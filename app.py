from flask import Flask, request, jsonify, render_template
import chess
import numpy as np

app = Flask(__name__)

from engine import engine_move, position_metrics, explain_move, cnn_model, board_to_planes

# ── global game state ─────────────────────────────────────────────────────────
board            = chess.Board()
move_history     = []          # SAN strings
captured_white   = []          # pieces captured FROM white (i.e. white's pieces taken)
captured_black   = []          # pieces captured FROM black (i.e. black's pieces taken)

# running metric snapshots for end-of-game stats
metric_snapshots = []          # list of dicts, one per half-move
forfeited        = False       # set True on resignation

PIECE_NAMES = {
    chess.PAWN: 'P', chess.KNIGHT: 'N', chess.BISHOP: 'B',
    chess.ROOK: 'R', chess.QUEEN: 'Q', chess.KING: 'K'
}


def board_to_san(b, move):
    try:
        return b.san(move)
    except Exception:
        return move.uci()


def record_capture(b, move):
    """Before pushing a move, record any capture."""
    if b.is_capture(move):
        target_sq = move.to_square
        # en passant: captured pawn is not on to_square
        if b.is_en_passant(move):
            ep_rank = 4 if b.turn == chess.WHITE else 3
            ep_file = chess.square_file(move.to_square)
            target_sq = chess.square(ep_file, ep_rank)
        piece = b.piece_at(target_sq)
        if piece:
            code = ('w' if piece.color == chess.WHITE else 'b') + PIECE_NAMES[piece.piece_type]
            if piece.color == chess.WHITE:
                captured_white.append(code)
            else:
                captured_black.append(code)


def get_state():
    metrics = position_metrics(board) if not board.is_game_over() else {}

    pieces = {}
    for sq, piece in board.piece_map().items():
        name  = chess.square_name(sq)
        color = 'w' if piece.color == chess.WHITE else 'b'
        pieces[name] = color + PIECE_NAMES[piece.piece_type]

    legal_map = {}
    for mv in board.legal_moves:
        frm = chess.square_name(mv.from_square)
        to  = chess.square_name(mv.to_square)
        legal_map.setdefault(frm, []).append(to)

    game_over = board.is_game_over()
    result = None
    if game_over:
        outcome = board.outcome()
        result = {
            "result": board.result(),
            "reason": outcome.termination.name.replace("_", " ").title() if outcome else ""
        }

    return {
        "fen":            board.fen(),
        "pieces":         pieces,
        "legal_map":      legal_map,
        "turn":           "white" if board.turn == chess.WHITE else "black",
        "move_history":   move_history,
        "metrics":        metrics,
        "game_over":      game_over,
        "result":         result,
        "captured_white": captured_white,
        "captured_black": captured_black,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/state")
def state():
    return jsonify(get_state())


@app.route("/move", methods=["POST"])
def make_move():
    global board, move_history
    data = request.json or {}
    uci  = data.get("uci", "").strip()

    if board.is_game_over():
        return jsonify({"error": "Game is over"}), 400

    try:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return jsonify({"error": f"Illegal move: {uci}"}), 400
    except Exception:
        return jsonify({"error": f"Invalid UCI: {uci}"}), 400

    record_capture(board, move)
    san = board_to_san(board, move)
    board.push(move)
    move_history.append(san)

    # snapshot metrics after player move
    try:
        snap = position_metrics(board)
        snap['half_move'] = len(move_history)
        metric_snapshots.append(snap)
    except Exception:
        pass

    player_from = chess.square_name(move.from_square)
    player_to   = chess.square_name(move.to_square)

    return jsonify({
        "player_move": {"uci": uci, "san": san, "from": player_from, "to": player_to},
        **get_state(),
    })


@app.route("/engine_move", methods=["POST"])
def do_engine_move():
    global board, move_history

    if board.is_game_over():
        return jsonify({"error": "Game is over"}), 400
    if board.turn == chess.WHITE:
        return jsonify({"error": "Not engine's turn"}), 400

    try:
        eng_move, explanation, top3 = engine_move(board)
        if not eng_move or eng_move not in board.legal_moves:
            return jsonify({"error": "Engine has no legal moves"}), 400

        record_capture(board, eng_move)
        eng_san  = board_to_san(board, eng_move)
        eng_from = chess.square_name(eng_move.from_square)
        eng_to   = chess.square_name(eng_move.to_square)
        board.push(eng_move)
        move_history.append(eng_san)

        try:
            snap = position_metrics(board)
            snap['half_move'] = len(move_history)
            metric_snapshots.append(snap)
        except Exception:
            pass

        # Compute top moves for WHITE's next turn (board is now White's turn)
        top_moves_out = []
        if not board.is_game_over():
            try:
                _, _, white_top3 = engine_move(board)
                for m in white_top3:
                    mv = m["move"]
                    try:
                        mv_san = board.san(mv) if mv in board.legal_moves else mv.uci()
                    except Exception:
                        mv_san = mv.uci()
                    try:
                        reasons = explain_move(board, mv, m)
                    except Exception:
                        reasons = []
                    top_moves_out.append({
                        "uci":      mv.uci(),
                        "san":      mv_san,
                        "score":    round(float(m["score"]), 3),
                        "cnn_cp":   round(float(m["cnn_cp"]), 1),
                        "material": m["material"],
                        "space":    m["space"],
                        "reasons":  reasons,
                    })
            except Exception:
                pass

        return jsonify({
            "engine": {
                "uci":         eng_move.uci(),
                "san":         eng_san,
                "from":        eng_from,
                "to":          eng_to,
                "explanation": explanation,
                "top_moves":   top_moves_out,
            },
            **get_state(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyse", methods=["POST"])
def analyse():
    if board.is_game_over():
        return jsonify({"error": "Game is over"}), 400
    try:
        _, _, top3 = engine_move(board)
        metrics    = position_metrics(board)
        top_moves_out = []
        for m in top3:
            mv = m["move"]
            try:
                mv_san = board.san(mv) if mv in board.legal_moves else mv.uci()
            except Exception:
                mv_san = mv.uci()
            try:
                reasons = explain_move(board, mv, m)
            except Exception:
                reasons = []
            top_moves_out.append({
                "uci":      mv.uci(),
                "san":      mv_san,
                "score":    round(float(m["score"]), 3),
                "cnn_cp":   round(float(m["cnn_cp"]), 1),
                "material": m["material"],
                "space":    m["space"],
                "reasons":  reasons,
            })
        return jsonify({"top_moves": top_moves_out, "metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forfeit", methods=["POST"])
def forfeit():
    """Player resigns — frontend will call /game_stats after this."""
    global forfeited
    forfeited = True
    state = get_state()
    state["game_over"] = True
    state["result"]    = {"result": "0-1", "reason": "Resignation"}
    return jsonify({"forfeited": True, **state})


@app.route("/game_stats", methods=["POST"])
def game_stats():
    """
    Compute end-of-game statistics and a saliency map using
    Integrated Gradients on the final board position.
    """
    import tensorflow as tf

    total_moves   = len(move_history)
    white_moves   = [move_history[i] for i in range(0, len(move_history), 2)]
    black_moves   = [move_history[i] for i in range(1, len(move_history), 2)]

    # Average metrics over the game
    avg_material = 0.0
    avg_space    = 0.0
    avg_center   = 0.0
    if metric_snapshots:
        avg_material = round(sum(s.get('material', 0) for s in metric_snapshots) / len(metric_snapshots), 2)
        avg_space    = round(sum(s.get('space', 0)    for s in metric_snapshots) / len(metric_snapshots), 2)
        avg_center   = round(sum(s.get('center', 0)   for s in metric_snapshots) / len(metric_snapshots), 2)

    # ── Integrated Gradients saliency map ─────────────────────────────────────
    # Use current board position (end of game)
    try:
        tensor      = board_to_planes(board)          # (8,8,12)
        inp         = tf.constant(tensor[np.newaxis], dtype=tf.float32)  # (1,8,8,12)
        baseline    = tf.zeros_like(inp)
        steps       = 50
        alphas      = tf.linspace(0.0, 1.0, steps + 1)
        interpolated = baseline + alphas[:, None, None, None] * (inp - baseline)  # (51,8,8,12)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            preds = cnn_model(interpolated, training=False)

        grads = tape.gradient(preds, interpolated)      # (51,8,8,12)
        avg_grads = tf.reduce_mean(grads, axis=0)       # (8,8,12)
        ig = (inp[0] - baseline[0]) * avg_grads         # (8,8,12)

        # collapse channels → per-square importance
        saliency = tf.reduce_sum(tf.abs(ig), axis=-1).numpy()   # (8,8)

        # Normalise 0→1
        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        # Convert to list-of-lists (rank 8 first = row 0 = rank 8 visually)
        saliency_list = saliency.tolist()

    except Exception as e:
        saliency_list = None
        print(f"Saliency error: {e}")

    return jsonify({
        "total_moves":      total_moves,
        "white_moves":      len(white_moves),
        "black_moves":      len(black_moves),
        "captured_white":   captured_white,
        "captured_black":   captured_black,
        "avg_material":     avg_material,
        "avg_space":        avg_space,
        "avg_center":       avg_center,
        "saliency":         saliency_list,   # 8×8 float, or null if failed
    })


@app.route("/reset", methods=["POST"])
def reset():
    global board, move_history, captured_white, captured_black, metric_snapshots, forfeited
    board            = chess.Board()
    move_history     = []
    captured_white   = []
    captured_black   = []
    metric_snapshots = []
    forfeited        = False
    return jsonify(get_state())


if __name__ == "__main__":
    app.run(debug=True, port=5000)