from flask import Flask, request, jsonify, render_template
import chess

app = Flask(__name__)

from engine import engine_move, position_metrics, explain_move

board = chess.Board()
move_history = []


def board_to_san(b, move):
    try:
        return b.san(move)
    except Exception:
        return move.uci()


def get_state():
    metrics = position_metrics(board) if not board.is_game_over() else {}

    pieces = {}
    symbol_map = {
        chess.PAWN: 'P', chess.KNIGHT: 'N', chess.BISHOP: 'B',
        chess.ROOK: 'R', chess.QUEEN: 'Q', chess.KING: 'K'
    }
    for sq, piece in board.piece_map().items():
        name = chess.square_name(sq)
        color = 'w' if piece.color == chess.WHITE else 'b'
        pieces[name] = color + symbol_map[piece.piece_type]

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
        "fen": board.fen(),
        "pieces": pieces,
        "legal_map": legal_map,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "move_history": move_history,
        "metrics": metrics,
        "game_over": game_over,
        "result": result,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/state")
def state():
    return jsonify(get_state())


@app.route("/move", methods=["POST"])
def make_move():
    """Apply only the PLAYER'S move. Engine move is fetched separately."""
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

    san = board_to_san(board, move)
    board.push(move)
    move_history.append(san)
    player_from = chess.square_name(move.from_square)
    player_to   = chess.square_name(move.to_square)

    return jsonify({
        "player_move": {"uci": uci, "san": san, "from": player_from, "to": player_to},
        **get_state(),
    })


@app.route("/engine_move", methods=["POST"])
def do_engine_move():
    """Compute and apply the ENGINE'S move. Called by frontend after showing player move."""
    global board, move_history

    if board.is_game_over():
        return jsonify({"error": "Game is over"}), 400

    # It must be black's turn (engine plays black)
    if board.turn == chess.WHITE:
        return jsonify({"error": "Not engine's turn"}), 400

    try:
        eng_move, explanation, top3 = engine_move(board)
        if not eng_move or eng_move not in board.legal_moves:
            return jsonify({"error": "Engine has no legal moves"}), 400

        eng_san  = board_to_san(board, eng_move)
        eng_from = chess.square_name(eng_move.from_square)
        eng_to   = chess.square_name(eng_move.to_square)
        board.push(eng_move)
        move_history.append(eng_san)

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

        engine_data = {
            "uci":         eng_move.uci(),
            "san":         eng_san,
            "from":        eng_from,
            "to":          eng_to,
            "explanation": explanation,
            "top_moves":   top_moves_out,
        }

        return jsonify({
            "engine": engine_data,
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


@app.route("/reset", methods=["POST"])
def reset():
    global board, move_history
    board        = chess.Board()
    move_history = []
    return jsonify(get_state())


if __name__ == "__main__":
    app.run(debug=True, port=5000)