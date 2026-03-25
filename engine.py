import pickle
import numpy as np
import chess
from tensorflow.keras.models import load_model

# --- Load all models ---
cnn_model = load_model("models/cnn_model.keras", compile=False)

with open("models/rf_model.pkl", "rb") as f:
    rf = pickle.load(f)

with open("models/mlp_model.pkl", "rb") as f:
    mlp = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/weight_model.pkl", "rb") as f:
    weight_model = pickle.load(f)

print("✅ All models loaded")

# ─────────────────────────────────────────────
# BOARD TENSOR
# ─────────────────────────────────────────────

def board_to_planes(board):
    planes = np.zeros((8, 8, 12), dtype=np.float32)
    piece_map = board.piece_map()
    piece_to_index = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5}
    for square, piece in piece_map.items():
        row = 7 - (square // 8)
        col = square % 8
        offset = 0 if piece.color == chess.WHITE else 6
        planes[row, col, piece_to_index[piece.symbol().upper()] + offset] = 1
    return planes

# ─────────────────────────────────────────────
# EVALUATION FUNCTIONS
# ─────────────────────────────────────────────

def cnn_evaluate(board):
    tensor = board_to_planes(board)
    tensor = np.expand_dims(tensor, axis=0)
    pred = cnn_model.predict(tensor, verbose=0)[0][0]
    return float(pred)

def material_balance(board):
    values = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9}
    white = sum(len(board.pieces(p, chess.WHITE)) * v for p, v in values.items())
    black = sum(len(board.pieces(p, chess.BLACK)) * v for p, v in values.items())
    return white - black

def space_control(board):
    white = sum(1 for sq in chess.SQUARES if board.is_attacked_by(chess.WHITE, sq))
    black = sum(1 for sq in chess.SQUARES if board.is_attacked_by(chess.BLACK, sq))
    return white - black

def center_control(board):
    center = [chess.D4, chess.E4, chess.D5, chess.E5]
    white = sum(1 for sq in center if board.is_attacked_by(chess.WHITE, sq))
    black = sum(1 for sq in center if board.is_attacked_by(chess.BLACK, sq))
    return white - black

def mobility_score(board):
    my_moves = len(list(board.legal_moves))
    board.turn = not board.turn
    opp_moves = len(list(board.legal_moves))
    board.turn = not board.turn
    return my_moves - opp_moves

def move_impact(board, move):
    before = space_control(board)
    board.push(move)
    after = space_control(board)
    board.pop()
    return after - before

# ─────────────────────────────────────────────
# HEURISTIC BONUSES
# ─────────────────────────────────────────────

def development_bonus(board, move):
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
        return 0.2
    return 0

def pawn_push_penalty(board, move):
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        if chess.square_rank(move.from_square) == 1:
            return -0.2
    return 0

def opening_center_bonus(board, move):
    if move.uci() in ["e2e4","d2d4","c2c4"]:
        return 0.3
    return 0

def tactical_move_bonus(board, move):
    bonus = 0.0
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        capture_values = {
            chess.PAWN: 0.1, chess.KNIGHT: 0.25,
            chess.BISHOP: 0.25, chess.ROOK: 0.35, chess.QUEEN: 0.5
        }
        bonus += capture_values.get(captured.piece_type, 0.1) if captured else 0.1
    board.push(move)
    if board.is_check():
        bonus += 0.2
    board.pop()
    if move.promotion:
        bonus += 0.4
    if move.uci() in ["e2e4","d2d4","c2c4","e7e5","d7d5","c7c5"]:
        bonus += 0.25
    return bonus

# ─────────────────────────────────────────────
# HYBRID SCORE
# ─────────────────────────────────────────────

def hybrid_score(board):
    cnn_score = cnn_evaluate(board)
    cnn_norm = np.tanh(cnn_score / 200)
    mat = material_balance(board)
    space = space_control(board)
    center = center_control(board)
    mob = mobility_score(board)
    w = weight_model.coef_
    score = w[0]*cnn_norm + w[1]*mat + w[2]*space + w[3]*center + w[4]*mob
    return score, cnn_score, mat, space

# ─────────────────────────────────────────────
# RERANKER (BATCHED)
# ─────────────────────────────────────────────

def rerank_moves(board):
    candidates = list(board.legal_moves)
    if not candidates:
        return []

    # batch CNN predictions for all candidate moves
    tensors = []
    for mv in candidates:
        board.push(mv)
        tensors.append(board_to_planes(board))
        board.pop()

    cnn_scores = cnn_model.predict(np.array(tensors), verbose=0).flatten()

    move_scores = []
    for mv, cnn_score in zip(candidates, cnn_scores):
        board.push(mv)
        mat = material_balance(board)
        space = space_control(board)
        center = center_control(board)
        mob = mobility_score(board)
        cnn_norm = np.tanh(cnn_score / 200)
        try:
            w = weight_model.coef_
            score = w[0]*cnn_norm + w[1]*mat + w[2]*space + w[3]*center + w[4]*mob
        except:
            score = 0.5*cnn_norm + 0.25*mat + 0.15*space + 0.08*center + 0.02*mob
        board.pop()

        score += development_bonus(board, mv)
        score += pawn_push_penalty(board, mv)
        score += opening_center_bonus(board, mv)
        score += tactical_move_bonus(board, mv)

        move_scores.append({
            "move": mv,
            "score": round(float(score), 3),
            "cnn_cp": round(float(cnn_score), 2),
            "material": mat,
            "space": space,
            "center": center_control(board),
            "mobility": mob
        })

    # 1-ply shallow search — batch opponent responses
    opp_tensors, opp_move_map = [], []
    for i, entry in enumerate(move_scores):
        board.push(entry["move"])
        for opp_mv in board.legal_moves:
            board.push(opp_mv)
            opp_tensors.append(board_to_planes(board))
            opp_move_map.append(i)
            board.pop()
        board.pop()

    if opp_tensors:
        opp_scores = cnn_model.predict(np.array(opp_tensors), verbose=0).flatten()
        opp_best = {}
        for idx, sc in zip(opp_move_map, opp_scores):
            if idx not in opp_best or sc > opp_best[idx]:
                opp_best[idx] = sc
        for i, entry in enumerate(move_scores):
            if i in opp_best:
                entry["score"] -= 0.5 * np.tanh(opp_best[i] / 200)

    move_scores.sort(key=lambda x: x["score"], reverse=(board.turn == chess.WHITE))
    return move_scores

# ─────────────────────────────────────────────
# EXPLAIN MOVE
# ─────────────────────────────────────────────

def explain_move(board, move, info):
    reasons = []
    piece = board.piece_at(move.from_square)
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]

    if move.to_square in center_squares:
        reasons.append("strengthens control of the center")
    if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
        reasons.append("develops a minor piece")
    if piece and piece.piece_type == chess.PAWN:
        if chess.square_rank(move.to_square) >= 3:
            reasons.append("expands space with a pawn advance")
    if board.is_capture(move):
        reasons.append("captures an opponent piece")
    board.push(move)
    if board.is_check():
        reasons.append("delivers a check to the opponent king")
    board.pop()
    if move.promotion:
        reasons.append("promotes a pawn")
    if info.get("cnn_cp", 0) > 100:
        reasons.append("neural evaluation indicates a positional improvement")

    space_delta = move_impact(board, move)
    if space_delta > 0:
        reasons.append(f"increases board control by +{space_delta} squares")
    elif space_delta < 0:
        reasons.append(f"trades space for other compensation ({space_delta} squares)")

    if not reasons:
        reasons.append("improves overall piece coordination")

    return reasons

# ─────────────────────────────────────────────
# ENGINE MOVE (main entry point for UI)
# ─────────────────────────────────────────────

def engine_move(board):
    ranked = rerank_moves(board)
    if not ranked:
        return None, ["No legal moves available"], []
    best = ranked[0]
    move = best["move"]
    explanation = explain_move(board, move, best)
    return move, explanation, ranked[:3]

# ─────────────────────────────────────────────
# POSITION METRICS (for UI analysis panel)
# ─────────────────────────────────────────────

def position_metrics(board):
    return {
        "material": material_balance(board),
        "space": space_control(board),
        "center": center_control(board),
        "mobility": mobility_score(board),
        "cnn_eval": round(cnn_evaluate(board), 2)
    }