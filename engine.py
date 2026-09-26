import logging
import pickle
from contextlib import contextmanager
import numpy as np
import chess
from tensorflow.keras.models import load_model

from config import CNN_MODEL_PATH, WEIGHT_MODEL_PATH

logger = logging.getLogger(__name__)

# --- Load the models the engine actually uses ---
# Paths are repository-relative (see config.py), not relative to the current
# working directory, so importing this module does not depend on where Python
# was launched from.
#
# rf_model.pkl, mlp_model.pkl and scaler.pkl were previously loaded here but are
# never referenced by this module or by app.py. They are training-notebook
# artifacts; loading them cost ~1.07s and ~79MB of resident memory for no effect.
# The files are retained on disk for provenance. See docs/REPRODUCIBILITY.md.
cnn_model = load_model(str(CNN_MODEL_PATH), compile=False)

with open(WEIGHT_MODEL_PATH, "rb") as f:
    weight_model = pickle.load(f)

logger.info("Loaded CNN evaluator and Ridge weight model")

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

# C10: push/pop must be exception-safe.
#
# chess.Board.push() appends to move_stack and _stack BEFORE it validates the
# move (chess/__init__.py: the `piece_type is not None` assert fires after both
# appends). So a move from an empty square mutates the caller's board - the
# halfmove clock advances and a bogus frame is left on the stack - and then
# raises, which used to skip the pop() entirely and leave the caller's board
# corrupted.
#
# Unwinding to the recorded depth rather than calling pop() once means a failure
# EARLIER in push(), which would leave no frame to unwind, cannot raise
# IndexError here and mask the original exception.
@contextmanager
def _pushed(board, move):
    depth = len(board.move_stack)
    try:
        board.push(move)
        yield board
    finally:
        while len(board.move_stack) > depth:
            board.pop()


def move_impact(board, move):
    before = space_control(board)
    with _pushed(board, move):
        after = space_control(board)
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

# C10 fix (C5): the central pawn pushes, both colours.
#
# opening_center_bonus previously listed White's three pushes only, so Black's
# mirrored pushes scored 0 while White's scored 0.3. tactical_move_bonus already
# listed both colours, which is what establishes the intended semantics as
# colour-symmetric - the omission was in opening_center_bonus, not in the
# concept. Both helpers now read the same constant so they cannot drift apart.
#
# The magnitude (0.3) is unchanged, and the list is exactly the union
# tactical_move_bonus already used, so tactical_move_bonus is unaffected.
#
# app.py only ever lets the engine play Black (its /engine_move handler rejects
# White's turn), so the asymmetry was
# active in every game - the same structural argument that justified the C1 fix.
CENTRAL_PAWN_PUSHES = ("e2e4", "d2d4", "c2c4", "e7e5", "d7d5", "c7c5")


def opening_center_bonus(board, move):
    if move.uci() in CENTRAL_PAWN_PUSHES:
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
    with _pushed(board, move):
        if board.is_check():
            bonus += 0.2
    if move.promotion:
        bonus += 0.4
    if move.uci() in CENTRAL_PAWN_PUSHES:
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

        # C1 fix: apply the heuristic bonuses in the direction that is BETTER for
        # the side to move.
        #
        # The weighted score above is on a White-positive scale, and the sort at
        # the end of this function is descending for White but ascending for
        # Black (lower is better for Black). The bonuses were previously added
        # with a fixed positive sign, so for Black a "good move" bonus pushed the
        # move DOWN Black's own preference list. The app only ever lets the
        # engine play Black, so this was active in every game.
        #
        # Multiplying by 1.0 is exact in IEEE 754 and the addition order is
        # unchanged, so White's scores remain bit-identical.
        bonus_sign = 1.0 if board.turn == chess.WHITE else -1.0
        score += bonus_sign * development_bonus(board, mv)
        score += bonus_sign * pawn_push_penalty(board, mv)
        score += bonus_sign * opening_center_bonus(board, mv)
        score += bonus_sign * tactical_move_bonus(board, mv)

        move_scores.append({
            "move": mv,
            "score": round(float(score), 3),
            "cnn_cp": round(float(cnn_score), 2),
            "material": mat,
            "space": space,
            # C10 fix (C4): this used to call center_control(board) HERE, which is
            # after the board.pop() above, so the reported value was the PRE-move
            # centre control while material/space/mobility were post-move. `center`
            # is the post-move value captured inside the push, and is the same
            # value the weighted score above already used - so this is a reporting
            # fix only: no score, ordering or selection changes.
            "center": center,
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

        # C2 fix: make this an actual minimax step.
        #
        # The CNN is a WHITE-POSITIVE evaluator and the candidate score is on the
        # same White-positive scale, with no sign transform in between. So the
        # opponent picks the reply that is best for THEM on that scale:
        #   - we are White  -> the opponent is Black -> they MINIMISE
        #   - we are Black  -> the opponent is White -> they MAXIMISE
        # The previous code always took the maximum, which selected the reply most
        # favourable to White regardless of who was actually replying.
        #
        # The resulting value is then blended in POSITIVELY, because the sort
        # below already encodes direction (descending for White, ascending for
        # Black). The previous code subtracted it, which inverted the term for
        # both sides: a reply that is good for White made White's move look worse
        # and Black's move look better.
        opponent_maximises = (board.turn == chess.BLACK)
        opp_best = {}
        for idx, sc in zip(opp_move_map, opp_scores):
            if idx not in opp_best:
                opp_best[idx] = sc
            elif opponent_maximises:
                opp_best[idx] = max(opp_best[idx], sc)
            else:
                opp_best[idx] = min(opp_best[idx], sc)
        for i, entry in enumerate(move_scores):
            if i in opp_best:
                entry["score"] += 0.5 * np.tanh(opp_best[i] / 200)

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
    with _pushed(board, move):
        if board.is_check():
            reasons.append("delivers a check to the opponent king")
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