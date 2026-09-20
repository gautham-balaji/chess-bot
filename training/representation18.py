"""C6-A3 board tensor encoding: the 18-plane representation.

Implements exactly the layout specified in docs/C6_TRAINING_PIPELINE_AUDIT.md §K.
Planes 0-11 are the existing 12-plane encoding, byte-for-byte unchanged; six
planes are APPENDED, never reordered, so the meaning of every existing channel
is preserved.

    plane  meaning                        values           spatial?
    0-11   the 12-plane piece encoding    0/1              yes
    12     side to move is White          all-1 or all-0   constant
    13     White kingside castling        constant 0/1     constant
    14     White queenside castling       constant 0/1     constant
    15     Black kingside castling        constant 0/1     constant
    16     Black queenside castling       constant 0/1     constant
    17     en-passant target square       single cell 1.0  yes

Everything stays binary {0, 1} float32 with absolute orientation and no
normalisation, matching the existing design. All six fields are plain attributes
of `chess.Board`, available for every legal position; `ep_square` is usually
None, which gives an all-zero plane.

--------------------------------------------------------------------------
THIS REPRESENTATION IS NOT LOADABLE BY THE SHIPPED ENGINE
--------------------------------------------------------------------------
`engine.board_to_planes` hardcodes `np.zeros((8, 8, 12))`, so an 18-channel CNN
cannot be fed by the unmodified engine at all - it raises a shape error. That is
a property of the engine, not a bug here, and it is why A3's engine evaluation
needs `training/evaluate18_runner.py`. See docs/C6_A3_REPORT.md.

Planes 0-11 are guarded against drift by tests/unit/test_training_representation18.py,
which asserts they equal `training.representation.board_to_planes` on every
position in both evaluation suites.
"""
from __future__ import annotations

import chess
import numpy as np

from training import representation as _r12

N_PLANES = 18
BOARD_SHAPE = (8, 8, N_PLANES)

# The six appended channels, in order.
SIDE_TO_MOVE_PLANE = 12
CASTLING_PLANES = {
    (chess.WHITE, "kingside"): 13,
    (chess.WHITE, "queenside"): 14,
    (chess.BLACK, "kingside"): 15,
    (chess.BLACK, "queenside"): 16,
}
EN_PASSANT_PLANE = 17

PLANE_NAMES = _r12.PLANE_NAMES + [
    "side_to_move_is_white",
    "white_kingside_castling",
    "white_queenside_castling",
    "black_kingside_castling",
    "black_queenside_castling",
    "en_passant_target",
]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a board as an (8, 8, 18) binary float32 tensor."""
    planes = np.zeros(BOARD_SHAPE, dtype=np.float32)

    # --- planes 0-11: identical to the 12-plane encoder ----------------------
    planes[:, :, :_r12.N_PLANES] = _r12.board_to_planes(board)

    # --- plane 12: side to move ---------------------------------------------
    if board.turn == chess.WHITE:
        planes[:, :, SIDE_TO_MOVE_PLANE] = 1.0

    # --- planes 13-16: castling rights ---------------------------------------
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[:, :, CASTLING_PLANES[(chess.WHITE, "kingside")]] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[:, :, CASTLING_PLANES[(chess.WHITE, "queenside")]] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[:, :, CASTLING_PLANES[(chess.BLACK, "kingside")]] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[:, :, CASTLING_PLANES[(chess.BLACK, "queenside")]] = 1.0

    # --- plane 17: en-passant target, using the existing orientation ---------
    if board.ep_square is not None:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        planes[row, col, EN_PASSANT_PLANE] = 1.0

    return planes


def fen_to_planes(fen: str) -> np.ndarray:
    return board_to_planes(chess.Board(fen))


def encode_many(fens) -> np.ndarray:
    """Stack an iterable of FENs into an (N, 8, 8, 18) array."""
    fens = list(fens)
    out = np.zeros((len(fens), *BOARD_SHAPE), dtype=np.float32)
    for i, fen in enumerate(fens):
        out[i] = fen_to_planes(fen)
    return out


def representation_summary() -> dict:
    """Recorded in every run's metadata so the arm is self-describing."""
    return {
        "name": "planes18",
        "shape": list(BOARD_SHAPE),
        "n_planes": N_PLANES,
        "dtype": "float32",
        "values": "binary {0, 1}",
        "plane_names": PLANE_NAMES,
        "orientation": "absolute; row 0 = rank 8, col 0 = file a; never flipped",
        "encodes_side_to_move": True,
        "encodes_castling_rights": True,
        "encodes_en_passant": True,
        "planes_0_to_11_identical_to_planes12": True,
        "loadable_by_unmodified_engine": False,
        "note": "docs/C6_TRAINING_PIPELINE_AUDIT.md section K. engine.board_to_planes "
                "hardcodes 12 channels, so the shipped engine cannot feed this model.",
    }
