"""C6-A13 board tensor encoding: 12 piece planes + 4 castling-right planes.

A13 isolates the ONE component of A3's 18-plane representation that showed
measured signal. A3 added six planes at once and regressed; the plane-ablation
probe (docs/C6_A3_REPORT.md section 12) attributed the damage to the
side-to-move plane (304 cp of prediction leverage on a feature the
White-positive labels made irrelevant) and found the en-passant plane inert
(0.15 cp, 0.27% dataset coverage). Castling sat in between at 92.6 cp.

A13 therefore keeps castling and drops the other two:

    plane  meaning                        values           spatial?
    0-11   the 12-plane piece encoding    0/1              yes
    12     White kingside castling        constant 0/1     constant
    13     White queenside castling       constant 0/1     constant
    14     Black kingside castling        constant 0/1     constant
    15     Black queenside castling       constant 0/1     constant

    NOT PRESENT: side to move, en-passant target.

16 channels, not 13: "A13" is the arm name, not the plane count. Castling is
four separate binary rights, so encoding them takes four planes.

Channel indices 12-15 deliberately differ from representation18, where castling
occupies 13-16 because side-to-move takes 12. Planes 0-11 are identical in both
and in the shipped engine, which is what keeps the arms comparable; the added
channels are a per-representation choice.

Everything stays binary {0, 1} float32 with absolute orientation and no
normalisation, matching the existing design.

--------------------------------------------------------------------------
THIS REPRESENTATION IS NOT LOADABLE BY THE SHIPPED ENGINE
--------------------------------------------------------------------------
`engine.board_to_planes` hardcodes `np.zeros((8, 8, 12))`, so a 16-channel CNN
raises a shape error under the unmodified engine, exactly as A3's 18-channel one
does. A13 evaluation therefore goes through the same experiment-only in-process
shim. No production file is modified.

Planes 0-11 are guarded against drift by tests/unit/test_training_representation16.py,
which asserts they equal `training.representation.board_to_planes` on every
position in both evaluation suites.
"""
from __future__ import annotations

import chess
import numpy as np

from training import representation as _r12

N_PLANES = 16
BOARD_SHAPE = (8, 8, N_PLANES)

# The four appended channels, in order. Indices are relative to this encoding.
CASTLING_PLANES = {
    (chess.WHITE, "kingside"): 12,
    (chess.WHITE, "queenside"): 13,
    (chess.BLACK, "kingside"): 14,
    (chess.BLACK, "queenside"): 15,
}

PLANE_NAMES = _r12.PLANE_NAMES + [
    "white_kingside_castling",
    "white_queenside_castling",
    "black_kingside_castling",
    "black_queenside_castling",
]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a board as an (8, 8, 16) binary float32 tensor."""
    planes = np.zeros(BOARD_SHAPE, dtype=np.float32)

    # --- planes 0-11: identical to the 12-plane encoder ----------------------
    planes[:, :, :_r12.N_PLANES] = _r12.board_to_planes(board)

    # --- planes 12-15: castling rights ---------------------------------------
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[:, :, CASTLING_PLANES[(chess.WHITE, "kingside")]] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[:, :, CASTLING_PLANES[(chess.WHITE, "queenside")]] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[:, :, CASTLING_PLANES[(chess.BLACK, "kingside")]] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[:, :, CASTLING_PLANES[(chess.BLACK, "queenside")]] = 1.0

    return planes


def fen_to_planes(fen: str) -> np.ndarray:
    return board_to_planes(chess.Board(fen))


def encode_many(fens) -> np.ndarray:
    """Stack an iterable of FENs into an (N, 8, 8, 16) array."""
    fens = list(fens)
    out = np.zeros((len(fens), *BOARD_SHAPE), dtype=np.float32)
    for i, fen in enumerate(fens):
        out[i] = fen_to_planes(fen)
    return out


def representation_summary() -> dict:
    """Recorded in every run's metadata so the arm is self-describing."""
    return {
        "name": "planes16",
        "shape": list(BOARD_SHAPE),
        "n_planes": N_PLANES,
        "dtype": "float32",
        "values": "binary {0, 1}",
        "plane_names": PLANE_NAMES,
        "orientation": "absolute; row 0 = rank 8, col 0 = file a; never flipped",
        "encodes_side_to_move": False,
        "encodes_castling_rights": True,
        "encodes_en_passant": False,
        "planes_0_to_11_identical_to_planes12": True,
        "loadable_by_unmodified_engine": False,
        "note": "A13: castling rights only. Isolates the one component of A3's "
                "18-plane set that had measured signal, excluding side-to-move "
                "(harmful per C6_A3_REPORT section 12) and en-passant (inert).",
    }
