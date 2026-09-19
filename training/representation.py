"""Board tensor encoding for training.

This is a standalone copy of the engine's 12-plane encoder. It is duplicated
deliberately: importing `engine` would load TensorFlow and the *production* CNN
just to encode a board, which couples the training harness to the shipped model.

The duplication is guarded by `tests/unit/test_training_representation.py`, which
asserts this function and `engine.board_to_planes` agree on every position in both
evaluation suites. If either ever drifts, that test fails.

A0 MUST use exactly these 12 planes. The side-to-move, castling and en-passant
planes belong to A3 and are not implemented here.
"""
from __future__ import annotations

import chess
import numpy as np

# Channel layout (identical to engine.board_to_planes):
#   0-5   White  P N B R Q K
#   6-11  Black  p n b r q k
# row = 7 - (square // 8)  -> row 0 is rank 8
# col = square % 8         -> col 0 is file a
PIECE_TO_INDEX = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
BLACK_OFFSET = 6
N_PLANES = 12
BOARD_SHAPE = (8, 8, N_PLANES)

PLANE_NAMES = [
    "white_pawn", "white_knight", "white_bishop", "white_rook",
    "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop", "black_rook",
    "black_queen", "black_king",
]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a board as an (8, 8, 12) binary float32 tensor."""
    planes = np.zeros(BOARD_SHAPE, dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        offset = 0 if piece.color == chess.WHITE else BLACK_OFFSET
        planes[row, col, PIECE_TO_INDEX[piece.symbol().upper()] + offset] = 1
    return planes


def fen_to_planes(fen: str) -> np.ndarray:
    return board_to_planes(chess.Board(fen))


def encode_many(fens) -> np.ndarray:
    """Stack an iterable of FENs into an (N, 8, 8, 12) array."""
    fens = list(fens)
    out = np.zeros((len(fens), *BOARD_SHAPE), dtype=np.float32)
    for i, fen in enumerate(fens):
        out[i] = fen_to_planes(fen)
    return out


def representation_summary() -> dict:
    """Recorded in every run's metadata so the arm is self-describing."""
    return {
        "name": "planes12",
        "shape": list(BOARD_SHAPE),
        "n_planes": N_PLANES,
        "dtype": "float32",
        "values": "binary {0, 1}",
        "plane_names": PLANE_NAMES,
        "orientation": "absolute; row 0 = rank 8, col 0 = file a; never flipped",
        "encodes_side_to_move": False,
        "encodes_castling_rights": False,
        "encodes_en_passant": False,
        "note": "identical to engine.board_to_planes; A3 will extend this to 18 planes",
    }
