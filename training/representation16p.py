"""C6-A13P board tensor encoding: 12 piece planes + 4 PLACEBO constant planes.

A13P is the control for A13. A13 added four spatially-constant castling-right
planes and regressed. Two hypotheses survive:

  (H1) the CASTLING CONTENT is what hurts
  (H2) adding ANY spatially-constant broadcast channel hurts this architecture
       at this dataset size

A13P keeps the channel count, the tensor shape and the parameter count of A13
while stripping every bit of chess information out of the four added planes.

    plane  meaning                        values           spatial?
    0-11   the 12-plane piece encoding    0/1              yes
    12-15  PLACEBO: fixed constant 1.0    all-1            constant

    NOT PRESENT: castling rights, side to move, en-passant.

--------------------------------------------------------------------------
WHY ALL-ONES RATHER THAN ALL-ZEROS
--------------------------------------------------------------------------
An all-zero plane is provably inert: it contributes nothing to any convolution
and receives no gradient, so its 2,304 weights would be dead and the control
would test only "does the tensor have more columns". All-ones planes actually
flow signal and gradient into the first Conv2D, which is the closest
information-free analogue of a broadcast channel that is switched on.

What they do to the network, stated precisely. The first layer is
`Conv2D(64, 3x3, padding="same", activation="relu")`, so the order is
conv -> ReLU -> BatchNormalization. A constant input plane adds a FIXED spatial
pattern to each filter's pre-activation: constant over the interior, and smaller
along the 1-pixel border where zero padding means some of the 3x3 taps see 0.
That pattern is learnable (2,304 new weights) and it shifts each filter's ReLU
threshold, so it is not a no-op - but it is identical for every position, so it
carries no information that could discriminate one board from another.

--------------------------------------------------------------------------
WHAT THIS CONTROL DOES AND DOES NOT SEPARATE
--------------------------------------------------------------------------
A13's castling planes are constant WITHIN a board but VARY ACROSS boards
(measured: per-plane std ~0.50 across the 9,667 records). That cross-position
variation is precisely what carries the castling information.

A position-independent placebo therefore differs from A13 in TWO ways at once:
it carries no chess semantics, AND it does not vary across positions at all.
So A13P can show that mere tensor width is harmless, but it CANNOT by itself
separate "castling semantics are harmful" from "any across-position varying
broadcast channel is harmful". Separating those needs a third arm whose added
planes vary across positions while carrying no chess meaning.
See docs/C6_A13P_REPORT.md.
"""
from __future__ import annotations

import chess
import numpy as np

from training import representation as _r12

N_PLANES = 16
BOARD_SHAPE = (8, 8, N_PLANES)

# The four appended placebo channels and their fixed values. Position
# independent by construction: this tuple is the whole definition.
PLACEBO_PLANES = (12, 13, 14, 15)
PLACEBO_VALUE = 1.0

PLANE_NAMES = _r12.PLANE_NAMES + [
    "placebo_constant_0",
    "placebo_constant_1",
    "placebo_constant_2",
    "placebo_constant_3",
]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a board as an (8, 8, 16) binary float32 tensor.

    `board` determines planes 0-11 only. Planes 12-15 never read it.
    """
    planes = np.zeros(BOARD_SHAPE, dtype=np.float32)

    # --- planes 0-11: identical to the 12-plane encoder ----------------------
    planes[:, :, :_r12.N_PLANES] = _r12.board_to_planes(board)

    # --- planes 12-15: fixed constants, independent of `board` ---------------
    for idx in PLACEBO_PLANES:
        planes[:, :, idx] = PLACEBO_VALUE

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
        "name": "planes16p",
        "shape": list(BOARD_SHAPE),
        "n_planes": N_PLANES,
        "dtype": "float32",
        "values": "binary {0, 1}",
        "plane_names": PLANE_NAMES,
        "orientation": "absolute; row 0 = rank 8, col 0 = file a; never flipped",
        "encodes_side_to_move": False,
        "encodes_castling_rights": False,
        "encodes_en_passant": False,
        "planes_0_to_11_identical_to_planes12": True,
        "added_planes_are_placebo": True,
        "added_planes_value": PLACEBO_VALUE,
        "added_planes_depend_on_position": False,
        "loadable_by_unmodified_engine": False,
        "note": "A13P: control for A13. Same shape and parameter count as "
                "planes16, but the four added channels are fixed constants "
                "carrying no chess information.",
    }
