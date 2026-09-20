"""C6-A14 encoding: 12 piece planes for the CNN + 4 castling SCALARS for the head.

A14 is the architectural follow-up to A3 / A13 / A13P / A13R. Those four arms
all put global state into the CONVOLUTIONAL INPUT as spatially-constant
broadcast planes, and all four regressed against A2 - including A13R, whose
added planes contained nothing but hash noise. The surviving hypothesis is that
the damage is caused by WHERE the global state enters the network, not by what
it contains.

A14 therefore keeps the castling information and changes only its entry point:

    A13  : [12 piece planes | 4 castling planes] -> Conv/BN stack -> Flatten -> Dense
    A14  :  12 piece planes                      -> Conv/BN stack -> Flatten -+
                                                                              |-> concat -> Dense
            4 castling scalars ---------------------------------------------- +

--------------------------------------------------------------------------
WHY THIS MODULE STILL PRODUCES A 16-CHANNEL TENSOR
--------------------------------------------------------------------------
Every consumer in this repo - `engine.cnn_evaluate`, the engine's two batched
call sites, `training.train`, `training.a2_saturation_probe` - moves a position
through the network as ONE numpy array:

    engine:   cnn_model.predict(np.array([board_to_planes(b) for b in ...]))

so a two-input Keras model could not be fed without changing production call
sites, which this experiment is forbidden to do. The tensor produced here is a
TRANSPORT container, not a convolutional input:

    channel  meaning                              consumed by
    0-11     the 12-plane piece encoding          the Conv/BN stack
    12       White kingside castling right        the dense head, as a scalar
    13       White queenside castling right       the dense head, as a scalar
    14       Black kingside castling right        the dense head, as a scalar
    15       Black queenside castling right       the dense head, as a scalar

Channels 12-15 are broadcast over all 64 cells ONLY so that the four bits can
ride inside the same array. `training.train.build_model_postconv_castling`
splits the tensor as its FIRST graph operation: the Conv2D stack receives
`x[:, :, :, :12]`, shape (None, 8, 8, 12), and the four rights are read back as
`x[:, 0, 0, 12:]`, shape (None, 4), which is concatenated AFTER Flatten.

Castling therefore never reaches a convolution. `tests/unit/test_training_representation12c4.py`
pins that from the built graph, not from this docstring.

--------------------------------------------------------------------------
RELATION TO A13
--------------------------------------------------------------------------
Channels 12-15 here are bit-for-bit identical to `representation16`'s castling
planes, in the same order. That is deliberate and is what makes the A13 vs A14
contrast exact: the two arms receive THE SAME INPUT TENSOR and differ only in
where the network consumes channels 12-15. This module does not import
`representation16`; the equality is asserted by the test suite instead, so
neither arm can silently drift into the other.

--------------------------------------------------------------------------
FEATURE ORDER (canonical, deterministic, never reordered)
--------------------------------------------------------------------------
    index 0  white_kingside_castling
    index 1  white_queenside_castling
    index 2  black_kingside_castling
    index 3  black_queenside_castling

Same order as `representation16` and as the castling block of `representation18`.

--------------------------------------------------------------------------
NOT LOADABLE BY THE SHIPPED ENGINE
--------------------------------------------------------------------------
`engine.board_to_planes` hardcodes `np.zeros((8, 8, 12))`, so - exactly as for
A3, A13, A13P and A13R - an A14 model cannot be fed by the unmodified engine and
is evaluated through the experiment-only in-process shim
`training/evaluate_planes_runner.py`. No production file is modified.
"""
from __future__ import annotations

import chess
import numpy as np

from training import representation as _r12

N_PLANES = 16
BOARD_SHAPE = (8, 8, N_PLANES)

# What the convolutional stack actually sees.
CONV_PLANES = _r12.N_PLANES            # 12
CONV_INPUT_SHAPE = _r12.BOARD_SHAPE    # (8, 8, 12)

# The four castling scalars, in canonical order. Index i of this tuple is
# transported in channel CONV_PLANES + i and is feature i of the dense head.
CASTLING_FEATURE_ORDER = (
    (chess.WHITE, "kingside"),
    (chess.WHITE, "queenside"),
    (chess.BLACK, "kingside"),
    (chess.BLACK, "queenside"),
)
CASTLING_FEATURE_NAMES = (
    "white_kingside_castling",
    "white_queenside_castling",
    "black_kingside_castling",
    "black_queenside_castling",
)
N_CASTLING_FEATURES = len(CASTLING_FEATURE_ORDER)
CASTLING_CHANNELS = tuple(range(CONV_PLANES, CONV_PLANES + N_CASTLING_FEATURES))

PLANE_NAMES = _r12.PLANE_NAMES + [f"{n}_scalar" for n in CASTLING_FEATURE_NAMES]


def castling_scalars(board: chess.Board) -> np.ndarray:
    """The four castling rights as a (4,) float32 vector in canonical order."""
    out = np.zeros(N_CASTLING_FEATURES, dtype=np.float32)
    for i, (colour, side) in enumerate(CASTLING_FEATURE_ORDER):
        has = (board.has_kingside_castling_rights(colour) if side == "kingside"
               else board.has_queenside_castling_rights(colour))
        out[i] = 1.0 if has else 0.0
    return out


def conv_planes(board: chess.Board) -> np.ndarray:
    """The (8, 8, 12) tensor the convolutional stack consumes - nothing else."""
    return _r12.board_to_planes(board)


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a board as the (8, 8, 16) TRANSPORT tensor described above."""
    planes = np.zeros(BOARD_SHAPE, dtype=np.float32)
    planes[:, :, :CONV_PLANES] = conv_planes(board)
    for i, value in enumerate(castling_scalars(board)):
        planes[:, :, CONV_PLANES + i] = value
    return planes


def fen_to_planes(fen: str) -> np.ndarray:
    return board_to_planes(chess.Board(fen))


def fen_to_castling_scalars(fen: str) -> np.ndarray:
    return castling_scalars(chess.Board(fen))


def encode_many(fens) -> np.ndarray:
    """Stack an iterable of FENs into an (N, 8, 8, 16) transport array."""
    fens = list(fens)
    out = np.zeros((len(fens), *BOARD_SHAPE), dtype=np.float32)
    for i, fen in enumerate(fens):
        out[i] = fen_to_planes(fen)
    return out


def representation_summary() -> dict:
    """Recorded in every run's metadata so the arm is self-describing."""
    return {
        "name": "planes12c4",
        "shape": list(BOARD_SHAPE),
        "n_planes": N_PLANES,
        "dtype": "float32",
        "values": "binary {0, 1}",
        "plane_names": PLANE_NAMES,
        "orientation": "absolute; row 0 = rank 8, col 0 = file a; never flipped",
        "conv_input_shape": list(CONV_INPUT_SHAPE),
        "conv_channels": list(range(CONV_PLANES)),
        "castling_channels": list(CASTLING_CHANNELS),
        "castling_feature_names": list(CASTLING_FEATURE_NAMES),
        "castling_enters_convolution": False,
        "castling_enters_dense_head": True,
        "encodes_side_to_move": False,
        "encodes_castling_rights": True,
        "encodes_en_passant": False,
        "planes_0_to_11_identical_to_planes12": True,
        "loadable_by_unmodified_engine": False,
        "note": "channels 12-15 are a TRANSPORT container for four scalars; the "
                "model slices them off before the first Conv2D and concatenates "
                "them after Flatten. See training/representation12c4.py.",
    }
