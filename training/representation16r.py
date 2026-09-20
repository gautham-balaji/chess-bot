"""C6-A13R board tensor encoding: 12 piece planes + 4 hash-derived placebo planes.

A13R is the repaired control for A13, replacing the invalid A13P.

    plane  meaning                              values  spatial?  varies by position?
    0-11   the 12-plane piece encoding          0/1     yes       yes
    12-15  PLACEBO: one deterministic hash bit  0/1     constant  YES

--------------------------------------------------------------------------
WHY A13P WAS INVALID, AND WHAT A13R FIXES
--------------------------------------------------------------------------
A13P used FIXED constants (all-ones) for the four added channels. Every sample
in a batch then received the SAME offset in the first convolution, so
BatchNormalization's batch-mean subtraction removed it exactly and the training
loss became blind to those 2,304 weights. They drifted along an unconstrained
direction; validation loss (which uses BatchNorm's lagging running statistics)
oscillated by 6-13x; early stopping fired at epoch 11-14 and restore_best_weights
returned near-initialisation checkpoints. See docs/C6_A13P_REPORT.md section 11.

A13R's planes are constant WITHIN a board but DIFFER BETWEEN boards, exactly as
A13's castling planes do. The per-sample offset therefore varies across a batch,
no batch mean can absorb it, and the flat direction does not exist.

--------------------------------------------------------------------------
CONSTRUCTION
--------------------------------------------------------------------------
For plane i, the whole board is filled with a single bit:

    bit_i(fen) = sha256(f"{SALTS[i]}|{fen}").digest()[0] & 1

  - `fen` is the canonical python-chess FEN, `board.fen()`. Verified to
    round-trip exactly (`chess.Board(f).fen() == f`) on all 9,667 dataset
    records and all 212 evaluation positions.
  - SALTS are four distinct domain-separation strings, so the four planes are
    independent draws rather than four views of one hash.
  - SHA-256 is deterministic and platform independent, so the encoding is
    reproducible across runs and machines.

NO chess state is consulted. The generator reads the FEN string only, as an
opaque byte sequence, and never inspects castling rights, side to move,
en-passant, piece placement, legality, material or anything else. The string is
the hash input; the hash destroys its structure.

--------------------------------------------------------------------------
WHAT THIS PLACEBO MATCHES, AND WHAT IT DOES NOT
--------------------------------------------------------------------------
Measured against A13's castling planes over the 9,667 records:

    property                       A13 castling      A13R placebo
    marginal P(plane = 1)          0.507 - 0.547     0.498 - 0.504   matched
    correlation with the label     -0.006 to +0.025  -0.006 to +0.006 matched
    spatially constant per board   yes               yes             matched
    varies across positions        yes               yes             matched
    mean |inter-plane correlation| 0.431             0.003           NOT matched
    effective rank of the 4 planes 2                 4               NOT matched

A13's four rights are near-duplicates in pairs (WK-WQ r = 0.932, BK-BQ r = 0.908),
so they carry roughly two independent dimensions. Independent salts give four.
This is a deliberate consequence of the specified construction and is recorded
as a limitation in docs/C6_A13R_REPORT.md, not silently absorbed.

The placebo is also memorisable but not generalisable: a hash bit is a fixed
function of the position, so a network can fit it on the training split and gain
nothing on the test split. Real castling rights generalise. That difference is
intrinsic to any hash placebo and is likewise documented rather than fixed.
"""
from __future__ import annotations

import hashlib

import chess
import numpy as np

from training import representation as _r12

N_PLANES = 16
BOARD_SHAPE = (8, 8, N_PLANES)

PLACEBO_PLANES = (12, 13, 14, 15)

# Domain separation: four distinct salts so the planes are independent draws.
SALTS = (
    "C6-A13R/placebo-plane-0",
    "C6-A13R/placebo-plane-1",
    "C6-A13R/placebo-plane-2",
    "C6-A13R/placebo-plane-3",
)

PLANE_NAMES = _r12.PLANE_NAMES + [
    "placebo_hash_bit_0",
    "placebo_hash_bit_1",
    "placebo_hash_bit_2",
    "placebo_hash_bit_3",
]


def placebo_bit(salt: str, fen: str) -> int:
    """One deterministic, position-dependent, chess-meaningless bit.

    `fen` is treated as an opaque string. No parsing, no chess semantics.
    """
    digest = hashlib.sha256(f"{salt}|{fen}".encode("utf-8")).digest()
    return digest[0] & 1


def placebo_values(fen: str) -> list[int]:
    """The four placebo bits for a position, in plane order."""
    return [placebo_bit(salt, fen) for salt in SALTS]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a board as an (8, 8, 16) binary float32 tensor."""
    planes = np.zeros(BOARD_SHAPE, dtype=np.float32)

    # --- planes 0-11: identical to the 12-plane encoder ----------------------
    planes[:, :, :_r12.N_PLANES] = _r12.board_to_planes(board)

    # --- planes 12-15: hash bits of the canonical FEN ------------------------
    fen = board.fen()
    for idx, value in zip(PLACEBO_PLANES, placebo_values(fen)):
        if value:
            planes[:, :, idx] = 1.0

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
        "name": "planes16r",
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
        "added_planes_depend_on_position": True,
        "added_planes_construction": "sha256(salt|canonical_fen).digest()[0] & 1",
        "added_planes_salts": list(SALTS),
        "loadable_by_unmodified_engine": False,
        "note": "A13R: variation-matched placebo control for A13. Spatially "
                "constant per board and varying across boards, like A13's "
                "castling planes, but carrying no chess information. Repairs "
                "A13P's BatchNorm flat-direction flaw.",
    }
