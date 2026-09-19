"""Tests for training/representation.py - the 12-plane encoder used by A0.

The most important test here is the DRIFT GUARD: training/representation.py is a
deliberate standalone copy of engine.board_to_planes, so it must be proven equal
to it rather than assumed. A0 is only a valid control if it trains on exactly the
representation the engine uses at inference.
"""
import json
from pathlib import Path

import chess
import numpy as np
import pytest

from training import representation as R

REPO_ROOT = Path(__file__).resolve().parents[2]


def _suite_fens(name):
    path = REPO_ROOT / "evaluation" / "positions" / f"{name}.json"
    return [p["fen"] for p in json.loads(path.read_text(encoding="utf-8"))["positions"]]


# ==================================================================== drift guard

@pytest.mark.parametrize("suite", ["phase0_52", "extended"])
def test_training_encoder_matches_the_engine_encoder(engine_mod, suite):
    """training/representation.py must be bit-identical to engine.board_to_planes.

    If this fails, A0 is training on a different representation than the engine
    evaluates with, and every arm comparison is invalid.
    """
    mismatches = [
        fen for fen in _suite_fens(suite)
        if not np.array_equal(R.fen_to_planes(fen),
                              engine_mod.board_to_planes(chess.Board(fen)))
    ]
    assert mismatches == [], f"{len(mismatches)} encoding mismatches, e.g. {mismatches[:3]}"


def test_encoders_agree_on_edge_case_positions(engine_mod):
    for fen in [
        chess.STARTING_FEN,
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",                      # kings only
        "8/4P3/8/8/8/8/8/k2K4 w - - 0 1",                     # promotion imminent
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",               # all castling rights
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",                  # en passant
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",  # checkmate
    ]:
        assert np.array_equal(R.fen_to_planes(fen),
                              engine_mod.board_to_planes(chess.Board(fen))), fen


# ==================================================================== shape / dtype

def test_shape_is_8x8x12():
    assert R.fen_to_planes(chess.STARTING_FEN).shape == (8, 8, 12) == R.BOARD_SHAPE


def test_dtype_is_float32():
    assert R.fen_to_planes(chess.STARTING_FEN).dtype == np.float32


def test_values_are_binary():
    planes = R.fen_to_planes(chess.STARTING_FEN)
    assert set(np.unique(planes)).issubset({0.0, 1.0})


def test_start_position_has_32_active_cells():
    assert R.fen_to_planes(chess.STARTING_FEN).sum() == 32


def test_channel_order_is_white_then_black():
    planes = R.fen_to_planes(chess.STARTING_FEN)
    assert list(planes.sum(axis=(0, 1))) == pytest.approx([8, 2, 2, 2, 1, 1] * 2)
    assert R.PLANE_NAMES[0] == "white_pawn" and R.PLANE_NAMES[6] == "black_pawn"


def test_encode_many_stacks_correctly():
    fens = _suite_fens("phase0_52")[:5]
    batch = R.encode_many(fens)
    assert batch.shape == (5, 8, 8, 12)
    assert np.array_equal(batch[2], R.fen_to_planes(fens[2]))


def test_encode_many_handles_an_empty_iterable():
    assert R.encode_many([]).shape == (0, 8, 8, 12)


# ==================================================================== A0 scope

def test_a0_representation_has_no_c6_planes():
    """A0 must NOT include the side-to-move / castling / en-passant planes."""
    summary = R.representation_summary()
    assert summary["n_planes"] == 12
    assert summary["encodes_side_to_move"] is False
    assert summary["encodes_castling_rights"] is False
    assert summary["encodes_en_passant"] is False


def test_side_to_move_remains_unrepresented_in_a0():
    """Characterisation: this is C6 and is deliberately still absent in A0."""
    white = R.fen_to_planes("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    black = R.fen_to_planes("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert np.array_equal(white, black)


def test_representation_summary_is_serialisable():
    json.dumps(R.representation_summary())
