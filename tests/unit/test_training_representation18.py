"""Tests for the C6-A3 18-plane representation and its evaluation shim.

The audit's section K fixes the layout exactly, so these tests check the encoder
against that specification rather than against itself. The most important one is
the drift guard: planes 0-11 must remain byte-identical to the 12-plane encoder,
because A3's whole claim is that it changes ONLY the representation by appending.
"""
import json
from pathlib import Path

import chess
import numpy as np
import pytest

from training import representation as R12
from training import representation18 as R18
from training import representations as REPS

REPO_ROOT = Path(__file__).resolve().parents[2]

START = chess.STARTING_FEN
# after 1. e4 -- Black to move, en-passant target on e3, all castling intact
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
NO_CASTLING = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


def _suite_fens():
    fens = []
    for suite in ("extended", "phase0_52"):
        path = REPO_ROOT / "evaluation" / "positions" / f"{suite}.json"
        fens += [p["fen"] for p in json.loads(path.read_text(encoding="utf-8"))["positions"]]
    return fens


# ==================================================================== shape

def test_shape_is_eight_by_eight_by_eighteen():
    assert R18.BOARD_SHAPE == (8, 8, 18)
    assert R18.N_PLANES == 18
    assert R18.fen_to_planes(START).shape == (8, 8, 18)


def test_encoding_is_binary_float32():
    planes = R18.fen_to_planes(AFTER_E4)
    assert planes.dtype == np.float32
    assert set(np.unique(planes)).issubset({0.0, 1.0})


def test_plane_names_cover_every_channel():
    assert len(R18.PLANE_NAMES) == 18
    assert R18.PLANE_NAMES[:12] == R12.PLANE_NAMES
    assert R18.PLANE_NAMES[12] == "side_to_move_is_white"
    assert R18.PLANE_NAMES[17] == "en_passant_target"


# ==================================================================== drift guard

def test_first_twelve_planes_match_the_twelve_plane_encoder_on_both_suites():
    """THE critical guard: A3 appends, it never reorders or alters."""
    fens = _suite_fens()
    assert len(fens) > 200
    for fen in fens:
        board = chess.Board(fen)
        assert np.array_equal(R18.board_to_planes(board)[:, :, :12],
                              R12.board_to_planes(board)), fen


def test_first_twelve_planes_match_the_engine_encoder():
    """The 12-plane encoder is itself pinned to engine.board_to_planes, so this
    transitively pins A3's piece planes to production."""
    import engine as E
    for fen in (START, AFTER_E4, NO_CASTLING):
        board = chess.Board(fen)
        assert np.array_equal(R18.board_to_planes(board)[:, :, :12],
                              E.board_to_planes(board))


# ==================================================================== plane 12

def test_side_to_move_plane_is_all_ones_for_white_and_all_zeros_for_black():
    white = R18.fen_to_planes(START)[:, :, 12]
    black = R18.fen_to_planes(AFTER_E4)[:, :, 12]
    assert np.all(white == 1.0)
    assert np.all(black == 0.0)


def test_side_to_move_plane_makes_identical_placements_distinguishable():
    """This is the defect A3 exists to fix, and the xfail
    test_side_to_move_should_be_representable documents it for 12 planes."""
    w = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    b = chess.Board("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert np.array_equal(R12.board_to_planes(w), R12.board_to_planes(b))
    assert not np.array_equal(R18.board_to_planes(w), R18.board_to_planes(b))


# ==================================================================== planes 13-16

def test_castling_planes_are_set_from_the_board_rights():
    planes = R18.fen_to_planes(START)
    for idx in (13, 14, 15, 16):
        assert np.all(planes[:, :, idx] == 1.0), idx


def test_castling_planes_are_zero_when_rights_are_gone():
    planes = R18.fen_to_planes(NO_CASTLING)
    for idx in (13, 14, 15, 16):
        assert np.all(planes[:, :, idx] == 0.0), idx


def test_each_castling_plane_tracks_its_own_right():
    """White kingside only."""
    planes = R18.fen_to_planes("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    assert np.all(planes[:, :, 13] == 1.0)   # white kingside
    assert np.all(planes[:, :, 14] == 0.0)   # white queenside
    assert np.all(planes[:, :, 15] == 0.0)   # black kingside
    assert np.all(planes[:, :, 16] == 0.0)   # black queenside


def test_castling_planes_distinguish_boards_with_identical_pieces():
    with_rights = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    without = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert np.array_equal(R12.board_to_planes(with_rights),
                          R12.board_to_planes(without))
    assert not np.array_equal(R18.board_to_planes(with_rights),
                              R18.board_to_planes(without))


# ==================================================================== plane 17

def test_en_passant_plane_marks_exactly_one_square():
    planes = R18.fen_to_planes(AFTER_E4)
    ep = planes[:, :, 17]
    assert ep.sum() == 1.0
    # e3 is square 20 -> row 7 - 20//8 = 5, col 20 % 8 = 4
    assert ep[5, 4] == 1.0


def test_en_passant_plane_uses_the_same_orientation_as_the_piece_planes():
    """Derive the expected cell from the shared formula, not from a constant."""
    board = chess.Board(AFTER_E4)
    row = 7 - (board.ep_square // 8)
    col = board.ep_square % 8
    assert R18.board_to_planes(board)[row, col, 17] == 1.0


def test_en_passant_plane_is_all_zero_when_there_is_no_target():
    assert R18.fen_to_planes(START)[:, :, 17].sum() == 0.0
    assert chess.Board(START).ep_square is None


# ==================================================================== summary

def test_summary_reports_the_three_gaps_as_closed():
    s = R18.representation_summary()
    assert s["name"] == "planes18"
    assert s["n_planes"] == 18
    assert s["encodes_side_to_move"] is True
    assert s["encodes_castling_rights"] is True
    assert s["encodes_en_passant"] is True
    assert s["planes_0_to_11_identical_to_planes12"] is True


def test_summary_records_that_the_shipped_engine_cannot_load_it():
    assert R18.representation_summary()["loadable_by_unmodified_engine"] is False


def test_twelve_plane_summary_still_reports_the_gaps_as_open():
    """A3 must not have quietly changed what A0/A1/A2 recorded."""
    s = R12.representation_summary()
    assert s["n_planes"] == 12
    assert s["encodes_side_to_move"] is False
    assert s["encodes_castling_rights"] is False
    assert s["encodes_en_passant"] is False


# ==================================================================== registry

def test_registry_resolves_both_representations():
    assert REPS.get("planes12") is R12
    assert REPS.get("planes18") is R18


def test_registry_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown representation"):
        REPS.get("planes24")


def test_every_registered_encoder_exposes_the_same_surface():
    for name in REPS.NAMES:
        mod = REPS.get(name)
        for attr in ("BOARD_SHAPE", "N_PLANES", "PLANE_NAMES", "board_to_planes",
                     "fen_to_planes", "encode_many", "representation_summary"):
            assert hasattr(mod, attr), f"{name} is missing {attr}"


def test_encode_many_stacks_in_order():
    fens = [START, AFTER_E4, NO_CASTLING]
    out = R18.encode_many(fens)
    assert out.shape == (3, 8, 8, 18)
    for i, fen in enumerate(fens):
        assert np.array_equal(out[i], R18.fen_to_planes(fen))


# ==================================================================== arm wiring

def test_a3_is_the_only_eighteen_plane_arm():
    from training import train as T
    assert sorted(T.ARMS) == ["A0", "A1", "A13", "A2", "A3"]
    assert T.ARMS["A3"]["representation"] == "planes18"
    for arm in ("A0", "A1", "A2"):
        assert T.ARMS[arm]["representation"] == "planes12"
    # A13 is 16 planes, so A3 remains the only 18-plane arm
    assert T.ARMS["A13"]["representation"] == "planes16"
    assert [a for a, spec in T.ARMS.items()
            if spec["representation"] == "planes18"] == ["A3"]


def test_a3_differs_from_a2_in_representation_only():
    """A3's single experimental variable."""
    from training import train as T
    a2, a3 = T.ARMS["A2"], T.ARMS["A3"]
    assert a2["label_policy"] == a3["label_policy"]
    differing = {k for k in set(a2) | set(a3) if a2.get(k) != a3.get(k)} - {"description"}
    assert differing == {"representation"}, (
        f"A2 and A3 must differ in representation alone, also differ in {differing}")


def test_a3_adds_exactly_the_audit_s_predicted_parameter_count():
    """Audit section K: 3x3x12x64 -> 3x3x18x64 adds 3,456 parameters."""
    from training import train as T
    p12 = T.build_model(R12).count_params()
    p18 = T.build_model(R18).count_params()
    assert p12 == 2_360_129
    assert p18 - p12 == 3 * 3 * 6 * 64 == 3_456
    assert p18 == 2_363_585


# ==================================================================== eval shim

def test_engine_resolves_board_to_planes_through_the_module_global(monkeypatch):
    """The shim rebinds one name. That only works if every call site looks the
    name up on the module at call time rather than capturing it.

    Verified behaviourally rather than by grepping: rebind the encoder to a
    counting wrapper around the REAL 12-plane one, then drive the engine and
    check the wrapper was actually used. `rerank_moves` exercises both the
    candidate-scoring path and the 1-ply lookahead path.
    """
    import engine as E

    calls = []
    real = E.board_to_planes

    def counting(board):
        calls.append(1)
        return real(board)

    monkeypatch.setattr(E, "board_to_planes", counting)

    E.cnn_evaluate(chess.Board(START))
    assert calls, "cnn_evaluate did not go through the module global"

    before = len(calls)
    E.rerank_moves(chess.Board(START))
    assert len(calls) > before, "rerank_moves did not go through the module global"


def test_shim_targets_the_unmodified_evaluator():
    """The implementation moved to evaluate_planes_runner when A13 added a
    16-plane arm; the mechanism is unchanged."""
    from training import evaluate_planes_runner as RUN
    source = Path(RUN.__file__).read_text(encoding="utf-8")
    assert "from evaluation import evaluate as ev" in source
    assert "ev.main()" in source


def test_eighteen_plane_entry_point_delegates_to_the_generic_shim():
    """docs/C6_A3_REPORT.md documents `python -m training.evaluate18_runner`."""
    from training import evaluate18_runner as OLD
    from training import evaluate_planes_runner as RUN
    assert OLD.REPRESENTATION == "planes18"
    assert OLD._runner is RUN


def test_run_suite_defaults_to_the_direct_evaluator_path():
    """Default-preserving guarantee for the A0/A1/A2 runs."""
    import inspect
    from training import evaluate_arm as EA
    sig = inspect.signature(EA.run_suite)
    assert sig.parameters["representation"].default == "planes12"
    source = inspect.getsource(EA.run_suite)
    assert 'evaluation" / "evaluate.py"' in source
    assert "training.evaluate_planes_runner" in source
