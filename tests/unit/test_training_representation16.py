"""Tests for the C6-A13 16-plane representation (12 piece + 4 castling).

A13's claim is that it changes exactly one thing relative to A2: four castling
planes are appended and nothing else. These tests pin that claim from both
sides - what A13 adds, and what it must NOT contain (no side-to-move plane, no
en-passant plane), since those are the two A3 additions A13 exists to exclude.
"""
import json
from pathlib import Path

import chess
import numpy as np
import pytest

from training import dataset as D
from training import representation as R12
from training import representation16 as R16
from training import representation18 as R18
from training import representations as REPS

REPO_ROOT = Path(__file__).resolve().parents[2]

START = chess.STARTING_FEN
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
NO_CASTLING = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
ALL_RIGHTS = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"


def _suite_fens():
    fens = []
    for suite in ("extended", "phase0_52"):
        path = REPO_ROOT / "evaluation" / "positions" / f"{suite}.json"
        fens += [p["fen"] for p in json.loads(path.read_text(encoding="utf-8"))["positions"]]
    return fens


# ============================================ 1. shape

def test_shape_is_eight_by_eight_by_sixteen():
    assert R16.BOARD_SHAPE == (8, 8, 16)
    assert R16.N_PLANES == 16
    assert R16.fen_to_planes(START).shape == (8, 8, 16)


def test_encoding_is_binary_float32():
    planes = R16.fen_to_planes(ALL_RIGHTS)
    assert planes.dtype == np.float32
    assert set(np.unique(planes)).issubset({0.0, 1.0})


def test_plane_names_cover_every_channel():
    assert len(R16.PLANE_NAMES) == 16
    assert R16.PLANE_NAMES[:12] == R12.PLANE_NAMES
    assert R16.PLANE_NAMES[12:] == [
        "white_kingside_castling", "white_queenside_castling",
        "black_kingside_castling", "black_queenside_castling"]


# ============================================ 2. channels 0-11 unchanged

def test_first_twelve_planes_match_the_twelve_plane_encoder_on_both_suites():
    """THE drift guard: A13 appends, it never reorders or alters."""
    fens = _suite_fens()
    assert len(fens) > 200
    for fen in fens:
        board = chess.Board(fen)
        assert np.array_equal(R16.board_to_planes(board)[:, :, :12],
                              R12.board_to_planes(board)), fen


def test_first_twelve_planes_match_the_engine_encoder():
    import engine as E
    for fen in (START, AFTER_E4, NO_CASTLING, ALL_RIGHTS):
        board = chess.Board(fen)
        assert np.array_equal(R16.board_to_planes(board)[:, :, :12],
                              E.board_to_planes(board))


def test_first_twelve_planes_also_match_a3_s_first_twelve():
    """A13 and A3 share the same piece planes, so the arms stay comparable."""
    for fen in (START, AFTER_E4, ALL_RIGHTS):
        board = chess.Board(fen)
        assert np.array_equal(R16.board_to_planes(board)[:, :, :12],
                              R18.board_to_planes(board)[:, :, :12])


# ============================================ 3. castling planes

def test_castling_planes_are_set_when_all_rights_exist():
    planes = R16.fen_to_planes(START)
    for idx in (12, 13, 14, 15):
        assert np.all(planes[:, :, idx] == 1.0), idx


def test_castling_planes_are_zero_when_rights_are_gone():
    planes = R16.fen_to_planes(NO_CASTLING)
    for idx in (12, 13, 14, 15):
        assert np.all(planes[:, :, idx] == 0.0), idx


@pytest.mark.parametrize("fen,expected", [
    ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", (1, 0, 0, 0)),
    ("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", (0, 1, 0, 0)),
    ("4k2r/8/8/8/8/8/8/4K3 b k - 0 1", (0, 0, 1, 0)),
    ("r3k3/8/8/8/8/8/8/4K3 b q - 0 1", (0, 0, 0, 1)),
    (ALL_RIGHTS, (1, 1, 1, 1)),
])
def test_each_castling_plane_tracks_its_own_right(fen, expected):
    planes = R16.fen_to_planes(fen)
    got = tuple(int(planes[0, 0, i]) for i in (12, 13, 14, 15))
    assert got == expected


def test_castling_planes_are_spatially_constant():
    planes = R16.fen_to_planes(ALL_RIGHTS)
    for idx in (12, 13, 14, 15):
        assert len(np.unique(planes[:, :, idx])) == 1


def test_castling_planes_distinguish_boards_with_identical_pieces():
    """The information the 12-plane encoder cannot represent."""
    with_rights = chess.Board(ALL_RIGHTS)
    without = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert np.array_equal(R12.board_to_planes(with_rights),
                          R12.board_to_planes(without))
    assert not np.array_equal(R16.board_to_planes(with_rights),
                              R16.board_to_planes(without))


def test_castling_plane_order_matches_the_declared_mapping():
    assert R16.CASTLING_PLANES == {
        (chess.WHITE, "kingside"): 12, (chess.WHITE, "queenside"): 13,
        (chess.BLACK, "kingside"): 14, (chess.BLACK, "queenside"): 15}


# ============================================ 4 & 5. exclusions

def test_there_is_no_side_to_move_channel():
    """A13's defining exclusion. Two boards differing ONLY in side to move must
    encode identically - exactly as they do under 12 planes, and unlike A3."""
    w = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    b = chess.Board("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert np.array_equal(R16.board_to_planes(w), R16.board_to_planes(b))
    assert not np.array_equal(R18.board_to_planes(w), R18.board_to_planes(b))


def test_there_is_no_en_passant_channel():
    """Two boards differing ONLY in the en-passant target must encode
    identically under A13, and differently under A3."""
    with_ep = chess.Board(AFTER_E4)
    without_ep = chess.Board(AFTER_E4.replace(" e3 ", " - "))
    assert with_ep.ep_square is not None and without_ep.ep_square is None
    assert np.array_equal(R16.board_to_planes(with_ep),
                          R16.board_to_planes(without_ep))
    assert not np.array_equal(R18.board_to_planes(with_ep),
                              R18.board_to_planes(without_ep))


def test_summary_declares_the_exclusions():
    s = R16.representation_summary()
    assert s["name"] == "planes16"
    assert s["n_planes"] == 16
    assert s["encodes_castling_rights"] is True
    assert s["encodes_side_to_move"] is False
    assert s["encodes_en_passant"] is False
    assert s["planes_0_to_11_identical_to_planes12"] is True
    assert s["loadable_by_unmodified_engine"] is False


def test_a13_is_exactly_a3_minus_side_to_move_and_en_passant():
    """The four castling values must agree between the two encodings, even
    though they sit at different channel indices."""
    for fen in (START, ALL_RIGHTS, "4k3/8/8/8/8/8/8/4K2R w K - 0 1", NO_CASTLING):
        p16 = R16.fen_to_planes(fen)
        p18 = R18.fen_to_planes(fen)
        for i in range(4):
            assert np.array_equal(p16[:, :, 12 + i], p18[:, :, 13 + i]), (fen, i)


# ============================================ 6. parameter count

def test_a13_parameter_count_is_exactly_2_362_433():
    from training import train as T
    p12 = T.build_model(R12).count_params()
    p16 = T.build_model(R16).count_params()
    assert p12 == 2_360_129
    assert p16 == 2_362_433
    # only the first Conv2D kernel changes: 3x3x(16-12)x64
    assert p16 - p12 == 3 * 3 * 4 * 64 == 2_304


def test_a13_sits_between_a2_and_a3_in_size():
    from training import train as T
    assert (T.build_model(R12).count_params()
            < T.build_model(R16).count_params()
            < T.build_model(R18).count_params())


# ============================================ 7 & 8. labels and split

def test_a13_uses_exactly_the_same_labels_as_a2():
    from training import train as T
    assert T.ARMS["A13"]["label_policy"] == T.ARMS["A2"]["label_policy"]

    path = REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl"
    if not path.is_file():
        pytest.skip("dataset_v1.jsonl not generated in this checkout")
    records = D.load_records(path)
    a2 = D.apply_label_policy(records, T.ARMS["A2"]["label_policy"])
    a13 = D.apply_label_policy(records, T.ARMS["A13"]["label_policy"])
    assert np.array_equal(a2, a13)


def test_a13_uses_exactly_the_same_split_as_a2():
    from training import train as T
    records = [{"fen": f"fen{i:04d}", "raw_stockfish_value": i - 50,
                "eval_type": "cp", "label": i - 50, "side_to_move": "white",
                "raw_value_perspective": "side_to_move", "is_checkmate": False}
               for i in range(400)]
    s2 = D.build_arm_data(records, T.ARMS["A2"]["label_policy"]).split
    s13 = D.build_arm_data(records, T.ARMS["A13"]["label_policy"]).split
    assert np.array_equal(s2.test_index, s13.test_index)
    assert np.array_equal(s2.train_index, s13.train_index)
    assert s2.split_seed == s13.split_seed == 42


def test_a13_differs_from_a2_in_representation_only():
    from training import train as T
    a2, a13 = T.ARMS["A2"], T.ARMS["A13"]
    differing = {k for k in set(a2) | set(a13) if a2.get(k) != a13.get(k)} - {"description"}
    assert differing == {"representation"}, (
        f"A2 and A13 must differ in representation alone, also differ in {differing}")


def test_a13_differs_from_a3_in_representation_only():
    from training import train as T
    a3, a13 = T.ARMS["A3"], T.ARMS["A13"]
    differing = {k for k in set(a3) | set(a13) if a3.get(k) != a13.get(k)} - {"description"}
    assert differing == {"representation"}


# ============================================ registry and dispatch

def test_registry_resolves_planes16():
    assert REPS.get("planes16") is R16
    assert set(REPS.NAMES) == {"planes12", "planes16", "planes18"}


def test_every_registered_encoder_exposes_the_same_surface():
    for name in REPS.NAMES:
        mod = REPS.get(name)
        for attr in ("BOARD_SHAPE", "N_PLANES", "PLANE_NAMES", "board_to_planes",
                     "fen_to_planes", "encode_many", "representation_summary"):
            assert hasattr(mod, attr), f"{name} is missing {attr}"


def test_encode_many_stacks_in_order():
    fens = [START, ALL_RIGHTS, NO_CASTLING]
    out = R16.encode_many(fens)
    assert out.shape == (3, 8, 8, 16)
    for i, fen in enumerate(fens):
        assert np.array_equal(out[i], R16.fen_to_planes(fen))


def test_dispatch_sends_non_twelve_plane_arms_to_the_shim():
    """Keyed off plane count, so it covers planes16 without a name list."""
    import inspect
    from training import evaluate_arm as EA
    source = inspect.getsource(EA.run_suite)
    assert "N_PLANES == 12" in source
    assert "training.evaluate_planes_runner" in source
    assert inspect.signature(EA.run_suite).parameters["representation"].default == "planes12"


def test_shim_rejects_a_twelve_plane_representation():
    from training import evaluate_planes_runner as RUN
    with pytest.raises(SystemExit, match="needs no shim"):
        RUN.main(["--representation", "planes12", "--dataset", "x"])


def test_shim_requires_an_explicit_representation():
    from training import evaluate_planes_runner as RUN
    with pytest.raises(SystemExit, match="--representation is required"):
        RUN.main(["--dataset", "x"])


def test_legacy_eighteen_plane_entry_point_still_pins_planes18():
    """docs/C6_A3_REPORT.md documents this command; it must keep working."""
    from training import evaluate18_runner as OLD
    assert OLD.REPRESENTATION == "planes18"


# ============================================ 9. production untouched

def test_production_encoder_is_still_twelve_planes():
    """A13 must not have altered the shipped engine's encoder."""
    import engine as E
    assert E.board_to_planes(chess.Board(START)).shape == (8, 8, 12)


def test_production_encoder_ignores_castling_rights():
    """Restates the gap A13 works around: production cannot see these rights."""
    import engine as E
    a = E.board_to_planes(chess.Board(ALL_RIGHTS))
    b = E.board_to_planes(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"))
    assert np.array_equal(a, b)
