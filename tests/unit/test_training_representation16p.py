"""Tests for the C6-A13P placebo representation (12 piece + 4 constant planes).

A13P is A13's control. Its whole value rests on two claims: the added channels
are shaped and counted exactly like A13's, and they carry no information about
the board. These tests pin both, and in particular try hard to FALSIFY the
second - if any board property leaked into planes 12-15, the control would be
worthless.
"""
import json
from pathlib import Path

import chess
import numpy as np
import pytest

from training import dataset as D
from training import representation as R12
from training import representation16 as R16
from training import representation16p as R16P
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
    assert R16P.BOARD_SHAPE == (8, 8, 16)
    assert R16P.N_PLANES == 16
    assert R16P.fen_to_planes(START).shape == (8, 8, 16)


def test_encoding_is_binary_float32():
    planes = R16P.fen_to_planes(ALL_RIGHTS)
    assert planes.dtype == np.float32
    assert set(np.unique(planes)).issubset({0.0, 1.0})


def test_plane_names_cover_every_channel():
    assert len(R16P.PLANE_NAMES) == 16
    assert R16P.PLANE_NAMES[:12] == R12.PLANE_NAMES
    assert all(n.startswith("placebo_constant_") for n in R16P.PLANE_NAMES[12:])


# ============================================ 2. channels 0-11 identical to A2

def test_first_twelve_planes_match_the_twelve_plane_encoder_on_both_suites():
    fens = _suite_fens()
    assert len(fens) > 200
    for fen in fens:
        board = chess.Board(fen)
        assert np.array_equal(R16P.board_to_planes(board)[:, :, :12],
                              R12.board_to_planes(board)), fen


def test_first_twelve_planes_match_the_engine_encoder():
    import engine as E
    for fen in (START, AFTER_E4, NO_CASTLING, ALL_RIGHTS):
        board = chess.Board(fen)
        assert np.array_equal(R16P.board_to_planes(board)[:, :, :12],
                              E.board_to_planes(board))


def test_first_twelve_planes_match_a13():
    """A13 and A13P must differ ONLY in channels 12-15."""
    for fen in _suite_fens()[:50]:
        board = chess.Board(fen)
        assert np.array_equal(R16P.board_to_planes(board)[:, :, :12],
                              R16.board_to_planes(board)[:, :, :12])


# ============================================ 3. channels 12-15 spatially constant

def test_placebo_planes_are_spatially_constant():
    for fen in (START, ALL_RIGHTS, NO_CASTLING, AFTER_E4):
        planes = R16P.fen_to_planes(fen)
        for idx in R16P.PLACEBO_PLANES:
            assert len(np.unique(planes[:, :, idx])) == 1, (fen, idx)


def test_placebo_planes_hold_the_declared_constant():
    planes = R16P.fen_to_planes(START)
    for idx in R16P.PLACEBO_PLANES:
        assert np.all(planes[:, :, idx] == R16P.PLACEBO_VALUE)


def test_placebo_value_is_nonzero():
    """An all-zero plane would be provably inert - no contribution, no gradient -
    so the control would test nothing beyond tensor width."""
    assert R16P.PLACEBO_VALUE != 0.0


def test_placebo_planes_are_channels_twelve_to_fifteen():
    assert R16P.PLACEBO_PLANES == (12, 13, 14, 15)


# ============================================ 4. NO board information (falsification)

def test_placebo_planes_are_identical_across_every_suite_position():
    """THE control's core claim. If any board property leaked in, some position
    would differ."""
    fens = _suite_fens()
    reference = R16P.fen_to_planes(fens[0])[:, :, 12:]
    for fen in fens:
        assert np.array_equal(R16P.fen_to_planes(fen)[:, :, 12:], reference), fen


def test_placebo_planes_are_identical_across_the_whole_dataset():
    path = REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl"
    if not path.is_file():
        pytest.skip("dataset_v1.jsonl not generated in this checkout")
    records = D.load_records(path)
    X = R16P.encode_many(r["fen"] for r in records)
    added = X[:, :, :, 12:]
    # zero variance across positions == zero information
    assert float(added.std()) == 0.0
    assert float(added.min()) == float(added.max()) == R16P.PLACEBO_VALUE


@pytest.mark.parametrize("a,b", [
    # castling rights differ
    (ALL_RIGHTS, "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"),
    # side to move differs
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/8/4K3 b - - 0 1"),
    # en-passant target differs
    (AFTER_E4, AFTER_E4.replace(" e3 ", " - ")),
    # entirely different positions
    (START, NO_CASTLING),
])
def test_no_board_property_reaches_the_placebo_planes(a, b):
    pa = R16P.fen_to_planes(a)[:, :, 12:]
    pb = R16P.fen_to_planes(b)[:, :, 12:]
    assert np.array_equal(pa, pb)


def test_placebo_planes_do_not_read_the_board_at_all():
    """Structural check: the encoder must produce the constants even for a board
    object whose accessors would raise if consulted."""
    class Exploding(chess.Board):
        def has_kingside_castling_rights(self, colour):
            raise AssertionError("placebo planes must not read castling rights")

        def has_queenside_castling_rights(self, colour):
            raise AssertionError("placebo planes must not read castling rights")

    planes = R16P.board_to_planes(Exploding(ALL_RIGHTS))
    for idx in R16P.PLACEBO_PLANES:
        assert np.all(planes[:, :, idx] == R16P.PLACEBO_VALUE)


def test_a13p_differs_from_a13_only_in_the_added_channels():
    """And it must actually differ there, or the arms would be identical."""
    board = chess.Board(NO_CASTLING)          # A13 gives zeros, placebo gives ones
    p16, p16p = R16.board_to_planes(board), R16P.board_to_planes(board)
    assert np.array_equal(p16[:, :, :12], p16p[:, :, :12])
    assert not np.array_equal(p16[:, :, 12:], p16p[:, :, 12:])


def test_a13_and_a13p_coincide_exactly_when_all_four_rights_are_present():
    """A consequence of PLACEBO_VALUE = 1.0 that the report must disclose: on a
    position with every castling right, A13's planes ARE all-ones, so the two
    encodings are byte-identical there. The arms differ only on the positions
    where at least one right is absent."""
    board = chess.Board(ALL_RIGHTS)
    assert np.array_equal(R16.board_to_planes(board), R16P.board_to_planes(board))


def test_summary_declares_the_placebo_and_the_exclusions():
    s = R16P.representation_summary()
    assert s["name"] == "planes16p"
    assert s["n_planes"] == 16
    assert s["added_planes_are_placebo"] is True
    assert s["added_planes_depend_on_position"] is False
    assert s["encodes_castling_rights"] is False
    assert s["encodes_side_to_move"] is False
    assert s["encodes_en_passant"] is False
    assert s["planes_0_to_11_identical_to_planes12"] is True
    assert s["loadable_by_unmodified_engine"] is False


# ============================================ 5 & 9. parameters and shape parity

def test_a13p_parameter_count_is_exactly_2_362_433():
    from training import train as T
    assert T.build_model(R16P).count_params() == 2_362_433


def test_a13_and_a13p_have_identical_shape_and_parameter_count():
    from training import train as T
    assert R16P.BOARD_SHAPE == R16.BOARD_SHAPE
    assert R16P.N_PLANES == R16.N_PLANES
    assert T.build_model(R16P).count_params() == T.build_model(R16).count_params()


def test_a13p_adds_the_same_2304_parameters_over_a2():
    from training import train as T
    assert (T.build_model(R16P).count_params()
            - T.build_model(R12).count_params()) == 3 * 3 * 4 * 64 == 2_304


# ============================================ 6 & 7. labels and split

def test_a13p_uses_exactly_the_same_labels_as_a2_and_a13():
    from training import train as T
    assert (T.ARMS["A13P"]["label_policy"] == T.ARMS["A2"]["label_policy"]
            == T.ARMS["A13"]["label_policy"])

    path = REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl"
    if not path.is_file():
        pytest.skip("dataset_v1.jsonl not generated in this checkout")
    records = D.load_records(path)
    ys = [D.apply_label_policy(records, T.ARMS[a]["label_policy"])
          for a in ("A2", "A13", "A13P")]
    assert np.array_equal(ys[0], ys[1])
    assert np.array_equal(ys[0], ys[2])


def test_a13p_uses_exactly_the_same_split_as_a2_and_a13():
    from training import train as T
    records = [{"fen": f"fen{i:04d}", "raw_stockfish_value": i - 50,
                "eval_type": "cp", "label": i - 50, "side_to_move": "white",
                "raw_value_perspective": "side_to_move", "is_checkmate": False}
               for i in range(400)]
    splits = [D.build_arm_data(records, T.ARMS[a]["label_policy"]).split
              for a in ("A2", "A13", "A13P")]
    for s in splits[1:]:
        assert np.array_equal(s.test_index, splits[0].test_index)
        assert np.array_equal(s.train_index, splits[0].train_index)
        assert s.split_seed == 42


def test_a13p_differs_from_a13_in_representation_only():
    from training import train as T
    a13, a13p = T.ARMS["A13"], T.ARMS["A13P"]
    differing = {k for k in set(a13) | set(a13p)
                 if a13.get(k) != a13p.get(k)} - {"description"}
    assert differing == {"representation"}


def test_a13p_differs_from_a2_in_representation_only():
    from training import train as T
    a2, a13p = T.ARMS["A2"], T.ARMS["A13P"]
    differing = {k for k in set(a2) | set(a13p)
                 if a2.get(k) != a13p.get(k)} - {"description"}
    assert differing == {"representation"}


# ============================================ registry and dispatch

def test_registry_resolves_planes16p():
    assert REPS.get("planes16p") is R16P
    assert set(REPS.NAMES) == {"planes12", "planes16", "planes16p", "planes18"}


def test_every_registered_encoder_exposes_the_same_surface():
    for name in REPS.NAMES:
        mod = REPS.get(name)
        for attr in ("BOARD_SHAPE", "N_PLANES", "PLANE_NAMES", "board_to_planes",
                     "fen_to_planes", "encode_many", "representation_summary"):
            assert hasattr(mod, attr), f"{name} is missing {attr}"


def test_encode_many_stacks_in_order():
    fens = [START, ALL_RIGHTS, NO_CASTLING]
    out = R16P.encode_many(fens)
    assert out.shape == (3, 8, 8, 16)
    for i, fen in enumerate(fens):
        assert np.array_equal(out[i], R16P.fen_to_planes(fen))


def test_ablation_tooling_knows_about_planes16p():
    from training import a3_plane_ablation as AB
    assert "planes16p" in AB.GROUPS_BY_REPRESENTATION
    assert AB.GROUPS_BY_REPRESENTATION["planes16p"]["placebo_constants"] == [12, 13, 14, 15]


# ============================================ 8. production untouched

def test_production_encoder_is_still_twelve_planes():
    import engine as E
    assert E.board_to_planes(chess.Board(START)).shape == (8, 8, 12)


def test_production_encoder_is_unchanged_by_a13p():
    """A13P must not have perturbed the shipped encoder in any way."""
    import engine as E
    for fen in (START, AFTER_E4, ALL_RIGHTS, NO_CASTLING):
        board = chess.Board(fen)
        assert np.array_equal(E.board_to_planes(board), R12.board_to_planes(board))
