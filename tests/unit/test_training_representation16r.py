"""Tests for the C6-A13R variation-matched placebo representation.

A13R repairs A13P. Its value rests on three claims, each of which these tests
try to falsify:

  - the added channels are spatially constant per board (like A13's),
  - they VARY across boards (unlike A13P's, which is the whole point),
  - they carry no chess information whatsoever.

Plus the usual controlled-experiment invariants: shape, parameter count, labels,
split, and that production is untouched.
"""
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import pytest

from training import dataset as D
from training import representation as R12
from training import representation16 as R16
from training import representation16p as R16P
from training import representation16r as R16R
from training import representations as REPS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl"

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


def _scalar_values(fens):
    """(N, 4) array of the four placebo values, one row per position."""
    return np.array([R16R.fen_to_planes(f)[0, 0, 12:] for f in fens], dtype=float)


# ============================================ 1. shape

def test_shape_is_eight_by_eight_by_sixteen():
    assert R16R.BOARD_SHAPE == (8, 8, 16)
    assert R16R.N_PLANES == 16
    assert R16R.fen_to_planes(START).shape == (8, 8, 16)


def test_encoding_is_binary_float32():
    planes = R16R.fen_to_planes(ALL_RIGHTS)
    assert planes.dtype == np.float32
    assert set(np.unique(planes)).issubset({0.0, 1.0})


def test_plane_names_cover_every_channel():
    assert len(R16R.PLANE_NAMES) == 16
    assert R16R.PLANE_NAMES[:12] == R12.PLANE_NAMES
    assert all(n.startswith("placebo_hash_bit_") for n in R16R.PLANE_NAMES[12:])


# ============================================ 2. channels 0-11 identical to A2

def test_first_twelve_planes_match_the_twelve_plane_encoder_on_both_suites():
    fens = _suite_fens()
    assert len(fens) > 200
    for fen in fens:
        board = chess.Board(fen)
        assert np.array_equal(R16R.board_to_planes(board)[:, :, :12],
                              R12.board_to_planes(board)), fen


def test_first_twelve_planes_match_the_engine_encoder():
    import engine as E
    for fen in (START, AFTER_E4, NO_CASTLING, ALL_RIGHTS):
        board = chess.Board(fen)
        assert np.array_equal(R16R.board_to_planes(board)[:, :, :12],
                              E.board_to_planes(board))


def test_first_twelve_planes_match_a13_and_a13p():
    for fen in _suite_fens()[:40]:
        board = chess.Board(fen)
        piece = R12.board_to_planes(board)
        for mod in (R16, R16P, R16R):
            assert np.array_equal(mod.board_to_planes(board)[:, :, :12], piece)


# ============================================ 3. spatially constant

def test_each_placebo_plane_is_spatially_constant():
    for fen in _suite_fens()[:60]:
        planes = R16R.fen_to_planes(fen)
        for idx in R16R.PLACEBO_PLANES:
            assert len(np.unique(planes[:, :, idx])) == 1, (fen, idx)


def test_placebo_planes_are_channels_twelve_to_fifteen():
    assert R16R.PLACEBO_PLANES == (12, 13, 14, 15)


# ============================================ 4. VARIES across positions

def test_placebo_values_vary_across_positions():
    """THE fix for A13P. If these were constant across boards, BatchNorm would
    absorb them and the arm would be untrainable again."""
    V = _scalar_values(_suite_fens())
    for i in range(4):
        assert V[:, i].std() > 0.3, f"plane {i} does not vary across positions"
        assert 0.3 < V[:, i].mean() < 0.7, f"plane {i} is nearly always the same value"


def test_all_sixteen_bit_patterns_occur_on_the_real_dataset():
    if not DATASET.is_file():
        pytest.skip("dataset_v1.jsonl not generated in this checkout")
    records = D.load_records(DATASET)
    V = _scalar_values(r["fen"] for r in records)
    assert len(set(map(tuple, V.astype(int)))) == 16


def test_placebo_is_not_the_a13p_fixed_constant():
    """A13P used all-ones; A13R must not collapse to that."""
    V = _scalar_values(_suite_fens())
    assert not np.all(V == 1.0)
    assert not np.all(V == 0.0)


# ============================================ 5 & 6. deterministic

def test_repeated_encoding_of_the_same_fen_is_identical():
    for fen in (START, AFTER_E4, ALL_RIGHTS):
        a = R16R.fen_to_planes(fen)
        b = R16R.fen_to_planes(fen)
        assert np.array_equal(a, b)


def test_encoding_is_independent_of_how_the_board_was_reached():
    """Same position via a move sequence vs via FEN must encode identically."""
    board = chess.Board()
    board.push_uci("e2e4")
    direct = chess.Board(AFTER_E4)
    assert board.fen() == direct.fen()
    assert np.array_equal(R16R.board_to_planes(board),
                          R16R.board_to_planes(direct))


def test_placebo_bit_matches_the_documented_formula():
    """Pin the exact construction, so a future refactor cannot silently change
    the encoding and invalidate stored A13R models."""
    for salt in R16R.SALTS:
        for fen in (START, ALL_RIGHTS):
            expected = hashlib.sha256(f"{salt}|{fen}".encode("utf-8")).digest()[0] & 1
            assert R16R.placebo_bit(salt, fen) == expected


def test_salts_are_distinct_for_domain_separation():
    assert len(set(R16R.SALTS)) == 4


def test_the_four_planes_are_not_copies_of_one_another():
    V = _scalar_values(_suite_fens())
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(V[:, i], V[:, j]), (i, j)


def test_canonical_fen_round_trips_so_the_hash_input_is_well_defined():
    """board.fen() is the hash input; it must equal the stored FEN, or the same
    position could hash differently depending on how it was loaded."""
    for fen in _suite_fens():
        assert chess.Board(fen).fen() == fen, fen


# ============================================ 7. no chess semantics consulted

def test_generator_reads_only_the_fen_string():
    """`placebo_values` takes a string. Give it a FEN-shaped string that is not a
    legal position and it must still produce bits - proof that nothing chess
    specific is consulted."""
    assert R16R.placebo_values("not-a-real-position") == [
        R16R.placebo_bit(s, "not-a-real-position") for s in R16R.SALTS]


def test_board_accessors_are_never_consulted_for_the_placebo():
    """A board whose chess-state accessors raise must still encode."""
    class Exploding(chess.Board):
        def has_kingside_castling_rights(self, colour):
            raise AssertionError("placebo must not read castling rights")

        def has_queenside_castling_rights(self, colour):
            raise AssertionError("placebo must not read castling rights")

        def is_legal(self, move):
            raise AssertionError("placebo must not read legality")

    planes = R16R.board_to_planes(Exploding(ALL_RIGHTS))
    assert planes.shape == (8, 8, 16)


def test_placebo_is_uncorrelated_with_the_label():
    """A13's castling planes correlate with the label at |r| <= 0.026. The
    placebo must be no more informative than that."""
    if not DATASET.is_file():
        pytest.skip("dataset_v1.jsonl not generated in this checkout")
    records = D.load_records(DATASET)
    y = D.apply_label_policy(records, D.LABEL_POLICY_CORRECTED_MATE_WHITE).astype(float)
    V = _scalar_values(r["fen"] for r in records)
    for i in range(4):
        assert abs(float(np.corrcoef(V[:, i], y)[0, 1])) < 0.03


# ============================================ 8. marginals match A13

def test_marginals_are_approximately_fifty_percent():
    if not DATASET.is_file():
        pytest.skip("dataset_v1.jsonl not generated in this checkout")
    records = D.load_records(DATASET)
    V = _scalar_values(r["fen"] for r in records)
    for i in range(4):
        assert 0.45 <= V[:, i].mean() <= 0.55, f"plane {i} marginal off target"


def test_marginals_are_comparable_to_a13_castling_planes():
    """A13's are 0.507-0.547; A13R's must land in the same broad regime."""
    if not DATASET.is_file():
        pytest.skip("dataset_v1.jsonl not generated in this checkout")
    records = D.load_records(DATASET)
    fens = [r["fen"] for r in records]
    a13 = np.array([R16.fen_to_planes(f)[0, 0, 12:] for f in fens], dtype=float)
    a13r = _scalar_values(fens)
    for i in range(4):
        assert abs(a13r[:, i].mean() - a13[:, i].mean()) < 0.10, i


# ============================================ 9. parameter count

def test_a13r_parameter_count_is_exactly_2_362_433():
    from training import train as T
    assert T.build_model(R16R).count_params() == 2_362_433


def test_all_three_sixteen_plane_arms_have_identical_size():
    from training import train as T
    counts = {T.build_model(m).count_params() for m in (R16, R16P, R16R)}
    assert counts == {2_362_433}
    assert R16R.BOARD_SHAPE == R16.BOARD_SHAPE == R16P.BOARD_SHAPE


# ============================================ 10. labels and split

def test_a13r_uses_exactly_the_same_labels_as_a2_and_a13():
    from training import train as T
    assert (T.ARMS["A13R"]["label_policy"] == T.ARMS["A2"]["label_policy"]
            == T.ARMS["A13"]["label_policy"])
    if not DATASET.is_file():
        pytest.skip("dataset_v1.jsonl not generated in this checkout")
    records = D.load_records(DATASET)
    ys = [D.apply_label_policy(records, T.ARMS[a]["label_policy"])
          for a in ("A2", "A13", "A13R")]
    assert np.array_equal(ys[0], ys[1]) and np.array_equal(ys[0], ys[2])


def test_a13r_uses_exactly_the_same_split_as_a2_and_a13():
    from training import train as T
    records = [{"fen": f"fen{i:04d}", "raw_stockfish_value": i - 50,
                "eval_type": "cp", "label": i - 50, "side_to_move": "white",
                "raw_value_perspective": "side_to_move", "is_checkmate": False}
               for i in range(400)]
    splits = [D.build_arm_data(records, T.ARMS[a]["label_policy"]).split
              for a in ("A2", "A13", "A13R")]
    for s in splits[1:]:
        assert np.array_equal(s.test_index, splits[0].test_index)
        assert np.array_equal(s.train_index, splits[0].train_index)
        assert s.split_seed == 42


def test_a13r_differs_from_a2_and_a13_in_representation_only():
    from training import train as T
    for other in ("A2", "A13"):
        a, b = T.ARMS[other], T.ARMS["A13R"]
        differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - {"description"}
        assert differing == {"representation"}, (other, differing)


# ============================================ registry / tooling

def test_registry_resolves_planes16r():
    assert REPS.get("planes16r") is R16R
    assert {"planes12", "planes16", "planes16p", "planes16r", "planes18"} <= set(REPS.NAMES)


def test_every_registered_encoder_exposes_the_same_surface():
    for name in REPS.NAMES:
        mod = REPS.get(name)
        for attr in ("BOARD_SHAPE", "N_PLANES", "PLANE_NAMES", "board_to_planes",
                     "fen_to_planes", "encode_many", "representation_summary"):
            assert hasattr(mod, attr), f"{name} is missing {attr}"


def test_encode_many_stacks_in_order():
    fens = [START, ALL_RIGHTS, NO_CASTLING]
    out = R16R.encode_many(fens)
    assert out.shape == (3, 8, 8, 16)
    for i, fen in enumerate(fens):
        assert np.array_equal(out[i], R16R.fen_to_planes(fen))


def test_ablation_tooling_knows_about_planes16r():
    from training import a3_plane_ablation as AB
    assert AB.GROUPS_BY_REPRESENTATION["planes16r"]["placebo_hash_bits"] == [12, 13, 14, 15]


def test_summary_declares_the_placebo_and_that_it_varies():
    s = R16R.representation_summary()
    assert s["name"] == "planes16r"
    assert s["added_planes_are_placebo"] is True
    assert s["added_planes_depend_on_position"] is True     # the A13P fix
    assert s["encodes_castling_rights"] is False
    assert s["encodes_side_to_move"] is False
    assert s["encodes_en_passant"] is False
    assert s["loadable_by_unmodified_engine"] is False


# ============================================ 11 & 12. production untouched

def test_production_encoder_is_still_twelve_planes():
    import engine as E
    assert E.board_to_planes(chess.Board(START)).shape == (8, 8, 12)


def test_production_encoder_is_unchanged():
    import engine as E
    for fen in (START, AFTER_E4, ALL_RIGHTS, NO_CASTLING):
        board = chess.Board(fen)
        assert np.array_equal(E.board_to_planes(board), R12.board_to_planes(board))


def test_no_production_path_consumes_the_sixteen_plane_model():
    """evaluate_arm sends non-12-plane arms to the experiment-only shim; the
    direct evaluator path stays reserved for 12-plane arms."""
    import inspect
    from training import evaluate_arm as EA
    source = inspect.getsource(EA.run_suite)
    assert "N_PLANES == 12" in source
    assert "training.evaluate_planes_runner" in source
    assert inspect.signature(EA.run_suite).parameters["representation"].default == "planes12"
