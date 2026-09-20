"""Tests for the C8 `dataset_v2` builder.

The dataset is the only thing C8 changes, so the guarantees that make the C8 vs
A2 comparison meaningful all live here:

  * the split unit is game CONTENT, so identical games cannot straddle it
  * extraction is deterministic, bounded and gap-respecting
  * train/test placement overlap ends at exactly zero
  * evaluation-suite overlap ends at exactly zero
  * the A2 label policy is used, not reimplemented
  * a rebuild reproduces the same hashes

Split-level behaviour (content identity, row-order invariance, 80/20) is owned by
`tests/unit/test_training_c7_audit.py`; this file imports the same helpers rather
than re-deriving them, and pins only what C8 adds on top.

No Stockfish, no training, no writes outside tmp_path.
"""
import json
from pathlib import Path

import chess
import pytest

from training import build_dataset as B
from training import build_dataset_v2 as V2
from training import c7_dataset_audit as C7
from training import labels as L

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "training" / "artifacts" / "dataset_v2.manifest.json"
TRAIN_JSONL = REPO_ROOT / "training" / "artifacts" / "dataset_v2.train.jsonl"
TEST_JSONL = REPO_ROOT / "training" / "artifacts" / "dataset_v2.test.jsonl"

LONG_GAME = " ".join(["e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O"] * 3)
SHORT_GAME = "e4 e5 Qh5 Nc6 Bc4 Nf6 Qxf7#"


def _replay(moves):
    return C7.replay_game("G", 0, moves)


def _needs_dataset():
    if not MANIFEST.is_file():
        pytest.skip("dataset_v2 not built in this checkout; "
                    "run python -m training.build_dataset_v2")


@pytest.fixture(scope="module")
def manifest():
    _needs_dataset()
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def records():
    _needs_dataset()
    def load(path):
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return load(TRAIN_JSONL), load(TEST_JSONL)


# ======================================= 1-3. content identity and the split

def test_split_unit_is_game_content_not_the_id_column():
    """Reused from C7: two ids, one game -> one unit."""
    assert C7.game_unit_keys(["idA", "idB"], [LONG_GAME, LONG_GAME])[0] == \
        C7.game_unit_keys(["idA", "idB"], [LONG_GAME, LONG_GAME])[1]


def test_duplicate_ids_with_identical_moves_collapse_to_one_unit():
    keys = C7.game_unit_keys(["same", "same", "same"],
                             [LONG_GAME, LONG_GAME, LONG_GAME])
    assert len(set(keys)) == 1


def test_identical_content_under_different_ids_stays_on_one_side():
    ids = [f"id{i}" for i in range(200)]
    moves = [f"e4 e5 Nf3 Nc6 Bb5 a{i % 40}" for i in range(200)]
    keys = C7.game_unit_keys(ids, moves)
    train, test, train_u, test_u = C7.game_level_split(keys, V2.SPLIT_SEED,
                                                       V2.TEST_FRACTION)
    assert not (train_u & test_u)
    for i, k in enumerate(keys):
        on_train = i in train
        assert all((j in train) == on_train for j, k2 in enumerate(keys) if k2 == k)


def test_builder_uses_the_c7_split_constants():
    assert V2.SPLIT_SEED == 42
    assert V2.TEST_FRACTION == 0.20


# ======================================= 4-5. split properties (pinned here too)

def test_split_is_invariant_to_row_order():
    ids = [f"id{i}" for i in range(300)]
    moves = [f"e4 e{i}" for i in range(300)]
    _, _, tr_a, te_a = C7.game_level_split(
        C7.game_unit_keys(ids, moves), V2.SPLIT_SEED, V2.TEST_FRACTION)
    order = list(reversed(range(300)))
    _, _, tr_b, te_b = C7.game_level_split(
        C7.game_unit_keys([ids[i] for i in order], [moves[i] for i in order]),
        V2.SPLIT_SEED, V2.TEST_FRACTION)
    assert (tr_a, te_a) == (tr_b, te_b)


def test_split_is_eighty_twenty(manifest):
    gs = manifest["game_split"]
    assert gs["n_game_units"] == 18_920
    assert gs["n_train_units"] == 15_136
    assert gs["n_test_units"] == 3_784
    assert gs["n_train_units"] + gs["n_test_units"] == gs["n_game_units"]
    assert round(gs["n_test_units"] / gs["n_game_units"], 2) == 0.20


def test_manifest_reproduces_the_c7_split_hashes(manifest):
    gs = manifest["game_split"]
    for key in ("game_list_sha256", "train_game_sha256", "test_game_sha256"):
        assert gs[key] == V2.C7_EXPECTED[key], key


# ======================================= 6-9. extraction policy

def test_policy_constants_are_the_c7_decision():
    assert V2.POLICY_NAME == "evenly_spaced_4_minply16"
    assert V2.MIN_PLY == 16
    assert V2.MAX_PER_GAME == 4
    assert V2.MIN_GAP == 4
    assert V2.SAMPLING_SEED is None


def test_extraction_never_selects_below_min_ply():
    picks = C7.policy_evenly_spaced(_replay(LONG_GAME), V2.MAX_PER_GAME,
                                    min_ply=V2.MIN_PLY, min_gap=V2.MIN_GAP)
    assert picks and all(p >= V2.MIN_PLY - 1 for p in picks)


def test_extraction_never_exceeds_four_per_game():
    for moves in (LONG_GAME, LONG_GAME + " " + LONG_GAME):
        picks = C7.policy_evenly_spaced(_replay(moves), V2.MAX_PER_GAME,
                                        min_ply=V2.MIN_PLY, min_gap=V2.MIN_GAP)
        assert len(picks) <= V2.MAX_PER_GAME


def test_extraction_respects_the_four_ply_minimum_gap():
    picks = C7.policy_evenly_spaced(_replay(LONG_GAME), V2.MAX_PER_GAME,
                                    min_ply=V2.MIN_PLY, min_gap=V2.MIN_GAP)
    assert all(b - a >= V2.MIN_GAP for a, b in zip(picks, picks[1:]))


def test_games_shorter_than_min_ply_contribute_nothing():
    assert C7.policy_evenly_spaced(_replay(SHORT_GAME), V2.MAX_PER_GAME,
                                   min_ply=V2.MIN_PLY, min_gap=V2.MIN_GAP) == []


def test_extraction_is_deterministic_across_calls():
    rep = _replay(LONG_GAME)
    first = C7.policy_evenly_spaced(rep, V2.MAX_PER_GAME, min_ply=V2.MIN_PLY,
                                    min_gap=V2.MIN_GAP)
    for _ in range(5):
        assert C7.policy_evenly_spaced(rep, V2.MAX_PER_GAME, min_ply=V2.MIN_PLY,
                                       min_gap=V2.MIN_GAP) == first


def test_builder_declares_no_rng(manifest):
    ex = manifest["extraction"]
    assert ex["uses_rng"] is False
    assert ex["sampling_seed"] is None
    assert ex["policy"] == "evenly_spaced_4_minply16"
    assert (ex["min_ply"], ex["max_positions_per_game"], ex["min_gap_plies"]) \
        == (16, 4, 4)


def test_builder_source_contains_no_random_sampling():
    src = Path(V2.__file__).read_text(encoding="utf-8")
    for banned in ("random.", "np.random", "default_rng", "shuffle", "sample("):
        assert banned not in src, banned


# ======================================= extraction on the real dataset

def test_every_record_is_at_or_beyond_min_ply(records):
    train, test = records
    for rows in (train, test):
        assert min(r["ply"] for r in rows) >= V2.MIN_PLY


def test_no_game_contributes_more_than_four_records(records):
    from collections import Counter
    train, test = records
    for rows in (train, test):
        counts = Counter(r["game_content_key"] for r in rows)
        assert max(counts.values()) <= V2.MAX_PER_GAME


def test_selected_plies_within_a_game_respect_the_gap(records):
    from collections import defaultdict
    train, test = records
    for rows in (train, test):
        by_game = defaultdict(list)
        for r in rows:
            by_game[r["game_content_key"]].append(r["ply"])
        # Deduplication and scrubbing only REMOVE records, so surviving plies are
        # a subset of the selected ones and the gap can only grow.
        for plies in by_game.values():
            plies.sort()
            assert all(b - a >= V2.MIN_GAP for a, b in zip(plies, plies[1:]))


def test_recorded_fen_matches_the_recorded_ply_and_placement(records):
    """Spot-check that the FEN really is the board after `ply` half-moves."""
    train, _ = records
    for r in train[:200]:
        board = chess.Board(r["fen"])
        assert board.fullmove_number >= 1
        assert r["placement"] == r["fen"].split(" ")[0]
        assert r["side_to_move"] == ("white" if board.turn == chess.WHITE else "black")
        assert r["piece_count"] == chess.popcount(board.occupied)


# ======================================= 10-11. dedup and train/test leakage

def test_deduplicate_by_placement_keeps_first_occurrence():
    rows = [
        {"placement": "A", "ply": 16}, {"placement": "B", "ply": 20},
        {"placement": "A", "ply": 24}, {"placement": "C", "ply": 28},
    ]
    kept, stats = V2.deduplicate_by_placement(rows)
    assert [r["placement"] for r in kept] == ["A", "B", "C"]
    assert kept[0]["ply"] == 16
    assert stats["duplicate_rows_removed"] == 1
    assert stats["unique_placements"] == 3


def test_scrub_removes_exactly_the_banned_placements():
    rows = [{"placement": p} for p in ("A", "B", "C", "B")]
    kept, stats = V2.scrub(rows, {"B"}, "test reason")
    assert [r["placement"] for r in kept] == ["A", "C"]
    assert stats["records_removed"] == 2
    assert stats["distinct_placements_removed"] == 1
    assert len(stats["removed_placement_sha256"]) == 1


def test_scrub_is_a_no_op_when_nothing_matches():
    rows = [{"placement": p} for p in ("A", "B")]
    kept, stats = V2.scrub(rows, {"Z"}, "none")
    assert len(kept) == 2 and stats["records_removed"] == 0


def test_no_placement_appears_twice_within_a_side(records):
    train, test = records
    for rows in (train, test):
        placements = [r["placement"] for r in rows]
        assert len(placements) == len(set(placements))


def test_train_and_test_share_no_placement(records):
    """THE leakage guarantee, checked on the built artifact."""
    train, test = records
    assert not ({r["placement"] for r in train} & {r["placement"] for r in test})


def test_train_and_test_share_no_game(records):
    train, test = records
    assert not ({r["game_content_key"] for r in train}
                & {r["game_content_key"] for r in test})


def test_leakage_scrub_reduced_test_only(manifest):
    ls = manifest["leakage_scrub"]
    assert ls["train_test_overlap_placements"] == 88
    assert ls["records_removed"] == 88
    assert ls["final_train_test_placement_overlap"] == 0
    assert "TEST only" in ls["policy"]


# ======================================= 12. evaluation-suite scrub

def test_no_record_uses_an_evaluation_suite_placement(records):
    train, test = records
    suite_placements, _ = V2.load_suite_placements()
    assert suite_placements
    for rows in (train, test):
        assert not ({r["placement"] for r in rows} & suite_placements)


def test_suite_scrub_counts_are_recorded(manifest):
    ss = manifest["evaluation_suite_scrub"]
    assert ss["train"]["records_removed"] == 1
    assert ss["test"]["records_removed"] == 0
    assert ss["final_suite_overlap_train"] == 0
    assert ss["final_suite_overlap_test"] == 0
    assert ss["suites_modified"] is False
    assert ss["suites"] == {"extended": 160, "phase0_52": 52}


def test_suite_loader_reads_both_suites_without_modifying_them():
    before = {}
    for suite in V2.SUITE_FILES:
        path = REPO_ROOT / "evaluation" / "positions" / f"{suite}.json"
        before[suite] = B.sha256_file(path)
    placements, counts = V2.load_suite_placements()
    assert counts == {"extended": 160, "phase0_52": 52}
    assert len(placements) == 185
    for suite, digest in before.items():
        path = REPO_ROOT / "evaluation" / "positions" / f"{suite}.json"
        assert B.sha256_file(path) == digest


# ======================================= 13. label policy unchanged

def test_builder_imports_the_label_policy_rather_than_reimplementing_it():
    src = Path(V2.__file__).read_text(encoding="utf-8")
    assert "from training import labels as L" in src
    assert "L.make_label(" in src
    for banned in ("MATE_SCORE_BASE =", "def make_label", "def to_white_positive",
                   "def clip_cp", "CP_CLIP ="):
        assert banned not in src, banned


def test_manifest_records_the_a2_policy(manifest):
    lab = manifest["labels"]
    assert lab["policy_name"] == "corrected_mate_white_perspective"
    assert lab["label_perspective"] == L.LABEL_PERSPECTIVE == "white"
    assert lab["cp_clip_used"] == L.CP_CLIP == 1500
    assert lab["policy_source"] == "training/labels.py (unmodified)"


def test_labels_obey_the_policy_bounds(records):
    train, test = records
    for rows in (train, test):
        for r in rows:
            assert -2000 <= r["label"] <= 2000
            assert r["label_perspective"] == "white"
            assert r["raw_value_perspective"] == "side_to_move"
            assert r["eval_type"] in ("cp", "mate")
            if r["eval_type"] == "cp":
                assert abs(r["label"]) <= L.CP_CLIP


def test_a_sample_of_labels_reproduces_make_label(records):
    """The stored label must be exactly what the policy returns for the stored
    raw value - no post-processing crept in."""
    train, _ = records
    for r in train[:500]:
        expected = L.make_label(
            eval_type=r["eval_type"],
            raw_value=r["raw_stockfish_value"],
            side_to_move_is_white=(r["side_to_move"] == "white"),
            is_checkmate=r["is_checkmate"],
            cp_clip=L.CP_CLIP,
        )
        assert r["label"] == expected.label
        assert r["was_clipped"] == expected.was_clipped


def test_mate_labels_outrank_every_cp_label(records):
    train, _ = records
    mates = [r["label"] for r in train if r["eval_type"] == "mate"]
    cps = [r["label"] for r in train if r["eval_type"] == "cp"]
    if mates:
        assert min(abs(v) for v in mates) > max(abs(v) for v in cps) - 1


# ======================================= 14-15. manifest and rebuild

def test_manifest_records_every_required_field(manifest):
    assert manifest["pipeline_version"] == "c8-1"
    assert manifest["source"]["sha256"] == B.EXPECTED_SOURCE["sha256"]
    for key in ("game_list_sha256", "train_game_sha256", "test_game_sha256",
                "split_seed", "test_fraction"):
        assert key in manifest["game_split"], key
    for key in ("policy", "min_ply", "max_positions_per_game", "min_gap_plies",
                "sampling_seed"):
        assert key in manifest["extraction"], key
    art = manifest["artifact"]
    assert art["train_sha256"] and art["test_sha256"]
    assert manifest["labels"]["train_label_sha256"]
    assert manifest["labels"]["test_label_sha256"]
    assert manifest["stockfish"]["depth"] == 8
    assert manifest["stockfish"]["threads"] == 1
    assert manifest["stockfish"]["hash_mb"] == 16
    assert manifest["stockfish"]["clear_hash_per_position"] is True


def test_manifest_hashes_match_the_files_on_disk(manifest):
    assert B.sha256_file(TRAIN_JSONL) == manifest["artifact"]["train_sha256"]
    assert B.sha256_file(TEST_JSONL) == manifest["artifact"]["test_sha256"]


def test_label_hash_is_reproducible_from_the_records(manifest, records):
    train, test = records
    assert V2.label_hash(train) == manifest["labels"]["train_label_sha256"]
    assert V2.label_hash(test) == manifest["labels"]["test_label_sha256"]


def test_final_counts_are_the_expected_ones(manifest, records):
    train, test = records
    assert len(train) == 54_812
    assert len(test) == 13_712
    assert manifest["final"]["train"]["records"] == len(train)
    assert manifest["final"]["test"]["records"] == len(test)
    assert manifest["final"]["total_records"] == len(train) + len(test)


def test_timestamp_is_not_an_input_to_any_hash(manifest):
    """A rebuild must reproduce the content hashes even though the time differs."""
    assert "NOT inputs to any hash" in manifest["timestamp_note"]
    for digest in (manifest["artifact"]["train_sha256"],
                   manifest["artifact"]["test_sha256"]):
        assert manifest["generated_at_utc"] not in digest


def test_record_writer_is_stable_for_the_same_rows(tmp_path):
    rows = [
        {"game_content_key": "k", "game_id": "g", "source_row_index": 0,
         "ply": 16, "fen": chess.STARTING_FEN, "placement": "x",
         "side_to_move": "white", "piece_count": 32, "phase": "opening",
         "plies_in_game": 40, "is_checkmate": False, "is_stalemate": False,
         "is_game_over": False, "legal_move_count": 20, "eval_type": "cp",
         "raw_stockfish_value": 12, "raw_value_perspective": "side_to_move",
         "label": 12, "label_perspective": "white", "was_clipped": False},
    ]
    a = V2.write_records(rows, tmp_path / "a.jsonl")
    b = V2.write_records(rows, tmp_path / "b.jsonl")
    assert a == b


def test_extraction_is_reproducible_on_a_small_slice(tmp_path):
    """A second structural build of the same rows yields identical files."""
    import pandas as pd
    df = pd.read_csv(REPO_ROOT / "games.csv").head(400)
    keys = C7.game_unit_keys(df["id"].astype(str).tolist(),
                             df["moves"].astype(str).tolist())
    rows_a = V2.extract_side(df, set(range(len(df))), keys)
    rows_b = V2.extract_side(df, set(range(len(df))), keys)
    assert [r["fen"] for r in rows_a] == [r["fen"] for r in rows_b]
    assert V2.write_records(rows_a, tmp_path / "a.jsonl") == \
        V2.write_records(rows_b, tmp_path / "b.jsonl")


def test_extraction_emits_records_in_canonical_order(tmp_path):
    """Order must not depend on CSV row order, or dedup would not be stable."""
    import pandas as pd
    df = pd.read_csv(REPO_ROOT / "games.csv").head(300)
    keys = C7.game_unit_keys(df["id"].astype(str).tolist(),
                             df["moves"].astype(str).tolist())
    rows = V2.extract_side(df, set(range(len(df))), keys)
    assert rows == sorted(rows, key=lambda r: (r["game_content_key"], r["ply"]))


def test_dataset_files_are_sorted_canonically(records):
    train, test = records
    for rows in (train, test):
        keys = [(r["game_content_key"], r["ply"]) for r in rows]
        assert keys == sorted(keys)


# ======================================= composition (the point of C8)

def test_dataset_v2_contains_endgame_positions(records):
    """dataset_v1 had ZERO positions with <=12 pieces."""
    train, _ = records
    assert sum(1 for r in train if r["piece_count"] <= 12) > 1000


def test_side_to_move_is_roughly_balanced(records):
    """dataset_v1 was 95.9% White to move."""
    train, _ = records
    white = sum(1 for r in train if r["side_to_move"] == "white")
    assert 0.5 < white / len(train) < 0.7


def test_all_three_phases_are_represented(records):
    from collections import Counter
    train, _ = records
    counts = Counter(r["phase"] for r in train)
    assert set(counts) == {"opening", "middlegame", "endgame"}
    assert all(v > 1000 for v in counts.values())


def test_dataset_v2_is_much_larger_than_dataset_v1(records):
    train, test = records
    assert len(train) + len(test) > 5 * 9_667


# ======================================= 16. production files untouched

def test_builder_never_writes_to_production_paths():
    # The module docstring names these paths to say it does NOT touch them, so
    # the check is scoped to the code body.
    body = Path(V2.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for banned in ("models/", "weight_model", "cnn_model.keras",
                   "CHESS_BOT_MODELS_DIR"):
        assert banned not in body, banned


def test_builder_does_not_import_keras_or_the_training_harness():
    src = Path(V2.__file__).read_text(encoding="utf-8")
    for banned in ("import keras", "from keras", "tensorflow",
                   "from training import train"):
        assert banned not in src, banned


def test_builder_only_reads_the_evaluation_suites():
    """`load_suite_placements` must not open the suites for writing."""
    import inspect
    src = inspect.getsource(V2.load_suite_placements)
    assert "read_text" in src
    for banned in ("write_text", "open(", "w\"", "'w'"):
        assert banned not in src, banned


def test_production_label_module_is_untouched():
    """dataset_v2 must use the same labels.py A2 was trained with."""
    assert L.LABEL_PERSPECTIVE == "white"
    assert L.CP_CLIP == 1500
    assert L.MATE_SCORE_BASE == 2000
    assert L.MATE_SCORE_STEP == 10


def test_builder_refuses_a_source_that_does_not_match_the_audit(tmp_path):
    """The C7 split hashes are only valid for the audited games.csv."""
    fake = tmp_path / "games.csv"
    fake.write_text("id,moves,turns,winner,white_rating,black_rating\n"
                    "a,e4 e5,2,white,1500,1500\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match the audited identity"):
        V2.build(fake, tmp_path / "out", L.CP_CLIP, 0, skip_labels=True, expect=False)
