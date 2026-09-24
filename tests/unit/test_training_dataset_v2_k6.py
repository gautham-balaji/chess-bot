"""Tests for C8b: the `dataset_v2_k6` scaling dataset and its training arm.

C8b's claim is "identical to C8a except positions-per-game". These tests pin
that from both sides:

  * the k=6 dataset obeys every C7/C8 invariant the k=4 one does - min ply 16,
    <=6 per game, 4-ply gap, placement dedup, zero train/test placement overlap,
    zero evaluation-suite overlap
  * it uses the SAME game-level split as C8a, so the two arms differ only in
    how densely the same train games are sampled
  * parameterising the builder did not change C8a's dataset_v2
  * the C8b training arm reuses A2's architecture, recipe and label policy

Structural policy behaviour is owned by `test_training_build_dataset_v2.py` and
`test_training_c7_audit.py`; this file imports the same helpers rather than
re-deriving them and pins only what C8b adds.

No Stockfish, no training, no writes outside tmp_path.
"""
import json
from pathlib import Path

import chess
import pytest

from training import build_dataset as B
from training import build_dataset_v2 as V2B
from training import c7_dataset_audit as C7
from training import labels as L
from training import train_v2 as V2T

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "training" / "artifacts"

K6_POLICY = "evenly_spaced_6_minply16"
K4_POLICY = "evenly_spaced_4_minply16"
K6_PREFIX = ARTIFACTS / "dataset_v2_k6"
K4_MANIFEST = ARTIFACTS / "dataset_v2.manifest.json"

LONG_GAME = " ".join(["e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O"] * 4)


def _needs_k6():
    if not Path(f"{K6_PREFIX}.manifest.json").is_file():
        pytest.skip("dataset_v2_k6 not built; run "
                    "python -m training.build_dataset_v2 "
                    "--policy evenly_spaced_6_minply16")


@pytest.fixture(scope="module")
def manifest():
    _needs_k6()
    return json.loads(Path(f"{K6_PREFIX}.manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def records():
    _needs_k6()
    def load(p):
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    return (load(Path(f"{K6_PREFIX}.train.jsonl")),
            load(Path(f"{K6_PREFIX}.test.jsonl")))


# ======================================= policy registration

def test_both_policies_are_registered():
    assert sorted(V2B.POLICIES) == [K4_POLICY, K6_POLICY]


def test_the_default_policy_is_still_c8as():
    assert V2B.DEFAULT_POLICY == K4_POLICY
    assert V2B.POLICY_NAME == K4_POLICY
    assert V2B.MAX_PER_GAME == 4
    assert V2B.MIN_PLY == 16


def test_k6_differs_from_k4_only_in_positions_per_game():
    k4, k6 = V2B.POLICIES[K4_POLICY], V2B.POLICIES[K6_POLICY]
    assert k4["min_ply"] == k6["min_ply"] == 16
    assert k4["min_gap"] == k6["min_gap"] == 4
    assert (k4["max_per_game"], k6["max_per_game"]) == (4, 6)
    differing = {k for k in set(k4) | set(k6) if k4.get(k) != k6.get(k)}
    assert differing == {"max_per_game", "default_out", "dataset_name", "expected"}


def test_both_policies_declare_the_same_game_split():
    """The split is a property of games.csv and the seed, not of the policy."""
    for key in ("game_list_sha256", "train_game_sha256", "test_game_sha256"):
        assert (V2B.POLICIES[K4_POLICY]["expected"][key]
                == V2B.POLICIES[K6_POLICY]["expected"][key]
                == V2B.SPLIT_HASHES[key])
    for key in ("game_units", "train_units", "test_units"):
        assert (V2B.POLICIES[K4_POLICY]["expected"][key]
                == V2B.POLICIES[K6_POLICY]["expected"][key])


def test_builder_rejects_an_unknown_policy(tmp_path):
    with pytest.raises(SystemExit, match="unknown policy"):
        V2B.build(REPO_ROOT / "games.csv", tmp_path / "out", L.CP_CLIP, 0,
                  skip_labels=True, expect=False, policy="evenly_spaced_9")


def test_parameterising_the_builder_did_not_change_c8as_expectations():
    """Regression guard: dataset_v2's recorded expectations are untouched."""
    ex = V2B.POLICIES[K4_POLICY]["expected"]
    assert ex["selected_total"] == 68_901
    assert ex["deduped_train"] == 54_813
    assert ex["deduped_test"] == 13_800
    assert ex["train_test_overlap_placements"] == 88
    assert ex["final_test"] == 13_712


def test_c8as_committed_manifest_still_describes_the_k4_policy():
    """The parameterisation must not have rewritten C8a's dataset."""
    if not K4_MANIFEST.is_file():
        pytest.skip("dataset_v2 manifest not present")
    m = json.loads(K4_MANIFEST.read_text(encoding="utf-8"))
    assert m["extraction"]["policy"] == K4_POLICY
    assert m["extraction"]["max_positions_per_game"] == 4
    assert m["final"]["train"]["records"] == 54_812
    assert m["final"]["test"]["records"] == 13_712


# ======================================= extraction respects the k=6 bounds

def test_extract_side_honours_the_max_per_game_argument():
    rep = C7.replay_game("G", 0, LONG_GAME)
    for k in (4, 6):
        picks = C7.policy_evenly_spaced(rep, k, min_ply=16, min_gap=4)
        assert len(picks) <= k
    assert len(C7.policy_evenly_spaced(rep, 6, min_ply=16, min_gap=4)) >= \
        len(C7.policy_evenly_spaced(rep, 4, min_ply=16, min_gap=4))


def test_every_k6_record_is_at_or_beyond_ply_16(records):
    train, test = records
    for rows in (train, test):
        assert min(r["ply"] for r in rows) >= 16


def test_no_game_contributes_more_than_six_k6_records(records):
    from collections import Counter
    train, test = records
    for rows in (train, test):
        assert max(Counter(r["game_content_key"] for r in rows).values()) <= 6


def test_k6_selected_plies_respect_the_four_ply_gap(records):
    from collections import defaultdict
    train, test = records
    for rows in (train, test):
        by_game = defaultdict(list)
        for r in rows:
            by_game[r["game_content_key"]].append(r["ply"])
        for plies in by_game.values():
            plies.sort()
            assert all(b - a >= 4 for a, b in zip(plies, plies[1:]))


def test_k6_is_strictly_larger_than_k4(manifest):
    assert manifest["final"]["train"]["records"] > 54_812
    assert manifest["final"]["test"]["records"] > 13_712


# ======================================= split, dedup, leakage

def test_k6_uses_the_same_game_split_as_c8a(manifest):
    gs = manifest["game_split"]
    for key in ("game_list_sha256", "train_game_sha256", "test_game_sha256"):
        assert gs[key] == V2B.SPLIT_HASHES[key]
    assert gs["split_seed"] == 42
    assert gs["n_train_units"] == 15_136
    assert gs["n_test_units"] == 3_784


def test_k6_train_games_are_the_same_games_as_c8as(records):
    """Same split => C8b samples the SAME train games, just more densely."""
    if not (ARTIFACTS / "dataset_v2.train.jsonl").is_file():
        pytest.skip("dataset_v2 train file not built")
    train, _ = records
    k4 = {json.loads(l)["game_content_key"] for l in
          (ARTIFACTS / "dataset_v2.train.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()}
    k6 = {r["game_content_key"] for r in train}
    assert k4 <= k6


def test_no_placement_repeats_within_a_k6_side(records):
    train, test = records
    for rows in (train, test):
        placements = [r["placement"] for r in rows]
        assert len(placements) == len(set(placements))


def test_k6_train_and_test_share_no_placement(records):
    train, test = records
    assert not ({r["placement"] for r in train} & {r["placement"] for r in test})


def test_k6_train_and_test_share_no_game(records):
    train, test = records
    assert not ({r["game_content_key"] for r in train}
                & {r["game_content_key"] for r in test})


def test_k6_uses_no_evaluation_suite_placement(records):
    train, test = records
    suite, _ = V2B.load_suite_placements()
    for rows in (train, test):
        assert not ({r["placement"] for r in rows} & suite)


def test_k6_leakage_and_suite_scrubs_are_recorded(manifest):
    ls = manifest["leakage_scrub"]
    assert ls["train_test_overlap_placements"] == 90
    assert ls["records_removed"] == 90
    assert ls["final_train_test_placement_overlap"] == 0
    ss = manifest["evaluation_suite_scrub"]
    assert ss["final_suite_overlap_train"] == 0
    assert ss["final_suite_overlap_test"] == 0
    assert ss["suites_modified"] is False


# ======================================= labels unchanged

def test_k6_uses_the_a2_label_policy(manifest):
    lab = manifest["labels"]
    assert lab["policy_name"] == "corrected_mate_white_perspective"
    assert lab["label_perspective"] == "white"
    assert lab["cp_clip_used"] == L.CP_CLIP == 1500
    assert lab["policy_source"] == "training/labels.py (unmodified)"


def test_k6_labels_reproduce_make_label(records):
    train, _ = records
    for r in train[:500]:
        expected = L.make_label(
            eval_type=r["eval_type"], raw_value=r["raw_stockfish_value"],
            side_to_move_is_white=(r["side_to_move"] == "white"),
            is_checkmate=r["is_checkmate"], cp_clip=L.CP_CLIP)
        assert r["label"] == expected.label


def test_k6_keeps_checkmate_records(records):
    """C8's decision was to keep them; C8b must preserve that exactly."""
    train, _ = records
    assert sum(1 for r in train if r["is_checkmate"]) > 4_000


def test_k6_labels_stay_within_the_policy_bounds(records):
    train, test = records
    for rows in (train, test):
        for r in rows:
            assert -2000 <= r["label"] <= 2000
            assert r["label_perspective"] == "white"


# ======================================= manifest / reproducibility

def test_k6_manifest_hashes_match_the_files(manifest):
    assert B.sha256_file(Path(f"{K6_PREFIX}.train.jsonl")) == \
        manifest["artifact"]["train_sha256"]
    assert B.sha256_file(Path(f"{K6_PREFIX}.test.jsonl")) == \
        manifest["artifact"]["test_sha256"]


def test_k6_manifest_records_the_policy(manifest):
    ex = manifest["extraction"]
    assert ex["policy"] == K6_POLICY
    assert ex["min_ply"] == 16
    assert ex["max_positions_per_game"] == 6
    assert ex["min_gap_plies"] == 4
    assert ex["sampling_seed"] is None
    assert ex["uses_rng"] is False
    assert manifest["dataset_name"] == "dataset_v2_k6"
    assert manifest["pipeline_version"] == "c8-1"


def test_k6_final_counts_match_the_manifest(manifest, records):
    train, test = records
    assert len(train) == manifest["final"]["train"]["records"]
    assert len(test) == manifest["final"]["test"]["records"]


def test_k6_records_are_in_canonical_order(records):
    for rows in records:
        keys = [(r["game_content_key"], r["ply"]) for r in rows]
        assert keys == sorted(keys)


def test_k6_fens_are_self_consistent(records):
    train, _ = records
    for r in train[:200]:
        board = chess.Board(r["fen"])
        assert r["placement"] == r["fen"].split(" ")[0]
        assert r["side_to_move"] == ("white" if board.turn == chess.WHITE else "black")
        assert r["piece_count"] == chess.popcount(board.occupied)


# ======================================= the C8b training arm

def test_c8b_arm_points_at_the_k6_dataset():
    assert V2T.ARMS["C8b"]["dataset_prefix"] == "training/artifacts/dataset_v2_k6"


def test_c8b_matches_c8a_in_everything_but_the_dataset():
    a, b = V2T.ARMS["C8a"], V2T.ARMS["C8b"]
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differing == {"dataset_prefix", "description"}


def test_c8b_uses_a2s_label_policy_and_representation():
    from training import dataset as D
    spec = V2T.ARMS["C8b"]
    assert spec["label_policy"] == D.LABEL_POLICY_CORRECTED_MATE_WHITE
    assert spec["representation"] == "planes12"


def test_c8b_loads_through_the_same_verified_loader():
    _needs_k6()
    bundle = V2T.load_split(K6_PREFIX)
    assert len(bundle["train"]) == 78_901
    assert len(bundle["test"]) == 19_802


def test_c8b_is_routed_through_the_direct_evaluator():
    """12-plane arm: it must not use the experiment-only shim."""
    import inspect
    from training import evaluate_arm as EA
    from training import representations as REPS
    from training import train as T
    rep = T.ARMS.get("C8b", {}).get("representation", "planes12")
    assert REPS.get(rep).N_PLANES == 12
    assert "N_PLANES == 12" in inspect.getsource(EA.run_suite)


# ======================================= production safety

def test_builder_still_touches_no_production_path():
    body = Path(V2B.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for banned in ("models/", "weight_model", "cnn_model.keras",
                   "CHESS_BOT_MODELS_DIR"):
        assert banned not in body, banned


def test_k6_dataset_files_are_gitignored():
    """Only the manifest is committed, as dataset_v1 and dataset_v2 are."""
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "training/artifacts/*.jsonl" in ignore
