"""Tests for training/dataset.py - label policies and the deterministic split.

No TensorFlow, no training, no dataset file required (synthetic records), so these
run in milliseconds. The heavy training path is exercised by a separate smoke run,
not by the test suite.
"""
import numpy as np
import pytest

from training import dataset as D


def _rec(fen, raw, eval_type="cp", label=None, stm="white"):
    return {
        "fen": fen, "raw_stockfish_value": raw, "eval_type": eval_type,
        "label": raw if label is None else label, "side_to_move": stm,
        "raw_value_perspective": "side_to_move", "is_checkmate": False,
    }


def _records(n=100):
    return [_rec(f"fen{i:04d}", i - 50) for i in range(n)]


# ==================================================================== validation

def test_validate_accepts_a_well_formed_dataset():
    assert D.validate_records(_records(10))["n_records"] == 10


def test_validate_rejects_duplicate_fens():
    with pytest.raises(ValueError, match="duplicate FENs"):
        D.validate_records([_rec("same", 1), _rec("same", 2)])


def test_validate_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing fields"):
        D.validate_records([{"fen": "a", "label": 1}])


def test_validate_rejects_a_non_side_to_move_raw_perspective():
    bad = _rec("a", 1)
    bad["raw_value_perspective"] = "white"
    with pytest.raises(ValueError, match="side-to-move relative"):
        D.validate_records([bad])


# ==================================================================== A0 label policy

def test_legacy_policy_is_a_plain_clip_of_the_raw_value():
    assert D.legacy_notebook_label(217) == 217
    assert D.legacy_notebook_label(-217) == -217
    assert D.legacy_notebook_label(9000) == 1500
    assert D.legacy_notebook_label(-9000) == -1500


def test_legacy_policy_ignores_eval_type_and_keeps_mate_distances():
    """This IS the pre-C6 defect, and A0 must reproduce it faithfully."""
    records = [_rec("a", 1, "mate"), _rec("b", 0, "mate"), _rec("c", -3, "mate")]
    y = D.apply_label_policy(records, D.LABEL_POLICY_LEGACY)
    assert list(y) == [1.0, 0.0, -3.0], "mate distances must survive as raw values"


def test_legacy_policy_keeps_side_to_move_relative_signs():
    """No perspective normalisation in A0 - that is A2."""
    records = [_rec("a", -572, stm="black")]
    assert D.apply_label_policy(records, D.LABEL_POLICY_LEGACY)[0] == -572.0


def test_c6prep_policy_uses_the_stored_repaired_label():
    records = [_rec("a", 1, "mate", label=1990)]
    assert D.apply_label_policy(records, D.LABEL_POLICY_C6PREP)[0] == 1990.0


def test_the_two_policies_differ_on_mate_records():
    records = [_rec("a", 0, "mate", label=-2000)]
    legacy = D.apply_label_policy(records, D.LABEL_POLICY_LEGACY)[0]
    c6 = D.apply_label_policy(records, D.LABEL_POLICY_C6PREP)[0]
    assert legacy == 0.0 and c6 == -2000.0
    assert legacy != c6


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="unknown label policy"):
        D.apply_label_policy(_records(3), "a1_mate_fix")


def test_policy_never_mutates_the_records():
    records = _records(5)
    snapshot = [dict(r) for r in records]
    D.apply_label_policy(records, D.LABEL_POLICY_LEGACY)
    assert records == snapshot


def test_a0_policy_summary_declares_it_is_the_control_and_not_historical():
    s = D.label_policy_summary(D.LABEL_POLICY_LEGACY)
    assert s["is_control_arm"] is True
    assert s["reproduces_historical_labels"] is False
    assert "side_to_move" in s["perspective"]


# ==================================================================== split

def test_split_is_80_20():
    split = D.make_split(_records(1000))
    assert len(split.train_index) == 800
    assert len(split.test_index) == 200


def test_split_partitions_without_overlap():
    split = D.make_split(_records(500))
    assert set(split.train_index) & set(split.test_index) == set()
    assert len(set(split.train_index) | set(split.test_index)) == 500


def test_split_is_identical_across_calls():
    a, b = D.make_split(_records(300)), D.make_split(_records(300))
    assert np.array_equal(a.test_index, b.test_index)
    assert np.array_equal(a.train_index, b.train_index)


def test_split_is_independent_of_the_training_seed():
    """The whole point: seed-to-seed spread must be TRAINING variance, not data
    variance, so every seed has to see exactly the same test set."""
    records = _records(400)
    for training_seed in (0, 1, 2):
        # the training seed is never passed to make_split at all
        split = D.make_split(records, split_seed=D.DEFAULT_SPLIT_SEED)
        assert np.array_equal(split.test_index,
                              D.make_split(records).test_index), training_seed


def test_different_split_seeds_give_different_splits():
    a = D.make_split(_records(400), split_seed=42)
    b = D.make_split(_records(400), split_seed=7)
    assert not np.array_equal(a.test_index, b.test_index)


def test_split_is_independent_of_record_order():
    """Sorting by FEN first means a reordered dataset yields the same partition."""
    records = _records(200)
    shuffled = list(reversed(records))

    a = D.make_split(records)
    b = D.make_split(shuffled)
    assert {records[i]["fen"] for i in a.test_index} == \
           {shuffled[i]["fen"] for i in b.test_index}


def test_split_summary_records_the_contract():
    s = D.make_split(_records(100)).summary()
    assert s["independent_of_training_seed"] is True
    assert s["split_seed"] == D.DEFAULT_SPLIT_SEED
    assert s["n_train"] == 80 and s["n_test"] == 20


# ==================================================================== arm data

def test_arm_data_aligns_labels_with_the_split():
    records = _records(100)
    arm = D.build_arm_data(records, D.LABEL_POLICY_LEGACY)
    assert len(arm.y_train) == len(arm.split.train_index) == 80
    assert len(arm.y_test) == len(arm.split.test_index) == 20
    for i, rec in zip(arm.split.test_index, arm.test_records):
        assert records[i]["fen"] == rec["fen"]


def test_arm_data_label_values_match_the_policy():
    records = [_rec(f"f{i}", i * 10) for i in range(50)]
    arm = D.build_arm_data(records, D.LABEL_POLICY_LEGACY)
    for idx in arm.split.test_index:
        assert arm.labels[idx] == float(records[idx]["raw_stockfish_value"])


def test_arm_data_stats_cover_all_three_partitions():
    stats = D.build_arm_data(_records(100), D.LABEL_POLICY_LEGACY).label_stats()
    assert set(stats) == {"all", "train", "test"}
    assert stats["all"]["n"] == 100 and stats["train"]["n"] == 80


# ==================================================================== A1 label policy
#
# A1 changes EXACTLY ONE thing from A0: how mate scores are represented.
# It must NOT apply perspective normalisation - that is A2. These tests are the
# guard against A1 silently becoming A2.

def _mate(fen, raw, stm="white", label=None):
    return _rec(fen, raw, eval_type="mate", label=label, stm=stm)


def test_a1_leaves_cp_labels_byte_identical_to_a0():
    """cp records must be untouched, otherwise A1 is not isolated."""
    records = [_rec(f"f{i}", v) for i, v in enumerate([-9000, -572, -1, 0, 7, 598, 9000])]
    a0 = D.apply_label_policy(records, D.LABEL_POLICY_LEGACY)
    a1 = D.apply_label_policy(records, D.LABEL_POLICY_CORRECTED_MATE)
    assert list(a0) == list(a1)


def test_a1_repairs_mate_labels():
    """A0 stored mate-in-1 as 1 and checkmate as 0; A1 must not."""
    records = [_mate("a", 1), _mate("b", 0), _mate("c", -3)]
    a0 = D.apply_label_policy(records, D.LABEL_POLICY_LEGACY)
    a1 = D.apply_label_policy(records, D.LABEL_POLICY_CORRECTED_MATE)
    assert list(a0) == [1.0, 0.0, -3.0]
    assert list(a1) == [1990.0, -2000.0, -1970.0]


def test_a1_mate_labels_exceed_the_cp_clip():
    records = [_mate("a", d) for d in (0, 1, 5, 49, 200)]
    for v in D.apply_label_policy(records, D.LABEL_POLICY_CORRECTED_MATE):
        assert abs(v) > D.CP_CLIP_LEGACY


def test_a1_mate_magnitude_is_monotonic_in_distance():
    records = [_mate("a", 1), _mate("b", 5), _mate("c", 20)]
    v = D.apply_label_policy(records, D.LABEL_POLICY_CORRECTED_MATE)
    assert v[0] > v[1] > v[2] > 0, "a faster mate must score better"


def test_a1_checkmate_is_negative_for_the_side_to_move():
    """mate == 0 means the side to move is checkmated - it has LOST."""
    for stm in ("white", "black"):
        v = D.apply_label_policy([_mate("a", 0, stm=stm)], D.LABEL_POLICY_CORRECTED_MATE)
        assert v[0] == -2000.0, f"stm={stm}"


# ---------------------------------------------------------------- A1 is NOT A2

def test_a1_does_not_apply_perspective_normalisation():
    """THE critical guard. A black-to-move mate-in-1 is +1990 under A1 (the side
    to move mates) but -1990 under the White-positive A2 policy."""
    rec = _mate("a", 1, stm="black", label=-1990)
    a1 = D.apply_label_policy([rec], D.LABEL_POLICY_CORRECTED_MATE)[0]
    a2 = D.apply_label_policy([rec], D.LABEL_POLICY_C6PREP)[0]
    assert a1 == 1990.0, "A1 must stay side-to-move relative"
    assert a2 == -1990.0, "A2 (c6prep) is White-positive"
    assert a1 != a2, "A1 and A2 must be distinguishable"


def test_a1_and_a2_agree_on_white_to_move_records():
    """They differ only in perspective, so White-to-move records must match."""
    rec = _mate("a", 1, stm="white", label=1990)
    assert (D.apply_label_policy([rec], D.LABEL_POLICY_CORRECTED_MATE)[0]
            == D.apply_label_policy([rec], D.LABEL_POLICY_C6PREP)[0] == 1990.0)


def test_a1_keeps_black_to_move_cp_signs_unflipped():
    """A2 would flip this to +572; A1 must leave it at -572, exactly like A0."""
    rec = _rec("a", -572, stm="black")
    assert D.apply_label_policy([rec], D.LABEL_POLICY_CORRECTED_MATE)[0] == -572.0
    assert D.apply_label_policy([rec], D.LABEL_POLICY_LEGACY)[0] == -572.0


def test_policy_summaries_distinguish_the_three_arms():
    a0 = D.label_policy_summary(D.LABEL_POLICY_LEGACY)
    a1 = D.label_policy_summary(D.LABEL_POLICY_CORRECTED_MATE)
    a2 = D.label_policy_summary(D.LABEL_POLICY_C6PREP)
    assert a1["applies_perspective_normalisation"] is False
    assert a2["applies_perspective_normalisation"] is True
    assert a1["changed_from_a0"] == "mate-label representation ONLY"
    assert len({a0["name"], a1["name"], a2["name"]}) == 3


def test_a1_policy_never_mutates_records():
    records = [_mate("a", 1), _rec("b", 50)]
    snapshot = [dict(r) for r in records]
    D.apply_label_policy(records, D.LABEL_POLICY_CORRECTED_MATE)
    assert records == snapshot


def test_a1_uses_the_same_split_as_a0():
    records = _records(400)
    a0 = D.build_arm_data(records, D.LABEL_POLICY_LEGACY)
    a1 = D.build_arm_data(records, D.LABEL_POLICY_CORRECTED_MATE)
    assert np.array_equal(a0.split.test_index, a1.split.test_index)
    assert np.array_equal(a0.split.train_index, a1.split.train_index)
