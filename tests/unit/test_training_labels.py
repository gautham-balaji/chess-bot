"""Tests for training/labels.py - the C6-Prep label policy.

Pure arithmetic against hand-computed answers. No Stockfish, no games.csv, no
machine-specific paths, so these run in milliseconds anywhere.

The reference semantics were established empirically against Stockfish 17.1 via
the `stockfish` 4.0.8 wrapper (see docs/C6_TRAINING_PIPELINE_AUDIT.md):

    White to move, White mates in 1        -> {'type':'mate','value':  1}
    Black to move, Black mates in 1        -> {'type':'mate','value':  1}   (stm-relative)
    already checkmated (either colour)     -> {'type':'mate','value':  0}
    Black to move, White up a queen        -> {'type':'cp',  'value': -572} (stm-relative)
"""
import pytest

from training import labels as L


# ==================================================================== perspective

def test_white_to_move_value_is_unchanged():
    assert L.to_white_positive(120, side_to_move_is_white=True) == 120


def test_black_to_move_value_is_negated():
    """Stockfish is side-to-move relative, so +120 for Black is -120 for White."""
    assert L.to_white_positive(120, side_to_move_is_white=False) == -120


def test_perspective_is_self_inverse():
    for raw in (-1500, -572, -1, 0, 7, 598, 1500):
        assert L.to_white_positive(raw, True) == -L.to_white_positive(raw, False)


def test_perspective_reproduces_the_observed_wrapper_conversion():
    """Black to move, White up a queen: stm-relative -572 must read +572 for White.

    This is the exact pair observed from the wrapper with turn_perspective
    True/False, so our explicit conversion must agree with it.
    """
    assert L.to_white_positive(-572, side_to_move_is_white=False) == 572


def test_zero_is_unchanged_either_way():
    assert L.to_white_positive(0, True) == 0
    assert L.to_white_positive(0, False) == 0


# ==================================================================== mate magnitude

def test_mate_magnitude_is_maximal_at_distance_zero():
    assert L.mate_magnitude(0) == L.MATE_SCORE_BASE == 2000


def test_mate_magnitude_decreases_with_distance():
    assert L.mate_magnitude(1) == 1990
    assert L.mate_magnitude(5) == 1950
    assert L.mate_magnitude(1) > L.mate_magnitude(2) > L.mate_magnitude(10)


def test_mate_magnitude_is_capped():
    floor = L.MATE_SCORE_BASE - L.MATE_SCORE_STEP * L.MATE_MAX_DISTANCE
    assert L.mate_magnitude(L.MATE_MAX_DISTANCE) == floor
    assert L.mate_magnitude(999) == floor


def test_every_mate_outranks_every_clipped_cp_label():
    """The whole point of the mate scale: a mate must never look like a cp score."""
    floor = L.mate_magnitude(999)
    assert floor > L.CP_CLIP, f"mate floor {floor} must exceed cp clip {L.CP_CLIP}"


def test_mate_magnitude_ignores_the_sign_of_the_distance():
    assert L.mate_magnitude(-3) == L.mate_magnitude(3)


# ==================================================================== mate sign

def test_positive_mate_means_the_side_to_move_wins():
    assert L.mate_to_white_positive(1, side_to_move_is_white=True) == 1990
    assert L.mate_to_white_positive(1, side_to_move_is_white=False) == -1990


def test_negative_mate_means_the_side_to_move_loses():
    assert L.mate_to_white_positive(-1, side_to_move_is_white=True) == -1990
    assert L.mate_to_white_positive(-1, side_to_move_is_white=False) == 1990


def test_mate_zero_means_the_side_to_move_is_checkmated_and_has_lost():
    """Zero carries no sign, so the winner comes from the side to move."""
    assert L.mate_to_white_positive(0, side_to_move_is_white=True) == -2000
    assert L.mate_to_white_positive(0, side_to_move_is_white=False) == 2000


def test_mate_mapping_is_colour_symmetric():
    for d in (0, 1, 3, 12, 60):
        assert L.mate_to_white_positive(d, True) == -L.mate_to_white_positive(d, False)


def test_faster_mates_score_better_for_the_winner():
    fast = L.mate_to_white_positive(1, side_to_move_is_white=True)
    slow = L.mate_to_white_positive(9, side_to_move_is_white=True)
    assert fast > slow > 0


# ==================================================================== mate-zero check

def test_mate_zero_is_consistent_when_the_board_really_is_checkmate():
    assert L.classify_mate_zero(is_checkmate=True) is True


def test_mate_zero_is_flagged_when_the_board_is_not_checkmate():
    assert L.classify_mate_zero(is_checkmate=False) is False


# ==================================================================== clipping

def test_clip_leaves_in_range_values_alone():
    assert L.clip_cp(0) == 0
    assert L.clip_cp(1499) == 1499
    assert L.clip_cp(-1499) == -1499


def test_clip_bounds_are_inclusive():
    assert L.clip_cp(1500) == 1500
    assert L.clip_cp(-1500) == -1500


def test_clip_clamps_beyond_the_bounds():
    assert L.clip_cp(5000) == 1500
    assert L.clip_cp(-5000) == -1500


def test_clip_bound_is_configurable():
    assert L.clip_cp(900, bound=500) == 500
    assert L.clip_cp(-900, bound=500) == -500


# ==================================================================== make_label

def test_cp_label_for_white_to_move():
    lab = L.make_label("cp", 598, side_to_move_is_white=True)
    assert lab.label == 598
    assert lab.eval_type == "cp"
    assert lab.raw_value == 598
    assert lab.side_to_move == "white"
    assert lab.was_clipped is False
    assert lab.mate_zero_consistent is None


def test_cp_label_for_black_to_move_is_negated():
    """The defect this pipeline fixes: the notebook stored this as -572."""
    lab = L.make_label("cp", -572, side_to_move_is_white=False)
    assert lab.label == 572, "White is up a queen, so the label must be positive"
    assert lab.raw_value == -572, "the raw stm-relative value stays auditable"
    assert lab.side_to_move == "black"


def test_cp_label_records_clipping():
    lab = L.make_label("cp", 9000, side_to_move_is_white=True)
    assert lab.label == 1500
    assert lab.was_clipped is True


def test_mate_label_is_never_the_raw_distance():
    """The notebook stored mate-in-1 as the label 1. It must not any more."""
    lab = L.make_label("mate", 1, side_to_move_is_white=True)
    assert lab.label == 1990
    assert lab.label != 1
    assert lab.raw_value == 1


def test_checkmate_label_is_never_zero():
    """The notebook stored an already-checkmated position as 0 - "equal"."""
    lab = L.make_label("mate", 0, side_to_move_is_white=True, is_checkmate=True)
    assert lab.label == -2000, "White to move and checkmated: worst possible for White"
    assert lab.label != 0
    assert lab.mate_zero_consistent is True


def test_black_checkmated_is_maximally_good_for_white():
    lab = L.make_label("mate", 0, side_to_move_is_white=False, is_checkmate=True)
    assert lab.label == 2000


def test_mate_zero_on_a_non_checkmate_board_is_flagged_not_guessed():
    lab = L.make_label("mate", 0, side_to_move_is_white=True, is_checkmate=False)
    assert lab.mate_zero_consistent is False
    assert lab.label == -2000, "still labelled by policy, but the anomaly is recorded"


def test_mate_labels_are_never_clipped():
    """Clipping mates would collapse the mate scale back into the cp range."""
    lab = L.make_label("mate", 0, side_to_move_is_white=False, is_checkmate=True)
    assert lab.label == 2000 > L.CP_CLIP
    assert lab.was_clipped is False


def test_unknown_eval_type_raises():
    with pytest.raises(ValueError):
        L.make_label("wdl", 10, side_to_move_is_white=True)


def test_label_is_serialisable():
    d = L.make_label("cp", 100, side_to_move_is_white=True).as_dict()
    assert set(d) == {"label", "eval_type", "raw_value", "side_to_move",
                      "was_clipped", "mate_zero_consistent"}


# ==================================================================== policy summary

def test_policy_summary_declares_white_perspective():
    summary = L.policy_summary()
    assert summary["label_perspective"] == "white"
    assert summary["clip_bounds"] == [-1500, 1500]
    assert "cp labels only" in summary["clip_applies_to"]


def test_policy_summary_documents_the_mate_zero_interpretation():
    assert "side to move" in L.policy_summary()["mate_zero_policy"].lower()


def test_policy_ordering_is_documented():
    assert "mate mapping" in L.policy_summary()["ordering"]
