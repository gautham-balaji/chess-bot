"""Tests for evaluation/metrics.py - the harness's own mathematics.

These test ARITHMETIC, not chess strength. Every case uses hand-constructed
inputs with a known correct answer. If these fail, every number the harness
reports is suspect, so they are cheap and exhaustive by design.
"""
import chess
import chess.engine
import pytest

from evaluation import metrics as M


# ==================================================================== PovEval

def test_poveval_requires_exactly_one_of_cp_or_mate():
    with pytest.raises(ValueError):
        M.PovEval()
    with pytest.raises(ValueError):
        M.PovEval(cp=10, mate_in=2)


def test_poveval_cp_is_not_mate():
    assert M.PovEval(cp=50).is_mate is False


def test_poveval_mate_flags():
    assert M.PovEval(mate_in=3).is_mate is True
    assert M.PovEval(mate_in=3).mate_is_winning is True
    assert M.PovEval(mate_in=-3).mate_is_winning is False
    assert M.PovEval(mate_in=0).mate_is_winning is True, "mate on the board, POV delivered"


def test_poveval_mate_is_winning_rejects_cp_scores():
    with pytest.raises(ValueError):
        _ = M.PovEval(cp=10).mate_is_winning


# ==================================================================== POV conversion
#
# The single most important thing to get right. A PovScore built as
# Cp(+120) relative to WHITE must read +120 for White and -120 for Black.

def test_pov_conversion_white_positive():
    pov = chess.engine.PovScore(chess.engine.Cp(120), chess.WHITE)
    assert M.score_from_pov(pov, chess.WHITE) == M.PovEval(cp=120)


def test_pov_conversion_flips_sign_for_black():
    pov = chess.engine.PovScore(chess.engine.Cp(120), chess.WHITE)
    assert M.score_from_pov(pov, chess.BLACK) == M.PovEval(cp=-120)


def test_pov_conversion_black_relative_score():
    """A score stored relative to BLACK reads positive for Black."""
    pov = chess.engine.PovScore(chess.engine.Cp(80), chess.BLACK)
    assert M.score_from_pov(pov, chess.BLACK) == M.PovEval(cp=80)
    assert M.score_from_pov(pov, chess.WHITE) == M.PovEval(cp=-80)


def test_pov_conversion_is_self_inverse():
    """Converting to both POVs must give exactly opposite centipawns."""
    for raw in (-450, -1, 0, 1, 37, 900):
        pov = chess.engine.PovScore(chess.engine.Cp(raw), chess.WHITE)
        white = M.score_from_pov(pov, chess.WHITE).cp
        black = M.score_from_pov(pov, chess.BLACK).cp
        assert white == -black


def test_pov_conversion_mate_white():
    pov = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
    assert M.score_from_pov(pov, chess.WHITE) == M.PovEval(mate_in=3)


def test_pov_conversion_mate_flips_for_black():
    """White mating in 3 means Black is mated in 3."""
    pov = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
    assert M.score_from_pov(pov, chess.BLACK) == M.PovEval(mate_in=-3)


def test_pov_conversion_never_returns_centipawns_for_a_mate():
    pov = chess.engine.PovScore(chess.engine.Mate(-2), chess.WHITE)
    result = M.score_from_pov(pov, chess.WHITE)
    assert result.cp is None and result.mate_in == -2


# ==================================================================== regret

def test_regret_is_zero_for_identical_evaluations():
    regret, status = M.compute_regret(M.PovEval(cp=40), M.PovEval(cp=40))
    assert regret == 0
    assert status == M.MATE_STATUS_NONE


def test_regret_is_positive_when_engine_move_is_worse():
    """Reference leaves +50, engine leaves -30 => 80cp given up."""
    regret, _ = M.compute_regret(M.PovEval(cp=50), M.PovEval(cp=-30))
    assert regret == 80


def test_regret_is_negative_when_engine_move_scores_better():
    """Retained, not clamped - clamping would bias the mean upward."""
    regret, _ = M.compute_regret(M.PovEval(cp=10), M.PovEval(cp=45))
    assert regret == -35


@pytest.mark.parametrize(
    "ref_cp,eng_cp,expected",
    [(0, 0, 0), (100, 0, 100), (0, -100, 100), (-200, -500, 300), (-50, 20, -70)],
)
def test_regret_arithmetic_table(ref_cp, eng_cp, expected):
    regret, _ = M.compute_regret(M.PovEval(cp=ref_cp), M.PovEval(cp=eng_cp))
    assert regret == expected


def test_regret_is_none_whenever_a_mate_is_involved():
    for ref, eng in [
        (M.PovEval(mate_in=2), M.PovEval(cp=100)),
        (M.PovEval(cp=100), M.PovEval(mate_in=-2)),
        (M.PovEval(mate_in=2), M.PovEval(mate_in=4)),
    ]:
        regret, status = M.compute_regret(ref, eng)
        assert regret is None
        assert status != M.MATE_STATUS_NONE


# ==================================================================== mate taxonomy

def test_mate_status_none_for_two_cp_scores():
    assert M.classify_mate(M.PovEval(cp=10), M.PovEval(cp=-5)) == M.MATE_STATUS_NONE


def test_mate_status_missed_forced_mate():
    """Reference forces mate; the engine's move only reaches a cp score."""
    assert M.classify_mate(M.PovEval(mate_in=3), M.PovEval(cp=250)) == \
        M.MATE_STATUS_MISSED_MATE


def test_mate_status_engine_found_mate_reference_did_not():
    assert M.classify_mate(M.PovEval(cp=250), M.PovEval(mate_in=2)) == \
        M.MATE_STATUS_ENGINE_FOUND_MATE


def test_mate_status_engine_move_allows_mate():
    """Reference wins by force, engine's move gets mated instead."""
    assert M.classify_mate(M.PovEval(mate_in=3), M.PovEval(mate_in=-1)) == \
        M.MATE_STATUS_ENGINE_ALLOWS_MATE


def test_mate_status_engine_move_allows_mate_from_cp_reference():
    assert M.classify_mate(M.PovEval(cp=0), M.PovEval(mate_in=-4)) == \
        M.MATE_STATUS_ENGINE_ALLOWS_MATE


def test_mate_status_both_win():
    assert M.classify_mate(M.PovEval(mate_in=1), M.PovEval(mate_in=5)) == \
        M.MATE_STATUS_BOTH_WIN


def test_mate_status_both_lose():
    """Position is lost by force whatever is played - not the engine's fault."""
    assert M.classify_mate(M.PovEval(mate_in=-2), M.PovEval(mate_in=-1)) == \
        M.MATE_STATUS_BOTH_LOSE


def test_mate_on_the_board_counts_as_a_win_for_the_mover():
    assert M.classify_mate(M.PovEval(mate_in=0), M.PovEval(mate_in=0)) == \
        M.MATE_STATUS_BOTH_WIN


# ==================================================================== agreement

def test_top1_agreement_matching_and_not():
    assert M.top1_agreement("e2e4", "e2e4") is True
    assert M.top1_agreement("e2e4", "d2d4") is False


def test_top1_agreement_handles_missing_moves():
    assert M.top1_agreement(None, "e2e4") is False
    assert M.top1_agreement("e2e4", None) is False
    assert M.top1_agreement(None, None) is False


def test_top3_containment_hit_and_miss():
    assert M.top_n_containment("d2d4", ["e2e4", "d2d4", "g1f3"]) is True
    assert M.top_n_containment("a2a3", ["e2e4", "d2d4", "g1f3"]) is False


def test_top3_containment_with_empty_reference_list():
    assert M.top_n_containment("e2e4", []) is False


def test_top3_containment_handles_missing_engine_move():
    assert M.top_n_containment(None, ["e2e4"]) is False


# ==================================================================== chance baseline

def test_random_baseline_top1_is_mean_of_reciprocals():
    # 1/2 and 1/4 -> mean 0.375
    assert M.random_move_baseline([2, 4], 1) == pytest.approx(0.375)


def test_random_baseline_top3_caps_at_the_legal_move_count():
    """With only 2 legal moves, a top-3 list must contain the move with p=1."""
    assert M.random_move_baseline([2], 3) == pytest.approx(1.0)


def test_random_baseline_top3_for_a_wide_position():
    assert M.random_move_baseline([30], 3) == pytest.approx(0.1)


def test_random_baseline_ignores_invalid_counts():
    assert M.random_move_baseline([0, 4], 1) == pytest.approx(0.25)


def test_random_baseline_returns_none_for_empty_input():
    assert M.random_move_baseline([]) is None


# ==================================================================== aggregation

def test_percentile_nearest_rank():
    """Nearest-rank: index = round(q * (n-1)).

    Note round() is banker's rounding, so an exact .5 index ties to even:
    for n=10, q=0.5 gives round(4.5) == 4, i.e. value 5. This matches the
    percentile used in the Phase 0 baseline, so the two phases stay comparable.
    """
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert M.percentile(values, 0.0) == 1
    assert M.percentile(values, 1.0) == 10
    assert M.percentile(values, 0.5) == 5
    assert M.percentile(values, 0.95) == 10


def test_percentile_p95_on_52_points_matches_phase0_indexing():
    """n=52 is the evaluation suite size; index = round(0.95*51) = 48."""
    values = list(range(52))
    assert M.percentile(values, 0.95) == 48


def test_percentile_of_empty_is_none():
    assert M.percentile([], 0.95) is None


def test_summarize_known_values():
    s = M.summarize([10, 20, 30, 40, 50])
    assert s["n"] == 5
    assert s["min"] == 10
    assert s["max"] == 50
    assert s["median"] == 30
    assert s["mean"] == 30


def test_summarize_handles_negatives():
    s = M.summarize([-50, 0, 50])
    assert s["min"] == -50 and s["mean"] == 0


def test_summarize_empty_returns_nulls_not_an_exception():
    s = M.summarize([])
    assert s == {"n": 0, "min": None, "median": None, "mean": None,
                 "p95": None, "max": None}


def test_summarize_is_order_independent():
    assert M.summarize([5, 1, 9, 3]) == M.summarize([9, 3, 1, 5])


def test_blunder_rate_threshold_is_strict():
    """Exactly at the threshold is not a blunder; one above is."""
    assert M.blunder_rate([300], threshold=300)["blunders"] == 0
    assert M.blunder_rate([301], threshold=300)["blunders"] == 1


def test_blunder_rate_counts_and_denominator():
    result = M.blunder_rate([0, 100, 400, 900], threshold=300)
    assert result["blunders"] == 2
    assert result["denominator"] == 4
    assert result["rate"] == pytest.approx(0.5)


def test_blunder_rate_empty_input():
    result = M.blunder_rate([])
    assert result["blunders"] == 0 and result["rate"] is None


def test_rate_reports_numerator_and_denominator():
    r = M.rate(10, 52)
    assert r["numerator"] == 10 and r["denominator"] == 52
    assert r["rate"] == pytest.approx(0.1923, abs=1e-4)
    assert r["percent"] == pytest.approx(19.23, abs=1e-2)


def test_rate_matches_the_phase0_headline_number():
    """Phase 0 reported 10/52 = 19.23% top-1 agreement."""
    assert M.rate(10, 52)["percent"] == pytest.approx(19.23, abs=0.01)


def test_rate_handles_zero_denominator():
    r = M.rate(0, 0)
    assert r["rate"] is None and r["percent"] is None


# ==================================================================== spearman

def test_spearman_identical_rankings_is_one():
    assert M.spearman_rank_correlation([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)


def test_spearman_reversed_rankings_is_minus_one():
    assert M.spearman_rank_correlation([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1.0)


def test_spearman_requires_at_least_three_points():
    assert M.spearman_rank_correlation([0, 1], [1, 0]) is None


def test_spearman_returns_none_for_constant_input():
    """rho is undefined when one ranking has no variance."""
    assert M.spearman_rank_correlation([1, 1, 1], [0, 1, 2]) is None


def test_spearman_returns_none_on_length_mismatch():
    assert M.spearman_rank_correlation([0, 1, 2], [0, 1]) is None


# ==================================================================== end-to-end math
#
# A worked example combining POV conversion, regret and classification, with the
# answer computed by hand.

def test_worked_example_black_to_move():
    """Black to move. Both resulting positions are stored White-relative.

    After Stockfish's move: -150 White  => +150 Black
    After the engine's move: +40 White  =>  -40 Black
    Black therefore gave up 150 - (-40) = 190 centipawns.
    """
    after_sf = M.score_from_pov(
        chess.engine.PovScore(chess.engine.Cp(-150), chess.WHITE), chess.BLACK)
    after_engine = M.score_from_pov(
        chess.engine.PovScore(chess.engine.Cp(40), chess.WHITE), chess.BLACK)

    assert after_sf == M.PovEval(cp=150)
    assert after_engine == M.PovEval(cp=-40)

    regret, status = M.compute_regret(after_sf, after_engine)
    assert regret == 190
    assert status == M.MATE_STATUS_NONE


def test_worked_example_white_to_move_missed_mate():
    """White to move. Stockfish's move mates in 2; the engine's leaves +90."""
    after_sf = M.score_from_pov(
        chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE), chess.WHITE)
    after_engine = M.score_from_pov(
        chess.engine.PovScore(chess.engine.Cp(90), chess.WHITE), chess.WHITE)

    regret, status = M.compute_regret(after_sf, after_engine)
    assert regret is None, "a mate score must not be forced onto the centipawn axis"
    assert status == M.MATE_STATUS_MISSED_MATE
