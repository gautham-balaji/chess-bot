"""Regression tests against the ACCEPTED engine baseline.

WHAT THIS IS
    A change-detector. It answers: "did anything about engine output move
    without us intending it?"

WHAT THIS IS NOT
    A definition of correct play. regression/engine_results_post_c2.json records
    what the engine DOES at the accepted post-C2 state - remaining known defects
    included. It does not say those moves are good.

TWO BASELINES, ON PURPOSE
    baseline/engine_results.json        ORIGINAL Phase 0 capture. Frozen historical
                                        evidence, never re-recorded.
    regression/engine_results_post_c2.json
                                        Current accepted expectations, re-recorded
                                        after C1 and C2 were accepted.

    C1 changed top-3 SCORE values on 15 Black positions; C2 changed them on all 52.
    Neither changed a selected move, a top-3 ordering, legality, determinism or
    board integrity - so the score expectations were re-recorded while every
    behavioural assertion was kept. Both fixes were correctness fixes with NO
    measured change in playing quality.

    `test_selected_moves_and_ordering_still_match_original_phase0` keeps the link
    to the original capture, so the two can always be distinguished.

    When a future accepted change alters output, re-record
    regression/engine_results_post_c2.json deliberately and document the diff -
    never revert the engine to satisfy this file, and never edit the JSON by hand.

Marked `slow`: 52 positions at ~600ms each. Deselect with -m "not slow".
"""
import chess
import pytest

pytestmark = pytest.mark.slow


def _ids(positions):
    return [p["id"] for p in positions]


@pytest.fixture(scope="module")
def replayed(request):
    """Run the engine once over all 52 baseline positions and cache the result.

    Module-scoped so the ~35s replay is paid once, not once per assertion.
    """
    engine_mod = request.getfixturevalue("engine_mod")
    positions = request.getfixturevalue("baseline_fens")

    out = {}
    for entry in positions:
        board = chess.Board(entry["fen"])
        move, _, top3 = engine_mod.engine_move(board)
        out[entry["id"]] = {
            "move_uci": move.uci() if move else None,
            "top3": [(c["move"].uci(), c["score"]) for c in top3],
            "legal": bool(move is not None and move in chess.Board(entry["fen"]).legal_moves),
            "fen_preserved": board.fen() == entry["fen"],
        }
    return out


# ==================================================================== legality

def test_every_baseline_position_still_yields_a_legal_move(replayed, baseline_fens):
    illegal = [pid for pid in _ids(baseline_fens) if not replayed[pid]["legal"]]
    assert illegal == [], f"illegal moves returned for: {illegal}"


def test_legality_rate_matches_baseline(replayed, baseline_fens, current_engine_results):
    now = sum(1 for pid in _ids(baseline_fens) if replayed[pid]["legal"])
    then = sum(1 for pid in _ids(baseline_fens) if current_engine_results[pid]["is_legal"])
    assert now == then == len(baseline_fens)


# ==================================================================== selected move

def test_selected_moves_match_baseline(replayed, baseline_fens, current_engine_results):
    """The headline regression check. A diff here means engine behaviour changed."""
    diffs = []
    for pid in _ids(baseline_fens):
        before = current_engine_results[pid]["engine_move_uci"]
        after = replayed[pid]["move_uci"]
        if before != after:
            diffs.append(f"{pid}: {before} -> {after}")
    assert diffs == [], "engine move selection changed:\n  " + "\n  ".join(diffs)


# ==================================================================== top-3

def test_top3_ordering_matches_baseline(replayed, baseline_fens, current_engine_results):
    diffs = []
    for pid in _ids(baseline_fens):
        before = [c["uci"] for c in current_engine_results[pid]["top_candidates"]]
        after = [uci for uci, _ in replayed[pid]["top3"]]
        if before != after:
            diffs.append(f"{pid}: {before} -> {after}")
    assert diffs == [], "top-3 ordering changed:\n  " + "\n  ".join(diffs)


def test_top3_scores_match_baseline(replayed, baseline_fens, current_engine_results):
    """Scores are compared exactly: rerank_moves rounds to 3 decimals, and the
    engine was verified deterministic within a process in Phase 0 and Phase 1."""
    diffs = []
    for pid in _ids(baseline_fens):
        before = [c["score"] for c in current_engine_results[pid]["top_candidates"]]
        after = [score for _, score in replayed[pid]["top3"]]
        if before != pytest.approx(after, abs=1e-3):
            diffs.append(f"{pid}: {before} -> {after}")
    assert diffs == [], "top-3 scores changed:\n  " + "\n  ".join(diffs)


# ============================================================ historical continuity

def test_selected_moves_and_ordering_still_match_original_phase0(
    replayed, baseline_fens, baseline_engine_results
):
    """The C1 and C2 fixes changed SCORES but must not have changed CHOICES.

    Compares against the ORIGINAL frozen Phase 0 capture, not the re-recorded
    expectations, so the claim "neither fix altered a selected move or a top-3
    ordering" stays continuously verified rather than resting on a report.

    If this ever fails, engine behaviour has diverged from the pre-C1 state and
    that divergence needs explaining before any baseline is re-recorded.
    """
    move_diffs, order_diffs = [], []
    for pid in _ids(baseline_fens):
        original = baseline_engine_results[pid]
        now = replayed[pid]
        if original["engine_move_uci"] != now["move_uci"]:
            move_diffs.append(f"{pid}: {original['engine_move_uci']} -> {now['move_uci']}")
        original_order = [c["uci"] for c in original["top_candidates"]]
        now_order = [uci for uci, _ in now["top3"]]
        if original_order != now_order:
            order_diffs.append(f"{pid}: {original_order} -> {now_order}")

    joined_moves = "; ".join(move_diffs)
    joined_orders = "; ".join(order_diffs)
    assert move_diffs == [], (
        "selected moves diverged from the ORIGINAL Phase 0 capture: " + joined_moves)
    assert order_diffs == [], (
        "top-3 ordering diverged from the ORIGINAL Phase 0 capture: " + joined_orders)


def test_scores_are_expected_to_differ_from_original_phase0(
    replayed, baseline_fens, baseline_engine_results
):
    """Characterisation: C1 and C2 DID change score values on all 52 positions.

    Pins the reason the baseline was re-recorded. If this ever stops being true,
    the re-recording rationale no longer holds and should be revisited.
    """
    differing = 0
    for pid in _ids(baseline_fens):
        before = [c["score"] for c in baseline_engine_results[pid]["top_candidates"]]
        after = [score for _, score in replayed[pid]["top3"]]
        if before != pytest.approx(after, abs=1e-3):
            differing += 1
    assert differing == len(baseline_fens), (
        f"expected all {len(baseline_fens)} positions to differ in score from the "
        f"original Phase 0 capture, got {differing}")


# ==================================================================== invariants

def test_no_baseline_position_mutates_its_board(replayed, baseline_fens):
    mutated = [pid for pid in _ids(baseline_fens) if not replayed[pid]["fen_preserved"]]
    assert mutated == [], f"input board mutated for: {mutated}"


def test_replay_is_deterministic_for_a_sample(engine_mod, sample_fens):
    """Re-running a subset must reproduce itself exactly within this process.

    Scoped to a sample to keep runtime sane. This does NOT claim cross-machine or
    cross-version determinism - TensorFlow's oneDNN notice makes that unverified.
    """
    for entry in sample_fens:
        first, _, first_top = engine_mod.engine_move(chess.Board(entry["fen"]))
        second, _, second_top = engine_mod.engine_move(chess.Board(entry["fen"]))
        assert first == second, entry["id"]
        assert [c["score"] for c in first_top] == [c["score"] for c in second_top]


def test_baseline_suite_is_intact(baseline_fens):
    """Guard the fixture itself: if fens.json is edited, every comparison above
    silently changes meaning."""
    assert len(baseline_fens) == 52
    sides = [p["side_to_move"] for p in baseline_fens]
    assert sides.count("white") == 30
    assert sides.count("black") == 22
    for entry in baseline_fens:
        board = chess.Board(entry["fen"])
        assert board.is_valid()
        assert not board.is_game_over()
