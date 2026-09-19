"""Regression tests against the frozen Phase 0 baseline.

WHAT THIS IS
    A change-detector. It answers: "did anything about engine output move
    without us intending it?"

WHAT THIS IS NOT
    A definition of correct play. baseline/engine_results.json records what the
    engine DID on 2026-09-19 at commit ccd24c4 - bugs included. Several of those
    recorded moves are produced by defects documented in docs/PHASE_1_REPORT.md
    (the Black-side bonus sign, the inverted 1-ply lookahead, the dropped Ridge
    intercept).

    Phase 4 will DELIBERATELY change some of these outputs. When that happens the
    correct response is to re-record the baseline and document the diff - not to
    revert the engine to satisfy this file.

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


def test_legality_rate_matches_baseline(replayed, baseline_fens, baseline_engine_results):
    now = sum(1 for pid in _ids(baseline_fens) if replayed[pid]["legal"])
    then = sum(1 for pid in _ids(baseline_fens) if baseline_engine_results[pid]["is_legal"])
    assert now == then == len(baseline_fens)


# ==================================================================== selected move

def test_selected_moves_match_baseline(replayed, baseline_fens, baseline_engine_results):
    """The headline regression check. A diff here means engine behaviour changed."""
    diffs = []
    for pid in _ids(baseline_fens):
        before = baseline_engine_results[pid]["engine_move_uci"]
        after = replayed[pid]["move_uci"]
        if before != after:
            diffs.append(f"{pid}: {before} -> {after}")
    assert diffs == [], "engine move selection changed:\n  " + "\n  ".join(diffs)


# ==================================================================== top-3

def test_top3_ordering_matches_baseline(replayed, baseline_fens, baseline_engine_results):
    diffs = []
    for pid in _ids(baseline_fens):
        before = [c["uci"] for c in baseline_engine_results[pid]["top_candidates"]]
        after = [uci for uci, _ in replayed[pid]["top3"]]
        if before != after:
            diffs.append(f"{pid}: {before} -> {after}")
    assert diffs == [], "top-3 ordering changed:\n  " + "\n  ".join(diffs)


def test_top3_scores_match_baseline(replayed, baseline_fens, baseline_engine_results):
    """Scores are compared exactly: rerank_moves rounds to 3 decimals, and the
    engine was verified deterministic within a process in Phase 0 and Phase 1."""
    diffs = []
    for pid in _ids(baseline_fens):
        before = [c["score"] for c in baseline_engine_results[pid]["top_candidates"]]
        after = [score for _, score in replayed[pid]["top3"]]
        if before != pytest.approx(after, abs=1e-3):
            diffs.append(f"{pid}: {before} -> {after}")
    assert diffs == [], "top-3 scores changed:\n  " + "\n  ".join(diffs)


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
