"""Integration tests for engine.rerank_moves and engine.engine_move.

These exercise the CNN, the Ridge fusion and the heuristic layer together.
Each engine_move() call costs ~600ms, so the position sets are deliberately small.
"""
import chess
import pytest


def bonus_sum(engine_mod, board, move):
    """Total heuristic bonus rerank_moves adds to a candidate, using the engine's
    own helpers rather than re-deriving the constants."""
    return (
        engine_mod.development_bonus(board, move)
        + engine_mod.pawn_push_penalty(board, move)
        + engine_mod.opening_center_bonus(board, move)
        + engine_mod.tactical_move_bonus(board, move)
    )


# ==================================================================== legality

def test_engine_move_returns_a_legal_move(engine_mod, sample_fens):
    """The single most important property: never return an illegal move."""
    for entry in sample_fens:
        board = chess.Board(entry["fen"])
        move, _, _ = engine_mod.engine_move(board)
        fresh = chess.Board(entry["fen"])  # re-check on a pristine board
        assert move in fresh.legal_moves, f"{entry['id']} returned illegal {move}"


def test_engine_move_returns_a_parseable_uci_move(engine_mod, white_to_move_board):
    move, _, _ = engine_mod.engine_move(white_to_move_board)
    assert chess.Move.from_uci(move.uci()) == move


def test_engine_move_result_structure(engine_mod, start_board):
    move, explanation, top3 = engine_mod.engine_move(start_board)
    assert isinstance(move, chess.Move)
    assert isinstance(explanation, list) and explanation
    assert isinstance(top3, list) and 0 < len(top3) <= 3


def test_top_candidate_fields(engine_mod, start_board):
    _, _, top3 = engine_mod.engine_move(start_board)
    expected = {"move", "score", "cnn_cp", "material", "space", "center", "mobility"}
    for entry in top3:
        assert set(entry) == expected


# ==================================================================== ranking contract

def test_rerank_returns_one_entry_per_legal_move(engine_mod, white_to_move_board):
    ranked = engine_mod.rerank_moves(white_to_move_board)
    assert len(ranked) == white_to_move_board.legal_moves.count()


def test_rerank_candidates_are_unique(engine_mod, white_to_move_board):
    ranked = engine_mod.rerank_moves(white_to_move_board)
    ucis = [e["move"].uci() for e in ranked]
    assert len(ucis) == len(set(ucis))


def test_white_ranking_is_descending_by_score(engine_mod, white_to_move_board):
    """rerank_moves sorts reverse=True when White is to move."""
    scores = [e["score"] for e in engine_mod.rerank_moves(white_to_move_board)]
    assert scores == sorted(scores, reverse=True)


def test_black_ranking_is_ascending_by_score(engine_mod, black_to_move_board):
    """...and reverse=False for Black, so lower scores rank first."""
    scores = [e["score"] for e in engine_mod.rerank_moves(black_to_move_board)]
    assert scores == sorted(scores)


def test_engine_move_returns_the_top_ranked_candidate(engine_mod, white_to_move_board):
    ranked = engine_mod.rerank_moves(white_to_move_board)
    move, _, top3 = engine_mod.engine_move(white_to_move_board)
    assert move == ranked[0]["move"]
    assert [e["move"] for e in top3] == [e["move"] for e in ranked[:3]]


# ==================================================================== determinism

def test_engine_move_is_deterministic(engine_mod, white_to_move_board):
    first_move, _, first_top = engine_mod.engine_move(white_to_move_board)
    second_move, _, second_top = engine_mod.engine_move(white_to_move_board)
    assert first_move == second_move
    assert [e["score"] for e in first_top] == [e["score"] for e in second_top]


def test_rerank_scores_are_stable_across_calls(engine_mod, black_to_move_board):
    first = [(e["move"].uci(), e["score"]) for e in engine_mod.rerank_moves(black_to_move_board)]
    second = [(e["move"].uci(), e["score"]) for e in engine_mod.rerank_moves(black_to_move_board)]
    assert first == second


# ==================================================================== board safety

def test_engine_move_does_not_mutate_the_input_board(engine_mod, sample_fens):
    for entry in sample_fens:
        board = chess.Board(entry["fen"])
        engine_mod.engine_move(board)
        assert board.fen() == entry["fen"], f"{entry['id']} board was mutated"
        assert len(board.move_stack) == 0


# ==================================================================== edge cases

def test_checkmate_returns_no_move_without_raising(engine_mod, checkmate_board):
    move, explanation, top3 = engine_mod.engine_move(checkmate_board)
    assert move is None
    assert top3 == []
    assert explanation == ["No legal moves available"]


def test_stalemate_returns_no_move_without_raising(engine_mod, stalemate_board):
    move, _, top3 = engine_mod.engine_move(stalemate_board)
    assert move is None
    assert top3 == []


def test_rerank_returns_empty_list_for_terminal_positions(engine_mod, checkmate_board):
    assert engine_mod.rerank_moves(checkmate_board) == []


def test_handles_position_with_few_legal_moves(engine_mod, endgame_board):
    assert endgame_board.legal_moves.count() < 10
    move, _, _ = engine_mod.engine_move(endgame_board)
    assert move in endgame_board.legal_moves


def test_handles_position_with_many_legal_moves(engine_mod):
    """A wide-open middlegame - exercises the O(branching^2) lookahead path."""
    board = chess.Board("r2q1rk1/pp2ppbp/2n3p1/2pp4/3PnB2/2P1PN2/PP1N1PPP/R2QKB1R w KQ - 0 10")
    assert board.legal_moves.count() > 30
    move, _, _ = engine_mod.engine_move(board)
    assert move in board.legal_moves


def test_handles_a_position_where_the_side_to_move_is_in_check(engine_mod, in_check_board):
    move, _, _ = engine_mod.engine_move(in_check_board)
    assert move in in_check_board.legal_moves


@pytest.mark.parametrize("fen_key", ["white_to_move_board", "black_to_move_board"])
def test_both_sides_are_handled(engine_mod, request, fen_key):
    board = request.getfixturevalue(fen_key)
    move, _, top3 = engine_mod.engine_move(board)
    assert move in board.legal_moves
    assert len(top3) > 0


# ==================================================================== C1 (FIXED)
#
# The heuristic bonuses used to be added with a fixed positive sign while the
# final sort direction flips by side (engine.py). For White, where higher scores
# rank first, a positive bonus correctly improves a move. For Black, where LOWER
# scores rank first, the same positive bonus pushed the move DOWN Black's own
# preference list - and app.py only ever lets the engine play Black.
#
# Phase 4A fixed this by applying the bonuses with a side-dependent sign. The
# contract below is asserted identically for both colours and must now pass for
# both. It previously passed for White and failed for Black.
#
# The applied bonus is measured directly, by re-running rerank_moves with the
# four bonus helpers stubbed to zero and differencing the scores. Everything else
# (the weighted term, the 1-ply lookahead adjustment) is deterministic and
# identical between the two calls, so the difference IS the signed bonus
# contribution. This avoids re-deriving any engine arithmetic in the test.

BONUS_HELPERS = (
    "development_bonus",
    "pawn_push_penalty",
    "opening_center_bonus",
    "tactical_move_bonus",
)


def _ranking_direction_is_lower_is_better(ranked):
    """Derive the sort direction from the engine's own output rather than
    hardcoding it, so this test does not restate the implementation."""
    return ranked[0]["score"] < ranked[-1]["score"]


def _scores_by_uci(ranked):
    return {e["move"].uci(): e["score"] for e in ranked}


@pytest.mark.parametrize(
    "board_fixture", ["white_to_move_board", "black_to_move_board"]
)
def test_positive_bonus_must_not_worsen_a_move_for_the_side_to_move(
    engine_mod, request, monkeypatch, board_fixture
):
    board = request.getfixturevalue(board_fixture)

    ranked = engine_mod.rerank_moves(board.copy())
    lower_is_better = _ranking_direction_is_lower_is_better(ranked)
    with_bonus = _scores_by_uci(ranked)

    # Raw bonuses MUST be captured before the helpers are stubbed out.
    raw = {e["move"].uci(): bonus_sum(engine_mod, board, e["move"]) for e in ranked}

    for name in BONUS_HELPERS:
        monkeypatch.setattr(engine_mod, name, lambda b, m: 0.0)
    without_bonus = _scores_by_uci(engine_mod.rerank_moves(board.copy()))

    checked = 0
    for uci, raw_bonus in raw.items():
        if raw_bonus <= 0:
            continue
        checked += 1
        applied = with_bonus[uci] - without_bonus[uci]

        if lower_is_better:
            assert applied <= 0, (
                f"{uci}: raw bonus {raw_bonus:+.3f} was applied as {applied:+.3f}, "
                f"raising the score, but LOWER scores rank better for this side - "
                f"the bonus made the move worse"
            )
        else:
            assert applied >= 0, (
                f"{uci}: raw bonus {raw_bonus:+.3f} was applied as {applied:+.3f}, "
                f"but HIGHER scores rank better for this side"
            )

    assert checked > 0, "no bonused move found - test position is unsuitable"


@pytest.mark.parametrize(
    "board_fixture,expect_negated",
    [("white_to_move_board", False), ("black_to_move_board", True)],
)
def test_bonus_magnitude_is_preserved_and_only_the_sign_changes(
    engine_mod, request, monkeypatch, board_fixture, expect_negated
):
    """The C1 fix must flip the bonus SIGN for Black, not rescale it.

    Guards against a future 'fix' that clamps, halves or drops the bonuses, and
    pins White's behaviour as unchanged.
    """
    board = request.getfixturevalue(board_fixture)
    ranked = engine_mod.rerank_moves(board.copy())
    with_bonus = _scores_by_uci(ranked)
    raw = {e["move"].uci(): bonus_sum(engine_mod, board, e["move"]) for e in ranked}

    for name in BONUS_HELPERS:
        monkeypatch.setattr(engine_mod, name, lambda b, m: 0.0)
    without_bonus = _scores_by_uci(engine_mod.rerank_moves(board.copy()))

    checked = 0
    for uci, raw_bonus in raw.items():
        if raw_bonus == 0:
            continue
        checked += 1
        applied = with_bonus[uci] - without_bonus[uci]
        expected = -raw_bonus if expect_negated else raw_bonus
        # rerank_moves rounds scores to 3 decimals, hence the tolerance.
        assert applied == pytest.approx(expected, abs=2e-3), (
            f"{uci}: raw bonus {raw_bonus:+.3f} applied as {applied:+.3f}, "
            f"expected {expected:+.3f}"
        )

    assert checked > 0, "no bonused move found - test position is unsuitable"


# ==================================================================== known gap C4

@pytest.mark.deferred
@pytest.mark.xfail(
    strict=True,
    reason="DEFERRED (Phase 4, C4): the 'center' value in each candidate dict is "
           "computed after board.pop() (engine.py:181), so it reports the PRE-move "
           "centre control, while 'material'/'space'/'mobility' in the same dict "
           "are POST-move. The reported metrics are internally inconsistent.",
)
def test_candidate_center_should_be_the_post_move_value(engine_mod, white_to_move_board):
    ranked = engine_mod.rerank_moves(white_to_move_board)
    for entry in ranked:
        board = white_to_move_board.copy()
        board.push(entry["move"])
        expected = engine_mod.center_control(board)
        assert entry["center"] == expected, (
            f"{entry['move'].uci()}: reported centre {entry['center']} but the "
            f"post-move value is {expected}"
        )


def test_candidate_material_and_space_are_post_move_values(engine_mod, white_to_move_board):
    """Characterisation: these three ARE post-move, which is what makes the
    'center' field above inconsistent with its siblings."""
    ranked = engine_mod.rerank_moves(white_to_move_board)
    for entry in ranked[:5]:
        board = white_to_move_board.copy()
        board.push(entry["move"])
        assert entry["material"] == engine_mod.material_balance(board)
        assert entry["space"] == engine_mod.space_control(board)
        assert entry["mobility"] == engine_mod.mobility_score(board)


# ==================================================================== C2 lookahead
#
# The 1-ply lookahead must implement minimax over the opponent's replies.
#
# Established empirically (see docs/PHASE_4B_REPORT.md):
#   * the CNN is a WHITE-POSITIVE evaluator (+342cp with White a queen up,
#     -486cp with Black a queen up), and is side-to-move blind;
#   * the candidate score is also White-positive (material_balance is
#     White-minus-Black and every Ridge coefficient is positive);
#   * the opponent-reply scores are raw CNN output, with no sign transform.
#
# Therefore the opponent picks the reply best for THEM on the White-positive
# scale - Black minimises, White maximises - and that value must be blended in
# POSITIVELY, because the final sort already encodes direction (descending for
# White, ascending for Black).
#
# These tests drive rerank_moves with a stubbed CNN so the reply values are fully
# controlled, and measure the APPLIED lookahead term by differencing two runs
# that are identical except for the reply values. The candidate-batch value is
# held constant, so the weighted term and the heuristic bonuses cancel exactly in
# the difference and cannot confound the result.
#
# The expected deltas are derived from the minimax contract, NOT from the
# implementation, and they differ in sign from what the pre-fix code produces -
# so these cannot pass by sharing a wrong assumption with the implementation.

import numpy as np

LOOKAHEAD_WEIGHT = 0.5      # matches engine.py; the term is 0.5 * tanh(V / 200)
LOOKAHEAD_SCALE = 200.0


class _StubCNN:
    """Returns controlled values: constant for the candidate batch, then a caller
    supplied list for the opponent-reply batch (rerank_moves calls predict twice)."""

    def __init__(self, candidate_value, reply_values):
        self.candidate_value = candidate_value
        self.reply_values = reply_values
        self.calls = 0

    def predict(self, arr, verbose=0):
        self.calls += 1
        n = len(arr)
        if self.calls == 1:
            return np.full((n, 1), self.candidate_value, dtype=np.float32)
        assert n == len(self.reply_values), (
            f"opponent batch size {n} != prepared reply values {len(self.reply_values)}"
        )
        return np.asarray(self.reply_values, dtype=np.float32).reshape(n, 1)


def _reply_layout(board):
    """Candidate order and reply counts, mirroring how rerank_moves batches."""
    candidates = list(board.legal_moves)
    counts = []
    for mv in candidates:
        board.push(mv)
        counts.append(board.legal_moves.count())
        board.pop()
    return candidates, counts


def _run_with_reply_values(engine_mod, monkeypatch, board, values):
    stub = _StubCNN(candidate_value=0.0, reply_values=values)
    monkeypatch.setattr(engine_mod, "cnn_model", stub)
    ranked = engine_mod.rerank_moves(board.copy())
    return {e["move"].uci(): float(e["score"]) for e in ranked}


def _expected_term(value):
    return LOOKAHEAD_WEIGHT * np.tanh(value / LOOKAHEAD_SCALE)


@pytest.mark.parametrize(
    "fen,side,injected,expected_delta_desc",
    [
        # White to move: the opponent is Black, who MINIMISES the White-positive
        # score. Injecting a reply that is great for White must NOT change the
        # lookahead term - Black would never choose it.
        ("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", "white", +1000.0, "no change"),
        # ...whereas injecting a reply that is terrible for White MUST lower it.
        ("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", "white", -1000.0, "lower"),
        # Black to move: the opponent is White, who MAXIMISES. A reply that is
        # great for White MUST raise the score (worse for Black, who sorts
        # ascending).
        ("8/8/8/4k3/8/4K3/4P3/8 b - - 0 1", "black", +1000.0, "raise"),
        # ...and a reply that is terrible for White must NOT change it.
        ("8/8/8/4k3/8/4K3/4P3/8 b - - 0 1", "black", -1000.0, "no change"),
    ],
)
def test_lookahead_uses_the_opponents_best_reply(
    engine_mod, monkeypatch, fen, side, injected, expected_delta_desc
):
    board = chess.Board(fen)
    assert (board.turn == chess.WHITE) == (side == "white")

    candidates, counts = _reply_layout(board)
    total = sum(counts)

    baseline_values = [0.0] * total
    baseline = _run_with_reply_values(engine_mod, monkeypatch, board, baseline_values)

    # Inject the extreme value into the FIRST reply of the FIRST candidate only.
    injected_values = list(baseline_values)
    injected_values[0] = injected
    injected_run = _run_with_reply_values(engine_mod, monkeypatch, board, injected_values)

    target = candidates[0].uci()
    delta = injected_run[target] - baseline[target]

    if expected_delta_desc == "no change":
        assert delta == pytest.approx(0.0, abs=2e-3), (
            f"{side} to move: injecting a reply worth {injected:+.0f} changed the "
            f"lookahead by {delta:+.4f}, but the opponent would never pick that reply"
        )
    elif expected_delta_desc == "raise":
        expected = _expected_term(injected) - _expected_term(0.0)
        assert delta == pytest.approx(expected, abs=2e-3), (
            f"black to move: a reply worth {injected:+.0f} to White must RAISE the "
            f"White-positive score by {expected:+.4f} (worse for Black), got {delta:+.4f}"
        )
    else:  # lower
        expected = _expected_term(injected) - _expected_term(0.0)
        assert delta == pytest.approx(expected, abs=2e-3), (
            f"white to move: a reply worth {injected:+.0f} to White must LOWER the "
            f"score by {expected:+.4f}, got {delta:+.4f}"
        )

    # Every other candidate is untouched, so its score must be unchanged.
    for uci, score in baseline.items():
        if uci == target:
            continue
        assert injected_run[uci] == pytest.approx(score, abs=2e-3), (
            f"{uci} changed although none of its replies were modified"
        )


@pytest.mark.parametrize(
    "fen,side", [("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", "white"),
                 ("8/8/8/4k3/8/4K3/4P3/8 b - - 0 1", "black")]
)
def test_lookahead_term_matches_the_minimax_value_exactly(
    engine_mod, monkeypatch, fen, side
):
    """The applied term must equal 0.5*tanh(V/200) where V is min(replies) for
    White to move and max(replies) for Black to move."""
    board = chess.Board(fen)
    candidates, counts = _reply_layout(board)

    # Distinct spread per candidate so min and max are clearly different.
    values, offsets, cursor = [], [], 0
    for k in counts:
        offsets.append(cursor)
        spread = [(-400.0 + 200.0 * j) for j in range(k)]
        values.extend(spread)
        cursor += k

    zero = _run_with_reply_values(engine_mod, monkeypatch, board, [0.0] * sum(counts))
    actual = _run_with_reply_values(engine_mod, monkeypatch, board, values)

    opponent_maximises = (board.turn == chess.BLACK)
    for idx, mv in enumerate(candidates):
        block = values[offsets[idx]:offsets[idx] + counts[idx]]
        v = max(block) if opponent_maximises else min(block)
        expected = _expected_term(v) - _expected_term(0.0)
        got = actual[mv.uci()] - zero[mv.uci()]
        assert got == pytest.approx(expected, abs=2e-3), (
            f"{side} to move, {mv.uci()}: replies {block} -> minimax value {v}, "
            f"expected applied term {expected:+.4f}, got {got:+.4f}"
        )
