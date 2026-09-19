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


# ==================================================================== known gap C1
#
# The heuristic bonuses are added with a fixed positive sign (engine.py:172-175)
# while the final sort direction flips by side (engine.py:207). For White, where
# higher scores rank first, a positive bonus correctly improves a move. For Black,
# where LOWER scores rank first, the same positive bonus pushes the move DOWN
# Black's own preference list.
#
# The contract below - "a positive heuristic bonus must not make a move rank
# worse for the side to move" - is asserted identically for both colours. It
# PASSES for White and FAILS for Black, which is what makes this a defect in the
# engine rather than a quirk of the test.
#
# This matters in production: app.py only ever lets the engine play Black.

def _ranking_direction_is_lower_is_better(ranked):
    """Derive the sort direction from the engine's own output rather than
    hardcoding it, so this test does not restate the implementation."""
    return ranked[0]["score"] < ranked[-1]["score"]


@pytest.mark.parametrize(
    "board_fixture",
    [
        "white_to_move_board",
        pytest.param(
            "black_to_move_board",
            marks=[
                pytest.mark.deferred,
                pytest.mark.xfail(
                    strict=True,
                    reason="DEFERRED (Phase 4, C1): heuristic bonuses are added with a "
                           "fixed positive sign while Black's ranking sorts ascending, "
                           "so a positive bonus makes a move rank WORSE for Black. The "
                           "identical assertion passes for White.",
                ),
            ],
        ),
    ],
)
def test_positive_bonus_must_not_worsen_a_move_for_the_side_to_move(
    engine_mod, request, board_fixture
):
    board = request.getfixturevalue(board_fixture)
    ranked = engine_mod.rerank_moves(board)
    lower_is_better = _ranking_direction_is_lower_is_better(ranked)

    checked = 0
    for entry in ranked:
        bonus = bonus_sum(engine_mod, board, entry["move"])
        if bonus <= 0:
            continue
        checked += 1
        score_with_bonus = entry["score"]
        score_without_bonus = entry["score"] - bonus

        if lower_is_better:
            assert score_with_bonus <= score_without_bonus, (
                f"{entry['move'].uci()}: bonus {bonus:+.3f} raised the score to "
                f"{score_with_bonus:.3f} from {score_without_bonus:.3f}, but lower "
                f"scores rank better for this side - the bonus made the move worse"
            )
        else:
            assert score_with_bonus >= score_without_bonus, (
                f"{entry['move'].uci()}: bonus {bonus:+.3f} did not improve the score"
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
