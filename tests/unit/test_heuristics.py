"""Unit tests for the pure heuristic helpers in engine.py.

These functions are deterministic, model-free and fast - the most testable part
of the codebase. Tests assert mathematical invariants (sign, symmetry, state
restoration) rather than re-deriving the implementation.

Colour symmetry is checked with chess.Board.mirror(), which swaps piece colours,
flips the board vertically and flips the side to move.
"""
import chess
import pytest

MIRROR_ANTISYMMETRIC_FENS = [
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "8/5pk1/8/8/8/8/5PK1/R6r w - - 0 1",
    "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 1",
]


# ==================================================================== material

def test_material_balance_is_zero_at_start(engine_mod, start_board):
    assert engine_mod.material_balance(start_board) == 0


def test_material_balance_is_zero_for_kings_only(engine_mod, empty_board):
    assert engine_mod.material_balance(empty_board) == 0


@pytest.mark.parametrize(
    "fen,expected",
    [
        ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 9),    # White queen
        ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 5),    # White rook
        ("4k3/8/8/8/8/8/8/1N2K3 w - - 0 1", 3),   # White knight
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", 1),   # White pawn
        ("3qk3/8/8/8/8/8/8/4K3 w - - 0 1", -9),   # Black queen
        ("r3k3/8/8/8/8/8/8/4K3 w - - 0 1", -5),   # Black rook
    ],
)
def test_material_balance_signs_and_values(engine_mod, fen, expected):
    """White material is positive, Black material negative, kings excluded."""
    assert engine_mod.material_balance(chess.Board(fen)) == expected


def test_adding_white_material_increases_balance(engine_mod):
    base = engine_mod.material_balance(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
    richer = engine_mod.material_balance(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1"))
    assert richer > base


def test_adding_black_material_decreases_balance(engine_mod):
    base = engine_mod.material_balance(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
    poorer = engine_mod.material_balance(chess.Board("3qk3/8/8/8/8/8/8/4K3 w - - 0 1"))
    assert poorer < base


def test_material_balance_excludes_the_king(engine_mod):
    """Two kings only must be 0 - the king carries no value in this function."""
    assert engine_mod.material_balance(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")) == 0


# ==================================================================== symmetry

@pytest.mark.parametrize("fen", MIRROR_ANTISYMMETRIC_FENS)
def test_material_balance_is_antisymmetric_under_colour_mirror(engine_mod, fen):
    board = chess.Board(fen)
    assert engine_mod.material_balance(board.mirror()) == -engine_mod.material_balance(board)


@pytest.mark.parametrize("fen", MIRROR_ANTISYMMETRIC_FENS)
def test_space_control_is_antisymmetric_under_colour_mirror(engine_mod, fen):
    board = chess.Board(fen)
    assert engine_mod.space_control(board.mirror()) == -engine_mod.space_control(board)


@pytest.mark.parametrize("fen", MIRROR_ANTISYMMETRIC_FENS)
def test_center_control_is_antisymmetric_under_colour_mirror(engine_mod, fen):
    board = chess.Board(fen)
    assert engine_mod.center_control(board.mirror()) == -engine_mod.center_control(board)


@pytest.mark.parametrize("fen", MIRROR_ANTISYMMETRIC_FENS)
def test_mobility_is_invariant_under_colour_mirror(engine_mod, fen):
    """mobility_score is relative to the side to move, and mirror() flips that
    too, so the value should be unchanged rather than negated."""
    board = chess.Board(fen)
    assert engine_mod.mobility_score(board.mirror()) == engine_mod.mobility_score(board)


# ==================================================================== center

def test_center_control_is_bounded(engine_mod, sample_fens):
    """Only four squares are inspected, so the value lies in [-4, 4]."""
    for entry in sample_fens:
        value = engine_mod.center_control(chess.Board(entry["fen"]))
        assert -4 <= value <= 4, entry["id"]


def test_center_control_is_zero_in_a_symmetric_position(engine_mod, start_board):
    assert engine_mod.center_control(start_board) == 0


def test_center_control_detects_white_advantage(engine_mod):
    """A White pawn on e4 attacks d5, giving White centre presence Black lacks."""
    board = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    assert engine_mod.center_control(board) > 0


# ==================================================================== space

def test_space_control_is_zero_in_a_symmetric_position(engine_mod, start_board):
    assert engine_mod.space_control(start_board) == 0


def test_space_control_favours_the_side_with_more_pieces(engine_mod):
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert engine_mod.space_control(board) > 0


# ==================================================================== mobility
#
# mobility_score temporarily assigns board.turn to count the opponent's replies.
# Phase 1 deliberately left that implementation alone, so these tests assert the
# externally observable invariant: nothing may leak.

@pytest.mark.parametrize(
    "fen",
    [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # en passant
        "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 1",               # castling
        "8/8/8/4k3/8/4K3/4P3/8 b - - 0 1",                                 # Black to move
    ],
)
def test_mobility_score_restores_all_board_state(engine_mod, fen):
    board = chess.Board(fen)
    before = (
        board.fen(),
        board.turn,
        board.castling_rights,
        board.ep_square,
        len(board.move_stack),
    )
    engine_mod.mobility_score(board)
    after = (
        board.fen(),
        board.turn,
        board.castling_rights,
        board.ep_square,
        len(board.move_stack),
    )
    assert before == after, "mobility_score leaked board state"


def test_mobility_score_is_deterministic(engine_mod, white_to_move_board):
    first = engine_mod.mobility_score(white_to_move_board)
    second = engine_mod.mobility_score(white_to_move_board)
    assert first == second


def test_mobility_score_sign_matches_who_has_more_moves(engine_mod):
    """White queen out, Black has only a king: White must have more mobility."""
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert engine_mod.mobility_score(board) > 0


# ==================================================================== move_impact

def test_move_impact_restores_the_board(engine_mod, start_board):
    before = start_board.fen()
    engine_mod.move_impact(start_board, chess.Move.from_uci("e2e4"))
    assert start_board.fen() == before
    assert len(start_board.move_stack) == 0


def test_move_impact_is_deterministic(engine_mod, start_board):
    move = chess.Move.from_uci("e2e4")
    assert engine_mod.move_impact(start_board, move) == engine_mod.move_impact(
        start_board, move
    )


def test_move_impact_returns_an_integer_delta(engine_mod, start_board):
    assert isinstance(engine_mod.move_impact(start_board, chess.Move.from_uci("e2e4")), int)


def test_advancing_a_central_pawn_increases_space(engine_mod, start_board):
    assert engine_mod.move_impact(start_board, chess.Move.from_uci("e2e4")) > 0


# ==================================================================== bonuses

def test_development_bonus_rewards_minor_pieces(engine_mod, start_board):
    assert engine_mod.development_bonus(start_board, chess.Move.from_uci("g1f3")) == 0.2


def test_development_bonus_ignores_pawn_moves(engine_mod, start_board):
    assert engine_mod.development_bonus(start_board, chess.Move.from_uci("e2e4")) == 0


def test_pawn_push_penalty_applies_to_second_rank_pawns(engine_mod, start_board):
    assert engine_mod.pawn_push_penalty(start_board, chess.Move.from_uci("e2e4")) == -0.2


def test_pawn_push_penalty_ignores_pieces(engine_mod, start_board):
    assert engine_mod.pawn_push_penalty(start_board, chess.Move.from_uci("g1f3")) == 0


def test_tactical_bonus_rewards_captures(engine_mod):
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    capture = chess.Move.from_uci("e4d5")
    assert capture in board.legal_moves
    assert engine_mod.tactical_move_bonus(board, capture) > 0


def test_tactical_bonus_rewards_checks(engine_mod):
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    check = chess.Move.from_uci("d1d8")
    assert check in board.legal_moves
    assert engine_mod.tactical_move_bonus(board, check) > 0


def test_tactical_bonus_rewards_promotion(engine_mod):
    board = chess.Board("8/4P3/8/8/8/8/8/k2K4 w - - 0 1")
    promo = chess.Move.from_uci("e7e8q")
    assert promo in board.legal_moves
    assert engine_mod.tactical_move_bonus(board, promo) >= 0.4


def test_tactical_bonus_is_deterministic(engine_mod, start_board):
    move = chess.Move.from_uci("e2e4")
    assert engine_mod.tactical_move_bonus(start_board, move) == \
        engine_mod.tactical_move_bonus(start_board, move)


def test_tactical_bonus_restores_the_board(engine_mod, start_board):
    """It pushes to test for check, so it must pop cleanly."""
    before = start_board.fen()
    engine_mod.tactical_move_bonus(start_board, chess.Move.from_uci("e2e4"))
    assert start_board.fen() == before
    assert len(start_board.move_stack) == 0


def test_tactical_bonus_treats_central_pawn_pushes_symmetrically(engine_mod):
    """Unlike opening_center_bonus, this helper lists both colours' pushes."""
    white = chess.Board()
    black = chess.Board()
    black.push_san("e4")
    assert engine_mod.tactical_move_bonus(white, chess.Move.from_uci("e2e4")) == \
        engine_mod.tactical_move_bonus(black, chess.Move.from_uci("e7e5"))


# ==================================================================== known gaps
#
# The tests below state contracts the current implementation does not satisfy.
# Phase 2 does not fix them. strict xfail means that if Phase 4 fixes the bug,
# these XPASS and fail the suite, forcing the marker to be removed deliberately.

@pytest.mark.deferred
@pytest.mark.xfail(
    strict=True,
    reason="DEFERRED (Phase 4, C5): opening_center_bonus matches only White's UCI "
           "strings ('e2e4','d2d4','c2c4'), so Black's mirrored central pushes get 0.",
)
def test_opening_center_bonus_should_be_colour_symmetric(engine_mod):
    white = chess.Board()
    black = chess.Board()
    black.push_san("e4")
    assert engine_mod.opening_center_bonus(white, chess.Move.from_uci("e2e4")) == \
        engine_mod.opening_center_bonus(black, chess.Move.from_uci("e7e5"))


def test_opening_center_bonus_current_behaviour_is_white_only(engine_mod):
    """Characterisation test: records today's asymmetry as fact, so the pair of
    tests together make the defect unambiguous rather than merely 'failing'."""
    white = chess.Board()
    black = chess.Board()
    black.push_san("e4")
    assert engine_mod.opening_center_bonus(white, chess.Move.from_uci("e2e4")) == 0.3
    assert engine_mod.opening_center_bonus(black, chess.Move.from_uci("e7e5")) == 0


def test_bonuses_are_positive_regardless_of_side_to_move(engine_mod):
    """Characterisation test underpinning the Black-side ranking defect (C1).

    development_bonus returns the same POSITIVE value for White and Black. On its
    own that is fine; it becomes a defect only when combined with rerank_moves
    sorting ascending for Black. See tests/integration/test_engine.py.
    """
    white = chess.Board()
    black = chess.Board()
    black.push_san("e4")
    white_bonus = engine_mod.development_bonus(white, chess.Move.from_uci("g1f3"))
    black_bonus = engine_mod.development_bonus(black, chess.Move.from_uci("g8f6"))
    assert white_bonus == black_bonus == 0.2


@pytest.mark.deferred
@pytest.mark.xfail(
    strict=True,
    reason="DEFERRED (Phase 4, new): tactical_move_bonus and move_impact push "
           "before validating. An illegal move from an EMPTY square raises "
           "AssertionError after the push, so the pop never runs and the caller's "
           "board is left mutated. Needs try/finally - a production change.",
)
@pytest.mark.parametrize("helper_name", ["tactical_move_bonus", "move_impact"])
def test_helpers_should_not_corrupt_board_when_move_is_illegal(engine_mod, helper_name):
    helper = getattr(engine_mod, helper_name)
    board = chess.Board()
    before = board.fen()
    with pytest.raises(Exception):
        helper(board, chess.Move.from_uci("a3a4"))  # no piece on a3
    assert board.fen() == before, "board was left mutated after the exception"


def test_helpers_tolerate_an_illegal_move_from_an_occupied_square(engine_mod):
    """Characterisation: python-chess allows pushing such a move, so the helpers
    return a value rather than raising, and the board IS restored."""
    board = chess.Board()
    before = board.fen()
    result = engine_mod.tactical_move_bonus(board, chess.Move.from_uci("e2e5"))
    assert isinstance(result, float)
    assert board.fen() == before
