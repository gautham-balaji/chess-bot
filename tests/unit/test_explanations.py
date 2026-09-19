"""Unit tests for engine.explain_move - the user-facing XAI layer.

explain_move produces the strings the UI renders directly, so its contract is
"always a non-empty list of strings, never raises, never mutates the board".
It pushes the move internally to test for check, so restoration matters.
"""
import chess
import pytest


def test_returns_a_list_of_strings(engine_mod, start_board):
    reasons = engine_mod.explain_move(
        start_board, chess.Move.from_uci("e2e4"), {"cnn_cp": 50}
    )
    assert isinstance(reasons, list)
    assert all(isinstance(r, str) for r in reasons)


def test_is_never_empty(engine_mod, sample_fens):
    """The UI renders this directly, so an empty list would be a blank panel.
    explain_move has a fallback branch for exactly this reason."""
    for entry in sample_fens:
        board = chess.Board(entry["fen"])
        move = next(iter(board.legal_moves))
        reasons = engine_mod.explain_move(board, move, {"cnn_cp": 0})
        assert len(reasons) > 0, entry["id"]


def test_does_not_mutate_the_board(engine_mod, start_board):
    before = start_board.fen()
    engine_mod.explain_move(start_board, chess.Move.from_uci("e2e4"), {"cnn_cp": 0})
    assert start_board.fen() == before
    assert len(start_board.move_stack) == 0


def test_is_deterministic(engine_mod, start_board):
    move = chess.Move.from_uci("e2e4")
    first = engine_mod.explain_move(start_board, move, {"cnn_cp": 120})
    second = engine_mod.explain_move(start_board, move, {"cnn_cp": 120})
    assert first == second


def test_tolerates_missing_cnn_cp_key(engine_mod, start_board):
    """explain_move uses info.get('cnn_cp', 0), so an empty dict must be safe."""
    reasons = engine_mod.explain_move(start_board, chess.Move.from_uci("e2e4"), {})
    assert len(reasons) > 0


def test_mentions_center_for_a_central_move(engine_mod, start_board):
    reasons = engine_mod.explain_move(
        start_board, chess.Move.from_uci("e2e4"), {"cnn_cp": 0}
    )
    assert any("center" in r for r in reasons)


def test_mentions_development_for_a_knight_move(engine_mod, start_board):
    reasons = engine_mod.explain_move(
        start_board, chess.Move.from_uci("g1f3"), {"cnn_cp": 0}
    )
    assert any("minor piece" in r for r in reasons)


def test_mentions_capture_for_a_capture(engine_mod):
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    reasons = engine_mod.explain_move(
        board, chess.Move.from_uci("e4d5"), {"cnn_cp": 0}
    )
    assert any("captures" in r for r in reasons)


def test_mentions_check_for_a_checking_move(engine_mod):
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    reasons = engine_mod.explain_move(
        board, chess.Move.from_uci("d1d8"), {"cnn_cp": 0}
    )
    assert any("check" in r for r in reasons)


def test_mentions_promotion_for_a_promotion(engine_mod):
    board = chess.Board("8/4P3/8/8/8/8/8/k2K4 w - - 0 1")
    reasons = engine_mod.explain_move(
        board, chess.Move.from_uci("e7e8q"), {"cnn_cp": 0}
    )
    assert any("promotes" in r for r in reasons)


@pytest.mark.parametrize("cnn_cp,should_mention", [(500, True), (0, False)])
def test_neural_evaluation_reason_is_threshold_driven(
    engine_mod, start_board, cnn_cp, should_mention
):
    """The 'neural evaluation' reason appears only above cnn_cp > 100."""
    reasons = engine_mod.explain_move(
        start_board, chess.Move.from_uci("e2e4"), {"cnn_cp": cnn_cp}
    )
    mentioned = any("neural evaluation" in r for r in reasons)
    assert mentioned is should_mention
