"""Shared pytest fixtures.

Two cost constraints shape this file:

1. Importing `engine` pulls in TensorFlow and loads the CNN. Measured at
   13-24s warm. Python caches modules, so the cost is paid once per pytest
   session regardless of how many tests import it - but the import is kept in a
   session-scoped fixture so collection-time cost is explicit and so tests that
   do not need the model never trigger it.

2. `engine_move()` costs ~600ms per position. Tests therefore use small curated
   position sets; the full 52-position baseline is exercised only by tests
   marked `slow`.

Global state: app.py keeps the game in module-level globals, so every API test
gets a fresh reset via the `client` fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "baseline"        # historical Phase 0 evidence, frozen
REGRESSION_DIR = REPO_ROOT / "regression"    # current accepted expectations


# ---------------------------------------------------------------- module loading

@pytest.fixture(scope="session")
def engine_mod():
    """The engine module. Session-scoped: TensorFlow + CNN load once."""
    import engine
    return engine


@pytest.fixture(scope="session")
def app_mod():
    """The Flask application module."""
    import app
    return app


@pytest.fixture
def client(app_mod):
    """Flask test client with game state reset before AND after each test.

    app.py stores board/history/captures in module-level globals, so without
    this tests would leak into each other and become order-dependent.
    """
    test_client = app_mod.app.test_client()
    test_client.post("/reset")
    yield test_client
    test_client.post("/reset")


# ---------------------------------------------------------------- boards

@pytest.fixture
def start_board():
    """Standard initial position. White to move."""
    return chess.Board()


@pytest.fixture
def empty_board():
    """Board with kings only - the minimum legal position."""
    return chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")


@pytest.fixture
def white_to_move_board():
    """Quiet opening position, White to move."""
    return chess.Board(
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    )


@pytest.fixture
def black_to_move_board():
    """Position after 1.e4 - Black to move.

    This side matters: app.py only ever lets the engine play Black.
    """
    board = chess.Board()
    board.push_san("e4")
    return board


@pytest.fixture
def tactical_board():
    """White has a back-rank mate available (Ra8#)."""
    return chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")


@pytest.fixture
def endgame_board():
    """King and pawn vs king - few legal moves, no queens."""
    return chess.Board("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1")


@pytest.fixture
def checkmate_board():
    """Terminal: fool's mate. Black has delivered mate, White to move, 0 legal moves."""
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        board.push_san(san)
    assert board.is_checkmate()
    return board


@pytest.fixture
def stalemate_board():
    """Terminal: Black to move, no legal moves, not in check."""
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    return board


@pytest.fixture
def in_check_board():
    """Black is in check and must respond."""
    board = chess.Board("rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 3")
    assert board.is_check()
    return board


# ---------------------------------------------------------------- baseline data

@pytest.fixture(scope="session")
def baseline_fens():
    """The frozen Phase 0 52-position suite."""
    with open(BASELINE_DIR / "fens.json", encoding="utf-8") as fh:
        return json.load(fh)["positions"]


@pytest.fixture(scope="session")
def baseline_engine_results():
    """ORIGINAL Phase 0 engine output, keyed by position id. Historical evidence.

    Captured before the C1 and C2 correctness fixes. Score values here no longer
    match the current engine; selected moves and top-3 ordering still do, which is
    what the historical-continuity test asserts.

    This file is never re-recorded. For the live change-detector expectations, use
    `current_engine_results`.
    """
    with open(BASELINE_DIR / "engine_results.json", encoding="utf-8") as fh:
        data = json.load(fh)
    return {row["id"]: row for row in data["pass1"]}


@pytest.fixture(scope="session")
def current_engine_results():
    """The ACCEPTED post-C2 engine output - the live regression expectations.

    Re-recorded after C1 and C2 were accepted, by running
    `baseline/scripts/measure_engine.py` against the committed engine. Generated,
    never hand-edited. See regression/README.md.

    This still records what the engine DOES, not what it SHOULD do. A later
    accepted change is expected to require re-recording it again.
    """
    with open(REGRESSION_DIR / "engine_results_post_c2.json", encoding="utf-8") as fh:
        data = json.load(fh)
    return {row["id"]: row for row in data["pass1"]}


@pytest.fixture(scope="session")
def sample_fens(baseline_fens):
    """A small cross-section of the baseline suite: one position per category,
    balanced across sides to move. Keeps integration tests to a few seconds."""
    wanted = ["OP01", "OP02", "MG02", "MG12", "EG01", "EG12", "TC01", "DF01"]
    by_id = {p["id"]: p for p in baseline_fens}
    return [by_id[i] for i in wanted if i in by_id]


# ---------------------------------------------------------------- stockfish

@pytest.fixture(scope="session")
def stockfish_path():
    """Resolved Stockfish path, or None. Never raises."""
    import config
    return config.find_stockfish()


def pytest_collection_modifyitems(config, items):  # noqa: A002 - pytest hook signature
    """Skip `needs_stockfish` tests when no binary can be resolved."""
    import config as project_config  # shadows the hook arg deliberately, hence the alias

    if project_config.find_stockfish() is not None:
        return
    skip = pytest.mark.skip(reason="no Stockfish binary resolvable on this machine")
    for item in items:
        if "needs_stockfish" in item.keywords:
            item.add_marker(skip)
