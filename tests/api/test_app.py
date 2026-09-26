"""Flask API contract tests - happy paths and response schemas.

Uses app.app.test_client(); no live server, no network. The `client` fixture
resets the shared module-level game state before and after every test, because
app.py stores the board in globals (app.py:18-25).
"""
import chess
import pytest


# ==================================================================== /

def test_index_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<html" in response.data.lower()


# ==================================================================== /state

def test_state_returns_200_json(client):
    response = client.get("/state")
    assert response.status_code == 200
    assert response.is_json


def test_state_schema(client):
    body = client.get("/state").get_json()
    expected = {
        "fen", "pieces", "legal_map", "turn", "move_history",
        "metrics", "game_over", "result", "captured_white", "captured_black",
    }
    assert expected <= set(body)


def test_state_is_the_initial_position_after_reset(client):
    body = client.get("/state").get_json()
    assert body["fen"] == chess.STARTING_FEN
    assert body["turn"] == "white"
    assert body["move_history"] == []
    assert body["game_over"] is False
    assert body["result"] is None
    assert body["captured_white"] == []
    assert body["captured_black"] == []


def test_state_reports_32_pieces_initially(client):
    assert len(client.get("/state").get_json()["pieces"]) == 32


def test_state_legal_map_matches_python_chess(client):
    legal_map = client.get("/state").get_json()["legal_map"]
    flattened = sum(len(v) for v in legal_map.values())
    assert flattened == chess.Board().legal_moves.count()


def test_state_metrics_present_while_game_is_live(client):
    metrics = client.get("/state").get_json()["metrics"]
    assert set(metrics) == {"material", "space", "center", "mobility", "cnn_eval"}


# ==================================================================== /move

def test_move_applies_a_legal_move(client):
    response = client.post("/move", json={"uci": "e2e4"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["player_move"]["uci"] == "e2e4"
    assert body["player_move"]["san"] == "e4"
    assert body["turn"] == "black"
    assert body["move_history"] == ["e4"]


def test_move_advances_the_fen(client):
    before = client.get("/state").get_json()["fen"]
    client.post("/move", json={"uci": "e2e4"})
    after = client.get("/state").get_json()["fen"]
    assert before != after


def test_move_records_a_capture(client):
    for uci in ["e2e4", "d7d5", "e4d5"]:
        assert client.post("/move", json={"uci": uci}).status_code == 200
    captured = client.get("/state").get_json()["captured_black"]
    assert "bP" in captured, "capturing a black pawn must be recorded"


def test_move_handles_promotion_uci(client):
    """A promotion needs the 5-character UCI form. Here the b-pawn walks to a7
    and promotes by capturing on b8."""
    for uci in ["b2b4", "g8f6", "b4b5", "f6g8", "b5b6", "g8f6", "b6a7", "f6g8"]:
        assert client.post("/move", json={"uci": uci}).status_code == 200, uci

    response = client.post("/move", json={"uci": "a7b8q"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["player_move"]["uci"] == "a7b8q"
    assert "Q" in body["player_move"]["san"], "SAN should record the promotion piece"


# ==================================================================== /engine_move

def test_engine_move_plays_a_legal_black_move(client):
    client.post("/move", json={"uci": "e2e4"})
    response = client.post("/engine_move")
    assert response.status_code == 200

    body = response.get_json()
    engine_block = body["engine"]
    assert set(engine_block) >= {"uci", "san", "from", "to", "explanation", "top_moves"}

    board = chess.Board()
    board.push_uci("e2e4")
    assert chess.Move.from_uci(engine_block["uci"]) in board.legal_moves


def test_engine_move_explanation_is_non_empty(client):
    client.post("/move", json={"uci": "e2e4"})
    explanation = client.post("/engine_move").get_json()["engine"]["explanation"]
    assert isinstance(explanation, list) and explanation


def test_engine_move_top_moves_schema(client):
    client.post("/move", json={"uci": "e2e4"})
    top_moves = client.post("/engine_move").get_json()["engine"]["top_moves"]
    for entry in top_moves:
        assert set(entry) >= {"uci", "san", "score", "cnn_cp", "material", "space", "reasons"}


def test_engine_move_is_deterministic(client):
    client.post("/move", json={"uci": "e2e4"})
    first = client.post("/engine_move").get_json()["engine"]["uci"]

    client.post("/reset")
    client.post("/move", json={"uci": "e2e4"})
    second = client.post("/engine_move").get_json()["engine"]["uci"]

    assert first == second


def test_engine_move_appends_to_history(client):
    client.post("/move", json={"uci": "e2e4"})
    body = client.post("/engine_move").get_json()
    assert len(body["move_history"]) == 2
    assert body["turn"] == "white"


# ==================================================================== /analyse

def test_analyse_returns_top_moves_and_metrics(client):
    response = client.post("/analyse")
    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"top_moves", "metrics"}
    assert 0 < len(body["top_moves"]) <= 3


def test_analyse_does_not_change_the_board(client):
    before = client.get("/state").get_json()["fen"]
    client.post("/analyse")
    assert client.get("/state").get_json()["fen"] == before


def test_analyse_suggests_legal_moves(client):
    top_moves = client.post("/analyse").get_json()["top_moves"]
    board = chess.Board()
    for entry in top_moves:
        assert chess.Move.from_uci(entry["uci"]) in board.legal_moves


# ==================================================================== /reset

def test_reset_restores_the_initial_position(client):
    client.post("/move", json={"uci": "e2e4"})
    body = client.post("/reset").get_json()
    assert body["fen"] == chess.STARTING_FEN
    assert body["move_history"] == []
    assert body["turn"] == "white"


def test_reset_clears_captures(client):
    for uci in ["e2e4", "d7d5", "e4d5"]:
        client.post("/move", json={"uci": uci})
    body = client.post("/reset").get_json()
    assert body["captured_white"] == []
    assert body["captured_black"] == []


# ==================================================================== /game_stats

def test_game_stats_schema(client):
    response = client.post("/game_stats")
    assert response.status_code == 200
    body = response.get_json()
    expected = {
        "total_moves", "white_moves", "black_moves", "captured_white",
        "captured_black", "avg_material", "avg_space", "avg_center", "saliency",
    }
    assert expected <= set(body)


def test_game_stats_counts_moves(client):
    client.post("/move", json={"uci": "e2e4"})
    client.post("/move", json={"uci": "e7e5"})
    body = client.post("/game_stats").get_json()
    assert body["total_moves"] == 2
    assert body["white_moves"] == 1
    assert body["black_moves"] == 1


def test_game_stats_saliency_is_8x8_normalised(client):
    saliency = client.post("/game_stats").get_json()["saliency"]
    assert saliency is not None, "expected Integrated Gradients to succeed here"
    assert len(saliency) == 8
    assert all(len(row) == 8 for row in saliency)
    flat = [v for row in saliency for v in row]
    assert min(flat) >= 0.0 and max(flat) <= 1.0


def test_game_stats_degrades_to_null_saliency_on_failure(client, app_mod, monkeypatch):
    """Characterisation: saliency failure is swallowed and reported as null with
    HTTP 200. Recorded, not fixed - a silent 200 hides the failure from callers."""
    def boom(*args, **kwargs):
        raise RuntimeError("forced saliency failure")

    monkeypatch.setattr(app_mod, "cnn_model", boom)
    response = client.post("/game_stats")
    assert response.status_code == 200
    assert response.get_json()["saliency"] is None


# ==================================================================== /forfeit

def test_forfeit_marks_the_game_over(client):
    response = client.post("/forfeit")
    assert response.status_code == 200
    body = response.get_json()
    assert body["forfeited"] is True
    assert body["game_over"] is True


def test_forfeit_always_reports_0_1(client):
    """Characterisation of a known defect: /forfeit hardcodes '0-1' regardless of
    who resigned. Deferred - recorded here so a future fix is a deliberate,
    visible change. Harmless while the app only ever lets the engine play Black,
    which it enforces. See docs/FINAL_QA_REPORT.md section 7."""
    body = client.post("/forfeit").get_json()
    assert body["result"]["result"] == "0-1"
    assert body["result"]["reason"] == "Resignation"


# ==================================================================== /model_info

def test_model_info_schema(client):
    response = client.get("/model_info")
    assert response.status_code == 200
    body = response.get_json()
    assert {"architecture", "input_shape", "layers", "total_params", "training"} <= set(body)


def test_model_info_reports_real_layers(client):
    body = client.get("/model_info").get_json()
    assert body["input_shape"] == [8, 8, 12]
    assert body["total_params"] > 0
    assert len(body["layers"]) > 0
    for layer in body["layers"]:
        assert {"index", "name", "type", "output_shape", "params"} <= set(layer)


def test_model_info_training_figures_match_the_notebook(client):
    """Phase 1 corrected these from fabricated values. Guard against regression."""
    training = client.get("/model_info").get_json()["training"]
    assert training["correlation_vs_stockfish"] == 0.708
    assert training["dataset_size"] == "10,000 positions"
    assert "correlation_note" in training
    assert "source" in training


def test_model_info_has_no_fabricated_claims(client):
    """The pre-Phase-1 values were 0.506 and '~50,000 positions', neither of which
    matched the notebook."""
    training = client.get("/model_info").get_json()["training"]
    assert training["correlation_vs_stockfish"] != 0.506
    assert "50,000" not in training["dataset_size"]


# ==================================================================== /benchmark

def test_benchmark_returns_the_expected_blocks(client):
    response = client.post("/benchmark")
    assert response.status_code == 200
    assert {"fen", "hybrid", "stockfish", "comparison"} <= set(response.get_json())


def test_benchmark_never_reports_the_invalid_score_difference(client):
    """Phase 1 removed eval_diff_cp: it subtracted a side-to-move-relative
    Stockfish centipawn score from the engine's non-centipawn score."""
    comparison = client.post("/benchmark").get_json().get("comparison", {})
    assert "eval_diff_cp" not in comparison


def test_benchmark_never_reports_a_misleading_speedup(client):
    """'speedup' read as a speed-up but was < 1 because the engine is slower."""
    comparison = client.post("/benchmark").get_json().get("comparison", {})
    assert "speedup" not in comparison
    assert "time_saved_ms" not in comparison


def test_benchmark_reports_unavailable_when_stockfish_is_missing(
    client, app_mod, monkeypatch
):
    """Must degrade gracefully, not 500."""
    monkeypatch.setattr(app_mod.config, "find_stockfish", lambda: None)
    response = client.post("/benchmark")
    assert response.status_code == 200

    body = response.get_json()
    assert body["stockfish"]["available"] is False
    assert "STOCKFISH_PATH" in body["stockfish"]["note"]
    assert body["hybrid"]["move"] is not None, "the engine half must still run"


@pytest.mark.needs_stockfish
def test_benchmark_with_stockfish_available(client):
    body = client.post("/benchmark").get_json()
    assert body["stockfish"].get("available") is not False

    board = chess.Board()
    assert chess.Move.from_uci(body["stockfish"]["move"]) in board.legal_moves
    assert body["stockfish"]["depth"] == 8

    comparison = body["comparison"]
    assert isinstance(comparison["agreement"], bool)
    assert comparison["engine_time_ratio_vs_stockfish"] > 0
    assert "greater than 1" in comparison["ratio_note"]


@pytest.mark.needs_stockfish
def test_benchmark_agreement_is_computed_not_randomised(client):
    """The pre-Phase-1 script set agreement with np.random.rand() < 0.7. The
    endpoint's version must be a real comparison, so it is stable across calls."""
    first = client.post("/benchmark").get_json()
    second = client.post("/benchmark").get_json()

    assert first["comparison"]["agreement"] == second["comparison"]["agreement"]
    expected = first["hybrid"]["move"] == first["stockfish"]["move"]
    assert first["comparison"]["agreement"] == expected
