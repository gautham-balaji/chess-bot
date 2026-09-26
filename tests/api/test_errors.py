"""Flask API error-path tests.

These assert the contract the API has: every rejected request returns HTTP 400
with a JSON `{"error": ...}` body, and no input shape produces a 500.

Phase 2 recorded two violations of that contract as paired
characterisation/`xfail(strict)` tests rather than fixing them. C10 fixed both in
app.py and replaced the characterisation tests with assertions on the intended
contract. See docs/C10_FINAL_QA.md.
"""
import chess
import pytest


# ==================================================================== /move errors

def test_illegal_move_returns_400_json(client):
    response = client.post("/move", json={"uci": "e2e5"})
    assert response.status_code == 400
    assert response.is_json
    assert "Illegal move" in response.get_json()["error"]


def test_malformed_uci_returns_400_json(client):
    response = client.post("/move", json={"uci": "zzzz"})
    assert response.status_code == 400
    assert "Invalid UCI" in response.get_json()["error"]


def test_missing_uci_field_returns_400_json(client):
    response = client.post("/move", json={})
    assert response.status_code == 400
    assert response.is_json
    assert "error" in response.get_json()


def test_empty_uci_string_returns_400(client):
    response = client.post("/move", json={"uci": ""})
    assert response.status_code == 400


def test_whitespace_uci_returns_400(client):
    response = client.post("/move", json={"uci": "   "})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "uci",
    ["e2e4e", "1234", "!!!!", "e9e4", "x" * 200, "e2", "null"],
)
def test_assorted_garbage_uci_is_rejected_without_500(client, uci):
    """Whatever the input, the server must reject cleanly - never crash."""
    response = client.post("/move", json={"uci": uci})
    assert response.status_code == 400, f"{uci!r} produced {response.status_code}"
    assert response.is_json


def test_move_out_of_turn_is_rejected_as_illegal(client):
    """It is White's move; a black move is simply not in legal_moves."""
    response = client.post("/move", json={"uci": "e7e5"})
    assert response.status_code == 400
    assert "Illegal move" in response.get_json()["error"]


def test_move_after_game_over_is_rejected(client):
    """Play fool's mate, then attempt another move."""
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        assert client.post("/move", json={"uci": uci}).status_code == 200

    assert client.get("/state").get_json()["game_over"] is True

    response = client.post("/move", json={"uci": "a2a3"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Game is over"


def test_wrong_http_method_returns_405(client):
    assert client.get("/move").status_code == 405


# =============================================== malformed bodies (FIXED IN C10)
#
# Both cases below were xfail(strict) from Phase 2 to C9.
#
# The /move handler read `request.json or {}`. In Flask >= 2.1 `request.json` RAISES
# before the `or {}` guard can apply - UnsupportedMediaType (415 + HTML) for a
# non-JSON content-type, BadRequest (400 + HTML) for a malformed JSON body - so
# /move was the one endpoint that did not honour the "errors are 400 +
# {'error': ...} JSON" contract every other path returns.
#
# C10 replaced it with get_json(silent=True), which returns None instead of
# raising, so an unusable body falls through to the same rejection as any other
# unusable input. The two former characterisation tests
# (test_non_json_body_currently_returns_415_html and its pair) asserted the 415 +
# HTML behaviour and have been REPLACED by the intended contract below.

def test_non_json_body_returns_400_json(client):
    response = client.post("/move", data="notjson", content_type="text/plain")
    assert response.status_code == 400
    assert response.is_json
    assert "error" in response.get_json()


def test_malformed_json_body_returns_400_json(client):
    response = client.post("/move", data="{not valid json", content_type="application/json")
    assert response.status_code == 400
    assert response.is_json
    assert "error" in response.get_json()


@pytest.mark.parametrize(
    "data,content_type",
    [
        ("", "application/json"),
        ("", "text/plain"),
        ("<xml/>", "application/xml"),
        ("notjson", "application/json"),
        ("[1, 2, 3]", "application/json"),
        ('"just a string"', "application/json"),
        ("null", "application/json"),
        ("17", "application/json"),
    ],
)
def test_every_unusable_body_shape_returns_400_json(client, data, content_type):
    """The contract is uniform: whatever the body, reject with 400 + JSON.

    `[1, 2, 3]` and `17` matter specifically - a truthy non-dict body used to
    reach `.get` on a list/int and raise AttributeError, i.e. a 500.
    """
    response = client.post("/move", data=data, content_type=content_type)
    assert response.status_code == 400, f"{data!r} produced {response.status_code}"
    assert response.is_json
    assert "error" in response.get_json()


# ------------------------------------------------- C10-A: non-string uci values
#
# Found during the C10 audit, not previously recorded. `data.get("uci", "")`
# returns the JSON value as-is, so a non-string uci reached `.strip()` and raised
# AttributeError -> HTTP 500. Confirmed against the pre-C10 expression for
# null, 123 and true.

@pytest.mark.parametrize("uci", [None, 123, True, 1.5, {"a": 1}, ["e2e4"]])
def test_non_string_uci_is_rejected_with_400_not_500(client, uci):
    """The regression test for C10-A."""
    response = client.post("/move", json={"uci": uci})
    assert response.status_code == 400, f"uci={uci!r} produced {response.status_code}"
    assert response.is_json
    assert "error" in response.get_json()


def test_a_valid_move_still_works_after_the_body_parsing_change(client):
    """Guards the happy path the C10 body-parsing fix runs through."""
    response = client.post("/move", json={"uci": "e2e4"})
    assert response.status_code == 200
    assert response.get_json()["player_move"]["uci"] == "e2e4"



# ==================================================================== /engine_move

def test_engine_move_on_whites_turn_returns_400(client):
    """The engine only ever plays Black - /engine_move rejects White's turn."""
    response = client.post("/engine_move")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Not engine's turn"


def test_engine_move_after_game_over_returns_400(client):
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        client.post("/move", json={"uci": uci})
    response = client.post("/engine_move")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Game is over"


def test_engine_move_ignores_an_irrelevant_body(client):
    """The handler reads no fields from the body, so junk must be harmless."""
    client.post("/move", json={"uci": "e2e4"})
    response = client.post("/engine_move", json={"unexpected": "field", "uci": "zzzz"})
    assert response.status_code == 200


def test_engine_move_with_no_body_succeeds(client):
    client.post("/move", json={"uci": "e2e4"})
    assert client.post("/engine_move").status_code == 200


# ==================================================================== /analyse

def test_analyse_after_game_over_returns_400(client):
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        client.post("/move", json={"uci": uci})
    response = client.post("/analyse")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Game is over"


def test_analyse_ignores_an_irrelevant_body(client):
    assert client.post("/analyse", json={"junk": True}).status_code == 200


# ==================================================================== /benchmark

def test_benchmark_after_game_over_returns_400(client):
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        client.post("/move", json={"uci": uci})
    response = client.post("/benchmark")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Game is over"


def test_benchmark_survives_a_stockfish_launch_failure(client, app_mod, monkeypatch):
    """A bad path must degrade to available:false, not 500.

    Phase 1 narrowed `except (FileNotFoundError, Exception)` to `except Exception`
    and now surfaces the real error text instead of always claiming
    'not installed'.
    """
    monkeypatch.setattr(app_mod.config, "find_stockfish", lambda: "/nonexistent/stockfish")
    response = client.post("/benchmark")
    assert response.status_code == 200
    assert response.get_json()["stockfish"]["available"] is False


# ==================================================================== unknown routes

def test_unknown_route_returns_404(client):
    assert client.get("/does-not-exist").status_code == 404


def test_unknown_route_post_returns_404(client):
    assert client.post("/also-missing").status_code == 404


# ==================================================================== state isolation

def test_state_is_reset_between_tests(client):
    """Guards the `client` fixture itself. app.py holds one global board, so if
    reset ever stopped working every API test above would become order-dependent."""
    assert client.get("/state").get_json()["fen"] == chess.STARTING_FEN
    client.post("/move", json={"uci": "d2d4"})
    assert client.get("/state").get_json()["fen"] != chess.STARTING_FEN


def test_state_is_reset_between_tests_again(client):
    """Deliberate near-duplicate of the test above: it only passes if the fixture
    cleaned up after that one."""
    assert client.get("/state").get_json()["fen"] == chess.STARTING_FEN
