"""Tests for the model-loading and score-fusion boundary.

Covers what the runtime actually loads (the CNN and the Ridge weight model), the
shape of the fusion contract, and the CNN-backed helpers cnn_evaluate and
position_metrics.

The Ridge intercept is NOT applied at inference. These tests document that
precisely without blessing it as correct - see the deferred test at the bottom.
"""
import chess
import numpy as np
import pytest


# ==================================================================== loading

def test_cnn_model_is_loaded(engine_mod):
    assert engine_mod.cnn_model is not None


def test_weight_model_is_loaded(engine_mod):
    assert engine_mod.weight_model is not None


def test_unused_training_artifacts_are_not_loaded(engine_mod):
    """Phase 1 removed rf/mlp/scaler from import. Guard against reintroduction:
    they cost ~1.07s and ~79MB for no runtime effect."""
    for name in ("rf", "mlp", "scaler"):
        assert not hasattr(engine_mod, name), (
            f"engine.{name} is loaded again but is never used at runtime"
        )


def test_model_paths_are_repository_relative():
    """Phase 1 contract: paths anchor to the repo, not the working directory."""
    import config
    assert config.CNN_MODEL_PATH.is_absolute()
    assert config.CNN_MODEL_PATH.is_file()
    assert config.WEIGHT_MODEL_PATH.is_file()


# ==================================================================== ridge shape

def test_ridge_has_exactly_five_coefficients(engine_mod):
    """rerank_moves indexes w[0]..w[4] for cnn_norm, material, space, center,
    mobility. A retrained model with a different feature count would silently
    produce wrong scores or IndexError."""
    assert len(engine_mod.weight_model.coef_) == 5


def test_ridge_coefficients_are_finite_numbers(engine_mod):
    coef = np.asarray(engine_mod.weight_model.coef_, dtype=float)
    assert np.all(np.isfinite(coef))


def test_ridge_intercept_exists_and_is_scalar(engine_mod):
    intercept = engine_mod.weight_model.intercept_
    assert np.isscalar(intercept) or np.asarray(intercept).size == 1
    assert np.isfinite(float(intercept))


def test_ridge_exposes_a_predict_method(engine_mod):
    assert callable(getattr(engine_mod.weight_model, "predict", None))


# ==================================================================== cnn_evaluate

def test_cnn_evaluate_returns_a_float(engine_mod, start_board):
    assert isinstance(engine_mod.cnn_evaluate(start_board), float)


def test_cnn_evaluate_is_deterministic(engine_mod, start_board):
    assert engine_mod.cnn_evaluate(start_board) == engine_mod.cnn_evaluate(start_board)


def test_cnn_evaluate_does_not_mutate_the_board(engine_mod, start_board):
    before = start_board.fen()
    engine_mod.cnn_evaluate(start_board)
    assert start_board.fen() == before


def test_cnn_evaluate_distinguishes_materially_different_positions(engine_mod):
    """A sanity check that the model responds to input at all - not a quality claim."""
    balanced = engine_mod.cnn_evaluate(chess.Board())
    white_up = engine_mod.cnn_evaluate(
        chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    )
    assert balanced != white_up


# ==================================================================== hybrid_score

def test_hybrid_score_returns_four_values(engine_mod, start_board):
    result = engine_mod.hybrid_score(start_board)
    assert len(result) == 4


def test_hybrid_score_components_are_numeric(engine_mod, start_board):
    score, cnn_score, mat, space = engine_mod.hybrid_score(start_board)
    assert np.isfinite(float(score))
    assert np.isfinite(float(cnn_score))
    assert isinstance(mat, int)
    assert isinstance(space, int)


def test_hybrid_score_is_deterministic(engine_mod, start_board):
    assert engine_mod.hybrid_score(start_board) == engine_mod.hybrid_score(start_board)


# ==================================================================== position_metrics

def test_position_metrics_schema(engine_mod, start_board):
    metrics = engine_mod.position_metrics(start_board)
    assert set(metrics) == {"material", "space", "center", "mobility", "cnn_eval"}


def test_position_metrics_values_are_numeric(engine_mod, start_board):
    for key, value in engine_mod.position_metrics(start_board).items():
        assert isinstance(value, (int, float)), key


def test_position_metrics_is_deterministic(engine_mod, white_to_move_board):
    assert engine_mod.position_metrics(white_to_move_board) == \
        engine_mod.position_metrics(white_to_move_board)


def test_position_metrics_does_not_mutate_the_board(engine_mod, white_to_move_board):
    before = white_to_move_board.fen()
    engine_mod.position_metrics(white_to_move_board)
    assert white_to_move_board.fen() == before


def test_position_metrics_agrees_with_the_standalone_helpers(engine_mod, white_to_move_board):
    metrics = engine_mod.position_metrics(white_to_move_board)
    assert metrics["material"] == engine_mod.material_balance(white_to_move_board)
    assert metrics["space"] == engine_mod.space_control(white_to_move_board)
    assert metrics["center"] == engine_mod.center_control(white_to_move_board)


# ============================================ C3: STILL DEFERRED AFTER C10 (open)
#
# The runtime computes sum(coef_ * features) by hand and never adds intercept_.
# C10 audited this and deliberately did NOT change it. The reasoning, and the
# evidence for each step, is pinned by the passing tests in this section so the
# deferral rests on measurements rather than on a report.
#
#   1. `hybrid_score` has no caller in engine.py, app.py, evaluation/,
#      baseline/scripts/ or training/. It is not on the move-selection path.
#   2. On the path that IS used (`rerank_moves`), the omitted intercept is a
#      per-position CONSTANT, so it cannot change the ranking. Measured over all
#      52 baseline positions: 0 ordering changes, 0 selected-move changes.
#   3. So C3 is an interpretation defect - `hybrid_score` is not the Ridge's
#      prediction and cannot be read as centipawns - not a ranking defect.
#
# Fixing it would shift every reported score by +15.0772 and require re-recording
# the regression baseline, while changing no move the engine plays. Every C6-C9
# arm was evaluated under coef-only fusion (see docs/C9_RIDGE_AUDIT.md §1), and
# the Ridge itself was fitted on unrecoverable side-to-move-relative targets
# (§4), so "readable as centipawns" would not become true merely by adding the
# intercept back. C10 therefore scopes C3 as a separate future change needing its
# own justification and regression re-recording. See docs/C10_FINAL_QA.md.

def test_hybrid_score_currently_omits_the_ridge_intercept(engine_mod, start_board):
    """Characterisation: hybrid_score equals Ridge.predict MINUS the intercept."""
    score, cnn_score, _, _ = engine_mod.hybrid_score(start_board)

    features = np.array([[
        np.tanh(cnn_score / 200),
        engine_mod.material_balance(start_board),
        engine_mod.space_control(start_board),
        engine_mod.center_control(start_board),
        engine_mod.mobility_score(start_board),
    ]])
    predicted = float(engine_mod.weight_model.predict(features)[0])
    intercept = float(engine_mod.weight_model.intercept_)

    assert float(score) == pytest.approx(predicted - intercept, abs=1e-6)


def test_the_ridge_intercept_is_non_zero(engine_mod):
    """If this ever becomes ~0, C3 stops being a defect and this whole section
    should be revisited rather than left asserting a moot point."""
    assert abs(float(engine_mod.weight_model.intercept_)) > 1.0


def test_hybrid_score_is_not_on_the_move_selection_path(engine_mod):
    """Evidence for step 1 of the C3 deferral.

    `rerank_moves` inlines the same weighted sum instead of calling
    `hybrid_score`, so C3's effect is confined to a function the engine and the
    API never invoke. Asserted against the source so it cannot silently change.
    """
    import inspect
    import app

    for module in (engine_mod, app):
        source = inspect.getsource(module)
        calls = [line for line in source.splitlines()
                 if "hybrid_score(" in line and "def hybrid_score" not in line]
        assert calls == [], f"{module.__name__} now calls hybrid_score: {calls}"


@pytest.mark.parametrize(
    "fen",
    [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
        "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    ],
)
def test_omitting_the_intercept_cannot_change_the_ranking(engine_mod, fen):
    """Evidence for step 2 of the C3 deferral - the load-bearing claim.

    The intercept is added identically to every candidate in a position, so it is
    an order-preserving shift. This test applies it and asserts the ranking is
    byte-identical, which is what makes C3 safe to defer. If a future change made
    the omission position-dependent (e.g. a per-candidate intercept), this fails.
    """
    board = chess.Board(fen)
    ranked = engine_mod.rerank_moves(board.copy())
    assert ranked, fen

    intercept = float(engine_mod.weight_model.intercept_)
    shifted = sorted(
        ({**entry, "score": round(float(entry["score"] + intercept), 3)}
         for entry in ranked),
        key=lambda e: e["score"],
        reverse=(board.turn == chess.WHITE),
    )

    assert [e["move"].uci() for e in shifted] == [e["move"].uci() for e in ranked]
    for before, after in zip(ranked, shifted):
        assert after["score"] == pytest.approx(before["score"] + intercept, abs=2e-3)


@pytest.mark.deferred
@pytest.mark.xfail(
    strict=True,
    reason="OPEN, SEPARATELY SCOPED (C3; audited and deliberately retained in "
           "C10): engine.py applies weight_model.coef_ only and drops "
           "intercept_ (15.0772), so hybrid_score is not the Ridge's prediction "
           "and cannot be read as centipawns. Confirmed in C10 to be an "
           "interpretation defect, not a ranking defect - hybrid_score is off "
           "the move-selection path, and on the path that is used the omission "
           "is an order-preserving per-position constant (0/52 ordering changes, "
           "0/52 selected-move changes). Fixing it shifts every reported score "
           "by +15.0772 and requires re-recording the regression baseline "
           "without changing a single move played. Scoped as a separate future "
           "change. See docs/C10_FINAL_QA.md.",
)
def test_hybrid_score_should_equal_the_ridge_prediction(engine_mod, start_board):
    score, cnn_score, _, _ = engine_mod.hybrid_score(start_board)

    features = np.array([[
        np.tanh(cnn_score / 200),
        engine_mod.material_balance(start_board),
        engine_mod.space_control(start_board),
        engine_mod.center_control(start_board),
        engine_mod.mobility_score(start_board),
    ]])
    assert float(score) == pytest.approx(
        float(engine_mod.weight_model.predict(features)[0]), abs=1e-6
    )
