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


# ==================================================================== known gap
#
# The runtime computes sum(coef_ * features) by hand and never adds intercept_.
# The characterisation test below pins the current arithmetic; the deferred test
# states the contract a corrected implementation would satisfy. Keeping both
# means the suite records the fact without declaring "ignoring the intercept" to
# be correct behaviour.

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


@pytest.mark.deferred
@pytest.mark.xfail(
    strict=True,
    reason="DEFERRED (Phase 4, C3): the runtime applies weight_model.coef_ only "
           "and drops weight_model.intercept_ (15.0772), so hybrid_score is not "
           "the Ridge model's prediction and cannot be read as centipawns.",
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
