"""Tests for training/train.py and training/evaluate_arm.py.

Covers configuration, metrics arithmetic, artifact layout and - importantly - the
guarantee that an experiment can never overwrite the production model. Nothing
here trains a network; the model-building test is the only one that touches Keras
and it only inspects shapes.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from training import dataset as D
from training import evaluate_arm as EA
from training import representation as R
from training import train as T

REPO_ROOT = Path(__file__).resolve().parents[2]


# ==================================================================== arm config

def test_a0_is_the_control_arm_with_legacy_labels_and_12_planes():
    spec = T.ARMS["A0"]
    assert spec["label_policy"] == D.LABEL_POLICY_LEGACY
    assert spec["representation"] == "planes12"


def test_a0_does_not_use_the_c6prep_label_policy():
    """A0 must not silently inherit A1's mate fix or A2's perspective fix."""
    assert T.ARMS["A0"]["label_policy"] != D.LABEL_POLICY_C6PREP


def test_hyperparameters_match_the_documented_original_pipeline():
    hp = T.HP
    assert hp["loss"] == "huber"
    assert hp["optimizer"] == "adam"
    assert hp["learning_rate"] == 1e-3
    assert hp["batch_size"] == 64
    assert hp["max_epochs"] == 100
    assert hp["reduce_lr_factor"] == 0.5
    assert hp["reduce_lr_patience"] == 5
    assert hp["early_stopping_patience"] == 10
    assert hp["restore_best_weights"] is True
    assert hp["validation_split"] == 0.1


def test_architecture_constants_match_the_saved_production_artifact():
    """Verified against models/cnn_model.keras config.json."""
    assert T.HP["conv_filters"] == [64, 128, 128]
    assert T.HP["dense_units"] == [256, 128]
    assert T.HP["dropout_rates"] == [0.3, 0.2]
    assert T.HP["pooling"] is None


# ==================================================================== model

def test_built_model_has_the_expected_io_shapes():
    model = T.build_model()
    assert model.input_shape == (None, 8, 8, 12)
    assert model.output_shape == (None, 1)


def test_built_model_parameter_count_matches_production():
    """The production CNN has 2,360,129 parameters; A0 must rebuild the same net."""
    assert T.build_model().count_params() == 2_360_129


def test_built_model_layer_sequence():
    names = [l.__class__.__name__ for l in T.build_model().layers]
    assert names == [
        "Conv2D", "BatchNormalization", "Conv2D", "BatchNormalization",
        "Conv2D", "BatchNormalization", "Flatten",
        "Dense", "Dropout", "Dense", "Dropout", "Dense",
    ]


def test_model_predicts_the_expected_output_shape():
    model = T.build_model()
    batch = R.encode_many(["4k3/8/8/8/8/8/8/4K3 w - - 0 1"] * 3)
    preds = model.predict(batch, verbose=0)
    assert preds.shape == (3, 1)
    assert np.isfinite(preds).all()


# ==================================================================== metrics

def test_huber_matches_a_hand_computed_value():
    # |err| = 0.5 -> quadratic branch: 0.5 * 0.25 = 0.125
    assert T.huber_loss(np.array([0.0]), np.array([0.5])) == pytest.approx(0.125)


def test_huber_uses_the_linear_branch_beyond_delta():
    # |err| = 3 -> 0.5*1 + 1*(3-1) = 2.5
    assert T.huber_loss(np.array([0.0]), np.array([3.0])) == pytest.approx(2.5)


def test_metrics_are_exact_on_a_known_vector():
    y = np.array([0.0, 10.0, 20.0])
    p = np.array([0.0, 12.0, 17.0])
    m = T.evaluate_predictions(y, p)
    # evaluate_predictions rounds to 4 decimals, hence the tolerance.
    assert m["mae_centipawns"] == pytest.approx(5 / 3, abs=1e-4)
    assert m["rmse_centipawns"] == pytest.approx(np.sqrt(13 / 3), abs=1e-4)
    assert m["n_predictions"] == 3
    assert m["n_invalid_predictions"] == 0


def test_perfect_predictions_give_pearson_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert T.evaluate_predictions(y, y.copy())["pearson_r"] == pytest.approx(1.0)


def test_invalid_predictions_are_counted_not_silently_dropped():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, np.nan, 3.0])
    m = T.evaluate_predictions(y, p)
    assert m["n_predictions"] == 3
    assert m["n_invalid_predictions"] == 1


def test_pearson_is_none_when_undefined():
    y = np.array([5.0, 5.0, 5.0])
    assert T.evaluate_predictions(y, y.copy())["pearson_r"] is None


def test_metrics_distinguish_huber_from_mse():
    """The original notebook printed the Huber value labelled 'Test MSE'."""
    m = T.evaluate_predictions(np.array([0.0, 0.0]), np.array([3.0, 3.0]))
    assert m["huber_loss_delta1"] != m["mse_centipawns_squared"]
    assert "Huber" in m["metric_note"] or "huber" in m["metric_note"]


# ==================================================================== safety

def test_experiment_output_never_points_at_the_production_models_dir():
    assert T.EXPERIMENTS_DIR.resolve() != (REPO_ROOT / "models").resolve()
    assert "experiments" in str(T.EXPERIMENTS_DIR)


def test_evaluate_arm_stages_a_separate_models_dir(tmp_path):
    """Injection must COPY into a scratch dir, never write into models/."""
    fake_model = tmp_path / "cnn_model.keras"
    fake_model.write_bytes(b"not-a-real-model")
    staging = tmp_path / "staged"

    result = EA.stage_models_dir(fake_model, staging)
    assert result == staging
    assert (staging / "cnn_model.keras").read_bytes() == b"not-a-real-model"
    assert (staging / "weight_model.pkl").is_file(), "production Ridge must be copied"
    assert staging.resolve() != EA.PRODUCTION_MODELS.resolve()


def test_production_model_is_not_modified_by_staging(tmp_path):
    before = {p.name: (p.stat().st_size, p.stat().st_mtime)
              for p in EA.PRODUCTION_MODELS.iterdir()}
    fake = tmp_path / "cnn_model.keras"
    fake.write_bytes(b"x")
    EA.stage_models_dir(fake, tmp_path / "staged")
    after = {p.name: (p.stat().st_size, p.stat().st_mtime)
             for p in EA.PRODUCTION_MODELS.iterdir()}
    assert before == after


def test_evaluate_arm_targets_the_existing_phase3_suites():
    for name, path in EA.SUITES.items():
        assert path.is_file(), f"{name} suite missing at {path}"
    assert set(EA.SUITES) == {"extended", "phase0_52"}


def test_injection_uses_the_documented_config_env_var():
    """config.MODELS_DIR already honours CHESS_BOT_MODELS_DIR, which is why no
    production file needs changing."""
    import config
    source = (REPO_ROOT / "config.py").read_text(encoding="utf-8")
    assert "CHESS_BOT_MODELS_DIR" in source
    assert config.MODELS_DIR.name == "models"


# ==================================================================== determinism cfg

def test_determinism_config_reports_what_it_seeded():
    cfg = T.configure_determinism(0, op_determinism=False)
    assert cfg["seed"] == 0
    assert "numpy" in cfg["seeded"] and "tf.random" in cfg["seeded"]
    assert cfg["op_determinism_requested"] is False
    assert cfg["op_determinism_enabled"] is False


def test_determinism_config_is_json_serialisable():
    json.dumps(T.configure_determinism(1, op_determinism=False))


# ==================================================================== A1 arm

def test_a1_is_registered_with_the_corrected_mate_policy():
    spec = T.ARMS["A1"]
    assert spec["label_policy"] == D.LABEL_POLICY_CORRECTED_MATE
    assert spec["representation"] == "planes12", "A1 must stay 12-plane"


def test_a1_does_not_use_the_a2_perspective_policy():
    assert T.ARMS["A1"]["label_policy"] != D.LABEL_POLICY_C6PREP


def test_a0_still_selects_legacy_labels():
    """A1's addition must not have disturbed the control arm."""
    assert T.ARMS["A0"]["label_policy"] == D.LABEL_POLICY_LEGACY


def test_a0_and_a1_differ_only_in_label_policy():
    a0, a1 = T.ARMS["A0"], T.ARMS["A1"]
    assert a0["representation"] == a1["representation"]
    assert a0["label_policy"] != a1["label_policy"]


def test_a2_is_not_implemented_yet():
    assert sorted(T.ARMS) == ["A0", "A1"]
