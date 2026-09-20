"""Tests for the C8a training harness.

C8a's whole claim is "only the dataset changed". These tests pin that claim from
both sides:

  * the architecture, hyperparameters, callbacks and metrics are IMPORTED from
    `training/train.py`, not copied, so they cannot drift from A2
  * the label policy is A2's, re-derived and verified against the stored labels
  * the dataset_v2 split is loaded as built, never re-derived
  * dataset_v1 is unreachable from the C8a path
  * `training/train.py` itself is untouched

No training, no Stockfish, no writes outside tmp_path.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from training import dataset as D
from training import representation as R
from training import train as T
from training import train_v2 as V2

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFIX = REPO_ROOT / "training" / "artifacts" / "dataset_v2"


def _needs_dataset():
    if not Path(f"{PREFIX}.manifest.json").is_file():
        pytest.skip("dataset_v2 not built; run python -m training.build_dataset_v2")


@pytest.fixture(scope="module")
def bundle():
    _needs_dataset()
    return V2.load_split(PREFIX)


# ============================================== arm registration

def test_c8a_is_registered_with_a2s_label_policy_and_representation():
    spec = V2.ARMS["C8a"]
    assert spec["label_policy"] == D.LABEL_POLICY_CORRECTED_MATE_WHITE
    assert spec["representation"] == "planes12"
    assert spec["dataset_prefix"] == "training/artifacts/dataset_v2"


def test_c8a_is_the_only_arm_here_and_is_absent_from_train_arms():
    """C8a must not leak into train.ARMS, whose roster other tests pin."""
    assert sorted(V2.ARMS) == ["C8a"]
    assert "C8a" not in T.ARMS


def test_c8a_uses_the_same_label_policy_as_a2():
    assert V2.ARMS["C8a"]["label_policy"] == T.ARMS["A2"]["label_policy"]


def test_c8a_uses_the_same_representation_as_a2():
    assert V2.ARMS["C8a"]["representation"] == T.ARMS["A2"]["representation"] \
        == "planes12"


# ============================================== recipe is imported, not copied

def test_harness_imports_the_architecture_from_train_py():
    src = Path(V2.__file__).read_text(encoding="utf-8")
    assert "from training import train as T" in src
    assert "T.build_model(R)" in src
    # No local redefinition of the net or the recipe.
    for banned in ("Sequential(", "layers.Conv2D", "layers.Dense", "HP = {",
                   "def build_model", "def make_callbacks", "keras.optimizers"):
        assert banned not in src, banned


def test_hyperparameters_are_train_pys_object_not_a_copy():
    assert V2.T.HP is T.HP


def test_the_a2_recipe_values_are_the_ones_c8a_will_use():
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


def test_the_model_c8a_builds_is_a2s_exactly():
    model = T.build_model(R)
    assert model.input_shape == (None, 8, 8, 12)
    assert model.output_shape == (None, 1)
    assert model.count_params() == 2_360_129


def test_harness_refuses_a_non_a2_architecture(monkeypatch):
    """The parameter-count guard must actually fire."""
    from keras import layers, models
    tiny = models.Sequential([layers.Input(shape=(8, 8, 12)),
                              layers.Flatten(), layers.Dense(1)])
    tiny.compile(optimizer="adam", loss="huber")
    monkeypatch.setattr(T, "build_model", lambda *a, **k: tiny)
    with pytest.raises(SystemExit, match="not A2's"):
        V2.run("C8a", 0, PREFIX, Path("."), op_determinism=False, max_epochs=1)


# ============================================== dataset isolation

def test_c8a_never_loads_dataset_v1():
    """dataset_v1 must be unreachable from the C8a training path."""
    src = Path(V2.__file__).read_text(encoding="utf-8")
    body = src.split('"""', 2)[2]
    assert "dataset_v1.jsonl" not in body
    assert "load_manifest" not in body


def test_c8a_never_re_derives_the_split():
    """make_split would replace the game-level split with a position-level one."""
    body = Path(V2.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for banned in ("make_split", "build_arm_data", "train_index", "test_index"):
        assert banned not in body, banned


def test_default_prefix_points_at_dataset_v2():
    assert V2.DEFAULT_PREFIX.name == "dataset_v2"


def test_loader_rejects_a_missing_dataset(tmp_path):
    with pytest.raises(SystemExit, match="missing"):
        V2.load_split(tmp_path / "nope")


# ============================================== split integrity

def test_split_sizes_are_the_built_ones(bundle):
    assert len(bundle["train"]) == 54_812
    assert len(bundle["test"]) == 13_712


def test_loader_verifies_file_hashes_against_the_manifest(bundle, tmp_path):
    """Corrupting a file must stop the run, not train on unidentified data."""
    manifest = bundle["manifest"]
    prefix = tmp_path / "dataset_v2"
    Path(f"{prefix}.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    Path(f"{prefix}.train.jsonl").write_text('{"fen": "x"}\n', encoding="utf-8")
    Path(f"{prefix}.test.jsonl").write_text('{"fen": "x"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match its manifest"):
        V2.load_split(prefix)


def test_loader_enforces_game_separation(bundle):
    train_games = {r["game_content_key"] for r in bundle["train"]}
    test_games = {r["game_content_key"] for r in bundle["test"]}
    assert not (train_games & test_games)


def test_loader_enforces_placement_separation(bundle):
    train_pl = {r["placement"] for r in bundle["train"]}
    test_pl = {r["placement"] for r in bundle["test"]}
    assert not (train_pl & test_pl)


def test_split_is_game_level_not_position_level(bundle):
    assert bundle["manifest"]["game_split"]["split_unit"].startswith("game")
    assert bundle["manifest"]["game_split"]["split_seed"] == 42


# ============================================== labels

def test_labels_are_re_derived_through_the_a2_policy(bundle):
    y = V2.labels_for(bundle["test"], D.LABEL_POLICY_CORRECTED_MATE_WHITE, "test")
    stored = np.array([r["label"] for r in bundle["test"]], dtype=np.float32)
    assert np.array_equal(y, stored)
    assert y.dtype == np.float32


def test_label_mismatch_is_fatal():
    """A stored label that the policy would not produce must stop the run."""
    records = [{"raw_stockfish_value": 40, "eval_type": "cp",
                "side_to_move": "white", "label": 999}]
    with pytest.raises(SystemExit, match="Refusing to train"):
        V2.labels_for(records, D.LABEL_POLICY_CORRECTED_MATE_WHITE, "train")


def test_labels_obey_the_a2_bounds(bundle):
    y = V2.labels_for(bundle["train"], D.LABEL_POLICY_CORRECTED_MATE_WHITE, "train")
    assert y.min() >= -2000 and y.max() <= 2000


def test_harness_does_not_reimplement_label_logic():
    body = Path(V2.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for banned in ("def make_label", "MATE_SCORE_BASE", "CP_CLIP =",
                   "def to_white_positive", "def clip_cp"):
        assert banned not in body, banned


# ============================================== composition (recorded, not asserted as good)

def test_composition_summary_reports_what_changed(bundle):
    y = V2.labels_for(bundle["train"], D.LABEL_POLICY_CORRECTED_MATE_WHITE, "train")
    desc = V2.describe_side(bundle["train"], y)
    assert desc["records"] == 54_812
    assert set(desc["phase_counts"]) == {"opening", "middlegame", "endgame"}
    assert desc["side_to_move_counts"]["black"] > 20_000
    assert desc["checkmate_records"] == 4_658


# ============================================== reproducibility helpers

def test_history_hash_is_stable_and_order_independent():
    a = V2.sha256_json({"loss": [1.0, 2.0], "val_loss": [3.0]})
    b = V2.sha256_json({"val_loss": [3.0], "loss": [1.0, 2.0]})
    assert a == b
    assert a != V2.sha256_json({"loss": [1.0, 2.1], "val_loss": [3.0]})


def test_harness_version_is_recorded():
    assert V2.HARNESS_VERSION == "c8a-1"


# ============================================== production safety

def test_train_py_is_not_modified_by_importing_train_v2():
    """Importing the C8a harness must not mutate the shared harness."""
    assert sorted(T.ARMS) == ["A0", "A1", "A13", "A13P", "A13R", "A14", "A2", "A3"]
    assert T.HP["batch_size"] == 64
    assert T.HARNESS_VERSION == "a0-1"


def test_harness_writes_nothing_to_production_paths():
    body = Path(V2.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for banned in ("models/", "weight_model", "CHESS_BOT_MODELS_DIR", "engine"):
        assert banned not in body, banned


def test_experiment_output_root_is_the_gitignored_experiments_dir():
    assert V2.EXPERIMENTS_DIR.name == "experiments"
    assert V2.EXPERIMENTS_DIR.parent.name == "training"


def test_evaluate_arm_routes_c8a_through_the_direct_evaluator():
    """C8a is a 12-plane arm, so it must NOT use the experiment-only shim."""
    from training import evaluate_arm as EA
    from training import representations as REPS
    rep = T.ARMS.get("C8a", {}).get("representation", "planes12")
    assert rep == "planes12"
    assert REPS.get(rep).N_PLANES == 12
    import inspect
    assert "N_PLANES == 12" in inspect.getsource(EA.run_suite)
