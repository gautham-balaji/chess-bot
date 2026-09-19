"""Controlled CNN retraining harness (C6 experiment arms).

    python -m training.train --arm A0 --seed 0

Trains one arm/seed, writes the model and full metadata under
`training/experiments/<arm>/seed_<n>/`, and never touches `models/` or any
production file.

--------------------------------------------------------------------------
A0 - the control / noise-floor arm
--------------------------------------------------------------------------
  representation : planes12          (unchanged - NOT the C6 18-plane encoding)
  label policy   : legacy_notebook   (pre-C6 policy; NOT A1's mate fix, NOT A2's
                                      perspective normalisation)
  architecture   : verified against models/cnn_model.keras config.json
  split          : fixed, identical across seeds

A0 exists to measure how much the architecture's results move from SEED ALONE,
so that a later arm's difference can be read against that noise floor rather
than assumed to be signal.

Engine-level evaluation is run separately by `training/evaluate_arm.py`, which
injects the trained model via CHESS_BOT_MODELS_DIR and calls the unmodified
Phase 3 evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

from training import dataset as D  # noqa: E402
from training import representation as R  # noqa: E402

HARNESS_VERSION = "a0-1"

DEFAULT_DATASET = REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "training" / "artifacts" / "dataset_v1.manifest.json"
EXPERIMENTS_DIR = REPO_ROOT / "training" / "experiments"

# --- architecture / training config, verified against models/cnn_model.keras ---
HP = {
    "conv_filters": [64, 128, 128],
    "conv_kernel": [3, 3],
    "conv_activation": "relu",
    "conv_padding": "same",
    "batch_norm_after_each_conv": True,
    "pooling": None,
    "dense_units": [256, 128],
    "dense_activation": "relu",
    "dropout_rates": [0.3, 0.2],
    "output_units": 1,
    "output_activation": "linear",
    "loss": "huber",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "batch_size": 64,
    "max_epochs": 100,
    "validation_split": 0.1,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 5,
    "early_stopping_patience": 10,
    "restore_best_weights": True,
    "monitor": "val_loss",
}

ARMS = {
    "A0": {
        "label_policy": D.LABEL_POLICY_LEGACY,
        "representation": "planes12",
        "description": "control / noise floor: pre-C6 label policy, 12 planes",
    },
    "A1": {
        "label_policy": D.LABEL_POLICY_CORRECTED_MATE,
        "representation": "planes12",
        "description": (
            "mate-label correction only: repaired mate scale, LEGACY "
            "side-to-move perspective retained, 12 planes"
        ),
    },
    "A2": {
        "label_policy": D.LABEL_POLICY_CORRECTED_MATE_WHITE,
        "representation": "planes12",
        "description": (
            "label perspective normalisation: A1's repaired mate scale plus "
            "White-positive labels, 12 planes. Differs from A1 ONLY in "
            "perspective"
        ),
    },
}


# ============================================================ determinism

def configure_determinism(seed: int, op_determinism: bool) -> dict:
    """Seed every RNG Keras uses; optionally force deterministic kernels."""
    import random
    import tensorflow as tf
    import keras

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    enabled, error = False, None
    if op_determinism:
        try:
            tf.config.experimental.enable_op_determinism()
            enabled = True
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)

    return {
        "seed": seed,
        "python_hash_seed": str(seed),
        "seeded": ["random", "numpy", "tf.random", "keras.utils.set_random_seed"],
        "op_determinism_requested": op_determinism,
        "op_determinism_enabled": enabled,
        "op_determinism_error": error,
    }


# ============================================================ model

def build_model():
    """Rebuild the architecture verified from models/cnn_model.keras."""
    from keras import layers, models

    model = models.Sequential([
        layers.Input(shape=R.BOARD_SHAPE),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    import keras
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=HP["learning_rate"]),
                  loss=HP["loss"])
    return model


def make_callbacks():
    import keras
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor=HP["monitor"], factor=HP["reduce_lr_factor"],
            patience=HP["reduce_lr_patience"], verbose=0),
        keras.callbacks.EarlyStopping(
            monitor=HP["monitor"], patience=HP["early_stopping_patience"],
            restore_best_weights=HP["restore_best_weights"], verbose=0),
    ]


# ============================================================ metrics

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """Keras' default Huber (delta=1.0), computed explicitly so the reported
    number is unambiguous. The original notebook printed this value under the
    label 'Test MSE', which it is not."""
    err = np.abs(y_true - y_pred)
    quadratic = np.minimum(err, delta)
    linear = err - quadratic
    return float(np.mean(0.5 * quadratic ** 2 + delta * linear))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    finite = np.isfinite(y_pred)
    n_invalid = int((~finite).sum())
    yt, yp = y_true[finite], y_pred[finite]

    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mse = float(np.mean((yt - yp) ** 2))

    if len(yt) > 1 and np.std(yt) > 0 and np.std(yp) > 0:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pearson = None

    return {
        "n_predictions": int(len(y_pred)),
        "n_invalid_predictions": n_invalid,
        "huber_loss_delta1": round(huber_loss(yt, yp), 6),
        "mse_centipawns_squared": round(mse, 4),
        "rmse_centipawns": round(rmse, 4),
        "mae_centipawns": round(mae, 4),
        "pearson_r": round(pearson, 6) if pearson is not None else None,
        "metric_note": (
            "huber_loss_delta1 is the TRAINED objective. MSE/RMSE/MAE are in "
            "centipawns and are reported separately; the original notebook "
            "printed the Huber value labelled 'Test MSE'."
        ),
    }


# ============================================================ helpers

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_model_weights(keras_path: Path) -> str:
    """Hash ONLY the weights inside a .keras archive.

    The .keras container is not byte-stable across runs: it embeds a
    `date_saved` timestamp and Keras `shared_object_id` values, which are
    Python id() addresses. Verified on two identical seed-0 runs: the
    container hashes differed while model.weights.h5 was bit-identical.
    This hash is therefore the meaningful reproducibility check.
    """
    import zipfile
    with zipfile.ZipFile(keras_path) as archive:
        return hashlib.sha256(archive.read("model.weights.h5")).hexdigest()


def _rel(path: Path) -> str:
    """Repo-relative path when possible, absolute otherwise (scratch out-roots)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT).as_posix())
    except ValueError:
        return str(path.resolve().as_posix())


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def package_versions() -> dict:
    import importlib.metadata as md

    def v(pkg):
        try:
            return md.version(pkg)
        except Exception:  # noqa: BLE001
            return None
    return {p: v(p) for p in ("tensorflow", "keras", "numpy", "scikit-learn",
                              "chess", "pandas")}


# ============================================================ run

def run(arm: str, seed: int, dataset_path: Path, manifest_path: Path,
        out_root: Path, op_determinism: bool, max_epochs: int | None) -> dict:
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; implemented arms: {sorted(ARMS)}")
    arm_spec = ARMS[arm]

    out_dir = out_root / arm / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- dataset ---------------------------------------------------------------
    if not dataset_path.is_file():
        raise SystemExit(
            f"ERROR: dataset not found: {dataset_path}\n"
            f"       Regenerate it with:\n"
            f"         python training/build_dataset.py --out "
            f"training/artifacts/dataset_v1 --verify-source"
        )
    records = D.load_records(dataset_path)
    manifest = D.load_manifest(manifest_path)
    validation = D.validate_records(records)

    dataset_sha = sha256_file(dataset_path)
    manifest_sha = manifest["artifact"]["records_sha256"]
    if dataset_sha != manifest_sha:
        raise SystemExit(
            f"ERROR: dataset checksum does not match its manifest.\n"
            f"       dataset : {dataset_sha}\n"
            f"       manifest: {manifest_sha}\n"
            f"       Refusing to train on an unidentified dataset."
        )

    data = D.build_arm_data(records, arm_spec["label_policy"])
    print(f"arm={arm} seed={seed}")
    print(f"  dataset  : {len(records)} records, sha256 {dataset_sha[:16]}... (manifest OK)")
    print(f"  labels   : {arm_spec['label_policy']}")
    print(f"  split    : {len(data.split.train_index)} train / "
          f"{len(data.split.test_index)} test (split_seed={data.split.split_seed})")

    # --- determinism -----------------------------------------------------------
    determinism = configure_determinism(seed, op_determinism)
    print(f"  seeding  : op_determinism={determinism['op_determinism_enabled']}")

    # --- encode ----------------------------------------------------------------
    X = R.encode_many(r["fen"] for r in records)
    X_train, X_test = X[data.split.train_index], X[data.split.test_index]
    y_train, y_test = data.y_train, data.y_test

    # --- train -----------------------------------------------------------------
    model = build_model()
    epochs = max_epochs or HP["max_epochs"]
    started = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=HP["batch_size"],
        validation_split=HP["validation_split"],
        callbacks=make_callbacks(),
        verbose=2,
    )
    train_seconds = time.perf_counter() - started

    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    val_losses = hist.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else None

    # --- evaluate --------------------------------------------------------------
    y_pred_test = model.predict(X_test, verbose=0).flatten().astype(np.float64)
    y_pred_train = model.predict(X_train, verbose=0).flatten().astype(np.float64)

    test_metrics = evaluate_predictions(y_test.astype(np.float64), y_pred_test)
    train_metrics = evaluate_predictions(y_train.astype(np.float64), y_pred_train)

    # --- artifacts -------------------------------------------------------------
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "cnn_model.keras"
    model.save(model_path)

    predictions_path = out_dir / "test_predictions.json"
    predictions_path.write_text(json.dumps({
        "split_seed": data.split.split_seed,
        "n_test": int(len(y_test)),
        "fens": [r["fen"] for r in data.test_records],
        "y_true": [float(v) for v in y_test],
        "y_pred": [float(v) for v in y_pred_test],
    }, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "harness_version": HARNESS_VERSION,
        "arm": arm,
        "arm_description": arm_spec["description"],
        "seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seconds": round(train_seconds, 1),
        "dataset": {
            "path": _rel(dataset_path),
            "sha256": dataset_sha,
            "manifest_records_sha256": manifest_sha,
            "manifest_matches": True,
            "pipeline_version": manifest.get("pipeline_version"),
            "source_sha256": manifest["dataset"]["source_sha256"],
            "n_records": validation["n_records"],
        },
        "representation": R.representation_summary(),
        "label_policy": D.label_policy_summary(arm_spec["label_policy"]),
        "label_stats": data.label_stats(),
        "split": data.split.summary(),
        "hyperparameters": HP,
        "epochs_requested": epochs,
        "epochs_run": len(val_losses),
        "best_epoch": best_epoch,
        "best_val_loss": round(min(val_losses), 6) if val_losses else None,
        "final_train_loss": round(hist["loss"][-1], 6) if hist.get("loss") else None,
        "final_val_loss": round(val_losses[-1], 6) if val_losses else None,
        "model_parameters": int(model.count_params()),
        "determinism": determinism,
        "metrics": {"train": train_metrics, "test": test_metrics},
        "history": hist,
        "artifacts": {
            "model_path": _rel(model_path),
            "model_sha256": sha256_file(model_path),
            "model_weights_sha256": sha256_model_weights(model_path),
            "hash_note": (
                "model_sha256 covers the .keras container, which embeds a save "
                "timestamp and Python object ids and is therefore NOT stable "
                "across runs. model_weights_sha256 is the reproducibility check."
            ),
            "test_predictions_sha256": sha256_file(predictions_path),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "packages": package_versions(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
        },
        "scope_note": (
            "Model-level metrics are measured on the held-out split of the "
            "TRAINING dataset. They say nothing about chess playing strength; "
            "engine-level evidence comes only from the Phase 3 evaluator."
        ),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"\n  epochs run  : {metadata['epochs_run']} (best {best_epoch})")
    print(f"  test huber  : {test_metrics['huber_loss_delta1']}")
    print(f"  test RMSE   : {test_metrics['rmse_centipawns']} cp")
    print(f"  test MAE    : {test_metrics['mae_centipawns']} cp")
    print(f"  test Pearson: {test_metrics['pearson_r']}")
    print(f"  weights sha : {metadata['artifacts']['model_weights_sha256'][:16]}...")
    print(f"  container   : {metadata['artifacts']['model_sha256'][:16]}... "
          f"(not byte-stable: embeds timestamp + object ids)")
    print(f"  WROTE {out_dir}")
    return metadata


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-root", type=Path, default=EXPERIMENTS_DIR)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override the epoch cap (smoke runs only)")
    ap.add_argument("--no-op-determinism", action="store_true",
                    help="skip tf.config.experimental.enable_op_determinism()")
    args = ap.parse_args()

    run(args.arm, args.seed, args.dataset, args.manifest, args.out_root,
        op_determinism=not args.no_op_determinism, max_epochs=args.max_epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
