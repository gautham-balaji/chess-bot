"""C8a training harness: the A2 recipe, trained on `dataset_v2`.

    python -m training.train_v2 --arm C8a --seed 0

Additive and experiment-only. `training/train.py` is NOT modified: this module
IMPORTS its architecture, hyperparameters, callbacks, determinism setup and
metric functions, so C8a cannot drift from A2 in any of them. The only thing
that differs is where the data comes from.

--------------------------------------------------------------------------
WHY A SEPARATE MODULE IS NEEDED AT ALL
--------------------------------------------------------------------------
`train.py` assumes the dataset_v1 shape: one `.jsonl` of records, with the
train/test split DERIVED at training time by `dataset.make_split` (sort by FEN,
seeded permutation, 80/20 over POSITIONS).

`dataset_v2` cannot work that way. Its split is over GAMES, was computed before
the positions were extracted, and is already materialised as two separate files.
Re-deriving a position-level split over dataset_v2 would destroy the property the
whole of C7/C8 exists to guarantee - that no game contributes to both sides.

So this module loads the two files as given and never calls `make_split`.

--------------------------------------------------------------------------
WHAT IS HELD IDENTICAL TO A2
--------------------------------------------------------------------------
    architecture    train.build_model(representation)   -> (8,8,12), 2,360,129 params
    hyperparameters train.HP                            -> Huber, Adam 1e-3, batch 64,
                                                           cap 100, ReduceLROnPlateau
                                                           (0.5, patience 5),
                                                           EarlyStopping (patience 10,
                                                           restore best)
    callbacks       train.make_callbacks()
    determinism     train.configure_determinism(seed)
    metrics         train.evaluate_predictions()
    representation  training.representation (planes12)

--------------------------------------------------------------------------
LABELS ARE RE-DERIVED, NOT TRUSTED
--------------------------------------------------------------------------
The label vector is produced by `dataset.apply_label_policy(records,
corrected_mate_white_perspective)` - the SAME function and policy A2 used -
applied to dataset_v2's stored `raw_stockfish_value` / `eval_type` /
`side_to_move`. The result is then asserted equal to the `label` field the
builder wrote. If the builder and the A2 policy ever disagreed, this run would
fail rather than train on a silently different target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
from training import train as T  # noqa: E402

HARNESS_VERSION = "c8a-1"

DEFAULT_PREFIX = REPO_ROOT / "training" / "artifacts" / "dataset_v2"
EXPERIMENTS_DIR = REPO_ROOT / "training" / "experiments"

ARMS = {
    "C8a": {
        "dataset_prefix": "training/artifacts/dataset_v2",
        "label_policy": D.LABEL_POLICY_CORRECTED_MATE_WHITE,
        "representation": "planes12",
        "description": (
            "dataset expansion: A2's architecture, representation, label policy "
            "and training recipe are all unchanged; the training data is "
            "dataset_v2 (game-level split, 54,812 train / 13,712 test) instead "
            "of dataset_v1 (9,667 records, position-level split). Differs from "
            "A2 ONLY in the dataset - in both its SIZE and its COMPOSITION"
        ),
    },
}


def load_split(prefix: Path) -> dict:
    """Load dataset_v2's pre-computed train/test split and verify it."""
    manifest_path = Path(f"{prefix}.manifest.json")
    train_path = Path(f"{prefix}.train.jsonl")
    test_path = Path(f"{prefix}.test.jsonl")
    for path in (manifest_path, train_path, test_path):
        if not path.is_file():
            raise SystemExit(
                f"ERROR: missing {path}\n"
                f"       Rebuild with: python -m training.build_dataset_v2")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = D.load_records(train_path)
    test = D.load_records(test_path)

    # --- the files must be the ones the manifest describes ---------------------
    for side, path, rows in (("train", train_path, train), ("test", test_path, test)):
        digest = T.sha256_file(path)
        expected = manifest["artifact"][f"{side}_sha256"]
        if digest != expected:
            raise SystemExit(
                f"ERROR: {path.name} does not match its manifest.\n"
                f"       file     {digest}\n"
                f"       manifest {expected}")
        declared = manifest["final"][side]["records"]
        if len(rows) != declared:
            raise SystemExit(
                f"ERROR: {path.name} holds {len(rows)} records, manifest says "
                f"{declared}")

    # --- the split guarantee C7/C8 exists to provide ---------------------------
    train_games = {r["game_content_key"] for r in train}
    test_games = {r["game_content_key"] for r in test}
    if train_games & test_games:
        raise SystemExit(
            f"ERROR: {len(train_games & test_games)} games appear in BOTH splits; "
            f"dataset_v2 is not game-separated")
    train_pl = {r["placement"] for r in train}
    test_pl = {r["placement"] for r in test}
    if train_pl & test_pl:
        raise SystemExit(
            f"ERROR: {len(train_pl & test_pl)} piece placements appear in BOTH "
            f"splits; the leakage scrub did not hold")

    return {"manifest": manifest, "manifest_path": manifest_path,
            "train_path": train_path, "test_path": test_path,
            "train": train, "test": test}


def labels_for(records: list[dict], policy: str, side: str) -> np.ndarray:
    """Re-derive labels through the A2 policy and check them against the file."""
    derived = D.apply_label_policy(records, policy)
    stored = np.array([r["label"] for r in records], dtype=np.float32)
    if not np.array_equal(derived, stored):
        n = int((derived != stored).sum())
        raise SystemExit(
            f"ERROR: {n} {side} labels derived by "
            f"dataset.apply_label_policy({policy!r}) differ from the labels "
            f"dataset_v2 stores. Refusing to train on an ambiguous target.")
    return derived


def sha256_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def describe_side(records: list[dict], y: np.ndarray) -> dict:
    from collections import Counter
    return {
        "records": len(records),
        "distinct_games": len({r["game_content_key"] for r in records}),
        "side_to_move_counts": dict(Counter(r["side_to_move"] for r in records)),
        "phase_counts": dict(Counter(r["phase"] for r in records)),
        "eval_type_counts": dict(Counter(r["eval_type"] for r in records)),
        "checkmate_records": sum(1 for r in records if r["is_checkmate"]),
        "label": {"min": float(y.min()), "max": float(y.max()),
                  "mean": round(float(y.mean()), 4),
                  "std": round(float(y.std()), 4)},
    }


def run(arm: str, seed: int, prefix: Path, out_root: Path,
        op_determinism: bool, max_epochs: int | None) -> dict:
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; implemented arms: {sorted(ARMS)}")
    spec = ARMS[arm]

    out_dir = out_root / arm / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data ------------------------------------------------------------------
    bundle = load_split(prefix)
    manifest, train_records, test_records = (
        bundle["manifest"], bundle["train"], bundle["test"])

    y_train = labels_for(train_records, spec["label_policy"], "train")
    y_test = labels_for(test_records, spec["label_policy"], "test")

    print(f"arm={arm} seed={seed}")
    print(f"  dataset  : {manifest['dataset_name']} "
          f"(pipeline {manifest['pipeline_version']})")
    print(f"  train sha: {manifest['artifact']['train_sha256'][:16]}... (manifest OK)")
    print(f"  test  sha: {manifest['artifact']['test_sha256'][:16]}... (manifest OK)")
    print(f"  split    : {len(train_records)} train / {len(test_records)} test "
          f"(game-level, split_seed={manifest['game_split']['split_seed']})")
    print(f"  labels   : {spec['label_policy']} (re-derived and verified)")

    # --- determinism -----------------------------------------------------------
    determinism = T.configure_determinism(seed, op_determinism)
    print(f"  seeding  : op_determinism={determinism['op_determinism_enabled']}")

    # --- encode ----------------------------------------------------------------
    print(f"  encoding : {spec['representation']} {R.BOARD_SHAPE}")
    X_train = R.encode_many(r["fen"] for r in train_records)
    X_test = R.encode_many(r["fen"] for r in test_records)

    # --- train -----------------------------------------------------------------
    model = T.build_model(R)
    if model.count_params() != 2_360_129:
        raise SystemExit(
            f"ERROR: architecture is not A2's: {model.count_params()} params, "
            f"expected 2,360,129")
    epochs = max_epochs or T.HP["max_epochs"]
    started = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=T.HP["batch_size"],
        validation_split=T.HP["validation_split"],
        callbacks=T.make_callbacks(),
        verbose=2,
    )
    train_seconds = time.perf_counter() - started

    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    val_losses = hist.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else None

    # --- evaluate --------------------------------------------------------------
    y_pred_test = model.predict(X_test, verbose=0).flatten().astype(np.float64)
    y_pred_train = model.predict(X_train, verbose=0).flatten().astype(np.float64)
    test_metrics = T.evaluate_predictions(y_test.astype(np.float64), y_pred_test)
    train_metrics = T.evaluate_predictions(y_train.astype(np.float64), y_pred_train)

    # --- artifacts -------------------------------------------------------------
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "cnn_model.keras"
    model.save(model_path)

    predictions_path = out_dir / "test_predictions.json"
    predictions_path.write_text(json.dumps({
        "dataset": "dataset_v2",
        "n_test": int(len(y_test)),
        "fens": [r["fen"] for r in test_records],
        "side_to_move": [r["side_to_move"] for r in test_records],
        "phase": [r["phase"] for r in test_records],
        "is_checkmate": [bool(r["is_checkmate"]) for r in test_records],
        "y_true": [float(v) for v in y_test],
        "y_pred": [float(v) for v in y_pred_test],
    }, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "harness_version": HARNESS_VERSION,
        "arm": arm,
        "arm_description": spec["description"],
        "seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seconds": round(train_seconds, 1),
        "dataset": {
            "name": manifest["dataset_name"],
            "pipeline_version": manifest["pipeline_version"],
            "train_path": T._rel(bundle["train_path"]),
            "test_path": T._rel(bundle["test_path"]),
            "train_sha256": manifest["artifact"]["train_sha256"],
            "test_sha256": manifest["artifact"]["test_sha256"],
            "manifest_sha256": T.sha256_file(bundle["manifest_path"]),
            "source_sha256": manifest["source"]["sha256"],
            "n_train": len(train_records),
            "n_test": len(test_records),
            "split_unit": manifest["game_split"]["split_unit"],
            "split_seed": manifest["game_split"]["split_seed"],
            "extraction_policy": manifest["extraction"]["policy"],
            "train_game_sha256": manifest["game_split"]["train_game_sha256"],
            "test_game_sha256": manifest["game_split"]["test_game_sha256"],
            "is_dataset_v1": False,
        },
        "representation": R.representation_summary(),
        "label_policy": D.label_policy_summary(spec["label_policy"]),
        "label_verification": "re-derived via dataset.apply_label_policy and "
                              "asserted equal to the stored label field",
        "composition": {"train": describe_side(train_records, y_train),
                        "test": describe_side(test_records, y_test)},
        "hyperparameters": T.HP,
        "hyperparameters_source": "training/train.py HP (imported, not copied)",
        "architecture_source": "training/train.py build_model (imported)",
        "epochs_requested": epochs,
        "epochs_run": len(val_losses),
        "best_epoch": best_epoch,
        "best_val_loss": round(min(val_losses), 6) if val_losses else None,
        "final_train_loss": round(hist["loss"][-1], 6) if hist.get("loss") else None,
        "best_train_loss": round(min(hist["loss"]), 6) if hist.get("loss") else None,
        "final_val_loss": round(val_losses[-1], 6) if val_losses else None,
        "model_parameters": int(model.count_params()),
        "determinism": determinism,
        "metrics": {"train": train_metrics, "test": test_metrics},
        "history": hist,
        "history_sha256": sha256_json(hist),
        "artifacts": {
            "model_path": T._rel(model_path),
            "model_sha256": T.sha256_file(model_path),
            "model_weights_sha256": T.sha256_model_weights(model_path),
            "hash_note": (
                "model_sha256 covers the .keras container, which embeds a save "
                "timestamp and Python object ids and is NOT stable across runs. "
                "model_weights_sha256 is the reproducibility check."
            ),
            "test_predictions_sha256": T.sha256_file(predictions_path),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "packages": T.package_versions(),
            "git_commit": T._git("rev-parse", "HEAD"),
            "git_dirty": bool(T._git("status", "--porcelain")),
        },
        "scope_note": (
            "Model-level metrics are measured on dataset_v2's held-out split, "
            "which is NOT the same held-out set A2 was scored on. They are "
            "therefore not directly comparable with A2's test metrics; see the "
            "cross-evaluation in training/c8a_cross_eval.py. Engine-level "
            "evidence comes only from the Phase 3 evaluator, which uses the same "
            "suites for every arm and IS directly comparable."
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
    print(f"  history sha : {metadata['history_sha256'][:16]}...")
    print(f"  WROTE {out_dir}")
    return metadata


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="C8a", choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset-prefix", type=Path, default=DEFAULT_PREFIX)
    ap.add_argument("--out-root", type=Path, default=EXPERIMENTS_DIR)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override the epoch cap (smoke runs only)")
    ap.add_argument("--no-op-determinism", action="store_true")
    args = ap.parse_args(argv)

    run(args.arm, args.seed, args.dataset_prefix, args.out_root,
        op_determinism=not args.no_op_determinism, max_epochs=args.max_epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
