"""C6-A13P diagnostic: why does A13P train fine but validate catastrophically?

    python -m training.a13p_bn_diagnostic

Read-only. No training, no Stockfish, no engine. Writes
training/experiments/A13P/bn_diagnostic.json.

--------------------------------------------------------------------------
THE OBSERVATION
--------------------------------------------------------------------------
A13P's TRAINING loss descends normally and tracks A2's almost exactly
(226 -> 109 over 11 epochs, against A2's 222 -> 107). Its VALIDATION loss
explodes and oscillates: 237 -> 699 -> 2165 -> 1151 -> 576 -> ... All three
seeds stop early with best_epoch 1, 2 and 4.

Healthy train loss with a diverging val loss, in a network whose only change is
four information-free constant input channels, points at BatchNormalization
rather than at anything representational.

--------------------------------------------------------------------------
THE HYPOTHESIS
--------------------------------------------------------------------------
The first layer is Conv2D(..., activation="relu") followed by
BatchNormalization. A spatially constant input channel contributes the SAME
offset to every sample in the batch. In TRAINING mode BatchNorm subtracts the
batch mean, so that offset is removed exactly and the training loss is blind to
it. The 2,304 weights feeding those channels therefore sit in a direction that
training loss cannot constrain - they can drift freely.

At INFERENCE BatchNorm uses running averages instead, which are an EMA and lag
behind the drifting activations. The further the weights wander along the
unconstrained direction, the worse the mismatch, and validation loss blows up.

A13's castling planes do NOT create this flat direction, because they vary
across positions, so the offset differs per sample and the batch mean cannot
absorb it.

--------------------------------------------------------------------------
THE TESTS
--------------------------------------------------------------------------
1. Predict the held-out set with BatchNorm in TRAINING mode (batch statistics)
   and in INFERENCE mode (running statistics). If the hypothesis holds,
   training-mode predictions are good and inference-mode ones are not - for
   A13P, but not for A2 or A13.
2. Compare the learned first-layer weight magnitude on the added channels
   against the piece channels. Unconstrained drift should show up as inflation.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from training import dataset as D  # noqa: E402
from training import representations as REPS  # noqa: E402

EXP = REPO_ROOT / "training" / "experiments"
OUT = EXP / "A13P" / "bn_diagnostic.json"
DATASET = REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl"
SEEDS = (0, 1, 2)
ARMS = ("A2", "A13", "A13P")


def metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    finite = np.isfinite(y_pred)
    r = (float(np.corrcoef(y_true[finite], y_pred[finite])[0, 1])
         if finite.sum() > 2 else float("nan"))
    return {"mae": round(float(np.abs(err).mean()), 4),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
            "pearson_r": round(r, 6)}


def main() -> int:
    import keras
    from training import train as T

    records = D.load_records(DATASET)
    split = D.make_split(records)
    test_records = [records[i] for i in split.test_index]
    y = D.apply_label_policy(
        test_records, D.LABEL_POLICY_CORRECTED_MATE_WHITE).astype(np.float64)

    out = {"stage": "C6-A13P BatchNorm diagnostic",
           "status": "EXPLORATORY - not pre-registered",
           "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_positions": len(test_records),
           "hypothesis": ("constant input channels create a direction the training "
                          "loss cannot constrain, because BatchNorm removes a "
                          "batch-constant offset in training mode; the inference-mode "
                          "running statistics then mismatch"),
           "per_run": {}}

    encoded = {}

    def encode_for(arm):
        name = T.ARMS[arm]["representation"]
        if name not in encoded:
            encoded[name] = REPS.get(name).encode_many(r["fen"] for r in test_records)
        return encoded[name]

    print("=" * 96)
    print("BATCHNORM MODE COMPARISON on the 1,933 held-out positions")
    print("=" * 96)
    print(f"\n{'run':10s}{'inference MAE':>16s}{'training-mode MAE':>20s}"
          f"{'ratio':>9s}{'inference r':>13s}{'train-mode r':>14s}")

    ratios = {a: [] for a in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            path = EXP / arm / f"seed_{seed}" / "models" / "cnn_model.keras"
            if not path.is_file():
                continue
            model = keras.models.load_model(path, compile=False)
            X = encode_for(arm)

            # inference mode: BatchNorm uses its running statistics
            infer = np.asarray(model.predict(X, verbose=0, batch_size=64)).reshape(-1)
            # training mode: BatchNorm uses the statistics of each batch
            train_mode = []
            for i in range(0, len(X), 64):
                train_mode.append(
                    np.asarray(model(X[i:i + 64], training=True)).reshape(-1))
            train_mode = np.concatenate(train_mode)

            mi, mt = metrics(y, infer.astype(np.float64)), metrics(y, train_mode.astype(np.float64))
            ratio = mi["mae"] / mt["mae"] if mt["mae"] else float("nan")
            ratios[arm].append(ratio)
            out["per_run"][f"{arm}_seed_{seed}"] = {
                "inference_mode": mi, "training_mode": mt,
                "mae_ratio_inference_over_training": round(ratio, 4)}
            print(f"  {arm:5s}s{seed}{mi['mae']:16.2f}{mt['mae']:20.2f}"
                  f"{ratio:9.2f}{mi['pearson_r']:13.4f}{mt['pearson_r']:14.4f}")

    print(f"\n{'-' * 96}\nMEAN inference/training MAE ratio (1.0 = the two modes agree)\n{'-' * 96}")
    out["summary"] = {}
    for arm in ARMS:
        if ratios[arm]:
            out["summary"][arm] = {"per_seed": [round(x, 4) for x in ratios[arm]],
                                   "mean": round(st.fmean(ratios[arm]), 4)}
            print(f"  {arm:6s} {[round(x, 2) for x in ratios[arm]]}   "
                  f"mean {st.fmean(ratios[arm]):.2f}")

    # ---------------------------------------------------------------- weights
    print(f"\n{'-' * 96}\nFIRST-LAYER WEIGHT MAGNITUDE: added channels vs piece "
          f"channels\n{'-' * 96}")
    print(f"{'run':10s}{'mean |w| ch 0-11':>19s}{'mean |w| ch 12+':>18s}{'ratio':>9s}")
    out["first_layer_weights"] = {}
    for arm in ("A13", "A13P"):
        for seed in SEEDS:
            path = EXP / arm / f"seed_{seed}" / "models" / "cnn_model.keras"
            if not path.is_file():
                continue
            model = keras.models.load_model(path, compile=False)
            kernel = np.asarray(model.layers[0].get_weights()[0])   # (3,3,C,64)
            piece = float(np.abs(kernel[:, :, :12, :]).mean())
            added = float(np.abs(kernel[:, :, 12:, :]).mean())
            ratio = added / piece if piece else float("nan")
            out["first_layer_weights"][f"{arm}_seed_{seed}"] = {
                "mean_abs_piece_channels": round(piece, 6),
                "mean_abs_added_channels": round(added, 6),
                "ratio": round(ratio, 4)}
            print(f"  {arm:5s}s{seed}{piece:19.5f}{added:18.5f}{ratio:9.2f}")

    # ---------------------------------------------------------------- verdict
    a13p = out["summary"].get("A13P", {}).get("mean")
    a2 = out["summary"].get("A2", {}).get("mean")
    a13 = out["summary"].get("A13", {}).get("mean")
    if a13p is not None and a2 is not None and a13 is not None:
        supported = a13p > 2 * max(a2, a13)
        out["hypothesis_supported"] = bool(supported)
        out["verdict"] = (
            f"SUPPORTED: A13P's inference/training MAE ratio is {a13p:.1f} vs "
            f"{a2:.1f} (A2) and {a13:.1f} (A13). The placebo arm's damage is a "
            f"BatchNorm train/inference mismatch, not a representational effect."
            if supported else
            f"NOT SUPPORTED: A13P ratio {a13p:.1f}, A2 {a2:.1f}, A13 {a13:.1f}")
        print(f"\n{out['verdict']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWROTE {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
