"""C6-A3 diagnostic: does the 18-plane model actually use the six added planes?

    python -m training.a3_plane_ablation

Read-only. No training, no Stockfish, no engine. Writes
training/experiments/A3/plane_ablation.json.

--------------------------------------------------------------------------
STATUS: EXPLORATORY
--------------------------------------------------------------------------
Not pre-registered. A3's pre-registered result is the paired A2 -> A3 contrast.
This probe only tries to explain that result, and its findings are
hypothesis-generating rather than confirmatory.

--------------------------------------------------------------------------
THE QUESTION
--------------------------------------------------------------------------
A3 completed the representation with side-to-move, castling rights and
en-passant, and fitted WORSE than A2. Two very different explanations:

  (a) the model ignores the new planes, and the deficit is the cost of six
      extra input channels (more first-layer parameters, same 7,734 samples);
  (b) the model uses them, but they mislead on this dataset.

Ablation separates these. For each A3 seed, re-predict the 1,933 held-out
positions with one plane group zeroed at a time and measure how far the
predictions move. A plane the model ignores produces no change.

Zeroing is the right ablation here because every added plane is already 0 for
"absent" - no castling right, not White to move, no en-passant square - so a
zeroed plane is a valid input the model saw during training, not an off-manifold
one.

A0/A1/A2 are 12-plane models and have no planes to ablate; they appear only as
the reference points for A3's own accuracy.
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
from training import representation18 as R18  # noqa: E402

EXP = REPO_ROOT / "training" / "experiments"
OUT = EXP / "A3" / "plane_ablation.json"
DATASET = REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl"
SEEDS = (0, 1, 2)

# Plane groups to ablate, by channel index.
GROUPS = {
    "side_to_move": [12],
    "castling": [13, 14, 15, 16],
    "en_passant": [17],
    "all_six_added": [12, 13, 14, 15, 16, 17],
}


def metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else float("nan")
    return {"mae": round(float(np.abs(err).mean()), 4),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
            "pearson_r": round(r, 6)}


def main() -> int:
    import keras

    records = D.load_records(DATASET)
    split = D.make_split(records)
    test_records = [records[i] for i in split.test_index]
    y = D.apply_label_policy(
        test_records, D.LABEL_POLICY_CORRECTED_MATE_WHITE).astype(np.float64)

    X = R18.encode_many(r["fen"] for r in test_records)
    print(f"held-out positions: {X.shape}")

    out = {"stage": "C6-A3 plane ablation",
           "status": "EXPLORATORY - not pre-registered",
           "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_positions": int(X.shape[0]),
           "ablation": "zero the plane group (a valid in-distribution value for "
                       "every added plane)",
           "groups": {k: v for k, v in GROUPS.items()},
           "per_seed": {}}

    print(f"\n{'seed':6s}{'ablation':18s}{'MAE':>10s}{'dMAE':>9s}"
          f"{'pearson':>10s}{'d pearson':>11s}{'mean |dpred|':>14s}{'max |dpred|':>13s}")

    summary = {g: [] for g in GROUPS}
    for seed in SEEDS:
        path = EXP / "A3" / f"seed_{seed}" / "models" / "cnn_model.keras"
        if not path.is_file():
            print(f"  seed {seed}: MISSING {path}")
            continue
        model = keras.models.load_model(path, compile=False)

        base_pred = model.predict(X, verbose=0, batch_size=512).reshape(-1).astype(np.float64)
        base = metrics(y, base_pred)
        print(f"  {seed:<4d}{'(none)':18s}{base['mae']:10.2f}{'':>9s}"
              f"{base['pearson_r']:10.4f}")
        row = {"baseline": base, "ablations": {}}

        for name, channels in GROUPS.items():
            X_ab = X.copy()
            X_ab[:, :, :, channels] = 0.0
            pred = model.predict(X_ab, verbose=0, batch_size=512).reshape(-1).astype(np.float64)
            m = metrics(y, pred)
            dpred = np.abs(pred - base_pred)
            entry = {
                **m,
                "delta_mae": round(m["mae"] - base["mae"], 4),
                "delta_pearson": round(m["pearson_r"] - base["pearson_r"], 6),
                "mean_abs_prediction_change": round(float(dpred.mean()), 4),
                "max_abs_prediction_change": round(float(dpred.max()), 4),
                "predictions_identical": bool(np.allclose(pred, base_pred, atol=1e-4)),
            }
            row["ablations"][name] = entry
            summary[name].append(entry["mean_abs_prediction_change"])
            print(f"  {'':4s}  {name:18s}{m['mae']:10.2f}{entry['delta_mae']:+9.2f}"
                  f"{m['pearson_r']:10.4f}{entry['delta_pearson']:+11.4f}"
                  f"{entry['mean_abs_prediction_change']:14.2f}"
                  f"{entry['max_abs_prediction_change']:13.2f}")

        out["per_seed"][f"seed_{seed}"] = row

    # ------------------------------------------------------------------ verdict
    print(f"\n{'-' * 92}\nMEAN |prediction change| when each group is zeroed, "
          f"across seeds\n{'-' * 92}")
    out["summary"] = {}
    for name, vals in summary.items():
        if vals:
            out["summary"][name] = {"per_seed": vals,
                                    "mean": round(st.fmean(vals), 4)}
            print(f"  {name:18s} {[round(v, 2) for v in vals]}   mean {st.fmean(vals):8.2f} cp")

    if out["summary"]:
        used = out["summary"]["all_six_added"]["mean"]
        out["model_uses_added_planes"] = bool(used > 1.0)
        out["verdict"] = (
            f"the added planes DO affect predictions (mean |change| {used:.1f} cp "
            f"when all six are zeroed)"
            if used > 1.0 else
            f"the added planes are effectively IGNORED (mean |change| {used:.1f} cp)")
        print(f"\n  {out['verdict']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWROTE {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
