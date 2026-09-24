"""C8a cross-evaluation: score every arm on BOTH held-out sets.

    python -m training.c8a_cross_eval

Read-only over trained models. No Stockfish, no training, no production write.
Writes training/experiments/C8a/cross_eval.json.

--------------------------------------------------------------------------
WHY THIS IS NECESSARY
--------------------------------------------------------------------------
A2 and C8a do NOT share a held-out set:

    A2   trained on dataset_v1 -> tested on dataset_v1's 1,933 position-split
         records, every one an opening position at <= ply 20
    C8a  trained on dataset_v2 -> tested on dataset_v2's 13,712 game-split
         records, spanning opening/middlegame/endgame

So their reported test Huber, MAE, RMSE and Pearson are measured on different
distributions and **cannot be compared directly**. A lower number on an easier
or merely different set says nothing.

This module removes that ambiguity by scoring every model on BOTH sets. The
diagonal reproduces each arm's own reported metrics; the off-diagonal is the
comparison that actually means something.

Engine-level metrics do not need this treatment: every arm is evaluated on the
same two suites by the same unmodified evaluator, so those numbers are already
comparable. Model-level and engine-level results are kept separate throughout.

--------------------------------------------------------------------------
LABEL CONSISTENCY
--------------------------------------------------------------------------
Both held-out sets are labelled with the SAME A2 policy
(`corrected_mate_white_perspective`), re-derived here through
`dataset.apply_label_policy` rather than read from either file, so the target is
identical in both columns of the table.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import chess  # noqa: E402
import numpy as np  # noqa: E402

from training import dataset as D  # noqa: E402
from training import representation as R  # noqa: E402
from training import train as T  # noqa: E402

EXP = REPO_ROOT / "training" / "experiments"
ARTIFACTS = REPO_ROOT / "training" / "artifacts"
OUT = EXP / "C8a" / "cross_eval.json"
SEEDS = (0, 1, 2)
POLICY = D.LABEL_POLICY_CORRECTED_MATE_WHITE


def load_v1_test() -> list[dict]:
    """dataset_v1's held-out split, rebuilt with the same code A2 used."""
    records = D.load_records(ARTIFACTS / "dataset_v1.jsonl")
    split = D.make_split(records)
    return [records[i] for i in split.test_index]


def load_v2_test() -> list[dict]:
    return D.load_records(ARTIFACTS / "dataset_v2.test.jsonl")


def load_v2_k6_test() -> list[dict]:
    """C8b's held-out split. Same game-level split as dataset_v2's, sampled more
    densely, so it is a superset in games but not in positions."""
    return D.load_records(ARTIFACTS / "dataset_v2_k6.test.jsonl")


def annotate(records: list[dict]) -> dict:
    """Per-record side to move and phase, derived identically for both sets."""
    stm, phase, mated = [], [], []
    for r in records:
        board = chess.Board(r["fen"])
        stm.append("white" if board.turn == chess.WHITE else "black")
        pieces = chess.popcount(board.occupied)
        if "phase" in r:
            phase.append(r["phase"])
        else:
            # dataset_v1 records carry no phase field; classify the same way the
            # C7 audit does so the two sets are described on one scale.
            ply = r.get("plies_played", 0)
            phase.append("endgame" if pieces <= 12
                         else "opening" if ply <= 20 else "middlegame")
        mated.append(bool(board.is_checkmate()))
    return {"side_to_move": np.array(stm), "phase": np.array(phase),
            "is_checkmate": np.array(mated)}


def metrics(y, p) -> dict:
    m = T.evaluate_predictions(y.astype(np.float64), p.astype(np.float64))
    return {"huber": m["huber_loss_delta1"], "mae": m["mae_centipawns"],
            "rmse": m["rmse_centipawns"], "pearson": m["pearson_r"],
            "n": m["n_predictions"]}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Score every arm on both held-out sets.")
    ap.add_argument("--arms", nargs="+", default=["A2", "C8a", "C8b"])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    import keras

    loaders = [("dataset_v1_test", load_v1_test),
               ("dataset_v2_test", load_v2_test)]
    if (ARTIFACTS / "dataset_v2_k6.test.jsonl").is_file():
        loaders.append(("dataset_v2_k6_test", load_v2_k6_test))

    sets = {}
    for name, loader in loaders:
        records = loader()
        y = D.apply_label_policy(records, POLICY)
        X = R.encode_many(r["fen"] for r in records)
        sets[name] = {"records": records, "y": y, "X": X, "meta": annotate(records)}
        print(f"{name}: {len(records)} records, X{X.shape}")

    out = {
        "stage": "C8a cross-evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_policy": POLICY,
        "note": ("A2 and C8a have different held-out sets; the off-diagonal cells "
                 "are the comparable ones. Engine metrics need no such correction."),
        "held_out_sets": {k: {"n": len(v["records"])} for k, v in sets.items()},
        "per_run": {},
        "by_arm": {},
    }

    print(f"\n{'run':12s}{'held-out set':18s}{'huber':>10s}{'MAE':>10s}"
          f"{'RMSE':>10s}{'Pearson':>10s}")
    collected = defaultdict(lambda: defaultdict(list))
    for arm in args.arms:
        for seed in SEEDS:
            path = EXP / arm / f"seed_{seed}" / "models" / "cnn_model.keras"
            if not path.is_file():
                print(f"  {arm} s{seed}: MISSING {path}")
                continue
            model = keras.models.load_model(path, compile=False)
            run_key = f"{arm}_seed_{seed}"
            out["per_run"][run_key] = {}
            for set_name, bundle in sets.items():
                p = np.asarray(model.predict(bundle["X"], verbose=0,
                                             batch_size=512)).reshape(-1)
                block = metrics(bundle["y"], p)

                meta = bundle["meta"]
                err = np.abs(bundle["y"].astype(np.float64) - p)
                block["mae_by_side"] = {
                    s: round(float(err[meta["side_to_move"] == s].mean()), 4)
                    for s in ("white", "black")
                    if (meta["side_to_move"] == s).any()}
                block["mae_by_phase"] = {
                    ph: round(float(err[meta["phase"] == ph].mean()), 4)
                    for ph in ("opening", "middlegame", "endgame")
                    if (meta["phase"] == ph).any()}
                if meta["is_checkmate"].any():
                    block["mae_checkmate"] = round(
                        float(err[meta["is_checkmate"]].mean()), 4)
                    block["mae_non_checkmate"] = round(
                        float(err[~meta["is_checkmate"]].mean()), 4)
                block["pred_abs_mean"] = round(float(np.abs(p).mean()), 4)
                block["pred_std"] = round(float(p.std()), 4)

                out["per_run"][run_key][set_name] = block
                for k in ("huber", "mae", "rmse", "pearson", "pred_abs_mean"):
                    collected[(arm, set_name)][k].append(block[k])
                print(f"  {run_key:10s}{set_name:18s}{block['huber']:>10.2f}"
                      f"{block['mae']:>10.2f}{block['rmse']:>10.2f}"
                      f"{block['pearson']:>10.4f}")

    print(f"\n{'arm':8s}{'held-out set':18s}{'huber':>10s}{'MAE':>10s}"
          f"{'RMSE':>10s}{'Pearson':>10s}   (3-seed means)")
    for (arm, set_name), vals in collected.items():
        mean = {k: round(float(np.mean(v)), 4) for k, v in vals.items()}
        out["by_arm"].setdefault(arm, {})[set_name] = {
            "mean": mean,
            "per_seed": {k: [round(x, 4) for x in v] for k, v in vals.items()},
        }
        print(f"  {arm:6s}{set_name:18s}{mean['huber']:>10.2f}{mean['mae']:>10.2f}"
              f"{mean['rmse']:>10.2f}{mean['pearson']:>10.4f}")

    print("\nTHE COMPARABLE READ: same held-out set, different training data")
    for set_name in sets:
        row = {a: out["by_arm"].get(a, {}).get(set_name, {}).get("mean")
               for a in args.arms}
        if all(row.values()):
            a, b = args.arms[0], args.arms[-1]
            print(f"  on {set_name}: {a} MAE {row[a]['mae']:.2f} vs "
                  f"{b} MAE {row[b]['mae']:.2f}  "
                  f"({row[b]['mae'] - row[a]['mae']:+.2f})   "
                  f"Pearson {row[a]['pearson']:.4f} vs {row[b]['pearson']:.4f} "
                  f"({row[b]['pearson'] - row[a]['pearson']:+.4f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    try:
        shown = args.out.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:          # an --out outside the repo, e.g. a scratch dir
        shown = str(args.out.resolve().as_posix())
    print(f"\nWROTE {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
