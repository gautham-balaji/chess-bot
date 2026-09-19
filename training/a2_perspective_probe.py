"""C6-A2 mechanism probe: is the label frame learnable from a 12-plane input?

    python -m training.a2_perspective_probe

Read-only. Writes training/experiments/A2/perspective_probe.json.

--------------------------------------------------------------------------
THE HYPOTHESIS
--------------------------------------------------------------------------
`training/representation.py` reports encodes_side_to_move = False: the 12-plane
tensor carries piece placement only. Under A0 and A1 the target is
side-to-move-relative, so for the 407 Black-to-move positions the label's SIGN
depends on a fact the input does not contain. The network is asked to output
two different values for inputs drawn from one indistinguishable distribution,
which is irreducible label noise rather than a learnable pattern.

A2 normalises the frame, so its target is a function of the position alone.

--------------------------------------------------------------------------
THE TEST (within-arm, so label scales are never compared across arms)
--------------------------------------------------------------------------
For each arm and seed, split the held-out test set by side to move and compare
the arm's OWN error on each group. Comparing an arm against itself sidesteps the
objection that A1 and A2 are fitted to different targets.

    prediction: A0 and A1 show a large Black-vs-White error gap
                A2 shows a much smaller one

A signed-error check is included because the failure mode is directional: if the
network cannot see whose turn it is, it should predict the White-frame value and
therefore be wrong by roughly -2 * label on Black-to-move rows.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

import chess
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from training import representation as R  # noqa: E402

EXP = REPO_ROOT / "training" / "experiments"
OUT = EXP / "A2" / "perspective_probe.json"
SEEDS = (0, 1, 2)
ARMS = ("A0", "A1", "A2")


def group_stats(err, signed):
    err, signed = np.asarray(err), np.asarray(signed)
    return {"n": int(err.size),
            "mae": round(float(np.abs(err).mean()), 4),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
            "mean_signed_error": round(float(signed.mean()), 4),
            "median_abs_error": round(float(np.median(np.abs(err))), 4)}


def main() -> int:
    rep = R.representation_summary() if hasattr(R, "representation_summary") else None
    if rep is None:
        for name in ("summary", "describe", "planes_summary"):
            if hasattr(R, name):
                rep = getattr(R, name)()
                break
    encodes_stm = bool(rep["encodes_side_to_move"]) if rep else None

    out = {"stage": "C6-A2 perspective probe",
           "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "representation": {"n_planes": rep["n_planes"] if rep else None,
                              "encodes_side_to_move": encodes_stm,
                              "encodes_castling_rights": rep["encodes_castling_rights"] if rep else None,
                              "encodes_en_passant": rep["encodes_en_passant"] if rep else None},
           "hypothesis": ("with encodes_side_to_move = False, a side-to-move-relative "
                          "target is not a function of the input; A2 removes that"),
           "per_run": {}, "summary": {}}

    print("=" * 92)
    print("C6-A2 PERSPECTIVE PROBE - held-out test error split by side to move")
    print("=" * 92)
    print(f"\nrepresentation: {rep['n_planes'] if rep else '?'} planes, "
          f"encodes_side_to_move={encodes_stm}, "
          f"castling={rep['encodes_castling_rights'] if rep else '?'}, "
          f"en_passant={rep['encodes_en_passant'] if rep else '?'}")

    print(f"\n{'run':8s}{'white n':>9s}{'white MAE':>11s}{'black n':>9s}{'black MAE':>11s}"
          f"{'gap':>10s}{'ratio':>8s}{'black signed err':>18s}")
    gaps = {a: [] for a in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            path = EXP / arm / f"seed_{seed}" / "test_predictions.json"
            if not path.is_file():
                print(f"  {arm} s{seed}: MISSING {path}")
                continue
            d = json.loads(path.read_text(encoding="utf-8"))
            y_true = np.asarray(d["y_true"], dtype=np.float64)
            y_pred = np.asarray(d["y_pred"], dtype=np.float64)
            is_white = np.array([chess.Board(f).turn == chess.WHITE for f in d["fens"]])

            err = y_pred - y_true
            w = group_stats(err[is_white], err[is_white])
            b = group_stats(err[~is_white], err[~is_white])
            gap = round(b["mae"] - w["mae"], 4)
            ratio = round(b["mae"] / w["mae"], 4) if w["mae"] else None
            gaps[arm].append(gap)

            # directional check: is the black-row error close to -2 * label?
            naive = -2.0 * y_true[~is_white]
            corr = float(np.corrcoef(err[~is_white], naive)[0, 1]) if (~is_white).sum() > 2 else None

            out["per_run"][f"{arm}_seed_{seed}"] = {
                "white": w, "black": b, "mae_gap_black_minus_white": gap,
                "mae_ratio_black_over_white": ratio,
                "corr_black_error_with_minus_2x_label": round(corr, 4) if corr is not None else None,
                "overall_mae": round(float(np.abs(err).mean()), 4),
            }
            print(f"  {arm} s{seed} {w['n']:>8d}{w['mae']:>11.2f}{b['n']:>9d}{b['mae']:>11.2f}"
                  f"{gap:>10.2f}{ratio:>8.2f}{b['mean_signed_error']:>18.2f}")

    # ------------------------------------------------------------------ why A1
    # Splitting the Black rows by eval_type shows what A1's mate repair did: it
    # moved 19 test rows from a target of 0 to a target of magnitude 2000 while
    # leaving the frame unlearnable, so the repair itself became the damage.
    from training import dataset as D

    eval_type = {r["fen"]: r["eval_type"]
                 for r in D.load_records(REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl")}
    print(f"\n{'-' * 92}\nBLACK-TO-MOVE TEST ROWS, SPLIT BY EVAL TYPE\n{'-' * 92}")
    print(f"{'run':8s}{'cp n':>7s}{'cp MAE':>10s}{'mate n':>9s}{'mate MAE':>11s}"
          f"{'mean |mate label|':>20s}")
    out["black_rows_by_eval_type"] = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = EXP / arm / f"seed_{seed}" / "test_predictions.json"
            if not path.is_file():
                continue
            d = json.loads(path.read_text(encoding="utf-8"))
            y_true = np.asarray(d["y_true"], dtype=np.float64)
            err = np.abs(np.asarray(d["y_pred"], dtype=np.float64) - y_true)
            blk = np.array([chess.Board(f).turn == chess.BLACK for f in d["fens"]])
            mate = np.array([eval_type[f] == "mate" for f in d["fens"]])
            b_cp, b_mate = blk & ~mate, blk & mate
            row = {
                "black_cp": {"n": int(b_cp.sum()), "mae": round(float(err[b_cp].mean()), 4)},
                "black_mate": {"n": int(b_mate.sum()), "mae": round(float(err[b_mate].mean()), 4),
                               "mean_abs_label": round(float(np.abs(y_true[b_mate]).mean()), 4)},
            }
            out["black_rows_by_eval_type"][f"{arm}_seed_{seed}"] = row
            print(f"  {arm} s{seed} {row['black_cp']['n']:>6d}{row['black_cp']['mae']:>10.2f}"
                  f"{row['black_mate']['n']:>9d}{row['black_mate']['mae']:>11.2f}"
                  f"{row['black_mate']['mean_abs_label']:>20.1f}")

    print(f"\n{'-' * 92}\nMEAN Black-minus-White MAE gap, across the three seeds\n{'-' * 92}")
    for arm in ARMS:
        if gaps[arm]:
            out["summary"][arm] = {"mae_gap_per_seed": gaps[arm],
                                   "mae_gap_mean": round(st.fmean(gaps[arm]), 4)}
            print(f"  {arm}: {[round(x, 1) for x in gaps[arm]]}   mean {st.fmean(gaps[arm]):+.2f}")

    if all(a in out["summary"] for a in ARMS):
        a1g, a2g = out["summary"]["A1"]["mae_gap_mean"], out["summary"]["A2"]["mae_gap_mean"]
        supported = a2g < a1g
        out["hypothesis_supported"] = bool(supported)
        out["verdict"] = (
            f"A2's Black-vs-White error gap is {a1g - a2g:.1f} cp SMALLER than A1's"
            if supported else
            f"NOT SUPPORTED: A2's gap ({a2g:.1f}) is not smaller than A1's ({a1g:.1f})")
        print(f"\n{out['verdict']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWROTE {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
