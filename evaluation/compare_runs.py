"""Compare two evaluation runs produced by evaluate.py.

Used in Phase 4A to measure the effect of the C1 fix. Reads two result JSON
files and reports metric deltas plus a per-position move-change breakdown,
split by side to move.

Neither input file is modified.

Usage:
    python evaluation/compare_runs.py BEFORE.json AFTER.json [--label-before X --label-after Y]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(value, suffix=""):
    return "n/a" if value is None else f"{value}{suffix}"


def delta(before, after, invert=False):
    """Signed change with a direction marker. `invert=True` means lower is better."""
    if before is None or after is None:
        return "n/a"
    d = after - before
    if abs(d) < 1e-9:
        return "0 (no change)"
    better = (d < 0) if invert else (d > 0)
    return f"{d:+.4g} ({'better' if better else 'worse'})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--label-before", default="before")
    ap.add_argument("--label-after", default="after")
    args = ap.parse_args()

    b, a = load(args.before), load(args.after)
    mb, ma = b["metrics"], a["metrics"]
    lb, la = args.label_before, args.label_after

    print("=" * 78)
    print(f"{b['dataset']['name']}  n={b['dataset']['count']}   {lb} -> {la}")
    print("=" * 78)

    rows = [
        ("legality rate %", mb["legality_rate"]["percent"], ma["legality_rate"]["percent"], False),
        ("top-1 agreement %", mb["top1_agreement"]["percent"], ma["top1_agreement"]["percent"], False),
        ("top-3 containment %", mb["top3_containment"]["percent"], ma["top3_containment"]["percent"], False),
        ("mean regret cp", mb["move_regret_cp"]["mean"], ma["move_regret_cp"]["mean"], True),
        ("median regret cp", mb["move_regret_cp"]["median"], ma["move_regret_cp"]["median"], True),
        ("p95 regret cp", mb["move_regret_cp"]["p95"], ma["move_regret_cp"]["p95"], True),
        ("max regret cp", mb["move_regret_cp"]["max"], ma["move_regret_cp"]["max"], True),
        ("blunder rate", mb["blunder_rate"]["rate"], ma["blunder_rate"]["rate"], True),
        ("regret coverage %", mb["regret_coverage"]["percent"], ma["regret_coverage"]["percent"], False),
        ("spearman mean", mb["spearman_rho_vs_stockfish_topk"]["mean"],
         ma["spearman_rho_vs_stockfish_topk"]["mean"], False),
        ("spearman median", mb["spearman_rho_vs_stockfish_topk"]["median"],
         ma["spearman_rho_vs_stockfish_topk"]["median"], False),
    ]
    print(f"\n{'metric':24s} {lb:>14s} {la:>14s}   change")
    print("-" * 78)
    for name, x, y, inv in rows:
        print(f"{name:24s} {fmt(x):>14s} {fmt(y):>14s}   {delta(x, y, inv)}")

    print(f"\n{'counts':24s} {lb:>14s} {la:>14s}")
    print("-" * 78)
    for label, key in [("top-1 agreeing", "top1_agreement"),
                       ("top-3 containing", "top3_containment"),
                       ("legal moves", "legality_rate")]:
        print(f"{label:24s} {mb[key]['numerator']:>7d}/{mb[key]['denominator']:<6d} "
              f"{ma[key]['numerator']:>7d}/{ma[key]['denominator']:<6d}")
    print(f"{'blunders':24s} {mb['blunder_rate']['blunders']:>7d}/"
          f"{mb['blunder_rate']['denominator']:<6d} "
          f"{ma['blunder_rate']['blunders']:>7d}/{ma['blunder_rate']['denominator']:<6d}")

    print("\nmate statuses")
    print("-" * 78)
    keys = sorted(set(mb["mate_status_counts"]) | set(ma["mate_status_counts"]))
    for k in keys:
        print(f"{k:40s} {mb['mate_status_counts'].get(k, 0):>6d} "
              f"{ma['mate_status_counts'].get(k, 0):>6d}")

    # ---------------------------------------------------------------- per position
    bp = {r["id"]: r for r in b["per_position"]}
    ap_ = {r["id"]: r for r in a["per_position"]}
    changed, regret_deltas = [], []
    by_side = Counter()
    for pid, rb in bp.items():
        ra = ap_.get(pid)
        if ra is None:
            continue
        if rb["engine_move"] != ra["engine_move"]:
            changed.append((pid, rb, ra))
            by_side[rb["side_to_move"]] += 1
            if rb["move_regret_cp"] is not None and ra["move_regret_cp"] is not None:
                regret_deltas.append(ra["move_regret_cp"] - rb["move_regret_cp"])

    total_by_side = Counter(r["side_to_move"] for r in bp.values())
    print(f"\nper-position engine move changes: {len(changed)}/{len(bp)}")
    print("-" * 78)
    for side in sorted(total_by_side):
        print(f"  {side:6s}: {by_side[side]:3d} changed / {total_by_side[side]:3d} positions")

    if changed:
        print(f"\n{'id':8s} {'side':6s} {'before':7s} {'after':7s} {'regret b':>9s} "
              f"{'regret a':>9s} {'delta':>9s}")
        print("-" * 78)
        for pid, rb, ra in sorted(changed):
            rbv = rb["move_regret_cp"]
            rav = ra["move_regret_cp"]
            d = (rav - rbv) if (rbv is not None and rav is not None) else None
            print(f"{pid:8s} {rb['side_to_move']:6s} {rb['engine_move']:7s} "
                  f"{ra['engine_move']:7s} {fmt(rbv):>9s} {fmt(rav):>9s} {fmt(d):>9s}")

        if regret_deltas:
            improved = sum(1 for d in regret_deltas if d < 0)
            worsened = sum(1 for d in regret_deltas if d > 0)
            same = sum(1 for d in regret_deltas if d == 0)
            print(f"\n  of the changed positions with defined regret on both sides "
                  f"(n={len(regret_deltas)}): {improved} improved, {worsened} worsened, "
                  f"{same} unchanged")
            print(f"  net regret change across changed positions: "
                  f"{sum(regret_deltas):+.1f} cp "
                  f"(mean {sum(regret_deltas) / len(regret_deltas):+.1f} cp)")

    # ---------------------------------------------------------------- by side
    print("\nmetrics by side to move")
    print("-" * 78)
    print(f"{'side':6s} {'n':>4s} {'top1 b':>8s} {'top1 a':>8s} "
          f"{'meanR b':>9s} {'meanR a':>9s} {'medR b':>8s} {'medR a':>8s}")
    for side in sorted(b["breakdowns"]["by_side_to_move"]):
        db = b["breakdowns"]["by_side_to_move"][side]
        da = a["breakdowns"]["by_side_to_move"].get(side, {})
        print(f"{side:6s} {db['n']:>4d} "
              f"{fmt(db['top1_agreement']['percent']):>8s} "
              f"{fmt(da.get('top1_agreement', {}).get('percent')):>8s} "
              f"{fmt(db['regret_cp']['mean']):>9s} "
              f"{fmt(da.get('regret_cp', {}).get('mean')):>9s} "
              f"{fmt(db['regret_cp']['median']):>8s} "
              f"{fmt(da.get('regret_cp', {}).get('median')):>8s}")

    print("\nNOTE: latency is deliberately excluded - Phase 3 established it is "
          "host-dependent\n      and does not reproduce across runs.")


if __name__ == "__main__":
    main()
