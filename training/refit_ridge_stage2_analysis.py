"""C6-A1R Stage 2 analysis - the 2x2 matched-fusion comparison.

    python -m training.refit_ridge_stage2_analysis

Reads (never writes) the production-fusion results recorded by A0/A1 and the
matched-fusion results produced by Stage 2, and answers the central question:

    Does the A0 -> A1 engine-level difference survive when BOTH arms use a Ridge
    matched to their own CNN and label policy?

Writes training/experiments/A1R/stage2_analysis.json and prints the tables.
Descriptive only - three seeds, no significance testing.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

A1R = REPO_ROOT / "training" / "experiments" / "A1R"
SEEDS = (0, 1, 2)
ARMS = ("A0", "A1")
SUITES = ("extended", "phase0_52")

METRICS = [
    ("legality_pct", "legality %", lambda m: m["legality_rate"]["percent"], 2),
    ("top1_pct", "top-1 agreement %", lambda m: m["top1_agreement"]["percent"], 2),
    ("top3_pct", "top-3 containment %", lambda m: m["top3_containment"]["percent"], 2),
    ("mean_regret", "mean regret cp", lambda m: m["move_regret_cp"]["mean"], 2),
    ("median_regret", "median regret cp", lambda m: m["move_regret_cp"]["median"], 2),
    ("p95_regret", "p95 regret cp", lambda m: m["move_regret_cp"]["p95"], 2),
    ("max_regret", "max regret cp", lambda m: m["move_regret_cp"]["max"], 2),
    ("blunder_rate", "blunder rate >300cp", lambda m: m["blunder_rate"]["rate"], 4),
    ("regret_coverage", "regret coverage %", lambda m: m["regret_coverage"]["percent"], 2),
    ("spearman_mean", "spearman mean", lambda m: m["spearman_rho_vs_stockfish_topk"]["mean"], 4),
    ("spearman_median", "spearman median", lambda m: m["spearman_rho_vs_stockfish_topk"]["median"], 4),
]


def production_path(arm, seed, suite):
    return REPO_ROOT / "training" / "experiments" / arm / f"seed_{seed}" / "evaluation" / f"{suite}.json"


def matched_path(arm, seed, suite):
    return A1R / f"{arm}_seed_{seed}" / f"{suite}.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def spread(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(st.fmean(vals), 4),
            "sd": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "range": round(max(vals) - min(vals), 4)}


def moves_of(result):
    return {r["id"]: r["engine_move"] for r in result["per_position"]}


def contain_of(result):
    return {r["id"]: r["top3_containment"] for r in result["per_position"]}


def main() -> int:
    data = {}
    for arm in ARMS:
        for seed in SEEDS:
            for suite in SUITES:
                data[(arm, seed, suite, "production")] = load(production_path(arm, seed, suite))
                data[(arm, seed, suite, "matched")] = load(matched_path(arm, seed, suite))

    out = {"note": "descriptive only; three seeds; no significance testing",
           "suites": {}}

    for suite in SUITES:
        n = data[("A0", 0, suite, "production")]["dataset"]["count"]
        print("=" * 104)
        print(f"SUITE {suite} (n={n})")
        print("=" * 104)
        suite_out = {"n": n, "per_run": {}, "paired": {}, "arm_deltas": {}}

        # ---------------- per-run values
        print(f"\n{'metric':22s}" + "".join(f"{a} s{s} {f[:4]}".rjust(14)
                                            for a in ARMS for s in SEEDS for f in ("prod", "match"))[:0]
              + "  (prod -> matched, per arm/seed)")
        for key, label, get, nd in METRICS:
            cells = []
            for arm in ARMS:
                for seed in SEEDS:
                    p = get(data[(arm, seed, suite, "production")]["metrics"])
                    m = get(data[(arm, seed, suite, "matched")]["metrics"])
                    cells.append(f"{p:.{nd}f}->{m:.{nd}f}".rjust(17))
                    suite_out["per_run"].setdefault(f"{arm}_seed_{seed}", {})[key] = {
                        "production": p, "matched": m, "delta": round(m - p, 6)}
            print(f"  {label:20s}" + "".join(cells))

        # ---------------- 1 & 2: production vs matched, per arm
        print(f"\n{'-'*104}\n1&2. PRODUCTION -> MATCHED fusion, paired per seed\n{'-'*104}")
        for arm in ARMS:
            print(f"  {arm}:")
            for key, label, get, nd in METRICS:
                d = [get(data[(arm, s, suite, "matched")]["metrics"])
                     - get(data[(arm, s, suite, "production")]["metrics"]) for s in SEEDS]
                same = "all 3 same direction" if (all(x > 0 for x in d) or all(x < 0 for x in d)) else ""
                print(f"    {label:20s} " + "".join(f"{x:+10.4g}" for x in d) +
                      f"   mean {st.fmean(d):+9.4g}   {same}")
                suite_out["arm_deltas"].setdefault(arm, {})[key] = {
                    "per_seed": [round(x, 6) for x in d], "mean": round(st.fmean(d), 6),
                    "consistent_direction": bool(all(x > 0 for x in d) or all(x < 0 for x in d))}

        # ---------------- 3 & 4: A0 -> A1 under each fusion
        print(f"\n{'-'*104}\n3&4. A0 -> A1 difference, under each fusion (paired by seed)\n{'-'*104}")
        print(f"    {'metric':20s}{'PRODUCTION fusion':>36s}{'MATCHED fusion':>36s}")
        for key, label, get, nd in METRICS:
            dp = [get(data[("A1", s, suite, "production")]["metrics"])
                  - get(data[("A0", s, suite, "production")]["metrics"]) for s in SEEDS]
            dm = [get(data[("A1", s, suite, "matched")]["metrics"])
                  - get(data[("A0", s, suite, "matched")]["metrics"]) for s in SEEDS]
            print(f"    {label:20s}" +
                  "".join(f"{x:+11.4g}" for x in dp) + f" (m {st.fmean(dp):+8.3g})" +
                  "".join(f"{x:+11.4g}" for x in dm) + f" (m {st.fmean(dm):+8.3g})")
            suite_out["paired"][key] = {
                "a0_to_a1_production": {"per_seed": [round(x, 6) for x in dp],
                                        "mean": round(st.fmean(dp), 6),
                                        "consistent": bool(all(x > 0 for x in dp) or all(x < 0 for x in dp))},
                "a0_to_a1_matched": {"per_seed": [round(x, 6) for x in dm],
                                     "mean": round(st.fmean(dm), 6),
                                     "consistent": bool(all(x > 0 for x in dm) or all(x < 0 for x in dm))},
                "effect_change": round(st.fmean(dm) - st.fmean(dp), 6),
            }

        # ---------------- 7: ranges
        print(f"\n{'-'*104}\n7. RANGES across seeds\n{'-'*104}")
        print(f"    {'metric':20s}{'A0 prod':>22s}{'A0 matched':>22s}{'A1 prod':>22s}{'A1 matched':>22s}")
        for key, label, get, nd in METRICS:
            cells = []
            for arm in ARMS:
                for fusion in ("production", "matched"):
                    v = [get(data[(arm, s, suite, fusion)]["metrics"]) for s in SEEDS]
                    sp = spread(v)
                    cells.append(f"[{sp['min']:8.4g}..{sp['max']:8.4g}]".rjust(22))
                    suite_out.setdefault("ranges", {}).setdefault(f"{arm}_{fusion}", {})[key] = sp
            # reorder to A0 prod, A0 matched, A1 prod, A1 matched
            print(f"    {label:20s}" + cells[0] + cells[1] + cells[2] + cells[3])

        # ---------------- 5 & 6: selected-move / containment changes
        print(f"\n{'-'*104}\n5&6. SELECTED-MOVE and TOP-3 CONTAINMENT changes (production -> matched)\n{'-'*104}")
        for arm in ARMS:
            for seed in SEEDS:
                p = data[(arm, seed, suite, "production")]
                m = data[(arm, seed, suite, "matched")]
                pm, mm = moves_of(p), moves_of(m)
                changed = [k for k in pm if pm[k] != mm.get(k)]
                pc, mc = contain_of(p), contain_of(m)
                gained = [k for k in pc if not pc[k] and mc.get(k)]
                lost = [k for k in pc if pc[k] and not mc.get(k)]
                print(f"    {arm} seed {seed}: moves changed {len(changed):3d}/{n}  "
                      f"top3 gained {len(gained):3d}  lost {len(lost):3d}")
                suite_out["per_run"][f"{arm}_seed_{seed}"]["selected_move_changes"] = len(changed)
                suite_out["per_run"][f"{arm}_seed_{seed}"]["top3_gained"] = len(gained)
                suite_out["per_run"][f"{arm}_seed_{seed}"]["top3_lost"] = len(lost)

        # ---------------- mate + breakdowns
        print(f"\n{'-'*104}\nMATE STATUSES\n{'-'*104}")
        for arm in ARMS:
            for seed in SEEDS:
                p = data[(arm, seed, suite, "production")]["metrics"]["mate_status_counts"]
                m = data[(arm, seed, suite, "matched")]["metrics"]["mate_status_counts"]
                print(f"    {arm} s{seed} prod {p}")
                print(f"    {arm} s{seed} match {m}")
                suite_out["per_run"][f"{arm}_seed_{seed}"]["mate_production"] = p
                suite_out["per_run"][f"{arm}_seed_{seed}"]["mate_matched"] = m

        for bd, title in (("by_side_to_move", "WHITE/BLACK"), ("by_category", "PHASE / CATEGORY")):
            print(f"\n{'-'*104}\n{title} BREAKDOWN (matched fusion)\n{'-'*104}")
            groups = sorted(data[("A0", 0, suite, "matched")]["breakdowns"][bd])
            print(f"    {'group':14s}" + "".join(f"{a} s{s}".rjust(11) for a in ARMS for s in SEEDS))
            for g in groups:
                cells = []
                for arm in ARMS:
                    for seed in SEEDS:
                        d = data[(arm, seed, suite, "matched")]["breakdowns"][bd][g]
                        cells.append(f"{d['regret_cp']['mean']}".rjust(11))
                        suite_out["per_run"][f"{arm}_seed_{seed}"].setdefault(bd, {})[g] = {
                            "n": d["n"],
                            "top1_pct": d["top1_agreement"]["percent"],
                            "top3_pct": d["top3_containment"]["percent"],
                            "mean_regret": d["regret_cp"]["mean"],
                            "median_regret": d["regret_cp"]["median"]}
                n_g = data[("A0", 0, suite, "matched")]["breakdowns"][bd][g]["n"]
                print(f"    {g:10s} n={n_g:<3d}" + "".join(cells) + "   (mean regret cp)")

        out["suites"][suite] = suite_out
        print()

    path = A1R / "stage2_analysis.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
