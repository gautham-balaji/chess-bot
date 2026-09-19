"""C6-A2 analysis: the paired A1 -> A2 contrast, read against the A0 noise band.

    python -m training.a2_analysis

Read-only over A0/A1/A2 evaluation output. Writes
training/experiments/A2/a2_analysis.json and prints the tables.

All three arms are evaluated with the FROZEN PRODUCTION Ridge, so the fusion is
held constant and the only varying input is the CNN's label policy. The A1R
matched Ridge is deliberately NOT used here.

Descriptive only: three seeds, no significance testing. The decision rule is the
pre-registered one - a change counts only if it exceeds the A0 seed-noise band.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

EXP = REPO_ROOT / "training" / "experiments"
OUT = EXP / "A2" / "a2_analysis.json"
SEEDS = (0, 1, 2)
ARMS = ("A0", "A1", "A2")
SUITES = ("extended", "phase0_52")

METRICS = [
    ("mean_regret", "mean regret cp", lambda m: m["move_regret_cp"]["mean"], True),
    ("median_regret", "median regret cp", lambda m: m["move_regret_cp"]["median"], True),
    ("p95_regret", "p95 regret cp", lambda m: m["move_regret_cp"]["p95"], True),
    ("max_regret", "max regret cp", lambda m: m["move_regret_cp"]["max"], True),
    ("top1_pct", "top-1 agreement %", lambda m: m["top1_agreement"]["percent"], False),
    ("top3_pct", "top-3 containment %", lambda m: m["top3_containment"]["percent"], False),
    ("blunder_rate", "blunder rate >300cp", lambda m: m["blunder_rate"]["rate"], True),
    ("legality_pct", "legality %", lambda m: m["legality_rate"]["percent"], False),
    ("spearman_mean", "spearman mean", lambda m: m["spearman_rho_vs_stockfish_topk"]["mean"], False),
    ("regret_coverage", "regret coverage %", lambda m: m["regret_coverage"]["percent"], False),
]


def path_for(arm, seed, suite):
    return EXP / arm / f"seed_{seed}" / "evaluation" / f"{suite}.json"


def load(arm, seed, suite):
    return json.loads(path_for(arm, seed, suite).read_text(encoding="utf-8"))


def band(vals):
    return {"min": round(min(vals), 4), "max": round(max(vals), 4),
            "mean": round(st.fmean(vals), 4),
            "range": round(max(vals) - min(vals), 4),
            "sd": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0}


def regret_of(result, breakdown, group):
    return result["breakdowns"][breakdown][group]["regret_cp"]["mean"]


def breakdown_block(data, suite, breakdown, title, note=""):
    """Per-arm means plus the paired A1 -> A2 delta, for one breakdown."""
    print(f"\n{'-' * 100}\n{title}{note}\n{'-' * 100}")
    groups = sorted(data[("A0", 0, suite)]["breakdowns"][breakdown])
    header = "".join(f"{a} s{s}".rjust(11) for a in ARMS for s in SEEDS)
    print(f"  {'group':12s}{header}   A1->A2 mean")

    block = {}
    for g in groups:
        cells = [f"{regret_of(data[(a, s, suite)], breakdown, g):11.4g}"
                 for a in ARMS for s in SEEDS]
        deltas = [regret_of(data[("A2", s, suite)], breakdown, g)
                  - regret_of(data[("A1", s, suite)], breakdown, g) for s in SEEDS]
        n_g = data[("A0", 0, suite)]["breakdowns"][breakdown][g]["n"]
        print(f"  {g:8s} n={n_g:<3d}" + "".join(cells) + f"   {st.fmean(deltas):+10.4g}")
        block[g] = {
            "n": n_g,
            "per_arm": {f"{a}_seed_{s}": regret_of(data[(a, s, suite)], breakdown, g)
                        for a in ARMS for s in SEEDS},
            "a1_to_a2_mean_regret_delta": round(st.fmean(deltas), 4),
            "a1_to_a2_per_seed": [round(x, 4) for x in deltas],
            "consistent": bool(all(x > 0 for x in deltas) or all(x < 0 for x in deltas)),
        }
    return block


def main() -> int:
    missing = [str(path_for(a, s, q).relative_to(REPO_ROOT))
               for a in ARMS for s in SEEDS for q in SUITES
               if not path_for(a, s, q).is_file()]
    if missing:
        raise SystemExit("ERROR: missing evaluation output:\n  " + "\n  ".join(missing))

    data = {(a, s, q): load(a, s, q) for a in ARMS for s in SEEDS for q in SUITES}
    out = {"stage": "C6-A2 analysis",
           "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "fusion": "frozen PRODUCTION Ridge for all arms (A1R matched Ridge NOT used)",
           "note": "descriptive only; three seeds; no significance testing",
           "suites": {}}

    for suite in SUITES:
        n = data[("A0", 0, suite)]["dataset"]["count"]
        print("=" * 100)
        print(f"SUITE {suite} (n={n})")
        print("=" * 100)
        so = {"n": n, "per_seed": {}, "bands": {}, "a0_noise_band": {},
              "a1_to_a2": {}, "a0_to_a2": {}}

        # ------------------------------------------------ per-seed values
        print(f"\n{'metric':22s}" + "".join(f"{a} s{s}".rjust(11) for a in ARMS for s in SEEDS))
        for key, label, get, _ in METRICS:
            cells = []
            for a in ARMS:
                for s in SEEDS:
                    v = get(data[(a, s, suite)]["metrics"])
                    cells.append(f"{v:11.4g}")
                    so["per_seed"].setdefault(f"{a}_seed_{s}", {})[key] = v
            print(f"  {label:20s}" + "".join(cells))

        # ------------------------------------------------ A0 noise band
        print(f"\n{'-' * 100}\nA0 NOISE BAND (seed-only variation) vs A1 and A2 spreads\n{'-' * 100}")
        print(f"  {'metric':20s}{'A0 band':>30s}{'A1 band':>30s}{'A2 band':>30s}")
        for key, label, get, _ in METRICS:
            row = []
            for a in ARMS:
                b = band([get(data[(a, s, suite)]["metrics"]) for s in SEEDS])
                row.append(f"[{b['min']:9.4g} .. {b['max']:9.4g}]".rjust(30))
                so["bands"].setdefault(a, {})[key] = b
            so["a0_noise_band"][key] = so["bands"]["A0"][key]
            print(f"  {label:20s}" + "".join(row))

        # ------------------------------------------------ paired contrasts
        for src, dst, slot in (("A1", "A2", "a1_to_a2"), ("A0", "A2", "a0_to_a2")):
            print(f"\n{'-' * 100}\n{src} -> {dst}, PAIRED BY SEED (production fusion)\n{'-' * 100}")
            for key, label, get, lower_better in METRICS:
                d = [get(data[(dst, s, suite)]["metrics"]) - get(data[(src, s, suite)]["metrics"])
                     for s in SEEDS]
                mean_d = st.fmean(d)
                consistent = all(x > 0 for x in d) or all(x < 0 for x in d)
                noise_range = so["a0_noise_band"][key]["range"]
                exceeds = abs(mean_d) > noise_range
                if mean_d == 0:
                    direction = "unchanged"
                else:
                    direction = "better" if ((mean_d < 0) == lower_better) else "worse"
                flag = "EXCEEDS A0 noise" if exceeds else "within A0 noise"
                verdict = direction if consistent else "inconsistent"
                print(f"    {label:20s}" + "".join(f"{x:+11.4g}" for x in d)
                      + f"   mean {mean_d:+10.4g}  A0 range {noise_range:9.4g}  "
                        f"{flag:16s} {verdict}")
                so[slot][key] = {
                    "per_seed": [round(x, 6) for x in d], "mean": round(mean_d, 6),
                    "consistent_direction": bool(consistent),
                    "a0_noise_range": noise_range,
                    "exceeds_a0_noise": bool(exceeds),
                    "direction": direction,
                    "lower_is_better": lower_better,
                }

        # ------------------------------------------------ breakdowns
        so["by_side_to_move"] = breakdown_block(
            data, suite, "by_side_to_move", "BY SIDE TO MOVE - mean regret cp",
            "  (the 4.2% of labels A2 changed are all Black-to-move)")
        so["by_category"] = breakdown_block(
            data, suite, "by_category", "BY CATEGORY - mean regret cp")

        # ------------------------------------------------ mate behaviour
        print(f"\n{'-' * 100}\nMATE STATUS COUNTS\n{'-' * 100}")
        so["mate_status"] = {}
        for a in ARMS:
            for s in SEEDS:
                mc = data[(a, s, suite)]["metrics"]["mate_status_counts"]
                print(f"  {a} s{s}: {mc}")
                so["mate_status"][f"{a}_seed_{s}"] = mc

        # ------------------------------------------------ move agreement
        print(f"\n{'-' * 100}\nSELECTED-MOVE CHANGES A1 -> A2 (paired by seed)\n{'-' * 100}")
        so["selected_move_changes"] = {}
        for s in SEEDS:
            rows1 = data[("A1", s, suite)]["per_position"]
            m1 = {r["id"]: r["engine_move"] for r in rows1}
            m2 = {r["id"]: r["engine_move"] for r in data[("A2", s, suite)]["per_position"]}
            stm = {r["id"]: r["side_to_move"] for r in rows1}
            changed = [k for k in m1 if m1[k] != m2.get(k)]
            by_stm = dict(Counter(stm[k] for k in changed))
            print(f"  seed {s}: {len(changed):3d}/{n} moves changed   by side to move {by_stm}")
            so["selected_move_changes"][f"seed_{s}"] = {
                "changed": len(changed), "n": n, "by_side_to_move": by_stm}

        out["suites"][suite] = so
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
