"""C6-A2R Stage 1 analysis - does A2 pose a DIFFERENT Ridge compatibility problem?

    python -m training.a2r_stage1_analysis

Read-only over training/experiments/A2R/stage1_results.json (produced by
`training/refit_ridge.py`). Writes training/experiments/A2R/stage1_analysis.json
and prints the tables. No model is loaded, no engine is run, nothing is fitted.

--------------------------------------------------------------------------
TWO GATES, NOT ONE
--------------------------------------------------------------------------
A1R asked one question: is the refit fusion materially different from the
PRODUCTION fusion? That gate tripped, Stage 2 ran, and it answered the question
Stage 2 existed to answer - matched fusion did not explain A1's regression.

So re-deciding the A1R gate for A2 is not enough. A2R must also ask whether A2
differs from the arms already measured. Two gates:

  GATE 1 (absolute, identical to A1R)
      Is A2's refit close to a positive scalar multiple of production, AND are
      its ranking shares inside the A0 production-coefficient seed band?

  GATE 2 (relative, specific to A2R)
      Does A2 sit outside the envelope already established by the A0 and A1
      refits? If A2 behaves like an arm whose matched-fusion question Stage 2
      has already answered, another Stage 2 buys nothing.

Stage 2 for A2 is warranted only if GATE 2 also indicates A2 is a genuinely new
case. A tripped GATE 1 alone merely reconfirms what A1R already established.
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

from training import refit_ridge as RR  # noqa: E402

A2R = REPO_ROOT / "training" / "experiments" / "A2R"
RESULTS = A2R / "stage1_results.json"
A1R_RESULTS = REPO_ROOT / "training" / "experiments" / "A1R" / "stage1_results.json"
OUT = A2R / "stage1_analysis.json"

ARMS = ("A0", "A1", "A2")
SEEDS = (0, 1, 2)
FEATURES = RR.FEATURE_NAMES

# The A1R gate reference band: A0's CNN ranking share under PRODUCTION
# coefficients, i.e. what the share does from training seed alone.
GATE_BAND_LOW, GATE_BAND_HIGH = 82.85, 84.93


def rows_by(results, arm):
    return [r for r in results if r["arm"] == arm]


def spread(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(st.fmean(vals), 6),
            "sd": round(st.stdev(vals), 6) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 6), "max": round(max(vals), 6),
            "range": round(max(vals) - min(vals), 6)}


def overlaps(a, b) -> bool:
    """Do two [min, max] bands overlap at all?"""
    return not (a["max"] < b["min"] or b["max"] < a["min"])


def main() -> int:
    if not RESULTS.is_file():
        raise SystemExit(
            f"ERROR: {RESULTS} not found. Run:\n"
            f"  python -m training.refit_ridge --arms A0 A1 A2 "
            f"--out-dir training/experiments/A2R --stage A2R-stage1")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    results = data["results"]
    w_prod = np.asarray(data["production_coef"], dtype=float)
    prod_ratios = RR.normalised_ratios(w_prod)

    out = {"stage": "C6-A2R Stage 1 analysis",
           "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": str(RESULTS.relative_to(REPO_ROOT).as_posix()),
           "production_coef": [float(x) for x in w_prod],
           "production_intercept": data["production_intercept"],
           "note": "descriptive; three seeds per arm; no significance testing"}

    # ------------------------------------------------------------ reproduction
    # The A0/A1 rows here were refitted from scratch. They must reproduce the
    # committed A1R Stage 1 numbers, or something has drifted.
    if A1R_RESULTS.is_file():
        a1r = {(r["arm"], r["seed"]): r["coef"]
               for r in json.loads(A1R_RESULTS.read_text(encoding="utf-8"))["results"]}
        diffs = []
        for r in results:
            key = (r["arm"], r["seed"])
            if key in a1r:
                d = float(np.max(np.abs(np.asarray(r["coef"]) - np.asarray(a1r[key]))))
                diffs.append((f"{key[0]} s{key[1]}", d))
        worst = max((d for _, d in diffs), default=None)
        out["a1r_reproduction"] = {
            "n_compared": len(diffs),
            "max_abs_coefficient_difference": worst,
            "reproduces_a1r": bool(worst is not None and worst < 1e-6),
            "per_run": {k: v for k, v in diffs},
        }
        print("=" * 104)
        print("REPRODUCTION CHECK vs committed A1R Stage 1")
        print("=" * 104)
        status = "EXACT" if out["a1r_reproduction"]["reproduces_a1r"] else "DRIFT"
        print(f"  {len(diffs)} A0/A1 runs re-fitted; max |Δcoef| = {worst:.3e}  -> {status}\n")

    # ------------------------------------------------------------ 1. coefficients
    print("=" * 104)
    print("1. COEFFICIENT VECTORS  (absolute value, and sign)")
    print("=" * 104)
    hdr = f"  {'model':14s}" + "".join(f"{f:>13s}" for f in FEATURES) + f"{'intercept':>12s}"
    print(hdr)
    print(f"  {'production':14s}" + "".join(f"{v:13.4f}" for v in w_prod)
          + f"{data['production_intercept']:12.4f}")
    out["coefficients"] = {"production": [float(x) for x in w_prod]}
    for r in results:
        name = f"{r['arm']} seed {r['seed']}"
        print(f"  {name:14s}" + "".join(f"{v:13.4f}" for v in r["coef"])
              + f"{r['intercept']:12.4f}")
        out["coefficients"][f"{r['arm']}_seed_{r['seed']}"] = r["coef"]

    # cross-seed range per arm, per coefficient
    print(f"\n  cross-seed range per arm (max - min):")
    print(f"  {'arm':14s}" + "".join(f"{f:>13s}" for f in FEATURES))
    out["coefficient_cross_seed_range"] = {}
    for arm in ARMS:
        rs = rows_by(results, arm)
        rngs = [spread([r["coef"][i] for r in rs])["range"] for i in range(5)]
        print(f"  {arm:14s}" + "".join(f"{v:13.4f}" for v in rngs))
        out["coefficient_cross_seed_range"][arm] = {
            FEATURES[i]: spread([r["coef"][i] for r in rs]) for i in range(5)}

    # ------------------------------------------------------------ 2. ratios
    print("\n" + "=" * 104)
    print("2. RATIOS NORMALISED TO cnn_norm  (what can reorder moves)")
    print("=" * 104)
    print(f"  {'model':14s}" + "".join(f"{f:>13s}" for f in FEATURES))
    print(f"  {'production':14s}" + "".join(f"{v:13.6f}" for v in prod_ratios))
    out["ratios_normalised_to_cnn"] = {"production": prod_ratios}
    for r in results:
        print(f"  {r['arm']} seed {r['seed']:<5d}"
              + "".join(f"{v:13.6f}" for v in r["ratios_normalised_to_cnn"]))
        out["ratios_normalised_to_cnn"][f"{r['arm']}_seed_{r['seed']}"] = \
            r["ratios_normalised_to_cnn"]

    print(f"\n  multiple of the production ratio (sign preserved):")
    print(f"  {'model':14s}" + "".join(f"{f:>13s}" for f in FEATURES[1:]))
    out["ratio_multiple_of_production"] = {}
    for r in results:
        mult = [r["ratios_normalised_to_cnn"][i] / prod_ratios[i] for i in range(1, 5)]
        print(f"  {r['arm']} seed {r['seed']:<5d}" + "".join(f"{v:13.2f}" for v in mult))
        out["ratio_multiple_of_production"][f"{r['arm']}_seed_{r['seed']}"] = \
            [round(x, 4) for x in mult]

    # sign agreement
    out["sign_flips_vs_production"] = {}
    for r in results:
        flips = [FEATURES[i] for i in range(5)
                 if np.sign(r["coef"][i]) != np.sign(w_prod[i])]
        out["sign_flips_vs_production"][f"{r['arm']}_seed_{r['seed']}"] = flips

    # ------------------------------------------------------------ 3. scalars
    print("\n" + "=" * 104)
    print("3. COSINE, SCALE, R-SQUARED, RANKING SHARE, DECISIVENESS")
    print("=" * 104)
    print(f"  {'model':14s}{'cosine':>10s}{'scale':>8s}{'R2 tr':>8s}{'R2 hold':>9s}"
          f"{'share prod':>12s}{'share refit':>13s}{'shift':>9s}"
          f"{'decisive':>10s}{'topmove chg':>13s}")
    out["per_run"] = {}
    for r in results:
        key = f"{r['arm']}_seed_{r['seed']}"
        sp, sr = r["ranking_spread_production_coef"], r["ranking_spread_refit_coef"]
        dp, dr = r["decisive_production_coef"], r["decisive_refit_coef"]
        tm = r["top_move_agreement_production_vs_refit"]
        print(f"  {r['arm']} seed {r['seed']:<5d}"
              f"{r['cosine_similarity_to_production']:10.6f}"
              f"{r['scale_factor_vs_production']:8.4f}"
              f"{r['inner_split']['r2_train']:8.4f}{r['inner_split']['r2_holdout']:9.4f}"
              f"{sp['cnn_share_pct']:11.2f}%{sr['cnn_share_pct']:12.2f}%"
              f"{r['cnn_share_shift_pct_points']:9.2f}"
              f"{dr['cnn_decisive_pct']:9.1f}%{tm['changed_top_move_pct']:12.1f}%")
        out["per_run"][key] = {
            "cosine_to_production": r["cosine_similarity_to_production"],
            "scale_factor_vs_production": r["scale_factor_vs_production"],
            "r2_train": r["inner_split"]["r2_train"],
            "r2_holdout": r["inner_split"]["r2_holdout"],
            "cnn_share_production_coef": sp["cnn_share_pct"],
            "cnn_share_refit_coef": sr["cnn_share_pct"],
            "cnn_share_shift": r["cnn_share_shift_pct_points"],
            "cnn_decisive_production_coef": dp["cnn_decisive_pct"],
            "cnn_decisive_refit_coef": dr["cnn_decisive_pct"],
            "cnn_dictates_production_coef": dp["cnn_dictates_top_move_pct"],
            "cnn_dictates_refit_coef": dr["cnn_dictates_top_move_pct"],
            "top_move_changed_pct": tm["changed_top_move_pct"],
            "median_term_sd_refit": sr["median_term_sd"],
        }

    print("\n  cosine perturbation sensitivity of the production vector "
          "(why cosine cannot carry the gate):")
    for k, v in data.get("cosine_perturbation_sensitivity_of_production", {}).items():
        if k.endswith(("_x10", "_x100")) or k in ("uniform_x10", "space_sign_flip"):
            print(f"    {k:22s} {v:.6f}")
    out["cosine_perturbation_sensitivity"] = \
        data.get("cosine_perturbation_sensitivity_of_production", {})

    # ------------------------------------------------------------ 4. per-arm bands
    print("\n" + "=" * 104)
    print("4. PER-ARM BANDS  (three seeds)")
    print("=" * 104)
    quantities = {
        "cnn_share_refit_coef": "refit CNN ranking share %",
        "cnn_share_production_coef": "prod-coef CNN ranking share %",
        "scale_factor_vs_production": "scale factor w0/prod",
        "r2_holdout": "R2 holdout",
        "cnn_decisive_refit_coef": "CNN decisive % (refit)",
        "cnn_decisive_production_coef": "CNN decisive % (prod)",
        "top_move_changed_pct": "top move changed by refit %",
    }
    out["bands"] = {}
    print(f"  {'quantity':32s}" + "".join(f"{a:>26s}" for a in ARMS))
    for key, label in quantities.items():
        cells = []
        for arm in ARMS:
            b = spread([out["per_run"][f"{arm}_seed_{s}"][key] for s in SEEDS])
            out["bands"].setdefault(arm, {})[key] = b
            cells.append(f"[{b['min']:9.4g} .. {b['max']:9.4g}]".rjust(26))
        print(f"  {label:32s}" + "".join(cells))

    # ratio bands
    print(f"\n  normalised-ratio bands per arm:")
    for i, f in enumerate(FEATURES[1:], start=1):
        cells = []
        for arm in ARMS:
            b = spread([r["ratios_normalised_to_cnn"][i] for r in rows_by(results, arm)])
            out["bands"].setdefault(arm, {}).setdefault("ratios", {})[f] = b
            cells.append(f"[{b['min']:9.5f} .. {b['max']:9.5f}]".rjust(26))
        print(f"  {f:32s}" + "".join(cells) + f"   prod {prod_ratios[i]:+.6f}")

    # ------------------------------------------------------------ GATE 1
    print("\n" + "=" * 104)
    print("GATE 1 (absolute, identical to A1R): is A2's refit ~ a positive scalar "
          "multiple of production,")
    print("                                     and are its shares inside the A0 "
          f"band {GATE_BAND_LOW}-{GATE_BAND_HIGH}%?")
    print("=" * 104)
    a2_rows = rows_by(results, "A2")
    a2_flips = {f"A2_seed_{r['seed']}": out["sign_flips_vs_production"][f"A2_seed_{r['seed']}"]
                for r in a2_rows}
    any_flip = any(a2_flips.values())
    max_ratio_mult = max(
        abs(out["ratio_multiple_of_production"][f"A2_seed_{r['seed']}"][i])
        for r in a2_rows for i in range(4))
    a2_share_band = out["bands"]["A2"]["cnn_share_refit_coef"]
    shares_in_band = (a2_share_band["min"] >= GATE_BAND_LOW
                      and a2_share_band["max"] <= GATE_BAND_HIGH)
    near_proportional = (not any_flip) and max_ratio_mult < 2.0

    print(f"  sign flips vs production        : {a2_flips}")
    print(f"  max |ratio multiple|            : {max_ratio_mult:.1f}x  "
          f"(near-proportional needs < 2x)")
    print(f"  A2 refit CNN share band         : "
          f"{a2_share_band['min']:.2f}% .. {a2_share_band['max']:.2f}%")
    print(f"  condition A near-proportional?  : {'YES' if near_proportional else 'NO'}")
    print(f"  condition B shares within band? : {'YES' if shares_in_band else 'NO'}")
    gate1_tripped = not (near_proportional and shares_in_band)
    print(f"\n  GATE 1: {'TRIPPED' if gate1_tripped else 'CLEAR'}")

    out["gate_1_absolute"] = {
        "sign_flips": a2_flips,
        "max_abs_ratio_multiple": round(max_ratio_mult, 4),
        "a2_refit_cnn_share_band": a2_share_band,
        "reference_band": [GATE_BAND_LOW, GATE_BAND_HIGH],
        "near_proportional_to_production": bool(near_proportional),
        "shares_within_a0_band": bool(shares_in_band),
        "tripped": bool(gate1_tripped),
    }

    # ------------------------------------------------------------ GATE 2
    print("\n" + "=" * 104)
    print("GATE 2 (relative, A2R-specific): is A2 outside the envelope A0 and A1 "
          "already established?")
    print("=" * 104)
    comparisons, distinct = {}, {}
    for key, label in quantities.items():
        a2b = out["bands"]["A2"][key]
        a0b, a1b = out["bands"]["A0"][key], out["bands"]["A1"][key]
        ov0, ov1 = overlaps(a2b, a0b), overlaps(a2b, a1b)
        comparisons[key] = {
            "A0": a0b, "A1": a1b, "A2": a2b,
            "overlaps_A0": bool(ov0), "overlaps_A1": bool(ov1),
            "distinct_from_both": bool(not ov0 and not ov1),
        }
        distinct[key] = not ov0 and not ov1
        flag = "DISTINCT from both" if distinct[key] else (
            "overlaps " + ", ".join(x for x, o in (("A0", ov0), ("A1", ov1)) if o))
        print(f"  {label:32s} A2 [{a2b['min']:8.4g}..{a2b['max']:8.4g}]   {flag}")
    out["gate_2_relative"] = {"comparisons": comparisons}

    # Which of these actually bear on RANKING, rather than on fit quality?
    ranking_keys = ["cnn_share_refit_coef", "cnn_decisive_refit_coef",
                    "top_move_changed_pct"]
    ranking_distinct = [k for k in ranking_keys if distinct[k]]
    gate2_tripped = len(ranking_distinct) > 0
    print(f"\n  ranking-relevant quantities distinct from BOTH A0 and A1: "
          f"{ranking_distinct or 'none'}")
    print(f"\n  GATE 2: {'TRIPPED' if gate2_tripped else 'CLEAR'}")
    out["gate_2_relative"].update({
        "ranking_relevant_quantities": ranking_keys,
        "distinct_from_both_arms": ranking_distinct,
        "tripped": bool(gate2_tripped),
    })

    # ------------------------------------------------------------ decision
    print("\n" + "=" * 104)
    print("DECISION")
    print("=" * 104)
    warranted = gate1_tripped and gate2_tripped
    out["stage2_warranted"] = bool(warranted)
    if warranted:
        verdict = ("Stage 2 WARRANTED: A2's fusion is materially different from "
                   "production AND from the A0/A1 refits already measured, on at "
                   f"least one ranking-relevant quantity ({', '.join(ranking_distinct)}).")
    elif gate1_tripped:
        verdict = ("Stage 2 NOT warranted: GATE 1 trips, but that only reconfirms "
                   "what A1R Stage 1 already established for every arm. A2 sits "
                   "inside the A0/A1 envelope on every ranking-relevant quantity, "
                   "so an A2 matched-Ridge run would re-answer a question A1R "
                   "Stage 2 has already answered.")
    else:
        verdict = ("Stage 2 NOT warranted: A2's refit is near-proportional to "
                   "production and its ranking shares sit inside the A0 seed band.")
    out["verdict"] = verdict
    print(f"  GATE 1 (vs production) : {'TRIPPED' if gate1_tripped else 'CLEAR'}")
    print(f"  GATE 2 (vs A0/A1)      : {'TRIPPED' if gate2_tripped else 'CLEAR'}")
    print(f"\n  {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWROTE {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
