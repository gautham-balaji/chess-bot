"""C9 Stage 2b — matched-fusion ENGINE evaluation for C8a.

    python -m training.c9_stage2b

Runs the two new engine cells the Stage 2 design specifies, using the UNMODIFIED
Phase 3 evaluator:

    |      | production Ridge, d=200        | matched fusion            |
    |------|--------------------------------|---------------------------|
    | C8a  | already measured (the control) | C9b (d=200), C9c (d=d*)   |

No CNN is trained. No production file is written. The C8a control's evaluation
output is never touched - everything new lands under training/experiments/C9/.

Design: docs/C9_STAGE2_DESIGN.md sections 5, 13 and 16.
Selection: training/experiments/C9/stage2_selection.json (written by Stage 2a).

--------------------------------------------------------------------------
HOW EACH VARIANT IS STAGED, WITHOUT TOUCHING PRODUCTION
--------------------------------------------------------------------------
C9b - matched Ridge, production divisor
      A Ridge is reconstructed from the exact coefficients Stage 2a recorded for
      `A_include_checkmate|d=200` and staged via
      `evaluate_arm.stage_models_dir(..., ridge_model=...)`. The CNN is C8a's,
      copied unchanged.

C9c - jointly matched Ridge AND divisor
      `engine.py` hardcodes `tanh(x/200)`, so the divisor cannot be staged
      directly. But the model's output layer is `Dense(1, use_bias=True)`, so
      scaling ONLY that layer's kernel and bias by k = 200/d* gives

          output' = k * output   =>   tanh(output'/200) == tanh(output/d*)

      exactly. The staged CNN is therefore a genuine .keras artifact that
      reproduces divisor d* with NO engine change. Its matched Ridge comes from
      the `A_include_checkmate|d=d*` row.

      CAVEAT, stated in the report and not hidden: `engine.py` applies /200 in
      THREE places, including the unscaled 1-ply lookahead `0.5*tanh(opp/200)`.
      Rescaling the CNN changes the effective divisor there too. C9c is therefore
      NOT a single-variable contrast against the control; C9b is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from training import evaluate_arm as EA  # noqa: E402

STAGE = "C9-stage2b"
ARM = "C8a"
SEEDS = (0, 1, 2)
SUITES = ("extended", "phase0_52")
OUT_DIR = REPO_ROOT / "training" / "experiments" / "C9"
SELECTION = OUT_DIR / "stage2_selection.json"
RIDGE_DIR = OUT_DIR / "matched_ridges"
CNN_DIR = OUT_DIR / "scaled_cnns"
PRODUCTION_MODELS = REPO_ROOT / "models"
PRODUCTION_DIVISOR = 200.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT).as_posix())
    except ValueError:
        return str(path.resolve().as_posix())


# ============================================================ staged artifacts

def build_matched_ridge(coef, intercept, out_path: Path) -> dict:
    """Reconstruct a Ridge from recorded coefficients and persist it.

    Stage 2a recorded coef/intercept but not the estimator. A Ridge is fully
    determined by those, and `engine.py` reads only `coef_`, so reconstruction is
    exact by construction - and is verified after pickling.
    """
    import numpy as np
    from sklearn.linear_model import Ridge

    ridge = Ridge(alpha=1.0)
    ridge.coef_ = np.asarray(coef, dtype=float)
    ridge.intercept_ = float(intercept)
    ridge.n_features_in_ = len(ridge.coef_)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(ridge, fh)

    reloaded = pickle.load(open(out_path, "rb"))
    if not np.array_equal(reloaded.coef_, np.asarray(coef, dtype=float)):
        raise SystemExit(f"ERROR: {out_path.name} did not round-trip its coefficients")
    return {"path": _rel(out_path), "sha256": sha256_file(out_path),
            "coef": [float(x) for x in reloaded.coef_],
            "intercept": float(reloaded.intercept_)}


def build_scaled_cnn(source: Path, divisor: float, out_path: Path) -> dict:
    """Copy the CNN with ONLY its final Dense(1) kernel and bias scaled by 200/d.

    Verifies afterwards that every other layer is bit-identical and that the
    final layer is exactly k x the source's.
    """
    import numpy as np
    import keras

    k = PRODUCTION_DIVISOR / float(divisor)
    src = keras.models.load_model(source, compile=False)
    dst = keras.models.load_model(source, compile=False)

    last = dst.layers[-1]
    if last.__class__.__name__ != "Dense" or last.units != 1 or not last.use_bias:
        raise SystemExit(f"ERROR: expected a Dense(1, use_bias=True) output layer, "
                         f"found {last.__class__.__name__}")
    original = [w.copy() for w in last.get_weights()]
    last.set_weights([w * k for w in original])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst.save(out_path)

    # --- verification ---------------------------------------------------------
    check = keras.models.load_model(out_path, compile=False)
    src_w = [w for l in src.layers for w in l.get_weights()]
    chk_w = [w for l in check.layers for w in l.get_weights()]
    if len(src_w) != len(chk_w):
        raise SystemExit("ERROR: staged CNN has a different weight structure")

    n_last = len(original)
    earlier_identical = all(np.array_equal(a, b)
                            for a, b in zip(src_w[:-n_last], chk_w[:-n_last]))
    final_scaled = all(np.allclose(b, a * k, rtol=0, atol=1e-6)
                       for a, b in zip(original, check.layers[-1].get_weights()))
    if not earlier_identical:
        raise SystemExit("ERROR: staged CNN changed a layer other than the output")
    if not final_scaled:
        raise SystemExit("ERROR: staged CNN output layer is not exactly k x source")

    return {"path": _rel(out_path), "sha256": sha256_file(out_path),
            "source_sha256": sha256_file(source), "divisor": float(divisor),
            "k": k, "earlier_layers_bit_identical": True,
            "final_layer_scaled_by_k": True}


# ============================================================ evaluation

def run_cell(variant: str, seed: int, cnn: Path, ridge: Path,
             expected_coef, out_root: Path) -> dict:
    """Evaluate one (variant, seed) on both suites via the unmodified evaluator."""
    import numpy as np

    run_dir = out_root / variant / f"seed_{seed}"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    before = {p.name: (p.stat().st_mtime, p.stat().st_size)
              for p in PRODUCTION_MODELS.iterdir()}

    with tempfile.TemporaryDirectory(prefix="chessbot_c9_") as tmp:
        staging = EA.stage_models_dir(cnn, Path(tmp), ridge_model=ridge)

        staged_cnn_sha = sha256_file(staging / "cnn_model.keras")
        staged_ridge = pickle.load(open(staging / "weight_model.pkl", "rb"))
        production_ridge = pickle.load(open(PRODUCTION_MODELS / "weight_model.pkl", "rb"))
        checks = {
            "staged_cnn_matches_source": staged_cnn_sha == sha256_file(cnn),
            "staged_ridge_matches_selection": bool(np.allclose(
                staged_ridge.coef_, np.asarray(expected_coef, float), rtol=0, atol=0)),
            "staged_ridge_is_not_production": bool(not np.allclose(
                staged_ridge.coef_, production_ridge.coef_)),
        }
        if not all(checks.values()):
            raise SystemExit(f"ERROR: staging verification failed for "
                             f"{variant} seed {seed}: {checks}")
        print(f"  staged OK: cnn={staged_cnn_sha[:12]}... "
              f"ridge coef[0]={staged_ridge.coef_[0]:.4f} (not production)")

        started = time.perf_counter()
        for suite in SUITES:
            rc = EA.run_suite(staging, suite, eval_dir / suite, representation="planes12")
            if rc != 0:
                raise SystemExit(f"ERROR: evaluation failed for {variant} seed {seed} "
                                 f"suite {suite} (exit {rc})")
        elapsed = time.perf_counter() - started

    after = {p.name: (p.stat().st_mtime, p.stat().st_size)
             for p in PRODUCTION_MODELS.iterdir()}
    if before != after:
        raise SystemExit("ERROR: production models/ directory was modified")

    summary = {}
    for suite in SUITES:
        result = json.loads((eval_dir / f"{suite}.json").read_text(encoding="utf-8"))
        m = result["metrics"]
        summary[suite] = {
            "n": result["dataset"]["count"],
            "legality": m["legality_rate"],
            "top1_agreement": m["top1_agreement"],
            "top3_containment": m["top3_containment"],
            "move_regret_cp": m["move_regret_cp"],
            "blunder_rate": m["blunder_rate"],
            "regret_coverage": m["regret_coverage"],
            "spearman": m["spearman_rho_vs_stockfish_topk"],
            "mate_status_counts": m["mate_status_counts"],
            "by_side_to_move": result["breakdowns"]["by_side_to_move"],
            "by_category": result["breakdowns"]["by_category"],
        }
        print(f"    {suite:10s} legality {m['legality_rate']['percent']}%  "
              f"top1 {m['top1_agreement']['percent']}%  "
              f"top3 {m['top3_containment']['percent']}%  "
              f"regret {m['move_regret_cp']['mean']}/{m['move_regret_cp']['median']}  "
              f"blunder {m['blunder_rate']['rate']}")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")
    return {"variant": variant, "seed": seed, "staging_checks": checks,
            "wall_clock_seconds": round(elapsed, 1), "summary": summary}


# ============================================================ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="C9 Stage 2b: matched-fusion engine evaluation.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--selection", type=Path, default=SELECTION)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args(argv)

    if not args.selection.is_file():
        raise SystemExit(f"ERROR: missing {args.selection}. Run Stage 2a first.")
    sel = json.loads(args.selection.read_text(encoding="utf-8"))
    choice = sel["selection"]
    d_star = choice["divisor"]
    variant_key = choice["key"]

    print(f"{STAGE}: matched-fusion engine evaluation")
    print(f"  Stage 2a selected: {variant_key}  (d*={d_star}, "
          f"{choice['fit_variant']})")
    print(f"  collapses_to_variant_b: {choice['collapses_to_variant_b']}\n")

    # C9b always uses the production-divisor row of the SELECTED fit variant.
    b_key = f"{choice['fit_variant']}|d=200"
    cells = [("C9b", 200, b_key)]
    if d_star != 200:
        cells.append(("C9c", d_star, variant_key))
    else:
        print("  d* == 200: C collapses onto B; only C9b will be evaluated.\n")

    results = {
        "stage": STAGE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "control": {
            "name": "C8a production fusion",
            "reused_from": "training/experiments/C8a/seed_*/evaluation",
            "rerun": False,
            "ridge": "models/weight_model.pkl (production)",
            "divisor": PRODUCTION_DIVISOR,
        },
        "selection": choice,
        "cells": {},
        "lookahead_caveat": (
            "engine.py applies /200 in three places, including the unscaled "
            "1-ply lookahead 0.5*tanh(opp_best/200). C9c's rescaled CNN "
            "therefore changes the effective divisor of the lookahead as well "
            "as of the candidate term. C9c is NOT a single-variable contrast; "
            "C9b is."),
        "per_run": [],
    }

    for variant, divisor, grid_key in cells:
        print(f"{'=' * 96}\n{variant}: divisor {divisor}, Ridge row '{grid_key}'\n{'=' * 96}")
        row = sel["grid"][grid_key]
        results["cells"][variant] = {"divisor": divisor, "grid_key": grid_key,
                                     "fit_variant": row["fit_variant"],
                                     "artifacts": {}}
        for seed in args.seeds:
            coef = row["per_seed"][f"seed_{seed}"]["coef"]
            intercept = row["per_seed"][f"seed_{seed}"]["intercept"]
            print(f"\n{variant} seed {seed}")

            ridge_info = build_matched_ridge(
                coef, intercept, RIDGE_DIR / f"{variant}_seed_{seed}.pkl")
            print(f"  ridge   coef[0]={ridge_info['coef'][0]:.4f} "
                  f"sha={ridge_info['sha256'][:12]}...")

            source_cnn = (REPO_ROOT / "training" / "experiments" / ARM /
                          f"seed_{seed}" / "models" / "cnn_model.keras")
            if divisor == PRODUCTION_DIVISOR:
                cnn_path = source_cnn
                cnn_info = {"path": _rel(source_cnn),
                            "sha256": sha256_file(source_cnn),
                            "rescaled": False, "divisor": PRODUCTION_DIVISOR}
                print(f"  cnn     C8a seed {seed} unchanged "
                      f"sha={cnn_info['sha256'][:12]}...")
            else:
                cnn_path = CNN_DIR / f"{variant}_seed_{seed}.keras"
                cnn_info = build_scaled_cnn(source_cnn, divisor, cnn_path)
                cnn_info["rescaled"] = True
                print(f"  cnn     rescaled k={cnn_info['k']:.4f} "
                      f"(d={divisor}) sha={cnn_info['sha256'][:12]}... "
                      f"earlier layers identical: "
                      f"{cnn_info['earlier_layers_bit_identical']}")

            results["cells"][variant]["artifacts"][f"seed_{seed}"] = {
                "ridge": ridge_info, "cnn": cnn_info}
            results["per_run"].append(
                run_cell(variant, seed, cnn_path, Path(ridge_info["path"]),
                         coef, args.out_dir))

    path = args.out_dir / "stage2_results.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'=' * 96}")
    print(f"WROTE {_rel(path)}")
    print(f"suite runs completed: {len(results['per_run']) * len(SUITES)}")
    print("Production models/ verified unmodified after every cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
