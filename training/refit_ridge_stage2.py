"""C6-A1R Stage 2 - matched-fusion engine evaluation.

    python -m training.refit_ridge_stage2

Runs the 2x2 diagnostic's missing cells: each A0/A1 CNN paired with the Ridge
refitted for it in Stage 1, evaluated by the UNMODIFIED Phase 3 evaluator.

    |    | production Ridge        | matched Ridge |
    |----|-------------------------|---------------|
    | A0 | already measured (A0)   | RUN HERE      |
    | A1 | already measured (A1)   | RUN HERE      |

No CNN is trained. No production file is written. Existing A0/A1 evaluation
output is never touched - everything lands under training/experiments/A1R/.

--------------------------------------------------------------------------
WHERE THE MATCHED RIDGE COMES FROM
--------------------------------------------------------------------------
Stage 1 recorded each refit's `coef` and `intercept` in stage1_results.json but
did not persist the estimator. Rather than refit (which would risk drift), the
Ridge objects are RECONSTRUCTED from those exact recorded numbers. A Ridge is
fully determined by coef_/intercept_, and engine.py reads only `coef_`, so the
reconstruction is exact by construction and is verified after pickling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from training import evaluate_arm as EA  # noqa: E402

STAGE = "A1R-stage2"
OUT_DIR = REPO_ROOT / "training" / "experiments" / "A1R"
STAGE1 = OUT_DIR / "stage1_results.json"
RIDGE_DIR = OUT_DIR / "matched_ridges"
PRODUCTION_MODELS = REPO_ROOT / "models"
SUITES = ("extended", "phase0_52")


def _rel(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (scratch/tmp paths)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT).as_posix())
    except ValueError:
        return str(path.resolve().as_posix())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================ matched ridges

def build_matched_ridge(coef, intercept, out_path: Path) -> dict:
    """Reconstruct and persist a Ridge with the exact Stage 1 coefficients."""
    from sklearn.linear_model import Ridge
    import numpy as np

    model = Ridge(alpha=1.0)
    model.coef_ = np.asarray(coef, dtype=float)
    model.intercept_ = float(intercept)
    model.n_features_in_ = len(coef)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(model, fh)

    # Verify the round-trip: what the engine will load must equal Stage 1.
    reloaded = pickle.load(open(out_path, "rb"))
    if not np.allclose(reloaded.coef_, np.asarray(coef, dtype=float), rtol=0, atol=0):
        raise SystemExit(f"ERROR: Ridge round-trip mismatch for {out_path}")

    return {"path": _rel(out_path),
            "sha256": sha256_file(out_path),
            "coef": [float(x) for x in reloaded.coef_],
            "intercept": float(reloaded.intercept_)}


# ============================================================ one run

def run_one(arm: str, seed: int, ridge_path: Path, expected_coef, out_root: Path) -> dict:
    cnn = REPO_ROOT / "training" / "experiments" / arm / f"seed_{seed}" / "models" / "cnn_model.keras"
    if not cnn.is_file():
        raise SystemExit(f"ERROR: missing CNN {cnn}")

    run_dir = out_root / f"{arm}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prod_before = {p.name: sha256_file(p) for p in sorted(PRODUCTION_MODELS.iterdir())}
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="a1r_stage2_") as tmp:
        staging = EA.stage_models_dir(cnn, Path(tmp), ridge_model=ridge_path)

        # --- verify the staged artifacts are the intended arm/seed -------------
        staged_cnn_sha = sha256_file(staging / "cnn_model.keras")
        staged_ridge = pickle.load(open(staging / "weight_model.pkl", "rb"))
        import numpy as np
        checks = {
            "staged_cnn_matches_source": staged_cnn_sha == sha256_file(cnn),
            "staged_ridge_matches_stage1": bool(
                np.allclose(staged_ridge.coef_, np.asarray(expected_coef, float),
                            rtol=0, atol=0)),
            "staged_ridge_is_not_production": bool(not np.allclose(
                staged_ridge.coef_,
                pickle.load(open(PRODUCTION_MODELS / "weight_model.pkl", "rb")).coef_)),
        }
        if not all(checks.values()):
            raise SystemExit(f"ERROR: staging verification failed for {arm} seed {seed}: {checks}")
        print(f"  staged OK: cnn={staged_cnn_sha[:12]}… "
              f"ridge coef[0]={staged_ridge.coef_[0]:.4f} (not production)")

        env = dict(os.environ)
        env["CHESS_BOT_MODELS_DIR"] = str(staging)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        for suite in SUITES:
            cmd = [sys.executable, str(REPO_ROOT / "evaluation" / "evaluate.py"),
                   "--dataset", str(REPO_ROOT / "evaluation" / "positions" / f"{suite}.json"),
                   "--out-prefix", str(run_dir / suite)]
            print(f"    -> {suite}", flush=True)
            if subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode != 0:
                raise SystemExit(f"ERROR: evaluation failed for {arm} seed {seed} / {suite}")

    prod_after = {p.name: sha256_file(p) for p in sorted(PRODUCTION_MODELS.iterdir())}
    if prod_before != prod_after:
        raise SystemExit("ERROR: production models/ changed during the run")

    return {
        "arm": arm, "seed": seed, "fusion": "matched",
        "cnn": _rel(cnn),
        "cnn_sha256": staged_cnn_sha,
        "ridge": _rel(ridge_path),
        "staging_checks": checks,
        "seconds": round(time.perf_counter() - started, 1),
        "results": {s: _rel(run_dir / f"{s}.json") for s in SUITES},
    }


# ============================================================ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to e.g. A0:0 A1:2")
    ap.add_argument("--out-root", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if not STAGE1.is_file():
        raise SystemExit(f"ERROR: Stage 1 results not found at {STAGE1}. Run "
                         f"`python -m training.refit_ridge` first.")
    stage1 = json.loads(STAGE1.read_text(encoding="utf-8"))
    print(f"{STAGE}: matched-fusion engine evaluation")
    print(f"  Stage 1 source: {_rel(STAGE1)}")
    print(f"  production coef: {[round(x, 4) for x in stage1['production_coef']]}\n")

    wanted = set(args.only) if args.only else None
    manifest = {"stage": STAGE,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "stage1_results": _rel(STAGE1),
                "production_coef": stage1["production_coef"],
                "evaluator": "evaluation/evaluate.py (unmodified)",
                "suites": list(SUITES),
                "matched_ridges": [], "runs": []}

    started = time.perf_counter()
    for row in stage1["results"]:
        arm, seed = row["arm"], row["seed"]
        if wanted and f"{arm}:{seed}" not in wanted:
            continue
        print(f"{arm} seed {seed}")
        ridge_info = build_matched_ridge(row["coef"], row["intercept"],
                                         RIDGE_DIR / f"{arm}_seed_{seed}_weight_model.pkl")
        ridge_info.update({"arm": arm, "seed": seed})
        manifest["matched_ridges"].append(ridge_info)
        manifest["runs"].append(
            run_one(arm, seed, REPO_ROOT / ridge_info["path"], row["coef"], args.out_root))

    manifest["total_seconds"] = round(time.perf_counter() - started, 1)
    out = args.out_root / "stage2_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWROTE {out}")
    print(f"total runtime: {manifest['total_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
