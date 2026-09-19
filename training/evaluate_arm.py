"""Engine-level evaluation of an experimental model, via the unmodified Phase 3 evaluator.

    python -m training.evaluate_arm --arm A0 --seed 0

--------------------------------------------------------------------------
HOW THE MODEL IS INJECTED  (no production file is modified)
--------------------------------------------------------------------------
`engine.py` loads the CNN from `config.CNN_MODEL_PATH`, and `config.py` already
resolves that through:

    MODELS_DIR = Path(os.environ.get("CHESS_BOT_MODELS_DIR") or (REPO_ROOT / "models"))

So this script assembles a models directory containing the experimental
`cnn_model.keras` plus a COPY of the production `weight_model.pkl`, sets
CHESS_BOT_MODELS_DIR to it, and runs `evaluation/evaluate.py` as a subprocess.

Nothing in engine.py, app.py, evaluation/ or models/ is touched, and the
production models directory is never written to. The subprocess boundary also
guarantees the experimental model cannot leak into any other process.

--------------------------------------------------------------------------
WHAT IS HELD CONSTANT  (fairness rule)
--------------------------------------------------------------------------
Same evaluator, same position files, same Stockfish configuration (depth 8,
Threads=1, Hash=16, Clear Hash per position), same engine code, same Ridge
fusion weights.

The Ridge weights are DELIBERATELY not refitted per arm. They stay at the
production `weight_model.pkl` for every arm, so "engine configuration" is
identical across A0/A1/A2/A3. Note this is a known limitation rather than a
neutral choice: those coefficients were fitted against the ORIGINAL CNN's output
scale, so a retrained CNN is being fused with weights tuned for a different
model. It biases all arms in the same direction, which keeps the comparison
fair, but it means no arm should be read as "the best this architecture can do".
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PRODUCTION_MODELS = REPO_ROOT / "models"
EXPERIMENTS_DIR = REPO_ROOT / "training" / "experiments"
SUITES = {
    "extended": REPO_ROOT / "evaluation" / "positions" / "extended.json",
    "phase0_52": REPO_ROOT / "evaluation" / "positions" / "phase0_52.json",
}


def stage_models_dir(experiment_model: Path, staging: Path,
                     ridge_model: Path | None = None) -> Path:
    """Build a models/ directory holding the experimental CNN + a Ridge.

    `ridge_model` defaults to None, which copies the PRODUCTION Ridge - the
    behaviour every A0 and A1 run used, and the behaviour the existing tests
    pin. Passing an explicit path instead stages that Ridge, which is what the
    A1R Stage 2 matched-fusion diagnostic needs.

    Either way the production directory is only ever READ.
    """
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(experiment_model, staging / "cnn_model.keras")
    shutil.copy2(ridge_model or (PRODUCTION_MODELS / "weight_model.pkl"),
                 staging / "weight_model.pkl")
    return staging


def run_suite(models_dir: Path, suite: str, out_prefix: Path) -> int:
    env = dict(os.environ)
    env["CHESS_BOT_MODELS_DIR"] = str(models_dir)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    cmd = [sys.executable, str(REPO_ROOT / "evaluation" / "evaluate.py"),
           "--dataset", str(SUITES[suite]),
           "--out-prefix", str(out_prefix)]
    print(f"  -> {suite}: {' '.join(cmd[-4:])}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--suites", nargs="+", default=["extended", "phase0_52"],
                    choices=sorted(SUITES))
    ap.add_argument("--out-root", type=Path, default=EXPERIMENTS_DIR)
    args = ap.parse_args()

    run_dir = args.out_root / args.arm / f"seed_{args.seed}"
    model = run_dir / "models" / "cnn_model.keras"
    if not model.is_file():
        raise SystemExit(f"ERROR: no trained model at {model}. Run training/train.py first.")

    before = {p.name: p.stat().st_mtime for p in PRODUCTION_MODELS.iterdir()}

    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="chessbot_arm_models_") as tmp:
        models_dir = stage_models_dir(model, Path(tmp))
        print(f"injecting via CHESS_BOT_MODELS_DIR={models_dir}")
        for suite in args.suites:
            rc = run_suite(models_dir, suite, eval_dir / suite)
            if rc != 0:
                raise SystemExit(f"ERROR: evaluation failed for {suite} (exit {rc})")

    after = {p.name: p.stat().st_mtime for p in PRODUCTION_MODELS.iterdir()}
    if before != after:
        raise SystemExit("ERROR: production models/ directory was modified - aborting")
    print("verified: production models/ untouched")

    summary = {}
    for suite in args.suites:
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
        }
        print(f"\n{suite} (n={summary[suite]['n']}):")
        print(f"  legality        {m['legality_rate']['percent']}%")
        print(f"  top-1 agreement {m['top1_agreement']['percent']}% "
              f"({m['top1_agreement']['numerator']}/{m['top1_agreement']['denominator']})")
        print(f"  top-3 contain.  {m['top3_containment']['percent']}%")
        print(f"  regret mean/median/p95  {m['move_regret_cp']['mean']} / "
              f"{m['move_regret_cp']['median']} / {m['move_regret_cp']['p95']} cp "
              f"(n={m['move_regret_cp']['n']})")
        print(f"  blunder rate    {m['blunder_rate']['blunders']}/"
              f"{m['blunder_rate']['denominator']} = {m['blunder_rate']['rate']}")
        print(f"  spearman        mean {m['spearman_rho_vs_stockfish_topk']['mean']} "
              f"median {m['spearman_rho_vs_stockfish_topk']['median']}")
        print(f"  mate statuses   {m['mate_status_counts']}")

    out = eval_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
