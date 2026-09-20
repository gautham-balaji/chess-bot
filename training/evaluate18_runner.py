"""Run the UNMODIFIED Phase 3 evaluator against an 18-plane CNN.

    python -m training.evaluate18_runner --dataset <suite.json> --out-prefix <prefix>

--------------------------------------------------------------------------
WHY THIS EXISTS
--------------------------------------------------------------------------
`engine.board_to_planes` hardcodes `np.zeros((8, 8, 12))`. An 18-channel model
therefore cannot be fed by the shipped engine at all - Keras raises a shape
error on the first prediction. A0, A1 and A2 needed no such shim because they
are 12-plane arms; A3 cannot be evaluated without one.

This is a real property of the engine, not a workaround for a test problem, and
it is itself an A3 finding: **the 18-plane model is not deployable to the current
engine without a production change.** See docs/C6_A3_REPORT.md.

--------------------------------------------------------------------------
WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT DO
--------------------------------------------------------------------------
It imports `engine`, rebinds the single name `engine.board_to_planes` to the
18-plane encoder, and then calls `evaluation.evaluate.main()`.

  - No file on disk is modified. `engine.py` and `evaluation/evaluate.py` are
    read and imported exactly as committed; the rebinding lives in this process
    only and dies with it.
  - All three of the engine's call sites (`cnn_evaluate`, `rerank_moves`, and
    the 1-ply lookahead) resolve `board_to_planes` through the module global, so
    one rebinding covers every path. A test asserts that.
  - Nothing else about the engine changes: the same heuristics, the same Ridge
    fusion, the same C1/C2 fixes, the same 1-ply lookahead, the same evaluator,
    the same Stockfish configuration.

So the only difference between an A3 run and an A2 run is the board encoding -
which is precisely A3's experimental variable. The comparison stays clean.

The evaluator is imported and its `main()` called in-process rather than
re-implemented, so the measurement path is the committed one.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Import the engine FIRST, then rebind. `evaluation.evaluate.main()` does
    # `import engine as engine_mod` internally, which returns this same, already
    # patched module object out of sys.modules.
    import engine as engine_mod
    from training import representation18 as rep18

    original = engine_mod.board_to_planes
    if original.__module__ == rep18.__name__:
        raise SystemExit("ERROR: engine.board_to_planes is already the 18-plane encoder")

    engine_mod.board_to_planes = rep18.board_to_planes
    print(f"patched engine.board_to_planes -> {rep18.__name__}.board_to_planes "
          f"{rep18.BOARD_SHAPE} (in-process only; no file modified)")

    # Fail fast and clearly if the staged model is not actually 18-channel,
    # rather than surfacing a Keras shape error 100 positions in.
    expected = (None, *rep18.BOARD_SHAPE)
    actual = tuple(engine_mod.cnn_model.input_shape)
    if actual[1:] != expected[1:]:
        raise SystemExit(
            f"ERROR: staged CNN expects input {actual}, but this runner feeds "
            f"{expected}. Is the staged model really an 18-plane arm?")
    print(f"staged CNN input shape {actual} - matches the 18-plane encoder")

    from evaluation import evaluate as ev

    sys.argv = ["evaluate.py", *argv]
    return ev.main()


if __name__ == "__main__":
    raise SystemExit(main())
