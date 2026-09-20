"""Run the UNMODIFIED Phase 3 evaluator against a CNN whose input is not 12 planes.

    python -m training.evaluate_planes_runner --representation planes16 \\
        --dataset <suite.json> --out-prefix <prefix>

--------------------------------------------------------------------------
WHY THIS EXISTS
--------------------------------------------------------------------------
`engine.board_to_planes` hardcodes `np.zeros((8, 8, 12))`. Any model with a
different channel count therefore cannot be fed by the shipped engine - Keras
raises a shape error on the first prediction. A0, A1 and A2 need no shim
because they are 12-plane arms; A3 (18 planes) and A13 (16 planes) cannot be
evaluated without one.

This is a real property of the engine, not a workaround for a test problem, and
it is itself a finding: **those models are not deployable to the current engine
without a production change.** See docs/C6_A3_REPORT.md and docs/C6_A13_REPORT.md.

--------------------------------------------------------------------------
WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT DO
--------------------------------------------------------------------------
It imports `engine`, rebinds the single name `engine.board_to_planes` to the
requested encoder, and then calls `evaluation.evaluate.main()`.

  - No file on disk is modified. `engine.py` and `evaluation/evaluate.py` are
    read and imported exactly as committed; the rebinding lives in this process
    only and dies with it.
  - All three of the engine's call sites (`cnn_evaluate`, `rerank_moves`, and
    the 1-ply lookahead) resolve `board_to_planes` through the module global, so
    one rebinding covers every path. A test asserts that behaviourally.
  - Nothing else about the engine changes: the same heuristics, the same Ridge
    fusion, the same C1/C2 fixes, the same 1-ply lookahead, the same evaluator,
    the same Stockfish configuration.

So the only difference between one of these runs and an A2 run is the board
encoding - which is precisely the experimental variable. The comparison stays
clean.

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

    # Pull our own flag out; everything else is forwarded to the evaluator
    # untouched, so its argument surface stays the committed one.
    representation = None
    forwarded = []
    i = 0
    while i < len(argv):
        if argv[i] == "--representation":
            representation = argv[i + 1]
            i += 2
        elif argv[i].startswith("--representation="):
            representation = argv[i].split("=", 1)[1]
            i += 1
        else:
            forwarded.append(argv[i])
            i += 1

    if representation is None:
        raise SystemExit("ERROR: --representation is required "
                         "(e.g. --representation planes16)")

    from training import representations as REPS

    rep = REPS.get(representation)
    if rep.N_PLANES == 12:
        raise SystemExit(
            f"ERROR: {representation} has 12 planes and needs no shim - run "
            f"evaluation/evaluate.py directly, as A0/A1/A2 do.")

    # Import the engine FIRST, then rebind. `evaluation.evaluate.main()` does
    # `import engine as engine_mod` internally, which returns this same, already
    # patched module object out of sys.modules.
    import engine as engine_mod

    if engine_mod.board_to_planes.__module__ == rep.__name__:
        raise SystemExit(f"ERROR: engine.board_to_planes is already {representation}")

    engine_mod.board_to_planes = rep.board_to_planes
    print(f"patched engine.board_to_planes -> {rep.__name__}.board_to_planes "
          f"{rep.BOARD_SHAPE} (in-process only; no file modified)")

    # Fail fast and clearly if the staged model does not match, rather than
    # surfacing a Keras shape error 100 positions in.
    actual = tuple(engine_mod.cnn_model.input_shape)
    if actual[1:] != rep.BOARD_SHAPE:
        raise SystemExit(
            f"ERROR: staged CNN expects input {actual}, but this runner feeds "
            f"(None, {rep.BOARD_SHAPE}). Is the staged model really a "
            f"{representation} arm?")
    print(f"staged CNN input shape {actual} - matches {representation}")

    from evaluation import evaluate as ev

    sys.argv = ["evaluate.py", *forwarded]
    return ev.main()


if __name__ == "__main__":
    raise SystemExit(main())
