"""Backwards-compatible entry point for A3's 18-plane evaluation shim.

    python -m training.evaluate18_runner --dataset <suite.json> --out-prefix <prefix>

The shim was generalised when A13 added a 16-plane arm, because the mechanism
has nothing to do with the number 18 - it is needed for any channel count the
shipped `engine.board_to_planes` does not produce. The implementation now lives
in `training/evaluate_planes_runner.py`.

This module remains so that the command documented in docs/C6_A3_REPORT.md keeps
working unchanged. It only pins the representation to `planes18` and delegates.
"""
from __future__ import annotations

import sys

from training import evaluate_planes_runner as _runner

REPRESENTATION = "planes18"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    return _runner.main(["--representation", REPRESENTATION, *argv])


if __name__ == "__main__":
    raise SystemExit(main())
