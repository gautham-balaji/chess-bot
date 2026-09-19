"""DISABLED — this benchmark did not produce real measurements.

Phase 0 verification found that this script's headline outputs were fabricated:

  * line 57 (original): ``results["agreement"] = np.random.rand() < 0.7``
    The genuine ``hybrid_move`` and ``stockfish_move`` were computed on the two
    preceding lines and then discarded. The printed "top move agreement" was a
    random number.

  * lines 113-126 (original): the script computed ``hybrid_times`` and
    ``stockfish_times`` from real timings and then printed hardcoded literals
    instead - "Average time 9.2ms / 15.1ms", "Min 7.10ms", "Max 12.30ms",
    "Speedup 0.60x faster".

Because those numbers were presented as measurements, the script is disabled
rather than quietly corrected. The unmodified original is preserved at
``archive/phase1_invalid_benchmarks/benchmark_stockfish.py.original``.

Real, reproducible measurements of the same comparison already exist:

    baseline/BASELINE.md           human-readable report
    baseline/comparison.json       top-1 agreement, top-3 containment,
                                   legality rate, latency ratio, chance baselines
    baseline/scripts/compare.py    the code that produced them

A replacement evaluation harness is deliberately NOT built here; that is planned
work, not part of the Phase 1 integrity pass. Running this file exits non-zero.
"""
import sys

REPLACEMENT = "baseline/BASELINE.md  (data: baseline/comparison.json)"
ARCHIVED_ORIGINAL = "archive/phase1_invalid_benchmarks/benchmark_stockfish.py.original"


def main() -> int:
    sys.stderr.write(
        "\n"
        "benchmark_stockfish.py is DISABLED.\n"
        "\n"
        "Its agreement rate was generated with np.random.rand() and its timing\n"
        "statistics were hardcoded, so its output was not a measurement.\n"
        "\n"
        f"  Verified results: {REPLACEMENT}\n"
        f"  Original script:  {ARCHIVED_ORIGINAL}\n"
        "\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
