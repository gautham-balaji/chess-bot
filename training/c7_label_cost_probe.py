"""C7 labelling-cost probe: how long does Stockfish take per position, by phase?

    python -m training.c7_label_cost_probe --per-phase 120

Read-only measurement. Uses the EXACT Phase 3 / build_dataset configuration
(depth 8, Threads=1, Hash=16MB, hash cleared per position) and the exact A2
label policy, so the timings and the label mix transfer directly to a dataset
rebuild. Writes no dataset and trains nothing.

--------------------------------------------------------------------------
WHY A PROBE IS NEEDED
--------------------------------------------------------------------------
`dataset_v1` cost 187s for 9,667 positions, but every one of those positions is
at most 20 plies deep. An expanded dataset would be dominated by middlegame and
endgame positions, whose depth-8 search cost is not the same. Extrapolating the
opening-only rate would give a wrong estimate in an unknown direction, so the
rate is measured per phase instead.

The same run also measures the cp/mate label mix per phase, which the existing
dataset cannot show: dataset_v1 is 98.1% cp because near-opening positions are
almost never mate-scored.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import chess  # noqa: E402
import pandas as pd  # noqa: E402

import config as project_config  # noqa: E402
from training import build_dataset as B  # noqa: E402
from training import labels as L  # noqa: E402
from training.c7_dataset_audit import (  # noqa: E402
    MIN_PLY_DEFAULT, classify_phase, replay_game,
)

PROBE_VERSION = "c7-cost-1"
DEFAULT_OUT = REPO_ROOT / "training" / "artifacts" / "c7_label_cost.json"
PROBE_SEED = 20260920      # distinct from every split/training seed in the repo


def collect_positions(df: pd.DataFrame, per_phase: int, seed: int) -> list[dict]:
    """Gather `per_phase` positions for each phase, walking games deterministically."""
    import random
    rng = random.Random(seed)
    order = list(range(len(df)))
    rng.shuffle(order)

    wanted = {"opening": per_phase, "middlegame": per_phase, "endgame": per_phase}
    got = defaultdict(list)

    for row_pos in order:
        if all(len(got[p]) >= wanted[p] for p in wanted):
            break
        row = df.iloc[row_pos]
        rep = replay_game(row.get("id"), row_pos, row.get("moves"))
        if not rep.n_plies:
            continue
        # One position per game, so the probe is not dominated by a few games.
        candidates = list(range(MIN_PLY_DEFAULT - 1, rep.n_plies))
        if not candidates:
            continue
        rng.shuffle(candidates)
        for idx in candidates:
            phase = classify_phase(idx + 1, rep.piece_counts[idx])
            if len(got[phase]) >= wanted[phase]:
                continue
            board = chess.Board()
            for token in str(row.get("moves")).split()[: idx + 1]:
                board.push_san(token)
            got[phase].append({
                "game_id": rep.game_id, "ply": idx + 1, "phase": phase,
                "fen": board.fen(),
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "is_checkmate": board.is_checkmate(),
                "piece_count": chess.popcount(board.occupied),
            })
            break

    return [p for phase in ("opening", "middlegame", "endgame") for p in got[phase]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure Stockfish labelling cost by phase.")
    ap.add_argument("--source", type=Path, default=REPO_ROOT / "games.csv")
    ap.add_argument("--per-phase", type=int, default=120)
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.source)
    print(f"source: {args.source.name}, {len(df)} rows")
    print(f"collecting {args.per_phase} positions per phase...")
    positions = collect_positions(df, args.per_phase, args.seed)
    print(f"  collected {len(positions)}: "
          f"{dict(Counter(p['phase'] for p in positions))}")

    engine, info, path = B.open_engine()
    print(f"stockfish: {info.version}")
    print(f"  depth={B.SF_DEPTH} Threads={B.SF_THREADS} Hash={B.SF_HASH_MB}MB "
          f"clear_hash={B.SF_CLEAR_HASH_PER_POSITION}")

    per_phase_ms = defaultdict(list)
    eval_types = defaultdict(Counter)
    clipped = defaultdict(int)
    labels_seen = defaultdict(list)

    try:
        for i, pos in enumerate(positions, 1):
            started = time.perf_counter()
            if B.SF_CLEAR_HASH_PER_POSITION:
                engine.send_ucinewgame_command()
            engine.set_fen_position(pos["fen"])
            evaluation = engine.get_evaluation()
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            label = L.make_label(
                eval_type=evaluation["type"],
                raw_value=int(evaluation["value"]),
                side_to_move_is_white=(pos["side_to_move"] == "white"),
                is_checkmate=pos["is_checkmate"],
                cp_clip=L.CP_CLIP,
            )
            per_phase_ms[pos["phase"]].append(elapsed_ms)
            eval_types[pos["phase"]][label.eval_type] += 1
            labels_seen[pos["phase"]].append(label.label)
            if label.was_clipped:
                clipped[pos["phase"]] += 1

            if i % 100 == 0:
                print(f"    {i}/{len(positions)}", flush=True)
    finally:
        engine.send_quit_command()

    out = {
        "probe_version": PROBE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "stockfish": {
            "version": info.version, "depth": B.SF_DEPTH, "threads": B.SF_THREADS,
            "hash_mb": B.SF_HASH_MB,
            "clear_hash_per_position": B.SF_CLEAR_HASH_PER_POSITION,
        },
        "label_policy": L.policy_summary(),
        "per_phase": {},
    }

    print(f"\n{'phase':14s}{'n':>6s}{'mean ms':>10s}{'median':>9s}{'p90':>9s}"
          f"{'pos/sec':>10s}{'cp':>7s}{'mate':>7s}{'clipped':>9s}")
    all_ms = []
    for phase in ("opening", "middlegame", "endgame"):
        ms = per_phase_ms.get(phase)
        if not ms:
            continue
        all_ms.extend(ms)
        s = sorted(ms)
        block = {
            "n": len(ms),
            "mean_ms": round(st.fmean(ms), 3),
            "median_ms": round(s[len(s) // 2], 3),
            "p90_ms": round(s[int(0.9 * (len(s) - 1))], 3),
            "max_ms": round(s[-1], 3),
            "positions_per_second": round(1000.0 / st.fmean(ms), 2),
            "eval_type_counts": dict(eval_types[phase]),
            "mate_fraction": round(eval_types[phase]["mate"] / len(ms), 4),
            "clipped": clipped[phase],
            "label_mean": round(st.fmean(labels_seen[phase]), 2),
        }
        out["per_phase"][phase] = block
        print(f"  {phase:12s}{block['n']:>6}{block['mean_ms']:>10.2f}"
              f"{block['median_ms']:>9.2f}{block['p90_ms']:>9.2f}"
              f"{block['positions_per_second']:>10.2f}"
              f"{eval_types[phase]['cp']:>7}{eval_types[phase]['mate']:>7}"
              f"{clipped[phase]:>9}")

    overall_rate = 1000.0 / st.fmean(all_ms)
    out["overall"] = {
        "n": len(all_ms),
        "mean_ms": round(st.fmean(all_ms), 3),
        "positions_per_second": round(overall_rate, 2),
    }
    out["projected_wall_clock_minutes"] = {
        str(n): round(n / overall_rate / 60.0, 1)
        for n in (20_000, 50_000, 100_000, 200_000)
    }
    print(f"\noverall: {out['overall']['mean_ms']:.2f} ms/position "
          f"= {overall_rate:.1f} positions/sec (single process)")
    for n, mins in out["projected_wall_clock_minutes"].items():
        print(f"  {int(n):>7,} positions -> {mins:>6.1f} min")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWROTE {args.out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
