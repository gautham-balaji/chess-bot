"""Phase 0 baseline: run Stockfish over the fixed FEN suite.

Uses the EXISTING configuration found in the repository, unchanged:
  - binary path: the literal hardcoded path in app.py:383 / benchmark_stockfish.py:12
  - depth 8:     STOCKFISH_DEPTH in both files
  - access via chess.engine.SimpleEngine.popen_uci, as app.py:386 does

The Stockfish integration is deliberately NOT redesigned here. The only
deviations from app.py, both recorded in the output, are:
  1. one engine process is reused for all positions instead of one per call
     (app.py spawns per request); this affects timing only, and is noted.
  2. scores are additionally captured from White's POV so that later phases
     have an unambiguous frame. The side-to-move-relative value app.py uses
     is recorded too, unchanged.

Usage:
  python baseline/scripts/measure_stockfish.py baseline/fens.json baseline/stockfish_results.json
"""
import json
import os
import statistics
import sys
import time

import chess
import chess.engine

# Exactly the literal from app.py:383 and benchmark_stockfish.py:12.
HARDCODED_PATH = (
    r"C:\Users\vsriv\Downloads\stockfish-windows-x86-64-avx2\stockfish"
    r"\stockfish-windows-x86-64-avx2.exe"
)
DEPTH = 8          # STOCKFISH_DEPTH in app.py:384 and benchmark_stockfish.py:13
MULTIPV = 3        # additive: not used anywhere in the existing repo


def pct(s, q):
    if not s:
        return None
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return round(s[k], 2)


def summarize(vals):
    if not vals:
        return None
    s = sorted(vals)
    return {"n": len(s), "min_ms": round(s[0], 2),
            "median_ms": round(statistics.median(s), 2),
            "mean_ms": round(statistics.fmean(s), 2),
            "p95_ms": pct(s, 0.95), "max_ms": round(s[-1], 2)}


def main():
    fens_path, out_path = sys.argv[1], sys.argv[2]
    positions = json.load(open(fens_path, encoding="utf-8"))["positions"]

    path = os.environ.get("STOCKFISH_PATH", HARDCODED_PATH)
    if not os.path.isfile(path):
        json.dump({"available": False, "path_tried": path,
                   "reason": "binary not found"},
                  open(out_path, "w", encoding="utf-8"), indent=2)
        print("Stockfish NOT available at", path)
        return

    rows, times = [], []
    with chess.engine.SimpleEngine.popen_uci(path) as sf:
        engine_id = dict(sf.id)
        # Record options as configured. The existing repo sets NOTHING here:
        # no Threads, no Hash, no seed. Defaults are whatever Stockfish chose.
        opts = {}
        for key in ("Threads", "Hash", "MultiPV", "Skill Level", "UCI_Chess960"):
            if key in sf.options:
                opts[key] = sf.options[key].default

        for i, p in enumerate(positions, 1):
            board = chess.Board(p["fen"])

            t0 = time.perf_counter()
            play = sf.play(board, chess.engine.Limit(depth=DEPTH))
            play_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            infos = sf.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=MULTIPV)
            analyse_ms = (time.perf_counter() - t1) * 1000.0
            if isinstance(infos, dict):
                infos = [infos]

            top_n = []
            for info in infos:
                pv = info.get("pv") or []
                score = info["score"]
                top_n.append({
                    "uci": pv[0].uci() if pv else None,
                    "score_white_cp": score.white().score(mate_score=100000),
                    "score_relative_cp": score.relative.score(mate_score=100000),
                    "is_mate": score.relative.is_mate(),
                    "mate_in": score.relative.mate(),
                    "depth": info.get("depth"),
                })

            times.append(play_ms)
            rows.append({
                "id": p["id"], "fen": p["fen"], "category": p["category"],
                "side_to_move": p["side_to_move"],
                "best_move_uci": play.move.uci() if play.move else None,
                "best_move_san": board.san(play.move) if play.move else None,
                "play_time_ms": round(play_ms, 2),
                "analyse_time_ms": round(analyse_ms, 2),
                "depth": DEPTH,
                "top_n": top_n,
                "score_white_cp": top_n[0]["score_white_cp"] if top_n else None,
                "score_relative_cp": top_n[0]["score_relative_cp"] if top_n else None,
                "is_mate_score": top_n[0]["is_mate"] if top_n else None,
            })
            print(f"  {i:2d}/{len(positions)} {p['id']} {rows[-1]['best_move_uci']} "
                  f"{play_ms:.0f}ms cp_white={rows[-1]['score_white_cp']}", flush=True)

    payload = {
        "schema_version": 1,
        "generated_by": "baseline/scripts/measure_stockfish.py",
        "available": True,
        "binary_path": path,
        "path_source": "hardcoded literal from app.py:383 / benchmark_stockfish.py:12",
        "engine_id": engine_id,
        "depth": DEPTH,
        "multipv": MULTIPV,
        "uci_options_as_configured_by_repo": opts,
        "options_note": (
            "The existing repository sets NO UCI options (no Threads, no Hash, no "
            "Skill Level). Values shown are Stockfish defaults. Because Threads is "
            "not pinned, Stockfish is not guaranteed to be reproducible run-to-run."
        ),
        "deviations_from_app_py": [
            "One engine process reused for all positions; app.py:386 spawns one "
            "process per /benchmark request. Affects timing only, and makes these "
            "timings FASTER than app.py would achieve (no process spawn per call).",
            "MultiPV=3 captured; the existing repo never requests MultiPV.",
            "score.white() captured alongside score.relative; app.py:395 uses "
            "score.relative only.",
        ],
        "position_count": len(rows),
        "play_latency_ms": summarize(times),
        "results": rows,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print("WROTE", out_path)
    print("latency:", payload["play_latency_ms"])


if __name__ == "__main__":
    main()
