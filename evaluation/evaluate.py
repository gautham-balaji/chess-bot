"""Reproducible evaluation harness: engine move quality vs a Stockfish reference.

Measures how good the engine's chosen moves are by having the SAME Stockfish
configuration evaluate the position after the engine's move and after
Stockfish's own move, then comparing those two evaluations from the mover's
point of view. The engine's internal score is never compared to Stockfish's -
see evaluation/metrics.py for why.

Usage:
    python evaluation/evaluate.py --dataset evaluation/positions/phase0_52.json
    python evaluation/evaluate.py --dataset ... --out-prefix evaluation/results/run2

Stockfish is configured explicitly (Threads=1, Hash=16, depth=8) for
reproducibility. This is the HARNESS's configuration and does not touch the
application's gameplay settings.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import chess  # noqa: E402
import chess.engine  # noqa: E402

import config as project_config  # noqa: E402
from evaluation import metrics as M  # noqa: E402

# --- Harness Stockfish configuration (reproducibility, not gameplay) ----------
SF_DEPTH = 8          # matches Phase 0 and the CNN's training-label depth
SF_THREADS = 1        # pinned: multi-threaded search is nondeterministic
SF_HASH_MB = 16       # pinned: table size changes results
SF_MULTIPV_AGREEMENT = 3   # matches Phase 0 exactly, for like-for-like comparison
SF_MULTIPV_RANKING = 8     # wider list, used only for rank correlation

# Clear Stockfish's transposition table before each position.
#
# This is NOT cosmetic. A shared hash table makes a fixed-depth search
# order-dependent: entries written while analysing position N change the search
# at position N+1. Measured on this 52-position suite, running the Phase 0 call
# sequence versus this harness's (longer) sequence with a shared hash produced
# DIFFERENT Stockfish best moves on 19 of 52 positions. With the hash cleared per
# position, the two sequences agree on 52/52 - the reference depends only on the
# position, which is what a reference must do.
SF_CLEAR_HASH_PER_POSITION = True


@contextmanager
def stockfish_session(path: str):
    """One Stockfish process for the whole run, closed even on failure."""
    engine = chess.engine.SimpleEngine.popen_uci(path)
    try:
        engine.configure({"Threads": SF_THREADS, "Hash": SF_HASH_MB})
        yield engine
    finally:
        engine.quit()


def evaluate_resulting_position(sf, board: chess.Board, pov_color) -> M.PovEval:
    """Evaluate `board` from `pov_color`'s perspective.

    Terminal positions are resolved directly rather than sent to Stockfish: a
    mated or stalemated position has no search result, and asking for one
    produces engine-specific behaviour we would rather not depend on.
    """
    if board.is_checkmate():
        # This is always a position reached by pov_color making a move, so the
        # side to move here is the opponent and pov_color delivered the mate.
        # mate_in == 0 means "mate is on the board, delivered by the POV side".
        assert board.turn != pov_color, "unexpected: POV side is the mated side"
        return M.PovEval(mate_in=0)
    if board.is_game_over():
        return M.PovEval(cp=0)  # stalemate / insufficient material / draw rules

    info = sf.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
    return M.score_from_pov(info["score"], pov_color)


def run_position(sf, engine_mod, entry: dict) -> dict:
    """Evaluate one position. Raises on unexpected failure - errors are not
    swallowed here, because a silently-skipped position would bias the run."""
    fen = entry["fen"]
    board = chess.Board(fen)
    pov = board.turn  # the side choosing the move; every score below uses this

    # Make this position's reference independent of everything evaluated before it.
    if SF_CLEAR_HASH_PER_POSITION:
        sf.configure({"Clear Hash": None})

    # --- our engine -----------------------------------------------------------
    t0 = time.perf_counter()
    engine_move, _explanation, _top3 = engine_mod.engine_move(board.copy())
    engine_latency_ms = (time.perf_counter() - t0) * 1000.0

    legal = engine_move is not None and engine_move in chess.Board(fen).legal_moves

    # Full ranking, used only for rank correlation (not timed).
    ranked = engine_mod.rerank_moves(chess.Board(fen))
    engine_rank_of = {e["move"].uci(): i for i, e in enumerate(ranked)}

    # --- stockfish reference --------------------------------------------------
    t1 = time.perf_counter()
    play_result = sf.play(board, chess.engine.Limit(depth=SF_DEPTH))
    sf_latency_ms = (time.perf_counter() - t1) * 1000.0
    sf_best = play_result.move

    infos3 = sf.analyse(board, chess.engine.Limit(depth=SF_DEPTH),
                        multipv=SF_MULTIPV_AGREEMENT)
    if isinstance(infos3, dict):
        infos3 = [infos3]
    sf_top3 = [i["pv"][0].uci() for i in infos3 if i.get("pv")]

    # --- regret ---------------------------------------------------------------
    regret_cp, mate_status, after_eng, after_sf = None, "not_evaluated", None, None
    if legal and sf_best is not None:
        b_eng = chess.Board(fen)
        b_eng.push(engine_move)
        after_eng = evaluate_resulting_position(sf, b_eng, pov)

        if engine_move == sf_best:
            # Same move => same position => regret is exactly 0 by construction.
            after_sf = after_eng
        else:
            b_sf = chess.Board(fen)
            b_sf.push(sf_best)
            after_sf = evaluate_resulting_position(sf, b_sf, pov)

        regret_cp, mate_status = M.compute_regret(after_sf, after_eng)

    # --- rank correlation over Stockfish's top-K ------------------------------
    rho = None
    k = min(SF_MULTIPV_RANKING, board.legal_moves.count())
    if k >= 3:
        infos_k = sf.analyse(board, chess.engine.Limit(depth=SF_DEPTH), multipv=k)
        if isinstance(infos_k, dict):
            infos_k = [infos_k]
        sf_ordered = [i["pv"][0].uci() for i in infos_k if i.get("pv")]
        pairs = [(engine_rank_of[u], r) for r, u in enumerate(sf_ordered)
                 if u in engine_rank_of]
        if len(pairs) >= 3:
            rho = M.spearman_rank_correlation([p[0] for p in pairs],
                                              [p[1] for p in pairs])

    return {
        "id": entry["id"],
        "fen": fen,
        "category": entry.get("category"),
        "side_to_move": entry.get("side_to_move"),
        "legal_move_count": entry.get("legal_move_count", board.legal_moves.count()),
        "engine_move": engine_move.uci() if engine_move else None,
        "engine_move_is_legal": legal,
        # engine.py's score becomes numpy.float32 once the 1-ply adjustment is
        # applied (float - np.float32 -> np.float32), which is not JSON
        # serialisable. Coerced here; the value is unchanged.
        "engine_top_score": float(ranked[0]["score"]) if ranked else None,
        "engine_latency_ms": round(engine_latency_ms, 2),
        "stockfish_best_move": sf_best.uci() if sf_best else None,
        "stockfish_top3": sf_top3,
        "stockfish_latency_ms": round(sf_latency_ms, 2),
        "eval_after_engine_move_pov_mover": after_eng.as_dict() if after_eng else None,
        "eval_after_stockfish_move_pov_mover": after_sf.as_dict() if after_sf else None,
        "move_regret_cp": regret_cp,
        "mate_status": mate_status,
        "top1_agreement": M.top1_agreement(
            engine_move.uci() if engine_move else None,
            sf_best.uci() if sf_best else None),
        "top3_containment": M.top_n_containment(
            engine_move.uci() if engine_move else None, sf_top3),
        "spearman_rho_vs_stockfish_topk": (round(rho, 4) if rho is not None else None),
    }


def aggregate(rows: list[dict], dataset: dict, sf_id: dict, sf_path: str) -> dict:
    n = len(rows)
    regrets = [r["move_regret_cp"] for r in rows if r["move_regret_cp"] is not None]
    latencies = [r["engine_latency_ms"] for r in rows]
    sf_latencies = [r["stockfish_latency_ms"] for r in rows]
    rhos = [r["spearman_rho_vs_stockfish_topk"] for r in rows
            if r["spearman_rho_vs_stockfish_topk"] is not None]

    mate_counts: dict[str, int] = {}
    for r in rows:
        mate_counts[r["mate_status"]] = mate_counts.get(r["mate_status"], 0) + 1

    def by_group(key):
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r.get(key) or "unknown", []).append(r)
        out = {}
        for name, grp in sorted(groups.items()):
            grp_regrets = [g["move_regret_cp"] for g in grp
                           if g["move_regret_cp"] is not None]
            out[name] = {
                "n": len(grp),
                "top1_agreement": M.rate(sum(1 for g in grp if g["top1_agreement"]), len(grp)),
                "top3_containment": M.rate(sum(1 for g in grp if g["top3_containment"]), len(grp)),
                "regret_cp": M.summarize(grp_regrets),
            }
        return out

    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python_version": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "stockfish_version": sf_id.get("name"),
            "stockfish_path": sf_path,
            "stockfish_depth": SF_DEPTH,
            "stockfish_threads": SF_THREADS,
            "stockfish_hash_mb": SF_HASH_MB,
            "stockfish_multipv_agreement": SF_MULTIPV_AGREEMENT,
            "stockfish_multipv_ranking": SF_MULTIPV_RANKING,
            "stockfish_clear_hash_per_position": SF_CLEAR_HASH_PER_POSITION,
        },
        "dataset": {
            "name": dataset.get("name") or dataset.get("description", "")[:80],
            "source_file": dataset.get("_source_file"),
            "count": n,
        },
        "pov_convention": (
            "All evaluations are expressed from the perspective of the side to "
            "move in the ORIGINAL position (the player choosing the move). "
            "Positive centipawns are good for that player."
        ),
        "regret_definition": (
            "move_regret_cp = eval_after_stockfish_move - eval_after_engine_move, "
            "both produced by the same Stockfish configuration and both converted "
            "to the mover's POV. Positive means the engine gave up that many "
            "centipawns. Negative values are retained, not clamped."
        ),
        "metrics": {
            "legality_rate": M.rate(sum(1 for r in rows if r["engine_move_is_legal"]), n),
            "top1_agreement": M.rate(sum(1 for r in rows if r["top1_agreement"]), n),
            "top3_containment": M.rate(sum(1 for r in rows if r["top3_containment"]), n),
            "random_move_baseline": {
                "top1": round(M.random_move_baseline(
                    [r["legal_move_count"] for r in rows], 1) or 0, 4),
                "top3": round(M.random_move_baseline(
                    [r["legal_move_count"] for r in rows], 3) or 0, 4),
                "method": "mean over positions of min(N, legal_moves)/legal_moves",
            },
            "move_regret_cp": M.summarize(regrets),
            "regret_coverage": M.rate(len(regrets), n),
            "blunder_rate": M.blunder_rate(regrets),
            "mate_status_counts": mate_counts,
            "spearman_rho_vs_stockfish_topk": M.summarize(rhos),
            "engine_latency_ms": M.summarize(latencies),
            "stockfish_latency_ms": M.summarize(sf_latencies),
        },
        "breakdowns": {
            "by_category": by_group("category"),
            "by_side_to_move": by_group("side_to_move"),
        },
        "per_position": rows,
    }


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def write_markdown(result: dict, path: Path) -> None:
    m, meta = result["metrics"], result["metadata"]
    L = []
    L.append(f"# Evaluation run - {result['dataset']['name']}\n")
    L.append(f"**{meta['timestamp_utc']}** | commit `{meta['git_commit'][:8]}`"
             f"{' (dirty)' if meta['git_dirty'] else ''} | "
             f"{meta['stockfish_version']} depth {meta['stockfish_depth']}, "
             f"Threads={meta['stockfish_threads']}, Hash={meta['stockfish_hash_mb']}MB\n")
    L.append(f"Positions: **{result['dataset']['count']}**\n")

    L.append("\n## Headline metrics\n")
    L.append("| Metric | Value | n |")
    L.append("|---|---:|---:|")
    for label, key in [("Legality rate", "legality_rate"),
                       ("Top-1 agreement", "top1_agreement"),
                       ("Top-3 containment", "top3_containment")]:
        d = m[key]
        L.append(f"| {label} | {d['percent']}% | {d['numerator']}/{d['denominator']} |")
    r = m["move_regret_cp"]
    L.append(f"| Mean move regret | {r['mean']} cp | {r['n']} |")
    L.append(f"| Median move regret | {r['median']} cp | {r['n']} |")
    L.append(f"| p95 move regret | {r['p95']} cp | {r['n']} |")
    b = m["blunder_rate"]
    L.append(f"| Blunder rate (>{b['threshold_cp']}cp) | "
             f"{round(100 * b['rate'], 2) if b['rate'] is not None else 'n/a'}% | "
             f"{b['blunders']}/{b['denominator']} |")
    lat = m["engine_latency_ms"]
    L.append(f"| Engine latency p50 / p95 | {lat['median']} / {lat['p95']} ms | {lat['n']} |")
    sfl = m["stockfish_latency_ms"]
    L.append(f"| Stockfish latency p50 / p95 | {sfl['median']} / {sfl['p95']} ms | {sfl['n']} |")
    L.append("\n> **Latency is host-dependent and is the one metric here that does "
             "NOT reproduce.** Repeated runs of identical code on the same machine "
             "in a single session produced median engine latencies between roughly "
             "620ms and 1300ms depending on what else the host was doing. Quality "
             "metrics above reproduced exactly across runs; latency did not. Do not "
             "read a latency change between runs or phases as an engine change.\n")

    L.append("\n### Chance reference\n")
    rb = m["random_move_baseline"]
    L.append(f"A uniformly random legal move would agree with Stockfish's top move "
             f"**{round(100 * rb['top1'], 2)}%** of the time and fall in its top 3 "
             f"**{round(100 * rb['top3'], 2)}%** of the time on this suite "
             f"({rb['method']}).\n")

    L.append("\n## Mate handling\n")
    L.append(f"Regret is defined for **{m['regret_coverage']['numerator']}/"
             f"{m['regret_coverage']['denominator']}** positions. Mate scores are "
             f"ordinals, not centipawns, so mate-involved positions are excluded "
             f"from every centipawn aggregate and counted here instead.\n")
    L.append("| Mate status | Count |")
    L.append("|---|---:|")
    for status, count in sorted(m["mate_status_counts"].items(), key=lambda x: -x[1]):
        L.append(f"| `{status}` | {count} |")

    for title, key in [("By category", "by_category"), ("By side to move", "by_side_to_move")]:
        L.append(f"\n## {title}\n")
        L.append("| Group | n | Top-1 | Top-3 | Mean regret | Median regret |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for name, d in result["breakdowns"][key].items():
            rr = d["regret_cp"]
            L.append(f"| {name} | {d['n']} | {d['top1_agreement']['percent']}% | "
                     f"{d['top3_containment']['percent']}% | "
                     f"{rr['mean'] if rr['mean'] is not None else 'n/a'} | "
                     f"{rr['median'] if rr['median'] is not None else 'n/a'} |")
        L.append("\nSmall groups are descriptive only - no significance is implied.\n")

    rho = m["spearman_rho_vs_stockfish_topk"]
    L.append("\n## Rank correlation\n")
    L.append(f"Spearman rho between the engine's ranking and Stockfish's ranking of "
             f"the **same** candidate moves (Stockfish's top "
             f"{meta['stockfish_multipv_ranking']}), computed on ranks only so the "
             f"engine's non-centipawn scale is irrelevant.\n")
    L.append(f"Mean **{rho['mean']}**, median **{rho['median']}**, over {rho['n']} positions.\n")

    L.append("\n## Per-position results\n")
    L.append("| ID | Cat | STM | Engine | Stockfish | Regret cp | Mate status | ms |")
    L.append("|---|---|---|---|---|---:|---|---:|")
    for row in result["per_position"]:
        L.append(f"| {row['id']} | {row['category']} | {row['side_to_move'][:1].upper()} | "
                 f"`{row['engine_move']}` | `{row['stockfish_best_move']}` | "
                 f"{row['move_regret_cp'] if row['move_regret_cp'] is not None else '-'} | "
                 f"{'' if row['mate_status'] == 'none' else row['mate_status']} | "
                 f"{row['engine_latency_ms']:.0f} |")

    L.append("\n---\n")
    L.append(f"POV convention: {result['pov_convention']}\n")
    L.append(f"Regret: {result['regret_definition']}\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-prefix", default=None,
                        help="output path prefix; defaults to evaluation/results/<dataset stem>")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N positions")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["_source_file"] = str(dataset_path.as_posix())
    positions = dataset["positions"]
    if args.limit:
        positions = positions[: args.limit]

    sf_path = project_config.find_stockfish()
    if sf_path is None:
        print("ERROR: no Stockfish binary. Set STOCKFISH_PATH or install stockfish.",
              file=sys.stderr)
        return 2

    print(f"dataset : {dataset_path} ({len(positions)} positions)")
    print(f"stockfish: {sf_path}")
    print(f"config   : depth={SF_DEPTH} Threads={SF_THREADS} Hash={SF_HASH_MB}MB", flush=True)

    import engine as engine_mod  # after chdir; loads TensorFlow + CNN

    started = time.perf_counter()
    rows = []
    with stockfish_session(sf_path) as sf:
        sf_id = dict(sf.id)
        for i, entry in enumerate(positions, 1):
            row = run_position(sf, engine_mod, entry)
            rows.append(row)
            print(f"  {i:3d}/{len(positions)} {row['id']:6s} "
                  f"eng={row['engine_move']:6s} sf={row['stockfish_best_move']:6s} "
                  f"regret={row['move_regret_cp'] if row['move_regret_cp'] is not None else '-':>6} "
                  f"{row['engine_latency_ms']:6.0f}ms", flush=True)

    result = aggregate(rows, dataset, sf_id, sf_path)
    result["metadata"]["wall_clock_seconds"] = round(time.perf_counter() - started, 1)

    prefix = Path(args.out_prefix) if args.out_prefix else \
        REPO_ROOT / "evaluation" / "results" / dataset_path.stem
    prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = prefix.with_suffix(".json")
    # default=float is a safety net for any stray numpy scalar; it must never be
    # needed for a value we care about, so anything it catches is a coercion bug.
    json_path.write_text(json.dumps(result, indent=2, default=float) + "\n",
                         encoding="utf-8")
    md_path = prefix.with_suffix(".md")
    write_markdown(result, md_path)

    m = result["metrics"]
    print(f"\nWROTE {json_path}")
    print(f"WROTE {md_path}")
    print(f"  legality        {m['legality_rate']['percent']}%")
    print(f"  top-1 agreement {m['top1_agreement']['percent']}% "
          f"({m['top1_agreement']['numerator']}/{m['top1_agreement']['denominator']})")
    print(f"  top-3 contain.  {m['top3_containment']['percent']}%")
    print(f"  regret mean/median/p95  {m['move_regret_cp']['mean']} / "
          f"{m['move_regret_cp']['median']} / {m['move_regret_cp']['p95']} cp "
          f"(n={m['move_regret_cp']['n']})")
    print(f"  blunder rate    {m['blunder_rate']}")
    print(f"  latency p50/p95 {m['engine_latency_ms']['median']} / "
          f"{m['engine_latency_ms']['p95']} ms")
    print(f"  wall clock      {result['metadata']['wall_clock_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
