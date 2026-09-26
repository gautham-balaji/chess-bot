"""Phase 0 baseline: run the EXISTING engine over the fixed FEN suite.

Calls engine.engine_move() exactly as app.py does. Nothing in engine.py or
app.py is modified or monkeypatched. Each position is run twice so that
determinism can be observed rather than assumed.

Usage:  python baseline/scripts/measure_engine.py baseline/fens.json baseline/engine_results.json
"""
import json
import os
import statistics
import sys
import time

# engine.py is not an installed package, so the repo root must be on sys.path.
#
# The chdir is retained for belt-and-braces only. When this script was written for
# Phase 0, engine.py loaded "models/cnn_model.keras" relative to the CURRENT
# WORKING DIRECTORY; Phase 1 moved path resolution into config.py, which anchors
# to its own file, so model loading no longer depends on the working directory.
# Corrected in C10 - the original comment had become false.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import chess


def pct(sorted_vals, q):
    """Nearest-rank percentile; explicit so the reported number is unambiguous."""
    if not sorted_vals:
        return None
    k = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return round(sorted_vals[k], 2)


def summarize(vals):
    if not vals:
        return None
    s = sorted(vals)
    return {
        "n": len(s),
        "min_ms": round(s[0], 2),
        "median_ms": round(statistics.median(s), 2),
        "mean_ms": round(statistics.fmean(s), 2),
        "p95_ms": pct(s, 0.95),
        "max_ms": round(s[-1], 2),
        "total_s": round(sum(s) / 1000.0, 2),
    }


def run_once(engine, positions, pass_label):
    rows = []
    for i, p in enumerate(positions, 1):
        board = chess.Board(p["fen"])
        err = None
        move = None
        explanation = None
        top = []
        t0 = time.perf_counter()
        try:
            move, explanation, top3 = engine.engine_move(board)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            for m in top3:
                mv = m["move"]
                top.append({
                    "uci": mv.uci(),
                    "score": round(float(m["score"]), 4),
                    "cnn_cp": round(float(m["cnn_cp"]), 3),
                    "material": int(m["material"]),
                    "space": int(m["space"]),
                    "center": int(m["center"]),
                    "mobility": int(m["mobility"]),
                })
        except Exception as exc:  # noqa: BLE001 - record, do not mask
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err = repr(exc)

        # Legality is re-checked against a FRESH board built from the FEN,
        # so a mutated board inside the engine cannot make an illegal move look legal.
        fresh = chess.Board(p["fen"])
        legal = bool(move is not None and move in fresh.legal_moves)
        fen_preserved = (board.fen() == p["fen"])

        rows.append({
            "id": p["id"],
            "fen": p["fen"],
            "category": p["category"],
            "side_to_move": p["side_to_move"],
            "engine_move_uci": move.uci() if move else None,
            "engine_move_san": (fresh.san(move) if legal else None),
            "is_legal": legal,
            "engine_top_score": top[0]["score"] if top else None,
            "top_candidates": top,
            "explanation": explanation,
            "time_ms": round(elapsed_ms, 2),
            "error": err,
            "board_fen_preserved_after_call": fen_preserved,
        })
        print(f"  [{pass_label}] {i:2d}/{len(positions)} {p['id']} "
              f"{rows[-1]['engine_move_uci']} {rows[-1]['time_ms']:.0f}ms "
              f"legal={legal}", flush=True)
    return rows


def main():
    fens_path, out_path = sys.argv[1], sys.argv[2]
    positions = json.load(open(fens_path, encoding="utf-8"))["positions"]

    t0 = time.perf_counter()
    import engine  # noqa: PLC0415 - timing this import is the point
    import_s = time.perf_counter() - t0
    print(f"import engine: {import_s:.2f}s", flush=True)

    pass1 = run_once(engine, positions, "pass1")
    pass2 = run_once(engine, positions, "pass2")

    # Determinism: compare move AND full top-candidate scores between passes.
    det_rows = []
    for a, b in zip(pass1, pass2):
        same_move = a["engine_move_uci"] == b["engine_move_uci"]
        same_scores = [c["score"] for c in a["top_candidates"]] == \
                      [c["score"] for c in b["top_candidates"]]
        det_rows.append({"id": a["id"], "same_move": same_move,
                         "same_top_scores": same_scores})

    times = [r["time_ms"] for r in pass1 if r["error"] is None]
    legal_n = sum(1 for r in pass1 if r["is_legal"])
    errors = [r for r in pass1 if r["error"]]

    payload = {
        "schema_version": 1,
        "generated_by": "baseline/scripts/measure_engine.py",
        "engine_entry_point": "engine.engine_move(chess.Board(fen))",
        "import_engine_seconds": round(import_s, 2),
        "ridge_coefficients": [float(x) for x in engine.weight_model.coef_],
        "ridge_intercept": float(engine.weight_model.intercept_),
        "ridge_intercept_used_at_inference": False,
        # Stated without line numbers: the previous wording named lines 131-132
        # and 162-163, which had already drifted by C10. Both sites are the
        # `w = weight_model.coef_` assignments in hybrid_score and rerank_moves.
        "ridge_intercept_note": (
            "Both `w = weight_model.coef_` sites (hybrid_score and rerank_moves) "
            "use the coefficients only; weight_model.intercept_ is never added. "
            "Known open defect C3; audited and deliberately retained in C10 - it "
            "is an order-preserving per-position constant, so it changes no move. "
            "See docs/C10_FINAL_QA.md."
        ),
        "position_count": len(positions),
        "latency_summary_pass1_ms": summarize(times),
        "latency_summary_pass2_ms": summarize(
            [r["time_ms"] for r in pass2 if r["error"] is None]),
        "latency_by_category_ms": {
            cat: summarize([r["time_ms"] for r in pass1
                            if r["category"] == cat and r["error"] is None])
            for cat in sorted({p["category"] for p in positions})
        },
        "legality": {
            "legal_moves_returned": legal_n,
            "total": len(pass1),
            "rate": round(legal_n / len(pass1), 4) if pass1 else None,
        },
        "error_count": len(errors),
        "determinism": {
            "method": "same process, two sequential passes over the same 52 FENs",
            "identical_move_count": sum(1 for d in det_rows if d["same_move"]),
            "identical_top_score_count": sum(1 for d in det_rows if d["same_top_scores"]),
            "total": len(det_rows),
            "fully_deterministic": all(d["same_move"] and d["same_top_scores"]
                                       for d in det_rows),
            "caveat": (
                "This shows run-to-run stability within one process. It does not "
                "prove determinism across processes, machines, or TF/BLAS versions."
            ),
            "per_position": det_rows,
        },
        "board_mutation_check": {
            "positions_where_input_fen_preserved": sum(
                1 for r in pass1 if r["board_fen_preserved_after_call"]),
            "total": len(pass1),
        },
        "pass1": pass1,
        "pass2": pass2,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print("WROTE", out_path)
    print("latency pass1:", payload["latency_summary_pass1_ms"])
    print("legality:", payload["legality"])
    print("deterministic:", payload["determinism"]["fully_deterministic"])


if __name__ == "__main__":
    main()
