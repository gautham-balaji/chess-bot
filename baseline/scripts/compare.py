"""Phase 0 baseline: compute ONLY the comparisons that are mathematically valid.

Deliberately NOT computed here:

  average centipawn delta ( engine_score - stockfish_score )
      The engine's score is a Ridge-weighted sum of a tanh-squashed CNN output
      plus raw material / space / centre / mobility counts (engine.py:160-165).
      It is not in centipawns, the Ridge intercept is discarded, and it is not
      anchored to a stated point of view. Stockfish's value is in centipawns.
      Subtracting one from the other is a unit error, so it is omitted rather
      than reported with a caveat.

  move-quality regret (a.k.a. average centipawn loss)
      This IS the right metric, but it requires a SECOND Stockfish evaluation
      of the position after the engine's move. The existing repository never
      does that, so computing it here would exceed "measure what exists".
      Deferred to a later phase; noted as the recommended metric.

Usage:
  python baseline/scripts/compare.py baseline/engine_results.json \
      baseline/stockfish_results.json baseline/comparison.json
"""
import json
import sys
from collections import Counter, defaultdict


def main():
    eng_path, sf_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    eng = json.load(open(eng_path, encoding="utf-8"))
    sf = json.load(open(sf_path, encoding="utf-8"))

    eng_by_id = {r["id"]: r for r in eng["pass1"]}
    fens = json.load(open(sys.argv[4], encoding="utf-8"))["positions"] \
        if len(sys.argv) > 4 else []
    legal_counts = {p["id"]: p["legal_move_count"] for p in fens}

    if not sf.get("available"):
        payload = {"stockfish_available": False,
                   "reason": sf.get("reason"),
                   "engine_only_metrics": {"legality": eng["legality"]}}
        json.dump(payload, open(out_path, "w", encoding="utf-8"), indent=2)
        print("Stockfish unavailable; wrote engine-only metrics.")
        return

    rows = []
    for s in sf["results"]:
        e = eng_by_id.get(s["id"])
        if not e:
            continue
        sf_top_n = [t["uci"] for t in s["top_n"] if t["uci"]]
        rows.append({
            "id": s["id"],
            "category": s["category"],
            "side_to_move": s["side_to_move"],
            "engine_move": e["engine_move_uci"],
            "engine_legal": e["is_legal"],
            "stockfish_move": s["best_move_uci"],
            "top1_agree": e["engine_move_uci"] == s["best_move_uci"],
            "in_stockfish_top3": e["engine_move_uci"] in sf_top_n,
            "stockfish_top3": sf_top_n,
            "stockfish_is_mate_score": s["is_mate_score"],
            "legal_move_count": legal_counts.get(s["id"]),
            "engine_time_ms": e["time_ms"],
            "stockfish_time_ms": s["play_time_ms"],
        })

    n = len(rows)
    agree = sum(1 for r in rows if r["top1_agree"])
    top3 = sum(1 for r in rows if r["in_stockfish_top3"])
    legal = sum(1 for r in rows if r["engine_legal"])

    by_cat = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    by_side = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    for r in rows:
        for bucket, key in ((by_cat, r["category"]), (by_side, r["side_to_move"])):
            bucket[key]["n"] += 1
            bucket[key]["top1"] += int(r["top1_agree"])
            bucket[key]["top3"] += int(r["in_stockfish_top3"])

    def rates(d):
        return {k: {"n": v["n"],
                    "top1_agreement": round(v["top1"] / v["n"], 4) if v["n"] else None,
                    "top3_containment": round(v["top3"] / v["n"], 4) if v["n"] else None}
                for k, v in sorted(d.items())}

    eng_times = [r["engine_time_ms"] for r in rows]
    sf_times = [r["stockfish_time_ms"] for r in rows]

    payload = {
        "schema_version": 1,
        "generated_by": "baseline/scripts/compare.py",
        "position_count": n,
        "stockfish_depth": sf["depth"],
        "valid_metrics": {
            "engine_legality_rate": {
                "value": round(legal / n, 4) if n else None,
                "legal": legal, "total": n,
                "definition": "fraction of engine moves that are legal in the "
                              "position, re-checked on a fresh board from the FEN",
            },
            "top1_move_agreement": {
                "value": round(agree / n, 4) if n else None,
                "agree": agree, "total": n,
                "definition": "fraction of positions where the engine's chosen move "
                              "equals Stockfish's chosen move at depth 8. Compares "
                              "MOVES, so it is unit-free and POV-free.",
            },
            "top3_containment": {
                "value": round(top3 / n, 4) if n else None,
                "hits": top3, "total": n,
                "definition": "fraction of positions where the engine's move appears "
                              "in Stockfish's MultiPV=3 list at depth 8.",
            },
            "latency_ratio_engine_over_stockfish": {
                "value": round(sum(eng_times) / sum(sf_times), 2) if sum(sf_times) else None,
                "engine_total_ms": round(sum(eng_times), 1),
                "stockfish_total_ms": round(sum(sf_times), 1),
                "definition": "total engine wall time divided by total Stockfish "
                              "wall time over the same suite. >1 means the engine "
                              "is SLOWER. Same machine, same positions, sequential.",
            },
        },
        "chance_baseline": {
            "note": (
                "Agreement rates are meaningless without a chance reference. A "
                "uniform-random legal move agrees with Stockfish's top move with "
                "probability 1/legal_move_count. Positions with few legal moves "
                "(endgames) inflate agreement, so raw rates must not be compared "
                "across categories without this."
            ),
            "expected_top1_agreement_if_random": round(
                sum(1.0 / r["legal_move_count"] for r in rows if r["legal_move_count"]) / n, 4
            ) if n and all(r["legal_move_count"] for r in rows) else None,
            "expected_top3_containment_if_random": round(
                sum(min(3, r["legal_move_count"]) / r["legal_move_count"]
                    for r in rows if r["legal_move_count"]) / n, 4
            ) if n and all(r["legal_move_count"] for r in rows) else None,
            "by_category_expected_top1_if_random": {
                cat: round(
                    sum(1.0 / r["legal_move_count"] for r in rows
                        if r["category"] == cat and r["legal_move_count"])
                    / max(1, sum(1 for r in rows if r["category"] == cat)), 4)
                for cat in sorted({r["category"] for r in rows})
            },
            "mean_legal_moves": round(
                sum(r["legal_move_count"] for r in rows if r["legal_move_count"]) / n, 2
            ) if n else None,
        },
        "breakdowns": {"by_category": rates(by_cat), "by_side_to_move": rates(by_side)},
        "deliberately_not_computed": {
            "average_centipawn_delta": (
                "INVALID for this implementation. engine score is not in centipawns "
                "(tanh-squashed CNN term weighted 330.9 plus raw feature counts, "
                "engine.py:160-165), the Ridge intercept is dropped, and the POV is "
                "unstated. app.py:411 computes exactly this as 'eval_diff_cp'; that "
                "field should not be treated as a quality measure."
            ),
            "move_quality_regret": (
                "The CORRECT metric, but not computed in Phase 0: it needs a second "
                "Stockfish evaluation of the post-engine-move position, which the "
                "existing repository never performs. Recommended for a later phase."
            ),
        },
        "caveats": [
            "Top-1 agreement against a depth-8 reference is a weak notion of quality: "
            "many positions have several near-equal good moves, and a disagreement is "
            "not necessarily an error.",
            "Stockfish ran with default UCI options (Threads/Hash not pinned), so the "
            "reference moves are not guaranteed bit-reproducible across runs.",
            "The 52-position suite is a fixed comparison set, not a statistically "
            "representative sample; rates carry no confidence interval.",
        ],
        "per_position": rows,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print("WROTE", out_path)
    for k, v in payload["valid_metrics"].items():
        print(f"  {k}: {v['value']}")
    print("  by side:", rates(by_side))
    print("  mate-score positions:", Counter(r["stockfish_is_mate_score"] for r in rows))


if __name__ == "__main__":
    main()
