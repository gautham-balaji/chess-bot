"""Phase 1 behaviour-preservation check.

Compares post-Phase-1 engine output against the frozen Phase 0 baseline over the
same 52 FENs. Phase 1 was a reproducibility/integrity pass: it must not change
a single selected move or score.

Reads (never writes) baseline/engine_results.json.

Usage:
  python verification/compare_phase0_phase1.py \
      baseline/engine_results.json \
      verification/phase1_engine_results.json \
      verification/behaviour_diff.json
"""
import json
import sys


def main():
    before_path, after_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    before = json.load(open(before_path, encoding="utf-8"))
    after = json.load(open(after_path, encoding="utf-8"))

    b_by_id = {r["id"]: r for r in before["pass1"]}
    a_by_id = {r["id"]: r for r in after["pass1"]}

    ids = sorted(set(b_by_id) | set(a_by_id))
    move_diffs, score_diffs, top3_diffs, legality_diffs, missing = [], [], [], [], []

    for pid in ids:
        b, a = b_by_id.get(pid), a_by_id.get(pid)
        if not b or not a:
            missing.append(pid)
            continue

        if b["engine_move_uci"] != a["engine_move_uci"]:
            move_diffs.append({"id": pid, "phase0": b["engine_move_uci"],
                               "phase1": a["engine_move_uci"]})

        if b["is_legal"] != a["is_legal"]:
            legality_diffs.append({"id": pid, "phase0": b["is_legal"],
                                   "phase1": a["is_legal"]})

        b_top = [(c["uci"], c["score"]) for c in b["top_candidates"]]
        a_top = [(c["uci"], c["score"]) for c in a["top_candidates"]]
        if b_top != a_top:
            top3_diffs.append({"id": pid, "phase0": b_top, "phase1": a_top})

        if b["engine_top_score"] != a["engine_top_score"]:
            score_diffs.append({
                "id": pid,
                "phase0": b["engine_top_score"],
                "phase1": a["engine_top_score"],
                "delta": (None if None in (b["engine_top_score"], a["engine_top_score"])
                          else round(a["engine_top_score"] - b["engine_top_score"], 6)),
            })

    n = len([p for p in ids if p in b_by_id and p in a_by_id])
    preserved = not (move_diffs or score_diffs or top3_diffs or legality_diffs or missing)

    payload = {
        "positions_compared": n,
        "behaviour_preserved": preserved,
        "summary": {
            "selected_move_unchanged": f"{n - len(move_diffs)}/{n}",
            "top3_uci_and_scores_unchanged": f"{n - len(top3_diffs)}/{n}",
            "top_score_unchanged": f"{n - len(score_diffs)}/{n}",
            "legality_unchanged": f"{n - len(legality_diffs)}/{n}",
        },
        "phase0": {
            "legality_rate": before["legality"]["rate"],
            "deterministic": before["determinism"]["fully_deterministic"],
            "mean_ms": before["latency_summary_pass1_ms"]["mean_ms"],
            "median_ms": before["latency_summary_pass1_ms"]["median_ms"],
            "import_engine_seconds": before["import_engine_seconds"],
            "ridge_coefficients": before["ridge_coefficients"],
        },
        "phase1": {
            "legality_rate": after["legality"]["rate"],
            "deterministic": after["determinism"]["fully_deterministic"],
            "mean_ms": after["latency_summary_pass1_ms"]["mean_ms"],
            "median_ms": after["latency_summary_pass1_ms"]["median_ms"],
            "import_engine_seconds": after["import_engine_seconds"],
            "ridge_coefficients": after["ridge_coefficients"],
        },
        "ridge_coefficients_identical":
            before["ridge_coefficients"] == after["ridge_coefficients"],
        "differences": {
            "selected_move": move_diffs,
            "top_score": score_diffs,
            "top3_ordering_or_scores": top3_diffs,
            "legality": legality_diffs,
            "missing_positions": missing,
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print("WROTE", out_path)
    print("behaviour_preserved:", preserved)
    for k, v in payload["summary"].items():
        print(f"  {k}: {v}")
    print("  ridge coefficients identical:", payload["ridge_coefficients_identical"])
    if not preserved:
        print("\n!!! DIFFERENCES DETECTED - investigate, do NOT alter the engine to mask them")
        print(json.dumps(payload["differences"], indent=2)[:3000])
    return 0 if preserved else 1


if __name__ == "__main__":
    raise SystemExit(main())
