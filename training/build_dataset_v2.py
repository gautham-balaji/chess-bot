"""C8 dataset builder: `dataset_v2`, the game-split expansion of dataset_v1.

    python -m training.build_dataset_v2 --out training/artifacts/dataset_v2

Writes `<out>.train.jsonl`, `<out>.test.jsonl` and `<out>.manifest.json`.
Trains nothing and never writes to `models/`, `engine.py` or the evaluation
suites.

--------------------------------------------------------------------------
WHAT THIS CHANGES RELATIVE TO dataset_v1, AND WHAT IT DOES NOT
--------------------------------------------------------------------------
CHANGES: only which positions are in the training data.

  dataset_v1   10,000 sampled games -> ONE position per game at <=20 plies
               -> 9,667 records, position-level 80/20 split
  dataset_v2   18,920 unique games -> GAME-LEVEL 80/20 split FIRST
               -> up to 4 evenly spaced positions per game from ply 16
               -> ~68.9k records, placement-deduplicated, leakage-scrubbed

UNCHANGED: the label policy (`training/labels.py`, A2's
`corrected_mate_white_perspective`), the Stockfish configuration, the model
architecture, the 12-plane representation, the evaluator, the Ridge fusion and
every production file. This builder imports the label policy rather than
reimplementing it, so it cannot drift.

--------------------------------------------------------------------------
THE PIPELINE ORDER IS THE POINT
--------------------------------------------------------------------------
    games.csv rows
      -> group by game CONTENT (sha256 of the moves string)   [18,920 units]
      -> seeded split over sorted units                       [seed 42, 20% test]
      -> extract positions from each side INDEPENDENTLY
      -> deduplicate within each side at PLACEMENT level
      -> scrub: drop from TEST any placement present in TRAIN
      -> scrub: drop from BOTH any placement in either evaluation suite
      -> label the survivors with the A2 policy

Splitting before extraction is what makes `train_games n test_games = 0` a
statement about game content rather than about row indices. Labelling last means
the expensive Stockfish pass runs only over records that actually survive.

The split unit is the move string, NOT the `id` column: 813 ids appear on more
than one row (verified exact duplicates), and 50 move sequences are filed under
243 different ids. Keying on `id` would let identical game content straddle the
split. See docs/C7_DATASET_EXPANSION_DESIGN.md section 1.2.

--------------------------------------------------------------------------
WHY PLACEMENT-LEVEL DEDUPLICATION
--------------------------------------------------------------------------
`dataset_v1` deduplicates on the exact FEN. The 12-plane encoder reads ONLY the
piece-placement field, so two records with the same placement are literally the
same CNN input whatever their side to move, castling rights or move counters.
Placement is therefore the level at which train/test overlap actually leaks into
this model, and it is the level used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import chess  # noqa: E402
import pandas as pd  # noqa: E402

from training import build_dataset as B  # noqa: E402
from training import labels as L  # noqa: E402
from training import c7_dataset_audit as C7  # noqa: E402

PIPELINE_VERSION = "c8-1"

DEFAULT_SOURCE = REPO_ROOT / "games.csv"
DEFAULT_OUT = REPO_ROOT / "training" / "artifacts" / "dataset_v2"
SUITE_FILES = ("extended", "phase0_52")

# --- the C7 decision, implemented verbatim -----------------------------------
SPLIT_SEED = 42
TEST_FRACTION = 0.20
POLICY_NAME = "evenly_spaced_4_minply16"
MIN_PLY = 16
MAX_PER_GAME = 4
MIN_GAP = 4
SAMPLING_SEED = None          # the policy is closed-form; there is no RNG

# Counts C7 measured for this exact policy. Recorded in the manifest and checked
# by --expect so a silent drift in pandas, python-chess or the source file is
# caught at build time rather than in the C8 results.
C7_EXPECTED = {
    "game_units": 18920,
    "train_units": 15136,
    "test_units": 3784,
    "selected_total": 68901,
    "selected_train": 55082,
    "selected_test": 13819,
    "deduped_train": 54813,
    "deduped_test": 13800,
    "train_test_overlap_placements": 88,
    "final_test": 13712,
    "game_list_sha256":
        "b90881bf3c6245ba3634800da8879ccf8ecb9a64fb41f5116c5439983dc64e62",
    "train_game_sha256":
        "cb1308d9af1ee45709f272492c0a22a02b8e7026f055d8ff43e1d81f881c19fb",
    "test_game_sha256":
        "d8632c3ac89be36bced244f4370eec09c3c80b2e6a0fe841f62ca67df97809f3",
}

RECORD_KEYS = [
    "game_content_key", "game_id", "source_row_index", "ply", "fen",
    "placement", "side_to_move", "piece_count", "phase", "plies_in_game",
    "is_checkmate", "is_stalemate", "is_game_over", "legal_move_count",
    "eval_type", "raw_stockfish_value", "raw_value_perspective", "label",
    "label_perspective", "was_clipped", "mate_zero_consistent",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================ extraction

def extract_side(df: pd.DataFrame, row_indices: set, unit_keys: list) -> list[dict]:
    """Replay each game unit once and select its positions.

    One canonical row per game unit: `games.csv` holds 1,138 exact duplicate
    rows, which would otherwise contribute the same positions two to five times.
    Records are emitted in a canonical (game_content_key, ply) order so that
    downstream "keep the first occurrence" deduplication does not depend on row
    order in the CSV.
    """
    seen_units, rows = set(), []
    for i in sorted(row_indices):
        key = unit_keys[i]
        if key in seen_units:
            continue
        seen_units.add(key)

        row = df.iloc[i]
        moves = row.get("moves")
        rep = C7.replay_game(row.get("id"), i, moves)
        picks = C7.policy_evenly_spaced(rep, MAX_PER_GAME, min_ply=MIN_PLY,
                                        min_gap=MIN_GAP)
        if not picks:
            continue

        # Replay once more, emitting a board only at the selected plies. The
        # audit's GameReplay stores digests, not boards, so the FEN is rebuilt
        # here from the same token stream.
        board = chess.Board()
        tokens = str(moves).strip().split()
        wanted = set(picks)
        for idx, token in enumerate(tokens):
            try:
                board.push_san(token)
            except ValueError:
                break
            if idx not in wanted:
                continue
            fen = board.fen()
            rows.append({
                "game_content_key": key,
                "game_id": row.get("id"),
                "source_row_index": int(i),
                "ply": idx + 1,
                "fen": fen,
                "placement": fen.split(" ")[0],
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "piece_count": chess.popcount(board.occupied),
                "phase": C7.classify_phase(idx + 1, chess.popcount(board.occupied)),
                "plies_in_game": rep.n_plies,
                "is_checkmate": board.is_checkmate(),
                "is_stalemate": board.is_stalemate(),
                "is_game_over": board.is_game_over(),
                "legal_move_count": board.legal_moves.count(),
            })

    rows.sort(key=lambda r: (r["game_content_key"], r["ply"]))
    return rows


def deduplicate_by_placement(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep the first record of each placement, in canonical order."""
    seen, kept = set(), []
    for row in rows:
        if row["placement"] in seen:
            continue
        seen.add(row["placement"])
        kept.append(row)
    return kept, {
        "level": "fen piece-placement field",
        "input_rows": len(rows),
        "unique_placements": len(kept),
        "duplicate_rows_removed": len(rows) - len(kept),
    }


def load_suite_placements() -> tuple[set, dict]:
    """Piece placements of every evaluation-suite position. Read-only."""
    placements, counts = set(), {}
    for suite in SUITE_FILES:
        path = REPO_ROOT / "evaluation" / "positions" / f"{suite}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        fens = [p["fen"] for p in data["positions"]]
        counts[suite] = len(fens)
        placements.update(f.split(" ")[0] for f in fens)
    return placements, counts


def scrub(rows: list[dict], banned: set, reason: str) -> tuple[list[dict], dict]:
    """Drop every record whose placement is in `banned`, recording what went."""
    kept, removed = [], []
    for row in rows:
        (removed if row["placement"] in banned else kept).append(row)
    return kept, {
        "reason": reason,
        "records_removed": len(removed),
        "distinct_placements_removed": len({r["placement"] for r in removed}),
        "removed_placement_sha256": sorted(
            sha256_text(p) for p in {r["placement"] for r in removed}),
    }


# ============================================================ labelling

def label_rows(engine, rows: list[dict], cp_clip: int, progress_every: int) -> dict:
    """Attach the A2 label to every record, in place.

    Identical policy call to `build_dataset.label_positions`: the hash is cleared
    before each position so labels never depend on iteration order, and
    `training/labels.py` does the mate mapping, perspective and clipping.
    """
    stats, anomalies = Counter(), []
    for i, row in enumerate(rows, 1):
        if B.SF_CLEAR_HASH_PER_POSITION:
            engine.send_ucinewgame_command()
        engine.set_fen_position(row["fen"])
        evaluation = engine.get_evaluation()
        eval_type, raw_value = evaluation["type"], int(evaluation["value"])

        label = L.make_label(
            eval_type=eval_type,
            raw_value=raw_value,
            side_to_move_is_white=(row["side_to_move"] == "white"),
            is_checkmate=row["is_checkmate"],
            cp_clip=cp_clip,
        )
        row.update({
            "eval_type": label.eval_type,
            "raw_stockfish_value": label.raw_value,
            "raw_value_perspective": "side_to_move",
            "label": label.label,
            "label_perspective": L.LABEL_PERSPECTIVE,
            "was_clipped": label.was_clipped,
        })
        if label.mate_zero_consistent is not None:
            row["mate_zero_consistent"] = label.mate_zero_consistent
            if not label.mate_zero_consistent:
                stats["mate_zero_inconsistent"] += 1
                anomalies.append({"fen": row["fen"],
                                  "reason": "mate==0 but board is not checkmate"})
        stats[f"eval_type_{eval_type}"] += 1
        if label.was_clipped:
            stats["clipped"] += 1
        if progress_every and i % progress_every == 0:
            print(f"    labelled {i}/{len(rows)}", flush=True)
    return {"counts": dict(stats), "anomalies": anomalies}


# ============================================================ artifacts

def write_records(rows: list[dict], path: Path) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps({k: row[k] for k in RECORD_KEYS if k in row},
                                sort_keys=False) + "\n")
    return B.sha256_file(path)


def label_hash(rows: list[dict]) -> str:
    """Hash of the label vector alone, in file order."""
    return sha256_text("\n".join(str(r["label"]) for r in rows))


def describe_side(rows: list[dict]) -> dict:
    # `label` is absent under --skip-labels, which builds the record set without
    # calling Stockfish. Everything structural is still reported.
    values = [r["label"] for r in rows if "label" in r]
    return {
        "records": len(rows),
        "distinct_games": len({r["game_content_key"] for r in rows}),
        "side_to_move_counts": dict(Counter(r["side_to_move"] for r in rows)),
        "phase_counts": dict(Counter(r["phase"] for r in rows)),
        "eval_type_counts": dict(Counter(r["eval_type"] for r in rows
                                         if "eval_type" in r)),
        "clipped": sum(1 for r in rows if r.get("was_clipped")),
        "checkmate_positions": sum(1 for r in rows if r["is_checkmate"]),
        "ply": {
            "min": min(r["ply"] for r in rows),
            "max": max(r["ply"] for r in rows),
            "mean": round(sum(r["ply"] for r in rows) / len(rows), 3),
        },
        "piece_count": {
            "min": min(r["piece_count"] for r in rows),
            "max": max(r["piece_count"] for r in rows),
        },
        "label": {
            "min": min(values), "max": max(values),
            "mean": round(sum(values) / len(values), 3),
        } if values else None,
    }


# ============================================================ build

def build(source: Path, out_prefix: Path, cp_clip: int, progress_every: int,
          skip_labels: bool, expect: bool) -> dict:
    started = time.perf_counter()

    # ---------------------------------------------------------------- source
    df = pd.read_csv(source)
    source_sha = B.sha256_file(source)
    print(f"source     : {source.name} rows={len(df)} sha256={source_sha[:16]}...")
    if source_sha != B.EXPECTED_SOURCE["sha256"]:
        raise SystemExit(
            f"ERROR: games.csv does not match the audited identity.\n"
            f"       expected {B.EXPECTED_SOURCE['sha256']}\n"
            f"       actual   {source_sha}\n"
            f"       The C7 split hashes are only valid for the audited file.")

    ids = df["id"].astype(str).tolist()
    moves = df["moves"].astype(str).tolist()
    unit_keys = C7.game_unit_keys(ids, moves)

    # ---------------------------------------------------------------- split
    train_rows_idx, test_rows_idx, train_units, test_units = C7.game_level_split(
        unit_keys, SPLIT_SEED, TEST_FRACTION)
    if train_units & test_units:
        raise SystemExit("ERROR: game split overlaps by content")

    split_hashes = {
        "game_list_sha256": sha256_text("\n".join(sorted(set(unit_keys)))),
        "train_game_sha256": sha256_text("\n".join(sorted(train_units))),
        "test_game_sha256": sha256_text("\n".join(sorted(test_units))),
    }
    print(f"split      : {len(train_units)} train / {len(test_units)} test game units "
          f"(seed {SPLIT_SEED}, test_fraction {TEST_FRACTION})")
    for name, value in split_hashes.items():
        match = " OK" if value == C7_EXPECTED[name] else " *** DIFFERS FROM C7 ***"
        print(f"  {name:20s} {value[:16]}...{match}")

    # ---------------------------------------------------------------- extract
    print(f"extract    : {POLICY_NAME} (min_ply={MIN_PLY}, max={MAX_PER_GAME}, "
          f"min_gap={MIN_GAP}, no RNG)")
    train = extract_side(df, train_rows_idx, unit_keys)
    test = extract_side(df, test_rows_idx, unit_keys)
    selected = {"train": len(train), "test": len(test),
                "total": len(train) + len(test)}
    print(f"  selected : {selected['total']} "
          f"({selected['train']} train / {selected['test']} test)")

    # ---------------------------------------------------------------- dedup
    train, dedup_train = deduplicate_by_placement(train)
    test, dedup_test = deduplicate_by_placement(test)
    print(f"  deduped  : {len(train)} train / {len(test)} test "
          f"(placement level; removed {dedup_train['duplicate_rows_removed']} / "
          f"{dedup_test['duplicate_rows_removed']})")

    # ---------------------------------------------------------------- leakage
    train_placements = {r["placement"] for r in train}
    overlap = train_placements & {r["placement"] for r in test}
    test, leak_scrub = scrub(test, overlap,
                             "placement also present in the train split")
    print(f"  leakage  : {len(overlap)} placements in both; removed "
          f"{leak_scrub['records_removed']} TEST records -> {len(test)} test")
    if {r["placement"] for r in train} & {r["placement"] for r in test}:
        raise SystemExit("ERROR: train/test placement overlap survived the scrub")

    # ---------------------------------------------------------------- suites
    suite_placements, suite_counts = load_suite_placements()
    train, suite_train = scrub(train, suite_placements,
                               "placement appears in an evaluation suite")
    test, suite_test = scrub(test, suite_placements,
                             "placement appears in an evaluation suite")
    print(f"  suites   : removed {suite_train['records_removed']} train / "
          f"{suite_test['records_removed']} test records "
          f"({sum(suite_counts.values())} suite positions, "
          f"{len(suite_placements)} distinct placements)")
    for side, rows in (("train", train), ("test", test)):
        if {r["placement"] for r in rows} & suite_placements:
            raise SystemExit(f"ERROR: suite overlap survived in {side}")

    print(f"  FINAL    : {len(train)} train / {len(test)} test")

    # ---------------------------------------------------------------- labels
    engine_info = None
    if skip_labels:
        print("labels     : SKIPPED (--skip-labels); records carry no label")
        label_stats = {"counts": {}, "anomalies": [], "skipped": True}
    else:
        engine, engine_info, _ = B.open_engine()
        print(f"stockfish  : {engine_info.version}")
        print(f"  depth={B.SF_DEPTH} Threads={B.SF_THREADS} Hash={B.SF_HASH_MB}MB "
              f"clear_hash={B.SF_CLEAR_HASH_PER_POSITION}")
        try:
            stats_train = label_rows(engine, train, cp_clip, progress_every)
            stats_test = label_rows(engine, test, cp_clip, progress_every)
        finally:
            engine.send_quit_command()
        merged = Counter(stats_train["counts"]) + Counter(stats_test["counts"])
        label_stats = {
            "counts": dict(merged),
            "train_counts": stats_train["counts"],
            "test_counts": stats_test["counts"],
            "anomalies": stats_train["anomalies"] + stats_test["anomalies"],
            "skipped": False,
        }

    # ---------------------------------------------------------------- write
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    train_path = Path(f"{out_prefix}.train.jsonl")
    test_path = Path(f"{out_prefix}.test.jsonl")
    train_sha = write_records(train, train_path)
    test_sha = write_records(test, test_path)

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "dataset_name": "dataset_v2",
        "purpose": (
            "C8 training dataset. Changes ONLY which positions are used: the "
            "architecture, representation, label policy, evaluator, Ridge and "
            "every production file are untouched."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generation_seconds": round(time.perf_counter() - started, 1),
        "timestamp_note": (
            "generated_at_utc and generation_seconds are informational only and "
            "are NOT inputs to any hash in this manifest."
        ),
        "source": {
            "filename": source.name,
            "sha256": source_sha,
            "size_bytes": source.stat().st_size,
            "row_count": len(df),
            "matches_audited_identity": source_sha == B.EXPECTED_SOURCE["sha256"],
        },
        "game_split": {
            "split_unit": "game move-sequence content (sha256 of the moves string)",
            "why_not_id": (
                "813 ids appear on multiple rows (verified exact duplicates) and "
                "50 move sequences are filed under 243 different ids; keying on "
                "id would let identical content straddle the split"
            ),
            "split_seed": SPLIT_SEED,
            "test_fraction": TEST_FRACTION,
            "row_order_invariant": True,
            "n_rows": len(df),
            "n_game_units": len(set(unit_keys)),
            "n_train_units": len(train_units),
            "n_test_units": len(test_units),
            "train_units_intersect_test_units": 0,
            **split_hashes,
        },
        "extraction": {
            "policy": POLICY_NAME,
            "min_ply": MIN_PLY,
            "max_positions_per_game": MAX_PER_GAME,
            "min_gap_plies": MIN_GAP,
            "sampling_seed": SAMPLING_SEED,
            "uses_rng": False,
            "description": (
                "at most 4 positions per game, evenly spaced from ply 16 to the "
                "final ply, never closer than 4 plies apart; closed-form, no RNG"
            ),
            "one_canonical_row_per_game_unit": True,
            "record_order": "sorted by (game_content_key, ply)",
            "selected_counts": selected,
        },
        "deduplication": {
            "level": "fen piece-placement field",
            "rationale": (
                "the 12-plane encoder reads only piece squares, so identical "
                "placements are identical CNN inputs"
            ),
            "train": dedup_train,
            "test": dedup_test,
        },
        "leakage_scrub": {
            "train_test_overlap_placements": len(overlap),
            **leak_scrub,
            "policy": "remove from TEST only; the train split is never reduced "
                      "to resolve train/test overlap",
            "final_train_test_placement_overlap": 0,
        },
        "evaluation_suite_scrub": {
            "suites": suite_counts,
            "distinct_suite_placements": len(suite_placements),
            "train": suite_train,
            "test": suite_test,
            "final_suite_overlap_train": 0,
            "final_suite_overlap_test": 0,
            "suites_modified": False,
        },
        "final": {
            "train": describe_side(train) if train else {},
            "test": describe_side(test) if test else {},
            "total_records": len(train) + len(test),
        },
        "labels": {
            **L.policy_summary(),
            "policy_name": "corrected_mate_white_perspective",
            "policy_source": "training/labels.py (unmodified)",
            "cp_clip_used": cp_clip,
            "counts": label_stats["counts"],
            "anomalies": label_stats["anomalies"],
            "skipped": label_stats["skipped"],
            "train_label_sha256": label_hash(train) if not skip_labels else None,
            "test_label_sha256": label_hash(test) if not skip_labels else None,
        },
        "stockfish": None if engine_info is None else {
            "version": engine_info.version,
            "wrapper_package": engine_info.wrapper_package,
            "wrapper_version": engine_info.wrapper_version,
            "depth": engine_info.depth,
            "threads": engine_info.threads,
            "hash_mb": engine_info.hash_mb,
            "clear_hash_per_position": engine_info.clear_hash_per_position,
            "turn_perspective_used": engine_info.turn_perspective_used,
        },
        "artifact": {
            "train_file": train_path.name,
            "test_file": test_path.name,
            "train_sha256": train_sha,
            "test_sha256": test_sha,
            "records_format": "JSON Lines (one position per line, UTF-8, LF)",
            "record_keys": RECORD_KEYS,
        },
        "c7_expected": C7_EXPECTED,
        "reproducibility": {
            "python_version": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "packages": {
                "pandas": _version("pandas"),
                "python-chess": _version("chess"),
                "stockfish": _version("stockfish"),
                "numpy": _version("numpy"),
            },
            "git_commit": B._git("rev-parse", "HEAD"),
            "git_dirty": bool(B._git("status", "--porcelain")),
            "determinism": (
                "Deterministic for a fixed games.csv, Stockfish binary/version, "
                "configuration and pipeline version. The extraction policy uses "
                "no RNG; the only seed is the game split seed."
            ),
        },
    }

    manifest_path = Path(f"{out_prefix}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nWROTE {train_path.name}  {train_sha[:16]}...")
    print(f"WROTE {test_path.name}   {test_sha[:16]}...")
    print(f"WROTE {manifest_path.name}")

    if expect:
        check_expectations(manifest)
    return manifest


def _version(pkg):
    import importlib.metadata as md
    try:
        return md.version(pkg)
    except Exception:  # noqa: BLE001
        return None


def check_expectations(manifest: dict) -> None:
    """Compare the build against C7's measured values and fail loudly on drift."""
    gs, ex = manifest["game_split"], C7_EXPECTED
    got = {
        "game_units": gs["n_game_units"],
        "train_units": gs["n_train_units"],
        "test_units": gs["n_test_units"],
        "selected_total": manifest["extraction"]["selected_counts"]["total"],
        "selected_train": manifest["extraction"]["selected_counts"]["train"],
        "selected_test": manifest["extraction"]["selected_counts"]["test"],
        "deduped_train": manifest["deduplication"]["train"]["unique_placements"],
        "deduped_test": manifest["deduplication"]["test"]["unique_placements"],
        "train_test_overlap_placements":
            manifest["leakage_scrub"]["train_test_overlap_placements"],
        "game_list_sha256": gs["game_list_sha256"],
        "train_game_sha256": gs["train_game_sha256"],
        "test_game_sha256": gs["test_game_sha256"],
    }
    print(f"\n{'check':34s}{'expected (C7)':>20s}{'actual':>20s}  status")
    bad = []
    for key, want in ex.items():
        if key not in got:
            continue
        have = got[key]
        ok = have == want
        if not ok:
            bad.append(key)
        w = str(want)[:18] + ("..." if len(str(want)) > 18 else "")
        h = str(have)[:18] + ("..." if len(str(have)) > 18 else "")
        print(f"  {key:32s}{w:>20s}{h:>20s}  {'OK' if ok else 'MISMATCH'}")
    if bad:
        raise SystemExit(f"ERROR: build differs from C7 measurements: {bad}")
    print("  all C7 expectations reproduced")


# ============================================================ cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="prefix; writes <out>.train.jsonl, <out>.test.jsonl, "
                         "<out>.manifest.json")
    ap.add_argument("--cp-clip", type=int, default=L.CP_CLIP)
    ap.add_argument("--progress-every", type=int, default=5000)
    ap.add_argument("--skip-labels", action="store_true",
                    help="build the record set without calling Stockfish "
                         "(structure checks only; produces an unlabelled dataset)")
    ap.add_argument("--no-expect", action="store_true",
                    help="do not check the build against C7's measured counts")
    args = ap.parse_args(argv)

    build(args.source, args.out, args.cp_clip, args.progress_every,
          args.skip_labels, expect=not args.no_expect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
