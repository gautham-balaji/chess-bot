"""Reproducible training-dataset and label builder (C6-Prep).

Extracts the dataset/position/label generation that previously existed only as
cells in `chess_model_FINAL.ipynb`, so the later A0-A3 experiments can be run
fairly. It does NOT train anything and does not touch any model artifact.

    python training/build_dataset.py --out training/artifacts/dataset_v1

writes two files:

    <out>.jsonl     one inspectable record per position
    <out>.manifest.json   full provenance and policy

Deterministic for a fixed games.csv, Stockfish binary/version, configuration,
seed and pipeline version.

--------------------------------------------------------------------------
DIFFERENCES FROM THE NOTEBOOK  (all deliberate; see training/README.md)
--------------------------------------------------------------------------
  * labels are WHITE-POSITIVE, not side-to-move-relative
  * mate scores are mapped onto the centipawn axis instead of being stored as
    raw mate distances
  * Stockfish is pinned to Threads=1, Hash=16 and the hash is CLEARED before
    every position, so labels no longer depend on iteration order
  * positions are deduplicated by exact FEN
  * malformed SAN and invalid boards are counted and recorded, not swallowed
  * the Stockfish path comes from config.find_stockfish(), not a hardcoded one

PRESERVED unchanged: the population, `df.sample(10000, random_state=42)`, and
the `tokens[:20]` position derivation (at MOST 20 plies, not exactly 20).
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import chess  # noqa: E402
import pandas as pd  # noqa: E402

import config as project_config  # noqa: E402
from training import labels as L  # noqa: E402

PIPELINE_VERSION = "c6prep-1"

# --- preserved from the notebook ----------------------------------------------
SAMPLE_SIZE = 10_000
SAMPLE_SEED = 42
MAX_FULLMOVES = 10                 # board_after_fullmove(moves, 10) -> tokens[:20]
MAX_PLIES = MAX_FULLMOVES * 2

# --- Phase 3 reference Stockfish configuration --------------------------------
SF_DEPTH = 8
SF_THREADS = 1
SF_HASH_MB = 16
SF_CLEAR_HASH_PER_POSITION = True

# --- dataset identity recorded by the audit -----------------------------------
EXPECTED_SOURCE = {
    "filename": "games.csv",
    "sha256": "e7aadff104a610afb5403caf81c1461babecb0dc87760ff5eea7dc0e7a4a8129",
    "size_bytes": 7_672_655,
    "row_count": 20_058,
}
REQUIRED_COLUMNS = ["id", "moves", "turns", "winner", "white_rating", "black_rating"]


# ============================================================ source dataset

@dataclass
class SourceInfo:
    path: str
    filename: str
    sha256: str
    size_bytes: int
    row_count: int
    columns: list
    matches_expected: dict = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source(path: Path, verify: bool) -> tuple[pd.DataFrame, SourceInfo]:
    """Load games.csv, validate columns, and record its identity.

    A checksum mismatch is RECORDED, not fatal, unless --verify-source is given:
    a future dataset may legitimately differ, and the manifest distinguishes the
    expected identity from the actual one.
    """
    if not path.is_file():
        raise SystemExit(
            f"ERROR: source dataset not found: {path}\n"
            f"       Expected {EXPECTED_SOURCE['filename']} "
            f"(sha256 {EXPECTED_SOURCE['sha256'][:16]}...). Nothing is downloaded "
            f"automatically; see training/README.md."
        )

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: {path} is missing required columns: {missing}")

    digest = sha256_file(path)
    size = path.stat().st_size
    info = SourceInfo(
        path=str(path.as_posix()),
        filename=path.name,
        sha256=digest,
        size_bytes=size,
        row_count=len(df),
        columns=list(df.columns),
        matches_expected={
            "sha256": digest == EXPECTED_SOURCE["sha256"],
            "size_bytes": size == EXPECTED_SOURCE["size_bytes"],
            "row_count": len(df) == EXPECTED_SOURCE["row_count"],
        },
    )

    if not all(info.matches_expected.values()):
        message = (
            f"source dataset differs from the identity recorded in the audit: "
            f"{info.matches_expected}"
        )
        if verify:
            raise SystemExit(f"ERROR (--verify-source): {message}")
        print(f"  WARNING: {message}", flush=True)

    return df, info


# ============================================================ sampling

def sample_games(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Deterministic uniform sample. Identical semantics to the notebook."""
    return df.sample(size, random_state=seed)


# ============================================================ position generation

@dataclass
class PositionOutcome:
    board: chess.Board | None
    plies_played: int
    san_tokens_available: int
    truncated_by_bad_san: bool
    bad_san_token: str | None
    invalid_board: bool


def board_after_fullmove(moves_san, fullmove_number: int) -> PositionOutcome:
    """Replay at most `fullmove_number * 2` plies of a SAN move string.

    Behaviourally identical to the notebook's function, except that a malformed
    SAN token is RECORDED rather than silently swallowed. The notebook's bare
    `except: break` is why 845 of its positions ended up shorter than 20 plies -
    though most of those are simply short games, not parse failures.
    """
    board = chess.Board()
    if not isinstance(moves_san, str):
        return PositionOutcome(board, 0, 0, False, None, False)

    tokens = moves_san.strip().split()
    ply_cap = fullmove_number * 2
    bad_token = None
    for token in tokens[:ply_cap]:
        try:
            board.push_san(token)
        except ValueError:
            bad_token = token
            break

    return PositionOutcome(
        board=board,
        plies_played=len(board.move_stack),
        san_tokens_available=len(tokens),
        truncated_by_bad_san=bad_token is not None,
        bad_san_token=bad_token,
        invalid_board=not board.is_valid(),
    )


def generate_positions(sampled: pd.DataFrame) -> tuple[list[dict], dict]:
    """Derive one position per sampled game and record how it was produced."""
    rows, stats = [], Counter()
    ply_hist = Counter()

    for row_index, row in sampled.iterrows():
        outcome = board_after_fullmove(row["moves"], MAX_FULLMOVES)
        stats["games_processed"] += 1
        if outcome.truncated_by_bad_san:
            stats["bad_san_truncations"] += 1
        if outcome.invalid_board:
            stats["invalid_boards_rejected"] += 1
            continue

        board = outcome.board
        ply_hist[outcome.plies_played] += 1
        rows.append({
            "game_id": row.get("id"),
            "source_row_index": int(row_index),
            "fen": board.fen(),
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
            "plies_played": outcome.plies_played,
            "reached_ply_cap": outcome.plies_played == MAX_PLIES,
            "san_tokens_available": outcome.san_tokens_available,
            "truncated_by_bad_san": outcome.truncated_by_bad_san,
            "bad_san_token": outcome.bad_san_token,
            "is_checkmate": board.is_checkmate(),
            "is_stalemate": board.is_stalemate(),
            "is_game_over": board.is_game_over(),
            "legal_move_count": board.legal_moves.count(),
        })

    stats["valid_positions"] = len(rows)
    return rows, {"counts": dict(stats), "ply_histogram": dict(sorted(ply_hist.items()))}


# ============================================================ deduplication

def deduplicate_by_fen(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep the FIRST occurrence of each exact FEN.

    Exact FEN - not piece placement - so that side to move, castling rights and
    en-passant state remain distinguishing. First-occurrence keeps the result
    deterministic given the deterministic sample order.
    """
    seen, kept, duplicate_counts = set(), [], Counter()
    for row in rows:
        fen = row["fen"]
        duplicate_counts[fen] += 1
        if fen in seen:
            continue
        seen.add(fen)
        kept.append(row)

    repeated = {f: n for f, n in duplicate_counts.items() if n > 1}
    return kept, {
        "method": "exact FEN, first occurrence kept",
        "input_rows": len(rows),
        "unique_fens": len(kept),
        "duplicate_rows_removed": len(rows) - len(kept),
        "fens_occurring_more_than_once": len(repeated),
        "max_occurrences_of_one_fen": max(duplicate_counts.values()) if duplicate_counts else 0,
    }


# ============================================================ stockfish

@dataclass
class EngineInfo:
    version: str
    wrapper_package: str
    wrapper_version: str
    depth: int
    threads: int
    hash_mb: int
    clear_hash_per_position: bool
    turn_perspective_used: str


def open_engine():
    """Start Stockfish with the Phase 3 reference configuration."""
    path = project_config.find_stockfish()
    if path is None:
        raise SystemExit(
            "ERROR: no Stockfish binary found. Set STOCKFISH_PATH or install "
            "stockfish on PATH. See docs/REPRODUCIBILITY.md."
        )

    from stockfish import Stockfish
    import importlib.metadata as md

    engine = Stockfish(path=path, depth=SF_DEPTH)
    # Native, side-to-move-relative output. The White-positive conversion is done
    # explicitly in training/labels.py rather than via the wrapper's
    # turn_perspective=False mode, which decides perspective with a substring test.
    engine.set_turn_perspective(True)
    engine.update_engine_parameters({"Threads": SF_THREADS, "Hash": SF_HASH_MB})

    info = EngineInfo(
        version=_engine_version(path),
        wrapper_package="stockfish",
        wrapper_version=md.version("stockfish"),
        depth=SF_DEPTH,
        threads=SF_THREADS,
        hash_mb=SF_HASH_MB,
        clear_hash_per_position=SF_CLEAR_HASH_PER_POSITION,
        turn_perspective_used="True (native side-to-move relative); converted to "
                              "White-positive in training/labels.py",
    )
    return engine, info, path


def _engine_version(path: str) -> str:
    """Read the engine's UCI identification banner."""
    try:
        proc = subprocess.run([path], input="quit\n", capture_output=True,
                              text=True, timeout=20)
        first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
        return first.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unknown: {exc}>"


def label_positions(engine, rows: list[dict], cp_clip: int, progress_every: int) -> dict:
    """Attach a label to every position. Mutates `rows` in place."""
    stats = Counter()
    anomalies = []

    for i, row in enumerate(rows, 1):
        if SF_CLEAR_HASH_PER_POSITION:
            # Clears Stockfish's hash table, so each label is independent of
            # everything evaluated before it. The notebook never did this, which
            # made its labels depend on iteration order.
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
                anomalies.append({"fen": row["fen"], "reason":
                                  "mate==0 reported but board is not checkmate"})

        stats[f"eval_type_{eval_type}"] += 1
        if label.was_clipped:
            stats["clipped"] += 1

        if progress_every and i % progress_every == 0:
            print(f"    labelled {i}/{len(rows)}", flush=True)

    return {"counts": dict(stats), "anomalies": anomalies}


# ============================================================ artifacts

def write_records(rows: list[dict], path: Path) -> str:
    """One JSON object per line - greppable, diffable, streamable."""
    ordered_keys = [
        "game_id", "source_row_index", "fen", "side_to_move", "plies_played",
        "reached_ply_cap", "san_tokens_available", "truncated_by_bad_san",
        "bad_san_token", "is_checkmate", "is_stalemate", "is_game_over",
        "legal_move_count", "eval_type", "raw_stockfish_value",
        "raw_value_perspective", "label", "label_perspective", "was_clipped",
        "mate_zero_consistent",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps({k: row[k] for k in ordered_keys if k in row},
                                sort_keys=False) + "\n")
    return sha256_file(path)


def build_manifest(source: SourceInfo, engine_info: EngineInfo, position_stats: dict,
                   dedup_stats: dict, label_stats: dict, rows: list[dict],
                   cp_clip: int, records_path: Path, records_sha: str,
                   elapsed: float, sample_size: int, sample_seed: int) -> dict:
    values = [r["label"] for r in rows]
    stm = Counter(r["side_to_move"] for r in rows)
    import importlib.metadata as md

    def version(pkg):
        try:
            return md.version(pkg)
        except Exception:  # noqa: BLE001
            return None

    return {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generation_seconds": round(elapsed, 1),
        "purpose": (
            "C6-Prep reproducible dataset/label artifact. Inputs for the planned "
            "A0-A3 experiments. No model is trained by this pipeline."
        ),
        "dataset": {
            "source_filename": source.filename,
            "source_sha256": source.sha256,
            "source_size_bytes": source.size_bytes,
            "source_row_count": source.row_count,
            "source_columns": source.columns,
            "expected_source": EXPECTED_SOURCE,
            "source_matches_expected": source.matches_expected,
            "sample_size": sample_size,
            "sample_seed": sample_seed,
            "sampling_method": (
                f"pandas DataFrame.sample(n={sample_size}, "
                f"random_state={sample_seed}) - uniform, without replacement, "
                f"not stratified"
            ),
            "deduplication": dedup_stats,
            "final_record_count": len(rows),
            "side_to_move_counts": dict(stm),
        },
        "position_generation": {
            "max_fullmoves": MAX_FULLMOVES,
            "max_plies": MAX_PLIES,
            "derivation": (
                f"replay SAN tokens[:{MAX_PLIES}] from the initial position - "
                f"AT MOST {MAX_PLIES} plies, not exactly {MAX_PLIES}. Games with "
                f"fewer plies yield earlier positions. Preserved from the notebook."
            ),
            "san_policy": (
                "python-chess push_san; a malformed token stops the replay and is "
                "recorded per-record in truncated_by_bad_san / bad_san_token"
            ),
            "counts": position_stats["counts"],
            "ply_histogram": position_stats["ply_histogram"],
            "positions_at_ply_cap": sum(1 for r in rows if r["reached_ply_cap"]),
            "positions_below_ply_cap": sum(1 for r in rows if not r["reached_ply_cap"]),
        },
        "stockfish": {
            "version": engine_info.version,
            "wrapper_package": engine_info.wrapper_package,
            "wrapper_version": engine_info.wrapper_version,
            "depth": engine_info.depth,
            "threads": engine_info.threads,
            "hash_mb": engine_info.hash_mb,
            "clear_hash_per_position": engine_info.clear_hash_per_position,
            "turn_perspective_used": engine_info.turn_perspective_used,
            "note": (
                "Executable path is deliberately NOT part of dataset identity. "
                "Clearing the hash before each position makes labels independent "
                "of iteration order; the notebook did not do this."
            ),
        },
        "labels": {
            **L.policy_summary(),
            "cp_clip_used": cp_clip,
            "counts": label_stats["counts"],
            "cp_label_count": label_stats["counts"].get("eval_type_cp", 0),
            "mate_label_count": label_stats["counts"].get("eval_type_mate", 0),
            "clipped_count": label_stats["counts"].get("clipped", 0),
            "anomalies": label_stats["anomalies"],
            "label_min": min(values) if values else None,
            "label_max": max(values) if values else None,
            "label_mean": round(sum(values) / len(values), 3) if values else None,
        },
        "artifact": {
            "records_file": records_path.name,
            "records_sha256": records_sha,
            "records_format": "JSON Lines (one position per line, UTF-8, LF)",
        },
        "reproducibility": {
            "python_version": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "packages": {
                "pandas": version("pandas"),
                "python-chess": version("chess"),
                "stockfish": version("stockfish"),
                "numpy": version("numpy"),
            },
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "determinism": (
                "Deterministic for fixed games.csv, Stockfish binary/version, "
                "configuration, seed and pipeline version. No model training and "
                "no RNG beyond the pandas sample seed."
            ),
        },
    }


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# ============================================================ cli

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="games.csv")
    ap.add_argument("--out", default="training/artifacts/dataset_v1",
                    help="output prefix; writes <out>.jsonl and <out>.manifest.json")
    ap.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--cp-clip", type=int, default=L.CP_CLIP)
    ap.add_argument("--limit", type=int, default=None,
                    help="label only the first N deduplicated positions (smoke runs)")
    ap.add_argument("--verify-source", action="store_true",
                    help="fail if games.csv does not match the recorded identity")
    ap.add_argument("--progress-every", type=int, default=1000)
    args = ap.parse_args()

    started = time.perf_counter()
    print(f"source     : {args.source}")
    df, source = load_source(Path(args.source), args.verify_source)
    print(f"  rows={source.row_count} sha256={source.sha256[:16]}... "
          f"matches_expected={all(source.matches_expected.values())}")

    sampled = sample_games(df, args.sample_size, args.seed)
    print(f"sampled    : {len(sampled)} games (seed {args.seed})")

    rows, position_stats = generate_positions(sampled)
    print(f"positions  : {len(rows)} valid, "
          f"{position_stats['counts'].get('invalid_boards_rejected', 0)} rejected, "
          f"{position_stats['counts'].get('bad_san_truncations', 0)} bad-SAN truncations")

    rows, dedup_stats = deduplicate_by_fen(rows)
    print(f"deduped    : {dedup_stats['unique_fens']} unique FENs "
          f"({dedup_stats['duplicate_rows_removed']} duplicate rows removed)")

    if args.limit:
        rows = rows[: args.limit]
        print(f"  --limit  : labelling only {len(rows)}")

    engine, engine_info, sf_path = open_engine()
    print(f"stockfish  : {engine_info.version}")
    print(f"  depth={SF_DEPTH} Threads={SF_THREADS} Hash={SF_HASH_MB}MB "
          f"clear_hash={SF_CLEAR_HASH_PER_POSITION}")
    try:
        label_stats = label_positions(engine, rows, args.cp_clip, args.progress_every)
    finally:
        engine.send_quit_command()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records_path = out.with_suffix(".jsonl")
    records_sha = write_records(rows, records_path)

    manifest = build_manifest(source, engine_info, position_stats, dedup_stats,
                              label_stats, rows, args.cp_clip, records_path,
                              records_sha, time.perf_counter() - started,
                              args.sample_size, args.seed)
    manifest_path = Path(str(out) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lab = manifest["labels"]
    print(f"\nWROTE {records_path}  (sha256 {records_sha[:16]}...)")
    print(f"WROTE {manifest_path}")
    print(f"  records      : {len(rows)}")
    print(f"  cp / mate    : {lab['cp_label_count']} / {lab['mate_label_count']}")
    print(f"  clipped      : {lab['clipped_count']}")
    print(f"  label range  : min={lab['label_min']} max={lab['label_max']} "
          f"mean={lab['label_mean']}")
    print(f"  elapsed      : {manifest['generation_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
