"""C7 dataset audit: what does games.csv actually support?

    python -m training.c7_dataset_audit --out training/artifacts/c7_audit.json

Read-only. Replays every game in `games.csv` once, then answers the fifteen
audit questions from the C7 brief from measurements rather than estimates.
No Stockfish call, no training, no production write.

--------------------------------------------------------------------------
WHY THIS EXISTS
--------------------------------------------------------------------------
`dataset_v1` was built by `training/build_dataset.py`, which samples 10,000
games and derives **exactly one position per game** - the board after at most
10 fullmoves (`tokens[:20]`). The dataset is therefore 9,667 near-opening
positions, one per game, and its 80/20 split is taken over POSITIONS.

Because each position came from a distinct game, that position-level split
happens to be game-level too. That coincidence disappears the moment more than
one position is taken per game, which is exactly what a dataset expansion must
do. This audit measures the headroom and the leakage risk before any pipeline
change is written.

--------------------------------------------------------------------------
THREE IDENTITY LEVELS
--------------------------------------------------------------------------
Leakage is measured at three levels, because they are not the same question:

    exact_fen     board.fen() - placement, side to move, castling, en-passant,
                  halfmove clock, fullmove number. What build_dataset.py
                  deduplicates on today.
    position      the first four FEN fields. The chess position proper; the
                  move counters are dropped. Two records identical here are the
                  same position reached at different move numbers.
    placement     the piece-placement field alone. THIS IS WHAT THE MODEL SEES:
                  the 12-plane encoder reads only piece squares, so two records
                  identical here are literally the same CNN input, whatever
                  their side to move or castling rights.

`placement` is the strictest and the one that matters for train/test leakage
against an A2-architecture model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import chess  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AUDIT_VERSION = "c7-audit-1"

DEFAULT_SOURCE = REPO_ROOT / "games.csv"
DEFAULT_OUT = REPO_ROOT / "training" / "artifacts" / "c7_audit.json"
DATASET_V1 = REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl"

# The existing pipeline's constants, for the "what we have today" comparison.
V1_SAMPLE_SIZE = 10_000
V1_SAMPLE_SEED = 42
V1_MAX_PLIES = 20

# Game-level split, proposed. Separate from the model training seed.
GAME_SPLIT_SEED = 42
TEST_FRACTION = 0.2

# Positions closer together than this many plies are near-duplicates by
# construction: consecutive plies differ by one move.
NEAR_DUPLICATE_PLY_GAP = 4

# Audit-only phase classifier. NOT a label input and NOT the evaluation suites'
# hand-assigned categories - those are curated per position. Defined here purely
# so the audit can report a reproducible phase mix.
OPENING_MAX_PLY = 20
ENDGAME_MAX_PIECES = 12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def key64(text: str) -> int:
    """A 64-bit digest, so 1.2M position keys fit in a numpy array."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(),
                          "big")


def classify_phase(ply: int, piece_count: int) -> str:
    if piece_count <= ENDGAME_MAX_PIECES:
        return "endgame"
    if ply <= OPENING_MAX_PLY:
        return "opening"
    return "middlegame"


# ============================================================ replay

class GameReplay:
    """Every ply of one game, recorded as compact parallel arrays."""

    __slots__ = ("game_id", "row_index", "n_tokens", "n_plies", "bad_san_token",
                 "exact", "position", "placement", "piece_counts", "white_to_move",
                 "terminal_checkmate", "terminal_stalemate")

    def __init__(self, game_id, row_index):
        self.game_id = game_id
        self.row_index = row_index
        self.n_tokens = 0
        self.n_plies = 0
        self.bad_san_token = None
        self.exact = []          # per ply: 64-bit key of board.fen()
        self.position = []       # per ply: 64-bit key of the first four fields
        self.placement = []      # per ply: 64-bit key of the placement field
        self.piece_counts = []
        self.white_to_move = []
        self.terminal_checkmate = False
        self.terminal_stalemate = False


def replay_game(game_id, row_index, moves_san) -> GameReplay:
    """Replay every legal SAN token, recording the position after each ply.

    Ply 0 (the initial position) is deliberately NOT recorded: it is identical
    in every game and would be a guaranteed cross-split duplicate.
    """
    rep = GameReplay(game_id, row_index)
    board = chess.Board()
    if not isinstance(moves_san, str):
        return rep

    tokens = moves_san.strip().split()
    rep.n_tokens = len(tokens)

    for token in tokens:
        try:
            board.push_san(token)
        except ValueError:
            rep.bad_san_token = token
            break
        fen = board.fen()
        fields = fen.split(" ")
        rep.exact.append(key64(fen))
        rep.position.append(key64(" ".join(fields[:4])))
        rep.placement.append(key64(fields[0]))
        rep.piece_counts.append(chess.popcount(board.occupied))
        rep.white_to_move.append(board.turn == chess.WHITE)

    rep.n_plies = len(rep.exact)
    if rep.n_plies:
        rep.terminal_checkmate = board.is_checkmate()
        rep.terminal_stalemate = board.is_stalemate()
    return rep


def replay_all(df: pd.DataFrame, progress_every: int) -> list[GameReplay]:
    out = []
    for i, (row_index, row) in enumerate(df.iterrows(), 1):
        out.append(replay_game(row.get("id"), int(row_index), row.get("moves")))
        if progress_every and i % progress_every == 0:
            print(f"    replayed {i}/{len(df)}", flush=True)
    return out


# ============================================================ sampling policies

def policy_last_of_first_20(rep: GameReplay) -> list[int]:
    """dataset_v1's policy: one position, the board after at most 20 plies."""
    if not rep.n_plies:
        return []
    return [min(rep.n_plies, V1_MAX_PLIES) - 1]


def policy_evenly_spaced(rep: GameReplay, k: int, min_ply: int = 1,
                         min_gap: int = NEAR_DUPLICATE_PLY_GAP) -> list[int]:
    """At most `k` plies spread evenly over the game, never closer than min_gap.

    Deterministic: no RNG. Positions before `min_ply` are skipped so the
    universally-shared first moves do not dominate.
    """
    if rep.n_plies < min_ply:
        return []
    lo, hi = min_ply - 1, rep.n_plies - 1
    span = hi - lo
    if span <= 0:
        return [lo]
    max_k = max(1, span // min_gap + 1)
    k = min(k, max_k)
    if k == 1:
        return [hi]
    picks = [lo + round(i * span / (k - 1)) for i in range(k)]
    return sorted(set(picks))


def policy_phase_stratified(rep: GameReplay, per_phase: int, min_ply: int = 1,
                            min_gap: int = NEAR_DUPLICATE_PLY_GAP) -> list[int]:
    """Up to `per_phase` positions from each phase bucket the game reaches.

    Deterministic: within a bucket the candidate plies are spread evenly, so a
    long middlegame contributes spaced positions rather than consecutive ones.
    """
    buckets = defaultdict(list)
    for idx in range(min_ply - 1, rep.n_plies):
        buckets[classify_phase(idx + 1, rep.piece_counts[idx])].append(idx)

    chosen = []
    for _, plies in sorted(buckets.items()):
        if not plies:
            continue
        span = plies[-1] - plies[0]
        k = min(per_phase, max(1, span // min_gap + 1))
        if k == 1:
            chosen.append(plies[len(plies) // 2])
        else:
            chosen.extend(plies[round(i * (len(plies) - 1) / (k - 1))]
                          for i in range(k))
    return sorted(set(chosen))


# `min_ply` matters more than anything else here. The first few plies of a game
# are drawn from a tiny set - there are 20 legal first moves and 400 two-ply
# openings - so positions before roughly ply 12 are shared across many games by
# construction and are guaranteed cross-split duplicates however the games are
# split. The per-ply profile in `ply_uniqueness_profile` measures exactly where
# that stops being true.
MIN_PLY_DEFAULT = 12

POLICIES = {
    "v1_one_per_game": ("dataset_v1: one position per game, board after <=20 plies",
                        policy_last_of_first_20),
    "evenly_spaced_2_minply1": ("<=2 evenly spaced plies per game from ply 1",
                                lambda r: policy_evenly_spaced(r, 2, min_ply=1)),
    "evenly_spaced_4_minply1": ("<=4 evenly spaced plies per game from ply 1",
                                lambda r: policy_evenly_spaced(r, 4, min_ply=1)),
    "evenly_spaced_1": ("1 ply per game from ply 12 (the ~20k target)",
                        lambda r: policy_evenly_spaced(r, 1, min_ply=MIN_PLY_DEFAULT)),
    "evenly_spaced_2": ("<=2 evenly spaced plies per game from ply 12, min gap 4",
                        lambda r: policy_evenly_spaced(r, 2, min_ply=MIN_PLY_DEFAULT)),
    "evenly_spaced_3": ("<=3 evenly spaced plies per game from ply 12 (the ~50k target)",
                        lambda r: policy_evenly_spaced(r, 3, min_ply=MIN_PLY_DEFAULT)),
    "evenly_spaced_4": ("<=4 evenly spaced plies per game from ply 12, min gap 4",
                        lambda r: policy_evenly_spaced(r, 4, min_ply=MIN_PLY_DEFAULT)),
    "evenly_spaced_1_minply16": ("1 ply per game from ply 16 (the ~20k target)",
                                 lambda r: policy_evenly_spaced(r, 1, min_ply=16)),
    "evenly_spaced_2_minply16": ("<=2 evenly spaced plies per game from ply 16",
                                 lambda r: policy_evenly_spaced(r, 2, min_ply=16)),
    "evenly_spaced_3_minply16": ("<=3 evenly spaced plies per game from ply 16 (the ~50k target)",
                                 lambda r: policy_evenly_spaced(r, 3, min_ply=16)),
    "evenly_spaced_4_minply16": ("<=4 evenly spaced plies per game from ply 16",
                                 lambda r: policy_evenly_spaced(r, 4, min_ply=16)),
    "evenly_spaced_4_minply20": ("<=4 evenly spaced plies per game from ply 20",
                                 lambda r: policy_evenly_spaced(r, 4, min_ply=20)),
    "evenly_spaced_6": ("<=6 evenly spaced plies per game from ply 12, min gap 4",
                        lambda r: policy_evenly_spaced(r, 6, min_ply=MIN_PLY_DEFAULT)),
    "evenly_spaced_6_minply16": ("<=6 evenly spaced plies per game from ply 16",
                                 lambda r: policy_evenly_spaced(r, 6, min_ply=16)),
    "evenly_spaced_8": ("<=8 evenly spaced plies per game from ply 12, min gap 4",
                        lambda r: policy_evenly_spaced(r, 8, min_ply=MIN_PLY_DEFAULT)),
    "evenly_spaced_12": ("<=12 evenly spaced plies per game from ply 12, min gap 4",
                         lambda r: policy_evenly_spaced(r, 12, min_ply=MIN_PLY_DEFAULT)),
    "phase_stratified_2": ("<=2 per phase bucket (opening/middlegame/endgame), from ply 12",
                           lambda r: policy_phase_stratified(r, 2, min_ply=MIN_PLY_DEFAULT)),
    "phase_stratified_3": ("<=3 per phase bucket, from ply 12",
                           lambda r: policy_phase_stratified(r, 3, min_ply=MIN_PLY_DEFAULT)),
    "phase_stratified_4": ("<=4 per phase bucket, from ply 12",
                           lambda r: policy_phase_stratified(r, 4, min_ply=MIN_PLY_DEFAULT)),
    "all_plies_from_12": ("every ply from 12 onward (upper bound, NOT a recommendation)",
                          lambda r: list(range(MIN_PLY_DEFAULT - 1, r.n_plies))),
    "all_plies": ("every ply of every game (absolute upper bound)",
                  lambda r: list(range(r.n_plies))),
}


def ply_uniqueness_profile(replays: list[GameReplay], max_ply: int = 120) -> dict:
    """How shared is the position at ply N, across the whole corpus?

    This is what sets `min_ply`. At ply 1 every game is one of 20 positions; the
    question is where the corpus stops overlapping with itself.
    """
    profile = {}
    for ply in list(range(1, 21)) + list(range(25, max_ply + 1, 5)):
        idx = ply - 1
        keys = [r.placement[idx] for r in replays if r.n_plies > idx]
        if not keys:
            continue
        counts = Counter(keys)
        profile[ply] = {
            "games_reaching_ply": len(keys),
            "distinct_placements": len(counts),
            "uniqueness": round(len(counts) / len(keys), 6),
            "rows_sharing_a_placement": sum(n for n in counts.values() if n > 1),
            "max_games_sharing_one_placement": max(counts.values()),
        }
    return profile


# ============================================================ game-level split

def game_unit_keys(ids: list[str], moves: list[str]) -> list[str]:
    """The split unit for each row: the game's CONTENT, not its id.

    Two measured properties of games.csv force this:

      * 813 ids appear on more than one row (1,758 rows). Every such group was
        verified to carry identical `moves`, so they are exact duplicate rows.
      * 50 distinct move strings are shared by 243 DIFFERENT ids - the same game
        content filed under different identifiers.

    Splitting on `id` would let the second group straddle the split: identical
    move sequences, hence identical positions, on both sides. Keying on the move
    string collapses both cases into one unit and makes
    `train_games n test_games = 0` a statement about game CONTENT.
    """
    return [sha256_text(m if isinstance(m, str) else "")[:32] for m in moves]


def game_level_split(unit_keys: list[str], seed: int, test_fraction: float):
    """Deterministic split over GAME UNITS, not positions.

    Every row sharing a unit key is assigned to the same side. Units are sorted
    first, so the split depends only on the SET of game contents - not on row
    order in games.csv - and a seeded permutation then assigns the first
    `test_fraction` of units to test. Identical inputs give identical output on
    any machine.
    """
    units = sorted(set(unit_keys))
    rng = np.random.default_rng(seed)
    permuted = np.asarray(units, dtype=object)[rng.permutation(len(units))]
    n_test = int(round(len(units) * test_fraction))
    test_units = set(permuted[:n_test])
    train_units = set(permuted[n_test:])

    train = {i for i, k in enumerate(unit_keys) if k in train_units}
    test = {i for i, k in enumerate(unit_keys) if k in test_units}
    return train, test, train_units, test_units


# ============================================================ analysis

def describe(values) -> dict:
    v = sorted(values)
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "min": v[0], "max": v[-1],
        "mean": round(st.fmean(v), 3),
        "median": v[len(v) // 2],
        "p10": v[int(0.10 * (len(v) - 1))],
        "p25": v[int(0.25 * (len(v) - 1))],
        "p75": v[int(0.75 * (len(v) - 1))],
        "p90": v[int(0.90 * (len(v) - 1))],
        "p99": v[int(0.99 * (len(v) - 1))],
    }


def evaluate_policy(name: str, description: str, selector, replays: list[GameReplay],
                    train_games: set, test_games: set,
                    canonical_rows: set | None = None,
                    suite_keys_by_level: dict | None = None) -> dict:
    """Apply one sampling policy and measure yield, duplication and leakage.

    `canonical_rows` restricts extraction to one row per game unit, so the 1,138
    exact duplicate rows in games.csv do not inflate the yield with positions
    that would be deduplicated away anyway.
    """
    per_game_counts = []
    # level -> key -> [n_train_occurrences, n_test_occurrences]
    seen = {lvl: defaultdict(lambda: [0, 0]) for lvl in ("exact", "position", "placement")}
    within_game_dupes = {lvl: 0 for lvl in ("exact", "position", "placement")}
    phase_counts, stm_counts = Counter(), Counter()
    n_train_positions = n_test_positions = 0
    consecutive_pairs = 0

    for gi, rep in enumerate(replays):
        if canonical_rows is not None and gi not in canonical_rows:
            continue
        picks = selector(rep)
        if not picks:
            per_game_counts.append(0)
            continue
        per_game_counts.append(len(picks))
        is_test = gi in test_games
        slot = 1 if is_test else 0
        if is_test:
            n_test_positions += len(picks)
        else:
            n_train_positions += len(picks)

        for lvl in seen:
            keys = [getattr(rep, lvl)[p] for p in picks]
            within_game_dupes[lvl] += len(keys) - len(set(keys))
            for k in keys:
                seen[lvl][k][slot] += 1

        for a, b in zip(picks, picks[1:]):
            if b - a < NEAR_DUPLICATE_PLY_GAP:
                consecutive_pairs += 1

        for p in picks:
            phase_counts[classify_phase(p + 1, rep.piece_counts[p])] += 1
            stm_counts["white" if rep.white_to_move[p] else "black"] += 1

    total = n_train_positions + n_test_positions
    leakage = {}
    for lvl, table in seen.items():
        both = sum(1 for tr, te in table.values() if tr > 0 and te > 0)
        leaked_test_rows = sum(te for tr, te in table.values() if tr > 0 and te > 0)
        n_unique = len(table)
        # After deduplication within each side, dropping every colliding key from
        # the TEST side leaves this many usable test positions. Train is left
        # untouched, so the scrub costs test size only.
        train_unique = sum(1 for tr, _ in table.values() if tr > 0)
        test_unique = sum(1 for _, te in table.values() if te > 0)
        leakage[lvl] = {
            "unique_keys": n_unique,
            "duplicate_rows_total": total - n_unique,
            "keys_in_both_train_and_test": both,
            "test_rows_whose_key_also_appears_in_train": leaked_test_rows,
            "leaked_test_fraction": round(leaked_test_rows / n_test_positions, 6)
            if n_test_positions else 0.0,
            "deduped_train_positions": train_unique,
            "deduped_test_positions": test_unique,
            "scrubbed_test_positions": test_unique - both,
            "scrub_cost_fraction": round(both / test_unique, 6) if test_unique else 0.0,
        }

    suite_hits = {}
    if suite_keys_by_level:
        for lvl, keys in suite_keys_by_level.items():
            train_keys = {k for k, (tr, _) in seen[lvl].items() if tr > 0}
            test_keys = {k for k, (_, te) in seen[lvl].items() if te > 0}
            suite_hits[lvl] = {
                "suite_keys_in_selected_train": len(keys & train_keys),
                "suite_keys_in_selected_test": len(keys & test_keys),
                "suite_keys_total": len(keys),
            }

    return {
        "description": description,
        "positions_total": total,
        "evaluation_suite_positions_selected": suite_hits,
        "positions_train": n_train_positions,
        "positions_test": n_test_positions,
        "games_contributing": sum(1 for c in per_game_counts if c),
        "positions_per_game": describe([c for c in per_game_counts if c]),
        "within_game_duplicate_rows": within_game_dupes,
        "selected_pairs_closer_than_min_gap": consecutive_pairs,
        "phase_counts": dict(phase_counts),
        "phase_fractions": {k: round(v / total, 4) for k, v in phase_counts.items()}
        if total else {},
        "side_to_move_counts": dict(stm_counts),
        "side_to_move_fractions": {k: round(v / total, 4) for k, v in stm_counts.items()}
        if total else {},
        "leakage": leakage,
    }


# ============================================================ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C7 dataset audit (read-only).")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="replay only the first N games (smoke runs)")
    ap.add_argument("--progress-every", type=int, default=5000)
    args = ap.parse_args(argv)

    print(f"source: {args.source}")
    df = pd.read_csv(args.source)
    if args.limit:
        df = df.head(args.limit)
    source_sha = sha256_file(args.source)
    print(f"  rows={len(df)} sha256={source_sha[:16]}...")

    # -------------------------------------------------- 1-3. games and identifiers
    ids = df["id"].astype(str).tolist()
    id_counts = Counter(ids)
    repeated_ids = {i: n for i, n in id_counts.items() if n > 1}
    moves_counts = Counter(df["moves"].astype(str).tolist())
    repeated_moves = {m: n for m, n in moves_counts.items() if n > 1}

    identifiers = {
        "row_count": len(df),
        "id_column_present": "id" in df.columns,
        "unique_ids": len(id_counts),
        "ids_appearing_more_than_once": len(repeated_ids),
        "rows_involved_in_duplicate_ids": sum(repeated_ids.values()),
        "max_rows_sharing_one_id": max(id_counts.values()),
        "id_is_unique_per_row": len(id_counts) == len(df),
        "distinct_move_strings": len(moves_counts),
        "move_strings_appearing_more_than_once": len(repeated_moves),
        "rows_involved_in_duplicate_move_strings": sum(repeated_moves.values()),
    }
    print(f"  unique ids: {identifiers['unique_ids']} / {len(df)}  "
          f"(duplicate ids: {identifiers['ids_appearing_more_than_once']})")

    # -------------------------------------------------- 4-5. replay every game
    print("replaying every game...")
    replays = replay_all(df, args.progress_every)

    n_tokens = [r.n_tokens for r in replays]
    n_plies = [r.n_plies for r in replays]
    bad_san = [r for r in replays if r.bad_san_token]
    empty = [r for r in replays if r.n_plies == 0]

    ply_histogram = Counter()
    for r in replays:
        ply_histogram[r.n_plies] += 1

    positions_at_or_beyond = {}
    for threshold in (1, 10, 20, 30, 40, 60, 80, 100, 150):
        positions_at_or_beyond[threshold] = sum(1 for p in n_plies if p >= threshold)

    total_plies = sum(n_plies)
    game_lengths = {
        "declared_turns_column": describe(df["turns"].astype(int).tolist()),
        "san_tokens_in_moves_column": describe(n_tokens),
        "legally_replayed_plies": describe(n_plies),
        "total_legally_replayed_plies": total_plies,
        "games_with_zero_plies": len(empty),
        "games_with_bad_san": len(bad_san),
        "games_reaching_ply": positions_at_or_beyond,
        "ply_histogram_head": dict(sorted(ply_histogram.items())[:30]),
    }
    print(f"  total legal plies across all games: {total_plies:,}")
    print(f"  median game length: {game_lengths['legally_replayed_plies']['median']} plies")
    print(f"  games with bad SAN: {len(bad_san)}   zero-ply games: {len(empty)}")

    # -------------------------------------------------- 7-8. duplicates, all plies
    print("measuring duplicate positions across ALL plies...")
    all_dupes = {}
    for lvl in ("exact", "position", "placement"):
        within = 0
        table = defaultdict(int)
        for rep in replays:
            keys = getattr(rep, lvl)
            within += len(keys) - len(set(keys))
            for k in set(keys):
                table[k] += 1          # count GAMES containing this key
        multi_game = sum(1 for n in table.values() if n > 1)
        all_dupes[lvl] = {
            "unique_keys_across_corpus": len(table),
            "within_game_duplicate_plies": within,
            "keys_appearing_in_more_than_one_game": multi_game,
            "fraction_of_keys_in_multiple_games": round(multi_game / len(table), 6)
            if table else 0.0,
        }
        print(f"  {lvl:10s} unique={len(table):>9,}  within-game dupes={within:>8,}  "
              f"in >1 game={multi_game:>8,}")

    print("profiling per-ply uniqueness...")
    ply_profile = ply_uniqueness_profile(replays)
    for ply in (1, 2, 4, 8, 12, 16, 20, 30, 40):
        if ply in ply_profile:
            p = ply_profile[ply]
            print(f"  ply {ply:>3}: {p['games_reaching_ply']:>6,} games, "
                  f"{p['distinct_placements']:>6,} distinct  "
                  f"uniqueness={p['uniqueness']:.4f}")

    # -------------------------------------------------- game-level split
    unit_keys = game_unit_keys(ids, df["moves"].astype(str).tolist())
    train_games, test_games, train_units, test_units = game_level_split(
        unit_keys, GAME_SPLIT_SEED, TEST_FRACTION)
    assert not (train_games & test_games), "game split overlaps by row"
    assert not (train_units & test_units), "game split overlaps by content"

    # Every id that appears on any train row, and likewise for test. If an id
    # were filed under two different contents it could appear in both; measured.
    train_id_set = {ids[i] for i in train_games}
    test_id_set = {ids[i] for i in test_games}
    split_info = {
        "method": ("rows grouped by game CONTENT (sha256 of the move string), "
                   "units sorted, seeded permutation, first test_fraction to "
                   "test; positions are extracted AFTER the split"),
        "split_unit": "game move-sequence content, not the id column",
        "split_seed": GAME_SPLIT_SEED,
        "test_fraction": TEST_FRACTION,
        "n_rows": len(df),
        "n_game_units": len(set(unit_keys)),
        "n_train_units": len(train_units),
        "n_test_units": len(test_units),
        "n_train_rows": len(train_games),
        "n_test_rows": len(test_games),
        "train_units_intersect_test_units": len(train_units & test_units),
        "train_ids_intersect_test_ids": len(train_id_set & test_id_set),
        "game_list_sha256": sha256_text("\n".join(sorted(set(unit_keys)))),
        "train_game_sha256": sha256_text("\n".join(sorted(train_units))),
        "test_game_sha256": sha256_text("\n".join(sorted(test_units))),
    }
    print(f"game-level split: {len(train_units)} train / {len(test_units)} test units "
          f"({len(train_games)} / {len(test_games)} rows)")
    print(f"  train_units n test_units = {len(train_units & test_units)}")
    print(f"  train_ids   n test_ids   = {len(train_id_set & test_id_set)}")

    # -------------------------------------------------- 6, 9-11, 14. policies
    # One canonical row per game unit: duplicate rows contribute nothing new.
    canonical_rows, seen_units = set(), set()
    for i, k in enumerate(unit_keys):
        if k not in seen_units:
            seen_units.add(k)
            canonical_rows.add(i)
    print(f"canonical rows (one per game unit): {len(canonical_rows)} "
          f"of {len(df)} ({len(df) - len(canonical_rows)} duplicate rows skipped)")

    # Load the evaluation-suite keys FIRST: each policy reports how many suite
    # positions its selected train side would contain.
    suite_keys, suite_counts = {}, {}
    for suite in ("extended", "phase0_52"):
        path = REPO_ROOT / "evaluation" / "positions" / f"{suite}.json"
        if not path.is_file():
            continue
        fens = [p["fen"] for p in
                json.loads(path.read_text(encoding="utf-8"))["positions"]]
        suite_counts[suite] = len(fens)
        for fen in fens:
            suite_keys.setdefault("exact", set()).add(key64(fen))
            suite_keys.setdefault("position", set()).add(key64(" ".join(fen.split(" ")[:4])))
            suite_keys.setdefault("placement", set()).add(key64(fen.split(" ")[0]))

    print("evaluating sampling policies...")
    policies = {}
    for name, (desc, selector) in POLICIES.items():
        policies[name] = evaluate_policy(name, desc, selector, replays,
                                         train_games, test_games, canonical_rows,
                                         suite_keys)
        p = policies[name]
        lk = p["leakage"]["placement"]
        print(f"  {name:22s} total={p['positions_total']:>9,}  "
              f"train={p['positions_train']:>8,}  test={p['positions_test']:>7,}  "
              f"placement-leak={lk['leaked_test_fraction']:.4%}")

    # -------------------------------------------------- evaluation-suite overlap
    # The engine-level suites are the measurement instrument for every C6/C8 arm.
    # If an expanded training set contained suite positions, that instrument
    # would be compromised. Measured at placement level, the strictest.
    print("checking overlap against the evaluation suites...")
    suite_overlap = {"suite_position_counts": suite_counts}
    if suite_keys:
        for lvl in ("exact", "position", "placement"):
            all_corpus, from_min_ply = set(), set()
            for rep in replays:
                keys = getattr(rep, lvl)
                all_corpus.update(keys)
                from_min_ply.update(keys[MIN_PLY_DEFAULT - 1:])
            hits_all = suite_keys[lvl] & all_corpus
            hits_min = suite_keys[lvl] & from_min_ply
            suite_overlap[lvl] = {
                "suite_keys": len(suite_keys[lvl]),
                "present_anywhere_in_games_csv": len(hits_all),
                "fraction_anywhere": round(len(hits_all) / len(suite_keys[lvl]), 6),
                f"present_at_ply_{MIN_PLY_DEFAULT}_or_later": len(hits_min),
                "fraction_from_min_ply": round(len(hits_min) / len(suite_keys[lvl]), 6),
            }
            print(f"  {lvl:10s} suite positions in games.csv: "
                  f"{len(hits_all)}/{len(suite_keys[lvl])} anywhere, "
                  f"{len(hits_min)} at ply>={MIN_PLY_DEFAULT}")

        # Does the EXISTING dataset_v1 already contain suite positions? If so this
        # is a pre-existing property of the A2 baseline, not something C7 creates.
        if DATASET_V1.is_file():
            v1_records = [json.loads(line) for line in
                          DATASET_V1.read_text(encoding="utf-8").splitlines()
                          if line.strip()]
            v1_levels = {
                "exact": {key64(r["fen"]) for r in v1_records},
                "position": {key64(" ".join(r["fen"].split(" ")[:4])) for r in v1_records},
                "placement": {key64(r["fen"].split(" ")[0]) for r in v1_records},
            }
            suite_overlap["dataset_v1_vs_suites"] = {
                lvl: {
                    "overlapping_keys": len(suite_keys[lvl] & v1_levels[lvl]),
                    "suite_keys": len(suite_keys[lvl]),
                    "fraction": round(len(suite_keys[lvl] & v1_levels[lvl])
                                      / len(suite_keys[lvl]), 6),
                } for lvl in ("exact", "position", "placement")
            }
            for lvl, d in suite_overlap["dataset_v1_vs_suites"].items():
                print(f"  dataset_v1 {lvl:10s} overlaps suites: "
                      f"{d['overlapping_keys']}/{d['suite_keys']}")

    # -------------------------------------------------- dataset_v1 comparison
    v1 = None
    if DATASET_V1.is_file():
        records = [json.loads(line) for line in
                   DATASET_V1.read_text(encoding="utf-8").splitlines() if line.strip()]
        gids = [r.get("game_id") for r in records]
        gid_counts = Counter(gids)
        placements = Counter(r["fen"].split(" ")[0] for r in records)
        positions4 = Counter(" ".join(r["fen"].split(" ")[:4]) for r in records)
        v1 = {
            "records": len(records),
            "distinct_game_ids": len(gid_counts),
            "max_records_from_one_game": max(gid_counts.values()),
            "one_position_per_game": max(gid_counts.values()) == 1,
            "distinct_exact_fens": len({r["fen"] for r in records}),
            "distinct_positions_first_four_fields": len(positions4),
            "distinct_placements": len(placements),
            "placements_appearing_more_than_once":
                sum(1 for n in placements.values() if n > 1),
            "records_sharing_a_placement":
                sum(n for n in placements.values() if n > 1),
            "side_to_move_counts": dict(Counter(r["side_to_move"] for r in records)),
            "eval_type_counts": dict(Counter(r["eval_type"] for r in records)),
            "plies_played": describe([r["plies_played"] for r in records]),
            "label_stats": {
                "min": min(r["label"] for r in records),
                "max": max(r["label"] for r in records),
                "mean": round(st.fmean(r["label"] for r in records), 3),
            },
        }
        print(f"dataset_v1: {v1['records']} records, "
              f"{v1['distinct_game_ids']} distinct games, "
              f"one_position_per_game={v1['one_position_per_game']}")
        print(f"  distinct placements: {v1['distinct_placements']} "
              f"({v1['records_sharing_a_placement']} records share a placement)")

    out = {
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": str(args.source.as_posix()),
            "sha256": source_sha,
            "size_bytes": args.source.stat().st_size,
            "rows_audited": len(df),
            "columns": list(df.columns),
        },
        "identifiers": identifiers,
        "game_lengths": game_lengths,
        "duplicates_all_plies": all_dupes,
        "ply_uniqueness_profile": ply_profile,
        "game_level_split": split_info,
        "canonical_rows": len(canonical_rows),
        "sampling_policies": policies,
        "evaluation_suite_overlap": suite_overlap,
        "dataset_v1": v1,
        "definitions": {
            "exact": "board.fen(): placement, stm, castling, ep, halfmove, fullmove",
            "position": "first four FEN fields (move counters dropped)",
            "placement": "piece-placement field only - WHAT THE 12-PLANE MODEL SEES",
            "phase_classifier": (
                f"endgame if piece_count <= {ENDGAME_MAX_PIECES}; else opening if "
                f"ply <= {OPENING_MAX_PLY}; else middlegame. Audit-only heuristic."
            ),
            "near_duplicate_ply_gap": NEAR_DUPLICATE_PLY_GAP,
            "ply_0_excluded": "the initial position is never recorded",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWROTE {args.out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
