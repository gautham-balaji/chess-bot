"""C9 Stage 2a — leakage-safe divisor and fit-variant selection.

    python -m training.c9_stage2a

Selects the divisor `d*` and the Ridge fit variant for C9 Stage 2 **without ever
reading an evaluation suite**. Writes
`training/experiments/C9/stage2_selection.json` and
`training/experiments/C9/tune_candidates.json`.

This module does NOT evaluate the suites and does NOT stage any engine variant.
That is Stage 2b (`training/c9_stage2b.py`). No production file is written.

Design: docs/C9_STAGE2_DESIGN.md sections 6 and 7.

--------------------------------------------------------------------------
WHY SELECTION NEEDS ITS OWN POPULATION
--------------------------------------------------------------------------
`corr(tanh(cnn/d), label)` is MONOTONICALLY INCREASING in `d` (Stage 1 section
8), so maximising it picks `d -> infinity`, where the CNN term vanishes. That is
a BETWEEN-position criterion; move ranking is WITHIN-position. Selecting `d` by
correlation is therefore invalid.

The suites cannot be used either: they are the measurement instrument, and
tuning on them would invalidate every C6-C8 number measured there.

So Stage 2a builds its own within-position benchmark:

    dataset_v2 test (13,712 positions / 3,614 games)
      -> game-level 80/20 split, seed 9          [FIT | TUNE]
      -> Ridge fitted on FIT ONLY, per (d, variant, seed)
      -> TUNE positions' legal moves scored by the SAME pinned Stockfish
      -> mean within-position Spearman of fused score vs Stockfish ordering

FIT and TUNE share no game. Neither shares a placement with either suite. The
Ridge never sees TUNE.

--------------------------------------------------------------------------
THE SELECTION SET IS CLOSED
--------------------------------------------------------------------------
`d` is chosen from exactly Stage 1's swept values - no new ones are invented -
and 200 (production) is included so "keep production" can win. Ties within 0.005
Spearman resolve toward the smaller `d`, then toward variant A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import chess  # noqa: E402
import numpy as np  # noqa: E402

from training import build_dataset as B  # noqa: E402
from training import c9_stage1 as S1  # noqa: E402
from training import dataset as D  # noqa: E402
from training import representation as R  # noqa: E402
from training import refit_ridge as RR  # noqa: E402

STAGE = "C9-stage2a"
ARM = "C8a"
SEEDS = (0, 1, 2)
DATASET_PREFIX = REPO_ROOT / "training" / "artifacts" / "dataset_v2"
OUT_DIR = REPO_ROOT / "training" / "experiments" / "C9"

# --- the design's frozen parameters -------------------------------------------
TUNE_SPLIT_SEED = 9              # distinct from dataset seed 42 and inner seed 1234
TUNE_FRACTION = 0.20
DIVISORS = (200, 400, 600, 800, 1000, 1500, 2000)   # CLOSED: Stage 1's sweep
FIT_VARIANTS = ("A_include_checkmate", "B_exclude_checkmate")
TIE_SPEARMAN = 0.005             # ties resolve to smaller d, then variant A
TARGET_TUNE_CANDIDATES = 6000    # design section 14


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================ FIT / TUNE split

def fit_tune_split(records: list[dict], seed: int = TUNE_SPLIT_SEED,
                   tune_fraction: float = TUNE_FRACTION):
    """Game-level split of the fit population. Deterministic, no sampling.

    Games are SORTED before the seeded permutation, so the split depends only on
    the SET of game keys - not on record order - exactly as the C7/C8 game split
    does.
    """
    games = sorted({r["game_content_key"] for r in records})
    rng = np.random.default_rng(seed)
    permuted = np.asarray(games, dtype=object)[rng.permutation(len(games))]
    n_tune = int(round(len(games) * tune_fraction))
    tune_games = set(permuted[:n_tune])
    fit_games = set(permuted[n_tune:])
    if tune_games & fit_games:
        raise SystemExit("ERROR: FIT and TUNE game sets overlap")

    fit = [r for r in records if r["game_content_key"] in fit_games]
    tune = [r for r in records if r["game_content_key"] in tune_games]
    return fit, tune, fit_games, tune_games


def select_tune_positions(tune: list[dict], target_candidates: int) -> list[dict]:
    """Take TUNE positions in file order until the candidate budget is met.

    The file is already sorted canonically by (game_content_key, ply), so this is
    deterministic and needs no RNG. Mated boards are skipped: they have no legal
    moves and cannot contribute a ranking.
    """
    chosen, total = [], 0
    for r in tune:
        if r["is_checkmate"]:
            continue
        n = chess.Board(r["fen"]).legal_moves.count()
        if n < 2:                      # nothing to rank
            continue
        chosen.append(r)
        total += n
        if total >= target_candidates:
            break
    return chosen


# ============================================================ TUNE labelling

def label_tune_candidates(positions: list[dict], progress_every: int = 25) -> dict:
    """Stockfish-label every legal move's resulting board for each TUNE position.

    Uses the SAME pinned configuration as `build_dataset` and the evaluator:
    depth 8, Threads=1, Hash=16MB, hash cleared before every position. The
    evaluator itself is NOT invoked and no suite is read.

    Labels are White-positive via the A2 policy, so a higher value is better for
    White. The MOVER's preference is recovered at scoring time from side to move.
    """
    from training import labels as L

    engine, info, _ = B.open_engine()
    out = []
    try:
        for i, rec in enumerate(positions, 1):
            board = chess.Board(rec["fen"])
            mover_is_white = board.turn == chess.WHITE
            kids = []
            for mv in board.legal_moves:
                board.push(mv)
                child_fen = board.fen()
                child_white = board.turn == chess.WHITE
                is_mate = board.is_checkmate()
                board.pop()

                engine.send_ucinewgame_command()
                engine.set_fen_position(child_fen)
                ev = engine.get_evaluation()
                lab = L.make_label(eval_type=ev["type"], raw_value=int(ev["value"]),
                                   side_to_move_is_white=child_white,
                                   is_checkmate=is_mate, cp_clip=L.CP_CLIP)
                kids.append({"uci": mv.uci(), "fen": child_fen,
                             "label": lab.label, "eval_type": lab.eval_type})
            out.append({
                "fen": rec["fen"],
                "game_content_key": rec["game_content_key"],
                "mover_is_white": mover_is_white,
                "candidates": kids,
            })
            if progress_every and i % progress_every == 0:
                print(f"    labelled {i}/{len(positions)} positions "
                      f"({sum(len(p['candidates']) for p in out)} candidates)",
                      flush=True)
    finally:
        engine.send_quit_command()

    return {
        "stockfish": {"version": info.version, "depth": B.SF_DEPTH,
                      "threads": B.SF_THREADS, "hash_mb": B.SF_HASH_MB,
                      "clear_hash_per_position": B.SF_CLEAR_HASH_PER_POSITION},
        "n_positions": len(out),
        "n_candidates": sum(len(p["candidates"]) for p in out),
        "positions": out,
    }


# ============================================================ scoring

def spearman(a, b) -> float:
    """Rank correlation, average ranks for ties. Small n, so this is exact."""
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def score_tune(coef, tune_labels: dict, cnn_by_position: dict,
               heur_by_position: dict, divisor: float) -> dict:
    """Mean within-position Spearman and top-1 agreement over TUNE positions.

    The fused score is the production five-term weighted sum at the candidate
    divisor. The engine's unscaled heuristic bonuses and 1-ply lookahead are NOT
    applied here - this is the same proxy Stage 1 used, and it is used only to
    SELECT. Stage 2b measures the real engine.
    """
    coef = np.asarray(coef, dtype=float)
    rhos, top1, n = [], 0, 0
    for pos in tune_labels["positions"]:
        fen = pos["fen"]
        y = np.array([c["label"] for c in pos["candidates"]], dtype=float)
        if len(y) < 2:
            continue
        cnn = cnn_by_position[fen]
        feats = heur_by_position[fen]
        fused = (coef[0] * np.tanh(cnn / divisor)
                 + sum(coef[i + 1] * feats[:, i] for i in range(4)))

        # Both scores are White-positive; the mover prefers max if White else min.
        sign = 1.0 if pos["mover_is_white"] else -1.0
        rho = spearman(sign * fused, sign * y)
        if not np.isnan(rho):
            rhos.append(rho)
        top1 += int(np.argmax(sign * fused) == np.argmax(sign * y))
        n += 1
    return {"mean_spearman": float(np.mean(rhos)) if rhos else float("nan"),
            "median_spearman": float(np.median(rhos)) if rhos else float("nan"),
            "top1_agreement_pct": round(100 * top1 / n, 3) if n else None,
            "n_positions": n}


# ============================================================ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="C9 Stage 2a: leakage-safe divisor / fit-variant selection.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dataset-prefix", type=Path, default=DATASET_PREFIX)
    ap.add_argument("--target-candidates", type=int, default=TARGET_TUNE_CANDIDATES)
    ap.add_argument("--reuse-labels", action="store_true",
                    help="reuse tune_candidates.json if present (skips Stockfish)")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{STAGE}: divisor / fit-variant selection "
          f"(NO evaluation suite is read, NO engine evaluation)\n")

    # ------------------------------------------------------------- inputs
    models_meta = S1.verify_models()
    print("C8a models verified against the C9 audit:")
    for k, v in models_meta.items():
        print(f"  {k}: {v['model_weights_sha256'][:20]}... OK")

    records, _, manifest, digest = RR.load_fit_population(args.dataset_prefix)
    print(f"\nfit population: {len(records)} records, sha256 {digest[:16]}... (manifest OK)")
    leak = S1.leakage_checks(records)
    print(f"  leakage: suite placements {leak['overlap_with_suite_placements']}, "
          f"suite FENs {leak['overlap_with_suite_exact_fens']}, "
          f"C8a-train placements {leak['overlap_with_c8a_train_placements']} -> CLEAN")

    y_all = D.apply_label_policy(records, RR.ARM_POLICIES[ARM]).astype(np.float64)
    if not np.array_equal(y_all, np.array([r["label"] for r in records], dtype=np.float64)):
        raise SystemExit("ERROR: re-derived labels differ from stored labels")

    # ------------------------------------------------------------- FIT / TUNE
    fit, tune, fit_games, tune_games = fit_tune_split(records)
    assert not (fit_games & tune_games)
    print(f"\nFIT/TUNE split (game-level, seed {TUNE_SPLIT_SEED}):")
    print(f"  FIT  {len(fit):6d} positions / {len(fit_games):5d} games")
    print(f"  TUNE {len(tune):6d} positions / {len(tune_games):5d} games")
    print(f"  FIT games n TUNE games = {len(fit_games & tune_games)}")
    S1.leakage_checks(tune)      # raises if TUNE touches a suite
    print("  TUNE suite overlap -> CLEAN")

    tune_positions = select_tune_positions(tune, args.target_candidates)
    print(f"  TUNE benchmark: {len(tune_positions)} positions "
          f"(~{sum(chess.Board(r['fen']).legal_moves.count() for r in tune_positions)} candidates)")

    # ------------------------------------------------------------- TUNE labels
    labels_path = args.out_dir / "tune_candidates.json"
    if args.reuse_labels and labels_path.is_file():
        tune_labels = json.loads(labels_path.read_text(encoding="utf-8"))
        print(f"\nreusing {labels_path.name}: {tune_labels['n_positions']} positions, "
              f"{tune_labels['n_candidates']} candidates")
    else:
        print(f"\nlabelling TUNE candidates with Stockfish "
              f"(depth {B.SF_DEPTH}, Threads={B.SF_THREADS}, Hash={B.SF_HASH_MB}MB, "
              f"clear hash per position)...")
        tune_labels = label_tune_candidates(tune_positions)
        tune_labels["fit_tune_split"] = {
            "seed": TUNE_SPLIT_SEED, "tune_fraction": TUNE_FRACTION,
            "n_fit_positions": len(fit), "n_tune_positions": len(tune),
            "n_fit_games": len(fit_games), "n_tune_games": len(tune_games),
            "fit_games_sha256": sha256_text("\n".join(sorted(fit_games))),
            "tune_games_sha256": sha256_text("\n".join(sorted(tune_games))),
        }
        labels_path.write_text(json.dumps(tune_labels, indent=2) + "\n", encoding="utf-8")
        print(f"  {tune_labels['n_positions']} positions, "
              f"{tune_labels['n_candidates']} candidate evaluations")
        print(f"  WROTE {labels_path.name}")

    # ------------------------------------------------------------- caches
    import engine as E
    import keras

    tune_fens = [p["fen"] for p in tune_labels["positions"]]
    heur_by_position, child_fens_by_position = {}, {}
    for pos in tune_labels["positions"]:
        rows = []
        for c in pos["candidates"]:
            b = chess.Board(c["fen"])
            rows.append([E.material_balance(b), E.space_control(b),
                         E.center_control(b), E.mobility_score(b)])
        heur_by_position[pos["fen"]] = np.asarray(rows, dtype=float)
        child_fens_by_position[pos["fen"]] = [c["fen"] for c in pos["candidates"]]

    fit_boards = [chess.Board(r["fen"]) for r in fit]
    X_fit = R.encode_many(r["fen"] for r in fit)
    y_fit = D.apply_label_policy(fit, RR.ARM_POLICIES[ARM]).astype(np.float64)
    mated_fit = np.array([bool(r["is_checkmate"]) for r in fit])
    print(f"\nFIT: {len(fit)} positions, {int(mated_fit.sum())} checkmate "
          f"({100 * mated_fit.mean():.1f}%)")

    # ------------------------------------------------------------- grid
    print(f"\n{'=' * 100}\nSELECTION GRID (Ridge fitted on FIT only; scored on TUNE)\n{'=' * 100}")
    grid = {}
    per_seed_cnn = {}
    for seed in SEEDS:
        model, _ = RR.load_model_for(ARM, seed)
        per_seed_cnn[seed] = {
            "fit": model.predict(X_fit, verbose=0).flatten().astype(np.float64),
            "tune": {fen: model.predict(R.encode_many(child_fens_by_position[fen]),
                                        verbose=0).flatten().astype(np.float64)
                     for fen in tune_fens},
        }
        del model
        keras.backend.clear_session()

    print(f"{'variant':22s}{'d':>6s}" + "".join(f"{'s'+str(s)+' rho':>11s}" for s in SEEDS)
          + f"{'mean rho':>11s}{'mean top1':>11s}")
    for variant in FIT_VARIANTS:
        mask = np.ones(len(fit), bool) if variant.startswith("A") else ~mated_fit
        for d in DIVISORS:
            rhos, top1s, per_seed = [], [], {}
            for seed in SEEDS:
                cnn_fit = per_seed_cnn[seed]["fit"]
                X = np.array([RR.board_features(b, c, tanh_scale=float(d))
                              for b, c, m in zip(fit_boards, cnn_fit, mask) if m])
                ridge = RR.fit_ridge(X, y_fit[mask])
                w = np.asarray(ridge.coef_, dtype=float)
                sc = score_tune(w, tune_labels, per_seed_cnn[seed]["tune"],
                                heur_by_position, float(d))
                per_seed[f"seed_{seed}"] = {
                    "coef": [float(x) for x in w],
                    "intercept": float(ridge.intercept_), **sc,
                }
                rhos.append(sc["mean_spearman"])
                top1s.append(sc["top1_agreement_pct"])
            key = f"{variant}|d={d}"
            grid[key] = {
                "fit_variant": variant, "divisor": d,
                "n_fit_positions": int(mask.sum()),
                "mean_spearman_across_seeds": float(np.mean(rhos)),
                "mean_top1_across_seeds": float(np.mean(top1s)),
                "per_seed": per_seed,
            }
            print(f"{variant:22s}{d:>6d}" + "".join(f"{r:>11.4f}" for r in rhos)
                  + f"{np.mean(rhos):>11.4f}{np.mean(top1s):>10.2f}%")

    # ------------------------------------------------------------- selection
    best_score = max(g["mean_spearman_across_seeds"] for g in grid.values())
    contenders = [(k, g) for k, g in grid.items()
                  if best_score - g["mean_spearman_across_seeds"] <= TIE_SPEARMAN]
    contenders.sort(key=lambda kv: (kv[1]["divisor"],
                                    0 if kv[1]["fit_variant"].startswith("A") else 1))
    chosen_key, chosen = contenders[0]
    runner_up = sorted(
        ((k, g) for k, g in grid.items() if k != chosen_key),
        key=lambda kv: -kv[1]["mean_spearman_across_seeds"])[0]

    print(f"\n{'=' * 100}\nSELECTION\n{'=' * 100}")
    print(f"  best mean Spearman        {best_score:.4f}")
    print(f"  contenders within {TIE_SPEARMAN}   {len(contenders)}  "
          f"{[k for k, _ in contenders]}")
    print(f"  SELECTED                  {chosen_key}")
    print(f"    d*                      {chosen['divisor']}")
    print(f"    fit variant             {chosen['fit_variant']}")
    print(f"    mean Spearman           {chosen['mean_spearman_across_seeds']:.4f}")
    print(f"    mean top-1              {chosen['mean_top1_across_seeds']:.2f}%")
    print(f"  best non-selected         {runner_up[0]} "
          f"({runner_up[1]['mean_spearman_across_seeds']:.4f})")
    print(f"  margin over it            "
          f"{chosen['mean_spearman_across_seeds'] - runner_up[1]['mean_spearman_across_seeds']:+.4f}")
    prod_key = "A_include_checkmate|d=200"
    print(f"  production divisor row    {prod_key}: "
          f"{grid[prod_key]['mean_spearman_across_seeds']:.4f}")

    out = {
        "stage": STAGE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Selection only. No evaluation suite was read; no engine "
                 "evaluation was run. The candidate set is closed and was fixed "
                 "by docs/C9_STAGE2_DESIGN.md before this ran."),
        "arm": ARM, "seeds": list(SEEDS),
        "label_policy": RR.ARM_POLICIES[ARM],
        "ridge_alpha": RR.RIDGE_ALPHA,
        "c8a_models": models_meta,
        "fit_population": {"records": len(records), "test_sha256": digest, **leak},
        "fit_tune_split": tune_labels.get("fit_tune_split", {
            "seed": TUNE_SPLIT_SEED, "tune_fraction": TUNE_FRACTION,
            "n_fit_positions": len(fit), "n_tune_positions": len(tune),
            "n_fit_games": len(fit_games), "n_tune_games": len(tune_games),
            "fit_games_sha256": sha256_text("\n".join(sorted(fit_games))),
            "tune_games_sha256": sha256_text("\n".join(sorted(tune_games))),
        }),
        "tune_benchmark": {
            "n_positions": tune_labels["n_positions"],
            "n_candidates": tune_labels["n_candidates"],
            "stockfish": tune_labels["stockfish"],
            "labels_sha256": B.sha256_file(labels_path),
        },
        "candidate_set": {"divisors": list(DIVISORS),
                          "fit_variants": list(FIT_VARIANTS),
                          "closed": True},
        "tie_rule": {"spearman_window": TIE_SPEARMAN,
                     "then": "smaller divisor, then variant A"},
        "grid": grid,
        "selection": {
            "key": chosen_key,
            "divisor": chosen["divisor"],
            "fit_variant": chosen["fit_variant"],
            "mean_spearman": chosen["mean_spearman_across_seeds"],
            "mean_top1_pct": chosen["mean_top1_across_seeds"],
            "n_contenders_within_tie": len(contenders),
            "contenders": [k for k, _ in contenders],
            "runner_up": runner_up[0],
            "runner_up_mean_spearman": runner_up[1]["mean_spearman_across_seeds"],
            "margin_over_runner_up": (chosen["mean_spearman_across_seeds"]
                                      - runner_up[1]["mean_spearman_across_seeds"]),
            "production_divisor_row_mean_spearman":
                grid[prod_key]["mean_spearman_across_seeds"],
            "collapses_to_variant_b": chosen["divisor"] == 200,
        },
    }
    path = args.out_dir / "stage2_selection.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWROTE {path.relative_to(REPO_ROOT).as_posix()}")
    print("\nHARD STOP: Stage 2b (suite evaluation) is a separate module and was NOT run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
