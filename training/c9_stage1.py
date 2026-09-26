"""C9 Stage 1 — matched-Ridge and squash-divisor diagnostic for C8a.

    python -m training.c9_stage1

Read-only over trained models. **No Stockfish, no engine evaluation, no
training, no production write.** Writes
`training/experiments/C9/stage1_results.json`.

Design: docs/C9_RIDGE_AUDIT.md   Prior stages: docs/C6_A1R_STAGE1_REPORT.md,
docs/C6_A1R_STAGE2_REPORT.md, docs/C6_A2R_STAGE1_REPORT.md

--------------------------------------------------------------------------
WHAT THIS MEASURES, AND WHY THERE ARE TWO PARTS
--------------------------------------------------------------------------
The engine scores a candidate move as

    w[0]*tanh(cnn/200) + w[1]*material + w[2]*space + w[3]*center + w[4]*mobility

`w` comes from `models/weight_model.pkl`. The divisor `200` is a literal in
`engine.py`. The C9 audit measured that C8a's CNN correlates 0.876 with the
target but only 0.704 after the production squash, so the two components must be
separated:

  PART 1  matched Ridge - refit `w` to C8a's own output distribution.
          This can only rescale and rebalance the five terms.

  PART 2  divisor sweep - measure corr(tanh(cnn/d), label) for a range of `d`.
          This quantifies signal the squash DISCARDS BEFORE the Ridge sees it,
          which no choice of `w` can recover.

**A large Part 2 effect is not evidence that the Ridge refit works.** The two are
reported separately and the gate is evaluated on Part 1 alone, exactly as the
audit pre-registered.

--------------------------------------------------------------------------
TWO RIDGE VARIANTS, BOTH REPORTED
--------------------------------------------------------------------------
1,129 of the 13,712 fit positions are checkmated boards with zero legal moves and
labels of +/-2000. The engine never ranks moves in such a position. They are
neither silently kept nor silently dropped: variant A fits all 13,712, variant B
excludes the 1,129, and both are reported.

--------------------------------------------------------------------------
TWO RANKING POPULATIONS, BOTH REPORTED
--------------------------------------------------------------------------
Ranking spread is a property of candidate moves, so it needs post-move boards.

  `extended`  the first 80 positions, as A1R/A2R used. Needed because the gate's
              A0 reference band (82.85-84.93%) was measured there; comparing
              against it on a different population would be meaningless. No
              label and no fit ever touches these - they supply board features
              and CNN outputs only.
  `heldout`   candidate moves generated from dataset_v2 test positions. Carries
              no evaluation-suite contact at all, and confirms the `extended`
              reading is not an artifact of that population.

The Ridge is fitted ONLY on dataset_v2's test split. Neither ranking population
contributes a single row to any fit.
"""
from __future__ import annotations

import json
import os
import pickle
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

from training import dataset as D  # noqa: E402
from training import representation as R  # noqa: E402
from training import refit_ridge as RR  # noqa: E402

STAGE = "C9-stage1"
ARM = "C8a"
SEEDS = (0, 1, 2)
DATASET_PREFIX = REPO_ROOT / "training" / "artifacts" / "dataset_v2"
OUT_DIR = REPO_ROOT / "training" / "experiments" / "C9"

# Divisors swept in Part 2. 200 is production.
DIVISORS = (200, 400, 600, 800, 1000, 1500, 2000)
SATURATION_THRESHOLD = 0.95

# The A0 seed spread of CNN ranking share under PRODUCTION coefficients,
# measured in A1R Stage 1 (docs/C6_A1R_STAGE1_REPORT.md). The gate compares the
# C8a refit's share against this band.
A0_REFERENCE_BAND = (82.85, 84.93)

# How many held-out positions supply the leakage-free ranking population. Taken
# in file order - the file is already sorted canonically by (game, ply), so this
# is deterministic and needs no RNG.
HELDOUT_RANKING_POSITIONS = 80

# C8a model weight hashes recorded in docs/C9_RIDGE_AUDIT.md. Verified before
# anything is fitted, so the diagnostic cannot silently run on the wrong models.
EXPECTED_C8A_WEIGHTS = {
    0: "6bd57636a2747ec479992fb0873385167956352d776808ee2ef79af023ad36e2",
    1: "144594ffff30bdfa3b59eafc8897e811bbaa6ab269c1d5c82a29364e79a24b97",
    2: "41ee1626feff429a956c2125d83b2dcf5cf86b93ad04a86e3fb556d27152071c",
}


# ============================================================ verification

def verify_models() -> dict:
    """Confirm the three C8a models are the ones the audit measured."""
    out = {}
    for seed in SEEDS:
        meta_path = (REPO_ROOT / "training" / "experiments" / ARM /
                     f"seed_{seed}" / "metadata.json")
        if not meta_path.is_file():
            raise SystemExit(f"ERROR: missing {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        got = meta["artifacts"]["model_weights_sha256"]
        if got != EXPECTED_C8A_WEIGHTS[seed]:
            raise SystemExit(
                f"ERROR: C8a seed {seed} weights sha256 {got} does not match the "
                f"audited {EXPECTED_C8A_WEIGHTS[seed]}. Refusing to run.")
        if meta["model_parameters"] != 2_360_129:
            raise SystemExit(f"ERROR: C8a seed {seed} is not the A2 architecture")
        out[f"seed_{seed}"] = {
            "model_weights_sha256": got,
            "matches_audit": True,
            "model_parameters": meta["model_parameters"],
            "trained_on": meta["dataset"]["name"],
            "train_records": meta["dataset"]["n_train"],
        }
    return out


def leakage_checks(fit_records: list[dict]) -> dict:
    """Confirm the fit population touches neither evaluation suite nor C8a train."""
    train = D.load_records(Path(f"{DATASET_PREFIX}.train.jsonl"))
    train_pl = {r["placement"] for r in train}
    train_g = {r["game_content_key"] for r in train}
    fit_pl = {r["placement"] for r in fit_records}
    fit_g = {r["game_content_key"] for r in fit_records}

    suite_pl, suite_fen, counts = set(), set(), {}
    for suite in ("extended", "phase0_52"):
        data = json.loads((REPO_ROOT / "evaluation" / "positions" /
                           f"{suite}.json").read_text(encoding="utf-8"))
        fens = [p["fen"] for p in data["positions"]]
        counts[suite] = len(fens)
        suite_pl.update(f.split(" ")[0] for f in fens)
        suite_fen.update(fens)

    checks = {
        "fit_records": len(fit_records),
        "fit_games": len(fit_g),
        "suite_position_counts": counts,
        "overlap_with_c8a_train_placements": len(fit_pl & train_pl),
        "overlap_with_c8a_train_games": len(fit_g & train_g),
        "overlap_with_suite_placements": len(fit_pl & suite_pl),
        "overlap_with_suite_exact_fens": len({r["fen"] for r in fit_records} & suite_fen),
    }
    bad = [k for k in checks if k.startswith("overlap_") and checks[k]]
    if bad:
        raise SystemExit(f"ERROR: fit population is contaminated: "
                         f"{ {k: checks[k] for k in bad} }")
    checks["all_clean"] = True
    return checks


# ============================================================ ranking populations

def candidate_population(fens: list[str]) -> dict:
    """Post-move board features and metadata for each position's legal moves."""
    import engine as E
    board_feats, posts, maximise = {}, {}, {}
    for fen in fens:
        b = chess.Board(fen)
        rows, children = [], []
        maximise[fen] = b.turn == chess.WHITE
        for mv in b.legal_moves:
            b.push(mv)
            rows.append([E.material_balance(b), E.space_control(b),
                         E.center_control(b), E.mobility_score(b)])
            children.append(b.fen())
            b.pop()
        if not children:            # mated / stalemated: no moves to rank
            continue
        board_feats[fen] = np.asarray(rows, dtype=float)
        posts[fen] = children
        maximise[fen] = maximise[fen]
    return {"features": board_feats, "candidates": posts, "maximise": maximise}


def cnn_on_candidates(model, pop: dict) -> dict:
    return {fen: model.predict(R.encode_many(children), verbose=0)
            .flatten().astype(np.float64)
            for fen, children in pop["candidates"].items()}


# ============================================================ main

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="C9 Stage 1: matched Ridge + divisor sweep. No engine evaluation.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dataset-prefix", type=Path, default=DATASET_PREFIX)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args(argv)

    seeds = tuple(args.seeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{STAGE}: matched-Ridge + divisor diagnostic "
          f"(Stage 1 only - NO engine evaluation, NO Stockfish)\n")

    # ---------------------------------------------------------------- inputs
    models_meta = verify_models()
    print("C8a model verification:")
    for k, v in models_meta.items():
        print(f"  {k}: {v['model_weights_sha256'][:20]}... matches audit, "
              f"{v['model_parameters']:,} params, trained on {v['trained_on']}")

    fit_records, _, manifest, digest = RR.load_fit_population(args.dataset_prefix)
    print(f"\nfit population: {Path(args.dataset_prefix).name}.test.jsonl")
    print(f"  {len(fit_records)} records, sha256 {digest[:16]}... (manifest OK)")

    leak = leakage_checks(fit_records)
    print(f"  leakage: suite placements {leak['overlap_with_suite_placements']}, "
          f"suite FENs {leak['overlap_with_suite_exact_fens']}, "
          f"C8a-train placements {leak['overlap_with_c8a_train_placements']}, "
          f"C8a-train games {leak['overlap_with_c8a_train_games']}  -> CLEAN")

    policy = RR.ARM_POLICIES[ARM]
    y_all = D.apply_label_policy(fit_records, policy).astype(np.float64)
    stored = np.array([r["label"] for r in fit_records], dtype=np.float64)
    if not np.array_equal(y_all, stored):
        raise SystemExit("ERROR: re-derived labels differ from the stored labels")
    print(f"  labels: {policy} (re-derived and verified)")

    mated = np.array([bool(r["is_checkmate"]) for r in fit_records])
    print(f"  checkmate positions: {int(mated.sum())} "
          f"({100 * mated.mean():.1f}%) - variant B excludes these")

    production = pickle.load(open(RR.PRODUCTION_RIDGE, "rb"))
    w_prod = np.asarray(production.coef_, dtype=float)
    print(f"\nproduction Ridge coef : {[round(float(x), 4) for x in w_prod]}")
    print(f"production intercept  : {float(production.intercept_):.4f} "
          f"(NOT applied by engine.py - defect C3)")

    # ---------------------------------------------------------------- ranking pops
    ext_fens = [p["fen"] for p in json.loads(
        (REPO_ROOT / "evaluation" / "positions" / "extended.json")
        .read_text(encoding="utf-8"))["positions"]][:RR.RANKING_POSITIONS]
    held_fens = [r["fen"] for r in fit_records
                 if not r["is_checkmate"]][:HELDOUT_RANKING_POSITIONS]
    pops = {"extended": candidate_population(ext_fens),
            "heldout": candidate_population(held_fens)}
    for name, pop in pops.items():
        print(f"ranking population '{name}': {len(pop['candidates'])} positions, "
              f"{sum(len(v) for v in pop['candidates'].values())} candidate moves")

    boards = [chess.Board(r["fen"]) for r in fit_records]
    X_fit = R.encode_many(r["fen"] for r in fit_records)

    out = {
        "stage": STAGE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Stage 1 diagnostic only. No engine evaluation, no Stockfish, no "
                 "training, no production write. A large divisor effect is NOT "
                 "evidence that the coefficient refit works; the gate is "
                 "evaluated on the coefficient refit alone."),
        "arm": ARM,
        "label_policy": policy,
        "seeds": list(seeds),
        "ridge_alpha": RR.RIDGE_ALPHA,
        "production_ridge": {
            "coef": [float(x) for x in w_prod],
            "intercept": float(production.intercept_),
            "intercept_applied_by_engine": False,
            "tanh_divisor": RR.TANH_SCALE,
        },
        "c8a_models": models_meta,
        "fit_population": {
            "prefix": str(Path(args.dataset_prefix).as_posix()),
            "test_sha256": digest,
            "dataset_name": manifest.get("dataset_name"),
            "records": len(fit_records),
            "checkmate_records": int(mated.sum()),
            **leak,
        },
        "a0_reference_band_cnn_share_pct": list(A0_REFERENCE_BAND),
        "divisors": list(DIVISORS),
        "per_seed": {},
    }

    import keras
    print(f"\n{'=' * 108}")
    for seed in seeds:
        model, model_path = RR.load_model_for(ARM, seed)
        row = {"model_path": str(model_path.relative_to(REPO_ROOT).as_posix())}

        cnn_fit = model.predict(X_fit, verbose=0).flatten().astype(np.float64)
        cnn_by_pos = {name: cnn_on_candidates(model, pop) for name, pop in pops.items()}

        # ------------------------------------------------ PART 1: matched Ridge
        variants = {}
        for label, mask in (("all_positions", np.ones(len(fit_records), bool)),
                            ("excluding_checkmate", ~mated)):
            X = np.array([RR.board_features(b, c)
                          for b, c, m in zip(boards, cnn_fit, mask) if m])
            y = y_all[mask]
            tr, te = RR.inner_split_indices(len(X))
            inner = RR.fit_ridge(X[tr], y[tr])
            refit = RR.fit_ridge(X, y)
            w = np.asarray(refit.coef_, dtype=float)

            block = {
                "n_fit_positions": int(len(X)),
                "coef": [float(x) for x in w],
                "intercept": float(refit.intercept_),
                "ratios_normalised_to_cnn": RR.normalised_ratios(w),
                "production_ratios": RR.normalised_ratios(w_prod),
                "cosine_similarity_to_production": RR.cosine_similarity(w, w_prod),
                "scale_factor_vs_production": float(w[0] / w_prod[0]),
                "inner_split": {
                    "seed": RR.INNER_SPLIT_SEED,
                    "n_train": int(len(tr)), "n_test": int(len(te)),
                    "r2_train": RR.r_squared(y[tr], inner.predict(X[tr])),
                    "r2_heldout": RR.r_squared(y[te], inner.predict(X[te])),
                },
                "r2_full_fit": RR.r_squared(y, refit.predict(X)),
                "ranking": {},
            }
            for pname, pop in pops.items():
                sp = RR.ranking_spread(w, cnn_by_pos[pname], pop["features"])
                sp_prod = RR.ranking_spread(w_prod, cnn_by_pos[pname], pop["features"])
                agree = RR.top_move_agreement(w_prod, w, cnn_by_pos[pname],
                                              pop["features"], pop["maximise"])
                block["ranking"][pname] = {
                    "production": sp_prod, "refit": sp,
                    "cnn_share_delta_pp": round(
                        sp["cnn_share_pct"] - sp_prod["cnn_share_pct"], 3),
                    "top_move_agreement": agree,
                }
            variants[label] = block

        row["matched_ridge"] = variants

        # ------------------------------------------------ PART 2: divisor sweep
        sweep = {}
        for d in DIVISORS:
            t = np.tanh(cnn_fit / d)
            sweep[str(d)] = {
                "corr_with_label": float(np.corrcoef(t, y_all)[0, 1]),
                "saturated_fraction": float(np.mean(np.abs(t) > SATURATION_THRESHOLD)),
                "mean_abs_tanh": float(np.abs(t).mean()),
            }
            # implied CNN ranking spread at this divisor, production coefficients
            for pname, pop in pops.items():
                sp = RR.ranking_spread(w_prod, cnn_by_pos[pname], pop["features"],
                                       tanh_scale=float(d))
                sweep[str(d)].setdefault("cnn_term_sd", {})[pname] = \
                    sp["median_term_sd"]["cnn_norm"]
                sweep[str(d)].setdefault("cnn_share_pct", {})[pname] = sp["cnn_share_pct"]
        row["divisor_sweep"] = {
            "raw_cnn": {
                "corr_with_label": float(np.corrcoef(cnn_fit, y_all)[0, 1]),
                "mean_abs": float(np.abs(cnn_fit).mean()),
                "std": float(cnn_fit.std()),
            },
            "by_divisor": sweep,
        }
        out["per_seed"][f"seed_{seed}"] = row

        v = variants["all_positions"]
        print(f"C8a seed {seed}")
        print(f"  refit coef      {[round(x, 4) for x in v['coef']]}")
        print(f"  cosine vs prod  {v['cosine_similarity_to_production']:.6f}   "
              f"w0 scale {v['scale_factor_vs_production']:.4f}")
        for pname in pops:
            rk = v["ranking"][pname]
            print(f"  [{pname:8s}] CNN share  production "
                  f"{rk['production']['cnn_share_pct']:.2f}%  ->  refit "
                  f"{rk['refit']['cnn_share_pct']:.2f}%   "
                  f"({rk['cnn_share_delta_pp']:+.2f} pp)   "
                  f"top move changed {rk['top_move_agreement']['changed_top_move_pct']:.2f}%")
        sw = row["divisor_sweep"]
        print(f"  corr(tanh(cnn/d), y): " + "  ".join(
            f"d={d}:{sw['by_divisor'][str(d)]['corr_with_label']:.4f}" for d in DIVISORS))
        print(f"  raw corr {sw['raw_cnn']['corr_with_label']:.4f}   "
              f"saturated@200 {100 * sw['by_divisor']['200']['saturated_fraction']:.1f}%")

    # ---------------------------------------------------------------- gate
    print(f"\n{'=' * 108}\nGATE (coefficient refit only; the divisor sweep is NOT an input)\n{'=' * 108}")
    lo, hi = A0_REFERENCE_BAND
    gate = {"reference_band_cnn_share_pct": list(A0_REFERENCE_BAND),
            "population_used": "extended (same population the A0 band was measured on)",
            "per_seed": {}, "note": "evaluated on variant A (all positions)"}
    outside, changed = [], []
    for seed in seeds:
        rk = out["per_seed"][f"seed_{seed}"]["matched_ridge"]["all_positions"]["ranking"]["extended"]
        share = rk["refit"]["cnn_share_pct"]
        tmc = rk["top_move_agreement"]["changed_top_move_pct"]
        is_out = not (lo <= share <= hi)
        outside.append(is_out); changed.append(tmc)
        gate["per_seed"][f"seed_{seed}"] = {
            "refit_cnn_share_pct": share,
            "production_cnn_share_pct": rk["production"]["cnn_share_pct"],
            "outside_a0_band": bool(is_out),
            "top_move_changed_pct": tmc,
        }
        print(f"  seed {seed}: refit CNN share {share:7.2f}%  "
              f"(band {lo}-{hi})  outside={is_out}   top move changed {tmc:.2f}%")
    gate["all_seeds_outside_band"] = bool(all(outside))
    gate["any_seed_outside_band"] = bool(any(outside))
    gate["mean_top_move_changed_pct"] = round(st.fmean(changed), 3)
    gate["tripped"] = bool(all(outside))
    print(f"\n  GATE: {'TRIPPED' if gate['tripped'] else 'NOT TRIPPED'} "
          f"(all three seeds outside the A0 band: {gate['all_seeds_outside_band']}; "
          f"mean top-move change {gate['mean_top_move_changed_pct']:.2f}%)")
    print("  Stage 2 is NOT run by this module regardless of the gate result.")
    out["gate"] = gate

    path = args.out_dir / "stage1_results.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    try:
        shown = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        shown = str(path.resolve().as_posix())
    print(f"\nWROTE {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
