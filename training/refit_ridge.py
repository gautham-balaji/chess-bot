"""C6 Stage 1 - Ridge compatibility diagnostic (coefficient inspection only).

    python -m training.refit_ridge                       # A1R default: A0 + A1
    python -m training.refit_ridge --arms A0 A1 A2 \\
        --out-dir training/experiments/A2R --stage A2R-stage1

Answers: would refitting the Ridge fusion layer for each arm's own CNN materially
change how the engine RANKS moves? Stage 1 does NOT run the engine; it fits the
refit Ridges and inspects their coefficients. Stage 2 (engine evaluation) is
deliberately not implemented here.

Design: docs/C6_A1R_RIDGE_DIAGNOSTIC_DESIGN.md

--------------------------------------------------------------------------
METHOD
--------------------------------------------------------------------------
For each of the six trained models (A0 x3, A1 x3):

  1. Compute cnn_norm = tanh(cnn(position) / 200) on the 1,933 dataset_v1 TEST
     split positions. Out-of-sample for that model, which is the point: the
     production Ridge was fitted partly in-sample.
  2. Assemble [cnn_norm, material, space, center, mobility] on the POSITION
     ITSELF - matching how the production Ridge was fitted (notebook cell 31).
  3. Fit Ridge(alpha=1.0) against THAT ARM'S OWN label policy.
  4. Report coefficients, ratios normalised to w[0], cosine similarity to the
     production vector, inner-split R2, and the implied per-term RANKING spread
     on evaluation positions.

  5. Report how often the CNN term is DECISIVE for the top-ranked candidate, and
     whether the refit coefficients would rank a different candidate first than
     the production coefficients do. Both are proxies for the engine's choice -
     `rerank_moves` also adds unscaled heuristic bonuses and a 1-ply lookahead -
     but they are far tighter evidence about rankings than coefficients alone.

Nothing production is read-write: models/weight_model.pkl is loaded read-only as
the comparison reference. Output goes to an experiments directory (gitignored).

The arm set, seeds, output directory and stage label are all selectable, so the
same methodology serves A1R (A0 + A1) and A2R (A0 + A1 + A2) without being
reimplemented. Running the module with NO arguments reproduces the committed A1R
Stage 1 configuration exactly.
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

STAGE = "A1R-stage1"

# Every arm this diagnostic knows how to fit, with the label policy whose target
# it must use. ARMS below stays at the A1R default so that running the module
# with no arguments reproduces the committed A1R Stage 1 artifact byte for byte;
# A2R selects its arms explicitly on the command line.
ARM_POLICIES = {
    "A0": D.LABEL_POLICY_LEGACY,
    "A1": D.LABEL_POLICY_CORRECTED_MATE,
    "A2": D.LABEL_POLICY_CORRECTED_MATE_WHITE,
    # C9: the C8a CNN. Same label policy as A2; its fit population is
    # dataset_v2's test split, selected with --dataset-prefix.
    "C8a": D.LABEL_POLICY_CORRECTED_MATE_WHITE,
}
ARMS = {"A0": D.LABEL_POLICY_LEGACY, "A1": D.LABEL_POLICY_CORRECTED_MATE}
SEEDS = (0, 1, 2)
FEATURE_NAMES = ["cnn_norm", "material", "space", "center", "mobility"]

DATASET = REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl"
MANIFEST = REPO_ROOT / "training" / "artifacts" / "dataset_v1.manifest.json"
PRODUCTION_RIDGE = REPO_ROOT / "models" / "weight_model.pkl"
EXPERIMENTS = REPO_ROOT / "training" / "experiments"
OUT_DIR = EXPERIMENTS / "A1R"

RIDGE_ALPHA = 1.0           # matches notebook cell 31
INNER_SPLIT_SEED = 1234     # sanity-check split, distinct from the dataset split seed
INNER_TEST_FRACTION = 0.2
RANKING_POSITIONS = 80      # same subset as the design document's section 5 table
TANH_SCALE = 200.0


# ============================================================ pure helpers

def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def normalised_ratios(coef) -> list:
    """Coefficients divided by the cnn_norm coefficient.

    Ranking within a position depends only on the RELATIVE weighting of terms, so
    these ratios - not the raw magnitudes - are what can reorder moves through
    what the design document calls channel (a).
    """
    coef = np.asarray(coef, dtype=float)
    if coef[0] == 0:
        return [float("nan")] * len(coef)
    return [float(c / coef[0]) for c in coef]


def r_squared(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def board_features(board: chess.Board, cnn_score: float,
                   tanh_scale: float = TANH_SCALE) -> list:
    """The five Ridge features, in the order notebook cell 31 used.

    `tanh_scale` defaults to the production 200.0, so every existing caller is
    unaffected. C9's divisor sweep varies it to measure how much signal the fixed
    squash discards; that sweep is a MEASUREMENT and changes no production file.
    """
    import engine as E  # read-only use of the identical helpers
    return [
        float(np.tanh(cnn_score / tanh_scale)),
        float(E.material_balance(board)),
        float(E.space_control(board)),
        float(E.center_control(board)),
        float(E.mobility_score(board)),
    ]


# ============================================================ data

def load_test_split():
    """The 1,933 dataset_v1 test-split records, checksum-verified."""
    import hashlib
    digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    manifest = D.load_manifest(MANIFEST)
    if digest != manifest["artifact"]["records_sha256"]:
        raise SystemExit("ERROR: dataset checksum does not match its manifest")

    records = D.load_records(DATASET)
    D.validate_records(records)
    split = D.make_split(records)
    return [records[i] for i in split.test_index], records, split, digest


def load_fit_population(prefix=None):
    """The positions the Ridge is fitted on.

    `prefix=None` is the A1R/A2R default and delegates to `load_test_split()`
    unchanged - dataset_v1's 1,933-record test split.

    A prefix such as `training/artifacts/dataset_v2` instead loads that dataset's
    ALREADY-MATERIALISED test split (`<prefix>.test.jsonl`), verified against its
    manifest. That split is game-level separated from its own train split and was
    scrubbed of evaluation-suite placements when it was built, so it is a strictly
    cleaner fit population than dataset_v1's. No split is re-derived here.
    """
    if prefix is None:
        return load_test_split()

    import hashlib
    prefix = Path(prefix)
    manifest_path = Path(f"{prefix}.manifest.json")
    test_path = Path(f"{prefix}.test.jsonl")
    for path in (manifest_path, test_path):
        if not path.is_file():
            raise SystemExit(f"ERROR: missing {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if digest != manifest["artifact"]["test_sha256"]:
        raise SystemExit(
            f"ERROR: {test_path.name} does not match its manifest: "
            f"file {digest} vs manifest "
            + manifest['artifact']['test_sha256'])

    records = D.load_records(test_path)
    declared = manifest["final"]["test"]["records"]
    if len(records) != declared:
        raise SystemExit(f"ERROR: {test_path.name} holds {len(records)} records, "
                         f"manifest says {declared}")
    return records, records, manifest, digest


def load_model_for(arm: str, seed: int):
    from keras.models import load_model
    path = EXPERIMENTS / arm / f"seed_{seed}" / "models" / "cnn_model.keras"
    if not path.is_file():
        raise SystemExit(f"ERROR: missing trained model {path}")
    return load_model(path, compile=False), path


# ============================================================ fitting

def fit_ridge(X, y, alpha: float = RIDGE_ALPHA):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


def inner_split_indices(n: int, seed: int = INNER_SPLIT_SEED,
                        frac: float = INNER_TEST_FRACTION):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(n * frac))
    return np.sort(order[n_test:]), np.sort(order[:n_test])


# ============================================================ ranking spread

def ranking_spread(coef, cnn_by_position, features_by_position,
                   tanh_scale: float = TANH_SCALE) -> dict:
    """Median across positions of each weighted term's spread ACROSS candidates.

    This is the quantity that decides move ORDER. Absolute output scale does not
    reorder anything; only the relative spread of the weighted terms does.
    """
    coef = np.asarray(coef, dtype=float)
    per_term = {name: [] for name in FEATURE_NAMES}
    for fen, cnn in cnn_by_position.items():
        board_feats = features_by_position[fen]          # (n_candidates, 4)
        cnn_norm = np.tanh(np.asarray(cnn, float) / tanh_scale)
        terms = np.column_stack([coef[0] * cnn_norm] +
                                [coef[i + 1] * board_feats[:, i] for i in range(4)])
        for i, name in enumerate(FEATURE_NAMES):
            per_term[name].append(float(np.std(terms[:, i])))

    medians = {name: st.median(vals) for name, vals in per_term.items()}
    total = sum(abs(v) for v in medians.values())
    return {
        "median_term_sd": {k: round(v, 4) for k, v in medians.items()},
        "cnn_share_pct": round(100 * abs(medians["cnn_norm"]) / total, 3) if total else None,
        "total_term_sd": round(total, 4),
    }


# ============================================================ decisive contribution
#
# Ranking share says how much spread each term contributes. It does not say
# whether that spread ever CHANGES the chosen move. These two measures do, and
# they are what the Stage 1 gate needs in order to talk about rankings rather
# than about coefficients.
#
# IMPORTANT: this is a PROXY for the engine's choice, not the engine's choice.
# `rerank_moves` adds unscaled heuristic bonuses and a 1-ply lookahead term on
# top of the weighted sum, so a proxy disagreement does not guarantee an engine
# disagreement, nor the reverse. Only Stage 2 measures the engine.

def _weighted_terms(coef, cnn_values, board_feats):
    coef = np.asarray(coef, dtype=float)
    cnn_norm = np.tanh(np.asarray(cnn_values, float) / TANH_SCALE)
    return np.column_stack([coef[0] * cnn_norm] +
                           [coef[i + 1] * board_feats[:, i] for i in range(4)])


def _best_index(scores, maximise: bool) -> int:
    """The engine scores White-positive and encodes the side in the sort
    direction, so the mover's best candidate is the max for White and the min
    for Black."""
    return int(np.argmax(scores)) if maximise else int(np.argmin(scores))


def decisive_contribution(coef, cnn_by_position, features_by_position,
                          maximise_by_position) -> dict:
    """How often the CNN term actually determines which move ranks first."""
    decisive = dictates = n = 0
    for fen, cnn in cnn_by_position.items():
        feats = features_by_position[fen]
        if len(feats) < 2:
            continue
        up = maximise_by_position[fen]
        terms = _weighted_terms(coef, cnn, feats)
        full = _best_index(terms.sum(axis=1), up)
        without_cnn = _best_index(terms[:, 1:].sum(axis=1), up)
        cnn_only = _best_index(terms[:, 0], up)
        decisive += int(full != without_cnn)
        dictates += int(full == cnn_only)
        n += 1
    return {
        "n_positions": n,
        # the CNN changed the top move relative to the heuristics alone
        "cnn_decisive_pct": round(100 * decisive / n, 3) if n else None,
        # the top move is simply the CNN's own favourite
        "cnn_dictates_top_move_pct": round(100 * dictates / n, 3) if n else None,
    }


def top_move_agreement(coef_a, coef_b, cnn_by_position, features_by_position,
                       maximise_by_position) -> dict:
    """Do two coefficient vectors rank the same candidate first?

    This is the most direct Stage 1 evidence about whether a refit could change
    engine rankings - still a proxy (see the note above), but a far tighter one
    than a coefficient comparison.
    """
    same = n = 0
    for fen, cnn in cnn_by_position.items():
        feats = features_by_position[fen]
        if len(feats) < 2:
            continue
        up = maximise_by_position[fen]
        a = _best_index(_weighted_terms(coef_a, cnn, feats).sum(axis=1), up)
        b = _best_index(_weighted_terms(coef_b, cnn, feats).sum(axis=1), up)
        same += int(a == b)
        n += 1
    return {
        "n_positions": n,
        "same_top_move_pct": round(100 * same / n, 3) if n else None,
        "changed_top_move_pct": round(100 * (n - same) / n, 3) if n else None,
        "note": "proxy: the weighted sum only, without the engine's unscaled "
                "heuristic bonuses or its 1-ply lookahead term",
    }


def perturbation_sensitivity(reference) -> dict:
    """How much each coefficient must move before cosine notices.

    A1R Stage 1 found cosine to be a poor discriminator here because w[0]
    dominates the norm. This records that as a measured artifact rather than a
    remark, so the A2R gate cannot lean on cosine by accident.
    """
    ref = np.asarray(reference, dtype=float)
    out = {}
    for i, name in enumerate(FEATURE_NAMES):
        for factor in (2.0, 10.0, 100.0):
            v = ref.copy()
            v[i] *= factor
            out[f"{name}_x{factor:g}"] = round(cosine_similarity(ref, v), 6)
    v = ref.copy()
    v[2] *= -1.0
    out["space_sign_flip"] = round(cosine_similarity(ref, v), 6)
    out["uniform_x10"] = round(cosine_similarity(ref, ref * 10.0), 6)
    return out


# ============================================================ main

def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="C6 Ridge compatibility diagnostic, Stage 1 (no engine evaluation).")
    ap.add_argument("--arms", nargs="+", default=sorted(ARMS),
                    choices=sorted(ARM_POLICIES),
                    help="arms to fit (default: the A1R pair A0 A1)")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="output directory (default: the A1R directory)")
    ap.add_argument("--stage", default=STAGE, help="stage label recorded in the artifact")
    ap.add_argument("--dataset-prefix", default=None,
                    help="fit-population prefix, e.g. training/artifacts/dataset_v2. "
                         "Default None = dataset_v1 test split (the A1R behaviour)")
    args = ap.parse_args(argv)

    arms = {a: ARM_POLICIES[a] for a in args.arms}
    seeds = tuple(args.seeds)
    out_dir, stage = args.out_dir, args.stage

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{stage}: Ridge compatibility diagnostic (Stage 1 only - no engine evaluation)")
    print(f"arms: {sorted(arms)}   seeds: {list(seeds)}   out: {out_dir}\n")

    test_records, all_records, split, dataset_sha = load_test_split()
    print(f"dataset_v1 sha256 {dataset_sha[:16]}... (manifest OK)")
    print(f"test split: {len(test_records)} records (split_seed={split.split_seed})")

    production = pickle.load(open(PRODUCTION_RIDGE, "rb"))
    w_prod = np.asarray(production.coef_, dtype=float)
    print(f"production Ridge coef: {[round(float(x), 4) for x in w_prod]}")
    print(f"production intercept : {float(production.intercept_):.4f} "
          f"(NOT applied by engine.py - defect C3)\n")

    # Board features are model-independent, so compute them once.
    print("caching board features...")
    test_boards = [chess.Board(r["fen"]) for r in test_records]

    eval_fens = [p["fen"] for p in json.loads(
        (REPO_ROOT / "evaluation" / "positions" / "extended.json").read_text(
            encoding="utf-8"))["positions"]][:RANKING_POSITIONS]
    import engine as E
    rank_board_feats, rank_candidates, rank_maximise = {}, {}, {}
    for fen in eval_fens:
        b = chess.Board(fen)
        rows, posts = [], []
        rank_maximise[fen] = b.turn == chess.WHITE
        for mv in b.legal_moves:
            b.push(mv)
            rows.append([E.material_balance(b), E.space_control(b),
                         E.center_control(b), E.mobility_score(b)])
            posts.append(b.fen())
            b.pop()
        rank_board_feats[fen] = np.asarray(rows, dtype=float)
        rank_candidates[fen] = posts
    print(f"ranking analysis on {len(eval_fens)} extended positions "
          f"({sum(len(v) for v in rank_candidates.values())} candidate moves)\n")

    X_test_encoded = R.encode_many(r["fen"] for r in test_records)
    results = []

    for arm, policy in arms.items():
        y = D.apply_label_policy(test_records, policy).astype(np.float64)
        for seed in seeds:
            model, model_path = load_model_for(arm, seed)

            # --- features on the position itself, as cell 31 did ---------------
            cnn_test = model.predict(X_test_encoded, verbose=0).flatten().astype(np.float64)
            X = np.array([board_features(b, c) for b, c in zip(test_boards, cnn_test)])

            # --- inner split sanity check --------------------------------------
            tr, te = inner_split_indices(len(X))
            inner = fit_ridge(X[tr], y[tr])
            r2_in = r_squared(y[tr], inner.predict(X[tr]))
            r2_out = r_squared(y[te], inner.predict(X[te]))

            # --- final refit on all 1,933 --------------------------------------
            refit = fit_ridge(X, y)
            w = np.asarray(refit.coef_, dtype=float)

            # --- ranking spread on evaluation positions ------------------------
            cnn_by_pos = {}
            for fen in eval_fens:
                enc = R.encode_many(rank_candidates[fen])
                cnn_by_pos[fen] = model.predict(enc, verbose=0).flatten().astype(np.float64)

            spread_prod = ranking_spread(w_prod, cnn_by_pos, rank_board_feats)
            spread_refit = ranking_spread(w, cnn_by_pos, rank_board_feats)

            row = {
                "arm": arm, "seed": seed, "label_policy": policy,
                "model_path": str(model_path.relative_to(REPO_ROOT).as_posix()),
                "n_fit_positions": int(len(X)),
                "coef": [float(x) for x in w],
                "intercept": float(refit.intercept_),
                "ratios_normalised_to_cnn": normalised_ratios(w),
                "production_ratios": normalised_ratios(w_prod),
                "cosine_similarity_to_production": cosine_similarity(w, w_prod),
                "scale_factor_vs_production": float(w[0] / w_prod[0]),
                "inner_split": {"seed": INNER_SPLIT_SEED, "n_train": int(len(tr)),
                                "n_test": int(len(te)),
                                "r2_train": round(r2_in, 6), "r2_holdout": round(r2_out, 6)},
                "ranking_spread_production_coef": spread_prod,
                "ranking_spread_refit_coef": spread_refit,
                "cnn_share_shift_pct_points": round(
                    spread_refit["cnn_share_pct"] - spread_prod["cnn_share_pct"], 3),
                "decisive_production_coef": decisive_contribution(
                    w_prod, cnn_by_pos, rank_board_feats, rank_maximise),
                "decisive_refit_coef": decisive_contribution(
                    w, cnn_by_pos, rank_board_feats, rank_maximise),
                "top_move_agreement_production_vs_refit": top_move_agreement(
                    w_prod, w, cnn_by_pos, rank_board_feats, rank_maximise),
            }
            results.append(row)
            print(f"  {arm} seed {seed}: coef={[round(x, 3) for x in w]}")
            print(f"      cos={row['cosine_similarity_to_production']:.6f}  "
                  f"scale={row['scale_factor_vs_production']:.4f}  "
                  f"R2(holdout)={r2_out:.4f}  "
                  f"CNN share {spread_prod['cnn_share_pct']:.2f}% -> "
                  f"{spread_refit['cnn_share_pct']:.2f}%")
            print(f"      CNN decisive {row['decisive_production_coef']['cnn_decisive_pct']:.1f}%"
                  f" -> {row['decisive_refit_coef']['cnn_decisive_pct']:.1f}%   "
                  f"top move changed by refit: "
                  f"{row['top_move_agreement_production_vs_refit']['changed_top_move_pct']:.1f}%")

    payload = {
        "stage": stage,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": {
            "arms": {a: p for a, p in arms.items()},
            "seeds": list(seeds),
            "fit_positions": "dataset_v1 TEST split (out-of-sample for every model)",
            "features": FEATURE_NAMES,
            "features_computed_on": "the position itself (matches notebook cell 31)",
            "target": "each arm's own label policy",
            "ridge_alpha": RIDGE_ALPHA,
            "inner_split_seed": INNER_SPLIT_SEED,
            "ranking_positions": len(eval_fens),
            "ranking_features_computed_on": "post-move positions (matches rerank_moves)",
        },
        "dataset_sha256": dataset_sha,
        "production_coef": [float(x) for x in w_prod],
        "production_intercept": float(production.intercept_),
        "cosine_perturbation_sensitivity_of_production": perturbation_sensitivity(w_prod),
        "results": results,
    }
    out = out_dir / "stage1_results.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
