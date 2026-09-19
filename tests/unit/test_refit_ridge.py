"""Tests for the C6-A1R Stage 1 diagnostic maths (training/refit_ridge.py).

Pure arithmetic against hand-computed answers. No models, no Stockfish, no
dataset file - these are the helpers the Stage 1 conclusions rest on, so they are
checked independently of the run itself.
"""
import numpy as np
import pytest

from training import refit_ridge as RR


# ==================================================================== cosine

def test_cosine_of_identical_vectors_is_one():
    v = [330.9, 32.4, 0.79, 5.17, 0.019]
    assert RR.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_is_invariant_to_positive_scaling():
    """A pure rescale leaves cosine at 1 - which is exactly why cosine alone
    cannot detect a scale change. Recorded so the limitation is explicit."""
    v = [330.9, 32.4, 0.79, 5.17, 0.019]
    assert RR.cosine_similarity(v, [10 * x for x in v]) == pytest.approx(1.0)


def test_cosine_detects_a_direction_change():
    assert RR.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_sensitivity_is_uneven_across_coefficients():
    """Measured, not assumed. With w[0] ~ 331 dominating the norm, cosine reacts
    to changes in `material` but is nearly blind to the three small coefficients:

        material x10   -> 0.7793     (detected)
        material x2    -> 0.9954     (barely)
        center   x10   -> 0.9904     (barely)
        space    x10   -> 0.9998     (blind)
        mobility x100  -> 1.0000     (blind)
        uniform  x10   -> 1.0000     (blind by construction)

    So a high cosine does NOT by itself establish that the ranking balance is
    unchanged. The gate therefore relies on the normalised ratios and the ranking
    share, with cosine as a coarse summary only.
    """
    prod = [330.9, 32.4, 0.79, 5.17, 0.019]
    assert RR.cosine_similarity(prod, [330.9, 324.0, 0.79, 5.17, 0.019]) < 0.80
    assert RR.cosine_similarity(prod, [330.9, 32.4, 7.9, 5.17, 0.019]) > 0.999
    assert RR.cosine_similarity(prod, [330.9, 32.4, 0.79, 5.17, 1.9]) > 0.9999


def test_cosine_of_zero_vector_is_nan():
    assert np.isnan(RR.cosine_similarity([0, 0], [1, 1]))


# ==================================================================== ratios

def test_ratios_are_normalised_to_the_cnn_coefficient():
    r = RR.normalised_ratios([100.0, 10.0, 1.0, 5.0, 0.5])
    assert r == pytest.approx([1.0, 0.1, 0.01, 0.05, 0.005])


def test_ratios_are_scale_invariant():
    a = RR.normalised_ratios([100.0, 10.0, 1.0, 5.0, 0.5])
    b = RR.normalised_ratios([1000.0, 100.0, 10.0, 50.0, 5.0])
    assert a == pytest.approx(b)


def test_ratios_detect_a_changed_balance_that_cosine_hides():
    """The discriminator the gate actually uses."""
    prod = RR.normalised_ratios([330.9, 32.4, 0.79, 5.17, 0.019])
    altered = RR.normalised_ratios([330.9, 324.0, 0.79, 5.17, 0.019])
    assert altered[1] / prod[1] == pytest.approx(10.0)


def test_ratios_handle_a_zero_cnn_coefficient():
    assert all(np.isnan(x) for x in RR.normalised_ratios([0.0, 1.0, 2.0, 3.0, 4.0]))


# ==================================================================== r squared

def test_r2_of_perfect_prediction_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert RR.r_squared(y, y.copy()) == pytest.approx(1.0)


def test_r2_of_the_mean_predictor_is_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert RR.r_squared(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_r2_can_be_negative():
    y = np.array([1.0, 2.0, 3.0])
    assert RR.r_squared(y, np.array([10.0, -10.0, 30.0])) < 0


def test_r2_is_nan_for_constant_truth():
    assert np.isnan(RR.r_squared(np.array([5.0, 5.0]), np.array([1.0, 2.0])))


# ==================================================================== inner split

def test_inner_split_is_deterministic():
    a_tr, a_te = RR.inner_split_indices(1000)
    b_tr, b_te = RR.inner_split_indices(1000)
    assert np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)


def test_inner_split_partitions_without_overlap():
    tr, te = RR.inner_split_indices(1933)
    assert set(tr) & set(te) == set()
    assert len(set(tr) | set(te)) == 1933
    assert len(te) == pytest.approx(1933 * 0.2, abs=1)


def test_inner_split_seed_differs_from_the_dataset_split_seed():
    """Must not reuse split seed 42, or the sanity check would be correlated
    with the partition it is validating."""
    from training import dataset as D
    assert RR.INNER_SPLIT_SEED != D.DEFAULT_SPLIT_SEED


# ==================================================================== ranking spread

def _spread_inputs(cnn_values, board_rows):
    fen = "x"
    return ({fen: np.asarray(cnn_values, float)},
            {fen: np.asarray(board_rows, float)})


def test_ranking_spread_is_zero_when_all_candidates_are_identical():
    cnn, feats = _spread_inputs([100.0, 100.0], [[0, 0, 0, 0], [0, 0, 0, 0]])
    out = RR.ranking_spread([330.9, 32.4, 0.79, 5.17, 0.019], cnn, feats)
    assert all(v == 0 for v in out["median_term_sd"].values())


def test_ranking_spread_isolates_the_cnn_term():
    """Board features constant, CNN varying -> all spread must be in cnn_norm."""
    cnn, feats = _spread_inputs([-400.0, 400.0], [[1, 1, 1, 1], [1, 1, 1, 1]])
    out = RR.ranking_spread([330.9, 32.4, 0.79, 5.17, 0.019], cnn, feats)
    assert out["median_term_sd"]["cnn_norm"] > 0
    assert out["median_term_sd"]["material"] == 0
    assert out["cnn_share_pct"] == pytest.approx(100.0)


def test_ranking_spread_isolates_a_board_term():
    cnn, feats = _spread_inputs([100.0, 100.0], [[0, 0, 0, 0], [5, 0, 0, 0]])
    out = RR.ranking_spread([330.9, 32.4, 0.79, 5.17, 0.019], cnn, feats)
    assert out["median_term_sd"]["cnn_norm"] == 0
    assert out["median_term_sd"]["material"] > 0
    assert out["cnn_share_pct"] == pytest.approx(0.0)


def test_ranking_spread_responds_to_the_coefficient_balance():
    """Raising the material coefficient must lower the CNN share - this is the
    channel (a) effect the gate is looking for."""
    cnn, feats = _spread_inputs([-300.0, 300.0], [[0, 0, 0, 0], [3, 0, 0, 0]])
    base = RR.ranking_spread([330.9, 32.4, 0.79, 5.17, 0.019], cnn, feats)
    heavier = RR.ranking_spread([330.9, 324.0, 0.79, 5.17, 0.019], cnn, feats)
    assert heavier["cnn_share_pct"] < base["cnn_share_pct"]


def test_ranking_spread_is_invariant_to_uniform_coefficient_scaling():
    """A uniform rescale cannot reorder moves, so the SHARE must not move.

    Note this is precisely why the design document flags channel (b) separately:
    a uniform rescale is invisible here, yet it still changes how the unscaled
    heuristic bonuses and lookahead term compete with the weighted score.
    """
    cnn, feats = _spread_inputs([-300.0, 300.0], [[0, 0, 0, 0], [3, 1, 2, 4]])
    w = [330.9, 32.4, 0.79, 5.17, 0.019]
    a = RR.ranking_spread(w, cnn, feats)
    b = RR.ranking_spread([10 * x for x in w], cnn, feats)
    assert a["cnn_share_pct"] == pytest.approx(b["cnn_share_pct"])


# ==================================================================== config

def test_diagnostic_covers_both_arms_with_their_own_policies():
    from training import dataset as D
    assert RR.ARMS == {"A0": D.LABEL_POLICY_LEGACY,
                       "A1": D.LABEL_POLICY_CORRECTED_MATE}
    assert RR.SEEDS == (0, 1, 2)


def test_diagnostic_matches_the_notebook_ridge_alpha():
    assert RR.RIDGE_ALPHA == 1.0


def test_diagnostic_output_is_isolated_and_not_production():
    assert RR.OUT_DIR.name == "A1R"
    assert "experiments" in str(RR.OUT_DIR)
    assert RR.OUT_DIR.resolve() != (RR.REPO_ROOT / "models").resolve()


# ==================================================================== Stage 2 staging
#
# Stage 2 adds an optional `ridge_model` override to evaluate_arm.stage_models_dir.
# It must be ADDITIVE: the default path (used by every A0 and A1 run) has to keep
# copying the production Ridge, unchanged.

def test_default_staging_still_copies_the_production_ridge(tmp_path):
    """Default-preserving guarantee for the A0/A1 runs."""
    import pickle
    from training import evaluate_arm as EA

    fake_cnn = tmp_path / "cnn_model.keras"
    fake_cnn.write_bytes(b"cnn")
    staging = EA.stage_models_dir(fake_cnn, tmp_path / "default")

    staged = pickle.load(open(staging / "weight_model.pkl", "rb"))
    production = pickle.load(open(EA.PRODUCTION_MODELS / "weight_model.pkl", "rb"))
    assert np.allclose(staged.coef_, production.coef_)


def test_explicit_ridge_override_is_staged_instead(tmp_path):
    from training import evaluate_arm as EA

    fake_cnn = tmp_path / "cnn_model.keras"
    fake_cnn.write_bytes(b"cnn")
    custom = tmp_path / "custom_weight_model.pkl"
    custom.write_bytes(b"custom-ridge-bytes")

    staging = EA.stage_models_dir(fake_cnn, tmp_path / "override", ridge_model=custom)
    assert (staging / "weight_model.pkl").read_bytes() == b"custom-ridge-bytes"


def test_staging_override_never_writes_to_production(tmp_path):
    from training import evaluate_arm as EA

    before = {p.name: (p.stat().st_size, p.stat().st_mtime)
              for p in EA.PRODUCTION_MODELS.iterdir()}
    fake_cnn = tmp_path / "cnn_model.keras"; fake_cnn.write_bytes(b"cnn")
    custom = tmp_path / "r.pkl"; custom.write_bytes(b"x")
    EA.stage_models_dir(fake_cnn, tmp_path / "s", ridge_model=custom)
    after = {p.name: (p.stat().st_size, p.stat().st_mtime)
             for p in EA.PRODUCTION_MODELS.iterdir()}
    assert before == after


def test_matched_ridge_round_trips_the_stage1_coefficients(tmp_path):
    """Stage 2 reconstructs Ridges from recorded numbers; the reload must be exact."""
    from training import refit_ridge_stage2 as S2

    coef = [245.4067, 37.5425, -18.5303, 14.5807, 12.0704]
    info = S2.build_matched_ridge(coef, 37.6215, tmp_path / "w.pkl")

    import pickle
    reloaded = pickle.load(open(tmp_path / "w.pkl", "rb"))
    assert list(reloaded.coef_) == coef
    assert reloaded.intercept_ == 37.6215
    assert info["coef"] == coef
    assert len(info["sha256"]) == 64


def test_matched_ridge_exposes_coef_the_way_engine_reads_it(tmp_path):
    """engine.py uses weight_model.coef_ and indexes w[0]..w[4]."""
    from training import refit_ridge_stage2 as S2
    import pickle

    coef = [200.0, 40.0, -3.0, 14.0, 3.0]
    S2.build_matched_ridge(coef, 50.0, tmp_path / "w.pkl")
    w = pickle.load(open(tmp_path / "w.pkl", "rb")).coef_
    assert len(w) == 5
    assert float(w[0]) == 200.0 and float(w[4]) == 3.0


def test_stage2_writes_only_under_the_a1r_directory():
    from training import refit_ridge_stage2 as S2
    assert S2.OUT_DIR.name == "A1R"
    assert S2.OUT_DIR.resolve() != (S2.REPO_ROOT / "models").resolve()
    assert S2.RIDGE_DIR.is_relative_to(S2.OUT_DIR)
