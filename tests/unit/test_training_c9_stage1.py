"""Tests for C9 Stage 1 — matched-Ridge and divisor diagnostic.

Stage 1 produces the numbers a Stage 2 go/no-go would rest on, so the parts that
would silently corrupt that decision are pinned here:

  * the A1R default path through `refit_ridge` is untouched by the extension
  * the fit population is dataset_v2's test split and is verified against its
    manifest, with the leakage guards actually firing
  * `tanh_scale` is a parameter whose default is still the production 200.0
  * the gate is computed from the COEFFICIENT refit only - never from the
    divisor sweep
  * Stage 1 runs no engine, no Stockfish, and writes nothing to production

No training, no Stockfish, no writes outside tmp_path.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from training import c9_stage1 as C9
from training import dataset as D
from training import refit_ridge as RR

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "training" / "experiments" / "C9" / "stage1_results.json"


def _needs_results():
    if not RESULTS.is_file():
        pytest.skip("C9 Stage 1 not run; python -m training.c9_stage1")


@pytest.fixture(scope="module")
def results():
    _needs_results()
    return json.loads(RESULTS.read_text(encoding="utf-8"))


# ===================================== the A1R extension stays additive

def test_a1r_defaults_are_untouched():
    assert RR.ARMS == {"A0": D.LABEL_POLICY_LEGACY,
                       "A1": D.LABEL_POLICY_CORRECTED_MATE}
    assert RR.OUT_DIR.name == "A1R"
    assert RR.STAGE == "A1R-stage1"
    assert RR.RIDGE_ALPHA == 1.0


def test_load_fit_population_default_is_the_a1r_dataset_v1_path():
    import inspect
    src = inspect.getsource(RR.load_fit_population)
    assert "if prefix is None:" in src
    assert "return load_test_split()" in src


def test_c8a_is_registered_with_a2s_label_policy():
    assert RR.ARM_POLICIES["C8a"] == D.LABEL_POLICY_CORRECTED_MATE_WHITE
    assert RR.ARM_POLICIES["C8a"] == RR.ARM_POLICIES["A2"]
    assert "C8a" not in RR.ARMS          # must not widen the A1R default


def test_tanh_scale_defaults_to_production_everywhere():
    import inspect
    assert RR.TANH_SCALE == 200.0
    for fn in (RR.board_features, RR.ranking_spread):
        assert inspect.signature(fn).parameters["tanh_scale"].default == 200.0


def test_board_features_honour_a_custom_divisor():
    import chess
    a = RR.board_features(chess.Board(), 400.0)
    b = RR.board_features(chess.Board(), 400.0, tanh_scale=2000.0)
    assert a[0] == pytest.approx(np.tanh(400.0 / 200.0))
    assert b[0] == pytest.approx(np.tanh(400.0 / 2000.0))
    assert a[1:] == b[1:]            # the four board terms are divisor-independent


def test_ranking_spread_honours_a_custom_divisor():
    cnn = {"p": np.array([100.0, 800.0, -400.0])}
    feats = {"p": np.zeros((3, 4))}
    coef = [1.0, 0.0, 0.0, 0.0, 0.0]
    tight = RR.ranking_spread(coef, cnn, feats, tanh_scale=200.0)
    loose = RR.ranking_spread(coef, cnn, feats, tanh_scale=2000.0)
    # A larger divisor shrinks the term: tanh is compressive toward linearity.
    assert loose["median_term_sd"]["cnn_norm"] < tight["median_term_sd"]["cnn_norm"]


# ===================================== fit population and its guards

def test_fit_population_prefix_points_at_dataset_v2():
    assert C9.DATASET_PREFIX.name == "dataset_v2"


def test_loader_rejects_a_missing_dataset(tmp_path):
    with pytest.raises(SystemExit, match="missing"):
        RR.load_fit_population(tmp_path / "nope")


def test_loader_rejects_a_file_that_does_not_match_its_manifest(tmp_path):
    manifest = json.loads(
        (REPO_ROOT / "training" / "artifacts" / "dataset_v2.manifest.json")
        .read_text(encoding="utf-8"))
    prefix = tmp_path / "dataset_v2"
    Path(f"{prefix}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    Path(f"{prefix}.test.jsonl").write_text('{"fen": "x"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match its manifest"):
        RR.load_fit_population(prefix)


def test_leakage_guard_raises_on_a_contaminated_population():
    """The guard must actually fire, not just report."""
    suite = json.loads((REPO_ROOT / "evaluation" / "positions" / "extended.json")
                       .read_text(encoding="utf-8"))["positions"][0]
    poisoned = [{"fen": suite["fen"], "placement": suite["fen"].split(" ")[0],
                 "game_content_key": "x", "is_checkmate": False, "label": 0}]
    with pytest.raises(SystemExit, match="contaminated"):
        C9.leakage_checks(poisoned)


def test_recorded_fit_population_is_clean(results):
    fp = results["fit_population"]
    assert fp["records"] == 13_712
    assert fp["overlap_with_suite_placements"] == 0
    assert fp["overlap_with_suite_exact_fens"] == 0
    assert fp["overlap_with_c8a_train_placements"] == 0
    assert fp["overlap_with_c8a_train_games"] == 0
    assert fp["all_clean"] is True
    assert fp["dataset_name"] == "dataset_v2"


def test_model_verification_pins_the_audited_hashes(results):
    for seed in (0, 1, 2):
        block = results["c8a_models"][f"seed_{seed}"]
        assert block["model_weights_sha256"] == C9.EXPECTED_C8A_WEIGHTS[seed]
        assert block["matches_audit"] is True
        assert block["model_parameters"] == 2_360_129
        assert block["trained_on"] == "dataset_v2"


def test_model_verification_rejects_a_wrong_hash(monkeypatch):
    monkeypatch.setitem(C9.EXPECTED_C8A_WEIGHTS, 0, "0" * 64)
    with pytest.raises(SystemExit, match="does not match the audited"):
        C9.verify_models()


# ===================================== both Ridge variants are reported

def test_both_ridge_variants_are_present_for_every_seed(results):
    for seed in (0, 1, 2):
        variants = results["per_seed"][f"seed_{seed}"]["matched_ridge"]
        assert set(variants) == {"all_positions", "excluding_checkmate"}


def test_checkmate_variant_excludes_exactly_the_mated_boards(results):
    n_mate = results["fit_population"]["checkmate_records"]
    assert n_mate == 1_129
    for seed in (0, 1, 2):
        v = results["per_seed"][f"seed_{seed}"]["matched_ridge"]
        assert v["all_positions"]["n_fit_positions"] == 13_712
        assert v["excluding_checkmate"]["n_fit_positions"] == 13_712 - n_mate


def test_every_refit_has_five_coefficients_and_an_intercept(results):
    for seed in (0, 1, 2):
        for variant in ("all_positions", "excluding_checkmate"):
            b = results["per_seed"][f"seed_{seed}"]["matched_ridge"][variant]
            assert len(b["coef"]) == 5
            assert isinstance(b["intercept"], float)
            assert len(b["ratios_normalised_to_cnn"]) == 5


def test_ranking_is_reported_on_both_populations(results):
    for seed in (0, 1, 2):
        for variant in ("all_positions", "excluding_checkmate"):
            rk = results["per_seed"][f"seed_{seed}"]["matched_ridge"][variant]["ranking"]
            assert set(rk) == {"extended", "heldout"}
            for pop in rk.values():
                assert "production" in pop and "refit" in pop
                assert pop["top_move_agreement"]["n_positions"] > 0


# ===================================== divisor sweep

def test_sweep_covers_the_required_divisors(results):
    assert results["divisors"] == [200, 400, 600, 800, 1000, 1500, 2000]
    for seed in (0, 1, 2):
        sweep = results["per_seed"][f"seed_{seed}"]["divisor_sweep"]["by_divisor"]
        assert set(sweep) == {str(d) for d in results["divisors"]}


def test_correlation_rises_monotonically_with_the_divisor(results):
    for seed in (0, 1, 2):
        sweep = results["per_seed"][f"seed_{seed}"]["divisor_sweep"]["by_divisor"]
        corrs = [sweep[str(d)]["corr_with_label"] for d in results["divisors"]]
        assert corrs == sorted(corrs)


def test_saturation_falls_as_the_divisor_rises(results):
    for seed in (0, 1, 2):
        sweep = results["per_seed"][f"seed_{seed}"]["divisor_sweep"]["by_divisor"]
        sat = [sweep[str(d)]["saturated_fraction"] for d in results["divisors"]]
        assert sat == sorted(sat, reverse=True)


def test_production_divisor_is_the_worst_of_the_swept_values(results):
    """The audit's claim, reproduced rather than assumed."""
    for seed in (0, 1, 2):
        sweep = results["per_seed"][f"seed_{seed}"]["divisor_sweep"]["by_divisor"]
        at200 = sweep["200"]["corr_with_label"]
        assert all(sweep[str(d)]["corr_with_label"] > at200
                   for d in results["divisors"] if d != 200)


def test_raw_correlation_is_recorded_and_is_the_ceiling(results):
    for seed in (0, 1, 2):
        sw = results["per_seed"][f"seed_{seed}"]["divisor_sweep"]
        raw = sw["raw_cnn"]["corr_with_label"]
        assert raw > sw["by_divisor"]["200"]["corr_with_label"]
        assert raw == pytest.approx(sw["by_divisor"]["2000"]["corr_with_label"], abs=0.01)


# ===================================== the gate

def test_gate_uses_the_a0_reference_band(results):
    assert results["a0_reference_band_cnn_share_pct"] == [82.85, 84.93]
    assert results["gate"]["reference_band_cnn_share_pct"] == [82.85, 84.93]


def test_gate_is_computed_on_the_extended_population(results):
    """The A0 band was measured there; comparing on another population would be
    meaningless."""
    assert "extended" in results["gate"]["population_used"]


def test_gate_is_evaluated_on_the_coefficient_refit_only():
    """A large divisor effect must never feed the gate."""
    import inspect
    src = inspect.getsource(C9.main)
    gate_block = src.split("GATE (coefficient refit only")[1]
    assert "divisor_sweep" not in gate_block
    assert "matched_ridge" in gate_block


def test_gate_result_is_consistent_with_the_recorded_shares(results):
    lo, hi = results["gate"]["reference_band_cnn_share_pct"]
    outside = []
    for seed in (0, 1, 2):
        g = results["gate"]["per_seed"][f"seed_{seed}"]
        share = g["refit_cnn_share_pct"]
        assert g["outside_a0_band"] == (not (lo <= share <= hi))
        outside.append(g["outside_a0_band"])
    assert results["gate"]["tripped"] == all(outside)


def test_stage1_never_runs_the_engine_or_stockfish():
    """Checked against the parsed module, not raw text: the banner string
    legitimately contains the word "Stockfish" while promising not to use it."""
    import ast
    tree = ast.parse(Path(C9.__file__).read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    for banned in ("subprocess", "stockfish", "evaluation.evaluate",
                   "training.evaluate_arm", "training.evaluate_planes_runner"):
        assert not any(m == banned or m.endswith(f".{banned}") for m in imported), banned

    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for banned in ("rerank_moves", "cnn_evaluate", "hybrid_score", "run",
                   "Popen", "check_output"):
        assert banned not in called, banned


def test_stage1_writes_nothing_to_production():
    body = Path(C9.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for banned in ("models/", "weight_model.pkl", "CHESS_BOT_MODELS_DIR"):
        assert banned not in body, banned


def test_output_directory_is_the_gitignored_experiments_tree():
    assert C9.OUT_DIR.name == "C9"
    assert C9.OUT_DIR.parent.name == "experiments"
    assert C9.OUT_DIR.resolve() != (REPO_ROOT / "models").resolve()


def test_stage1_declares_itself_a_diagnostic(results):
    assert results["stage"] == "C9-stage1"
    assert "No engine evaluation" in results["note"]
    assert "not evidence" in results["note"] or "NOT" in results["note"]


def test_production_ridge_is_recorded_read_only(results):
    pr = results["production_ridge"]
    assert pr["coef"] == pytest.approx(
        [330.900549, 32.389872, 0.791884, 5.165416, 0.018828], abs=1e-6)
    assert pr["intercept_applied_by_engine"] is False
    assert pr["tanh_divisor"] == 200.0


# ===================================== determinism

def test_no_random_sampling_is_introduced():
    body = Path(C9.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for banned in ("random.", "np.random", "shuffle", "default_rng"):
        assert banned not in body, banned


def test_heldout_ranking_population_is_taken_in_file_order():
    assert C9.HELDOUT_RANKING_POSITIONS == 80
    body = Path(C9.__file__).read_text(encoding="utf-8")
    assert "[:HELDOUT_RANKING_POSITIONS]" in body
