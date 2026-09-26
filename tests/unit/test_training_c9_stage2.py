"""Tests for C9 Stage 2 — selection (2a) and matched-fusion engine staging (2b).

Stage 2 produces the numbers a fusion decision would rest on, so the parts that
would silently corrupt it are pinned here:

  * the FIT/TUNE split is game-level, deterministic, and disjoint
  * the Ridge is never fitted on TUNE, and no suite is read during selection
  * the candidate set is closed and the tie rule is applied as pre-registered
  * the divisor is staged by scaling ONLY the output layer, exactly
  * staged artifacts are never production artifacts
  * Stage 2b never writes to models/

No Stockfish, no training, no writes outside tmp_path.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from training import c9_stage2a as S2A
from training import c9_stage2b as S2B
from training import refit_ridge as RR

REPO_ROOT = Path(__file__).resolve().parents[2]
C9 = REPO_ROOT / "training" / "experiments" / "C9"
SELECTION = C9 / "stage2_selection.json"
RESULTS = C9 / "stage2_results.json"


def _need(path):
    if not path.is_file():
        pytest.skip(f"{path.name} not present; run the C9 Stage 2 modules")


@pytest.fixture(scope="module")
def selection():
    _need(SELECTION)
    return json.loads(SELECTION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results():
    _need(RESULTS)
    return json.loads(RESULTS.read_text(encoding="utf-8"))


# ===================================== frozen design parameters

def test_candidate_set_is_the_closed_stage1_sweep():
    assert S2A.DIVISORS == (200, 400, 600, 800, 1000, 1500, 2000)
    assert S2A.FIT_VARIANTS == ("A_include_checkmate", "B_exclude_checkmate")


def test_production_divisor_is_in_the_candidate_set():
    """"Keep production" must be able to win."""
    assert 200 in S2A.DIVISORS


def test_split_seed_is_distinct_from_every_other_seed_in_the_programme():
    from training import dataset as D
    assert S2A.TUNE_SPLIT_SEED == 9
    assert S2A.TUNE_SPLIT_SEED != D.DEFAULT_SPLIT_SEED        # 42
    assert S2A.TUNE_SPLIT_SEED != RR.INNER_SPLIT_SEED         # 1234


def test_tie_rule_constants_match_the_design():
    assert S2A.TIE_SPEARMAN == 0.005
    assert S2A.TUNE_FRACTION == 0.20


# ===================================== FIT / TUNE split

def _records(n_games=50, per_game=4):
    return [{"game_content_key": f"g{g:03d}", "fen": f"f{g}-{i}",
             "is_checkmate": False, "label": 0}
            for g in range(n_games) for i in range(per_game)]


def test_fit_tune_split_is_game_level_and_disjoint():
    fit, tune, fg, tg = S2A.fit_tune_split(_records())
    assert not (fg & tg)
    assert {r["game_content_key"] for r in fit} == fg
    assert {r["game_content_key"] for r in tune} == tg
    assert len(fit) + len(tune) == 200


def test_fit_tune_split_respects_the_fraction():
    _, _, fg, tg = S2A.fit_tune_split(_records(n_games=100))
    assert len(tg) == 20 and len(fg) == 80


def test_fit_tune_split_is_deterministic():
    a = S2A.fit_tune_split(_records())
    b = S2A.fit_tune_split(_records())
    assert a[2] == b[2] and a[3] == b[3]


def test_fit_tune_split_is_invariant_to_record_order():
    recs = _records()
    _, _, fg_a, tg_a = S2A.fit_tune_split(recs)
    _, _, fg_b, tg_b = S2A.fit_tune_split(list(reversed(recs)))
    assert fg_a == fg_b and tg_a == tg_b


def test_fit_tune_split_changes_with_the_seed():
    _, _, _, tg_9 = S2A.fit_tune_split(_records(), seed=9)
    _, _, _, tg_7 = S2A.fit_tune_split(_records(), seed=7)
    assert tg_9 != tg_7


def test_recorded_split_is_disjoint_and_game_level(selection):
    sp = selection["fit_tune_split"]
    assert sp["seed"] == 9
    assert sp["n_fit_games"] + sp["n_tune_games"] == 3_614
    assert sp["n_fit_positions"] + sp["n_tune_positions"] == 13_712
    assert sp["fit_games_sha256"] != sp["tune_games_sha256"]


# ===================================== TUNE benchmark selection

def test_tune_position_selection_skips_unrankable_boards():
    import chess
    mate = {"fen": "7k/5KQ1/8/8/8/8/8/8 b - - 0 1", "is_checkmate": True,
            "game_content_key": "g"}
    assert chess.Board(mate["fen"]).is_checkmate()
    assert S2A.select_tune_positions([mate], 100) == []


def test_tune_position_selection_is_deterministic_and_in_file_order():
    recs = [{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "is_checkmate": False, "game_content_key": f"g{i}"} for i in range(10)]
    a = S2A.select_tune_positions(recs, 60)
    b = S2A.select_tune_positions(recs, 60)
    assert [r["game_content_key"] for r in a] == [r["game_content_key"] for r in b]
    assert [r["game_content_key"] for r in a] == \
        [r["game_content_key"] for r in recs[:len(a)]]


def test_recorded_benchmark_meets_the_design_budget(selection):
    tb = selection["tune_benchmark"]
    assert tb["n_positions"] > 0
    assert tb["n_candidates"] >= 5_000          # design targeted ~6,000
    sf = tb["stockfish"]
    assert sf["depth"] == 8 and sf["threads"] == 1 and sf["hash_mb"] == 16
    assert sf["clear_hash_per_position"] is True
    assert "17.1" in sf["version"]


# ===================================== leakage

def test_selection_never_reads_a_suite():
    """No suite file may be opened by the selection module."""
    body = Path(S2A.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for banned in ("extended.json", "phase0_52.json", "evaluation/positions"):
        assert banned not in body, banned


def test_recorded_fit_population_is_suite_clean(selection):
    fp = selection["fit_population"]
    assert fp["overlap_with_suite_placements"] == 0
    assert fp["overlap_with_suite_exact_fens"] == 0
    assert fp["overlap_with_c8a_train_placements"] == 0
    assert fp["overlap_with_c8a_train_games"] == 0


def test_no_unseeded_randomness_in_selection():
    """A SEEDED permutation is the deterministic mechanism (same as the C7 game
    split); what must not appear is unseeded RNG, shuffling or sampling."""
    import ast
    body = Path(S2A.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for banned in ("shuffle", "np.random.seed", "random.choice", "random.sample",
                   ".sample(", "randint"):
        assert banned not in body, banned

    # Every default_rng call must pass an explicit seed argument.
    tree = ast.parse(Path(S2A.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "default_rng"]
    assert calls, "expected the seeded permutation to use default_rng"
    for call in calls:
        assert call.args or call.keywords, "default_rng called without a seed"


# ===================================== the grid and the tie rule

def test_grid_covers_every_divisor_and_variant(selection):
    expected = {f"{v}|d={d}" for v in S2A.FIT_VARIANTS for d in S2A.DIVISORS}
    assert set(selection["grid"]) == expected
    assert len(expected) == 14


def test_every_grid_cell_has_all_three_seeds(selection):
    for key, cell in selection["grid"].items():
        assert set(cell["per_seed"]) == {"seed_0", "seed_1", "seed_2"}, key
        for s in cell["per_seed"].values():
            assert len(s["coef"]) == 5


def test_candidate_set_is_recorded_as_closed(selection):
    cs = selection["candidate_set"]
    assert cs["closed"] is True
    assert cs["divisors"] == list(S2A.DIVISORS)


def test_selection_is_the_argmax_or_a_pre_registered_tie_break(selection):
    grid = selection["grid"]
    best = max(c["mean_spearman_across_seeds"] for c in grid.values())
    chosen = selection["selection"]
    assert best - chosen["mean_spearman"] <= S2A.TIE_SPEARMAN


def test_tie_break_selected_the_smallest_divisor_among_contenders(selection):
    chosen = selection["selection"]
    divisors = [int(k.split("d=")[1]) for k in chosen["contenders"]]
    assert chosen["divisor"] == min(divisors)


def test_tie_break_prefers_variant_a_when_divisors_match(selection):
    """Pre-registered order: smaller d, then variant A."""
    chosen = selection["selection"]
    same_d = [k for k in chosen["contenders"]
              if int(k.split("d=")[1]) == chosen["divisor"]]
    if len(same_d) > 1:
        assert chosen["fit_variant"].startswith("A")


def test_collapse_flag_is_consistent_with_the_chosen_divisor(selection):
    chosen = selection["selection"]
    assert chosen["collapses_to_variant_b"] == (chosen["divisor"] == 200)


# ===================================== Stage 2b staging

def test_matched_ridge_round_trips_its_coefficients(tmp_path):
    coef = [1272.683, 6.3911, 8.7604, -19.2777, -0.548]
    info = S2B.build_matched_ridge(coef, 5.6457, tmp_path / "r.pkl")
    assert info["coef"] == pytest.approx(coef)
    import pickle
    reloaded = pickle.load(open(tmp_path / "r.pkl", "rb"))
    assert reloaded.coef_ == pytest.approx(np.asarray(coef))
    assert reloaded.intercept_ == pytest.approx(5.6457)


def test_matched_ridge_exposes_coef_the_way_engine_reads_it(tmp_path):
    import pickle
    info = S2B.build_matched_ridge([1.0, 2.0, 3.0, 4.0, 5.0], 0.5, tmp_path / "r.pkl")
    r = pickle.load(open(tmp_path / "r.pkl", "rb"))
    w = r.coef_
    assert len(w) == 5 and float(w[0]) == 1.0
    assert info["sha256"]


def test_scaled_cnn_scales_only_the_output_layer(tmp_path):
    import keras
    src = REPO_ROOT / "training" / "experiments" / "C8a" / "seed_0" / "models" / "cnn_model.keras"
    if not src.is_file():
        pytest.skip("C8a seed 0 model not present")
    info = S2B.build_scaled_cnn(src, 1000, tmp_path / "c.keras")
    assert info["k"] == pytest.approx(0.2)
    assert info["earlier_layers_bit_identical"] is True
    assert info["final_layer_scaled_by_k"] is True

    a = keras.models.load_model(src, compile=False)
    b = keras.models.load_model(tmp_path / "c.keras", compile=False)
    for la, lb in list(zip(a.layers, b.layers))[:-1]:
        for wa, wb in zip(la.get_weights(), lb.get_weights()):
            assert np.array_equal(wa, wb)
    for wa, wb in zip(a.layers[-1].get_weights(), b.layers[-1].get_weights()):
        assert np.allclose(wb, wa * 0.2, rtol=0, atol=1e-6)


def test_scaled_cnn_reproduces_the_target_divisor_exactly(tmp_path):
    """tanh(scaled/200) must equal tanh(original/d) - the whole mechanism."""
    import keras
    from training import representation as R
    src = REPO_ROOT / "training" / "experiments" / "C8a" / "seed_0" / "models" / "cnn_model.keras"
    if not src.is_file():
        pytest.skip("C8a seed 0 model not present")
    S2B.build_scaled_cnn(src, 1000, tmp_path / "c.keras")
    a = keras.models.load_model(src, compile=False)
    b = keras.models.load_model(tmp_path / "c.keras", compile=False)
    fens = [json.loads(l)["fen"] for l in
            (REPO_ROOT / "training" / "artifacts" / "dataset_v2.test.jsonl")
            .read_text(encoding="utf-8").splitlines()[:64]]
    X = R.encode_many(fens)
    raw = np.asarray(a.predict(X, verbose=0)).reshape(-1)
    scaled = np.asarray(b.predict(X, verbose=0)).reshape(-1)
    assert np.max(np.abs(np.tanh(scaled / 200) - np.tanh(raw / 1000))) < 1e-5


def test_scaled_cnn_rejects_an_unexpected_output_layer(tmp_path):
    import keras
    from keras import layers, models
    wrong = models.Sequential([layers.Input(shape=(8, 8, 12)), layers.Flatten(),
                               layers.Dense(2)])
    wrong.compile(optimizer="adam", loss="huber")
    p = tmp_path / "wrong.keras"
    wrong.save(p)
    with pytest.raises(SystemExit, match="Dense\\(1"):
        S2B.build_scaled_cnn(p, 1000, tmp_path / "out.keras")


def test_production_divisor_constant_matches_engine():
    assert S2B.PRODUCTION_DIVISOR == 200.0
    src = (REPO_ROOT / "engine.py").read_text(encoding="utf-8")
    assert "np.tanh(cnn_score / 200)" in src


# ===================================== Stage 2b never touches production

def test_stage2b_never_writes_to_production():
    body = Path(S2B.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for banned in ("shutil.copy2(PRODUCTION", "open(PRODUCTION_MODELS / \"weight_model.pkl\", \"wb\")"):
        assert banned not in body, banned
    assert "PRODUCTION_MODELS" in body        # it is read, for verification only


def test_stage2b_uses_the_existing_staging_helper():
    body = Path(S2B.__file__).read_text(encoding="utf-8")
    assert "EA.stage_models_dir(cnn, Path(tmp), ridge_model=ridge)" in body


def test_stage2b_uses_the_unmodified_evaluator():
    body = Path(S2B.__file__).read_text(encoding="utf-8")
    assert "EA.run_suite(" in body
    assert 'representation="planes12"' in body


def test_stage2b_output_tree_is_the_gitignored_experiments_dir():
    assert S2B.OUT_DIR.name == "C9"
    assert S2B.OUT_DIR.parent.name == "experiments"
    for d in (S2B.RIDGE_DIR, S2B.CNN_DIR):
        assert d.is_relative_to(S2B.OUT_DIR)


def test_stage2b_never_overwrites_the_c8a_control():
    body = Path(S2B.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    assert "experiments/C8a" not in body.replace("\\", "/") or "source_cnn" in body


# ===================================== recorded Stage 2b results

def test_control_is_reused_not_rerun(results):
    assert results["control"]["rerun"] is False
    assert "C8a" in results["control"]["reused_from"]


def test_lookahead_caveat_is_recorded(results):
    caveat = results["lookahead_caveat"]
    assert "three places" in caveat
    assert "lookahead" in caveat
    assert "NOT a single-variable contrast" in caveat


def test_every_run_passed_its_staging_checks(results):
    assert results["per_run"], "no runs recorded"
    for run in results["per_run"]:
        assert all(run["staging_checks"].values()), run


def test_staged_ridge_was_never_the_production_vector(results):
    for run in results["per_run"]:
        assert run["staging_checks"]["staged_ridge_is_not_production"] is True


def test_every_run_covers_both_suites(results):
    for run in results["per_run"]:
        assert set(run["summary"]) == {"extended", "phase0_52"}
        assert run["summary"]["extended"]["n"] == 160
        assert run["summary"]["phase0_52"]["n"] == 52


def test_legality_never_regressed_below_the_control(results):
    for run in results["per_run"]:
        for suite in ("extended", "phase0_52"):
            assert run["summary"][suite]["legality"]["percent"] == 100.0


def test_cells_match_the_selection(results, selection):
    d_star = selection["selection"]["divisor"]
    assert "C9b" in results["cells"]
    assert results["cells"]["C9b"]["divisor"] == 200
    if d_star == 200:
        assert "C9c" not in results["cells"]
    else:
        assert results["cells"]["C9c"]["divisor"] == d_star


def test_c9c_artifacts_record_the_rescale(results):
    if "C9c" not in results["cells"]:
        pytest.skip("d* == 200; C collapsed onto B")
    for seed in ("seed_0", "seed_1", "seed_2"):
        cnn = results["cells"]["C9c"]["artifacts"][seed]["cnn"]
        assert cnn["rescaled"] is True
        assert cnn["earlier_layers_bit_identical"] is True
        assert cnn["final_layer_scaled_by_k"] is True
        assert cnn["k"] == pytest.approx(200.0 / results["cells"]["C9c"]["divisor"])
