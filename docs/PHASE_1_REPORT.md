# Phase 1 Report — Repository Integrity + Reproducibility

**Objective:** make the repository honest, reproducible and clean **without changing the
engine's decision-making behaviour.**

**Baseline reference:** commit `ccd24c417826bb20048ac19b2cb50202189cf870`, artifacts in
`baseline/`. Nothing under `baseline/` was modified.

**Outcome:** behaviour preserved exactly — 52/52 positions return identical moves,
identical top-3 orderings and identical scores. Nothing committed.

---

## 1. Changes implemented

### 1.1 Repository-relative model paths

- **Problem:** `engine.py:7` loaded `"models/cnn_model.keras"` relative to the current
  working directory. Importing from anywhere other than the repo root failed. This was
  hit for real during Phase 0 when running a script from `baseline/scripts/`.
- **Old behaviour:** `import engine` raised `ModuleNotFoundError` / failed to find models
  unless CWD was the repo root.
- **New behaviour:** new `config.py` anchors `MODELS_DIR` to `Path(__file__).resolve().parent`.
  Verified working with CWD set to `C:\Users\vsriv`.
- **Why Phase 1:** pure path resolution. No model, preprocessing or inference change.

### 1.2 Stockfish configuration

- **Problem:** an absolute Windows path (`C:\Users\vsriv\Downloads\...`) was hardcoded in
  `app.py:383` and `benchmark_stockfish.py:12`.
- **Old behaviour:** Stockfish-dependent code only worked on one machine.
- **New behaviour:** `config.find_stockfish()` resolves `STOCKFISH_PATH` → `PATH` →
  per-platform defaults, and returns `None` when unavailable. Optional gitignored `.env`
  support (~15 lines, no new dependency) so the existing local install keeps working.
  `STOCKFISH_DEPTH = 8` is **unchanged**.
- **Why Phase 1:** configuration only. Depth, algorithm and quality settings untouched.

### 1.3 Dependency specification corrected

- **Problem:** `requirements.txt` pinned `tensorflow==2.15.0`, which bundles Keras 2.x,
  while `models/cnn_model.keras` is a Keras 3 archive (`keras_version: 3.11.1`,
  layout `config.json` + `metadata.json` + `model.weights.h5`). A clean install produced
  an app that could not load its own model. Everything else was unpinned, `streamlit` was
  listed but unused, and `python-chess==1.11.2` is **not installable** (that alias stops
  at 1.999 — the distribution is `chess`).
- **New behaviour:** pinned to the empirically verified environment; notebook-only deps
  split into `requirements-notebook.txt`; `streamlit` dropped.
- **Verification:** `pip install --dry-run` resolves both files with exit 0, and reports
  only `gunicorn` as needing installation — confirming every other pin already matches the
  working environment.

### 1.4 Python version declaration aligned

- **Problem:** `runtime.txt` / `.python-version` declared `3.10.13`; the project actually
  ran on `3.12.10`.
- **New behaviour:** both declare `3.12.10`.

### 1.5 Unused model loading removed

- **Problem:** `engine.py` unpickled `rf_model.pkl` (77.9 MB), `mlp_model.pkl` and
  `scaler.pkl` at import. None is referenced anywhere in `engine.py` or `app.py`.
- **New behaviour:** only `cnn_model.keras` and `weight_model.pkl` are loaded. **The three
  files are retained on disk** (see §6).
- **Measured effect:** ~1.07 s and ~79 MB of resident memory. **Not** a fix for the ~19 s
  TensorFlow import.

### 1.6 Dead `TF_CPP_MIN_LOG_LEVEL` line fixed

- **Problem:** `app.py:10` set the variable *after* line 8 imported `engine`, which pulls
  in TensorFlow — so it had no effect.
- **New behaviour:** moved above the import, using `setdefault` so an explicit value wins.

### 1.7 Emoji `print` at import replaced

- **Problem:** `engine.py:21` printed `✅ All models loaded`, raising `UnicodeEncodeError`
  on a cp1252 Windows console.
- **New behaviour:** `logging.getLogger(__name__).info(...)`, ASCII only.

### 1.8 Invalid benchmark logic disabled — see §4

### 1.9 README corrected — see §5

---

## 2. Dependency / reproducibility result

**Supported Python:** 3.12 (verified 3.12.10). Declared consistently in `runtime.txt`
and `.python-version`.

| Package | Pin |
|---|---|
| tensorflow | 2.21.0 |
| keras | 3.13.2 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| numpy | 2.4.3 |
| chess | 1.11.2 |
| flask | 3.1.3 |
| flask-cors | 6.0.2 |
| gunicorn | 23.0.0 (deploy only, Linux/macOS) |

**Model compatibility:** the CNN loads under TF 2.21 / Keras 3.13.2. The Ridge model
unpickles with coefficients byte-identical to Phase 0.

### Remaining compatibility risks

1. **scikit-learn pickle skew persists.** Pickles were written by 1.7.1, loaded under
   1.8.0, emitting `InconsistentVersionWarning`. Not suppressed. Low risk for the Ridge
   model (a 5-element array plus a scalar, both verified identical), but real. Fixing it
   properly means re-serialising model files — out of scope.
2. **The TF 2.15 incompatibility was never empirically demonstrated**, only inferred from
   the Keras version and archive layout inside the model file. Installing TF 2.15 would
   have destroyed the verified environment.
3. **`gunicorn==23.0.0` is unverified locally** — it does not run on Windows.
4. **Cross-machine determinism unverified.** TensorFlow warns about oneDNN
   floating-point ordering at every import.
5. **107 MB of models still tracked in git** without LFS.

---

## 3. Stockfish configuration

**How it is configured:** `STOCKFISH_PATH` env var (or `.env`) → `stockfish` on `PATH` →
per-platform common locations. Depth fixed at 8, unchanged from before Phase 1.

**Required setup:** none for gameplay. For `/benchmark`, install Stockfish
(`apt-get install stockfish`, `brew install stockfish`, or download on Windows) and
optionally set `STOCKFISH_PATH`.

**Version used for the baseline:** Stockfish 17.1.

**When unavailable:** `find_stockfish()` returns `None`; `/benchmark` returns HTTP 200
with `{"stockfish": {"available": false, "note": "..."}}`. No other endpoint is affected.
The previous `except (FileNotFoundError, Exception)` — which reported every failure,
including genuine bugs, as "not installed" — is now `except Exception` with the actual
error text surfaced.

---

## 4. Invalid benchmark cleanup

Originals preserved verbatim in `archive/phase1_invalid_benchmarks/` **before** any edit.
This mattered: `benchmark_stockfish.py` and `generate_visualizations.py` were **untracked**,
so editing them in place would have destroyed the evidence permanently. The four PNGs were
tracked at `ccd24c4` and remain recoverable from git history as well.

| # | Invalid metric | Location | Action |
|---|---|---|---|
| 1 | Agreement rate from `np.random.rand() < 0.7`, discarding the real moves computed two lines above | `benchmark_stockfish.py:57` | **Script disabled** — exits non-zero with an explanation |
| 2 | Hardcoded "Average time 9.2ms / 15.1ms", "Min 7.10ms", "Max 12.30ms", "Speedup 0.60x", printed while the measured lists went unused | `benchmark_stockfish.py:113-126` | **Script disabled** |
| 3 | `eval_diff_cp` — subtracted a side-to-move-relative Stockfish centipawn score from the engine's non-centipawn score | `app.py:411` | **Field removed** from the API response |
| 4 | `speedup` = `stockfish_time / hybrid_time` — reads as a speed-up but is < 1 because the engine is slower; rendered in green with a `×` | `app.py:409` + `index.html` | **Replaced** by `engine_time_ratio_vs_stockfish` with an explicit "greater than 1 means slower" note; UI label and styling updated |
| 5 | `times = [40, 250, 2500]`, "6.2x faster", "62x faster", `sizes = [11.2, 50]`, `values = [0.506, 0.55, 180]` — all hardcoded | `generate_visualizations.py` | **Script disabled** |
| 6 | Synthetic "training curves" from `25000*np.exp(-epochs/15)+15000+np.random.normal(...)` | `generate_visualizations.py:135-136,147` | **Script disabled** |
| 7 | Four generated charts displayed in the web UI's Model Metrics panel | `static/*.png`, `index.html:1597-1600` | **PNGs deleted; UI panel replaced** with an explanation of why they were removed and a pointer to `baseline/BASELINE.md` |

`agreement` in `/benchmark` was **kept** — unlike the script's version, it is computed
honestly as `hybrid_move == stockfish_move`.

No replacement evaluation harness was built, and no new metric was introduced.

---

## 5. README corrections

| Claim | Was | Now | Evidence |
|---|---|---|---|
| Dataset size | "~50,000 positions" (×3 places) | **10,000 positions**, all at ply 20 | notebook cell 14 |
| Train/test split | "held-out test set of 50 positions" | **20% holdout, ~2,000 positions** | `train_test_split(test_size=0.2)` |
| Pearson correlation | 0.506 | **0.708**, framed as label fit not playing strength | notebook cell 16 |
| Starting-position agreement | "✅ CNN agrees with Stockfish (`e2e4`)" | **Removed.** The full engine plays `d2d4` at the start; Stockfish plays `e2e4`. The old claim described a raw-CNN argmax, not the engine | `baseline/engine_results.json` OP01 |
| Speed | "6.2x / 62x faster than Stockfish" | **~121× slower**, with measured mean/median/p95 for both | `baseline/comparison.json` |
| Model performance table | Mixed real and fabricated figures | Rebuilt as **Measured Performance** with method, suite, and a random-move chance baseline | `baseline/BASELINE.md` |
| Top-1 agreement | (absent; chart claimed 0.55) | **19.2% on the 52-position suite**, vs 5.9% chance — explicitly *not* called "accuracy" | `baseline/comparison.json` |
| MAE 180 cp | Plotted in a chart | **Removed** — no MAE is computed anywhere in the project | — |
| Model size | "11.2 MB" / "~9.1 MB" | Replaced by real file sizes (27.1 MB on disk, 2,360,129 params) | `/model_info` |
| API reference | 8 of 10 endpoints | All 10, plus the shared-state and 415 quirks documented | `baseline/api_results.json` |
| Python version | "3.10 or higher" | **3.12**, matching the pins | `baseline/environment.json` |
| Model files | "all five required" | Two required, three marked training-only | source inspection |

Also added: a note that the Ridge layer weights the CNN term at **330.9** against **32.4**
for a pawn of material, so the hand-crafted heuristics contribute well under 1% of the
score — i.e. the explanations are post-hoc rationalisation of a neural decision. And an
explicit **"What is not yet measured"** block: no centipawn-loss metric, no MAE, no Elo,
no playing-strength result.

`/model_info` was corrected in the same way (0.506 → 0.708, ~50,000 → 10,000, plus
`dataset_note`, `test_split`, `correlation_note` and `source` fields).

---

## 6. Artifact cleanup

**Removed from runtime loading:** `rf_model.pkl`, `mlp_model.pkl`, `scaler.pkl`.

Safety checks performed before the change:

| Check | Result |
|---|---|
| Referenced anywhere in `engine.py` after load? | No |
| Referenced anywhere in `app.py`? | No (`from engine import` names only `engine_move`, `position_metrics`, `explain_move`, `cnn_model`, `board_to_planes`) |
| Used by any endpoint? | No |
| Needed by the training workflow? | The notebook **creates** them; it does not read them from disk |
| Documented functionality depends on them? | Yes — the README documents the RF and MLP results |

**The files were NOT deleted.** They are genuine outputs of the ML work, the README
documents their results (RF R² = 0.5146, MLP accuracy = 0.7228), and deleting them would
remove evidence of the project's ML scope. They are now labelled as training artifacts in
both the README and `docs/REPRODUCIBILITY.md`. Deleting the 77.9 MB RF pickle from git
history is deferred — it is a git-size decision, best taken alongside CI work.

**Deleted:** the four fabricated `static/*.png` charts (copies preserved in `archive/`,
plus recoverable from git history).

> **Explicitly not claimed:** this did not fix the slow startup. Import is dominated by
> TensorFlow (~19 s); removing the pickles saved ~1 s. This corrects an assumption stated
> in the pre-Phase-0 audit.

---

## 7. Behaviour preservation

Method: re-ran `baseline/scripts/measure_engine.py` over the same
`baseline/fens.json`, then diffed against the frozen Phase 0 results with
`verification/compare_phase0_phase1.py`. Raw output: `verification/behaviour_diff.json`.

| Metric | Phase 0 | Phase 1 |
|---|---:|---:|
| Legal moves | 52 / 52 (100%) | **52 / 52 (100%)** |
| Selected moves unchanged | — | **52 / 52** |
| Top-3 UCI + scores unchanged | — | **52 / 52** |
| Top score unchanged | — | **52 / 52** |
| Determinism (2 passes) | True | **True** |
| Ridge coefficients | `[330.9005, 32.3899, 0.7919, 5.1654, 0.0188]` | **identical** |
| Ridge intercept | 15.0772 | **identical** |
| API smoke tests (17 requests) | all as expected | **all status codes identical** |
| Model loading | OK | **OK** |

**Differences in selected moves: none.** `behaviour_preserved: true`.

Timing differed as expected from wall-clock noise — mean 617.0 ms → 660.8 ms, median
638.0 ms → 642.2 ms. Timing is not behaviour; no move or score changed.

`import engine` measured 16.50 s → 12.86 s in these two runs. This is **within run-to-run
variance** (Phase 0 fresh-subprocess runs spanned 17.3–24.3 s) and should not be read as a
speed-up caused by dropping the unused pickles, which accounted for ~1.07 s.

### Unexpected behaviour changes

**None in the engine.** Two deliberate, documented API changes:

- `/benchmark` no longer returns `eval_diff_cp` or `speedup`; it returns
  `engine_time_ratio_vs_stockfish`, `ratio_note` and `eval_comparison`. The frontend was
  updated to match. This was the point of Objective 4.
- `/model_info` returns corrected training figures plus four new descriptive fields.

One incidental improvement: `/benchmark` now works on any machine with Stockfish
installed, rather than only on the machine matching the hardcoded path.

---

## 8. Remaining issues

### Phase 2 (testing)
- No pytest suite, no `pyproject.toml`/`conftest.py`, pytest not installed.
- Silent-failure handlers would hide defects from assertions: bare `except:`
  (`engine.py`), `except Exception: pass` (`app.py` ×2).
- Module-level globals make API tests order-dependent and non-parallelisable.
- Import cost (13–24 s) is paid before the first assertion.

### Phase 3 (CI)
- No `.github/`.
- 107 MB of models in git without LFS; slow clones.
- `PYTHONIOENCODING=utf-8` still needed for some scripts on Windows consoles.
- Stockfish unpinned (`Threads`/`Hash` at defaults) — reference not reproducible.
- `gunicorn` untestable on the Windows dev machine.

### Phase 4 (correctness — deferred because each changes engine output)
- **C1** Heuristic bonuses added with fixed positive sign while the sort direction flips
  by side; for Black they push good moves down its own preference list. Always active —
  the engine only ever plays Black.
- **C2** 1-ply lookahead takes `max` over opponent replies, selecting the reply best for
  the mover rather than the opponent's best.
- **C3** Ridge `intercept_` (15.0772) never applied at inference.
- **C4** `center` in each candidate dict is the pre-move value while `material`/`space`/
  `mobility` are post-move.
- **C5** `opening_center_bonus` matches only White's UCI strings.
- **C6** *Hypothesis:* `board_to_planes` encodes no side-to-move / castling / en-passant
  plane, yet post-move positions are evaluated with the opponent to move. Unmeasured.

### Also deferred (safe, but not Phase 1 scope)
- `POST /move` with a non-JSON body returns 415 HTML instead of JSON.
- `/engine_move` runs the engine twice per request (~2× latency).
- `/forfeit` hardcodes result `"0-1"`.
- No evaluation harness: no centipawn-loss/regret metric, no MultiPV in app code.
- `rf_model.pkl` still 77.9 MB in git history.
- `ogapp.py`, `check_strucutre.py`, `chess_model.ipynb` remain gitignored dev leftovers.

---

## 9. Files changed

### Modified (tracked)
```
.gitignore              +7      .env, .DS_Store, *.pyc
.python-version          1      3.10.13 -> 3.12.10
runtime.txt              1      python-3.10.13 -> python-3.12.10
requirements.txt        60      repinned, split, streamlit dropped, chess name fixed
engine.py               26      repo-relative paths, unused loads removed, logging
app.py                  73      stockfish config, TF log fix, benchmark + model_info
templates/index.html    37      benchmark panel fields, metrics panel replaced
README.md              175      all quantitative claims corrected
```

### Added (untracked)
```
config.py                            paths + Stockfish resolution + .env loader
requirements-notebook.txt            notebook/training deps
.env.example                         config template
docs/REPRODUCIBILITY.md              setup and limitations
docs/PHASE_1_REPORT.md               this file
verification/compare_phase0_phase1.py
verification/phase1_engine_results.json
verification/phase1_api_results.json
verification/behaviour_diff.json
verification/phase1_engine_run.log
verification/server_start.log
archive/phase1_invalid_benchmarks/   originals + README explaining why
.env                                 gitignored, local Stockfish path
```

### Deleted (tracked)
```
static/accuracy_metrics.png
static/architecture_breakdown.png
static/performance_comparison.png
static/training_progress.png
```

### Rewritten in place (were untracked, originals archived first)
```
benchmark_stockfish.py       -> disabled stub
generate_visualizations.py   -> disabled stub
```

### Pre-existing, unrelated — untouched
```
baseline/                    Phase 0 artifacts, read-only this phase
games.csv, ogapp.py, check_strucutre.py,
chess_model.ipynb, lichess_data_analysis.ipynb, venv/    (all gitignored)
```

---

## 10. Verification commands

```bash
# Models load; Ridge coefficients match baseline; Stockfish resolves
python -c "import engine, config, json; \
  print([round(float(x),4) for x in engine.weight_model.coef_]); \
  print(json.dumps(config.stockfish_status(), indent=2))"

# CWD independence (run from a different directory)
cd /c/Users/vsriv && PYTHONPATH='/path/to/chess-bot' python -c \
  "import engine, chess; print(engine.engine_move(chess.Board())[0])"

# Behaviour preservation over the frozen 52-FEN suite
python baseline/scripts/measure_engine.py \
    baseline/fens.json verification/phase1_engine_results.json
python verification/compare_phase0_phase1.py \
    baseline/engine_results.json \
    verification/phase1_engine_results.json \
    verification/behaviour_diff.json

# API smoke test (17 requests)
python baseline/scripts/measure_api.py verification/phase1_api_results.json

# Real HTTP server end-to-end
PORT=5055 python app.py &
curl -X POST http://127.0.0.1:5055/reset
curl -X POST http://127.0.0.1:5055/move -H "Content-Type: application/json" -d '{"uci":"e2e4"}'
curl -X POST http://127.0.0.1:5055/engine_move

# Dependency resolution without installing
python -m pip install --dry-run --report - -r requirements.txt
python -m pip install --dry-run --report - -r requirements-notebook.txt

# Disabled scripts exit non-zero
python benchmark_stockfish.py ; echo "exit=$?"
python generate_visualizations.py ; echo "exit=$?"
```
