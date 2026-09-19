# Phase 0 — Reproducible Baseline

**Status:** measurement only. No application code, model, dependency, README, test or
CI file was created, modified or deleted. Nothing was committed.

**Repository state at measurement time**

| | |
|---|---|
| Branch | `main` |
| Commit | `ccd24c417826bb20048ac19b2cb50202189cf870` |
| Commit date | 2026-04-16 10:12:33 +0530 |
| Commit subject | `Comparision, Metrics, Architecture` |
| Working tree | clean for tracked files; two pre-existing untracked files (`benchmark_stockfish.py`, `generate_visualizations.py`) left untouched |
| Measured on | Windows 11 (10.0.26200), AMD64, Intel 13th-gen class, 22 logical CPUs |
| Date of run | 2026-09-19 |

---

## 1. Executive summary

The engine works and is well-behaved in the two ways that matter most for building a
test suite on top of it: it returned a **legal move in 52 of 52 positions (100%)**, and
it is **bit-for-bit deterministic** across repeated runs in the same process. Those two
properties are the foundation every later phase depends on, and they hold today.

Measured against Stockfish 17.1 at depth 8 on the same 52 positions:

- **Top-1 move agreement: 19.2%** (10/52), against a uniform-random chance level of
  **5.9%**. The engine is roughly **3.3× better than chance** — a real signal, not noise,
  and considerably more modest than the repository currently advertises.
- **Top-3 containment: 36.5%**, against a chance level of 17.7%.
- **Latency: the engine is ~121× slower than Stockfish depth 8** on identical positions
  (32,083 ms total vs 264 ms total). The repository's committed charts claim the engine
  is 6.2× and 62× *faster*. That claim is contradicted by measurement, and by the app's
  own `/benchmark` endpoint.

Three existing "metrics" in the repository are **not measurements** and were excluded
from this baseline rather than recorded: a randomly generated agreement rate, a set of
hardcoded timing statistics, and a fully synthetic chart-generation script. Details in
§10 (Evaluation) and §9.

One assumption from the earlier audit was **corrected by measurement**: the 82 MB unused
`rf_model.pkl` is *not* a major share of import cost (1.04 s). Import time is dominated
by **TensorFlow itself at 19.4 s**. Removing the dead pickle is still correct, but it is
a ~1 s win, not a ~20 s win. This is recorded so Phase 1 prioritises the right thing.

---

## 2. Environment

Full detail in [`environment.json`](environment.json).

### Interpreter and platform

| Item | Value |
|---|---|
| Python (actually used) | **3.12.10** (CPython, `venv/Scripts/python.exe`) |
| Python (declared in `runtime.txt`) | `python-3.10.13` |
| Python (declared in `.python-version`) | `3.10.13` |
| OS | Windows 11, 10.0.26200, AMD64 |
| Logical CPUs | 22 |

### Installed package versions

| Package | Installed | Declared in `requirements.txt` |
|---|---|---|
| tensorflow | **2.21.0** | `tensorflow==2.15.0` |
| keras | **3.13.2** | (implied by TF) |
| scikit-learn | **1.8.0** | unpinned |
| python-chess (`chess`) | 1.11.2 | unpinned |
| flask | 3.1.3 | unpinned |
| flask-cors | 6.0.2 | unpinned |
| numpy | 2.4.3 | unpinned |
| scipy | 1.17.1 | unpinned |
| pandas | 2.3.3 | unpinned |
| stockfish (PyPI wrapper) | 4.0.8 | unpinned |
| matplotlib / seaborn | 3.10.8 / 0.13.2 | unpinned |
| streamlit / ipython / tqdm | 1.55.0 / 9.11.0 / 4.67.3 | unpinned |
| h5py / protobuf | 3.14.0 / 6.33.6 | not listed |
| **gunicorn** | **not installed** | listed; `Procfile` runs `gunicorn app:app` |
| **pytest** | **not installed** | not listed |

### Can the repository be reproduced from a clean environment?

**No — not from the declared configuration.** Six concrete conflicts, all recorded in
`environment.json` under `reproducibility_conflicts`:

| ID | Conflict |
|---|---|
| `PY-VERSION-MISMATCH` | `runtime.txt` / `.python-version` declare 3.10.13; the working interpreter is 3.12.10 |
| `TF-PIN-CANNOT-LOAD-MODEL` | `requirements.txt` pins `tensorflow==2.15.0` (ships Keras 2.x). `models/cnn_model.keras` records `keras_version: 3.11.1` inside its own `metadata.json`. A Keras 2 runtime is not expected to load a Keras 3 archive. **Hypothesis, not empirically verified** — installing TF 2.15 was out of scope for Phase 0 |
| `SKLEARN-PICKLE-SKEW` | Pickles were written by scikit-learn 1.7.1; 1.8.0 is installed. Loading emits `InconsistentVersionWarning` ("might lead to breaking code or invalid results") |
| `GUNICORN-MISSING` | The documented production start command cannot run in this venv |
| `STOCKFISH-HARDCODED-PATH` | Absolute Windows path literal in `app.py:383` and `benchmark_stockfish.py:12`. Present on this machine; not on `PATH` |
| `CWD-RELATIVE-MODEL-PATHS` | `engine.py:7` loads `"models/cnn_model.keras"` relative to the current working directory. **Observed during this phase:** running a script from `baseline/scripts/` failed with `ModuleNotFoundError: No module named 'engine'` until the repo root was added to `sys.path` and set as CWD |

---

## 3. Repository and model inventory

### Model artifacts (exact bytes, SHA-256 in `environment.json`)

| File | Bytes | MB | Used at runtime? |
|---|---:|---:|---|
| `models/rf_model.pkl` | 81,720,956 | 77.9 | **No** — loaded at `engine.py:10`, never referenced again |
| `models/cnn_model.keras` | 28,381,519 | 27.1 | Yes |
| `models/mlp_model.pkl` | 1,061,773 | 1.01 | **No** — loaded at `engine.py:13`, never referenced again |
| `models/scaler.pkl` | 862 | 0.00 | **No** — loaded at `engine.py:16`, never referenced again |
| `models/weight_model.pkl` | 480 | 0.00 | Yes (Ridge coefficients) |
| **Total tracked in git** | **112,165,590** | **107.0** | — |

`cnn_model.keras` internal metadata: `{"keras_version": "3.11.1", "date_saved": "2026-03-11@09:16:31"}`.

### Source inventory

| File | Bytes | Lines | Tracked |
|---|---:|---:|---|
| `app.py` | 17,542 | 479 | yes |
| `engine.py` | 10,590 | 268 | yes |
| `templates/index.html` | 68,990 | 1,664 | yes |
| `README.md` | 20,688 | — | yes |
| `chess_model_FINAL.ipynb` | 399,502 | 47 cells | yes |
| `benchmark_stockfish.py` | 5,076 | 137 | **untracked** |
| `generate_visualizations.py` | 7,302 | 175 | **untracked** |

**Tests: 0 files. CI: 0 files.** No `tests/`, `conftest.py`, `pytest.ini`,
`pyproject.toml`, `setup.py`, or `.github/`.

### Stockfish

| Item | Value |
|---|---|
| Binary | `C:\Users\vsriv\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe` |
| Version | **Stockfish 17.1** |
| Size | 79,835,136 bytes (76.1 MB) |
| On `PATH` | **No** |
| Present on this machine | Yes |

---

## 4. Runtime measurements

### 4.1 Import / startup — fresh subprocess per run

From [`startup_results.json`](startup_results.json). Each run is a **new interpreter**;
`engine.py` executes model loading at module scope, so this cost is paid on every
process start.

| Module | Run 1 | Run 2 | Run 3 | Min | Max |
|---|---:|---:|---:|---:|---:|
| `import engine` | 17.29 s | 17.91 s | 24.34 s | **17.29 s** | **24.34 s** |
| `import app` | 23.45 s | 20.85 s | — | **20.85 s** | **23.45 s** |

**A genuinely cold first import in this session measured 76.3 s** (observed during the
preceding audit, before OS file caches were warm). The table above reflects a **warm
filesystem cache**. No post-reboot cold-boot measurement was taken. Both numbers are
reported because the gap between them is large and CI will see the cold path.

**Import cost attribution** (measured separately, warm cache):

| Component | Seconds | Share |
|---|---:|---|
| `import tensorflow` | **19.37** | dominant |
| `pickle.load(rf_model.pkl)` (82 MB, unused) | 1.04 | small |
| `load_model(cnn_model.keras)` | 0.91 | small |
| `pickle.load(mlp_model.pkl)` (unused) | 0.022 | negligible |
| `pickle.load(scaler.pkl)` + `weight_model.pkl` (scaler unused) | 0.002 | negligible |

> **Correction to the earlier audit.** That audit stated the 82 MB pickle was "a large
> share of the import cost". Measurement shows it is ~1 s of a ~20 s import. TensorFlow
> import is the real cost. Removing the dead pickles remains correct for clone size and
> hygiene, but it will not meaningfully reduce startup time.

### 4.2 Engine latency — 52 positions

From [`engine_results.json`](engine_results.json). Entry point: `engine.engine_move(chess.Board(fen))`,
called exactly as `app.py:149` does.

| Statistic | Pass 1 | Pass 2 |
|---|---:|---:|
| n | 52 | 52 |
| min | 257.08 ms | — |
| **median** | **638.03 ms** | — |
| **mean** | **616.99 ms** | — |
| p95 (nearest-rank) | 1,004.90 ms | — |
| max | 1,112.25 ms | — |
| total | 32.08 s | — |
| errors | **0** | 0 |

By category (pass 1):

| Category | n | min | median | mean | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| endgame | 12 | 260.8 | 312.4 | 332.9 | 422.1 | 432.5 |
| defensive | 4 | 322.4 | 428.7 | 436.5 | 566.1 | 566.1 |
| tactical | 6 | 257.1 | 415.9 | 509.3 | 937.4 | 937.4 |
| opening | 18 | 598.3 | 674.7 | 729.0 | 1,004.9 | 1,018.2 |
| middlegame | 12 | 660.6 | 842.7 | 847.0 | 1,015.7 | 1,112.3 |

Latency tracks legal-move count, as expected from the O(branching²) 1-ply lookahead at
`engine.py:183-202`.

**Determinism: confirmed.** Across two sequential passes, **52/52 identical moves** and
**52/52 identical top-candidate score lists**. `fully_deterministic: true`.
*Caveat:* this demonstrates run-to-run stability within a single process on one machine.
It does not prove determinism across processes, machines, or TF/BLAS versions. TensorFlow
emitted `oneDNN custom operations are on. You may see slightly different numerical
results due to floating-point round-off errors from different computation orders` at every
import, so cross-machine determinism should be treated as **unverified**.

**Board mutation: clean.** In 52/52 cases the caller's `Board` had an unchanged FEN after
`engine_move()` returned — the push/pop bookkeeping balances.

### 4.3 API latency

From [`api_results.json`](api_results.json). Flask `test_client()`; no live server, no
network. `import app` in this process: 12.33 s.

| Endpoint | Method | Status | Time | JSON? |
|---|---|---:|---:|---|
| `/reset` (initial) | POST | 200 | 498 ms | yes |
| `/` | GET | 200 | 21 ms | no (HTML) |
| `/state` | GET | 200 | 143 ms | yes |
| `/model_info` | GET | 200 | 155 ms | yes |
| `/move` `{"uci":"e2e4"}` | POST | 200 | 264 ms | yes |
| `/engine_move` | POST | 200 | **1,695 ms** | yes |
| `/analyse` | POST | 200 | 748 ms | yes |
| `/benchmark` | POST | 200 | 1,300 ms | yes |
| `/game_stats` | POST | 200 | 268 ms | yes |
| `/move` `{"uci":"e2e5"}` (illegal) | POST | **400** | 1 ms | yes |
| `/move` `{"uci":"zzzz"}` | POST | **400** | 1 ms | yes |
| `/move` `{}` | POST | **400** | 0 ms | yes |
| `/move` `text/plain` body | POST | **415** | 1 ms | **no** |
| `/move` wrong method | GET | **405** | 1 ms | no |
| `/forfeit` | POST | 200 | 182 ms | yes |
| `/reset` (final) | POST | 200 | 128 ms | yes |
| `/engine_move` on White's turn | POST | **400** | 1 ms | yes |

**All 6 required endpoints succeed.** Two observations recorded without fixing:

- `/engine_move` at 1,695 ms is ~2.7× the mean single `engine_move()` call, consistent
  with `app.py` invoking the engine twice per request (`app.py:149` and `app.py:171`).
- A non-JSON body returns **415 with an HTML body**, breaking the `{"error": ...}` JSON
  contract every other error path honours. `request.json` at `app.py:104` raises before
  the `or {}` guard can apply.

---

## 5. Fixed FEN suite

[`fens.json`](fens.json) — **52 positions**, built by
[`scripts/build_fens.py`](scripts/build_fens.py).

No prior baseline dataset existed. The only pre-existing FEN list in the repository is
the 5-position `test_positions` array in `benchmark_stockfish.py:72-78`; **all five are
folded into this suite** and carry `"source": "benchmark_stockfish.py test_positions"`
so continuity is preserved.

**Deterministic:** regenerating the file twice produced an identical SHA-256
(`ea52608148719280f8ab58afc08bc44406a2686f012d956cc36915b33a51e511`). There is no
randomness in the generator.

| Category | Count |
|---|---:|
| opening | 18 |
| middlegame | 12 |
| endgame | 12 |
| tactical | 6 |
| defensive | 4 |
| **Total** | **52** |

| Side to move | Count |
|---|---:|
| White | 30 |
| Black | 22 |

Every position was validated: `board.is_valid()` is true, none is terminal (both engine
and Stockfish must be able to return a move), and 0 candidates were dropped. Mean legal
move count: **24.87**.

Each record carries `id`, `fen`, `side_to_move`, `category`, `description`, `source`,
`provenance`, `legal_move_count`, `is_check`, `fullmove_number`, `piece_count`. Opening
and middlegame positions store their **full SAN move list from the start position**, so
their provenance is self-verifying rather than asserted.

### Stated limitations of this suite

- **It is not a representative sample.** It is a fixed comparison set, hand-curated. No
  sampling frame, no confidence intervals. Rates computed on it describe *this suite*.
- 52 slightly exceeds the 30–50 target; the extra positions were added specifically to
  raise Black-to-move coverage from 13 to 22, because the application only ever lets the
  engine play Black (`app.py:145`).
- Category labels are descriptive conveniences. Endgame positions are described by literal
  material ("king and pawn vs king") rather than by theory names, to avoid asserting
  theoretical claims that were not verified.
- 4 positions produce mate scores from Stockfish (`TC01`, `TC02`, `TC05`, `DF03`). These
  are flagged and must be excluded from any future centipawn aggregate.

---

## 6. Engine results

Per-position records in [`engine_results.json`](engine_results.json) (`pass1` / `pass2`),
each with selected move, legality, top-3 candidates with component features
(`score`, `cnn_cp`, `material`, `space`, `center`, `mobility`), explanation strings, and
execution time.

| Metric | Value |
|---|---|
| Positions | 52 |
| **Legal moves returned** | **52 / 52 = 100%** |
| Errors / exceptions | 0 |
| Deterministic (move + scores, 2 passes) | **52 / 52** |
| Input board FEN preserved after call | 52 / 52 |

Legality was re-checked against a **freshly constructed board** from the FEN, so a mutated
board inside the engine could not make an illegal move appear legal.

**Ridge model as loaded** (`models/weight_model.pkl`):

```
coef_      = [330.9005, 32.3899, 0.7919, 5.1654, 0.0188]
             [cnn_norm, material, space,  center, mobility]
intercept_ = 15.0772      # NEVER APPLIED at inference
```

`engine.py:131-132` and `engine.py:162-163` use `weight_model.coef_` only; the intercept
is not added. Recorded in `engine_results.json` as
`ridge_intercept_used_at_inference: false`. This is a source-inspection finding, not an
inference.

Sample record (`OP01`, initial position): move `d2d4`, score `92.9968`, explanation
`["strengthens control of the center", "expands space with a pawn advance", "increases
board control by +6 squares"]`.

---

## 7. Stockfish results

[`stockfish_results.json`](stockfish_results.json). Configuration taken **unchanged** from
the repository.

| Item | Value | Source |
|---|---|---|
| Binary | `...\stockfish-windows-x86-64-avx2.exe` | hardcoded literal, `app.py:383` / `benchmark_stockfish.py:12` |
| Version | **Stockfish 17.1** | UCI banner |
| Access | `chess.engine.SimpleEngine.popen_uci` | as `app.py:386` |
| Depth | **8** | `STOCKFISH_DEPTH`, `app.py:384` / `benchmark_stockfish.py:13` |
| Threads | **1** (Stockfish default) | **repo sets nothing** |
| Hash | **16 MB** (Stockfish default) | **repo sets nothing** |
| Skill Level | 20 (default) | repo sets nothing |

> The repository sets **no UCI options at all**. Threads and Hash are Stockfish defaults,
> not deliberate choices. Because they are not pinned by the code, the reference moves
> are **not guaranteed bit-reproducible** across runs or machines.

| Statistic | `engine.play()` latency |
|---|---:|
| n | 52 |
| min | 0.98 ms |
| **median** | **3.42 ms** |
| **mean** | **5.09 ms** |
| p95 | 15.39 ms |
| max | 18.05 ms |
| total | 264.4 ms |

**Deviations from `app.py`, all recorded in the artifact:**

1. One engine process is reused for all 52 positions; `app.py:386` spawns a process per
   `/benchmark` request. This affects **timing only**, and makes these Stockfish timings
   **faster** than `app.py` would achieve. The engine-vs-Stockfish latency gap reported
   below is therefore, if anything, generous to Stockfish — and the engine still loses by
   121×.
2. `MultiPV=3` was requested so top-3 containment could be computed. The existing
   repository never requests MultiPV.
3. `score.white()` (absolute POV) was captured alongside `score.relative` (what
   `app.py:395` uses), so later phases have an unambiguous frame.

Both `score_white_cp` and `score_relative_cp` are stored per position, plus `is_mate`,
`mate_in`, and `depth`.

---

## 8. Valid comparisons

[`comparison.json`](comparison.json). **Only unit-free, POV-free metrics are reported.**

| Metric | Value | Chance level | Ratio to chance |
|---|---:|---:|---:|
| **Engine legality rate** | **100%** (52/52) | — | — |
| **Top-1 move agreement** | **19.23%** (10/52) | 5.91% | **3.25×** |
| **Top-3 containment** | **36.54%** (19/52) | 17.72% | **2.06×** |
| **Latency ratio (engine ÷ Stockfish)** | **121.33×** | — | — |

Latency ratio detail: engine total **32,083.4 ms**, Stockfish total **264.4 ms**, same
machine, same 52 positions, run sequentially.

### By category

| Category | n | Top-1 | chance | Top-3 |
|---|---:|---:|---:|---:|
| endgame | 12 | **41.67%** | 12.15% | 58.33% |
| middlegame | 12 | 16.67% | 2.84% | 25.00% |
| tactical | 6 | 16.67% | 4.73% | 16.67% |
| opening | 18 | 11.11% | 3.58% | 33.33% |
| defensive | 4 | **0.00%** | 8.62% | 50.00% |

### By side to move

| Side | n | Top-1 | Top-3 |
|---|---:|---:|---:|
| White | 30 | 23.33% | 33.33% |
| Black | 22 | 13.64% | 40.91% |

**Reading these honestly.** Endgame top-1 looks strongest at 41.7%, but endgames have far
fewer legal moves, so chance agreement there is also highest (12.2%). Relative to chance,
middlegame is actually the best category (5.9× chance) and endgame is 3.4×. Defensive
scored 0/4, below its chance level — but n=4 is far too small to support any conclusion,
and it is reported only for completeness. **No per-category claim in this table is
statistically supported**; they are descriptive counts on a fixed 52-position set.

Positions where the engine matched Stockfish's top move: `OP09`, `OP13`, `MG08`, `MG11`,
`EG01`, `EG05`, `EG08`, `EG10`, `EG11`, `TC06`.

### Deliberately NOT computed

**`average centipawn delta` (engine score − Stockfish score) — invalid, omitted.**
The engine's score is a Ridge-weighted sum of a `tanh`-squashed CNN output plus raw
material / space / centre / mobility counts (`engine.py:160-165`). It is not in
centipawns; the Ridge intercept (15.0772) is discarded; and its point of view is never
stated. Stockfish's value is in centipawns from a defined POV. Subtracting one from the
other is a unit error, so it was excluded rather than reported with a caveat.

Note that **`app.py:411` computes exactly this quantity** and returns it as
`comparison.eval_diff_cp`. That field should not be treated as a quality measure.

**`move-quality regret` (average centipawn loss) — correct metric, not yet computed.**
This is the metric that *would* be valid, because both sides of the subtraction are
measured by Stockfish on one scale. It requires a **second** Stockfish evaluation of the
position after the engine's move, which the existing repository never performs. Computing
it would exceed "measure what exists", so it is deferred and flagged as the recommended
metric for a later phase.

### Caveats on everything in §8

- Top-1 agreement against a depth-8 reference is a weak notion of quality. Many positions
  have several near-equal good moves; a disagreement is not necessarily an error.
- Stockfish ran with unpinned Threads/Hash, so reference moves are not guaranteed
  bit-reproducible.
- 52 positions, no confidence intervals, not a random sample.

---

## 9. Documentation integrity check

README was **not edited**. Claims compared against measurement and against the notebook's
own retained outputs.

| # | Claim | Location | Actual evidence | Status |
|---|---|---|---|---|
| 1 | Pearson correlation **0.506** vs Stockfish | `README.md:123`, `README.md:456`, `README.md:462`, `app.py:470`, `generate_visualizations.py:69,154` | Notebook cell 16 retained output: **`Pearson Correlation with Stockfish: 0.708`**; cell 17 confirms `Correlation: 0.708` | **inaccurate** (understates the real figure) |
| 2 | "held-out test set of **50 positions**" | `README.md:123` | `train_test_split(test_size=0.2)` on 10,000 → **~2,000** positions; cell 16 output shows 63 prediction batches | **inaccurate** |
| 3 | "**~50,000** positions" labelled by Stockfish | `README.md:106`, `README.md:472`, `app.py:469` | Notebook cell 14: `df.sample(10000, random_state=42)`; retained output `Collected: 10000 valid positions` | **inaccurate** (5× overstated) |
| 4 | Engine is "**6.2x faster**" than Stockfish depth 8 | `generate_visualizations.py:37` → `static/performance_comparison.png` | Measured: engine mean **616.99 ms**, Stockfish depth-8 mean **5.09 ms**; ratio **121.33× slower** | **inaccurate — inverted** |
| 5 | Engine is "**62x faster**" than Stockfish depth 15 | `generate_visualizations.py:39` → same chart | Depth 15 not measured; but depth-8 result already contradicts the direction of the claim | **inaccurate** (direction contradicted; exact depth-15 figure unverified) |
| 6 | Chart source values `times = [40, 250, 2500]` ms | `generate_visualizations.py:23` | Hardcoded literals. No measurement in that file. Engine measured at 617 ms mean, not 40 ms | **fabricated** |
| 7 | "Top Move Agreement **0.55**" | `generate_visualizations.py:69` → `static/accuracy_metrics.png` | Hardcoded literal. Measured top-1 agreement: **0.1923** | **fabricated** |
| 8 | "MAE **180** centipawns" | `generate_visualizations.py:69` | Hardcoded literal; no MAE is computed anywhere in the repository | **fabricated / unverifiable** |
| 9 | Training-loss and correlation curves in `static/training_progress.png` | `generate_visualizations.py:135-136,147` | Generated from `25000*np.exp(-epochs/15)+15000+np.random.normal(...)` and `0.1+0.406*(1-np.exp(-epochs/10))+noise`. Purely synthetic, while real per-epoch history exists in notebook cell 15 output | **fabricated** |
| 10 | Model size "**11.2 MB**" / "**~9.1 MB**" | `generate_visualizations.py:44`, `:120` | `/model_info` reports **2,360,129** params ≈ 9.0 MB at fp32; file on disk is **28,381,519 bytes (27.1 MB)** | **inaccurate / inconsistent** |
| 11 | "Stockfish 50 MB", "4.5x smaller" | `generate_visualizations.py:44,57` | Stockfish binary on this machine: **79,835,136 bytes (76.1 MB)** | **inaccurate** |
| 12 | Top-move agreement rate printed by the benchmark suite | `benchmark_stockfish.py:57,129-130` | `results["agreement"] = np.random.rand() < 0.7` — the computed `hybrid_move` and `stockfish_move` on the two preceding lines are discarded | **fabricated (randomly generated)** |
| 13 | "Average time 9.2ms / 15.1ms", "Min 7.10ms", "Max 12.30ms", "Speedup 0.60x" | `benchmark_stockfish.py:113-126` | Hardcoded literals printed while the genuinely measured `hybrid_times` / `stockfish_times` lists (computed at `:109-110`) go unused | **fabricated** |
| 14 | `correlation_vs_stockfish: 0.506` served by the live API | `app.py:470` | Same as #1 | **inaccurate** |
| 15 | Prerequisite "Python 3.10 or higher" | `README.md` prerequisites | True as stated, but `requirements.txt` pins `tensorflow==2.15.0`, which is not installable on 3.12 — the interpreter actually in use | **misleading** |
| 16 | API reference lists 8 endpoints | `README.md:405-414` | 10 routes exist; `/benchmark` and `/model_info` are undocumented | **incomplete** |
| 17 | "Stockfish binary (only needed to re-run training — not required to play)" | README prerequisites | **Accurate.** Gameplay paths never call Stockfish; only `/benchmark` does | **accurate** |
| 18 | Batched inference across all legal moves in a single CNN call | `README.md:193` | **Accurate.** `engine.py:151` and `engine.py:195` each issue one batched `predict` | **accurate** |
| 19 | Ridge weights fitted against Stockfish targets | README §4 | **Accurate.** Notebook cell 31 fits `Ridge(alpha=1.0)` on `sf_targets` | **accurate** |
| 20 | Integrated Gradients saliency over the final position | README §8 | **Accurate.** `app.py:274-303`: baseline, 50-step interpolation, gradient averaging, channel collapse, normalisation. `/game_stats` returned a populated 8×8 map | **accurate** |

**Summary: 20 claims checked — 4 accurate, 6 fabricated, 9 inaccurate, 1 incomplete.**

---

## 10. Known issues

Observed during this phase. **Nothing was fixed.** Items marked *hypothesis* were not
empirically confirmed.

### Correctness

| ID | Issue | Evidence |
|---|---|---|
| C1 | Heuristic bonuses are added with a fixed positive sign (`engine.py:168-171`) while the sort direction flips by side (`engine.py:204`). For Black, lower is better, so a "good move" bonus pushes the move **down** Black's own preference list. The app only ever lets the engine play Black (`app.py:145`), so this path is always active | Source inspection + `engine_results.json` (bonus values recorded per candidate) |
| C2 | The 1-ply lookahead takes `max` over opponent replies (`engine.py:198`), selecting the reply most favourable to the mover, then penalises for it (`engine.py:202`). The opponent minimises, so the extremum is inverted | Source inspection |
| C3 | `center` reported in the candidate dict (`engine.py:179`) is computed **after** `board.pop()` (`engine.py:166`) — the pre-move value — while `material`, `space`, `mobility` in the same dict are post-move | Source inspection; both values present in `engine_results.json` |
| C4 | Ridge `intercept_` (15.0772) is never applied, so the score is not the Ridge model's prediction and cannot be read as centipawns | `engine_results.json: ridge_intercept_used_at_inference=false` |
| C5 | `opening_center_bonus` (`engine.py:97`) matches only White UCI strings `e2e4/d2d4/c2c4` | Source inspection |
| C6 | `/forfeit` hardcodes result `"0-1"` (`app.py:249`) regardless of who resigned | Source inspection; `api_results.json` |
| C7 | `mobility_score` mutates `board.turn` in place (`engine.py:67-69`); restored, but `ep_square` is untouched while flipped | Source inspection. *No incorrect output observed*: 52/52 input FENs were preserved |
| C8 | *Hypothesis:* `board_to_planes` encodes no side-to-move, castling or en-passant plane, so the CNN cannot distinguish whose turn it is — yet `rerank_moves` evaluates post-move positions where it is the opponent's turn. Not measured in Phase 0 | Source inspection (`engine.py:27-36`) |

### Performance

| ID | Issue | Evidence |
|---|---|---|
| P1 | Engine is **121× slower** than Stockfish depth 8 on identical positions | `comparison.json` |
| P2 | `import engine` costs **17–24 s warm, 76 s cold**, dominated by **TensorFlow (19.4 s)**, paid on every process start | `startup_results.json` + attribution measurement |
| P3 | `/engine_move` calls the engine twice (`app.py:149`, `app.py:171`) — 1,695 ms vs ~617 ms for one call | `api_results.json` |
| P4 | 1-ply lookahead is O(branching²): ~400 CNN evaluations vs ~20 for the main pass. Latency tracks legal-move count (endgame 333 ms → middlegame 847 ms mean) | `engine_results.json: latency_by_category_ms` |
| P5 | 79 MB of unused model pickles loaded at import (`rf`, `mlp`, `scaler`). Costs ~1.07 s and 79 MB of clone weight — **not** the 20 s previously assumed | Attribution measurement |
| P6 | 107 MB of model files tracked in git without LFS | `environment.json: models` |

### Reproducibility

| ID | Issue |
|---|---|
| R1 | `requirements.txt` pins `tensorflow==2.15.0` (Keras 2) but the model records `keras_version: 3.11.1`. *Hypothesis:* a clean install cannot load the model — **not empirically verified** |
| R2 | Declared Python 3.10.13 vs actual 3.12.10 |
| R3 | scikit-learn unpinned; pickles written by 1.7.1, loaded under 1.8.0 with `InconsistentVersionWarning` |
| R4 | Stockfish path hardcoded in 3 places (`app.py:383`, `benchmark_stockfish.py:12`, notebook cells 11/14/42) |
| R5 | `engine.py:7` uses CWD-relative model paths — **reproduced this phase** (`ModuleNotFoundError` from `baseline/scripts/`) |
| R6 | `gunicorn` in `Procfile` and `requirements.txt` but not installed |
| R7 | `games.csv` is gitignored, so `chess_model_FINAL.ipynb` cannot be re-run from a clean clone |
| R8 | TensorFlow warns `oneDNN ... may see slightly different numerical results` at every import — cross-machine determinism unverified |
| R9 | Emoji in `print()` (`engine.py:21`) crashes under a cp1252 stdout; `PYTHONIOENCODING=utf-8` was required throughout this phase |

### Testing

| ID | Issue |
|---|---|
| T1 | **Zero test files**, zero assertions, no `pytest` installed, no `pyproject.toml`/`pytest.ini` |
| T2 | **Zero CI configuration** — no `.github/` |
| T3 | `app.py` module-level globals (`app.py:13-20`) make requests order-dependent and shared; tests would need a `/reset` fixture and could not run in parallel |
| T4 | Model loading at import scope means any test session pays 17–76 s before the first assertion |
| T5 | Silent-failure handlers would hide defects from assertions: bare `except:` (`engine.py:164`), `except Exception: pass` (`app.py:127`, `app.py:164`), `except (FileNotFoundError, Exception)` (`app.py:413`) |
| T6 | `check_strucutre.py` is `print`-based with no assertions or exit code |

### Evaluation

| ID | Issue |
|---|---|
| E1 | `benchmark_stockfish.py:57` generates its agreement rate with `np.random.rand() < 0.7`, discarding the real moves computed immediately above |
| E2 | `benchmark_stockfish.py:113-126` prints hardcoded timings while ignoring measured lists |
| E3 | `generate_visualizations.py` is entirely synthetic, including simulated training curves — and its outputs are the committed `static/*.png` files the README displays |
| E4 | `app.py:411` computes `eval_diff_cp` by subtracting a side-to-move-relative Stockfish score from a non-centipawn engine score — two different units and two different POVs |
| E5 | No MultiPV, no regret/centipawn-loss, no legality check, no chance baseline anywhere in the existing code |
| E6 | Stockfish invoked without pinned Threads/Hash, so the reference is not reproducible by construction |
| E7 | `except (FileNotFoundError, Exception)` (`app.py:413`) reports every failure as "Stockfish not installed", masking real errors |

### Documentation

See §9. **6 fabricated claims, 9 inaccurate, 1 incomplete, 4 accurate.** The most serious
is #4/#12/#13: the repository ships a chart and a benchmark script asserting the engine is
6–62× faster than Stockfish, while measurement — and the app's own `/benchmark` endpoint,
which returns `"speedup": 0.01` — show it is ~121× slower.

---

## 11. Exact commands used

All run from the repository root with the project venv. `PYTHONIOENCODING=utf-8` is
required because of emoji in `engine.py:21` (see R9).

```bash
# --- 0. Safety checks ---
test -e baseline && echo "CONFLICT" || echo "OK"
git status --porcelain
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

# --- 1. Fixed FEN suite (deterministic; rerun gives identical bytes) ---
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe \
    baseline/scripts/build_fens.py baseline/fens.json

# --- 2. Engine over the suite, two passes (determinism + latency) ---
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe \
    baseline/scripts/measure_engine.py \
    baseline/fens.json baseline/engine_results.json

# --- 3. Stockfish over the same suite, existing config (depth 8) ---
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe \
    baseline/scripts/measure_stockfish.py \
    baseline/fens.json baseline/stockfish_results.json

# --- 4. Valid comparisons only ---
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe \
    baseline/scripts/compare.py \
    baseline/engine_results.json baseline/stockfish_results.json \
    baseline/comparison.json baseline/fens.json

# --- 5. Flask endpoints via test client ---
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe \
    baseline/scripts/measure_api.py baseline/api_results.json

# --- 6. Environment / inventory / reproducibility conflicts ---
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe \
    baseline/scripts/capture_environment.py baseline/environment.json

# --- 7. Import cost in fresh subprocesses ---
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe \
    baseline/scripts/measure_startup.py . baseline/startup_results.json

# --- 8. Import cost attribution (ad hoc, one-off) ---
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe -c "
import time,pickle
t=time.perf_counter(); import tensorflow; print('tf', time.perf_counter()-t)
from tensorflow.keras.models import load_model
t=time.perf_counter(); load_model('models/cnn_model.keras', compile=False); print('cnn', time.perf_counter()-t)
for n in ['rf_model','mlp_model','scaler','weight_model']:
    t=time.perf_counter(); pickle.load(open('models/'+n+'.pkl','rb')); print(n, time.perf_counter()-t)
"
```

Re-running steps 1–7 in order reproduces every artifact except wall-clock timings.

---

## 12. What could not be measured reliably

1. **True cold-start import.** The 76.3 s figure was observed once, early in the session,
   before OS caches warmed. Subsequent fresh-subprocess runs measured 17–24 s. No
   post-reboot measurement was taken. Both are reported; neither is a clean cold-boot number.

2. **Whether `requirements.txt` actually fails to load the model.** Installing
   `tensorflow==2.15.0` was out of scope (no dependency installation permitted in this
   phase). The conflict is inferred from `keras_version: 3.11.1` recorded inside the model
   archive versus the Keras 2.x that TF 2.15 ships. **Labelled a hypothesis.**

3. **Cross-machine / cross-process determinism.** Verified only within one process on one
   machine. TensorFlow's oneDNN warning explicitly flags possible floating-point variation
   from differing computation orders. **Unverified.**

4. **Stockfish reproducibility.** Threads/Hash were left at defaults because the repository
   sets nothing. Reference moves are therefore not guaranteed stable run-to-run. No
   repeat-run check of Stockfish was performed.

5. **Depth-15 Stockfish comparison** (claim #5). Not measured; only the direction of the
   claim is contradicted, via the depth-8 result.

6. **MAE in centipawns** (claim #8). Cannot be computed: it would require the invalid
   score subtraction described in §8. Marked unverifiable rather than false.

7. **Statistical significance of any rate.** 52 positions, purposively selected. No
   confidence intervals are given because the suite has no sampling frame. Per-category
   rates (n as low as 4) are descriptive only.

8. **API timings are sequence-dependent.** Because `app.py` holds global state, the
   numbers in §4.3 reflect that specific request order, not isolated per-endpoint cost.

9. **`/benchmark` timing includes Stockfish process spawn.** Its 1,300 ms is not
   comparable to the 5.09 ms per-position Stockfish figure, which reuses one process.

10. **Correction recorded.** The earlier audit's claim that the 82 MB pickle was a large
    share of import cost was wrong; measurement attributes the cost to TensorFlow. Noted
    here so the error is not silently carried into Phase 1.

---

## 13. Artifact index

| File | Contents |
|---|---|
| `BASELINE.md` | This report |
| `environment.json` | Git state, interpreter, packages, model SHA-256s, Stockfish, 6 reproducibility conflicts |
| `fens.json` | 52-position fixed suite with provenance |
| `engine_results.json` | Both passes, per-position moves/scores/explanations/timings, determinism, legality |
| `stockfish_results.json` | Per-position best move, MultiPV-3, both POV scores, timings, UCI options |
| `comparison.json` | Valid metrics only, chance baselines, breakdowns, explicit exclusions |
| `api_results.json` | 17 requests: status, latency, JSON-ness, error fields |
| `startup_results.json` | Fresh-subprocess import timings |
| `logs/engine_run.log` | Raw stdout of the engine measurement run |
| `scripts/*.py` | The seven scripts that produce all of the above |
