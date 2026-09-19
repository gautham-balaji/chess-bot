# Reproducibility

How to set this project up, what it depends on, and what still does not work.
Written after the Phase 1 integrity pass.

---

## Supported Python version

**Python 3.12** — verified on **3.12.10** (Windows 11, AMD64, CPython).

`runtime.txt` and `.python-version` both declare `3.12.10`. Before Phase 1 they
declared `3.10.13`, which was not the interpreter the project actually ran on.

Other 3.12.x patch releases should work. 3.13 is untested. 3.10/3.11 are untested
against the pinned TensorFlow build.

---

## Dependency installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

To additionally re-run the training notebook:

```bash
pip install -r requirements.txt -r requirements-notebook.txt
```

### Runtime pins (`requirements.txt`)

| Package | Version | Why pinned here |
|---|---|---|
| `tensorflow` | 2.21.0 | Verified to load the shipped Keras 3 model |
| `keras` | 3.13.2 | Ships with TF 2.21; pinned for determinism |
| `scikit-learn` | 1.8.0 | Needed to unpickle the Ridge weight model |
| `scipy` | 1.17.1 | scikit-learn dependency |
| `numpy` | 2.4.3 | Verified environment |
| `chess` | 1.11.2 | Move generation and UCI client |
| `flask` | 3.1.3 | Web server |
| `flask-cors` | 6.0.2 | CORS |
| `gunicorn` | 23.0.0 | Used by `Procfile`; **Linux/macOS only**, not installed locally |

Every pin except `gunicorn` is already satisfied by the verified development
environment — confirmed with `pip install --dry-run`, which reported only
`gunicorn` as needing installation.

> **Package name gotcha.** The distribution is **`chess`**, not `python-chess`.
> The `python-chess` alias on PyPI stops at version `1.999`, so
> `python-chess==1.11.2` is **not installable**. Earlier revisions of
> `requirements.txt` listed `python-chess`; this was caught by a dry-run resolve.

### Removed from the dependency set

`streamlit` was listed in the original `requirements.txt` but is imported nowhere
in this repository. It was removed.

---

## Model / dependency compatibility notes

### TensorFlow and the shipped CNN — resolved

`requirements.txt` previously pinned `tensorflow==2.15.0`. That pin was wrong.

- `models/cnn_model.keras` internally reports `keras_version: 3.11.1`
- its archive layout is the Keras 3 format: `config.json` + `metadata.json` +
  `model.weights.h5`
- TensorFlow 2.15 bundles Keras 2.x, which does not implement that format

So a clean install from the old `requirements.txt` produced an application that
could not load its own model. The pin is now `tensorflow==2.21.0` /
`keras==3.13.2`, under which the model loads.

> This was **not** verified by actually installing TF 2.15 — doing so would have
> destroyed the verified working environment. The conclusion rests on the Keras
> version and archive layout recorded inside the model file itself.

### scikit-learn pickle version skew — present, benign, not hidden

The `.pkl` model files were serialised with **scikit-learn 1.7.1**. The pinned
runtime is **1.8.0**, so loading them emits:

```
InconsistentVersionWarning: Trying to unpickle estimator Ridge from version
1.7.1 when using version 1.8.0. This might lead to breaking code or invalid results.
```

**This warning is real and is deliberately not suppressed.** For the only pickle
the runtime actually loads (`weight_model.pkl`, a `Ridge`), the risk is minimal:
the object carries a 5-element `coef_` array and a scalar `intercept_`. Both were
verified byte-identical to the Phase 0 baseline after Phase 1:

```
coef_      = [330.9005, 32.3899, 0.7919, 5.1654, 0.0188]
intercept_ = 15.0772
```

Re-serialising the pickles under 1.8.0 would remove the warning, but it would
change model files, which is explicitly out of scope for an integrity pass.

### NumPy

NumPy 2.4.3 is in use. TensorFlow 2.21 and scikit-learn 1.8 both support NumPy 2.

---

## Model files

Committed under `models/`. Paths resolve **relative to the repository**, anchored
to `config.py`'s own location — not to the shell's working directory.

| File | Size | Loaded at runtime? |
|---|---:|---|
| `cnn_model.keras` | 27.1 MB | **Yes** — CNN position evaluator |
| `weight_model.pkl` | 480 B | **Yes** — Ridge fusion weights |
| `rf_model.pkl` | 77.9 MB | No — training artifact |
| `mlp_model.pkl` | 1.01 MB | No — training artifact |
| `scaler.pkl` | 862 B | No — training artifact |

Before Phase 1, `engine.py` unpickled all five at import. The three unused ones
cost ~1.07 s and ~79 MB of resident memory for no effect; they are no longer
loaded. **The files are retained on disk**: they are genuine outputs of the
training notebook, they are documented in the README, and the notebook's
`R² = 0.5146` / accuracy `0.7228` results refer to them.

To load models from elsewhere, set `CHESS_BOT_MODELS_DIR`.

---

## Stockfish setup

Stockfish is **optional**. Gameplay never calls it. It is used only by the
`/benchmark` endpoint and by the training notebook.

**Version used for the recorded baseline: Stockfish 17.1.** Any recent Stockfish
that speaks UCI should work. Search depth is fixed at **8** in `config.py`
(`STOCKFISH_DEPTH`), matching the depth the CNN's training labels were generated at.

### How the executable is configured

`config.find_stockfish()` resolves in this order:

1. **`STOCKFISH_PATH` environment variable** — explicit wins
2. **`stockfish` on `PATH`**
3. **Common per-platform install locations**
   - Linux: `/usr/games/stockfish`, `/usr/bin/stockfish`, `/usr/local/bin/stockfish`
   - macOS: `/opt/homebrew/bin/stockfish`, `/usr/local/bin/stockfish`
   - Windows: `C:\Program Files\Stockfish\stockfish.exe`, `C:\stockfish\stockfish.exe`

Before Phase 1 the path was a hardcoded absolute Windows path in three places, so
Stockfish-dependent code only worked on one machine.

### Installing

```bash
# Debian / Ubuntu
sudo apt-get install -y stockfish

# macOS
brew install stockfish

# Windows: download from stockfishchess.org, then set STOCKFISH_PATH
```

### When Stockfish is unavailable

Nothing breaks. `find_stockfish()` returns `None` rather than raising, and
`/benchmark` returns HTTP 200 with:

```json
{
  "stockfish": {
    "available": false,
    "note": "Stockfish not found. Set the STOCKFISH_PATH environment variable or install stockfish on PATH."
  },
  "comparison": { "note": "Stockfish comparison unavailable" }
}
```

The UI shows "Not installed". All other endpoints are unaffected.

---

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `STOCKFISH_PATH` | No | auto-detect | Absolute path to the Stockfish executable |
| `CHESS_BOT_MODELS_DIR` | No | `<repo>/models` | Override model directory |
| `PORT` | No | `5000` | Port for `python app.py` |
| `TF_CPP_MIN_LOG_LEVEL` | No | `2` (set by `app.py`) | TensorFlow log verbosity |

### `.env`

`config.py` reads an optional `.env` at the repository root (`KEY=VALUE`, `#`
comments). **Real environment variables always take precedence.** `.env` is
gitignored because it holds machine-specific paths.

```bash
cp .env.example .env
# edit STOCKFISH_PATH
```

No third-party dotenv dependency is used; the parser is ~15 lines in `config.py`
and fails silently if the file is malformed, so it can never break startup.

---

## Starting the application

```bash
python app.py                 # http://127.0.0.1:5000
PORT=5055 python app.py       # custom port
```

Because model paths are repo-relative, `python /abs/path/to/app.py` also works
from any directory.

Production (Linux/macOS only):

```bash
gunicorn app:app
```

> **Do not run gunicorn with more than one worker.** `app.py` keeps a single game
> in module-level globals, so multiple workers would each hold a different board
> and players would see the game jump between states.

Importing the modules from another directory requires the repo root on
`PYTHONPATH`, since this is a flat script layout rather than an installed package:

```bash
PYTHONPATH=/path/to/chess-bot python -c "import engine"
```

---

## Verification

Quick checks that the setup is sound:

```bash
# models resolve and load; Stockfish resolution reported
python -c "import config, engine; print(config.stockfish_status()); print(engine.weight_model.coef_)"

# engine produces a move
python -c "import chess, engine; print(engine.engine_move(chess.Board())[0])"

# full API surface
python baseline/scripts/measure_api.py /tmp/api.json

# engine behaviour still matches the frozen baseline
python baseline/scripts/measure_engine.py baseline/fens.json verification/engine.json
python verification/compare_phase0_phase1.py \
    baseline/engine_results.json verification/engine.json verification/diff.json
```

---

## Known limitations

### Cannot be fixed by configuration

1. **The training notebook cannot be re-run from a clean clone.** Its input
   `games.csv` (7.3 MB) is gitignored and not distributed. The notebook retains
   its outputs, so its results are inspectable but not reproducible here.
2. **107 MB of model files are tracked in git without LFS**, ~79 MB of which the
   runtime never loads. Clones are slow.
3. **Import takes 13–24 s warm, and was measured at 76 s cold.** This is
   dominated by **TensorFlow itself (~19 s)**, not by model loading. Removing the
   unused pickles saved ~1 s — it did not meaningfully change startup.
4. **`gunicorn` does not run on Windows**, so the `Procfile` path is untestable
   on the primary development machine.
5. **Single shared game state.** One board for all clients; requests are
   order-dependent.
6. **Cross-machine determinism is unverified.** TensorFlow reports
   `oneDNN custom operations are on. You may see slightly different numerical
   results...` at every import. Determinism was confirmed only within one process
   on one machine.
7. **Stockfish is not pinned for reproducibility.** `Threads` and `Hash` are left
   at Stockfish defaults, so reference moves are not guaranteed identical
   run-to-run.
8. **Emoji in source crashed Windows consoles.** `engine.py` printed
   `✅ All models loaded` at import, which raises `UnicodeEncodeError` under a
   cp1252 stdout. Replaced with a `logging` call. Other project scripts may still
   need `PYTHONIOENCODING=utf-8`.

### Deliberately left alone by Phase 1

Phase 1 changed **no engine behaviour**. Verified: all 52 baseline positions
return identical moves, identical top-3 orderings and identical scores.

These known defects were therefore **not** fixed, because fixing them would
change engine output:

- Heuristic move bonuses are added with a fixed positive sign while the ranking
  sort direction flips by side, so for Black the bonuses push good moves *down*
  its own preference list. The engine only ever plays Black in the app.
- The 1-ply lookahead takes `max` over opponent replies, selecting the reply most
  favourable to the mover rather than the opponent's best.
- The Ridge `intercept_` (15.0772) is never applied at inference.
- `center` reported in each candidate dict is the pre-move value, while
  `material` / `space` / `mobility` in the same dict are post-move.
- `opening_center_bonus` matches only White's UCI strings.
- `POST /move` with a non-JSON body returns 415 HTML instead of a JSON error.
- `/engine_move` runs the engine twice per request.
- `/forfeit` hardcodes the result `"0-1"`.

These are recorded in `docs/PHASE_1_REPORT.md` under remaining issues.
