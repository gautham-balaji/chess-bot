# Testing

The automated test suite, added in Phase 2.

---

## What this suite does and does not answer

This distinction matters and is easy to blur.

**pytest answers:** *"Does the implementation honour its software contracts?"*
Does `board_to_planes` produce the right shape? Does `engine_move` always return
a legal move? Does `POST /move` reject an illegal move with 400 and a JSON body?
Is the output deterministic? These have objectively right answers.

**A chess-evaluation harness answers:** *"How good is this engine?"*
How often does it agree with Stockfish? How much centipawn quality does it give
up per move? These have no pass/fail answer — they are measurements on a
distribution, and they belong in a separate harness with its own report.

**The suite in this directory is entirely the first kind.** It contains no
quality judgements about chess play. The closest thing to a quality measurement
currently in the project is the frozen Phase 0 baseline in `baseline/`, and even
that is used here only as a *change detector* — see below.

---

## Installing

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

`requirements-dev.txt` adds exactly one package: `pytest==9.1.1`.

---

## Running

```bash
# everything (~3m15s)
pytest

# skip the 52-position baseline replay (~2m10s)
pytest -m "not slow"

# unit tests only - fast feedback loop (~21s)
pytest tests/unit

# integration only
pytest tests/integration

# API only
pytest tests/api

# just the Phase 0 regression check
pytest tests/integration/test_regression_baseline.py

# see which deferred tests exist and why
pytest -m deferred -v
```

On Windows, prefix with `PYTHONIOENCODING=utf-8` if your console is cp1252.

### Expected runtime

| Selection | Tests | Time |
|---|---:|---:|
| `pytest` (full) | 214 | **~196s** |
| `pytest -m "not slow"` | 206 | ~133s |
| `pytest tests/unit` | 115 | ~21s |

Most of the time is unavoidable: importing TensorFlow costs ~13–24s once per
session, and every `engine_move()` call costs ~600ms because the engine runs
~400 CNN forward passes per position.

---

## Layout

```
tests/
├── conftest.py                         fixtures: modules, boards, baseline data
├── unit/
│   ├── test_board_encoding.py          board_to_planes
│   ├── test_heuristics.py              material, space, centre, mobility, bonuses
│   ├── test_explanations.py            explain_move
│   └── test_model_fusion.py            model loading, Ridge shape, position_metrics
├── integration/
│   ├── test_engine.py                  rerank_moves + engine_move
│   └── test_regression_baseline.py     replay of the frozen Phase 0 suite  [slow]
└── api/
    ├── test_app.py                     endpoint contracts and schemas
    └── test_errors.py                  error paths and edge cases
```

Configuration lives in `pyproject.toml`: `pythonpath = ["."]` so `import engine`
works without the repo root being the working directory, plus marker
registration and `--strict-markers`.

### Markers

| Marker | Meaning |
|---|---|
| `slow` | Replays all 52 baseline positions (~48s). Deselect with `-m "not slow"` |
| `needs_stockfish` | Auto-skipped when no Stockfish binary resolves |
| `deferred` | Documents a known defect Phase 4 is expected to fix |

### Cost and isolation

Two constraints shape `conftest.py`:

- **`engine_mod` and `app_mod` are session-scoped.** TensorFlow and the CNN load
  once per pytest session rather than once per test.
- **The `client` fixture resets game state before *and* after each test.**
  `app.py` keeps the board in module-level globals, so without this the API tests
  would leak into each other and become order-dependent. `test_errors.py` contains
  a deliberate pair of near-duplicate tests that only pass if that cleanup works.

---

## How the Phase 0 baseline is used

`tests/integration/test_regression_baseline.py` replays all 52 positions from
`baseline/fens.json` and compares against `baseline/engine_results.json`,
checking legality, selected move, top-3 ordering and top-3 scores.

**It is a change detector, not a correctness oracle.**

`baseline/engine_results.json` records what the engine *did* at commit `ccd24c4`
on 2026-09-19 — **bugs included**. Several recorded moves are produced by the
defects listed below. The file does not say those moves are good.

> **When Phase 4 fixes an engine bug, this file is expected to fail.** That is the
> suite working correctly. The right response is to re-record the baseline and
> document the diff — never to revert the engine to make the test green again.

---

## Known expected failures

All 11 are `xfail(strict=True)`, meaning **if the underlying bug is fixed, the
test XPASSes and the suite fails.** That is deliberate: it forces someone to come
back here and remove the marker rather than letting a stale "known bug" note
linger after the bug is gone.

Every one of these is a **known engine/API defect**, not a test defect and not an
environment problem.

| Test | Defect |
|---|---|
| `test_positive_bonus_must_not_worsen_a_move_for_the_side_to_move[black]` | **C1** — heuristic bonuses are added with a fixed positive sign while Black's ranking sorts ascending, so a bonus pushes a move *down* Black's own preference list. The identical assertion **passes for White** |
| `test_hybrid_score_should_equal_the_ridge_prediction` | **C3** — the runtime uses `weight_model.coef_` only and drops `intercept_` (15.0772), so the score is not the Ridge model's prediction |
| `test_candidate_center_should_be_the_post_move_value` | **C4** — `center` in each candidate dict is the *pre*-move value while `material`/`space`/`mobility` are *post*-move |
| `test_opening_center_bonus_should_be_colour_symmetric` | **C5** — matches only White's UCI strings, so Black's mirrored pushes score 0 |
| `test_side_to_move_should_be_representable` | **C6** — no side-to-move plane in the encoding |
| `test_castling_rights_should_be_representable` | C6 — castling rights not encoded |
| `test_en_passant_should_be_representable` | C6 — en-passant state not encoded |
| `test_helpers_should_not_corrupt_board_when_move_is_illegal[tactical_move_bonus]` | **New in Phase 2** — pushes before validating; an illegal move from an empty square raises after the push, so the pop never runs and the caller's board is left mutated |
| `test_helpers_should_not_corrupt_board_when_move_is_illegal[move_impact]` | same |
| `test_non_json_body_should_return_400_json` | `request.json` raises `UnsupportedMediaType` before the handler's guard, giving 415 + HTML instead of 400 + JSON |
| `test_malformed_json_body_should_return_400_json` | same root cause |

### Paired characterisation tests

Each deferred test has a **passing** counterpart that pins today's actual
behaviour — for example `test_opening_center_bonus_current_behaviour_is_white_only`
and `test_non_json_body_currently_returns_415_html`.

The pairing is deliberate. One test says *"this is what it does"*, the other says
*"this is what it should do"*. Together they make the defect unambiguous, and they
mean the suite never silently blesses a bug as intended behaviour.

### C2 is not covered

The inverted extremum in the 1-ply lookahead (`max` over opponent replies where
the opponent minimises) has **no test**. Writing an honest one would require
asserting a chess-quality outcome, which this suite deliberately does not do —
and any behavioural proxy would be confounded by the CNN term, which outweighs
the lookahead penalty by roughly 600:1. It is better measured by the evaluation
harness than asserted here.

---

## Adding tests

- Reuse the board fixtures in `conftest.py` rather than inlining FENs.
- Keep unit tests model-free where possible; they run in milliseconds.
- If a test needs `engine_move`, budget ~600ms per call.
- If a test exposes a bug, **do not fix the bug to make it pass.** Add a
  characterisation test for current behaviour plus an `xfail(strict=True)` test
  for the intended contract, and record it in the table above.
