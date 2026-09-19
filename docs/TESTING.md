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
currently in the project is the recorded engine baseline, and even that is used
here only as a *change detector* — see below.

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

# just the regression baseline check
pytest tests/integration/test_regression_baseline.py

# see which deferred tests exist and why
pytest -m deferred -v
```

On Windows, prefix with `PYTHONIOENCODING=utf-8` if your console is cp1252.

### Expected runtime

| Selection | Tests | Time |
|---|---:|---:|
| `pytest` (full) | 282 | **~120s** |
| `pytest -m "not slow"` | 272 | ~60s |
| `pytest tests/unit` | 173 | ~20s |

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
│   └── test_regression_baseline.py     replay of the 52-position suite      [slow]
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
| `deferred` | Documents a known defect a later phase is expected to fix |

### Cost and isolation

Two constraints shape `conftest.py`:

- **`engine_mod` and `app_mod` are session-scoped.** TensorFlow and the CNN load
  once per pytest session rather than once per test.
- **The `client` fixture resets game state before *and* after each test.**
  `app.py` keeps the board in module-level globals, so without this the API tests
  would leak into each other and become order-dependent. `test_errors.py` contains
  a deliberate pair of near-duplicate tests that only pass if that cleanup works.

---

## How the regression baseline is used

`tests/integration/test_regression_baseline.py` replays all 52 positions from
`baseline/fens.json` and checks legality, selected move, top-3 ordering, top-3
scores, board integrity and determinism.

**It is a change detector, not a correctness oracle.** The recorded files say what
the engine *does*, remaining defects included — not that those moves are good.

### Two baselines, on purpose

| File | Role |
|---|---|
| `baseline/engine_results.json` | **ORIGINAL Phase 0 capture** (commit `ccd24c4`, 2026-09-19). Frozen historical evidence, never re-recorded |
| `regression/engine_results_post_c2.json` | **Current accepted expectations**, re-recorded after C1 and C2 |

The live assertions compare against the current file; two dedicated tests keep the
link to the original so the two states can always be told apart.

### Why it was re-recorded after C2

- **C1** (Phase 4A) changed top-3 **score values** on **15 Black positions**.
- **C2** (Phase 4B) changed top-3 **score values** on **all 52 positions**.
- **Neither changed a selected move, a top-3 ordering, legality, determinism or
  board integrity** — verified directly before re-recording (0 move changes,
  0 ordering changes, 0 illegal moves, all score deltas within ±1.0).

So only the score expectations were stale. Every behavioural assertion was kept and
the score assertion was **updated, not removed or weakened**.

> **Both were correctness fixes with no measured change in playing quality.** Across
> 212 positions on two independent suites, neither altered top-1 agreement, top-3
> containment, regret, blunder rate, Spearman or mate statuses. They made the engine
> *right*, not *stronger*.

`test_selected_moves_and_ordering_still_match_original_phase0` asserts that against
the original capture continuously, and
`test_scores_are_expected_to_differ_from_original_phase0` pins the counterpart.

### Re-recording

Only after a change to engine output has been reviewed and **accepted**:

```bash
python baseline/scripts/measure_engine.py \
    baseline/fens.json regression/engine_results_post_c2.json
```

Generated, never hand-edited; `baseline/` stays untouched. Full rules in
[`regression/README.md`](../regression/README.md).

> **Never revert the engine to make this file green, and never weaken an
> assertion.** A failure here means output moved — decide whether that was intended.

---

## Known expected failures

All 10 are `xfail(strict=True)`, meaning **if the underlying bug is fixed, the
test XPASSes and the suite fails.** That is deliberate: it forces someone to come
back here and remove the marker rather than letting a stale "known bug" note
linger after the bug is gone.

Every one of these is a **known engine/API defect**, not a test defect and not an
environment problem.

| Test | Defect |
|---|---|
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

### C1 and C2 are fixed, not deferred

Both were accepted in Phase 4A and Phase 4B and are now covered by **passing**
tests, so neither appears in the table above:

| Defect | Covering tests |
|---|---|
| **C1** — heuristic bonus sign inverted for Black | `test_positive_bonus_must_not_worsen_a_move_for_the_side_to_move` (both colours), `test_bonus_magnitude_is_preserved_and_only_the_sign_changes` |
| **C2** — 1-ply lookahead minimax | `test_lookahead_uses_the_opponents_best_reply` (4 cases), `test_lookahead_term_matches_the_minimax_value_exactly` (2 cases) |

Phase 2 recorded that C2 could not be honestly tested without asserting a
chess-quality outcome. Phase 4B found a way that does not: drive `rerank_moves`
with a **stubbed CNN** and measure the applied lookahead term by differencing two
runs that are identical except for the reply values, so the weighted term and the
bonuses cancel exactly. The expected values come from the minimax contract and
differ *in sign* from the pre-fix output — confirmed by running them against the
unfixed engine, where 5 of 6 failed.

Phase 4B also found the original C2 description was incomplete: the extremum was
wrong for White only, and a second, separate sign inversion affected both sides.

---

## Adding tests

- Reuse the board fixtures in `conftest.py` rather than inlining FENs.
- Keep unit tests model-free where possible; they run in milliseconds.
- If a test needs `engine_move`, budget ~600ms per call.
- If a test exposes a bug, **do not fix the bug to make it pass.** Add a
  characterisation test for current behaviour plus an `xfail(strict=True)` test
  for the intended contract, and record it in the table above.
