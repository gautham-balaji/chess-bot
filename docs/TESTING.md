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
| `pytest` (full) | 958 | **~170s** |
| `pytest -m "not slow"` | 948 | ~120s |
| `pytest tests/unit` | 833 | ~50s |
| `pytest tests/integration` | 44 | ~35s |
| `pytest tests/api` | 81 | ~60s |

Counts are as of C10. Most of the growth since Phase 2 (282 tests) is the C6-C9
training/experiment harness under `tests/unit/test_training_*.py`, which is
model-free and fast.

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
│   ├── test_model_fusion.py            model loading, Ridge shape, position_metrics
│   ├── test_evaluation_metrics.py      evaluation/metrics.py definitions
│   ├── test_refit_ridge.py             training/refit_ridge.py
│   └── test_training_*.py              the C6-C9 harness: dataset builders, label
│                                       policies, representations, arms, C7 audit,
│                                       C8 dataset_v2, C9 Stage 1 / Stage 2
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
| `deferred` | Documents a known, **open** defect that is deliberately not fixed. Paired with `xfail(strict)`. Classified in [`C10_FINAL_QA.md`](C10_FINAL_QA.md) |

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

### Why it was re-recorded again in C10

Same shape as C1/C2, smaller footprint. Two C10 correctness fixes moved recorded
output:

- **C5** (colour-symmetric `opening_center_bonus`) changed top-3 **score values**
  on **5 Black positions**, by exactly ±0.300 each — the bonus magnitude.
- **C4** (candidate `center` reported post-move) changed the recorded **`center`
  field** on **46 of 52 positions**. No test asserts that field, so it did not
  cause the failure; it was re-recorded for accuracy.

Verified directly before re-recording: **0** selected-move changes, **0** top-3
ordering changes, **0** illegal moves, **0** boards mutated, max absolute score
delta **0.300**, and `material`/`space`/`mobility`/`cnn_cp`/explanations all
unchanged. `ridge_coefficients` and `ridge_intercept` in the file are identical,
so no model changed.

Again only the score expectations were stale, and again the score assertion was
**updated, not removed or weakened**. Details in
[`C10_FINAL_QA.md`](C10_FINAL_QA.md).

> The pre-C10 `center` values are worth one line: they were **identical for all
> three candidates** in a position, which is the visible signature of the C4
> defect — a per-position value masquerading as a per-candidate one.

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

**4 as of C10** (was 10 from Phase 2 to C9). All are `xfail(strict=True)`, meaning
**if the underlying defect is fixed, the test XPASSes and the suite fails.** That
is deliberate: it forces someone to come back here and remove the marker rather
than letting a stale "known bug" note linger after the bug is gone.

Every one is a **known, open engine defect** — not a test defect and not an
environment problem. Each was re-audited in C10 against the production source and
the measured C6-C9 evidence; the full classification table, including the six that
were fixed, is in [`C10_FINAL_QA.md`](C10_FINAL_QA.md).

| Test | Defect | Why it is still deferred |
|---|---|---|
| `test_hybrid_score_should_equal_the_ridge_prediction` | **C3** — the runtime uses `weight_model.coef_` only and drops `intercept_` (15.0772), so the score is not the Ridge model's prediction and cannot be read as centipawns | `hybrid_score` is **off the move-selection path** (0 callers in `engine.py`, `app.py`, `evaluation/`, `training/`), and on the path that *is* used the omission is an **order-preserving per-position constant** — 0/52 ordering changes, 0/52 selected-move changes. An interpretation defect, not a ranking defect. Scoped as a separate future change |
| `test_side_to_move_should_be_representable` | **C6** — no side-to-move plane in the encoding | The production CNN takes **(None, 8, 8, 12)**, so this needs a retrain — and **C6-A3 built exactly this encoding and regressed the engine +46.5 cp** mean regret on `extended` across all three paired seeds (19.7 cp noise band) |
| `test_castling_rights_should_be_representable` | C6 — castling rights not encoded | Same, plus **C6-A13** showed castling rights alone reproduce most of that regression (+32.6 cp) |
| `test_en_passant_should_be_representable` | C6 — en-passant state not encoded | Same |

### Fixed in C10

Six xfails were closed by production fixes, each with a focused regression test.
They no longer appear above:

| Defect | Fix | Covering tests |
|---|---|---|
| **C4** — candidate `center` was the *pre*-move value while `material`/`space`/`mobility` were post-move | `engine.py` reports the post-move value the scorer already used | `test_candidate_center_is_the_post_move_value`, `test_candidate_center_is_no_longer_the_pre_move_value`, `test_all_four_candidate_metrics_are_post_move_values`, `test_c4_fix_does_not_change_candidate_scores` |
| **C5** — `opening_center_bonus` matched only White's UCI strings | one shared `CENTRAL_PAWN_PUSHES` constant, both colours | `test_opening_center_bonus_is_colour_symmetric` + 13 parametrized cases |
| **board corruption** (×2) — `tactical_move_bonus` and `move_impact` left the caller's board mutated when `push` rejected the move | `engine._pushed` unwinds in a `finally` | `test_helpers_do_not_corrupt_board_when_move_is_illegal` and two siblings, × 3 helpers |
| **415 + HTML** (×2) — `POST /move` with a non-JSON or malformed body escaped the JSON error contract | `request.get_json(silent=True)` | `test_non_json_body_returns_400_json`, `test_malformed_json_body_returns_400_json`, `test_every_unusable_body_shape_returns_400_json` |

C10 also found and fixed two defects that had **no** xfail:
`explain_move` had the identical board-corruption bug, and a non-string `uci`
(e.g. `{"uci": null}`) returned **HTTP 500**. See
[`C10_FINAL_QA.md`](C10_FINAL_QA.md) §2.

### Paired characterisation tests

Each deferred test has a **passing** counterpart that pins today's actual
behaviour — `test_production_encoding_collapses_these_states` for the three C6
gaps, and `test_hybrid_score_currently_omits_the_ridge_intercept` for C3.

The pairing is deliberate. One test says *"this is what it does"*, the other says
*"this is what it should do"*. Together they make the defect unambiguous, and they
mean the suite never silently blesses a bug as intended behaviour — and never
silently loses the limitation either.

When a defect is fixed, its characterisation test is **replaced** by assertions on
the corrected behaviour, not deleted and not relaxed. C10 did this twice (C5 and
the API error contract); both replacements assert more than the originals did.


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
- If a test exposes a bug, **do not fix the bug to make it pass** in the same
  breath. Add a characterisation test for current behaviour plus an
  `xfail(strict=True)` test for the intended contract, and record it in the table
  above. Fixing it is a separate, deliberate decision.
- When a deferred defect **is** fixed: remove the `xfail`/`deferred` markers,
  **replace** the characterisation test with assertions on the corrected
  behaviour rather than deleting it, and add a focused regression test that would
  fail on the unfixed code. Never weaken an assertion to reduce the xfail count.
- Re-record `regression/engine_results_post_c2.json` only if the fix moved
  recorded output, and only after verifying exactly what moved. See
  [`regression/README.md`](../regression/README.md).
