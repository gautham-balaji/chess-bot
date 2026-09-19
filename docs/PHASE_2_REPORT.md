# Phase 2 Report — Testing Infrastructure and Regression Coverage

**Objective:** establish a meaningful automated pytest suite **before** changing
any known engine correctness issue.

**Result:** 214 tests. **203 pass, 11 xfail, 0 fail.** Full suite 196s.
No production code was modified. The Phase 0 regression check passes on all 52
positions, confirming engine behaviour is unchanged since commit `ccd24c4`.

---

## 1. What was added

| File | Purpose |
|---|---|
| `pyproject.toml` | pytest config: `pythonpath`, markers, `--strict-markers` |
| `requirements-dev.txt` | one dependency: `pytest==9.1.1` |
| `tests/conftest.py` | session-scoped module fixtures, 10 board fixtures, baseline loaders |
| `tests/unit/test_board_encoding.py` | `board_to_planes` |
| `tests/unit/test_heuristics.py` | material, space, centre, mobility, move_impact, bonuses |
| `tests/unit/test_explanations.py` | `explain_move` |
| `tests/unit/test_model_fusion.py` | model loading, Ridge shape, `cnn_evaluate`, `position_metrics` |
| `tests/integration/test_engine.py` | `rerank_moves`, `engine_move` |
| `tests/integration/test_regression_baseline.py` | replay of the frozen Phase 0 suite |
| `tests/api/test_app.py` | endpoint contracts and schemas |
| `tests/api/test_errors.py` | error paths and edge cases |
| `docs/TESTING.md` | how to run, what fails and why |

`pytest` was installed into the venv — the only environment change in this phase.

---

## 2. Test architecture

Three layers, matching the cost of what they exercise:

- **unit** (115 tests, ~21s) — pure functions. Mostly model-free; runs in the
  fast feedback loop.
- **integration** (32 tests) — CNN + Ridge + heuristics together via
  `rerank_moves` / `engine_move`, plus the baseline replay.
- **api** (67 tests) — Flask `test_client()`. No live server, no network.

**Cost control.** Importing TensorFlow costs 13–24s and `engine_move()` ~600ms.
So `engine_mod` / `app_mod` are **session-scoped** (models load once), the
baseline replay is **module-scoped** (52 positions replayed once, then asserted
against six times), and the replay is marked `slow` for deselection.

**Isolation.** `app.py` keeps the game in module-level globals, so the `client`
fixture resets state *before and after* every test. `test_errors.py` includes a
deliberate near-duplicate pair that only passes if that cleanup works — the
fixture is itself under test.

**The characterisation/contract pairing.** Where behaviour is wrong, two tests
are written: a passing one pinning what the code does today, and an
`xfail(strict=True)` one stating what it should do. One test alone would either
bless the bug or just fail noisily. `strict=True` means a Phase 4 fix makes the
xfail XPASS and turn the suite red, forcing the marker to be removed deliberately
rather than leaving a stale note behind.

---

## 3. Number of tests

| Layer | Collected |
|---|---:|
| unit | 115 |
| integration | 32 |
| api | 67 |
| **Total** | **214** |

Above the 20–30 / 8–15 / 15–25 guidance, driven by parametrisation (e.g. one
symmetry test runs over 4 FENs, one garbage-input test over 7 payloads) rather
than by padding with trivial cases.

---

## 4. Passing tests

**203 passing**, covering:

- **Encoding** — shape `(8,8,12)`, `float32`, binary values, piece counts equal
  active cells, per-channel distribution, White/Black channel separation,
  orientation swept across all 64 squares, distinctness, determinism, no mutation.
- **Heuristics** — material signs and values, king excluded, colour antisymmetry
  under `Board.mirror()` for material/space/centre, mobility invariance under
  mirror, centre bounded to [-4, 4], and **`mobility_score` fully restores FEN,
  turn, castling rights, en-passant square and move stack** across four position
  types including en-passant and castling.
- **Bonuses** — development, pawn-push penalty, captures, checks, promotions;
  board restoration after push/pop; determinism.
- **Explanations** — always a non-empty `list[str]`, never mutates, deterministic,
  tolerates a missing `cnn_cp` key, and the right reason fires for centre /
  development / capture / check / promotion / threshold.
- **Fusion** — CNN and Ridge load; **`rf`/`mlp`/`scaler` are asserted absent** so
  the Phase 1 cleanup cannot regress; Ridge has exactly 5 finite coefficients;
  model paths are absolute and repo-relative.
- **Engine** — legal move on every sampled position, result structure, candidate
  field schema, one entry per legal move, uniqueness, **White ranks descending and
  Black ascending**, `engine_move` returns `ranked[0]`, determinism, input board
  never mutated, checkmate/stalemate return `(None, [...], [])`, few-move and
  30+-move positions, in-check positions, both sides.
- **API** — all 10 endpoints; `/state` schema and initial position; `/move` legal
  moves, captures, promotions; `/engine_move` legality, schema, determinism;
  `/analyse` non-mutating; `/reset`; `/game_stats` schema, move counts, 8×8
  normalised saliency; `/forfeit`; `/model_info` structure **and that the Phase 1
  corrected figures (0.708, "10,000 positions") are served**; `/benchmark`
  blocks, graceful degradation when Stockfish is missing, and that **no
  `eval_diff_cp` / `speedup` / `time_saved_ms` field has returned**.
- **Errors** — illegal move, malformed UCI, missing field, empty/whitespace UCI,
  7 garbage payloads, out-of-turn, post-game-over on `/move` `/engine_move`
  `/analyse` `/benchmark`, 405, 404, Stockfish launch failure.

---

## 5. Expected / deferred failures

**11 `xfail(strict=True)`.** Every one is a **known engine or API defect** — none
is a test defect or an environment problem.

| # | Test | Defect | ID |
|---|---|---|---|
| 1 | `test_positive_bonus_must_not_worsen_a_move_for_the_side_to_move[black_to_move_board]` | Bonuses added with fixed positive sign while Black sorts ascending | **C1** |
| 2 | `test_hybrid_score_should_equal_the_ridge_prediction` | `intercept_` (15.0772) dropped at inference | **C3** |
| 3 | `test_candidate_center_should_be_the_post_move_value` | `center` is pre-move, siblings are post-move | **C4** |
| 4 | `test_opening_center_bonus_should_be_colour_symmetric` | White-only UCI strings | **C5** |
| 5 | `test_side_to_move_should_be_representable` | No side-to-move plane | **C6** |
| 6 | `test_castling_rights_should_be_representable` | Castling not encoded | C6 |
| 7 | `test_en_passant_should_be_representable` | En passant not encoded | C6 |
| 8 | `test_helpers_should_not_corrupt_board_when_move_is_illegal[tactical_move_bonus]` | Push-before-validate leaves board mutated on exception | **NEW** |
| 9 | `...[move_impact]` | same | **NEW** |
| 10 | `test_non_json_body_should_return_400_json` | 415 + HTML breaks the JSON error contract | — |
| 11 | `test_malformed_json_body_should_return_400_json` | same root cause | — |

### Why `xfail` here is justified

These are not being hidden. Each carries an explicit `reason=` naming the defect
and the source line, each is tagged `@pytest.mark.deferred` (`pytest -m deferred`
lists them), each is documented in `docs/TESTING.md`, and each is paired with a
**passing** characterisation test recording current behaviour. `strict=True`
guarantees the suite goes red the moment a fix lands, so none can go stale.

The alternative — leaving 11 hard failures — would make `pytest` red by default
and train everyone to ignore it.

---

## 6. Infrastructure failures

**None.** No environment, dependency or collection failures. `pytest` installed
cleanly and the suite runs end to end.

One environment note: `PYTHONIOENCODING=utf-8` is needed on a cp1252 Windows
console for some project scripts. It is **not** required for pytest itself.

---

## 7. Known bugs exposed by tests

### New — not previously documented

**Push-before-validate corrupts the caller's board.**
`tactical_move_bonus` and `move_impact` call `board.push(move)` before any
legality check. For an illegal move from an **empty** square, python-chess raises
`AssertionError` *after* the push, so `board.pop()` never executes and the
caller's board is left mutated (FEN changed, `move_stack` length 1).

Evidence, fresh board per case:

```
e2e5   legal=False  tactical_move_bonus -> 0.0                    fen_intact=True   stack=0
a3a4   legal=False  tactical_move_bonus -> RAISED AssertionError  fen_intact=False  stack=1
a3a4   legal=False  move_impact         -> RAISED AssertionError  fen_intact=False  stack=1
e2e4   legal=True   tactical_move_bonus -> 0.25                   fen_intact=True   stack=0
```

Not reachable from the application today — `rerank_moves` only ever passes legal
moves — so it is a latent contract violation, not a live bug. The fix is
`try/finally`, which is a production change and therefore Phase 4.

### Confirmed with stronger evidence

**C1, the Black-side bonus sign, is now demonstrated rather than asserted.**
The test applies one contract — *"a positive heuristic bonus must not make a move
rank worse for the side to move"* — parametrised over both colours. It **passes
for White and fails for Black.** The sort direction is derived from the engine's
own output (`ranked[0]["score"] < ranked[-1]["score"]`) rather than hardcoded, so
the test does not restate the implementation it is testing.

That symmetry is what makes this a defect in the engine rather than a quirk of
the test.

### Confirmed as previously documented

C3 (dropped intercept — `hybrid_score` equals `Ridge.predict` *minus* the
intercept, verified numerically), C4 (pre/post-move `center`), C5 (White-only
opening bonus), C6 (encoding gaps), and the 415 JSON-contract hole.

### Not covered: C2

The inverted extremum in the 1-ply lookahead has **no test**. An honest one would
have to assert a chess-quality outcome, which this suite deliberately avoids, and
any behavioural proxy is confounded: the CNN term is weighted 330.9 while the
lookahead penalty is bounded to ±0.5, roughly 600:1. Better measured by the
evaluation harness than asserted here. Recorded rather than faked.

---

## 8. Phase 0 regression results

`pytest tests/integration/test_regression_baseline.py` — **8 passed in 48.09s.**

| Check | Result |
|---|---|
| Legal move on all 52 positions | **52/52** |
| Legality rate matches baseline | **52/52 = 52/52** |
| Selected move matches baseline | **52/52, zero diffs** |
| Top-3 ordering matches baseline | **52/52, zero diffs** |
| Top-3 scores match baseline (±1e-3) | **52/52, zero diffs** |
| Input board never mutated | **52/52** |
| Within-process determinism (sample) | pass |
| Baseline suite intact (52 positions, 30W/22B, all valid and non-terminal) | pass |

**Engine behaviour is unchanged since commit `ccd24c4`.** Phase 2 added tests
only.

The file states in its own docstring that the baseline is a **change detector,
not a correctness oracle** — it records what the engine did on 2026-09-19, bugs
included, and Phase 4 is expected to change some of those outputs deliberately.

---

## 9. Full test runtime

| Selection | Tests | Time |
|---|---:|---:|
| `pytest` | 214 | **196.20s (3m16s)** |
| `pytest -m "not slow"` | 206 run, 8 deselected | 132.75s (2m12s) |
| `pytest tests/unit` | 115 | **20.80s** |

Slowest contributors:

```
48.13s  setup  test_regression_baseline (52-position replay, module-scoped)
14.02s  setup  test_app (import app -> TensorFlow + CNN)
12.40s  call   test_replay_is_deterministic_for_a_sample (16 engine_move calls)
 6.74s  call   test_engine_move_does_not_mutate_the_input_board
 6.41s  call   test_engine_move_returns_a_legal_move
```

The floor is structural: TensorFlow import (~13–24s once) plus ~600ms per
`engine_move`. The 21s unit loop is the practical development cadence.

---

## 10. Files changed

### Added
```
pyproject.toml
requirements-dev.txt
tests/conftest.py
tests/unit/test_board_encoding.py
tests/unit/test_heuristics.py
tests/unit/test_explanations.py
tests/unit/test_model_fusion.py
tests/integration/test_engine.py
tests/integration/test_regression_baseline.py
tests/api/test_app.py
tests/api/test_errors.py
docs/TESTING.md
docs/PHASE_2_REPORT.md
```

### Modified
**None.**

### Deleted
**None.**

### Environment
`pytest==9.1.1` installed into `venv/`.

Verified with `git diff --stat -- engine.py app.py config.py templates/ static/
models/ README.md requirements.txt runtime.txt .python-version` → empty.

---

## 11. Commands used

```bash
pip install pytest

pytest tests/unit/test_board_encoding.py -q
pytest tests/unit -q
pytest tests/integration/test_engine.py -q
pytest tests/integration/test_regression_baseline.py -q
pytest tests/api -q

pytest -q --durations=8          # full suite + timing
pytest -q -m "not slow"          # fast subset
pytest --collect-only -q         # counts
pytest --collect-only -q -m deferred

git diff --stat -- engine.py app.py config.py templates/ static/ models/ \
    README.md requirements.txt runtime.txt .python-version
git status --porcelain
```

---

## 12. What was intentionally NOT changed

No engine bug was fixed. Specifically **not** touched:

- the Black-side heuristic bonus sign (C1)
- the 1-ply lookahead extremum (C2)
- the dropped Ridge intercept (C3)
- the pre/post-move `center` inconsistency (C4)
- the White-only `opening_center_bonus` (C5)
- the side-to-move / castling / en-passant encoding gaps (C6)
- the newly found push-before-validate board corruption
- the 415 non-JSON response
- `/forfeit`'s hardcoded `"0-1"`
- `/engine_move` running the engine twice
- `/game_stats` returning 200 with `saliency: null` on failure

Also not done, per scope: no GitHub Actions (Phase 6), no Stockfish evaluation
harness (Phase 3), no model retraining, no architectural refactor, no frontend
changes, no TensorFlow/model-loading optimisation.

No testability refactor of production code proved necessary. Everything above was
reachable through existing public functions, `monkeypatch`, and Flask's
`test_client()`.
