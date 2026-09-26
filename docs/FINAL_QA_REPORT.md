# Final QA Report — repository state at the end of C10

**Scope.** C10 was final correctness QA, test cleanup, repository hygiene and
documentation. It opened no research arm, retrained nothing, modified no dataset
or label, ran no new Stockfish experiment, and changed no evaluation methodology
or suite.

This document is the summary. The per-defect audit is
[`C10_FINAL_QA.md`](C10_FINAL_QA.md).

---

## 1. Baseline state entering C10

C9 closed the fusion branch with no production change. Verified by running the
suite rather than reading the C9 report:

```
$ python -m pytest -ra -q
903 passed, 10 xfailed, 74 warnings in 209.23s
```

0 failed, 0 skipped, 0 xpassed — matching the stated C9 baseline exactly.

All four production artifacts were byte-identical to the hashes C9 recorded
(§6). `git status` was dirty in five files, all pre-existing and unrelated to C10
(§8).

---

## 2. The C10 xfail audit

Ten `xfail(strict=True)` tests, all carrying the `deferred` marker. Every one was
classified against the production source and the measured C6–C9 evidence.

| Classification | Count | Which |
|---|---:|---|
| Genuine defect → **fixed** | **6** | C4 (`center` pre/post), C5 (colour asymmetry), board corruption ×2, API 415/malformed JSON ×2 |
| Genuine defect → **genuinely deferred**, documented | **1** | C3 (Ridge intercept) |
| Real limitation → **genuinely deferred**, documented | **3** | C6 side-to-move / castling / en-passant |
| Intentional behaviour → assert normally | 0 | — |
| Obsolete test or assumption | **0** | none turned out to be stale |

Two further defects with **no** xfail were found during the audit and fixed:

- **C10-A** — `POST /move` with a non-string `uci` (`{"uci": null}`, `123`,
  `true`) returned **HTTP 500**; a truthy non-dict body did too.
- **C10-B** — `explain_move` had the identical board-corruption defect as the two
  helpers the xfail named, but was not covered.

The full table, with the evidence behind each classification, is in
[`C10_FINAL_QA.md`](C10_FINAL_QA.md) §2.

---

## 3. Fixes made

### Production code — 4 changes, all with a correctness justification

| # | File | Change | Behavioural footprint |
|---|---|---|---|
| 1 | `engine.py` | New `_pushed` context manager; `move_impact`, `tactical_move_bonus` and `explain_move` use it instead of an unguarded `push` … `pop`. `chess.Board.push` appends to `move_stack` before it asserts pseudo-legality, so a rejected move left the caller's board mutated and skipped the `pop` | **None on any legal move.** Every production caller iterates `board.legal_moves`, so the path is unreachable in normal operation. Robustness only |
| 2 | `engine.py` | C4: the candidate dict's `center` now reports the post-move value (the one the scorer already consumed) instead of calling `center_control` after the `pop` | **No score, ordering or selected-move change.** Changes the recorded `center` field on 46/52 baseline positions |
| 3 | `engine.py` | C5: `opening_center_bonus` and `tactical_move_bonus` share one `CENTRAL_PAWN_PUSHES` constant covering both colours. Magnitude (0.3) unchanged; the list is exactly the union `tactical_move_bonus` already used, so that helper is unaffected | **0 selected-move changes, 0 ordering changes** on the 52-position baseline; 5 Black positions' top-3 scores move by exactly ±0.300 |
| 4 | `app.py` | `request.json or {}` → `request.get_json(silent=True)` with an `isinstance(dict)` guard and `str(... or "")` on `uci`. `request.json` raises before the handler's own validation | Error paths only. Non-JSON body 415+HTML → 400+JSON; malformed JSON 400+HTML → 400+JSON; non-string `uci` 500 → 400. **No gameplay change** |

**Justification standard applied:** each is a defect where the code contradicts
its own stated or sibling-established intent — not a tuning change. Nothing was
changed to improve a metric. Fixes 1 and 4 cannot affect move selection at all;
fixes 2 and 3 were measured before being accepted.

### Production artifacts — unchanged

`models/cnn_model.keras` and `models/weight_model.pkl` were **not** modified. No
model was retrained, refitted or rescaled.

### Tests

| File | Before | After | Net | What |
|---|---:|---:|---:|---|
| `tests/unit/test_heuristics.py` | 58 | 77 | **+19** | 9 board-corruption regression tests (3 helpers × 3 properties: FEN/depth/clock restored, a pre-existing stack preserved, success path unchanged); 14 C5 tests replacing 2 |
| `tests/api/test_errors.py` | 30 | 44 | **+14** | 2 contract tests, an 8-case body-shape matrix, a 6-case non-string-`uci` matrix (C10-A), a happy-path guard |
| `tests/unit/test_model_fusion.py` | 22 | 28 | **+6** | C3 evidence: intercept non-zero, `hybrid_score` off the selection path (asserted against module source), order-preservation over 4 positions |
| `tests/unit/test_board_encoding.py` | 23 | 27 | **+4** | C6 evidence: production CNN input shape pinned at `(None, 8, 8, 12)`; 3 characterisation tests pinning the current collapse |
| `tests/integration/test_engine.py` | 32 | 34 | **+2** | 4 C4 tests, including one that fails if `center` reverts to a per-position constant and one that checks scores remain reconstructible |
| `tests/api/test_app.py` | 37 | 37 | 0 | docstring only: removed a stale `app.py:253` reference |
| **total** | **202** | **247** | **+45** | matches the suite-level delta (913 → 958 collected) |

Counts are `pytest --collect-only` on each file, measured against the pre-C10
version of that file.

**No test was weakened, relaxed or deleted to reduce the xfail count.** Two
characterisation tests were **replaced** because the behaviour they characterised
was fixed — `test_opening_center_bonus_current_behaviour_is_white_only` and
`test_non_json_body_currently_returns_415_html`. Both replacements assert *more*
than the originals: the C5 replacement still pins the magnitude at exactly 0.3
(now for both colours) and adds six zero cases; the API replacement adds a
14-case input matrix.

The four retained xfails keep `strict=True` and their **assertions are
unchanged**. Only their `reason` strings were rewritten, from a stale
"DEFERRED (Phase 4)" to an accurate statement of why they are open.

---

## 4. Final test counts

```
$ python -m pytest -ra -q
954 passed, 4 xfailed, 74 warnings in 168.51s
```

| | entry | exit |
|---|---:|---:|
| **passed** | 903 | **954** |
| **failed** | 0 | **0** |
| **skipped** | 0 | **0** |
| **xfailed** | 10 | **4** |
| **xpassed** | 0 | **0** |
| collected | 913 | **958** |

**Zero unexpected failures.**

`skipped = 0` because Stockfish resolves on this machine, so the 3
`needs_stockfish` tests ran. On a machine without it they skip — expected, not a
failure.

### xpasses

**None in the final run.** One transient xpass occurred mid-audit: after the C5
production fix, `test_opening_center_bonus_should_be_colour_symmetric` XPASSed
and, being `strict=True`, failed the suite. That is precisely the mechanism the
strict markers exist for; it was resolved by removing the marker in the same
change, not by suppressing it.

---

## 5. Regression validation

`tests/integration/test_regression_baseline.py` replays all 52 positions from
`baseline/fens.json`. Every behavioural assertion passed **before** the baseline
was re-recorded, which is what made the re-record safe:

| check | result |
|---|---|
| legality | **100%** (52/52 legal moves) |
| selected moves vs the accepted post-C2 baseline | **0 changes** |
| top-3 ordering vs the accepted post-C2 baseline | **0 changes** |
| selected moves and ordering vs the **original Phase 0** capture | **0 changes** (historical continuity intact) |
| boards mutated by the call | **0** |
| determinism (two passes, same process) | **identical** moves and scores, 52/52 |
| top-3 scores | **5 positions moved**, by exactly ±0.300 — the C5 bonus magnitude |
| `material` / `space` / `mobility` / `cnn_cp` | **0 field diffs** |
| `explanation` | **0 diffs** |
| `ridge_coefficients`, `ridge_intercept` in the recorded file | **identical** |

### The re-record

`regression/engine_results_post_c2.json` was regenerated with the documented
command, never hand-edited:

```bash
python baseline/scripts/measure_engine.py \
    baseline/fens.json regression/engine_results_post_c2.json
```

This follows the precedent set by C1 and C2, which changed score values on 15 and
52 positions respectively without changing a single choice. The C10 footprint is
smaller: 5 score values and the `center` field, which no test asserts.

`baseline/` recorded evidence was left untouched — `engine_results.json`,
`fens.json` and `api_results.json` are all byte-identical (§6). The frozen Phase 0
capture is deliberately **not** re-recorded, including
`baseline/api_results.json`, which still records the pre-C10 415 response.

Rationale and the full diff are in
[`regression/README.md`](../regression/README.md).

> **Not a quality claim.** The regression suite is a change detector, not a
> correctness oracle. Nothing in C10 was evaluated against Stockfish, and no C10
> fix is claimed to make the engine play better. C4 and fix 1 cannot change a
> move at all; C5 changed no move on the 52 positions measured.

---

## 6. Production integrity

All four required artifacts are **byte-identical** to the hashes C9 recorded:

```
models/cnn_model.keras              972d81199a1355667fca554b06dd05b35fdec3c4dc789968ac4a1b0599d8dec3
models/weight_model.pkl             59a731127cc9b440c866c7a5d27ce39d905c9c5dadd8116d7f97a8542f18bcd0
evaluation/positions/phase0_52.json 6078d0e8db9e4124b984bbf3b5ad018de6b9c84b49361b01058ff58d875c0208
evaluation/positions/extended.json  9d5419cceec43bac98f9cdcf8109a4242e93791fa267f1df54fc1dddc41f929a
```

Frozen historical evidence, also unchanged:

```
baseline/engine_results.json        64be5a7f4047129338ae6de2a76c96ebf37b353e75990a87d4e295721f057031
baseline/fens.json                  ea52608148719280f8ab58afc08bc44406a2686f012d956cc36915b33a51e511
baseline/api_results.json           48ffd4eea9d582748ab4b36e59fb72aa443a0270e0ff64368677453b3caf2126
```

`git status` over `models/`, `evaluation/`, `training/artifacts/` and `config.py`
is **empty**.

### C8/C9 experiment artifacts remain gitignored

```
$ git check-ignore -v training/experiments/ training/artifacts/dataset_v2.test.jsonl
.gitignore:23:training/experiments/    training/experiments/
.gitignore:19:training/artifacts/*.jsonl    training/artifacts/dataset_v2.test.jsonl
```

Nothing under `training/experiments/` is staged or tracked.

### Repository hygiene

One orphaned, **empty** directory was removed: `C8a/seed_0/` at the repository
root. It was untracked, contained zero files, and no script references that path —
all real C8a artifacts live under `training/experiments/C8a/`. Removing it changes
nothing in git.

### Production changes, explicitly listed

| File | In production path? | Justification |
|---|---|---|
| `engine.py` | **yes** | Three correctness fixes (§3): exception-safe push/pop, C4 reporting, C5 colour symmetry |
| `app.py` | **yes** | JSON error-contract fix + C10-A (HTTP 500 on non-string `uci`) |
| `regression/engine_results_post_c2.json` | no — test expectations | Regenerated after C4 and C5; generated, never hand-edited |
| `baseline/scripts/measure_engine.py` | no — measurement tool | Its `ridge_intercept_note` cited "engine.py lines 131-132 and 162-163", which had drifted. Now names the two `w = weight_model.coef_` sites instead of line numbers. Recorded values are unaffected |
| `pyproject.toml` | no — test config | The `deferred` marker description said "a known defect that Phase 4 is expected to fix", which was no longer true |

Nothing else in production was touched.

---

## 7. Remaining known limitations

This repository is **not** claimed to be bug-free. These are the limitations known
at the end of C10.

### Open defects, deliberately retained

| ID | Defect | Why it is still open |
|---|---|---|
| **C3** | The Ridge `intercept_` (15.0772) is never applied at inference, so `hybrid_score` is not the Ridge's prediction and cannot be read as centipawns | An **interpretation** defect, not a ranking one. `hybrid_score` has no caller on the selection path, and on the path that is used the omission is an order-preserving per-position constant (0/52 ordering changes, 0/52 selected-move changes). Fixing it would shift every reported score by +15.0772 and force another re-record while changing no move played. And it would not actually make the score readable as centipawns: the Ridge was fitted on unrecoverable side-to-move-relative targets ([C9 audit §4](C9_RIDGE_AUDIT.md)). **Separately scoped** |
| **C6** ×3 | The board encoding carries piece placement only — no side to move, castling rights or en-passant state | Architecturally locked: the production CNN takes `(None, 8, 8, 12)`. And measured as harmful: **C6-A3 built exactly this encoding and regressed the engine +46.5 cp** mean regret on `extended` across all three paired seeds against a 19.7 cp noise band; C6-A13 showed castling rights alone reproduce most of it (+32.6 cp). Documented, **not scheduled** |

Both keep `xfail(strict=True)`, so if either is ever closed the suite fails and
forces the marker to be removed deliberately.

### Other known limitations, not defects C10 was scoped to fix

- **Single shared game state.** `app.py` keeps board, history and captures in
  module-level globals, so all clients share one game and requests are
  order-dependent.
- **`/engine_move` runs the engine twice per request** — once for Black's move,
  again to precompute White's top-3. Roughly doubles per-request latency.
- **`/forfeit` hardcodes `"0-1"`.** Harmless while the app only ever lets the
  engine play Black, which it enforces, but wrong if that changed.
- **Determinism is within-process only.** Verified across two passes in one
  process; cross-process, cross-machine and cross-TF/BLAS determinism is
  unverified.
- **`InconsistentVersionWarning`** — `weight_model.pkl` was pickled under
  scikit-learn 1.7.1 and is loaded under 1.8.0. Not suppressed in production;
  silenced only in test output.
- **The evaluation suites are small.** `phase0_52` is 52 positions and `extended`
  is 160, with category cells as small as n=5. All C6–C9 conclusions are
  descriptive, three-seed, with no significance testing.
- **The production Ridge is not reproducible.** Its original training labels were
  never persisted ([C9 audit §4](C9_RIDGE_AUDIT.md)), and it is partly in-sample.
  "Not beaten in C9" is not "optimal".
- **C6–C9 reports contain `engine.py` line references** that predate C10's edits
  and no longer resolve. The reports are historical records and were deliberately
  not rewritten; their described behaviour and results remain accurate.

---

## 8. Git status

Pre-existing dirty files recorded at the start of C10, **unrelated to C10 and left
untouched**:

```
tests/unit/test_training_representation16p.py
tests/unit/test_training_representation18.py
tests/unit/test_training_train_harness.py
training/a2_analysis.py
training/a2_perspective_probe.py
```

Verified rather than assumed: the three test files carry arm-roster maintenance
from the C6 A13R/A14 work (loosening exact-set assertions to subset assertions);
the two training scripts carry an argparse/CLI refactor. C10 did not modify any of
them.

C10's own changes:

| File | Category |
|---|---|
| `engine.py` | production fix |
| `app.py` | production fix |
| `regression/engine_results_post_c2.json` | regenerated expectations |
| `baseline/scripts/measure_engine.py` | stale-line-number fix in a generated note |
| `pyproject.toml` | stale marker description |
| `tests/unit/test_heuristics.py` | tests |
| `tests/unit/test_model_fusion.py` | tests |
| `tests/unit/test_board_encoding.py` | tests |
| `tests/integration/test_engine.py` | tests |
| `tests/api/test_errors.py` | tests |
| `tests/api/test_app.py` | stale line reference in a docstring |
| `README.md` | documentation |
| `docs/TESTING.md` | documentation |
| `docs/REPRODUCIBILITY.md` | documentation |
| `regression/README.md` | documentation |
| `docs/C10_FINAL_QA.md` | new |
| `docs/FINAL_QA_REPORT.md` | new |

That is 17 files. No experiment artifact is staged or tracked. **Nothing was committed.**

---

## 9. Documentation changes

Only factual inconsistencies discovered during C10 were fixed. **No historical
experiment report was rewritten**, and no result was restated to look better.

| File | Inconsistency found | Fix |
|---|---|---|
| `README.md` | Reranker step 5 still described the **pre-C2** lookahead: "take the max … and *subtract* a penalty". C2 replaced that with minimax plus a positive blend | Rewritten to the actual algorithm; the C1 bonus-sign step added; both attributed to the Phase 4A/4B reports |
| `README.md` | "Known API quirk: a non-JSON body returns 415 with an HTML body … not yet fixed" | Replaced with the post-C10 contract, stating what the behaviour was before and noting `baseline/api_results.json` is frozen pre-fix evidence |
| `README.md` | Opening-centre bonus documented as "e4, d4, or c4" | Now lists both colours' pushes |
| `README.md` | Project structure listed `docs/` as 2 files (it holds 30) and omitted `tests/`, `evaluation/`, `training/`, `regression/` | Updated |
| `README.md` | "Building a proper evaluation harness is **planned work**. Until it exists, this README does not report those numbers." The harness was built in Phase 3 and has been in use through C6-C9 | Points to [`EVALUATION.md`](EVALUATION.md) for the current numbers. **No numbers were imported into the README** - in particular no C6-C9 experimental metric is restated as a production result |
| `README.md` | The "Measured Performance" table is the **Phase 0** capture, presented without noting that C1, C2, C4 and C5 have landed since | Labelled as the historical Phase 0 reference point, with a pointer to the current harness |
| `docs/TESTING.md` | Test counts frozen at Phase 2: "282 full / 272 not-slow / 173 unit" | Now 958 / 948 / 833, with per-directory rows and a note on where the growth came from |
| `docs/TESTING.md` | Layout block omitted every test file added since Phase 2 | Updated |
| `docs/TESTING.md` | "Known expected failures: all 10" with the pre-C10 roster | Now 4, with the reason each is still open, plus a "Fixed in C10" table |
| `docs/TESTING.md` | Marker table: `deferred` = "a known defect a later phase is expected to fix" | Corrected |
| `docs/TESTING.md` | No record of the C10 re-record; "Adding tests" had no guidance for *closing* a deferred defect | Both added |
| `docs/REPRODUCIBILITY.md` | "Deliberately left alone by Phase 1" listed 8 defects as if current, including the 415 quirk | Converted to a table keeping the Phase 1 position **and** adding each item's status now. The historical statement is preserved, not overwritten |
| `regression/README.md` | No record of the C10 re-record | Added, with the full verification table |
| `baseline/scripts/measure_engine.py` | `ridge_intercept_note` cited line numbers that had drifted | Now names the two code sites instead |
| `baseline/scripts/measure_engine.py` | Its header comment claimed "engine.py loads its models via paths relative to the CURRENT WORKING DIRECTORY". **False since Phase 1** moved path resolution into `config.py`, which anchors to its own file | Corrected, with the Phase 0 history kept |
| `tests/api/test_app.py` | The `/forfeit` characterisation docstring cited `app.py:253`; the hardcode is at line 266 | Line number removed; the limitation and its containment stated instead |
| `engine.py`, `tests/api/test_errors.py` | Comments cited `app.py:109` and `app.py:149`, both drifted (and `app.py:109` drifted further from C10's own edit) | Line numbers removed; the handlers are named instead |

C6–C9 reports were **not** edited. Their "10 xfailed" counts, metrics and
`engine.py` line references are accurate records of the repository at the time
they were written, and remain reproducible from the artifacts they cite.

---

## 10. Final repository status

| Criterion | Status |
|---|---|
| Every xfail explicitly classified | **yes** — 10/10, in [`C10_FINAL_QA.md`](C10_FINAL_QA.md) §2 |
| Genuine defects have focused regression tests | **yes** — 6 xfails plus 2 newly found defects |
| No test weakened to remove an xfail | **yes** — 2 characterisation tests replaced with stronger assertions; retained xfails' assertions unchanged |
| Full pytest has zero failures | **yes** — 954 passed, 0 failed, 0 skipped, 4 xfailed, 0 xpassed |
| xpasses investigated | **yes** — one transient, accounted for in §4 |
| Regression validation passes | **yes** — legality 100%, 0 move changes, 0 ordering changes, historical continuity intact |
| Production artifacts verified | **yes** — all four byte-identical to C9's hashes |
| Historical experiment results intact | **yes** — no C6–C9 report edited, no artifact re-recorded |
| README/docs factually consistent | **yes** — 17 inconsistencies fixed, listed in §9 |
| Final QA report exists | this document |
| Git status contains only intended changes | **yes** — 17 C10 files plus 5 pre-existing unrelated (§8) |
| No new research/optimization work opened | **yes** |

### What this report does not claim

- **Not** that production is bug-free. Four defects are open and documented in
  §7, two of them in production code.
- **Not** that C10 made the engine stronger. No C10 fix was evaluated against
  Stockfish, and none is claimed to improve playing quality. Two of the four fixes
  cannot change a move at all.
- **Not** that the C6–C9 experimental results say anything about production. C9
  closed with no production change justified, and C10 did not revisit that.

**C10 is complete. Nothing was committed. C11 was not started.**
