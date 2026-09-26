# C10 — Final QA: the deferred-test audit

**Status: complete.** C10 is QA, test cleanup, repository hygiene and
documentation. No research arm was opened, no model was retrained, no dataset or
label was modified, no Stockfish experiment was run, and no evaluation
methodology or suite was changed.

Prior: [C9 audit](C9_RIDGE_AUDIT.md) · [C9 Stage 1](C9_STAGE1_REPORT.md) ·
[C9 Stage 2 design](C9_STAGE2_DESIGN.md) · [C9 Stage 2](C9_STAGE2_REPORT.md) ·
[C8a](C8A_REPORT.md) · [C8b](C8B_REPORT.md) · [final state](FINAL_QA_REPORT.md)

---

## 1. Entry state

Taken from an actual run, not from the C9 report:

```
$ python -m pytest -ra -q
903 passed, 10 xfailed, 74 warnings in 209.23s
```

0 failed, 0 skipped, 0 xpassed. All 10 xfails were `xfail(strict=True)` and
carried the `deferred` marker. The roster below is that run's output, not a
historical summary.

Production artifacts at entry, all byte-identical to the hashes C9 recorded:

```
models/cnn_model.keras              972d81199a1355667fca554b06dd05b35fdec3c4dc789968ac4a1b0599d8dec3
models/weight_model.pkl             59a731127cc9b440c866c7a5d27ce39d905c9c5dadd8116d7f97a8542f18bcd0
evaluation/positions/phase0_52.json 6078d0e8db9e4124b984bbf3b5ad018de6b9c84b49361b01058ff58d875c0208
evaluation/positions/extended.json  9d5419cceec43bac98f9cdcf8109a4242e93791fa267f1df54fc1dddc41f929a
```

---

## 2. The audit

Every xfail was classified against the production source, the surrounding tests,
and the measured C6–C9 evidence. **Six were fixed. Four remain open and are now
documented with evidence rather than with a stale "Phase 4 will fix it" note.**

| ID | Test | Current reason (pre-C10) | Classification | Action | Evidence |
|---|---|---|---|---|---|
| **C3** | `test_hybrid_score_should_equal_the_ridge_prediction` | `DEFERRED (Phase 4, C3)`: runtime applies `coef_` only, drops `intercept_` (15.0772), so `hybrid_score` is not the Ridge's prediction | **Genuine defect, but interpretation-only — genuinely deferred** | **Kept `xfail(strict)`; reason rewritten to state the measured scope. 3 new *passing* evidence tests added.** | `hybrid_score` has **0 callers** in `engine.py`, `app.py`, `evaluation/`, `baseline/scripts/`, `training/` — it is off the move-selection path. On the path that *is* used (`rerank_moves`), the omitted intercept is a per-position **constant**: applying it changes **0/52** top-3 orderings and **0/52** selected moves. So C3 cannot change a move the engine plays. §4.1 |
| **C4** | `test_candidate_center_should_be_the_post_move_value` | `DEFERRED (Phase 4, C4)`: `center` computed after `board.pop()`, so pre-move, while `material`/`space`/`mobility` are post-move | **Genuine defect** | **FIXED** (`engine.py`) | The reported `center` was **identical for all three top candidates** on 46/52 baseline positions — the signature of a per-position (pre-move) value. Now per-candidate. Reporting-only: the scorer already used the post-move value, and `evaluation/evaluate.py` never reads the field. **0** score, ordering or selected-move changes. §3.2 |
| **C5** | `test_opening_center_bonus_should_be_colour_symmetric` | `DEFERRED (Phase 4, C5)`: matches only White's UCI strings, so Black's mirrored pushes get 0 | **Genuine defect** | **FIXED** (`engine.py`) | The sibling helper `tactical_move_bonus` already listed **both** colours' pushes, which establishes the intended semantics as symmetric — the omission was in `opening_center_bonus`. `app.py` only ever lets the engine play Black, so the asymmetry was active in every game (the C1 argument). Measured: **0** selected-move changes, **0** ordering changes, 5 score changes of exactly ±0.300. §3.3 |
| **C6a** | `test_side_to_move_should_be_representable` | `DEFERRED (Phase 4)`: no side-to-move plane | **Genuinely deferred — architecturally locked and measured as harmful** | **Kept `xfail(strict)`; reason rewritten. 1 + 3 new *passing* tests added.** | `models/cnn_model.keras` input shape is **(None, 8, 8, 12)**, 2,360,129 params: a 13th plane needs a CNN retrain. **C6-A3 built exactly this encoding** (18 planes = side-to-move + castling + en-passant) and was the worst arm in the programme: Pearson 0.631–0.656 vs A2's 0.733–0.745 (no overlap), mean regret **+46.5 cp** on `extended` across all 3 paired seeds against a 19.7 cp noise band. §4.2 |
| **C6b** | `test_castling_rights_should_be_representable` | `DEFERRED (Phase 4)`: castling rights not encoded | same | same | Same, plus **C6-A13** (castling *only*, 16 planes) reproduced most of the regression on its own: **+32.6 cp**. §4.2 |
| **C6c** | `test_en_passant_should_be_representable` | `DEFERRED (Phase 4)`: en-passant not encoded | same | same | Same. §4.2 |
| **B1** | `test_helpers_should_not_corrupt_board_when_move_is_illegal[tactical_move_bonus]` | `DEFERRED (Phase 4, new)`: pushes before validating; raises after the push so the pop never runs | **Genuine defect** (stated mechanism was inaccurate) | **FIXED** (`engine.py`) | `chess.Board.push` appends to `move_stack` and `_stack` **at its top**, then asserts pseudo-legality — so the exception fires *inside* push, with the board already mutated (halfmove clock 0→1, one bogus frame). Because the frame exists, unwinding in a `finally` fully restores the board. Verified directly. §3.1 |
| **B2** | `…[move_impact]` | same | **Genuine defect** | **FIXED** (`engine.py`) | Same. §3.1 |
| **A1** | `test_non_json_body_should_return_400_json` | `DEFERRED (Phase 4)`: `request.json` raises `UnsupportedMediaType` → 415 + HTML | **Genuine defect** | **FIXED** (`app.py`) | `/move` was the only endpoint not honouring the "errors are 400 + JSON" contract every other path returns. `get_json(silent=True)` returns `None` instead of raising, so a bad body falls through to the handler's own rejection. §3.4 |
| **A2** | `test_malformed_json_body_should_return_400_json` | `DEFERRED (Phase 4)`: same root cause | **Genuine defect** | **FIXED** (`app.py`) | Same. §3.4 |

### Nothing was reclassified as obsolete

No xfail turned out to be testing a stale assumption, and none was removed
because it had become moot. Every one of the ten either got a production fix with
a focused regression test, or kept its `xfail(strict=True)` marker with its
assertion **unchanged**.

### Two additional defects were found during the audit

| ID | Defect | Status |
|---|---|---|
| **C10-A** | `POST /move` with a non-string `uci` — `{"uci": null}`, `{"uci": 123}`, `{"uci": true}` — reached `.strip()` on a non-string and raised `AttributeError`, i.e. **HTTP 500**. A truthy non-dict body (`[1,2,3]`) reached `.get` on a list, also 500. Not previously recorded. | **FIXED** in the same `app.py` change, with a 6-case parametrized regression test. Confirmed against the pre-C10 expression. |
| **C10-B** | `explain_move` had the **identical** unguarded push/pop as `tactical_move_bonus` and `move_impact`, but was not covered by the xfail. Confirmed by replaying the old push/pop shape: board mutated, one frame left behind. | **FIXED** with the same `_pushed` helper; the regression test now covers all three helpers. |

---

## 3. The fixes

### 3.1 Exception-safe push/pop — `engine.py`

One new helper, used by the three affected functions:

```python
@contextmanager
def _pushed(board, move):
    depth = len(board.move_stack)
    try:
        board.push(move)
        yield board
    finally:
        while len(board.move_stack) > depth:
            board.pop()
```

Unwinding to the **recorded depth** rather than calling `pop()` once is
deliberate: if `push` ever fails *before* appending a frame, a bare `pop()` would
raise `IndexError` and mask the original exception.

Regression tests: 9 in `tests/unit/test_heuristics.py` — 3 helpers × 3 properties.
The board is restored on the failure path (FEN, stack depth **and** halfmove
clock); a **pre-existing** move stack is left intact rather than emptied; and the
success path is unchanged.

**Reachability:** every production caller iterates `board.legal_moves`, so this
path is not reachable in normal operation. It is a robustness fix, not a
behaviour fix, and it changes no output for any legal move.

### 3.2 C4 — `center` reported pre-move (`engine.py`)

```diff
-            "center": center_control(board),   # runs AFTER board.pop() -> pre-move
+            "center": center,                  # the post-move value already scored
```

The local `center` was computed inside the push and is the value the weighted
score already consumed, so **scores are unaffected**. 4 tests, including one that
fails if the field reverts to a per-position constant, and one that asserts all
four positional fields now agree on the same board.

### 3.3 C5 — colour-asymmetric opening bonus (`engine.py`)

The two helpers now share one constant, so they cannot drift apart again:

```python
CENTRAL_PAWN_PUSHES = ("e2e4", "d2d4", "c2c4", "e7e5", "d7d5", "c7c5")
```

The magnitude (0.3) is unchanged and the list is exactly the union
`tactical_move_bonus` already used, so **`tactical_move_bonus` is unaffected**.
14 tests, pinning 0.3 for all six pushes, 0 for six non-central moves, and the
shared list.

The old characterisation test `test_opening_center_bonus_current_behaviour_is_white_only`
asserted the asymmetry as fact. It was **replaced, not relaxed** — the magnitude
is still pinned exactly, now for both colours.

### 3.4 API error contract (`app.py`)

```diff
-    data = request.json or {}
-    uci  = data.get("uci", "").strip()
+    data = request.get_json(silent=True)
+    if not isinstance(data, dict):
+        data = {}
+    uci  = str(data.get("uci") or "").strip()
```

`request.json` raises before the `or {}` guard can apply. `get_json(silent=True)`
returns `None`, so an unusable body falls through to the same rejection as any
other unusable input. The `isinstance` and `str(...)` guards close **C10-A**.

The characterisation test that asserted 415 + HTML was **replaced** by the
intended contract, plus an 8-case body-shape matrix and a 6-case non-string-`uci`
matrix. 17 test cases added against 3 removed, so
`tests/api/test_errors.py` goes 30 → 44.

---

## 4. The deferrals

### 4.1 C3 — the Ridge intercept stays open, and stays scoped out

The brief's C3 rule was followed: production scoring was **not** changed. The
decision is that C3 **does not belong in C10**, on measured grounds:

1. `hybrid_score` — the function the xfail is written against — has **no caller**
   in `engine.py`, `app.py`, `evaluation/`, `baseline/scripts/` or `training/`.
   It is not on the move-selection path. Pinned by
   `test_hybrid_score_is_not_on_the_move_selection_path`, which asserts against
   the module source.
2. On the path that *is* used, the omission is an **order-preserving per-position
   constant**: 0/52 ordering changes, 0/52 selected-move changes. Pinned by
   `test_omitting_the_intercept_cannot_change_the_ranking` over 4 positions.
3. So C3 is an **interpretation** defect — `hybrid_score` is not the Ridge's
   prediction and cannot be read as centipawns — **not a ranking defect**.
4. Fixing it would shift every reported score by +15.0772 and require another
   regression re-record, while changing **no move the engine plays**.
5. Every C6–C9 arm was evaluated under coef-only fusion
   ([C9 audit §1](C9_RIDGE_AUDIT.md)), and the Ridge was fitted on unrecoverable
   side-to-move-relative targets ([C9 audit §4](C9_RIDGE_AUDIT.md)) — so
   "readable as centipawns" would **not** become true merely by adding the
   intercept back. That is the substantive reason this is not a one-line fix.

**C3 is therefore a separately scoped future change**, needing its own
justification, its own regression re-record, and honest treatment of point 5.

### 4.2 C6 — the representation gap stays open, on evidence

Reclassified from "deferred, pending Phase 4" to **"open, evidence-based"**. Two
independent grounds, both pinned by tests:

1. **Architecturally locked.** Input shape **(None, 8, 8, 12)**. Asserted by
   `test_production_cnn_input_is_twelve_planes`, so a future model swap forces
   these three deferrals to be re-decided rather than silently inherited.
2. **Measured as harmful.** [C6-A3](C6_A3_REPORT.md) built precisely the
   18-plane encoding these tests demand and regressed the engine by **+46.5 cp**
   mean regret on `extended`, consistently across all three paired seeds, against
   a 19.7 cp noise band — the first arm whose degradation clearly exceeded the
   noise band. [C6-A13](C6_A13_REPORT.md) showed castling rights **alone**
   reproduce most of it (+32.6 cp). C6-A3 also found an 18-plane model is not
   loadable by the shipped engine at all.

The three assertions are **unchanged** and `strict=True` is retained, so if the
representation is ever extended these XPASS and fail the suite. Three new
characterisation tests pin the current collapse as fact, so the limitation cannot
be lost silently either.

`tests/unit/test_training_representation18.py` covers the 18-plane research
encoder, which does distinguish all three states — the limitation is
production's, not the repository's.

---

## 5. Exit state

```
$ python -m pytest -ra -q
954 passed, 4 xfailed, 74 warnings in 168.51s
```

| | entry | exit |
|---|---:|---:|
| passed | 903 | **954** |
| failed | 0 | **0** |
| skipped | 0 | **0** |
| xfailed | 10 | **4** |
| xpassed | 0 | **0** |
| collected | 913 | **958** |

**+45 test cases. 6 xfails closed by production fixes, 4 retained deliberately.**

No test was weakened, relaxed or deleted to reduce the xfail count. Two
characterisation tests were **replaced** because the behaviour they characterised
was fixed; both replacements assert *more* than the originals.

### xpasses

**None in the final run.** One transient xpass occurred mid-audit and is
accounted for: after the C5 production fix,
`test_opening_center_bonus_should_be_colour_symmetric` XPASSed and, being
`strict=True`, failed the suite — which is exactly the mechanism the strict
markers exist for. It was resolved by removing the marker in the same change.

`skipped = 0` because Stockfish resolves on this machine, so the 3
`needs_stockfish` tests ran. On a machine without it they skip; that is expected
and is not a failure.

---

## 6. Reproduction

```bash
# full suite
python -m pytest -ra -q

# just the deferred roster and its reasons
python -m pytest -m deferred -v

# the C10 regression tests
python -m pytest tests/unit/test_heuristics.py tests/unit/test_model_fusion.py \
                 tests/unit/test_board_encoding.py tests/api/test_errors.py \
                 tests/integration/test_engine.py -q

# the 52-position regression replay
python -m pytest tests/integration/test_regression_baseline.py -q
```

See [FINAL_QA_REPORT.md](FINAL_QA_REPORT.md) for the regression re-record, the
production-integrity check and the full list of changed files.
