# Regression baseline (current accepted engine state)

`engine_results_post_c2.json` holds the **live regression expectations** — what the
engine produces at the accepted post-C2 state, over the frozen 52-position suite in
`baseline/fens.json`.

It is **generated, never hand-edited**:

```bash
python baseline/scripts/measure_engine.py \
    baseline/fens.json regression/engine_results_post_c2.json
```

Consumed by `tests/integration/test_regression_baseline.py` via the
`current_engine_results` fixture.

---

## Why there are two baselines

| File | Role |
|---|---|
| `baseline/engine_results.json` | **ORIGINAL Phase 0 capture.** Frozen historical evidence. Never re-recorded |
| `regression/engine_results_post_c2.json` | **Current accepted expectations.** Re-recorded after C1 and C2 |

Keeping both means the historical Phase 0 behaviour and the current accepted
behaviour can always be told apart, rather than the original being overwritten.

## Why it was re-recorded

Two correctness fixes were accepted:

- **C1** (Phase 4A — Black-side heuristic bonus sign) changed top-3 **score values**
  on **15 Black positions**.
- **C2** (Phase 4B — 1-ply lookahead minimax) changed top-3 **score values** on
  **all 52 positions** (30 White, 22 Black).

**Neither fix changed a selected move, a top-3 ordering, legality, determinism or
board integrity.** Verified directly before re-recording: 0 move changes, 0 ordering
changes, 0 illegal moves, 52 score changes, all score deltas within ±1.0 —
consistent with the C1 bonus swing (≤1.3) and the C2 lookahead swing (≤1.0).

So only the score expectations were stale. Every behavioural assertion was kept, and
the score assertion was **updated rather than removed or weakened**.

> **C1 and C2 were correctness fixes with no measured change in playing quality.**
> Across 212 evaluated positions on two independent suites, neither altered top-1
> agreement, top-3 containment, mean/median/p95 regret, blunder rate, regret
> coverage, Spearman correlation or mate statuses. They made the engine *right*,
> not *stronger*. See `docs/PHASE_4A_REPORT.md` and `docs/PHASE_4B_REPORT.md`.

## Why it was re-recorded again in C10

Two C10 correctness fixes moved recorded output. Same shape as C1/C2: scores moved,
choices did not.

- **C5** (`opening_center_bonus` made colour-symmetric — it previously matched only
  White's UCI strings) changed top-3 **score values** on **5 Black positions**
  (`OP02`, `OP16`, `OP17`, `TC03`, `DF03`), by exactly **±0.300** each — the bonus
  magnitude, applied with the C1 sign convention.
- **C4** (candidate `center` now reports the *post*-move value, as
  `material`/`space`/`mobility` already did) changed the recorded **`center`
  field** on **46 of 52** positions. No test asserts that field, so this did not
  cause a failure; it was re-recorded for accuracy.

Verified directly before re-recording:

| check | result |
|---|---:|
| selected-move changes | **0** |
| top-3 ordering changes | **0** |
| illegal moves returned | **0** |
| boards mutated by the call | **0** |
| positions with a score change | 5 |
| max absolute score delta | **0.300** |
| `material` / `space` / `mobility` / `cnn_cp` field diffs | **0** |
| `explanation` diffs | **0** |
| `ridge_coefficients`, `ridge_intercept` | **identical** |
| legality / determinism / board-mutation summaries | **identical** |

So again only the score expectations were stale, and again the score assertion was
**updated rather than removed or weakened**. `baseline/` was left untouched
(`engine_results.json`, `fens.json` and `api_results.json` are all byte-identical).

> The pre-C10 `center` values were **identical for all three candidates** in a
> position — the visible signature of the C4 defect, a per-position value
> masquerading as a per-candidate one.

One other field changed: `ridge_intercept_note` previously cited "engine.py lines
131-132 and 162-163", which had drifted. The generator now names the two
`w = weight_model.coef_` sites instead of line numbers.

Full audit: [`docs/C10_FINAL_QA.md`](../docs/C10_FINAL_QA.md).

## Historical continuity is still asserted

`test_selected_moves_and_ordering_still_match_original_phase0` compares the current
engine against the **original Phase 0 capture** and requires the selected move and
top-3 ordering to be unchanged. The claim "neither fix altered a move" is therefore
continuously verified by the suite, not just recorded in a report.

`test_scores_are_expected_to_differ_from_original_phase0` pins the counterpart: all
52 positions *do* differ in score from the original capture. If that ever stops
being true, the reason for re-recording no longer holds and should be revisited.

## When to re-record again

Only when a change to engine output has been **reviewed and accepted**. Then:

1. Confirm the diff is what the accepted change should produce.
2. Regenerate with the command above — never edit the JSON by hand.
3. Record what changed and why, in that phase's report and the commit message.
4. Leave `baseline/` untouched.

**Never** revert the engine to satisfy this file, and never weaken an assertion to
make the suite green.
