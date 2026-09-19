# Phase 4A Report — C1: Black-side heuristic bonus sign

**Scope:** one defect only (C1). No other engine issue touched, no model retrained,
no evaluation methodology changed.

**Starting point:** commit `0f7ad6a` (Phase 3), clean tree.

**Headline:** the fix is correct and verified, White is provably unchanged — and it
changes **zero engine moves** on all 212 evaluated positions. Every quality metric
is identical before and after. The reason is measurable and is the real finding of
this phase.

---

## A. Exact root cause

Two pieces of `rerank_moves` in `engine.py` disagreed about what "better" means.

**1. Bonuses were added with a fixed positive sign** (pre-fix, `engine.py:172-175`):

```python
score += development_bonus(board, mv)
score += pawn_push_penalty(board, mv)
score += opening_center_bonus(board, mv)
score += tactical_move_bonus(board, mv)
```

**2. The final sort direction flips by side** (`engine.py:208`, unchanged):

```python
move_scores.sort(key=lambda x: x["score"], reverse=(board.turn == chess.WHITE))
```

The weighted score is on a **White-positive scale** (the Ridge model was fitted
against White-perspective Stockfish targets), so:

| Side to move | Sort | "Better" means | Effect of `+bonus` |
|---|---|---|---|
| White | `reverse=True` (descending) | higher score | **correct** — improves the move |
| Black | `reverse=False` (ascending) | lower score | **inverted** — pushes the move *down* Black's own preference list |

So for Black, a bonus intended to reward developing a knight, capturing, checking
or promoting made the move rank **worse**. Since `app.py:149` only ever lets the
engine play Black, the defect was active in every game the application played.

---

## B. Exact code change

One hunk in `engine.py`. No other production file touched.

```python
bonus_sign = 1.0 if board.turn == chess.WHITE else -1.0
score += bonus_sign * development_bonus(board, mv)
score += bonus_sign * pawn_push_penalty(board, mv)
score += bonus_sign * opening_center_bonus(board, mv)
score += bonus_sign * tactical_move_bonus(board, mv)
```

`board` is back at the original position at this point (`board.pop()` on line 170),
so `board.turn` is the side to move.

**Why this form rather than summing then negating:** multiplying by `1.0` is exact
in IEEE 754 and the *sequence of additions is unchanged*, so White's scores are
bit-identical. Summing the four terms first and then adding would have changed the
floating-point accumulation order for White — a gratuitous risk in a controlled
experiment. This was verified empirically (§G).

Explicitly **not** changed: the sort direction, the 1-ply lookahead subtraction
(C2 — also sign-suspect, deliberately left alone), the Ridge intercept (C3), the
`center` field (C4), `opening_center_bonus`'s White-only matching (C5), the board
encoding (C6), candidate generation, the CNN, model weights, the API, and the
evaluation harness methodology.

---

## C. Tests added / changed

`tests/integration/test_engine.py` only.

**Changed — the C1 strict xfail is now a real passing test.** The `xfail(strict=True)`
and `deferred` markers were removed from the Black parametrisation, so the same
contract is now asserted and *passed* for both colours.

The test's measurement method also had to change. It previously reconstructed a
"score without bonus" as `score - bonus`, which silently assumed the bonus is
always *added* — an assumption the fix invalidates. It now measures the **applied**
bonus directly:

```python
with_bonus    = rerank_moves(board)                     # real helpers
raw           = {uci: bonus_sum(...)}                   # captured BEFORE patching
monkeypatch   -> the four bonus helpers return 0.0
without_bonus = rerank_moves(board)                     # stubbed helpers
applied       = with_bonus[uci] - without_bonus[uci]    # the signed contribution
```

Everything else (weighted term, 1-ply lookahead) is deterministic and identical
between the two calls, so the difference *is* the signed bonus contribution. No
engine arithmetic is re-derived in the test, and the check is not circular.

**Added — 2 tests:**

| Test | Purpose |
|---|---|
| `test_bonus_magnitude_is_preserved_and_only_the_sign_changes[white/black]` | The fix must flip the sign, not rescale. Guards against a future "fix" that clamps, halves or drops the bonuses. The White parametrisation also pins White's behaviour as unchanged |

Net: `test_engine.py` went from 24 to 24 test functions (one xfail pair became a
passing pair; one added parametrised pair; one redundant draft removed).

> **One test defect of my own, found and fixed:** the first version called
> `bonus_sum` *after* monkeypatching, so it read the stubs and compared against
> zero. Corrected by capturing raw bonuses before patching.

---

## D. Full pytest result

```
1 failed, 263 passed, 10 xfailed in 262.25s (4m22s)
```

**The single failure is `test_top3_scores_match_baseline`, and it is expected.**
It was not weakened, skipped or deleted. See §H.

xfail count dropped 11 → 10: the C1 entry is gone, the other nine (C3, C4, C5, C6
×3, the two push-before-validate cases, and the two 415 JSON-contract cases) remain
untouched.

Focused run first, as required:

```
pytest tests/integration/test_engine.py -k "bonus or ranking or descending or ascending"
  -> 6 passed, 20 deselected in 25.00s
```

---

## E. Phase 0 suite (52 positions) — before / after

| Metric | pre-C1 | post-C1 | Change |
|---|---:|---:|---|
| legality rate | 100.0% | 100.0% | none |
| top-1 agreement | 19.23% (10/52) | 19.23% (10/52) | none |
| top-3 containment | 32.69% (17/52) | 32.69% (17/52) | none |
| mean regret | 107.51 cp | 107.51 cp | none |
| median regret | 45 cp | 45 cp | none |
| p95 regret | 419 cp | 419 cp | none |
| max regret | 614 cp | 614 cp | none |
| blunder rate (>300cp) | 15.56% (7/45) | 15.56% (7/45) | none |
| regret coverage | 86.54% | 86.54% | none |
| Spearman mean / median | 0.24 / 0.37 | 0.24 / 0.37 | none |
| `missed_forced_mate` | 4 | 4 | none |
| `engine_move_allows_forced_mate` | 3 | 3 | none |
| **engine moves changed** | — | **0 / 52** | — |

Methodology unchanged: Stockfish 17.1, depth 8, Threads=1, Hash=16 MB, Clear Hash
per position, same `evaluation/metrics.py`.

---

## F. Extended suite (160 positions) — before / after

| Metric | pre-C1 | post-C1 | Change |
|---|---:|---:|---|
| legality rate | 100.0% | 100.0% | none |
| top-1 agreement | 18.12% (29/160) | 18.12% (29/160) | none |
| top-3 containment | 33.75% (54/160) | 33.75% (54/160) | none |
| mean regret | 144.07 cp | 144.07 cp | none |
| median regret | 25 cp | 25 cp | none |
| p95 regret | 533 cp | 533 cp | none |
| blunder rate (>300cp) | 24.34% (37/152) | 24.34% (37/152) | none |
| regret coverage | 95.0% | 95.0% | none |
| Spearman mean / median | 0.22 / 0.30 | 0.22 / 0.30 | none |
| `missed_forced_mate` | 4 | 4 | none |
| `engine_move_allows_forced_mate` | 4 | 4 | none |
| **engine moves changed** | — | **0 / 160** | — |

---

## G. White vs Black

### White is provably unchanged

| Evidence | Result |
|---|---|
| Regression suite: positions with changed top-3 scores | **15, all Black. Zero White.** |
| Evaluation, phase0_52 | 0 / 30 White moves changed |
| Evaluation, extended | 0 / 85 White moves changed |
| White metrics, phase0_52 | top-1 26.67%, mean regret 122.52, median 29 — identical |
| White metrics, extended | top-1 17.65%, mean regret 154.64, median 19 — identical |
| Unit test | `test_bonus_magnitude_is_preserved_and_only_the_sign_changes[white_to_move_board-False]` passes: applied bonus == raw bonus |

This is by construction: `1.0 * x` is exact in IEEE 754 and the addition order is
unchanged. **No White behaviour changed, and no explanation is needed for a change
that did not occur.**

### Black changed in score, not in choice

| Evidence | Result |
|---|---|
| Black positions with changed top-3 **scores** | **15 / 22** (phase0_52) |
| Black positions with changed top-3 **ordering** | **0** (`test_top3_ordering_matches_baseline` passed) |
| Black positions with changed **selected move** | **0 / 22** and **0 / 75** |
| Black metrics, phase0_52 | top-1 9.09%, mean regret 85.0, median 46 — identical |
| Black metrics, extended | top-1 18.67%, mean regret 132.01, median 29 — identical |

The 7 Black positions with no score change are those where no bonus applied to any
top-3 move (quiet endgames — no development, capture, check, promotion or centre
push available).

---

## H. Per-position impact summary

**0 of 212 engine moves changed.** Score values changed on 15 Black positions; no
ranking changed anywhere.

### Why — measured, not asserted

Across all **97 Black positions** in the two suites:

| Quantity | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| rank-1 → rank-2 score gap | 0.220 | **40.978** | 76.008 | 490.205 |
| Max bonus *swing* (2 × bonus) | 0.000 | **0.500** | 0.492 | 1.300 |

The sign flip moves a score by `2 × bonus` (from `+b` to `−b`). The median gap
between the best and second-best move is **82× larger** than the median swing.

**Only 1 of 97 Black positions has a rank-1/rank-2 gap small enough for the swing
to reorder the top two at all** — and in that position it did not.

This is the same magnitude problem documented since the Phase 0 audit: the Ridge
model weights the CNN term at **330.9** and material at **32.4**, while the
heuristic bonuses are 0.1–0.65. The bonuses were never capable of deciding a move.
C1 was a real inversion of a term that contributes well under 1% of the score.

### Historical baseline assertions that changed

One assertion, `test_top3_scores_match_baseline`, on 15 positions:

```
OP02 OP07 OP10 OP15 OP16 OP17 OP18
MG09 MG10 MG11 MG12
EG11 TC03 TC06 DF03
```

All 15 are **Black to move**. The shift is exactly the doubled bonus, e.g.
`OP02: 55.9358 → 55.4358` (−0.500), `OP16: 12.6342 → 12.1342` (−0.500).

| Baseline assertion | Result | Why |
|---|---|---|
| `test_selected_moves_match_baseline` | **PASS** | No move changed |
| `test_top3_ordering_matches_baseline` | **PASS** | No ordering changed |
| `test_top3_scores_match_baseline` | **FAIL** | Black scores shifted by the intended bonus sign flip |
| `test_every_baseline_position_still_yields_a_legal_move` | PASS | 52/52 |
| `test_legality_rate_matches_baseline` | PASS | 52/52 |
| `test_no_baseline_position_mutates_its_board` | PASS | 52/52 |
| `test_replay_is_deterministic_for_a_sample` | PASS | — |
| `test_baseline_suite_is_intact` | PASS | — |

**This change is expected from C1 and is the intended consequence.** The test was
not weakened, skipped or deleted. Whether to re-record the baseline is a review
decision — see §L.

---

## I. Unexpected behaviour

**1. The fix changed no moves at all.** I expected a small number of Black
reorderings on 212 positions. Zero occurred. Quantified in §H: the bonuses are
~82× too small relative to typical score gaps. Worth stating plainly, because the
naive reading of "we fixed a bug affecting every game the engine plays" is that
play should improve — and on this evidence, it does not measurably.

**2. Top-3 *scores* changed while top-3 *ordering* did not.** The regression suite
separates these, so the diff landed precisely on the one assertion that should
have caught it and on none of the others. That separation was worth having.

**3. A test defect of mine.** The first version of the rewritten C1 test called
`bonus_sum` after monkeypatching the helpers, so it compared against zero and five
tests failed. The engine fix was fine; the test was wrong. Fixed by capturing raw
bonuses first.

**Not investigated (out of scope):** latency figures differ between runs
(p50 616–779 ms here). Phase 3 established latency is host-dependent and does not
reproduce; it is excluded from all comparisons above.

---

## J. Files changed

### Modified (2)
```
engine.py                        +21 -4    the C1 fix (one hunk)
tests/integration/test_engine.py +126 -45  xfail -> passing, + 2 tests
```

### Added (5)
```
evaluation/compare_runs.py                    before/after comparison tool
evaluation/results/phase0_52_postC1.json/.md  post-C1 evaluation
evaluation/results/extended_postC1.json/.md   post-C1 evaluation
docs/PHASE_4A_REPORT.md                       this file
```

### Deleted
**None.**

### Preserved as historical evidence
```
baseline/                              untouched (sha256 ea526081... verified)
evaluation/results/phase0_52_run1/2    Phase 3 pre-C1 results, untouched
evaluation/results/extended_run1       Phase 3 pre-C1 results, untouched
evaluation/positions/*                 datasets untouched
evaluation/metrics.py, evaluate.py     methodology untouched
app.py, config.py, models/, templates/ untouched
```

---

## K. Should C1 be accepted?

**Yes — accept, but on correctness grounds, not on measured playing strength.**

**For:**

- It is an unambiguous logic defect: a term meaning "this move is better" was
  applied in the direction meaning "this move is worse", for one side only, in the
  only side the application ever plays.
- The fix is 5 lines, in one function, with no side effects.
- **White is provably unchanged** — bit-identical by construction and confirmed on
  115 White positions across two suites plus the regression baseline.
- **Nothing regressed.** All 212 positions keep 100% legality; every quality metric
  is identical; no new mate failures.
- It is now covered by 4 passing tests including a sign-and-magnitude guard.

**Against (stated honestly):**

- **It does not improve measured playing quality at all.** Not top-1 agreement, not
  regret, not blunder rate, not mate handling, not rank correlation — on 212
  positions across two independent suites.
- It costs one historical baseline assertion.

**The honest reading:** this is a correctness fix with no measurable behavioural
benefit, because the code path it corrects is numerically inert. Accepting it makes
the engine *right*; it does not make it *better*. Both statements should appear in
any write-up of this work — reporting only the first would be the same kind of
overclaiming the earlier phases were cleaning up.

Per the brief's instruction not to judge on top-1 agreement alone: **every** signal
— regret distribution, blunder rate, mate statuses, Spearman, and agreement — is
unchanged. There is no metric on which this helps, and none on which it hurts.

---

## L. Recommended next step

1. **Accept the C1 change.** It is correct, minimal, well-tested and provably
   White-neutral.

2. **Do not re-record the Phase 0 baseline yet.** Only score *values* moved; moves
   and ordering are intact, so the baseline still detects the changes that matter.
   Re-record once a change actually alters move selection — probably after C2 —
   and record both diffs in one documented step. If you prefer a green suite now,
   re-record deliberately and note the 15 Black positions in the commit message.

3. **Prioritise C2 next, and expect it to matter more.** The 1-ply lookahead has the
   same inverted-extremum character but a larger lever: it takes `max` over opponent
   replies where the opponent minimises, and it is the single most expensive
   computation in the engine (~400 CNN evaluations per move versus ~20 for the main
   pass). Its penalty is bounded to ±0.5, so it may prove equally inert — the
   harness will now answer that in one run rather than by argument.

4. **Recognise the structural finding.** C1, C2, C4 and C5 all live in the heuristic
   layer, which the measurements say contributes under 1% of the score. Fixing them
   one by one will keep producing correct-but-inert results. The metrics that would
   actually move — mean regret 144 cp, blunder rate 24% — are dominated by the CNN
   term, and by its ply-20-only training distribution. If the goal is measurably
   better play rather than measurably correct code, that is where the effort
   belongs, and it is a retraining question, not a bug-fix question.
