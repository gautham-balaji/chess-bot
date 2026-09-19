# Phase 4B Report — C2: 1-ply lookahead minimax

**Scope:** C2 only. No other defect touched, no retraining, no change to the
evaluation methodology, Stockfish configuration or datasets.

**Starting point:** commit `863cd8a` (Phase 4A / C1 accepted), clean tree.

**Headline:** C2 is real and was **worse than previously diagnosed** — it contained
two independent errors, not one, and the second had been missed. The fix is
verified by a test that fails against the old code. It changes **zero engine moves**
on all 212 evaluated positions, and every quality metric is identical.

---

## A. Exact root cause

The pre-fix implementation:

```python
opp_scores = cnn_model.predict(np.array(opp_tensors), verbose=0).flatten()
opp_best = {}
for idx, sc in zip(opp_move_map, opp_scores):
    if idx not in opp_best or sc > opp_best[idx]:   # (1) always MAX
        opp_best[idx] = sc
for i, entry in enumerate(move_scores):
    if i in opp_best:
        entry["score"] -= 0.5 * np.tanh(opp_best[i] / 200)   # (2) always MINUS
```

**Two defects, not one:**

| # | Defect | Wrong for |
|---|---|---|
| 1 | Always takes `max` over the opponent's replies | **White to move only** (Black's opponent *is* White, who maximises — so `max` was already correct there) |
| 2 | **Subtracts** the term instead of adding it | **Both sides** |

> **The previous diagnosis was incomplete.** Phase 2 recorded C2 as "the lookahead
> takes `max` over opponent replies, selecting the reply most favourable to the
> mover rather than the opponent's best". That describes defect 1 and is correct
> for White — but it is *wrong for Black*, where `max` is the right extremum, and
> it misses defect 2 entirely. The sign inversion is the error that affected every
> position on both sides. This is exactly why the brief said not to just change
> `max()` to `min()`: doing so would have fixed nothing for Black and left the
> dominant error in place on both sides.

---

## B. Score-perspective / minimax reasoning

Established **empirically**, not assumed:

**A. What perspective is the CNN score?** White-positive centipawns.

| Position | CNN output |
|---|---:|
| White up a queen | **+342.24** |
| Balanced start | +21.59 |
| Black up a queen | **−485.69** |

Also confirmed side-to-move blind: the identical placement scores `−485.69` with
either side to move (no side-to-move plane — that is C6, untouched here).

**B. What perspective is the candidate score?** White-positive. `material_balance`
is White-minus-Black, `space_control` and `center_control` likewise, and all five
Ridge coefficients are positive (`[330.901, 32.390, 0.792, 5.165, 0.019]`).

**C. What perspective are the opponent-reply scores?** White-positive — the same
CNN applied to the post-reply position.

**D. Is any sign transform applied before the extremum?** No. The raw CNN output
goes straight into the comparison.

**E. What must the correct minimax operation be?**

The opponent picks the reply best for *them* on the White-positive scale:

| Side to move | Opponent | Opponent wants | Correct extremum |
|---|---|---|---|
| White | Black | White-positive score **low** | `min(replies)` |
| Black | White | White-positive score **high** | `max(replies)` |

That value must then be blended in **positively**, because the final sort already
encodes direction (`reverse=(board.turn == chess.WHITE)`: descending for White,
ascending for Black). Adding a White-favourable value correctly makes a position
better for White and worse for Black simultaneously.

**F. Is the current implementation inverted under those semantics?** Yes, worked
through explicitly:

*Black to move* (the only case the app plays). Sorts ascending, lower is better.
Suppose a candidate allows White a crushing reply, `V = +500`.
- Correct: `score += 0.5·tanh(2.5) = +0.4999` → score rises → ranks **worse** for Black ✓
- Pre-fix: `score −= 0.4999` → score falls → ranks **better** for Black ✗

*White to move.* Sorts descending, higher is better. Suppose Black's best reply
gives `min = −500` while some other reply gives `max = +100`.
- Correct: `V = min = −500`, `score += 0.5·tanh(−2.5) = −0.4999` → ranks worse ✓
- Pre-fix: `V = max = +100`, `score −= 0.5·tanh(0.5) = −0.231` → the *wrong reply*
  is selected **and** the wrong sign is applied ✗

The two errors do not cancel: picking `max` instead of `min` selects a different
value entirely, and the sign error is then applied to it.

---

## C. Exact production-code change

One hunk in `engine.py`. No other production file touched.

```python
opponent_maximises = (board.turn == chess.BLACK)
opp_best = {}
for idx, sc in zip(opp_move_map, opp_scores):
    if idx not in opp_best:
        opp_best[idx] = sc
    elif opponent_maximises:
        opp_best[idx] = max(opp_best[idx], sc)
    else:
        opp_best[idx] = min(opp_best[idx], sc)
for i, entry in enumerate(move_scores):
    if i in opp_best:
        entry["score"] += 0.5 * np.tanh(opp_best[i] / 200)
```

`board` is back at the original position when this runs, so `board.turn` is the
side to move — the same expression the sort on the next line relies on.

The weight (`0.5`) and scale (`200`) are **unchanged**. Explicitly not touched:
the CNN, candidate generation, heuristic bonuses (C1 stays fixed), the sort, the
Ridge model/intercept (C3), the `center` field (C4), `opening_center_bonus` (C5),
board encoding (C6), the API, and the evaluation harness.

---

## D. Tests added / changed

**6 new tests**, written and confirmed failing **before** the fix.

They drive `rerank_moves` with a **stubbed CNN** (`_StubCNN`) that returns a
constant for the candidate batch and caller-supplied values for the opponent-reply
batch. The applied lookahead term is measured by **differencing two runs that are
identical except for the reply values** — so the weighted term and the heuristic
bonuses cancel exactly and cannot confound the result. No Stockfish is involved.

| Test | Cases | What it pins |
|---|---|---|
| `test_lookahead_uses_the_opponents_best_reply` | 4 | Injecting one extreme reply must change the term **only if the opponent would actually choose it** |
| `test_lookahead_term_matches_the_minimax_value_exactly` | 2 | The applied term equals `0.5·tanh(V/200)` for the true minimax `V`, across a spread of reply values |

Both White-to-move and Black-to-move semantics are covered.

**Why these cannot pass by sharing a wrong assumption:** the expected values are
derived from the minimax contract in the test, and they differ *in sign* from what
the pre-fix code produces. Verified empirically — against the unfixed engine:

```
5 failed, 1 passed
AssertionError: black to move, e5f6: replies [-400..800] -> minimax value 800.0,
                expected applied term +0.4997, got -0.4997
```

The one pre-fix pass is the Black/low-injection case, where `max` was already the
correct extremum and the injected value does not change it — a case that is
genuinely non-discriminating for Black, by construction.

After the fix: **6 passed**. C1's tests re-verified: **6 passed**.

---

## E. Full pytest result

```
1 failed, 269 passed, 10 xfailed in 79.93s
```

280 tests collected (274 before + 6 new). The single failure is
`test_top3_scores_match_baseline` — expected, not weakened, not deleted. See §N.

xfail count unchanged at 10 (C3, C4, C5, C6×3, push-before-validate ×2, 415 ×2).
C2 never had an xfail — Phase 2 deliberately did not write one.

---

## F. phase0_52 (52 positions): Phase 4A → 4B

| Metric | post-C1 | post-C2 | Change |
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

Latency reported separately: p50 526.9 ms, p95 1955.8 ms. **No speed claim is made**
— see §K.

---

## G. extended (160 positions): Phase 4A → 4B

| Metric | post-C1 | post-C2 | Change |
|---|---:|---:|---|
| legality rate | 100.0% | 100.0% | none |
| top-1 agreement | 18.12% (29/160) | 18.12% (29/160) | none |
| top-3 containment | 33.75% (54/160) | 33.75% (54/160) | none |
| mean regret | 144.07 cp | 144.07 cp | none |
| median regret | 25 cp | 25 cp | none |
| p95 regret | 533 cp | 533 cp | none |
| max regret | 676 cp | 676 cp | none |
| blunder rate (>300cp) | 24.34% (37/152) | 24.34% (37/152) | none |
| regret coverage | 95.0% | 95.0% | none |
| Spearman mean / median | 0.22 / 0.30 | 0.22 / 0.30 | none |
| `missed_forced_mate` | 4 | 4 | none |
| `engine_move_allows_forced_mate` | 4 | 4 | none |
| **engine moves changed** | — | **0 / 160** | — |

Latency: p50 1189.9 ms, p95 4663.3 ms. Again, no speed claim.

---

## H. White vs Black impact

**C2 affects both sides — unlike C1, which was Black-only.**

| | C1 (Phase 4A) | C2 (Phase 4B) |
|---|---|---|
| Baseline positions with changed top-3 **scores** | 15 (all Black) | **52 (all of them — 30 White + 22 Black)** |
| Positions with changed top-3 **ordering** | 0 | **0** |
| Positions with changed **selected move** | 0 | **0** |
| White moves changed | 0 / 115 | **0 / 115** |
| Black moves changed | 0 / 97 | **0 / 97** |

**Code-level reason for the asymmetry between C1 and C2:** C1 was a sign error on a
term (`bonus`) whose sign only mattered relative to the sort direction, so it was
wrong for exactly one side. C2's sign error is on the lookahead term itself, which
is applied to every candidate on both sides — so *every* position's scores moved.
The extremum error is additionally White-only, because Black's opponent is White,
who genuinely maximises.

Per-side metrics are unchanged on both suites (phase0_52: White top-1 26.67%, mean
regret 122.52; Black 9.09%, 85.0. extended: White 17.65%, 154.64; Black 18.67%,
132.01).

---

## I. Number of moves changed

**0 of 212.** Score-only changes: 52/52 on the regression suite. Ordering changes: 0.

---

## J. Per-position changed-move analysis

**There are no changed moves to analyse** — the diff is empty on both suites, for
both colours. Nothing is being withheld or cherry-picked.

### Why — measured, not asserted

Across all **212 positions**, comparing the pre-C2 and post-C2 lookahead terms
computed from the *same* CNN reply evaluations:

| Quantity | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| rank-1 → rank-2 score gap | 0.134 | **32.684** | 59.736 | 490.957 |
| Max **relative** C2 swing (worst candidate pair) | 0.003 | **0.570** | 0.569 | 1.300 |
| Max per-candidate \|C2 delta\| | 0.254 | 0.854 | 0.782 | 1.000 |

The median gap between the best and second-best move is **57× larger** than the
median relative swing the fix introduces. **Only 3 of 212 positions** have a gap
small enough for the swing to reorder the top two at all — and in none of them did
it.

For context, C2's per-candidate swing (median 0.854, max 1.000) is **larger** than
C1's (median 0.500, max 1.300 swing but smaller typical values) — the fix moves
scores more than C1 did, and it still is not close to mattering.

Root cause of the inertness is unchanged from Phase 4A: the Ridge model weights the
CNN term at **330.9** and material at **32.4**, while the lookahead term is bounded
to ±0.5 (a ±1.0 swing when the sign flips). The lookahead was never capable of
deciding a move.

---

## K. Runtime impact

**The CNN workload is provably identical.** Instrumenting `cnn_model.predict` over
the 52-position suite:

| | pre- and post-C2 |
|---|---|
| `predict` calls | **104** (exactly 2 per position) |
| Position rows evaluated | **36,779** |

The fix changes only a `min`/`max` branch inside a loop that already existed. It
adds no `predict` call, no tensor construction and no board push/pop.

Wall-clock timings are reported but **no speed claim is made**. Three repeated
isolated trials on the same 12 positions gave medians of **2001 / 904 / 1038 ms** —
a 2.2× spread on identical code, consistent with Phase 3's finding that latency on
this host does not reproduce. Harness wall clocks (phase0_52: 65.1 s → 91.3 s;
extended: 261.6 s → 570.2 s) are dominated by the same host variance and should not
be read as a C2 effect.

No optimisation was attempted — correctness only, as instructed.

---

## L. Unexpected findings

**1. The prior C2 diagnosis was incomplete, and the missing half was the bigger
error.** Phase 2 recorded only the `max`/`min` issue. Investigation found a second,
independent sign inversion that affected *both* sides and *every* position, whereas
the extremum error affected White only. Following the earlier description literally
— changing `max()` to `min()` — would have made Black strictly worse (breaking a
case that was already correct) while leaving the dominant error untouched. The
brief's instruction to re-derive the semantics rather than trust the report was
load-bearing.

**2. `max` was already correct for Black.** Because Black's opponent is White, who
genuinely maximises the White-positive score. Easy to miss, and the reason the
original one-line description was misleading.

**3. C2 is inert despite moving scores more than C1 did.** Per-candidate deltas up
to 1.000 versus C1's smaller typical swing, and it touches all 212 positions rather
than 15 — yet still zero move changes. The 57× gap ratio explains it.

**4. Two independent bugs, two identical outcomes.** C1 and C2 were different
defects in different terms, and both produced exactly the same result: correct code,
no behavioural change. That consistency is itself the finding — see §Q.

**Not investigated (out of scope):** the mate failures (4 missed forced mates, 3–4
mate-allowing moves) are unchanged and remain unexplained by either fix.

---

## M. Files changed / added

### Modified (2)
```
engine.py                        +25 -2     the C2 fix (one hunk)
tests/integration/test_engine.py +173 -0    6 new C2 tests
```

### Added (5)
```
evaluation/results/phase0_52_postC2.json / .md
evaluation/results/extended_postC2.json / .md
docs/PHASE_4B_REPORT.md
```

`evaluation/compare_runs.py` from Phase 4A was reused unchanged — no new tooling
was needed.

### Deleted
**None.**

### Preserved untouched
```
baseline/                                   historical Phase 0 evidence
evaluation/results/*_run1, *_run2           Phase 3 results
evaluation/results/*_postC1                 Phase 4A results (the pre-C2 reference)
evaluation/positions/*                      datasets
evaluation/metrics.py, evaluate.py          methodology
app.py, config.py, models/, templates/      untouched
```

---

## N. Historical baseline assertions affected

One assertion, on **all 52 positions**:

| Assertion | Result | Why |
|---|---|---|
| `test_every_baseline_position_still_yields_a_legal_move` | **PASS** | 52/52 legal |
| `test_legality_rate_matches_baseline` | **PASS** | 52/52 |
| `test_selected_moves_match_baseline` | **PASS** | No move changed |
| `test_top3_ordering_matches_baseline` | **PASS** | No ordering changed |
| `test_top3_scores_match_baseline` | **FAIL** | Lookahead term changed sign (both sides) and extremum (White) |
| `test_no_baseline_position_mutates_its_board` | **PASS** | 52/52 |
| `test_replay_is_deterministic_for_a_sample` | **PASS** | — |
| `test_baseline_suite_is_intact` | **PASS** | — |

All 52 positions are affected because the sign inversion applied to every candidate
on both sides. This is the intended consequence of the fix. The test was not
weakened, skipped or deleted, and the baseline was not re-recorded.

---

## O. Should C2 be accepted?

Answering the three questions separately, as instructed:

**1. Is C2 logically incorrect?** **Yes — on two counts**, one more than previously
documented. The extremum selected the wrong reply for White; the sign inverted the
term for both sides. Established from first principles by measuring the CNN's
perspective, not by trusting the earlier report.

**2. Does fixing C2 change engine behaviour?** **It changes scores but not
behaviour.** All 52 baseline positions shifted in score; 0 of 212 positions changed
their selected move or top-3 ordering.

**3. Does fixing C2 improve measured playing quality?** **No.** Not top-1 agreement,
top-3 containment, mean/median/p95/max regret, blunder rate, regret coverage,
Spearman, or mate statuses — on 212 positions across two independent suites. It also
does not make anything worse.

**Recommendation: accept**, on the same basis as C1 — correctness, not strength.

- It is a genuine minimax error in the most expensive computation the engine
  performs (~36,779 CNN position evaluations across 52 positions, ~700 per move).
- The fix is ~10 lines in one function, adds no computational cost, and is pinned
  by 6 tests that provably fail against the old code.
- Nothing regressed: 100% legality maintained, no metric moved in either direction.
- It corrects a documented defect whose description was itself wrong, and the report
  now records the correct semantics.

**What must not be claimed:** that this improves the engine. It does not, on any
measurement available.

---

## P. Is a new regression baseline justified?

**Not yet — but the case is stronger than after C1.**

Arguments for re-recording now:
- All 52 positions now differ in score, so `test_top3_scores_match_baseline` is
  permanently red and provides no ongoing signal.
- Two accepted changes (C1, C2) have accumulated against it.

Arguments against:
- The baseline's *valuable* assertions — selected move, ordering, legality,
  determinism, board integrity — all still pass and still detect what matters.
- Re-recording discards the only artefact tying current behaviour to the measured
  Phase 0 state.

**Recommendation:** re-record once, deliberately, as its own reviewed commit,
recording that the score deltas come from C1 (15 Black positions) and C2 (all 52),
that no move or ordering changed in either, and carrying `baseline/` forward
untouched as historical evidence. Do it before C3/C4/C5 so the next phase starts
from a green suite — but as an explicit decision, not as a side effect.

---

## Q. Recommended next step

**Stop fixing heuristic-layer defects one at a time.** Two phases have now produced
the same result:

| Phase | Defect | Logically wrong? | Moves changed | Quality change |
|---|---|---|---|---|
| 4A | C1 — bonus sign | Yes | 0 / 212 | none |
| 4B | C2 — lookahead minimax | Yes (×2) | 0 / 212 | none |

C3 (Ridge intercept) is a **constant offset** — it provably cannot change any
ranking, so it will be inert by construction. C4 (`center` pre/post-move) affects a
*reported* field, not the score. C5 (`opening_center_bonus` White-only) is another
bonus-layer term of the same magnitude C1 showed to be 57–82× too small.

The measurements point somewhere else. Mean regret is **107–144 cp** and the blunder
rate is **15–24%**, and both are dominated by the CNN term (Ridge weight 330.9)
whose training distribution was **10,000 positions all at ply 20**. The category
breakdown from Phase 3 is consistent: middlegame mean regret 185–232 cp versus
endgame 12–40 cp.

Concrete options, in order of expected value:

1. **C6 — add side-to-move / castling / en-passant planes and retrain.** The CNN is
   currently blind to whose turn it is while being asked to evaluate post-move
   positions where the turn has flipped. This is the one open defect that plausibly
   moves regret. It is a retraining task, not a bug fix, and needs `games.csv`
   (currently gitignored and undistributed) — that blocker should be resolved first.
2. **Re-record the baseline** (§P) — cheap, unblocks a green suite.
3. **Batch C3/C4/C5 into a single "known-inert corrections" phase** with one
   evaluation run rather than three, since the per-phase measurement cost is no
   longer buying new information.

If the goal is a demonstrably stronger engine rather than demonstrably correct code,
option 1 is the only one of these that can deliver it.
