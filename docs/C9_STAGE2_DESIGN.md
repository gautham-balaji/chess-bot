# C9 Stage 2 — Matched-Fusion Engine Experiment (design only)

## 1. Status

**DESIGN ONLY. Stage 2 has NOT been implemented and has NOT been run in this
task.** No Stockfish was invoked, no evaluation suite was touched, no production
file was modified, no CNN was retrained, nothing was committed.

One read-only feasibility check *was* performed (§13.2) to confirm the divisor
mechanism is arithmetically exact before committing to it. It wrote only to a
scratch directory and involved no evaluator and no Stockfish.

Prior: [C9 audit](C9_RIDGE_AUDIT.md) · [C9 Stage 1](C9_STAGE1_REPORT.md) ·
[A1R S1](C6_A1R_STAGE1_REPORT.md) · [A1R S2](C6_A1R_STAGE2_REPORT.md) ·
[A2R S1](C6_A2R_STAGE1_REPORT.md) · [C8a](C8A_REPORT.md) · [C8b](C8B_REPORT.md)

---

## 2. Objective

> **Does changing the fusion of the C8a CNN into the final move-ranking score
> improve actual engine behaviour against the frozen Stockfish evaluator?**

Stage 1 measured coefficients and proxies. Stage 2 measures moves. Nothing short
of the engine answers this.

---

## 3. Stage 1 findings that constrain Stage 2

1. **The coefficient gate tripped** on all three seeds and both Ridge variants.
   CNN ranking share falls from 81.5–83.4% (production) to 45.5–49.6% (variant A)
   or 54.1–58.4% (variant B), against an A0 band of 82.85–84.93%. Top-ranked
   candidate changes on 23.8–36.3% (A) or 16.3–25.0% (B) of positions.

2. **The two levers are coupled.** Raising the divisor raises correlation
   (0.704 → 0.874) but *shrinks* the CNN term (SD 48.58 → 8.19, share
   82.2% → 43.7%), because `tanh(x/d) ≈ x/d` for large `d`.
   **A divisor change with `w0` held fixed measures the wrong thing** and is
   excluded from this design by construction.

3. **The two Ridge variants disagree in direction.** Including the 1,129 mated
   boards gives `w0` ≈ 377–389 (1.14–1.18× production); excluding them gives
   `w0` ≈ 246–252 (0.74–0.76×). The refit does not even agree on whether the CNN
   should be weighted *more* or *less*. §7 resolves this without engine time.

4. **Matched fusion is not a known good.** A1R Stage 2 ran this exact comparison
   for A0 and A1: it **improved A0 and degraded A1**. It is a different fitting
   methodology, not a quality upgrade.

---

## 4. Exact control

**Variant A = C8a with production fusion.** This is **already measured** — it is
the C8a arm reported in [C8A_REPORT.md](C8A_REPORT.md), produced by the same
evaluator, suites, Stockfish configuration and seeds. **It is not re-run.**

| | value |
|---|---|
| CNN | C8a seeds 0/1/2, weights `6bd57636…`, `144594ff…`, `41ee1626…` |
| Ridge | production `models/weight_model.pkl`, `coef_ = [330.900549, 32.389872, 0.791884, 5.165416, 0.018828]` |
| divisor | 200 (the `engine.py` literal) |
| intercept | **not applied** (defect C3) |
| extended | top-1 21.88%, top-3 39.17%, mean regret 107.03, median 17.00, blunder 0.160, Spearman 0.297 |
| phase0_52 | top-1 27.57%, top-3 46.16%, mean regret 95.38, median 15.17, blunder 0.148, Spearman 0.293 |

Re-running the control would cost 6 suite runs for numbers we already hold under
identical conditions. Reusing it is only valid because *nothing* about the
control's configuration changes — which is true here, and is asserted in §16
step 0 before any comparison is made.

---

## 5. Candidate variants

Four were required for consideration. The proposed engine matrix is **two new
cells**, because two of the four are resolved without engine time.

| | variant | Ridge | divisor | engine-evaluated? |
|---|---|---|---|---|
| **A** | production fusion | production | 200 | **no — already measured** (§4) |
| **B** | matched Ridge only | C8a-matched at d=200 | 200 | **YES** |
| **C** | jointly matched | C8a-matched at d\* | d\* | **YES** |
| **D** | checkmate inclusion | A-vs-B fit variant | — | **no — resolved in Stage 2a** (§7) |

**Why B is included.** It answers the literal question the C9 audit posed — "does
a Ridge fitted to C8a's output distribution improve ranking?" — and it is the
only variant directly comparable to A1R Stage 2's methodology. Dropping it would
leave the programme's original fusion question unanswered.

**Why C is included.** Stage 1 showed the coefficient lever alone cannot recover
the ~0.17 of correlation the squash discards. C is the only variant that can.

**Why D is not an engine cell.** It is a property of *how each of B and C is
fitted*, not a third deployment. Running both fit variants through the engine
would double the matrix to four cells for a question that a leakage-free
ranking benchmark can settle beforehand (§7).

**Engine matrix: 2 cells × 3 seeds × 2 suites = 12 suite runs.**

---

## 6. Divisor-selection methodology (leakage-safe)

### The constraint

The suites are the measurement instrument. Selecting `d` by running the suites
and keeping the best value would convert them into a tuning set and invalidate
every number C6–C8 produced on them. **This must not happen.**

### Why the Stage 1 sweep cannot select `d` on its own

`corr(tanh(cnn/d), label)` is **monotonically increasing** in `d` across the
whole swept range. Maximising it selects `d → ∞`, where the CNN term vanishes
(SD 8.19 at d=2000 and still falling). Correlation is a *between-position*
criterion; ranking is *within-position*. **A between-position criterion cannot
select a parameter whose effect is within-position.**

### Proposed method — Stage 2a, a leakage-free ranking benchmark

Build a within-position ranking benchmark that never touches the suites:

1. **Split the fit population by game.** `dataset_v2.test.jsonl` (13,712
   positions, 3,614 games) splits into **FIT (80% of games)** and
   **TUNE (20% of games)**, deterministic, sorted-then-permuted with a fixed
   seed **9** — deliberately distinct from the dataset split seed 42 and the
   inner-split seed 1234. Game-level, matching C7/C8 discipline, so no game
   contributes to both.
2. **Label TUNE candidate moves with Stockfish.** For each of ~200 TUNE
   positions, generate every legal move and score the resulting board at the
   **same** Stockfish configuration (17.1, depth 8, Threads=1, Hash=16MB, hash
   cleared per position). ~30 candidates per position ≈ **6,000 evaluations**.
   These are new labels on held-out positions — **no suite position, no suite
   label, and no evaluator output is read.**
3. **For each `(d, fit-variant)` pair**, fit the Ridge on **FIT only** at divisor
   `d`, then score TUNE's candidate rankings with the full fused score.
4. **Criterion (pre-registered):** mean within-position **Spearman** between the
   fused candidate score and Stockfish's candidate ordering, averaged over TUNE
   positions and over the three seeds. Top-1 agreement is recorded as a secondary
   read but does **not** break ties.
5. **Selection rule (pre-registered):** `argmax` over the candidate set. Ties
   within 0.005 Spearman resolve toward the **smaller `d`** (conservative, closer
   to production), then toward **variant A** (includes everything, matching the
   production Ridge's own methodology).

### The candidate set is pre-declared and closed

`d ∈ {200, 400, 600, 800, 1000, 1500, 2000}` — **exactly Stage 1's swept values,
no new ones invented.** Including 200 means "keep the production divisor" is
always available to win. The set is fixed before Stage 2a runs and is not
extended afterwards.

### Why this is leakage-safe

| property | status |
|---|---|
| suites read during selection | **never** |
| suite labels used | **never** |
| selection population overlaps suites | **0** placements, **0** exact FENs (Stage 1 §4 verified) |
| selection population overlaps C8a train | **0** placements, **0** games |
| Ridge fitted on the selection population | **no** — FIT and TUNE are disjoint by game |
| deterministic | yes — fixed seed 9, no sampling |

The Ridge deployed in Stage 2b is the **FIT-only** Ridge, never refitted on TUNE.
That is a textbook train/validate separation and needs no "refit on everything"
justification.

### If selection returns d\* = 200

Then variant C collapses onto variant B and **only one new cell is run**. That is
a legitimate outcome, not a failure, and it must be reported as such.

---

## 7. Checkmate-inclusion methodology

Stage 1 showed the choice is not cosmetic: `w0` ≈ 377–389 with mates, ≈ 246–252
without, reversing direction relative to production.

**The choice is resolved in Stage 2a, by the same ranking criterion, not by
engine evaluation and not by fiat.** Both fit variants are carried through the
`(d, variant)` grid in §6 step 3; the selection in step 5 picks one.

The justification for letting the benchmark decide: the engine's task *is*
within-position ranking of legal moves. Mated boards have no legal moves, so a
fit variant's merit for ranking is exactly what the TUNE benchmark measures.
Choosing by argument rather than measurement would be guessing.

**The eliminated variant is reported, not discarded** — its TUNE score appears in
the Stage 2a artifact so the margin of the decision is visible. If the two
variants score within the 0.005 tie threshold, the tie-break selects A and the
report must state that the choice was effectively arbitrary.

---

## 8. Leakage controls

| control | mechanism |
|---|---|
| suites never tuned on | selection completes in Stage 2a before any suite is run; the candidate set is closed |
| suite positions absent from fit/tune | asserted at runtime by `c9_stage1.leakage_checks`, reused |
| Ridge never fitted on TUNE | game-level FIT/TUNE split, seed 9 |
| Ridge never fitted on C8a train | fit population is C8a's held-out split |
| CNN never retrained | models loaded read-only, hash-verified against the audit |
| no post-hoc variant addition | §5 matrix and §6 candidate set frozen by this document |
| suites unmodified | staged models only; suites read-only |

**One residual, stated plainly:** the Stage 2a benchmark and the Stage 2b suites
are different populations. A configuration selected as best on TUNE is not
guaranteed best on the suites. That is the honest cost of refusing to tune on the
instrument, and it means a null Stage 2b result does **not** prove the fusion
cannot be improved — only that this selection did not transfer.

---

## 9. Exact evaluator configuration

Unchanged in every respect. No evaluator edit of any kind.

| | value |
|---|---|
| evaluator | `evaluation/evaluate.py`, unmodified |
| Stockfish | 17.1 |
| depth | 8 |
| Threads | 1 |
| Hash | 16 MB |
| Clear Hash | per position |
| MultiPV | agreement 3, ranking 8 (existing constants) |
| suites | `extended` (160), `phase0_52` (52) |
| regret definition | existing |
| mate handling | existing |
| legality metric | existing |

---

## 10. Metrics

The existing headline set, unchanged: legality, top-1 agreement, top-3
containment, mean/median/p95/max regret, >300cp blunder rate, regret coverage,
Spearman mean and median, mate status counts, White/Black split, phase/category
split.

Reported per seed and as three-seed means. **No new metric is introduced and no
composite score is formed.**

---

## 11. Pre-registered decision rule

### Applying the A0 noise band

For each contrast — **A→B** and **A→C** — paired by seed:

1. Per metric, compute the three per-seed deltas and their mean.
2. Compare `|mean delta|` to the **A0 three-seed range** for that metric on that
   suite (the existing band used by every C6/C8 arm).
3. A metric **counts** only if `|mean delta| >` band **and** all three per-seed
   deltas share a sign.

### Verdict per variant, per suite

- **IMPROVEMENT** — ≥1 metric counts as better, and **0** metrics count as worse.
- **REGRESSION** — ≥1 metric counts as worse, and **0** count as better.
- **MIXED** — at least one of each.
- **NO EFFECT** — no metric counts.

### Overall C9 conclusion (pre-registered)

> **"Matched fusion improves the engine"** may be claimed only if a variant is
> **IMPROVEMENT on both suites**.
>
> Any other combination is reported as **"no engine-level benefit demonstrated"**.

This is deliberately conservative and matches how C8a's claim was made.

### Explicitly pre-registered edge cases

| case | ruling |
|---|---|
| B improves, C regresses (or vice versa) | Report both factually. No aggregate ranking of variants. The improving variant may be claimed only if it meets the both-suites bar on its own. |
| Improvement on `extended`, none on `phase0_52` | **Not** an improvement. Reported as inconsistent across suites. |
| MIXED on either suite | No claim. Report which metrics moved which way. |
| **C improves model-level correlation but not engine metrics** | **NO engine benefit.** Pre-registered explicitly because C8b already produced exactly this pattern — better regression, no better chess. Model-level gains are reported separately and never substitute for engine metrics. |
| d\* = 200 selected | C collapses onto B; report as a finding that the production divisor survived a fair comparison. |
| Both variants NO EFFECT | C9 closes; the fusion branch is exhausted (§ summary). |

**No overall score, no weighting, no variant ranking.** The output is a factual
per-metric table plus the four-way verdict above.

---

## 12. Seed pairing

C8a seeds 0/1/2 are **paired throughout**: seed *n*'s matched Ridge is fitted
from seed *n*'s CNN outputs and evaluated against seed *n*'s control. No seed is
selected, ranked, dropped or averaged before pairing. Per-seed and three-seed
aggregate results are both reported. This matches A1R Stage 2's "Option B"
(one Ridge per seed), which Stage 1 already follows.

---

## 13. Temporary staging strategy

### 13.1 Matched Ridge — the established mechanism

`training/evaluate_arm.py::stage_models_dir(experiment_model, staging,
ridge_model=…)` already exists, built for A1R Stage 2, and copies a chosen Ridge
into a temporary directory that `CHESS_BOT_MODELS_DIR` points at.
`models/weight_model.pkl` is only ever **read**. `training/refit_ridge_stage2.py`
additionally verifies, after staging, that the staged Ridge matches Stage 1's
recorded coefficients and is **not** the production vector. C9 Stage 2 reuses
both.

### 13.2 Divisor — staged as a rescaled CNN, no `engine.py` change

The divisor is a literal at `engine.py:130`, `:164` and `:240`. It cannot be
staged. But because the model's output layer is `Dense(1, use_bias=True)`:

```
scale the final kernel and bias by k = 200/d
  =>  output' = k * output
  =>  tanh(output'/200) = tanh(output/d)     exactly
```

**Verified read-only before writing this design** (scratch only, no evaluator, no
Stockfish, on C8a seed 0 over 400 held-out positions):

| check | result |
|---|---:|
| `max |scaled − k·base|` | 6.1e-05 |
| `max |tanh(scaled/200) − tanh(base/d)|` | **1.2e-07** |
| survives `.keras` save/load | **0.0e+00** |

So variant C stages *a CNN artifact whose weights differ only by a scalar on the
final layer*, plus its matched Ridge. **No `engine.py` edit, no in-process
patching, no production write.**

**A property of this mechanism that must not be hidden:** `engine.py` applies
`/200` in **three** places, including the unscaled 1-ply lookahead term
`0.5*tanh(opp_best/200)`. Rescaling the CNN therefore changes the effective
divisor in the lookahead too. This is arguably the *correct* reading of "change
the divisor" — the CNN's output scale is a property of the model, not of one call
site — but it means variant C changes two things relative to variant B, and the
report must say so. Isolating them would require an `engine.py` change, which is
out of scope.

The staged CNN must be verified after staging: its final-layer weights equal
`k ×` the source's, and every other layer is bit-identical.

---

## 14. Runtime and cost estimate

### Stage 2a (selection) — no suites

| item | estimate |
|---|---|
| TUNE positions | ~200 (20% of 3,614 games, capped) |
| candidate moves | ~6,000 (~30/position, from Stage 1's measured 2,416 over 80 positions) |
| Stockfish evaluations | **~6,000** at depth 8 ≈ **1–2 min** (C7 measured 154.5 pos/s) |
| Ridge fits | 7 divisors × 2 variants × 3 seeds = 42 fits, seconds total |
| CNN inference | ~6,000 boards × 3 seeds, seconds |
| **total** | **≈ 5–10 min** |

### Stage 2b (engine) — the suites

Observed C8a rates: `extended` 198–261 s, `phase0_52` 52–73 s per seed.

| item | estimate |
|---|---|
| suite runs | 2 cells × 3 seeds × 2 suites = **12** |
| positions evaluated | 2 × 3 × 212 = **1,272** |
| wall clock | 12 runs ≈ **28 min** at observed rates |
| **realistic budget** | **45–90 min** — C8b saw 6× epoch-time degradation under external machine load, so the point estimate is not the planning number |

If d\* = 200, both figures halve (6 suite runs, ~14 min).

### Storage

C8a's per-seed evaluation output is 236 KB. Stage 2b adds ≈ **2 cells × 3 seeds
× 236 KB ≈ 1.4 MB**, plus 6 matched Ridge pickles (~1 KB each) and, for variant
C, 3 rescaled CNN copies (~9 MB each ≈ **27 MB**). All under
`training/experiments/C9/`, which is gitignored. **Nothing large is committed** —
only the JSON results artifact and the report.

---

## 15. Risks and confounds

1. **Matched fusion is not known to help.** A1R Stage 2 measured it improving A0
   and degrading A1. A null or negative Stage 2 is a genuinely likely outcome.
2. **Methodological asymmetry (inherited).** The matched Ridge is fitted
   **out-of-sample**; the production Ridge was fitted **partly in-sample** on
   data that no longer exists. Part of any measured difference is attributable to
   that, not to matching. **Unavoidable** — the original fitting population is
   unrecoverable — and it applies equally to B and C.
3. **Variant C changes two things** (Ridge *and* effective divisor, the latter in
   three call sites including the lookahead). A→C is therefore not a clean
   single-variable contrast; A→B is. This is why both are run.
4. **Selection may not transfer** from TUNE to the suites (§8).
5. **Small suites, three seeds, descriptive only.** n=160 and n=52; category
   cells as small as n=4–10. No significance testing. The A0 band is wide
   relative to plausible fusion effects.
6. **Defect C3 persists.** The Ridge is fitted *with* an intercept the runtime
   discards. Identical for control and variants, so the comparison is fair, but
   the fitted object is not the deployed function. **C3 is not fixed here.**
7. **The unscaled additive terms.** Heuristic bonuses (±0.20–0.65) and the
   lookahead (±0.5) are not scaled by the Ridge. At `w0 ≈ 331` they are
   negligible; if a matched `w0` were far smaller they would gain influence.
   Stage 1's `w0` values (246–389) keep them negligible, but variant C's rescaled
   CNN changes the lookahead's *behaviour* (point 3).
8. **Reusing the control's existing measurement** assumes the C8a evaluation
   environment is reproducible. Mitigated by step 0 in §16, which re-verifies the
   control's inputs before any comparison.

---

## 16. Implementation plan

**Step 0 — preflight (no Stockfish).** Verify C8a model hashes against the audit;
verify `dataset_v2.test.jsonl` against its manifest; verify production
`weight_model.pkl` and both suites are byte-identical to the values in
[C9_STAGE1_REPORT.md](C9_STAGE1_REPORT.md) §14; confirm the C8a control
evaluation artifacts exist and record their hashes. Abort on any mismatch.

**Step 1 — FIT/TUNE split.** Game-level, seed 9, deterministic. Assert
`FIT_games ∩ TUNE_games = ∅` and that both are suite-clean via the existing
`leakage_checks`.

**Step 2 — label TUNE candidates.** Generate legal moves for each TUNE position,
score resulting boards with the pinned Stockfish configuration, persist with
hashes. ~6,000 evaluations.

**Step 3 — grid and selection.** For each `(d ∈ {200,400,600,800,1000,1500,2000},
variant ∈ {A,B})` and each seed: fit Ridge on FIT at divisor `d`; score TUNE
candidate rankings; record mean within-position Spearman. Apply §6 step 5.
Persist the **full grid**, not just the winner. **Stop and report** — selection is
reviewable before any suite time is spent.

**Step 4 — stage artifacts.** Build 3 matched Ridges for variant B (d=200) and,
if `d* ≠ 200`, 3 matched Ridges plus 3 rescaled CNNs for variant C. Verify each
staged artifact (§13).

**Step 5 — engine evaluation.** 12 suite runs (or 6 if `d* = 200`) via
`evaluate_arm` with `ridge_model=`, into `training/experiments/C9/`. Existing C8a
evaluation output is never overwritten.

**Step 6 — analysis.** Reuse `training/a2_analysis.py` with
`--arms A0 C8a C9b C9c --baseline A0 --contrasts C8a:C9b C8a:C9c`. Apply §11.

**Step 7 — report.** `docs/C9_STAGE2_REPORT.md`, plus production-integrity and
full-suite verification.

A hard gate sits between steps 3 and 4: **selection is reported and reviewed
before the suites are touched.**

---

## 17. Files expected to change or be created

**New:**

```
training/c9_stage2.py                       selection (2a) + staging + evaluation (2b)
tests/unit/test_training_c9_stage2.py       tests for the above
docs/C9_STAGE2_REPORT.md                    results
training/experiments/C9/stage2_selection.json   full grid + choice   (gitignored)
training/experiments/C9/stage2_results.json     engine results       (gitignored)
training/experiments/C9/tune_candidates.json    TUNE labels          (gitignored)
training/experiments/C9/{matched_ridges,scaled_cnns}/                (gitignored)
training/experiments/C9/{C9b,C9c}/seed_*/evaluation/                 (gitignored)
```

**Modified (additive only, defaults preserved):**

```
training/refit_ridge.py     possibly a fit-mask/FIT-subset argument, if step 3
                            cannot be expressed through the existing surface
```

`training/evaluate_arm.py` is expected to need **no change** — `ridge_model=`
already exists. If a change proves necessary it must be additive with defaults
preserved, and the existing A1R/A2R tests must continue to pass unmodified.

**Expected test-count change:** +N new tests only. The baseline is **861 passed,
10 xfailed**. No existing test may be weakened, relaxed or deleted; if a roster
assertion must widen (as happened twice in Stage 1), the change must be stated
explicitly with its justification.

---

## 18. No production file will be modified

Stage 2 will not modify — and its implementation must assert byte-identity for —
`engine.py`, `app.py`, `config.py`, `evaluation/` (including both suites),
`models/` (CNN **and** Ridge), `baseline/`, `regression/`, `training/labels.py`,
`training/dataset.py`, `training/train.py`, `dataset_v1`, `dataset_v2`,
`dataset_v2_k6`, and all C8a/C8b artifacts.

Matched Ridges and rescaled CNNs exist only in temporary staging directories and
under the gitignored `training/experiments/C9/` tree. The production Ridge is
read; it is never written. **The Ridge intercept remains excluded from runtime
scoring** for control and variants alike — C3 is not fixed by C9.

---

## 19. Stage 2 is not being run in this task

This document is the design. No selection, no staging, no evaluation, no
Stockfish, no commit.

---

## Summary

**What exact variants would Stage 2 run?**
Two new engine cells. **B** = C8a + matched Ridge at divisor 200. **C** = C8a +
Ridge and divisor matched *jointly* at the selected `d*`. The control **A**
(production fusion) is reused from the C8a report, not re-run. The checkmate
question (**D**) is settled in Stage 2a without engine time. If `d* = 200`, C
collapses onto B and only one new cell runs.

**How is the divisor chosen without leakage?**
By a pre-registered ranking benchmark built entirely outside the suites: a
game-level 80/20 FIT/TUNE split (seed 9) of C8a's held-out set, ~6,000 TUNE
candidate moves freshly labelled by the same pinned Stockfish, Ridge fitted on
FIT only, `d` chosen by mean within-position Spearman on TUNE from the closed set
`{200, 400, 600, 800, 1000, 1500, 2000}` — Stage 1's own sweep values, with 200
included so production can win. Ties favour the smaller `d`, then variant A. The
suites are never read during selection.

**How many Stockfish evaluations?**
~6,000 for selection (≈1–2 min) plus 1,272 position evaluations across 12 suite
runs (≈28 min at observed C8a rates; budget 45–90 min). Halved if `d* = 200`.

**What is the decision rule?**
Per metric, paired by seed: a change counts only if `|mean delta|` exceeds the A0
three-seed band **and** all three seeds agree in sign. A variant is IMPROVEMENT
only with ≥1 counting improvement and **zero** counting regressions, on **both**
suites. Anything else is "no engine-level benefit demonstrated". No composite
score, no variant ranking.

**What would cause us to stop C9 after Stage 2?**
Both variants returning NO EFFECT or MIXED, or any variant regressing. Given
A1R Stage 2's arm-dependent result and C8b's precedent of model-level gains not
transferring, this is a likely outcome and would close the fusion branch.

**What would justify documenting an experimental improvement?**
A variant meeting the both-suites IMPROVEMENT bar. Even then the claim is
"experimental, under frozen production Ridge methodology" — shipping would need
a separate decision, because the matched Ridge is fitted out-of-sample against a
production Ridge fitted partly in-sample, and that asymmetry is not resolvable.

**What remains uncertain?**
Whether TUNE-based selection transfers to the suites; how much of any difference
is the in-sample/out-of-sample asymmetry rather than matching; that variant C
changes the divisor in three call sites including the lookahead, so A→C is not a
single-variable contrast; and whether the A0 band is even narrow enough to
resolve a fusion-sized effect on n=160 and n=52 with three seeds.
