# C6-A1R — Ridge Compatibility Diagnostic (design only)

**This document is audit and design only. Nothing was trained, no diagnostic was
run, no production or experiment artifact was modified, nothing was committed.**

Repository state: `6f8ff5d`, clean tree. A0 and A1 complete and committed.

---

## 1. Question

> Does A1's engine-level behaviour materially depend on pairing the **production
> Ridge coefficients** with A1's **changed CNN output scale**?

A1 showed better label fit (Pearson 0.592–0.606 vs A0's 0.509–0.530) but
consistently worse engine metrics on all three paired seeds. A1's report flagged
one unresolved confound: the production Ridge was fitted against the *original*
CNN's output scale, and A1 changes that scale.

This document determines whether that confound can be isolated with the artifacts
available, and if so, specifies the experiment.

---

## 2. What the current Ridge actually does

Verified from source, not from prior summaries.

**The only Ridge-fitting code in the repository is `chess_model_FINAL.ipynb`
cell 31.** No script or module fits it; every other reference merely loads
`models/weight_model.pkl`.

```python
for board, sf_cp in zip(sample_boards, sf_evaluations):
    cnn_score = cnn_evaluate(board)
    cnn_norm  = np.tanh(cnn_score / 200)
    hybrid_features.append([cnn_norm, material_balance(board), space_control(board),
                            center_control(board), mobility_score(board)])
    sf_targets.append(sf_cp)

weight_model = Ridge(alpha=1.0)
weight_model.fit(hybrid_features, sf_targets)
```

Answering the seven questions in the brief:

| # | Question | Answer | Status |
|---|---|---|---|
| 1 | Features | `[cnn_norm, material, space, center, mobility]`, where `cnn_norm = tanh(cnn/200)` | **VERIFIED** |
| 2 | Target | `sf_evaluations` — the **original 10,000 labels**: side-to-move relative, mate stored as raw distance. That is **A0-policy-like**, *not* A1-policy | **VERIFIED** |
| 3 | CNN outputs used | The **original** trained CNN, evaluated on the same 10,000 `sample_boards` | **VERIFIED** |
| 4 | Are the original 10k / original CNN required? | To **reproduce** the production Ridge: yes, and it is **impossible** — the original labels were never persisted and are unrecoverable. To **refit** for an arm: no; `dataset_v1` and the arm's own CNN suffice | **VERIFIED** |
| 5 | Can a refit avoid leakage? | Partially — see §6. A clean-ish refit is possible but constrained | **VERIFIED** |
| 6 | Is the coefficient used as assumed? | Yes. `engine.py:166-167` uses `weight_model.coef_` only; `intercept_` (15.077) is **never applied** (defect C3, still open) | **VERIFIED** |
| 7 | Is the CNN term the only changed feature? | **Yes.** `material`, `space`, `center`, `mobility` are pure board functions and do not involve the CNN. Measured identically across all six models (§5) | **VERIFIED** |

### Three pre-existing properties of the production Ridge that matter here

1. **It was fitted on all 10,000 positions**, including the ~2,000 the CNN held
   out *and* the ~8,000 the CNN trained on. Its `cnn_norm` feature is therefore
   partly in-sample, which inflates the CNN coefficient relative to what would be
   appropriate on unseen positions. **The original fitting process already had
   this leakage.**
2. **It was fitted on features of the position itself**, but the engine applies it
   to features of **post-move** positions and uses it to *rank moves within* a
   position. It was fitted for between-position accuracy and is used for
   within-position discrimination. These are different objectives.
3. **Its target was A0-style labels.** So A1 pairs an A1-trained CNN with a fusion
   layer fitted to A0-style targets — a mismatch in **target semantics**, not only
   in output scale.

---

## 3. Why A1 creates a potential confound

A1 changed 188 mate labels from raw distances (≈0) to ±1510…2000. That enlarges
the target range the CNN learns to span, which can change its output distribution.
The engine then feeds that output through a **fixed** `tanh(x/200)` and multiplies
by a **fixed** `w[0] = 330.90` fitted for a different model.

Two distinct channels by which a Ridge refit could change engine behaviour:

- **(a) Changed ratios between the five coefficients.** Ranking within a position
  depends only on the *relative* weighting of the terms. If a refit shifts weight
  from `cnn_norm` toward `material`, the ordering can change.
- **(b) Changed overall scale relative to the unscaled terms.** The heuristic
  bonuses (±0.2–0.65) and the 1-ply lookahead term (±0.5) are added **after** the
  weighted sum and are **not** scaled by the Ridge. Uniformly shrinking all five
  coefficients would make those terms proportionally more influential.

Channel (b) is easy to overlook: a pure uniform rescale of the weighted score is
*not* ranking-neutral in this engine, precisely because of the unscaled additive
terms.

---

## 4. Existing evidence

From the A1 phase (reused, not rerun) — CNN outputs on 1,856 post-move positions
from `extended`:

| Model | mean abs output | p95 abs | max | `abs(tanh)>0.99` | `tanh` sd |
|---|---:|---:|---:|---:|---:|
| A0 s0 / s1 / s2 | 203.3 / 124.8 / 125.6 | 1140.9 / 586.5 / 532.7 | 2139 / 1025 / 1342 | 14.87% / 5.93% / 5.17% | 0.415 / 0.467 / 0.453 |
| A1 s0 / s1 / s2 | 181.2 / 123.6 / 169.6 | 986.9 / 459.3 / 735.8 | 1795 / 971 / 1343 | 10.72% / 3.72% / 8.51% | 0.454 / 0.509 / 0.505 |

**A1's raw outputs are not systematically larger**, and A1 saturates `tanh` *less*
on two of three seeds. The tanh-saturation hypothesis was tested and rejected in
A1. The seed-to-seed spread within each arm is comparable to the difference
between arms.

---

## 5. Output-scale analysis — what actually drives ranking

Absolute output scale is the wrong quantity. Ranking depends on score
**differences between candidate moves within one position**, so the relevant
measure is how much spread each weighted term contributes *across candidates*.

Measured (read-only) on 80 `extended` positions — **zero overlap with either
dataset split** — as the median across positions of the per-term standard
deviation across that position's candidate moves:

| Model | `330.90·tanh` | `32.39·mat` | `0.79·space` | `5.17·center` | `0.019·mob` | CNN share |
|---|---:|---:|---:|---:|---:|---:|
| A0 seed 0 | 50.675 | 5.960 | 1.508 | 2.939 | 0.083 | **82.8%** |
| A0 seed 1 | 59.116 | 5.960 | 1.508 | 2.939 | 0.083 | **84.9%** |
| A0 seed 2 | 58.491 | 5.960 | 1.508 | 2.939 | 0.083 | **84.8%** |
| A1 seed 0 | 62.690 | 5.960 | 1.508 | 2.939 | 0.083 | **85.7%** |
| A1 seed 1 | 84.676 | 5.960 | 1.508 | 2.939 | 0.083 | **89.0%** |
| A1 seed 2 | 77.288 | 5.960 | 1.508 | 2.939 | 0.083 | **88.0%** |

Findings:

- **The four board terms are numerically identical across all six models**
  (5.960 / 1.508 / 2.939 / 0.083), confirming question 7: the CNN term is the only
  feature the arm change touches.
- **The CNN term dominates ranking in both arms** — 82.8–84.9% of total term
  spread for A0, 85.7–89.0% for A1.
- **A1's CNN term contributes *more* ranking spread than A0's** (62.7–84.7 vs
  50.7–59.1), i.e. A1's CNN dominates the fusion *more*, not less.
- The A0↔A1 difference (≈ +3 to +4 percentage points of CNN share) is **of similar
  size to the within-arm seed spread** (A0 spans 82.8–84.9%).

**Interpretation:** the confound is real but the measured scale shift is modest,
and it moves in the direction of *more* CNN dominance. Since `w[0]` already
accounts for ~85% of ranking spread, a refit would have to change the coefficient
**ratios** substantially — not merely rescale — to alter rankings materially.
Whether it does is exactly what the diagnostic must establish, and it is cheaply
checkable before any engine evaluation (§8, Stage 1).

---

## 6. Leakage analysis

| Data | Available? | Notes |
|---|---|---|
| Ridge features | **Yes** — computable from any CNN + `dataset_v1` | — |
| Ridge target | **Yes** — `dataset_v1` raw Stockfish values, under a stated label policy | — |
| Original Ridge target (`sf_evaluations`) | **No** — never persisted, unrecoverable | Production Ridge cannot be reproduced |
| Original CNN | Yes (`models/cnn_model.keras`) | But without its labels, no reproduction is possible |

### Split composition (verified)

`dataset_v1`: 7,734 train / 1,933 test, split seed 42, identical for every arm.

| Split | White stm | Black stm | mate records |
|---|---:|---:|---:|
| train | 7,407 | 327 | 153 |
| test | 1,853 | **80** | **35** |

### Evaluation-suite overlap (verified)

| Suite | in dataset TRAIN | in dataset TEST |
|---|---|---|
| **`extended` (primary)** | **0** | **0** |
| `phase0_52` | 5 (`OP02`,`OP03`,`OP05`,`OP06`,`OP17`) | 1 (`OP04`) |

**`extended` has zero overlap with either split**, so fitting a Ridge on *either*
split leaks nothing into the primary evaluation suite. This is the single most
important fact enabling the diagnostic.

### Where to fit the refit Ridge

| Option | Leakage | Verdict |
|---|---|---|
| **Fit on the 7,734 train split** | The arm's CNN trained on these positions, so `cnn_norm` is **in-sample** and optimistically accurate. Ridge would over-weight the CNN term — the same flaw the production Ridge has | **Reject.** Reproduces the original defect |
| **Fit on the 1,933 test split** | CNN outputs are genuinely **out-of-sample**. No leakage into `extended`. Touches 1 of 52 `phase0_52` positions (`OP04`) | **Preferred**, with the `OP04` caveat disclosed |
| Fit on evaluation positions | Direct leakage into the metric being reported | **Reject outright** |

**Cost of the preferred option:** it consumes the only clean holdout, so there is
no independent set left to validate the refit's own generalisation. Mitigation: an
inner split of the 1,933 (e.g. 80/20 by FEN, fixed seed) to report the refit
Ridge's own held-out R², purely as a sanity check. The final refit for engine use
should then be fitted on all 1,933 with the inner-split result reported alongside.

**Thin strata warning:** the test split holds only **80 Black-to-move** and **35
mate** records. A refit is therefore poorly constrained on exactly the subgroups
A1 changed. This is a genuine limitation, not a fixable one within this dataset.

### Which target should the refit use?

To preserve A1's isolation, the refit target must be the **arm's own label
policy** — A1's corrected-mate, side-to-move-relative labels; A0's legacy labels.
Using a common target across arms would reintroduce the mismatch the diagnostic is
meant to remove.

Consequence: the refit changes **both** the features and the target relative to
production. That is correct for the diagnostic — it makes each arm's pipeline
internally consistent — but it means A1R does not isolate "output scale" alone; it
isolates **"fusion layer matched to the arm" vs "production fusion layer"**. That
distinction must be stated in any result.

---

## 7. Possible diagnostic designs

### Design 1 — A1 only, production vs refit Ridge

```
A1 CNN + production Ridge   (already measured)
A1 CNN + A1-refit Ridge     (new: 3 engine evals)
```

Cheapest. Answers "does the refit change A1's behaviour?" but **cannot attribute
the A0→A1 engine difference**, because A0 would still be measured under the
production Ridge. If the refit changes A1, we would not know whether it would
equally change A0.

### Design 2 — 2×2, both arms under both fusion layers  *(recommended)*

| | production Ridge | refit Ridge |
|---|---|---|
| **A0** | done (A0 report) | **new: 3 evals** |
| **A1** | done (A1 report) | **new: 3 evals** |

Six new engine evaluations, no retraining. Lets the A0→A1 difference be re-measured
with the fusion layer matched to each arm, which is exactly what removes the
confound.

### Fairness to A0 — why the A0 refit arm is required

If only A1 gets a refit, any change is unattributable: a refit Ridge is fitted on
**out-of-sample** CNN outputs while the production Ridge was fitted partly
**in-sample**. That methodological difference alone could shift results for *any*
CNN, A0 included. **The A0 refit control is what separates "refitting helps" from
"refitting helps A1 specifically".** Design 2 is the minimum defensible design.

### Option A vs Option B — one shared Ridge, or one per seed?

| | Description | Assessment |
|---|---|---|
| **Option A** | One Ridge fitted across all three seeds' CNN outputs | **Reject.** The three CNNs have different output distributions; a single `w[0]` fitted to their mixture is coherent for none of them, and it breaks the A0↔A1 seed pairing |
| **Option B** | One Ridge per (arm, seed) | **Recommended.** Each fusion layer matches the CNN it serves, and the paired structure `A0 sN ↔ A1 sN` is preserved |

Option B's cost: the Ridge fit becomes a second source of per-seed variance, so the
refit arms' spread is not directly comparable to A0's pure-training noise floor.
This must be stated, and it is why the A0-refit arm doubles as the noise reference
for the refit condition.

---

## 8. Recommended design

**Outcome A — a clean refit is possible**, with the constraints in §6.

Two stages, with a cheap gate so the expensive stage only runs if it can matter.

### Stage 1 — coefficient inspection (minutes, no engine evaluation)

For each of the six models (A0 ×3, A1 ×3):

1. Compute `cnn_norm` on the **1,933 test-split positions** using that model.
2. Assemble `[cnn_norm, material, space, center, mobility]` — features on the
   position itself, matching how the production Ridge was fitted.
3. Fit `Ridge(alpha=1.0)` against **that arm's own label policy** on those
   positions.
4. Report the six coefficient vectors, their ratios normalised to `w[0]`, the
   implied per-term ranking spread (as in §5), and the inner-split R².

**Gate:** if every refit coefficient vector is close to a positive scalar multiple
of the production vector **and** the implied per-term ranking shares are within the
A0 seed spread, then the refit cannot materially change rankings through channel
(a), and only channel (b) — overall scale versus the unscaled bonus/lookahead
terms — remains. Report that and **stop**; do not spend the engine evaluations.

### Stage 2 — engine evaluation (only if Stage 1 shows a material change)

Six runs of the **unmodified** Phase 3 evaluator via the existing
`CHESS_BOT_MODELS_DIR` staging mechanism, with the staged directory holding the
arm's CNN **and its refit `weight_model.pkl`** instead of the production one.

- Primary suite `extended` (zero overlap). Secondary `phase0_52`, with the `OP04`
  caveat disclosed.
- Everything else identical: Stockfish 17.1, depth 8, Threads 1, Hash 16 MB,
  Clear Hash per position, same engine code, same evaluator.

**Estimated cost:** Stage 1 ≈ 5 minutes. Stage 2 ≈ 70–80 minutes
(6 × ~11 min on `extended`, plus ~2.5 min each if `phase0_52` is included).

### Implementation notes (not yet written)

- New code should live in a diagnostic module, e.g. `training/refit_ridge.py`,
  writing to `training/experiments/A1R/<arm>_seed_<n>/` (already gitignored).
- `evaluate_arm.py` currently hardcodes copying the **production**
  `weight_model.pkl` into the staging directory. A refit run needs an optional
  override for that file. This is a **small additive change to experiment-only
  code**, not to production — but it is a change to a file A0 and A1 used, so it
  must be additive and default-preserving, and the existing tests that assert
  staging copies the production Ridge must continue to pass.
- No production file, no A0/A1 artifact, and no evaluator behaviour may change.

---

## 9. Required data and artifacts

Everything needed **already exists**. Nothing must be generated or obtained:

- `training/artifacts/dataset_v1.jsonl` + manifest (9,667 records, checksum-verified)
- `training/experiments/A0/seed_{0,1,2}/models/cnn_model.keras`
- `training/experiments/A1/seed_{0,1,2}/models/cnn_model.keras`
- `models/weight_model.pkl` (production reference, read-only)
- `evaluation/positions/{extended,phase0_52}.json`
- The unmodified Phase 3 evaluator and a Stockfish binary

**Nothing is missing.** This is Outcome A, not Outcome B.

---

## 10. Expected metrics

**Stage 1:** six coefficient vectors; ratios normalised to `w[0]`; per-term ranking
spread and CNN share (comparable to §5); refit inner-split R²; cosine similarity
between each refit vector and the production vector.

**Stage 2:** the standard Phase 3 set, unchanged — legality, top-1 agreement, top-3
containment, mean/median/p95 regret, blunder rate >300cp, regret coverage,
Spearman, mate statuses, White/Black breakdown. Plus paired deltas in the existing
A0↔A1 style, and the re-measured A0→A1 difference **under matched fusion**.

---

## 11. Interpretation rules

Fixed in advance, to prevent post-hoc reasoning:

1. **Three seeds. No significance testing.** Report ranges and paired directions.
2. **The A0-refit arm is the reference for the refit condition**, not A0-production.
   Refitting changes the fitting methodology (out-of-sample vs partly in-sample),
   so refit arms must be compared to each other.
3. **The headline question is whether the A0→A1 engine difference survives matched
   fusion.** If A1's regret increase persists under refit Ridge, the production
   Ridge is not the explanation. If it disappears, the confound was material and
   A1's engine-level conclusion must be re-stated.
4. **A1R cannot isolate output scale alone** — the refit changes features *and*
   target. Any result describes "arm-matched fusion vs production fusion".
5. **A1R does not fix the objective mismatch** (§2, property 2): a Ridge fitted for
   between-position accuracy is still being used for within-position ranking. A
   refit narrows the scale mismatch, not this one.
6. **Do not treat A1R as a new primary arm.** It cannot change A0's or A1's
   recorded results; it only changes how they are interpreted.
7. **Mate statuses remain underpowered** (7–8 mate positions of 160, ±1 seed
   noise). Do not read mate-count changes as evidence.

---

## 12. Limitations

1. **The production Ridge cannot be reproduced** — its target labels are
   unrecoverable. A1R compares against it as an artifact, never reconstructs it.
2. **The refit consumes the only clean holdout.** Fitting on the 1,933 test split
   leaves no independent set; the inner-split R² is a sanity check, not validation.
3. **Thin strata:** only 80 Black-to-move and 35 mate records in the test split —
   precisely the subgroups A1 alters.
4. **Methodological asymmetry:** refit Ridges are fitted out-of-sample while the
   production Ridge was fitted partly in-sample. This is why the A0-refit control
   is mandatory.
5. **`OP04`** (1 of 52 `phase0_52` positions) sits in the refit fitting set. Zero
   impact on `extended`.
6. **The measured scale shift is modest** (§5: CNN ranking share 82.8–84.9% for A0
   vs 85.7–89.0% for A1, against within-arm seed spread of ~2 pp). A null result is
   a plausible and legitimate outcome.
7. **The intercept remains unused** (defect C3). Refit Ridges will also have their
   intercept dropped by `engine.py`, so the refit is fitted with an intercept the
   engine will discard — a pre-existing inconsistency A1R inherits rather than
   fixes.
8. **Three seeds per cell**, six cells.

---

## 13. Should A2 wait for this diagnostic?

**Recommendation: yes — run A1R (at least Stage 1) before A2.**

Reasoning:

- **A2 changes the CNN's target semantics more than A1 did** (perspective
  normalisation flips sign on Black-to-move labels). It therefore inherits the
  *same* Ridge confound, and more strongly. Running A2 first would add a second
  uninterpreted result on top of an uninterpreted one.
- **A1's engine-level finding is currently unattributable.** The consistent regret
  increase across three paired seeds is the most interesting result of the
  programme so far, and we cannot presently say whether it reflects the labels or
  the fusion mismatch. A2's interpretation depends on that answer.
- **Stage 1 is cheap (~5 minutes) and may settle it.** If the refit coefficients
  are near-proportional to production, the confound is bounded and A2 can proceed
  under the existing fixed-Ridge protocol with a documented justification.

**The counter-argument, stated fairly:** A0, A1 and A2 under a *fixed* Ridge form a
consistent protocol, and switching fusion layers mid-programme introduces its own
comparability problem. If the priority is completing the planned arms on schedule,
running A2 with the production Ridge and then doing **one** refit diagnostic
covering A0/A1/A2 together is defensible — and cheaper in total, since it amortises
the refit work across three arms.

**On balance:** run **Stage 1 only** now. It is inexpensive, it either bounds the
confound or demonstrates it matters, and either answer improves how A2 is designed
and read. Defer Stage 2 until Stage 1 says it is warranted.

This is a recommendation, not a decision — the schedule tradeoff is yours.
