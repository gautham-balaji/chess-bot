# C6-A2R Stage 1 — Ridge Compatibility Diagnostic for A2 (results)

**Stage 1 only. The matched-Ridge Stage 2 was NOT run. A3 was not started.**
No CNN was trained, no production file was modified, nothing was committed.

Design reused verbatim from [`C6_A1R_RIDGE_DIAGNOSTIC_DESIGN.md`](C6_A1R_RIDGE_DIAGNOSTIC_DESIGN.md).
Prior stages: [Stage 1](C6_A1R_STAGE1_REPORT.md) · [Stage 2](C6_A1R_STAGE2_REPORT.md) · [A2](C6_A2_REPORT.md)
Repository state: `43862ac`.

---

## Gate decision (up front)

> **GATE 1 (vs production): TRIPPED. GATE 2 (vs A0/A1): CLEAR.**
>
> **An A2 matched-Ridge Stage 2 is NOT warranted. Proceed directly to A3.**

A2's refit coefficients are **qualitatively unlike any arm measured so far** — it
is the only arm whose refit keeps `space` positive, and the only one that drives
`center` and `mobility` negative. But a changed coefficient vector is not the
question. On every quantity that bears on **move ranking**, A2 lands inside the
envelope A0 and A1 already established:

| ranking-relevant quantity | A0 | A1 | **A2** | A2 vs others |
|---|---|---|---|---|
| refit CNN ranking share % | 45.66–54.95 | 30.98–34.50 | **47.59–50.37** | **inside A0** |
| CNN decisive % (refit coef) | 41.25–53.75 | 36.25–47.50 | **32.50–36.25** | touches A1 |
| top move changed by refit % | 28.75–32.50 | 70.00–72.50 | **27.50–36.25** | **inside A0** |

A2 behaves like A0, not like A1. **A1R Stage 2 already ran the A0-matched cell**
— it is one of the four cells of the 2×2 — so the matched-fusion behaviour of an
arm with A2's profile has already been measured. Re-running it for A2 would
re-answer a question the programme has answered.

---

## 1. Methodology

Identical to A1R Stage 1. The existing diagnostic was **extended, not replaced**:
`training/refit_ridge.py` gained optional `--arms`, `--seeds`, `--out-dir` and
`--stage` arguments. Run with no arguments it still fits exactly A0 and A1 and
still writes `training/experiments/A1R/stage1_results.json`, which A1R Stage 2
reads — a test pins that default.

For each of the nine trained models (A0 ×3, A1 ×3, A2 ×3):

1. **Fit positions:** the 1,933 `dataset_v1` test-split records (split seed 42),
   out-of-sample for every model.
2. **Features:** `[cnn_norm, material, space, center, mobility]` with
   `cnn_norm = tanh(cnn(position)/200)`, computed on the position itself,
   matching notebook cell 31.
3. **Target:** each arm's own label policy — A0 `legacy_notebook`, A1
   `corrected_mate_legacy_perspective`, A2 `corrected_mate_white_perspective`.
4. **Fit:** `Ridge(alpha=1.0)`.
5. **Sanity check:** inner 80/20 split, seed 1234; reported coefficients come
   from the final fit on all 1,933.
6. **Ranking spread:** 80 `extended` positions (2,701 candidate moves), features
   on **post-move** positions, matching `rerank_moves`. `extended` has zero
   overlap with either dataset split.

**Two diagnostics added for A2R**, because A1R Stage 1's own limitation #1 was
that coefficient change does not demonstrate ranking change:

- **CNN decisive %** — how often the CNN term changes which candidate ranks
  first, relative to the four board terms alone.
- **Top move changed %** — how often the refit coefficients rank a *different*
  candidate first than the production coefficients do.

Both respect the side to move (the engine scores White-positive and encodes the
side in the sort direction, so the mover's best candidate is the max for White
and the min for Black).

> **Both are proxies, not the engine's choice.** `rerank_moves` adds unscaled
> heuristic bonuses and a 1-ply lookahead term on top of the weighted sum, so a
> proxy disagreement neither guarantees nor excludes an engine disagreement.
> They are nonetheless far tighter evidence about rankings than coefficients are,
> which is the gap A1R Stage 1 flagged and could not close.

### Reproduction check

The A0 and A1 rows were refitted from scratch rather than copied. They reproduce
the committed A1R Stage 1 coefficients **exactly**:

```
6 A0/A1 runs re-fitted; max |Δcoef| = 0.000e+00  -> EXACT
```

This also confirms the production Ridge reference vector is being read
identically to when A1R ran, despite the pre-existing
`InconsistentVersionWarning` (the pickle was written with scikit-learn 1.7.1; the
environment now has 1.8.0).

---

## 2. Coefficient vectors

| model | `cnn_norm` | `material` | `space` | `center` | `mobility` | intercept |
|---|---:|---:|---:|---:|---:|---:|
| **production** | **330.9005** | **32.3899** | **+0.7919** | **+5.1654** | **+0.0188** | 15.0772 |
| A0 seed 0 | 197.3612 | 41.8343 | −3.4932 | +14.2285 | +3.0745 | 52.7727 |
| A0 seed 1 | 197.5795 | 40.4798 | −2.7576 | +13.5918 | +2.8425 | 60.0146 |
| A0 seed 2 | 200.9706 | 40.6948 | −2.3270 | +10.3456 | +2.5740 | 39.6432 |
| A1 seed 0 | 245.4067 | 37.5425 | −18.5303 | +14.5807 | +12.0704 | 37.6215 |
| A1 seed 1 | 209.3882 | 37.1902 | −16.9606 | +15.1350 | +12.2764 | 46.5361 |
| A1 seed 2 | 208.1213 | 38.7785 | −17.5500 | +15.8633 | +12.5012 | 42.5777 |
| **A2 seed 0** | **263.2468** | **46.9059** | **+15.8569** | **−2.6374** | **−4.2915** | 25.3621 |
| **A2 seed 1** | **260.7108** | **47.6721** | **+16.3671** | **−2.4401** | **−4.5923** | 44.2649 |
| **A2 seed 2** | **277.4567** | **46.5877** | **+16.2640** | **−2.0181** | **−4.0322** | 46.9629 |

**Cross-seed range (max − min) per arm:**

| arm | `cnn_norm` | `material` | `space` | `center` | `mobility` |
|---|---:|---:|---:|---:|---:|
| A0 | 3.6093 | 1.3545 | 1.1663 | 3.8829 | 0.5006 |
| A1 | **37.2854** | 1.5883 | 1.5697 | 1.2826 | 0.4308 |
| A2 | 16.7459 | 1.0844 | **0.5101** | **0.6192** | 0.5601 |

A2 is the **most seed-stable arm on the three small coefficients** and sits
between A0 and A1 on `cnn_norm`.

### Sign pattern — where A2 is genuinely unique

| coefficient | production | A0 refits | A1 refits | **A2 refits** |
|---|---|---|---|---|
| `space` | **+** | − (all 3) | − (all 3) | **+ (all 3)** |
| `center` | **+** | + (all 3) | + (all 3) | **− (all 3)** |
| `mobility` | **+** | + (all 3) | + (all 3) | **− (all 3)** |

**A2 is the only arm that agrees with production on `space`**, and the only arm
that disagrees with production on `center` and `mobility`. The unexplained
`space` sign flip that A1R Stage 1 reported in all six of its refits **does not
occur for A2**.

This is a real structural difference. Section 5 shows it does not translate into
a ranking difference.

---

## 3. Ratios normalised to `cnn_norm`

Ranking depends only on relative weighting, so these — not raw magnitudes — are
what can reorder moves.

| model | `material` | `space` | `center` | `mobility` |
|---|---:|---:|---:|---:|
| **production** | 0.097884 | +0.002393 | +0.015610 | +0.000057 |
| A0 seed 0 | 0.211968 | −0.017700 | +0.072094 | +0.015578 |
| A0 seed 1 | 0.204879 | −0.013957 | +0.068792 | +0.014387 |
| A0 seed 2 | 0.202491 | −0.011579 | +0.051478 | +0.012808 |
| A1 seed 0 | 0.152981 | −0.075509 | +0.059414 | +0.049185 |
| A1 seed 1 | 0.177614 | −0.081001 | +0.072282 | +0.058630 |
| A1 seed 2 | 0.186327 | −0.084326 | +0.076222 | +0.060067 |
| **A2 seed 0** | 0.178182 | **+0.060236** | **−0.010019** | **−0.016302** |
| **A2 seed 1** | 0.182854 | **+0.062779** | **−0.009359** | **−0.017614** |
| **A2 seed 2** | 0.167910 | **+0.058618** | **−0.007274** | **−0.014533** |

**Multiple of the production ratio (sign preserved):**

| model | `material` | `space` | `center` | `mobility` |
|---|---:|---:|---:|---:|
| A0 seed 0 / 1 / 2 | 2.17 / 2.09 / 2.07 | −7.40 / −5.83 / −4.84 | 4.62 / 4.41 / 3.30 | 274 / 253 / 225 |
| A1 seed 0 / 1 / 2 | 1.56 / 1.81 / 1.90 | −31.6 / −33.9 / −35.2 | 3.81 / 4.63 / 4.88 | 864 / 1030 / 1056 |
| **A2 seed 0 / 1 / 2** | 1.82 / 1.87 / 1.72 | **+25.2 / +26.2 / +24.5** | **−0.64 / −0.60 / −0.47** | **−287 / −310 / −255** |

**Ratio bands are fully disjoint between A2 and the other two arms** on `space`,
`center` and `mobility`:

| ratio | A0 | A1 | **A2** | disjoint from both? |
|---|---|---|---|---|
| `material` | 0.20249…0.21197 | 0.15298…0.18633 | 0.16791…0.18285 | no (inside A1) |
| `space` | −0.01770…−0.01158 | −0.08433…−0.07551 | **+0.05862…+0.06278** | **yes** |
| `center` | +0.05148…+0.07209 | +0.05941…+0.07622 | **−0.01002…−0.00727** | **yes** |
| `mobility` | +0.01281…+0.01558 | +0.04918…+0.06007 | **−0.01761…−0.01453** | **yes** |

So on the coefficient balance itself, **A2 is a distinctly different fusion**.
That is the finding the gate must not over-read.

---

## 4. Cosine, scale, R², and the cosine trap

| model | cosine → prod | scale `w₀`/prod | R² train | R² holdout |
|---|---:|---:|---:|---:|
| A0 seed 0 | 0.992017 | 0.5964 | 0.3831 | 0.3901 |
| A0 seed 1 | 0.992993 | 0.5971 | 0.3817 | 0.4001 |
| A0 seed 2 | 0.994009 | 0.6073 | 0.3888 | 0.4156 |
| A1 seed 0 | 0.993505 | 0.7416 | 0.4000 | 0.3790 |
| A1 seed 1 | 0.990457 | 0.6328 | 0.3840 | 0.3697 |
| A1 seed 2 | 0.989226 | 0.6290 | 0.3893 | 0.3479 |
| **A2 seed 0** | **0.994840** | 0.7955 | 0.4455 | 0.3795 |
| **A2 seed 1** | **0.994329** | 0.7879 | 0.4450 | 0.3940 |
| **A2 seed 2** | **0.995747** | 0.8385 | 0.4486 | 0.3890 |

### The cosine trap, demonstrated rather than asserted

**A2 has the highest cosine similarity to production of any of the nine refits
(0.9943–0.9957) while having the most divergent sign pattern.** Reading cosine
alone would rank A2 the *most* compatible arm. It is, structurally, the least
like the others.

Measured perturbation sensitivity of the production vector (now recorded as an
artifact, not just a remark):

| perturbation | cosine |
|---|---:|
| `material` ×10 | 0.779404 (detected) |
| `center` ×10 | 0.990411 (barely) |
| `space` ×10 | 0.999770 (blind) |
| `space` **sign flip** | **0.999989 (blind)** |
| `mobility` ×10 | 1.000000 (blind) |
| `mobility` ×100 | 0.999984 (blind) |
| uniform ×10 | 1.000000 (blind by construction) |

Cosine is **essentially blind to a sign flip of `space`** — precisely the change
that distinguishes these arms. Per instruction, cosine carries no weight in the
gate; the gate rests on the normalised ratios and on ranking behaviour.

### Scale — channel (b)

A2's `w₀` is **0.788–0.839×** production, distinct from A0 (0.596–0.607) and A1
(0.629–0.742). A2 is therefore the arm whose refit is **closest to production in
scale**, so the second channel the design document identified — a smaller
weighted score letting the unscaled heuristic bonuses (±0.2–0.65) and the 1-ply
lookahead term (±0.5) compete more strongly — is **weakest for A2**, not
strongest. A2 is distinct here, but in the benign direction.

### R²

A2 has the highest **train** R² (0.4450–0.4486) but a **holdout** R²
(0.3795–0.3940) comparable to A0 (0.3901–0.4156) and better than A1
(0.3479–0.3790). The train→holdout drop is larger for A2 (≈0.06) than for A0
(≈0.00), so A2's fusion fits its own labels better without generalising better.
All nine refits explain roughly a third to two-fifths of label variance
out-of-sample; none is a strong predictor.

---

## 5. Ranking share and decisive contribution

| model | CNN share (prod coef) | CNN share (refit coef) | shift | CNN decisive % (refit) | **top move changed by refit %** |
|---|---:|---:|---:|---:|---:|
| A0 seed 0 | 82.85% | 45.66% | −37.19 | 53.8% | **30.0%** |
| A0 seed 1 | 84.93% | 51.73% | −33.20 | 41.2% | **32.5%** |
| A0 seed 2 | 84.79% | 54.95% | −29.84 | 52.5% | **28.8%** |
| A1 seed 0 | 85.67% | 30.98% | −54.68 | 36.2% | **71.2%** |
| A1 seed 1 | 88.98% | 34.50% | −54.48 | 47.5% | **72.5%** |
| A1 seed 2 | 88.05% | 31.74% | −56.31 | 40.0% | **70.0%** |
| **A2 seed 0** | 86.56% | 47.59% | −38.97 | 36.2% | **36.2%** |
| **A2 seed 1** | 88.31% | 50.37% | −37.94 | 32.5% | **27.5%** |
| **A2 seed 2** | 86.36% | 48.80% | −37.56 | 33.8% | **36.2%** |

The decisive quantity is the last column. On the same 80 positions:

- **A1's refit would rank a different move first on 70.0–72.5% of positions.**
- **A2's would on 27.5–36.2% — statistically the same as A0's 28.8–32.5%.**

A2's fusion mismatch, measured where it matters, is an **A0-sized** mismatch, not
an A1-sized one. Under production coefficients A2's CNN share (86.36–88.31%)
overlaps A1's (85.67–88.98%) and exceeds A0's — yet A2's *engine* results were
fine while A1's regressed. That is consistent with the A2 report's conclusion
that A1's regression was caused by unlearnable labels, not by the fusion.

---

## 6. The two gates

### GATE 1 — absolute, identical to A1R

> *Is every refit coefficient vector close to a positive scalar multiple of the
> production vector, **and** are the implied ranking shares within the A0 seed
> spread (82.85–84.93%)?*

| condition | A2 result |
|---|---|
| Near-proportional to production? | **NO.** `center` and `mobility` flip sign in all three seeds; max \|ratio multiple\| is 310× |
| Ranking shares within the A0 band? | **NO.** A2 refit shares are 47.59–50.37%, i.e. 32–37 pp below the band |

**GATE 1: TRIPPED.**

This is not news. GATE 1 trips for **every arm**, including the A0 control. It
established in A1R that the production Ridge over-weights the CNN because it was
fitted partly in-sample. It does not distinguish A2 from anything.

### GATE 2 — relative, A2R-specific

> *Is A2 outside the envelope A0 and A1 already established, on a quantity that
> bears on ranking?*

| quantity | A2 band | verdict |
|---|---|---|
| refit CNN ranking share % | 47.59–50.37 | overlaps A0 (45.66–54.95) |
| CNN decisive % (refit) | 32.50–36.25 | touches A1 (36.25–47.50) |
| top move changed by refit % | 27.50–36.25 | overlaps A0 (28.75–32.50) |
| — *non-ranking quantities* | | |
| scale factor `w₀`/prod | 0.7879–0.8385 | **distinct from both** (benign direction) |
| R² holdout | 0.3795–0.3940 | overlaps A0 |
| CNN decisive % (prod coef) | 46.25–50.00 | overlaps A0 |

**Ranking-relevant quantities distinct from both A0 and A1: none.**

**GATE 2: CLEAR.**

Touching bands are counted as overlapping, which is the conservative choice — it
makes GATE 2 *harder* to trip, so the gate cannot be tripped by a boundary
coincidence. A test pins that behaviour.

---

## 7. Is an A2 matched-Ridge Stage 2 warranted?

**No.** Four reasons, in order of weight:

1. **A2's ranking behaviour is A0's, and A0-matched has already been measured.**
   A1R Stage 2 was a 2×2: A0 and A1, each under production and matched fusion.
   The A0-matched cell is exactly the analogue of what an A2 Stage 2 would
   measure. There, matching the Ridge **improved** A0's mean regret by 11.63 cp —
   a modest, benign effect. A2's proxy profile predicts the same.
2. **The question Stage 2 exists to answer has been answered.** A1R Stage 2
   asked whether fusion mismatch explained an arm's engine result. It did not —
   the A0→A1 gap *widened* from +16.6 to +36.8 cp under matched fusion. There is
   no live hypothesis that A2's engine numbers are a fusion artifact.
3. **Channel (b) is weakest for A2.** Its `w₀` is the closest to production of
   any refit (0.79–0.84× vs 0.60–0.74×), so the unscaled bonus and lookahead
   terms are least disturbed.
4. **Cost/benefit.** Stage 2 is ~70–80 minutes of Stockfish evaluation to
   re-measure a cell whose analogue is known and whose predicted effect is small.

**The honest counter-argument.** A2's coefficient *balance* is disjoint from both
other arms on three of four ratios, and the sign pattern is unique. It is
logically possible that a fusion which is A0-like in aggregate ranking statistics
is nonetheless different on the specific positions that drive regret — the proxy
measures *whether* the top move changes, not *how much worse* the new move is,
and the two need not correlate. Stage 1 cannot exclude that. What tips the
balance is reason 2: even if A2's matched fusion did shift regret, there is no
longer a hypothesis it would test, because A1R Stage 2 already removed fusion
mismatch as the explanation for the programme's central anomaly.

**Recommendation: proceed directly to A3.**

If A3 produces an engine result that is surprising in the way A1's was, the
correct response is a matched-Ridge run covering A3 — and A2 could be folded into
that run cheaply at the same time.

---

## 8. Limitations

1. **Stage 1 measures the fusion layer, not the engine.** The top-move proxy
   omits the unscaled heuristic bonuses and the 1-ply lookahead, so it neither
   guarantees nor excludes an engine-level change.
2. **The proxy counts changes, not their cost.** A 30% top-move change rate says
   nothing about whether the new moves are better or worse.
3. **Thin strata, unchanged from A1R.** The 1,933-position fit set holds only 80
   Black-to-move and 35 mate records — and those are exactly the records A2's
   label policy alters. **A2's refit is least constrained precisely where A2
   differs**, which is a sharper version of the A1R limitation.
4. **The refit consumes the only clean holdout.** The inner 80/20 split is a
   sanity check, not independent validation.
5. **`OP04`** (1 of 52 `phase0_52` positions) is in the fit set; `extended` has
   zero overlap.
6. **Refits change features *and* target**, so this isolates "arm-matched fusion
   vs production fusion", not output scale alone.
7. **The objective mismatch is unfixed** — a Ridge fitted for between-position
   accuracy is still used for within-position ranking, in every arm.
8. **Intercepts are discarded by the engine** (defect C3, open). A2's refit
   intercepts are 25.4–47.0 and would be thrown away.
9. **Three seeds per arm.** Ranges are descriptive; no significance testing.
10. **A2's unique sign pattern is unexplained.** It is reported, not accounted
    for. No mechanism was investigated and none is asserted.
11. **80 ranking positions**, the same subset A1R used, chosen for comparability
    rather than coverage.

---

## 9. Production safety

- `engine.py`, `app.py`, `config.py`, `evaluation/`, `models/`, and the
  baseline/regression fixtures are **unmodified** — `git status` reports no
  changes under any of them.
- `models/weight_model.pkl` was opened **read-only**; its coefficients still read
  `[330.9005, 32.3899, 0.7919, 5.1654, 0.0188]` and all five files in `models/`
  retain their original SHA-256 digests.
- No CNN was trained; no engine evaluation was run; no Stockfish process started.
- The A1R artifacts are untouched: A2R writes only to
  `training/experiments/A2R/`, and a test pins that its output paths differ from
  `A1R/stage1_results.json`, which A1R Stage 2 reads.
- `training/experiments/` is gitignored, so all results stay local.

---

## Reproducing

```bash
# Stage 1 refits for all three arms (~5 minutes, no Stockfish, no training)
python -m training.refit_ridge --arms A0 A1 A2 \
    --out-dir training/experiments/A2R --stage A2R-stage1

# cross-arm comparison and the two gates
python -m training.a2r_stage1_analysis
```

Artifacts (gitignored): `training/experiments/A2R/stage1_results.json` and
`training/experiments/A2R/stage1_analysis.json`.

The A1R default is preserved — `python -m training.refit_ridge` with no arguments
still fits A0 and A1 into `training/experiments/A1R/`.
