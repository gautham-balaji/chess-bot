# C6-A1R Stage 2 — Matched-Fusion Engine Diagnostic (results)

**Stage 2 complete. A2 not started. Nothing committed.**

No CNN was trained. `engine.py`, production models, evaluation positions,
baseline/regression fixtures and all A0/A1 historical artifacts are unchanged.

Design: [`C6_A1R_RIDGE_DIAGNOSTIC_DESIGN.md`](C6_A1R_RIDGE_DIAGNOSTIC_DESIGN.md)
Stage 1: [`C6_A1R_STAGE1_REPORT.md`](C6_A1R_STAGE1_REPORT.md)

---

## Answer to the central question

> **Does the A0→A1 engine-level difference survive when both arms use a Ridge
> matched to their own CNN and label policy?**

**Yes — and it roughly doubles.**

| `extended`, A0→A1 mean regret | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| Under **production** fusion | +22.15 | +6.13 | +21.59 | **+16.6 cp** |
| Under **matched** fusion | +41.73 | +30.90 | +37.68 | **+36.8 cp** |

Median regret behaves the same way (+10.8 → **+37.3 cp**), and on `phase0_52`
mean regret goes +12.8 → **+40.1 cp**. Metrics that were *mixed* under production
fusion become *consistent* under matched fusion: top-1 agreement A0→A1 goes from
−1.04 pp (mixed signs) to **−10.0 pp (all three seeds negative)**, and top-3
containment from −0.21 pp (mixed) to **−14.0 pp (all three negative)**.

**Per the design document's pre-registered interpretation rule: the A1 regret
increase persists under matched fusion, therefore the production Ridge mismatch
is not sufficient to explain the A1 regression.** The confound identified in the
A1 report is resolved — and resolved *against* the hypothesis that it was driving
the result.

---

## 1. Methodology

For each of the six trained CNNs (A0 ×3, A1 ×3):

1. **Matched Ridge reconstructed** from the exact Stage 1 `coef`/`intercept`
   values (Stage 1 recorded the numbers, not the estimator). A Ridge is fully
   determined by those, and `engine.py` reads only `coef_`, so the reconstruction
   is exact; the pickle round-trip is verified with `atol=0, rtol=0`.
2. **Staged** the arm/seed CNN plus its matched Ridge into a temporary directory
   via `CHESS_BOT_MODELS_DIR`.
3. **Verified before every evaluation** that the staged CNN hash equals the source
   CNN, the staged Ridge coefficients equal Stage 1 exactly, and the staged Ridge
   is **not** the production Ridge. All 18 checks passed.
4. **Evaluated** with the **unmodified** Phase 3 evaluator — Stockfish 17.1,
   depth 8, Threads 1, Hash 16 MB, Clear Hash per position, identical metrics and
   thresholds, identical position suites.
5. **Verified after every run** that `models/` was byte-identical.

Staged Ridge `coef[0]` per run, confirming each got its own: 197.3612, 197.5795,
200.9706 (A0 s0–s2) and 245.4067, 209.3882, 208.1213 (A1 s0–s2) — versus
production 330.9005.

**The only code change to the shared harness** was an additive, default-preserving
`ridge_model=None` parameter on `evaluate_arm.stage_models_dir`. When omitted it
copies the production Ridge exactly as before; a test pins that default, and the
existing A0/A1 staging tests still pass.

**Runtime:** 2,364.5 s (39.4 min) for all six runs across both suites.

---

## 2. Six run results — `extended` (n=160), primary

Production → matched, per arm/seed:

| Metric | A0 s0 | A0 s1 | A0 s2 | A1 s0 | A1 s1 | A1 s2 |
|---|---|---|---|---|---|---|
| Legality % | 100→100 | 100→100 | 100→100 | 100→100 | 100→100 | 100→100 |
| Top-1 agreement % | 19.38→15.62 | 20.62→19.38 | 16.88→20.00 | 17.50→**7.50** | 16.88→**8.75** | 19.38→**8.75** |
| Top-3 containment % | 33.75→28.75 | 33.12→31.88 | 33.75→35.00 | 36.88→**20.62** | 32.50→**18.12** | 30.62→**15.00** |
| Mean regret cp | 128.73→**120.78** | 140.71→**129.15** | 148.43→**133.04** | 150.88→162.51 | 146.84→160.05 | 170.02→170.72 |
| Median regret cp | 30→36.5 | 27→29 | 37→38 | 38.5→**70.0** | 38→**70.5** | 50→**75.0** |
| p95 regret cp | 530→529 | 573→565 | 594→533 | 580→578 | 569→541 | 637→565 |
| Max regret cp | 678→678 | 678→678 | 696→696 | 694→**1218** | 784→**1218** | 758→**1218** |
| Blunder rate >300cp | 0.2105→0.1842 | 0.2288→0.2092 | 0.2500→0.2185 | 0.2566→0.1987 | 0.2368→0.2303 | 0.2810→0.2318 |
| Regret coverage % | 95.00→95.00 | 95.62→95.62 | 95.00→94.38 | 95.00→94.38 | 95.00→95.00 | 95.62→94.38 |
| Spearman mean | 0.21→0.22 | 0.21→0.21 | 0.22→0.20 | 0.17→**0.08** | 0.18→**0.14** | 0.22→**0.11** |
| Spearman median | 0.25→0.31 | 0.29→0.24 | 0.25→0.26 | 0.24→**0.11** | 0.24→0.19 | 0.27→0.19 |

**Legality stayed 100% in all twelve cells.**

---

## 3. Production → matched, paired per arm (`extended`)

### A0 — the refit *helps*

| Metric | s0 | s1 | s2 | mean | consistent? |
|---|---:|---:|---:|---:|---|
| Mean regret cp | −7.95 | −11.56 | −15.39 | **−11.63** | all 3 lower |
| Median regret cp | +6.5 | +2 | +1 | +3.17 | all 3 higher |
| p95 regret cp | −1 | −8 | −61 | −23.33 | all 3 lower |
| Blunder rate | −0.0263 | −0.0196 | −0.0315 | **−0.0258** | all 3 lower |
| Top-1 % | −3.76 | −1.24 | +3.12 | −0.63 | mixed |
| Spearman mean | +0.01 | 0 | −0.02 | −0.003 | mixed |

### A1 — the refit *hurts*

| Metric | s0 | s1 | s2 | mean | consistent? |
|---|---:|---:|---:|---:|---|
| Mean regret cp | +11.63 | +13.21 | +0.70 | **+8.51** | all 3 higher |
| Median regret cp | +31.5 | +32.5 | +25 | **+29.67** | all 3 higher |
| Max regret cp | +524 | +434 | +460 | **+472.7** | all 3 higher |
| Top-1 % | −10.00 | −8.13 | −10.63 | **−9.59** | all 3 lower |
| Top-3 % | −16.26 | −14.38 | −15.62 | **−15.42** | all 3 lower |
| Spearman mean | −0.09 | −0.04 | −0.11 | **−0.08** | all 3 lower |
| Blunder rate | −0.0579 | −0.0065 | −0.0492 | −0.0379 | all 3 lower |

**The matched Ridge improves A0 and degrades A1** — the opposite of what the
confound hypothesis predicted.

Note the divergence between median regret (**+29.7**, worse) and blunder rate
(**−0.038**, better) for A1: the matched fusion moved mass into the middle of the
distribution while trimming the >300 cp tail. Mean, median and tail metrics do not
agree here, and reporting only one would misrepresent the change.

---

## 4. A0 → A1 under each fusion

### `extended` (n=160)

| Metric | Production (s0/s1/s2, mean) | Matched (s0/s1/s2, mean) |
|---|---|---|
| Top-1 agreement % | −1.88 / −3.74 / +2.50 → **−1.04** (mixed) | −8.12 / −10.63 / −11.25 → **−10.00** (all −) |
| Top-3 containment % | +3.13 / −0.62 / −3.13 → **−0.21** (mixed) | −8.13 / −13.76 / −20.00 → **−14.00** (all −) |
| **Mean regret cp** | +22.15 / +6.13 / +21.59 → **+16.6** | +41.73 / +30.90 / +37.68 → **+36.8** |
| **Median regret cp** | +8.5 / +11 / +13 → **+10.8** | +33.5 / +41.5 / +37 → **+37.3** |
| p95 regret cp | +50 / −4 / +43 → +29.7 | +49 / −24 / +32 → +19.0 |
| Max regret cp | +16 / +106 / +62 → +61.3 | +540 / +540 / +522 → **+534** |
| Blunder rate | +0.046 / +0.008 / +0.031 → +0.028 | +0.015 / +0.021 / +0.013 → +0.016 |
| Spearman mean | −0.04 / −0.03 / 0 → −0.023 | −0.14 / −0.07 / −0.09 → **−0.10** |

### `phase0_52` (n=52) — secondary, 11.5% train overlap

| Metric | Production mean | Matched mean |
|---|---:|---:|
| Top-1 agreement % | −1.92 (mixed) | **−7.05** (all −) |
| Top-3 containment % | −10.3 | −6.41 |
| **Mean regret cp** | **+12.8** | **+40.1** |
| **Median regret cp** | +16.5 | +28.3 |
| Max regret cp | +7.33 | **+458** |
| Spearman mean | −0.04 | **−0.17** |

Both suites move the same way: the A0→A1 gap **widens** under matched fusion, and
becomes directionally consistent on metrics that were previously mixed.

---

## 5. Selected-move and top-3 containment changes

Production → matched, `extended` (n=160):

| Run | moves changed | top-3 gained | top-3 lost |
|---|---:|---:|---:|
| A0 seed 0 | 52 | 5 | 13 |
| A0 seed 1 | 50 | 10 | 12 |
| A0 seed 2 | 41 | 10 | 8 |
| **A1 seed 0** | **115** | 10 | **36** |
| **A1 seed 1** | **111** | 10 | **33** |
| **A1 seed 2** | **112** | 5 | **30** |

The refit perturbs A1 **more than twice as much** as A0 (111–115 of 160 moves vs
41–52) — consistent with Stage 1, where A1's coefficient shift was the larger one
(CNN ranking share 31–35% vs A0's 46–55%). On `phase0_52` the pattern repeats
(A0 seed 0: 27 of 52).

---

## 6. Breakdowns (matched fusion)

### White / Black — mean regret cp, `extended`

| Side | A0 s0 | A0 s1 | A0 s2 | A1 s0 | A1 s1 | A1 s2 |
|---|---:|---:|---:|---:|---:|---:|
| White (n=85) | 89.84 | 93.63 | 98.24 | 127.06 | 132.22 | 148.01 |
| Black (n=75) | 155.17 | 169.11 | 172.25 | 202.45 | 191.80 | 196.31 |

A1 is worse than A0 for both colours in every seed. Black remains worse than White
in every cell, as in all previous phases.

### Phase / category — mean regret cp, `extended`

| Category | n | A0 s0 | A0 s1 | A0 s2 | A1 s0 | A1 s1 | A1 s2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| opening | 60 | 115.35 | 114.78 | 140.00 | 133.25 | 125.85 | 128.13 |
| middlegame | 64 | 154.39 | 176.25 | 157.09 | 150.39 | 173.12 | 193.02 |
| **endgame** | 21 | **39.00** | **38.71** | **42.38** | **309.10** | **259.71** | **247.14** |
| tactical | 10 | 133.00 | 135.50 | 135.50 | 56.25 | 56.25 | 56.25 |
| defensive | 5 | 68.67 | 59.50 | 101.50 | 101.50 | 6.00 | 161.50 |

**The A1 matched-fusion regression is concentrated in endgames**: 247–309 cp mean
regret versus A0's 39–42 cp, a ~6× difference, consistent across all three seeds.
Opening and middlegame differences are comparatively small. The tactical and
defensive cells (n=10 and n=5, with identical values across A1 seeds) are too
small to interpret.

### Max-regret outlier

A1's matched max regret is **1218 cp in all three seeds**, traced to a single
position — `XEG07` (endgame, Black to move): the engine plays `h1h2` where
Stockfish plays `h1a1`. A1's other worst cases are also endgames (`XEG05` 1060,
`XEG15` 1058). A0's matched worst cases are middlegames at 671–678 cp.

### Mate statuses

Unchanged in character. `missed_forced_mate` and `engine_move_allows_forced_mate`
stay in the 2–4 band across all twelve cells — inside the ±1 seed noise A0
established. **No mate-status conclusion is drawn.**

---

## 7. Interpretation

**Descriptive, paired evidence** (what was measured):

- Under matched fusion the A0→A1 mean-regret gap is **+36.8 cp** on `extended` and
  **+40.1 cp** on `phase0_52`, versus **+16.6** and **+12.8** under production
  fusion. The gap is larger in **all six** paired comparisons.
- Metrics that were mixed under production fusion become consistent under matched
  fusion: top-1 (−10.0 pp, all three seeds negative) and top-3 (−14.0 pp, all
  three negative).
- The matched Ridge **lowers** A0's mean regret (−11.6 cp, all three seeds) and
  **raises** A1's (+8.5 cp, all three seeds).
- A1's degradation under matched fusion is concentrated in **endgames**.

**What this supports** (stated as inference, not proof):

The pre-registered rule says: if the A1 regret increase persists under matched
fusion, the production Ridge mismatch does not explain the A1 regression. It
persists and enlarges, so **the A1 engine-level result is not an artifact of
fusion mismatch.** The A1 report's stated confound is resolved, and A1's
conclusion stands as written.

A coherent reading of A0 improving while A1 degrades: Stage 1 showed the honest
out-of-sample refit down-weights the CNN in both arms, and **more so for A1**
(31–35% of ranking spread vs A0's 46–55%). If A1's CNN is a *worse ranker* despite
its better label fit, then giving it less weight should help — yet A1 got worse.
That points toward A1's degradation being carried by the CNN's *predictions*
rather than by the fusion weighting. **This is a hypothesis consistent with the
data, not a demonstrated mechanism**; no mechanism was isolated, and three seeds
cannot establish causality.

**What must not be concluded:** that matched fusion is "better" or "worse" in
general. It helps A0 and hurts A1; it is a different fitting methodology
(out-of-sample vs partly in-sample), not a quality upgrade.

---

## 8. Limitations

1. **Three seeds per cell, six cells.** No significance testing; all figures are
   descriptive ranges and paired directions.
2. **Matched Ridges were fitted on the 1,933-position test split**, which holds
   only 80 Black-to-move and 35 mate records — thin exactly where A1 differs. The
   endgame concentration of A1's regression sits in a 21-position category.
3. **The refit consumed the only clean holdout**; Stage 1's inner-split R²
   (0.348–0.416) is a sanity check, not independent validation.
4. **`OP04`** (1 of 52 `phase0_52` positions) is in the Ridge fitting set.
   `extended`, the primary suite, has **zero** overlap.
5. **Matched fusion changes features *and* target**, so this isolates "arm-matched
   fusion vs production fusion", not output scale alone.
6. **The objective mismatch is unfixed** for both fusions: the Ridge is fitted to
   predict absolute centipawns across positions but used to rank moves within one.
7. **The intercept is discarded by `engine.py`** (defect C3). Matched Ridges carry
   intercepts of 37.6–60.0 that the engine throws away, so the model evaluated is
   not the model fitted. This applies equally to production.
8. **Mean, median and tail metrics disagree** for A1 under matched fusion (median
   +29.7 worse, blunder rate −0.038 better). Any single-metric summary would
   mislead.
9. **No mechanism was isolated** for why A1 degrades. The reading in §7 is a
   hypothesis.
10. **Mate statuses remain underpowered** (7–8 mate positions of 160).

---

## 9. Is the A1 result still interpretable?

**Yes, and it is now better supported than when it was written.**

A1's report recorded a consistent engine-level regression across three paired
seeds and explicitly flagged the Ridge mismatch as an unresolved confound that
"this design cannot separate from a genuine label effect". Stage 2 separates it.
The regression is not only present under matched fusion — it is **larger and more
consistent**.

A1's recorded conclusions require **no revision**. One sentence can now be
strengthened: where A1 said the fusion confound "cannot be separated", it has been
separated, and it does not account for the result. A1's own artifacts and report
were not modified.

---

## 10. Should A2 proceed?

**Yes — A2 can now proceed, under the production Ridge, as originally planned.**

Reasoning:

- The question that justified pausing is answered. Fusion mismatch does not
  explain A1's engine-level result, so A2 does not inherit an unresolved confound
  of that kind.
- **A2 should keep the production Ridge**, matching A0 and A1. Stage 2 shows the
  fusion layer is not neutral — it helps A0 and hurts A1 — so switching arms to
  matched fusion mid-programme would introduce a new, larger comparability problem
  than it solves. A fixed production Ridge keeps all arms mutually comparable.
- The A1R results should be **cited in A2's report** as the evidence that the
  fixed-Ridge protocol is defensible, rather than merely assumed.

**Two things worth carrying into A2:**

1. **A2 changes perspective normalisation**, which will shift the CNN's target
   semantics again. If A2 also shows an engine-level regression, the same question
   will arise — but A1R now provides the template and the prior, so it can be
   answered with one Stage-1-style coefficient check rather than a full rerun.
2. **The endgame concentration** (A1 matched: 247–309 cp vs A0's 39–42) is the most
   striking unexplained finding here. It may be worth a targeted look independent
   of A2, since endgames are also where the ply-20 training distribution is
   weakest.

A separate standing observation, unchanged from Stage 1: **the production Ridge
over-weights the CNN** because it was fitted partly in-sample. That is a property
of the shipped engine and is independent of C6.

---

## Reproducing

```bash
python -m training.refit_ridge                    # Stage 1 (~6 min)
python -m training.refit_ridge_stage2             # Stage 2 (~40 min)
python -m training.refit_ridge_stage2_analysis    # tables + JSON
```

Artifacts (all gitignored) under `training/experiments/A1R/`:
`stage1_results.json`, `stage2_manifest.json`, `stage2_analysis.json`,
`matched_ridges/*.pkl`, and `{A0,A1}_seed_{0,1,2}/{extended,phase0_52}.{json,md}`.
