# C9 Stage 2 — Matched-Fusion Engine Experiment (results)

## 1. Status

Complete. Stage 2a (selection) and Stage 2b (12 suite runs) both executed
exactly as [C9_STAGE2_DESIGN.md](C9_STAGE2_DESIGN.md) specifies.

- **No production file was modified.** `engine.py`, `app.py`, `config.py`,
  `evaluation/`, `models/` (CNN *and* Ridge), `baseline/`, `regression/`,
  `training/labels.py`, `training/dataset.py`, `training/train.py` and all
  datasets are byte-identical.
- **The production Ridge was never replaced** — only read, and staged copies used.
- **C8a was not retrained.** The three CNNs were hash-verified and loaded read-only.
- **The C8a control was not re-run**; its existing artifacts are byte-identical.
- **The evaluator was not altered.** Defect C3 was not fixed.
- Nothing committed.

Prior: [C9 audit](C9_RIDGE_AUDIT.md) · [C9 Stage 1](C9_STAGE1_REPORT.md) ·
[design](C9_STAGE2_DESIGN.md) · [A1R S2](C6_A1R_STAGE2_REPORT.md) ·
[C8a](C8A_REPORT.md) · [C8b](C8B_REPORT.md)

---

## Verdict up front

> **Neither variant produced an engine-level benefit. Both regress.**
>
> | variant | `extended` | `phase0_52` | pre-registered overall |
> |---|---|---|---|
> | **C9b** matched Ridge, d=200 | **MIXED** (1 better, 6 worse) | **REGRESSION** (0 better, 3 worse) | **no engine-level benefit demonstrated** |
> | **C9c** jointly matched, d=1000 | **REGRESSION** (0 better, 6 worse) | **REGRESSION** (0 better, 3 worse) | **no engine-level benefit demonstrated** |
>
> Stage 2a's selection gain (**+0.027 Spearman** on a held-out ranking benchmark)
> **did not transfer** to the engine. This is exactly the risk the design
> pre-registered in §8 and §15.4.
>
> **The C9 fusion branch closes.** The production Ridge and the `/200` divisor
> survive a fair, leakage-free challenge.

One mechanism finding is worth keeping: **C9c cut `missed_forced_mate` from 11 to
4** on both suites and raised `both_forced_mate_for_mover` from 1 to 8 —
unsaturating the CNN genuinely improves mate detection — while simultaneously
degrading overall move quality and tripling worst-case regret (§22).

---

## 2. Objective

> Does changing the fusion of the C8a CNN into the final move-ranking score
> improve actual engine behaviour against the frozen Stockfish evaluator?

---

## 3. Preflight verification

All six checks passed before any Stockfish call. Any mismatch would have aborted.

| # | check | result |
|---|---|---|
| 1 | C8a CNN hashes vs the C9 audit | `6bd57636…`, `144594ff…`, `41ee1626…` — **all match**, 2,360,129 params each |
| 2 | `dataset_v2.test.jsonl` vs manifest | `70ca70f1…` — **match**, 13,712 records |
| 3 | production Ridge | `59a73112…` — **match**, `coef_[0]=330.900549` |
| 4 | `extended.json` | `9d5419cc…` — **match** |
| 4 | `phase0_52.json` | `6078d0e8…` — **match** |
| 5 | production CNN | `972d8119…` — **match** |
| 6 | Variant A control artifacts | all 6 present; Stockfish 17.1, depth 8, Threads=1, Hash=16MB, Clear Hash true, MultiPV 3/8 |

**Control reuse justified:** `evaluate_arm.stage_models_dir` copies
`ridge_model or models/weight_model.pkl`, and the C8a runs passed no
`ridge_model`, so they are production-Ridge by construction; divisor 200 is the
`engine.py` literal. The control was therefore **not re-run**.

---

## 4. Stage 2a FIT/TUNE methodology

Game-level 80/20 split of C8a's held-out set, seed **9** (distinct from the
dataset seed 42 and the inner-split seed 1234), sorted-then-permuted so it
depends only on the *set* of game keys.

| | value |
|---|---:|
| FIT | 10,969 positions / 2,891 games |
| TUNE | 2,743 positions / 723 games |
| FIT ∩ TUNE games | **0** |
| TUNE suite overlap | **0** (guard raises on any) |
| TUNE benchmark | **200 positions, 6,026 candidate moves** |
| Stockfish for TUNE | 17.1, depth 8, Threads=1, Hash=16MB, clear hash per position |
| TUNE labels sha256 | `91bbce9164c03914f7582f17…` |

The Ridge was fitted on **FIT only** and never saw TUNE. No suite was read
during selection.

---

## 5. Complete selection grid

Mean within-position Spearman on TUNE, averaged over the three paired seeds.

| d | 200 | 400 | 600 | 800 | 1000 | 1500 | 2000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A** include checkmate | 0.1552 | 0.1658 | 0.1739 | 0.1795 | **0.1826** | 0.1857 | **0.1859** |
| **B** exclude checkmate | 0.1656 | 0.1778 | 0.1779 | 0.1801 | 0.1761 | 0.1752 | 0.1726 |

Secondary diagnostic (top-1 agreement, variant A): 21.50% at d=200 →
22.67 / 22.00 / 21.33 / **20.83** / 20.83 / 21.00% — it *falls* as `d` rises,
pointing the opposite way to Spearman. Per the pre-registration it did **not**
break ties.

Per-seed Spearman at the selected cell: s0 0.1517, s1 0.2078, s2 0.1882.

## 6. Selected divisor

**d\* = 1000.** Raw best was d=2000 (0.1859), but d=1000, d=1500 and d=2000 all
fall inside the pre-registered 0.005 tie window, so the tie rule (**smaller d**,
then variant A) selected **1000**. `d* ≠ 200`, so C did not collapse onto B.

## 7. Selected fit variant

**A — include checkmate.** At d ≥ 800 variant A outscores B; the preference
*flips* below that (at d=200, B 0.1656 > A 0.1552).

## 8. Selection margin

| | value |
|---|---:|
| selected (A, d=1000) | 0.1826 |
| best non-selected (A, d=2000) | 0.1859 |
| margin over it | **−0.0034** (inside the tie window) |
| contenders within the window | 3 |
| vs production row (A, d=200) | **+0.0273** |

**The selection is a tie-break, not a clear win**, and absolute Spearman is low
(~0.18) — the fused proxy correlates only weakly with Stockfish's ordering.

---

## 9. Variant B configuration

C8a CNN **unchanged** + matched Ridge from the `A_include_checkmate|d=200` grid
row, divisor 200.

| seed | `w0` | `w1` mat | `w2` space | `w3` center | `w4` mob |
|---|---:|---:|---:|---:|---:|
| production | 330.9005 | 32.3899 | 0.7919 | 5.1654 | 0.0188 |
| 0 | 378.9157 | 37.0882 | 19.9156 | −23.3047 | −1.3017 |
| 1 | 381.1282 | 36.8645 | 20.1914 | −20.1395 | −1.5593 |
| 2 | 392.5073 | 35.7463 | 20.0311 | −21.2535 | −1.4497 |

## 10. Variant C configuration

Matched Ridge from `A_include_checkmate|d=1000` **plus** a CNN whose final
`Dense(1)` kernel and bias are scaled by `k = 200/1000 = 0.2`.

| seed | `w0` | `w1` mat | `w2` space | `w3` center | `w4` mob |
|---|---:|---:|---:|---:|---:|
| 0 | 1272.6830 | 6.3911 | 8.7604 | −19.2777 | −0.5480 |
| 1 | 1259.0590 | 8.8748 | 9.1771 | −14.1486 | −1.0320 |
| 2 | 1262.3404 | 8.5454 | 8.8345 | −11.6490 | −0.7164 |

`w0` is ~3.8× production — the compensation Stage 1 predicted a larger divisor
would need.

### ⚠️ Variant C is NOT a single-variable contrast

`engine.py` applies `/200` in **three** places (`:130`, `:164`, `:240`).
Rescaling the CNN changes the effective divisor in all of them. **C9c therefore
changes three things at once:**

1. the Ridge coefficients,
2. the effective CNN divisor in the candidate term,
3. **the effective divisor of the CNN-based 1-ply lookahead** `0.5*tanh(opp/200)`.

Isolating (3) would require an `engine.py` change, which is out of scope.
**C9b is the clean single-variable contrast; C9c is not.**

---

## 11. Staged-artifact verification

Every one of the 6 cells passed all three staging checks (recorded in
`stage2_results.json`):

| check | result |
|---|---|
| staged CNN matches its source | **pass** ×6 |
| staged Ridge matches the Stage 2a coefficients exactly (`atol=0`) | **pass** ×6 |
| staged Ridge is **not** the production vector | **pass** ×6 |
| C9c: earlier CNN layers bit-identical to source | **pass** ×3 |
| C9c: final layer exactly `k ×` source | **pass** ×3 |
| `models/` unmodified after every cell (mtime + size) | **pass** ×6 |

Mechanism verified independently: `max |tanh(scaled/200) − tanh(raw/1000)|` <
1e-5 over held-out positions, and the transformation survives `.keras`
save/load exactly.

---

## 12. Evaluator configuration

Unchanged: `evaluation/evaluate.py`, Stockfish 17.1, depth 8, Threads=1,
Hash=16MB, Clear Hash per position, MultiPV 3 (agreement) / 8 (ranking),
existing regret, mate and legality definitions. Suites `extended` (160) and
`phase0_52` (52).

---

## 13. Per-seed results

`top1 / top3 / mean regret / median regret / blunder`

| run | `extended` | `phase0_52` |
|---|---|---|
| C8a s0 | 24.38 / 38.12 / 119.97 / 19.0 / 0.1742 | 28.85 / 53.85 / 84.15 / 15.5 / 0.1250 |
| C8a s1 | 20.00 / 40.00 / 93.43 / 16.0 / 0.1429 | 25.00 / 34.62 / 121.23 / 25.0 / 0.1915 |
| C8a s2 | 21.25 / 39.38 / 107.68 / 16.0 / 0.1623 | 28.85 / 50.00 / 80.77 / 5.0 / 0.1277 |
| C9b s0 | 18.75 / 34.38 / 159.71 / 46.0 / 0.2500 | 26.92 / 48.08 / 104.36 / 24.0 / 0.1489 |
| C9b s1 | 15.62 / 32.50 / 145.88 / 42.0 / 0.2129 | 21.15 / 30.77 / 128.67 / 46.0 / 0.1957 |
| C9b s2 | 18.75 / 36.25 / 151.35 / 39.0 / 0.2387 | 23.08 / 44.23 / 104.02 / 15.5 / 0.1739 |
| C9c s0 | 26.88 / 36.25 / 149.81 / 26.5 / 0.2372 | 38.46 / 50.00 / 97.83 / 17.0 / 0.1489 |
| C9c s1 | 17.50 / 34.38 / 145.01 / 32.0 / 0.2013 | 28.85 / 34.62 / 145.71 / 38.0 / 0.2222 |
| C9c s2 | 20.00 / 34.38 / 157.46 / 46.0 / 0.2387 | 26.92 / 42.31 / 124.00 / 31.0 / 0.1957 |

Legality **100%** in all 12 runs.

## 14–17. Aggregates, deltas and the A0 band

### `extended` (n=160)

| metric | C8a | C9b | Δ(B) | C9c | Δ(C) | A0 band | B | C |
|---|---:|---:|---:|---:|---:|---:|---|---|
| legality | 100.000 | 100.000 | +0.000 | 100.000 | +0.000 | 0.000 | — | — |
| top-1 % | 21.877 | 17.707 | −4.170 | 21.460 | −0.417 | 3.740 | **WORSE** | within |
| top-3 % | 39.167 | 34.377 | −4.790 | 35.003 | −4.163 | 0.630 | **WORSE** | **WORSE** |
| mean regret | 107.027 | 152.313 | +45.287 | 150.760 | +43.733 | 19.700 | **WORSE** | **WORSE** |
| median regret | 17.000 | 42.333 | +25.333 | 34.833 | +17.833 | 10.000 | **WORSE** | **WORSE** |
| p95 regret | 502.333 | 557.667 | +55.333 | 558.000 | +55.667 | 64.000 | within | within |
| max regret | 680.000 | 704.000 | +24.000 | **1148.000** | **+468.000** | 18.000 | inconsistent | **WORSE** |
| blunder rate | 0.160 | 0.234 | +0.074 | 0.226 | +0.066 | 0.040 | **WORSE** | **WORSE** |
| regret coverage | 96.460 | 97.087 | +0.627 | 96.877 | +0.417 | 0.620 | **BETTER** | within |
| Spearman mean | 0.297 | 0.263 | −0.033 | 0.280 | −0.017 | 0.010 | **WORSE** | **WORSE** |

**C9b: 1 better, 6 worse → MIXED.  C9c: 0 better, 6 worse → REGRESSION.**

### `phase0_52` (n=52)

| metric | C8a | C9b | Δ(B) | C9c | Δ(C) | A0 band | B | C |
|---|---:|---:|---:|---:|---:|---:|---|---|
| legality | 100.000 | 100.000 | +0.000 | 100.000 | +0.000 | 0.000 | — | — |
| top-1 % | 27.567 | 23.717 | −3.850 | 31.410 | +3.843 | 5.770 | within | within |
| top-3 % | 46.157 | 41.027 | −5.130 | 42.310 | −3.847 | 3.850 | **WORSE** | within |
| mean regret | 95.383 | 112.350 | +16.967 | 122.513 | +27.130 | 31.650 | within | within |
| median regret | 15.167 | 28.500 | +13.333 | 28.667 | +13.500 | 7.500 | **WORSE** | **WORSE** |
| p95 regret | 459.333 | 462.000 | +2.667 | 480.333 | +21.000 | 53.000 | within | within |
| max regret | 614.000 | 614.000 | +0.000 | **904.667** | **+290.667** | 0.000 | within | inconsistent |
| blunder rate | 0.148 | 0.173 | +0.025 | 0.189 | +0.041 | 0.062 | within | within |
| regret coverage | 91.023 | 89.100 | −1.923 | 88.460 | −2.563 | 1.920 | **WORSE** | **WORSE** |
| Spearman mean | 0.293 | 0.233 | −0.060 | 0.220 | −0.073 | 0.060 | within | **WORSE** |

**C9b: 0 better, 3 worse → REGRESSION.  C9c: 0 better, 3 worse → REGRESSION.**

## 18–19. Verdicts

| variant | `extended` | `phase0_52` | overall |
|---|---|---|---|
| **C9b** | MIXED | REGRESSION | **no engine-level benefit demonstrated** |
| **C9c** | REGRESSION | REGRESSION | **no engine-level benefit demonstrated** |

The pre-registered bar — IMPROVEMENT on **both** suites — is met by neither.
The rule was applied exactly as written and was not reinterpreted after seeing
the numbers. **No variant is called a production upgrade; no overall ranking of
B against C is formed.**

---

## 20. White / Black analysis

Mean regret, three-seed means:

| suite | arm | black | white |
|---|---|---:|---:|
| extended | C8a | 105.52 | 108.37 |
| extended | C9b | 140.35 | 162.92 |
| extended | C9c | 153.90 | 148.02 |
| phase0_52 | C8a | 74.57 | 110.64 |
| phase0_52 | C9b | 99.93 | 121.00 |
| phase0_52 | C9c | 136.43 | 112.96 |

Both variants degrade both colours on both suites. C9c degrades Black
disproportionately on `phase0_52` (74.57 → 136.43).

## 21. Phase / category analysis

Mean regret by category, `extended`, three-seed means:

| arm | defensive | endgame | middlegame | opening | tactical |
|---|---:|---:|---:|---:|---:|
| C8a | 23.25 | 34.62 | 138.34 | 110.60 | 35.33 |
| C9b | **0.00** | 42.82 | 193.91 | 158.82 | 129.00 |
| C9c | 3.25 | **145.81** | 173.48 | 130.58 | **229.13** |

C9c's endgame regret quadruples (34.62 → 145.81) and tactical regret rises 6.5×
(35.33 → 229.13), both on small cells (n=21 and n=10) — directionally clear,
numerically fragile. Defensive (n=5) improves for both variants.

## 22. Mate analysis

Summed over three seeds:

| suite | arm | none | missed_forced_mate | allows_forced_mate | both_forced_mate |
|---|---|---:|---:|---:|---:|
| extended | C8a | 463 | 11 | 5 | 1 |
| extended | C9b | 466 | 11 | **2** | 1 |
| extended | **C9c** | 465 | **4** | 3 | **8** |
| phase0_52 | C8a | 142 | 11 | 2 | 1 |
| phase0_52 | C9b | 139 | 11 | 5 | 1 |
| phase0_52 | **C9c** | 138 | **4** | 6 | **8** |

**This is the one place a variant clearly beats the control.** C9c reduces
`missed_forced_mate` from **11 → 4** on *both* suites and raises
`both_forced_mate_for_mover` from 1 → 8. Unsaturating the CNN (40.1% → 3.8% of
positions saturated at d=1000) lets the ±2000 mate signal survive the squash, and
the engine finds forced mates it previously missed.

It does not rescue the overall verdict: the same change triples worst-case regret
(680 → 1148 on `extended`) and degrades six headline metrics. Mate statuses are
**not** in the pre-registered metric set used for the verdict, and are reported
here as mechanism, not as a claim.

---

## 23. Risks and confounds

1. **Selection did not transfer.** Stage 2a's +0.027 Spearman gain on TUNE
   produced no engine gain. The design pre-registered this risk (§8); it
   materialised. A held-out ranking benchmark is not the suites.
2. **The tie-break was arbitrary.** Three configurations sat within 0.0033
   Spearman. d=1500 or d=2000 might have behaved differently; they were not run,
   and running them now would be tuning on the suites.
3. **The secondary diagnostic disagreed.** Top-1 on TUNE *fell* as `d` rose,
   and top-1 on the suites is where C9c did best. Spearman may have been the
   wrong primary criterion — but it was pre-registered, and changing it
   post-hoc would invalidate the experiment.
4. **C9c is not a single-variable contrast** (§10) — three things change at once.
5. **Methodological asymmetry (inherited).** Matched Ridges are fitted
   out-of-sample; the production Ridge was fitted partly in-sample on
   unrecoverable data. Some of the regression is attributable to that, not to
   matching. A1R Stage 2 found the same asymmetry helped A0 and hurt A1.
6. **Defect C3 persists.** Ridges are fitted *with* an intercept the runtime
   discards — identical for control and variants, so the comparison is fair.
7. **Small suites, three seeds, descriptive only.** Category cells as small as
   n=5–10; no significance testing.
8. **Wall-clock variance.** C9c seed 1 took 16,053 s against C9c seed 2's 642 s,
   caused by external machine load (free RAM fell to 266 MB). This affects
   runtime only — the evaluator, seeds and configuration were identical.

---

## 24. What Stage 2 establishes

- Refitting the Ridge to C8a's own output distribution **does not improve** the
  engine; at d=200 it degrades it consistently across seeds and suites.
- Jointly matching the Ridge and the divisor **also does not improve** the
  engine, and makes the worst case substantially worse.
- The production fusion (`w = [330.90, 32.39, 0.79, 5.17, 0.019]`, divisor 200)
  **survives a fair, leakage-free challenge** on its own terms.
- Stage 1's coefficient gate tripping (a 33–36 pp shift in CNN ranking share)
  **did not predict** an engine-level gain — a proxy-versus-reality gap now
  measured twice in this programme.
- Reducing CNN saturation **does** improve forced-mate detection (11 → 4 missed).

## 25. What Stage 2 does NOT establish

- **That no fusion change could help.** Only two of a large space were tested,
  one divisor value of seven, and one selection criterion.
- **That the Ridge is optimal.** It is fitted on unrecoverable data with an
  unresolvable in-sample advantage; "not beaten here" ≠ "optimal".
- **That the divisor 200 is optimal.** d=1500 and d=2000 scored higher on TUNE
  and were never run. Running them now would tune on the instrument.
- **That the mate-detection gain is worthless.** It is real and consistent; it
  was simply outweighed. Exploiting it without the regret cost is an open
  question, not a result.
- **Any production recommendation.** Nothing here justifies changing
  `models/weight_model.pkl` or `engine.py`.

---

## 26. Production integrity

**PASS.** `git status` over `engine.py`, `app.py`, `config.py`, `evaluation/`,
`models/`, `baseline/`, `regression/`, `training/labels.py`,
`training/dataset.py`, `training/train.py` and `training/artifacts/` is
**empty**.

```
models/cnn_model.keras              972d81199a1355667fca554b06dd05b35fdec3c4dc789968ac4a1b0599d8dec3
models/weight_model.pkl             59a731127cc9b440c866c7a5d27ce39d905c9c5dadd8116d7f97a8542f18bcd0
evaluation/positions/extended.json  9d5419cceec43bac98f9cdcf8109a4242e93791fa267f1df54fc1dddc41f929a
evaluation/positions/phase0_52.json 6078d0e8db9e4124b984bbf3b5ad018de6b9c84b49361b01058ff58d875c0208
```

C8a control artifacts unchanged (`1cf6368f…`, `11752d8d…`, `3cba840e…`).
`models/` was re-checked after every cell by the runner itself.

## 27. Test results

| | count |
|---|---|
| baseline before Stage 2 | 861 passed, 10 xfailed |
| **after Stage 2** | **903 passed, 10 xfailed** |
| new in `test_training_c9_stage2.py` | **+42** |

No existing test was weakened, relaxed or deleted; no roster assertion changed.

## 28. Reproduction

```bash
# Stage 2a - selection (~10 min, ~6,000 Stockfish calls, no suites)
python -m training.c9_stage2a

# Stage 2b - engine evaluation (12 suite runs)
python -m training.c9_stage2b

# tests
python -m pytest tests/unit/test_training_c9_stage2.py -q
```

Artifacts under `training/experiments/C9/` (gitignored): `stage2_selection.json`,
`tune_candidates.json`, `stage2_results.json`, `matched_ridges/`,
`scaled_cnns/`, `C9b/seed_*/`, `C9c/seed_*/`.

**Final QA has not started. C10 has not started. Nothing was committed.**
