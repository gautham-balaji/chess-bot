# C8b — Dataset Scaling (`dataset_v2_k6`)

## 1. Status

Complete. 3 seeds, both suites, frozen production Ridge. **C8a remains the
control and is unchanged.** No production file changed. **C9 has NOT been run.**

---

## Result up front

> **SATURATION.** C8b is approximately equivalent to C8a at engine level.
>
> On `extended`, C8a → C8b is **within the A0 noise band and directionally
> inconsistent on 8 of 10 metrics**; the two that clear the band (top-3 +1.25,
> Spearman −0.01) are inconsistent across seeds and cancel. On `phase0_52`,
> **all 10 metrics are within the band and inconsistent**.
>
> C8b does **retain** C8a's gain over A2 — A2 → C8b still clears the bar on
> 6 of 10 `extended` metrics — but adds nothing on top of it.
>
> Model level tells a different story: C8b is *slightly but consistently* better
> than C8a on all three held-out sets (MAE −4.7 to −9.7 cp, Pearson +0.002 to
> +0.021). **The model metrics improve while the engine metrics do not.**

44% more training data bought a measurably better regressor and no better
chess. The dataset-expansion branch has reached diminishing returns.

---

## 2. Experiment question

Does increasing the C8a training dataset further continue to improve
model-level and engine-level performance? C8a is the control; only the training
dataset changes.

---

## 3. Fixed controls

| | value | how it is held |
|---|---|---|
| architecture | A2 Sequential, `(8,8,12)`, **2,360,129 params** | `train_v2` imports `train.build_model`; a guard aborts on any other count |
| representation | `planes12` | imported `training.representation` |
| labels | `corrected_mate_white_perspective` | `dataset.apply_label_policy`, re-derived and asserted equal to the stored labels |
| loss / optimiser | Huber, Adam 1e-3, batch 64 | `train.HP` (the same object, not a copy) |
| callbacks | ReduceLROnPlateau (0.5, p5), EarlyStopping (p10, restore best) | `train.make_callbacks()` |
| epoch cap | 100 | `train.HP` |
| seeds | 0, 1, 2 | — |
| fusion | **frozen production Ridge — not refitted** | C9 is the later matched-Ridge experiment |
| evaluator | `evaluation/evaluate.py`, unmodified | direct path (12-plane arm, no shim) |
| Stockfish | 17.1, depth 8, Threads=1, Hash=16MB, Clear Hash per position | — |
| suites | `extended` (160), `phase0_52` (52) | unmodified |

`training/train.py` was not modified. C8b is an additional arm in
`training/train_v2.py`; the arm spec differs from C8a's in exactly one key,
`dataset_prefix` (asserted by test).

---

## 4. Dataset construction

`dataset_v2_k6` is built by the same `training/build_dataset_v2.py` pipeline,
parameterised by extraction policy. The pipeline order is unchanged:

```
games.csv rows
  -> group by game CONTENT (sha256 of the moves string)   [18,920 units]
  -> seeded split over sorted units                       [seed 42, 20% test]
  -> extract positions from each side INDEPENDENTLY       [k=6, min_ply 16, gap 4]
  -> deduplicate within each side at PLACEMENT level
  -> scrub: drop from TEST any placement present in TRAIN
  -> scrub: drop from BOTH any placement in either evaluation suite
  -> label the survivors with the A2 policy
```

| parameter | C8a (`dataset_v2`) | C8b (`dataset_v2_k6`) |
|---|---|---|
| policy | `evenly_spaced_4_minply16` | `evenly_spaced_6_minply16` |
| min ply | 16 | 16 |
| **max positions per game** | **4** | **6** |
| min gap | 4 | 4 |
| RNG / sampling seed | none | none |
| split seed | 42 | 42 |

### C8a's dataset is provably untouched

Parameterising the builder was a regression risk, so it was verified directly:
rebuilding `dataset_v2` with the parameterised code reproduced the original
unlabelled artifact hashes **`aeda950b3716ab8d…` / `e873267d9dcb97d1…`**
exactly, and `dataset_v2`'s committed manifest still declares
`evenly_spaced_4_minply16`, max 4, 54,812 / 13,712. The default policy is still
C8a's. Both facts are pinned by tests.

---

## 5. Final dataset counts

Every count C7's audit predicted for k=6 was reproduced by the build:

| stage | train | test | total |
|---|---:|---:|---:|
| selected | 79,202 | 19,922 | **99,124** |
| after placement dedup | 78,902 | 19,892 | 98,794 |
| after train/test leakage scrub | 78,902 | 19,802 | 98,704 |
| after evaluation-suite scrub | **78,901** | **19,802** | **98,703** |

Against C8a: **+24,089 train records (+43.9%)** and +6,090 test.

### Composition — C8b is not simply "more of the same"

| | C8a train | C8b train |
|---|---:|---:|
| records | 54,812 | 78,901 |
| distinct games | 14,410 | **14,410 (identical)** |
| White to move | 58.6% | **62.3%** |
| opening | 27.0% | **20.1%** |
| middlegame | 62.9% | **70.2%** |
| endgame | 10.1% | 9.7% |
| **checkmate records** | **4,658 (8.5%)** | **4,658 (5.9%)** |
| mate-typed labels | 5,610 (10.2%) | 5,920 (7.5%) |

Two consequences worth stating explicitly:

1. **The same 14,410 games, sampled more densely.** C8b adds no new games, so
   this is a density experiment, not a coverage experiment.
2. **The checkmate count is identical in absolute terms** (4,658), because the
   policy takes one final ply per game either way. The extra 24,089 records are
   almost entirely non-terminal middlegame positions, which *dilutes* the
   checkmate share from 8.5% to 5.9%. This is a direct, unplanned test of C8a's
   documented over-representation caveat — see §13.

---

## 6. Leakage and split verification

| guarantee | measured |
|---|---|
| split unit | game move-sequence content, **not** the `id` column |
| game list / train / test sha256 | **byte-identical to C8a's** (§ below) |
| train units / test units | 15,136 / 3,784 |
| `train_units ∩ test_units` | **0** |
| train/test placement overlap, pre-scrub | 90 placements, 90 test records removed |
| **final train/test placement overlap** | **0** |
| **final train/test game overlap** | **0** |
| evaluation-suite records removed | 1 train, 0 test |
| **final suite overlap** | **0 train, 0 test** |
| min ply | 16 (both sides) |
| max records per game | 6 (both sides) |
| selected pairs closer than 4 plies | **0** |
| suites modified | **no** — read-only |

```
game list   b90881bf3c6245ba3634800da8879ccf8ecb9a64fb41f5116c5439983dc64e62
train games cb1308d9af1ee45709f272492c0a22a02b8e7026f055d8ff43e1d81f881c19fb
test games  d8632c3ac89be36bced244f4370eec09c3c80b2e6a0fe841f62ca67df97809f3
```

Because the split hashes match C8a's, C8b's train games **are** C8a's train
games. A test asserts `k4_train_games ⊆ k6_train_games`.

---

## 7. Training results

| run | epochs | best ep | best val Huber | test Huber | MAE | RMSE | Pearson | secs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C8a s0 | 18 | 8 | 217.34 | 219.64 | 220.14 | 360.51 | 0.8760 | 874 |
| C8a s1 | 25 | 15 | 221.92 | 221.75 | 222.25 | 362.76 | 0.8742 | 1,094 |
| C8a s2 | 22 | 12 | 217.73 | 220.13 | 220.63 | 364.46 | 0.8732 | 1,128 |
| **C8b s0** | 27 | 17 | 213.93 | 208.50 | 209.00 | 340.92 | 0.8614 | 2,411 |
| **C8b s1** | 25 | 15 | 211.31 | 206.50 | 206.99 | 338.84 | 0.8636 | 13,218 |
| **C8b s2** | 27 | 17 | 210.18 | 204.26 | 204.76 | 334.12 | 0.8677 | 1,792 |

> ⚠️ Each arm's own test metrics are measured on **its own** held-out set
> (C8a: 13,712 records; C8b: 19,802). They are not comparable directly. §10
> resolves this by scoring every model on every set.

C8b model weight hashes:

```
C8b s0  582ed0190ffa6986b5b522e5080cd7566249b614bbc5e49e1382508e5abd3bed
C8b s1  667930d4ed31ac338fcd54fd63e754def5ab2907d7945f015f9745b650b26a35
C8b s2  f500cf1859acbebb2ebe1aa8dde2351727a3fdb73676b7af97be6742fdb797bd
```

History hashes: `a29123bb…`, `0185520b…`, `d19c4617…`.

**Runtime note:** seed 1's 13,218s is an artifact of external machine load
(free RAM fell to ~1.8 GB of 16 GB, with unrelated Docker and Chrome processes
consuming more CPU than the training run). Epoch times ranged 58s–1,146s across
the run. This affects wall clock only — the recipe, seeds and determinism setup
are unchanged, and seed 1's results sit between seeds 0 and 2 on every metric.

---

## 8. Engine results

Three-seed means, production fusion, identical suites and evaluator.

### `extended` (n=160)

| arm | legality | top-1 % | top-3 % | mean regret | median | p95 | max | blunder | coverage | Spearman mean / median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 100 | 18.96 | 33.54 | 139.29 | 31.33 | 565.7 | 684.0 | 0.230 | 95.21 | 0.213 / 0.263 |
| A2 | 100 | 17.71 | 32.29 | 142.53 | 37.67 | 533.3 | 700.7 | 0.229 | 95.62 | 0.193 / 0.237 |
| **C8a** | 100 | 21.88 | 39.17 | **107.03** | **17.00** | **502.3** | 680.0 | **0.160** | 96.46 | **0.297** / 0.347 |
| **C8b** | 100 | **22.71** | **40.41** | 111.32 | 19.50 | 507.7 | 689.7 | 0.168 | **96.67** | 0.287 / 0.320 |

### `phase0_52` (n=52)

| arm | legality | top-1 % | top-3 % | mean regret | median | p95 | max | blunder | coverage | Spearman mean / median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 100 | 21.79 | 40.38 | 104.31 | 20.83 | 506.3 | 614.0 | 0.154 | 87.18 | 0.233 / 0.263 |
| A2 | 100 | 19.23 | 28.85 | 113.81 | 30.50 | 503.7 | 618.0 | 0.167 | 88.46 | 0.267 / 0.360 |
| **C8a** | 100 | **27.57** | **46.16** | **95.38** | **15.17** | 459.3 | 614.0 | **0.148** | 91.02 | 0.293 / 0.323 |
| **C8b** | 100 | 23.08 | 45.51 | 100.05 | 17.00 | **459.0** | 614.0 | 0.154 | **91.67** | **0.337** / **0.390** |

Legality 100% in all six runs. C8a and C8b trade wins metric by metric with no
consistent direction — which is what §9 quantifies.

---

## 9. A0 / A2 / C8a / C8b comparison

Pre-registered rule, unchanged: a change counts only if it **exceeds the A0
band** *and* is **consistent across all three paired seeds**.

### C8a → C8b, `extended` (the primary question)

| metric | per-seed Δ | mean Δ | A0 range | verdict |
|---|---|---:|---:|---|
| mean regret cp | −1.09 / +16.12 / −2.16 | +4.29 | 19.70 | within band, inconsistent |
| median regret cp | +5 / +2.5 / 0 | +2.50 | 10.0 | within band, inconsistent |
| p95 regret cp | −35 / +77 / −26 | +5.33 | 64.0 | within band, inconsistent |
| max regret cp | +29 / 0 / 0 | +9.67 | 18.0 | within band, inconsistent |
| top-1 % | −2.50 / +1.25 / +3.75 | +0.83 | 3.74 | within band, inconsistent |
| top-3 % | 0 / 0 / +3.74 | +1.25 | 0.63 | exceeds band, **inconsistent** |
| blunder rate | +0.013 / +0.019 / −0.008 | +0.008 | 0.0395 | within band, inconsistent |
| legality % | 0 / 0 / 0 | 0 | 0 | unchanged |
| Spearman mean | −0.02 / −0.01 / 0 | −0.01 | 0.01 | exceeds band, **inconsistent** |
| regret coverage % | 0 / 0 / +0.63 | +0.21 | 0.62 | within band, inconsistent |

**8 of 10 within the band; 0 of 10 pass both tests.**

### C8a → C8b, `phase0_52`

| metric | per-seed Δ | mean Δ | A0 range | verdict |
|---|---|---:|---:|---|
| mean regret cp | +18.52 / −30.08 / +25.56 | +4.67 | 31.65 | within band, inconsistent |
| median regret cp | +12.5 / −12 / +5 | +1.83 | 7.50 | within band, inconsistent |
| p95 regret cp | +52 / −89 / +36 | −0.33 | 53.0 | within band, inconsistent |
| top-1 % | −11.54 / +1.92 / −3.85 | −4.49 | 5.77 | within band, inconsistent |
| top-3 % | −11.54 / +7.69 / +1.92 | −0.64 | 3.85 | within band, inconsistent |
| blunder rate | +0.021 / −0.043 / +0.039 | +0.006 | 0.0624 | within band, inconsistent |
| Spearman mean | +0.09 / +0.04 / 0 | +0.043 | 0.06 | within band, inconsistent |
| regret coverage % | 0 / 0 / +1.93 | +0.64 | 1.92 | within band, inconsistent |
| max regret / legality | 0 / 0 / 0 | 0 | 0 | unchanged |

**All 10 within the band and inconsistent.**

### A2 → C8b — the C8a gain is retained

| metric (`extended`) | mean Δ | A0 range | verdict |
|---|---:|---:|---|
| mean regret cp | −31.21 | 19.70 | **exceeds, better** |
| median regret cp | −18.17 | 10.0 | **exceeds, better** |
| top-1 % | +5.00 | 3.74 | **exceeds, better** |
| top-3 % | +8.12 | 0.63 | **exceeds, better** |
| blunder rate | −0.061 | 0.0395 | **exceeds, better** |
| Spearman mean | +0.093 | 0.01 | **exceeds, better** |
| regret coverage % | +1.05 | 0.62 | **exceeds, better** |

7 of 10 clear the bar, essentially reproducing A2 → C8a (which was also 7/10).
C8b sits where C8a sits relative to A2, not beyond it.

### Seed bands (`extended`)

| metric | A2 band | C8a band | C8b band | A0 band |
|---|---|---|---|---|
| mean regret | [133.7 .. 152.3] | [93.43 .. 120.0] | [105.5 .. 118.9] | [128.7 .. 148.4] |
| top-1 % | [14.38 .. 19.38] | [20.00 .. 24.38] | [21.25 .. 25.00] | [16.88 .. 20.62] |
| top-3 % | [29.38 .. 34.38] | [38.12 .. 40.00] | [38.12 .. 43.12] | [33.12 .. 33.75] |
| blunder rate | [0.2026 .. 0.2484] | [0.1429 .. 0.1742] | [0.1548 .. 0.1871] | [0.2105 .. 0.2500] |

C8a's and C8b's bands overlap substantially on every metric. No seed is ranked
or selected; all comparisons are paired three-seed.

---

## 10. Model level vs engine level — they disagree

Every model scored on every held-out set (`training/c8a_cross_eval.py`):

| held-out set | arm | Huber | MAE | RMSE | Pearson |
|---|---|---:|---:|---:|---:|
| dataset_v1_test (1,933) | A2 | 162.38 | 162.88 | 261.61 | 0.7381 |
| | C8a | 138.46 | 138.96 | 230.45 | 0.8044 |
| | **C8b** | **128.72** | **129.22** | **222.12** | **0.8255** |
| dataset_v2_test (13,712) | A2 | 389.31 | 389.81 | 626.89 | 0.5770 |
| | C8a | 220.51 | 221.01 | 362.58 | 0.8745 |
| | **C8b** | **215.85** | **216.35** | **360.40** | **0.8761** |
| dataset_v2_k6_test (19,802) | A2 | 354.08 | 354.58 | 561.85 | 0.5739 |
| | C8a | 214.43 | 214.93 | 344.25 | 0.8599 |
| | **C8b** | **206.42** | **206.92** | **337.96** | **0.8642** |

**C8b beats C8a on all three sets, including C8a's own** (MAE −4.66,
Pearson +0.0016) and A2's (MAE −9.74, Pearson +0.0211). The direction is
consistent; the magnitude is small.

**This is an explicit model-improves / engine-does-not case.** The extra 44% of
data made the CNN a measurably better centipawn regressor and did not make the
engine pick better moves. Two non-exclusive readings, neither demonstrated here:

- the remaining engine error is dominated by the **fusion**, not the CNN — the
  frozen Ridge and its `tanh(cnn/200)` squash discard much of the improvement
  (§13 shows C8a/C8b losing 7–8% of candidates to saturation against A2's 3.7%);
- **regression accuracy and move ranking diverge** — a lower mean absolute error
  over a position distribution does not imply better *within-position* ordering,
  which is what top-1 and regret measure.

---

## 11. White / Black analysis

**Engine mean regret, `extended`, C8a → C8b paired:** black **−8.25**,
white **+15.52** — opposite directions, netting to nothing. On `phase0_52`:
black +9.78, white +0.56. No consistent side effect.

**Model MAE by side to move:**

| held-out set | arm | White | Black | gap |
|---|---|---:|---:|---:|
| dataset_v2_k6_test | A2 | 327.7 | 398.8 | +71.1 |
| | C8a | 197.1 | 244.2 | **+47.1** |
| | C8b | **187.3** | **239.2** | +51.9 |
| dataset_v1_test | A2 | 157.5 | 287.0 | +129.5 |
| | C8a | 134.6 | 239.9 | +105.3 |
| | C8b | **125.2** | **222.4** | **+97.2** |

C8b improves both sides in absolute terms on both sets. The *gap* narrows
further on dataset_v1_test (105.3 → 97.2) but widens slightly on
dataset_v2_k6_test (47.1 → 51.9) — consistent with C8b's training mix being
more White-skewed (62.3% vs 58.6%). The effect is small and not consistent
across sets.

---

## 12. Phase analysis

**Model MAE by phase, `dataset_v2_k6_test`:**

| phase | A2 | C8a | C8b |
|---|---:|---:|---:|
| opening | 122.2 | 114.5 | **108.2** |
| middlegame | 371.4 | 233.4 | **225.9** |
| endgame | 692.7 | 283.3 | **268.5** |

C8b improves every phase slightly over C8a. The large gains remain the A2 → C8a
ones.

**Engine mean regret by category, `extended`, C8a → C8b mean delta:**
defensive −3.08, endgame −7.67, middlegame +5.18, opening +1.33,
tactical **+81.81**. The tactical figure rests on n=10 with extreme per-seed
spread (C8a seeds scored 105 / 1 / 0; C8b 116 / 101 / 135) and should not be
read as a finding. On `phase0_52`: defensive −9.67, endgame +0.25,
middlegame +15.86, opening +0.43, tactical +22.28.

No phase shows a consistent, band-clearing move in either direction.

---

## 13. Mate analysis, saturation and candidate separation

### Mate statuses, summed across three seeds

| suite | arm | none | missed_forced_mate | allows_forced_mate | both_forced_mate |
|---|---|---:|---:|---:|---:|
| extended | A2 | 459 | 12 | 9 | 0 |
| extended | C8a | 463 | 11 | 5 | 1 |
| extended | **C8b** | **464** | 11 | **4** | 1 |
| phase0_52 | A2 | 138 | 12 | 6 | 0 |
| phase0_52 | C8a | 142 | 11 | 2 | 1 |
| phase0_52 | **C8b** | **143** | 11 | **1** | 1 |

`engine_move_allows_forced_mate` continues to fall (9 → 5 → 4 and 6 → 2 → 1),
but the movement from C8a to C8b is one position per suite — inside noise.
`missed_forced_mate` is unmoved at 11 for both arms.

**Model MAE on checkmate positions** (dataset_v2_k6_test): A2 1606.0,
C8a **476.9**, C8b **547.0**. C8b is *worse* on checkmates than C8a — consistent
with its checkmate share falling from 8.5% to 5.9% while the absolute count
stayed at 4,658.

### Saturation and candidate separation (5,639 candidate positions)

| arm | raw \|cnn\| | saturated (\|tanh\| > 0.95) | within-position tanh spread (White / Black) |
|---|---:|---:|---:|
| A0 | 98.8 | 6.2% | 0.6458 / 0.6353 |
| A2 | 110.0 | **3.7%** | **0.9547** / 0.8126 |
| C8a | 155.5 | 8.1% | 0.5819 / 0.6434 |
| **C8b** | **124.7** | **6.9%** | **0.6604** / 0.6766 |

**C8a's documented over-representation caveat is confirmed mechanically and
refuted as an explanation.** Diluting checkmates from 8.5% to 5.9% did exactly
what C8a predicted it would: output magnitude fell 155.5 → 124.7, saturation
fell 8.1% → 6.9%, and within-position candidate separation rose 0.582 → 0.660.

**And engine performance did not improve.** The partial relief of output-scale
inflation was real and measurable, and it bought nothing. That is evidence
against mate-driven saturation being the binding constraint on these arms — and
a reason *not* to pursue a checkmate-excluded dataset on saturation grounds.

Both C8a and C8b remain well above A2 on saturation (6.9–8.1% vs 3.7%) and well
below it on White-to-move separation (0.58–0.66 vs 0.95), while beating it
decisively on the engine metrics. The fusion still looks like the place where
signal is lost — which is C9's question, not C8b's.

---

## 14. Decision

**(B) SATURATION.**

- **Does C8b beat C8a?** No. 0 of 10 metrics on `extended` and 0 of 10 on
  `phase0_52` pass both the band and consistency tests. Directions are mixed.
- **Does C8b still clear the A0 band vs A2?** Yes — 7 of 10 on `extended`,
  matching C8a. The C8a gain is retained, not extended.
- **Consistent across all three seeds?** No. Every C8a → C8b metric is
  directionally inconsistent across seeds.
- **Improve, plateau or regress?** **Plateau** at engine level; **small,
  consistent improvement** at model level.
- **By side?** No consistent effect (black −8.25, white +15.52 on `extended`).
- **By phase?** No consistent effect; model-level MAE improves slightly in all
  three phases.
- **Mate metrics?** `allows_forced_mate` 5 → 4 and 2 → 1 (noise-level);
  `missed_forced_mate` unmoved; checkmate model MAE *worsens* 476.9 → 547.0.
- **Saturation / separation?** Both improve (8.1% → 6.9%, 0.582 → 0.660)
  without an engine gain.

**Recommendation: stop scaling this dataset.** Going 54.8k → 78.9k on the same
14,410 games produced no engine-level improvement. Further density increases on
the same corpus have no measured basis. C8a remains the reference arm; C8b is
not an upgrade over it and should not replace it.

---

## 15. Limitations

1. **Size and composition are confounded, and this is not hidden.** C8b changes
   record count (+43.9%) *and* mix (opening 27.0% → 20.1%, middlegame
   62.9% → 70.2%, White 58.6% → 62.3%, checkmate share 8.5% → 5.9%). **C8b does
   not isolate "more records" as a causal variable.** A null result is therefore
   a null for *this* denser sampling, not for data volume in general.
2. **Same games, not new games.** C8b adds zero new games. It tests sampling
   density on a fixed 14,410-game corpus, not corpus size.
3. **Different held-out sets.** Each arm's own test metrics are on its own
   split; only §10's cross-evaluation is comparable.
4. **Frozen Ridge caps every arm.** Not refitted, by design. §13 shows both C8
   arms are penalised more by it than A2 is.
5. **Three seeds, descriptive only.** No significance testing; the A0 band is
   the instrument. With bands this wide relative to the C8a → C8b deltas, a real
   effect smaller than the band would be invisible here.
6. **Small suites.** n=160 and n=52; category cells as small as n=4–10. The
   tactical deltas (+81.81, +22.28) are noise.
7. **Seed 1's wall clock is not comparable** (external machine load, §7). This
   affects runtime reporting only, not results.
8. **One dataset population.** 20,058 Lichess games throughout.

---

## 16. Production safety

**PASS.** `git status` over `engine.py`, `app.py`, `config.py`, `evaluation/`,
`models/`, `baseline/`, `regression/`, `training/labels.py`,
`training/dataset.py`, `training/train.py`, `dataset_v1.jsonl`,
`dataset_v1.manifest.json` and `dataset_v2.manifest.json` is **empty — all
unchanged**.

```
models/cnn_model.keras              972d81199a1355667fca554b06dd05b35fdec3c4dc789968ac4a1b0599d8dec3
models/weight_model.pkl             59a731127cc9b440c866c7a5d27ce39d905c9c5dadd8116d7f97a8542f18bcd0
evaluation/positions/extended.json  9d5419cceec43bac98f9cdcf8109a4242e93791fa267f1df54fc1dddc41f929a
evaluation/positions/phase0_52.json 6078d0e8db9e4124b984bbf3b5ad018de6b9c84b49361b01058ff58d875c0208
training/artifacts/dataset_v1.jsonl 5a689e3f37156a0540598cdebd0c7f42cdbb09976b5c9af61c013ebdcf3a430d
dataset_v2.manifest.json            5aae0a86a7d51a5531d9ebced92b2a8b09ce579ccb85cb9dacc988348ae91841
```

`dataset_v2`'s and `dataset_v2_k6`'s JSONL files both still match their
manifests. C8b models, predictions and evaluation output live under
`training/experiments/C8b/`, which is gitignored, as are the `.jsonl` data files
under `training/artifacts/`.

Full suite: **827 passed, 10 xfailed** (the pre-existing Phase-4 deferrals). No
test was weakened or deleted; one roster assertion in
`test_training_train_v2.py` was extended from `["C8a"]` to `["C8a", "C8b"]`,
keeping its actual guarantee (no leak into `train.ARMS`) intact.

### C8b dataset hashes

```
source games.csv  e7aadff104a610afb5403caf81c1461babecb0dc87760ff5eea7dc0e7a4a8129
train jsonl       86bc32e1721675cb6f95605b845ee841b611dd6d46cc6d749895be3bffc4e9be
test jsonl        811e2b63a1dc77e45f7640d475ae599615229aab5b05e37d32d46f9d43fb67bd
train labels      ece1145b13354727bd66d571136f9209b77172bd89c5d4ce93e1db8e6c689b68
test labels       bd5fc8d92b1e108f80cc43e7a544744529222d84399cac492d04c3737e3216c1
```

---

## 17. Reproduction

```bash
# build the k=6 dataset (~13 min, needs Stockfish)
python -m training.build_dataset_v2 --policy evenly_spaced_6_minply16

# C8a's dataset is still the default and is unaffected
python -m training.build_dataset_v2            # rebuilds dataset_v2 identically

# train (3 seeds)
python -m training.train_v2 --arm C8b --seed 0     # and 1, 2

# engine evaluation, unmodified evaluator
python -m training.evaluate_arm --arm C8b --seed 0 # and 1, 2

# analysis
python -m training.a2_analysis --arms A0 A2 C8a C8b --baseline A0 \
    --contrasts C8a:C8b A2:C8b --out training/experiments/C8b/c8b_analysis.json
python -m training.c8a_cross_eval --arms A2 C8a C8b \
    --out training/experiments/C8b/cross_eval.json
python -m training.a2_saturation_probe --arms A0 A2 C8a C8b --contrast C8a:C8b \
    --out training/experiments/C8b/saturation_probe.json

# verification
python -m pytest tests/unit/test_training_dataset_v2_k6.py -q
```

Artifacts: `training/experiments/C8b/` (gitignored) — per-seed `metadata.json`,
`test_predictions.json`, `models/cnn_model.keras`, `evaluation/{extended,
phase0_52}.{json,md}`, `summary.json`, plus `c8b_analysis.json`,
`cross_eval.json`, `saturation_probe.json`.

**C9 has not been run. The Ridge remains frozen. Production code is unchanged.**
