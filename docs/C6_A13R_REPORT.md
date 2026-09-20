# C6-A13R — Variation-Matched Placebo Control (16 planes)

**Status:** complete, 3 seeds, both suites. **The control executed validly** — the
A13P training pathology is gone.
**Scope:** one representation variable relative to A13. No production code changed.
**Fusion:** frozen production Ridge. No matched-Ridge run.
**Statistical standing:** descriptive. Three seeds, no significance testing.

---

## Result up front

> **A meaningless, position-varying broadcast channel reproduces the A13
> regression — and slightly exceeds it.**
>
> On `extended`, A2 → A13R degrades **seven** headline metrics consistently
> across all three paired seeds, and **all seven exceed the A0 noise band**.
> `phase0_52` replicates with five. This is the largest and most coherent
> regression measured anywhere in the C6 programme.
>
> The network allocates near-identical leverage to real castling rights
> (84.1 cp) and to pure hash noise (78.9 cp). But removing castling **hurts**
> (ΔPearson −0.095 to −0.108) while removing the noise **helps** (+0.002 to
> +0.028). Castling carries genuine signal, and the model is *still* worse for
> having been given it as an input plane.

**This supports the broadcast-channel hypothesis (H2).** Under this dataset,
architecture, training recipe and representation-as-input-plane design, adding
spatially-constant position-varying channels is associated with the regression,
independent of whether they carry chess meaning.

---

## 1. Objective

Determine whether A13's regression is caused by the **semantic content** of the
four castling-right channels, or by the **architectural effect** of adding
spatially-constant, position-varying broadcast channels.

A13P attempted this and failed for a reason unrelated to the question. A13R
repairs that specific flaw.

---

## 2. Hypotheses

- **H1 — semantics.** A13 regresses because of what the castling channels *mean*.
  Prediction: a meaningless placebo of the same shape does **not** regress.
- **H2 — architecture.** A13 regresses because of *any* spatially-constant,
  position-varying broadcast channel at this dataset size.
  Prediction: a meaningless placebo regresses **too**.

---

## 3. Exact representation

| | A2 | A13 | A13P | **A13R** |
|---|---|---|---|---|
| channels 0–11 | 12 piece planes | identical | identical | **identical** |
| channels 12–15 | *absent* | castling rights | fixed constant 1.0 | **hash bits of the FEN** |
| spatially constant | — | yes | yes | **yes** |
| **varies across positions** | — | **yes** | **NO** | **YES** |
| shape | (8,8,12) | (8,8,16) | (8,8,16) | **(8,8,16)** |
| parameters | 2,360,129 | 2,362,433 | 2,362,433 | **2,362,433** |

---

## 4. Why A13P was invalid

A13P filled channels 12–15 with a **fixed** constant. Every sample in a batch
then received the same offset in the first convolution, so BatchNormalization's
batch-mean subtraction removed it exactly and the training loss became blind to
those 2,304 weights. They drifted along an unconstrained direction.

Measured consequences (docs/C6_A13P_REPORT.md §11):

| | A2 | A13 | **A13P** | **A13R** |
|---|---|---|---|---|
| val loss max/min | 1.5 / 2.7 / 3.1 | 1.2 / 1.2 / 1.3 | **6.0 / 9.1 / 12.8** | **1.2 / 1.2 / 1.2** |
| epochs with val > 2× running best | 0 / 0 / 0 | 0 / 0 / 0 | **5 / 8 / 9** | **0 / 0 / 0** |
| epochs run | 25 / 58 / 46 | 22 / 44 / 46 | **11 / 12 / 14** | **35 / 36 / 29** |
| best epoch | 15 / 48 / 36 | 12 / 34 / 36 | **1 / 2 / 4** | **25 / 26 / 19** |
| Pearson spread | 0.012 | 0.034 | **0.502** | **0.030** |

**A13R's validation profile is identical to A13's.** The pathology is repaired,
not merely reduced. Per instruction, no A13P result is used as evidence here.

---

## 5. A13R placebo construction

For plane *i*, the entire board is filled with a single bit:

```python
bit_i(fen) = sha256(f"{SALTS[i]}|{fen}").digest()[0] & 1

SALTS = ("C6-A13R/placebo-plane-0", ..., "C6-A13R/placebo-plane-3")
```

- **Input** is `board.fen()`, the canonical python-chess FEN, treated as an
  opaque byte string. Verified `chess.Board(f).fen() == f` on **all 9,667
  dataset records and all 212 evaluation positions** (0 mismatches), so the hash
  input is well-defined however a board was reached. A test asserts a position
  reached by `push_uci` encodes identically to the same position built from FEN.
- **Domain separation:** four distinct salts, so the planes are independent
  draws rather than four views of one hash.
- **No chess state is consulted.** The generator never inspects castling rights,
  side to move, en-passant, piece placement, legality or material. Tests pass a
  `chess.Board` subclass whose accessors raise, and encoding still succeeds; and
  `placebo_values` accepts an arbitrary non-FEN string.
- **Deterministic and platform-independent** (SHA-256), so A13R models remain
  reproducible. A test pins the literal formula so a refactor cannot silently
  change the encoding.

---

## 6. Distribution comparison against A13

Measured over the 9,667 dataset records, **before training**:

| property | A13 castling | **A13R placebo** | matched? |
|---|---|---|---|
| marginal P(plane = 1) | 0.507 – 0.547 | **0.498 – 0.504** | yes |
| correlation with the label | −0.006 .. +0.025 | **−0.006 .. +0.006** | yes |
| spatially constant per board | yes | **yes** | yes |
| varies across positions (std) | 0.499 | **0.500** | yes |
| distinct 4-bit patterns present | 16/16 | **16/16** | yes |
| mean \|inter-plane correlation\| | **0.431** | **0.003** | **no** |
| effective rank (eigenvalues > 0.1) | **2** | **4** | **no** |

A13's four rights are near-duplicates in pairs — WK↔WQ *r* = 0.932, BK↔BQ
*r* = 0.908 — so they span roughly **two** independent dimensions. Independent
salts give **four**. This follows directly from the specified construction and is
carried into §16 as the principal limitation.

The mismatch is **asymmetric in the inference it permits**: A13R presents a
*larger* nuisance load than A13, so a regression in A13R is a conservative
finding, whereas a null result would have been weak. A13R regressed.

---

## 7. Experimental controls

Identical to A2/A13, verified before training:

| | value |
|---|---|
| dataset | `dataset_v1.jsonl`, 9,667 deduplicated records |
| **dataset sha256** | `5a689e3f37156a0540598cdebd0c7f42cdbb09976b5c9af61c013ebdcf3a430d` |
| split | 7,734 / 1,933, **split seed 42** |
| **train index sha256** | `9c1582d429845a1d7be4b8823301bdd01782c916819987aa3552fb2a0a2a1ff7` |
| **test index sha256** | `7d4c1ecdb4fbb0e701c449ec5e38f0670de298a5b735afaf2910dd8eddf61d0e` |
| label policy | `corrected_mate_white_perspective` (White-positive A2 labels) |
| **label sha256** | `424cd8a8b661a81364caab7e2fc34351a3748406e8cf57a174a695d26bd61493` |
| loss / optimiser | Huber, Adam, LR 1e-3, batch 64 |
| callbacks | ReduceLROnPlateau (0.5, patience 5), EarlyStopping (patience 10, restore best) |
| training cap | 100 epochs |
| seeds | 0, 1, 2 |
| fusion | frozen production Ridge `[330.9005, 32.3899, 0.7919, 5.1654, 0.0188]` |
| evaluator | `evaluation/evaluate.py`, unmodified |
| Stockfish | 17.1, depth 8, Threads=1, Hash=16MB, `Clear Hash` per position |
| suites | `extended` (n=160), `phase0_52` (n=52) |

Nothing was tuned on A13R results.

---

## 8. Implementation verification

All twelve required checks, covered by 35 tests in
`tests/unit/test_training_representation16r.py`:

| # | check | status |
|---|---|---|
| 1 | shape is `(8,8,16)` | pass |
| 2 | first 12 channels byte-identical to A2 | pass — both suites, `engine.board_to_planes`, A13 and A13P |
| 3 | each placebo channel spatially constant per board | pass |
| 4 | placebo values vary across positions | pass — std > 0.3 per plane; all 16 patterns occur |
| 5 | the four channels are deterministic | pass — formula pinned to SHA-256 literal |
| 6 | repeated encoding of the same FEN is identical | pass — including via a different move path |
| 7 | generator inspects only the FEN string | pass — exploding-accessor board; non-FEN string accepted |
| 8 | marginals approximately matched to A13 | pass — within 0.10 of A13's per plane |
| 9 | parameter count exactly 2,362,433 | pass |
| 10 | labels and split hashes identical to A2/A13 | pass |
| 11 | production engine/app/config/model/Ridge untouched | pass — §17 |
| 12 | no production path consumes the 16-plane model | pass — dispatch keys off `N_PLANES == 12` |

---

## 9. Training results, all three seeds

| run | rep | params | epochs | best ep | val Huber | test Huber | MAE | RMSE | Pearson | secs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 s0 | planes12 | 2,360,129 | 25 | 15 | 155.31 | 160.20 | 160.70 | 257.92 | 0.7450 | 185 |
| A2 s1 | planes12 | 2,360,129 | 58 | 48 | 157.32 | 162.48 | 162.98 | 262.09 | 0.7360 | 401 |
| A2 s2 | planes12 | 2,360,129 | 46 | 36 | 154.84 | 164.46 | 164.96 | 264.82 | 0.7332 | 329 |
| A13 s0 | planes16 | 2,362,433 | 22 | 12 | 188.22 | 194.03 | 194.53 | 307.25 | 0.6054 | 143 |
| A13 s1 | planes16 | 2,362,433 | 44 | 34 | 184.29 | 186.86 | 187.36 | 297.33 | 0.6389 | 383 |
| A13 s2 | planes16 | 2,362,433 | 46 | 36 | 180.86 | 189.85 | 190.35 | 300.32 | 0.6281 | 269 |
| **A13R s0** | planes16r | 2,362,433 | 35 | 25 | 188.10 | 192.54 | 193.04 | 308.70 | **0.6016** | 196 |
| **A13R s1** | planes16r | 2,362,433 | 36 | 26 | 183.35 | 193.53 | 194.03 | 306.74 | **0.6068** | 201 |
| **A13R s2** | planes16r | 2,362,433 | 29 | 19 | 188.69 | 200.73 | 201.23 | 316.39 | **0.5769** | 163 |

Weight hashes:

```
A13R seed 0  8cf0fac11b2cfbdb8e782e3c664c136c…
A13R seed 1  79643f08c670931ed04a45e2a46914ad…
A13R seed 2  f9302424815ae0ba7d5866c591db5b96…
```

Representation identifier `planes16r`; dataset / label / split hashes as §7.

```
Pearson   A2    [0.7332 .. 0.7450]  mean 0.7381
          A13   [0.6054 .. 0.6389]  mean 0.6241    real castling
          A13R  [0.5769 .. 0.6068]  mean 0.5951    meaningless hash bits
```

Because all three arms share identical labels, MAE/RMSE/Huber are directly
comparable, and A13R is worse than A2 on every one. **A13R's deficit vs A2
(−0.143 Pearson) is slightly larger than A13's (−0.114).** The A13 and A13R bands
touch only at their edges (0.6054 vs 0.6068).

---

## 10. Phase 0 (`phase0_52`, n=52) results

| metric | A13R s0 | A13R s1 | A13R s2 |
|---|---|---|---|
| legality % | 100 | 100 | 100 |
| top-1 agreement % | 11.54 | 19.23 | 17.31 |
| top-3 containment % | 19.23 | 30.77 | 26.92 |
| mean regret cp | 183.1 | 169.0 | 176.0 |
| median regret cp | 81 | 58 | 49 |
| p95 regret cp | 610 | 610 | 619 |
| max regret cp | 786 | 786 | 786 |
| blunder rate >300cp | 0.2667 | 0.2609 | 0.3043 |
| regret coverage % | 86.54 | 88.46 | 88.46 |
| spearman mean | 0.13 | 0.09 | 0.24 |

Mate statuses: s0/s1 `{none: 153, missed: 4, allows: 3}`, s2 `{none: 155,
missed: 4, allows: 1}`.

---

## 11. Extended (n=160) results

| metric | A13R s0 | A13R s1 | A13R s2 |
|---|---|---|---|
| legality % | 100 | 100 | 100 |
| top-1 agreement % | 14.38 | 13.75 | 11.25 |
| top-3 containment % | 25.62 | 26.88 | 25.62 |
| mean regret cp | 186.8 | 213.3 | 189.8 |
| median regret cp | 70 | 94 | 77 |
| p95 regret cp | 640 | 663 | 632 |
| max regret cp | 786 | 1136 | 786 |
| blunder rate >300cp | 0.2941 | 0.3399 | 0.3226 |
| regret coverage % | 95.62 | 95.62 | 96.88 |
| spearman mean | 0.16 | 0.20 | 0.19 |

Mate statuses: s0/s1 `{none: 153, missed: 4, allows: 3}`, s2 `{none: 155,
missed: 4, allows: 1}` — indistinguishable from A2 and A13 at these counts.

Legality is 100% in all six runs.

**By side to move, mean regret cp (3-seed mean):**

| suite | side | n | A0 | A2 | A13 | **A13R** | A2→A13R |
|---|---|---|---|---|---|---|---|
| extended | black | 75 | 175.54 | 139.07 | 157.98 | **184.63** | +45.56 (all 3) |
| extended | white | 85 | 106.82 | 145.60 | 190.13 | **207.38** | +61.77 (all 3) |
| phase0_52 | black | 22 | 91.79 | 63.18 | 109.85 | **182.21** | +119.03 (all 3) |
| phase0_52 | white | 30 | 113.33 | 149.45 | 167.30 | **172.02** | +22.58 (all 3) |

**By category, `extended`, A2→A13R:** endgame **+99.75**, middlegame +52.41,
opening +46.96, defensive +4.56, tactical +1.71.

**Selected-move changes A2→A13R:** 107 / 109 / 111 of 160 — the largest
behavioural divergence of any arm.

---

## 12. A2 → A13R comparison

**`extended` (n=160)** — every row below is worse in **all three** paired seeds:

| metric | s0 | s1 | s2 | mean Δ | A0 noise | verdict |
|---|---|---|---|---|---|---|
| mean regret cp | +53.15 | +71.70 | +37.46 | **+54.10** | 19.70 | **EXCEEDS** |
| median regret cp | +41 | +64 | +23 | **+42.67** | 10 | **EXCEEDS** |
| p95 regret cp | +110 | +124 | +101 | **+111.7** | 64 | **EXCEEDS** |
| max regret cp | +70 | +427 | +109 | **+202.0** | 18 | **EXCEEDS** |
| top-1 agreement % | −5.00 | −5.63 | −3.13 | **−4.59** | 3.74 | **EXCEEDS** |
| top-3 containment % | −7.50 | −7.50 | −3.76 | **−6.25** | 0.63 | **EXCEEDS** |
| blunder rate >300cp | +0.0915 | +0.1046 | +0.0742 | **+0.0901** | 0.0395 | **EXCEEDS** |
| legality % | 0 | 0 | 0 | 0 | 0 | unchanged |
| spearman mean | −0.05 | +0.01 | +0.01 | −0.01 | 0.01 | inconsistent |

**`phase0_52` (n=52)**

| metric | mean Δ | A0 noise | verdict |
|---|---|---|---|
| mean regret cp | **+62.22** | 31.65 | all 3 worse, **EXCEEDS** |
| median regret cp | **+32.17** | 7.5 | all 3 worse, **EXCEEDS** |
| p95 regret cp | **+109.3** | 53 | all 3 worse, **EXCEEDS** |
| max regret cp | **+168.0** | 0 | all 3 worse, **EXCEEDS** |
| blunder rate | **+0.1106** | 0.0624 | all 3 worse, **EXCEEDS** |
| top-1 agreement % | −3.20 | 5.77 | inconsistent |
| top-3 containment % | −3.21 | 3.85 | inconsistent |

**Seven metrics on `extended` and five on `phase0_52` degrade consistently and
exceed the noise band.** No metric improves consistently on either suite. This is
a coherent degradation across the headline set, not a cherry-picked cell.

---

## 13. A13 → A13R comparison

**`extended`**

| metric | s0 | s1 | s2 | mean Δ | A0 noise | verdict |
|---|---|---|---|---|---|---|
| mean regret cp | +39.88 | +13.70 | +10.81 | +21.46 | 19.70 | all 3 worse, exceeds |
| median regret cp | +40 | +26 | +38 | +34.67 | 10 | all 3 worse, exceeds |
| max regret cp | +28 | +420 | +28 | +158.7 | 18 | all 3 worse, exceeds |
| top-3 containment % | −10.63 | −3.74 | −5.63 | −6.67 | 0.63 | all 3 worse, exceeds |
| top-1 agreement % | −0.62 | −0.63 | −0.63 | −0.63 | 3.74 | all 3 worse, within noise |
| blunder rate | +0.0554 | +0.0044 | +0.0416 | +0.0338 | 0.0395 | all 3 worse, within noise |
| p95 regret cp | +46 | +31 | −31 | +15.33 | 64 | inconsistent |

**`phase0_52`:** median +20.0 and max +164.3 exceed; mean regret +31.32 and
blunder +0.051 are all-3-worse but within that suite's wider band.

**A13R is worse than A13**, consistently but by a smaller margin than either is
worse than A2. Real castling information therefore buys back *some* of the cost
of adding the channels — it does not come close to paying for it.

---

## 14. A0 noise-band comparison

The A0 band is the control arm's min–max over three seeds — the movement
attributable to training seed alone.

| metric (extended) | A0 band | A2 band | A13 band | **A13R band** |
|---|---|---|---|---|
| mean regret cp | 128.7–148.4 | 133.7–152.3 | 147.0–199.6 | **186.8–213.3** |
| median regret cp | 27–37 | 29–54 | 30–68 | **70–94** |
| p95 regret cp | 530–594 | 530–539 | 594–663 | **632–663** |
| top-1 agreement % | 16.88–20.62 | 14.38–19.38 | 11.88–15.00 | **11.25–14.38** |
| top-3 containment % | 33.12–33.75 | 29.38–34.38 | 30.62–36.25 | **25.62–26.88** |
| blunder rate | 0.2105–0.2500 | 0.2026–0.2484 | 0.2387–0.3355 | **0.2941–0.3399** |

A13R's mean-regret, median-regret and top-3 bands are **entirely disjoint** from
A0's and from A2's. Its top-1 band barely overlaps A13's and is disjoint from A0's.

---

## 15. Mechanism analysis

A3's saturation explanation was **not** assumed; A13 already showed low
saturation coexisting with regression.

### 15.1 Plane ablation — the decisive comparison

Zeroing the four added channels on the 1,933 held-out positions:

| arm | added channels | mean \|Δprediction\| | ΔPearson when zeroed |
|---|---|---:|---:|
| A13 | real castling | **84.11 cp** | **−0.095 / −0.089 / −0.108** |
| **A13R** | **hash noise** | **78.91 cp** | **+0.004 / +0.002 / +0.028** |
| *A13P* | *fixed constants* | *1,933.69 cp* | *(invalid arm)* |

Two facts sit side by side:

1. **The network allocates essentially the same leverage to pure noise as to
   real castling** — 78.9 cp vs 84.1 cp, both stable across seeds.
2. **Castling is genuinely used and the noise is genuinely a distractor.**
   Removing castling costs ~0.1 Pearson; removing the noise *improves* Pearson
   slightly.

Yet both arms regress against A2 by comparable amounts. The cost of the channels
is therefore not about what they carry. It is paid on arrival.

A13R's ablation magnitude is also **stable** (78.7 / 76.7 / 81.3) where A13P's
was wildly unstable (16.9 / 1,230.9 / 4,553.3) — independent confirmation that
A13P's flat direction does not exist here.

### 15.2 Raw output magnitude, saturation, candidate separation

5,639 candidate positions across both suites:

| arm | raw \|cnn\| | saturated (\|tanh\|>0.95) | within-position tanh spread (White) |
|---|---:|---:|---:|
| A0 | 98.8 | 6.2% | 0.6458 |
| A2 | 110.0 | 3.7% | 0.9547 |
| A13 | 111.2 | 3.1% | 1.1426 |
| **A13R** | **112.7** | **2.7%** | **1.2449** |

A13R's raw output magnitude is normal (112.7, within A2/A13's range — nothing
like A3's inflated 173.7), its saturation is the **lowest** of any arm, and its
candidate separation the **highest**. **Saturation does not explain A13R.** The
CNN is fully engaged in the fusion and still ranks worse.

### 15.3 White vs Black prediction error

| arm | White MAE | Black MAE | gap |
|---|---|---|---|
| A2 | 155.7 / 157.6 / 159.2 | 276.5 / 286.8 / 297.7 | +129.5 |
| A13 | 186.8 / 180.6 / 184.0 | 373.7 / 343.9 / 336.6 | +167.6 |
| **A13R** | 183.8 / 185.6 / 191.8 | 407.3 / 389.4 / 419.0 | **+218.2** |

A13R has the widest gap of any validly-trained arm, degrading both sides.

### 15.4 What the mechanism appears to be

Consistent with all of the above, and stated as a hypothesis rather than a
finding: with 7,734 training samples, four spatially-uniform channels give the
first convolution a cheap global degree of freedom that the network reliably
recruits (≈80 cp of prediction leverage in both arms). Because the signal is
global rather than spatial, it competes with rather than complements the piece
planes, and whatever it encodes — genuine castling rights or a hash bit — the net
effect on held-out accuracy is negative.

This is *not* demonstrated causally. §18 names the experiment that would test it.

---

## 16. Limitations

1. **Independence mismatch (principal).** A13's four planes have mean
   inter-plane correlation 0.431 and effective rank 2; A13R's are independent
   (0.003, rank 4). A13R therefore presents twice A13's effective nuisance
   dimensionality. This makes A13R's regression a *conservative* result, but it
   means the two arms are not distributionally identical.
2. **A hash placebo is memorisable but not generalisable.** A fixed function of
   the position can be fitted on the training split and yields nothing on the
   test split. Real castling generalises. This is intrinsic to any hash placebo.
3. **Entropy and structure differ.** A13R's bits are i.i.d. Bernoulli(0.5) over
   positions; castling rights are highly structured over a game (monotone loss of
   rights, correlated with phase). A13R does not reproduce that structure.
4. **Three seeds, no significance testing.** All statements are directions and
   magnitudes against a measured noise band.
5. **Small cells.** Category breakdowns at n=5–21 and mate counts at 1–4 of 160
   are not interpretable alone; the `tactical` and `defensive` cells move in ways
   that do not replicate across suites.
6. **Untuned recipe.** A13R used A2's exact hyperparameters, correct for a
   controlled experiment but not a test of what a tuned 16-channel model could do.
7. **Frozen production Ridge**; A13R's fusion compatibility was not measured and
   no matched-Ridge run was performed.
8. **The evaluation shim** is a path A0/A1/A2 did not use, though it changes
   exactly one binding and is test-covered.
9. **Nothing here generalises** beyond `dataset_v1`, this CNN, this recipe, and
   representation-as-input-plane.

---

## 17. Decision

**The evidence supports H2, the broadcast-channel hypothesis.**

Recording it in the form the interpretation gates specify:

> Under this dataset, architecture, training recipe and
> representation-as-input-plane design, **adding spatially-constant
> position-varying global-state channels to this CNN is associated with the
> regression**, and the association does not depend on the channels carrying
> chess meaning.

Supporting facts, none of which rests on a single metric:

1. A13R regresses against A2 on **seven** `extended` metrics and **five**
   `phase0_52` metrics, all consistent across three paired seeds, all exceeding
   the A0 noise band. No metric improves consistently on either suite.
2. A13R's offline Pearson deficit (−0.143) is **larger** than A13's (−0.114),
   despite carrying no information at all.
3. The network recruits near-identical leverage from noise (78.9 cp) and from
   castling (84.1 cp), while only castling is actually useful.
4. The A13P failure mode is demonstrably absent (val max/min 1.2, zero spikes,
   29–36 epochs, stable ablation), so this is not a repeat of that artifact.

**What this does not establish.** It does not show castling rights are useless —
§15.1 shows the opposite, that the model does extract value from them. It shows
that supplying them *as input planes* costs more than they return. And per §16.1
the comparison is not distributionally exact.

**Consequence for the programme:** stop adding global scalar state as image
planes. That direction has now been tested three times (A3, A13, A13R) and
regressed every time.

---

## 18. Recommended next phase

**Architectural change: inject global state after convolutional feature
extraction**, rather than as input planes.

```
piece planes (8,8,12) → Conv/BN stack → Flatten ─┐
                                                  ├→ concat → Dense(256) → … → output
global scalars (castling, side-to-move, …) ──────┘
```

This keeps the convolutional pathway exactly as A2 has it — A2 remains the
control — and gives global state a route that does not compete for spatial
filters. It directly tests §15.4's hypothesis: if the same castling bits help
when concatenated but hurt as planes, the mechanism is the input-plane design,
not the information.

Two caveats to set before running it. It changes the architecture, so parameter
count will no longer match A2 and the comparison becomes less clean than the
representation arms have been — that needs stating in the design. And it should
be run with **castling only** first, since A3 already showed side-to-move is
harmful under the A2 label policy (which makes it irrelevant to the target) and
en-passant inert at 0.27% coverage.

A cheaper intermediate, if an architecture change is too large a step: rerun A13
with the four castling planes but **a wider or regularised first layer**, to test
whether the cost is capacity contention specifically. Lower information value
than the concat arm, but it stays within the current architecture.

---

## Reproducing

```bash
python -m training.train --arm A13R --seed 0        # and 1, 2
python -m training.evaluate_arm --arm A13R --seed 0 # and 1, 2

python -m training.a2_analysis --arms A0 A2 A13 A13R \
    --contrasts A2:A13R A13:A13R \
    --out training/experiments/A13R/a13r_analysis.json
python -m training.a2_perspective_probe --arms A0 A2 A13 A13R --contrast A13:A13R \
    --out training/experiments/A13R/perspective_probe.json
python -m training.a2_saturation_probe --arms A0 A2 A13 A13R --contrast A13:A13R \
    --out training/experiments/A13R/saturation_probe.json
python -m training.a3_plane_ablation --arm A13R
```

Earlier arms' defaults are unchanged, so the A2, A3, A13 and A13P reports still
reproduce exactly. Artifacts are under the gitignored
`training/experiments/A13R/`.
