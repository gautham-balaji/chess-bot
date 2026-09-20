# C6-A13P — Placebo Constant-Plane Control (16 planes)

**Status:** run to completion, 3 seeds, both suites. **The control did not
execute validly** — see §5 and §11.
**Scope:** one representation variable relative to A13. No production code changed.
**Fusion:** frozen production Ridge. No matched-Ridge run.
**Statistical standing:** descriptive. Three seeds, no significance testing.

---

## Verdict up front

> **A13P cannot decide between the two hypotheses, because the placebo design
> itself creates a training pathology that A13 does not have.**
>
> All three A13P seeds stopped at epoch 11–14 with best epoch **1, 2 and 4**.
> Their *training* loss descended normally and tracked A2 almost exactly; their
> *validation* loss exploded and oscillated (max/min ratio **9.3**, with **5–9
> spikes above 2× the running best**, against **zero** such spikes for A2 and
> A13). Early stopping — patience 10, `restore_best_weights=True` — therefore
> returned near-initialisation checkpoints.
>
> **The reported A13P metrics measure an undertrained model, not a
> representation.** Forcing them into the H1/H2 decision would be wrong.

The mechanism is identified and measured (§11): information-free constant
channels create a weight direction that the training loss cannot constrain,
because BatchNorm removes a batch-constant offset in training mode. The ablation
is the clearest evidence — **the better-trained an A13P seed is, the more
catastrophically its output depends on channels carrying zero information**
(17 → 1,231 → 4,553 cp).

---

## 1. Objective

Distinguish:

- **H1** — A13 regresses because of the **castling content** carried by its four
  added planes.
- **H2** — A13 regresses because adding **any spatially-constant broadcast
  channel** to this CNN is harmful at this dataset size.

A13P was designed to hold shape, parameter count and every training variable
fixed while removing all chess information from the four added channels.

---

## 2. Exact controlled variable

**The contents of channels 12–15, and only that.**

| | A2 | A13 | **A13P** |
|---|---|---|---|
| channels 0–11 | 12 piece planes | identical | **identical** |
| channels 12–15 | *absent* | four castling rights | **four fixed constants (1.0)** |
| shape | (8,8,12) | (8,8,16) | **(8,8,16)** |
| parameters | 2,360,129 | 2,362,433 | **2,362,433** |

A13 and A13P are identical in shape, parameter count, labels, split, optimiser,
loss, callbacks and seeds. They differ only in what the four added channels hold.

---

## 3. Unchanged variables and hashes

All verified before training:

| | value |
|---|---|
| dataset | `dataset_v1.jsonl`, 9,667 deduplicated records |
| **dataset sha256** | `5a689e3f37156a0540598cdebd0c7f42cdbb09976b5c9af61c013ebdcf3a430d` ✓ |
| split | 7,734 train / 1,933 test, **split seed 42** |
| **train index sha256** | `9c1582d429845a1d7be4b8823301bdd01782c916819987aa3552fb2a0a2a1ff7` |
| **test index sha256** | `7d4c1ecdb4fbb0e701c449ec5e38f0670de298a5b735afaf2910dd8eddf61d0e` |
| label policy | `corrected_mate_white_perspective` |
| **label sha256** | `424cd8a8b661a81364caab7e2fc34351a3748406e8cf57a174a695d26bd61493` ✓ |
| loss / optimiser | Huber, Adam, LR 1e-3, batch 64 |
| callbacks | ReduceLROnPlateau (factor 0.5, patience 5), EarlyStopping (patience 10, restore best), monitor `val_loss` |
| max epochs | 100 |
| seeds | 0, 1, 2 |
| fusion | frozen production Ridge `[330.9005, 32.3899, 0.7919, 5.1654, 0.0188]` |
| evaluator | `evaluation/evaluate.py`, unmodified |
| Stockfish | 17.1, depth 8, Threads=1, Hash=16MB, `Clear Hash` per position |
| suites | `extended` (n=160), `phase0_52` (n=52) |

Dataset and label hashes match the values specified for this experiment exactly.
No hyperparameter was tuned.

---

## 4. Representation definition

`training/representation16p.py`, registered as `planes16p`:

```
plane  meaning                       values          spatial?
0-11   the 12-plane piece encoding   0/1             yes
12-15  PLACEBO: fixed constant 1.0   all-1           constant
```

`board_to_planes` reads the board for planes 0–11 only; planes 12–15 never
consult it. A test passes in a `chess.Board` subclass whose castling accessors
raise, and the placebo planes are still produced.

**Why all-ones rather than all-zeros.** An all-zero plane is provably inert — no
contribution to any convolution, no gradient — so the control would test only
whether the tensor has more columns. All-ones planes flow signal and gradient
into the first Conv2D. The layer is `Conv2D(64, 3x3, padding="same",
activation="relu")` followed by BatchNormalization, so the constant adds a fixed
spatial pattern to each filter's pre-activation (uniform in the interior, reduced
along the 1-pixel zero-padded border) and shifts that filter's ReLU threshold.

**Disclosed consequence.** With `PLACEBO_VALUE = 1.0`, a position holding all
four castling rights encodes *identically* under A13 and A13P. That is **2,878 of
9,667 records (29.77%)**. The two arms differ only on the remaining 70.23%. This
does not affect A13P's validity as an information-free encoding (its added
channels have exactly zero variance across positions) but it does mean the
A13↔A13P input contrast is not total.

**What this control does and does not separate.** A13's castling planes are
constant *within* a board but vary *across* boards — measured per-plane std ≈
0.50 over the 9,667 records — and that cross-position variation is precisely what
carries the information. A position-independent placebo therefore differs from
A13 in **two** ways at once: no chess semantics, *and* no across-position
variation at all. Even had A13P trained cleanly, it could not by itself separate
"castling semantics are harmful" from "any across-position-varying broadcast
channel is harmful". §15 names the arm that would.

---

## 5. Implementation checks

All nine, covered by 33 tests in `tests/unit/test_training_representation16p.py`:

| # | check | status |
|---|---|---|
| 1 | shape is `(8,8,16)` | pass |
| 2 | channels 0–11 identical to A2 | pass — all 212 suite positions, plus `engine.board_to_planes`, plus A13 |
| 3 | channels 12–15 spatially constant | pass |
| 4 | channels 12–15 carry no board information | pass — zero variance across all 9,667 records; parametrised falsification over castling / side-to-move / en-passant / unrelated positions; encoder does not read the board |
| 5 | exactly 2,362,433 parameters | pass |
| 6 | labels byte-identical to A2/A13 | pass |
| 7 | split identical to A2/A13 | pass |
| 8 | production encoder/engine unchanged | pass — §14 |
| 9 | A13 and A13P identical shape and parameter count | pass |

---

## 6. Per-seed training results

| run | rep | params | epochs | **best epoch** | val Huber | test Huber | MAE | RMSE | Pearson | secs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 s0 | planes12 | 2,360,129 | 25 | 15 | 155.31 | 160.20 | 160.70 | 257.92 | 0.7450 | 185 |
| A2 s1 | planes12 | 2,360,129 | 58 | 48 | 157.32 | 162.48 | 162.98 | 262.09 | 0.7360 | 401 |
| A2 s2 | planes12 | 2,360,129 | 46 | 36 | 154.84 | 164.46 | 164.96 | 264.82 | 0.7332 | 329 |
| A13 s0 | planes16 | 2,362,433 | 22 | 12 | 188.22 | 194.03 | 194.53 | 307.25 | 0.6054 | 143 |
| A13 s1 | planes16 | 2,362,433 | 44 | 34 | 184.29 | 186.86 | 187.36 | 297.33 | 0.6389 | 383 |
| A13 s2 | planes16 | 2,362,433 | 46 | 36 | 180.86 | 189.85 | 190.35 | 300.32 | 0.6281 | 269 |
| **A13P s0** | planes16p | 2,362,433 | **11** | **1** | 237.37 | 245.30 | 245.80 | 399.35 | **0.2245** | 75 |
| **A13P s1** | planes16p | 2,362,433 | **12** | **2** | 200.91 | 206.16 | 206.66 | 347.93 | **0.6238** | 88 |
| **A13P s2** | planes16p | 2,362,433 | **14** | **4** | 162.07 | 166.61 | 167.11 | 271.39 | **0.7266** | 80 |

Weight hashes (bit-stable artifact):

```
A13P seed 0  1dfc80f198222091cada268661a722d6…
A13P seed 1  f672678df21cb3d12d9be0ca106c8ea5…
A13P seed 2  7bcbb7ea301dc840ff2a0bf60fbc9cff…
```

Representation identifier: `planes16p`. Dataset / label / split hashes as §3.

### The runs did not converge

```
Pearson   A2    [0.7332 .. 0.7450]  mean 0.7381
          A13   [0.6054 .. 0.6389]  mean 0.6241
          A13P  [0.2245 .. 0.7266]  mean 0.5250     <- spread 0.50, not a band
```

A13P's spread is **0.50 Pearson wide**, against A2's 0.012 and A13's 0.034. That
is not seed noise around a value; it is three runs stopped at arbitrary points.

**Training loss was healthy.** First 11 epochs, A2 s0 vs A13P s0:

```
A2   s0 train  222.2 190.7 170.7 155.2 144.7 133.1 127.5 118.4 112.4 108.5 107.6
A13P s0 train  226.2 198.4 178.1 166.7 155.3 148.0 132.8 127.2 120.6 113.0 108.9
```

Nearly identical. **Validation loss was not:**

```
A2   s0 val    269.7 476.8 453.5 356.1 263.8 199.6 247.3 166.1 171.1 175.9 192.6
A13P s0 val    237.4 699.0 2164.8 1151.3 576.4 641.4 339.6 1552.6 1042.1 1147.7 595.3
```

| arm | val max/min | epochs with val > 2× running best |
|---|---:|---:|
| A2 | 1.5 / 2.7 / 3.1 | **0 / 0 / 0** |
| A13 | 1.2 / 1.2 / 1.3 | **0 / 0 / 0** |
| **A13P** | **6.0 / 9.1 / 12.8** | **5 / 8 / 9** |

Early stopping monitors `val_loss` with patience 10. Every A13P run stopped at
exactly best_epoch + 10, and `restore_best_weights=True` returned the epoch-1,
-2 and -4 weights.

---

## 7. Engine evaluation

Production Ridge, unmodified Phase 3 evaluator, via the existing plane shim
(`patched engine.board_to_planes -> training.representation16p.board_to_planes
(8, 8, 16)`, logged six times).

**`extended` (n=160)**

| metric | A13P s0 | A13P s1 | A13P s2 |
|---|---|---|---|
| legality % | 100 | 100 | 100 |
| top-1 agreement % | 19.38 | 21.88 | 18.12 |
| top-3 containment % | 35.62 | 37.50 | 29.38 |
| mean regret cp | 148.0 | 144.5 | 187.6 |
| median regret cp | 28 | 25 | 42 |
| p95 regret cp | 533 | 533 | 666 |
| max regret cp | 678 | 693 | 784 |
| blunder rate >300cp | 0.2597 | 0.2468 | 0.3137 |
| regret coverage % | 96.25 | 96.25 | 95.62 |
| spearman mean | 0.24 | 0.23 | 0.28 |

**`phase0_52` (n=52)**

| metric | A13P s0 | A13P s1 | A13P s2 |
|---|---|---|---|
| legality % | 100 | 100 | 100 |
| top-1 agreement % | 21.15 | 25.00 | 21.15 |
| top-3 containment % | 44.23 | 44.23 | 36.54 |
| mean regret cp | 120.6 | 108.4 | 132.0 |
| median regret cp | 21 | 16.5 | 19 |
| p95 regret cp | 516 | 516 | 579 |
| max regret cp | 614 | 614 | 614 |
| blunder rate >300cp | 0.2128 | 0.1957 | 0.2444 |
| regret coverage % | 90.38 | 88.46 | 86.54 |
| spearman mean | 0.29 | 0.23 | 0.27 |

**Mean regret cp by side to move** (mean over three seeds):

| suite | side | n | A0 | A2 | A13 | A13P | A2→A13P |
|---|---|---|---|---|---|---|---|
| extended | black | 75 | 175.54 | 139.07 | 157.98 | 181.07 | +41.98 |
| extended | white | 85 | 106.82 | 145.60 | 190.13 | 141.13 | −4.47 |

**By category, `extended`, A2→A13P:** middlegame +34.17, tactical +41.30,
opening +11.65, endgame −16.54, defensive −3.25.

Legality is 100% in all six runs.

---

## 8. A2 → A13P comparison

**`extended`**

| metric | s0 | s1 | s2 | mean Δ | A0 noise | verdict |
|---|---|---|---|---|---|---|
| mean regret cp | +14.33 | +2.85 | +35.32 | +17.50 | 19.70 | all 3 worse, within noise |
| blunder rate | +0.0571 | +0.0115 | +0.0653 | +0.0446 | 0.0395 | all 3 worse, **exceeds** |
| spearman mean | +0.03 | +0.04 | +0.10 | +0.0567 | 0.01 | all 3 better, **exceeds** |
| top-1 agreement % | 0.00 | +2.50 | +3.74 | +2.08 | 3.74 | inconsistent |
| top-3 containment % | +2.50 | +3.12 | 0.00 | +1.87 | 0.63 | inconsistent |

**`phase0_52`**

| metric | mean Δ | A0 noise | verdict |
|---|---|---|---|
| top-3 containment % | **+12.82** | 3.85 | all 3 better, **exceeds** |
| mean regret cp | +6.54 | 31.65 | inconsistent |
| blunder rate | +0.0509 | 0.0624 | inconsistent |

---

## 9. A13 → A13P comparison

**`extended`**

| metric | s0 | s1 | s2 | mean Δ | A0 noise | verdict |
|---|---|---|---|---|---|---|
| **top-1 agreement %** | +4.38 | +7.50 | +6.24 | **+6.04** | 3.74 | all 3 better, **exceeds** |
| **spearman mean** | +0.04 | +0.07 | +0.04 | **+0.050** | 0.01 | all 3 better, **exceeds** |
| mean regret cp | +1.06 | −55.15 | +8.67 | −15.14 | 19.70 | inconsistent |
| p95 regret cp | −61 | −99 | +3 | −52.33 | 64 | inconsistent |
| blunder rate | +0.021 | −0.0887 | +0.0327 | −0.0117 | 0.0395 | inconsistent |

A13P is consistently better than A13 on top-1 agreement and Spearman, and mixed
elsewhere. **This cannot be read as "constant channels are harmless",** because
§11 shows A13P's advantage on ranking metrics comes from its CNN being nearly
switched off inside the fusion, not from a better representation.

---

## 10. A0 noise-band comparison

The A0 band is the control arm's min–max over three seeds — what each metric does
from seed alone.

| metric (extended) | A0 band | A2 band | A13 band | A13P band |
|---|---|---|---|---|
| mean regret cp | 128.7–148.4 | 133.7–152.3 | 147.0–199.6 | 144.5–187.6 |
| top-1 agreement % | 16.88–20.62 | 14.38–19.38 | 11.88–15.00 | **18.12–21.88** |
| top-3 containment % | 33.12–33.75 | 29.38–34.38 | 30.62–36.25 | 29.38–37.50 |
| blunder rate | 0.2105–0.2500 | 0.2026–0.2484 | 0.2387–0.3355 | 0.2468–0.3137 |
| spearman mean | 0.21–0.22 | 0.18–0.21 | 0.16–0.24 | **0.23–0.28** |

Only two A2→A13P movements clear the band consistently: blunder rate (worse) and
Spearman (better). They point in opposite directions, which is itself a sign that
the arm is not measuring a coherent representational effect.

---

## 11. Mechanism probes

Per instruction, A3's saturation explanation was **not** assumed.

### 11.1 Placebo ablation — the decisive result

`python -m training.a3_plane_ablation --arm A13P` zeroes the four placebo
channels on the 1,933 held-out positions. Zeroing a channel that carries **no
information** should ideally change nothing.

| seed | baseline Pearson | mean \|Δprediction\| when zeroed | Pearson after |
|---|---:|---:|---:|
| 0 | 0.2245 | **16.9 cp** | 0.1900 |
| 1 | 0.6238 | **1,230.9 cp** | 0.4619 |
| 2 | 0.7266 | **4,553.3 cp** | 0.3077 |

**The dependence is monotone in how well the seed trained.** The better a run
fit, the more catastrophically its output relied on channels holding zero
information. A13's castling ablation, by contrast, is a stable 84.1 cp across
seeds.

That is the signature of an **unconstrained weight direction**. In training mode
BatchNorm subtracts the batch mean, and the offset a constant channel contributes
is identical for every sample, so it is removed exactly — the training loss is
blind to those 2,304 weights. They drift, the network's function comes to depend
on them, and only BatchNorm's statistics cancel the result. Remove the input and
the model collapses.

A13 does not have this failure mode: its planes vary across positions, so the
offset differs per sample and cannot be absorbed by a batch mean.

### 11.2 BatchNorm train/inference comparison

| arm | inference/training-mode MAE ratio (3 seeds) |
|---|---|
| A2 | 0.97 / 0.98 / 0.99 |
| A13 | 1.00 / 0.99 / 0.98 |
| A13P | 1.13 / 1.12 / 0.98 |

A13P's mismatch is visible but modest. **This understates the effect**, because
`restore_best_weights` saved the epoch-1/2/4 checkpoints — *before* the
divergence. The stored histories (§6) show what the discarded later epochs did.
A13P s0's Pearson is 0.2245 in inference mode but 0.4223 in training mode: the
same weights are much better when BatchNorm uses batch statistics.

### 11.3 Raw output magnitude, saturation, candidate separation

5,639 candidate positions across both suites:

| arm | raw \|cnn\| | saturated (\|tanh\|>0.95) | within-position tanh spread (White) |
|---|---:|---:|---:|
| A0 | 98.8 | 6.2% | 0.6458 |
| A2 | 110.0 | 3.7% | 0.9547 |
| A13 | 111.2 | 3.1% | 1.1426 |
| **A13P** | **61.4** | **0.3%** | **0.4386** |

Per seed, A13P's raw \|cnn\| is 28.0 / 78.0 / 78.2 and its White-side
within-position spread 0.047 / 0.331 / 0.938.

**A13P seed 0's CNN is effectively switched off inside the fusion**: a
within-position spread of 0.047 means the `330.9·tanh(cnn/200)` term barely
varies between candidate moves, so ranking falls back almost entirely on the
hand-crafted heuristics. That — not a good representation — is why A13P's top-1
agreement and Spearman look respectable in §9. The heuristics alone rank about as
well as a weak CNN.

### 11.4 White vs Black prediction error

| arm | White MAE | Black MAE | gap |
|---|---|---|---|
| A2 | 155.7 / 157.6 / 159.2 | 276.5 / 286.8 / 297.7 | +129.5 |
| A13 | 186.8 / 180.6 / 184.0 | 373.7 / 343.9 / 336.6 | +167.6 |
| **A13P** | 223.6 / 189.6 / 159.4 | 759.0 / 602.9 / 345.5 | **+378.2** |

A13P's gap tracks how undertrained each seed is (s0 worst, s2 best), consistent
with §6 rather than with any representational story.

---

## 12. Decision between the two hypotheses

**No decision is possible from this experiment.**

The pre-registered decision logic assumed A13P would land near A2 (→ H1) or near
A13 (→ H2), with an intermediate branch for mixed evidence. A13P landed in a
fourth regime the design did not anticipate: **the arm failed to train under the
fixed protocol**, for a reason specific to the placebo construction and not
shared by A13.

Recording the required statement in the form the instructions asked for, with its
condition unmet:

> Neither "the degradation is associated with the castling information rather
> than merely the addition of constant channels" nor "adding spatially-constant
> global-state channels to this CNN architecture is associated with the
> regression" is supported by A13P as run.

What can be said factually:

1. **Information-free constant channels are not a neutral control in a
   BatchNorm network.** They introduce a weight direction the training loss
   cannot constrain, and the resulting instability interacts with early stopping
   to terminate training near initialisation. §11.1 measures this directly.
2. **This failure mode is specific to *position-independent* constants.** A13's
   planes are spatially constant but vary across positions, so they do not create
   it. A13P therefore does not model A13's situation.
3. **The one weak signal available** is A13P seed 2, the best-trained of the
   three, which reached Pearson 0.7266 — close to A2's band (0.7332–0.7450) and
   well above A13's (0.6054–0.6389). If A13P had trained cleanly and continued to
   look like A2, that would favour H1. **One seed of three, from a run that
   stopped at epoch 14, is not evidence** — it is a reason to re-run, not a
   conclusion.

---

## 13. Limitations

1. **The control did not execute validly.** Everything in §7–§10 describes
   undertrained models. This is the dominant limitation and it subsumes most
   others.
2. **A position-independent placebo cannot separate semantics from
   across-position variation** even when it trains (§4). Two properties differ at
   once.
3. **29.77% input overlap with A13** by construction (`PLACEBO_VALUE = 1.0`
   equals A13's encoding wherever all four rights are present).
4. **Three seeds, no significance testing.** A13P's 0.50-wide Pearson spread is
   not a noise band.
5. **Dataset- and architecture-specific.** Nothing here generalises beyond
   `dataset_v1`, this CNN, this recipe, and representation-as-input-plane.
6. **Frozen production Ridge**; A13P's fusion compatibility was not measured and
   no matched-Ridge run was performed.
7. **The evaluation shim** is a path A0/A1/A2 did not use, though it changes
   exactly one binding and is test-covered.
8. **§11.2 understates the BatchNorm mismatch** because only best-epoch weights
   were saved; no per-epoch checkpoints exist to measure the divergence directly.

---

## 14. Production-safety verification

- `engine.py`, `app.py`, `config.py`, `evaluation/`, `models/`, baseline and
  regression fixtures: **unmodified** — `git status` reports no changes under any
  of them.
- **Production model hash unchanged:** `cnn_model.keras` `972d81199a1355667fca554b…`
- **Production Ridge hash unchanged:** `weight_model.pkl` `59a731127cc9b440c866c7a5…`,
  coefficients still `[330.9005, 32.3899, 0.7919, 5.1654, 0.0188]`
- `mlp_model.pkl`, `rf_model.pkl`, `scaler.pkl` digests unchanged.
- **Production engine still uses (8,8,12):** `engine.board_to_planes(chess.Board())`
  returns shape `(8, 8, 12)`; two tests assert the shipped encoder is unchanged.
- All three A13P evaluation runs logged `verified: production models/ untouched`.
- All A13P artifacts are under `training/experiments/A13P/`, inside the
  gitignored `training/experiments/` directory (confirmed by `git check-ignore`).
- **Full test suite: 564 passed, 10 xfailed, 0 failed.**

---

## 15. Recommended next step

**Re-run the control with a design that does not create the pathology.** Two
options, in order of preference:

**(a) `A13R` — a variation-matched placebo.** Four spatially-constant planes whose
values vary across positions with the *same marginal distribution as A13's
castling planes* (~50% ones), assigned by a deterministic function of the FEN
that carries no chess meaning — e.g. bits of a hash of the position string.

- This removes the BatchNorm flat direction, because the offset now differs per
  sample, so the arm can actually train.
- It matches A13 on **both** channel geometry *and* across-position variation,
  leaving **semantic content** as the single remaining difference — which is
  exactly the H1/H2 contrast.
- The original instruction excluded a per-position hash on the grounds that it
  introduces an arbitrary position-dependent signal. That concern is real: such
  planes are pure label-noise-free distractors. But A13P has now demonstrated
  that the alternative — position-independent constants — is *not inert either*;
  it is pathological in a different and less interpretable way. Between an
  interpretable distractor and an untrainable arm, the distractor is the better
  control. This is a recommendation for review, not a change made unilaterally.

**(b) Re-run A13P with early stopping relaxed** (larger patience, or select on a
smoothed validation curve) so the arm reaches convergence. This keeps the
placebo definition but changes a callback, which departs from the fixed protocol
— so it needs explicit approval, and its results would not be directly
comparable to A2/A13 on training length.

Option (a) is the stronger experiment; option (b) is the cheaper one.

**Independently of the control**, the ablation result in §11.1 is worth acting
on: a CNN whose output moves by thousands of centipawns when an
information-free input channel is zeroed is fragile in a way that would matter
for any future representation work. It reinforces the direction already flagged
at the end of the A13 report — that global scalar state probably should not be
injected as input planes at all, but concatenated after convolutional feature
extraction.

---

## Reproducing

```bash
python -m training.train --arm A13P --seed 0        # and 1, 2
python -m training.evaluate_arm --arm A13P --seed 0 # and 1, 2

python -m training.a2_analysis --arms A0 A2 A13 A13P \
    --contrasts A2:A13P A13:A13P \
    --out training/experiments/A13P/a13p_analysis.json
python -m training.a2_perspective_probe --arms A0 A2 A13 A13P --contrast A13:A13P \
    --out training/experiments/A13P/perspective_probe.json
python -m training.a2_saturation_probe --arms A0 A2 A13 A13P --contrast A13:A13P \
    --out training/experiments/A13P/saturation_probe.json
python -m training.a3_plane_ablation --arm A13P
python -m training.a13p_bn_diagnostic
```

Earlier arms' defaults are unchanged, so the A2, A3 and A13 reports still
reproduce exactly.
