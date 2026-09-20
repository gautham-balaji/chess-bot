# C6-A14 — Post-Convolution Castling Injection

**Status:** complete, 3 seeds, both suites.
**Scope:** one architectural variable relative to A2. No production code changed.
**Fusion:** frozen production Ridge. No matched-Ridge run.
**Statistical standing:** descriptive. Three seeds, no significance testing.

---

## Result up front

> **Moving castling out of the convolutional input repairs the A3/A13/A13R
> regression completely — and buys nothing.**
>
> A14 recovers A2's performance on every model-level metric (test Huber 161.7 /
> 161.7 / 165.1 against A2's 160.2 / 162.5 / 164.5; Pearson 0.7362 mean against
> A2's 0.7381) where A13 and A13R both lost ~0.11 Pearson. At engine level,
> A2 → A14 is **inside the A0 noise band and directionally inconsistent on
> 9 of 10 headline metrics on both suites**.
>
> The ablation says why. Zeroing A14's four castling scalars moves predictions
> by **0.50 cp** and changes Pearson by **0.0000**. A13's castling planes moved
> predictions by 84.11 cp; A13R's hash noise by 78.91 cp. **The network does not
> use the castling features at all.** A14 is A2 plus 1,024 parameters the
> optimiser declines to recruit.

**Both halves of this matter.** The architectural hypothesis is *confirmed on
the harm side*: the damage in A3/A13/A13R was caused by where global state
entered the network, not by what it contained. It is *refuted on the benefit
side*: given a place where it can be ignored safely, this recipe ignores it.
Castling is not a usable win for this model at this dataset size.

---

## 1. Objective

A3, A13, A13P and A13R all added spatially-constant, position-varying channels
to the **convolutional input** and all four regressed against A2 — including
A13R, whose channels contained nothing but hash noise. The surviving hypothesis
from that sequence was that the cost is paid on **arrival**: a global scalar
broadcast across an 8×8 convolutional input is harmful regardless of content.

A14 tests that directly by changing **only the entry point**. The castling
information is identical to A13's. The convolutional input is identical to A2's.
If the regression follows the information, A14 should regress like A13. If it
follows the entry point, A14 should look like A2.

This is the final architectural follow-up in the C6 representation branch. It is
not a new placebo and adds no new representation content.

---

## 2. A2 control architecture

Unchanged, and used as the control exactly as committed:

```
Input (8, 8, 12)                     12 piece planes
  → Conv2D(64, 3×3, same, relu)      6,976
  → BatchNormalization               256
  → Conv2D(128, 3×3, same, relu)     73,856
  → BatchNormalization               512
  → Conv2D(128, 3×3, same, relu)     147,584
  → BatchNormalization               512
  → Flatten                          8192
  → Dense(256, relu)                 2,097,408
  → Dropout(0.3)
  → Dense(128, relu)                 32,896
  → Dropout(0.2)
  → Dense(1)                         129
                                     ─────────
                             total   2,360,129
```

---

## 3. A14 architecture

The same Conv/BN stack, the same dense head, one `Concatenate` inserted between
`Flatten` and the first `Dense`:

```
board tensor (8, 8, 16)   ── TRANSPORT container, not a convolutional input
  │
  ├─ [:, :, :, :12] ──→ Conv2D(64) → BN → Conv2D(128) → BN → Conv2D(128) → BN
  │                       → Flatten ─────────────────────── 8192 ──┐
  │                                                                ├─ Concatenate (8196)
  └─ [:, 0, 0, 12:] ──→ 4 castling scalars ─────────────── 4 ──────┘
                                                                   │
                        Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.2) → Dense(1)
```

### Why the tensor is still 16 channels wide

Every consumer in this repository moves a position through the network as **one
numpy array** — `engine.cnn_evaluate` does
`cnn_model.predict(np.expand_dims(board_to_planes(b), 0))`, and the engine's two
batched call sites do `cnn_model.predict(np.array(tensors))`. A genuine
two-input Keras model could not be fed without editing those production call
sites, which this experiment is forbidden to do.

Channels 12–15 therefore broadcast the four castling bits over all 64 cells
**only so the bits can ride inside the same array**. The split is the graph's
first operation, so the convolutional stack's input is `(None, 8, 8, 12)` and no
castling value ever reaches a convolution. This is asserted against the built
Keras graph, not against documentation (§6).

A consequence worth stating: **A13 and A14 receive bit-identical input tensors.**
The two arms differ only in where the network consumes channels 12–15, which
makes the A13 ↔ A14 contrast exact.

### Castling feature order (canonical, deterministic)

| index | channel | feature |
|---|---|---|
| 0 | 12 | `white_kingside_castling` |
| 1 | 13 | `white_queenside_castling` |
| 2 | 14 | `black_kingside_castling` |
| 3 | 15 | `black_queenside_castling` |

Same order as `representation16` (A13) and as the castling block of
`representation18` (A3), so the arms stay directly comparable.

---

## 4. Exact parameter-count difference

| arm | route | params | Δ vs A2 | where the delta lives |
|---|---|---:|---:|---|
| A2 | — | 2,360,129 | — | — |
| A13 / A13P / A13R | input plane | 2,362,433 | **+2,304** | first Conv2D kernel, 3·3·4·64 |
| **A14** | **post-conv** | **2,361,153** | **+1,024** | `Dense(256)` kernel, 4·256 |

A14's convolutional and BatchNormalization parameters total **229,696**,
**bit-identical to A2's**. The first Conv2D has 6,976 parameters in both arms
(A13's has 9,280). `Dense(256)`'s fan-in grows 8192 → 8196 and nothing else
changes.

Parameter count was not matched to A2 — this is an architecture experiment and
matching was explicitly not required. It is recorded exactly instead.

---

## 5. Experimental controls

Identical to A2/A13/A13R, verified **before** training:

| | value |
|---|---|
| dataset | `dataset_v1.jsonl`, 9,667 records |
| **dataset sha256** | `5a689e3f37156a0540598cdebd0c7f42cdbb09976b5c9af61c013ebdcf3a430d` |
| split | 7,734 / 1,933, **split seed 42** |
| **train index sha256** | `9c1582d429845a1d7be4b8823301bdd01782c916819987aa3552fb2a0a2a1ff7` |
| **test index sha256** | `7d4c1ecdb4fbb0e701c449ec5e38f0670de298a5b735afaf2910dd8eddf61d0e` |
| label policy | `corrected_mate_white_perspective` (A2 White-positive labels) |
| **label sha256** | `424cd8a8b661a81364caab7e2fc34351a3748406e8cf57a174a695d26bd61493` |
| loss / optimiser | Huber, Adam, LR 1e-3, batch 64 |
| callbacks | ReduceLROnPlateau (0.5, patience 5), EarlyStopping (patience 10, restore best) |
| training cap | 100 epochs |
| seeds | 0, 1, 2 |
| fusion | frozen production Ridge `weight_model.pkl` |
| evaluator | `evaluation/evaluate.py`, unmodified |
| Stockfish | 17.1, depth 8, Threads=1, Hash=16MB, `Clear Hash` per position |
| suites | `extended` (n=160), `phase0_52` (n=52) |

All three hashes are byte-identical to A2, A13 and A13R. Nothing was tuned on
A14 results.

Not added, as specified: side-to-move, en-passant, hash features, extra
heuristics, new losses, new optimizers, new Ridge, hyperparameter tuning. The
only layer added anywhere is the `Concatenate` required to wire the injection.

---

## 6. Implementation verification

All fourteen required checks, covered by **68 tests** in
`tests/unit/test_training_representation12c4.py`. The checks that matter most are
run against the **built Keras graph**, not the docstring.

| # | check | status |
|---|---|---|
| 1 | same first-12 piece representation as A2 | pass — all 212 suite positions, plus `engine.board_to_planes` |
| 2 | convolutional input shape is `(8,8,12)` | pass — `conv_layers[0].input.shape == (None,8,8,12)` |
| 3 | castling never enters the convolutional tensor | pass — conv input channels are `[12, 64, 128]`; feeding two tensors differing **only** in channels 12–15 yields bit-identical Flatten output |
| 4 | four castling scalars extracted correctly | pass — 10 parameterised positions + agreement with `python-chess` on all 212 suite positions |
| 5 | feature order deterministic and documented | pass — order pinned, matches A13's, stable across repeated calls |
| 6 | a position's castling scalars are correct | pass — see #4; transport channels carry the scalars unchanged |
| 7 | concatenation occurs after Flatten | pass — `Concatenate` inputs are `[(None,8192), (None,4)]`; no `Dense` precedes it |
| 8 | output shape unchanged | pass — `(None, 1)`, and the engine's single-position call shape works |
| 9 | labels and split hashes identical to A2 | pass — all three sha256 values match exactly (§5) |
| 10 | existing Conv/BN stack unchanged | pass — layer configs, shapes and the 229,696 conv+BN parameter total all identical to A2 |
| 11 | production engine/app/config/model/Ridge untouched | pass — §14 hash verification |
| 12 | production 12-plane model remains the default path | pass — `evaluate_arm` dispatch still keys off `N_PLANES == 12`; no production file mentions A14 |
| 13 | parameter count recorded and compared | pass — 2,361,153, exactly A2 + 4·256 (§4) |
| 14 | no A13/A13R representation code used | pass — module imports only `training.representation`; source contains no reference to `representation16`/`representation18`/`hashlib` |

Two further guards worth naming:

- **Spatial constancy.** The model reads the scalars from cell `(0,0)`, which is
  exact only if the broadcast is uniform. Asserted over all 9,667 dataset records
  **and** both evaluation suites.
- **Engine loadability.** `engine.py` calls `load_model(path, compile=False)`
  with `safe_mode` on. A test saves and reloads the A14 graph through exactly
  that call and checks prediction equality — a graph that cannot deserialise
  that way would not be evaluable at all.

Full suite after the change: **666 passed, 10 xfailed**. One pre-existing pinned
arm-roster assertion in `tests/unit/test_training_representation18.py` was
updated to include A14; no other existing test needed changing.

---

## 7. Three-seed training results

| run | arch | params | epochs | best ep | val Huber | test Huber | MAE | RMSE | Pearson | secs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 s0 | sequential | 2,360,129 | 25 | 15 | 155.31 | 160.20 | 160.70 | 257.92 | 0.7450 | 185 |
| A2 s1 | sequential | 2,360,129 | 58 | 48 | 157.32 | 162.48 | 162.98 | 262.09 | 0.7360 | 401 |
| A2 s2 | sequential | 2,360,129 | 46 | 36 | 154.84 | 164.46 | 164.96 | 264.82 | 0.7332 | 329 |
| A13 s0 | sequential | 2,362,433 | 22 | 12 | 188.22 | 194.03 | 194.53 | 307.25 | 0.6054 | 143 |
| A13 s1 | sequential | 2,362,433 | 44 | 34 | 184.29 | 186.86 | 187.36 | 297.33 | 0.6389 | 383 |
| A13 s2 | sequential | 2,362,433 | 46 | 36 | 180.86 | 189.85 | 190.35 | 300.32 | 0.6281 | 269 |
| A13R s0 | sequential | 2,362,433 | 35 | 25 | 188.10 | 192.54 | 193.04 | 308.70 | 0.6016 | 196 |
| A13R s1 | sequential | 2,362,433 | 36 | 26 | 183.35 | 193.53 | 194.03 | 306.74 | 0.6068 | 201 |
| A13R s2 | sequential | 2,362,433 | 29 | 19 | 188.69 | 200.73 | 201.23 | 316.39 | 0.5769 | 163 |
| **A14 s0** | **post-conv** | **2,361,153** | 37 | 27 | 154.93 | **161.65** | 162.15 | 259.37 | **0.7427** | 218 |
| **A14 s1** | **post-conv** | **2,361,153** | 48 | 38 | 160.56 | **161.74** | 162.24 | 258.75 | **0.7441** | 261 |
| **A14 s2** | **post-conv** | **2,361,153** | 26 | 16 | 158.97 | **165.12** | 165.62 | 269.42 | **0.7217** | 143 |

**A14 lands inside A2's band on every model-level metric.** Mean Pearson 0.7362
vs A2's 0.7381; A13 0.6241, A13R 0.5951. Training is healthy — epochs run 26–48
with best epochs well clear of initialisation, and none of the A13P-style
val-loss pathology.

Model weight hashes: `b6c401ce…` / `a7521c9a…` / `f834d55a…`.

---

## 8. Phase 0 (`phase0_52`, n=52) results

Three-seed means, production fusion:

| arm | legality | top-1 % | top-3 % | mean regret | median | p95 | max | blunder | coverage | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 100 | 21.79 | 40.38 | 104.31 | 20.83 | 506.3 | 614.0 | 0.154 | 87.18 | 0.233 |
| A2 | 100 | 19.23 | 28.85 | 113.81 | 30.50 | 503.7 | 618.0 | 0.167 | 88.46 | 0.267 |
| A13 | 100 | 16.67 | 29.49 | 144.72 | 42.67 | 554.0 | 621.7 | 0.226 | 87.82 | 0.173 |
| A13R | 100 | 16.03 | 25.64 | 176.04 | 62.67 | 613.0 | 786.0 | 0.277 | 87.82 | 0.153 |
| **A14** | **100** | **17.95** | **31.41** | **118.03** | **26.00** | **506.7** | **617.7** | **0.180** | **89.10** | **0.167** |

Per-seed A14: top-1 19.23 / 15.38 / 19.23; mean regret 123.85 / 114.74 / 115.50;
median 19.0 / 40.0 / 19.0. Legality 100% in every run.

Mate statuses, A14: `{none 47, missed_forced_mate 4, allows_forced_mate 1}` /
`{46, 4, 2}` / `{46, 4, 2}` — matching A2's `{46, 4, 2}` in all three seeds.

**A2 → A14 paired by seed:** 9 of 10 metrics inside the A0 noise band and
directionally inconsistent. The single exception is Spearman (−0.10 mean,
consistent, exceeds the 0.06 band) — see §10 for why this does not survive
cross-suite reading.

## 9. Extended (n=160) results

| arm | legality | top-1 % | top-3 % | mean regret | median | p95 | max | blunder | coverage | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 100 | 18.96 | 33.54 | 139.29 | 31.33 | 565.7 | 684.0 | 0.230 | 95.21 | 0.213 |
| A2 | 100 | 17.71 | 32.29 | 142.53 | 37.67 | 533.3 | 700.7 | 0.229 | 95.62 | 0.193 |
| A13 | 100 | 13.75 | 32.71 | 175.17 | 45.67 | 629.7 | 744.0 | 0.285 | 96.46 | 0.200 |
| A13R | 100 | 13.13 | 26.04 | 196.63 | 80.33 | 645.0 | 902.7 | 0.319 | 96.04 | 0.183 |
| **A14** | **100** | **17.29** | **32.50** | **144.05** | **39.17** | **544.3** | **715.7** | **0.233** | **95.83** | **0.213** |

Per-seed A14: top-1 16.25 / 19.38 / 16.25; mean regret 144.81 / 140.01 / 147.33;
median 46.5 / 38.0 / 33.0.

**White/Black split, mean regret:** A14 133.3 / 142.0 / 133.7 (White) against
157.6 / 137.8 / 162.7 (Black). Paired against A2 this is −9.29 White, +13.61
Black — i.e. A14 *narrows* A2's White-side deficit slightly and gives some back
on Black. Both directions are inside the A0 band.

**Phase/category split, A2 → A14 mean regret delta:** defensive +10.83,
endgame −6.79, middlegame −9.55, opening +21.58, tactical −78.48. The tactical
figure rests on n=10 with extreme per-seed spread (A2 seed 1 scored 3.0 cp, A14
seed 1 scored −13.75) and should not be read as a finding.

**Selected-move churn, A2 → A14:** 57 / 60 / 57 of 160 moves change, roughly
evenly split by side. The arms are not producing the same engine — they are
producing a differently-wrong engine of the same quality.

**A2 → A14 paired by seed:** 9 of 10 metrics inside the A0 band and
inconsistent. The exception is Spearman (+0.02, consistent, band 0.01) — pointing
the *opposite* way to phase0's Spearman result.

---

## 10. A2 → A14 comparison

Pre-registered decision rule: a change counts only if it exceeds the A0
seed-noise band and is consistent across all three paired seeds.

**`extended`:**

| metric | per-seed Δ | mean Δ | A0 band | verdict |
|---|---|---:|---:|---|
| mean regret cp | +11.12 / −1.60 / −4.96 | +1.52 | 19.70 | within, inconsistent |
| median regret cp | +17.5 / +8 / −21 | +1.50 | 10.0 | within, inconsistent |
| p95 regret cp | +1 / −3 / +35 | +11.0 | 64.0 | within, inconsistent |
| top-1 % | −3.13 / 0 / +1.87 | −0.42 | 3.74 | within, inconsistent |
| top-3 % | −2.50 / −3.76 / +6.87 | +0.20 | 0.63 | within, inconsistent |
| blunder rate | +0.038 / −0.007 / −0.020 | +0.004 | 0.040 | within, inconsistent |
| legality % | 0 / 0 / 0 | 0 | 0 | unchanged |
| **Spearman** | +0.01 / +0.03 / +0.02 | **+0.02** | 0.01 | **exceeds, better** |
| regret coverage % | +0.63 / 0 / 0 | +0.21 | 0.62 | within, inconsistent |

**`phase0_52`:**

| metric | per-seed Δ | mean Δ | A0 band | verdict |
|---|---|---:|---:|---|
| mean regret cp | +19.52 / −13.0 / +6.13 | +4.22 | 31.65 | within, inconsistent |
| median regret cp | +3.5 / +5.5 / −22.5 | −4.50 | 7.50 | within, inconsistent |
| p95 regret cp | +53 / −52 / +8 | +3.0 | 53.0 | within, inconsistent |
| top-1 % | −3.85 / −3.85 / +3.85 | −1.28 | 5.77 | within, inconsistent |
| top-3 % | −1.92 / −1.93 / +11.54 | +2.56 | 3.85 | within, inconsistent |
| blunder rate | +0.039 / −0.022 / +0.022 | +0.013 | 0.062 | within, inconsistent |
| legality % | 0 / 0 / 0 | 0 | 0 | unchanged |
| **Spearman** | −0.13 / −0.11 / −0.06 | **−0.10** | 0.06 | **exceeds, worse** |
| regret coverage % | +1.92 / 0 / 0 | +0.64 | 1.92 | within, inconsistent |

**The two suites' only significant metrics contradict each other.** Spearman
improves consistently on `extended` (+0.02) and degrades consistently on
`phase0_52` (−0.10). Reporting either alone would be cherry-picking, which the
decision rule forbids. Every other metric on both suites is within noise and
inconsistent in sign.

**Verdict: A14 is equivalent to A2 at engine level.**

### A13 → A14: the regression is fully repaired

| metric | suite | mean Δ | verdict |
|---|---|---:|---|
| mean regret cp | extended | **−31.12** | exceeds band, better |
| p95 regret cp | extended | **−85.33** | exceeds band, better |
| top-1 % | extended | +3.54 | better (band 3.74) |
| blunder rate | extended | −0.052 | exceeds band |
| mean regret cp | phase0_52 | −26.69 | better |
| blunder rate | phase0_52 | −0.046 | better |

And at model level the repair is total: Pearson 0.624 → 0.736, back to A2's
0.738.

---

## 11. Mechanism analysis

### 11.1 Ablation — the decisive measurement

Zeroing the four castling channels on the 1,933 held-out positions:

| arm | channels carry | mean \|Δprediction\| | ΔPearson when zeroed |
|---|---|---:|---:|
| A13 | real castling, **into conv** | 84.11 cp | −0.095 / −0.089 / −0.108 |
| A13R | hash noise, **into conv** | 78.91 cp | +0.004 / +0.002 / +0.028 |
| **A14** | **real castling, post-conv** | **0.50 cp** | **0.0000 / 0.0000 / −0.0001** |

Per-seed A14: 0.61 / 0.42 / 0.46 cp, max single-position change 4.67 cp. The
script's own verdict: *"the added planes are effectively IGNORED."*

This is the whole result in one line. **The same four bits that command 84 cp of
leverage when fed to the first convolution command 0.5 cp when offered to the
dense head.**

### 11.2 Output magnitude, saturation, candidate separation

5,639 candidate positions across both suites:

| arm | raw \|cnn\| | saturated (\|tanh\|>0.95) | within-position tanh spread (White) |
|---|---:|---:|---:|
| A0 | 98.8 | 6.2% | 0.6458 |
| A2 | 110.0 | 3.7% | 0.9547 |
| A13 | 111.2 | 3.1% | 1.1426 |
| **A14** | **98.3** | **2.9%** | **0.9824** |

A14's magnitude, saturation and candidate separation all sit in A2's
neighbourhood. Saturation does not explain anything here, and nothing is
pathological.

### 11.3 White vs Black prediction error (held-out split)

| arm | White MAE | Black MAE | gap |
|---|---|---|---|
| A2 | 155.7 / 157.6 / 159.2 | 276.5 / 286.8 / 297.7 | +129.5 |
| A13 | 186.8 / 180.6 / 184.0 | 373.7 / 343.9 / 336.5 | +167.6 |
| A13R | 183.8 / 185.6 / 191.8 | 407.3 / 389.4 / 419.0 | +218.2 |
| **A14** | **157.4 / 157.3 / 159.3** | **272.1 / 276.0 / 311.3** | **+128.5** |

A14 reproduces A2's asymmetry almost exactly, where both input-plane arms
widened it substantially.

### 11.4 What the mechanism appears to be

Stated as a hypothesis consistent with all of the above, not as a causal
demonstration:

1. **The harm was structural, and it is now explained.** A spatially-constant
   channel entering `Conv2D` is multiplied by 2,304 new weights that see the
   same value at every one of 64 positions. With 7,734 training samples the
   optimiser reliably recruits that cheap global degree of freedom — ~80 cp of
   leverage in *both* A13 and A13R, whether the channel carried castling rights
   or a hash bit — and because the signal is global rather than spatial it
   competes with the piece planes instead of complementing them. Removing the
   entry point removes the recruitment, and the regression disappears entirely.
   **A14 confirms the A13R conclusion by the reverse route.**

2. **The benefit never materialises.** Offered through 1,024 dense weights
   downstream of Flatten, castling has to compete with 8,192 convolutional
   features for the same `Dense(256)` units, with no structural advantage and no
   weighting in its favour. The optimiser's answer is to leave it alone. Note
   A13's ablation showed castling *is* genuinely informative (removing it cost
   ~0.1 Pearson there) — so this is not "castling is useless information", it is
   **"this recipe has no mechanism that makes four extra scalars worth
   learning among 8,192 others"**.

Together: global state injected after spatial feature extraction is *safe*, and
under this dataset, architecture and training recipe it is also *inert*.

---

## 12. Limitations

1. **A14 is inert, not merely equal.** The 0.50 cp ablation means the
   equivalence to A2 is not evidence that post-conv injection *works* — it is
   evidence that it *does nothing*. Any claim that "global state should be
   injected after spatial features" is supported by this experiment only in the
   weak, negative sense that doing so stops the harm.
2. **One injection design was tested.** Raw concatenation into a 8,196-wide
   dense layer is the minimal wiring, and deliberately so. A design that gave
   castling structural weight — a small dedicated sub-network, a gating or
   FiLM-style modulation, or feature scaling — might recruit it. None was tried,
   because the brief was one clean controlled answer, not an experiment tree.
3. **Frozen production Ridge.** As in every C6 arm, the fusion weights were
   fitted against the original CNN's output scale. This biases all arms
   identically so the comparison stays fair, but no arm should be read as the
   best this architecture can do.
4. **Three seeds, descriptive only.** No significance testing. The A0 noise band
   is the decision instrument, and on several metrics the band is comparable to
   or larger than the effect.
5. **Contradictory Spearman.** The one metric that clears the band does so in
   opposite directions on the two suites. This is reported rather than resolved;
   with n=52 and n=160 and three seeds there is no basis to prefer either.
6. **Small suites.** The category breakdown in §9 rests on cells as small as
   n=4–10. The tactical −78 cp figure in particular is noise.
7. **Not deployable as-is.** A14's input tensor is 16 channels wide, so the
   unmodified engine cannot feed it; evaluation went through the same
   experiment-only shim A3/A13/A13R used. Shipping A14 would require a
   production change — for an arm that measurably ignores the feature it adds.

---

## 13. Decision

**A14 is similar to A2.** Per the pre-registered rule, this is the
"do not invent another experiment" branch.

- Engine level: 9 of 10 metrics inside the A0 noise band and inconsistent in
  sign, on **both** suites. The two band-clearing results contradict each other
  across suites.
- Model level: A14 matches A2 and fully repairs the A13/A13R regression.
- Mechanism: the castling features carry 0.50 cp of leverage. The network
  ignores them.

**The C6 representation/architecture branch is exhausted.** The question it was
opened to answer is now answered in both directions:

> Adding global state as CNN **input planes** consistently regresses, regardless
> of content (A3, A13, A13P, A13R). Adding the same state **after spatial
> feature extraction** is harmless and, under this dataset, architecture and
> training recipe, is simply not learned (A14).

A14 is **not** recommended as a candidate architecture direction. It costs 1,024
parameters and a production change to the engine's tensor path in exchange for a
feature the model demonstrably does not use.

---

## 14. Recommended next step

Move to the next engineering improvement. Representation and injection-point
changes have now been tested to exhaustion at this dataset size and have
produced no win.

The consistent signal across the whole C6 programme is that **7,734 training
positions is the binding constraint** — it is what makes a cheap global degree
of freedom worth recruiting in A13/A13R, and what leaves four honest features
unlearned in A14. The natural next lever is the dataset, not the tensor.

Two lower-cost items are also worth noting, neither of which is a representation
experiment:

- The frozen production Ridge is fitted to the *original* CNN's output scale and
  caps every retrained arm (§12.3). A matched-fusion refit is a known, bounded
  piece of work.
- The Phase 4 deferred defects (`xfail` in the suite) are real production bugs
  and independent of anything C6 has measured.

---

## Reproducing

```bash
# train
python -m training.train --arm A14 --seed 0     # and 1, 2

# evaluate both suites through the experiment-only shim
python -m training.evaluate_arm --arm A14 --seed 0   # and 1, 2

# analysis
python -m training.a2_analysis --arms A0 A2 A13 A14 --baseline A0 \
    --contrasts A2:A14 A13:A14 --out training/experiments/A14/a14_analysis.json
python -m training.a3_plane_ablation --arm A14
python -m training.a2_saturation_probe --arms A0 A2 A13 A14 --contrast A2:A14 \
    --out training/experiments/A14/saturation_probe.json

# verification
python -m pytest tests/unit/test_training_representation12c4.py -q
```

Artifacts land in `training/experiments/A14/` (gitignored). No production file is
written at any point; `evaluate_arm` asserts the production `models/` directory
is unmodified after every run.
