# C6-A2 — Label Perspective Normalisation

**Status:** complete, 3 seeds, both evaluation suites.
**Scope:** one experimental variable relative to A1. No production code changed.
**Fusion:** the frozen **production** Ridge for every arm. The A1R matched Ridge
is deliberately *not* used here.
**Statistical standing:** descriptive. Three seeds, no significance testing. A
change is only called real when it is directionally consistent across all three
paired seeds *and* exceeds the A0 seed-noise band.

---

## 1. Objective

Test whether converting the training labels from the legacy **side-to-move-relative**
perspective to a **White-positive** evaluation perspective changes model quality
or engine behaviour, relative to A1.

This closes the loop on a prediction the C6 training-pipeline audit made but never
measured. Audit §D.3 flagged that side-to-move-relative labels act as "~4.5% label
noise under the dominant White-positive convention", and §L established that the
engine's `rerank_moves` **requires** a White-positive CNN at inference. A0 and A1
both trained against the other convention. A2 is the first arm to align the
training frame with the inference frame.

---

## 2. Experimental delta

Exactly one variable moves relative to A1: **the frame the label is expressed in.**

| | A0 | A1 | **A2** |
|---|---|---|---|
| label policy | `legacy_notebook` | `corrected_mate_legacy_perspective` | **`corrected_mate_white_perspective`** |
| mate representation | raw distance as if cp | repaired ±2000 scale | repaired ±2000 scale *(same as A1)* |
| cp clip | ±1500 | ±1500 | ±1500 *(same as A1)* |
| **perspective** | side-to-move | side-to-move | **White-positive (CHANGED)** |
| representation | 12 planes | 12 planes | 12 planes |

A2's policy is implemented in `training/dataset.py` as an explicit *derivation*
from `raw_stockfish_value`, exactly as A1's is, so the two arms are two
derivations of the same raw inputs and the contrast is auditable in one function.
It is not a read of the stored `label` column — though a test pins that the two
agree on all 9,667 records, which cross-validates the derivation against the
C6-Prep pipeline.

```python
# A1 — hardcodes the mover as White, pinning the label to the mover's frame
mate: labels.mate_to_white_positive(d, side_to_move_is_white=True)
cp  : labels.clip_cp(raw, 1500)

# A2 — passes the record's ACTUAL side to move
mate: labels.mate_to_white_positive(d, side_to_move_is_white=stm_is_white)
cp  : labels.to_white_positive(labels.clip_cp(raw, 1500), side_to_move_is_white=stm_is_white)
```

---

## 3. Fixed configuration

Held identical to A0 and A1, and verified by test rather than by assertion:

| | value |
|---|---|
| dataset | `dataset_v1.jsonl`, 9,667 deduplicated records |
| split | sorted by FEN, seeded permutation, split seed 42 → 7,734 train / 1,933 test |
| representation | 12 planes, 8×8×12 float32, `encodes_side_to_move=False` |
| architecture | identical CNN, **2,360,129 parameters** in all nine runs |
| loss / optimiser | Huber, Adam, LR 1e-3, batch 64 |
| epochs | max 100, same callbacks (early stopping, best-weight restore) |
| seeds | 0, 1, 2 |
| fusion | frozen **production** Ridge, `coef = [330.9005, 32.3899, 0.7919, 5.1654, 0.0188]` |
| evaluator | `evaluation/evaluate.py`, unmodified |
| Stockfish | depth 8, Threads=1, Hash=16, `Clear Hash` per position |
| suites | `extended` (n=160), `phase0_52` (n=52) |

`test_a2_differs_from_a1_in_label_policy_only` asserts that the A1 and A2 arm
specs differ in the `label_policy` key and nothing else, and
`test_a2_uses_the_same_hyperparameters_as_a0_and_a1` asserts no arm can shadow a
hyperparameter at all.

---

## 4. Pre-training label audit

Run and gated **before** any training, via `python -m training.a2_label_audit`
(artifact: `training/experiments/A2/label_audit.json`). All eight gates passed.

| | value |
|---|---|
| records | 9,667 |
| White to move | 9,260 (95.79%) |
| Black to move | **407 (4.21%)** — 306 cp + 101 mate |
| labels changed A1 → A2 | **407 (4.21%)** — 327 train / 80 test |
| changed, by side to move | `{black: 407}` — **no White record moved** |
| every change an exact sign flip | yes |
| magnitudes identical everywhere | yes |
| derived A2 == stored `label` column | yes, 0 / 9,667 mismatches |

**Reconciliation with the audit.** The C6 audit reported 446 Black-to-move
positions out of 10,000. `dataset_v1` has 407 out of 9,667 because it is
deduplicated by FEN. Both figures describe the same defect: 4.46% pre-dedup,
4.21% post-dedup.

**The changed labels are not marginal.** Mean |Δ| over the 407 changed records is
**1,554 cp**. The Black-record mean moves from −671 to +671: these positions are
overwhelmingly ones where White is winning, so A1 labelled them strongly negative
and A2 labels them strongly positive. A2 changes few labels, but changes them
about as much as a label can be changed.

---

## 5. Model results — A2 seeds 0 / 1 / 2

| run | epochs | best epoch | best val loss | MAE cp | RMSE cp | **Pearson r** | secs |
|---|---|---|---|---|---|---|---|
| A0 s0 | 17 | 7 | 156.53 | 165.64 | 245.36 | 0.5090 | 124 |
| A0 s1 | 18 | 8 | 160.66 | 167.53 | 244.19 | 0.5204 | 136 |
| A0 s2 | 22 | 12 | 154.01 | 162.32 | 240.82 | 0.5304 | 161 |
| A1 s0 | 23 | 13 | 180.57 | 185.87 | 312.90 | 0.6055 | 160 |
| A1 s1 | 35 | 25 | 180.84 | 190.26 | 313.52 | 0.6040 | 227 |
| A1 s2 | 41 | 31 | 182.51 | 190.82 | 317.18 | 0.5919 | 297 |
| **A2 s0** | 25 | 15 | 155.31 | 160.70 | 257.92 | **0.7450** | 185 |
| **A2 s1** | 58 | 48 | 157.32 | 162.98 | 262.09 | **0.7360** | 401 |
| **A2 s2** | 46 | 36 | 154.84 | 164.96 | 264.82 | **0.7332** | 329 |

**Read Pearson, not MAE.** The three arms are fitted to different label
distributions (label std 286 / 402 / 395), so MAE and RMSE are not comparable
across arms and neither is the Huber validation loss. Pearson r is scale-invariant
and is the one directly comparable offline number.

On that metric the arms separate completely, with **no overlap between any two
arms across three seeds**:

```
A0  0.5090  0.5204  0.5304
A1  0.5919  0.6040  0.6055
A2  0.7332  0.7360  0.7450     <- +0.13 over A1, +0.22 over A0
```

This is the largest offline effect measured anywhere in the C6 programme.

---

## 6. Engine results — A2 seeds 0 / 1 / 2

Both suites, production Ridge, unmodified Phase 3 evaluator.

**`extended` (n=160)**

| metric | A2 s0 | A2 s1 | A2 s2 |
|---|---|---|---|
| legality % | 100 | 100 | 100 |
| top-1 agreement % | 19.38 | 19.38 | 14.38 |
| top-3 containment % | 33.12 | 34.38 | 29.38 |
| mean regret cp | 133.7 | 141.6 | 152.3 |
| median regret cp | 29 | 30 | 54 |
| p95 regret cp | 530 | 539 | 531 |
| blunder rate >300cp | 0.2026 | 0.2353 | 0.2484 |

**`phase0_52` (n=52)**

| metric | A2 s0 | A2 s1 | A2 s2 |
|---|---|---|---|
| legality % | 100 | 100 | 100 |
| top-1 agreement % | 23.08 | 19.23 | 15.38 |
| top-3 containment % | 30.77 | 28.85 | 26.92 |
| mean regret cp | 104.3 | 127.7 | 109.4 |
| median regret cp | 15.5 | 34.5 | 41.5 |
| p95 regret cp | 471 | 524 | 516 |
| blunder rate >300cp | 0.1522 | 0.1957 | 0.1522 |

Legality is 100% in all six runs, as in every previous arm.

---

## 7. A1 → A2 paired comparison, against the A0 noise band

The A0 noise band is the min–max range of the control arm over its three seeds —
what the metric does from **seed alone**, with no experimental change at all.

**`extended` (n=160)**

| metric | s0 | s1 | s2 | mean Δ | A0 noise range | verdict |
|---|---|---|---|---|---|---|
| mean regret cp | −17.19 | −5.23 | −17.73 | **−13.38** | 19.70 | all 3 better, **within noise** |
| p95 regret cp | −50 | −30 | −106 | −62.00 | 64 | all 3 better, within noise |
| blunder rate | −0.0540 | −0.0015 | −0.0326 | −0.0294 | 0.0395 | all 3 better, within noise |
| median regret cp | −9.5 | −8 | +4 | −4.50 | 10 | inconsistent |
| top-1 agreement % | +1.88 | +2.50 | −5.00 | −0.21 | 3.74 | inconsistent |
| top-3 containment % | −3.76 | +1.88 | −1.24 | −1.04 | 0.63 | inconsistent |

**`phase0_52` (n=52)**

| metric | s0 | s1 | s2 | mean Δ | A0 noise range | verdict |
|---|---|---|---|---|---|---|
| mean regret cp | −13.98 | +3.81 | +0.37 | −3.27 | 31.65 | inconsistent |
| spearman mean | +0.06 | +0.09 | +0.07 | **+0.073** | 0.06 | all 3 better, **EXCEEDS noise** |
| median regret cp | −15.5 | −5.5 | +0.5 | −6.83 | 7.5 | inconsistent |
| top-3 containment % | −1.92 | +3.85 | −5.77 | −1.28 | 3.85 | inconsistent |

**Verdict on aggregate engine metrics.** A2 recovers A1's regression — mean regret,
p95 and blunder rate all improve in all three paired seeds on `extended` — but
every one of those improvements sits **inside the A0 noise band**. Only the
`phase0_52` Spearman improvement (+0.073, all three seeds, band 0.06) clears the
pre-registered bar, and a single metric on the smaller suite is weak evidence.

**By the pre-registered rule, A2 does not demonstrate an aggregate engine-level
improvement over A1.** The aggregate is, however, the wrong place to look — §9.

---

## 8. A0 → A2 paired comparison

| suite | metric | mean Δ vs A0 | A0 noise range | verdict |
|---|---|---|---|---|
| extended | mean regret cp | +3.24 | 19.70 | within noise |
| extended | top-3 containment % | −1.25 | 0.63 | exceeds noise, inconsistent |
| phase0_52 | mean regret cp | +9.50 | 31.65 | within noise |
| phase0_52 | top-3 containment % | −11.54 | 3.85 | all 3 worse, **exceeds noise** |

A2 returns aggregate regret to the control's level (+3.24 cp on `extended`, well
inside noise) after A1 had pushed it up. It does **not** beat the control in
aggregate, and it inherits A1's top-3 containment regression on `phase0_52`,
which A2 does not repair.

---

## 9. White vs Black — the central finding

The label change applies only to Black-to-move positions, so the aggregate
necessarily dilutes it. Splitting by side to move is the pre-specified place to
look, and it is where the experiment resolves.

**Mean regret cp, averaged over the three seeds:**

| suite | side | n | A0 | A1 | **A2** | A1 → A2 | A0 → A2 |
|---|---|---|---|---|---|---|---|
| extended | **black** | 75 | 175.54 | 195.70 | **139.07** | **−56.63** | −36.46 |
| extended | white | 85 | 106.82 | 120.90 | 145.60 | +24.70 | +38.78 |
| phase0_52 | **black** | 22 | 91.79 | 123.73 | **63.18** | **−60.55** | −28.61 |
| phase0_52 | white | 30 | 113.33 | 112.59 | 149.45 | +36.86 | +36.11 |

**All four A1 → A2 deltas are consistent across all three paired seeds**, and the
pattern replicates across two independent suites:

- **A2 is the best of the three arms on Black-to-move positions, on both suites.**
  It beats A1 by ~57–61 cp and the control A0 by ~29–36 cp.
- **A2 is the worst of the three arms on White-to-move positions, on both suites**,
  by ~25–37 cp against A1 and ~36–39 cp against A0.

The improvement lands precisely where the label change applies, in the direction
predicted before training, replicated on two suites and three seeds. The
magnitudes (57–61 cp) are around three times the `extended` A0 noise range of
19.7 cp.

**A2 does not make the engine better overall. It redistributes accuracy from
White-to-move positions to Black-to-move positions**, and because the suites are
roughly half Black-to-move while the training set is only 4.2% Black-to-move, the
two effects very nearly cancel in the aggregate.

The White-side regression is real, consistent and **unexplained**. §13 records the
hypothesis that was tested for it and failed.

---

## 10. Game-phase / category breakdown

Mean regret cp, A1 → A2, averaged over three seeds:

| category | n (ext) | A1 → A2 (extended) | n (p52) | A1 → A2 (phase0_52) |
|---|---|---|---|---|
| defensive | 5 | −58.61 | 4 | +13.50 |
| endgame | 21 | +28.84 | 12 | −1.86 |
| middlegame | 64 | −22.65 | 12 | −29.08 |
| opening | 60 | −15.17 | 18 | +13.53 |
| tactical | 10 | −0.67 | 6 | −22.00 |

**Middlegame is the only category that improves on both suites** (−22.65, −29.08).
Every other category flips sign between suites, and the cells with n=4–10 are far
too small to read — a single position changes `tactical` on `extended` by tens of
centipawns.

Note for continuity with A1R: A1's degradation was concentrated in **endgames**
(247–309 cp under matched fusion). Under production fusion here, A2's endgame
regret rises on `extended` (+28.84) and is flat on `phase0_52` (−1.86). A2 does
not repair the endgame concentration, and this report does not explain it.

---

## 11. Mate behaviour

Mate status counts over `extended` (n=160):

| run | none | missed forced mate | move allows forced mate | both |
|---|---|---|---|---|
| A0 s0/s1/s2 | 152 / 153 / 152 | 4 / 4 / 3 | 4 / 3 / 4 | 0 / 0 / 1 |
| A1 s0/s1/s2 | 152 / 152 / 153 | 3 / 3 / 3 | 4 / 4 / 3 | 1 / 1 / 1 |
| **A2 s0/s1/s2** | **153 / 153 / 153** | **4 / 4 / 4** | **3 / 3 / 3** | **0 / 0 / 0** |

`phase0_52` shows the same shape: A2 is `{none: 46, missed: 4, allows: 2}` in all
three seeds.

Two observations:

1. **A2's mate profile is identical across all three seeds on both suites.** A0
   and A1 both vary seed to seed. A2 is the only arm whose mate behaviour is
   seed-stable.
2. The trade is one fewer *allowed* forced mate against one more *missed* forced
   mate, relative to A1. Allowing a mate is the more damaging error, so this is
   mildly favourable, but at counts of 3–4 out of 160 it is not a result — a
   single position moves the number.

The mate-label magnitudes are identical in A1 and A2 (§4), so any mate difference
here is attributable to the frame, not the scale.

---

## 12. Mechanism — the representation gap, measured

**Why A2 should work, stated before the numbers.** `representation_summary()`
reports `encodes_side_to_move = False`: the 12-plane tensor carries piece
placement only. Under A0 and A1 the target is side-to-move-relative, so for the
407 Black-to-move positions the *sign* of the label depends on a fact the input
does not contain. The network is asked to emit two different values for inputs it
cannot distinguish. That is irreducible label noise, not a learnable pattern. A2
makes the target a function of the position alone.

This is measurable **within each arm**, which avoids comparing arms fitted to
different label scales. `python -m training.a2_perspective_probe` splits each
arm's own held-out test error by side to move:

| arm | White MAE | Black MAE | **gap** | Black signed error |
|---|---|---|---|---|
| A0 s0/s1/s2 | 157.3 / 159.0 / 153.5 | 359.8 / 365.0 / 366.5 | **+202.6 / +206.0 / +212.9** | +170 / +172 / +186 |
| A1 s0/s1/s2 | 167.6 / 173.4 / 174.7 | 610.3 / 581.3 / 563.8 | **+442.7 / +407.9 / +389.1** | +393 / +328 / +325 |
| **A2 s0/s1/s2** | 155.7 / 157.6 / 159.2 | 276.5 / 286.8 / 297.7 | **+120.8 / +129.1 / +138.5** | +41 / −70 / −97 |

Mean gap: **A0 +207, A1 +413, A2 +129.** No overlap between arms. A2 more than
halves A1's gap, and the Black-row signed error collapses from a systematic +325
to +393 bias down to −97 to +41 — i.e. A0 and A1 are *directionally* wrong on
Black rows in the way an unlearnable frame predicts, and A2 is not.

**Why A1 was worse than A0 — the open question from the A1 report, answered.**
Splitting the Black test rows by eval type:

| arm | Black cp MAE (n=61) | Black mate MAE (n=19) | mean \|mate label\| |
|---|---|---|---|
| A0 | 445 / 453 / 449 | 86 / 83 / 103 | **0** |
| A1 | 459 / 448 / 449 | **1096 / 1011 / 933** | **2000** |
| A2 | 260 / 224 / 218 | 331 / 487 / 552 | 2000 |

A1's mate repair moved 19 test rows from a target of **0** — trivially
predictable — to a target of magnitude **2000** while leaving the frame
unlearnable. The repair itself became the damage. A2 keeps the ±2000 magnitude
but puts it in a frame the network can see, and the error drops by roughly half.

This is consistent with, and independent of, the A1R finding: A1R Stage 2 showed
fusion mismatch does *not* explain A1's regression. The cause was in the labels
all along, and it was the interaction of A1's mate repair with the unfixed frame.

It also empirically confirms audit §D.3's prediction, which had never been
measured.

---

## 13. An exploratory hypothesis that did NOT hold

**Not pre-registered.** This probe attempts to explain the White-to-move
regression in §9. Its result is reported because it failed, not because it
supports anything.

**Hypothesis.** The fusion squashes the CNN through `330.9 * tanh(cnn / 200)`.
A2's labels put real mass at ±2000, so if A2's CNN emits larger magnitudes, tanh
flattens, the CNN term stops separating candidate moves within a position, and
ranking falls back on the heuristics.

**Result** (`python -m training.a2_saturation_probe`, 5,639 candidate child
positions across both suites):

| arm | saturated (\|tanh\| > 0.95) | within-position tanh spread, White to move |
|---|---|---|
| A0 | 6.2% | 0.6458 |
| A1 | 6.1% | 0.7905 |
| **A2** | **3.7%** | **0.9547** |

**Not supported, and the reverse is true.** A2 saturates *less* than A1 and
separates White-to-move candidates *more*. A2's CNN is the least saturated and
most discriminative of the three.

**What this suggests, untested.** A larger within-position spread means the
`330.9 * tanh(...)` term now dominates the heuristic terms more strongly than it
did for A0 or A1. A1R Stage 1 independently found the production Ridge already
over-weights the CNN — its ranking share was 82.9–89.0%. A2 making the CNN a
louder voice inside an already CNN-dominated, mismatched fusion would help where
the CNN is now genuinely more correct (Black to move) and hurt where the
heuristics were carrying the ranking (White to move). That is consistent with
every number in §9, but it is a hypothesis: the direct test is a matched-Ridge
run, which is explicitly out of scope for A2.

---

## 14. Reproducibility and production safety

**Reproducibility.** `keras.utils.set_random_seed(seed)` plus
`tf.config.experimental.enable_op_determinism()`. The split is seeded separately
(42) from the training seed, so the test set is byte-identical across all nine
runs. Per-seed weight hashes (`model_weights_sha256`, the bit-stable artifact —
the `.keras` container embeds a save timestamp and is not byte-stable):

```
A2 seed 0  d7549acb5733d863…
A2 seed 1  4d9826fd18bdd345…
A2 seed 2  bbe635961c5d4ba8…
```

**Production safety.** Nothing in `engine.py`, `app.py`, `config.py`,
`evaluation/`, `models/` or the baseline/regression fixtures was modified.
Experimental models reach the evaluator only through the pre-existing
`CHESS_BOT_MODELS_DIR` injection point, in a subprocess, against a temporary
staging directory holding a **copy** of the production Ridge.
`training/evaluate_arm.py` hashes the production models directory before and
after each run; all three A2 runs logged `verified: production models/ untouched`.

All A2 artifacts are under `training/experiments/A2/` and `docs/`.

**Tests.** Full suite: **449 passed, 10 xfailed, 0 failed** (baseline before A2:
435 passed, 10 xfailed). The 14 new tests pin the six required A2 properties,
including `test_a2_matches_the_stored_c6prep_label_on_the_real_dataset`, which
checks the derivation against all 9,667 stored labels.

The xfail `test_side_to_move_should_be_representable` — "board_to_planes has no
side-to-move plane, so identical placements with opposite sides to move are
indistinguishable" — remains xfail. A2 works *around* that defect by removing the
labels' dependence on the missing input; it does not fix it.

---

## 15. Limitations and factual conclusion

**Limitations**

1. **Three seeds, descriptive only.** No significance testing. Everything here is
   a direction and a magnitude against a measured noise band.
2. **The suites are not the training distribution.** `extended` is 47% Black-to-move
   and `phase0_52` is 42%; the training set is 4.2%. The engine-level Black/White
   split is therefore far more sensitive to this change than the aggregate, and
   an aggregate number on a suite with a different side balance would differ.
3. **Small cells.** Category breakdowns at n=4–21 and mate counts at 3–4 out of
   160 are not interpretable on their own.
4. **Frozen production Ridge.** By design, so the fusion is constant across arms.
   But A1R Stage 1 established that this Ridge is mismatched to every retrained
   CNN, so no arm here should be read as the best this architecture can do.
5. **The White-to-move regression is unexplained.** The one hypothesis tested
   (§13) failed, and the alternative it suggests is untested.
6. **Offline gain ≠ engine gain.** The Pearson improvement is large and clean; it
   did not convert into an aggregate engine improvement. That gap is itself a
   finding about the fusion, not about the labels.

**Factual conclusion**

1. A2 changed exactly 407 of 9,667 labels (4.21%), all Black-to-move, every one an
   exact sign flip with magnitude preserved. All eight pre-training gates passed.
2. **Offline, A2 is decisively the best arm.** Pearson r 0.733–0.745 versus A1's
   0.592–0.606 and A0's 0.509–0.530, with no overlap across three seeds.
3. **On Black-to-move positions A2 is the best arm on both suites**, beating A1 by
   56.6 cp (`extended`) and 60.6 cp (`phase0_52`) and beating the control A0 by
   36.5 and 28.6 cp. All consistent across three paired seeds.
4. **On White-to-move positions A2 is the worst arm on both suites**, by 24.7 and
   36.9 cp against A1. Also consistent across three paired seeds. Unexplained.
5. **In aggregate the two effects cancel.** A1 → A2 mean regret improves in all
   three seeds on `extended` (−13.38 cp) but stays within the A0 noise band of
   19.7 cp. Against the control, A2 is +3.24 cp — statistically indistinguishable
   from A0. **A2 does not clear the pre-registered bar for an aggregate
   engine-level improvement.**
6. The mechanism is established and quantified: the 12-plane input does not encode
   side to move, so side-to-move-relative labels are partly unlearnable. The
   within-arm Black-minus-White test-error gap is +207 cp (A0), +413 cp (A1) and
   +129 cp (A2), with no overlap between arms.
7. **A1's regression is now explained.** A1's mate repair moved 19 Black test rows
   from a target of 0 to a target of ±2000 while leaving the frame unlearnable,
   raising their MAE from ~91 to ~1013. A2 keeps the magnitude, fixes the frame,
   and cuts that error roughly in half. This is independent of A1R Stage 2, which
   had already ruled out fusion mismatch as the cause.
8. Legality remained 100% in all six A2 runs. No production file was modified, and
   the production models directory was verified unchanged after every run.

**In one line:** the perspective fix is clearly correct and produces by far the
largest offline gain in the programme, and it demonstrably improves play exactly
where it applies — but under the frozen, CNN-dominated production fusion that gain
is offset by an unexplained White-side regression, so the engine as a whole is
statistically unchanged.

---

## Reproducing

```bash
# 1. pre-training label audit (gated; must print ALL GATES PASS)
python -m training.a2_label_audit

# 2. train
python -m training.train --arm A2 --seed 0     # and 1, 2

# 3. evaluate against both suites with the production Ridge
python -m training.evaluate_arm --arm A2 --seed 0   # and 1, 2

# 4. analysis
python -m training.a2_analysis             # paired deltas vs the A0 noise band
python -m training.a2_perspective_probe    # mechanism (§12)
python -m training.a2_saturation_probe     # failed hypothesis (§13)
```

Artifacts: `training/experiments/A2/label_audit.json`, `a2_analysis.json`,
`perspective_probe.json`, `saturation_probe.json`, and per-seed
`metadata.json` / `test_predictions.json` / `evaluation/*.json`.
