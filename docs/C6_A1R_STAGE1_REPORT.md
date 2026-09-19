# C6-A1R Stage 1 — Ridge Compatibility Diagnostic (results)

**Stage 1 only. Stage 2 was not run. A2 was not started.** No production code,
model or evaluation artifact was modified; nothing was committed.

Design: [`C6_A1R_RIDGE_DIAGNOSTIC_DESIGN.md`](C6_A1R_RIDGE_DIAGNOSTIC_DESIGN.md)
Repository state: `6f8ff5d`.

---

## Gate decision (up front)

> **The gate is TRIPPED. The refit changes the fusion layer materially.**
>
> Refitting moves the CNN's share of ranking spread from **82.9–89.0%** down to
> **31.0–55.0%** — a shift of **−29.8 to −56.3 percentage points**, against a
> gate reference band (A0 seed spread under production coefficients) of just
> **82.85–84.93%**, about 2 pp wide. The normalised coefficient ratios also change
> substantially, including a **sign flip on `space`** in all six refits.
>
> **Stage 2 is therefore warranted** — but per instruction it was **not run**.

---

## 1. Methodology

Exactly as specified in the design document.

For each of the six trained models (A0 ×3, A1 ×3):

1. **Fit positions:** the **1,933 `dataset_v1` test-split** records (split seed 42,
   identical to every arm). These are **out-of-sample** for every model — the
   point of the exercise, since the production Ridge was fitted partly in-sample.
2. **Features:** `[cnn_norm, material, space, center, mobility]` with
   `cnn_norm = tanh(cnn(position)/200)`, computed on the **position itself**,
   matching notebook cell 31.
3. **Target:** **each arm's own label policy** — A0 `legacy_notebook`,
   A1 `corrected_mate_legacy_perspective` — so the fusion layer is matched to the
   arm it serves.
4. **Fit:** `Ridge(alpha=1.0)`, matching cell 31.
5. **Sanity check:** an inner 80/20 split of the 1,933 (seed **1234**, deliberately
   different from the dataset split seed 42) to report a held-out R². The reported
   coefficients are from the final fit on all 1,933.
6. **Ranking spread:** on **80 `extended` positions** (2,701 candidate moves) — the
   same subset as the design document's §5 table, and a suite with **zero overlap**
   with either dataset split. Features here are computed on **post-move** positions,
   matching `rerank_moves`.

Dataset checksum verified against the committed manifest before fitting
(`5a689e3f37156a05…`). Production Ridge loaded **read-only** as the reference.

---

## 2. Coefficient vectors

| Model | `cnn_norm` | `material` | `space` | `center` | `mobility` | intercept |
|---|---:|---:|---:|---:|---:|---:|
| **production** | **330.9005** | **32.3899** | **0.7919** | **5.1654** | **0.0188** | 15.0772 |
| A0 seed 0 | 197.3612 | 41.8343 | −3.4932 | 14.2285 | 3.0745 | 52.7727 |
| A0 seed 1 | 197.5795 | 40.4798 | −2.7576 | 13.5918 | 2.8425 | 60.0146 |
| A0 seed 2 | 200.9706 | 40.6948 | −2.3270 | 10.3456 | 2.5740 | 39.6432 |
| A1 seed 0 | 245.4067 | 37.5425 | −18.5303 | 14.5807 | 12.0704 | 37.6215 |
| A1 seed 1 | 209.3882 | 37.1902 | −16.9606 | 15.1350 | 12.2764 | 46.5361 |
| A1 seed 2 | 208.1213 | 38.7785 | −17.5500 | 15.8633 | 12.5012 | 42.5777 |

Note the intercepts (37.6–60.0) are far larger than production's 15.08 — and
**`engine.py` discards all of them** (defect C3, still open).

---

## 3. Ratios normalised to `cnn_norm`

Ranking within a position depends only on the **relative** weighting of terms, so
these ratios — not the raw magnitudes — are what can reorder moves.

| Model | `cnn_norm` | `material` | `space` | `center` | `mobility` |
|---|---:|---:|---:|---:|---:|
| **production** | 1.000000 | 0.097884 | **+0.002393** | 0.015610 | 0.000057 |
| A0 seed 0 | 1.000000 | 0.211968 | **−0.017700** | 0.072094 | 0.015578 |
| A0 seed 1 | 1.000000 | 0.204879 | **−0.013957** | 0.068792 | 0.014387 |
| A0 seed 2 | 1.000000 | 0.202491 | **−0.011579** | 0.051478 | 0.012808 |
| A1 seed 0 | 1.000000 | 0.152981 | **−0.075509** | 0.059414 | 0.049185 |
| A1 seed 1 | 1.000000 | 0.177614 | **−0.081001** | 0.072282 | 0.058630 |
| A1 seed 2 | 1.000000 | 0.186327 | **−0.084326** | 0.076222 | 0.060067 |

### Multiple of the production ratio

| Model | `material` | `space` | `center` | `mobility` |
|---|---:|---:|---:|---:|
| A0 seed 0 | 2.2× | **−7.4×** | 4.6× | **274×** |
| A0 seed 1 | 2.1× | **−5.8×** | 4.4× | **253×** |
| A0 seed 2 | 2.1× | **−4.8×** | 3.3× | **225×** |
| A1 seed 0 | 1.6× | **−31.6×** | 3.8× | **864×** |
| A1 seed 1 | 1.8× | **−33.8×** | 4.6× | **1030×** |
| A1 seed 2 | 1.9× | **−35.2×** | 4.9× | **1056×** |

Three findings:

- **`space` flips sign in all six refits.** Production weights it slightly
  positive; every out-of-sample refit weights it negative. Consistently, and much
  more strongly for A1 (−32× to −35×) than A0 (−5× to −7×).
- **`mobility` goes from negligible to material** — production's 0.0188 is
  effectively zero; refits give 2.6–12.5, a 225–1056× ratio change.
- **`material` roughly doubles** relative to the CNN term.

---

## 4. Cosine similarity, scale, ranking share, R²

| Model | cosine → prod | scale (`w₀`/prod) | CNN share (prod coef) | CNN share (refit coef) | shift | R² train | **R² holdout** |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 seed 0 | 0.992017 | 0.5964 | 82.85% | **45.66%** | −37.19 | 0.3831 | **0.3901** |
| A0 seed 1 | 0.992993 | 0.5971 | 84.93% | **51.73%** | −33.20 | 0.3817 | **0.4001** |
| A0 seed 2 | 0.994009 | 0.6073 | 84.79% | **54.95%** | −29.84 | 0.3888 | **0.4156** |
| A1 seed 0 | 0.993505 | 0.7416 | 85.67% | **30.98%** | −54.68 | 0.4000 | **0.3790** |
| A1 seed 1 | 0.990457 | 0.6328 | 88.98% | **34.50%** | −54.48 | 0.3840 | **0.3697** |
| A1 seed 2 | 0.989226 | 0.6290 | 88.05% | **31.74%** | −56.31 | 0.3893 | **0.3479** |

### Cosine similarity is high — and misleading here

All six exceed **0.989**, which naively suggests "near-proportional". **It does
not.** Cosine is dominated by `w₀ ≈ 200–331`, so it is largely blind to the very
coefficients that changed most. Measured directly (and pinned by a test):

| Perturbation of the production vector | cosine |
|---|---:|
| `material` ×10 | 0.7793 (detected) |
| `material` ×2 | 0.9954 (barely) |
| `center` ×10 | 0.9904 (barely) |
| `space` ×10 | 0.9998 (blind) |
| `mobility` ×100 | 1.0000 (blind) |
| uniform ×10 | 1.0000 (blind by construction) |

**Reporting only the cosine would have produced the wrong gate decision.** The
gate correctly rests on the normalised ratios and the ranking share.

### R² sanity check

Held-out R² is **0.348–0.416** across all six, with train R² 0.382–0.400 — close
together, so the refits are not overfitting and are not degenerate. They are also
not strong predictors; the fusion layer explains roughly a third to two-fifths of
label variance out-of-sample.

### Scale factor — channel (b)

Every refit `w₀` is **0.60–0.74×** production. Because the heuristic bonuses
(±0.2–0.65) and the 1-ply lookahead term (±0.5) are added **after** the weighted
sum and are **not** Ridge-scaled, a smaller weighted score makes those unscaled
terms proportionally **~1.4–1.7× more influential**. This is a real second channel,
independent of the ratio changes, and it is invisible to both cosine and the
ranking-share metric.

---

## 5. A0 vs A1 and seed spread

| Quantity | A0 mean | A0 sd | A0 range | A1 mean | A1 sd | A1 range |
|---|---:|---:|---|---:|---:|---|
| **Refit CNN share %** | 50.78 | 4.72 | 45.66 – 54.95 | **32.41** | 1.85 | **30.98 – 34.50** |
| Scale factor | 0.6003 | 0.0061 | 0.596 – 0.607 | 0.6678 | 0.0640 | 0.629 – 0.742 |
| R² holdout | 0.4020 | 0.0129 | 0.390 – 0.416 | 0.3656 | 0.0160 | 0.348 – 0.379 |
| `material` ratio | 0.2064 | 0.0049 | 0.2025 – 0.2120 | 0.1723 | 0.0173 | 0.1530 – 0.1863 |
| `mobility` ratio | 0.0143 | 0.0014 | 0.0128 – 0.0156 | 0.0560 | 0.0059 | 0.0492 – 0.0601 |

**The refit CNN-share ranges do not overlap:** A0 45.66–54.95% vs A1 30.98–34.50%.
Under a matched fusion layer, A1's CNN would carry **substantially less** ranking
weight than A0's — the opposite of what happens under the production Ridge, where
A1's CNN share is *higher* (85.7–89.0% vs 82.8–84.9%).

That reversal is the single most consequential finding of Stage 1.

---

## 6. Comparison against the production Ridge

| Property | Production | Refits |
|---|---|---|
| Fitted on | all 10,000 (≈8,000 **in-sample** for the CNN) | 1,933, **fully out-of-sample** |
| Target | original A0-style labels | **each arm's own policy** |
| CNN ranking share | 82.9–89.0% | **31.0–55.0%** |
| `space` sign | **+** | **−** (all six) |
| `mobility` | ≈0 (0.0188) | 2.57–12.50 |
| `w₀` | 330.90 | 197–245 (0.60–0.74×) |

The direction is consistent with the leakage the design document predicted: the
production Ridge was fitted with partly in-sample CNN outputs, which inflates the
CNN term. Every honest out-of-sample refit reduces it sharply. **The production
Ridge over-weights the CNN relative to an out-of-sample fit — for both arms.**

---

## 7. Gate decision

The design document's gate:

> *If every refit coefficient vector is close to a positive scalar multiple of the
> production vector **and** the implied per-term ranking shares are within the A0
> seed spread, then the refit cannot materially change rankings through channel
> (a) — report and stop.*

| Gate condition | Result |
|---|---|
| Near-proportional to production? | **NO.** `space` flips sign in all six; `mobility` ratio changes 225–1056×; `material` ratio roughly doubles |
| Ranking shares within the A0 seed spread (82.85–84.93%)? | **NO.** Refit shares are 30.98–54.95%, i.e. **28–54 pp below the band** |

**Both conditions fail. The gate is tripped: channel (a) is material.** Channel (b)
is also active (scale 0.60–0.74×).

**Stage 2 is warranted. It was not run**, per instruction.

---

## 8. Limitations

1. **Stage 1 measures the fusion layer, not engine behaviour.** A large change in
   coefficient balance makes a ranking change *possible*; it does not demonstrate
   one. Only Stage 2 can measure selected moves and regret.
2. **The CNN term still leads under the refits** (31–55% of ranking spread, the
   largest single term in every case), so rankings may prove more robust than the
   coefficient shift suggests.
3. **Thin strata, as predicted:** the 1,933-position fit set holds only **80
   Black-to-move** and **35 mate** records — exactly the subgroups A1 alters. The
   A1 refits are poorly constrained where they matter most.
4. **The refit consumed the only clean holdout.** The inner 80/20 split is a sanity
   check, not independent validation.
5. **`OP04`** (1 of 52 `phase0_52` positions) is in the fit set. `extended`, the
   primary suite, has **zero** overlap.
6. **Refits change features *and* target**, so this isolates "arm-matched fusion vs
   production fusion", not output scale alone.
7. **The objective mismatch is unfixed:** the Ridge is fitted to predict absolute
   centipawns across positions, but used to rank moves within a position. Both
   production and refits share this flaw.
8. **The intercept is discarded by the engine** (C3). Refits produce intercepts of
   37.6–60.0 that would be thrown away, so the fitted model is not the model the
   engine would actually apply.
9. **Three seeds per arm.** No significance testing; ranges are descriptive.
10. **The `space` sign flip is unexplained.** Consistent across all six refits and
    stronger for A1, but no mechanism was investigated, and none is asserted.

---

## 9. Recommendation: is Stage 2 warranted?

**Yes.** Three findings justify the ~70–80 minutes:

1. The CNN ranking share collapses by **30–56 pp** under matched fusion — far
   outside the ~2 pp gate band.
2. The **A0 and A1 refit shares do not overlap** (45.7–55.0% vs 31.0–34.5%), and
   the ordering **reverses** versus production. If A1's engine regression is driven
   by fusion mismatch, this is the mechanism, and Stage 2 would show it.
3. Both channels are active: ratio change **and** a 0.60–0.74× scale change that
   alters how the unscaled bonus/lookahead terms compete.

**Stage 2 must keep the 2×2 design** (A0-refit and A1-refit, per-seed Ridge). The
A0-refit arm is now demonstrably essential: A0's refit also drops its CNN share by
~30 pp, so without it any A1 change would be unattributable.

**Stage 2 was not run, per instruction.**

---

## 10. Recommendation: should A2 proceed?

**Recommendation: run Stage 2 before A2.**

Stage 1 was intended to bound the confound cheaply. It did the opposite — it
showed the fusion layer is materially mismatched for **both** arms, and more so
for A1. Specifically:

- **A1's engine-level regression remains unattributable**, and Stage 1 has raised
  rather than lowered the probability that fusion mismatch contributes. Running A2
  now would add a third result interpreted under a fusion layer we have just
  measured to be substantially wrong for the arm it serves.
- **A2 changes target semantics more than A1 did** (perspective normalisation flips
  sign on Black-to-move labels), so it inherits this confound at least as strongly.
- Stage 2 is **~70–80 minutes with no retraining** — cheap relative to a full arm.

**The counter-argument, stated fairly:** A0/A1/A2 under a *fixed* production Ridge
remain a self-consistent protocol, and all three are comparable to each other
whatever the fusion layer's absolute merit. Deferring Stage 2 until after A2 and
then running one refit diagnostic across all three arms is defensible and cheaper
in total.

**On balance I recommend Stage 2 first**, because A1's result is the one currently
driving the programme's conclusions and it is the one most at risk of being
misread. But this is a schedule judgement and it is yours to make — the evidence
above supports either path, and neither is scientifically unsound.

A separate observation worth noting regardless of the path chosen: **the production
Ridge appears to over-weight the CNN** because it was fitted partly in-sample. That
is a property of the shipped engine, not just of these experiments, and it is
independent of C6.

---

## Reproducing

```bash
python -m training.refit_ridge
```

Output: `training/experiments/A1R/stage1_results.json` (gitignored).
Runtime ≈ 6 minutes. No engine evaluation, no training, no Stockfish.
