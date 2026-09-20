# C6-A13 — Castling Rights Only (16 planes)

**Status:** complete, 3 seeds, both evaluation suites.
**Scope:** one representation variable relative to A2. No production code changed.
**Fusion:** the frozen **production** Ridge. No matched-Ridge run (A2R settled the
decision path; nothing here made one necessary).
**Statistical standing:** descriptive. Three seeds, no significance testing. A
change counts only when it is directionally consistent across all three paired
seeds *and* exceeds the A0 seed-noise band.

---

## Result up front

> **A13 degrades in the same way A3 did, despite containing neither of the two
> planes the A3 report blamed.**
>
> Castling rights alone reproduce essentially the whole A3 regression — offline
> (Pearson 0.6241 mean vs A3's 0.6435 and A2's 0.7381) and at the engine level
> (mean regret **+32.6 cp** on `extended`, all three paired seeds, against an A0
> noise range of 19.7).
>
> **This revises the A3 report's stated mechanism.** A3 §12 attributed the
> damage principally to the side-to-move plane and to tanh saturation. A13 has
> no side-to-move plane and the *lowest* saturation of any arm (3.1%), yet
> regresses anyway. Neither is necessary.

Per the pre-registered interpretation rule, this is the third case:
**"A13 materially worse than A2 consistently → castling rights alone may be
contributing to the degradation."** That is a diagnostic conclusion about this
dataset, architecture and recipe — not a claim that castling rights are useless,
and not a production recommendation.

---

## 1. Exact controlled variable

**The board encoding, and only that.** 12 planes → 16 planes.

| | A2 | **A13** |
|---|---|---|
| **representation** | planes12 `(8,8,12)` | **planes16 `(8,8,16)`** |
| parameters | 2,360,129 | **2,362,433 (+2,304)** |
| everything else | — | **identical** |

The parameter delta is exactly `3·3·4·64 = 2,304` — only the first `Conv2D`
kernel changes, `3×3×12×64 → 3×3×16×64`. A test pins the literal 2,362,433.

### The four appended planes

| plane | meaning | values | spatial? |
|---:|---|---|---|
| 0–11 | the existing 12-plane piece encoding, **unchanged** | 0/1 | yes |
| 12 | White kingside castling | constant 0/1 | constant |
| 13 | White queenside castling | constant 0/1 | constant |
| 14 | Black kingside castling | constant 0/1 | constant |
| 15 | Black queenside castling | constant 0/1 | constant |

**Deliberately absent:** side-to-move and en-passant, the two A3 additions that
the A3 ablation found harmful and inert respectively.

16 channels, not 13 — "A13" is the arm name, not a plane count. Castling is four
independent binary rights.

Channel indices differ from A3's encoding (where castling sits at 13–16 because
side-to-move takes 12). A test asserts the four castling *values* agree between
the two encodings position by position, so the arms remain comparable.

### Dataset coverage of the added fields

| field | coverage in `dataset_v1` (n=9,667) |
|---|---|
| White kingside / queenside | 4,980 (51.5%) / 4,900 (50.7%) |
| Black kingside / queenside | 5,289 (54.7%) / 5,208 (53.9%) |
| any castling right | 7,180 (74.3%) |

Unlike en-passant (0.27%), castling is genuinely well-populated. This is not a
dead-channel experiment.

---

## 2. Unchanged variables, with hashes

| | value |
|---|---|
| dataset | `dataset_v1.jsonl`, 9,667 deduplicated records |
| **dataset sha256** | `5a689e3f37156a0540598cdebd0c7f42cdbb09976b5c9af61c013ebdcf3a430d` |
| split | 7,734 train / 1,933 test, **split seed 42** |
| **train index sha256** | `9c1582d429845a1d7be4b8823301bdd01782c916819987aa3552fb2a0a2a1ff7` |
| **test index sha256** | `7d4c1ecdb4fbb0e701c449ec5e38f0670de298a5b735afaf2910dd8eddf61d0e` |
| label policy | `corrected_mate_white_perspective` |
| **label vector sha256** | `424cd8a8b661a81364caab7e2fc34351a3748406e8cf57a174a695d26bd61493` |
| loss / optimiser | Huber, Adam, LR 1e-3, batch 64 |
| epochs | max 100, same callbacks |
| seeds | 0, 1, 2 |
| fusion | frozen production Ridge `[330.9005, 32.3899, 0.7919, 5.1654, 0.0188]` |
| evaluator | `evaluation/evaluate.py`, unmodified |
| Stockfish | 17.1, depth 8, Threads=1, Hash=16MB, `Clear Hash` per position |
| suites | `extended` (n=160), `phase0_52` (n=52) |

**A2 and A13 produce byte-identical label and split hashes.** No hyperparameter
was tuned.

---

## 3. Required implementation checks

All nine, each covered by a test in
`tests/unit/test_training_representation16.py` (32 tests):

| # | check | status |
|---|---|---|
| 1 | shape is `(8,8,16)` | pass |
| 2 | channels 0–11 match the 12-plane encoder | pass — on all 212 suite positions, and against `engine.board_to_planes` |
| 3 | channels 12–15 are the four castling rights | pass — parametrised per right, plus spatial-constancy |
| 4 | **no side-to-move channel** | pass — boards differing only in side to move encode identically (and differ under A3) |
| 5 | **no en-passant channel** | pass — boards differing only in ep target encode identically (and differ under A3) |
| 6 | parameter count is exactly 2,362,433 | pass |
| 7 | labels identical to A2 | pass — on all 9,667 records |
| 8 | split identical to A2 | pass — indices and seed 42 |
| 9 | production files untouched | pass — §10 |

---

## 4. Evaluation shim (unchanged mechanism, generalised)

`engine.board_to_planes` hardcodes `np.zeros((8, 8, 12))`, so a 16-channel model
fails under the unmodified engine exactly as A3's 18-channel one does.

The A3 shim was **generalised, not duplicated**:
`training/evaluate_planes_runner.py` now takes `--representation` and rebinds
`engine.board_to_planes` in-process; `training/evaluate18_runner.py` remains as a
thin delegating wrapper so the command documented in
[C6_A3_REPORT.md](C6_A3_REPORT.md) keeps working. `training/evaluate_arm.py`
dispatches on the encoder's **plane count** (`N_PLANES == 12`) rather than a
hardcoded name, so future representations need no change there.

No file on disk is modified; the rebinding dies with the process. The evaluation
log shows the confirmation six times:
`patched engine.board_to_planes -> training.representation16.board_to_planes (8, 8, 16) (in-process only; no file modified)`.

**Deployability, restated:** a 16-plane model is no more deployable than an
18-plane one. Both require editing `engine.board_to_planes`.

---

## 5. Per-seed model results

| run | rep | params | epochs | best epoch | val Huber | test Huber | MAE cp | RMSE cp | **Pearson r** | secs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 s0 | planes12 | 2,360,129 | 25 | 15 | 155.31 | 160.20 | 160.70 | 257.92 | 0.7450 | 185 |
| A2 s1 | planes12 | 2,360,129 | 58 | 48 | 157.32 | 162.48 | 162.98 | 262.09 | 0.7360 | 401 |
| A2 s2 | planes12 | 2,360,129 | 46 | 36 | 154.84 | 164.46 | 164.96 | 264.82 | 0.7332 | 329 |
| A3 s0 | planes18 | 2,363,585 | 20 | 10 | 180.25 | 185.98 | 186.48 | 292.41 | 0.6561 | 130 |
| A3 s1 | planes18 | 2,363,585 | 60 | 50 | 176.88 | 186.03 | 186.53 | 297.13 | 0.6433 | 381 |
| A3 s2 | planes18 | 2,363,585 | 43 | 33 | 180.27 | 187.98 | 188.48 | 299.85 | 0.6310 | 250 |
| **A13 s0** | planes16 | 2,362,433 | 22 | 12 | 188.22 | 194.03 | 194.53 | 307.25 | **0.6054** | 143 |
| **A13 s1** | planes16 | 2,362,433 | 44 | 34 | 184.29 | 186.86 | 187.36 | 297.33 | **0.6389** | 383 |
| **A13 s2** | planes16 | 2,362,433 | 46 | 36 | 180.86 | 189.85 | 190.35 | 300.32 | **0.6281** | 269 |

Weight hashes (bit-stable artifact; the `.keras` container is not byte-stable):

```
A13 seed 0  be615367aebdb8e83ebaa669f1580d67…
A13 seed 1  f2093755e7e5b02d4a6efa21c26cabce…
A13 seed 2  cff2718ca31dc4a428d983dec3ae5299…
```

Representation identifier recorded in every run's metadata: `planes16`.

### Aggregate

```
Pearson r   A2   [0.7332 .. 0.7450]  mean 0.7381
            A3   [0.6310 .. 0.6561]  mean 0.6435
            A13  [0.6054 .. 0.6389]  mean 0.6241
```

**A13 does not overlap A2. It does overlap A3.** Because all three arms share
identical labels, MAE/RMSE/Huber are directly comparable too, and every one is
worse for A13 than for A2.

A13's deficit vs A2 is **−0.114 Pearson**; A3's is −0.095. **Castling alone
accounts for the whole A3 offline regression, and slightly exceeds it.**

---

## 6. Engine results

**`extended` (n=160)**

| metric | A13 s0 | A13 s1 | A13 s2 |
|---|---|---|---|
| legality % | 100 | 100 | 100 |
| top-1 agreement % | 15.00 | 14.38 | 11.88 |
| top-3 containment % | 36.25 | 30.62 | 31.25 |
| mean regret cp | 147.0 | 199.6 | 178.9 |
| median regret cp | 30 | 68 | 39 |
| p95 regret cp | 594 | 632 | 663 |
| max regret cp | 758 | 716 | 758 |
| blunder rate >300cp | 0.2387 | 0.3355 | 0.2810 |
| regret coverage % | 96.88 | 96.88 | 95.62 |
| spearman mean | 0.20 | 0.16 | 0.24 |

**`phase0_52` (n=52)**

| metric | A13 s0 | A13 s1 | A13 s2 |
|---|---|---|---|
| legality % | 100 | 100 | 100 |
| top-1 agreement % | 11.54 | 19.23 | 19.23 |
| top-3 containment % | 26.92 | 28.85 | 32.69 |
| mean regret cp | 137.4 | 154.4 | 142.3 |
| median regret cp | 53.5 | 29.5 | 45 |
| p95 regret cp | 524 | 614 | 524 |
| max regret cp | 614 | 626 | 625 |
| blunder rate >300cp | 0.2174 | 0.2391 | 0.2222 |
| regret coverage % | 88.46 | 88.46 | 86.54 |
| spearman mean | 0.12 | 0.18 | 0.22 |

**Mate statuses, `extended`:** A13 s0/s1 `{none: 155, missed: 4, allows: 1}`,
s2 `{none: 153, missed: 4, allows: 3}`, against A2's `{none: 153, missed: 4,
allows: 3}` on all three seeds. A13 **allows fewer forced mates** on two of three
seeds. At counts of 1–3 out of 160 this is not a result, but it is the one place
A13 looks better and it is recorded rather than omitted.

---

## 7. A2 → A13 paired comparison, against the A0 noise band

**`extended` (n=160)**

| metric | s0 | s1 | s2 | mean Δ | A0 noise range | verdict |
|---|---|---|---|---|---|---|
| mean regret cp | +13.27 | +58.00 | +26.65 | **+32.64** | 19.70 | all 3 worse, **EXCEEDS** |
| p95 regret cp | +64 | +93 | +132 | **+96.33** | 64 | all 3 worse, **EXCEEDS** |
| max regret cp | +42 | +7 | +81 | **+43.33** | 18 | all 3 worse, **EXCEEDS** |
| top-1 agreement % | −4.38 | −5.00 | −2.50 | **−3.96** | 3.74 | all 3 worse, **EXCEEDS** |
| blunder rate | +0.0361 | +0.1002 | +0.0326 | **+0.0563** | 0.0395 | all 3 worse, **EXCEEDS** |
| median regret cp | +1 | +38 | −15 | +8.00 | 10 | inconsistent |
| top-3 containment % | +3.13 | −3.76 | +1.87 | +0.41 | 0.63 | inconsistent |
| spearman mean | −0.01 | −0.03 | +0.06 | +0.007 | 0.01 | inconsistent |

**`phase0_52` (n=52)**

| metric | s0 | s1 | s2 | mean Δ | A0 noise range | verdict |
|---|---|---|---|---|---|---|
| mean regret cp | +33.10 | +26.69 | +32.92 | +30.90 | 31.65 | all 3 worse, within noise |
| blunder rate | +0.0652 | +0.0434 | +0.0700 | +0.0595 | 0.0624 | all 3 worse, within noise |
| p95 regret cp | +53 | +90 | +8 | +50.33 | 53 | all 3 worse, within noise |
| spearman mean | −0.17 | −0.11 | 0.00 | −0.093 | 0.06 | exceeds, inconsistent |
| top-1 agreement % | −11.54 | 0.00 | +3.85 | −2.56 | 5.77 | inconsistent |

**Five metrics on `extended` degrade consistently across all three paired seeds
and all five exceed the A0 noise band.** On `phase0_52` the *direction* is also
consistent for mean regret, p95 and blunder rate, but the magnitudes sit just
inside that suite's wider band (n=52), so it corroborates without independently
clearing the bar.

No metric improves consistently and by more than noise. Per the instruction not
to call a result an improvement on one favourable movement: A13's better mate
counts and its `extended` top-3/regret-coverage movements are not improvements —
they are small, inconsistent, or within noise.

---

## 8. A0 → A13 (against the control)

| metric (extended) | mean Δ | A0 noise | verdict |
|---|---|---|---|
| mean regret cp | **+35.88** | 19.70 | all 3 worse, exceeds |
| top-1 agreement % | **−5.21** | 3.74 | all 3 worse, exceeds |
| max regret cp | +60.00 | 18 | all 3 worse, exceeds |
| blunder rate | +0.0553 | 0.0395 | all 3 worse, exceeds |
| p95 regret cp | +64.00 | 64 | all 3 worse, at the band edge |

A13, like A3, is worse than the A0 control on every headline `extended` metric.

---

## 9. Side and phase analysis

**Mean regret cp by side to move, averaged over three seeds:**

| suite | side | n | A0 | A2 | A3 | **A13** | A2 → A13 |
|---|---|---|---|---|---|---|---|
| extended | black | 75 | 175.54 | **139.07** | 174.07 | 157.98 | +18.90 (mixed) |
| extended | white | 85 | 106.82 | 145.60 | 202.37 | **190.13** | **+44.52 (all 3)** |
| phase0_52 | black | 22 | 91.79 | **63.18** | 120.93 | 109.85 | +46.68 (all 3) |
| phase0_52 | white | 30 | 113.33 | 149.45 | 149.01 | **167.30** | +17.86 (all 3) |

As with A3, **the damage is concentrated on White-to-move on `extended`** — the
95.8% majority in training and 53% of that suite — and A13 gives back most of
A2's Black-side advantage.

**Mean regret cp by category, A2 → A13:**

| category | n (ext) | extended | consistent | n (p52) | phase0_52 | consistent |
|---|---|---|---|---|---|---|
| opening | 60 | **+60.82** | yes | 18 | +21.13 | yes |
| endgame | 21 | +22.81 | yes | 12 | +26.85 | yes |
| middlegame | 64 | +17.74 | no | 12 | +62.50 | yes |
| defensive | 5 | −11.17 | no | 4 | −25.33 | no |
| tactical | 10 | −27.36 | no | 6 | −37.50 | no |

**Opening and endgame degrade consistently on both suites.** Openings again
degrade most — the phase where castling rights should carry the most
information. As in A3, the added planes hurt most exactly where they should have
helped.

The small cells (n=4–10) move in the opposite direction on both suites but are
inconsistent across seeds and not interpretable.

---

## 10. Mechanism probes — and a revision to the A3 report

### 10.1 The model does use castling

`python -m training.a3_plane_ablation --arm A13` zeroes the four castling planes
on the 1,933 held-out positions:

| arm | mean \|prediction change\| when castling is zeroed | Δ Pearson |
|---|---:|---:|
| A3 (castling at 13–16) | 92.55 cp | −0.089 to −0.100 |
| **A13 (castling at 12–15)** | **84.11 cp** | **−0.089 to −0.108** |

Consistent leverage across both arms. The model genuinely depends on castling —
and that dependence leaves it *worse* than the A2 model that never had the
information. Leverage is not benefit.

### 10.2 Saturation does NOT explain A13 — which revises A3 §12

`python -m training.a2_saturation_probe`, 5,639 candidate positions:

| arm | raw \|cnn\| | saturated (\|tanh\|>0.95) | within-position tanh spread, White to move |
|---|---:|---:|---:|
| A0 | 98.8 | 6.2% | 0.6458 |
| A2 | 110.0 | 3.7% | 0.9547 |
| A3 | 173.7 | **13.8%** | 0.7249 |
| **A13** | **111.2** | **3.1%** | **1.1426** |

**A13 has the lowest saturation and the best candidate separation of any arm —
and still regresses by +32.6 cp.**

The A3 report's §12 proposed the chain: *irrelevant feature → larger outputs →
tanh saturation → less candidate separation → worse move selection*. A13
falsifies that as the general mechanism. A3's saturation was real and specific to
A3; it is **not necessary** for a representation-driven engine regression.

I am flagging this explicitly because A3's report presented that chain as its
central explanation. It stands as a description of A3, not as the cause of
representation regressions in general.

### 10.3 White vs Black prediction error

| arm | White MAE (n=1,853) | Black MAE (n=80) | gap | Black cp MAE | Black mate MAE |
|---|---|---|---|---|---|
| A2 | 155.7 / 157.6 / 159.2 | 276.5 / 286.8 / 297.7 | **+129.5** | 218–260 | 331–552 |
| A3 | 180.6 / 180.7 / 182.1 | 321.8 / 321.1 / 335.9 | +145.1 | 321–344 | 248–336 |
| **A13** | **186.8 / 180.6 / 184.0** | 373.7 / 343.9 / 336.6 | **+167.6** | 264–290 | **537–641** |

A13 degrades the majority White rows by ~26 cp (A3: ~23 cp) and has the widest
Black-minus-White gap of the three. Its Black *mate* rows are the worst of any
arm since A1.

### 10.4 What the three arms have in common

The user's diagnostic question was whether A3's regression came from castling,
side-to-move, en-passant, or their interaction. The evidence:

| addition | present in | offline deficit vs A2 |
|---|---|---|
| side-to-move + castling + en-passant | A3 | −0.095 Pearson |
| **castling only** | **A13** | **−0.114 Pearson** |
| en-passant | A3 only | inert (0.15 cp ablation, 0.27% coverage) |

**Castling alone is sufficient to produce the regression.** Side-to-move is not
necessary. En-passant is inert. If anything A3 — which has all three — is
slightly *less* bad than A13, so the interaction does not compound.

That pattern points at a hypothesis neither arm was designed to test: the common
property of the harmful additions is that they are **spatially constant broadcast
planes** (A13 adds four; A3 adds five constant plus one sparse spatial). On
7,734 training samples, such channels may give the network a cheap global signal
to overfit, independent of semantic content.

**This is a hypothesis, not a finding.** Two arms cannot separate "castling
specifically" from "constant planes generally". §12 names the experiment that
would.

---

## 11. Limitations

1. **A13 used A2's exact recipe, untuned.** Correct for a controlled experiment,
   but it means the result is "these planes under A2's hyperparameters hurt",
   not "castling rights are useless". Unchanged from A3's limitation 1.
2. **`phase0_52` corroborates but does not independently clear the bar.** The
   direction is consistent on mean regret, p95 and blunder rate, but magnitudes
   sit inside that suite's wider noise band.
3. **The constant-plane hypothesis (§10.4) is untested.** It is consistent with
   both A3 and A13 but no experiment discriminates it from a castling-specific
   effect.
4. **Two arms, three seeds each.** No significance testing.
5. **Small cells.** Category breakdowns at n=4–21 and mate counts at 1–4 of 160
   are not interpretable alone.
6. **Frozen production Ridge.** A13's fusion compatibility was not measured; no
   matched-Ridge Stage 2 was run, as instructed. Unlike A3, A13 gives no specific
   reason to suspect a fusion interaction — its saturation is the lowest of any
   arm and its candidate separation the highest — so nothing in this evidence
   makes Stage 2 necessary.
7. **The evaluation shim** is a path A0/A1/A2 did not use. It changes exactly one
   binding and is test-covered, but it is not literally the same invocation.
8. **Dataset-specific.** `dataset_v1` is 95.8% White-to-move and 74.3% has some
   castling right; a differently distributed dataset could behave differently.

---

## 12. Factual interpretation and next experiment

**Interpretation, per the pre-registered rule:** A13 is materially and
consistently worse than A2, so **castling rights alone may be contributing to the
degradation** on this dataset, architecture and recipe. This is diagnostic, not a
production recommendation, and one experiment cannot establish that castling
rights are useless in general.

**What A13 settles about A3:** castling alone reproduces the regression;
side-to-move is not necessary for it; en-passant is inert; the three do not
compound. A3 §12's saturation chain does not generalise.

**Recommended next experiment — a placebo-plane control.** Train `A13P`:
12 piece planes plus **four spatially-constant planes carrying no positional
information** (e.g. four channels fixed by a per-position hash, or simply four
all-zero planes plus four all-one planes to match parameter count). Same labels,
split, seeds, recipe.

- If A13P regresses by a similar ~0.11 Pearson, the cause is **adding constant
  broadcast channels to this architecture on this sample size**, not castling
  semantics — and the whole representation-completion direction needs an
  architectural change (e.g. injecting scalar state at the dense layer rather
  than as input planes) before it can be tested fairly.
- If A13P matches A2, the harm is **specific to castling information**, which
  would be a genuinely surprising and interesting result worth pursuing.

This is the cheapest experiment that discriminates the two live hypotheses, and
it needs no new evaluation infrastructure.

---

## 13. Production-safety verification

- `engine.py`, `app.py`, `config.py`, `evaluation/`, `models/`, and the
  baseline/regression fixtures are **unmodified** — `git status` reports no
  changes under any of them.
- **Production model hash unchanged:** `cnn_model.keras`
  `972d81199a1355667fca554b…`
- **Production Ridge hash unchanged:** `weight_model.pkl`
  `59a731127cc9b440c866c7a5…`, coefficients still
  `[330.9005, 32.3899, 0.7919, 5.1654, 0.0188]`
- `mlp_model.pkl`, `rf_model.pkl`, `scaler.pkl` digests also unchanged.
- **Production engine behaviour unchanged:** `engine.board_to_planes` still
  allocates `(8, 8, 12)`; two tests assert it still produces 12 planes and still
  cannot distinguish boards differing only in castling rights.
- All three A13 evaluation runs logged `verified: production models/ untouched`.
- All A13 artifacts are under `training/experiments/A13/`, inside the
  already-gitignored `training/experiments/` directory.
- **Tests: 531 passed, 10 xfailed, 0 failed** before the A13 evaluation runs;
  re-run after completion below.

---

## Reproducing

```bash
# train
python -m training.train --arm A13 --seed 0      # and 1, 2

# evaluate (dispatches to the plane shim automatically, by plane count)
python -m training.evaluate_arm --arm A13 --seed 0   # and 1, 2

# analysis
python -m training.a2_analysis --arms A0 A2 A3 A13 --contrasts A2:A13 A0:A13 \
    --out training/experiments/A13/a13_analysis.json
python -m training.a2_perspective_probe --arms A0 A2 A3 A13 --contrast A2:A13 \
    --out training/experiments/A13/perspective_probe.json
python -m training.a2_saturation_probe --arms A0 A2 A3 A13 --contrast A2:A13 \
    --out training/experiments/A13/saturation_probe.json
python -m training.a3_plane_ablation --arm A13
```

The analysis scripts keep their A2 defaults and the ablation keeps its A3
default, so the earlier reports still reproduce unchanged (verified).

Artifacts (gitignored): `training/experiments/A13/a13_analysis.json`,
`perspective_probe.json`, `saturation_probe.json`, `plane_ablation.json`, and
per-seed `metadata.json` / `test_predictions.json` / `evaluation/*.json`.
