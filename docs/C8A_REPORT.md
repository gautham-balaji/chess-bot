# C8a — Dataset Expansion (`dataset_v2`)

## 1. Status

Complete. 3 seeds, both suites, frozen production Ridge. No production file
changed. **C8b has not been run.**

---

## Result up front

> **C8a is the first arm in the entire C6/C7/C8 programme to beat A2, and it
> does so decisively.**
>
> On `extended`, **7 of 10 headline metrics exceed the A0 noise band and are
> consistent across all three paired seeds**: mean regret −35.5 cp, median
> regret −20.7 cp, top-1 +4.2 pts, top-3 +6.9 pts, blunder rate −0.069,
> Spearman +0.103, regret coverage +0.84. Nothing regresses consistently.
> `phase0_52` replicates with 4 band-exceeding consistent wins.
>
> The model-level result is not a distribution artifact. Scored on **A2's own
> held-out set**, C8a still wins: MAE 138.96 vs 162.88, Pearson 0.8044 vs
> 0.7381.

After five arms that changed representation or architecture and all failed, the
dataset was the binding constraint.

---

## 2. Experiment question

Does replacing the opening-biased `dataset_v1` with the larger, game-diverse
`dataset_v2` improve the CNN's model-level and engine-level performance?
A2 is the control; **only the training dataset changes.**

---

## 3. Fixed controls

| | value | how it is held |
|---|---|---|
| architecture | A2 Sequential, `(8,8,12)`, **2,360,129 params** | `train_v2` imports `train.build_model`; a guard aborts on any other count |
| representation | `planes12` | imported `training.representation` |
| labels | `corrected_mate_white_perspective` | `dataset.apply_label_policy`, **re-derived and asserted equal** to the stored labels |
| loss / optimiser | Huber, Adam 1e-3, batch 64 | `train.HP` (the same object, not a copy) |
| callbacks | ReduceLROnPlateau (0.5, p5), EarlyStopping (p10, restore best) | `train.make_callbacks()` |
| epoch cap | 100 | `train.HP` |
| seeds | 0, 1, 2 | — |
| fusion | frozen production Ridge | not refitted |
| evaluator | `evaluation/evaluate.py`, unmodified | direct path (12-plane arm needs no shim) |
| Stockfish | 17.1, depth 8, Threads=1, Hash=16MB, Clear Hash per position | — |
| suites | `extended` (160), `phase0_52` (52) | unmodified |

`training/train.py` was **not modified**. C8a lives in `training/train_v2.py`,
which imports the architecture, hyperparameters, callbacks, determinism setup and
metric functions rather than restating them, so it cannot drift from A2. A
separate module was necessary because `train.py` derives its split at training
time via `dataset.make_split` (position-level); re-deriving that over
`dataset_v2` would destroy the game-level separation C7/C8 exists to guarantee.

**Hygiene, verified before training:** dataset_v2 manifest and both JSONL hashes
matched; train∩test game keys = 0; train∩test placements = 0; labels re-derived
through the A2 policy and equal to the stored field; architecture 2,360,129
params; `is_dataset_v1: false` recorded per run; production hashes snapshotted.

**Environment:** Python 3.12.10, Windows 11 AMD64, TensorFlow 2.21.0,
Keras 3.13.2, NumPy 2.4.3, scikit-learn 1.8.0, python-chess 1.11.2,
pandas 2.3.3, stockfish 4.0.8.

---

## 4. Dataset comparison

| | `dataset_v1` (A2) | `dataset_v2` (C8a) |
|---|---:|---:|
| train records | 7,734 | **54,812** |
| test records | 1,933 | **13,712** |
| split unit | position | **game content** |
| White to move (train) | **95.9%** | **58.6%** |
| phase mix (train) | 100% opening | 27.0 / 62.9 / 10.1 |
| endgame positions | **0** | 5,510 |
| min piece count | 22 | 2 |
| checkmate records | 148 (1.5%) | 4,658 (8.5%) |
| mate-typed labels | 1.9% | 10.2% |

**These are two changes at once — size (7.1×) and composition.** See §15.

---

## 5. Training results

| run | epochs | best ep | best val Huber | final train loss | test Huber | MAE | RMSE | Pearson | secs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 s0 | 25 | 15 | 155.31 | — | 160.20 | 160.70 | 257.92 | 0.7450 | 185 |
| A2 s1 | 58 | 48 | 157.32 | — | 162.48 | 162.98 | 262.09 | 0.7360 | 401 |
| A2 s2 | 46 | 36 | 154.84 | — | 164.46 | 164.96 | 264.82 | 0.7332 | 329 |
| **C8a s0** | 18 | 8 | 217.34 | 123.47 | 219.64 | 220.14 | 360.51 | 0.8760 | 874 |
| **C8a s1** | 25 | 15 | 221.92 | 94.31 | 221.75 | 222.25 | 362.76 | 0.8742 | 1094 |
| **C8a s2** | 22 | 12 | 217.73 | 111.35 | 220.13 | 220.63 | 364.46 | 0.8732 | 1128 |

> ⚠️ **These two blocks are NOT comparable.** A2 is scored on dataset_v1's 1,933
> opening-only held-out records; C8a on dataset_v2's 13,712 game-diverse ones.
> Higher raw Huber on a harder set means nothing. §10 resolves this.

Model weight hashes:

```
C8a s0  6bd57636a2747ec479992fb0873385167956352d776808ee2ef79af023ad36e2
C8a s1  144594ffff30bdfa3b59eafc8897e811bbaa6ab269c1d5c82a29364e79a24b97
C8a s2  41ee1626feff429a956c2125d83b2dcf5cf86b93ad04a86e3fb556d27152071c
```

History hashes: `e03818b3…`, `2512cb74…`, `1b69c48b…`.

C8a converges in fewer epochs (18–25 vs 25–58) — expected, since one epoch is
7.1× more gradient steps.

---

## 6. Engine results

Three-seed means, production fusion, identical suites and evaluator.

### `extended` (n=160)

| arm | legality | top-1 % | top-3 % | mean regret | median | p95 | max | blunder | coverage | Spearman mean / median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 100 | 18.96 | 33.54 | 139.29 | 31.33 | 565.7 | 684.0 | 0.230 | 95.21 | 0.213 / 0.263 |
| A2 | 100 | 17.71 | 32.29 | 142.53 | 37.67 | 533.3 | 700.7 | 0.229 | 95.62 | 0.193 / 0.237 |
| **C8a** | **100** | **21.88** | **39.17** | **107.03** | **17.00** | **502.3** | **680.0** | **0.160** | **96.46** | **0.297 / 0.347** |

Per-seed C8a: top-1 24.38 / 20.00 / 21.25; mean regret 119.97 / 93.43 / 107.68.

### `phase0_52` (n=52)

| arm | legality | top-1 % | top-3 % | mean regret | median | p95 | max | blunder | coverage | Spearman mean / median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 100 | 21.79 | 40.38 | 104.31 | 20.83 | 506.3 | 614.0 | 0.154 | 87.18 | 0.233 / 0.263 |
| A2 | 100 | 19.23 | 28.85 | 113.81 | 30.50 | 503.7 | 618.0 | 0.167 | 88.46 | 0.267 / 0.360 |
| **C8a** | **100** | **27.57** | **46.16** | **95.38** | **15.17** | **459.3** | **614.0** | **0.148** | **91.02** | **0.293 / 0.323** |

Per-seed C8a: top-1 28.85 / 25.00 / 28.85; mean regret 84.15 / 121.23 / 80.77.

Legality stays 100% in all six runs.

---

## 7. White / Black analysis

C7 predicted that fixing the 95.9%-White training skew should help Black.
**It helped both sides, and narrowed the gap.**

**Mean regret, `extended`** (A2 → C8a, paired):

| side | A2 | C8a | Δ |
|---|---:|---:|---:|
| black (n=75) | 139.07 | 105.54 | **−33.55** |
| white (n=85) | 145.60 | 108.39 | **−37.23** |

**Model MAE by side to move** (from the cross-evaluation):

| held-out set | arm | White | Black | gap |
|---|---|---:|---:|---:|
| dataset_v1_test | A2 | 157.5 | 287.0 | +129.5 |
| dataset_v1_test | **C8a** | **134.6** | **239.9** | **+105.3** |
| dataset_v2_test | A2 | 352.1 | 443.8 | +91.7 |
| dataset_v2_test | **C8a** | **194.6** | **258.8** | **+64.2** |

The White/Black asymmetry that survived every C6 arm **narrows for the first
time** — on both held-out sets. It does not vanish: Black is still harder.

On `phase0_52` the side split is mixed (White −38.8, Black +11.39) on n=22/30;
too small to read against the A0 band.

---

## 8. Phase analysis

**Model MAE by phase, `dataset_v2_test`:**

| phase | A2 | C8a | Δ |
|---|---:|---:|---:|
| opening | 123.4 | 113.8 | −9.6 |
| middlegame | 435.9 | 255.3 | −180.6 |
| **endgame** | **795.2** | **290.0** | **−505.2 (−64%)** |

A2 had never seen an endgame; its endgame MAE is 795 cp. This is the single
largest measured effect in C8a and it is exactly what C7 predicted.

**Engine mean regret by category, `extended`** (A2 → C8a mean delta):
defensive −29.67, endgame −11.24, middlegame −38.29, opening −39.20,
tactical −49.58. Every category improves. (`phase0_52`: −6.33, −2.36, −9.75,
−33.80, −57.78.)

Note the engine's endgame regret was already low for A2 (45.86 on `extended`)
despite terrible endgame *evaluation* — the heuristics carry those positions.
The model-level endgame gain is much larger than the engine-level one.

---

## 9. Mate analysis

Summed across three seeds:

| suite | arm | none | missed_forced_mate | allows_forced_mate | both_forced_mate |
|---|---|---:|---:|---:|---:|
| extended | A2 | 459 | 12 | 9 | 0 |
| extended | **C8a** | **463** | **11** | **5** | **1** |
| phase0_52 | A2 | 138 | 12 | 6 | 0 |
| phase0_52 | **C8a** | **142** | **11** | **2** | **1** |

`engine_move_allows_forced_mate` **falls by 44% and 67%** — C8a walks into a
lost forced mate far less often. `missed_forced_mate` improves only marginally
(12 → 11 on both suites), and one seed newly reaches `both_forced_mate_for_mover`.

**Model MAE on checkmate positions** (dataset_v2_test): A2 **1606.0** → C8a
**476.9**. C8a recognises mate far better, but a 477 cp error on a ±2000 target
means it still does not score mate decisively. This is consistent with the
modest `missed_forced_mate` movement: the C8 checkmate records helped, but did
not solve the mate-recognition failure.

---

## 10. Model-level vs engine-level

These are kept separate deliberately. The cross-evaluation scores **every model
on both held-out sets** (`training/c8a_cross_eval.py`), because A2 and C8a do not
share one:

| held-out set | arm | Huber | MAE | RMSE | Pearson |
|---|---|---:|---:|---:|---:|
| dataset_v1_test (1,933) | A2 | 162.38 | 162.88 | 261.61 | 0.7381 |
| dataset_v1_test | **C8a** | **138.46** | **138.96** | **230.45** | **0.8044** |
| dataset_v2_test (13,712) | A2 | 389.31 | 389.81 | 626.89 | 0.5770 |
| dataset_v2_test | **C8a** | **220.51** | **221.01** | **362.58** | **0.8745** |

**C8a wins on A2's own home turf** (−23.9 MAE, +0.066 Pearson on dataset_v1_test),
so the gain is not an artifact of being scored on a different distribution.

**Do model and engine metrics agree here?** Yes — unusually. Both improve, in the
same direction, on both suites. That has *not* been true elsewhere in this
programme: A3/A13/A13R degraded both, and A14 moved neither. No metric class
improves while the other regresses, so this report has no such case to flag.

One genuine divergence in *magnitude*: the model-level endgame gain (−64% MAE) is
far larger than the engine-level endgame regret gain (−11 cp), because the
heuristic terms already covered endgames in the fusion.

---

## 11. Seed variation

| metric (extended) | A2 band | C8a band | A0 band (noise floor) |
|---|---|---|---|
| mean regret | [133.7 .. 152.3] | [93.43 .. 120.0] | [128.7 .. 148.4] |
| top-1 % | [14.38 .. 19.38] | [20.00 .. 24.38] | [16.88 .. 20.62] |
| top-3 % | [29.38 .. 34.38] | [38.12 .. 40.00] | [33.12 .. 33.75] |
| blunder rate | [0.2026 .. 0.2484] | [0.1429 .. 0.1742] | [0.2105 .. 0.2500] |
| Spearman mean | [0.18 .. 0.21] | [0.28 .. 0.31] | [0.21 .. 0.22] |

**C8a's entire three-seed band sits outside A2's** on mean regret, top-1, top-3,
blunder rate and Spearman. Seeds are not ranked and no single seed is used.

C8a's spread is comparable to A2's, so the improvement is not one lucky run:
the worst C8a seed still beats the best A2 seed on top-1, top-3, blunder rate
and Spearman.

---

## 12. Comparison against the A0 noise band

Pre-registered rule: a change counts only if it **exceeds the A0 band** *and* is
**consistent across all three paired seeds**.

### `extended`, A2 → C8a

| metric | per-seed Δ | mean Δ | A0 range | verdict |
|---|---|---:|---:|---|
| mean regret cp | −13.72 / −48.18 / −44.61 | −35.50 | 19.70 | **exceeds, better** |
| median regret cp | −10 / −14 / −38 | −20.67 | 10.0 | **exceeds, better** |
| top-1 % | +5.00 / +0.62 / +6.87 | +4.16 | 3.74 | **exceeds, better** |
| top-3 % | +5.00 / +5.62 / +10.00 | +6.87 | 0.63 | **exceeds, better** |
| blunder rate | −0.028 / −0.092 / −0.086 | −0.069 | 0.0395 | **exceeds, better** |
| Spearman mean | +0.07 / +0.12 / +0.12 | +0.103 | 0.01 | **exceeds, better** |
| regret coverage % | +1.26 / +0.63 / +0.63 | +0.84 | 0.62 | **exceeds, better** |
| p95 regret cp | +8 / −99 / −2 | −31.00 | 64.0 | within band |
| max regret cp | −36 / −29 / +3 | −20.67 | 18.0 | exceeds, inconsistent |
| legality % | 0 / 0 / 0 | 0 | 0 | unchanged |

**7 of 10 metrics pass both tests. None fails in the other direction.**

### `phase0_52`, A2 → C8a

| metric | per-seed Δ | mean Δ | A0 range | verdict |
|---|---|---:|---:|---|
| top-1 % | +5.77 / +5.77 / +13.47 | +8.34 | 5.77 | **exceeds, better** |
| top-3 % | +23.08 / +5.77 / +23.08 | +17.31 | 3.85 | **exceeds, better** |
| regret coverage % | +3.85 / +1.92 / +1.92 | +2.56 | 1.92 | **exceeds, better** |
| median regret cp | 0 / −9.5 / −36.5 | −15.33 | 7.50 | exceeds, one seed flat |
| mean regret cp | −20.18 / −6.51 / −28.60 | −18.43 | 31.65 | within band, all better |
| blunder rate | −0.027 / −0.004 / −0.025 | −0.019 | 0.0624 | within band, all better |
| p95 regret cp | −52 / 0 / −81 | −44.33 | 53.0 | within band |
| Spearman mean | −0.06 / +0.06 / +0.08 | +0.027 | 0.06 | within band, inconsistent |
| max regret cp | 0 / 0 / −12 | −4.00 | 0 | exceeds, inconsistent |
| legality % | 0 / 0 / 0 | 0 | 0 | unchanged |

Also **A0 → C8a** on `extended`: mean regret, median regret, top-3, blunder rate,
Spearman and coverage all exceed the band consistently — C8a beats the noise-floor
control too, which A2 never did.

---

## 13. C8a decision

**C8a is a clear, consistent improvement over A2 at both model level and engine
level.** It is the first arm in this programme to clear the pre-registered bar.

- `extended`: 7/10 metrics exceed the A0 band and are consistent across all three
  paired seeds; none regresses consistently.
- `phase0_52`: 3–4 metrics clear the bar, the rest improve within the band. The
  smaller suite has a wider band, as expected at n=52.
- Model level: C8a wins on **both** held-out sets, including A2's own.
- Legality is unchanged at 100%.

**The dataset was the binding constraint.** C6 tested five representation and
architecture changes against A2 and none beat it; the first dataset change does,
substantially.

### A measured caveat that did *not* block the result

The saturation probe (`training/experiments/C8a/saturation_probe.json`,
5,639 candidate positions) shows C8a is **worse** on the fusion's own terms:

| arm | raw \|cnn\| | saturated (\|tanh\| > 0.95) | within-position tanh spread (White) |
|---|---:|---:|---:|
| A0 | 98.8 | 6.2% | 0.6458 |
| A2 | 110.0 | 3.7% | 0.9547 |
| **C8a** | **155.5** | **8.1%** | **0.5819** |

C8a's CNN output magnitude is inflated (mean \|pred\| 427.7 vs A2's 223.8 on
dataset_v2_test), partly because 8.45% of its training records carry ±2000 mate
labels. That inflation pushes **2.2× more candidates into tanh's saturated
region** and **cuts within-position candidate separation by 39%**.

**C8a wins anyway.** This is the documented checkmate/output-scale concern showing
up exactly where C8 predicted it would — but as *unrealised headroom*, not as a
failure. The pre-registered trigger ("if C8a underperforms **and** diagnostics
point at mate/output-scale saturation") has **not** fired, because C8a does not
underperform. No dataset change is warranted, and none was made.

---

## 14. Is C8b justified?

**Yes — the signal is clear enough to justify it, but it is NOT run here.**

The pre-registered condition was "a sufficiently clear positive signal or an
otherwise informative scaling result". C8a delivers the former: 7/10 band-clearing
consistent wins on the larger suite, replicated on the smaller one.

C8b (`evenly_spaced_6_minply16`, 99,124 records, 79,202 train) would answer
whether the gain keeps scaling with data or has saturated at ~55k. Measured cost:
the dataset labels in ~11 minutes and each seed trains in ~20 minutes, so the full
arm is roughly 2 hours.

**Recommendation and its competitor.** Before C8b, the saturation finding in §13
identifies a plausibly larger and cheaper win: the frozen production Ridge was
fitted to the *original* CNN's output scale, and C8a is losing 8.1% of its
candidates to tanh saturation under it. A matched-fusion refit against C8a could
recover signal C8a already has. That is a fusion experiment, explicitly out of
scope here, and is recorded as a candidate — **not launched**.

**C8b has not been started. C9 has not been started.**

---

## 15. Limitations

1. **Size and composition are confounded, and this is not hidden.** C8a differs
   from A2 in dataset size (7.1×) *and* composition (opening-only → phase-mixed,
   95.9% → 58.6% White). **No improvement here can be attributed to size alone.**
   Isolating them needs a size-matched control — `dataset_v2` subsampled to 7,734
   train records under the same split — which is one 3-seed run and was **not**
   performed.
2. **Different held-out sets.** A2's and C8a's own reported test metrics are not
   comparable; §10's cross-evaluation exists solely to fix this and is the only
   model-level comparison that should be quoted.
3. **Frozen Ridge caps every arm.** Its coefficients were fitted to the original
   CNN's output scale. This biases all arms identically so the comparison stays
   fair, but no arm should be read as the best this architecture can do — and §13
   shows C8a is penalised more than A2 by it.
4. **Three seeds, descriptive only.** No significance testing. The A0 band is the
   decision instrument.
5. **Small suites.** n=160 and n=52. Category cells run as small as n=4–10; the
   tactical deltas (−49.58, −57.78) rest on n=10 and n=6 and should not be read
   as findings.
6. **Mate recognition is improved but not fixed.** Checkmate MAE 1606 → 477 cp,
   yet `missed_forced_mate` only moves 12 → 11 on both suites.
7. **One dataset population.** 20,058 Lichess games. C8a says more and better-
   distributed positions from *this* corpus help; it says nothing about other
   provenance, rating bands or time controls.
8. **Training-set overfitting gap.** Train MAE 134–173 vs test 220–222 suggests
   C8a still has capacity headroom or needs regularisation — untested.

---

## 16. Production safety

**PASS — all 27 production artifacts byte-identical** to the pre-experiment
snapshot: `models/*.keras`, `models/*.pkl` (including the Ridge
`weight_model.pkl`), both evaluation suites, all `baseline/`, all `regression/`,
`dataset_v1.jsonl`, `dataset_v1.manifest.json`, `dataset_v2.manifest.json`, and
(LF-normalised) `engine.py`, `app.py`, `config.py`, `evaluation/evaluate.py`,
`training/labels.py`, `training/dataset.py`, `training/train.py`.

`dataset_v2.train.jsonl` and `dataset_v2.test.jsonl` still match their manifest
hashes. All C8a models, predictions and evaluation output live under
`training/experiments/C8a/`, which is gitignored.

Full suite: **792 passed, 10 xfailed** (the pre-existing Phase-4 deferrals). No
test was weakened or deleted.

---

## 17. Reproduction

```bash
# train (≈20 min per seed)
python -m training.train_v2 --arm C8a --seed 0     # and 1, 2

# engine evaluation, unmodified evaluator (12-plane arm, no shim)
python -m training.evaluate_arm --arm C8a --seed 0 # and 1, 2

# analysis
python -m training.a2_analysis --arms A0 A2 C8a --baseline A0 \
    --contrasts A2:C8a A0:C8a --out training/experiments/C8a/c8a_analysis.json
python -m training.c8a_cross_eval --arms A2 C8a
python -m training.a2_saturation_probe --arms A0 A2 C8a --contrast A2:C8a \
    --out training/experiments/C8a/saturation_probe.json

# verification
python -m pytest tests/unit/test_training_train_v2.py -q
```

Artifacts: `training/experiments/C8a/` (gitignored) — per-seed `metadata.json`,
`test_predictions.json`, `models/cnn_model.keras`, `evaluation/{extended,
phase0_52}.{json,md}`, `summary.json`, plus `c8a_analysis.json`,
`cross_eval.json`, `saturation_probe.json`.
