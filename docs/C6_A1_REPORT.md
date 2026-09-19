# C6-A1 — Mate-Label Correction

**Status: complete. Seeds 0, 1 and 2 trained and evaluated.**

No production file was modified. Every evaluation asserted afterwards that
`models/` was untouched.

---

## 1. Objective

Test whether correcting the Stockfish mate-label representation changes model
behaviour and downstream chess evaluation, measured against the A0 noise floor.

A0 used the legacy notebook policy, under which mate scores survived as raw mate
distances — mate-in-1 stored as `1`, an already-checkmated position as `0`. A1
replaces **only** that treatment with the repaired mate scale from C6-Prep.

---

## 2. Experimental delta

| | A0 | A1 |
|---|---|---|
| Representation | `planes12` | `planes12` (unchanged) |
| cp labels | `clip(raw_stm, ±1500)` | **identical** |
| mate labels | raw mate distance | **repaired magnitude scale** |
| Perspective | side-to-move relative | **side-to-move relative (unchanged)** |

**Changed:** mate-label treatment. **Nothing else.**

### A1 is not A2

A1 keeps the legacy side-to-move perspective. `training/labels.py`'s `make_label`
couples mate mapping with White-positive conversion, so A1 does **not** use it.
Instead it reuses the already-verified primitive:

```python
labels.mate_to_white_positive(d, side_to_move_is_white=True)
```

This is exact rather than a workaround: when the side to move **is** White,
"White-positive" and "side-to-move-relative" are the same frame by definition, so
passing `True` yields the stm-relative signed magnitude. **No new mate arithmetic
was written.**

The policy is recorded explicitly as `corrected_mate_legacy_perspective`, with
`applies_perspective_normalisation: False` in every run's metadata. Five tests
assert A1 ≠ A2, including that a Black-to-move mate-in-1 is `+1990` under A1 and
`−1990` under the A2-style policy.

---

## 3. Fixed configuration

Held identical to A0 and verified per seed from recorded metadata: dataset
(`5a689e3f…`, 9,667 records, manifest-checked before every run) · representation
`planes12` · split 7,734/1,933, seed 42, **byte-identical to A0** · architecture
**2,360,129 params** · Huber / Adam 1e-3 · batch 64 · epoch cap 100 ·
ReduceLROnPlateau 0.5/5 · EarlyStopping 10 + restore-best · `validation_split=0.1`
· production Ridge weights · unmodified Phase 3 evaluator · Stockfish 17.1,
depth 8, Threads 1, Hash 16 MB, Clear Hash per position.

Cross-checked: hyperparameters A0 == A1 ✓ · dataset sha A0 == A1 ✓ ·
representation A0 == A1 ✓ · split A0 == A1 ✓ · weights distinct across all six
runs ✓.

---

## 4. Label-policy audit

Performed **before** training, over all 9,667 records.

| Quantity | Value |
|---|---:|
| Total records | 9,667 |
| **Changed (A0 → A1)** | **188** |
| Unchanged | 9,479 |
| Changed by type | `mate`: 188 · **`cp`: 0** |
| Total mate records in dataset | 188 |
| Black-to-move records changed | 101 |
| Label delta min / max | −2000 / +1989 |
| A1 label range | −2000 … +1990 |

**Exactly the 188 mate records changed, and no cp record was touched** — which is
the definition of the intended isolation. This matched the prediction before
training, so no stop condition was triggered.

Examples: mate-in-1 `1 → 1990`; checkmate-on-board `0 → −2000`; mated-in-3
`−3 → −1970`.

---

## 5. Model results — A1 seeds 0 / 1 / 2

Test split n = 1,933, identical to A0.

| Metric | s0 | s1 | s2 | mean | sd | min | max | range | *A0 range* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Huber test loss | 185.375 | 189.760 | 190.320 | 188.48 | 2.71 | 185.37 | 190.32 | 4.95 | *161.8 – 167.0* |
| RMSE (cp) | 312.90 | 313.52 | 317.18 | 314.53 | 2.31 | 312.90 | 317.18 | 4.28 | *240.8 – 245.4* |
| MAE (cp) | 185.87 | 190.26 | 190.82 | 188.98 | 2.71 | 185.87 | 190.82 | 4.95 | *162.3 – 167.5* |
| **Pearson r** | **0.6055** | **0.6040** | **0.5919** | **0.6005** | 0.0075 | 0.5919 | 0.6055 | 0.0137 | *0.509 – 0.530* |
| Best epoch | 13 | 25 | 31 | 23.0 | 9.2 | 13 | 31 | 18 | *7 – 12* |
| Epochs run | 23 | 35 | 41 | 33.0 | 9.2 | 23 | 41 | 18 | *17 – 22* |
| Invalid predictions | 0 | 0 | 0 | — | — | — | — | 0 | *0* |

**Reading these correctly matters.**

- **Pearson r rose to 0.592–0.606, entirely above the A0 range (0.509–0.530).**
  Correlation is scale-invariant, so this is a like-for-like comparison: A1's
  labels are genuinely easier to rank-fit.
- **RMSE/MAE/Huber are NOT comparable between arms.** A1's targets now reach
  ±2000 where A0's mate targets sat near 0, so the target variance is far larger.
  The increase from ~243 to ~315 cp RMSE reflects a different label distribution,
  not a worse fit. Comparing them across arms would be a category error.
- A1 trained substantially longer (23–41 epochs vs A0's 17–22).

Per the brief, these are **not** compared to the old notebook's Pearson.

---

## 6. Engine results — A1 seeds 0 / 1 / 2

Unmodified Phase 3 evaluator. Full per-seed tables including White/Black
breakdowns are below in §7.

### `extended` (n=160) — primary

| Metric | s0 | s1 | s2 | A1 mean | A1 sd | A1 range |
|---|---:|---:|---:|---:|---:|---|
| Legality % | 100 | 100 | 100 | 100 | 0 | 100 – 100 |
| Top-1 agreement % | 17.50 | 16.88 | 19.38 | 17.92 | 1.30 | 16.88 – 19.38 |
| Top-3 containment % | 36.88 | 32.50 | 30.62 | 33.33 | 3.21 | 30.62 – 36.88 |
| Mean regret cp | 150.88 | 146.84 | 169.99 | 155.90 | 12.38 | 146.8 – 170.0 |
| Median regret cp | 38.5 | 38 | 50 | 42.17 | 6.79 | 38 – 50 |
| p95 regret cp | 580 | 569 | 637 | 595.3 | 36.5 | 569 – 637 |
| Blunder rate | 0.2566 | 0.2368 | 0.2810 | 0.2581 | 0.0221 | 0.237 – 0.281 |
| Regret coverage % | 95.00 | 95.00 | 95.62 | 95.21 | 0.36 | 95.0 – 95.6 |
| Spearman mean | 0.17 | 0.18 | 0.22 | 0.19 | 0.0265 | 0.17 – 0.22 |
| Spearman median | 0.24 | 0.24 | 0.27 | 0.25 | 0.0173 | 0.24 – 0.27 |

### `phase0_52` (n=52) — secondary

> Contains 6 positions (11.5%) that appear verbatim in the training set.

| Metric | s0 | s1 | s2 | A1 mean | A1 sd | A1 range |
|---|---:|---:|---:|---:|---:|---|
| Legality % | 100 | 100 | 100 | 100 | 0 | 100 – 100 |
| Top-1 agreement % | 21.15 | 15.38 | 23.08 | 19.87 | 4.01 | 15.38 – 23.08 |
| Top-3 containment % | 32.69 | 25.00 | 32.69 | 30.13 | 4.44 | 25.0 – 32.7 |
| Mean regret cp | 118.31 | 123.93 | 109.00 | 117.08 | 7.54 | 109.0 – 123.9 |
| Median regret cp | 31 | 40 | 41 | 37.33 | 5.51 | 31 – 41 |
| p95 regret cp | 579 | 524 | 472 | 525.0 | 53.51 | 472 – 579 |
| Blunder rate | 0.1556 | 0.1556 | 0.1522 | 0.1545 | 0.0020 | 0.152 – 0.156 |
| Spearman mean | 0.23 | 0.20 | 0.15 | 0.1933 | 0.0404 | 0.15 – 0.23 |

---

## 7. A0 → A1 paired comparison

Seed pairing preserved: A0 s0 ↔ A1 s0, etc.

### `extended` — per-seed deltas (A1 − A0)

| Metric | seed 0 | seed 1 | seed 2 | direction |
|---|---:|---:|---:|---|
| Legality % | +0 | +0 | +0 | unchanged |
| Top-1 agreement % | −1.88 | −3.74 | +2.50 | **mixed** |
| Top-3 containment % | +3.13 | −0.62 | −3.13 | **mixed** |
| **Mean regret cp** | **+22.15** | **+6.13** | **+21.59** | **all 3 higher** |
| **Median regret cp** | **+8.5** | **+11** | **+13** | **all 3 higher** |
| p95 regret cp | +50 | −4 | +43 | 2 of 3 higher |
| **Blunder rate** | **+0.0461** | **+0.0080** | **+0.0310** | **all 3 higher** |
| Regret coverage % | +0 | −0.62 | +0.62 | mixed |
| Spearman mean | −0.04 | −0.03 | +0.00 | 2 lower, 1 flat |

### Where A1 sits relative to the A0 observed range — `extended`

| Metric | A0 range | A1 range | Relation |
|---|---|---|---|
| Legality % | 100 – 100 | 100 – 100 | identical |
| Top-1 agreement % | 16.88 – 20.62 | 16.88 – 19.38 | **inside** |
| Top-3 containment % | 33.12 – 33.75 | 30.62 – 36.88 | wider both ways |
| **Mean regret cp** | 128.7 – 148.4 | **146.8 – 170.0** | **shifted up; overlap only at the edge** |
| **Median regret cp** | 27 – 37 | **38 – 50** | **no overlap — entirely above** |
| p95 regret cp | 530 – 594 | 569 – 637 | shifted up |
| **Blunder rate** | 0.2105 – 0.2500 | **0.2368 – 0.2810** | shifted up |
| Spearman mean | 0.21 – 0.22 | 0.17 – 0.22 | shifted down |
| Regret coverage % | 95.0 – 95.6 | 95.0 – 95.6 | identical |

### `phase0_52` — per-seed deltas (A1 − A0)

| Metric | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| Top-1 agreement % | +0 | −9.62 | +3.85 |
| Top-3 containment % | −9.62 | −13.46 | −7.69 |
| Mean regret cp | +27.13 | +1.10 | +10.07 |
| Median regret cp | +10 | +15.5 | +24 |
| p95 regret cp | +108 | +0 | −52 |
| Blunder rate | +0.0223 | −0.0401 | +0.0189 |
| Spearman mean | −0.03 | −0.04 | −0.05 |

On `phase0_52`, **top-3 containment fell in all three seeds** (−7.7 to −13.5 pp,
A1 range 25.0–32.7 entirely below A0's 38.5–42.3) and **Spearman mean fell in all
three**.

### By side to move — `extended` (A0 → A1 mean regret)

| Side | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| White (n=85) | 93.81 → 98.33 | 102.32 → 123.33 | 124.34 → 141.04 |
| Black (n=75) | 167.53 → 210.82 | 183.89 → 173.66 | 175.19 → 202.62 |

Mean regret rose for White in all three seeds and for Black in two of three.

### Legality and selected moves

**Legality was 100% for all three A1 seeds on both suites** — unchanged from A0,
and the only engine metric with zero movement across all twelve runs.

---

## 8. Mate-specific analysis

A1 targets mate labels directly, so this is the most focused test of its intent.

### `extended`

| Seed | Arm | `none` | `missed_forced_mate` | `engine_move_allows_forced_mate` | `both_forced_mate_for_mover` |
|---|---|---:|---:|---:|---:|
| 0 | A0 | 152 | 4 | 4 | — |
| 0 | A1 | 152 | **3** | 4 | 1 |
| 1 | A0 | 153 | 4 | 3 | — |
| 1 | A1 | 152 | **3** | 4 | 1 |
| 2 | A0 | 152 | 3 | 4 | 1 |
| 2 | A1 | 153 | 3 | **3** | 1 |

`missed_forced_mate` went 4/4/3 → 3/3/3. `engine_move_allows_forced_mate` went
4/3/4 → 4/4/3.

### `phase0_52`

`missed_forced_mate` 4/4/3 → 3/3/3; `engine_move_allows_forced_mate` 3/2/3 →
3/3/2.

### Interpretation

`missed_forced_mate` moved from {4,4,3} to {3,3,3} on both suites — a shift of at
most one position per seed. **A0 established that these categories vary by ±1
between seeds at these counts**, and indeed A0's own seeds ranged 3–4 on exactly
this metric. **A one-position change is inside the noise A0 already measured and
is not evidence of an effect.**

Only 7–8 of 160 positions involve mate at all, so this experiment has very little
resolving power on the metric it most directly targets. That is a limitation of
the evaluation suites, not of A1.

---

## 9. A tested mechanism hypothesis that did NOT hold

Because A1's labels now reach ±2000 while the engine computes
`cnn_norm = tanh(cnn_score / 200)` — which saturates beyond roughly ±600 cp — a
natural hypothesis was that A1's models produce larger outputs, saturate `tanh`
more often, and thereby lose discrimination between candidate moves.

Measured directly on 1,856 post-move positions from the `extended` suite:

| Model | mean abs output | p95 abs | `|tanh|>0.99` | `tanh` sd |
|---|---:|---:|---:|---:|
| A0 s0 / s1 / s2 | 203.3 / 124.8 / 125.6 | 1140.9 / 586.5 / 532.7 | 14.87% / 5.93% / 5.17% | 0.415 / 0.467 / 0.453 |
| A1 s0 / s1 / s2 | 181.2 / 123.6 / 169.6 | 986.9 / 459.3 / 735.8 | 10.72% / 3.72% / 8.51% | 0.454 / 0.509 / 0.505 |

**The hypothesis is not supported.** A1's outputs are not systematically larger —
A1 seed 1 has the *lowest* saturation of all six models — and A1's `tanh` spread
is if anything slightly *wider*, i.e. marginally more discrimination, not less.

This is recorded because the hypothesis was tested and rejected. **No mechanism
for the observed engine-level shift has been established**, and none is asserted.

---

## 10. Reproducibility

Per-seed verification (all three seeds, all checks passed): seed recorded
correctly · dataset checksum matches the committed manifest · split byte-identical
to A0 · `planes12` · 2,360,129 params · label policy
`corrected_mate_legacy_perspective` with `applies_perspective_normalisation: False`
· model weights sha256 recorded · **0 invalid predictions** · outputs isolated
under `training/experiments/A1/seed_<n>/`.

All six model weight hashes (A0 ×3, A1 ×3) are distinct. `enable_op_determinism()`
succeeded on every run. A0 already established that the harness reproduces a seed
exactly at the weights level; no seed was rerun here, and no anomaly prompted one.

---

## 11. Production safety

| Check | Result |
|---|---|
| `models/cnn_model.keras` | `972d81199a1355667f…` unchanged |
| `models/weight_model.pkl` | `59a731127cc9b440c8…` unchanged |
| `engine.py`, `app.py`, `config.py` | unchanged |
| `evaluation/`, `baseline/`, `regression/` | unchanged |
| Ridge weights | production file, not refitted |
| `models/` untouched assertion | passed after all 6 evaluation runs |

`git diff --stat` over all production paths is empty. Injection used the existing
`CHESS_BOT_MODELS_DIR` staging mechanism; no production behaviour was modified.

**Full test suite: 405 passed, 10 xfailed, 0 failures.**

---

## 12. Limitations

1. **Three seeds per arm.** No significance testing has been performed and none is
   appropriate. All spreads are descriptive over n=3.
2. **RMSE/MAE/Huber are not comparable between A0 and A1** — the label
   distributions differ by construction. Only Pearson (scale-invariant) and the
   engine-level metrics support cross-arm comparison.
3. **Historical labels remain unrecoverable**, so neither arm reproduces the
   shipped model. A0 is the control, not a reproduction.
4. **Ridge weights were not refitted** — fixed at the production values for every
   arm. Those coefficients were fitted against the original CNN's output scale.
   This biases all arms in the same direction (fair for comparison), but A1
   changes the label scale substantially, so the fixed fusion layer may suit A1
   less well than A0. **This is a plausible confound for the engine-level result
   and cannot be separated within this experiment's design.**
5. **Model-level test metrics are not chess strength.**
6. **`phase0_52` has 11.5% train/eval overlap**; `extended` is primary.
7. **Mate-status counts are tiny** (7–8 mate-involved positions of 160) and A0
   showed ±1 seed noise in these categories. The suites have little power to
   resolve the effect A1 most directly targets.
8. **A1 trained noticeably longer** (23–41 epochs vs 17–22). Callbacks were
   identical; this is a consequence of the changed label distribution, not a
   protocol difference.
9. **No mechanism has been established** for the engine-level shift; the one
   hypothesis tested (§9) was not supported.

---

## 13. Factual conclusion

- All three A1 seeds trained and evaluated successfully, with **0 invalid
  predictions** and **100% legality on both suites in every seed**.
- The experimental isolation held: **exactly 188 labels changed, all of them mate
  records; zero cp records changed**; perspective, representation, split,
  architecture and protocol were verified identical to A0.
- **Model-level label fit improved on the scale-invariant metric.** A1's Pearson r
  (0.592–0.606) lies entirely above the A0 range (0.509–0.530).
- **Engine-level metrics moved in the opposite direction, consistently.** On the
  primary `extended` suite, mean regret rose in all three paired seeds (+22.2,
  +6.1, +21.6 cp), median regret rose in all three (+8.5, +11, +13 cp), and the
  blunder rate rose in all three (+0.046, +0.008, +0.031). **A1's median-regret
  range (38–50 cp) does not overlap A0's (27–37 cp) at all.** On `phase0_52`,
  top-3 containment and Spearman fell in all three seeds.
- **Top-1 agreement moved inconsistently** (−1.88, −3.74, +2.50) and stayed inside
  the A0 range — it does not distinguish the arms.
- **The mate-status categories A1 targets barely moved.** `missed_forced_mate`
  went {4,4,3} → {3,3,3}, a change of at most one position per seed, which is
  inside the ±1 noise A0 already measured. This experiment has little power on
  that metric.

**A1 is not described as an improvement.** The consistent direction of the regret
and blunder deltas across all three paired seeds, and the non-overlapping
median-regret ranges, are **observations worth further investigation** — not a
demonstration of causality from n=3. In particular, limitation 4 (fixed Ridge
weights fitted to a different output scale) is an unresolved confound that this
design cannot separate from a genuine label effect.

No arm is ranked against another, and A2 has not been started.

---

## Reproducing

```bash
python -m training.train        --arm A1 --seed {0,1,2}
python -m training.evaluate_arm --arm A1 --seed {0,1,2}
```

Outputs in `training/experiments/A1/seed_<n>/` (gitignored).
