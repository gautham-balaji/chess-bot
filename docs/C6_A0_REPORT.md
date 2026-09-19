# C6-A0 — Controlled Retraining / Noise-Floor Experiment

**Status: complete. Seeds 0, 1 and 2 trained and evaluated.**

No production file was modified. `engine.py`, `app.py`, `config.py`, `evaluation/`,
`baseline/`, `regression/` and `models/` are byte-identical; every evaluation run
asserted afterwards that `models/` was untouched.

---

## 1. Objective

A0 measures **how much this architecture's results move from the training seed
alone**, with dataset, labels, split, architecture and hyperparameters held fixed.

It is the reference against which A1/A2/A3 must be read. Phases 4A and 4B each
showed a real correctness fix producing exactly zero measurable change; without a
noise floor, a later arm's difference cannot be distinguished from run-to-run
variance.

**A0 is not an attempt to improve anything.** The three A0 seeds are the control
group. The shipped model is a reference point only — **not a fourth seed**.

---

## 2. Fixed experimental configuration

Identical across all three seeds; the **only** intentional variable is the training
seed. Verified per-seed from the recorded metadata.

| Component | Value |
|---|---|
| Dataset | `training/artifacts/dataset_v1.jsonl`, 9,667 records |
| Representation | `planes12` — 12 planes, **no C6 additions** |
| Label policy | `legacy_notebook` |
| Split | 7,734 train / 1,933 test, split seed **42** |
| Architecture | Conv 64→BN→Conv 128→BN→Conv 128→BN→Flatten→Dense 256→Drop 0.3→Dense 128→Drop 0.2→Dense 1 |
| Parameters | **2,360,129** (matches production exactly) |
| Loss / optimizer | Huber / Adam, lr 1e-3 |
| Batch / epoch cap | 64 / 100 |
| ReduceLROnPlateau | factor 0.5, patience 5, monitor `val_loss` |
| EarlyStopping | patience 10, `restore_best_weights=True` |
| Validation | `validation_split=0.1` |
| Ridge fusion | production `weight_model.pkl`, **unchanged** |
| Evaluator | unmodified Phase 3 `evaluation/evaluate.py` |
| Stockfish | 17.1, depth 8, Threads 1, Hash 16 MB, Clear Hash per position |

Cross-seed verification: hyperparameters identical ✓ · split identical ✓ · dataset
sha identical ✓ · label policy identical ✓ · representation identical ✓ · **model
weights hashes all distinct** ✓ (so the seed did vary what it was supposed to vary).

---

## 3. Dataset identity

| | |
|---|---|
| Records | 9,667 |
| Dataset sha256 | `5a689e3f37156a05…` — **verified against the committed manifest before every run** |
| Source sha256 | `e7aadff104a610af…` (`games.csv`, 20,058 rows) |

The trainer refuses to run if the dataset checksum does not match its manifest.

---

## 4. A0 label policy

`dataset_v1` retains, per position, the **raw pre-policy Stockfish value** and
`eval_type` alongside the C6-Prep repaired label. The A0 control is therefore a
**derivation, not a reconstruction and not a synthetic label**:

```python
legacy_notebook_label(raw) = clip(raw_stockfish_value, -1500, +1500)   # eval_type ignored
```

This reproduces the pre-C6 notebook policy exactly: mate scores stay raw mate
distances (mate-in-1 → `1`, checkmate → `0`) and labels stay **side-to-move
relative**. 494 of 9,667 labels differ from the C6-Prep labels.

> **A0 deliberately does NOT use A1's mate correction or A2's perspective
> normalisation.** A test asserts this.

**Critical limitation, preserved from the seed-0 report:**

| | |
|---|---|
| A0 **is** | the pre-C6 label *policy* applied to `dataset_v1`'s reproducible raw values |
| A0 **is not** | the historical labels that trained `models/cnn_model.keras` |

The historical labels are unrecoverable — unknown Stockfish version, shared
transposition table (order-dependent), never persisted. **A0 is a control for the
later arms, not a reproduction of the shipped model.** This is the single most
important framing point in this report.

---

## 5. Split design

Records are sorted by FEN (canonical, order-independent), then permuted with a
**fixed split seed of 42, deliberately separate from the training seed**. Verified:
all three seeds produced a byte-identical split (7,734 / 1,933). Seed-to-seed
movement is therefore training variance, not data variance.

---

## 6. Model results — seeds 0 / 1 / 2

Test split, n = 1,933 for every seed.

| Metric | seed 0 | seed 1 | seed 2 | mean | sd | min | max | **range** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Huber test loss (δ=1) | 165.138 | 167.027 | 161.819 | 164.66 | 2.64 | 161.82 | 167.03 | **5.21** |
| RMSE (cp) | 245.36 | 244.19 | 240.82 | 243.45 | 2.36 | 240.82 | 245.36 | **4.54** |
| MAE (cp) | 165.64 | 167.53 | 162.32 | 165.16 | 2.64 | 162.32 | 167.53 | **5.21** |
| Pearson r | 0.5090 | 0.5204 | 0.5304 | 0.5199 | 0.0107 | 0.5090 | 0.5304 | **0.0215** |
| Best epoch | 7 | 8 | 12 | 9.0 | 2.6 | 7 | 12 | **5** |
| Epochs run | 17 | 18 | 22 | 19.0 | 2.6 | 17 | 22 | **5** |
| Invalid predictions | 0 | 0 | 0 | — | — | — | — | **0** |

> `huber_loss_delta1` is the objective actually trained; RMSE/MAE are separate,
> in centipawns. The original notebook printed the Huber value labelled "Test MSE".

> **These are model-level metrics on a held-out slice of the TRAINING dataset.**
> They say nothing about chess playing strength. Do not compare the ~0.52 Pearson
> to the notebook's recorded 0.708 — different labels, split and dataset.

---

## 7. Engine results — seeds 0 / 1 / 2

Unmodified Phase 3 evaluator, same suites, same Stockfish configuration, same Ridge
weights. The `shipped` column is the post-C2 production model, included **as a
reference point, not as a fourth seed**.

### 7.1 `extended` (n=160) — primary

| Metric | seed 0 | seed 1 | seed 2 | shipped | mean | sd | min | max | **range** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legality % | 100 | 100 | 100 | 100 | 100.00 | 0.00 | 100 | 100 | **0.00** |
| Top-1 agreement % | 19.38 | 20.62 | 16.88 | 18.12 | 18.96 | 1.91 | 16.88 | 20.62 | **3.74** |
| Top-3 containment % | 33.75 | 33.12 | 33.75 | 33.75 | 33.54 | 0.36 | 33.12 | 33.75 | **0.63** |
| Mean regret cp | 128.73 | 140.71 | 148.43 | 144.07 | 139.29 | 9.93 | 128.73 | 148.43 | **19.70** |
| Median regret cp | 30 | 27 | 37 | 25 | 31.33 | 5.13 | 27 | 37 | **10.00** |
| p95 regret cp | 530 | 573 | 594 | 533 | 565.67 | 32.62 | 530 | 594 | **64.00** |
| Blunder rate >300cp | 0.2105 | 0.2288 | 0.2500 | 0.2434 | 0.2298 | 0.0198 | 0.2105 | 0.2500 | **0.0395** |
| Regret coverage % | 95.00 | 95.62 | 95.00 | 95.00 | 95.21 | 0.36 | 95.00 | 95.62 | **0.62** |
| Spearman mean | 0.21 | 0.21 | 0.22 | 0.22 | 0.2133 | 0.0058 | 0.21 | 0.22 | **0.0100** |
| Spearman median | 0.25 | 0.29 | 0.25 | 0.30 | 0.2633 | 0.0231 | 0.25 | 0.29 | **0.0400** |

**Mate statuses**

| | `none` | `missed_forced_mate` | `engine_move_allows_forced_mate` | `both_forced_mate_for_mover` |
|---|---:|---:|---:|---:|
| seed 0 | 152 | 4 | 4 | — |
| seed 1 | 153 | 4 | 3 | — |
| seed 2 | 152 | 3 | 4 | **1** |
| shipped | 152 | 4 | 4 | — |

**By side to move**

| Side | seed 0 | seed 1 | seed 2 | mean | sd | **range** |
|---|---:|---:|---:|---:|---:|---:|
| White (n=85) top-1 % | 22.35 | 22.35 | 17.65 | 20.78 | 2.71 | **4.70** |
| White mean regret cp | 93.81 | 102.32 | 124.34 | 106.82 | 15.76 | **30.53** |
| Black (n=75) top-1 % | 16.00 | 18.67 | 16.00 | 16.89 | 1.54 | **2.67** |
| Black mean regret cp | 167.53 | 183.89 | 175.19 | 175.54 | 8.19 | **16.36** |

### 7.2 `phase0_52` (n=52) — secondary

> Contains **6 positions (11.5%) that appear verbatim in the training set**
> (audit §N). Reported for continuity only; `extended` is primary.

| Metric | seed 0 | seed 1 | seed 2 | shipped | mean | sd | min | max | **range** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legality % | 100 | 100 | 100 | 100 | 100.00 | 0.00 | 100 | 100 | **0.00** |
| Top-1 agreement % | 21.15 | 25.00 | 19.23 | 19.23 | 21.79 | 2.94 | 19.23 | 25.00 | **5.77** |
| Top-3 containment % | 42.31 | 38.46 | 40.38 | 32.69 | 40.38 | 1.93 | 38.46 | 42.31 | **3.85** |
| Mean regret cp | 91.18 | 122.83 | 98.93 | 107.51 | 104.31 | 16.50 | 91.18 | 122.83 | **31.65** |
| Median regret cp | 21 | 24.5 | 17 | 45 | 20.83 | 3.75 | 17 | 24.5 | **7.50** |
| p95 regret cp | 471 | 524 | 524 | 419 | 506.33 | 30.60 | 471 | 524 | **53.00** |
| Blunder rate >300cp | 0.1333 | 0.1957 | 0.1333 | 0.1556 | 0.1541 | 0.0360 | 0.1333 | 0.1957 | **0.0624** |
| Regret coverage % | 86.54 | 88.46 | 86.54 | 86.54 | 87.18 | 1.11 | 86.54 | 88.46 | **1.92** |
| Spearman mean | 0.26 | 0.24 | 0.20 | 0.24 | 0.2333 | 0.0306 | 0.20 | 0.26 | **0.0600** |
| Spearman median | 0.30 | 0.29 | 0.20 | 0.37 | 0.2633 | 0.0551 | 0.20 | 0.30 | **0.1000** |

**Mate statuses**

| | `none` | `missed_forced_mate` | `engine_move_allows_forced_mate` | `both_forced_mate_for_mover` |
|---|---:|---:|---:|---:|
| seed 0 | 45 | 4 | 3 | — |
| seed 1 | 46 | 4 | 2 | — |
| seed 2 | 45 | 3 | 3 | **1** |
| shipped | 45 | 4 | 3 | — |

**By side to move**

| Side | seed 0 | seed 1 | seed 2 | mean | sd | **range** |
|---|---:|---:|---:|---:|---:|---:|
| White (n=30) top-1 % | 23.33 | 23.33 | 20.00 | 22.22 | 1.92 | **3.33** |
| White mean regret cp | 103.46 | 133.04 | 103.50 | 113.33 | 17.07 | **29.58** |
| Black (n=22) top-1 % | 18.18 | 27.27 | 18.18 | 21.21 | 5.25 | **9.09** |
| Black mean regret cp | 74.37 | 108.32 | 92.68 | 91.79 | 16.99 | **33.95** |

---

## 8. Aggregate seed statistics

See the `mean | sd | min | max | range` columns in §6 and §7. All spreads are over
n=3 seeds. **No significance testing has been performed and none is appropriate
from three runs** — these are descriptive ranges only.

---

## 9. Observed noise range

This is the deliverable of A0. Stated factually:

**On `extended` (primary, n=160), across the three A0 seeds:**

- Legality was **100% in all three seeds** (range 0).
- Top-1 agreement ranged from **16.88% to 20.62%** (3.74 pp).
- Top-3 containment ranged from **33.12% to 33.75%** (0.63 pp).
- Mean regret ranged from **128.73 to 148.43 cp** (**19.70 cp**).
- Median regret ranged from **27 to 37 cp** (10 cp).
- p95 regret ranged from **530 to 594 cp** (64 cp).
- Blunder rate >300cp ranged from **21.05% to 25.00%** (3.95 pp).
- Spearman mean ranged from **0.21 to 0.22** (0.01).
- `missed_forced_mate` ranged from **3 to 4**; `engine_move_allows_forced_mate`
  from **3 to 4**.

**On `phase0_52` (n=52) the spreads are wider**, as expected from the smaller
sample: top-1 **19.23–25.00%** (5.77 pp), mean regret **91.18–122.83 cp**
(31.65 cp), blunder rate **13.33–19.57%** (6.24 pp).

### Where the shipped model sits relative to the A0 range

Reference only — the shipped model is not a seed, and it was trained on different
(unrecoverable) labels.

| Suite | Metric | A0 range | Shipped | Position |
|---|---|---|---:|---|
| extended | top-1 agreement | 16.88 – 20.62 % | 18.12 | inside |
| extended | top-3 containment | 33.12 – 33.75 % | 33.75 | inside |
| extended | mean regret | 128.73 – 148.43 cp | 144.07 | inside |
| extended | median regret | 27 – 37 cp | 25 | **outside (below)** |
| extended | p95 regret | 530 – 594 cp | 533 | inside |
| extended | blunder rate | 0.2105 – 0.2500 | 0.2434 | inside |
| extended | spearman mean | 0.21 – 0.22 | 0.22 | inside |
| phase0_52 | top-1 agreement | 19.23 – 25.00 % | 19.23 | inside (at boundary) |
| phase0_52 | top-3 containment | 38.46 – 42.31 % | 32.69 | **outside (below)** |
| phase0_52 | mean regret | 91.18 – 122.83 cp | 107.51 | inside |
| phase0_52 | median regret | 17 – 24.5 cp | 45 | **outside (above)** |
| phase0_52 | p95 regret | 471 – 524 cp | 419 | **outside (below)** |
| phase0_52 | blunder rate | 0.1333 – 0.1957 | 0.1556 | inside |
| phase0_52 | spearman mean | 0.20 – 0.26 | 0.24 | inside |

**On the primary suite, 6 of 7 headline metrics place the shipped model inside the
A0 seed range.**

> This directly retires the tentative seed-0 observations from the earlier version
> of this report. Seed 0's `extended` mean regret of 128.73 cp looked lower than the
> shipped model's 144.07 cp, but seed 2 produced 148.43 cp under identical
> conditions. **That apparent difference was within seed noise.** Reporting it as an
> improvement would have been wrong.

---

## 10. Reproducibility evidence

**Seed 0 was independently rerun during the seed-0 phase and was bit-identical**
at the weights and prediction level: `model.weights.h5` identical, 1,933 test
predictions identical (max abs diff **0**), metrics and full history identical.
Per instruction, seed 0 was **not** rerun again here.

The `.keras` **container** is not byte-stable — it embeds `date_saved` and
`config.json` `shared_object_id` values (Python `id()` addresses). The harness
therefore records `model_weights_sha256` as the meaningful check.

| Seed | Weights sha256 | Source |
|---|---|---|
| 0 | `be4d997dab427d26…` | computed from the saved artifact¹ |
| 1 | `1b56088b0ad3c24c…` | recorded in metadata |
| 2 | `b1a5319ddaa7c34d…` | recorded in metadata |

All three distinct, confirming the seed varied what it was meant to vary.

¹ Seed 0's metadata predates the `model_weights_sha256` field, which was added
during the seed-0 reproducibility investigation. Its hash was computed from the
stored `.keras` artifact rather than by rerunning. Seeds 1 and 2 record it natively.

`tf.config.experimental.enable_op_determinism()` succeeded on all three runs, with
`random`, `numpy`, `tf.random`, `keras.utils.set_random_seed` and `PYTHONHASHSEED`
all seeded. **No TensorFlow nondeterminism was observed** for this workload on this
machine — a statement about this CPU and TF build, not a general guarantee.

---

## 11. Production-safety verification

Checked before each run and asserted after each evaluation:

| Check | Result |
|---|---|
| Working tree clean (experiment outputs ignored) | ✓ |
| Dataset checksum vs committed manifest | ✓ all three runs |
| Label policy = `legacy_notebook` | ✓ all three runs |
| Split identical to seed 0 | ✓ all three runs |
| `models/` untouched | ✓ asserted by `evaluate_arm.py` after every suite |
| Phase 3 evaluator unchanged | ✓ `git diff` empty over `evaluation/` |
| `models/cnn_model.keras` sha256 | `972d81199a1355667f…` unchanged |
| `models/weight_model.pkl` sha256 | `59a731127cc9b440c8…` unchanged |

The experimental model is injected by pointing `CHESS_BOT_MODELS_DIR` at a temporary
staging directory holding the experimental CNN plus a **copy** of the production
Ridge weights. `config.py` already supported that variable, so no production code
needed changing.

**Full test suite: 390 passed, 10 xfailed, 0 failures** — matching the expected
baseline exactly.

---

## 12. Limitations

1. **A0 does not reproduce the historical labels or the shipped model.** It cannot:
   the original Stockfish version is unrecorded, its labels were order-dependent,
   and they were never persisted.
2. **Three seeds give a range, not a distribution.** No significance testing has
   been done and none is warranted. The sd values in §6–§7 are descriptive over
   n=3 and should not be used as if they were a population estimate.
3. **The Ridge fusion weights were not refitted** — all arms use the production
   `weight_model.pkl` so engine configuration is identical across arms. Those
   coefficients were fitted against the *original* CNN's output scale, so every
   retrained model is fused with weights tuned for a different network. This biases
   all arms in the same direction (fair for comparison) but means no arm should be
   read as "the best this architecture can do".
4. **Model-level test metrics are not chess evidence.** The 1,933-position test
   split measures label fit on the training distribution.
5. **`phase0_52` has 11.5% train/eval overlap** (6 of 52 positions); `extended` has
   none and is primary.
6. **Dataset distribution is unchanged from the audit** — 9,667 positions, ~91.5%
   at ply 20, heavily opening-weighted and heavily White-to-move. Every arm
   inherits this.
7. **Early stopping fired at epochs 17–22 (best 7–12)**, far earlier than the
   notebook's recorded 55/45. Not investigated; identical callbacks across all
   arms, so it affects them equally.
8. **Seed 2 produced a mate status the other seeds did not** — one
   `both_forced_mate_for_mover` on each suite. Mate classification is itself
   seed-sensitive at these counts (3–4 per category), so mate-status differences of
   ±1 between arms will not be interpretable.

---

## 13. Factual conclusion

- The A0 harness ran end to end for all three seeds. **All three trained
  successfully with 0 invalid predictions.**
- Configuration was verified identical across seeds on every axis except the
  training seed; **model weights hashes are distinct**, confirming the intended
  variable was the only one that moved.
- **Legality was 100% on both suites for all three seeds** — the only engine metric
  with zero observed spread.
- **An empirical noise band now exists.** On the primary `extended` suite, seed
  alone moves mean regret by up to **19.70 cp**, top-1 agreement by **3.74 pp**, and
  the blunder rate by **3.95 pp**.
- **On `extended`, 6 of 7 headline metrics place the shipped model inside the A0
  seed range.**
- Training is reproducible at the weights level; the `.keras` container is not
  byte-stable, and the harness records a weights-level hash instead.
- Production files, models and historical artifacts are unmodified; the full test
  suite matches its expected baseline.

**No seed is described as better or worse than another, and no claim is made that
A0 improved on the shipped model.** The purpose of A0 was to establish the observed
variation, and that is what the numbers above report.

---

## Reproducing

```bash
python -m training.train        --arm A0 --seed {0,1,2}
python -m training.evaluate_arm --arm A0 --seed {0,1,2}
```

Outputs land in `training/experiments/A0/seed_<n>/` (gitignored):
`models/cnn_model.keras`, `metadata.json`, `test_predictions.json`,
`evaluation/{extended,phase0_52}.{json,md}`, `evaluation/summary.json`.
