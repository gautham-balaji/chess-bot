# C6-A0 — Controlled Retraining / Noise-Floor Experiment

**Status: seed 0 complete. Seeds 1 and 2 not yet run, pending review.**

No production file was modified. `engine.py`, `app.py`, `config.py`, `evaluation/`,
`baseline/`, `regression/` and `models/` are all byte-identical to `83b4767`.

---

## Objective

A0 measures **how much the results of this architecture move from the training seed
alone**, holding dataset, labels, split, architecture and hyperparameters fixed.

It is the reference against which A1/A2/A3 must be read. Phases 4A and 4B both
showed a real correctness fix producing *exactly zero* measurable change; without a
noise floor, a later arm's difference cannot be distinguished from run-to-run
variance.

**A0 is not an attempt to improve anything, and one seed cannot establish a noise
floor.** See [Conclusion](#conclusion).

---

## Experimental configuration

| Component | Value |
|---|---|
| Dataset | `training/artifacts/dataset_v1.jsonl`, 9,667 records |
| Dataset sha256 | `5a689e3f37156a05…` — **verified against the committed manifest** |
| Source sha256 | `e7aadff104a610af…` (`games.csv`, 20,058 rows) |
| Representation | `planes12` — **unchanged 12 planes**, no C6 additions |
| Label policy | `legacy_notebook` (the A0 control — see below) |
| Split | 7,734 train / 1,933 test (80/20), split seed **42** |
| Architecture | Conv 64→BN→Conv 128→BN→Conv 128→BN→Flatten→Dense 256→Drop 0.3→Dense 128→Drop 0.2→Dense 1 |
| Parameters | **2,360,129** — matches the production model exactly |
| Loss / optimizer | Huber / Adam, lr 1e-3 |
| Batch / epoch cap | 64 / 100 |
| ReduceLROnPlateau | factor 0.5, patience 5, monitor `val_loss` |
| EarlyStopping | patience 10, `restore_best_weights=True` |
| Validation | `validation_split=0.1` inside `fit` |

Architecture and hyperparameters were **verified against `models/cnn_model.keras`'s
own `config.json`**, not taken from documentation. A test pins the parameter count
and layer sequence.

### The A0 label policy — and exactly what it is not

`dataset_v1` retains, per position, the **raw pre-policy Stockfish value** and the
`eval_type`, alongside the C6-Prep repaired `label`. The A0 control is therefore a
**derivation**, not a reconstruction and not a synthetic label:

```python
legacy_notebook_label(raw) = clip(raw_stockfish_value, -1500, +1500)   # eval_type ignored
```

This reproduces the pre-C6 notebook policy exactly: mate scores stay raw mate
distances (mate-in-1 → `1`, checkmate → `0`) and labels stay **side-to-move
relative**. 494 of 9,667 labels differ from the C6-Prep labels.

> **A0 deliberately does NOT use A1's mate correction or A2's perspective
> normalisation.** `tests/unit/test_training_arm_dataset.py` asserts this directly.

**Critical limitation, stated plainly:**

| | |
|---|---|
| A0 **is** | the pre-C6 label *policy* applied to `dataset_v1`'s reproducible raw values |
| A0 **is not** | the historical labels that trained `models/cnn_model.keras` |

The historical labels are unrecoverable — unknown Stockfish version, shared
transposition table (so order-dependent), never persisted. Nothing here attempts to
reconstruct them. **A0 is therefore a control for the later arms, not a
reproduction of the shipped model.**

### Split design

Records are sorted by FEN (canonical, order-independent), then permuted with a
**fixed split seed of 42 that is deliberately separate from the training seed**. The
test set is byte-identical across every seed and arm, so seed-to-seed spread is
training variance rather than data variance. Tests assert the split is unchanged
across training seeds and invariant to record ordering.

---

## Model results — seed 0

| Metric | Train (7,734) | **Test (1,933)** |
|---|---:|---:|
| Huber loss (δ=1, the trained objective) | 115.65 | **165.14** |
| RMSE (centipawns) | 184.86 | **245.36** |
| MAE (centipawns) | 116.15 | **165.64** |
| Pearson r | 0.770 | **0.509** |
| Predictions / invalid | 7,734 / 0 | **1,933 / 0** |

Training: 17 epochs run, **best epoch 7**, early-stopped. 123.5 s.
Final train loss 63.39, final val loss 165.60, best val loss 156.53.

> **Metric naming.** `huber_loss_delta1` is the objective actually trained; RMSE/MAE
> are reported separately in centipawns. The original notebook printed the Huber
> value labelled "Test MSE", which it is not. This harness never conflates them.

> **These are model-level metrics on a held-out slice of the TRAINING dataset.**
> They say nothing about chess playing strength. The only engine-level evidence is
> the Phase 3 evaluator output below.

**Do not compare the 0.509 Pearson to the notebook's recorded 0.708.** Different
labels (legacy policy on new raw values vs historical), a different split, and a
deduplicated dataset. The two numbers are not measuring the same thing.

---

## Engine results — seed 0

Produced by the **unmodified** Phase 3 evaluator (`evaluation/evaluate.py`), same
position files, same Stockfish configuration (17.1, depth 8, Threads=1, Hash=16,
Clear Hash per position), same engine code, same Ridge fusion weights.

### `extended` (n=160) — primary

| Metric | Shipped (post-C2) | A0 seed 0 |
|---|---:|---:|
| Legality | 100% | **100%** |
| Top-1 agreement | 18.12% (29/160) | 19.38% (31/160) |
| Top-3 containment | 33.75% | 33.75% |
| Mean regret | 144.07 cp | 128.73 cp |
| Median regret | 25.0 cp | 30.0 cp |
| p95 regret | 533 cp | 530 cp |
| Blunder rate >300cp | 24.34% (37/152) | 21.05% (32/152) |
| Regret coverage | 95.0% | 95.0% |
| Spearman mean / median | 0.22 / 0.30 | 0.21 / 0.25 |
| `missed_forced_mate` | 4 | **4** |
| `engine_move_allows_forced_mate` | 4 | **4** |

By side: White n=85 top-1 22.35%, mean regret 93.81, median 20.5 · Black n=75 top-1
16.00%, mean regret 167.53, median 58.5.

### `phase0_52` (n=52) — secondary

| Metric | Shipped (post-C2) | A0 seed 0 |
|---|---:|---:|
| Legality | 100% | **100%** |
| Top-1 agreement | 19.23% (10/52) | 21.15% (11/52) |
| Top-3 containment | 32.69% | 42.31% |
| Mean regret | 107.51 cp | 91.18 cp |
| Median regret | 45 cp | 21 cp |
| p95 regret | 419 cp | 471 cp |
| Blunder rate >300cp | 15.56% (7/45) | 13.33% (6/45) |
| Spearman mean / median | 0.24 / 0.37 | 0.26 / 0.30 |
| Mate statuses | 4 / 3 | **4 / 3** |

By side: White n=30 top-1 23.33%, mean regret 103.46 · Black n=22 top-1 18.18%,
mean regret 74.37.

> `phase0_52` contains **6 positions (11.5%) that appear verbatim in the training
> set** (audit §N). It is reported for continuity only; `extended` (0 overlap) is
> primary.

**These differences must not be read as improvement.** They are a single seed
against a single other model, with no variance estimate on either side. Several
metrics move in opposite directions (mean regret down but median regret up on
`extended`; p95 up on `phase0_52`), which is what noise looks like. Establishing
whether any of this is signal is precisely what seeds 1 and 2 are for.

---

## Reproducibility

Seed 0 was run **twice** with identical configuration.

| Artifact | Result |
|---|---|
| `model.weights.h5` (inside the `.keras`) | **bit-identical** |
| Test predictions (1,933 values) | **identical**, max abs diff **0** |
| Test metrics | **identical** |
| Full training history | **identical** |
| `test_predictions.json` sha256 | **identical** |
| `.keras` container sha256 | **differs** |

**The `.keras` container is not byte-stable, but the model is.** Diagnosed exactly:
the archive embeds `metadata.json.date_saved` (a wall-clock timestamp) and
`config.json` `shared_object_id` values, which are Python `id()` addresses and
therefore differ per process. `model.weights.h5` was bit-identical.

The harness now records **`model_weights_sha256`** alongside the container hash, and
documents that the weights hash is the meaningful reproducibility check. For seed 0:
`be4d997dab427d26…` in both runs.

`tf.config.experimental.enable_op_determinism()` was requested and **succeeded**
(recorded in metadata as `op_determinism_enabled: true`), together with seeding of
`random`, `numpy`, `tf.random`, `keras.utils.set_random_seed` and `PYTHONHASHSEED`.

**No TensorFlow nondeterminism was observed on this machine for this workload.**
That is a statement about this CPU, this TF build and this model — not a general
guarantee.

---

## Limitations

1. **A0 does not reproduce the historical labels or the shipped model.** It cannot:
   the original Stockfish version is unrecorded, its labels were order-dependent,
   and they were never persisted. A0 is a control for A1–A3, nothing more.
2. **One seed is not a noise floor.** Every engine-level difference above is n=1 vs
   n=1. No variance, no interval, no significance. Seeds 1 and 2 are required
   before any of it can be interpreted.
3. **The Ridge fusion weights were not refitted.** All arms use the production
   `weight_model.pkl` so that "engine configuration" is identical across arms. But
   those coefficients (CNN term weighted 330.9) were fitted against the *original*
   CNN's output scale, so every retrained model is being fused with weights tuned
   for a different network. This biases all arms in the same direction — fair for
   comparison, but it means no arm should be read as "the best this architecture
   can do".
4. **Model-level test metrics are not chess evidence.** The 1,933-position test
   split measures label fit on the training distribution (positions at ≤20 plies
   from one game collection). Chess claims come only from the Phase 3 evaluator.
5. **`phase0_52` has 11.5% train/eval overlap** (6 of 52 positions). `extended` has
   none and is primary.
6. **Dataset distribution is unchanged from the audit:** 9,667 positions, ~91.5% at
   ply 20, 9,554/446 White/Black to move before dedup — heavily opening-weighted
   and heavily White-to-move. A0 inherits that; so will every other arm.
7. **Early stopping fired at epoch 17 (best 7)**, far earlier than the notebook's
   recorded 55/45. Not investigated; the arms are compared under identical
   callbacks, so this affects all of them equally.

---

## Conclusion

**Factual statements only:**

- The A0 harness runs end to end, and seed 0 trained successfully.
- Training is **reproducible at the weights level**: two runs produced bit-identical
  weights and identical predictions and metrics.
- The A0 control is **exactly and non-synthetically defined** from `dataset_v1`'s
  retained raw values, and is verifiably distinct from the A1/A2 policies.
- The trained model integrates with the engine and returns **100% legal moves** on
  both suites via the unmodified Phase 3 evaluator.
- Production files, models and historical artifacts are **unmodified**.

**No claim of improvement is made.** Seed 0's engine metrics differ from the shipped
model in both directions depending on the metric, and a single seed provides no
basis for separating signal from training variance. That separation is the entire
purpose of running seeds 1 and 2.

---

## Reproducing this run

```bash
# train (~2 min)
python -m training.train --arm A0 --seed 0

# engine-level evaluation via the unmodified Phase 3 evaluator (~11 min)
python -m training.evaluate_arm --arm A0 --seed 0
```

Outputs land in `training/experiments/A0/seed_0/` (gitignored):
`models/cnn_model.keras`, `metadata.json`, `test_predictions.json`,
`evaluation/{extended,phase0_52}.{json,md}`, `evaluation/summary.json`.

The model is injected into the engine by setting `CHESS_BOT_MODELS_DIR` to a
temporary staging directory holding the experimental CNN plus a copy of the
production Ridge weights. `config.py` already supports that variable, so **no
production code needed changing**, and `evaluate_arm.py` asserts afterwards that
`models/` was not modified.
