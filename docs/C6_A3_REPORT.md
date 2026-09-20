# C6-A3 — Representation Completion (18 planes)

**Status:** complete, 3 seeds, both evaluation suites.
**Scope:** one experimental variable relative to A2. No production code changed.
**Fusion:** the frozen **production** Ridge for every arm. No matched-Ridge run.
**Statistical standing:** descriptive. Three seeds, no significance testing. A
change counts only when it is directionally consistent across all three paired
seeds *and* exceeds the A0 seed-noise band.

---

## Result up front

> **A3 is the worst arm in the programme, and it is the first arm whose
> degradation clearly exceeds the noise band.**
>
> Completing the board representation — adding side-to-move, castling rights and
> en-passant exactly as the C6 audit's §K specified — made the model worse
> offline (Pearson 0.631–0.656 vs A2's 0.733–0.745, no overlap) and worse at the
> engine level (mean regret **+46.5 cp** on `extended`, all three paired seeds,
> against an A0 noise range of 19.7 cp).
>
> **A separate, independent finding: an 18-plane model cannot be loaded by the
> shipped engine at all.** Even had A3 won, it would not have been deployable
> without a production change.

---

## 1. Objective

Test whether extending the board encoding from 12 to 18 planes changes model
quality or engine behaviour, relative to A2.

The C6 training-pipeline audit called this "the most important design finding"
(§L) and specified the layout in §K. A0, A1 and A2 all left the representation
alone; three xfail tests — `test_side_to_move_should_be_representable`,
`test_castling_rights_should_be_representable`,
`test_en_passant_should_be_representable` — have documented the gap since
Phase 2. A3 closes it and measures the result.

---

## 2. Experimental delta

Exactly one variable moves relative to A2: **the board encoding.**

| | A2 | **A3** |
|---|---|---|
| label policy | `corrected_mate_white_perspective` | **identical** |
| labels | — | **byte-identical, 0 of 9,667 differ** |
| split | 7,734 / 1,933, seed 42 | **identical** |
| **representation** | planes12 `(8,8,12)` | **planes18 `(8,8,18)`** |
| parameters | 2,360,129 | **2,363,585 (+3,456)** |

The parameter delta is exactly the audit's prediction: only the first `Conv2D`
kernel changes, `3×3×12×64 → 3×3×18×64`, i.e. `3·3·6·64 = 3,456` (+0.15%).
A test asserts that number.

### The six appended planes (audit §K, implemented verbatim)

| plane | meaning | values | spatial? |
|---:|---|---|---|
| 0–11 | the existing 12-plane piece encoding, **unchanged** | 0/1 | yes |
| 12 | side to move is White | all-1 or all-0 | constant |
| 13–16 | White K / White Q / Black K / Black Q castling rights | constant 0/1 | constant |
| 17 | en-passant target square | single cell 1.0 | yes |

Planes are **appended, never reordered**. A drift-guard test asserts planes 0–11
equal `training.representation.board_to_planes` on all 212 positions of both
evaluation suites, and equal `engine.board_to_planes` directly.

### How informative are the new planes on this dataset?

| field | coverage in `dataset_v1` (n=9,667) |
|---|---|
| side to move = White | 9,260 (**95.79%** — near-constant) |
| any castling right | 7,180 (74.3%); individual rights 50.7–54.7% |
| en-passant available | **26 (0.27%)** — plane 17 is all-zero otherwise |

Castling is genuinely informative. Side-to-move is 96/4 skewed. En passant is
very nearly a dead channel. §12 shows this matters.

---

## 3. Fixed configuration

Everything else held identical to A0/A1/A2:

| | value |
|---|---|
| dataset | `dataset_v1.jsonl`, 9,667 deduplicated records |
| split | sorted by FEN, seeded permutation, split seed 42 |
| loss / optimiser | Huber, Adam, LR 1e-3, batch 64 |
| epochs | max 100, same callbacks |
| seeds | 0, 1, 2 |
| fusion | frozen **production** Ridge |
| evaluator | `evaluation/evaluate.py`, unmodified |
| Stockfish | depth 8, Threads=1, Hash=16, `Clear Hash` per position |
| suites | `extended` (n=160), `phase0_52` (n=52) |

`test_a3_differs_from_a2_in_representation_only` asserts the two arm specs differ
in the `representation` key and nothing else.

---

## 4. An 18-plane model cannot be loaded by the shipped engine

This blocked the evaluation half of the task and is a finding in its own right.

`engine.py:31` hardcodes the channel count:

```python
def board_to_planes(board):
    planes = np.zeros((8, 8, 12), dtype=np.float32)
```

Running the unmodified Phase 3 evaluator against an A3 model fails immediately:

```
ValueError: Input 0 with name 'None' of layer 'conv2d' is incompatible with the
layer: expected axis -1 of input shape to have value 18, but received input with
shape (32, 8, 8, 12)
```

The audit predicted the converse ("an 18-channel model cannot load 12-channel
weights"); the operative direction is that the **engine's encoder**, not just the
weights file, is fixed at 12. The extra state is not recoverable from the 12
piece planes — that is precisely why it was missing — so no wrapper can bridge
it.

**How A3 was evaluated without touching production.**
`training/evaluate18_runner.py` imports `engine`, rebinds the single name
`engine.board_to_planes` to the 18-plane encoder in-process, and then calls
`evaluation.evaluate.main()`. No file on disk changes; the rebinding dies with
the process. All three of the engine's call sites (`cnn_evaluate`,
`rerank_moves`, and the 1-ply lookahead) resolve the name through the module
global, so one rebinding covers every path — asserted behaviourally by
`test_engine_resolves_board_to_planes_through_the_module_global`, which drives
the real engine through a counting wrapper.

Everything else is the committed evaluator: same heuristics, same Ridge, same
C1/C2 fixes, same lookahead, same Stockfish configuration. **The only difference
between an A3 run and an A2 run is the board encoding, which is A3's variable.**
The runner also asserts the staged model's input shape before starting, so a
mis-staged model fails in one line rather than 100 positions in.

`training/evaluate_arm.py` dispatches on the arm's representation and defaults to
`planes12`, so the A0/A1/A2 path is byte-identical to before; a test pins that.

**Deployability, stated plainly:** shipping an 18-plane CNN requires editing
`engine.board_to_planes`. That is a production change, out of scope here, and it
would invalidate `models/cnn_model.keras`.

---

## 5. Model results — A3 seeds 0 / 1 / 2

| run | planes | params | epochs | best epoch | best val loss | MAE cp | RMSE cp | **Pearson r** | secs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 s0/s1/s2 | 12 | 2,360,129 | 17/18/22 | 7/8/12 | 156.5/160.7/154.0 | 165.6/167.5/162.3 | 245/244/241 | 0.5090/0.5204/0.5304 | 124/136/161 |
| A1 s0/s1/s2 | 12 | 2,360,129 | 23/35/41 | 13/25/31 | 180.6/180.8/182.5 | 185.9/190.3/190.8 | 313/314/317 | 0.6055/0.6040/0.5919 | 160/227/297 |
| A2 s0/s1/s2 | 12 | 2,360,129 | 25/58/46 | 15/48/36 | 155.3/157.3/154.8 | 160.7/163.0/165.0 | 258/262/265 | **0.7450/0.7360/0.7332** | 185/401/329 |
| **A3 s0** | 18 | 2,363,585 | 20 | 10 | 180.25 | 186.48 | 292.41 | **0.6561** | 130 |
| **A3 s1** | 18 | 2,363,585 | 60 | 50 | 176.88 | 186.53 | 297.13 | **0.6433** | 381 |
| **A3 s2** | 18 | 2,363,585 | 43 | 33 | 180.27 | 188.48 | 299.85 | **0.6310** | 250 |

A2 and A3 are fitted to **identical labels**, so unlike the earlier cross-arm
comparisons, MAE and RMSE *are* directly comparable here — and every one of them
is worse for A3.

```
Pearson r   A2  0.7332  0.7360  0.7450
            A3  0.6310  0.6433  0.6561     <- no overlap, -0.09 mean
```

**A3 loses about 0.09 Pearson relative to A2**, giving back roughly 40% of the
+0.22 that A2 gained over the A0 control.

### Is it underfitting or overfitting?

| run | final train loss | final val loss | gap |
|---|---:|---:|---:|
| A2 s0/s1/s2 | 48.7 / 31.1 / 33.1 | 162.1 / 160.8 / 161.4 | 113 / 130 / 128 |
| A3 s0/s1/s2 | 84.2 / 30.9 / 35.7 | 198.0 / 178.0 / 184.3 | **114 / 147 / 149** |

Mixed: seed 0 stopped early at 20 epochs and underfit (train 84.2). Seeds 1 and 2
reached A2-comparable train loss (30.9, 35.7) but a **larger generalisation gap**.
So the deficit is not an early-stopping artifact — the two long runs fit the
training data as well as A2 did and still generalised worse.

---

## 6. Engine results — A3 seeds 0 / 1 / 2

**`extended` (n=160)**

| metric | A3 s0 | A3 s1 | A3 s2 |
|---|---|---|---|
| legality % | 100 | 100 | 100 |
| top-1 agreement % | 16.25 | 12.50 | 12.50 |
| top-3 containment % | 31.88 | 31.88 | 30.62 |
| mean regret cp | 181.0 | 178.2 | 208.0 |
| median regret cp | 87 | 51 | 54 |
| p95 regret cp | 559 | 633 | 680 |
| blunder rate >300cp | 0.2810 | 0.2810 | 0.3333 |

**`phase0_52` (n=52)**

| metric | A3 s0 | A3 s1 | A3 s2 |
|---|---|---|---|
| legality % | 100 | 100 | 100 |
| top-1 agreement % | 19.23 | 21.15 | 13.46 |
| top-3 containment % | 28.85 | 34.62 | 26.92 |
| mean regret cp | 151.8 | 100.8 | 160.7 |
| median regret cp | 88 | 29 | 54 |
| blunder rate >300cp | 0.2000 | 0.1333 | 0.2222 |

Legality is 100% in all six runs, as in every previous arm.

---

## 7. A2 → A3 paired comparison, against the A0 noise band

**`extended` (n=160)**

| metric | s0 | s1 | s2 | mean Δ | A0 noise range | verdict |
|---|---|---|---|---|---|---|
| mean regret cp | +47.32 | +36.55 | +55.68 | **+46.52** | 19.70 | all 3 worse, **EXCEEDS noise** |
| p95 regret cp | +29 | +94 | +149 | **+90.67** | 64 | all 3 worse, **EXCEEDS noise** |
| blunder rate | +0.0784 | +0.0457 | +0.0849 | **+0.0697** | 0.0395 | all 3 worse, **EXCEEDS noise** |
| top-1 agreement % | −3.13 | −6.88 | −1.88 | **−3.96** | 3.74 | all 3 worse, **EXCEEDS noise** |
| median regret cp | +58 | +21 | 0 | +26.33 | 10 | exceeds noise, inconsistent |
| top-3 containment % | −1.24 | −2.50 | +1.24 | −0.83 | 0.63 | inconsistent |
| **spearman mean** | +0.01 | +0.05 | +0.05 | **+0.0367** | 0.01 | all 3 **better**, EXCEEDS noise |

**`phase0_52` (n=52)**

| metric | s0 | s1 | s2 | mean Δ | A0 noise range | verdict |
|---|---|---|---|---|---|---|
| mean regret cp | +47.51 | −26.96 | +51.34 | +23.96 | 31.65 | inconsistent |
| spearman mean | −0.12 | −0.04 | −0.05 | −0.070 | 0.06 | all 3 worse, EXCEEDS noise |
| top-1 agreement % | −3.85 | +1.92 | −1.92 | −1.28 | 5.77 | inconsistent |

**On `extended`, four separate metrics degrade consistently across all three
paired seeds and all four exceed the A0 noise band.** No previous arm produced
that. `phase0_52` (n=52) is mixed — seed 1 improved there — so the aggregate
claim rests on the larger suite.

**Spearman is the one counter-signal, and it flips between suites**: +0.037 on
`extended` (all 3 better) and −0.070 on `phase0_52` (all 3 worse). It should not
be read as a genuine A3 strength.

---

## 8. A0 → A3 paired comparison

| suite | metric | mean Δ vs A0 | A0 noise | verdict |
|---|---|---|---|---|
| extended | mean regret cp | **+49.76** | 19.70 | all 3 worse, exceeds |
| extended | median regret cp | +32.67 | 10 | all 3 worse, exceeds |
| extended | top-1 agreement % | **−5.21** | 3.74 | all 3 worse, exceeds |
| extended | top-3 containment % | −2.08 | 0.63 | all 3 worse, exceeds |
| extended | blunder rate | +0.0687 | 0.0395 | all 3 worse, exceeds |
| phase0_52 | mean regret cp | +33.46 | 31.65 | inconsistent |

**A3 is worse than the A0 control on every headline `extended` metric,
consistently across seeds.** It is the only arm in the programme of which that
is true.

---

## 9. White vs Black

Mean regret cp, averaged over three seeds:

| suite | side | n | A0 | A1 | A2 | **A3** | A2 → A3 |
|---|---|---|---|---|---|---|---|
| extended | black | 75 | 175.54 | 195.70 | **139.07** | 174.07 | +35.00 (all 3) |
| extended | white | 85 | 106.82 | 120.90 | 145.60 | **202.37** | **+56.76 (all 3)** |
| phase0_52 | black | 22 | 91.79 | 123.73 | **63.18** | 120.93 | +57.75 (mixed) |
| phase0_52 | white | 30 | 113.33 | 112.59 | 149.45 | 149.01 | −0.44 (mixed) |

On `extended` both sides degrade consistently, and **the damage is larger on
White-to-move — the 95.8%-majority case in training and 53% of the suite**. A3
also gives back essentially all of A2's Black-side gain (139.07 → 174.07, back
near A0's 175.54).

This is the opposite of what the representation was supposed to do. Side-to-move
information should, if anything, help the minority Black case.

---

## 10. Game-phase / category breakdown

Mean regret cp, A2 → A3, averaged over three seeds:

| category | n (ext) | A2→A3 (extended) | consistent | n (p52) | A2→A3 (phase0_52) | consistent |
|---|---|---|---|---|---|---|
| opening | 60 | **+66.68** | yes | 18 | +36.65 | no |
| middlegame | 64 | **+45.37** | yes | 12 | +10.72 | no |
| tactical | 10 | +43.08 | yes | 6 | −106.00 | yes |
| endgame | 21 | +1.64 | no | 12 | +39.22 | yes |
| defensive | 5 | +1.58 | no | 4 | −47.22 | yes |

**Opening and middlegame — the two largest cells — degrade consistently on
`extended`.** Openings degrading most is notable, because openings are exactly
where castling rights should carry information. The added planes hurt most where
they should have helped.

The small cells (n=4–10) flip sign between suites and are not interpretable;
`tactical` moves by 43 cp one way and 106 cp the other on 10 and 6 positions.

---

## 11. Mate behaviour

`extended`, all three A3 seeds: `{none: 153, missed_forced_mate: 4,
engine_move_allows_forced_mate: 3}` — **identical to A2 on every seed**.

`phase0_52`: A3 is `{none: 45, missed: 4, allows: 3}` in all three seeds, against
A2's `{none: 46, missed: 4, allows: 2}` — one position worse, in all three seeds.

At counts of 3–4 out of 160 this is not a result. What is worth recording: A2 and
A3 are the only arms whose mate profiles are seed-stable, and A3 inherits that.

Offline, A3 is actually **better on Black-to-move mate rows** — MAE 248–336 vs
A2's 331–552, and far more stable (§12 table). That is the one place the extra
planes measurably helped, and it did not survive into engine metrics.

---

## 12. Mechanism — the model leans hard on a feature its labels made irrelevant

Two candidate explanations for A3's deficit: the model ignores the new planes and
pays only their parameter cost, or it uses them and they mislead. Ablation
separates these. `python -m training.a3_plane_ablation` zeroes each plane group
on the 1,933 held-out positions and measures how far predictions move. Zero is a
valid in-distribution value for every added plane, so this is not an
off-manifold probe.

| plane group | mean \|prediction change\| when zeroed (3 seeds) |
|---|---:|
| **side to move (plane 12)** | **304.4 cp** |
| castling (13–16) | 92.6 cp |
| en passant (17) | **0.15 cp** |
| all six | 666.8 cp |

**The model relies enormously on plane 12** — zeroing one near-constant channel
moves predictions by 304 cp and costs 0.12–0.24 Pearson.

**Why that is a problem.** A2 and A3 share White-positive labels, so the target
is a function of the *position alone*: side-to-move is **not needed** to predict
it. A3 handed the network an irrelevant feature that splits the data 95.8/4.2,
with only 327 Black-to-move examples in the 7,734-row training set. The network
used it to special-case that thin minority instead of learning board features —
and the cost landed on the majority. A2's White-row MAE is 155.7/157.6/159.2;
A3's is 180.6/180.7/182.1.

The side-to-move split by eval type:

| arm | White MAE | Black MAE | gap | Black cp MAE | Black mate MAE |
|---|---|---|---|---|---|
| A2 | 155.7/157.6/159.2 | 276.5/286.8/297.7 | **+129.5** | 218–260 | 331–552 |
| A3 | 180.6/180.7/182.1 | 321.8/321.1/335.9 | **+145.1** | 321–344 | **248–336** |

A3's Black-minus-White gap is *wider* than A2's, and the Black signed error
returns to a systematic positive bias (+74 to +119) after A2 had driven it near
zero (−97 to +41).

**En passant is inert.** At 0.27% coverage, zeroing plane 17 moves predictions by
0.15 cp. It contributes 576 first-layer parameters and no measurable signal. The
audit called it "defensible on frequency grounds" to omit; this measures that.

### How the offline deficit becomes an engine deficit

`python -m training.a2_saturation_probe`, 5,639 candidate positions:

| arm | raw \|cnn\| | **saturated (\|tanh\|>0.95)** | within-position tanh spread, White to move |
|---|---:|---:|---:|
| A0 | 98.8 | 6.2% | 0.6458 |
| A1 | 109.0 | 6.1% | 0.7905 |
| A2 | 110.0 | **3.7%** | **0.9547** |
| **A3** | **173.7** | **13.8%** | **0.7249** |

The fusion applies `330.9 · tanh(cnn/200)`. A3's raw outputs are ~58% larger than
A2's, so **13.8% of candidates land in tanh's flat region — 3.7× A2's rate and
more than double any other arm.** Saturation compresses the differences *between
candidate moves within a position*, so the CNN term discriminates less: the
White-to-move within-position spread falls from 0.955 to 0.725.

This is the same hypothesis that was tested for A2 and **failed** there. For A3
it holds. That asymmetry is itself evidence: the probe is capable of returning a
negative, and did, one arm earlier.

Chain, end to end: irrelevant high-leverage feature → over-confident, larger-
magnitude outputs → tanh saturation → less separation between candidate moves →
worse move selection, concentrated on White-to-move.

---

## 13. Limitations

1. **A3 used A2's exact recipe, untuned for 18 channels.** That is the correct
   controlled experiment — one variable — but it means the result is "adding
   these planes under A2's hyperparameters hurts", not "these planes are
   worthless". A representation change plausibly warrants its own learning-rate
   or regularisation; that was not tested and is the single largest caveat here.
2. **`phase0_52` does not replicate the aggregate degradation** (seed 1 improved).
   The consistent result rests on `extended`, n=160.
3. **The mechanism in §12 is correlational.** The ablation and saturation
   measurements are strong and mutually consistent, but no intervention was run
   to confirm causation — e.g. retraining with plane 12 removed, which would
   isolate it.
4. **Three seeds.** Ranges are descriptive; no significance testing.
5. **Small cells.** Category breakdowns at n=4–21 and mate counts at 3–4 of 160
   are not interpretable alone.
6. **Frozen production Ridge**, as designed. A1R Stage 1 established it is
   mismatched to every retrained CNN, and A2R found A2's mismatch A0-sized. **A3
   was not put through a Ridge diagnostic**, so its fusion compatibility is
   unmeasured — and §12 gives specific reason to think A3 interacts with the
   fusion differently (13.8% saturation). No matched-Ridge run was performed.
7. **The evaluation shim**, though faithful, is a path A0/A1/A2 did not use. It
   changes exactly one binding and is test-covered, but it is not literally the
   same process invocation.
8. **The dataset's 95.8/4.2 side-to-move skew is a property of `dataset_v1`,**
   which was sampled from games where positions were recorded predominantly at
   White-to-move. A balanced dataset could plausibly reverse this result.

---

## 14. Reproducibility and production safety

**Reproducibility.** `keras.utils.set_random_seed(seed)` plus
`tf.config.experimental.enable_op_determinism()`; split seed 42 held separate
from the training seed. Per-seed bit-stable weight hashes:

```
A3 seed 0  0fec4a240a625a6060d977a7…
A3 seed 1  772b4db9aa281730e9829086…
A3 seed 2  802e8e76b18fa2b4882ff3b1…
```

**Production safety.** `engine.py`, `app.py`, `config.py`, `evaluation/`,
`models/` and the baseline/regression fixtures are unmodified — `git status`
reports no changes under any of them, all five files in `models/` retain their
original SHA-256 digests, the production Ridge still reads
`[330.9005, 32.3899, 0.7919, 5.1654, 0.0188]`, and `engine.board_to_planes` still
allocates `(8, 8, 12)`. All three A3 evaluation runs logged `verified: production
models/ untouched`, and the log shows the shim's own confirmation six times
(`patched engine.board_to_planes → … (in-process only; no file modified)`).

All A3 artifacts are under `training/experiments/A3/` (gitignored).

**Tests.** Full suite: **495 passed, 10 xfailed, 0 failed** (468 before A3). The
27 new tests cover the §K layout plane by plane, the planes-0–11 drift guard
against both the 12-plane encoder and `engine.board_to_planes`, the registry, the
parameter-count prediction, and the shim mechanism.

The three representation xfails remain xfail: A3 demonstrates the gap *can* be
closed, and `engine.board_to_planes` still does not close it.

---

## 15. Factual conclusion

1. A3 changed exactly one variable from A2: 12 → 18 planes, implemented verbatim
   from audit §K. Labels are byte-identical (0 of 9,667 differ), the split is
   identical, and planes 0–11 are byte-identical to the 12-plane encoder on all
   212 evaluation positions. Parameters rose by exactly the predicted 3,456.
2. **Offline, A3 is clearly worse than A2**: Pearson 0.631–0.656 vs 0.733–0.745,
   no overlap across three seeds. Because the labels are identical, MAE and RMSE
   are directly comparable too, and both are worse.
3. **At the engine level on `extended`, four metrics degrade consistently across
   all three paired seeds and all four exceed the A0 noise band**: mean regret
   +46.5 cp (band 19.7), p95 +90.7 (band 64), blunder rate +0.070 (band 0.040),
   top-1 −3.96 pp (band 3.74). This is the first arm in the programme to clear
   that bar — in the wrong direction.
4. **A3 is also worse than the A0 control on every headline `extended` metric**,
   consistently across seeds. No other arm is.
5. `phase0_52` does not replicate the aggregate degradation; seed 1 improved
   there. Spearman is the only counter-signal and it flips sign between suites.
6. **Mechanism, measured:** the model leans on side-to-move for 304 cp of
   prediction movement, despite White-positive labels making it irrelevant to the
   target. It thereby special-cases a 4.2% minority with 327 training examples,
   and the cost falls on the White-to-move majority (+56.8 cp regret vs Black's
   +35.0). Larger outputs then saturate the fusion's tanh at 13.8% — 3.7× A2's
   rate — compressing within-position candidate separation from 0.955 to 0.725.
   The saturation hypothesis that failed for A2 holds for A3.
7. **En passant is inert**: 0.27% dataset coverage, 0.15 cp ablation effect.
8. **Independent finding: an 18-plane CNN is not loadable by the shipped
   engine.** `engine.board_to_planes` hardcodes 12 channels and the missing state
   is not recoverable from the piece planes, so evaluation required an
   experiment-only in-process shim. Deploying any 18-plane model would require a
   production change to the engine.
9. Legality remained 100% in all six A3 runs. No production file was modified.

**In one line:** completing the representation exactly as the audit specified
made the model measurably worse, by handing it a high-leverage feature that
A2's label fix had already made irrelevant — and the resulting model could not
have been deployed to the current engine anyway.

**Recommended next step, not taken:** the cheapest discriminating follow-up is a
**planes13 arm** — the 12 piece planes plus castling rights only, dropping the
side-to-move and en-passant planes that §12 identifies as harmful and inert
respectively. That isolates the one added field with measured signal. A
hyperparameter-tuned A3 would address limitation 1 but confounds two variables at
once.

---

## Reproducing

```bash
# train
python -m training.train --arm A3 --seed 0      # and 1, 2

# evaluate (dispatches to the 18-plane runner automatically)
python -m training.evaluate_arm --arm A3 --seed 0   # and 1, 2

# analysis
python -m training.a2_analysis --arms A0 A1 A2 A3 --contrasts A2:A3 A0:A3 \
    --out training/experiments/A3/a3_analysis.json
python -m training.a2_perspective_probe --arms A0 A1 A2 A3 --contrast A2:A3 \
    --out training/experiments/A3/perspective_probe.json
python -m training.a2_saturation_probe --arms A0 A1 A2 A3 --contrast A2:A3 \
    --out training/experiments/A3/saturation_probe.json
python -m training.a3_plane_ablation
```

The analysis scripts keep their A2 defaults: run with no arguments they still
reproduce the A2 report exactly (verified — all seven of its headline numbers).

Artifacts (gitignored): `training/experiments/A3/a3_analysis.json`,
`perspective_probe.json`, `saturation_probe.json`, `plane_ablation.json`, and
per-seed `metadata.json` / `test_predictions.json` / `evaluation/*.json`.
