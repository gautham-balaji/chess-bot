# C9 — Matched Fusion / Ridge Audit (audit and design only)

**Status: audit and design only.** Nothing was refitted, trained, or modified.
No production file changed. The matched Ridge has **not** been implemented.

Prior work reused, not repeated: [A1R design](C6_A1R_RIDGE_DIAGNOSTIC_DESIGN.md) ·
[A1R Stage 1](C6_A1R_STAGE1_REPORT.md) · [A1R Stage 2](C6_A1R_STAGE2_REPORT.md) ·
[A2R Stage 1](C6_A2R_STAGE1_REPORT.md) · [C8a](C8A_REPORT.md) · [C8b](C8B_REPORT.md)

---

## Headline finding

> **The frozen Ridge is not the binding constraint. The frozen `tanh(x/200)`
> squash in front of it is — and that squash lives in `engine.py`, not in
> `weight_model.pkl`.**
>
> On the proposed fit population, C8a's CNN correlates **0.876** with the target.
> After the production squash the Ridge only ever sees a **0.704**-correlated
> feature. The squash discards **0.17 of correlation** from C8a, against **0.07**
> from A2 — it penalises the better model more than twice as hard.
>
> Worse, measured on real candidate moves: **C8a's CNN term has *less*
> within-position ranking spread than A2's (194.03 vs 288.82)** despite being the
> far better regressor. 40% of C8a's positions land in tanh's saturated region,
> where candidates become indistinguishable.
>
> **A Ridge refit rescales `w[0]`. It cannot undo saturation that happened
> upstream.** A coefficient-only C9 is therefore likely to return null — and a
> null would be misread as "fusion is fine".

The audit's recommendation is a **conditional GO** with an explicit scope
decision for you to make (§12).

---

## 1. Current production Ridge pipeline

Verified from source at `355d6da`, not from prior summaries.

```
board
  -> board_to_planes (8,8,12)
  -> cnn_model.predict            -> cnn_score      (raw centipawn-scale float)
  -> np.tanh(cnn_score / 200)     -> cnn_norm       (hardcoded divisor)
  -> w[0]*cnn_norm + w[1]*material + w[2]*space + w[3]*center + w[4]*mobility
  -> + heuristic bonuses          (UNSCALED, +/-0.20..0.65)
  -> + 0.5*tanh(opp_best / 200)   (UNSCALED 1-ply lookahead)
  -> sort  (descending for White, ascending for Black)
```

The CNN enters the final score **twice**, and the Ridge governs only the first:

| entry point | site | weight | governed by the Ridge? |
|---|---|---|---|
| candidate evaluation | `engine.py:164` | `w[0] = 330.90` | **yes** |
| 1-ply lookahead | `engine.py:240` | fixed `0.5` | **no** |

`weight_model.pkl` is loaded read-only at `engine.py:22-23`. The divisor `200`
is a **literal at three sites** (`engine.py:130`, `:164`, `:240`) — it is not a
module constant and not part of the Ridge artifact.

**Defect C3 (open, pre-existing):** `engine.py` applies `weight_model.coef_`
only. `intercept_ = 15.0772` is **never applied**, so `hybrid_score` is not the
Ridge's prediction. Pinned as an `xfail` in `tests/unit/test_model_fusion.py`.

---

## 2. Exact production features

| # | feature | definition | CNN-dependent? |
|---|---|---|---|
| 0 | `cnn_norm` | `tanh(cnn(position) / 200)` | **yes** |
| 1 | `material` | `material_balance(board)` | no |
| 2 | `space` | `space_control(board)` | no |
| 3 | `center` | `center_control(board)` | no |
| 4 | `mobility` | `mobility_score(board)` | no |

Features 1–4 are pure board functions. **Only feature 0 changes between arms** —
verified in A1R §2 and unchanged here.

---

## 3. Current Ridge coefficients

```
sklearn.linear_model.Ridge(alpha=1.0), n_features_in_ = 5, fit_intercept=True

coef_      [330.900549, 32.389872, 0.791884, 5.165416, 0.018828]
intercept_ 15.077236450243632          <-- DROPPED at runtime (defect C3)
```

| feature | coefficient | mean within-position spread on 6,045 real candidates |
|---|---:|---:|
| `cnn_norm` | **330.900549** | A2 **288.82** · C8a **194.03** |
| `material` | 32.389872 | 38.65 |
| `space` | 0.791884 | 7.01 |
| `center` | 5.165416 | 11.45 |
| `mobility` | 0.018828 | 0.42 |

**The fusion is effectively a two-term model.** `space`, `center` and `mobility`
contribute 7.01 / 11.45 / 0.42 of ranking spread against the CNN's 194–289 and
material's 38.65. The unscaled heuristic bonuses (±0.65) and lookahead (±0.5) are
three orders of magnitude below the CNN term at production scale.

---

## 4. Current Ridge training population

**The only Ridge-fitting code in the repository is `chess_model_FINAL.ipynb`
cell 31.** No module fits it; everything else loads the pickle.

```python
for board, sf_cp in zip(sample_boards, sf_evaluations):
    cnn_score = cnn_evaluate(board)
    cnn_norm  = np.tanh(cnn_score / 200)
    hybrid_features.append([cnn_norm, material_balance(board), space_control(board),
                            center_control(board), mobility_score(board)])
    sf_targets.append(sf_cp)
weight_model = Ridge(alpha=1.0).fit(hybrid_features, sf_targets)
```

| property | value |
|---|---|
| population | the original **10,000** `sample_boards` |
| target | `sf_evaluations` — the original labels: **side-to-move relative**, mate as raw distance (**A0-policy-like**) |
| CNN | the **original** shipped CNN |
| features | on the **position itself**, not on post-move candidates |
| reproducible? | **No.** The original labels were never persisted and are unrecoverable |

### Three pre-existing defects in the production Ridge

1. **Partly in-sample.** Fitted on all 10,000 positions, including the ~8,000 the
   CNN trained on. Its `cnn_norm` is optimistically accurate, inflating `w[0]`.
2. **Objective mismatch.** Fitted for *between-position* centipawn accuracy;
   applied for *within-position* move ranking. Different objectives.
3. **Target-semantics mismatch.** Fitted against A0-style side-to-move-relative
   labels. Every arm since A2 uses White-positive labels with mapped mates.

---

## 5. Leakage analysis

### The proposed fit population is strictly cleaner than A1R's

Measured, not assumed:

| property | `dataset_v1` test (A1R/A2R used) | **`dataset_v2` test (proposed)** |
|---|---:|---:|
| records | 1,933 | **13,712** |
| distinct games | — | 3,614 |
| **evaluation-suite placement overlap** | **1** | **0** |
| overlap with C8a train placements | n/a | **0** |
| overlap with C8a train games | n/a | **0** |
| out-of-sample for the arm's CNN | yes | **yes** |

`dataset_v2`'s test split already passed C8's evaluation-suite scrub, so it
contains **zero** suite placements. `dataset_v1`'s split contains one. Using
dataset_v2's test split therefore *removes* a leak that prior Ridge diagnostics
carried.

### Against the brief's prohibitions

| requirement | status |
|---|---|
| must not use `phase0_52` positions | **satisfied** — 0 placement overlap, 0 exact-FEN overlap |
| must not use `extended` positions | **satisfied** — same |
| must not use any test labels | **satisfied** — targets are dataset_v2 test-split Stockfish labels, which are *not* evaluation-suite labels and never enter the evaluator |
| must not use anything derived from the suites | **satisfied** — the suites are read only for the scrub that *excludes* them |
| deterministic | **satisfied** — fixed split seed 42, fixed game-content split, no RNG |

### Residual leakage considerations

1. **The fit population is C8a's own held-out set.** It is out-of-sample for the
   CNN, which is the point. But it is then also the set C8a's model metrics were
   reported on, so those metrics stop being independent of the fusion. This is
   cosmetic (the decision rests on engine metrics) and should be stated.
2. **1,129 of the 13,712 fit positions are checkmates with zero legal moves**
   (8.2%), carrying ±2000 targets. The engine ranks *moves*, and a mated board
   has none. Including them pulls `w[0]` toward fitting extreme values the ranking
   task never encounters. **Recommend fitting with and without them** and
   reporting both; this is a free variant at Stage 1.

---

## 6. Why C8a's output distribution differs — measured

On the proposed fit population (13,712 positions):

| model | raw \|cnn\| | raw std | %saturated (\|tanh\|>0.95) | corr(tanh(cnn/200), y) | corr(raw, y) | **lost to squash** |
|---|---:|---:|---:|---:|---:|---:|
| A2 s0 | 246.0 | 336.3 | 22.4% | 0.5232 | 0.5943 | 0.071 |
| A2 s1 | 217.4 | 299.5 | 18.3% | 0.4827 | 0.5486 | 0.066 |
| A2 s2 | 208.2 | 302.3 | 17.3% | 0.5205 | 0.5882 | 0.068 |
| **C8a s0** | 431.3 | 664.8 | **40.7%** | 0.7044 | **0.8760** | **0.172** |
| **C8a s1** | 424.2 | 650.8 | **40.0%** | 0.7033 | **0.8742** | **0.171** |
| **C8a s2** | 427.6 | 668.0 | **39.8%** | 0.7086 | **0.8732** | **0.165** |

C8a trained on labels spanning ±2000 with 10.2% mate-typed targets, so it emits a
much wider distribution than A2 — roughly double the magnitude. The fixed
`/200` divisor was chosen for a model with ~1/2 that scale.

### How much the divisor alone costs — measurement, no fitting

`corr(tanh(cnn/d), label)` as a function of the divisor `d`:

| model | d=200 *(prod)* | d=400 | d=600 | d=800 | d=1000 | d=1500 | d=2000 | raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 s0 | 0.5232 | 0.5656 | 0.5839 | 0.5923 | 0.5962 | 0.5988 | 0.5984 | 0.5943 |
| **C8a s0** | **0.7044** | 0.7641 | 0.8041 | 0.8307 | 0.8479 | 0.8679 | **0.8742** | 0.8760 |
| **C8a s1** | **0.7033** | 0.7638 | 0.8040 | 0.8304 | 0.8472 | 0.8666 | 0.8726 | 0.8742 |
| **C8a s2** | **0.7086** | 0.7674 | 0.8069 | 0.8327 | 0.8490 | 0.8673 | 0.8726 | 0.8732 |

A2 is near its ceiling at d≈800. **C8a is still climbing at d=2000.** The
production divisor costs C8a about **0.17 of correlation**; no value of `w[0]`
recovers any of it, because `w[0]` is a linear scale applied *after* the
saturating nonlinearity.

### The consequence that matters for ranking

Mean within-position spread of the CNN term on 6,045 real candidate boards:

| model | `w0*tanh(cnn/200)` spread | `w1*material` spread | ratio |
|---|---:|---:|---:|
| A2 s0 | **288.82** | 38.65 | 7.5 : 1 |
| C8a s0 | **194.03** | 38.65 | 5.0 : 1 |

**C8a's CNN discriminates 33% *less* between candidates than A2's**, despite
correlating 0.876 with the target against A2's 0.594. C8a beats A2 decisively at
engine level while its fusion term is *handicapped*. That is the headroom C9
exists to test.

---

## 7. Proposed matched-Ridge training population

**`dataset_v2`'s test split — all 13,712 records.**

| property | value |
|---|---|
| source | `training/artifacts/dataset_v2.test.jsonl` |
| records | 13,712 (3,614 games) |
| target | `corrected_mate_white_perspective`, re-derived via `dataset.apply_label_policy` |
| features | `[cnn_norm, material, space, center, mobility]` on the **position itself** (matching cell 31 and A1R, so results stay comparable) |
| estimator | `Ridge(alpha=1.0)` — matching cell 31 |
| per-seed | one Ridge per C8a seed (A1R "Option B"; the three CNNs have different output distributions) |
| variant | a second fit excluding the 1,129 checkmate positions (§5.2) |

Rejected alternatives, with reasons:

| population | verdict |
|---|---|
| dataset_v2 **train** split | **Reject.** `cnn_norm` would be in-sample — reproduces the production Ridge's defect 1 |
| dataset_v1 test (A1R's) | **Reject.** 7× smaller, wrong label distribution for C8a, and carries 1 suite placement |
| evaluation-suite positions | **Reject.** Forbidden by the brief and destroys the measurement instrument |
| post-move candidate boards | **Defer.** Fixes defect 2 (objective mismatch) but needs ~411k new Stockfish labels and changes two things at once. See §12 |

---

## 8. Exact train/test separation

```
games.csv (18,920 game units)
  |
  +-- 15,136 train units --> dataset_v2 TRAIN (54,812) ---> C8a CNN WEIGHTS
  |
  +--  3,784 test  units --> dataset_v2 TEST  (13,712) ---> C9 RIDGE COEFFICIENTS
                                                            (CNN out-of-sample here)
evaluation/positions/{extended,phase0_52}  (212)      ---> ENGINE MEASUREMENT ONLY
                                                            scrubbed out of BOTH above
```

Three disjoint roles. Verified: train∩test games = 0, train∩test placements = 0,
suite∩dataset_v2 placements = 0 (both splits).

---

## 9. Exact experiment controls

Everything below is held at C8a's values; **only the fusion coefficients change.**

| | value |
|---|---|
| CNN weights | C8a seeds 0/1/2, **reused as-is, not retrained** (`6bd57636…`, `144594ff…`, `41ee1626…`) |
| representation | `planes12`, `(8,8,12)`, 2,360,129 params |
| dataset | `dataset_v2` (train sha `19009b33…`, test sha `70ca70f1…`) |
| labels | `corrected_mate_white_perspective`, unchanged |
| evaluator | `evaluation/evaluate.py`, unmodified |
| Stockfish | 17.1, depth 8, Threads=1, Hash=16MB, Clear Hash per position |
| suites | `extended` (160), `phase0_52` (52) |
| seeds | 0, 1, 2 — paired; no seed ranked or selected |
| decision rule | the existing A0 three-seed noise band; a change counts only if it exceeds the band **and** is consistent across all three paired seeds |

**Injection mechanism (no production change).** `engine.py` reads
`weight_model.coef_` from the staged models directory. `training/evaluate_arm.py`
already supports `stage_models_dir(experiment_model, staging, ridge_model=...)` —
built for A1R Stage 2 — so a matched Ridge is supplied by staging a different
`weight_model.pkl` into a temporary directory. **No file in `models/` or
`engine.py` is touched.** This path is already exercised and tested.

---

## 10. Proposed metrics

**Stage 1 (minutes, no engine, no Stockfish).** Reuses
`training/refit_ridge.py`'s established outputs:

- coefficient vector and intercept per seed, both fit variants
- ratios normalised to `w[0]`, cosine similarity to the production vector
- **CNN share of within-position ranking spread** (the A1R gate quantity)
- **% of positions where the top-ranked candidate changes** under refit coefficients
- inner-split R² on the fit population
- *added for C9:* `corr(tanh(cnn/d), y)` across divisors, and the implied CNN-term
  spread — so the coefficient headroom and the squash headroom are visible side
  by side before any engine time is spent

**Stage 2 (engine, only if Stage 1 trips the gate).** The standard suite, paired
by seed, C8a-production vs C8a-matched: legality, top-1, top-3, mean/median/p95/max
regret, >300cp blunder rate, regret coverage, Spearman mean/median, mate statuses,
White/Black split, phase/category split.

**Gate (pre-registered, following A1R):** proceed to Stage 2 only if the refit
moves the CNN's ranking share outside the A0 seed-spread reference band, or
changes the top-ranked candidate on a materially different fraction of positions.

---

## 11. Risks and confounds

1. **The stated hypothesis is probably wrong in mechanism (principal risk).**
   §6 shows the recoverable signal sits behind the squash, not in the
   coefficients. A coefficient-only refit can change the CNN:material *ratio*
   (currently 5:1 for C8a) but cannot restore discrimination lost to 40%
   saturation. **A null Stage 2 would not vindicate the production fusion** — it
   would only show that `w[0]` is not the lever.
2. **Matched fusion is not a quality upgrade — measured, not assumed.** A1R
   Stage 2 ran this exact comparison for A0 and A1: it **improved A0 and degraded
   A1**. The refit also changes *methodology* (out-of-sample vs partly in-sample),
   so any C8a change is partly attributable to that, not to matching per se.
3. **A2R Stage 1 declined Stage 2 for A2**, finding A2's refit profile sits inside
   A0's envelope. C8a shares A2's label policy, so the same gate may well clear —
   in which case C9 Stage 2 is not warranted either.
4. **Uniform rescaling is not ranking-neutral.** The heuristic bonuses (±0.65) and
   lookahead (±0.5) are unscaled. They are negligible at `w[0]=330.9`, but a refit
   that shrinks `w[0]` by ~100× would make them decisive. A1R's "channel (b)".
5. **The Ridge governs only one of two CNN entry points** (§1). The lookahead's
   `0.5*tanh(opp/200)` is untouched by any refit.
6. **Defect C3 persists.** The refit is fitted *with* an intercept the engine then
   discards. The matched Ridge inherits this, as A1R's did.
7. **Checkmate positions in the fit population** (§5.2) — mitigated by fitting both
   variants.
8. **Three seeds, descriptive only.** No significance testing; small suites
   (n=160, n=52).

---

## 12. Go / no-go recommendation

**CONDITIONAL GO — run Stage 1 only, then decide.** Stage 1 costs minutes, needs
no Stockfish, and is the established gate. Committing to Stage 2 now would risk
~2 hours of engine evaluation on a lever the audit suggests is the wrong one.

There is a scope decision that is **yours, not mine**, and it should be made
before Stage 2:

| option | what changes | can it address §6? | respects "only the Ridge fitting changes"? |
|---|---|---|---|
| **A. Coefficients only** *(the brief as written)* | `weight_model.coef_` | **No** — cannot undo saturation | Yes |
| **B. Coefficients + squash divisor** | `coef_` and the effective `/200` | **Yes** | No — the divisor is not part of the Ridge |

Option B needs **no production edit**. Because `tanh((raw·k)/200) = tanh(raw/(200/k))`,
staging a thin wrapper that rescales the CNN's output reproduces any divisor
exactly, using the same in-process injection pattern A3/A13/A14 already used for
the encoder. It is experiment-only and reversible.

My recommendation: **run Stage 1 covering both levers** (it is free), then choose
Option A or B with the numbers in hand. If Stage 1 shows — as §6 predicts — that
the coefficient lever is small and the divisor lever is large, Option A alone
would answer the brief's literal question while leaving the real one untested.

**Do not** refit the production Ridge artifact or ship a matched Ridge on the
strength of this audit. C8a's production-fusion results stand as the reference.

---

## 13. Recommended implementation steps (not performed)

1. Extend `training/refit_ridge.py` additively with `--dataset-prefix` (default
   unchanged, so `tests/unit/test_refit_ridge.py` and the committed A1R artifact
   stay byte-identical) and register C8a with the A2 label policy.
2. Add the divisor sweep and the no-checkmate fit variant to Stage 1 output.
3. Run Stage 1 for C8a ×3 seeds; write `training/experiments/C9/stage1_results.json`.
4. Evaluate the gate against the A0 reference band; **stop and report**.
5. Only if the gate trips: stage matched `weight_model.pkl` files via the existing
   `evaluate_arm.stage_models_dir(..., ridge_model=…)` and run both suites ×3 seeds.
6. Compare paired by seed against C8a-production using the A0 noise band.

---

## Verification performed for this audit

No test was modified. Full suite re-run after the audit: **827 passed,
10 xfailed** (the pre-existing Phase-4 deferrals).

Production integrity: `engine.py`, `app.py`, `config.py`, `evaluation/`,
`models/` (CNN **and** Ridge), `baseline/`, `regression/`, `training/labels.py`,
`training/dataset.py`, `training/train.py`, `dataset_v1`, `dataset_v2` and
`dataset_v2_k6` manifests — **all unchanged**.

**C9 has not been implemented. Nothing was refitted. Final QA has not started.
C10 has not started.**
