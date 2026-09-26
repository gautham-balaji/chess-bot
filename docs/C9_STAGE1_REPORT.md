# C9 Stage 1 — Matched-Ridge and Squash-Divisor Diagnostic (results)

## 1. Status

**Stage 1 only.** Stage 2 was **not** run and is not implemented.

- **No Stockfish engine evaluation was run.**
- **No production file was changed.** `engine.py`, `app.py`, `models/` (CNN *and*
  Ridge), `evaluation/`, baselines and regression fixtures are byte-identical.
- **C8a CNN weights were not retrained** — the three committed models were loaded
  read-only and hash-verified first.
- **The production Ridge was not replaced.** It was read once, as the comparison
  reference.

Design: [C9_RIDGE_AUDIT.md](C9_RIDGE_AUDIT.md) · Prior:
[A1R S1](C6_A1R_STAGE1_REPORT.md) · [A1R S2](C6_A1R_STAGE2_REPORT.md) ·
[A2R S1](C6_A2R_STAGE1_REPORT.md) · [C8a](C8A_REPORT.md)
Artifact: `training/experiments/C9/stage1_results.json` (gitignored)

---

## Gate decision up front

> **The coefficient gate is TRIPPED, on both Ridge variants and all three seeds.**
>
> Refitting moves the CNN's share of ranking spread from **81.5–83.4%** under
> production coefficients to **45.5–49.6%** (variant A) or **54.1–58.4%**
> (variant B), against an A0 reference band of **82.85–84.93%** — roughly 2 pp
> wide. The refit changes the top-ranked candidate on **23.8–36.3%** of positions.
>
> **Stage 2 was not run, per instruction.**

> **Separately — and this is not a gate input — the divisor sweep reproduces the
> audit.** `corr(tanh(cnn/d), label)` rises from **0.703–0.709** at the production
> `d=200` to **0.873–0.874** at `d=2000`, against a raw-CNN ceiling of
> **0.873–0.876**. Saturation falls from **40.1%** to **0.0%**.
>
> **A large divisor effect is NOT evidence that the coefficient refit works.**
> The two levers are reported separately throughout, and §8 shows they interact
> in a way that makes neither sufficient alone.

---

## 2. Objective

Test whether the C8a CNN's fusion is mismatched to its output distribution, by
measuring **two separable things**:

1. **Coefficient-only refit** — fit a Ridge to C8a's own output distribution and
   ask whether it would rank moves differently.
2. **Divisor/squash effect** — measure how much signal the fixed `tanh(x/200)`
   discards *before* the Ridge sees anything.

Stage 1 is a diagnostic. **No production strength claim follows from it.**

---

## 3. Exact controls

| | value |
|---|---|
| CNN weights | C8a seeds 0/1/2, loaded read-only, **not retrained** |
| representation | `planes12`, `(8,8,12)`, 2,360,129 params (asserted) |
| label policy | `corrected_mate_white_perspective`, re-derived via `dataset.apply_label_policy` and asserted equal to the stored labels |
| features | `[cnn_norm, material, space, center, mobility]` — the five production features, built by `refit_ridge.board_features` |
| estimator | `Ridge(alpha=1.0)`, `fit_intercept=True` — matching notebook cell 31 |
| fit population | `dataset_v2.test.jsonl`, manifest-verified |
| RNG | none introduced; the only seed is the pre-existing inner-split seed 1234 |
| engine / Stockfish | **not invoked** |

---

## 4. Fit population and leakage checks

`training/artifacts/dataset_v2.test.jsonl`, sha256 `70ca70f18872337c…`
(manifest OK), **13,712 records from 3,614 games**.

| check | measured |
|---|---:|
| overlap with `extended` + `phase0_52` placements | **0** |
| overlap with suite exact FENs | **0** |
| overlap with C8a **train** placements | **0** |
| overlap with C8a **train** games | **0** |
| labels re-derived == stored labels | **yes** |
| checkmate positions (zero legal moves) | 1,129 (8.2%) |

The guards are enforced, not merely reported: `leakage_checks` raises
`SystemExit` on any non-zero overlap, and a test feeds it a deliberately poisoned
population to confirm it fires.

**Ranking populations** (used for spread only — never fitted, no labels read):

- `extended` — first 80 positions, 2,701 candidate moves. Required because the
  A0 gate band was measured there.
- `heldout` — 80 dataset_v2 test positions, 2,416 candidate moves. Zero
  evaluation-suite contact; confirms `extended` is not an artifact.

---

## 5. C8a model verification

All three verified against `docs/C9_RIDGE_AUDIT.md` **before** any fitting:

| seed | weights sha256 | matches audit | params | trained on |
|---|---|---|---:|---|
| 0 | `6bd57636a2747ec479992fb0873385167956352d776808ee2ef79af023ad36e2` | yes | 2,360,129 | dataset_v2 |
| 1 | `144594ffff30bdfa3b59eafc8897e811bbaa6ab269c1d5c82a29364e79a24b97` | yes | 2,360,129 | dataset_v2 |
| 2 | `41ee1626feff429a956c2125d83b2dcf5cf86b93ad04a86e3fb556d27152071c` | yes | 2,360,129 | dataset_v2 |

A mismatch aborts the run.

---

## 6. Matched Ridge coefficients — variant A (all 13,712)

| run | `w0` cnn | `w1` mat | `w2` space | `w3` center | `w4` mob | cosine | `w0` scale | R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **production** | 330.9005 | 32.3899 | 0.7919 | 5.1654 | 0.0188 | — | — | — |
| C8a s0 | 376.8479 | 37.5751 | 20.2692 | −21.0870 | −1.3168 | 0.9962 | 1.1389 | 0.6115 |
| C8a s1 | 380.4040 | 37.3819 | 20.3837 | −18.4307 | −1.6115 | 0.9967 | 1.1496 | 0.6141 |
| C8a s2 | 388.9432 | 36.2826 | 20.3869 | −19.0929 | −1.4725 | 0.9967 | 1.1754 | 0.6137 |

The refit **raises** `w0` by 14–18%, yet `space` grows **25×** (0.79 → 20.3) and
both `center` and `mobility` **flip sign** (+5.17 → −19 to −21; +0.019 → −1.3 to
−1.6). Cosine similarity stays at 0.996 — reproducing A1R's finding that cosine
is a poor discriminator here because `w0` dominates the norm.

---

## 7. Checkmate-excluded Ridge — variant B (12,583)

| run | `w0` cnn | `w1` mat | `w2` space | `w3` center | `w4` mob | cosine | `w0` scale | R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C8a s0 | 245.6436 | 43.6440 | 9.3556 | −2.4256 | −0.4335 | 0.9960 | 0.7423 | 0.6429 |
| C8a s1 | 246.1096 | 43.6759 | 9.5956 | −0.7440 | −0.5428 | 0.9961 | 0.7438 | 0.6447 |
| C8a s2 | 251.6771 | 42.8550 | 9.5393 | −1.2943 | −0.4062 | 0.9967 | 0.7606 | 0.6435 |

**The 1,129 mated boards change the fit materially.** Variant A's `w0` is
**1.53×** variant B's (≈377–389 vs ≈246–252), and the direction reverses: with
mates the refit *raises* `w0` above production, without them it *lowers* it to
0.74–0.76×. `center`'s sign flip also largely disappears (−19 → −1.2).

8.2% of the fit population carries ±2000 labels on boards the engine can never
rank moves in, and they pull the CNN coefficient up by half again. This is why
the audit required both variants; neither is silently preferred here.

Variant B also fits slightly better (R² 0.643–0.645 vs 0.612–0.614).

---

## 8. Divisor sweep

Measured on the same 13,712 positions. **No Ridge was refitted per divisor** —
this quantifies what the squash discards, nothing more.

| d | s0 corr | s1 corr | s2 corr | mean saturated | CNN term SD (`extended`) | CNN share (`extended`) |
|---:|---:|---:|---:|---:|---:|---:|
| **200** *(production)* | 0.7044 | 0.7033 | 0.7086 | **40.1%** | **48.58** | **82.20%** |
| 400 | 0.7641 | 0.7638 | 0.7674 | 16.7% | 35.37 | 77.07% |
| 600 | 0.8041 | 0.8040 | 0.8069 | 9.8% | 25.65 | 70.82% |
| 800 | 0.8307 | 0.8304 | 0.8327 | 6.9% | 19.87 | 65.25% |
| 1000 | 0.8479 | 0.8472 | 0.8490 | 3.8% | 16.12 | 60.37% |
| 1500 | 0.8679 | 0.8666 | 0.8673 | 0.0% | 10.87 | 50.70% |
| 2000 | 0.8742 | 0.8726 | 0.8726 | 0.0% | 8.19 | 43.67% |
| **raw** | **0.8760** | **0.8742** | **0.8732** | — | — | — |

Raw C8a output: mean |cnn| **427.7**, std **661.2**.

The audit's claim reproduces exactly: the production divisor is the **worst** of
every value swept, C8a is still improving at d=2000, and d=2000 recovers
essentially the full raw correlation (0.8742 vs 0.8760).

### The interaction that matters, and that the audit did not anticipate

**Raising the divisor increases correlation but *decreases* the CNN term's
ranking spread under fixed coefficients** — from 48.58 at d=200 to 8.19 at
d=2000, and its share of total spread from 82.2% to 43.7%.

This is arithmetic, not a bug: for large `d`, `tanh(x/d) ≈ x/d`, so the feature
shrinks as `1/d`. More information survives, in a smaller-magnitude feature.

**Consequence: the two levers are not separable.** A larger divisor without a
compensating increase in `w0` hands ranking influence to `material` — the
opposite of the intent. Any Stage 2 that varies the divisor must refit `w0`
alongside it, or it will measure the wrong thing. Neither lever alone is
sufficient, which is precisely why Stage 1 measured both.

---

## 9. Ranking-spread diagnostics

CNN share of within-position ranking spread, production → refit:

| run | `extended` prod | `extended` refit | Δ pp | `heldout` prod | `heldout` refit | top move changed |
|---|---:|---:|---:|---:|---:|---:|
| A all s0 | 81.54 | 45.46 | −36.08 | 69.66 | 32.80 | 23.75% |
| A all s1 | 81.67 | 45.92 | −35.74 | 77.05 | 41.76 | 36.25% |
| A all s2 | 83.39 | 49.63 | −33.76 | 76.71 | 42.07 | 28.75% |
| B no-mate s0 | 81.54 | 54.14 | −27.40 | 69.66 | 39.23 | 17.50% |
| B no-mate s1 | 81.67 | 54.42 | −27.25 | 77.05 | 48.47 | 25.00% |
| B no-mate s2 | 83.39 | 58.37 | −25.02 | 76.71 | 49.10 | 16.25% |

The two ranking populations agree in direction and magnitude, so the finding is
not an artifact of the `extended` subset.

Note the apparent paradox: the refit **raises** `w0` (variant A) while the CNN's
ranking *share* falls sharply. Both are true — `space`'s 25× growth and
`center`'s sign flip add far more spread than the 14–18% `w0` increase does.
Share is relative.

**These are proxies, not the engine's choice.** `rerank_moves` adds unscaled
heuristic bonuses and a 1-ply lookahead on top of the weighted sum. Only Stage 2
measures the engine.

---

## 10. A0 gate comparison

Reference band: **82.85–84.93%** (A0 seed spread under production coefficients,
A1R Stage 1), measured on `extended` — so the gate is evaluated on `extended`
for like-with-like comparison.

| seed | production share | refit share (A) | outside band? | top move changed |
|---|---:|---:|---|---:|
| 0 | 81.54% | **45.46%** | **yes** | 23.75% |
| 1 | 81.67% | **45.92%** | **yes** | 36.25% |
| 2 | 83.39% | **49.63%** | **yes** | 28.75% |

Mean top-move change **29.58%**. Variant B also lands outside the band on all
three seeds (54.14–58.37%).

---

## 11. Decision

> **GATE: TRIPPED.**

Both gate conditions are met, on both variants and all three seeds:

1. CNN ranking share moves **33.8–36.1 pp** (variant A) or **25.0–27.4 pp**
   (variant B) below the A0 band — the band itself is ~2 pp wide.
2. The refit changes the top-ranked candidate on **23.8–36.3%** of positions.

**Stage 2 was not run.** Per instruction, the gate result does not trigger it.

---

## 12. What Stage 1 does and does not establish

**Establishes:**

- A Ridge fitted to C8a's own output distribution is **materially different**
  from the production Ridge in ranking-relevant terms.
- That difference is **robust to the checkmate question** — it survives both
  variants, though its size and the sign of the `w0` change do not.
- The production `tanh(x/200)` discards **~0.17 of correlation** from C8a
  (0.704 → 0.876 raw), with 40.1% of positions saturated.
- Raising the divisor recovers that correlation but **shrinks the CNN term**,
  so the divisor and the coefficients must move together.

**Does NOT establish:**

- **That a matched Ridge would make the engine play better.** Stage 1 measures
  coefficients and proxies, not moves. A1R Stage 2 measured the real thing for
  A0/A1 and found matched fusion **improved A0 and degraded A1** — it is a
  different fitting methodology, not a quality upgrade.
- **That the divisor is the better lever.** §8 shows it is not free.
- **Any production strength claim whatsoever.**
- **Which variant is correct.** A and B disagree on whether `w0` should rise or
  fall; that is a real open question, not a reporting detail.

A methodological asymmetry persists from A1R: the refit is fitted
**out-of-sample** while the production Ridge was fitted **partly in-sample**.
Some of the measured difference is attributable to that, not to matching per se.
Defect C3 also persists — the refit is fitted *with* an intercept `engine.py`
discards.

---

## 13. Is Stage 2 warranted?

**On the pre-registered gate: yes.** On judgement: **warranted but not
obviously worth it as currently scoped**, for three measured reasons.

1. **A1R Stage 2 already ran this comparison** for A0 and A1 and found matched
   fusion is arm-dependent, not a general improvement.
2. **A2R Stage 1 declined Stage 2** for A2, whose refit profile sits inside A0's
   envelope. C8a shares A2's label policy, though its refit profile here is
   distinctly its own (`space` +25×, `center` sign-flipped).
3. **§8's interaction means a coefficient-only Stage 2 tests the weaker lever.**
   It would answer the brief's literal question while leaving the larger measured
   effect untested.

If Stage 2 runs, the design should be decided first: coefficients only
(variant A and B differ, so pick deliberately), or coefficients **and** divisor
jointly. That is a scope decision, not a Stage 1 finding, and it is not made here.

---

## 14. Tests and production integrity

**Full suite: 861 passed, 10 xfailed.**

Baseline before this work was 827 passed / 10 xfailed. The +34 accounts exactly:

| source | delta | why |
|---|---:|---|
| `tests/unit/test_training_c9_stage1.py` | **+33** | new file |
| `tests/unit/test_refit_ridge.py` | **+1** | 49 → 50; one roster test was split into two (see below) |
| **total** | **+34** | 827 → **861** |

No test was weakened or deleted.

Two assertions in `tests/unit/test_refit_ridge.py` were updated, both roster
statements about the newly registered `C8a` arm:

- `set(RR.ARM_POLICIES)` now includes `"C8a"`.
- `test_every_registered_arm_has_a_distinct_label_policy` was **false by design**
  once C8a was registered, because C8a deliberately reuses A2's label policy. It
  was split into two sharper tests rather than relaxed: one asserting A0/A1/A2
  remain mutually distinct, and one asserting C8a's sharing of A2's policy is
  intentional.

`tests/unit/test_refit_ridge.py::test_default_arm_set_is_still_the_a1r_pair`
passes unchanged, so the committed A1R workflow is intact.

**Production integrity — all unchanged:**

```
models/cnn_model.keras              972d81199a1355667fca554b06dd05b35fdec3c4dc789968ac4a1b0599d8dec3
models/weight_model.pkl             59a731127cc9b440c866c7a5d27ce39d905c9c5dadd8116d7f97a8542f18bcd0
evaluation/positions/extended.json  9d5419cceec43bac98f9cdcf8109a4242e93791fa267f1df54fc1dddc41f929a
evaluation/positions/phase0_52.json 6078d0e8db9e4124b984bbf3b5ad018de6b9c84b49361b01058ff58d875c0208
```

`engine.py`, `app.py`, `config.py`, `evaluation/`, `baseline/`, `regression/`,
`training/labels.py`, `training/dataset.py`, `training/train.py`, `dataset_v1`,
`dataset_v2` and `dataset_v2_k6` manifests: **no change**. C8a and its artifacts
were read-only throughout.

---

## Reproduction

```bash
python -m training.c9_stage1                       # ~3 min, no Stockfish
python -m pytest tests/unit/test_training_c9_stage1.py tests/unit/test_refit_ridge.py -q

# the A1R default is unchanged and still reproduces its committed artifact
python -m training.refit_ridge
```

**Stage 2 has not been implemented or run. Final QA has not started. C10 has not
started.**
