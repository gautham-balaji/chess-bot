# C8 — `dataset_v2`

**Status:** dataset built, validated and documented. **No CNN has been trained.**
C8 training is a separate, later step.

> **C8 dataset construction changes only the training data. It does not change
> the model architecture, representation, labels, evaluator, Ridge, or
> production engine.**

---

## 1. Why `dataset_v2` exists

C6 closed the representation/architecture branch: A3, A13, A13P, A13R and A14 all
failed to beat A2, and A14 showed the network simply ignores features it is given
no reason to learn. C7 then measured the training data and found the constraint
is upstream of the model.

`dataset_v1` takes **exactly one position per game**, the board after at most 20
plies. Measured consequences:

| | `dataset_v1` | `dataset_v2` |
|---|---:|---:|
| records | 9,667 | **68,524** |
| source games used | 9,667 (one position each) | 18,024 |
| positions at exactly ply 20 | 8,932 (92.4%) | — |
| min piece count | **22** | **2** |
| endgame positions (≤12 pieces) | **0** | 6,949 |
| White to move | **95.9%** | **58.6%** |
| mate-typed labels | 1.9% | 10.2% |
| split unit | position | **game content** |

The evaluation suites it is judged on are 21–35% endgames and ~55% White. The old
dataset and the measurement instrument barely overlap; `dataset_v2` closes that gap.

Full measurements: `docs/C7_DATASET_EXPANSION_DESIGN.md`.

---

## 2. Extraction policy — `evenly_spaced_4_minply16`

At most **4** positions per game, evenly spaced from **ply 16** through the game's
**final ply**, never closer than **4 plies** apart.

| parameter | value |
|---|---|
| min ply | 16 |
| max positions per game | 4 |
| min gap between selected plies | 4 |
| RNG | **none** — the policy is closed-form |
| sampling seed | **none required** |
| canonical row per game unit | yes (1,138 duplicate CSV rows skipped) |
| record order | sorted by `(game_content_key, ply)` |

`min_ply = 16` is the measured knee of the corpus uniqueness curve: positions at
ply 1 are shared by 0.1% uniqueness (20 legal first moves), ply 12 reaches 0.85
and ply 16 reaches 0.94. Extracting from ply 1 produces **26% placement leakage**
across a game split; from ply 16 it is 0.69%, and 0 after scrubbing.

Phase buckets and random position sampling were both evaluated in C7 and
rejected — phase-stratified sampling produced 22,314 selected pairs closer than
the minimum gap.

### Measured outcome

| stage | train | test | total |
|---|---:|---:|---:|
| selected | 55,082 | 13,819 | **68,901** |
| after placement dedup | 54,813 | 13,800 | 68,613 |
| after train/test leakage scrub | 54,813 | 13,712 | 68,525 |
| after evaluation-suite scrub | **54,812** | **13,712** | **68,524** |

Selected pairs closer than 4 plies: **0**.
Positions per game: median 4, max 4.

---

## 3. Game split

The split unit is **game content** — the sha256 of the `moves` string — **not the
`id` column**. Measured in C7: 813 ids appear on multiple rows (all verified exact
duplicate rows), and 50 move sequences are filed under 243 different ids. Keying
on `id` would let identical game content straddle the split.

| | value |
|---|---|
| split seed | 42 |
| test fraction | 0.20 |
| ordering | unique content keys sorted, then seeded permutation; first 20% to test |
| row-order invariant | yes (sorting precedes permutation) |
| rows | 20,058 |
| **unique game units** | **18,920** |
| **train / test units** | **15,136 / 3,784** |
| `train_units ∩ test_units` | **0** |

Positions are extracted **after** the split, independently per side, so a game
can never contribute to both.

### Hashes (reproduce C7 exactly)

```
game list   b90881bf3c6245ba3634800da8879ccf8ecb9a64fb41f5116c5439983dc64e62
train games cb1308d9af1ee45709f272492c0a22a02b8e7026f055d8ff43e1d81f881c19fb
test games  d8632c3ac89be36bced244f4370eec09c3c80b2e6a0fe841f62ca67df97809f3
```

---

## 4. Leakage handling

**Deduplication is at piece-placement level**, not exact FEN. The 12-plane encoder
reads only piece squares, so two records with the same placement are literally
the same CNN input regardless of side to move, castling rights or move counters.
Placement is therefore the level at which overlap actually leaks into this model.

| step | measured |
|---|---|
| duplicate placements removed within train | 269 |
| duplicate placements removed within test | 19 |
| placements present in **both** train and test | **88** |
| records removed to resolve it | 88, **from TEST only** |
| final train/test placement overlap | **0** |

**Train is never reduced to resolve train/test overlap.** The scrub removes the
colliding records from the test side, so the training set keeps its full size and
the test set shrinks by 0.64%. Every removed placement's sha256 is recorded in the
manifest.

Residual transposition leakage is irreducible in principle — two different games
can reach the same position — which is why the scrub exists rather than relying on
the game split alone.

---

## 5. Evaluation-suite scrub

The suites are the measurement instrument for every C6/C8 arm, so no training
record may be one of their positions.

| | value |
|---|---|
| suites read | `extended` (160), `phase0_52` (52) |
| distinct suite placements | 185 |
| records removed from train | **1** |
| records removed from test | 0 |
| final suite overlap, train / test | **0 / 0** |
| suites modified | **no** — read-only |

For scale: without the ply-16 floor this scrub would have removed 25 training
records. `dataset_v1` contains 6 suite placements and was never scrubbed, so
**`dataset_v2` is cleaner than the A2 baseline on this axis.**

---

## 6. Labels — unchanged

`dataset_v2` uses the A2 policy `corrected_mate_white_perspective` by **importing
`training/labels.py`**, not by reimplementing it. A test asserts the builder
contains no local copy of `make_label`, `CP_CLIP`, `MATE_SCORE_BASE` or the
perspective logic, and another re-derives 500 stored labels through
`L.make_label` and compares.

Unchanged: mate mapping, perspective normalisation, clipping, scale.
C7 verified the policy on later-game positions, endgames, mate distances, mate-0,
both perspectives and clipping, and found no bug.

| | train | test |
|---|---:|---:|
| cp labels | 49,202 | 12,327 |
| mate labels | 5,610 | 1,385 |
| clipped | 2 | 0 |
| label min / max | −2000 / 2000 | −2000 / 2000 |
| label mean | +38.2 | +39.1 |
| anomalies | 0 | 0 |

Stockfish 17.1, depth 8, `Threads=1`, `Hash=16MB`, hash cleared before every
position — identical to `dataset_v1` and to the Phase 3 evaluator.

### Terminal positions: measured, and deliberately kept

The policy selects the game's **final ply**, which raises the question of how
many records are terminal. Three different things are easy to conflate, so they
are reported separately.

> An earlier revision of this section stated "26.2% of train records are terminal
> positions" and "the engine never evaluates a checkmated board at inference".
> **Both were wrong.** The 26.2% figure was *final-ply* records, not terminal
> ones, and the inference claim is contradicted by the engine's own code. The
> measured values below replace them.

#### Final-ply ≠ terminal ≠ checkmate

| measure | train | test | total | % of 68,524 |
|---|---:|---:|---:|---:|
| **final ply of the game** | 14,385 | 3,605 | 17,990 | 26.25% |
| **any terminal** (`is_game_over`) | 4,772 | 1,163 | 5,935 | **8.66%** |
| **checkmate** (zero legal moves) | 4,658 | 1,129 | 5,787 | **8.45%** |
| stalemate | 0 | 0 | 0 | 0.00% |

Final-ply records are 26.24% of the train split (14,385 / 54,812), but most are
not terminal at all. By the source game's `victory_status`, the 14,385 final-ply
train records break down as **55.1% resignations, 32.4% mates, 8.3% timeouts,
4.2% draws** — roughly two-thirds are ordinary legal positions that a player
resigned, lost on time, or agreed drawn. Only the mate subset is a game-over
state.

#### Terminal classification re-derived from the FENs

| class | records | share |
|---|---:|---:|
| checkmate | 5,787 | **8.45%** |
| insufficient material | 142 | 0.21% |
| fifty-move claimable | 9 | 0.01% |
| *non-checkmate terminal, combined* | *151* | *0.22%* |
| **non-terminal** | **62,586** | **91.33%** |
| total | 68,524 | 100.00% |

The 142 insufficient-material and 9 fifty-move records still have legal moves and
are ordinary evaluable positions. **Checkmates are the only records with zero
legal moves.** (The derived count differs from `is_game_over` by 3 records,
because a fifty-move draw is *claimable* rather than automatic.)

#### The production engine does evaluate checkmated boards

`rerank_moves` pushes each candidate move and encodes the **resulting** board, so
a candidate that delivers mate produces a checkmated board that reaches
`cnn_model.predict`. Measured, descriptively:

- Instrumenting the encoder on a Scholar's-mate position: **1,472 boards encoded,
  1 of them checkmated**, and the production CNN scored that mate-for-White at
  **≈ −4.25 cp**.
- Across both evaluation suites, **8 of 212 positions (3.8%)** feed at least one
  checkmated board to the CNN — 8 of 6,045 candidate encodings (**0.132%**).
- Those 8 positions correspond exactly to the 8 `missed_forced_mate` results
  reported for A2 (4 on `extended`, 4 on `phase0_52`). This is a correspondence
  between the two sets, not a demonstration of cause.
- `engine.py` contains no checkmate handling in the move-selection path; the only
  textual matches for "mate" are substrings of "material".

So checkmated boards are part of the candidate-evaluation path, not outside it.

#### Why the mate rate is 10.2%, not C7's ~4.6%

C7's ~4.6% was a **projection**, obtained by applying per-phase mate rates from a
300-position label-cost probe to the policy's phase mix. That probe sampled
phase-balanced positions and did not model the extraction policy, which samples
evenly from ply 16 **through each game's final ply**. Always including the final
ply raises exposure to late-game and checkmate positions, and the realised rate is
**10.2%** (5,610 of 54,812 train mate-typed labels, of which 4,658 are already
mated and 952 are forced-mate-ahead).

#### Decision: keep the checkmate positions

C8 keeps them, on this evidence:

- they lie on the actual candidate-evaluation path (above);
- they correspond to a documented, reproducible failure mode — A2 misses all 8
  mate-in-1 suite positions;
- removing them would drop **4,658 train records (−8.5%)** and **1,323 endgame
  records (−24.0% of the endgame bucket)**, the bucket `dataset_v2` exists to
  create;
- no measured evidence currently justifies the change ahead of the controlled C8
  training run.

**Caveat, recorded deliberately.** Training is **8.45%** checkmate against
**0.132%** of observed checkmated candidate encodings — a ~64× over-representation
and a legitimate distribution concern. It is not acted on now. If C8 underperforms
A2 and diagnostics point at CNN output-scale inflation or mate saturation, a
checkmate-excluded follow-up dataset is a reasonable next step. **That follow-up
is not part of C8 and should not be run now.**

---

## 7. Final dataset and hashes

| | value |
|---|---|
| train records | **54,812** (14,410 games) |
| test records | **13,712** (3,614 games) |
| total | **68,524** |
| train file | `training/artifacts/dataset_v2.train.jsonl` |
| test file | `training/artifacts/dataset_v2.test.jsonl` |
| manifest | `training/artifacts/dataset_v2.manifest.json` |

```
source games.csv  e7aadff104a610afb5403caf81c1461babecb0dc87760ff5eea7dc0e7a4a8129
train jsonl       19009b338615e6f03c83eb4ebec4ce13be219a80bba905dda61ce109821d92fb
test jsonl        70ca70f18872337cecdd6bc1818981d0808598996c5f2c87ce7a6bb2fb8d3906
train labels      4e859b8c7700d72352f4b686b5fee2fd47233c53201d09197dc676c9722c307d
test labels       6adc83737e748b39c4b3a3d766abb4146dd169029fcf619c991d1c77fcefe9c3
```

Composition:

| | train | test |
|---|---|---|
| side to move | white 32,125 / black 22,687 | white 8,073 / black 5,639 |
| phase | opening 14,805 / middlegame 34,497 / endgame 5,510 | 3,678 / 8,595 / 1,439 |
| ply | min 16, max 349, mean 40.8 | min 16, max 255, mean 41.3 |
| piece count | min 2, max 32 | min 2, max 32 |

The `.jsonl` files are **gitignored** by the existing
`training/artifacts/*.jsonl` rule — the same convention `dataset_v1` follows. Only
the manifest is committed; the data is rebuilt locally.

Per record: `game_content_key`, `game_id`, `source_row_index`, `ply`, `fen`,
`placement`, `side_to_move`, `piece_count`, `phase`, `plies_in_game`,
`is_checkmate`, `is_stalemate`, `is_game_over`, `legal_move_count`, `eval_type`,
`raw_stockfish_value`, `raw_value_perspective`, `label`, `label_perspective`,
`was_clipped`.

---

## 8. How to rebuild

```bash
python -m training.build_dataset_v2          # ~7 minutes, needs Stockfish
```

The build refuses to run if `games.csv` does not match the audited sha256, because
the C7 split hashes are only valid for that file. It then checks itself against
every C7-measured count and exits non-zero on any drift:

```
  game_units, train_units, test_units, selected_total/train/test,
  deduped_train/test, train_test_overlap_placements,
  game_list_sha256, train_game_sha256, test_game_sha256
```

Structure-only build, no Stockfish: `--skip-labels`.

**Determinism.** Rebuilding from the same `games.csv`, Stockfish version and
configuration, and pipeline version `c8-1` reproduces byte-identical files.
`generated_at_utc` and `generation_seconds` are informational and feed no hash.

## 9. How to verify

```bash
python -m pytest tests/unit/test_training_build_dataset_v2.py -q   # 52 tests
python -m pytest tests/unit/test_training_c7_audit.py -q           # 44 tests
```

The tests assert, against the built artifact: min ply 16, ≤4 records per game, the
4-ply gap, placement uniqueness within each side, **zero train/test placement
overlap**, **zero train/test game overlap**, **zero evaluation-suite overlap**,
stored labels re-derived through `L.make_label`, manifest hashes matching the
files on disk, and that the builder contains no RNG and no production path.

---

## 10. What is intentionally NOT changed

| | status |
|---|---|
| model architecture | untouched — A2 Sequential, 2,360,129 params |
| representation | untouched — `planes12`, `(8,8,12)`; **no evaluation shim needed** |
| `training/labels.py` | untouched — imported, not copied |
| `training/train.py`, `training/dataset.py` | untouched |
| `engine.py`, `app.py`, `config.py` | untouched |
| `evaluation/` and both suites | untouched, read-only |
| production models and Ridge | untouched |
| baseline / regression fixtures | untouched |
| `dataset_v1` | untouched — A2 remains reproducible |

Because the representation stays `planes12`, a C8 model loads in the unmodified
engine and evaluates through `evaluation/evaluate.py` directly — unlike every arm
since A3, which needed the experiment-only shim.

---

## 11. Next step — separate task

**CNN training has not started and must not be inferred from this document.**

C8 training, when authorised, should hold everything constant against A2 and vary
only the dataset: A2 architecture, `planes12`, the A2 label policy, Huber, Adam
1e-3, batch 64, cap 100, ReduceLROnPlateau, EarlyStopping patience 10 restore
best, seeds 0/1/2, frozen production Ridge, both suites, compared against the A0
three-seed noise band.

Note the confound, stated in C7 §8 and unchanged here: C8 differs from A2 in
dataset **size** *and* **composition**. That is the hypothesis, not a nuisance — but
a gain cannot be attributed to size alone without a size-matched control arm.
