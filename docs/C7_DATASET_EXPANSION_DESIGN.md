# C7 — Controlled Dataset Expansion: Audit and Design

**Status:** audit and design only. No CNN trained, no dataset rebuilt, no production file touched.
**Scope:** measure what `games.csv` supports and design a leakage-free game-level split for C8.
**Every number below is measured** by `training/c7_dataset_audit.py` and
`training/c7_label_cost_probe.py`, recorded in `training/artifacts/c7_audit.json`
and `training/artifacts/c7_label_cost.json`.

---

## Result up front

> **The dataset is not merely small — it is one position per game, all in the
> opening, and 95.9% White to move.**
>
> `dataset_v1` takes exactly one position per game: the board after at most 20
> plies. Measured consequences: **8,932 of 9,667 records sit at exactly ply 20**,
> every record has a fullmove number of at most 11, the minimum piece count is
> 22, and **the dataset contains zero endgame positions**. 95.89% are White to
> move, and the 407 Black-to-move records are the short games — mean label
> **+671 cp** against White's +50 cp.
>
> The evaluation suites it is judged on are nothing like this: `extended` has 33
> positions with ≤12 pieces and is 53% White, `phase0_52` has 18 and is 58% White.
>
> **Expansion is feasible and cheap.** The corpus holds **1,212,827 legal plies**
> across **18,920 unique games**. Labelling runs at **154.5 positions/second**, so
> 100,000 positions cost **≈11 minutes** of Stockfish time. A game-level split
> with a ply-16 floor yields **68,901 positions at 0.69% placement leakage before
> scrubbing and exactly 0 after**, while *reducing* evaluation-suite contamination
> from today's 6 positions to 1.

**Recommendation: proceed to C8.** The binding constraint is real, measured, and
removable. §10 gives the verdict; §8 gives the exact C8 design.

---

## 1. What data actually exists

### 1.1 Source file

| property | measured value |
|---|---|
| file | `games.csv` |
| sha256 | `e7aadff104a610afb5403caf81c1461babecb0dc87760ff5eea7dc0e7a4a8129` |
| size | 7,672,655 bytes |
| rows | 20,058 |
| columns | `id, rated, created_at, last_move_at, turns, victory_status, winner, increment_code, white_id, white_rating, black_id, black_rating, moves, opening_eco, opening_name, opening_ply` |

### 1.2 Games and identifiers (audit questions 1–3)

**Rows are not games, and `id` is not a safe split key.**

| measurement | value |
|---|---|
| rows | 20,058 |
| distinct `id` values | **19,113** |
| ids appearing on more than one row | 813 (1,758 rows; max 5 rows per id) |
| duplicate-id groups whose rows carry **identical** `moves` | **813 of 813** |
| duplicate-id groups whose rows carry different `moves` | 0 |
| distinct `moves` strings | 18,920 |
| `moves` strings shared by more than one **id** | 50, spanning 243 distinct ids |
| **unique games by content** | **18,920** |

Two separate collisions exist. Duplicate ids are exact duplicate rows —
verified, every group has identical moves. Separately, 50 move sequences are
filed under 243 different ids; these are almost all trivially short games
(median 3 plies, max 22, only 3 with ≥12 plies).

**Consequence for the split:** keying on `id` would let identical game content
land on both sides. The split unit must be the **move sequence**, which collapses
20,058 rows to 18,920 units and makes both collisions disappear. This is
implemented and tested.

### 1.3 Game lengths and available plies (questions 4–5)

Every game replayed with `python-chess`. **0 games had malformed SAN; 0 games
replayed to zero plies.**

| statistic | plies |
|---|---|
| min | 1 |
| p10 | 22 |
| median | **55** |
| mean | 60.47 |
| p90 | 107 |
| max | 349 |
| **total legal plies in the corpus** | **1,212,827** |

Games surviving to a given ply:

| ply | 1 | 10 | 20 | 30 | 40 | 60 | 80 | 100 | 150 |
|---|---|---|---|---|---|---|---|---|---|
| games | 20,058 | 19,441 | 18,397 | 16,715 | 14,383 | 8,907 | 4,900 | 2,622 | 311 |

**The corpus is ~125× larger than the dataset built from it.** `dataset_v1` uses
9,667 of 1,212,827 available plies — 0.8%.

### 1.4 What `dataset_v1` actually is

| measurement | value |
|---|---|
| records | 9,667 |
| distinct game ids | 9,667 — **exactly one position per game** |
| records at exactly ply 20 | **8,932 (92.4%)** |
| records below ply 20 | 735 |
| fullmove number | min 1, median 11, **max 11** |
| piece count | min **22**, median 29, max 32 |
| **records with ≤12 pieces (endgame)** | **0** |
| side to move | white 9,260 (95.89%) / black 407 (4.11%) |
| eval type | cp 9,479 / mate 188 (1.9%) |
| mean label, White to move | **+50.2 cp** |
| mean label, Black to move | **+671.4 cp** |

**Every Black-to-move record has an odd ply count** — verified. Extraction stops
at ply 20, an even number, so White is to move unless the game *ended* first. The
4.11% Black sample is therefore not a sample of Black-to-move positions at all;
it is a sample of games that ended early, which is why its mean label is 13×
White's.

This single fact explains the White/Black asymmetry that persisted through every
C6 arm (A2 White MAE ≈157, Black MAE ≈277–298). It is a property of the dataset,
not of the architecture — which is consistent with C6's finding that no
representation or architecture change moved it.

### 1.5 Train/evaluation distribution mismatch

| | `dataset_v1` | `extended` | `phase0_52` |
|---|---|---|---|
| n | 9,667 | 160 | 52 |
| White to move | **95.9%** | 53.1% | 57.7% |
| min piece count | **22** | 3 | 3 |
| positions with ≤12 pieces | **0** | 33 (20.6%) | 18 (34.6%) |
| categories | opening only | opening 60, middlegame 64, endgame 21, tactical 10, defensive 5 | opening 18, middlegame 12, endgame 12, tactical 6, defensive 4 |

The model is trained exclusively on move-11 opening positions and evaluated on a
suite that is one-fifth to one-third endgames. This is a first-order train/eval
mismatch and the strongest single argument for C7.

---

## 2. Duplicate and near-duplicate structure (questions 7–9)

Measured over all 1,212,827 plies, at three identity levels:

| level | what it is | unique keys | within-game duplicate plies | keys in >1 game |
|---|---|---:|---:|---:|
| `exact` | full `board.fen()` | 1,024,317 | **0** | 58,539 (5.71%) |
| `position` | first four FEN fields | 1,014,353 | 4,421 | 59,127 (5.83%) |
| `placement` | piece-placement field only | 1,013,462 | 4,689 | 59,368 (5.86%) |

`placement` is the decisive level: **the 12-plane encoder reads only piece
squares**, so two records with the same placement are literally the same CNN
input regardless of side to move, castling rights or move counters.

**Question 7 — duplicate FENs within one game:** zero at `exact` level (the
halfmove/fullmove counters always differ), 4,689 at `placement` level across the
whole corpus — genuine repetitions and shuffles. Negligible: 0.39% of all plies.

**Question 8 — duplicate FENs across different games:** 59,368 placements (5.86%
of distinct placements) occur in more than one game.

**Question 9 — would identical positions still cross a game-level split?**
**Yes, and this is the central design finding.** Game-level splitting prevents
*game* leakage but not *position* leakage, because different games transpose into
the same position. The magnitude depends almost entirely on the ply floor:

### Per-ply uniqueness — the measurement that sets the ply floor

| ply | games reaching it | distinct placements | uniqueness |
|---|---:|---:|---:|
| 1 | 20,058 | 20 | **0.0010** |
| 2 | 20,040 | 215 | 0.0107 |
| 4 | 19,768 | 2,226 | 0.1126 |
| 8 | 19,571 | 10,557 | 0.5394 |
| 12 | 19,281 | 16,398 | 0.8505 |
| 16 | 18,897 | 17,684 | **0.9358** |
| 20 | 18,397 | 17,502 | 0.9514 |
| 30 | 16,715 | 15,959 | 0.9548 |
| 40 | 14,383 | 13,727 | 0.9544 |

There are 20 legal first moves, so ply 1 is shared by construction. Uniqueness
rises steeply and plateaus around 0.955 from ply 20 onward — it never reaches
1.0 because transpositions and common simplified endings recur.

Taking positions from ply 1 is catastrophic: measured **49.97% placement leakage**
across a game-level split. A ply-16 floor is the knee of the curve.

---

## 3. Feasible expansion sizes (questions 6, 14, 15)

All figures measured under the game-level split of §4, one canonical row per game
unit, deduplicated at `placement` level.

| policy | total | train | test | deduped train | leakage (placement) | test after scrub | suite positions in train |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v1_one_per_game` *(today)* | 18,920 | 15,136 | 3,784 | 15,093 | 0.50% | 3,765 | 6 |
| **k=1, ply≥16** | 18,025 | 14,410 | 3,615 | 14,388 | **0.22%** | 3,606 | **0** |
| **k=2, ply≥16** | 35,583 | 28,450 | 7,133 | 28,183 | 1.35% | 7,028 | 1 |
| **k=3, ply≥16** ← ~50k | 52,567 | 42,027 | 10,540 | 41,757 | 0.91% | 10,433 | 1 |
| **k=4, ply≥16** ← recommended | **68,901** | **55,082** | **13,819** | **54,813** | **0.69%** | **13,712** | **1** |
| **k=6, ply≥16** ← ~100k | 99,124 | 79,202 | 19,922 | 78,902 | 0.49% | 19,802 | 1 |

*(k = maximum evenly-spaced positions per game; `v1_one_per_game` is scored here
under a game-level split over all 18,920 games, not the historical 10,000-game
sample.)*

**Answer to question 15: yes.** All three targets are supported:

| target | policy | measured yield |
|---|---|---:|
| ~20,000 | k=1, ply≥16 | 18,025 |
| ~50,000 | k=3, ply≥16 | 52,567 |
| ~100,000 | k=6, ply≥16 | 99,124 |

The absolute ceiling is 1,212,827 plies, but taking every ply is not a candidate:
it produces 13.04% placement leakage and consecutive near-identical positions.

### Why the ply floor matters more than k

The same k at different floors:

| policy | total | leakage | suite positions in train |
|---|---:|---:|---:|
| k=4, ply≥1 | 74,621 | **26.09%** | — |
| k=4, ply≥12 | 70,955 | 3.83% | 25 |
| **k=4, ply≥16** | 68,901 | **0.69%** | **1** |
| k=4, ply≥20 | 66,413 | 0.18% | 0 |

Moving the floor from 12 to 16 costs 2,054 positions (2.9%) and cuts leakage by
5.5× and suite contamination by 25×. Moving 16→20 buys a further 4× leakage
reduction for 3.6% of the data; §7 recommends 16 and explains the trade.

---

## 4. Game-level split design (question 4 of the deliverable)

```
games.csv rows
  → group by game CONTENT (sha256 of the moves string)   [18,920 units]
  → sort units, seeded permutation, first 20% to test    [split seed 42]
  → THEN extract positions from each side independently
  → label each position with the A2 policy
  → deduplicate within each side at placement level
  → scrub: drop from TEST any placement present in TRAIN
  → scrub: drop from BOTH any placement present in either evaluation suite
```

Measured on the real corpus:

| property | value |
|---|---|
| split unit | game move-sequence content, **not** the `id` column |
| split seed | 42 (separate from the model training seed) |
| test fraction | 0.20 |
| game units | 18,920 |
| train units / test units | **15,136 / 3,784** |
| train rows / test rows | 16,084 / 3,974 |
| **train_units ∩ test_units** | **0** |
| **train_ids ∩ test_ids** | **0** |
| game list sha256 | `b90881bf3c6245ba3634800da8879ccf8ecb9a64fb41f5116c5439983dc64e62` |
| train game sha256 | `cb1308d9af1ee45709f272492c0a22a02b8e7026f055d8ff43e1d81f881c19fb` |
| test game sha256 | `d8632c3ac89be36bced244f4370eec09c3c80b2e6a0fe841f62ca67df97809f3` |

The split is sorted before permuting, so it depends only on the **set** of game
contents — reordering `games.csv` cannot change which games are in test. This is
asserted by a test, as is the guarantee that every row sharing a content key
lands on the same side.

---

## 5. Leakage results (question 5 of the deliverable)

Under the recommended policy (k=4, ply≥16):

| guarantee | status |
|---|---|
| `train_games ∩ test_games = ∅` | **0 overlapping units, 0 overlapping ids** — measured |
| same game contributing to both sides | **impossible by construction** — positions extracted after the split |
| exact FEN in both train and test | 96 of 13,819 test rows (0.69%) |
| **position (4-field) in both** | 95 of 13,819 (0.69%) |
| **placement in both** — the level the CNN sees | **96 of 13,819 (0.69%)** |
| after scrubbing test | **0 by construction**, 13,712 test positions remain (−0.8%) |
| duplicate rows within one game | 0 selected pairs closer than the 4-ply minimum gap |

**Evaluation-suite contamination.** This is not in the brief's list but is
material, because the suites are the measurement instrument for every C6/C8 arm:

| | exact | position | placement |
|---|---:|---:|---:|
| suite positions present anywhere in `games.csv` | 130/190 | 138/190 | 138/185 |
| present at ply ≥16 | — | — | 34 |
| **present in `dataset_v1` today** | **6/190** | **6/190** | **6/185** |
| selected into train, k=4 ply≥12 | 22 | 25 | 25 |
| **selected into train, k=4 ply≥16** | **1** | **1** | **1** |

Without a ply floor the expansion would raise suite contamination from 6 to 25
positions and quietly bias C8 in its own favour. At ply≥16 it *falls* to 1, and
the scrub removes that one. **C8 would be cleaner than A2 on this axis, not
dirtier.**

---

## 6. Recommended extraction and sampling policy (question 3)

**`evenly_spaced_4_minply16`:** at most 4 positions per game, evenly spaced over
plies 16 to the end, never closer than 4 plies apart.

| property | measured |
|---|---|
| positions | 68,901 (train 55,082 / test 13,819) |
| games contributing | 18,025 of 18,920 |
| positions per game | median 4, max 4 |
| selected pairs closer than the 4-ply gap | **0** |
| placement leakage before scrub | 0.69% |
| suite positions in train | 1 (→ 0 after scrub) |

Why this one:

1. **Deterministic, no RNG.** Ply indices are a closed-form function of game
   length. No sampling seed is needed at all, which removes a reproducibility
   variable rather than documenting one.
2. **Diversity, not volume.** The 4-ply minimum gap is enforced and measured at
   zero violations, so no two selected positions are a single move apart.
3. **It fixes the two structural defects.** Phase mix goes from 100% opening to
   **27% opening / 63% middlegame / 10% endgame**; side to move goes from 95.9%
   White to **58.9% White / 41.1% Black**.
4. **It is ~7× `dataset_v1`** — a large enough change to move the result if the
   dataset really is the binding constraint, without being so large that a failure
   is unattributable.

**Phase-stratified sampling was measured and rejected.** `phase_stratified_3`
yields a comparable 117,311 positions but produced **22,314 selected pairs closer
than the minimum gap**, because buckets are filled independently and adjacent
buckets contribute adjacent plies. It would have delivered exactly the
near-identical-consecutive-positions problem the brief asks to avoid. The
evenly-spaced policies produce zero such pairs.

If a larger set is wanted, `evenly_spaced_6_minply16` (99,124) has the same
properties and slightly lower leakage. §8 recommends running both.

---

## 7. Expected label counts and Stockfish cost (questions 6, 7 of the deliverable)

### Labelling rate — measured, not extrapolated

`training/c7_label_cost_probe.py`, 300 positions, the exact `build_dataset`
configuration (depth 8, Threads=1, Hash=16MB, hash cleared per position):

| phase | n | mean ms | pos/sec | cp | mate | mate fraction | mean label |
|---|---:|---:|---:|---:|---:|---:|---:|
| opening | 100 | 6.10 | 164.0 | 98 | 2 | 2.0% | +23.6 |
| middlegame | 100 | 6.60 | 151.4 | 95 | 5 | 5.0% | +93.7 |
| endgame | 100 | 6.71 | 148.9 | 92 | 8 | 8.0% | −60.1 |
| **overall** | **300** | **6.47** | **154.5** | 285 | 15 | 5.0% | — |

Cost barely varies by phase, so the opening-only rate implied by `dataset_v1`
(187s for 9,667 positions) does transfer. **Labelling cost is not a constraint:**

| positions | projected wall clock (single process) |
|---|---|
| 20,000 | 2.2 min |
| 50,000 | 5.4 min |
| **100,000** | **10.8 min** |
| 200,000 | 21.6 min |

### Expected label composition at k=4, ply≥16

Applying the measured per-phase mate rates to the measured phase mix
(27% / 63% / 10%):

| | `dataset_v1` | projected C8 |
|---|---:|---:|
| cp labels | 98.1% | ≈95.4% |
| mate labels | 1.9% | **≈4.6%** (≈3,170 of 68,901) |
| clipped labels | 1 | small; 0 of 300 probe positions clipped |
| endgame positions | **0** | ≈6,890 |
| Black-to-move positions | 407 | **≈28,320** |

Roughly **70× more Black-to-move positions and 2.4× the mate-label rate**, which
is the point of the exercise.

---

## 8. Exact C8 experiment design (question 8)

### What changes

**Only the training dataset.** Everything else is pinned to A2.

| held constant | value |
|---|---|
| architecture | A2 Sequential, 12 piece planes, `(8,8,12)` input, 2,360,129 params |
| representation | `planes12` — unchanged, engine-loadable |
| label policy | `corrected_mate_white_perspective` — **byte-identical rules** |
| loss / optimiser | Huber, Adam, LR 1e-3, batch 64 |
| callbacks | ReduceLROnPlateau (0.5, patience 5), EarlyStopping (patience 10, restore best) |
| epoch cap | 100 |
| seeds | 0, 1, 2 |
| evaluation | `evaluation/evaluate.py`, unmodified |
| Stockfish | 17.1, depth 8, Threads=1, Hash=16MB, Clear Hash per position |
| suites | `extended` (160), `phase0_52` (52) |
| fusion | **frozen production Ridge**, initially |

Because the representation stays `planes12`, C8 needs **no evaluation shim** —
unlike A3/A13/A13R/A14, it runs through `evaluation/evaluate.py` directly, the
same path A0/A1/A2 used.

### Arms

| arm | dataset | positions | purpose |
|---|---|---:|---|
| **A2** | `dataset_v1` | 9,667 | existing control, already trained — not re-run |
| **C8a** | `dataset_v2` k=4 ply≥16 | 68,901 | the primary expansion arm |
| **C8b** | `dataset_v2-100k` k=6 ply≥16 | 99,124 | does more data keep helping, or saturate? |

C8b is worth running only because labelling is 11 minutes; if that changes, drop it.

### The confound that must be stated, not hidden

C8a differs from A2 in **two** ways at once: dataset *size* (9,667 → 68,901) and
dataset *composition* (opening-only → phase-balanced; 96% White → 59% White). A
gain cannot be attributed to size alone.

Two options, in preference order:

1. **Accept and state it.** The composition change is not a nuisance variable —
   it is the *hypothesis*. "The training distribution does not match the
   evaluation distribution" is the thing being fixed.
2. **Add a size-matched control** if attribution is later required: a
   `dataset_v2-9667` drawn from the same policy and split, subsampled
   deterministically to A2's exact size. That isolates composition from size at
   the cost of one more 3-seed run. **Not proposed now** — it is only worth
   running if C8a improves and the mechanism matters.

### Decision rule (pre-registered)

Compare C8a against A2 on both suites, paired by seed, against the **A0 three-seed
noise band**, exactly as C6 did. A change counts only if it exceeds the band and
is consistent across all three seeds. Report the White/Black split separately —
that is where the largest effect is predicted.

---

## 9. Reproducibility contract

Everything below is emitted into `dataset_v2.manifest.json` by the builder.

| artifact | definition |
|---|---|
| source hash | sha256 of `games.csv` = `e7aadff1…a4a8129` |
| game-list hash | sha256 of the sorted unique content keys = `b90881bf…3dc64e62` |
| train-game hash | sha256 of the sorted train content keys = `cb1308d9…f881c19fb` |
| test-game hash | sha256 of the sorted test content keys = `d8632c3a…f97809f3` |
| split seed | 42 |
| test fraction | 0.20 |
| extraction policy | `evenly_spaced`, k=4, `min_ply`=16, `min_gap`=4 |
| **sampling seed** | **none required** — the policy is closed-form and deterministic |
| position-record hash | sha256 of the emitted `.jsonl` |
| label hash | sha256 of the label vector, per split side |
| label policy | `corrected_mate_white_perspective`, `training/labels.py` unchanged |
| pipeline version | `c7-1` |

Rebuild determinism holds for a fixed `games.csv`, Stockfish binary/version and
configuration, exactly as `c6prep-1` does.

### Label control — no bug found

The A2 policy was re-verified against later-game inputs it has never seen
(endgames, mates at distance, mate-0, both perspectives, clipping at ±1500).
All cases behave as documented: mate 3 → ±1970, mate 0 with checkmate → ±2000
by side to move, cp ±1800 → ±1500 clipped, perspective negation correct.

**No mate-mapping, perspective, clipping or scale change is proposed.**
`training/labels.py` is not modified by C7 or C8.

---

## 10. Risks and limitations

1. **The source is one population.** 20,058 Lichess games with a median length of
   55 plies. Expanding within it increases *quantity* and *phase coverage* but not
   *provenance diversity* — the openings, rating range and time controls are
   whatever this file contains. C8 cannot answer "would better games help".
2. **Ratings are not stratified.** `white_rating`/`black_rating` exist and are
   unused by both the current and proposed policies. A dataset dominated by one
   skill band teaches that band's mistakes. Measured only as available, not acted on.
3. **Residual transposition leakage is irreducible.** Game-level splitting cannot
   prevent two different games reaching the same position. At ply≥16 this is 0.69%
   pre-scrub and 0 post-scrub, but scrubbing is not free: it removes precisely the
   *most common* positions from test, so the scrubbed test set is very slightly
   biased toward unusual positions.
4. **The phase classifier is an audit heuristic.** `endgame ≤12 pieces; else
   opening ≤ply 20; else middlegame`. It is reproducible and used only for
   reporting and for the rejected phase-stratified policy — never for labels.
5. **C8a confounds size with composition** (§8). Stated, not hidden.
6. **The frozen production Ridge still caps every arm.** Its coefficients were
   fitted to the *original* CNN's output scale. This biases all arms identically
   so the comparison stays fair, but no C8 arm should be read as "the best this
   architecture can do" until a matched-fusion refit is run.
7. **More data may not help.** C6 established that the architecture ignores extra
   inputs; it did not establish that the architecture has spare capacity. A
   2.36M-parameter CNN on 55,082 positions may simply fit the same function
   better without ranking moves better — the engine metric is what decides, and
   it is only loosely coupled to test Huber.
8. **The evaluation suites stay small.** 160 and 52 positions with three seeds.
   The A0 noise band is wide relative to plausible effects; C7 does not change that.

---

## 11. Verdict: is the dataset sufficiently expandable to justify C8?

**Yes — with higher confidence than for any prior C6 arm.**

| criterion | finding |
|---|---|
| Is there materially more data? | **Yes.** 1,212,827 plies available; 9,667 used (0.8%). |
| Can it be extracted without leakage? | **Yes.** Game-level split by content, 0 unit overlap; 0.69% position leakage at ply≥16, 0 after scrub. |
| Is the cost acceptable? | **Yes.** 154.5 positions/sec — 68,901 positions in ≈7.5 minutes. |
| Is there a concrete defect to fix? | **Yes, two.** Zero endgame positions; 95.9% White to move with an unrepresentative Black sample. |
| Would it compromise evaluation? | **No — it improves it.** Suite contamination falls from 6 positions to 1, then 0 after scrub. |
| Does it need a production change? | **No.** `planes12` stays; C8 runs on the unmodified evaluator. |

The C6 programme showed the model ignores what it is given. C7 shows it has never
been given the positions it is judged on.

### Recommended next step

Build `dataset_v2` under the §4 split and §6 policy, verify the manifest hashes,
then run C8a (3 seeds) against A2. Add C8b if C8a moves. Do not change labels,
architecture, representation, recipe or Ridge.

---

## Reproducing

```bash
# full audit (~5 min; replays all 20,058 games)
python -m training.c7_dataset_audit --out training/artifacts/c7_audit.json

# labelling cost and label mix by phase (needs Stockfish)
python -m training.c7_label_cost_probe --per-phase 100

# tests
python -m pytest tests/unit/test_training_c7_audit.py -q
```

Both scripts are read-only: no dataset is written, no model is trained, and no
production file is touched.
