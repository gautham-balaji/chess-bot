# Evaluation

How the engine's playing quality is measured against Stockfish, and what the
current numbers are.

> **This is not the test suite.** `pytest` asks *"does the implementation honour
> its software contracts?"* — pass/fail. This harness asks *"how good are the
> moves?"* — a measurement on a distribution, with no pass/fail answer. See
> [`TESTING.md`](TESTING.md).

---

## Methodology

### What is being evaluated

For each position in a fixed suite, the engine chooses a move and Stockfish
chooses a move. Stockfish then evaluates **both resulting positions** under the
same configuration, and the difference is how much the engine's choice cost.

The engine's own score is never compared against Stockfish's. It is not in
centipawns (it is a Ridge-weighted sum of a `tanh`-squashed CNN output plus raw
feature counts, with the Ridge intercept dropped), so subtracting it from a
centipawn value would be a unit error. The pre-Phase-1 code did exactly that;
this harness does not.

### Datasets

| Dataset | n | Purpose |
|---|---:|---|
| [`phase0_52.json`](../evaluation/positions/phase0_52.json) | 52 | Frozen verbatim copy of the Phase 0 suite. Enables like-for-like comparison with the original baseline |
| [`extended.json`](../evaluation/positions/extended.json) | 160 | Larger suite built in Phase 3 |

`phase0_52.json` is a byte-faithful copy of `baseline/fens.json`
(sha256 `ea526081…`), stored under `evaluation/` so the harness never reads or
writes the historical baseline directory.

The extended suite is built by
[`build_extended_suite.py`](../evaluation/build_extended_suite.py) and rebuilds
byte-identically. Opening and middlegame positions come from **32 named opening
lines** truncated at plies 6, 7, 11 and 12; each record stores its full move
list, so provenance is self-verifying — anyone can replay it. Endgame and
tactical positions are hand-constructed literal FENs described by their material
("rook and pawn vs rook"), not by theory names that would be an unverified claim.
Every position is validated (`is_valid()`, non-terminal, no duplicate FEN); the
builder dropped 5 of my own candidates for exactly those reasons.

Composition: opening 60, middlegame 64, endgame 21, tactical 10, defensive 5;
85 White to move, 75 Black.

> **Neither suite is statistically representative.** They are not sampled from
> master games or any tactics database. They are fixed comparison sets. No
> population is being estimated, so no confidence interval is implied and no
> significance is claimed.

### Stockfish configuration

| Setting | Value |
|---|---|
| Version | Stockfish 17.1 |
| Depth | 8 (matches Phase 0 and the CNN's training-label depth) |
| Threads | 1 (pinned — multi-threaded search is nondeterministic) |
| Hash | 16 MB (pinned — table size changes results) |
| MultiPV (agreement) | 3 (matches Phase 0 exactly) |
| MultiPV (rank correlation) | 8 |
| **Clear Hash** | **before every position** |

One Stockfish process is reused for the whole run and closed via `try/finally`.

**Why hash clearing matters.** A shared transposition table makes a fixed-depth
search *order-dependent*: entries written while analysing position N change the
search at position N+1. Measured on the 52-position suite, the Phase 0 call
sequence and this harness's longer sequence produced **different Stockfish best
moves on 19 of 52 positions** with a shared hash. With the hash cleared per
position, the two sequences agree on **52/52**. Clearing makes the reference
depend only on the position, which is what a reference must do.

### Engine version

Whatever is committed at run time; each result file records the git commit and
whether the tree was dirty. No engine behaviour was changed in this phase.

### POV convention

**One convention, everywhere:** every evaluation is expressed from the
perspective of the side to move in the *original* position — the player choosing
the move. Positive centipawns are good for that player.

After the move it is the opponent's turn, so a raw Stockfish score is relative to
the opponent. Conversion happens exactly once, at a single function
(`metrics.score_from_pov`), so POV handling cannot drift. It is covered by 9
dedicated unit tests including sign-flip and self-inverse checks.

### Regret calculation

```
move_regret_cp = eval_after_stockfish_move − eval_after_engine_move
```

both from the same Stockfish configuration, both converted to the mover's POV.

- `0` — the engine's move is as good as the reference (exactly 0 when it *is* the
  reference move)
- `> 0` — centipawns given up
- `< 0` — the engine's move scored better than Stockfish's own choice. Possible at
  fixed shallow depth through search instability. **Not clamped** — clamping would
  silently bias the mean upward.

### Mate handling

A mate score is an ordinal, not a centipawn quantity. Mapping it onto a
centipawn axis (e.g. `mate_score=10000`) would corrupt every aggregate, so
**mate-involved positions are excluded from all centipawn statistics and reported
by count instead.** Each is labelled:

| `mate_status` | Meaning |
|---|---|
| `none` | Both evaluations are centipawns; regret is defined |
| `missed_forced_mate` | Stockfish's move forces mate; the engine's does not |
| `engine_move_allows_forced_mate` | The engine's move walks into a forced mate |
| `engine_found_mate_reference_did_not` | The engine's move forces mate, the reference's does not |
| `both_forced_mate_for_mover` | Both moves force mate |
| `both_moves_lose_to_forced_mate` | Position is lost by force either way — not the engine's fault |

Positions whose resulting position is already terminal are resolved directly
rather than sent to Stockfish (checkmate → mate delivered; stalemate/draw → 0 cp).

### Blunder threshold

`regret > 300 cp`. **A measurement convention for this report, not a claim about
universal chess truth.** The denominator is positions with a *defined* regret.

---

## Results

Both runs below are fully reproducible — see [Reproducibility](#reproducibility).

### `phase0_52` (52 positions)

| Metric | Value | n |
|---|---:|---:|
| Legality rate | **100%** | 52/52 |
| Stockfish top-1 move agreement | **19.23%** | 10/52 |
| Top-3 containment | **32.69%** | 17/52 |
| Mean move regret | **107.51 cp** | 45 |
| Median move regret | **45 cp** | 45 |
| p95 move regret | **419 cp** | 45 |
| Min / max regret | −24 / 614 cp | 45 |
| Blunder rate (>300 cp) | **15.56%** | 7/45 |
| Regret coverage | 86.54% | 45/52 |
| Spearman rho (mean / median) | 0.24 / 0.37 | 52 |
| Stockfish latency p50 | 9.09 ms | 52 |

Chance reference: a uniformly random legal move would match Stockfish's top move
**5.91%** of the time and land in its top 3 **17.72%** of the time on this suite
(mean over positions of `min(N, legal_moves)/legal_moves`). Top-1 agreement is
therefore about **3.3× chance** and top-3 about **1.8× chance**.

Mate breakdown: 45 `none`, **4 `missed_forced_mate`**, **3
`engine_move_allows_forced_mate`**.

### `extended` (160 positions)

| Metric | Value | n |
|---|---:|---:|
| Legality rate | **100%** | 160/160 |
| Stockfish top-1 move agreement | **18.12%** | 29/160 |
| Top-3 containment | **33.75%** | 54/160 |
| Mean move regret | **144.07 cp** | 152 |
| Median move regret | **25 cp** | 152 |
| p95 move regret | **533 cp** | 152 |
| Min / max regret | −84 / 676 cp | 152 |
| Blunder rate (>300 cp) | **24.34%** | 37/152 |
| Spearman rho (mean / median) | 0.22 / 0.30 | 160 |

Chance reference on this suite: top-1 4.31%, top-3 12.93%.
Mate breakdown: 152 `none`, 4 `missed_forced_mate`, 4 `engine_move_allows_forced_mate`.

The two suites agree closely on top-1 (19.2% vs 18.1%), top-3 (32.7% vs 33.8%)
and legality (100%), which is mild evidence the 52-position figures were not a
fluke of that particular set.

### The mean/median gap

On both suites the **mean regret is far above the median** (107.5 vs 45;
144.1 vs 25). The distribution is heavily right-skewed: most moves are
near-reference, and a minority of large errors drag the mean. The median is the
better summary of typical play; the blunder rate is the better summary of the
tail. Reporting only the mean would overstate typical error, and reporting only
the median would hide the tail.

### By category

Small groups — descriptive only, no significance implied, no ranking intended.

**phase0_52**

| Category | n | Top-1 | Mean regret |
|---|---:|---:|---:|
| endgame | 12 | 41.67% | 12.42 |
| tactical | 6 | 16.67% | 0.00 |
| defensive | 4 | 25.00% | 84.00 |
| opening | 18 | 11.11% | 96.67 |
| middlegame | 12 | 8.33% | 231.75 |

**extended**

| Category | n | Top-1 | Mean regret |
|---|---:|---:|---:|
| defensive | 5 | 60.00% | 0.00 |
| endgame | 21 | 52.38% | 39.71 |
| middlegame | 64 | 12.50% | 185.30 |
| opening | 60 | 11.67% | 145.30 |
| tactical | 10 | 0.00% | 122.00 |

Both suites show the same ordering: endgames cheapest, middlegames most
expensive. Endgame agreement is partly a chance artefact (fewer legal moves), but
the *regret* figure is not subject to that inflation and points the same way.

### By side to move

| Suite | Side | n | Top-1 | Mean regret | Median regret |
|---|---|---:|---:|---:|---:|
| phase0_52 | White | 30 | 26.67% | 122.52 | 29 |
| phase0_52 | Black | 22 | 9.09% | 85.00 | 46 |
| extended | White | 85 | 17.65% | 154.64 | 19 |
| extended | Black | 75 | 18.67% | 132.01 | 29 |

On the 52-position suite White's top-1 agreement is roughly 3× Black's, which is
consistent with the known Black-side bonus-sign defect. **The extended suite does
not reproduce that gap** (17.65% vs 18.67%), so the 52-position split is more
likely small-sample noise than a measured effect. Median regret is worse for
Black on both suites, but by a small margin. This harness does not establish the
Black-side defect — the dedicated unit test in Phase 2 does.

### Rank correlation

Spearman rho between the engine's ranking and Stockfish's ranking of the **same**
candidate moves (Stockfish's top 8), computed on ranks only, so the engine's
non-centipawn scale is irrelevant.

Mean 0.24, median 0.37 on `phase0_52`; mean 0.22, median 0.30 on `extended`. The
full range spans −1.0 to 0.98: the engine orders moves roughly like Stockfish
more often than not, but weakly and inconsistently.

---

## Phase 0 comparison

Only like-for-like measurements are compared.

| Metric | Phase 0 | Phase 3 (`phase0_52`) | Comparable? |
|---|---:|---:|---|
| Legality | 100% (52/52) | 100% (52/52) | Yes |
| Top-1 agreement | 19.23% (10/52) | 19.23% (10/52) | Yes, with a caveat below |
| Top-3 containment | 36.54% (19/52) | 32.69% (17/52) | Reference changed |
| Median engine latency | 638 ms | see below | **No** |
| p95 engine latency | 1005 ms | see below | **No** |
| Mean move regret | not measured | 107.51 cp | New in Phase 3 |
| Blunder rate | not measured | 15.56% | New in Phase 3 |

### Top-1: same number, not quite the same positions

Both phases report 10/52. But the *sets* overlap on only **9 of 10**: Phase 3
gains `DF02` and loses `MG11`. The identical count is partly coincidence. This is
caused by the hash-clearing fix — Stockfish's own best move differs between the
two phases on 21 of 52 positions, because Phase 0's reference was itself
order-dependent.

Phase 0's numbers were **verified reproducible under Phase 0's own call
sequence** (0/52 differ when that sequence is replayed). Phase 0 was not wrong;
it was sequence-dependent. Phase 3's reference is position-independent, which is
the stronger property.

Top-3 containment moved from 36.54% to 32.69% for the same reason — the reference
lists changed, not the engine.

### Latency is not comparable

**Engine latency is the one metric that did not reproduce.** Measurements of
identical code on the same machine within a single session ranged from a median
of **626 ms to 1305 ms**:

| Measurement | Median | p95 |
|---|---:|---:|
| Phase 0 (isolated) | 638 ms | 1005 ms |
| Phase 3, early harness run | 641 ms | 1108 ms |
| Phase 3, later harness runs | 1172–1305 ms | 2881–4033 ms |
| Phase 3, isolated 12-position trials ×3 | 622 / 626 / 643 ms | 754–822 ms |

The engine is provably unchanged: Phase 2's regression suite confirms all 52
baseline positions return identical moves and identical scores. The variation is
host state (background load and thermal behaviour during a long session), not
code. **Do not read a latency change between runs or phases as an engine change.**

Stockfish latency (median ~9–11 ms) is reported separately. No "speedup" headline
is produced, because the two are not measured under controlled identical
conditions.

---

## Reproducibility

The harness was run twice end-to-end on `phase0_52`.

**Every quality field was identical across both runs** — 0 differences across all
52 positions on: engine move, Stockfish best move, Stockfish top-3, regret, mate
status, top-1 agreement, top-3 containment, Spearman rho, and both POV-converted
evaluations. Aggregate metrics were byte-identical excluding latency.

Latency differed and is documented above as host-dependent.

Both dataset files rebuild byte-identically (verified by sha256).

---

## Harness validation

The harness's own mathematics is covered by **58 unit tests** in
[`tests/unit/test_evaluation_metrics.py`](../tests/unit/test_evaluation_metrics.py),
running in under a second. They test arithmetic against hand-computed answers,
not chess strength:

- **POV conversion** — sign flips for Black, self-inverse property, mate scores
  never leaking into the centipawn field
- **Regret** — a 5-case arithmetic table, zero for identical evaluations,
  negative values preserved
- **Mate taxonomy** — all six statuses, including "mate already on the board"
- **Agreement / containment** — hit, miss, empty list, missing move
- **Chance baseline** — reciprocal mean, top-N capping at the legal-move count
- **Aggregation** — percentile indexing (including the banker's-rounding tie),
  order independence, empty input returning nulls not exceptions
- **Blunder threshold** — strict `>` at exactly 300
- **Spearman** — identical, reversed, constant and length-mismatched inputs
- **Two worked end-to-end examples** with regret computed by hand

---

## Limitations

1. **Suite size.** 52 and 160 positions. No confidence intervals; differences of a
   few percentage points are not meaningful.
2. **Provenance.** Neither suite is sampled from real games. Opening/middlegame
   positions come from named opening lines chosen by hand; endgame and tactical
   positions are constructed. They span phases by design, not by sampling.
3. **No statistical significance is claimed anywhere**, including every category
   breakdown. Categories with n = 4–10 are descriptive only.
4. **Stockfish is a reference, not ground truth.** Depth 8 is shallow. A
   disagreement is not automatically an engine error, and regret measured at
   depth 8 is not regret against perfect play.
5. **Regret is defined for 86–95% of positions.** Mate-involved positions are
   excluded from centipawn aggregates by design. The 4 missed forced mates and
   3–4 mate-allowing moves are real quality failures that the mean regret does
   **not** include.
6. **Latency is hardware- and host-state-dependent** and did not reproduce within
   a single session. Treat it as indicative only.
7. **Engine-vs-Stockfish is not an objective chess-strength measure.** No Elo
   estimate is produced. These numbers describe agreement and centipawn loss on
   two fixed sets, nothing more.
8. **Negative regret is retained**, so the mean is slightly lower than a
   clamped-at-zero convention would give. This is deliberate.
9. **The engine's non-centipawn score is never compared to Stockfish's.** Any
   metric requiring that comparison is absent by design, not by oversight.

---

## Running it

```bash
# 52-position suite (~2-3 min)
python evaluation/evaluate.py --dataset evaluation/positions/phase0_52.json

# 160-position suite (~9 min)
python evaluation/evaluate.py --dataset evaluation/positions/extended.json

# custom output location / quick smoke run
python evaluation/evaluate.py --dataset evaluation/positions/phase0_52.json \
    --out-prefix evaluation/results/myrun --limit 5

# rebuild the extended dataset (deterministic)
python evaluation/build_extended_suite.py evaluation/positions/extended.json

# validate the harness mathematics
pytest tests/unit/test_evaluation_metrics.py
```

Requires a resolvable Stockfish binary — see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Each run writes a `.json` (full
per-position detail) and a `.md` (human-readable summary) under
`evaluation/results/`.
