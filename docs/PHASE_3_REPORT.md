# Phase 3 Report — Reproducible AI Evaluation Harness

**Objective:** measure the engine's actual playing quality against Stockfish,
without invalid score subtraction.

**Result:** a reproducible harness producing move-quality regret, agreement,
blunder rate and rank correlation on two fixed suites. All quality metrics
reproduce exactly across runs. **No engine code was changed.** Full test suite:
261 passed, 11 xfailed.

---

## 1. What was implemented

| File | Purpose |
|---|---|
| `evaluation/metrics.py` | Pure math: POV conversion, regret, mate taxonomy, aggregation, Spearman. No engine calls, no I/O |
| `evaluation/evaluate.py` | Runner: one Stockfish process, per-position measurement, JSON + Markdown output |
| `evaluation/build_extended_suite.py` | Deterministic builder for the 160-position suite |
| `evaluation/positions/phase0_52.json` | Frozen verbatim copy of the Phase 0 suite |
| `evaluation/positions/extended.json` | 160-position suite |
| `evaluation/results/*.json` / `*.md` | Machine- and human-readable results |
| `tests/unit/test_evaluation_metrics.py` | 58 tests for the harness's own arithmetic |
| `docs/EVALUATION.md` | Methodology, results, limitations |

The split matters: all arithmetic lives in `metrics.py` with no dependencies, so
it is fully unit-testable in under a second. The runner only orchestrates.

---

## 2. Evaluation methodology

For each position, the engine picks `M_engine` and Stockfish picks `M_sf`. The
**same** Stockfish configuration then evaluates both resulting positions, both
converted to the mover's POV:

```
move_regret_cp = eval_after_stockfish_move − eval_after_engine_move
```

The engine's own score never enters the calculation — it is not in centipawns
(Ridge-weighted `tanh`-squashed CNN output plus raw feature counts, intercept
dropped), so subtracting it from a Stockfish value would be a unit error. The
pre-Phase-1 `eval_diff_cp` field did exactly that.

Negative regret is **retained, not clamped**; clamping would bias the mean upward.

---

## 3. Dataset

| Dataset | n | Composition |
|---|---:|---|
| `phase0_52` | 52 | Byte-faithful copy of `baseline/fens.json` (sha256 `ea526081…`) |
| `extended` | 160 | opening 60, middlegame 64, endgame 21, tactical 10, defensive 5; 85 W / 75 B |

The Phase 0 suite was **preserved, not replaced** — it is what makes the Phase 0
comparison like-for-like. It is copied under `evaluation/` so the harness never
touches `baseline/`.

The extended suite is built from **32 named opening lines** truncated at plies
6, 7, 11, 12 (chosen 2-and-2 so side-to-move stays balanced), plus hand-built
endgame and tactical FENs. Each opening record stores its full move list, so
provenance is self-verifying. Every position is validated; the builder rejected 5
of my own candidates (pawn on rank 1, adjacent kings, same-coloured bishops with
insufficient material) and 5 duplicate transpositions. Rebuilds byte-identically.

**Neither suite is statistically representative**, and the files say so in their
own metadata.

---

## 4. Stockfish configuration

| Setting | Value |
|---|---|
| Version | Stockfish 17.1 |
| Depth | 8 |
| Threads | **1** (explicitly pinned) |
| Hash | **16 MB** (explicitly pinned) |
| MultiPV | 3 for agreement (matches Phase 0), 8 for rank correlation |
| Clear Hash | **before every position** |
| Process | one, reused, closed via `try/finally` |

Phase 0 noted the application set no UCI options at all. This harness sets them
explicitly. This is the **harness's** configuration; gameplay settings in
`app.py`/`config.py` were not touched.

---

## 5. Metrics

Legality rate · Stockfish top-1 move agreement · top-3 containment · move regret
(min/median/mean/p95/max) · blunder rate (>300 cp) · mate-status counts · regret
coverage · Spearman rank correlation · engine and Stockfish latency, reported
separately. Every rate carries its numerator and denominator.

A **chance baseline** is reported alongside agreement (`mean(min(N, legal)/legal)`),
because an agreement rate is uninterpretable without one — endgames have few legal
moves and inflate agreement for trivial reasons.

No "speedup" headline is produced.

---

## 6. POV convention

Every evaluation is from the perspective of the side to move in the **original**
position. Positive centipawns are good for that player. Conversion happens once,
in `metrics.score_from_pov`, covered by 9 dedicated tests (sign flip, self-inverse,
mate never leaking into the cp field).

---

## 7. Mate handling

A mate score is an ordinal, not a centipawn quantity, so **mate-involved positions
are excluded from every centipawn aggregate** and counted separately under six
explicit `mate_status` labels. Terminal resulting positions are resolved directly
rather than sent to Stockfish (checkmate → mate delivered; draw → 0 cp).

Observed: `phase0_52` — 45 `none`, 4 `missed_forced_mate`, 3
`engine_move_allows_forced_mate`. `extended` — 152 / 4 / 4.

---

## 8. Actual results

### `phase0_52` (n=52)

| Metric | Value | n |
|---|---:|---:|
| Legality | **100%** | 52/52 |
| Top-1 agreement | **19.23%** | 10/52 |
| Top-3 containment | **32.69%** | 17/52 |
| Mean regret | **107.51 cp** | 45 |
| Median regret | **45 cp** | 45 |
| p95 regret | **419 cp** | 45 |
| Min / max regret | −24 / 614 cp | 45 |
| Blunder rate (>300cp) | **15.56%** | 7/45 |
| Regret coverage | 86.54% | 45/52 |
| Spearman (mean/median) | 0.24 / 0.37 | 52 |
| Stockfish latency p50 | 9.09 ms | 52 |

Chance: top-1 5.91%, top-3 17.72% → agreement is ~3.3× chance.

### `extended` (n=160)

| Metric | Value | n |
|---|---:|---:|
| Legality | **100%** | 160/160 |
| Top-1 agreement | **18.12%** | 29/160 |
| Top-3 containment | **33.75%** | 54/160 |
| Mean regret | **144.07 cp** | 152 |
| Median regret | **25 cp** | 152 |
| p95 regret | **533 cp** | 152 |
| Blunder rate | **24.34%** | 37/152 |
| Spearman (mean/median) | 0.22 / 0.30 | 160 |

The two suites agree closely on top-1, top-3 and legality — mild evidence the
52-position figures were not a fluke of that set.

---

## 9. Phase 0 comparison

| Metric | Phase 0 | Phase 3 | Comparable? |
|---|---:|---:|---|
| Legality | 100% (52/52) | 100% (52/52) | Yes |
| Top-1 agreement | 19.23% (10/52) | 19.23% (10/52) | Yes, with caveat |
| Top-3 containment | 36.54% (19/52) | 32.69% (17/52) | Reference changed |
| Median latency | 638 ms | **not comparable** | No |
| p95 latency | 1005 ms | **not comparable** | No |
| Mean regret | not measured | 107.51 cp | New |
| Blunder rate | not measured | 15.56% | New |

**The identical 10/52 is partly coincidence.** The agreeing sets overlap on only
9 of 10 — Phase 3 gains `DF02`, loses `MG11` — because the reference itself
changed (see §16).

---

## 10. Reproducibility results

The harness was run twice end-to-end on `phase0_52`.

**Zero differences across all 52 positions** on: engine move, Stockfish best move,
Stockfish top-3, regret, mate status, top-1 agreement, top-3 containment,
Spearman rho, and both POV-converted evaluations. Aggregates byte-identical
excluding latency.

Both dataset builders produce byte-identical files (sha256 verified).

**Latency did not reproduce** — see §16.

---

## 11. Harness validation tests

**58 tests, all passing, 0.21s** (`tests/unit/test_evaluation_metrics.py`).
They test arithmetic against hand-computed answers, not chess strength: POV
conversion and sign flips, a 5-case regret table, all six mate statuses,
agreement/containment edge cases, chance-baseline maths, percentile indexing
(including the banker's-rounding tie), strict blunder threshold, Spearman
degenerate inputs, and two worked end-to-end examples computed by hand.

One test initially failed — my expectation was wrong about `round(4.5)` returning
4 under banker's rounding, not the implementation. Fixed the test and documented
the tie behaviour, since it matches Phase 0's percentile and keeps the phases
comparable.

Full suite after Phase 3: **261 passed, 11 xfailed, 3m02s**.

---

## 12. Runtime

| Run | Wall clock |
|---|---:|
| `phase0_52` (52 positions) | 142–157 s |
| `extended` (160 positions) | 544 s |
| Metrics unit tests | 0.21 s |
| Full pytest suite | 182 s |

Per position: 1 `engine_move` (timed) + 1 `rerank_moves` (for rank correlation,
untimed) + 4–5 Stockfish calls. Stockfish is cheap (~9–11 ms median); the engine
dominates.

---

## 13. Known limitations

1. 52 and 160 positions — no confidence intervals, no significance claimed
   anywhere, including every category breakdown (some n = 4–10).
2. Neither suite is sampled from real games; provenance is "named opening lines"
   and "constructed FENs", stated as such.
3. Stockfish depth 8 is a **reference, not ground truth**. Disagreement ≠ error.
4. Regret is undefined for 7–8 positions per suite (mate involved). Those include
   4 missed forced mates — real failures the mean regret does not capture.
5. Latency is host-dependent and did not reproduce within one session.
6. No Elo estimate, no claim of objective chess strength.
7. Negative regret retained, so the mean is slightly below a clamped convention.

---

## 14. Files changed

### Added
```
evaluation/__init__.py
evaluation/metrics.py
evaluation/evaluate.py
evaluation/build_extended_suite.py
evaluation/positions/phase0_52.json
evaluation/positions/extended.json
evaluation/results/phase0_52_run1.json / .md
evaluation/results/phase0_52_run2.json / .md
evaluation/results/extended_run1.json / .md
tests/unit/test_evaluation_metrics.py
docs/EVALUATION.md
docs/PHASE_3_REPORT.md
```

### Modified
**None.**

### Deleted
**None.** `baseline/` untouched — verified by sha256.

---

## 15. Exact commands

```bash
# datasets
python evaluation/build_extended_suite.py evaluation/positions/extended.json

# evaluation runs
python evaluation/evaluate.py --dataset evaluation/positions/phase0_52.json \
    --out-prefix evaluation/results/phase0_52_run1
python evaluation/evaluate.py --dataset evaluation/positions/phase0_52.json \
    --out-prefix evaluation/results/phase0_52_run2
python evaluation/evaluate.py --dataset evaluation/positions/extended.json \
    --out-prefix evaluation/results/extended_run1

# validation
pytest tests/unit/test_evaluation_metrics.py -q
pytest -q

# reproducibility diff (run1 vs run2, per-position, 10 fields)
# isolated latency trials (no Stockfish process)
```

---

## 16. Unexpected findings

### A. Stockfish at fixed depth is order-dependent with a shared hash

The single most important methodological finding of this phase.

Phase 3's first runs disagreed with Phase 0 on Stockfish's **own best move** in
22 of 52 positions — same binary, same version, same depth, same Threads, same
Hash. A controlled four-way experiment isolated the cause:

| Variant | vs recorded Phase 0 |
|---|---:|
| A: Phase 0 call sequence, no hash clear | **0/52 differ** |
| B: Phase 3 call sequence, no hash clear | 19/52 differ |
| C: Phase 3 call sequence, **with** hash clear | 21/52 differ |
| D: Phase 0 call sequence, **with** hash clear | 21/52 differ |
| A vs B (sequence effect, no clear) | **19/52 differ** |
| C vs D (sequence effect, with clear) | **0/52 differ** |

Conclusions:

- **Phase 0 is exactly reproducible** under its own call sequence (A: 0/52). It
  was not wrong — it was sequence-dependent.
- Entries written while analysing position N change the fixed-depth search at
  position N+1. Adding analyse calls changed 19/52 reference moves.
- **With `Clear Hash` per position, the call sequence stops mattering** (C vs D:
  0/52). The reference depends only on the position.

The harness now clears the hash per position. This shifted top-3 containment from
40.38% (uncleared) to 32.69% (cleared) — a change in the *reference*, not the
engine.

An initial hypothesis that lingering MultiPV was responsible was **disproved**:
python-chess resets MultiPV to its default before every `play()`
(`chess/engine.py:1617`).

### B. Engine latency did not reproduce; the engine did not change

Median engine latency for identical code on the same machine in one session:

| Measurement | Median | p95 |
|---|---:|---:|
| Phase 0 (isolated) | 638 ms | 1005 ms |
| Phase 3, early run | 641 ms | 1108 ms |
| Phase 3, later runs | 1172–1305 ms | 2881–4033 ms |
| Phase 3, isolated trials ×3 | 622 / 626 / 643 ms | 754–822 ms |

Initially attributed to Stockfish contention; an isolated measurement with no
Stockfish process also showed ~1260 ms, disproving that. Three clean repeat
trials then returned 622–643 ms, matching Phase 0. The cause is host state
(background load / thermal behaviour during a long session), not code — Phase 2's
regression suite independently confirms all 52 positions return identical moves
and scores.

**Latency must not be compared across phases.** The harness's Markdown output now
says so explicitly.

### C. The regret distribution is heavily right-skewed

Mean 107.5 vs median 45 (`phase0_52`); mean 144.1 vs median 25 (`extended`).
Typical moves are close to the reference; a minority of large errors dominate the
mean. Reporting only the mean would overstate typical error; only the median
would hide the tail. Both are reported, plus the blunder rate.

### D. Middlegames are the most expensive phase

Consistent across both suites: mean regret 231.75 (phase0_52) and 185.30
(extended) in middlegames, versus 12.42 / 39.71 in endgames. Consistent with the
CNN having been trained **only on ply-20 positions** — but this harness does not
establish causation, and the category samples are small.

### E. The Black-side gap did not replicate

On `phase0_52`, White's top-1 agreement (26.67%) is ~3× Black's (9.09%), which
looks like confirmation of the known Black-side bonus-sign defect. **The extended
suite does not reproduce it** (17.65% W vs 18.67% B, n=85/75). The 52-position
split is therefore more likely small-sample noise than a measured effect.

This is worth stating plainly: it would have been easy to report the 52-position
split as evidence for a bug that is independently known to exist. The larger
sample does not support it. The Phase 2 unit test remains the actual evidence for
that defect.

---

## 17. Deferred to Phase 4

No engine bug was fixed. Unchanged: the Black-side heuristic bonus sign (C1), the
1-ply lookahead extremum (C2), the dropped Ridge intercept (C3), the pre/post-move
`center` inconsistency (C4), the White-only `opening_center_bonus` (C5), the
side-to-move / castling / en-passant encoding gaps (C6), the push-before-validate
board corruption, the 415 non-JSON response, `/forfeit`'s hardcoded `"0-1"`, and
`/engine_move` running the engine twice.

New material for Phase 4 to use:

- **A regret baseline now exists.** Any Phase 4 fix can be measured as a change in
  mean/median regret and blunder rate on both suites, rather than argued for.
- **4 missed forced mates and 3–4 mate-allowing moves per suite** are concrete,
  inspectable failure cases.
- **Middlegame regret is ~5× endgame regret**, a lead worth following when
  evaluating the C2 lookahead fix.
- **Re-baselining will be required**: fixing any of C1–C5 changes engine output,
  so `tests/integration/test_regression_baseline.py` will fail by design and the
  Phase 0 baseline must be re-recorded with the diff documented.
- **Use `Clear Hash` in any future Stockfish comparison**, or the reference will
  silently depend on call order.
