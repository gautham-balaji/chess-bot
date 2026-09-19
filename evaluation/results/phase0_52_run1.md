# Evaluation run - phase0_52

**2026-09-19T06:02:45+00:00** | commit `6cd14ebc` (dirty) | Stockfish 17.1 depth 8, Threads=1, Hash=16MB

Positions: **52**


## Headline metrics

| Metric | Value | n |
|---|---:|---:|
| Legality rate | 100.0% | 52/52 |
| Top-1 agreement | 19.23% | 10/52 |
| Top-3 containment | 32.69% | 17/52 |
| Mean move regret | 107.51 cp | 45 |
| Median move regret | 45 cp | 45 |
| p95 move regret | 419 cp | 45 |
| Blunder rate (>300cp) | 15.56% | 7/45 |
| Engine latency p50 / p95 | 1247.99 / 3440.98 ms | 52 |
| Stockfish latency p50 / p95 | 9.09 / 38.58 ms | 52 |

### Chance reference

A uniformly random legal move would agree with Stockfish's top move **5.91%** of the time and fall in its top 3 **17.72%** of the time on this suite (mean over positions of min(N, legal_moves)/legal_moves).


## Mate handling

Regret is defined for **45/52** positions. Mate scores are ordinals, not centipawns, so mate-involved positions are excluded from every centipawn aggregate and counted here instead.

| Mate status | Count |
|---|---:|
| `none` | 45 |
| `missed_forced_mate` | 4 |
| `engine_move_allows_forced_mate` | 3 |

## By category

| Group | n | Top-1 | Top-3 | Mean regret | Median regret |
|---|---:|---:|---:|---:|---:|
| defensive | 4 | 25.0% | 25.0% | 84.0 | 84.0 |
| endgame | 12 | 41.67% | 58.33% | 12.42 | 0.0 |
| middlegame | 12 | 8.33% | 16.67% | 231.75 | 176.0 |
| opening | 18 | 11.11% | 33.33% | 96.67 | 50.5 |
| tactical | 6 | 16.67% | 16.67% | 0.0 | 0 |

Small groups are descriptive only - no significance is implied.


## By side to move

| Group | n | Top-1 | Top-3 | Mean regret | Median regret |
|---|---:|---:|---:|---:|---:|
| black | 22 | 9.09% | 31.82% | 85.0 | 46.0 |
| white | 30 | 26.67% | 33.33% | 122.52 | 29 |

Small groups are descriptive only - no significance is implied.


## Rank correlation

Spearman rho between the engine's ranking and Stockfish's ranking of the **same** candidate moves (Stockfish's top 8), computed on ranks only so the engine's non-centipawn scale is irrelevant.

Mean **0.24**, median **0.37**, over 52 positions.


## Per-position results

| ID | Cat | STM | Engine | Stockfish | Regret cp | Mate status | ms |
|---|---|---|---|---|---:|---|---:|
| OP01 | opening | W | `d2d4` | `e2e4` | 29 |  | 2519 |
| OP02 | opening | B | `e7e5` | `c7c5` | 12 |  | 1136 |
| OP03 | opening | W | `d2d4` | `g1f3` | 50 |  | 3058 |
| OP04 | opening | W | `d1g4` | `g1f3` | 110 |  | 1543 |
| OP05 | opening | W | `d1h5` | `b1c3` | 118 |  | 1701 |
| OP06 | opening | W | `d1g4` | `d2d4` | 142 |  | 1170 |
| OP07 | opening | B | `c8g4` | `e7e6` | 122 |  | 1484 |
| OP08 | opening | W | `c4d5` | `e2e3` | 14 |  | 2712 |
| OP09 | opening | W | `f3e5` | `f3e5` | 0 |  | 1143 |
| OP10 | opening | B | `c6d4` | `a7a6` | 24 |  | 2454 |
| OP11 | opening | W | `f3e5` | `d2d3` | 419 |  | 1338 |
| OP12 | opening | W | `f3e5` | `c2c3` | 375 |  | 1672 |
| OP13 | opening | W | `e2e4` | `e2e4` | 0 |  | 3453 |
| OP14 | opening | W | `c4c5` | `c1d2` | 132 |  | 3862 |
| OP15 | opening | B | `b8c6` | `g8f6` | 19 |  | 2897 |
| OP16 | opening | B | `d7d5` | `c7c6` | 37 |  | 1197 |
| OP17 | opening | B | `e7e5` | `d8d5` | 86 |  | 1572 |
| OP18 | opening | B | `c6d4` | `e5d4` | 51 |  | 1420 |
| MG01 | middlegame | W | `b3d5` | `d2d4` | 77 |  | 1583 |
| MG02 | middlegame | W | `d1d6` | `f2f3` | 614 |  | 1576 |
| MG03 | middlegame | W | `h4f6` | `c4d5` | 17 |  | 1695 |
| MG04 | middlegame | W | `f1d3` | `g1f3` | -1 |  | 1123 |
| MG05 | middlegame | W | `f3e5` | `h2h3` | 471 |  | 2203 |
| MG06 | middlegame | W | `f3e5` | `c1e3` | 417 |  | 1299 |
| MG07 | middlegame | W | `f3e5` | `h4h5` | 150 |  | 1458 |
| MG08 | middlegame | W | `d1d8` | `d1d8` | 0 |  | 3441 |
| MG09 | middlegame | B | `f6e4` | `c6a5` | 202 |  | 1700 |
| MG10 | middlegame | B | `f6e4` | `e8g8` | 402 |  | 3677 |
| MG11 | middlegame | B | `e6d5` | `f6d5` | 61 |  | 3316 |
| MG12 | middlegame | B | `f6e4` | `g8h8` | 371 |  | 1451 |
| EG01 | endgame | W | `e3d3` | `e3d3` | 0 |  | 424 |
| EG02 | endgame | B | `e5d5` | `e5d6` | 74 |  | 383 |
| EG03 | endgame | W | `e3d3` | `e3d2` | -8 |  | 480 |
| EG04 | endgame | B | `e6f7` | `e6f6` | 1 |  | 507 |
| EG05 | endgame | W | `g2h1` | `g2h1` | 0 |  | 659 |
| EG06 | endgame | W | `h2g1` | `h2g3` | 14 |  | 477 |
| EG07 | endgame | B | `g7h7` | `g7h6` | 47 |  | 443 |
| EG08 | endgame | W | `f3g1` | `f3g1` | 0 |  | 696 |
| EG09 | endgame | B | `e6e5` | `e6f5` | -24 |  | 468 |
| EG10 | endgame | W | `g2h1` | `g2h1` | 0 |  | 1173 |
| EG11 | endgame | B | `h1a1` | `h1a1` | 0 |  | 669 |
| EG12 | endgame | B | `e6f7` | `e6e5` | 45 |  | 475 |
| TC01 | tactical | W | `a1a4` | `a1a8` | - | missed_forced_mate | 431 |
| TC02 | tactical | B | `a8a5` | `a8a1` | - | missed_forced_mate | 591 |
| TC03 | tactical | B | `f8b4` | `g8f6` | - | engine_move_allows_forced_mate | 1740 |
| TC04 | tactical | W | `e5f7` | `e5f3` | - | engine_move_allows_forced_mate | 646 |
| TC05 | tactical | W | `b1f5` | `b1b8` | - | missed_forced_mate | 547 |
| TC06 | tactical | B | `f6e4` | `f6e4` | 0 |  | 1092 |
| DF01 | defensive | B | `g8h8` | `h7h6` | - | engine_move_allows_forced_mate | 831 |
| DF02 | defensive | W | `h2h3` | `h2h3` | 0 |  | 519 |
| DF03 | defensive | B | `d7d5` | `d8h4` | - | missed_forced_mate | 905 |
| DF04 | defensive | W | `e1g1` | `e1c1` | 168 |  | 772 |

---

POV convention: All evaluations are expressed from the perspective of the side to move in the ORIGINAL position (the player choosing the move). Positive centipawns are good for that player.

Regret: move_regret_cp = eval_after_stockfish_move - eval_after_engine_move, both produced by the same Stockfish configuration and both converted to the mover's POV. Positive means the engine gave up that many centipawns. Negative values are retained, not clamped.

