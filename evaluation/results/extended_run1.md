# Evaluation run - extended

**2026-09-19T06:16:03+00:00** | commit `6cd14ebc` (dirty) | Stockfish 17.1 depth 8, Threads=1, Hash=16MB

Positions: **160**


## Headline metrics

| Metric | Value | n |
|---|---:|---:|
| Legality rate | 100.0% | 160/160 |
| Top-1 agreement | 18.12% | 29/160 |
| Top-3 containment | 33.75% | 54/160 |
| Mean move regret | 144.07 cp | 152 |
| Median move regret | 25.0 cp | 152 |
| p95 move regret | 533 cp | 152 |
| Blunder rate (>300cp) | 24.34% | 37/152 |
| Engine latency p50 / p95 | 1304.84 / 4032.65 ms | 160 |
| Stockfish latency p50 / p95 | 10.54 / 47.3 ms | 160 |

### Chance reference

A uniformly random legal move would agree with Stockfish's top move **4.31%** of the time and fall in its top 3 **12.93%** of the time on this suite (mean over positions of min(N, legal_moves)/legal_moves).


## Mate handling

Regret is defined for **152/160** positions. Mate scores are ordinals, not centipawns, so mate-involved positions are excluded from every centipawn aggregate and counted here instead.

| Mate status | Count |
|---|---:|
| `none` | 152 |
| `missed_forced_mate` | 4 |
| `engine_move_allows_forced_mate` | 4 |

## By category

| Group | n | Top-1 | Top-3 | Mean regret | Median regret |
|---|---:|---:|---:|---:|---:|
| defensive | 5 | 60.0% | 60.0% | 0.0 | 0 |
| endgame | 21 | 52.38% | 57.14% | 39.71 | 0 |
| middlegame | 64 | 12.5% | 23.44% | 185.3 | 39.5 |
| opening | 60 | 11.67% | 40.0% | 145.3 | 33.0 |
| tactical | 10 | 0.0% | 0.0% | 122.0 | 6.0 |

Small groups are descriptive only - no significance is implied.


## By side to move

| Group | n | Top-1 | Top-3 | Mean regret | Median regret |
|---|---:|---:|---:|---:|---:|
| black | 75 | 18.67% | 36.0% | 132.01 | 29 |
| white | 85 | 17.65% | 31.76% | 154.64 | 19 |

Small groups are descriptive only - no significance is implied.


## Rank correlation

Spearman rho between the engine's ranking and Stockfish's ranking of the **same** candidate moves (Stockfish's top 8), computed on ranks only so the engine's non-centipawn scale is irrelevant.

Mean **0.22**, median **0.3**, over 160 positions.


## Per-position results

| ID | Cat | STM | Engine | Stockfish | Regret cp | Mate status | ms |
|---|---|---|---|---|---:|---|---:|
| X01P06 | opening | W | `b5c6` | `b5a4` | 7 |  | 2783 |
| X01P07 | opening | B | `c6d4` | `g8f6` | 106 |  | 1706 |
| X01P11 | middlegame | B | `c6d4` | `b7b5` | 170 |  | 1167 |
| X01P12 | middlegame | W | `a4b5` | `a4b3` | 406 |  | 1426 |
| X02P06 | opening | W | `b5c6` | `e1g1` | 62 |  | 1154 |
| X02P07 | opening | B | `c6d4` | `f8c5` | 57 |  | 1092 |
| X02P11 | middlegame | B | `d7c6` | `d7c6` | 0 |  | 1094 |
| X02P12 | middlegame | W | `d4e5` | `d4e5` | 0 |  | 3337 |
| X03P06 | opening | W | `f3e5` | `c2c3` | 396 |  | 1354 |
| X03P07 | opening | B | `g8f6` | `g8f6` | 0 |  | 1419 |
| X03P11 | middlegame | B | `c8g4` | `a7a5` | 14 |  | 1477 |
| X03P12 | middlegame | W | `f3e5` | `h2h3` | 440 |  | 1447 |
| X04P06 | opening | W | `f3e5` | `d2d3` | 419 |  | 1966 |
| X04P07 | opening | B | `f6e4` | `d7d5` | 165 |  | 1371 |
| X04P11 | middlegame | B | `c7c6` | `c8d7` | 9 |  | 591 |
| X04P12 | middlegame | W | `g5f7` | `d5c6` | 379 |  | 1859 |
| X05P06 | opening | W | `d1d4` | `f3d4` | 594 |  | 1543 |
| X05P07 | opening | B | `c6d4` | `g8f6` | 101 |  | 2074 |
| X05P11 | middlegame | B | `f6d4` | `g8e7` | 529 |  | 2382 |
| X05P12 | middlegame | W | `d1g4` | `f1b5` | 138 |  | 5910 |
| X06P06 | opening | W | `e5f7` | `e5f3` | 234 |  | 1726 |
| X06P07 | opening | B | `c8g4` | `f6e4` | 174 |  | 2016 |
| X06P11 | middlegame | B | `c8f5` | `f8e7` | 1 |  | 1693 |
| X06P12 | middlegame | W | `d3e4` | `e1g1` | 228 |  | 2312 |
| X07P06 | opening | W | `d1d4` | `f3d4` | 4 |  | 1817 |
| X07P07 | opening | B | `c8g4` | `a7a6` | 528 |  | 3423 |
| X07P11 | middlegame | B | `f6e4` | `e7e5` | 367 |  | 1705 |
| X07P12 | middlegame | W | `c3d5` | `d4b3` | 269 |  | 1441 |
| X08P11 | middlegame | B | `f6e4` | `c8d7` | 357 |  | 2035 |
| X08P12 | middlegame | W | `d4e6` | `f2f3` | 599 |  | 2072 |
| X09P06 | opening | W | `d1d4` | `f3d4` | 510 |  | 1337 |
| X09P07 | opening | B | `c6d4` | `g7g6` | 70 |  | 1631 |
| X09P11 | middlegame | B | `f6e4` | `d7d6` | 216 |  | 2492 |
| X09P12 | middlegame | W | `d1d6` | `c3d5` | 533 |  | 3113 |
| X10P06 | opening | W | `d1d4` | `f3d4` | 3 |  | 3425 |
| X10P07 | opening | B | `b8c6` | `g8f6` | 0 |  | 1456 |
| X10P11 | middlegame | B | `c6d4` | `a7a6` | 41 |  | 5949 |
| X10P12 | middlegame | W | `d4e6` | `d4c6` | 402 |  | 3444 |
| X11P06 | opening | W | `d1g4` | `e4e5` | 26 |  | 813 |
| X11P07 | opening | B | `b4c3` | `c7c5` | 60 |  | 816 |
| X11P11 | middlegame | B | `c5d4` | `g8e7` | 20 |  | 3430 |
| X11P12 | middlegame | W | `f1d3` | `g1f3` | -1 |  | 3840 |
| X12P06 | opening | W | `c2c3` | `e4e5` | 129 |  | 1610 |
| X12P07 | opening | B | `f6e4` | `f6d7` | 20 |  | 1805 |
| X12P11 | middlegame | B | `c5d4` | `c5d4` | 0 |  | 1325 |
| X12P12 | middlegame | W | `g1f3` | `d2f3` | 38 |  | 941 |
| X13P06 | opening | W | `c3e4` | `c3e4` | 0 |  | 779 |
| X13P07 | opening | B | `g8f6` | `c8f5` | -84 |  | 812 |
| X13P11 | middlegame | B | `g6c2` | `h7h6` | 467 |  | 1052 |
| X13P12 | middlegame | W | `g3e4` | `g1f3` | 575 |  | 797 |
| X14P06 | opening | W | `d1e2` | `c1g5` | 6 |  | 3386 |
| X14P07 | opening | B | `f5c2` | `e7e6` | 513 |  | 3084 |
| X14P11 | middlegame | B | `f5c2` | `b8c6` | 403 |  | 3287 |
| X14P12 | middlegame | W | `d4c5` | `b1c3` | 86 |  | 7572 |
| X15P06 | opening | W | `c4d5` | `g1f3` | 12 |  | 2130 |
| X15P07 | opening | B | `d5c4` | `f8e7` | 11 |  | 2994 |
| X15P11 | middlegame | B | `d5c4` | `b8d7` | 9 |  | 3625 |
| X15P12 | middlegame | W | `g5f6` | `g5h4` | 16 |  | 9265 |
| X16P06 | opening | W | `d4d5` | `b1c3` | 270 |  | 2246 |
| X16P07 | opening | B | `c8g4` | `b7b5` | 20 |  | 3588 |
| X16P11 | middlegame | B | `c5d4` | `f8e7` | 6 |  | 1693 |
| X16P12 | middlegame | W | `c4e6` | `e3e4` | 392 |  | 2817 |
| X17P06 | opening | W | `c4d5` | `e2e3` | 3 |  | 3666 |
| X17P07 | opening | B | `d5c4` | `d5c4` | 0 |  | 1905 |
| X17P11 | middlegame | B | `f6e4` | `h7h6` | 4 |  | 906 |
| X17P12 | middlegame | W | `f1c4` | `f1c4` | 0 |  | 913 |
| X18P11 | middlegame | B | `f8b4` | `d5c4` | 16 |  | 1508 |
| X18P12 | middlegame | W | `f3e5` | `d3c4` | 569 |  | 2505 |
| X19P06 | opening | W | `c4c5` | `c1d2` | 132 |  | 695 |
| X19P07 | opening | B | `b4c3` | `c7c5` | 3 |  | 717 |
| X19P11 | middlegame | B | `b4c3` | `c7c5` | 27 |  | 842 |
| X19P12 | middlegame | W | `d4c5` | `c4d5` | 45 |  | 759 |
| X20P06 | opening | W | `b1c3` | `e2e3` | 2 |  | 640 |
| X20P07 | opening | B | `h7h6` | `c8b7` | 43 |  | 815 |
| X20P11 | middlegame | B | `b7f3` | `e8g8` | 94 |  | 759 |
| X20P12 | middlegame | W | `b1c3` | `b1c3` | 0 |  | 772 |
| X21P06 | opening | W | `e2e4` | `e2e4` | 0 |  | 671 |
| X21P07 | opening | B | `f6e4` | `d7d6` | 317 |  | 767 |
| X21P11 | middlegame | B | `f6e4` | `e7e5` | 334 |  | 819 |
| X21P12 | middlegame | W | `d4e5` | `d4d5` | 29 |  | 1889 |
| X22P06 | opening | W | `c4d5` | `g1f3` | -19 |  | 1846 |
| X22P07 | opening | B | `d8d5` | `f6d5` | 488 |  | 3421 |
| X22P11 | middlegame | B | `d8d4` | `c7c5` | 671 |  | 1815 |
| X22P12 | middlegame | W | `f1b5` | `g1f3` | 8 |  | 3703 |
| X23P06 | opening | W | `b1c3` | `b1c3` | 0 |  | 1037 |
| X23P07 | opening | B | `e6d5` | `e6d5` | 0 |  | 1285 |
| X23P11 | middlegame | B | `b8c6` | `g7g6` | 433 |  | 2538 |
| X23P12 | middlegame | W | `d1g4` | `a2a4` | 580 |  | 1499 |
| X24P06 | opening | W | `g2b7` | `c2c4` | 676 |  | 1355 |
| X24P07 | opening | B | `e8f7` | `d7d5` | 102 |  | 1112 |
| X24P11 | middlegame | B | `d7d5` | `d7d5` | 0 |  | 3112 |
| X24P12 | middlegame | W | `b1c3` | `d1b3` | -33 |  | 3874 |
| X25P06 | opening | W | `d2d4` | `e2e3` | 8 |  | 3371 |
| X25P07 | opening | B | `c6d4` | `g7g6` | 47 |  | 6285 |
| X25P11 | middlegame | B | `d7d5` | `e8g8` | 4 |  | 5203 |
| X25P12 | middlegame | W | `d2d4` | `d2d3` | -10 |  | 677 |
| X26P06 | opening | W | `d2d4` | `g2g3` | 11 |  | 759 |
| X26P07 | opening | B | `e5e4` | `d7d5` | 22 |  | 635 |
| X26P11 | middlegame | B | `d5c3` | `d5c3` | 0 |  | 771 |
| X26P12 | middlegame | W | `f3e5` | `d2d3` | 403 |  | 881 |
| X27P06 | opening | W | `b1c3` | `c4d5` | 38 |  | 1661 |
| X27P07 | opening | B | `b8c6` | `f8e7` | -1 |  | 4379 |
| X27P11 | middlegame | B | `c7c5` | `c7c5` | 0 |  | 2558 |
| X27P12 | middlegame | W | `c4d5` | `b2b3` | 14 |  | 1353 |
| X28P06 | opening | W | `f4c7` | `g1f3` | 495 |  | 3585 |
| X28P07 | opening | B | `c7c5` | `f8d6` | 16 |  | 4851 |
| X28P11 | middlegame | B | `d6g3` | `c7c5` | 35 |  | 2355 |
| X28P12 | middlegame | W | `g3d6` | `a2a4` | 6 |  | 4033 |
| X29P06 | opening | W | `f4e5` | `e4d5` | 14 |  | 1355 |
| X29P07 | opening | B | `d5e4` | `f6e4` | 355 |  | 2250 |
| X29P11 | middlegame | B | `e4c3` | `c7c5` | 29 |  | 1911 |
| X29P12 | middlegame | W | `c3e4` | `f1d3` | 208 |  | 3858 |
| X30P06 | opening | W | `d1h5` | `d2d4` | 642 |  | 1647 |
| X30P07 | opening | B | `a5c3` | `g8f6` | 529 |  | 915 |
| X30P11 | middlegame | B | `a5c3` | `c8f5` | 531 |  | 1146 |
| X30P12 | middlegame | W | `c4f7` | `f3e5` | 284 |  | 1122 |
| X31P06 | opening | W | `c1g5` | `g1f3` | 19 |  | 797 |
| X31P07 | opening | B | `f6e4` | `f8g7` | 335 |  | 805 |
| X31P11 | middlegame | B | `f6e4` | `b8c6` | 410 |  | 974 |
| X31P12 | middlegame | W | `e4e5` | `a2a4` | 30 |  | 758 |
| X32P06 | opening | W | `e5d6` | `g1f3` | 28 |  | 774 |
| X32P07 | opening | B | `d6e5` | `d6e5` | 0 |  | 777 |
| X32P11 | middlegame | B | `g4f3` | `b8c6` | 12 |  | 740 |
| X32P12 | middlegame | W | `e5d6` | `c2c4` | 17 |  | 714 |
| XEG01 | endgame | W | `e3d3` | `e3d3` | 0 |  | 275 |
| XEG02 | endgame | B | `e5d5` | `e5d6` | 74 |  | 266 |
| XEG03 | endgame | W | `g2g1` | `f3f4` | 152 |  | 321 |
| XEG04 | endgame | W | `g2h1` | `g2h1` | 0 |  | 415 |
| XEG05 | endgame | B | `h1a1` | `h1a1` | 0 |  | 460 |
| XEG06 | endgame | W | `g2h1` | `g2h1` | 0 |  | 386 |
| XEG07 | endgame | B | `h1a1` | `h1a1` | 0 |  | 399 |
| XEG08 | endgame | W | `e3d3` | `e3d2` | -8 |  | 282 |
| XEG09 | endgame | B | `e6f7` | `e6e5` | 45 |  | 293 |
| XEG10 | endgame | B | `e6f7` | `e6f6` | 1 |  | 327 |
| XEG11 | endgame | B | `g7h7` | `g7h6` | 47 |  | 297 |
| XEG12 | endgame | W | `f3g1` | `f3g1` | 0 |  | 351 |
| XEG13 | endgame | W | `e2f1` | `c3d5` | 9 |  | 374 |
| XEG14 | endgame | W | `g2h1` | `g2h1` | 0 |  | 451 |
| XEG15 | endgame | B | `h1d1` | `h1d1` | 0 |  | 448 |
| XEG16 | endgame | W | `g2g1` | `g2g1` | 0 |  | 410 |
| XEG17 | endgame | W | `f2f4` | `e2d3` | 16 |  | 328 |
| XEG18 | endgame | W | `g2h1` | `g2h1` | 0 |  | 375 |
| XEG19 | endgame | W | `e2f3` | `e2c4` | -6 |  | 333 |
| XEG20 | endgame | W | `g2h1` | `g2f3` | 504 |  | 280 |
| XEG21 | endgame | B | `g1f2` | `g1f2` | 0 |  | 326 |
| XTC01 | tactical | W | `a1a4` | `a1a8` | - | missed_forced_mate | 287 |
| XTC02 | tactical | B | `a8a5` | `a8a1` | - | missed_forced_mate | 304 |
| XTC03 | tactical | W | `e5f7` | `e5f3` | - | engine_move_allows_forced_mate | 358 |
| XTC04 | tactical | W | `e5f7` | `f2f4` | - | engine_move_allows_forced_mate | 435 |
| XTC05 | tactical | W | `b1f5` | `b1b8` | - | missed_forced_mate | 339 |
| XTC06 | tactical | W | `b4d6` | `g2g4` | -27 |  | 310 |
| XTC07 | tactical | W | `d1d5` | `d1b3` | 503 |  | 313 |
| XTC08 | tactical | W | `e5c4` | `d1b1` | -12 |  | 258 |
| XTC09 | tactical | W | `e7e8q` | `a1a2` | 24 |  | 302 |
| XTC10 | tactical | B | `f8b4` | `g8f6` | - | engine_move_allows_forced_mate | 885 |
| XDF01 | defensive | B | `g8h8` | `h7h6` | - | engine_move_allows_forced_mate | 288 |
| XDF02 | defensive | W | `h2h3` | `h2h3` | 0 |  | 336 |
| XDF03 | defensive | B | `c7c6` | `c7c6` | 0 |  | 386 |
| XDF04 | defensive | B | `d7d5` | `d8h4` | - | missed_forced_mate | 948 |
| XDF05 | defensive | W | `h2h3` | `h2h3` | 0 |  | 485 |

---

POV convention: All evaluations are expressed from the perspective of the side to move in the ORIGINAL position (the player choosing the move). Positive centipawns are good for that player.

Regret: move_regret_cp = eval_after_stockfish_move - eval_after_engine_move, both produced by the same Stockfish configuration and both converted to the mover's POV. Positive means the engine gave up that many centipawns. Negative values are retained, not clamped.

