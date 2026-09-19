# Evaluation run - extended

**2026-09-19T06:50:36+00:00** | commit `0f7ad6a4` (dirty) | Stockfish 17.1 depth 8, Threads=1, Hash=16MB

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
| Engine latency p50 / p95 | 779.04 / 1266.91 ms | 160 |
| Stockfish latency p50 / p95 | 7.61 / 23.47 ms | 160 |

> **Latency is host-dependent and is the one metric here that does NOT reproduce.** Repeated runs of identical code on the same machine in a single session produced median engine latencies between roughly 620ms and 1300ms depending on what else the host was doing. Quality metrics above reproduced exactly across runs; latency did not. Do not read a latency change between runs or phases as an engine change.


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
| X01P06 | opening | W | `b5c6` | `b5a4` | 7 |  | 1207 |
| X01P07 | opening | B | `c6d4` | `g8f6` | 106 |  | 648 |
| X01P11 | middlegame | B | `c6d4` | `b7b5` | 170 |  | 668 |
| X01P12 | middlegame | W | `a4b5` | `a4b3` | 406 |  | 579 |
| X02P06 | opening | W | `b5c6` | `e1g1` | 62 |  | 627 |
| X02P07 | opening | B | `c6d4` | `f8c5` | 57 |  | 645 |
| X02P11 | middlegame | B | `d7c6` | `d7c6` | 0 |  | 711 |
| X02P12 | middlegame | W | `d4e5` | `d4e5` | 0 |  | 764 |
| X03P06 | opening | W | `f3e5` | `c2c3` | 396 |  | 775 |
| X03P07 | opening | B | `g8f6` | `g8f6` | 0 |  | 818 |
| X03P11 | middlegame | B | `c8g4` | `a7a5` | 14 |  | 980 |
| X03P12 | middlegame | W | `f3e5` | `h2h3` | 440 |  | 832 |
| X04P06 | opening | W | `f3e5` | `d2d3` | 419 |  | 687 |
| X04P07 | opening | B | `f6e4` | `d7d5` | 165 |  | 1033 |
| X04P11 | middlegame | B | `c7c6` | `c8d7` | 9 |  | 341 |
| X04P12 | middlegame | W | `g5f7` | `d5c6` | 379 |  | 964 |
| X05P06 | opening | W | `d1d4` | `f3d4` | 594 |  | 815 |
| X05P07 | opening | B | `c6d4` | `g8f6` | 101 |  | 786 |
| X05P11 | middlegame | B | `f6d4` | `g8e7` | 529 |  | 1117 |
| X05P12 | middlegame | W | `d1g4` | `f1b5` | 138 |  | 1135 |
| X06P06 | opening | W | `e5f7` | `e5f3` | 234 |  | 721 |
| X06P07 | opening | B | `c8g4` | `f6e4` | 174 |  | 604 |
| X06P11 | middlegame | B | `c8f5` | `f8e7` | 1 |  | 890 |
| X06P12 | middlegame | W | `d3e4` | `e1g1` | 228 |  | 918 |
| X07P06 | opening | W | `d1d4` | `f3d4` | 4 |  | 710 |
| X07P07 | opening | B | `c8g4` | `a7a6` | 528 |  | 800 |
| X07P11 | middlegame | B | `f6e4` | `e7e5` | 367 |  | 845 |
| X07P12 | middlegame | W | `c3d5` | `d4b3` | 269 |  | 875 |
| X08P11 | middlegame | B | `f6e4` | `c8d7` | 357 |  | 879 |
| X08P12 | middlegame | W | `d4e6` | `f2f3` | 599 |  | 931 |
| X09P06 | opening | W | `d1d4` | `f3d4` | 510 |  | 783 |
| X09P07 | opening | B | `c6d4` | `g7g6` | 70 |  | 706 |
| X09P11 | middlegame | B | `f6e4` | `d7d6` | 216 |  | 895 |
| X09P12 | middlegame | W | `d1d6` | `c3d5` | 533 |  | 793 |
| X10P06 | opening | W | `d1d4` | `f3d4` | 3 |  | 766 |
| X10P07 | opening | B | `b8c6` | `g8f6` | 0 |  | 854 |
| X10P11 | middlegame | B | `c6d4` | `a7a6` | 41 |  | 1113 |
| X10P12 | middlegame | W | `d4e6` | `d4c6` | 402 |  | 1701 |
| X11P06 | opening | W | `d1g4` | `e4e5` | 26 |  | 759 |
| X11P07 | opening | B | `b4c3` | `c7c5` | 60 |  | 758 |
| X11P11 | middlegame | B | `c5d4` | `g8e7` | 20 |  | 703 |
| X11P12 | middlegame | W | `f1d3` | `g1f3` | -1 |  | 690 |
| X12P06 | opening | W | `c2c3` | `e4e5` | 129 |  | 673 |
| X12P07 | opening | B | `f6e4` | `f6d7` | 20 |  | 668 |
| X12P11 | middlegame | B | `c5d4` | `c5d4` | 0 |  | 743 |
| X12P12 | middlegame | W | `g1f3` | `d2f3` | 38 |  | 945 |
| X13P06 | opening | W | `c3e4` | `c3e4` | 0 |  | 882 |
| X13P07 | opening | B | `g8f6` | `c8f5` | -84 |  | 852 |
| X13P11 | middlegame | B | `g6c2` | `h7h6` | 467 |  | 745 |
| X13P12 | middlegame | W | `g3e4` | `g1f3` | 575 |  | 1025 |
| X14P06 | opening | W | `d1e2` | `c1g5` | 6 |  | 746 |
| X14P07 | opening | B | `f5c2` | `e7e6` | 513 |  | 704 |
| X14P11 | middlegame | B | `f5c2` | `b8c6` | 403 |  | 873 |
| X14P12 | middlegame | W | `d4c5` | `b1c3` | 86 |  | 846 |
| X15P06 | opening | W | `c4d5` | `g1f3` | 12 |  | 716 |
| X15P07 | opening | B | `d5c4` | `f8e7` | 11 |  | 805 |
| X15P11 | middlegame | B | `d5c4` | `b8d7` | 9 |  | 1181 |
| X15P12 | middlegame | W | `g5f6` | `g5h4` | 16 |  | 1371 |
| X16P06 | opening | W | `d4d5` | `b1c3` | 270 |  | 1189 |
| X16P07 | opening | B | `c8g4` | `b7b5` | 20 |  | 1058 |
| X16P11 | middlegame | B | `c5d4` | `f8e7` | 6 |  | 1565 |
| X16P12 | middlegame | W | `c4e6` | `e3e4` | 392 |  | 1253 |
| X17P06 | opening | W | `c4d5` | `e2e3` | 3 |  | 1182 |
| X17P07 | opening | B | `d5c4` | `d5c4` | 0 |  | 1156 |
| X17P11 | middlegame | B | `f6e4` | `h7h6` | 4 |  | 1278 |
| X17P12 | middlegame | W | `f1c4` | `f1c4` | 0 |  | 1439 |
| X18P11 | middlegame | B | `f8b4` | `d5c4` | 16 |  | 1520 |
| X18P12 | middlegame | W | `f3e5` | `d3c4` | 569 |  | 1412 |
| X19P06 | opening | W | `c4c5` | `c1d2` | 132 |  | 1071 |
| X19P07 | opening | B | `b4c3` | `c7c5` | 3 |  | 1121 |
| X19P11 | middlegame | B | `b4c3` | `c7c5` | 27 |  | 1240 |
| X19P12 | middlegame | W | `d4c5` | `c4d5` | 45 |  | 1221 |
| X20P06 | opening | W | `b1c3` | `e2e3` | 2 |  | 995 |
| X20P07 | opening | B | `h7h6` | `c8b7` | 43 |  | 968 |
| X20P11 | middlegame | B | `b7f3` | `e8g8` | 94 |  | 1135 |
| X20P12 | middlegame | W | `b1c3` | `b1c3` | 0 |  | 1162 |
| X21P06 | opening | W | `e2e4` | `e2e4` | 0 |  | 1083 |
| X21P07 | opening | B | `f6e4` | `d7d6` | 317 |  | 1129 |
| X21P11 | middlegame | B | `f6e4` | `e7e5` | 334 |  | 1196 |
| X21P12 | middlegame | W | `d4e5` | `d4d5` | 29 |  | 1849 |
| X22P06 | opening | W | `c4d5` | `g1f3` | -19 |  | 1238 |
| X22P07 | opening | B | `d8d5` | `f6d5` | 488 |  | 1114 |
| X22P11 | middlegame | B | `d8d4` | `c7c5` | 671 |  | 1180 |
| X22P12 | middlegame | W | `f1b5` | `g1f3` | 8 |  | 1135 |
| X23P06 | opening | W | `b1c3` | `b1c3` | 0 |  | 898 |
| X23P07 | opening | B | `e6d5` | `e6d5` | 0 |  | 958 |
| X23P11 | middlegame | B | `b8c6` | `g7g6` | 433 |  | 1259 |
| X23P12 | middlegame | W | `d1g4` | `a2a4` | 580 |  | 1267 |
| X24P06 | opening | W | `g2b7` | `c2c4` | 676 |  | 719 |
| X24P07 | opening | B | `e8f7` | `d7d5` | 102 |  | 806 |
| X24P11 | middlegame | B | `d7d5` | `d7d5` | 0 |  | 712 |
| X24P12 | middlegame | W | `b1c3` | `d1b3` | -33 |  | 681 |
| X25P06 | opening | W | `d2d4` | `e2e3` | 8 |  | 601 |
| X25P07 | opening | B | `c6d4` | `g7g6` | 47 |  | 608 |
| X25P11 | middlegame | B | `d7d5` | `e8g8` | 4 |  | 762 |
| X25P12 | middlegame | W | `d2d4` | `d2d3` | -10 |  | 652 |
| X26P06 | opening | W | `d2d4` | `g2g3` | 11 |  | 629 |
| X26P07 | opening | B | `e5e4` | `d7d5` | 22 |  | 753 |
| X26P11 | middlegame | B | `d5c3` | `d5c3` | 0 |  | 847 |
| X26P12 | middlegame | W | `f3e5` | `d2d3` | 403 |  | 834 |
| X27P06 | opening | W | `b1c3` | `c4d5` | 38 |  | 657 |
| X27P07 | opening | B | `b8c6` | `f8e7` | -1 |  | 757 |
| X27P11 | middlegame | B | `c7c5` | `c7c5` | 0 |  | 676 |
| X27P12 | middlegame | W | `c4d5` | `b2b3` | 14 |  | 793 |
| X28P06 | opening | W | `f4c7` | `g1f3` | 495 |  | 747 |
| X28P07 | opening | B | `c7c5` | `f8d6` | 16 |  | 769 |
| X28P11 | middlegame | B | `d6g3` | `c7c5` | 35 |  | 985 |
| X28P12 | middlegame | W | `g3d6` | `a2a4` | 6 |  | 844 |
| X29P06 | opening | W | `f4e5` | `e4d5` | 14 |  | 776 |
| X29P07 | opening | B | `d5e4` | `f6e4` | 355 |  | 805 |
| X29P11 | middlegame | B | `e4c3` | `c7c5` | 29 |  | 915 |
| X29P12 | middlegame | W | `c3e4` | `f1d3` | 208 |  | 849 |
| X30P06 | opening | W | `d1h5` | `d2d4` | 642 |  | 822 |
| X30P07 | opening | B | `a5c3` | `g8f6` | 529 |  | 821 |
| X30P11 | middlegame | B | `a5c3` | `c8f5` | 531 |  | 974 |
| X30P12 | middlegame | W | `c4f7` | `f3e5` | 284 |  | 998 |
| X31P06 | opening | W | `c1g5` | `g1f3` | 19 |  | 834 |
| X31P07 | opening | B | `f6e4` | `f8g7` | 335 |  | 752 |
| X31P11 | middlegame | B | `f6e4` | `b8c6` | 410 |  | 735 |
| X31P12 | middlegame | W | `e4e5` | `a2a4` | 30 |  | 783 |
| X32P06 | opening | W | `e5d6` | `g1f3` | 28 |  | 790 |
| X32P07 | opening | B | `d6e5` | `d6e5` | 0 |  | 757 |
| X32P11 | middlegame | B | `g4f3` | `b8c6` | 12 |  | 891 |
| X32P12 | middlegame | W | `e5d6` | `c2c4` | 17 |  | 719 |
| XEG01 | endgame | W | `e3d3` | `e3d3` | 0 |  | 266 |
| XEG02 | endgame | B | `e5d5` | `e5d6` | 74 |  | 325 |
| XEG03 | endgame | W | `g2g1` | `f3f4` | 152 |  | 295 |
| XEG04 | endgame | W | `g2h1` | `g2h1` | 0 |  | 398 |
| XEG05 | endgame | B | `h1a1` | `h1a1` | 0 |  | 409 |
| XEG06 | endgame | W | `g2h1` | `g2h1` | 0 |  | 389 |
| XEG07 | endgame | B | `h1a1` | `h1a1` | 0 |  | 421 |
| XEG08 | endgame | W | `e3d3` | `e3d2` | -8 |  | 318 |
| XEG09 | endgame | B | `e6f7` | `e6e5` | 45 |  | 295 |
| XEG10 | endgame | B | `e6f7` | `e6f6` | 1 |  | 379 |
| XEG11 | endgame | B | `g7h7` | `g7h6` | 47 |  | 352 |
| XEG12 | endgame | W | `f3g1` | `f3g1` | 0 |  | 342 |
| XEG13 | endgame | W | `e2f1` | `c3d5` | 9 |  | 354 |
| XEG14 | endgame | W | `g2h1` | `g2h1` | 0 |  | 441 |
| XEG15 | endgame | B | `h1d1` | `h1d1` | 0 |  | 484 |
| XEG16 | endgame | W | `g2g1` | `g2g1` | 0 |  | 329 |
| XEG17 | endgame | W | `f2f4` | `e2d3` | 16 |  | 325 |
| XEG18 | endgame | W | `g2h1` | `g2h1` | 0 |  | 357 |
| XEG19 | endgame | W | `e2f3` | `e2c4` | -6 |  | 311 |
| XEG20 | endgame | W | `g2h1` | `g2f3` | 504 |  | 289 |
| XEG21 | endgame | B | `g1f2` | `g1f2` | 0 |  | 341 |
| XTC01 | tactical | W | `a1a4` | `a1a8` | - | missed_forced_mate | 311 |
| XTC02 | tactical | B | `a8a5` | `a8a1` | - | missed_forced_mate | 295 |
| XTC03 | tactical | W | `e5f7` | `e5f3` | - | engine_move_allows_forced_mate | 360 |
| XTC04 | tactical | W | `e5f7` | `f2f4` | - | engine_move_allows_forced_mate | 426 |
| XTC05 | tactical | W | `b1f5` | `b1b8` | - | missed_forced_mate | 325 |
| XTC06 | tactical | W | `b4d6` | `g2g4` | -27 |  | 297 |
| XTC07 | tactical | W | `d1d5` | `d1b3` | 503 |  | 339 |
| XTC08 | tactical | W | `e5c4` | `d1b1` | -12 |  | 309 |
| XTC09 | tactical | W | `e7e8q` | `a1a2` | 24 |  | 297 |
| XTC10 | tactical | B | `f8b4` | `g8f6` | - | engine_move_allows_forced_mate | 795 |
| XDF01 | defensive | B | `g8h8` | `h7h6` | - | engine_move_allows_forced_mate | 289 |
| XDF02 | defensive | W | `h2h3` | `h2h3` | 0 |  | 330 |
| XDF03 | defensive | B | `c7c6` | `c7c6` | 0 |  | 331 |
| XDF04 | defensive | B | `d7d5` | `d8h4` | - | missed_forced_mate | 550 |
| XDF05 | defensive | W | `h2h3` | `h2h3` | 0 |  | 275 |

---

POV convention: All evaluations are expressed from the perspective of the side to move in the ORIGINAL position (the player choosing the move). Positive centipawns are good for that player.

Regret: move_regret_cp = eval_after_stockfish_move - eval_after_engine_move, both produced by the same Stockfish configuration and both converted to the mover's POV. Positive means the engine gave up that many centipawns. Negative values are retained, not clamped.

