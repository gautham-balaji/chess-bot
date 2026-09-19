"""Build an extended evaluation suite with verifiable provenance.

PROVENANCE POLICY
    Opening and middlegame positions are produced by playing a NAMED move
    sequence from the initial position. The move list is stored with each
    position, so the provenance is self-verifying: re-running this script
    reproduces every FEN exactly, and any reader can replay the moves.

    Endgame and tactical positions are hand-constructed literal FENs described
    by their literal material ("rook and pawn vs rook"), not by theory names
    that would be an unverified claim.

WHAT THIS IS NOT
    These positions are NOT sampled from master games, NOT drawn from any
    tactics database, and NOT a statistically representative sample of practical
    chess. They are a fixed, reproducible comparison set, deliberately spread
    across game phases and both sides to move. No population is being estimated,
    so no confidence interval is implied by results computed on them.

The Phase 0 52-position suite is NOT modified or replaced. This is an additional
dataset; phase0_52 remains the set used for Phase 0 comparison.

Usage:
    python evaluation/build_extended_suite.py evaluation/positions/extended.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import chess

# Named opening lines. Each is truncated at several plies to yield positions
# across the opening/early-middlegame boundary, alternating side to move.
OPENING_LINES = [
    ("Ruy Lopez, Closed", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6",
                           "O-O", "Be7", "Re1", "b5", "Bb3", "d6", "c3", "O-O"]),
    ("Ruy Lopez, Berlin", ["e4", "e5", "Nf3", "Nc6", "Bb5", "Nf6", "O-O", "Nxe4",
                           "d4", "Nd6", "Bxc6", "dxc6", "dxe5", "Nf5"]),
    ("Italian, Giuoco Pianissimo", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3",
                                    "Nf6", "d3", "d6", "O-O", "O-O", "Re1", "a6"]),
    ("Two Knights Defence", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5",
                             "exd5", "Na5", "Bb5+", "c6", "dxc6", "bxc6"]),
    ("Scotch Game", ["e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4", "Bc5",
                     "Be3", "Qf6", "c3", "Nge7"]),
    ("Petroff Defence", ["e4", "e5", "Nf3", "Nf6", "Nxe5", "d6", "Nf3", "Nxe4",
                         "d4", "d5", "Bd3", "Nc6", "O-O", "Be7"]),
    ("Sicilian Najdorf", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6",
                          "Nc3", "a6", "Be3", "e5", "Nb3", "Be7", "f3", "O-O"]),
    ("Sicilian Dragon", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6",
                         "Nc3", "g6", "Be3", "Bg7", "f3", "O-O", "Qd2", "Nc6"]),
    ("Sicilian Sveshnikov", ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "Nf6",
                             "Nc3", "e5", "Ndb5", "d6", "Bg5", "a6", "Na3", "b5"]),
    ("Sicilian Taimanov", ["e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4", "Nc6",
                           "Nc3", "Qc7", "Be3", "a6", "Qd2", "Nf6"]),
    ("French Winawer", ["e4", "e6", "d4", "d5", "Nc3", "Bb4", "e5", "c5", "a3",
                        "Bxc3+", "bxc3", "Ne7", "Qg4", "O-O"]),
    ("French Tarrasch", ["e4", "e6", "d4", "d5", "Nd2", "Nf6", "e5", "Nfd7",
                         "Bd3", "c5", "c3", "Nc6", "Ne2", "cxd4"]),
    ("Caro-Kann Classical", ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5",
                             "Ng3", "Bg6", "h4", "h6", "Nf3", "Nd7"]),
    ("Caro-Kann Advance", ["e4", "c6", "d4", "d5", "e5", "Bf5", "Nf3", "e6",
                           "Be2", "c5", "Be3", "Qb6", "Nc3", "Nc6"]),
    ("Queen's Gambit Declined", ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7",
                                 "e3", "O-O", "Nf3", "h6", "Bh4", "b6"]),
    ("Queen's Gambit Accepted", ["d4", "d5", "c4", "dxc4", "Nf3", "Nf6", "e3", "e6",
                                 "Bxc4", "c5", "O-O", "a6", "dxc5", "Bxc5"]),
    ("Slav Defence", ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "dxc4", "a4",
                      "Bf5", "e3", "e6", "Bxc4", "Bb4"]),
    ("Semi-Slav Meran", ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "e6", "e3",
                         "Nbd7", "Bd3", "dxc4", "Bxc4", "b5"]),
    ("Nimzo-Indian Rubinstein", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3", "O-O",
                                 "Bd3", "d5", "Nf3", "c5", "O-O", "Nc6"]),
    ("Queen's Indian Defence", ["d4", "Nf6", "c4", "e6", "Nf3", "b6", "g3", "Bb7",
                                "Bg2", "Be7", "O-O", "O-O", "Nc3", "Ne4"]),
    ("King's Indian Classical", ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6",
                                 "Nf3", "O-O", "Be2", "e5", "O-O", "Nc6"]),
    ("Gruenfeld Exchange", ["d4", "Nf6", "c4", "g6", "Nc3", "d5", "cxd5", "Nxd5",
                            "e4", "Nxc3", "bxc3", "Bg7", "Nf3", "c5"]),
    ("Benoni Modern", ["d4", "Nf6", "c4", "c5", "d5", "e6", "Nc3", "exd5", "cxd5",
                       "d6", "e4", "g6", "Nf3", "Bg7"]),
    ("Dutch Defence", ["d4", "f5", "g3", "Nf6", "Bg2", "e6", "Nf3", "Be7", "O-O",
                       "O-O", "c4", "d6", "Nc3", "Qe8"]),
    ("English Symmetrical", ["c4", "c5", "Nf3", "Nf6", "Nc3", "Nc6", "g3", "g6",
                             "Bg2", "Bg7", "O-O", "O-O", "d4", "cxd4"]),
    ("English, Reversed Sicilian", ["c4", "e5", "Nc3", "Nf6", "Nf3", "Nc6", "g3",
                                    "d5", "cxd5", "Nxd5", "Bg2", "Nb6"]),
    ("Reti Opening", ["Nf3", "d5", "c4", "e6", "g3", "Nf6", "Bg2", "Be7", "O-O",
                      "O-O", "d4", "Nbd7", "Nbd2", "c6"]),
    ("London System", ["d4", "d5", "Bf4", "Nf6", "e3", "e6", "Nf3", "Bd6", "Bg3",
                       "O-O", "Bd3", "c5", "c3", "Nc6"]),
    ("Vienna Game", ["e4", "e5", "Nc3", "Nf6", "f4", "d5", "fxe5", "Nxe4", "Nf3",
                     "Be7", "d4", "O-O"]),
    ("Scandinavian Defence", ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qa5", "d4", "Nf6",
                              "Nf3", "c6", "Bc4", "Bf5", "Bd2", "e6"]),
    ("Pirc Defence", ["e4", "d6", "d4", "Nf6", "Nc3", "g6", "Nf3", "Bg7", "Be2",
                      "O-O", "O-O", "c6", "a4", "Nbd7"]),
    ("Alekhine Defence", ["e4", "Nf6", "e5", "Nd5", "d4", "d6", "Nf3", "Bg4",
                          "Be2", "e6", "O-O", "Be7", "c4", "Nb6"]),
]

# Plies at which each line is sampled. An even ply leaves White to move and an
# odd ply leaves Black, so this set is chosen 2-and-2 to keep the suite balanced
# across sides. All lines above are at least 12 plies, so every truncation applies.
TRUNCATIONS = [6, 7, 11, 12]

ENDGAMES = [
    ("king and pawn vs king, White to move", "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"),
    ("king and pawn vs king, Black to move", "8/8/8/4k3/8/4K3/4P3/8 b - - 0 1"),
    ("king and two pawns vs king and pawn", "8/5pk1/8/8/8/5P2/5PK1/8 w - - 0 1"),
    ("rook and pawn vs rook, White to move", "8/5pk1/8/8/8/8/5PK1/R6r w - - 0 1"),
    ("rook and pawn vs rook, Black to move", "8/5pk1/8/8/8/8/5PK1/R6r b - - 0 1"),
    ("rook endgame, White an extra pawn", "8/5pk1/6p1/8/8/6P1/5PKP/R6r w - - 0 1"),
    ("rook endgame, Black an extra pawn", "8/5pkp/6p1/8/8/6P1/5PK1/R6r b - - 0 1"),
    ("king and rook vs king, White to move", "8/8/4k3/8/8/4K3/8/4R3 w - - 0 1"),
    ("king and rook vs king, Black to move", "8/8/4k3/8/8/4K3/8/4R3 b - - 0 1"),
    ("king and queen vs king, Black to move", "8/8/4k3/8/8/4K3/8/4Q3 b - - 0 1"),
    ("bishop and pawn vs bishop and pawn", "8/5pk1/8/8/8/2B5/5PK1/6b1 b - - 0 1"),
    ("knight and pawns vs bishop and pawns", "8/4kpp1/8/8/8/5N2/4KPP1/6b1 w - - 0 1"),
    ("two knights and pawn vs knight and pawn", "8/4kp2/8/8/3N4/2N5/4KP2/5n2 w - - 0 1"),
    ("queen vs rook and pawn, White to move", "8/5pk1/8/8/8/8/5PK1/3Q3r w - - 0 1"),
    ("queen vs rook and pawn, Black to move", "8/5pk1/8/8/8/8/5PK1/3Q3r b - - 0 1"),
    ("opposite-coloured bishops, White an extra pawn", "8/5pk1/8/3B4/8/6P1/5PK1/6b1 w - - 0 1"),
    ("four-pawn king and pawn endgame", "8/4kppp/8/8/8/8/4KPPP/8 w - - 0 1"),
    ("rook and two pawns vs rook and pawn", "8/5pk1/8/8/8/5P2/5PK1/R6r w - - 0 1"),
    ("king and two bishops vs king", "8/8/4k3/8/8/2B1K3/4B3/8 w - - 0 1"),
    ("passed pawns on opposite wings", "8/6k1/8/8/1p6/8/1P4K1/8 w - - 0 1"),
    ("knight and pawns vs bishop and pawns, Black to move",
     "8/4kpp1/8/8/8/5N2/4KPP1/6b1 b - - 0 1"),
]

TACTICAL_AND_DEFENSIVE = [
    ("tactical", "back-rank mate available to White", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"),
    ("tactical", "back-rank mate available to Black", "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1"),
    ("tactical", "knight fork against king and rook", "4r1k1/5ppp/8/4N3/8/8/5PPP/6K1 w - - 0 1"),
    ("tactical", "knight fork against king and queen", "3qr1k1/5ppp/8/4N3/8/8/5PPP/6K1 w - - 0 1"),
    ("tactical", "queen and king mating net", "6k1/5ppp/8/8/8/8/5PPP/1Q4K1 w - - 0 1"),
    ("tactical", "skewer on the long diagonal", "6k1/5ppp/8/8/1B6/8/5PPP/6K1 w - - 0 1"),
    ("tactical", "pinned knight in front of the king", "4k3/8/8/8/8/4n3/8/3QK3 w - - 0 1"),
    ("tactical", "discovered attack available", "4k3/8/8/4N3/8/8/8/3RK3 w - - 0 1"),
    ("tactical", "promotion race with a rook present", "8/4P3/8/8/8/5k2/8/R3K3 w Q - 0 1"),
    ("tactical", "early queen sortie, Black must cover f7",
     "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 4 3"),
    ("defensive", "Black a rook down, must defend", "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1"),
    ("defensive", "White a queen down, must defend", "6k1/5ppp/8/1q6/8/8/5PPP/6K1 w - - 0 1"),
    ("defensive", "Black to move, in check from a bishop",
     "rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 3"),
    ("defensive", "Black to move against a kingside pawn advance",
     "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"),
    ("defensive", "White a piece down in a simplified position",
     "6k1/5ppp/8/8/3b4/8/5PPP/6K1 w - - 0 1"),
    ("defensive", "Black a rook down in a rook-and-pawn endgame",
     "8/5pk1/8/8/8/8/5PK1/R6r b - - 0 1"),
]


def build():
    rows, dropped = [], []
    seen_fens = set()

    def add(pid, category, description, provenance, board):
        if not board.is_valid():
            dropped.append({"id": pid, "reason": "board.is_valid() is False"})
            return
        if board.is_game_over():
            dropped.append({"id": pid, "reason": "terminal position"})
            return
        fen = board.fen()
        if fen in seen_fens:
            dropped.append({"id": pid, "reason": "duplicate FEN"})
            return
        seen_fens.add(fen)
        rows.append({
            "id": pid,
            "category": category,
            "description": description,
            "source": "constructed for this suite",
            "provenance": provenance,
            "fen": fen,
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
            "legal_move_count": board.legal_moves.count(),
            "is_check": board.is_check(),
            "fullmove_number": board.fullmove_number,
            "piece_count": len(board.piece_map()),
        })

    # --- openings / middlegames from named lines ---------------------------
    for line_idx, (name, moves) in enumerate(OPENING_LINES, 1):
        for ply in TRUNCATIONS:
            if ply > len(moves):
                continue
            board = chess.Board()
            ok = True
            for san in moves[:ply]:
                try:
                    board.push_san(san)
                except ValueError:
                    dropped.append({"id": f"X{line_idx:02d}-{ply}",
                                    "reason": f"illegal SAN {san!r} in line {name!r}"})
                    ok = False
                    break
            if not ok:
                continue
            category = "opening" if ply <= 8 else "middlegame"
            add(
                f"X{line_idx:02d}P{ply:02d}",
                category,
                f"{name}, after {ply} plies",
                "startpos + " + " ".join(moves[:ply]),
                board,
            )

    # --- endgames -----------------------------------------------------------
    for idx, (description, fen) in enumerate(ENDGAMES, 1):
        add(f"XEG{idx:02d}", "endgame", description, "literal FEN", chess.Board(fen))

    # --- tactical / defensive ----------------------------------------------
    counters = Counter()
    for category, description, fen in TACTICAL_AND_DEFENSIVE:
        counters[category] += 1
        prefix = "XTC" if category == "tactical" else "XDF"
        add(f"{prefix}{counters[category]:02d}", category, description,
            "literal FEN", chess.Board(fen))

    return rows, dropped


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/positions/extended.json"
    rows, dropped = build()

    print(f"BUILT {len(rows)}  DROPPED {len(dropped)}")
    for d in dropped:
        print(f"  DROP {d['id']}: {d['reason']}")
    print("by category:", dict(Counter(r["category"] for r in rows)))
    print("by side to move:", dict(Counter(r["side_to_move"] for r in rows)))

    payload = {
        "schema_version": 1,
        "name": "extended",
        "description": (
            "Extended evaluation suite built in Phase 3. Opening and middlegame "
            "positions are produced by playing named move sequences from the "
            "initial position, stored with each record so provenance is "
            "self-verifying. Endgame and tactical positions are hand-constructed "
            "literal FENs described by their material."
        ),
        "not_representative_notice": (
            "These positions are NOT sampled from master games and are NOT a "
            "statistically representative sample of practical chess. No population "
            "is being estimated; results carry no confidence interval."
        ),
        "generated_by": "evaluation/build_extended_suite.py",
        "count": len(rows),
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "side_to_move_counts": dict(Counter(r["side_to_move"] for r in rows)),
        "positions": rows,
        "dropped": dropped,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
