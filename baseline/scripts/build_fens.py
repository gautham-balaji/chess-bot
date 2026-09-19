"""Phase 0 baseline: build the FIXED deterministic FEN suite.

No randomness anywhere. Running this again produces a byte-identical file.
Read-only with respect to application code; writes only baseline/fens.json.

Usage:  python baseline/scripts/build_fens.py baseline/fens.json
"""
import json
import sys
from collections import Counter

import chess

# (id, category, description, source, spec)
# spec is ("moves", [san, ...]) played from the start position, or ("fen", "<fen>").
# Move-list provenance is preferred where possible: it is self-verifying.
SPEC = [
    # ---- OPENING ----
    ("OP01", "opening", "Initial position", "standard", ("moves", [])),
    ("OP02", "opening", "After 1.e4 (Black to move)", "derived", ("moves", ["e4"])),
    ("OP03", "opening", "Open Game after 1.e4 e5", "derived", ("moves", ["e4", "e5"])),
    ("OP04", "opening", "Sicilian Defence after 1.e4 c5", "benchmark_stockfish.py test_positions",
     ("moves", ["e4", "c5"])),
    ("OP05", "opening", "French Defence after 1.e4 e6", "derived", ("moves", ["e4", "e6"])),
    ("OP06", "opening", "Caro-Kann after 1.e4 c6", "derived", ("moves", ["e4", "c6"])),
    ("OP07", "opening", "Queen pawn: 1.d4 d5 2.c4", "derived", ("moves", ["d4", "d5", "c4"])),
    ("OP08", "opening", "Slav: 1.d4 d5 2.c4 c6", "derived", ("moves", ["d4", "d5", "c4", "c6"])),
    ("OP09", "opening", "1.e4 e5 2.Nf3 Nf6", "benchmark_stockfish.py test_positions",
     ("moves", ["e4", "e5", "Nf3", "Nf6"])),
    ("OP10", "opening", "Ruy Lopez after 3.Bb5 (Black to move)", "derived",
     ("moves", ["e4", "e5", "Nf3", "Nc6", "Bb5"])),
    ("OP11", "opening", "Two Knights: 3.Bc4 Nf6", "benchmark_stockfish.py test_positions",
     ("moves", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"])),
    ("OP12", "opening", "Giuoco Piano: 3...Bc5 4.d3 Nf6", "benchmark_stockfish.py test_positions",
     ("moves", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "d3", "Nf6"])),
    ("OP13", "opening", "King's Indian setup", "derived",
     ("moves", ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7"])),
    ("OP14", "opening", "Nimzo-Indian after 3...Bb4", "derived",
     ("moves", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"])),
    ("OP15", "opening", "English: 1.c4 e5 2.Nc3", "derived", ("moves", ["c4", "e5", "Nc3"])),
    ("OP16", "opening", "1.d4 Nf6 2.c4 (Black to move)", "derived", ("moves", ["d4", "Nf6", "c4"])),
    ("OP17", "opening", "Scandinavian: 1.e4 d5 2.exd5 (Black to move)", "derived",
     ("moves", ["e4", "d5", "exd5"])),
    ("OP18", "opening", "Scotch: 1.e4 e5 2.Nf3 Nc6 3.d4 (Black to move)", "derived",
     ("moves", ["e4", "e5", "Nf3", "Nc6", "d4"])),

    # ---- MIDDLEGAME ----
    ("MG01", "middlegame", "Ruy Lopez Closed, both sides castled", "derived",
     ("moves", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7",
                "Re1", "b5", "Bb3", "d6", "c3", "O-O"])),
    ("MG02", "middlegame", "Sicilian Najdorf main line", "derived",
     ("moves", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6",
                "Be3", "e5", "Nb3", "Be7"])),
    ("MG03", "middlegame", "Queen's Gambit Declined, Orthodox", "derived",
     ("moves", ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O",
                "Nf3", "h6", "Bh4", "b6"])),
    ("MG04", "middlegame", "French Winawer structure", "derived",
     ("moves", ["e4", "e6", "d4", "d5", "Nc3", "Bb4", "e5", "c5", "a3", "Bxc3+",
                "bxc3", "Ne7"])),
    ("MG05", "middlegame", "Italian, both castled, central tension", "derived",
     ("moves", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6",
                "O-O", "O-O", "Re1", "a6"])),
    ("MG06", "middlegame", "King's Indian, closed centre", "derived",
     ("moves", ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O",
                "Be2", "e5", "d5", "Nbd7"])),
    ("MG07", "middlegame", "Caro-Kann Classical", "derived",
     ("moves", ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5", "Ng3", "Bg6",
                "h4", "h6", "Nf3", "Nd7"])),
    ("MG08", "middlegame", "Queen's Gambit Accepted, open centre", "derived",
     ("moves", ["d4", "d5", "c4", "dxc4", "Nf3", "Nf6", "e3", "e6", "Bxc4", "c5",
                "O-O", "a6", "dxc5", "Bxc5"])),

    ("MG09", "middlegame", "Ruy Lopez Closed, one ply deeper (Black to move)", "derived",
     ("moves", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7",
                "Re1", "b5", "Bb3", "d6", "c3", "O-O", "h3"])),
    ("MG10", "middlegame", "Sicilian Najdorf, one ply deeper (Black to move)", "derived",
     ("moves", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6",
                "Be3", "e5", "Nb3", "Be7", "f3"])),
    ("MG11", "middlegame", "QGD Orthodox, one ply deeper (Black to move)", "derived",
     ("moves", ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O",
                "Nf3", "h6", "Bh4", "b6", "cxd5"])),
    ("MG12", "middlegame", "Italian, one ply deeper (Black to move)", "derived",
     ("moves", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6",
                "O-O", "O-O", "Re1", "a6", "Nbd2"])),

    # ---- ENDGAME (material described literally; no theory-name claims) ----
    ("EG01", "endgame", "King and pawn vs king, White to move", "constructed",
     ("fen", "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1")),
    ("EG02", "endgame", "King and pawn vs king, Black to move", "constructed",
     ("fen", "8/8/8/4k3/8/4K3/4P3/8 b - - 0 1")),
    ("EG03", "endgame", "King and rook vs king, White to move", "constructed",
     ("fen", "8/8/4k3/8/8/4K3/8/4R3 w - - 0 1")),
    ("EG04", "endgame", "King and queen vs king, Black to move (losing side)", "constructed",
     ("fen", "8/8/4k3/8/8/4K3/8/4Q3 b - - 0 1")),
    ("EG05", "endgame", "Rook and pawn vs rook and pawn", "constructed",
     ("fen", "8/5pk1/8/8/8/8/5PK1/R6r w - - 0 1")),
    ("EG06", "endgame", "Symmetrical two-pawn king-and-pawn endgame", "constructed",
     ("fen", "8/5ppk/8/8/8/8/5PPK/8 w - - 0 1")),
    ("EG07", "endgame", "Bishop and pawn vs bishop and pawn", "constructed",
     ("fen", "8/5pk1/8/8/8/2B5/5PK1/6b1 b - - 0 1")),
    ("EG08", "endgame", "Knight and pawns vs bishop and pawns", "constructed",
     ("fen", "8/4kpp1/8/8/8/5N2/4KPP1/6b1 w - - 0 1")),
    ("EG09", "endgame", "Queen and rook vs lone king, Black to move", "constructed",
     ("fen", "8/8/4k3/8/8/4K3/8/3QR3 b - - 0 1")),
    ("EG10", "endgame", "Rook endgame, extra pawn for White", "constructed",
     ("fen", "8/5pk1/6p1/8/8/6P1/5PKP/R6r w - - 0 1")),
    ("EG11", "endgame", "Rook and pawn vs rook and pawn, Black to move", "constructed",
     ("fen", "8/5pk1/8/8/8/8/5PK1/R6r b - - 0 1")),
    ("EG12", "endgame", "King and rook vs king, Black to move (losing side)", "constructed",
     ("fen", "8/8/4k3/8/8/4K3/8/4R3 b - - 0 1")),

    # ---- TACTICAL ----
    ("TC01", "tactical", "Back-rank mate available to White", "constructed",
     ("fen", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")),
    ("TC02", "tactical", "Back-rank mate available to Black", "constructed",
     ("fen", "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1")),
    ("TC03", "tactical", "Early queen sortie, Black must address f7", "constructed",
     ("fen", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 4 3")),
    ("TC04", "tactical", "Knight fork available against king and rook", "constructed",
     ("fen", "4r1k1/5ppp/8/4N3/8/8/5PPP/6K1 w - - 0 1")),
    ("TC05", "tactical", "Queen vs exposed king, mating net available", "constructed",
     ("fen", "6k1/5ppp/8/8/8/8/5PPP/1Q4K1 w - - 0 1")),
    ("TC06", "tactical", "Open Italian position after White castles", "constructed",
     ("fen", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4")),

    # ---- DEFENSIVE (side to move is materially or positionally worse) ----
    ("DF01", "defensive", "Black a rook down, must defend", "constructed",
     ("fen", "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1")),
    ("DF02", "defensive", "White a queen down, must defend", "constructed",
     ("fen", "6k1/5ppp/8/1q6/8/8/5PPP/6K1 w - - 0 1")),
    ("DF03", "defensive", "Black to move against a weakening kingside advance", "constructed",
     ("fen", "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")),
    ("DF04", "defensive", "Symmetrical rook endgame, both sides may castle", "constructed",
     ("fen", "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 1")),
]


def build():
    rows, dropped = [], []
    for pid, cat, desc, src, spec in SPEC:
        try:
            if spec[0] == "moves":
                board = chess.Board()
                for san in spec[1]:
                    board.push_san(san)
                provenance = "startpos + " + " ".join(spec[1]) if spec[1] else "startpos"
            else:
                board = chess.Board(spec[1])
                provenance = "literal FEN"

            if not board.is_valid():
                dropped.append({"id": pid, "reason": "board.is_valid() is False",
                                "detail": board.fen()})
                continue
            if board.is_game_over():
                dropped.append({"id": pid,
                                "reason": f"terminal position ({board.result()}); "
                                          "neither engine nor Stockfish can return a move",
                                "detail": board.fen()})
                continue

            rows.append({
                "id": pid,
                "category": cat,
                "description": desc,
                "source": src,
                "provenance": provenance,
                "fen": board.fen(),
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "legal_move_count": board.legal_moves.count(),
                "is_check": board.is_check(),
                "fullmove_number": board.fullmove_number,
                "piece_count": len(board.piece_map()),
            })
        except Exception as exc:  # noqa: BLE001 - we want the reason recorded, not raised
            dropped.append({"id": pid, "reason": repr(exc), "detail": str(spec)})
    return rows, dropped


def main():
    out_path = sys.argv[1]
    rows, dropped = build()

    print(f"BUILT {len(rows)}  DROPPED {len(dropped)}")
    for d in dropped:
        print(f"  DROP {d['id']}: {d['reason']}")
    print("by category:", dict(Counter(r["category"] for r in rows)))
    print("by side to move:", dict(Counter(r["side_to_move"] for r in rows)))

    payload = {
        "schema_version": 1,
        "description": (
            "Phase 0 fixed baseline FEN suite. Hand-curated and fully deterministic: "
            "this file is regenerated byte-identically by rerunning build_fens.py. "
            "It is NOT claimed to be a statistically representative sample of any "
            "position distribution; it is a fixed comparison set only."
        ),
        "generated_by": "baseline/scripts/build_fens.py",
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
