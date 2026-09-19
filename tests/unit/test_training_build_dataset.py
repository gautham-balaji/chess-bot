"""Tests for training/build_dataset.py - sampling, position generation, dedup.

These cover everything that does NOT need a Stockfish binary, using tiny
in-memory DataFrames. No games.csv, no machine-specific paths.

The one test that does need a real engine is marked `needs_stockfish` and is
auto-skipped when no binary resolves (see tests/conftest.py).
"""
import json

import chess
import pandas as pd
import pytest

from training import build_dataset as B
from training import labels as L


def _frame(rows):
    """Minimal games.csv-shaped frame."""
    return pd.DataFrame([
        {"id": r.get("id", f"g{i}"), "moves": r["moves"], "turns": 0,
         "winner": "white", "white_rating": 1500, "black_rating": 1500}
        for i, r in enumerate(rows)
    ])


# ==================================================================== sampling

def test_same_seed_gives_the_same_sampled_game_ids():
    df = _frame([{"moves": "e4 e5", "id": f"game{i}"} for i in range(200)])
    a = list(B.sample_games(df, 50, seed=42)["id"])
    b = list(B.sample_games(df, 50, seed=42)["id"])
    assert a == b


def test_same_seed_gives_the_same_ORDER_too():
    """Order matters: dedup keeps the first occurrence of each FEN."""
    df = _frame([{"moves": "e4 e5", "id": f"game{i}"} for i in range(200)])
    assert list(B.sample_games(df, 30, 42)["id"]) == list(B.sample_games(df, 30, 42)["id"])


def test_different_seeds_give_different_samples():
    df = _frame([{"moves": "e4 e5", "id": f"game{i}"} for i in range(200)])
    assert list(B.sample_games(df, 50, 42)["id"]) != list(B.sample_games(df, 50, 7)["id"])


def test_sample_size_is_respected():
    df = _frame([{"moves": "e4 e5", "id": f"game{i}"} for i in range(200)])
    assert len(B.sample_games(df, 25, 42)) == 25


# ==================================================================== positions

def test_full_length_game_stops_at_the_ply_cap():
    """20 plies available -> exactly 20 played. Preserved notebook semantics."""
    moves = "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7"
    out = B.board_after_fullmove(moves, B.MAX_FULLMOVES)
    assert out.plies_played == B.MAX_PLIES == 20
    assert out.truncated_by_bad_san is False
    assert out.invalid_board is False


def test_short_game_yields_a_position_below_the_cap():
    """This is why 8.4% of the original dataset is NOT at ply 20."""
    out = B.board_after_fullmove("e4 e5 Nf3", B.MAX_FULLMOVES)
    assert out.plies_played == 3
    assert out.plies_played < B.MAX_PLIES


def test_malformed_san_is_recorded_not_swallowed():
    """The notebook's bare `except: break` hid this. It must now be observable."""
    out = B.board_after_fullmove("e4 e5 Zz9 Nf3", B.MAX_FULLMOVES)
    assert out.plies_played == 2
    assert out.truncated_by_bad_san is True
    assert out.bad_san_token == "Zz9"


def test_clean_game_is_not_flagged_as_bad_san():
    out = B.board_after_fullmove("e4 e5 Nf3", B.MAX_FULLMOVES)
    assert out.truncated_by_bad_san is False
    assert out.bad_san_token is None


def test_non_string_moves_yields_the_initial_position():
    out = B.board_after_fullmove(float("nan"), B.MAX_FULLMOVES)
    assert out.plies_played == 0
    assert out.board.fen() == chess.STARTING_FEN


def test_generate_positions_records_ply_and_side_to_move():
    df = _frame([{"moves": "e4"}, {"moves": "e4 e5"}])
    rows, stats = B.generate_positions(df)
    assert len(rows) == 2
    assert rows[0]["plies_played"] == 1 and rows[0]["side_to_move"] == "black"
    assert rows[1]["plies_played"] == 2 and rows[1]["side_to_move"] == "white"
    assert stats["counts"]["valid_positions"] == 2
    assert stats["ply_histogram"] == {1: 1, 2: 1}


def test_generate_positions_records_required_audit_fields():
    rows, _ = B.generate_positions(_frame([{"moves": "e4 e5", "id": "abc"}]))
    row = rows[0]
    for key in ("game_id", "source_row_index", "fen", "side_to_move",
                "plies_played", "reached_ply_cap", "truncated_by_bad_san",
                "is_checkmate", "legal_move_count"):
        assert key in row, f"missing audit field: {key}"
    assert row["game_id"] == "abc"


def test_bad_san_truncations_are_counted():
    _, stats = B.generate_positions(_frame([{"moves": "e4 Zz9"}, {"moves": "e4 e5"}]))
    assert stats["counts"]["bad_san_truncations"] == 1


def test_checkmate_positions_are_flagged():
    rows, _ = B.generate_positions(_frame([{"moves": "f3 e5 g4 Qh4#"}]))
    assert rows[0]["is_checkmate"] is True
    assert rows[0]["legal_move_count"] == 0


# ==================================================================== dedup

def test_duplicate_fens_collapse_to_one_record():
    rows = [{"fen": "A"}, {"fen": "B"}, {"fen": "A"}, {"fen": "A"}]
    kept, stats = B.deduplicate_by_fen(rows)
    assert [r["fen"] for r in kept] == ["A", "B"]
    assert stats["unique_fens"] == 2
    assert stats["duplicate_rows_removed"] == 2
    assert stats["fens_occurring_more_than_once"] == 1
    assert stats["max_occurrences_of_one_fen"] == 3


def test_dedup_keeps_the_first_occurrence():
    rows = [{"fen": "A", "game_id": "first"}, {"fen": "A", "game_id": "second"}]
    kept, _ = B.deduplicate_by_fen(rows)
    assert kept[0]["game_id"] == "first"


def test_dedup_uses_the_complete_fen_not_piece_placement():
    """Same placement, different side to move -> two distinct positions."""
    same_placement = "4k3/8/8/8/8/8/8/4K3"
    rows = [{"fen": f"{same_placement} w - - 0 1"},
            {"fen": f"{same_placement} b - - 0 1"}]
    kept, stats = B.deduplicate_by_fen(rows)
    assert stats["unique_fens"] == 2, "side to move must remain distinguishing"
    assert len(kept) == 2


def test_dedup_distinguishes_castling_rights():
    rows = [{"fen": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"},
            {"fen": "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"}]
    assert B.deduplicate_by_fen(rows)[1]["unique_fens"] == 2


def test_dedup_of_unique_input_is_a_no_op():
    rows = [{"fen": f"fen{i}"} for i in range(5)]
    kept, stats = B.deduplicate_by_fen(rows)
    assert len(kept) == 5 and stats["duplicate_rows_removed"] == 0


def test_dedup_is_deterministic():
    rows = [{"fen": "A"}, {"fen": "B"}, {"fen": "A"}, {"fen": "C"}, {"fen": "B"}]
    assert B.deduplicate_by_fen(rows)[0] == B.deduplicate_by_fen(rows)[0]


# ==================================================================== artifact

def test_records_are_written_as_one_json_object_per_line(tmp_path):
    rows = [
        {"game_id": "a", "source_row_index": 0, "fen": "f1", "side_to_move": "white",
         "plies_played": 20, "reached_ply_cap": True, "san_tokens_available": 40,
         "truncated_by_bad_san": False, "bad_san_token": None, "is_checkmate": False,
         "is_stalemate": False, "is_game_over": False, "legal_move_count": 30,
         "eval_type": "cp", "raw_stockfish_value": 12, "raw_value_perspective": "side_to_move",
         "label": 12, "label_perspective": "white", "was_clipped": False},
    ]
    path = tmp_path / "out.jsonl"
    digest = B.write_records(rows, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["fen"] == "f1"
    assert record["label"] == 12
    assert record["label_perspective"] == "white"
    assert len(digest) == 64


def test_writing_the_same_records_twice_gives_the_same_checksum(tmp_path):
    rows = [{"game_id": "a", "fen": "f1", "side_to_move": "white", "label": 5,
             "eval_type": "cp", "raw_stockfish_value": 5, "plies_played": 3,
             "label_perspective": "white"}]
    a = B.write_records(rows, tmp_path / "a.jsonl")
    b = B.write_records(rows, tmp_path / "b.jsonl")
    assert a == b


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "x.bin"
    p.write_bytes(b"chess")
    assert B.sha256_file(p) == hashlib.sha256(b"chess").hexdigest()


# ==================================================================== config

def test_pipeline_pins_the_phase3_stockfish_reference_configuration():
    assert B.SF_DEPTH == 8
    assert B.SF_THREADS == 1
    assert B.SF_HASH_MB == 16
    assert B.SF_CLEAR_HASH_PER_POSITION is True


def test_pipeline_preserves_the_notebook_sampling_semantics():
    assert B.SAMPLE_SIZE == 10_000
    assert B.SAMPLE_SEED == 42
    assert B.MAX_FULLMOVES == 10 and B.MAX_PLIES == 20


def test_expected_source_identity_matches_the_audit():
    assert B.EXPECTED_SOURCE["sha256"] == (
        "e7aadff104a610afb5403caf81c1461babecb0dc87760ff5eea7dc0e7a4a8129")
    assert B.EXPECTED_SOURCE["size_bytes"] == 7_672_655
    assert B.EXPECTED_SOURCE["row_count"] == 20_058


# ==================================================================== live engine

@pytest.mark.needs_stockfish
def test_live_engine_agrees_with_the_label_policy():
    """One small end-to-end check that the wrapper semantics still hold.

    Guards the empirical basis of the whole policy: if a future Stockfish or
    wrapper changes how mate or perspective is reported, this fails loudly
    instead of silently corrupting a dataset.
    """
    engine, info, _ = B.open_engine()
    try:
        # Black to move, White up a queen: stm-relative value must be negative,
        # and the policy must turn it into a positive (White-good) label.
        fen = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1"
        engine.send_ucinewgame_command()
        engine.set_fen_position(fen)
        ev = engine.get_evaluation()
        assert ev["type"] == "cp"
        assert ev["value"] < 0, "expected a side-to-move-relative (Black) value"

        lab = L.make_label(ev["type"], ev["value"], side_to_move_is_white=False)
        assert lab.label > 0, "White is up a queen, so the label must be positive"

        # An already-checkmated position must report mate 0.
        mated = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        assert chess.Board(mated).is_checkmate()
        engine.send_ucinewgame_command()
        engine.set_fen_position(mated)
        ev2 = engine.get_evaluation()
        assert ev2 == {"type": "mate", "value": 0}

        lab2 = L.make_label("mate", 0, side_to_move_is_white=True, is_checkmate=True)
        assert lab2.label == -L.MATE_SCORE_BASE
        assert lab2.mate_zero_consistent is True
    finally:
        engine.send_quit_command()

    assert info.threads == 1 and info.hash_mb == 16 and info.depth == 8
