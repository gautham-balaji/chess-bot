"""Tests for the C7 dataset-audit measurement code.

The audit's conclusions drive a dataset rebuild, so the measurement instrument
itself needs to be correct. These tests pin the parts that a wrong answer would
silently corrupt:

  * the game-level split really is game-level, by CONTENT not by id
  * the split is deterministic and order-independent
  * sampling policies are deterministic, respect their bounds, and never pick
    the same ply twice
  * leakage accounting counts what it claims to count
  * replay matches the existing build_dataset replay on the same input

No Stockfish, no training, no file writes.
"""
import hashlib

import chess
import pytest

from training import build_dataset as B
from training import c7_dataset_audit as A

START = chess.STARTING_FEN
E4_GAME = "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7"
SHORT_GAME = "e4 e5 Qh5 Nc6 Bc4 Nf6 Qxf7#"


def make_replay(moves, game_id="G", row=0):
    return A.replay_game(game_id, row, moves)


# ============================================================ replay

def test_replay_records_one_entry_per_ply():
    rep = make_replay(E4_GAME)
    assert rep.n_tokens == 20
    assert rep.n_plies == 20
    for arr in (rep.exact, rep.position, rep.placement, rep.piece_counts,
                rep.white_to_move):
        assert len(arr) == 20


def test_replay_excludes_the_initial_position():
    """Ply 0 is identical in every game and would leak by construction."""
    rep = make_replay(E4_GAME)
    start_placement = A.key64(START.split(" ")[0])
    assert start_placement not in rep.placement


def test_replay_side_to_move_alternates_starting_with_black():
    rep = make_replay(E4_GAME)
    # After White's first move it is Black to move.
    assert rep.white_to_move[0] is False
    assert rep.white_to_move[1] is True
    assert all(rep.white_to_move[i] != rep.white_to_move[i + 1]
               for i in range(rep.n_plies - 1))


def test_replay_detects_terminal_checkmate():
    rep = make_replay(SHORT_GAME)
    assert rep.n_plies == 7
    assert rep.terminal_checkmate is True
    assert rep.terminal_stalemate is False


def test_replay_stops_at_bad_san_and_records_the_token():
    rep = make_replay("e4 e5 Zz9 Nf3")
    assert rep.n_plies == 2
    assert rep.bad_san_token == "Zz9"


def test_replay_handles_a_non_string_moves_cell():
    rep = make_replay(None)
    assert rep.n_plies == 0 and rep.n_tokens == 0


def test_piece_count_decreases_on_a_capture():
    rep = make_replay("e4 d5 exd5")
    assert rep.piece_counts == [32, 32, 31]


def test_replay_agrees_with_build_dataset_on_the_first_twenty_plies():
    """The audit must describe the SAME pipeline dataset_v1 came from."""
    for moves in (E4_GAME, SHORT_GAME, "d4 d5 c4 c6 cxd5 e6"):
        outcome = B.board_after_fullmove(moves, B.MAX_FULLMOVES)
        rep = make_replay(moves)
        n = min(rep.n_plies, B.MAX_PLIES)
        assert outcome.plies_played == n
        assert rep.exact[n - 1] == A.key64(outcome.board.fen())


# ============================================================ identity levels

def test_the_three_identity_levels_separate_what_they_claim():
    # Same placement, different side to move / castling: placement collides,
    # position does not.
    a = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    b = "4k3/8/8/8/8/8/8/4K3 b - - 0 1"
    assert A.key64(a.split(" ")[0]) == A.key64(b.split(" ")[0])
    assert A.key64(" ".join(a.split(" ")[:4])) != A.key64(" ".join(b.split(" ")[:4]))


def test_exact_level_distinguishes_move_counters_but_position_level_does_not():
    a = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    b = "4k3/8/8/8/8/8/8/4K3 w - - 9 40"
    assert A.key64(a) != A.key64(b)
    assert A.key64(" ".join(a.split(" ")[:4])) == A.key64(" ".join(b.split(" ")[:4]))


# ============================================================ game-level split

def test_split_unit_is_content_not_id():
    """Two rows with different ids but identical moves are ONE unit."""
    keys = A.game_unit_keys(["idA", "idB"], [E4_GAME, E4_GAME])
    assert keys[0] == keys[1]


def test_split_unit_separates_different_games():
    keys = A.game_unit_keys(["idA", "idB"], [E4_GAME, SHORT_GAME])
    assert keys[0] != keys[1]


def test_duplicate_rows_land_on_the_same_side():
    ids = [f"id{i}" for i in range(100)]
    moves = [f"e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d{i % 7 + 1}"
             for i in range(100)]
    keys = A.game_unit_keys(ids, moves)
    train, test, train_u, test_u = A.game_level_split(keys, 42, 0.2)
    assert not (train_u & test_u)
    for i, k in enumerate(keys):
        side_train = i in train
        for j, k2 in enumerate(keys):
            if k2 == k:
                assert (j in train) == side_train


def test_split_partitions_every_row_exactly_once():
    keys = A.game_unit_keys([f"id{i}" for i in range(500)],
                            [f"e4 e5 Nf3 Nc6 Bb5 a{i}" for i in range(500)])
    train, test, _, _ = A.game_level_split(keys, 42, 0.2)
    assert train & test == set()
    assert len(train | test) == 500


def test_split_respects_the_test_fraction():
    keys = A.game_unit_keys([f"id{i}" for i in range(1000)],
                            [f"e4 e{i}" for i in range(1000)])
    _, _, train_u, test_u = A.game_level_split(keys, 42, 0.2)
    assert len(test_u) == 200
    assert len(train_u) == 800


def test_split_is_deterministic_across_calls():
    keys = A.game_unit_keys([f"id{i}" for i in range(300)],
                            [f"e4 e{i}" for i in range(300)])
    a = A.game_level_split(keys, 42, 0.2)
    b = A.game_level_split(keys, 42, 0.2)
    assert a[2] == b[2] and a[3] == b[3]


def test_split_is_independent_of_row_order():
    """Reordering games.csv must not change which games are in test."""
    ids = [f"id{i}" for i in range(300)]
    moves = [f"e4 e{i}" for i in range(300)]
    keys = A.game_unit_keys(ids, moves)
    _, _, train_a, test_a = A.game_level_split(keys, 42, 0.2)

    order = list(reversed(range(300)))
    keys_r = A.game_unit_keys([ids[i] for i in order], [moves[i] for i in order])
    _, _, train_b, test_b = A.game_level_split(keys_r, 42, 0.2)
    assert test_a == test_b and train_a == train_b


def test_different_seeds_give_different_splits():
    keys = A.game_unit_keys([f"id{i}" for i in range(300)],
                            [f"e4 e{i}" for i in range(300)])
    _, _, _, test_42 = A.game_level_split(keys, 42, 0.2)
    _, _, _, test_7 = A.game_level_split(keys, 7, 0.2)
    assert test_42 != test_7


# ============================================================ sampling policies

@pytest.mark.parametrize("k", [2, 4, 6, 8, 12])
def test_evenly_spaced_never_exceeds_k(k):
    rep = make_replay(E4_GAME + " " + E4_GAME.replace("e4", "h3", 1))
    picks = A.policy_evenly_spaced(rep, k, min_ply=A.MIN_PLY_DEFAULT)
    assert len(picks) <= k


def test_evenly_spaced_returns_strictly_increasing_unique_plies():
    rep = make_replay(" ".join(["e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5"] * 3))
    picks = A.policy_evenly_spaced(rep, 8, min_ply=A.MIN_PLY_DEFAULT)
    assert picks == sorted(set(picks))


def test_evenly_spaced_respects_min_ply():
    rep = make_replay(E4_GAME)
    picks = A.policy_evenly_spaced(rep, 6, min_ply=12)
    assert all(p >= 11 for p in picks), picks


def test_evenly_spaced_returns_nothing_for_a_game_shorter_than_min_ply():
    rep = make_replay("e4 e5")
    assert A.policy_evenly_spaced(rep, 4, min_ply=12) == []


def test_evenly_spaced_is_deterministic():
    rep = make_replay(E4_GAME)
    first = A.policy_evenly_spaced(rep, 4, min_ply=12)
    for _ in range(5):
        assert A.policy_evenly_spaced(rep, 4, min_ply=12) == first


def test_min_gap_bounds_the_count_on_a_short_span():
    """A 6-ply span cannot yield 8 positions at least 4 plies apart."""
    rep = make_replay(E4_GAME[: len("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O")])
    picks = A.policy_evenly_spaced(rep, 8, min_ply=1, min_gap=4)
    span = rep.n_plies - 1
    assert len(picks) <= max(1, span // 4 + 1)


def test_v1_policy_reproduces_the_dataset_v1_derivation():
    for moves in (E4_GAME, SHORT_GAME, "d4 d5"):
        rep = make_replay(moves)
        picks = A.policy_last_of_first_20(rep)
        assert len(picks) == 1
        outcome = B.board_after_fullmove(moves, B.MAX_FULLMOVES)
        assert picks[0] == outcome.plies_played - 1


def test_phase_stratified_respects_min_ply_and_is_sorted():
    rep = make_replay(" ".join(["e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7"] * 4))
    picks = A.policy_phase_stratified(rep, 3, min_ply=12)
    assert picks == sorted(set(picks))
    assert all(p >= 11 for p in picks)


def test_every_registered_policy_is_deterministic_and_in_range():
    rep = make_replay(" ".join(["e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5"] * 3))
    for name, (_, selector) in A.POLICIES.items():
        first = selector(rep)
        assert selector(rep) == first, name
        assert first == sorted(set(first)), name
        assert all(0 <= p < rep.n_plies for p in first), name


# ============================================================ phase classifier

def test_phase_classifier_is_total_and_reproducible():
    for ply in (1, 15, 21, 80):
        for pieces in (2, 12, 13, 32):
            assert A.classify_phase(ply, pieces) in {
                "opening", "middlegame", "endgame"}


def test_endgame_is_decided_by_piece_count_regardless_of_ply():
    assert A.classify_phase(3, 10) == "endgame"
    assert A.classify_phase(200, 12) == "endgame"


def test_opening_and_middlegame_are_split_at_the_ply_boundary():
    assert A.classify_phase(A.OPENING_MAX_PLY, 32) == "opening"
    assert A.classify_phase(A.OPENING_MAX_PLY + 1, 32) == "middlegame"


# ============================================================ leakage accounting

def _two_game_corpus():
    """Two games that share a placement at a known ply, and nothing else."""
    shared = "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6"
    g1 = make_replay(shared + " c3 O-O", "g1", 0)
    g2 = make_replay(shared + " h3 Nb8", "g2", 1)
    return [g1, g2]


def test_leakage_detects_a_shared_placement_across_the_split():
    reps = _two_game_corpus()
    # Force game 0 into train and game 1 into test, then take every ply.
    result = A.evaluate_policy(
        "t", "t", lambda r: list(range(r.n_plies)), reps, {0}, {1})
    lk = result["leakage"]["placement"]
    assert lk["keys_in_both_train_and_test"] > 0
    assert lk["test_rows_whose_key_also_appears_in_train"] > 0
    assert 0.0 < lk["leaked_test_fraction"] <= 1.0


def test_no_leakage_when_the_shared_prefix_is_excluded():
    reps = _two_game_corpus()
    result = A.evaluate_policy(
        "t", "t", lambda r: [r.n_plies - 1], reps, {0}, {1})
    assert result["leakage"]["placement"]["keys_in_both_train_and_test"] == 0
    assert result["leakage"]["placement"]["leaked_test_fraction"] == 0.0


def test_position_counts_add_up():
    reps = _two_game_corpus()
    result = A.evaluate_policy(
        "t", "t", lambda r: list(range(r.n_plies)), reps, {0}, {1})
    assert (result["positions_train"] + result["positions_test"]
            == result["positions_total"])
    assert result["positions_total"] == sum(r.n_plies for r in reps)


def test_canonical_rows_suppress_duplicate_games():
    reps = _two_game_corpus() + [make_replay("e4 e5 Nf3 Nc6", "g3", 2)]
    full = A.evaluate_policy("t", "t", lambda r: [0], reps, {0, 2}, {1})
    limited = A.evaluate_policy("t", "t", lambda r: [0], reps, {0, 2}, {1},
                                canonical_rows={0, 1})
    assert full["positions_total"] == 3
    assert limited["positions_total"] == 2


def test_scrubbed_test_count_removes_exactly_the_colliding_keys():
    reps = _two_game_corpus()
    result = A.evaluate_policy(
        "t", "t", lambda r: list(range(r.n_plies)), reps, {0}, {1})
    lk = result["leakage"]["placement"]
    assert (lk["scrubbed_test_positions"]
            == lk["deduped_test_positions"] - lk["keys_in_both_train_and_test"])


def test_side_to_move_and_phase_fractions_sum_to_one():
    reps = _two_game_corpus()
    result = A.evaluate_policy(
        "t", "t", lambda r: list(range(r.n_plies)), reps, {0}, {1})
    assert round(sum(result["side_to_move_fractions"].values()), 6) == 1.0
    assert round(sum(result["phase_fractions"].values()), 6) == 1.0


# ============================================================ hashing

def test_sha256_text_matches_hashlib():
    assert A.sha256_text("abc") == hashlib.sha256(b"abc").hexdigest()


def test_key64_is_stable_and_64_bit():
    k = A.key64("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert k == A.key64("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert 0 <= k < 2 ** 64


# ============================================================ read-only guarantee

def test_the_audit_module_never_writes_outside_its_out_path():
    source = (A.__file__ and open(A.__file__, encoding="utf-8").read()) or ""
    for banned in ("models/", "weight_model", "cnn_model.keras", "engine.py"):
        assert banned not in source, banned


def test_the_audit_imports_no_training_or_keras_code():
    source = open(A.__file__, encoding="utf-8").read()
    for banned in ("import keras", "from keras", "tensorflow",
                   "from training import train"):
        assert banned not in source, banned
