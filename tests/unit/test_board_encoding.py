"""Unit tests for engine.board_to_planes - the CNN's only input representation.

The encoding is 8x8x12: channels 0-5 are White P,N,B,R,Q,K and 6-11 the Black
equivalents. Row 0 is rank 8 (row = 7 - square // 8), column 0 is file a.

These tests describe the representation as built. They do not redesign it.
"""
import chess
import numpy as np
import pytest

PIECE_CHANNEL = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
BLACK_OFFSET = 6


# ------------------------------------------------------------------ shape / dtype

def test_shape_is_8x8x12(engine_mod, start_board):
    assert engine_mod.board_to_planes(start_board).shape == (8, 8, 12)


def test_dtype_is_float32(engine_mod, start_board):
    assert engine_mod.board_to_planes(start_board).dtype == np.float32


def test_values_are_binary(engine_mod, start_board):
    planes = engine_mod.board_to_planes(start_board)
    assert set(np.unique(planes)).issubset({0.0, 1.0})


# ------------------------------------------------------------------ piece counts

def test_kings_only_board_activates_exactly_two_cells(engine_mod, empty_board):
    planes = engine_mod.board_to_planes(empty_board)
    assert planes.sum() == 2, "a kings-only board must encode exactly two pieces"


def test_start_position_activates_32_cells(engine_mod, start_board):
    planes = engine_mod.board_to_planes(start_board)
    assert planes.sum() == 32


def test_active_cells_equal_piece_count(engine_mod, sample_fens):
    """The number of set cells must equal the number of pieces on the board."""
    for entry in sample_fens:
        board = chess.Board(entry["fen"])
        planes = engine_mod.board_to_planes(board)
        assert planes.sum() == len(board.piece_map()), f"mismatch for {entry['id']}"


def test_start_position_channel_distribution(engine_mod, start_board):
    planes = engine_mod.board_to_planes(start_board)
    per_channel = planes.sum(axis=(0, 1))
    expected = [8, 2, 2, 2, 1, 1] * 2  # P,N,B,R,Q,K for White then Black
    assert list(per_channel) == pytest.approx(expected)


# ------------------------------------------------------------------ single pieces

@pytest.mark.parametrize(
    "fen,symbol,square_name",
    [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "K", "e1"),
        ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "R", "a1"),
        ("4k3/8/8/8/4Q3/8/8/4K3 w - - 0 1", "Q", "e4"),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "P", "e2"),
    ],
)
def test_white_piece_lands_in_correct_channel_and_cell(
    engine_mod, fen, symbol, square_name
):
    planes = engine_mod.board_to_planes(chess.Board(fen))
    square = chess.parse_square(square_name)
    row, col = 7 - (square // 8), square % 8
    channel = PIECE_CHANNEL[symbol]

    assert planes[row, col, channel] == 1.0
    # and it is the only cell set in that channel
    assert planes[:, :, channel].sum() == 1.0


def test_black_pieces_use_the_upper_six_channels(engine_mod):
    planes = engine_mod.board_to_planes(chess.Board("4k3/8/8/8/4q3/8/8/4K3 w - - 0 1"))
    square = chess.parse_square("e4")
    row, col = 7 - (square // 8), square % 8

    assert planes[row, col, PIECE_CHANNEL["Q"] + BLACK_OFFSET] == 1.0
    assert planes[row, col, PIECE_CHANNEL["Q"]] == 0.0, "must not leak into White's channel"


def test_white_and_black_same_piece_type_do_not_collide(engine_mod):
    """A White and a Black queen on different squares occupy different channels."""
    planes = engine_mod.board_to_planes(chess.Board("4k3/8/8/8/8/8/3qQ3/4K3 w - - 0 1"))
    assert planes[:, :, PIECE_CHANNEL["Q"]].sum() == 1.0
    assert planes[:, :, PIECE_CHANNEL["Q"] + BLACK_OFFSET].sum() == 1.0


# ------------------------------------------------------------------ orientation

def test_orientation_a1_is_bottom_left(engine_mod):
    """a1 (square 0) must map to row 7, col 0 - the bottom-left of the array."""
    planes = engine_mod.board_to_planes(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"))
    assert planes[7, 0, PIECE_CHANNEL["R"]] == 1.0


def test_orientation_h8_is_top_right(engine_mod):
    """h8 (square 63) must map to row 0, col 7."""
    planes = engine_mod.board_to_planes(chess.Board("4k2r/8/8/8/8/8/8/4K3 b k - 0 1"))
    assert planes[0, 7, PIECE_CHANNEL["R"] + BLACK_OFFSET] == 1.0


def test_orientation_is_consistent_across_all_squares(engine_mod):
    """Sweep a rook over every square and confirm the row/col mapping holds."""
    for square in chess.SQUARES:
        if square in (chess.E1, chess.E8):
            continue  # occupied by the kings
        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        board.set_piece_at(square, chess.Piece(chess.ROOK, chess.WHITE))
        planes = engine_mod.board_to_planes(board)
        row, col = 7 - (square // 8), square % 8
        assert planes[row, col, PIECE_CHANNEL["R"]] == 1.0, chess.square_name(square)


# ------------------------------------------------------------------ distinctness

def test_different_piece_placements_encode_differently(engine_mod):
    a = engine_mod.board_to_planes(chess.Board("4k3/8/8/8/4Q3/8/8/4K3 w - - 0 1"))
    b = engine_mod.board_to_planes(chess.Board("4k3/8/8/4Q3/8/8/8/4K3 w - - 0 1"))
    assert not np.array_equal(a, b)


def test_different_piece_types_encode_differently(engine_mod):
    a = engine_mod.board_to_planes(chess.Board("4k3/8/8/8/4Q3/8/8/4K3 w - - 0 1"))
    b = engine_mod.board_to_planes(chess.Board("4k3/8/8/8/4R3/8/8/4K3 w - - 0 1"))
    assert not np.array_equal(a, b)


def test_encoding_is_deterministic(engine_mod, start_board):
    first = engine_mod.board_to_planes(start_board)
    second = engine_mod.board_to_planes(start_board)
    assert np.array_equal(first, second)


def test_encoding_does_not_mutate_the_board(engine_mod, start_board):
    before = start_board.fen()
    engine_mod.board_to_planes(start_board)
    assert start_board.fen() == before


# ================================ C6: STILL DEFERRED AFTER C10 (open, evidenced)
#
# `board_to_planes` carries piece placement ONLY. Side to move, castling rights
# and en-passant state are absent. That is a real representational limitation: the
# engine evaluates post-move positions where it is the opponent's turn, and the
# CNN cannot tell those apart from the same placement with the other side to move.
#
# These three tests were written in Phase 2 as "DEFERRED (Phase 4)", i.e. a gap
# expected to be closed. C10 re-audited them and reclassified the deferral as
# EVIDENCE-BASED rather than pending, on two independent grounds:
#
#   1. ARCHITECTURALLY LOCKED. models/cnn_model.keras has input shape
#      (None, 8, 8, 12) and 2,360,129 parameters. A 13th plane cannot be added
#      without retraining the CNN, which C10's brief forbids and which no
#      correctness argument justifies on its own.
#
#   2. MEASURED AS HARMFUL. C6-A3 built exactly the encoding these three tests
#      demand - 18 planes, side-to-move + castling + en-passant, the layout the
#      C6 audit specified - and it was the worst arm in the programme:
#      Pearson 0.631-0.656 against A2's 0.733-0.745 (no overlap), and mean regret
#      +46.5 cp on `extended` across all three paired seeds against an A0
#      seed-noise band of 19.7 cp. C6-A13 (castling only, 16 planes) reproduced
#      most of it: +32.6 cp. See docs/C6_A3_REPORT.md and docs/C6_A13_REPORT.md.
#      C6-A3 also found an 18-plane model is not loadable by the shipped engine.
#
# So the gap is documented, not scheduled. The assertions below are UNCHANGED and
# strict=True is retained: if the production representation is ever extended,
# these XPASS and fail the suite, forcing this file to be revisited rather than
# letting a stale note linger. See docs/C10_FINAL_QA.md.
#
# tests/unit/test_training_representation18.py covers the 18-plane research
# encoder, which does distinguish all three - the limitation is production's, not
# the repository's.

_C6_DEFERRAL = (
    "OPEN, EVIDENCE-BASED DEFERRAL (C6; re-audited in C10): production "
    "board_to_planes is 12-plane and models/cnn_model.keras takes "
    "(None, 8, 8, 12), so this cannot be closed without retraining the CNN. "
    "C6-A3 built exactly this encoding (18 planes) and regressed the engine by "
    "+46.5 cp mean regret on `extended` across all three paired seeds against a "
    "19.7 cp noise band; C6-A13 (castling only) regressed +32.6 cp. Documented, "
    "not scheduled. See docs/C6_A3_REPORT.md, docs/C6_A13_REPORT.md, "
    "docs/C10_FINAL_QA.md. Specifically: "
)


def test_production_cnn_input_is_twelve_planes(engine_mod):
    """Pins ground 1 of the C6 deferral: the encoding is locked by the model.

    If the production CNN is ever replaced by one taking more planes, this fails
    and the three deferrals below must be re-decided rather than inherited.
    """
    assert engine_mod.cnn_model.input_shape == (None, 8, 8, 12)
    assert engine_mod.board_to_planes(chess.Board()).shape == (8, 8, 12)


@pytest.mark.deferred
@pytest.mark.xfail(strict=True, reason=_C6_DEFERRAL +
                   "board_to_planes has no side-to-move plane, so identical "
                   "placements with opposite sides to move are indistinguishable.")
def test_side_to_move_should_be_representable(engine_mod):
    white_to_move = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    black_to_move = chess.Board("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert not np.array_equal(
        engine_mod.board_to_planes(white_to_move),
        engine_mod.board_to_planes(black_to_move),
    )


@pytest.mark.deferred
@pytest.mark.xfail(strict=True, reason=_C6_DEFERRAL +
                   "castling rights are not encoded.")
def test_castling_rights_should_be_representable(engine_mod):
    with_rights = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    without_rights = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert not np.array_equal(
        engine_mod.board_to_planes(with_rights),
        engine_mod.board_to_planes(without_rights),
    )


@pytest.mark.deferred
@pytest.mark.xfail(strict=True, reason=_C6_DEFERRAL +
                   "en-passant state is not encoded.")
def test_en_passant_should_be_representable(engine_mod):
    with_ep = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    without_ep = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 2")
    assert not np.array_equal(
        engine_mod.board_to_planes(with_ep),
        engine_mod.board_to_planes(without_ep),
    )


# The counterpart characterisation tests: these three states ARE currently
# collapsed, which is the fact the deferrals above record.

@pytest.mark.parametrize(
    "fen_a,fen_b,what",
    [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1",
         "4k3/8/8/8/8/8/8/4K3 b - - 0 1", "side to move"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
         "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1", "castling rights"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
         "4k3/8/8/3pP3/8/8/8/4K3 w - - 0 2", "en-passant square"),
    ],
)
def test_production_encoding_collapses_these_states(engine_mod, fen_a, fen_b, what):
    """Characterisation: pins the current limitation as fact, so the suite never
    silently blesses it as intended and never silently loses it either."""
    a = engine_mod.board_to_planes(chess.Board(fen_a))
    b = engine_mod.board_to_planes(chess.Board(fen_b))
    assert np.array_equal(a, b), (
        f"{what} is now encoded - the C6 deferrals must be re-decided")
