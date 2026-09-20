"""Tests for C6-A14: post-convolution castling injection.

A14's whole claim is about WHERE castling enters the network, so these tests are
written against the BUILT KERAS GRAPH, not against the encoder's docstring:

  * the convolutional stack's input is (None, 8, 8, 12) and equals A2's planes
  * no castling value reaches any Conv2D
  * the four castling scalars are concatenated after Flatten
  * the Conv/BN layers are configured exactly as A2's
  * the parameter difference against A2 is exactly +1,024 (Dense(256)'s fan-in)

Plus the usual controlled-experiment invariants: labels, split, arm registration,
and that production stays on the 12-plane path.

The section numbering matches the fourteen verification points the A14 design was
required to establish; see docs/C6_A14_REPORT.md section 6.
"""
import inspect
import json
from pathlib import Path

import chess
import numpy as np
import pytest

from training import dataset as D
from training import representation as R12
from training import representation12c4 as RC4
from training import representations as REPS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl"

START = chess.STARTING_FEN
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
NO_CASTLING = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
ALL_RIGHTS = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
NO_RIGHTS_SAME_PIECES = "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"
# Same pieces as ALL_RIGHTS, White kingside right removed.
WQ_ONLY = "r3k2r/8/8/8/8/8/8/R3K2R w Qkq - 0 1"

A2_PARAMS = 2_360_129          # pinned by test_training_train_harness.py
A14_PARAMS = 2_361_153         # A2 + 4 * 256


def _suite_fens():
    fens = []
    for suite in ("extended", "phase0_52"):
        path = REPO_ROOT / "evaluation" / "positions" / f"{suite}.json"
        fens += [p["fen"] for p in json.loads(path.read_text(encoding="utf-8"))["positions"]]
    return fens


@pytest.fixture(scope="module")
def a14_model():
    from training import train as T
    return T.build_model(RC4, T.ARCH_POSTCONV_CASTLING)


@pytest.fixture(scope="module")
def a2_model():
    from training import train as T
    return T.build_model(R12)


@pytest.fixture(scope="module")
def conv_layers(a14_model):
    from keras import layers
    return [l for l in a14_model.layers if isinstance(l, layers.Conv2D)]


# ================================================ 1. same 12 piece planes as A2

def test_first_twelve_channels_match_the_twelve_plane_encoder_on_both_suites():
    """THE drift guard: A14's convolutional input IS A2's, position by position."""
    fens = _suite_fens()
    assert len(fens) > 200
    for fen in fens:
        board = chess.Board(fen)
        assert np.array_equal(RC4.board_to_planes(board)[:, :, :12],
                              R12.board_to_planes(board)), fen


def test_conv_planes_helper_is_exactly_the_a2_encoder():
    for fen in (START, AFTER_E4, NO_CASTLING, ALL_RIGHTS, WQ_ONLY):
        board = chess.Board(fen)
        assert np.array_equal(RC4.conv_planes(board), R12.board_to_planes(board))


def test_first_twelve_channels_match_the_production_engine_encoder():
    import engine as E
    for fen in _suite_fens()[:40]:
        board = chess.Board(fen)
        assert np.array_equal(RC4.board_to_planes(board)[:, :, :12],
                              E.board_to_planes(board)), fen


# ================================================ 2. conv input shape is (8,8,12)

def test_declared_conv_input_shape_is_twelve_planes():
    assert RC4.CONV_PLANES == 12
    assert RC4.CONV_INPUT_SHAPE == (8, 8, 12) == R12.BOARD_SHAPE


def test_first_conv_layer_receives_eight_by_eight_by_twelve(conv_layers):
    assert tuple(conv_layers[0].input.shape) == (None, 8, 8, 12)


def test_conv_stack_shapes_are_identical_to_a2(conv_layers, a2_model):
    from keras import layers
    a2 = [l for l in a2_model.layers if isinstance(l, layers.Conv2D)]
    assert len(conv_layers) == len(a2) == 3
    for mine, theirs in zip(conv_layers, a2):
        assert tuple(mine.input.shape) == tuple(theirs.input.shape)
        assert tuple(mine.output.shape) == tuple(theirs.output.shape)


# ================================================ 3. castling never enters conv

def test_no_convolution_sees_more_than_twelve_channels(conv_layers):
    """If castling leaked into the conv path, some Conv2D's input channel count
    would have to grow. None does."""
    assert [int(l.input.shape[-1]) for l in conv_layers] == [12, 64, 128]


def test_first_conv_kernel_has_a2s_exact_parameter_count(conv_layers):
    # 3*3*12*64 + 64 = 6,976. A13's first conv is 3*3*16*64 + 64 = 9,280.
    assert conv_layers[0].count_params() == 3 * 3 * 12 * 64 + 64 == 6_976


def test_changing_only_castling_leaves_the_conv_channels_bit_identical():
    a = RC4.fen_to_planes(ALL_RIGHTS)
    b = RC4.fen_to_planes(NO_RIGHTS_SAME_PIECES)
    assert np.array_equal(a[:, :, :12], b[:, :, :12])
    assert not np.array_equal(a[:, :, 12:], b[:, :, 12:])


def test_conv_features_are_unaffected_by_the_castling_channels(a14_model):
    """End to end: run the real graph up to the Concatenate on two tensors that
    differ ONLY in channels 12-15. The convolutional features must be identical."""
    import keras
    trunk = keras.models.Model(a14_model.inputs,
                               a14_model.get_layer("inject_castling").input[0])
    x = RC4.encode_many([ALL_RIGHTS])
    y = RC4.encode_many([NO_RIGHTS_SAME_PIECES])
    assert not np.array_equal(x[:, :, :, 12:], y[:, :, :, 12:])
    assert np.array_equal(trunk.predict(x, verbose=0), trunk.predict(y, verbose=0))


# ================================================ 4 & 6. the four scalars

def test_there_are_exactly_four_castling_features():
    assert RC4.N_CASTLING_FEATURES == 4
    assert len(RC4.CASTLING_FEATURE_ORDER) == 4
    assert len(RC4.CASTLING_FEATURE_NAMES) == 4
    assert RC4.castling_scalars(chess.Board(START)).shape == (4,)


@pytest.mark.parametrize("fen,expected", [
    (START, (1, 1, 1, 1)),
    (ALL_RIGHTS, (1, 1, 1, 1)),
    (NO_CASTLING, (0, 0, 0, 0)),
    (NO_RIGHTS_SAME_PIECES, (0, 0, 0, 0)),
    ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", (1, 0, 0, 0)),
    ("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", (0, 1, 0, 0)),
    ("4k2r/8/8/8/8/8/8/4K3 b k - 0 1", (0, 0, 1, 0)),
    ("r3k3/8/8/8/8/8/8/4K3 b q - 0 1", (0, 0, 0, 1)),
    (WQ_ONLY, (0, 1, 1, 1)),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1", (1, 1, 1, 0)),
])
def test_castling_scalars_are_correct(fen, expected):
    assert tuple(int(v) for v in RC4.fen_to_castling_scalars(fen)) == expected


def test_scalars_agree_with_python_chess_on_every_suite_position():
    for fen in _suite_fens():
        board = chess.Board(fen)
        got = RC4.castling_scalars(board)
        want = (board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK))
        assert tuple(bool(v) for v in got) == want, fen


def test_scalars_are_binary_float32():
    v = RC4.fen_to_castling_scalars(WQ_ONLY)
    assert v.dtype == np.float32
    assert set(np.unique(v)).issubset({0.0, 1.0})


def test_transport_channels_carry_the_scalars_unchanged():
    for fen in (START, NO_CASTLING, WQ_ONLY, ALL_RIGHTS):
        planes = RC4.fen_to_planes(fen)
        scalars = RC4.fen_to_castling_scalars(fen)
        for i, ch in enumerate(RC4.CASTLING_CHANNELS):
            assert np.all(planes[:, :, ch] == scalars[i]), (fen, ch)


def test_transport_channels_are_spatially_constant_everywhere():
    """The model reads the scalars from cell (0, 0), which is exact only if the
    broadcast is constant. Checked over both suites AND the training dataset."""
    fens = _suite_fens()
    if DATASET.is_file():
        fens += [r["fen"] for r in D.load_records(DATASET)]
    assert len(fens) > 200
    for fen in fens:
        planes = RC4.fen_to_planes(fen)
        for ch in RC4.CASTLING_CHANNELS:
            assert len(np.unique(planes[:, :, ch])) == 1, (fen, ch)


# ================================================ 5. deterministic documented order

def test_feature_order_is_the_documented_one():
    assert RC4.CASTLING_FEATURE_NAMES == (
        "white_kingside_castling", "white_queenside_castling",
        "black_kingside_castling", "black_queenside_castling")
    assert RC4.CASTLING_FEATURE_ORDER == (
        (chess.WHITE, "kingside"), (chess.WHITE, "queenside"),
        (chess.BLACK, "kingside"), (chess.BLACK, "queenside"))
    assert RC4.CASTLING_CHANNELS == (12, 13, 14, 15)


def test_feature_order_matches_a13_so_the_arms_stay_comparable():
    from training import representation16 as R16
    assert [R16.CASTLING_PLANES[k] - 12 for k in RC4.CASTLING_FEATURE_ORDER] == [0, 1, 2, 3]


def test_order_is_stable_across_calls():
    board = chess.Board(WQ_ONLY)
    first = RC4.castling_scalars(board)
    for _ in range(5):
        assert np.array_equal(RC4.castling_scalars(board), first)


def test_plane_names_cover_every_channel_and_name_the_scalars():
    assert len(RC4.PLANE_NAMES) == 16
    assert RC4.PLANE_NAMES[:12] == R12.PLANE_NAMES
    assert RC4.PLANE_NAMES[12:] == [f"{n}_scalar" for n in RC4.CASTLING_FEATURE_NAMES]


# ================================================ 7. concatenation is after Flatten

def test_concatenate_consumes_the_flatten_output_and_the_scalars(a14_model):
    from keras import layers
    concat = a14_model.get_layer("inject_castling")
    assert isinstance(concat, layers.Concatenate)
    assert [tuple(t.shape) for t in concat.input] == [(None, 8192), (None, 4)]
    assert tuple(concat.output.shape) == (None, 8196)


def test_flatten_feeds_the_concatenate_directly(a14_model):
    from keras import layers
    flatten = next(l for l in a14_model.layers if isinstance(l, layers.Flatten))
    assert tuple(flatten.output.shape) == (None, 8192)
    assert (tuple(a14_model.get_layer("inject_castling").input[0].shape)
            == tuple(flatten.output.shape))


def test_no_dense_layer_precedes_the_concatenation(a14_model):
    order = [l.__class__.__name__ for l in a14_model.layers]
    assert order.index("Concatenate") < order.index("Dense")


def test_layer_order_is_conv_stack_then_flatten_then_concat_then_a2_head(a14_model):
    """A14's layer list is A2's list with exactly ONE layer inserted, the
    Concatenate, and it sits between Flatten and the first Dense. The two channel
    slices are graph ops with no weights, so they are not layers."""
    assert [l.__class__.__name__ for l in a14_model.layers] == [
        "InputLayer",
        "Conv2D", "BatchNormalization",
        "Conv2D", "BatchNormalization",
        "Conv2D", "BatchNormalization",
        "Flatten", "Concatenate",
        "Dense", "Dropout", "Dense", "Dropout", "Dense",
    ]


def test_a14_inserts_exactly_one_layer_into_a2s_sequence(a14_model, a2_model):
    a14 = [l.__class__.__name__ for l in a14_model.layers if l.__class__.__name__
           != "InputLayer"]
    a2 = [l.__class__.__name__ for l in a2_model.layers]
    assert [n for n in a14 if n != "Concatenate"] == a2
    assert a14.count("Concatenate") == 1


# ================================================ 8. output shape unchanged

def test_output_shape_is_unchanged(a14_model, a2_model):
    assert a14_model.output_shape == (None, 1) == a2_model.output_shape


def test_model_predicts_one_finite_scalar_per_position(a14_model):
    preds = a14_model.predict(RC4.encode_many([START, NO_CASTLING, ALL_RIGHTS]),
                              verbose=0)
    assert preds.shape == (3, 1)
    assert np.isfinite(preds).all()


def test_model_accepts_the_engines_single_position_call_shape(a14_model):
    """engine.cnn_evaluate does np.expand_dims(board_to_planes(b), axis=0)."""
    one = np.expand_dims(RC4.fen_to_planes(START), axis=0)
    assert one.shape == (1, 8, 8, 16)
    assert a14_model.predict(one, verbose=0).shape == (1, 1)


# ================================================ 9. labels and split match A2

def test_a14_uses_exactly_the_same_label_policy_as_a2():
    from training import train as T
    assert (T.ARMS["A14"]["label_policy"] == T.ARMS["A2"]["label_policy"]
            == D.LABEL_POLICY_CORRECTED_MATE_WHITE)


def test_a14_labels_are_identical_to_a2_on_the_real_dataset():
    from training import train as T
    if not DATASET.is_file():
        pytest.skip("dataset_v1.jsonl not generated in this checkout")
    records = D.load_records(DATASET)
    y2 = D.apply_label_policy(records, T.ARMS["A2"]["label_policy"])
    y14 = D.apply_label_policy(records, T.ARMS["A14"]["label_policy"])
    assert np.array_equal(y2, y14)


def test_a14_split_is_identical_to_a2_and_a13():
    from training import train as T
    records = [{"fen": f"fen{i:04d}", "raw_stockfish_value": i - 50,
                "eval_type": "cp", "label": i - 50, "side_to_move": "white",
                "raw_value_perspective": "side_to_move", "is_checkmate": False}
               for i in range(400)]
    splits = [D.build_arm_data(records, T.ARMS[a]["label_policy"]).split
              for a in ("A2", "A13", "A14")]
    for s in splits[1:]:
        assert np.array_equal(s.train_index, splits[0].train_index)
        assert np.array_equal(s.test_index, splits[0].test_index)
        assert s.split_seed == 42


def test_a14_differs_from_a2_in_representation_and_architecture_only():
    from training import train as T
    a, b = T.ARMS["A2"], T.ARMS["A14"]
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - {"description"}
    assert differing == {"representation", "architecture"}


def test_a14_differs_from_a13_in_representation_and_architecture_only():
    from training import train as T
    a, b = T.ARMS["A13"], T.ARMS["A14"]
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - {"description"}
    assert differing == {"representation", "architecture"}


def test_a14_uses_the_same_hyperparameters_as_every_other_arm():
    from training import train as T
    hp = T.HP
    assert hp["loss"] == "huber" and hp["optimizer"] == "adam"
    assert hp["learning_rate"] == 1e-3 and hp["batch_size"] == 64
    assert hp["max_epochs"] == 100 and hp["early_stopping_patience"] == 10
    assert hp["restore_best_weights"] is True
    assert hp["reduce_lr_patience"] == 5 and hp["validation_split"] == 0.1
    assert hp["dense_units"] == [256, 128] and hp["dropout_rates"] == [0.3, 0.2]
    assert set(T.ARMS["A14"]) <= {"label_policy", "representation", "architecture",
                                  "description"}


# ================================================ 10. Conv/BN stack unchanged

def test_conv_layer_configs_are_identical_to_a2(conv_layers, a2_model):
    from keras import layers
    a2 = [l for l in a2_model.layers if isinstance(l, layers.Conv2D)]
    for mine, theirs in zip(conv_layers, a2):
        for key in ("filters", "kernel_size", "strides", "padding", "activation",
                    "use_bias", "dilation_rate", "groups"):
            assert mine.get_config()[key] == theirs.get_config()[key], key


def test_batchnorm_configs_and_placement_are_identical_to_a2(a14_model, a2_model):
    from keras import layers
    mine = [l for l in a14_model.layers if isinstance(l, layers.BatchNormalization)]
    a2 = [l for l in a2_model.layers if isinstance(l, layers.BatchNormalization)]
    assert len(mine) == len(a2) == 3
    for m, t in zip(mine, a2):
        assert tuple(m.input.shape) == tuple(t.input.shape)
        for key in ("axis", "momentum", "epsilon", "center", "scale"):
            assert m.get_config()[key] == t.get_config()[key], key


def test_conv_and_bn_parameter_totals_are_identical_to_a2(a14_model, a2_model):
    from keras import layers

    def conv_bn_params(model):
        return sum(l.count_params() for l in model.layers
                   if isinstance(l, (layers.Conv2D, layers.BatchNormalization)))
    assert conv_bn_params(a14_model) == conv_bn_params(a2_model) == 229_696


def test_dense_head_widths_are_unchanged(a14_model):
    from keras import layers
    dense = [l for l in a14_model.layers if isinstance(l, layers.Dense)]
    assert [l.units for l in dense] == [256, 128, 1]
    dropout = [l for l in a14_model.layers if isinstance(l, layers.Dropout)]
    assert [l.rate for l in dropout] == [0.3, 0.2]


def test_the_sequential_builder_is_untouched_for_every_other_arm(a2_model):
    """A0 through A13R must still rebuild the exact net they were trained with."""
    from training import train as T
    assert a2_model.count_params() == A2_PARAMS
    assert a2_model.input_shape == (None, 8, 8, 12)
    assert [l.__class__.__name__ for l in a2_model.layers] == [
        "Conv2D", "BatchNormalization", "Conv2D", "BatchNormalization",
        "Conv2D", "BatchNormalization", "Flatten",
        "Dense", "Dropout", "Dense", "Dropout", "Dense"]
    for arm in ("A0", "A1", "A2", "A3", "A13", "A13P", "A13R"):
        assert T.ARMS[arm].get("architecture", T.ARCH_SEQUENTIAL) == T.ARCH_SEQUENTIAL


def test_unknown_architecture_is_rejected():
    from training import train as T
    with pytest.raises(ValueError, match="unknown architecture"):
        T.build_model(R12, "resnet")


# ================================================ 13. parameter count

def test_a14_parameter_count_is_exactly_2_361_153(a14_model):
    assert a14_model.count_params() == A14_PARAMS


def test_a14_adds_exactly_1024_parameters_over_a2(a14_model, a2_model):
    """The ONLY new parameters are Dense(256)'s four extra input columns."""
    assert a2_model.count_params() == A2_PARAMS
    assert a14_model.count_params() - a2_model.count_params() == 4 * 256 == 1_024


def test_a14_adds_fewer_parameters_than_a13(a14_model, a2_model):
    """A13 pays 3*3*4*64 = 2,304 in the first conv; A14 pays 1,024 in Dense."""
    from training import representation16 as R16
    from training import train as T
    a2 = a2_model.count_params()
    assert T.build_model(R16).count_params() - a2 == 3 * 3 * 4 * 64 == 2_304
    assert a14_model.count_params() - a2 == 1_024


# ================================================ 14. no A13/A13R code is used

def test_the_encoder_does_not_reference_any_other_representation_module():
    body = Path(RC4.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
    for banned in ("representation16", "representation18", "representations",
                   "hashlib", "sha256", "SALT", "ep_square", "board.turn"):
        assert banned not in body, banned


def test_the_encoder_imports_only_the_twelve_plane_module():
    imported = {v.__name__ for v in vars(RC4).values()
                if getattr(v, "__name__", "").startswith("training.representation")}
    assert imported == {"training.representation"}


def test_a14_resolves_to_its_own_encoder_not_a13s():
    from training import representation16 as R16
    from training import representation16p as R16P
    from training import representation16r as R16R
    from training import train as T
    assert T.ARMS["A14"]["representation"] == "planes12c4"
    assert REPS.get("planes12c4") is RC4
    for other in (R16, R16P, R16R):
        assert RC4 is not other
        assert REPS.get("planes12c4") is not other


def test_a14_transport_equals_a13_input_but_the_arms_stay_distinct():
    """The two arms receive THE SAME tensor; only the wiring differs. That is what
    makes the A13 vs A14 contrast exact - and it is asserted here rather than
    achieved by sharing code, so neither module can drift into the other."""
    from training import representation16 as R16
    from training import train as T
    for fen in _suite_fens()[:60]:
        board = chess.Board(fen)
        assert np.array_equal(RC4.board_to_planes(board),
                              R16.board_to_planes(board)), fen
    assert T.ARMS["A13"]["representation"] != T.ARMS["A14"]["representation"]
    assert (T.ARMS["A13"].get("architecture", T.ARCH_SEQUENTIAL)
            != T.ARMS["A14"]["architecture"])


def test_a14_carries_no_hash_noise_no_side_to_move_and_no_en_passant():
    s = RC4.representation_summary()
    assert s["encodes_side_to_move"] is False
    assert s["encodes_en_passant"] is False
    assert s["encodes_castling_rights"] is True
    # Two boards differing ONLY in side to move must encode identically.
    a = RC4.fen_to_planes("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    b = RC4.fen_to_planes("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert np.array_equal(a, b)
    # ...and likewise for two boards differing only in the en-passant square.
    c = RC4.fen_to_planes("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    d = RC4.fen_to_planes("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    assert np.array_equal(c, d)


def test_encoding_is_deterministic_and_position_only():
    for fen in (START, WQ_ONLY, NO_CASTLING):
        first = RC4.fen_to_planes(fen)
        for _ in range(3):
            assert np.array_equal(RC4.fen_to_planes(fen), first)


# ================================================ registry / tooling

def test_registry_resolves_planes12c4():
    assert REPS.get("planes12c4") is RC4
    assert {"planes12", "planes16", "planes16p", "planes16r", "planes18",
            "planes12c4"} <= set(REPS.NAMES)


def test_every_registered_encoder_exposes_the_same_surface():
    for name in REPS.NAMES:
        mod = REPS.get(name)
        for attr in ("BOARD_SHAPE", "N_PLANES", "PLANE_NAMES", "board_to_planes",
                     "fen_to_planes", "encode_many", "representation_summary"):
            assert hasattr(mod, attr), f"{name} is missing {attr}"


def test_encode_many_stacks_in_order():
    fens = [START, ALL_RIGHTS, NO_CASTLING]
    out = RC4.encode_many(fens)
    assert out.shape == (3, 8, 8, 16)
    for i, fen in enumerate(fens):
        assert np.array_equal(out[i], RC4.fen_to_planes(fen))


def test_summary_declares_where_castling_enters():
    s = RC4.representation_summary()
    assert s["name"] == "planes12c4"
    assert s["conv_input_shape"] == [8, 8, 12]
    assert s["castling_channels"] == [12, 13, 14, 15]
    assert s["castling_enters_convolution"] is False
    assert s["castling_enters_dense_head"] is True
    assert s["planes_0_to_11_identical_to_planes12"] is True
    assert s["loadable_by_unmodified_engine"] is False


def test_ablation_tooling_knows_about_planes12c4():
    from training import a3_plane_ablation as AB
    assert AB.GROUPS_BY_REPRESENTATION["planes12c4"]["castling_scalars"] == [12, 13, 14, 15]


def test_a14_model_round_trips_through_the_engines_loader(tmp_path, a14_model):
    """engine.py calls load_model(path, compile=False) with safe_mode ON. A graph
    that cannot be deserialised that way is not evaluable at all."""
    import keras
    path = tmp_path / "cnn_model.keras"
    a14_model.save(path)
    reloaded = keras.saving.load_model(path, compile=False)
    assert reloaded.input_shape == (None, 8, 8, 16)
    x = RC4.encode_many([START, ALL_RIGHTS, NO_CASTLING])
    assert np.allclose(a14_model.predict(x, verbose=0),
                       reloaded.predict(x, verbose=0))


# ================================================ 11 & 12. production untouched

def test_production_encoder_is_still_twelve_planes_and_unchanged():
    import engine as E
    for fen in (START, AFTER_E4, ALL_RIGHTS, NO_CASTLING, WQ_ONLY):
        board = chess.Board(fen)
        assert E.board_to_planes(board).shape == (8, 8, 12)
        assert np.array_equal(E.board_to_planes(board), R12.board_to_planes(board))


def test_the_default_shipped_path_is_still_the_twelve_plane_direct_evaluator():
    from training import evaluate_arm as EA
    source = inspect.getsource(EA.run_suite)
    assert "N_PLANES == 12" in source
    assert "training.evaluate_planes_runner" in source
    assert inspect.signature(EA.run_suite).parameters["representation"].default == "planes12"


def test_a14_is_routed_through_the_experiment_only_shim():
    """planes12c4's transport tensor is 16 channels wide, so evaluate_arm must not
    send it down the direct production evaluator path."""
    assert RC4.N_PLANES != 12


def test_no_production_file_mentions_the_a14_arm():
    for name in ("engine.py", "app.py", "config.py"):
        source = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "12c4" not in source, name
        assert "A14" not in source, name
        assert "Concatenate" not in source, name
