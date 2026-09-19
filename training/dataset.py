"""Dataset loading, label-policy selection and the deterministic train/test split.

No TensorFlow import, so all of this is unit-testable in milliseconds.

--------------------------------------------------------------------------
LABEL POLICIES
--------------------------------------------------------------------------
`dataset_v1.jsonl` stores, per position, BOTH:

    raw_stockfish_value   Stockfish's native side-to-move-relative output
    eval_type             "cp" or "mate"
    label                 the C6-Prep repaired label (White-positive, mate mapped)

Because the raw pre-policy value is retained, an arm's label policy is a
*derivation*, not a stored choice. Each policy below is applied to the same raw
values, so the arms differ in exactly one respect.

    legacy_notebook   clip(raw_stm_value, +/-1500), with NO check of eval_type.
                      Reproduces the pre-C6 notebook policy exactly: mate scores
                      stay raw mate distances (mate-in-1 -> 1, checkmate -> 0)
                      and labels stay side-to-move relative.
                      >>> THIS IS THE A0 CONTROL <<<

    c6prep            The repaired policy from training/labels.py: mates mapped
                      onto the centipawn axis, labels White-positive, clipping
                      applied to cp only. This is the `label` field as stored.
                      (Reserved for later arms; A0 must not use it.)

--------------------------------------------------------------------------
WHAT A0's CONTROL IS, AND IS NOT
--------------------------------------------------------------------------
IS:      the pre-C6 label POLICY applied to dataset_v1's reproducible raw values.
IS NOT:  the historical labels used to train models/cnn_model.keras.

The historical labels are unrecoverable: they were produced with an unrecorded
Stockfish version, with a shared transposition table (so order-dependent), and
were never persisted. See docs/C6_TRAINING_PIPELINE_AUDIT.md. Nothing here
attempts to reconstruct them, and no synthetic labels are created.

--------------------------------------------------------------------------
SPLIT
--------------------------------------------------------------------------
Records are sorted by FEN (a canonical, order-independent key) and then split
80/20 with a FIXED split seed that is deliberately separate from the training
seed. The test set is therefore byte-identical across every seed and every arm,
which is what lets seed-to-seed spread be read as training variance rather than
data variance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CP_CLIP_LEGACY = 1500
DEFAULT_SPLIT_SEED = 42
DEFAULT_TEST_FRACTION = 0.2

LABEL_POLICY_LEGACY = "legacy_notebook"
LABEL_POLICY_C6PREP = "c6prep"
LABEL_POLICIES = (LABEL_POLICY_LEGACY, LABEL_POLICY_C6PREP)


# ============================================================ loading

def load_records(path: Path) -> list[dict]:
    """Read dataset_v1.jsonl. Order is as written (deterministic)."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def validate_records(records: list[dict]) -> dict:
    """Fail loudly on anything that would silently corrupt an arm."""
    required = {"fen", "label", "raw_stockfish_value", "eval_type", "side_to_move"}
    missing_fields = required - set(records[0]) if records else required
    if missing_fields:
        raise ValueError(f"dataset records are missing fields: {sorted(missing_fields)}")

    fens = [r["fen"] for r in records]
    if len(set(fens)) != len(fens):
        raise ValueError(
            f"dataset contains duplicate FENs ({len(fens) - len(set(fens))}); "
            f"it should already be deduplicated by build_dataset.py"
        )
    perspectives = {r.get("raw_value_perspective") for r in records}
    if perspectives != {"side_to_move"}:
        raise ValueError(
            f"expected every raw value to be side-to-move relative, got {perspectives}"
        )
    return {"n_records": len(records), "unique_fens": len(set(fens))}


# ============================================================ label policies

def legacy_notebook_label(raw_value: int, clip: int = CP_CLIP_LEGACY) -> int:
    """The pre-C6 policy: clip the raw side-to-move value, ignore eval_type.

    Verbatim equivalent of the notebook's

        val = int(np.clip(engine.get_evaluation()['value'], -1500, 1500))

    which never inspected ['type'], so a mate distance was stored as if it were
    a centipawn amount.
    """
    return int(np.clip(int(raw_value), -clip, clip))


def apply_label_policy(records: list[dict], policy: str) -> np.ndarray:
    """Derive the label vector for an arm. Never mutates `records`."""
    if policy == LABEL_POLICY_LEGACY:
        return np.array([legacy_notebook_label(r["raw_stockfish_value"])
                         for r in records], dtype=np.float32)
    if policy == LABEL_POLICY_C6PREP:
        return np.array([r["label"] for r in records], dtype=np.float32)
    raise ValueError(f"unknown label policy {policy!r}; expected one of {LABEL_POLICIES}")


def label_policy_summary(policy: str) -> dict:
    if policy == LABEL_POLICY_LEGACY:
        return {
            "name": LABEL_POLICY_LEGACY,
            "perspective": "side_to_move (NOT normalised to White)",
            "mate_handling": "none - raw mate distance stored as if centipawns "
                             "(mate-in-1 -> 1, checkmate -> 0)",
            "clip": [-CP_CLIP_LEGACY, CP_CLIP_LEGACY],
            "clip_applies_to": "all labels",
            "derivation": "clip(raw_stockfish_value, +/-1500); eval_type ignored",
            "is_control_arm": True,
            "reproduces_historical_labels": False,
            "note": "the pre-C6 POLICY applied to dataset_v1's raw values. The "
                    "historical labels themselves are unrecoverable (unknown "
                    "Stockfish version, order-dependent, never persisted).",
        }
    if policy == LABEL_POLICY_C6PREP:
        return {
            "name": LABEL_POLICY_C6PREP,
            "perspective": "white",
            "mate_handling": "mapped onto the centipawn axis (see training/labels.py)",
            "clip": [-CP_CLIP_LEGACY, CP_CLIP_LEGACY],
            "clip_applies_to": "cp labels only",
            "derivation": "the stored `label` field",
            "is_control_arm": False,
            "reproduces_historical_labels": False,
        }
    raise ValueError(f"unknown label policy {policy!r}")


# ============================================================ split

@dataclass
class Split:
    train_index: np.ndarray
    test_index: np.ndarray
    split_seed: int
    test_fraction: float
    order_key: str

    def summary(self) -> dict:
        return {
            "method": (
                f"records sorted by {self.order_key}, then a seeded permutation; "
                f"the FIRST {1 - self.test_fraction:.0%} are train"
            ),
            "split_seed": self.split_seed,
            "test_fraction": self.test_fraction,
            "order_key": self.order_key,
            "n_train": int(len(self.train_index)),
            "n_test": int(len(self.test_index)),
            "independent_of_training_seed": True,
        }


def make_split(records: list[dict], split_seed: int = DEFAULT_SPLIT_SEED,
               test_fraction: float = DEFAULT_TEST_FRACTION) -> Split:
    """Deterministic 80/20 split, identical for every training seed.

    Sorting by FEN first makes the split independent of the order the records
    happen to sit in the file, so it survives a dataset regeneration that
    preserves content but changes ordering.
    """
    order = np.argsort(np.array([r["fen"] for r in records], dtype=object), kind="stable")
    rng = np.random.default_rng(split_seed)
    permuted = order[rng.permutation(len(order))]

    n_test = int(round(len(records) * test_fraction))
    test_index = np.sort(permuted[:n_test])
    train_index = np.sort(permuted[n_test:])
    return Split(train_index, test_index, split_seed, test_fraction, "fen")


# ============================================================ assembled arm data

@dataclass
class ArmData:
    records: list[dict]
    labels: np.ndarray
    split: Split
    label_policy: str

    @property
    def train_records(self):
        return [self.records[i] for i in self.split.train_index]

    @property
    def test_records(self):
        return [self.records[i] for i in self.split.test_index]

    @property
    def y_train(self) -> np.ndarray:
        return self.labels[self.split.train_index]

    @property
    def y_test(self) -> np.ndarray:
        return self.labels[self.split.test_index]

    def label_stats(self) -> dict:
        def describe(v):
            return {
                "n": int(len(v)),
                "min": float(np.min(v)), "max": float(np.max(v)),
                "mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v)), 4),
            }
        return {"all": describe(self.labels),
                "train": describe(self.y_train),
                "test": describe(self.y_test)}


def build_arm_data(records: list[dict], label_policy: str,
                   split_seed: int = DEFAULT_SPLIT_SEED,
                   test_fraction: float = DEFAULT_TEST_FRACTION) -> ArmData:
    labels = apply_label_policy(records, label_policy)
    split = make_split(records, split_seed, test_fraction)
    return ArmData(records=records, labels=labels, split=split, label_policy=label_policy)
