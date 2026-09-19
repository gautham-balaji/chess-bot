"""C6-A2 pre-training label audit: exactly what changes between A1 and A2.

    python -m training.a2_label_audit

Read-only. Writes training/experiments/A2/label_audit.json and prints the tables.
Run and inspect this BEFORE training: if anything other than Black-to-move
records moves, the perspective change is not isolated and A2 must not be trained.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from training import dataset as D  # noqa: E402

DATASET = REPO_ROOT / "training" / "artifacts" / "dataset_v1.jsonl"
OUT = REPO_ROOT / "training" / "experiments" / "A2" / "label_audit.json"


def describe(v) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {"n": int(v.size), "min": float(v.min()), "max": float(v.max()),
            "mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
            "median": round(float(np.median(v)), 4)}


def main() -> int:
    if not DATASET.is_file():
        raise SystemExit(f"ERROR: dataset not found: {DATASET}")
    records = D.load_records(DATASET)
    D.validate_records(records)

    a0 = D.apply_label_policy(records, D.LABEL_POLICY_LEGACY)
    a1 = D.apply_label_policy(records, D.LABEL_POLICY_CORRECTED_MATE)
    a2 = D.apply_label_policy(records, D.LABEL_POLICY_CORRECTED_MATE_WHITE)
    stored = D.apply_label_policy(records, D.LABEL_POLICY_C6PREP)

    stm = np.array([r["side_to_move"] for r in records])
    et = np.array([r["eval_type"] for r in records])
    changed = a1 != a2
    is_black = stm == "black"

    # ---------------------------------------------------------------- gates
    gates = {
        "derived_a2_equals_stored_label_field": bool(np.array_equal(a2, stored)),
        "every_change_is_a_black_to_move_record": bool(
            not (changed & ~is_black).any()),
        "every_black_to_move_record_changed_or_is_zero": bool(
            ((changed | (a1 == 0)) | ~is_black).all()),
        "every_change_is_an_exact_sign_flip": bool(
            np.array_equal(a2[changed], -a1[changed])),
        "no_white_to_move_label_moved": bool(np.array_equal(a1[~is_black], a2[~is_black])),
        "magnitudes_identical_everywhere": bool(np.array_equal(np.abs(a1), np.abs(a2))),
        "a2_differs_from_a0": bool(not np.array_equal(a0, a2)),
        "a2_differs_from_a1": bool(not np.array_equal(a1, a2)),
    }

    split = D.make_split(records)
    in_test = np.zeros(len(records), dtype=bool)
    in_test[split.test_index] = True

    audit = {
        "stage": "C6-A2 pre-training label audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(DATASET.relative_to(REPO_ROOT).as_posix()),
        "n_records": len(records),
        "policies": {
            "A0": D.label_policy_summary(D.LABEL_POLICY_LEGACY)["name"],
            "A1": D.label_policy_summary(D.LABEL_POLICY_CORRECTED_MATE)["name"],
            "A2": D.label_policy_summary(D.LABEL_POLICY_CORRECTED_MATE_WHITE)["name"],
        },
        "a2_policy_summary": D.label_policy_summary(D.LABEL_POLICY_CORRECTED_MATE_WHITE),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "composition": {
            "by_side_to_move": {k: int(v) for k, v in Counter(stm).items()},
            "by_eval_type": {k: int(v) for k, v in Counter(et).items()},
            "by_side_and_type": {f"{a}/{b}": int(v)
                                 for (a, b), v in Counter(zip(stm, et)).items()},
            "black_to_move_share_pct": round(100 * float(is_black.mean()), 4),
        },
        "a1_to_a2": {
            "n_changed": int(changed.sum()),
            "n_unchanged": int((~changed).sum()),
            "pct_changed": round(100 * float(changed.mean()), 4),
            "changed_by_side_to_move": {k: int(v) for k, v in Counter(stm[changed]).items()},
            "changed_by_eval_type": {k: int(v) for k, v in Counter(et[changed]).items()},
            "changed_in_train": int((changed & ~in_test).sum()),
            "changed_in_test": int((changed & in_test).sum()),
            "max_abs_delta": float(np.abs(a2 - a1).max()),
            "mean_abs_delta_over_changed": round(
                float(np.abs(a2 - a1)[changed].mean()), 4),
        },
        "a0_to_a2": {
            "n_changed": int((a0 != a2).sum()),
            "note": "A2 changes BOTH mate scale and perspective relative to A0; "
                    "the controlled single-variable contrast is A1 -> A2",
        },
        "label_distributions": {
            "A0": describe(a0), "A1": describe(a1), "A2": describe(a2),
            "A1_white_records": describe(a1[~is_black]),
            "A2_white_records": describe(a2[~is_black]),
            "A1_black_records": describe(a1[is_black]),
            "A2_black_records": describe(a2[is_black]),
        },
        "split": split.summary(),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- print
    print("=" * 78)
    print("C6-A2 PRE-TRAINING LABEL AUDIT")
    print("=" * 78)
    c = audit["composition"]
    print(f"\nrecords {len(records)}   "
          f"white {c['by_side_to_move'].get('white')}  "
          f"black {c['by_side_to_move'].get('black')} "
          f"({c['black_to_move_share_pct']}%)")
    print(f"  by side/type: {c['by_side_and_type']}")

    d = audit["a1_to_a2"]
    print(f"\nA1 -> A2 changes: {d['n_changed']} / {len(records)} "
          f"({d['pct_changed']}%)   unchanged {d['n_unchanged']}")
    print(f"  by side_to_move : {d['changed_by_side_to_move']}")
    print(f"  by eval_type    : {d['changed_by_eval_type']}")
    print(f"  train / test    : {d['changed_in_train']} / {d['changed_in_test']}")
    print(f"  max |delta|     : {d['max_abs_delta']:.0f}")

    print("\ndistributions:")
    for k in ("A0", "A1", "A2", "A1_black_records", "A2_black_records"):
        s = audit["label_distributions"][k]
        print(f"  {k:20s} n={s['n']:5d}  [{s['min']:8.0f} .. {s['max']:8.0f}]  "
              f"mean {s['mean']:9.3f}  std {s['std']:8.3f}")

    print("\ngates:")
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    verdict = "ALL GATES PASS - safe to train A2" if audit["all_gates_pass"] \
        else "GATE FAILURE - DO NOT TRAIN A2"
    print(f"\n{verdict}")
    print(f"\nWROTE {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0 if audit["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
