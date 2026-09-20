"""C6-A2 exploratory probe: why does A2 lose accuracy on White-to-move positions?

    python -m training.a2_saturation_probe

Read-only, no Stockfish, no training, no production write. Writes
training/experiments/A2/saturation_probe.json.

--------------------------------------------------------------------------
STATUS: EXPLORATORY
--------------------------------------------------------------------------
This was NOT pre-registered. A2's pre-registered result is the paired A1 -> A2
contrast in a2_analysis.py. This probe only tries to explain one observation
from it - that A2 improves Black-to-move regret and worsens White-to-move
regret - and its findings are hypothesis-generating, not confirmatory.

--------------------------------------------------------------------------
THE HYPOTHESIS
--------------------------------------------------------------------------
The fusion applies the CNN through a saturating squash:

    score = w0 * tanh(cnn / 200) + w1*mat + w2*space + w3*center + w4*mob

A2's labels put real mass at +/-2000 in the correct frame, so if A2's CNN emits
larger magnitudes than A1's, tanh(cnn/200) flattens and the CNN term stops
separating candidate moves - the ranking then falls back on the heuristics.

The same hypothesis was tested for A1 in an earlier phase and was NOT supported.
It is re-tested here because A2 changes the label distribution differently.

What is measured, per arm/seed, over the candidate positions of both suites:
  - the spread of raw CNN output
  - the spread of tanh(cnn/200), which is what the fusion actually consumes
  - the share of candidates in tanh's saturated region (|tanh| > 0.95)
  - the WITHIN-POSITION spread of the CNN term, which is what ranking depends on
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

EXP = REPO_ROOT / "training" / "experiments"
OUT = EXP / "A2" / "saturation_probe.json"
SEEDS = (0, 1, 2)
ARMS = ("A0", "A1", "A2")
CONTRAST = ("A1", "A2")   # (src, dst) for the verdict
SUITES = ("extended", "phase0_52")
TANH_DIVISOR = 200.0
SATURATED = 0.95


def candidate_positions() -> list[tuple[str, bool]]:
    """(fen, side_to_move_is_white) for every position in both suites."""
    import chess
    out, seen = [], set()
    for suite in SUITES:
        path = REPO_ROOT / "evaluation" / "positions" / f"{suite}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["positions"] if isinstance(data, dict) else data
        for row in rows:
            fen = row["fen"]
            if fen not in seen:
                seen.add(fen)
                out.append((fen, chess.Board(fen).turn == chess.WHITE))
    return out


def child_fens(fen: str) -> list[str]:
    """The positions the engine actually scores: one per legal move."""
    import chess
    board = chess.Board(fen)
    kids = []
    for mv in board.legal_moves:
        board.push(mv)
        kids.append(board.fen())
        board.pop()
    return kids


def describe(v: np.ndarray) -> dict:
    return {"n": int(v.size), "mean": round(float(v.mean()), 4),
            "std": round(float(v.std()), 4),
            "mean_abs": round(float(np.abs(v).mean()), 4),
            "p05": round(float(np.percentile(v, 5)), 4),
            "p95": round(float(np.percentile(v, 95)), 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Fusion saturation and within-position CNN separation, per arm.")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--contrast", default=f"{CONTRAST[0]}:{CONTRAST[1]}",
                    help="SRC:DST pair for the verdict, e.g. A2:A3")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    arms = tuple(args.arms)
    src_arm, dst_arm = args.contrast.split(":", 1)
    out_path = args.out

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import keras

    from training import representations as REPS
    from training import train as T

    positions = candidate_positions()
    print(f"positions: {len(positions)} "
          f"({sum(1 for _, w in positions if w)} white to move, "
          f"{sum(1 for _, w in positions if not w)} black to move)")

    # Encode every child position once; all arms score the identical tensor.
    groups, fens, is_white_parent = [], [], []
    for fen, white in positions:
        kids = child_fens(fen)
        if not kids:
            continue
        groups.append((len(fens), len(fens) + len(kids)))
        fens.extend(kids)
        is_white_parent.append(white)
    # Arms may use different board encodings, so encode once PER REPRESENTATION
    # and reuse. The candidate positions themselves are identical for every arm.
    encoded = {}

    def encode_for(arm):
        # Arms defined outside train.ARMS (C8a lives in train_v2.ARMS) default to
        # planes12, matching evaluate_arm.run_suite's existing dispatch.
        rep_name = T.ARMS.get(arm, {}).get("representation", "planes12")
        if rep_name not in encoded:
            encoded[rep_name] = REPS.get(rep_name).encode_many(fens)
            print(f"encoded {len(fens)} candidates as {rep_name} "
                  f"{encoded[rep_name].shape}")
        return encoded[rep_name]

    out = {"stage": "C6-A2 saturation probe",
           "status": "EXPLORATORY - not pre-registered",
           "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_parent_positions": len(groups),
           "n_candidate_positions": len(fens),
           "tanh_divisor": TANH_DIVISOR,
           "saturation_threshold": SATURATED,
           "per_run": {}}

    parent_white = np.array(is_white_parent)
    header = (f"{'run':8s}{'raw |cnn|':>11s}{'raw std':>10s}{'|tanh|':>9s}"
              f"{'%sat':>8s}{'within-pos tanh spread':>24s}")
    print(f"\n{header}")
    print(f"{'':8s}{'':11s}{'':10s}{'':9s}{'':8s}{'(white / black parents)':>24s}")

    for arm in arms:
        for seed in SEEDS:
            model_path = EXP / arm / f"seed_{seed}" / "models" / "cnn_model.keras"
            if not model_path.is_file():
                print(f"  {arm} s{seed}: MISSING {model_path}")
                continue
            model = keras.models.load_model(model_path, compile=False)
            raw = np.asarray(model.predict(encode_for(arm), verbose=0,
                                           batch_size=512)).reshape(-1)
            squashed = np.tanh(raw / TANH_DIVISOR)
            sat = float((np.abs(squashed) > SATURATED).mean())

            # Ranking depends on the SPREAD of the CNN term inside one position.
            spreads = np.array([squashed[a:b].max() - squashed[a:b].min()
                                for a, b in groups])
            w_spread = float(spreads[parent_white].mean())
            b_spread = float(spreads[~parent_white].mean())

            out["per_run"][f"{arm}_seed_{seed}"] = {
                "raw_cnn": describe(raw),
                "tanh_cnn": describe(squashed),
                "saturated_fraction": round(sat, 6),
                "within_position_tanh_spread": {
                    "mean_all": round(float(spreads.mean()), 6),
                    "mean_white_to_move": round(w_spread, 6),
                    "mean_black_to_move": round(b_spread, 6),
                    "median_all": round(float(np.median(spreads)), 6),
                },
            }
            print(f"  {arm} s{seed}{np.abs(raw).mean():>11.1f}{raw.std():>10.1f}"
                  f"{np.abs(squashed).mean():>9.4f}{100 * sat:>7.1f}%"
                  f"{w_spread:>12.4f} /{b_spread:>9.4f}")

    # ------------------------------------------------------------------ verdict
    def arm_mean(arm, path):
        vals = []
        for seed in SEEDS:
            d = out["per_run"].get(f"{arm}_seed_{seed}")
            if d:
                cur = d
                for k in path:
                    cur = cur[k]
                vals.append(cur)
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n{'-' * 72}\nVERDICT\n{'-' * 72}")
    sat_by_arm = {a: arm_mean(a, ["saturated_fraction"]) for a in arms}
    spr_by_arm = {a: arm_mean(a, ["within_position_tanh_spread", "mean_white_to_move"])
                  for a in arms}
    for a in arms:
        print(f"  {a}: saturated {100 * sat_by_arm[a]:5.1f}%   "
              f"within-position tanh spread (White to move) {spr_by_arm[a]:.4f}")

    supported = (sat_by_arm[dst_arm] > sat_by_arm[src_arm]
                 and spr_by_arm[dst_arm] < spr_by_arm[src_arm])
    out["contrast"] = {"src": src_arm, "dst": dst_arm}
    out["saturation_by_arm"] = {a: round(sat_by_arm[a], 6) for a in arms}
    out["white_within_position_spread_by_arm"] = {a: round(spr_by_arm[a], 6) for a in arms}
    out["hypothesis_supported"] = bool(supported)
    out["verdict"] = (
        f"SUPPORTED: {dst_arm} saturates more and separates White-to-move "
        f"candidates less than {src_arm}"
        if supported else
        f"NOT SUPPORTED: {dst_arm} does not both saturate more and separate less "
        f"than {src_arm}; the White-to-move regression is not explained by tanh "
        f"saturation")
    print(f"\n  {out['verdict']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    try:
        shown = out_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:          # an --out outside the repo, e.g. a scratch dir
        shown = str(out_path.resolve().as_posix())
    print(f"\nWROTE {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
