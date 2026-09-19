"""Pure label policy: perspective normalisation, mate mapping, clipping.

No Stockfish calls and no I/O, so every rule here is unit-testable in
milliseconds. `build_dataset.py` supplies raw engine output; everything that
turns it into a training label lives here.

--------------------------------------------------------------------------
LABEL PERSPECTIVE  -  "white"
--------------------------------------------------------------------------
Every label is WHITE-POSITIVE: positive means good for White, negative means
good for Black, regardless of whose turn it is.

Stockfish's native UCI output is relative to the side to move. The conversion is
done HERE, explicitly, from `side_to_move_is_white` - not by the `stockfish`
wrapper's `turn_perspective=False` mode, which decides perspective by testing
whether the substring "w" appears in the FEN. That happens to work, but it is a
string heuristic; doing the arithmetic ourselves keeps the rule visible and
testable, and lets the raw side-to-move value be stored for auditing.

This replaces the original notebook's behaviour, which used the wrapper default
(`turn_perspective=True`) and therefore produced side-to-move-relative labels -
mixing two sign conventions in one dataset, since 446 of its 10,000 positions
were Black to move.

--------------------------------------------------------------------------
MATE POLICY
--------------------------------------------------------------------------
Stockfish reports mates as {"type": "mate", "value": <distance>}, where the
distance is a small integer, NOT a centipawn amount. The original notebook read
["value"] without checking ["type"], so "mate in 1" was stored as the label 1
and an already-checkmated position as 0 - teaching the model that forced mates
are equal. This module maps them onto the centipawn axis instead:

    magnitude(d) = MATE_SCORE_BASE - MATE_SCORE_STEP * min(|d|, MATE_MAX_DISTANCE)

    d = 0  ->  2000   (mate already on the board: the most decisive value)
    d = 1  ->  1990
    d = 5  ->  1950
    d >= 49 -> 1510   (floor, still strictly above CP_CLIP)

The mapping is deterministic, symmetric in colour, and monotonically decreasing
in mate distance, so a faster mate always scores as better. Every mate magnitude
is strictly greater than the centipawn clip bound, so any mate outranks any
non-mate evaluation.

--------------------------------------------------------------------------
THE mate == 0 AMBIGUITY
--------------------------------------------------------------------------
VERIFIED against Stockfish 17.1 via the `stockfish` 4.0.8 wrapper: an
already-checkmated position returns {"type": "mate", "value": 0}. Zero carries
no sign, so the value alone cannot say who won.

INTERPRETATION (documented and enforced): mate == 0 means THE SIDE TO MOVE IS
CHECKMATED, so the side to move has LOST. The winner is therefore the opponent,
and the sign is taken from `side_to_move_is_white` rather than from the value.

If a caller reports mate == 0 for a board that is not actually checkmate, that
is an unexpected engine state; `classify_mate_zero` flags it so the builder can
record it rather than silently guessing.

--------------------------------------------------------------------------
ORDERING
--------------------------------------------------------------------------
    1. mate mapping        (mate distance -> signed magnitude on the cp axis)
    2. perspective         (side-to-move-relative -> White-positive)
    3. clipping            (CENTIPAWN LABELS ONLY)

Clipping is applied only to `type == "cp"` labels. Mate labels are produced by
step 1 at magnitudes above the clip bound by construction, so clipping them
would collapse the whole mate scale back onto the cp range and undo the policy.
The original notebook's `np.clip(val, -1500, 1500)` was commented "clip outlier
mate scores"; it clipped zero of 10,000 values, because mate values were small
distances rather than large centipawn numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

# --- policy constants (recorded verbatim in the manifest) ---------------------
LABEL_PERSPECTIVE = "white"

CP_CLIP = 1500            # centipawn labels are clamped to +/- this
MATE_SCORE_BASE = 2000    # magnitude for mate already on the board (d = 0)
MATE_SCORE_STEP = 10      # magnitude lost per additional move to mate
MATE_MAX_DISTANCE = 49    # distance cap -> floor magnitude 1510 > CP_CLIP

EVAL_TYPE_CP = "cp"
EVAL_TYPE_MATE = "mate"


def policy_summary() -> dict:
    """Machine-readable statement of the policy, embedded in the manifest."""
    return {
        "label_perspective": LABEL_PERSPECTIVE,
        "cp_policy": (
            "raw Stockfish centipawns, side-to-move-relative, converted to "
            "White-positive, then clipped"
        ),
        "mate_policy": (
            f"magnitude = {MATE_SCORE_BASE} - {MATE_SCORE_STEP} * "
            f"min(|mate_distance|, {MATE_MAX_DISTANCE}); signed so that positive "
            f"means White wins. Monotonic in mate distance and strictly above "
            f"the centipawn clip bound."
        ),
        "mate_zero_policy": (
            "mate == 0 means the SIDE TO MOVE is checkmated and has lost; the "
            "sign is taken from the side to move, not from the value, because "
            "zero carries no sign. Magnitude is the maximum "
            f"({MATE_SCORE_BASE})."
        ),
        "clip_bounds": [-CP_CLIP, CP_CLIP],
        "clip_applies_to": "cp labels only; mate labels are never clipped",
        "ordering": "mate mapping -> perspective normalisation -> clipping (cp only)",
        "mate_score_base": MATE_SCORE_BASE,
        "mate_score_step": MATE_SCORE_STEP,
        "mate_max_distance": MATE_MAX_DISTANCE,
    }


# ============================================================ perspective

def to_white_positive(value: int, side_to_move_is_white: bool) -> int:
    """Convert a side-to-move-relative score to a White-positive one.

    Stockfish's native output is relative to the side to move, so a Black-to-move
    value must be negated to express it from White's point of view.
    """
    return int(value) if side_to_move_is_white else -int(value)


# ============================================================ mate

def mate_magnitude(distance: int) -> int:
    """Centipawn-axis magnitude for a mate at `distance` moves. Always positive."""
    d = min(abs(int(distance)), MATE_MAX_DISTANCE)
    return MATE_SCORE_BASE - MATE_SCORE_STEP * d


def classify_mate_zero(is_checkmate: bool) -> bool:
    """True when a reported mate == 0 is consistent with the board.

    Returned rather than asserted so the builder can count and report anomalies
    instead of the pipeline guessing.
    """
    return bool(is_checkmate)


def mate_to_white_positive(mate_distance: int, side_to_move_is_white: bool) -> int:
    """Signed, White-positive label for a mate score.

    `mate_distance` is Stockfish's side-to-move-relative mate value:
        > 0  the side to move delivers mate
        < 0  the side to move gets mated
        == 0 the side to move is already checkmated (see module docstring)
    """
    d = int(mate_distance)
    if d > 0:
        winner_is_white = side_to_move_is_white          # side to move mates
    else:
        # d < 0: side to move gets mated.
        # d == 0: side to move is already checkmated.
        # Both mean the OPPONENT wins.
        winner_is_white = not side_to_move_is_white
    magnitude = mate_magnitude(d)
    return magnitude if winner_is_white else -magnitude


# ============================================================ clipping

def clip_cp(value: int, bound: int = CP_CLIP) -> int:
    """Clamp a centipawn label. Never applied to mate labels."""
    return max(-abs(bound), min(abs(bound), int(value)))


# ============================================================ combined

@dataclass(frozen=True)
class Label:
    """A finished training label plus everything needed to audit it."""
    label: int                  # final White-positive value used for training
    eval_type: str              # "cp" | "mate"
    raw_value: int              # Stockfish's side-to-move-relative output
    side_to_move: str           # "white" | "black"
    was_clipped: bool
    mate_zero_consistent: bool | None   # None unless eval_type == "mate" and raw == 0

    def as_dict(self) -> dict:
        return asdict(self)


def make_label(
    eval_type: str,
    raw_value: int,
    side_to_move_is_white: bool,
    is_checkmate: bool = False,
    cp_clip: int = CP_CLIP,
) -> Label:
    """Apply the full policy: mate mapping -> perspective -> clip (cp only)."""
    stm = "white" if side_to_move_is_white else "black"

    if eval_type == EVAL_TYPE_MATE:
        value = mate_to_white_positive(raw_value, side_to_move_is_white)
        consistent = classify_mate_zero(is_checkmate) if int(raw_value) == 0 else None
        return Label(
            label=value,
            eval_type=EVAL_TYPE_MATE,
            raw_value=int(raw_value),
            side_to_move=stm,
            was_clipped=False,
            mate_zero_consistent=consistent,
        )

    if eval_type == EVAL_TYPE_CP:
        white_positive = to_white_positive(raw_value, side_to_move_is_white)
        clipped = clip_cp(white_positive, cp_clip)
        return Label(
            label=clipped,
            eval_type=EVAL_TYPE_CP,
            raw_value=int(raw_value),
            side_to_move=stm,
            was_clipped=clipped != white_positive,
            mate_zero_consistent=None,
        )

    raise ValueError(f"unknown evaluation type: {eval_type!r}")
