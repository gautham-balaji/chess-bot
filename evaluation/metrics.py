"""Pure evaluation mathematics: POV handling, regret, mate policy, aggregation.

This module contains NO chess engine calls and NO I/O, so every rule below is
unit-testable in milliseconds. The runner (evaluate.py) supplies raw Stockfish
output; everything that turns it into a number lives here.

--------------------------------------------------------------------------
POV CONVENTION  (one convention, used everywhere)
--------------------------------------------------------------------------
Move quality is always judged from the perspective of THE PLAYER WHO CHOSE THE
MOVE - the side to move in the ORIGINAL position. Call that side S.

After S plays a move it is the opponent's turn, so a raw Stockfish score for the
resulting position is relative to the OPPONENT. Every score is therefore
converted to S's POV exactly once, at the boundary, by `score_from_pov`.

Positive cp = good for S. Negative cp = bad for S. No exceptions.

--------------------------------------------------------------------------
REGRET
--------------------------------------------------------------------------
    regret_cp = eval_after_stockfish_move - eval_after_engine_move

both evaluated by the SAME Stockfish configuration and both converted to S's POV.

- regret == 0  : the engine's move is as good as the reference (identical when
                 the engine picked the same move)
- regret >  0  : the engine's move gave up `regret` centipawns
- regret <  0  : the engine's move scored BETTER than Stockfish's own choice.
                 This is possible at fixed shallow depth (search instability)
                 and is NOT clamped - clamping would silently bias the mean.

The engine's own score never enters this calculation. That is the whole point:
the engine's score is not in centipawns, so subtracting it from a Stockfish
score would be a unit error.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

# Regret above this is counted a blunder. A measurement convention for this
# report - not a claim about universal chess truth. 300cp ~ three pawns.
BLUNDER_THRESHOLD_CP = 300


# ============================================================ score container

@dataclass(frozen=True)
class PovEval:
    """A position's evaluation from ONE fixed perspective.

    Exactly one of `cp` / `mate_in` is set.

    mate_in > 0 : the POV side delivers mate in that many moves
    mate_in < 0 : the POV side is mated in that many moves
    mate_in == 0: mate is already on the board, delivered by the POV side
    """
    cp: int | None = None
    mate_in: int | None = None

    def __post_init__(self):
        if (self.cp is None) == (self.mate_in is None):
            raise ValueError("exactly one of cp / mate_in must be set")

    @property
    def is_mate(self) -> bool:
        return self.mate_in is not None

    @property
    def mate_is_winning(self) -> bool:
        if self.mate_in is None:
            raise ValueError("not a mate score")
        return self.mate_in >= 0

    def as_dict(self) -> dict:
        return asdict(self)


def score_from_pov(pov_score, pov_color) -> PovEval:
    """Convert a python-chess PovScore to a PovEval from `pov_color`'s view.

    `pov_score` is chess.engine.PovScore; `pov_color` is chess.WHITE/BLACK.
    Kept as the single conversion point so POV logic cannot drift.
    """
    relative = pov_score.pov(pov_color)
    if relative.is_mate():
        return PovEval(mate_in=relative.mate())
    return PovEval(cp=relative.score())


# ============================================================ mate taxonomy

MATE_STATUS_NONE = "none"
MATE_STATUS_BOTH_WIN = "both_forced_mate_for_mover"
MATE_STATUS_MISSED_MATE = "missed_forced_mate"
MATE_STATUS_ENGINE_FOUND_MATE = "engine_found_mate_reference_did_not"
MATE_STATUS_ENGINE_ALLOWS_MATE = "engine_move_allows_forced_mate"
MATE_STATUS_BOTH_LOSE = "both_moves_lose_to_forced_mate"
MATE_STATUS_OTHER = "mixed_mate"


def classify_mate(after_reference: PovEval, after_engine: PovEval) -> str:
    """Label the mate situation of a position pair, both from the mover's POV.

    Any label other than MATE_STATUS_NONE means centipawn regret is NOT defined
    for that position: a mate score is an ordinal, not a centipawn quantity, and
    mapping it onto a cp axis (e.g. mate_score=10000) would corrupt the mean.
    Such positions are reported by count instead.
    """
    ref_mate, eng_mate = after_reference.is_mate, after_engine.is_mate

    if not ref_mate and not eng_mate:
        return MATE_STATUS_NONE

    if ref_mate and eng_mate:
        ref_win = after_reference.mate_is_winning
        eng_win = after_engine.mate_is_winning
        if ref_win and eng_win:
            return MATE_STATUS_BOTH_WIN
        if not ref_win and not eng_win:
            return MATE_STATUS_BOTH_LOSE
        if ref_win and not eng_win:
            return MATE_STATUS_ENGINE_ALLOWS_MATE
        return MATE_STATUS_ENGINE_FOUND_MATE

    if ref_mate and not eng_mate:
        # Reference forces mate, engine's move does not.
        return (MATE_STATUS_MISSED_MATE if after_reference.mate_is_winning
                else MATE_STATUS_OTHER)

    # Engine's move produces a mate score, reference's does not.
    return (MATE_STATUS_ENGINE_FOUND_MATE if after_engine.mate_is_winning
            else MATE_STATUS_ENGINE_ALLOWS_MATE)


def compute_regret(after_reference: PovEval, after_engine: PovEval):
    """Return (regret_cp | None, mate_status).

    regret_cp is None precisely when a mate score is involved.
    """
    status = classify_mate(after_reference, after_engine)
    if status != MATE_STATUS_NONE:
        return None, status
    return after_reference.cp - after_engine.cp, status


# ============================================================ agreement

def top1_agreement(engine_move_uci: str | None, reference_move_uci: str | None) -> bool:
    if engine_move_uci is None or reference_move_uci is None:
        return False
    return engine_move_uci == reference_move_uci


def top_n_containment(engine_move_uci: str | None, reference_moves: Sequence[str]) -> bool:
    if engine_move_uci is None:
        return False
    return engine_move_uci in reference_moves


def random_move_baseline(legal_move_counts: Iterable[int], top_n: int = 1) -> float | None:
    """Expected agreement if a legal move were picked uniformly at random.

    For top-1 this is mean(1/L); for top-N, mean(min(N, L)/L). Included because
    an agreement rate is uninterpretable without a chance reference - endgames
    have few legal moves and inflate agreement for trivial reasons.
    """
    counts = [c for c in legal_move_counts if c and c > 0]
    if not counts:
        return None
    return sum(min(top_n, c) / c for c in counts) / len(counts)


# ============================================================ aggregation

def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile, stated explicitly so the number is unambiguous."""
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def summarize(values: Sequence[float]) -> dict:
    """min / median / mean / p95 / max plus n. Returns nulls for an empty input
    rather than raising, so a run with no defined regrets still produces output."""
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None,
                "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 2),
        "median": round(statistics.median(ordered), 2),
        "mean": round(statistics.fmean(ordered), 2),
        "p95": round(percentile(ordered, 0.95), 2),
        "max": round(ordered[-1], 2),
    }


def blunder_rate(regrets: Sequence[float], threshold: int = BLUNDER_THRESHOLD_CP) -> dict:
    """Fraction of DEFINED regrets exceeding `threshold`.

    The denominator excludes mate-involved positions, because they have no
    centipawn regret. That is stated in the returned dict, not left implicit.
    """
    if not regrets:
        return {"blunders": 0, "denominator": 0, "rate": None, "threshold_cp": threshold}
    blunders = sum(1 for r in regrets if r > threshold)
    return {
        "blunders": blunders,
        "denominator": len(regrets),
        "rate": round(blunders / len(regrets), 4),
        "threshold_cp": threshold,
        "denominator_note": "positions with a defined centipawn regret; "
                            "mate-involved positions are excluded",
    }


def rate(numerator: int, denominator: int) -> dict:
    """A rate reported with its numerator and denominator, never bare."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": round(numerator / denominator, 4) if denominator else None,
        "percent": round(100 * numerator / denominator, 2) if denominator else None,
    }


# ============================================================ rank correlation

def spearman_rank_correlation(engine_ranks: Sequence[int], reference_ranks: Sequence[int]):
    """Spearman rho between two rankings of the SAME candidate moves.

    Operates on ranks, never on scores, so the engine's non-centipawn scale is
    irrelevant - only ordering is compared. Returns None when there are fewer
    than 3 shared candidates or either ranking is constant (rho undefined).
    """
    if len(engine_ranks) != len(reference_ranks) or len(engine_ranks) < 3:
        return None
    if len(set(engine_ranks)) < 2 or len(set(reference_ranks)) < 2:
        return None
    from scipy.stats import spearmanr  # local import: keeps this module import-light
    rho, _ = spearmanr(engine_ranks, reference_ranks)
    if rho != rho:  # NaN
        return None
    return float(rho)
