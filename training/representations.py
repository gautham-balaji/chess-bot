"""Registry mapping an arm's representation name to its encoder module.

Additive indirection so an arm can select its board encoding by name. Every
encoder module exposes the same surface:

    BOARD_SHAPE, N_PLANES, PLANE_NAMES
    board_to_planes(board) -> np.ndarray
    fen_to_planes(fen)     -> np.ndarray
    encode_many(fens)      -> np.ndarray
    representation_summary() -> dict

`planes12` is what A0, A1 and A2 used and is identical to `engine.board_to_planes`.
`planes18` is the C6 representation from the audit's section K, used by A3.
`planes16` is planes12 plus the four castling-right planes only, used by A13.
"""
from __future__ import annotations

from training import representation as _planes12
from training import representation16 as _planes16
from training import representation18 as _planes18

PLANES12 = "planes12"
PLANES16 = "planes16"
PLANES18 = "planes18"

REGISTRY = {PLANES12: _planes12, PLANES16: _planes16, PLANES18: _planes18}
NAMES = tuple(REGISTRY)


def get(name: str):
    """Return the encoder module for `name`, or fail loudly."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown representation {name!r}; expected one of {NAMES}") from None
