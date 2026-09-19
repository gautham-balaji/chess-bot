"""Central path and Stockfish configuration.

Phase 1 (reproducibility). This module exists so that nothing in the project
depends on the directory Python happens to be launched from, and so that no
machine-specific absolute path is embedded in source.

It deliberately contains NO engine logic: no scoring, no ranking, no model
inference. Search depth is re-exported unchanged from the value the project
already used.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# --- Repository-relative paths -------------------------------------------------
# Anchored to THIS FILE, not to the current working directory, so imports work
# regardless of where the interpreter was started.
REPO_ROOT: Path = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE per line, '#' comments, no dependencies.

    Existing environment variables always win, so an explicitly exported value
    is never silently overridden by the file. The file is gitignored: it holds
    machine-specific values such as where Stockfish is installed.
    """
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        # A malformed or unreadable .env must never break startup.
        pass


_load_dotenv(REPO_ROOT / ".env")

MODELS_DIR: Path = Path(os.environ.get("CHESS_BOT_MODELS_DIR") or (REPO_ROOT / "models"))

CNN_MODEL_PATH: Path = MODELS_DIR / "cnn_model.keras"
WEIGHT_MODEL_PATH: Path = MODELS_DIR / "weight_model.pkl"

# Artifacts produced by the training notebook but NOT used by the runtime engine.
# Kept for provenance; see docs/REPRODUCIBILITY.md.
UNUSED_TRAINING_ARTIFACTS = {
    "rf_model": MODELS_DIR / "rf_model.pkl",
    "mlp_model": MODELS_DIR / "mlp_model.pkl",
    "scaler": MODELS_DIR / "scaler.pkl",
}

# --- Stockfish ----------------------------------------------------------------
# Unchanged from the pre-Phase-1 value used in app.py and benchmark_stockfish.py.
# Phase 1 does not tune Stockfish strength in any way.
STOCKFISH_DEPTH: int = 8

_COMMON_STOCKFISH_LOCATIONS = {
    "win32": [
        r"C:\Program Files\Stockfish\stockfish.exe",
        r"C:\stockfish\stockfish.exe",
    ],
    "linux": [
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
    ],
    "darwin": [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
    ],
}


def find_stockfish() -> str | None:
    """Resolve a Stockfish executable, or return None if unavailable.

    Resolution order:
      1. ``STOCKFISH_PATH`` environment variable (explicit wins)
      2. ``stockfish`` on PATH
      3. common per-platform install locations

    Returns None rather than raising: Stockfish is optional at runtime, and
    every caller in this project already treats it as optional.
    """
    explicit = os.environ.get("STOCKFISH_PATH")
    if explicit and Path(explicit).is_file():
        return explicit

    on_path = shutil.which("stockfish")
    if on_path:
        return on_path

    for candidate in _COMMON_STOCKFISH_LOCATIONS.get(sys.platform, []):
        if Path(candidate).is_file():
            return candidate

    return None


def stockfish_status() -> dict:
    """Describe how Stockfish resolved, for diagnostics and docs."""
    path = find_stockfish()
    return {
        "available": path is not None,
        "path": path,
        "env_STOCKFISH_PATH": os.environ.get("STOCKFISH_PATH"),
        "found_on_PATH": shutil.which("stockfish"),
        "platform": sys.platform,
        "depth": STOCKFISH_DEPTH,
    }
