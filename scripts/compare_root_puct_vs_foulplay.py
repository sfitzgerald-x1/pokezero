#!/usr/bin/env python3
"""Run paired raw-checkpoint and root-PUCT benchmarks against external foul-play."""

from __future__ import annotations

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from pokezero.foulplay_bridge import comparison_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(comparison_main())
