"""Makes ``tests/`` importable as a package, and puts it on ``sys.path``.

The path insertion is what matters. Test modules import shared helpers as top-level modules
(``from _showdown_root import ...``, ``from test_explosion_fixture import ...``). pytest makes
that work by inserting each test file's rootdir; ``python -m unittest tests.test_x`` does not,
and that form is this repo's own convention -- ``.github/workflows/fleet-worker.yml`` runs it,
and the documented repro commands throughout ``docs/`` use it.

This file rather than ``conftest.py`` alone because unittest never imports conftest. It DOES
import this, when resolving ``tests.test_x``. Both exist so either runner works, and neither is
load-bearing on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
