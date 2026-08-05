"""Make `tests/` importable however the suite is invoked.

Test modules import shared helpers as top-level modules (``from _showdown_root import ...``,
``from test_explosion_fixture import ...``). Under ``pytest`` that resolves because pytest
inserts the rootdir of each test file into ``sys.path``. Under ``python -m unittest
tests.test_x`` it does NOT: only the repo root goes on the path, so the import raises
``ModuleNotFoundError`` and takes the whole module down.

That second form is this repo's own convention -- ``.github/workflows/fleet-worker.yml`` runs
it, and the documented repro commands throughout ``docs/`` use it. It is also the failure mode
that stays invisible: CI happens to name modules that do not use these helpers, so CI would
have stayed green while every documented manual repro broke.

conftest.py is imported by pytest but NOT by unittest, so the insertion is repeated in
``tests/__init__.py`` -- which unittest DOES import when resolving ``tests.test_x``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Same reason as tests/__init__.py: pin ``pokezero`` to this checkout ahead of the
# editable install, so a pytest run from a scratch clone tests the clone's tree.
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
