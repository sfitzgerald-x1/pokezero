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

# And pin ``pokezero`` itself to THIS checkout, ahead of the editable install's .pth.
#
# 157 of 207 test modules have no ``sys.path.insert`` of their own, so they resolve
# ``pokezero`` through the editable install -- whatever tree that points at, which is not
# necessarily the tree the test file lives in. From the primary checkout the two are the
# same directory and nothing is wrong. From a SECOND clone or a ``git worktree`` sharing
# the venv they are not, and every one of those 157 modules silently exercises the other
# tree while appearing to test this one.
#
# That is not hypothetical and it is not rare: reviewing a PR from a scratch clone is a
# workflow this repo's own guidance prescribes, and a reviewer hit exactly this -- a
# module-level run read a false GREEN against a tree that did not contain the change.
# Worse, it is ORDER-DEPENDENT: ``sys.modules`` caches by name, so in one
# ``python -m unittest A B C`` the first module to import ``pokezero`` fixes the tree for
# all of them. A mutation matrix can therefore be correct only by argument ordering.
#
# Inserting here makes the 157 per-module inserts redundant rather than load-bearing,
# because unittest imports this package to resolve ``tests.test_x``.
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
