"""Makes ``tests/`` importable as a package, and pins imports to THIS checkout.

Two insertions, with different reasons.

``tests/`` is what lets shared helpers resolve. Test modules import them as top-level
modules (``from _showdown_root import ...``, ``from test_explosion_fixture import ...``). pytest makes
that work by inserting each test file's rootdir; ``python -m unittest tests.test_x`` does not,
and that form is this repo's own convention -- ``.github/workflows/fleet-worker.yml`` runs it,
and the documented repro commands throughout ``docs/`` use it.

``src/`` is what pins ``pokezero`` itself to this checkout rather than to whatever tree
the editable install points at. See the comment on that insertion below.

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
# 180 of the 208 test modules resolve ``pokezero`` through the editable install rather
# than through the tree they live in: 157 insert nothing at all, and 23 of the 51 that do
# insert only non-``src`` paths (22 ``scripts/``, one ``docs/token-format``). The install
# points wherever its .pth says, which need not be this checkout.
#
# From the primary checkout the two are the same directory and nothing is wrong. From a
# SECOND clone or a ``git worktree`` sharing the venv they are not, and all 180 silently
# exercise the other tree while appearing to test this one. That is not hypothetical:
# reviewing from a scratch clone is a workflow this repo's guidance prescribes, and a
# reviewer hit exactly this -- a module-level run read a false GREEN against a tree that
# did not contain the change under review.
#
# It is also ORDER-DEPENDENT, which is worse than a plain gap: ``sys.modules`` caches by
# name, so in one ``python -m unittest A B C`` the first module to import ``pokezero``
# fixes the tree for all of them. A mutation matrix can be correct only by argument order.
#
# Inserting here fixes all 180 for every invocation style this repo uses -- unittest
# imports this package to resolve ``tests.test_x``, which is what CI and every documented
# repro command use. It does NOT make the 51 existing inserts redundant: the 23 above add
# paths nothing here provides, and stay load-bearing. Two styles are also still uncovered
# because neither this file nor conftest is imported: ``python tests/test_x.py`` and
# ``unittest discover -s tests`` (which makes ``tests/`` the top-level dir). Neither
# appears in ``.github/workflows/`` or ``docs/``.
#
# And ``sys.path`` does NOT cross a subprocess boundary -- see ``tests/_subproc_env.py``
# for the child-process half, which is the same hazard and needed its own fix.
# Move-to-front, NOT insert-if-absent. A membership test is satisfied by the string
# appearing anywhere on the path, including behind another checkout's ``src``
# inherited via PYTHONPATH -- and then the wrong tree still wins. Collapsing this
# back to ``if _SRC_DIR not in sys.path`` silently reopens that hole.
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR in sys.path:
    sys.path.remove(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)
