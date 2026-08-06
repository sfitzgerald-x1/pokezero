"""Environment for test subprocesses that import ``pokezero``.

``sys.path`` does not cross a ``subprocess`` boundary. So the ``src`` insertion in
``tests/__init__.py`` fixes in-process imports and does nothing for a child, which
resolves ``pokezero`` through the editable install instead -- i.e. through whatever
tree the ``.pth`` points at, not the checkout under test.

That is the same false-GREEN class the in-process fix closes, and it was proven live
rather than argued: mutating a clone's ``neural_policy.require_torch`` to raise
unconditionally left
``tests.test_neural_policy`` ... ``test_require_torch_applies_thread_env_to_real_torch_in_fresh_process``
**passing**, because the child imported the primary tree. One of the affected gates
is a panic gate (``tests/test_engine_search_no_panic.py``), where a false green is
exactly the outcome the gate exists to prevent.

``tests/test_promotion.py`` already did this correctly by hand; this is that pattern
in one place so the next spawn site inherits it.
"""

from __future__ import annotations

import os
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def subproc_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return ``base`` (default ``os.environ``) with this checkout's ``src`` prepended.

    Prepended, not replaced: a caller may already be setting ``PYTHONPATH`` for its
    own reasons, and dropping that would trade one silent resolution bug for another.
    """

    environment = dict(os.environ if base is None else base)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = _SRC if not existing else f"{_SRC}{os.pathsep}{existing}"
    return environment
