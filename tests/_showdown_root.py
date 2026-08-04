"""One place tests resolve the pokemon-showdown checkout.

Before this existed, ~23 tracked test files each wrote

    os.environ.get("POKEZERO_SHOWDOWN_ROOT", "/Users/<maintainer>/...")

which put a username and a local filesystem layout into a PUBLIC repo, and gave every other
contributor a default that cannot exist — so the tests silently skipped, or failed with a
confusing missing-file error, depending on how each file happened to guard.

Resolution order lives in :func:`pokezero.local_showdown.default_showdown_root` so the library
and the tests cannot drift: ``POKEZERO_SHOWDOWN_ROOT`` first, then conventional locations
expressed via ``Path.home()`` and the repo root, none of which name a user.

Use :func:`showdown_root` for the path and :func:`requires_showdown` to skip a test that needs a
real checkout. Prefer the skip decorator over an ad-hoc ``os.path.exists`` check: it reports WHY
the test did not run, which the silent-skip behaviour this module replaces did not.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest

from pokezero.local_showdown import default_showdown_root

__all__ = ["showdown_root", "showdown_root_str", "has_showdown", "requires_showdown"]


def showdown_root() -> Path:
    """The checkout to test against. Never a user-specific literal."""
    return default_showdown_root()


def showdown_root_str() -> str:
    """:func:`showdown_root` as ``str``, for the many APIs that take a string path."""
    return str(showdown_root())


def has_showdown() -> bool:
    """Whether the resolved root actually looks like a Showdown checkout.

    Checks ``data/`` rather than the directory itself, because an empty or partial directory at
    a conventional location is the case that produces a confusing failure deep inside a loader
    rather than an honest skip.
    """
    return (showdown_root() / "data").is_dir()


def requires_showdown(reason: str = "needs a pokemon-showdown checkout"):
    """Skip decorator that names the resolved path, so a skip is diagnosable.

    A bare ``skipUnless(...)`` tells you a test did not run; it does not tell you WHERE it
    looked, which is the only thing you need to know to fix it.
    """
    return unittest.skipUnless(
        has_showdown(),
        f"{reason} (looked in {showdown_root()}; set POKEZERO_SHOWDOWN_ROOT to override)",
    )


def showdown_env(**extra: str) -> "dict[str, str]":
    """``os.environ`` plus an explicit ``POKEZERO_SHOWDOWN_ROOT``, for subprocess tests.

    A subprocess inherits the variable only if the parent had it SET. When the root came from a
    conventional-location fallback instead, the child would resolve it independently — usually
    to the same place, but not if the child runs from a different working directory. Passing it
    explicitly removes the ambiguity.
    """
    return {**os.environ, "POKEZERO_SHOWDOWN_ROOT": showdown_root_str(), **extra}
