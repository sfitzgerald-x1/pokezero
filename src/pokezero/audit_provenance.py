"""Public-source provenance helpers for audit artifacts.

Runtime images deliberately omit the public repository's ``.git`` directory.
The private image builder injects the immutable source revision through an
environment variable, while local development keeps using ``git rev-parse``.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


PUBLIC_REPO_COMMIT_ENV = "POKEZERO_PUBLIC_REPO_COMMIT"
_FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def public_repo_commit(repo_root: Path) -> str | None:
    """Return the injected source revision or the local checkout revision.

    The injected value is a trusted image-build contract. Cluster launchers
    must not override it with an arbitrary runtime environment value.

    An invalid injected value is never recorded as provenance. Complete-artifact
    validation can therefore fail closed instead of publishing an apparently
    source-pinned report.
    """

    injected = os.environ.get(PUBLIC_REPO_COMMIT_ENV, "").strip().lower()
    if _FULL_GIT_SHA.fullmatch(injected):
        return injected
    try:
        commit = subprocess.check_output(
            ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().lower()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit if _FULL_GIT_SHA.fullmatch(commit) else None
