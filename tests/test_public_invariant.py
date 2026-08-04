"""Public-repo invariant guard: no internal-environment identifiers in tracked files.

Covers two classes. Fixed internal identifiers (cluster, registry, namespace) via _FORBIDDEN,
and PERSONAL FILESYSTEM PATHS via _FORBIDDEN_PATTERNS. The second was added on 2026-08-03 after
137 occurrences of a maintainer home directory were found across 48 tracked files -- 23 of them
test files that hardcoded it as the default Showdown checkout root, which leaked a username and
silently skipped for every other contributor.

The internal cluster deployment must leave zero trace in this public repo —
no private-repo names, cluster or node-pool identifiers, internal registry or
storage paths, namespaces, or kube contexts. Docs that need to reference such
things use neutral placeholders (``<private-store>/...``,
``<internal-registry>:...``, "the internal GPU environment") with the real
values recorded in the private deployment tooling.

This guard exists because the invariant was violated four separate times by
committed docs and audit artifacts before 2026-07-30 (see the divergence
ledger's invariant-scrub entries): documentation of the rule did not enforce
it, and reviewer greps only caught what a reviewer happened to scan. A test
runs every time. If this test fails, REWORD the file (see the scrub commit for
patterns) — do not add exceptions here without the owner's sign-off.

The patterns below are assembled from fragments so this file does not match
its own scan.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Assembled from fragments so the guard does not flag itself.
_FORBIDDEN = [
    ("private deploy repo name", "pokezero" + "-deploy"),
    ("cluster name", "olf" + "usa"),
    ("infra provider", "cru" + "soe"),
    ("node-pool identifier", "node" + "pool"),
    ("internal storage root", "/sha" + "red/"),
    ("internal namespace prefix", "scott-" + "experiment"),
    ("controller job prefix", "scott-" + "fnd-"),
    ("gpu pool label", "scott-" + "gpu-slice"),
    ("kube context flag", "kubectl " + "--context"),
]

# Regex rules, for classes of leak rather than fixed strings. Assembled from fragments for the
# same reason as _FORBIDDEN: an unfragmented pattern would match this file.
_FORBIDDEN_PATTERNS = [
    (
        "maintainer home directory",
        # Any user's home, not one specific username: a default naming SOMEONE's home is
        # useless to everyone else, so this must fail for a new contributor's path too.
        re.compile(r"/(?:Us" + r"ers|ho" + r"me)/[A-Za-z0-9._-]+/"),
    ),
]

_ALLOWED_FILES = {
    # Recorded provenance inside a TAMPER-EVIDENT golden corpus: every row carries a
    # `row_sha256` over its own payload, so the paths cannot be scrubbed in place without
    # forging those hashes -- which would defeat the guarantee the field exists to provide.
    # The corpus must be REGENERATED instead, after `randbat.py` stops recording absolute
    # `sets_path`/`generator_path` values. Tracked as a follow-up; this entry should shrink to
    # nothing, and is here so the exception is a visible, reviewable line rather than a
    # weakened pattern.
    "tests/data/golden_corpus_sample/rows.jsonl",
}


class PublicInvariantTest(unittest.TestCase):
    def test_fleet_worker_workflow_runs_for_every_tracked_change(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "fleet-worker.yml").read_text(encoding="utf-8")
        self.assertIn("pull_request:\n", workflow)
        self.assertNotIn("paths:", workflow)

    def test_no_internal_identifiers_in_tracked_files(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()

        violations: list[str] = []
        for rel in tracked:
            if rel in _ALLOWED_FILES:
                continue
            path = REPO_ROOT / rel
            try:
                text = path.read_text(errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue
            for label, needle in _FORBIDDEN:
                for match in re.finditer(re.escape(needle), text, re.IGNORECASE):
                    line = text.count("\n", 0, match.start()) + 1
                    violations.append(f"{rel}:{line}: {label} ({needle!r})")
            for label, pattern in _FORBIDDEN_PATTERNS:
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    violations.append(f"{rel}:{line}: {label} ({match.group(0)!r})")

        self.assertEqual(
            violations,
            [],
            "internal-environment identifiers in tracked files — reword with "
            "neutral placeholders (the private deployment tooling holds the "
            "real values):\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
