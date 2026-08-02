"""Public-repo invariant guard: no internal-environment identifiers in tracked files.

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

_ALLOWED_FILES = {
    # This guard assembles the patterns from fragments and never contains them,
    # but keep an explicit empty allowlist here so any future exception is a
    # visible, reviewable diff rather than a pattern tweak.
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

        self.assertEqual(
            violations,
            [],
            "internal-environment identifiers in tracked files — reword with "
            "neutral placeholders (the private deployment tooling holds the "
            "real values):\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
