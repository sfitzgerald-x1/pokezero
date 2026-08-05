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

import gzip
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
        #
        # NO trailing slash requirement -- a bare reference with nothing after it is the most
        # likely reintroduction form, and the first version of this rule missed it. Escaped
        # (JSON `\/Users\/`) and Windows (`C:\Users\`) separators count as separators.
        # IGNORECASE because macOS filesystems are case-insensitive, so `/users/` names the
        # same directory and would otherwise be a trivial bypass.
        re.compile(r"[/\\](?:Us" + r"ers|ho" + r"me)[/\\]+[A-Za-z0-9._-]+", re.IGNORECASE),
    ),
    (
        "home directory flattened into a path segment",
        # The shape a temp-dir namer produces: a home path with its separators turned into
        # hyphens. It carries the username just as plainly but survives any rule looking for a
        # real path prefix, and it was still sitting in two tracked files after the first
        # scrub. (Not spelled out here -- this guard must not match itself.)
        re.compile(r"-(?:Us" + r"ers|ho" + r"me)-[A-Za-z0-9._]+-", re.IGNORECASE),
    ),
]

# PER-RULE, deliberately: a blanket file allowlist would exempt the file from the
# internal-cluster checks as well, silently weakening the older invariant to accommodate the
# newer one. Keyed by rule label.
# EMPTY, and that is the point. This held one carve-out -- the golden corpus sample's
# `rows.jsonl`, whose recorded provenance embedded absolute `sets_path` / `generator_path` /
# `showdown_root` values. Those could not be scrubbed in place (each row carries a `row_sha256`
# over its own payload, so editing them would mean forging the hash that makes the corpus
# tamper-evident), so the corpus had to be REGENERATED after the writer was fixed to emit
# relative paths. It has been: both committed samples are v4/v3 regenerations carrying
# `sets_path: data/random-battles/gen3/sets.json` and no absolute paths at all.
#
# Kept as an empty dict rather than deleted, so the mechanism stays available and the next
# exception has to be written down here to exist.
_ALLOWED_FOR_RULE: dict[str, set[str]] = {}


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
            path = REPO_ROOT / rel
            try:
                if path.suffix == ".gz":
                    # Compressed tracked files are DECOMPRESSED and scanned. `read_text` on a
                    # gzip yields bytes that decode to nothing resembling a path, so a leak inside
                    # one was invisible to this guard -- and the file that motivated the guard's
                    # last carve-out, the golden corpus, ships exactly such a sidecar
                    # (`fold.jsonl.gz`, 51,913 bytes of JSON carrying the same provenance fields as
                    # rows.jsonl). Both committed samples are clean today; the point is that they
                    # were clean unverifiably before this.
                    with gzip.open(path, "rt", errors="ignore") as handle:
                        text = handle.read()
                else:
                    text = path.read_text(errors="ignore")
            except (OSError, UnicodeDecodeError, EOFError, gzip.BadGzipFile):
                continue
            for label, needle in _FORBIDDEN:
                for match in re.finditer(re.escape(needle), text, re.IGNORECASE):
                    line = text.count("\n", 0, match.start()) + 1
                    violations.append(f"{rel}:{line}: {label} ({needle!r})")
            for label, pattern in _FORBIDDEN_PATTERNS:
                if rel in _ALLOWED_FOR_RULE.get(label, ()):
                    continue
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
