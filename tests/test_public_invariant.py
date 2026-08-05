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


class TestFileStructureTest(unittest.TestCase):
    """No test file may define tests below its `unittest.main()` block.

    A mid-file `if __name__ == "__main__": unittest.main()` makes direct execution and pytest
    disagree about what ran, and everything below it is invisible to `python <file>`. Measured
    before this guard existed: `test_belief.py` ran 16 tests directly against 69 under pytest,
    `test_belief_variant_narrowing.py` 10 against 13, and `test_randbat.py` broke two ways at once
    -- two helpers were defined below the block, so with POKEZERO_SHOWDOWN_ROOT set direct
    execution raised NameError, and with it unset it silently ran 23 of its 26 tests. (An earlier
    revision of this docstring said only "NameErrored", which is the half that needs the env var;
    the sibling comment in test_randbat.py was corrected and this copy was missed.)

    That is not a hypothetical tidiness rule. In every one of those three files the stranded region
    held guards added specifically to catch a defect -- so the tests written to prevent a regression
    were the ones not running. It happened twice in one pull request: the second time, new tests
    were appended below a stranded block in one file by the same change that fixed a stranded block
    in another.

    A file-structure rule catches the whole class; fixing each instance does not.
    """

    def test_no_tracked_test_defines_anything_below_unittest_main(self) -> None:
        import ast

        offenders: list[str] = []
        for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            # Detected structurally. A first version of this guard tested
            # `"unittest.main" in ast.dump(node)`, which NEVER matches: `ast.dump` renders the call
            # as `Attribute(value=Name(id='unittest'), attr='main')`, so that substring cannot
            # appear. The guard silently matched nothing in all 206 test files and reported a clean
            # repo -- caught by mutating a file to violate it and watching the guard pass.
            def _is_main_guard(node) -> bool:
                if not isinstance(node, ast.If):
                    return False
                if not any(
                    isinstance(sub, ast.Name) and sub.id == "__name__"
                    for sub in ast.walk(node.test)
                ):
                    return False
                # Both spellings: `unittest.main()` (Attribute) and `from unittest import main`
                # then `main()` (Name). The Attribute-only form missed the latter -- no test file
                # uses it today, but the miss was silent and the widening is one clause.
                return any(
                    isinstance(sub, ast.Call)
                    and (
                        (isinstance(sub.func, ast.Attribute) and sub.func.attr == "main")
                        or (isinstance(sub.func, ast.Name) and sub.func.id == "main")
                    )
                    for sub in ast.walk(node)
                )

            main_line = next(
                (node.lineno for node in tree.body if _is_main_guard(node)), None
            )
            if main_line is None:
                continue
            below = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.lineno > main_line
            ]
            if below:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}: unittest.main() at line {main_line}, "
                    f"but these are defined after it: {', '.join(below)}"
                )

        self.assertEqual(
            offenders,
            [],
            "move `unittest.main()` to the end of the file — anything below it is invisible to "
            "direct execution, and pytest and `python <file>` will report different counts",
        )


class DuplicateTestClassTest(unittest.TestCase):
    """No test module may define the same top-level class twice.

    A redefinition silently shadows the first, so whichever copy is earlier never runs and the two
    can drift apart unnoticed. Found for real: `tests/test_mcts_eval_manifest.py` held two
    copies of `TrimmedEncoderTablesTest` at lines 392 and 578 with identical SHA-256 -- so which
    one Python bound (the second; binding is top-to-bottom) made no behavioural difference, and the
    deletion of the second left the previously-dead first copy as the survivor. Nothing failed,
    pytest collected 39 either way, and the only visible symptom was a class list counting it twice.
    (A byte count stood here and reviewer and author got different figures for it. Reconciled: the
    class body holds three em dashes at 3 bytes each in UTF-8, so 5,010 BYTES is 5,004 CHARACTERS --
    both measurements were right about different units. The digest is kept anyway, being the fact
    the argument rests on and immune to that confusion.)

    The structural guard next door catches classes stranded BELOW `unittest.main()`; this catches
    classes hidden BEHIND another definition. Same failure — a test that looks present and is not.
    """

    def test_no_tracked_test_module_defines_a_class_twice(self) -> None:
        import ast
        from collections import Counter

        offenders: list[str] = []
        for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            counts = Counter(
                node.name for node in tree.body if isinstance(node, ast.ClassDef)
            )
            for name, count in sorted(counts.items()):
                if count > 1:
                    lines = [
                        node.lineno
                        for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == name
                    ]
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}: {name} defined {count}x at lines "
                        f"{', '.join(map(str, lines))}"
                    )

        self.assertEqual(
            offenders,
            [],
            "a redefined class shadows the earlier one, so the earlier copy never runs",
        )


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
