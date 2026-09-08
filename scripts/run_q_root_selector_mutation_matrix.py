#!/usr/bin/env python3
"""Prove the Q root selector's source-level guardrails are live.

This is deliberately a small semantic mutation matrix, not a generic coverage
number.  Each mutation corrupts a behaviour the candidate needs in order to be
an interpretable, one-world Q-selector experiment: it must play the completed
tree's Q arm, fail closed on an incomplete root, use the acting player's value
frame, retain every fixed-budget fence, and reach the actual bridge child.

The runner copies the checked-out source to a temporary directory, changes one
unique source anchor in that copy, and requires focused tests to fail.  It
never changes the source checkout, Kubernetes state, or an existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TEST_COUNT = 10
EVIDENCE_FILES = (
    "scripts/run_q_root_selector_mutation_matrix.py",
    "src/pokezero/engine_search.py",
    "src/pokezero/foulplay_bridge.py",
    "scripts/foulplay_paired_eval.py",
    "tests/test_engine_search.py",
    "tests/test_foulplay_bridge.py",
    "tests/test_foulplay_paired_eval.py",
    "tests/test_run_q_root_selector_mutation_matrix.py",
)
TEST_TARGETS = (
    "tests.test_engine_search.RootDecisionTelemetryTest.test_root_selector_q_plays_the_witnessed_q_arm_from_the_same_tree",
    "tests.test_engine_search.RootDecisionTelemetryTest.test_root_selector_q_refuses_a_partly_unmappable_completed_root",
    "tests.test_engine_search.RootDecisionTelemetryTest.test_root_selector_q_uses_the_acting_seats_q_frame",
    "tests.test_engine_search.RootDecisionTelemetryTest.test_root_selector_q_has_the_same_one_world_full_budget_fence",
    "tests.test_foulplay_paired_eval.RootSelectorQPassthroughTest",
    "tests.test_foulplay_bridge.FoulPlayBridgeTest.test_config_from_args_carries_q_selector_to_the_controlled_bridge",
    "tests.test_foulplay_bridge.FoulPlayBridgeTest.test_build_policy_makes_q_selector_fail_closed",
)


class Mutation(NamedTuple):
    name: str
    relative_path: str
    before: str
    after: str


MUTATIONS = (
    Mutation(
        "play_visit_max_instead_of_completed_tree_q_max",
        "src/pokezero/engine_search.py",
        'action_index = int(comparison["q_action"])',
        'action_index = int(comparison["visit_action"])',
    ),
    Mutation(
        "silently_drop_an_unmappable_visited_root_arm",
        "src/pokezero/engine_search.py",
        'return None, "visited_arm_unmapped"',
        "continue",
    ),
    Mutation(
        "forget_the_p2_q_frame_reflection",
        "src/pokezero/engine_search.py",
        'q_acting = entry["q"] if acting_side_one else 1.0 - entry["q"]',
        'q_acting = entry["q"]',
    ),
    Mutation(
        "allow_q_selection_without_root_arm_telemetry",
        "src/pokezero/engine_search.py",
        'if not self.override_telemetry:\n                raise ValueError(\n                    f"{selector_name} requires override_telemetry=True so its "',
        'if False:\n                raise ValueError(\n                    f"{selector_name} requires override_telemetry=True so its "',
    ),
    Mutation(
        "allow_multiworld_q_selection_without_an_aggregation_rule",
        "src/pokezero/engine_search.py",
        'if self.worlds != 1:\n                raise ValueError(\n                    f"{selector_name} requires worlds=1: the first selector "',
        'if False:\n                raise ValueError(\n                    f"{selector_name} requires worlds=1: the first selector "',
    ),
    Mutation(
        "allow_q_selection_from_an_early_stopped_prefix",
        "src/pokezero/engine_search.py",
        'if self.early_stop:\n                raise ValueError(\n                    f"{selector_name} requires early_stop=False: a visit-lock "',
        'if False:\n                raise ValueError(\n                    f"{selector_name} requires early_stop=False: a visit-lock "',
    ),
    Mutation(
        "allow_q_selection_with_a_dynamic_budget",
        "src/pokezero/engine_search.py",
        'if self.depth_min is not None or self.worlds_min is not None:\n                raise ValueError(\n                    f"{selector_name} requires a fixed search allocation; "',
        'if False:\n                raise ValueError(\n                    f"{selector_name} requires a fixed search allocation; "',
    ),
    Mutation(
        "drop_q_selector_while_building_the_paired_child_argv",
        "scripts/foulplay_paired_eval.py",
        'if args.engine_root_selector_q:\n            argv.append("--engine-root-selector-q")',
        'if False:\n            argv.append("--engine-root-selector-q")',
    ),
    Mutation(
        "drop_q_selector_between_bridge_parser_and_engine_config",
        "src/pokezero/foulplay_bridge.py",
        'engine_root_selector_q=getattr(args, "engine_root_selector_q", False),',
        "engine_root_selector_q=False,",
    ),
    Mutation(
        "drop_q_selector_while_constructing_the_engine_policy",
        "src/pokezero/foulplay_bridge.py",
        "root_selector_q=config.engine_root_selector_q,",
        "root_selector_q=False,",
    ),
)


class MutationError(RuntimeError):
    """A source anchor changed, or a semantic mutant survived the focused tests."""


def _source_commit(override: str | None) -> str:
    """Read the immutable source revision, allowing a source-image override."""
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        if override is None:
            raise MutationError(
                "source commit override is required when the source tree has no Git metadata"
            ) from exc
        if re.fullmatch(r"[0-9a-f]{40}", override) is None:
            raise MutationError("--source-commit must be a 40-character lowercase Git commit")
        return override
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise MutationError("Git did not report a 40-character lowercase source commit")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    if status:
        raise MutationError("source tree is not clean; refusing to misattribute evidence")
    if override is not None and override != value:
        raise MutationError("--source-commit does not match this checked-out Git revision")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, payload: dict[str, object]) -> None:
    """Create an artifact atomically; never replace conflicting evidence."""
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise MutationError(f"refusing to replace a different mutation artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise MutationError(f"concurrent mutation artifact differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _mutated_copy(mutation: Mutation) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temporary = tempfile.TemporaryDirectory(prefix="q-root-selector-mutation-")
    checkout = Path(temporary.name) / "pokezero"
    shutil.copytree(
        ROOT,
        checkout,
        ignore=shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__", ".venv"),
    )
    target = checkout / mutation.relative_path
    source = target.read_text(encoding="utf-8")
    if source.count(mutation.before) != 1:
        temporary.cleanup()
        raise MutationError(f"mutation anchor is not unique for {mutation.name}")
    target.write_text(source.replace(mutation.before, mutation.after), encoding="utf-8")
    return temporary, checkout


def _run_focused_tests(checkout: Path) -> subprocess.CompletedProcess[str]:
    """Run tests against the copied ``src`` tree, never an installed package."""
    environment = dict(os.environ)
    source_path = str(checkout / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not inherited else source_path + os.pathsep + inherited
    )
    return subprocess.run(
        [sys.executable, "-m", "unittest", "-q", *TEST_TARGETS],
        cwd=checkout,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _validated_test_run(
    completed: subprocess.CompletedProcess[str], *, clean_baseline: bool
) -> dict[str, object]:
    """Refuse a harness/runtime failure from masquerading as a killed mutant.

    A nonzero process status is not itself mutation evidence: an import error,
    a missing dependency, or a test-selection typo would also produce one.  The
    original source must first pass the exact target count without skips.  Each
    mutant must then execute that same complete set and end in unittest's normal
    ``FAILED (...)`` summary.  Import, syntax, and partial-run failures are
    deliberately evidence failures, not kills.
    """
    output = completed.stdout
    match = re.search(r"^Ran (\d+) tests? in ", output, flags=re.MULTILINE)
    if match is None or int(match.group(1)) != EXPECTED_TEST_COUNT:
        found = "none" if match is None else match.group(1)
        raise MutationError(
            f"focused tests did not execute exactly {EXPECTED_TEST_COUNT} targets "
            f"(reported {found})"
        )
    if "skipped=" in output:
        raise MutationError("focused tests skipped a target; skipped evidence is not a kill")
    if any(marker in output for marker in ("ImportError", "ModuleNotFoundError", "SyntaxError")):
        raise MutationError("focused tests had an import or syntax failure, not a semantic kill")
    if clean_baseline:
        if completed.returncode != 0 or re.search(r"^OK$", output, flags=re.MULTILINE) is None:
            raise MutationError("unmutated focused-test baseline is not clean")
        status = "CLEAN"
    else:
        if completed.returncode == 0:
            raise MutationError("focused tests passed against a semantic mutant")
        if re.search(r"^FAILED ", output, flags=re.MULTILINE) is None:
            raise MutationError("mutant did not end in a unittest failure summary")
        status = "KILLED"
    return {
        "status": status,
        "exit_code": completed.returncode,
        "tests_run": EXPECTED_TEST_COUNT,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def run(*, source_commit: str | None = None) -> dict[str, object]:
    """Kill every declared selector mutant and return a provenance-bound receipt."""
    baseline = _validated_test_run(_run_focused_tests(ROOT), clean_baseline=True)
    results: list[dict[str, object]] = []
    for mutation in MUTATIONS:
        temporary, checkout = _mutated_copy(mutation)
        try:
            completed = _run_focused_tests(checkout)
            result = _validated_test_run(completed, clean_baseline=False)
            results.append({"name": mutation.name, **result})
        finally:
            temporary.cleanup()
    return {
        "schema_version": "pokezero.q-root-selector-mutation-battery.v1",
        "complete": True,
        "source_commit": _source_commit(source_commit),
        "source_files_sha256": {
            path: _sha256(ROOT / path) for path in EVIDENCE_FILES
        },
        "test_targets": list(TEST_TARGETS),
        "baseline": baseline,
        "mutations": results,
        "all_killed": len(results) == len(MUTATIONS)
        and all(item["status"] == "KILLED" for item in results),
        "is_b2_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit")
    args = parser.parse_args(argv)
    try:
        result = run(source_commit=args.source_commit)
        if result["all_killed"] is not True:
            raise MutationError("targeted Q-selector mutation battery is incomplete")
        _write_once(args.out, result)
    except (MutationError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
