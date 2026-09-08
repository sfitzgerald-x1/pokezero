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


ROOT = Path(__file__).resolve().parents[1]
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
    if override is not None:
        if re.fullmatch(r"[0-9a-f]{40}", override) is None:
            raise MutationError("--source-commit must be a 40-character lowercase Git commit")
        return override
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
        raise MutationError(
            "source commit override is required when the source tree has no Git metadata"
        ) from exc
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise MutationError("Git did not report a 40-character lowercase source commit")
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
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    try:
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


def run(*, source_commit: str | None = None) -> dict[str, object]:
    """Kill every declared selector mutant and return a provenance-bound receipt."""
    results: list[dict[str, object]] = []
    for mutation in MUTATIONS:
        temporary, checkout = _mutated_copy(mutation)
        try:
            completed = _run_focused_tests(checkout)
            if completed.returncode == 0:
                raise MutationError(f"SURVIVED {mutation.name}: focused tests passed")
            results.append(
                {
                    "name": mutation.name,
                    "status": "KILLED",
                    "exit_code": completed.returncode,
                    "output_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                }
            )
        finally:
            temporary.cleanup()
    return {
        "schema_version": "pokezero.q-root-selector-mutation-battery.v1",
        "complete": True,
        "source_commit": _source_commit(source_commit),
        "source_files_sha256": {
            path: _sha256(ROOT / path)
            for path in sorted({mutation.relative_path for mutation in MUTATIONS})
        },
        "test_targets": list(TEST_TARGETS),
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
