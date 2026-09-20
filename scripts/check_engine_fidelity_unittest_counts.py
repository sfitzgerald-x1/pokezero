#!/usr/bin/env python3
"""Verify the engine-fidelity workflow's exact unittest-count guards locally.

The native fidelity workflow deliberately treats both a shrinking and a growing
test suite as a failure.  That is a useful anti-vacuity property, but its
``Ran N tests`` values live in YAML, after an expensive native build.  Importing
every guarded module locally is not an answer: several intentionally import the
native extension at module load time.

This verifier instead reads the workflow and counts unittest methods directly
from the test modules' ASTs.  It is deliberately limited to the workflow's
explicit ``python -m unittest`` invocations with an exact ``Ran N tests``
guard.  A new guard it cannot parse is a failure, not a silent omission.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "engine-fidelity-gates.yml"
TESTS = ROOT / "tests"


@dataclass(frozen=True)
class CountGuard:
    step: str
    expected: int
    targets: tuple[str, ...]


def _steps(workflow: str) -> list[tuple[str, str]]:
    """Return each named workflow step and its YAML body without parsing YAML."""

    parts = re.split(r"^      - name: ", workflow, flags=re.MULTILINE)
    result: list[tuple[str, str]] = []
    for part in parts[1:]:
        name, _, body = part.partition("\n")
        result.append((name, body))
    return result


def _unittest_command(body: str) -> str | None:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if "python -m unittest" not in line or line.lstrip().startswith("#"):
            continue
        command = [line]
        while "2>&1" not in command[-1]:
            index += 1
            if index >= len(lines):
                raise ValueError("unterminated python -m unittest command")
            command.append(lines[index])
        return "\n".join(command)
    return None


def _expected_count(body: str) -> int | None:
    for line in body.splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = re.search(r"Ran (\d+) tests", line)
        if match and "grep" in line:
            return int(match.group(1))
    return None


def _guards() -> list[CountGuard]:
    guards: list[CountGuard] = []
    for step, body in _steps(WORKFLOW.read_text()):
        expected = _expected_count(body)
        command = _unittest_command(body)
        if expected is None and command is None:
            continue
        if expected is None or command is None:
            raise ValueError(
                f"{step!r} has only one of an unittest command and exact Ran N guard"
            )
        targets = tuple(re.findall(r"\btests\.[A-Za-z_]\w*(?:\.\w+)*", command))
        if not targets:
            raise ValueError(f"{step!r} has no parseable unittest targets")
        guards.append(CountGuard(step=step, expected=expected, targets=targets))
    if not guards:
        raise ValueError("found no exact unittest-count guards")
    return guards


def _test_count(target: str) -> int:
    """Count an explicit unittest module or selected test method from its AST."""

    pieces = target.split(".")
    if len(pieces) < 2 or pieces[0] != "tests":
        raise ValueError(f"unsupported unittest target {target!r}")
    module = pieces[1]
    path = TESTS / f"{module}.py"
    if not path.is_file():
        raise ValueError(f"test target {target!r} has no module at {path.relative_to(ROOT)}")
    tree = ast.parse(path.read_text(), filename=str(path))

    if len(pieces) == 2:
        return sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(tree)
        )

    selected = pieces[-1]
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == selected
    ]
    if len(matches) != 1:
        raise ValueError(
            f"selected target {target!r} matched {len(matches)} test methods in "
            f"{path.relative_to(ROOT)}"
        )
    if not selected.startswith("test_"):
        raise ValueError(f"selected target {target!r} is not a unittest test method")
    return 1


def main() -> int:
    failed = False
    guards = _guards()
    for guard in guards:
        actual = sum(_test_count(target) for target in guard.targets)
        if actual != guard.expected:
            failed = True
            print(
                f"ERROR: {guard.step}: workflow expects {guard.expected} tests, "
                f"but source defines {actual} ({', '.join(guard.targets)})",
                file=sys.stderr,
            )
    if failed:
        return 1
    print(f"Verified {len(guards)} engine-fidelity unittest count guards against test source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
