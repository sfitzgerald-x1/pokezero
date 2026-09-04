#!/usr/bin/env python3
"""Execute targeted, source-level mutation evidence for the R42 selector.

This is deliberately not a generic coverage metric.  Each mutant is one of
the behaviors whose regression would make an R42 positive result misleading:
raw-tie preservation, subject-relative terminal-loss detection, safe-tie
selection, or receipt validation of the previous lowest-index rule.

The runner copies the checked-out source to a temporary directory, mutates one
file there, and requires the focused test command to fail.  It never changes
the source checkout, a Kubernetes object, or a durable shared artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple, Sequence
import re


ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = (
    "tests.test_live_foulplay_continuation",
    "tests.test_live_foulplay_continuation_oracle_b2_eval",
)


class Mutation(NamedTuple):
    name: str
    relative_path: str
    before: str
    after: str


MUTATIONS = (
    Mutation(
        "restore_lowest_index_tie_break_instead_of_raw_preservation",
        "src/pokezero/live_foulplay_continuation.py",
        "if raw_candidate is not None:\n        return raw_candidate",
        "if False:\n        return raw_candidate",
    ),
    Mutation(
        "keep_raw_immediate_loss_despite_safe_tie",
        "src/pokezero/live_foulplay_continuation.py",
        "eligible = non_losing_tied or tied",
        "eligible = tied",
    ),
    Mutation(
        "invert_subject_relative_immediate_loss_orientation",
        "src/pokezero/live_foulplay_continuation.py",
        "terminal.get(\"winner\") == foulplay_player",
        "terminal.get(\"winner\") != foulplay_player",
    ),
    Mutation(
        "validator_accepts_the_old_lowest_index_tie_rule",
        "scripts/live_foulplay_continuation_oracle_b2_eval.py",
        "return int(raw_candidate[\"action_index\"] if raw_candidate is not None else\n               min(eligible, key=lambda candidate: int(candidate[\"action_index\"]))[\"action_index\"])",
        "return int(min(tied, key=lambda candidate: int(candidate[\"action_index\"]))[\"action_index\"])",
    ),
)


class MutationError(RuntimeError):
    """The targeted source mutation battery is incomplete or a mutant survived."""


def _source_commit(override: str | None) -> str:
    """Return the frozen source commit without requiring a runtime ``.git`` tree.

    The source image intentionally need not contain Git metadata.  The image
    builder receipt already carries the immutable commit, so a provenance-bound
    preflight passes that value explicitly while local developer invocation can
    retain the convenient checked-out default.
    """

    if override is not None:
        if re.fullmatch(r"[0-9a-f]{40}", override) is None:
            raise MutationError("--source-commit must be a 40-character lowercase Git commit")
        return override
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise MutationError("source commit override is required when the source tree has no Git metadata") from exc
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise MutationError("Git did not report a 40-character lowercase source commit")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, payload: dict[str, object]) -> None:
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
    temporary = tempfile.TemporaryDirectory(prefix="r42-selector-mutation-")
    checkout = Path(temporary.name) / "pokezero"
    shutil.copytree(ROOT, checkout, ignore=shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__", ".venv"))
    target = checkout / mutation.relative_path
    source = target.read_text(encoding="utf-8")
    if source.count(mutation.before) != 1:
        temporary.cleanup()
        raise MutationError(f"mutation anchor is not unique for {mutation.name}")
    target.write_text(source.replace(mutation.before, mutation.after), encoding="utf-8")
    return temporary, checkout


def _run_focused_tests(checkout: Path) -> subprocess.CompletedProcess[str]:
    """Run the source-level tests against precisely ``checkout``.

    The R42 source image deliberately avoids carrying a test-only dependency
    such as pytest.  More importantly, merely changing the current directory
    is not sufficient for a ``src/`` package: an installed base-image
    ``pokezero`` could otherwise be imported instead of the deliberately
    mutated copy.  A fresh interpreter with the copied ``src`` prepended to
    ``PYTHONPATH`` establishes both facts before any mutant can be credited as
    killed.
    """

    environment = dict(os.environ)
    source_path = str(checkout / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not inherited else source_path + os.pathsep + inherited
    )
    return subprocess.run(
        [sys.executable, "-m", "unittest", "-q", *TEST_MODULES],
        cwd=checkout,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def run(*, source_commit: str | None = None) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for mutation in MUTATIONS:
        temporary, checkout = _mutated_copy(mutation)
        try:
            completed = _run_focused_tests(checkout)
            if completed.returncode == 0:
                raise MutationError(f"SURVIVED {mutation.name}: focused tests passed")
            results.append({"name": mutation.name, "status": "KILLED", "exit_code": completed.returncode,
                            "output_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest()})
        finally:
            temporary.cleanup()
    return {
        "schema_version": "pokezero.r42-selector-mutation-battery.v1",
        "complete": True,
        "source_commit": _source_commit(source_commit),
        "source_files_sha256": {
            "src/pokezero/live_foulplay_continuation.py": _sha256(ROOT / "src/pokezero/live_foulplay_continuation.py"),
            "scripts/live_foulplay_continuation_oracle_b2_eval.py": _sha256(ROOT / "scripts/live_foulplay_continuation_oracle_b2_eval.py"),
        },
        "mutations": results,
        "all_killed": len(results) == len(MUTATIONS) and all(item["status"] == "KILLED" for item in results),
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
            raise MutationError("targeted R42 mutation battery is incomplete")
        _write_once(args.out, result)
    except (MutationError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
