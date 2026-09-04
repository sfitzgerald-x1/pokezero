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


ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "tests/test_live_foulplay_continuation.py",
    "tests/test_live_foulplay_continuation_oracle_b2_eval.py",
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


def run() -> dict[str, object]:
    results: list[dict[str, object]] = []
    for mutation in MUTATIONS:
        temporary, checkout = _mutated_copy(mutation)
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *TESTS], cwd=checkout,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            if completed.returncode == 0:
                raise MutationError(f"SURVIVED {mutation.name}: focused tests passed")
            results.append({"name": mutation.name, "status": "KILLED", "exit_code": completed.returncode,
                            "output_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest()})
        finally:
            temporary.cleanup()
    return {
        "schema_version": "pokezero.r42-selector-mutation-battery.v1",
        "complete": True,
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stdout=subprocess.PIPE, check=True).stdout.strip(),
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
    args = parser.parse_args(argv)
    try:
        result = run()
        if result["all_killed"] is not True:
            raise MutationError("targeted R42 mutation battery is incomplete")
        _write_once(args.out, result)
    except (MutationError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
