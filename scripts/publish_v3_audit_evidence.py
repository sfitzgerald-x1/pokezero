#!/usr/bin/env python3
"""Publish a vetted, public-safe summary of a completed v3 audit cycle.

The cluster aggregates intentionally retain operational paths and the fully
qualified image reference needed to resume work.  Those details do not belong
in the public evidence ledger.  This tool verifies the terminal aggregates and
their constituent audit artifacts, then copies only a small whitelist of
reproducibility fields into a tracked JSON file.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


PUBLIC_SCHEMA_VERSION = "pokezero.v3-audit-public-evidence.v2"
OBSERVATION_SCHEMA = "pokezero.observation.v3"
PROTOCOL_SIGNATURE_SCHEMA = "pokezero.protocol-signature-census.v2"
SILENT_MUTATION_AUDIT_SCHEMA = "pokezero.silent-mutation-audit.v1"
COLLISION_AUDIT_SCHEMA = "pokezero.encoding-collision-audit.v1"
COLLISION_MODEL_INPUT_NUMERIC_DTYPE = "float32"
COVERAGE_AUDIT_SCHEMA = "pokezero.deep-line-audit.v1"
REQUIRED_BOUNDED_DEPTH_ROUNDS = 8
MINIMUM_SILENT_RANDOM_GAMES = 8
MINIMUM_SILENT_MAX_ROUNDS = 120
REQUIRED_COLLISION_RECORDS = 100_000
REQUIRED_PROVENANCE_KEYS = (
    "public_repo_commit",
    "showdown_source_hash",
    "observation_schema",
    "image_digest",
)
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SOURCE_HASH_RE = re.compile(r"[0-9a-f]{8,64}")
_PUBLIC_AUDIT_SCRIPTS = frozenset(
    {
        "coverage_enumeration_audit.py",
        "deep_line_audit.py",
        "silent_mutation_audit.py",
        "encoding_collision_audit.py",
        "protocol_emission_inventory.py",
    }
)
_PATH_FLAGS = frozenset(
    {
        "--collision-sketch",
        "--corpus",
        "--coverage-json",
        "--failure-dir",
        "--json",
        "--observed-audit",
        "--out",
        "--showdown-root",
    }
)
_NUMERIC_FLAGS = frozenset(
    {
        "--depth-rounds",
        "--max-decisions",
        "--max-games",
        "--max-move-rounds",
        "--max-rounds",
        "--random-games",
        "--seed-start",
        "--start-decision",
    }
)
_LITERAL_FLAG_VALUES = {
    "--capture-driver": frozenset({"foul-play", "random-legal"}),
    "--observation-schema": frozenset({"v3"}),
}
_BOOLEAN_FLAGS = frozenset(
    {
        "--exact-variants",
        "--gap-fill",
        "--interaction-registry",
        "--no-gap-fill",
        "--no-universal-lane",
        "--protocol-fixtures",
        "--scenarios",
        "--universal-lane",
        "--use-moves",
    }
)
_SHARD_RE = re.compile(r"[0-9]+/[1-9][0-9]*")
_PUBLIC_ATOM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}")
_COVERAGE_UNCOVERED_KEYS = frozenset({"species", "ability_pairs", "moves", "items", "variants"})


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing required audit artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"audit artifact must be a JSON object: {path}")
    return value


def _require_schema(payload: Mapping[str, Any], expected: str, *, label: str) -> None:
    if payload.get("schema_version") != expected:
        raise ValueError(
            f"{label} has schema {payload.get('schema_version')!r}; expected {expected!r}"
        )


def _require_terminal_status(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("status") not in {"clean", "needs-triage"}:
        raise ValueError(f"{label} has non-terminal audit status {payload.get('status')!r}")


def _require_derived_status(*, status: object, clean: bool, label: str) -> str:
    """Prevent a terminal marker from calling nonzero evidence ``clean``."""

    _require_terminal_status({"status": status}, label=label)
    expected = "clean" if clean else "needs-triage"
    if status != expected:
        raise ValueError(f"{label} status {status!r} disagrees with its validated counters")
    return expected


def _immutable_digest(value: object, *, label: str) -> str:
    matches = _DIGEST_RE.findall(str(value))
    if len(matches) != 1:
        raise ValueError(f"{label} must contain exactly one immutable sha256 image digest")
    return matches[0]


def _public_command(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a non-empty command list")
    script = Path(value[0]).name
    if script not in _PUBLIC_AUDIT_SCRIPTS:
        raise ValueError(f"{label} does not name a public audit script")
    command = [f"scripts/{script}"]
    index = 1
    while index < len(value):
        token = value[index]
        if not token.startswith("--"):
            raise ValueError(f"{label} has an unrecognized positional argument")
        if token in _BOOLEAN_FLAGS:
            command.append(token)
            index += 1
            continue
        if token in _PATH_FLAGS:
            if index + 1 >= len(value):
                raise ValueError(f"{label} has a path flag without a value: {token}")
            command.extend((token, "<artifact-path>"))
            index += 2
            continue
        if index + 1 >= len(value):
            raise ValueError(f"{label} has a value flag without a value: {token}")
        argument = value[index + 1]
        if token in _NUMERIC_FLAGS:
            _require_nonnegative_int_text(argument, label=f"{label} {token}")
        elif token == "--shard":
            if _SHARD_RE.fullmatch(argument) is None:
                raise ValueError(f"{label} has an invalid shard argument")
        elif token in _LITERAL_FLAG_VALUES:
            if argument not in _LITERAL_FLAG_VALUES[token]:
                raise ValueError(f"{label} has an invalid value for {token}")
        elif token == "--pass":
            if argument not in {"A", "B", "both"}:
                raise ValueError(f"{label} has an invalid value for --pass")
        elif token in {"--scenario", "--suppress-kind"}:
            if _PUBLIC_ATOM_RE.fullmatch(argument) is None:
                raise ValueError(f"{label} has an unsafe value for {token}")
        else:
            raise ValueError(f"{label} has an unrecognized flag: {token}")
        command.extend((token, argument))
        index += 2
    return command


def _public_provenance(value: object, *, label: str) -> dict[str, Any]:
    result = _validated_provenance_identity(value, label=label)
    if not isinstance(value, Mapping):  # Kept for static type narrowing after validation.
        raise AssertionError("validated provenance must be a mapping")
    result["command"] = _public_command(value.get("command"), label=label)
    result["execution_scope"] = _public_execution_scope(value.get("execution_scope"), label=label)
    _require_execution_scope_for_script(
        command=result["command"], scope=result["execution_scope"], label=label
    )
    if "seed_range" in value:
        result["seed_range"] = _public_seed_range(value["seed_range"], label=label)
    if "shard" in value:
        result["shard"] = _public_shard(value["shard"], label=label)
    return result


def _require_execution_scope_for_script(
    *, command: Sequence[str], scope: Mapping[str, Any], label: str
) -> None:
    """Require the range/configuration dimensions that define each audit lane."""

    required_fields = {
        "scripts/coverage_enumeration_audit.py": {"seed_range", "shard"},
        "scripts/deep_line_audit.py": {"seed_range", "max_rounds", "scenario_names", "protocol_fixtures"},
        "scripts/silent_mutation_audit.py": {"seed_range", "max_rounds", "scenario_names"},
        "scripts/encoding_collision_audit.py": {"decision_range", "input_kind", "input_artifact_count"},
        "scripts/protocol_emission_inventory.py": {"input_audit_count", "seed_range", "shard"},
    }
    missing = sorted(required_fields[command[0]] - set(scope))
    if missing:
        raise ValueError(f"{label} execution scope is missing required fields: {', '.join(missing)}")


def _public_execution_scope(value: object, *, label: str) -> dict[str, Any]:
    """Copy only reproducibility metadata that is safe for the public ledger."""

    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} is missing execution scope")
    allowed = {
        "seed_range",
        "shard",
        "max_rounds",
        "max_decision_rounds",
        "capture_driver",
        "scenario_names",
        "protocol_fixtures",
        "decision_range",
        "input_kind",
        "input_artifact_count",
        "input_audit_count",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} execution scope has unsupported fields: {', '.join(unknown)}")
    result: dict[str, Any] = {}
    if "seed_range" in value:
        result["seed_range"] = _public_execution_seed_range(value["seed_range"], label=label)
    if "shard" in value:
        result["shard"] = _public_shard(value["shard"], label=label) if value["shard"] is not None else None
    for key in ("max_rounds", "max_decision_rounds", "input_artifact_count", "input_audit_count"):
        if key in value:
            result[key] = _require_nonnegative_int(value[key], label=f"{label} execution scope {key}")
    if "capture_driver" in value:
        if value["capture_driver"] not in {"checkpoint", "random-legal"}:
            raise ValueError(f"{label} execution scope has invalid capture driver")
        result["capture_driver"] = value["capture_driver"]
    if "scenario_names" in value:
        names = value["scenario_names"]
        if not isinstance(names, list) or not all(isinstance(name, str) and _PUBLIC_ATOM_RE.fullmatch(name) for name in names):
            raise ValueError(f"{label} execution scope has invalid scenario names")
        result["scenario_names"] = list(names)
    if "protocol_fixtures" in value:
        if not isinstance(value["protocol_fixtures"], bool):
            raise ValueError(f"{label} execution scope protocol_fixtures must be boolean")
        result["protocol_fixtures"] = value["protocol_fixtures"]
    if "decision_range" in value:
        decision_range = value["decision_range"]
        if not isinstance(decision_range, Mapping) or set(decision_range) != {"start", "limit", "end_exclusive"}:
            raise ValueError(f"{label} execution scope has invalid decision range")
        start = _require_nonnegative_int(decision_range.get("start"), label=f"{label} decision start")
        limit = _require_nonnegative_int(decision_range.get("limit"), label=f"{label} decision limit")
        end = _require_nonnegative_int(decision_range.get("end_exclusive"), label=f"{label} decision end")
        if end != start + limit:
            raise ValueError(f"{label} execution scope has inconsistent decision range")
        result["decision_range"] = {"start": start, "limit": limit, "end_exclusive": end}
    if "input_kind" in value:
        if value["input_kind"] not in {"corpus", "collision-sketch"}:
            raise ValueError(f"{label} execution scope has invalid input kind")
        result["input_kind"] = value["input_kind"]
    return result


def _validated_provenance_identity(value: object, *, label: str) -> dict[str, Any]:
    """Validate a provenance tuple without copying command or path-bearing fields."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing audit_provenance")
    missing = [key for key in REQUIRED_PROVENANCE_KEYS if not value.get(key)]
    if missing:
        raise ValueError(f"{label} has incomplete audit provenance: {', '.join(missing)}")
    if value.get("observation_schema") != OBSERVATION_SCHEMA:
        raise ValueError(f"{label} did not use {OBSERVATION_SCHEMA}")
    recorded_at = value.get("recorded_at")
    if not isinstance(recorded_at, str) or not recorded_at:
        raise ValueError(f"{label} is missing provenance timestamp")
    try:
        datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} has an invalid provenance timestamp") from exc
    commit = str(value["public_repo_commit"])
    source_hash = str(value["showdown_source_hash"])
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError(f"{label} has an invalid public repository commit")
    if _SOURCE_HASH_RE.fullmatch(source_hash) is None:
        raise ValueError(f"{label} has an invalid Showdown source hash")
    return {
        "public_repo_commit": commit,
        "showdown_source_hash": source_hash,
        "observation_schema": OBSERVATION_SCHEMA,
        "image_digest": _immutable_digest(value["image_digest"], label=label),
        "recorded_at": recorded_at,
    }


def _validated_aggregate_identity(value: object, *, label: str) -> dict[str, Any]:
    """Validate the canonical four-field identity emitted by a ledger merger."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing aggregate audit provenance")
    missing = [key for key in REQUIRED_PROVENANCE_KEYS if not value.get(key)]
    if missing:
        raise ValueError(f"{label} has incomplete aggregate provenance: {', '.join(missing)}")
    if value.get("observation_schema") != OBSERVATION_SCHEMA:
        raise ValueError(f"{label} did not use {OBSERVATION_SCHEMA}")
    commit = str(value["public_repo_commit"])
    source_hash = str(value["showdown_source_hash"])
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError(f"{label} has an invalid public repository commit")
    if _SOURCE_HASH_RE.fullmatch(source_hash) is None:
        raise ValueError(f"{label} has an invalid Showdown source hash")
    return {
        "public_repo_commit": commit,
        "showdown_source_hash": source_hash,
        "observation_schema": OBSERVATION_SCHEMA,
        "image_digest": _immutable_digest(value["image_digest"], label=label),
    }


def _require_matching_provenance(entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    if not entries:
        raise ValueError("no audit provenance entries were supplied")
    first = entries[0]
    result = {key: str(first[key]) for key in REQUIRED_PROVENANCE_KEYS}
    for entry in entries[1:]:
        mismatch = [key for key in REQUIRED_PROVENANCE_KEYS if str(entry[key]) != result[key]]
        if mismatch:
            raise ValueError("mixed audit provenance across layers: " + ", ".join(mismatch))
    return result


def _require_protocol_signature_schema(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("protocol_signature_schema_version") != PROTOCOL_SIGNATURE_SCHEMA:
        raise ValueError(f"{label} is missing {PROTOCOL_SIGNATURE_SCHEMA}")


def _public_seed_range(value: object, *, label: str) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"start", "end"}:
        raise ValueError(f"{label} has an invalid seed range")
    start = _require_nonnegative_int(value.get("start"), label=f"{label} seed start")
    end = _require_nonnegative_int(value.get("end"), label=f"{label} seed end")
    if end < start:
        raise ValueError(f"{label} seed range ends before it starts")
    return {"start": start, "end": end}


def _public_execution_seed_range(value: object, *, label: str) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) not in ({"start", "end"}, {"start", "end", "count"}):
        raise ValueError(f"{label} execution scope has an invalid seed range")
    start = _require_nonnegative_int(value.get("start"), label=f"{label} seed start")
    end = _require_nonnegative_int(value.get("end"), label=f"{label} seed end")
    if end < start:
        raise ValueError(f"{label} seed range ends before it starts")
    result: dict[str, int] = {"start": start, "end": end}
    if "count" in value:
        count = _require_nonnegative_int(value.get("count"), label=f"{label} seed count")
        if count != result["end"] - result["start"] + 1:
            raise ValueError(f"{label} execution scope has inconsistent seed range")
        result["count"] = count
    return result


def _public_shard(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"index", "count"}:
        raise ValueError(f"{label} has an invalid shard descriptor")
    index = _require_nonnegative_int(value.get("index"), label=f"{label} shard index")
    count = _require_nonnegative_int(value.get("count"), label=f"{label} shard count")
    if count < 1 or index >= count:
        raise ValueError(f"{label} has an out-of-range shard descriptor")
    return {"index": index, "count": count}


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_nonnegative_int_text(value: str, *, label: str) -> None:
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"{label} must be a non-negative integer")


def _require_command_script(command: Sequence[str], script: str, *, label: str) -> None:
    if not command or command[0] != f"scripts/{script}":
        raise ValueError(f"{label} did not run scripts/{script}")


def _require_command_flag(command: Sequence[str], flag: str, *, label: str) -> None:
    if flag not in command:
        raise ValueError(f"{label} is missing required {flag}")


def _require_command_value(command: Sequence[str], flag: str, *, label: str) -> str:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"{label} has no unambiguous {flag} value")
    return command[positions[0] + 1]


def _require_command_int_at_least(command: Sequence[str], flag: str, minimum: int, *, label: str) -> int:
    value = _require_command_value(command, flag, label=label)
    _require_nonnegative_int_text(value, label=f"{label} {flag}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{label} {flag} must be at least {minimum}")
    return parsed


def _public_uncovered(value: object, *, label: str) -> dict[str, list[str]]:
    if not isinstance(value, Mapping) or set(value) != _COVERAGE_UNCOVERED_KEYS:
        raise ValueError(f"{label} has an invalid uncovered-atom ledger")
    public: dict[str, list[str]] = {}
    for key in sorted(_COVERAGE_UNCOVERED_KEYS):
        atoms = value[key]
        if not isinstance(atoms, list) or not all(
            isinstance(atom, str) and _PUBLIC_ATOM_RE.fullmatch(atom) for atom in atoms
        ):
            raise ValueError(f"{label} has unsafe uncovered atoms for {key}")
        public[key] = list(atoms)
    return public


def _coverage_stage(root: Path, stage: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _read_json(root / stage / "summary.json")
    if stage == "party":
        _require_schema(summary, "pokezero.coverage-audit-party.v1", label="party summary")
        if summary.get("status") != "complete":
            raise ValueError("party summary is not complete")
        audits = [_read_json(root / stage / "audit.json")]
    else:
        _require_schema(summary, "pokezero.coverage-audit-stage.v1", label=f"{stage} summary")
        if summary.get("stage") != stage or summary.get("status") != "complete":
            raise ValueError(f"{stage} summary is not a completed {stage} stage")
        shard_count = summary.get("shards")
        if not isinstance(shard_count, int) or shard_count < 1:
            raise ValueError(f"{stage} summary has no valid shard count")
        audits = [_read_json(root / stage / "shards" / f"shard-{index:02d}" / "audit.json") for index in range(shard_count)]
    public_runs: list[dict[str, Any]] = []
    for index, audit in enumerate(audits):
        _require_schema(audit, COVERAGE_AUDIT_SCHEMA, label=f"{stage} audit {index}")
        _require_protocol_signature_schema(audit, label=f"{stage} audit {index}")
        public_runs.append(_public_provenance(audit.get("audit_provenance"), label=f"{stage} audit {index}"))
    summary_provenance = _public_provenance(summary.get("audit_provenance"), label=f"{stage} summary")
    _require_matching_provenance([summary_provenance, *public_runs])
    audit_finding_count = sum(
        _require_nonnegative_int(audit.get("finding_count"), label=f"{stage} audit finding_count")
        for audit in audits
    )
    audit_decisions_checked = sum(
        _require_nonnegative_int(audit.get("decisions_checked"), label=f"{stage} audit decisions_checked")
        for audit in audits
    )
    if stage == "party":
        for index, run in enumerate(public_runs):
            command = run["command"]
            label = f"party audit {index}"
            _require_command_script(command, "deep_line_audit.py", label=label)
            for flag in ("--scenarios", "--interaction-registry", "--protocol-fixtures"):
                _require_command_flag(command, flag, label=label)
            if _require_command_value(command, "--observation-schema", label=label) != "v3":
                raise ValueError(f"{label} did not use the v3 observation schema")
            if _require_command_int_at_least(command, "--random-games", 0, label=label) != 0:
                raise ValueError(f"{label} must use curated fixtures without random games")
        if audit_decisions_checked == 0:
            raise ValueError("party audit checked no decision boundaries")
        audit_failure_artifact_count = 0
    else:
        expected_depth = 0 if stage == "static" else REQUIRED_BOUNDED_DEPTH_ROUNDS
        audit_failure_artifact_count = 0
        for index, audit in enumerate(audits):
            command = public_runs[index]["command"]
            label = f"{stage} audit {index}"
            _require_command_script(command, "coverage_enumeration_audit.py", label=label)
            for flag in ("--exact-variants", "--no-universal-lane"):
                _require_command_flag(command, flag, label=label)
            if _require_command_value(command, "--observation-schema", label=label) != "v3":
                raise ValueError(f"{label} did not use the v3 observation schema")
            if stage == "static":
                if "--depth-rounds" in command:
                    raise ValueError(f"{label} unexpectedly ran a depth script")
            elif _require_command_int_at_least(
                command, "--depth-rounds", REQUIRED_BOUNDED_DEPTH_ROUNDS, label=label
            ) != REQUIRED_BOUNDED_DEPTH_ROUNDS:
                raise ValueError(f"{label} did not run the required bounded-depth scope")
            execution = audit.get("coverage_execution")
            if not isinstance(execution, Mapping):
                raise ValueError(f"{stage} audit {index} has no coverage execution ledger")
            if _require_nonnegative_int(
                execution.get("depth_rounds"), label=f"{stage} audit {index} depth_rounds"
            ) != expected_depth:
                raise ValueError(f"{stage} audit {index} depth does not match its stage")
            audit_failure_artifact_count += _require_nonnegative_int(
                execution.get("failure_artifact_count"),
                label=f"{stage} audit {index} failure_artifact_count",
            )
    summary_finding_count = _require_nonnegative_int(summary.get("finding_count"), label=f"{stage} finding_count")
    summary_decisions_checked = _require_nonnegative_int(
        summary.get("decisions_checked"), label=f"{stage} decisions_checked"
    )
    if stage == "party":
        summary_failure_artifact_count = 0
    else:
        summary_failure_artifact_count = _require_nonnegative_int(
            summary.get("failure_artifact_count"), label=f"{stage} failure_artifact_count"
        )
    if (
        (summary_finding_count, summary_decisions_checked, summary_failure_artifact_count)
        != (audit_finding_count, audit_decisions_checked, audit_failure_artifact_count)
    ):
        raise ValueError(f"{stage} summary differs from its constituent audit artifacts")
    if stage == "party":
        # Party fixtures use a curated registry rather than a source-atom ledger.
        coverage_complete = True
        uncovered: dict[str, list[str]] = {}
    else:
        coverage_complete = summary.get("coverage_complete")
        if not isinstance(coverage_complete, bool):
            raise ValueError(f"{stage} summary has no boolean coverage_complete")
        uncovered = _public_uncovered(summary.get("uncovered"), label=f"{stage} summary")
        merged_ledger = _read_json(root / stage / "ledger-merged.json")
        merged_provenance = _validated_aggregate_identity(
            merged_ledger.get("audit_provenance"), label=f"{stage} merged coverage ledger"
        )
        _require_matching_provenance([summary_provenance, merged_provenance, *public_runs])
        ledger_complete = merged_ledger.get("complete")
        if not isinstance(ledger_complete, bool):
            raise ValueError(f"{stage} merged coverage ledger has no boolean completion verdict")
        ledger_uncovered = _public_uncovered(
            merged_ledger.get("uncovered"), label=f"{stage} merged coverage ledger"
        )
        if (coverage_complete, uncovered) != (ledger_complete, ledger_uncovered):
            raise ValueError(f"{stage} summary differs from its merged coverage ledger")
    public = {
        "status": summary.get("status"),
        "finding_count": summary_finding_count,
        "decisions_checked": summary_decisions_checked,
        "coverage_complete": coverage_complete,
        "failure_artifact_count": summary_failure_artifact_count,
        "uncovered": uncovered,
        "command_runs": public_runs,
    }
    return public, [summary_provenance, *public_runs]


def _coverage(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    complete = _read_json(root / "complete.json")
    _require_schema(complete, "pokezero.coverage-audit-job.v1", label="coverage terminal aggregate")
    _require_terminal_status(complete, label="coverage terminal aggregate")
    if complete.get("terminal_stage") != "full":
        raise ValueError("coverage terminal aggregate did not complete the full audit")
    terminal_stages = complete.get("stages")
    if not isinstance(terminal_stages, Mapping) or set(terminal_stages) != {"static", "depth", "party"}:
        raise ValueError("coverage terminal aggregate does not declare every required stage")
    stages: dict[str, Any] = {}
    provenance: list[dict[str, Any]] = []
    for stage in ("static", "depth", "party"):
        summary = _read_json(root / stage / "summary.json")
        if terminal_stages.get(stage) != summary:
            raise ValueError(f"coverage terminal aggregate differs from {stage} summary")
        public_stage, stage_provenance = _coverage_stage(root, stage)
        stages[stage] = public_stage
        provenance.extend(stage_provenance)
    terminal_digest = _immutable_digest(complete.get("image_digest"), label="coverage terminal aggregate")
    common = _require_matching_provenance(provenance)
    if terminal_digest != common["image_digest"]:
        raise ValueError("coverage terminal image digest differs from stage provenance")
    all_clean = all(
        stage["coverage_complete"] and stage["finding_count"] == 0 and stage["failure_artifact_count"] == 0
        for stage in stages.values()
    )
    status = _require_derived_status(
        status=complete.get("status"), clean=all_clean, label="coverage terminal aggregate"
    )
    return {"status": status, "terminal_stage": "full", "stages": stages}, provenance


def _silent(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    complete = _read_json(root / "complete.json")
    audit = _read_json(root / "audit.json")
    _require_schema(complete, "pokezero.silent-mutation-audit-job.v1", label="silent-mutation terminal aggregate")
    _require_terminal_status(complete, label="silent-mutation terminal aggregate")
    _require_schema(audit, SILENT_MUTATION_AUDIT_SCHEMA, label="silent-mutation audit")
    provenance = _public_provenance(audit.get("audit_provenance"), label="silent-mutation audit")
    complete_provenance = _public_provenance(
        complete.get("audit_provenance"), label="silent-mutation terminal aggregate"
    )
    _require_matching_provenance([provenance, complete_provenance])
    if _immutable_digest(complete.get("image_digest"), label="silent-mutation terminal aggregate") != provenance["image_digest"]:
        raise ValueError("silent-mutation terminal image digest differs from audit provenance")
    command = provenance["command"]
    _require_command_script(command, "silent_mutation_audit.py", label="silent-mutation audit")
    _require_command_flag(command, "--interaction-registry", label="silent-mutation audit")
    if _require_command_value(command, "--observation-schema", label="silent-mutation audit") != "v3":
        raise ValueError("silent-mutation audit did not use the v3 observation schema")
    _require_command_int_at_least(
        command, "--random-games", MINIMUM_SILENT_RANDOM_GAMES, label="silent-mutation audit"
    )
    _require_command_int_at_least(
        command, "--max-rounds", MINIMUM_SILENT_MAX_ROUNDS, label="silent-mutation audit"
    )
    steps = _require_nonnegative_int(audit.get("steps_audited"), label="silent-mutation audit steps_audited")
    if steps == 0:
        raise ValueError("silent-mutation audit checked no state transitions")
    candidates = _require_nonnegative_int(
        audit.get("silent_candidate_count"), label="silent-mutation audit silent_candidate_count"
    )
    if complete.get("steps_audited") != steps or complete.get("silent_candidate_count") != candidates:
        raise ValueError("silent-mutation terminal aggregate differs from its audit artifact")
    status = _require_derived_status(
        status=complete.get("status"), clean=candidates == 0, label="silent-mutation terminal aggregate"
    )
    return {
        "status": status,
        "steps_audited": steps,
        "silent_candidate_count": candidates,
        "command_run": provenance,
    }, [provenance]


def _collision(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    complete_path = root / "controller" / "complete.json"
    audit_path = root / "audit" / "collision-audit.json"
    complete = _read_json(complete_path)
    _require_schema(complete, "pokezero.collision-audit-controller-complete.v1", label="collision terminal aggregate")
    _require_terminal_status(complete, label="collision terminal aggregate")
    audit = _read_json(audit_path)
    _require_schema(audit, COLLISION_AUDIT_SCHEMA, label="collision audit")
    if audit.get("expected_observation_schema") != OBSERVATION_SCHEMA:
        raise ValueError("collision audit did not encode the v3 observation schema")
    if audit.get("model_input_numeric_dtype") != COLLISION_MODEL_INPUT_NUMERIC_DTYPE:
        raise ValueError("collision audit did not hash the model float32 input boundary")
    expected_sha = complete.get("audit_sha256")
    actual_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if not isinstance(expected_sha, str) or expected_sha != actual_sha:
        raise ValueError("collision terminal aggregate does not authenticate its audit artifact")
    provenance = _public_provenance(audit.get("audit_provenance"), label="collision audit")
    if _immutable_digest(complete.get("image_digest"), label="collision terminal aggregate") != provenance["image_digest"]:
        raise ValueError("collision terminal image digest differs from audit provenance")
    records_scanned = _require_nonnegative_int(audit.get("records_scanned"), label="collision records_scanned")
    input_group_count = _require_nonnegative_int(audit.get("input_group_count"), label="collision input_group_count")
    collision_group_count = _require_nonnegative_int(
        audit.get("collision_group_count"), label="collision collision_group_count"
    )
    actionable_collision_group_count = _require_nonnegative_int(
        audit.get("actionable_collision_group_count"), label="collision actionable_collision_group_count"
    )
    if input_group_count > records_scanned:
        raise ValueError("collision input_group_count exceeds records_scanned")
    if collision_group_count > input_group_count:
        raise ValueError("collision collision_group_count exceeds input_group_count")
    if actionable_collision_group_count > collision_group_count:
        raise ValueError("collision actionable_collision_group_count exceeds collision_group_count")
    status = _require_derived_status(
        status=complete.get("status"),
        clean=actionable_collision_group_count == 0,
        label="collision terminal aggregate",
    )
    return {
        "status": status,
        "records_scanned": records_scanned,
        "input_group_count": input_group_count,
        "collision_group_count": collision_group_count,
        "actionable_collision_group_count": actionable_collision_group_count,
        "minimum_records_required": REQUIRED_COLLISION_RECORDS,
        "command_run": provenance,
    }, [provenance]


def _inventory(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    complete_path = root / "complete.json"
    inventory_path = root / "inventory.json"
    complete = _read_json(complete_path)
    _require_schema(complete, "pokezero.protocol-inventory-job.v1", label="protocol inventory terminal aggregate")
    _require_terminal_status(complete, label="protocol inventory terminal aggregate")
    inventory = _read_json(inventory_path)
    _require_schema(inventory, "pokezero.protocol-emission-inventory.v2", label="protocol inventory")
    provenance = _public_provenance(inventory.get("audit_provenance"), label="protocol inventory")
    observed = inventory.get("observed")
    if not isinstance(observed, Mapping):
        raise ValueError("protocol inventory is missing observed-census metadata")
    observed_inputs = observed.get("audit_provenance")
    if not isinstance(observed_inputs, list) or not observed_inputs:
        raise ValueError("protocol inventory is missing observed-census provenance")
    observed_provenance: list[dict[str, Any]] = []
    for index, entry in enumerate(observed_inputs):
        if not isinstance(entry, Mapping):
            raise ValueError(f"protocol inventory observed provenance {index} is malformed")
        observed_provenance.append(
            _validated_provenance_identity(
                entry.get("audit_provenance"), label=f"protocol inventory observed provenance {index}"
            )
        )
    _require_matching_provenance([provenance, *observed_provenance])
    if _immutable_digest(complete.get("image_digest"), label="protocol inventory terminal aggregate") != provenance["image_digest"]:
        raise ValueError("protocol inventory terminal image digest differs from audit provenance")
    expected_sha = complete.get("inventory_sha256")
    actual_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    if not isinstance(expected_sha, str) or expected_sha != actual_sha:
        raise ValueError("protocol inventory terminal aggregate does not authenticate its inventory artifact")
    differential = inventory.get("differential")
    if not isinstance(differential, Mapping):
        raise ValueError("protocol inventory is missing its E/O/C differential")
    engine = inventory.get("engine_emittable")
    consumed = inventory.get("consumer_dispatch")
    if not all(isinstance(value, Mapping) for value in (engine, observed, consumed)):
        raise ValueError("protocol inventory is missing E/O/C counts")
    terminal_counts = (
        _require_nonnegative_int(complete.get("observed_tag_count"), label="inventory terminal observed_tag_count"),
        _require_nonnegative_int(
            complete.get("observed_but_unconsumed_count"), label="inventory terminal observed_but_unconsumed_count"
        ),
        _require_nonnegative_int(
            complete.get("observed_but_unconsumed_unclassified_count"),
            label="inventory terminal observed_but_unconsumed_unclassified_count",
        ),
    )
    inventory_counts = (
        _require_nonnegative_int(observed.get("tag_count"), label="O tag_count"),
        _differential_count(differential, "observed_but_unconsumed"),
        _differential_count(differential, "observed_but_unconsumed_unclassified"),
    )
    if terminal_counts != inventory_counts:
        raise ValueError("protocol inventory terminal aggregate differs from its inventory artifact")
    semantic_coverage_gap_count = _differential_count(
        differential, "observed_signatures_without_semantic_coverage"
    )
    emittable_but_unobserved_count = _differential_count(differential, "emittable_but_unobserved")
    consumer_not_emittable_count = _differential_count(differential, "consumer_not_emittable")
    status = _require_derived_status(
        status=complete.get("status"),
        clean=semantic_coverage_gap_count == 0,
        label="protocol inventory terminal aggregate",
    )
    return {
        "status": status,
        "protocol_signature_schema": PROTOCOL_SIGNATURE_SCHEMA,
        "engine_emittable_tag_count": _require_nonnegative_int(engine.get("tag_count"), label="E tag_count"),
        "observed_tag_count": inventory_counts[0],
        "consumer_tag_count": _require_nonnegative_int(consumed.get("tag_count"), label="C tag_count"),
        "observed_but_unconsumed_count": inventory_counts[1],
        "observed_but_unconsumed_unclassified_count": inventory_counts[2],
        "observed_signatures_without_semantic_coverage_count": semantic_coverage_gap_count,
        "emittable_but_unobserved_count": emittable_but_unobserved_count,
        "consumer_not_emittable_count": consumer_not_emittable_count,
        "command_run": provenance,
    }, [provenance]


def _differential_count(differential: Mapping[str, Any], key: str) -> int:
    value = differential.get(key)
    if not isinstance(value, list):
        raise ValueError(f"protocol inventory differential {key} must be a list")
    return len(value)


def publish(
    *,
    coverage_root: Path,
    silent_root: Path,
    collision_root: Path,
    inventory_root: Path,
    output: Path,
) -> dict[str, Any]:
    coverage, coverage_provenance = _coverage(coverage_root)
    silent, silent_provenance = _silent(silent_root)
    collision, collision_provenance = _collision(collision_root)
    inventory, inventory_provenance = _inventory(inventory_root)
    provenance = _require_matching_provenance(
        [*coverage_provenance, *silent_provenance, *collision_provenance, *inventory_provenance]
    )
    if collision["records_scanned"] < REQUIRED_COLLISION_RECORDS:
        raise ValueError(
            "collision audit did not reach the required decision count: "
            f"{collision['records_scanned']} < {REQUIRED_COLLISION_RECORDS}"
        )
    payload = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "artifact_id": (
            f"v3-observation-audit-{provenance['public_repo_commit'][:12]}-"
            f"{provenance['showdown_source_hash'][:12]}"
        ),
        "published_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "layers": {
            "coverage": coverage,
            "silent_mutation": silent,
            "encoding_collision": collision,
            "protocol_inventory": inventory,
        },
    }
    _write_json_atomic(output, payload)
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--silent-root", type=Path, required=True)
    parser.add_argument("--collision-root", type=Path, required=True)
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = publish(
            coverage_root=args.coverage_root,
            silent_root=args.silent_root,
            collision_root=args.collision_root,
            inventory_root=args.inventory_root,
            output=args.out,
        )
    except ValueError as exc:
        print(f"audit evidence publication failed: {exc}", file=sys.stderr)
        return 2
    print(
        "V3_AUDIT_EVIDENCE_PUBLISHED "
        f"artifact_id={payload['artifact_id']} status=complete output={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
