#!/usr/bin/env python
"""Produce the file-backed certification execution manifest.

The readout consumes ``engine-cert-execution-manifest/2``.  This producer is
the only supported way to create that schema: it hashes the supplied files,
copies the contract and readout into content-addressed blobs, and derives shard
identity from the completed JSONL records instead of accepting handwritten
commit or probe-hash fields.

Example::

    python scripts/cert_execution_manifest.py \
      --contract reports/final-cert-contract.json \
      --output artifacts/execution-manifest.json \
      --shard-report artifacts/shard-0.json --checkpoint artifacts/shard-0.jsonl \
      --completion-marker artifacts/shard-0.complete \
      --behavioral-probe-log artifacts/shard-0.behavior.log \
      --branch-events-probe-log artifacts/shard-0.branch.log \
      --aggregate-behavioral-probe-log artifacts/aggregate.behavior.log \
      --aggregate-branch-events-probe-log artifacts/aggregate.branch.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.audit_provenance import public_repo_commit  # noqa: E402

CHECKPOINT_SCHEMA = "engine-transition-differential/1"
MANIFEST_SCHEMA = "engine-cert-execution-manifest/2"
STAMP_SCHEMA = "pokezero-engine-build/2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_PROBE_PASS_RE = re.compile(r"^\[[^]]+\] PASS\b", re.MULTILINE)
_PROBE_FAIL_RE = re.compile(r"^\[[^]]+\] FAIL\b", re.MULTILINE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_int(value: object) -> int | None:
    return value if type(value) is int else None


def validate_execution_manifest_schema(payload: object) -> list[str]:
    """Validate the bounded v2 schema without adding a runtime dependency.

    The checked-in JSON Schema remains the interoperable specification.  This
    deliberately small validator enforces the same required fields at both
    producer and consumer seams, so a missing per-shard probe record cannot
    become valid merely because the local image lacks ``jsonschema``.
    """

    errors: list[str] = []

    def mapping(value: object, label: str, required: Sequence[str]) -> Mapping[str, Any] | None:
        if not isinstance(value, Mapping):
            errors.append(f"{label} is not an object")
            return None
        for field in required:
            if field not in value:
                errors.append(f"{label} is missing required field {field!r}")
        return value

    def file(value: object, label: str) -> None:
        entry = mapping(value, label, ("path", "sha256"))
        if entry is None:
            return
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            errors.append(f"{label}.path is not a non-empty string")
        if not isinstance(entry.get("sha256"), str) or not _SHA256_RE.fullmatch(entry["sha256"]):
            errors.append(f"{label}.sha256 is not a lowercase SHA-256")

    def behavioral_probe(value: object, label: str) -> None:
        file(value, label)
        entry = mapping(value, label, ("path", "sha256", "passed", "total"))
        if entry is None:
            return
        passed, total = _strict_int(entry.get("passed")), _strict_int(entry.get("total"))
        if passed is None or passed < 0 or total is None or total < 0 or passed > total:
            errors.append(f"{label} has malformed passed/total counts")

    def branch_probe(value: object, label: str) -> None:
        file(value, label)
        entry = mapping(value, label, ("path", "sha256", "passed"))
        if entry is not None and type(entry.get("passed")) is not bool:
            errors.append(f"{label}.passed is not a boolean")

    root = mapping(
        payload,
        "manifest",
        ("schema", "producer", "source", "contract_blob", "readout_blob", "engine_provenance", "aggregate_provenance", "shards"),
    )
    if root is None:
        return errors
    if root.get("schema") != MANIFEST_SCHEMA:
        errors.append(f"manifest.schema is not {MANIFEST_SCHEMA}")
    file(root.get("producer"), "producer")
    file(root.get("contract_blob"), "contract_blob")
    file(root.get("readout_blob"), "readout_blob")
    source = mapping(root.get("source"), "source", ("commit", "checkout"))
    if source is not None:
        if not isinstance(source.get("commit"), str) or not _GIT_RE.fullmatch(source["commit"]):
            errors.append("source.commit is not a lowercase Git SHA")
        if not isinstance(source.get("checkout"), str) or not source["checkout"]:
            errors.append("source.checkout is not a non-empty string")
    engine = mapping(root.get("engine_provenance"), "engine_provenance", ("fingerprint", "stamp"))
    if engine is not None:
        if not isinstance(engine.get("fingerprint"), str) or not _SHA256_RE.fullmatch(engine["fingerprint"]):
            errors.append("engine_provenance.fingerprint is not a lowercase SHA-256")
        file(engine.get("stamp"), "engine_provenance.stamp")
    aggregate = mapping(root.get("aggregate_provenance"), "aggregate_provenance", ("behavioral_probes", "branch_events_probe"))
    if aggregate is not None:
        behavioral_probe(aggregate.get("behavioral_probes"), "aggregate_provenance.behavioral_probes")
        branch_probe(aggregate.get("branch_events_probe"), "aggregate_provenance.branch_events_probe")
    shards = root.get("shards")
    if not isinstance(shards, list) or not shards:
        errors.append("shards is not a non-empty array")
        return errors
    for index, raw in enumerate(shards):
        label = f"shards[{index}]"
        shard = mapping(
            raw,
            label,
            ("seed_start", "report", "checkpoint", "completion_marker", "image_commit", "behavioral_probes", "branch_events_probe"),
        )
        if shard is None:
            continue
        seed_start = _strict_int(shard.get("seed_start"))
        if seed_start is None or seed_start < 0:
            errors.append(f"{label}.seed_start is not a non-negative integer")
        file(shard.get("report"), f"{label}.report")
        file(shard.get("completion_marker"), f"{label}.completion_marker")
        if not isinstance(shard.get("image_commit"), str) or not _GIT_RE.fullmatch(shard["image_commit"]):
            errors.append(f"{label}.image_commit is not a lowercase Git SHA")
        behavioral_probe(shard.get("behavioral_probes"), f"{label}.behavioral_probes")
        branch_probe(shard.get("branch_events_probe"), f"{label}.branch_events_probe")
        checkpoint = mapping(shard.get("checkpoint"), f"{label}.checkpoint", ("path", "sha256", "records", "resume_provenance"))
        if checkpoint is not None:
            file(checkpoint, f"{label}.checkpoint")
            records = _strict_int(checkpoint.get("records"))
            if records is None or records <= 0:
                errors.append(f"{label}.checkpoint.records is not a positive integer")
            provenance = mapping(checkpoint.get("resume_provenance"), f"{label}.checkpoint.resume_provenance", ("source_commit", "engine_fingerprint", "image_commit"))
            if provenance is not None:
                for field, pattern in (("source_commit", _GIT_RE), ("engine_fingerprint", _SHA256_RE), ("image_commit", _GIT_RE)):
                    value = provenance.get(field)
                    if not isinstance(value, str) or not pattern.fullmatch(value):
                        errors.append(f"{label}.checkpoint.resume_provenance.{field} is malformed")
    return errors


def validate_final_contract_schema(contract: object) -> list[str]:
    """Validate public final-contract fields shared by producer and readout.

    This intentionally validates only public, execution-relevant state. Cluster
    configuration and any private rate-table implementation stay outside this
    repository and cannot become implicit certification inputs.
    """

    if not isinstance(contract, Mapping):
        return ["final contract root is not an object"]
    final = contract.get("registered_before_launch") is True or contract.get("requires_execution_contract") is True
    if not final:
        return []
    errors: list[str] = []
    if contract.get("registered_before_launch") is not True:
        errors.append("final contract registered_before_launch is not true")
    if contract.get("requires_execution_contract") is not True:
        errors.append("final contract requires_execution_contract is not true")
    gates = contract.get("certification_gates")
    if not isinstance(gates, Mapping):
        return errors + ["final contract has no certification_gates object"]
    for field, pattern in (
        ("required_source_commit", _GIT_RE),
        ("required_image_commit", _GIT_RE),
        ("required_engine_fingerprint", _SHA256_RE),
        ("required_readout_sha256", _SHA256_RE),
        ("required_execution_manifest_producer_sha256", _SHA256_RE),
    ):
        value = gates.get(field)
        if not isinstance(value, str) or not pattern.fullmatch(value):
            errors.append(f"final contract {field} is not a valid lowercase hash")
    table = contract.get("pre_registered_family_rate_table")
    if not isinstance(table, Mapping):
        errors.append("final contract has no pre_registered_family_rate_table object")
    else:
        for field in ("documented_families", "new_mechanisms_post_fix"):
            if not isinstance(table.get(field), Mapping):
                errors.append(f"final contract pre_registered_family_rate_table.{field} is not an object")
    if not isinstance(contract.get("predicted_class_rates_10k"), Mapping):
        errors.append("final contract predicted_class_rates_10k is not an object")
    return errors


def _file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"artifact is not a file: {path}")
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _source_checkout(repo_root: Path, *, source_commit: str | None = None) -> dict[str, str]:
    """Resolve image-injected source identity before falling back to local git."""

    commit = (source_commit or public_repo_commit(repo_root) or "").strip().lower()
    if not _GIT_RE.fullmatch(commit):
        raise ValueError(
            "cannot resolve public source commit; set POKEZERO_PUBLIC_REPO_COMMIT "
            "in no-.git images or pass --source-commit"
        )
    return {"commit": commit, "checkout": str(repo_root.resolve())}


def _freeze_blob(source: Path, artifact_dir: Path, label: str) -> dict[str, str]:
    """Copy a blob by digest once, then make the stored evidence read-only."""

    source = source.resolve()
    digest = _sha256(source)
    target = artifact_dir / f"{label}-{digest}{source.suffix}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or _sha256(target) != digest:
            raise ValueError(f"content-addressed {label} blob conflicts at {target}")
    else:
        target.write_bytes(source.read_bytes())
        target.chmod(0o444)
    return {"path": str(target.resolve()), "sha256": digest}


def _engine_provenance(stamp_path: Path) -> dict[str, Any]:
    stamp = _file(stamp_path)
    try:
        payload = json.loads(stamp_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"engine build stamp is invalid JSON: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != STAMP_SCHEMA:
        raise ValueError(f"engine build stamp must use {STAMP_SCHEMA}")
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
        raise ValueError("engine build stamp has no valid fingerprint")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"poke_engine", "pokezero_search"}:
        raise ValueError("engine build stamp does not attest both installed consumers")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"engine build stamp has malformed {name} artifact")
        path, digest = artifact.get("module_path"), artifact.get("module_sha256")
        if not isinstance(path, str) or not _SHA256_RE.fullmatch(str(digest)):
            raise ValueError(f"engine build stamp has malformed {name} module identity")
        module = Path(path)
        if not module.is_file() or _sha256(module) != digest:
            raise ValueError(f"engine build stamp {name} module no longer matches the artifact")
    return {"fingerprint": fingerprint, "stamp": stamp}


def _probe(path: Path, *, branch_probe: bool) -> dict[str, Any]:
    evidence: dict[str, Any] = _file(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    failures = len(_PROBE_FAIL_RE.findall(text))
    passes = len(_PROBE_PASS_RE.findall(text))
    if branch_probe:
        evidence["passed"] = failures == 0 and "[search-crate-branch-events] PASS" in text
    else:
        evidence["passed"] = passes
        evidence["total"] = passes + failures
    return evidence


def _checkpoint(path: Path, *, report: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    evidence: dict[str, Any] = _file(path)
    records: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid checkpoint JSON at line {line_number}: {error}") from error
        if not isinstance(record, Mapping) or record.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"{path}: invalid checkpoint record at line {line_number}")
        seed = _strict_int(record.get("seed"))
        if seed is None:
            raise ValueError(f"{path}: checkpoint record {line_number} has a malformed seed")
        if record.get("build_check") != "gated":
            raise ValueError(f"{path}: checkpoint record {line_number} is not build-gated")
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{path}: checkpoint record {line_number} has no resume provenance")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: checkpoint has no completed-game records")
    report_seeds = report.get("seeds")
    if not isinstance(report_seeds, Mapping):
        raise ValueError(f"{path}: paired shard report has no seed summary")
    start, end, distinct = (
        _strict_int(report_seeds.get("min")),
        _strict_int(report_seeds.get("max")),
        _strict_int(report_seeds.get("distinct")),
    )
    games = _strict_int(report.get("games"))
    if start is None or end is None or distinct is None or games is None or games <= 0:
        raise ValueError(f"{path}: paired shard report has malformed seed summary")
    if end != start + games - 1 or distinct != games:
        raise ValueError(f"{path}: paired shard report does not describe a complete contiguous game range")
    report_provenance = report.get("checkpoint_provenance")
    if not isinstance(report_provenance, Mapping):
        raise ValueError(f"{path}: paired shard report has no checkpoint provenance summary")
    if report_provenance.get("complete") is not True or _strict_int(
            report_provenance.get("records_with_provenance")) != games:
        raise ValueError(f"{path}: paired shard report has incomplete checkpoint provenance")
    seeds = [_strict_int(record.get("seed")) for record in records]
    if any(seed is None or not start <= seed <= end for seed in seeds):
        raise ValueError(f"{path}: checkpoint seed is outside paired shard seed band")
    if len(records) != games or len(set(seeds)) != games:
        raise ValueError(f"{path}: checkpoint record population does not match paired shard report")
    provenance_rows = [record["provenance"] for record in records]

    def one_hash(field: str, pattern: re.Pattern[str], label: str) -> str:
        values = [value.get(field) for value in provenance_rows]
        if not all(isinstance(value, str) and pattern.fullmatch(value) for value in values):
            raise ValueError(f"{path}: checkpoint has mixed or malformed {label} provenance")
        unique = set(values)
        if len(unique) != 1:
            raise ValueError(f"{path}: checkpoint has mixed or malformed {label} provenance")
        return next(iter(unique))

    source_commit = one_hash("source_commit", _GIT_RE, "source")
    fingerprint = one_hash("engine_fingerprint", _SHA256_RE, "engine")
    image_commit = one_hash("image_commit", _GIT_RE, "image")
    if not _GIT_RE.fullmatch(source_commit):
        raise ValueError(f"{path}: checkpoint has mixed or malformed source provenance")
    evidence["records"] = len(records)
    evidence["resume_provenance"] = {
        "source_commit": source_commit,
        "engine_fingerprint": fingerprint,
        "image_commit": image_commit,
    }
    return evidence, image_commit


def produce_manifest(
    *,
    contract: Path,
    readout: Path,
    output: Path,
    reports: Sequence[Path],
    checkpoints: Sequence[Path],
    completion_markers: Sequence[Path],
    behavioral_logs: Sequence[Path],
    branch_logs: Sequence[Path],
    aggregate_behavioral_log: Path,
    aggregate_branch_log: Path,
    engine_stamp: Path,
    repo_root: Path = REPO_ROOT,
    source_commit: str | None = None,
) -> dict[str, Any]:
    sequences = (reports, checkpoints, completion_markers, behavioral_logs, branch_logs)
    if not reports or any(len(values) != len(reports) for values in sequences[1:]):
        raise ValueError("each shard needs exactly one report, checkpoint, completion marker, and two probe logs")
    try:
        contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"contract is invalid JSON: {error}") from error
    if not isinstance(contract_payload, Mapping):
        raise ValueError("contract root is not an object")
    contract_errors = validate_final_contract_schema(contract_payload)
    if contract_errors:
        raise ValueError("final contract schema validation failed: " + "; ".join(contract_errors))
    artifact_dir = output.resolve().parent / "cert-provenance-blobs"
    source = _source_checkout(repo_root, source_commit=source_commit)
    engine = _engine_provenance(engine_stamp)
    gates = contract_payload.get("certification_gates")
    if isinstance(gates, Mapping):
        if gates.get("required_source_commit") != source["commit"]:
            raise ValueError("source checkout commit does not match final contract")
        if gates.get("required_engine_fingerprint") != engine["fingerprint"]:
            raise ValueError("engine build fingerprint does not match final contract")
        if gates.get("required_readout_sha256") != _sha256(readout):
            raise ValueError("readout hash does not match final contract")
        if gates.get("required_execution_manifest_producer_sha256") != _sha256(Path(__file__)):
            raise ValueError("execution manifest producer hash does not match final contract")
    shards: list[dict[str, Any]] = []
    for report_path, checkpoint_path, marker_path, behavior_log, branch_log in zip(
        reports, checkpoints, completion_markers, behavioral_logs, branch_logs
    ):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{report_path}: invalid shard report JSON: {error}") from error
        if not isinstance(report, Mapping):
            raise ValueError(f"{report_path}: shard report root is not an object")
        seeds = report.get("seeds")
        seed_start = _strict_int(seeds.get("min")) if isinstance(seeds, Mapping) else None
        if seed_start is None:
            raise ValueError(f"{report_path}: shard report has no valid seed start")
        checkpoint, image_commit = _checkpoint(checkpoint_path, report=report)
        provenance = checkpoint["resume_provenance"]
        if provenance["source_commit"] != source["commit"]:
            raise ValueError(f"{checkpoint_path}: source commit is not this producer checkout")
        if provenance["engine_fingerprint"] != engine["fingerprint"]:
            raise ValueError(f"{checkpoint_path}: engine fingerprint is not the supplied engine stamp")
        if isinstance(gates, Mapping) and image_commit != gates.get("required_image_commit"):
            raise ValueError(f"{checkpoint_path}: image commit does not match final contract")
        shards.append({
            "seed_start": seed_start,
            "report": _file(report_path),
            "checkpoint": checkpoint,
            "completion_marker": _file(marker_path),
            "image_commit": image_commit,
            "behavioral_probes": _probe(behavior_log, branch_probe=False),
            "branch_events_probe": _probe(branch_log, branch_probe=True),
        })
    if len({entry["seed_start"] for entry in shards}) != len(shards):
        raise ValueError("shard reports repeat a seed start")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "producer": _file(Path(__file__)),
        "source": source,
        "contract_blob": _freeze_blob(contract, artifact_dir, "contract"),
        "readout_blob": _freeze_blob(readout, artifact_dir, "readout"),
        "engine_provenance": engine,
        "aggregate_provenance": {
            "behavioral_probes": _probe(aggregate_behavioral_log, branch_probe=False),
            "branch_events_probe": _probe(aggregate_branch_log, branch_probe=True),
        },
        "shards": sorted(shards, key=lambda entry: entry["seed_start"]),
    }
    schema_errors = validate_execution_manifest_schema(manifest)
    if schema_errors:
        raise ValueError("produced execution manifest violates v2 schema: " + "; ".join(schema_errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--readout", type=Path, default=REPO_ROOT / "scripts" / "cert_sweep_readout.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-report", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--completion-marker", type=Path, action="append", required=True)
    parser.add_argument("--behavioral-probe-log", type=Path, action="append", required=True)
    parser.add_argument("--branch-events-probe-log", type=Path, action="append", required=True)
    parser.add_argument("--aggregate-behavioral-probe-log", type=Path, required=True)
    parser.add_argument("--aggregate-branch-events-probe-log", type=Path, required=True)
    parser.add_argument("--engine-stamp", type=Path, default=Path(sys.prefix) / ".engine-build-fingerprint.json")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--source-commit",
        default=None,
        help="explicit immutable public source commit for a no-.git image",
    )
    args = parser.parse_args(argv)
    try:
        manifest = produce_manifest(
            contract=args.contract,
            readout=args.readout,
            output=args.output,
            reports=args.shard_report,
            checkpoints=args.checkpoint,
            completion_markers=args.completion_marker,
            behavioral_logs=args.behavioral_probe_log,
            branch_logs=args.branch_events_probe_log,
            aggregate_behavioral_log=args.aggregate_behavioral_probe_log,
            aggregate_branch_log=args.aggregate_branch_events_probe_log,
            engine_stamp=args.engine_stamp,
            repo_root=args.repo_root,
            source_commit=args.source_commit,
        )
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.output} ({len(manifest['shards'])} shards, file-backed provenance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
