"""Promotion registry helpers for accepted checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from .evaluation import PromotionGateConfig, PromotionGateResult, evaluate_promotion_gate
from .opponents import historical_opponent_policy_specs

PROMOTION_REGISTRY_SCHEMA_VERSION = "pokezero.promotion_registry.v1"
NEURAL_SELFPLAY_SOURCE_TYPE = "pokezero.neural_selfplay_run.v1"


@dataclass(frozen=True)
class PromotionRegistryEntry:
    sequence: int
    policy_id: str | None
    checkpoint_path: str | None
    manifest_path: str
    source_type: str
    source_iteration: int | None
    promoted_at: str
    label: str | None
    notes: str | None
    gate_result: Mapping[str, Any]
    source_checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None

    @property
    def checkpoint_policy_spec(self) -> str | None:
        if not self.checkpoint_path:
            return None
        prefix = "neural:" if self.source_type == NEURAL_SELFPLAY_SOURCE_TYPE else "linear:"
        return f"{prefix}{self.checkpoint_path}"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "sequence": self.sequence,
            "policy_id": self.policy_id,
            "checkpoint_path": self.checkpoint_path,
            "manifest_path": self.manifest_path,
            "source_type": self.source_type,
            "source_iteration": self.source_iteration,
            "promoted_at": self.promoted_at,
            "label": self.label,
            "notes": self.notes,
            "gate_result": dict(self.gate_result),
        }
        if self.source_checkpoint_path is not None:
            payload["source_checkpoint_path"] = self.source_checkpoint_path
        if self.checkpoint_sha256 is not None:
            payload["checkpoint_sha256"] = self.checkpoint_sha256
        return payload


@dataclass(frozen=True)
class PromotionRegistry:
    path: Path
    entries: tuple[PromotionRegistryEntry, ...] = ()

    @property
    def latest(self) -> PromotionRegistryEntry | None:
        return self.entries[-1] if self.entries else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROMOTION_REGISTRY_SCHEMA_VERSION,
            "registry_path": str(self.path),
            "latest_policy_id": self.latest.policy_id if self.latest is not None else None,
            "latest_checkpoint_path": self.latest.checkpoint_path if self.latest is not None else None,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def checkpoint_policy_specs(self) -> tuple[str, ...]:
        return tuple(
            entry.checkpoint_policy_spec
            for entry in self.entries
            if entry.checkpoint_policy_spec is not None
        )

    def opponent_pool_policy_specs(
        self,
        *,
        max_historical_opponents: int,
        current_policy_spec: str | None = None,
    ) -> tuple[str, ...]:
        return historical_opponent_policy_specs(
            self.checkpoint_policy_specs(),
            current_policy_spec=current_policy_spec,
            max_historical_opponents=max_historical_opponents,
        )


@dataclass(frozen=True)
class PromotionRecordResult:
    registry_path: Path
    gate_result: PromotionGateResult
    entry: PromotionRegistryEntry | None
    registry: PromotionRegistry

    @property
    def recorded(self) -> bool:
        return self.entry is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded": self.recorded,
            "registry_path": str(self.registry_path),
            "entry": self.entry.to_dict() if self.entry is not None else None,
            "gate_result": self.gate_result.to_dict(),
            "registry": self.registry.to_dict(),
        }


@dataclass(frozen=True)
class PromotionRegistryVerificationCheck:
    name: str
    passed: bool
    entry_sequence: int | None
    observed: str | int | bool | None
    expected: str | int | bool | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "entry_sequence": self.entry_sequence,
            "observed": self.observed,
            "expected": self.expected,
            "message": self.message,
        }


@dataclass(frozen=True)
class PromotionRegistryVerificationResult:
    registry_path: Path
    entry_count: int
    checked_checkpoint_count: int
    verified_checksum_count: int
    verified_loadable_count: int
    checks: tuple[PromotionRegistryVerificationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_path": str(self.registry_path),
            "entry_count": self.entry_count,
            "checked_checkpoint_count": self.checked_checkpoint_count,
            "verified_checksum_count": self.verified_checksum_count,
            "verified_loadable_count": self.verified_loadable_count,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


def load_promotion_registry(path: Path) -> PromotionRegistry:
    registry_path = path.expanduser()
    if not registry_path.exists():
        return PromotionRegistry(path=registry_path, entries=())
    if not registry_path.is_file():
        raise ValueError(f"promotion registry path must be a file: {registry_path}")
    payload = _mapping(json.loads(registry_path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != PROMOTION_REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported promotion registry schema: {payload.get('schema_version')!r}.")
    entries = tuple(_entry_from_payload(entry) for entry in _sequence(payload.get("entries", ())))
    return PromotionRegistry(path=registry_path, entries=entries)


def record_promotion(
    manifest_path: Path,
    *,
    registry_path: Path,
    config: PromotionGateConfig = PromotionGateConfig(),
    label: str | None = None,
    notes: str | None = None,
    promoted_at: str | None = None,
    artifact_dir: Path | None = None,
    allow_duplicate: bool = False,
) -> PromotionRecordResult:
    gate_result = evaluate_promotion_gate(manifest_path, config=config)
    registry = load_promotion_registry(registry_path)
    if not gate_result.passed:
        return PromotionRecordResult(
            registry_path=registry.path,
            gate_result=gate_result,
            entry=None,
            registry=registry,
        )
    if not allow_duplicate:
        _reject_duplicate(registry, gate_result)
    sequence = len(registry.entries) + 1
    checkpoint_path = gate_result.checkpoint_path
    source_checkpoint_path = None
    checkpoint_sha256 = None
    if artifact_dir is not None:
        # Copy before writing the registry so retries can safely overwrite the same
        # sequence-named orphan if the later registry write fails.
        checkpoint_path, checkpoint_sha256 = _copy_checkpoint_artifact(
            gate_result,
            artifact_dir=artifact_dir,
            sequence=sequence,
        )
        source_checkpoint_path = gate_result.checkpoint_path
    entry = PromotionRegistryEntry(
        sequence=sequence,
        policy_id=gate_result.candidate_policy_id,
        checkpoint_path=checkpoint_path,
        manifest_path=str(gate_result.manifest_path),
        source_type=gate_result.source_type,
        source_iteration=gate_result.source_iteration,
        promoted_at=promoted_at or _utc_now_iso(),
        label=label,
        notes=notes,
        gate_result=gate_result.to_dict(),
        source_checkpoint_path=source_checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )
    updated = PromotionRegistry(path=registry.path, entries=(*registry.entries, entry))
    _write_registry(updated)
    return PromotionRecordResult(
        registry_path=registry.path,
        gate_result=gate_result,
        entry=entry,
        registry=updated,
    )


def verify_promotion_registry(
    path: Path,
    *,
    verify_checksums: bool = True,
    require_checksums: bool = False,
    verify_loadable: bool = False,
) -> PromotionRegistryVerificationResult:
    registry = load_promotion_registry(path)
    checks: list[PromotionRegistryVerificationCheck] = [
        _sequence_check(registry),
    ]
    checked_checkpoint_count = 0
    verified_checksum_count = 0
    verified_loadable_count = 0
    for entry in registry.entries:
        checks.append(_gate_result_passed_check(entry))
        if not entry.checkpoint_path:
            checks.append(
                PromotionRegistryVerificationCheck(
                    name="checkpoint_path_present",
                    passed=False,
                    entry_sequence=entry.sequence,
                    observed=entry.checkpoint_path,
                    expected="non-empty",
                    message="promotion entry must include a checkpoint path",
                )
            )
            continue
        resolved_checkpoint = _resolve_selection_checkpoint_path(entry.checkpoint_path)
        checks.append(
            PromotionRegistryVerificationCheck(
                name="checkpoint_exists",
                passed=resolved_checkpoint is not None,
                entry_sequence=entry.sequence,
                observed=entry.checkpoint_path,
                expected="existing file",
                message="promotion checkpoint path resolves to an existing file",
            )
        )
        if resolved_checkpoint is None:
            continue
        checked_checkpoint_count += 1
        if verify_loadable:
            loadable_check, policy_id_check = _policy_loadable_checks(entry)
            checks.append(loadable_check)
            if loadable_check.passed:
                verified_loadable_count += 1
            if policy_id_check is not None:
                checks.append(policy_id_check)
        if verify_checksums and entry.checkpoint_sha256 is not None:
            observed_sha256 = _sha256_file(resolved_checkpoint)
            verified_checksum_count += 1
            checks.append(
                PromotionRegistryVerificationCheck(
                    name="checkpoint_sha256",
                    passed=observed_sha256 == entry.checkpoint_sha256,
                    entry_sequence=entry.sequence,
                    observed=observed_sha256,
                    expected=entry.checkpoint_sha256,
                    message="promotion checkpoint checksum matches registry metadata",
                )
            )
        elif require_checksums:
            checks.append(
                PromotionRegistryVerificationCheck(
                    name="checkpoint_sha256_present",
                    passed=False,
                    entry_sequence=entry.sequence,
                    observed=None,
                    expected="sha256 metadata",
                    message="promotion checkpoint checksum metadata is required",
                )
            )
    return PromotionRegistryVerificationResult(
        registry_path=registry.path,
        entry_count=len(registry.entries),
        checked_checkpoint_count=checked_checkpoint_count,
        verified_checksum_count=verified_checksum_count,
        verified_loadable_count=verified_loadable_count,
        checks=tuple(checks),
    )


def _entry_from_payload(payload: Any) -> PromotionRegistryEntry:
    entry = _mapping(payload)
    return PromotionRegistryEntry(
        sequence=int(entry["sequence"]),
        policy_id=_optional_str(entry.get("policy_id")),
        checkpoint_path=_optional_str(entry.get("checkpoint_path")),
        manifest_path=str(entry["manifest_path"]),
        source_type=str(entry["source_type"]),
        source_iteration=_optional_int(entry.get("source_iteration")),
        promoted_at=str(entry["promoted_at"]),
        label=_optional_str(entry.get("label")),
        notes=_optional_str(entry.get("notes")),
        gate_result=_mapping(entry.get("gate_result", {})),
        source_checkpoint_path=_optional_str(entry.get("source_checkpoint_path")),
        checkpoint_sha256=_optional_str(entry.get("checkpoint_sha256")),
    )


def _sequence_check(registry: PromotionRegistry) -> PromotionRegistryVerificationCheck:
    observed = tuple(entry.sequence for entry in registry.entries)
    expected = tuple(range(1, len(registry.entries) + 1))
    return PromotionRegistryVerificationCheck(
        name="sequence_contiguous",
        passed=observed == expected,
        entry_sequence=None,
        observed=",".join(str(sequence) for sequence in observed),
        expected=",".join(str(sequence) for sequence in expected),
        message="promotion registry sequences are contiguous and ordered",
    )


def _gate_result_passed_check(entry: PromotionRegistryEntry) -> PromotionRegistryVerificationCheck:
    gate_result = _mapping(entry.gate_result)
    observed = bool(gate_result.get("passed"))
    return PromotionRegistryVerificationCheck(
        name="gate_result_passed",
        passed=observed,
        entry_sequence=entry.sequence,
        observed=observed,
        expected=True,
        message="promotion entry embeds a passing gate result",
    )


def _policy_loadable_checks(
    entry: PromotionRegistryEntry,
) -> tuple[PromotionRegistryVerificationCheck, PromotionRegistryVerificationCheck | None]:
    policy_spec = entry.checkpoint_policy_spec
    if policy_spec is None:
        return (
            PromotionRegistryVerificationCheck(
                name="checkpoint_policy_loadable",
                passed=False,
                entry_sequence=entry.sequence,
                observed=None,
                expected="loadable policy spec",
                message="promotion checkpoint must have a loadable policy spec",
            ),
            None,
        )
    try:
        from .collection import policy_from_spec

        policy = policy_from_spec(policy_spec)
    except Exception as exc:
        return (
            PromotionRegistryVerificationCheck(
                name="checkpoint_policy_loadable",
                passed=False,
                entry_sequence=entry.sequence,
                observed=f"{type(exc).__name__}: {exc}",
                expected="loadable policy spec",
                message="promotion checkpoint loads through the policy selection path",
            ),
            None,
        )
    loaded_policy_id = getattr(policy, "policy_id", None)
    loadable_check = PromotionRegistryVerificationCheck(
        name="checkpoint_policy_loadable",
        passed=True,
        entry_sequence=entry.sequence,
        observed=policy_spec,
        expected="loadable policy spec",
        message="promotion checkpoint loads through the policy selection path",
    )
    if entry.policy_id is None:
        return loadable_check, None
    policy_id_check = PromotionRegistryVerificationCheck(
        name="checkpoint_policy_id",
        passed=loaded_policy_id == entry.policy_id,
        entry_sequence=entry.sequence,
        observed=str(loaded_policy_id) if loaded_policy_id is not None else None,
        expected=entry.policy_id,
        message="loaded policy id matches registry metadata",
    )
    return loadable_check, policy_id_check


def _reject_duplicate(registry: PromotionRegistry, gate_result: PromotionGateResult) -> None:
    for entry in registry.entries:
        if gate_result.checkpoint_path is None:
            continue
        if entry.checkpoint_path == gate_result.checkpoint_path or entry.source_checkpoint_path == gate_result.checkpoint_path:
            raise ValueError(f"checkpoint is already promoted: {gate_result.checkpoint_path}")


def _copy_checkpoint_artifact(
    gate_result: PromotionGateResult,
    *,
    artifact_dir: Path,
    sequence: int,
) -> tuple[str, str]:
    if gate_result.checkpoint_path is None:
        raise ValueError("cannot copy promoted artifact: gate result has no checkpoint path.")
    source_path = _resolve_checkpoint_path(
        gate_result.checkpoint_path,
        manifest_path=gate_result.manifest_path,
    )
    if source_path is None:
        raise FileNotFoundError(f"Promoted checkpoint does not exist: {gate_result.checkpoint_path}")
    source_sha256 = _sha256_file(source_path)
    target_dir = artifact_dir.expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / _artifact_file_name(
        sequence=sequence,
        policy_id=gate_result.candidate_policy_id,
        source_path=source_path,
    )
    temporary_path = target_path.with_name(f".{target_path.name}.tmp")
    shutil.copy2(source_path, temporary_path)
    temporary_path.replace(target_path)
    if _sha256_file(target_path) != source_sha256:
        raise OSError(f"Promoted checkpoint copy checksum mismatch: {target_path}")
    return str(target_path), source_sha256


def _resolve_checkpoint_path(checkpoint_path: str, *, manifest_path: Path) -> Path | None:
    raw_path = Path(checkpoint_path).expanduser()
    candidates = (raw_path,)
    if not raw_path.is_absolute():
        candidates = (
            manifest_path.parent / raw_path,
            manifest_path.parent.parent / raw_path,
            raw_path,
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_selection_checkpoint_path(checkpoint_path: str) -> Path | None:
    raw_path = Path(checkpoint_path).expanduser()
    if raw_path.exists() and raw_path.is_file():
        return raw_path
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_file_name(*, sequence: int, policy_id: str | None, source_path: Path) -> str:
    safe_policy_id = _safe_path_component(policy_id or "unknown-policy")
    suffix = source_path.suffix or ".json"
    return f"{sequence:06d}-{safe_policy_id}{suffix}"


def _safe_path_component(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_", "."} else "-" for character in value)
    return cleaned.strip(".-") or "unknown"


def _write_registry(registry: PromotionRegistry) -> None:
    registry.path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = registry.path.with_name(f".{registry.path.name}.tmp")
    temporary_path.write_text(json.dumps(registry.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(registry.path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object payload.")
    return value


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__iter__"):
        raise ValueError("expected JSON array payload.")
    return tuple(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
