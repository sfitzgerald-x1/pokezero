"""Promotion registry helpers for accepted checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from .evaluation import PromotionGateConfig, PromotionGateResult, evaluate_promotion_gate

PROMOTION_REGISTRY_SCHEMA_VERSION = "pokezero.promotion_registry.v1"


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
            f"linear:{entry.checkpoint_path}"
            for entry in self.entries
            if entry.checkpoint_path
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
    if artifact_dir is not None:
        checkpoint_path = _copy_checkpoint_artifact(
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
    )
    updated = PromotionRegistry(path=registry.path, entries=(*registry.entries, entry))
    _write_registry(updated)
    return PromotionRecordResult(
        registry_path=registry.path,
        gate_result=gate_result,
        entry=entry,
        registry=updated,
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
    )


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
) -> str:
    if gate_result.checkpoint_path is None:
        raise ValueError("cannot copy promoted artifact: gate result has no checkpoint path.")
    source_path = _resolve_checkpoint_path(
        gate_result.checkpoint_path,
        manifest_path=gate_result.manifest_path,
    )
    if source_path is None:
        raise FileNotFoundError(f"Promoted checkpoint does not exist: {gate_result.checkpoint_path}")
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
    return str(target_path)


def _resolve_checkpoint_path(checkpoint_path: str, *, manifest_path: Path) -> Path | None:
    raw_path = Path(checkpoint_path).expanduser()
    candidates = (raw_path,)
    if not raw_path.is_absolute():
        candidates = (
            raw_path,
            Path.cwd() / raw_path,
            manifest_path.parent / raw_path,
            manifest_path.parent.parent / raw_path,
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


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
