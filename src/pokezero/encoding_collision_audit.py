"""Detect public-state distinctions collapsed by an observation encoding.

The audit intentionally consumes only :mod:`public_decision_corpus` records.
Those records contain the acting player's model input plus public replay facts;
they exclude the opponent's request and all other private simulator state.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .observation import OBSERVATION_SCHEMA_VERSION_V3
from .public_decision_corpus import (
    PublicDecisionCorpusStream,
    PublicDecisionRecord,
    canonical_json_sha256,
    open_public_decision_corpus,
)


ENCODING_COLLISION_AUDIT_SCHEMA_VERSION = "pokezero.encoding-collision-audit.v1"
COLLISION_SKETCH_SCHEMA_VERSION = "pokezero.encoding-collision-sketch.v1"
WHITELIST_VERSION = "pokezero.encoding-collision-whitelist.v1"
DELIBERATE_ABSTRACTION_WHITELIST = frozenset(
    {
        "hp-quantization",
        "tendency-bucketing",
        "transition-window-truncation",
    }
)


def _input_payload(record: PublicDecisionRecord) -> dict[str, Any]:
    observation = record.observation
    return {
        "schema_version": observation.schema_version,
        "categorical_ids": observation.categorical_ids,
        "numeric_features": observation.numeric_features,
        "token_type_ids": observation.token_type_ids,
        "attention_mask": observation.attention_mask,
        "legal_action_mask": observation.legal_action_mask,
    }


def encoded_input_hash(record: PublicDecisionRecord) -> str:
    """Hash only values the policy can receive at a decision boundary."""

    return canonical_json_sha256(_input_payload(record))


def decision_kind(record: PublicDecisionRecord) -> str:
    """Classify the public request shape without using the chosen action."""

    mask = record.current_legal_action_mask
    moves_legal = any(mask[:4])
    switches_legal = any(mask[4:])
    if moves_legal and switches_legal:
        return "move-or-switch"
    if moves_legal:
        return "move-only"
    if switches_legal:
        return "switch-only"
    return "no-legal-action"


def _public_payload(record: PublicDecisionRecord) -> dict[str, Any]:
    """Canonical public reference independent of raw protocol formatting."""

    return {
        "format_id": record.format_id,
        "turn_index": record.turn_index,
        "acting_player_state": dict(record.observation.acting_player_state),
        "public_belief_view": dict(record.public_belief_view),
        "public_resolved_action_rounds": [round_.to_dict() for round_ in record.public_resolved_action_rounds],
        "history": [
            {
                "turn_index": entry.turn_index,
                "acting_player_state": dict(entry.observation.acting_player_state),
            }
            for entry in record.history
        ],
    }


def canonical_public_fingerprint(record: PublicDecisionRecord) -> tuple[str, dict[str, Any]]:
    payload = _public_payload(record)
    return canonical_json_sha256(payload), payload


@dataclass(frozen=True)
class CollisionSketchRecord:
    """Compact public locator for a potential input/public-state collision.

    Sketches intentionally retain hashes instead of model tensors or public
    state payloads. A later, bounded hydration pass can replay only a concrete
    collision pair to recover the field-level explanation.
    """

    decision_id: str
    battle_id: str
    seed: int
    format_id: str
    acting_player: str
    turn_index: int
    observation_schema: str
    decision_kind: str
    input_hash: str
    public_fingerprint: str

    @classmethod
    def from_record(cls, record: PublicDecisionRecord) -> "CollisionSketchRecord":
        public_fingerprint, _ = canonical_public_fingerprint(record)
        return cls(
            decision_id=record.decision_id,
            battle_id=record.battle_id,
            seed=record.seed,
            format_id=record.format_id,
            acting_player=record.acting_player,
            turn_index=record.turn_index,
            observation_schema=record.observation.schema_version,
            decision_kind=decision_kind(record),
            input_hash=encoded_input_hash(record),
            public_fingerprint=public_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "sketch",
            "schema_version": COLLISION_SKETCH_SCHEMA_VERSION,
            "decision_id": self.decision_id,
            "battle_id": self.battle_id,
            "seed": self.seed,
            "format_id": self.format_id,
            "acting_player": self.acting_player,
            "turn_index": self.turn_index,
            "observation_schema": self.observation_schema,
            "decision_kind": self.decision_kind,
            "input_hash": self.input_hash,
            "public_fingerprint": self.public_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CollisionSketchRecord":
        expected = {
            "record_type",
            "schema_version",
            "decision_id",
            "battle_id",
            "seed",
            "format_id",
            "acting_player",
            "turn_index",
            "observation_schema",
            "decision_kind",
            "input_hash",
            "public_fingerprint",
        }
        if set(payload) != expected:
            raise ValueError("collision sketch record has unsupported fields.")
        if payload.get("record_type") != "sketch" or payload.get("schema_version") != COLLISION_SKETCH_SCHEMA_VERSION:
            raise ValueError("not a collision sketch record.")
        hashes = (payload.get("input_hash"), payload.get("public_fingerprint"))
        if any(not isinstance(value, str) or not _is_sha256(value) for value in hashes):
            raise ValueError("collision sketch hashes must be lowercase SHA-256 values.")
        player = payload.get("acting_player")
        if player not in {"p1", "p2"}:
            raise ValueError("collision sketch acting_player must be p1 or p2.")
        kind = payload.get("decision_kind")
        if kind not in {"move-only", "move-or-switch", "switch-only", "no-legal-action"}:
            raise ValueError("collision sketch decision_kind is invalid.")
        observation_schema = payload.get("observation_schema")
        if not isinstance(observation_schema, str) or not observation_schema:
            raise ValueError("collision sketch observation_schema is required.")
        for field in ("decision_id", "battle_id", "format_id"):
            if not isinstance(payload.get(field), str) or not payload[field]:
                raise ValueError(f"collision sketch {field} is required.")
        seed = payload.get("seed")
        turn_index = payload.get("turn_index")
        if not isinstance(seed, int) or seed < 0 or not isinstance(turn_index, int) or turn_index < 0:
            raise ValueError("collision sketch seed and turn_index must be non-negative integers.")
        return cls(
            decision_id=payload["decision_id"],
            battle_id=payload["battle_id"],
            seed=seed,
            format_id=payload["format_id"],
            acting_player=player,
            turn_index=turn_index,
            observation_schema=observation_schema,
            decision_kind=kind,
            input_hash=payload["input_hash"],
            public_fingerprint=payload["public_fingerprint"],
        )


def collision_sketch_manifest(*, capture_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable public provenance for one compact sketch shard."""

    safe_capture_manifest = json.loads(json.dumps(dict(capture_manifest), sort_keys=True))
    if safe_capture_manifest.get("opponent_legal_mask_mode") != "hidden":
        raise ValueError("collision sketch capture requires hidden opponent legal masks.")
    root_noise = safe_capture_manifest.get("root_noise")
    if not isinstance(root_noise, Mapping) or root_noise.get("enabled") is not False:
        raise ValueError("collision sketch capture requires root noise to be disabled.")
    return {
        "record_type": "manifest",
        "schema_version": COLLISION_SKETCH_SCHEMA_VERSION,
        "capture_manifest": safe_capture_manifest,
        "capture_manifest_sha256": canonical_json_sha256(safe_capture_manifest),
    }


class CollisionSketchWriter:
    """Write or resume a small public-only collision sketch sidecar.

    Shards replay deterministic seeds after a pod retry. Resuming valid records
    lets that retry preserve already-fsynced work while de-duplicating replayed
    decisions. Only an incomplete final JSON line may be removed, because a
    later replay deterministically restores it.
    """

    def __init__(self, path: Path, *, manifest: Mapping[str, Any], resume: bool = False) -> None:
        self.path = path
        self.manifest = _validated_collision_sketch_manifest(manifest)
        self._seen_decision_ids: set[str] = set()
        self.resumed_record_count = 0
        self.record_count = 0
        self.recovered_trailing_partial = False
        if path.exists() and not resume:
            raise FileExistsError(f"collision sketch already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.recovered_trailing_partial = _normalize_collision_sketch_tail(path)
            existing_manifest, records = _iter_collision_sketch(path)
            if existing_manifest != self.manifest:
                raise ValueError("existing collision sketch has incompatible immutable capture manifest.")
            for record in records:
                if record.decision_id in self._seen_decision_ids:
                    raise ValueError(f"existing collision sketch duplicates decision_id {record.decision_id!r}.")
                self._seen_decision_ids.add(record.decision_id)
                self.resumed_record_count += 1
            self.record_count = self.resumed_record_count
            self._handle = path.open("a", encoding="utf-8")
        else:
            self._handle = path.open("x", encoding="utf-8")
            self._write(self.manifest, sync=True)

    def append_record(self, record: PublicDecisionRecord) -> int:
        return self._append_record(record, sync=True)

    def _append_record(self, record: PublicDecisionRecord, *, sync: bool) -> int:
        sketch = CollisionSketchRecord.from_record(record)
        if sketch.observation_schema != OBSERVATION_SCHEMA_VERSION_V3:
            raise ValueError(
                "collision sketch capture requires schema "
                f"{OBSERVATION_SCHEMA_VERSION_V3!r}; encountered {sketch.observation_schema!r}"
            )
        if sketch.decision_id in self._seen_decision_ids:
            return 0
        self._write(sketch.to_dict(), sync=sync)
        self._seen_decision_ids.add(sketch.decision_id)
        self.record_count += 1
        return 1

    def append_trajectory(self, records: Iterable[PublicDecisionRecord]) -> int:
        written = sum(self._append_record(record, sync=False) for record in records)
        if written:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        return written

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "CollisionSketchWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _write(self, payload: Mapping[str, Any], *, sync: bool) -> None:
        self._handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        self._handle.write("\n")
        self._handle.flush()
        if sync:
            os.fsync(self._handle.fileno())


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _normalize_collision_sketch_tail(path: Path) -> bool:
    """Repair only a retry-interrupted final JSON line, then return whether changed."""

    content = path.read_bytes()
    if not content or content.endswith(b"\n"):
        return False
    newline = content.rfind(b"\n")
    tail = content[newline + 1 :]
    try:
        json.loads(tail)
    except json.JSONDecodeError:
        with path.open("r+b") as handle:
            handle.truncate(newline + 1)
            handle.flush()
            os.fsync(handle.fileno())
        return True
    with path.open("ab") as handle:
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _validated_collision_sketch_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"record_type", "schema_version", "capture_manifest", "capture_manifest_sha256"}
    if set(payload) != expected:
        raise ValueError("collision sketch manifest has unsupported fields.")
    if payload.get("record_type") != "manifest" or payload.get("schema_version") != COLLISION_SKETCH_SCHEMA_VERSION:
        raise ValueError("not a collision sketch manifest.")
    capture_manifest = payload.get("capture_manifest")
    if not isinstance(capture_manifest, Mapping):
        raise ValueError("collision sketch capture_manifest must be an object.")
    expected_hash = canonical_json_sha256(capture_manifest)
    if payload.get("capture_manifest_sha256") != expected_hash:
        raise ValueError("collision sketch capture manifest hash does not match its payload.")
    if capture_manifest.get("opponent_legal_mask_mode") != "hidden":
        raise ValueError("collision sketch capture manifest must use hidden opponent legal masks.")
    root_noise = capture_manifest.get("root_noise")
    if not isinstance(root_noise, Mapping) or root_noise.get("enabled") is not False:
        raise ValueError("collision sketch capture manifest must disable root noise.")
    return dict(payload)


def _difference_paths(left: Any, right: Any, *, prefix: str = "", limit: int = 64) -> list[str]:
    """Return bounded, deterministic public-field differences for one collision pair."""

    if left == right:
        return []
    if len(prefix) >= 1_024:
        return [prefix]
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = _difference_paths(left.get(key), right.get(key), prefix=f"{prefix}.{key}".lstrip("."), limit=limit)
            paths.extend(child)
            if len(paths) >= limit:
                return paths[:limit]
        return paths
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        paths = []
        for index in range(max(len(left), len(right))):
            first = left[index] if index < len(left) else None
            second = right[index] if index < len(right) else None
            paths.extend(_difference_paths(first, second, prefix=f"{prefix}[{index}]", limit=limit))
            if len(paths) >= limit:
                return paths[:limit]
        return paths
    return [prefix or "value"]


def _whitelist_classification(left: Mapping[str, Any], right: Mapping[str, Any], differences: list[str]) -> str | None:
    """Classify only well-defined deliberate abstractions; everything else remains actionable."""

    if not differences:
        return None
    if (
        left.get("acting_player_state") == right.get("acting_player_state")
        and left.get("public_belief_view") == right.get("public_belief_view")
        and left.get("public_resolved_action_rounds") != right.get("public_resolved_action_rounds")
    ):
        return "transition-window-truncation"
    if all(path.endswith("hp_fraction") for path in differences):
        return "hp-quantization"
    return None


def _representative(record: PublicDecisionRecord, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": record.decision_id,
        "battle_id": record.battle_id,
        "seed": record.seed,
        "turn_index": record.turn_index,
        "public_state": dict(payload),
    }


@dataclass
class _FingerprintSample:
    count: int
    payload: Mapping[str, Any]
    representative: Mapping[str, Any]


@dataclass
class _InputGroup:
    count: int = 0
    public_fingerprints: dict[str, _FingerprintSample] = field(default_factory=dict)


@dataclass
class EncodingCollisionAudit:
    """Streaming aggregation of public-state collisions for one corpus selection."""

    expected_observation_schema: str = OBSERVATION_SCHEMA_VERSION_V3
    groups: dict[tuple[str, str, str, str], _InputGroup] = field(default_factory=dict)
    records_scanned: int = 0

    def add(self, record: PublicDecisionRecord) -> None:
        if record.observation.schema_version != self.expected_observation_schema:
            raise ValueError(
                "encoding collision audit requires schema "
                f"{self.expected_observation_schema!r}; encountered {record.observation.schema_version!r}"
            )
        self.records_scanned += 1
        input_hash = encoded_input_hash(record)
        # Keep p1/p2 separate even though observations are side-relative: this
        # prevents a corpus capture convention from masquerading as a schema
        # collision. Request shape is separate for the same reason.
        scope = (record.observation.schema_version, record.acting_player, decision_kind(record), input_hash)
        group = self.groups.setdefault(scope, _InputGroup())
        group.count += 1
        public_hash, payload = canonical_public_fingerprint(record)
        sample = group.public_fingerprints.get(public_hash)
        if sample is None:
            group.public_fingerprints[public_hash] = _FingerprintSample(
                count=1,
                payload=payload,
                representative=_representative(record, payload),
            )
        else:
            sample.count += 1

    def to_json_dict(self, *, corpus: Mapping[str, Any]) -> dict[str, Any]:
        collisions: list[dict[str, Any]] = []
        whitelist_counts: Counter[str] = Counter()
        for (schema, player, kind, input_hash), group in sorted(self.groups.items()):
            if len(group.public_fingerprints) < 2:
                continue
            samples = sorted(group.public_fingerprints.items())
            base_hash, base = samples[0]
            alternatives = []
            classifications: set[str] = set()
            for alternative_hash, alternative in samples[1:]:
                differences = _difference_paths(base.payload, alternative.payload)
                classification = _whitelist_classification(base.payload, alternative.payload, differences)
                if classification is not None:
                    classifications.add(classification)
                    whitelist_counts[classification] += 1
                alternatives.append(
                    {
                        "public_fingerprint": alternative_hash,
                        "records": alternative.count,
                        "difference_paths": differences,
                        "whitelist_classification": classification,
                        "representative": dict(alternative.representative),
                    }
                )
            actionable = any(item["whitelist_classification"] is None for item in alternatives)
            collisions.append(
                {
                    "input_hash": input_hash,
                    "observation_schema": schema,
                    "acting_player": player,
                    "decision_kind": kind,
                    "records": group.count,
                    "public_fingerprint_count": len(samples),
                    "base": {
                        "public_fingerprint": base_hash,
                        "records": base.count,
                        "representative": dict(base.representative),
                    },
                    "alternatives": alternatives,
                    "whitelist_classifications": sorted(classifications),
                    "actionable": actionable,
                }
            )
        actionable = sum(1 for collision in collisions if collision["actionable"])
        return {
            "schema_version": ENCODING_COLLISION_AUDIT_SCHEMA_VERSION,
            "expected_observation_schema": self.expected_observation_schema,
            "whitelist": {
                "version": WHITELIST_VERSION,
                "rules": sorted(DELIBERATE_ABSTRACTION_WHITELIST),
                "matched_pair_counts": dict(sorted(whitelist_counts.items())),
            },
            "corpus": dict(corpus),
            "records_scanned": self.records_scanned,
            "input_group_count": len(self.groups),
            "collision_group_count": len(collisions),
            "actionable_collision_group_count": actionable,
            "collision_groups": collisions,
        }


@dataclass
class _SketchFingerprintSample:
    count: int
    locator: Mapping[str, Any]


@dataclass
class _SketchInputGroup:
    count: int = 0
    public_fingerprints: dict[str, _SketchFingerprintSample] = field(default_factory=dict)


@dataclass
class EncodingCollisionSketchAudit:
    """Aggregate compact collision sketches without retaining model tensors.

    A non-empty result is intentionally a candidate rather than a verdict: the
    compact first pass proves distinct public fingerprints shared one model
    input, while a later replay hydrates the exact field differences.
    """

    expected_observation_schema: str = OBSERVATION_SCHEMA_VERSION_V3
    groups: dict[tuple[str, str, str, str], _SketchInputGroup] = field(default_factory=dict)
    records_scanned: int = 0

    def add(self, record: CollisionSketchRecord) -> None:
        if record.observation_schema != self.expected_observation_schema:
            raise ValueError(
                "collision sketch audit requires schema "
                f"{self.expected_observation_schema!r}; encountered {record.observation_schema!r}"
            )
        self.records_scanned += 1
        scope = (record.observation_schema, record.acting_player, record.decision_kind, record.input_hash)
        group = self.groups.setdefault(scope, _SketchInputGroup())
        group.count += 1
        sample = group.public_fingerprints.get(record.public_fingerprint)
        if sample is None:
            group.public_fingerprints[record.public_fingerprint] = _SketchFingerprintSample(
                count=1,
                locator={
                    "decision_id": record.decision_id,
                    "battle_id": record.battle_id,
                    "seed": record.seed,
                    "turn_index": record.turn_index,
                },
            )
        else:
            sample.count += 1

    def to_json_dict(self, *, corpus: Mapping[str, Any]) -> dict[str, Any]:
        collisions: list[dict[str, Any]] = []
        for (schema, player, kind, input_hash), group in sorted(self.groups.items()):
            if len(group.public_fingerprints) < 2:
                continue
            samples = sorted(group.public_fingerprints.items())
            base_hash, base = samples[0]
            alternatives = [
                {
                    "public_fingerprint": fingerprint,
                    "records": sample.count,
                    "locator": dict(sample.locator),
                }
                for fingerprint, sample in samples[1:]
            ]
            collisions.append(
                {
                    "input_hash": input_hash,
                    "observation_schema": schema,
                    "acting_player": player,
                    "decision_kind": kind,
                    "records": group.count,
                    "public_fingerprint_count": len(samples),
                    "base": {
                        "public_fingerprint": base_hash,
                        "records": base.count,
                        "locator": dict(base.locator),
                    },
                    "alternatives": alternatives,
                    "requires_public_replay_hydration": True,
                    "actionable": True,
                }
            )
        return {
            "schema_version": ENCODING_COLLISION_AUDIT_SCHEMA_VERSION,
            "expected_observation_schema": self.expected_observation_schema,
            "whitelist": {
                "version": WHITELIST_VERSION,
                "rules": sorted(DELIBERATE_ABSTRACTION_WHITELIST),
                "matched_pair_counts": {},
                "applied": False,
                "reason": "compact sketches require public replay hydration before whitelist adjudication",
            },
            "corpus": dict(corpus),
            "records_scanned": self.records_scanned,
            "input_group_count": len(self.groups),
            "collision_group_count": len(collisions),
            "actionable_collision_group_count": len(collisions),
            "collision_groups": collisions,
        }


def audit_public_decision_corpus(
    path: Path,
    *,
    max_decisions: int = 100_000,
    start_decision: int = 0,
    expected_observation_schema: str = OBSERVATION_SCHEMA_VERSION_V3,
) -> dict[str, Any]:
    """Stream a bounded public corpus and return collision evidence with provenance."""

    stream: PublicDecisionCorpusStream = open_public_decision_corpus(
        path, max_decisions=max_decisions, start_decision=start_decision
    )
    audit = EncodingCollisionAudit(expected_observation_schema=expected_observation_schema)
    for record in stream.iter_decisions():
        audit.add(record)
    return audit.to_json_dict(
        corpus={
            "manifest_schema_version": stream.manifest.get("schema_version"),
            "manifest_sha256": canonical_json_sha256(stream.manifest),
            "selected_decision_count": stream.selected_decision_count,
            "selected_content_sha256": stream.selected_content_sha256,
            "selected_decision_start": start_decision,
            "selected_decision_limit": max_decisions,
        }
    )


def audit_collision_sketches(
    paths: Iterable[Path],
    *,
    max_decisions: int = 100_000,
    start_decision: int = 0,
    expected_observation_schema: str = OBSERVATION_SCHEMA_VERSION_V3,
) -> dict[str, Any]:
    """Audit multiple compact sketch shards in stable path/record order."""

    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive.")
    if start_decision < 0:
        raise ValueError("start_decision must be non-negative.")
    selected = 0
    source_index = 0
    selected_digest = hashlib.sha256()
    manifest_hashes: list[str] = []
    capture_manifest_hash: str | None = None
    seen_decision_ids: set[str] = set()
    audit = EncodingCollisionSketchAudit(expected_observation_schema=expected_observation_schema)
    sorted_paths = tuple(sorted(Path(path) for path in paths))
    if not sorted_paths:
        raise ValueError("collision sketch audit requires at least one sketch path.")
    sources: list[tuple[Mapping[str, Any], Iterator[CollisionSketchRecord]]] = []
    for path in sorted_paths:
        manifest, records = _iter_collision_sketch(path)
        manifest_hashes.append(canonical_json_sha256(manifest))
        current_capture_manifest_hash = str(manifest["capture_manifest_sha256"])
        if capture_manifest_hash is None:
            capture_manifest_hash = current_capture_manifest_hash
        elif capture_manifest_hash != current_capture_manifest_hash:
            raise ValueError("collision sketches have incompatible immutable capture manifests.")
        sources.append((manifest, records))
    for _manifest, records in sources:
        for record in records:
            if record.decision_id in seen_decision_ids:
                raise ValueError(f"collision sketches contain duplicate decision_id {record.decision_id!r}.")
            seen_decision_ids.add(record.decision_id)
            if source_index < start_decision:
                source_index += 1
                continue
            if selected >= max_decisions:
                break
            source_index += 1
            selected += 1
            selected_digest.update(_canonical_json_line(record.to_dict()))
            audit.add(record)
        if selected >= max_decisions:
            break
    return audit.to_json_dict(
        corpus={
            "source_kind": "collision-sketch",
            "sketch_count": len(sorted_paths),
            "sketch_manifest_sha256": sorted(manifest_hashes),
            "capture_manifest_sha256": capture_manifest_hash,
            "selected_decision_count": selected,
            "selected_content_sha256": selected_digest.hexdigest(),
            "selected_decision_start": start_decision,
            "selected_decision_limit": max_decisions,
        }
    )


def _iter_collision_sketch(path: Path) -> tuple[Mapping[str, Any], Iterator[CollisionSketchRecord]]:
    if not path.is_file():
        raise ValueError(f"collision sketch does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid collision sketch JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"collision sketch line must be an object: {path}:{line_number}")
            manifest = _validated_collision_sketch_manifest(payload)
            break
        else:
            raise ValueError(f"collision sketch is empty: {path}")

    def records() -> Iterator[CollisionSketchRecord]:
        with path.open(encoding="utf-8") as handle:
            first_record = True
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid collision sketch JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(payload, Mapping):
                    raise ValueError(f"collision sketch line must be an object: {path}:{line_number}")
                if first_record:
                    if _validated_collision_sketch_manifest(payload) != manifest:
                        raise ValueError(f"collision sketch manifest changed while reading: {path}")
                    first_record = False
                    continue
                yield CollisionSketchRecord.from_dict(payload)

    return manifest, records()


def _canonical_json_line(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
