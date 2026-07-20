"""Detect public-state distinctions collapsed by an observation encoding.

The audit intentionally consumes only :mod:`public_decision_corpus` records.
Those records contain the acting player's model input plus public replay facts;
they exclude the opponent's request and all other private simulator state.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .observation import OBSERVATION_SCHEMA_VERSION_V3
from .public_decision_corpus import (
    PublicDecisionCorpusStream,
    PublicDecisionRecord,
    canonical_json_sha256,
    open_public_decision_corpus,
)


ENCODING_COLLISION_AUDIT_SCHEMA_VERSION = "pokezero.encoding-collision-audit.v1"
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
