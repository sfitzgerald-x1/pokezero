"""Validated, immutable Gen 3 generator priors in source-variant order.

No consumer default is selected here: a caller must supply the generator's
team context. How to marginalize hidden team-generation context is a separate
policy decision. This module does not change world or feature production.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


SCHEMA = "gen3-generator-weights-v1"
SHOWDOWN_PIN = "f76228a1354b5d0f307ca2d16101294ad3a2308b"
EXECUTED_PIN = "a27248481987e94a95cd2a1261ec7f2900b097c84bc1b670ab85f92c0db63f78"
CONTEXT_KEYS = ("rain", "sun", "statusCure", "rapidSpin")


def digest(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def source_manifest(source):
    return {
        "source_hash": source.metadata.source_hash,
        "species": {
            key: [dict(variant_id=v.variant_id, species=v.species, level=v.level,
                       moves=list(v.moves), ability=v.ability, item=v.item) for v in universe.variants]
            for key, universe in source.universes.items()
        },
    }


def context_index(team_context):
    spikes = team_context.get("spikes", 0)
    if type(spikes) not in (bool, int) or spikes not in (0, 1):
        raise ValueError("spikes >= 2 is unreachable in pinned randomTeam")
    index = 0
    for bit, key in enumerate(CONTEXT_KEYS):
        value = team_context.get(key, False)
        if type(value) not in (bool, int) or value not in (0, 1):
            raise ValueError(f"invalid generator context: {key}")
        index |= int(value) << bit
    return index


@dataclass(frozen=True)
class SpeciesWeights:
    denominator: int
    variant_ids: tuple[str, ...]
    contexts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GeneratorWeights:
    provenance: str
    source_hash: str
    species: Mapping[str, SpeciesWeights]
    variant_manifest_sha256: str | None = None

    def weights(self, species: str, *, team_context: Mapping[str, object]) -> tuple[int, ...]:
        """Exact unnormalized integer prior; requires an explicit context."""
        return self.species[species].contexts[context_index(team_context)]


def load_generator_weights(source, path: Path | str | None = None) -> GeneratorWeights:
    path = Path(path) if path is not None else Path(__file__).with_name("data") / "gen3_generator_weights.json"
    payload = json.loads(path.read_text())
    table_hash = payload.pop("sha256")
    if table_hash != digest(payload):
        raise ValueError("generator table hash mismatch")
    if (payload["schema"] != SCHEMA or payload["showdown_commit"] != SHOWDOWN_PIN
            or payload["executed_sha256"] != EXECUTED_PIN or payload["complete"] is not True
            or payload["probability_model"] != "independent-uniform-api-choices"
            or tuple(payload["context_keys"]) != CONTEXT_KEYS):
        raise ValueError("generator table provenance mismatch")
    manifest = source_manifest(source)
    if payload["source_hash"] != source.metadata.source_hash or payload["variant_manifest_sha256"] != digest(manifest):
        raise ValueError("generator table source/variant binding mismatch")
    if payload["species"].keys() != source.universes.keys():
        raise ValueError("generator table species coverage mismatch")
    species = {}
    for key, row in payload["species"].items():
        ids = tuple(v.variant_id for v in source.universes[key].variants)
        if not 0 < len(ids) <= 64 or tuple(row["variant_ids"]) != ids:
            raise ValueError(f"generator variant order mismatch: {key}")
        denominator = _integer(row["denominator"], positive=True)
        if set(row["contexts"]) != {str(i) for i in range(16)}:
            raise ValueError(f"generator context coverage mismatch: {key}")
        contexts = tuple(tuple(_integer(w) for w in row["contexts"][str(i)]) for i in range(16))
        if any(len(weights) != len(ids) or sum(weights) != denominator for weights in contexts):
            raise ValueError(f"generator probability mass mismatch: {key}")
        species[key] = SpeciesWeights(denominator, ids, contexts)
    return GeneratorWeights(table_hash, payload["source_hash"], MappingProxyType(species), digest(manifest))


def _integer(value, *, positive=False):
    if not isinstance(value, str) or not value.isascii() or not value.isdigit() or str(int(value)) != value:
        raise ValueError("generator table integers must be canonical decimal strings")
    integer = int(value)
    if positive and integer == 0:
        raise ValueError("generator denominator must be positive")
    return integer
