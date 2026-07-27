"""Representative decision corpus for the Phase-A timing lattice (plan D3/A2).

``pokezero.engine-mcts-timing-corpus.v1`` holds exactly N legal PokeZero
decisions replayed from held-out FoulPlay games. It deliberately does NOT reuse
``public-decision-corpus.v1``: that artifact omits the request-derived action
candidates and legal mask, and a timing row must exercise the same root-action
mapping the live policy performs.

Privacy is a schema property, not a convention: a record carries the acting
player's public event prefix, its own request-derived candidates, seeds, and the
public belief inputs — never the opponent's request or hidden team data.

Stratification (plan A2) is for COVERAGE ONLY; it does not create a dynamic
policy. Strata may overlap, and the manifest records the deterministic held-out
seed range, the selection algorithm, and the count in every bucket, so changing
any of them changes the corpus hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

TIMING_CORPUS_SCHEMA_VERSION = "pokezero.engine-mcts-timing-corpus.v1"
DEFAULT_DECISION_COUNT = 256

# Fields a record must carry (plan A2). Kept explicit so the schema hash moves
# when the contract moves.
TIMING_CORPUS_SCHEMA_DESCRIPTION = {
    "schema_version": TIMING_CORPUS_SCHEMA_VERSION,
    "record_types": ("manifest", "decision"),
    "decision_fields": (
        "decision_id",
        "battle_id",
        "seat",
        "turn_index",
        "team_seed",
        "battle_seed",
        "bot_rng_seed",
        "event_prefix",
        "action_candidates",
        "legal_action_mask",
        "public_belief_inputs",
        "strata",
    ),
    "privacy": (
        "acting-seat public event prefix, its own request-derived candidates/mask, "
        "seeds, and public belief inputs only; no opponent request or hidden team data"
    ),
}
TIMING_CORPUS_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(TIMING_CORPUS_SCHEMA_DESCRIPTION, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

# Plan A2 strata. Every axis is derived from PUBLIC state so a record can be
# labeled without touching hidden information.
REMAINING_BUCKETS = ("remaining_6", "remaining_4", "remaining_2", "remaining_1")
HP_BUCKETS = ("hp_high", "hp_medium", "hp_low")
BOOST_BUCKETS = ("boost_none", "boost_offensive", "boost_defensive")
REQUEST_BUCKETS = ("request_move", "request_forced_switch")
UNCERTAINTY_BUCKETS = ("uncertainty_low", "uncertainty_high")
PHASE_BUCKETS = ("phase_early", "phase_middle", "phase_late")
STRATA_AXES = (
    REMAINING_BUCKETS,
    HP_BUCKETS,
    BOOST_BUCKETS,
    REQUEST_BUCKETS,
    UNCERTAINTY_BUCKETS,
    PHASE_BUCKETS,
)


class CorpusError(RuntimeError):
    """Terminal corpus contract failure."""


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def remaining_bucket(remaining: int) -> str:
    """Bucket by living Pokemon on the acting side (6/4/2/1 in the plan)."""
    if remaining <= 1:
        return "remaining_1"
    if remaining <= 2:
        return "remaining_2"
    if remaining <= 4:
        return "remaining_4"
    return "remaining_6"


def hp_bucket(team_hp_fraction: float) -> str:
    if team_hp_fraction >= 2.0 / 3.0:
        return "hp_high"
    if team_hp_fraction >= 1.0 / 3.0:
        return "hp_medium"
    return "hp_low"


def boost_bucket(boosts: Mapping[str, int] | None) -> str:
    """Offensive wins ties: an offensive boost changes the search's threat model
    more than a defensive one, and the plan only asks for coverage."""
    if not boosts:
        return "boost_none"
    offensive = sum(int(boosts.get(stat, 0)) for stat in ("attack", "special_attack", "speed"))
    defensive = sum(int(boosts.get(stat, 0)) for stat in ("defense", "special_defense"))
    if offensive > 0:
        return "boost_offensive"
    if defensive > 0:
        return "boost_defensive"
    return "boost_none"


def request_bucket(forced_switch: bool) -> str:
    return "request_forced_switch" if forced_switch else "request_move"


def uncertainty_bucket(hidden_world_count: int, *, high_threshold: int = 2) -> str:
    """Low/high hidden-world uncertainty by the number of distinct unresolved
    opponent possibilities the belief inputs admit."""
    return "uncertainty_high" if hidden_world_count >= high_threshold else "uncertainty_low"


def phase_bucket(turn_index: int, *, early_max: int = 8, middle_max: int = 20) -> str:
    if turn_index <= early_max:
        return "phase_early"
    if turn_index <= middle_max:
        return "phase_middle"
    return "phase_late"


@dataclass(frozen=True)
class TimingDecisionRecord:
    """One replayable decision. ``event_prefix`` is replayed to warm the
    incremental fold before the decision timer starts (plan A2)."""

    decision_id: str
    battle_id: str
    seat: str
    turn_index: int
    team_seed: int
    battle_seed: int
    bot_rng_seed: int
    event_prefix: tuple[str, ...]
    action_candidates: tuple[Mapping[str, Any], ...]
    legal_action_mask: tuple[bool, ...]
    public_belief_inputs: Mapping[str, Any]
    strata: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.seat not in {"p1", "p2"}:
            raise ValueError("seat must be p1 or p2.")
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")
        if not any(self.legal_action_mask):
            raise ValueError("a timing decision must have at least one legal action.")
        if not self.action_candidates:
            raise ValueError(
                "action_candidates is required: the request-derived mapping is exactly "
                "what public-decision-corpus.v1 lacks and what this study must time."
            )
        forbidden = {"opponent_request", "hidden_team", "opponent_private"}
        leaked = forbidden.intersection(self.public_belief_inputs)
        if leaked:
            raise ValueError(
                f"public_belief_inputs carries non-public keys {sorted(leaked)}; the timing "
                "corpus is an information-set artifact."
            )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["record_type"] = "decision"
        payload["event_prefix"] = list(self.event_prefix)
        payload["action_candidates"] = [dict(candidate) for candidate in self.action_candidates]
        payload["legal_action_mask"] = list(self.legal_action_mask)
        payload["strata"] = list(self.strata)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TimingDecisionRecord":
        return cls(
            decision_id=str(payload["decision_id"]),
            battle_id=str(payload["battle_id"]),
            seat=str(payload["seat"]),
            turn_index=int(payload["turn_index"]),
            team_seed=int(payload["team_seed"]),
            battle_seed=int(payload["battle_seed"]),
            bot_rng_seed=int(payload["bot_rng_seed"]),
            event_prefix=tuple(payload["event_prefix"]),
            action_candidates=tuple(dict(item) for item in payload["action_candidates"]),
            legal_action_mask=tuple(bool(value) for value in payload["legal_action_mask"]),
            public_belief_inputs=dict(payload["public_belief_inputs"]),
            strata=tuple(payload["strata"]),
        )


def label_strata(
    *,
    remaining: int,
    team_hp_fraction: float,
    boosts: Mapping[str, int] | None,
    forced_switch: bool,
    hidden_world_count: int,
    turn_index: int,
) -> tuple[str, ...]:
    """Deterministic public-state labels; strata may overlap by design."""
    return (
        remaining_bucket(remaining),
        hp_bucket(team_hp_fraction),
        boost_bucket(boosts),
        request_bucket(forced_switch),
        uncertainty_bucket(hidden_world_count),
        phase_bucket(turn_index),
    )


def select_stratified(
    records: Sequence[TimingDecisionRecord],
    *,
    count: int = DEFAULT_DECISION_COUNT,
) -> tuple[TimingDecisionRecord, ...]:
    """Deterministically choose ``count`` records with the flattest strata coverage.

    Round-robin over every bucket of every axis, taking the lowest-``decision_id``
    unused record in the currently least-covered bucket. Deterministic in the
    input order and total by construction — the same candidate pool always yields
    the same corpus, which is what makes the corpus hash meaningful.
    """
    if count <= 0:
        raise ValueError("count must be positive.")
    if len(records) < count:
        raise CorpusError(
            f"candidate pool has {len(records)} decisions, need {count}; widen the held-out "
            "seed range rather than reusing decisions."
        )
    pool = sorted(records, key=lambda record: record.decision_id)
    chosen: list[TimingDecisionRecord] = []
    used: set[str] = set()
    coverage: dict[str, int] = {
        bucket: 0 for axis in STRATA_AXES for bucket in axis
    }
    while len(chosen) < count:
        # Target the least-covered bucket that still has an unused record.
        candidates = [
            (coverage[bucket], bucket)
            for bucket in coverage
            if any(bucket in record.strata and record.decision_id not in used for record in pool)
        ]
        if not candidates:  # pool exhausted of labeled records; take the rest in order
            for record in pool:
                if record.decision_id not in used:
                    chosen.append(record)
                    used.add(record.decision_id)
                    if len(chosen) == count:
                        break
            break
        _, target = min(candidates)
        for record in pool:
            if target in record.strata and record.decision_id not in used:
                chosen.append(record)
                used.add(record.decision_id)
                for bucket in record.strata:
                    coverage[bucket] += 1
                break
    return tuple(chosen[:count])


def bucket_counts(records: Iterable[TimingDecisionRecord]) -> dict[str, int]:
    counts = {bucket: 0 for axis in STRATA_AXES for bucket in axis}
    for record in records:
        for bucket in record.strata:
            if bucket in counts:
                counts[bucket] += 1
    return counts


@dataclass(frozen=True)
class TimingCorpusManifest:
    """Everything that makes the corpus reproducible; any change moves the hash."""

    held_out_seed_start: int
    held_out_seed_end: int
    selection_algorithm: str
    decision_count: int
    bucket_counts: Mapping[str, int]
    corpus_sha256: str
    schema_version: str = TIMING_CORPUS_SCHEMA_VERSION
    schema_sha256: str = TIMING_CORPUS_SCHEMA_SHA256

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["record_type"] = "manifest"
        payload["bucket_counts"] = dict(self.bucket_counts)
        return payload


def build_corpus(
    records: Sequence[TimingDecisionRecord],
    *,
    held_out_seed_start: int,
    held_out_seed_end: int,
    count: int = DEFAULT_DECISION_COUNT,
    selection_algorithm: str = "least-covered-bucket-round-robin.v1",
) -> tuple[TimingCorpusManifest, tuple[TimingDecisionRecord, ...]]:
    selected = select_stratified(records, count=count)
    corpus_sha256 = canonical_json_sha256([record.to_payload() for record in selected])
    manifest = TimingCorpusManifest(
        held_out_seed_start=held_out_seed_start,
        held_out_seed_end=held_out_seed_end,
        selection_algorithm=selection_algorithm,
        decision_count=len(selected),
        bucket_counts=bucket_counts(selected),
        corpus_sha256=corpus_sha256,
    )
    return manifest, selected


def write_corpus(
    path: str | Path,
    manifest: TimingCorpusManifest,
    records: Sequence[TimingDecisionRecord],
) -> None:
    """Atomic write: a partially written corpus must never be adopted."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest.to_payload(), sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record.to_payload(), sort_keys=True) + "\n")
    temp.replace(target)


def read_corpus(
    path: str | Path,
) -> tuple[TimingCorpusManifest, tuple[TimingDecisionRecord, ...]]:
    """Read + fail closed on hash or schema drift."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise CorpusError(f"empty corpus: {path}")
    header = json.loads(lines[0])
    if header.get("record_type") != "manifest":
        raise CorpusError(f"{path}: first record must be the manifest.")
    if header.get("schema_version") != TIMING_CORPUS_SCHEMA_VERSION:
        raise CorpusError(
            f"{path}: schema {header.get('schema_version')!r} != {TIMING_CORPUS_SCHEMA_VERSION}"
        )
    if header.get("schema_sha256") != TIMING_CORPUS_SCHEMA_SHA256:
        raise CorpusError(
            f"{path}: corpus was built against a different record contract "
            f"({header.get('schema_sha256')} != {TIMING_CORPUS_SCHEMA_SHA256})."
        )
    records = tuple(
        TimingDecisionRecord.from_payload(json.loads(line)) for line in lines[1:] if line.strip()
    )
    manifest = TimingCorpusManifest(
        held_out_seed_start=int(header["held_out_seed_start"]),
        held_out_seed_end=int(header["held_out_seed_end"]),
        selection_algorithm=str(header["selection_algorithm"]),
        decision_count=int(header["decision_count"]),
        bucket_counts=dict(header["bucket_counts"]),
        corpus_sha256=str(header["corpus_sha256"]),
    )
    actual = canonical_json_sha256([record.to_payload() for record in records])
    if actual != manifest.corpus_sha256:
        raise CorpusError(
            f"{path}: corpus content hash {actual} != manifest {manifest.corpus_sha256}; "
            "a timing lattice must not run on a mutated corpus."
        )
    if len(records) != manifest.decision_count:
        raise CorpusError(
            f"{path}: {len(records)} records but manifest declares {manifest.decision_count}."
        )
    return manifest, records
