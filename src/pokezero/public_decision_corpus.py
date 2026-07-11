"""Public-only replay-prefix corpus for adaptive root-search profiling.

The corpus deliberately does not reuse rollout serialization. Rollouts retain both
players' decision observations, while this artifact is an information-set audit
surface: it contains the acting player's observation/history, public resolved
actions, and the public belief view only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .actions import ACTION_COUNT
from .observation import PokeZeroObservationV0
from .trajectory import BattleTrajectory


PUBLIC_DECISION_CORPUS_SCHEMA_VERSION = "pokezero.public-decision-corpus.v1"
PUBLIC_DECISION_CORPUS_SCHEMA_DESCRIPTION = {
    "schema_version": PUBLIC_DECISION_CORPUS_SCHEMA_VERSION,
    "record_types": ("manifest", "decision"),
    "decision_fields": (
        "decision_id",
        "battle_id",
        "seed",
        "format_id",
        "acting_player",
        "turn_index",
        "recorded_action_index",
        "observation",
        "history",
        "current_legal_action_mask",
        "public_resolved_action_rounds",
        "public_belief_view",
    ),
    "privacy": "acting-player observation/history plus public resolved action rounds only",
}
PUBLIC_DECISION_CORPUS_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(PUBLIC_DECISION_CORPUS_SCHEMA_DESCRIPTION, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

_FORBIDDEN_KEY_FRAGMENTS = (
    "request",
    "opponent_observation",
    "requested_observations",
    "opponent_legal",
    "private_observation",
    "raw_choice",
)
_PUBLIC_ACTION_IDENTIFIER_FORBIDDEN_FIELDS = frozenset(
    {
        "action_index",
        "move_slot",
        "switch_slot",
        "team_index",
        "raw_choice",
        "slot",
    }
)
_PUBLIC_OPPONENT_POKEMON_FIELDS = frozenset(
    {"ident", "showdown_slot", "species", "condition", "hp_fraction", "status", "fainted", "active", "details"}
)
_SELF_POKEMON_FIELDS = frozenset(
    {
        "ident",
        "showdown_slot",
        "species",
        "condition",
        "hp_fraction",
        "status",
        "fainted",
        "active",
        "details",
        "moves",
        "ability",
        "item",
        "stats",
    }
)
_ACTING_PLAYER_STATE_FIELDS = frozenset(
    {
        "showdown_slot",
        "opponent_showdown_slot",
        "self_side_conditions",
        "opponent_side_conditions",
        "self_side_condition_counts",
        "opponent_side_condition_counts",
        "weather",
        "turn_number",
        "self_active_boosts",
        "opponent_active_boosts",
        "self_active_volatiles",
        "opponent_active_volatiles",
        "self_future_sight_turns",
        "opponent_future_sight_turns",
        "self_toxic_stage",
        "opponent_toxic_stage",
        "self_active",
        "opponent_active",
        "self_team",
        "opponent_team",
        "recent_public_events",
        "transition_token_count",
        "self_sleep_clause_used",
        "opponent_sleep_clause_used",
        "weather_turns_remaining",
        "weather_permanent",
        "self_wish_pending",
        "opponent_wish_pending",
        "action_candidates",
    }
)


def canonical_json_sha256(payload: object) -> str:
    """Return the stable SHA256 used for corpus, configuration, and record IDs."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PublicObservation:
    """An encoded acting-player observation plus only deterministic public metadata."""

    schema_version: str
    categorical_ids: tuple[tuple[int, ...], ...]
    numeric_features: tuple[tuple[float, ...], ...]
    token_type_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    legal_action_mask: tuple[bool, ...]
    acting_player_state: Mapping[str, Any]

    @classmethod
    def from_observation(cls, observation: PokeZeroObservationV0) -> "PublicObservation":
        return cls(
            schema_version=observation.schema_version,
            categorical_ids=tuple(tuple(int(value) for value in row) for row in observation.categorical_ids),
            numeric_features=tuple(tuple(float(value) for value in row) for row in observation.numeric_features),
            token_type_ids=tuple(int(value) for value in observation.token_type_ids),
            attention_mask=tuple(bool(value) for value in observation.attention_mask),
            legal_action_mask=tuple(bool(value) for value in observation.legal_action_mask),
            acting_player_state=_public_acting_player_state(observation.metadata),
        )

    def to_observation(self, *, belief_view: Mapping[str, Any]) -> PokeZeroObservationV0:
        """Rehydrate the model input without adding a request or opponent-private data."""

        return PokeZeroObservationV0(
            categorical_ids=self.categorical_ids,
            numeric_features=self.numeric_features,
            token_type_ids=self.token_type_ids,
            attention_mask=self.attention_mask,
            legal_action_mask=self.legal_action_mask,
            metadata={**dict(self.acting_player_state), "belief_view": dict(belief_view)},
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "categorical_ids": [list(row) for row in self.categorical_ids],
            "numeric_features": [list(row) for row in self.numeric_features],
            "token_type_ids": list(self.token_type_ids),
            "attention_mask": list(self.attention_mask),
            "legal_action_mask": list(self.legal_action_mask),
            "acting_player_state": _json_value(self.acting_player_state),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicObservation":
        _reject_forbidden_payload(payload)
        state = _mapping(payload.get("acting_player_state"), "acting_player_state")
        unknown_state_fields = set(state) - _ACTING_PLAYER_STATE_FIELDS
        if unknown_state_fields:
            raise ValueError(f"public observation has unsupported state fields: {sorted(unknown_state_fields)}")
        legal_action_mask = tuple(bool(value) for value in _sequence(payload.get("legal_action_mask"), "legal_action_mask"))
        if len(legal_action_mask) != ACTION_COUNT:
            raise ValueError(f"public observation legal_action_mask must contain {ACTION_COUNT} values.")
        return cls(
            schema_version=_nonempty_string(payload.get("schema_version"), "schema_version"),
            categorical_ids=tuple(
                tuple(int(value) for value in _sequence(row, "categorical_ids row"))
                for row in _sequence(payload.get("categorical_ids"), "categorical_ids")
            ),
            numeric_features=tuple(
                tuple(float(value) for value in _sequence(row, "numeric_features row"))
                for row in _sequence(payload.get("numeric_features"), "numeric_features")
            ),
            token_type_ids=tuple(int(value) for value in _sequence(payload.get("token_type_ids"), "token_type_ids")),
            attention_mask=tuple(bool(value) for value in _sequence(payload.get("attention_mask"), "attention_mask")),
            legal_action_mask=legal_action_mask,
            acting_player_state=dict(state),
        )


@dataclass(frozen=True)
class PublicActorObservation:
    """One actor observation annotated with its source decision round."""

    turn_index: int
    observation: PublicObservation

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("actor observation turn_index must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {"turn_index": self.turn_index, "observation": self.observation.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicActorObservation":
        return cls(
            turn_index=int(payload.get("turn_index")),
            observation=PublicObservation.from_dict(_mapping(payload.get("observation"), "actor observation")),
        )


@dataclass(frozen=True)
class PublicActionIdentifier:
    """A resolved public action without a request-local slot or action index.

    Index resolution is intentionally deferred to a sampled belief world. This
    prevents a persisted opponent action from revealing request-local ordering.
    """

    kind: str
    move_id: str | None = None
    switched_species: str | None = None
    event_id: str | None = None

    def __post_init__(self) -> None:
        expected_field = {
            "move": "move_id",
            "switch": "switched_species",
            "event": "event_id",
        }.get(self.kind)
        if expected_field is None:
            raise ValueError("public action identifier kind must be move, switch, or event.")
        values = {
            "move_id": self.move_id,
            "switched_species": self.switched_species,
            "event_id": self.event_id,
        }
        if not isinstance(values[expected_field], str) or not values[expected_field]:
            raise ValueError(f"public {self.kind} action identifier requires {expected_field}.")
        if any(value is not None for field, value in values.items() if field != expected_field):
            raise ValueError("public action identifier must contain exactly one identifier field.")

    def to_dict(self) -> dict[str, str]:
        if self.kind == "move":
            return {"kind": "move", "move_id": str(self.move_id)}
        if self.kind == "switch":
            return {"kind": "switch", "switched_species": str(self.switched_species)}
        return {"kind": "event", "event_id": str(self.event_id)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicActionIdentifier":
        _reject_public_action_identifier_fields(payload)
        kind = _nonempty_string(payload.get("kind"), "public action kind")
        expected_keys = {
            "move": {"kind", "move_id"},
            "switch": {"kind", "switched_species"},
            "event": {"kind", "event_id"},
        }.get(kind)
        if expected_keys is None or set(payload) != expected_keys:
            raise ValueError("public action identifier has unsupported fields.")
        return cls(
            kind=kind,
            move_id=payload.get("move_id"),
            switched_species=payload.get("switched_species"),
            event_id=payload.get("event_id"),
        )


@dataclass(frozen=True)
class PublicResolvedActionRound:
    """Resolved public action identifiers from one past decision round."""

    turn_index: int
    actions: Mapping[str, PublicActionIdentifier]

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("public action round turn_index must be non-negative.")
        if not self.actions:
            raise ValueError("public action round must contain at least one resolved action.")
        normalized = {
            str(player): action if isinstance(action, PublicActionIdentifier) else PublicActionIdentifier.from_dict(action)
            for player, action in sorted(self.actions.items())
        }
        object.__setattr__(self, "actions", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {"turn_index": self.turn_index, "actions": {player: action.to_dict() for player, action in self.actions.items()}}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicResolvedActionRound":
        return cls(
            turn_index=int(payload.get("turn_index")),
            actions={
                str(player): PublicActionIdentifier.from_dict(_mapping(action, "public action identifier"))
                for player, action in _mapping(payload.get("actions"), "actions").items()
            },
        )


@dataclass(frozen=True)
class PublicDecisionRecord:
    """One decision-point replay prefix from the acting player's information set."""

    decision_id: str
    battle_id: str
    seed: int
    format_id: str
    acting_player: str
    turn_index: int
    recorded_action_index: int
    observation: PublicObservation
    history: tuple[PublicActorObservation, ...]
    current_legal_action_mask: tuple[bool, ...]
    public_resolved_action_rounds: tuple[PublicResolvedActionRound, ...]
    public_belief_view: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.acting_player not in {"p1", "p2"}:
            raise ValueError("acting_player must be p1 or p2.")
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")
        if not 0 <= self.recorded_action_index < ACTION_COUNT:
            raise ValueError(f"recorded_action_index must be between 0 and {ACTION_COUNT - 1}.")
        if len(self.current_legal_action_mask) != ACTION_COUNT:
            raise ValueError(f"current_legal_action_mask must contain {ACTION_COUNT} values.")
        if tuple(self.observation.legal_action_mask) != self.current_legal_action_mask:
            raise ValueError("current_legal_action_mask must match observation.legal_action_mask.")
        if not self.current_legal_action_mask[self.recorded_action_index]:
            raise ValueError("recorded_action_index must be legal in the acting-player observation.")
        history_turns = tuple(entry.turn_index for entry in self.history)
        if history_turns != tuple(sorted(set(history_turns))) or any(turn >= self.turn_index for turn in history_turns):
            raise ValueError("history must contain unique source turns before the profiled turn.")
        if any(round_.turn_index >= self.turn_index for round_ in self.public_resolved_action_rounds):
            raise ValueError("public resolved action rounds must precede the profiled turn.")
        if [round_.turn_index for round_ in self.public_resolved_action_rounds] != list(range(len(self.public_resolved_action_rounds))):
            raise ValueError("public resolved action rounds must be contiguous from turn zero.")
        _reject_forbidden_payload(self.public_belief_view)

    def observations(self) -> tuple[PokeZeroObservationV0, ...]:
        return tuple(
            observation.to_observation(belief_view=self.public_belief_view)
            for observation in (*(entry.observation for entry in self.history), self.observation)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "decision",
            "schema_version": PUBLIC_DECISION_CORPUS_SCHEMA_VERSION,
            "decision_id": self.decision_id,
            "battle_id": self.battle_id,
            "seed": self.seed,
            "format_id": self.format_id,
            "acting_player": self.acting_player,
            "turn_index": self.turn_index,
            "recorded_action_index": self.recorded_action_index,
            "observation": self.observation.to_dict(),
            "history": [observation.to_dict() for observation in self.history],
            "current_legal_action_mask": list(self.current_legal_action_mask),
            "public_resolved_action_rounds": [round_.to_dict() for round_ in self.public_resolved_action_rounds],
            "public_belief_view": _json_value(self.public_belief_view),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicDecisionRecord":
        _reject_forbidden_payload(payload)
        expected_keys = set(PUBLIC_DECISION_CORPUS_SCHEMA_DESCRIPTION["decision_fields"]) | {"record_type", "schema_version"}
        unknown_keys = set(payload) - expected_keys
        if unknown_keys:
            raise ValueError(f"public decision record has unsupported fields: {sorted(unknown_keys)}")
        if payload.get("record_type") != "decision" or payload.get("schema_version") != PUBLIC_DECISION_CORPUS_SCHEMA_VERSION:
            raise ValueError("not a public decision corpus v1 decision record.")
        record = cls(
            decision_id=_nonempty_string(payload.get("decision_id"), "decision_id"),
            battle_id=_nonempty_string(payload.get("battle_id"), "battle_id"),
            seed=int(payload.get("seed")),
            format_id=_nonempty_string(payload.get("format_id"), "format_id"),
            acting_player=_nonempty_string(payload.get("acting_player"), "acting_player"),
            turn_index=int(payload.get("turn_index")),
            recorded_action_index=int(payload.get("recorded_action_index")),
            observation=PublicObservation.from_dict(_mapping(payload.get("observation"), "observation")),
            history=tuple(
                PublicActorObservation.from_dict(_mapping(item, "history item"))
                for item in _sequence(payload.get("history"), "history")
            ),
            current_legal_action_mask=tuple(
                bool(value) for value in _sequence(payload.get("current_legal_action_mask"), "current_legal_action_mask")
            ),
            public_resolved_action_rounds=tuple(
                PublicResolvedActionRound.from_dict(_mapping(item, "public_resolved_action_round"))
                for item in _sequence(payload.get("public_resolved_action_rounds"), "public_resolved_action_rounds")
            ),
            public_belief_view=dict(_mapping(payload.get("public_belief_view"), "public_belief_view")),
        )
        expected_id = public_decision_id(record)
        if record.decision_id != expected_id:
            raise ValueError("public decision record checksum does not match its public payload.")
        return record


@dataclass(frozen=True)
class PublicDecisionCorpus:
    manifest: Mapping[str, Any]
    decisions: tuple[PublicDecisionRecord, ...]
    path: Path | None = None
    selected_content_sha256: str | None = None
    selected_decision_limit: int | None = None

    @property
    def source_file_sha256(self) -> str | None:
        """Return a raw source-file digest only when the complete file was read."""

        if self.selected_decision_limit is not None:
            return None
        if self.path is not None:
            return sha256_file(self.path)
        return canonical_json_sha256({"manifest": self.manifest, "decisions": [record.to_dict() for record in self.decisions]})

    @property
    def corpus_sha256(self) -> str | None:
        """Compatibility alias for the complete source-file digest, never a capped selection hash."""

        return self.source_file_sha256


def public_decision_id(record: PublicDecisionRecord) -> str:
    """Checksum a record excluding its own ID, so private input changes cannot affect it."""

    payload = record.to_dict()
    payload.pop("decision_id")
    return canonical_json_sha256(payload)


def public_decision_records_from_trajectory(
    trajectory: BattleTrajectory,
    *,
    acting_player: str = "p1",
) -> tuple[PublicDecisionRecord, ...]:
    """Project a full trajectory to public replay prefixes without reading opponent observations."""

    if acting_player not in {"p1", "p2"}:
        raise ValueError("acting_player must be p1 or p2.")
    steps_by_turn: dict[int, list[Any]] = {}
    own_steps = []
    for step in trajectory.steps:
        steps_by_turn.setdefault(step.turn_index, []).append(step)
        if step.player_id == acting_player:
            own_steps.append(step)
    supplied_rounds = _public_rounds_from_trajectory_metadata(trajectory.metadata)
    records: list[PublicDecisionRecord] = []
    prior_observations: list[PublicActorObservation] = []
    for step in sorted(own_steps, key=lambda item: item.turn_index):
        belief_view = _public_belief_view(step.observation.metadata)
        observation = PublicObservation.from_observation(step.observation)
        rounds = tuple(
            supplied_rounds.get(
                turn_index,
                _fallback_public_action_round(
                    turn_index,
                    steps_by_turn.get(turn_index, ()),
                    acting_player=acting_player,
                ),
            )
            for turn_index in range(step.turn_index)
        )
        prototype = PublicDecisionRecord(
            decision_id="pending",
            battle_id=trajectory.battle_id,
            seed=trajectory.seed,
            format_id=trajectory.format_id,
            acting_player=acting_player,
            turn_index=step.turn_index,
            recorded_action_index=step.action_index,
            observation=observation,
            history=tuple(prior_observations),
            current_legal_action_mask=tuple(step.legal_action_mask),
            public_resolved_action_rounds=rounds,
            public_belief_view=belief_view,
        )
        record = PublicDecisionRecord(**{**prototype.__dict__, "decision_id": public_decision_id(prototype)})
        records.append(record)
        prior_observations.append(PublicActorObservation(turn_index=step.turn_index, observation=observation))
    return tuple(records)


def _public_rounds_from_trajectory_metadata(metadata: Mapping[str, Any]) -> dict[int, PublicResolvedActionRound]:
    """Read bridge-produced public action IDs without inspecting either request."""

    payload = metadata.get("public_resolved_action_rounds") if isinstance(metadata, Mapping) else None
    if payload is None:
        return {}
    _reject_forbidden_payload(payload)
    rounds = tuple(
        PublicResolvedActionRound.from_dict(_mapping(item, "public_resolved_action_round"))
        for item in _sequence(payload, "public_resolved_action_rounds")
    )
    indexes = tuple(round_.turn_index for round_ in rounds)
    if indexes != tuple(sorted(set(indexes))):
        raise ValueError("public resolved action rounds in trajectory metadata must have unique sorted turns.")
    return {round_.turn_index: round_ for round_ in rounds}


def _fallback_public_action_round(
    turn_index: int,
    steps: Sequence[Any],
    *,
    acting_player: str,
) -> PublicResolvedActionRound:
    """Use only the actor's own observation when bridge public events are absent.

    This compatibility path never inspects an opposing observation or its action
    index. Unresolved opposing actions remain public event identifiers and will
    be rejected by replay until a sampled world can resolve a real identifier.
    """

    actions: dict[str, PublicActionIdentifier] = {}
    for step in steps:
        player = str(step.player_id)
        if player == acting_player:
            actions[player] = _acting_action_identifier(step)
        else:
            actions[player] = PublicActionIdentifier(kind="event", event_id="unresolved-public-action")
    if not actions:
        actions[acting_player] = PublicActionIdentifier(kind="event", event_id="unresolved-public-action")
    return PublicResolvedActionRound(turn_index=turn_index, actions=actions)


def _acting_action_identifier(step: Any) -> PublicActionIdentifier:
    candidates = step.observation.metadata.get("action_candidates")
    if isinstance(candidates, Sequence):
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or candidate.get("action_index") != step.action_index:
                continue
            if candidate.get("kind") == "move" and isinstance(candidate.get("move_id"), str):
                return PublicActionIdentifier(kind="move", move_id=candidate["move_id"])
            if candidate.get("kind") == "switch":
                pokemon = candidate.get("pokemon")
                if isinstance(pokemon, Mapping) and isinstance(pokemon.get("species"), str):
                    return PublicActionIdentifier(kind="switch", switched_species=pokemon["species"])
    return PublicActionIdentifier(kind="event", event_id="unresolved-public-action")


def public_corpus_manifest(
    *,
    checkpoint_sha256: str,
    belief_set_source_hash: str | None,
    capture_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a public manifest. Root noise and privileged masks are forbidden by contract."""

    if capture_config.get("opponent_legal_mask_mode", "hidden") != "hidden":
        raise ValueError("public decision corpus capture requires opponent_legal_mask_mode='hidden'.")
    if capture_config.get("root_dirichlet_alpha") not in (None, False):
        raise ValueError("public decision corpus capture requires root noise to be off.")
    safe_config = _json_value(dict(capture_config))
    return {
        "record_type": "manifest",
        "schema_version": PUBLIC_DECISION_CORPUS_SCHEMA_VERSION,
        "schema_sha256": PUBLIC_DECISION_CORPUS_SCHEMA_SHA256,
        "checkpoint_sha256": str(checkpoint_sha256),
        "belief_set_source_hash": belief_set_source_hash,
        "opponent_legal_mask_mode": "hidden",
        "root_noise": {"enabled": False, "root_dirichlet_alpha": None},
        "capture_config": safe_config,
        "capture_config_sha256": canonical_json_sha256(safe_config),
    }


class PublicDecisionCorpusWriter:
    """Append valid public decisions while keeping one immutable provenance manifest."""

    def __init__(self, path: Path, *, manifest: Mapping[str, Any], append: bool = False) -> None:
        self.path = path
        self.manifest = dict(manifest)
        self._seen_decision_ids: set[str] = set()
        if path.exists():
            if not append:
                raise FileExistsError(f"public decision corpus already exists: {path}")
            existing = load_public_decision_corpus(path)
            _require_compatible_manifest(existing.manifest, self.manifest)
            self._seen_decision_ids = {record.decision_id for record in existing.decisions}
            self._handle = path.open("a", encoding="utf-8")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("x", encoding="utf-8")
            self._write_line(self.manifest)

    def append_trajectory(self, trajectory: BattleTrajectory, *, acting_player: str = "p1") -> int:
        written = 0
        for record in public_decision_records_from_trajectory(trajectory, acting_player=acting_player):
            written += self.append(record)
        return written

    def append(self, record: PublicDecisionRecord) -> int:
        if record.decision_id in self._seen_decision_ids:
            return 0
        self._write_line(record.to_dict())
        self._seen_decision_ids.add(record.decision_id)
        return 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "PublicDecisionCorpusWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _write_line(self, payload: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        self._handle.write("\n")
        self._handle.flush()


def load_public_decision_corpus(path: Path, *, max_decisions: int | None = None) -> PublicDecisionCorpus:
    """Read a validated public-only corpus, optionally stopping at a deterministic prefix."""

    if max_decisions is not None and max_decisions <= 0:
        raise ValueError("max_decisions must be positive when provided.")

    manifest: Mapping[str, Any] | None = None
    decisions: list[PublicDecisionRecord] = []
    seen_ids: set[str] = set()
    selected_digest = hashlib.sha256() if max_decisions is not None else None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid public corpus JSON at line {line_number}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"public corpus line {line_number} must be a JSON object.")
            if payload.get("record_type") == "manifest":
                if manifest is not None or decisions:
                    raise ValueError("public corpus manifest must be the first non-empty record.")
                manifest = _validated_manifest(payload)
                if selected_digest is not None:
                    selected_digest.update(_canonical_json_line(manifest))
                continue
            if manifest is None:
                raise ValueError("public corpus is missing its manifest.")
            record = PublicDecisionRecord.from_dict(payload)
            if record.decision_id in seen_ids:
                raise ValueError(f"public corpus contains duplicate decision_id {record.decision_id!r}.")
            seen_ids.add(record.decision_id)
            decisions.append(record)
            if selected_digest is not None:
                selected_digest.update(_canonical_json_line(record.to_dict()))
            if max_decisions is not None and len(decisions) >= max_decisions:
                break
    if manifest is None:
        raise ValueError("public corpus is empty or missing its manifest.")
    return PublicDecisionCorpus(
        manifest=manifest,
        decisions=tuple(decisions),
        path=path,
        selected_content_sha256=selected_digest.hexdigest() if selected_digest is not None else None,
        selected_decision_limit=max_decisions,
    )


def _canonical_json_line(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _validated_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    _reject_forbidden_payload(payload)
    expected = {
        "record_type",
        "schema_version",
        "schema_sha256",
        "checkpoint_sha256",
        "belief_set_source_hash",
        "opponent_legal_mask_mode",
        "root_noise",
        "capture_config",
        "capture_config_sha256",
    }
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"public corpus manifest has unsupported fields: {sorted(unknown)}")
    if payload.get("record_type") != "manifest" or payload.get("schema_version") != PUBLIC_DECISION_CORPUS_SCHEMA_VERSION:
        raise ValueError("not a public decision corpus v1 manifest.")
    if payload.get("schema_sha256") != PUBLIC_DECISION_CORPUS_SCHEMA_SHA256:
        raise ValueError("public corpus schema hash does not match v1.")
    if payload.get("opponent_legal_mask_mode") != "hidden":
        raise ValueError("public corpus must use hidden opponent legal-mask mode.")
    root_noise = _mapping(payload.get("root_noise"), "root_noise")
    if root_noise.get("enabled") is not False or root_noise.get("root_dirichlet_alpha") is not None:
        raise ValueError("public corpus must record root noise as disabled.")
    capture_config = _mapping(payload.get("capture_config"), "capture_config")
    if payload.get("capture_config_sha256") != canonical_json_sha256(capture_config):
        raise ValueError("public corpus capture configuration hash does not match its payload.")
    return dict(payload)


def _require_compatible_manifest(existing: Mapping[str, Any], requested: Mapping[str, Any]) -> None:
    fields = ("schema_version", "schema_sha256", "checkpoint_sha256", "belief_set_source_hash", "opponent_legal_mask_mode")
    mismatched = [field for field in fields if existing.get(field) != requested.get(field)]
    if mismatched:
        raise ValueError(f"cannot append incompatible public corpus manifest fields: {', '.join(mismatched)}")


def _public_belief_view(metadata: Mapping[str, Any]) -> dict[str, Any]:
    value = metadata.get("belief_view") if isinstance(metadata, Mapping) else None
    belief_view = _mapping(value, "belief_view")
    _reject_forbidden_payload(belief_view)
    if belief_view.get("self_slot") not in {"p1", "p2"} or belief_view.get("opponent_slot") not in {"p1", "p2"}:
        raise ValueError("acting-player observation is missing a valid public belief view.")
    return _json_value(belief_view)


def _public_acting_player_state(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("acting-player observation metadata is required for public corpus capture.")
    state: dict[str, Any] = {}
    for field in _ACTING_PLAYER_STATE_FIELDS:
        if field not in metadata:
            continue
        value = metadata[field]
        if field in {"self_active"}:
            state[field] = _sanitize_pokemon(value, allowed=_SELF_POKEMON_FIELDS)
        elif field in {"opponent_active"}:
            state[field] = _sanitize_pokemon(value, allowed=_PUBLIC_OPPONENT_POKEMON_FIELDS)
        elif field == "self_team":
            state[field] = [_sanitize_pokemon(item, allowed=_SELF_POKEMON_FIELDS) for item in _sequence_or_empty(value)]
        elif field == "opponent_team":
            state[field] = [
                _sanitize_pokemon(item, allowed=_PUBLIC_OPPONENT_POKEMON_FIELDS)
                for item in _sequence_or_empty(value)
            ]
        elif field == "action_candidates":
            state[field] = _public_action_candidates(value)
        else:
            state[field] = _json_value(value)
    _reject_forbidden_payload(state)
    return state


def _sanitize_pokemon(value: Any, *, allowed: frozenset[str]) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = _mapping(value, "pokemon")
    return {key: _json_value(payload[key]) for key in sorted(set(payload) & allowed)}


def _public_action_candidates(value: Any) -> list[dict[str, Any]]:
    """Keep actor-visible semantic action IDs while dropping request-local slots."""

    candidates: list[dict[str, Any]] = []
    for candidate in _sequence_or_empty(value):
        if not isinstance(candidate, Mapping):
            continue
        action_index = candidate.get("action_index")
        kind = candidate.get("kind")
        if kind is None and isinstance(candidate.get("move_id"), str):
            kind = "move"
        if kind is None and isinstance(candidate.get("pokemon"), Mapping):
            kind = "switch"
        if not isinstance(action_index, int) or not isinstance(kind, str):
            continue
        public_candidate: dict[str, Any] = {
            "action_index": action_index,
            "kind": kind,
            "legal": bool(candidate.get("legal")),
        }
        if kind == "move" and isinstance(candidate.get("move_id"), str):
            public_candidate["move_id"] = candidate["move_id"]
        elif kind == "switch":
            pokemon = candidate.get("pokemon")
            if isinstance(pokemon, Mapping) and isinstance(pokemon.get("species"), str):
                public_candidate["switched_species"] = pokemon["species"]
        candidates.append(public_candidate)
    return candidates


def _reject_forbidden_payload(payload: Any) -> None:
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key).lower()
            if key == "opponent_legal_mask_mode":
                _reject_forbidden_payload(value)
                continue
            if any(fragment in key for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise ValueError(f"public corpus payload contains forbidden private field {raw_key!r}.")
            _reject_forbidden_payload(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _reject_forbidden_payload(value)


def _reject_public_action_identifier_fields(payload: Mapping[str, Any]) -> None:
    forbidden = _PUBLIC_ACTION_IDENTIFIER_FORBIDDEN_FIELDS.intersection(str(key).lower() for key in payload)
    if forbidden:
        raise ValueError(
            "public action identifier contains request-local field(s): " + ", ".join(sorted(forbidden))
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a JSON array.")
    return value


def _sequence_or_empty(value: Any) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _json_value(value: Any) -> Any:
    """Normalize immutable mappings/tuples so persisted records are canonical JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
