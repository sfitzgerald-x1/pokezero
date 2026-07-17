"""Belief-backed start-state materialization for hidden-information search."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import random
import re
from typing import Any, Mapping, Sequence

from .actions import MOVE_ACTION_COUNT, canonical_switch_action_map, is_switch_action
from .belief import (
    BeliefEvidence,
    PlayerBeliefView,
    RevealedPokemonBelief,
)
from .env import BattleStartOverride
from .observation import PokeZeroObservationV0
from .policy import PolicyContext
from .public_action_capture import public_action_rounds_from_trajectory_metadata
from .randbat import Gen3RandbatSource, Gen3RandbatVariant, canonical_gen3_randbat_species_id
from .search import StartOverrideSource
from .search_policy import OpponentActionScenario, StartOverridePlanner
from .showdown_fixture import FixturePokemon, pack_team
from .tier2 import canonical_move_id


DEFAULT_RANDBAT_TEAM_SIZE = 6
DEFAULT_BELIEF_WORLD_SAMPLE_CAP = 4
_STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")
_PINCH_BERRIES = {"salacberry", "petayaberry", "liechiberry"}
_HIDDEN_POWER_IVS: Mapping[str, Mapping[str, int]] = {
    "bug": {"atk": 30, "def": 30, "spd": 30},
    "dark": {},
    "dragon": {"atk": 30},
    "electric": {"spa": 30},
    "fighting": {"def": 30, "spa": 30, "spd": 30, "spe": 30},
    "fire": {"atk": 30, "spa": 30, "spe": 30},
    "flying": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30},
    "ghost": {"def": 30, "spd": 30},
    "grass": {"atk": 30, "spa": 30},
    "ground": {"spa": 30, "spd": 30},
    "ice": {"atk": 30, "def": 30},
    "poison": {"def": 30, "spa": 30, "spd": 30},
    "psychic": {"atk": 30, "spe": 30},
    "rock": {"def": 30, "spd": 30, "spe": 30},
    "steel": {"spd": 30},
    "water": {"atk": 30, "def": 30, "spa": 30},
}


@dataclass(frozen=True)
class BeliefWorldSamplingProfile:
    """Public-belief-derived world-count and audit metadata for root search.

    ``sample_count`` is deliberately bounded by the number of concrete public
    variant combinations. The profile never reads the real opponent team.
    """

    sample_cap: int
    sample_count: int
    combination_count: int
    uncertainty_bits: float
    uncertain_slot_count: int
    public_checksum: str

    def to_metadata(self) -> dict[str, object]:
        return {
            "root_puct_belief_world_sample_cap": self.sample_cap,
            "root_puct_belief_world_sample_count": self.sample_count,
            "root_puct_belief_world_combination_count": self.combination_count,
            "root_puct_belief_world_uncertainty_bits": self.uncertainty_bits,
            "root_puct_belief_world_uncertain_slot_count": self.uncertain_slot_count,
            "root_puct_belief_public_checksum": self.public_checksum,
        }


def belief_world_sampling_profile(
    context: PolicyContext,
    *,
    sample_cap: int = DEFAULT_BELIEF_WORLD_SAMPLE_CAP,
    set_source: Gen3RandbatSource | None = None,
    team_size: int = DEFAULT_RANDBAT_TEAM_SIZE,
) -> BeliefWorldSamplingProfile | None:
    """Derive a bounded PIMC world count from player-relative public belief.

    The pre-registered mapping is ``K = min(sample_cap, combination_count)``.
    ``combination_count`` covers both surviving revealed variants and the
    distinct-species hidden backline worlds the materializer can sample. When a
    set source is unavailable, the profile deliberately falls back to revealed
    variants only rather than inventing a hidden-team distribution.
    """

    if sample_cap <= 0:
        raise ValueError("sample_cap must be positive.")
    if team_size <= 0:
        raise ValueError("team_size must be positive.")
    metadata = context.observation.metadata
    if not isinstance(metadata, Mapping):
        return None
    view = player_belief_view_from_payload(metadata.get("belief_view"))
    if view is None:
        return None

    revealed_candidate_counts = tuple(
        max(1, len(pokemon.candidate_variants))
        for pokemon in view.opponent_pokemon
    )
    revealed_combination_count = math.prod(revealed_candidate_counts) if revealed_candidate_counts else 1
    hidden_combination_count, hidden_slot_count = _public_hidden_backline_combination_count(
        context=context,
        view=view,
        set_source=set_source,
        team_size=team_size,
    )
    combination_count = revealed_combination_count * hidden_combination_count
    return BeliefWorldSamplingProfile(
        sample_cap=sample_cap,
        sample_count=min(sample_cap, combination_count),
        combination_count=combination_count,
        uncertainty_bits=math.log2(combination_count),
        uncertain_slot_count=sum(count > 1 for count in revealed_candidate_counts) + hidden_slot_count,
        public_checksum=_belief_world_public_checksum(context=context, view=view),
    )


def gen3_randbat_belief_start_override_planner(
    set_source: Gen3RandbatSource,
    *,
    team_size: int = DEFAULT_RANDBAT_TEAM_SIZE,
    world_sample_cap: int = DEFAULT_BELIEF_WORLD_SAMPLE_CAP,
) -> StartOverridePlanner:
    """Create a root-PUCT start-override planner from player-relative public belief.

    The returned planner is hidden-info safe: it reads the acting player's observation metadata,
    which contains the player's own request-known team plus public belief about the opponent. It
    does not inspect the opponent's private observation or legal-action mask. Each sampled world is
    shared across all candidate root actions for its scenario.
    """

    if team_size <= 0:
        raise ValueError("team_size must be positive.")
    if world_sample_cap <= 0:
        raise ValueError("world_sample_cap must be positive.")

    def planner(
        context: PolicyContext,
        scenario: OpponentActionScenario,
        scenario_index: int,
        rng: random.Random,
    ) -> StartOverrideSource:
        del scenario, scenario_index
        if not set_source.supports(context.format_id):
            return None
        sampled_override, failure_reason = _gen3_randbat_belief_start_override_result(
            context=context,
            set_source=set_source,
            rng=rng,
            team_size=team_size,
        )
        if sampled_override is None:
            reason = failure_reason or "unknown reason"

            def missing_override() -> BattleStartOverride:
                raise ValueError(f"start override planner did not produce a sampled world: {reason}")

            missing_override.start_override_id = "gen3-randbat-belief-missing"  # type: ignore[attr-defined]
            return missing_override

        def sample_override() -> BattleStartOverride:
            return sampled_override

        sample_override.start_override_id = "gen3-randbat-belief"  # type: ignore[attr-defined]
        return sample_override

    def sample_count_for_context(context: PolicyContext) -> int:
        profile = belief_world_sampling_profile(
            context,
            sample_cap=world_sample_cap,
            set_source=set_source,
            team_size=team_size,
        )
        return profile.sample_count if profile is not None else 1

    def sampling_metadata_for_context(context: PolicyContext) -> Mapping[str, object]:
        profile = belief_world_sampling_profile(
            context,
            sample_cap=world_sample_cap,
            set_source=set_source,
            team_size=team_size,
        )
        return profile.to_metadata() if profile is not None else {
            "root_puct_belief_world_sampling_profile": "missing"
        }

    planner.planner_id = "gen3-randbat-belief"  # type: ignore[attr-defined]
    planner.scenario_independent = True  # type: ignore[attr-defined]
    planner.sample_count_for_context = sample_count_for_context  # type: ignore[attr-defined]
    planner.sampling_metadata_for_context = sampling_metadata_for_context  # type: ignore[attr-defined]
    return planner


def _belief_world_public_checksum(
    *,
    context: PolicyContext,
    view: PlayerBeliefView,
) -> str:
    """Hash only the player-relative public state that determines world sampling."""

    payload = {
        "format_id": context.format_id,
        "self_slot": view.self_slot,
        "opponent_slot": view.opponent_slot,
        "opponent_pokemon": [pokemon.to_overlay_payload() for pokemon in view.opponent_pokemon],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _public_hidden_backline_combination_count(
    *,
    context: PolicyContext,
    view: PlayerBeliefView,
    set_source: Gen3RandbatSource | None,
    team_size: int,
) -> tuple[int, int]:
    """Count the public hidden-team branch space used by ``_sample_hidden_backline``.

    The sampler draws distinct species and then one set variant for each. The
    elementary-symmetric recurrence counts that weighted without-replacement
    space exactly, including public switch/move-slot constraints and ordering
    because packed-team order is observable to replay materialization.
    """

    hidden_slots = max(0, team_size - len(view.opponent_pokemon))
    if hidden_slots == 0 or set_source is None:
        return 1, 0
    used_species = {_normalize_species_id(pokemon.species) for pokemon in view.opponent_pokemon}
    team_index_constraints = _public_opponent_team_index_constraints(
        context,
        opponent_slot=view.opponent_slot,
        team_size=team_size,
    )
    if team_index_constraints is None:
        return 1, hidden_slots
    move_slot_constraints = _public_opponent_move_slot_constraints(context, view.opponent_slot)
    constrained_count = 1
    constrained_uncertain_slots = 0
    for species in sorted(team_index_constraints):
        if species in used_species:
            continue
        universe = set_source.universe_for(species)
        if universe is None:
            return 1, hidden_slots
        moves = tuple(
            move
            for _slot, move in sorted(move_slot_constraints.get(species, {}).items())
            if str(move).strip()
        )
        summary = (
            set_source.summarize(
                format_id=context.format_id,
                species=universe.species,
                revealed_moves=moves,
            )
            if moves
            else None
        )
        variants = (
            tuple(summary.candidate_variants)
            if summary is not None
            else tuple(variant.to_summary() for variant in universe.variants)
        )
        if not variants:
            return 1, hidden_slots
        constrained_count *= len(variants)
        constrained_uncertain_slots += int(len(variants) > 1)
        used_species.add(species)
        hidden_slots -= 1
    if hidden_slots < 0:
        return 1, 0
    variant_counts = tuple(
        len(universe.variants)
        for universe in set_source.universes.values()
        if universe.variants and _normalize_species_id(universe.species) not in used_species
    )
    if len(variant_counts) < hidden_slots:
        return 1, constrained_uncertain_slots + hidden_slots
    return (
        constrained_count * _ordered_distinct_variant_combination_count(variant_counts, hidden_slots),
        constrained_uncertain_slots + hidden_slots,
    )


def _ordered_distinct_variant_combination_count(
    variant_counts: Sequence[int],
    slots: int,
) -> int:
    """Return ordered distinct-species variant assignments for ``slots`` team slots."""

    elementary = [0] * (slots + 1)
    elementary[0] = 1
    for count in variant_counts:
        for index in range(slots, 0, -1):
            elementary[index] += elementary[index - 1] * count
    return elementary[slots] * math.factorial(slots)


def gen3_randbat_belief_start_override(
    *,
    context: PolicyContext,
    set_source: Gen3RandbatSource,
    rng: random.Random,
    team_size: int = DEFAULT_RANDBAT_TEAM_SIZE,
) -> BattleStartOverride | None:
    """Sample one complete custom-game start override from the acting player's belief.

    Returns ``None`` when the current observation lacks enough request-known self-team data to build
    a faithful packed team. Replay consistency checks then keep bad sampled worlds from being scored.
    """
    override, _reason = _gen3_randbat_belief_start_override_result(
        context=context,
        set_source=set_source,
        rng=rng,
        team_size=team_size,
    )
    return override


def _gen3_randbat_belief_start_override_result(
    *,
    context: PolicyContext,
    set_source: Gen3RandbatSource,
    rng: random.Random,
    team_size: int = DEFAULT_RANDBAT_TEAM_SIZE,
) -> tuple[BattleStartOverride | None, str | None]:
    """Return a sampled start override plus a public-data failure reason for diagnostics."""

    if team_size <= 0:
        raise ValueError("team_size must be positive.")
    if not set_source.supports(context.format_id):
        return None, "unsupported format"
    metadata = context.observation.metadata
    if not isinstance(metadata, Mapping):
        return None, "observation metadata is missing"
    view = player_belief_view_from_payload(metadata.get("belief_view"))
    if view is None:
        return None, "belief_view is missing or invalid"
    self_team, self_team_failure = _self_team_from_metadata_result(
        _root_self_team_payload(context, team_size=team_size) or metadata.get("self_team"),
        team_size=team_size,
        set_source=set_source,
    )
    if self_team is None:
        return None, self_team_failure or "request-known self_team is missing or inconsistent"
    team_index_constraints = _public_opponent_team_index_constraints(
        context,
        opponent_slot=view.opponent_slot,
        team_size=team_size,
    )
    if team_index_constraints is None:
        return None, "public opponent switch constraints are inconsistent"
    opponent_team, opponent_team_failure = _opponent_team_from_belief_result(
        view,
        set_source=set_source,
        format_id=context.format_id,
        rng=rng,
        team_size=team_size,
        move_slot_constraints=_public_opponent_move_slot_constraints(context, view.opponent_slot),
        team_index_constraints=team_index_constraints,
    )
    if opponent_team is None:
        return None, opponent_team_failure or "opponent belief could not be materialized"
    if view.self_slot not in {"p1", "p2"} or view.opponent_slot not in {"p1", "p2"}:
        return None, "belief_view player slots are invalid"
    return (
        BattleStartOverride(
            player_teams={
                view.self_slot: pack_team(self_team),
                view.opponent_slot: pack_team(opponent_team),
            },
            observation_format_id=context.format_id,
        ),
        None,
    )


def _root_self_team_payload(context: PolicyContext, *, team_size: int) -> Any:
    """Return the earliest request-known self-team snapshot for root replay materialization."""

    for step in context.trajectory.steps:
        if step.player_id != context.player_id:
            continue
        metadata = step.observation.metadata
        if not isinstance(metadata, Mapping):
            continue
        rows = _as_sequence(metadata.get("self_team"))
        if len(rows) == team_size:
            return rows
    return None


def _public_opponent_move_slot_constraints(
    context: PolicyContext,
    opponent_slot: str,
) -> dict[str, dict[int, str]]:
    """Map public opponent moves back to recorded move slots for replay-prefix fidelity.

    Prefix replay must resubmit historic opponent choices by move slot. The slot index is part of the
    recorded trajectory, but the move name is taken only from the acting player's public event log.
    This avoids reading opponent-private request moves while still forcing already-revealed moves into
    the slots needed to reproduce the public history.
    """

    own_observations = _own_observations_by_decision_round(context)
    public_rounds = public_action_rounds_from_trajectory_metadata(context.trajectory)
    active_species_by_turn = {
        turn_index: species
        for turn_index, observation in own_observations.items()
        if (species := _public_opponent_active_species(observation)) is not None
    }
    constraints: dict[str, dict[int, str]] = {}
    for step in context.trajectory.steps:
        if step.player_id != opponent_slot or not 0 <= step.action_index < 4:
            continue
        species = active_species_by_turn.get(step.turn_index)
        if species is None:
            continue
        identifier = public_rounds.get(step.turn_index, None)
        public_action = identifier.actions.get(opponent_slot) if identifier is not None else None
        move = public_action.move_id if public_action is not None and public_action.kind == "move" else None
        if move is None:
            move = _public_move_after_decision_round(
                own_observations,
                opponent_slot=opponent_slot,
                self_slot=context.player_id,
                species=species,
                turn_index=step.turn_index,
            )
        if move is None:
            continue
        species_key = _normalize_species_id(species)
        slot_constraints = constraints.setdefault(species_key, {})
        existing = slot_constraints.get(step.action_index)
        if existing is not None and _normalize_id(existing) != _normalize_id(move):
            continue
        slot_constraints[step.action_index] = move
    return constraints


def _public_opponent_team_index_constraints(
    context: PolicyContext,
    *,
    opponent_slot: str,
    team_size: int,
) -> dict[str, int] | None:
    """Map public opponent switch targets back to recorded packed-team indices.

    Replay submits historical opponent switch actions by action index. In Showdown, that action
    index decodes through the opponent's private team order, so a sampled packed team must put a
    publicly switched-in species at the same party index that the recorded action targeted. The
    switch action index comes from the replay trajectory; the target species comes only from public
    switch/drag/replace lines visible in the acting player's observations.
    """

    if team_size <= 0:
        return None
    own_observations = _own_observations_by_decision_round(context)
    public_rounds = public_action_rounds_from_trajectory_metadata(context.trajectory)
    constraints: dict[str, int] = {}
    current_order = list(range(team_size))
    active_position: int | None = None
    active_species: str | None = None
    if own_observations:
        first_turn = min(own_observations)
        if first_turn == 0:
            active_species = _public_opponent_active_species(own_observations[first_turn])
            if active_species is not None:
                if not _assign_team_index_constraint(
                    constraints,
                    species=active_species,
                    team_index=0,
                    team_size=team_size,
                ):
                    return None
                active_position = 0

    opponent_steps = sorted(
        (
            step
            for step in context.trajectory.steps
            if step.player_id == opponent_slot and step.turn_index < context.decision_round_index
        ),
        key=lambda step: step.turn_index,
    )
    for step in opponent_steps:
        next_active = _public_opponent_active_species(own_observations.get(step.turn_index + 1))
        if is_switch_action(step.action_index) and active_position is not None:
            switch_slot = step.action_index - MOVE_ACTION_COUNT
            try:
                switch_targets = canonical_switch_action_map(active_position, team_size=team_size)
            except ValueError:
                switch_targets = ()
            if switch_slot < len(switch_targets):
                identifier = public_rounds.get(step.turn_index, None)
                public_action = identifier.actions.get(opponent_slot) if identifier is not None else None
                switch_species = (
                    public_action.switched_species
                    if public_action is not None and public_action.kind == "switch"
                    else None
                )
                if switch_species is None:
                    switch_species = _public_switch_after_decision_round(
                        own_observations,
                        opponent_slot=opponent_slot,
                        self_slot=context.player_id,
                        turn_index=step.turn_index,
                    )
                if switch_species is not None:
                    target_position = switch_targets[switch_slot]
                    target_index = current_order[target_position]
                    if not _assign_team_index_constraint(
                        constraints,
                        species=switch_species,
                        team_index=target_index,
                        team_size=team_size,
                    ):
                        return None
                    current_order[active_position], current_order[target_position] = (
                        current_order[target_position],
                        current_order[active_position],
                    )
                    active_species = switch_species
                    active_position = 0
                    continue

        if next_active is None:
            continue
        next_key = _normalize_species_id(next_active)
        if next_key in constraints:
            active_species = next_active
            active_position = _move_constrained_species_to_active_position(
                current_order,
                constraints[next_key],
                active_position=active_position,
            )
        elif active_species is not None and next_key != _normalize_species_id(active_species):
            active_species = next_active
            active_position = None
    return constraints


def _move_constrained_species_to_active_position(
    current_order: list[int],
    initial_index: int,
    *,
    active_position: int | None,
) -> int | None:
    try:
        current_position = current_order.index(initial_index)
    except ValueError:
        return None
    if active_position is None:
        return None
    if current_position != active_position:
        current_order[active_position], current_order[current_position] = (
            current_order[current_position],
            current_order[active_position],
        )
    return 0


def _assign_team_index_constraint(
    constraints: dict[str, int],
    *,
    species: str,
    team_index: int,
    team_size: int,
) -> bool:
    if team_index < 0 or team_index >= team_size:
        return False
    species_key = _normalize_species_id(species)
    existing = constraints.get(species_key)
    if existing is not None:
        return existing == team_index
    if any(index == team_index for key, index in constraints.items() if key != species_key):
        return False
    constraints[species_key] = team_index
    return True


def _public_switch_after_decision_round(
    observations_by_turn: Mapping[int, PokeZeroObservationV0],
    *,
    opponent_slot: str,
    self_slot: str,
    turn_index: int,
) -> str | None:
    next_observation = observations_by_turn.get(turn_index + 1)
    if next_observation is None:
        return None
    for line in reversed(_recent_public_events(next_observation)):
        species = _switch_species_from_public_event_line(
            line,
            opponent_slot=opponent_slot,
            self_slot=self_slot,
        )
        if species is not None:
            return species
    return None


def _switch_species_from_public_event_line(
    line: str,
    *,
    opponent_slot: str,
    self_slot: str,
) -> str | None:
    parts = str(line).split("|")
    if len(parts) < 4 or parts[1] not in {"switch", "drag", "replace"}:
        return None
    actor = parts[2]
    if not _public_actor_matches_slot(actor, slot=opponent_slot, self_slot=self_slot):
        return None
    return _species_from_switch_details(parts[3]) or _species_from_public_actor(actor)


def _species_from_switch_details(details: str) -> str | None:
    species = str(details).split(",", 1)[0].strip()
    return species or None


def _own_observations_by_decision_round(context: PolicyContext) -> dict[int, PokeZeroObservationV0]:
    observations: dict[int, PokeZeroObservationV0] = {
        context.decision_round_index: context.observation,
    }
    for step in context.trajectory.steps:
        if step.player_id == context.player_id:
            observations.setdefault(step.turn_index, step.observation)
    return observations


def _public_opponent_active_species(observation: PokeZeroObservationV0 | None) -> str | None:
    if observation is None:
        return None
    metadata = observation.metadata
    if not isinstance(metadata, Mapping):
        return None
    active = metadata.get("opponent_active")
    if not isinstance(active, Mapping):
        return None
    return _optional_text(active.get("species"))


def _public_move_after_decision_round(
    observations_by_turn: Mapping[int, PokeZeroObservationV0],
    *,
    opponent_slot: str,
    self_slot: str,
    species: str,
    turn_index: int,
) -> str | None:
    next_observation = observations_by_turn.get(turn_index + 1)
    if next_observation is None:
        return None
    # The public-event window is rolling, so the same active mon's older move can still be present.
    # Walk backward through the next decision observation and take the newest matching move line.
    for line in reversed(_recent_public_events(next_observation)):
        move = _move_from_public_event_line(
            line,
            opponent_slot=opponent_slot,
            self_slot=self_slot,
            species=species,
        )
        if move is not None:
            return move
    return None


def _recent_public_events(observation: PokeZeroObservationV0) -> tuple[str, ...]:
    metadata = observation.metadata
    if not isinstance(metadata, Mapping):
        return ()
    return tuple(str(line) for line in _as_sequence(metadata.get("recent_public_events")))


def _move_from_public_event_line(
    line: str,
    *,
    opponent_slot: str,
    self_slot: str,
    species: str,
) -> str | None:
    parts = str(line).split("|")
    if len(parts) < 4 or parts[1] != "move":
        return None
    if _called_move_line(parts):
        return None
    actor = parts[2]
    if not _public_actor_matches_slot(actor, slot=opponent_slot, self_slot=self_slot):
        return None
    actor_species = _species_from_public_actor(actor)
    if actor_species is not None and _normalize_species_id(actor_species) != _normalize_species_id(species):
        return None
    return _optional_text(parts[3])


def _called_move_line(parts: Sequence[str]) -> bool:
    for token in parts[4:]:
        text = str(token).strip()
        if not text.startswith("[from]"):
            continue
        normalized = _normalize_id(text)
        if "lockedmove" in normalized:
            continue
        return True
    return False


def _public_actor_matches_slot(actor: str, *, slot: str, self_slot: str) -> bool:
    normalized = str(actor).strip().lower()
    if normalized.startswith(slot.lower()):
        return True
    if normalized.startswith("opponent"):
        return slot != self_slot
    if normalized.startswith("self"):
        return slot == self_slot
    return False


def _species_from_public_actor(actor: str) -> str | None:
    if ":" not in actor:
        return None
    return _optional_text(actor.split(":", 1)[1])


def player_belief_view_from_payload(payload: Any) -> PlayerBeliefView | None:
    if not isinstance(payload, Mapping):
        return None
    self_slot = _optional_slot(payload.get("self_slot"))
    opponent_slot = _optional_slot(payload.get("opponent_slot"))
    if self_slot is None or opponent_slot is None or self_slot == opponent_slot:
        return None
    return PlayerBeliefView(
        self_slot=self_slot,
        opponent_slot=opponent_slot,
        self_pokemon=tuple(
            pokemon
            for raw in _as_sequence(payload.get("self_pokemon"))
            if (pokemon := _revealed_pokemon_from_payload(raw)) is not None
        ),
        opponent_pokemon=tuple(
            pokemon
            for raw in _as_sequence(payload.get("opponent_pokemon"))
            if (pokemon := _revealed_pokemon_from_payload(raw)) is not None
        ),
    )


def _self_team_from_metadata(
    payload: Any,
    *,
    team_size: int,
    set_source: Gen3RandbatSource,
) -> tuple[FixturePokemon, ...] | None:
    """Return a reconstructible request-known team without exposing diagnostic details."""

    team, _failure = _self_team_from_metadata_result(
        payload,
        team_size=team_size,
        set_source=set_source,
    )
    return team


def _self_team_from_metadata_result(
    payload: Any,
    *,
    team_size: int,
    set_source: Gen3RandbatSource,
) -> tuple[tuple[FixturePokemon, ...] | None, str | None]:
    """Return a fixture team or a stable public failure category for Root-PUCT telemetry."""

    rows = _as_sequence(payload)
    if len(rows) != team_size:
        return None, "request-known self_team has an unexpected member count"
    team: list[FixturePokemon] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None, "request-known self_team contains an invalid member"
        species = _optional_text(row.get("species"))
        request_moves = _moves_from_payload(row.get("moves"))
        if species is None or not request_moves:
            return None, "request-known self_team member is missing species or moves"
        # Showdown uses resolved-power request IDs such as ``return102``. Custom
        # replay teams require the canonical move ID (``return``) instead.
        moves = tuple(canonical_move_id(move) for move in request_moves)
        level = _level_from_details(_optional_text(row.get("details"))) or 100
        spread = _gen3_randbat_fixture_spread(
            row,
            species=species,
            moves=moves,
            item=_optional_text(row.get("item")),
            level=level,
            set_source=set_source,
        )
        if spread is None:
            return None, "request-known self_team fixture stats cannot be reconstructed"
        team.append(
            FixturePokemon(
                species=species,
                moves=moves,
                ability=_optional_text(row.get("ability")),
                item=_optional_text(row.get("item")),
                level=level,
                evs=spread["evs"],
                ivs=spread["ivs"],
            )
        )
    return tuple(team), None


def _gen3_randbat_fixture_spread(
    row: Mapping[str, Any],
    *,
    species: str,
    moves: tuple[str, ...],
    item: str | None,
    level: int,
    set_source: Gen3RandbatSource,
) -> dict[str, Mapping[str, int]] | None:
    """Mirror Showdown's Gen 3 randbat EV/IV recipe for request-known self Pokemon.

    If the request includes actual stats, fail closed unless the reconstructed spread reproduces
    them exactly. Simplified fixture rows without stats keep the default randbat spread.
    """

    evs = {stat: 85 for stat in _STAT_ORDER}
    ivs = {stat: 31 for stat in _STAT_ORDER}
    # Request payloads encode dynamic-power moves as ids such as ``return102``.
    # Normalize aliases before applying Showdown's Gen 3 stat rules.
    normalized_moves = tuple(canonical_move_id(move) for move in moves)
    hidden_power_type = _hidden_power_type(normalized_moves)
    if hidden_power_type is not None:
        for stat, value in _HIDDEN_POWER_IVS.get(hidden_power_type, {}).items():
            ivs[stat] = value

    base_stats = _base_stats_for_species(set_source, species)
    observed_stats = _stats_from_payload(row.get("stats"))
    if observed_stats and base_stats is None:
        return None

    if base_stats is not None:
        _adjust_hp_evs(
            evs=evs,
            ivs=ivs,
            base_hp=base_stats["hp"],
            level=level,
            moves=normalized_moves,
            item=item,
        )

    if _should_minimize_confusion_damage(normalized_moves, set_source.move_metadata):
        evs["atk"] = 0
        ivs["atk"] = (ivs["atk"] or 31) - 28 if hidden_power_type is not None else 0

    if base_stats is not None:
        _adjust_post_attack_hp_evs(
            evs=evs,
            ivs=ivs,
            base_hp=base_stats["hp"],
            level=level,
            moves=normalized_moves,
            item=item,
        )

    if observed_stats:
        computed = _computed_stats(base_stats, evs=evs, ivs=ivs, level=level) if base_stats is not None else None
        if computed is None or any(computed.get(stat) != value for stat, value in observed_stats.items()):
            return None
    return {"evs": evs, "ivs": ivs}


def _opponent_team_from_belief(
    view: PlayerBeliefView,
    *,
    set_source: Gen3RandbatSource,
    format_id: str,
    rng: random.Random,
    team_size: int,
    move_slot_constraints: Mapping[str, Mapping[int, str]] | None = None,
    team_index_constraints: Mapping[str, int] | None = None,
) -> tuple[FixturePokemon, ...] | None:
    team, _reason = _opponent_team_from_belief_result(
        view,
        set_source=set_source,
        format_id=format_id,
        rng=rng,
        team_size=team_size,
        move_slot_constraints=move_slot_constraints,
        team_index_constraints=team_index_constraints,
    )
    return team


def _opponent_team_from_belief_result(
    view: PlayerBeliefView,
    *,
    set_source: Gen3RandbatSource,
    format_id: str,
    rng: random.Random,
    team_size: int,
    move_slot_constraints: Mapping[str, Mapping[int, str]] | None = None,
    team_index_constraints: Mapping[str, int] | None = None,
) -> tuple[tuple[FixturePokemon, ...] | None, str | None]:
    team: list[FixturePokemon] = []
    used_species: set[str] = set()
    for belief in view.opponent_pokemon:
        fixture = _sample_revealed_opponent_fixture(
            belief,
            set_source=set_source,
            format_id=format_id,
            rng=rng,
            move_slot_constraints=move_slot_constraints,
        )
        if fixture is None:
            return None, f"opponent {belief.species} could not be sampled from public belief"
        team.append(fixture)
        used_species.add(_normalize_species_id(fixture.species))
    hidden_needed = team_size - len(team)
    if hidden_needed < 0:
        return None, "opponent belief has more revealed pokemon than team slots"
    constrained_hidden = _sample_constrained_hidden_backline(
        set_source,
        format_id=format_id,
        used_species=used_species,
        team_index_constraints=team_index_constraints or {},
        move_slot_constraints=move_slot_constraints or {},
        count_limit=hidden_needed,
        rng=rng,
    )
    if constrained_hidden is None:
        return None, "public constrained hidden opponent species could not be sampled"
    team.extend(constrained_hidden)
    hidden_needed = team_size - len(team)
    hidden = _sample_hidden_backline(
        set_source,
        used_species=used_species,
        count=hidden_needed,
        rng=rng,
    )
    if hidden is None:
        return None, "random hidden opponent backline could not be sampled"
    constrained_team = _fixture_team_with_index_constraints(
        tuple(team + list(hidden)),
        team_index_constraints or {},
        team_size=team_size,
    )
    if constrained_team is None:
        return None, "sampled opponent team could not satisfy public team slot constraints"
    return constrained_team, None


def _fixture_team_with_index_constraints(
    team: tuple[FixturePokemon, ...],
    constraints: Mapping[str, int],
    *,
    team_size: int,
) -> tuple[FixturePokemon, ...] | None:
    if len(team) != team_size:
        return None
    if not constraints:
        return team
    slots: list[FixturePokemon | None] = [None] * team_size
    unconstrained: list[FixturePokemon] = []
    for fixture in team:
        species_key = _normalize_species_id(fixture.species)
        target_index = constraints.get(species_key)
        if target_index is None:
            unconstrained.append(fixture)
            continue
        if target_index < 0 or target_index >= team_size or slots[target_index] is not None:
            return None
        slots[target_index] = fixture
    constrained_species = {_normalize_species_id(fixture.species) for fixture in team}
    missing_constraints = set(constraints) - constrained_species
    if missing_constraints:
        return None
    remaining = iter(unconstrained)
    for index, value in enumerate(slots):
        if value is None:
            try:
                slots[index] = next(remaining)
            except StopIteration:
                return None
    if any(slot is None for slot in slots):
        return None
    return tuple(slot for slot in slots if slot is not None)


def _sample_revealed_opponent_fixture(
    pokemon: RevealedPokemonBelief,
    *,
    set_source: Gen3RandbatSource,
    format_id: str,
    rng: random.Random,
    move_slot_constraints: Mapping[str, Mapping[int, str]] | None = None,
) -> FixturePokemon | None:
    variants = pokemon.candidate_variants
    if not variants:
        summary = set_source.summarize(
            format_id=format_id,
            species=pokemon.species,
            revealed_moves=pokemon.revealed_moves,
            revealed_ability=pokemon.revealed_ability,
            revealed_item=pokemon.revealed_item,
            ruled_out_abilities=pokemon.ruled_out_abilities,
        )
        variants = tuple(summary.candidate_variants) if summary is not None else ()
    if not variants:
        return None
    public_max_hp = _condition_max_hp(pokemon.condition)
    if public_max_hp is not None:
        hp_matched = tuple(
            variant
            for variant in variants
            if _variant_public_max_hp(
                variant,
                fallback_species=pokemon.species,
                set_source=set_source,
            )
            == public_max_hp
        )
        if not hp_matched:
            return None
        variants = hp_matched
    species_constraints = (move_slot_constraints or {}).get(_normalize_species_id(pokemon.species), {})
    if species_constraints:
        slot_matched = tuple(
            variant
            for variant in variants
            if _fixture_from_variant_payload(
                variant,
                fallback_species=pokemon.species,
                set_source=set_source,
                move_slot_constraints=species_constraints,
            )
            is not None
        )
        if not slot_matched:
            return None
        variants = slot_matched
    variant = variants[rng.randrange(len(variants))]
    return _fixture_from_variant_payload(
        variant,
        fallback_species=pokemon.species,
        set_source=set_source,
        move_slot_constraints=species_constraints,
    )


def _condition_max_hp(condition: str | None) -> int | None:
    if not condition:
        return None
    match = re.match(r"^\s*\d+\s*/\s*(\d+)\s*(?:\s|$)", condition)
    if match is None:
        return None
    try:
        max_hp = int(match.group(1))
    except ValueError:
        return None
    # Opponent public conditions are percentages in Showdown's request/public state
    # (e.g. "70/100"). Only request-known absolute HP denominators can safely filter variants.
    if max_hp <= 100:
        return None
    return max_hp


def _variant_public_max_hp(
    payload: Mapping[str, Any],
    *,
    fallback_species: str,
    set_source: Gen3RandbatSource,
) -> int | None:
    fixture = _fixture_from_variant_payload(
        payload,
        fallback_species=fallback_species,
        set_source=set_source,
    )
    if fixture is None:
        return None
    base_stats = _base_stats_for_species(set_source, fixture.species)
    if base_stats is None:
        return None
    ivs = fixture.ivs or {}
    evs = fixture.evs or {}
    return _hp_stat(
        base_hp=base_stats["hp"],
        iv=int(ivs.get("hp", 31)),
        ev=int(evs.get("hp", 0)),
        level=fixture.level,
    )


def _sample_constrained_hidden_backline(
    set_source: Gen3RandbatSource,
    *,
    format_id: str,
    used_species: set[str],
    team_index_constraints: Mapping[str, int],
    move_slot_constraints: Mapping[str, Mapping[int, str]],
    count_limit: int,
    rng: random.Random,
) -> tuple[FixturePokemon, ...] | None:
    required_species = [
        species
        for species, _index in sorted(team_index_constraints.items(), key=lambda item: (item[1], item[0]))
        if species not in used_species
    ]
    if len(required_species) > count_limit:
        return None
    fixtures: list[FixturePokemon] = []
    for species in required_species:
        universe = set_source.universe_for(species)
        if universe is None or not universe.variants:
            return None
        species_move_slot_constraints = move_slot_constraints.get(_normalize_species_id(species), {})
        revealed_moves = tuple(
            move
            for _slot, move in sorted(species_move_slot_constraints.items())
            if str(move).strip()
        )
        if revealed_moves:
            summary = set_source.summarize(
                format_id=format_id,
                species=universe.species,
                revealed_moves=revealed_moves,
            )
            variants = tuple(summary.candidate_variants) if summary is not None else ()
        else:
            variants = tuple(variant.to_summary() for variant in universe.variants)
        if not variants:
            return None
        variant = variants[rng.randrange(len(variants))]
        fixture = _fixture_from_variant_payload(
            variant,
            fallback_species=universe.species,
            set_source=set_source,
            move_slot_constraints=species_move_slot_constraints,
        )
        if fixture is None:
            return None
        used_species.add(_normalize_species_id(fixture.species))
        fixtures.append(fixture)
    return tuple(fixtures)


def _sample_hidden_backline(
    set_source: Gen3RandbatSource,
    *,
    used_species: set[str],
    count: int,
    rng: random.Random,
) -> tuple[FixturePokemon, ...] | None:
    candidates = [
        universe
        for universe in set_source.universes.values()
        if universe.variants and _normalize_species_id(universe.species) not in used_species
    ]
    team: list[FixturePokemon] = []
    for _ in range(count):
        if not candidates:
            return None
        universe_index = rng.randrange(len(candidates))
        universe = candidates.pop(universe_index)
        variant = universe.variants[rng.randrange(len(universe.variants))]
        used_species.add(_normalize_species_id(universe.species))
        fixture = _fixture_from_variant(variant, set_source=set_source)
        if fixture is None:
            return None
        team.append(fixture)
    return tuple(team)


def _hidden_power_type(normalized_moves: Sequence[str]) -> str | None:
    for move in normalized_moves:
        if move.startswith("hiddenpower") and len(move) > len("hiddenpower"):
            return move[len("hiddenpower") :]
    return None


def _base_stats_for_species(
    set_source: Gen3RandbatSource,
    species: str,
) -> dict[str, int] | None:
    raw = set_source.species_metadata.get(_normalize_species_id(species))
    if not isinstance(raw, Mapping):
        return None
    base_stats = raw.get("baseStats")
    if not isinstance(base_stats, Mapping):
        return None
    stats: dict[str, int] = {}
    for stat in _STAT_ORDER:
        value = base_stats.get(stat)
        if not isinstance(value, int):
            return None
        stats[stat] = value
    return stats


def _stats_from_payload(payload: Any) -> dict[str, int] | None:
    if not isinstance(payload, Mapping):
        return None
    stats: dict[str, int] = {}
    for stat in _STAT_ORDER:
        value = payload.get(stat)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            stats[stat] = value
    return stats or None


def _adjust_hp_evs(
    *,
    evs: dict[str, int],
    ivs: Mapping[str, int],
    base_hp: int,
    level: int,
    moves: Sequence[str],
    item: str | None,
) -> None:
    while evs["hp"] > 1:
        hp = _hp_stat(base_hp=base_hp, iv=ivs["hp"], ev=evs["hp"], level=level)
        if "substitute" in moves and any(move in moves for move in ("flail", "reversal")):
            if hp % 4 > 0:
                break
        elif "substitute" in moves and _normalize_id(item or "") in _PINCH_BERRIES:
            if hp % 4 == 0:
                break
        elif "bellydrum" in moves:
            if hp % 2 > 0:
                break
        else:
            break
        evs["hp"] -= 4


def _adjust_post_attack_hp_evs(
    *,
    evs: dict[str, int],
    ivs: Mapping[str, int],
    base_hp: int,
    level: int,
    moves: Sequence[str],
    item: str | None,
) -> None:
    hp = _hp_stat(base_hp=base_hp, iv=ivs["hp"], ev=evs["hp"], level=level)
    if "substitute" in moves and any(move in moves for move in ("endeavor", "flail", "reversal")):
        if hp % 4 == 0:
            evs["hp"] -= 4
    elif "substitute" in moves and _normalize_id(item or "") in _PINCH_BERRIES:
        while evs["hp"] > 1 and hp % 4 > 0:
            evs["hp"] -= 4
            hp = _hp_stat(base_hp=base_hp, iv=ivs["hp"], ev=evs["hp"], level=level)


def _should_minimize_confusion_damage(
    normalized_moves: Sequence[str],
    move_metadata: Mapping[str, Mapping[str, Any]],
) -> bool:
    if "transform" in normalized_moves:
        return False
    for move in normalized_moves:
        metadata = move_metadata.get(_normalize_id(move), {})
        # Showdown's random-team counter treats Seismic Toss/Night Shade/etc. as fixed-damage
        # moves, not physical attacks, when deciding whether to minimize confusion damage.
        if metadata.get("damage") or metadata.get("damageCallback"):
            continue
        if _gen3_move_category(move, move_metadata) == "Physical":
            return False
    return True


def _gen3_move_category(
    move: str,
    move_metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    metadata = move_metadata.get(_normalize_id(move), {})
    if str(metadata.get("category") or "") == "Status":
        return "Status"
    normalized_move = _normalize_id(move)
    if normalized_move.startswith("hiddenpower") and len(normalized_move) > len("hiddenpower"):
        raw_type = normalized_move[len("hiddenpower") :]
        move_type = raw_type[:1].upper() + raw_type[1:]
    else:
        move_type = str(metadata.get("type") or "")
    if move_type in {"Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel"}:
        return "Physical"
    if move_type in {"Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark"}:
        return "Special"
    return str(metadata.get("category") or "Status")


def _computed_stats(
    base_stats: Mapping[str, int],
    *,
    evs: Mapping[str, int],
    ivs: Mapping[str, int],
    level: int,
) -> dict[str, int]:
    stats = {
        "hp": _hp_stat(base_hp=base_stats["hp"], iv=ivs["hp"], ev=evs["hp"], level=level),
    }
    for stat in ("atk", "def", "spa", "spd", "spe"):
        stats[stat] = _battle_stat(
            base=base_stats[stat],
            iv=ivs[stat],
            ev=evs[stat],
            level=level,
        )
    return stats


def _hp_stat(*, base_hp: int, iv: int, ev: int, level: int) -> int:
    return ((2 * base_hp + iv + ev // 4 + 100) * level) // 100 + 10


def _battle_stat(*, base: int, iv: int, ev: int, level: int) -> int:
    return ((2 * base + iv + ev // 4) * level) // 100 + 5


def _fixture_from_determinized(pokemon: Any, *, set_source: Gen3RandbatSource) -> FixturePokemon | None:
    moves = tuple(str(move) for move in getattr(pokemon, "moves", ()) if str(move))
    species = _optional_text(getattr(pokemon, "species", None))
    if species is None or not moves:
        return None
    level = int(getattr(pokemon, "level", None) or 100)
    item = _optional_text(getattr(pokemon, "item", None))
    spread = _gen3_randbat_fixture_spread(
        {},
        species=species,
        moves=moves,
        item=item,
        level=level,
        set_source=set_source,
    )
    if spread is None:
        return None
    return FixturePokemon(
        species=species,
        moves=moves,
        ability=_optional_text(getattr(pokemon, "ability", None)),
        item=item,
        level=level,
        evs=spread["evs"],
        ivs=spread["ivs"],
    )


def _fixture_from_variant_payload(
    payload: Mapping[str, Any],
    *,
    fallback_species: str,
    set_source: Gen3RandbatSource,
    move_slot_constraints: Mapping[int, str] | None = None,
) -> FixturePokemon | None:
    moves = _moves_from_payload(payload.get("moves"))
    if not moves:
        return None
    # Randbat summary payloads intentionally omit species, so fallback_species preserves public
    # cosmetic formes such as Unown-Z while stats/source lookup canonicalizes to base Unown.
    species = _optional_text(payload.get("species")) or fallback_species
    level = payload.get("level")
    resolved_level = int(level) if isinstance(level, int) else 100
    item = _optional_text(payload.get("item"))
    spread = _gen3_randbat_fixture_spread(
        {},
        species=species,
        moves=moves,
        item=item,
        level=resolved_level,
        set_source=set_source,
    )
    if spread is None:
        return None
    fixture = FixturePokemon(
        species=species,
        moves=moves,
        ability=_optional_text(payload.get("ability")),
        item=item,
        level=resolved_level,
        evs=spread["evs"],
        ivs=spread["ivs"],
    )
    if move_slot_constraints:
        fixture = _fixture_with_move_slot_constraints(fixture, move_slot_constraints)
    return fixture


def _fixture_with_move_slot_constraints(
    fixture: FixturePokemon,
    move_slot_constraints: Mapping[int, str],
) -> FixturePokemon | None:
    moves = list(fixture.moves)
    if not moves:
        return None
    assigned: list[str | None] = [None] * len(moves)
    used: set[str] = set()
    for slot, move in sorted(move_slot_constraints.items()):
        if slot < 0 or slot >= len(assigned):
            return None
        actual_move = _fixture_move_for_public_constraint(moves, move)
        if actual_move is None:
            return None
        normalized = _normalize_id(actual_move)
        if normalized in used:
            return None
        assigned[slot] = actual_move
        used.add(normalized)
    remaining = [move for move in moves if _normalize_id(move) not in used]
    for index, value in enumerate(assigned):
        if value is None:
            assigned[index] = remaining.pop(0)
    return replace(fixture, moves=tuple(move for move in assigned if move is not None))


def _fixture_move_for_public_constraint(moves: Sequence[str], public_move: str) -> str | None:
    public_id = _normalize_id(public_move)
    # Some request/public labels include the Gen 3 Hidden Power base power, e.g. Hidden Power Ice 70.
    if public_id.endswith("70"):
        public_id = public_id[:-2]
    for move in moves:
        move_id = _normalize_id(move)
        if move_id == public_id:
            return move
        if public_id == "hiddenpower" and move_id.startswith("hiddenpower"):
            return move
    return None


def _fixture_from_variant(
    variant: Gen3RandbatVariant,
    *,
    set_source: Gen3RandbatSource,
) -> FixturePokemon | None:
    spread = _gen3_randbat_fixture_spread(
        {},
        species=variant.species,
        moves=variant.moves,
        item=variant.item,
        level=variant.level,
        set_source=set_source,
    )
    if spread is None:
        return None
    return FixturePokemon(
        species=variant.species,
        moves=variant.moves,
        ability=variant.ability,
        item=variant.item,
        level=variant.level,
        evs=spread["evs"],
        ivs=spread["ivs"],
    )


def _revealed_pokemon_from_payload(payload: Any) -> RevealedPokemonBelief | None:
    if not isinstance(payload, Mapping):
        return None
    showdown_slot = _optional_text(payload.get("showdown_slot"))
    species = _optional_text(payload.get("species"))
    if showdown_slot is None or species is None:
        return None
    return RevealedPokemonBelief(
        showdown_slot=showdown_slot,
        species=species,
        condition=_optional_text(payload.get("condition")),
        status=_optional_text(payload.get("status")),
        active=bool(payload.get("active")),
        revealed_moves=_moves_from_payload(payload.get("revealed_moves")),
        revealed_ability=_optional_text(payload.get("revealed_ability")),
        revealed_item=_optional_text(payload.get("revealed_item")),
        ruled_out_abilities=_moves_from_payload(payload.get("ruled_out_abilities")),
        candidate_set_count=_optional_int(payload.get("candidate_set_count")),
        uncertainty=_optional_float(payload.get("uncertainty"), default=1.0),
        possible_abilities=_moves_from_payload(payload.get("possible_abilities")),
        possible_items=_moves_from_payload(payload.get("possible_items")),
        possible_moves=_moves_from_payload(payload.get("possible_moves")),
        candidate_variants=tuple(
            dict(variant)
            for variant in _as_sequence(payload.get("candidate_variants"))
            if isinstance(variant, Mapping)
        ),
        source_metadata=dict(payload["source_metadata"])
        if isinstance(payload.get("source_metadata"), Mapping)
        else None,
        evidence=tuple(
            BeliefEvidence(
                kind=str(item.get("kind") or ""),
                detail=str(item.get("detail") or ""),
                source_line=_optional_text(item.get("source_line")),
            )
            for item in _as_sequence(payload.get("evidence"))
            if isinstance(item, Mapping)
        ),
        transformed=bool(payload.get("transformed")),
        transform_species=_optional_text(payload.get("transform_species")),
    )


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _moves_from_payload(value: Any) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in _as_sequence(value) if str(item).strip())


def _optional_slot(value: Any) -> str | None:
    text = _optional_text(value)
    return text if text in {"p1", "p2"} else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _optional_float(value: Any, *, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _level_from_details(details: str | None) -> int | None:
    if not details:
        return None
    for part in details.split(","):
        token = part.strip()
        if token.startswith("L") and token[1:].isdigit():
            return int(token[1:])
    return None


def _normalize_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _normalize_species_id(value: str) -> str:
    return canonical_gen3_randbat_species_id(value)
