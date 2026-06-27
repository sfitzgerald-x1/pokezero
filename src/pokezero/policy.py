"""Policy interfaces and baseline policies for early rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .actions import ACTION_COUNT, MOVE_ACTION_COUNT
from .dex import ShowdownDex, load_showdown_dex_cached, normalize_id
from .observation import PokeZeroObservationV0
from .trajectory import BattleTrajectory


_STATUS_CURE_WEIGHTS = {
    "par": 1.0,
    "psn": 1.0,
    "brn": 1.25,
    "tox": 1.5,
    "slp": 2.0,
    "frz": 2.0,
}
_SIDE_HAZARDS = {"spikes", "stealthrock", "toxicspikes"}
# Showdown can expose Recharge as a forced pseudo-move after Hyper Beam-style moves.
# Other forced Gen 3 moves seen in requests, including Struggle and lock-in moves,
# are ordinary dex moves and should keep using normal strict validation.
_FORCED_PSEUDO_MOVE_IDS = {"recharge"}
SCRIPTED_TEACHER_BRANCHES = (
    "damaging_move",
    "damaging_no_effect",
    "fallback",
    "forced_pseudo_move",
    "low_impact_status",
    "rapid_spin_blocked_by_ghost",
    "rapid_spin_clear_hazards",
    "rapid_spin_no_hazards",
    "recovery",
    "setup",
    "spikes_available",
    "spikes_maxed",
    "status_no_effect",
    "status_pressure",
    "switch",
    "switch_missing_target",
    "team_status_cure",
    "team_status_cure_no_status",
    "unknown_move",
)


@dataclass(frozen=True)
class PolicyDecision:
    action_index: int
    policy_id: str
    action_probability: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    value_estimate: Optional[float] = None

    def __post_init__(self) -> None:
        if self.action_index < 0 or self.action_index >= ACTION_COUNT:
            raise ValueError(f"action_index must be between 0 and {ACTION_COUNT - 1}.")
        if self.action_probability is not None and not 0.0 <= self.action_probability <= 1.0:
            raise ValueError("action_probability must be between 0 and 1 when set.")
        if self.value_estimate is not None and not math.isfinite(float(self.value_estimate)):
            raise ValueError("value_estimate must be finite when set.")


@dataclass(frozen=True)
class PolicyContext:
    player_id: str
    decision_round_index: int
    battle_id: str
    format_id: str
    seed: int
    observation: PokeZeroObservationV0
    requested_players: tuple[str, ...]
    trajectory: BattleTrajectory
    requested_legal_action_masks: Mapping[str, tuple[bool, ...]] = field(default_factory=dict)
    # Privileged: contains every requested player's private observation. Context-aware policies
    # should only read other players' observations when intentionally building search/eval tooling.
    requested_observations: Mapping[str, PokeZeroObservationV0] = field(default_factory=dict)


@runtime_checkable
class Policy(Protocol):
    policy_id: str

    def select_action(
        self,
        observation: PokeZeroObservationV0,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        ...


@runtime_checkable
class ContextAwarePolicy(Policy, Protocol):
    def select_action_with_context(
        self,
        context: PolicyContext,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        ...


@dataclass(frozen=True)
class RandomLegalPolicy:
    policy_id: str = "random-legal"

    def select_action(
        self,
        observation: PokeZeroObservationV0,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        legal = legal_action_indices(observation.legal_action_mask)
        action_index = rng.choice(legal)
        return PolicyDecision(
            action_index=action_index,
            policy_id=self.policy_id,
            action_probability=1.0 / len(legal),
        )


@dataclass(frozen=True)
class SimpleLegalPolicy:
    """A non-strategic baseline with explicit switch participation.

    This policy is intentionally weak. It exists to verify rollout plumbing and
    avoid a benchmark that never exercises legal switch actions.
    """

    policy_id: str = "simple-legal"
    switch_probability: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.switch_probability <= 1.0:
            raise ValueError("switch_probability must be between 0 and 1.")

    def select_action(
        self,
        observation: PokeZeroObservationV0,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        legal_moves = legal_move_action_indices(observation.legal_action_mask)
        legal_switches = legal_switch_action_indices(observation.legal_action_mask)
        legal_action_indices(observation.legal_action_mask)

        if legal_switches and (not legal_moves or rng.random() < self.switch_probability):
            action_index = rng.choice(legal_switches)
            action_pool = legal_switches
            action_family = "switch"
            family_probability = 1.0 if not legal_moves else self.switch_probability
        elif legal_moves:
            action_index = rng.choice(legal_moves)
            action_pool = legal_moves
            action_family = "move"
            family_probability = 1.0 if not legal_switches else 1.0 - self.switch_probability

        return PolicyDecision(
            action_index=action_index,
            policy_id=self.policy_id,
            action_probability=family_probability / len(action_pool),
            metadata={"action_family": action_family},
        )


@dataclass
class ScriptedTeacherPolicy:
    """Metadata-backed Gen 3 randbat teacher for bootstrap data generation."""

    policy_id: str = "scripted-teacher"
    showdown_root: Path | str | None = None
    dex: ShowdownDex | None = None
    allow_fallback: bool = False
    allow_unknown_moves: bool = False
    # Move scores are roughly base_power * effectiveness * STAB (0-250+);
    # switch scores are roughly hp*40 plus matchup/context bonuses.
    switch_margin: float = 8.0
    poor_move_threshold: float = 35.0
    team_status_cure_score: float = 64.0
    status_pressure_score: float = 55.0
    statused_switch_penalty: float = 10.0
    low_hp_switch_bonus: float = 35.0
    active_danger_switch_bonus: float = 45.0
    tie_breaker: str = "random"

    def __post_init__(self) -> None:
        if self.tie_breaker not in {"random", "first"}:
            raise ValueError("tie_breaker must be 'random' or 'first'.")

    def select_action(
        self,
        observation: PokeZeroObservationV0,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        dex = self._dex()
        if dex is None:
            return self._fallback(observation, rng=rng, reason="dex unavailable")
        if not observation.metadata:
            return self._fallback(observation, rng=rng, reason="missing observation metadata")

        candidates = _legal_candidate_metadata(observation)
        if not candidates:
            return self._fallback(observation, rng=rng, reason="missing legal candidate metadata")
        unknown_moves = _unknown_legal_move_names(candidates, dex)
        if unknown_moves and not self.allow_unknown_moves:
            raise ValueError(f"scripted-teacher could not resolve legal move(s): {', '.join(unknown_moves)}")
        move_scores = tuple(
            _move_score(
                candidate,
                observation.metadata,
                dex,
                allow_unknown_moves=self.allow_unknown_moves,
                team_status_cure_score=self.team_status_cure_score,
                status_pressure_score=self.status_pressure_score,
            )
            for candidate in candidates
            if candidate.get("kind") == "move"
        )
        switch_scores = tuple(
            _switch_score(
                candidate,
                observation.metadata,
                dex,
                statused_switch_penalty=self.statused_switch_penalty,
                low_hp_switch_bonus=self.low_hp_switch_bonus,
                active_danger_switch_bonus=self.active_danger_switch_bonus,
            )
            for candidate in candidates
            if candidate.get("kind") == "switch"
        )
        if not move_scores and not switch_scores:
            return self._fallback(observation, rng=rng)
        best_move = max(move_scores, key=lambda score: score.score, default=None)
        best_switch = max(switch_scores, key=lambda score: score.score, default=None)
        selected_pool = move_scores
        if best_switch is not None and (
            best_move is None
            or best_move.score < self.poor_move_threshold
            or best_switch.score > best_move.score + self.switch_margin
        ):
            selected_pool = switch_scores
        if not selected_pool:
            return self._fallback(observation, rng=rng)

        best_score = max(score.score for score in selected_pool)
        tied = tuple(score for score in selected_pool if abs(score.score - best_score) < 1e-9)
        selected = min(tied, key=lambda score: score.action_index) if self.tie_breaker == "first" else rng.choice(tied)
        return PolicyDecision(
            action_index=selected.action_index,
            policy_id=self.policy_id,
            action_probability=1.0 / len(tied),
            metadata={
                "policy_family": "scripted-teacher",
                "action_family": selected.kind,
                "teacher_score": selected.score,
                "teacher_reason": selected.reason,
                "teacher_branch": selected.branch,
                "teacher_tie_count": len(tied),
                "teacher_tie_breaker": self.tie_breaker,
            },
        )

    def _fallback(self, observation: PokeZeroObservationV0, *, rng: random.Random, reason: str = "fallback") -> PolicyDecision:
        if not self.allow_fallback:
            raise ValueError(f"scripted-teacher cannot select a teacher action: {reason}")
        decision = SimpleLegalPolicy(policy_id=self.policy_id, switch_probability=0.05).select_action(
            observation,
            rng=rng,
        )
        return PolicyDecision(
            action_index=decision.action_index,
            policy_id=self.policy_id,
            action_probability=decision.action_probability,
            metadata={
                **dict(decision.metadata),
                "policy_family": "scripted-teacher",
                "teacher_reason": reason,
                "teacher_branch": "fallback",
            },
        )

    def _dex(self) -> ShowdownDex | None:
        if self.dex is not None:
            return self.dex
        root = self.showdown_root or os.environ.get("POKEZERO_SHOWDOWN_ROOT")
        if root is None:
            return None
        self.dex = load_showdown_dex_cached(root)
        return self.dex


@dataclass
class MaxDamagePolicy:
    """Fixed baseline that always selects the highest estimated-damage legal move.

    Damage is estimated the same way the scripted teacher scores damaging moves
    (base power x type effectiveness x STAB x accuracy, with priority/recoil/selfdestruct
    adjustments). Status / zero-power moves count as 0 damage. When no move is legal
    (forced switch) it picks a legal switch. Intended as an evaluation/benchmark opponent
    only -- a tougher reference than random-legal -- not for training-data generation.
    """

    policy_id: str = "max-damage"
    showdown_root: Path | str | None = None
    dex: ShowdownDex | None = None

    def select_action(
        self,
        observation: PokeZeroObservationV0,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        dex = self._dex()
        if dex is None:
            raise ValueError(
                "max-damage policy requires a Showdown dex; pass showdown_root or set POKEZERO_SHOWDOWN_ROOT."
            )
        if not observation.metadata:
            return self._random_fallback(observation, rng=rng, reason="missing observation metadata")
        candidates = _legal_candidate_metadata(observation)
        if not candidates:
            return self._random_fallback(observation, rng=rng, reason="missing legal candidate metadata")

        self_types = _metadata_species_types(observation.metadata.get("self_active"), dex)
        opponent_types = _metadata_species_types(observation.metadata.get("opponent_active"), dex)
        hp_fraction = _metadata_hp_fraction(observation.metadata.get("self_active"), default=1.0)
        move_candidates = tuple(c for c in candidates if c.get("kind") == "move")
        switch_candidates = tuple(c for c in candidates if c.get("kind") == "switch")

        if move_candidates:
            scored = [
                (
                    int(c["action_index"]),
                    _max_damage_estimate(c, dex, self_types, opponent_types, hp_fraction),
                )
                for c in move_candidates
            ]
            best = max(score for _, score in scored)
            tied = [index for index, score in scored if abs(score - best) < 1e-9]
            action_index = rng.choice(tied)
            return PolicyDecision(
                action_index=action_index,
                policy_id=self.policy_id,
                action_probability=1.0 / len(tied),
                metadata={"policy_family": "max-damage", "branch": "max_damage_move", "damage_estimate": best},
            )
        if switch_candidates:
            switch_indices = [int(c["action_index"]) for c in switch_candidates]
            action_index = rng.choice(switch_indices)
            return PolicyDecision(
                action_index=action_index,
                policy_id=self.policy_id,
                action_probability=1.0 / len(switch_indices),
                metadata={"policy_family": "max-damage", "branch": "forced_switch"},
            )
        return self._random_fallback(observation, rng=rng, reason="no legal move or switch candidates")

    def _random_fallback(self, observation: PokeZeroObservationV0, *, rng: random.Random, reason: str) -> PolicyDecision:
        legal = legal_action_indices(observation.legal_action_mask)
        action_index = rng.choice(legal)
        return PolicyDecision(
            action_index=action_index,
            policy_id=self.policy_id,
            action_probability=1.0 / len(legal),
            metadata={"policy_family": "max-damage", "branch": "fallback", "reason": reason},
        )

    def _dex(self) -> ShowdownDex | None:
        if self.dex is not None:
            return self.dex
        root = self.showdown_root or os.environ.get("POKEZERO_SHOWDOWN_ROOT")
        if root is None:
            return None
        self.dex = load_showdown_dex_cached(root)
        return self.dex


def _max_damage_estimate(
    candidate: Mapping[str, Any],
    dex: ShowdownDex,
    self_types: tuple[str, ...],
    opponent_types: tuple[str, ...],
    hp_fraction: float,
) -> float:
    """Estimated damage for a legal move candidate; status / 0-power moves count as 0."""
    raw_move_name = str(candidate.get("move_id") or candidate.get("move_name") or "")
    move_id = normalize_id(raw_move_name)
    if move_id in _FORCED_PSEUDO_MOVE_IDS:
        return 1.0  # e.g. recharge: only chosen when it is the sole legal move
    move = dex.move_info(raw_move_name)
    if move is None:
        return 0.0  # unknown move: cannot estimate damage, rank lowest
    if move.gen3_category == "Status" or move.base_power <= 0:
        return 0.0
    return _damaging_move_score(
        int(candidate["action_index"]), move, dex, self_types, opponent_types, hp_fraction
    ).score


@dataclass(frozen=True)
class _ActionScore:
    action_index: int
    kind: str
    score: float
    reason: str
    branch: str = "unknown"


def legal_action_indices(legal_action_mask: Sequence[bool]) -> tuple[int, ...]:
    _require_legal_mask(legal_action_mask)
    legal = tuple(index for index, allowed in enumerate(legal_action_mask) if allowed)
    if not legal:
        raise ValueError("legal_action_mask must contain at least one legal action.")
    return legal


def legal_move_action_indices(legal_action_mask: Sequence[bool]) -> tuple[int, ...]:
    _require_legal_mask(legal_action_mask)
    return tuple(index for index in range(MOVE_ACTION_COUNT) if legal_action_mask[index])


def legal_switch_action_indices(legal_action_mask: Sequence[bool]) -> tuple[int, ...]:
    _require_legal_mask(legal_action_mask)
    return tuple(index for index in range(MOVE_ACTION_COUNT, ACTION_COUNT) if legal_action_mask[index])


def _require_legal_mask(legal_action_mask: Sequence[bool]) -> None:
    if len(legal_action_mask) != ACTION_COUNT:
        raise ValueError(f"legal_action_mask must contain {ACTION_COUNT} values.")


def _legal_candidate_metadata(observation: PokeZeroObservationV0) -> tuple[Mapping[str, Any], ...]:
    raw_candidates = observation.metadata.get("action_candidates") if isinstance(observation.metadata, Mapping) else None
    if not isinstance(raw_candidates, list):
        return ()
    candidates: list[Mapping[str, Any]] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            continue
        action_index = raw_candidate.get("action_index")
        if not isinstance(action_index, int) or action_index < 0 or action_index >= ACTION_COUNT:
            continue
        if not observation.legal_action_mask[action_index]:
            continue
        candidates.append(raw_candidate)
    return tuple(candidates)


def _unknown_legal_move_names(candidates: Sequence[Mapping[str, Any]], dex: ShowdownDex) -> tuple[str, ...]:
    missing: list[str] = []
    for candidate in candidates:
        if candidate.get("kind") != "move":
            continue
        move_name = str(candidate.get("move_id") or candidate.get("move_name") or "")
        if normalize_id(move_name) in _FORCED_PSEUDO_MOVE_IDS:
            continue
        if not dex.move_info(move_name):
            missing.append(str(candidate.get("move_name") or candidate.get("move_id") or "unknown"))
    return tuple(missing)


def _move_score(
    candidate: Mapping[str, Any],
    metadata: Mapping[str, Any],
    dex: ShowdownDex,
    *,
    allow_unknown_moves: bool,
    team_status_cure_score: float,
    status_pressure_score: float,
) -> _ActionScore:
    action_index = int(candidate["action_index"])
    raw_move_name = str(candidate.get("move_id") or candidate.get("move_name") or "")
    raw_move_id = normalize_id(raw_move_name)
    if raw_move_id in _FORCED_PSEUDO_MOVE_IDS:
        display_name = str(candidate.get("move_name") or candidate.get("move_id") or "forced move")
        return _ActionScore(action_index, "move", 1.0, f"{display_name}: forced pseudo-move", "forced_pseudo_move")
    move = dex.move_info(raw_move_name)
    if move is None:
        if not allow_unknown_moves:
            raise ValueError(f"scripted-teacher could not resolve move: {candidate.get('move_name') or candidate.get('move_id')}")
        return _ActionScore(action_index, "move", 12.0, "unknown move", "unknown_move")

    self_types = _metadata_species_types(metadata.get("self_active"), dex)
    opponent_types = _metadata_species_types(metadata.get("opponent_active"), dex)
    hp_fraction = _metadata_hp_fraction(metadata.get("self_active"), default=1.0)
    move_id = normalize_id(move.id or move.name)
    if move_id == "rapidspin":
        return _rapid_spin_score(action_index, move, metadata, dex, self_types, opponent_types, hp_fraction)
    if move.gen3_category == "Status" or move.base_power <= 0:
        return _status_move_score(
            action_index,
            move,
            metadata,
            dex,
            hp_fraction,
            team_status_cure_score=team_status_cure_score,
            status_pressure_score=status_pressure_score,
        )

    return _damaging_move_score(action_index, move, dex, self_types, opponent_types, hp_fraction)


def _damaging_move_score(
    action_index: int,
    move,
    dex: ShowdownDex,
    self_types: tuple[str, ...],
    opponent_types: tuple[str, ...],
    hp_fraction: float,
) -> _ActionScore:
    effectiveness = dex.effectiveness(move.type, opponent_types)
    if effectiveness == 0.0:
        return _ActionScore(action_index, "move", 0.0, f"{move.name} has no effect", "damaging_no_effect")
    stab = 1.5 if move.type in self_types else 1.0
    accuracy = max(0.5, min(1.0, move.accuracy / 100.0 if move.accuracy else 1.0))
    score = move.base_power * effectiveness * stab * accuracy
    if move.priority > 0:
        score += 8.0 * move.priority
    if move.recoil and hp_fraction < 0.35:
        score -= 15.0
    if move.selfdestruct and hp_fraction > 0.35:
        score *= 0.35
    return _ActionScore(
        action_index,
        "move",
        score,
        f"{move.name}: bp={move.base_power} type={move.type} eff={effectiveness:g} stab={stab:g}",
        "damaging_move",
    )


def _status_move_score(
    action_index: int,
    move,
    metadata: Mapping[str, Any],
    dex: ShowdownDex,
    hp_fraction: float,
    *,
    team_status_cure_score: float,
    status_pressure_score: float,
) -> _ActionScore:
    move_id = normalize_id(move.id or move.name)
    opponent_status = _metadata_status(metadata.get("opponent_active"))
    if move_id in {"spikes"}:
        return _spikes_score(action_index, move, metadata)
    if move.status and opponent_status == "none":
        opponent_types = _metadata_species_types(metadata.get("opponent_active"), dex)
        if _status_move_has_no_effect(move, opponent_types, dex):
            type_label = "/".join(opponent_types) if opponent_types else "unknown type"
            return _ActionScore(
                action_index,
                "move",
                4.0,
                f"{move.name}: status has no effect on {type_label}",
                "status_no_effect",
            )
        return _ActionScore(action_index, "move", status_pressure_score, f"{move.name}: status pressure", "status_pressure")
    if move.heal:
        score = 58.0 if hp_fraction < 0.45 else 8.0
        return _ActionScore(action_index, "move", score, f"{move.name}: recovery", "recovery")
    if any(value > 0 for value in move.boosts.values()):
        score = 36.0 if hp_fraction >= 0.55 else 12.0
        return _ActionScore(action_index, "move", score, f"{move.name}: setup", "setup")
    if move_id in {"healbell", "aromatherapy"}:
        status_weight = _team_status_cure_weight(metadata.get("self_team"))
        if status_weight > 0.0:
            score = min(team_status_cure_score, 36.0 + (14.0 * status_weight))
            return _ActionScore(
                action_index,
                "move",
                score,
                f"{move.name}: team status cure weight={status_weight:g}",
                "team_status_cure",
            )
        return _ActionScore(action_index, "move", 10.0, f"{move.name}: no team status", "team_status_cure_no_status")
    return _ActionScore(action_index, "move", 10.0, f"{move.name}: low-impact status", "low_impact_status")


def _status_move_has_no_effect(move, opponent_types: tuple[str, ...], dex: ShowdownDex) -> bool:
    status = normalize_id(move.status)
    normalized_types = {normalize_id(value) for value in opponent_types}
    # In Gen 3, Thunder Wave follows Electric immunity, but other paralysis
    # status moves such as Glare should not inherit Normal-type damage immunity.
    if (
        status == "par"
        and normalize_id(move.type) == "electric"
        and dex.effectiveness(move.type, opponent_types) == 0.0
    ):
        return True
    if status in {"psn", "tox"} and normalized_types.intersection({"poison", "steel"}):
        return True
    if status == "brn" and "fire" in normalized_types:
        return True
    return False


def _rapid_spin_score(
    action_index: int,
    move,
    metadata: Mapping[str, Any],
    dex: ShowdownDex,
    self_types: tuple[str, ...],
    opponent_types: tuple[str, ...],
    hp_fraction: float,
) -> _ActionScore:
    hazard_count = _side_hazard_count(metadata.get("self_side_conditions"))
    if dex.effectiveness(move.type, opponent_types) == 0.0:
        return _ActionScore(action_index, "move", 4.0, f"{move.name}: blocked by Ghost", "rapid_spin_blocked_by_ghost")
    if hazard_count <= 0:
        damage_score = _damaging_move_score(action_index, move, dex, self_types, opponent_types, hp_fraction)
        return _ActionScore(
            action_index,
            "move",
            damage_score.score,
            f"{move.name}: no side hazards; {damage_score.reason}",
            "rapid_spin_no_hazards",
        )
    return _ActionScore(
        action_index,
        "move",
        min(76.0, 58.0 + (10.0 * hazard_count)),
        f"{move.name}: clears hazards={hazard_count}",
        "rapid_spin_clear_hazards",
    )


def _spikes_score(action_index: int, move, metadata: Mapping[str, Any]) -> _ActionScore:
    known_layers = _side_condition_count(
        "spikes",
        metadata.get("opponent_side_conditions"),
        metadata.get("opponent_side_condition_counts"),
    )
    if known_layers >= 3:
        return _ActionScore(action_index, "move", 10.0, f"{move.name}: opponent Spikes already maxed", "spikes_maxed")
    return _ActionScore(action_index, "move", 62.0, f"{move.name}: hazard pressure layers={known_layers}/3", "spikes_available")


def _switch_score(
    candidate: Mapping[str, Any],
    metadata: Mapping[str, Any],
    dex: ShowdownDex,
    *,
    statused_switch_penalty: float,
    low_hp_switch_bonus: float,
    active_danger_switch_bonus: float,
) -> _ActionScore:
    action_index = int(candidate["action_index"])
    pokemon = candidate.get("pokemon")
    if not isinstance(pokemon, Mapping):
        return _ActionScore(action_index, "switch", 0.0, "missing switch target", "switch_missing_target")
    species = str(pokemon.get("species") or "unknown")
    hp_fraction = _metadata_hp_fraction(pokemon, default=0.0)
    candidate_types = _metadata_species_types(pokemon, dex)
    opponent_types = _metadata_species_types(metadata.get("opponent_active"), dex)
    incoming = _opponent_incoming_pressure(metadata, dex, candidate_types, opponent_types)
    active_types = _metadata_species_types(metadata.get("self_active"), dex)
    active_incoming = _opponent_incoming_pressure(metadata, dex, active_types, opponent_types)
    matchup_bonus = 0.0
    if incoming.effectiveness == 0.0:
        matchup_bonus = 35.0
    elif incoming.pressure <= 45.0:
        matchup_bonus = 20.0
    elif incoming.pressure <= 80.0:
        matchup_bonus = 10.0
    elif incoming.pressure >= 140.0:
        matchup_bonus = -35.0
    elif incoming.pressure >= 100.0:
        matchup_bonus = -18.0
    active_hp_fraction = _metadata_hp_fraction(metadata.get("self_active"), default=1.0)
    active_danger_bonus = _active_danger_switch_score(
        active_pressure=active_incoming.pressure,
        switch_pressure=incoming.pressure,
        active_hp_fraction=active_hp_fraction,
        max_bonus=active_danger_switch_bonus,
    )
    preservation_bonus = 0.0
    if active_hp_fraction < 0.35:
        preservation_bonus = ((0.35 - active_hp_fraction) / 0.35) * low_hp_switch_bonus
        preservation_bonus *= hp_fraction * _switch_preservation_scale(incoming.pressure, incoming.effectiveness)
    status_penalty = statused_switch_penalty if _has_status(pokemon) else 0.0
    score = (hp_fraction * 40.0) + matchup_bonus + active_danger_bonus + preservation_bonus - status_penalty
    return _ActionScore(
        action_index,
        "switch",
        score,
        (
            f"switch to {species}: hp={hp_fraction:.2f} incoming={incoming.pressure:.1f} "
            f"eff={incoming.effectiveness:g} source={incoming.source} "
            f"active_incoming={active_incoming.pressure:.1f} danger={active_danger_bonus:.1f} "
            f"preserve={preservation_bonus:.1f} status_penalty={status_penalty:.1f}"
        ),
        "switch",
    )


def _active_danger_switch_score(
    *,
    active_pressure: float,
    switch_pressure: float,
    active_hp_fraction: float,
    max_bonus: float,
) -> float:
    if max_bonus <= 0.0 or active_pressure < 80.0 or switch_pressure >= active_pressure:
        return 0.0
    # Full-health actives may still need to pivot out of a predicted super-effective
    # max-damage hit, while low-health actives deserve stronger preservation pressure.
    hp_pressure = max(0.75, min(1.25, 1.25 - (active_hp_fraction * 0.5)))
    reduction = min(1.0, (active_pressure - switch_pressure) / 160.0)
    return max_bonus * reduction * hp_pressure


@dataclass(frozen=True)
class _IncomingPressure:
    pressure: float
    effectiveness: float
    source: str


def _opponent_incoming_pressure(
    metadata: Mapping[str, Any],
    dex: ShowdownDex,
    candidate_types: tuple[str, ...],
    opponent_types: tuple[str, ...],
) -> _IncomingPressure:
    move_names = _metadata_move_names(metadata.get("opponent_active_possible_moves")) or _metadata_move_names(
        metadata.get("opponent_active_revealed_moves")
    )
    worst_move: _IncomingPressure | None = None
    for move_name in move_names:
        move = dex.move_info(move_name)
        if move is None or move.gen3_category == "Status" or move.base_power <= 0:
            continue
        effectiveness = dex.effectiveness(move.type, candidate_types)
        stab = 1.5 if move.type in opponent_types else 1.0
        accuracy = max(0.5, min(1.0, move.accuracy / 100.0 if move.accuracy else 1.0))
        pressure = move.base_power * effectiveness * stab * accuracy
        candidate = _IncomingPressure(pressure=pressure, effectiveness=effectiveness, source="opponent_moves")
        if worst_move is None or candidate.pressure > worst_move.pressure:
            worst_move = candidate
    if worst_move is not None:
        return worst_move
    if opponent_types:
        fallback = max(
            (
                _IncomingPressure(
                    pressure=80.0 * dex.effectiveness(opponent_type, candidate_types) * 1.5,
                    effectiveness=dex.effectiveness(opponent_type, candidate_types),
                    source="opponent_types",
                )
                for opponent_type in opponent_types
            ),
            key=lambda candidate: candidate.pressure,
            default=_IncomingPressure(pressure=80.0, effectiveness=1.0, source="opponent_types"),
        )
        return fallback
    return _IncomingPressure(pressure=80.0, effectiveness=1.0, source="unknown")


def _metadata_move_names(raw_moves: Any) -> tuple[str, ...]:
    if not isinstance(raw_moves, Sequence) or isinstance(raw_moves, (str, bytes)):
        return ()
    return tuple(str(move) for move in raw_moves if str(move).strip())


def _metadata_species_types(raw_pokemon: Any, dex: ShowdownDex) -> tuple[str, ...]:
    if not isinstance(raw_pokemon, Mapping):
        return ()
    species = raw_pokemon.get("species")
    info = dex.species_info(str(species or ""))
    return info.types if info is not None else ()


def _metadata_hp_fraction(raw_pokemon: Any, *, default: float) -> float:
    if not isinstance(raw_pokemon, Mapping):
        return default
    value = raw_pokemon.get("hp_fraction")
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


def _metadata_status(raw_pokemon: Any) -> str:
    if not isinstance(raw_pokemon, Mapping):
        return "none"
    status = str(raw_pokemon.get("status") or "none")
    return status or "none"


def _switch_preservation_scale(incoming_pressure: float, incoming_effectiveness: float) -> float:
    if incoming_effectiveness > 1.0 or incoming_pressure >= 120.0:
        return 0.0
    if incoming_pressure >= 75.0:
        return 0.25
    if incoming_effectiveness == 1.0:
        return 0.5
    return 1.0


def _team_status_cure_weight(raw_team: Any) -> float:
    if not isinstance(raw_team, Sequence) or isinstance(raw_team, (str, bytes)):
        return 0.0
    total = 0.0
    for pokemon in raw_team:
        if _has_status(pokemon):
            total += _STATUS_CURE_WEIGHTS.get(_metadata_status(pokemon).lower(), 1.0)
    return total


def _side_hazard_count(raw_conditions: Any) -> int:
    if not isinstance(raw_conditions, Sequence) or isinstance(raw_conditions, (str, bytes)):
        return 0
    return sum(1 for condition in raw_conditions if normalize_id(str(condition)) in _SIDE_HAZARDS)


def _side_condition_count(condition: str, raw_conditions: Any, raw_counts: Any) -> int:
    normalized = normalize_id(condition)
    if isinstance(raw_counts, Mapping):
        value = raw_counts.get(normalized)
        if isinstance(value, (int, float)):
            return max(0, int(value))
    if not isinstance(raw_conditions, Sequence) or isinstance(raw_conditions, (str, bytes)):
        return 0
    return sum(1 for raw_condition in raw_conditions if normalize_id(str(raw_condition)) == normalized)


def _has_status(raw_pokemon: Any) -> bool:
    status = _metadata_status(raw_pokemon).lower()
    if status in {"", "none", "fnt", "unknown"}:
        return False
    hp_fraction = _metadata_hp_fraction(raw_pokemon, default=1.0)
    return hp_fraction > 0.0
