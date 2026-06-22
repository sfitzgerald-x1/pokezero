"""Policy interfaces and baseline policies for early rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import random
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .actions import ACTION_COUNT, MOVE_ACTION_COUNT
from .dex import ShowdownDex, load_showdown_dex_cached, normalize_id
from .observation import PokeZeroObservationV0


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


@dataclass(frozen=True)
class PolicyDecision:
    action_index: int
    policy_id: str
    action_probability: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_index < 0 or self.action_index >= ACTION_COUNT:
            raise ValueError(f"action_index must be between 0 and {ACTION_COUNT - 1}.")
        if self.action_probability is not None and not 0.0 <= self.action_probability <= 1.0:
            raise ValueError("action_probability must be between 0 and 1 when set.")


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
    statused_switch_penalty: float = 10.0
    low_hp_switch_bonus: float = 35.0

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
        selected = rng.choice(tied)
        return PolicyDecision(
            action_index=selected.action_index,
            policy_id=self.policy_id,
            action_probability=1.0 / len(tied),
            metadata={
                "policy_family": "scripted-teacher",
                "action_family": selected.kind,
                "teacher_score": selected.score,
                "teacher_reason": selected.reason,
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
            metadata={**dict(decision.metadata), "policy_family": "scripted-teacher", "teacher_reason": reason},
        )

    def _dex(self) -> ShowdownDex | None:
        if self.dex is not None:
            return self.dex
        root = self.showdown_root or os.environ.get("POKEZERO_SHOWDOWN_ROOT")
        if root is None:
            return None
        self.dex = load_showdown_dex_cached(root)
        return self.dex


@dataclass(frozen=True)
class _ActionScore:
    action_index: int
    kind: str
    score: float
    reason: str


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
) -> _ActionScore:
    action_index = int(candidate["action_index"])
    raw_move_name = str(candidate.get("move_id") or candidate.get("move_name") or "")
    raw_move_id = normalize_id(raw_move_name)
    if raw_move_id in _FORCED_PSEUDO_MOVE_IDS:
        display_name = str(candidate.get("move_name") or candidate.get("move_id") or "forced move")
        return _ActionScore(action_index, "move", 1.0, f"{display_name}: forced pseudo-move")
    move = dex.move_info(raw_move_name)
    if move is None:
        if not allow_unknown_moves:
            raise ValueError(f"scripted-teacher could not resolve move: {candidate.get('move_name') or candidate.get('move_id')}")
        return _ActionScore(action_index, "move", 12.0, "unknown move")

    self_types = _metadata_species_types(metadata.get("self_active"), dex)
    opponent_types = _metadata_species_types(metadata.get("opponent_active"), dex)
    hp_fraction = _metadata_hp_fraction(metadata.get("self_active"), default=1.0)
    move_id = normalize_id(move.id or move.name)
    if move_id == "rapidspin":
        return _rapid_spin_score(action_index, move, metadata, dex, self_types, opponent_types, hp_fraction)
    if move.gen3_category == "Status" or move.base_power <= 0:
        return _status_move_score(action_index, move, metadata, hp_fraction, team_status_cure_score=team_status_cure_score)

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
        return _ActionScore(action_index, "move", 0.0, f"{move.name} has no effect")
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
    )


def _status_move_score(
    action_index: int,
    move,
    metadata: Mapping[str, Any],
    hp_fraction: float,
    *,
    team_status_cure_score: float,
) -> _ActionScore:
    move_id = normalize_id(move.id or move.name)
    opponent_status = _metadata_status(metadata.get("opponent_active"))
    if move_id in {"spikes"}:
        return _spikes_score(action_index, move, metadata)
    if move.status and opponent_status == "none":
        return _ActionScore(action_index, "move", 55.0, f"{move.name}: status pressure")
    if move.heal:
        score = 58.0 if hp_fraction < 0.45 else 8.0
        return _ActionScore(action_index, "move", score, f"{move.name}: recovery")
    if any(value > 0 for value in move.boosts.values()):
        score = 36.0 if hp_fraction >= 0.55 else 12.0
        return _ActionScore(action_index, "move", score, f"{move.name}: setup")
    if move_id in {"healbell", "aromatherapy"}:
        status_weight = _team_status_cure_weight(metadata.get("self_team"))
        if status_weight > 0.0:
            score = min(team_status_cure_score, 36.0 + (14.0 * status_weight))
            return _ActionScore(
                action_index,
                "move",
                score,
                f"{move.name}: team status cure weight={status_weight:g}",
            )
        return _ActionScore(action_index, "move", 10.0, f"{move.name}: no team status")
    return _ActionScore(action_index, "move", 10.0, f"{move.name}: low-impact status")


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
        return _ActionScore(action_index, "move", 4.0, f"{move.name}: blocked by Ghost")
    if hazard_count <= 0:
        damage_score = _damaging_move_score(action_index, move, dex, self_types, opponent_types, hp_fraction)
        return _ActionScore(
            action_index,
            "move",
            damage_score.score,
            f"{move.name}: no side hazards; {damage_score.reason}",
        )
    return _ActionScore(action_index, "move", min(76.0, 58.0 + (10.0 * hazard_count)), f"{move.name}: clears hazards={hazard_count}")


def _spikes_score(action_index: int, move, metadata: Mapping[str, Any]) -> _ActionScore:
    known_layers = _side_condition_count(
        "spikes",
        metadata.get("opponent_side_conditions"),
        metadata.get("opponent_side_condition_counts"),
    )
    if known_layers >= 3:
        return _ActionScore(action_index, "move", 10.0, f"{move.name}: opponent Spikes already maxed")
    score = 62.0 - (8.0 * known_layers)
    return _ActionScore(action_index, "move", score, f"{move.name}: hazard pressure layers={known_layers}/3")


def _switch_score(
    candidate: Mapping[str, Any],
    metadata: Mapping[str, Any],
    dex: ShowdownDex,
    *,
    statused_switch_penalty: float,
    low_hp_switch_bonus: float,
) -> _ActionScore:
    action_index = int(candidate["action_index"])
    pokemon = candidate.get("pokemon")
    if not isinstance(pokemon, Mapping):
        return _ActionScore(action_index, "switch", 0.0, "missing switch target")
    species = str(pokemon.get("species") or "unknown")
    hp_fraction = _metadata_hp_fraction(pokemon, default=0.0)
    candidate_types = _metadata_species_types(pokemon, dex)
    opponent_types = _metadata_species_types(metadata.get("opponent_active"), dex)
    incoming = max((dex.effectiveness(opponent_type, candidate_types) for opponent_type in opponent_types), default=1.0)
    matchup_bonus = 0.0
    if incoming == 0.0:
        matchup_bonus = 35.0
    elif incoming < 1.0:
        matchup_bonus = 20.0
    elif incoming > 1.0:
        matchup_bonus = -22.0
    active_hp_fraction = _metadata_hp_fraction(metadata.get("self_active"), default=1.0)
    preservation_bonus = 0.0
    if active_hp_fraction < 0.35:
        preservation_bonus = ((0.35 - active_hp_fraction) / 0.35) * low_hp_switch_bonus
        preservation_bonus *= hp_fraction * _switch_preservation_scale(incoming)
    status_penalty = statused_switch_penalty if _has_status(pokemon) else 0.0
    score = (hp_fraction * 40.0) + matchup_bonus + preservation_bonus - status_penalty
    return _ActionScore(
        action_index,
        "switch",
        score,
        (
            f"switch to {species}: hp={hp_fraction:.2f} incoming={incoming:g} "
            f"preserve={preservation_bonus:.1f} status_penalty={status_penalty:.1f}"
        ),
    )


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


def _switch_preservation_scale(incoming_effectiveness: float) -> float:
    if incoming_effectiveness > 1.0:
        return 0.0
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
