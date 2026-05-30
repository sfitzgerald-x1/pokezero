"""Policy interfaces and baseline policies for early rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .actions import ACTION_COUNT, MOVE_ACTION_COUNT
from .observation import PokeZeroObservationV0


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
