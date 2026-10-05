"""Opt-in trajectory MCTS reference kernel, not a qualified Pokémon experiment.

Implements Wang (MIT, 2024), pp. 21-22 and 26-28: persistent Q/N/M/P,
fresh determinization per trajectory, first-new-state leaves, sampled chance,
own-policy opponent sampling, and maximum-visit selection. A simulator adapter
must supply public/player-known state identities, exact randbats sampling and
private-safe inference. Those obligations are NOT established by this kernel.
Single-worker execution is an explicit deviation from the paper's 20 workers.
No production policy imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import random
import time
from typing import Callable, Protocol


class ReferenceRefusal(ValueError):
    """Invalid state, work, or inference is evidence of refusal, not a fallback."""


@dataclass(frozen=True)
class DecisionState:
    # Adapter-owned public/own-private information identity, NEVER latent truth.
    key: bytes
    actions: tuple[str, ...]
    faint_count: int

    def __post_init__(self) -> None:
        if (not isinstance(self.key, bytes) or not self.key or not self.actions
                or not isinstance(self.actions, tuple)
                or any(not isinstance(a, str) or not a for a in self.actions)
                or len(set(self.actions)) != len(self.actions)
                or type(self.faint_count) is not int or not 0 <= self.faint_count <= 12):
            raise ReferenceRefusal("invalid reference decision state")


@dataclass(frozen=True)
class Evaluation:
    priors: tuple[float, ...]
    # Subject-seat terminal-reward scale [-1, 1], not native win probability.
    value: float


def signed_win_probability(probability: float) -> float:
    if (type(probability) not in (float, int) or not math.isfinite(probability)
            or not 0 <= probability <= 1):
        raise ReferenceRefusal("critic is not a finite win probability")
    return 2.0 * probability - 1.0


def _value(value: float) -> float:
    if (type(value) not in (float, int) or not math.isfinite(value)
            or not -1 <= value <= 1):
        raise ReferenceRefusal("critic/terminal value must be subject-signed in [-1, 1]")
    return float(value)


def _distribution(priors: tuple[float, ...], size: int) -> tuple[float, ...]:
    if (not isinstance(priors, tuple) or len(priors) != size or not priors
            or any(type(p) not in (float, int) or not math.isfinite(p) or p < 0 for p in priors)
            or not math.isclose(math.fsum(priors), 1.0, rel_tol=0, abs_tol=2e-5)):
        raise ReferenceRefusal("policy must supply its complete normalized legal distribution")
    # Float32 model softmax has rounding error; reject genuine lost mass first.
    total = math.fsum(priors)
    return tuple(float(p) / total for p in priors)


@dataclass(frozen=True)
class Frame:
    # None is a real opponent-only replacement boundary, never a synthetic WAIT.
    subject: DecisionState | None
    opponent_actions: tuple[str, ...] = ()
    opponent_priors: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.opponent_actions, tuple)
                or any(not isinstance(a, str) or not a for a in self.opponent_actions)
                or len(set(self.opponent_actions)) != len(self.opponent_actions)
                or (self.subject is not None and not isinstance(self.subject, DecisionState))):
            raise ReferenceRefusal("invalid reference request boundary")
        if self.opponent_actions:
            _distribution(self.opponent_priors, len(self.opponent_actions))
        elif self.opponent_priors or self.subject is None:
            raise ReferenceRefusal("empty or inconsistent reference request boundary")


@dataclass(frozen=True)
class Terminal:
    value: float

    def __post_init__(self) -> None:
        _value(self.value)
        if self.value not in (-1, 0, 1):
            raise ReferenceRefusal("terminal must be an actual win, loss, or draw")


class World(Protocol):
    """One freshly sampled hypothetical world, never a live hidden snapshot.

    ``advance`` samples simulator chance using only its supplied chance RNG.
    ``frame`` emits the opponent's OWN masked policy from this sampled world;
    it must not read another world's party slots or the live opponent request.
    ``evaluate`` sees subject-relative inputs and returns the same champion.
    ``close`` releases this trajectory's resources, including on refusal.
    """

    def frame(self) -> Frame: ...
    def advance(self, subject_action: str | None, opponent_action: str | None,
                chance_rng: random.Random) -> Frame | Terminal: ...
    def evaluate(self, state: DecisionState) -> Evaluation: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class ReferenceConfig:
    # Both constants are chosen/preregistered, not claimed as unpublished paper settings.
    alpha: float
    beta: float
    max_transitions: int = 256

    def __post_init__(self) -> None:
        if any(type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 1
               for x in (self.alpha, self.beta)):
            raise ReferenceRefusal("paper-scale alpha and beta must lie in [0, 1]")
        if type(self.max_transitions) is not int or self.max_transitions <= 0:
            raise ReferenceRefusal("transition safety limit must be a positive integer")


@dataclass
class Node:
    state: DecisionState
    priors: tuple[float, ...]
    visits: list[int] = field(default_factory=list)
    totals: list[float] = field(default_factory=list)
    count: int = 0

    @classmethod
    def from_evaluation(cls, state: DecisionState, evaluation: Evaluation) -> Node:
        _value(evaluation.value)
        priors = _distribution(evaluation.priors, len(state.actions))
        return cls(state, priors, [0] * len(priors), [0.0] * len(priors))

    def select(self, config: ReferenceConfig) -> int:
        # Empty Q is zero; ties resolve in the adapter's fixed action order.
        return max(range(len(self.priors)), key=lambda i: (
            (self.totals[i] / self.visits[i] if self.visits[i] else 0.0)
            + config.alpha * self.priors[i] ** config.beta
            * math.sqrt(self.count) / (1 + self.visits[i]), -i))


@dataclass(frozen=True)
class SearchResult:
    action: str
    trajectories: int
    transitions: int
    world_draws: int
    root_visits: tuple[int, ...]
    root_means: tuple[float | None, ...]
    elapsed_seconds: float
    deadline_exhausted: bool
    deadline_overrun_seconds: float


def _rng(seed: int, ordinal: int, domain: str) -> random.Random:
    digest = hashlib.sha256(b"pokezero.paper-reference.v1\0" + seed.to_bytes(8, "big")
                            + ordinal.to_bytes(8, "big") + domain.encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


class TrajectorySearch:
    """Battle-scoped reference statistics. Refusals poison reuse until reset.

    Repeated calls retain statistics; ``reset_battle`` is mandatory on a new
    battle. Pruning uses only the real current faint count, not simulated KOs.
    This deliberately has no production fallback or leaf-depth truncation.
    """

    def __init__(self, config: ReferenceConfig) -> None:
        self.config = config
        self.nodes: dict[bytes, Node] = {}
        self._battle_id: str | None = None
        self._faint_floor = 0
        self._ordinal = 0
        self._usable = True

    def reset_battle(self, battle_id: str) -> None:
        if not isinstance(battle_id, str) or not battle_id:
            raise ReferenceRefusal("reference requires an explicit battle identity")
        self.nodes.clear()
        self._battle_id = battle_id
        self._faint_floor = self._ordinal = 0
        self._usable = True

    def _existing(self, state: DecisionState) -> Node | None:
        node = self.nodes.get(state.key)
        if node is not None and node.state != state:
            raise ReferenceRefusal("information-state key aliased different legal actions or faint count")
        return node

    def _trajectory(self, world: World, root: DecisionState,
                    opponent_rng: random.Random, chance_rng: random.Random,
                    expired: Callable[[], bool]) -> tuple[int, bool]:
        frame = world.frame()
        if not isinstance(frame, Frame) or frame.subject != root:
            raise ReferenceRefusal("sampled world changed the player-known root or legality")
        path: list[tuple[Node, int]] = []
        leaf: Node | None = None
        transitions = 0
        faint_floor = root.faint_count
        while True:
            if expired():
                # No incomplete trajectory enters Q/N/M, including a staged new leaf.
                return transitions, False
            if frame.subject is not None:
                state = frame.subject
                if state.faint_count < faint_floor:
                    raise ReferenceRefusal("trajectory decreased monotone faint count")
                faint_floor = state.faint_count
                node = self._existing(state)
                if node is None:
                    evaluation = world.evaluate(state)
                    leaf = Node.from_evaluation(state, evaluation)
                    value = _value(evaluation.value)
                    break
                index = node.select(self.config)
                subject_action = state.actions[index]
                path.append((node, index))
            else:
                subject_action = None
            if transitions >= self.config.max_transitions:
                raise ReferenceRefusal("transition safety limit reached; truncated value is not accepted")
            opponent_action = None
            if frame.opponent_actions:
                opponent_action = opponent_rng.choices(
                    frame.opponent_actions, weights=frame.opponent_priors, k=1)[0]
            outcome = world.advance(subject_action, opponent_action, chance_rng)
            transitions += 1
            if isinstance(outcome, Terminal):
                value = _value(outcome.value)
                break
            if not isinstance(outcome, Frame):
                raise ReferenceRefusal("simulator returned neither a request boundary nor a terminal")
            frame = outcome
        if expired():
            return transitions, False
        if leaf is not None:
            self.nodes[leaf.state.key] = leaf
        # Sampled trajectory mean, NOT exact chance expectation or sign-alternating minimax.
        for node, index in path:
            node.visits[index] += 1
            node.totals[index] += value
            node.count += 1
        return transitions, True

    def search(self, root: DecisionState, *, battle_id: str,
               evaluate_root: Callable[[DecisionState], Evaluation],
               sample_world: Callable[[random.Random], World], seed: int,
               trajectories: int, deadline_seconds: float | None = None,
               clock: Callable[[], float] = time.perf_counter) -> SearchResult:
        started = clock()  # Includes root inference, determinization, simulator and cleanup.
        if (battle_id != self._battle_id or not self._usable
                or type(seed) is not int or not 0 <= seed < 2**64
                or type(trajectories) is not int or trajectories <= 0
                or (deadline_seconds is not None and (type(deadline_seconds) not in (int, float)
                    or not math.isfinite(deadline_seconds) or deadline_seconds <= 0))):
            raise ReferenceRefusal("invalid battle/work/deadline contract; reset battle before reuse")
        deadline = None if deadline_seconds is None else started + deadline_seconds
        expired = lambda: deadline is not None and clock() >= deadline
        completed = transitions = draws = 0
        try:
            if root.faint_count < self._faint_floor:
                raise ReferenceRefusal("real faint count decreased without a battle reset")
            self._faint_floor = root.faint_count
            self.nodes = {key: node for key, node in self.nodes.items()
                          if node.state.faint_count >= self._faint_floor}
            node = self._existing(root)
            if node is None:
                node = Node.from_evaluation(root, evaluate_root(root))
                self.nodes[root.key] = node
            for _ in range(trajectories):
                if expired():
                    break
                ordinal = self._ordinal
                self._ordinal += 1
                world = sample_world(_rng(seed, ordinal, "hidden"))
                draws += 1
                try:
                    steps, backed = self._trajectory(world, root, _rng(seed, ordinal, "opponent"),
                                                    _rng(seed, ordinal, "chance"), expired)
                    transitions += steps
                    completed += int(backed)
                finally:
                    world.close()
                if not backed:
                    break
            if completed == 0:
                raise ReferenceRefusal("zero complete trajectories; no stale-statistics or raw-policy fallback")
            action_index = max(range(len(root.actions)), key=lambda i: (node.visits[i], -i))
            elapsed = clock() - started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ReferenceRefusal("invalid decision clock")
            return SearchResult(root.actions[action_index], completed, transitions, draws,
                tuple(node.visits), tuple(total / count if count else None
                    for total, count in zip(node.totals, node.visits)), elapsed,
                deadline is not None and started + elapsed >= deadline,
                0.0 if deadline_seconds is None else max(0.0, elapsed - deadline_seconds))
        except Exception:
            # A simulator/cleanup/inference refusal cannot leave a reusable successful tree.
            self._usable = False
            raise
