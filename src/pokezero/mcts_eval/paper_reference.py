"""Opt-in trajectory MCTS reference kernel, not a qualified Pokémon experiment.

Implements Wang (MIT, 2024), pp. 21-22 and 26-28: persistent Q/N/M/P,
fresh determinization per trajectory, first-new-state leaves, sampled chance,
own-policy opponent sampling, and maximum-visit selection. A simulator adapter
must supply public/player-known state identities, exact randbats sampling and
private-safe inference. Those obligations are NOT established by this kernel.
This kernel is one worker; paper_reference_parallel supplies the process pool.
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


@dataclass(frozen=True)
class BatchResult:
    """Worker work receipt, never a decision or a stale-action fallback.

    A shared deadline can expire before an individual worker finishes any
    trajectory. Only the coordinator can decide whether aggregate NEW work
    suffices. An empty batch is not a simulator failure and does not poison a
    worker, but ordinary single-worker ``search`` still refuses zero work.
    """

    trajectories: int
    transitions: int
    world_draws: int
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
        self._local_statistics = {}
        self._shared_statistics = {}
        self._master_version = -1
        self._exported_statistics = None
        self._last_master_snapshot = None

    def reset_battle(self, battle_id: str) -> None:
        if not isinstance(battle_id, str) or not battle_id:
            raise ReferenceRefusal("reference requires an explicit battle identity")
        self.nodes.clear()
        self._battle_id = battle_id
        self._faint_floor = self._ordinal = 0
        self._usable = True
        self._local_statistics.clear()
        self._shared_statistics.clear()
        self._master_version = -1
        self._exported_statistics = None
        self._last_master_snapshot = None

    def export_statistics(self, *, worker_id: str, sequence: int):
        from .paper_reference_exchange import WorkerUpdate
        if not self._usable or self._battle_id is None:
            raise ReferenceRefusal("cannot export poisoned or unbound worker statistics")
        message = WorkerUpdate(self._battle_id, worker_id, sequence,
            tuple(r for _, r in sorted(self._local_statistics.items())
                  if r.state.faint_count >= self._faint_floor))
        old = self._exported_statistics
        if old is not None:
            if old.worker_id != worker_id or sequence < old.sequence:
                raise ReferenceRefusal("worker identity/sequence changed before battle reset")
            old_rows = tuple(r for r in old.rows if r.state.faint_count >= self._faint_floor)
            if sequence == old.sequence and message.rows != old_rows:
                raise ReferenceRefusal("same export sequence has different own evidence")
        self._exported_statistics = message
        return message

    def merge_statistics(self, snapshot) -> None:
        from .paper_reference_exchange import MasterSnapshot, _nonregressing
        if (not isinstance(snapshot, MasterSnapshot) or not self._usable
                or snapshot.battle_id != self._battle_id
                or snapshot.faint_floor < self._faint_floor):
            raise ReferenceRefusal("invalid master/worker battle or faint-floor contract")
        if snapshot.version < self._master_version:
            raise ReferenceRefusal("worker cannot roll back to an older master snapshot")
        if snapshot.version == self._master_version and snapshot != self._last_master_snapshot:
            raise ReferenceRefusal("same master version has different evidence or metadata")
        shared = {r.state.key: r for r in snapshot.rows}
        sent = self._exported_statistics
        own = {key: row for key, row in self._local_statistics.items()
               if row.state.faint_count >= snapshot.faint_floor}
        if own:
            exported = {} if sent is None else {r.state.key: r for r in sent.rows
                                               if r.state.faint_count >= snapshot.faint_floor}
            if own != exported:
                raise ReferenceRefusal("master merge would overwrite unexported completed work")
            if dict(snapshot.acknowledged).get(sent.worker_id) != sent.sequence:
                raise ReferenceRefusal("master has not acknowledged the latest worker contribution")
        for key, old in self._shared_statistics.items():
            if old.state.faint_count >= snapshot.faint_floor:
                if key not in shared:
                    raise ReferenceRefusal("master discarded unpruned shared evidence")
                _nonregressing(old, shared[key])
        for key, local in self._local_statistics.items():
            if local.state.faint_count >= snapshot.faint_floor:
                if key not in shared:
                    raise ReferenceRefusal("master omitted the worker's own completed evidence")
                _nonregressing(local, shared[key])
        for key, row in shared.items():
            self._existing(row.state)
        # Preserve local priors. Remote known states acquire priors lazily when
        # encountered; imported evidence is never marked as local work.
        self.nodes = {key: node for key, node in self.nodes.items()
                      if node.state.faint_count >= snapshot.faint_floor}
        for key, node in self.nodes.items():
            if key in shared:
                row = shared[key]
                node.visits[:] = row.visits
                node.totals[:] = row.totals
                node.count = row.count
        self._shared_statistics = shared
        self._local_statistics = {key: row for key, row in self._local_statistics.items()
                                  if row.state.faint_count >= snapshot.faint_floor}
        self._faint_floor = snapshot.faint_floor
        self._master_version = snapshot.version
        self._last_master_snapshot = snapshot

    def _discovered(self, node: Node) -> None:
        from .paper_reference_exchange import Statistics
        if node.state.key not in self._local_statistics:
            self._local_statistics[node.state.key] = Statistics(node.state,
                (0,) * len(node.priors), (0.,) * len(node.priors), 0)

    def _own_backup(self, node: Node, index: int, value: float) -> None:
        from .paper_reference_exchange import Statistics
        self._discovered(node)
        old = self._local_statistics[node.state.key]
        visits, totals = list(old.visits), list(old.totals)
        visits[index] += 1
        totals[index] += value
        self._local_statistics[node.state.key] = Statistics(node.state,
            tuple(visits), tuple(totals), old.count + 1)

    def _node_from_evaluation(self, state: DecisionState, evaluation: Evaluation) -> Node:
        node = Node.from_evaluation(state, evaluation)
        row = self._shared_statistics.get(state.key)
        if row is not None:
            if row.state != state:
                raise ReferenceRefusal("remote information-state identity changed")
            node.visits[:], node.totals[:], node.count = row.visits, row.totals, row.count
        return node

    def _existing(self, state: DecisionState) -> Node | None:
        node = self.nodes.get(state.key)
        if node is not None and node.state != state:
            raise ReferenceRefusal("information-state key aliased different legal actions or faint count")
        shared = self._shared_statistics.get(state.key)
        if shared is not None and shared.state != state:
            raise ReferenceRefusal("imported information-state key aliased different actions or faint count")
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
                    node = self._node_from_evaluation(state, evaluation)
                    if state.key not in self._shared_statistics:
                        leaf = node
                        value = _value(evaluation.value)
                        break
                    # Another worker already expanded this information state.
                    # Recompute P locally, but keep traversing its shared tree.
                    self.nodes[state.key] = node
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
            self._discovered(leaf)
        # Sampled trajectory mean, NOT exact chance expectation or sign-alternating minimax.
        for node, index in path:
            node.visits[index] += 1
            node.totals[index] += value
            node.count += 1
            self._own_backup(node, index, value)
        return transitions, True

    def search(self, root: DecisionState, *, battle_id: str,
               evaluate_root: Callable[[DecisionState], Evaluation],
               sample_world: Callable[[random.Random], World], seed: int,
               trajectories: int, deadline_seconds: float | None = None,
               clock: Callable[[], float] = time.perf_counter) -> SearchResult:
        result = self._run_batch(root, battle_id=battle_id, evaluate_root=evaluate_root,
            sample_world=sample_world, seed=seed, trajectories=trajectories,
            deadline_seconds=deadline_seconds, deadline_at=None, clock=clock,
            require_complete=True)
        assert isinstance(result, SearchResult)
        return result

    def search_batch(self, root: DecisionState, *, battle_id: str,
                     evaluate_root: Callable[[DecisionState], Evaluation],
                     sample_world: Callable[[random.Random], World], seed: int,
                     trajectories: int, deadline_at: float | None = None,
                     clock: Callable[[], float] = time.perf_counter) -> BatchResult:
        """At most one exchange batch, using the coordinator's absolute clock."""
        result = self._run_batch(root, battle_id=battle_id, evaluate_root=evaluate_root,
            sample_world=sample_world, seed=seed, trajectories=trajectories,
            deadline_seconds=None, deadline_at=deadline_at, clock=clock,
            require_complete=False)
        assert isinstance(result, BatchResult)
        return result

    def _run_batch(self, root: DecisionState, *, battle_id: str,
                   evaluate_root: Callable[[DecisionState], Evaluation],
                   sample_world: Callable[[random.Random], World], seed: int,
                   trajectories: int, deadline_seconds: float | None,
                   deadline_at: float | None, clock: Callable[[], float],
                   require_complete: bool) -> BatchResult | SearchResult:
        started = clock()  # Includes root inference, determinization, simulator and cleanup.
        if (battle_id != self._battle_id or not self._usable
                or type(seed) is not int or not 0 <= seed < 2**64
                or type(trajectories) is not int or trajectories <= 0
                or (deadline_seconds is not None and (type(deadline_seconds) not in (int, float)
                    or not math.isfinite(deadline_seconds) or deadline_seconds <= 0))
                or (deadline_at is not None and (type(deadline_at) not in (int, float)
                    or not math.isfinite(deadline_at)))
                or (deadline_seconds is not None and deadline_at is not None)):
            raise ReferenceRefusal("invalid battle/work/deadline contract; reset battle before reuse")
        deadline = deadline_at if deadline_at is not None else (
            None if deadline_seconds is None else started + deadline_seconds)
        expired = lambda: deadline is not None and clock() >= deadline
        completed = transitions = draws = 0
        try:
            if root.faint_count < self._faint_floor:
                raise ReferenceRefusal("real faint count decreased without a battle reset")
            self._faint_floor = root.faint_count
            self.nodes = {key: node for key, node in self.nodes.items()
                          if node.state.faint_count >= self._faint_floor}
            self._shared_statistics = {key: row for key, row in self._shared_statistics.items()
                                      if row.state.faint_count >= self._faint_floor}
            self._local_statistics = {key: row for key, row in self._local_statistics.items()
                                     if row.state.faint_count >= self._faint_floor}
            node = self._existing(root)
            if node is None and not expired():
                node = self._node_from_evaluation(root, evaluate_root(root))
                self.nodes[root.key] = node
                self._discovered(node)
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
            if completed == 0 and require_complete:
                raise ReferenceRefusal("zero complete trajectories; no stale-statistics or raw-policy fallback")
            if require_complete:
                node = self.nodes[root.key]
                action_index = max(range(len(root.actions)), key=lambda i: (node.visits[i], -i))
            elapsed = clock() - started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ReferenceRefusal("invalid decision clock")
            exhausted = deadline is not None and started + elapsed >= deadline
            overrun = 0.0 if deadline is None else max(0.0, started + elapsed - deadline)
            if require_complete:
                return SearchResult(root.actions[action_index], completed, transitions, draws,
                    tuple(node.visits), tuple(q / n if n else None
                        for q, n in zip(node.totals, node.visits)), elapsed, exhausted, overrun)
            return BatchResult(completed, transitions, draws, elapsed, exhausted, overrun)
        except Exception:
            # A simulator/cleanup/inference refusal cannot leave a reusable successful tree.
            self._usable = False
            raise
