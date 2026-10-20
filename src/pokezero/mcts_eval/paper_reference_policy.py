"""Opt-in controlled-bridge policy for the persistent trajectory reference.

This is the existing one-snapshot model runtime, not staged collector or
scientific qualification. No live opponent request or trajectory is transported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Any

from ..policy import PolicyContext, PolicyDecision
from .paper_reference import ReferenceConfig, ReferenceRefusal
from .paper_reference_parallel import ParallelRefusal, ParallelTrajectorySearch, sanitized_worker_failures
from .paper_reference_runtime import PublicRootRequest, ShowdownWorkerFactory
from .paper_reference_showdown import decision_state


@dataclass(frozen=True)
class ReferencePolicyConfig:
    decision_seconds: float
    workers: int = 20
    alpha: float = .5
    beta: float = 1.
    max_transitions: int = 256
    allow_earlier_compatible_template: bool = False

    def __post_init__(self) -> None:
        ReferenceConfig(self.alpha, self.beta, self.max_transitions)
        if (type(self.decision_seconds) not in (int, float)
                or not math.isfinite(self.decision_seconds) or self.decision_seconds <= 0):
            raise ReferenceRefusal("reference decision ceiling must be finite and positive")
        if type(self.workers) is not int or not 1 <= self.workers <= 20:
            raise ReferenceRefusal("reference workers must be an integer in [1, 20]")
        if type(self.allow_earlier_compatible_template) is not bool:
            raise ReferenceRefusal("reference compatible-template adaptation must be explicit bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_seconds": self.decision_seconds, "workers": self.workers,
            "alpha": self.alpha, "beta": self.beta,
            "max_transitions": self.max_transitions, "batch_size": 10,
            "allow_earlier_compatible_template": self.allow_earlier_compatible_template,
            "device": "cpu", "leaf": "unchanged_one_snapshot_model",
            "selector": "cumulative_root_visit_max", "raw_fallback": False,
            "atomic_work_instrumented": False, "scientifically_qualified": False,
            "complete_workflow_cost_qualified": False,
            "cpu_seconds": None, "cpu_accounting": "not_instrumented",
        }


@dataclass
class ReferencePolicyStats:
    binding: dict[str, Any] = field(default_factory=dict)
    searches: int = 0
    forced_decisions: int = 0
    refusals: int = 0
    trajectories: int = 0
    transitions: int = 0
    world_draws: int = 0
    neural_forwards: int = 0
    decision_wall_seconds: float = 0.
    pool_startup_seconds: float = 0.
    deadline_overrun_seconds: float = 0.
    last_decision: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {name: (dict(value) if isinstance(value, dict) else value)
                for name, value in vars(self).items()}


class TrajectoryReferencePolicy:
    requires_public_materialization_state = True

    def __init__(self, *, config: ReferencePolicyConfig,
                 factory: ShowdownWorkerFactory, policy_id: str) -> None:
        self.config, self.factory = config, factory
        self.policy_id = f"{policy_id}+trajectory-reference"
        self.stats = ReferencePolicyStats(binding={
            "checkpoint_sha256": factory.checkpoint_sha256,
            "set_source_hash": factory.set_source_hash,
            "showdown_root": factory.showdown_root,
            "node_binary": factory.node_binary,
        })
        self._pool: ParallelTrajectorySearch | None = None
        self._closed = self._failed = False

    def select_action(self, observation: Any, *, rng: random.Random) -> PolicyDecision:
        raise ReferenceRefusal("trajectory reference requires public materialization context")

    def select_action_with_context(self, context: PolicyContext,
                                   *, rng: random.Random) -> PolicyDecision:
        if self._closed or self._failed:
            raise ReferenceRefusal("trajectory reference is closed or refused; no restart")
        started, phase = time.perf_counter(), "public_root"
        try:
            state = context.public_materialization_state
            if (getattr(state, "player_id", None) != context.player_id
                    or context.format_id != "gen3randombattle"
                    or getattr(state, "format_id", None) != context.format_id
                    or getattr(state, "observation_format_id", None) != context.format_id
                    or getattr(getattr(state, "replay", None), "battle_id", None) != context.battle_id):
                raise ReferenceRefusal("reference actor/format materialization mismatch")
            # Capture is the sole transport boundary. Neither the full context,
            # trajectory nor simultaneous opponent information reaches a worker.
            request = PublicRootRequest.capture(state, context.observation)
            if request.set_source_hash != self.factory.set_source_hash:
                raise ReferenceRefusal("reference public root catalog binding drift")
            root = decision_state(request.observation, player=context.player_id)
            if not root.actions:
                raise ReferenceRefusal("reference root has no legal action")
            if len(root.actions) == 1:
                metadata = {"status": "FORCED", "new_trajectories": 0,
                            "root_visits": [0], "raw_fallback": False}
            else:
                if self._pool is None:
                    phase = "pool_startup"
                    startup_at = time.perf_counter()
                    try:
                        self._pool = ParallelTrajectorySearch(
                            ReferenceConfig(self.config.alpha, self.config.beta,
                                            self.config.max_transitions), self.factory,
                            workers=self.config.workers, batch_size=10)
                    finally:
                        startup = time.perf_counter() - startup_at
                        self.stats.pool_startup_seconds += startup
                        # Public capture still counts against this decision.
                        started += startup
                phase = "search"
                remaining = self.config.decision_seconds - (time.perf_counter() - started)
                if remaining <= 0:
                    raise ReferenceRefusal("public preparation exhausted reference ceiling")
                parallel = self._pool.search(request, root, battle_id=context.battle_id,
                    seed=rng.getrandbits(64), deadline_seconds=remaining)
                result = parallel.result
                if result.trajectories <= 0 or result.action not in root.actions:
                    raise ReferenceRefusal("reference returned zero work or a different action")
                if (len(result.root_visits) != len(root.actions)
                        or any(type(n) is not int or n < 0 for n in result.root_visits)
                        or result.action != root.actions[max(range(len(root.actions)),
                            key=lambda i: (result.root_visits[i], -i))]):
                    raise ReferenceRefusal("reference selector is not cumulative visit-max")
                forward_counts = [receipt["evidence"]["neural_forwards"]
                    for receipt in parallel.worker_receipts]
                if any(type(n) is not int or n < 0 for n in forward_counts):
                    raise ReferenceRefusal("reference forward counts are invalid")
                forwards = sum(forward_counts)
                elapsed = time.perf_counter() - started
                metadata = {"status": "SEARCHED", "new_trajectories": result.trajectories,
                    "transitions": result.transitions, "world_draws": result.world_draws,
                    "neural_forwards": forwards, "root_visits": list(result.root_visits),
                    "root_means": list(result.root_means), "decision_wall_seconds": elapsed,
                    "search_wall_seconds": result.elapsed_seconds,
                    "deadline_exhausted": result.deadline_exhausted,
                    "deadline_overrun_seconds": max(0., elapsed - self.config.decision_seconds),
                    "pool_startup_seconds": self.stats.pool_startup_seconds,
                    "worker_pids": list(parallel.worker_pids),
                    "exchange_count": parallel.exchange_count,
                    "decision_id": parallel.decision_id, "raw_fallback": False}
            action = root.actions[0] if len(root.actions) == 1 else result.action
            try:
                action_index = int(action.removeprefix("action:"))
            except (ValueError, AttributeError) as error:
                raise ReferenceRefusal("reference action is not a bridge action") from error
            if (action != f"action:{action_index}" or not 0 <= action_index
                    < len(context.observation.legal_action_mask)
                    or not context.observation.legal_action_mask[action_index]):
                raise ReferenceRefusal("reference action is illegal at the original bridge root")
            metadata.update(battle_id=context.battle_id, root_sha256=root.key.hex(),
                set_source_hash=self.factory.set_source_hash,
                checkpoint_sha256=self.factory.checkpoint_sha256,
                atomic_work_instrumented=False, scientifically_qualified=False)
            if len(root.actions) == 1:
                self.stats.forced_decisions += 1
                metadata["decision_wall_seconds"] = time.perf_counter() - started
            else:
                self.stats.searches += 1
                self.stats.trajectories += result.trajectories
                self.stats.transitions += result.transitions
                self.stats.world_draws += result.world_draws
                self.stats.neural_forwards += forwards
                self.stats.deadline_overrun_seconds += metadata["deadline_overrun_seconds"]
            self.stats.last_decision = dict(metadata)
            self.stats.decision_wall_seconds += time.perf_counter() - started
            return PolicyDecision(action_index, self.policy_id,
                                  metadata={"trajectory_reference": metadata})
        except BaseException as error:
            self.stats.decision_wall_seconds += time.perf_counter() - started
            self.record_failure(phase, error)
            raise

    def record_failure(self, phase: str, error: BaseException) -> None:
        if not self._failed:
            self.stats.refusals += 1
            # Arbitrary exception strings/worker traces can contain private
            # input. Retain structured status, not the full error transport.
            self.stats.last_decision = {"status": "REFUSED", "phase": phase,
                "error_type": type(error).__name__, "raw_fallback": False}
            # Constructor failures carry their own evidence: self._pool has
            # not been assigned yet. Do not drop those worker startup sites.
            if isinstance(error, ParallelRefusal) and isinstance(error.evidence, dict):
                worker_failures = sanitized_worker_failures(error.evidence.get("errors", ()))
                if worker_failures:
                    self.stats.last_decision["worker_failures"] = worker_failures
            if (phase == "search" and self._pool is not None
                    and self._pool.last_evidence.get("status") != "COMPLETE"):
                receipts = self._pool.last_evidence.get("receipts", ())
                self.stats.last_decision.update(
                    observed_partial_trajectories=sum(r["batch"].trajectories for r in receipts),
                    observed_partial_transitions=sum(r["batch"].transitions for r in receipts),
                    observed_partial_world_draws=sum(r["batch"].world_draws for r in receipts),
                    refused_work_accounting_complete=False)
        self._failed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            self._pool.close()
