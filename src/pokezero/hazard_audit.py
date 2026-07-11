"""Public-only, reproducible Step 3 hazard blind-spot audit helpers.

The corpus deliberately stores only an acting player's observations, legal mask,
and completed public actions.  In particular it never retains an opponent
request, opponent observation, or opponent legal-action mask.  A synthetic
``BattleTrajectory`` is reconstructed solely to reuse the existing replay/PUCT
implementation; non-actor steps contain an all-legal placeholder, never a
captured opponent mask.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

from .actions import ACTION_COUNT
from .determinization import (
    BeliefWorldSamplingProfile,
    belief_world_sampling_profile,
    gen3_randbat_belief_start_override_planner,
)
from .env import BattleStartOverride, PlayerId, PokeZeroEnv
from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
from .observation import PokeZeroObservationV0
from .policy import MaxDamagePolicy, Policy, RandomLegalPolicy, SimpleLegalPolicy
from .policy import PolicyContext
from .prior_belief_profile import public_policy_context
from .public_decision_corpus import (
    PUBLIC_DECISION_CORPUS_SCHEMA_VERSION,
    PublicDecisionCorpus,
    PublicDecisionRecord,
    PublicResolvedActionRound,
    load_public_decision_corpus,
    public_decision_records_from_trajectory,
)
from .public_replay_materializer import PublicReplayError, replay_public_action_rounds
from .randbat import Gen3RandbatSource
from .rollout import RolloutConfig, RolloutDriver
from .search import ActionPriorVector, ObservationValueFunction, puct_branch_search
from .search_policy import OpponentActionScenario, _root_dirichlet_action_priors
from .trajectory import BattleTrajectory, TrajectoryStep


HAZARD_AUDIT_SCHEMA_VERSION = "pokezero.hazard-blind-spot-audit.v1"
DEFAULT_EXTRA_VISITS = (0, 24, 120)
DEFAULT_LOW_PRIOR_THRESHOLD = 0.01
DEFAULT_DIRICHLET_ALPHA = 0.3
DEFAULT_DIRICHLET_MIX = 0.25
DEFAULT_DIRICHLET_SEED = 20260710
_PRIVATE_PAYLOAD_KEYS = frozenset(
    {
        "request",
        "opponent_request",
        "requested_observations",
        "requested_legal_action_masks",
        "opponent_legal_action_mask",
        "opponent_legal_mask",
        "opponent_actions",
        "opponent_action_index",
        "opponent_observation",
        "true_opponent_observation",
        "requested_opponent_observation",
        "start_override",
        "start_overrides",
        "true_opponent_request",
        "true_opponent_legal_mask",
    }
)

ActionPriorFunction = Callable[[tuple[PokeZeroObservationV0, ...]], ActionPriorVector]
WorldProvider = Callable[["HazardAuditDecision"], Sequence["AuditWorld"]]


@dataclass(frozen=True)
class PublicActorObservation:
    """One retained observation from the acting player's own information set."""

    turn_index: int
    observation: PokeZeroObservationV0

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")


@dataclass(frozen=True)
class HazardAuditDecision:
    """A Step 3 target derived from one canonical Step 2 public record."""

    public_record: PublicDecisionRecord
    driver_id: str
    target_action_index: int
    target_move_id: str

    def __post_init__(self) -> None:
        if self.target_move_id not in {"spikes", "rapidspin"}:
            raise ValueError("target_move_id must be 'spikes' or 'rapidspin'.")
        if not 0 <= self.target_action_index < ACTION_COUNT:
            raise ValueError(f"target_action_index must be in 0..{ACTION_COUNT - 1}.")
        if not self.observation.legal_action_mask[self.target_action_index]:
            raise ValueError("the hazard target action must be legal in the actor observation.")
        _assert_public_payload(self.public_payload())

    @property
    def battle_id(self) -> str:
        return self.public_record.battle_id

    @property
    def format_id(self) -> str:
        return self.public_record.format_id

    @property
    def seed(self) -> int:
        return self.public_record.seed

    @property
    def player_id(self) -> PlayerId:
        return self.public_record.acting_player

    @property
    def decision_round(self) -> int:
        return self.public_record.turn_index

    @property
    def observation(self) -> PokeZeroObservationV0:
        return self.public_record.observation.to_observation(belief_view=self.public_record.public_belief_view)

    @property
    def observation_history(self) -> tuple[PublicActorObservation, ...]:
        return tuple(
            PublicActorObservation(
                turn_index=entry.turn_index,
                observation=entry.observation.to_observation(belief_view=self.public_record.public_belief_view),
            )
            for entry in (*self.public_record.history,)
        ) + (PublicActorObservation(turn_index=self.decision_round, observation=self.observation),)

    @property
    def public_action_rounds(self) -> tuple[PublicResolvedActionRound, ...]:
        return self.public_record.public_resolved_action_rounds

    @property
    def actor_history(self) -> tuple[PokeZeroObservationV0, ...]:
        return tuple(entry.observation for entry in self.observation_history)

    @property
    def state_id(self) -> str:
        return canonical_hash(self.public_payload())[:20]

    def public_payload(self) -> dict[str, object]:
        """Serializable corpus payload, excluding every opponent-private input."""

        return {
            "public_decision": self.public_record.to_dict(),
            "driver_id": self.driver_id,
            "target": {
                "action_index": self.target_action_index,
                "move_id": self.target_move_id,
            },
        }

    def to_public_decision_record(self) -> dict[str, object]:
        """Return the exact canonical Step 2 decision representation."""

        return self.public_record.to_dict()

    def to_dict(self) -> dict[str, object]:
        return {
            "state_id": self.state_id,
            "public_state_hash": canonical_hash(self.public_payload()),
            **self.public_payload(),
        }

    def to_search_trajectory(self, *, replay_actions: Mapping[int, Mapping[PlayerId, int]]) -> BattleTrajectory:
        """Build an ephemeral raw replay only after sampled-world ID resolution."""

        history_by_turn = {entry.turn_index: entry.observation for entry in self.observation_history}
        trajectory = BattleTrajectory(
            battle_id=self.battle_id,
            format_id=self.format_id,
            seed=self.seed,
            metadata={"hazard_audit_public_only": True, "driver_id": self.driver_id},
        )
        for round_ in self.public_action_rounds:
            actions = replay_actions.get(round_.turn_index)
            if actions is None or set(actions) != set(round_.actions):
                raise ValueError("sampled replay actions do not match the public prefix.")
            for player_id, action_index in sorted(actions.items()):
                observation = (
                    history_by_turn[round_.turn_index]
                    if player_id == self.player_id and round_.turn_index in history_by_turn
                    else _public_replay_placeholder_observation()
                )
                trajectory.append(
                    TrajectoryStep(
                        player_id=player_id,
                        turn_index=round_.turn_index,
                        observation=observation,
                        legal_action_mask=tuple(observation.legal_action_mask),
                        action_index=action_index,
                        metadata={
                            "hazard_audit_public_actor_observation": player_id == self.player_id,
                            "hazard_audit_public_replay_placeholder": player_id != self.player_id,
                        },
                    )
                )
        # The current actor observation makes the existing value/PUCT history API
        # see the branch point. Its action is not replayed because it lies at N.
        trajectory.append(
            TrajectoryStep(
                player_id=self.player_id,
                turn_index=self.decision_round,
                observation=self.observation,
                legal_action_mask=tuple(self.observation.legal_action_mask),
                action_index=self.target_action_index,
                metadata={"hazard_audit_branch_point": True},
            )
        )
        return trajectory

    def policy_context(self) -> PolicyContext:
        return public_policy_context(self.public_record)


@dataclass(frozen=True)
class AuditConfig:
    """Pinned Step 3 search configuration; no capstone configuration is changed."""

    cpuct: float = 1.25
    extra_visits: tuple[int, ...] = DEFAULT_EXTRA_VISITS
    low_prior_threshold: float = DEFAULT_LOW_PRIOR_THRESHOLD
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
    dirichlet_mix: float = DEFAULT_DIRICHLET_MIX
    dirichlet_seed: int = DEFAULT_DIRICHLET_SEED

    def __post_init__(self) -> None:
        if self.cpuct < 0:
            raise ValueError("cpuct must be non-negative.")
        if tuple(self.extra_visits) != DEFAULT_EXTRA_VISITS:
            raise ValueError("Step 3 requires the fixed extra-visit sweep (0, 24, 120).")
        if not 0.0 <= self.low_prior_threshold <= 1.0:
            raise ValueError("low_prior_threshold must be in [0, 1].")
        if self.dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be positive.")
        if not 0.0 < self.dirichlet_mix <= 1.0:
            raise ValueError("dirichlet_mix must be in (0, 1].")

    def to_dict(self) -> dict[str, object]:
        return {
            "cpuct": self.cpuct,
            "extra_visits": list(self.extra_visits),
            "low_prior_threshold": self.low_prior_threshold,
            "dirichlet": {
                "enabled": True,
                "audit_only": True,
                "alpha": self.dirichlet_alpha,
                "mix": self.dirichlet_mix,
                "base_seed": self.dirichlet_seed,
            },
            "mandatory_legal_sweep": True,
            "entrenchment": "target receives no visits beyond its mandatory one per valid world",
        }


@dataclass(frozen=True)
class AuditWorld:
    """One determinized public-belief world and its inferred opponent actions."""

    world_id: str
    opponent_actions: Mapping[PlayerId, int]
    start_override: BattleStartOverride | None = None
    replay_actions: Mapping[int, Mapping[PlayerId, int]] = field(default_factory=dict)
    metadata: Mapping[str, object] | None = None
    available: bool = True
    rejection_code: str | None = None

    def __post_init__(self) -> None:
        normalized = {str(player): int(action) for player, action in sorted(self.opponent_actions.items())}
        if any(action < 0 or action >= ACTION_COUNT for action in normalized.values()):
            raise ValueError(f"opponent action indices must be in 0..{ACTION_COUNT - 1}.")
        object.__setattr__(self, "opponent_actions", normalized)
        normalized_replay_actions = {
            int(turn_index): {
                str(player): int(action)
                for player, action in sorted(actions.items())
            }
            for turn_index, actions in sorted(self.replay_actions.items())
        }
        if any(
            action < 0 or action >= ACTION_COUNT
            for actions in normalized_replay_actions.values()
            for action in actions.values()
        ):
            raise ValueError(f"sampled replay action indices must be in 0..{ACTION_COUNT - 1}.")
        object.__setattr__(self, "replay_actions", normalized_replay_actions)
        _assert_public_payload(dict(self.metadata or {}))

    def to_dict(self) -> dict[str, object]:
        return {
            "world_id": self.world_id,
            "available": self.available,
            "rejection_code": self.rejection_code,
            "metadata": dict(self.metadata or {}),
            "opponent_action_source": "public-belief sampled world",
        }


@dataclass
class PublicBeliefWorldProvider:
    """Materialize PIMC worlds without reading the true opponent request or mask.

    Opponent action validity is checked only on the sampled public-belief world.
    The true battle's opponent observation/mask never enters this object.
    """

    env_factory: Callable[[], PokeZeroEnv]
    set_source: Gen3RandbatSource
    sampled_world_opponent_policy: Policy
    world_sample_cap: int = 4

    def __call__(self, decision: HazardAuditDecision) -> tuple[AuditWorld, ...]:
        context = decision.policy_context()
        profile = belief_world_sampling_profile(
            context,
            sample_cap=self.world_sample_cap,
            set_source=self.set_source,
        )
        if profile is None:
            return (
                AuditWorld(
                    world_id=f"{decision.state_id}-missing-public-belief",
                    opponent_actions={},
                    available=False,
                    rejection_code="missing_public_belief",
                ),
            )
        planner = gen3_randbat_belief_start_override_planner(
            self.set_source,
            world_sample_cap=self.world_sample_cap,
        )
        worlds: list[AuditWorld] = []
        for world_index in range(profile.sample_count):
            world_id = f"{decision.state_id}-world-{world_index}"
            rng = random.Random(_stable_seed(decision.state_id, "belief-world", world_index))
            env: PokeZeroEnv | None = None
            try:
                source = planner(
                    context,
                    OpponentActionScenario(actions={}, weight=1.0, label="public-audit"),
                    world_index,
                    rng,
                )
                if source is None:
                    raise ValueError("missing sampled world")
                start_override = source() if callable(source) else source
                env = self.env_factory()
                try:
                    materialization = replay_public_action_rounds(
                        env,
                        seed=decision.seed,
                        format_id=decision.format_id,
                        public_action_rounds=decision.public_action_rounds,
                        start_override=start_override,
                    )
                    if decision.player_id not in materialization.requested_players:
                        raise ValueError("sampled world does not request the actor")
                    opponent_actions = _sampled_world_opponent_actions(
                        env=env,
                        actor=decision.player_id,
                        requested_players=materialization.requested_players,
                        policy=self.sampled_world_opponent_policy,
                        seed=_stable_seed(decision.state_id, "sampled-world-opponent-policy", world_index),
                    )
                finally:
                    if env is not None:
                        _close_env(env)
            except PublicReplayError as exc:
                worlds.append(
                    AuditWorld(
                        world_id=world_id,
                        opponent_actions={},
                        available=False,
                        rejection_code=exc.reason,
                        metadata=_public_profile_metadata(
                            profile,
                            world_index,
                            opponent_policy_id=self.sampled_world_opponent_policy.policy_id,
                        ),
                    )
                )
                continue
            except (ValueError, RuntimeError):
                worlds.append(
                    AuditWorld(
                        world_id=world_id,
                        opponent_actions={},
                        available=False,
                        rejection_code="public_world_replay_rejected",
                        metadata=_public_profile_metadata(
                            profile,
                            world_index,
                            opponent_policy_id=self.sampled_world_opponent_policy.policy_id,
                        ),
                    )
                )
                continue
            worlds.append(
                AuditWorld(
                    world_id=world_id,
                    opponent_actions=opponent_actions,
                    start_override=start_override,
                    replay_actions=materialization.replay_actions,
                    metadata=_public_profile_metadata(
                        profile,
                        world_index,
                        opponent_policy_id=self.sampled_world_opponent_policy.policy_id,
                        event_canonicalizations=materialization.event_canonicalizations,
                    ),
                )
            )
        return tuple(worlds)


def capture_hazard_audit_corpus(
    *,
    env_config: LocalShowdownConfig,
    games: int,
    seed_start: int = 1,
    max_states: int = 800,
    max_decision_rounds: int = 250,
) -> tuple[HazardAuditDecision, ...]:
    """Capture checkpoint-independent driver states and discard private rollout data."""

    if games <= 0 or max_states <= 0:
        raise ValueError("games and max_states must be positive.")
    drivers = _hazard_audit_driver_pairs(env_config.resolved_showdown_root())
    collected: list[HazardAuditDecision] = []
    for offset in range(games):
        if len(collected) >= max_states:
            break
        seed = seed_start + offset
        driver_id, policies = drivers[offset % len(drivers)]
        with LocalShowdownEnv(env_config) as env:
            result = RolloutDriver(
                env=env,
                policies=policies,
                config=RolloutConfig(max_decision_rounds=max_decision_rounds),
            ).run(seed=seed, battle_id=f"hazard-audit-{driver_id}-{seed}")
        collected.extend(
            hazard_audit_decisions_from_trajectory(result.trajectory, driver_id=driver_id)
        )
    return tuple(sorted(collected[:max_states], key=lambda decision: decision.state_id))


def hazard_audit_decisions_from_trajectory(
    trajectory: BattleTrajectory,
    *,
    driver_id: str,
) -> tuple[HazardAuditDecision, ...]:
    """Capture Step 3 targets from canonical p1 public corpus records only."""

    return _hazard_decisions_from_public_records(
        public_decision_records_from_trajectory(trajectory, acting_player="p1"),
        driver_id=driver_id,
    )


def hazard_audit_decisions_from_public_corpus(
    corpus: PublicDecisionCorpus | Path,
) -> tuple[HazardAuditDecision, ...]:
    """Adapt the canonical Step 2 JSONL corpus without a second schema reader."""

    loaded = load_public_decision_corpus(corpus) if isinstance(corpus, Path) else corpus
    if not isinstance(loaded, PublicDecisionCorpus):
        raise ValueError("hazard audit requires a canonical PublicDecisionCorpus or its JSONL path.")
    if loaded.manifest.get("schema_version") != PUBLIC_DECISION_CORPUS_SCHEMA_VERSION:
        raise ValueError(f"hazard audit requires {PUBLIC_DECISION_CORPUS_SCHEMA_VERSION!r}.")
    return _hazard_decisions_from_public_records(loaded.decisions, driver_id="public-decision-corpus")


def _hazard_decisions_from_public_records(
    records: Sequence[PublicDecisionRecord],
    *,
    driver_id: str,
) -> tuple[HazardAuditDecision, ...]:
    decisions: list[HazardAuditDecision] = []
    for record in records:
        observation = record.observation.to_observation(belief_view=record.public_belief_view)
        for target_action_index, target_move_id in _hazard_target_moves(
            observation,
            record.current_legal_action_mask,
        ):
            decisions.append(
                HazardAuditDecision(
                    public_record=record,
                    driver_id=driver_id,
                    target_action_index=target_action_index,
                    target_move_id=target_move_id,
                )
            )
    return tuple(sorted(decisions, key=lambda decision: decision.state_id))


def run_hazard_blind_spot_audit(
    *,
    decisions: Iterable[HazardAuditDecision],
    env_factory: Callable[[], PokeZeroEnv],
    action_priors: ActionPriorFunction,
    value_fn: ObservationValueFunction,
    world_provider: WorldProvider,
    config: AuditConfig = AuditConfig(),
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run deterministic/noise PUCT sweeps and return a JSON-ready audit payload."""

    decision_list = tuple(sorted(decisions, key=lambda decision: decision.state_id))
    config_payload = config.to_dict()
    records: list[dict[str, object]] = []
    corpus_states = [decision.to_dict() for decision in decision_list]
    low_prior_state_ids: set[str] = set()
    low_prior_available_world_state_ids: set[str] = set()
    low_prior_available_world_pairs: set[tuple[str, str]] = set()
    for decision in decision_list:
        priors = _validated_priors(action_priors(decision.actor_history))
        target_prior = _normalized_target_prior(
            priors,
            legal_mask=tuple(decision.observation.legal_action_mask),
            target_action_index=decision.target_action_index,
        )
        low_prior = target_prior <= config.low_prior_threshold
        if low_prior:
            low_prior_state_ids.add(decision.state_id)
        worlds = tuple(sorted(world_provider(decision), key=lambda world: world.world_id))
        for world in worlds:
            if low_prior and world.available:
                low_prior_available_world_state_ids.add(decision.state_id)
                low_prior_available_world_pairs.add((decision.state_id, world.world_id))
            for arm in ("deterministic", "dirichlet_audit_only"):
                search_priors, noise_metadata = _arm_priors(
                    arm=arm,
                    decision=decision,
                    priors=priors,
                    config=config,
                )
                for extra_visits in config.extra_visits:
                    records.append(
                        _run_record(
                            decision=decision,
                            world=world,
                            arm=arm,
                            extra_visits=extra_visits,
                            target_prior=target_prior,
                            low_prior=low_prior,
                            action_priors=search_priors,
                            noise_metadata=noise_metadata,
                            env_factory=env_factory,
                            value_fn=value_fn,
                            config=config,
                        )
                    )
    records.sort(key=lambda record: (str(record["state_id"]), str(record["world_id"]), str(record["arm"]), int(record["extra_visits"])))
    aggregate = aggregate_hazard_audit_records(
        records,
        eligibility_funnel={
            "hazard_legal_target_states": len({decision.state_id for decision in decision_list}),
            "low_prior_target_states": len(low_prior_state_ids),
            "low_prior_target_states_with_available_belief_worlds": len(low_prior_available_world_state_ids),
            "low_prior_state_world_pairs_with_available_belief_worlds": len(low_prior_available_world_pairs),
        },
    )
    corpus_hash = canonical_hash(corpus_states)
    provenance_payload = dict(provenance or {})
    _assert_public_payload(provenance_payload)
    payload = {
        "schema_version": HAZARD_AUDIT_SCHEMA_VERSION,
        "provenance": provenance_payload,
        "config": config_payload,
        "hashes": {
            "config_hash": canonical_hash(config_payload),
            "corpus_hash": corpus_hash,
            "records_hash": canonical_hash(records),
            "provenance_hash": canonical_hash(provenance_payload),
        },
        "corpus": {
            "state_count": len(decision_list),
            "states": corpus_states,
        },
        "records": records,
        "aggregate": aggregate,
    }
    # AuditWorld holds the sampled override and action map in memory, but the
    # persisted artifact must never carry either private simulation input.
    _assert_public_payload(payload)
    return payload


def aggregate_hazard_audit_records(
    records: Iterable[Mapping[str, Any]],
    *,
    eligibility_funnel: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Aggregate the Step 3 metrics with the mandatory-sweep semantics made explicit."""

    rows = tuple(records)
    state_low_prior: dict[str, bool] = {}
    for row in rows:
        state_id = str(row["state_id"])
        low_prior = bool(row["low_prior"])
        if state_id in state_low_prior and state_low_prior[state_id] != low_prior:
            raise ValueError("a state has inconsistent low-prior classification across records.")
        state_low_prior[state_id] = low_prior
    low_prior_count = sum(state_low_prior.values())
    total_states = len(state_low_prior)

    paired_search_by_budget: dict[str, dict[str, int]] = {}
    for budget in DEFAULT_EXTRA_VISITS:
        paired_rows: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in rows:
            if (
                int(row.get("extra_visits", -1)) == budget
                and row.get("status") == "searched"
                and bool(row.get("low_prior"))
                and str(row.get("arm")) in {"deterministic", "dirichlet_audit_only"}
            ):
                paired_rows[(str(row["state_id"]), str(row["world_id"]))].add(str(row["arm"]))
        complete_pairs = {
            pair for pair, arms in paired_rows.items() if arms == {"deterministic", "dirichlet_audit_only"}
        }
        paired_search_by_budget[str(budget)] = {
            "target_states": len({state_id for state_id, _world_id in complete_pairs}),
            "state_world_pairs": len(complete_pairs),
        }

    derived_available_pairs = {
        (str(row["state_id"]), str(row["world_id"]))
        for row in rows
        if bool(row.get("low_prior"))
        and (
            not isinstance(row.get("world"), Mapping)
            or bool(row["world"].get("available", False))
        )
    }
    funnel = {
        "hazard_legal_target_states": total_states,
        "low_prior_target_states": low_prior_count,
        "low_prior_target_states_with_available_belief_worlds": len(
            {state_id for state_id, _world_id in derived_available_pairs}
        ),
        "low_prior_state_world_pairs_with_available_belief_worlds": len(derived_available_pairs),
    }
    if eligibility_funnel is not None:
        funnel.update({key: int(value) for key, value in eligibility_funnel.items()})
    funnel.update(
        {
            "paired_searched_target_states_by_extra_visits": {
                budget: values["target_states"] for budget, values in paired_search_by_budget.items()
            },
            "paired_searched_state_world_pairs_by_extra_visits": {
                budget: values["state_world_pairs"] for budget, values in paired_search_by_budget.items()
            },
            "interpretation": (
                "Counts are a stage-by-stage eligibility funnel: legal hazard/spin targets, low-prior "
                "targets, sampled public-belief worlds, then complete deterministic/Dirichlet search pairs."
            ),
        }
    )
    eligibility_low_prior_count = int(funnel["low_prior_target_states"])
    eligibility_total_states = int(funnel["hazard_legal_target_states"])

    off_rescue: dict[str, dict[str, object]] = {}
    for budget in (24, 120):
        per_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            if (
                row.get("arm") == "deterministic"
                and int(row.get("extra_visits", -1)) == budget
                and row.get("status") == "searched"
                and bool(row.get("low_prior"))
            ):
                per_state[str(row["state_id"])].append(row)
        rescued = sum(
            any(int(row["target_revisits"]) > 0 for row in state_rows)
            for state_rows in per_state.values()
        )
        denominator = len(per_state)
        off_rescue[str(budget)] = {
            "rescued_low_prior_lines": rescued,
            "eligible_low_prior_lines": denominator,
            "rate": _rate(rescued, denominator),
            "definition": "a legal low-prior target is rescued iff any valid public-belief world re-visits it beyond the mandatory sweep",
        }

    choice_delta: dict[str, dict[str, object]] = {}
    for budget in DEFAULT_EXTRA_VISITS:
        paired: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in rows:
            if int(row.get("extra_visits", -1)) != budget or row.get("status") != "searched":
                continue
            if not bool(row.get("low_prior")):
                continue
            arm = str(row.get("arm"))
            if arm not in {"deterministic", "dirichlet_audit_only"}:
                continue
            paired[(str(row["state_id"]), str(row["world_id"]))][arm] = row
        paired_by_state: dict[str, list[Mapping[str, Mapping[str, Any]]]] = defaultdict(list)
        for (state_id, _world_id), arms in paired.items():
            if set(arms) == {"deterministic", "dirichlet_audit_only"}:
                paired_by_state[state_id].append(arms)
        state_choices = {
            state_id: {
                "deterministic": _strict_majority_choice(
                    bool(arms["deterministic"]["target_selected"])
                    for arms in world_pairs
                ),
                "dirichlet_audit_only": _strict_majority_choice(
                    bool(arms["dirichlet_audit_only"]["target_selected"])
                    for arms in world_pairs
                ),
            }
            for state_id, world_pairs in paired_by_state.items()
        }
        complete = tuple(state_choices.values())
        off_rate = _rate(sum(choice["deterministic"] for choice in complete), len(complete))
        on_rate = _rate(sum(choice["dirichlet_audit_only"] for choice in complete), len(complete))
        toward_target = sum(
            not choice["deterministic"] and choice["dirichlet_audit_only"]
            for choice in complete
        )
        away_from_target = sum(
            choice["deterministic"] and not choice["dirichlet_audit_only"]
            for choice in complete
        )
        choice_delta[str(budget)] = {
            "paired_low_prior_target_states": len(complete),
            "paired_state_world_pairs": sum(len(world_pairs) for world_pairs in paired_by_state.values()),
            "toward_low_prior_target_states": toward_target,
            "away_from_low_prior_target_states": away_from_target,
            "choice_rate_off": off_rate,
            "choice_rate_on": on_rate,
            "delta_choice_on": _rate(toward_target - away_from_target, len(complete)),
            "world_aggregation": "strict-majority target selection over paired valid belief worlds; exact ties resolve false",
            "interpretation": (
                "noise_only_choice_sensitivity"
                if budget == 0
                else "search_choice_change_toward_low_prior_target"
            ),
            "definition": "per-state changes toward minus away from the low-prior hazard/spin target after strict-majority aggregation over paired belief worlds; this is a choice metric, not a rescue metric",
        }

    searched = [row for row in rows if row.get("status") == "searched"]
    status_counts = Counter(str(row.get("status") or "missing_status") for row in rows)
    invalid_reason_counts = Counter(
        str(row.get("invalid_reason") or "missing_invalid_reason")
        for row in rows
        if row.get("status") == "search_invalid"
    )
    world_unavailable_records = status_counts["world_unavailable"]
    rejected_records = status_counts["search_rejected"] + status_counts["target_branch_unavailable"]
    invalid_records = status_counts["search_invalid"]
    other_non_search_records = len(rows) - len(searched) - world_unavailable_records - rejected_records - invalid_records
    return {
        "definitions": {
            "E": "share of unique legal hazard/spin target lines with normalized legal prior at or below low_prior_threshold",
            "R_off": "deterministic rescue rate; a rescue is at least one re-visit, not merely the mandatory initial visit",
            "DeltaChoice_on": "per-state changes toward minus away from the low-prior hazard/spin target after strict-majority aggregation over paired belief worlds; budget 0 is noise-only sensitivity, never rescue",
            "entrenchment": "target_revisits == 0. Each legal target receives one mandatory initial-sweep visit per valid world, so entrenchment is never defined as target_visits == 0.",
        },
        "E": {
            "low_prior_lines": eligibility_low_prior_count,
            "legal_target_lines": eligibility_total_states,
            "rate": _rate(eligibility_low_prior_count, eligibility_total_states),
        },
        "R_off": off_rescue,
        "DeltaChoice_on": choice_delta,
        "eligibility_funnel": funnel,
        "coverage": {
            "records": len(rows),
            "searched_records": len(searched),
            "status_counts": dict(sorted(status_counts.items())),
            "world_unavailable_records": world_unavailable_records,
            "rejected_records": rejected_records,
            "invalid_records": invalid_records,
            "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
            "other_non_search_records": other_non_search_records,
            "unique_target_lines": total_states,
        },
    }


def canonical_hash(payload: object) -> str:
    """Stable SHA-256 for corpus/provenance/config/record payloads."""

    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_record(
    *,
    decision: HazardAuditDecision,
    world: AuditWorld,
    arm: str,
    extra_visits: int,
    target_prior: float,
    low_prior: bool,
    action_priors: ActionPriorVector,
    noise_metadata: Mapping[str, object],
    env_factory: Callable[[], PokeZeroEnv],
    value_fn: ObservationValueFunction,
    config: AuditConfig,
) -> dict[str, object]:
    configured_legal_action_count = sum(bool(value) for value in decision.observation.legal_action_mask)
    root_visit_budget = configured_legal_action_count + extra_visits
    base = {
        "state_id": decision.state_id,
        "public_state_hash": canonical_hash(decision.public_payload()),
        "driver_id": decision.driver_id,
        "seed": decision.seed,
        "actor": decision.player_id,
        "decision_round": decision.decision_round,
        "target_move_id": decision.target_move_id,
        "target_action_index": decision.target_action_index,
        "normalized_legal_target_prior": target_prior,
        "low_prior": low_prior,
        "world_id": world.world_id,
        "world": world.to_dict(),
        "arm": arm,
        "dirichlet_audit_only": arm == "dirichlet_audit_only",
        "extra_visits": extra_visits,
        "configured_legal_action_count": configured_legal_action_count,
        "requested_root_visit_budget": root_visit_budget,
        "mandatory_sweep_candidate_count": None,
        "expected_total_visits": None,
        "noise": dict(noise_metadata),
    }
    if not world.available:
        return {**base, "status": "world_unavailable", "target_visits": None, "target_revisits": None, "entrenched": None, "target_selected": None}
    env = env_factory()
    try:
        result = puct_branch_search(
            env=env,
            trajectory=decision.to_search_trajectory(replay_actions=world.replay_actions),
            player_id=decision.player_id,
            prefix_decision_round_count=decision.decision_round,
            legal_action_mask=tuple(bool(value) for value in decision.observation.legal_action_mask),
            opponent_actions=world.opponent_actions,
            value_fn=value_fn,
            action_priors=action_priors,
            cpuct=config.cpuct,
            root_visit_budget=root_visit_budget,
            start_override=world.start_override,
            expected_current_observation=decision.observation if world.start_override is not None else None,
        )
    except ValueError:
        return {**base, "status": "search_rejected", "target_visits": None, "target_revisits": None, "entrenched": None, "target_selected": None}
    finally:
        _close_env(env)
    target = next((candidate for candidate in result.candidates if candidate.action_index == decision.target_action_index), None)
    candidate_count = len(result.candidates)
    expected_total_visits = candidate_count + extra_visits
    visit_accounting = {
        "mandatory_sweep_candidate_count": candidate_count,
        "expected_total_visits": expected_total_visits,
        "total_visits": result.total_visits,
    }
    if result.total_visits != expected_total_visits:
        return {
            **base,
            **visit_accounting,
            "status": "search_invalid",
            "invalid_reason": "mandatory_sweep_visit_mismatch",
            "target_visits": None,
            "target_revisits": None,
            "entrenched": None,
            "target_selected": None,
        }
    if target is None:
        return {
            **base,
            **visit_accounting,
            "status": "target_branch_unavailable",
            "target_visits": None,
            "target_revisits": None,
            "entrenched": None,
            "target_selected": None,
        }
    candidate_visits = {str(candidate.action_index): candidate.visits for candidate in result.candidates}
    revisits = max(0, target.visits - 1)
    return {
        **base,
        **visit_accounting,
        "status": "searched",
        "selected_action_index": result.action_index,
        "target_selected": result.action_index == decision.target_action_index,
        "target_visits": target.visits,
        "target_revisits": revisits,
        "entrenched": revisits == 0,
        "candidate_visits": candidate_visits,
        "search_result_hash": canonical_hash(
            {
                "selected_action_index": result.action_index,
                "total_visits": result.total_visits,
                "candidate_visits": candidate_visits,
            }
        ),
    }


def _arm_priors(
    *,
    arm: str,
    decision: HazardAuditDecision,
    priors: ActionPriorVector,
    config: AuditConfig,
) -> tuple[ActionPriorVector, Mapping[str, object]]:
    if arm == "deterministic":
        return priors, {"root_puct_root_dirichlet_enabled": False}
    if arm != "dirichlet_audit_only":
        raise ValueError(f"unsupported audit arm: {arm!r}.")
    return _root_dirichlet_action_priors(
        priors,
        context=decision.policy_context(),
        legal_action_mask=tuple(decision.observation.legal_action_mask),
        alpha=config.dirichlet_alpha,
        mix=config.dirichlet_mix,
        base_seed=config.dirichlet_seed,
    )


def _hazard_target_moves(
    observation: PokeZeroObservationV0,
    legal_mask: Sequence[bool],
) -> tuple[tuple[int, str], ...]:
    candidates = observation.metadata.get("action_candidates")
    if not isinstance(candidates, Sequence):
        return ()
    targets: list[tuple[int, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        action_index = candidate.get("action_index")
        move_id = _normalized_move_id(candidate.get("move_id"))
        if not isinstance(action_index, int) or move_id not in {"spikes", "rapidspin"}:
            continue
        if 0 <= action_index < len(legal_mask) and bool(legal_mask[action_index]):
            targets.append((action_index, move_id))
    return tuple(targets)


def _normalized_move_id(value: object) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def _hazard_audit_driver_pairs(showdown_root: Path) -> tuple[tuple[str, Mapping[PlayerId, Policy]], ...]:
    return (
        (
            "max-damage-vs-max-damage",
            {"p1": MaxDamagePolicy(showdown_root=showdown_root), "p2": MaxDamagePolicy(showdown_root=showdown_root)},
        ),
        (
            "max-damage-vs-random-legal",
            {"p1": MaxDamagePolicy(showdown_root=showdown_root), "p2": RandomLegalPolicy()},
        ),
        ("simple-legal-vs-simple-legal", {"p1": SimpleLegalPolicy(), "p2": SimpleLegalPolicy()}),
        ("random-legal-vs-random-legal", {"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()}),
    )


def _sampled_world_opponent_actions(
    *,
    env: PokeZeroEnv,
    actor: PlayerId,
    requested_players: Sequence[PlayerId],
    policy: Policy,
    seed: int,
) -> dict[PlayerId, int]:
    """Select opponents from the materialized world, never actor policy priors.

    The observation and its legal mask come from ``env`` only after replaying a
    public-belief start override. They are not serialized, and no observation
    from the source battle's opposing seat is available to this function.
    """

    actions: dict[PlayerId, int] = {}
    for player_id in requested_players:
        if player_id == actor:
            continue
        observation = env.observe(player_id)
        legal_mask = tuple(bool(value) for value in observation.legal_action_mask)
        if not any(legal_mask):
            raise ValueError("sampled world has no legal opponent action")
        policy_rng = random.Random(_stable_seed(seed, player_id))
        action_index = policy.select_action(observation, rng=policy_rng).action_index
        if not legal_mask[action_index]:
            raise ValueError("sampled-world opponent policy selected an illegal action")
        actions[player_id] = action_index
    return actions


def _public_profile_metadata(
    profile: BeliefWorldSamplingProfile,
    world_index: int,
    *,
    opponent_policy_id: str,
    event_canonicalizations: Sequence[Any] = (),
) -> dict[str, object]:
    return {
        "world_index": world_index,
        "belief_sampling": profile.to_metadata(),
        "world_source": "public-belief-determinization",
        "sampled_world_opponent_policy": opponent_policy_id,
        "public_event_canonicalizations": [
            canonicalization.to_dict() for canonicalization in event_canonicalizations
        ],
    }


def _validated_priors(values: Sequence[float]) -> ActionPriorVector:
    priors = tuple(float(value) for value in values)
    if len(priors) != ACTION_COUNT:
        raise ValueError(f"action priors must contain {ACTION_COUNT} values.")
    if any(value < 0.0 or value != value or value == float("inf") for value in priors):
        raise ValueError("action priors must be finite non-negative values.")
    return priors


def _normalized_target_prior(
    priors: ActionPriorVector,
    *,
    legal_mask: tuple[bool, ...],
    target_action_index: int,
) -> float:
    legal = tuple(index for index, is_legal in enumerate(legal_mask) if is_legal)
    total = sum(priors[index] for index in legal)
    if total <= 0.0:
        return 1.0 / len(legal)
    return priors[target_action_index] / total


def _public_replay_placeholder_observation() -> PokeZeroObservationV0:
    # This placeholder exists only because BattleTrajectory requires an observation
    # per replay action. All True is intentionally non-informative and cannot encode
    # the source opponent's actual legal mask.
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=(True,) * ACTION_COUNT,
        metadata={"hazard_audit_public_replay_placeholder": True},
        schema_version="pokezero.hazard-audit.placeholder.v1",
    )


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 8)


def _strict_majority_choice(choices: Iterable[bool]) -> bool:
    """Collapse paired belief worlds to a conservative per-state target choice."""

    values = tuple(bool(choice) for choice in choices)
    return sum(values) * 2 > len(values)


def _close_env(env: PokeZeroEnv) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _assert_public_payload(payload: object) -> None:
    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            forbidden = _PRIVATE_PAYLOAD_KEYS.intersection(str(key) for key in value)
            if forbidden:
                raise ValueError(
                    "hazard audit public payload contains forbidden private key(s): "
                    + ", ".join(sorted(forbidden))
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(payload)


def _jsonable(value: object) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(child) for child in sorted(value, key=repr)]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    if isinstance(value, Path):
        return str(value)
    return value
