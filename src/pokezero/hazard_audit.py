"""Public-only, reproducible Step 3 hazard blind-spot audit helpers.

The corpus deliberately stores only an acting player's observations, legal mask,
and completed public actions.  In particular it never retains an opponent
request, opponent observation, or opponent legal-action mask.  A synthetic
``BattleTrajectory`` is reconstructed solely to reuse the existing replay/PUCT
implementation; non-actor steps contain an all-legal placeholder, never a
captured opponent mask.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass, replace
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
from .observation import ObservationPerspective, PokeZeroObservationV0
from .policy import MaxDamagePolicy, Policy, RandomLegalPolicy, SimpleLegalPolicy
from .policy import PolicyContext
from .randbat import Gen3RandbatSource
from .replay_branching import replay_trajectory_prefix
from .rollout import RolloutConfig, RolloutDriver
from .search import ActionPriorVector, ObservationValueFunction, puct_branch_search
from .search_policy import OpponentActionScenario, _root_dirichlet_action_priors
from .trajectory import BattleTrajectory, TrajectoryStep


HAZARD_AUDIT_SCHEMA_VERSION = "pokezero.hazard-blind-spot-audit.v1"
PUBLIC_DECISION_CORPUS_SCHEMA_VERSION = "pokezero.public-decision-corpus.v1"
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
class PublicActionRound:
    """Completed public actions used to replay a decision prefix.

    An action becomes public after its round resolves.  This class intentionally
    carries no request or legality data for either player.
    """

    turn_index: int
    actions: Mapping[PlayerId, int]

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")
        if not self.actions:
            raise ValueError("public action rounds must contain at least one action.")
        normalized = {str(player): int(action) for player, action in sorted(self.actions.items())}
        if any(action < 0 or action >= ACTION_COUNT for action in normalized.values()):
            raise ValueError(f"public action indices must be in 0..{ACTION_COUNT - 1}.")
        object.__setattr__(self, "actions", normalized)

    def to_dict(self) -> dict[str, object]:
        return {"turn_index": self.turn_index, "actions": dict(self.actions)}


@dataclass(frozen=True)
class PublicActorObservation:
    """One retained observation from the acting player's own information set."""

    turn_index: int
    observation: PokeZeroObservationV0

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "observation": public_observation_to_dict(self.observation),
        }


@dataclass(frozen=True)
class HazardAuditDecision:
    """A replayable legal Spikes/Rapid Spin decision with public-only inputs."""

    battle_id: str
    format_id: str
    seed: int
    driver_id: str
    player_id: PlayerId
    decision_round: int
    target_action_index: int
    target_move_id: str
    observation_history: tuple[PublicActorObservation, ...]
    public_action_rounds: tuple[PublicActionRound, ...]

    def __post_init__(self) -> None:
        if self.decision_round < 0:
            raise ValueError("decision_round must be non-negative.")
        if self.target_move_id not in {"spikes", "rapidspin"}:
            raise ValueError("target_move_id must be 'spikes' or 'rapidspin'.")
        if not 0 <= self.target_action_index < ACTION_COUNT:
            raise ValueError(f"target_action_index must be in 0..{ACTION_COUNT - 1}.")
        history_turns = tuple(entry.turn_index for entry in self.observation_history)
        if not history_turns or history_turns[-1] != self.decision_round:
            raise ValueError("observation_history must end at decision_round.")
        if history_turns != tuple(sorted(set(history_turns))):
            raise ValueError("observation_history turn indexes must be unique and sorted.")
        if not self.observation.legal_action_mask[self.target_action_index]:
            raise ValueError("the hazard target action must be legal in the actor observation.")
        round_indexes = tuple(round_.turn_index for round_ in self.public_action_rounds)
        if round_indexes != tuple(range(self.decision_round)):
            raise ValueError("public_action_rounds must be contiguous from 0 through the replay prefix.")
        _assert_public_payload(self.public_payload())

    @property
    def observation(self) -> PokeZeroObservationV0:
        return self.observation_history[-1].observation

    @property
    def actor_history(self) -> tuple[PokeZeroObservationV0, ...]:
        return tuple(entry.observation for entry in self.observation_history)

    @property
    def state_id(self) -> str:
        return canonical_hash(self.public_payload())[:20]

    def public_payload(self) -> dict[str, object]:
        """Serializable corpus payload, excluding every opponent-private input."""

        return {
            "public_decision": self.to_public_decision_record(),
            "target": {
                "action_index": self.target_action_index,
                "move_id": self.target_move_id,
            },
        }

    def to_public_decision_record(self) -> dict[str, object]:
        """Emit the generic Step 2 replay-prefix corpus shape consumed by this audit.

        The record stays independent of the hazard target, so a state with both
        legal lines can produce two audit rows without forking corpus semantics.
        """

        return {
            "schema_version": PUBLIC_DECISION_CORPUS_SCHEMA_VERSION,
            "battle_id": self.battle_id,
            "format_id": self.format_id,
            "seed": self.seed,
            "driver_id": self.driver_id,
            "actor": self.player_id,
            "decision_round": self.decision_round,
            "replay_prefix": {
                "public_action_rounds": [round_.to_dict() for round_ in self.public_action_rounds],
            },
            "own_observation_history": [entry.to_dict() for entry in self.observation_history[:-1]],
            "current_observation": public_observation_to_dict(self.observation),
            "current_legal_action_mask": list(self.observation.legal_action_mask),
        }

    def to_dict(self) -> dict[str, object]:
        trajectory = self.to_search_trajectory()
        return {
            "state_id": self.state_id,
            "public_state_hash": canonical_hash(self.public_payload()),
            **self.public_payload(),
            # The replay trajectory is derived from the public payload.  Its non-actor
            # observations are explicit all-legal placeholders, never captured data.
            "replay_trajectory": _public_trajectory_to_dict(trajectory),
        }

    def to_search_trajectory(self) -> BattleTrajectory:
        """Build a replay trajectory without retaining any opponent-private mask."""

        history_by_turn = {entry.turn_index: entry.observation for entry in self.observation_history}
        trajectory = BattleTrajectory(
            battle_id=self.battle_id,
            format_id=self.format_id,
            seed=self.seed,
            metadata={"hazard_audit_public_only": True, "driver_id": self.driver_id},
        )
        for round_ in self.public_action_rounds:
            for player_id, action_index in sorted(round_.actions.items()):
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
        """A deliberately actor-only context for public-belief determinization/noise."""

        return PolicyContext(
            player_id=self.player_id,
            decision_round_index=self.decision_round,
            battle_id=self.battle_id,
            format_id=self.format_id,
            seed=self.seed,
            observation=self.observation,
            requested_players=(self.player_id,),
            trajectory=self.to_search_trajectory(),
        )


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
    metadata: Mapping[str, object] | None = None
    available: bool = True
    rejection_code: str | None = None

    def __post_init__(self) -> None:
        normalized = {str(player): int(action) for player, action in sorted(self.opponent_actions.items())}
        if any(action < 0 or action >= ACTION_COUNT for action in normalized.values()):
            raise ValueError(f"opponent action indices must be in 0..{ACTION_COUNT - 1}.")
        object.__setattr__(self, "opponent_actions", normalized)
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
                    prefix = replay_trajectory_prefix(
                        env,
                        decision.to_search_trajectory(),
                        decision_round_count=decision.decision_round,
                        start_override=start_override,
                        consistency_player_id=decision.player_id,
                        expected_current_observation=decision.observation,
                        check_prefix_observations=False,
                    )
                    if decision.player_id not in prefix.requested_players:
                        raise ValueError("sampled world does not request the actor")
                    opponent_actions = _sampled_world_opponent_actions(
                        env=env,
                        actor=decision.player_id,
                        requested_players=prefix.requested_players,
                        policy=self.sampled_world_opponent_policy,
                        seed=_stable_seed(decision.state_id, "sampled-world-opponent-policy", world_index),
                    )
                finally:
                    if env is not None:
                        _close_env(env)
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
                    metadata=_public_profile_metadata(
                        profile,
                        world_index,
                        opponent_policy_id=self.sampled_world_opponent_policy.policy_id,
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
    """Extract legal hazard targets while retaining no opponent-private step data."""

    actions_by_round: dict[int, dict[PlayerId, int]] = defaultdict(dict)
    for step in trajectory.steps:
        actions_by_round[step.turn_index][step.player_id] = step.action_index
    decisions: list[HazardAuditDecision] = []
    for step in trajectory.steps:
        targets = _hazard_target_moves(step.observation, step.legal_action_mask)
        if not targets:
            continue
        actor_history = tuple(
            PublicActorObservation(turn_index=other.turn_index, observation=other.observation)
            for other in trajectory.steps
            if other.player_id == step.player_id and other.turn_index <= step.turn_index
        )
        public_rounds = tuple(
            PublicActionRound(turn_index=turn_index, actions=actions_by_round[turn_index])
            for turn_index in range(step.turn_index)
        )
        for target_action_index, target_move_id in targets:
            decisions.append(
                HazardAuditDecision(
                    battle_id=trajectory.battle_id,
                    format_id=trajectory.format_id,
                    seed=trajectory.seed,
                    driver_id=driver_id,
                    player_id=step.player_id,
                    decision_round=step.turn_index,
                    target_action_index=target_action_index,
                    target_move_id=target_move_id,
                    observation_history=actor_history,
                    public_action_rounds=public_rounds,
                )
            )
    return tuple(decisions)


def hazard_audit_decisions_from_public_corpus(payload: Mapping[str, Any]) -> tuple[HazardAuditDecision, ...]:
    """Load legal hazard targets from Step 2's public replay-prefix corpus.

    The adapter accepts the announced v1 schema and the two harmless container
    names used by corpus writers (``records`` and ``decisions``). It reads only
    public action rounds, the actor's own history/current observation, and the
    actor's current legal mask; any request or opponent legality key is refused.
    """

    schema_version = str(payload.get("schema_version") or "")
    if schema_version != PUBLIC_DECISION_CORPUS_SCHEMA_VERSION:
        raise ValueError(
            "hazard audit requires "
            f"{PUBLIC_DECISION_CORPUS_SCHEMA_VERSION!r}, got {schema_version!r}."
        )
    raw_records = payload.get("records", payload.get("decisions", payload.get("states")))
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, str | bytes):
        raise ValueError("public decision corpus must contain a records, decisions, or states array.")
    decisions: list[HazardAuditDecision] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("public decision corpus records must be JSON objects.")
        record = raw_record.get("public_decision", raw_record)
        if not isinstance(record, Mapping):
            raise ValueError("public decision corpus record has an invalid public_decision payload.")
        _assert_public_payload(record)
        record_schema = str(record.get("schema_version") or schema_version)
        if record_schema != PUBLIC_DECISION_CORPUS_SCHEMA_VERSION:
            raise ValueError("public decision record schema does not match the corpus schema.")
        decisions.extend(_hazard_decisions_from_public_record(record))
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
    for decision in decision_list:
        priors = _validated_priors(action_priors(decision.actor_history))
        target_prior = _normalized_target_prior(
            priors,
            legal_mask=tuple(decision.observation.legal_action_mask),
            target_action_index=decision.target_action_index,
        )
        low_prior = target_prior <= config.low_prior_threshold
        worlds = tuple(sorted(world_provider(decision), key=lambda world: world.world_id))
        for world in worlds:
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
    aggregate = aggregate_hazard_audit_records(records)
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


def aggregate_hazard_audit_records(records: Iterable[Mapping[str, Any]]) -> dict[str, object]:
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
        complete = [arms for arms in paired.values() if set(arms) == {"deterministic", "dirichlet_audit_only"}]
        off_rate = _rate(sum(bool(arms["deterministic"]["target_selected"]) for arms in complete), len(complete))
        on_rate = _rate(sum(bool(arms["dirichlet_audit_only"]["target_selected"]) for arms in complete), len(complete))
        toward_target = sum(
            not bool(arms["deterministic"]["target_selected"])
            and bool(arms["dirichlet_audit_only"]["target_selected"])
            for arms in complete
        )
        away_from_target = sum(
            bool(arms["deterministic"]["target_selected"])
            and not bool(arms["dirichlet_audit_only"]["target_selected"])
            for arms in complete
        )
        any_choice_changed = sum(
            arms["deterministic"].get("selected_action_index")
            != arms["dirichlet_audit_only"].get("selected_action_index")
            for arms in complete
        )
        choice_delta[str(budget)] = {
            "paired_low_prior_target_lines": len(complete),
            "toward_low_prior_target_lines": toward_target,
            "away_from_low_prior_target_lines": away_from_target,
            "choice_rate_off": off_rate,
            "choice_rate_on": on_rate,
            "delta_choice_on": _rate(toward_target - away_from_target, len(complete)),
            "supplementary_any_action_change_rate": _rate(any_choice_changed, len(complete)),
            "interpretation": (
                "noise_only_choice_sensitivity"
                if budget == 0
                else "search_choice_change_toward_low_prior_target"
            ),
            "definition": "paired changes toward minus away from the low-prior hazard/spin target; this is a choice metric, not a rescue metric",
        }

    searched = [row for row in rows if row.get("status") == "searched"]
    return {
        "definitions": {
            "E": "share of unique legal hazard/spin target lines with normalized legal prior at or below low_prior_threshold",
            "R_off": "deterministic rescue rate; a rescue is at least one re-visit, not merely the mandatory initial visit",
            "DeltaChoice_on": "paired changes toward minus away from the low-prior hazard/spin target; budget 0 is noise-only sensitivity, never rescue",
            "entrenchment": "target_revisits == 0. Each legal target receives one mandatory initial-sweep visit per valid world, so entrenchment is never defined as target_visits == 0.",
        },
        "E": {
            "low_prior_lines": low_prior_count,
            "legal_target_lines": total_states,
            "rate": _rate(low_prior_count, total_states),
        },
        "R_off": off_rescue,
        "DeltaChoice_on": choice_delta,
        "coverage": {
            "records": len(rows),
            "searched_records": len(searched),
            "unavailable_records": len(rows) - len(searched),
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


def public_observation_to_dict(observation: PokeZeroObservationV0) -> dict[str, object]:
    """Serialize an actor observation and refuse accidental raw request fields."""

    payload = {
        "schema_version": observation.schema_version,
        "categorical_ids": observation.categorical_ids,
        "numeric_features": observation.numeric_features,
        "token_type_ids": observation.token_type_ids,
        "attention_mask": observation.attention_mask,
        "legal_action_mask": observation.legal_action_mask,
        "perspective": None
        if observation.perspective is None
        else {
            "player_id": observation.perspective.player_id,
            "showdown_slot": observation.perspective.showdown_slot,
            "opponent_showdown_slot": observation.perspective.opponent_showdown_slot,
        },
        "metadata": dict(observation.metadata),
    }
    _assert_public_payload(payload)
    return _jsonable(payload)


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
            trajectory=decision.to_search_trajectory(),
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


def _hazard_decisions_from_public_record(record: Mapping[str, Any]) -> tuple[HazardAuditDecision, ...]:
    replay_prefix = record.get("replay_prefix", {})
    if not isinstance(replay_prefix, Mapping):
        raise ValueError("public decision replay_prefix must be an object.")
    actor = record.get("actor", record.get("player_id"))
    if not isinstance(actor, str) or not actor:
        raise ValueError("public decision record must name an actor.")
    try:
        decision_round = int(record["decision_round"])
        seed = int(record["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("public decision record must include integer seed and decision_round.") from exc
    current_payload = record.get(
        "current_observation",
        record.get("own_observation", record.get("observation")),
    )
    current = _public_observation_from_dict(current_payload)
    mask_payload = record.get("current_legal_action_mask", record.get("legal_action_mask"))
    if mask_payload is not None:
        mask = tuple(bool(value) for value in _require_sequence(mask_payload, "current_legal_action_mask"))
        if len(mask) != ACTION_COUNT:
            raise ValueError(f"current_legal_action_mask must contain {ACTION_COUNT} values.")
        current = replace(current, legal_action_mask=mask)
    history_payload = record.get("own_observation_history", record.get("public_observation_history", ()))
    history = tuple(_public_actor_observation_from_dict(value) for value in _require_sequence(history_payload, "own_observation_history"))
    if not history or history[-1].turn_index != decision_round:
        history = (*history, PublicActorObservation(turn_index=decision_round, observation=current))
    else:
        history = (*history[:-1], PublicActorObservation(turn_index=decision_round, observation=current))
    rounds_payload = replay_prefix.get(
        "public_action_rounds",
        record.get("public_action_rounds", replay_prefix.get("action_rounds", ())),
    )
    public_rounds = tuple(
        _public_action_round_from_dict(value)
        for value in _require_sequence(rounds_payload, "public_action_rounds")
    )
    targets = _hazard_target_moves(current, current.legal_action_mask)
    return tuple(
        HazardAuditDecision(
            battle_id=str(record.get("battle_id") or "public-decision-corpus"),
            format_id=str(record.get("format_id") or "gen3randombattle"),
            seed=seed,
            driver_id=str(record.get("driver_id") or "public-decision-corpus"),
            player_id=actor,
            decision_round=decision_round,
            target_action_index=target_action_index,
            target_move_id=target_move_id,
            observation_history=history,
            public_action_rounds=public_rounds,
        )
        for target_action_index, target_move_id in targets
    )


def _public_actor_observation_from_dict(value: object) -> PublicActorObservation:
    if not isinstance(value, Mapping):
        raise ValueError("own_observation_history entries must be objects.")
    try:
        turn_index = int(value["turn_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("own_observation_history entry is missing turn_index.") from exc
    return PublicActorObservation(
        turn_index=turn_index,
        observation=_public_observation_from_dict(value.get("observation")),
    )


def _public_action_round_from_dict(value: object) -> PublicActionRound:
    if not isinstance(value, Mapping):
        raise ValueError("public_action_rounds entries must be objects.")
    actions = value.get("actions")
    if not isinstance(actions, Mapping):
        raise ValueError("public action rounds must contain an actions object.")
    try:
        turn_index = int(value["turn_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("public action round is missing turn_index.") from exc
    return PublicActionRound(turn_index=turn_index, actions=actions)


def _public_observation_from_dict(value: object) -> PokeZeroObservationV0:
    if isinstance(value, PokeZeroObservationV0):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("current_observation must be an object.")
    _assert_public_payload(value)
    perspective_payload = value.get("perspective")
    perspective = None
    if isinstance(perspective_payload, Mapping):
        perspective = ObservationPerspective(
            player_id=str(perspective_payload["player_id"]),
            showdown_slot=str(perspective_payload["showdown_slot"]),
            opponent_showdown_slot=str(perspective_payload["opponent_showdown_slot"]),
        )
    return PokeZeroObservationV0(
        categorical_ids=tuple(tuple(int(item) for item in row) for row in _require_sequence(value.get("categorical_ids", ()), "categorical_ids")),
        numeric_features=tuple(tuple(float(item) for item in row) for row in _require_sequence(value.get("numeric_features", ()), "numeric_features")),
        token_type_ids=tuple(int(item) for item in _require_sequence(value.get("token_type_ids", ()), "token_type_ids")),
        attention_mask=tuple(bool(item) for item in _require_sequence(value.get("attention_mask", ()), "attention_mask")),
        legal_action_mask=tuple(bool(item) for item in _require_sequence(value.get("legal_action_mask", ()), "legal_action_mask")),
        perspective=perspective,
        metadata=dict(value.get("metadata", {})),
        schema_version=str(value.get("schema_version") or "pokezero.observation.unversioned"),
    )


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{name} must be an array.")
    return value


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
) -> dict[str, object]:
    return {
        "world_index": world_index,
        "belief_sampling": profile.to_metadata(),
        "world_source": "public-belief-determinization",
        "sampled_world_opponent_policy": opponent_policy_id,
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


def _public_trajectory_to_dict(trajectory: BattleTrajectory) -> dict[str, object]:
    return {
        "battle_id": trajectory.battle_id,
        "format_id": trajectory.format_id,
        "seed": trajectory.seed,
        "metadata": dict(trajectory.metadata),
        "steps": [
            {
                "player_id": step.player_id,
                "turn_index": step.turn_index,
                "action_index": step.action_index,
                "metadata": dict(step.metadata),
                "observation": public_observation_to_dict(step.observation),
                "legal_action_mask": list(step.legal_action_mask),
            }
            for step in trajectory.steps
        ],
    }


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 8)


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
