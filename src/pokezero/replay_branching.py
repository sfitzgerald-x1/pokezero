"""Replay-from-root helpers for future search/forking code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .actions import ACTION_COUNT
from .env import BattleFormat, BattleStartOverride, PlayerId, PokeZeroEnv, StepResult, TerminalState
from .observation import (
    ACTION_CANDIDATE_TOKEN_COUNT,
    FIELD_TOKEN_COUNT,
    OPPONENT_POKEMON_TOKEN_COUNT,
    PokeZeroObservationV0,
    SELF_POKEMON_TOKEN_COUNT,
)
from .policy import Policy
from .rollout import RolloutConfig, RolloutResult, continue_rollout_from_current_state
from .trajectory import BattleTrajectory


_RECENT_EVENT_TOKEN_OFFSET = (
    FIELD_TOKEN_COUNT
    + SELF_POKEMON_TOKEN_COUNT
    + OPPONENT_POKEMON_TOKEN_COUNT
    + ACTION_CANDIDATE_TOKEN_COUNT
)
_NUMERIC_HP_FRACTION_INDEX = 0
_SELF_POKEMON_TOKEN_OFFSET = FIELD_TOKEN_COUNT
_OPPONENT_POKEMON_TOKEN_OFFSET = _SELF_POKEMON_TOKEN_OFFSET + SELF_POKEMON_TOKEN_COUNT


@dataclass(frozen=True)
class ReplayActionRound:
    """One environment decision boundary worth of recorded player actions."""

    turn_index: int
    actions: Mapping[PlayerId, int]
    expected_observations: Mapping[PlayerId, PokeZeroObservationV0] | None = None

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")
        if not self.actions:
            raise ValueError("actions must be non-empty.")
        normalized = {
            str(player): int(action_index)
            for player, action_index in sorted(self.actions.items(), key=lambda item: str(item[0]))
        }
        invalid_actions = [
            action_index
            for action_index in normalized.values()
            if action_index < 0 or action_index >= ACTION_COUNT
        ]
        if invalid_actions:
            raise ValueError(f"action indices must be between 0 and {ACTION_COUNT - 1}.")
        object.__setattr__(self, "actions", normalized)
        if self.expected_observations is not None:
            normalized_observations = {
                str(player): observation
                for player, observation in sorted(
                    self.expected_observations.items(),
                    key=lambda item: str(item[0]),
                )
            }
            unknown_observation_players = sorted(set(normalized_observations) - set(normalized))
            if unknown_observation_players:
                raise ValueError(
                    "expected_observations must only include players with recorded actions; "
                    f"unexpected: {', '.join(unknown_observation_players)}."
                )
            object.__setattr__(self, "expected_observations", normalized_observations)


@dataclass(frozen=True)
class ReplayPrefixResult:
    """State summary after replaying a prefix into an environment."""

    replayed_round_count: int
    requested_players: tuple[PlayerId, ...]
    terminal: TerminalState | None


@dataclass(frozen=True)
class ReplayBranchResult:
    """State summary after replaying a prefix and submitting one branch action round."""

    prefix: ReplayPrefixResult
    branch_round: ReplayActionRound
    step_result: StepResult


@dataclass(frozen=True)
class ReplayBranchRolloutResult:
    """A branch action plus policy rollout continuation from the resulting state."""

    branch: ReplayBranchResult
    continuation: RolloutResult


def action_rounds_from_trajectory(
    trajectory: BattleTrajectory,
    *,
    decision_round_count: int | None = None,
) -> tuple[ReplayActionRound, ...]:
    """Group trajectory steps into replayable action rounds.

    ``TrajectoryStep.turn_index`` is the rollout driver's decision-round index. Search can replay
    the first ``decision_round_count`` rounds from the original seed, then submit a different next
    action to explore a branch.
    """

    if decision_round_count is not None and decision_round_count < 0:
        raise ValueError("decision_round_count must be non-negative when set.")

    grouped: dict[int, dict[PlayerId, int]] = {}
    observations_by_round: dict[int, dict[PlayerId, PokeZeroObservationV0]] = {}
    for step in trajectory.steps:
        if decision_round_count is not None and step.turn_index >= decision_round_count:
            continue
        actions = grouped.setdefault(step.turn_index, {})
        if step.player_id in actions:
            raise ValueError(
                f"trajectory has duplicate action for player {step.player_id!r} "
                f"at decision round {step.turn_index}."
            )
        actions[step.player_id] = step.action_index
        observations_by_round.setdefault(step.turn_index, {})[step.player_id] = step.observation

    expected_turn = 0
    rounds: list[ReplayActionRound] = []
    for turn_index in sorted(grouped):
        if turn_index != expected_turn:
            raise ValueError(
                f"trajectory action rounds must be contiguous from 0; "
                f"missing decision round {expected_turn}."
            )
        rounds.append(
            ReplayActionRound(
                turn_index=turn_index,
                actions=grouped[turn_index],
                expected_observations=observations_by_round.get(turn_index),
            )
        )
        expected_turn += 1
    if decision_round_count is not None and len(rounds) != decision_round_count:
        raise ValueError(
            f"trajectory contains {len(rounds)} replayable decision rounds, "
            f"but {decision_round_count} were requested."
        )
    return tuple(rounds)


def replay_action_rounds(
    env: PokeZeroEnv,
    *,
    seed: int,
    format_id: BattleFormat = "gen3randombattle",
    action_rounds: tuple[ReplayActionRound, ...],
    start_override: BattleStartOverride | None = None,
    consistency_player_id: PlayerId | None = None,
    expected_current_observation: PokeZeroObservationV0 | None = None,
    check_prefix_observations: bool = True,
    hp_fraction_tolerance: float = 0.0,
) -> ReplayPrefixResult:
    """Reset ``env`` and replay a recorded action prefix from the battle root.

    Prefix observation checks remain enabled by default for strict replay audits. Search callers
    that supply ``expected_current_observation`` may disable them when only the branch-point state
    must match the recorded battle.
    """

    _reset_env(env, seed=seed, format_id=format_id, start_override=start_override)
    for expected_index, action_round in enumerate(action_rounds):
        if action_round.turn_index != expected_index:
            raise ValueError(
                f"action_rounds must be contiguous from 0; expected decision round "
                f"{expected_index}, got {action_round.turn_index}."
            )
        terminal = env.terminal()
        if terminal is not None:
            raise ValueError(
                f"cannot replay decision round {action_round.turn_index}; "
                "environment reached terminal early."
            )
        _require_requested_players(
            action_round,
            requested_players=env.requested_players(),
        )
        if (
            check_prefix_observations
            and start_override is not None
            and consistency_player_id is not None
        ):
            _require_expected_observation(
                env,
                action_round,
                player_id=consistency_player_id,
                ignore_recent_events=True,
                hp_fraction_tolerance=hp_fraction_tolerance,
            )
        env.step(action_round.actions)

    if (
        start_override is not None
        and consistency_player_id is not None
        and expected_current_observation is not None
    ):
        _require_observation_match(
            env,
            expected=expected_current_observation,
            player_id=consistency_player_id,
            turn_index=len(action_rounds),
            ignore_recent_events=True,
            hp_fraction_tolerance=hp_fraction_tolerance,
        )

    return ReplayPrefixResult(
        replayed_round_count=len(action_rounds),
        requested_players=env.requested_players(),
        terminal=env.terminal(),
    )


def replay_trajectory_prefix(
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    *,
    decision_round_count: int,
    start_override: BattleStartOverride | None = None,
    consistency_player_id: PlayerId | None = None,
    expected_current_observation: PokeZeroObservationV0 | None = None,
    check_prefix_observations: bool = True,
    hp_fraction_tolerance: float = 0.0,
) -> ReplayPrefixResult:
    """Replay the first N decision rounds from a trajectory into ``env``."""

    return replay_action_rounds(
        env,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        action_rounds=action_rounds_from_trajectory(
            trajectory,
            decision_round_count=decision_round_count,
        ),
        start_override=start_override,
        consistency_player_id=consistency_player_id,
        expected_current_observation=expected_current_observation,
        check_prefix_observations=check_prefix_observations,
        hp_fraction_tolerance=hp_fraction_tolerance,
    )


def replay_trajectory_branch(
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    *,
    prefix_decision_round_count: int,
    branch_actions: Mapping[PlayerId, int],
    start_override: BattleStartOverride | None = None,
    consistency_player_id: PlayerId | None = None,
    expected_current_observation: PokeZeroObservationV0 | None = None,
    check_prefix_observations: bool = True,
    hp_fraction_tolerance: float = 0.0,
) -> ReplayBranchResult:
    """Replay a trajectory prefix, submit one explicit branch action, and leave ``env`` there."""

    prefix = replay_trajectory_prefix(
        env,
        trajectory,
        decision_round_count=prefix_decision_round_count,
        start_override=start_override,
        consistency_player_id=consistency_player_id,
        expected_current_observation=expected_current_observation,
        check_prefix_observations=check_prefix_observations,
        hp_fraction_tolerance=hp_fraction_tolerance,
    )
    if prefix.terminal is not None:
        raise ValueError("cannot branch from a terminal replay prefix.")
    branch_round = ReplayActionRound(
        turn_index=prefix_decision_round_count,
        actions=branch_actions,
    )
    _require_requested_players(
        branch_round,
        requested_players=prefix.requested_players,
    )
    step_result = env.step(branch_round.actions)
    return ReplayBranchResult(
        prefix=prefix,
        branch_round=branch_round,
        step_result=step_result,
    )


def replay_trajectory_branch_rollout(
    env: PokeZeroEnv,
    trajectory: BattleTrajectory,
    *,
    prefix_decision_round_count: int,
    branch_actions: Mapping[PlayerId, int],
    policies: Mapping[PlayerId, Policy],
    rollout_config: RolloutConfig,
    battle_id: str = "replay-branch-rollout",
    reset_policies: bool = True,
    start_override: BattleStartOverride | None = None,
    consistency_player_id: PlayerId | None = None,
    expected_current_observation: PokeZeroObservationV0 | None = None,
    check_prefix_observations: bool = True,
    hp_fraction_tolerance: float = 0.0,
) -> ReplayBranchRolloutResult:
    """Replay, branch once, then continue the rollout with policies until terminal or cap."""

    branch = replay_trajectory_branch(
        env,
        trajectory,
        prefix_decision_round_count=prefix_decision_round_count,
        branch_actions=branch_actions,
        start_override=start_override,
        consistency_player_id=consistency_player_id,
        expected_current_observation=expected_current_observation,
        check_prefix_observations=check_prefix_observations,
        hp_fraction_tolerance=hp_fraction_tolerance,
    )
    continuation = continue_rollout_from_current_state(
        env=env,
        policies=policies,
        config=rollout_config,
        seed=trajectory.seed,
        battle_id=battle_id,
        starting_decision_round_index=prefix_decision_round_count + 1,
        available_observations=branch.step_result.observations,
        reset_policies=reset_policies,
    )
    return ReplayBranchRolloutResult(
        branch=branch,
        continuation=continuation,
    )


def _require_requested_players(
    action_round: ReplayActionRound,
    *,
    requested_players: tuple[PlayerId, ...],
) -> None:
    requested_set = set(requested_players)
    action_players = set(action_round.actions)
    if action_players == requested_set:
        return
    missing = sorted(requested_set - action_players)
    extra = sorted(action_players - requested_set)
    details: list[str] = []
    if missing:
        details.append(f"missing requested players: {', '.join(missing)}")
    if extra:
        details.append(f"unexpected players: {', '.join(extra)}")
    raise ValueError(
        f"replay actions for decision round {action_round.turn_index} "
        f"do not match environment request ({'; '.join(details)})."
    )


def _reset_env(
    env: PokeZeroEnv,
    *,
    seed: int,
    format_id: BattleFormat,
    start_override: BattleStartOverride | None,
) -> None:
    if start_override is None:
        env.reset(seed=seed, format_id=format_id)
        return
    resetter = getattr(env, "reset_with_start_override", None)
    if not callable(resetter):
        raise ValueError("environment does not support replay start overrides.")
    resetter(seed=seed, format_id=start_override.format_id, start_override=start_override)


def _require_expected_observation(
    env: PokeZeroEnv,
    action_round: ReplayActionRound,
    *,
    player_id: PlayerId,
    ignore_recent_events: bool = False,
    hp_fraction_tolerance: float = 0.0,
) -> None:
    if not action_round.expected_observations:
        return
    expected = action_round.expected_observations.get(player_id)
    if expected is None:
        return
    _require_observation_match(
        env,
        expected=expected,
        player_id=player_id,
        turn_index=action_round.turn_index,
        ignore_recent_events=ignore_recent_events,
        hp_fraction_tolerance=hp_fraction_tolerance,
    )


def _require_observation_match(
    env: PokeZeroEnv,
    *,
    expected: PokeZeroObservationV0,
    player_id: PlayerId,
    turn_index: int,
    ignore_recent_events: bool = False,
    hp_fraction_tolerance: float = 0.0,
) -> None:
    actual = env.observe(player_id)
    if not _observations_match_for_replay(
        actual,
        expected,
        ignore_recent_events=ignore_recent_events,
        hp_fraction_tolerance=hp_fraction_tolerance,
    ):
        details = _observation_replay_mismatch_details(
            actual,
            expected,
            ignore_recent_events=ignore_recent_events,
            hp_fraction_tolerance=hp_fraction_tolerance,
        )
        detail_suffix = f" ({details})" if details else ""
        raise ValueError(
            "start override does not reproduce recorded replay prefix observations "
            f"for decision round {turn_index}: {player_id}.{detail_suffix}"
        )


def _observations_match_for_replay(
    actual: PokeZeroObservationV0,
    expected: PokeZeroObservationV0,
    *,
    ignore_recent_events: bool = False,
    hp_fraction_tolerance: float = 0.0,
) -> bool:
    categorical_actual = actual.categorical_ids
    categorical_expected = expected.categorical_ids
    numeric_actual = actual.numeric_features
    numeric_expected = expected.numeric_features
    token_type_actual = actual.token_type_ids
    token_type_expected = expected.token_type_ids
    attention_actual = actual.attention_mask
    attention_expected = expected.attention_mask
    if ignore_recent_events:
        # Start overrides for randbats use gen3customgame because Showdown only honors arbitrary
        # packed teams there. Its startup rule/tier protocol lines differ from gen3randombattle
        # even when the current battle state, request, teams, and legal actions are faithful.
        categorical_actual = categorical_actual[:_RECENT_EVENT_TOKEN_OFFSET]
        categorical_expected = categorical_expected[:_RECENT_EVENT_TOKEN_OFFSET]
        numeric_actual = numeric_actual[:_RECENT_EVENT_TOKEN_OFFSET]
        numeric_expected = numeric_expected[:_RECENT_EVENT_TOKEN_OFFSET]
        token_type_actual = token_type_actual[:_RECENT_EVENT_TOKEN_OFFSET]
        token_type_expected = token_type_expected[:_RECENT_EVENT_TOKEN_OFFSET]
        attention_actual = attention_actual[:_RECENT_EVENT_TOKEN_OFFSET]
        attention_expected = attention_expected[:_RECENT_EVENT_TOKEN_OFFSET]
    return (
        actual.schema_version == expected.schema_version
        and categorical_actual == categorical_expected
        and _numeric_features_match_for_replay(
            numeric_actual,
            numeric_expected,
            hp_fraction_tolerance=hp_fraction_tolerance,
        )
        and token_type_actual == token_type_expected
        and attention_actual == attention_expected
        and tuple(actual.legal_action_mask) == tuple(expected.legal_action_mask)
        and actual.perspective == expected.perspective
    )


def _observation_replay_mismatch_details(
    actual: PokeZeroObservationV0,
    expected: PokeZeroObservationV0,
    *,
    ignore_recent_events: bool = False,
    hp_fraction_tolerance: float = 0.0,
) -> str | None:
    if actual.schema_version != expected.schema_version:
        return (
            f"schema_version: actual={_format_mismatch_value(actual.schema_version)} "
            f"expected={_format_mismatch_value(expected.schema_version)}"
        )

    categorical_actual = actual.categorical_ids
    categorical_expected = expected.categorical_ids
    numeric_actual = actual.numeric_features
    numeric_expected = expected.numeric_features
    token_type_actual = actual.token_type_ids
    token_type_expected = expected.token_type_ids
    attention_actual = actual.attention_mask
    attention_expected = expected.attention_mask
    if ignore_recent_events:
        categorical_actual = categorical_actual[:_RECENT_EVENT_TOKEN_OFFSET]
        categorical_expected = categorical_expected[:_RECENT_EVENT_TOKEN_OFFSET]
        numeric_actual = numeric_actual[:_RECENT_EVENT_TOKEN_OFFSET]
        numeric_expected = numeric_expected[:_RECENT_EVENT_TOKEN_OFFSET]
        token_type_actual = token_type_actual[:_RECENT_EVENT_TOKEN_OFFSET]
        token_type_expected = token_type_expected[:_RECENT_EVENT_TOKEN_OFFSET]
        attention_actual = attention_actual[:_RECENT_EVENT_TOKEN_OFFSET]
        attention_expected = attention_expected[:_RECENT_EVENT_TOKEN_OFFSET]

    for name, actual_values, expected_values in (
        ("categorical_ids", categorical_actual, categorical_expected),
        ("numeric_features", numeric_actual, numeric_expected),
        ("token_type_ids", token_type_actual, token_type_expected),
        ("attention_mask", attention_actual, attention_expected),
        ("legal_action_mask", actual.legal_action_mask, expected.legal_action_mask),
    ):
        mismatch = _first_mismatch(
            actual_values,
            expected_values,
            hp_fraction_tolerance=hp_fraction_tolerance if name == "numeric_features" else 0.0,
        )
        if mismatch is not None:
            path, actual_value, expected_value = mismatch
            return (
                f"{_format_mismatch_path(name, path, token_segmented=name != 'legal_action_mask')}: "
                f"actual={_format_mismatch_value(actual_value)} "
                f"expected={_format_mismatch_value(expected_value)}"
            )

    if actual.perspective != expected.perspective:
        return (
            f"perspective: actual={_format_mismatch_value(actual.perspective)} "
            f"expected={_format_mismatch_value(expected.perspective)}"
        )
    return None


def _numeric_features_match_for_replay(
    actual: tuple[tuple[float, ...], ...],
    expected: tuple[tuple[float, ...], ...],
    *,
    hp_fraction_tolerance: float,
) -> bool:
    return _first_mismatch(
        actual,
        expected,
        hp_fraction_tolerance=hp_fraction_tolerance,
    ) is None


def _first_mismatch(
    actual,
    expected,
    path: tuple[int | str, ...] = (),
    *,
    hp_fraction_tolerance: float = 0.0,
):
    try:
        if actual == expected or _within_hp_fraction_tolerance(
            actual,
            expected,
            path=path,
            hp_fraction_tolerance=hp_fraction_tolerance,
        ):
            return None
    except ValueError:
        # Some array types return elementwise comparisons. Fall through to indexed comparison.
        pass
    if _is_indexable_sequence(actual) and _is_indexable_sequence(expected):
        actual_len = len(actual)
        expected_len = len(expected)
        if actual_len != expected_len:
            return (path + ("len",), actual_len, expected_len)
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            mismatch = _first_mismatch(
                actual_item,
                expected_item,
                path + (index,),
                hp_fraction_tolerance=hp_fraction_tolerance,
            )
            if mismatch is not None:
                return mismatch
        return None
    return (path, actual, expected)


def _within_hp_fraction_tolerance(
    actual,
    expected,
    *,
    path: tuple[int | str, ...],
    hp_fraction_tolerance: float,
) -> bool:
    if hp_fraction_tolerance <= 0.0:
        return False
    if (
        len(path) < 2
        or path[-1] != _NUMERIC_HP_FRACTION_INDEX
        or not _is_pokemon_token_path(path)
    ):
        return False
    if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
        return False
    return abs(float(actual) - float(expected)) <= hp_fraction_tolerance


def _is_pokemon_token_path(path: tuple[int | str, ...]) -> bool:
    token_index = path[-2]
    if not isinstance(token_index, int):
        return False
    return (
        _SELF_POKEMON_TOKEN_OFFSET <= token_index < _OPPONENT_POKEMON_TOKEN_OFFSET + OPPONENT_POKEMON_TOKEN_COUNT
    )


def _is_indexable_sequence(value) -> bool:
    return not isinstance(value, (str, bytes)) and hasattr(value, "__len__") and hasattr(value, "__getitem__")


def _format_mismatch_path(
    name: str,
    path: tuple[int | str, ...],
    *,
    token_segmented: bool,
) -> str:
    if not path:
        return name
    first = path[0]
    segment = f"/{_token_segment(first)}" if token_segmented and isinstance(first, int) else ""
    suffix = "".join(f"[{part}]" for part in path)
    return f"{name}{segment}{suffix}"


def _token_segment(index: int) -> str:
    if index < FIELD_TOKEN_COUNT:
        return "field"
    if index < FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT:
        return "self_pokemon"
    if index < FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT + OPPONENT_POKEMON_TOKEN_COUNT:
        return "opponent_pokemon"
    if index < _RECENT_EVENT_TOKEN_OFFSET:
        return "action_candidates"
    return "recent_events"


def _format_mismatch_value(value) -> str:
    text = repr(value)
    if len(text) > 120:
        return text[:117] + "..."
    return text
