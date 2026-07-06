"""G4 refutation-mining primitives.

The miner post-mortems games a champion won, searches the loser's legal
single-turn deviations, and certifies a deviation only from terminal rollout
outcomes.  It deliberately has no value-head dependency: callers inject an
evaluator that returns terminal winners for branch/reseed pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence, TextIO

from .actions import ACTION_COUNT
from .collection import RolloutRecord
from .env import PokeZeroEnv, TerminalState
from .policy import Policy, legal_action_indices
from .replay_branching import action_rounds_from_trajectory, replay_trajectory_branch
from .rollout import RolloutConfig, continue_rollout_from_current_state
from .trajectory import BattleTrajectory, TrajectoryStep


FRAGILE_STATE_SCHEMA_VERSION = "pokezero.fragile_state.v1"
REFUTATION_REPORT_SCHEMA_VERSION = "pokezero.refutation_report.v1"
TERMINAL_ROLLOUT_EVALUATION_SOURCE = "terminal_rollout"


@dataclass(frozen=True)
class RefutationMiningConfig:
    """Configuration for R0 single-turn refutation mining."""

    champion_policy_id: str | None = None
    champion_player_id: str | None = None
    max_wins: int = 200
    max_decision_points_per_game: int | None = None
    max_deviations_per_state: int | None = None
    certification_seed_count: int = 20
    min_flip_rate: float = 0.60
    mode: str = "oracle"

    def __post_init__(self) -> None:
        if self.champion_policy_id is None and self.champion_player_id is None:
            raise ValueError("champion_policy_id or champion_player_id is required.")
        if self.max_wins <= 0:
            raise ValueError("max_wins must be positive.")
        if self.max_decision_points_per_game is not None and self.max_decision_points_per_game <= 0:
            raise ValueError("max_decision_points_per_game must be positive when set.")
        if self.max_deviations_per_state is not None and self.max_deviations_per_state <= 0:
            raise ValueError("max_deviations_per_state must be positive when set.")
        if self.certification_seed_count < 20:
            raise ValueError("certification_seed_count must be at least 20.")
        if not 0.0 < self.min_flip_rate < 1.0:
            raise ValueError("min_flip_rate must be between 0 and 1.")
        if self.mode not in {"oracle", "fair"}:
            raise ValueError("mode must be 'oracle' or 'fair'.")


@dataclass(frozen=True)
class RefutationCandidate:
    """A loser-seat single-turn deviation at a recorded decision point."""

    battle_id: str
    source_record_index: int
    seed: int
    format_id: str
    champion_player_id: str
    loser_player_id: str
    decision_round_index: int
    step_index: int
    recorded_action_index: int
    deviation_action_index: int
    branch_actions: Mapping[str, int]

    def __post_init__(self) -> None:
        for name, action_index in (
            ("recorded_action_index", self.recorded_action_index),
            ("deviation_action_index", self.deviation_action_index),
        ):
            if action_index < 0 or action_index >= ACTION_COUNT:
                raise ValueError(f"{name} must be between 0 and {ACTION_COUNT - 1}.")
        if self.recorded_action_index == self.deviation_action_index:
            raise ValueError("deviation_action_index must differ from recorded_action_index.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "battle_id": self.battle_id,
            "source_record_index": self.source_record_index,
            "seed": self.seed,
            "format_id": self.format_id,
            "champion_player_id": self.champion_player_id,
            "loser_player_id": self.loser_player_id,
            "decision_round_index": self.decision_round_index,
            "step_index": self.step_index,
            "recorded_action_index": self.recorded_action_index,
            "deviation_action_index": self.deviation_action_index,
            "branch_actions": dict(sorted(self.branch_actions.items())),
        }


@dataclass(frozen=True)
class BranchTerminalResult:
    """Terminal outcome for one branch/reseed evaluation."""

    certification_seed: int
    winner: str | None
    capped: bool = False
    turn_count: int | None = None

    @classmethod
    def from_terminal(cls, *, certification_seed: int, terminal: TerminalState) -> "BranchTerminalResult":
        return cls(
            certification_seed=certification_seed,
            winner=terminal.winner,
            capped=terminal.capped,
            turn_count=terminal.turn_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "certification_seed": self.certification_seed,
            "winner": self.winner,
            "capped": self.capped,
            **({"turn_count": self.turn_count} if self.turn_count is not None else {}),
        }


class TerminalBranchEvaluator(Protocol):
    """Evaluates a branch by playing to terminal, never by querying a value head."""

    evaluation_source: str

    def evaluate(
        self,
        *,
        record: RolloutRecord,
        candidate: RefutationCandidate,
        certification_seed: int,
    ) -> BranchTerminalResult:
        ...


@dataclass(frozen=True)
class ReplayTerminalBranchEvaluator:
    """Replay-from-root branch evaluator backed by the live environment.

    Replay uses the recorded battle seed to reach the branch state.  The
    certification seed is then used for continuation policy RNG.  This keeps R0
    compatible with today's replay-from-root harness; full simulator-RNG
    reseeding needs a future snapshot/restore backend.
    """

    env_factory: Callable[[], PokeZeroEnv]
    policies: Mapping[str, Policy]
    rollout_config: RolloutConfig
    reset_policies: bool = True
    check_prefix_observations: bool = False
    evaluation_source: str = TERMINAL_ROLLOUT_EVALUATION_SOURCE
    reseed_scope: str = "continuation_policy_rng"

    def evaluate(
        self,
        *,
        record: RolloutRecord,
        candidate: RefutationCandidate,
        certification_seed: int,
    ) -> BranchTerminalResult:
        env: PokeZeroEnv = self.env_factory()
        try:
            branch = replay_trajectory_branch(
                env,
                record.trajectory,
                prefix_decision_round_count=candidate.decision_round_index,
                branch_actions=candidate.branch_actions,
                check_prefix_observations=self.check_prefix_observations,
            )
            continuation = continue_rollout_from_current_state(
                env=env,
                policies=self.policies,
                config=self.rollout_config,
                seed=certification_seed,
                battle_id=f"refutation-{record.battle_id}-{candidate.decision_round_index}-{certification_seed}",
                starting_decision_round_index=candidate.decision_round_index + 1,
                available_observations=branch.step_result.observations,
                reset_policies=self.reset_policies,
            )
            return BranchTerminalResult.from_terminal(
                certification_seed=certification_seed,
                terminal=continuation.terminal,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()


@dataclass(frozen=True)
class CertifiedRefutation:
    """A deviation that flips the recorded winner over repeated terminal rollouts."""

    candidate: RefutationCandidate
    evaluation_source: str
    mode: str
    certification_seed_count: int
    min_flip_rate: float
    deviation_wins: int
    champion_wins: int
    ties_or_caps: int
    flip_rate: float
    terminal_results: tuple[BranchTerminalResult, ...]
    search_stats: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FRAGILE_STATE_SCHEMA_VERSION,
            "mode": self.mode,
            "evaluation_source": self.evaluation_source,
            "candidate": self.candidate.to_dict(),
            "certification": {
                "seed_count": self.certification_seed_count,
                "min_flip_rate": self.min_flip_rate,
                "deviation_wins": self.deviation_wins,
                "champion_wins": self.champion_wins,
                "ties_or_caps": self.ties_or_caps,
                "flip_rate": self.flip_rate,
                "passed": self.flip_rate > self.min_flip_rate,
            },
            "terminal_results": [result.to_dict() for result in self.terminal_results],
            "search_stats": dict(self.search_stats),
        }


@dataclass(frozen=True)
class RefutationMiningReport:
    config: RefutationMiningConfig
    source_record_count: int
    sampled_win_count: int
    scanned_decision_count: int
    candidate_deviation_count: int
    evaluated_deviation_count: int
    certified_refutations: tuple[CertifiedRefutation, ...]
    archive_path: Path

    @property
    def refutation_rate(self) -> float:
        return len(self.certified_refutations) / self.sampled_win_count if self.sampled_win_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REFUTATION_REPORT_SCHEMA_VERSION,
            "config": {
                "champion_policy_id": self.config.champion_policy_id,
                "champion_player_id": self.config.champion_player_id,
                "max_wins": self.config.max_wins,
                "max_decision_points_per_game": self.config.max_decision_points_per_game,
                "max_deviations_per_state": self.config.max_deviations_per_state,
                "certification_seed_count": self.config.certification_seed_count,
                "min_flip_rate": self.config.min_flip_rate,
                "mode": self.config.mode,
            },
            "source_record_count": self.source_record_count,
            "sampled_win_count": self.sampled_win_count,
            "scanned_decision_count": self.scanned_decision_count,
            "candidate_deviation_count": self.candidate_deviation_count,
            "evaluated_deviation_count": self.evaluated_deviation_count,
            "certified_refutation_count": len(self.certified_refutations),
            "refutation_rate": self.refutation_rate,
            "archive_path": str(self.archive_path),
            "examples": [refutation.to_dict() for refutation in self.certified_refutations[:10]],
        }


def mine_refutations(
    *,
    records: Iterable[RolloutRecord],
    config: RefutationMiningConfig,
    evaluator: TerminalBranchEvaluator,
    archive_path: Path,
) -> RefutationMiningReport:
    """Mine and certify loser-seat deviations from champion wins."""

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    source_records = tuple(records)
    sampled_wins = tuple(_iter_champion_wins(source_records, config=config))
    scanned_decision_count = 0
    candidate_deviation_count = 0
    evaluated_deviation_count = 0
    certified: list[CertifiedRefutation] = []

    with archive_path.open("w", encoding="utf-8") as handle:
        for record_index, record, champion_player_id, loser_player_id in sampled_wins:
            decision_steps = _loser_decision_steps(record.trajectory, loser_player_id)
            if config.max_decision_points_per_game is not None:
                decision_steps = decision_steps[: config.max_decision_points_per_game]
            scanned_decision_count += len(decision_steps)
            for step_index, step in decision_steps:
                candidates = _deviation_candidates(
                    record=record,
                    source_record_index=record_index,
                    champion_player_id=champion_player_id,
                    loser_player_id=loser_player_id,
                    step_index=step_index,
                    step=step,
                    max_deviations=config.max_deviations_per_state,
                )
                candidate_deviation_count += len(candidates)
                for candidate in candidates:
                    evaluated_deviation_count += 1
                    maybe = certify_candidate(
                        record=record,
                        candidate=candidate,
                        config=config,
                        evaluator=evaluator,
                    )
                    if maybe is None:
                        continue
                    certified.append(maybe)
                    _write_fragile_state(handle, maybe)

    return RefutationMiningReport(
        config=config,
        source_record_count=len(source_records),
        sampled_win_count=len(sampled_wins),
        scanned_decision_count=scanned_decision_count,
        candidate_deviation_count=candidate_deviation_count,
        evaluated_deviation_count=evaluated_deviation_count,
        certified_refutations=tuple(certified),
        archive_path=archive_path,
    )


def certify_candidate(
    *,
    record: RolloutRecord,
    candidate: RefutationCandidate,
    config: RefutationMiningConfig,
    evaluator: TerminalBranchEvaluator,
) -> CertifiedRefutation | None:
    terminal_results = tuple(
        evaluator.evaluate(
            record=record,
            candidate=candidate,
            certification_seed=record.seed + seed_offset + 1,
        )
        for seed_offset in range(config.certification_seed_count)
    )
    deviation_wins = sum(1 for result in terminal_results if result.winner == candidate.loser_player_id)
    champion_wins = sum(1 for result in terminal_results if result.winner == candidate.champion_player_id)
    ties_or_caps = len(terminal_results) - deviation_wins - champion_wins
    flip_rate = deviation_wins / len(terminal_results) if terminal_results else 0.0
    if flip_rate <= config.min_flip_rate:
        return None
    return CertifiedRefutation(
        candidate=candidate,
        evaluation_source=evaluator.evaluation_source,
        mode=config.mode,
        certification_seed_count=config.certification_seed_count,
        min_flip_rate=config.min_flip_rate,
        deviation_wins=deviation_wins,
        champion_wins=champion_wins,
        ties_or_caps=ties_or_caps,
        flip_rate=flip_rate,
        terminal_results=terminal_results,
        search_stats={
            "search_method": "enumerate_single_turn_legal_deviations",
            "depth": 1,
            "value_head_used": False,
            "reseed_scope": str(getattr(evaluator, "reseed_scope", "terminal_rollout")),
        },
    )


def candidate_count_for_records(
    *,
    records: Iterable[RolloutRecord],
    config: RefutationMiningConfig,
) -> dict[str, int]:
    """Cheap planning helper for report preflights."""

    source_records = tuple(records)
    win_count = 0
    decisions = 0
    deviations = 0
    for _, record, champion_player_id, loser_player_id in _iter_champion_wins(source_records, config=config):
        win_count += 1
        decision_steps = _loser_decision_steps(record.trajectory, loser_player_id)
        if config.max_decision_points_per_game is not None:
            decision_steps = decision_steps[: config.max_decision_points_per_game]
        decisions += len(decision_steps)
        for step_index, step in decision_steps:
            deviations += len(
                _deviation_candidates(
                    record=record,
                    source_record_index=0,
                    champion_player_id=champion_player_id,
                    loser_player_id=loser_player_id,
                    step_index=step_index,
                    step=step,
                    max_deviations=config.max_deviations_per_state,
                )
            )
    return {
        "source_record_count": len(source_records),
        "sampled_win_count": win_count,
        "scanned_decision_count": decisions,
        "candidate_deviation_count": deviations,
    }


def write_refutation_report(path: Path, report: RefutationMiningReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def iter_fragile_states(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _iter_champion_wins(
    records: Sequence[RolloutRecord],
    *,
    config: RefutationMiningConfig,
) -> Iterator[tuple[int, RolloutRecord, str, str]]:
    emitted = 0
    for record_index, record in enumerate(records):
        terminal = record.terminal
        if terminal.capped or terminal.winner is None:
            continue
        champion_player_id = _champion_player_id(record, config=config)
        if champion_player_id is None or terminal.winner != champion_player_id:
            continue
        loser_player_id = _single_loser_player(record.trajectory, champion_player_id)
        if loser_player_id is None:
            continue
        yield record_index, record, champion_player_id, loser_player_id
        emitted += 1
        if emitted >= config.max_wins:
            return


def _champion_player_id(record: RolloutRecord, *, config: RefutationMiningConfig) -> str | None:
    if config.champion_player_id is not None:
        return config.champion_player_id if record.policy_ids.get(config.champion_player_id) is not None else None
    assert config.champion_policy_id is not None
    winners = [
        player_id
        for player_id, policy_id in record.policy_ids.items()
        if policy_id == config.champion_policy_id
    ]
    if not winners:
        return None
    winner = record.terminal.winner
    if winner in winners:
        return winner
    return None


def _single_loser_player(trajectory: BattleTrajectory, champion_player_id: str) -> str | None:
    players = tuple(player for player in trajectory.players() if player != champion_player_id)
    return players[0] if len(players) == 1 else None


def _loser_decision_steps(
    trajectory: BattleTrajectory,
    loser_player_id: str,
) -> tuple[tuple[int, TrajectoryStep], ...]:
    return tuple(
        (step_index, step)
        for step_index, step in enumerate(trajectory.steps)
        if step.player_id == loser_player_id
    )


def _deviation_candidates(
    *,
    record: RolloutRecord,
    source_record_index: int,
    champion_player_id: str,
    loser_player_id: str,
    step_index: int,
    step: TrajectoryStep,
    max_deviations: int | None,
) -> tuple[RefutationCandidate, ...]:
    rounds = action_rounds_from_trajectory(
        record.trajectory,
        decision_round_count=step.turn_index + 1,
    )
    recorded_round = rounds[step.turn_index]
    legal = tuple(action for action in legal_action_indices(step.legal_action_mask) if action != step.action_index)
    if max_deviations is not None:
        legal = legal[:max_deviations]
    candidates: list[RefutationCandidate] = []
    for deviation_action_index in legal:
        branch_actions = dict(recorded_round.actions)
        branch_actions[loser_player_id] = deviation_action_index
        candidates.append(
            RefutationCandidate(
                battle_id=record.battle_id,
                source_record_index=source_record_index,
                seed=record.seed,
                format_id=record.format_id,
                champion_player_id=champion_player_id,
                loser_player_id=loser_player_id,
                decision_round_index=step.turn_index,
                step_index=step_index,
                recorded_action_index=step.action_index,
                deviation_action_index=deviation_action_index,
                branch_actions=branch_actions,
            )
        )
    return tuple(candidates)


def _write_fragile_state(handle: TextIO, refutation: CertifiedRefutation) -> None:
    handle.write(json.dumps(refutation.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False))
    handle.write("\n")
    handle.flush()
