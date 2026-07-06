"""G4 refutation-mining primitives.

The miner post-mortems games a champion won, searches the loser's legal
bounded-depth deviations, and certifies a deviation only from terminal rollout
outcomes. It deliberately has no value-head dependency: callers inject an
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


FRAGILE_STATE_SCHEMA_VERSION = "pokezero.fragile_state.v2"
REFUTATION_REPORT_SCHEMA_VERSION = "pokezero.refutation_report.v1"
TERMINAL_ROLLOUT_EVALUATION_SOURCE = "terminal_rollout"
DEFAULT_R0_MIN_SAMPLED_WINS = 200
DEFAULT_R0_MIN_CERTIFIED_REFUTATIONS = 10


class InfeasibleRefutationLineError(ValueError):
    """A bounded-depth recorded continuation line is impossible after branching."""


@dataclass(frozen=True)
class RefutationMiningConfig:
    """Configuration for R0 bounded-depth refutation mining."""

    champion_policy_id: str | None = None
    champion_player_id: str | None = None
    max_wins: int = 200
    max_decision_points_per_game: int | None = None
    max_deviations_per_state: int | None = None
    max_line_depth: int = 1
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
        if self.max_line_depth <= 0 or self.max_line_depth > 3:
            raise ValueError("max_line_depth must be between 1 and 3.")
        if self.certification_seed_count < 20:
            raise ValueError("certification_seed_count must be at least 20.")
        if not 0.0 < self.min_flip_rate < 1.0:
            raise ValueError("min_flip_rate must be between 0 and 1.")
        if self.mode not in {"oracle", "fair"}:
            raise ValueError("mode must be 'oracle' or 'fair'.")


@dataclass(frozen=True)
class RefutationCandidate:
    """A loser-seat deviation line at a recorded decision point."""

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
    branch_action_sequence: tuple[Mapping[str, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name, action_index in (
            ("recorded_action_index", self.recorded_action_index),
            ("deviation_action_index", self.deviation_action_index),
        ):
            if action_index < 0 or action_index >= ACTION_COUNT:
                raise ValueError(f"{name} must be between 0 and {ACTION_COUNT - 1}.")
        if self.recorded_action_index == self.deviation_action_index:
            raise ValueError("deviation_action_index must differ from recorded_action_index.")
        branch_actions = _normalize_action_map(self.branch_actions)
        sequence = self.branch_action_sequence or (branch_actions,)
        normalized_sequence = tuple(_normalize_action_map(round_actions) for round_actions in sequence)
        if not normalized_sequence:
            raise ValueError("branch_action_sequence must be non-empty.")
        if len(normalized_sequence) > 3:
            raise ValueError("branch_action_sequence cannot exceed 3 rounds.")
        if normalized_sequence[0] != branch_actions:
            raise ValueError("branch_action_sequence first round must match branch_actions.")
        object.__setattr__(self, "branch_actions", branch_actions)
        object.__setattr__(self, "branch_action_sequence", normalized_sequence)

    @property
    def line_depth(self) -> int:
        return len(self.branch_action_sequence)

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
            "branch_action_sequence": [
                dict(sorted(round_actions.items()))
                for round_actions in self.branch_action_sequence
            ],
            "line_depth": self.line_depth,
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
    value_head_used: bool
    reseed_scope: str

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
    value_head_used: bool = False
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
            step_result = branch.step_result
            forced_round_count = 1
            for offset, branch_actions in enumerate(candidate.branch_action_sequence[1:], start=1):
                terminal = step_result.terminal or env.terminal()
                if terminal is not None:
                    return BranchTerminalResult.from_terminal(
                        certification_seed=certification_seed,
                        terminal=terminal,
                    )
                _require_branch_requested_players(
                    branch_actions,
                    requested_players=env.requested_players(),
                    turn_index=candidate.decision_round_index + offset,
                )
                try:
                    step_result = env.step(branch_actions)
                except RuntimeError as exc:
                    raise InfeasibleRefutationLineError(
                        f"branch action sequence round {candidate.decision_round_index + offset} "
                        f"failed during forced continuation: {exc}"
                    ) from exc
                forced_round_count += 1
            terminal = step_result.terminal or env.terminal()
            if terminal is not None:
                return BranchTerminalResult.from_terminal(
                    certification_seed=certification_seed,
                    terminal=terminal,
                )
            continuation = continue_rollout_from_current_state(
                env=env,
                policies=self.policies,
                config=self.rollout_config,
                seed=certification_seed,
                battle_id=(
                    f"refutation-{record.battle_id}-{candidate.decision_round_index}-"
                    f"d{candidate.line_depth}-{certification_seed}"
                ),
                starting_decision_round_index=candidate.decision_round_index + forced_round_count,
                available_observations=step_result.observations,
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
    reseed_scope: str
    simulator_rng_reseeded: bool
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
                "reseed_scope": self.reseed_scope,
                "simulator_rng_reseeded": self.simulator_rng_reseeded,
                "limitation": (
                    "certification varied continuation policy RNG only; true simulator-RNG "
                    "reseeding requires a snapshot/restore backend"
                    if not self.simulator_rng_reseeded
                    else None
                ),
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
    skipped_candidate_error_count: int
    candidate_error_examples: tuple[Mapping[str, Any], ...]
    certified_refutations: tuple[CertifiedRefutation, ...]
    archive_path: Path

    @property
    def refutation_rate(self) -> float:
        return self.refuted_game_count / self.sampled_win_count if self.sampled_win_count else 0.0

    @property
    def refuted_game_count(self) -> int:
        return len(
            {
                refutation.candidate.source_record_index
                for refutation in self.certified_refutations
            }
        )

    @property
    def certified_refutations_per_sampled_win(self) -> float:
        return len(self.certified_refutations) / self.sampled_win_count if self.sampled_win_count else 0.0

    @property
    def distinct_certified_root_count(self) -> int:
        return len(
            {
                (
                    refutation.candidate.source_record_index,
                    refutation.candidate.decision_round_index,
                    refutation.candidate.step_index,
                    refutation.candidate.deviation_action_index,
                )
                for refutation in self.certified_refutations
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REFUTATION_REPORT_SCHEMA_VERSION,
            "config": {
                "champion_policy_id": self.config.champion_policy_id,
                "champion_player_id": self.config.champion_player_id,
                "max_wins": self.config.max_wins,
                "max_decision_points_per_game": self.config.max_decision_points_per_game,
                "max_deviations_per_state": self.config.max_deviations_per_state,
                "max_line_depth": self.config.max_line_depth,
                "certification_seed_count": self.config.certification_seed_count,
                "min_flip_rate": self.config.min_flip_rate,
                "mode": self.config.mode,
            },
            "source_record_count": self.source_record_count,
            "sampled_win_count": self.sampled_win_count,
            "scanned_decision_count": self.scanned_decision_count,
            "candidate_deviation_count": self.candidate_deviation_count,
            "evaluated_deviation_count": self.evaluated_deviation_count,
            "skipped_candidate_error_count": self.skipped_candidate_error_count,
            "candidate_error_examples": [dict(example) for example in self.candidate_error_examples[:10]],
            "certified_refutation_count": len(self.certified_refutations),
            "distinct_certified_root_count": self.distinct_certified_root_count,
            "refuted_game_count": self.refuted_game_count,
            "refutation_rate": self.refutation_rate,
            "certified_refutations_per_sampled_win": self.certified_refutations_per_sampled_win,
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
    skipped_candidate_error_count = 0
    candidate_error_examples: list[Mapping[str, Any]] = []
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
                    max_line_depth=config.max_line_depth,
                )
                candidate_deviation_count += len(candidates)
                for candidate in candidates:
                    evaluated_deviation_count += 1
                    try:
                        maybe = certify_candidate(
                            record=record,
                            candidate=candidate,
                            config=config,
                            evaluator=evaluator,
                        )
                    except InfeasibleRefutationLineError as exc:
                        skipped_candidate_error_count += 1
                        if len(candidate_error_examples) < 10:
                            candidate_error_examples.append(
                                {
                                    "candidate": candidate.to_dict(),
                                    "error": str(exc),
                                }
                            )
                        continue
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
        skipped_candidate_error_count=skipped_candidate_error_count,
        candidate_error_examples=tuple(candidate_error_examples),
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
    if evaluator.evaluation_source != TERMINAL_ROLLOUT_EVALUATION_SOURCE:
        raise ValueError(
            "refutation certification requires terminal-rollout evaluation; "
            f"got {evaluator.evaluation_source!r}."
        )
    if evaluator.value_head_used:
        raise ValueError("refutation certification cannot use a value-head-backed evaluator.")
    reseed_scope = evaluator.reseed_scope
    simulator_rng_reseeded = reseed_scope == "simulator_rng"
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
        reseed_scope=reseed_scope,
        simulator_rng_reseeded=simulator_rng_reseeded,
        search_stats={
            "search_method": "enumerate_recorded_continuation_deviation_lines",
            "depth": candidate.line_depth,
            "max_line_depth": config.max_line_depth,
            "value_head_used": evaluator.value_head_used,
            "reseed_scope": reseed_scope,
            "simulator_rng_reseeded": simulator_rng_reseeded,
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
                    max_line_depth=config.max_line_depth,
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


def validate_refutation_report_payload(
    *,
    report: Mapping[str, Any],
    fragile_states: Iterable[Mapping[str, Any]],
    min_sampled_wins: int = DEFAULT_R0_MIN_SAMPLED_WINS,
    min_certified_refutations: int = DEFAULT_R0_MIN_CERTIFIED_REFUTATIONS,
    min_certification_seed_count: int = 20,
) -> dict[str, Any]:
    """Validate an R0 refutation report/archive against the artifact-level gate."""

    if min_sampled_wins <= 0:
        raise ValueError("min_sampled_wins must be positive.")
    if min_certified_refutations <= 0:
        raise ValueError("min_certified_refutations must be positive.")
    if min_certification_seed_count < 20:
        raise ValueError("min_certification_seed_count must be at least 20.")

    rows = tuple(dict(row) for row in fragile_states)
    checks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, *, observed: Any = None, threshold: Any = None, message: str) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "threshold": threshold,
                "message": message,
            }
        )

    add_check(
        "report_schema",
        report.get("schema_version") == REFUTATION_REPORT_SCHEMA_VERSION,
        observed=report.get("schema_version"),
        threshold=REFUTATION_REPORT_SCHEMA_VERSION,
        message="report schema version is supported",
    )
    sampled_win_count = _int_value(report.get("sampled_win_count"))
    add_check(
        "sampled_win_count",
        sampled_win_count is not None and sampled_win_count >= min_sampled_wins,
        observed=sampled_win_count,
        threshold=min_sampled_wins,
        message="report mined enough sampled champion wins",
    )
    reported_certified_count = _int_value(report.get("certified_refutation_count"))
    reported_distinct_root_count = _int_value(report.get("distinct_certified_root_count"))
    certified_gate_count = (
        reported_distinct_root_count
        if reported_distinct_root_count is not None
        else reported_certified_count
    )
    add_check(
        "distinct_certified_root_count",
        certified_gate_count is not None and certified_gate_count >= min_certified_refutations,
        observed=certified_gate_count,
        threshold=min_certified_refutations,
        message="report contains enough distinct certified refutation roots",
    )
    add_check(
        "archive_count_matches_report",
        reported_certified_count is not None and len(rows) == reported_certified_count,
        observed=len(rows),
        threshold=reported_certified_count,
        message="fragile-state archive row count matches the report",
    )

    terminal_rollout_rows = 0
    no_value_head_rows = 0
    seed_count_rows = 0
    distinct_seed_rows = 0
    certification_consistency_rows = 0
    flip_rate_rows = 0
    replay_coordinate_rows = 0
    terminal_result_rows = 0
    simulator_rng_reseeded_rows = 0
    reseed_scope_counts: dict[str, int] = {}
    for row in rows:
        certification = _mapping_or_empty(row.get("certification"))
        candidate = _mapping_or_empty(row.get("candidate"))
        search_stats = _mapping_or_empty(row.get("search_stats"))
        seed_count = _int_value(certification.get("seed_count"))
        min_flip_rate = _float_value(certification.get("min_flip_rate"))
        terminal_results = row.get("terminal_results")
        result_rows = tuple(_mapping_or_empty(result) for result in terminal_results) if isinstance(terminal_results, list) else ()
        result_seeds = tuple(_int_value(result.get("certification_seed")) for result in result_rows)
        champion_player_id = candidate.get("champion_player_id")
        loser_player_id = candidate.get("loser_player_id")
        computed_deviation_wins = sum(1 for result in result_rows if result.get("winner") == loser_player_id)
        computed_champion_wins = sum(1 for result in result_rows if result.get("winner") == champion_player_id)
        computed_ties_or_caps = len(result_rows) - computed_deviation_wins - computed_champion_wins
        computed_flip_rate = computed_deviation_wins / len(result_rows) if result_rows else None
        reseed_scope = certification.get("reseed_scope") or search_stats.get("reseed_scope") or "unknown"
        reseed_scope_counts[str(reseed_scope)] = reseed_scope_counts.get(str(reseed_scope), 0) + 1

        if row.get("schema_version") == FRAGILE_STATE_SCHEMA_VERSION and row.get("evaluation_source") == TERMINAL_ROLLOUT_EVALUATION_SOURCE:
            terminal_rollout_rows += 1
        if search_stats.get("value_head_used") is False:
            no_value_head_rows += 1
        if seed_count is not None and seed_count >= min_certification_seed_count:
            seed_count_rows += 1
        if seed_count is not None and len(result_seeds) == seed_count and None not in result_seeds and len(set(result_seeds)) == seed_count:
            distinct_seed_rows += 1
        if (
            seed_count is not None
            and computed_flip_rate is not None
            and _int_value(certification.get("deviation_wins")) == computed_deviation_wins
            and _int_value(certification.get("champion_wins")) == computed_champion_wins
            and _int_value(certification.get("ties_or_caps")) == computed_ties_or_caps
            and _float_equal(_float_value(certification.get("flip_rate")), computed_flip_rate)
            and len(result_rows) == seed_count
        ):
            certification_consistency_rows += 1
        if (
            computed_flip_rate is not None
            and min_flip_rate is not None
            and computed_flip_rate > min_flip_rate
            and certification.get("passed") is True
        ):
            flip_rate_rows += 1
        if _candidate_has_replay_coordinates(candidate):
            replay_coordinate_rows += 1
        if isinstance(terminal_results, list) and seed_count is not None and len(result_rows) == seed_count:
            terminal_result_rows += 1
        if certification.get("simulator_rng_reseeded") is True:
            simulator_rng_reseeded_rows += 1

    add_check(
        "terminal_rollout_rows",
        terminal_rollout_rows == len(rows),
        observed=terminal_rollout_rows,
        threshold=len(rows),
        message="all archived refutations use terminal-rollout evaluation",
    )
    add_check(
        "no_value_head_rows",
        no_value_head_rows == len(rows),
        observed=no_value_head_rows,
        threshold=len(rows),
        message="all archived refutations avoid champion value-head evaluation",
    )
    add_check(
        "certification_seed_count_rows",
        seed_count_rows == len(rows),
        observed=seed_count_rows,
        threshold=len(rows),
        message="all archived refutations use enough certification seeds",
    )
    add_check(
        "distinct_certification_seed_rows",
        distinct_seed_rows == len(rows),
        observed=distinct_seed_rows,
        threshold=len(rows),
        message="all archived refutations include distinct certification seeds",
    )
    add_check(
        "certification_consistency_rows",
        certification_consistency_rows == len(rows),
        observed=certification_consistency_rows,
        threshold=len(rows),
        message="all archived certification summaries match terminal results",
    )
    add_check(
        "flip_rate_rows",
        flip_rate_rows == len(rows),
        observed=flip_rate_rows,
        threshold=len(rows),
        message="all archived refutations exceed their configured flip-rate threshold",
    )
    add_check(
        "replay_coordinate_rows",
        replay_coordinate_rows == len(rows),
        observed=replay_coordinate_rows,
        threshold=len(rows),
        message="all archived refutations carry replay coordinates needed for reproduction",
    )
    add_check(
        "terminal_result_rows",
        terminal_result_rows == len(rows),
        observed=terminal_result_rows,
        threshold=len(rows),
        message="all archived refutations include one terminal result per certification seed",
    )
    if rows and simulator_rng_reseeded_rows < len(rows):
        warnings.append(
            {
                "name": "simulator_rng_not_fully_reseeded",
                "observed": simulator_rng_reseeded_rows,
                "threshold": len(rows),
                "message": (
                    "some archived refutations vary continuation policy RNG only; true "
                    "simulator-RNG reseeding requires a snapshot/restore backend"
                ),
            }
        )

    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": "pokezero.refutation_report_validation.v1",
        "passed": passed,
        "checks": checks,
        "warnings": warnings,
        "summary": {
            "sampled_win_count": sampled_win_count,
            "certified_refutation_count": reported_certified_count,
            "distinct_certified_root_count": certified_gate_count,
            "archive_row_count": len(rows),
            "min_sampled_wins": min_sampled_wins,
            "min_certified_refutations": min_certified_refutations,
            "min_certification_seed_count": min_certification_seed_count,
            "simulator_rng_reseeded_count": simulator_rng_reseeded_rows,
            "reseed_scope_counts": dict(sorted(reseed_scope_counts.items())),
        },
    }


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
    max_line_depth: int = 1,
) -> tuple[RefutationCandidate, ...]:
    rounds = action_rounds_from_trajectory(
        record.trajectory,
        decision_round_count=step.turn_index + 1,
    )
    recorded_round = rounds[step.turn_index]
    all_rounds = action_rounds_from_trajectory(record.trajectory)
    continuation_rounds = all_rounds[step.turn_index + 1 : step.turn_index + max_line_depth]
    legal = tuple(action for action in legal_action_indices(step.legal_action_mask) if action != step.action_index)
    if max_deviations is not None:
        legal = legal[:max_deviations]
    candidates: list[RefutationCandidate] = []
    for deviation_action_index in legal:
        branch_actions = dict(recorded_round.actions)
        branch_actions[loser_player_id] = deviation_action_index
        branch_action_sequences = [(branch_actions,)]
        for depth in range(2, max_line_depth + 1):
            continuation = continuation_rounds[: depth - 1]
            if len(continuation) != depth - 1:
                break
            branch_action_sequences.append(
                (branch_actions, *tuple(dict(round_actions.actions) for round_actions in continuation))
            )
        for branch_action_sequence in branch_action_sequences:
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
                    branch_action_sequence=branch_action_sequence,
                )
            )
    return tuple(candidates)


def _write_fragile_state(handle: TextIO, refutation: CertifiedRefutation) -> None:
    handle.write(json.dumps(refutation.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False))
    handle.write("\n")
    handle.flush()


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _float_equal(left: float | None, right: float | None, *, tolerance: float = 1e-12) -> bool:
    return left is not None and right is not None and abs(left - right) <= tolerance


def _candidate_has_replay_coordinates(candidate: Mapping[str, Any]) -> bool:
    required = (
        "battle_id",
        "source_record_index",
        "seed",
        "format_id",
        "champion_player_id",
        "loser_player_id",
        "decision_round_index",
        "step_index",
        "recorded_action_index",
        "deviation_action_index",
        "branch_actions",
        "branch_action_sequence",
        "line_depth",
    )
    return (
        all(key in candidate for key in required)
        and isinstance(candidate.get("branch_actions"), Mapping)
        and isinstance(candidate.get("branch_action_sequence"), list)
        and _int_value(candidate.get("line_depth")) is not None
    )


def _normalize_action_map(actions: Mapping[str, int]) -> dict[str, int]:
    if not actions:
        raise ValueError("branch action rounds must be non-empty.")
    normalized = {
        str(player_id): int(action_index)
        for player_id, action_index in sorted(actions.items(), key=lambda item: str(item[0]))
    }
    for action_index in normalized.values():
        if action_index < 0 or action_index >= ACTION_COUNT:
            raise ValueError(f"branch action indices must be between 0 and {ACTION_COUNT - 1}.")
    return normalized


def _require_branch_requested_players(
    actions: Mapping[str, int],
    *,
    requested_players: tuple[str, ...],
    turn_index: int,
) -> None:
    requested = set(requested_players)
    supplied = set(actions)
    if supplied == requested:
        return
    raise InfeasibleRefutationLineError(
        f"branch action sequence round {turn_index} does not match environment request: "
        f"requested={sorted(requested)} supplied={sorted(supplied)}"
    )
