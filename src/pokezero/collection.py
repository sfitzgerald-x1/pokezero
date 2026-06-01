"""Rollout collection and JSONL persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, Mapping, TextIO
from urllib.parse import parse_qsl, urlencode

from .env import PokeZeroEnv, TerminalState
from .policy import Policy, RandomLegalPolicy, ScriptedTeacherPolicy, SimpleLegalPolicy
from .rollout import RolloutConfig, RolloutDriver, RolloutResult
from .trajectory import BattleTrajectory, trajectory_from_dict, trajectory_to_dict

ROLLOUT_RECORD_SCHEMA_VERSION = "pokezero.rollout_record.v1"
LINEAR_POLICY_SPEC_PREFIX = "linear:"


@dataclass(frozen=True)
class BenchmarkMatchup:
    label: str
    p1_policy: Policy
    p2_policy: Policy

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("benchmark matchup label must be non-empty.")


@dataclass(frozen=True)
class BenchmarkMatchupResult:
    label: str
    p1_policy_id: str
    p2_policy_id: str
    seed_start: int
    metrics: "CollectionMetrics"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "p1_policy_id": self.p1_policy_id,
            "p2_policy_id": self.p2_policy_id,
            "seed_start": self.seed_start,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class BenchmarkHeadToHeadResult:
    label: str
    first_policy_id: str
    second_policy_id: str
    games: int
    first_policy_wins: int
    second_policy_wins: int
    ties: int
    capped_games: int

    @property
    def first_policy_win_rate(self) -> float:
        return self.first_policy_wins / self.games if self.games else 0.0

    @property
    def second_policy_win_rate(self) -> float:
        return self.second_policy_wins / self.games if self.games else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "first_policy_id": self.first_policy_id,
            "second_policy_id": self.second_policy_id,
            "games": self.games,
            "first_policy_wins": self.first_policy_wins,
            "second_policy_wins": self.second_policy_wins,
            "ties": self.ties,
            "capped_games": self.capped_games,
            "first_policy_win_rate": self.first_policy_win_rate,
            "second_policy_win_rate": self.second_policy_win_rate,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    format_id: str
    max_decision_rounds: int
    games_per_matchup: int
    matchups: tuple[BenchmarkMatchupResult, ...]

    @property
    def total_games(self) -> int:
        return sum(result.metrics.games for result in self.matchups)

    @property
    def elapsed_seconds(self) -> float:
        return sum(result.metrics.elapsed_seconds for result in self.matchups)

    @property
    def total_decision_rounds(self) -> int:
        return sum(result.metrics.total_decision_rounds for result in self.matchups)

    @property
    def games_per_second(self) -> float:
        return self.total_games / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def decisions_per_second(self) -> float:
        return self.total_decision_rounds / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def head_to_head_results(self) -> tuple[BenchmarkHeadToHeadResult, ...]:
        return aggregate_benchmark_head_to_heads(self.matchups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_id": self.format_id,
            "max_decision_rounds": self.max_decision_rounds,
            "games_per_matchup": self.games_per_matchup,
            "total_games": self.total_games,
            "elapsed_seconds": self.elapsed_seconds,
            "games_per_second": self.games_per_second,
            "decisions_per_second": self.decisions_per_second,
            "matchups": [result.to_dict() for result in self.matchups],
            "head_to_heads": [result.to_dict() for result in self.head_to_head_results],
        }


@dataclass(frozen=True)
class RolloutRecord:
    battle_id: str
    seed: int
    format_id: str
    policy_ids: Mapping[str, str]
    decision_round_count: int
    elapsed_seconds: float
    terminal: TerminalState
    trajectory: BattleTrajectory


@dataclass(frozen=True)
class CollectionMetrics:
    games: int
    elapsed_seconds: float
    total_decision_rounds: int
    total_simulator_turns: int
    p1_wins: int
    p2_wins: int
    ties: int
    capped_games: int

    @property
    def games_per_second(self) -> float:
        return self.games / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def decisions_per_second(self) -> float:
        return self.total_decision_rounds / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def average_decision_rounds(self) -> float:
        return self.total_decision_rounds / self.games if self.games else 0.0

    @property
    def average_simulator_turns(self) -> float:
        return self.total_simulator_turns / self.games if self.games else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "elapsed_seconds": self.elapsed_seconds,
            "total_decision_rounds": self.total_decision_rounds,
            "total_simulator_turns": self.total_simulator_turns,
            "p1_wins": self.p1_wins,
            "p2_wins": self.p2_wins,
            "ties": self.ties,
            "capped_games": self.capped_games,
            "games_per_second": self.games_per_second,
            "decisions_per_second": self.decisions_per_second,
            "average_decision_rounds": self.average_decision_rounds,
            "average_simulator_turns": self.average_simulator_turns,
        }


def collect_rollouts(
    *,
    output_path: Path,
    games: int,
    env_factory: Callable[[], PokeZeroEnv],
    policies: Mapping[str, Policy],
    rollout_config: RolloutConfig,
    seed_start: int = 1,
    append: bool = False,
) -> CollectionMetrics:
    if games <= 0:
        raise ValueError("games must be positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accumulator = _MetricsAccumulator()
    collection_start = perf_counter()
    write_path = output_path if append else _temporary_output_path(output_path)
    try:
        with write_path.open("a" if append else "w", encoding="utf-8") as handle:
            for game_index in range(games):
                seed = seed_start + game_index
                record = run_rollout_record(
                    env_factory=env_factory,
                    policies=policies,
                    rollout_config=rollout_config,
                    seed=seed,
                    battle_id=f"rollout-{seed}",
                )
                accumulator.add(record)
                write_rollout_record(handle, record)
        if not append:
            write_path.replace(output_path)
    except Exception:
        if not append:
            write_path.unlink(missing_ok=True)
        raise
    elapsed = perf_counter() - collection_start
    return accumulator.to_metrics(elapsed_seconds=elapsed)


def benchmark_rollouts(
    *,
    games: int,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    seed_start: int = 1,
    matchups: Iterable[BenchmarkMatchup] | None = None,
) -> BenchmarkReport:
    if games <= 0:
        raise ValueError("games must be positive.")
    selected_matchups = tuple(matchups) if matchups is not None else default_benchmark_matchups()
    if not selected_matchups:
        raise ValueError("at least one benchmark matchup is required.")

    results: list[BenchmarkMatchupResult] = []
    for matchup in selected_matchups:
        policies = {
            "p1": matchup.p1_policy,
            "p2": matchup.p2_policy,
        }
        accumulator = _MetricsAccumulator()
        matchup_start = perf_counter()
        for game_index in range(games):
            seed = seed_start + game_index
            record = run_rollout_record(
                env_factory=env_factory,
                policies=policies,
                rollout_config=rollout_config,
                seed=seed,
                battle_id=f"benchmark-{_slugify_label(matchup.label)}-{seed}",
            )
            accumulator.add(record)
        elapsed = perf_counter() - matchup_start
        results.append(
            BenchmarkMatchupResult(
                label=matchup.label,
                p1_policy_id=matchup.p1_policy.policy_id,
                p2_policy_id=matchup.p2_policy.policy_id,
                seed_start=seed_start,
                metrics=accumulator.to_metrics(elapsed_seconds=elapsed),
            )
        )

    return BenchmarkReport(
        format_id=rollout_config.format_id,
        max_decision_rounds=rollout_config.max_decision_rounds,
        games_per_matchup=games,
        matchups=tuple(results),
    )


def default_benchmark_matchups() -> tuple[BenchmarkMatchup, ...]:
    return (
        BenchmarkMatchup("random-legal vs random-legal", RandomLegalPolicy(), RandomLegalPolicy()),
        BenchmarkMatchup("simple-legal vs random-legal", SimpleLegalPolicy(), RandomLegalPolicy()),
        BenchmarkMatchup("random-legal vs simple-legal", RandomLegalPolicy(), SimpleLegalPolicy()),
        BenchmarkMatchup("simple-legal vs simple-legal", SimpleLegalPolicy(), SimpleLegalPolicy()),
    )


def aggregate_benchmark_head_to_heads(
    matchup_results: Iterable[BenchmarkMatchupResult],
) -> tuple[BenchmarkHeadToHeadResult, ...]:
    accumulators: dict[tuple[str, str], _HeadToHeadAccumulator] = {}
    ordered_keys: list[tuple[str, str]] = []

    for result in matchup_results:
        p1_policy_id = result.p1_policy_id
        p2_policy_id = result.p2_policy_id
        if p1_policy_id == p2_policy_id:
            continue
        unordered_key = tuple(sorted((p1_policy_id, p2_policy_id)))
        accumulator = accumulators.get(unordered_key)
        if accumulator is None:
            accumulator = _HeadToHeadAccumulator(
                first_policy_id=p1_policy_id,
                second_policy_id=p2_policy_id,
            )
            accumulators[unordered_key] = accumulator
            ordered_keys.append(unordered_key)
        accumulator.add(result)

    return tuple(accumulators[key].to_result() for key in ordered_keys)


def run_rollout_record(
    *,
    env_factory: Callable[[], PokeZeroEnv],
    policies: Mapping[str, Policy],
    rollout_config: RolloutConfig,
    seed: int,
    battle_id: str,
) -> RolloutRecord:
    env = env_factory()
    start = perf_counter()
    try:
        result = RolloutDriver(env=env, policies=policies, config=rollout_config).run(seed=seed, battle_id=battle_id)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    elapsed = perf_counter() - start
    return record_from_result(result, policies=policies, elapsed_seconds=elapsed)


def record_from_result(
    result: RolloutResult,
    *,
    policies: Mapping[str, Policy],
    elapsed_seconds: float,
) -> RolloutRecord:
    return RolloutRecord(
        battle_id=result.trajectory.battle_id,
        seed=result.trajectory.seed,
        format_id=result.trajectory.format_id,
        policy_ids={player: policy.policy_id for player, policy in policies.items()},
        decision_round_count=result.decision_round_count,
        elapsed_seconds=elapsed_seconds,
        terminal=result.terminal,
        trajectory=result.trajectory,
    )


def write_rollout_record(handle: TextIO, record: RolloutRecord) -> None:
    handle.write(json.dumps(rollout_record_to_dict(record), separators=(",", ":"), sort_keys=True))
    handle.write("\n")
    handle.flush()


def read_rollout_records(path: Path) -> tuple[RolloutRecord, ...]:
    return tuple(iter_rollout_records(path))


def iter_rollout_records(path: Path) -> Iterator[RolloutRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield rollout_record_from_dict(json.loads(line))


def rollout_record_to_dict(record: RolloutRecord) -> dict[str, Any]:
    return {
        "schema_version": ROLLOUT_RECORD_SCHEMA_VERSION,
        "battle_id": record.battle_id,
        "seed": record.seed,
        "format_id": record.format_id,
        "policy_ids": dict(record.policy_ids),
        "decision_round_count": record.decision_round_count,
        "elapsed_seconds": record.elapsed_seconds,
        "terminal": _terminal_to_dict(record.terminal),
        "trajectory": trajectory_to_dict(record.trajectory),
    }


def rollout_record_from_dict(payload: Mapping[str, Any]) -> RolloutRecord:
    if payload.get("schema_version") != ROLLOUT_RECORD_SCHEMA_VERSION:
        raise ValueError(f"Unsupported rollout record schema: {payload.get('schema_version')!r}.")
    return RolloutRecord(
        battle_id=str(payload["battle_id"]),
        seed=int(payload["seed"]),
        format_id=str(payload["format_id"]),
        policy_ids={str(player): str(policy) for player, policy in _mapping(payload["policy_ids"]).items()},
        decision_round_count=int(payload["decision_round_count"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        terminal=_terminal_from_dict(_mapping(payload["terminal"])),
        trajectory=trajectory_from_dict(_mapping(payload["trajectory"])),
    )


def summarize_records(records: Iterable[RolloutRecord], *, elapsed_seconds: float) -> CollectionMetrics:
    accumulator = _MetricsAccumulator()
    for record in records:
        accumulator.add(record)
    return accumulator.to_metrics(elapsed_seconds=elapsed_seconds)


def policy_from_spec(spec: str) -> Policy:
    return policy_factory_from_spec(spec)()


def policy_factory_from_spec(spec: str) -> Callable[[], Policy]:
    normalized = spec.strip()
    policy_body, options = _split_policy_spec_options(normalized)
    lowered = policy_body.lower()
    if lowered == "random-legal":
        if options:
            raise ValueError("random-legal does not support policy spec options.")
        return RandomLegalPolicy
    if lowered == "simple-legal":
        if options:
            raise ValueError("simple-legal does not support policy spec options.")
        return SimpleLegalPolicy
    if lowered == "scripted-teacher":
        teacher_options = _scripted_teacher_options(options)
        return lambda: ScriptedTeacherPolicy(**teacher_options)
    if lowered.startswith(LINEAR_POLICY_SPEC_PREFIX):
        from .linear_policy import LinearSoftmaxPolicy, load_linear_model

        checkpoint = policy_body[len(LINEAR_POLICY_SPEC_PREFIX) :].strip()
        if not checkpoint:
            raise ValueError("linear policy spec must include a checkpoint path after 'linear:'.")
        linear_options = _linear_policy_options(options)
        model = load_linear_model(Path(checkpoint))
        return lambda: LinearSoftmaxPolicy(model=model, **linear_options)
    raise ValueError(
        f"Unsupported policy spec: {spec!r}. Expected random-legal, simple-legal, "
        "scripted-teacher, or linear:/path/to/checkpoint.json."
    )


def policy_from_name(name: str) -> Policy:
    return policy_from_spec(name)


def policy_spec_with_showdown_root(spec: str, showdown_root: Path | str | None) -> str:
    if showdown_root is None:
        return spec
    policy_body, options = _split_policy_spec_options(spec.strip())
    if policy_body.lower() != "scripted-teacher" or "showdown_root" in options:
        return spec
    options = {**options, "showdown_root": str(showdown_root)}
    return f"{policy_body}?{urlencode(options)}"


def _split_policy_spec_options(spec: str) -> tuple[str, dict[str, str]]:
    body, separator, query = spec.partition("?")
    if not separator:
        return body, {}
    options: dict[str, str] = {}
    for key, value in parse_qsl(query, keep_blank_values=True):
        normalized_key = key.strip().lower()
        if not normalized_key:
            raise ValueError("policy spec option names must be non-empty.")
        if normalized_key in options:
            raise ValueError(f"duplicate policy spec option: {normalized_key}.")
        options[normalized_key] = value.strip()
    return body, options


def _linear_policy_options(options: Mapping[str, str]) -> dict[str, object]:
    supported = {"sample", "deterministic", "epsilon", "temperature"}
    unknown = sorted(set(options) - supported)
    if unknown:
        raise ValueError(f"Unsupported linear policy option(s): {', '.join(unknown)}.")

    sample = _optional_bool(options, "sample")
    deterministic = _optional_bool(options, "deterministic")
    if sample is not None and deterministic is not None and sample == deterministic:
        raise ValueError("linear policy options 'sample' and 'deterministic' conflict.")
    deterministic_policy = deterministic if deterministic is not None else False
    if sample is not None:
        deterministic_policy = not sample

    exploration_epsilon = _optional_float(options, "epsilon", default=0.0)
    sampling_temperature = _optional_float(options, "temperature", default=1.0)
    if not 0.0 <= exploration_epsilon <= 1.0:
        raise ValueError("linear policy epsilon must be between 0 and 1.")
    if sampling_temperature <= 0.0:
        raise ValueError("linear policy temperature must be positive.")
    return {
        "deterministic": deterministic_policy,
        "exploration_epsilon": exploration_epsilon,
        "sampling_temperature": sampling_temperature,
    }


def _scripted_teacher_options(options: Mapping[str, str]) -> dict[str, object]:
    supported = {"showdown_root", "switch_margin", "poor_move_threshold"}
    unknown = sorted(set(options) - supported)
    if unknown:
        raise ValueError(f"Unsupported scripted-teacher option(s): {', '.join(unknown)}.")
    teacher_options: dict[str, object] = {}
    if options.get("showdown_root"):
        teacher_options["showdown_root"] = Path(options["showdown_root"])
    if "switch_margin" in options:
        teacher_options["switch_margin"] = _optional_float(options, "switch_margin", default=8.0)
    if "poor_move_threshold" in options:
        teacher_options["poor_move_threshold"] = _optional_float(options, "poor_move_threshold", default=35.0)
    return teacher_options


def _optional_bool(options: Mapping[str, str], key: str) -> bool | None:
    if key not in options:
        return None
    value = options[key].strip().lower()
    if value == "":
        return True
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"policy spec option {key!r} must be a boolean value.")


def _optional_float(options: Mapping[str, str], key: str, *, default: float) -> float:
    if key not in options:
        return default
    try:
        return float(options[key])
    except ValueError as exc:
        raise ValueError(f"policy spec option {key!r} must be numeric.") from exc


def _terminal_to_dict(terminal: TerminalState) -> dict[str, Any]:
    return {
        "winner": terminal.winner,
        "turn_count": terminal.turn_count,
        "capped": terminal.capped,
    }


def _terminal_from_dict(payload: Mapping[str, Any]) -> TerminalState:
    winner = payload.get("winner")
    return TerminalState(
        winner=str(winner) if winner is not None else None,
        turn_count=int(payload["turn_count"]),
        capped=bool(payload.get("capped", False)),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object payload.")
    return value


def _temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.tmp")


def _slugify_label(label: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in label.strip())
    return "-".join(part for part in slug.split("-") if part)


@dataclass
class _MetricsAccumulator:
    games: int = 0
    total_decision_rounds: int = 0
    total_simulator_turns: int = 0
    p1_wins: int = 0
    p2_wins: int = 0
    ties: int = 0
    capped_games: int = 0

    def add(self, record: RolloutRecord) -> None:
        self.games += 1
        self.total_decision_rounds += record.decision_round_count
        self.total_simulator_turns += record.terminal.turn_count
        if record.terminal.winner == "p1":
            self.p1_wins += 1
        elif record.terminal.winner == "p2":
            self.p2_wins += 1
        elif not record.terminal.capped:
            self.ties += 1
        if record.terminal.capped:
            self.capped_games += 1

    def to_metrics(self, *, elapsed_seconds: float) -> CollectionMetrics:
        return CollectionMetrics(
            games=self.games,
            elapsed_seconds=elapsed_seconds,
            total_decision_rounds=self.total_decision_rounds,
            total_simulator_turns=self.total_simulator_turns,
            p1_wins=self.p1_wins,
            p2_wins=self.p2_wins,
            ties=self.ties,
            capped_games=self.capped_games,
        )


@dataclass
class _HeadToHeadAccumulator:
    first_policy_id: str
    second_policy_id: str
    games: int = 0
    first_policy_wins: int = 0
    second_policy_wins: int = 0
    ties: int = 0
    capped_games: int = 0

    def add(self, result: BenchmarkMatchupResult) -> None:
        metrics = result.metrics
        self.games += metrics.games
        self.ties += metrics.ties
        self.capped_games += metrics.capped_games
        if result.p1_policy_id == self.first_policy_id and result.p2_policy_id == self.second_policy_id:
            self.first_policy_wins += metrics.p1_wins
            self.second_policy_wins += metrics.p2_wins
        elif result.p1_policy_id == self.second_policy_id and result.p2_policy_id == self.first_policy_id:
            self.first_policy_wins += metrics.p2_wins
            self.second_policy_wins += metrics.p1_wins
        else:
            raise ValueError("matchup result does not match head-to-head policies.")

    def to_result(self) -> BenchmarkHeadToHeadResult:
        return BenchmarkHeadToHeadResult(
            label=f"{self.first_policy_id} vs {self.second_policy_id}",
            first_policy_id=self.first_policy_id,
            second_policy_id=self.second_policy_id,
            games=self.games,
            first_policy_wins=self.first_policy_wins,
            second_policy_wins=self.second_policy_wins,
            ties=self.ties,
            capped_games=self.capped_games,
        )
