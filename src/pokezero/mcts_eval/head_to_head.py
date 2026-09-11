"""Small, fail-closed plumbing for fresh MCTS-versus-MCTS games.

This module deliberately does *not* contain another evaluation framework.  It
only fills the gap between :class:`~pokezero.rollout.RolloutDriver` (which can
already play a fresh battle with two policies) and the mirrored-pair scoring
contract.  In particular, the historic ``mcts_acceptance_h2h.py`` script is
search-versus-raw and must not be relabelled as this comparison.

An in-process run is useful for identical-policy controls and for a future
selector comparison.  It refuses two different engine source identities: a
pre-fix-versus-fixed backup comparison needs isolated native builds/processes,
not two labels attached to today's loaded extension.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence, cast

from ..policy import PolicyContext
from .scoring import (
    GameResult,
    MergeError,
    bootstrap_indices,
    bootstrap_mean,
    outcome_record,
    pair_scores,
)


GAME_SCHEMA_VERSION = "pokezero.mcts-h2h-game.v3"
_SEATS = ("p1", "p2")


class HeadToHeadError(RuntimeError):
    """A comparison is invalid, incomplete, or unsafe to resume."""


@dataclass(frozen=True)
class MctsPolicySpec:
    """The frozen identity and explicit MCTS knobs for one side of a game.

    ``config`` is intentionally data, rather than an ``EngineMctsConfig``
    instance: it is serialised into every durable game record and can therefore
    be checked before a partial pair is resumed.  The caller must include every
    non-default knob that could change the search, not merely depth and sims.
    """

    config_id: str
    policy_id: str
    source_commit: str
    source_tree_sha256: str
    engine_fingerprint: str
    checkpoint_sha256: str
    showdown_source_sha256: str
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "policy_id",
            "source_commit",
            "source_tree_sha256",
            "engine_fingerprint",
            "checkpoint_sha256",
            "showdown_source_sha256",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if not isinstance(self.config, Mapping):
            raise TypeError("config must be a mapping of explicit MCTS settings.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "policy_id": self.policy_id,
            "source_commit": self.source_commit,
            "source_tree_sha256": self.source_tree_sha256,
            "engine_fingerprint": self.engine_fingerprint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "showdown_source_sha256": self.showdown_source_sha256,
            "config": dict(self.config),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MctsPolicySpec":
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise HeadToHeadError("policy specification has no mapping-valued config.")
        return cls(
            config_id=str(payload.get("config_id", "")),
            policy_id=str(payload.get("policy_id", "")),
            source_commit=str(payload.get("source_commit", "")),
            source_tree_sha256=str(payload.get("source_tree_sha256", "")),
            engine_fingerprint=str(payload.get("engine_fingerprint", "")),
            checkpoint_sha256=str(payload.get("checkpoint_sha256", "")),
            showdown_source_sha256=str(payload.get("showdown_source_sha256", "")),
            config=dict(config),
        )

    @property
    def provenance_sha256(self) -> str:
        encoded = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def require_inprocess_compatibility(
    candidate: MctsPolicySpec, incumbent: MctsPolicySpec
) -> None:
    """Refuse a false source comparison in a single Python/native process."""

    if candidate.checkpoint_sha256 != incumbent.checkpoint_sha256:
        raise HeadToHeadError(
            "candidate and incumbent checkpoints differ; this runner requires one "
            "frozen checkpoint so a game result has a single model anchor."
        )
    if candidate.source_commit != incumbent.source_commit:
        raise HeadToHeadError(
            "candidate and incumbent source commits differ. A single process cannot load "
            "two source-bound pokezero native extensions; use an isolated-build comparison "
            "instead of relabelling the current engine as the historical incumbent."
        )
    if candidate.source_tree_sha256 != incumbent.source_tree_sha256:
        raise HeadToHeadError(
            "candidate and incumbent public source-tree identities differ in an in-process "
            "run; the loaded Python runtime has one identity."
        )
    if candidate.engine_fingerprint != incumbent.engine_fingerprint:
        raise HeadToHeadError(
            "candidate and incumbent engine fingerprints differ in an in-process run; "
            "the loaded extension has one identity, so this provenance is contradictory."
        )
    if candidate.showdown_source_sha256 != incumbent.showdown_source_sha256:
        raise HeadToHeadError(
            "candidate and incumbent Showdown source identities differ in an in-process run; "
            "one live simulator cannot support a source-different comparison."
        )


def require_isolated_build_compatibility(
    candidate: MctsPolicySpec, incumbent: MctsPolicySpec
) -> None:
    """Validate the invariants for a source-different native-engine comparison.

    The two policies intentionally may have different source receipts and engine
    fingerprints.  They must still share the model and battle engine, otherwise
    a score would confound the search implementation with a checkpoint or
    simulator change.
    """

    if candidate.checkpoint_sha256 != incumbent.checkpoint_sha256:
        raise HeadToHeadError(
            "candidate and incumbent checkpoints differ; an isolated-build comparison "
            "still requires one frozen checkpoint."
        )
    if candidate.showdown_source_sha256 != incumbent.showdown_source_sha256:
        raise HeadToHeadError(
            "candidate and incumbent Showdown source identities differ; an isolated-build "
            "comparison requires one frozen simulator."
        )


@dataclass(frozen=True)
class PublicTrajectoryStep:
    """Only the fields engine determinization needs from a historic step.

    The acting side retains its own request-local observation.  The other side
    retains only its already-executed action index, which is needed to replay
    public switch order.  Its private observation and legal mask are absent.
    """

    player_id: str
    turn_index: int
    action_index: int
    observation: object | None


@dataclass(frozen=True)
class PublicTrajectory:
    """Structural, public-only substitute for ``BattleTrajectory`` in MCTS."""

    battle_id: str
    format_id: str
    seed: int
    steps: tuple[PublicTrajectoryStep, ...]
    terminal: object | None
    metadata: Mapping[str, Any]


def public_only_context(context: PolicyContext) -> PolicyContext:
    """Strip all opponent-private request state before an MCTS decision.

    ``hide_opponent_legal_action_masks`` alone only protects the simultaneous
    request.  The stock rollout snapshot still contains both seats' historical
    observations, so passing it through would make a supposedly fresh MCTS game
    depend on information the policy does not receive in production.
    """

    player_id = context.player_id
    public_rounds = context.trajectory.metadata.get("public_resolved_action_rounds")
    metadata: dict[str, Any] = {}
    if isinstance(public_rounds, Sequence) and not isinstance(public_rounds, (str, bytes)):
        # Copy the values, not the whole original metadata mapping.  This is the
        # only trajectory metadata consumed by engine determinization.
        metadata["public_resolved_action_rounds"] = list(public_rounds)
    trajectory = PublicTrajectory(
        battle_id=context.trajectory.battle_id,
        format_id=context.trajectory.format_id,
        seed=context.trajectory.seed,
        steps=tuple(
            PublicTrajectoryStep(
                player_id=step.player_id,
                turn_index=step.turn_index,
                action_index=step.action_index,
                observation=step.observation if step.player_id == player_id else None,
            )
            for step in context.trajectory.steps
        ),
        terminal=context.trajectory.terminal,
        metadata=metadata,
    )
    own_mask = context.requested_legal_action_masks.get(
        player_id, tuple(context.observation.legal_action_mask)
    )
    return replace(
        context,
        trajectory=cast(Any, trajectory),
        requested_legal_action_masks={player_id: tuple(own_mask)},
        requested_observations={player_id: context.observation},
    )


class PublicOnlyMctsPolicy:
    """A narrow context-sanitising wrapper for a context-aware MCTS policy."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy
        self.policy_id = str(policy.policy_id)
        self.requires_public_materialization_state = bool(
            getattr(policy, "requires_public_materialization_state", False)
        )

    @property
    def stats(self) -> Any:
        return self._policy.stats

    @property
    def is_source_isolated(self) -> bool:
        """Whether decisions are made by a separately source-bound process."""

        return bool(getattr(self._policy, "is_source_isolated", False))

    def select_action(self, observation: Any, *, rng: Any) -> Any:
        return self._policy.select_action(observation, rng=rng)

    def select_action_with_context(self, context: PolicyContext, *, rng: Any) -> Any:
        selector = getattr(self._policy, "select_action_with_context", None)
        if not callable(selector):
            raise HeadToHeadError(
                f"{self.policy_id} has no context-aware selector; refusing to degrade MCTS to "
                "the context-free path."
            )
        return selector(public_only_context(context), rng=rng)

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def close(self) -> None:
        close = getattr(self._policy, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class PolicyTelemetry:
    """Counters accumulated by one policy over one completed game."""

    decisions: int = 0
    searched_decisions: int = 0
    fallback_decisions: int = 0
    model_evals: int = 0
    total_iterations: int = 0
    worlds_constructed: int = 0
    worlds_searched: int = 0
    prior_fallbacks: int = 0
    root_prior_fallbacks: int = 0
    branch_prior_fallbacks: int = 0
    opponent_prior_arm_decisions: int = 0
    decision_wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        counters = (
            self.decisions,
            self.searched_decisions,
            self.fallback_decisions,
            self.model_evals,
            self.total_iterations,
            self.worlds_constructed,
            self.worlds_searched,
            self.prior_fallbacks,
            self.root_prior_fallbacks,
            self.branch_prior_fallbacks,
            self.opponent_prior_arm_decisions,
        )
        if any(value < 0 for value in counters):
            raise ValueError("policy telemetry counters must be non-negative.")
        if self.prior_fallbacks != self.root_prior_fallbacks + self.branch_prior_fallbacks:
            raise ValueError(
                "policy prior fallback aggregate must equal root plus branch fallbacks."
            )
        if not math.isfinite(self.decision_wall_seconds) or self.decision_wall_seconds < 0:
            raise ValueError("policy decision wall time must be finite and non-negative.")

    @classmethod
    def capture(cls, policy: Any) -> "PolicyTelemetry":
        stats = getattr(policy, "stats", None)
        if stats is None:
            raise HeadToHeadError("MCTS policy exposes no stats; required work counters are absent.")
        prior_fallbacks = int(getattr(stats, "prior_fallbacks", 0))
        root_prior_fallbacks = getattr(stats, "root_prior_fallbacks", None)
        branch_prior_fallbacks = getattr(stats, "branch_prior_fallbacks", None)
        if root_prior_fallbacks is None and branch_prior_fallbacks is None:
            # A caller that cannot distinguish scope cannot establish that its
            # live root action was clean. Preserve its total as a conservative
            # root failure instead of silently making it eligible.
            root_prior_fallbacks = prior_fallbacks
            branch_prior_fallbacks = 0
        elif root_prior_fallbacks is None or branch_prior_fallbacks is None:
            raise HeadToHeadError(
                "MCTS policy exposes incomplete root/branch prior fallback telemetry."
            )
        if int(root_prior_fallbacks) + int(branch_prior_fallbacks) != prior_fallbacks:
            raise HeadToHeadError(
                "MCTS policy reports a prior fallback aggregate that does not equal root plus "
                "branch."
            )
        return cls(
            decisions=int(getattr(stats, "decisions", 0)),
            searched_decisions=int(getattr(stats, "searched_decisions", 0)),
            fallback_decisions=int(getattr(stats, "fallback_decisions", 0)),
            model_evals=int(getattr(stats, "model_evals", 0)),
            total_iterations=int(getattr(stats, "total_iterations", 0)),
            worlds_constructed=int(getattr(stats, "worlds_constructed", 0)),
            worlds_searched=int(getattr(stats, "worlds_searched", 0)),
            prior_fallbacks=prior_fallbacks,
            root_prior_fallbacks=int(root_prior_fallbacks),
            branch_prior_fallbacks=int(branch_prior_fallbacks),
            # Historical, non-isolated records may predate this observability
            # counter. The source-isolated transport itself requires it, while
            # durable readers retain backwards compatibility for old artifacts.
            opponent_prior_arm_decisions=int(
                getattr(stats, "opponent_prior_arm_decisions", 0)
            ),
            decision_wall_seconds=float(getattr(stats, "decision_wall_seconds", 0.0)),
        )

    def delta(self, before: "PolicyTelemetry") -> "PolicyTelemetry":
        values = {
            field: getattr(self, field) - getattr(before, field)
            for field in self.__dataclass_fields__
        }
        if any(value < 0 for value in values.values()):
            raise HeadToHeadError(
                "policy telemetry regressed during a game; counters must be monotonic to "
                "report realized work."
            )
        return PolicyTelemetry(**values)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeadToHeadGame:
    """One durable game, always scored from the candidate policy's seat."""

    seed: int
    candidate_seat: str
    candidate: MctsPolicySpec
    incumbent: MctsPolicySpec
    result: GameResult
    terminal_winner: str | None
    terminal_capped: bool
    terminal_turn_count: int
    candidate_telemetry: PolicyTelemetry
    incumbent_telemetry: PolicyTelemetry
    incumbent_decision_walls_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.candidate_seat not in _SEATS:
            raise ValueError("candidate_seat must be p1 or p2.")
        if self.result.seed != self.seed or self.result.seat != self.candidate_seat:
            raise ValueError("GameResult must be keyed by this game's seed and candidate seat.")
        if self.result.config_id != self.candidate.config_id:
            raise ValueError("GameResult config_id must be the candidate config id.")
        if self.result.provenance_sha256 != self.candidate.provenance_sha256:
            raise ValueError("GameResult provenance must bind the candidate specification.")
        if self.terminal_winner not in {None, "p1", "p2"}:
            raise ValueError("terminal winner must be p1, p2, or None.")
        if not isinstance(self.terminal_capped, bool):
            raise ValueError("terminal capped must be a JSON boolean.")
        if self.terminal_turn_count < 0:
            raise ValueError("terminal turn count must be non-negative.")
        if self.result.turns != self.terminal_turn_count:
            raise ValueError("GameResult turns must match the recorded terminal turn count.")
        expected_outcome = outcome_for_candidate(
            winner=self.terminal_winner,
            candidate_seat=self.candidate_seat,
            capped=self.terminal_capped,
        )
        if self.result.outcome != expected_outcome:
            raise ValueError(
                "GameResult outcome must agree with terminal winner/cap and the candidate seat."
            )
        if (
            self.candidate_telemetry.fallback_decisions
            or self.incumbent_telemetry.fallback_decisions
        ):
            raise ValueError("a scored MCTS-vs-MCTS game cannot contain fallback decisions.")
        if self.result.opponent_crashed:
            raise ValueError("a scored MCTS-vs-MCTS game cannot accept an opponent crash.")
        if any(
            not math.isfinite(float(value)) or float(value) < 0
            for value in self.result.decision_walls_s
        ):
            raise ValueError("GameResult decision wall times must be finite and non-negative.")
        if any(
            not math.isfinite(float(value)) or float(value) < 0
            for value in self.incumbent_decision_walls_s
        ):
            raise ValueError("incumbent decision wall times must be finite and non-negative.")
        if any(not isinstance(value, str) for value in self.result.chosen_actions):
            raise ValueError("GameResult chosen actions must be strings.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GAME_SCHEMA_VERSION,
            "seed": self.seed,
            "candidate_seat": self.candidate_seat,
            "candidate": self.candidate.to_payload(),
            "incumbent": self.incumbent.to_payload(),
            "result": asdict(self.result),
            "terminal": {
                "winner": self.terminal_winner,
                "capped": self.terminal_capped,
                "turn_count": self.terminal_turn_count,
            },
            "candidate_telemetry": self.candidate_telemetry.to_payload(),
            "incumbent_telemetry": self.incumbent_telemetry.to_payload(),
            "incumbent_decision_walls_s": list(self.incumbent_decision_walls_s),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HeadToHeadGame":
        if payload.get("schema_version") != GAME_SCHEMA_VERSION:
            raise HeadToHeadError(
                f"game schema {payload.get('schema_version')!r} is not {GAME_SCHEMA_VERSION!r}."
            )
        candidate_raw = payload.get("candidate")
        incumbent_raw = payload.get("incumbent")
        result_raw = payload.get("result")
        terminal_raw = payload.get("terminal")
        candidate_metrics = payload.get("candidate_telemetry")
        incumbent_metrics = payload.get("incumbent_telemetry")
        incumbent_walls = payload.get("incumbent_decision_walls_s")
        if not all(
            isinstance(value, Mapping)
            for value in (
                candidate_raw,
                incumbent_raw,
                result_raw,
                terminal_raw,
                candidate_metrics,
                incumbent_metrics,
            )
        ):
            raise HeadToHeadError("game record is missing a required mapping payload.")
        if not isinstance(incumbent_walls, list):
            raise HeadToHeadError(
                "game record is missing incumbent_decision_walls_s as a JSON list."
            )
        try:
            return cls(
                seed=int(payload["seed"]),
                candidate_seat=str(payload["candidate_seat"]),
                candidate=MctsPolicySpec.from_payload(candidate_raw),
                incumbent=MctsPolicySpec.from_payload(incumbent_raw),
                result=GameResult(**dict(result_raw)),
                terminal_winner=(
                    str(terminal_raw["winner"])
                    if terminal_raw.get("winner") is not None
                    else None
                ),
                terminal_capped=terminal_raw["capped"],
                terminal_turn_count=int(terminal_raw["turn_count"]),
                candidate_telemetry=PolicyTelemetry(**dict(candidate_metrics)),
                incumbent_telemetry=PolicyTelemetry(**dict(incumbent_metrics)),
                incumbent_decision_walls_s=tuple(float(value) for value in incumbent_walls),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HeadToHeadError(f"invalid game record: {error}") from error


def outcome_for_candidate(*, winner: str | None, candidate_seat: str, capped: bool) -> str:
    if capped:
        return "cap"
    if winner is None:
        return "tie"
    return "win" if winner == candidate_seat else "loss"


def _policy_decision_walls(result: Any, player_id: str) -> tuple[float, ...]:
    values: list[float] = []
    for step in result.trajectory.steps:
        if step.player_id != player_id:
            continue
        elapsed = step.metadata.get("policy_elapsed_seconds")
        if elapsed is not None:
            values.append(float(elapsed))
    return tuple(values)


def _candidate_actions(result: Any, candidate_seat: str) -> tuple[str, ...]:
    return tuple(
        str(step.action_index)
        for step in result.trajectory.steps
        if step.player_id == candidate_seat
    )


def _decision_wall_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return comparable latency tails without discarding the raw samples.

    A game can terminate before one side makes a decision, so an empty series is
    a legitimate diagnostic state rather than an error. Its tail values remain
    explicit ``None`` instead of being silently presented as zero-latency work.
    """

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "total_s": 0.0,
            "min_s": None,
            "p50_s": None,
            "p95_s": None,
            "max_s": None,
        }

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "count": len(ordered),
        "total_s": sum(ordered),
        "min_s": ordered[0],
        "p50_s": quantile(0.50),
        "p95_s": quantile(0.95),
        "max_s": ordered[-1],
    }


DriverFactory = Callable[[int, str, Any, Any], Any]
PolicyFactory = Callable[[], Any]
SessionFactory = Callable[[int, str], tuple[Any, Any, Any]]
GameSink = Callable[["HeadToHeadGame"], None]


def play_mirrored_pair(
    *,
    seed: int,
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
    candidate_factory: PolicyFactory,
    incumbent_factory: PolicyFactory,
    driver_factory: DriverFactory,
    completed: Mapping[str, HeadToHeadGame] | None = None,
    session_factory: SessionFactory | None = None,
    on_game: GameSink | None = None,
    execution_mode: str = "in_process",
) -> tuple[HeadToHeadGame, ...]:
    """Play (or resume) the two candidate-seat orientations for one battle seed.

    A completed exact seat is reused; a partial pair is not scored by this
    function.  The caller persists each returned game immediately, then uses
    :func:`complete_pair` before any aggregate is read.
    """

    if execution_mode == "in_process":
        require_inprocess_compatibility(candidate, incumbent)
    elif execution_mode == "isolated_build":
        if session_factory is None:
            raise HeadToHeadError(
                "isolated-build comparisons require a session_factory that installs the "
                "source-isolated policy in the live battle driver."
            )
        require_isolated_build_compatibility(candidate, incumbent)
    else:
        raise HeadToHeadError(
            f"unknown execution_mode {execution_mode!r}; expected 'in_process' or "
            "'isolated_build'."
        )
    reusable = dict(completed or {})
    unknown_seats = set(reusable).difference(_SEATS)
    if unknown_seats:
        raise HeadToHeadError(f"completed games contain invalid seats: {sorted(unknown_seats)}")
    output: list[HeadToHeadGame] = []
    for candidate_seat in _SEATS:
        incumbent_seat = "p2" if candidate_seat == "p1" else "p1"
        existing = reusable.get(candidate_seat)
        if existing is not None:
            _assert_game_identity(existing, seed=seed, candidate=candidate, incumbent=incumbent)
            output.append(existing)
            continue
        if session_factory is None:
            candidate_policy = PublicOnlyMctsPolicy(candidate_factory())
            incumbent_policy = PublicOnlyMctsPolicy(incumbent_factory())
            driver = driver_factory(seed, candidate_seat, candidate_policy, incumbent_policy)
        else:
            # LocalShowdownEnv owns the live public-materialization state, and
            # ``EnvTier2AnnotationSource`` is bound to that env.  The real
            # runner consequently needs to build env + both policies together;
            # building a policy first and attaching a later env would make its
            # fold annotations refer to a different battle.
            driver, candidate_policy, incumbent_policy = session_factory(seed, candidate_seat)
            if not isinstance(candidate_policy, PublicOnlyMctsPolicy) or not isinstance(
                incumbent_policy, PublicOnlyMctsPolicy
            ):
                raise HeadToHeadError(
                    "session_factory must install PublicOnlyMctsPolicy wrappers in the live "
                    "driver; wrapping only the telemetry reference would leak opponent-private "
                    "history to the policy that actually plays."
                )
            if execution_mode == "isolated_build" and not (
                candidate_policy.is_source_isolated and incumbent_policy.is_source_isolated
            ):
                raise HeadToHeadError(
                    "isolated-build comparison must source-isolate both policies; refusing to "
                    "relabel the host engine as either declared build."
                )
        candidate_before = PolicyTelemetry.capture(candidate_policy)
        incumbent_before = PolicyTelemetry.capture(incumbent_policy)
        try:
            result = driver.run(seed=seed, battle_id=f"mcts-h2h-{seed}-{candidate_seat}")
        finally:
            close = getattr(getattr(driver, "env", None), "close", None)
            if callable(close):
                close()
            for policy in (candidate_policy, incumbent_policy):
                policy_close = getattr(policy, "close", None)
                if callable(policy_close):
                    policy_close()
        candidate_after = PolicyTelemetry.capture(candidate_policy)
        incumbent_after = PolicyTelemetry.capture(incumbent_policy)
        candidate_delta = candidate_after.delta(candidate_before)
        incumbent_delta = incumbent_after.delta(incumbent_before)
        if candidate_delta.fallback_decisions or incumbent_delta.fallback_decisions:
            raise HeadToHeadError(
                f"seed {seed} seat {candidate_seat} recorded MCTS fallback(s): "
                f"candidate={candidate_delta.fallback_decisions}, "
                f"incumbent={incumbent_delta.fallback_decisions}."
            )
        terminal = result.terminal
        game_result = GameResult(
            config_id=candidate.config_id,
            seed=seed,
            seat=candidate_seat,
            outcome=outcome_for_candidate(
                winner=terminal.winner,
                candidate_seat=candidate_seat,
                capped=bool(terminal.capped),
            ),
            turns=int(terminal.turn_count),
            provenance_sha256=candidate.provenance_sha256,
            decision_walls_s=_policy_decision_walls(result, candidate_seat),
            chosen_actions=_candidate_actions(result, candidate_seat),
        )
        game = HeadToHeadGame(
            seed=seed,
            candidate_seat=candidate_seat,
            candidate=candidate,
            incumbent=incumbent,
            result=game_result,
            terminal_winner=terminal.winner,
            terminal_capped=bool(terminal.capped),
            terminal_turn_count=int(terminal.turn_count),
            candidate_telemetry=candidate_delta,
            incumbent_telemetry=incumbent_delta,
            incumbent_decision_walls_s=_policy_decision_walls(result, incumbent_seat),
        )
        if on_game is not None:
            # Persist before the next seat begins.  If that next game dies, the
            # completed orientation is recoverable but ``complete_pair`` keeps
            # it out of every score until its mirror is also present.
            on_game(game)
        output.append(game)
    return tuple(output)


def _assert_game_identity(
    game: HeadToHeadGame,
    *,
    seed: int,
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
) -> None:
    if game.seed != seed or game.candidate != candidate or game.incumbent != incumbent:
        raise HeadToHeadError(
            "existing game does not exactly match this pair's seed/candidate/incumbent identity; "
            "never resume across provenance drift."
        )


def game_path(root: str | Path, *, seed: int, candidate_seat: str) -> Path:
    if candidate_seat not in _SEATS:
        raise ValueError("candidate_seat must be p1 or p2.")
    return Path(root) / "games" / f"seed-{seed}-{candidate_seat}.json"


def _canonical_game_bytes(game: HeadToHeadGame) -> bytes:
    return (json.dumps(game.to_payload(), sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def write_game_immutable(root: str | Path, game: HeadToHeadGame) -> Path:
    """Atomically create one durable game without replacing an existing result."""

    target = game_path(root, seed=game.seed, candidate_seat=game.candidate_seat)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = _canonical_game_bytes(game)
    try:
        existing = target.read_bytes()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != expected:
            raise HeadToHeadError(
                f"refusing to replace extant game record {target}; its immutable result differs."
            )
        return target

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != expected:
                raise HeadToHeadError(
                    f"refusing to replace concurrently-created game record {target}; it differs."
                ) from None
        return target
    finally:
        temporary.unlink(missing_ok=True)


def read_game(
    root: str | Path,
    *,
    seed: int,
    candidate_seat: str,
) -> HeadToHeadGame | None:
    path = game_path(root, seed=seed, candidate_seat=candidate_seat)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeadToHeadError(f"cannot read durable game record {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise HeadToHeadError(f"durable game record {path} is not a JSON object.")
    game = HeadToHeadGame.from_payload(payload)
    if game.seed != seed or game.candidate_seat != candidate_seat:
        raise HeadToHeadError(f"durable game record {path} is addressed by the wrong seed or seat.")
    return game


def load_pair(
    root: str | Path,
    *,
    seed: int,
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
) -> dict[str, HeadToHeadGame]:
    """Load every extant orientation, rejecting provenance drift before reuse."""

    records: dict[str, HeadToHeadGame] = {}
    for seat in _SEATS:
        game = read_game(root, seed=seed, candidate_seat=seat)
        if game is None:
            continue
        _assert_game_identity(game, seed=seed, candidate=candidate, incumbent=incumbent)
        records[seat] = game
    return records


def complete_pair(
    games: Sequence[HeadToHeadGame],
    *,
    seed: int,
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
) -> tuple[HeadToHeadGame, HeadToHeadGame]:
    """Require exactly both orientations before an aggregate can use this seed."""

    by_seat: dict[str, HeadToHeadGame] = {}
    for game in games:
        _assert_game_identity(game, seed=seed, candidate=candidate, incumbent=incumbent)
        if game.candidate_seat in by_seat:
            raise HeadToHeadError(f"seed {seed} has duplicate candidate seat {game.candidate_seat}.")
        by_seat[game.candidate_seat] = game
    missing = [seat for seat in _SEATS if seat not in by_seat]
    if missing:
        raise HeadToHeadError(
            f"seed {seed} is an incomplete mirrored pair (missing {', '.join(missing)}); "
            "it must not be scored."
        )
    return by_seat["p1"], by_seat["p2"]


def summarize_complete_pairs(
    games: Sequence[HeadToHeadGame],
    *,
    seeds: Sequence[int],
    candidate: MctsPolicySpec,
    incumbent: MctsPolicySpec,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    bootstrap_confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Return the candidate's direct MCTS-vs-MCTS score against 0.5.

    Each game is scored from the candidate's seat.  The two complementary seat
    games are first averaged into one team-seed value, so no bootstrap treats
    p1/p2 outcomes of the same matchup as independent observations.
    """

    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive.")
    required = [
        game
        for seed in seeds
        for game in complete_pair(
            [entry for entry in games if entry.seed == seed],
            seed=seed,
            candidate=candidate,
            incumbent=incumbent,
        )
    ]
    try:
        values = pair_scores(
            (game.result for game in required), seeds=seeds, config_id=candidate.config_id
        )
    except MergeError as error:
        raise HeadToHeadError(f"candidate scoring refused: {error}") from error
    interval = bootstrap_mean(
        values,
        bootstrap_indices(
            sample_size=len(values), resamples=bootstrap_resamples, seed=bootstrap_seed
        ),
        confidence_level=bootstrap_confidence_level,
    )
    candidate_walls = tuple(
        wall for game in required for wall in game.result.decision_walls_s
    )
    incumbent_walls = tuple(
        wall for game in required for wall in game.incumbent_decision_walls_s
    )
    return {
        "schema_version": "pokezero.mcts-h2h-summary.v1",
        "candidate": candidate.to_payload(),
        "incumbent": incumbent.to_payload(),
        "seeds": list(seeds),
        "pair_scores": values,
        "candidate_score": interval.to_payload(),
        "outcomes": outcome_record(
            (game.result for game in required), config_id=candidate.config_id
        ),
        "candidate_decision_walls_s": list(candidate_walls),
        "incumbent_decision_walls_s": list(incumbent_walls),
        "candidate_decision_wall_summary": _decision_wall_summary(candidate_walls),
        "incumbent_decision_wall_summary": _decision_wall_summary(incumbent_walls),
        "candidate_model_evals": sum(game.candidate_telemetry.model_evals for game in required),
        "incumbent_model_evals": sum(game.incumbent_telemetry.model_evals for game in required),
        "candidate_iterations": sum(game.candidate_telemetry.total_iterations for game in required),
        "incumbent_iterations": sum(game.incumbent_telemetry.total_iterations for game in required),
        # Prior fallbacks do not necessarily force a chosen-action fallback: the
        # engine can fail closed to uniform priors and continue its search. Keep
        # them visible for the acceptance policy instead of silently treating
        # them as either zero or a game-level policy fallback.
        "candidate_prior_fallbacks": sum(
            game.candidate_telemetry.prior_fallbacks for game in required
        ),
        "incumbent_prior_fallbacks": sum(
            game.incumbent_telemetry.prior_fallbacks for game in required
        ),
        "candidate_root_prior_fallbacks": sum(
            game.candidate_telemetry.root_prior_fallbacks for game in required
        ),
        "incumbent_root_prior_fallbacks": sum(
            game.incumbent_telemetry.root_prior_fallbacks for game in required
        ),
        "candidate_branch_prior_fallbacks": sum(
            game.candidate_telemetry.branch_prior_fallbacks for game in required
        ),
        "incumbent_branch_prior_fallbacks": sum(
            game.incumbent_telemetry.branch_prior_fallbacks for game in required
        ),
        # A flag-on opponent-prior candidate is eligible only when this
        # source-instrumented counter proves that model-priced opponent arms
        # actually reached the tree. Keep both arms explicit in the immutable
        # summary; a generic policy outcome alone cannot establish application.
        "candidate_opponent_prior_arm_decisions": sum(
            game.candidate_telemetry.opponent_prior_arm_decisions for game in required
        ),
        "incumbent_opponent_prior_arm_decisions": sum(
            game.incumbent_telemetry.opponent_prior_arm_decisions for game in required
        ),
    }
