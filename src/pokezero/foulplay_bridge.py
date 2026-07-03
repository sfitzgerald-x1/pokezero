"""Controlled foul-play benchmark harness for context-aware PokeZero policies.

The existing live-server foul-play benchmark is useful for raw online play, but it cannot exercise
context-aware replay-from-root search: the online client only has protocol lines, while
``RootPUCTSearchPolicy`` needs a deterministic seed, action trajectory, and both players' current
legal requests.

This module keeps foul-play across the GPL boundary by running it as a separate websocket client,
but owns the Showdown ``BattleStream`` process so PokeZero can build the exact ``PolicyContext``
required by root-PUCT.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

from .category_vocab import CategoryVocabulary
from .dex import ShowdownDex, load_showdown_dex_cached
from .env import PlayerId, TerminalState
from .local_showdown import BRIDGE_PATH, LocalShowdownConfig, LocalShowdownEnv, showdown_seed_from_int
from .neural_policy import (
    TransformerSoftmaxPolicy,
    evaluate_transformer_action_priors,
    evaluate_transformer_observation_value,
    evaluate_transformer_opponent_action_priors,
    load_transformer_checkpoint,
)
from .observation import PokeZeroObservationV0
from .policy import Policy, PolicyContext, PolicyDecision
from .randbat_vocab import gen3_category_vocabulary
from .rollout import RolloutConfig
from .search_policy import (
    RootPUCTSearchPolicy,
    greedy_opponent_action_planner,
    prior_top_k_opponent_action_scenario_planner,
)
from .showdown import (
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    PlayerRelativeBattleState,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
    showdown_choice_for_action,
)
from .teacher_capture import action_index_from_choice_string
from .trajectory import BattleTrajectory, TrajectoryStep


SCHEMA_VERSION = "pokezero.controlled-foulplay-benchmark.v1"
DEFAULT_FOULPLAY_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "foul-play"
DEFAULT_BATTLE_ID_PREFIX = "battle-gen3randombattle-controlled"


@dataclass(frozen=True)
class ControlledFoulPlayConfig:
    checkpoint: Path
    showdown_root: Path
    foulplay_root: Path = DEFAULT_FOULPLAY_ROOT
    foulplay_python: Path | None = None
    games: int = 1
    seed_start: int = 1
    search_time_ms: int = 1000
    max_decision_rounds: int = 250
    format_id: str = "gen3randombattle"
    policy_mode: str = "root-puct"
    device: str | None = None
    temperature: float = 1.0
    cpuct: float = 1.25
    selection_mode: str = "puct"
    minimum_value_improvement: float | None = None
    root_visit_budget: int | None = None
    root_opponent_action_scenarios: int = 1
    leaf_rollout_rounds: int = 0
    opponent_legal_mask_mode: str = "hidden"
    allow_search_fallback: bool = True
    node_binary: str = "node"
    pokezero_username: str = "PokeZeroBot"
    foulplay_username: str = "FoulPlayBot"
    websocket_host: str = "127.0.0.1"

    def __post_init__(self) -> None:
        if self.games <= 0:
            raise ValueError("games must be positive.")
        if self.seed_start < 0:
            raise ValueError("seed_start must be non-negative.")
        if self.search_time_ms <= 0:
            raise ValueError("search_time_ms must be positive.")
        if self.max_decision_rounds <= 0:
            raise ValueError("max_decision_rounds must be positive.")
        if self.policy_mode not in {"raw", "root-puct"}:
            raise ValueError("policy_mode must be 'raw' or 'root-puct'.")
        if self.selection_mode not in {"puct", "value", "visits"}:
            raise ValueError("selection_mode must be 'puct', 'value', or 'visits'.")
        if self.minimum_value_improvement is not None and self.minimum_value_improvement < 0.0:
            raise ValueError("minimum_value_improvement must be non-negative when set.")
        if self.root_visit_budget is not None and self.root_visit_budget <= 0:
            raise ValueError("root_visit_budget must be positive when set.")
        if self.root_opponent_action_scenarios <= 0:
            raise ValueError("root_opponent_action_scenarios must be positive.")
        if self.leaf_rollout_rounds < 0:
            raise ValueError("leaf_rollout_rounds must be non-negative.")
        if self.opponent_legal_mask_mode not in {"hidden", "privileged"}:
            raise ValueError("opponent_legal_mask_mode must be 'hidden' or 'privileged'.")

    @property
    def resolved_foulplay_python(self) -> Path:
        if self.foulplay_python is not None:
            return self.foulplay_python
        return self.foulplay_root / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class ControlledFoulPlayGameResult:
    battle_id: str
    seed: int
    winner: str | None
    pokezero_won: bool
    decision_rounds: int
    pokezero_decisions: int
    root_puct_searches: int
    root_puct_fallbacks: int
    root_puct_total_visits: int = 0
    root_puct_fallback_reasons: Mapping[str, int] = field(default_factory=dict)
    root_puct_average_elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "battle_id": self.battle_id,
            "seed": self.seed,
            "winner": self.winner,
            "pokezero_won": self.pokezero_won,
            "decision_rounds": self.decision_rounds,
            "pokezero_decisions": self.pokezero_decisions,
            "root_puct_searches": self.root_puct_searches,
            "root_puct_fallbacks": self.root_puct_fallbacks,
            "root_puct_total_visits": self.root_puct_total_visits,
        }
        if self.root_puct_average_elapsed_seconds is not None:
            payload["root_puct_average_elapsed_seconds"] = self.root_puct_average_elapsed_seconds
        if self.root_puct_fallback_reasons:
            payload["root_puct_fallback_reasons"] = dict(sorted(self.root_puct_fallback_reasons.items()))
        return payload


@dataclass(frozen=True)
class ControlledFoulPlayBenchmarkResult:
    config: ControlledFoulPlayConfig
    policy_id: str
    games: tuple[ControlledFoulPlayGameResult, ...]

    @property
    def completed_games(self) -> int:
        return len(self.games)

    @property
    def wins(self) -> int:
        return sum(1 for game in self.games if game.pokezero_won)

    @property
    def win_rate(self) -> float:
        return self.wins / self.completed_games if self.completed_games else 0.0

    def to_dict(self) -> dict[str, Any]:
        root_searches = sum(game.root_puct_searches for game in self.games)
        root_fallbacks = sum(game.root_puct_fallbacks for game in self.games)
        root_total_visits = sum(game.root_puct_total_visits for game in self.games)
        root_fallback_reasons: dict[str, int] = {}
        for game in self.games:
            for reason, count in game.root_puct_fallback_reasons.items():
                root_fallback_reasons[reason] = root_fallback_reasons.get(reason, 0) + count
        elapsed_values = [
            game.root_puct_average_elapsed_seconds
            for game in self.games
            if game.root_puct_average_elapsed_seconds is not None
        ]
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": str(self.config.checkpoint),
            "format_id": self.config.format_id,
            "policy_id": self.policy_id,
            "policy_mode": self.config.policy_mode,
            "opponent_policy_id": "foul-play",
            "games": self.config.games,
            "completed_games": self.completed_games,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "seed_start": self.config.seed_start,
            "max_decision_rounds": self.config.max_decision_rounds,
            "root_puct": {
                "cpuct": self.config.cpuct,
                "selection_mode": self.config.selection_mode,
                "minimum_value_improvement": self.config.minimum_value_improvement,
                "root_visit_budget": self.config.root_visit_budget,
                "root_opponent_action_scenarios": self.config.root_opponent_action_scenarios,
                "leaf_rollout_rounds": self.config.leaf_rollout_rounds,
                "opponent_legal_mask_mode": self.config.opponent_legal_mask_mode,
                "foulplay_search_time_ms": self.config.search_time_ms,
                "allow_search_fallback": self.config.allow_search_fallback,
                "searches": root_searches,
                "fallbacks": root_fallbacks,
                "total_visits": root_total_visits,
            },
            "game_results": [game.to_dict() for game in self.games],
        }
        if elapsed_values:
            payload["root_puct"]["average_elapsed_seconds"] = sum(elapsed_values) / len(elapsed_values)
        if root_fallback_reasons:
            payload["root_puct"]["fallback_reasons"] = dict(sorted(root_fallback_reasons.items()))
        return payload


class FoulPlayProtocolError(RuntimeError):
    """Raised when the foul-play websocket client emits an unsupported protocol message."""


@dataclass
class _ProcessLogBuffer:
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)

    def append_stdout(self, line: str) -> None:
        self.stdout.append(line)
        if len(self.stdout) > 200:
            del self.stdout[: len(self.stdout) - 200]

    def append_stderr(self, line: str) -> None:
        self.stderr.append(line)
        if len(self.stderr) > 200:
            del self.stderr[: len(self.stderr) - 200]

    def tail(self) -> str:
        parts = []
        if self.stderr:
            parts.append("stderr:\n" + "\n".join(self.stderr[-40:]))
        if self.stdout:
            parts.append("stdout:\n" + "\n".join(self.stdout[-40:]))
        return "\n\n".join(parts) or "(no foul-play output captured)"


class _FoulPlayWebsocketServer:
    def __init__(self, *, username: str, host: str) -> None:
        self.username = username
        self.host = host
        self.port: int | None = None
        self.websocket: Any = None
        self.server: Any = None
        self.challenge_queue: asyncio.Queue[str] = asyncio.Queue()
        self.choice_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    @property
    def uri(self) -> str:
        if self.port is None:
            raise RuntimeError("server has not started.")
        return f"ws://{self.host}:{self.port}/showdown/websocket"

    async def start(self) -> None:
        import websockets

        self.server = await websockets.serve(self._handle_connection, self.host, 0, max_size=None)
        socket = self.server.sockets[0]
        self.port = int(socket.getsockname()[1])

    async def close(self) -> None:
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handle_connection(self, websocket: Any) -> None:
        self.websocket = websocket
        await websocket.send("|challstr|1|pokezero-controlled")
        try:
            async for message in websocket:
                await self._handle_message(str(message))
        except Exception:
            # The caller monitors the foul-play process and will report its stderr/stdout. Avoid
            # leaking a noisy websocket traceback as the primary error.
            self.websocket = None
            return

    async def _handle_message(self, message: str) -> None:
        room, body = _split_outgoing_showdown_message(message)
        if room and (choice := _choice_body_from_outgoing_message(body)):
            await self.choice_queue.put((room, choice))
            return
        if body.startswith("/trn "):
            await self.send_global(f"|updateuser|{self.username}|1|0|")
            return
        if body.startswith("/challenge "):
            target = body[len("/challenge ") :].split(",", 1)[0].strip()
            await self.challenge_queue.put(target)
            return
        if body.startswith("/leave "):
            battle_id = body[len("/leave ") :].strip()
            await self.send_room_lines(battle_id, ["|deinit|"])
            return
        # /utm, /timer, chat, and /savereplay are accepted no-ops for this controlled harness.

    async def send_global(self, message: str) -> None:
        if self.websocket is None:
            raise FoulPlayProtocolError("foul-play websocket is not connected.")
        await self.websocket.send(message)

    async def send_room_lines(self, battle_id: str, lines: Sequence[str]) -> None:
        if self.websocket is None:
            raise FoulPlayProtocolError("foul-play websocket is not connected.")
        if not lines:
            return
        await self.websocket.send(f">{battle_id}\n" + "\n".join(lines))

    async def wait_for_challenge(self, *, expected_target: str, timeout_seconds: float = 30.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for foul-play challenge.")
            target = await asyncio.wait_for(self.challenge_queue.get(), timeout=remaining)
            if _showdown_id(target) == _showdown_id(expected_target):
                return

    async def wait_for_choice(self, *, battle_id: str, timeout_seconds: float = 120.0) -> str:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for foul-play choice.")
            room, choice = await asyncio.wait_for(self.choice_queue.get(), timeout=remaining)
            if room == battle_id:
                return choice


class _BattleBridge:
    def __init__(self, *, showdown_root: Path, node_binary: str) -> None:
        self.showdown_root = showdown_root
        self.node_binary = node_binary
        self.process: asyncio.subprocess.Process | None = None
        self.events: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        self.stderr_lines: list[str] = []
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.process = await asyncio.create_subprocess_exec(
            self.node_binary,
            str(BRIDGE_PATH),
            "--showdown-root",
            str(self.showdown_root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                "PATH": os.environ.get("PATH", ""),
                "POKEZERO_SHOWDOWN_ROOT": str(self.showdown_root),
            },
        )
        self._stdout_task = asyncio.create_task(self._drain_stdout())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def close(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.returncode is None:
                try:
                    await self.send({"type": "close"})
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except Exception:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
        finally:
            for task in (self._stdout_task, self._stderr_task):
                if task is not None:
                    task.cancel()
            self.process = None

    async def send(self, command: Mapping[str, Any]) -> None:
        if self.process is None or self.process.stdin is None or self.process.returncode is not None:
            raise RuntimeError(self._exit_message())
        self.process.stdin.write(json.dumps(command, separators=(",", ":")).encode("utf-8") + b"\n")
        await self.process.stdin.drain()

    async def next_event(self, *, timeout_seconds: float = 120.0) -> Mapping[str, Any]:
        event = await asyncio.wait_for(self.events.get(), timeout=timeout_seconds)
        if event.get("type") == "error":
            raise RuntimeError(str(event.get("message") or "BattleStream bridge error."))
        return event

    async def _drain_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        async for raw in self.process.stdout:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            self.events.put_nowait(json.loads(line))

    async def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        async for raw in self.process.stderr:
            self.stderr_lines.append(raw.decode("utf-8", errors="replace").rstrip())
            if len(self.stderr_lines) > 100:
                del self.stderr_lines[: len(self.stderr_lines) - 100]

    def _exit_message(self) -> str:
        if self.process is not None and self.process.returncode is not None:
            stderr = "\n".join(self.stderr_lines[-20:])
            suffix = f" Stderr:\n{stderr}" if stderr else ""
            return f"BattleStream bridge exited with status {self.process.returncode}.{suffix}"
        return "BattleStream bridge is not running."


@dataclass
class _ControlledBattleState:
    battle_id: str
    seed: int
    format_id: str
    public_lines: list[str] = field(default_factory=list)
    request_lines: dict[PlayerId, str] = field(default_factory=dict)
    trajectory: BattleTrajectory | None = None
    decisions: list[PolicyDecision] = field(default_factory=list)
    next_foulplay_rqid: int = 1
    foulplay_terminal_sent: bool = False

    def all_lines(self) -> list[str]:
        return [*self.public_lines, *self.request_lines.values()]


async def run_controlled_foulplay_benchmark(
    config: ControlledFoulPlayConfig,
) -> ControlledFoulPlayBenchmarkResult:
    """Run PokeZero vs foul-play with a known BattleStream seed and context-aware policy."""

    _validate_external_paths(config)
    model, result = load_transformer_checkpoint(config.checkpoint, map_location=config.device)
    policy_id = str(result.model_config.policy_id)
    observation_spec = replace(
        DEFAULT_REPLAY_OBSERVATION_SPEC,
        categorical_feature_count=result.model_config.categorical_feature_count,
        numeric_feature_count=result.model_config.numeric_feature_count,
    )
    vocab = gen3_category_vocabulary(config.showdown_root)
    dex = load_showdown_dex_cached(config.showdown_root)
    env_config = LocalShowdownConfig(
        showdown_root=config.showdown_root,
        node_binary=config.node_binary,
        observation_spec=observation_spec,
        category_vocab=vocab,
    )
    rollout_config = RolloutConfig(
        max_decision_rounds=config.max_decision_rounds,
        format_id=config.format_id,
    )
    policy = _build_policy(
        config=config,
        model=model,
        result=result,
        env_config=env_config,
        rollout_config=rollout_config,
        policy_id=policy_id,
    )

    server = _FoulPlayWebsocketServer(username=config.foulplay_username, host=config.websocket_host)
    bridge = _BattleBridge(showdown_root=config.showdown_root, node_binary=config.node_binary)
    foulplay_process: asyncio.subprocess.Process | None = None
    foulplay_logs = _ProcessLogBuffer()
    foulplay_log_tasks: list[asyncio.Task[None]] = []
    game_results: list[ControlledFoulPlayGameResult] = []
    try:
        await server.start()
        foulplay_process = await _spawn_foulplay(config, server.uri)
        foulplay_log_tasks = [
            asyncio.create_task(_drain_process_stream(foulplay_process.stdout, foulplay_logs.append_stdout)),
            asyncio.create_task(_drain_process_stream(foulplay_process.stderr, foulplay_logs.append_stderr)),
        ]
        await bridge.start()
        for offset in range(config.games):
            seed = config.seed_start + offset
            await _wait_for_foulplay_challenge_or_exit(
                server=server,
                expected_target=config.pokezero_username,
                process=foulplay_process,
                logs=foulplay_logs,
            )
            game_results.append(
                await _run_single_game(
                    config=config,
                    bridge=bridge,
                    server=server,
                    policy=policy,
                    vocab=vocab,
                    dex=dex,
                    observation_spec=observation_spec,
                    seed=seed,
                    foulplay_process=foulplay_process,
                    foulplay_logs=foulplay_logs,
                )
            )
    finally:
        await bridge.close()
        if foulplay_process is not None and foulplay_process.returncode is None:
            foulplay_process.terminate()
            try:
                await asyncio.wait_for(foulplay_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                foulplay_process.kill()
                await foulplay_process.wait()
        for task in foulplay_log_tasks:
            task.cancel()
        await server.close()

    return ControlledFoulPlayBenchmarkResult(
        config=config,
        policy_id=(policy.policy_id if hasattr(policy, "policy_id") else policy_id),
        games=tuple(game_results),
    )


def _validate_external_paths(config: ControlledFoulPlayConfig) -> None:
    if not config.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {config.checkpoint}")
    if not (config.showdown_root / "dist" / "sim" / "index.js").exists():
        raise FileNotFoundError(
            f"built Pokemon Showdown simulator not found under {config.showdown_root}; "
            "set --showdown-root to a built checkout."
        )
    if not (config.foulplay_root / "run.py").exists():
        raise FileNotFoundError(
            f"foul-play checkout not found at {config.foulplay_root}; initialize third_party/foul-play "
            "or pass --foulplay-root."
        )
    if not config.resolved_foulplay_python.exists():
        raise FileNotFoundError(
            f"foul-play Python not found at {config.resolved_foulplay_python}; run "
            "scripts/setup_foulplay_eval.sh or pass --foulplay-python."
        )


def _build_policy(
    *,
    config: ControlledFoulPlayConfig,
    model: Any,
    result: Any,
    env_config: LocalShowdownConfig,
    rollout_config: RolloutConfig,
    policy_id: str,
) -> Policy:
    def raw_policy(policy_id_override: str | None = None) -> TransformerSoftmaxPolicy:
        return TransformerSoftmaxPolicy(
            model=model,
            result=result,
            deterministic=True,
            sampling_temperature=config.temperature,
            device=config.device,
            policy_id=policy_id_override,
        )

    if config.policy_mode == "raw":
        return raw_policy(policy_id)

    search_policy_id = f"{policy_id}+root-puct"

    def value_fn(history: tuple[PokeZeroObservationV0, ...]) -> float:
        return evaluate_transformer_observation_value(
            model=model,
            result=result,
            observations=history,
            device=config.device,
        )

    def prior_fn(history: tuple[PokeZeroObservationV0, ...]) -> tuple[float, ...]:
        return evaluate_transformer_action_priors(
            model=model,
            result=result,
            observations=history,
            temperature=config.temperature,
            device=config.device,
        )

    def opponent_prior_fn(history: tuple[PokeZeroObservationV0, ...]) -> tuple[float, ...]:
        return evaluate_transformer_opponent_action_priors(
            model=model,
            result=result,
            observations=history,
            temperature=config.temperature,
            device=config.device,
        )

    scenario_planner = None
    if config.root_opponent_action_scenarios > 1:
        scenario_planner = prior_top_k_opponent_action_scenario_planner(
            opponent_prior_fn,
            scenario_count=config.root_opponent_action_scenarios,
        )

    leaf_rollout_policy_factory = None
    if config.leaf_rollout_rounds:
        leaf_rollout_policy_factory = lambda player_id: raw_policy(f"{search_policy_id}-leaf-{player_id}")

    return RootPUCTSearchPolicy(
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=rollout_config,
        value_fn=value_fn,
        prior_fn=prior_fn,
        opponent_action_planner=greedy_opponent_action_planner(opponent_prior_fn),
        opponent_action_scenario_planner=scenario_planner,
        fallback_policy=raw_policy(f"{search_policy_id}-fallback"),
        allow_fallback=config.allow_search_fallback,
        policy_id=search_policy_id,
        cpuct=config.cpuct,
        selection_mode=config.selection_mode,
        minimum_value_improvement=config.minimum_value_improvement,
        root_visit_budget=config.root_visit_budget,
        leaf_rollout_decision_rounds=config.leaf_rollout_rounds,
        leaf_rollout_policy_factory=leaf_rollout_policy_factory,
        leaf_rollout_metadata={"root_puct_leaf_rollout_opponent_policy": "checkpoint"}
        if config.leaf_rollout_rounds
        else {},
    )


async def _spawn_foulplay(
    config: ControlledFoulPlayConfig,
    websocket_uri: str,
) -> asyncio.subprocess.Process:
    env = {
        **os.environ,
        "FOULPLAY_LOCAL_NOSEC": "1",
        "PYTHONPATH": str(config.foulplay_root),
    }
    return await asyncio.create_subprocess_exec(
        str(config.resolved_foulplay_python),
        str(config.foulplay_root / "run.py"),
        "--websocket-uri",
        websocket_uri,
        "--ps-username",
        config.foulplay_username,
        "--bot-mode",
        "challenge_user",
        "--user-to-challenge",
        config.pokezero_username,
        "--pokemon-format",
        config.format_id,
        "--run-count",
        str(config.games),
        "--search-time-ms",
        str(config.search_time_ms),
        cwd=str(config.foulplay_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )


async def _drain_process_stream(
    stream: asyncio.StreamReader | None,
    append: Any,
) -> None:
    if stream is None:
        return
    async for raw in stream:
        append(raw.decode("utf-8", errors="replace").rstrip())


async def _run_single_game(
    *,
    config: ControlledFoulPlayConfig,
    bridge: _BattleBridge,
    server: _FoulPlayWebsocketServer,
    policy: Policy,
    vocab: CategoryVocabulary,
    dex: ShowdownDex,
    observation_spec: Any,
    seed: int,
    foulplay_process: asyncio.subprocess.Process,
    foulplay_logs: _ProcessLogBuffer,
) -> ControlledFoulPlayGameResult:
    battle_id = f"{DEFAULT_BATTLE_ID_PREFIX}-{seed}"
    state = _ControlledBattleState(
        battle_id=battle_id,
        seed=seed,
        format_id=config.format_id,
        trajectory=BattleTrajectory(
            battle_id=battle_id,
            format_id=config.format_id,
            seed=seed,
            metadata={"opponent_policy_id": "foul-play", "controlled_foulplay_bridge": True},
        ),
    )
    await server.send_room_lines(
        battle_id,
        ["|init|battle", f"|title|{config.pokezero_username} vs. {config.foulplay_username}"],
    )
    await bridge.send(
        {
            "type": "start",
            "battleId": battle_id,
            "formatid": config.format_id,
            "seed": showdown_seed_from_int(seed),
            "players": {
                "p1": config.pokezero_username,
                "p2": config.foulplay_username,
            },
        }
    )

    requested_players: tuple[PlayerId, ...] = ()
    decision_round = 0
    terminal: TerminalState | None = None

    while terminal is None:
        if decision_round >= config.max_decision_rounds:
            terminal = TerminalState(winner=None, turn_count=config.max_decision_rounds, capped=True)
            break
        event = await bridge.next_event()
        if event.get("battleId") != battle_id:
            continue
        event_type = event.get("type")
        if event_type == "stream":
            await _handle_stream_event(state, server, event)
            terminal = _terminal_from_public_lines(state.public_lines, config)
            continue
        if event_type == "ready":
            requested_players = tuple(str(player) for player in event.get("requested") or ())
            if not requested_players:
                continue
            terminal = await _handle_decision_boundary(
                config=config,
                bridge=bridge,
                server=server,
                state=state,
                policy=policy,
                vocab=vocab,
                dex=dex,
                observation_spec=observation_spec,
                decision_round=decision_round,
                requested_players=requested_players,
                foulplay_process=foulplay_process,
                foulplay_logs=foulplay_logs,
            )
            decision_round += 1
            continue
        if event_type == "terminal":
            terminal = _terminal_from_public_lines(state.public_lines, config) or TerminalState(
                winner=None,
                turn_count=decision_round,
            )
            break

    await _notify_foulplay_terminal(
        state=state,
        server=server,
        terminal=terminal,
        config=config,
    )
    winner_name = _winner_name(terminal, config)
    if state.trajectory is not None:
        state.trajectory.record_terminal(terminal)
    elapsed = [
        float(decision.metadata["root_puct_elapsed_seconds"])
        for decision in state.decisions
        if "root_puct_elapsed_seconds" in decision.metadata
    ]
    root_searches = sum(
        1
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
    )
    root_fallbacks = sum(1 for decision in state.decisions if decision.metadata.get("root_puct_fallback"))
    root_fallback_reasons: dict[str, int] = {}
    for decision in state.decisions:
        if not decision.metadata.get("root_puct_fallback"):
            continue
        reason = str(decision.metadata.get("root_puct_fallback_reason") or "unknown")
        root_fallback_reasons[reason] = root_fallback_reasons.get(reason, 0) + 1
    root_total_visits = sum(
        int(decision.metadata.get("root_puct_total_visits") or 0)
        for decision in state.decisions
        if decision.metadata.get("policy_family") == "root-puct-search"
        and not decision.metadata.get("root_puct_fallback")
    )
    return ControlledFoulPlayGameResult(
        battle_id=battle_id,
        seed=seed,
        winner=winner_name,
        pokezero_won=winner_name == config.pokezero_username,
        decision_rounds=decision_round,
        pokezero_decisions=len(state.decisions),
        root_puct_searches=root_searches,
        root_puct_fallbacks=root_fallbacks,
        root_puct_total_visits=root_total_visits,
        root_puct_fallback_reasons=root_fallback_reasons,
        root_puct_average_elapsed_seconds=(sum(elapsed) / len(elapsed) if elapsed else None),
    )


async def _handle_stream_event(
    state: _ControlledBattleState,
    server: _FoulPlayWebsocketServer,
    event: Mapping[str, Any],
) -> None:
    stream = event.get("stream")
    raw_lines = event.get("lines")
    if not isinstance(stream, str) or not isinstance(raw_lines, list):
        raise RuntimeError(f"malformed BattleStream event: {event!r}")
    lines = [str(line) for line in raw_lines if str(line)]
    if stream == "omniscient":
        state.public_lines.extend(lines)
    elif stream in {"p1", "p2"}:
        for line in lines:
            if line.startswith("|request|"):
                state.request_lines[stream] = line
        if stream == "p2":
            forwarded = [_line_for_foulplay(state, line) for line in lines]
            for chunk in _line_chunks_safe_for_foulplay(forwarded):
                await server.send_room_lines(state.battle_id, chunk)
            if any(_is_terminal_protocol_line(line) for line in forwarded):
                state.foulplay_terminal_sent = True


async def _notify_foulplay_terminal(
    *,
    state: _ControlledBattleState,
    server: _FoulPlayWebsocketServer,
    terminal: TerminalState,
    config: ControlledFoulPlayConfig,
) -> None:
    if state.foulplay_terminal_sent:
        return
    line = _terminal_line_for_foulplay(terminal, config)
    await server.send_room_lines(state.battle_id, [line])
    state.foulplay_terminal_sent = True


def _terminal_line_for_foulplay(
    terminal: TerminalState,
    config: ControlledFoulPlayConfig,
) -> str:
    winner = _winner_name(terminal, config)
    if winner is None:
        return "|tie|"
    return f"|win|{winner}"


def _is_terminal_protocol_line(line: str) -> bool:
    return line.startswith("|win|") or line == "|tie" or line.startswith("|tie|")


def _line_for_foulplay(state: _ControlledBattleState, line: str) -> str:
    if not line.startswith("|request|"):
        return line
    payload = json.loads(line[len("|request|") :])
    if isinstance(payload, dict) and "rqid" not in payload:
        payload = dict(payload)
        payload["rqid"] = state.next_foulplay_rqid
        state.next_foulplay_rqid += 1
        return "|request|" + json.dumps(payload, separators=(",", ":"))
    return line


def _line_chunks_safe_for_foulplay(lines: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Filter and chunk BattleStream lines into messages foul-play can parse.

    foul-play uses the first pipe-delimited command in a websocket message to decide how to parse
    the whole block. BattleStream can put metadata before ``|player|`` or ``|request|`` in the same
    chunk, so force those parser-sensitive lines to the front of their own messages.
    """

    safe_lines = tuple(
        line
        for line in lines
        if line and line != "|" and not line.startswith("|t:|")
    )
    chunks: list[tuple[str, ...]] = []
    current: list[str] = []
    for line in safe_lines:
        if line.startswith("|player|") or line.startswith("|request|"):
            if current:
                chunks.append(tuple(current))
                current = []
            chunks.append((line,))
        else:
            current.append(line)
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


async def _handle_decision_boundary(
    *,
    config: ControlledFoulPlayConfig,
    bridge: _BattleBridge,
    server: _FoulPlayWebsocketServer,
    state: _ControlledBattleState,
    policy: Policy,
    vocab: CategoryVocabulary,
    dex: ShowdownDex,
    observation_spec: Any,
    decision_round: int,
    requested_players: tuple[PlayerId, ...],
    foulplay_process: asyncio.subprocess.Process,
    foulplay_logs: _ProcessLogBuffer,
) -> TerminalState | None:
    assert state.trajectory is not None
    player_states = {
        player: _player_state(state, player)
        for player in requested_players
    }
    observations = {
        player: observation_from_player_state(
            player_states[player],
            category_vocab=vocab,
            spec=observation_spec,
            dex=dex,
        )
        for player in requested_players
    }
    choices: dict[PlayerId, str] = {}
    decisions: dict[PlayerId, PolicyDecision] = {}
    if "p1" in requested_players:
        p1_context = PolicyContext(
            player_id="p1",
            decision_round_index=decision_round,
            battle_id=state.battle_id,
            format_id=config.format_id,
            seed=state.seed,
            observation=observations["p1"],
            requested_players=requested_players,
            trajectory=state.trajectory,
            requested_legal_action_masks=_requested_legal_action_masks_for_context(
                observations,
                acting_player="p1",
                opponent_legal_mask_mode=config.opponent_legal_mask_mode,
            ),
            requested_observations=dict(observations),
        )
        decisions["p1"] = await asyncio.to_thread(
            _select_policy_decision,
            policy,
            observations["p1"],
            p1_context,
            seed=state.seed,
        )
        choices["p1"] = showdown_choice_for_action(player_states["p1"], decisions["p1"].action_index)
    if "p2" in requested_players:
        choice = await _wait_for_foulplay_choice_or_exit(
            server=server,
            battle_id=state.battle_id,
            process=foulplay_process,
            logs=foulplay_logs,
        )
        p2_action = action_index_from_choice_string(player_states["p2"], choice)
        if p2_action is None:
            raise RuntimeError(f"unable to decode foul-play choice {choice!r}.")
        choices["p2"] = choice
        decisions["p2"] = PolicyDecision(
            action_index=p2_action,
            policy_id="foul-play",
            metadata={"raw_choice": choice},
        )

    for player in requested_players:
        decision = decisions.get(player)
        if decision is None:
            continue
        state.trajectory.append(
            TrajectoryStep(
                player_id=player,
                turn_index=decision_round,
                observation=observations[player],
                legal_action_mask=tuple(observations[player].legal_action_mask),
                action_index=decision.action_index,
                metadata={"policy_id": decision.policy_id, **dict(decision.metadata)},
            )
        )
        if player == "p1":
            state.decisions.append(decision)

    await bridge.send({"type": "choices", "battleId": state.battle_id, "choices": choices})
    return None


async def _wait_for_foulplay_choice_or_exit(
    *,
    server: _FoulPlayWebsocketServer,
    battle_id: str,
    process: asyncio.subprocess.Process,
    logs: _ProcessLogBuffer,
) -> str:
    if process.returncode is not None:
        raise RuntimeError(f"foul-play exited with status {process.returncode} before choosing.\n{logs.tail()}")
    choice_task = asyncio.create_task(server.wait_for_choice(battle_id=battle_id))
    process_task = asyncio.create_task(process.wait())
    try:
        done, pending = await asyncio.wait(
            {choice_task, process_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if choice_task in done:
            return choice_task.result()
        raise RuntimeError(
            f"foul-play exited with status {process.returncode} before choosing.\n{logs.tail()}"
        )
    finally:
        for task in (choice_task, process_task):
            if not task.done():
                task.cancel()


async def _wait_for_foulplay_challenge_or_exit(
    *,
    server: _FoulPlayWebsocketServer,
    expected_target: str,
    process: asyncio.subprocess.Process,
    logs: _ProcessLogBuffer,
) -> None:
    if process.returncode is not None:
        raise RuntimeError(f"foul-play exited with status {process.returncode} before challenging.\n{logs.tail()}")
    challenge_task = asyncio.create_task(server.wait_for_challenge(expected_target=expected_target))
    process_task = asyncio.create_task(process.wait())
    try:
        done, pending = await asyncio.wait(
            {challenge_task, process_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if challenge_task in done:
            challenge_task.result()
            return
        raise RuntimeError(
            f"foul-play exited with status {process.returncode} before challenging.\n{logs.tail()}"
        )
    finally:
        for task in (challenge_task, process_task):
            if not task.done():
                task.cancel()


def _select_policy_decision(
    policy: Policy,
    observation: PokeZeroObservationV0,
    context: PolicyContext,
    *,
    seed: int,
) -> PolicyDecision:
    rng = random.Random(f"{seed}:{context.player_id}:{context.decision_round_index}")
    selector = getattr(policy, "select_action_with_context", None)
    if callable(selector):
        return selector(context, rng=rng)
    return policy.select_action(observation, rng=rng)


def _requested_legal_action_masks_for_context(
    observations: Mapping[PlayerId, PokeZeroObservationV0],
    *,
    acting_player: PlayerId,
    opponent_legal_mask_mode: str,
) -> dict[PlayerId, tuple[bool, ...]]:
    masks: dict[PlayerId, tuple[bool, ...]] = {}
    for player, observation in observations.items():
        if player != acting_player and opponent_legal_mask_mode == "hidden":
            continue
        masks[player] = tuple(observation.legal_action_mask)
    return masks


def _player_state(state: _ControlledBattleState, player: PlayerId) -> PlayerRelativeBattleState:
    replay = parse_showdown_replay(state.all_lines(), battle_id=state.battle_id)
    return normalize_for_player(
        replay,
        player_id=player,
        configured_showdown_slot=player,
        format_id=state.format_id,
    )


def _terminal_from_public_lines(
    lines: Sequence[str],
    config: ControlledFoulPlayConfig,
) -> TerminalState | None:
    turn = 0
    winner: PlayerId | None = None
    for line in lines:
        if line.startswith("|turn|"):
            try:
                turn = int(line.split("|", 2)[2])
            except (IndexError, ValueError):
                pass
        elif line.startswith("|win|"):
            winner_name = line.split("|", 2)[2] if len(line.split("|", 2)) >= 3 else ""
            if winner_name == config.pokezero_username:
                winner = "p1"
            elif winner_name == config.foulplay_username:
                winner = "p2"
            return TerminalState(winner=winner, turn_count=turn)
        elif line == "|tie" or line.startswith("|tie|"):
            return TerminalState(winner=None, turn_count=turn)
    return None


def _winner_name(terminal: TerminalState, config: ControlledFoulPlayConfig) -> str | None:
    if terminal.winner == "p1":
        return config.pokezero_username
    if terminal.winner == "p2":
        return config.foulplay_username
    return None


def _split_outgoing_showdown_message(message: str) -> tuple[str, str]:
    if "|" not in message:
        return "", message.strip()
    room, body = message.split("|", 1)
    return room.strip(), body.strip()


def _choice_body_from_outgoing_message(body: str) -> str | None:
    command = body.split("|", 1)[0].strip()
    if command.startswith("/choose "):
        return command[len("/choose ") :].strip()
    if command.startswith("/switch "):
        return f"switch {command[len('/switch ') :].strip()}"
    return None


def _showdown_id(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a controlled BattleStream benchmark: PokeZero policy vs external foul-play.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Transformer checkpoint path.")
    parser.add_argument(
        "--showdown-root",
        type=Path,
        default=Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT", "")) if os.environ.get("POKEZERO_SHOWDOWN_ROOT") else None,
        help="Built Pokemon Showdown checkout root, or POKEZERO_SHOWDOWN_ROOT.",
    )
    parser.add_argument("--foulplay-root", type=Path, default=DEFAULT_FOULPLAY_ROOT, help="foul-play checkout path.")
    parser.add_argument("--foulplay-python", type=Path, default=None, help="Python executable for foul-play.")
    parser.add_argument("--games", type=int, default=1, help="Number of games.")
    parser.add_argument("--seed-start", type=int, default=1, help="First deterministic BattleStream seed.")
    parser.add_argument("--search-time-ms", type=int, default=1000, help="foul-play search time per move.")
    parser.add_argument("--max-decision-rounds", type=int, default=250, help="Decision-round cap.")
    parser.add_argument("--format", dest="format_id", default="gen3randombattle", help="Showdown format id.")
    parser.add_argument("--policy-mode", choices=("raw", "root-puct"), default="root-puct")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu, cuda, mps.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Checkpoint policy softmax temperature.")
    parser.add_argument("--cpuct", type=float, default=1.25, help="Root PUCT exploration constant.")
    parser.add_argument(
        "--selection-mode",
        choices=("puct", "value", "visits"),
        default="puct",
        help="Root search candidate selection rule.",
    )
    parser.add_argument(
        "--minimum-value-improvement",
        type=float,
        default=None,
        help=(
            "Require the search-selected action to beat the prior-best action by this value margin; "
            "otherwise use the prior-best action."
        ),
    )
    parser.add_argument(
        "--root-visit-budget",
        type=int,
        default=None,
        help="Total root visits per searched decision; defaults to one visit per legal action.",
    )
    parser.add_argument(
        "--root-opponent-action-scenarios",
        type=int,
        default=1,
        help="Number of checkpoint-prior opponent root-action scenarios to average.",
    )
    parser.add_argument(
        "--leaf-rollout-rounds",
        type=int,
        default=0,
        help="Decision rounds to continue each root branch before leaf value evaluation.",
    )
    parser.add_argument(
        "--opponent-legal-mask-mode",
        choices=("hidden", "privileged"),
        default="hidden",
        help=(
            "Whether root opponent-action planning withholds the opponent's private legal mask "
            "(hidden, default) or uses it as a privileged benchmark safety guard."
        ),
    )
    parser.add_argument(
        "--no-search-fallback",
        action="store_true",
        help="Raise on search failure instead of falling back to the raw checkpoint action.",
    )
    parser.add_argument("--node-binary", default="node", help="Node executable for BattleStream bridge.")
    parser.add_argument("--pokezero-username", default="PokeZeroBot")
    parser.add_argument("--foulplay-username", default="FoulPlayBot")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional JSON result path.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.showdown_root is None:
        parser.error("--showdown-root is required, or set POKEZERO_SHOWDOWN_ROOT.")
    config = ControlledFoulPlayConfig(
        checkpoint=args.checkpoint,
        showdown_root=args.showdown_root,
        foulplay_root=args.foulplay_root,
        foulplay_python=args.foulplay_python,
        games=args.games,
        seed_start=args.seed_start,
        search_time_ms=args.search_time_ms,
        max_decision_rounds=args.max_decision_rounds,
        format_id=args.format_id,
        policy_mode=args.policy_mode,
        device=args.device,
        temperature=args.temperature,
        cpuct=args.cpuct,
        selection_mode=args.selection_mode,
        minimum_value_improvement=args.minimum_value_improvement,
        root_visit_budget=args.root_visit_budget,
        root_opponent_action_scenarios=args.root_opponent_action_scenarios,
        leaf_rollout_rounds=args.leaf_rollout_rounds,
        opponent_legal_mask_mode=args.opponent_legal_mask_mode,
        allow_search_fallback=not args.no_search_fallback,
        node_binary=args.node_binary,
        pokezero_username=args.pokezero_username,
        foulplay_username=args.foulplay_username,
    )
    result = await run_controlled_foulplay_benchmark(config)
    payload = result.to_dict()
    if args.summary_out is not None:
        _write_json(args.summary_out, payload)
        print(f"controlled_foulplay_summary: {args.summary_out}", file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"RESULT: {result.policy_id} won {result.wins}/{result.completed_games} "
            f"vs foul-play ({result.win_rate:.1%})"
        )
        root = payload["root_puct"]
        if isinstance(root, Mapping) and root.get("searches"):
            print(
                "root-puct: "
                f"searches={root.get('searches')} fallbacks={root.get('fallbacks')} "
                f"avg_elapsed={root.get('average_elapsed_seconds', 'n/a')}"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
