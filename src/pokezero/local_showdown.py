"""Local Pokemon Showdown BattleStream-backed PokeZero environment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping, Optional, TextIO

from .dex import load_showdown_dex_cached
from .env import BattleFormat, PlayerId, StepResult, TerminalState
from .observation import ObservationSpec, PokeZeroObservationV0
from .randbat_vocab import gen3_category_vocabulary
from .showdown import (
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    PlayerRelativeBattleState,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
    showdown_choice_for_action,
)

DEFAULT_SHOWDOWN_ROOT = Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown")
BRIDGE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "battle_bridge.mjs"
PLAYER_IDS: tuple[PlayerId, PlayerId] = ("p1", "p2")


class LocalShowdownError(RuntimeError):
    """Raised when the local BattleStream bridge or simulator rejects a step."""


@dataclass(frozen=True)
class LocalShowdownConfig:
    showdown_root: Path | str | None = None
    bridge_path: Path | str = BRIDGE_PATH
    node_binary: str = "node"
    observation_spec: ObservationSpec = DEFAULT_REPLAY_OBSERVATION_SPEC
    read_timeout_seconds: float = 10.0

    def resolved_showdown_root(self) -> Path:
        configured = self.showdown_root or os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT
        return Path(configured).expanduser().resolve()

    def resolved_bridge_path(self) -> Path:
        return Path(self.bridge_path).expanduser().resolve()


class LocalShowdownEnv:
    """Synchronous `PokeZeroEnv` backed by a one-battle Node BattleStream bridge."""

    def __init__(self, config: LocalShowdownConfig | None = None) -> None:
        self.config = config or LocalShowdownConfig()
        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: list[str] = []
        self._battle_id = "local-showdown"
        self._format_id: BattleFormat = "gen3randombattle"
        self._lines: list[str] = []
        self._latest_requests: dict[PlayerId, Mapping[str, Any]] = {}
        self._latest_turn = 0
        self._terminal: TerminalState | None = None
        self._last_step_had_error = False

    @property
    def protocol_lines(self) -> tuple[str, ...]:
        return tuple(self._lines)

    def reset(self, *, seed: int, format_id: BattleFormat = "gen3randombattle") -> None:
        self.close()
        self._validate_runtime()
        self._battle_id = f"local-{format_id}-{seed}"
        self._format_id = format_id
        self._lines = []
        self._latest_requests = {}
        self._latest_turn = 0
        self._terminal = None
        self._last_step_had_error = False
        self._start_bridge()
        try:
            self._send_command(
                {
                    "type": "start",
                    "formatid": format_id,
                    "seed": showdown_seed_from_int(seed),
                    "players": {"p1": "PokeZero p1", "p2": "PokeZero p2"},
                }
            )
            self._read_until_boundary()
        except Exception:
            self.close()
            raise

    def requested_players(self) -> tuple[PlayerId, ...]:
        return requested_players_from_requests(self._latest_requests)

    def observe(self, player: PlayerId) -> PokeZeroObservationV0:
        state = self._state_for_player(player)
        root = self.config.resolved_showdown_root()
        # Both cached: rows align deterministically with the model's vocab built from the same root.
        return observation_from_player_state(
            state,
            category_vocab=gen3_category_vocabulary(root),
            spec=self.config.observation_spec,
            dex=load_showdown_dex_cached(root),
        )

    def legal_actions(self, player: PlayerId) -> tuple[bool, ...]:
        return self.observe(player).legal_action_mask

    def step(self, actions: Mapping[PlayerId, int]) -> StepResult:
        requested = self.requested_players()
        if not requested:
            raise LocalShowdownError("Cannot step without requested players.")
        missing = [player for player in requested if player not in actions]
        if missing:
            raise LocalShowdownError(f"Missing actions for requested players: {', '.join(missing)}.")

        states: dict[PlayerId, PlayerRelativeBattleState] = {
            player: self._state_for_player(player) for player in requested
        }
        self._last_step_had_error = False
        self._latest_requests = {}
        choices = {
            player: showdown_choice_for_action(states[player], actions[player])
            for player in requested
        }
        self._send_command({"type": "choices", "choices": choices})
        self._read_until_boundary()
        if self._last_step_had_error:
            raise LocalShowdownError("Showdown rejected a submitted choice.")

        next_requested = self.requested_players()
        observations = {player: self.observe(player) for player in next_requested}
        rewards = self._rewards()
        terminal = self.terminal()
        if terminal is not None:
            self.close()
        return StepResult(
            observations=observations,
            rewards=rewards,
            terminal=terminal,
            requested_players=next_requested,
        )

    def terminal(self) -> Optional[TerminalState]:
        return self._terminal

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin:
                try:
                    self._send_command({"type": "close"})
                except Exception:
                    pass
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1.0)
        finally:
            _close_process_pipes(process)
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout=1.0)
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=1.0)
            self._process = None
            self._stdout_queue = None
            self._stdout_thread = None
            self._stderr_thread = None

    def __enter__(self) -> "LocalShowdownEnv":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _validate_runtime(self) -> None:
        showdown_root = self.config.resolved_showdown_root()
        bridge_path = self.config.resolved_bridge_path()
        if not bridge_path.exists():
            raise FileNotFoundError(f"Missing BattleStream bridge: {bridge_path}")
        if not (showdown_root / "dist" / "sim" / "index.js").exists():
            raise FileNotFoundError(
                f"Missing built Pokemon Showdown simulator at {showdown_root / 'dist' / 'sim' / 'index.js'}. "
                "Set POKEZERO_SHOWDOWN_ROOT to a built Pokemon Showdown checkout."
            )
        if shutil.which(self.config.node_binary) is None:
            raise FileNotFoundError(f"Node binary not found: {self.config.node_binary}")

    def _start_bridge(self) -> None:
        showdown_root = self.config.resolved_showdown_root()
        env = {
            "PATH": os.environ.get("PATH", ""),
            "POKEZERO_SHOWDOWN_ROOT": str(showdown_root),
        }
        self._process = subprocess.Popen(
            [
                self.config.node_binary,
                str(self.config.resolved_bridge_path()),
                "--showdown-root",
                str(showdown_root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        if self._process.stdout is None:
            raise LocalShowdownError("Bridge stdout was not created.")
        self._stdout_queue = queue.Queue()
        self._stdout_thread = threading.Thread(
            target=_drain_stdout,
            args=(self._process.stdout, self._stdout_queue),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(
            target=_drain_stderr,
            args=(self._process.stderr, self._stderr_lines),
            daemon=True,
        )
        self._stderr_thread.start()

    def _send_command(self, payload: Mapping[str, Any]) -> None:
        if self._process is None or self._process.stdin is None or self._process.poll() is not None:
            raise LocalShowdownError(self._bridge_exit_message())
        self._process.stdin.write(f"{json.dumps(payload, separators=(',', ':'))}\n")
        self._process.stdin.flush()

    def _read_until_boundary(self) -> None:
        deadline = time.monotonic() + self.config.read_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalShowdownError(self._timeout_message())
            event = self._read_event(timeout=remaining)
            if event is None:
                continue
            if self._apply_event(event):
                return

    def _read_event(self, *, timeout: float) -> Mapping[str, Any] | None:
        if self._stdout_queue is None:
            raise LocalShowdownError("Bridge is not running.")
        try:
            line = self._stdout_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            raise LocalShowdownError(self._bridge_exit_message())
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LocalShowdownError(f"Bridge emitted invalid JSON: {line.rstrip()}") from exc
        if not isinstance(event, Mapping):
            raise LocalShowdownError(f"Bridge emitted non-object event: {event!r}")
        return event

    def _apply_event(self, event: Mapping[str, Any]) -> bool:
        event_type = event.get("type")
        if event_type == "error":
            raise LocalShowdownError(str(event.get("message") or "Bridge error."))
        if event_type == "ready":
            return True
        if event_type == "terminal":
            if self._terminal is None:
                self._terminal = TerminalState(winner=None, turn_count=self._latest_turn)
            return True
        if event_type != "stream":
            return False
        stream = event.get("stream")
        lines = event.get("lines")
        if not isinstance(stream, str) or not isinstance(lines, list):
            raise LocalShowdownError(f"Malformed bridge stream event: {event!r}")
        clean_lines = [str(line) for line in lines if str(line)]
        for line in clean_lines:
            if line.startswith("|error|"):
                self._last_step_had_error = True
                raise LocalShowdownError(f"Showdown emitted error: {line}")
        if stream == "omniscient":
            for line in clean_lines:
                self._lines.append(line)
                self._update_public_state(line)
            return False
        if stream in PLAYER_IDS:
            for line in clean_lines:
                if line.startswith("|request|"):
                    request = _decode_request_line(line)
                    side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
                    side_id = side.get("id") if isinstance(side, Mapping) else None
                    if side_id == stream:
                        self._latest_requests[stream] = request
                        self._lines.append(line)
        return False

    def _update_public_state(self, line: str) -> None:
        if line.startswith("|turn|"):
            try:
                self._latest_turn = int(line.split("|", 2)[2])
            except (IndexError, ValueError):
                pass
            return
        if line.startswith("|win|"):
            winner_name = line.split("|", 2)[2] if len(line.split("|", 2)) >= 3 else ""
            self._terminal = TerminalState(winner=self._winner_slot(winner_name), turn_count=self._latest_turn)
            return
        if line == "|tie" or line.startswith("|tie|"):
            self._terminal = TerminalState(winner=None, turn_count=self._latest_turn)

    def _state_for_player(self, player: PlayerId) -> PlayerRelativeBattleState:
        if player not in PLAYER_IDS:
            raise ValueError(f"player must be one of {', '.join(PLAYER_IDS)}; got {player!r}.")
        replay = parse_showdown_replay(self._lines, battle_id=self._battle_id)
        return normalize_for_player(
            replay,
            player_id=player,
            configured_showdown_slot=player,
            format_id=self._format_id,
        )

    def _winner_slot(self, winner_name: str) -> PlayerId | None:
        replay = parse_showdown_replay(self._lines, battle_id=self._battle_id)
        for slot, name in replay.players.items():
            if name == winner_name:
                return slot
        return None

    def _rewards(self) -> dict[PlayerId, float]:
        if self._terminal is None:
            return {"p1": 0.0, "p2": 0.0}
        if self._terminal.winner is None:
            return {"p1": 0.0, "p2": 0.0}
        return {
            "p1": 1.0 if self._terminal.winner == "p1" else -1.0,
            "p2": 1.0 if self._terminal.winner == "p2" else -1.0,
        }

    def _timeout_message(self) -> str:
        return f"Timed out waiting for BattleStream bridge output. {self._bridge_exit_message()}"

    def _bridge_exit_message(self) -> str:
        if self._process is not None and self._process.poll() is not None:
            stderr = "\n".join(self._stderr_lines[-20:])
            suffix = f" Stderr:\n{stderr}" if stderr else ""
            return f"BattleStream bridge exited with status {self._process.returncode}.{suffix}"
        return "BattleStream bridge is still running."


def showdown_seed_from_int(seed: int) -> str:
    digest = hashlib.sha256(str(int(seed)).encode("utf-8")).digest()
    parts = [int.from_bytes(digest[index : index + 2], "big") for index in range(0, 8, 2)]
    return ",".join(str(part) for part in parts)


def requested_players_from_requests(requests: Mapping[PlayerId, Mapping[str, Any]]) -> tuple[PlayerId, ...]:
    return tuple(player for player in PLAYER_IDS if _is_actionable_request(requests.get(player)))


def _is_actionable_request(request: Mapping[str, Any] | None) -> bool:
    if not isinstance(request, Mapping):
        return False
    if request.get("wait"):
        return False
    if request.get("teamPreview"):
        return False
    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list) and any(bool(slot) for slot in force_switch):
        return True
    active = request.get("active")
    return isinstance(active, list) and bool(active)


def _decode_request_line(line: str) -> Mapping[str, Any]:
    prefix = "|request|"
    if not line.startswith(prefix):
        raise ValueError("request line must start with |request|")
    payload = json.loads(line[len(prefix) :])
    if not isinstance(payload, Mapping):
        raise ValueError("request payload must be a JSON object.")
    return payload


def _drain_stdout(stream: TextIO, target: queue.Queue[str | None]) -> None:
    try:
        for line in stream:
            target.put(line)
    except (OSError, ValueError):
        pass
    finally:
        target.put(None)


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass


def _drain_stderr(stream: TextIO | None, target: list[str]) -> None:
    if stream is None:
        return
    try:
        for line in stream:
            target.append(line.rstrip())
            if len(target) > 100:
                del target[: len(target) - 100]
    except (OSError, ValueError):
        pass
