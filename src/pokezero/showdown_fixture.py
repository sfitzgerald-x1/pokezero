"""Showdown ground-truth one-turn fixture plumbing.

This module is a small, reusable harness for running a single deterministic turn
of curated custom Gen 3 singles teams through the existing Node ``BattleStream``
bridge (``scripts/battle_bridge.mjs``). It exists so step 4 of
``docs/poke_engine_assessment.md`` (one-turn instruction outcome validation) has a
Showdown oracle to compare against.

Scope: **Showdown ground truth only.** Nothing here imports or compares against
``poke-engine``; that equivalence work is still owned by the assessment doc's
later steps. Imports are kept lazy/minimal and the runner reuses the existing
:mod:`pokezero.local_showdown` config and bridge conventions rather than spinning
up a parallel transport. The runner submits one pair of choices and returns at
the next boundary; fixtures that intentionally create a faint followed by a
forced-switch request will need a follow-up driver to resolve the replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .local_showdown import LocalShowdownConfig

# Custom Game is the only Gen 3 format that accepts arbitrary curated teams without
# random-battle set generation or Team Preview (discovered from the built checkout's
# config/formats.ts: "[Gen 3] Custom Game", id derived by stripping non-alphanumerics).
DEFAULT_GEN3_CUSTOM_FORMAT = "gen3customgame"

PLAYER_IDS = ("p1", "p2")

# Packed-set EV/IV stat order matches Pokemon Showdown's Teams.pack (hp, atk, def, spa, spd, spe).
_STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")
_MAX_IV = 31


def _pack_name(name: str | None) -> str:
    """Mirror Showdown's ``Teams.packName``: drop non-alphanumerics, keep case."""

    if not name:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "", name)


@dataclass(frozen=True)
class FixturePokemon:
    """A simple curated Gen 3 set.

    Only the knobs needed for deterministic one-turn fixtures are exposed. ``evs``
    default to all-zero and ``ivs`` default to a perfect 31 spread, matching the
    packed-team conventions Showdown assumes when those fields are omitted.
    Ability names are packed as display names; future ability-sensitive fixtures
    should use explicit, known-good names because Custom Game does not enforce
    random-battle legality.
    """

    species: str
    moves: Sequence[str]
    ability: str | None = None
    item: str | None = None
    level: int = 100
    nature: str = ""
    gender: str | None = None
    evs: Mapping[str, int] | None = None
    ivs: Mapping[str, int] | None = None


def pack_pokemon(mon: FixturePokemon) -> str:
    """Pack one :class:`FixturePokemon` into Showdown's ``|``-delimited set format."""

    if not mon.species:
        raise ValueError("FixturePokemon.species must be a non-empty species name")
    if not mon.moves:
        raise ValueError(f"FixturePokemon {mon.species!r} must list at least one move")

    # name | species | item | ability | moves | nature | evs | gender | ivs | shiny | level | happiness
    # The set is keyed by species with no nickname, so the species field stays empty (Showdown
    # recovers it from the name on unpack).
    parts = [
        mon.species,
        "",
        _pack_name(mon.item),
        _pack_name(mon.ability),
        ",".join(_pack_name(move) for move in mon.moves),
        mon.nature or "",
        _pack_evs(mon.evs),
        mon.gender or "",
        _pack_ivs(mon.ivs),
        "",  # shiny
        "" if mon.level == 100 else str(mon.level),
        "",  # happiness (default 255)
    ]
    return "|".join(parts)


def pack_team(team: Sequence[FixturePokemon]) -> str:
    """Pack an ordered party into a single ``]``-delimited Showdown team string."""

    if not team:
        raise ValueError("team must contain at least one Pokemon")
    return "]".join(pack_pokemon(mon) for mon in team)


def _pack_evs(evs: Mapping[str, int] | None) -> str:
    if not evs:
        return ""
    values = [str(int(evs.get(stat, 0))) if evs.get(stat) else "" for stat in _STAT_ORDER]
    packed = ",".join(values)
    return "" if packed == ",,,,," else packed


def _pack_ivs(ivs: Mapping[str, int] | None) -> str:
    if not ivs:
        return ""
    # A 31 in any slot packs as empty (Showdown's getIv); an all-31 spread collapses to "".
    values = ["" if int(ivs.get(stat, _MAX_IV)) == _MAX_IV else str(int(ivs.get(stat, _MAX_IV))) for stat in _STAT_ORDER]
    packed = ",".join(values)
    return "" if packed == ",,,,," else packed


@dataclass(frozen=True)
class OneTurnFixtureResult:
    """Structured outcome of running a single curated turn through Showdown."""

    format_id: str
    seed: int
    choices: Mapping[str, str]
    protocol_lines: tuple[str, ...]
    p1_request: Mapping[str, Any] | None
    p2_request: Mapping[str, Any] | None
    terminal: bool
    error_lines: tuple[str, ...] = field(default_factory=tuple)

    def move_names(self) -> tuple[str, ...]:
        """Move names that actually fired this turn, in protocol order."""

        names = []
        for line in self.protocol_lines:
            if line.startswith("|move|"):
                fields = line.split("|")
                if len(fields) >= 4:
                    names.append(fields[3])
        return tuple(names)


def build_start_payload(
    *,
    battle_id: str,
    format_id: str,
    seed: int,
    p1_team: str,
    p2_team: str,
    p1_name: str = "PokeZero p1",
    p2_name: str = "PokeZero p2",
) -> dict[str, Any]:
    """Construct the bridge ``start`` command payload for a custom-team battle.

    Factored out so the payload shape can be unit-tested without a live bridge.
    """

    from .local_showdown import showdown_seed_from_int

    return {
        "type": "start",
        "battleId": battle_id,
        "formatid": format_id,
        "seed": showdown_seed_from_int(seed),
        "players": {
            "p1": {"name": p1_name, "team": p1_team},
            "p2": {"name": p2_name, "team": p2_team},
        },
    }


def run_one_turn_fixture(
    *,
    p1_team: Sequence[FixturePokemon],
    p2_team: Sequence[FixturePokemon],
    p1_choice: str,
    p2_choice: str,
    seed: int,
    format_id: str = DEFAULT_GEN3_CUSTOM_FORMAT,
    config: "LocalShowdownConfig | None" = None,
) -> OneTurnFixtureResult:
    """Run one deterministic turn of two curated teams and collect the outcome.

    Starts a fresh one-battle ``BattleStream`` via the bridge, waits for the
    opening decision boundary, submits ``p1_choice``/``p2_choice``, then waits for
    the next boundary (or terminal) and returns the omniscient protocol plus both
    seats' opening requests. This is Showdown ground-truth plumbing only.
    """

    session = _BridgeFixtureSession(config)
    try:
        session.start(
            format_id=format_id,
            seed=seed,
            p1_team=pack_team(p1_team),
            p2_team=pack_team(p2_team),
        )
        session.read_until_boundary()
        p1_request = session.requests.get("p1")
        p2_request = session.requests.get("p2")
        choices = {"p1": p1_choice, "p2": p2_choice}
        session.send_choices(choices)
        session.read_until_boundary()
        return OneTurnFixtureResult(
            format_id=format_id,
            seed=seed,
            choices=choices,
            protocol_lines=tuple(session.protocol_lines),
            p1_request=p1_request,
            p2_request=p2_request,
            terminal=session.terminal,
            error_lines=tuple(session.error_lines),
        )
    finally:
        session.close()


class _BridgeFixtureSession:
    """Minimal single-battle driver over the Node bridge for fixture runs.

    Deliberately thinner than :class:`pokezero.local_showdown.LocalShowdownEnv`: no
    observation/belief machinery, just protocol collection. Reuses the shared
    config and drain helpers so transport conventions stay in one place.
    """

    def __init__(self, config: "LocalShowdownConfig | None") -> None:
        from .local_showdown import LocalShowdownConfig

        self.config = config or LocalShowdownConfig()
        self._battle_id = "fixture"
        self.protocol_lines: list[str] = []
        self.requests: dict[str, Mapping[str, Any]] = {}
        self.error_lines: list[str] = []
        self.terminal = False
        self._process: Any = None
        self._stdout_queue: Any = None
        self._stdout_thread: Any = None
        self._stderr_thread: Any = None
        self._stderr_lines: list[str] = []

    def start(self, *, format_id: str, seed: int, p1_team: str, p2_team: str) -> None:
        self._validate_runtime()
        self._spawn()
        payload = build_start_payload(
            battle_id=self._battle_id,
            format_id=format_id,
            seed=seed,
            p1_team=p1_team,
            p2_team=p2_team,
        )
        self._send(payload)

    def send_choices(self, choices: Mapping[str, str]) -> None:
        self.requests = {}
        self._send({"type": "choices", "battleId": self._battle_id, "choices": dict(choices)})

    def read_until_boundary(self) -> None:
        import queue

        deadline = time.monotonic() + self.config.read_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _fixture_error(f"Timed out waiting for bridge boundary. {self._exit_message()}")
            assert self._stdout_queue is not None
            try:
                line = self._stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise _fixture_error(self._exit_message())
            event = json.loads(line)
            if self._apply_event(event):
                return

    def _apply_event(self, event: Mapping[str, Any]) -> bool:
        battle_id = event.get("battleId")
        if battle_id is not None and battle_id != self._battle_id:
            return False
        event_type = event.get("type")
        if event_type == "error":
            message = str(event.get("message") or "Bridge error.")
            self.error_lines.append(message)
            raise _fixture_error(message)
        if event_type == "ready":
            return True
        if event_type == "terminal":
            self.terminal = True
            return True
        if event_type != "stream":
            return False
        stream = event.get("stream")
        lines = event.get("lines") or []
        for raw in lines:
            line = str(raw)
            if not line:
                continue
            if line.startswith("|error|"):
                self.error_lines.append(line)
            if stream == "omniscient":
                self.protocol_lines.append(line)
            elif stream in PLAYER_IDS and line.startswith("|request|"):
                request = json.loads(line[len("|request|") :])
                side = request.get("side") if isinstance(request, Mapping) else None
                if isinstance(side, Mapping) and side.get("id") == stream:
                    self.requests[stream] = request
        return False

    def _validate_runtime(self) -> None:
        import shutil

        root = self.config.resolved_showdown_root()
        bridge = self.config.resolved_bridge_path()
        if not bridge.exists():
            raise FileNotFoundError(f"Missing BattleStream bridge: {bridge}")
        if not (root / "dist" / "sim" / "index.js").exists():
            raise FileNotFoundError(
                f"Missing built Pokemon Showdown simulator at {root / 'dist' / 'sim' / 'index.js'}. "
                "Set POKEZERO_SHOWDOWN_ROOT to a built Pokemon Showdown checkout."
            )
        if shutil.which(self.config.node_binary) is None:
            raise FileNotFoundError(f"Node binary not found: {self.config.node_binary}")

    def _spawn(self) -> None:
        import os
        import queue
        import subprocess
        import threading

        from .local_showdown import _drain_stderr, _drain_stdout

        root = self.config.resolved_showdown_root()
        env = {"PATH": os.environ.get("PATH", ""), "POKEZERO_SHOWDOWN_ROOT": str(root)}
        self._process = subprocess.Popen(
            [
                self.config.node_binary,
                str(self.config.resolved_bridge_path()),
                "--showdown-root",
                str(root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._stdout_queue = queue.Queue()
        self._stdout_thread = threading.Thread(
            target=_drain_stdout, args=(self._process.stdout, self._stdout_queue), daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._process.stderr, self._stderr_lines), daemon=True
        )
        self._stderr_thread.start()

    def _send(self, payload: Mapping[str, Any]) -> None:
        if self._process is None or self._process.stdin is None or self._process.poll() is not None:
            raise _fixture_error(self._exit_message())
        self._process.stdin.write(f"{json.dumps(payload, separators=(',', ':'))}\n")
        self._process.stdin.flush()

    def close(self) -> None:
        from .local_showdown import _close_process_pipes

        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    self._send({"type": "close"})
                    process.wait(timeout=1.0)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except Exception:
                        process.kill()
        finally:
            _close_process_pipes(process)
            for thread in (self._stdout_thread, self._stderr_thread):
                if thread is not None:
                    thread.join(timeout=1.0)
            self._process = None

    def _exit_message(self) -> str:
        if self._process is not None and self._process.poll() is not None:
            stderr = "\n".join(self._stderr_lines[-20:])
            suffix = f" Stderr:\n{stderr}" if stderr else ""
            return f"BattleStream bridge exited with status {self._process.returncode}.{suffix}"
        return "BattleStream bridge is still running."


def _fixture_error(message: str) -> RuntimeError:
    # Reuse the shared error type so callers can catch one class across the harness.
    from .local_showdown import LocalShowdownError

    return LocalShowdownError(message)


def charmander_squirtle_fixture() -> tuple[list[FixturePokemon], list[FixturePokemon]]:
    """A curated one-mon-per-side Gen 3 fixture: Charmander/Ember vs. Squirtle/Water Gun.

    Mirrors the curated pairing used by the poke-engine adapter smoke so both sides
    of the eventual equivalence comparison start from the same simple matchup.
    """

    charmander = FixturePokemon(
        species="Charmander",
        ability="Blaze",
        moves=("Ember", "Tackle"),
        level=100,
    )
    squirtle = FixturePokemon(
        species="Squirtle",
        ability="Torrent",
        moves=("Water Gun", "Tackle"),
        level=100,
    )
    return [charmander], [squirtle]
