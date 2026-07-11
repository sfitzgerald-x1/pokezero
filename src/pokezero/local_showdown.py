"""Local Pokemon Showdown BattleStream-backed PokeZero environment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TextIO

if TYPE_CHECKING:
    from .category_vocab import CategoryVocabulary

from .belief import PublicBattleBeliefEngine
from .dex import load_showdown_dex_cached
from .env import BattleFormat, BattleStartOverride, PlayerId, StepResult, TerminalState
from .observation import (
    DEFAULT_OBSERVATION_FEATURE_MASKS,
    OBSERVATION_SCHEMA_VERSION_V2_2,
    ObservationFeatureMasks,
    ObservationSpec,
    PokeZeroObservationV0,
)
from .randbat import load_gen3_randbat_source_cached
from .randbat_vocab import gen3_category_vocabulary
from .showdown import (
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    PlayerRelativeBattleState,
    _ReplayParser,
    normalize_for_player,
    observation_from_player_state,
    showdown_choice_for_action,
)
from .investment import InvestmentLiveTracker
from .tier2 import Tier2LiveTracker, cb_whitelist_for_source, own_team_from_request

DEFAULT_SHOWDOWN_ROOT = Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown")
BRIDGE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "battle_bridge.mjs"
PLAYER_IDS: tuple[PlayerId, PlayerId] = ("p1", "p2")
_DEFAULT_PLAYER_NAMES: Mapping[PlayerId, str] = {"p1": "PokeZero p1", "p2": "PokeZero p2"}


class LocalShowdownError(RuntimeError):
    """Raised when the local BattleStream bridge or simulator rejects a step."""


def env_config_with_checkpoint_masks(
    env_config: LocalShowdownConfig,
    required_masks: "ObservationFeatureMasks | Sequence[ObservationFeatureMasks]",
    *,
    context: str,
    required_specs: "ObservationSpec | Sequence[ObservationSpec]" = (),
) -> "LocalShowdownConfig":
    """Derive the env's encode-time feature masks AND observation spec from checkpoint provenance.

    The train/eval consistency latch for the mask axis (same failure shape as the #492
    belief-source mismatch): a checkpoint stamped with ablation masks (K=32 budget, stats-off,
    exact-state-off) must be evaluated on observations encoded the same way. Semantics:

    - no transformer checkpoints in play -> env config unchanged;
    - checkpoints agree on one mask set -> env adopts it (overriding the untouched default);
    - checkpoints DISAGREE -> hard fail (one env cannot encode two ways);
    - env carries an EXPLICIT non-default mask config that differs from the checkpoints'
      -> hard fail loudly (never silently prefer either side).

    ``required_specs`` extends the latch to the observation SCHEMA axis with identical
    semantics (the dual-schema window's core mechanism): pass each loaded checkpoint's
    ``observation_spec_from_model_config`` so a v2 checkpoint gets the v2 encode (121
    columns, no v2.1 blocks) and a v2.1 checkpoint the v2.1 encode — resolved from stamped
    provenance, never from the build's default. A v2 and a v2.1 checkpoint in one env, or an
    explicit non-default env spec that disagrees with the checkpoints', hard-fails loudly.
    """
    from .observation import ObservationFeatureMasks, ObservationSpec

    if isinstance(required_masks, ObservationFeatureMasks):
        required_masks = (required_masks,)
    distinct: list[ObservationFeatureMasks] = []
    for masks in required_masks:
        if masks not in distinct:
            distinct.append(masks)
    if isinstance(required_specs, ObservationSpec):
        required_specs = (required_specs,)
    distinct_specs: list[ObservationSpec] = []
    for spec in required_specs:
        if spec not in distinct_specs:
            distinct_specs.append(spec)
    if not distinct and not distinct_specs:
        return env_config
    if len(distinct) > 1:
        raise ValueError(
            f"{context}: checkpoints require conflicting observation feature masks "
            f"({', '.join(repr(masks) for masks in distinct)}); one env cannot encode both — "
            "evaluate them in separate runs."
        )
    if len(distinct_specs) > 1:
        schemas = sorted({spec.schema_version for spec in distinct_specs})
        raise ValueError(
            f"{context}: checkpoints require conflicting observation specs "
            f"({', '.join(repr(spec) for spec in distinct_specs)}); one env cannot encode "
            "two observation schemas. For eval, score them in separate runs; for "
            "iterate/resume, a training line keeps its own stamped schema "
            f"({' vs '.join(repr(schema) for schema in schemas)}) — continue it on the "
            "build it is pinned to instead of mixing it with fresh-stamped configs."
        )
    resolved = env_config
    if distinct:
        required = distinct[0]
        if resolved.feature_masks != required:
            if resolved.feature_masks != DEFAULT_OBSERVATION_FEATURE_MASKS:
                raise ValueError(
                    f"{context}: env feature masks {resolved.feature_masks!r} conflict with the "
                    f"loaded checkpoint's trained masks {required!r}. Refusing to encode observations "
                    "the model never trained on (the #492 train/eval-mismatch class); drop the "
                    "explicit env masks or evaluate a matching checkpoint."
                )
            resolved = replace(resolved, feature_masks=required)
    if distinct_specs:
        required_spec = distinct_specs[0]
        if resolved.observation_spec != required_spec:
            if resolved.observation_spec != DEFAULT_REPLAY_OBSERVATION_SPEC:
                raise ValueError(
                    f"{context}: env observation spec {resolved.observation_spec!r} conflicts "
                    f"with the loaded checkpoint's trained spec {required_spec!r} "
                    f"(schema {required_spec.schema_version!r}). Refusing to encode a schema "
                    "the model never trained on (the census-mismatch class); drop the explicit "
                    "env spec or evaluate a matching checkpoint."
                )
            resolved = replace(resolved, observation_spec=required_spec)
    return resolved


def belief_set_source_env_enabled() -> bool:
    """The single env flip point for candidate-set belief features (training AND eval sides).

    Every consumer must call this rather than re-parsing the variable: two independent parsers
    drifting apart is exactly the silent train/eval observation mismatch class.
    """
    return os.environ.get("POKEZERO_BELIEF_SET_SOURCE", "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LocalShowdownConfig:
    showdown_root: Path | str | None = None
    bridge_path: Path | str = BRIDGE_PATH
    node_binary: str = "node"
    observation_spec: ObservationSpec = DEFAULT_REPLAY_OBSERVATION_SPEC
    # Ablation-arm feature masks (config, not spec): masked-off blocks are zeroed +
    # attention-masked at encode time. Callers pairing the env with a model must keep these
    # consistent with the model config's stats_block_enabled / exact_state_enabled /
    # transition_token_budget fields.
    feature_masks: ObservationFeatureMasks = DEFAULT_OBSERVATION_FEATURE_MASKS
    # Category vocabulary used to convert token strings to embedding rows. When None it is built
    # from showdown_root; callers that pair the env with a specific model MUST pass the model's
    # vocabulary here so encode-time rows match the embedding exactly (no silent row drift).
    category_vocab: "CategoryVocabulary | None" = None
    read_timeout_seconds: float = 10.0
    # Whether the belief engine narrows opponent candidate sets via the Gen 3 randbats set source
    # (populates possible_moves / candidate_variants / possible ability+item). None defers to the
    # POKEZERO_BELIEF_SET_SOURCE env var so training and eval images flip together from one place;
    # set explicitly (True/False) to pin it (e.g. in tests). Revealed moves/ability/item do NOT
    # depend on this — they come straight from the protocol.
    set_belief_source: bool | None = None

    def resolved_showdown_root(self) -> Path:
        configured = self.showdown_root or os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT
        return Path(configured).expanduser().resolve()

    def resolved_bridge_path(self) -> Path:
        return Path(self.bridge_path).expanduser().resolve()

    def belief_set_source_enabled(self) -> bool:
        if self.set_belief_source is not None:
            return self.set_belief_source
        return belief_set_source_env_enabled()


@dataclass(frozen=True)
class LocalShowdownSnapshot:
    """Restorable simulator plus local public-state snapshot for a live bridge battle."""

    battle_token: str
    battle_id: str
    format_id: BattleFormat
    observation_format_id: BattleFormat
    bridge_snapshot: Mapping[str, Any]
    protocol_lines: tuple[str, ...]
    latest_requests: Mapping[PlayerId, Mapping[str, Any]]
    latest_turn: int
    terminal: TerminalState | None


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
        self._observation_format_id: BattleFormat = self._format_id
        self._lines: list[str] = []
        self._latest_requests: dict[PlayerId, Mapping[str, Any]] = {}
        self._latest_turn = 0
        self._terminal: TerminalState | None = None
        self._last_step_had_error = False
        # Persistent incremental state: the parser + belief engine are fed each new protocol line
        # / event exactly once (see _sync_incremental_state), so observations cost O(state) instead
        # of re-parsing and re-ingesting the whole accumulated log every call (O(n^2) per battle).
        self._parser = _ReplayParser(self._battle_id)
        # Shared, immutable candidate-set source (built once per process, cached). None when the
        # belief set source is disabled, in which case only protocol-revealed facts populate.
        self._belief_set_source = (
            load_gen3_randbat_source_cached(self.config.resolved_showdown_root())
            if self.config.belief_set_source_enabled()
            else None
        )
        self._belief_engine = PublicBattleBeliefEngine(
            format_id=self._observation_format_id, set_source=self._belief_set_source
        )
        self._parsed_line_count = 0
        self._belief_fed_count = 0
        # Tier-2 live residual trackers (#505 follow-up): one per perspective, created
        # lazily once that player's first request arrives (it carries the exact own-team
        # stats the residual math needs). Active only when the encode-time masks keep the
        # channel on AND the candidate-set source is enabled — mask-off arms and
        # pre-#505 checkpoints (whose provenance latches tier2_residuals=False) pay
        # nothing and encode byte-identically.
        self._first_requests: dict[PlayerId, Mapping[str, Any]] = {}
        self._tier2_trackers: dict[PlayerId, Tier2LiveTracker] = {}
        # Defender-side investment trackers (v2.1 batch 2): same lazy per-perspective
        # pattern, active only under the tier2 channel AND the tier2_investment mask
        # (default off — see ObservationFeatureMasks).
        self._investment_trackers: dict[PlayerId, InvestmentLiveTracker] = {}
        # Warm pool: the bridge process is reused across battles. Each battle gets a unique routing
        # token; events from a prior battle carry a stale token and are ignored (see _apply_event).
        self._battle_counter = 0
        self._battle_token: str | None = None

    @property
    def belief_set_source_hash(self) -> str | None:
        """Provenance hash of the candidate-set source encoding observations (None when disabled)."""
        if self._belief_set_source is None:
            return None
        return self._belief_set_source.metadata.source_hash

    @property
    def protocol_lines(self) -> tuple[str, ...]:
        return tuple(self._lines)

    def reset(self, *, seed: int, format_id: BattleFormat = "gen3randombattle") -> None:
        self._reset(seed=seed, format_id=format_id, start_override=None)

    def reset_with_start_override(
        self,
        *,
        seed: int,
        format_id: BattleFormat | None = None,
        start_override: BattleStartOverride,
    ) -> None:
        effective_format_id = start_override.format_id if format_id is None else str(format_id)
        if effective_format_id != start_override.format_id:
            raise ValueError(
                "reset_with_start_override format_id must match "
                f"start_override.format_id {start_override.format_id!r}."
            )
        self._reset(seed=seed, format_id=effective_format_id, start_override=start_override)

    def _reset(
        self,
        *,
        seed: int,
        format_id: BattleFormat = "gen3randombattle",
        start_override: BattleStartOverride | None,
    ) -> None:
        previous_token = self._battle_token
        self._battle_id = f"local-{format_id}-{seed}"
        self._format_id = format_id
        self._observation_format_id = (
            str(start_override.observation_format_id)
            if start_override is not None and start_override.observation_format_id is not None
            else format_id
        )
        self._battle_counter += 1
        self._battle_token = f"b{self._battle_counter}"
        self._lines = []
        self._latest_requests = {}
        self._latest_turn = 0
        self._terminal = None
        self._last_step_had_error = False
        self._parser = _ReplayParser(self._battle_id)
        self._belief_engine = PublicBattleBeliefEngine(
            format_id=self._observation_format_id, set_source=self._belief_set_source
        )
        self._parsed_line_count = 0
        self._belief_fed_count = 0
        self._first_requests = {}
        self._tier2_trackers = {}
        self._investment_trackers = {}
        # Reuse a live bridge process across battles (warm pool); only spawn when there is none or
        # the previous one died. Stale events from the prior battle carry previous_token and are
        # ignored by _apply_event, so a clean queue drain is not required.
        reuse = self._process is not None and self._process.poll() is None
        if not reuse:
            self.close()  # clean up a dead process / drain threads, then spawn fresh
            self._validate_runtime()
            self._start_bridge()
        elif previous_token is not None:
            self._send_command({"type": "end", "battleId": previous_token})
        try:
            self._send_command(
                {
                    "type": "start",
                    "battleId": self._battle_token,
                    "formatid": format_id,
                    "seed": showdown_seed_from_int(seed),
                    "players": _start_players_payload(start_override),
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
        # Prefer the explicitly-paired model vocabulary; otherwise build it from the root.
        # A v2.2 (turn-merged) spec needs the tt_phase/tt2_* families or every merged
        # label would land in the OOV band.
        vocab = self.config.category_vocab or gen3_category_vocabulary(
            root,
            include_turn_merged=(
                self.config.observation_spec.schema_version == OBSERVATION_SCHEMA_VERSION_V2_2
            ),
        )
        observation = observation_from_player_state(
            state,
            category_vocab=vocab,
            spec=self.config.observation_spec,
            dex=load_showdown_dex_cached(root),
            feature_masks=self.config.feature_masks,
        )
        # The belief view is derived from the same public protocol transcript as
        # the observation. Keeping it in metadata makes public-corpus capture
        # consistent across fixed-driver and controlled FoulPlay games without
        # exposing either player's request payload.
        return replace(
            observation,
            metadata={**dict(observation.metadata), "belief_view": state.belief_view.to_overlay_payload()},
        )

    def legal_actions(self, player: PlayerId) -> tuple[bool, ...]:
        return self.observe(player).legal_action_mask

    def snapshot(self) -> LocalShowdownSnapshot:
        """Capture a restorable snapshot of the current live battle.

        The snapshot includes the Node simulator state plus the Python-side protocol parser inputs.
        It is an oracle simulator snapshot; hidden-info callers must not use it as a replacement for
        explicit belief sampling.
        """

        if self._battle_token is None:
            raise LocalShowdownError("Cannot snapshot before reset.")
        self._send_command({"type": "snapshot", "battleId": self._battle_token})
        event = self._read_until_event_type("snapshot")
        snapshot = event.get("snapshot")
        if not isinstance(snapshot, Mapping):
            raise LocalShowdownError(f"Bridge emitted malformed snapshot event: {event!r}")
        return LocalShowdownSnapshot(
            battle_token=self._battle_token,
            battle_id=self._battle_id,
            format_id=self._format_id,
            observation_format_id=self._observation_format_id,
            bridge_snapshot=_json_clone_mapping(snapshot),
            protocol_lines=tuple(self._lines),
            latest_requests=_json_clone_requests(self._latest_requests),
            latest_turn=self._latest_turn,
            terminal=self._terminal,
        )

    def restore(self, snapshot: LocalShowdownSnapshot) -> None:
        """Restore a snapshot captured from this live bridge battle."""

        if self._battle_token is None:
            raise LocalShowdownError("Cannot restore before reset.")
        if snapshot.battle_token != self._battle_token:
            raise ValueError("LocalShowdownSnapshot can only be restored into its original live battle.")
        self._send_command(
            {
                "type": "restore",
                "battleId": self._battle_token,
                "snapshot": snapshot.bridge_snapshot,
            }
        )
        self._read_until_event_type("restored")
        self._battle_id = snapshot.battle_id
        self._format_id = snapshot.format_id
        self._observation_format_id = snapshot.observation_format_id
        self._lines = list(snapshot.protocol_lines)
        self._latest_requests = _json_clone_requests(snapshot.latest_requests)
        # Trackers rebuild lazily from the restored line prefix; the earliest request in
        # the restored snapshot stands in for the battle's first (own-team stats are
        # immutable within a battle, which is all the trackers read from it).
        self._first_requests = dict(self._latest_requests)
        self._tier2_trackers = {}
        self._investment_trackers = {}
        self._latest_turn = snapshot.latest_turn
        self._terminal = snapshot.terminal
        self._last_step_had_error = False
        self._parser = _ReplayParser(self._battle_id)
        self._belief_engine = PublicBattleBeliefEngine(
            format_id=self._observation_format_id, set_source=self._belief_set_source
        )
        self._parsed_line_count = 0
        self._belief_fed_count = 0

    def reseed_simulator_rng(self, seed: int) -> None:
        """Reset Showdown's battle PRNG at the current simulator state."""

        if self._battle_token is None:
            raise LocalShowdownError("Cannot reseed before reset.")
        showdown_seed = showdown_seed_from_int(seed)
        self._send_command(
            {
                "type": "reseed",
                "battleId": self._battle_token,
                "seed": showdown_seed,
            }
        )
        self._read_until_event_type("reseeded")

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
        choices = {}
        for player in requested:
            try:
                choices[player] = showdown_choice_for_action(states[player], actions[player])
            except ValueError as exc:
                raise ValueError(f"{player}: {exc}") from exc
        self._send_command({"type": "choices", "battleId": self._battle_token, "choices": choices})
        self._read_until_boundary()
        if self._last_step_had_error:
            raise LocalShowdownError("Showdown rejected a submitted choice.")

        next_requested = self.requested_players()
        observations = {player: self.observe(player) for player in next_requested}
        rewards = self._rewards()
        terminal = self.terminal()
        # On terminal we leave the bridge process alive (warm pool): the finished battle is freed
        # by the next reset()'s "end" command, or by close() on shutdown. This avoids a node
        # respawn per game.
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

    def _read_until_event_type(self, event_type: str) -> Mapping[str, Any]:
        deadline = time.monotonic() + self.config.read_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalShowdownError(self._timeout_message())
            event = self._read_event(timeout=remaining)
            if event is None:
                continue
            if event.get("type") == "error":
                raise LocalShowdownError(str(event.get("message") or "Bridge error."))
            battle_id = event.get("battleId")
            if battle_id is not None and self._battle_token is not None and battle_id != self._battle_token:
                continue
            if event.get("type") == event_type:
                return event
            self._apply_event(event)

    def _apply_event(self, event: Mapping[str, Any]) -> bool:
        event_type = event.get("type")
        # On a reused (warm) process, events from a finished battle still drain through the queue;
        # they carry that battle's routing token, so ignore anything not for the current battle.
        # Global events (process-level errors, "closed") carry no battleId and are not filtered.
        battle_id = event.get("battleId")
        if battle_id is not None and self._battle_token is not None and battle_id != self._battle_token:
            return False
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
                        self._first_requests.setdefault(stream, request)
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

    def _sync_incremental_state(self) -> None:
        """Feed newly-appended protocol lines to the persistent parser and belief engine once."""
        if len(self._lines) > self._parsed_line_count:
            self._parser.feed(self._lines[self._parsed_line_count :])
            self._parsed_line_count = len(self._lines)
        events = self._parser.public_events
        if len(events) > self._belief_fed_count:
            for event in events[self._belief_fed_count :]:
                self._belief_engine.ingest_event(event)
            self._belief_fed_count = len(events)

    def _state_for_player(self, player: PlayerId) -> PlayerRelativeBattleState:
        if player not in PLAYER_IDS:
            raise ValueError(f"player must be one of {', '.join(PLAYER_IDS)}; got {player!r}.")
        self._sync_incremental_state()
        replay = self._parser.snapshot()
        # v2.2 (turn-merged) specs need the merged stream populated alongside the
        # per-action one (which stays the Tier-2 annotation substrate + pinned-bit source).
        turn_merged = (
            self.config.observation_spec.schema_version == OBSERVATION_SCHEMA_VERSION_V2_2
        )
        state = normalize_for_player(
            replay,
            player_id=player,
            configured_showdown_slot=player,
            format_id=self._observation_format_id,
            belief_engine=self._belief_engine,
            include_turn_merged=turn_merged,
        )
        tracker = self._tier2_tracker_for(player)
        if tracker is not None:
            state = replace(
                state,
                transition_tokens=tracker.annotate(
                    replay, state.transition_tokens, self._belief_engine
                ),
            )
        investment_tracker = self._investment_tracker_for(player)
        if investment_tracker is not None:
            codes = investment_tracker.observe(
                replay, state.transition_tokens, self._belief_engine
            )
            if codes:
                state = replace(
                    state,
                    transition_tokens=tuple(
                        replace(token, investment=codes[index]) if index in codes else token
                        for index, token in enumerate(state.transition_tokens)
                    ),
                )
        # v2.2: map the FINAL annotated per-action stream (tier2 residual/CB +
        # investment codes) onto the merged sub-blocks; the per-action stream stays
        # the annotation substrate and the per-mon pinned-surface derivation source.
        if turn_merged and (tracker is not None or investment_tracker is not None):
            from .turn_merged import annotate_turn_merged_tokens

            state = replace(
                state,
                turn_merged_tokens=annotate_turn_merged_tokens(
                    state.turn_merged_tokens, state.transition_tokens
                ),
            )
        return state

    def tier2_residuals_active(self) -> bool:
        """Whether this env populates Tier-2 residuals into transition tokens.

        Requires both the encode-time mask (checkpoint-latched via
        ``env_config_with_checkpoint_masks``) AND the candidate-set source — without
        candidate variants every strike is unassessable, so the tracker is skipped
        outright and encodes stay byte-identical to a pre-#505 pipeline.
        """
        return bool(self.config.feature_masks.tier2_residuals) and self._belief_set_source is not None

    def _tier2_tracker_for(self, player: PlayerId) -> Tier2LiveTracker | None:
        if not self.tier2_residuals_active():
            return None
        tracker = self._tier2_trackers.get(player)
        if tracker is not None:
            return tracker
        request = self._first_requests.get(player) or self._latest_requests.get(player)
        if request is None:
            return None
        own_team = own_team_from_request(request)
        if not own_team:
            return None
        root = self.config.resolved_showdown_root()
        dex = load_showdown_dex_cached(root)
        tracker = Tier2LiveTracker(
            perspective_slot=player,
            own_team=own_team,
            dex=dex,
            whitelist=cb_whitelist_for_source(self._belief_set_source, dex),
        )
        self._tier2_trackers[player] = tracker
        return tracker

    def investment_active(self) -> bool:
        """Whether this env populates defender-side investment codes into tokens.

        Requires the tier2 channel (mask + candidate-set source) AND the separate
        tier2_investment provenance mask — default off, so existing pipelines encode
        byte-identically until v2.1 training adopts the column.
        """
        return self.tier2_residuals_active() and bool(self.config.feature_masks.tier2_investment)

    def _investment_tracker_for(self, player: PlayerId) -> InvestmentLiveTracker | None:
        if not self.investment_active():
            return None
        tracker = self._investment_trackers.get(player)
        if tracker is not None:
            return tracker
        request = self._first_requests.get(player) or self._latest_requests.get(player)
        if request is None:
            return None
        own_team = own_team_from_request(request)
        if not own_team:
            return None
        dex = load_showdown_dex_cached(self.config.resolved_showdown_root())
        tracker = InvestmentLiveTracker(
            perspective_slot=player,
            own_team=own_team,
            dex=dex,
        )
        self._investment_trackers[player] = tracker
        return tracker

    def _winner_slot(self, winner_name: str) -> PlayerId | None:
        self._sync_incremental_state()
        for slot, name in self._parser.players.items():
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


def _start_players_payload(start_override: BattleStartOverride | None) -> dict[PlayerId, str | dict[str, str]]:
    player_teams = start_override.player_teams if start_override is not None else {}
    players: dict[PlayerId, str | dict[str, str]] = {}
    for player in PLAYER_IDS:
        name = _DEFAULT_PLAYER_NAMES[player]
        team = player_teams.get(player)
        players[player] = {"name": name, "team": team} if team else name
    return players


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


def _json_clone_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    cloned = json.loads(json.dumps(value, separators=(",", ":")))
    if not isinstance(cloned, Mapping):
        raise ValueError("expected JSON object clone.")
    return cloned


def _json_clone_requests(
    value: Mapping[PlayerId, Mapping[str, Any]],
) -> dict[PlayerId, Mapping[str, Any]]:
    cloned = _json_clone_mapping(value)
    return {player: request for player in PLAYER_IDS if isinstance((request := cloned.get(player)), Mapping)}


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
