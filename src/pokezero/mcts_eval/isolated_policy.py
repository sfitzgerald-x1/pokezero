"""Public-only subprocess policy adapter for source-different MCTS comparisons.

The live RolloutDriver and its Showdown environment remain in the parent
process. Each policy's search runs in a persistent child started by that
policy's declared Python/source build. This is a small transport adapter, not
a second rollout or evaluation framework.

Pickle is used only over inherited stdin/stdout pipes of a locally started
child. The host sanitises the context before serialisation; a child never sees
the other player's request observation or legal mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import random
import select
import struct
import subprocess
import math
from time import monotonic
from typing import Any, BinaryIO, Mapping, Sequence

from ..policy import PolicyContext, PolicyDecision
from .head_to_head import HeadToHeadError, MctsPolicySpec, public_only_context


PROTOCOL_VERSION = "pokezero.isolated-mcts-policy.v1"
_MAX_FRAME_BYTES = 64 * 1024 * 1024
_STATS_FIELDS = (
    "decisions",
    "searched_decisions",
    "fallback_decisions",
    "model_evals",
    "total_iterations",
    "worlds_constructed",
    "worlds_searched",
    "prior_fallbacks",
    "decision_wall_seconds",
)


class IsolatedPolicyError(HeadToHeadError):
    """A source-isolated policy worker cannot safely provide a decision."""


def write_frame(stream: BinaryIO, payload: Mapping[str, Any]) -> None:
    """Write one bounded pickle frame, flushing it before the next request."""

    encoded = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
    if not encoded or len(encoded) > _MAX_FRAME_BYTES:
        raise IsolatedPolicyError(
            f"isolated policy frame has invalid size {len(encoded)} bytes."
        )
    stream.write(struct.pack(">Q", len(encoded)))
    stream.write(encoded)
    stream.flush()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("isolated policy worker closed its protocol stream.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO) -> Mapping[str, Any]:
    """Read one bounded pickle frame and require a mapping payload."""

    size = struct.unpack(">Q", _read_exact(stream, 8))[0]
    if not size or size > _MAX_FRAME_BYTES:
        raise IsolatedPolicyError(f"isolated policy frame has invalid size {size} bytes.")
    try:
        payload = pickle.loads(_read_exact(stream, size))
    except (pickle.PickleError, EOFError) as error:
        raise IsolatedPolicyError(f"cannot decode isolated policy frame: {error}") from error
    if not isinstance(payload, Mapping):
        raise IsolatedPolicyError("isolated policy frame must decode to a mapping.")
    return payload


@dataclass(frozen=True)
class IsolatedPolicyLaunch:
    """Immutable local-process launch contract for one source-bound policy."""

    policy: MctsPolicySpec
    command: tuple[str, ...]
    worker_config: Mapping[str, Any]
    response_timeout_seconds: float = 60.0
    stderr_path: Path | None = None
    working_directory: Path | None = None

    def __post_init__(self) -> None:
        if not self.command or not all(str(part) for part in self.command):
            raise ValueError("isolated policy worker command must be non-empty.")
        if self.response_timeout_seconds <= 0:
            raise ValueError("isolated policy worker response timeout must be positive.")
        if not isinstance(self.worker_config, Mapping):
            raise TypeError("isolated policy worker config must be a mapping.")
        if self.working_directory is not None and not self.working_directory.is_dir():
            raise ValueError("isolated policy worker working_directory must be an existing directory.")


@dataclass
class IsolatedPolicyStats:
    """The monotonic telemetry surface consumed by PolicyTelemetry.capture."""

    decisions: int = 0
    searched_decisions: int = 0
    fallback_decisions: int = 0
    model_evals: int = 0
    total_iterations: int = 0
    worlds_constructed: int = 0
    worlds_searched: int = 0
    prior_fallbacks: int = 0
    decision_wall_seconds: float = 0.0

    def update(self, payload: Mapping[str, Any]) -> None:
        for field_name in _STATS_FIELDS:
            if field_name not in payload:
                raise IsolatedPolicyError(
                    f"isolated policy worker omitted telemetry field {field_name!r}."
                )
            value = payload[field_name]
            if field_name == "decision_wall_seconds":
                try:
                    value = float(value)
                except (TypeError, ValueError) as error:
                    raise IsolatedPolicyError(
                        "isolated policy worker reported a non-numeric decision wall time."
                    ) from error
                if not math.isfinite(value) or value < 0:
                    raise IsolatedPolicyError(
                        "isolated policy worker reported an invalid decision wall time."
                    )
            elif not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise IsolatedPolicyError(
                    f"isolated policy worker reported invalid {field_name}={value!r}."
                )
            previous = getattr(self, field_name)
            if value < previous:
                raise IsolatedPolicyError(
                    f"isolated policy telemetry regressed for {field_name}: "
                    f"{previous!r} -> {value!r}."
                )
            setattr(self, field_name, value)


def snapshot_annotation_source(source: Any | None, *, player_id: str) -> dict[str, Any]:
    """Serialise only the current public Tier-2 overlay needed by search."""

    if source is None:
        return {"active": False, "overlay": {}}
    active = getattr(source, "active", None)
    if not callable(active):
        raise IsolatedPolicyError("isolated policy annotation source has no active() method.")
    if not bool(active()):
        return {"active": False, "overlay": {}}
    overlay_for = getattr(source, "overlay_for", None)
    if not callable(overlay_for):
        raise IsolatedPolicyError(
            "active isolated policy annotation source has no overlay_for() method."
        )
    overlay = overlay_for(player_id)
    if not isinstance(overlay, Mapping):
        raise IsolatedPolicyError("isolated policy annotation overlay is not a mapping.")
    copied: dict[int, tuple[Any, ...]] = {}
    for index, value in overlay.items():
        if (
            not isinstance(index, int)
            or index < 0
            or isinstance(value, (str, bytes, bytearray))
            or not isinstance(value, Sequence)
        ):
            raise IsolatedPolicyError("isolated policy annotation overlay has an invalid entry.")
        copied[index] = tuple(value)
    return {"active": True, "overlay": copied}


@dataclass
class IsolatedMctsPolicy:
    """A context-aware policy backed by one persistent local worker process."""

    launch: IsolatedPolicyLaunch
    annotation_source: Any | None = None
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _stderr_handle: BinaryIO | None = field(default=None, init=False, repr=False)
    _stats: IsolatedPolicyStats = field(default_factory=IsolatedPolicyStats, init=False)
    _closed: bool = field(default=False, init=False)
    worker_receipt: Mapping[str, Any] | None = field(default=None, init=False)

    @property
    def policy_id(self) -> str:
        return self.launch.policy.policy_id

    @property
    def stats(self) -> IsolatedPolicyStats:
        return self._stats

    @property
    def requires_public_materialization_state(self) -> bool:
        return True

    @property
    def is_source_isolated(self) -> bool:
        """Marker required before the paired runner waives in-process identity."""

        return True

    def select_action(self, observation: Any, *, rng: random.Random) -> PolicyDecision:
        raise IsolatedPolicyError(
            "source-isolated MCTS requires public materialisation context; "
            "refusing the context-free policy path."
        )

    def select_action_with_context(
        self, context: PolicyContext, *, rng: random.Random
    ) -> PolicyDecision:
        if self._closed:
            raise IsolatedPolicyError("source-isolated MCTS worker is already closed.")
        if context.public_materialization_state is None:
            raise IsolatedPolicyError(
                "source-isolated MCTS requires public_materialization_state."
            )
        sanitized = public_only_context(context)
        response = self._request(
            {
                "type": "decide",
                "context": sanitized,
                "rng_state": rng.getstate(),
                "annotation": snapshot_annotation_source(
                    self.annotation_source, player_id=context.player_id
                ),
            }
        )
        if response.get("type") != "decision":
            self._raise_worker_response(response, expected="decision")
        decision = _decision_from_payload(response.get("decision"), policy=self.launch.policy)
        legal_mask = tuple(context.observation.legal_action_mask)
        if decision.action_index >= len(legal_mask) or not legal_mask[decision.action_index]:
            raise IsolatedPolicyError(
                "source-isolated MCTS worker selected an action not legal in the host request."
            )
        telemetry = response.get("stats")
        if not isinstance(telemetry, Mapping):
            raise IsolatedPolicyError("isolated policy worker omitted telemetry.")
        self._stats.update(telemetry)
        return decision

    def reset(self) -> None:
        if self._closed:
            return
        response = self._request({"type": "reset"})
        if response.get("type") != "reset":
            self._raise_worker_response(response, expected="reset")

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process is not None and self._process.poll() is None:
                try:
                    self._request({"type": "close"})
                except IsolatedPolicyError:
                    pass
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=5)
        finally:
            if self._process is not None:
                for stream in (self._process.stdin, self._process.stdout):
                    if stream is not None:
                        stream.close()
            if self._stderr_handle is not None:
                self._stderr_handle.close()
            self._closed = True

    def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        process = self._ensure_started()
        if process.stdin is None or process.stdout is None:
            raise IsolatedPolicyError("isolated policy worker has no binary protocol pipes.")
        try:
            write_frame(process.stdin, payload)
            response = self._receive(process)
        except (BrokenPipeError, EOFError, OSError) as error:
            raise IsolatedPolicyError(
                f"isolated policy worker protocol failed: {error}{self._worker_exit_detail(process)}"
            ) from error
        return response

    def _ensure_started(self) -> subprocess.Popen[bytes]:
        if self._process is not None:
            return self._process
        stderr: Any = subprocess.DEVNULL
        if self.launch.stderr_path is not None:
            self.launch.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_handle = self.launch.stderr_path.open("xb")
            stderr = self._stderr_handle
        try:
            process = subprocess.Popen(
                self.launch.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                cwd=(str(self.launch.working_directory) if self.launch.working_directory else None),
                close_fds=True,
            )
        except OSError as error:
            raise IsolatedPolicyError(
                f"cannot start source-isolated MCTS worker: {error}"
            ) from error
        self._process = process
        response = self._request_start(process)
        if response.get("type") != "hello":
            self._raise_worker_response(response, expected="hello")
        receipt = response.get("receipt")
        if not isinstance(receipt, Mapping):
            raise IsolatedPolicyError("isolated policy worker hello has no provenance receipt.")
        if dict(receipt.get("policy", {})) != self.launch.policy.to_payload():
            raise IsolatedPolicyError(
                "isolated policy worker provenance receipt does not match its declared policy."
            )
        self.worker_receipt = dict(receipt)
        return process

    def _request_start(self, process: subprocess.Popen[bytes]) -> Mapping[str, Any]:
        if process.stdin is None:
            raise IsolatedPolicyError("isolated policy worker has no input pipe.")
        write_frame(
            process.stdin,
            {
                "type": "start",
                "protocol_version": PROTOCOL_VERSION,
                "policy": self.launch.policy.to_payload(),
                "worker_config": dict(self.launch.worker_config),
            },
        )
        return self._receive(process)

    def _receive(self, process: subprocess.Popen[bytes]) -> Mapping[str, Any]:
        if process.stdout is None:
            raise IsolatedPolicyError("isolated policy worker has no output pipe.")
        deadline = monotonic() + self.launch.response_timeout_seconds
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise IsolatedPolicyError(
                    "timed out waiting for source-isolated MCTS worker response"
                    + self._worker_exit_detail(process)
                )
            readable, _, _ = select.select([process.stdout], [], [], remaining)
            if readable:
                return read_frame(process.stdout)
            if process.poll() is not None:
                raise IsolatedPolicyError(
                    "source-isolated MCTS worker exited before responding"
                    + self._worker_exit_detail(process)
                )

    def _worker_exit_detail(self, process: subprocess.Popen[bytes]) -> str:
        code = process.poll()
        return "" if code is None else f" (exit code {code})"

    @staticmethod
    def _raise_worker_response(response: Mapping[str, Any], *, expected: str) -> None:
        if response.get("type") == "error":
            message = str(response.get("message", "unspecified worker error"))
            raise IsolatedPolicyError(f"source-isolated MCTS worker refused: {message}")
        raise IsolatedPolicyError(
            f"source-isolated MCTS worker returned {response.get('type')!r}; "
            f"expected {expected!r}."
        )


def _decision_from_payload(payload: object, *, policy: MctsPolicySpec) -> PolicyDecision:
    if not isinstance(payload, Mapping):
        raise IsolatedPolicyError("isolated policy worker decision is not a mapping.")
    if str(payload.get("policy_id", "")) != policy.policy_id:
        raise IsolatedPolicyError(
            "isolated policy worker decision policy_id does not match its declared policy."
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise IsolatedPolicyError("isolated policy worker decision metadata is not a mapping.")
    try:
        return PolicyDecision(
            action_index=int(payload.get("action_index")),
            policy_id=policy.policy_id,
            action_probability=payload.get("action_probability"),
            value_estimate=payload.get("value_estimate"),
            metadata=dict(metadata),
        )
    except (TypeError, ValueError) as error:
        raise IsolatedPolicyError(
            f"isolated policy worker returned an invalid decision: {error}"
        ) from error
