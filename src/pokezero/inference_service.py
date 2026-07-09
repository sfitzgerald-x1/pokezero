"""WS-L1 batched GPU inference service.

Serves a transformer checkpoint's *forward* to remote collectors so the expensive 50M
model forward runs once on a GPU (bf16, batched) instead of per-collector on CPU. The
service owns ONLY the forward: the collector-side policy (``TransformerSoftmaxPolicy`` with
an injected ``forward_fn``) keeps the entire decision path — per-player history, tensorize,
legal-action masking, rng sampling, behavior-probability, value — so served-vs-local parity
is structural and determinism (sampling stays client-side) is untouched.

Requests are dynamically batched (see _BatchingForwarder) so N concurrent collector requests
cost one GPU forward. Wire protocol (JSON over HTTP; a raw-bytes payload is a possible follow-up):
  GET  /config  -> {"window_size": int, "policy_id": str, "action_count": int}
  POST /forward -> body is the observation-window tensors (nested lists, leading batch dim 1):
                   {categorical_ids, numeric_features, token_type_ids, attention_mask, history_mask}
                -> {"policy_logits": [[...]], "value": [...], "opponent_action_logits": [[...]] | null}
"""

from __future__ import annotations

import contextlib
import http.client
import json
import queue
import random
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy

from .actions import ACTION_COUNT
from .neural_policy import (
    TransformerPolicyOutput,
    TransformerSoftmaxPolicy,
    load_transformer_policy,
    require_torch,
)

_FORWARD_TENSOR_KEYS = (
    "categorical_ids",
    "numeric_features",
    "token_type_ids",
    "attention_mask",
    "history_mask",
)

# A collector fleet can start hundreds of clients immediately after the inference server becomes
# ready. Retry transient connection failures locally so that short-lived socket admission pressure
# does not burn an entire Job-index retry. This affects request timing only, never policy outputs.
_REMOTE_RETRY_ATTEMPTS = 6
_REMOTE_RETRY_INITIAL_BACKOFF_S = 0.05
_REMOTE_RETRY_MAX_BACKOFF_S = 0.8
_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class _RetryableInferenceResponse(RuntimeError):
    """Marks a transient HTTP response that should use the bounded client retry path."""


def _retry_backoff_seconds(attempt: int) -> float:
    """Return exponential backoff with bounded jitter to desynchronize a collector startup burst."""
    base = min(_REMOTE_RETRY_INITIAL_BACKOFF_S * (2**attempt), _REMOTE_RETRY_MAX_BACKOFF_S)
    return base * random.uniform(0.75, 1.25)


def _sleep_before_remote_retry(attempt: int) -> None:
    time.sleep(_retry_backoff_seconds(attempt))


def _autocast_context(torch_module: Any, device: Any, amp: str | None):
    if amp != "bf16":
        return contextlib.nullcontext()
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    return torch_module.autocast(device_type=device_type, dtype=torch_module.bfloat16)


def encode_forward_request(tensors: Mapping[str, Any]) -> bytes:
    """Serialize the 5 window tensors as a 4-byte header length + JSON header (shapes/dtypes) +
    raw little-endian array bytes. Avoids per-float JSON encode/decode of the ~100 KB
    numeric_features (the measured bottleneck: server GPU sat at 2% while collect crawled)."""
    header: dict[str, Any] = {"__order__": list(_FORWARD_TENSOR_KEYS)}
    chunks: list[bytes] = []
    for key in _FORWARD_TENSOR_KEYS:
        arr = numpy.ascontiguousarray(tensors[key].detach().cpu().numpy())
        header[key] = {"shape": list(arr.shape), "dtype": arr.dtype.str}
        chunks.append(arr.tobytes())
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack(">I", len(header_bytes)) + header_bytes + b"".join(chunks)


def decode_forward_request(body: bytes) -> "dict[str, Any]":
    """Inverse of encode_forward_request → {key: numpy array with leading batch dim}."""
    (header_len,) = struct.unpack(">I", body[:4])
    header = json.loads(body[4 : 4 + header_len].decode("utf-8"))
    offset = 4 + header_len
    out: dict[str, Any] = {}
    for key in header["__order__"]:
        meta = header[key]
        dtype = numpy.dtype(meta["dtype"])
        shape = tuple(meta["shape"])
        count = int(numpy.prod(shape)) if shape else 1
        nbytes = count * dtype.itemsize
        # .copy() → writable + contiguous (frombuffer is read-only, which torch dislikes)
        out[key] = numpy.frombuffer(body[offset : offset + nbytes], dtype=dtype).reshape(shape).copy()
        offset += nbytes
    return out


def _forward_batch(
    policy: TransformerSoftmaxPolicy,
    payloads: "list[Mapping[str, Any]]",
    *,
    device: Any = None,
    amp: str | None = None,
) -> "list[dict[str, Any]]":
    """Run ONE batched model forward for N request payloads and return one result dict each.

    Each payload's tensors carry a leading batch dim 1 (from the client's
    observation_window_to_torch); we concatenate along that dim into a [B, …] batch, run a single
    forward, then split back per request — so B concurrent collector requests cost one GPU call.
    fp32 is emitted on the wire even under bf16 (the collector's softmax/masking/sampling stay
    fp32, identical to local). Batch-of-1 is numerically identical to a single forward → parity holds.
    """
    torch_module = require_torch()

    def _stack(key: str, dtype: Any) -> Any:
        values = [p[key] for p in payloads]  # each carries leading batch dim 1
        if isinstance(values[0], numpy.ndarray):
            # binary path: concat [1,…] arrays → [B,…] and adopt as one tensor (fast, no per-float parse)
            stacked = numpy.concatenate(values, axis=0)
            return torch_module.as_tensor(stacked).to(dtype=dtype, device=device)
        # list path (tests): nested Python lists
        return torch_module.tensor([v[0] for v in values], dtype=dtype, device=device)

    tensors = {
        "categorical_ids": _stack("categorical_ids", torch_module.long),
        "numeric_features": _stack("numeric_features", torch_module.float32),
        "token_type_ids": _stack("token_type_ids", torch_module.long),
        "attention_mask": _stack("attention_mask", torch_module.bool),
        "history_mask": _stack("history_mask", torch_module.bool),
    }
    with torch_module.no_grad(), _autocast_context(torch_module, device, amp):
        output = policy._default_forward(tensors)
    policy_logits = output.policy_logits.float().cpu().tolist()
    value = output.value.float().cpu().tolist()
    opp = getattr(output, "opponent_action_logits", None)
    opp_logits = None if opp is None else opp.float().cpu().tolist()
    results: list[dict[str, Any]] = []
    for i in range(len(payloads)):
        results.append(
            {
                "policy_logits": [policy_logits[i]],
                "value": [value[i]],
                "opponent_action_logits": None if opp_logits is None else [opp_logits[i]],
            }
        )
    return results


def run_forward_from_payload(
    policy: TransformerSoftmaxPolicy,
    payload: Mapping[str, Any],
    *,
    device: Any = None,
    amp: str | None = None,
) -> dict[str, Any]:
    """Single-payload forward (batch-of-1). Kept for in-process parity tests."""
    return _forward_batch(policy, [payload], device=device, amp=amp)[0]


class _BatchingForwarder:
    """Dynamic-batching front-end: coalesces concurrent /forward requests within a small time
    window into ONE GPU forward, so 64 collectors don't each pay a batch-1 GPU call + serialize
    on the server. A background thread drains the queue; each request waits on its own Event."""

    def __init__(
        self,
        policy: TransformerSoftmaxPolicy,
        *,
        device: Any,
        amp: str | None,
        max_batch: int = 64,
        max_delay_s: float = 0.010,
    ) -> None:
        self._policy = policy
        self._device = device
        self._amp = amp
        self._max_batch = max_batch
        self._max_delay = max_delay_s
        self._queue: "queue.Queue[tuple[Mapping[str, Any], dict[str, Any], threading.Event]]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_policy(self, policy: TransformerSoftmaxPolicy) -> None:
        # Atomic reference swap (GIL): in-flight batches finish on the old policy; the next batch
        # picks up the new one. This is the checkpoint hot-swap at the iteration boundary.
        self._policy = policy

    def submit(self, payload: Mapping[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
        slot: dict[str, Any] = {}
        event = threading.Event()
        self._queue.put((payload, slot, event))
        if not event.wait(timeout):
            raise TimeoutError("inference batch timed out")
        if "error" in slot:
            raise RuntimeError(slot["error"])
        return slot["result"]

    def _loop(self) -> None:
        while True:
            batch = [self._queue.get()]  # block for the first request
            deadline = time.monotonic() + self._max_delay
            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            payloads = [item[0] for item in batch]
            try:
                results = _forward_batch(self._policy, payloads, device=self._device, amp=self._amp)
                for (_, slot, event), result in zip(batch, results):
                    slot["result"] = result
                    event.set()
            except Exception as exc:  # noqa: BLE001 — fail the whole batch diagnosably, don't hang collectors
                for _, slot, event in batch:
                    slot["error"] = f"{type(exc).__name__}: {exc}"
                    event.set()


class RemoteForward:
    """A ``forward_fn`` for TransformerSoftmaxPolicy that RPCs the window tensors to an inference
    server and returns an equivalent TransformerPolicyOutput (fp32 torch tensors).

    Reuses ONE persistent HTTP connection (keep-alive) rather than opening a socket per forward —
    a collector does ~70 forwards/game, and 64 collectors each churning new connections overflows
    the server's listen backlog (Errno 104 connection-reset). Retries bounded transient transport
    failures with exponential backoff; permanent client responses still fail immediately. Guarded
    by a lock in case a pod runs games concurrently on the same policy."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        parsed = urlsplit(base_url if "://" in base_url else "http://" + base_url)
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._timeout = timeout
        self._lock = threading.Lock()
        self._conn: "http.client.HTTPConnection | None" = None

    def _request(self, body: bytes) -> dict[str, Any]:
        for attempt in range(_REMOTE_RETRY_ATTEMPTS):
            try:
                if self._conn is None:
                    self._conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
                self._conn.request("POST", "/forward", body=body, headers={"Content-Type": "application/octet-stream"})
                response = self._conn.getresponse()
                data = response.read()
                if response.status != 200:
                    if response.status in _RETRYABLE_HTTP_STATUSES:
                        raise _RetryableInferenceResponse(
                            f"inference server {response.status}: {data[:200]!r}"
                        )
                    raise RuntimeError(f"inference server {response.status}: {data[:200]!r}")
                return json.loads(data.decode("utf-8"))
            except (http.client.HTTPException, ConnectionError, OSError, _RetryableInferenceResponse):
                if self._conn is not None:
                    self._conn.close()
                self._conn = None
                if attempt == _REMOTE_RETRY_ATTEMPTS - 1:
                    raise
                _sleep_before_remote_retry(attempt)
        raise RuntimeError("unreachable")

    def __call__(self, tensors: Mapping[str, Any]) -> TransformerPolicyOutput:
        torch_module = require_torch()
        body = encode_forward_request(tensors)
        with self._lock:
            result = self._request(body)
        opp = result.get("opponent_action_logits")
        return TransformerPolicyOutput(
            policy_logits=torch_module.tensor(result["policy_logits"], dtype=torch_module.float32),
            value=torch_module.tensor(result["value"], dtype=torch_module.float32),
            opponent_action_logits=None if opp is None else torch_module.tensor(opp, dtype=torch_module.float32),
        )


class _StubModel:
    """Placeholder model for a remote policy: the forward is served remotely, so no weights
    are held client-side. Only .eval()/.to() are exercised by TransformerSoftmaxPolicy."""

    def eval(self) -> "_StubModel":
        return self

    def to(self, *args: Any, **kwargs: Any) -> "_StubModel":
        return self


def fetch_remote_config(base_url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    config_url = base_url.rstrip("/") + "/config"
    for attempt in range(_REMOTE_RETRY_ATTEMPTS):
        try:
            with urlopen(config_url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP_STATUSES or attempt == _REMOTE_RETRY_ATTEMPTS - 1:
                raise
        except (URLError, OSError):
            if attempt == _REMOTE_RETRY_ATTEMPTS - 1:
                raise
        _sleep_before_remote_retry(attempt)
    raise RuntimeError("unreachable")


def remote_inference_policy(base_url: str, **policy_options: Any) -> TransformerSoftmaxPolicy:
    """Build a collector-side policy whose forward is served by the inference server at
    ``base_url``. Fetches only window_size/policy_id from the server — no checkpoint weights."""
    config = fetch_remote_config(base_url)
    result = SimpleNamespace(
        model_config=SimpleNamespace(
            window_size=int(config["window_size"]),
            policy_id=config.get("policy_id"),
        )
    )
    return TransformerSoftmaxPolicy(
        model=_StubModel(),
        result=result,  # type: ignore[arg-type]
        forward_fn=RemoteForward(base_url),
        **policy_options,
    )


def build_request_handler(policy: TransformerSoftmaxPolicy, forwarder: "_BatchingForwarder", *, device: Any = None):
    window_size = int(policy.result.model_config.window_size)
    # Mutable so /config reflects the current checkpoint after a hot-swap. reload_lock serializes
    # concurrent /reload calls (the controller issues one per iteration boundary).
    state = {"policy_id": str(policy.policy_id)}
    reload_lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive so the persistent RemoteForward reuses the socket
        # Close idle kept-alive connections so a collector that dies ungracefully (OOM/evict/SIGKILL
        # — routine at 64 pods) doesn't leak a server thread+FD until OS TCP-keepalive (~2h). Set
        # above the client's 30s request / 60s batch timeout so it never cuts a live request.
        timeout = 65

        def log_message(self, *args: Any) -> None:  # silence per-request stderr spam
            pass

        def _send(self, code: int, obj: Any) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/config":
                self._send(200, {"window_size": window_size, "policy_id": state["policy_id"], "action_count": ACTION_COUNT})
            elif self.path.rstrip("/") in ("/health", "/healthz"):
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.rstrip("/")
            if path == "/forward":
                # Submit to the batcher (one bad item returns a diagnosable error, never hangs the
                # collector or takes down the batch).
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = decode_forward_request(self.rfile.read(length))
                    result = forwarder.submit(payload)
                except TimeoutError as exc:  # server overload, not a client error
                    self._send(504, {"error": f"TimeoutError: {exc}"})
                    return
                except Exception as exc:  # noqa: BLE001 — malformed/bad-shape body is a client error
                    self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
                    return
                self._send(200, result)
            elif path == "/reload":
                # Hot-swap the served checkpoint (controller calls this at each iteration boundary).
                # Load the new policy WITHOUT holding the batcher — forwards keep hitting the old
                # policy until the atomic set_policy swap, so there's no serving gap.
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    req = json.loads(self.rfile.read(length).decode("utf-8"))
                    checkpoint = str(req["checkpoint_path"])
                    with reload_lock:
                        new_policy = load_transformer_policy(Path(checkpoint), device=device)
                        new_window = int(new_policy.result.model_config.window_size)
                        if new_window != window_size:
                            raise ValueError(
                                f"reload window_size {new_window} != served {window_size}; "
                                "collectors tensorize at the served window — refusing arch-mismatched swap."
                            )
                        forwarder.set_policy(new_policy)
                        state["policy_id"] = str(new_policy.policy_id)
                except Exception as exc:  # noqa: BLE001
                    self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
                    return
                self._send(200, {"reloaded": checkpoint, "policy_id": state["policy_id"]})
            else:
                self._send(404, {"error": "not found"})

    return _Handler


def serve_inference(
    checkpoint_path: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8600,
    device: str | None = None,
    amp: str | None = None,
    max_batch: int = 64,
    batch_window_ms: float = 10.0,
) -> ThreadingHTTPServer:
    """Load a checkpoint and start a threaded, dynamically-batched HTTP inference server. Returns
    the server (call .serve_forever() to block, or use in a thread for tests)."""
    policy = load_transformer_policy(Path(checkpoint_path), device=device)
    forwarder = _BatchingForwarder(
        policy, device=device, amp=amp, max_batch=max_batch, max_delay_s=batch_window_ms / 1000.0
    )
    handler = build_request_handler(policy, forwarder, device=device)

    class _BatchedHTTPServer(ThreadingHTTPServer):
        # 64+ collectors connect near-simultaneously; the default listen backlog of 5 overflows
        # (Errno 104 connection-reset). Widen it. daemon_threads so workers don't block shutdown.
        request_queue_size = 512
        daemon_threads = True

    server = _BatchedHTTPServer((host, port), handler)
    return server


def serve_forever(checkpoint_path: str, **kwargs: Any) -> None:
    server = serve_inference(checkpoint_path, **kwargs)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
