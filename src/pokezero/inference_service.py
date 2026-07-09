"""WS-L1 batched GPU inference service.

Serves a transformer checkpoint's *forward* to remote collectors so the expensive 50M
model forward runs once on a GPU (bf16, batched) instead of per-collector on CPU. The
service owns ONLY the forward: the collector-side policy (``TransformerSoftmaxPolicy`` with
an injected ``forward_fn``) keeps the entire decision path — per-player history, tensorize,
legal-action masking, rng sampling, behavior-probability, value — so served-vs-local parity
is structural and determinism (sampling stays client-side) is untouched.

Wire protocol (JSON over HTTP; raw-bytes/dynamic-batching optimization is a follow-up):
  GET  /config  -> {"window_size": int, "policy_id": str, "action_count": int}
  POST /forward -> body is the observation-window tensors (nested lists, leading batch dim 1):
                   {categorical_ids, numeric_features, token_type_ids, attention_mask, history_mask}
                -> {"policy_logits": [[...]], "value": [...], "opponent_action_logits": [[...]] | null}
"""

from __future__ import annotations

import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.request import Request, urlopen

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


def _autocast_context(torch_module: Any, device: Any, amp: str | None):
    if amp != "bf16":
        return contextlib.nullcontext()
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    return torch_module.autocast(device_type=device_type, dtype=torch_module.bfloat16)


def run_forward_from_payload(
    policy: TransformerSoftmaxPolicy,
    payload: Mapping[str, Any],
    *,
    device: Any = None,
    amp: str | None = None,
) -> dict[str, Any]:
    """Rebuild the window tensors from a request payload, run the model forward, and return
    logits/value as plain lists. Shared by the HTTP server and in-process parity tests.

    fp32 is emitted on the wire even when the forward runs in bf16 — the collector applies
    softmax/masking/sampling in fp32 exactly like the local path (the bf16 numerics are what
    the parity gate bounds, not the serialization).
    """
    torch_module = require_torch()
    tensors = {
        "categorical_ids": torch_module.tensor(payload["categorical_ids"], dtype=torch_module.long, device=device),
        "numeric_features": torch_module.tensor(payload["numeric_features"], dtype=torch_module.float32, device=device),
        "token_type_ids": torch_module.tensor(payload["token_type_ids"], dtype=torch_module.long, device=device),
        "attention_mask": torch_module.tensor(payload["attention_mask"], dtype=torch_module.bool, device=device),
        "history_mask": torch_module.tensor(payload["history_mask"], dtype=torch_module.bool, device=device),
    }
    with torch_module.no_grad(), _autocast_context(torch_module, device, amp):
        output = policy._default_forward(tensors)
    opp = getattr(output, "opponent_action_logits", None)
    return {
        "policy_logits": output.policy_logits.float().cpu().tolist(),
        "value": output.value.float().cpu().tolist(),
        "opponent_action_logits": None if opp is None else opp.float().cpu().tolist(),
    }


class RemoteForward:
    """A ``forward_fn`` for TransformerSoftmaxPolicy that RPCs the window tensors to an
    inference server and returns an equivalent TransformerPolicyOutput (fp32 torch tensors)."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self._forward_url = base_url.rstrip("/") + "/forward"
        self._timeout = timeout

    def __call__(self, tensors: Mapping[str, Any]) -> TransformerPolicyOutput:
        torch_module = require_torch()
        payload = {key: tensors[key].detach().cpu().tolist() for key in _FORWARD_TENSOR_KEYS}
        body = json.dumps(payload).encode("utf-8")
        request = Request(self._forward_url, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self._timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
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
    with urlopen(base_url.rstrip("/") + "/config", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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


def build_request_handler(policy: TransformerSoftmaxPolicy, *, device: Any, amp: str | None):
    window_size = int(policy.result.model_config.window_size)
    policy_id = str(policy.policy_id)

    class _Handler(BaseHTTPRequestHandler):
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
                self._send(200, {"window_size": window_size, "policy_id": policy_id, "action_count": ACTION_COUNT})
            elif self.path.rstrip("/") in ("/health", "/healthz"):
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/forward":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send(200, run_forward_from_payload(policy, payload, device=device, amp=amp))

    return _Handler


def serve_inference(
    checkpoint_path: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8600,
    device: str | None = None,
    amp: str | None = None,
) -> ThreadingHTTPServer:
    """Load a checkpoint and start a threaded HTTP inference server. Returns the server
    (call .serve_forever() to block, or use in a thread for tests)."""
    from pathlib import Path

    policy = load_transformer_policy(Path(checkpoint_path), device=device)
    handler = build_request_handler(policy, device=device, amp=amp)
    server = ThreadingHTTPServer((host, port), handler)
    return server


def serve_forever(checkpoint_path: str, **kwargs: Any) -> None:
    server = serve_inference(checkpoint_path, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    thread.join()
