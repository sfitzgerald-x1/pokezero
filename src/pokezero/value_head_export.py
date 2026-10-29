"""A1 validation-selected value head on the unchanged champion inference trunk.

The positional five-input, three-output interface is the existing Rust
NativeLeafModel interface. No checkpoint is rewritten, and root/opponent
policies continue to load the original champion checkpoint. All paths use FP32.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

from .neural_policy import require_torch
from .value_precision_probe import make_value_head


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_validation_head(summary: dict[str, Any]) -> str:
    """Select architecture using validation alone, with a fixed tie order."""
    kinds = ("linear", "attention_mlp", "attention_hl_gauss")
    if (summary.get("schema") != "pokezero.value_probe_head_refits.v1"
            or summary.get("status") != "THREE_OUTCOME_HEAD_REFITS_COMPLETE_ZEROS_UNRESOLVED"
            or summary.get("model_fits") != 3 or set(summary.get("heads", {})) != set(kinds)):
        raise ValueError("completed three-head A1 refit summary required")
    for kind in kinds:
        entry = summary["heads"][kind]
        mse = entry.get("validation_mse")
        if (type(mse) not in (int, float) or not math.isfinite(mse) or mse < 0
                or type(entry.get("selected_epoch")) is not int or entry["selected_epoch"] < 1):
            raise ValueError("finite validation MSE and selected epoch required")
    return min(kinds, key=lambda kind: summary["heads"][kind]["validation_mse"])


def load_selected_head(summary_path: str | Path, *, summary_sha256: str,
                       checkpoint_contract: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Bind selection, head bytes, feature lineage, and champion before loading."""
    path = Path(summary_path)
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != summary_sha256:
        raise ValueError("head summary binding mismatch")
    summary = json.loads(blob)
    if summary.get("checkpoint_contract") != checkpoint_contract:
        raise ValueError("refit checkpoint contract differs from champion")
    kind = select_validation_head(summary)
    head_path = path.parent / (kind + ".pt")
    head_blob = head_path.read_bytes()
    head_sha = hashlib.sha256(head_blob).hexdigest()
    if head_sha != summary["heads"][kind]["artifact_sha256"]:
        raise ValueError("selected head bytes differ from summary")
    torch = require_torch()
    payload = torch.load(io.BytesIO(head_blob), map_location="cpu", weights_only=True)
    if (payload.get("schema") != "pokezero.value_probe_head.v1"
            or payload.get("kind") != kind or payload.get("width") != checkpoint_contract["width"]
            or payload.get("features_manifest_sha256") != summary["features_manifest_sha256"]
            or payload.get("selected_epoch") != summary["heads"][kind]["selected_epoch"]):
        raise ValueError("selected head provenance mismatch")
    head = make_value_head(kind, checkpoint_contract["width"]).float().eval()
    head.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(tensor).all() for tensor in head.state_dict().values()):
        raise ValueError("nonfinite selected head")
    return head, {"kind": kind, "selected_epoch": payload["selected_epoch"],
        "validation_mse": summary["heads"][kind]["validation_mse"],
        "summary_sha256": summary_sha256, "head_sha256": head_sha,
        "features_manifest_sha256": payload["features_manifest_sha256"],
        "selection": "minimum validation MSE; fixed linear/attention_mlp/attention_hl_gauss tie order; no test selection"}


def build_value_head_export(model: Any, head: Any | None) -> Any:
    """Use production embedding/encoder/pool/policy modules; replace only value.

    This one-frame expanded-input adapter follows the production forward, and is
    checked against the independent encoder-hook oracle before export. It avoids
    hooks or mutable per-forward state in the served/traced model.
    """
    torch = require_torch()
    from .showdown import ACTION_CANDIDATE_TOKEN_OFFSET
    from .actions import ACTION_COUNT
    config = model.config
    if (config.window_size != 1 or config.temporal_aggregator != "mean"
            or model.encoder is None or config.token_count != 23
            or config.observation_schema_version != "pokezero.observation.v4"):
        raise ValueError("one-frame 23-token v4 transformer required")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("FP32 champion parameters required")
    if head is not None and any(parameter.dtype != torch.float32 for parameter in head.parameters()):
        raise ValueError("FP32 refit parameters required")

    class ValueHeadExport(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.champion = model
            self.refit = head

        def forward(self, categorical_ids, numeric_features, token_type_ids,
                    attention_mask, history_mask):
            x = self.champion._embed_expanded_inputs(categorical_ids=categorical_ids,
                numeric_features=numeric_features, token_type_ids=token_type_ids)
            batch_size, window_size, token_count, _ = x.shape
            positions = torch.arange(window_size, device=x.device)
            x = x + self.champion.history_position_embedding(positions).view(1, window_size, 1, -1)
            x = x.view(batch_size, window_size * token_count, config.embedding_dim)
            valid = (attention_mask.bool() & history_mask.bool().unsqueeze(-1)).view(batch_size, -1)
            encoded = self.champion.encoder(x, src_key_padding_mask=~valid)
            pooled = self.champion._pool_encoded_history(encoded=encoded,
                attention_mask=attention_mask.bool(), history_mask=history_mask.bool())
            action_tokens = encoded[:, ACTION_CANDIDATE_TOKEN_OFFSET:ACTION_CANDIDATE_TOKEN_OFFSET + ACTION_COUNT, :]
            policy = self.champion.policy_head(action_tokens).squeeze(-1)
            opponent = self.champion.opponent_action_head(pooled)
            if self.refit is None:
                raw = self.champion.value_head(pooled).squeeze(-1)
                value = torch.tanh(raw) if config.value_activation == "tanh" else raw
            else:
                value = self.refit(encoded, valid)[0]
            return policy, value, opponent

    return ValueHeadExport().float().eval()


def validate_export(module: Any, traced: Any, model: Any, head: Any | None,
                    inputs: tuple[Any, ...], *, tolerance: float = 1e-5) -> dict[str, float]:
    """Independent production forward and encoder hook, including masked tokens."""
    from .value_precision_probe import frozen_token_features
    torch = require_torch()
    names = ("categorical_ids", "numeric_features", "token_type_ids", "attention_mask", "history_mask")
    tensors = dict(zip(names, inputs))
    with torch.no_grad():
        tokens, baseline = frozen_token_features(model, tensors)
        reference_value = baseline.value if head is None else head(tokens, inputs[3][:, 0])[0]
        reference = (baseline.policy_logits, reference_value, baseline.opponent_action_logits)
        eager = module(*inputs)
        actual = traced(*inputs)
    result = {}
    for index, name in enumerate(("policy_logits", "value", "opponent_action_logits")):
        for label, candidate in (("eager", eager), ("traced", actual)):
            if not torch.isfinite(candidate[index]).all():
                raise ValueError("nonfinite export outputs")
            error = float((candidate[index] - reference[index]).abs().max())
            result[f"{label}_{name}_max_abs_diff"] = error
            if error > tolerance:
                raise ValueError(f"{label} {name} export error {error} exceeds {tolerance}")
    return result
