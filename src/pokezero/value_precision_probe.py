"""Frozen-trunk value probes on saved observations, separate from PPO training.

Cached returns are outcome labels only under the checked unshaped-return contract.
Zero returns remain unresolved without separate historical terminal provenance.
Nothing here collects games, changes the production policy, or infers sibling values.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def checkpoint_contract(path: str | Path) -> dict[str, Any]:
    """Bind the champion contract to the exact bytes safely deserialized."""
    from .neural_policy import parse_transformer_model_config, require_torch
    torch = require_torch()
    blob = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    config = parse_transformer_model_config(payload)
    if (config.observation_schema_version != "pokezero.observation.v4"
            or config.window_size != 1 or config.temporal_aggregator != "mean"
            or config.token_count != 23 or config.numeric_feature_count != 132
            or config.categorical_feature_count != 41):
        raise ValueError("champion one-frame v4 contract required")
    if not isinstance(payload.get("belief_set_source_hash"), str) or not payload["belief_set_source_hash"]:
        raise ValueError("bound belief source required")
    masks = {"stats_block": config.stats_block_enabled, "exact_state": config.exact_state_enabled,
             "transition_token_budget": config.transition_token_budget,
             "tier2_residuals": config.tier2_residuals, "tier2_investment": config.tier2_investment,
             "feature_pack_last_move": config.feature_pack_last_move,
             "investment_belief_narrowing": config.investment_belief_narrowing,
             "item_belief_narrowing": config.item_belief_narrowing}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"schema": "pokezero.value_probe_checkpoint_contract.v1",
            "checkpoint_sha256": hashlib.sha256(blob).hexdigest(),
            "model_config_sha256": hashlib.sha256(canonical(config.to_dict())).hexdigest(),
            "category_vocab_sha256": hashlib.sha256(canonical(config.category_vocab)).hexdigest(),
            "categorical_vocab_size": config.categorical_vocab_size,
            "token_type_vocab_size": config.token_type_vocab_size,
            "belief_set_source_hash": payload["belief_set_source_hash"], "feature_masks": masks,
            "width": config.embedding_dim, "value_activation": config.value_activation,
            "value_calibration_transform": payload.get("value_calibration_transform")}


def outcome_labels(returns: Any, capped: Any, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return outcomes and status: 0 known, 1 capped, 2 unresolved zero.

    In dataset._discounted_returns_by_step_index the unshaped return is a
    terminal outcome times a positive discount power. Nonzero signs therefore
    recover outcomes without treating GAE estimates as labels. Underflowed zeros
    and absent terminals cannot be distinguished from draws in this cache.
    Zero labels require separately bound historical terminal provenance and a
    non-underflow bound. This provisional reader never imputes them as ties.
    """
    discount = config.get("discount")
    if (isinstance(discount, bool) or not isinstance(discount, (int, float))
            or not math.isfinite(discount) or not 0 < discount <= 1):
        raise ValueError("finite positive discount at most one required")
    for name in ("hp_delta_return_weight", "faint_delta_return_weight", "turn_penalty"):
        if config.get(name, 0) != 0:
            raise ValueError(f"outcome recovery refuses {name}")
    if config.get("potential_shaping") is not None:
        raise ValueError("outcome recovery refuses potential shaping")
    values = np.asarray(returns)
    caps = np.asarray(capped)
    if values.ndim != 1 or caps.shape != values.shape or caps.dtype != np.bool_:
        raise ValueError("one-dimensional returns and boolean capped flags required")
    if not np.isfinite(values).all() or (np.abs(values) > 1.000001).any():
        raise ValueError("invalid cached return")
    labels = np.sign(values).astype(np.float32)
    status = np.zeros(values.shape, dtype=np.uint8)
    status[values == 0] = 2
    status[caps] = 1
    labels[status != 0] = np.nan
    return labels, status


def battle_split(seed: int, *, lineage: str, salt: str) -> str:
    """80/10/10 train/validation/test; a seed stays together across all shards.

    Grouping copied seeds together is conservative when explicit battle/seat IDs
    were omitted. This holds data out from the head refit, not from the champion
    trunk's historical training, nor from the historical data-generating policy.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("integer game seed required")
    if not lineage or not salt:
        raise ValueError("lineage and split salt required")
    key = f"{lineage}\0{int(seed)}\0{salt}".encode()
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10000
    return "test" if bucket < 1000 else "validation" if bucket < 2000 else "train"


def validate_cache_metadata(metadata: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    """Check every selected shard against the frozen champion input contract."""
    expected = {"schema_version": "pokezero.training_cache.v2",
                "observation_schema": "pokezero.observation.v4",
                "collection_env": "showdown", "belief_fidelity": "full",
                "belief_set_source_hash": contract["belief_set_source_hash"],
                "feature_masks": contract["feature_masks"]}
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"cache contract mismatch: {name}")
    if metadata.get("belief_set_source_mixed"):
        raise ValueError("mixed source cache")
    shapes = metadata["observation_shapes"]
    if (shapes.get("numeric_features") != [23, 132]
            or shapes.get("attention_mask") != [23]
            or shapes.get("token_type_ids") != [23]
            or shapes.get("window_size") != 1):
        raise ValueError("one-frame champion observation shape required")
    categorical = shapes.get("categorical_ids", [])
    if len(categorical) != 2 or categorical[0] != 23 or not 1 <= categorical[1] <= 41:
        raise ValueError("invalid categorical storage shape")
    storage = metadata.get("categorical_storage", {})
    if categorical[1] != 41 and (storage.get("mode") != "compact-nonzero"
                               or storage.get("original_feature_count") != 41):
        raise ValueError("compressed categories require summed-embedding contract")
    outcome_labels(np.array([], np.float32), np.array([], bool), metadata["dataset_config"])


def hl_gauss_targets(targets: Any, *, bins: int = 101, sigma_ratio: float = 0.75) -> Any:
    """Gaussian CDF mass per equal-width bin on [-1,1], renormalized.

    Farebrother et al., ICML2024, section3.1 / appendixA:
    https://proceedings.mlr.press/v235/farebrother24a.html
    The paper's default sigma/bin-width ratio is0.75. This is scalar-target
    smoothing, not an estimate of the distribution of future outcomes.
    """
    from .neural_policy import require_torch
    torch = require_torch()
    if (isinstance(bins, bool) or not isinstance(bins, int) or bins < 3
            or not math.isfinite(sigma_ratio) or sigma_ratio <= 0):
        raise ValueError("at least three bins and positive finite smoothing required")
    if not torch.isfinite(targets).all() or torch.any(torch.abs(targets) > 1):
        raise ValueError("finite outcome targets in [-1,1] required")
    # Compute labels in double precision so boundary normalization is stable.
    edges = torch.linspace(-1, 1, bins + 1, dtype=torch.float64, device=targets.device)
    sigma = sigma_ratio * (2 / bins)
    cdf = 0.5 * (1 + torch.erf((edges - targets.double().unsqueeze(-1)) / (sigma * math.sqrt(2))))
    mass = torch.diff(cdf, dim=-1)
    return (mass / mass.sum(dim=-1, keepdim=True)).float()


def make_value_head(kind: str, width: int, *, hidden: int = 256, bins: int = 101,
                    tanh_value: bool = False) -> Any:
    """Linear/mean, attention MLP, or the same attention MLP with HL-Gauss."""
    from .neural_policy import require_torch
    torch = require_torch()
    nn = torch.nn
    if kind not in {"linear", "attention_mlp", "attention_hl_gauss"}:
        raise ValueError("unknown value probe head")
    if width <= 0 or hidden <= 0 or bins < 3:
        raise ValueError("positive head dimensions required")

    class ValueProbeHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kind = kind
            self.query = None if kind == "linear" else nn.Parameter(torch.zeros(width))
            outputs = bins if kind == "attention_hl_gauss" else 1
            self.network = (nn.Linear(width, 1) if kind == "linear" else nn.Sequential(
                nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, outputs)))
            edges = torch.linspace(-1, 1, bins + 1)
            self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2)

        def forward(self, tokens: Any, valid: Any) -> tuple[Any, Any | None]:
            if tokens.ndim != 3 or tokens.shape[-1] != width or valid.shape != tokens.shape[:2]:
                raise ValueError("batch/token/width features and matching attention required")
            if valid.dtype != torch.bool or not valid.any(dim=-1).all():
                raise ValueError("each observation needs an attended token")
            if self.query is None:
                pooled = torch.where(valid[..., None], tokens, 0).sum(1) / valid.sum(1, keepdim=True)
            else:
                score = (tokens * self.query).sum(-1) / math.sqrt(width)
                weights = torch.softmax(score.masked_fill(~valid, -torch.inf), dim=-1)
                pooled = (weights[..., None] * torch.where(valid[..., None], tokens, 0)).sum(1)
            raw = self.network(pooled)
            if kind == "attention_hl_gauss":
                return (torch.softmax(raw, -1) * self.centers).sum(-1), raw
            values = raw.squeeze(-1)
            return (torch.tanh(values) if tanh_value else values), None

    return ValueProbeHead()


def frozen_token_features(model: Any, tensors: Mapping[str, Any]) -> tuple[Any, Any]:
    """Capture the actual production encoder output, without a second trunk port."""
    from .neural_policy import model_forward_from_training_tensors, require_torch
    torch = require_torch()
    if model.config.window_size != 1 or model.config.temporal_aggregator != "mean" or model.encoder is None:
        raise ValueError("one-frame mean-pooled transformer trunk required")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captured = []
    hook = model.encoder.register_forward_hook(lambda _module, _args, output: captured.append(output.detach()))
    try:
        with torch.no_grad():
            output = model_forward_from_training_tensors(model, tensors)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError("expected exactly one encoder invocation")
    return captured[0], output


def calibration_metrics(predictions: Any, outcomes: Any, *, bins: int = 20) -> dict[str, Any]:
    """Outcome residual moments and ECE, not an epistemic/aleatoric decomposition."""
    p, y = np.asarray(predictions, np.float64), np.asarray(outcomes, np.float64)
    if p.ndim != 1 or p.shape != y.shape or not p.size or bins < 1:
        raise ValueError("nonempty paired predictions/outcomes required")
    if not np.isfinite(p).all() or not np.isfinite(y).all() or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("finite predictions and actual outcome labels required")
    residual = p - y
    indices = np.minimum(bins - 1, (((np.clip(p, -1, 1) + 1) / 2) * bins).astype(int))
    groups = []
    for index in range(bins):
        selected = indices == index
        if selected.any():
            groups.append({"bin": index, "count": int(selected.sum()),
                           "mean_prediction": float(p[selected].mean()),
                           "mean_outcome": float(y[selected].mean())})
    bias = float(residual.mean())
    return {"examples": int(p.size), "mse": float(np.mean(residual ** 2)),
            "mae": float(np.mean(np.abs(residual))), "bias": bias, "bias_squared": bias ** 2,
            "centered_outcome_residual_variance": float(np.var(residual)),
            "ece_return_scale": sum(g["count"] / p.size * abs(g["mean_prediction"] - g["mean_outcome"])
                                    for g in groups), "bins": groups,
            "interpretation": "Residual variance includes outcome randomness and model error; not separately identified.",
            "top_two_action_resolution": {"status": "UNMEASURED_REQUIRES_ACTUAL_SIBLING_OBSERVATIONS"}}
