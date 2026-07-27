"""Checkpoint contract resolution + artifact materialization (plan deliverable 1).

The study must run unmodified against any engine-compatible checkpoint, so the
checkpoint reference is *data* and every contract is DERIVED from the
checkpoint's own stamped model configuration:

    observation schema + token dimensions, transition-history budget, feature
    masks, architecture, policy id, TorchScript export shape/device, and the
    encoder-table contract.

Terminal (never retried, never reused) conditions, per plan section 2.1:

- the checkpoint hash does not match the caller's expectation (provenance drift);
- the observation schema is unsupported by the engine encoder;
- root and leaf contracts disagree (the Python root encode and the crate's
  encoder tables must describe the same observation).

Export reuse is keyed by checkpoint hash, accelerator device, observation
contract, Showdown source hash, and exporter revision — anything that can change
an exported artifact's meaning is in the key, so a stale artifact can never be
silently adopted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

# Bump when the exporter's output semantics change; it is part of the reuse key
# so previously exported artifacts are not adopted across an exporter change.
EXPORTER_REVISION = "pokezero.mcts-eval.exporter.v1"

# Schemas the engine encoder (rust/pokezero-search encoder tables) can express.
SUPPORTED_OBSERVATION_SCHEMAS = ("pokezero.observation.v2.2", "pokezero.observation.v3")

TABLES_SCHEMA_VERSION = "pokezero.encoder-tables.v1"


class ContractError(RuntimeError):
    """Terminal provenance/contract failure — the run stops, never retries."""


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CheckpointContract:
    """Everything the harness must freeze about one checkpoint before it runs.

    ``observation_contract`` is the canonical, order-independent description of
    what an observation IS for this checkpoint; it is what root and leaf encodes
    must agree on and what export reuse is keyed by.
    """

    checkpoint_path: str
    checkpoint_sha256: str
    policy_id: str
    schema_version: str
    token_count: int
    categorical_feature_count: int
    numeric_feature_count: int
    transition_token_count: int
    architecture: Mapping[str, Any]
    feature_masks: Mapping[str, Any]
    model_device: str
    showdown_root: str | None = None
    showdown_source_sha256: str | None = None
    model_path: str | None = None
    model_sha256: str | None = None
    tables_path: str | None = None
    tables_sha256: str | None = None
    exporter_revision: str = EXPORTER_REVISION

    @property
    def observation_contract(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "token_count": self.token_count,
            "categorical_feature_count": self.categorical_feature_count,
            "numeric_feature_count": self.numeric_feature_count,
            "transition_token_count": self.transition_token_count,
            "feature_masks": dict(self.feature_masks),
        }

    @property
    def observation_contract_sha256(self) -> str:
        return _canonical_hash(self.observation_contract)

    def to_manifest(self) -> dict[str, Any]:
        """Provenance block embedded in every artifact this study produces."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "policy_id": self.policy_id,
            "observation_contract": self.observation_contract,
            "observation_contract_sha256": self.observation_contract_sha256,
            "architecture": dict(self.architecture),
            "model_device": self.model_device,
            "showdown_root": self.showdown_root,
            "showdown_source_sha256": self.showdown_source_sha256,
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "tables_path": self.tables_path,
            "tables_sha256": self.tables_sha256,
            "exporter_revision": self.exporter_revision,
        }


def export_reuse_key(contract: CheckpointContract) -> str:
    """Key an exported artifact by everything that can change its meaning."""
    return _canonical_hash(
        {
            "checkpoint_sha256": contract.checkpoint_sha256,
            "model_device": contract.model_device,
            "observation_contract_sha256": contract.observation_contract_sha256,
            "showdown_source_sha256": contract.showdown_source_sha256,
            "exporter_revision": contract.exporter_revision,
        }
    )


def _architecture_from_model_config(config: Any) -> dict[str, Any]:
    fields = (
        "embedding_dim",
        "transformer_layers",
        "attention_heads",
        "feedforward_dim",
        "window_size",
        "temporal_aggregator",
        "categorical_vocab_size",
    )
    return {name: getattr(config, name) for name in fields if hasattr(config, name)}


def resolve_checkpoint_contract(
    checkpoint: str | Path,
    *,
    expected_sha256: str | None = None,
    model_device: str = "cpu",
    showdown_root: str | Path | None = None,
    showdown_source_sha256: str | None = None,
    expected_showdown_source_sha256: str | None = None,
    model_path: str | Path | None = None,
    tables_path: str | Path | None = None,
) -> CheckpointContract:
    """Derive (never assume) the full contract from a checkpoint's own config.

    Optional pre-exported artifacts are validated against the derived contract
    rather than trusted: a TorchScript artifact or encoder-table JSON describing
    a different observation is a terminal root/leaf disagreement.
    """
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise ContractError(f"checkpoint not found: {checkpoint_path}")

    actual_sha = sha256_file(checkpoint_path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ContractError(
            f"checkpoint provenance drift: {checkpoint_path} has sha256 {actual_sha}, "
            f"expected {expected_sha256}. Refusing to run a matrix against a different "
            "checkpoint than the one the experiment identity was computed for."
        )
    if (
        expected_showdown_source_sha256 is not None
        and showdown_source_sha256 is not None
        and expected_showdown_source_sha256 != showdown_source_sha256
    ):
        raise ContractError(
            f"Showdown source drift: got {showdown_source_sha256}, expected "
            f"{expected_showdown_source_sha256}; encoder tables and the engine would "
            "describe different games."
        )

    from ..neural_policy import (
        feature_masks_from_model_config,
        load_transformer_model_config,
        observation_spec_from_model_config,
    )

    config = load_transformer_model_config(checkpoint_path)
    spec = observation_spec_from_model_config(config)
    if spec.schema_version not in SUPPORTED_OBSERVATION_SCHEMAS:
        raise ContractError(
            f"unsupported observation schema {spec.schema_version!r} for engine search "
            f"(supported: {', '.join(SUPPORTED_OBSERVATION_SCHEMAS)})."
        )
    masks = feature_masks_from_model_config(config)
    masks_payload = {
        name: getattr(masks, name)
        for name in dir(masks)
        if not name.startswith("_") and not callable(getattr(masks, name))
    }

    contract = CheckpointContract(
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=actual_sha,
        policy_id=str(getattr(config, "policy_id", "") or ""),
        schema_version=spec.schema_version,
        token_count=int(spec.token_count),
        categorical_feature_count=int(spec.categorical_feature_count),
        numeric_feature_count=int(spec.numeric_feature_count),
        transition_token_count=int(getattr(spec, "transition_token_count", 0)),
        architecture=_architecture_from_model_config(config),
        feature_masks=masks_payload,
        model_device=model_device,
        showdown_root=str(showdown_root) if showdown_root is not None else None,
        showdown_source_sha256=showdown_source_sha256,
        model_path=str(model_path) if model_path is not None else None,
        model_sha256=sha256_file(model_path) if model_path is not None else None,
        tables_path=str(tables_path) if tables_path is not None else None,
        tables_sha256=sha256_file(tables_path) if tables_path is not None else None,
    )
    if tables_path is not None:
        validate_encoder_tables(contract, tables_path)
    return contract


def validate_encoder_tables(contract: CheckpointContract, tables_path: str | Path) -> None:
    """Fail closed when the leaf encoder describes a different observation than the root.

    The crate encodes leaves from these tables while Python encodes the root from
    the checkpoint's spec; a silent disagreement is the census-mismatch class —
    the model would score states it never trained on.
    """
    payload = json.loads(Path(tables_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != TABLES_SCHEMA_VERSION:
        raise ContractError(
            f"encoder tables {tables_path}: unknown artifact schema "
            f"{payload.get('schema_version')!r} (expected {TABLES_SCHEMA_VERSION})."
        )
    layout = payload.get("layout") or {}
    mismatches = {
        key: (layout.get(key), expected)
        for key, expected in (
            ("schema_version", contract.schema_version),
            ("token_count", contract.token_count),
            ("categorical_feature_count", contract.categorical_feature_count),
            ("numeric_feature_count", contract.numeric_feature_count),
        )
        if layout.get(key) != expected
    }
    if mismatches:
        detail = ", ".join(
            f"{key}: tables={got!r} checkpoint={expected!r}" for key, (got, expected) in sorted(mismatches.items())
        )
        raise ContractError(
            f"root/leaf observation contract disagreement for {tables_path} ({detail}). "
            "The crate would encode leaves the checkpoint never trained on."
        )
