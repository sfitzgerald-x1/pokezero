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
# v2: encoder tables are derived from the checkpoint (--checkpoint) rather than
# the schema default, so region-trimmed models get tables of THEIR width.
# v3: `default_feature_masks` is derived from the checkpoint too. v2 only threaded
# the observation SPEC through, so the masks stayed at the dataclass defaults and
# the tables described feature gating no trained checkpoint actually used.
# v4: the categorical VOCABULARY is derived from the checkpoint too. v3 still
# enumerated it from the build, so tables exported for an older checkpoint on a
# newer build described a different string->row map than the model was trained
# with. v3-era artifacts are therefore not reusable.
# Bumping this invalidates every artifact exported by v1/v2/v3 — which is the whole
# point of having the field: a behaviour change in the exporter must not be
# silently reused. (Missing this bump caused a trimmed run to reuse cached
# 87-token tables and fail the root/leaf contract check.)
EXPORTER_REVISION = "pokezero.mcts-eval.exporter.v4"

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
    # The token->row enumeration the checkpoint was TRAINED with. This is the
    # authority for what a categorical id means: the model's embedding rows were
    # learned against these positions, so an encode that uses a different
    # enumeration indexes rows that mean something else. Never re-derive it from
    # the build — the build's enumeration is only correct for a model created by
    # that same build.
    category_vocab: tuple[str, ...]
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
            # In the contract, so the reuse key changes when the enumeration does:
            # a tables file exported against a different vocabulary can no longer
            # be adopted for this checkpoint, whatever wrote it.
            "category_vocab_sha256": hashlib.sha256(
                "\n".join(self.category_vocab).encode("utf-8")
            ).hexdigest(),
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
        category_vocab=tuple(str(t) for t in (getattr(config, "category_vocab", ()) or ())),
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

    Everything here is compared against the CHECKPOINT, never against the build.
    The checkpoint is the only authority for what its own observation means: the
    embedding rows were learned against the enumeration stamped in it, so a table
    that agrees with today's build but not with the checkpoint is the defective
    one, however freshly it was exported.
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
    # The four scalars above describe the observation's SHAPE. They were the only
    # thing checked, and shape agreement is not contract agreement: the crate also
    # gates encode work on `default_feature_masks`, so tables of the right width
    # can still fill regions the checkpoint masks (or blank regions it expects).
    # That hole let every engine-search eval run with `tier2_investment: false`
    # tables against `tier2_investment: True` checkpoints, and fed a budget-0
    # Markov checkpoint a full 64-token synthesized history region.
    table_masks = layout.get("default_feature_masks") or {}
    contract_masks = dict(contract.feature_masks)
    # Tables name this field `stats_block`; the checkpoint config spells it out.
    expected_masks = {
        "exact_state": contract_masks.get("exact_state"),
        "stats_block": contract_masks.get("opponent_tendency_stats_block"),
        "tier2_residuals": contract_masks.get("tier2_residuals"),
        "tier2_investment": contract_masks.get("tier2_investment"),
        # The exporter clamps the budget to the physical region, so compare against
        # the same clamp rather than the raw config value.
        "transition_token_budget": min(
            int(contract_masks.get("transition_token_budget") or 0),
            contract.transition_token_count,
        ),
    }
    mismatches.update(
        {
            f"default_feature_masks.{key}": (table_masks.get(key), expected)
            for key, expected in expected_masks.items()
            if expected is not None and table_masks.get(key) != expected
        }
    )
    # Compare against the CHECKPOINT, not against this build. Getting that
    # backwards is worse than not checking: on 2026-07-29 both 5M checkpoints had
    # trained on a 1216-token enumeration while the build had grown to 1217
    # (`volatile:solarbeam` inserted at index 1204, renumbering the 13 volatiles
    # after it). A build-anchored check REJECTS the cached tables that correctly
    # matched the model and ACCEPTS a fresh export that silently shifts those rows.
    stored = list((payload.get("vocab") or {}).get("tokens") or [])
    trained = list(contract.category_vocab)
    if trained and stored != trained:
        first_shift = next(
            (i for i, (a, b) in enumerate(zip(stored, trained)) if a != b), None
        )
        unknown = [t for t in stored if t not in set(trained)]
        absent = [t for t in trained if t not in set(stored)]
        mismatches["vocab.tokens"] = (
            f"{len(stored)} tokens (first divergence at index {first_shift}; "
            f"not in the trained vocabulary: {unknown[:5]}; "
            f"trained tokens absent from the tables: {absent[:5]})",
            f"{len(trained)} tokens as trained",
        )

    if mismatches:
        detail = ", ".join(
            f"{key}: tables={got!r} checkpoint={expected!r}" for key, (got, expected) in sorted(mismatches.items())
        )
        raise ContractError(
            f"root/leaf observation contract disagreement for {tables_path} ({detail}). "
            "The crate would encode leaves the checkpoint never trained on."
        )
