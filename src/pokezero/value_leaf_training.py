"""Value-only training caches from source-bound continuation successors.

The live continuation oracle can prove outcomes for counterfactual actions, but
older B2a summaries deliberately did not retain model-ready successor inputs.
This module is the narrow bridge for *new* collections: it accepts only an
exact post-fixed-step PokeZero observation plus an uncapped terminal outcome,
and writes a cache that is explicitly incompatible with policy training.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .actions import ACTION_COUNT
from .dataset import (
    TRAINING_CACHE_OBJECTIVE_CONTRACT_SCHEMA_VERSION,
    TrajectoryDatasetConfig,
    TrajectoryExample,
    TrainingCacheSummary,
    write_training_cache_from_examples,
)
from .live_foulplay_continuation import LIVE_FOULPLAY_SUCCESSOR_CAPTURE_SCHEMA_VERSION
from .observation import PokeZeroObservationV0


VALUE_LEAF_TRAINING_CACHE_SCHEMA_VERSION = "pokezero.value_leaf_training_cache.v1"
VALUE_LEAF_TRAINING_COMPATIBLE_OBJECTIVES = ("value-only",)
_REQUIRED_SOURCE_BINDING_FIELDS = frozenset(
    {
        "checkpoint_sha256",
        "checkpoint_iteration",
        "source_commit",
        "source_tree_sha256",
        "source_image_digest",
        "observation_schema_version",
        "collection_manifest_sha256",
    }
)


@dataclass(frozen=True)
class ValueLeafTrainingSummary:
    """Durable facts for a cache built from certified successor leaves."""

    capture_count: int
    example_count: int
    value_target_counts: Mapping[str, int]
    source_binding: Mapping[str, Any]
    cache: TrainingCacheSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_count": self.capture_count,
            "example_count": self.example_count,
            "value_target_counts": dict(self.value_target_counts),
            "source_binding": dict(self.source_binding),
            "cache": self.cache.to_dict(),
        }


def value_leaf_training_example(capture: Mapping[str, Any]) -> TrajectoryExample:
    """Materialize exactly one value-only example from a successor capture.

    The model sees the post-fixed-step successor, never the generic Showdown
    snapshot.  A terminal fixed step is intentionally rejected: it is a valid
    oracle decision outcome, but it has no leaf state for value supervision.
    """

    _require_capture_schema(capture)
    continuation = _mapping(capture.get("continuation"), label="capture.continuation")
    if continuation.get("terminal_after_fixed_joint_step") is not False:
        raise ValueError("value-leaf capture must contain a non-terminal fixed joint step")
    decision_count = _int(
        continuation.get("decision_round_count"),
        label="capture.continuation.decision_round_count",
    )
    if decision_count < 1:
        raise ValueError("value-leaf capture requires at least one continuation decision")
    terminal = _mapping(continuation.get("terminal"), label="capture.continuation.terminal")
    if terminal.get("capped") is not False:
        raise ValueError("value-leaf capture requires an uncapped terminal outcome")
    pokezero_player = _player(capture.get("pokezero_player"), label="capture.pokezero_player")
    foulplay_player = _player(capture.get("foulplay_player"), label="capture.foulplay_player")
    if pokezero_player == foulplay_player:
        raise ValueError("value-leaf capture must have distinct PokeZero and FoulPlay seats")
    winner = terminal.get("winner")
    if winner == pokezero_player:
        return_value = 1.0
    elif winner == foulplay_player:
        return_value = -1.0
    elif winner is None:
        return_value = 0.0
    else:
        raise ValueError("value-leaf capture terminal winner is not one of the registered seats")
    observation = capture.get("successor_observation")
    if not isinstance(observation, PokeZeroObservationV0):
        raise ValueError("value-leaf capture successor_observation must be a PokeZeroObservationV0")
    legal_action_mask = tuple(bool(value) for value in observation.legal_action_mask)
    if len(legal_action_mask) != ACTION_COUNT or not any(legal_action_mask):
        raise ValueError("value-leaf capture successor observation has no valid legal-action mask")
    source_battle_id = _nonempty_text(capture.get("source_battle_id"), label="capture.source_battle_id")
    source_seed = _int(capture.get("source_seed"), label="capture.source_seed")
    source_round = _int(capture.get("source_decision_round"), label="capture.source_decision_round")
    joint_step = _mapping(capture.get("first_restored_joint_step"), label="capture.first_restored_joint_step")
    action = _int(joint_step.get(pokezero_player), label="capture.first_restored_joint_step.pokezero_action")
    opponent_action = _int(joint_step.get(foulplay_player), label="capture.first_restored_joint_step.foulplay_action")
    if not 0 <= action < ACTION_COUNT or not 0 <= opponent_action < ACTION_COUNT:
        raise ValueError("value-leaf capture joint action is outside the PokeZero action space")
    action_target = next(index for index, legal in enumerate(legal_action_mask) if legal)
    return TrajectoryExample(
        battle_id=f"{source_battle_id}:successor:{source_round}:a{action}:o{opponent_action}",
        seed=source_seed,
        format_id=_format_id(capture),
        player_id=pokezero_player,
        turn_index=source_round + 1,
        categorical_ids=(deepcopy(observation.categorical_ids),),
        numeric_features=(deepcopy(observation.numeric_features),),
        token_type_ids=(deepcopy(observation.token_type_ids),),
        attention_mask=(deepcopy(observation.attention_mask),),
        history_mask=(True,),
        legal_action_mask=legal_action_mask,
        # This cache is explicitly value-only; retain a deterministic valid
        # action solely because the generic cache format requires one.
        action_index=action_target,
        reward=0.0,
        return_value=return_value,
        step_metadata={
            "value_leaf_training": {
                "target_kind": "uncapped-terminal-continuation",
                "policy_target": "not-used-value-only",
                "continuation_decision_round_count": decision_count,
                "source_action": action,
                "foulplay_action": opponent_action,
            }
        },
        terminal_capped=False,
    )


def source_bound_successor_capture_callback(
    *,
    source_binding: Mapping[str, Any],
    sink: Callable[[Mapping[str, Any]], None],
) -> Callable[[Mapping[str, Any]], None]:
    """Attach one sealed binding to every capture before it reaches a collector.

    The live callback intentionally knows no checkpoint or image identity.  A
    collection driver resolves those immutable facts once, then uses this
    wrapper so each individual successor receipt carries the exact same binding.
    Existing bindings are refused rather than silently overwritten.
    """

    binding = _validated_source_binding(source_binding)

    def record(capture: Mapping[str, Any]) -> None:
        if "source_binding" in capture:
            raise ValueError("value-leaf successor capture already has a source binding")
        enriched = dict(capture)
        enriched["source_binding"] = dict(binding)
        sink(enriched)

    return record


def write_value_leaf_training_cache(
    *,
    captures: Iterable[Mapping[str, Any]],
    output_path: Path,
    source_binding: Mapping[str, Any],
    dataset_config: TrajectoryDatasetConfig | None = None,
    overwrite: bool = False,
) -> ValueLeafTrainingSummary:
    """Write a value-only cache from unique, source-bound successor captures.

    A v4 checkpoint is window-one; accepting any other window here would silently
    train on an incomplete model input.  The source binding is required rather
    than inferred, so a collection driver must seal its checkpoint, source tree,
    observation schema, and manifest before this cache can become trainable.
    """

    config = dataset_config or TrajectoryDatasetConfig(window_size=1)
    if config.window_size != 1:
        raise ValueError("value-leaf successor caches require window_size=1")
    binding = _validated_source_binding(source_binding)
    captures_tuple = tuple(captures)
    if not captures_tuple:
        raise ValueError("value-leaf training cache requires at least one capture")
    capture_bindings = tuple(
        _validated_source_binding(
            _mapping(capture.get("source_binding"), label="capture.source_binding")
        )
        for capture in captures_tuple
    )
    if any(capture_binding != binding for capture_binding in capture_bindings):
        raise ValueError("value-leaf capture source binding does not match the sealed cache binding")
    examples = tuple(value_leaf_training_example(capture) for capture in captures_tuple)
    identities = tuple(_capture_identity(capture) for capture in captures_tuple)
    if len(set(identities)) != len(identities):
        raise ValueError("value-leaf training cache contains duplicate successor captures")
    observed_schemas = {
        cast_observation(capture["successor_observation"]).schema_version
        for capture in captures_tuple
    }
    if observed_schemas != {binding["observation_schema_version"]}:
        raise ValueError("value-leaf capture observation schema does not match the sealed source binding")
    target_counts = {
        "win": sum(example.return_value > 0.0 for example in examples),
        "tie": sum(example.return_value == 0.0 for example in examples),
        "loss": sum(example.return_value < 0.0 for example in examples),
    }
    cache = write_training_cache_from_examples(
        examples,
        output_path,
        config=config,
        overwrite=overwrite,
        metadata_overrides={
            "training_objective_contract": {
                "schema_version": TRAINING_CACHE_OBJECTIVE_CONTRACT_SCHEMA_VERSION,
                "compatible_objectives": list(VALUE_LEAF_TRAINING_COMPATIBLE_OBJECTIVES),
            },
            "value_leaf_training": _value_leaf_metadata(
                source_binding=binding,
                capture_count=len(captures_tuple),
                value_target_counts=target_counts,
            ),
        },
    )
    return ValueLeafTrainingSummary(
        capture_count=len(captures_tuple),
        example_count=len(examples),
        value_target_counts=target_counts,
        source_binding=binding,
        cache=cache,
    )


def cast_observation(value: object) -> PokeZeroObservationV0:
    if not isinstance(value, PokeZeroObservationV0):
        raise ValueError("value-leaf capture successor_observation must be a PokeZeroObservationV0")
    return value


def _value_leaf_metadata(
    *,
    source_binding: Mapping[str, Any],
    capture_count: int,
    value_target_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": VALUE_LEAF_TRAINING_CACHE_SCHEMA_VERSION,
        "target_mode": "uncapped-terminal-successor-value",
        "compatible_objectives": list(VALUE_LEAF_TRAINING_COMPATIBLE_OBJECTIVES),
        "required_window_size": 1,
        "capture_count": capture_count,
        "value_target_counts": dict(value_target_counts),
        "source_binding": dict(source_binding),
    }


def _capture_identity(capture: Mapping[str, Any]) -> str:
    step = _mapping(capture.get("first_restored_joint_step"), label="capture.first_restored_joint_step")
    return json.dumps(
        {
            "source_battle_id": capture.get("source_battle_id"),
            "source_seed": capture.get("source_seed"),
            "source_decision_round": capture.get("source_decision_round"),
            "pokezero_player": capture.get("pokezero_player"),
            "foulplay_player": capture.get("foulplay_player"),
            "joint_step": dict(step),
            "source_request_sha256": capture.get("source_request_sha256"),
            "snapshot_request_sha256": capture.get("snapshot_request_sha256"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _validated_source_binding(source_binding: Mapping[str, Any]) -> dict[str, Any]:
    binding = dict(source_binding)
    missing = sorted(_REQUIRED_SOURCE_BINDING_FIELDS - set(binding))
    if missing:
        raise ValueError("value-leaf source binding missing: " + ", ".join(missing))
    for field in {"observation_schema_version"}:
        _nonempty_text(binding[field], label=f"source_binding.{field}")
    for field in {
        "checkpoint_sha256",
        "source_tree_sha256",
        "source_image_digest",
        "collection_manifest_sha256",
    }:
        _sha256(binding[field], label=f"source_binding.{field}")
    _git_commit_sha(binding["source_commit"], label="source_binding.source_commit")
    if _int(binding["checkpoint_iteration"], label="source_binding.checkpoint_iteration") < 0:
        raise ValueError("source_binding.checkpoint_iteration must be non-negative")
    try:
        json.dumps(binding, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("value-leaf source binding must be JSON-serializable") from error
    return binding


def _require_capture_schema(capture: Mapping[str, Any]) -> None:
    if capture.get("schema_version") != LIVE_FOULPLAY_SUCCESSOR_CAPTURE_SCHEMA_VERSION:
        raise ValueError("unsupported value-leaf successor capture schema")


def _format_id(capture: Mapping[str, Any]) -> str:
    # The live boundary format is intentionally not carried as a generic state
    # object. Collection drivers bind it in the manifest and put it on a capture.
    return _nonempty_text(capture.get("format_id"), label="capture.format_id")


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _player(value: object, *, label: str) -> str:
    player = _nonempty_text(value, label=label)
    if player not in {"p1", "p2"}:
        raise ValueError(f"{label} must be p1 or p2")
    return player


def _nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = _nonempty_text(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text.lower()):
        raise ValueError(f"{label} must be a lowercase-or-uppercase SHA-256 hex digest")
    return text


def _git_commit_sha(value: object, *, label: str) -> str:
    text = _nonempty_text(value, label=label)
    if len(text) not in {40, 64} or any(character not in "0123456789abcdef" for character in text.lower()):
        raise ValueError(f"{label} must be a full git commit hex digest")
    return text
