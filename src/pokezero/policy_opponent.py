"""Strict own-policy provider for the isolated native opponent-sampling mode.

This is not the auxiliary opponent head. The callback constructs this seat's
canonical information state at root and child nodes; it is opt-in and is not
enabled in the live high-level policy or any benchmark by default.
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from typing import Any, Callable, Mapping, Sequence

from .actions import ACTION_COUNT
from .category_vocab import CategoryVocabulary
from .dex import ShowdownDex
from .policy_opponent_diagnostics import PolicyOpponentDiagnostics
from .policy_opponent_view import (
    PolicyOpponentView, PolicyOpponentViewError,
    _PublicPolicyPrefix, build_policy_opponent_view_from_native_bundle, public_policy_lines,
)


def policy_opponent_distribution(
    view: PolicyOpponentView, *, native_action_indices: Sequence[int | None],
    model: Any, result: Any, category_vocab: CategoryVocabulary, dex: ShowdownDex,
    device: Any = None, timing: Any = None, inference_lock: Any = None,
    raw_argmax: bool = False,
    diagnostics: PolicyOpponentDiagnostics | None = None,
) -> tuple[float, ...]:
    """Return probabilities in the exact native legal-option order, or refuse.

    Single-choice/no-action boundaries skip the network only AFTER the complete
    private-safe view and legal surface are certified. This reference requires
    a one-snapshot checkpoint: passing
    one current view to a multi-window model would silently discard history.
    No epsilon floor, temperature sweep, or fallback distribution is applied.
    """
    from .neural_policy import (
        evaluate_transformer_action_priors, feature_masks_from_model_config,
        observation_spec_from_model_config,
    )

    if diagnostics is not None and type(diagnostics) is not PolicyOpponentDiagnostics:
        raise PolicyOpponentViewError("diagnostics must be the aggregate timing collector")
    with diagnostics.phase("distribution_binding") if diagnostics is not None else nullcontext():
        if type(raw_argmax) is not bool:
            raise PolicyOpponentViewError("raw_argmax must be an explicit boolean")
        config = result.model_config
        if getattr(model, "config", None) != config:
            raise PolicyOpponentViewError("policy model/config binding mismatch")
        if config.window_size != 1:
            raise PolicyOpponentViewError("policy opponent requires complete multi-window history")
        if observation_spec_from_model_config(config) != view.spec:
            raise PolicyOpponentViewError("policy opponent observation spec differs from checkpoint")
        if feature_masks_from_model_config(config) != view.feature_masks:
            raise PolicyOpponentViewError("policy opponent feature masks differ from checkpoint")
        if category_vocab.tokens != tuple(config.category_vocab) or category_vocab.oov_buckets != config.category_oov_buckets:
            raise PolicyOpponentViewError("policy opponent vocabulary differs from checkpoint")
        expected_source_hash = getattr(result, "belief_set_source_hash", None)
        source = view.materialization.belief_engine.set_source
        source_hash = getattr(getattr(source, "metadata", None), "source_hash", None)
        if expected_source_hash is not None and source_hash != expected_source_hash:
            raise PolicyOpponentViewError("policy opponent belief source differs from checkpoint")
        indices = tuple(native_action_indices)
        if view.native_action_indices is not None and indices != view.native_action_indices:
            raise PolicyOpponentViewError("policy opponent action map differs from certified native bundle")
        waiting = indices == (None,) and view.native_action_indices == (None,)
        if not waiting and (not indices or any(
            type(index) is not int or not 0 <= index < ACTION_COUNT for index in indices
        )):
            raise PolicyOpponentViewError("unmapped native policy-opponent action")
        if len(set(indices)) != len(indices):
            raise PolicyOpponentViewError("duplicate native policy-opponent action")
    with diagnostics.phase("observation_and_surface") if diagnostics is not None else nullcontext():
        observation = view.observation(category_vocab=category_vocab, dex=dex)
        observation.validate(view.spec)
        legal = {index for index, enabled in enumerate(observation.legal_action_mask) if enabled}
        if (set() if waiting else set(indices)) != legal:
            raise PolicyOpponentViewError("native and sampled-request legal surfaces differ")
    if len(indices) == 1:
        return (1.0,)
    # The canonical evaluator calls eval()/to() before each forward. Parallel
    # trees may share weights, but must never mutate that module concurrently.
    # Only inference is serialized; observations remain invocation-owned.
    with diagnostics.phase("model_evaluation") if diagnostics is not None else nullcontext():
        with inference_lock if inference_lock is not None else nullcontext():
            probabilities = evaluate_transformer_action_priors(
                model=model, result=result, observations=(observation,), temperature=1.0,
                device=device, timing=timing,
            )
    with diagnostics.phase("output_certification") if diagnostics is not None else nullcontext():
        if len(probabilities) != ACTION_COUNT or any(not math.isfinite(value) or value < 0 for value in probabilities):
            raise PolicyOpponentViewError("invalid own-policy probability row")
        if any(probabilities[index] != 0 for index in range(ACTION_COUNT) if index not in legal):
            raise PolicyOpponentViewError("own-policy row contains illegal action mass")
        weights = tuple(probabilities[index] for index in indices)
        total = math.fsum(weights)
        if not math.isfinite(total) or total <= 0:
            raise PolicyOpponentViewError("own-policy legal distribution has no mass")
        if raw_argmax:
            # Raw's canonical action-slot tie break, NOT native option order. The
            # strict view and legal-surface certification above still apply in full.
            chosen = min(range(len(indices)), key=lambda i: (-weights[i], indices[i]))
            return tuple(float(i == chosen) for i in range(len(indices)))
        return tuple(weight / total for weight in weights)


def make_policy_opponent_callback(
    *, public_lines: Sequence[str], hp_visibility: Mapping[str, str], opponent_slot: str,
    battle_id: str, battle_seed: int, format_id: str, set_source: Any,
    model: Any, result: Any, category_vocab: CategoryVocabulary, dex: ShowdownDex,
    device: Any = None, timing: Any = None, inference_lock: Any = None,
    raw_argmax: bool = False,
    diagnostics: PolicyOpponentDiagnostics | None = None,
) -> Callable[[str], tuple[float, ...]]:
    """Native search callback using ONLY this seat's canonical own policy.

    Each reached node supplies a complete projected branch suffix and an
    updated sampled own-party request. Clone the immutable public root parser,
    not the last callback's state: traversal order can revisit a parent or
    alternate between sibling branches. Beliefs and private request surfaces
    are fully reconstructed on each call. The native sampler caches
    each returned distribution for its specific node/ordered action surface.
    """
    from .neural_policy import feature_masks_from_model_config, observation_spec_from_model_config

    if diagnostics is not None and type(diagnostics) is not PolicyOpponentDiagnostics:
        raise PolicyOpponentViewError("diagnostics must be the aggregate timing collector")
    if type(raw_argmax) is not bool:
        raise PolicyOpponentViewError("raw_argmax must be an explicit boolean")
    root_prefix = public_policy_lines(public_lines, hp_visibility=hp_visibility)
    public_prefix = _PublicPolicyPrefix(root_prefix, battle_id=battle_id)
    spec = observation_spec_from_model_config(result.model_config)
    masks = feature_masks_from_model_config(result.model_config)

    def provide(payload_json: str) -> tuple[float, ...]:
        with diagnostics.phase("callback_total") if diagnostics is not None else nullcontext():
            with diagnostics.phase("payload_binding") if diagnostics is not None else nullcontext():
                try:
                    payload = json.loads(payload_json)
                except (TypeError, ValueError) as error:
                    raise PolicyOpponentViewError("malformed native policy-opponent payload") from error
                if not isinstance(payload, dict) or set(payload) != {
                    "native_request_bundle", "public_branch_lines", "opponent_slot",
                } or payload["opponent_slot"] != opponent_slot:
                    raise PolicyOpponentViewError("native policy-opponent payload seat/field binding mismatch")
                suffix = payload["public_branch_lines"]
                if not isinstance(suffix, list) or not all(isinstance(line, str) for line in suffix):
                    raise PolicyOpponentViewError("native policy-opponent public suffix must be lines")
            with diagnostics.phase("view_reconstruction") if diagnostics is not None else nullcontext():
                view = build_policy_opponent_view_from_native_bundle(
                    native_request_bundle=payload["native_request_bundle"],
                    public_lines=(*root_prefix, *suffix), hp_visibility={"p1": "percentage", "p2": "percentage"},
                    opponent_slot=opponent_slot, battle_id=battle_id, battle_seed=battle_seed,
                    format_id=format_id, set_source=set_source, spec=spec, feature_masks=masks,
                    _public_prefix=public_prefix,
                )
            return policy_opponent_distribution(
                view, native_action_indices=view.native_action_indices or (),
                model=model, result=result, category_vocab=category_vocab, dex=dex,
                device=device, timing=timing, inference_lock=inference_lock,
                raw_argmax=raw_argmax, diagnostics=diagnostics,
            )

    return provide
