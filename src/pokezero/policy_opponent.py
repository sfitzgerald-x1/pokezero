"""Strict own-policy provider for the paper-opponent correctness reference.

This is not the auxiliary opponent head and not an enabled search mode. It
provides the reference distribution against which native integration must be
checked at roots and reached children before that mode can be enabled.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .actions import ACTION_COUNT
from .category_vocab import CategoryVocabulary
from .dex import ShowdownDex
from .policy_opponent_view import PolicyOpponentView, PolicyOpponentViewError


def policy_opponent_distribution(
    view: PolicyOpponentView, *, native_action_indices: Sequence[int | None],
    model: Any, result: Any, category_vocab: CategoryVocabulary, dex: ShowdownDex,
    device: Any = None, timing: Any = None,
) -> tuple[float, ...]:
    """Return probabilities in the exact native legal-option order, or refuse.

    Single-choice/no-action boundaries are handled by the native sampler without
    a network call. This reference requires a one-snapshot checkpoint: passing
    one current view to a multi-window model would silently discard history.
    No epsilon floor, temperature sweep, or fallback distribution is applied.
    """
    from .neural_policy import (
        evaluate_transformer_action_priors, feature_masks_from_model_config,
        observation_spec_from_model_config,
    )

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
    if not indices or any(type(index) is not int or not 0 <= index < ACTION_COUNT for index in indices):
        raise PolicyOpponentViewError("unmapped native policy-opponent action")
    if len(set(indices)) != len(indices):
        raise PolicyOpponentViewError("duplicate native policy-opponent action")
    observation = view.observation(category_vocab=category_vocab, dex=dex)
    observation.validate(view.spec)
    legal = {index for index, enabled in enumerate(observation.legal_action_mask) if enabled}
    if set(indices) != legal:
        raise PolicyOpponentViewError("native and sampled-request legal surfaces differ")
    probabilities = evaluate_transformer_action_priors(
        model=model, result=result, observations=(observation,), temperature=1.0,
        device=device, timing=timing,
    )
    if len(probabilities) != ACTION_COUNT or any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise PolicyOpponentViewError("invalid own-policy probability row")
    if any(probabilities[index] != 0 for index in range(ACTION_COUNT) if index not in legal):
        raise PolicyOpponentViewError("own-policy row contains illegal action mass")
    weights = tuple(probabilities[index] for index in indices)
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise PolicyOpponentViewError("own-policy legal distribution has no mass")
    return tuple(weight / total for weight in weights)
