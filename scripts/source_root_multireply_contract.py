"""Frozen, public-safe registration for the source-root multireply study.

The study evaluates the distinct actions selected by the recorded raw policy,
model-leaf MCTS, and rollout-leaf MCTS.  An arm may select the same action as
another arm; the durable ledger registers that action once and retains aliases
so an apparent third candidate is never fabricated.  Opponent opening actions
remain private to the controller: only their reproducible sample ordinal is
durable.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping


OPPONENT_REPLY_SAMPLE_COUNT = 8
ARM_LABELS = ("raw_policy", "model_leaf", "rollout_leaf")


class MultireplyContractError(ValueError):
    """The proposed diagnostic inputs cannot be registered safely."""


def opponent_reply_selector_seed(decision_id: str, sample: int) -> int:
    """Return a public-root-bound selector seed without exposing its action."""
    if not isinstance(decision_id, str) or not decision_id:
        raise MultireplyContractError("decision id must be non-empty")
    if isinstance(sample, bool) or not isinstance(sample, int) or not 0 <= sample < OPPONENT_REPLY_SAMPLE_COUNT:
        raise MultireplyContractError("opponent reply sample is out of range")
    encoded = f"source-root-multireply:{decision_id}:{sample}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def register_candidate_actions(choices: Mapping[str, int]) -> tuple[dict[str, int], dict[str, str]]:
    """Register each distinct action once and bind every arm to its row label."""
    if set(choices) != set(ARM_LABELS):
        raise MultireplyContractError("candidate choices must cover raw, model, and rollout arms exactly")
    registered: dict[str, int] = {}
    aliases: dict[str, str] = {}
    by_action: dict[int, str] = {}
    for arm in ARM_LABELS:
        action = choices[arm]
        if isinstance(action, bool) or not isinstance(action, int) or action < 0:
            raise MultireplyContractError(f"{arm}: action index is invalid")
        existing = by_action.get(action)
        if existing is None:
            registered[arm] = action
            by_action[action] = arm
            aliases[arm] = arm
        else:
            aliases[arm] = existing
    if len(registered) < 2:
        raise MultireplyContractError("multireply diagnostic requires at least two distinct candidate actions")
    return registered, aliases


def contract_projection(*, decision_id: str, choices: Mapping[str, int]) -> dict[str, Any]:
    """Return the manifest-safe input projection for one source root."""
    actions, aliases = register_candidate_actions(choices)
    return {
        "candidate_actions": actions,
        "candidate_aliases": aliases,
        "opponent_reply_samples": list(range(OPPONENT_REPLY_SAMPLE_COUNT)),
        "opponent_selector": {
            "kind": "sampled_raw_policy",
            "deterministic": False,
            "sampling_temperature": 1.0,
            "action_hidden": True,
            "selector_seed_commitment": hashlib.sha256(
                "".join(f"{sample}:{opponent_reply_selector_seed(decision_id, sample)};" for sample in range(OPPONENT_REPLY_SAMPLE_COUNT)).encode("utf-8")
            ).hexdigest(),
        },
    }
