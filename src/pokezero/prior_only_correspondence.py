"""Independent low-work prior replay; never a selected-action/work proof.

Only the two prospectively authorized four-world model budgets are supported.
Historical exact correspondence remains unchanged. Every declared boundary is
still joined using original identity, legal mapping and pre-display priors.
"""
from copy import deepcopy

from .root_prior_correspondence import exact_collector_gate, fingerprint


def prior_only_binding(selection_binding):
    """Permit exactly search_sims 1024/4096 -> 16, not arbitrary drift."""
    result = deepcopy(selection_binding)
    selectors = result.get("selector_configurations")
    if not isinstance(selectors, dict) or set(selectors) != {"sampled4"}:
        raise ValueError("one four-world public selector required")
    config = selectors["sampled4"]
    if not isinstance(config, dict) or config.get("worlds") != 4 \
            or type(config.get("search_sims")) is not int \
            or config["search_sims"] not in (1024, 4096) \
            or config.get("leaf_eval") != "model" \
            or config.get("model_leaf_override") is not None \
            or config.get("strict_fallbacks") is not True \
            or config.get("root_prior_exact_aggregate") is not True \
            or result.get("world_rng") != "per-decision":
        raise ValueError("declared fixed-work public model selection required")
    config["search_sims"] = 16
    return result


def prior_only_collector_gate(collector, live):
    """Check every expected prior with exact TV zero, not final actions.

    Both reports retain their original bindings. The exact historical reader
    is reused only after checking the single permitted configuration difference.
    No source, binary, seed or model pin is erased.
    """
    try:
        expected = prior_only_binding(collector["bindings"])
        if fingerprint(expected) != fingerprint(live.get("bindings")):
            raise ValueError("prior-only replay differs beyond its declared work budget")
        normalized = deepcopy(live)
        normalized["bindings"] = deepcopy(collector["bindings"])
        verdict = exact_collector_gate(collector, normalized)
    except (KeyError, TypeError, ValueError):
        verdict = dict(passed=False, errors=["invalid_prior_only_budget_binding"])
    verdict.update(scope="root priors only; not selected-action or accepted-work validation",
        prior_iterations_per_world=16, selected_action_compared=False,
        allowed_binding_difference="sampled4.search_sims:1024/4096->16 only")
    return verdict
