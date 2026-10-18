"""Passive initial-team capture for the narrowly fixed one-world A1 check.

No simulator commands, source truth, selector replacement or launch authority.
The caller must bind the complete runtime/configuration and original source
receipts, attach these callbacks to the public selector for ONE decision, and
derive the counters from that decision's production-stat deltas. Construction
alone is insufficient; these observations are not atomic-work validation or
an explicit comparison with a native invocation's input hash.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib

from ..root_prior_correspondence import boundary_fingerprint


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _recipe(config):
    expected = dict(worlds=1, search_sims=1024, search_batch=16, search_depth=6,
        threads=1, model_world_workers=1, leaf_eval="model", model_leaf_override=None,
        early_stop=False, model_decision_time_ms=None, depth_min=None, worlds_min=None,
        strict_fallbacks=True)
    actual = {name: getattr(config, name, object()) for name in expected}
    # Main's original model path predates the optional leaf-override seam.
    actual["model_leaf_override"] = getattr(config, "model_leaf_override", None)
    _require(all(type(actual[k]) is type(v) and actual[k] == v for k, v in expected.items()),
             "fixed one-world model configuration required")
    return actual


class OneWorldTeamCapture:
    """Copy actual attempted completions; reconcile only a successful decision.

    Engine observers swallow exceptions. Consequently the reader must reject
    gaps against attempted/constructed/searched counters, not infer that an
    absent callback means no construction refusal occurred. Original packed
    rosters describe INITIAL sets, not current items, moves, HP or abilities.
    Truth must remain outside this public-selector observer.
    """

    def __init__(self, context, config, *, boundary_key=boundary_fingerprint):
        self.recipe = _recipe(config)
        self.boundary_key = boundary_key
        self.binding = self._binding(context)
        self.attempts = []
        self.constructed = []

    def _binding(self, context):
        seed, index = context.seed, context.decision_round_index
        _require(type(seed) is int and 0 <= seed < 2**32 and type(index) is int and index >= 0
                 and type(context.battle_id) is str and context.battle_id
                 and context.player_id == "p1", "original p1 source boundary required")
        digest = self.boundary_key(context)
        _require(type(digest) is str and len(digest) == 64
                 and all(c in "0123456789abcdef" for c in digest), "boundary hash required")
        return dict(seed=seed, battle_id=context.battle_id, round=index,
                    player_id=context.player_id, boundary_sha256=digest)

    def _check(self, context):
        _require(self._binding(context) == self.binding, "world callback boundary changed")

    def attempt(self, context, ordinal, override):
        self._check(context)
        _require(type(ordinal) is int and ordinal == len(self.attempts) and not self.constructed,
                 "complete original attempt order required")
        teams = None
        if override is not None:
            supplied = override.player_teams
            _require(isinstance(supplied, Mapping) and set(supplied) == {"p1", "p2"}
                     and all(type(v) is str and v for v in supplied.values()),
                     "actual packed initial teams required")
            teams = deepcopy(dict(supplied))
        self.attempts.append(dict(ordinal=ordinal, initial_packed_teams=teams))

    def world(self, context, world, state):
        self._check(context)
        _require(self.attempts and not self.constructed
                 and self.attempts[-1]["initial_packed_teams"] is not None,
                 "exactly one constructed world needs its actual attempt")
        sides, parties = dict(world.slot_sides), dict(world.party_species)
        _require(set(sides) == set(parties) == {"p1", "p2"}
                 and set(sides.values()) == {"side_one", "side_two"}
                 and all(isinstance(p, (tuple, list)) and p
                         and all(type(s) is str and s for s in p) for p in parties.values()),
                 "constructed world identity maps required")
        serialized = state.to_string()
        _require(type(serialized) is str and serialized, "constructed state serialization required")
        self.constructed.append(dict(attempt_ordinal=self.attempts[-1]["ordinal"],
            slot_sides=deepcopy(sides), party_species={k: list(v) for k, v in parties.items()},
            constructed_state_sha256=hashlib.sha256(serialized.encode()).hexdigest()))

    def finish(self, context, config, *, attempts, constructed, searched, iterations):
        self._check(context)
        _require(_recipe(config) == self.recipe, "capture configuration changed")
        _require(all(type(n) is int for n in (attempts, constructed, searched, iterations))
                 and attempts == len(self.attempts) and attempts > 0
                 and constructed == len(self.constructed) == searched == 1 and iterations == 1024,
                 "complete callbacks and successful one-world work required")
        return dict(schema="pokezero.search-over-raw.initial-team-capture.v1",
            binding=deepcopy(self.binding), recipe=deepcopy(self.recipe),
            attempts=deepcopy(self.attempts), constructed=deepcopy(self.constructed),
            work=dict(attempts=attempts, constructed=constructed, searched=searched, iterations=iterations),
            association="pinned one-world fixed-work path and callback cardinality",
            explicit_native_input_hash_comparison=False, source_truth_bound=False,
            current_state_accuracy_established=False, scientific_strength_claim=False,
            work_evidence="production counters; not atomic backup validation")
