"""Oracle functions the CUDA engine replaces, exactly as the V2 collector calls them.

Each factory closes over the prepared tables and returns a pure per-lane
function of the lane carry. gpu_engine.v2e_gen compiles them to CUDA; the
harness (gpu_engine.v2e_diff) and the collector use the same definitions.
"""
from contextlib import contextmanager

import jax
import jax.numpy as j

from gpu_engine import transition, engine_device_trackers as trackers, engine_device_observation as encoder
from gpu_engine import benchmark_device_selfplay as sampling, v5_device_rollout as rollout
from gpu_engine import public_belief_features


# ---------------------------------------------------------------------------
# Trace-time substitutions. The generator keeps one lane per thread, so oracle
# helpers that flatten the lane axis into rows for a vmap are traced in an
# equivalent unflattened form. Each substitute performs the same integer
# operations per row; the harness compares the generated kernel against the
# unmodified oracle, so any difference would show up as a diff failure.

def _first_occurrences_unfused(ids, selected, count):
    length = ids.shape[-1]
    def one(row, keep):
        positions = j.where(keep & (row > 0), j.arange(length, dtype=j.int32), length)
        return j.full(count, length, j.int32).at[row].min(positions).at[0].set(length)
    f = one
    for _ in range(ids.ndim - 1):f = jax.vmap(f)
    return f(ids, selected)


SUBSTITUTIONS = ((public_belief_features, '_first_occurrences', _first_occurrences_unfused),)


@contextmanager
def trace_substitutions():
    saved = [(module, name, getattr(module, name)) for module, name, _ in SUBSTITUTIONS]
    try:
        for module, name, value in SUBSTITUTIONS:setattr(module, name, value)
        yield
    finally:
        for module, name, value in saved:setattr(module, name, value)


def observe(engine_tables, observation_tables):
    """Action masks, request seats, physical slots and both seats' encoder views.

    Matches v2_collector._make_tick.advance before the forward; error_counts
    (a batch-wide diagnostic the collector never reads) is dropped.
    """
    def fn(state, public):
        b = state.hp.shape[0]
        masks = transition.action_mask(state, engine_tables)
        observations = tuple(encoder.encode(state, trackers.readout(public, j.full(b, s, j.int32)), observation_tables,
            observer_sides=j.full(b, s, j.int32), legal_action_mask=masks[:, s])._replace(error_counts=None) for s in range(2))
        return masks, sampling.request_seats(state), rollout._physical(public), observations
    return fn
