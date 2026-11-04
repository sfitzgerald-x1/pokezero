"""Oracle functions the CUDA engine replaces, exactly as the V2 collector calls them.

Each factory closes over the prepared tables and returns a pure per-lane
function of the lane carry. gpu_engine.v2e_gen compiles them to CUDA; the
harness (gpu_engine.v2e_diff) and the collector use the same definitions.
"""
from contextlib import contextmanager
import threading

import jax
import jax.numpy as j

from gpu_engine import transition, engine_device_trackers as trackers, engine_device_observation as encoder
from gpu_engine import benchmark_device_selfplay as sampling, v5_device_rollout as rollout
from gpu_engine import public_belief_features, lane_index


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


_lane_index_check = lane_index._check


def _check_plain_safe(x, rows, indices):
    """lane_index._check, refusing the one index type whose plain form differs from the select form:
    unsigned 32-bit indices at or above 2**31 (plain indexing converts them to int32, so they wrap
    negative and clamp to 0, where the select form clamps to n - 1)."""
    for index in indices:
        dtype = j.asarray(index).dtype
        if j.issubdtype(dtype, j.unsignedinteger) and dtype.itemsize >= 4:
            raise TypeError('unsigned 32-bit lane index: the plain indexing form differs from the select form at >= 2**31')
    return _lane_index_check(x, rows, indices)


# lane_index's one-hot select forms exist so XLA fuses per-lane point reads and writes. For the CUDA
# trace they only inflate the program (the step jaxpr drops from 1.59M to 0.60M equations without
# them), so it uses the plain indexing they were proven bitwise equal to in stage 1; the harness
# still compares against the unmodified oracle.
SUBSTITUTIONS = ((public_belief_features, '_first_occurrences', _first_occurrences_unfused), (lane_index, 'LIMIT', 0),
    (lane_index, '_check', _check_plain_safe))
_substituting = threading.RLock()


@contextmanager
def trace_substitutions():
    """Apply SUBSTITUTIONS for one CUDA trace. Process-wide, so serialized; JAX's trace caches are
    cleared on exit so no jaxpr traced under them is reused by later (oracle) tracing."""
    with _substituting:
        saved = [(module, name, getattr(module, name)) for module, name, _ in SUBSTITUTIONS]
        try:
            for module, name, value in SUBSTITUTIONS:setattr(module, name, value)
            yield
        finally:
            for module, name, value in saved:setattr(module, name, value)
            jax.clear_caches()


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


def step(engine_tables, observation_tables):
    """The collector's engine step with public carry: transition.step_with_public as v2_collector calls it."""
    def fn(state, actions, public):
        return transition.step_with_public(state, actions, engine_tables, public, belief_tables=observation_tables.belief)
    return fn
