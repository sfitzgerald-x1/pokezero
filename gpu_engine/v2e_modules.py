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
    cleared on entry (so no oracle jaxpr traced earlier in the process is reused under the
    substitutions) and on exit (so no jaxpr traced under them is reused by later oracle tracing)."""
    with _substituting:
        saved = [(module, name, getattr(module, name)) for module, name, _ in SUBSTITUTIONS]
        try:
            jax.clear_caches()
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


# ---------------------------------------------------------------------------
# The D3 frame pricer's per-world engine and encoder work (device_public_teacher_batch.make_kernel),
# as pure per-lane functions. The pricer maps one-row worlds (every state and public leaf carries a
# batch axis of 1) with jax.vmap; each lane function here is that same vmap of the same oracle
# calls that device_public_teacher.make_executor's callables make, so the lane axis is the mapped
# world axis. The joint maps (world, branch) pairs: lane = world * 81 + branch, every lane carrying
# its world's root state, public carry, dispatch targets and chance capsule, and its own action pair.

PRICER_JOINT_LANES = 81
PRICER_MODES = dict(views='warp', sample='thread', joint='thread')
_DISPATCH_FIELDS = ('switch_targets', 'move_targets')   # what observation.map_actions reads of a root Observation
_ROOT_FIELDS = ('legal_action_mask', 'error') + _DISPATCH_FIELDS   # what the pricer reads of a root Observation


def _only(observation, fields):
    return observation._replace(**{name: None for name in observation._fields if name not in fields})


def dispatch_targets(observation):
    """The parts of a root Observation the joint's action dispatch reads (observation.map_actions):
    the switch targets and, when present, the request-slot move targets."""
    return _only(observation, _DISPATCH_FIELDS)


def pricer_lane_functions(legal, observe, dispatch, advance, sample):
    """{'views', 'sample', 'joint'} lane functions over one-row-world callables with the signatures of
    device_public_teacher.Executor's (legal(state), observe(state, public, owner, mask) -> Leaf,
    dispatch(observation, state, owner, action), advance(state, public, actions, chance), sample(state))."""
    def views_one(state, public, owner):
        # make_kernel's `views`: both seats' root observations (legal masks, errors and dispatch
        # targets; the encoder arrays are returned once, as actor inputs).
        masks = legal(state);seat = j.asarray([owner], j.int32)
        own = observe(state, public, seat, masks[:, owner])
        peer = observe(state, public, 1 - seat, masks[:, 1 - owner])
        return _only(own.observation, _ROOT_FIELDS), _only(peer.observation, _ROOT_FIELDS), own.inputs, peer.inputs
    def joint_one(state, public, own, peer, owner, requested, pair, chance):
        # device_public_teacher._make_joint's `single`; own/peer are dispatch_targets of the root observations.
        actions = j.full((1, 2), -1, j.int32)
        chosen = dispatch(own, state, owner, pair[:1])
        actions = actions.at[:, owner[0]].set(chosen)
        reply = dispatch(peer, state, 1 - owner, pair[1:])
        actions = actions.at[:, 1 - owner[0]].set(j.where(requested, reply, -1))
        after = advance(state, public, actions, chance)
        masks = legal(after.step.state)
        view = observe(after.step.state, after.trackers, owner, masks[:, owner[0]])
        return (after.step.invalid_actions, after.step.state.done, after.step.state.winner, view.observation.error, view.inputs)
    def views(states, publics, owners):
        return jax.vmap(views_one)(states, publics, owners)
    def joint(states, publics, own, peer, owners, requested, pairs, chance):
        return jax.vmap(joint_one)(states, publics, own, peer, owners, requested, pairs, chance)
    def chance(states):
        return jax.vmap(sample)(states)
    return dict(views=views, sample=chance, joint=joint)


def _pricer_parts(engine_tables, observation_tables):
    """device_public_teacher.make_executor's legal/observe/dispatch/advance/sample, unjitted."""
    from gpu_engine import device_public_leaf as leaf, engine_device_observation as observation
    def legal(state):
        return transition.action_mask(state, engine_tables) & sampling.request_seats(state)[..., None]
    def observe(state, public, owner, mask):
        return leaf.from_public(state, public, observation_tables, observer_sides=owner, legal_action_mask=mask,
            enable_device_leaf=True)
    def advance(state, public, actions, chance):
        return transition.step_with_public(state, actions, engine_tables, public,
            belief_tables=observation_tables.belief, chance_inputs=chance)
    return legal, observe, observation.map_actions, advance, transition.sample_chance_inputs


_pricer_functions = []


def pricer_functions(engine_tables, observation_tables):
    """(index, {'views', 'sample', 'joint'}) lane functions for these tables; the same function
    objects on every call for the same tables (v2e_gen's registry keys a generated module by name
    and function), index distinguishing table sets within the process."""
    for index, (e, o, fns) in enumerate(_pricer_functions):
        if e is engine_tables and o is observation_tables:return index, fns
    fns = pricer_lane_functions(*_pricer_parts(engine_tables, observation_tables))
    _pricer_functions.append((engine_tables, observation_tables, fns))
    return len(_pricer_functions) - 1, fns
