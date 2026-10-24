"""Finite chance seam contracts, not a fidelity or probability certification."""
from unittest.mock import patch
import importlib.util
import unittest

AVAILABLE=bool(importlib.util.find_spec('jax'))
if AVAILABLE:
    import jax
    import jax.numpy as j
    import numpy as np
    from gpu_engine import transition as t
    from gpu_engine.state import empty_state


def fixture(batch=3):
    state = empty_state(batch,seed=29)
    state = state._replace(
        stats=j.full_like(state.stats,200).at[...,0].set(100),
        level=j.full_like(state.level,100),hp=j.full_like(state.hp,100),
        max_hp=j.full_like(state.hp,100),moves=j.ones_like(state.moves),
        pp=j.full_like(state.pp,10),types=state.types.at[...,0].set(1),
        done=j.zeros_like(state.done))
    table = t.MoveTable(j.array([0,50,0,50]),j.array([0,1,1,0]),
        j.array([False,True,False,True]),j.array([100]*4),j.array([True]*4),
        j.zeros(4,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(3))
    return state,table


def equal(a,b):
    for left,right in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
        np.testing.assert_array_equal(left,right)


def test_original_random_streams_are_unchanged():
    state,_ = fixture()
    chance = t.sample_chance_inputs(state)
    keys = jax.vmap(lambda key:jax.random.split(key,2))(state.keys)[:,1]
    equal(chance.draws,jax.vmap(lambda key:jax.random.randint(key,(2,12),0,
        j.array([2,100,100,100,16,16,100,4,8,2,2,2],j.int32)))(keys))
    for field,salt,limit in [('phazing',271828,60),('secondary',1618033,100),
                             ('sleep_talk',1414213,60)]:
        equal(getattr(chance,field),jax.vmap(lambda key:jax.random.randint(
            jax.random.fold_in(key,salt),(2,),0,limit))(keys))
    equal(chance.entry_ties,jax.vmap(lambda key,count:jax.random.randint(
        jax.random.fold_in(key,j.uint32(982451653)+count.astype(j.uint32)),(2,),0,2))(
            state.keys,state.replacement_count))
    equal(chance.initial_tie,jax.vmap(lambda key:jax.random.randint(
        jax.random.fold_in(key,314159),(),0,2))(state.keys))


def _check_sampled_and_explicit_paths_are_identical(boundary):
    state,table = fixture()
    actions = j.zeros((3,2),j.int32)
    if boundary in ('replacement','deferred'):
        state = state._replace(hp=state.hp.at[:,0,0].set(0))
        actions = actions.at[:,0].set(4)
    if boundary == 'deferred':
        old = t.sample_chance_inputs(state)
        state = state._replace(pending_residual=j.ones(3,j.bool_),
            residual_draws=old.draws,phazing_draws=old.phazing,
            secondary_draws=old.secondary,sleep_talk_draws=old.sleep_talk)
    if boundary == 'invalid':actions=actions.at[:,0].set(9)
    if boundary == 'terminal':state=state._replace(done=j.ones(3,j.bool_))
    chance = t.sample_chance_inputs(state)
    expected = t.step(state,actions,table)
    # No hidden randint is permitted inside the explicit transition body.
    with patch.object(jax.random,'randint',side_effect=AssertionError('hidden chance')):
        actual = t.step_with_chance(state,actions,table,chance)
    equal(expected,actual)


def test_sampled_and_explicit_paths_are_identical():
    for boundary in ('ordinary','replacement','deferred','invalid','terminal'):
        _check_sampled_and_explicit_paths_are_identical(boundary)


def test_explicit_path_is_jittable_and_preserves_reused_draws():
    state,table = fixture(1)
    actions = j.zeros((1,2),j.int32)
    chance = t.sample_chance_inputs(state)
    equal(t.step_with_chance(state,actions,table,chance),
          jax.jit(t.step_with_chance)(state,actions,table,chance))


def test_deferred_turn_reuses_saved_draws_not_new_supplied_draws():
    state,table = fixture(1)
    old = t.sample_chance_inputs(state)
    state = state._replace(hp=state.hp.at[:,0,0].set(0),
        pending_residual=j.ones(1,j.bool_),queued_actor=j.ones(1,j.int32),
        residual_draws=old.draws,phazing_draws=old.phazing,
        secondary_draws=old.secondary,sleep_talk_draws=old.sleep_talk)
    new = old._replace(draws=j.zeros_like(old.draws),
        phazing=j.zeros_like(old.phazing),secondary=j.zeros_like(old.secondary),
        sleep_talk=j.zeros_like(old.sleep_talk))
    assert not np.array_equal(old.draws,new.draws)
    actions=j.array([[4,0]],j.int32)
    equal(t.step_with_chance(state,actions,table,old),
          t.step_with_chance(state,actions,table,new))


def test_initial_entry_tie_is_explicit_and_changes_weather_order():
    from gpu_engine import entry,weather
    state,table=fixture(1)
    meta={'vocabulary':{'abilities':['_none','drizzle','drought'],
        'types':['_none','flying','ground'],'moves':['_none','tackle','splash','struggle']}}
    table=table._replace(entry=entry.from_metadata(meta))
    state=state._replace(entry_pending=j.ones((1,2),j.bool_),
        ability=state.ability.at[0,0,0].set(1).at[0,1,0].set(2))
    chance=t.sample_chance_inputs(state)
    results=[]
    with patch.object(jax.random,'randint',side_effect=AssertionError('hidden chance')):
        for coin in (0,1):
            results.append(t.step_with_chance(state,j.zeros((1,2),j.int32),table,
                chance._replace(initial_tie=j.array([coin],j.int32))).state)
    assert [int(s.weather[0]) for s in results]==[weather.SUN,weather.RAIN]
    assert all(not np.asarray(s.entry_pending).any() for s in results)


def test_shape_and_dtype_rejected_before_transition():
    state,table = fixture(1)
    chance = t.sample_chance_inputs(state)
    for wrong in (j.zeros((1,2,11),j.int32),chance.draws.astype(j.float32)):
        with unittest.TestCase().assertRaisesRegex(ValueError,'chance inputs'):
            t.step_with_chance(state,j.zeros((1,2),j.int32),table,
                chance._replace(draws=wrong))


def load_tests(loader,tests,pattern):
    """Keep optional-JAX checks discoverable by the repository's unittest CI."""
    return unittest.TestSuite(unittest.FunctionTestCase(
        unittest.skipUnless(AVAILABLE,'requires isolated spike JAX')(function))
        for name,function in globals().items() if name.startswith('test_') and callable(function))
