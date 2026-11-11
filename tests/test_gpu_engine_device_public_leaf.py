"""Synthetic root/world facts only: no engine/model/forward/game execution."""
import ast
from pathlib import Path
import unittest
import jax
import jax.numpy as j
import numpy as np

from gpu_engine import device_public_leaf as leaf
from gpu_engine import engine_device_observation as observation, engine_device_trackers as public
import test_gpu_engine_device_observation as fixtures


class DevicePublicLeafTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeviceObservationTests.setUpClass()
        cls.fixture = fixtures.DeviceObservationTests()
        cls.tables = cls.fixture.tables

    def facts(self):
        state, _, _, legal = self.fixture.facts(b=3)
        hp = state.hp.at[:, :, 1].set(j.asarray([[101, 120], [140, 160], [180, 200]], j.int32))
        state = state._replace(hp=hp, turn=j.asarray([11, 22, 33], j.int32),
            spikes=j.asarray([[0, 1], [1, 2], [2, 3]], j.int32))
        root = public.initialize(state)  # actual ROOT fixture, not a sampled world
        indices = j.asarray([2, 0, 2, 1], j.int32)
        worlds = jax.tree.map(lambda value: value[indices], state)
        owners = j.asarray([0, 1, 1, 0], j.int32)
        return root, indices, worlds, owners, legal[indices]

    def same(self, a, b):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(x, y)

    def test_mixed_roots_seats_repeat_same_encoder_and_dispatch_jit(self):
        root, indices, worlds, owners, legal = self.facts()
        fn = jax.jit(lambda w, r, i, o, m: leaf.from_roots(w, r, i, self.tables,
            observer_sides=o, legal_action_mask=m, enable_device_leaf=True))
        got = fn(worlds, root, indices, owners, legal)
        cloned = public.clone_for_worlds(root, indices)
        direct = observation.encode(worlds, public.readout(cloned, owners), self.tables,
            observer_sides=owners, legal_action_mask=legal)
        self.same(got.public, cloned); self.same(got.observation, direct)
        self.same(got.inputs, observation.actor_inputs(direct))
        self.assertEqual(got.inputs['numeric_features'].shape, (4, 1, 29, 132))
        actions = j.asarray([0, 4, 1, 4], j.int32)
        self.same(leaf.map_actions(got, worlds, actions), observation.map_actions(direct, worlds, owners, actions))
        np.testing.assert_allclose(got.observation.numeric_features[:, 0, observation._index('TURN_COUNT')], [33/1000, 11/1000, 33/1000, 22/1000], atol=1e-7)

    def test_sampled_foe_is_inert_own_world_stats_are_not_a_privileged_critic(self):
        root, indices, worlds, owners, legal = self.facts()
        call = lambda w: leaf.from_roots(w, root, indices, self.tables,
            observer_sides=owners, legal_action_mask=legal, enable_device_leaf=True)
        baseline = call(worlds); changed = worlds; rows=j.arange(4); foe=1-owners
        for name in ('species', 'level', 'gender', 'hp', 'max_hp', 'stats', 'types', 'ability',
                'base_ability', 'item', 'moves', 'pp', 'disabled', 'choice_move', 'encore_move', 'encore_counter', 'charge_move'):
            value=getattr(changed,name)
            value=value.at[rows,foe].set(~value[rows,foe] if value.dtype==j.bool_ else value[rows,foe]+999)
            changed=changed._replace(**{name:value})
        self.same(call(changed), baseline)
        own=worlds._replace(stats=worlds.stats.at[rows,owners,1,1].set(300))
        got=call(own)
        np.testing.assert_allclose(got.observation.numeric_features[:,1,observation._index('ACTUAL_ATK')],300/714,atol=1e-6)
        self.same(got.observation.numeric_features[:,7:13],baseline.observation.numeric_features[:,7:13])
        self.same(got.observation.numeric_features[:,23:],baseline.observation.numeric_features[:,23:])

    def test_advanced_public_carry_legacy_explicit_and_no_world_initialization(self):
        root, indices, worlds, owners, legal = self.facts()
        start=leaf.from_roots(worlds,root,indices,self.tables,observer_sides=owners,
            legal_action_mask=legal,enable_device_leaf=True)
        # Declared synthetic actual-public-step result: these are literal public
        # writes, not a reconstructed event or a physical engine execution.
        advanced=start.public._replace(turn=start.public.turn+1,
            hp_percent=start.public.hp_percent.at[:, :, 0].set(25))
        got=leaf.from_public(worlds,advanced,self.tables,observer_sides=owners,
            legal_action_mask=legal,enable_device_leaf=True)
        np.testing.assert_allclose(got.observation.numeric_features[:,7,observation._index('HP_FRACTION')],.25,atol=1e-7)
        self.same(start.public,public.clone_for_worlds(root,indices))
        legacy=leaf.from_public(worlds,advanced,self.tables,observer_sides=owners,
            legal_action_mask=legal,enable_device_leaf=True,legacy_champion=True)
        self.same(legacy.inputs,observation.champion_inputs(got.observation))
        for enabled in (False,1,None):
            with self.assertRaises(ValueError):leaf.from_roots(None,None,None,None,
                observer_sides=None,legal_action_mask=None,enable_device_leaf=enabled)
        tree=ast.parse(Path(leaf.__file__).read_text())
        called={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertIn('clone_for_worlds',called);self.assertIn('actor_inputs',called)
        self.assertFalse({'initialize','empty_state','step','step_with_chance'}&called)


if __name__=='__main__':unittest.main()
