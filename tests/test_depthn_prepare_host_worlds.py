"""Host-built protocol worlds (phase_d_protocol_world.HOST_WORLDS): each host (NumPy) step, and a whole materialize_protocol on
the batch-2 transport fixture, equals the eager device build byte for byte (values, dtypes, shapes, leaf types, weak_type,
committed), so a world that differs fails here. Named test_depthn_* so the depthn-light CI job runs it; the full-root parity
runs pod-side on replayed screen roots."""
import unittest

import numpy as np

from gpu_engine import device_public_world_state as states, phase_d_protocol_world as protocol, phase_d_tail as tail
from gpu_engine import v5_hypothetical_state as mapping
from gpu_engine.state import empty_state
import test_gpu_engine_v5_public_world as mapping_tests

SEEDS = (0, 1, 17, 2**31 - 1, 2**31, 2**32 - 1)


def assert_same(test, device, host, label=''):
    for name in device._fields:
        x, y = np.asarray(getattr(device, name)), np.asarray(getattr(host, name))
        test.assertEqual((x.dtype, x.shape), (y.dtype, y.shape), f'{label}{name}')
        test.assertEqual(x.tobytes(), y.tobytes(), f'{label}{name}')


class HostTemplate(unittest.TestCase):
    def test_a_bad_seed_refuses_as_empty_state_does(self):
        for seed in (-1, 2**32, 1.5, np.uint32(3)):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'unsigned 32-bit'):protocol._host_arrays(seed, None)

    def test_host_arrays_equal_empty_state_for_every_seed_and_batched_key(self):
        keys = states.world_keys(list(SEEDS))
        for seed, key in zip(SEEDS, keys):
            device = empty_state(1, seed=seed)
            for batched in (None, key):
                host = protocol._host_arrays(seed, batched)
                self.assertEqual(set(host), set(device._fields))
                for name, value in device._asdict().items():
                    x = np.asarray(value)
                    self.assertEqual((host[name].dtype, host[name].shape, host[name].tobytes()), (x.dtype, x.shape, x.tobytes()), (seed, name))

    def test_each_world_gets_its_own_copy(self):
        a = protocol._host_arrays(5, None);a['hp'][0, 0, 0] = 99;a['keys'][...] = 0
        b = protocol._host_arrays(5, None)
        self.assertEqual(int(b['hp'][0, 0, 0]), 0);self.assertEqual(b['keys'].tobytes(), np.asarray(empty_state(1, seed=5).keys).tobytes())


class HostPut(unittest.TestCase):
    def test_put_equals_the_device_scatter_and_leaves_its_input(self):
        device = empty_state(1, seed=3);host = protocol._host_arrays(3, None)
        for name, index, value in (('hp', (0, 1, 2), 57), ('rest_origin', (0, 0, 1), True), ('volatile', (0, 1, 0), np.uint32(4)),
                                   ('active', (0, 1), 3), ('disabled', (0, 0, 1), np.array([True, False, True, False])),
                                   ('stats', (0, 0, 2), np.arange(6, dtype=np.int32)), ('last_action_index', (0, 0), np.int32(-1))):
            before = host[name].copy()
            x = np.asarray(tail._put(getattr(device, name), index, value));y = tail._put(host[name], index, value)
            self.assertIsInstance(y, np.ndarray)
            self.assertEqual((x.dtype, x.shape, x.tobytes()), (y.dtype, y.shape, y.tobytes()), name)
            self.assertEqual(host[name].tobytes(), before.tobytes(), name)


class HostMapping(unittest.TestCase):
    def test_map_spec_host_build_equals_the_device_build(self):
        spec, metadata = mapping_tests.TypedMappingTests().fixture()
        args = dict(metadata=metadata, turn=7, active_turns=(2, 4), happiness=((7,), (255,)))
        for seed in SEEDS:
            device = mapping.map_spec(spec, seed=seed, **args);host = mapping.map_spec(spec, seed=seed, arrays=protocol._host_arrays(seed, None), **args)
            self.assertEqual(device.refusals, host.refusals);self.assertFalse(host.refusals)
            self.assertIsInstance(host.state.hp, np.ndarray)
            assert_same(self, device.state, host.state, f'seed {seed} ')

    def test_tail_restore_host_equals_device_with_a_pending_residual_wish_and_fresh_substitute(self):
        spec, metadata = mapping_tests.TypedMappingTests().fixture()
        args = dict(metadata=metadata, turn=7, active_turns=(2, 4), happiness=((7,), (255,)))
        facts = dict(replacement=dict(pending_residual=True, queued_actor=-1, turn=6, ages=(1, 2)), fresh_substitutes=dict(p2='snorlax'),
                     substitute_origins={}, wish_set_turns=dict(p1=6), turn=7)
        for seed in SEEDS[:3]:
            device = mapping.map_spec(spec, seed=seed, **args).state;host = mapping.map_spec(spec, seed=seed, arrays=protocol._host_arrays(seed, None), **args).state
            a = tail.restore(device, spec, (0, 0), facts, metadata=metadata, checks=[]);b = tail.restore(host, spec, (0, 0), facts, metadata=metadata, checks=[])
            assert_same(self, a, b, f'seed {seed} ')
            self.assertEqual(int(np.asarray(b.substitute_hp)[0, 1, 0]), 50)   # max HP 200 // 4
            self.assertEqual(int(np.asarray(b.wish_counter)[0, 0]), 1)


class HostRestoreReplacement(unittest.TestCase):
    def test_a_fainted_cursors_last_move_choice_lock_and_struggle_restore_alike(self):
        from dataclasses import replace
        spec, metadata = mapping_tests.TypedMappingTests().fixture()
        args = dict(metadata=metadata, turn=7, active_turns=(2, 4), happiness=((7,), (255,)))
        facts = dict(replacement=dict(pending_residual=False, queued_actor=-1, turn=6, ages=(1, 2)), fresh_substitutes={},
                     substitute_origins={}, wish_set_turns={}, turn=7)
        for last, item in (('move:1', 'choiceband'), ('move:struggle', None), ('switch:0', None)):
            fainted = replace(spec.side_one, pokemon=(replace(spec.side_one.pokemon[0], hp=0, item=item),), last_used_move=last)
            original = replace(spec, side_one=fainted)
            device = mapping.map_spec(spec, seed=4, **args).state;host = mapping.map_spec(spec, seed=4, arrays=protocol._host_arrays(4, None), **args).state
            a = tail.restore(device, original, (0, 0), facts, metadata=metadata, checks=[]);b = tail.restore(host, original, (0, 0), facts, metadata=metadata, checks=[])
            with self.subTest(last=last):assert_same(self, a, b, f'{last} ')
        self.assertEqual(int(np.asarray(b.last_action_kind)[0, 0]), 1)


class EndToEndMaterialize(unittest.TestCase):
    """materialize_protocol on the batch-2 production-transport fixture (own update, foe carry, sleep restore, held counts,
    rebase and transfer all real; only the bridge, the start unpacker and the revealed posterior are patched), host versus
    device: the world's state and public carry leaf for leaf."""
    def test_host_and_device_worlds_are_identical_leaf_for_leaf(self):
        import jax
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        import test_phase_d_batch2_facts as transport
        public, start, inputs, physical, world = transport.ProductionTransport().fixture()
        # The fixture's carry is a namespace for the rebase it patches; here the rebase is real, so the carry is a NamedTuple with
        # the leaves _rebase writes as from_b2 builds them (uncommitted device arrays) and the fixture's own leaves alongside.
        import dataclasses
        from typing import NamedTuple
        import jax.numpy as jnp
        History = NamedTuple('History', [('hp', object), ('maximum', object)])
        own = vars(inputs.root.public)
        Carry = NamedTuple('Carry', [(name, object) for name in (*own, 'damage_history', 'physical_to_public', 'active_physical', 'own_order')])
        carry = Carry(**own, damage_history=History(jnp.full((1, 2), 7, jnp.int32), jnp.full((1, 2), 9, jnp.int32)),
                      physical_to_public=jnp.full((1, 2, 6), -1, jnp.int32), active_physical=jnp.zeros((1, 2), jnp.int32),
                      own_order=jnp.tile(jnp.arange(6, dtype=jnp.int32), (1, 2, 1)))
        inputs = dataclasses.replace(inputs, root=dataclasses.replace(inputs.root, public=carry))
        unpack =lambda team: tuple(NS(species=part.split('|')[0], ability='', item='') for part in team.split(']'))
        made = {}
        for host in (False, True):
            state = jax.tree_util.tree_map(lambda x: np.array(x), physical) if host else physical
            with patch.object(protocol, 'HOST_WORLDS', host), patch.object(protocol, 'battle_spec_from_payload', return_value=world), \
                    patch.object(protocol, 'map_spec', return_value=NS(state=state)), patch.object(protocol, 'unpack_team', side_effect=unpack), \
                    patch.object(protocol.public_worlds, '_revealed', return_value=()):
                made[host] = protocol.materialize_protocol(public, start, inputs=inputs, seed=17)
        a, b = made[False], made[True]
        self.assertIsNotNone(a.state, a.refusals);self.assertEqual(a.refusals, b.refusals)
        for label, x, y in (('state', a.state, b.state), ('public', a.public, b.public)):
            xs, ys = jax.tree_util.tree_flatten_with_path(x)[0], jax.tree_util.tree_flatten_with_path(y)[0]
            self.assertEqual([jax.tree_util.keystr(p) for p, _ in xs], [jax.tree_util.keystr(p) for p, _ in ys])
            for (path, u), (_, v) in zip(xs, ys):
                name = label + jax.tree_util.keystr(path)
                self.assertEqual(type(u), type(v), name)
                self.assertEqual((getattr(u, 'weak_type', None), getattr(u, 'committed', None)), (getattr(v, 'weak_type', None), getattr(v, 'committed', None)), name)
                if hasattr(u, 'dtype'):
                    p, q = np.asarray(u), np.asarray(v)
                    self.assertEqual((p.dtype, p.shape, p.tobytes()), (q.dtype, q.shape, q.tobytes()), name)
                else:self.assertEqual(u, v, name)


class DeviceTransfer(unittest.TestCase):
    def test_device_leaves_keep_dtype_and_bytes(self):
        import jax
        spec, metadata = mapping_tests.TypedMappingTests().fixture()
        host = mapping.map_spec(spec, metadata=metadata, seed=9, turn=7, active_turns=(2, 4), happiness=((7,), (255,)),
                                arrays=protocol._host_arrays(9, None)).state
        moved = protocol._device(host)
        for name in host._fields:
            x = getattr(moved, name);self.assertIsInstance(x, jax.Array, name)
            self.assertEqual((np.asarray(x).dtype, np.asarray(x).tobytes()), (getattr(host, name).dtype, getattr(host, name).tobytes()), name)
            self.assertFalse(getattr(x, 'weak_type', False), name)


if __name__ == '__main__':
    unittest.main()
