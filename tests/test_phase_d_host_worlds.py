"""Host-built protocol worlds (phase_d_protocol_world.HOST_WORLDS): each host (NumPy) step equals the eager device build byte for
byte (values, dtypes, shapes), so a world that differs fails here. The full-root parity runs pod-side on replayed screen roots."""
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
            self.assertEqual(int(np.asarray(b.substitute_hp)[0, 1, 0]), 50)   # max HP 200 // 4;self.assertEqual(int(np.asarray(b.wish_counter)[0, 0]), 1)


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
