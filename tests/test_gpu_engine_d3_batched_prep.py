"""Batched D3 preparation pieces equal their one-root originals (CPU)."""
import unittest

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_gumbel
from gpu_engine import device_public_worlds as plans
from gpu_engine import device_public_world_state as mapping
from gpu_engine.state import empty_state
from gpu_engine.v5_device_rollout import RootContext
import test_gpu_engine_device_public_teacher as teacher_fixtures


def batch_context(rows=5):
    """facts()'s synthetic root, tiled to `rows` rows with distinct per-row public HP percentages
    (so the encoder's hp_fraction = percent / 100 takes many values), streams and frames (two of
    them frame 0, where the history flag depends on the view)."""
    context, _, _, tables, _ = teacher_fixtures.facts()
    tile = lambda x: j.concatenate([x] * rows)
    state, public = jax.tree.map(tile, context.state), jax.tree.map(tile, context.public)
    # Own-only cells that materialization copies unchecked, distinct by row and side, so a wrong-row
    # or wrong-side own gather cannot pass.
    row, side = j.arange(rows)[:, None, None], j.arange(2)[None, :, None]
    state = state._replace(happiness=(200 + 10 * row + 3 * side + j.arange(6)[None, None, :]).astype(state.happiness.dtype),
        pp=(10 - (row + 2 * side) % 7)[..., None].repeat(4, axis=-1).repeat(6, axis=-2).astype(state.pp.dtype))
    percent = (100 - 7 * j.arange(rows)[:, None, None] - 3 * j.arange(2)[None, :, None] - 11 * j.arange(6)[None, None, :]) % 101
    percent = percent.at[0].set(100).astype(public.hp_percent.dtype)   # row 0: full health, frame 0
    public = public._replace(hp_percent=percent)
    frames = j.asarray([0, 2, 3, 0, 5][:rows] + [1] * max(0, rows - 5), j.int32)
    return RootContext(state, public, 81 + j.arange(rows, dtype=j.int32), frames), tables


class DetachRootsTest(unittest.TestCase):
    def same_bytes(self, a, b, path='root'):
        """Same containers (types, dict key ORDER: jax's tree utilities sort keys), same array bytes."""
        if hasattr(a, 'keys') and hasattr(a, 'items'):
            self.assertTrue(hasattr(b, 'keys'), path);self.assertEqual(list(a.keys()), list(b.keys()), path)
            for key in a.keys():self.same_bytes(a[key], b[key], f'{path}.{key}')
        elif isinstance(a, (tuple, list)):
            self.assertIs(type(a), type(b), path);self.assertEqual(len(a), len(b), path)
            for i, (x, y) in enumerate(zip(a, b)):self.same_bytes(x, y, f'{path}[{getattr(a, "_fields", range(len(a)))[i]}]')
        elif a is None or b is None:
            self.assertIs(a, b, path)
        else:
            x, y = np.asarray(a), np.asarray(b)
            self.assertEqual((x.dtype, x.shape), (y.dtype, y.shape), path)
            self.assertEqual(x.tobytes(), y.tobytes(), path)

    def test_batched_roots_equal_one_root_detach_and_materialize(self):
        context, tables = batch_context()
        source = tables.source
        pairs = [(r, o) for r in range(5) for o in (0, 1)]
        reference = {(r, o): plans.detach_root(context, source, row=r, owner=o, enable_device_worlds=True) for r, o in pairs}
        self.assertEqual(len({bytes(np.asarray(x.own['happiness'])) for x in reference.values()}), len(pairs))
        for chosen in (pairs[3:4], pairs, [pairs[i % 10] for i in range(64)], [pairs[(7 * i + 3) % 10] for i in range(65)]):
            batch = plans.detach_roots(context, source, rows=[r for r, _ in chosen], owners=[o for _, o in chosen],
                enable_device_worlds=True)
            self.assertEqual(len(batch), len(chosen))
            for (row, owner), new in zip(chosen, batch):
                old = reference[(row, owner)]
                self.assertEqual(new.address, old.address)
                self.assertEqual((new.source_binding_sha256, new.reply_private_history_available),
                    (old.source_binding_sha256, old.reply_private_history_available))
                self.same_bytes(dict(new.own), dict(old.own))
                self.same_bytes(dict(new.view), dict(old.view))
                self.same_bytes(new.public, old.public)
                self.assertTrue(all(not np.asarray(x).flags.writeable for x in jax.tree.leaves(new.public)))
        # Distinct percentages really reach the float encoder view.
        fractions = {float(x) for root in batch for x in np.asarray(root.view['hp_fraction']).ravel()}
        self.assertGreater(len(fractions), 20)
        self.assertTrue(any(root.reply_private_history_available for root in batch))

    def test_host_materialization_equals_original(self):
        context, tables = batch_context()
        source = tables.source;checked = 0
        pairs = [(r, o) for r in range(5) for o in (0, 1)]
        batch = plans.detach_roots(context, source, rows=[r for r, _ in pairs], owners=[o for _, o in pairs], enable_device_worlds=True)
        for (row, owner), new in zip(pairs, batch):
            old = plans.detach_root(context, source, row=row, owner=owner, enable_device_worlds=True)
            recipe = next(v5_gumbel.Recipe(seed) for seed in range(400) if v5_gumbel.Recipe(seed).selected(old.address))
            starts_old, starts_new = plans.plan_starts(old, source, recipe), plans.plan_starts(new, source, recipe)
            self.assertEqual((starts_old.selected, starts_old.starts), (starts_new.selected, starts_new.starts))
            keys = dict(zip(*(lambda seeds: (seeds, mapping.world_keys(seeds)))(
                [mapping.kernel_seed(new, s.world, recipe) for s in starts_new.starts])))
            for start in starts_old.starts:
                x = mapping.materialize(old, start, tables, recipe)
                y = mapping.materialize_host(new, start, tables, recipe, keys=keys)
                self.assertEqual((x.index, x.refusals, x.kernel_seed, x.reply_private_history_available),
                    (y.index, y.refusals, y.kernel_seed, y.reply_private_history_available))
                self.assertEqual(x.state is None, y.state is None)
                if x.state is not None:
                    self.same_bytes(x.state, y.state);self.same_bytes(x.public, y.public);checked += 1
        self.assertGreater(checked, 0)

    def test_detach_failure_is_not_a_refusal(self):
        context, tables = batch_context()
        with self.assertRaises(ValueError):
            plans.detach_roots(context, tables.source, rows=[0, 9], owners=[0, 0], enable_device_worlds=True)


class WorldKeysTest(unittest.TestCase):
    def test_world_keys_equal_empty_state_keys(self):
        rng = np.random.default_rng(0)
        edge = [0, 1, 2, 2**31 - 1, 2**31, 2**32 - 2, 2**32 - 1]
        seeds = edge + [int(x) for x in rng.integers(0, 2**32, 64, dtype=np.uint64)]
        for chosen in (seeds[:1], seeds[:64], seeds, seeds[::-1][:65]):   # bucket edges: 1, 64, 71, 65
            keys = mapping.world_keys(chosen)
            self.assertEqual(keys.shape, (len(chosen), 1, 2))
            for seed, key in zip(chosen, keys):
                np.testing.assert_array_equal(key, np.asarray(empty_state(1, seed=seed).keys))

    def test_empty_template_matches_empty_state(self):
        arrays = mapping._empty_arrays()
        reference = empty_state(1, seed=12345)._asdict()
        self.assertEqual(set(arrays), set(reference))
        for name, value in reference.items():
            if name == 'keys':continue
            value = np.asarray(value)
            self.assertEqual((arrays[name].dtype, arrays[name].shape), (value.dtype, value.shape), name)
            np.testing.assert_array_equal(arrays[name], value)

    def test_template_copies_are_independent(self):
        a = mapping._empty_arrays();a['hp'][0, 0, 0] = 99
        self.assertEqual(int(mapping._empty_arrays()['hp'][0, 0, 0]), 0)


class GateCompareTest(unittest.TestCase):
    def test_structure_is_exact_and_floats_have_tolerance(self):
        from gpu_engine.d3_saved_root_gate import differences, FLOAT_TOLERANCE
        row = dict(status='MATCH', worlds=[1, 2], target=dict(delta=0.5, top=True))
        self.assertEqual(differences(row, dict(row, target=dict(delta=0.5 + FLOAT_TOLERANCE / 2, top=True))), [])
        self.assertEqual(differences(row, dict(row, target=dict(delta=0.5 + 1e-6, top=True))), ['.target.delta:float 1e-06'])
        self.assertEqual(differences(row, dict(row, status='MISMATCH')), ['.status:value'])
        self.assertEqual(differences(row, dict(row, worlds=[1, 2, 3])), ['.worlds:length'])
        self.assertEqual(differences(row, dict(row, worlds=[1, 2.0])), ['.worlds[1]:value'])
        self.assertEqual(differences(row, dict(row, extra=1)), [':keys'])
        self.assertEqual(differences(row, dict(row, target=dict(delta=0.5, top=1))), ['.target.top:value'])


if __name__ == '__main__':
    unittest.main()
