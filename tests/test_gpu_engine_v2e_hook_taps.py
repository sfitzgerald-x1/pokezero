"""CPU tests of the hook taps (gpu_engine.v2e_hook_taps) and the CUDA hook registry (gpu_engine.v2e_hooks): attribute
wrapping with call-site identification, nesting and occurrence numbering, restoration, written-leaf detection, the
per-site outputs, and the oracle-as-cuda / negative self-tests on synthetic trackers (no real tables, no GPU)."""
from pathlib import Path
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_device_public_hooks as H, v2e_hook_taps as taps, v2e_hooks
from gpu_engine.state import empty_state
from gpu_engine.v2e_diff import unsigned

HERE = Path(__file__).stem


def synthetic(b=5):
    """A small battle with living actives and an initialized tracker Context (no belief tables)."""
    state = empty_state(b)
    state = state._replace(hp=j.full((b, 2, 6), 100, j.int32), max_hp=j.full((b, 2, 6), 100, j.int32), species=j.ones((b, 2, 6), j.int32),
        level=j.full((b, 2, 6), 100, j.int32), status=j.zeros((b, 2, 6), j.int32).at[:, 0, 0].set(4))
    c = H.initialize(state)
    return state, c


def bits_equal(a, b):
    return bool(np.array_equal(np.asarray(unsigned(a)), np.asarray(unsigned(b))))


class ArgumentTest(unittest.TestCase):
    def test_parse_argument(self):
        self.assertEqual(taps.parse_argument('observe@all'), (['observe'], 'all'))
        self.assertEqual(taps.parse_argument('observe'), (['observe'], 'all'))
        self.assertEqual(taps.parse_argument('observe,hp_line@756'), (['observe', 'hp_line'], 'transition:756'))
        self.assertEqual(taps.parse_argument('port.condition@engine_device_public_hooks:232'), (['port.condition'], 'engine_device_public_hooks:232'))
        with self.assertRaises(ValueError):taps.parse_argument('@all')

    def test_tappable_names_and_context_detection(self):
        names = set(taps.tappable())
        for name in v2e_hooks.M1_HOOKS + v2e_hooks.M3_HOOKS + v2e_hooks.M2_HOOKS + v2e_hooks.CALLEES:self.assertIn(name, names)
        self.assertTrue(taps.context_first('observe'));self.assertTrue(taps.context_first('_refresh'))
        self.assertFalse(taps.context_first('_write'));self.assertFalse(taps.context_first('port.condition'));self.assertFalse(taps.context_first('damage_history._add'))
        self.assertIs(taps.original('observe'), H.observe)
        with self.assertRaises(KeyError):taps.original('nowhere.fn')
        with self.assertRaises(KeyError):taps.original('Context')


class TapTest(unittest.TestCase):
    def test_direct_call_is_recorded_with_its_site(self):
        state, c = synthetic();before = c.trackers;use = j.ones(5, bool)
        with taps.tap(['observe']) as calls:
            H.observe(c, state, use);line = __import__('inspect').currentframe().f_lineno
        self.assertEqual(len(calls), 1);call = calls[0]
        self.assertEqual((call.name, call.site, call.occurrence), ('observe', f'{HERE}:{line}', None))
        self.assertTrue(call.context_first);self.assertIs(call.before, before);self.assertIs(call.after, c.trackers)
        self.assertIs(call.args[0], state);self.assertEqual(call.kwargs, {})
        written = taps.written_paths(call.before, call.after)
        self.assertIn('.hp_percent', written);self.assertIn('.weather', written);self.assertNotIn('.observed_moves', written)
        self.assertIs(H.observe, taps.original('observe'), 'restored after the block')

    def test_nested_calls_and_occurrences(self):
        state, c = synthetic();use = j.ones(5, bool)
        with taps.tap(['hp_line', 'observe', 'negative_hp_line', 'negative_items.hp_line', 'feature']) as calls:
            H.hp_line(c, state, 0, use, source_class=H.HP_ACTION)
            for _ in range(2):H.observe(c, state, use)
        by_name = {}
        for call in calls:by_name.setdefault(call.name, []).append(call)
        self.assertEqual(len(by_name['hp_line']), 1)
        nested = [c_ for c_ in by_name['observe'] if c_.site.startswith('engine_device_public_hooks:')]
        self.assertEqual([c_.site for c_ in nested], ['engine_device_public_hooks:254'], 'observe called from hp_line is a site of its own')
        self.assertEqual([c_.site for c_ in by_name['negative_hp_line']], ['engine_device_public_hooks:253'])
        self.assertEqual([c_.site for c_ in by_name['negative_items.hp_line']], ['engine_device_public_hooks:442'])
        looped = [c_ for c_ in by_name['observe'] if c_.site.startswith(HERE)]
        self.assertEqual([c_.occurrence for c_ in looped], [0, 1], 'a site hit twice per step numbers its occurrences')
        self.assertEqual([c_.key for c_ in looped], [f'observe@{looped[0].site}#0', f'observe@{looped[0].site}#1'])
        self.assertIsNone(nested[0].occurrence)
        self.assertGreater(len(by_name['feature']), 10, 'feature is reached through observe\'s module-global calls')
        self.assertTrue(all(isinstance(c_.args[0], int) for c_ in by_name['feature']), 'feature(c, side, ...) records side as a static')
        hp = by_name['hp_line'][0];first_nested = min((c_ for c_ in calls if c_.site.startswith('engine_device_public_hooks:')), key=lambda c_: c_.order)
        self.assertLess(hp.order, first_nested.order, 'the enclosing call is recorded at entry, before its nested calls')
        self.assertEqual([c_.order for c_ in calls], list(range(len(calls))))
        self.assertTrue(all(c_.after is not None for c_ in calls));self.assertIsNone(hp.step_line, 'no step_with_chance frame in a direct test call')

    def test_restored_after_exception_and_reentry_refused(self):
        fn = H.observe
        with self.assertRaises(RuntimeError):
            with taps.tap(['observe']):
                self.assertIsNot(H.observe, fn);self.assertIs(taps.original('observe'), fn)
                with self.assertRaises(RuntimeError):
                    with taps.tap(['observe']):pass
                raise RuntimeError('inside')
        self.assertIs(H.observe, fn);self.assertEqual(taps._installed, {})

    def test_two_names_for_one_function_are_refused_and_nothing_leaks(self):
        from gpu_engine import engine_device_damage_history as dh, engine_device_public_port as port
        fn = dh._add
        with self.assertRaises(ValueError):
            with taps.tap(['damage_history._add', 'port._add']):pass
        self.assertIs(dh._add, fn);self.assertIs(port._add, fn);self.assertEqual(taps._installed, {})
        self.assertIs(taps.original('port._add'), fn)
        with taps.tap(['observe']):
            with self.assertRaises(RuntimeError):   # re-entry is refused before any wrapping
                with taps.tap(['hp_line']):pass
        self.assertFalse(getattr(H.observe, '__v2e_tap__', False))

    def test_every_binding_of_a_function_is_wrapped(self):
        """_add is bound in damage_history and imported by name into public_port: both bindings are tapped."""
        from gpu_engine import engine_device_damage_history as dh, engine_device_public_port as port
        fn = dh._add;self.assertIs(port._add, fn)
        with taps.tap(['damage_history._add']) as calls:
            self.assertIsNot(port._add, fn);self.assertIs(port._add, dh._add)
            pair = j.zeros((5, 2), j.float32);port._add(pair, j.ones(5, j.float32))
        self.assertIs(port._add, fn);self.assertEqual(len(calls), 1);self.assertIs(calls[0].before, pair)
        self.assertEqual(taps.written_paths(calls[0].before, calls[0].after), [''], 'an array result is one written leaf')


class OutputsTest(unittest.TestCase):
    def test_site_outputs_written_and_untouched(self):
        state, c = synthetic();before = c.trackers
        H.observe(c, state, j.ones(5, bool));after = c.trackers;written = taps.written_paths(before, after)
        oracle = taps.site_outputs('observe@x', before, after, written, oracle=True)
        self.assertEqual(set(oracle), {f'observe@x/{p}' for p in written} | {'observe@x/untouched_equal'})
        self.assertTrue(bool(np.all(np.asarray(oracle['observe@x/untouched_equal']))))
        same = taps.site_outputs('observe@x', before, after, written, oracle=False)
        self.assertTrue(bool(np.all(np.asarray(same['observe@x/untouched_equal']))))
        bad = after._replace(observed_moves=after.observed_moves.at[2, 0, 0, 0].set(7))   # an untouched leaf changed at lane 2
        flagged = taps.site_outputs('observe@x', before, bad, written, oracle=False)
        np.testing.assert_array_equal(np.asarray(flagged['observe@x/untouched_equal']), [True, True, False, True, True])
        with self.assertRaises(ValueError):taps.site_outputs('observe@x', before, after._replace(observed_moves=after.observed_moves[:, :1]), written, oracle=False)

    def test_structure_change_falls_back_to_the_whole_result(self):
        out = taps.site_outputs('p', j.zeros(3), (j.ones(3), j.zeros(3, bool)), None, oracle=False)
        self.assertEqual(sorted(out), ['p/[0]', 'p/[1]', 'p/err'])

    def test_error_word_channel(self):
        state, c = synthetic();before = c.trackers;H.observe(c, state, j.ones(5, bool));after = c.trackers;written = taps.written_paths(before, after)
        zeros = taps.site_outputs('o', before, after, written, oracle=False)['o/err']
        np.testing.assert_array_equal(np.asarray(zeros), np.zeros(5, np.uint32))
        word = taps.site_outputs('o', before, after, written, oracle=False, err=j.asarray([0x4c41594f], j.uint32))['o/err']
        np.testing.assert_array_equal(np.asarray(word), np.full(5, 0x4c41594f, np.uint32))
        per_lane = taps.site_outputs('o', before, after, written, oracle=False, err=j.asarray([0, 1, 0, 2, 0], j.uint32))['o/err']
        np.testing.assert_array_equal(np.asarray(per_lane), [0, 1, 0, 2, 0])
        with self.assertRaises(ValueError):taps.error_word(j.asarray([1, 2], j.uint32), 5)
        with self.assertRaises(ValueError):taps.error_word(j.asarray([1], j.int32), 5)
        carry, err = taps.split_result((after, j.zeros(1, j.uint32)));self.assertIs(carry, after);self.assertIsNotNone(err)
        carry, err = taps.split_result(after);self.assertIs(carry, after);self.assertIsNone(err)
        carry, err = taps.split_result((j.zeros(3), j.zeros(3)));self.assertIsNotNone(err, 'a 2-tuple whose second item is an array is (carry, err)')

    def test_exercised_lanes(self):
        state, c = synthetic();before = c.trackers;use = j.asarray([True, False, True, False, True])
        H.observe(c, state, use);after = c.trackers;written = taps.written_paths(before, after)
        hit = np.asarray(taps.exercised(before, after, written))
        self.assertTrue(hit[0] and hit[2] and hit[4], 'observe changes hp_percent/status/weather on the used lanes')
        self.assertFalse(hit[1] or hit[3])

    def test_statics_and_signature(self):
        state, c = synthetic();use = j.ones(5, bool)
        with taps.tap(['hp_line', 'feature', 'observe']) as calls:
            H.hp_line(c, state, 1, use, source_class=H.HP_POST_ITEM, record_damage=False)
        hp = next(c_ for c_ in calls if c_.name == 'hp_line')
        self.assertEqual(taps.statics(hp), {'side': 1, 'source_class': H.HP_POST_ITEM, 'record_damage': False})
        self.assertEqual(taps.signature(hp), 'hp_line(record_damage=False, source_class=2)')
        obs = next(c_ for c_ in calls if c_.name == 'observe' and c_.site == 'engine_device_public_hooks:254')
        self.assertEqual(taps.signature(obs), 'observe(record_damage=False, rest_mask=False, rest_side=None)')
        feats = {taps.signature(c_) for c_ in calls if c_.name == 'feature'}
        self.assertIn("feature(add=False, name='toxic_stage')", feats);self.assertIn("feature(add=False, name='sleep_turns')", feats)

    def test_signatures_report(self):
        diffs = [dict(results=[dict(module='hook:x@all', lane_steps=10, exercised_lane_steps_total=10,
            exercised_by_site={'x@m:1#0': 5, 'x@m:1#1': 0, 'x@m:2': 0, 'y@m:3': 7}, signatures={'x@m:1#0': 'x(a=1)', 'x@m:1#1': 'x(a=1)', 'x@m:2': 'x(a=2)', 'y@m:3': 'y()'})])]
        report = taps.signatures_report(diffs)
        self.assertEqual(report['signatures'], 3);self.assertEqual(report['signatures_never_exercised'], ['x(a=2)'])
        self.assertEqual((report['sites'], report['sites_never_exercised'], report['sites_under_100']), (4, 2, 2))
        self.assertIn('`x(a=2)`', taps.signatures_markdown(report))


class SelfTestTest(unittest.TestCase):
    def test_registry_and_selection(self):
        v2e_hooks.selftest(None)
        with self.assertRaises(KeyError):v2e_hooks.resolve('observe')   # normal mode: a missing hook is a hard error
        v2e_hooks.CUDA_HOOKS['observe'] = marker = lambda *a, **k: None
        try:
            self.assertIs(v2e_hooks.resolve('observe'), marker)
            self.assertTrue(v2e_hooks.implementation('observe')['registered'])
            with self.assertRaises(KeyError):v2e_hooks.resolve('hp_line')   # no fallback for the others
            with self.assertRaises(ValueError):v2e_hooks.selftest('oracle')   # refused while anything is registered
        finally:del v2e_hooks.CUDA_HOOKS['observe']
        with self.assertRaises(ValueError):v2e_hooks.selftest('bogus')
        previous = v2e_hooks.selftest('oracle-as-cuda', source='test')
        try:
            self.assertEqual(v2e_hooks.selftest_mode(), 'oracle')
            info = v2e_hooks.implementation('observe')
            self.assertEqual((info['selftest'], info['selftest_source'], info['registered']), ('oracle', 'test', False))
            self.assertIn('selftest_oracle:observe', info['implementation'])
            v2e_hooks.CUDA_HOOKS['observe'] = lambda *a, **k: None
            try:
                with self.assertRaises(ValueError):v2e_hooks.resolve('observe')   # a registration under an active self-test is refused too
            finally:del v2e_hooks.CUDA_HOOKS['observe']
        finally:v2e_hooks.selftest(previous)
        self.assertIsNone(v2e_hooks.selftest_mode())

    def test_flip_one_bit(self):
        for x in (j.arange(40, dtype=j.int32).reshape(5, 8), j.zeros((5, 2), bool), j.ones((5, 3), j.float32), j.zeros((2,), j.uint32)):
            flipped = v2e_hooks._flip_one_bit((x, j.zeros(5)), 0)
            a = np.asarray(unsigned(x)).reshape(x.shape[0], -1).astype(np.uint64);b = np.asarray(unsigned(flipped[0])).reshape(x.shape[0], -1).astype(np.uint64)
            lane = v2e_hooks.NEGATIVE_LANE if x.shape[0] > v2e_hooks.NEGATIVE_LANE else 0
            diff = a ^ b;self.assertEqual(int((diff != 0).sum()), 1);self.assertEqual(int(diff[lane, 0]), 1, x.dtype)
            self.assertEqual(flipped[0].dtype, x.dtype);self.assertIs(flipped[1], flipped[1])

    def test_oracle_as_cuda_reproduces_the_hook_and_negatives_differ(self):
        state, c = synthetic();before = c.trackers;use = j.ones(5, bool)
        tables = v2e_hooks.HookTables(None, None, None)
        want = H.observe(H.Context(before, None, None), state, use).trackers;written = taps.written_paths(before, want)
        got = v2e_hooks.selftest_hook('observe', 'oracle')(before, state, use, tables=tables)
        self.assertEqual(jax.tree.structure(got), jax.tree.structure(want))
        for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got)):self.assertTrue(bits_equal(a, b))
        paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(want)[0]]
        bad = v2e_hooks.selftest_hook('observe', 'negative')(before, state, use, tables=tables)
        differing = [p for p, a, b in zip(paths, jax.tree.leaves(want), jax.tree.leaves(bad)) if not bits_equal(a, b)]
        self.assertEqual(len(differing), 1);self.assertIn(differing[0], written, 'negative flips a written leaf')
        bad = v2e_hooks.selftest_hook('observe', 'negative-untouched')(before, state, use, tables=tables)
        differing = [p for p, a, b in zip(paths, jax.tree.leaves(want), jax.tree.leaves(bad)) if not bits_equal(a, b)]
        self.assertEqual(len(differing), 1);self.assertNotIn(differing[0], written, 'negative-untouched flips an untouched leaf')
        self.assertFalse(np.all(np.asarray(taps.site_outputs('o', before, bad, written, oracle=False)['o/untouched_equal'])))
        # a carry-first hook: _write(t, name, side, physical, value, use)
        want = H._write(before, 'shown_item', 0, j.zeros(5, j.int32), j.full(5, 9, j.int32), use)
        got = v2e_hooks.selftest_hook('_write', 'oracle')(before, 'shown_item', 0, j.zeros(5, j.int32), j.full(5, 9, j.int32), use, tables=tables)
        for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got)):self.assertTrue(bits_equal(a, b))


if __name__ == '__main__':unittest.main()
