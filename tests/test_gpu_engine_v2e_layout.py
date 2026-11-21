"""CPU tests of the generated CUDA layout header (gpu_engine.v2e_layout): deterministic generation, digests that move
with any leaf change, the mini spec and M0a's real spec, the committed header's freshness, the numpy emulation of JAX's
gather/scatter semantics, and the header's accessors compiled with a host C++ compiler against that emulation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gpu_engine import v2e_layout as L

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests' / 'fixtures'
MINI_SPEC = FIXTURES / 'v2e_phases_spec_mini.json'
# numpy results of jnp indexing on GPU (jax 0.12.0), the reference the emulation and the accessors must match.
JAX_GATHER = {(-7,): 0, (-1,): 5, (6,): 5, (100,): 5, (-14,): 0, (-6,): 0, (5,): 5}
JAX_SCATTER_KEPT = {-7: False, -6: True, -1: True, 5: True, 6: False, 100: False}


def description():
    return L.load_description(L.DEFAULT_DESCRIPTION)


def tiny_description():
    return dict(schema=L.SCHEMA,
        state=[dict(path='.hp', dtype='int32', shape=[2, 6]), dict(path='.done', dtype='bool', shape=[]), dict(path='.keys', dtype='uint32', shape=[2])],
        actions=[dict(path='', dtype='int32', shape=[2])],
        trackers=[dict(path=".mon_features['toxic_stage']", dtype='int32', shape=[2, 6]), dict(path='.tendency.matchup', dtype='int32', shape=[2, 6, 6, 2]),
            dict(path='.damage_history.taken', dtype='float32', shape=[2, 2, 2]), dict(path='.public_volatiles', dtype='bool', shape=[2, 6, 40])],
        step_output=[dict(path='.step.state.hp', dtype='int32', shape=[2, 6])],
        tables=[dict(path='.power', dtype='int32', shape=[7]), dict(path='.struggle', dtype='int32', shape=[]), dict(path='.effects.self_boosts', dtype='int32', shape=[7, 7]),
            dict(path='belief.spreads', dtype='uint16', shape=[3, 4, 5, 6]), dict(path=L.ACCURACY_PATH, dtype='int32', shape=[6, 101, 13, 13])])


class GenerationTest(unittest.TestCase):
    def test_generation_is_deterministic(self):
        d = description();spec = L.load_spec(MINI_SPEC)
        a = L.generate(d, spec);b = L.generate(json.loads(json.dumps(d)), L.load_spec(MINI_SPEC))
        self.assertEqual(a, b)
        self.assertIn('struct TrackerView', a);self.assertIn('struct C_b1', a);self.assertIn('struct EngineTables', a)

    def test_real_leaf_counts(self):
        d = description()
        self.assertEqual((len(d['state']), len(d['actions']), len(d['trackers']), len(d['step_output'])), (60, 1, 86, 148))
        self.assertEqual(len(d['tables']), 146 + 31 + 1)
        self.assertEqual(d['tables'][-1]['path'], L.ACCURACY_PATH);self.assertEqual(d['tables'][-1]['shape'], [6, 101, 13, 13])

    def test_committed_header_is_fresh(self):
        """The committed gpu_engine/cuda/v2e_layout.cuh equals a regeneration from the committed description and spec."""
        spec = L.load_spec(L.DEFAULT_SPEC) if L.DEFAULT_SPEC.exists() else None
        self.assertEqual(L.HEADER.read_text(), L.generate(description(), spec))

    def test_struct_members_follow_flatten_order(self):
        text = L.generate(tiny_description(), L.load_spec(MINI_SPEC))
        self.assertIn('Leaf<int32_t, 2, 6> hp;   // arg 0: .hp int32[2, 6]', text)
        self.assertIn('Leaf<bool> done;   // arg 1: .done bool[]', text)
        self.assertIn('static constexpr int arg_keys = 2;', text)
        self.assertIn('Leaf<int32_t, 2, 6, 6, 2> tendency_matchup;', text)
        self.assertIn('static constexpr int kStepOffset = 4;   // first operand', text)   # trackers follow 3 state leaves and 1 action
        self.assertIn('Leaf<int32_t, 2> speed;   // arg 2: speed int32[2] (source local', text)   # mini spec b1 carry order
        self.assertIn('constexpr int kStepLeaves = 8;', text)

    def test_table_offsets_are_aligned_and_packed_in_order(self):
        d = tiny_description();rows, total = L.table_layout(d['tables'])
        self.assertEqual([o for _, o, _ in rows], [0, 32, 48, 256, 256 + (3 * 4 * 5 * 6 * 2 + 15) // 16 * 16])
        self.assertEqual(total % L.TABLE_ALIGN, 0)
        text = L.generate(d)
        self.assertIn('static constexpr int64_t off_effects_self_boosts = 48;', text)
        self.assertIn(f'static constexpr int64_t kBytes = {total};', text)
        self.assertIn('Table<int32_t, 6, 101, 13, 13> mechanics__ACCURACY_COUNTS() const', text)

    def test_identifiers(self):
        self.assertEqual(L.identifier(".mon_features['turns_active']"), 'mon_features_turns_active')
        self.assertEqual(L.identifier('.tendency.matchup'), 'tendency_matchup')
        self.assertEqual(L.identifier('state.hp'), 'state_hp');self.assertEqual(L.identifier(''), 'leaf')
        self.assertEqual(L.identifier('[0].hp'), '_0_hp');self.assertEqual(L.identifier('.default'), 'default_')
        taken = set();self.assertEqual([L.identifier('.a.b', taken), L.identifier('.a_b', taken)], ['a_b', 'a_b_2'])

    def test_unsupported_dtype_is_refused(self):
        d = tiny_description();d['state'][0]['dtype'] = 'float64'
        with self.assertRaises(ValueError):L.generate(d)


class DigestTest(unittest.TestCase):
    def test_digest_changes_with_any_leaf_change(self):
        d = tiny_description();base = L.digests(d)
        for key, change in (('shape', [2, 7]), ('dtype', 'int16'), ('path', '.hp2')):
            e = copy.deepcopy(d);e['state'][0][key] = change
            self.assertNotEqual(L.digests(e)['StateView'], base['StateView'], key)
            self.assertEqual(L.digests(e)['TrackerView'], base['TrackerView'], 'other structs keep their digest')
        e = copy.deepcopy(d);e['trackers'].append(dict(path='.added', dtype='int32', shape=[]))
        self.assertNotEqual(L.digests(e)['TrackerView'], base['TrackerView'])
        e = copy.deepcopy(d);e['trackers'] = e['trackers'][::-1]
        self.assertNotEqual(L.digests(e)['TrackerView'], base['TrackerView'], 'order is part of the digest')

    def test_header_digest_constants_match_python(self):
        d = tiny_description();spec = L.load_spec(MINI_SPEC);text = L.generate(d, spec)
        for name, (d64, hexdigest) in L.digests(d, spec).items():
            self.assertIn(f'static constexpr uint64_t kDigest = 0x{d64:016x}ull;', text, name)
            self.assertIn(f'kDigestHex = "{hexdigest}"', text, name)
            self.assertEqual(int(hexdigest[:16], 16), d64)
        self.assertIn(f'kStaleLayout = 0x{L.STALE_LAYOUT:08x}u', text)

    def test_carry_digests_follow_the_spec(self):
        d = tiny_description();spec = L.load_spec(MINI_SPEC);base = L.digests(d, spec)
        other = copy.deepcopy(spec);other['boundaries'][1]['carry'][2]['shape'] = [3]
        self.assertNotEqual(L.digests(d, other)['C_b1'], base['C_b1']);self.assertEqual(L.digests(d, other)['C_b0'], base['C_b0'])


class SpecTest(unittest.TestCase):
    def test_mini_spec_loads_and_duplicates_are_refused(self):
        spec = L.load_spec(MINI_SPEC)
        self.assertEqual([b['name'] for b in spec['boundaries']], ['b0', 'b1', 'b2'])
        bad = copy.deepcopy(spec);bad['boundaries'][0]['carry'].append(dict(bad['boundaries'][0]['carry'][0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'spec.json';path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):L.load_spec(path)

    def test_real_spec_when_present(self):
        """M0a's committed spec, once present: 9 boundaries b0..b8, b8 carrying every step output leaf, every entry a Leaf."""
        if not L.DEFAULT_SPEC.exists():self.skipTest('gpu_engine/v2e_phases_spec.json not committed yet')
        spec = L.load_spec(L.DEFAULT_SPEC);d = description()
        self.assertEqual([b['name'] for b in spec['boundaries']], [f'b{k}' for k in range(9)])
        self.assertEqual(len(spec['boundaries'][-1]['carry']), len(d['step_output']))
        text = L.generate(d, spec)
        for b in spec['boundaries']:self.assertIn(f"struct C_{b['name']} {{", text)


class EmulationTest(unittest.TestCase):
    def test_gather_matches_jax(self):
        x = np.arange(4 * 6, dtype=np.int32).reshape(4, 6)
        for (i,), col in JAX_GATHER.items():
            got = L.emulate_get(x, np.full(4, i, np.int32))
            np.testing.assert_array_equal(got, x[np.arange(4), col])

    def test_scatter_drops_and_converts_like_jax(self):
        x = np.arange(4 * 6, dtype=np.int32).reshape(4, 6)
        for i, kept in JAX_SCATTER_KEPT.items():
            out = L.emulate_set(x, [np.full(4, i, np.int32)], np.int32(-99))
            self.assertEqual(bool((out != x).any()), kept, i)
        # float updates into int32: toward zero, saturating, NaN -> 0 (jnp .at[].set on GPU: [2, -2, 2147483647, 0])
        out = L.emulate_set(x, [np.arange(4, dtype=np.int32)], np.array([2.7, -2.7, 1e10, np.nan], np.float32))
        np.testing.assert_array_equal(out[np.arange(4), np.arange(4)], [2, -2, 2147483647, 0])
        np.testing.assert_array_equal(L.convert(np.array([0, 2, -1, 5], np.int32), np.bool_), [False, True, True, True])
        np.testing.assert_array_equal(L.convert(np.array([256, -1, 300, 7], np.int32), np.uint8), [0, 255, 44, 7])
        np.testing.assert_array_equal(L.convert(np.array([16777217, -1], np.int32), np.float32), np.array([16777216., -1.], np.float32))
        np.testing.assert_array_equal(L.convert(np.array([2**31, 2**32 - 1], np.uint32), np.int32), [-2147483648, -1])
        np.testing.assert_array_equal(L.convert(np.array([40000., -40000., np.nan, 2.9, -2.9], np.float32), np.int16), [32767, -32768, 0, 2, -2])
        np.testing.assert_array_equal(L.convert(np.array([300., -5., np.nan, 2.9, 255.9], np.float32), np.uint8), [255, 0, 0, 2, 255])
        np.testing.assert_array_equal(L.convert(np.array([5e9, -5., np.nan, 4294967296.0, 2.9], np.float32), np.uint32), [4294967295, 0, 0, 4294967295, 2])
        np.testing.assert_array_equal(L.convert(np.array([0., -0., np.nan, 2.9], np.float32), np.bool_), [False, False, True, True])

    def test_promotion_round_trips_the_whole_operand_like_jax(self):
        """jax: x.at[].set(float) on an int32 operand promotes x to float32, so untouched large values change; the point
        semantics of Leaf::set converts the update alone. The oracle has no promoting scatter (scatter-census)."""
        self.assertTrue(L.promotes('int32', 'float32'));self.assertTrue(L.promotes('uint32', 'float32'))
        for a, b in (('int32', 'int32'), ('bool', 'int32'), ('uint8', 'int32'), ('int32', 'uint32'), ('float32', 'int32'), ('int32', 'bool')):
            self.assertEqual(L.promotes(a, b), np.dtype(L.promoted_dtype(a, b)) != np.dtype(a), (a, b))
        self.assertFalse(L.promotes('float32', 'int32'));self.assertFalse(L.promotes('int32', 'int32'))
        x = np.array([[2**24 + 1, 5], [7, -(2**24 + 3)]], np.int32);v = np.array([1.5, 2.5], np.float32)
        exact = L.emulate_set(x, [np.array([1, 0])], v);point = L.emulate_set_point(x, [np.array([1, 0])], v)
        np.testing.assert_array_equal(point, [[2**24 + 1, 1], [2, -(2**24 + 3)]])
        np.testing.assert_array_equal(exact, [[2**24, 1], [2, -(2**24 + 4)]])   # the untouched elements rounded through float32
        np.testing.assert_array_equal(L.emulate_set(x, [np.array([1, 0])], np.array([9, 8], np.int32)), [[2**24 + 1, 9], [8, -(2**24 + 3)]])

    def test_multi_index_scatter_drops_the_whole_write(self):
        x = np.zeros((3, 2, 6), np.int32)
        out = L.emulate_set(x, [np.array([0, 1, 5]), np.array([5, 6, 0])], np.int32(1))   # lane 1 side in range, slot 6 out; lane 2 side 5 out
        self.assertEqual(int(out.sum()), 1);self.assertEqual(int(out[0, 0, 5]), 1)


class HostAccessorTest(unittest.TestCase):
    """The header's Leaf accessors, compiled as plain C++ (-x c++), against the numpy emulation for every dtype in use."""
    def test_accessors_match_the_emulation(self):
        if L.cxx_compiler() is None:self.skipTest('no C++ compiler')
        d = description();spec = L.load_spec(MINI_SPEC);text = L.generate(d, spec)
        dtypes = sorted({e['dtype'] for key in ('state', 'actions', 'trackers') for e in d[key]} | {'int16', 'uint8', 'uint16'})
        cases = L.probe_cases(dtypes)
        with tempfile.TemporaryDirectory() as tmp:
            rows = L.host_probe(text, cases, 48, work=tmp)
        self.assertEqual(len(rows), len(cases));mismatches = [];seen_clamp = seen_drop = 0
        for row in rows:
            rank = len(row['dims']);idx = [row['idx'][:, k] for k in range(rank)]
            got = L.emulate_get(row['x'], *idx);out = L.emulate_set_point(row['x'], idx, row['v']);fill = L.emulate_get_fill(row['x'], *idx)
            kept = np.ones(row['x'].shape[0], bool)
            for i, n in zip(idx, row['dims']):kept &= L.scatter_index(i, n)[1]
            ok = (L._bits_equal(got, row['got']) and L._bits_equal(out, row['out']) and L._bits_equal(fill, row['fill']) and np.array_equal(kept, row['kept'])
                and row['drops'] == int((~kept).sum()))
            if not ok:mismatches.append(row['case'])
            seen_clamp += row['clamps'];seen_drop += row['drops']
        self.assertEqual(mismatches, []);self.assertGreater(seen_clamp, 0);self.assertGreater(seen_drop, 0)


class ListingTest(unittest.TestCase):
    """The device-path pruner (gpu_engine.v2e_listing): static facts on a stand-in, and on the real step."""
    def test_stand_in_pruning(self):
        from gpu_engine import v2e_listing as LS
        code = ['from . import helper as h', 'def f(x, table, _device_public=None, _public_trace=None, _lifecycle_trace=False, _toxic_context=None):',
            '    y = x + 1', '    if _lifecycle_trace:', '        y = y * 2', '        y = y * 3', '    elif _public_trace is not None:', '        y = y - 1',
            '    else:', '        y = y + 2', '    if _device_public is not None and table.entry is not None:', '        y = h.g(y)',
            '    if _toxic_context is not None:', '        y = 0', '    if x > 0:', '        y = y + 1', '    else:', '        y = y - 1',
            '    carry = table.items', '    if carry is None:', '        return y', '    return y', '', 'def g(v):', '    return v']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stand_in.py';path.write_text('\n'.join(code) + '\n')
            module = LS.Module('stand_in', path)
        pruned = LS.prune_function(module, module.functions['f'], LS.Facts({'table': {'entry', 'items'}, 'belief': set()}))
        self.assertEqual(sorted(pruned.lines), [2, 3, 10, 11, 12, 15, 16, 17, 18, 19, 22])
        self.assertEqual(sorted(pruned.static_headers), [11]);self.assertEqual(sorted(pruned.unknown_headers), [15])
        self.assertEqual(LS.reached_calls(module, pruned), {('helper', 'g')})
        text = LS.render(module, pruned, {'stand_in': {2, 3, 10}})
        self.assertIn('    2: def f(', text);self.assertIn('   11:~    if _device_public', text, 'a line without equations is marked ~')
        code2 = ['def g(table, x):', '    if table.items is not None:', '        return x', '    return x + 1', '', 'def h(table, x, original=True):',
            '    y = x if table.items is not None else 0', '    return y if original else 1']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stand_in2.py';path.write_text('\n'.join(code2) + '\n');module2 = LS.Module('stand_in2', path)
        facts = LS.Facts({'table': {'items'}, 'belief': set()})
        g = LS.prune_function(module2, module2.functions['g'], facts)
        self.assertEqual(sorted(g.lines), [1, 2, 3], 'the statement after a statically taken return is dead');self.assertEqual(sorted(g.dropped), [4])
        h = LS.prune_function(module2, module2.functions['h'], LS.Facts({'table': {'items'}, 'belief': set()}, {'original': True}))
        self.assertIn('else-branch', h.notes[7]);self.assertIn('else-branch', h.notes[8])

    def test_call_site_facts_need_a_static_at_every_call(self):
        from gpu_engine import v2e_listing as LS
        # observe: rest_side is None at most sites but an array (omitted from statics) at transition:1196 -> no fact;
        # record_damage is a static at every call with one value -> a fact; reveal_item's original is a fact.
        sites = dict(sites={'observe': {'transition:756': dict(statics=[{'record_damage': True, 'rest_side': None, 'rest_mask': False}]),
            'transition:1196': dict(statics=[{'record_damage': True}])}, 'reveal_item': {'transition:1872': dict(statics=[{'side': 0, 'original': True}, {'side': 1, 'original': True}])},
            'wish': {'transition:1499': dict(statics=[{'side': 0, 'start': True, 'resolved': False}]), 'transition:1741': dict(statics=[{'side': 0, 'start': False}])}})
        facts = LS.call_site_facts(sites)
        self.assertEqual(facts[('engine_device_public_hooks', 'observe')], {'record_damage': True})
        self.assertEqual(facts[('engine_device_public_hooks', 'reveal_item')], {'original': True})
        self.assertNotIn('resolved', facts[('engine_device_public_hooks', 'wish')]);self.assertNotIn('start', facts[('engine_device_public_hooks', 'wish')])

    def test_real_step_pruning(self):
        from gpu_engine import v2e_listing as LS
        modules = LS.load_modules();listed = LS.listing(modules, LS.present_attributes())
        step = listed[LS.STEP_FUNCTION];text = modules['transition'].lines
        self.assertIn(1124, step.static_headers, 'if table.effects is not None: is a statically true guard')
        for n in step.lines:self.assertNotIn('_public_trace=', text[n - 1].replace(' ', '').split('#')[0][:40], n)
        dead = [n for n in range(step.first, step.last + 1) if text[n - 1].strip().startswith('if _lifecycle_trace')]
        self.assertTrue(dead);self.assertFalse(set(dead) & step.lines)
        self.assertIn(('engine_device_public_hooks', 'observe'), listed);self.assertIn(('pursuit', 'intercept'), listed)
        self.assertIn(('engine_device_trackers', '_gated'), listed, 'functions passed as callbacks are reached')
        self.assertLess(len(step.lines), 800);self.assertGreater(len(step.lines) - len(step.static_headers), 550)


class CommandTest(unittest.TestCase):
    """The one-command regeneration M0a runs once its spec exists, exercised with the mini spec:
    python -m gpu_engine.v2e_layout generate --spec gpu_engine/v2e_phases_spec.json (defaults: the committed description and header path)."""
    def test_generate_and_check_with_a_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'v2e_layout.cuh'
            self.assertEqual(L.main(['generate', '--spec', str(MINI_SPEC), '--out', str(out)]), 0)
            text = out.read_text()
            self.assertEqual(text, L.generate(description(), L.load_spec(MINI_SPEC)))
            for name in ('C_b0', 'C_b1', 'C_b2'):self.assertIn(f'struct {name} {{', text)
            self.assertIn('// phases: A b0->b1, BC b1->b2', text)
            self.assertEqual(L.main(['check', '--spec', str(MINI_SPEC), '--header', str(out)]), 0)
            self.assertEqual(L.main(['check', '--spec', str(Path(tmp) / 'absent.json'), '--header', str(out)]), 1, 'a header with carries is stale without the spec')
            self.assertEqual(L.main(['check', '--header', str(L.HEADER), '--spec', str(MINI_SPEC)]), 1, 'the committed header has no carries yet')


class FillAndDigestTest(unittest.TestCase):
    def test_fill_gather_emulation(self):
        """jnp.take_along_axis / .at[].get(mode='fill'): wrap once, then the fill value (jax defaults per dtype)."""
        x = np.arange(4 * 6, dtype=np.int32).reshape(4, 6)
        np.testing.assert_array_equal(L.emulate_get_fill(x, np.array([-7, -1, 6, 100])), [np.iinfo(np.int32).min, 11, np.iinfo(np.int32).min, np.iinfo(np.int32).min])
        np.testing.assert_array_equal(L.emulate_get_fill(x, np.array([-6, 0, 5, 3])), [0, 6, 17, 21])
        self.assertEqual(L.fill_value('uint32'), 2**32 - 1);self.assertEqual(L.fill_value('bool'), True);self.assertTrue(np.isnan(L.fill_value('float32')))
        self.assertEqual(L.fill_value('int16'), -32768);self.assertEqual(L.fill_value('uint8'), 255)
        f = L.emulate_get_fill(np.zeros((2, 3), np.float32), np.array([5, 1]));self.assertTrue(np.isnan(f[0]) and f[1] == 0)

    def test_header_wide_digest_and_attribute(self):
        d = tiny_description();spec = L.load_spec(MINI_SPEC);text = L.generate(d, spec)
        d64, hexdigest = L.layout_digest(d, spec)
        self.assertIn(f'constexpr uint64_t kLayoutDigest = 0x{d64:016x}ull;', text);self.assertIn(f'kLayoutDigestHex = "{hexdigest}"', text)
        e = copy.deepcopy(d);e['tables'][0]['shape'] = [8]
        self.assertNotEqual(L.layout_digest(e, spec)[0], d64, 'any struct change moves the header-wide digest')
        self.assertNotEqual(L.layout_digest(d)[0], d64, 'the carries are part of it')
        for v in (0, 2**63 - 1, 2**63, 2**64 - 1, d64):
            self.assertEqual(int(np.int64(L.digest_attr(v)).view(np.uint64)), v)

    def test_live_description_matches_describe_shapes(self):
        """live_description over arrays built from the fixture reproduces the fixture's tables entries exactly."""
        d = description()
        import jax
        from gpu_engine.engine_device_trackers import Trackers
        state = tuple(np.zeros([1] + e['shape'], e['dtype']) for e in d['state'])   # a tuple pytree: paths differ, dtypes/shapes agree
        live = L.live_description(state, np.zeros((1, 2), np.int32), Trackers(*[None] * len(Trackers._fields)), tuple(), None)
        self.assertEqual([(e['dtype'], e['shape']) for e in live['state']], [(e['dtype'], e['shape']) for e in d['state']])
        self.assertEqual(live['tables'][-1], d['tables'][-1])
        diffs = L.check_live(d, state, np.zeros((1, 2), np.int32), Trackers(*[None] * len(Trackers._fields)), tuple(), None)
        self.assertTrue(any(s.startswith('state: leaf order') for s in diffs), 'the tuple pytree has other paths: reported')


class HostSpanAndStaticAssertTest(unittest.TestCase):
    def _compile(self, body, expect_ok=True):
        import subprocess
        compiler = L.cxx_compiler()
        if compiler is None:self.skipTest('no C++ compiler')
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'v2e_layout.cuh').write_text(L.generate(description()))
            src = Path(tmp) / 'main.cpp';src.write_text('#define V2E_LAYOUT_DEBUG 1\n#include "v2e_layout.cuh"\n#include <cstdio>\nusing namespace v2e_layout;\n' + body)
            result = subprocess.run([compiler, '-std=c++17', '-x', 'c++', '-o', str(Path(tmp) / 'a.out'), str(src)], capture_output=True, text=True)
            if not expect_ok:return result.returncode, result.stderr
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            run = subprocess.run([str(Path(tmp) / 'a.out')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr);return run.stdout

    def test_span_and_writable(self):
        out = self._compile(r'''
int main() {
  int32_t data[3 * 2 * 6]; for (int k = 0; k < 36; ++k) data[k] = k;
  uint32_t dbg[2] = {0u, 0u};
  Leaf<int32_t, 2, 6> leaf; leaf.p = data; leaf.lanes = 3; leaf.dbg = dbg;
  // span(b) is the lane; span(b, i0) clamps i0 like a gather (-1 -> 1, 7 -> 1, -3 -> 0)
  std::printf("%ld %ld %ld %ld %ld\n", (long)(leaf.span(1) - data), (long)(leaf.span(1, -1) - data), (long)(leaf.span(1, 7) - data), (long)(leaf.span(1, -3) - data), (long)(leaf.span(2, 0) - data));
  // writable follows the scatter rule: wrap once, in range or dropped
  std::printf("%d %d %d %d %d\n", (int)leaf.writable(0), (int)leaf.writable(0, -1), (int)leaf.writable(0, -3), (int)leaf.writable(0, 2), (int)leaf.writable(0, 1));
  // get_fill at a lane: in range reads, out of range fills INT_MIN; clamps counted
  // sequence the reads: argument evaluation order is unspecified, and get_fill updates dbg
  const int32_t wrapped = leaf.get_fill(0, 1, -1); const bool filled = leaf.get_fill(0, 1, 6) == INT32_MIN; const uint32_t clamps = dbg[0];
  std::printf("%d %d %u\n", wrapped, (int)filled, clamps);
  return 0;
}
''')
        self.assertEqual(out.splitlines(), ['12 18 18 12 24', '1 1 0 0 1', '11 1 3'])

    def test_set_refuses_unsupported_update_types(self):
        ok = self._compile('int main() { int32_t d[6] = {}; Leaf<int32_t, 6> l; l.p = d; l.lanes = 1; l.set(0, 1, 2.5f); l.set(0, 2, true); l.set(0, 3, uint8_t(7)); return d[1] != 2; }')
        self.assertEqual(ok, '')
        rc, err = self._compile('int main() { int32_t d[6] = {}; Leaf<int32_t, 6> l; l.p = d; l.lanes = 1; l.set(0, 1, 2.5); return 0; }', expect_ok=False)
        self.assertNotEqual(rc, 0);self.assertIn('scatter update type', err)
        rc, err = self._compile('int main() { int32_t d[6] = {}; Leaf<int32_t, 6> l; l.p = d; l.lanes = 1; l.set(0, 1, int64_t(3)); return 0; }', expect_ok=False)
        self.assertNotEqual(rc, 0)


class CoverageTest(unittest.TestCase):
    """gpu_engine.v2e_coverage pieces on synthetic data (no tables, no trace)."""
    def test_paths_summary_and_markdown(self):
        from gpu_engine import v2e_coverage as cov, v2e_census as cz
        self.assertEqual(cov.path_of('hook:observe@transition:756#1'), 'hook:observe@transition:756');self.assertEqual(cov.path_of('step:defer'), 'step:defer')
        counts = {g: {} for g in cz.GROUPS};counts['E'] = {'hook:observe@transition:756#0': 500, 'hook:observe@transition:756#1': 700, 'effect:.x@caller': 3, 'input:kind:done': 10}
        groups = cov.summarize(counts, 1000, {}, rare=1000)
        self.assertEqual(groups['E']['paths']['hook:observe@transition:756'], dict(occurrences=1200, per_lane_step=1.2))
        self.assertEqual(groups['E']['rare'], ['input:kind:done', 'effect:.x@caller']);self.assertEqual(len(groups['E']['features']), 4)
        report = dict(shards=1, lane_steps=1000, rare=1000, census=None, weights='equal weights (unweighted mean of the groups)', design_uniform_share=0.7,
            groups={g: dict(features={}, paths={}, rare=[], union=None) for g in cz.GROUPS}, distinct_paths_factor_across_groups=None, design_factor_across_groups=None, census_factor_across_groups=None)
        report['groups']['E'] = groups['E'];report['groups']['E']['union'] = dict(distinct_paths_warp_over_lane=2.5, union_factor_design_uniform=1.07, union_factor=1.1, distinct_classes_per_warp=dict(mean=3.0, p90=5))
        md = cov.markdown(report)
        self.assertIn('| E | 4 | 3 | 2 | 2.500 | 1.070 | None | 3.0 / 5 |', md);self.assertIn('no --census: no census-weighted figure', md)
        self.assertIn('| `hook:observe@transition:756` | 1,200 | 120.00% |', md)

    def test_lane_mask_and_group_of_line(self):
        import jax.numpy as j
        from gpu_engine import v2e_coverage as cov
        np.testing.assert_array_equal(np.asarray(cov.lane_mask(True, 3)), [True, True, True])
        np.testing.assert_array_equal(np.asarray(cov.lane_mask(j.asarray([[True, False], [False, False], [True, True]]), 3)), [True, False, True])
        self.assertIsNone(cov.lane_mask(j.zeros(3, j.int32), 3));self.assertIsNone(cov.lane_mask(j.zeros((4,), bool), 3))
        self.assertEqual((cov.group_of_line(511), cov.group_of_line(1700), cov.group_of_line(2092), cov.group_of_line(153), cov.group_of_line(None)), ('B', 'F', 'G', None, None))


if __name__ == '__main__':unittest.main()
