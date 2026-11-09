"""CPU tests of the phase slicer (gpu_engine.v2e_phases), no real tables and no GPU: a toy function with known boundaries
(loop occurrences included) whose sliced chain equals the whole function bitwise; the incomplete-carry failure; alias
deduplication by var identity; recomputation of constant-derived values inside the phase (never carried); the lane
dependence of call outputs; the spec format (and the committed spec when present); chain-spec parsing; the boundary
table against transition.py and v2e_census; and the trace function's restoration."""
from pathlib import Path
import inspect
import json
import sys
import unittest

import jax
import jax.numpy as j
import numpy as np
from jax._src.core import jaxpr_as_fun

from gpu_engine import v2e_phases as V

LANES = 37
SCALE = np.asarray([1.5, 2.5, 4.0], np.float32)


class Box:
    """Not a pytree: an attribute of it is invisible to the taps."""


def toy(x, y, box):
    a = x * 2
    alias = y
    k = j.asarray(SCALE) * np.float32(2.0)
    rows = j.arange(x.shape[0])
    dead = x - 7
    c = a + alias
    for i in range(2):
        c = c + k[i] * x[:, :1]
    out = c / k[2] + rows[:, None].astype(j.float32)
    return dict(out=out, y=y)


def toy_hidden(x, y, box):
    box.hidden = x + 1
    del x
    a = y * 2
    return dict(out=a + box.hidden)


def line_of(fn, statement, occurrence=0):
    lines, first = inspect.getsourcelines(fn)
    found = [first + i for i, t in enumerate(lines) if t.strip().startswith(statement)]
    return found[occurrence]


def toy_boundaries():
    loop = line_of(toy, 'c = c + k[i] * x[:, :1]')
    return (V.Boundary('b0', None, 0, 'entry'), V.Boundary('b1', line_of(toy, 'c = a + alias'), 0, 'c = a + alias'),
        V.Boundary('b2', loop, 0, 'c = c + k[i]'), V.Boundary('b3', loop, 1, 'c = c + k[i]'), V.Boundary('b4', None, 0, 'return'))


def example():
    return (np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))


def toy_plan():
    return V.trace_plan(lambda x, y: toy(x, y, Box()), example(), toy_boundaries(), snapshot=V.local_snapshot, code=toy.__code__, lanes=LANES)


def bits(x):
    x = np.asarray(x)
    return x.view(np.uint32) if x.dtype == np.float32 else x


def run_chain(plan, values):
    """Evaluate every consecutive slice from the b0 carry; {boundary: {name: value}}."""
    carry = [values[e.name] for e in plan.carry[plan.order[0]]];seen = {plan.order[0]: dict(zip([e.name for e in plan.carry[plan.order[0]]], carry))}
    for start, end in zip(plan.order[:-1], plan.order[1:]):
        s = plan.slices[(start, end)]
        carry = list(jaxpr_as_fun(s.closed)(*carry))
        seen[end] = dict(zip([e.name for e in plan.carry[end]], carry))
    return seen


class SlicerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = toy_plan()

    def names(self, b):
        return [e.name for e in self.plan.carry[b]]

    def test_carries_are_the_live_lane_locals(self):
        self.assertEqual(self.names('b0'), ['x', 'y'])
        self.assertEqual(self.names('b1'), ['x', 'y', 'a'])          # dead, k, rows and alias are not carried
        self.assertEqual(self.names('b2'), ['x', 'y', 'c'])
        self.assertEqual(self.names('b3'), ['x', 'y', 'c'])
        self.assertEqual(self.names('b4'), ["['out']", "['y']"])
        self.assertTrue(all(e.shape == (3,) and e.dtype == 'float32' for b in self.plan.order for e in self.plan.carry[b]))
        self.assertEqual(self.plan.recomputed(), 0)

    def test_loop_occurrences_are_distinct_boundaries(self):
        c2 = next(e for e in self.plan.carry['b2'] if e.name == 'c');c3 = next(e for e in self.plan.carry['b3'] if e.name == 'c')
        self.assertIsNot(c2.atom, c3.atom)

    def test_sliced_chain_equals_the_whole_function_bitwise(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((LANES, 3)).astype(np.float32);y = rng.standard_normal((LANES, 3)).astype(np.float32)
        seen = run_chain(self.plan, {'x': j.asarray(x), 'y': j.asarray(y)})
        want = toy(j.asarray(x), j.asarray(y), Box())
        np.testing.assert_array_equal(bits(seen['b4']["['out']"]), bits(want['out']))
        np.testing.assert_array_equal(bits(seen['b4']["['y']"]), bits(want['y']))
        # The intermediate carries are the locals' values at the boundary.
        np.testing.assert_array_equal(bits(seen['b1']['a']), bits(j.asarray(x) * 2))

    def test_alias_deduplication_prefers_the_first_name(self):
        y = next(e for e in self.plan.carry['b1'] if e.name == 'y')
        self.assertEqual(y.aliases, ['alias'])
        self.assertNotIn('alias', self.names('b1'))

    def test_pass_through_outputs_alias_their_inputs(self):
        a = self.plan.slices[('b0', 'b1')]
        self.assertEqual(a.aliases, {0: 0, 1: 1})    # x and y pass through phase b0 -> b1 unchanged
        last = self.plan.slices[('b3', 'b4')]
        self.assertEqual(last.aliases, {1: 1})        # output y is input y

    def test_constant_derived_values_are_recomputed_not_carried(self):
        for b in self.plan.order:
            self.assertFalse({'k', 'rows'} & set(self.names(b)))
        last = self.plan.slices[('b3', 'b4')].closed
        prims = {e.primitive.name for e in last.jaxpr.eqns}
        self.assertIn('iota', prims)                    # rows, recomputed in the last phase
        consts = [np.asarray(c) for c in last.consts]
        self.assertTrue(any(c.shape == SCALE.shape and np.array_equal(c, SCALE) for c in consts), consts)
        loop = self.plan.slices[('b2', 'b3')].closed  # k[1] recomputed in the second loop phase too
        self.assertTrue(any(np.asarray(c).shape == SCALE.shape for c in loop.consts))

    def test_incomplete_carry_fails_loudly(self):
        bs = (V.Boundary('b0', None, 0, 'entry'), V.Boundary('b1', line_of(toy_hidden, 'a = y * 2'), 0, 'a = y * 2'), V.Boundary('b2', None, 0, 'return'))
        with self.assertRaises(V.IncompleteCarry) as caught:
            V.trace_plan(lambda x, y: toy_hidden(x, y, Box()), example(), bs, snapshot=V.local_snapshot, code=toy_hidden.__code__, lanes=LANES)
        self.assertIn('[0]', str(caught.exception))    # the step input x (example leaf 0) is outside C_b1

    def test_trace_function_is_restored(self):
        before = sys.gettrace()
        toy_plan()
        self.assertIs(sys.gettrace(), before)

    def test_unreached_boundary_fails(self):
        bs = (V.Boundary('b0', None, 0, 'entry'), V.Boundary('b1', line_of(toy, 'c = c + k[i]'), 2, 'c = c + k[i]'), V.Boundary('b2', None, 0, 'return'))
        with self.assertRaises(RuntimeError):
            V.trace_plan(lambda x, y: toy(x, y, Box()), example(), bs, snapshot=V.local_snapshot, code=toy.__code__, lanes=LANES)


class DependenceTest(unittest.TestCase):
    def test_constant_output_of_a_call_is_not_lane_dependent(self):
        @jax.jit
        def inner(x, c):
            return c * 2, x + c
        closed = jax.make_jaxpr(lambda x: inner(x, j.float32(3.0)))(np.zeros((LANES,), np.float32))
        dep = V.lane_dependence(closed.jaxpr)
        call = closed.jaxpr.eqns[-1]
        self.assertEqual([dep[v] for v in call.outvars], [False, True])


def mini_spec():
    carry = lambda *names: [dict(name=n, source='local', path='', dtype='int32', shape=[2]) for n in names]
    bs = [dict(name=b.name, line=b.line, occurrence=b.occurrence, statement=b.statement, carry=carry('a', 'b')) for b in V.BOUNDARIES]
    return dict(schema=V.SCHEMA, oracle_digest='x', lanes_traced=37, boundaries=bs, phases=[{'name': n, 'from': a, 'to': b} for n, a, b in V.PHASES])


class SpecFormatTest(unittest.TestCase):
    def test_valid_spec_passes(self):
        V.validate_spec(mini_spec())

    def test_invalid_specs_fail(self):
        def broken(edit):
            s = json.loads(json.dumps(mini_spec()));edit(s)
            with self.assertRaises(ValueError):V.validate_spec(s)
        broken(lambda s: s.update(schema='other'))
        broken(lambda s: s['boundaries'][3]['carry'].append(dict(s['boundaries'][3]['carry'][0])))   # duplicate name
        broken(lambda s: s['boundaries'][1]['carry'][0].update(dtype='int64'))
        broken(lambda s: s['boundaries'][1]['carry'][0].update(source='nowhere'))
        broken(lambda s: s['boundaries'][1]['carry'][0].pop('path'))
        broken(lambda s: s['boundaries'][2]['carry'][0].update(shape=[0]))
        broken(lambda s: s['phases'].pop())
        broken(lambda s: s['boundaries'].pop(4))
        with self.assertRaises(ValueError):V.validate_spec(mini_spec(), output_leaves=148)

    def test_entry_json_has_the_contract_keys(self):
        e = V.Entry('work.hp', 'local', '.hp', 'int32', (2, 6), aliases=['state.hp'])
        self.assertEqual(e.json(), dict(name='work.hp', source='local', path='.hp', dtype='int32', shape=[2, 6], aliases=['state.hp']))

    def test_committed_spec_when_present(self):
        if not V.SPEC.exists():self.skipTest('gpu_engine/v2e_phases_spec.json not committed yet')
        spec = V.load_spec()
        self.assertEqual(len(spec['boundaries'][-1]['carry']), 148)
        b0 = {e['name'] for e in spec['boundaries'][0]['carry']}
        self.assertTrue({'state.hp', 'actions'} <= b0)
        self.assertTrue(any(n.startswith('trackers.') for n in b0) and any(n.startswith('chance_inputs.') for n in b0))
        for b in spec['boundaries']:
            for e in b['carry']:self.assertTrue(e['name'].startswith(('state', 'actions', 'trackers', 'chance_inputs')) or e['source'] == 'local', e)


class ChainSpecTest(unittest.TestCase):
    hand = {'FG': V.HandPhase('FG', 'b6', 'b8', None)}

    def test_all_generated(self):
        units = V.parse_chain('gen-A-G', self.hand)
        self.assertEqual([u.name for u in units], [p[0] for p in V.PHASES])

    def test_mixed_chain(self):
        units = V.parse_chain('gen-A-E+hand-FG', self.hand)
        self.assertEqual([(u.kind, u.name) for u in units][-2:], [('gen', 'E_order1'), ('hand', 'FG')])
        self.assertEqual([u.name for u in V.parse_chain('gen-A-E+gen-FG', self.hand)][-1], 'FG')
        self.assertEqual(len(V.parse_chain('gen-A+gen-B-D+gen-E-G', self.hand)), 8)

    def test_bad_chains(self):
        for spec in ('gen-A-D', 'gen-FG', 'gen-A-E', 'gen-Z', 'gen-G-A', 'oops-A-G', 'gen-A-E+gen-F'):
            with self.assertRaises(ValueError, msg=spec):V.parse_chain(spec, self.hand)
        with self.assertRaises(KeyError):V.parse_chain('gen-A-E+hand-XX', self.hand)

    def test_phase_arguments(self):
        self.assertEqual(V._phase_range('E_order1'), ('b5', 'b6', 'E_order1'))
        self.assertEqual(V._phase_range('FG'), ('b6', 'b8', 'FG'))
        self.assertEqual(V._phase_range('b4..b4.865'), ('b4', 'b4.865', 'b4..b4.865'))
        with self.assertRaises(KeyError):V._phase_range('Q')


class BoundaryTableTest(unittest.TestCase):
    def test_boundaries_match_transition_and_census(self):
        info = V.check_boundaries(V.boundaries(sub=True))
        self.assertEqual(info['b4']['line'], 689)
        self.assertGreater(info['b4.1582']['event_line'], 1582)   # a comment line taps the next line with code

    def test_a_moved_statement_fails(self):
        with self.assertRaises(V.BoundaryMoved):V.check_boundaries((V.Boundary('b1', 512, 0, 'speed = _speeds(work,table)', 'B'),))

    def test_boundary_names_are_ordered(self):
        names = [b.name for b in V.boundaries(sub=True)]
        self.assertEqual([n for n in names if '.' not in n], [b.name for b in V.BOUNDARIES])
        self.assertLess(names.index('b4.1582'), names.index('b5'));self.assertLess(names.index('b6.1804.9'), names.index('b7'))

    def test_boundary_observer_reads_both_key_quotings(self):
        self.assertEqual(V._boundary_of("['b3/work.hp']"), 'b3')
        self.assertEqual(V._boundary_of('["b0/trackers.mon_features[\'x\']"]'), 'b0')


if __name__ == '__main__':
    unittest.main()
