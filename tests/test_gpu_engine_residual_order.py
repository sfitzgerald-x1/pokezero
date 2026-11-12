"""R1 (Fidelity M2's compile-size restructure): the residual event positions that gate the Wrap and Yawn blocks.

transition.step_with_chance emits M2's partiallytrapped and Yawn handler blocks only where those events can sort,
residual_order.POSITIONS (Wrap 4, 8, 9, 11; Yawn 6, 12, 13). That is exact only under two invariants, checked here on
transition.py's own source:
(a) a position is a rank among all 14 keys whether or not its event is active: no sentinel key for an inactive event, and
    the loop visits every position (no compaction);
(b) Speed and the tie are fixed before the loop, and nothing inside the loop reorders.
The reachable position x kind table is recomputed by brute force from the step's own constants and key function (imported,
never copied) over faster, slower and equal Speeds and every tie bit, and the gating sets must equal it. A changed key
formula, multiplier, subOrder or event list fails here instead of silently dropping events. Pure Python: no JAX import.
"""
import ast
import itertools
import os
from pathlib import Path
import subprocess
import sys
import unittest

from gpu_engine import residual_order as ro

ROOT = Path(__file__).resolve().parents[1]
WRAP, YAWN = {4, 8, 9, 11}, {6, 12, 13}
# Faster, slower and equal: every ordered pair of these end_speed values, small and large gaps included.
SPEEDS = (0, 1, 2, 31, 63, 64, 65, 200, 201, 2000)


def brute_force():
    """({kind: positions}, number of orders, all keys distinct): each event's position is its rank among ALL the keys."""
    table = {kind: set() for kind in ro.KINDS};distinct = True;cases = 0
    pairs = sorted(set(ro.TIE_PAIR))
    for speed, bits in itertools.product(itertools.product(SPEEDS, repeat=2), itertools.product((0, 1), repeat=len(pairs))):
        draw = dict(zip(pairs, bits))
        keys = [ro.event_key(speed[ro.EVENT_SIDES[e]], ro.EVENT_SUBORDER[e], draw[ro.TIE_PAIR[e]] ^ ro.EVENT_SIDES[e])
                for e in range(ro.EVENTS)]
        distinct &= len(set(keys)) == ro.EVENTS
        for e in range(ro.EVENTS):table[ro.EVENT_KIND[e]].add(sum(k < keys[e] for k in keys))
        cases += 1
    return table, cases, distinct


def step_source():
    tree = ast.parse((ROOT / 'gpu_engine' / 'transition.py').read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'step_with_chance')


def same(node, source):
    return ast.dump(node) == ast.dump(ast.parse(source).body[0])


def stored(nodes):
    """Names written anywhere in these statements (assignment, augmented assignment, loop or with target, walrus)."""
    return [n.id for node in nodes for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]


def loaded(nodes):
    return {n.id for node in nodes for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


class ConstantsTests(unittest.TestCase):
    def test_event_tables_are_one_event_per_side_per_kind(self):
        """Each kind owns two adjacent events, side 0's then side 1's (the step's `side = event % 2`), at one subOrder and one
        shared tie draw; the pairs' subOrders differ; is_kind (the step's masks) names exactly EVENT_KIND."""
        self.assertEqual(ro.EVENTS, 14)
        self.assertEqual(ro.EVENT_SIDES, tuple(e % 2 for e in range(ro.EVENTS)))
        for kind in ro.KINDS:
            first, stop = ro.kind_events(kind)
            self.assertEqual(stop - first, 2, kind)
            self.assertEqual(len({ro.EVENT_SUBORDER[e] for e in range(first, stop)}), 1, kind)
            self.assertEqual(len({ro.TIE_PAIR[e] for e in range(first, stop)}), 1, kind)
        self.assertEqual(len(set(ro.EVENT_SUBORDER)), len(ro.KINDS));self.assertEqual(len(set(ro.TIE_PAIR)), len(ro.KINDS))
        for e in range(ro.EVENTS):
            self.assertEqual([kind for kind in ro.KINDS if ro.is_kind(e, kind)], [ro.EVENT_KIND[e]], e)
        self.assertEqual(ro.EVENT_SUBORDER, (3, 3, 4, 4, 5, 5, 6, 6, 9, 9, 14, 14, 19, 19))

    def test_key_separation_premise(self):
        """The multiplier exceeds the span of subOrder*2+tie (so Speed decides every cross-side comparison) and every key is
        distinct, for every Speed relation and tie draw: the order is a strict total order."""
        parts = [ro.event_key(0, suborder, tie) for suborder in ro.EVENT_SUBORDER for tie in (0, 1)]
        self.assertEqual(parts, [suborder * 2 + tie for suborder in ro.EVENT_SUBORDER for tie in (0, 1)])
        self.assertEqual((ro.SPEED_MULTIPLIER, max(parts) - min(parts), max(parts)), (64, 33, 39))
        self.assertGreater(ro.SPEED_MULTIPLIER, max(parts) - min(parts))
        self.assertEqual(ro.event_key(1, 0, 0) - ro.event_key(0, 0, 0), -ro.SPEED_MULTIPLIER)   # faster sorts first
        table, cases, distinct = brute_force()
        self.assertEqual(cases, len(SPEEDS) ** 2 * 2 ** len(set(ro.TIE_PAIR)));self.assertTrue(distinct)
        self.assertEqual(list(ro.speed_gaps()), [-1, 0, 1])
        for *_, keys, order in ro.orders():
            self.assertEqual(len(set(keys)), ro.EVENTS);self.assertEqual(sorted(order), list(range(ro.EVENTS)))

    def test_gating_sets_equal_the_brute_force_table(self):
        table, _, _ = brute_force()
        self.assertEqual(table['wrap'], WRAP);self.assertEqual(table['yawn'], YAWN)
        self.assertEqual({kind: set(positions) for kind, positions in ro.POSITIONS.items()}, table)
        self.assertEqual(sum(len(positions) for positions in table.values()), 26)   # of 7 x 14
        self.assertEqual(set(ro.POSITIONS['wrap']), WRAP);self.assertEqual(set(ro.POSITIONS['yawn']), YAWN)

    def test_module_is_jax_free(self):
        code = 'import sys\nimport gpu_engine.residual_order\nsys.exit(int(any(m == "jax" or m.startswith("jax.") for m in sys.modules)))'
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)),
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class StepSourceTests(unittest.TestCase):
    """transition.step_with_chance's residual list: invariants (a) and (b) and the R1 gating."""
    @classmethod
    def setUpClass(cls):
        cls.fn = step_source()
        loops = [n for n in ast.walk(cls.fn) if isinstance(n, ast.For) and same(ast.Expr(n.iter), 'range(residual_order.EVENTS)')]
        assert len(loops) == 1, 'transition.step_with_chance: one `for order in range(residual_order.EVENTS)` loop'
        cls.loop = loops[0];body = cls.fn.body
        cls.before = body[:body.index(cls.loop)]

    def assignment(self, name):
        found = [n for n in ast.walk(self.fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        self.assertEqual(len(found), 1, name)
        return found[0]

    def test_keys_are_built_once_from_the_constants_before_the_loop(self):
        """Invariant (a): every event keeps its real key (no sentinel, no mask in the keys), all 14 are argsorted, and the
        loop visits every position in rank order. Invariant (b): Speed, the tie, the keys and the order are each assigned
        once, before the loop."""
        self.assertIsInstance(self.loop.target, ast.Name);self.assertEqual(self.loop.target.id, 'order')
        for name, source in (('event_sides', 'event_sides = jnp.array(residual_order.EVENT_SIDES)'),
                             ('event_suborder', 'event_suborder = jnp.array(residual_order.EVENT_SUBORDER)'),
                             ('event_keys', 'event_keys = residual_order.event_key(end_speed[:,event_sides],event_suborder[None,:],tie)'),
                             ('event_order', 'event_order = jnp.argsort(event_keys,axis=1)')):
            node = self.assignment(name)
            self.assertTrue(same(node, source), f'{name}: {ast.unparse(node)}')
            self.assertIn(node, self.before, name)
        for name in ('end_speed', 'tie'):self.assertIn(self.assignment(name), self.before, name)
        self.assertLess(self.before.index(self.assignment('end_speed')), self.before.index(self.assignment('event_keys')))
        self.assertTrue(same(self.loop.body[1], 'event = event_order[:,order]'))
        self.assertEqual(self.before[-1], self.assignment('event_order'))

    def test_nothing_in_the_loop_reorders(self):
        """Invariant (b): the loop never rewrites the ordering names, its position or the event it took, and calls no sort."""
        writes = stored(self.loop.body)
        for name in ('end_speed', 'tie', 'event_sides', 'event_suborder', 'event_keys', 'event_order', 'order'):
            self.assertNotIn(name, writes, name)
        self.assertEqual(writes.count('event'), 1)
        calls = [n.func.attr for n in ast.walk(ast.Module(body=self.loop.body, type_ignores=[]))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse({'argsort', 'sort', 'lexsort', 'sort_key_val'} & set(calls))

    def test_kind_masks_come_from_the_constants(self):
        masks = {f'{kind}_event' for kind in ro.KINDS}
        found = {n.targets[0].id: n for n in self.loop.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in masks}
        self.assertEqual(set(found), masks)
        for kind in ro.KINDS:self.assertTrue(same(found[f'{kind}_event'], f"{kind}_event = residual_order.is_kind(event,'{kind}')"))
        self.assertTrue(same(self.loop.body[2], 'side = event % 2'))

    def test_only_wrap_and_yawn_are_gated_on_their_derived_positions(self):
        """R1 gates exactly two blocks, on residual_order.POSITIONS (never a hand-written tuple), with no else; the other
        handlers run at every position. Each gated block reads only its own kind's mask, and that mask is read nowhere else."""
        gated = [n for n in self.loop.body if isinstance(n, ast.If) and 'order' in loaded([n.test])]
        self.assertEqual([ast.unparse(n.test) for n in gated],
                         ["order in residual_order.POSITIONS['wrap']", "order in residual_order.POSITIONS['yawn']"])
        self.assertFalse([n for n in ast.walk(ast.Module(body=self.loop.body, type_ignores=[])) if isinstance(n, ast.If)
                          and n not in gated and 'order' in loaded([n.test])])
        rest = [n for n in self.loop.body if n not in gated]
        for block, kind in zip(gated, ('wrap', 'yawn')):
            self.assertEqual(block.orelse, [])
            self.assertEqual({name for name in loaded(block.body) if name.endswith('_event')}, {f'{kind}_event'})
            self.assertNotIn(f'{kind}_event', loaded(rest))
        # The v2e_regress M2 recorder keys on the Wrap block's locals: its one HP-line call with use=_wrap_tick sits there.
        wrap_lines = [n for n in ast.walk(ast.Module(body=self.loop.body, type_ignores=[])) if isinstance(n, ast.Call)
                      and ast.unparse(n.func) == 'device_public.hp_line' and '_wrap_tick' in loaded(n.args)]
        self.assertEqual(len(wrap_lines), 1)
        self.assertIn(wrap_lines[0], list(ast.walk(gated[0])))
        self.assertIn('_wrap_holder', stored(gated[0].body));self.assertIn('_yawn_sleep', stored(gated[1].body))


if __name__ == '__main__':
    unittest.main()
