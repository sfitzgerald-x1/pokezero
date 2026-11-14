"""R1 and R2 (Fidelity M2's compile-size restructures): the residual event positions that gate every residual handler.

transition.step_with_chance emits each order-10 residual handler only where its events can sort, residual_order.POSITIONS:
Speed Boost 0, 1, 7; Leftovers and the pinch berries 1, 2, 3, 8; Leech Seed 2, 4, 5, 9; brn/psn/tox 3, 6, 7, 10; Wrap 4, 8,
9, 11 (R1); Encore 5, 10, 11, 12; Yawn 6, 12, 13 (R1). That is exact only under two invariants, checked here on
transition.py's own source:
(a) a position is a rank among all 14 keys whether or not its event is active: no sentinel key for an inactive event, and
    the loop visits every position (no compaction);
(b) Speed and the tie are fixed before the loop, and nothing inside the loop reorders.
The reachable position x kind table is recomputed by brute force from the step's own constants and key function (imported,
never copied) over faster, slower and equal Speeds and every tie bit, and the gating sets must equal it. A changed key
formula, multiplier, subOrder or event list fails here instead of silently dropping events. R2's shared Leftovers/status
HP write, its lifecycle-mode rules (Encore at every position; every handler without the entry table) and the untouched
per-position common code are checked on the AST too. Pure Python: no JAX import.
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
# Every kind's positions (R2 gates the five older handlers on these too).
EXPECTED = {'ability': {0, 1, 7}, 'heal': {1, 2, 3, 8}, 'seed': {2, 4, 5, 9}, 'status': {3, 6, 7, 10}, 'wrap': WRAP,
            'encore': {5, 10, 11, 12}, 'yawn': YAWN}
OLD = ('ability', 'heal', 'seed', 'status', 'encore')
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
        # R1 review: pin which handler each event runs and which draw its pair shares, not only the subOrders. A swapped
        # label (e.g. 'status' <-> 'encore') would run brn/psn/tox at subOrder 14 and Encore at 6, and still pass the rest.
        self.assertEqual(ro.EVENT_KIND, ('ability', 'ability', 'heal', 'heal', 'seed', 'seed', 'status', 'status',
                                         'wrap', 'wrap', 'encore', 'encore', 'yawn', 'yawn'))
        self.assertEqual(ro.TIE_PAIR, (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6))
        self.assertEqual({ro.EVENT_KIND[e]: ro.EVENT_SUBORDER[e] for e in range(ro.EVENTS)},
                         {'ability': 3, 'heal': 4, 'seed': 5, 'status': 6, 'wrap': 9, 'encore': 14, 'yawn': 19})

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
        self.assertEqual(table, EXPECTED)   # all seven kinds (R2)
        self.assertEqual({kind: set(positions) for kind, positions in ro.POSITIONS.items()}, table)
        self.assertEqual(sum(len(positions) for positions in table.values()), 26)   # of 7 x 14
        self.assertEqual(sum(len(table[kind]) for kind in OLD), 19)                 # R2: of the older handlers' 5 x 14
        for position in range(ro.EVENTS):                                            # each position holds at most 2 kinds
            self.assertIn(len([kind for kind in ro.KINDS if position in table[kind]]), (1, 2), position)
        self.assertEqual(set(ro.POSITIONS['wrap']), WRAP);self.assertEqual(set(ro.POSITIONS['yawn']), YAWN)
        with self.assertRaises(TypeError):ro.POSITIONS['wrap'] = frozenset()     # read-only (R1 review)
        self.assertTrue(all(isinstance(positions, frozenset) for positions in ro.POSITIONS.values()))

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
        # R1 review: the derivation assumes every tie is one bit, draw ^ side (complementary per pair; the Yawn pair's fixed
        # `sides`). Pin the step's tie expression and the bounds of every column it reads, so a wider column (e.g. a
        # 0..99 roll) cannot widen a kind's positions past the gated sets unnoticed.
        tie_source = ('tie = jnp.concatenate((draws[:,1,0,None] ^ sides, draws[:,0,9,None] ^ sides,draws[:,0,10,None] ^ sides,'
                      'draws[:,1,9,None] ^ sides, effect_draws[:,0,3,None] ^ sides,draws[:,1,11,None] ^ sides,'
                      ' jnp.broadcast_to(sides,(draws.shape[0],2))),axis=1)')
        self.assertTrue(same(self.assignment('tie'), tie_source), ast.unparse(self.assignment('tie')))
        self.assertTrue(same(self.assignment('sides'), 'sides = jnp.arange(2)[None, :]'), ast.unparse(self.assignment('sides')))
        module = ast.parse((ROOT / 'gpu_engine' / 'transition.py').read_text())
        bounds = [ast.literal_eval(n.args[0]) for n in ast.walk(module) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == 'jnp.array' and n.args and isinstance(n.args[0], ast.List) and len(n.args[0].elts) == 12]
        self.assertEqual(bounds, [[2, 100, 100, 100, 16, 16, 100, 4, 8, 2, 2, 2]])
        self.assertEqual([bounds[0][c] for c in (0, 9, 10, 11)], [2, 2, 2, 2])
        effect = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'EFFECT_BOUNDS' for t in n.targets))
        self.assertEqual(effect[3], 2)
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

    def gates(self):
        """Every If in the loop whose test reads the position, in program order, with the kind it gates."""
        found = [n for n in ast.walk(ast.Module(body=self.loop.body, type_ignores=[])) if isinstance(n, ast.If)
                 and 'order' in loaded([n.test])]
        found.sort(key=lambda n: (n.lineno, n.col_offset))
        kinds = [ast.literal_eval(next(c for c in ast.walk(n.test) if isinstance(c, ast.Subscript)).slice) for n in found]
        return found, kinds

    def test_every_handler_is_gated_on_its_derived_positions(self):
        """R1 and R2 gate each handler's blocks on residual_order.POSITIONS, never a hand-written tuple. In lifecycle mode
        Encore runs at every position (its kernel-risk branch counts each visit) and, without the entry table, so does every
        older handler (_all_positions); Wrap and Yawn (R1) have neither clause and no else."""
        found, kinds = self.gates()
        form = lambda kind, extra=None: f"order in residual_order.POSITIONS['{kind}']" + (f' or {extra}' if extra else '')
        every = '_all_positions'
        self.assertEqual([ast.unparse(n.test) for n in found], [
            form('ability', every), form('heal', every), form('status', every),
            form('heal'),                                   # the status gate's elif: the heal-only HP write
            form('heal', every), form('status', every),     # the device observe (heal) and hp_line (status)
            form('heal', every), form('status', every), form('heal', every), form('seed', every),
            form('wrap'), form('yawn'), form('encore', '_lifecycle_trace')])
        self.assertTrue(same(self.assignment('_all_positions'), '_all_positions = _lifecycle_trace and table.entry is None'))
        self.assertIn(self.assignment('_all_positions'), self.before)
        for block, kind in zip(found, kinds):
            if kind in ('wrap', 'yawn'):self.assertEqual(block.orelse, [], kind)
        # The v2e_regress M2 recorder keys on the Wrap block's locals: its one HP-line call with use=_wrap_tick sits there.
        wrap, yawn = found[kinds.index('wrap')], found[kinds.index('yawn')]
        wrap_lines = [n for n in ast.walk(ast.Module(body=self.loop.body, type_ignores=[])) if isinstance(n, ast.Call)
                      and ast.unparse(n.func) == 'device_public.hp_line' and '_wrap_tick' in loaded(n.args)]
        self.assertEqual(len(wrap_lines), 1)
        self.assertIn(wrap_lines[0], list(ast.walk(wrap)))
        self.assertIn('_wrap_holder', stored(wrap.body));self.assertIn('_yawn_sleep', stored(yawn.body))

    def test_each_mask_is_read_only_under_its_own_gates(self):
        """A kind's mask is read only inside that kind's gates, and the gated code reads no other kind's mask, except the
        shared Leftovers/status HP write's heal_event, under the status gate."""
        found, kinds = self.gates()
        shared = found[kinds.index('status')].body[-1]
        for block, kind in zip(found, kinds):
            body = [n for n in block.body if n is not shared]
            self.assertLessEqual({name for name in loaded(body) if name.endswith('_event')}, {f'{kind}_event'}, kind)
        self.assertEqual({name for name in loaded([shared]) if name.endswith('_event')}, {'heal_event'})
        inner = {id(n) for g in found for n in ast.walk(ast.Module(body=g.body, type_ignores=[]))}
        outside = [n for n in ast.walk(ast.Module(body=self.loop.body, type_ignores=[])) if isinstance(n, ast.Name)
                   and isinstance(n.ctx, ast.Load) and n.id.endswith('_event') and id(n) not in inner]
        self.assertEqual(outside, [])

    def test_the_shared_heal_and_status_write_reduces_exactly(self):
        """The Leftovers/status HP write sits last under the first status gate. Where no status event sorts (the heal-only
        positions) its elif writes where(heal_event, healed, hp): major_status_residual(active all false) deals 0 and returns
        the counter unchanged, so residual.hp is hp and the counter write is the identity. Where no heal event sorts, healed
        stays the common `healed = hp`, assigned before the first heal gate and elsewhere only under heal gates."""
        found, kinds = self.gates()
        status = found[kinds.index('status')]
        self.assertTrue(same(status.body[-1], 'work = work._replace(hp=_set_active(work.hp, work, side,jnp.where(heal_event,'
            'healed,residual.hp)), status_counter=_set_active(work.status_counter, work, side, residual.counter))'))
        self.assertTrue(same(status.body[-2], 'residual = m.major_status_residual(hp, _active(work.max_hp, work, side), '
            '_active(work.status, work, side), _active(work.status_counter, work, side), active=eligible & status_event)'))
        self.assertEqual(len(status.orelse), 1);elif_ = status.orelse[0]
        self.assertIs(elif_, found[3]);self.assertEqual(elif_.orelse, [])
        self.assertEqual(len(elif_.body), 1)
        self.assertTrue(same(elif_.body[0], 'work = work._replace(hp=_set_active(work.hp,work,side,jnp.where(heal_event,healed,hp)))'))
        top = self.loop.body
        healed = [n for n in top if same(n, 'healed = hp')]
        self.assertEqual(len(healed), 1)
        self.assertLess(top.index(healed[0]), top.index(found[kinds.index('heal')]))
        heal_blocks = [g for g, k in zip(found, kinds) if k == 'heal']
        writes = [n for n in ast.walk(ast.Module(body=top, type_ignores=[])) if isinstance(n, ast.Name)
                  and isinstance(n.ctx, ast.Store) and n.id == 'healed']
        in_heal = {id(n) for g in heal_blocks for n in ast.walk(g)}
        self.assertEqual([n for n in writes if id(n) not in in_heal], [healed[0].targets[0]])

    def test_gated_names_are_not_read_stale(self):
        """A name stored only under a kind's gates is read only under that kind's gates (else a skipped position would read
        the previous position's value), or is a carry assigned before the loop (the lifecycle tape)."""
        found, kinds = self.gates()
        def kind_of(node):   # the innermost gate wins (the elif and the device gates nest)
            best = None
            for block, kind in zip(found, kinds):
                if any(x is node for x in ast.walk(ast.Module(body=block.body, type_ignores=[]))):
                    if best is None or block.lineno > best[0]:best = (block.lineno, kind)
            return 'common' if best is None else best[1]
        names = [n for n in ast.walk(ast.Module(body=self.loop.body, type_ignores=[])) if isinstance(n, ast.Name)]
        where = {}
        for n in names:
            if isinstance(n.ctx, ast.Store):where.setdefault(n.id, set()).add(kind_of(n))
        carried = set(stored(self.before))
        stale = sorted({(n.id, kind_of(n)) for n in names if isinstance(n.ctx, ast.Load) and n.id in where
                        and 'common' not in where[n.id] and kind_of(n) not in where[n.id] and n.id not in carried})
        self.assertEqual(stale, [])
        self.assertIn('_public_trace', carried)

    def test_the_per_position_common_code_is_not_gated(self):
        """eligible, battle_continues, the faint snapshot, the device baseline, the fainted-boost clear and the two observer
        retains with eligible (the status residual and Leftovers contexts' per-position profile carry) run at every position."""
        top = self.loop.body
        found, _ = self.gates()
        self.assertTrue(same(top[0], '_faint_before_hp=work.hp'))
        self.assertTrue(same(top[-1], 'work=_clear_fainted_boosts(work,_faint_before_hp,eligible=eligible)'))
        for source in ('battle_continues = jnp.all(jnp.any(work.hp > 0, axis=2), axis=1)',
                       'eligible = residual_ready & battle_continues', 'hp = _active(work.hp,work,side)',
                       'if _device_public is not None:_device_residual_before=work'):
            self.assertEqual(len([n for n in top if same(n, source)]), 1, source)
        inner = {id(n) for g in found for n in ast.walk(g)}
        retains = [n for n in ast.walk(ast.Module(body=top, type_ignores=[])) if isinstance(n, ast.Call)
                   and ast.unparse(n.func).endswith('.retain_profile_device')
                   and any(k.arg == 'eligible' and ast.unparse(k.value) == 'eligible' for k in n.keywords)]
        self.assertEqual(sorted(ast.unparse(n.func) for n in retains),
                         ['leftovers_callbacks.retain_profile_device', 'status_residual_callbacks.retain_profile_device'])
        self.assertFalse([n for n in retains if id(n) in inner])


if __name__ == '__main__':
    unittest.main()
