"""Causal Lum source receipts on literal states; no games/checkpoints/public port."""
import ast
import hashlib
import inspect
import json
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import effects, items, mechanics as m, secondary, transition
from gpu_engine import engine_item_callbacks as item, engine_status_callbacks as status
from gpu_engine import engine_public_events as public, engine_lifecycle_callbacks as life
from gpu_engine.state import empty_state


def table():
    types = ['_none', 'normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug',
        'ghost', 'steel', 'fire', 'water', 'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark']
    metadata = dict(vocabulary=dict(types=types, moves=['_none', 'splash', 'thunderwave', 'struggle'],
        species=['_none', 'mew'], items=['_none', 'lumberry', 'leftovers'],
        abilities=['_none', 'synchronize']))
    return transition.MoveTable(j.array([0, 0, 0, 50], j.int32), j.array([0, 1, 13, 0], j.int32),
        j.array([False, False, False, True]), j.full(4, 100, j.int32), j.ones(4, bool),
        j.zeros(4, j.int32), j.ones((18, 18), j.int32), j.ones((18, 18), j.int32),
        j.array(3, j.int32), effects=effects.from_metadata(metadata), items=items.from_metadata(metadata),
        secondary=secondary.from_metadata(metadata))


def state(batch=1):
    s = empty_state(batch, seed=71)
    return s._replace(species=j.ones_like(s.species), level=j.full_like(s.level, 100),
        stats=j.full_like(s.stats, 200), hp=j.full_like(s.hp, 1000), max_hp=j.full_like(s.hp, 1000),
        pp=j.full_like(s.pp, 10), moves=j.ones_like(s.moves), types=s.types.at[..., 0].set(1),
        done=j.zeros_like(s.done))


def tape(s, capacity=16, *, legacy=False):
    """Literal internal typed lifecycle storage, never a public registration."""
    b = s.hp.shape[0]
    zero = j.zeros(b, j.int32); sides = j.zeros((b, 2), j.int32)
    base = public.CallbackTape(j.zeros((b, capacity, 10), j.int32), zero, zero, zero,
        *[j.zeros(8, j.uint32) for _ in range(4)])
    if legacy:
        return base
    rows = j.arange(b)[:, None]; seats = j.arange(2)[None, :]
    mapping = j.full((b, 2, 6), -1, j.int32).at[rows, seats, s.active].set(0)
    cursor = life.Cursor(j.arange(1, b + 1, dtype=j.int32), j.zeros((b, 8), j.uint32), sides,
        s.turn + 1, zero, j.ones(b, bool), j.ones(b, bool), s.active, sides,
        j.ones((b, 2), j.int32), j.ones((b, 2), bool), j.ones((b, 2), j.int32), mapping,
        j.ones((b, 2), j.int32))
    return life.LifecycleTape(*base, cursor.stream_id, j.zeros((b, capacity), j.int32),
        j.zeros((b, capacity), j.int32), j.zeros((b, capacity, 2), j.int32),
        j.zeros((b, capacity, 6), j.int32), cursor)


class ItemCallbacksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = table()

    def same(self, a, b):
        for actual, expected in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(actual, expected)

    def words(self, trace, row=0):
        return np.asarray(trace.words[row, :int(trace.count[row])])

    def test_default_numeric_ast_matches_main_before_trace_instrumentation(self):
        class StripTrace(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == '_status_trace is not None':
                    return None
                return self.generic_visit(node)
        def canonical(node):
            # Python3.12 added empty type_params to FunctionDef. Normalize that
            # nonnumeric field so the frozen3.9/3.13 fingerprints agree.
            if isinstance(node, ast.AST):
                return [type(node).__name__, [(key, canonical(value)) for key, value in ast.iter_fields(node)
                    if key != 'type_params']]
            if isinstance(node, list):
                return [canonical(value) for value in node]
            return node
        for function, expected in ((transition._lum_update,
                'a3b44aafc4a5cb5c12e806430ddc19042bcd56602f01c8f7bdfa3349359edd9f'),
                (secondary.after_status_event,
                '6143ed888b4c1903a3686fc1e8750b84a7c0cd7383fd82c82fdc90bdf82fb41c')):
            parsed = ast.parse(inspect.getsource(function)).body[0]
            value = json.dumps(canonical(StripTrace().visit(parsed)), separators=(',', ':'))
            self.assertEqual(hashlib.sha256(value.encode()).hexdigest(), expected)

    def test_actual_lum_update_status_confusion_both_and_no_consume_guards(self):
        s = state(7)._replace(active=j.array([[2, 4]] * 7, j.int32))
        rows = j.arange(7)
        s = s._replace(item=s.item.at[rows, 0, 2].set(j.array([1, 1, 1, 1, 1, 2, 1])),
            status=s.status.at[rows, 0, 2].set(j.array([m.PAR, 0, m.SLP, m.PSN, 0, m.BRN, m.FRZ])),
            status_counter=s.status_counter.at[rows, 0, 2].set(99),
            confusion_counter=s.confusion_counter.at[rows, 0, 2].set(j.array([0, 42, 7, 0, 0, 0, 0])),
            hp=s.hp.at[3, 0, 2].set(0))
        eligible = j.array([True, True, True, True, True, True, False])
        expected = transition._lum_update(s, self.table, eligible)
        actual, trace = transition._lum_update(s, self.table, eligible, tape(s))
        self.same(actual, expected)
        self.assertEqual(trace.count.tolist(), [1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(self.words(trace, 0).tolist(), [[11, -1, -1, 0, 2, 1, m.PAR, 0, 0, 0]])
        self.assertEqual(self.words(trace, 1).tolist(), [[11, -1, -1, 0, 2, 1, 0, 0, 1, 0]])
        self.assertEqual(self.words(trace, 2).tolist(), [[11, -1, -1, 0, 2, 1, m.SLP, 0, 1, 0]])
        self.assertEqual((trace.refusal & status.UNKNOWN_STATUS).tolist(), [128, 128, 128, 0, 0, 0, 0])
        self.assertFalse(np.isin(np.asarray(trace.words), [42, 99]).any())
        self.assertEqual(trace.event_generation[1, 0].tolist(), [0, 0])
        self.assertEqual(trace.event_turn[1, 0].item(), 1)

    def test_confusion_duration_counterfactual_has_same_receipt(self):
        s = state()._replace(item=state().item.at[0, 1, 0].set(1))
        receipts = []
        for duration in (1, 2, 17, 123):
            changed = s._replace(confusion_counter=s.confusion_counter.at[0, 1, 0].set(duration))
            _, trace = transition._lum_update(changed, self.table, j.ones(1, bool), tape(changed))
            receipts.append(trace)
        for trace in receipts[1:]:
            self.same(trace, receipts[0])

    def test_jit_mixed_lum_rows_preserves_state_and_causal_receipt(self):
        s = state(3)._replace(item=state(3).item.at[:, 1, 0].set(j.array([1, 1, 2])),
            status=state(3).status.at[:, 1, 0].set(j.array([m.SLP, m.PSN, m.PAR])),
            confusion_counter=state(3).confusion_counter.at[0, 1, 0].set(27))
        eligible = j.array([True, False, True])
        expected = transition._lum_update(s, self.table, eligible, tape(s))
        compiled = jax.jit(lambda value, trace: transition._lum_update(value, self.table, eligible, trace))
        actual = jax.block_until_ready(compiled(s, tape(s)))
        self.same(actual, expected)
        self.assertEqual(actual[1].count.tolist(), [1, 0, 0])
        self.assertEqual(actual[1].words[0, 0, 8].item(), 1)

    def test_no_consume_never_reveals_held_item_or_bench_or_dead_holder(self):
        s = state(3)
        s = s._replace(item=s.item.at[:, 1, 0].set(j.array([1, 2, 1])).at[:, 0, 5].set(1),
            status=s.status.at[1, 1, 0].set(m.PAR).at[:, 0, 5].set(m.PSN),
            confusion_counter=s.confusion_counter.at[2, 1, 0].set(19), hp=s.hp.at[2, 1, 0].set(0))
        _, trace = transition._lum_update(s, self.table, j.ones(3, bool), tape(s))
        self.assertFalse(bool(trace.words.any())); self.assertEqual(trace.count.tolist(), [0, 0, 0])

    def test_nested_synchronize_consumes_source_lum_before_original_holder(self):
        s = state(2)._replace(active=j.array([[2, 4], [3, 1]], j.int32))
        rows = j.arange(2); source = j.array([0, 1], j.int32); target = 1 - source
        own, afflicted = s.active[rows, source], s.active[rows, target]
        s = s._replace(item=s.item.at[rows, source, own].set(1).at[rows, target, afflicted].set(1),
            status=s.status.at[rows, target, afflicted].set(m.TOX),
            ability=s.ability.at[rows, target, afflicted].set(1))
        expected = secondary.after_status_event(s, source, target, j.full(2, m.TOX), self.table.secondary,
            applied=j.ones(2, bool), item_table=self.table.items)
        actual, trace = secondary.after_status_event(s, source, target, j.full(2, m.TOX), self.table.secondary,
            applied=j.ones(2, bool), item_table=self.table.items, _status_trace=tape(s))
        self.same(actual, expected)
        self.assertEqual(trace.count.tolist(), [2, 2])
        for row in range(2):
            self.assertEqual(self.words(trace, row).tolist(),
                [[11, -1, -1, int(source[row]), int(own[row]), 1, m.PSN, 0, 0, 0],
                 [11, -1, -1, int(target[row]), int(afflicted[row]), 1, m.TOX, 0, 0, 0]])
        self.assertTrue(bool((trace.refusal & status.UNKNOWN_STATUS).all()))

    def test_after_status_event_no_applied_or_nonreflecting_or_dead_does_not_consume(self):
        s = state(3)._replace(item=state(3).item.at[:, 0, 0].set(1).at[:, 1, 0].set(1),
            status=state(3).status.at[:, 1, 0].set(m.PAR),
            ability=state(3).ability.at[:, 1, 0].set(j.array([1, 0, 1])),
            hp=state(3).hp.at[2, 0, 0].set(0))
        applied = j.array([False, True, True])
        actual, trace = secondary.after_status_event(s, 0, 1, j.full(3, m.PAR), self.table.secondary,
            applied=applied, item_table=self.table.items, _status_trace=tape(s))
        self.assertEqual(trace.count.tolist(), [0, 1, 1])
        self.assertEqual(self.words(trace, 1)[0, 3], 1)
        self.assertEqual(self.words(trace, 2)[0, 3], 1)
        self.assertEqual(actual.item[:, 0, 0].tolist(), [1, 1, 1])

    def test_actual_core_primary_then_nested_lum_order_preserves_default_state_rng_and_legacy_tape(self):
        s = state(2)._replace(active=j.array([[2, 4], [3, 1]], j.int32))
        rows = j.arange(2); first = j.array([0, 1], j.int32); second = 1 - first
        s = s._replace(item=s.item.at[rows, first, s.active[rows, first]].set(1)
                .at[rows, second, s.active[rows, second]].set(1),
            ability=j.ones_like(s.ability),
            stats=s.stats.at[rows, first, s.active[rows, first], 5].set(300),
            moves=s.moves.at[rows, first, s.active[rows, first], 0].set(2))
        draws = j.zeros((2, 2, 12), j.int32).at[:, :, 4].set(1)
        chance = transition.ChanceInputs(draws, *[j.zeros((2, 2), j.int32) for _ in range(4)], j.zeros(2, j.int32))
        actions = j.zeros((2, 2), j.int32)
        expected = transition.step_with_chance(s, actions, self.table, chance)
        actual = transition.step_with_chance(s, actions, self.table, chance,
            _public_trace=tape(s, 32), _lifecycle_trace=True)
        legacy = transition.step_with_chance(s, actions, self.table, chance, _public_trace=tape(s, 32, legacy=True))
        self.same(actual.step, expected); self.same(legacy.step, expected)
        self.assertTrue(bool((legacy.callbacks.words[:, :, 0] <= 3).all()))
        for row in range(2):
            words = self.words(actual.callbacks, row)
            selected = words[np.isin(words[:, 0], [status.STATUS_APPLIED, item.LUM_CONSUMED])]
            self.assertEqual(selected[:, 0].tolist(), [9, 11, 11])
            self.assertEqual(selected[1:, 3].tolist(), [int(first[row]), int(second[row])])
            self.assertEqual(selected[1:, 4].tolist(), [int(s.active[row, first[row]]), int(s.active[row, second[row]])])
        self.assertTrue(bool((actual.callbacks.refusal & status.UNKNOWN_STATUS).all()))

    def test_overflow_and_legacy_abi_do_not_claim_success_or_new_kinds(self):
        s = state()._replace(item=j.ones_like(state().item), status=j.full_like(state().status, m.PAR))
        _, trace = transition._lum_update(s, self.table, j.ones(1, bool), tape(s, 1))
        self.assertEqual(trace.count.tolist(), [1])
        self.assertTrue(bool((trace.refusal & public.OVERFLOW).all()))
        old = tape(s, legacy=True)
        result = item.lum_consumed(old, 0, j.zeros(1, j.int32), j.ones(1, j.int32),
            j.full(1, m.PAR), j.zeros(1, j.int32), j.zeros(1, bool), eligible=j.ones(1, bool))
        self.same(result, old)
        self.assertEqual(public.CallbackTape._fields, ('words', 'count', 'refusal', 'turn',
            'engine_binding', 'source_binding', 'move_binding', 'producer_binding'))

    def test_lum_latch_is_local_to_receipt_helper_without_clearing_sync_unknown(self):
        s = state()._replace(item=state().item.at[0, 1, 0].set(1),
            status=state().status.at[0, 1, 0].set(m.PAR))
        with patch.object(item, 'lum_consumed', side_effect=lambda trace, *args, **kwargs: trace):
            # A future reviewed Lum consumer can remove the helper's own latch
            # without masking any aggregate status refusal bits on mixed paths.
            _, direct = transition._lum_update(s, self.table, j.ones(1, bool), tape(s))
            self.assertEqual(direct.refusal.tolist(), [0])
            _, ordinary = secondary.after_status_event(s, 0, 1, j.full(1, m.PAR), self.table.secondary,
                applied=j.ones(1, bool), item_table=self.table.items, _status_trace=tape(s))
            self.assertEqual(ordinary.refusal.tolist(), [0])
            reflected = s._replace(ability=s.ability.at[0, 1, 0].set(1))
            _, sync = secondary.after_status_event(reflected, 0, 1, j.full(1, m.PAR), self.table.secondary,
                applied=j.ones(1, bool), item_table=self.table.items, _status_trace=tape(reflected))
            self.assertEqual(sync.refusal.tolist(), [status.UNKNOWN_STATUS])


if __name__ == '__main__':
    unittest.main()
