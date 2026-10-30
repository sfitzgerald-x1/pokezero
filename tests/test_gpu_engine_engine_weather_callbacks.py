"""Literal entry weather-write controls, no games/models/public formatter."""
import ast
import hashlib
import inspect
import json
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import entry, weather, engine_weather_callbacks as callbacks
from gpu_engine import engine_public_events as public, engine_lifecycle_callbacks as life
from gpu_engine.state import empty_state


def table():
    types = ['_none', 'normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug',
        'ghost', 'steel', 'fire', 'water', 'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark']
    return entry.from_metadata(dict(vocabulary=dict(types=types, moves=['_none', 'splash', 'struggle', 'spikes'],
        abilities=['_none', 'drizzle', 'drought', 'sandstream', 'trace', 'intimidate', 'airlock', 'cloudnine', 'levitate'])))


def state(batch=1):
    s = empty_state(batch, seed=83)
    return s._replace(species=j.ones_like(s.species), stats=j.full_like(s.stats, 200),
        hp=j.full_like(s.hp, 100), max_hp=j.full_like(s.hp, 100), types=s.types.at[..., 0].set(1),
        active=j.broadcast_to(j.array([2, 4], j.int32), (batch, 2)), done=j.zeros_like(s.done))


def tape(s, capacity=8, *, legacy=False):
    b = len(s.turn); z = j.zeros(b, j.int32); side = j.zeros((b, 2), j.int32)
    base = public.CallbackTape(j.zeros((b, capacity, 10), j.int32), z, z, z,
        *[j.zeros(8, j.uint32) for _ in range(4)])
    if legacy:
        return base
    rows = j.arange(b)[:, None]; seats = j.arange(2)[None, :]
    mapping = j.full((b, 2, 6), -1, j.int32).at[rows, seats, s.active].set(0)
    cursor = life.Cursor(j.arange(1, b + 1, dtype=j.int32), j.zeros((b, 8), j.uint32), side,
        s.turn + 1, z, j.ones(b, bool), j.ones(b, bool), s.active, side, j.ones((b, 2), j.int32),
        j.ones((b, 2), bool), j.ones((b, 2), j.int32), mapping, j.ones((b, 2), j.int32))
    return life.LifecycleTape(*base, cursor.stream_id, j.zeros((b, capacity), j.int32),
        j.zeros((b, capacity), j.int32), j.zeros((b, capacity, 2), j.int32),
        j.zeros((b, capacity, 6), j.int32), cursor)


class WeatherCallbacksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = table()

    def same(self, a, b):
        for actual, expected in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(actual, expected)

    def words(self, trace, row=0):
        return np.asarray(trace.words[row, :int(trace.count[row])])

    def test_default_numeric_ast_and_table_abi_are_unchanged(self):
        class StripTrace(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == '_weather_trace is not None':
                    return None
                return self.generic_visit(node)
        def canonical(node):
            if isinstance(node, ast.AST):
                return [type(node).__name__, [(key, canonical(value)) for key, value in ast.iter_fields(node)
                    if key != 'type_params']]
            if isinstance(node, list):
                return [canonical(value) for value in node]
            return node
        parsed = ast.parse(inspect.getsource(entry.enter)).body[0]
        index = [arg.arg for arg in parsed.args.kwonlyargs].index('_weather_trace')
        parsed.args.kwonlyargs.pop(index); parsed.args.kw_defaults.pop(index)
        text = json.dumps(canonical(StripTrace().visit(parsed)), separators=(',', ':'))
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), 'c15078731b3eb22676586f59f5a322d443764d8f5e8eb2049443e26a84bd2eaf')
        self.assertEqual(entry.EntryTable._fields, ('weather_ability', 'intimidate', 'natural_cure',
            'levitate', 'attack_drop_immune', 'swift_swim', 'chlorophyll', 'flying_type', 'ground_type',
            'spikes_move', 'speed_boost', 'truant', 'trace', 'status_cure', 'own_tempo'))

    def test_actual_writes_mixed_same_weather_dead_and_noop_rows(self):
        s = state(8); rows = j.arange(8)
        abilities = j.array([1, 2, 3, 1, 5, 1, 1, 3], j.int32)
        s = s._replace(ability=s.ability.at[rows, 0, 2].set(abilities),
            base_ability=s.base_ability.at[rows, 0, 2].set(abilities),
            weather=j.array([0, 1, 3, 1, 0, 2, 2, 1], j.int32),
            weather_counter=j.array([0, 2, 123, 0, 0, 0, 0, 0], j.int32),
            hp=s.hp.at[5, 0, 2].set(0).at[7, 0, 2].set(1), spikes=s.spikes.at[7, 0].set(1))
        eligible = j.array([True, True, True, True, True, True, False, True])
        expected = entry.enter(s, 0, self.table, eligible=eligible)
        actual, ko, trace = entry.enter(s, 0, self.table, eligible=eligible, _weather_trace=tape(s))
        self.same((actual, ko), expected)
        self.assertEqual(trace.count.tolist(), [1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(self.words(trace, 0).tolist(), [[12, 0, 2, -1, -1, 0, 1, 1, 0, 1]])
        self.assertEqual(self.words(trace, 1).tolist(), [[12, 0, 2, -1, -1, 1, 2, 2, 1, 1]])
        self.assertEqual(self.words(trace, 2).tolist(), [[12, 0, 2, -1, -1, 3, 3, 3, 1, 1]])
        self.assertEqual(trace.refusal.tolist(), [0]*8)
        self.assertTrue(bool(ko[7])); self.assertFalse(bool((trace.words == 123).any()))
        self.assertEqual(trace.event_turn[0, 0].item(), 1)

    def test_noop_permanent_weather_reveals_no_hidden_weather_ability(self):
        receipts = []
        for ability in (0, 1, 6, 7, 8):
            s = state()._replace(weather=j.array([weather.RAIN], j.int32))
            s = s._replace(ability=s.ability.at[0, 0, 2].set(ability),
                base_ability=s.base_ability.at[0, 0, 2].set(ability))
            _, _, trace = entry.enter(s, 0, self.table, eligible=j.ones(1, bool), _weather_trace=tape(s))
            receipts.append(trace)
        for trace in receipts[1:]:
            self.same(trace, receipts[0])
        self.assertFalse(bool(receipts[0].words.any()))

    def test_trace_copy_has_no_copied_start_but_keeps_unknown_even_without_weather(self):
        s = state(3)
        s = s._replace(ability=s.ability.at[:, 0, 2].set(4).at[:, 1, 4].set(j.array([1, 2, 3])),
            base_ability=s.base_ability.at[:, 0, 2].set(4).at[:, 1, 4].set(j.array([1, 2, 3])))
        actual, _, trace = entry.enter(s, 0, self.table, eligible=j.ones(3, bool), _weather_trace=tape(s))
        self.assertEqual(actual.ability[:, 0, 2].tolist(), [1, 2, 3])
        self.assertEqual(actual.weather.tolist(), [0, 0, 0])
        self.assertEqual(trace.count.tolist(), [0, 0, 0])
        self.assertEqual(trace.refusal.tolist(), [256, 256, 256])
        self.assertFalse(bool(trace.words.any()))

    def test_unattributed_copied_weather_proc_and_noop_keep_independent_unknown(self):
        s = state(2)
        s = s._replace(ability=s.ability.at[:, 0, 2].set(1), base_ability=s.base_ability.at[:, 0, 2].set(4),
            weather=j.array([weather.SUN, weather.RAIN], j.int32))
        expected = entry.enter(s, 0, self.table, eligible=j.ones(2, bool))
        actual, ko, trace = entry.enter(s, 0, self.table, eligible=j.ones(2, bool), _weather_trace=tape(s))
        self.same((actual, ko), expected)
        self.assertEqual(trace.count.tolist(), [1, 0])
        self.assertEqual(trace.refusal.tolist(), [256, 256])
        # Future formatter handling may remove its own latch, never copy unknown.
        with patch.object(callbacks, 'entry_weather', side_effect=lambda trace, *args, **kwargs: trace):
            _, _, preserved = entry.enter(s, 0, self.table, eligible=j.ones(2, bool), _weather_trace=tape(s))
        self.assertEqual(preserved.refusal.tolist(), [256, 256])

    def test_actual_ordered_entry_calls_and_jit_preserve_state_rng_and_receipt_order(self):
        s = state(2)
        s = s._replace(ability=s.ability.at[:, 0, 2].set(1).at[:, 1, 4].set(2),
            base_ability=s.base_ability.at[:, 0, 2].set(1).at[:, 1, 4].set(2))
        first = j.array([0, 1], j.int32); second = 1 - first
        initial = tape(s)
        expected, _ = entry.enter(s, first, self.table, eligible=j.ones(2, bool))
        expected, _ = entry.enter(expected, second, self.table, eligible=j.ones(2, bool))
        compiled = jax.jit(lambda value, trace, side:
            entry.enter(value, side, self.table, eligible=j.ones(2, bool), _weather_trace=trace))
        actual, _, trace = jax.block_until_ready(compiled(s, initial, first))
        actual, _, trace = jax.block_until_ready(compiled(actual, trace, second))
        self.same(actual, expected); self.same(actual.keys, s.keys)
        self.assertEqual(trace.count.tolist(), [2, 2])
        self.assertEqual(self.words(trace, 0)[:, [1, 2, 5, 6]].tolist(), [[0, 2, 0, 1], [1, 4, 1, 2]])
        self.assertEqual(self.words(trace, 1)[:, [1, 2, 5, 6]].tolist(), [[1, 4, 0, 2], [0, 2, 2, 1]])

    def test_suppressors_do_not_cancel_public_weather_start_or_reveal_their_identity(self):
        for suppressor in (0, 6, 7):
            s = state()._replace(ability=state().ability.at[0, 0, 2].set(1).at[0, 1, 4].set(suppressor),
                base_ability=state().base_ability.at[0, 0, 2].set(1).at[0, 1, 4].set(suppressor))
            actual, _, trace = entry.enter(s, 0, self.table, eligible=j.ones(1, bool), _weather_trace=tape(s))
            self.assertEqual(actual.weather.tolist(), [weather.RAIN])
            self.assertEqual(self.words(trace).tolist(), [[12, 0, 2, -1, -1, 0, 1, 1, 0, 1]])

    def test_unproved_nonphysical_or_nonpermanent_or_noncanonical_trigger_is_unknown_only(self):
        s = state(5)
        trace = callbacks.entry_weather(tape(s), j.array([2, 0, 0, 0, 0]), j.array([2, 6, 2, 2, 2]),
            j.array([0, 0, 0, 1, 0]), j.array([1, 1, 1, 1, 4]),
            j.zeros(5, j.int32), j.array([0, 0, 1, 0, 0]), eligible=j.ones(5, bool))
        self.assertEqual(trace.count.tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(trace.refusal.tolist(), [256] * 5)
        self.assertFalse(bool(trace.words.any()))
        # The actual write still happens for an unsupported injected Hail row;
        # trace instrumentation must not repair numerics or invent a Gen3 proc.
        modified = self.table._replace(weather_ability=self.table.weather_ability.at[8].set(weather.HAIL))
        source = state()._replace(ability=state().ability.at[0, 0, 2].set(8),
            base_ability=state().base_ability.at[0, 0, 2].set(8))
        normal = entry.enter(source, 0, modified, eligible=j.ones(1, bool))
        actual, ko, unknown = entry.enter(source, 0, modified, eligible=j.ones(1, bool), _weather_trace=tape(source))
        self.same((actual, ko), normal); self.assertEqual(actual.weather.tolist(), [weather.HAIL])
        self.assertEqual(unknown.count.tolist(), [0]); self.assertEqual(unknown.refusal.tolist(), [256])

    def test_dead_or_ineligible_trace_does_not_copy_or_add_unknown(self):
        s = state(3)
        s = s._replace(ability=s.ability.at[:, 0, 2].set(4).at[:, 1, 4].set(1),
            base_ability=s.base_ability.at[:, 0, 2].set(4).at[:, 1, 4].set(1),
            hp=s.hp.at[0, 0, 2].set(0).at[2, 1, 4].set(0))
        actual, _, trace = entry.enter(s, 0, self.table, eligible=j.array([True, False, True]), _weather_trace=tape(s))
        self.assertEqual(actual.ability[:, 0, 2].tolist(), [4, 4, 4])
        self.assertEqual(trace.count.tolist(), [0, 0, 0]); self.assertEqual(trace.refusal.tolist(), [0, 0, 0])

    def test_legacy_tape_abi_and_overflow_and_other_unknowns_are_preserved(self):
        s = state()._replace(ability=state().ability.at[0, 0, 2].set(1).at[0, 1, 4].set(2),
            base_ability=state().base_ability.at[0, 0, 2].set(1).at[0, 1, 4].set(2))
        normal = entry.enter(s, 0, self.table, eligible=j.ones(1, bool))
        actual, ko, old = entry.enter(s, 0, self.table, eligible=j.ones(1, bool), _weather_trace=tape(s, legacy=True))
        self.same((actual, ko), normal); self.assertFalse(bool(old.words.any()))
        small = tape(s, 1)._replace(refusal=j.array([128], j.int32))
        actual, _, small = entry.enter(s, 0, self.table, eligible=j.ones(1, bool), _weather_trace=small)
        _, _, small = entry.enter(actual, 1, self.table, eligible=j.ones(1, bool), _weather_trace=small)
        self.assertEqual(small.count.tolist(), [1])
        self.assertEqual(small.refusal.tolist(), [128 | public.OVERFLOW])


if __name__ == '__main__':
    unittest.main()
