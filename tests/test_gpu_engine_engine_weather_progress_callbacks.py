"""Literal causal weather operation controls, no games/models/public formatter."""
import ast
import hashlib
import inspect
import json
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import weather, engine_weather_progress_callbacks as callbacks
from gpu_engine import engine_public_events as public
import test_gpu_engine_engine_item_callbacks as fixtures


class WeatherProgressTests(unittest.TestCase):
    def tape(self, batch=1, capacity=16, legacy=False):
        return fixtures.tape(fixtures.state(batch), capacity, legacy=legacy)

    def same(self, a, b):
        for actual, expected in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(actual, expected)

    def words(self, trace, row=0):
        return np.asarray(trace.words[row, :int(trace.count[row])])

    def test_default_numeric_ast_and_weather_table_abi_are_unchanged(self):
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
        expected = {weather.set_move: '2b5b6b05368770863aab049d011af115690e9b36b1e049f71f8fef1ffd156fae',
            weather.residual_clock: 'c9e04d658d330ef333b58d4dd45ca3f73dc424cf05a147b1cb0f4dde12afa8aa'}
        for function, digest in expected.items():
            parsed = ast.parse(inspect.getsource(function)).body[0]
            for i in range(len(parsed.args.kwonlyargs) - 1, -1, -1):
                if parsed.args.kwonlyargs[i].arg != 'eligible':
                    parsed.args.kwonlyargs.pop(i); parsed.args.kw_defaults.pop(i)
            text = json.dumps(canonical(StripTrace().visit(parsed)), separators=(',', ':'))
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), digest)
        self.assertEqual(weather.WeatherTable._fields, ('move_weather', 'thunder', 'fire_type', 'water_type',
            'sand_immune_type', 'ice_type', 'suppressor', 'sand_veil'))

    def test_all_four_actual_weather_move_writes_bind_exact_called_identity(self):
        old = j.zeros(4, j.int32); new = j.arange(1, 5, dtype=j.int32)
        moves = j.array([16777217, 92, 93, 94], j.int32)
        actual, counter, trace = weather.set_move(old, old, new, _weather_trace=self.tape(4),
            _actor_side=j.array([0, 1, 0, 1], j.int32), _actor_physical=j.array([2, 4, 3, 1], j.int32),
            _move_id=moves)
        self.same((actual, counter), weather.set_move(old, old, new))
        for row in range(4):
            self.assertEqual(self.words(trace, row).tolist(),
                [[13, row % 2, [2, 4, 3, 1][row], -1, -1, 0, row + 1, int(moves[row]), 0, 5]])
        self.assertEqual(trace.refusal.tolist(), [0] * 4)

    def test_same_weather_temporary_permanent_none_and_ineligible_do_not_reveal_uncalled_ids(self):
        old = j.array([0, 1, 1, 3, 4, 0, 2, 0], j.int32)
        counters = j.array([0, 2, 0, 0, 1, 0, 3, 0], j.int32)
        requested = j.array([1, 1, 1, 2, 4, 0, 3, 1], j.int32)
        eligible = j.array([True] * 7 + [False])
        moves = j.array([101, 999998, 999997, 104, 999996, 999995, 107, 999994], j.int32)
        actual, counter, trace = weather.set_move(old, counters, requested, eligible=eligible,
            _weather_trace=self.tape(8), _actor_side=0, _actor_physical=2, _move_id=moves)
        self.same((actual, counter), weather.set_move(old, counters, requested, eligible=eligible))
        self.assertEqual(trace.count.tolist(), [1, 0, 0, 1, 0, 0, 1, 0])
        self.assertEqual(trace.refusal.tolist(), [0, 256, 256, 0, 256, 0, 0, 0])
        self.assertFalse(np.isin(np.asarray(trace.words), [999998, 999997, 999996, 999995, 999994]).any())
        self.assertEqual(counter.tolist(), [5, 2, 0, 5, 1, 0, 5, 0])

    def test_repeat_does_not_refresh_and_source_order_is_move_then_upkeep_then_expiry(self):
        old = j.array([0], j.int32); count = j.array([0], j.int32)
        w, c, trace = weather.set_move(old, count, j.array([weather.SAND], j.int32),
            _weather_trace=self.tape(), _actor_side=0, _actor_physical=2, _move_id=91)
        w, c, trace = weather.set_move(w, c, j.array([weather.SAND], j.int32),
            _weather_trace=trace, _actor_side=1, _actor_physical=4, _move_id=999998)
        self.assertEqual(c.tolist(), [5]); self.assertEqual(trace.count.tolist(), [1])
        for _ in range(5):
            w, c, trace = weather.residual_clock(w, c, _weather_trace=trace)
        self.assertEqual((w.tolist(), c.tolist()), ([0], [0]))
        words = self.words(trace)
        self.assertEqual(words[:, 0].tolist(), [13] + [14] * 5)
        self.assertEqual(words[1:, 9].tolist(), [callbacks.UPKEEP] * 4 + [callbacks.EXPIRE])
        self.assertEqual(words[1:, 7:9].tolist(), [[5, 4], [4, 3], [3, 2], [2, 1], [1, 0]])
        self.assertEqual(words[-1, 5:7].tolist(), [weather.SAND, weather.NONE])
        _, _, after = weather.residual_clock(w, c, _weather_trace=trace)
        self.same(after, trace)

    def test_clock_mixed_permanent_expiring_live_and_inactive_rows(self):
        old = j.array([1, 2, 3, 4, 3, 0, 1], j.int32)
        counters = j.array([2, 1, 0, 5, 1, 0, 2], j.int32)
        eligible = j.array([True, True, True, True, False, True, False])
        actual, counter, trace = weather.residual_clock(old, counters, eligible=eligible, _weather_trace=self.tape(7))
        self.same((actual, counter), weather.residual_clock(old, counters, eligible=eligible))
        self.assertEqual(trace.count.tolist(), [1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(self.words(trace, 0).tolist(), [[14, -1, -1, -1, -1, 1, 1, 2, 1, 1]])
        self.assertEqual(self.words(trace, 1).tolist(), [[14, -1, -1, -1, -1, 2, 0, 1, 0, 2]])
        self.assertEqual(self.words(trace, 2).tolist(), [[14, -1, -1, -1, -1, 3, 3, 0, 0, 1]])
        self.assertEqual(trace.refusal.tolist(), [0] * 7)

    def test_missing_or_invalid_source_identity_stays_unknown_without_fabricated_move(self):
        old, count, request = j.array([0], j.int32), j.array([0], j.int32), j.array([1], j.int32)
        for values in ({}, dict(_actor_side=2, _actor_physical=2, _move_id=91),
                dict(_actor_side=0, _actor_physical=6, _move_id=91),
                dict(_actor_side=0, _actor_physical=2, _move_id=0),
                dict(_actor_side=0, _actor_physical=2, _move_id=j.array([1.25], j.float32)),
                dict(_actor_side=0, _actor_physical=2, _move_id=np.array([2**32 + 1], np.int64))):
            with self.subTest(values=values):
                w, c, trace = weather.set_move(old, count, request, _weather_trace=self.tape(), **values)
                self.assertEqual((w.tolist(), c.tolist()), ([1], [5]))
                self.assertEqual(trace.count.tolist(), [0]); self.assertEqual(trace.refusal.tolist(), [256])
        _, _, no_op = weather.set_move(request, count, request, _weather_trace=self.tape())
        self.assertEqual(no_op.refusal.tolist(), [256]); self.assertFalse(bool(no_op.words.any()))

    def test_invalid_weather_or_clock_operation_is_unknown_not_a_lexical_event(self):
        old = j.array([6, 1, 0], j.int32); counters = j.array([2, -1, 2], j.int32)
        w, c, trace = weather.residual_clock(old, counters, _weather_trace=self.tape(3))
        self.same((w, c), weather.residual_clock(old, counters))
        self.assertEqual(trace.count.tolist(), [0, 0, 0]); self.assertEqual(trace.refusal.tolist(), [256] * 3)
        invalid = callbacks.weather_clock(self.tape(), j.array([1], j.int32), j.array([1], j.int32),
            j.array([1], j.int32), j.array([0], j.int32), eligible=j.array([True]))
        self.assertEqual(invalid.count.tolist(), [0]); self.assertEqual(invalid.refusal.tolist(), [256])

    def test_malformed_empty_to_weather_clock_is_unknown_but_empty_noop_is_not(self):
        # Mixed rows distinguish a malformed unsolicited weather creation, the
        # legitimate empty no-op, and an ineligible row outside this operation.
        trace = callbacks.weather_clock(self.tape(3), j.zeros(3, j.int32),
            j.array([weather.RAIN, weather.NONE, weather.RAIN], j.int32),
            j.zeros(3, j.int32), j.zeros(3, j.int32), eligible=j.array([True, True, False]))
        self.assertEqual(trace.count.tolist(), [0, 0, 0])
        self.assertEqual(trace.refusal.tolist(), [256, 0, 0])
        self.assertFalse(bool(trace.words.any()))

    def test_suppression_and_suppressor_life_do_not_stop_raw_weather_clock_messages(self):
        types = ['_none', 'normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug',
            'ghost', 'steel', 'fire', 'water', 'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark']
        table = weather.from_metadata(dict(vocabulary=dict(types=types, moves=['_none', 'raindance'],
            abilities=['_none', 'airlock', 'cloudnine'])))
        receipts = []
        raw, clock = j.array([weather.RAIN], j.int32), j.array([2], j.int32)
        for ability, hp in ((0, 100), (1, 100), (2, 100), (1, 0)):
            effective = weather.effective(table, raw, j.array([[ability, 0]], j.int32), j.array([[hp, 100]], j.int32))
            self.assertEqual(effective.tolist(), [0 if ability and hp else weather.RAIN])
            _, _, trace = weather.residual_clock(raw, clock, _weather_trace=self.tape())
            receipts.append(trace)
        for trace in receipts[1:]:
            self.same(trace, receipts[0])
        self.assertEqual(self.words(receipts[0])[0, 9], callbacks.UPKEEP)

    def test_jit_matches_eager_operation_order_without_rng_or_other_state_mutation(self):
        fixture = fixtures.state(2); before = jax.tree.map(lambda value: np.asarray(value).copy(), fixture)
        def run(w, c, requested, trace):
            w, c, trace = weather.set_move(w, c, requested, _weather_trace=trace,
                _actor_side=j.array([0, 1], j.int32), _actor_physical=j.array([2, 4], j.int32),
                _move_id=j.array([91, 92], j.int32))
            return weather.residual_clock(w, c, _weather_trace=trace)
        args = (j.array([0, 2], j.int32), j.array([0, 1], j.int32), j.array([1, 2], j.int32), self.tape(2))
        expected = run(*args); actual = jax.block_until_ready(jax.jit(run)(*args))
        self.same(actual, expected); self.same(fixture, before)
        self.assertEqual(self.words(actual[2], 0)[:, 0].tolist(), [13, 14])
        self.assertEqual(self.words(actual[2], 1)[:, 0].tolist(), [14])
        self.assertEqual(self.words(actual[2], 1)[0, 9], callbacks.EXPIRE)

    def test_legacy_abi_overflow_and_central_latch_never_clear_other_refusals(self):
        w, c, old = weather.set_move(j.array([0], j.int32), j.array([0], j.int32), j.array([1], j.int32),
            _weather_trace=self.tape(legacy=True), _actor_side=0, _actor_physical=2, _move_id=91)
        _, _, old = weather.residual_clock(w, c, _weather_trace=old)
        self.assertFalse(bool(old.words.any())); self.assertEqual(old.refusal.tolist(), [0])
        initial = self.tape(capacity=1)._replace(refusal=j.array([128], j.int32))
        w, c, trace = weather.set_move(j.array([0], j.int32), j.array([0], j.int32), j.array([1], j.int32),
            _weather_trace=initial, _actor_side=0, _actor_physical=2, _move_id=91)
        _, _, trace = weather.residual_clock(w, c, _weather_trace=trace)
        self.assertEqual(trace.count.tolist(), [1]); self.assertEqual(trace.refusal.tolist(), [128 | public.OVERFLOW])
        with patch.object(callbacks, '_unknown', side_effect=lambda trace, eligible: trace):
            _, _, preserved = weather.set_move(j.array([0], j.int32), j.array([0], j.int32), j.array([1], j.int32),
                _weather_trace=initial, _actor_side=0, _actor_physical=2, _move_id=91)
        self.assertEqual(preserved.refusal.tolist(), [128])


if __name__ == '__main__':
    unittest.main()
