"""Synthetic route/accounting controls, no battle engine or actor execution."""
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from typing import NamedTuple
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import benchmark_device_selfplay as b, device_selfplay_profile as profile

BASE = '09633d5478b8ec46848efbfb380126d67d34e440'


class State(NamedTuple):
    done: object
    turn: object
    goal: object


def fixture(width):
    n = width+20
    initial = State(j.zeros(n, bool), j.zeros(n, j.int32),
        j.ones(n, j.int32).at[-3:].set(4))
    public = j.arange(n, dtype=j.uint32)[:, None]
    roots = j.arange(n, dtype=j.uint32)+100
    return initial, public, roots


@jax.jit
def synthetic_chunk(c, initial, public, roots, params):
    # Original finite retire/reset code with an inert deterministic state.
    live = (c.ids >= 0) & ~c.state.done
    turn = c.state.turn+live.astype(j.int32)
    after = c.state._replace(turn=turn, done=c.state.done | (turn >= c.state.goal))
    return b.retire(c, after, c.trackers+live[:, None].astype(j.uint32), initial, public,
        failure=j.zeros_like(live), requested=j.broadcast_to(live[:, None], (len(live), 2)),
        errors=j.zeros(10, j.int32), max_requests=100, observation_checksum=1, error_views=0)


class AdaptiveDenseTailTests(unittest.TestCase):
    def test_host_boundary_dispatch_preserves_every_cell_without_gathering(self):
        for width in (2048, 8192, 16):
            initial, public, roots = fixture(width)
            start = b.initialize(initial, public, width)
            calls = []; counts = {}
            def route(label):
                def execute(c, *args):
                    calls.append((label, len(c.ids), int(c.cursor), int(j.sum(c.ids >= 0))))
                    return synthetic_chunk(c, *args)
                return execute
            expected, _ = b.run_finite(synthetic_chunk, start, initial, public, roots, {})
            actual, _ = b.run_finite(route('scatter'), start, initial, public, roots, {},
                dense_tail_compiled=route('dense'), route_counts=counts)
            for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                np.testing.assert_array_equal(x, y)
            for label, physical, cursor, active in calls:
                self.assertEqual(physical, width)
                if label == 'dense':
                    self.assertEqual(width, 2048); self.assertEqual(cursor, len(initial.done))
                    self.assertTrue(1 <= active <= 16)
            self.assertEqual(counts.get('dense_chunks', 0), 3 if width == 2048 else 0)
            self.assertEqual(sum(counts.values()), int(actual.ticks))
            self.assertTrue(bool(actual.completed.all()))
            self.assertEqual(int(actual.cursor), len(initial.done))
            self.assertEqual(int(actual.physical_views), 2*width*int(actual.ticks))
            self.assertEqual(int(actual.requested_views), int(actual.decisions.sum()))

    def test_both_static_drivers_compile_before_either_phase_and_count_routes(self):
        initial, public, roots = fixture(2048); events = []
        class Driver:
            def __init__(self, dense): self.dense = dense
            def lower(self, *args):
                events.append(('lower', self.dense))
                lowered = synthetic_chunk.lower(*args)
                dense = self.dense
                class Lowered:
                    def compile(self):
                        events.append(('compile', dense)); fn = lowered.compile()
                        def execute(*args):
                            events.append(('execute', dense)); return fn(*args)
                        return execute
                return Lowered()
        def make(*args, **kwargs): return Driver(kwargs.get('dense_public_updates', False))
        with patch.object(b, 'make_driver', make):
            result = b.measure_pair({}, None, None, None, initial, initial, public, public,
                roots+len(roots), roots, batch=2048, sampling_seed=1, chunk_steps=1,
                adaptive_dense_tail=True)
        self.assertEqual(events[:4], [('lower', False), ('compile', False), ('lower', True), ('compile', True)])
        self.assertTrue(result['dense_tail_width_eligible'])
        self.assertGreaterEqual(result['dense_tail_lowering_service_seconds'], 0)
        self.assertGreaterEqual(result['dense_tail_compile_service_seconds'], 0)
        self.assertEqual(result['warm_route_counts'], {'scatter_chunks': 2, 'dense_chunks': 3})
        self.assertEqual(result['measured_route_counts'], result['warm_route_counts'])
        self.assertEqual(result['completed_games'], len(initial.done))
        self.assertEqual(result['warmup_summary']['completed_games'], len(initial.done))
        self.assertEqual(sum(result['measured_route_counts'].values()), result['ticks'])
        for value in (1, None, j.asarray(True)):
            with self.assertRaisesRegex(ValueError, 'adaptive dense-tail'):
                b.measure_pair(None, None, None, None, None, None, None, None, None, None,
                    batch=2048, sampling_seed=1, adaptive_dense_tail=value)

    def test_static_flag_forwards_only_to_public_step_and_other_bodies_unchanged(self):
        from gpu_engine import transition
        sentinel = object(); calls = []
        def step(*args, **kwargs): calls.append(kwargs); return sentinel
        with patch.object(transition, 'step_with_public', step):
            for flag in (False, True):
                parts = profile.make_components(None, NS(belief='source-only'), None, dense_public_updates=flag)
                self.assertIs(parts['step_with_public']('s', 'a', 'p', 'same_chance'), sentinel)
                self.assertEqual(calls[-1], dict(belief_tables='source-only', chance_inputs='same_chance',
                    **(dict(dense_public_updates=True) if flag else {})))
        changed = {'gpu_engine/benchmark_device_selfplay.py': {'make_driver','run_finite','measure_pair'},
                   'gpu_engine/device_selfplay_profile.py': {'make_components','profile_fixed'}}
        for path, allowed in changed.items():
            original = ast.parse(subprocess.check_output(['git','show',BASE+':'+path], text=True))
            current = ast.parse(Path(path).read_text())
            definitions = {n.name:n for n in original.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
            current.body = [definitions[n.name] if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in allowed else n for n in current.body]
            self.assertEqual(ast.dump(original), ast.dump(current), path)

    def test_actual_driver_forwards_static_dense_mode_with_fresh_same_carry(self):
        from gpu_engine import transition
        from test_gpu_engine_device_carry_donation import fixtures, fake_pipeline
        initial, public, params = fixtures(); roots = j.arange(5, dtype=j.uint32)+100
        observed = []; results = []
        with fake_pipeline() as tables:
            original = transition.step_with_public
            def step(*args, dense_public_updates=False, **kwargs):
                observed.append(dense_public_updates)
                return original(*args, **kwargs)
            with patch.object(transition, 'step_with_public', step):
                for flag in (False, True):
                    driver = b.make_driver(None, None, tables, sampling_seed=1,
                        chunk_steps=1, dense_public_updates=flag)
                    results.append(jax.block_until_ready(driver(b.initialize(initial, public, 2),
                        initial, public, roots, params)))
        self.assertEqual(observed, [False, True])
        for old, new in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
            np.testing.assert_array_equal(old, new)


if __name__ == '__main__': unittest.main()
