"""Convention rows on pinned, real Showdown callbacks (no engine import)."""
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from _showdown_root import requires_showdown, showdown_root

ROOT = Path(__file__).resolve().parents[1]
NODE = os.environ.get('NODE') or shutil.which('node')


@requires_showdown()
@unittest.skipUnless(NODE, 'requires node')
class ConventionRowsTests(unittest.TestCase):
    def check_version(self, version):
        result = subprocess.check_output([NODE, str(ROOT/'tests/chance_hooks_v4.cjs'), str(showdown_root()),
                                          str(ROOT/'gpu_engine/chance_hooks.cjs'), str(version)], text=True)
        self.assertGreaterEqual(json.loads(result)['checks'], 700)

    def test_legacy_v4_callback_laws_and_commuting_orders(self):
        self.check_version(4)

    def test_current_v6_callback_laws_and_commuting_orders(self):
        self.check_version(6)


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires JAX')
class TapeV4Tests(unittest.TestCase):
    def test_all_coordinates_match_engine_including_resumed_entry_stream(self):
        import jax.numpy as j
        import numpy as np
        from gpu_engine.fidelity import chance_tape, initial_chance, TAPE_SCHEMA
        from gpu_engine.transition import sample_chance_inputs
        from gpu_engine.state import empty_state
        for seed in (1, 19, 719, 50200):
            state = empty_state(1, seed=seed)
            expected = sample_chance_inputs(state)
            tape = chance_tape(state, seed, 0)
            self.assertEqual(tape['schema'], TAPE_SCHEMA)
            self.assertEqual(TAPE_SCHEMA, 'pokezero.gpu_chance_tape.v6')
            self.assertEqual([len(row) for row in tape['effect']], [5, 5])
            for name in ('draws', 'phazing', 'secondary', 'sleep_talk', 'effect', 'entry_ties'):
                self.assertEqual(tape[name], np.asarray(getattr(expected, name))[0].tolist(), name)
            self.assertEqual(initial_chance(state, allow_inert_gender=False)['initial_tie'], int(expected.initial_tie[0]))
            resumed = state._replace(keys=state.keys+1, replacement_count=j.array([3]), pending_residual=j.array([True]),
                residual_draws=expected.draws, phazing_draws=expected.phazing, secondary_draws=expected.secondary,
                sleep_talk_draws=expected.sleep_talk, effect_draws=expected.effect)
            after = chance_tape(resumed, seed, 1)
            self.assertTrue(after['resuming'])
            for name in ('draws', 'phazing', 'secondary', 'sleep_talk', 'effect'):
                self.assertEqual(after[name], tape[name], name)
            self.assertEqual(after['entry_ties'], np.asarray(sample_chance_inputs(resumed).entry_ties)[0].tolist())


if __name__ == '__main__':
    unittest.main()
