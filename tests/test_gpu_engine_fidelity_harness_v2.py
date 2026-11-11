"""PR-T comparator harness: tape v2 `effect` field and the Sleep Clause comparator format."""
import importlib.util
import unittest

import numpy as np


class OverrideFormatTests(unittest.TestCase):
    def test_start_override_accepts_custom_game_with_and_without_sleep_clause(self):
        from pokezero.env import BattleStartOverride, SLEEP_CLAUSE_BATTLE_START_OVERRIDE_FORMAT
        teams = dict(p1='x', p2='y')
        self.assertEqual(BattleStartOverride(teams).format_id, 'gen3customgame')
        self.assertEqual(BattleStartOverride(teams, format_id=SLEEP_CLAUSE_BATTLE_START_OVERRIDE_FORMAT).format_id,
                         'gen3customgame@@@sleepclausemod')
        for bad in ('gen3randombattle', 'gen3customgame@@@freezeclausemod', 'gen3customgame@@@sleepclausemod,x'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):BattleStartOverride(teams, format_id=bad)


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires JAX')
class TapeV2Tests(unittest.TestCase):
    def test_effect_is_the_design_stream_of_the_split_key(self):
        import jax
        import jax.numpy as j
        from gpu_engine.fidelity import EFFECT_BOUNDS, EFFECT_SALT, TAPE_SCHEMA, chance_tape
        from gpu_engine.state import empty_state
        s = empty_state(1, seed=23)
        tape = chance_tape(s, 50000, 0)
        self.assertEqual(tape['schema'], TAPE_SCHEMA)
        key = jax.random.split(s.keys[0], 2)[1]
        expected = jax.random.randint(jax.random.fold_in(key, EFFECT_SALT), (2, 4), 0, j.array(EFFECT_BOUNDS, j.int32))
        self.assertEqual(tape['effect'], np.asarray(expected).tolist())
        self.assertTrue(all(0 <= v < b for row in tape['effect'] for v, b in zip(row, EFFECT_BOUNDS)))
        # The six legacy fields keep their v1 values: draws is still the plain randint of the split key.
        draws = jax.random.randint(key, (2, 12), 0, j.array([2, 100, 100, 100, 16, 16, 100, 4, 8, 2, 2, 2], j.int32))
        self.assertEqual(tape['draws'], np.asarray(draws).tolist())

    def test_suspended_turn_replays_the_saved_effect(self):
        import jax.numpy as j
        from gpu_engine.fidelity import chance_tape
        from gpu_engine.state import empty_state
        s = empty_state(1, seed=29);a = chance_tape(s, 50000, 0)
        # Fidelity R: the engine saves the block in state.effect_draws, which the tape replays (over a caller's copy).
        pending = s._replace(pending_residual=j.array([True]), residual_draws=j.asarray([a['draws']]),
            secondary_draws=j.asarray([a['secondary']]), sleep_talk_draws=j.asarray([a['sleep_talk']]),
            effect_draws=j.asarray([a['effect']]), keys=s.keys + 1)
        b = chance_tape(pending, 50000, 1, [[0] * 4] * 2)
        self.assertEqual((b['draws'], b['effect']), (a['draws'], a['effect']))
        self.assertNotEqual(chance_tape(s._replace(keys=s.keys + 1), 50000, 2)['effect'], a['effect'])
        # An engine without the leaf: the caller's copy of the suspended turn's tape.
        legacy = type('Legacy', (), {k: getattr(pending, k) for k in pending._fields if k != 'effect_draws'})
        self.assertEqual(chance_tape(legacy, 50000, 3, a['effect'])['effect'], a['effect'])

    def test_comparator_formats(self):
        from gpu_engine import fidelity
        self.assertEqual(fidelity.SLEEP_CLAUSE_FORMAT, 'gen3customgame@@@sleepclausemod')
        self.assertEqual(set(fidelity.FORMATS), {'gen3customgame', 'gen3customgame@@@sleepclausemod'})
        with self.assertRaises(ValueError):fidelity.run('missing.npz', 'x', None, None, None, format_id='gen3randombattle')


if __name__ == '__main__':
    unittest.main()
