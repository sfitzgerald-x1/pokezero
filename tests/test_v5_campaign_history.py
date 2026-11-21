"""The optional campaign opponent bank shares one resolver across V2 paths."""
from dataclasses import replace
import unittest

from gpu_engine.v2_fused import history_window
from pokezero import v5_run as run
from test_v5_run import plan


def snapshot(iteration):
    digest = f'{iteration+1:064x}'
    return run.Snapshot(iteration, f'full-{iteration}', digest, f'actor-{iteration}', digest, 'a'*64)


class CampaignHistoryTests(unittest.TestCase):
    def setUp(self):
        self.standard = replace(plan(), historical_fraction=.5, historical_window=32)
        self.anchored = replace(self.standard, historical_anchor_slots=8)

    def test_early_history_fills_distinct_slots_without_displacing_recent_actors(self):
        for count in (0, 1, 24, 25, 31, 32, 40):
            with self.subTest(count=count):
                past = tuple(snapshot(i) for i in range(count))
                window = run.opponent_history(self.anchored, snapshot(count), past)
                self.assertEqual(len(set(window)), min(32, count))
                self.assertEqual(window[-24:], past[-24:])
                self.assertEqual(window, history_window(self.anchored, snapshot(count), past))

    def test_mature_log_ages_are_exact_and_assignment_uses_the_same_roster(self):
        current = snapshot(8192)
        past = tuple(snapshot(i) for i in range(8192))
        window = run.opponent_history(self.anchored, current, past)
        ages = tuple(current.iteration - actor.iteration for actor in reversed(window[:-24]))
        self.assertEqual(ages, tuple(32 * 2**i for i in range(1, 9)))
        game = self.anchored.games[0]
        for ordinal in range(15):
            actual = run.assignment(self.anchored, game, ordinal, current, past, history=window)
            self.assertEqual(actual, run.assignment(self.anchored, game, ordinal, current, past))
            self.assertIn(actual['opponent'], (*window, current))

    def test_plan_opt_in_and_legacy_declaration_identity(self):
        before = self.standard.manifest()
        self.assertNotIn('historical_anchor_slots', before)
        anchored = self.anchored.manifest()
        self.assertEqual(anchored.pop('historical_anchor_slots'), 8)
        self.assertNotEqual(anchored.pop('opponent_rule'), before.pop('opponent_rule'))
        self.assertEqual(before, anchored)
        with self.assertRaisesRegex(ValueError, 'eight of 32'):
            replace(self.anchored, historical_window=16)
        past = tuple(snapshot(i) for i in range(60))
        self.assertEqual(run.opponent_history(self.standard, snapshot(60), past), past[-32:])
