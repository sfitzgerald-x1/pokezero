"""The optional campaign opponent bank shares one resolver across V2 paths."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from gpu_engine import v2_collector
from gpu_engine.v2_fused import collector_history_slots, history_window
from pokezero import v5_run as run
from test_v5_run import plan


def snapshot(iteration):
    digest = f'{iteration+1:064x}'
    return run.Snapshot(iteration, f'full-{iteration}', digest, f'actor-{iteration}', digest, 'a'*64)


class CampaignHistoryTests(unittest.TestCase):
    def setUp(self):
        self.standard = replace(plan(), historical_fraction=.5, historical_window=32)
        self.anchored = replace(self.standard, historical_anchor_slots=8)

    def test_compact_collector_uses_one_layout_from_scratch_to_full_history(self):
        compact = SimpleNamespace(compact=True, model=self.anchored.model, historical_fraction=.5,
                                  historical_window=32, historical_anchor_slots=8)
        first_groups = None
        for iteration in (0, 1, 2, 3, 25, 8192):
            with self.subTest(iteration=iteration):
                past = tuple(snapshot(i) for i in range(-1, iteration - 1))
                current = snapshot(iteration - 1)
                selected = tuple(actor.archive_sha256 for actor in history_window(compact, current, past)
                                 if actor is not None)
                slots = collector_history_slots(compact, len(selected))
                archives, groups = v2_collector.canonical_groups(current.archive_sha256, selected, slots=slots)
                self.assertEqual(archives[1:1 + len(selected)], selected)
                self.assertEqual(len(archives), 33)
                self.assertTrue(all(a.startswith(v2_collector.PAD_PREFIX) for a in archives[1 + len(selected):]))
                if first_groups is None:
                    first_groups = groups
                self.assertEqual(groups, first_groups)
        self.assertEqual(collector_history_slots(self.anchored, 3), 4)
        compact.historical_fraction = 0.
        self.assertEqual(collector_history_slots(compact, 0), 0)

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
