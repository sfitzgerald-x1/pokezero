"""CPU tests of later campaign phases using an LR-start interface double."""
from dataclasses import asdict, dataclass, field, replace
import json
import unittest

from pokezero.v2_campaign_schedule import CampaignSchedule, CheckpointAnchor
from pokezero.v5_training import DeclaredGameWindow, TrainConfig, train_config_metadata


@dataclass
class StubLRStart:
    initial_lr: float = .002
    warmup_updates: int = 3
    scaling: str = 'linear'
    calls: list[int] = field(default_factory=list)

    def rate_at(self, completed_updates):
        self.calls.append(completed_updates)
        return (self.initial_lr / 5, self.initial_lr / 2, self.initial_lr)[completed_updates]


class CampaignScheduleTests(unittest.TestCase):
    def test_start_delegates_exact_counters_and_never_scales_peak_again(self):
        window = DeclaredGameWindow(0, 20, 20)
        schedule = CampaignSchedule()
        for scaling, peak in (('linear', .002), ('sqrt', .00067)):
            with self.subTest(scaling=scaling):
                start = StubLRStart(peak, scaling=scaling)
                schedule.validate_next_window(start, window, 0)
                rates = [schedule.window_rate(start, window, completed_updates=i, offset=i, window_updates=5)
                         for i in range(5)]
                self.assertEqual(rates, [peak / 5, peak / 2, peak, peak, peak])
                self.assertEqual(start.calls, [0, 1, 2])

    def test_steps_use_absolute_game_boundaries_with_no_declared_horizon_dependency(self):
        start = StubLRStart()
        schedule = CampaignSchedule((20, 40))
        for horizon in (60, 100_000_000):
            for games, steps in ((0, 0), (19, 0), (20, 1), (39, 1), (40, 2), (59, 2)):
                window = DeclaredGameWindow(games, games + 1, horizon)
                schedule.validate_next_window(start, window, 8)
                actual = schedule.window_rate(start, window, completed_updates=8, offset=0, window_updates=1)
                self.assertEqual(actual, start.initial_lr * .3 ** steps)
        self.assertEqual(start.calls, [])

    def test_step_requires_an_iteration_boundary_and_completed_warmup(self):
        start = StubLRStart()
        schedule = CampaignSchedule((20,))
        with self.assertRaisesRegex(ValueError, 'iteration boundary'):
            schedule.validate_next_window(start, DeclaredGameWindow(10, 30, 60), 10)
        with self.assertRaisesRegex(ValueError, 'completed warmup'):
            schedule.validate_next_window(start, DeclaredGameWindow(20, 30, 60), 2)
        with self.assertRaisesRegex(ValueError, 'linear/sqrt'):
            schedule.validate_next_window(StubLRStart(scaling='other'), DeclaredGameWindow(0, 10, 60), 0)

    def test_child_uses_actual_parent_rate_and_keeps_future_parent_step_separate(self):
        start = StubLRStart()
        original = CampaignSchedule((20, 40))
        anchor = CheckpointAnchor('a' * 64, 40, 12, start.initial_lr * .3)
        child = original.fork_cooldown(start, anchor, end_game=60)
        self.assertEqual(original.step_down_games, (20, 40))
        self.assertIsNone(original.cooldown)
        self.assertEqual(child.step_down_games, (20,))
        self.assertEqual(child.cooldown.parent, anchor)
        for game, factor in ((40, 1.), (50, .55), (60, .1)):
            self.assertAlmostEqual(child.cooldown.rate_at_game(game), anchor.applied_lr * factor)
        with self.assertRaisesRegex(ValueError, 'outside'):
            child.cooldown.rate_at_game(61)

    def test_restored_cooldown_replays_the_same_next_rates(self):
        start = StubLRStart()
        original = CampaignSchedule()
        anchor = CheckpointAnchor('a' * 64, 40, 12, start.initial_lr)
        child = original.fork_cooldown(start, anchor, end_game=60)
        bound = DeclaredGameWindow(40, 50, 60)
        child.validate_next_window(start, bound, 12)
        expected = [child.window_rate(start, bound, completed_updates=12+i, offset=i, window_updates=5)
                    for i in range(5)]
        state = json.loads(json.dumps(dict(schedule=asdict(child), window=asdict(bound), completed_updates=14, offset=2)))
        restored = CampaignSchedule(**state['schedule'])
        window = DeclaredGameWindow(**state['window'])
        actual = [restored.window_rate(start, window, completed_updates=state['completed_updates']+i,
                                       offset=state['offset']+i, window_updates=5) for i in range(3)]
        self.assertEqual((actual, restored.cooldown.parent), (expected[2:], anchor))
        self.assertEqual(start.calls, [])

    def test_rejects_inconsistent_parent_and_branch_extent(self):
        start = StubLRStart()
        schedule = CampaignSchedule()
        anchor = CheckpointAnchor('a' * 64, 40, 12, start.initial_lr)
        for changed in (replace(anchor, completed_updates=2), replace(anchor, applied_lr=.003)):
            with self.assertRaisesRegex(ValueError, 'warmup|parent applied LR'):
                schedule.fork_cooldown(start, changed, end_game=60)
        with self.assertRaisesRegex(ValueError, 'after the committed'):
            schedule.fork_cooldown(start, anchor, end_game=40)
        child = schedule.fork_cooldown(start, anchor, end_game=60)
        for window, updates in ((DeclaredGameWindow(40, 61, 70), 12), (DeclaredGameWindow(40, 60, 70), 13)):
            with self.assertRaisesRegex(ValueError, 'outside the finite cooldown lineage'):
                child.validate_next_window(start, window, updates)

    def test_existing_trainer_metadata_has_no_schedule_option(self):
        config = TrainConfig(seed=0)
        metadata = train_config_metadata(config)
        self.assertNotIn('campaign', metadata)
        self.assertNotIn('learning_rate_schedule', metadata)
        self.assertEqual(TrainConfig(**metadata), config)
