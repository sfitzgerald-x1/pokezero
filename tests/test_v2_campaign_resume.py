"""Checkpoint and fused-ledger tests with the reviewed, real LR-start provider."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from gpu_engine import v2_fused
from pokezero import v2_trainer as v2, v5_training as v5
from pokezero.v2_campaign_schedule import BACKSTOP_STEPS, CampaignSchedule
from pokezero.v2_lr_start import LRStart, scaled_peak_rate
from test_v2_lr_start import fixture, tensor_bytes
from test_v2_trainer import store_of, varied_batch


class CampaignResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model, self.loss, self.train, self.config, self.first = fixture()

    def create(self, schedule, warmup=8):
        config = replace(self.config, lr_start=LRStart('sqrt', warmup), campaign=schedule)
        return v2.create(self.model, self.loss, self.train, config, enable_v5=True, self_moves=True)

    def store(self, slot):
        return store_of(varied_batch(seed=730 + slot), self.model, self.loss,
                        segments=(5, 4), store_id=f'campaign-window-{slot}')

    def restore(self, path, **options):
        return v2.load_checkpoint(path, enable_v5=True, expected_self_moves=True, **options)

    def assert_same_state(self, full, resumed):
        self.assertEqual((full.steps, full.rows, full.cursor, full.window, full.lr_start_origin, full.v2_config),
                         (resumed.steps, resumed.rows, resumed.cursor, resumed.window, resumed.lr_start_origin,
                          resumed.v2_config))
        for left, right in zip((full.model.state_dict(), full.optimizer.state_dict(), full.rng),
                               (resumed.model.state_dict(), resumed.optimizer.state_dict(), resumed.rng)):
            self.assertEqual(tensor_bytes(left), tensor_bytes(right))

    def test_exact_partial_resume_through_all_schedule_phases(self):
        schedule = CampaignSchedule((40, 60), final_cooldown=(80, 100))
        full, split = self.create(schedule), self.create(schedule)
        with TemporaryDirectory() as directory:
            for slot, count in enumerate((3, 1, 0, 2, 4)):
                window = v5.DeclaredGameWindow(20 * slot, 20 * (slot + 1), 100)
                store = self.store(slot)
                expected = v2.Trainer(full).train(store, window=window)
                prefix = v2.Trainer(split).train(store, window=window, max_updates=count)
                path = Path(directory) / f'phase-{slot}.pt'
                v2.save_checkpoint(path, split)
                split = self.restore(path)
                suffix = v2.Trainer(split).train(store, window=window)
                self.assertEqual(prefix['updates'] + suffix['updates'], expected['updates'])
                self.assert_same_state(full, split)
                peak = scaled_peak_rate(self.train.batch_size, 'sqrt')
                rates = [update['learning_rate'] for update in expected['updates']]
                first = (peak / 8, 7 * peak / 8, .3 * peak, .3 ** 2 * peak, .3 ** 2 * peak)
                self.assertAlmostEqual(rates[0], first[slot])
                if slot == 4:
                    self.assertGreater(rates[0], rates[-1])

    def test_fused_ledger_accepts_one_advancement_at_the_committed_boundary(self):
        original = CampaignSchedule()
        runtime = self.create(original, warmup=3)
        reports = []
        for slot in range(2):
            window = v5.DeclaredGameWindow(20 * slot, 20 * (slot + 1), 40)
            if slot:
                with self.assertRaisesRegex(ValueError, 'current completed game boundary'):
                    v2.advance_campaign_step(runtime, 19)
                advanced = v2.advance_campaign_step(runtime, 20)
                self.assertEqual(advanced.step_down_games, (20, BACKSTOP_STEPS[1]))
                self.assertEqual(v2.advance_campaign_step(runtime, 20), advanced)
            report = v2.Trainer(runtime).train(self.store(slot), window=window)
            reports.append(dict(iteration=slot, games=20, training=dict(
                steps=report['steps'], rows=report['rows'], updates=len(report['updates']), lr_start=report['lr_start'],
                campaign_schedule=asdict(runtime.v2_config.campaign))))
        with self.assertRaisesRegex(ValueError, 'current completed game boundary'):
            v2.advance_campaign_step(runtime, 20)
        plan = SimpleNamespace(games=range(40), iterations=((SimpleNamespace(games=range(20)),),) * 2)
        persisted = json.loads(json.dumps(reports))
        v2_fused._validate_lr_start_reports(runtime, persisted, plan, asdict(runtime.v2_config.lr_start))
        persisted[0]['training']['campaign_schedule'] = persisted[1]['training']['campaign_schedule']
        with self.assertRaisesRegex(ValueError, 'eligible committed boundary'):
            v2_fused._validate_lr_start_reports(runtime, persisted, plan, asdict(runtime.v2_config.lr_start))

    def test_verified_child_resumes_and_cannot_change_the_parent(self):
        runtime = self.create(CampaignSchedule(), warmup=3)
        first = v5.DeclaredGameWindow(0, 20, 40)
        next_window = v5.DeclaredGameWindow(20, 40, 40)
        v2.Trainer(runtime).train(self.first, window=first)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / 'parent.pt'
            receipt = v2.save_checkpoint(parent, runtime)
            original = parent.read_bytes()
            opts = dict(parent_checkpoint_sha256=receipt['sha256'], start_game=20, end_game=40,
                        enable_v5=True, expected_self_moves=True)
            full = v2.load_campaign_cooldown(parent, **opts)
            self.assertEqual(full.v2_config.campaign.cooldown.parent.sha256, receipt['sha256'])
            with self.assertRaisesRegex(ValueError, 'original campaign schedule'):
                self.restore(parent, v2_config=full.v2_config)
            origin = root / 'child-origin.pt'
            v2.save_checkpoint(origin, full)
            partial = self.restore(origin)
            expected = v2.Trainer(full).train(self.store(1), window=next_window)
            prefix = v2.Trainer(partial).train(self.store(1), window=next_window, max_updates=2)
            fork = root / 'child-partial.pt'
            v2.save_checkpoint(fork, partial)
            partial = self.restore(fork)
            suffix = v2.Trainer(partial).train(self.store(1), window=next_window)
            self.assertEqual(prefix['updates'] + suffix['updates'], expected['updates'])
            self.assert_same_state(full, partial)
            self.assertEqual(parent.read_bytes(), original)

    def test_campaign_requires_warmup_and_flags_off_serialization_stays_unextended(self):
        with self.assertRaisesRegex(ValueError, 'positive LR-start warmup'):
            replace(self.config, lr_start=LRStart('sqrt', 0), campaign=CampaignSchedule())
        self.assertNotIn('campaign', v2._v2_section(self.create(None), [])['config'])
        runtime = v2.create(self.model, self.loss, self.train, enable_v5=True, self_moves=True)
        config = v2._v2_section(runtime, [])['config']
        self.assertNotIn('campaign', config)
        self.assertNotIn('lr_start', config)
