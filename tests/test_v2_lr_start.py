"""CPU proof for the opt-in V2 update clock and its two resume layers."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from gpu_engine import v2_fused
from pokezero import v2_trainer as v2, v5_training as v5
from pokezero.v2_lr_start import LRStart, scaled_peak_rate
from test_v2_trainer import configurations, store_of, varied_batch


POLICY = LRStart('sqrt', 8)
FIRST = v5.DeclaredGameWindow(0, 20, 40)
SECOND = v5.DeclaredGameWindow(20, 40, 40)


def fixture():
    model, loss, train = configurations(epochs=2, batch_size=4)
    train = replace(train, learning_rate=scaled_peak_rate(4, POLICY.scaling))
    config = v2.V2Config(optimizer='single', lr_start=POLICY)
    store = store_of(varied_batch(seed=713), model, loss, segments=(5, 4), store_id='lr-start-first')
    return model, loss, train, config, store


def resume_file(source, target):
    torch.set_num_threads(1)
    resumed = v2.load_checkpoint(source, enable_v5=True, expected_self_moves=True)
    report = v2.Trainer(resumed).train(fixture()[-1], window=FIRST)
    v2.save_checkpoint(target, resumed)
    return report


def tensor_bytes(value):
    # Compare persisted state independently of pickle's object alias table.
    if isinstance(value, torch.Tensor):
        return value.dtype, value.shape, value.stride(), value.reshape(-1).view(torch.uint8).numpy().tobytes()
    if isinstance(value, dict):
        return {key: tensor_bytes(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map(tensor_bytes, value))
    return value


@dataclass(frozen=True)
class PlanFixture:
    train: v5.TrainConfig

    def manifest(self):
        return asdict(self)


class LRStartTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def make_runtime(self):
        model, loss, train, config, store = fixture()
        rt = v2.create(model, loss, train, config, enable_v5=True, self_moves=True)
        return rt, store

    def test_peaks_and_zero_one_update_edges(self):
        self.assertEqual([scaled_peak_rate(4096, 'linear'), scaled_peak_rate(2048, 'linear')],
                         [4.72e-4, 2.36e-4])
        self.assertAlmostEqual(scaled_peak_rate(4096, 'sqrt'), 0.00016687720036002522)
        self.assertEqual(scaled_peak_rate(2048, 'sqrt'), 1.18e-4)
        for count in (0, 1):
            for completed in (0, 1, 100):
                self.assertEqual(LRStart('linear', count).rate(4.72e-4, completed), 4.72e-4)
        for value in (-1, False, 2.5):
            with self.assertRaises(ValueError):
                LRStart('sqrt', value)
        with self.assertRaises(ValueError):
            LRStart('square', 4)

    def test_partial_segments_two_epochs_and_next_window_use_one_global_clock(self):
        runtime, store = self.make_runtime()
        trainer = v2.Trainer(runtime)
        first = trainer.train(store, window=FIRST)
        self.assertTrue(first['complete'])
        self.assertEqual([x['owner_rows'] for x in first['updates']], [4, 1, 4, 4, 1, 4])
        self.assertEqual(first['lr_start'], dict(policy=asdict(POLICY), declared_window=asdict(FIRST),
            start_update=0, end_update=6, start_row=0, end_row=18))
        other = store_of(varied_batch(seed=717), runtime.model_config, runtime.loss_config,
            segments=(4, 5), store_id='lr-start-second')
        second = trainer.train(other, window=SECOND, max_updates=3)
        peak = runtime.train_config.learning_rate
        self.assertEqual([x['learning_rate'] for x in first['updates'] + second['updates']],
                         [peak * min(step / 8, 1.) for step in range(1, 10)])
        self.assertEqual(second['lr_start'], dict(policy=asdict(POLICY), declared_window=asdict(SECOND),
            start_update=6, end_update=9, start_row=18, end_row=27))

    def test_mid_warmup_resume_is_byte_exact_in_same_and_fresh_processes(self):
        runtime, store = self.make_runtime()
        trainer = v2.Trainer(runtime)
        trainer.train(store, window=FIRST, max_updates=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, full, same, fresh = (root / f'{name}.pt' for name in ('prefix', 'full', 'same', 'fresh'))
            v2.save_checkpoint(source, runtime)
            suffix = trainer.train(store, window=FIRST)
            v2.save_checkpoint(full, runtime)
            self.assertEqual(resume_file(source, same), suffix)
            child = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                    'resume-file', str(source), str(fresh)],
                                   check=True, text=True, capture_output=True, timeout=90)
            self.assertEqual(json.loads(child.stdout), suffix)
            expected = tensor_bytes(torch.load(full, weights_only=True))
            self.assertEqual(tensor_bytes(torch.load(same, weights_only=True)), expected)
            self.assertEqual(tensor_bytes(torch.load(fresh, weights_only=True)), expected)

    def test_resume_refuses_changed_parameters_context_and_semantic_corruption(self):
        runtime, store = self.make_runtime()
        trainer = v2.Trainer(runtime)
        with self.assertRaisesRegex(ValueError, 'declared-game'):
            trainer.train(store)
        trainer.train(store, window=FIRST, max_updates=2)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'base.pt'
            v2.save_checkpoint(source, runtime)
            for changed in (None, LRStart('linear', 8), LRStart('sqrt', 7)):
                config = replace(runtime.v2_config, lr_start=changed)
                with self.assertRaisesRegex(ValueError, 'original LR-start'):
                    v2.load_checkpoint(source, v2_config=config, enable_v5=True, expected_self_moves=True)
            with self.assertRaisesRegex(ValueError, 'require V2 resume'):
                v5.load_checkpoint(source, enable_v5=True, expected_self_moves=True)
            resumed = v2.load_checkpoint(source, enable_v5=True, expected_self_moves=True)
            with self.assertRaisesRegex(ValueError, 'original row store, order and declared window'):
                v2.Trainer(resumed).train(store, window=SECOND)
            cases = {
                'origin': ('missing its window origin', lambda p: p['v2'].pop('lr_start_origin')),
                'declaration': ('incomplete LR-start', lambda p: p['v2']['config'].pop('lr_start')),
                'cursor': ('cursor and counters differ', lambda p: p['v2']['cursor'].update(step=1)),
                'optimizer': ('optimizer rate differs', lambda p: p['optimizer_state']['param_groups'][0].update(lr=1.)),
            }
            for name, (message, corrupt) in cases.items():
                with self.subTest(name=name):
                    payload = torch.load(source, weights_only=True)
                    corrupt(payload)
                    payload['v2_sha256'] = v2._v2_digest(payload['v2'])
                    changed_path = source.with_name(f'{name}.pt')
                    torch.save(payload, changed_path)
                    with self.assertRaisesRegex(ValueError, message):
                        v2.load_checkpoint(changed_path, enable_v5=True, expected_self_moves=True)

    def test_direct_creation_refuses_a_conflicting_base_rate(self):
        model, loss, train, config, _ = fixture()
        for changed in (replace(train, learning_rate=2e-5),
                        replace(train, learning_rate_schedule='mit-thesis')):
            with self.assertRaisesRegex(ValueError, 'scaled peak and constant'):
                v2.create(model, loss, changed, config, enable_v5=True, self_moves=True)

    def test_fused_selector_preserves_the_original_declaration_and_legacy_branch(self):
        args = v2_fused.argument_parser().parse_args(['--reference-run', 'root', '--out', 'out'])
        self.assertIsNone(args.lr_start)
        self.assertIsNone(args.lr_warmup_updates)
        original = PlanFixture(v5.TrainConfig(seed=17, batch_size=4096,
            learning_rate=4.72e-4, learning_rate_schedule='mit-thesis'))
        context = dict(plan=original, declared={'declaration_sha256': 'original'})
        self.assertIs(v2_fused.select_learner_lr_start(context, None), original)
        self.assertEqual(context['declared'], {'declaration_sha256': 'original'})
        effective = v2_fused.select_learner_lr_start(context, POLICY)
        self.assertEqual(effective.train, replace(original.train, learning_rate=scaled_peak_rate(4096, 'sqrt'),
                                                learning_rate_schedule='constant'))
        self.assertEqual(context['declared']['declaration_sha256'], 'original')
        self.assertEqual(context['declared']['lr_start'], asdict(POLICY))

    def test_fused_parser_combines_lr_start_and_the_production_profile(self):
        args = v2_fused.parse_args([
            '--reference-run', 'root', '--out', 'out', '--profile', 'v2-production',
            '--lr-start', POLICY.scaling, '--lr-warmup-updates', str(POLICY.warmup_updates),
        ])
        self.assertEqual(args.lr_start, POLICY)
        self.assertTrue(args.freeze_gc_after_warmup)

    def test_campaign_parser_composes_with_lr_start_and_gc_profile(self):
        args = v2_fused.parse_args([
            '--reference-run', 'root', '--out', 'out', '--profile', 'v2-production',
            '--lr-start', POLICY.scaling, '--lr-warmup-updates', str(POLICY.warmup_updates),
            '--campaign-anneal', '--advance-campaign-step-at', '8',
        ])
        self.assertEqual(args.lr_start, POLICY)
        self.assertTrue(args.freeze_gc_after_warmup)
        self.assertTrue(args.campaign_anneal)
        self.assertEqual(args.advance_campaign_step_at, 8)

    def test_fused_committed_game_ledger_must_match_the_checkpoint(self):
        runtime, store = self.make_runtime()
        trained = v2.Trainer(runtime).train(store, window=FIRST)
        plan = SimpleNamespace(games=range(40), iterations=((SimpleNamespace(games=range(20)),),))
        report = dict(iteration=0, games=20, training=dict(steps=6, rows=18, updates=6,
            lr_start=trained['lr_start']))
        v2_fused._validate_lr_start_reports(runtime, [report], plan, asdict(POLICY))
        report['training']['lr_start']['end_row'] = 19
        with self.assertRaisesRegex(ValueError, 'report progress differs'):
            v2_fused._validate_lr_start_reports(runtime, [report], plan, asdict(POLICY))


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'resume-file':
        print(json.dumps(resume_file(Path(sys.argv[2]), Path(sys.argv[3]))))
    else:
        unittest.main()
