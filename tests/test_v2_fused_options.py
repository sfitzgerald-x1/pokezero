"""The fused production entry keeps learner acceleration options explicit."""
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from gpu_engine import v2_fused
from gpu_engine.v2_fused import argument_parser, parse_args, select_learner_precision
from pokezero import v5_run, v5_training


class LearnerOptionsTests(unittest.TestCase):
    def parse(self, *options):
        return argument_parser().parse_args([
            '--config', 'diagnostic.json', '--reference-run', 'reference', '--out', 'output', *options])

    def test_existing_commands_leave_both_options_disabled(self):
        args = self.parse()
        self.assertFalse(args.native_embedding_backward)
        self.assertFalse(args.fallback_random)
        self.assertEqual(args.mode, 'cudagraphs')

    def test_native_backward_does_not_enable_random_fallback(self):
        args = self.parse('--native-embedding-backward')
        self.assertTrue(args.native_embedding_backward)
        self.assertFalse(args.fallback_random)

    def test_random_fallback_can_be_selected_independently_for_controls(self):
        for native in (False, True):
            with self.subTest(native=native):
                args = self.parse('--fallback-random', *(['--native-embedding-backward'] if native else []))
                self.assertEqual(args.native_embedding_backward, native)
                self.assertTrue(args.fallback_random)


class FreezeOptionsTests(unittest.TestCase):
    def parse(self, *options):
        return parse_args(['--reference-run', 'reference', '--out', 'output', *options])

    def test_freeze_defaults_to_production_profile_only(self):
        self.assertFalse(self.parse().freeze_gc_after_warmup)
        self.assertTrue(self.parse('--profile', 'v2-production').freeze_gc_after_warmup)

    def test_explicit_gc_options_override_profile_default(self):
        self.assertTrue(self.parse('--freeze-gc-after-warmup').freeze_gc_after_warmup)
        self.assertFalse(self.parse('--profile', 'v2-production', '--no-freeze-gc-after-warmup').freeze_gc_after_warmup)


@dataclass(frozen=True)
class PlanFixture:
    train: v5_training.TrainConfig

    def manifest(self):
        return asdict(self)


class LearnerPrecisionTests(unittest.TestCase):
    def fixture(self):
        return dict(plan=PlanFixture(v5_training.TrainConfig(seed=23, precision='bfloat16')),
            declared=dict(declaration_sha256='original', recipe='R5'))

    def test_omitted_precision_preserves_original_plan_and_binding(self):
        context = self.fixture(); original = context['plan']; declared = dict(context['declared'])
        args = argument_parser().parse_args(['--reference-run', 'reference', '--out', 'output'])
        self.assertIsNone(args.learner_precision)
        self.assertIs(select_learner_precision(context, args.learner_precision), original)
        self.assertEqual(context['declared'], declared)

    def test_explicit_FP32_records_effective_plan_and_preserves_initial_train(self):
        context = self.fixture(); initial_train = context['plan'].train
        args = argument_parser().parse_args(['--reference-run', 'reference', '--out', 'output',
            '--learner-precision', 'float32'])
        plan = select_learner_precision(context, args.learner_precision)
        self.assertEqual(plan.train.precision, 'float32')
        self.assertEqual(initial_train.precision, 'bfloat16')
        self.assertEqual(context['declared']['recipe'], 'R5')
        self.assertEqual(context['declared']['declaration_sha256'], 'original')
        self.assertEqual(context['declared']['learner_precision'], 'float32')
        self.assertEqual(context['declared']['effective_plan_sha256'], v5_run._sha(plan.manifest()))

    def test_invalid_precision_refused_without_mutating_context(self):
        context = self.fixture(); original = context['plan']; declared = dict(context['declared'])
        with self.assertRaises(ValueError):
            select_learner_precision(context, 'float16')
        self.assertIs(context['plan'], original)
        self.assertEqual(context['declared'], declared)


class ResumeOptionsTests(unittest.TestCase):
    def test_require_resume_refuses_missing_out_before_initialization(self):
        with TemporaryDirectory() as directory:
            out = Path(directory) / 'new-output'
            with self.assertRaisesRegex(ValueError, 'needs a committed state.json'):
                v2_fused.main(['--config', 'unused', '--reference-run', 'unused', '--out', str(out), '--require-resume'])
            self.assertFalse(out.exists())

    def test_cooldown_resolves_only_committed_parent_and_does_not_touch_it(self):
        from pokezero.v2_campaign_schedule import CampaignSchedule

        with TemporaryDirectory() as directory:
            root = Path(directory)
            policy = dict(scaling='sqrt', warmup_updates=8000)
            snapshots = [v5_run.Snapshot(i, f'ckpt-{i}', f'{i + 2:064x}', f'actor-{i}', f'{i + 3:064x}', 'a'*64)
                         for i in range(-1, 100)]
            state = dict(snapshots=[asdict(snapshot) for snapshot in snapshots], lr_start=policy, row_order='file',
                         reports=[dict(iteration=i, training=dict(steps=240 * (i + 1), rows=980_000 * (i + 1),
                             lr_start=dict(declared_window=dict(end_game=20_000 * (i + 1))),
                             campaign_schedule=asdict(CampaignSchedule()))) for i in range(100)])
            manifest = root / 'state.json'
            original = v5_run._json(state).encode()
            manifest.write_bytes(original)
            last = snapshots[-1]
            parent = v2_fused.cooldown_parent(root, last.checkpoint_sha256, last.model_sha256, policy)
            self.assertEqual(parent['origin'], (24000, 98_000_000, 2_000_000))
            self.assertEqual(parent['history'][0].iteration, 33)
            self.assertEqual(parent['snapshot'], last)
            with self.assertRaisesRegex(ValueError, 'committed 2M or 5M'):
                v2_fused.cooldown_parent(root, snapshots[-2].checkpoint_sha256, last.model_sha256, policy)
            self.assertEqual(manifest.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'distinct, non-nested'):
                v2_fused.main(['--config', 'unused', '--reference-run', 'unused', '--out', str(root / 'child'),
                               '--lr-start', 'sqrt', '--lr-warmup-updates', '8000', '--campaign-anneal',
                               '--cooldown-child', str(root), '--cooldown-parent-sha256', last.checkpoint_sha256])
            self.assertEqual(list(root.iterdir()), [manifest])


if __name__ == '__main__':
    unittest.main()
