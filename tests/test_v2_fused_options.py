"""The fused production entry keeps learner acceleration options explicit."""
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

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
    def test_adopted_prefix_preserves_old_report_horizon_and_checks_successor(self):
        from pokezero.v2_campaign_schedule import CampaignSchedule

        plan = SimpleNamespace(iterations=[(SimpleNamespace(games=range(20_000)),)] * 2)
        policy = dict(scaling='sqrt', warmup_updates=8000)
        campaign = CampaignSchedule()
        prefix = dict(rounds=1, previous_total_games=5_000_000)
        reports = []
        for index, horizon in enumerate((5_000_000, 40_000)):
            window = dict(start_game=index * 20_000, end_game=(index + 1) * 20_000, total_games=horizon)
            lr_start = dict(policy=policy, declared_window=window, start_update=index * 10,
                            end_update=(index + 1) * 10, start_row=index * 100, end_row=(index + 1) * 100)
            reports.append(dict(iteration=index, games=20_000, training=dict(updates=10, steps=(index + 1) * 10,
                                rows=(index + 1) * 100, lr_start=lr_start, campaign_schedule=asdict(campaign))))
        runtime = SimpleNamespace(v2_config=SimpleNamespace(campaign=campaign), cursor=dict(complete=True),
                                  window=v5_training.DeclaredGameWindow(**window), lr_start_origin=(10, 100),
                                  steps=20, rows=200)
        v2_fused._validate_lr_start_reports(runtime, reports, plan, policy, adopted_prefix=prefix)
        reports[0]['training']['lr_start']['declared_window']['total_games'] = 40_000
        with self.assertRaisesRegex(ValueError, 'report progress differs'):
            v2_fused._validate_lr_start_reports(runtime, reports, plan, policy, adopted_prefix=prefix)

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
            parent = v2_fused.cooldown_parent(root, last.checkpoint_sha256, last.model_sha256, policy, 400_000)
            self.assertEqual(parent['origin'], (24000, 98_000_000, 2_000_000))
            self.assertEqual(parent['history'][0].iteration, 33)
            self.assertEqual(parent['snapshot'], last)
            with self.assertRaisesRegex(ValueError, 'declared campaign decision boundary'):
                v2_fused.cooldown_parent(root, snapshots[-2].checkpoint_sha256, last.model_sha256, policy, 400_000)
            self.assertEqual(manifest.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'distinct, non-nested'):
                v2_fused.main(['--config', 'unused', '--reference-run', 'unused', '--out', str(root / 'child'),
                               '--lr-start', 'sqrt', '--lr-warmup-updates', '8000', '--campaign-anneal',
                               '--cooldown-child', str(root), '--cooldown-parent-sha256', last.checkpoint_sha256])
            self.assertEqual(list(root.iterdir()), [manifest])

    def test_longer_cooldown_binds_parent_length_and_writer_mount(self):
        from pokezero.v2_campaign_schedule import CampaignSchedule

        with TemporaryDirectory() as directory:
            root = Path(directory)
            policy = dict(scaling='sqrt', warmup_updates=8000)
            parent = v5_run.Snapshot(674, 'checkpoint.pt', 'a' * 64, 'actor.npz', 'b' * 64, 'c' * 64)
            initial = v5_run.Snapshot(-1, 'initial.pt', 'd' * 64, 'initial.npz', 'e' * 64, parent.model_sha256)
            reports = [dict(iteration=i, training=dict(steps=240 * (i + 1), rows=980_000 * (i + 1),
                        lr_start=dict(declared_window=dict(end_game=20_000 * (i + 1))),
                        campaign_schedule=asdict(CampaignSchedule()))) for i in range(675)]
            state = dict(snapshots=[asdict(initial), asdict(parent)], lr_start=policy, row_order='file', reports=reports)
            receipt = dict(path='/out/checkpoint.pt.encoding.json', sha256='f' * 64)
            state['checkpoint_encoding_receipts'] = {parent.checkpoint_sha256: receipt}
            (root / 'state.json').write_text(v5_run._json(state))
            bound = v2_fused.cooldown_parent(root, parent.checkpoint_sha256, parent.model_sha256, policy, 1_500_000)
            self.assertEqual(bound['origin'][2], 13_500_000)
            self.assertEqual(bound['proof'], dict(receipt, path=str(root.resolve() / 'checkpoint.pt.encoding.json')))
            local = root / 'checkpoint.pt.encoding.json'
            local.write_text('{}')
            alias = root / 'writer' / local.name
            alias.parent.mkdir()
            alias.hardlink_to(local)
            receipt['path'] = str(alias)
            (root / 'state.json').write_text(v5_run._json(state))
            mounted = v2_fused.cooldown_parent(root, parent.checkpoint_sha256, parent.model_sha256, policy, 1_500_000)
            self.assertEqual(mounted['proof'], bound['proof'])
            with self.assertRaisesRegex(ValueError, 'cooldown duration differs'):
                v2_fused.cooldown_parent(root, parent.checkpoint_sha256, parent.model_sha256, policy, 400_000)
            receipt['path'] = '/out/other.pt.encoding.json'
            (root / 'state.json').write_text(v5_run._json(state))
            with self.assertRaisesRegex(ValueError, 'receipt does not belong'):
                v2_fused.cooldown_parent(root, parent.checkpoint_sha256, parent.model_sha256, policy, 1_500_000)

    def test_cooldown_checks_later_commits_and_reserved_s0_addresses(self):
        from pokezero.v2_campaign_schedule import CampaignSchedule
        from pokezero.v5_deferred_evaluation import evaluation_declaration_path

        def batch(start):
            return [dict(games=dict(game_count=1, stream_start=start, identity_seed_start=start, team_start=2*start),
                         constructor_seed=start, sampling_seed=start, gender_root_start=start)]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            policy = dict(scaling='sqrt', warmup_updates=1)
            initial = v5_run.Snapshot(-1, 'initial.pt', 'd' * 64, 'initial.npz', 'e' * 64, 'c' * 64)
            parent = v5_run.Snapshot(99, 'parent.pt', 'a' * 64, 'parent.npz', 'b' * 64, initial.model_sha256)
            later = v5_run.Snapshot(100, 'later.pt', 'f' * 64, 'later.npz', '0' * 64, initial.model_sha256)
            original = dict(iterations=[batch(i) for i in range(250)])
            successor = dict(iterations=original['iterations'][:100] + [batch(1000)])
            digest = v5_run._sha(successor)
            state = dict(snapshots=[asdict(item) for item in (initial, parent, later)], lr_start=policy,
                         row_order='file', declaration_sha256=digest,
                         reports=[dict(iteration=i, training=dict(steps=i+1, rows=i+1,
                                      lr_start=dict(declared_window=dict(end_game=20_000*(i+1))),
                                      campaign_schedule=asdict(CampaignSchedule()))) for i in range(101)])
            state_path = root / 'state.json'
            state_path.write_text(v5_run._json(state))
            legacy = root / 'legacy-effective.json'
            legacy.write_text(v5_run._json(successor))

            def resolve(start, fallback=None):
                return v2_fused.cooldown_parent(root, parent.checkpoint_sha256, parent.model_sha256, policy, 400_000,
                    child_declaration=dict(iterations=[batch(start)]), parent_declaration_path=fallback)

            self.assertEqual(resolve(2000, legacy)['snapshot'], parent)
            with self.assertRaisesRegex(ValueError, 'reuses a committed parent'):
                resolve(1000, legacy)
            with self.assertRaisesRegex(ValueError, 'declaration differs'):
                legacy.write_text(v5_run._json(original))
                resolve(2000, legacy)

            (root / 'declaration.json').write_text(v5_run._json(original))
            version = evaluation_declaration_path(root, digest)
            version.parent.mkdir(parents=True)
            version.write_text(v5_run._json(successor))
            state['adopted_prefix'] = dict(rounds=100, previous_declaration_sha256=v5_run._sha(original))
            state_path.write_text(v5_run._json(state))
            with self.assertRaisesRegex(ValueError, 'reuses a committed parent'):
                resolve(110)
            self.assertEqual(resolve(2000)['snapshot'], parent)


if __name__ == '__main__':
    unittest.main()
