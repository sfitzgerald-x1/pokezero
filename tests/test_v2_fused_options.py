"""The fused production entry keeps learner acceleration options explicit."""
import unittest
from dataclasses import asdict, dataclass

from gpu_engine.v2_fused import argument_parser, select_learner_precision
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


if __name__ == '__main__':
    unittest.main()
