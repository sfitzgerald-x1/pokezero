"""Fail-closed frozen-stat checks do not require Torch or a GPU."""
import math
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpu_engine.v2_profile_learner import annotated, compare_statistics, counter_delta


class StatisticTests(unittest.TestCase):
    def setUp(self):
        self.expected = dict(optimizer_step=1, owner_rows=128, epoch=0, epoch_step=0,
            learning_rate_epoch=1, learning_rate=.0003, policy_loss=1., entropy=.5,
            policy_loss_count=128, gradient_norm=2.)

    def compare(self, **changes):
        return compare_statistics([self.expected], [dict(self.expected, **changes)])

    def test_relative_and_absolute_bounds(self):
        self.assertTrue(self.compare(policy_loss=1.00021)['passed'])
        self.assertFalse(self.compare(policy_loss=1.00023)['passed'])
        self.expected['policy_loss'] = 0.
        self.assertTrue(self.compare(policy_loss=1.99e-5)['passed'])
        self.assertFalse(self.compare(policy_loss=2.01e-5)['passed'])

    def test_accounting_counts_and_rate_are_exact(self):
        for name in ('owner_rows', 'epoch_step', 'policy_loss_count', 'learning_rate'):
            with self.subTest(name=name):
                self.assertFalse(self.compare(**{name:self.expected[name]+1e-8})['passed'])

    def test_missing_fields_and_updates_fail(self):
        self.assertFalse(compare_statistics([], [])['passed'])
        self.assertFalse(compare_statistics([self.expected], [])['passed'])
        actual = dict(self.expected)
        actual.pop('entropy')
        self.assertFalse(compare_statistics([self.expected], [actual])['passed'])

    def test_nonfinite_or_nonnumeric_metrics_fail(self):
        for value in (math.nan, math.inf, -math.inf, True, '1'):
            with self.subTest(value=value):
                self.assertFalse(self.compare(policy_loss=value)['passed'])

    def test_signed_near_zero_bound(self):
        self.expected['policy_loss'] = -1.
        self.assertTrue(self.compare(policy_loss=-1.00021)['passed'])
        self.assertFalse(self.compare(policy_loss=-1.00023)['passed'])

    def test_compile_counter_delta_keeps_new_groups(self):
        self.assertEqual(counter_delta({'stats':{'unique_graphs':2}},
            {'stats':{'unique_graphs':3}, 'graph_break':{'reason':1}}),
            {'stats':{'unique_graphs':1}, 'graph_break':{'reason':1}})


class AnnotationTests(unittest.TestCase):
    def fixture(self):
        self.labels = []

        @contextmanager
        def region(name):
            self.labels.append(name)
            yield

        def compute(value, *, add=0):
            return value+add

        torch = SimpleNamespace(profiler=SimpleNamespace(record_function=region),
            autograd=SimpleNamespace(backward=compute),
            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=compute)))
        optimizer = SimpleNamespace(step=compute)
        trainer = SimpleNamespace(_step=compute, _sum=compute, runtime=SimpleNamespace(optimizer=optimizer))
        return torch, trainer, compute

    def test_annotations_preserve_results_arguments_and_restore_methods(self):
        torch, trainer, original = self.fixture()
        with patch.dict('sys.modules', torch=torch):
            with annotated(trainer):
                self.assertEqual(trainer._step(3, add=4), 7)
                self.assertEqual(torch.autograd.backward(5, add=6), 11)
                self.assertEqual(trainer.runtime.optimizer.step(7, add=8), 15)
            self.assertIs(trainer._step, original)
            self.assertIs(torch.autograd.backward, original)
            self.assertIs(torch.nn.utils.clip_grad_norm_, original)
            self.assertIs(trainer.runtime.optimizer.step, original)
        self.assertEqual(self.labels, ['v2/update', 'v2/backward', 'v2/optimizer'])

    def test_exception_restores_all_wrapped_methods(self):
        torch, trainer, original = self.fixture()
        with patch.dict('sys.modules', torch=torch):
            with self.assertRaisesRegex(RuntimeError, 'failed update'):
                with annotated(trainer):
                    raise RuntimeError('failed update')
            self.assertIs(trainer._step, original)
            self.assertIs(torch.autograd.backward, original)
            self.assertIs(trainer.runtime.optimizer.step, original)


if __name__ == '__main__':
    unittest.main()
