"""Whole-game receipt gates retain cancellations without treating them as work."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest

from wider_search_comparison import (
    register, staged_conditioning_enabled, validate_reference_measurement,
)


def fixture():
    good = dict(status='ROOT_VALIDATED', released=True)
    cancelled = dict(status='DEADLINE_CANCELLED', sampling_diagnostic=dict(
        schema='pokezero.world-sampling-deadline.v1', checked_at=11., deadline_at=10.,
        accepted_world=False, backed_up=False))
    batch = NS(world_draws=2, trajectories=1, transitions=3, deadline_exhausted=True)
    return NS(worker_pids=tuple(range(20)), result=NS(world_draws=2, trajectories=1, transitions=3),
        worker_receipts=[dict(worker=0, batch=batch, evidence=dict(draws=[good, cancelled]))])


class ReferenceDeadlineReceiptTests(unittest.TestCase):
    def test_completed_work_and_cancelled_rng_attempt_both_retained(self):
        measured = fixture()
        validate_reference_measurement(measured)
        self.assertEqual(len(measured.worker_receipts[0]['evidence']['draws']), 2)

    def test_cancelled_only_batch_is_allowed_when_another_batch_completed(self):
        measured = fixture()
        measured.worker_receipts.append(dict(worker=1,
            batch=NS(world_draws=1, trajectories=0, transitions=0, deadline_exhausted=True),
            evidence=dict(draws=[deepcopy(measured.worker_receipts[0]['evidence']['draws'][1])])) )
        measured.result.world_draws = 3
        validate_reference_measurement(measured)

    def test_cancellation_cannot_be_counted_as_completed_or_transition_work(self):
        for mutate in ('trajectories', 'transitions'):
            measured = fixture()
            measured.worker_receipts[0]['evidence']['draws'] = [
                deepcopy(measured.worker_receipts[0]['evidence']['draws'][1])]
            batch = measured.worker_receipts[0]['batch']
            batch.world_draws = 1
            batch.trajectories = 0
            batch.transitions = 0
            setattr(batch, mutate, 1)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(RuntimeError, 'counted as work'):
                validate_reference_measurement(measured)

    def test_missing_or_false_clock_and_zero_backup_proof_refuse(self):
        for key, value in (('accepted_world', True), ('backed_up', True), ('checked_at', 9.),
                           ('checked_at', float('nan')), ('deadline_at', None),
                           ('schema', 'unknown')):
            measured = fixture()
            measured.worker_receipts[0]['evidence']['draws'][1]['sampling_diagnostic'][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(RuntimeError, 'zero-backup'):
                validate_reference_measurement(measured)

    def test_unreleased_refused_or_uncounted_world_refuses(self):
        for status, release in (('REFUSED', True), ('STARTED', False), ('ROOT_VALIDATED', False)):
            measured = fixture()
            measured.worker_receipts[0]['evidence']['draws'][0].update(status=status, released=release)
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, 'unreleased'):
                validate_reference_measurement(measured)
        measured = fixture()
        measured.result.world_draws = 3
        with self.assertRaisesRegex(RuntimeError, 'aggregate'):
            validate_reference_measurement(measured)

    def test_zero_new_work_or_missing_worker_refuses_even_with_old_statistics(self):
        for field in ('work', 'workers'):
            measured = fixture()
            if field == 'work':
                measured.result.trajectories = 0
            else:
                measured.worker_pids = tuple(range(19))
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'NEW complete'):
                validate_reference_measurement(measured)

    def test_kernel_is_default_off_and_registered_boolean_only(self):
        self.assertFalse(staged_conditioning_enabled({}))
        self.assertTrue(staged_conditioning_enabled(dict(reference=dict(staged_substitute_conditioning=True))))
        for value in (1, 'true', None):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'registered boolean'):
                staged_conditioning_enabled(dict(reference=dict(staged_substitute_conditioning=value)))

    def test_opt_in_does_not_authorize_unqualified_confirmation(self):
        with self.assertRaisesRegex(RuntimeError, 'fresh technical qualification'):
            register(NS(staged_substitute_conditioning=True, qualification=False))


if __name__ == '__main__':
    unittest.main()
