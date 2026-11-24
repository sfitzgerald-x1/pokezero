import hashlib
import json
import shlex
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from gpu_engine.v2_fused import parse_args
from pokezero.v2_fused_state import verify_segment_start
from pokezero.v5_deferred_evaluation import MilestoneDispatch
from scripts.supervise_v2_campaign import ADMISSION_SECONDS, build_launch, disposition, segment_order, verify_attempt_pointer


class CampaignSupervisorTests(unittest.TestCase):
    def test_worker_start_accepts_only_admitted_or_same_job_progress(self):
        job = 'v2a-012abc-gpu:2'
        first = (0, '-')
        state = dict(round_ledger=dict(completed_rounds=12), segment_job='v2a-012abc-gpu:1')
        pointer = json.dumps(state).encode()
        retry = (12, hashlib.sha256(pointer).hexdigest())
        check = 'startup pointer differs'
        verify_segment_start(None, None, first, job)
        verify_segment_start(state, pointer, retry, job)
        with self.assertRaisesRegex(ValueError, check):
            verify_segment_start(state, pointer, first, job)
        with self.assertRaisesRegex(ValueError, check):
            verify_segment_start(state, pointer + b'\n', retry, job)
        with self.assertRaisesRegex(ValueError, check):
            verify_segment_start(None, None, retry, job)
        advanced = dict(state, round_ledger=dict(completed_rounds=13), segment_job=job)
        verify_segment_start(advanced, json.dumps(advanced).encode(), retry, job)
        with self.assertRaisesRegex(ValueError, check):
            verify_segment_start(advanced, json.dumps(advanced).encode(), retry, 'v2a-012abc-gpu:3')

    def test_admission_rejects_a_foreign_child_or_a_rolled_back_main(self):
        state = dict(round_ledger=dict(completed_rounds=150, latest_snapshot=dict(iteration=149)))
        pointer = json.dumps(state)
        digest = hashlib.sha256(pointer.encode()).hexdigest()
        verify_attempt_pointer('output', '', None, None)
        with self.assertRaisesRegex(ValueError, 'first admission'):
            verify_attempt_pointer('output', pointer, state, None)
        attempt = dict(committed_round=150, state_sha256=digest)
        self.assertEqual(verify_attempt_pointer('output', pointer, state, attempt), (150, digest))
        with self.assertRaisesRegex(ValueError, 'prior owned attempt'):
            verify_attempt_pointer('output', pointer + '\n', state, attempt)
        with self.assertRaisesRegex(ValueError, 'prior owned attempt'):
            verify_attempt_pointer('output', pointer, state, dict(attempt, committed_round=151), restore={})

    def test_segment_requires_durable_milestone_publication_after_retry(self):
        dispatch = MilestoneDispatch.__new__(MilestoneDispatch)
        dispatch.milestones = {49: 1_000_000}
        snapshot = SimpleNamespace(iteration=49)
        report_error = Mock()
        dispatch.publish_committed = Mock()
        dispatch.request_missing = Mock(side_effect=(True, False))
        dispatch.finish_committed_publications([snapshot], report_error=report_error)
        dispatch.publish_committed.assert_called_once_with(snapshot, report_error=report_error)
        dispatch.request_missing = Mock(return_value=True)
        with self.assertRaisesRegex(ValueError, 'milestone 1000000 publication remains pending'):
            dispatch.finish_committed_publications([snapshot], report_error=report_error)

    def test_main_stops_interleave_only_committed_cooldown_parents(self):
        main = {'output': 'v2a-012abc-gpu', 'stops': [1250, 2500, 3750, 5000]}
        late = {'output': 'v2a-111111-gpu', 'parent_round': 1350, 'stop': 150}
        early = {'output': 'v2a-222222-gpu', 'parent_round': 450, 'stop': 50}
        order = segment_order(main, [late, early])
        self.assertEqual([item['stop'] for item in order], [1250, 50, 2500, 150, 3750, 5000])

    def test_requeue_does_not_retry_and_terminal_requires_no_writer(self):
        attempt = {'submitted_at': '1970-01-01T00:00:00Z', 'ever_admitted': True}
        observation = {'admitted': False, 'live': False, 'complete': False, 'failed': False}
        at = ADMISSION_SECONDS + 1
        self.assertIsNone(disposition(observation, attempt, 120, 1250, at=at))
        failed_with_writer = dict(observation, failed=True, live=True)
        self.assertIsNone(disposition(failed_with_writer, attempt, 120, 1250, at=at))
        failed = dict(observation, failed=True)
        self.assertEqual(disposition(failed, attempt, 120, 1250, at=at), 'RETRY')
        self.assertEqual(disposition(failed, attempt, 1250, 1250, at=at), 'RETRY')
        self.assertEqual(disposition(failed, attempt, 1250, 1250, at=at, terminal_status='COMPLETE'), 'ADVANCE')
        self.assertEqual(disposition(failed, attempt, 1250, 1250, at=at, terminal_status='STOPPED'), 'SIGNED_STOP')
        self.assertEqual(disposition(dict(observation, complete=True), attempt, 120, 1250, at=at),
                         'STOPPED_SHORT')

    def test_pending_admission_expires_without_replacing_an_unknown_job(self):
        attempt = {'submitted_at': '1970-01-01T00:00:00Z'}
        pending = {'admitted': False, 'live': True, 'complete': False, 'failed': False}
        self.assertIsNone(disposition(pending, attempt, 100, 1250, at=10))
        self.assertEqual(disposition(pending, attempt, 100, 1250, at=ADMISSION_SECONDS),
                         'ADMISSION_EXPIRED')
        self.assertEqual(disposition(None, attempt, 100, 1250, at=ADMISSION_SECONDS), 'AMBIGUOUS')

    def test_main_adopts_and_child_reuses_its_exact_parent_checkpoint(self):
        main = {'output': 'v2a-012abc-gpu', 'stop': 1250, 'declaration': {'path': '/out/i', 'sha256': 'd'},
                'restore': {'milestone_sha256s': {'1000000': 'c'*64, '2000000': 'd'*64}}}
        packet = dict(main=main, launcher={'path': 'launch.py'}, checkout='/tree', reference_run='/reference',
                      previous_runtime='/previous', runtime={'path': '/current', 'sha256': 'r'},
                      resilience={'path': '/support', 'phase_observer_sha256': 'p', 'signal_driver_sha256': 's'},
                      cpu=32, memory='896Gi', stop_key_reference='env=existing:key')
        attempt = {'number': 2, 'before_round': 100, 'before_state_sha256': 'b'*64, 'cap_seconds': 3600}
        name, argv = build_launch(packet, main, attempt, None)
        self.assertEqual(name, 'scott-v2a-012abc-gpu-r2')
        self.assertEqual(argv[argv.index('--resume-run')+1], main['output'])
        entry = shlex.split(argv[-1])
        self.assertEqual(entry[entry.index('--adopt-2m-prefix-runtime')+1], '/previous')
        options = parse_args([*entry[entry.index('--out'):], '--require-resume'])
        self.assertEqual(options.segment_start, (100, 'b'*64, 2))
        self.assertEqual(options.adopted_milestone_sha256s, ['c'*64, 'd'*64])

        child = dict(main, output='v2a-111111-gpu', parent_round=450, stop=50)
        name, argv = build_launch(packet, child, dict(attempt, before_round=0, before_state_sha256='-'), 'a'*64)
        self.assertEqual(name, 'scott-v2a-111111-gpu')
        self.assertNotIn('--require-fresh', argv)
        self.assertNotIn('--resume-run', argv)
        self.assertEqual(argv[argv.index('--resilient-support')+1], '/support')
        entry = shlex.split(argv[-1])
        self.assertEqual(entry[entry.index('--cooldown-parent-sha256')+1], 'a'*64)
        self.assertEqual(parse_args(entry[entry.index('--out'):]).segment_start, (0, '-', 2))
