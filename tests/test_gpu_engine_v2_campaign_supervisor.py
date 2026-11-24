import shlex
import unittest

from scripts.supervise_v2_campaign import ADMISSION_SECONDS, build_launch, disposition, segment_order


class CampaignSupervisorTests(unittest.TestCase):
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
        main = {'output': 'v2a-012abc-gpu', 'stop': 1250, 'declaration': {'path': '/out/i', 'sha256': 'd'}}
        packet = dict(main=main, launcher={'path': 'launch.py'}, checkout='/tree', reference_run='/reference',
                      previous_runtime='/previous', runtime={'path': '/current', 'sha256': 'r'},
                      resilience={'path': '/support', 'phase_observer_sha256': 'p', 'signal_driver_sha256': 's'},
                      cpu=32, memory='896Gi', stop_key_reference='env=existing:key')
        attempt = {'number': 2, 'before_round': 100, 'cap_seconds': 3600}
        name, argv = build_launch(packet, main, attempt, None)
        self.assertEqual(name, 'scott-v2a-012abc-gpu-r2')
        self.assertEqual(argv[argv.index('--resume-run')+1], main['output'])
        entry = shlex.split(argv[-1])
        self.assertEqual(entry[entry.index('--adopt-2m-prefix-runtime')+1], '/previous')

        child = dict(main, output='v2a-111111-gpu', parent_round=450, stop=50)
        name, argv = build_launch(packet, child, dict(attempt, before_round=0), 'a'*64)
        self.assertEqual(name, 'scott-v2a-111111-gpu')
        self.assertNotIn('--require-fresh', argv)
        self.assertNotIn('--resume-run', argv)
        self.assertEqual(argv[argv.index('--resilient-support')+1], '/support')
        entry = shlex.split(argv[-1])
        self.assertEqual(entry[entry.index('--cooldown-parent-sha256')+1], 'a'*64)
