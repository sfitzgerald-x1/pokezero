"""Monitor coverage without Kubernetes or live Slack calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from pokezero import v5_progress_evaluation as panels, v5_progress_monitor as monitor, v5_run as run


class ProgressMonitorTests(unittest.TestCase):
    def test_observer_schema_error_leaves_admitted_job_untouched(self):
        record = dict(name='v2q-0123456789ab-cpu-r1', time=0, state='ADMITTING')
        release, log = Mock(), Mock()
        monitor._observe_owned(dict(attempts=[record]), Mock(side_effect=KeyError), release, log)
        self.assertEqual(record['state'], 'ADMITTING')
        release.assert_not_called()
        log.assert_called_once_with('job observation deferred: KeyError')

    def test_queues_once_then_releases_own_terminal_job_and_posts_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root/'artifacts/cpu-panels/1000000/request.json'
            request.parent.mkdir(parents=True)
            request.write_text('test request only')
            protocol = root/'protocol.json'
            protocol.write_text('{}')
            source = dict(staged_run_root=str(root), registry=str(root/'registry'), milestones=1,
                          protocol=dict(path=str(protocol), sha256=run._file_sha(protocol)),
                          max_active_jobs=1, max_attempts=2)
            def row(complete):
                def panel(n):
                    return dict(games_declared=n, games_completed=n if complete else 0, unknown=0,
                                pending=0 if complete else n, win_rate=.5 if complete else None,
                                ci95=[.46,.54] if complete else None)
                return dict(schema=panels.SCHEMA, declared_games=1_000_000, protocol_sha256=run._file_sha(protocol),
                    declaration_sha256='declaration', dispatch_sha256=run._file_sha(request), published_unix=10,
                    observed_unix=30 if complete else 20, complete=complete, work_started=complete,
                    queue_wait_seconds=0, eval_lag_seconds=10, panel={
                        n: panel(1000 if n=='foul_play' else 2000) for n in panels.OPPONENTS})
            launch, observe, release, send = Mock(return_value=0), Mock(return_value=None), Mock(), Mock(return_value='1.1')
            folder = root/'report'
            options = dict(allow_launch=True, channel='test-channel', send=send, observe=observe,
                           release=release, launch=launch)
            with patch.object(panels, 'summarize', side_effect=[row(False), row(True)]), \
                    patch.object(panels, 'pending_shards', return_value=[dict(opponent='foul_play', shard=0, cells_pending=5)]):
                self.assertEqual(monitor.tick(source, folder, **options)['queued'], 1)
                opaque = launch.call_args.args[-1]
                self.assertRegex(opaque, r'^v2q-[0-9a-f]{12}-cpu-r1$')
                observe.return_value = ('own-job-uid', 'Complete', None)
                self.assertEqual(monitor.tick(source, folder, **options)['finalized'], 1)
                monitor.tick(source, folder, **options)
            launch.assert_called_once()
            release.assert_called_once_with(opaque, 'own-job-uid')
            send.assert_called_once()
            text = monitor._message(send.call_args.args[1])
            self.assertIn('MD: 50.0% (2000/2000 completed)', text)
            self.assertIn('H2H: 50.0% (2000/2000 completed)', text)
            self.assertIn('FP: 50.0% (1000/1000 completed)', text)
            timeline = (folder/'eval-timeline.jsonl').read_text().splitlines()
            self.assertEqual(len(timeline), 1)
            self.assertEqual(json.loads(timeline[0])['declared_games'], 1_000_000)

    def test_exhausted_shard_reports_its_last_failure_without_relaunch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dispatch = root/'artifacts/cpu-panels/1000000/request.json'
            dispatch.parent.mkdir(parents=True)
            dispatch.write_text('{}')
            protocol = root/'protocol.json'
            protocol.write_text('{}')
            config = dict(staged_run_root=str(root), registry=str(root/'registry'), milestones=1,
                          protocol=dict(path=str(protocol), sha256=run._file_sha(protocol)),
                          max_active_jobs=1, max_attempts=1)
            folder = root/'report'
            launch, release = Mock(return_value=0), Mock()
            observe = Mock(return_value=('owned', 'Failed', 'DeadlineExceeded'))
            pending = dict(opponent='champion_9375', shard=2, cells_pending=17)
            with patch.object(panels, 'summarize', return_value=dict(complete=False)), \
                    patch.object(panels, 'pending_shards', return_value=[pending]), \
                    patch.object(monitor.report, 'update_report'):
                first = monitor.tick(config, folder, allow_launch=True, launch=launch, observe=observe, release=release)
                second = monitor.tick(config, folder, allow_launch=True, launch=launch, observe=observe, release=release)
            self.assertEqual(first['exhausted_shards'], 0)
            self.assertEqual(second['exhausted_shards'], 1)
            self.assertEqual(second['queued'], 1)
            issue, = json.loads((folder/'eval-queue.json').read_text())['exhausted']
            self.assertEqual(issue['last_reason'], 'DeadlineExceeded')
            self.assertEqual(issue['cells_pending'], 17)
            launch.assert_called_once()
            release.assert_called_once_with(launch.call_args.args[-1], 'owned')

    def test_ambiguous_slack_failure_gets_receipt_without_repost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            send = Mock(side_effect=TimeoutError)
            row = dict(declared_games=1_000_000, dispatch_sha256='request')
            monitor._notify(root, row, 'test-channel', send)
            monitor._notify(root, row, 'test-channel', send)
            send.assert_called_once()
            receipt = next((root/'slack').glob('*.receipt.json'))
            self.assertEqual(json.loads(receipt.read_bytes())['status'], 'UNCONFIRMED')


if __name__ == '__main__':
    unittest.main()
