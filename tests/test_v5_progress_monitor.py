"""Monitor coverage without Kubernetes or live Slack calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from pokezero import v5_deferred_evaluation as deferred
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
            request.write_text('{}')
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

    def test_immutable_mismatch_stays_pending_but_voids_monitor_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root/'protocol.json'
            protocol.write_text('{}')
            policy = run.EvaluationProtocol(run._file_sha(protocol), 1_000_000, 'deferred')
            dispatch = deferred.MilestoneDispatch(root, policy, {0: 1_000_000}, 'a'*64)
            request = dispatch.path(1_000_000)
            request.parent.mkdir(parents=True)
            request.write_text(run._json(dict(schema=deferred.MILESTONE, different_binding=True)))
            original = request.read_bytes()
            snapshot = run.Snapshot(0, 'train.pt', 'b'*64, 'actor.npz', 'c'*64, 'd'*64)
            later = run.Snapshot(1, 'later.pt', 'e'*64, 'later.npz', 'f'*64, 'd'*64)
            errors = []
            self.assertIsNone(dispatch.publish_committed(snapshot, report_error=errors.append))
            with patch.object(deferred, 'publish_dispatch', wraps=deferred.publish_dispatch) as publish:
                self.assertIsNone(dispatch.publish_committed(later, report_error=errors.append))
                publish.assert_called_once()
            self.assertEqual(request.read_bytes(), original)
            self.assertEqual(list(dispatch.pending), [1_000_000])
            self.assertTrue(dispatch.request_missing(snapshot))
            self.assertTrue(all(error.startswith('EVALUATION_VOID_ALERT') for error in errors))
            record = deferred.read_eval_void(request, 1_000_000)
            self.assertEqual(record['existing_request_sha256'], run._sha(json.loads(original)))
            self.assertNotEqual(record['existing_request_sha256'], record['proposed_request_sha256'])
            self.assertEqual(record['diff']['different_binding'], dict(existing={'different_binding': True}, proposed={}))

            config = dict(staged_run_root=str(root), registry=str(root/'registry'), milestones=1,
                          protocol=dict(path=str(protocol), sha256=policy.protocol_sha256),
                          max_active_jobs=1, max_attempts=2)
            folder = root/'report'
            launch, log = Mock(), Mock()
            with patch.object(panels, 'summarize') as summarize:
                result = monitor.tick(config, folder, allow_launch=True, launch=launch, log=log)
                monitor.tick(config, folder, allow_launch=True, launch=launch, log=log)
            self.assertEqual((result['queued'], result['finalized'], result['published_games']), (0, 0, 1_000_000))
            summarize.assert_not_called()
            launch.assert_not_called()
            log.assert_called_once()
            self.assertIn('EVALUATION_VOID_ALERT', log.call_args.args[0])
            self.assertEqual(json.loads((folder/'eval-queue.json').read_bytes())['void'], [record])
            self.assertEqual((folder/'eval-timeline.jsonl').read_text(), '')
            self.assertIn('| 1,000,000 | VOID |', (folder/'showdown-progress.md').read_text())

    def test_cached_staged_result_is_hidden_after_published_request_goes_void(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, staged = root/'original', root/'staged'
            protocol = root/'protocol.json'
            protocol.write_text('{}')
            policy = run.EvaluationProtocol(run._file_sha(protocol), 1_000_000, 'deferred')
            dispatch = deferred.MilestoneDispatch(original, policy, {0: 1_000_000}, 'a'*64)
            snapshot = run.Snapshot(0, 'train.pt', 'b'*64, 'actor.npz', 'c'*64, 'd'*64)
            published = dispatch.publish_committed(snapshot, report_error=self.fail)
            request = staged/published.request
            request.parent.mkdir(parents=True)
            request.write_bytes((original/published.request).read_bytes())
            folder = root/'report'
            cached = folder/'milestones/1000000.json'
            cached.parent.mkdir(parents=True)
            # A cached result needs no model or bank load to be disqualified.
            cached.write_text(json.dumps({'dispatch_sha256': published.request_sha256,
                                         'protocol_sha256': policy.protocol_sha256, 'complete': True}))
            dispatch.publish_committed(snapshot, report_error=lambda _: None,
                full_checkpoint=dict(path='different.pt', sha256='e'*64))
            config = dict(staged_run_root=str(staged), registry=str(root/'registry'), milestones=1,
                          protocol=dict(path=str(protocol), sha256=policy.protocol_sha256))
            log = Mock()
            with patch.object(panels, 'summarize') as summarize:
                result = monitor.tick(config, folder, log=log)
            self.assertEqual((result['queued'], result['finalized']), (0, 0))
            summarize.assert_not_called()
            log.assert_called_once()
            self.assertIn('EVALUATION_VOID_ALERT', log.call_args.args[0])
            self.assertIn('| 1,000,000 | VOID |', (folder/'showdown-progress.md').read_text())
            self.assertEqual((folder/'eval-timeline.jsonl').read_text(), '')

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
