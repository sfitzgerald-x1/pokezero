"""Declared deferred evaluation with fake training and separate eval workers."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from pokezero import v5_deferred_evaluation as deferred, v5_evaluation as evaluation, v5_run as run

import test_v5_run as fixture
from test_v5_milestone_evaluation import MilestoneEvaluationTests as Harness


ROUTE = 'd' * 64
UNIT_KEY = b'unit-test-only-key'.ljust(32, b'.')


def artifact(path, data):
    path.write_text(json.dumps(data))
    return evaluation.Artifact(str(path), run._file_sha(path))


class DeferredTrainingTests(unittest.TestCase):
    def test_one_million_declared_games_at_round_50_and_100(self):
        policy = run.EvaluationProtocol('c' * 64, 1_000_000, 'deferred', ROUTE)
        self.assertEqual(policy.metadata()['interval_games'], 1_000_000)
        self.assertEqual(run.evaluation_milestones([20_000] * 100, policy.interval_games),
                         {49: 1_000_000, 99: 2_000_000})
        dates = run.evaluation_milestones([20_000] * 5_000, policy.interval_games)
        self.assertEqual((len(dates), dates[4999]), (100, 100_000_000))
        with self.assertRaisesRegex(ValueError, 'crosses'):
            run.evaluation_milestones([20_000] * 49 + [20_001], policy.interval_games)

    def test_slow_failing_and_unknown_hooks_never_execute_in_training(self):
        for behavior in ('slow', 'failure', 'unknown'):
            with self.subTest(behavior=behavior), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'run'
                protocol = artifact(Path(directory) / 'protocol.json', {})
                plan = replace(fixture.plan(), evaluation=run.EvaluationProtocol(protocol.sha256, 2, 'deferred'))
                invoked = threading.Event()
                def hook(_):
                    invoked.set()
                    if behavior == 'slow':
                        raise AssertionError('blocking hook entered training')
                    if behavior == 'failure':
                        raise RuntimeError('inert failure')
                    return run.EvaluationOutcome('UNKNOWN', reason='inert unknown')
                result = Harness().execute(root, plan, fixture.FakeBackend(), fixture.FakeTraining(), hook)
                self.assertEqual((result['status'], result['steps']), ('COMPLETE_DECLARATION', 6))
                self.assertFalse(invoked.is_set())
                store = run.Store(root, plan, resume=True)
                self.assertEqual(len(store.matching('ITERATION_COMPLETE')), 2)
                for games in (2, 4):
                    pending = root / 'artifacts' / 'cpu-panels' / str(games) / 'request.json'
                    ref = evaluation.Artifact(str(pending), run._file_sha(pending))
                    self.assertEqual(deferred.read_milestone(ref).declared_games, games)
                    self.assertFalse((pending.parent / 'outcomes').exists())

    def test_training_finishes_while_an_actual_eval_attempt_is_still_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            protocol = artifact(Path(directory) / 'protocol.json', {})
            plan = replace(fixture.plan(), evaluation=run.EvaluationProtocol(protocol.sha256, 2, 'deferred'))
            entered, release = threading.Event(), threading.Event()
            outcomes, threads = [], []
            class SlowEvaluator:
                def __init__(self, *_):
                    pass
                def __call__(self, request):
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError('trainer did not finish while evaluator was running')
                    raise RuntimeError('eval failed after trainer finished')
            def evaluate_next():
                path = root / 'artifacts' / 'cpu-panels' / '2' / 'request.json'
                reference = evaluation.Artifact(str(path), run._file_sha(path))
                result = deferred.execute_dispatch(reference, protocol=protocol, enable_cpu_panel=True)
                outcomes.append(result)
            class Backend(fixture.FakeBackend):
                def collect(inner, prepared, *args, **kwargs):
                    if inner.collect_calls:
                        thread = threading.Thread(target=evaluate_next)
                        threads.append(thread)
                        thread.start()
                        self.assertTrue(entered.wait(5))
                    return super().collect(prepared, *args, **kwargs)
            with patch.object(evaluation, 'EvaluationHook', SlowEvaluator), patch.dict(
                    os.environ, {'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'}):
                try:
                    result = Harness().execute(root, plan, Backend(), fixture.FakeTraining(), None)
                    self.assertEqual(result['steps'], 6)
                    self.assertFalse(release.is_set())
                finally:
                    release.set()
                    for thread in threads:
                        thread.join(5)
                self.assertEqual(outcomes[0].status, 'UNKNOWN')
                path = root / 'artifacts' / 'cpu-panels' / '2' / 'request.json'
                ref = evaluation.Artifact(str(path), run._file_sha(path))
                with patch.object(evaluation, 'EvaluationHook', return_value=lambda _: run.EvaluationOutcome(
                        'UNKNOWN', reason='second independent attempt')):
                    second = deferred.execute_dispatch(ref, protocol=protocol, enable_cpu_panel=True, resume=True)
                self.assertEqual(second.status, 'UNKNOWN')
                self.assertEqual(len(list((path.parent / 'outcomes').glob('*.json'))), 2)

    def test_only_bound_fail_stops_before_next_collection_and_relaunch(self):
        for disposition in ('FAIL', 'VOID', 'UNKNOWN', 'foreign-run', 'bad-signature'):
            with self.subTest(disposition=disposition), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'run'
                plan = replace(fixture.plan(), evaluation=run.EvaluationProtocol('c' * 64, 2, 'deferred', ROUTE))
                backend = fixture.FakeBackend()
                original = deferred.publish_dispatch
                def publish(request, *args, **kwargs):
                    receipt = original(request, *args, **kwargs)
                    if request.declared_games != 2:
                        return receipt
                    dispatch_ref = evaluation.Artifact(str(root / receipt.request), receipt.request_sha256)
                    verdict = artifact(Path(directory) / 'verdict.json', dict(
                        schema='s0.transfer.r2', verdict=disposition, validity='VALID', route_declaration_sha256=ROUTE,
                        raw_cells_sha256='a' * 64, primary_result_sha256='b' * 64, independent_code_sha256='c' * 64,
                        completion_endpoints_pp={name: dict(L_min=-4.0, U_max=-3.0) if name == 'D' else
                            dict(L_min=-1.0, U_max=1.0) for name in ('D', 'G', 'J', 'Q_FP', 'Q_H2H')},
                        dispatch_sha256s=[dispatch_ref.sha256]))
                    if disposition == 'FAIL':
                        deferred.publish_route_failure(dispatch_ref, verdict, key=UNIT_KEY)
                    else:
                        if disposition in ('VOID', 'UNKNOWN'):
                            with self.assertRaisesRegex(ValueError, 'failed r2 verdict'):
                                deferred.publish_route_failure(dispatch_ref, verdict, key=UNIT_KEY)
                        payload = dict(schema=deferred.STOP, verdict=disposition, route_schema='s0.transfer.r2',
                            route_declaration_sha256=ROUTE, declaration_sha256=request.declaration_sha256,
                            protocol_sha256=request.protocol_sha256, iteration=0, declared_games=2,
                            dispatch_sha256=dispatch_ref.sha256, independent_verdict_sha256=verdict.sha256)
                        if disposition in ('foreign-run', 'bad-signature'):
                            payload['verdict'] = 'FAIL'
                        if disposition == 'foreign-run':
                            payload['declaration_sha256'] = '1' * 64
                        key = b'wrong' if disposition == 'bad-signature' else UNIT_KEY
                        deferred._publish_equal(root / 'evaluation-stop.json', dict(
                            payload=payload, signature=deferred._signature(payload, key)))
                    return receipt
                with patch.dict(os.environ, {deferred.STOP_KEY_ENV: UNIT_KEY.hex()}), patch.object(
                        deferred, 'publish_dispatch', side_effect=publish):
                    result = Harness().execute(root, plan, backend, fixture.FakeTraining(), None)
                    if disposition == 'FAIL':
                        self.assertEqual(result['reason'], 'S0_TRANSFER_FAIL')
                        self.assertEqual(backend.collect_calls, [1, 2])
                        resume = fixture.FakeBackend()
                        again = Harness().execute(root, plan, resume, fixture.FakeTraining(), None, resume=True)
                        self.assertEqual(again['reason'], 'S0_TRANSFER_FAIL')
                        self.assertEqual(resume.collect_calls, [])
                    else:
                        self.assertEqual(result['status'], 'COMPLETE_DECLARATION')
                        self.assertEqual(backend.collect_calls, [1, 2, 3, 4])

    def test_retains_milestone_full_latest_and_each_actor_across_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            base = fixture.plan()
            third = run.Batch((run.Game(5, 'fifth', 95, (8, 9)), run.Game(6, 'sixth', 96, (10, 11))), 104, 204)
            plan = replace(base, iterations=base.iterations + ((third,),),
                           evaluation=run.EvaluationProtocol('c' * 64, 4, 'deferred'))
            root = Path(directory) / 'run'
            result = Harness().execute(root, plan, fixture.FakeBackend(), fixture.FakeTraining(), None)
            self.assertEqual(result['steps'], 9)
            store = run.Store(root, plan, resume=True)
            snapshots = [run.Snapshot(**row) for row in store.matching('SNAPSHOT')]
            self.assertEqual([s.iteration for s in snapshots if (root / s.checkpoint).exists()], [-1, 1, 2])
            self.assertTrue(all((root / s.archive).is_file() for s in snapshots))
            self.assertEqual(len(list((root / 'artifacts').glob('train-*.pt'))), 3)
            backend = fixture.FakeBackend()
            result = Harness().execute(root, plan, backend, fixture.FakeTraining(), None, resume=True)
            self.assertEqual(result['steps'], 9)
            self.assertEqual(backend.collect_calls, [])


    def test_dispatch_repair_on_resume_reuses_committed_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            policy = run.EvaluationProtocol('c' * 64, 4, 'deferred')
            plan = replace(fixture.plan(), evaluation=policy)
            with patch.object(deferred, 'publish_dispatch', side_effect=OSError('publication unavailable')):
                result = Harness().execute(root, plan, fixture.FakeBackend(), fixture.FakeTraining(), None)
            self.assertEqual(result['steps'], 6)
            store = run.Store(root, plan, resume=True)
            self.assertEqual(len(store.matching('EVALUATION_REQUEST_ERROR')), 1)
            path = root / 'artifacts' / 'cpu-panels' / '4' / 'request.json'
            self.assertFalse(path.exists())
            backend = fixture.FakeBackend()
            result = Harness().execute(root, plan, backend, fixture.FakeTraining(), None, resume=True)
            self.assertEqual((result['steps'], backend.collect_calls), (6, []))
            ref = evaluation.Artifact(str(path), run._file_sha(path))
            snapshot = deferred.read_milestone(ref).snapshot
            self.assertEqual(snapshot, run.Snapshot(**store.matching('SNAPSHOT', iteration=1)[0]))

    def test_failed_void_receipt_is_repaired_on_restart_with_request_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            policy = run.EvaluationProtocol('c'*64, 4, 'deferred')
            plan = replace(fixture.plan(), evaluation=policy)
            publish = deferred.publish_dispatch

            def publish_stale(request, protocol, expected, **kwargs):
                extra = root / 'artifacts' / 'different-full.pt'
                extra.write_bytes(b'other full checkpoint')
                kwargs['full_checkpoint'] = dict(path=str(extra.relative_to(root)), sha256=run._file_sha(extra))
                return publish(request, protocol, expected, **kwargs)

            with patch.object(deferred, 'publish_dispatch', side_effect=publish_stale):
                first = Harness().execute(root, plan, fixture.FakeBackend(), fixture.FakeTraining(), None)
            self.assertEqual(first['steps'], 6)
            path = root / 'artifacts' / 'cpu-panels' / '4' / 'request.json'
            original = path.read_bytes()
            reference = evaluation.Artifact(str(path), run._file_sha(path))
            request = deferred.read_milestone(reference)
            recovered = deferred.MilestoneDispatch(root, policy, {1: 4}, request.declaration_sha256)
            self.assertTrue(recovered.request_missing(request.snapshot))
            self.assertIsNone(deferred.read_eval_void(path, 4))
            write = deferred._publish_equal

            def unavailable(path, document):
                if path.name == 'evaluation-void.json':
                    raise OSError('injected VOID log write failure')
                return write(path, document)

            backend = fixture.FakeBackend()
            with patch.object(deferred, '_publish_equal', side_effect=unavailable):
                failed_log = Harness().execute(root, plan, backend, fixture.FakeTraining(), None, resume=True)
            self.assertEqual((failed_log['steps'], backend.collect_calls), (6, []))
            self.assertIsNone(deferred.read_eval_void(path, 4))
            self.assertEqual(path.read_bytes(), original)
            errors = run.Store(root, plan, resume=True).matching('EVALUATION_REQUEST_ERROR')
            self.assertTrue(any('EVALUATION_VOID_LOG_ERROR' in e['reason'] for e in errors))

            backend = fixture.FakeBackend()
            again = Harness().execute(root, plan, backend, fixture.FakeTraining(), None, resume=True)
            self.assertEqual((again['steps'], backend.collect_calls), (6, []))
            self.assertEqual(path.read_bytes(), original)
            receipt = deferred.read_eval_void(path, 4)
            self.assertEqual(receipt['status'], 'VOID')
            self.assertNotEqual(receipt['existing_request_sha256'], receipt['proposed_request_sha256'])
            self.assertIn('full_checkpoint', receipt['diff'])

    def test_void_revokes_live_and_resumed_dispatch_consumers(self):
        from pokezero import v5_progress_evaluation as progress

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            profile = dict(schema=progress.SCHEMA, banks=dict.fromkeys(progress.OPPONENTS))
            protocol = artifact(Path(directory) / 'protocol.json', profile)
            policy = run.EvaluationProtocol(protocol.sha256, 2, 'deferred')
            plan = replace(fixture.plan(), evaluation=policy)
            Harness().execute(root, plan, fixture.FakeBackend(), fixture.FakeTraining(), None)
            path = root / 'artifacts' / 'cpu-panels' / '2' / 'request.json'
            reference = evaluation.Artifact(str(path), run._file_sha(path))
            request = deferred.read_milestone(reference)
            dispatch = deferred.MilestoneDispatch(root, policy, {0: 2}, request.declaration_sha256)

            def finish(actual):
                self.assertIsNone(deferred.read_eval_void(path, 2))
                dispatch.publish_committed(request.snapshot, report_error=lambda _: None,
                    full_checkpoint=dict(path='artifacts/different.pt', sha256='f'*64))
                return Harness().result(actual)

            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'}), \
                    patch.object(evaluation, 'EvaluationHook', return_value=finish) as hook:
                outcome = deferred.execute_dispatch(reference, protocol=protocol, enable_cpu_panel=True)
                self.assertEqual(outcome.status, 'UNKNOWN')
                self.assertIn('VOID milestone', outcome.reason)
                self.assertIsNone(outcome.report)
                with self.assertRaises(deferred.VoidEvaluation):
                    deferred.execute_dispatch(reference, protocol=protocol, enable_cpu_panel=True, resume=True)
                hook.assert_called_once()
            for read in (
                lambda: deferred.read_milestone(reference),
                lambda: deferred.read_dispatched_result(reference, protocol=protocol),
                lambda: deferred.execute_assay(reference, protocol, 'external_fp_greedy', resume=True),
                lambda: progress.summarize(reference, protocol, root/'registry'),
            ):
                with self.assertRaises(deferred.VoidEvaluation):
                    read()
            self.assertEqual(run._file_sha(path), reference.sha256)

    def test_assay_discarded_if_request_is_voided_while_child_runs(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace/'run'
            entry = artifact(workspace/'entry.json', {})
            inputs = artifact(workspace/'inputs.json', {})
            assay = 'external_fp_greedy'
            protocol = artifact(workspace/'protocol.json', dict(schema=deferred.ASSAY_PROTOCOL,
                milestones={'2': {assay: dict(entry=entry.__dict__, input_manifest=inputs.__dict__, deadline_seconds=10)}}))
            policy = run.EvaluationProtocol(protocol.sha256, 2, 'deferred')
            Harness().execute(root, replace(fixture.plan(), evaluation=policy), fixture.FakeBackend(), fixture.FakeTraining(), None)
            path = root/'artifacts/cpu-panels/2/request.json'
            reference = evaluation.Artifact(str(path), run._file_sha(path))
            request = deferred.read_milestone(reference)
            dispatch = deferred.MilestoneDispatch(root, policy, {0: 2}, request.declaration_sha256)

            def finish(command, **kwargs):
                dispatch.publish_committed(request.snapshot, report_error=lambda _: None,
                    full_checkpoint=dict(path='artifacts/different.pt', sha256='f'*64))
                output = Path(command[command.index('--output') + 1])
                output.write_text(json.dumps(dict(status='RESULT', assay=assay, dispatch_sha256=reference.sha256)))
                return SimpleNamespace(returncode=0)

            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'}), \
                    patch('subprocess.run', side_effect=finish):
                outcome = deferred.execute_assay(reference, protocol, assay)
            self.assertEqual(outcome['status'], 'UNKNOWN')
            self.assertIn('VOID milestone', outcome['reason'])
            self.assertNotIn('result', outcome)

    def test_assay_attempts_require_separate_device_and_preserve_failure_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / 'run'
            inputs = artifact(workspace / 'inputs.json', {'registered_fixed_cells': []})
            entry = workspace / 'adapter.py'
            entry.write_text(
                'import argparse, json\n'
                'from pathlib import Path\n'
                'p=argparse.ArgumentParser()\n'
                'for field in ("dispatch", "dispatch-sha256", "inputs", "inputs-sha256", "ledger", "output"):\n'
                '    p.add_argument("--"+field, required=True)\n'
                'p.add_argument("--resume", action="store_true"); a=p.parse_args()\n'
                'if not a.resume: raise SystemExit(7)\n'
                'assert json.loads(Path(a.inputs).read_text())["registered_fixed_cells"] == []\n'
                'with Path(a.output).open("x") as f:\n'
                '    json.dump(dict(status="RESULT", assay="matched_h2h_engine", '
                'dispatch_sha256=a.dispatch_sha256), f)\n'
            )
            protocol = artifact(workspace / 'protocol.json', dict(schema=deferred.ASSAY_PROTOCOL,
                milestones={'2': {'matched_h2h_engine': dict(entry=dict(path=str(entry), sha256=run._file_sha(entry)),
                    input_manifest=inputs.__dict__, deadline_seconds=10)}}))
            policy = run.EvaluationProtocol(protocol.sha256, 2, 'deferred', ROUTE)
            plan = replace(fixture.plan(), evaluation=policy)
            with patch.dict(os.environ, {deferred.STOP_KEY_ENV: UNIT_KEY.hex()}):
                Harness().execute(root, plan, fixture.FakeBackend(), fixture.FakeTraining(), None)
            path = root / 'artifacts' / 'cpu-panels' / '2' / 'request.json'
            ref = evaluation.Artifact(str(path), run._file_sha(path))
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'}):
                with self.assertRaisesRegex(ValueError, 'separate GPU'):
                    deferred.execute_assay(ref, protocol, 'matched_h2h_engine')
            # The inert child only writes JSON. No CUDA or cluster API is called.
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'JAX_PLATFORMS': 'cuda'}):
                first = deferred.execute_assay(ref, protocol, 'matched_h2h_engine')
                self.assertEqual((first['status'], first['exit_code']), ('UNKNOWN', 7))
                second = deferred.execute_assay(ref, protocol, 'matched_h2h_engine', resume=True)
                self.assertEqual((second['status'], second['attempt']), ('RESULT', 1))
                with self.assertRaisesRegex(ValueError, 'cannot replay'):
                    deferred.execute_assay(ref, protocol, 'matched_h2h_engine', resume=True)
            evidence = path.parent / 'assays' / 'matched_h2h_engine' / 'outcomes' / '000000.json'
            self.assertEqual(json.loads(evidence.read_text()), first)
            self.assertFalse((root / 'evaluation-stop.json').exists())



if __name__ == '__main__':
    unittest.main()
