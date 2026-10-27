"""Authored timer/owner processes only; no battle roots, native calls or allocation."""
from copy import deepcopy
import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

from gpu_engine import panel_cost_runtime as runtime,panel_owner as owner
from tests.test_gpu_engine_panel_cost import entry


class TimingRuntimeTests(unittest.TestCase):
    def setUp(self):
        # Direct filesystem-envelope tests have no owned child/channel. Actual
        # subprocess tests run a fresh interpreter and exercise the real pipe.
        patcher=mock.patch.object(runtime,'notify_start')
        patcher.start();self.addCleanup(patcher.stop)
        patcher=mock.patch.object(runtime,'notification_fd',return_value=-1)
        patcher.start();self.addCleanup(patcher.stop)

    def config(self,limit=60):
        from gpu_engine import panel_cost
        cfg=entry();cfg.update(mass_tolerance=1e-5,probability_tolerance=1e-5)
        cfg['operation_budget']['limit_seconds']=limit
        cfg['cost_admission']['decision_limit_seconds']=limit
        cfg['cost_admission']['operation_budget_sha256']=panel_cost.operation_budget(cfg)[1]
        return cfg

    def source(self,panel):
        panel.mkdir()
        payload=b''.join(json.dumps(dict(slot=i)).encode()+b'\n' for i in range(20))
        with gzip.open(panel/'inputs.jsonl.gz','wb') as stream:stream.write(payload)
        frozen=dict(slots=20,input_jsonl_sha256=hashlib.sha256(payload).hexdigest(),
            compressed_sha256=hashlib.sha256((panel/'inputs.jsonl.gz').read_bytes()).hexdigest())
        (panel/'inputs-frozen.json').write_text(json.dumps(frozen))
        return frozen

    def result(self,panel,slot,frozen):
        record=dict(slot=slot,status='UNKNOWN',category='INPUT_OR_PLAN_REFUSED',
            identified_mass='0',unknown_mass='1',input_jsonl_sha256=frozen['input_jsonl_sha256'])
        with gzip.open(panel/f'slot-{slot:04d}.json.gz','wb') as stream:
            stream.write(json.dumps(record).encode())
        runtime.sync_file(panel/f'slot-{slot:04d}.json.gz')
        runtime.exclusive_json(panel/f'authored-progress-{slot:04d}.json',dict(completed_slots=slot+1))
        return record

    def test_complete_timing_binds_full_result_and_retains_refusal_path(self):
        cfg=self.config();attempt='a'*64;started=time.monotonic()
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{runtime.ENV_ATTEMPT:attempt}):
            panel=Path(directory)/'panel';frozen=self.source(panel)
            session=runtime.Session(cfg,panel)
            for i in range(20):
                frame=session.begin(i,frozen)
                record=self.result(panel,i,frozen)
                session.finish(frame,record,'refused',0)
            watch=runtime.Watch(cfg,panel,attempt,os.getpid(),started)
            result=watch.reconstruct(True)
            self.assertTrue(result['timing_evidence_valid'],result['timing_issues'])
            self.assertEqual(result['latency']['timed_roots'],20)
            self.assertEqual(result['latency']['exercised_paths'],dict(refused=20))
            self.assertEqual(len(result['witness_publication_seconds']),20)
            self.assertFalse(result['latency']['comparison_cost_qualified'])
            self.assertFalse(watch.overdue())
            # Result mutation invalidates timing even when timestamps are unchanged.
            (panel/'slot-0000.json.gz').write_bytes(b'authored corrupt replacement')
            changed=watch.reconstruct(True)
            self.assertFalse(changed['timing_evidence_valid'])
            self.assertEqual(changed['observations'][0]['status'],'MISSING')

    def test_wrong_attempt_entry_pid_input_or_absolute_clock_refuses(self):
        cfg=self.config();attempt='b'*64;start=time.monotonic()
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{runtime.ENV_ATTEMPT:attempt}):
            panel=Path(directory)/'panel';frozen=self.source(panel)
            frame=runtime.Session(cfg,panel).begin(0,frozen)
            bound=runtime.binding(cfg,attempt,os.getpid())
            for field,value in [('attempt','c'*64),('worker_pid',os.getpid()+1),
                    ('entry_sha256','d'*64),('input_freeze_sha256','e'*64),
                    ('start_monotonic',float('nan')),('start_monotonic',start-1),
                    ('start_monotonic',time.monotonic()+100),('slot',20)]:
                changed=dict(frame);changed[field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):
                    runtime.validate_start(changed,bound,frozen,20,start,time.monotonic())

    def test_missing_done_has_no_inferred_positive_latency_lower_bound(self):
        cfg=self.config(limit=1);attempt='c'*64;start=time.monotonic()-10
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{runtime.ENV_ATTEMPT:attempt}):
            panel=Path(directory)/'panel';frozen=self.source(panel)
            with mock.patch.object(runtime.time,'monotonic',return_value=start+1):
                runtime.Session(cfg,panel).begin(0,frozen)
            watch=runtime.Watch(cfg,panel,attempt,os.getpid(),start)
            self.assertTrue(watch.overdue())
            result=watch.reconstruct(True)
            self.assertEqual(result['observations'][0],dict(slot=0,status='MISSING'))
            self.assertEqual(result['latency']['p95_lower_seconds'],0)
            self.assertEqual(result['latency']['p95_upper_seconds'],'unbounded')

    def test_done_before_limit_but_missing_publication_is_not_a_proven_overrun(self):
        cfg=self.config(limit=1);attempt='d'*64;start=time.monotonic()-10
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{runtime.ENV_ATTEMPT:attempt}):
            panel=Path(directory)/'panel';frozen=self.source(panel)
            session=runtime.Session(cfg,panel)
            with mock.patch.object(runtime.time,'monotonic',return_value=start+1):frame=session.begin(0,frozen)
            record=self.result(panel,0,frozen)
            with mock.patch.object(runtime.time,'monotonic',return_value=start+1.2):session.finish(frame,record,'refused',0)
            (panel/'slot-0000-timing-publication.json').unlink()
            watch=runtime.Watch(cfg,panel,attempt,os.getpid(),start)
            self.assertTrue(watch.overdue())
            result=watch.reconstruct(True)
            self.assertEqual(result['observations'][0]['status'],'COMPLETE')
            self.assertLess(result['observations'][0]['seconds'],1)
            self.assertFalse(result['timing_evidence_valid'])

    def test_future_and_cost_entries_must_reference_the_identical_budget(self):
        from gpu_engine import panel_cost
        cfg=self.config();changed=deepcopy(cfg);changed['operation_budget']['limit_seconds']=61
        with self.assertRaises(ValueError):panel_cost.contract(changed)
        future=deepcopy(cfg);future.pop('cost_admission')
        self.assertEqual(panel_cost.operation_budget(future),panel_cost.operation_budget(cfg))

    def test_immutable_publication_is_atomic_and_refuses_existing_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'witness.json';actual_link=runtime.os.link;seen=[]
            def publish(source,target):
                self.assertFalse(path.exists())
                self.assertEqual(json.loads(Path(source).read_bytes()),{'complete':'authored'})
                seen.append(True);return actual_link(source,target)
            with mock.patch.object(runtime.os,'link',side_effect=publish):
                runtime.exclusive_json(path,{'complete':'authored'})
            self.assertEqual(seen,[True])
            with self.assertRaises(FileExistsError):runtime.exclusive_json(path,{'replacement':True})
            self.assertEqual(json.loads(path.read_bytes()),{'complete':'authored'})

    def test_malformed_publication_types_become_issues_without_losing_known_time(self):
        cfg=self.config();attempt='f'*64;start=time.monotonic()
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{runtime.ENV_ATTEMPT:attempt}):
            panel=Path(directory)/'panel';frozen=self.source(panel)
            session=runtime.Session(cfg,panel);frame=session.begin(0,frozen)
            session.finish(frame,self.result(panel,0,frozen),'refused',0)
            for value in ([],None,1):
                (panel/'slot-0000-timing-publication.json').write_text(json.dumps(value))
                result=runtime.Watch(cfg,panel,attempt,os.getpid(),start).reconstruct(True)
                self.assertFalse(result['timing_evidence_valid'])
                self.assertEqual(result['observations'][0]['status'],'COMPLETE')

    def test_proven_overrun_retains_point_and_prevents_another_slot(self):
        cfg=self.config(limit=1);attempt='1'*64;start=time.monotonic()-10
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{runtime.ENV_ATTEMPT:attempt}):
            panel=Path(directory)/'panel';frozen=self.source(panel);session=runtime.Session(cfg,panel)
            with mock.patch.object(runtime.time,'monotonic',return_value=start+1):frame=session.begin(0,frozen)
            record=self.result(panel,0,frozen)
            with mock.patch.object(runtime.time,'monotonic',return_value=start+3),self.assertRaises(runtime.OperationBudgetExceeded):
                session.finish(frame,record,'refused',0)
            with self.assertRaises(runtime.OperationBudgetExceeded):session.begin(1,frozen)
            watch=runtime.Watch(cfg,panel,attempt,os.getpid(),start)
            with mock.patch.object(runtime,'validate_done',side_effect=AssertionError('large law parsed live')):
                self.assertTrue(watch.overdue())
            result=watch.reconstruct(True)
            self.assertEqual(result['operation_overruns'],[0])
            self.assertEqual(result['observations'][0]['seconds'],2)

    def test_owned_watchdog_reaps_child_and_keeps_missing_timing_unknown(self):
        cfg=self.config(limit=.1)
        child='''import json,sys,time
from pathlib import Path
from gpu_engine.panel_cost_runtime import Session
from tests.test_gpu_engine_panel_cost_runtime import TimingRuntimeTests
cfg=json.loads(sys.argv[1]);panel=Path(sys.argv[2]);frozen=TimingRuntimeTests().source(panel)
Session(cfg,panel).begin(0,frozen)
time.sleep(30)
'''
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            out=Path(directory)/'owned';panel=out/'panel'
            code=owner.own_cost([sys.executable,'-c',child,json.dumps(cfg),str(panel)],
                out=out,panel=panel,entry=cfg,calendar_seconds=10)
            terminal=json.loads((out/'owner-terminal.json').read_bytes())
        self.assertEqual(code,124)
        self.assertTrue(terminal['slot_watchdog_stopped_worker'])
        self.assertTrue(terminal['worker_reaped'])
        self.assertTrue(terminal['owned_group_absent'])
        self.assertNotEqual(terminal['actual_worker_exit'],0)
        self.assertEqual(terminal['slot_timing']['observations'][0]['status'],'MISSING')
        self.assertEqual(terminal['slot_timing']['latency']['p95_lower_seconds'],0)

    def test_first_start_fsync_hang_is_visible_via_owned_pipe(self):
        cfg=self.config(limit=.1)
        child='''import json,sys,time
from pathlib import Path
from gpu_engine import panel_cost_runtime as r
from tests.test_gpu_engine_panel_cost_runtime import TimingRuntimeTests
cfg=json.loads(sys.argv[1]);panel=Path(sys.argv[2]);frozen=TimingRuntimeTests().source(panel)
r.exclusive_json=lambda *args:time.sleep(30)
r.Session(cfg,panel).begin(0,frozen)
'''
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            out=Path(directory)/'owned';panel=out/'panel'
            code=owner.own_cost([sys.executable,'-c',child,json.dumps(cfg),str(panel)],
                out=out,panel=panel,entry=cfg,calendar_seconds=10)
            terminal=json.loads((out/'owner-terminal.json').read_bytes())
        self.assertEqual(code,124)
        self.assertTrue(terminal['slot_watchdog_stopped_worker'])
        self.assertFalse(terminal['timed_out'])
        self.assertTrue(terminal['worker_reaped'])
        self.assertTrue(terminal['owned_group_absent'])
        self.assertEqual(terminal['attempted_slots'],1)
        self.assertEqual(terminal['slot_timing']['observations'][0]['status'],'MISSING')

    def test_late_done_between_polls_cannot_be_operational_success_at_exit_zero(self):
        cfg=self.config(limit=2)
        child='''import json,sys,time
from pathlib import Path
from gpu_engine import panel_cost_runtime as r
from tests.test_gpu_engine_panel_cost_runtime import TimingRuntimeTests
cfg=json.loads(sys.argv[1]);panel=Path(sys.argv[2]);helper=TimingRuntimeTests();frozen=helper.source(panel)
session=r.Session(cfg,panel)
for i in range(20):
    frame=session.begin(i,frozen);record=helper.result(panel,i,frozen)
    if i==19:time.sleep(2.1)
    try:session.finish(frame,record,'refused',0)
    except r.OperationBudgetExceeded:pass
'''
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(runtime.Watch,'observe',return_value=dict(expired=False,deadline=None)):
            out=Path(directory)/'owned';panel=out/'panel'
            code=owner.own_cost([sys.executable,'-c',child,json.dumps(cfg),str(panel)],
                out=out,panel=panel,entry=cfg,calendar_seconds=15)
            terminal=json.loads((out/'owner-terminal.json').read_bytes())
        self.assertEqual(terminal['actual_worker_exit'],0)
        self.assertEqual(terminal['valid_completed_slots'],20)
        self.assertEqual(terminal['slot_timing']['operation_overruns'],[19])
        self.assertEqual(terminal['slot_timing']['latency']['latency_threshold'],'MET')
        self.assertEqual(code,124)
        self.assertEqual(terminal['status'],'FAILED_OR_INTERRUPTED')

    def test_unexpected_timing_reconstruction_error_still_publishes_actual_exit(self):
        cfg=self.config()
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(runtime.Watch,'reconstruct',side_effect=AttributeError('authored malformed witness')):
            out=Path(directory)/'owned'
            code=owner.own_cost([sys.executable,'-c','pass'],out=out,panel=out/'panel',
                entry=cfg,calendar_seconds=5)
            terminal=json.loads((out/'owner-terminal.json').read_bytes())
        self.assertEqual(code,1)
        self.assertEqual(terminal['actual_worker_exit'],0)
        self.assertTrue(terminal['worker_reaped'])
        self.assertTrue(terminal['owned_group_absent'])
        self.assertIn('AttributeError',terminal['slot_timing']['reconstruction_error'])


if __name__=='__main__':unittest.main()
