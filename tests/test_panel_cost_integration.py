"""Authored source-freeze/timer integration; no real pool or scientific launch."""
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from tests import test_prospective_fidelity_panel as fixtures
from gpu_engine import collect_fidelity_panel as collector,panel_cost_runtime as runtime
from tests.test_gpu_engine_panel_cost import entry as cost_entry


@unittest.skipUnless(fixtures.AVAILABLE,'isolated JAX and native constructor fixture required')
class CostCollectorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.ProspectivePanelTests.setUpClass()

    def test_twenty_authored_sources_freeze_before_plan_and_timer_precedes_binding(self):
        from gpu_engine import prospective_panel,panel_owner
        helper=fixtures.ProspectivePanelTests('test_all_inputs_and_context_close_before_any_plan')
        self.addCleanup(helper.doCleanups)
        state,sources,source,config,_=helper.fixture(trace=False)
        config.update(cost_entry());config.update(constructor_provenance={})
        # Repeated tiny authored fixture teams test lifecycle ordering only.
        # Production cost_source_rows itself is separately checked for40 distinct
        # registered team indices; no real pool is loaded in this test.
        rows=tuple(dict(slot=i,teams=[0,1],prefix_requests=0) for i in range(20))
        config['prefix_cycle']=1;attempt='e'*64;calls=[];start=time.monotonic()
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{runtime.ENV_ATTEMPT:attempt}):
            out=Path(directory)/'panel'
            actual_asarray=helper.j.asarray;transfers=[]
            def observed_asarray(value,*args,**kwargs):
                if ((out/'inputs-frozen.json').exists() and isinstance(value,helper.np.ndarray)
                        and value.ndim and value.shape[0]==1):
                    self.assertTrue((out/'timing-current.json').exists(),
                        'root transfer happened before the full-slot clock')
                    transfers.append(True)
                return actual_asarray(value,*args,**kwargs)
            def dispatch(root,initial,record,original,bound,table,entry,*,record_sha256):
                frozen=json.loads((out/'inputs-frozen.json').read_bytes())
                with gzip.open(out/'inputs.jsonl.gz','rt') as stream:records=[json.loads(line) for line in stream]
                self.assertEqual(len(records),20)
                self.assertEqual([r['slot'] for r in records],list(range(20)))
                frame=json.loads((out/'timing-current.json').read_bytes())
                self.assertEqual(frame['slot'],record['slot'])
                self.assertEqual(frame['input_freeze_sha256'],runtime.sha(frozen))
                self.assertTrue((out/f"slot-{record['slot']:04d}-timing-start.json").is_file())
                self.assertEqual(helper.s._sha(record),record_sha256)
                calls.append(record['slot'])
                return helper.f.Refused((helper.f.Gap('authored','UNSUPPORTED','retained'),))
            with mock.patch.object(collector,'runtime_admission',return_value={}), \
                    mock.patch.object(runtime,'notify_start'), \
                    mock.patch.object(runtime,'notification_fd',return_value=-1), \
                    mock.patch.object(helper.j,'asarray',side_effect=observed_asarray), \
                    mock.patch('gpu_engine.benchmark.sole_gpu',return_value=type('Device',(),{'device_kind':'authored CPU'})()), \
                    mock.patch.object(prospective_panel,'admit',return_value=sources), \
                    mock.patch.object(collector,'cost_source_rows',return_value=rows), \
                    mock.patch.object(prospective_panel,'plan_for_record',side_effect=dispatch), \
                    contextlib.redirect_stdout(io.StringIO()):
                collector.run_cost(config,Path('authored-unused.npz'),out)
            self.assertEqual(calls,list(range(20)))
            self.assertGreater(len(transfers),20)
            physical=panel_owner.reconstruct(out,config)
            self.assertTrue(physical['input_freeze_valid'])
            self.assertEqual(physical['counts'],{'UNKNOWN':20})
            timing=runtime.Watch(config,out,attempt,os.getpid(),start).reconstruct(True)
            self.assertTrue(timing['timing_evidence_valid'],timing['timing_issues'])
            self.assertEqual(timing['latency']['exercised_paths'],dict(refused=20))
            self.assertEqual(len(timing['execution_evidence']),20)
            self.assertFalse(timing['latency']['comparison_cost_qualified'])


if __name__=='__main__':unittest.main()
