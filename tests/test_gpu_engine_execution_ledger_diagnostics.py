"""Inert trace fixtures only; no kernel, encoder, model or game execution."""
import ast
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from gpu_engine import engine_gameplay_trace as trace
import test_gpu_engine_gameplay_trace_reuse as reuse


def ledger(batch=8, capacity=4):
    counts = np.arange(batch, dtype=np.int32) % (capacity + 1)
    values = {name: np.arange(batch, dtype=np.int32) + offset
        for offset, name in enumerate(trace._EXECUTION_SCALARS)}
    values['count'] = counts
    for offset, name in enumerate(trace._EXECUTION_ENTRIES):
        shape = (batch, capacity, 2) if name == 'generation' else (batch, capacity)
        values[name] = (np.arange(np.prod(shape), dtype=np.int32).reshape(shape) + offset * 100)
    return SimpleNamespace(**values)


def result(batch=2, **kwargs):
    value = reuse.result(batch, **kwargs)
    return value._replace(state=value.state._replace(core=value.state.core._replace(
        execution=reuse.immutable(ledger(batch)))))


class ExecutionLedgerDiagnosticsTests(unittest.TestCase):
    def test_literal_entered_prefix_b8_row_before_host_and_malformed(self):
        original = ledger()
        class SliceOnly:
            def __init__(self, value):self.value=value;self.shape=value.shape;self.dtype=value.dtype
            def __array__(self, *args):raise AssertionError('whole batch transferred')
            def __getitem__(self, key):
                self.calls.append(key)
                return self.value[key]
        wrapped = {}
        for name, value in vars(original).items():
            wrapped[name] = SliceOnly(value);wrapped[name].calls=[]
        packet = trace._execution(SimpleNamespace(**wrapped), 7)
        self.assertTrue(packet['available']);self.assertEqual(packet['count'],2)
        for name in trace._EXECUTION_SCALARS:self.assertEqual(packet[name],int(getattr(original,name)[7]))
        for name in trace._EXECUTION_ENTRIES:
            self.assertEqual(packet['entries'][name],getattr(original,name)[7,:2].tolist())
            self.assertEqual(wrapped[name].calls,[(7,slice(None,2))])
        self.assertEqual(trace._execution(original,0)['entries']['site'],[])
        original.count[7]=5
        self.assertEqual(trace._execution(original,7)['reason'],'MALFORMED_ACTUAL_EXECUTION_COUNT')
        original.count[7]=-1
        self.assertEqual(trace._execution(original,7)['reason'],'MALFORMED_ACTUAL_EXECUTION_COUNT')
        original.generation=np.zeros((8,4),np.int32)
        self.assertEqual(trace._execution(original,7)['reason'],'MALFORMED_ACTUAL_EXECUTION_LEDGER')
        self.assertFalse(trace._execution(None,0)['available'])
        self.assertFalse(trace._execution(SimpleNamespace(refusal=np.zeros(8,np.int32)),0)['available'])

    def test_additive_boundary_preserves_every_old_field(self):
        old = subprocess.check_output(['git','show',
            '55a7c142991685ef13cb3b7fd8a3f5c80c468472:gpu_engine/engine_gameplay_trace.py'],text=True)
        function=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)and n.name=='boundary')
        scope=dict(vars(trace));exec(compile(ast.Module(body=[function],type_ignores=[]),'original-boundary','exec'),scope)
        value=result(8);actual=trace.boundary(value,7);extra=actual.pop('execution')
        self.assertEqual(actual,scope['boundary'](value,7));self.assertTrue(extra['available'])
        self.assertEqual(extra['entries']['generation'],np.asarray(value.state.core.execution.generation[7,:2]).tolist())

    def test_every_new_leaf_in_immutable_proof_and_mutable_leaf_falls_back(self):
        value=result(8);execution=value.state.core.execution;proof=trace._boundary_proof(value)
        self.assertIsNotNone(proof)
        for name in (*trace._EXECUTION_SCALARS,*trace._EXECUTION_ENTRIES):
            leaf=getattr(execution,name)
            self.assertTrue(any(edge[0]=='array'and edge[1]is leaf for edge in proof),name)
            bad=execution._replace(**{name:np.asarray(leaf).copy()})
            changed=value._replace(state=value.state._replace(core=value.state.core._replace(execution=bad)))
            self.assertIsNone(trace._boundary_proof(changed),name)
        # A writable newly captured leaf cannot leave a stale cached stage.
        stages=np.asarray(execution.stage).copy()
        changed=value._replace(state=value.state._replace(core=value.state.core._replace(
            execution=execution._replace(stage=stages))))
        with TemporaryDirectory()as directory:
            writer=trace.TraceWriter(Path(directory)/'trace')
            writer.begin(initial=changed,**reuse.header(8));self.assertEqual(writer._boundary_packets,{})
            stages[7,0]=197
            self.assertEqual(writer._before_boundary(changed,7)['execution']['entries']['stage'][0],197)
            writer.finish([])

    def test_complete_trace_summary_cache_equivalence_with_ledger(self):
        original_result=reuse.result
        def with_ledger(batch=2, **kwargs):
            value=original_result(batch,**kwargs)
            return value._replace(state=value.state._replace(core=value.state.core._replace(
                execution=reuse.immutable(ledger(batch)))))
        with TemporaryDirectory()as directory,patch.object(reuse,'result',side_effect=with_ledger):
            path=Path(directory)/'trace'
            fresh,summary,_=reuse.trace_run(path,False);path.rename(Path(directory)/'fresh')
            actual,current,_=reuse.trace_run(path,True)
            self.assertEqual(fresh,actual);self.assertEqual(trace.encoded(summary),trace.encoded(current))
            records=[json.loads(line)for line in actual.splitlines()]
            packets=[r['payload']['after']for r in records if r['kind']=='DECISION']
            self.assertTrue(all(p['execution']['available']for p in packets))
            self.assertTrue(any(p['execution']['entries']['stage']for p in packets))


if __name__=='__main__':unittest.main()
