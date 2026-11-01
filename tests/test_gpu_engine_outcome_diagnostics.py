"""Literal observational transport only, no kernel/game/model execution."""
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_outcome_diagnostics as d,engine_kernel_frame_risk as risk

Context=namedtuple('DiagnosticFixtureContext','ledger risks diagnostics')
Tape=namedtuple('DiagnosticFixtureTape','count stream_id cursor refusal execution words')

class OutcomeDiagnosticTests(unittest.TestCase):
    def tape(self,b=8):
        z=j.zeros(b,j.int32);yes=j.ones(b,bool)
        cursor=NS(turn=j.arange(b,dtype=j.int32)+1,phase=z,generation=j.stack((z+2,z+3),axis=1))
        context=Context(NS(refusal=j.arange(b,dtype=j.int32)+10,frame=z+5),NS(carry=NS(risk=j.arange(b,dtype=j.int32)+20)),d.empty(b))
        tape=Tape(z+7,z+12001,cursor,z+30,context,j.arange(b*10,dtype=j.int32).reshape(b,1,10))
        outcome=risk.Outcome(z+70,yes,~yes,yes,z+2,z+4,~yes,yes,~yes,yes,z+3,~yes)
        return tape,outcome
    def record(self,t,o,active):
        b=len(t.count);z=j.zeros(b,j.int32)
        return d.record_on_tape(t,o,z+91,z,z+2,z+4,eligible=~active,active=active)
    def test_exact_twelve_operands_stamps_prior_bits_b8_inactive_and_overflow(self):
        t,o=self.tape();active=j.array([True,False,True,False,True,False,True,False])
        a=self.record(t,o,active)
        np.testing.assert_array_equal(a.words,t.words);np.testing.assert_array_equal(a.refusal,t.refusal)
        self.assertIs(a.execution.ledger,t.execution.ledger);self.assertIs(a.execution.risks,t.execution.risks)
        self.assertEqual(a.execution.diagnostics.count.tolist(),[1,0,1,0,1,0,1,0])
        records=d.refused_rows(a.execution.diagnostics,np.array([True,False,False,False,False,False,False,False]))['rows']
        self.assertEqual(len(records),1);r=records[0]['records'][0]
        for key,value in zip(d.OUTCOME_FIELDS,o):self.assertEqual(r[key],int(value[0]))
        self.assertEqual((r['stream_id'],r['frame'],r['raw_count'],r['turn'],r['generation0'],r['generation1']),(12001,5,7,1,2,3))
        self.assertEqual((r['prior_tape_refusal'],r['prior_execution_refusal'],r['prior_kernel_risk']),(30,10,20))
        b=self.record(a,o,active);c=self.record(b,o,active)
        self.assertEqual(c.execution.diagnostics.count.tolist(),[2,0,2,0,2,0,2,0])
        self.assertEqual(c.execution.diagnostics.overflow.tolist(),active.tolist())
        np.testing.assert_array_equal(c.execution.diagnostics.words,b.execution.diagnostics.words)
        np.testing.assert_array_equal(c.refusal,t.refusal);np.testing.assert_array_equal(c.words,t.words)
        for field in d.Capture._fields:np.testing.assert_array_equal(getattr(c.execution.diagnostics,field)[1::2],getattr(t.execution.diagnostics,field)[1::2])
    def test_hot_jit_and_missing_malformed_capture_are_inert(self):
        t,o=self.tape();C=namedtuple('Cursor','turn phase generation');t=t._replace(cursor=C(t.cursor.turn,t.cursor.phase,t.cursor.generation))
        L=namedtuple('Ledger','refusal frame');R=namedtuple('Risk','carry');Carry=namedtuple('Carry','risk')
        t=t._replace(execution=t.execution._replace(ledger=L(t.execution.ledger.refusal,t.execution.ledger.frame),risks=R(Carry(t.execution.risks.carry.risk))))
        fast=jax.jit(self.record);active=j.ones(8,bool)
        for _ in range(2):
            got=fast(t,o,active);self.assertEqual(got.execution.diagnostics.count.tolist(),[1]*8)
        for bad in (None,object(),d.empty(1),d.empty(8)._replace(words=j.zeros((8,2,len(d.FIELDS)),j.float32))):
            before=t._replace(execution=t.execution._replace(diagnostics=bad));self.assertIs(self.record(before,o,active),before)
        bad=t._replace(execution=t.execution._replace(risks=None));self.assertIs(self.record(bad,o,active),bad)
        self.assertEqual(d.refused_rows(None,np.array([False,True]))['rows'],[dict(row=1,available=False)])
    def test_refused_only_serializer_hoists_host_copies_without_device_row_slices(self):
        class CountCopies:
            def __init__(self,values):self.values=values;self.shape=values.shape;self.dtype=values.dtype;self.copies=0
            def __array__(self,dtype=None,copy=None):self.copies+=1;return np.asarray(self.values,dtype=dtype)
            def __getitem__(self,index):raise AssertionError('device row slice/gather')
        t,o=self.tape();a=self.record(t,o,j.ones(8,bool)).execution.diagnostics
        wrapped=d.Capture(*(CountCopies(np.asarray(v))for v in a))
        out=d.refused_rows(wrapped,np.array([False,True,False,False,False,False,False,True]))
        self.assertEqual([v['row']for v in out['rows']],[1,7]);self.assertTrue(all(v['available']for v in out['rows']))
        self.assertEqual(d.refused_rows(wrapped,np.zeros(8,bool))['rows'],[])
        self.assertTrue(all(value.copies==1 for value in wrapped))
        for mask in (np.array([1,0]),np.array([[True]])):
            with self.assertRaises(ValueError):d.refused_rows(a,mask)
    def test_observation_is_not_used_by_admission_or_policy(self):
        source=Path('gpu_engine/engine_source_frame_witness.py').read_text()
        body=source[source.index('def consume_pair_device'):source.index('def commit_rows')]
        before,after=body.split('report=Report(',1)
        self.assertNotIn('outcome_diagnostics',before)
        self.assertIn('tape.execution.diagnostics',after)
        for name in ('_network','_admission','_decide'):
            import ast,subprocess
            def fn(text):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)
            self.assertEqual(fn(Path('gpu_engine/benchmark_retained_games.py').read_text()),fn(subprocess.check_output(['git','show','7c6c6042:gpu_engine/benchmark_retained_games.py'],text=True)))

    def test_benchmark_keeps_refused_only_original_frame_data(self):
        from gpu_engine import benchmark_retained_games as bench,champion
        import test_gpu_engine_benchmark_retained_games as fixture
        Report=namedtuple('SyntheticReport','raw_count outcome_diagnostics')
        first,config=fixture.result(reasons=np.array([[256,0],[0,0],[0,0]]))
        tape,outcome=self.tape(3);capture=self.record(tape,outcome,j.ones(3,bool)).execution.diagnostics
        first.state.core.frame_input_report=Report(j.array([11,12,13]),capture)
        def network(_,args):
            n=len(args[0]);return champion.Output(j.zeros((n,9)),j.zeros(n),j.zeros((n,9))),0.,1.
        def advance(session,engine_actions,*,active_rows,timing_callback):
            result,_=fixture.result(frames=[0,1,1],done=[False,True,False],reasons=np.array([[256,0],[256,256],[0,0]]))
            result.state.core.frame_input_report=Report(j.array([11,21,22]),capture)
            session.state,session.result=result.state,result
            session.diagnostics=NS(active_rows=active_rows,executed_rows=active_rows,invalid_actions=j.zeros(3,bool),placeholder_moves=j.zeros(3,j.int32))
            for name in ('engine','requests','public_p1','public_p2','encode_p1','encode_p2','admission_p1','admission_p2'):
                timing_callback(name,.001)
            return result
        measured,model,advancing=fixture.RetainedBenchmarkTests().measure(first,config,network=network,advance=advance,cap=1)
        self.assertEqual((measured['completed_games'],measured['refused_games'],measured['censored_games']),(1,1,1))
        self.assertEqual([row['row']for row in measured['refusal_outcomes']['rows']],[0])
        self.assertEqual(measured['refusal_outcomes']['rows'][0]['records'][0]['raw_count'],7)
        for key in ('initial_frame_input_witness','final_frame_input_witness'):
            self.assertNotIn('outcome_diagnostics',measured[key])
        self.assertEqual(measured['final_frame_input_witness']['raw_count'],[11,21,22])
        self.assertEqual(model.call_count,1);self.assertEqual(advancing.call_count,1)
