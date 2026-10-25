"""Transport/capture contracts, not playing-strength or cost qualification."""
import json
import os
from pathlib import Path
import random
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.exact_session import FixedShapeModel,NativeSessionBackend
from pokezero.failure_capture import CaptureFailed,Quarantine,decision_scope,current
from pokezero.shallow_exact import ExactSearchRefused

SPEC=dict(window=1,tokens=1,categorical_features=1,numeric_features=1)

class Model:
    def __init__(self):self.calls=[];self.barriers=0;self.fail=False
    def synchronize(self):self.barriers+=1
    def eval_obs_packed(self,count,*columns,byte_order):
        self.calls.append((count,columns,byte_order))
        if self.fail:raise RuntimeError('authored model failure')
        values=list(struct.unpack('<'+'d'*count,columns[1]))
        return values,[0.]*(count*9),[1/9]*(count*9),9
    def drain_transport_timings(self):return (0,0.,0,0.)

class Session:
    def __init__(self,*args):self.calls=[];self.reconstructed=[]
    def root_terminal(self):return None
    def root_actions(self):return ['move'],['opponent']
    def enumerate_cell(self,*args):return 0,[(.4,None),(.6,None)],[]
    def encode_packed(self,addresses,workers):
        self.calls.append((addresses,workers))
        n=len(addresses)
        return dict(batch_rows=n,categorical_ids=struct.pack('<'+'i'*n,*[b for _,b in addresses]),
            numeric_features=struct.pack('<'+'d'*n,*[b*.5 for _,b in addresses]),
            token_type_ids=bytes(2*n),attention_mask=b'\x01'*n)
    def reconstruct(self,cell,branch):
        self.reconstructed.append((cell,branch))
        return json.dumps(dict(engine_state='root',post_state='post-'+str(branch),
            instruction_list=['authored-'+str(branch)]))

class ExactSessionContracts(unittest.TestCase):
    def test_fixed_shape_warms_three_times_then_preserves_real_rows_and_discards_padding(self):
        raw=Model();model=FixedShapeModel(raw,SPEC,4)
        self.assertEqual(len(raw.calls),3)
        self.assertEqual(raw.barriers,6)
        self.assertTrue(all(row['real_rows']==0 and row['model_rows']==4 for row in model.warmup))
        for count in (1,3,4):
            values=[.125*i for i in range(count)]
            cols=(bytes(count*4),struct.pack('<'+'d'*count,*values),bytes(count*2),b'\x01'*count,b'\x01'*count)
            actual=model.eval_obs_packed(count,*cols,byte_order='little')
            self.assertEqual(actual[0],values)
            self.assertEqual(len(actual[1]),count*9)
            self.assertEqual(len(actual[2]),count*9)
            n,encoded,_=raw.calls[-1]
            self.assertEqual(n,4)
            for src,full,dummy in zip(cols,encoded,model.dummy):
                self.assertEqual(full,src+dummy*(4-count))
        self.assertEqual(raw.barriers,12)

    def test_malformed_inputs_and_outputs_never_return_partial_values(self):
        raw=Model();model=FixedShapeModel(raw,SPEC,2)
        with self.assertRaises(ExactSearchRefused):model.eval_obs_packed(3,*[b'']*5,byte_order='little')
        with self.assertRaises(ExactSearchRefused):model.eval_obs_packed(1,*[b'']*5,byte_order='little')
        raw.eval_obs_packed=lambda *a,**k:([0.],[],[],9)
        with self.assertRaises(ExactSearchRefused):model.eval_obs_packed(0,*[b'']*5,byte_order='little')
        self.assertFalse(model.last_call['returned'])

    def test_forward_failure_stays_primary_when_post_barrier_also_fails(self):
        raw=Model();model=FixedShapeModel(raw,SPEC,2)
        failure=RuntimeError('original forward');failure.pokezero_failure_witness='original witness'
        def forward(*a,**k):raise failure
        raw.eval_obs_packed=forward
        def synchronize():
            raw.barriers+=1
            if raw.barriers==8:raise ValueError('secondary barrier')
        raw.synchronize=synchronize
        with self.assertRaises(RuntimeError) as raised:
            model.eval_obs_packed(0,*[b'']*5,byte_order='little')
        self.assertIs(raised.exception,failure)
        self.assertEqual(raised.exception.pokezero_failure_witness,'original witness')
        self.assertEqual(model.last_call['timing_status'],'UNAVAILABLE_SYNCHRONIZATION')
        self.assertIsNone(model.last_call['synchronized_wrapper_seconds'])
        self.assertEqual(model.last_call['synchronization_error']['message'],'secondary barrier')

    def fixture(self):
        native=SimpleNamespace(ExactSession=Session)
        raw=Model();model=FixedShapeModel(raw,SPEC,4)
        backend=NativeSessionBackend(native,model,packed_leaves=True,bulk_encode_workers=2,
            max_outcomes_per_call=100,observation_spec=SPEC)
        root=SimpleNamespace(lines=(),self_side='side_two',party_species={'p1':['a'],'p2':['b']},
            turn=3,state_str='root',fold=SimpleNamespace(to_payload=lambda:{'root':'fold'}),
            origin={'world_ordinal':0},encoder=object())
        backend.prepare_roots([root])
        return root,raw,backend

    def test_native_batches_keep_order_duplicates_and_value_frame_without_capture_copies(self):
        with patch.dict(os.environ,POKEZERO_ENUMERATE_ROLLS='1'):
            root,raw,b=self.fixture()
            leaves=[x.state for x in b.outcomes(root,'move','opponent')]
            ordered=[leaves[1],leaves[0],leaves[1]]
            self.assertEqual(b.evaluate_leaves(ordered),[-.5,-0.,-.5])
            self.assertIs(b._last_batch,ordered)
            self.assertEqual(leaves[0].session.calls,[([(0,1),(0,0),(0,1)],2)])
            self.assertEqual(leaves[0].session.reconstructed,[])
            self.assertEqual(b.captured_leaf_rows,0)
            self.assertEqual(b.capture_context_updates,0)
            self.assertEqual(b.model_call_timings[0]['fixed_shape']['padded_rows'],1)

    def test_quarantine_reconstructs_entire_ordered_real_batch_and_padding_only_on_failure(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,POKEZERO_ENUMERATE_ROLLS='1'):
            q=Quarantine(d,build={'fingerprint':'authored'})
            ctx=SimpleNamespace(seed=17,player_id='p2')
            with self.assertRaisesRegex(RuntimeError,'authored model failure'):
                with decision_scope(q,ctx,random.Random(17)):
                    root,raw,b=self.fixture()
                    leaves=[x.state for x in b.outcomes(root,'move','opponent')]
                    raw.fail=True
                    b.evaluate_leaves([leaves[1],leaves[0],leaves[1]])
            bundle=json.loads(next(Path(d).glob('failure-*.json')).read_text())
            self.assertEqual(bundle['exact_roots'][0]['engine_state'],'root')
            batch=bundle['lazy_evidence']['native_one_ply']
            self.assertEqual([r['branch_index'] for r in batch['rows']],[1,0,1])
            self.assertEqual((batch['real_rows'],batch['model_rows'],batch['padded_rows']),(3,4,1))
            self.assertEqual(batch['rows'][0]['native']['post_state'],'post-1')
            self.assertEqual(q.successes,1)

    def test_deadline_before_forward_keeps_unsubmitted_pending_rows_without_warmup_attribution(self):
        from pokezero.one_ply_exact import exact_one_ply
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,POKEZERO_ENUMERATE_ROLLS='1'):
            q=Quarantine(d,build={});clock=[0.]
            with self.assertRaises(ExactSearchRefused):
                with decision_scope(q,SimpleNamespace(),random.Random(1)):
                    root,raw,b=self.fixture()
                    terminal=b.terminal_value
                    def tick(p):
                        result=terminal(p)
                        if getattr(p,'branch',None)==1:clock[0]=2.
                        return result
                    b.terminal_value=tick
                    with patch('pokezero.one_ply_exact.time.monotonic',side_effect=lambda:clock[0]):
                        exact_one_ply([root],b,batch_size=4,max_seconds=1,cross_cell_batches=True)
            batch=json.loads(next(Path(d).glob('failure-*.json')).read_text())['lazy_evidence']['native_one_ply']
            self.assertEqual(batch['kind'],'ordered_pending_batch')
            self.assertEqual([r['branch_index'] for r in batch['rows']],[0])
            self.assertIsNone(batch['model_call'])
            self.assertFalse(batch['model_invoked'])
            self.assertEqual(len(raw.calls),3,'only the declared startup warmup ran')

    def test_encode_failure_identifies_group_and_does_not_claim_an_earlier_forward(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,POKEZERO_ENUMERATE_ROLLS='1'):
            q=Quarantine(d,build={})
            with self.assertRaisesRegex(ValueError,'authored encode failure'):
                with decision_scope(q,SimpleNamespace(),random.Random(1)):
                    root,raw,b=self.fixture()
                    leaves=[x.state for x in b.outcomes(root,'move','opponent')]
                    def fail(*args):raise ValueError('authored encode failure')
                    leaves[0].session.encode_packed=fail
                    b.evaluate_leaves(leaves)
            batch=json.loads(next(Path(d).glob('failure-*.json')).read_text())['lazy_evidence']['native_one_ply']
            self.assertEqual(batch['stage'],'encoding')
            self.assertIsNone(batch['model_call'])
            self.assertFalse(batch['model_invoked'])
            self.assertEqual(batch['encode_group']['start'],0)
            self.assertEqual(batch['encode_group']['stop'],2)

    def test_lazy_provider_never_runs_on_success_and_capture_loss_stays_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            q=Quarantine(d,build={});ctx=SimpleNamespace();calls=[]
            with decision_scope(q,ctx,random.Random(1)):
                current().defer('unused',lambda:calls.append(True))
            self.assertEqual(calls,[])
            with self.assertRaises(CaptureFailed):
                with decision_scope(q,ctx,random.Random(1)):
                    def bad():raise RuntimeError('capture reconstruction failed')
                    current().defer('bad',bad)
                    raise ValueError('authored contract failure')
            self.assertEqual(q.failures,1)
