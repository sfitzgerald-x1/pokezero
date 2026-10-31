"""Exact original PRIMARY status words; continuation contributes no duplicates."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_toxic_callbacks as toxic,engine_lifecycle_callbacks as life
from gpu_engine import engine_full_v4_event_device as full,engine_public_events as events
from gpu_engine import engine_execution_completeness as execution,public_wire,v4_encoder as v4
from gpu_engine.state import empty_state
import test_gpu_engine_toxic_runtime as fixtures

class ToxicSuccessFullWordsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ToxicRuntimeTests.setUpClass();cls.f=fixtures.ToxicRuntimeTests;cls.r=cls.f.runtime
        cls.fast=staticmethod(jax.jit(lambda stream,tape,before,after:
            full.project_device(stream,tape,before,after,cls.r['public'].device)))

    @staticmethod
    def words(p,row):
        return np.asarray(p.frames.words).transpose(1,0,2,3)[row][np.asarray(p.frames.valid).transpose(1,0,2)[row]]

    def test_status_write_emits_original_status_once_and_pair_fault_rolls_back(self):
        producer=self.r['request']['lifecycle'];b=2
        cursor=life.register(j.array([9811,9812],j.int32),catalog=producer,expected_binding=producer.binding,
            expected_engine_binding=np.array(producer.engine_binding,np.uint32))
        state=empty_state(b,seed=31)._replace(species=j.ones((b,2,6),j.int32).at[:,1].set(2).at[:,1,0].set(1),
            level=j.full((b,2,6),80,j.int32),gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),
            max_hp=j.full((b,2,6),100,j.int32),active=j.tile(j.array([3,5],j.int32),(b,1)))
        before=life.snapshot(cursor);tape=life._tape(cursor,producer,64)
        for side in (0,1):
            tape=life._reveal(tape,state,j.full(b,side,j.int32),j.full(b,-1,j.int32),j.full(b,0,j.int32) if side==1 else state.active[:,side],
                eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
        tape=tape._replace(cursor=tape.cursor._replace(initialized=j.ones(b,bool)));tape=life._turn(tape,j.ones(b,bool))
        # A second distinct public encounter must not be mistaken for a side.
        tape=life._reveal(tape,state,j.ones(b,j.int32),j.zeros(b,j.int32),j.full(b,5,j.int32),
            eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=False)
        stream=full.register(tape.stream_id,j.array([0,1],j.int32),compiled=self.r['public'])
        initial=self.fast(stream,tape,before,life.snapshot(tape.cursor));self.assertEqual(initial.error.tolist(),[0,0])
        names=[self.f.catalog.species.names[int(self.r['public'].device.causal.species[i])] for i in (1,2)]
        prefix=[f'|switch|p1a: {names[0]}|{names[0]}, L80, M|100/100',
            f'|switch|p2a: {names[0]}|{names[0]}, L80, M|100/100','|turn|1',
            f'|switch|p2a: {names[1]}|{names[1]}, L80, M|100/100']
        wire,_=public_wire.project_lines(public_wire.WireState(),prefix,**v4.wire_kwargs(self.f.catalog))
        for code in (toxic.STATUS_WRITE,):
            current=life._tape(tape.cursor,producer,64);side=j.zeros(b,j.int32);actor=j.full(b,3,j.int32);target=j.full(b,5,j.int32)
            move=j.full(b,self.f.move_id,j.int32);current=events.move(current,side,actor,target,move,eligible=j.ones(b,bool))
            flags=[False,False,False,False,False,True,False,False,False,False,False]
            flags[0]=flags[1]=flags[6]=flags[10]=True
            attempt=toxic.Attempt(*[j.full(b,x,bool) for x in flags],j.zeros(b,j.int32),j.full(b,4,j.int32),j.zeros(b,j.int32),j.zeros(b,j.int32))
            account=self.r['execution_compiled'];ledger=execution.initialize(tape.stream_id,compiled=account,capacity=16)
            current=execution.begin_on_tape(current,ledger,account.device,active=j.ones(b,bool))
            from gpu_engine import engine_status_callbacks as status
            current=status.changed(current,1-side,target,attempt.old_status,attempt.new_status,status.PRIMARY,eligible=j.ones(b,bool))
            current=toxic.record_device(current,side,actor,target,move,attempt,self.r['toxic_context'],eligible=j.ones(b,bool))
            p=self.fast(initial.state,current,life.snapshot(tape.cursor),life.snapshot(current.cursor))
            self.assertEqual(p.error.tolist(),[0,0])
            lines=[f'|move|p1a: {names[0]}|Toxic|p2a: {names[1]}',f'|-status|p2a: {names[1]}|tox']
            _,ps=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.f.catalog))
            expected=np.array([e.words() for projection in ps for e in projection.events],np.int32)
            for row in range(b):np.testing.assert_array_equal(self.words(p,row),expected)
            self.assertFalse(bool(p.B4_qualified.any()))
            # A repeated continuation has no new primary write or declaration.
            duplicate=toxic.record_device(current,side,actor,target,move,attempt,self.r['toxic_context'],eligible=j.array([True,False]))
            bad=self.fast(initial.state,duplicate,life.snapshot(tape.cursor),life.snapshot(duplicate.cursor))
            self.assertNotEqual(int(bad.error[0]),0);self.assertEqual(int(bad.error[1]),0)
            keep=j.array([False,True])
            def rows(old,new):
                if getattr(old,'shape',()) and old.shape[0]==b:
                    return j.where(keep.reshape((b,)+(1,)*(old.ndim-1)),new,old)
                return old
            expected_state=jax.tree.map(rows,initial.state,p.state)
            expected_state=expected_state._replace(causal=expected_state.causal._replace(admitted=keep,
                cursor=expected_state.causal.cursor._replace(admitted=keep)))
            for old,new in zip(jax.tree.leaves(expected_state),jax.tree.leaves(bad.state)):np.testing.assert_array_equal(old,new)
