"""Exact pinned original FULL words for two literal no-effect branches."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_toxic_callbacks as toxic,engine_lifecycle_callbacks as life
from gpu_engine import engine_full_v4_event_device as full,engine_public_events as events
from gpu_engine import engine_execution_completeness as execution,public_wire,v4_encoder as v4
from gpu_engine.state import empty_state
import test_gpu_engine_toxic_runtime as fixtures

class ToxicFullWordsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ToxicRuntimeTests.setUpClass();cls.f=fixtures.ToxicRuntimeTests;cls.r=cls.f.runtime
        cls.fast=staticmethod(jax.jit(lambda stream,tape,before,after:
            full.project_device(stream,tape,before,after,cls.r['public'].device)))

    @staticmethod
    def words(p,row):
        return np.asarray(p.frames.words).transpose(1,0,2,3)[row][np.asarray(p.frames.valid).transpose(1,0,2)[row]]

    def test_both_owners_nonphysical_target_exact_miss_and_type_immunity_words(self):
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
        for code in (toxic.MISS,toxic.PLAIN_TYPE_IMMUNITY):
            current=life._tape(tape.cursor,producer,64);side=j.zeros(b,j.int32);actor=j.full(b,3,j.int32);target=j.full(b,5,j.int32)
            move=j.full(b,self.f.move_id,j.int32);current=events.move(current,side,actor,target,move,eligible=j.ones(b,bool))
            flags=[False,False,False,False,False,True,False,False,False,False,False]
            if code==toxic.PLAIN_TYPE_IMMUNITY:flags[0]=flags[1]=flags[2]=flags[10]=True
            attempt=toxic.Attempt(*[j.full(b,x,bool) for x in flags],*[j.zeros(b,j.int32) for _ in range(4)])
            account=self.r['execution_compiled'];ledger=execution.initialize(tape.stream_id,compiled=account,capacity=16)
            current=execution.begin_on_tape(current,ledger,account.device,active=j.ones(b,bool))
            current=toxic.record_device(current,side,actor,target,move,attempt,self.r['toxic_context'],eligible=j.ones(b,bool))
            p=self.fast(initial.state,current,life.snapshot(tape.cursor),life.snapshot(current.cursor))
            self.assertEqual(p.error.tolist(),[0,0])
            lines=[f'|move|p1a: {names[0]}|Toxic|p2a: {names[1]}'+('|[miss]' if code==toxic.MISS else ''),
                f'|-miss|p1a: {names[0]}|p2a: {names[1]}' if code==toxic.MISS else f'|-immune|p2a: {names[1]}']
            _,ps=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.f.catalog))
            expected=np.array([e.words() for projection in ps for e in projection.events],np.int32)
            for row in range(b):np.testing.assert_array_equal(self.words(p,row),expected)
            self.assertFalse(bool(p.B4_qualified.any()))
