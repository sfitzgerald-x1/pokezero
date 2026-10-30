"""Actual kernel status writes and both genuine owner drains.

The Session birth/warm-up is real. Explicit component status pre-state and
original FULL status prefixes isolate this kernel-event integration; they do
not attest original Showdown status-hook history or physics equivalence.
"""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_retained_encoder_driver as d, engine_lifecycle_callbacks as life
from gpu_engine import engine_status_callbacks as statuses, engine_full_v4_event_device as full
from gpu_engine import engine_execution_completeness as e, mechanics as m, transition, public_mon_state
import test_gpu_engine_status_residual_runtime as fixtures


class StatusKernelOwnerDrainTests(unittest.TestCase):
    construct=fixtures.StatusResidualRuntimeTests.construct
    chance=fixtures.StatusResidualRuntimeTests.chance
    same=fixtures.StatusResidualRuntimeTests.same

    @classmethod
    def setUpClass(cls):
        fixtures.StatusResidualRuntimeTests.setUpClass();base=fixtures.StatusResidualRuntimeTests
        cls.f=base.f
        cls.prepared=d.prepare(*base.args,**dict(base.kwargs,status_residual_mode='kernel-event'))
        runtime=d._checked(cls.prepared)
        cls.seed_full=staticmethod(jax.jit(lambda stream,frame,public:full.project_and_fold(
            stream,frame,runtime['public'].device,public,runtime['catalog'])))

    def component_before(self):
        session=self.construct();r=d._checked(self.prepared)
        warm=d.advance(session,j.tile(j.array([[0,8]],j.int32),(2,1)),chance_inputs=self.chance())
        self.assertEqual(warm.state.core.admitted.tolist(),[True,True])
        before=warm.state;core=before.core
        self.assertEqual(core.engine.active.tolist(),[[0,5],[0,5]])
        engine=core.engine
        tape=life._tape(core.cursor,r['request']['lifecycle'],64)
        for side,kind in ((0,m.BRN),(1,m.PSN)):
            physical=engine.active[:,side];rows=j.arange(2)
            old=engine.status[rows,side,physical]
            self.assertEqual(old.tolist(),[m.NONE,m.NONE])
            tape=statuses.changed(tape,j.full(2,side,j.int32),physical,old,j.full(2,kind,j.int32),
                statuses.PRIMARY,eligible=j.ones(2,bool))
            engine=engine._replace(status=engine.status.at[rows,side,physical].set(kind))
        prefix=life.LifecycleResult(engine,tape.cursor,tape,life.snapshot(core.cursor),
            life.snapshot(tape.cursor),j.zeros(2,bool),j.zeros(2,j.int32))
        observers=[]
        for observer in before.observers:
            # Original full projection and all real public consumers. No emitter
            # patch, aggregate-bit removal or invented consumption result.
            result=self.seed_full(observer.projection,prefix,observer.retained)
            self.assertEqual(result.projection.error.tolist(),[0,0])
            observers.append(observer._replace(projection=result.projection.state,
                retained=result.public,retained_healthy=observer.retained_healthy))
        before=before._replace(core=core._replace(engine=engine),observers=tuple(observers))
        return r,before

    def actual_frame(self,r,before):
        actions=j.zeros((2,2),j.int32)
        frame=r['frame'](before.core,actions,self.chance(),j.ones(2,bool))
        ordinary=transition.step_with_chance(before.core.engine,actions,r['request']['engine_table'],self.chance())
        self.same(frame.state,ordinary.state)
        self.assertEqual(frame.callbacks.refusal.tolist(),[1,1])
        ledger=frame.callbacks.execution.ledger
        for row in range(2):
            selected=(j.arange(ledger.site.shape[1])<ledger.count[row])&(ledger.site[row]==int(e.Site.STATUS_RESIDUAL))
            self.assertEqual(ledger.stage[row][selected].tolist(),[2,2])
            raw=np.asarray(frame.callbacks.words[row,:int(frame.callbacks.count[row])])
            actual=raw[raw[:,0]==20]
            self.assertEqual(actual[:,4].tolist(),[5,0])
            self.assertEqual(actual[:,8].tolist(),[m.PSN,m.BRN])
            self.assertTrue(bool((actual[:,5]>actual[:,6]).all()))
            self.assertTrue(bool((actual[:,6]>0).all()))
        self.assertFalse(bool(e.report(ledger,tables=r['execution_compiled'].device).scoped_drained.any()))
        # Same trusted requests-stage input used by the real consume path.
        core=before.core._replace(engine=frame.state,cursor=frame.cursor,
            weather_profile=frame.weather_profile,execution=ledger)
        return frame,before._replace(core=core)

    def final_hp(self,result):
        state=result.state.core.engine
        for owner,observer in enumerate(result.state.observers):
            self.assertEqual(observer.retained_healthy.tolist(),[True,True])
            for row in range(2):
                for side in (0,1):
                    physical=int(state.active[row,side]);slot=int(observer.retained.mons.active[row,side])
                    hp=int(state.hp[row,side,physical]);maximum=int(state.max_hp[row,side,physical])
                    visible=hp if owner==side else min(99,(100*hp+maximum-1)//maximum)
                    self.assertEqual(int(observer.retained.mons.hp_numerator[row,side,slot]),visible)
                    self.assertEqual(int(observer.retained.mons.hp_denominator[row,side,slot]),maximum if owner==side else 100)
                    self.assertEqual(int(observer.retained.mons.status[row,side,slot]),public_mon_state.BURN if side==0 else public_mon_state.POISON)

    def test_actual_kernel_status_spans_drain_both_original_owner_consumers(self):
        r,before=self.component_before();frame,pending=self.actual_frame(r,before)
        result=d._consume(r,pending,frame,j.ones(2,bool))
        ledger=result.state.core.execution;tables=r['execution_compiled'].device
        self.assertEqual(result.state.core.admitted.tolist(),[True,True])
        self.assertEqual(e.report(ledger,tables=tables).scoped_drained.tolist(),[True,True])
        for row in range(2):
            present=j.arange(ledger.stage.shape[1])<ledger.count[row]
            self.assertTrue(bool((ledger.stage[row][present]==3).all()))
        self.same(result.state.observers[0].accounting,result.state.observers[1].accounting)
        self.final_hp(result)
        for owner in result.owners:
            self.assertFalse(bool(owner.full_line_consumers_complete.any()))
            self.assertFalse(bool(owner.full_encoder_admitted.any()))
            self.assertFalse(bool(owner.B4_qualified.any()))
            self.assertFalse(bool(owner.scored.any()))

    def test_one_row_second_owner_failure_rolls_back_both_and_does_not_drain(self):
        r,before=self.component_before();frame,pending=self.actual_frame(r,before)
        actual_public,actual_encode=r['public_fast'],r['encode_fast'];calls=[]
        def public(observer,frame,profile,active):
            calls.append(int(observer.owner[0]))
            if int(observer.owner[0])==1:
                observer=observer._replace(retained=observer.retained._replace(cursor=observer.retained.cursor._replace(
                    sequence=observer.retained.cursor.sequence.at[0].add(1))))
            return actual_public(observer,frame,profile,active)
        def encode(core,observer):
            self.assertEqual(calls[:2],[0,1])
            self.assertEqual(core.admitted.tolist(),[False,True])
            return actual_encode(core,observer)
        r.update(public_fast=public,encode_fast=encode)
        try:result=d._consume(r,pending,frame,j.ones(2,bool))
        finally:r.update(public_fast=actual_public,encode_fast=actual_encode)
        ledger=result.state.core.execution
        self.assertEqual(result.state.core.admitted.tolist(),[False,True])
        for row in (0,1):
            present=j.arange(ledger.stage.shape[1])<ledger.count[row]
            self.assertTrue(bool((ledger.stage[row][present]==(2 if row==0 else 3)).all()))
        self.assertTrue(int(ledger.refusal[0])&e.CONSUMPTION)
        self.assertEqual(int(ledger.refusal[1]),0)
        for old,new in zip(before.observers,result.state.observers):
            self.same(jax.tree.map(lambda x:x[0],old.projection),jax.tree.map(lambda x:x[0],new.projection))
            self.same(d._public_rows(j.array([False,True]),old.retained,new.retained),new.retained)
            self.assertEqual(new.retained_healthy.tolist(),[False,True])
        for owner in result.owners:
            self.assertFalse(bool(owner.encoded.observation.attention_mask[0].any()))
            self.assertFalse(bool(owner.own.admitted[0]))
            self.assertFalse(bool(owner.scored.any()))
            self.assertFalse(bool(owner.B4_qualified.any()))


if __name__=='__main__':unittest.main()
