"""Status residual writes through both real owner drains; eager actual frame, no models/games.

Light replacement for status_kernel_owner_drain and status_residual_runtime. The prepared runtimes, the
constructor and the requests/public/encode/score phases are the real ones, each compiled once per class.
Only the instrumented step runs eagerly: the prepared frame closure's own unjitted body
(``r['frame'].__wrapped__``), with its row selection and global-binding exclusion. The compiled frame
stays covered end to end by the heavy owning_drain_end_to_end module. Explicit component status
pre-state and FULL status prefixes isolate the kernel-event integration; they do not attest original
Showdown status-hook history or physics equivalence (the effect probes cover burn and poison physics).
"""
from contextlib import contextmanager
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_retained_encoder_driver as d, engine_lifecycle_callbacks as life
from gpu_engine import engine_status_callbacks as statuses, engine_status_residual_callbacks as status
from gpu_engine import engine_full_v4_event_device as full, engine_execution_completeness as e
from gpu_engine import engine_request_source as source
from gpu_engine import mechanics as m, transition, public_mon_state
import gpu_engine_owning_runtime_fixture as fixtures
import test_gpu_engine_leftovers_registered_runtime as heal_fixture


class StopConstruct(Exception):
    pass


class OwningStatusDrainTests(unittest.TestCase):
    construct=fixtures.ExecutionRuntimePlumbingTests.construct
    chance=fixtures.ExecutionRuntimePlumbingTests.chance
    same=fixtures.ExecutionRuntimePlumbingTests.same
    check_final_public_hp=heal_fixture.RegisteredLeftoversRuntimeTests.check_final_public_hp

    @classmethod
    def setUpClass(cls):
        # Host-only: the same three prepared runtimes as the heavy modules (plain accounting,
        # diagnostic status, kernel-event status). Nothing is compiled here.
        fixtures.ExecutionRuntimePlumbingTests.setUpClass();base=fixtures.ExecutionRuntimePlumbingTests
        cls.f=base.f;cls.plain=base.prepared;r=d._checked(cls.plain)
        cls.helper=status.compile_status_residual(json.loads(r['request']['metadata']),
            engine_catalog=r['request']['lifecycle'])
        cls.args=(r['request_compiled'],r['own_compiled'])
        cls.kwargs=dict(public_mode='retained-full',leftovers_compiled=r['leftovers_compiled'],
            leftovers_mode='retained',account_source_sites=True,status_residual_compiled=cls.helper,
            fake_forward=lambda _:d.FakeScores(j.ones(9),j.array(1,j.int32)))
        cls.diagnostic=d.prepare(*cls.args,**cls.kwargs)
        cls.kernel=d.prepare(*cls.args,**dict(cls.kwargs,status_residual_mode='kernel-event'))
        cls.birth_cache=None;cls.heal_cache=None;cls.next_cache=None;cls.warm_cache=None

    # The heavy owning_drain_end_to_end module reuses this flow with the compiled frame.
    compiled_frame=False

    def frame_call(self,r,*args):
        return r['frame'](*args) if self.compiled_frame else r['frame'].__wrapped__(*args)

    @contextmanager
    def frame_phase(self,r,frames):
        # The real advance; its frame phase runs eagerly (or compiled, in the heavy module) and is
        # recorded. Every other phase is the prepared runtime's compiled closure.
        compiled=r['frame']
        def capture(*args):
            frame=self.frame_call(dict(frame=compiled),*args);frames.append(frame);return frame
        r['frame']=capture
        try:yield
        finally:r['frame']=compiled

    def birth(self):
        # One real kernel-event construct per class: constructor, both owners' public folds, join,
        # encode and score. Shared read-only by every test; only heal_turn and next_turn advance its session.
        cls=type(self)
        if cls.birth_cache is None:
            session=self.construct(prepared=self.kernel)
            cls.birth_cache=(session,session.state,session.result)
        return cls.birth_cache

    def heal_turn(self):
        # The real Hail turn of #2's heal drain: the frame captures both Leftovers heals and the real
        # advance drains them through both owners. A switch installs side 1 at physical 5.
        cls=type(self)
        if cls.heal_cache is None:
            session,birth,_=self.birth();r=d._checked(self.kernel);frames=[]
            actions=j.tile(j.array([[0,8]],j.int32),(2,1))
            with self.frame_phase(r,frames):result=d.advance(session,actions,chance_inputs=self.chance())
            self.assertEqual(len(frames),1)
            cls.heal_cache=(birth,actions,frames[0],result)
        return d._checked(self.kernel),cls.heal_cache

    def next_turn(self):
        # The next real turn after the drained heal turn, through the same advance (requests phase,
        # session continuity, executed mask) and the same frame phase.
        cls=type(self)
        if cls.next_cache is None:
            r,(_,_,_,heal)=self.heal_turn();session,_,_=self.birth();frames=[]
            self.assertIs(session.result,heal)
            with self.frame_phase(r,frames):later=d.advance(session,j.zeros((2,2),j.int32),chance_inputs=self.chance())
            self.assertEqual(len(frames),1)
            cls.next_cache=(frames[0],later)
        return d._checked(self.kernel),cls.next_cache

    def warm(self):
        # #1's component_before: after the real warm (heal) turn, BRN (side 0) and PSN (side 1) are
        # injected and seeded through the original full projection of both owners.
        cls=type(self)
        if cls.warm_cache is None:
            r,(_,_,_,warm)=self.heal_turn()
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
            seed=jax.jit(lambda stream,frame,public:full.project_and_fold(
                stream,frame,r['public'].device,public,r['catalog']))
            observers=[]
            for observer in before.observers:
                # Original full projection and all real public consumers. No emitter patch,
                # aggregate-bit removal or invented consumption result.
                result=seed(observer.projection,prefix,observer.retained)
                self.assertEqual(result.projection.error.tolist(),[0,0])
                observers.append(observer._replace(projection=result.projection.state,
                    retained=result.public,retained_healthy=observer.retained_healthy))
            cls.warm_cache=before._replace(core=core._replace(engine=engine),observers=tuple(observers))
        return d._checked(self.kernel),cls.warm_cache

    def actual_frame(self,r,before):
        actions=j.zeros((2,2),j.int32)
        # The same argument structure as the advance's frame phase, so a compiled frame compiles once.
        frame=self.frame_call(r,before.core,actions,self.chance(),j.ones(2,bool),None,None)
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
                    self.assertEqual(int(observer.retained.mons.status[row,side,slot]),
                        public_mon_state.BURN if side==0 else public_mon_state.POISON)

    # execution_runtime_plumbing: the heal drain (its fixture is now tests/gpu_engine_owning_runtime_fixture.py)

    def test_actual_heal_turn_captures_and_drains_both_owner_leftovers_spans(self):
        r,(birth,actions,frame,result)=self.heal_turn();tables=r['execution_compiled'].device
        ordinary=transition.step_with_chance(birth.core.engine,actions,r['request']['engine_table'],self.chance())
        self.same(frame.state,ordinary.state)
        # UNINSTRUMENTED only: no UNKNOWN bit from an unhandled Leftovers ticket or any other site.
        self.assertEqual(frame.callbacks.refusal.tolist(),[1,1])
        ledger=frame.callbacks.execution.ledger;core=result.state.core
        self.assertEqual(ledger.frame.tolist(),[2,2])
        for row in range(2):
            # The turn's entries: one Leftovers entry per side and the kernel's turn-loop prefix (site
            # counts, not their order), every one captured.
            present=np.arange(ledger.site.shape[1])<int(ledger.count[row])
            sites=np.asarray(ledger.site[row])[present];stage=np.asarray(ledger.stage[row])[present]
            self.assertEqual(sorted(sites.tolist()),sorted([int(e.Site.LEFTOVERS)]*2+[int(e.Site.TURNLOOP_PREFIX)]))
            leftovers=sites==int(e.Site.LEFTOVERS)
            self.assertEqual(sorted(np.asarray(ledger.side[row])[present][leftovers].tolist()),[0,1])
            self.assertTrue(bool((stage==2).all()))
            # The kernel's own capture claims every emitted heal word (no test-side capture seam): two
            # distinct Leftovers entries, each on its heal word's side.
            words=np.asarray(frame.callbacks.words[row,:int(frame.callbacks.count[row])])
            heals=np.flatnonzero(words[:,0]==17);self.assertEqual(len(heals),2)
            slots=np.asarray(ledger.claimed[row])[heals]-1
            self.assertTrue(bool((slots>=0).all()));self.assertEqual(len(set(slots.tolist())),2)
            self.assertTrue(bool((np.asarray(ledger.site[row])[slots]==int(e.Site.LEFTOVERS)).all()))
            self.assertEqual(np.asarray(ledger.side[row])[slots].tolist(),words[heals,3].tolist())
            # Both owners drained every entry of the frame.
            self.assertEqual(int(core.execution.count[row]),int(ledger.count[row]))
            self.assertTrue(bool((np.asarray(core.execution.stage[row])[present]==3).all()))
        self.assertEqual(core.admitted.tolist(),[True,True])
        self.assertEqual(e.report(core.execution,tables=tables).scoped_drained.tolist(),[True,True])
        self.same(result.state.observers[0].accounting,result.state.observers[1].accounting)
        self.check_final_public_hp(result)
        # The next real turn opens ledger frame 3 and is admitted.
        _,(_,later)=self.next_turn()
        self.assertEqual(later.state.core.admitted.tolist(),[True,True])
        self.assertEqual(later.state.core.execution.frame.tolist(),[3,3])
        self.assertEqual(e.report(later.state.core.execution,tables=tables).scoped_drained.tolist(),[True,True])
        for owner in (*result.owners,*later.owners):
            self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))

    # status_kernel_owner_drain

    def test_actual_kernel_status_spans_drain_both_original_owner_consumers(self):
        r,before=self.warm();frame,pending=self.actual_frame(r,before)
        result=d._consume(r,pending,frame,j.ones(2,bool))
        ledger=result.state.core.execution;tables=r['execution_compiled'].device
        self.assertEqual(result.state.core.admitted.tolist(),[True,True])
        # The next actual turn after the drained heal turn opens frame 3.
        self.assertEqual(ledger.frame.tolist(),[3,3])
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
        r,before=self.warm();frame,pending=self.actual_frame(r,before)
        actual={name:r[name] for name in ('public_fast','encode_fast','score_fast')};calls=[]
        def public(observer,frame,profile,active):
            calls.append('public'+str(int(observer.owner[0])))
            if int(observer.owner[0])==1:
                observer=observer._replace(retained=observer.retained._replace(cursor=observer.retained.cursor._replace(
                    sequence=observer.retained.cursor.sequence.at[0].add(1))))
            return actual['public_fast'](observer,frame,profile,active)
        def encode(core,observer):
            self.assertEqual(calls[:2],['public0','public1'])
            self.assertEqual(core.admitted.tolist(),[False,True]);calls.append('encode')
            return actual['encode_fast'](core,observer)
        def score(*args):
            self.assertEqual(args[5].tolist(),[False,True]);calls.append('score')
            return actual['score_fast'](*args)
        r.update(public_fast=public,encode_fast=encode,score_fast=score)
        try:result=d._consume(r,pending,frame,j.ones(2,bool))
        finally:r.update(actual)
        # Both public folds finish before any encoding; no fake credit reaches the scorer.
        self.assertEqual(calls,['public0','public1','encode','score','encode','score'])
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

    # status_residual_runtime

    def test_original_host_guards_and_default_actual_frame_remain_intact(self):
        with self.assertRaises(ValueError):d.prepare(*self.args,**dict(self.kwargs,public_mode='basic'))
        with self.assertRaises(ValueError):d.prepare(*self.args,**dict(self.kwargs,status_residual_compiled=replace(self.helper)))
        for call in (lambda:jax.jit(lambda:d.prepare(*self.args,**self.kwargs).binding)(),
                     lambda:jax.jit(lambda dummy:d.prepare(*self.args,**self.kwargs).binding)(j.array(0))):
            with self.assertRaisesRegex(ValueError,'outside outer JIT'):call()
        # The status helper never reaches the actual constructor: the diagnostic and plain prepared
        # drivers construct the same engine (the real constructor, stopped before both owners' folds).
        constructed=[];actual=source.construct_from_pool
        def stop(*args,**kwargs):
            constructed.append(actual(*args,**kwargs));raise StopConstruct
        with patch.object(source,'construct_from_pool',side_effect=stop):
            for prepared in (self.diagnostic,self.plain):
                with self.assertRaises(StopConstruct):self.construct(prepared=prepared)
        self.same(constructed[0].state,constructed[1].state)
        # With no BRN/PSN/TOX present the diagnostic frame is the plain accounting frame, bit for bit.
        _,birth,_=self.birth()
        actions=j.full((2,2),8,j.int32)
        frame=self.frame_call(d._checked(self.diagnostic),birth.core,actions,self.chance())
        ordinary=self.frame_call(d._checked(self.plain),birth.core,actions,self.chance())
        self.same(frame,ordinary)
        # advance revalidates the status helper before any device phase: the guard precedes every closure.
        record=dict(advancing=False)
        with patch.object(d,'_session',return_value=record),\
                patch.object(status,'check_compiled',side_effect=ValueError('status metadata changed')):
            with self.assertRaisesRegex(ValueError,'status metadata changed'):
                d.advance(SimpleNamespace(prepared=self.diagnostic),actions,chance_inputs=self.chance())
        self.assertFalse(record['advancing'])

    def test_internal_actual_status_write_stays_explicitly_unknown(self):
        _,birth,result=self.birth();core=birth.core;r=d._checked(self.diagnostic)
        # Explicit INTERNAL fault fixture, not a sealed owning/public history:
        # it tests the real kernel's literal diagnostic write, never admission.
        engine=core.engine._replace(status=core.engine.status.at[:,0,0].set(m.BRN))
        core=core._replace(engine=engine)
        actions=j.tile(j.array([[0,8]],j.int32),(2,1))
        frame=self.frame_call(r,core,actions,self.chance())
        legacy=transition.step_with_chance(engine,actions,r['request']['engine_table'],self.chance())
        self.same(frame.state,legacy.state)
        for row in range(2):
            raw=np.asarray(frame.callbacks.words[row,:int(frame.callbacks.count[row])])
            writes=raw[raw[:,0]==status.STATUS_RESIDUAL]
            self.assertEqual(writes.shape,(1,10));write=writes[0]
            self.assertEqual(write[1:5].tolist(),[0,0,0,0])
            self.assertEqual(int(write[6]),int(frame.state.hp[row,0,0]))
            self.assertEqual(int(write[7]),int(engine.max_hp[row,0,0]))
            self.assertEqual(write[8:].tolist(),[m.BRN,status.BURN])
            self.assertGreater(int(write[5]),int(write[6]))
            self.assertTrue(int(frame.callbacks.refusal[row])&status.UNKNOWN_STATUS_RESIDUAL)
            self.assertTrue(int(frame.callbacks.refusal[row])&512)
            self.assertNotEqual(int(frame.callbacks.execution.ledger.refusal[row]),0)
        for owner in result.owners:
            self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))

    def test_explicit_kernel_event_preparation_requires_original_helper_accounting_and_fresh_host_guards(self):
        options=dict(self.kwargs,status_residual_mode='kernel-event')
        prepared=d.prepare(*self.args,**options);runtime=d._checked(prepared)
        self.assertIs(type(runtime['status_residual_context']),status.RegisteredStatusResidualContext)
        self.assertIs(runtime['status_residual_context'].data,self.helper.device)
        self.assertIs(runtime['status_residual_context'].weather_profile_data,runtime['speed_profile'].device)
        self.assertNotEqual(prepared.binding,self.diagnostic.binding)
        for change in (dict(account_source_sites=False),dict(status_residual_compiled=None),
                       dict(public_mode='basic'),dict(status_residual_mode='implicit')):
            with self.assertRaises(ValueError):d.prepare(*self.args,**dict(options,**change))
        for call in (lambda:jax.jit(lambda:d.prepare(*self.args,**options).binding)(),
                     lambda:jax.jit(lambda dummy:d.prepare(*self.args,**options).binding)(j.array(0))):
            with self.assertRaisesRegex(ValueError,'outside outer JIT'):call()
        with patch.object(status,'check_compiled',side_effect=ValueError('status original changed')):
            with self.assertRaisesRegex(ValueError,'status original changed'):d._checked(prepared)

    def test_prepared_contexts_reach_the_actual_frame_and_both_owner_consumers(self):
        # Wiring of each prepared runtime into its unjitted frame and public closures; no trace.
        sentinel=RuntimeError('wiring')
        for prepared,expected in ((self.diagnostic,life.StatusResidualContext),
                                  (self.kernel,status.RegisteredStatusResidualContext)):
            r=d._checked(prepared);seen={}
            def frame(*args,**kwargs):seen['frame']=kwargs;raise sentinel
            def public(*args,**kwargs):seen['public']=kwargs;raise sentinel
            with patch.object(d,'_frame',side_effect=frame),patch.object(d,'_full_public',side_effect=public):
                with self.assertRaises(RuntimeError):r['frame'].__wrapped__(None,None,None)
                with self.assertRaises(RuntimeError):r['public_fast'].__wrapped__(None,None,None)
            context=seen['frame']['status_residual_context']
            self.assertIs(type(context),expected);self.assertIs(context.data,self.helper.device)
            self.assertIs(seen['frame']['leftovers_context'],r['leftovers_context'])
            self.assertIs(seen['frame']['execution_tables'],r['execution_compiled'].device)
            self.assertIs(seen['public']['execution_tables'],r['execution_compiled'].device)
            self.assertIs(seen['public']['leftovers'],r['leftovers_context'].data)
            self.assertIs(seen['public']['leftovers_no_output'],r['no_output_compiled'].device)
            # Diagnostic mode never hands the helper to the consumers; kernel-event mode hands the same data.
            self.assertIs(seen['public']['status_residual'],None if prepared is self.diagnostic else self.helper.device)


if __name__=='__main__':unittest.main()
