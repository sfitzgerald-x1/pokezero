"""Owning partial accounting plumbing: birth closure, second-owner rollback, compiler guards; eager frame, no models/games.

Light replacement coverage for execution_runtime_plumbing's T2.1, T2.3, T2.5 and T2.6 (its heal drain T2.2 is
in owning_status_drain, its B=8 row test T2.4 in execution_plumbing_b8_rows). The prepared runtime, the
constructor and the requests/public/join/encode/score phases are the real ones, each compiled once per class;
the frame phase runs the prepared frame closure's own body eagerly (tests/gpu_engine_eager_frame.py, which
traces it under jit once). No test-side Leftovers capture seam: the kernel's own capture must hold.
"""
from dataclasses import replace
import gc
import unittest
from unittest.mock import patch
import weakref

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_execution_completeness as e
from gpu_engine import engine_retained_encoder_driver as d, engine_request_source as source
from gpu_engine import engine_lifecycle_callbacks as life
from gpu_engine_eager_frame import eager_frame
import gpu_engine_owning_runtime_fixture as plumbing


class StopConstruct(Exception):
    pass


class ExecutionPlumbingClosureTests(unittest.TestCase):
    construct=plumbing.ExecutionRuntimePlumbingTests.construct
    chance=plumbing.ExecutionRuntimePlumbingTests.chance
    same=plumbing.ExecutionRuntimePlumbingTests.same

    @classmethod
    def setUpClass(cls):
        # Host-only: the same plain (registered Leftovers) and accounting prepared runtimes.
        plumbing.ExecutionRuntimePlumbingTests.setUpClass();base=plumbing.ExecutionRuntimePlumbingTests
        cls.f=base.f;cls.plain=base.plain;cls.prepared=base.prepared;cls.birth_cache=None

    def birth(self):
        # One real accounting construct per class (constructor, both owners' folds, join, encode, score),
        # shared read-only; no test advances this session.
        cls=type(self)
        if cls.birth_cache is None:
            session=self.construct()
            cls.birth_cache=(session,session.state,session.result)
        return cls.birth_cache

    def test_actual_initial_context_sealed_before_reveals_and_both_owner_closure(self):
        session,state,result=self.birth()
        # The plain prepared's real constructor, stopped before both owners' folds: the same
        # constructed state and frozen callback ABI without compiling a second consumer set.
        constructed=[];actual=source.construct_from_pool
        def stop(*args,**kwargs):
            constructed.append(actual(*args,**kwargs));raise StopConstruct
        with patch.object(source,'construct_from_pool',side_effect=stop):
            with self.assertRaises(StopConstruct):self.construct(prepared=self.plain)
        plain=constructed[0]
        self.same(state.core.engine,plain.state)
        ledger=state.core.execution;r=d._checked(self.prepared);tables=r['execution_compiled'].device
        self.assertEqual(ledger.frame.tolist(),[1,1]);self.assertEqual(ledger.opened.tolist(),[False,False])
        self.assertEqual(e.report(ledger,tables=tables).scoped_drained.tolist(),[True,True])
        original=d._session(session)['constructed'];source._constructed(original)
        self.assertTrue(bool(original.callbacks.execution.ledger.opened.all()))
        self.assertEqual(original.callbacks.execution.ledger.frame.tolist(),[1,1])
        self.assertEqual(len(original.callbacks[:8]),8)
        for x,y in zip(original.callbacks[:8],plain.callbacks[:8]):self.same(x,y)
        for observer in state.observers:self.same(observer.accounting,ledger)
        self.assertIs(original.callbacks.execution.tables,tables)
        with self.assertRaisesRegex(ValueError,'two-owner'):
            source.advance_source(original,j.full((2,2),8,j.int32),compiled=r['request_compiled'],chance_inputs=self.chance())
        for owner in result.owners:
            self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))

    def test_second_owner_real_failure_rolls_back_both_before_encoding_no_fake_credit(self):
        self.birth()   # the consumer closures compile once, before the spies wrap them
        session=self.construct();before=session.state;r=d._checked(self.prepared)
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
        r.update(public_fast=public,encode_fast=encode,score_fast=score);frames=[]
        try:
            with eager_frame(self.prepared):
                eager=r['frame']
                def counted(*args):
                    frames.append(1);return eager(*args)
                r['frame']=counted
                try:result=d.advance(session,j.tile(j.array([[0,8]],j.int32),(2,1)),chance_inputs=self.chance())
                finally:r['frame']=eager
        finally:r.update(actual)
        self.assertEqual(len(frames),1)   # one frame invocation per advance
        # Both public folds finish before any encoding; no fake credit reaches the scorer.
        self.assertEqual(calls,['public0','public1','encode','score','encode','score'])
        ledger=result.state.core.execution
        for row,stage in ((0,2),(1,3)):
            # The Hail turn's entries (two Leftovers heals and the turn-loop prefix), all captured by the
            # kernel: row 0 is rolled back at stage 2, row 1 drains every entry to stage 3.
            present=np.arange(ledger.stage.shape[1])<int(ledger.count[row])
            self.assertEqual(sorted(np.asarray(ledger.site[row])[present].tolist()),
                sorted([int(e.Site.LEFTOVERS)]*2+[int(e.Site.TURNLOOP_PREFIX)]))
            self.assertTrue(bool((np.asarray(ledger.stage[row])[present]==stage).all()))
        self.assertTrue(int(ledger.refusal[0])&e.CONSUMPTION)
        self.assertEqual(int(ledger.refusal[1]),0)
        for old,new in zip(before.observers,result.state.observers):
            self.same(jax.tree.map(lambda x:x[0],old.projection),jax.tree.map(lambda x:x[0],new.projection))
            # Public state has GLOBAL namespaces; use the original row reducer.
            selected=d._public_rows(j.array([False,True]),old.retained,new.retained)
            self.same(selected,new.retained)
            self.assertEqual(new.retained_healthy.tolist(),[False,True])
        for owner in result.owners:
            self.assertFalse(bool(owner.scored.any()))
            self.assertFalse(bool(owner.encoded.observation.attention_mask[0].any()))
            self.assertFalse(bool(owner.own.admitted[0]))

    def test_phase_guard_handle_is_weak_not_a_registry_self_reference(self):
        r=d._checked(self.prepared)
        self.assertIs(r['prepared'](),self.prepared)
        temporary=d.prepare(r['request_compiled'],r['own_compiled'],public_mode='retained-full',
            leftovers_compiled=r['leftovers_compiled'],leftovers_mode='retained',account_source_sites=True,
            fake_forward=lambda _:d.FakeScores(j.zeros(9),j.array(0,j.int32)))
        reference=weakref.ref(temporary)
        del temporary
        gc.collect()
        self.assertIsNone(reference())

    def test_explicit_original_compiler_modes_and_host_tracing_boundaries(self):
        r=d._checked(self.prepared)
        for kwargs in (dict(public_mode='basic',account_source_sites=True),
                       dict(public_mode='retained-full',account_source_sites=True),
                       dict(public_mode='retained-full',leftovers_compiled=r['leftovers_compiled'],account_source_sites=True),
                       dict(account_source_sites=1)):
            with self.assertRaises(ValueError):d.prepare(r['request_compiled'],r['own_compiled'],fake_forward=lambda _:None,**kwargs)
        with self.assertRaises(ValueError):e._checked(replace(r['execution_compiled']))
        session,_,_=self.birth();original=d._session(session)['constructed'];cursor=life.register(original.cursor.stream_id,
            catalog=r['request']['lifecycle'],expected_binding=r['request']['lifecycle'].binding,
            expected_engine_binding=r['request']['rules'].device.engine_binding)
        call=lambda:life.initialize_with_lifecycle(original.birth_state,r['request']['engine_table'],cursor=cursor,
            catalog=r['request']['lifecycle'],expected_binding=r['request']['lifecycle'].binding,
            expected_engine_binding=r['request']['rules'].device.engine_binding,execution_compiled=r['execution_compiled'])
        with self.assertRaisesRegex(ValueError,'outside outer JIT'):jax.jit(lambda unused:call())(j.array(0,j.int32))
        # Every cached phase validates accounting compiler edges through the ordinary prepared guard,
        # not just at initial setup: advance refuses before any device phase (the session is unchanged).
        with eager_frame(self.prepared):
            with patch.object(e,'_checked',side_effect=ValueError('changed accounting compiler')):
                with self.assertRaisesRegex(ValueError,'changed accounting compiler'):
                    d.advance(session,j.full((2,2),8,j.int32),chance_inputs=self.chance())
        self.assertIs(d._session(session)['state'],session.state)


if __name__=='__main__':unittest.main()
