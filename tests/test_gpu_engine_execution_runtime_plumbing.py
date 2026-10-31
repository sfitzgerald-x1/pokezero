"""Owning partial accounting plumbing; no exhaustive-source/model/game proof."""
from dataclasses import replace
import json
import gc
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_execution_completeness as e
from gpu_engine import engine_retained_encoder_driver as d, engine_request_source as source
from gpu_engine import engine_lifecycle_callbacks as life, source_provenance
import test_gpu_engine_leftovers_registered_runtime as fixtures


class ExecutionRuntimePlumbingTests(unittest.TestCase):
    def setUp(self):
        # Keep setup/previous-case executables out of this heavy runtime case.
        # Prepared inputs remain shared; all within-case warm-cache checks run.
        jax.clear_caches()
        gc.collect()

    @classmethod
    def setUpClass(cls):
        fixtures.RegisteredLeftoversRuntimeTests.setUpClass();cls.f=fixtures.RegisteredLeftoversRuntimeTests.f
        cls.plain=fixtures.RegisteredLeftoversRuntimeTests.prepared
        r=d._checked(cls.plain)
        cls.prepared=d.prepare(r['request_compiled'],r['own_compiled'],public_mode='retained-full',
            leftovers_compiled=r['leftovers_compiled'],leftovers_mode='retained',account_source_sites=True,
            fake_forward=lambda _:d.FakeScores(j.ones(9),j.array(1,j.int32)))

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def construct(self,b=2,prepared=None):
        arrays=self.f().arrays('pressure','pressure');arrays['moves'][0,0,:2]=[self.f.ids['hail'],1]
        with TemporaryDirectory() as directory:
            path=Path(directory)/'literal.npz'
            np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.f.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(path)
            return d.construct(pool,np.tile([[0,1]],(b,1)).astype(np.int64),prepared=prepared or self.prepared,
                stream_ids=j.arange(1101,1101+b,dtype=j.int32),seed=31)

    def chance(self,b=2):
        x=j.zeros((b,2,12),j.int32).at[:,:,4].set(1)
        from gpu_engine import transition
        return transition.ChanceInputs(x,*[j.zeros((b,2),j.int32) for _ in range(4)],j.zeros(b,j.int32))

    @staticmethod
    def capture_actual_leftovers_spans(frame):
        # EXPLICIT internal test seam: root actual-site wiring is not in this
        # frozen plumbing base. Use ONLY real emitted words/stamps, never HP or
        # final-state reconstruction. This does not attest site exhaustiveness.
        tape=frame.callbacks;original=tape
        for index in range(tape.words.shape[1]):
            raw=original.words[:,index];active=(index<original.count)&(raw[:,0]==17)&(tape.execution.ledger.claimed[:,index]==0)
            if not bool(j.any(active)):continue
            cursor=original.cursor._replace(turn=original.event_turn[:,index],phase=original.event_phase[:,index],
                generation=original.event_generation[:,index])
            prefix=tape._replace(count=j.full_like(tape.count,index),cursor=cursor,turn=cursor.turn)
            prefix,key=e.open_attempt(prefix,e.Site.LEFTOVERS,raw[:,3],active=active)
            appended=prefix._replace(count=prefix.count+1)
            captured=e.capture_leftovers_write(appended,key,requested=active,positive_write=active)
            tape=tape._replace(execution=captured.execution,refusal=tape.refusal|captured.refusal)
        return frame._replace(callbacks=tape)

    def advance_with_capture(self,session,actions=None):
        r=d._checked(self.prepared);actual=r['frame'];frames=[]
        def capture(*args):
            frame=self.capture_actual_leftovers_spans(actual(*args));frames.append(frame);return frame
        r['frame']=capture
        try:result=d.advance(session,j.tile(j.array([[0,8]],j.int32),(2,1)) if actions is None else actions,
            chance_inputs=self.chance())
        finally:r['frame']=actual
        self.assertEqual(len(frames),1)
        return result,frames[0]

    def test_actual_initial_context_sealed_before_reveals_and_both_owner_closure(self):
        session=self.construct();plain=self.construct(prepared=self.plain)
        self.same(session.state.core.engine,plain.state.core.engine)
        ledger=session.state.core.execution;r=d._checked(self.prepared);tables=r['execution_compiled'].device
        self.assertEqual(ledger.frame.tolist(),[1,1]);self.assertEqual(ledger.opened.tolist(),[False,False])
        self.assertEqual(e.report(ledger,tables=tables).scoped_drained.tolist(),[True,True])
        original=d._session(session)['constructed'];source._constructed(original)
        self.assertTrue(bool(original.callbacks.execution.ledger.opened.all()))
        self.assertEqual(original.callbacks.execution.ledger.frame.tolist(),[1,1])
        self.assertEqual(len(original.callbacks[:8]),8)
        for x,y in zip(original.callbacks[:8],d._session(plain)['constructed'].callbacks[:8]):self.same(x,y)
        for observer in session.state.observers:self.same(observer.accounting,ledger)
        self.assertIs(original.callbacks.execution.tables,tables)
        with self.assertRaisesRegex(ValueError,'two-owner'):
            source.advance_source(original,j.full((2,2),8,j.int32),compiled=r['request_compiled'],chance_inputs=self.chance())
        for owner in session.result.owners:
            self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))

    def test_genuine_both_owner_heal_drain_then_next_actual_turn(self):
        session=self.construct()
        result,frame=self.advance_with_capture(session)
        tables=d._checked(self.prepared)['execution_compiled'].device
        self.assertEqual(result.state.core.admitted.tolist(),[True,True])
        self.assertEqual(result.state.core.execution.count.tolist(),[2,2])
        self.assertEqual(result.state.core.execution.stage[:,:2].tolist(),[[3,3],[3,3]])
        self.assertEqual(e.report(result.state.core.execution,tables=tables).scoped_drained.tolist(),[True,True])
        self.assertEqual(frame.callbacks.execution.ledger.stage[:,:2].tolist(),[[2,2],[2,2]])
        self.assertEqual(frame.callbacks.execution.ledger.frame.tolist(),[2,2])
        self.same(result.state.observers[0].accounting,result.state.observers[1].accounting)
        later,frame=self.advance_with_capture(session,j.full((2,2),0,j.int32))
        self.assertEqual(later.state.core.admitted.tolist(),[True,True])
        self.assertEqual(later.state.core.execution.frame.tolist(),[3,3])
        for result in (result,later):
            for owner in result.owners:
                self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))

    def test_second_owner_real_failure_rolls_back_both_before_encoding_no_fake_credit(self):
        session=self.construct();before=session.state;r=d._checked(self.prepared)
        actual_public,actual_encode,actual_score=r['public_fast'],r['encode_fast'],r['score_fast'];calls=[]
        def public(observer,frame,profile,active):
            calls.append('public'+str(int(observer.owner[0])))
            if int(observer.owner[0])==1:
                observer=observer._replace(retained=observer.retained._replace(cursor=observer.retained.cursor._replace(
                    sequence=observer.retained.cursor.sequence.at[0].add(1))))
            return actual_public(observer,frame,profile,active)
        def encode(core,observer):
            self.assertEqual(calls[:2],['public0','public1'])
            self.assertEqual(core.admitted.tolist(),[False,True]);calls.append('encode')
            return actual_encode(core,observer)
        def score(*args):
            self.assertEqual(args[5].tolist(),[False,True]);calls.append('score')
            return actual_score(*args)
        r.update(public_fast=public,encode_fast=encode,score_fast=score)
        try:result,frame=self.advance_with_capture(session)
        finally:r.update(public_fast=actual_public,encode_fast=actual_encode,score_fast=actual_score)
        self.assertEqual(result.state.core.execution.stage[0,:2].tolist(),[2,2])
        self.assertEqual(result.state.core.execution.stage[1,:2].tolist(),[3,3])
        self.assertTrue(int(result.state.core.execution.refusal[0])&e.CONSUMPTION)
        self.assertEqual(int(result.state.core.execution.refusal[1]),0)
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

    def test_b8_zero_and_mixed_rows_preserve_globals_and_private_history(self):
        session=self.construct(8);before=session.result;r=d._checked(self.prepared)
        # Constructor outputs are real inputs to the unchanged B8 advances.
        # Release constructor executables before compiling the larger frame.
        jax.block_until_ready(before)
        jax.clear_caches()
        gc.collect()
        result=d.advance(session,j.full((8,2),8,j.int32),active_rows=j.zeros(8,bool),chance_inputs=self.chance(8))
        self.same(result,before)
        mask=j.array([False,True]*4)
        later=d.advance(session,j.full((8,2),8,j.int32),active_rows=mask,chance_inputs=self.chance(8))
        self.assertEqual(later.state.core.execution.frame.tolist(),[1,2]*4)
        self.assertEqual(later.state.core.frame_number.tolist(),[0,1]*4)
        self.same(later.state.core.execution.inventory_binding,before.state.core.execution.inventory_binding)
        for old,new in zip(before.state.observers,later.state.observers):
            self.same(new.accounting.inventory_binding,old.accounting.inventory_binding)
            self.same(d._observer_rows(mask,old,new),new)
        self.same(d._core_rows(mask,before.state.core._replace(originals=None,constructors=None),
            later.state.core._replace(originals=None,constructors=None)),later.state.core._replace(originals=None,constructors=None))

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
        session=self.construct();original=d._session(session)['constructed'];cursor=life.register(original.cursor.stream_id,
            catalog=r['request']['lifecycle'],expected_binding=r['request']['lifecycle'].binding,
            expected_engine_binding=r['request']['rules'].device.engine_binding)
        call=lambda:life.initialize_with_lifecycle(original.birth_state,r['request']['engine_table'],cursor=cursor,
            catalog=r['request']['lifecycle'],expected_binding=r['request']['lifecycle'].binding,
            expected_engine_binding=r['request']['rules'].device.engine_binding,execution_compiled=r['execution_compiled'])
        with self.assertRaisesRegex(ValueError,'outside outer JIT'):jax.jit(lambda unused:call())(j.array(0,j.int32))
        # Every cached phase validates accounting compiler edges through the
        # ordinary prepared guard, not just at initial setup.
        with patch.object(e,'_checked',side_effect=ValueError('changed accounting compiler')):
            with self.assertRaisesRegex(ValueError,'changed accounting compiler'):
                d.advance(session,j.full((2,2),8,j.int32),chance_inputs=self.chance())


if __name__=='__main__':unittest.main()
