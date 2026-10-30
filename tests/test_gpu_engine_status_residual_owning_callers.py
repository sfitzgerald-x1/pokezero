"""Actual residual assignment diagnostics, not source/public qualification."""
import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import transition as t, mechanics as m, engine_lifecycle_callbacks as life
from gpu_engine import engine_status_residual_callbacks as c, engine_public_events as public
from gpu_engine import engine_execution_completeness as execution
from gpu_engine import engine_public_event_device as base, engine_full_v4_event_device as full
from gpu_engine import engine_request_source as source
import test_gpu_engine_engine_own_facts as fixtures


class StatusResidualOwningCallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeviceOwnFactsTests.setUpClass();cls.f=fixtures.DeviceOwnFactsTests
        cls.metadata=deepcopy(cls.f.metadata)
        cls.compiled=c.compile_status_residual(cls.metadata,engine_catalog=cls.f.lifecycle)

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def chance(self,b=2):
        draws=j.zeros((b,2,12),j.int32).at[:,:,4].set(1)
        return t.ChanceInputs(draws,*[j.zeros((b,2),j.int32) for _ in range(4)],j.zeros(b,j.int32))

    def start(self,b=2):
        original=source.construct_from_pool(self.f.pool,np.tile([[0,1]],(b,1)).astype(np.int64),
            compiled=self.f.request,stream_ids=j.arange(1201,1201+b,dtype=j.int32),seed=31)
        # Explicit tiny component states. Status history/party legality are not
        # certified by these fixtures or by an actual numerical callback.
        state=original.state._replace(hp=j.full_like(original.state.hp,1000),max_hp=j.full_like(original.state.max_hp,1000))
        return state,original.cursor

    def trace(self,state,cursor,*,compiled=True,chance=None,actions=None):
        return life.trace_step(state,j.zeros((state.hp.shape[0],2),j.int32) if actions is None else actions,
            self.f.table,cursor=cursor,catalog=self.f.lifecycle,expected_binding=self.f.lifecycle.binding,
            expected_engine_binding=np.array(self.f.lifecycle.engine_binding,np.uint32),
            chance_inputs=self.chance(state.hp.shape[0]) if chance is None else chance,
            status_residual_compiled=self.compiled if compiled else None)

    @staticmethod
    def words(frame,row):
        words=np.asarray(frame.callbacks.words[row,:int(frame.callbacks.count[row])])
        return words[words[:,0]==20]

    def status_state(self):
        state,cursor=self.start()
        return state._replace(status=state.status.at[:,0,0].set(j.array([m.BRN,m.TOX],j.int32)).at[:,1,0].set(j.array([m.PSN,m.TOX],j.int32)),
            status_counter=state.status_counter.at[:,0,0].set(j.array([0,14],j.int32)).at[:,1,0].set(j.array([0,15],j.int32))),cursor

    def test_actual_burn_poison_toxic_assignment_literal_hp_counter_and_default_world(self):
        state,cursor=self.status_state();seen=[];actual=c.record_status_residual_device
        def capture(*args,**kwargs):
            # Spy at the REAL after-write callback. No event is built from a
            # neighbouring/final state; compare literal fields with its receipt.
            seen.append((args,kwargs));return actual(*args,**kwargs)
        with patch.object(c,'record_status_residual_device',side_effect=capture):frame=self.trace(state,cursor)
        default=self.trace(state,cursor,compiled=False)
        self.same(frame.state,default.state)
        self.assertEqual(len(seen),10)
        for row in range(2):
            words=self.words(frame,row);self.assertEqual(len(words),2)
            self.assertTrue(int(frame.callbacks.refusal[row])&256)
            for word in words:
                side,physical=map(int,word[3:5]);self.assertEqual(physical,0)
                self.assertEqual(int(word[8]),int(state.status[row,side,physical]))
                self.assertEqual(int(word[1]),int(state.status_counter[row,side,physical]))
                self.assertEqual(int(word[6]),int(frame.state.hp[row,side,physical]))
                self.assertEqual(int(word[2]),int(frame.state.status_counter[row,side,physical]))
                matching=[a for a,k in seen if bool(k['eligible'][row]) and int(a[1][row])==side]
                self.assertEqual(len(matching),1);args=matching[0]
                self.assertEqual(word[1:3].tolist(),[int(args[7][row]),int(args[8][row])])
                self.assertEqual(word[5:9].tolist(),[int(args[i][row]) for i in (3,4,5,6)])
        counters={int(w[1]):int(w[2]) for w in self.words(frame,1)}
        self.assertEqual(counters,{14:15,15:15})
        legacy=public.trace_step(state,j.zeros((2,2),j.int32),self.f.table,catalog=self.f.lifecycle,
            expected_binding=self.f.lifecycle.binding,expected_engine_binding=np.array(self.f.lifecycle.engine_binding,np.uint32),
            chance_inputs=self.chance())
        self.same(legacy.step.state,frame.state)
        self.assertTrue(bool((legacy.callbacks.words[:,:,0]<=3).all()))

    def test_literal_numerical_write_fault_does_not_recompute_damage_or_stage(self):
        state,cursor=self.start();state=state._replace(status=state.status.at[:,0,0].set(m.TOX),
            status_counter=state.status_counter.at[:,0,0].set(3))
        actual=m.major_status_residual
        def literal(hp,maximum,status,counter,*,active):
            result=actual(hp,maximum,status,counter,active=active)
            mask=active & (status==m.TOX) & (hp>0)
            return result._replace(hp=j.where(mask,hp-2,result.hp),counter=j.where(mask,9,result.counter))
        with patch.object(m,'major_status_residual',side_effect=literal):
            frame=self.trace(state,cursor);plain=self.trace(state,cursor,compiled=False)
        self.same(frame.state,plain.state)
        for row in range(2):
            words=self.words(frame,row);self.assertEqual(len(words),1)
            self.assertEqual(words[0,1:3].tolist(),[3,9])
            self.assertEqual(int(words[0,5])-int(words[0,6]),2)
            self.assertEqual(int(words[0,6]),int(frame.state.hp[row,0,0]))

    def test_mixed_b8_inactive_and_unsupported_status_emit_no_phantom_receipts(self):
        state,cursor=self.start(8)
        statuses=j.array([m.BRN,m.PSN,m.TOX,m.NONE,m.PAR,m.SLP,m.FRZ,m.TOX],j.int32)
        state=state._replace(status=state.status.at[:,0,0].set(statuses),
            status_counter=state.status_counter.at[:,0,0].set(j.array([0,0,2,0,0,4,0,14],j.int32)),
            done=state.done.at[1].set(True).at[7].set(True))
        frame=self.trace(state,cursor);plain=self.trace(state,cursor,compiled=False)
        self.same(frame.state,plain.state)
        self.assertEqual([len(self.words(frame,row)) for row in range(8)],[1,0,1,0,0,0,0,0])
        for row in (1,7):
            self.assertEqual(int(frame.callbacks.count[row]),0)
            self.same(frame.state.hp[row],state.hp[row])
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):
            self.assertEqual(getattr(frame.callbacks,name).shape,(8,))
            self.same(getattr(frame.callbacks,name),getattr(self.compiled.device,name))

    def test_status_execution_obligation_remains_unhandled_despite_diagnostic20(self):
        state,cursor=self.status_state()
        projector=base.compile_projector(self.f.lifecycle,expected_binding=self.f.lifecycle.binding,
            expected_engine_binding=np.array(self.f.lifecycle.engine_binding,np.uint32))
        adapter=full.compile_adapter(self.f.metadata,projector,catalog=self.f.catalog)
        compiled=execution.compile_accounting(adapter);tables=compiled.device
        ledger=execution.initialize(cursor.stream_id,compiled=compiled,capacity=32)
        tape=execution.begin_on_tape(life._tape(cursor,self.f.lifecycle,64),ledger,tables,active=j.ones(2,bool))
        result=t.step_with_chance(state,j.zeros((2,2),j.int32),self.f.table,self.chance(),
            _public_trace=tape,_lifecycle_trace=True,_status_residual_context=life.StatusResidualContext(self.compiled.device))
        for row in range(2):
            ledger=result.callbacks.execution.ledger
            present=np.arange(ledger.site.shape[1])<int(ledger.count[row])
            status=present & (np.asarray(ledger.site[row])==int(execution.Site.STATUS_RESIDUAL))
            self.assertEqual(int(status.sum()),2)
            self.assertTrue(bool((np.asarray(ledger.stage[row])[status]==1).all()))
            self.assertTrue(int(result.callbacks.refusal[row])&512)
            self.assertTrue(int(result.callbacks.refusal[row])&256)
            self.assertFalse(bool(execution.report(ledger,tables=tables).scoped_drained[row]))

    def test_original_compiler_catalog_context_and_all_outer_tracing_shapes_closed(self):
        state,cursor=self.start();actions=j.zeros((2,2),j.int32);chance=self.chance()
        def invoke(cur=cursor,draws=chance):return self.trace(state,cur,chance=draws,actions=actions)
        for fn,args in ((jax.jit(lambda value:invoke(cur=value)),(cursor,)),
                        (jax.jit(lambda value:invoke(draws=value)),(chance,)),
                        (jax.jit(lambda unused:invoke()),(j.array(0,j.int32),)),(jax.jit(lambda:invoke()),())):
            with patch.object(life.public,'check_catalog',side_effect=AssertionError('host work reached')):
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError,'outside outer JIT'):fn(*args)
        old=self.metadata['vocabulary']['items'][1];self.metadata['vocabulary']['items'][1]='changed'
        try:
            with self.assertRaises(ValueError):invoke()
        finally:self.metadata['vocabulary']['items'][1]=old
        kwargs=dict(cursor=cursor,catalog=self.f.lifecycle,expected_binding=self.f.lifecycle.binding,
            expected_engine_binding=np.array(self.f.lifecycle.engine_binding,np.uint32),chance_inputs=chance)
        with self.assertRaises(ValueError):life.trace_step(state,actions,self.f.table,status_residual_compiled=replace(self.compiled),**kwargs)
        catalog2=public.compile_catalog(self.f.metadata,self.f.table,public_moves=self.f.catalog.moves,dex=self.f.dex,
            exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
        other=c.compile_status_residual(self.f.metadata,engine_catalog=catalog2)
        with self.assertRaisesRegex(ValueError,'same original engine catalog'):
            life.trace_step(state,actions,self.f.table,status_residual_compiled=other,**kwargs)
        for context in ((self.compiled.device,),life.StatusResidualContext(None),life.LeftoversContext(self.compiled.device)):
            with self.assertRaises(ValueError):t.step_with_chance(state,actions,self.f.table,chance,
                _public_trace=life._tape(cursor,self.f.lifecycle,64),_lifecycle_trace=True,_status_residual_context=context)
        with self.assertRaises(ValueError):t.step_with_chance(state,actions,self.f.table,chance,
            _status_residual_context=life.StatusResidualContext(self.compiled.device))

    def test_default_entire_transition_ast_unchanged_outside_optional_status_observer(self):
        old=ast.parse(subprocess.check_output(['git','show','12739035:gpu_engine/transition.py'],text=True))
        new=ast.parse(Path(t.__file__).read_text())
        class Normalize(ast.NodeTransformer):
            def visit_If(self,node):
                if ast.unparse(node.test)=='_status_residual_context is not None':return None
                return self.generic_visit(node)
            def visit_FunctionDef(self,node):
                keep=[i for i,arg in enumerate(node.args.kwonlyargs) if arg.arg!='_status_residual_context']
                node.args.kwonlyargs=[node.args.kwonlyargs[i] for i in keep]
                node.args.kw_defaults=[node.args.kw_defaults[i] for i in keep]
                return self.generic_visit(node)
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(Normalize().visit(new),include_attributes=False))


if __name__=='__main__':unittest.main()
