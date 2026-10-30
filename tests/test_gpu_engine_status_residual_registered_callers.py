"""Selected actual kernel-event coverage; original physics remains unqualified."""
import ast
import inspect
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import transition as t, mechanics as m, engine_lifecycle_callbacks as life
from gpu_engine import engine_status_residual_callbacks as c, engine_execution_completeness as e
from gpu_engine import engine_retained_encoder_driver as driver, engine_public_events as public
from gpu_engine import engine_weather_speed_profile as speed
import test_gpu_engine_weather_hp_runtime as fixtures
import test_gpu_engine_engine_status_residual_callbacks as helper_fixture


class RegisteredStatusResidualCallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.WeatherHPRuntimeTests.setUpClass();cls.f=fixtures.WeatherHPRuntimeTests.f
        cls.prepared=fixtures.WeatherHPRuntimeTests.prepared
        cls.runtime=driver._checked(cls.prepared);cls.profile=cls.runtime['speed_profile']
        cls.helper=c.compile_status_residual(cls.f.metadata,engine_catalog=cls.f.producer)
        cls.accounting=e.compile_accounting(cls.runtime['public'])

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def start(self,*,toxic=False,lethal=False,tie=False,warmup=True):
        session=fixtures.WeatherHPRuntimeTests.construct(self,tie=tie)
        core=session.state.core
        # Establish the nonphysical current target with an ACTUAL clean
        # kernel switch first. Injecting PSN into the incoming bench would
        # correctly trigger the separate uninstrumented entry-status family.
        clean=core.engine._replace(item=j.zeros_like(core.engine.item),
            moves=core.engine.moves.at[:,:,:,0].set(self.f.ids['raindance']))
        if warmup:
            warm=t.step_with_chance(clean,j.tile(j.array([[0,8]],j.int32),(2,1)),self.f.table,
                self.f().chance(),_public_trace=life._tape(core.cursor,self.f.producer,64),
                _lifecycle_trace=True,_weather_hp_context=(core.weather_profile,self.profile.device))
            current,birth,raw=warm.callbacks.cursor,warm.weather_profile,warm.step.state
        else:
            current,birth,raw=core.cursor,core.weather_profile,clean
        # Explicit component pre-state after a real installation. This proves
        # the next kernel write, not original Showdown status-hook history.
        state=raw._replace(
            status=raw.status.at[:,0,0].set(m.TOX if toxic else m.BRN).at[:,1,5].set(m.PSN),
            status_counter=raw.status_counter.at[:,0,0].set(15 if toxic else 0))
        if lethal:state=state._replace(hp=state.hp.at[:,0,0].set(1))
        return state,current,birth

    def run_step(self,state,cursor,record,*,registered=True,prior=1,capacity=64,context=None,weather_data=None,actions=None):
        tape=life._tape(cursor,self.f.producer,capacity)._replace(refusal=j.full_like(cursor.turn,prior))
        ledger=e.initialize(cursor.stream_id,compiled=self.accounting,capacity=32,receipt_capacity=capacity)
        tape=e.begin_on_tape(tape,ledger,self.accounting.device,active=~state.done)
        data=self.profile.device if weather_data is None else weather_data
        ctx=(c.RegisteredStatusResidualContext(self.helper.device,data) if registered else life.StatusResidualContext(self.helper.device)) if context is None else context
        actions=j.tile(j.array([[0,0]] if actions is None else [actions],j.int32),(state.hp.shape[0],1))
        return t.step_with_chance(state,actions,self.f.table,
            self.f().chance(),_public_trace=tape,_lifecycle_trace=True,_weather_hp_context=(record,data),
            _status_residual_context=ctx)

    def status_entries(self,result,row):
        ledger=result.callbacks.execution.ledger
        mask=(j.arange(ledger.site.shape[1])<ledger.count[row])&(ledger.site[row]==int(e.Site.STATUS_RESIDUAL))
        return ledger.stage[row][mask]

    def test_actual_kernel_positive_captures_literal_spans_without_claiming_original_hooks(self):
        state,cursor,record=self.start();result=self.run_step(state,cursor,record)
        default=t.step_with_chance(state,j.tile(j.array([[0,0]],j.int32),(2,1)),self.f.table,self.f().chance())
        self.same(result.step.state,default.state)
        self.assertTrue(bool(result.weather_profile.qualified_birth.all()))
        for row in range(2):
            self.assertEqual(self.status_entries(result,row).tolist(),[2,2])
            words=np.asarray(result.callbacks.words[row,:int(result.callbacks.count[row])]);status=words[words[:,0]==20]
            self.assertEqual(len(status),2);self.assertTrue(bool((status[:,6]>0).all()))
            self.assertEqual(int(result.callbacks.refusal[row])&768,0)
            self.assertEqual(sorted(status[:,4].tolist()),[0,5])
            self.assertEqual(sorted(status[:,8].tolist()),[m.BRN,m.PSN])
            self.assertFalse(bool(e.report(result.callbacks.execution.ledger,tables=self.accounting.device).scoped_drained[row]))
        diagnostic=self.run_step(state,cursor,record,registered=False)
        self.same(diagnostic.step.state,result.step.state)
        self.assertTrue(bool((diagnostic.callbacks.refusal&256).all()))
        self.assertNotIn('_defer_own_unknown',inspect.signature(c.record_status_residual).parameters)

    def test_private_emitter_defer_is_only_new_bit_and_invalid_paths_still_latch(self):
        state,cursor,record=self.start();base=life._tape(cursor,self.f.producer,1)
        def emit(tape,*,status=m.BRN,old=100,new=90,before=0,after=0):
            return c.record_status_residual_device(tape,0,0,old,new,201,status,before,after,
                self.helper.device,eligible=True,_defer_own_unknown=True)
        good=emit(base);self.assertEqual(good.refusal.tolist(),[1,1]);self.assertEqual(good.count.tolist(),[1,1])
        for prior in (256,128,512):
            result=emit(base._replace(refusal=j.full(2,prior,j.int32)))
            self.assertEqual(result.refusal.tolist(),[prior|256]*2)
        for kwargs in (dict(status=m.TOX,before=15,after=15),dict(new=0),dict(new=100),dict(after=1)):
            got=emit(base,**kwargs);self.assertTrue(bool((got.refusal&256).all()))
        overflow=emit(good);self.assertTrue(bool((overflow.refusal&public.OVERFLOW).all()))
        self.assertTrue(bool((overflow.refusal&256).all()))

    def test_false_profile_tie_and_toxic_cap_noop_lethal_cannot_credit_source(self):
        for kwargs in (dict(tie=True),dict(toxic=True),dict(lethal=True),{}):
            state,cursor,record=self.start(**kwargs)
            if not kwargs:record=record._replace(qualified_birth=j.array([False,True]))
            result=self.run_step(state,cursor,record)
            ledger=result.callbacks.execution.ledger
            if kwargs:
                self.assertTrue(bool((result.callbacks.refusal&256).all()))
                self.assertTrue(bool((result.callbacks.refusal&512).all()))
                for row in range(2):
                    selected=(j.arange(ledger.site.shape[1])<ledger.count[row])&(ledger.site[row]==int(e.Site.STATUS_RESIDUAL))
                    unsafe=selected if kwargs.get('tie') else selected&(ledger.side[row]==0)
                    self.assertTrue(bool((ledger.stage[row][unsafe]==1).all()))
            else:
                self.assertEqual(result.weather_profile.qualified_birth.tolist(),[False,True])
                self.assertEqual(self.status_entries(result,0).tolist(),[1,1])
                self.assertEqual(self.status_entries(result,1).tolist(),[2,2])
                self.assertEqual(int(result.callbacks.refusal[1])&768,0)
            if kwargs.get('toxic'):
                for row in range(2):
                    words=np.asarray(result.callbacks.words[row]);tox=words[(words[:,0]==20)&(words[:,8]==m.TOX)]
                    self.assertEqual(tox[0,1:3].tolist(),[15,15])
            if kwargs.get('lethal'):
                words=np.asarray(result.callbacks.words[0]);burn=words[(words[:,0]==20)&(words[:,8]==m.BRN)]
                self.assertEqual(int(burn[0,6]),0)
        state,cursor,record=self.start();actual=m.major_status_residual
        def noop(hp,maximum,status,counter,*,active):return actual(hp,maximum,status,counter,active=active)._replace(hp=hp)
        with patch.object(m,'major_status_residual',side_effect=noop):result=self.run_step(state,cursor,record)
        self.assertFalse(bool((result.callbacks.words[:,:,0]==20).any()))
        for row in range(2):self.assertEqual(self.status_entries(result,row).tolist(),[1,1])
        self.assertTrue(bool((result.callbacks.refusal&768).all()))

    def test_registered_requires_actual_record_same_profile_and_no_hook_boolean(self):
        state,cursor,record=self.start();data=self.profile.device
        for context in ((self.helper.device,data),c.RegisteredStatusResidualContext(self.helper.device,data._replace()),
                c.RegisteredStatusResidualContext(self.helper.device,data,True),
                c.RegisteredStatusResidualContext(self.helper.device,data,object())):
            with self.assertRaises(ValueError):self.run_step(state,cursor,record,context=context)
        tape=life._tape(cursor,self.f.producer,64)
        with self.assertRaises(ValueError):t.step_with_chance(state,j.full((2,2),0,j.int32),self.f.table,self.f().chance(),
            _public_trace=tape,_lifecycle_trace=True,
            _status_residual_context=c.RegisteredStatusResidualContext(self.helper.device,data))
        kwargs=dict(cursor=cursor,catalog=self.f.producer,expected_binding=self.f.producer.binding,
            expected_engine_binding=np.array(self.f.producer.engine_binding,np.uint32),chance_inputs=self.f().chance(),
            status_residual_compiled=self.helper)
        for mode in ('kernel-event','retained',True,'accepted'):
            with self.assertRaises(ValueError):life.trace_step(state,j.full((2,2),0,j.int32),self.f.table,
                status_residual_mode=mode,**kwargs)
        call=lambda:life.trace_step(state,j.full((2,2),0,j.int32),self.f.table,**kwargs)
        for fn,args in ((jax.jit(call),()),(jax.jit(lambda unused:call()),(j.array(0,j.int32),))):
            with self.assertRaisesRegex(ValueError,'outside outer JIT'):fn(*args)

    def test_b8_inactive_global_data_and_false_history_retention(self):
        state,cursor,record=self.start()
        tile=lambda x:j.tile(x,(4,)+(1,)*(x.ndim-1))
        state,cursor,record=jax.tree.map(tile,(state,cursor,record))
        ids=j.arange(1501,1509,dtype=j.int32)
        cursor=cursor._replace(stream_id=ids)
        record=record._replace(stream_id=ids,qualified_birth=j.array([True,True,False,True]*2))
        state=state._replace(done=j.array([False,True]*4))
        # Fixed draws match actual batch; no helper/global table is row-sliced.
        draws=j.zeros((8,2,12),j.int32).at[:,:,4].set(1)
        chance=t.ChanceInputs(draws,*[j.zeros((8,2),j.int32) for _ in range(4)],j.zeros(8,j.int32))
        tape=life._tape(cursor,self.f.producer,64)
        ledger=e.initialize(ids,compiled=self.accounting,capacity=32)
        tape=e.begin_on_tape(tape,ledger,self.accounting.device,active=~state.done)
        result=t.step_with_chance(state,j.tile(j.array([[0,0]],j.int32),(8,1)),self.f.table,chance,
            _public_trace=tape,_lifecycle_trace=True,_weather_hp_context=(record,self.profile.device),
            _status_residual_context=c.RegisteredStatusResidualContext(self.helper.device,self.profile.device))
        self.assertEqual(result.weather_profile.qualified_birth[2::4].tolist(),[False,False])
        for row in (1,3,5,7):
            self.assertEqual(int(result.callbacks.count[row]),0)
            self.same(result.step.state.hp[row],state.hp[row])
            self.same(jax.tree.map(lambda x:x[row],result.weather_profile),jax.tree.map(lambda x:x[row],record))
        for row in (0,4):self.assertEqual(self.status_entries(result,row).tolist(),[2,2])
        for row in (2,6):self.assertEqual(self.status_entries(result,row).tolist(),[1,1])
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):
            self.assertEqual(getattr(result.callbacks,name).shape,(8,))
            self.same(getattr(result.callbacks,name),getattr(self.helper.device,name))
    def test_prior_refusals_and_overflow_cannot_open_kernel_event_coverage(self):
        state,cursor,record=self.start()
        for prior in (257,129,513):
            result=self.run_step(state,cursor,record,prior=prior)
            self.assertTrue(bool(((result.callbacks.refusal&prior)==prior).all()))
            for row in range(2):self.assertTrue(bool((self.status_entries(result,row)==1).all()))
            self.assertFalse(bool(e.report(result.callbacks.execution.ledger,tables=self.accounting.device).scoped_drained.any()))
        # A different actual kernel family remains unhandled: newly arriving
        # status on the original physical5 arrival emits UNKNOWN_STATUS128.
        # The selected observer must preserve it and claim neither span.
        incoming,entry_cursor,entry_record=self.start(warmup=False)
        blocked=self.run_step(incoming,entry_cursor,entry_record,actions=[0,8])
        self.assertTrue(bool((blocked.callbacks.refusal&128).all()))
        for row in range(2):self.assertTrue(bool((self.status_entries(blocked,row)==1).all()))
        result=self.run_step(state,cursor,record,capacity=1)
        self.assertTrue(bool((result.callbacks.refusal&public.OVERFLOW).all()))
        self.assertFalse(bool((result.callbacks.execution.ledger.stage==2).any()))

    def test_default_numeric_ast_parent_identity_and_detect_changed_residual_input(self):
        old=ast.parse(subprocess.check_output(['git','show','df4c0dc5:gpu_engine/transition.py'],text=True))
        new=ast.parse(Path(t.__file__).read_text())
        private={'_public_trace':None,'_lifecycle_trace':False,'_weather_trace':None,
            '_status_trace':None,'_weather_hp_context':None,'_leftovers_context':None,
            '_status_residual_context':None}
        class Normalize(ast.NodeTransformer):
            def truth(self,test):
                names={v.id for v in ast.walk(test) if isinstance(v,ast.Name)}
                if not names or not names<=set(private):return None
                if any(isinstance(v,(ast.Call,ast.Attribute,ast.Subscript)) for v in ast.walk(test)):return None
                return bool(eval(compile(ast.Expression(test),'<private-default-proof>','eval'),{'__builtins__':{}},private))
            def visit_If(self,node):
                result=self.truth(node.test)
                if result is not None:
                    out=[]
                    for statement in node.body if result else node.orelse:
                        value=self.visit(statement)
                        if isinstance(value,list):out.extend(value)
                        elif value is not None:out.append(value)
                    return out
                return self.generic_visit(node)
            def visit_IfExp(self,node):
                result=self.truth(node.test)
                return self.visit(node.body if result else node.orelse) if result is not None else self.generic_visit(node)
        dump=lambda tree:ast.dump(Normalize().visit(tree),include_attributes=False)
        baseline=dump(old);self.assertEqual(dump(new),baseline)
        # Negative control: the proof cannot silently erase changed NUMERICAL
        # input outside the observer guards.
        changed=ast.parse(Path(t.__file__).read_text())
        call=next(v for v in ast.walk(changed) if isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute)
            and v.func.attr=='major_status_residual')
        call.args[0]=ast.BinOp(call.args[0],ast.Add(),ast.Constant(1))
        self.assertNotEqual(dump(changed),baseline)


if __name__=='__main__':unittest.main()
