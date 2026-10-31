"""Pure host clock/call-order controls: no actual kernel, model or games."""
from collections import namedtuple
from contextlib import ExitStack
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from gpu_engine import engine_retained_encoder_driver as d


class HostTimingTests(unittest.TestCase):
    def test_wait_is_outside_original_phase_and_observer_is_once(self):
        trace=[]; timings={}; ticks=iter((0, 2_000_000, 5_000_000, 12_000_000))
        def function(value):
            trace.append(('function',value));return 'result'
        def observe(name,ms):trace.append(('observer',name,ms))
        with patch.object(d.time,'perf_counter_ns',side_effect=lambda:next(ticks)), \
                patch.object(d.jax,'block_until_ready',side_effect=lambda x:trace.append(('wait',x))):
            self.assertEqual(d._timed(observe,'engine',function,3,_host_timings=timings),'result')
        self.assertEqual(trace,[('wait',(3,)),('function',3),('wait','result'),('observer','engine',7.)])
        self.assertEqual(timings,{'input_block_wait.engine_ms':2.})

    def test_disabled_instrumentation_has_no_clock_wait_or_observer(self):
        with patch.object(d.time,'perf_counter_ns') as clock, patch.object(d.jax,'block_until_ready') as wait:
            self.assertEqual(d._timed(None,'engine',lambda:5),5)
            with d._host_span(None,'unused'):pass
        clock.assert_not_called();wait.assert_not_called()
        self.assertIsNone(d.FrameDiagnostics(None,None,None,None).host_timings)

    def test_guard_and_device_calls_are_exact_after_host_only_erasure(self):
        import ast
        import subprocess
        from pathlib import Path
        baseline=subprocess.check_output(['git','show',
            'a10b916e:gpu_engine/engine_retained_encoder_driver.py'],text=True)
        current=Path(d.__file__).read_text()
        class EraseHost(ast.NodeTransformer):
            def visit_With(self,node):
                node=self.generic_visit(node)
                if len(node.items)==1 and isinstance(node.items[0].context_expr,ast.Call) and isinstance(node.items[0].context_expr.func,ast.Name) and node.items[0].context_expr.func.id=='_host_span':
                    return node.body
                return node
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Name) and t.id in ('host_timings','advance_start') for t in node.targets):return None
                return self.generic_visit(node)
            def visit_If(self,node):
                if ast.unparse(node.test)=='host_timings is not None':return None
                return self.generic_visit(node)
            def visit_Call(self,node):
                node=self.generic_visit(node)
                node.keywords=[k for k in node.keywords if k.arg!='_host_timings']
                if isinstance(node.func,ast.Name) and node.func.id in ('_consume','_consume_accounted','FrameDiagnostics'):
                    node.args=[a for a in node.args if not (isinstance(a,ast.Name) and a.id=='host_timings')]
                return node
            def visit_FunctionDef(self,node):
                node=self.generic_visit(node)
                if node.args.args and node.args.args[-1].arg=='host_timings':
                    node.args.args.pop();node.args.defaults.pop()
                if node.args.kwonlyargs and node.args.kwonlyargs[-1].arg=='_host_timings':
                    node.args.kwonlyargs.pop();node.args.kw_defaults.pop()
                return node
        names={'_checked','_checked_witness','_timed','_consume','_consume_accounted','construct','advance','latest_diagnostics','_frame'}
        def functions(source,normalize=False):
            tree=ast.parse(source)
            if normalize:tree=EraseHost().visit(tree)
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
        self.assertEqual(functions(baseline),functions(current,True))

    def run_fake_advance(self, *, mutate=False, callback_enabled=True):
        Core=namedtuple('Core','engine frame_number admitted requests orders unsafe originals constructors weather_profile execution risks cursor')
        State=namedtuple('State','core observers')
        engine=NS(done=np.zeros(1,bool));core=Core(engine,np.zeros(1,np.int32),np.ones(1,bool),(),(),(),(),(),None,None,None,None)
        state=State(core,());result=d.Result(state,());session=d.Session(NS(),state,result)
        record=dict(advancing=False,constructed=object(),state=state,result=result)
        trace=[];poisoned=False
        r=dict(execution_compiled=object(),retain_speed=None,public_mode='retained-full')
        def checked(_):
            trace.append('checked')
            if poisoned:raise ValueError('static mutation')
            return r
        def frame(*_):
            trace.append('engine');return NS(state=engine,cursor=NS(admitted=np.ones(1,bool)),
                callbacks=NS(refusal=np.zeros(1,np.int32),execution=None),weather_profile=None,
                invalid_actions=np.zeros(1,bool),placeholder_moves=np.zeros(1,np.int32))
        r['frame']=frame
        def consume(r,state,frame,executed,observe,previous,host_timings):
            for name in ('public_pair','encode_p1','admission_p1','encode_p2','admission_p2'):
                d._timed(observe,name,lambda:None,_host_timings=host_timings)
            return d.Result(state,())
        def callback(name,ms):
            nonlocal poisoned
            trace.append('observer:'+name)
            if mutate:poisoned=True
        with ExitStack() as stack:
            stack.enter_context(patch.object(d,'_session',side_effect=lambda _:trace.append('session') or record))
            stack.enter_context(patch.object(d,'_checked',side_effect=checked))
            stack.enter_context(patch.object(d.source,'_constructed',side_effect=lambda _:trace.append('constructed')))
            stack.enter_context(patch.object(d,'_checked_witness',side_effect=lambda *a,**k:trace.append('witness')))
            stack.enter_context(patch.object(d,'_core_rows',side_effect=lambda active,before,after:after))
            stack.enter_context(patch.object(d,'_consume',side_effect=consume))
            stack.enter_context(patch.object(d.jax,'block_until_ready',side_effect=lambda _:None))
            if mutate:
                with self.assertRaisesRegex(ValueError,'static mutation'):
                    d.advance(session,np.zeros((1,2),np.int32),active_rows=np.ones(1,bool),timing_callback=callback,chance_inputs=object())
                self.assertIs(session.state,state);self.assertFalse(record['advancing'])
            else:
                d.advance(session,np.zeros((1,2),np.int32),active_rows=np.ones(1,bool),
                    timing_callback=callback if callback_enabled else None,chance_inputs=object())
        return trace,record

    def test_original_guard_order_counts_and_seven_callbacks_preserved(self):
        trace,record=self.run_fake_advance()
        names=('engine','requests','public_pair','encode_p1','admission_p1','encode_p2','admission_p2')
        expected=['session','checked','constructed','witness','engine']
        for index,name in enumerate(names):
            expected += ['observer:'+name,'session','checked','constructed','witness']
            if index==0:expected += ['checked']
        expected+=['session']
        self.assertEqual(trace,expected)
        timings=record['diagnostics'].host_timings
        self.assertEqual(set(timings),{'initial_guards_ms','timing_observer_ms','post_observer_guards_ms',
            'extra_guards_ms','final_session_guard_ms','advance_wall_ms',
            *(f'input_block_wait.{name}_ms' for name in names)})
        self.assertTrue(all(np.isfinite(value) and value>=0 for value in timings.values()))

    def test_callback_mutation_still_stops_before_next_phase(self):
        trace,_=self.run_fake_advance(mutate=True)
        self.assertEqual(trace,['session','checked','constructed','witness','engine',
            'observer:engine','session','checked'])

    def test_no_callback_keeps_same_original_nonobserver_guards(self):
        trace,record=self.run_fake_advance(callback_enabled=False)
        self.assertEqual(trace,['session','checked','constructed','witness','engine','checked','session'])
        self.assertIsNone(record['diagnostics'].host_timings)


if __name__=='__main__':unittest.main()
