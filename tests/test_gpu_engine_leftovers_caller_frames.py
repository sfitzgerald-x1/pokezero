"""Actual owning Leftovers write seam through the real frame and driver, always UNKNOWN; no models/games.

Light replacement coverage for leftovers_owning_callers (all of its tests but the AST pin, which follows #1819's
re-pin). The prepared runtimes, constructor and requests/public/encode/score phases are the real ones; the frame
runs the prepared frame closure's own body eagerly (tests/gpu_engine_eager_frame.py, which traces it under jit
once), and every frame-vs-plain-step comparison is eager against eager.
"""
from dataclasses import replace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_leftovers_callbacks as leftovers
from gpu_engine import engine_lifecycle_callbacks as life, engine_retained_encoder_driver as driver
from gpu_engine import transition
from gpu_engine_eager_frame import eager_call, eager_frame
import test_gpu_engine_leftovers_owning_callers as fixture


class LeftoversCallerFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.LeftoversOwningCallerTests.setUpClass();base=fixture.LeftoversOwningCallerTests
        for name in ('f','leftovers','request','own','plain','prepared','item'):setattr(cls,name,getattr(base,name))
        cls.metadata=base.metadata   # by reference: the drift test mutates it and the prepared guard must see it

    same=fixture.LeftoversOwningCallerTests.same
    construct=fixture.LeftoversOwningCallerTests.construct
    chance=fixture.LeftoversOwningCallerTests.chance
    trace=fixture.LeftoversOwningCallerTests.trace
    words=staticmethod(fixture.LeftoversOwningCallerTests.words)

    def test_actual_weather_then_heal_and_next_turn_nonphysical_both_sides(self):
        session=self.construct();state,cursor=session.state.core.engine,session.state.core.cursor
        for actions in (j.tile(j.array([[0,8]],j.int32),(2,1)),j.tile(j.array([[8,4]],j.int32),(2,1))):
            default=self.trace(state,cursor,actions,compiled=False)
            actual=self.trace(state,cursor,actions)
            self.same(actual.state,default.state)
            for row in range(2):
                words=self.words(actual,row);heals=words[words[:,0]==17];chips=words[words[:,0]==16]
                self.assertGreaterEqual(len(heals),1)
                self.assertTrue(int(actual.callbacks.refusal[row]) & leftovers.UNKNOWN_LEFTOVERS)
                for heal in heals:
                    side,physical=map(int,heal[3:5]);self.assertEqual(physical,int(actual.state.active[row,side]))
                    matching=[chip for chip in chips if int(chip[3])==side]
                    self.assertEqual(len(matching),1)
                    self.assertEqual(int(heal[5]),int(matching[0][6]))
                    self.assertEqual(int(heal[6]),int(actual.state.hp[row,side,physical]))
                    self.assertEqual(int(heal[9]),self.item)
                self.assertTrue(any(int(w[4])>0 for w in heals))
            state,cursor=actual.state,actual.cursor
        # Both owner observers remain independently present; no policy forward
        # is authorized by this diagnostic receipt or by the physical fixture.
        self.assertEqual([o.owner.tolist() for o in session.state.observers],[[0,0],[1,1]])
        self.assertFalse(any(bool(o.B4_qualified.any()) for o in session.result.owners))

    def test_literal_before_item_and_after_actual_write_without_amount_recalculation(self):
        from gpu_engine import items
        session=self.construct();state,cursor=session.state.core.engine,session.state.core.cursor
        actions=j.tile(j.array([[0,8]],j.int32),(2,1))
        actual_heal=items.leftovers_heal;actual_pinch=items.consume_pinch
        # Fault the NUMERICAL helper return and later item consumption at their
        # actual seams, to distinguish literal write capture from reconstruction.
        def exact_heal(table,item,hp,maximum,*,eligible):
            result=actual_heal(table,item,hp,maximum,eligible=eligible)
            return j.where(eligible & (item==self.item) & (hp>0) & (hp<maximum),j.minimum(maximum,hp+3),result)
        def consume(table,item,boosts,hp,maximum,*,eligible):
            new,after=actual_pinch(table,item,boosts,hp,maximum,eligible=eligible)
            return j.where(eligible & (item==self.item),0,new),after
        with patch.object(items,'leftovers_heal',side_effect=exact_heal),patch.object(items,'consume_pinch',side_effect=consume):
            default=self.trace(state,cursor,actions,compiled=False)
            actual=self.trace(state,cursor,actions)
        self.same(actual.state,default.state)
        for row in range(2):
            heals=self.words(actual,row);heals=heals[heals[:,0]==17]
            self.assertEqual(len(heals),2)
            for heal in heals:
                side,physical=map(int,heal[3:5])
                self.assertEqual(int(heal[6])-int(heal[5]),3)
                self.assertEqual(int(heal[9]),self.item)
                self.assertEqual(int(actual.state.item[row,side,physical]),0)
                self.assertEqual(int(heal[6]),int(actual.state.hp[row,side,physical]))

    def test_b8_actual_hot_core_noop_dead_other_item_inactive_and_global_bindings(self):
        session=self.construct(8,move='raindance');core=session.state.core
        hp=core.engine.hp.at[2,0,0].set(0).at[4,0,0].add(-10).at[6,0,0].add(-10)
        engine=core.engine._replace(hp=hp,done=core.engine.done.at[2].set(True),item=core.engine.item.at[4,0,0].set(0))
        core=core._replace(engine=engine)
        mask=j.array([True,False,True,False,True,False,True,False]);actions=j.tile(j.array([[0,8]],j.int32),(8,1))
        r=driver._checked(self.prepared);frame=eager_call(r['frame'],core,actions,self.chance(8),mask)
        counts=[int((self.words(frame,row)[:,0]==17).sum()) for row in range(8)]
        self.assertEqual(counts,[0,0,0,0,0,0,1,0])
        for row in (1,3,5,7):
            self.assertEqual(int(frame.callbacks.count[row]),0)
            self.same(frame.state.hp[row],core.engine.hp[row])
            self.same(jax.tree.map(lambda x:x[row],frame.cursor),jax.tree.map(lambda x:x[row],core.cursor))
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):
            self.assertEqual(getattr(frame.callbacks,name).shape,(8,))
            self.same(getattr(frame.callbacks,name),getattr(self.leftovers.device,name))
        expected=transition.step_with_chance(engine._replace(done=engine.done|~mask),actions,self.f.table,self.chance(8))
        self.same(frame.state,driver._rows(mask,engine,expected.state))
        # Repeated hot invocation uses the same immutable closure, no callbacks.
        self.same(frame,eager_call(r['frame'],core,actions,self.chance(8),mask))

    def test_optional_owning_driver_preserves_world_then_refuses_without_fake_forward(self):
        # Both advances run their frame phase eagerly; the capture wraps the eager body.
        with eager_frame(self.prepared):
            session=self.construct();before=session.state;r=driver._checked(self.prepared);actual=r['frame'];frames=[]
            def capture(*args):
                frame=actual(*args);frames.append(frame);return frame
            r['frame']=capture
            actions=j.tile(j.array([[0,8]],j.int32),(2,1))
            try:result=driver.advance(session,actions,chance_inputs=self.chance())
            finally:r['frame']=actual
            self.assertEqual(len(frames),1)
            expected=transition.step_with_chance(before.core.engine,actions,self.f.table,self.chance())
            self.same(result.state.core.engine,expected.state)
            self.assertTrue(all((self.words(frames[0],r)[:,0]==17).any() for r in range(2)))
            self.assertEqual(result.state.core.admitted.tolist(),[False,False])
            for old,new,owner in zip(before.observers,result.state.observers,result.owners):
                self.same(old.retained,new.retained)
                self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))
            old=session.result
            later=driver.advance(session,actions,chance_inputs=self.chance())
            self.same(later,old)
            self.assertEqual(driver.latest_diagnostics(session).executed_rows.tolist(),[False,False])

    def test_original_host_guard_each_invocation_and_wrong_context_closed(self):
        session=self.construct();before=session.result
        # The guard refuses before any device phase; the eager scope only keeps a regression from compiling the frame.
        with eager_frame(self.prepared):
            self.metadata['vocabulary']['items'][self.item]='changed'
            try:
                with self.assertRaises(ValueError):driver.advance(session,j.full((2,2),8,j.int32),chance_inputs=self.chance())
                with self.assertRaises(ValueError):self.trace(session.state.core.engine,session.state.core.cursor,j.full((2,2),8,j.int32))
            finally:self.metadata['vocabulary']['items'][self.item]='leftovers'
        self.assertIs(session.result,before)
        with self.assertRaises(ValueError):
            driver.prepare(self.request,self.own,fake_forward=lambda _:None,leftovers_compiled=replace(self.leftovers))
        state=session.state.core.engine
        with self.assertRaises(ValueError):
            jax.jit(lambda value:self.trace(value,session.state.core.cursor,j.full((2,2),8,j.int32)))(state)
        with self.assertRaises(ValueError):
            transition.step_with_chance(state,j.full((2,2),8,j.int32),self.f.table,self.chance(),
                _leftovers_context=life.LeftoversContext(self.leftovers.device))
        with self.assertRaises(ValueError):
            transition.step_with_chance(state,j.full((2,2),8,j.int32),self.f.table,self.chance(),
                _public_trace=None,_lifecycle_trace=False,_leftovers_context=(self.leftovers.device,))

    def test_checked_lifecycle_rejects_cursor_chance_and_allclosed_before_host_work(self):
        session=self.construct();core=session.state.core
        actions=j.full((2,2),8,j.int32);chance=self.chance()
        def invoke(cursor=core.cursor,draws=chance):
            return life.trace_step(core.engine,actions,self.f.table,cursor=cursor,catalog=self.f.producer,
                expected_binding=self.f.producer.binding,
                expected_engine_binding=np.array(self.f.producer.engine_binding,np.uint32),
                chance_inputs=draws,leftovers_compiled=self.leftovers)
        cases=(('cursor-only',jax.jit(lambda cursor:invoke(cursor=cursor)),(core.cursor,)),
               ('chance-only',jax.jit(lambda draws:invoke(draws=draws)),(chance,)),
               ('allclosed-dummy',jax.jit(lambda unused:invoke()),(j.array(0,j.int32),)),
               ('allclosed-noargs',jax.jit(lambda:invoke()),()))
        # The fresh tracing probe must run BEFORE all host catalog/compiler
        # checks, and before the first numerical invocation can become cached.
        with patch.object(life.public,'check_catalog',side_effect=AssertionError('host work reached')) as catalog_guard, \
                patch.object(leftovers,'check_compiled',side_effect=AssertionError('helper guard reached')) as helper_guard, \
                patch.object(transition,'step_with_chance',side_effect=AssertionError('core reached')) as core_call:
            for label,fast,args in cases:
                with self.subTest(case=label):
                    for _ in range(2):
                        with self.assertRaisesRegex(ValueError,'outside outer JIT'):fast(*args)
            catalog_guard.assert_not_called();helper_guard.assert_not_called();core_call.assert_not_called()
        # An ordinary checked host invocation is still usable afterward, and
        # the absent opt-in continues through the existing default path.
        plain=self.trace(core.engine,core.cursor,actions,compiled=False)
        checked=self.trace(core.engine,core.cursor,actions)
        self.same(checked.state,plain.state)


if __name__=='__main__':unittest.main()
