"""Compiled actual lifecycle step: hot jit equals the eager step twice, fresh static guards, no host callbacks.

engine_status_callbacks' hot-jit test in its own module (the rest of its tests are
test_gpu_engine_status_callbacks_public); it compiles the instrumented step once at B=2.
"""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np
import jax
import jax.numpy as j

from pokezero.dex import ShowdownDex
from pokezero.category_vocab import CategoryVocabulary
from gpu_engine import transition as t, engine_lifecycle_callbacks as life, engine_public_events as events
from gpu_engine import engine_public_event_device as base, engine_full_v4_event_device as full
from gpu_engine import engine_status_callbacks as status, public_wire, public_inputs as inputs, v4_encoder as v4
import gpu_engine_status_callbacks_fixture as fixture


class StatusCallbacksHotJitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StatusCallbacksTests.setUpClass();base=fixture.StatusCallbacksTests
        for attr in ('f','metadata','ids','catalog','table','producer','compiled'):setattr(cls,attr,getattr(base,attr))
        cls.fast=staticmethod(base.fast)   # the fixture's one compiled project_and_fold (class access unwraps it)

    kwargs=fixture.StatusCallbacksTests.kwargs
    initial=fixture.StatusCallbacksTests.initial
    chance=fixture.StatusCallbacksTests.chance
    step=fixture.StatusCallbacksTests.step
    same=fixture.StatusCallbacksTests.same
    start=fixture.StatusCallbacksTests.start

    def test_hot_jit_same_actual_core_callback_order_and_fresh_static_guards(self):
        frame=self.initial();actions=j.zeros((2,2),j.int32)
        eager=self.step(frame,actions)
        tape=life._tape(frame.cursor,self.producer,64)
        fast=jax.jit(lambda s,a,c,tape:t.step_with_chance(s,a,self.table,c,
            _public_trace=tape,_lifecycle_trace=True))
        for _ in range(2):
            events.check_catalog(self.producer,self.table,expected_binding=self.producer.binding,
                expected_engine_binding=np.array(self.producer.engine_binding,np.uint32))
            actual=fast(frame.state,actions,self.chance(),tape)
            self.same(actual.step.state,eager.state);self.same(actual.callbacks,eager.callbacks)
        self.assertEqual(fast._cache_size(),1)
        original=self.compiled.device
        object.__setattr__(self.compiled,'device',original._replace(statuses=np.array(original.statuses)))
        try:
            with self.assertRaises(ValueError):full.check_adapter(self.compiled)
        finally:object.__setattr__(self.compiled,'device',original)
        closed=jax.make_jaxpr(fast)(frame.state,actions,self.chance(),tape)
        def primitives(program):
            names=set()
            for equation in program.jaxpr.eqns:
                names.add(equation.primitive.name)
                for value in equation.params.values():
                    for child in value if isinstance(value,(tuple,list)) else (value,):
                        if hasattr(child,'jaxpr'):names.update(primitives(child))
            return names
        self.assertFalse(primitives(closed)&{'pure_callback','io_callback','host_callback'})


if __name__=='__main__':unittest.main()
