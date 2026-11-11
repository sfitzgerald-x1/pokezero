"""Run a prepared retained driver's frame phase eagerly. Test helper, not a test module.

driver.advance runs the prepared runtime's frame closure, a jax.jit of the whole instrumented step, and
compiles it once per prepared runtime and batch shape. Component tests of the plumbing around the step run
that closure's own unjitted body instead: the same _frame code, with its row selection and global-binding
exclusion, without the whole-step XLA compile. Every other phase stays the prepared runtime's compiled
closure. The compiled frame stays covered end to end in the heavy tier (test_gpu_engine_owning_drain_end_to_end).

Use it around single advances (`with eager_frame(prepared): driver.advance(...)`) or per test, never in a
setUpClass that other modules borrow: the swap must not outlive the test that asked for it.
"""
from contextlib import contextmanager

from gpu_engine import engine_retained_encoder_driver as driver


@contextmanager
def eager_frame(*prepared, frames=None):
    held=[]
    for p in prepared:
        r=driver._checked(p);compiled=r['frame']
        def body(*args,_compiled=compiled):
            out=_compiled.__wrapped__(*args)
            if frames is not None:frames.append(out)
            return out
        held.append((r,compiled,compiled._cache_size()));r['frame']=body
    try:yield
    finally:
        for r,compiled,_ in held:r['frame']=compiled
    for _,compiled,size in held:
        if compiled._cache_size()!=size:raise AssertionError('the compiled frame ran inside an eager-frame scope')
