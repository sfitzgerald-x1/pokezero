"""Run a prepared retained driver's frame phase eagerly. Test helper, not a test module.

driver.advance runs the prepared runtime's frame closure, a jax.jit of the whole instrumented step, and
compiles it once per prepared runtime and batch shape. Component tests of the plumbing around the step run
that closure's own unjitted body instead: the same _frame code, with its row selection and global-binding
exclusion, without the whole-step XLA compile. Every other phase stays the prepared runtime's compiled
closure. Each prepared runtime's frame (one per public/accounting mode) is also traced under jit once per
process, on the arguments of its first eager call (no lowering or compile), so jit-only failures still
surface; the heavy tier runs the compiled frame end to end.

Use it around single advances (`with eager_frame(prepared): driver.advance(...)`) or per test, never in a
setUpClass that other modules borrow: the swap must not outlive the test that asked for it.
"""
from contextlib import ExitStack, contextmanager

from gpu_engine import engine_retained_encoder_driver as driver

# id of a traced compiled frame -> that frame, kept alive so ids stay unique.
_TRACED={}


def eager_call(compiled,*args,**kwargs):
    """The compiled frame closure's own body on these arguments, eagerly."""
    if id(compiled) not in _TRACED:
        # Trace the compiled frame under jit on these arguments, once per frame and process, without
        # lowering or compiling it: jit-only failures (Python control flow on a traced value, tracer
        # leaks) are raised while tracing.
        compiled.trace(*args,**kwargs);_TRACED[id(compiled)]=compiled
    return compiled.__wrapped__(*args,**kwargs)


@contextmanager
def eager_frame(*prepared):
    with ExitStack() as stack:
        compiled_frames=[]
        for p in prepared:
            r=driver._checked(p);compiled=r['frame']
            if not hasattr(compiled,'__wrapped__'):raise ValueError('frame already eager: nested or repeated eager_frame scope')
            r['frame']=lambda *args,_compiled=compiled,**kwargs:eager_call(_compiled,*args,**kwargs)
            stack.callback(r.__setitem__,'frame',compiled)   # restored in reverse order, also on errors
            compiled_frames.append((compiled,compiled._cache_size()))
        yield
        # A new compile of the frame inside the scope fails the scope. (A call through a reference to the
        # compiled frame captured before the swap that hit an existing executable would not show here.)
        for compiled,size in compiled_frames:
            if compiled._cache_size()!=size:raise AssertionError('the frame was compiled inside an eager-frame scope')
