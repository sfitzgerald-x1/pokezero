"""Run a prepared source-witness runtime's frame and witness eagerly. Test helper, not a test module.

A source-witness driver.advance compiles three whole programs: the frame, the witness pair (frame_witness_fast), and
execution.consume_frame_device, which both of them call. Component tests of what one advance produces run the frame and
witness closures' own unjitted bodies instead, on the same _frame and consume_pair_device code. Each compiled closure is traced
under jit once per process, on the arguments of its first eager call (gpu_engine_eager_frame.eager_call: no lowering or
compile), so jit-only failures still surface.

Run eagerly, consume_frame_device would recompile its projection scans on every call, because each call builds fresh scan
closures. Inside the scope it therefore runs under one jit per distinct set of non-array arguments, with a stable identity.
Only the execution route changes: every value is computed by the same code. The heavy anchor
(test_gpu_engine_source_witness_compiled_anchor) checks compiled == eager on the frame and witness outputs.

Use it around single advances (`with eager_source_witness(prepared): driver.advance(...)`) or inside a test or a test's own
helper, never in a setUpClass that other modules borrow: the swap must not outlive the test that asked for it. Call clear()
from the module's tearDownModule, because the jit cache holds compiled executables and the objects they close over.
"""
from contextlib import ExitStack, contextmanager

import jax

from gpu_engine import engine_execution_completeness as execution
from gpu_engine import engine_retained_encoder_driver as driver
from gpu_engine_eager_frame import eager_call

_CONSUME = execution.consume_frame_device
# key -> (jitted, non-array arguments). The entry holds those arguments, so the identities in its key stay unique.
_JIT = {}


def _arrays_only(value):
    return all(hasattr(leaf, 'dtype') for leaf in jax.tree.leaves(value))


def stable_consume(*args, **kwargs):
    """consume_frame_device under one jit per set of non-array arguments; array pytrees are its dynamic arguments."""
    named = {f'_{i}': value for i, value in enumerate(args)}
    named.update(kwargs)
    closed = {name: value for name, value in named.items() if not _arrays_only(value)}
    dynamic = {name: value for name, value in named.items() if name not in closed}
    key = (len(args), tuple(sorted(dynamic)), tuple(sorted((name, id(value)) for name, value in closed.items())))
    entry = _JIT.get(key)
    if entry is None:
        def body(values, _closed=closed, _count=len(args)):
            merged = dict(_closed, **values)
            return _CONSUME(*(merged.pop(f'_{i}') for i in range(_count)), **merged)
        entry = _JIT[key] = (jax.jit(body), closed)
    return entry[0](dynamic)


def clear():
    """Drop the cached consume executables (and the objects they close over)."""
    _JIT.clear()


@contextmanager
def eager_source_witness(*prepared):
    with ExitStack() as stack:
        compiled = []
        for p in prepared:
            r = driver._checked(p)
            if r['frame_witness_fast'] is None:raise ValueError('source-witness prepared runtime required')
            for key in ('frame', 'frame_witness_fast'):
                closure = r[key]
                if not hasattr(closure, '__wrapped__'):raise ValueError(f'{key} already eager: nested or repeated eager scope')
                r[key] = lambda *args, _closure=closure, **kwargs: eager_call(_closure, *args, **kwargs)
                stack.callback(r.__setitem__, key, closure)   # restored in reverse order, also on errors
                compiled.append((key, closure, closure._cache_size()))
        if execution.consume_frame_device is not _CONSUME:raise ValueError('consume_frame_device already replaced')
        execution.consume_frame_device = stable_consume
        stack.callback(setattr, execution, 'consume_frame_device', _CONSUME)
        yield
        # A new compile of the frame or witness inside the scope fails the scope.
        for key, closure, size in compiled:
            if closure._cache_size() != size:raise AssertionError(f'the {key} was compiled inside an eager scope')
