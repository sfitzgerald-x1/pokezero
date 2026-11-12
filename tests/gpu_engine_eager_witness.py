"""Run a prepared source-witness runtime's frame and witness eagerly. Test helper, not a test module.

On a source-witness runtime, driver.construct and driver.advance run the witness pair (frame_witness_fast), and advance
also runs the frame. Both are jit closures over whole programs: the frame, and the witness with
execution.consume_frame_device (called once per owner) inlined. Each one compiles once per prepared runtime.

Component tests of what a construct or an advance produces run the two closures' own unjitted bodies instead, on the same
_frame and consume_pair_device code. Each compiled closure is traced under jit (no lowering or compile) once per process and
argument structure: construction's witness call and an advance's differ, because the frame-input report is still None at
construction. Jit-only failures therefore still surface on both.

Run eagerly, consume_frame_device would recompile its projection scans on every call, because each call builds fresh scan
closures. Inside the scope it therefore runs under a jit with a stable identity. Its array leaves are dynamic arguments; the
table NamedTuples flatten into them. Its other leaves (the encoder Catalog, which is not a registered pytree, and Python
scalars) are closed over, with one compile per distinct set of them.
Only the execution route changes; every value is computed by the same code. The heavy anchor
(test_gpu_engine_source_witness_compiled_anchor) checks compiled == eager on the frame and witness outputs.

Enter it around a construct or an advance, inside a test or a test's own helper. Never enter it in a setUpClass that
other modules borrow: the swap must not outlive the test that asked for it. Call clear() from the module's tearDownModule,
because the jit cache holds compiled executables and the objects they close over.
"""
from contextlib import ExitStack, contextmanager

import jax

from gpu_engine import engine_execution_completeness as execution
from gpu_engine import engine_retained_encoder_driver as driver

_CONSUME = execution.consume_frame_device
# key -> (jitted, closed-over leaves). The entry holds those leaves, so the identities in its key stay unique.
_JIT = {}
# (id of a compiled closure, argument structure) -> that closure, kept alive so ids stay unique.
_TRACED = {}


def eager_call(compiled, *args, **kwargs):
    """The compiled closure's own body on these arguments, after one jit trace per argument structure (no compile)."""
    key = (id(compiled), jax.tree.structure((args, kwargs)))
    if key not in _TRACED:
        compiled.trace(*args, **kwargs);_TRACED[key] = compiled
    return compiled.__wrapped__(*args, **kwargs)


def _closed_key(leaf):
    """A closed-over leaf in a cache key: Python scalars and None by value, everything else by identity."""
    if type(leaf) is float:return ('float', leaf.hex())   # keeps 0.0 and -0.0 apart; every NaN shares one key
    if leaf is None or type(leaf) in (bool, int, str):return (type(leaf).__name__, leaf)
    return ('id', id(leaf))


def stable_consume(*args, **kwargs):
    """consume_frame_device under one jit per tree structure and set of closed-over (non-array) leaves."""
    named = {f'_{i}': value for i, value in enumerate(args)}
    named.update(kwargs)
    leaves, tree = jax.tree_util.tree_flatten(named)
    dynamic_at = tuple(i for i, leaf in enumerate(leaves) if hasattr(leaf, 'dtype'))
    closed = tuple((i, leaf) for i, leaf in enumerate(leaves) if not hasattr(leaf, 'dtype'))
    key = (len(args), tree, dynamic_at, tuple((i, _closed_key(leaf)) for i, leaf in closed))   # an unhashable tree raises
    entry = _JIT.get(key)
    if entry is None:
        def body(values, _closed=dict(closed), _tree=tree, _count=len(args), _total=len(leaves)):
            dynamic = iter(values)
            merged = jax.tree_util.tree_unflatten(_tree, [_closed[i] if i in _closed else next(dynamic) for i in range(_total)])
            # Pop the positional arguments before the call: `**merged` would otherwise be evaluated first.
            positional = [merged.pop(f'_{i}') for i in range(_count)]
            return _CONSUME(*positional, **merged)
        entry = _JIT[key] = (jax.jit(body), closed)
    return entry[0]([leaves[i] for i in dynamic_at])


def clear():
    """Drop the cached consume executables and traced closures (and the objects they hold)."""
    _JIT.clear();_TRACED.clear()


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
