"""Release the traced programs that jit's tracing caches keep after a program is compiled.

Why. jit caches each trace it makes in two caches, each keyed by a weak reference to the traced Python function:
jax._src.interpreters.partial_eval.trace_to_jaxpr (no size bound) and jax._src.pjit._infer_params_cached (2048 entries).
A cache entry lives as long as its function does. These functions live as long as the process:
- the live executor's jax.jit-wrapped legal, observe, dispatch, sample and advance (device_public_teacher.make_executor);
- module-level jitted helpers;
- a search's reply, label and root-forward programs.
A pricer program's trace calls the executor's jitted functions on its own tracers. So every engine-sized program that a
warmup lowers stays alive in these caches after it is compiled, together with every jaxpr derived from it and cached by
weak reference to it. Compiled objects hold none of it. After a DN2W16 server warmup that is about 66M Python objects, and
every full (gen-2) collection rescans them for about 13.5 s with the GIL held.

Why releasing is safe. A compiled program never needs its trace again:
- a Compiled (the pricers' program tables) runs without tracing;
- a jitted function called with concrete arguments at a signature it has already run uses its own executable cache (the
  C++ fast path), which release() leaves alone.
Clearing changes no program. A function traced again later re-traces to the same jaxpr, and nothing already compiled is
dropped or recompiled.
"""


def _caches():
    """jit's two tracing caches under the names this JAX gives them. A JAX that renames them yields fewer, and then
    release() frees less; tests/test_gpu_engine_trace_release.py fails when a compiled program's trace survives."""
    found = []
    try:
        from jax._src.interpreters import partial_eval
        found.append(partial_eval.trace_to_jaxpr)
    except (ImportError, AttributeError):
        pass
    try:
        from jax._src import pjit
        found.append(pjit._infer_params_cached)
    except (ImportError, AttributeError):
        pass
    return tuple(c for c in found if callable(getattr(c, 'cache_clear', None)))


def release():
    """Empty jit's tracing caches (see the module docstring). Returns the number of cache entries dropped."""
    dropped = 0
    for cache in _caches():
        try:
            dropped += int(cache.cache_info().currsize)
        except Exception:
            pass
        cache.cache_clear()
    return dropped
