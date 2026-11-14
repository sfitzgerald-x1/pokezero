"""Release the traced programs that jit's tracing caches keep after a program is compiled.

Why. jit caches each trace it makes in two caches, each keyed by a weak reference to the traced Python function:
jax._src.interpreters.partial_eval.trace_to_jaxpr (no size bound) and jax._src.pjit._infer_params_cached (2048 entries).
A cache entry lives as long as its function does. These functions live as long as the process:
- the live executor's jax.jit-wrapped legal, observe, dispatch, sample and advance (device_public_teacher.make_executor);
- module-level jitted helpers;
- a search's reply, label and root-forward programs.
A pricer program's trace calls the executor's jitted functions on its own tracers. So the traces of every engine-sized
program that a warmup lowers stay alive in these caches after it is compiled, together with every jaxpr derived from them
and cached by weak reference to them. Compiled objects hold none of it. After a DN2W16 server warmup that is about 66M Python objects, and
every full (gen-2) collection rescans them for about 13.5 s with the GIL held.

Why releasing is safe. A compiled program never needs its trace again:
- a Compiled (the pricers' program tables) runs without tracing;
- a jitted function called with concrete arguments at a signature it has already run uses its own executable cache (the
  C++ fast path), which release() leaves alone.
Clearing changes no program. A function traced again later re-traces to the same jaxpr. One kind of executable is keyed on
a trace rather than on a function: an eagerly evaluated jaxpr, such as a jitted function vmapped outside jit (vmapped
jax.random.fold_in in device_public_world_state.world_keys). That kind compiles once more on its next call after a
release, from the persistent cache when one is configured. So release once, after the last compile, then run the serving
path once more before serving (DepthNLiveSearch.warmup does both).

Both caches are JAX internals, verified on jax 0.12.0. release() raises RuntimeError when either is missing or cannot be
cleared (a JAX upgrade that moved them), rather than silently leaving the traces alive.
"""
VERIFIED_JAX = '0.12.0'
CACHES = (('jax._src.interpreters.partial_eval', 'trace_to_jaxpr'), ('jax._src.pjit', '_infer_params_cached'))
_found = None


def caches():
    """jit's two tracing caches (CACHES). RuntimeError when either is missing or has no cache_clear / cache_info."""
    global _found
    if _found is None:
        import importlib
        import jax
        out = []
        for module, name in CACHES:
            try:
                cache = getattr(importlib.import_module(module), name)
            except (ImportError, AttributeError) as error:
                raise RuntimeError(f'trace_release: {module}.{name} not found in jax {jax.__version__} (verified on {VERIFIED_JAX}): '
                    f'{type(error).__name__}: {error}') from error
            if not (callable(getattr(cache, 'cache_clear', None)) and callable(getattr(cache, 'cache_info', None))):
                raise RuntimeError(f'trace_release: {module}.{name} in jax {jax.__version__} has no cache_clear/cache_info '
                    f'(verified on {VERIFIED_JAX})')
            out.append(cache)
        _found = tuple(out)
    return _found


def release():
    """Empty jit's tracing caches (see the module docstring). Returns the number of cache entries dropped."""
    dropped = 0
    for cache in caches():
        dropped += int(cache.cache_info().currsize)
        cache.cache_clear()
    return dropped
