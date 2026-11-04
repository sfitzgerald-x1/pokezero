"""Compile oracle JAX functions to generated per-lane CUDA kernels and call them.

compile_function(name, fn, example) traces fn at the generator's lane size,
emits CUDA with gpu_engine.v2e_j2c, builds it (nvcc flags of v2e_cuda, cached
by source hash), registers the XLA FFI target and returns a callable with the
same pytree signature as fn for any lane count. The callable returns
(outputs, error_word); a nonzero error word means a generated cross-lane
access check failed, which never happens for row-independent programs.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from gpu_engine import v2e_cuda, v2e_j2c

_registered = {}


def build_module(module, build_dir=None, workers=None):
    """Compile the generated translation units in parallel and link one shared library."""
    import jax.ffi
    from concurrent.futures import ThreadPoolExecutor
    build_dir = Path(build_dir or os.environ.get('V2E_CUDA_BUILD') or Path.home() / '.cache' / 'v2e-cuda')
    h = hashlib.sha256(module.digest.encode())
    for path in sorted(v2e_cuda.SOURCE_DIR.glob('*.cuh')):h.update(path.read_bytes())
    h.update(repr(v2e_cuda.NVCC_FLAGS).encode());h.update(v2e_cuda.toolchain_id().encode())
    key = h.hexdigest()[:16]
    work = build_dir / f'gen-{module.name}-{key}';target = work / f'libv2e-gen-{module.name}.so'
    if target.exists():return target
    work.mkdir(parents=True, exist_ok=True)
    for name, text in module.files.items():(work / name).write_text(text)
    flags = [f for f in v2e_cuda.NVCC_FLAGS if f not in ('-shared',)]
    include = [f'-I{jax.ffi.include_dir()}', f'-I{v2e_cuda.SOURCE_DIR}', f'-I{work}']
    def compile_one(name):
        obj = work / (name + '.o')
        result = subprocess.run([v2e_cuda.nvcc(), *flags, *include, '-c', '-o', str(obj), str(work / name)], capture_output=True, text=True)
        if result.returncode:
            errors = [line for line in result.stderr.splitlines() if 'error' in line.lower()][:20]
            raise RuntimeError(f'nvcc failed for {module.name}/{name} ({result.returncode}):\n' + '\n'.join(errors))
        return obj
    units = sorted(n for n in module.files if n.endswith('.cu'))
    workers = workers or int(os.environ.get('V2E_BUILD_WORKERS', 0)) or min(32, os.cpu_count() or 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        objects = list(pool.map(compile_one, units))
    tmp = target.with_suffix(f'.so.tmp{os.getpid()}')
    result = subprocess.run([v2e_cuda.nvcc(), f'-arch={v2e_cuda.ARCH}', '-shared', '-o', str(tmp), *map(str, objects)], capture_output=True, text=True)
    if result.returncode:raise RuntimeError(f'link failed for {module.name}: {result.stderr[-2000:]}')
    os.replace(tmp, target)
    return target


class Generated:
    def __init__(self, module, in_tree, out_tree, consts, target):
        self.module = module;self.in_tree = in_tree;self.out_tree = out_tree;self.consts = consts;self.target = target

    def __call__(self, *args):
        import jax
        flat, tree = jax.tree.flatten(args)
        if tree != self.in_tree:raise ValueError('generated kernel called with a different pytree structure')
        lanes = flat[0].shape[0]
        for x, (dtype, shape) in zip(flat, self.module.inputs):
            if x.shape[1:] != tuple(shape[1:]) or x.shape[0] != lanes or np.dtype(x.dtype) != dtype:
                raise ValueError(f'generated kernel input {x.shape}{x.dtype} vs traced {shape}{dtype}')
        results = [jax.ShapeDtypeStruct((lanes,) + tuple(shape[1:]), dtype) for dtype, shape in self.module.outputs]
        results += [jax.ShapeDtypeStruct((self.module.arena_words * lanes,), np.uint32),
            jax.ShapeDtypeStruct((1 + len(getattr(self.module, 'marks', ())),), np.uint32)]
        out = jax.ffi.ffi_call(self.target, results)(*flat, *self.consts)
        return jax.tree.unflatten(self.out_tree, out[:-2]), out[-1]


def _walk(jaxpr):
    from jax.extend import core as jcore
    for e in jaxpr.eqns:
        yield e
        for sub in jcore.jaxprs_in_params(e.params):yield from _walk(sub)


def trace(fn, example):
    """Closed jaxpr of fn over example's pytree with every leaf's axis 0 set to LANES."""
    import jax
    flat, in_tree = jax.tree.flatten(example)
    avals = [jax.ShapeDtypeStruct((v2e_j2c.LANES,) + tuple(np.shape(x)[1:]), x.dtype) for x in flat]
    def flat_fn(*xs):
        return jax.tree.leaves(fn(*jax.tree.unflatten(in_tree, xs)))
    closed = jax.make_jaxpr(flat_fn)(*avals)
    from jax._src.interpreters import partial_eval as pe
    from jax.extend import core as jcore
    dced, _ = pe.dce_jaxpr(closed.jaxpr, [True] * len(closed.jaxpr.outvars), instantiate=True)
    closed = jcore.ClosedJaxpr(dced, closed.consts)
    out_tree = jax.tree.structure(jax.eval_shape(lambda *xs: fn(*jax.tree.unflatten(in_tree, xs)), *avals))
    return closed, avals, in_tree, out_tree


def compile_function(name, fn, example, build_dir=None, profile=False, mode='warp'):
    """example: tuple of positional arguments (pytrees of arrays with a lane axis).

    mode: 'warp' (one warp per lane, array-heavy modules) or 'thread' (one thread per lane, scalar-heavy).
    """
    import jax
    import jax.numpy as j
    from gpu_engine import v2e_modules
    cached = _registered.get(name)
    if cached is not None:
        if cached.fn is not fn or cached.mode != mode:raise ValueError(f'generated module {name!r} already compiled for a different function')
        return cached
    begin = time.perf_counter()
    with v2e_modules.trace_substitutions():
        closed, avals, in_tree, out_tree = trace(lambda *a: fn(*a), tuple(example))
    traced = time.perf_counter()
    module = v2e_j2c.generate(name, closed, avals, profile=profile, mode=mode)
    generated = time.perf_counter()
    path = build_module(module, build_dir)
    module.stats.update(trace_s=traced - begin, generate_s=generated - traced, build_s=time.perf_counter() - generated,
        equations=sum(1 for _ in _walk(closed.jaxpr)))
    lib = ctypes.cdll.LoadLibrary(str(path))
    target = f'v2e_gen_{name}_{module.digest}'
    jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(getattr(lib, f'V2eGen_{name}')), platform='CUDA')
    consts = [j.asarray(c) for c in module.consts]
    fn_ = Generated(module, in_tree, out_tree, consts, target)
    fn_.library = lib;fn_.path = path
    fn_.fn = fn;fn_.mode = mode
    _registered[name] = fn_
    return fn_
