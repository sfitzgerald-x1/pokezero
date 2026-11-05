"""V2 CUDA engine: build, load and JAX entry points (jax.ffi custom calls).

The JAX engine (frozen at d892a306) stays the bitwise oracle; every CUDA entry
point here is checked against it by gpu_engine.v2e_diff. Sources live in
gpu_engine/cuda/. The shared library is built once per source hash with nvcc
(-fmad=false, IEEE division/sqrt, no flush-to-zero) into a caller-chosen
directory, then its handlers are registered as CUDA FFI targets.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent / 'cuda'
ARCH = 'sm_103'
NVCC_FLAGS = ('-O3', '-std=c++17', f'-arch={ARCH}', '-fmad=false', '-prec-div=true', '-prec-sqrt=true', '-ftz=false',
    '-Xcompiler', '-fPIC', '-shared', '-lineinfo')
TARGETS = ('V2eRoundTrip', 'V2eChance', 'V2eDivVariants', 'V2ePrepWorlds', 'V2eMtProbe')
_loaded = {}


def toolchain_id():
    """nvcc, jax and jaxlib versions (part of every build cache key)."""
    if 'toolchain' not in _loaded:
        import jax, jaxlib
        version = subprocess.run([nvcc(), '--version'], capture_output=True, text=True).stdout
        _loaded['toolchain'] = f'{version}|jax {jax.__version__}|jaxlib {jaxlib.__version__}'
    return _loaded['toolchain']


def source_hash():
    h = hashlib.sha256(toolchain_id().encode())
    for path in sorted(SOURCE_DIR.glob('*')):
        if path.suffix in ('.cu', '.cuh', '.h'):
            h.update(path.name.encode());h.update(path.read_bytes())
    h.update(repr(NVCC_FLAGS).encode())
    return h.hexdigest()


def nvcc():
    found = shutil.which('nvcc') or ('/usr/local/cuda/bin/nvcc' if Path('/usr/local/cuda/bin/nvcc').exists() else None)
    if found is None:raise RuntimeError('nvcc not found')
    return found


def build(build_dir=None, *, verbose=False):
    """Compile gpu_engine/cuda/*.cu into one shared library keyed by source hash."""
    import jax.ffi
    build_dir = Path(build_dir or os.environ.get('V2E_CUDA_BUILD') or Path.home() / '.cache' / 'v2e-cuda')
    build_dir.mkdir(parents=True, exist_ok=True)
    digest = source_hash()[:16];target = build_dir / f'libv2e-{digest}.so'
    if target.exists():return target
    sources = [str(p) for p in sorted(SOURCE_DIR.glob('*.cu'))]
    tmp = target.with_suffix(f'.so.tmp{os.getpid()}')
    command = [nvcc(), *NVCC_FLAGS, f'-I{jax.ffi.include_dir()}', f'-I{SOURCE_DIR}', '-o', str(tmp), *sources]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:raise RuntimeError(f'nvcc failed ({result.returncode}):\n{result.stderr[-4000:]}')
    if verbose and result.stderr:print(result.stderr[-2000:])
    os.replace(tmp, target)
    return target


def load(build_dir=None):
    """Build (if needed), dlopen and register every handler once per process."""
    import jax
    if 'lib' in _loaded:return _loaded['lib']
    path = build(build_dir);lib = ctypes.cdll.LoadLibrary(str(path))
    for name in TARGETS:
        jax.ffi.register_ffi_target(f'v2e_{name}', jax.ffi.pycapsule(getattr(lib, name)), platform='CUDA')
    _loaded.update(lib=lib, path=path)
    return lib


def roundtrip(tree):
    """P0: pass every leaf through one in-place CUDA kernel; returns (tree, checksum[2])."""
    import jax
    load();leaves, treedef = jax.tree.flatten(tree)
    lanes = leaves[0].shape[0]
    if any(x.ndim == 0 or x.shape[0] != lanes for x in leaves):raise ValueError('every leaf needs the same leading lane axis')
    shapes = [jax.ShapeDtypeStruct((2,), np.uint32)] + [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in leaves]
    out = jax.ffi.ffi_call('v2e_V2eRoundTrip', shapes, input_output_aliases={i: i + 1 for i in range(len(leaves))})(
        *leaves, lanes=np.int64(lanes))
    return jax.tree.unflatten(treedef, out[1:]), out[0]


def chance(keys, replacement_count):
    """P2: transition.sample_chance_inputs(state) fields plus split(state.keys, 2)[:, 0].

    Returns (draws [B,2,12], phazing [B,2], secondary [B,2], sleep_talk [B,2],
    entry_ties [B,2], initial_tie [B], next_keys [B,2]).
    """
    import jax
    import jax.numpy as j
    load();keys = j.asarray(keys, j.uint32);count = j.asarray(replacement_count, j.int32);b = count.shape[0]
    if keys.shape != (b, 2):raise ValueError('[B,2] uint32 keys and [B] int32 replacement counts required')
    i32 = lambda *shape: jax.ShapeDtypeStruct(shape, np.int32)
    return tuple(jax.ffi.ffi_call('v2e_V2eChance', (i32(b, 2, 12), i32(b, 2), i32(b, 2), i32(b, 2), i32(b, 2), i32(b),
        jax.ShapeDtypeStruct((b, 2), np.uint32)))(keys, count))
