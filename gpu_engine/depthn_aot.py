"""Single-device executable cache for the pinned DepthN serving stack.

The directory must be trusted: JAX executable deserialization can run code.
The checksum detects damaged files. It does not authenticate a cache writer.
"""

from collections import Counter
import gc
import hashlib
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

import jax
import jaxlib
from jax.experimental import serialize_executable
from jax.extend.core import Jaxpr


LOG = logging.getLogger(__name__)
SCHEMA = 1


def _version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _stack(device):
    plugins = tuple(_version(f'jax-cuda{cuda}-{part}') for cuda in (12, 13) for part in ('plugin', 'pjrt'))
    # The executable is independent of the location of the ordinary XLA cache.
    config = tuple(sorted((key, repr(value)) for key, value in jax.config.values.items()
                          if not key.startswith(('jax_compilation_cache', 'jax_persistent_cache'))))
    flags = tuple(os.environ.get(name) for name in ('XLA_FLAGS', 'JAX_XLA_FLAGS', 'NVIDIA_TF32_OVERRIDE'))
    return (sys.implementation.cache_tag, jax.__version__, jaxlib.__version__, plugins, device.platform,
            device.device_kind, device.client.platform_version, device.id, config, flags)


def input_signature(args):
    """Tree and concrete input types for these positional, single-device programs."""
    leaves, tree = jax.tree.flatten(args)
    avals = tuple((tuple(x.shape), str(x.dtype), bool(getattr(x, 'weak_type', False)),
                   repr(x.format) if isinstance(x, jax.Array) and x.committed else None) for x in leaves)
    return tree, avals


def _tree_key(tree):
    node = tree.node_data()
    if node is None:
        return None
    kind, data = node
    return kind.__module__, kind.__qualname__, data, tuple(_tree_key(child) for child in tree.children())


def _read(path, device):
    with path.open('rb') as file:
        expected = file.read(32)
        body = memoryview(file.read())
    if hashlib.sha256(body).digest() != expected:
        raise ValueError('executable checksum differs')
    end = 8 + int.from_bytes(body[:8], 'big')
    in_tree, out_tree = pickle.loads(body[8:end])
    return serialize_executable.deserialize_and_load(body[end:], in_tree, out_tree,
                                                     backend=device.client, execution_devices=[device])


def _write(path, compiled):
    binary, in_tree, out_tree = serialize_executable.serialize(compiled)
    trees = pickle.dumps((in_tree, out_tree), protocol=4)
    length = len(trees).to_bytes(8, 'big')
    digest = hashlib.sha256(length + trees)
    digest.update(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.tmp-', delete=False) as file:
        tmp = Path(file.name)
        try:
            file.write(digest.digest())
            file.write(length)
            file.write(trees)
            file.write(binary)
            file.flush()
            os.fsync(file.fileno())
            file.close()
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


class ExecutableCache:
    def __init__(self, directory, *, source, static, device):
        self.directory = Path(directory).resolve()
        self.device = device
        self.namespace = (SCHEMA, source, static, _stack(device))
        self.events = Counter()
        self.load_seconds = 0.

    def acquire(self, program, build, args, *, signature=None):
        """Return (executable, lower seconds, compile seconds), with zero trace time on a hit."""
        if signature is None:
            signature = input_signature(args)
        tree, avals = signature
        identity = (self.namespace, program, _tree_key(tree), avals)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        path = self.directory / (key + '.jax')
        start = time.perf_counter()
        try:
            compiled = _read(path, self.device)
        except FileNotFoundError:
            pass
        except MemoryError:
            raise
        except Exception as error:
            self.events['read_errors'] += 1
            LOG.warning('AOT executable %s rejected: %s', key, error)
        else:
            self.events['hits'] += 1
            self.load_seconds += time.perf_counter() - start
            return compiled, 0., 0.

        self.events['misses'] += 1
        start = time.perf_counter()
        lowered = build().lower(*args)
        lower_seconds = time.perf_counter() - start
        start = time.perf_counter()
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - start
        del lowered
        try:
            _write(path, compiled)
        except MemoryError:
            raise
        except Exception as error:
            self.events['write_errors'] += 1
            LOG.warning('AOT executable %s could not be written: %s', key, error)
        else:
            self.events['writes'] += 1
        return compiled, lower_seconds, compile_seconds

    def stats(self):
        return dict(self.events, load_seconds=self.load_seconds)


def heap_stats():
    """Current host RSS on Linux and tracked Python objects after warmup."""
    try:
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except OSError:
        rss = None
    objects = gc.get_objects()
    return dict(rss_bytes=rss, gc_objects=len(objects), jaxprs=sum(type(obj) is Jaxpr for obj in objects))
