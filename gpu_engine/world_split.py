"""Opt-in measurement adapter: dispatch the supported world-range seam to helpers.

The owner captures and carries the root once. Helpers call protocol.world_range
for contiguous global index ranges; no sampler implementation is copied here.
Completion remains in prepare_roots' existing chain after the ranges return.
Outside split(), the installed hook delegates to the original range function.
A requested split must actually use the hook; a serial fallback is an error.
"""
from dataclasses import replace
import threading
import time

_local = threading.local()
_installed = {}


def worker_range(constructor, blob):
    """Build and pack one supported range on a worker's own static inputs."""
    import jax
    from . import prepare_pool as pp, phase_d_protocol_world as protocol
    with jax.default_device(jax.devices('cpu')[0]):
        root, detached, lines, seed, start, stop, count = pp.pickle.loads(blob)
        part = protocol.world_range(root, detached, start=start, stop=stop, lines=lines,
            catalog=constructor.catalog, metadata=constructor.metadata, source=constructor.source,
            config=constructor.owning_config, battle_id=root.context.battle_id, seed=seed, count=count)
        jax.block_until_ready([(w.state, w.public) for w in part.worlds if w.state is not None])
    shells, packed = pp.pack_worlds(part.worlds)
    return pp.dumps((replace(part, root=None, worlds=()), shells, packed))


class Backend:
    """Send pickled contiguous ranges to helpers and validate their ordered replies."""
    def __init__(self, k, ranges):
        if type(k) is not int or k < 1 or not callable(ranges):
            raise ValueError('world split requires k >= 1 and a range dispatcher')
        self.k, self.ranges = k, ranges
        self.parts = {}
        self._refused = False

    def require_split(self):
        if not self.parts or self.parts.get('k', 0) < min(self.k, self.parts.get('worlds', self.k)):
            raise RuntimeError('world split hook did not execute the requested ranges; serial fallback forbidden')

    def check_result(self, item):
        """A genuine constructor refusal has no worlds to split; preserve it."""
        from . import constructor_worlds as cw
        self._refused = type(item) is cw.RootRefusal
        if not self._refused:
            self.require_split()

    def worlds(self, root, detached, lines, seed, count):
        from .constructor_worlds import ROOT_ERRORS
        try:
            return self._worlds(root, detached, lines, seed, count)
        except ROOT_ERRORS as error:
            # These errors describe broken dispatch/replies, not a constructor
            # refusing the public root. Keep them outside prepare_roots' net.
            raise RuntimeError(f'world split dispatch/gather failed: {type(error).__name__}: {error}') from error

    def _worlds(self, root, detached, lines, seed, count):
        from . import prepare_pool as pp, phase_d_protocol_world as protocol
        import jax
        if self.parts:
            raise RuntimeError('one decision per world split context required')
        k = min(self.k, count)
        bounds = [(count * i // k, count * (i + 1) // k) for i in range(k)]
        clock = time.perf_counter()
        blobs = [pp.dumps((root, detached, tuple(lines), seed, a, b, count)) for a, b in bounds]
        shipped = time.perf_counter()
        replies = tuple(self.ranges(blobs));back = time.perf_counter()
        if len(replies) != k:
            raise RuntimeError('world split did not return every dispatched range')
        parts = []
        for (start, stop), reply in zip(bounds, replies):
            header, shells, packed = pp.pickle.loads(reply)
            if type(header) is not protocol.WorldRange or (header.start, header.stop) != (start, stop):
                raise RuntimeError('world split reply does not match its dispatched range')
            parts.append(replace(header, root=detached, worlds=pp.unpack_worlds(shells, packed, jax.devices('cpu')[0])))
        made = protocol.assemble(parts, count=count)
        self.parts = dict(k=k, worlds=count, ship=shipped-clock, helpers=back-shipped,
            gather=time.perf_counter()-back, bytes=sum(map(len, blobs)) + sum(map(len, replies)))
        return replace(parts[0], start=0, stop=count, worlds=made.worlds, seeds=made.seeds,
            mapper_diagnostics=made.mapper_diagnostics)


def _split_range(root, detached, *, start, stop, lines, catalog, source, metadata, config, battle_id, seed, count):
    backend = getattr(_local, 'backend', None)
    if backend is None:
        return _installed['world_range'](root, detached, start=start, stop=stop, lines=lines, catalog=catalog,
            source=source, metadata=metadata, config=config, battle_id=battle_id, seed=seed, count=count)
    if type(count) is not int or count not in (2, 8, 16, 32, 64) or start != 0 or stop != count:
        raise RuntimeError('world split requires one full declared world budget from the constructor seam')
    return backend.worlds(root, detached, lines, seed, count)


def install():
    """Install once; refuse a missing or replaced hook instead of silently running serially."""
    from . import phase_d_protocol_world as protocol
    current = getattr(protocol, 'world_range', None)
    if 'world_range' not in _installed:
        if not callable(current):
            raise RuntimeError('world split requires the protocol.world_range seam hook')
        _installed['world_range'] = current
        protocol.world_range = _split_range
    elif current is not _split_range:
        raise RuntimeError('world split seam hook is missing or replaced')


class split:
    """Route one decision's range through backend on this thread; fail if bypassed.

    Call backend.check_result on prepare_roots' result to preserve a RootRefusal
    before any worlds were built. A successful prepare must dispatch ranges.
    """
    def __init__(self, backend):
        self.backend = backend

    def __enter__(self):
        if getattr(_local, 'backend', None) is not None:
            raise RuntimeError('nested world split contexts are not supported')
        install()
        self.backend.parts = {};self.backend._refused = False
        _local.backend = self.backend
        return self.backend

    def __exit__(self, exc_type, *exc):
        _local.backend = None
        if exc_type is None and not self.backend._refused:
            self.backend.require_split()
