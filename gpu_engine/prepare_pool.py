"""Prepare workers for batched DepthN serving: constructor_worlds.prepare_roots, as a black box, in separate processes.

Why processes. Prepare (ProtocolConstructor.prepare plus the carry completion) is eager JAX on tiny arrays: its Python dispatch
holds the GIL between ops, so a thread pool does not scale it (measurement's host-stage profile, 2026-11-14: 1, 2 and 4
threads gave 0.91, 0.91 and 0.70 roots/s). Each worker is one process with its own ProtocolConstructor, built from the
server's own arguments, on the CPU (JAX_PLATFORMS=cpu, no GPU visible): the live server already prepares under
jax.default_device(cpu).

Battle affinity. Every request of a battle goes to the one worker the battle was assigned at first sight (the live worker
with the fewest assigned battles; ties to the lowest index), in submission order, one task at a time per worker. A worker
holds that battle's constructor sessions, attempted addresses, incremental root sessions and peer wire folds
(constructor_worlds._PEER_FOLDS), so each prepare sees exactly the session history serial prepare would: a prepare reads
only its own battle's state and the constructor's static inputs. The caller keeps the session LRU (DepthNLiveSearch.battles,
session_cap) and sends evictions here (evict): each is queued behind that battle's pending prepares on its worker, so it
lands where it would serially. reset() clears every worker and the affinity.

Shipping. A prepared root crosses the pipe as one pickle: the Prepared with each constructed world's (state, public)
replaced by a slot, and per world structure group (the pytree and every leaf's kind, dtype, shape, weak type and
commitment) one stacked buffer per leaf across its worlds, not one pickled array per leaf. unpack() rebuilds every world
leaf as the same kind of value: a jax.Array of the same dtype, shape, weak type and commitment, placed on the device the
caller names (the CPU, where prepare_roots leaves them), with identical bytes. digest() is a canonical hash of everything a
prepared root holds (arrays by dtype, shape, weak type, commitment, platform and bytes), the determinism proofs' check.

Failure. A worker that exits or closes its pipe fails its in-flight task (a 'died' result: the caller answers
server_error) and is restarted with empty sessions; its battles then fold from scratch on their next request (an evicted
battle's path: the same state, more host time). A task running longer than timeout_seconds kills its worker the same way.
A worker that cannot start fails its queued tasks. Nothing waits forever.
"""
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import gc
import hashlib
import importlib
import io
import os
import pickle
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import types

import numpy as np

PROTOCOL = 5
IDENTITY = ('jax', 'platforms', 'x64', 'matmul_precision', 'binding', 'count')   # what every worker of a pool must share
WORKER_ENVIRONMENT = {'JAX_PLATFORMS': 'cpu', 'CUDA_VISIBLE_DEVICES': ''}


# Shipping ---------------------------------------------------------------------------------------------------------------------------
def _proxy(mapping):
    return types.MappingProxyType(mapping)


def _readonly(array):
    array.flags.writeable = False
    return array


class _Pickler(pickle.Pickler):
    """pickle, plus what a prepared root holds that pickle alone does not keep: MappingProxyType views (device_public_worlds'
    frozen readouts) and read-only numpy arrays (their flag)."""
    def reducer_override(self, obj):
        if type(obj) is types.MappingProxyType:
            return (_proxy, (dict(obj),))
        if type(obj) is np.ndarray and not obj.flags.writeable:
            return (_readonly, (np.array(obj),))
        return NotImplemented


def dumps(value):
    buffer = io.BytesIO();_Pickler(buffer, protocol=PROTOCOL).dump(value)
    return buffer.getvalue()


@dataclass(frozen=True)
class Slot:
    """A shipped world's place: row `row` of structure group `group`."""
    group: int
    row: int


def _kind(x):
    import jax
    if isinstance(x, jax.Array):
        return ('jax', str(x.dtype), tuple(x.shape), bool(x.aval.weak_type), bool(x.committed))
    if type(x) is np.ndarray and x.dtype != object:
        return ('np', x.dtype.str, tuple(x.shape))
    return ('obj',)


def _weak(value):
    """A host value that device_put makes a weak-typed jax.Array: jax.Array's own unpickling's way (jax._src.array._reconstruct_array)."""
    from jax._src import core, literals
    return literals.TypedNdArray(value, aval=core.ShapedArray(value.shape, value.dtype, weak_type=True))


def pack_worlds(worlds):
    """(shells, groups): `worlds` with each constructed world's state a Slot and public None, and per structure group
    (id, skeleton: the (state, public) pytree with leaf indices for leaves, kinds, columns: one stacked numpy array per array
    leaf, a list of values per other leaf)."""
    import jax
    groups = {};shells = []
    for w in worlds:
        if w.state is None:
            shells.append(w);continue
        leaves, tree = jax.tree_util.tree_flatten((w.state, w.public))
        kinds = tuple(_kind(x) for x in leaves);key = (tree, kinds)
        g = groups.get(key)
        if g is None:
            g = groups[key] = dict(id=len(groups), tree=tree, kinds=kinds, rows=[])
        shells.append(replace(w, state=Slot(g['id'], len(g['rows'])), public=None));g['rows'].append(leaves)
    packed = []
    for g in groups.values():
        skeleton = jax.tree_util.tree_unflatten(g['tree'], list(range(len(g['kinds']))))
        columns = []
        for k, kind in enumerate(g['kinds']):
            column = [row[k] for row in g['rows']]
            if kind[0] == 'jax':
                columns.append(np.stack(jax.device_get(column)))
            elif kind[0] == 'np':
                columns.append(np.stack(column))
            else:
                columns.append(column)
        packed.append((g['id'], skeleton, g['kinds'], columns))
    return tuple(shells), tuple(packed)


_UNSTACK = {}


def _unstack(count, rows):
    """One jitted program per (leaf count, row count): stacked [rows, ...] leaves -> each row of each leaf, leaf-major."""
    key = (count, rows)
    if key not in _UNSTACK:
        import jax
        _UNSTACK[key] = jax.jit(lambda xs: tuple(x[i] for x in xs for i in range(rows)))
    return _UNSTACK[key]


def unpack_worlds(shells, packed, device):
    """The worlds pack_worlds shipped, every array leaf rebuilt on `device` (a jax Device) as it was (module docstring). The
    common case (every array leaf a strong, uncommitted jax.Array): one transfer per stacked leaf and one jitted unstack, its
    columns padded to the world count so one program serves every root (its outputs: uncommitted, strong, the same bytes)."""
    import jax
    built = {};pad = len(shells)
    for gid, skeleton, kinds, columns in packed:
        _, tree = jax.tree_util.tree_flatten(skeleton)
        rows = len(columns[0]) if columns else 0
        leaves = [[None] * len(kinds) for _ in range(rows)];plain = [];where = []
        if kinds and all(kind[0] == 'jax' and not kind[3] and not kind[4] for kind in kinds) and 0 < rows <= pad:
            stacked = [c if rows == pad else np.concatenate([c, np.zeros((pad - rows,) + c.shape[1:], c.dtype)]) for c in columns]
            with jax.default_device(device):
                out = _unstack(len(kinds), pad)(jax.device_put(stacked))
            for r in range(rows):
                built[(gid, r)] = jax.tree_util.tree_unflatten(tree, [out[k * pad + r] for k in range(len(kinds))])
            continue
        for k, kind in enumerate(kinds):
            for r in range(rows):
                if kind[0] == 'jax':
                    if kind[4]:   # committed: placed explicitly, as it was
                        leaves[r][k] = jax.device_put(columns[k][r], device)
                    else:
                        plain.append(columns[k][r]);where.append((r, k))
                elif kind[0] == 'np':
                    leaves[r][k] = np.array(columns[k][r])
                else:
                    leaves[r][k] = columns[k][r]
        if plain:
            with jax.default_device(device):
                arrays = jax.device_put([_weak(x) if kinds[k][3] else x for x, (r, k) in zip(plain, where)])
            for (r, k), a in zip(where, arrays):
                leaves[r][k] = a
        for r in range(rows):
            built[(gid, r)] = jax.tree_util.tree_unflatten(tree, leaves[r])
    out = []
    for w in shells:
        if isinstance(w.state, Slot):
            state, public = built[(w.state.group, w.state.row)]
            w = replace(w, state=state, public=public)
        out.append(w)
    return tuple(out)


def pack(item):
    """One prepare result (a constructor_worlds.Prepared, or anything else: pickled as it is) as bytes."""
    from . import constructor_worlds as cw
    if type(item) is cw.Prepared:
        shells, packed = pack_worlds(item.made.worlds)
        item = replace(item, made=replace(item.made, worlds=shells))
        return dumps(('prepared', item, packed))
    return dumps(('plain', item))


def unpack(blob, device):
    """pack's item, its world leaves (and any other jax.Array it holds) placed on `device`."""
    import jax
    with jax.default_device(device):   # jax.Array's own unpickling places on the default device
        out = pickle.loads(blob)
    if out[0] == 'plain':
        return out[1]
    _, item, packed = out
    return replace(item, made=replace(item.made, worlds=unpack_worlds(item.made.worlds, packed, device)))


# Canonical digest -------------------------------------------------------------------------------------------------------------------
def canonical(x, memo=None):
    """A canonical, process-independent description of `x`'s value (sets sorted, floats as hex, arrays by dtype, shape, weak
    type, commitment, platform and the sha256 of their bytes); TypeError on a value it does not know. Values, not object
    identity: a sub-object shared by two places (a world's carry reusing a piece of the root's) is described in each place,
    so a shipped root, whose world leaves are rebuilt apart, describes as the original (only a cycle is a reference)."""
    import jax
    memo = {} if memo is None else memo
    if x is None or type(x) in (bool, int, str, bytes):
        return (type(x).__name__, x)
    if type(x) is float:
        return ('float', x.hex())
    if isinstance(x, jax.Array):
        a = np.asarray(jax.device_get(x))
        return ('jax', str(x.dtype), tuple(x.shape), bool(x.aval.weak_type), bool(x.committed),
                sorted({d.platform for d in x.devices()}), hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            return ('ndarray-object', x.shape, bool(x.flags.writeable), [canonical(v, memo) for v in x.reshape(-1).tolist()])
        return ('ndarray', x.dtype.str, x.shape, bool(x.flags.writeable), hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())
    if isinstance(x, np.generic):
        return ('npscalar', x.dtype.str, x.tobytes())
    if id(x) in memo:
        return ('cycle', memo[id(x)])
    memo[id(x)] = len(memo)
    try:
        return _canonical(x, memo)
    finally:
        del memo[id(x)]


def _canonical(x, memo):
    import jax
    name = f'{type(x).__module__}.{type(x).__qualname__}'
    if type(x) in (tuple, list):
        return (name, [canonical(v, memo) for v in x])
    if isinstance(x, tuple) and hasattr(x, '_fields'):   # a namedtuple
        return (name, [(f, canonical(getattr(x, f), memo)) for f in x._fields])
    if isinstance(x, Mapping):
        items = [(canonical(k, memo), canonical(v, memo)) for k, v in x.items()]
        return (name, sorted(items, key=repr))
    if isinstance(x, (set, frozenset)):
        return (name, sorted((canonical(v, memo) for v in x), key=repr))
    if is_dataclass(x) and not isinstance(x, type):
        return (name, [(f.name, canonical(getattr(x, f.name), memo)) for f in fields(x)])
    if isinstance(x, (type, type(len))):
        return ('type', f'{getattr(x, "__module__", "")}.{getattr(x, "__qualname__", repr(x))}')
    state = getattr(x, '__dict__', None)
    if isinstance(state, dict):
        return (name, canonical(dict(state), memo))
    slots = [s for c in type(x).__mro__ for s in getattr(c, '__slots__', ())]
    if slots:
        return (name, [(s, canonical(getattr(x, s, None), memo)) for s in slots])
    raise TypeError(f'canonical: {name} unsupported')


def digest(x):
    """sha256 of canonical(x); a prepare result (Prepared, RootRefusal) without its wall clock (`seconds`), its one timing."""
    if is_dataclass(x) and not isinstance(x, type) and any(f.name == 'seconds' for f in fields(x)):
        x = replace(x, seconds=None)
    return hashlib.sha256(repr(canonical(x)).encode()).hexdigest()


# Worker side ------------------------------------------------------------------------------------------------------------------------
def _resolve(spec):
    module, name = spec.split(':')
    return getattr(importlib.import_module(module), name)


def drop_battles(constructor, battles):
    """depthn_live_search.drop_battles (DepthNLiveSearch._evict's constructor part)."""
    from .depthn_live_search import drop_battles
    drop_battles(constructor, set(battles))


def reset_constructor(constructor):
    """depthn_live_search.reset_constructor (DepthNLiveSearch.reset_sessions' constructor part)."""
    from .depthn_live_search import reset_constructor
    reset_constructor(constructor)


def prepare_one(constructor, request, *, device='cpu', redaction=None):
    """One request through constructor_worlds.prepare_roots and the live search's classification (depthn_live_search
    .classify_prepared): ('prepared', packed item, labels) or ('error', text) for an exception prepare_roots raised (the live
    search's server_error text). MemoryError and other non-Exception errors propagate (the worker exits; the caller answers
    server_error)."""
    from . import constructor_worlds as cw, depthn_live_search as live
    try:
        item = cw.prepare_roots(constructor, [request], device=device, **(redaction or {}))[0]
    except live.FATAL:
        raise
    except Exception as error:
        return ('error', f'{type(error).__name__}: {error}')
    labels = live.classify_prepared(constructor, request, item)
    return ('prepared', pack(item), labels)


def warm_in(constructor, requests, *, device='cpu', redaction=None, compiles=(0,)):
    """A worker's warm-in, before it is ready: every request prepared and classified once, in order, sessions kept through a
    pass (a battle's later decisions fold incrementally, an unseen battle's late decision from scratch, as live), so the
    first-use compiles of the shapes they reach happen now; then every session, attempt, held root and peer fold is forgotten.
    `requests`: one pass (protocol requests), or a sequence of passes (each a sequence of them) with every session forgotten
    between passes, so two lists may repeat an address or go back in a battle (the server's: its warmup requests, then the
    pinned list). dict(requests, passes, seconds, compiles, kinds: count per result kind)."""
    from . import constructor_worlds as cw
    passes = [list(p) for p in requests] if requests and isinstance(requests[0], (list, tuple)) else [list(requests)]
    started = time.perf_counter();before = compiles[0];kinds = {}
    for one in passes:
        for request in one:
            kind = prepare_one(constructor, request, device=device, redaction=redaction)[0];kinds[kind] = kinds.get(kind, 0) + 1
        reset_constructor(constructor);cw._PEER_FOLDS.clear()
    rss = rss_bytes();gc.collect();_trim()   # the warm-in's garbage, given back before serving (the RSS reported is after it)
    gc.freeze()   # what is left (the constructor, compiled programs' caches) leaves the collector's generations: serving's
                  # collections never rescan it (r5a's live servers froze after warmup the same way)
    return dict(requests=sum(len(p) for p in passes), passes=len(passes), seconds=time.perf_counter() - started, compiles=compiles[0] - before, kinds=kinds,
                rss_before_collect=rss, rss=rss_bytes())


def _trim():
    """glibc's malloc_trim(0) where there is one: freed heap pages back to the system."""
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError, AttributeError):
        pass


def worker_main(fd):
    """A worker's loop on the socket `fd` (module docstring): ('init', factory, payload, options) first, then 'prepare',
    'drop', 'reset', 'stop' messages, in order."""
    from multiprocessing.connection import Connection
    conn = Connection(fd)
    try:
        _, factory, payload, options = pickle.loads(conn.recv_bytes())
    except EOFError:
        return 0
    try:
        import jax
        from . import atomic_compilation_cache
        jax_config = options.get('jax_config') or {}
        if not atomic_compilation_cache.install() and jax_config.get('jax_compilation_cache_dir'):   # a shared cache: whole entries only
            raise RuntimeError(f'the persistent cache needs whole-entry writes: {atomic_compilation_cache.STATUS["reason"]}')
        jax.config.update('jax_enable_x64', False)
        for name, value in jax_config.items():
            jax.config.update(name, value)
        constructor = _resolve(factory)(payload)
        from . import constructor_worlds as cw
        ident = dict(pid=os.getpid(), jax=jax.__version__, atomic_cache=dict(atomic_compilation_cache.STATUS), platforms=sorted({d.platform for d in jax.devices()}),
                     x64=bool(jax.config.jax_enable_x64), matmul_precision=str(jax.config.jax_default_matmul_precision or 'default'),
                     binding=cw.constructor_binding(constructor), count=getattr(constructor, 'count', None))
    except BaseException as error:
        conn.send_bytes(pickle.dumps(('init_failed', f'{type(error).__name__}: {error}'), protocol=PROTOCOL));return 1
    device, redaction = options.get('device', 'cpu'), options.get('redaction') or {}
    compiles = [0]

    def listen(event, duration, **kw):
        if event == '/jax/core/compile/backend_compile_duration':
            compiles[0] += 1
    try:
        jax.monitoring.register_event_duration_secs_listener(listen)
    except Exception:   # noqa: BLE001 -- no monitoring: counts stay 0
        pass
    try:
        ident['warm_in'] = warm_in(constructor, options.get('warm_in') or (), device=device, redaction=redaction, compiles=compiles)
    except BaseException as error:
        conn.send_bytes(pickle.dumps(('init_failed', f'warm-in: {type(error).__name__}: {error}'), protocol=PROTOCOL));return 1
    conn.send_bytes(pickle.dumps(('ready', ident), protocol=PROTOCOL))
    while True:
        try:
            message = pickle.loads(conn.recv_bytes())
        except (EOFError, OSError):
            return 0
        kind = message[0]
        if kind == 'prepare':
            task, request = message[1], message[2];clock = time.perf_counter();before = compiles[0]
            out = prepare_one(constructor, request, device=device, redaction=redaction)
            conn.send_bytes(pickle.dumps(out[:1] + (task, time.perf_counter() - clock, compiles[0] - before) + out[1:], protocol=PROTOCOL))
        elif kind == 'worlds':   # MEASUREMENT ONLY (gpu_engine.world_split): one decision's world range, on this worker's constructor
            task, blob = message[1], message[2];clock = time.perf_counter();before = compiles[0]
            try:
                from . import world_split
                out = ('worlds_done', world_split.worker_range(constructor, blob))
            except Exception as error:   # noqa: BLE001 -- that range's error: the split request's server_error
                out = ('error', f'{type(error).__name__}: {error}')
            conn.send_bytes(pickle.dumps(out[:1] + (task, time.perf_counter() - clock, compiles[0] - before) + out[1:], protocol=PROTOCOL))
        elif kind == 'drop':
            drop_battles(constructor, message[1])
        elif kind == 'reset':
            reset_constructor(constructor);conn.send_bytes(pickle.dumps(('reset_done', message[1]), protocol=PROTOCOL))
        elif kind == 'ping':
            conn.send_bytes(pickle.dumps(('pong', message[1], rss_bytes()), protocol=PROTOCOL))
        elif kind == 'stop':
            return 0


def rss_bytes(pid=None):
    """Resident set size of a process (this one by default), from /proc; None where there is none."""
    try:
        with open(f'/proc/{pid or "self"}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


# Pool side --------------------------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    """A task's end: kind 'prepared' (item, labels, seconds: the worker's prepare and classify wall), 'error' (text: an
    exception prepare_roots raised), 'died' (text: the worker exited, was killed or could not start) or 'refused' (the
    task's on_start declined it: `veto`, whatever on_start returned)."""
    task: object
    kind: str
    worker: int
    item: object = None
    labels: object = None
    seconds: float = None
    text: str = None
    veto: object = None
    received: float = None       # time.monotonic() when the result reached this process
    unpack_seconds: float = None
    compiles: int = None         # the worker's JAX backend compiles during this task
    bytes: int = None            # the shipped payload's size


class _Worker:
    def __init__(self, index):
        self.index, self.generation = index, 0
        self.process = self.conn = self.reader = None
        self.state = 'starting'   # starting -> ready -> (died -> starting) | failed | stopped
        self.queue = deque()      # ('prepare', task) | ('drop', battles), in order
        self.busy = None          # the task in flight
        self.started = None       # time.monotonic() its task was sent
        self.ident = None
        self.restarts = []


class _Local(_Worker):
    """The pool's in-process slot (PreparePool.attach_local): this process's own constructor (the live search's), on one thread,
    taking the same messages as a worker in the same order. Its results are the items prepare_roots returned here, never
    shipped: a battle homed here is prepared exactly as the serial server prepares it."""
    def __init__(self, index, constructor, pool):
        super().__init__(index);self.constructor, self.pool = constructor, pool;self.inbox = queue.SimpleQueue();self.state = 'ready'
        self.generation = 1;self.ident = dict(local=True)
        self.thread = threading.Thread(target=self._run, name='prepare-local', daemon=True);self.thread.start()

    def _run(self):
        from . import constructor_worlds as cw, depthn_live_search as live
        device, redaction = self.pool.options['device'], self.pool.options['redaction']
        while (message := self.inbox.get())[0] != 'stop':
            kind = message[0]
            if kind == 'prepare':
                task, request = message[1], message[2];backend = message[3] if len(message) > 3 else None;clock = time.perf_counter()
                try:
                    if backend is not None:   # MEASUREMENT ONLY: this decision's worlds across idle workers (gpu_engine.world_split)
                        from . import world_split
                        with world_split.split(backend):
                            item = cw.prepare_roots(self.constructor, [request], device=device, **redaction)[0]
                    else:
                        item = cw.prepare_roots(self.constructor, [request], device=device, **redaction)[0]
                    labels = live.classify_prepared(self.constructor, request, item)
                    out = ('prepared', task, time.perf_counter() - clock, None, item, labels, 0., 0)
                except BaseException as error:   # noqa: BLE001 -- the request's server_error (MemoryError too: this thread goes on)
                    out = ('error', task, time.perf_counter() - clock, None, f'{type(error).__name__}: {error}')
            elif kind == 'drop':
                try:
                    drop_battles(self.constructor, message[1])
                except Exception:   # noqa: BLE001 -- a battle it cannot drop is forgotten by the next reset; the slot goes on
                    pass
                continue
            elif kind == 'reset':
                try:
                    reset_constructor(self.constructor)
                except Exception:   # noqa: BLE001 -- acknowledged all the same: the reset's caller must not hang on the slot
                    pass
                out = ('reset_done', message[1])
            else:
                continue
            self.pool.events.put(('pool', self.index, self.generation, out + (time.monotonic(),)))


class PreparePool:
    """P prepare workers (module docstring). Single-owner: every method but the reader threads' is called from one thread
    (the live search's executor, or the batching pipeline's scheduler). Results arrive on `events` (a queue.SimpleQueue the
    caller may share with its own events); pass each to handle()."""
    def __init__(self, factory, payload, *, workers, device='cpu', redaction=None, cpus=None, environment=None,
                 timeout_seconds=30., startup_seconds=900., events=None, jax_config=None, python=None, max_restarts=3, warm_in=()):
        if type(workers) is not int or workers < 1:
            raise ValueError('at least one prepare worker required')
        if cpus is not None and len(cpus) != workers:
            raise ValueError('one CPU set per worker required')
        self.factory, self.payload, self.size = factory, payload, workers
        self.options = dict(device=device, redaction=dict(redaction or {}), jax_config=dict(jax_config or {}), warm_in=tuple(warm_in))
        self.cpus, self.timeout, self.startup = cpus, float(timeout_seconds), float(startup_seconds)
        self.environment = dict(environment or {});self.python = python or sys.executable;self.max_restarts = max_restarts
        self.events = queue.SimpleQueue() if events is None else events
        self.workers = [_Worker(i) for i in range(workers)];self.affinity = {};self.closed = False;self.tasks = 0
        self._device = None;self.identity = None   # the first ready worker's (IDENTITY); every later one must match it
        self.local = None   # attach_local's in-process slot (index `size`, after the process workers)
        self.local_extra = None   # task -> an extra the in-process slot's prepare takes (a world_split.Backend), or None

    # lifecycle -----------------------------------------------------------------------------------------------------------------
    def attach_local(self, constructor):
        """An in-process slot on `constructor` (the live search's own): battles submitted with prefer_local may be homed here,
        and are then prepared on this process's constructor, as the serial server prepares (no shipping)."""
        if self.local is None:
            self.local = _Local(self.size, constructor, self);self.workers.append(self.local)
        elif self.local.constructor is not constructor:   # another search over this pool (a test, a bench): idle, re-pointed
            if self.local.busy is not None or self.local.queue:
                raise RuntimeError('the in-process slot is busy')
            self.local.constructor = constructor
        return self.local

    def start(self, wait=True):
        for w in self.workers[:self.size]:
            self._spawn(w)
        if wait:
            self.wait_ready()
        return self

    def wait_ready(self):
        """Block until every worker is ready (RuntimeError naming those that failed or did not start in startup_seconds)."""
        deadline = time.monotonic() + self.startup;stash = []
        try:
            while any(w.state == 'starting' for w in self.workers):
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RuntimeError('prepare workers did not start in time: ' + ', '.join(str(w.index) for w in self.workers if w.state == 'starting'))
                event = self._get(min(left, 1.), stash)
                for result in [] if event is None else self.handle(event):
                    raise RuntimeError(f'a task ended before the pool was ready: {result}')
        finally:
            self._restore(stash)
        failed = [w for w in self.workers if w.state != 'ready']
        if failed:
            raise RuntimeError('prepare workers failed to start: ' + '; '.join(f'{w.index}: {w.ident}' for w in failed))

    def _environment(self):
        env = dict(os.environ);env.update(self.environment);env.update(WORKER_ENVIRONMENT)
        return env

    def _spawn(self, w):
        parent, child = socket.socketpair()
        from multiprocessing.connection import Connection
        w.generation += 1;w.state = 'starting';w.busy = None;w.started = time.monotonic();w.ident = None
        w.process = subprocess.Popen([self.python, '-u', '-m', 'gpu_engine.prepare_pool', '--worker', str(child.fileno())],
            pass_fds=(child.fileno(),), env=self._environment(), stdin=subprocess.DEVNULL)
        child.close()
        if self.cpus is not None:   # set from here (no preexec_fn in a threaded parent); it applies before the worker's JAX starts its pools
            try:
                os.sched_setaffinity(w.process.pid, sorted(self.cpus[w.index]))
            except (OSError, AttributeError):
                pass
        w.conn = Connection(parent.detach())
        w.conn.send_bytes(pickle.dumps(('init', self.factory, self.payload, self.options), protocol=PROTOCOL))
        w.reader = threading.Thread(target=self._read, args=(w, w.generation, w.conn), name=f'prepare-reader-{w.index}', daemon=True)
        w.reader.start()

    def _read(self, w, generation, conn):
        """Reader thread: every message of this worker generation, unpacked here (off the owner's thread), onto events; a
        closed pipe ends with a 'gone' event."""
        while True:
            try:
                blob = conn.recv_bytes()
            except (EOFError, OSError):
                self.events.put(('pool', w.index, generation, ('gone',)));return
            received = time.monotonic();message = pickle.loads(blob)
            if message[0] == 'prepared':
                _, task, seconds, compiles, packed, labels = message;clock = time.perf_counter()
                try:
                    item = unpack(packed, self.device())
                except Exception as error:   # noqa: BLE001 -- a payload that does not rebuild is that task's error
                    message = ('error', task, seconds, compiles, f'PREPARE_UNPACK_FAILED: {type(error).__name__}: {error}')
                else:
                    message = ('prepared', task, seconds, compiles, item, labels, time.perf_counter() - clock, len(packed))
            self.events.put(('pool', w.index, generation, message + (received,)))

    def device(self):
        if self._device is None:
            import jax
            self._device = jax.devices(self.options['device'])[0]
        return self._device

    def close(self, timeout=10.):
        """Stop every worker (a 'stop' message, then SIGKILL after `timeout`)."""
        self.closed = True
        if self.local is not None:
            self.local.inbox.put(('stop',));self.local.state = 'stopped'
        for w in self.workers:
            if w.process is not None and w.process.poll() is None:
                try:
                    w.conn.send_bytes(pickle.dumps(('stop',), protocol=PROTOCOL))
                except OSError:
                    pass
        deadline = time.monotonic() + timeout
        for w in self.workers:
            if w.process is None:
                continue
            try:
                w.process.wait(max(0., deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                w.process.kill();w.process.wait()
            w.state = 'stopped'
            try:
                w.conn.close()
            except OSError:
                pass

    # affinity and sessions ----------------------------------------------------------------------------------------------------
    def assign(self, battle, exclude=None, prefer_local=False):
        """The worker index of `battle` (assigned at first sight: the in-process slot when prefer_local and there is one, else
        the live process worker with the fewest battles, lowest index; a ready one first, then one starting)."""
        if battle in self.affinity:
            return self.affinity[battle]
        if prefer_local and self.local is not None and self.local.state == 'ready':
            self.affinity[battle] = self.local.index
            return self.local.index
        counts = [0] * self.size
        for index in self.affinity.values():
            if index < self.size:
                counts[index] += 1
        process = self.workers[:self.size]
        live = ([w.index for w in process if w.state == 'ready' and w.index != exclude]
                or [w.index for w in process if w.state not in ('failed', 'stopped') and w.index != exclude] or list(range(self.size)))
        index = min(live, key=lambda i: (counts[i], i));self.affinity[battle] = index
        return index

    def evict(self, battles):
        """Forget these battles: each owning worker drops their sessions after its pending prepares (module docstring)."""
        by = {}
        for b in battles:
            index = self.affinity.pop(b, None)
            if index is not None:
                by.setdefault(index, set()).add(b)
        for index, group in by.items():
            self.workers[index].queue.append(('drop', group))   # sent by the next pump(), ahead of anything queued after it

    def reset(self, timeout=None):
        """Clear every worker's constructor sessions (no task may be pending) and the affinity; waits for every worker."""
        if any(w.busy is not None or any(kind == 'prepare' for kind, _ in w.queue) for w in self.workers):
            raise RuntimeError('reset with prepares pending')
        self.affinity.clear();token = object.__repr__(object());waiting = set()
        for w in self.workers:
            w.queue.clear()   # queued drops: the reset forgets everything anyway
            if w.state == 'ready':
                self._send(w, ('reset', token));waiting.add(w.index)
        deadline = time.monotonic() + (self.startup if timeout is None else timeout);stash = []
        try:
            while waiting:
                event = self._get(max(.01, min(1., deadline - time.monotonic())), stash)
                if event is None:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'prepare workers {sorted(waiting)} did not acknowledge a reset')
                    continue
                if event[3][0] == 'reset_done' and event[3][1] == token:
                    waiting.discard(event[1]);continue
                for result in self.handle(event):
                    raise RuntimeError(f'a task ended during a reset: {result}')
                waiting &= {w.index for w in self.workers if w.state == 'ready'}
        finally:
            self._restore(stash)

    # tasks ---------------------------------------------------------------------------------------------------------------------
    def submit(self, task, request, battle, on_start=None, prefer_local=False):
        """Queue `request` (a protocol request: what prepare_roots takes) on its battle's worker (a new battle homed in-process
        when prefer_local). `on_start(task)` runs as it is sent: a non-None return declines it (a 'refused' result with that
        veto; the worker never sees it)."""
        w = self.workers[self.assign(battle, prefer_local=prefer_local)];w.queue.append(('prepare', (task, request, on_start)));self.tasks += 1
        return w.index

    def idle_workers(self):
        """The ready process workers with nothing in flight or queued (world_split's helpers)."""
        return [w.index for w in self.workers[:self.size] if w.state == 'ready' and w.busy is None and not w.queue]

    def send_worlds(self, index, task, blob):
        """MEASUREMENT ONLY (gpu_engine.world_split): one pickled world range to idle worker `index`; its end is a 'worlds'
        Result (item: the packed worlds) or an 'error' / 'died' one."""
        w = self.workers[index];w.busy, w.started = task, time.monotonic();self._send(w, ('worlds', task, blob))

    def submit_to(self, index, task, request):
        """Queue `request` on worker `index` itself, whatever its battle's affinity (warm_in's)."""
        self.workers[index].queue.append(('prepare', (task, request, None)));self.tasks += 1

    def _get(self, timeout, stash):
        """The next pool event within `timeout` seconds (None if none came); a caller's own events on a shared queue are kept
        in `stash` (put back by _restore)."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                event = self.events.get(timeout=max(0., deadline - time.monotonic()))
            except queue.Empty:
                return None
            if type(event) is tuple and len(event) == 4 and event[0] == 'pool':
                return event
            stash.append(event)

    def _restore(self, stash):
        for event in stash:
            self.events.put(event)

    def _run(self, tasks):
        done = {};stash = []
        try:
            for result in self.pump():
                done[result.task] = result
            while len(done) < len(tasks):
                nxt = self.watchdog();wait = 1. if nxt is None else max(.01, min(1., nxt - time.monotonic()))
                event = self._get(wait, stash)
                for result in [] if event is None else self.handle(event):
                    done[result.task] = result
        finally:
            self._restore(stash)
        return [done[t] for t in tasks]

    def pending(self):
        return sum(1 for w in self.workers for kind, _ in w.queue if kind == 'prepare') + sum(w.busy is not None for w in self.workers)

    def pump(self):
        """Send what can be sent now: each worker's control messages at its queue head, and its next prepare when it is idle and
        ready. Returns the 'refused' results of declined tasks."""
        out = []
        for w in self.workers:
            while w.queue and w.state == 'ready':
                kind, body = w.queue[0]
                if kind == 'drop':
                    w.queue.popleft();self._send(w, ('drop', body));continue
                if w.busy is not None:
                    break
                w.queue.popleft();task, request, on_start = body
                veto = on_start(task) if on_start is not None else None
                if veto is not None:
                    out.append(Result(task, 'refused', w.index, veto=veto));continue
                w.busy, w.started = task, time.monotonic()
                extra = self.local_extra(task) if w is self.local and self.local_extra is not None else None
                self._send(w, ('prepare', task, request) + ((extra,) if extra is not None else ()))
            if w.state == 'failed':   # never coming back: its queued prepares fail, its battles go elsewhere next time
                while w.queue:
                    kind, body = w.queue.popleft()
                    if kind == 'prepare':
                        out.append(Result(body[0], 'died', w.index, text=f'prepare worker {w.index} unavailable: {w.ident}'))
        return out

    def _send(self, w, message):
        if w is self.local:
            w.inbox.put(message);return
        try:
            w.conn.send_bytes(pickle.dumps(message, protocol=PROTOCOL))
        except OSError:
            self._kill(w)   # its reader sees the pipe close and reports it gone

    def _kill(self, w):
        if w.process is not None and w.process.poll() is None:
            try:
                w.process.send_signal(signal.SIGKILL)
            except OSError:
                pass

    def watchdog(self, now=None):
        """Kill every worker whose task has run longer than timeout_seconds (its reader then reports it gone). Returns the next
        time (monotonic) a running task would time out, or None."""
        now = time.monotonic() if now is None else now;nxt = None
        for w in self.workers:   # a worker process that exited: its 'gone' now, whether or not its reader's arrives (handle takes one)
            if (w is not self.local and w.process is not None and w.state in ('ready', 'starting') and w.process.poll() is not None
                    and getattr(w, 'gone_sent', None) != w.generation):
                w.gone_sent = w.generation;self.events.put(('pool', w.index, w.generation, ('gone',)))
        for w in self.workers:
            if w.busy is not None and w.state == 'ready':
                due = w.started + self.timeout
                if now < due:
                    nxt = due if nxt is None else min(nxt, due)
                elif w is self.local:   # a thread cannot be killed: the slot is given up (its battles move to the workers)
                    w.state = 'killing';self.events.put(('pool', w.index, w.generation, ('gone',)))
                else:
                    w.state = 'killing';self._kill(w)
        return nxt

    def handle(self, event):
        """Results of one `events` entry: [] for a ready, pong or ack; [Result] for a task's end; a gone worker's in-flight task
        as 'died' (then the worker is restarted, or fails for good after max_restarts within 10 minutes)."""
        if not (type(event) is tuple and len(event) == 4 and event[0] == 'pool'):
            return []   # a caller's own event on a shared queue (a closed pipeline's last words): not the pool's
        _, index, generation, message = event;w = self.workers[index]
        if generation != w.generation:
            return []   # a previous incarnation's last words
        kind = message[0];out = []
        if kind == 'gone':   # one per incarnation: the reader's and the watchdog's may both arrive
            if getattr(w, 'gone_handled', None) == generation:
                return []
            w.gone_handled = generation
        if kind == 'ready':
            same = lambda ident: {k: ident.get(k) for k in IDENTITY}
            if self.identity is None:
                self.identity = same(message[1])
            if same(message[1]) != self.identity:   # a restarted worker that builds otherwise never serves
                w.ident = f'restarted with another identity: {same(message[1])}';w.state = 'failed';self._kill(w)
            else:
                w.state = 'ready';w.ident = message[1]
        elif kind == 'init_failed':
            w.ident = message[1];w.state = 'failed'
        elif kind == 'worlds_done':
            task = message[1]
            if task is not w.busy and task != w.busy:
                return []
            w.busy = None;out.append(Result(task, 'worlds', index, item=message[4], seconds=message[2], compiles=message[3], received=message[5]))
        elif kind in ('prepared', 'error'):
            task = message[1]
            if task is not w.busy and task != w.busy:
                return []
            w.busy = None
            if kind == 'prepared':
                out.append(Result(task, 'prepared', index, item=message[4], labels=message[5], seconds=message[2], compiles=message[3],
                                  unpack_seconds=message[6], bytes=message[7], received=message[8]))
            else:
                out.append(Result(task, 'error', index, seconds=message[2], compiles=message[3], text=message[4], received=message[5]))
        elif kind == 'gone' and w is self.local:   # its prepare ran past the timeout: fail it, give the slot up, move its battles
            if w.busy is not None:
                out.append(Result(w.busy, 'died', index, text='RuntimeError: the in-process prepare ran past the prepare timeout'));w.busy = None
            w.generation += 1;w.state = 'failed';w.ident = dict(local=True, failed='prepare timeout')   # a late result is ignored
            queued = [item for item in w.queue if item[0] == 'prepare'];w.queue = deque()
            for battle in [b for b, i in self.affinity.items() if i == index]:
                del self.affinity[battle]
            for _, (task, request, on_start) in queued:
                battle = request.get('battle_id') if isinstance(request, dict) else None
                self.workers[self.assign(battle if type(battle) is str else None, exclude=index)].queue.append(('prepare', (task, request, on_start)))
        elif kind == 'gone':
            code = w.process.poll() if w.process is not None else None
            if code is None and w.process is not None:
                self._kill(w);code = w.process.wait()
            text = f'RuntimeError: prepare worker {index} exited (code {code})' + (' after its task ran past the prepare timeout' if w.state == 'killing' else '')
            if w.busy is not None:
                out.append(Result(w.busy, 'died', index, text=text));w.busy = None
            queued = [item for item in w.queue if item[0] == 'prepare'];w.queue = deque()   # its drops: for sessions it no longer has
            if self.closed or w.state == 'stopped':
                w.state = 'stopped'
            else:
                now = time.monotonic();w.restarts = [t for t in w.restarts if now - t < 600.] + [now]
                if w.state == 'failed' or len(w.restarts) > self.max_restarts:
                    w.ident = w.ident if w.state == 'failed' and w.ident else text;w.state = 'failed'
                else:
                    try:
                        self._spawn(w)   # it comes back empty (its warm-in first) and takes new battles
                    except Exception as error:   # noqa: BLE001 -- it fails for good; its died task and queue are still answered below
                        w.ident = f'restart failed: {type(error).__name__}: {error}';w.state = 'failed'
                # Its battles move now, in their order, to the live workers (they fold from scratch there: an evicted
                # battle's path), so none waits for the restart's warm-in.
                for battle in [b for b, i in self.affinity.items() if i == index]:
                    del self.affinity[battle]
                for _, (task, request, on_start) in queued:
                    battle = request.get('battle_id') if isinstance(request, dict) else None
                    others = [x for x in self.workers if x.state == 'ready' and x is not w]
                    if others:
                        target = self.workers[self.assign(battle if type(battle) is str else None, exclude=index)]
                    elif w.state != 'failed':
                        target = w   # it restarts: they wait for it
                    else:   # no live worker to take it: answered now, never left on a dead worker's queue
                        out.append(Result(task, 'died', index, text=text + ' (no live prepare worker)'));continue
                    target.queue.append(('prepare', (task, request, on_start)))
        out.extend(self.pump())
        return out

    # synchronous use (the live search's process_many, warmup) ------------------------------------------------------------------
    def prepare_many(self, requests):
        """Every request prepared, results in order (Result per request), in parallel across workers (battle affinity holds)."""
        tasks = []
        for i, request in enumerate(requests):
            battle = request.get('battle_id') if isinstance(request, dict) else None
            key = ('task', self.tasks, i)
            self.submit(key, request, battle if type(battle) is str else None);tasks.append(key)
        return self._run(tasks)

    def rss(self):
        """{worker index: RSS bytes} of the live workers (/proc)."""
        return {w.index: rss_bytes(w.process.pid) for w in self.workers if w.process is not None and w.process.poll() is None}


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        from gpu_engine import prepare_pool   # the module itself, not __main__: what it pickles must unpickle in the server
        sys.exit(prepare_pool.worker_main(int(sys.argv[2])))
    sys.exit('usage: python -m gpu_engine.prepare_pool --worker FD')
