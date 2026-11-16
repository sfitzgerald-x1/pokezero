"""MEASUREMENT ONLY: one decision's worlds built across idle prepare workers (the batching pipeline's opt-in --split-worlds).

Never on by default, and never to be enabled in production before worlds' own world-range seam lands (prepare_root,
sample_range, assemble: worlds' PR after its prepare speedups). Until then this module is the narrow internal interface that
seam will replace: one decision's per-world loop over an index range (world_range), run by a pool worker on its own constructor,
and a sampler that farms phase_d_protocol_world.sample's world loop out to a backend (Backend.worlds) for the calls made inside
`split(backend)` on this thread, and is phase_d_protocol_world.sample itself everywhere else.

world_range is phase_d_protocol_world.sample's per-world body as it stands at 43937f12, copied, not changed (scripts' k-split
harness runs the same copy): the start sampler and v5_public_world.materialize on the decision's own public root and carried
root, its own ProtocolInputs (the per-root _once memo recomputed per range). A split sample returns the same WorldSet
serial sample() returns: the carried root, the worlds, their seeds and diagnostics in index order (the k-split harness checks
it byte for byte, prepare_pool.digest, on W16/W32/W64 roots; a worlds change to that body breaks the equality, so it must be
re-checked on every tree before a measurement). The ship cost (pickling the roots for the helpers, their unpickling, the
worlds' way back) is inside prepare_roots, so it is counted in prepare_seconds."""
import random
import threading
import time

_local = threading.local()
_installed = {}


def world_range(root, detached, lines, seed, indices, *, catalog, metadata, source):
    """[(index, world_seed, World, diagnostic)] for these world indices of one decision: sample()'s per-world body, verbatim."""
    from . import phase_d_protocol_world as protocol, v5_public_world, device_public_world_state as states
    from pokezero.determinization import _gen3_randbat_belief_start_override_result
    inputs = protocol.ProtocolInputs(detached, tuple(lines), catalog, metadata, source);game = protocol.game_lines_sha256(lines);out = []
    for index in indices:
        world_seed, kernel_seed = protocol.world_seeds(root, detached, game, seed, index)
        try:
            start, reason = _gen3_randbat_belief_start_override_result(context=root.context, set_source=source.original,
                rng=random.Random(world_seed), team_size=6, canonicalize_revealed_variants=True)
            if start is None:
                unknown = [];protocol._downstream_unknowns(unknown, 'Original public sampler unavailable: ' + str(reason))
                made = protocol.ProtocolWorld(None, None, (reason or 'public-sampler-unavailable',), mapper_diagnostics=tuple(unknown))
            else:
                made = v5_public_world.materialize(root, start, prepared=inputs, history=None, own_static=None, seed=kernel_seed)
            out.append((index, world_seed, states.World(index, made.state, made.public, made.refusals, kernel_seed, False),
                        dict(world=index, checks=list(made.mapper_diagnostics))))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            unknown = [];protocol._downstream_unknowns(unknown, 'Production sample prerequisite refused: ' + str(error))
            out.append((index, world_seed, states.World(index, None, None, (str(error),), kernel_seed, False), dict(world=index, checks=unknown)))
    return out


def worker_range(constructor, blob):
    """A pool worker's side: (root, detached, lines, seed, indices) pickled -> the range's worlds packed for the way back."""
    import jax
    from . import prepare_pool as pp
    with jax.default_device(jax.devices('cpu')[0]):
        root, detached, lines, seed, indices = pp.pickle.loads(blob)
        rows = world_range(root, detached, lines, seed, indices, catalog=constructor.catalog, metadata=constructor.metadata, source=constructor.source)
        jax.block_until_ready([(w.state, w.public) for _, _, w, _ in rows if w.state is not None])
    shells, packed = pp.pack_worlds([w for _, _, w, _ in rows])
    return pp.dumps(([(i, s, d) for i, s, _, d in rows], shells, packed))


class Backend:
    """Where a split sample's world ranges go: `ranges(blobs)` sends each pickled range to a helper and returns their packed
    replies in the same order (the pipeline's scheduler does the sending, its idle workers the work)."""
    def __init__(self, k, ranges):
        self.k, self.ranges = int(k), ranges
        self.parts = {}

    def worlds(self, root, detached, lines, seed, count):
        from . import prepare_pool as pp
        import jax
        k = max(1, min(self.k, count));indices = [list(range(count))[i::k] for i in range(k)];clock = time.perf_counter()
        blobs = [pp.dumps((root, detached, tuple(lines), seed, r)) for r in indices];shipped = time.perf_counter()
        replies = self.ranges(blobs);back = time.perf_counter();rows = {}
        for reply in replies:
            meta, shells, packed = pp.pickle.loads(reply)
            for (i, s, d), w in zip(meta, pp.unpack_worlds(shells, packed, jax.devices('cpu')[0])):
                rows[i] = (s, w, d)
        self.parts = dict(k=k, ship=shipped - clock, helpers=back - shipped, gather=time.perf_counter() - back,
                          bytes=sum(len(b) for b in blobs) + sum(len(r) for r in replies))
        return tuple(rows[i][1] for i in range(count)), tuple(rows[i][0] for i in range(count)), tuple(rows[i][2] for i in range(count))


def _split_sample(root, detached, *, lines, catalog, source, metadata, config, battle_id, seed, count):
    """phase_d_protocol_world.sample, or inside split(backend) on this thread: its preamble, then the backend's worlds."""
    from . import phase_d_protocol_world as protocol, phase_d_live_worlds as live_worlds
    backend = getattr(_local, 'backend', None)
    if backend is None:
        return _installed['sample'](root, detached, lines=lines, catalog=catalog, source=source, metadata=metadata, config=config,
                                    battle_id=battle_id, seed=seed, count=count)
    if count not in (2, 8, 16, 32, 64):
        raise ValueError('declared2/8/16/32/64 world budget required')
    if detached.source_binding_sha256 != source.binding_sha256:
        raise live_worlds.Refusal('SAME_PUBLIC_ROOT_SOURCE_BINDING_REQUIRED')
    detached = protocol._carried_root(root, detached, lines, metadata)
    worlds, seeds, diagnostics = backend.worlds(root, detached, lines, seed, count)
    return live_worlds.WorldSet(detached, worlds, seeds, count, diagnostics)


def install():
    """Route phase_d_protocol_world.sample through _split_sample in this process (idempotent). Outside split(), it calls the
    original: every other caller's sample is unchanged."""
    from . import phase_d_protocol_world as protocol
    if 'sample' not in _installed:
        _installed['sample'] = protocol.sample;protocol.sample = _split_sample


class split:
    """Context: this thread's sample() calls farm their world loop to `backend`."""
    def __init__(self, backend):
        self.backend = backend

    def __enter__(self):
        install();_local.backend = self.backend
        return self.backend

    def __exit__(self, *exc):
        _local.backend = None
