"""Depth-2 turn-2 leaf stepping: generated CUDA against XLA on the same lanes, from real D3 chunk children.

The question: is generated CUDA faster than XLA at depth-2 block sizes? The depth-2 pricer
(device_depth2_pricer) spends about half its device time in turn 2, whose blocks step child_block
children x k_own*k_opp action pairs (2,187 lanes at child_block 27, full width) through
dispatch -> advance (transition.step_with_public under the child's own chance) -> the learner's leaf
observe, under jax.lax.map. The D pricer's generated CUDA leaves (branch scott/v2e-p7-cuda-leaves-parked)
were bitwise but 1.8x slower than XLA at depth 1; this measures the same trade at depth-2 lane counts.

Inputs are real: one D3 bank chunk's requests (built as d3_depth2_gate --d3-root --chunk builds them;
the loader is copied below), their worlds materialized on host as the pricers do, then D3's turn 1
(the depth-2 kernel's own _make_child_joint over the pricing-order x legal-reply grid) gives every
child's engine state, trackers and both seats' child masks. Children are the kernel's expanded ones
by the codes that need no actor forward (enumerated, not invalid, not terminal, no encoder error on
the learner's view, no encoder error on a requested opponent's view); the forward-dependent codes
(4 actor refused, 6 child prior refused, 9 root prior refused) are not evaluated, which only changes
which children are available, never a lane's work. Lanes are packed as the kernel's block does at a
uniform child prior: each child's 81 (own action, opponent action) slots from `ranked` with zero
logits (each seat's legal encoder lanes ascending, dead slots repeating the first, 0 for a waiting
seat), with the child's own executor.sample draw.

Two paths run on the SAME children, views, pairs and chance capsules:
 (a) XLA: jax.vmap of device_depth2_pricer._make_leaf_joint (the kernel's own turn-2 function: one
     child per vmap row, pairs mapped inside), one jitted call per lane count;
 (b) CUDA: v2e_modules.pricer_lane_functions' `leaf` (the same dispatch/advance/observe sequence as a
     per-lane function) compiled by v2e_gen to a generated kernel, called as the parked cuda_leaves
     path calls its joint: every lane carries its child's state, trackers, dispatch targets and chance
     (j.repeat x81 inside the timed program), outputs reshaped back to [children, 81].
Both re-observe nothing: the child's dispatch views and chance are computed once, outside the timing,
and shared (the kernel recomputes them per block; that work is identical in both paths and about
2/81 of a block's encodes). Every output both paths produce is compared bit for bit on device:
invalid/done/winner, the learner's view error, every encoder input array, and depth2_digest of the
encoder inputs. --check-state also compiles `leaf_state` (the same lane function returning the
stepped engine state) and compares every state leaf against the lane function run through XLA; it is
not timed. Any mismatch (or a nonzero generated error word) fails the run (exit 1).

Per lane count L: XLA and CUDA compile seconds, warm device ms (min and median of --repeats after one
untimed call), lanes per second, and the CUDA/XLA ratio (> 1: CUDA slower). The generated module's
trace/generate/build seconds and arena words are reported once. --xla-lanes also times the lane
function itself through XLA in the generated (per-lane, x81 replicated) layout, separating the
layout's cost from the code generator's. Progress is JSON lines; the result is bench.json in a new
--out directory. All paths are arguments.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np

SCHEMA = 'pokezero.depth2-leaf-bench.v1'
LANES = 81                                  # D3's joint lanes per world = the depth-2 full-width pair grid (9 x 9)
DEFAULT_LANES = (2187, 6561, 19683, 59049)  # child_block 27, 81, 243, 729 at full width
OUTPUTS = ('invalid', 'done', 'winner', 'error', 'inputs')
SCALARS = ('invalid', 'done', 'winner', 'error')   # one value per lane (the leaf joint's [.., 1] row axis squeezed)
INPUT_ORDER = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'history_mask')
PEER_VIEW_ERROR = 5
UNEVALUATED_CODES = 'pre 9 (root prior refused), child 4 (actor refused), child 6 (child prior refused): need the actor forward'


def log(message, **values):
    print(json.dumps(dict(message=message, time=time.strftime('%H:%M:%S'), **values), default=str), flush=True)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--d3-root', required=True, help='the D3 bank run root (prespec.json, runtime.json, run/)')
    parser.add_argument('--chunk', type=int, required=True, help='bank chunk index; chunks 0..N are collected')
    parser.add_argument('--memory-policy', action='store_true', help="apply the bank runtime's JAX memory policy")
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--lanes', type=int, nargs='+', default=list(DEFAULT_LANES), help='lane counts L (multiples of 81; L/81 children each)')
    parser.add_argument('--repeats', type=int, default=5, help='timed warm calls per path and lane count (after one untimed call)')
    parser.add_argument('--max-worlds', type=int, default=256, help='worlds stepped through turn 1 (81 children each)')
    parser.add_argument('--max-roots', type=int, help='materialize only the first N requests of the chunk')
    parser.add_argument('--build-dir', help='generated CUDA build/cache directory (default: $V2E_CUDA_BUILD or ~/.cache/v2e-cuda)')
    parser.add_argument('--build-workers', type=int, help='parallel nvcc workers (default: $V2E_BUILD_WORKERS or min(32, cpus))')
    parser.add_argument('--check-state', action='store_true', help='also build leaf_state and compare every stepped state leaf (not timed)')
    parser.add_argument('--xla-lanes', action='store_true', help='also time the lane function through XLA in the per-lane layout')
    parser.add_argument('--skip-cuda', action='store_true', help='XLA only (no nvcc): the preparation and XLA timing; nothing is compared')
    args = parser.parse_args(argv)
    if args.chunk < 0:parser.error('--chunk N >= 0 required')
    if args.repeats < 1 or args.max_worlds < 1:parser.error('positive --repeats and --max-worlds required')
    if args.max_roots is not None and args.max_roots < 1:parser.error('positive --max-roots required')
    if args.build_workers is not None and args.build_workers < 1:parser.error('positive --build-workers required')
    if any(L <= 0 or L % LANES for L in args.lanes):parser.error(f'--lanes must be positive multiples of {LANES}')
    if len(set(args.lanes)) != len(args.lanes):parser.error('--lanes repeats a value')
    return args


# ---------------------------------------------------------------------------
# Pure helpers (numpy; CPU-tested).

def lane_pairs(own_legal, peer_legal, own_requested, peer_requested, k_own=9, k_opp=9):
    """[children, k_own*k_opp, 2] int32 (owner action, opponent action): the depth-2 kernel's turn-2 slots at a
    uniform child prior. `ranked` with zero logits orders each seat's legal ENCODER lanes ascending; slots past
    the legal count repeat the first choice (dead slots, never live); a waiting seat enumerates action 0 alone.
    Slot order within a child is (own slot major, opponent slot minor), as the kernel's `pair`."""
    own_legal, peer_legal = np.asarray(own_legal, bool), np.asarray(peer_legal, bool)
    own_requested, peer_requested = np.asarray(own_requested, bool), np.asarray(peer_requested, bool)
    if own_legal.shape != peer_legal.shape or own_legal.shape[1:] != (9,) or own_requested.shape != peer_requested.shape != (len(own_legal),):
        raise ValueError('[children, 9] legal masks and [children] request flags required')
    def ranked(mask, requested, k):
        order = np.argsort(np.where(mask, 0., np.inf), axis=-1, kind='stable')
        count = np.where(requested, np.minimum(mask.sum(axis=-1), k), 1)
        slot = np.arange(k);live = slot < count[:, None]
        chosen = np.take_along_axis(order, np.where(live, slot, 0), axis=-1)
        return np.where(requested[:, None], chosen, 0).astype(np.int32)
    own, opp = ranked(own_legal, own_requested, k_own), ranked(peer_legal, peer_requested, k_opp)
    n = len(own_legal)
    grid = np.stack((np.broadcast_to(own[:, :, None], (n, k_own, k_opp)), np.broadcast_to(opp[:, None, :], (n, k_own, k_opp))), axis=-1)
    return np.ascontiguousarray(grid.reshape(n, k_own * k_opp, 2))


def select_children(expanded, needed):
    """`needed` flat child indices among the expanded ones: evenly strided over them (index order kept) when
    there are enough, cycled from the start when there are not. Returns (indices, cycled)."""
    flat = np.flatnonzero(np.asarray(expanded, bool))
    if type(needed) is not int or needed <= 0:raise ValueError('positive child count required')
    if not len(flat):raise ValueError('no expanded child')
    if len(flat) >= needed:return flat[np.arange(needed) * len(flat) // needed], False
    return flat[np.arange(needed) % len(flat)], True


def _bits(x):
    import jax
    import jax.numpy as j
    if j.issubdtype(x.dtype, j.floating):
        return jax.lax.bitcast_convert_type(x, {2: j.uint16, 4: j.uint32, 8: j.uint64}[x.dtype.itemsize])
    return x


def compare_trees(a, b):
    """{leaf path: rows differing (leading axis), or 'geometry ...' / 'structure'} between two pytrees of arrays
    (device or host); {} when every leaf is byte-equal. Floats compare by their bits (NaN payloads included)."""
    import jax
    import jax.numpy as j
    la, ta = jax.tree_util.tree_flatten_with_path(a);lb, tb = jax.tree_util.tree_flatten_with_path(b)
    if ta != tb:return {'': 'structure'}
    report = {}
    for (path, x), (_, y) in zip(la, lb):
        key = jax.tree_util.keystr(path) or '.'
        if np.dtype(x.dtype) != np.dtype(y.dtype) or tuple(x.shape) != tuple(y.shape):
            report[key] = f'geometry {tuple(x.shape)}{np.dtype(x.dtype)} vs {tuple(y.shape)}{np.dtype(y.dtype)}';continue
        if not np.prod(x.shape):continue
        x, y = _bits(j.asarray(x)), _bits(j.asarray(y))
        differing = int(x != y) if x.ndim == 0 else int(j.sum(j.any((x != y).reshape(x.shape[0], -1), axis=1)))
        if differing:report[key] = differing
    return report


def summarize(seconds, lanes):
    seconds = list(seconds)
    return dict(warm_seconds=seconds, min_ms=1e3 * min(seconds), median_ms=1e3 * statistics.median(seconds),
        lanes_per_second_min=lanes / min(seconds), lanes_per_second_median=lanes / statistics.median(seconds))


# ---------------------------------------------------------------------------
# JAX stages (JAX imported lazily: the bank's memory policy must precede the first import).

def make_turn1(executor):
    """jitted: flat worlds -> flat children (lane = world * 81 + ordinal) with the depth-2 kernel's turn-1 facts.

    The kernel's `run` up to its child joint, verbatim where no actor forward is involved: both root views,
    the pre codes 1/2/3../8/10, D's pricing-order x ascending-legal-reply pairs padded with the first pair,
    the root's chance capsule and _make_child_joint; then the request flags from the engine masks, both
    seats' encoder masks and the child codes 1/2/3/5. `expanded` is the kernel's, minus the forward codes."""
    import jax
    import jax.numpy as j
    from gpu_engine import device_depth2_pricer as pricer
    child_joint = pricer._make_child_joint(executor)
    def views(state, public, owner):
        masks = executor.legal(state);seat = j.asarray([owner], j.int32)
        own = executor.observe(state, public, seat, masks[:, owner])
        peer = executor.observe(state, public, 1 - seat, masks[:, 1 - owner])
        return own.observation, peer.observation, own.inputs
    def run(a):
        states, publics, owners = a['states'], a['publics'], a['owner'];n = owners.shape[0]
        own, peer, own_inputs = jax.vmap(views)(states, publics, owners)
        legal = peer.legal_action_mask[:, 0];requested = j.any(legal, axis=-1)
        pre = j.where(own.error[:, 0] != 0, 1, 0)
        pre = j.where((pre == 0) & ~j.all(own.legal_action_mask[:, 0] == a['legal'], axis=-1), 2, pre)
        if set(own_inputs) != set(a['expected']):pre = j.where(pre == 0, 10, pre)
        for index, key in enumerate(k for k in INPUT_ORDER if k in a['expected']):
            if key not in own_inputs:continue
            value = own_inputs[key];same = j.all(value == a['expected'][key], axis=tuple(range(1, value.ndim)))
            pre = j.where((pre == 0) & ~same, 3 + index, pre)
        pre = j.where((pre == 0) & (peer.error[:, 0] != 0), 8, pre)
        reply_count = j.where(requested, j.sum(legal, axis=-1), 1)
        reply_order = j.argsort(j.where(legal, j.arange(9), j.arange(9) + 9), axis=-1)
        reply_order = j.where(requested[:, None], reply_order, 0)
        order, sizes = a['order'], a['order_size']
        ordinal = j.broadcast_to(j.arange(LANES), (n, LANES));required = sizes * reply_count
        action_row = j.where(ordinal < required[:, None], ordinal // reply_count[:, None], 0)
        reply_row = j.where(ordinal < required[:, None], ordinal % reply_count[:, None], 0)
        pairs = j.stack((j.take_along_axis(order, action_row, axis=1), j.take_along_axis(reply_order, reply_row, axis=1)), axis=-1)
        chance = jax.vmap(executor.sample)(states)
        (invalid, done, winner, error, _, child_states, child_publics, masks, own_legal2, peer_legal2, peer_error, _
            ) = jax.vmap(child_joint)(states, publics, own, peer, owners[:, None], requested[:, None], pairs, chance)
        invalid, done, winner, error, peer_error = (x[:, :, 0] for x in (invalid, done, winner, error, peer_error))
        masks = masks[:, :, 0];seat = owners[:, None, None, None]
        own_req2 = j.any(j.take_along_axis(masks, seat, axis=2)[:, :, 0], axis=-1)
        peer_req2 = j.any(j.take_along_axis(masks, 1 - seat, axis=2)[:, :, 0], axis=-1)
        own_legal2, peer_legal2 = own_legal2[:, :, 0], peer_legal2[:, :, 0]
        terminal_ok = (winner >= 0) & (winner <= 2)
        errors = j.where(invalid, 1, j.where(done & ~terminal_ok, 2, j.where(~done & (error != 0), 3, 0)))
        errors = j.where((errors == 0) & ~done & peer_req2 & (peer_error != 0), PEER_VIEW_ERROR, errors)
        enumerated = (ordinal < required[:, None]) & (pre == 0)[:, None]
        expanded = enumerated & (errors == 0) & ~done
        flat = lambda x: x.reshape((n * LANES,) + x.shape[2:])
        requested2 = j.where(owners[:, None, None] == j.arange(2), own_req2[..., None], peer_req2[..., None])
        return dict(child_states=jax.tree.map(flat, child_states), child_publics=jax.tree.map(flat, child_publics),
            owner=j.repeat(owners, LANES), requested=flat(requested2), own_requested=flat(own_req2), peer_requested=flat(peer_req2),
            own_legal=flat(own_legal2), peer_legal=flat(peer_legal2), pair1=flat(pairs), enumerated=flat(enumerated),
            expanded=flat(expanded), done=flat(done & enumerated), error=flat(j.where(enumerated, errors, 0)), pre=pre)
    return jax.jit(run)


def make_child_views(executor):
    """jitted: the kernel's block re-observation of the children (both seats' dispatch views) and each child's chance."""
    import jax
    import jax.numpy as j
    def views(state, public, owner):
        masks = executor.legal(state);seat = j.asarray([owner], j.int32)
        own = executor.observe(state, public, seat, masks[:, owner])
        peer = executor.observe(state, public, 1 - seat, masks[:, 1 - owner])
        return own.observation, peer.observation
    def run(states, publics, owners):
        own, peer = jax.vmap(views)(states, publics, owners)
        return own, peer, jax.vmap(executor.sample)(states)
    return jax.jit(run)


def make_xla_runner(executor):
    """(a) the kernel's turn-2 block step: jax.vmap over children of _make_leaf_joint (pairs mapped inside)."""
    import jax
    from gpu_engine import device_depth2_pricer as pricer
    leaf_joint = pricer._make_leaf_joint(executor)
    def run(states, publics, own, peer, owners, requested, pairs, chance):
        invalid, done, winner, error, inputs = jax.vmap(leaf_joint)(states, publics, own, peer, owners[:, None], requested, pairs, chance)
        return dict(invalid=invalid[:, :, 0], done=done[:, :, 0], winner=winner[:, :, 0], error=error[:, :, 0], inputs=inputs)
    return jax.jit(run)


def make_lane_runner(leaf, names=OUTPUTS, lanes=LANES):
    """(b) the generated layout: lane = child * lanes + pair; every lane carries its child's state, trackers,
    dispatch targets, owner, request flags and chance (j.repeat inside the timed program, as the parked joint),
    pairs flattened in the same order; outputs reshaped back to [children, lanes, ...] and the per-lane row
    axis squeezed on the scalar outputs. `leaf(*lane_args) -> (outputs, error_word)`: a v2e_gen.Generated module,
    or a stand-in that runs the lane function through JAX."""
    import jax
    import jax.numpy as j
    def run(states, publics, own, peer, owners, requested, pairs, chance):
        children = owners.shape[0]
        rep = lambda tree: jax.tree.map(lambda x: j.repeat(x, lanes, axis=0), tree)
        outs, word = leaf(*rep((states, publics, own, peer, owners[:, None], requested)), pairs.reshape(children * lanes, 2), rep(chance))
        unbranch = lambda x: x.reshape((children, lanes) + x.shape[1:])
        result = {name: jax.tree.map(unbranch, x) for name, x in zip(names, outs)}
        for name in SCALARS:result[name] = result[name][:, :, 0]
        return result, word
    return jax.jit(run)


def jax_stand_in(fn):
    """A lane function with the Generated calling convention (outputs, error word), run by JAX."""
    import jax.numpy as j
    return lambda *args: (fn(*args), j.zeros(1, j.uint32))


def measure(fn, args, repeats):
    """Compile, one untimed call, then `repeats` timed calls (block_until_ready each)."""
    import jax
    clock = time.perf_counter();compiled = fn.lower(*args).compile();compile_s = time.perf_counter() - clock
    clock = time.perf_counter();out = jax.block_until_ready(compiled(*args));first_s = time.perf_counter() - clock
    times = []
    for _ in range(repeats):
        clock = time.perf_counter();jax.block_until_ready(compiled(*args));times.append(time.perf_counter() - clock)
    return out, dict(compile_seconds=compile_s, first_call_seconds=first_s), times


def pack_worlds(requests, worlds, recipe, max_worlds):
    """The turn-1 data of the first `max_worlds` materialized worlds (host numpy; one row per world)."""
    import jax
    rows = [(q, wi) for q, pair in enumerate(worlds) for wi, world in enumerate(pair) if world.state is not None][:max_worlds]
    if not rows:raise SystemExit('no materialized world in the selected requests')
    def stacked(values):
        values = jax.device_get(tuple(values))
        return jax.tree.map(lambda *leaves: np.stack([np.asarray(x) for x in leaves]), *values)
    data = dict(states=stacked([worlds[q][wi].state for q, wi in rows]), publics=stacked([worlds[q][wi].public for q, wi in rows]),
        expected=stacked([requests[q].root_inputs for q, _ in rows]),
        owner=np.asarray([requests[q].plans.root.address.owner for q, _ in rows], np.int32),
        legal=np.asarray([requests[q].root_legal for q, _ in rows], bool),
        order=np.zeros((len(rows), 9), np.int32), order_size=np.ones(len(rows), np.int32))
    for i, (q, _) in enumerate(rows):
        r = requests[q];order = recipe.pricing_order(r.plans.root.address, r.root_logits, r.root_legal)
        data['order'][i] = order + (order[0],) * (9 - len(order));data['order_size'][i] = len(order)
    return data, rows


def histogram(values):
    keys, counts = np.unique(np.asarray(values), return_counts=True)
    return {str(int(k)): int(c) for k, c in zip(keys, counts)}


def memory_stats():
    import jax
    stats = jax.local_devices()[0].memory_stats() or {}
    return dict(peak_bytes_in_use=stats.get('peak_bytes_in_use'), bytes_in_use=stats.get('bytes_in_use'))


def module_report(generated):
    """The generated module's identity and v2e_j2c/v2e_gen stats (trace_s, generate_s, build_s, equations, chunks, lines, mode...)."""
    m = generated.module
    report = dict(name=m.name, library=str(generated.path), arena_words=int(m.arena_words), inputs=len(m.inputs), outputs=len(m.outputs),
        lanes_traced=int(m.inputs[0][1][0]) if m.inputs else None)
    report.update(dict(m.stats));report.setdefault('mode', generated.mode)
    return report


# ---------------------------------------------------------------------------
# The D3 bank chunk's requests: gpu_engine.d3_depth2_gate.bank_chunk_requests (branch scott/v2e-depth2-gate,
# 4e5fa015), copied so this tool imports at the depth-2 pricer merge base; `prepared` is added to the result
# (the engine and observation tables the generated modules trace over).

def _probe(worker, runtime, jax):
    try:worker._probe(runtime)
    except ValueError as error:
        # The probe pins the device count/kind of the original worker; keep its numeric settings (as the D3 gates).
        log('PROBE_DIFFERS', error=str(error));jax.config.update('jax_enable_x64', False);jax.config.update('jax_default_matmul_precision', 'highest')


def bank_chunk_requests(args):
    """One D3 bank chunk's Requests, as d3_chunk_equal builds the chunk and Provider._prepare_batched the requests."""
    root = Path(args.d3_root)
    from pokezero import v5_worker as worker, v5_run as runs
    runtime_path = root / 'runtime.json'
    runtime = worker._record(dict(path=str(runtime_path), sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest()))
    if args.memory_policy:worker._memory(runtime['memory_policy'], apply=True)   # must precede the first JAX import
    import jax
    from pokezero.v5_model import V5ModelConfig
    from pokezero.v5_loss import V5LossConfig
    from pokezero import v5_training as training
    from gpu_engine import v5_device_rollout as rollout, v5_model
    from gpu_engine.v5_device_backend import RuntimeBackend
    from gpu_engine.v5_device_teacher import Provider
    profile, vocabulary, gender, _ = worker._runtime_metadata(runtime, expected_pool_sha256=runtime['constructor_pool']['sha256'])
    d = json.loads((root / 'run' / 'declaration.json').read_text())
    loss_fields = {k: v for k, v in d['loss'].items() if k in V5LossConfig.__dataclass_fields__}
    batches = tuple(tuple(runs.Batch(tuple(runs.Game(**g) for g in x['games']), x['constructor_seed'], x['sampling_seed'],
        x.get('gender_seed'), x.get('gender_domain'), tuple(x.get('gender_root_ids', ()))) for x in it) for it in d['iterations'])
    plan = runs.Plan(d['run_id'], V5ModelConfig.from_metadata(d['model']), V5LossConfig(**loss_fields), training.TrainConfig(**d['train']),
        batches, d['pool_sha256'], d['selection_seed'], d['historical_fraction'], d['historical_window'], d['max_requests'],
        runs.Watchdog(**d['watchdog']), checkpoint_updates=d['checkpoint_updates'], discount=d['discount'])
    declared = json.loads((root / 'run' / 'journal' / '000000000.json').read_text())['payload']['declaration_sha256']
    if runs._sha(plan.manifest()) != declared:raise SystemExit('rebuilt plan differs from the D3 declaration')
    _probe(worker, runtime, jax)
    inputs = worker.Inputs(plan, profile, json.dumps(runtime), json.dumps(vocabulary), json.dumps(gender), runs._sha(plan.manifest()))
    prepared, pool, _, _ = worker.prepare_device_runtime(inputs)
    provider = Provider(prepared, plan, frame_batch=True, chunk_batch=True, vectorized=True, batched_prep=True)
    backend = RuntimeBackend(prepared, pool, gender_policy=gender, teacher=provider);backend.check_plan(plan)
    events = (json.loads(p.read_text()) for p in sorted((root / 'run' / 'journal').glob('*.json')))
    snapshot = next(e['payload'] for e in events if e['kind'] == 'SNAPSHOT' and e['payload']['iteration'] == -1)
    params, config, _ = v5_model.load(root / 'run' / snapshot['archive'], expected_archive_sha256=snapshot['archive_sha256'],
        expected_checkpoint_sha256=snapshot['checkpoint_sha256'], expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
        expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256, enable_v5=True)
    if config != plan.model:raise SystemExit('actor configuration differs')
    batch = plan.iterations[0][0];ordinal = {g.stream_id: i for i, g in enumerate(plan.games)};snap = runs.Snapshot(**snapshot)
    assignments = json.loads(runs._json(tuple(runs.assignment(plan, g, ordinal[g.stream_id], snap, ()) for g in batch.games)))
    archive = snapshot['archive_sha256']
    bindings = rollout.Bindings(plan.model.category_vocabulary_sha256, plan.model.label_vocabulary, prepared.source_sha256,
        (archive,) * 2, (True, True), backend.declaration_sha256)
    actor_bank = rollout.actor_bank_from_assignments(bindings, (archive,), current_archive=archive, assignments=assignments)
    collector = rollout.make_collector(prepared.engine_tables, prepared.observation_tables, plan.model, backend.labels, bindings,
        sampling_seed=batch.sampling_seed, chunk_steps=8, max_requests=plan.max_requests, actor_bank=actor_bank, retain_teacher_roots=True)
    carry = jax.block_until_ready(backend._construct(batch));log('CONSTRUCTED', games=len(batch.games))
    for _ in range(args.chunk + 1):
        device = jax.block_until_ready(collector(carry, (params,)));carry = device.carry
        host = rollout.boundary(device._replace(carry=carry._replace(state=None, public=None, truth=None), roots=None))
    provider.bind(actor_bank, (params,));provider.reset_metrics()
    records, grouped = provider._prepare_batched(device.roots, host.frames, actor_bank)
    groups = [[request for _, request in items] for _, items in grouped.items()]
    log('BANK_CHUNK', chunk=args.chunk, records=len(records), selected=sum(r['status'] != 'UNSELECTED' for r in records),
        frames=sorted(grouped), requests=sum(map(len, groups)))
    return SimpleNamespace(provider=provider, prepared=prepared, plan=plan, params=(params,), groups=groups,
        source=dict(d3_root=str(root), chunk=args.chunk, declaration_sha256=declared),
        snapshot={k: snapshot.get(k) for k in ('iteration', 'checkpoint', 'checkpoint_sha256', 'archive', 'archive_sha256', 'model_sha256')})


# ---------------------------------------------------------------------------

def prepare_children(executor, requests, materializer, recipe, *, max_worlds, needed):
    """Turn 1 on the chunk's first worlds, then `needed` expanded children with their block inputs (device arrays)."""
    import jax
    import jax.numpy as j
    from gpu_engine import device_public_teacher_batch as batch
    clock = time.perf_counter()
    take = min(len(requests), (max_worlds + 1) // 2 + 8)   # two worlds per request; a few spare for refusals
    worlds = batch._numpy_worlds(requests[:take], materializer, recipe)
    data, rows = pack_worlds(requests, worlds, recipe, max_worlds);materialize_s = time.perf_counter() - clock
    clock = time.perf_counter();device_data = jax.device_put(data)
    turn1 = make_turn1(executor);lowered = turn1.lower(device_data);compiled = lowered.compile();compile_s = time.perf_counter() - clock
    clock = time.perf_counter();out = jax.block_until_ready(compiled(device_data));turn1_s = time.perf_counter() - clock
    host = {k: np.asarray(out[k]) for k in ('expanded', 'enumerated', 'done', 'error', 'pre', 'owner', 'requested', 'own_requested', 'peer_requested',
        'own_legal', 'peer_legal', 'pair1')}
    indices, cycled = select_children(host['expanded'], needed)
    pick = lambda tree: jax.tree.map(lambda x: x[j.asarray(indices)], tree)
    states, publics, owners, requested = pick(out['child_states']), pick(out['child_publics']), pick(out['owner']), pick(out['requested'])
    del out, device_data
    clock = time.perf_counter()
    own, peer, chance = jax.block_until_ready(make_child_views(executor)(states, publics, owners));views_s = time.perf_counter() - clock
    pairs = lane_pairs(host['own_legal'][indices], host['peer_legal'][indices], host['own_requested'][indices], host['peer_requested'][indices])
    inputs = (states, publics, own, peer, owners, requested, jax.device_put(pairs), chance)
    n = len(rows)
    report = dict(requests_materialized=take, worlds=n, children=n * LANES, enumerated=int(host['enumerated'].sum()), expanded=int(host['expanded'].sum()),
        terminal=int(host['done'].sum()), child_errors=histogram(host['error'][host['enumerated']]), pre_codes=histogram(host['pre']),
        unevaluated_codes=UNEVALUATED_CODES, needed=needed, cycled=cycled, selected_worlds=int(len(np.unique(indices // LANES))),
        live_pairs_per_child=float(np.mean([len(set(map(tuple, p))) for p in pairs])),
        own_requested=int(host['own_requested'][indices].sum()), peer_requested=int(host['peer_requested'][indices].sum()),
        materialize_seconds=materialize_s, turn1_compile_seconds=compile_s, turn1_seconds=turn1_s, child_views_seconds=views_s)
    return inputs, indices, report


def main(argv=None):
    args = parse(argv)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    if args.build_workers:os.environ['V2E_BUILD_WORKERS'] = str(args.build_workers)
    loaded = bank_chunk_requests(args)   # applies the memory policy before the first JAX import
    import jax
    from gpu_engine import depth2_digest as digests, v2e_modules
    devices = [str(d) for d in jax.local_devices()]
    provider = loaded.provider;executor = provider.executor
    requests = [r for g in loaded.groups for r in g][:args.max_roots]
    log('REQUESTS', count=len(requests), devices=devices)
    lanes = sorted(args.lanes);needed = max(lanes) // LANES
    inputs, indices, prep = prepare_children(executor, requests, provider.materializer, provider.recipe, max_worlds=args.max_worlds, needed=needed)
    log('CHILDREN', **prep)
    result = dict(schema=SCHEMA, created=time.strftime('%Y-%m-%dT%H:%M:%S'), source=loaded.source, snapshot=loaded.snapshot, devices=devices,
        args={k: v for k, v in vars(args).items()}, preparation=prep, modules={}, lanes=[], layout=dict(lanes_per_child=LANES,
        xla='jax.vmap over children of device_depth2_pricer._make_leaf_joint (pairs mapped inside), shared child views and chance',
        cuda='v2e_modules leaf lane function via v2e_gen; lane = child*81 + pair; child inputs repeated per lane inside the timed program',
        shared='child dispatch views (both seats) and executor.sample(child) computed once by XLA, outside the timing, for both paths'))
    write = lambda: (out / 'bench.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    states, publics, own, peer, owners, requested, pairs, chance = inputs
    targets = (states, publics, v2e_modules.dispatch_targets(own), v2e_modules.dispatch_targets(peer), owners, requested, pairs, chance)
    xla = make_xla_runner(executor)
    digest = jax.jit(lambda x: digests.batched(x, axes=2))
    generated = generated_state = None;fns = None
    if not args.skip_cuda:
        from gpu_engine import v2e_gen
        index, fns = v2e_modules.pricer_functions(loaded.prepared.engine_tables, loaded.prepared.observation_tables)
        one = lambda tree: jax.tree.map(lambda x: jax.ShapeDtypeStruct((1,) + tuple(x.shape[1:]), x.dtype), tree)
        example = (one(states), one(publics), one(targets[2]), one(targets[3]), jax.ShapeDtypeStruct((1, 1), np.int32),
            jax.ShapeDtypeStruct((1, 2), np.bool_), jax.ShapeDtypeStruct((1, 2), np.int32), one(chance))
        for name in ('leaf',) + (('leaf_state',) if args.check_state else ()):
            clock = time.perf_counter()
            module = v2e_gen.compile_function(f'depth2_{name}_{index}', fns[name], example, build_dir=args.build_dir, mode=v2e_modules.DEPTH2_MODES[name])
            result['modules'][name] = dict(module_report(module), wall_seconds=time.perf_counter() - clock)
            log('MODULE', module=name, **result['modules'][name])
            if name == 'leaf':generated = module
            else:generated_state = module
        write()
    cuda = None if generated is None else make_lane_runner(generated)
    xla_lanes = None
    if args.xla_lanes:
        if fns is None:_, fns = v2e_modules.pricer_functions(loaded.prepared.engine_tables, loaded.prepared.observation_tables)
        xla_lanes = make_lane_runner(jax_stand_in(fns['leaf']))
    equal_all = True
    for L in lanes:
        children = L // LANES;slice_ = lambda tree: jax.tree.map(lambda x: x[:children], tree)
        kernel_args = jax.block_until_ready(slice_(inputs));lane_args = jax.block_until_ready(slice_(targets))
        entry = dict(lanes=L, children=children, paths={})
        out_xla, info, times = measure(xla, kernel_args, args.repeats)
        entry['paths']['xla'] = dict(info, **summarize(times, L), memory=memory_stats());log('XLA', lanes=L, **entry['paths']['xla'])
        digest_xla = jax.block_until_ready(digest(out_xla['inputs']))
        entry['digest_sha256'] = hashlib.sha256(np.asarray(digest_xla).tobytes()).hexdigest()
        if xla_lanes is not None:
            (out_lanes, _), info, times = measure(xla_lanes, lane_args, args.repeats)
            entry['paths']['xla_lanes'] = dict(info, **summarize(times, L), memory=memory_stats(), differing=compare_trees(out_xla, out_lanes))
            log('XLA_LANES', lanes=L, **entry['paths']['xla_lanes'])
            del out_lanes
        if cuda is not None:
            (out_cuda, word), info, times = measure(cuda, lane_args, args.repeats)
            differing = compare_trees(out_xla, out_cuda)
            digest_cuda = jax.block_until_ready(digest(out_cuda['inputs']))
            digest_differing = compare_trees(dict(digest=digest_xla.reshape(L, 2)), dict(digest=digest_cuda.reshape(L, 2)))
            word = [int(x) for x in np.asarray(word)]
            entry['paths']['cuda'] = dict(info, **summarize(times, L), memory=memory_stats(), error_word=word,
                arena_bytes=4 * int(generated.module.arena_words) * L)
            entry.update(differing=differing, digest_differing=digest_differing,
                equal=not differing and not digest_differing and not any(word),
                ratio_cuda_over_xla_min=entry['paths']['cuda']['min_ms'] / entry['paths']['xla']['min_ms'],
                ratio_cuda_over_xla_median=entry['paths']['cuda']['median_ms'] / entry['paths']['xla']['median_ms'])
            if generated_state is not None:
                reference = make_lane_runner(jax_stand_in(fns['leaf_state']), names=OUTPUTS + ('state',))
                checked = make_lane_runner(generated_state, names=OUTPUTS + ('state',))
                want, _ = jax.block_until_ready(reference(*lane_args));got, word2 = jax.block_until_ready(checked(*lane_args))
                state_differing = compare_trees(want, got);kernel_differing = compare_trees(out_xla, {k: got[k] for k in OUTPUTS})
                word2 = [int(x) for x in np.asarray(word2)]
                entry['state_check'] = dict(differing=state_differing, kernel_differing=kernel_differing, error_word=word2,
                    state_leaves=len(jax.tree.leaves(got['state'])), equal=not state_differing and not kernel_differing and not any(word2))
                entry['equal'] = entry['equal'] and entry['state_check']['equal']
                del want, got
            equal_all = equal_all and entry['equal']
            log('CUDA', lanes=L, equal=entry['equal'], ratio_min=entry['ratio_cuda_over_xla_min'], ratio_median=entry['ratio_cuda_over_xla_median'],
                differing=differing, digest_differing=digest_differing, **{k: v for k, v in entry['paths']['cuda'].items() if k != 'warm_seconds'})
            del out_cuda
        del out_xla
        result['lanes'].append(entry);write()
    result.update(compared=cuda is not None, equal=equal_all if cuda is not None else None,
        status=('PASS' if equal_all else 'FAIL') if cuda is not None else 'XLA_ONLY')
    write()
    log('DONE', status=result['status'], out=str(out / 'bench.json'),
        ratios={str(e['lanes']): e.get('ratio_cuda_over_xla_median') for e in result['lanes']})
    return 0 if result['status'] != 'FAIL' else 1


if __name__ == '__main__':raise SystemExit(main())
