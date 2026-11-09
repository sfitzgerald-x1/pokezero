"""Two-pass regression replayer over V2 capture shards (fidelity-patch design 5.2).

`base` runs the jitted transition.step_with_public of this tree on every
captured (state, actions, trackers) input and stores, per shard, a 64-bit
digest of every output leaf for every lane, plus the full post arrays of the
faint-clear tracker leaves on lanes where an active faints. `patched` runs a
later tree on the same shards: inputs are migrated by leaf path (existing
leaves must keep shape and dtype; new leaves are default-filled), the patched
step runs, and every lane-step whose outputs differ from the base digests is
attributed through a per-PR table of involvement predicates (pre-state and
actions only, so a conservative superset). It reports unchanged,
changed-and-involved per effect, changed-but-uninvolved (must be 0) and
involved-but-unchanged lane-steps, and lists every changed lane-step.

Special rules (PR0): `trackers.public_order` is checked against a numpy replay
of its swap rule over the recorded install calls instead of bitwise, and with
--faint-clear the PR0 faint-time public clear may change only the listed
tracker entries on lanes where an active faints.

Both passes must run on the same JAX version and lane padding. Paths are
command-line arguments.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

SCHEMA = 'pokezero.v2e_regress.v1'
ROOT = Path(__file__).resolve().parents[1]
DIGEST_SEEDS = (0x243F6A88, 0x85A308D3)
# PR0's faint-time public clear (design 2.5): these tracker entries of the
# fainted mon may be cleared; meanlook_trap may be cleared on both seats.
FAINT_CLEAR_MON = ('confusion_turns', 'encore_turns', 'wrap_trap_turns', 'stall_counter', 'sub_present',
                   'item_swapped', 'choice_locked')
FAINT_CLEAR_SEATS = ('meanlook_trap',)
FAINT_CLEAR_LEAVES = tuple(['trackers.public_volatiles'] + [f"trackers.mon_features['{k}']" for k in FAINT_CLEAR_MON + FAINT_CLEAR_SEATS])
PUBLIC_ORDER = 'trackers.public_order'
DESTINY_BOND = 64
SLEEP_MOVES = ('spore', 'hypnosis', 'sleeppowder', 'sing', 'lovelykiss', 'grasswhistle')


def log(message, **values):
    print(json.dumps(dict(message=message, time=time.strftime('%H:%M:%S'), **values), default=str), flush=True)


# ---------------------------------------------------------------------------
# Leaf naming. Inputs (state, actions, trackers) and the PublicStepResult
# output share names for the state and tracker leaves, so a post leaf can be
# compared with its own (migrated) pre value.

def _named(prefix, tree):
    import jax
    return {prefix + jax.tree_util.keystr(path): leaf for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


def name_inputs(state, actions, trackers):
    return {**_named('state', state), 'actions': actions, **_named('trackers', trackers)}


def name_outputs(out):
    step = out.step
    return {**_named('state', step.state), 'step.invalid_actions': step.invalid_actions,
            'step.placeholder_moves': step.placeholder_moves, **_named('trackers', out.trackers)}


def capture_name(path):
    """capture.json leaf path ([0].x / [1] / [2].y) -> input name."""
    for prefix, name in (('[0]', 'state'), ('[2]', 'trackers')):
        if path.startswith(prefix):return name + path[len(prefix):]
    if path == '[1]':return 'actions'
    raise ValueError(f'unexpected capture leaf path {path}')


# ---------------------------------------------------------------------------
# Digests: per lane, two independent 32-bit sums of a bijective mix of every
# element (so any single-element change always changes both words).

def _as_u32(x):
    import jax
    import jax.numpy as j
    if x.dtype == j.bool_:return x.astype(j.uint32)
    size = x.dtype.itemsize
    width = {1: j.uint8, 2: j.uint16, 4: j.uint32, 8: j.uint32}[size]
    if size == 8:return jax.lax.bitcast_convert_type(x, j.uint32)   # [...,2]
    u = x if x.dtype == width else jax.lax.bitcast_convert_type(x, width)
    return u.astype(j.uint32)


def _fmix(h):
    import jax.numpy as j
    h = h ^ (h >> 16);h = h * j.uint32(0x85EBCA6B);h = h ^ (h >> 13);h = h * j.uint32(0xC2B2AE35)
    return h ^ (h >> 16)


def lane_digest(x, lanes):
    import jax.numpy as j
    x = j.asarray(x)
    if x.ndim == 0 or x.shape[0] != lanes:x = j.broadcast_to(x, (lanes,) + x.shape)   # a lane-global leaf
    u = _as_u32(x).reshape(lanes, -1)
    index = j.arange(u.shape[1], dtype=j.uint32)[None, :] * j.uint32(0x9E3779B9)
    return j.stack([j.sum(_fmix(u ^ (index + j.uint32(seed))), axis=1, dtype=j.uint32) for seed in DIGEST_SEEDS], axis=1)


def digest_named(named, lanes):
    import jax.numpy as j
    names = list(named)
    return names, j.stack([lane_digest(named[n], lanes) for n in names], axis=1)   # [L, N, 2]


# ---------------------------------------------------------------------------
# Migration by leaf path.

class MigrationError(ValueError):
    pass


def migrate(capture_names, leaves, template_named, *, defaults=None):
    """Shard leaves (capture order) -> {name: array} in the template's leaf set.

    Existing leaves must keep their per-lane shape and dtype. Template leaves
    absent from the capture are filled from `defaults[name](lanes)` when given,
    else by broadcasting the template's lane 0 (the initializer's neutral value).
    A captured leaf missing from the template is refused.
    """
    defaults = defaults or {}
    lanes = int(leaves[0].shape[0])
    stored = dict(zip(capture_names, leaves))
    dropped = sorted(set(stored) - set(template_named))
    if dropped:raise MigrationError(f'captured leaves missing from this tree: {dropped[:8]}')
    out, filled = {}, []
    for name, template in template_named.items():
        template = np.asarray(template)
        if name in stored:
            value = np.asarray(stored[name])
            if value.shape[1:] != template.shape[1:] or value.dtype != template.dtype:
                raise MigrationError(f'{name}: captured {value.shape[1:]}{value.dtype} vs tree {template.shape[1:]}{template.dtype}')
            out[name] = value
        else:
            value = defaults[name](lanes) if name in defaults else np.broadcast_to(template[:1], (lanes,) + template.shape[1:])
            out[name] = np.ascontiguousarray(value, dtype=template.dtype);filled.append(name)
    return out, filled


def pad_lanes(named, lanes):
    """Pad every lane-batched leaf to `lanes` with copies of lane 0 (one compile for every shard)."""
    current = int(np.asarray(named['state.hp']).shape[0])
    if lanes <= current:return named, current
    def pad(x):
        x = np.asarray(x)
        if x.ndim == 0 or x.shape[0] != current:return x
        return np.concatenate([x, np.repeat(x[:1], lanes - current, axis=0)], axis=0)
    return {k: pad(v) for k, v in named.items()}, current


# ---------------------------------------------------------------------------
# Involvement predicates (numpy, pre-state and actions only).

class Pre:
    """Numpy view of a migrated pre-state plus the move resolution of the step."""
    def __init__(self, named, vocab, sleep_talk_callable, struggle_mask):
        self.s = {k[len('state.'):]: np.asarray(v) for k, v in named.items() if k.startswith('state.')}
        self.actions = np.asarray(named['actions']);self.vocab = vocab
        self.lanes = self.actions.shape[0];self.ids = {m: i for i, m in enumerate(vocab['moves'])}
        self.sleep_talk_callable = np.asarray(sleep_talk_callable);self.struggle = np.asarray(struggle_mask)
        self.candidates = self._candidates()

    def active(self, name):
        rows = np.arange(self.lanes)[:, None];sides = np.arange(2)[None, :]
        return self.s[name][rows, sides, self.s['active']]

    def _candidates(self):
        """[L,2,K] move ids a side may execute this step (0 = none); a superset."""
        s = self.s;rows = np.arange(self.lanes)[:, None];sides = np.arange(2)[None, :]
        saved = np.where(s['pending_residual'], s['queued_actor'], -1)
        acts = np.where(saved[:, None] == sides, s['queued_actions'], self.actions)
        slot = np.clip(acts, 0, 3)
        chosen = np.where(acts < 4, s['moves'][rows, sides, s['active'], slot], 0)
        charge = self.active('charge_move')
        encore = np.where(self.active('encore_counter') > 0, self.active('encore_move'), 0)
        struggle = np.where(self.struggle, self.ids.get('struggle', 0), 0)
        base = np.stack([chosen, charge, encore, struggle], axis=-1)
        talk = self.ids.get('sleeptalk', -1)
        own = s['moves'][rows, sides, s['active']]                      # [L,2,4]
        called = np.where(self.sleep_talk_callable[own] & (base == talk).any(-1, keepdims=True), own, 0)
        return np.concatenate([base, called], axis=-1)

    def uses(self, *names):
        ids = [self.ids[n] for n in names if n in self.ids]
        return np.isin(self.candidates, ids).any(axis=(1, 2)) if ids else np.zeros(self.lanes, bool)

    def side_uses(self, *names):
        ids = [self.ids[n] for n in names if n in self.ids]
        return np.isin(self.candidates, ids).any(axis=2) if ids else np.zeros((self.lanes, 2), bool)



def sleep_clause(p):
    """A primary-sleep move while the target's side has another living non-Rest sleeper."""
    s = p.s;slp = 5
    sleepers = (s['hp'] > 0) & (s['status'] == slp) & ~s['rest_origin'].astype(bool)     # [L,2,6]
    other = np.arange(6)[None, None, :] != s['active'][..., None]
    switching = (p.actions >= 4)[..., None]          # a side switching may bring the sleeper in as target
    has_other_sleeper = (sleepers & (other | switching)).any(-1)                         # [L,2] per target side
    return (p.side_uses(*SLEEP_MOVES) & has_other_sleeper[:, ::-1]).any(axis=1)


def sleep_immunity(p):
    """A primary-sleep move or Yawn aimed at a side holding Insomnia or Vital Spirit."""
    names = p.vocab['abilities'];ids = [names.index(n) for n in ('insomnia', 'vitalspirit') if n in names]
    holder = np.isin(p.s['ability'], ids)                                                # [L,2,6]
    active = holder[np.arange(p.lanes)[:, None], np.arange(2)[None, :], p.s['active']]
    exposed = active | (holder.any(-1) & (p.actions >= 4))                              # per holder side
    return (p.side_uses(*SLEEP_MOVES, 'yawn') & exposed[:, ::-1]).any(axis=1)


def counter(name):
    def predicate(p):
        return (p.s[name] > 0).any(axis=(1, 2)) if name in p.s else np.zeros(p.lanes, bool)
    return predicate


def destiny_bond_bit(p):
    return ((p.s['volatile'] & DESTINY_BOND) != 0).any(axis=(1, 2))


def white_herb(p):
    items = p.vocab['items']
    return (p.active('item') == items.index('whiteherb')).any(axis=1) if 'whiteherb' in items else np.zeros(p.lanes, bool)


def confusion_active(p):
    return (p.active('confusion_counter') > 0).any(axis=1)


def move(*names):
    return lambda p: p.uses(*names)


# Per-PR plug-in table: effect -> predicate. Choice unlock and the effect stream
# are never involvement (design 5.2). `none` is the base-against-base self-test.
PLUGINS = {
    'none': {},
    'A1': dict(curse=move('curse'), superpower=move('superpower'), charm=move('charm'), lovelykiss=move('lovelykiss'),
               aromatherapy=move('aromatherapy'), refresh=move('refresh'), sleep_clause=sleep_clause,
               sleep_immunity=sleep_immunity),
    'A2': dict(knockoff=move('knockoff'), gigadrain=move('gigadrain'), counter=move('counter'), mirrorcoat=move('mirrorcoat'),
               destinybond=move('destinybond'), destinybond_trigger=destiny_bond_bit),
    'R': dict(bonemerang=move('bonemerang')),
    'B': dict(irontail=move('irontail'), signalbeam=move('signalbeam'), steelwing=move('steelwing'),
              meteormash=move('meteormash'), headbutt=move('headbutt'), needlearm=move('needlearm'),
              confusion=confusion_active),
    'M1': dict(hyperbeam=move('hyperbeam'), trick=move('trick'), recharge=counter('recharge_counter')),
    'M2': dict(perishsong=move('perishsong'), yawn=move('yawn'), wrap=move('wrap'), perish=counter('perish_counter'),
               drowsy=counter('yawn_counter'), partial_trap=counter('partial_trap_counter')),
    'C': dict(psychoboost=move('psychoboost'), whiteherb=white_herb, sacredfire=move('sacredfire'), facade=move('facade')),
}


def predicates(prs):
    table = {}
    for pr in prs:
        if pr not in PLUGINS:raise ValueError(f'unknown plug-in {pr}; choose from {sorted(PLUGINS)}')
        table.update(PLUGINS[pr])
    return table


# ---------------------------------------------------------------------------
# Special rules.

def replay_public_order(pre_order, calls):
    """Numpy reference of PR0's swap rule over the recorded install calls.

    pre_order [L,2,6]; calls: (side [L], slot [L], use [L]) in call order, with
    use already excluding initial installs. Each used row moves `slot` to 0.
    """
    order = np.array(pre_order, copy=True)
    rows = np.arange(order.shape[0])
    for side, slot, use in calls:
        side, slot, use = np.asarray(side), np.asarray(slot), np.asarray(use, bool)
        current = order[rows, side]                                           # [L,6]
        position = np.argmax(current == slot[:, None], axis=1)
        found = (current == slot[:, None]).any(axis=1)
        swapped = current.copy()
        swapped[rows, position] = current[:, 0];swapped[:, 0] = slot
        order[rows, side] = np.where((use & found)[:, None], swapped, current)
    return order


def faint_clear_ok(lane_changes, post, base, fainted, physical_to_public):
    """True when every changed faint-clear leaf differs only on allowed, cleared entries.

    post/base: {name: array for this lane}; fainted [2,6] physical; physical_to_public [2,6].
    """
    for name in lane_changes:
        if name not in FAINT_CLEAR_LEAVES:return False
        a, b = np.asarray(post[name]), np.asarray(base[name])
        if a.shape != b.shape:return False
        differ = a != b
        allowed = np.zeros_like(differ)
        if name.endswith(tuple(f"['{k}']" for k in FAINT_CLEAR_SEATS)):
            allowed[...] = True
        else:
            for side, mon in zip(*np.nonzero(fainted)):
                slot = int(physical_to_public[side, mon])
                if 0 <= slot < 6:allowed[side, slot] = True
        if (differ & ~allowed).any() or (a[differ] != 0).any():return False
    return True


# ---------------------------------------------------------------------------
# Engine plumbing.

def context_and_template(args):
    import jax
    import jax.numpy as j
    from gpu_engine.v2_bench_collector import journal, prepare, actor, construct
    events = journal(args.reference_run);context = prepare(args)
    snapshot, _, _ = actor(args, context['plan'], events)
    _, roster, _ = construct(context, snapshot, events, games=1)
    first = lambda tree: jax.tree.map(lambda x: x[:1], tree)
    template = (first(roster.device.state), j.zeros((1, 2), j.int32), first(roster.device.public))
    return context, template


class Stepper:
    """The jitted step plus per-lane digests, faint data and (optionally) the install log.

    base_names: output names of the base pass; outputs outside it are new, and
    their own pre values are digested too (new leaves must keep their defaults).
    """
    def __init__(self, step_fn, treedef, *, base_names=None, faint_leaves=FAINT_CLEAR_LEAVES, record_installs=False):
        import jax
        import jax.numpy as j
        names = {}
        def traced(leaves):
            state, actions, trackers = jax.tree.unflatten(treedef, leaves)
            lanes = state.hp.shape[0];calls = []
            if record_installs:
                from gpu_engine import engine_device_public_hooks as hooks
                original = hooks.install
                def recorder(c, st, side, use, *, initial=False):
                    out = original(c, st, side, use, initial=initial)
                    s = j.broadcast_to(j.asarray(side, j.int32), (lanes,))
                    slot = out.trackers.active_slot[j.arange(lanes), s]
                    calls.append((s, slot, j.broadcast_to(j.asarray(use, j.bool_), (lanes,)) & (not initial)))
                    return out
                hooks.install = recorder
                try:out = step_fn(state, actions, trackers)
                finally:hooks.install = original
            else:
                out = step_fn(state, actions, trackers)
            named_in = name_inputs(state, actions, trackers);named_out = name_outputs(out)
            names['out'], digests = digest_named(named_out, lanes)
            new = [n for n in sorted(named_out) if base_names is not None and n not in base_names and n in named_in]
            names['pre'] = new
            pre = digest_named({n: named_in[n] for n in new}, lanes)[1] if new else j.zeros((lanes, 0, 2), j.uint32)
            return dict(digests=digests, pre_digests=pre, fainted=(state.hp > 0) & (out.step.state.hp <= 0),
                extra={n: named_out[n] for n in faint_leaves if n in named_out}, calls=calls,
                physical_to_public=out.trackers.physical_to_public, public_order=named_out.get(PUBLIC_ORDER))
        self._jit = jax.jit(traced);self._names = names

    def __call__(self, leaves):
        import jax
        result = jax.device_get(self._jit(leaves))
        result.update(names=self._names['out'], pre_names=self._names['pre'])
        return result


def git_source():
    rev = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'gpu_engine', 'src'], capture_output=True, text=True).stdout
    return rev, bool(dirty.strip())


def load_shard(path):
    with np.load(path) as z:
        keys = sorted((k for k in z.files if k[0] == 'l' and k[1:].isdigit()), key=lambda k: int(k[1:]))
        return [z[k] for k in keys], int(z['tick']), int(z['level'])


def prefetched(paths, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(load_shard, p) for p in paths[:workers]]
        for i, path in enumerate(paths):
            loaded = pending[i].result();pending[i] = None
            if i + workers < len(paths):pending.append(pool.submit(load_shard, paths[i + workers]))
            yield path, loaded


def shard_list(args, capture):
    shards = sorted(Path(args.shards).glob('shard-*.npz'))
    if args.limit:shards = shards[:args.limit]
    if not shards:raise SystemExit(f'no capture shards under {args.shards}')
    if not args.limit and capture.get('shards') is not None and len(shards) != capture['shards']:
        raise SystemExit(f"{len(shards)} shards under {args.shards}, capture.json records {capture['shards']}")
    return shards


def capture_record(args):
    path = Path(args.shards).parent / 'capture.json';blob = path.read_bytes()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Passes.

def base_pass(args, *, step_override=None, prepared=None):
    """step_override/prepared ((context, template)) let tests drive the pass without a V2 runtime."""
    import jax
    from gpu_engine.v2e_capture import leaf_paths
    from gpu_engine.v2e_diff import oracle_digest
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    capture, capture_sha = capture_record(args)
    context, template = prepared or context_and_template(args)
    if capture['paths'] != leaf_paths(template):raise ValueError('capture leaf paths differ from this tree (base must equal the capture)')
    treedef = jax.tree.structure(template)
    tables = context['prepared']
    stepper = Stepper(step_override or _engine_step(tables), treedef)
    shards = shard_list(args, capture);pad = args.pad_lanes or 0
    rev, dirty = git_source()
    names = None;lane_steps = 0;live = 0;begin = time.perf_counter()
    for index, (path, (leaves, tick, level)) in enumerate(prefetched(shards)):
        lanes = int(leaves[0].shape[0])
        named = dict(zip([capture_name(p) for p in capture['paths']], leaves))
        named, real = pad_lanes(named, pad)
        result = stepper([named[capture_name(p)] for p in capture['paths']])
        names = result['names']
        done = np.asarray(named['state.done'])[:real]
        fainted = result['fainted'][:real]
        faint_lanes = np.flatnonzero(fainted.any(axis=(1, 2)) & ~done)
        record = dict(digests=result['digests'][:real], done=done, tick=np.int64(tick), level=np.int64(level),
            fainted=fainted, faint_lanes=faint_lanes)
        for i, n in enumerate(sorted(result['extra'])):record[f'faint_{i:02d}'] = result['extra'][n][faint_lanes]
        np.savez_compressed(out / f'digest-{path.stem}.npz', **record)
        lane_steps += real;live += int((~done).sum())
        if index % 50 == 0:log('BASE_SHARD', index=index, shard=path.name, lanes=lanes, padded=pad or lanes, seconds=time.perf_counter() - begin)
    digest, modules = oracle_digest()
    manifest = dict(schema=SCHEMA, pass_='base', source_commit=rev, source_dirty=dirty, oracle_digest=digest, oracle_modules=len(modules),
        capture_sha256=capture_sha, capture_commit=capture.get('source_commit'), shards=[p.name for p in shards],
        lane_steps=lane_steps, live_lane_steps=live, output_names=names, input_names=[capture_name(p) for p in capture['paths']],
        faint_leaves=sorted(n for n in FAINT_CLEAR_LEAVES if n in names), pad_lanes=pad, digest_seeds=list(DIGEST_SEEDS),
        jax=jax.__version__, backend=jax.default_backend(), seconds=time.perf_counter() - begin)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    log('BASE_DONE', shards=len(shards), lane_steps=lane_steps, live_lane_steps=live, seconds=manifest['seconds'])
    return 0


def classify(changed_names, involved, rule_ok):
    if not changed_names:return 'unchanged'
    if rule_ok:return 'rule'
    return 'involved' if involved else 'uninvolved'


def patched_pass(args, *, step_override=None, prepared=None):
    """step_override: a deliberately perturbed step for self-tests; prepared as in base_pass."""
    import jax
    from gpu_engine import state as state_module
    from gpu_engine.v2e_diff import oracle_digest
    base_dir = Path(args.base);manifest = json.loads((base_dir / 'manifest.json').read_text())
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    capture, capture_sha = capture_record(args)
    if capture_sha != manifest['capture_sha256']:raise ValueError('capture differs from the base pass')
    if jax.__version__ != manifest['jax'] or jax.default_backend() != manifest['backend']:
        raise ValueError(f"base pass ran on jax {manifest['jax']}/{manifest['backend']}, this is {jax.__version__}/{jax.default_backend()}")
    context, template = prepared or context_and_template(args)
    treedef = jax.tree.structure(template)
    template_named = {}
    for name, leaf in zip(_template_names(template), jax.tree.leaves(template)):template_named[name] = np.asarray(leaf)
    capture_names = [capture_name(p) for p in capture['paths']]
    new_inputs = sorted(set(template_named) - set(capture_names))
    table = predicates(args.pr.split(','))
    tables = context['prepared'];metadata = context['pool'].metadata();vocab = metadata['vocabulary']
    public_order_rule = PUBLIC_ORDER in new_inputs
    base_names = manifest['output_names']
    stepper = Stepper(step_override or _engine_step(tables), treedef, base_names=set(base_names), record_installs=public_order_rule)
    shards = [p for p in shard_list(args, capture) if p.name in set(manifest['shards'])]
    if args.limit == 0 and len(shards) != len(manifest['shards']):raise ValueError('shard set differs from the base pass')
    usable = jax.jit(state_module.usable_moves)
    callable_moves = np.asarray(tables.engine_tables.sleep_talk.callable)
    counts = dict(lane_steps=0, done_skipped=0, unchanged=0, changed_and_involved=0, changed_but_uninvolved=0, rule_explained=0)
    by_effect = {k: dict(changed=0, involved_unchanged=0) for k in table}
    uninvolved_leaves = {};new_outputs = None
    listing = (out / 'changed.jsonl').open('x');begin = time.perf_counter();rev, dirty = git_source()
    names_order = _template_names(template)
    try:
        for index, (path, (leaves, tick, level)) in enumerate(prefetched(shards)):
            named, filled = migrate(capture_names, leaves, template_named)
            named, real = pad_lanes(named, manifest['pad_lanes'])
            result = stepper([named[n] for n in names_order])
            names = result['names'];position = {n: i for i, n in enumerate(names)}
            missing = [n for n in base_names if n not in position]
            if missing:raise ValueError(f'output leaves removed by this tree: {missing[:8]}')
            new_outputs = sorted(set(names) - set(base_names))
            digests = result['digests'][:real]
            base = np.load(base_dir / f'digest-{path.stem}.npz')
            diff = np.zeros((real, len(names)), bool)
            shared = [position[n] for n in base_names]
            diff[:, shared] = (digests[:, shared] != base['digests']).any(-1)
            # New leaves must keep their default-filled pre values unless involved.
            for k, n in enumerate(result['pre_names']):
                if n == PUBLIC_ORDER and public_order_rule:continue
                diff[:, position[n]] = (digests[:, position[n]] != result['pre_digests'][:real, k]).any(-1)
            for n in new_outputs:
                if n not in result['pre_names']:diff[:, position[n]] = True          # a new output with no pre value
            if public_order_rule:
                calls = [(np.asarray(a)[:real], np.asarray(b)[:real], np.asarray(c)[:real]) for a, b, c in result['calls']]
                replay = replay_public_order(np.asarray(named[PUBLIC_ORDER])[:real], calls)
                diff[:, position[PUBLIC_ORDER]] = (replay != result['public_order'][:real]).reshape(real, -1).any(-1)
            struggle = ~np.asarray(usable(_state_only(named, template, real))).any(-1)
            pre = Pre({n: np.asarray(named[n])[:real] for n in named if n.startswith('state.') or n == 'actions'},
                vocab, callable_moves, struggle)
            involved = {k: np.asarray(f(pre), bool) for k, f in table.items()}
            done = np.asarray(named['state.done'])[:real]
            faint_base = {}
            if args.faint_clear:
                lanes_list = base['faint_lanes'].tolist()
                for i, n in enumerate(manifest['faint_leaves']):faint_base[n] = dict(zip(lanes_list, base[f'faint_{i:02d}']))
                faint_set = set(lanes_list)
            for lane in range(real):
                counts['lane_steps'] += 1
                if done[lane]:counts['done_skipped'] += 1;continue
                changed = [names[i] for i in np.flatnonzero(diff[lane])]
                effects = sorted(k for k, v in involved.items() if v[lane])
                rule_ok = False
                if changed and args.faint_clear and lane in faint_set and all(n in faint_base for n in changed):
                    post = {n: result['extra'][n][lane] for n in changed}
                    based = {n: faint_base[n][lane] for n in changed}
                    rule_ok = faint_clear_ok(changed, post, based, result['fainted'][lane], result['physical_to_public'][lane])
                kind = classify(changed, bool(effects), rule_ok)
                if kind == 'unchanged':
                    counts['unchanged'] += 1
                    for e in effects:by_effect[e]['involved_unchanged'] += 1
                    continue
                if kind == 'rule':counts['rule_explained'] += 1
                elif kind == 'involved':
                    counts['changed_and_involved'] += 1
                    for e in effects:by_effect[e]['changed'] += 1
                else:
                    counts['changed_but_uninvolved'] += 1
                    for n in changed:uninvolved_leaves[n] = uninvolved_leaves.get(n, 0) + 1
                listing.write(json.dumps(dict(shard=path.name, lane=lane, tick=tick, kind=kind, involved=effects, leaves=changed)) + '\n')
            if index % 50 == 0:
                log('PATCHED_SHARD', index=index, shard=path.name, seconds=time.perf_counter() - begin, **counts)
    finally:
        listing.close()
    digest, modules = oracle_digest()
    report = dict(schema=SCHEMA, pass_='patched', plug_ins=args.pr, source_commit=rev, source_dirty=dirty, oracle_digest=digest,
        base_source_commit=manifest['source_commit'], base_oracle_digest=manifest['oracle_digest'], capture_sha256=capture_sha,
        shards=len(shards), migrated_new_inputs=new_inputs, new_outputs=new_outputs,
        public_order_rule=public_order_rule, faint_clear_rule=bool(args.faint_clear), counts=counts, by_effect=by_effect,
        uninvolved_leaves=dict(sorted(uninvolved_leaves.items(), key=lambda x: -x[1])),
        passed=counts['changed_but_uninvolved'] == 0, seconds=time.perf_counter() - begin)
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    log('PATCHED_DONE', passed=report['passed'], **counts)
    return 0 if report['passed'] else 1


def _engine_step(tables):
    from gpu_engine import v2e_modules
    return v2e_modules.step(tables.engine_tables, tables.observation_tables)


def _template_names(template):
    state, actions, trackers = template
    return list(name_inputs(state, actions, trackers))


def _state_only(named, template, real):
    import jax
    state = template[0]
    names = list(_named('state', state))
    return jax.tree.unflatten(jax.tree.structure(state), [np.asarray(named[n])[:real] for n in names])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('base', 'patched'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--reference-run', required=True)
        p.add_argument('--shards', required=True, help='directory of capture shards (capture.json beside it)')
        p.add_argument('--out', required=True, help='new output directory')
        p.add_argument('--arm', default='B', choices=('A', 'B'))
        p.add_argument('--limit', type=int, default=0, help='first N shards (0: all)')
        if name == 'base':
            p.add_argument('--pad-lanes', type=int, default=2048, help='pad every shard to this many lanes (one compile); 0 keeps native counts')
        else:
            p.add_argument('--base', required=True, help='output directory of the base pass')
            p.add_argument('--pr', default='none', help=f'comma list of involvement plug-ins: {",".join(PLUGINS)}')
            p.add_argument('--faint-clear', action='store_true', help="apply PR0's faint-time public clear rule")
    args = parser.parse_args(argv)
    return base_pass(args) if args.command == 'base' else patched_pass(args)


if __name__ == '__main__':
    sys.exit(main())
