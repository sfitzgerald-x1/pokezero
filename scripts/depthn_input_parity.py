#!/usr/bin/env python3
"""Why does the live DepthN server's root policy prior disagree with the live actor's greedy policy? Input parity on four paths.

The actor plays argmax of champion 9375 (torch, CPU fp32) on B2's own encoding of the literal root (pokezero.observation.v4, 23
tokens). The server's root prior is the same champion (gpu_engine.champion, bf16, weights.npz exported from the same checkpoint) on
world 0's own V2 view (29 tokens; the champion reads the 23-token prefix). Four paths separate the inputs from the model:
  path 1: B2 encoding        -> torch actor forward   (the actor reference; must reproduce the live greedy)
  path 2: the recorded prior -> server forward        (re-run on the inputs build gave it, must reproduce root_logits bit for bit:
                                                        world 0's encoding, or B2-spliced when the root's root_view_source is 'b2')
  path 3: B2 encoding        -> server forward        (root_inputs with the first 23 tokens of each token tensor replaced by B2's)
  path 4: world-0 prefix     -> torch actor forward

Subcommands (`depthn_input_parity.py capture|torch|report ...`):

capture (GPU pod: the exact server replay). Selection from the RR re-pricer rows (--rr-rows rows-*.jsonl): the disagree set is
every row with replay COMPLETE whose argmax of root.root_logits over root.root_legal (lowest index on ties) differs from
live.greedy_action_index; the control set an equal-size random.Random(--select-seed) sample of the agreeing rows (keys in
(ordinal, decision_index) order). The battles holding selected decisions are sharded (--shard I --shards N over their ordinals,
then --battles / --limit-battles for smoke runs) and replayed whole, in decision order, by scripts/depthn_rr_reprice.main itself
(its parser and server flags, tree guard, identity check, warmup and session replay; everything after the capture flags passes
through to it, with --cells set to this shard's battles and --out to <out>/replay). Two module globals of the re-pricer are swapped
for the duration of its main (reprice_hooks; check_reprice_hooks verifies from main's bytecode that it looks them up at call time):
- rr.Capture -> ParityCapture, whose installed() (rr's own context manager: decide and prepare_roots wrapped) additionally wraps
  gpu_engine.constructor_worlds.build for the same process_many call. DepthNLiveSearch._search calls `cw.build(...)` through the
  module attribute (check_build_call_site, as rr.check_call_sites checks its two). The wrapper calls the original with the same
  objects, returns its result unchanged and is restored when the call returns or raises. Then, for every PRICED root whose
  request_id (rr-<ordinal>-<decision_index>-<pid>) is selected, Recorder.after_build records (never raising into the search):
  (a) root_inputs, the Request's world-0 V2 encoding (five keys, exact arrays); (b) B2's detached.view['_b2_root_inputs'] of the
  prepared root of that index; (c) the root dict's b2_view, root_logits, root_value, root_legal, root_view_world; (d) path 3:
  constructor_worlds._root_forward(pricer.forward)(params[learner_actor], X), the jitted callable build used, X = splice(root_inputs,
  B2); valid, logits and value recorded; (e) path 2: the same callable on the inputs build's root prior read (root_inputs, or X when
  the root dict's root_view_source is 'b2': constructor_worlds.splice is build's own), compared bit for bit with the recorded
  root_logits and root_value (path2.logits_equal / value_equal, path2.inputs; a miss is reported, never raised); with a 'b2' source,
  `world0` also records the callable on the unmodified root_inputs (logits, value, top). Cross-world: OUR (owner)
  view of every constructed world through depth2._make_views(pricer.executor)(state, public, owner)'s third output, on the
  host-fetched state as build does; per root whether every world's 23-token prefix (and the full 29 tokens) is bit-identical to the
  root view world's, which keys / tokens / columns differ, the max |diff|; arrays kept for worlds 0, 1 and any world that differs.
  Once per run, the device encoder's installed column manifest (engine_device_trackers.readout of world 0's public carry ->
  engine_device_observation.column_manifest) for the report's labels.
- rr.decision_row -> a wrapper that returns the original's record unchanged and hands (cell, capture row, request, record) to
  Recorder.after_row: the record's live join and parity flags complete the decision's index entry.
Output (--out, a new directory): selection.json (the whole selection, identical in every shard), header.json, index.jsonl (one
entry per selected decision this shard replayed: scalars), arrays/<ordinal>.npz (one per battle: every array of (a), (b), (d), (e)
and the kept cross-world views, keys d<decision_index>__<group>__<name>), capture-summary.json (also on interruption), replay/ (the
re-pricer's own rows/header/summary).

torch (CPU; paths 1 and 4). The checkpoint loads as the actor loads it (foulplay_bridge.load_transformer_checkpoint(path,
map_location='cpu'); its bytes must be --checkpoint-sha256, which must be the pinned champion's); the live worker's precision
(depthn_live_panel.serve: one torch thread each, matmul 'highest', MHA fastpath off, TF32 off, eval) is set, then
RawPolicy.from_transformer(ActorContract.champion(), TransformerSoftmaxPolicy(...)) checks the actor's own preconditions. Per
decision the observation is built as LiveProtocolActor._observation builds it (float32 numerics, tolist), tensorized by
observation_window_to_torch (window 1: [1,1,23,...], history_mask [[True]], legal mask = root_legal), probabilities by
_masked_action_probabilities and greedy by _greedy_action_index (both imported); RawPolicy.select_action on the same observation must
agree. Path 1 is B2's encoding, path 4 the world-0 encoding's 23-token prefix.

report (CPU): summary.json and summary.md over the capture directories and torch JSONL files: the gate (path 1 reproduces the live
greedy on >= 99.5% of decisions, path 2 reproduces root_logits bit for bit, else HARNESS_NOT_EXACT with the misses), per path and set
the argmax agreement with the live greedy and the server prior and max |dlogit| over legal actions against path 1 and the recorded
prior (p50/p95/max), the B2 vs world-0 encoding diff (per key, token, column and (section, kind, column) cell, by set, ranked by
the disagree/control share ratio, tagged by side and information status), the cross-world diff, and the attribution.
"""
import argparse
import collections
from contextlib import contextmanager
import dis
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))
import numpy as np   # noqa: E402
from scripts import depthn_rr_reprice as rr   # noqa: E402
from gpu_engine import constructor_worlds as cw, depthn_live_search as live   # noqa: E402

INDEX_SCHEMA = 'pokezero.depthn-input-parity.index.v1'
TORCH_SCHEMA = 'pokezero.depthn-input-parity.torch.v1'
SUMMARY_SCHEMA = 'pokezero.depthn-input-parity.summary.v1'
CHAMPION_CHECKPOINT_SHA256 = '0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be'
TOKENS, V2_TOKENS, NUMERIC, CATEGORICAL = 23, 29, 132, 41
INPUT_KEYS = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'history_mask')
TOKEN_KEYS = {'numeric_features': 2, 'categorical_ids': 2, 'token_type_ids': 1, 'attention_mask': 1}   # trailing dims from the token axis
B2_KEYS = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask')
REQUEST_ID = re.compile(r'rr-(\d+)-(\d+)-\d+')
GATE_PATH1 = .995
FATAL = (MemoryError, KeyboardInterrupt)
SETS = ('disagree', 'control')
PATHS = {1: 'B2 encoding -> torch actor (the actor reference)', 2: 'the recorded root prior (world-0 or B2-spliced inputs, root_view_source)',
    3: 'B2 encoding -> server forward', 4: 'world-0 23-token prefix -> torch actor'}


# Small exact helpers --------------------------------------------------------------------------------------------------------------
def argmax_legal(values, legal):
    """The legal action with the largest value, the lowest index on ties (None without a legal action)."""
    acts = [a for a in range(len(legal)) if legal[a]]
    return max(acts, key=lambda a: (values[a], -a)) if acts else None


def bits(a):
    """An array as integers of its width: floats compared bit for bit (signed zeros and NaN payloads told apart)."""
    a = np.ascontiguousarray(np.asarray(a))
    return a.view({2: np.uint16, 4: np.uint32, 8: np.uint64}[a.itemsize]) if a.dtype.kind == 'f' else a


def bit_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(bits(a), bits(b)))


def floats(a):
    return [float(x) for x in np.asarray(a).reshape(-1)]


def request_key(request_id):
    """(ordinal, decision_index) of a re-pricer request_id (rr-<ordinal>-<decision_index>-<pid>), else None."""
    m = REQUEST_ID.fullmatch(request_id) if isinstance(request_id, str) else None
    return None if m is None else (int(m.group(1)), int(m.group(2)))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value, indent=1, sort_keys=True, default=repr) + '\n');stream.flush();os.fsync(stream.fileno())


def encode(value):
    try:return json.dumps(value, sort_keys=True, allow_nan=False, default=repr)
    except ValueError:return json.dumps(dict(value, nonfinite=True), sort_keys=True, default=repr)


def read_jsonl(path):
    with open(path, 'rb') as stream:
        for raw in stream:
            if raw.strip():yield json.loads(raw)


# Selection --------------------------------------------------------------------------------------------------------------------------
def rr_rows(paths):
    """What selection reads of the re-pricer's rows (rows-*.jsonl), in (ordinal, decision_index) order; ValueError on a duplicate."""
    out = {}
    for path in sorted(map(str, paths)):
        for r in read_jsonl(path):
            key = (int(r['ordinal']), int(r['decision_index']))
            if key in out:raise ValueError(f'decision {key} appears twice in the re-pricer rows ({out[key]["source"]}, {path})')
            root, found, replay = r.get('root') or {}, r.get('live') or {}, r.get('replay') or {}
            out[key] = dict(ordinal=key[0], decision_index=key[1], battle_id=r.get('battle_id'), replay_status=replay.get('status'),
                root_logits=root.get('root_logits'), root_legal=root.get('root_legal'), live_greedy=found.get('greedy_action_index'),
                live_source=found.get('source'), source=path)
    return [out[k] for k in sorted(out)]


def classify(rows):
    """dict(disagree, agree, unclassified): row keys whose server prior argmax over root_legal (lowest index on ties) differs from /
    equals the live greedy, over the rows with replay COMPLETE; unclassified (key, reason) otherwise."""
    out = dict(disagree=[], agree=[], unclassified=[])
    for r in rows:
        key = (r['ordinal'], r['decision_index']);logits, legal, greedy = r['root_logits'], r['root_legal'], r['live_greedy']
        if r['replay_status'] != 'COMPLETE':out['unclassified'].append((key, f'replay {r["replay_status"]}'));continue
        if logits is None or legal is None or type(greedy) is not int:out['unclassified'].append((key, 'no root prior or live greedy'));continue
        out['disagree' if argmax_legal(logits, legal) != greedy else 'agree'].append(key)
    return out


def select(rows, seed):
    """{(ordinal, decision_index): dict(set, live_greedy, root_logits, root_legal, server_top)}: every disagreeing row and an
    equal-size random.Random(seed) sample of the agreeing ones (module docstring); plus the classification counts."""
    c = classify(rows);by = {(r['ordinal'], r['decision_index']): r for r in rows}
    agree = sorted(c['agree']);n = min(len(c['disagree']), len(agree))
    control = sorted(random.Random(seed).sample(agree, n))
    selected = {}
    for name, keys in (('disagree', sorted(c['disagree'])), ('control', control)):
        for k in keys:
            r = by[k];selected[k] = dict(set=name, live_greedy=r['live_greedy'], root_logits=r['root_logits'], root_legal=r['root_legal'],
                server_top=argmax_legal(r['root_logits'], r['root_legal']), battle_id=r['battle_id'])
    counts = dict(rows=len(rows), complete=sum(r['replay_status'] == 'COMPLETE' for r in rows), disagree=len(c['disagree']), agree=len(agree),
        control=len(control), unclassified=len(c['unclassified']), unclassified_reasons=dict(collections.Counter(x[1] for x in c['unclassified'])))
    return selected, counts


def shard_battles(selected, shard, shards, *, battles=None, limit=None):
    """The battle ordinals of the selected decisions, every `shards`-th from the `shard`-th; then only `battles` (each must be a
    selected battle of this shard) and the first `limit`."""
    mine = sorted({k[0] for k in selected})[shard::shards]
    if battles:
        odd = sorted(set(battles) - set(mine))
        if odd:raise ValueError(f'battles {odd} hold no selected decision in shard {shard}/{shards}')
        mine = [b for b in mine if b in set(battles)]
    return mine[:limit] if limit is not None else mine


# The splice and the diffs -----------------------------------------------------------------------------------------------------------
_prefix_index, splice = cw._prefix_index, cw.splice   # build's own splice (constructor_worlds): the root prior's B2 inputs


def token_view(value, key):
    """[29, F] / [29] of a V2 token tensor with one leading row ([1, 1, 29, ...])."""
    a = np.asarray(value);trailing = a.shape[a.ndim - TOKEN_KEYS[key]:]
    flat = a.reshape((-1,) + trailing)
    if flat.shape[0] != 1:raise ValueError(f'{key}: one root row required, got {a.shape}')
    return flat[0]


def own_view_diff(views, reference, tokens=TOKENS):
    """Every world's own V2 inputs against the reference world's, bit for bit (module docstring): prefix_equal (the 23 tokens the
    champion reads and history_mask), full_equal (all 29 tokens), value_equal (== as the pricer's turn-1 check compares), the
    differing worlds and keys, the differing prefix cells as [key, token, column] (column -1 for a per-token key, token -1 for
    history_mask), keys differing beyond the prefix, and the largest |numeric difference|."""
    ref = views[reference];triples, keys, beyond, worlds = set(), set(), set(), []
    out = dict(reference=reference, worlds=sorted(views), prefix_equal=True, full_equal=True, value_equal=True, max_abs_diff=0.)
    for w in sorted(views):
        if w == reference:continue
        differs = False
        for key in INPUT_KEYS:
            a, b = np.asarray(views[w][key]), np.asarray(ref[key])
            if a.shape != b.shape or a.dtype != b.dtype:
                keys.add(key);differs = True;out.update(prefix_equal=False, full_equal=False, value_equal=False);triples.add((key, -1, -1));continue
            mask = bits(a) != bits(b)
            if not mask.any():continue
            differs = True;keys.add(key);out['full_equal'] = False
            if not bool(np.all(a == b)):out['value_equal'] = False
            if a.dtype.kind == 'f':out['max_abs_diff'] = max(out['max_abs_diff'], float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
            if key not in TOKEN_KEYS:triples.add((key, -1, -1));continue
            m = token_view(mask, key)
            if m[tokens:].any():beyond.add(key)
            prefix = m[:tokens]
            if prefix.ndim == 2:triples.update((key, int(t), int(c)) for t, c in zip(*np.nonzero(prefix)))
            else:triples.update((key, int(t), -1) for t in np.nonzero(prefix)[0])
        if differs:worlds.append(w)
    out.update(prefix_equal=not triples, differing_worlds=worlds, differing_keys=sorted(keys), beyond_prefix_keys=sorted(beyond),
        cells=sorted([k, t, c] for k, t, c in triples))
    return out


def encoding_diff(b2, root_inputs, tokens=TOKENS):
    """B2's 23-token encoding against the world-0 V2 root inputs' prefix: per key equal; per element differs (numeric by value, NaN
    equal to NaN), [23, 132] / [23, 41] / [23] / [23]; the largest |numeric difference|; bit_only (values equal, bits not)."""
    out = dict(keys={}, max_numeric_diff=0., bit_only=False)
    for key in B2_KEYS:
        a = np.asarray(b2[key]);r = token_view(root_inputs[key], key)[:tokens]
        if a.shape != r.shape:raise ValueError(f'{key}: B2 {a.shape} against the root prefix {r.shape}')
        if key == 'numeric_features':
            x, y = a.astype(np.float64), r.astype(np.float64)
            differs = ~((x == y) | (np.isnan(x) & np.isnan(y)))
            gap = np.abs(x - y);gap = gap[~np.isnan(gap)]
            out['max_numeric_diff'] = float(gap.max()) if gap.size else 0.
            out['bit_only'] = not differs.any() and not bit_equal(a.astype(np.float32), r.astype(np.float32))
        else:differs = a.astype(np.int64) != r.astype(np.int64)
        out['keys'][key] = not bool(differs.any());out[key] = differs
    return out


# Capture -----------------------------------------------------------------------------------------------------------------------------
def check_build_call_site(function=None):
    """True when `function` (DepthNLiveSearch._search) reads `build` as an attribute of the global `cw` at call time, so the
    wrapper takes effect (rr.check_call_sites' rule); RuntimeError otherwise."""
    function = live.DepthNLiveSearch._search if function is None else function
    found, previous = False, None
    for instruction in dis.get_instructions(function):
        if (instruction.opname in ('LOAD_ATTR', 'LOAD_METHOD') and instruction.argval == 'build' and previous is not None
                and previous.opname == 'LOAD_GLOBAL' and previous.argval == 'cw'):found = True
        previous = instruction
    if not found or function.__globals__ is not vars(live) or live.cw is not cw:
        raise RuntimeError('the search does not look cw.build up through its module at call time')
    return True


def check_reprice_hooks(function=None):
    """True when the re-pricer's main reads `Capture` and `decision_row` as module globals at call time; RuntimeError otherwise."""
    function = rr.main if function is None else function
    loaded = {i.argval for i in dis.get_instructions(function) if i.opname == 'LOAD_GLOBAL'}
    if {'Capture', 'decision_row'} - loaded or function.__globals__ is not vars(rr):
        raise RuntimeError(f'the re-pricer main does not look Capture / decision_row up at call time: {sorted({"Capture", "decision_row"} - loaded)}')
    return True


BUILD = inspect.signature(cw.build)


def root_forward_of(pricer):
    """The jitted root forward build itself uses (constructor_worlds._root_forward's cache, keyed by the evaluator object)."""
    return cw._root_forward(pricer.forward)


def views_of(pricer):
    """build's root-view function: device_depth2_pricer._make_views on the pricer's executor."""
    from gpu_engine import device_depth2_pricer as depth2
    return depth2._make_views(pricer.executor)


def installed_manifest(public, owner):
    """The device encoder's column manifest under the hooks world 0's public carry installs (module docstring)."""
    import jax.numpy as jnp
    from gpu_engine import engine_device_observation, engine_device_trackers
    return engine_device_observation.column_manifest(public=engine_device_trackers.readout(public, jnp.asarray([owner], jnp.int32)))


def capture_class(recorder):
    """rr.Capture whose installed() also wraps constructor_worlds.build (module docstring)."""
    class ParityCapture(rr.Capture):
        @contextmanager
        def installed(self):
            with super().installed():
                build = cw.build

                def build_wrapper(*arguments, **keywords):
                    out = build(*arguments, **keywords)
                    recorder.after_build(arguments, keywords, out)
                    return out
                cw.build = build_wrapper
                try:yield self
                finally:cw.build = build
    return ParityCapture


@contextmanager
def reprice_hooks(recorder):
    """rr.Capture -> ParityCapture and rr.decision_row -> its pass-through wrapper while the re-pricer's main runs."""
    check_reprice_hooks();check_build_call_site()
    capture, decision_row = rr.Capture, rr.decision_row

    def row_wrapper(*arguments, **keywords):
        record = decision_row(*arguments, **keywords)
        recorder.after_row(arguments[0], arguments[1], arguments[2], record)
        return record
    rr.Capture, rr.decision_row = capture_class(recorder), row_wrapper
    try:yield
    finally:rr.Capture, rr.decision_row = capture, decision_row


class Recorder:
    """The selected decisions' records (module docstring). after_build and after_row never raise into the replay (FATAL aside):
    their failures are the decision's `errors`."""
    def __init__(self, selected, out, *, log=None):
        self.selected, self.out, self.log = selected, Path(out), log
        (self.out / 'arrays').mkdir()
        self.index = (self.out / 'index.jsonl').open('x')
        self.pending, self.battle, self.arrays, self.entries = {}, None, {}, []
        self.errors, self.manifest, self.manifest_error, self.counts, self.done = [], None, None, collections.Counter(), set()

    # build ----------------------------------------------------------------------------------------------------------------------
    def after_build(self, arguments, keywords, result):
        try:
            bound = BUILD.bind(*arguments, **keywords);bound.apply_defaults();a = bound.arguments
            prepared, pricer, params, learner = a['prepared'], a['pricer'], a['params'], a['learner_actor']
            roots = list(result.roots)
        except FATAL:raise
        except Exception as error:self.errors.append(f'build arguments: {type(error).__name__}: {error}');return
        for root in roots:
            key = request_key(root.get('request_id'))
            if key is None or key not in self.selected:continue
            record = dict(key=key, build_status=root.get('status'), build_reason=root.get('reason'), errors=[], arrays={})
            self.pending[root['request_id']] = record
            if root.get('status') != 'PRICED':continue
            clock = time.perf_counter()
            try:self._root(record, root, prepared, pricer, params, learner, result)
            except FATAL:raise
            except Exception as error:record['errors'].append(f'{type(error).__name__}: {error}')
            record['capture_seconds'] = time.perf_counter() - clock

    def _root(self, record, root, prepared, pricer, params, learner, result):
        import jax
        item = prepared[root['index']]
        if item.request_id != root['request_id']:raise ValueError('the prepared root of this index answers another request')
        detached = item.detached;owner = int(detached.address.owner);rows = tuple(root['rows'])
        root_inputs = result.requests[rows[0]].root_inputs
        if any(result.requests[r].root_inputs is not root_inputs for r in rows):raise ValueError('the root\'s requests carry different root inputs')
        arrays = record['arrays']
        for k, v in root_inputs.items():arrays['root__' + k] = np.array(v, copy=True)
        view = getattr(detached, 'view', None);b2 = view.get('_b2_root_inputs') if hasattr(view, 'get') else None
        logits = np.asarray(root['root_logits'], np.float32);value = np.float32(root['root_value'])
        arrays.update(root__logits=logits, root__value=np.asarray(value), root__legal=np.asarray(root['root_legal'], bool))
        source = root.get('root_view_source')
        record.update(root_logits=list(root['root_logits']), root_legal=[bool(x) for x in root['root_legal']], root_value=root['root_value'],
            root_view_world=root.get('root_view_world'), b2_view=root.get('b2_view'), owner=owner, root_view_source=source,
            recorded_float32_exact=floats(logits) == list(root['root_logits']) and float(value) == root['root_value'])
        forward, p = root_forward_of(pricer), params[learner]
        X, lossy = splice(root_inputs, b2) if b2 is not None else (None, None)
        if source == 'b2' and X is None:raise ValueError('root_view_source b2 without _b2_root_inputs')
        # Path 2: the recorded prior again, through the same jitted callable on the inputs build gave it.
        out = jax.device_get(forward(p, X if source == 'b2' else root_inputs))
        l2, v2 = np.asarray(out.policy_logits, np.float32).reshape(-1), np.asarray(out.value, np.float32).reshape(-1)[0]
        arrays.update(path2__logits=l2, path2__value=np.asarray(v2), path2__valid=np.asarray(out.valid).reshape(-1))
        record['path2'] = dict(logits_equal=bit_equal(l2, logits), value_equal=bit_equal(v2, value), valid=bool(np.asarray(out.valid).reshape(-1)[0]),
            inputs='b2_spliced' if source == 'b2' else 'world0')
        record['world0'] = None
        if source == 'b2':   # world 0's own encoding through the same callable (what the prior read before the B2 root view)
            out = jax.device_get(forward(p, root_inputs))
            l0, v0 = np.asarray(out.policy_logits, np.float32).reshape(-1), np.asarray(out.value, np.float32).reshape(-1)[0]
            arrays.update(world0__logits=l0, world0__value=np.asarray(v0))
            record['world0'] = dict(logits=floats(l0), value=float(v0), valid=bool(np.asarray(out.valid).reshape(-1)[0]), top=argmax_legal(floats(l0), root['root_legal']))
        if b2 is None:record['errors'].append('the root carries no _b2_root_inputs');record['path3'] = None
        else:
            for k in b2:arrays['b2__' + k] = np.array(b2[k], copy=True)
            # Path 3: the server forward on the actor's encoding (the champion reads only the 23-token prefix).
            out = jax.device_get(forward(p, X))
            l3, v3 = np.asarray(out.policy_logits, np.float32).reshape(-1), np.asarray(out.value, np.float32).reshape(-1)[0]
            valid = bool(np.asarray(out.valid).reshape(-1)[0])
            arrays.update(path3__logits=l3, path3__value=np.asarray(v3), path3__valid=np.asarray(out.valid).reshape(-1))
            record['path3'] = dict(logits=floats(l3), value=float(v3), valid=valid, lossy_keys=lossy, top=argmax_legal(floats(l3), root['root_legal']))
            if not valid:record['errors'].append('path 3 forward not valid')
        # Cross-world: our own view in every constructed world, as build computes world 0's.
        views, own = views_of(pricer), {}
        for w in item.made.worlds:
            if w.state is None:continue
            state, public = jax.device_get((w.state, w.public))
            _, _, inputs, _ = views(state, public, owner)
            own[w.index] = {k: np.asarray(v) for k, v in inputs.items()}
            if self.manifest is None and self.manifest_error is None:
                try:self.manifest = installed_manifest(public, owner)
                except FATAL:raise
                except Exception as error:self.manifest_error = f'{type(error).__name__}: {error}'
        reference = root.get('root_view_world', 0)
        cross = own_view_diff(own, reference)
        cross['reference_reproduces_root_inputs'] = all(bit_equal(own[reference][k], root_inputs[k]) for k in root_inputs)
        record['crossworld'] = cross
        others = [w for w in sorted(own) if w != reference]
        for w in sorted({reference, *others[:1], *cross['differing_worlds']}):
            for k, v in own[w].items():arrays[f'xw__{w}__{k}'] = v

    # decision rows --------------------------------------------------------------------------------------------------------------------
    def after_row(self, cell, row, request, record):
        try:self._row(cell, row, request, record)
        except FATAL:raise
        except Exception as error:self.errors.append(f'row {cell.get("ordinal")}/{row.get("decision_index")}: {type(error).__name__}: {error}')

    def _row(self, cell, row, request, record):
        ordinal = cell['ordinal']
        if ordinal != self.battle:self.flush();self.battle = ordinal
        key = (ordinal, row['decision_index'])
        if key not in self.selected:return
        chosen = self.selected[key];rid = request['request_id'];captured = self.pending.pop(rid, None) or dict(errors=[], arrays={})
        found = record.get('live') or {};replay = record.get('replay') or {};root = record.get('root') or {}
        entry = dict(schema=INDEX_SCHEMA, ordinal=ordinal, decision_index=key[1], set=chosen['set'], battle_id=row.get('battle_id'), request_id=rid,
            selection=dict(live_greedy=chosen['live_greedy'], server_top=chosen['server_top']),
            live=dict(greedy_action_index=found.get('greedy_action_index'), source=found.get('source'), request_kind=found.get('request_kind'),
                turn=found.get('turn'), search_action_index=found.get('search_action_index')),
            parity=record.get('parity'), replay_status=replay.get('status'), replay_cause=replay.get('cause'), replay_action_index=replay.get('action_index'),
            build_status=captured.get('build_status'), build_reason=captured.get('build_reason'),
            captured=bool(captured['arrays']) and 'path2' in captured, errors=list(captured['errors']),
            **{k: captured.get(k) for k in ('root_logits', 'root_legal', 'root_value', 'root_view_world', 'root_view_source', 'b2_view', 'owner',
                'recorded_float32_exact', 'path2', 'path3', 'world0', 'crossworld', 'capture_seconds')})
        entry['live_greedy_matches_selection'] = entry['live']['greedy_action_index'] == chosen['live_greedy']
        entry['root_logits_match_selection'] = entry['root_logits'] is not None and entry['root_logits'] == chosen['root_logits']
        entry['decide_root_logits_match'] = entry['root_logits'] is not None and list(root.get('root_logits') or ()) == entry['root_logits']
        prefix = f'd{key[1]}__'
        for name, value in captured['arrays'].items():self.arrays[prefix + name] = value
        entry['arrays'] = f'arrays/{ordinal}.npz' if captured['arrays'] else None
        self.entries.append(entry);self.done.add(key)
        self.counts['recorded'] += 1;self.counts['captured'] += entry['captured']
        self.counts['path2_exact'] += bool(entry['path2'] and entry['path2']['logits_equal'] and entry['path2']['value_equal'])
        p3 = entry['path3'] or {}
        print(json.dumps(dict(parity_capture=dict(ordinal=ordinal, decision_index=key[1], set=chosen['set'], replay=entry['replay_status'],
            captured=entry['captured'], path2=entry['path2'], path3_top=p3.get('top'), live_greedy=chosen['live_greedy'], server_top=chosen['server_top'],
            crossworld_prefix_equal=(entry['crossworld'] or {}).get('prefix_equal'), errors=entry['errors']))), file=self.log or sys.stderr, flush=True)

    def flush(self):
        """The current battle's npz (written first) and its index entries."""
        if self.arrays:
            with (self.out / 'arrays' / f'{self.battle}.npz').open('xb') as stream:np.savez_compressed(stream, **self.arrays);stream.flush();os.fsync(stream.fileno())
        for entry in self.entries:self.index.write(encode(entry) + '\n')
        self.index.flush();os.fsync(self.index.fileno());self.arrays, self.entries = {}, []

    def close(self):
        self.flush();self.index.close()


def capture_parser():
    p = argparse.ArgumentParser(prog='depthn_input_parity.py capture', allow_abbrev=False, description='The exact server replay of the battles holding '
        'selected decisions (module docstring). Every other argument goes to scripts/depthn_rr_reprice.py (its server flags, --run-directory, --arm, '
        '--warmup-requests, --compilation-cache, --tree-commit, --max-decisions).')
    p.add_argument('--rr-rows', nargs='+', required=True, help='the RR re-pricer rows (rows-*.jsonl of every shard)')
    p.add_argument('--select-seed', type=int, required=True, help='the control sample\'s seed')
    p.add_argument('--out', required=True, help='a new directory')
    p.add_argument('--shard', type=int, default=0);p.add_argument('--shards', type=int, default=1)
    p.add_argument('--battles', type=int, nargs='+', metavar='ORD', help='only these battles (of this shard)')
    p.add_argument('--limit-battles', type=int, help='only the first N battles of this shard (smoke)')
    return p


def capture_main(argv):
    p = capture_parser();args, rest = p.parse_known_args(argv)
    for flag in ('--cells', '--sample', '--sample-seed'):
        if any(x == flag or x.startswith(flag + '=') for x in rest):p.error(f'{flag} is chosen here (from --rr-rows), not passed to the re-pricer')
    if args.shards < 1 or not 0 <= args.shard < args.shards:p.error('--shard I --shards N with 0 <= I < N required')
    if args.limit_battles is not None and args.limit_battles < 1:p.error('--limit-battles must be >= 1')
    try:check_reprice_hooks();check_build_call_site()
    except RuntimeError as error:raise SystemExit(f'depthn_input_parity: {error}')
    rows = rr_rows(args.rr_rows);selected, counts = select(rows, args.select_seed)
    battles = shard_battles(selected, args.shard, args.shards, battles=args.battles, limit=args.limit_battles)
    if not battles:raise SystemExit('depthn_input_parity: no battle in this shard')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    listing = [dict(ordinal=k[0], decision_index=k[1], **v) for k, v in sorted(selected.items())]
    write_json(out / 'selection.json', dict(counts=counts, select_seed=args.select_seed, rr_rows={str(x): file_sha256(x) for x in sorted(args.rr_rows)},
        selection_sha256=hashlib.sha256(encode(listing).encode()).hexdigest(), decisions=listing))
    mine = {k: v for k, v in selected.items() if k[0] in set(battles)}
    reprice = rest + ['--cells', *map(str, battles), '--shard', '0', '--shards', '1', '--out', str(out / 'replay')]
    head = None
    try:head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):pass
    write_json(out / 'header.json', dict(argv=list(argv), reprice_argv=reprice, head=head, shard=args.shard, shards=args.shards, battles=battles,
        selected_in_shard=len(mine), selected_sets=dict(collections.Counter(v['set'] for v in mine.values())), counts=counts, pid=os.getpid(),
        python=sys.version))
    recorder = Recorder(selected, out);started = time.perf_counter();code = None;finished = False
    try:
        with reprice_hooks(recorder):code = rr.main(reprice)
        finished = True
    finally:
        recorder.close()
        missing = sorted(k for k in mine if k not in recorder.done)
        write_json(out / 'capture-summary.json', dict(shard=args.shard, shards=args.shards, battles=battles, selected_in_shard=len(mine),
            recorded=recorder.counts['recorded'], captured=recorder.counts['captured'], path2_exact=recorder.counts['path2_exact'],
            missing=[list(k) for k in missing], orphaned=sorted(recorder.pending), errors=recorder.errors, reprice_exit=code, interrupted=not finished,
            installed_manifest=recorder.manifest, installed_manifest_error=recorder.manifest_error, seconds=time.perf_counter() - started))
    print(json.dumps(dict(done=str(out), recorded=recorder.counts['recorded'], captured=recorder.counts['captured'], path2_exact=recorder.counts['path2_exact'],
        missing=len(missing), errors=len(recorder.errors))), flush=True)
    return code


# Reading a capture ---------------------------------------------------------------------------------------------------------------------
def capture_entries(directories):
    """Every index entry of the capture directories, in (ordinal, decision_index) order, each with its directory; ValueError on a
    duplicate or on shards selected differently."""
    out, selections = {}, set()
    for d in map(Path, directories):
        selections.add(json.loads((d / 'selection.json').read_text())['selection_sha256'])
        for entry in read_jsonl(d / 'index.jsonl'):
            key = (entry['ordinal'], entry['decision_index'])
            if key in out:raise ValueError(f'decision {key} is in two captures')
            out[key] = dict(entry, directory=str(d))
    if len(selections) > 1:raise ValueError('the capture directories were selected from different rows or seeds')
    return [out[k] for k in sorted(out)]


def decision_arrays(entries):
    """(entry, {name: array}) per captured entry, loading each battle's npz once."""
    loaded, path = None, None
    for entry in entries:
        if not entry.get('arrays'):yield entry, None;continue
        where = Path(entry['directory']) / entry['arrays']
        if where != path:
            if loaded is not None:loaded.close()
            loaded, path = np.load(where), where
        prefix = f'd{entry["decision_index"]}__'
        yield entry, {name[len(prefix):]: loaded[name] for name in loaded.files if name.startswith(prefix)}
    if loaded is not None:loaded.close()


def b2_arrays(arrays):
    return {k: arrays['b2__' + k] for k in B2_KEYS}


def prefix_arrays(arrays):
    """The world-0 encoding's 23-token prefix in B2's shapes ([23, 132], [23, 41], [23], [23])."""
    return {k: token_view(arrays['root__' + k], k)[:TOKENS] for k in B2_KEYS}


# Torch (paths 1 and 4) ------------------------------------------------------------------------------------------------------------------
def actor_observation(arrays, legal):
    """A PokeZeroObservationV0 with the field values LiveProtocolActor._observation builds from the owning arrays: float32 numerics,
    every tensor as nested tuples (tolist); the legal mask given."""
    from pokezero.observation import PokeZeroObservationV0
    from pokezero.recorded_encoder_parity import SHAPES
    values = dict(arrays, legal_action_mask=np.asarray(legal, bool));result = {}
    for name, shape in SHAPES.items():
        a = np.asarray(values[name])
        if a.shape != shape:raise ValueError(f'{name}: {a.shape} is not the owning shape {shape}')
        if name == 'numeric_features':a = a.astype(np.float32)
        result[name] = tuple(tuple(row) for row in a.tolist()) if a.ndim == 2 else tuple(a.tolist())
    return PokeZeroObservationV0(**result, schema_version='pokezero.observation.v4')


def actor_tensors(observation, *, device='cpu'):
    """observation_window_to_torch on the one-frame window, as TransformerSoftmaxPolicy.select_action tensorizes it."""
    from pokezero.neural_policy import observation_window_to_torch
    return observation_window_to_torch([observation], window_size=1, device=device)


def precision_state(torch):
    return dict(mha_fastpath=bool(torch.backends.mha.get_fastpath_enabled()), matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32), cudnn_tf32=bool(torch.backends.cudnn.allow_tf32), threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(), version=torch.__version__)


# The live worker's settings (depthn_live_panel.serve); inter-op threads are set to 1 too but only recorded (no per-op numerics depend on them).
WANT_PRECISION = dict(mha_fastpath=False, matmul_precision='highest', cuda_matmul_tf32=False, cudnn_tf32=False, threads=1)


class ActorRuntime:
    """The live actor's champion: checkpoint, precision and preconditions (module docstring)."""
    def __init__(self, checkpoint, digest):
        from pokezero.neural_policy import TransformerSoftmaxPolicy, require_torch
        torch = require_torch();torch.set_num_threads(1)
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass   # already started: precision_state reports what holds
        torch.set_float32_matmul_precision('highest');torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
        from pokezero import foulplay_bridge as bridge
        from pokezero.live_protocol_actor import ActorContract, RawPolicy
        model, result = bridge.load_transformer_checkpoint(checkpoint, map_location='cpu');model.eval()
        policy = TransformerSoftmaxPolicy(model=model, result=result, deterministic=True, sampling_temperature=1., device='cpu',
            policy_id=result.model_config.policy_id, checkpoint_path=str(Path(checkpoint).resolve()), weights_sha256=digest)
        self.raw = RawPolicy.from_transformer(ActorContract.champion(), policy)   # the actor's own preconditions; raises otherwise
        self.state = precision_state(torch)
        odd = {k: (self.state[k], v) for k, v in WANT_PRECISION.items() if self.state[k] != v}
        if odd or model.training or result.model_config.window_size != 1:raise RuntimeError(f'the live actor\'s torch preconditions do not hold: {odd}')
        self.torch, self.policy = torch, policy

    def run(self, arrays, legal):
        from pokezero.neural_policy import _greedy_action_index, _masked_action_probabilities
        from pokezero.policy import legal_action_indices
        observation = actor_observation(arrays, legal);tensors = actor_tensors(observation, device=self.policy.device)
        with self.torch.no_grad():
            out = self.policy.forward_fn(tensors)
            probabilities = _masked_action_probabilities(out.policy_logits[0], tensors['legal_action_mask'][0], temperature=self.policy.sampling_temperature)
            values = tuple(float(probabilities[i].detach().cpu().item()) for i in range(9))
            logits = [float(x) for x in out.policy_logits[0].detach().cpu().tolist()];value = float(out.value[0].detach().cpu().item())
        greedy = _greedy_action_index(probabilities=values, legal=legal_action_indices(observation.legal_action_mask), family_gated=self.policy.family_gated_selection)
        chosen = self.raw.select_action(observation, rng=random.Random(0)).action_index;self.policy.reset()
        return dict(logits=logits, probabilities=list(values), greedy=greedy, value=value, select_action=chosen, select_agrees=chosen == greedy)


def torch_main(argv):
    p = argparse.ArgumentParser(prog='depthn_input_parity.py torch', allow_abbrev=False)
    p.add_argument('--capture', nargs='+', required=True, help='capture directories')
    p.add_argument('--checkpoint', required=True);p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--out', required=True, help='a new JSONL file');p.add_argument('--limit', type=int, help='only the first N captured decisions')
    args = p.parse_args(argv)
    if args.checkpoint_sha256 != CHAMPION_CHECKPOINT_SHA256:p.error(f'--checkpoint-sha256 must be the champion\'s pinned {CHAMPION_CHECKPOINT_SHA256}')
    digest = file_sha256(args.checkpoint)
    if digest != args.checkpoint_sha256:raise SystemExit(f'depthn_input_parity: {args.checkpoint} is {digest}, not {args.checkpoint_sha256}')
    runtime = ActorRuntime(args.checkpoint, digest);entries = capture_entries(args.capture);n = 0;clock = time.perf_counter()
    write_json(str(args.out) + '.header.json', dict(argv=list(argv), checkpoint=str(args.checkpoint), checkpoint_sha256=digest, precision=runtime.state,
        captures=list(args.capture), decisions=len(entries)))
    with open(args.out, 'x') as stream:
        for entry, arrays in decision_arrays(entries):
            if args.limit is not None and n >= args.limit:break
            if arrays is None or 'b2__numeric_features' not in arrays:continue
            legal = [bool(x) for x in arrays['root__legal']]
            row = dict(schema=TORCH_SCHEMA, ordinal=entry['ordinal'], decision_index=entry['decision_index'], set=entry['set'], legal=legal,
                checkpoint_sha256=digest, b2_legal_equals_root_legal=bool(np.array_equal(np.asarray(arrays.get('b2__legal_action_mask', legal), bool), legal)),
                path1=runtime.run(b2_arrays(arrays), legal), path4=runtime.run(prefix_arrays(arrays), legal))
            stream.write(encode(row) + '\n');stream.flush();n += 1
            if n % 100 == 0:print(json.dumps(dict(torch_done=n, seconds=time.perf_counter() - clock)), file=sys.stderr, flush=True)
        os.fsync(stream.fileno())
    print(json.dumps(dict(done=str(args.out), decisions=n, precision=runtime.state)), flush=True)
    return 0 if n else 2


# Feature layout and tags ------------------------------------------------------------------------------------------------------------
SECTIONS = (('field', 0, 1, 0), ('own', 1, 7, 1), ('opponent', 7, 13, 2), ('actions', 13, 22, 3), ('stats', 22, 23, 5))   # name, tokens, role


def token_layout():
    """Per token 0..22: its section, TOKEN_ROLES role and champion expert group, checked against the code (v5_model.TOKEN_ROLES,
    engine_device_observation.TOKEN_TYPES, champion.Config's default expert_groups, observation.py's token counts)."""
    from pokezero import observation as o
    from pokezero.v5_model import TOKEN_ROLES
    from gpu_engine import champion, engine_device_observation as device
    counts = (o.FIELD_TOKEN_COUNT, o.SELF_POKEMON_TOKEN_COUNT, o.OPPONENT_POKEMON_TOKEN_COUNT, o.ACTION_CANDIDATE_TOKEN_COUNT, o.OPPONENT_TENDENCY_STATS_TOKEN_COUNT)
    if tuple(stop - start for _, start, stop, _ in SECTIONS) != counts:raise AssertionError('token sections differ from observation.py')
    groups = champion.Config(numeric=NUMERIC, categorical=CATEGORICAL, tokens=TOKENS).expert_groups
    out = []
    for name, start, stop, role in SECTIONS:
        for t in range(start, stop):
            if TOKEN_ROLES[t] != role or device.TOKEN_TYPES[t] != role:raise AssertionError(f'token {t}: role {TOKEN_ROLES[t]} is not {name}\'s {role}')
            out.append(dict(token=t, section=name, slot=t - start, role=role, expert_group=groups[t], label=f'{name}[{t - start}]'))
    return out


def numeric_layout():
    """Per v4 numeric column: its writer names (showdown NUMERIC_* through v4_numeric_index; ranges as NAME[k]) and v4 layout group."""
    from pokezero import showdown as s
    names = {c: [] for c in range(NUMERIC)}
    for name, value in vars(s).items():
        if name.startswith('NUMERIC_') and type(value) is int and not name.endswith('_OFFSET'):   # an _OFFSET names a range (below)
            try:names[s.v4_numeric_index(value)].append(name[8:])
            except (ValueError, KeyError):pass
    for base, count in (('OPP_MOVE_PP', s.BELIEF_MOVE_BUCKET_COUNT), ('OPP_MOVE_PP_VALID', s.BELIEF_MOVE_BUCKET_COUNT), ('STAT_WEATHER_REVEAL', 8)):
        offset = getattr(s, f'NUMERIC_{base}_OFFSET')
        for k in range(count):
            try:column = s.v4_numeric_index(offset + k)
            except (ValueError, KeyError):continue
            if not names[column]:names[column].append(f'{base}[{k}]')
    group = {}
    for g, indices in s.V4_NUMERIC_LAYOUT_GROUPS:
        for legacy in indices:group[s.v4_numeric_index(legacy)] = g
    return [dict(column=c, names=sorted(names[c]) or ['unknown'], group=group.get(c, 'unknown')) for c in range(NUMERIC)]


def categorical_layout():
    """Per v4 categorical column: its name (showdown CATEGORY_*: nine fixed columns, the belief ability/item/move and volatile
    buckets, then the v4 pack's last used move and traced ability)."""
    from pokezero import showdown as s
    names = {}
    for name, value in vars(s).items():
        if name.startswith('CATEGORY_') and type(value) is int and 0 <= value < s.CATEGORY_FIXED_COUNT and not name.endswith(('_COUNT', '_OFFSET', '_BUCKETS')):
            names.setdefault(value, []).append(name[9:])
    for base, offset, count in (('BELIEF_ABILITY', s.CATEGORY_BELIEF_ABILITY_OFFSET, s.BELIEF_ABILITY_BUCKET_COUNT),
            ('BELIEF_ITEM', s.CATEGORY_BELIEF_ITEM_OFFSET, s.BELIEF_ITEM_BUCKET_COUNT), ('BELIEF_MOVE', s.CATEGORY_BELIEF_MOVE_OFFSET, s.BELIEF_MOVE_BUCKET_COUNT),
            ('VOLATILE', s.CATEGORY_VOLATILE_OFFSET, s.VOLATILE_BUCKET_COUNT)):
        for k in range(count):names.setdefault(offset + k, []).append(f'{base}[{k}]')
    names.setdefault(s.CATEGORY_LAST_USED_MOVE, []).append('LAST_USED_MOVE');names.setdefault(s.CATEGORY_TRACED_ABILITY, []).append('TRACED_ABILITY')
    return [dict(column=c, names=sorted(names.get(c, [])) or ['unknown']) for c in range(CATEGORICAL)]


OWN_PRIVATE_NUMERIC = ('ACTUAL_', 'POSSIBLE_ABILITY_COUNT', 'POSSIBLE_ITEM_COUNT', 'LEGAL')
OWN_PUBLIC_NUMERIC = ('BOOST_',)


def feature_tags(section, kind, column, names, reason):
    """(side, information status, rationale) of one (section, kind, column) cell (summary.md states the rules). Side: own, opponent
    or field; status: public, own-private, opponent-revealed, opponent-unrevealed, 'none' (a structural zero of this section in the
    device encoder's column manifest) or 'unknown' (the code does not settle it)."""
    from pokezero import showdown as s
    from gpu_engine.public_belief_features import NUMERIC_COLUMNS as BELIEF, CATEGORY_COLUMNS as BELIEF_CATEGORIES
    from gpu_engine.public_expected_stats import NUMERIC_COLUMNS as EXPECTED
    names = [n for n in names if n != 'unknown']
    if section in ('own', 'actions'):side = 'own'
    elif section == 'opponent':side = 'opponent'
    elif kind == 'numeric' and any(n.startswith('SELF_') or n.endswith('_SELF') or n.startswith('STAT_MY_') for n in names):side = 'own'
    elif kind == 'numeric' and any(n.startswith(('OPP_', 'STAT_')) or n.endswith('_OPP') for n in names):side = 'opponent'
    else:side = 'field'
    if kind in ('numeric', 'categorical') and str(reason or '').startswith('structurally'):
        return side, 'none', 'structural zero for this token section (device encoder column manifest)'
    if section == 'stats':return side, 'public', 'opponent-tendency counters from the public battle history'
    if section == 'field':
        if any(n.endswith('HAZARD_EXPECTED') for n in names):return side, 'unknown', 'manifest: own request facts + public belief (mixed)'
        return side, 'public', 'field token: public battle state'
    if section == 'actions':return side, 'own-private', 'own legal actions from the own request (moveset, PP, switch targets)'
    belief_categories = set(range(s.CATEGORY_BELIEF_ABILITY_OFFSET, s.CATEGORY_VOLATILE_OFFSET))
    if section == 'opponent':
        if kind == 'numeric' and (column in set(BELIEF) | set(EXPECTED) or 'TRAPPER_ALIVE' in names):
            return side, 'opponent-unrevealed', 'belief over the opponent\'s unrevealed set (public_belief_features / public_expected_stats)'
        if kind == 'categorical' and column in set(BELIEF_CATEGORIES):
            return side, 'opponent-unrevealed', 'belief possibility buckets (public_belief_features.CATEGORY_COLUMNS)'
        return side, 'opponent-revealed', 'opponent token outside the belief columns: the public encounter tracker'
    if kind == 'numeric' and any(n.startswith(OWN_PRIVATE_NUMERIC) for n in names):return side, 'own-private', 'own request facts only the owner has'
    if kind == 'categorical' and column in belief_categories:return side, 'own-private', 'own ability / item / moves from the own request'
    if kind == 'numeric' and any(n.startswith(OWN_PUBLIC_NUMERIC) for n in names):return side, 'public', 'boosts are announced on the wire'
    if kind == 'categorical' and column in {s.CATEGORY_LAST_USED_MOVE, s.CATEGORY_TRACED_ABILITY, *range(s.CATEGORY_VOLATILE_OFFSET, s.CATEGORY_VOLATILE_OFFSET + 6)}:
        return side, 'public', 'active-mon volatile / last move / traced ability: announced on the wire'
    return side, 'unknown', 'own token: request-exact own party; whether the opponent observed it depends on reveal history'


def feature_layout(manifest=None):
    """Every (section, kind, column) cell of the 23-token prefix with names, group, tags and the device encoder's manifest status
    (the installed manifest from the capture when given, else the static one: names only)."""
    from gpu_engine import engine_device_observation as device
    static = {(r['section'], r['kind'], r['column']): r for r in device.column_manifest()}
    installed = None if manifest is None else {(r['section'], r['kind'], r['column']): r for r in manifest}
    numeric, categorical = numeric_layout(), categorical_layout();cells = {}
    for section, start, stop, _ in SECTIONS:
        for kind, table in (('numeric', numeric), ('categorical', categorical)):
            for row in table:
                c = row['column'];basis = static[section, kind, c];status = None if installed is None else installed.get((section, kind, c), {})
                side, info, why = feature_tags(section, kind, c, row['names'], (status or basis)['reason'])
                cells[section, kind, c] = dict(section=section, kind=kind, column=c, tokens=[start, stop], names=row['names'], group=row.get('group'),
                    side=side, info=info, rationale=why, static_reason=basis['reason'], installed_status=None if status is None else status.get('status'),
                    installed_reason=None if status is None else status.get('reason'))
        for kind in ('token_type', 'attention'):
            side = {'own': 'own', 'actions': 'own', 'opponent': 'opponent'}.get(section, 'field')
            cells[section, kind, -1] = dict(section=section, kind=kind, column=-1, tokens=[start, stop], names=[kind], group=None, side=side,
                info='unknown', rationale='token structure', static_reason=None, installed_status=None, installed_reason=None)
    cells['all', 'history_mask', -1] = dict(section='all', kind='history_mask', column=-1, tokens=None, names=['history_mask'], group=None, side='field',
        info='unknown', rationale='frame mask', static_reason=None, installed_status=None, installed_reason=None)
    return cells


def section_of(token):
    return next(name for name, start, stop, _ in SECTIONS if start <= token < stop)


def diff_cells(diff):
    """The (section, kind, column) cells where an encoding_diff differs."""
    out = set()
    for kind, key in (('numeric', 'numeric_features'), ('categorical', 'categorical_ids')):
        for t, c in zip(*np.nonzero(diff[key])):out.add((section_of(int(t)), kind, int(c)))
    for kind, key in (('token_type', 'token_type_ids'), ('attention', 'attention_mask')):
        for t in np.nonzero(diff[key])[0]:out.add((section_of(int(t)), kind, -1))
    return out


def crossworld_cells(cross):
    out = set()
    for key, t, c in cross.get('cells') or ():
        if key == 'history_mask' or t < 0:out.add(('all', 'history_mask', -1) if key == 'history_mask' else ('all', key, -1));continue
        kind = {'numeric_features': 'numeric', 'categorical_ids': 'categorical', 'token_type_ids': 'token_type', 'attention_mask': 'attention'}[key]
        out.add((section_of(t), kind, c if kind in ('numeric', 'categorical') else -1))
    return out


# Report -------------------------------------------------------------------------------------------------------------------------------
def quantiles(values):
    """p50 / p95 / max (nearest rank) of the finite values, with the count and the count of non-finite ones."""
    v = sorted(x for x in values if x is not None and math.isfinite(x));bad = sum(1 for x in values if x is None or not math.isfinite(x))
    at = lambda q: v[max(0, math.ceil(q * len(v)) - 1)] if v else None
    return dict(n=len(v), nonfinite=bad, p50=at(.5), p95=at(.95), max=v[-1] if v else None)


def max_gap(x, y, legal):
    """max |x - y| over the legal actions (None when either side is missing; NaN propagates as inf)."""
    if x is None or y is None:return None
    gaps = [abs(x[a] - y[a]) for a in range(9) if legal[a]]
    return max((g if math.isfinite(g) else math.inf) for g in gaps) if gaps else None


def decision_paths(entry, torch_row):
    """dict(live, server, legal, logits {path: [9]}, action {path: int}) of one decision (paths without data omitted)."""
    legal = entry['root_legal'];logits, action = {}, {}
    if entry.get('root_logits') is not None:logits[2] = entry['root_logits'];action[2] = argmax_legal(entry['root_logits'], legal)
    p3 = entry.get('path3')
    if p3 and p3.get('logits') is not None:logits[3] = p3['logits'];action[3] = argmax_legal(p3['logits'], legal) if p3.get('valid') else None
    if torch_row is not None:
        for path, key in ((1, 'path1'), (4, 'path4')):logits[path] = torch_row[key]['logits'];action[path] = torch_row[key]['greedy']
    return dict(live=entry['selection']['live_greedy'], server=action.get(2), legal=legal, logits=logits, action=action)


def path_metrics(decisions, path):
    have = [d for d in decisions if path in d['logits']]
    return dict(n=len(have), agree_live=sum(d['action'][path] == d['live'] for d in have), agree_server=sum(d['action'][path] == d['server'] for d in have),
        agree_live_share=sum(d['action'][path] == d['live'] for d in have) / len(have) if have else None,
        agree_server_share=sum(d['action'][path] == d['server'] for d in have) / len(have) if have else None,
        dlogit_vs_path1=quantiles([max_gap(d['logits'][path], d['logits'].get(1), d['legal']) for d in have if 1 in d['logits']]),
        dlogit_vs_recorded=quantiles([max_gap(d['logits'][path], d['logits'].get(2), d['legal']) for d in have if 2 in d['logits']]))


def gate(entries, decisions):
    """Path 1 reproduces the live greedy on >= GATE_PATH1 of the decisions it covers; path 2 reproduces every captured root prior
    bit for bit. dict(verdict HARNESS_EXACT / HARNESS_NOT_EXACT, both checks, the misses)."""
    p1 = [(e, d) for e, d in zip(entries, decisions) if 1 in d['logits']]
    p1_miss = [dict(ordinal=e['ordinal'], decision_index=e['decision_index'], set=e['set'], live=d['live'], path1=d['action'][1]) for e, d in p1 if d['action'][1] != d['live']]
    captured = [e for e in entries if e.get('captured')]
    p2_miss = [dict(ordinal=e['ordinal'], decision_index=e['decision_index'], set=e['set'], path2=e.get('path2')) for e in captured
        if not (e.get('path2') and e['path2'].get('logits_equal') and e['path2'].get('value_equal'))]
    rate = 1 - len(p1_miss) / len(p1) if p1 else None
    ok1 = rate is not None and rate >= GATE_PATH1;ok2 = bool(captured) and not p2_miss
    return dict(verdict='HARNESS_EXACT' if ok1 and ok2 else 'HARNESS_NOT_EXACT',
        path1=dict(n=len(p1), reproduced=len(p1) - len(p1_miss), share=rate, threshold=GATE_PATH1, passed=ok1, misses=p1_miss),
        path2=dict(n=len(captured), bit_exact=len(captured) - len(p2_miss), passed=ok2, misses=p2_miss))


class Tally:
    """Per set: decisions, and the decisions in which each feature (any hashable) differs."""
    def __init__(self):
        self.n, self.counts = collections.Counter(), {s: collections.Counter() for s in SETS}

    def add(self, name, features):
        self.n[name] += 1;self.counts[name].update(set(features))

    def share(self, name, feature):
        return self.counts[name][feature] / self.n[name] if self.n[name] else None

    def ranked(self, label=lambda f: {}):
        """Every feature that differs somewhere, by the disagree/control share ratio (Jeffreys-smoothed for ranking: (d + .5) /
        (n_d + 1) over (c + .5) / (n_c + 1)), then the disagree count."""
        nd, nc = self.n['disagree'], self.n['control'];rows = []
        for f in set(self.counts['disagree']) | set(self.counts['control']):
            d, c = self.counts['disagree'][f], self.counts['control'][f]
            ds, cs = (d / nd if nd else None), (c / nc if nc else None)
            rows.append(dict(feature=list(f) if isinstance(f, tuple) else f, **label(f), disagree=d, control=c, disagree_share=ds, control_share=cs,
                ratio=None if not cs else ds / cs, control_zero=c == 0, smoothed_ratio=((d + .5) / (nd + 1)) / ((c + .5) / (nc + 1))))
        return sorted(rows, key=lambda r: (-r['smoothed_ratio'], -r['disagree'], str(r['feature'])))

    def table(self, features):
        return [dict(feature=f, **{s: self.counts[s][f] for s in SETS}, **{f'{s}_share': self.share(s, f) for s in SETS}) for f in features]


def report(entries, torch_rows, *, manifest=None, selection=None, coverage=None):
    """The summary dict (module docstring) over capture entries, {(ordinal, decision_index): torch row} and the arrays."""
    cells = feature_layout(manifest);tokens = token_layout()
    label = lambda f: dict(names=cells[f]['names'], side=cells[f]['side'], info=cells[f]['info'], group=cells[f]['group']) if f in cells else {}
    usable = [e for e in entries if e.get('root_legal') is not None]
    decisions = [decision_paths(e, torch_rows.get((e['ordinal'], e['decision_index']))) for e in usable]
    out = dict(schema=SUMMARY_SCHEMA, gate=gate(usable, decisions), paths={}, coverage=coverage, selection=selection)
    for path, what in PATHS.items():
        out['paths'][str(path)] = dict(what=what, **{s: path_metrics([d for e, d in zip(usable, decisions) if e['set'] == s], path) for s in SETS})
    # Encoding diff: B2 against the world-0 prefix.
    keys, tok, num, cat, cell, tag = (Tally() for _ in range(6));noise = collections.Counter();bit_only = collections.Counter();compared = 0
    for entry, arrays in decision_arrays(usable):
        if arrays is None or 'b2__numeric_features' not in arrays:continue
        diff = encoding_diff(b2_arrays(arrays), {k: arrays['root__' + k] for k in B2_KEYS});name = entry['set'];compared += 1
        keys.add(name, [k for k, equal in diff['keys'].items() if not equal])
        token = diff['numeric_features'].any(1) | diff['categorical_ids'].any(1) | diff['token_type_ids'] | diff['attention_mask']
        tok.add(name, [int(t) for t in np.nonzero(token)[0]]);num.add(name, [int(c) for c in np.nonzero(diff['numeric_features'].any(0))[0]])
        cat.add(name, [int(c) for c in np.nonzero(diff['categorical_ids'].any(0))[0]])
        differing = diff_cells(diff);cell.add(name, differing)
        tag.add(name, {x for f in differing for x in (('side', cells[f]['side']), ('info', cells[f]['info']), ('side_info', cells[f]['side'] + '|' + cells[f]['info']))})
        if not all(diff['keys'].values()):
            noise[name, 'differing'] += 1
            if diff['max_numeric_diff'] < 1e-6:noise[name, 'max_numeric_diff_below_1e-6'] += 1
            if diff['max_numeric_diff'] < 1e-6 and all(diff['keys'][k] for k in B2_KEYS[1:]):noise[name, 'numeric_only_below_1e-6'] += 1
        bit_only[name] += diff['bit_only']
    out['encoding_diff'] = dict(compared=compared, decisions=dict(keys.n),
        keys=keys.table(list(B2_KEYS)),
        tokens=[dict(r, label=tokens[r['feature']]['label']) for r in tok.table(list(range(TOKENS)))],
        numeric_columns=[dict(r, names=cells['own', 'numeric', r['feature']]['names'], group=cells['own', 'numeric', r['feature']]['group']) for r in num.table(list(range(NUMERIC)))],
        categorical_columns=[dict(r, names=cells['own', 'categorical', r['feature']]['names']) for r in cat.table(list(range(CATEGORICAL)))],
        ranked_cells=cell.ranked(label), ranked_tokens=[dict(r, label=tokens[r['feature']]['label']) for r in tok.ranked()],
        ranked_numeric_columns=num.ranked(lambda c: dict(names=cells['own', 'numeric', c]['names'])),
        ranked_categorical_columns=cat.ranked(lambda c: dict(names=cells['own', 'categorical', c]['names'])),
        by_tag=tag.ranked(), float_noise={f'{s}:{k}': v for (s, k), v in sorted(noise.items())}, bit_only_decisions=dict(bit_only))
    # Cross-world: our own view across the worlds of each root.
    cross, cross_cells = Tally(), Tally();worlds_compared = collections.Counter()
    for e in usable:
        c = e.get('crossworld')
        if not c:continue
        worlds_compared[e['set']] += 1
        cross.add(e['set'], [k for k in ('prefix_differs', 'full_differs', 'value_differs', 'reference_not_reproduced')
            if {'prefix_differs': not c['prefix_equal'], 'full_differs': not c['full_equal'], 'value_differs': not c['value_equal'],
                'reference_not_reproduced': not c.get('reference_reproduces_root_inputs', True)}[k]])
        cross_cells.add(e['set'], crossworld_cells(c))
    out['crossworld'] = dict(roots=dict(worlds_compared), any_difference=cross.table(['prefix_differs', 'full_differs', 'value_differs', 'reference_not_reproduced']),
        ranked_cells=cross_cells.ranked(label), turn1_input_check=TURN1_INPUT_CHECK)
    # Attribution (descriptive).
    attr = {}
    for s in SETS:
        ds = [d for e, d in zip(usable, decisions) if e['set'] == s]
        p3 = [d for d in ds if d['action'].get(3) is not None];p4 = [d for d in ds if 4 in d['action']]
        attr[s] = dict(path3_n=len(p3), path3_agrees_live=sum(d['action'][3] == d['live'] for d in p3),
            path3_agrees_live_share=sum(d['action'][3] == d['live'] for d in p3) / len(p3) if p3 else None,
            path4_n=len(p4), path4_disagrees_live=sum(d['action'][4] != d['live'] for d in p4),
            path4_disagrees_live_share=sum(d['action'][4] != d['live'] for d in p4) / len(p4) if p4 else None)
    out['attribution'] = dict(attr, reading='path3_agrees_live_share high in the disagree set: the inputs carry the disagreement (the server model '
        'reproduces the actor on the actor\'s encoding); path4_disagrees_live_share high: the world-0 encoding moves the actor too. Both low: the model '
        '(bf16 champion vs torch fp32) carries it.')
    out['replay'] = dict(entries=len(entries), captured=sum(bool(e.get('captured')) for e in entries),
        action_equal=sum(bool((e.get('parity') or {}).get('action_equal')) for e in entries), q_equal=sum(bool((e.get('parity') or {}).get('q_equal')) for e in entries),
        root_logits_match_selection=sum(bool(e.get('root_logits_match_selection')) for e in entries),
        live_greedy_matches_selection=sum(bool(e.get('live_greedy_matches_selection')) for e in entries),
        b2_view_equal=sum(bool((e.get('b2_view') or {}).get('equal')) for e in entries), path3_invalid=sum(bool(e.get('path3')) and not e['path3']['valid'] for e in entries),
        path3_lossy=sum(bool(e.get('path3') and e['path3']['lossy_keys']) for e in entries),
        torch_select_disagrees=sum(not r[p]['select_agrees'] for r in torch_rows.values() for p in ('path1', 'path4')),
        errors=[dict(ordinal=e['ordinal'], decision_index=e['decision_index'], errors=e['errors']) for e in entries if e.get('errors')])
    out['layout'] = dict(tokens=tokens, cells=[dict(v) for v in cells.values()], manifest='installed (capture)' if manifest is not None else 'static (names only)',
        tag_rules=TAG_RULES)
    return out


TURN1_INPUT_CHECK = ('device_depth2_pricer.make_turn1_stage.turn1 (the depth-N pricer\'s program P1, device_depthn_pricer.py:246): for every world slot '
    'of every block (both worlds of each pair, so all W worlds of a root) it recomputes OUR view jax.vmap(views)(states, publics, owners) with '
    'views = _make_views(executor) and compares each key of INPUT_KEYS present in request.root_inputs (all five: the full 29 tokens, every column, '
    'and history_mask) elementwise with == against request.root_inputs, which build set to world 0\'s view; the first differing key k sets pre = 3 + '
    'index(k) (code 10 when the key sets differ). A world with pre != 0 is refused (device_depthn_pricer._reports: "actual learner root input '
    'differs: <key>"), its block refuses (depthn_backups.refusal_summary cause "pre") and DepthNLiveSearch._search refuses the root. So no priced, '
    'COMPLETE root can carry a cross-world difference in values; only == -equal bits (+0.0 / -0.0) can survive. NaN never compares equal and refuses.')
TAG_RULES = ('Side: tokens 1-6 and 13-21 own, 7-12 opponent; field (token 0) and stats (token 22) columns by name (SELF_/_SELF/STAT_MY_ own, '
    'OPP_/_OPP/STAT_ opponent, else field). Status: structurally-not-applicable cells of the device encoder\'s column manifest "none"; stats token '
    'public (tendency counters from the public history); field token public except *_HAZARD_EXPECTED unknown (the manifest: own request facts + public '
    'belief); action tokens own-private (own moveset, PP, switch targets); opponent tokens opponent-unrevealed on the belief columns '
    '(public_belief_features NUMERIC/CATEGORY_COLUMNS, public_expected_stats, TRAPPER_ALIVE) else opponent-revealed; own tokens own-private for '
    'ACTUAL_*, POSSIBLE_ABILITY/ITEM_COUNT, LEGAL and the belief ability/item/move buckets, public for boosts, volatiles, last move and traced ability, '
    'unknown otherwise (request-exact values whose visibility depends on reveal history); token_type / attention / history_mask unknown.')


def markdown(s):
    """summary.md: the gate, the four paths, the encoding diff (top cells and tags), the cross-world diff and the attribution."""
    pct = lambda x: '-' if x is None else f'{100 * x:.1f}%'
    num = lambda x: '-' if x is None else f'{x:.4g}'
    g = s['gate'];lines = ['# DepthN root prior vs live greedy: input parity', '', f'**Gate: {g["verdict"]}**', '',
        f'- path 1 reproduces the live greedy: {g["path1"]["reproduced"]}/{g["path1"]["n"]} ({pct(g["path1"]["share"])}; threshold {pct(GATE_PATH1)})',
        f'- path 2 reproduces the recorded prior bit for bit: {g["path2"]["bit_exact"]}/{g["path2"]["n"]}']
    for name in ('path1', 'path2'):
        if g[name]['misses']:lines.append(f'- {name} misses ({len(g[name]["misses"])}): ' + ', '.join(f'{m["ordinal"]}/{m["decision_index"]}' for m in g[name]['misses'][:40]))
    lines += ['', '## Paths', '', '| path | set | n | = live greedy | = server prior | max dlogit vs path 1 p50 / p95 / max | vs recorded p50 / p95 / max |', '|---|---|---|---|---|---|---|']
    for p, row in s['paths'].items():
        for name in SETS:
            m = row[name];a, b = m['dlogit_vs_path1'], m['dlogit_vs_recorded']
            lines.append(f'| {p}: {row["what"]} | {name} | {m["n"]} | {pct(m["agree_live_share"])} | {pct(m["agree_server_share"])} | '
                f'{num(a["p50"])} / {num(a["p95"])} / {num(a["max"])} | {num(b["p50"])} / {num(b["p95"])} / {num(b["max"])} |')
    a = s['attribution']
    lines += ['', '## Attribution', '', '| set | path 3 (B2 -> server) = live | path 4 (world 0 -> torch) != live |', '|---|---|---|']
    for name in SETS:lines.append(f'| {name} | {a[name]["path3_agrees_live"]}/{a[name]["path3_n"]} ({pct(a[name]["path3_agrees_live_share"])}) | '
        f'{a[name]["path4_disagrees_live"]}/{a[name]["path4_n"]} ({pct(a[name]["path4_disagrees_live_share"])}) |')
    lines += ['', a['reading']]
    e = s['encoding_diff']
    lines += ['', '## Encoding diff: B2 vs world-0 prefix', '', f'decisions compared: {e["compared"]} ({e["decisions"]})', '', '| key | disagree | control |', '|---|---|---|']
    lines += [f'| {r["feature"]} differs | {r["disagree"]} ({pct(r["disagree_share"])}) | {r["control"]} ({pct(r["control_share"])}) |' for r in e['keys']]
    lines += ['', f'float noise: {e["float_noise"]}; bit-only (values equal): {e["bit_only_decisions"]}', '', '### By tag', '',
        '| tag | disagree | control | ratio |', '|---|---|---|---|']
    lines += [f'| {r["feature"][0]}: {r["feature"][1]} | {r["disagree"]} ({pct(r["disagree_share"])}) | {r["control"]} ({pct(r["control_share"])}) | '
        f'{"inf" if r["control_zero"] and r["disagree"] else num(r["ratio"])} |' for r in e['by_tag']]
    lines += ['', '### Top cells (section, kind, column) by disagree/control ratio', '', '| cell | names | side | status | disagree | control | ratio |', '|---|---|---|---|---|---|---|']
    lines += [f'| {r["feature"][0]} {r["feature"][1]} {r["feature"][2]} | {", ".join(r.get("names", []))} | {r.get("side")} | {r.get("info")} | '
        f'{r["disagree"]} ({pct(r["disagree_share"])}) | {r["control"]} ({pct(r["control_share"])}) | {"inf" if r["control_zero"] and r["disagree"] else num(r["ratio"])} |'
        for r in e['ranked_cells'][:40]]
    lines += ['', '### Tokens', '', '| token | disagree | control |', '|---|---|---|']
    lines += [f'| {r["feature"]} {r["label"]} | {pct(r["disagree_share"])} | {pct(r["control_share"])} |' for r in e['tokens']]
    c = s['crossworld']
    lines += ['', '## Cross-world: our own view across the worlds of a root', '', f'roots compared: {c["roots"]}', '', '| difference | disagree | control |', '|---|---|---|']
    lines += [f'| {r["feature"]} | {r["disagree"]} ({pct(r["disagree_share"])}) | {r["control"]} ({pct(r["control_share"])}) |' for r in c['any_difference']]
    if c['ranked_cells']:
        lines += ['', '| cell | names | side | status | disagree | control |', '|---|---|---|---|---|---|']
        lines += [f'| {r["feature"]} | {", ".join(r.get("names", []))} | {r.get("side")} | {r.get("info")} | {r["disagree"]} | {r["control"]} |' for r in c['ranked_cells'][:40]]
    lines += ['', c['turn1_input_check'], '', '## Replay', '', json.dumps({k: v for k, v in s['replay'].items() if k != 'errors'}), '',
        f'errors: {len(s["replay"]["errors"])}', '', '## Tag rules', '', TAG_RULES, f'Manifest: {s["layout"]["manifest"]}.', '']
    return '\n'.join(lines)


def report_main(argv):
    p = argparse.ArgumentParser(prog='depthn_input_parity.py report', allow_abbrev=False)
    p.add_argument('--capture', nargs='+', required=True);p.add_argument('--torch', nargs='*', default=[]);p.add_argument('--out', required=True, help='a new directory')
    args = p.parse_args(argv)
    entries = capture_entries(args.capture);torch_rows = {}
    for path in args.torch:
        for r in read_jsonl(path):
            key = (r['ordinal'], r['decision_index'])
            if key in torch_rows:raise SystemExit(f'depthn_input_parity: torch row {key} twice')
            torch_rows[key] = r
    manifest, summaries = None, []
    for d in map(Path, args.capture):
        summary = json.loads((d / 'capture-summary.json').read_text()) if (d / 'capture-summary.json').exists() else {}
        summaries.append(dict(directory=str(d), **{k: summary.get(k) for k in ('selected_in_shard', 'recorded', 'captured', 'path2_exact', 'interrupted', 'reprice_exit')},
            missing=len(summary.get('missing') or ()), errors=len(summary.get('errors') or ())))
        manifest = manifest or summary.get('installed_manifest')
    selection = json.loads((Path(args.capture[0]) / 'selection.json').read_text())
    coverage = dict(selected=selection['counts'], selected_decisions=len(selection['decisions']), entries=len(entries), torch_rows=len(torch_rows),
        captures=summaries, not_in_captures=len({(d['ordinal'], d['decision_index']) for d in selection['decisions']} - {(e['ordinal'], e['decision_index']) for e in entries}))
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    s = report(entries, torch_rows, manifest=manifest, selection=dict(counts=selection['counts'], select_seed=selection['select_seed']), coverage=coverage)
    write_json(out / 'summary.json', s)
    with (out / 'summary.md').open('x') as stream:stream.write(markdown(s))
    print(json.dumps(dict(done=str(out), verdict=s['gate']['verdict'], attribution={k: s['attribution'][k] for k in SETS})), flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = dict(capture=capture_main, torch=torch_main, report=report_main)
    if not argv or argv[0] not in commands:raise SystemExit('usage: depthn_input_parity.py {capture,torch,report} ...')
    return commands[argv[0]](argv[1:])


if __name__ == '__main__':raise SystemExit(main())
