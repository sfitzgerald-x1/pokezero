#!/usr/bin/env python3
"""Identity gate for the live search server's latency changes: the same recorded battles, the same outputs, byte for byte.

record: recorded battles (a panel run's owner captures, owning-wire/<ordinal>.jsonl, search-arm cells) are replayed decision by
decision through one ProtocolConstructor on the CPU with per-battle sessions on (saved-root admission off), as the live server
does; a single-legal decision is not sent (the live actor never sends one). For every sent decision it writes a canonical deep
digest of prepare_roots' output (Prepared or RootRefusal; arrays by dtype, shape and bytes, floats exactly, dataclasses, mappings
and sequences element by element in order; only timing fields and the request id left out), of classify's output, and of the
decision's public root (the root the constructor holds with --incremental-roots, else capture_root's from scratch), with the
unprofiled wall of prepare and of classify. Run it once on the original tree and once on the candidate tree (same inputs); the header records the imported tree's code digest
and git HEAD (code_identity), and compare refuses two records of the same code with the same flag (a vacuous comparison).

compare: two record files must hold the same decisions with equal digests; it reports every mismatch and both trees' walls by
decision band (0-2, 3-9, 10-19, 20+). Exit 0 only when every decision matches.

usage: depthn_root_identity.py record --showdown-root ... --run RUN_DIR --battles N --out NEW.jsonl [--incremental-roots]
       depthn_root_identity.py compare ORIGINAL.jsonl CANDIDATE.jsonl --out SUMMARY.json
"""
import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time

TIMING = re.compile(r'(^|_)seconds$|^_root_puct_branch_')
EXCLUDED = ('request_id',)


def canonical(value, shared, *, depth=0, stack=None):
    """A nested, JSON-encodable canonical form of value (module docstring); objects in `shared` (id -> name) become their name."""
    stack = set() if stack is None else stack
    if id(value) in shared:return ['shared', shared[id(value)]]
    if value is None or type(value) in (bool, int, str):return value
    if type(value) is float:return ['float', value.hex()]
    if depth > 60:return ['depth', type(value).__name__]
    if hasattr(value, '__array__') and hasattr(value, 'dtype') and hasattr(value, 'shape'):
        import numpy as np
        a = np.asarray(value);return ['array', a.dtype.str, list(a.shape), hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()]
    if isinstance(value, (bytes, bytearray)):return ['bytes', hashlib.sha256(bytes(value)).hexdigest()]
    if inspect.ismodule(value) or inspect.isroutine(value) or inspect.isclass(value):
        return ['code', getattr(value, '__module__', ''), getattr(value, '__qualname__', getattr(value, '__name__', repr(value)))]
    if id(value) in stack:return ['cycle', type(value).__name__]
    stack.add(id(value))
    try:
        rec = lambda v: canonical(v, shared, depth=depth + 1, stack=stack)
        name = type(value).__module__ + '.' + type(value).__qualname__
        if isinstance(value, dict):
            return ['dict', name, [[rec(k), rec(v)] for k, v in value.items() if not (type(k) is str and (TIMING.search(k) or k in EXCLUDED))]]
        if isinstance(value, (list, tuple)):
            fields = getattr(type(value), '_fields', None)
            if fields:return ['namedtuple', name, [[f, rec(getattr(value, f))] for f in fields if not (TIMING.search(f) or f in EXCLUDED)]]
            return ['seq', name, [rec(v) for v in value]]
        if isinstance(value, (set, frozenset)):return ['set', name, sorted(json.dumps(rec(v), sort_keys=True) for v in value)]
        if dataclasses.is_dataclass(value):
            return ['dataclass', name, [[f.name, rec(getattr(value, f.name))] for f in dataclasses.fields(value)
                                        if not (TIMING.search(f.name) or f.name in EXCLUDED)]]
        attrs = {}
        if hasattr(value, '__dict__'):attrs.update(vars(value))
        for slot in getattr(type(value), '__slots__', ()) or ():
            if isinstance(slot, str) and hasattr(value, slot):attrs[slot] = getattr(value, slot)
        if attrs:return ['object', name, [[k, rec(v)] for k, v in attrs.items() if not (TIMING.search(k) or k in EXCLUDED)]]
        text = repr(value)
        return ['opaque', name] if ' at 0x' in text else ['repr', name, text]   # a memory address differs between processes
    finally:
        stack.discard(id(value))


def digest(value, shared):
    return hashlib.sha256(json.dumps(canonical(value, shared), separators=(',', ':')).encode()).hexdigest()


def code_identity():
    """(root, sha256 of the imported tree's gpu_engine and src/pokezero .py files, git HEAD or None) of the code this process runs."""
    import subprocess
    import gpu_engine
    root = Path(gpu_engine.__file__).resolve().parents[1];h = hashlib.sha256()
    for base in ('gpu_engine', 'src/pokezero'):
        for path in sorted((root / base).rglob('*.py')):
            if '__pycache__' not in path.parts:h.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes())
    try:head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):head = None
    import pokezero
    if Path(pokezero.__file__).resolve().parents[2] != root:raise SystemExit('gpu_engine and pokezero come from different trees')
    return dict(root=str(root), code_sha256=h.hexdigest(), git_head=head)


def record(args):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import jax
    jax.config.update('jax_enable_x64', False);jax.config.update('jax_default_matmul_precision', 'highest')
    from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor, load_inputs
    from gpu_engine import constructor_worlds as cw
    from gpu_engine.depthn_live_search import classify
    from pokezero import phase_d_owning_capture as capture
    from pokezero.phase_d_search_server import SCHEMA
    from pokezero.live_protocol_actor import ActorContract, _packet
    from pokezero.showdown import _legal_action_mask
    from pokezero.v5_contract import DecisionAddress
    from pokezero.v5_teacher_public_root import capture_root
    keys = ('battle_id', 'owner', 'stream_id', 'decision_index', 'generation', 'prefix_sha256', 'lines', 'own_request_line')
    catalog, source, metadata, config = load_inputs(args)
    extra = {}
    if args.incremental_roots:
        if 'incremental_roots' not in inspect.signature(ProtocolConstructor).parameters:raise SystemExit('this tree has no incremental roots')
        extra['incremental_roots'] = True
    if args.eager_peer_fold:
        if 'eager_peer_fold' not in inspect.signature(ProtocolConstructor).parameters:raise SystemExit('this tree has no eager peer fold')
        extra['eager_peer_fold'] = True
    constructor = ProtocolConstructor(catalog=catalog, source=source, metadata=metadata, owning_config=config, count=16, seed=args.seed,
                                      rung='DN2W16', saved_root_admission=False, **extra)
    shared = {id(x): n for n, x in (('catalog', catalog), ('source', source), ('metadata', metadata), ('owning_config', config),
                                    ('functions', constructor.functions)) if x is not None}
    run = Path(args.run);cells = []
    for path in sorted((run / 'results').glob('*.json'), key=lambda p: int(p.stem)):
        r = json.loads(path.read_text())
        if (r.get('cell') or {}).get('arm') == args.arm and r.get('status') == 'COMPLETE' and (run / 'owning-wire' / f'{path.stem}.jsonl').exists():
            cells.append(int(path.stem))
    step = max(1, len(cells) // args.battles);chosen = cells[::step][:args.battles]
    header = dict(event='HEADER', run=str(run), arm=args.arm, battles=chosen, incremental_roots=args.incremental_roots, eager_peer_fold=args.eager_peer_fold, seed=args.seed,
                  tree=os.environ.get('TREE_COMMIT'), code=code_identity(), source=str(Path(__file__).resolve()))
    index = 0
    with open(args.out, 'x') as stream:
        stream.write(json.dumps(header) + '\n')
        for ordinal in chosen:
            for row in capture.read_capture(run / 'owning-wire' / f'{ordinal}.jsonl'):
                req = {k: row[k] for k in keys}
                try:legal = sum(map(bool, _legal_action_mask(_packet(req['own_request_line'], req['owner']))))
                except Exception:legal = None
                if legal == 1:continue
                body = dict(req, schema=SCHEMA, request_id=f'identity-{ordinal}-{req["decision_index"]}')
                clock = time.perf_counter();item = cw.prepare_roots(constructor, [body], start_index=index, device='cpu', peer_view_b=True)[0]
                prepare_wall = time.perf_counter() - clock;index += 1
                clock = time.perf_counter()
                try:classified = classify(constructor, body)
                except Exception as error:classified = ['error', type(error).__name__, str(error)[:200]]
                classify_wall = time.perf_counter() - clock
                held = getattr(constructor, 'roots', {}).get(((constructor.rung, req['battle_id'], req['owner'], req['stream_id']), req['decision_index'],
                                                               req['prefix_sha256'])) if args.incremental_roots else None
                if held is None:
                    try:held = capture_root(config=config, contract=ActorContract.champion(), lines=tuple(req['lines']), battle_id=req['battle_id'],
                                            address=DecisionAddress(req['stream_id'], req['decision_index'], int(req['owner'][1]) - 1), seed=args.seed & 0xffffffff)
                    except Exception as error:held = ['error', type(error).__name__, str(error)[:200]]
                out = dict(ordinal=ordinal, decision_index=req['decision_index'], lines=len(req['lines']), kind=type(item).__name__,
                           prepare=digest(item, shared), classify=digest(classified, shared), root=digest(held, shared),
                           prepare_wall=round(prepare_wall, 5), classify_wall=round(classify_wall, 5))
                stream.write(json.dumps(out) + '\n');stream.flush()
            print(json.dumps(dict(event='BATTLE', ordinal=ordinal, decisions=index)), flush=True)
    return 0


def compare(args):
    def load(path):
        rows = [json.loads(l) for l in open(path) if l.strip()];return rows[0], {(r['ordinal'], r['decision_index']): r for r in rows[1:]}
    (h0, a), (h1, b) = load(args.original), load(args.candidate)
    if h0['battles'] != h1['battles'] or h0['run'] != h1['run'] or h0['arm'] != h1['arm'] or h0['seed'] != h1['seed']:
        raise SystemExit('the two records replay different battles, runs, arms or seeds')
    c0, c1 = ((h.get('code') or {}).get('code_sha256') for h in (h0, h1))
    flags = lambda h: (h.get('incremental_roots'), h.get('eager_peer_fold'))
    if not c0 or not c1 or (c0 == c1 and flags(h0) == flags(h1)):
        raise SystemExit('two records of different code (or of different flags) required: a record compared with itself proves nothing')
    head = (h0.get('code') or {}).get('git_head') or ''
    if args.expect_original_head and not head.startswith(args.expect_original_head):
        raise SystemExit(f'the original record ran {head or "no git tree"}, not {args.expect_original_head}')
    mismatches = []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if x is None or y is None:mismatches.append(dict(decision=key, missing='original' if x is None else 'candidate'));continue
        diff = [f for f in ('kind', 'prepare', 'classify', 'root') if x[f] != y[f]]
        if diff:mismatches.append(dict(decision=key, fields=diff, kind=(x['kind'], y['kind'])))
    def pct(xs):
        xs = sorted(xs);return None if not xs else dict(n=len(xs), p50=round(xs[len(xs) // 2], 4), p95=round(xs[min(len(xs) - 1, int(.95 * len(xs)))], 4))
    bands = {'0-2': (0, 2), '3-9': (3, 9), '10-19': (10, 19), '20+': (20, 10**6)}
    walls = {name: {stage: {band: pct([r[stage] for r in rows.values() if lo <= r['decision_index'] <= hi]) for band, (lo, hi) in bands.items()}
                    for stage in ('prepare_wall', 'classify_wall')} for name, rows in (('original', a), ('candidate', b))}
    summary = dict(decisions=len(a), candidate_decisions=len(b), mismatches=len(mismatches), first_mismatches=mismatches[:20],
                   identical=not mismatches and len(a) == len(b), original=h0, candidate=h1, walls=walls)
    Path(args.out).write_text(json.dumps(summary, indent=1) + '\n');print(json.dumps(dict(identical=summary['identical'], decisions=len(a), mismatches=len(mismatches))))
    return 0 if summary['identical'] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0]);sub = p.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('record')
    for name in ('showdown-root', 'node-binary', 'metadata', 'expected-metadata-sha256', 'vocabulary', 'expected-vocabulary-sha256', 'run', 'out'):
        r.add_argument('--' + name, required=True)
    r.add_argument('--battles', type=int, default=24);r.add_argument('--arm', default='DN2W16');r.add_argument('--seed', type=int, default=20261103)
    r.add_argument('--incremental-roots', action='store_true');r.add_argument('--eager-peer-fold', action='store_true')
    c = sub.add_parser('compare');c.add_argument('original');c.add_argument('candidate');c.add_argument('--out', required=True)
    c.add_argument('--expect-original-head', help='the original record must have run this commit (a prefix of its git HEAD)')
    args = p.parse_args(argv)
    return record(args) if args.mode == 'record' else compare(args)


if __name__ == '__main__':
    sys.exit(main())
