#!/usr/bin/env python3
"""Wide-vs-blocks frame gate for the DepthN wide leaf layout (device_depthn_pricer leaf_layout='wide'), on the served stack.

Bind-time rule: every scored run that serves with --leaf-layout wide runs this gate at bind time, on its exact pinned stack
(tree, jax/jaxlib, GPU kind, XLA flags) and compilation cache, over about 30 roots at the bound rung, and launches only on
PASS (exit 0). Why: the wide layout's placement and digest are the blocks layout's by construction, and so are its leaf status
and error except the forward-usability code (error 4, and the status 3 it sets: the forward's `valid`, finite outputs and
|v| <= 1, computed on the lane's own row). That code and the leaf values equal the blocks layout's only while the champion
forward is row-position invariant at the compiled GEMM choices (compaction moves a lane to another row of its 2,187-row forward
chunk). That held on the GB300 frame gate under the pinned stack and autotune cache and does NOT hold on CPU. Re-gate after any
jax/XLA, cuBLAS or autotune-cache change.

usage: depthn_wide_leaf_frame_gate.py --roots ROOTS.jsonl --rung DN2W32 --out REPORT.json [--cache DIR] [--chunks C]
           [--wide-nodes N] [--max-roots N] [--min-priced N] [--negative-control] -- SERVER_FLAGS...
  SERVER_FLAGS  the scored run's own scripts/depthn_live_server flags (its inputs, champion, reply model, view and guard flags,
                --block-lanes, --blocks-per-call, ...). The gate builds that search (depthn_live_server.build, so ONE visible
                GPU and the pinned matmul precision) and prices every root on two pricers sharing its executor, evaluator and
                one program table: blocks first (its leaf program compiles or loads before the wide one), then wide.
  --roots       JSONL, one saved request per line with the eight address keys of a live request (battle_id, owner, stream_id,
                decision_index, generation, prefix_sha256, lines, own_request_line), as --warmup-requests.
  --rung        the bound rung (overrides the server flags'); --cache the persistent compilation cache (--compilation-cache).
  --chunks      the wide layout's forward chunks per call (default the server's --blocks-per-call); --wide-nodes its node
                capacity (default the server's, else lanes // 16).
  --min-priced  roots that must price WITH leaves (enumerated > 0) for a PASS (default 25): a root whose worlds were all
                refused or whose turn-1 cells are all terminal never runs a wide call, so it proves nothing.
  --negative-control  plants an off-by-one lane placement in the wide path: the gate then PASSES only when every priced root
                differs (the gate has power); use it once per stack, not for the bind decision.
Every pricer output array (device_depthn_pricer.arrays: every frame array, plies included) and the world reports are compared
bitwise per root. PASS iff at least --min-priced roots priced with leaves, the leaf total is positive, no root differs, no root
errs, and a ('leaf', 'wide', ...) program is in the table. Exit 0 PASS, 1 FAIL, 2 bad arguments, 3 setup error (the search
could not be built, or the report could not be written). Prints one line 'WIDE_LEAF_FRAME_GATE PASS|FAIL|SETUP_ERROR ...' and
writes the JSON report: per root the differing arrays and first positions, totals, the stack (jax and jaxlib versions, device
kind, XLA_FLAGS, the effective compilation cache and autotune directories, the tree's real HEAD beside the server's
--tree-commit label), the pricer identity, and the sha256 of every compiled program's HLO text (the leaf programs carry the
forward and its GEMM choices).
Follow-up, not yet available: the scored server does not log its programs' HLO sha256 at READY, so these hashes cannot yet be
matched against the run's own executables; until it does, the bind record ties the gate to the run by tree, stack and cache."""
import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TAG = 'WIDE_LEAF_FRAME_GATE'


def gate_pricers(search, *, chunks, wide_nodes=None):
    """(blocks, wide) DepthNPricers for one root per call, as make_pricers' bucket of the search's W/2 requests: the search's
    executor, evaluator and block lanes, the blocks pricer at the search's blocks per call and the wide one at `chunks`, one
    program table."""
    from gpu_engine import device_depthn_pricer as dn
    from gpu_engine.depthn_live_search import DEPTH, WIDTH
    first = search.pricers[search.buckets[0]]; table = {}; out = []
    for layout, calls in (('blocks', first.blocks_per_call), ('wide', chunks)):
        p = dn.DepthNPricer(first.executor, None, capacity=search.K, depth=DEPTH, k_own=WIDTH, k_opp=WIDTH, k1_own=WIDTH, k1_opp=WIDTH,
            block_lanes=first.block_lanes, blocks_per_call=calls, selection='prior', fixed_capacity=True, forward=search.forward,
            actor_capacity=len(search.params), **({} if layout == 'blocks' else dict(leaf_layout='wide', wide_nodes=wide_nodes)))
        p.bind(search.params); p.programs = table; out.append(p)
    return tuple(out)


def differ(a, b):
    """The names of the pricer output arrays (and 'world_reports') that differ between two frames, sorted."""
    import numpy as np
    from gpu_engine import device_depthn_pricer as dn
    A, B = dict(dn.arrays(a)), dict(dn.arrays(b))
    names = sorted(n for n in set(A) | set(B) if n not in A or n not in B or np.shape(A[n]) != np.shape(B[n])
                   or np.asarray(A[n]).dtype != np.asarray(B[n]).dtype or np.asarray(A[n]).tobytes() != np.asarray(B[n]).tobytes())
    if json.dumps(a.world_reports, sort_keys=True, default=repr) != json.dumps(b.world_reports, sort_keys=True, default=repr):
        names.append('world_reports')
    return names


def _off_by_one(real):
    """place_wide_leaf reading each packed lane's neighbour (the negative control's planted bug)."""
    def place(pack, fields, *args, **kw):
        last = max(0, pack.calls * pack.lanes - 1)
        return real(dataclasses.replace(pack, lane_pos=(pack.lane_pos + 1).clip(0, last)), fields, *args, **kw)
    return place


def gate(search, blocks, wide, requests, *, min_priced=25, negative_control=False, log=print):
    """Price every request (a dict with depthn_live_search.PROTOCOL_KEYS) on both pricers from one prepare (fresh sessions,
    the search's own prepare flags) and compare their frames. A root counts as priced only with enumerated leaves (a leafless one
    is NO_LEAVES: no wide call ran). Returns the report dict (verdict PASS / FAIL and, on FAIL, its reasons)."""
    import numpy as np
    from unittest.mock import patch
    from gpu_engine import constructor_worlds as cw, depthn_live_search as live, device_depthn_pricer as dn
    rows = []; tally = dict(roots=0, priced=0, leafless=0, refused=0, error=0, equal=0, differ=0, priced_differ=0, leaves=0)
    planted = patch.object(dn, 'place_wide_leaf', _off_by_one(dn.place_wide_leaf)) if negative_control else None
    for request in requests:
        row = dict(request_id=request.get('request_id')); tally['roots'] += 1
        try:
            search.reset_sessions()
            item = cw.prepare_roots(search.constructor, [{k: request[k] for k in live.PROTOCOL_KEYS}], device=search.device, **search._redaction())[0]
            if type(item) is not cw.Prepared or not item.made.fully_constructed:
                row.update(status='REFUSED', reason=getattr(item, 'reason', None) or 'world refusals'); tally['refused'] += 1; rows.append(row); continue
            made = cw.build(search.constructor, [item], blocks, search.params, binding=search.binding, recipe=search.recipe, only_complete=True)
            if not made.requests:
                row.update(status='REFUSED', reason='not built'); tally['refused'] += 1; rows.append(row); continue
            clock = time.perf_counter(); a, _ = cw.price(blocks, made, search.binding, search.recipe, search.params); row['blocks_seconds'] = time.perf_counter() - clock
            clock = time.perf_counter()
            if planted is not None:
                with planted:
                    b, _ = cw.price(wide, made, search.binding, search.recipe, search.params)
            else:
                b, _ = cw.price(wide, made, search.binding, search.recipe, search.params)
            row['wide_seconds'] = time.perf_counter() - clock
            names = differ(a, b); A, B = dict(dn.arrays(a)), dict(dn.arrays(b)); enumerated = int(np.count_nonzero(a.leaf_status))
            row.update(status='DIFFER' if names else ('EQUAL' if enumerated else 'NO_LEAVES'), differ=names, enumerated=enumerated,
                       blocks_device_seconds=a.timing['device_seconds'], wide_device_seconds=b.timing['device_seconds'],
                       wide_ply={k: v for k, v in b.timing['plies'][-1].items() if not k.endswith('_seconds')},
                       first={n: np.argwhere(np.asarray(A[n]) != np.asarray(B[n]))[:5].tolist() for n in names
                              if n in A and n in B and np.shape(A[n]) == np.shape(B[n])})
            tally['differ' if names else 'equal'] += 1
            if enumerated:
                tally['priced'] += 1; tally['leaves'] += enumerated; tally['priced_differ'] += bool(names)
            else:
                tally['leafless'] += 1
        except MemoryError:
            raise
        except Exception as error:   # a root that cannot be priced fails the gate (recorded, the gate goes on)
            row.update(status='ERROR', error=f'{type(error).__name__}: {error}'[:500]); tally['error'] += 1
        rows.append(row)
        log(f'{TAG} root {row["request_id"]} {row["status"]} {row.get("differ") or ""}')
    wide_ran = any(type(k) is tuple and len(k) > 1 and k[:2] == ('leaf', 'wide') for k in wide.programs)
    reasons = [f'{tally["priced"]} roots priced with leaves, under --min-priced {min_priced}'] if tally['priced'] < min_priced else []
    reasons += ['no enumerated leaves'] if not tally['leaves'] else []
    reasons += [f'{tally["error"]} root errors'] if tally['error'] else []
    reasons += [] if wide_ran else ['no wide leaf program in the table']
    if negative_control:   # the planted bug must show on every priced root
        reasons += [f'the planted bug showed on {tally["priced_differ"]} of {tally["priced"]} priced roots'] if tally['priced_differ'] != tally['priced'] else []
    else:
        reasons += [f'{tally["differ"]} roots differ'] if tally['differ'] else []
    return dict(verdict='FAIL' if reasons else 'PASS', reasons=reasons, min_priced=min_priced, negative_control=bool(negative_control), tally=tally, roots=rows,
                programs=program_hashes(blocks.programs), wide_geometry=wide._wide_geometry(wide.depth),
                blocks_geometry=dict(block_lanes=blocks.block_lanes, blocks_per_call=blocks.blocks_per_call))


def program_hashes(table):
    """{program key: sha256 of its compiled HLO text} (None where the executable has no text)."""
    out = {}
    for key, compiled in table.items():
        try:
            out[repr(key)] = hashlib.sha256(compiled.as_text().encode()).hexdigest()
        except Exception:
            out[repr(key)] = None
    return out


def stack_record(tree_head=None):
    """What the gate ran on: jax/jaxlib, the device, XLA_FLAGS, the effective compilation cache and the autotune directory JAX
    derives inside it (jax_persistent_cache_enable_xla_caches), and the tree's real HEAD."""
    import importlib.metadata
    import os
    import jax
    try:
        jaxlib = importlib.metadata.version('jaxlib')
    except importlib.metadata.PackageNotFoundError:
        jaxlib = None
    device = jax.devices()[0]; cache = jax.config.jax_compilation_cache_dir; xla_caches = getattr(jax.config, 'jax_persistent_cache_enable_xla_caches', '') or ''
    autotune = os.path.join(cache, 'xla_gpu_per_fusion_autotune_cache_dir') if cache and ('all' in xla_caches or 'xla_gpu_per_fusion_autotune_cache_dir' in xla_caches) else None
    return dict(jax=jax.__version__, jaxlib=jaxlib, platform=device.platform, device_kind=device.device_kind, devices=len(jax.devices()),
                xla_flags=os.environ.get('XLA_FLAGS'), compilation_cache_dir=cache, compilation_cache_enabled=bool(jax.config.jax_enable_compilation_cache),
                persistent_cache_enable_xla_caches=xla_caches or None, autotune_cache_dir=autotune, tree_head=tree_head)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--roots', required=True, help='JSONL of saved requests (the eight address keys per line)')
    p.add_argument('--rung', required=True, help='the bound rung, e.g. DN2W32')
    p.add_argument('--out', required=True, help='the JSON report path (must not exist)')
    p.add_argument('--cache', help='the persistent compilation cache directory (the server\'s --compilation-cache)')
    p.add_argument('--chunks', type=int, help='wide forward chunks per call (default the server\'s --blocks-per-call)')
    p.add_argument('--wide-nodes', type=int, help='wide node slots per call (default the server\'s --wide-nodes, else lanes // 16)')
    p.add_argument('--max-roots', type=int, help='gate the first N roots of --roots')
    p.add_argument('--min-priced', type=int, default=25, help='roots that must price with leaves for a PASS (default 25)')
    p.add_argument('--negative-control', action='store_true', help='plant an off-by-one wide placement: PASS iff every priced root differs')
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cut = argv.index('--') if '--' in argv else len(argv)
    p = parser(); a = p.parse_args(argv[:cut]); server_argv = argv[cut + 1:]
    if any(x is not None and x <= 0 for x in (a.chunks, a.wide_nodes, a.max_roots)) or a.min_priced <= 0:
        p.error('--chunks, --wide-nodes, --max-roots and --min-priced are positive')
    if a.max_roots is not None and a.max_roots < a.min_priced:
        p.error('--max-roots must be at least --min-priced')
    if Path(a.out).exists():
        p.error(f'{a.out} exists')
    if not Path(a.out).resolve().parent.is_dir():
        p.error(f'{a.out}: its directory does not exist')
    from scripts import depthn_live_server as srv
    from pokezero.phase_d_search_server import SCHEMA
    server = srv.options(server_argv + ['--rung', a.rung, '--buckets', '1', '--max-requests', '1'] + (['--compilation-cache', a.cache] if a.cache else []))
    try:
        saved = srv.read_requests(a.roots)
    except (OSError, ValueError) as error:
        p.error(f'--roots: {error}')
    saved = saved[:a.max_roots] if a.max_roots else saved
    try:   # setup: the scored run's search and the two pricers (exit 3, not a FAIL)
        search = srv.build(server)
        blocks, wide = gate_pricers(search, chunks=a.chunks or server.blocks_per_call, wide_nodes=a.wide_nodes if a.wide_nodes is not None else server.wide_nodes)
    except MemoryError:
        raise
    except Exception as error:
        print(f'{TAG} SETUP_ERROR build: {type(error).__name__}: {error}'[:1000], flush=True); return 3
    requests = [dict(r, schema=SCHEMA, request_id=f'gate-{i}') for i, r in enumerate(saved)]
    started = time.perf_counter()
    try:
        report = gate(search, blocks, wide, requests, min_priced=a.min_priced, negative_control=a.negative_control, log=lambda line: print(line, flush=True))
    except MemoryError:
        raise
    except Exception as error:
        print(f'{TAG} SETUP_ERROR gate report: {type(error).__name__}: {error}'[:1000], flush=True); return 3
    try:
        head = srv.tree_commit()
    except Exception as error:
        head = f'unavailable: {type(error).__name__}: {error}'[:200]
    try:
        report.update(rung=a.rung, roots_file=a.roots, cache=a.cache, chunks=a.chunks or server.blocks_per_call, seconds=time.perf_counter() - started,
                      stack=stack_record(head), tree_commit_label=search.tree_commit, pricer_identity=search.pricer_identity, argv=argv)
        Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True, default=repr))
    except MemoryError:
        raise
    except Exception as error:
        print(f'{TAG} SETUP_ERROR report: {type(error).__name__}: {error}'[:1000], flush=True); return 3
    t = report['tally']
    print(f'{TAG} {report["verdict"]}{" (negative control)" if a.negative_control else ""} rung={a.rung} chunks={report["chunks"]} '
          f'roots={t["roots"]} priced={t["priced"]}/{a.min_priced} leafless={t["leafless"]} equal={t["equal"]} differ={t["differ"]} '
          f'refused={t["refused"]} error={t["error"]} leaves={t["leaves"]} device={report["stack"]["device_kind"]} jax={report["stack"]["jax"]} '
          f'head={head} report={a.out}' + (f' reasons={"; ".join(report["reasons"])}' if report['reasons'] else ''), flush=True)
    return 0 if report['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
