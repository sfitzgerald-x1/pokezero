#!/usr/bin/env python3
"""R3pa live lane labels: parity with the offline labels on real prepared roots, the per-root cost, and the flag-off proof (CPU or
the one visible device; no model, no pricer).

The live server's constructor and executor are built from its own flags (--server-args: a JSON list of depthn_live_server's argv;
as depthn_live_server.build builds them, without the pricers, champion or reply model). Each request of --requests (one protocol
request per line, depthn_live_search.PROTOCOL_KEYS; production order, so the constructor's sessions fold as live) is prepared alone
with constructor_worlds.prepare_roots on the CPU (with the server's private-reveal redaction / view b). For every fully constructed
root, block order:
- live: gpu_engine.depthn_r3pa_labels.LaneLabels(executor, vocabulary).views(worlds, owner), then lane_labels_live;
- offline: per world, as the RR re-pricer records it (scripts/depthn_rr_reprice.py describe: jax.device_get of the world's state
  and public, opponent_facts, PeerLanes, lane_labels), then mc_ladder_backline.labels(world, seen_species(worlds)).
Both are compared string for string on every world and lane (world_sets_missing must agree too), and the views' legal masks
against PeerLanes'. Costs per root: views_seconds (stack, the device pass, the fetch), map_seconds (lane_labels_live), split into
each seat's first root (its compile) and the rest. Flag-off: the backend compile events (jax.monitoring) while the module is
imported and LaneLabels constructed must be 0; traces per seat after the run.

usage: python scripts/depthn_r3pa_labels_parity.py --server-args ARGV.json --requests REQUESTS.jsonl --out NEW.json [--limit N]
Exit 1 on any mismatch, or when nothing is compared.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / 'src')):
    if _path not in sys.path:sys.path.insert(0, _path)
EXAMPLES = 20


def stats(values):
    v = sorted(values)
    if not v:return None
    return dict(n=len(v), mean=statistics.fmean(v), median=statistics.median(v), p90=v[min(len(v) - 1, int(.9 * len(v)))], max=v[-1])


def build(argv):
    """(server args, constructor, executor) as depthn_live_server.build makes the constructor and the executor."""
    from scripts import depthn_live_server as srv
    from gpu_engine import constructor_worlds as cw, depthn_live_search as live
    from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor, load_inputs
    args = srv.options(argv);catalog, source, metadata, config = load_inputs(args)
    constructor = ProtocolConstructor(catalog=catalog, source=source, metadata=metadata, owning_config=config, count=live.RUNGS[args.rung], seed=args.seed,
        rung=args.rung, saved_root_admission=args.saved_root_admission, incremental_roots=bool(getattr(args, 'incremental_roots', False)),
        eager_peer_fold=bool(getattr(args, 'eager_peer_fold', False)))
    return args, constructor, cw.live_executor(metadata, catalog)


def offline_labels(worlds, owner, vocabulary, peer_lanes):
    """(labels or None for world_sets_missing, the peer legal masks, seconds): the RR re-pricer's capture, then the backline."""
    import jax
    from scripts import depthn_rr_reprice as R
    from gpu_engine import mc_ladder_backline as B
    started = time.perf_counter();rows = []
    for w in worlds:
        state, public = jax.device_get((w.state, w.public))
        facts = R.opponent_facts(state, owner, vocabulary, public);peer = peer_lanes(state, public, owner)
        peer['lanes'] = R.lane_labels(facts, peer);rows.append(dict(opponent=facts, peer=peer))
    seen = B.seen_species(rows)
    labels = None if seen is None else [B.labels(r, seen) for r in rows]
    return labels, [r['peer']['legal'] for r in rows], time.perf_counter() - started


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--server-args', required=True);p.add_argument('--requests', required=True);p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args(argv)
    out_path = Path(a.out)
    if out_path.exists():raise SystemExit(f'{out_path} exists: a fresh path required')
    server_argv = json.loads(Path(a.server_args).read_text())
    requests = [json.loads(line) for line in Path(a.requests).read_text().splitlines() if line.strip()][:a.limit]
    import jax
    compiles = []
    jax.monitoring.register_event_duration_secs_listener(lambda name, *x, **k: compiles.append(name) if name.endswith('backend_compile_duration') else None)
    args, constructor, executor = build(server_argv)
    from gpu_engine import constructor_worlds as cw
    from scripts import depthn_rr_reprice as R
    vocabulary = constructor.metadata['vocabulary'];peer_lanes = R.PeerLanes(executor)
    before = len(compiles);started = time.perf_counter()
    from gpu_engine import depthn_r3pa_labels as L
    labeller = L.LaneLabels(executor, vocabulary)
    flag_off = dict(backend_compiles_import_and_construct=len(compiles) - before, programs_after_construct=len(labeller.programs),
        traces_after_construct=dict(labeller.traces), construct_seconds=time.perf_counter() - started)
    redaction = dict(redact_private_reveals=True) if args.redact_private_reveals or args.peer_view_b else {}
    if args.peer_view_b:redaction['peer_view_b'] = True
    roots, examples, refused = [], [], {}
    totals = dict(roots=0, worlds=0, lanes=0, mismatched_lanes=0, legal_mismatched_worlds=0, both_world_sets_missing=0, missing_disagrees=0)
    seats_seen = set()
    for request in requests:
        item = cw.prepare_roots(constructor, [request], device='cpu', **redaction)[0]
        if type(item) is not cw.Prepared or not item.made.fully_constructed:
            kind = getattr(item, 'kind', 'not_fully_constructed');refused[kind] = refused.get(kind, 0) + 1;continue
        owner = int(item.detached.address.owner);worlds = item.made.worlds
        compiles_before = len(compiles);views = labeller.views(worlds, owner);view_compiles = len(compiles) - compiles_before
        clock = time.perf_counter()
        try:live, missing = L.lane_labels_live(views, labeller.names), False
        except L.WorldSetsMissing:live, missing = None, True
        map_seconds = time.perf_counter() - clock
        want, legal, offline_seconds = offline_labels(worlds, owner, vocabulary, peer_lanes)
        row = dict(request_id=request['request_id'], owner=owner, worlds=len(worlds), first_of_seat=owner not in seats_seen, views_seconds=views.seconds,
            view_compiles=view_compiles, map_seconds=map_seconds, offline_seconds=offline_seconds, world_sets_missing=missing)
        seats_seen.add(owner);totals['roots'] += 1;totals['worlds'] += len(worlds)
        legal_bad = [w for w in range(len(worlds)) if views.legal[w].tolist() != legal[w]];totals['legal_mismatched_worlds'] += len(legal_bad)
        if (want is None) != missing:
            totals['missing_disagrees'] += 1;row['mismatch'] = dict(live_missing=missing, offline_missing=want is None)
        elif missing:totals['both_world_sets_missing'] += 1
        else:
            bad = [(w, b, live[w][b], want[w][b]) for w in range(len(worlds)) for b in range(9) if live[w][b] != want[w][b]]
            totals['lanes'] += 9 * len(worlds);totals['mismatched_lanes'] += len(bad);row['mismatched_lanes'] = len(bad)
            examples.extend(dict(request_id=request['request_id'], world=w, lane=b, live=x, offline=y) for w, b, x, y in bad[:EXAMPLES - len(examples)])
            row['labels'] = live
        if legal_bad:row['legal_mismatched_worlds'] = legal_bad
        roots.append(row)
    warm = [r for r in roots if not r['first_of_seat']]
    report = dict(server_rung=args.rung, requests=len(requests), refused=refused, totals=totals, flag_off=flag_off,
        traces_after_run=dict(labeller.traces), programs_after_run=len(labeller.programs),
        cost=dict(first_of_seat=[dict(owner=r['owner'], views_seconds=r['views_seconds'], view_compiles=r['view_compiles']) for r in roots if r['first_of_seat']],
            warm_views_seconds=stats([r['views_seconds'] for r in warm]), warm_map_seconds=stats([r['map_seconds'] for r in warm]),
            warm_total_seconds=stats([r['views_seconds'] + r['map_seconds'] for r in warm]), warm_view_compiles=sum(r['view_compiles'] for r in warm),
            offline_seconds=stats([r['offline_seconds'] for r in warm])),
        label_kinds=dict(sorted(_kinds(roots).items())), examples=examples, devices=[str(d) for d in jax.devices()], jax=jax.__version__)
    with out_path.open('x') as f:json.dump(dict(report, roots=roots), f, indent=1, default=repr)
    print(json.dumps(report, default=repr))
    ok = totals['lanes'] > 0 and not (totals['mismatched_lanes'] or totals['missing_disagrees'] or totals['legal_mismatched_worlds'])
    return 0 if ok else 1


def _kinds(roots):
    out = {}
    for r in roots:
        for row in r.get('labels') or ():
            for x in row:
                k = 'none' if x is None else 'switch:?' if x == 'switch:?' else 'move:struggle' if x == 'move:struggle' else x.split(':')[0]
                out[k] = out.get(k, 0) + 1
    return out


if __name__ == '__main__':
    sys.exit(main())
