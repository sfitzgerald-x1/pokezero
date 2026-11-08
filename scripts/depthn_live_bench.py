#!/usr/bin/env python3
"""Gap 4B: the live search path's warm per-decision cost and coverage on Codex's frozen 200-root Showdown rosters.

Design. One process, one GPU, one W (the rung). It builds the live server's own DepthNLiveSearch (depthn_live_server.build,
the same flags, at bucket 1) and serves nothing: it warms every program up on the first --warmup-count roster roots
(DepthNLiveSearch.warmup, which resets the constructor's sessions and attempts after itself), then hands every roster root
to process_many ALONE, one after another, and records the response. So each row is one decision at capacity 1 with no
queue: Codex's ProtocolConstructor.prepare, the build, one DepthN price call and the reduction (search_hedge.averaging's
action), exactly the server's path; nothing is re-implemented here.
- Rosters (--roster PATH SHA256, repeatable; --producer-review PATH SHA256 optional): Codex's pinned 200-root rosters, read as
  scripts/phase_d_constructor_coverage.py reads them (phase_d_ladder_panel.read_bound; each selected row's capture_row
  found in its capture_path through phase_d_owning_capture.read_capture, its address checked). A root whose capture
  is unavailable is recorded UNAVAILABLE and never priced. Requests carry the eight address keys, a fresh request_id, the
  rung and a budget large enough never to expire (BUDGET).
- --saved-root-admission (default here) folds every prefix from scratch, as Codex's coverage tool: roots of one battle
  never share a session.
- Output (--out, a new directory): records.jsonl (one row per root, fsynced: roster, ordinal, address, status, cause,
  refusal, world refusal kinds, timings, decision and wall seconds, action, mean_q_agrees, root classes, structural),
  warmup.json, summary.json and summary.md: per roster and pooled, the searched share (COMPLETE / priced roots), the
  causes and constructor refusal kinds, and p50 / p95 / p99 / max of each stage and of the wall over COMPLETE roots; the
  lead's W=16 condition (p95 wall <= 2.6 s) is reported as a flag, never decided here.
- Root classes (the live search's root_classes: forced_switch, fresh_substitute, wish, sleep): the same breakdown per class,
  per roster and pooled (roots, priced, COMPLETE, searched share, causes, the stage table). A root in several classes counts
  in each, one in none counts as `other`, one the search could not classify (or never classified) as `unclassified`.
  Structural failures (cause server_error, or a WORLD_STRUCTURE_MISMATCH refusal) are counted apart from clean refusals,
  and `structural_classes` lists the classes with any (pooled): the lead's input on --refuse-classes, which the bench passes
  through to the search (as the server does) and never decides.
Informational: no verdict, no bound. Exit 0 when any root priced, 2 when none did.
"""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))
from scripts import depthn_live_server as server   # noqa: E402

BUDGET = 600.
TIMINGS = ('prepare_seconds', 'classify_seconds', 'build_seconds', 'price_seconds', 'reduce_seconds')   # the response's own stages
STAGES = TIMINGS + ('decision_seconds', 'wall_seconds')
GROUPS = server.live.CLASSES + ('other', 'unclassified')
W16_P95_SECONDS = 2.6
KEYS = ('battle_id', 'owner', 'stream_id', 'decision_index', 'generation', 'prefix_sha256', 'lines', 'own_request_line')


def parser():
    p = server.parser();p.prog = 'depthn_live_bench.py'
    for action in p._actions:   # the server's listener flags are unused here
        if action.dest in ('port', 'warmup_requests'):action.required = False
    p.add_argument('--roster', nargs=2, action='append', required=True, metavar=('PATH', 'SHA256'),
        help="one of Codex's frozen 200-root rosters and its sha256 (repeatable)")
    p.add_argument('--producer-review', nargs=2, metavar=('PATH', 'SHA256'), help="the rosters' producer review and its sha256 (checked when given)")
    p.add_argument('--warmup-count', type=int, default=8, help='roster roots used for the warmup (default 8)')
    p.add_argument('--limit', type=int, help='at most this many roots per roster (default all)')
    p.add_argument('--out', required=True, help='a new output directory')
    p.set_defaults(saved_root_admission=True, buckets=(1,))
    return p


def options(argv=None):
    p = parser();args = p.parse_args(argv);server.evaluators.check_arguments(p, args)
    if not args.champion_archive:p.error('--champion-archive with its two sha256 flags is required: the champion prices every root and leaf')
    if args.buckets != (1,):p.error('the bench prices at bucket 1 only (one decision per call)')
    if args.warmup_count < 1:p.error('--warmup-count must be >= 1')
    if args.limit is not None and args.limit < 1:p.error('--limit must be >= 1')
    args.port = args.port or 0;args.max_requests = 1
    return args


def roster_requests(path, digest, *, limit=None):
    """[(ordinal, address, request or None, problem)] for one pinned roster, as Codex's coverage tool resolves it."""
    from pokezero import phase_d_owning_capture as capture, phase_d_ladder_admission as admission
    from pokezero.phase_d_ladder_panel import read_bound
    selected = read_bound(path, digest)['selected']
    if not isinstance(selected, list) or not selected:raise ValueError(f'{path}: a roster with selected roots required')
    out = []
    for ordinal, meta in enumerate(selected[:limit]):
        address = admission.address(meta)
        try:
            found = next((row for row in capture.read_capture(meta['capture_path']) if row['capture_row'] == meta['capture_row']), None)
            if found is None or admission.address(found) != address:raise ValueError('same original captured root unavailable')
            out.append((ordinal, address, {k: found[k] for k in KEYS}, None))
        except (ValueError, TypeError, KeyError, OSError) as error:
            if isinstance(error, PermissionError):raise
            out.append((ordinal, address, None, f'{type(error).__name__}: {error}'))
    return out


def percentiles(values):
    """p50 / p95 / p99 / max (nearest rank) of finite values, or None when there are none."""
    xs = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not xs:return None
    rank = lambda q: xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]
    return dict(n=len(xs), p50=rank(.5), p95=rank(.95), p99=rank(.99), max=xs[-1])


def structural(response):
    """A structural failure, apart from clean refusals: a server_error, or a root refused for its world carry structure."""
    return response.get('cause') == 'server_error' or str(response.get('refusal') or '').startswith('WORLD_STRUCTURE_MISMATCH')


def groups(row):
    """The class groups a record row counts in: its root classes, else 'other'; 'unclassified' when the search gave none."""
    classes = row.get('root_classes')
    return ['unclassified'] if classes is None else list(classes) or ['other']


def by_class(priced):
    """Per class group (GROUPS order) over priced rows: roots, priced, COMPLETE, searched share, causes, structural and clean
    refusals, and the stage table over COMPLETE roots."""
    out = {}
    for group in GROUPS:
        mine = [r for r in priced if group in groups(r)];complete = [r for r in mine if r['status'] == 'COMPLETE']
        refused = [r for r in mine if r['status'] != 'COMPLETE']
        out[group] = dict(roots=len(mine), priced=len(mine), complete=len(complete), searched_share=None if not mine else len(complete) / len(mine),
            causes=dict(collections.Counter(r['cause'] for r in refused)), structural=sum(r['structural'] for r in refused),
            clean_refusals=sum(not r['structural'] for r in refused), stages={s: percentiles(r[s] for r in complete) for s in STAGES})
    return out


def summarize(rows, W):
    """The summary of record rows (one roster or pooled)."""
    priced = [r for r in rows if r['status'] != 'UNAVAILABLE'];complete = [r for r in priced if r['status'] == 'COMPLETE']
    stages = {s: percentiles(r[s] for r in complete) for s in STAGES}
    wall = stages['wall_seconds'];classes = by_class(priced)
    return dict(roots=len(rows), unavailable=len(rows) - len(priced), priced=len(priced), complete=len(complete),
        searched_share=None if not priced else len(complete) / len(priced), classes=classes,
        structural=sum(r['structural'] for r in priced), structural_classes=[g for g in GROUPS if classes[g]['structural']],
        causes=dict(collections.Counter(r['cause'] for r in priced if r['status'] != 'COMPLETE')),
        world_refusal_kinds=dict(collections.Counter(k for r in priced for k in r['world_refusal_kinds'])),
        mean_q_disagreements=sum(r['mean_q_agrees'] is False for r in complete), stages=stages,
        w16_condition=None if W != 16 or wall is None else dict(p95_wall_seconds=wall['p95'], bound=W16_P95_SECONDS, met=wall['p95'] <= W16_P95_SECONDS))


def record(roster, ordinal, address, response, wall):
    """One records.jsonl row from a process_many response (or None for an unavailable root)."""
    if response is None:
        return dict(roster=roster, ordinal=ordinal, address=list(address), status='UNAVAILABLE', cause=None, refusal=wall, world_refusal_kinds=[],
            **dict.fromkeys(STAGES), action_index=None, mean_q_agrees=None, constructed_worlds=0, root_classes=None, root_class=None,
            root_class_source=None, structural=False)
    timings = response.get('timings') or {}
    kinds = sorted({k for w in (response.get('world_refusals') or ()) for k in (w.get('kinds') or ())})
    return dict(roster=roster, ordinal=ordinal, address=list(address), status=response['status'], cause=response.get('cause'),
        refusal=None if response.get('refusal') is None else str(response['refusal'])[:500], world_refusal_kinds=kinds,
        **{s: timings.get(s) for s in TIMINGS}, decision_seconds=response.get('decision_seconds'), wall_seconds=wall,
        action_index=response.get('action_index'), mean_q_agrees=response.get('mean_q_agrees'), constructed_worlds=response.get('constructed_worlds'),
        root_classes=response.get('root_classes'), root_class=response.get('root_class'), root_class_source=response.get('root_class_source'),
        structural=structural(response))


def stage_table(stages, title):
    lines = [f'| {title} | n | p50 | p95 | p99 | max |', '| --- | --- | --- | --- | --- | --- |']
    for stage, p in stages.items():
        lines.append(f'| {stage} | - | - | - | - | - |' if p is None else f"| {stage} | {p['n']} | {p['p50']:.3f} | {p['p95']:.3f} | {p['p99']:.3f} | {p['max']:.3f} |")
    return lines


def class_lines(classes):
    """The per-class section: one row per class group, then each populated group's stage table over its COMPLETE roots."""
    lines = ['| Class | roots | priced | COMPLETE | searched share | structural | clean refusals | causes |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for group, c in classes.items():
        share = '-' if c['searched_share'] is None else f"{c['searched_share']:.3f}"
        lines.append(f"| {group} | {c['roots']} | {c['priced']} | {c['complete']} | {share} | {c['structural']} | {c['clean_refusals']} | {c['causes']} |")
    for group, c in classes.items():
        if c['roots']:lines += ['', *stage_table(c['stages'], f'{group}: stage (COMPLETE roots)')]
    return lines


def markdown(summary):
    lines = [f"# Gap 4B: live search cost and coverage, {summary['rung']} (W={summary['W']}), bucket 1", '',
        f"- Tree `{summary['tree_commit']}`; warmup {summary['warmup_seconds']:.1f} s on {summary['warmup_count']} roots; saved-root admission {summary['saved_root_admission']}", '']
    for name, s in [*summary['rosters'].items(), ('pooled', summary['pooled'])]:
        share = '-' if s['searched_share'] is None else f"{s['searched_share']:.3f}"
        lines += [f"## {name}", '', f"- roots {s['roots']}, unavailable {s['unavailable']}, priced {s['priced']}, COMPLETE {s['complete']} (searched share {share})",
            f"- causes {s['causes']}; world refusal kinds {s['world_refusal_kinds']}; action_mean_q disagreements {s['mean_q_disagreements']}"]
        if s['w16_condition'] is not None:lines.append(f"- W=16 condition: p95 wall {s['w16_condition']['p95_wall_seconds']:.3f} s against {W16_P95_SECONDS} s: met {s['w16_condition']['met']}")
        lines += [f"- structural failures {s['structural']}; structural classes {s['structural_classes']}", '', *stage_table(s['stages'], 'Stage (COMPLETE roots)'),
            '', f'### {name}: by root class', '', *class_lines(s['classes']), '']
    lines += [f"structural_classes (pooled): {summary['structural_classes']}", '']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    args = options(argv);out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.producer_review:
        from pokezero.phase_d_ladder_panel import read_bound
        read_bound(*args.producer_review)
    rosters = {Path(path).name: roster_requests(path, digest, limit=args.limit) for path, digest in args.roster}
    available = [req for items in rosters.values() for _, _, req, _ in items if req is not None]
    if not available:raise SystemExit('no roster root is available')
    search = server.build(args);header = server.header(args, search)
    started = time.perf_counter();warm = search.warmup(available[:args.warmup_count]);warm_seconds = time.perf_counter() - started
    (out / 'warmup.json').write_text(json.dumps(dict(header=header, warmup=warm, seconds=warm_seconds), indent=1, sort_keys=True, default=repr) + '\n')
    rows = []
    with (out / 'records.jsonl').open('x') as stream:
        for name, items in rosters.items():
            for ordinal, address, request, problem in items:
                if request is None:row = record(name, ordinal, address, None, problem)
                else:
                    body = dict(request, schema=server.original.SCHEMA, request_id=f'bench-{name}-{ordinal}-{os.getpid()}', rung=search.rung, budget_seconds=BUDGET)
                    clock = time.perf_counter();response = search.process_many([body])[0];row = record(name, ordinal, address, response, time.perf_counter() - clock)
                rows.append(row);stream.write(json.dumps(row, sort_keys=True, default=repr) + '\n');stream.flush();os.fsync(stream.fileno())
                print(json.dumps(dict(roster=name, ordinal=ordinal, status=row['status'], cause=row['cause'], wall=row['wall_seconds'])), flush=True)
    W = search.W
    summary = dict(header=header, rung=search.rung, W=W, tree_commit=search.tree_commit, saved_root_admission=args.saved_root_admission,
        warmup_count=args.warmup_count, warmup_seconds=warm_seconds, rosters={n: summarize([r for r in rows if r['roster'] == n], W) for n in rosters},
        pooled=summarize(rows, W), roster_sha256={Path(p).name: d for p, d in args.roster}, refuse_classes=list(args.refuse_classes),
        scope='INFORMATIONAL: warm per-decision cost at bucket 1 with no queue, and coverage; no verdict')
    summary['structural_classes'] = summary['pooled']['structural_classes']
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True, default=repr) + '\n')
    (out / 'summary.md').write_text(markdown(summary))
    print(json.dumps(dict(structural_classes=summary['structural_classes'])), flush=True)
    print(json.dumps(dict(done=str(out), pooled=dict((k, summary['pooled'][k]) for k in ('roots', 'priced', 'complete', 'searched_share')))), flush=True)
    return 0 if summary['pooled']['complete'] else 2


if __name__ == '__main__':raise SystemExit(main())
