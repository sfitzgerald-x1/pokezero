"""Opponent sidecars for the MC ladder's true-world rungs (R1, R2): what scripts/depthn_rr_reprice.py --true-worlds builds the
opponent's TRUE team from at a bank root. CPU only (Showdown replays; the champion in torch for the self roots' draws), no JAX.

Per root: the opponent seat's round-0 observation metadata['self_team'] rows (its true team in Showdown's original party order,
as scripts/depthn_opponent_view_parity.py reads Foul Play's), its literal round-0 request (proves happiness) and its literal
request at the root (true_private's HP / PP overlay; None when it waits there: the re-pricer then falls back to true and records
it). One file per re-pricer cell, OUT/opponent/<kind>-<ordinal>.json, its roots keyed by our decision index (the bank target):
- foulplay: the gate bank's rows (roots-foulplay-*.jsonl: ordinal, target, seed, ours) on the Foul Play collection; the cell
  ordinal is the collection ordinal. The game is replayed by the OV harness's own load_game and replay_rounds (each of our
  requests held to the owner capture, Foul Play's journal action legal in its own view, the whole game). A root's round is its
  capture row's request_count - 1, which must be the round the replay gives that decision (else the root holds an error).
- self: the self-capture manifest (DIR/manifest*.jsonl: ordinal, seed, ours, target); the cell ordinal is the capture ordinal.
  The opponent's round-0 observation and request are read after env.reset(seed); the battle is then replayed by
  mc_root_labeller.self_root (the labeller's env and draws) and the opponent's request read at the root. Our own request there
  must equal the owner capture's (DIR/owning-wire/<ordinal>.jsonl) when the capture exists.
Both kinds use mc_root_labeller.make_env on the labeller's base config (the collection's first numeric request's base_config) and
the champion's model config: foulplay the plain env (as the OV harness replays), self the labeller's caching env.

usage: python -m gpu_engine.mc_ladder_opponents foulplay --bank 'DIR/roots-foulplay-*.jsonl' --collection DIR --checkpoint PT --out OUT
       python -m gpu_engine.mc_ladder_opponents self --captures DIR --collection DIR --checkpoint PT --out OUT
       [--worker W --workers N: every N-th ordinal from the W-th]
Writes OUT/opponent/<kind>-<ordinal>.json (an existing file is kept and counted, so a rerun resumes) and
OUT/opponent-summary-<kind>[-<worker>].json (counts and every failed cell's error). Exit 0 when no cell failed, 1 otherwise.
"""
import argparse
import collections
import glob
import json
import os
from pathlib import Path
import sys

SCHEMA = 'pokezero.mc-ladder-opponent.v1'
KINDS = ('foulplay', 'self')
TEAM = 6
OTHER = {'p1': 'p2', 'p2': 'p1'}
ROOT = Path(__file__).resolve().parents[1]


def _ov():
    """The OV harness (scripts/depthn_opponent_view_parity.py: numpy at import)."""
    if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))
    from scripts import depthn_opponent_view_parity as ov
    return ov


def clone(value):
    return json.loads(json.dumps(value, default=repr))


def capture_rows(path):
    """The owner capture's ROOT rows (pokezero.phase_d_owning_capture.read_capture)."""
    from pokezero import phase_d_owning_capture as owning
    return list(owning.read_capture(path))


def document(kind, ordinal, seed, ours, team_rows, round0_request, roots, **source):
    """One sidecar: the opponent's six round-0 self_team rows, its round-0 request and the roots {decision index: entry}."""
    rows = clone(list(team_rows or ()))
    if len(rows) != TEAM:raise ValueError(f'the opponent\'s round-0 observation has {len(rows)} self_team rows')
    if not isinstance(round0_request, dict):raise ValueError('the opponent has no round-0 request')
    return dict(schema=SCHEMA, kind=kind, ordinal=int(ordinal), seed=int(seed), ours=ours, opponent=OTHER[ours], team_rows=rows,
        round0_request=clone(round0_request), roots={str(k): v for k, v in sorted(roots.items())}, source=source)


# Foul Play roots --------------------------------------------------------------------------------------------------------------------
def bank_roots(paths):
    """{ordinal: dict(seed, ours, targets)} from the gate bank's foulplay rows; one ordinal's rows must agree on seed and ours."""
    out = {}
    for path in paths:
        for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
            if not raw.strip():continue
            row = json.loads(raw);ordinal = int(row['ordinal'])
            entry = out.setdefault(ordinal, dict(seed=int(row['seed']), ours=row['ours'], targets=set()))
            if (entry['seed'], entry['ours']) != (int(row['seed']), row['ours']):raise ValueError(f'{path}:{number}: cell {ordinal}\'s seed or seat differs between rows')
            entry['targets'].add(int(row['target']))
    return {o: dict(e, targets=sorted(e['targets'])) for o, e in sorted(out.items())}


def foulplay_sidecar(env, collection, ordinal, entry):
    """One Foul Play game's sidecar (module docstring). The OV harness's CellError / Mismatch when the game cannot be replayed."""
    ov = _ov();collection = Path(collection)
    game = ov.load_game(collection, dict(ordinal=ordinal, seed=entry['seed']), reference='fp-wire-b')   # no screen-band exclusion here
    if game['ours'] != entry['ours']:raise ov.CellError('seat_differs', f'bank {entry["ours"]}, game {game["ours"]}')
    captured = capture_rows(collection / 'owning-wire' / f'{ordinal}.jsonl')
    if any(r['owner'] != game['ours'] for r in captured):raise ov.CellError('capture_owner')
    table, team = ov.replay_rounds(env, game, {r['decision_index']: r['own_request_line'] for r in captured})
    by_index = {r['decision_index']: r for r in captured};roots = {}
    for target in entry['targets']:
        row = by_index.get(target)
        if row is None:roots[target] = dict(decision_index=target, error='no owner capture row');continue
        round_ = int(row['request_count']) - 1;found = table.get(round_)
        if found is None or found['our_decision'] != target:
            roots[target] = dict(decision_index=target, round=round_, error=f'the replay gives round {round_} to decision '
                f'{None if found is None else found["our_decision"]}');continue
        roots[target] = dict(decision_index=target, round=round_, turn=found['turn'], opponent_requested=found['fp_request'] is not None,
            root_request=found['fp_request'])
    return document('foulplay', ordinal, game['seed'], game['ours'], team, table[0]['fp_request'], roots, collection=str(collection))


# Self roots -----------------------------------------------------------------------------------------------------------------------
def manifest_rows(captures):
    """The self captures' manifest rows, in ordinal order: the merged manifest.jsonl when it exists, else every worker's
    manifest-<w>.jsonl (a sharded capture run)."""
    merged, workers = Path(captures) / 'manifest.jsonl', sorted(Path(captures).glob('manifest-*.jsonl'))
    read = lambda paths: [json.loads(x) for p in paths for x in Path(p).read_text().splitlines() if x.strip()]
    rows = read([merged]) if merged.exists() else read(workers)
    ordinals = [int(r['ordinal']) for r in rows]
    if len(set(ordinals)) != len(ordinals):raise ValueError(f'{captures}: a capture ordinal appears twice in the manifests')
    if merged.exists() and workers and sorted(ordinals) != sorted(int(r['ordinal']) for r in read(workers)):
        raise ValueError(f'{captures}: the merged manifest is not the union of the workers\' manifests')
    if not rows:raise ValueError(f'{captures}: no manifest rows')
    return sorted(rows, key=lambda r: int(r['ordinal']))


def self_sidecar(env, champion, row, captures=None):
    """One self root's sidecar (module docstring). ValueError when the replay reaches no root or differs from the capture."""
    from gpu_engine.mc_root_labeller import self_root
    seed, ours, target, ordinal = int(row['seed']), row['ours'], int(row['target']), int(row['ordinal']);opp = OTHER[ours]
    env.reset(seed=seed)
    if opp not in env.requested_players():raise ValueError(f'the opponent {opp} is not requested in round 0')
    team = (env.observe(opp).metadata or {}).get('self_team');round0 = clone(env._latest_requests.get(opp))
    if not self_root(env, champion, seed, ours, target):raise ValueError(f'decision {target} is no root (the battle ends first or one legal action)')
    path = None if captures is None else Path(captures) / 'owning-wire' / f'{ordinal}.jsonl'
    checked = path is not None and path.exists()
    if checked:
        mine = {r['decision_index']: r for r in capture_rows(path)}.get(target);ov = _ov()
        if mine is None or mine['owner'] != ours or ov.request_body(mine['own_request_line']) != ov.request_body(json.dumps(env._latest_requests.get(ours))):
            raise ValueError(f'our request at decision {target} differs from the owner capture\'s')
    requested = opp in env.requested_players()
    roots = {target: dict(decision_index=target, round=None, opponent_requested=requested,
        root_request=clone(env._latest_requests.get(opp)) if requested else None)}
    return document('self', ordinal, seed, ours, team, round0, roots, manifest_status=row.get('status'), capture_checked=checked)


# Main -------------------------------------------------------------------------------------------------------------------------------
def labeller_base(collection):
    """The labeller's env base config: requests/<first numeric result ordinal>.json's base_config (mc_root_labeller.main)."""
    stems = sorted(int(p.stem) for p in (Path(collection) / 'results').glob('*.json') if p.stem.isdigit())
    if not stems:raise ValueError(f'{collection}: no results/<ordinal>.json')
    return json.loads((Path(collection) / 'requests' / f'{stems[0]}.json').read_bytes())['base_config']


def write_all(kind, items, out, make_env, build):
    """Writes each item's sidecar to OUT/opponent/<kind>-<ordinal>.json (an existing one is kept and counted). items: (ordinal,
    item); build(env, ordinal, item) -> document. A failing item is counted with its error and the env rebuilt (it may be
    wedged). Returns the summary."""
    folder = Path(out) / 'opponent';folder.mkdir(parents=True, exist_ok=True)
    counts, errors, env = collections.Counter(), [], None
    try:
        for ordinal, item in items:
            path = folder / f'{kind}-{ordinal}.json'
            if path.exists():   # resume: a complete earlier file is kept (written atomically, so it parses or is absent)
                json.loads(path.read_text());counts['existing'] += 1;continue
            if env is None:env = make_env()
            try:doc = build(env, ordinal, item)
            except Exception as error:
                errors.append(dict(ordinal=ordinal, error=f'{type(error).__name__}: {error}'[:400]));counts['failed'] += 1
                try:env.close()
                finally:env = None
                continue
            partial = path.with_suffix('.json.partial')   # atomic: written, fsynced, then renamed into place
            with partial.open('w') as stream:stream.write(json.dumps(doc, sort_keys=True) + '\n');stream.flush();os.fsync(stream.fileno())
            os.replace(partial, path)
            counts['written'] += 1;roots = doc['roots'].values()
            counts['roots'] += len(roots);counts['root_errors'] += sum('error' in r for r in roots)
            counts['opponent_waits'] += sum('error' not in r and not r['opponent_requested'] for r in roots)
            print(json.dumps(dict(kind=kind, ordinal=ordinal, roots=len(roots))), flush=True)
    finally:
        if env is not None:env.close()
    return dict(schema=SCHEMA + '.summary', kind=kind, cells=len(items), **counts, errors=errors)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('kind', choices=KINDS)
    p.add_argument('--bank', help='foulplay: the gate bank\'s roots-foulplay-*.jsonl (a glob)')
    p.add_argument('--captures', help='self: the self-capture directory (manifest*.jsonl, owning-wire/)')
    p.add_argument('--collection', required=True, help='the Foul Play collection (foulplay: its games; both: the labeller\'s base config)')
    p.add_argument('--checkpoint', required=True, help='the champion\'s .pt (the env\'s observation config; the self roots\' draws)')
    p.add_argument('--out', required=True);p.add_argument('--worker', type=int, default=0);p.add_argument('--workers', type=int, default=1)
    args = p.parse_args(argv)
    if not 0 <= args.worker < args.workers:p.error('--worker W --workers N with 0 <= W < N required')
    if (args.kind == 'foulplay') != bool(args.bank) or (args.kind == 'self') != bool(args.captures):p.error('foulplay takes --bank, self takes --captures')
    if args.kind == 'foulplay':
        paths = sorted(glob.glob(args.bank))
        if not paths:p.error(f'--bank {args.bank}: no file')
        items = list(bank_roots(paths).items())
    else:items = [(int(r['ordinal']), r) for r in manifest_rows(args.captures) if r.get('status') == 'ok']   # only captures whose root is the bank's
    items = items[args.worker::args.workers]
    import torch
    torch.set_num_threads(1)
    from gpu_engine import mc_root_labeller as labeller
    champion = labeller.Model(args.checkpoint, torch);base = labeller_base(args.collection)
    make_env = lambda: labeller.make_env(base, champion.result, caching=args.kind == 'self')
    if args.kind == 'foulplay':build = lambda env, ordinal, entry: foulplay_sidecar(env, args.collection, ordinal, entry)
    else:build = lambda env, ordinal, row: self_sidecar(env, champion, row, args.captures)
    summary = write_all(args.kind, items, args.out, make_env, build)
    summary.update(bank=args.bank, captures=args.captures, collection=args.collection, checkpoint=args.checkpoint, worker=args.worker, workers=args.workers)
    suffix = '' if args.workers == 1 else f'-{args.worker}'
    (Path(args.out) / f'opponent-summary-{args.kind}{suffix}.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'errors'}), flush=True)
    return 0 if not summary['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
