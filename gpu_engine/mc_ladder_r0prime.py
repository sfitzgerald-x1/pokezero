"""The MC-truth ladder's R0′ rung: R0 (the real-Showdown depth-1 baseline at each gate-bank root) with one change, the estimator.

R0 values each child (action a, reply b, draw c) by sampling both seats' policies until our seat is requested again, then the
champion's value head there. The engine rungs (R1-R4) value the state right after the joint turn. R0′ does that on Showdown:
mc_root_labeller.depth1_baseline(post_turn=True), with R0's actions, replies (pi_opp > 0.001 from the bank row's root policy),
draws and draw seeds. So R0 -> R0′ isolates the estimator and R0′ -> R1 the engine.

Per bank row: the root is replayed exactly as the labeller found it (foulplay_root on the collection game, self_root from the
battle seed), on the labeller's CachingEnv; the replayed root's legal actions must equal the row's. With --check-r0 N, the first
N roots of this worker also recompute R0 (post_turn=False) and compare its completed_q with the bank's (the bank's root policy is
stored at 6 decimals, so weights may differ in the 7th place).

Output: OUT/r0prime-<worker>.jsonl, one mc_ladder_score rung row per root (kind, ordinal or seed + ours, target, action: the
argmax of the post-turn completed_q, the legal actions echoed, completed_q, child_values), and OUT/r0prime-<worker>.summary.json.

usage: python -m gpu_engine.mc_ladder_r0prime BANK_DIR OUT_DIR --collection DIR --checkpoint CHAMPION.pt [--draws 8]
           [--worker W --workers N] [--check-r0 N]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np


def bank_rows(directory):
    rows = [json.loads(l) for p in sorted(glob.glob(str(Path(directory) / 'roots-*.jsonl'))) for l in open(p) if l.strip()]
    return sorted(rows, key=lambda r: (r['kind'], r.get('ordinal', -1), r['seed'], r['ours'], r['target']))


def replay_root(env, row, champion, collection, fpd):
    """Replay a bank row's root on env; True when the root's legal actions are the row's."""
    from gpu_engine import mc_root_labeller as L
    if row['kind'] == 'foulplay':
        game, why = fpd.load_game(collection, str(Path(collection) / 'results' / f'{row["ordinal"]}.json'))
        if game is None:raise ValueError(f'game {row["ordinal"]} not loaded: {why}')
        ok = L.foulplay_root(env, game, row['target'], fpd)
    else:ok = L.self_root(env, champion, row['seed'], row['ours'], row['target'])
    return bool(ok) and [int(a) for a in np.flatnonzero(L.legal_of(env.observe(row['ours'])))] == row['actions']


def label(env, snapshot, row, champion, draws, post_turn):
    """depth1_baseline at the bank row's root from `snapshot` (taken right after the replay: every child restores it)."""
    from gpu_engine import mc_root_labeller as L
    ours = row['ours']
    root = dict(ours=ours, opp={'p1': 'p2', 'p2': 'p1'}[ours], actions=row['actions'], root_seed=row['root_seed'], opp_policy=row.get('opp_policy'))
    return L.depth1_baseline(env, snapshot, root, dict(ours=champion, opp=champion), draws, post_turn=post_turn)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bank');parser.add_argument('out');parser.add_argument('--collection', required=True)
    parser.add_argument('--checkpoint', required=True);parser.add_argument('--draws', type=int, default=8)
    parser.add_argument('--worker', type=int, default=0);parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--check-r0', type=int, default=0)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    from gpu_engine import fp_reply_data as fpd, mc_root_labeller as L
    rows = bank_rows(args.bank)[args.worker::args.workers]
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True);path = out / f'r0prime-{args.worker}.jsonl'
    if path.exists() and path.stat().st_size:raise SystemExit(f'{path} already holds rows')
    collection = Path(args.collection);first = sorted(glob.glob(str(collection / 'requests' / '*.json')))[0]
    base = json.loads(Path(first).read_bytes())['base_config'];champion = L.Model(args.checkpoint, torch)
    env = L.make_env(base, champion.result, caching=True);counts = dict(roots=0, written=0, not_replayed=0, errors=0, r0_checked=0, r0_max_abs=0.)
    try:
        with open(path, 'w') as stream:
            for i, row in enumerate(rows):
                counts['roots'] += 1
                try:
                    if not replay_root(env, row, champion, collection, fpd):counts['not_replayed'] += 1;continue
                    snapshot = env.snapshot()   # the root, once: both estimators restore it for every child
                    b = label(env, snapshot, row, champion, args.draws, post_turn=True)
                    if i < args.check_r0:   # R0 again on the same replay: the bank's completed_q, to the 6-decimal policy
                        b0 = label(env, snapshot, row, champion, args.draws, post_turn=False);counts['r0_checked'] += 1
                        counts['r0_max_abs'] = max(counts['r0_max_abs'], float(np.max(np.abs(np.asarray(b0['completed_q']) - np.asarray(row['baseline']['completed_q'])))))
                except Exception as error:
                    counts['errors'] += 1;print(json.dumps(dict(error=f'{type(error).__name__}: {error}'[:300], kind=row['kind'], seed=row['seed'])), flush=True);continue
                q = b['completed_q'];action = int(row['actions'][int(np.argmax(q))])
                identity = dict(ordinal=row['ordinal']) if row['kind'] == 'foulplay' else dict(seed=row['seed'], ours=row['ours'])
                stream.write(json.dumps(dict(kind=row['kind'], **identity, target=row['target'], action=action, actions=row['actions'],
                                             completed_q=q, child_values=b['child_values'])) + '\n');stream.flush()
                counts['written'] += 1
    finally:
        env.close()
    (out / f'r0prime-{args.worker}.summary.json').write_text(json.dumps(dict(counts, draws=args.draws, worker=args.worker, workers=args.workers)) + '\n')
    print(json.dumps(dict(event='DONE', **counts)), flush=True)
    return 0 if counts['written'] and not counts['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
