"""The MC-truth ladder's R0′ rung: R0 (the real-Showdown depth-1 baseline at each gate-bank root) with one change, the estimator.

R0 values each child (action a, reply b, draw c) by sampling both seats' policies until our seat is requested again, then the
champion's value head there. The engine rungs (R1-R4) value the state right after the joint turn. R0′ does that on Showdown:
mc_root_labeller.depth1_baseline(post_turn=True), with R0's actions, replies (pi_opp > 0.001 from the bank row's root policy),
draws (the bank row's own count) and draw seeds; a root whose R0′ reply set differs from the bank's R0 is refused
(reply_mismatch: the stored policy's 6-decimal rounding can move a reply across the 0.001 cut). So R0 -> R0′ isolates the
estimator and R0′ -> R1 the engine.

Per bank row: the root is replayed exactly as the labeller found it (foulplay_root on the collection game, self_root from the
battle seed), on the labeller's CachingEnv; the replayed root's legal actions must equal the row's, and the champion's policy and
value there the row's policy and root_value (stored at 6 decimals; within ROOT_TOLERANCE), or the root is not_replayed /
root_mismatch. With --check-r0 N, the first N roots of this worker also recompute R0 (post_turn=False) and compare its
completed_q with the bank's (the bank's root policy is stored at 6 decimals, so weights may differ in the 7th place).

Output: OUT/r0prime-<worker>.jsonl, one mc_ladder_score rung row per root (kind, ordinal or seed + ours, target, action: the
argmax of the post-turn completed_q, the legal actions echoed, completed_q, child_values), and OUT/r0prime-<worker>.summary.json.
Exit 0 only when every root of the worker was written and the R0 re-check (if any) stayed within R0_TOLERANCE; 1 otherwise.

usage: python -m gpu_engine.mc_ladder_r0prime BANK_DIR OUT_DIR --collection DIR --checkpoint CHAMPION.pt
           [--worker W --workers N] [--check-r0 N]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT_TOLERANCE = 1e-5   # the replayed root's policy and value against the bank's 6-decimal copies
R0_TOLERANCE = 1e-5     # the re-checked R0 completed_q against the bank's


def bank_draws(row):
    """The bank row's depth-1 draw count (every child's), or ValueError."""
    counts = {len(v) for v in row['baseline']['child_values'].values()}
    if len(counts) != 1 or not min(counts):raise ValueError(f'the bank row has draw counts {sorted(counts)}')
    return counts.pop()


def verdict(counts):
    """The exit code of a worker's counts (module docstring)."""
    return 0 if counts['written'] and counts['written'] == counts['roots'] and counts['r0_max_abs'] <= R0_TOLERANCE else 1


def bank_rows(directory):
    rows = [json.loads(l) for p in sorted(glob.glob(str(Path(directory) / 'roots-*.jsonl'))) for l in open(p) if l.strip()]
    return sorted(rows, key=lambda r: (r['kind'], r.get('ordinal', -1), r['seed'], r['ours'], r['target']))


def replay_root(env, row, champion, collection, fpd):
    """Replay a bank row's root on env: None when it is the row's root, else why not (not_replayed: the game ends first or the legal
    actions differ; root_mismatch: the champion's policy or value there is not the row's)."""
    from gpu_engine import mc_root_labeller as L
    if row['kind'] == 'foulplay':
        game, why = fpd.load_game(collection, str(Path(collection) / 'results' / f'{row["ordinal"]}.json'))
        if game is None:raise ValueError(f'game {row["ordinal"]} not loaded: {why}')
        ok = L.foulplay_root(env, game, row['target'], fpd)
    else:ok = L.self_root(env, champion, row['seed'], row['ours'], row['target'])
    if not ok or [int(a) for a in np.flatnonzero(L.legal_of(env.observe(row['ours'])))] != row['actions']:return 'not_replayed'
    probs, value = champion.evaluate([env.observe(row['ours'])])
    if (np.max(np.abs(np.asarray(probs[0], float) - np.asarray(row['policy'], float))) > ROOT_TOLERANCE
            or abs(float(value[0]) - row['root_value']) > ROOT_TOLERANCE):return 'root_mismatch'
    return None


def label(env, snapshot, row, champion, post_turn):
    """depth1_baseline at the bank row's root from `snapshot` (taken right after the replay: every child restores it)."""
    from gpu_engine import mc_root_labeller as L
    ours = row['ours']
    root = dict(ours=ours, opp={'p1': 'p2', 'p2': 'p1'}[ours], actions=row['actions'], root_seed=row['root_seed'], opp_policy=row.get('opp_policy'))
    return L.depth1_baseline(env, snapshot, root, dict(ours=champion, opp=champion), bank_draws(row), post_turn=post_turn)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bank');parser.add_argument('out');parser.add_argument('--collection', required=True)
    parser.add_argument('--checkpoint', required=True)
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
    env = L.make_env(base, champion.result, caching=True);counts = dict(roots=0, written=0, not_replayed=0, root_mismatch=0, reply_mismatch=0, errors=0, r0_checked=0, r0_max_abs=0.)
    draws = set()
    try:
        with open(path, 'w') as stream:
            for i, row in enumerate(rows):
                counts['roots'] += 1
                try:
                    why = replay_root(env, row, champion, collection, fpd)
                    if why is not None:counts[why] += 1;continue
                    snapshot = env.snapshot()   # the root, once: both estimators restore it for every child
                    b = label(env, snapshot, row, champion, post_turn=True);draws.add(bank_draws(row))
                    if b['replies'] != row['baseline']['replies']:counts['reply_mismatch'] += 1;continue
                    if i < args.check_r0:   # R0 again on the same replay: the bank's completed_q, to the 6-decimal policy
                        b0 = label(env, snapshot, row, champion, post_turn=False);counts['r0_checked'] += 1
                        counts['r0_max_abs'] = max(counts['r0_max_abs'], float(np.max(np.abs(np.asarray(b0['completed_q']) - np.asarray(row['baseline']['completed_q'])))))
                except Exception as error:
                    counts['errors'] += 1;print(json.dumps(dict(error=f'{type(error).__name__}: {error}'[:300], kind=row['kind'], ordinal=row.get('ordinal'), seed=row['seed'],
                                                         ours=row['ours'], target=row['target'])), flush=True);continue
                q = b['completed_q'];action = int(row['actions'][int(np.argmax(q))])
                identity = dict(ordinal=row['ordinal']) if row['kind'] == 'foulplay' else dict(seed=row['seed'], ours=row['ours'])
                stream.write(json.dumps(dict(kind=row['kind'], **identity, target=row['target'], action=action, actions=row['actions'],
                                             completed_q=q, child_values=b['child_values'])) + '\n');stream.flush()
                counts['written'] += 1
    finally:
        env.close()
    (out / f'r0prime-{args.worker}.summary.json').write_text(json.dumps(dict(counts, draws=sorted(draws), worker=args.worker, workers=args.workers)) + '\n')
    print(json.dumps(dict(event='DONE', **counts)), flush=True)
    return verdict(counts)


if __name__ == '__main__':
    raise SystemExit(main())
