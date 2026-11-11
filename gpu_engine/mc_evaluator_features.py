"""Model A's features: the champion's frozen encoder tokens for each labelled root and its depth-1 children, with Monte Carlo targets.

Per bank row (mc_root_labeller's roots, any K): the root is replayed exactly as the labeller found it (mc_ladder_r0prime.replay_root:
the legal actions and the champion's policy and value at the root must be the row's). Then every (a, b) is expanded: a over the
row's actions, b over the opponent root policy's support (opp_policy > 0.001; a waiting opponent gives the single reply None), and
C chance draws per (a, b) from a seed bank disjoint from the rollouts' and the U1 baseline's (root_seed, 8_000_000 + c). A child is
our seat's view right after the joint turn (R0' and the engine rungs' estimator state), observed whether or not our seat is
requested; a terminal child keeps its outcome and has no tokens.

Targets, from the row: Q_MC(a) and its standard error (capped rollouts censored), the CRN-paired standard error of every action
pair (se_pair: the rollouts share their seed and root reply across actions), and for each (a, b) the outcomes of the rollouts
whose sampled root reply was b. The rollouts' replies are recomputed exactly as the labeller drew them
(sample(opp_policy, the opponent's legal mask at the root, draw(root_seed, k, 0, 1))), from the row's 6-decimal opp_policy; the
index records how many rollouts each (a, b) has.

The champion's own value head output is kept per observation (old_value: the comparator on identical children). The tokens
come from the production forward through one encoder hook (value_precision_probe.frozen_token_features' method): float16
[n, tokens, width] with the boolean attention mask [n, tokens], the root first, then the children in (a, b, c) order.

Output per worker: OUT/features-<worker>/{tokens,valid,owner,old_value}.npy (owner: the index row of each observation) and OUT/index-<worker>.jsonl (one row per
root: identity, actions, replies with their prior, Q_MC, SE, per-(a, b) outcomes, the children's (a, b, c, terminal outcome or
None, observation index)), and OUT/summary-<worker>.json. Exit 0 only when every root of the worker was written.

usage: python -m gpu_engine.mc_evaluator_features BANK_DIR OUT_DIR --collection DIR --checkpoint CHAMPION.pt [--draws 2]
           [--worker W --workers N] [--kinds foulplay,self] [--limit N]
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

CAPPED = -0.25
DRAW_BANK = 8_000_000   # the children's chance draws; the rollouts use k and the U1 baseline 7_000_000 + c


def censored_q(outcomes):
    """(Q_MC [A], SE [A]) of a row's outcomes [A, K] with capped rollouts left out (None where an action has none / one)."""
    o = np.asarray(outcomes, float);o = np.where(o == CAPPED, np.nan, o);q, se = [], []
    for row in o:
        x = row[~np.isnan(row)]
        q.append(None if not len(x) else float(x.mean()));se.append(None if len(x) < 2 else float(x.std(ddof=1) / math.sqrt(len(x))))
    return q, se


def pair_se(outcomes):
    """[A][A] standard errors of the CRN-paired differences o[i, k] - o[j, k] over rollouts k where neither is capped (None: < 2)."""
    o = np.asarray(outcomes, float);o = np.where(o == CAPPED, np.nan, o);A = len(o);out = [[None] * A for _ in range(A)]
    for i in range(A):
        for j in range(A):
            d = o[i] - o[j];d = d[~np.isnan(d)]
            if i != j and len(d) >= 2:out[i][j] = float(d.std(ddof=1) / math.sqrt(len(d)))
    return out


def reply_outcomes(outcomes, replies, support):
    """{a_index: {b: [outcomes of the rollouts whose reply was b, capped left out]}} for b in support (None: a waiting opponent)."""
    o = np.asarray(outcomes, float);out = {}
    for i, row in enumerate(o):
        out[i] = {b: [float(x) for k, x in enumerate(row) if x != CAPPED and (b is None or replies[k] == b)] for b in support}
    return out


def encoder_tokens(model, torch, observations):
    """(tokens float16 [n, tokens, width], valid bool [n, tokens], the champion's own value [n]) on single observations."""
    from pokezero.neural_policy import observation_windows_to_torch
    t = observation_windows_to_torch([[o] for o in observations], window_size=1, device='cpu');captured = []
    hook = model.encoder.register_forward_hook(lambda _m, _a, output: captured.append(output.detach()))
    try:
        with torch.no_grad():output = model(categorical_ids=t['categorical_ids'], numeric_features=t['numeric_features'], token_type_ids=t['token_type_ids'],
                                            attention_mask=t['attention_mask'], history_mask=t['history_mask'])
    finally:hook.remove()
    if len(captured) != 1:raise ValueError('expected exactly one encoder invocation')
    tokens = captured[0].reshape(len(observations), -1, captured[0].shape[-1])
    valid = t['attention_mask'].reshape(len(observations), -1).bool()
    if valid.shape != tokens.shape[:2]:raise ValueError(f'attention mask {tuple(valid.shape)} vs tokens {tuple(tokens.shape)}')
    return tokens.numpy().astype(np.float16), valid.numpy(), output.value.reshape(-1).numpy().astype(np.float32)


def expand(env, snapshot, row, champion, draws, torch):
    """(index row without the identity, observations) of one replayed root (module docstring)."""
    from gpu_engine import mc_root_labeller as L
    ours = row['ours'];opp = {'p1': 'p2', 'p2': 'p1'}[ours];pi = row.get('opp_policy');K = len(row['outcomes'][0])
    support = [None] if pi is None else [b for b in range(9) if pi[b] > .001]
    replies = None
    if pi is not None:
        legal_opp = L.legal_of(env.observe(opp))
        replies = [L.sample(pi, legal_opp, L.draw(row['root_seed'], k, 0, 1)) for k in range(K)]
    observations = [env.observe(ours)];children = []
    for i, a in enumerate(row['actions']):
        for b in support:
            for c in range(draws):
                env.restore(snapshot);env.reseed_simulator_rng(int(np.random.default_rng([row['root_seed'], DRAW_BANK + c]).integers(1, 2**31 - 1)))
                first = {ours: a}
                if b is not None:first[opp] = b
                env.step(first)
                if env.terminal() is not None:children.append(dict(a=a, b=b, c=c, terminal=L.outcome(env, ours), obs=None));continue
                children.append(dict(a=a, b=b, c=c, terminal=None, obs=len(observations)));observations.append(env.observe(ours))
    q, se = censored_q(row['outcomes'])
    groups = reply_outcomes(row['outcomes'], replies, support)
    index = dict(actions=row['actions'], replies=support, reply_prior=None if pi is None else [float(pi[b]) for b in support], K=K,
                 q_mc=q, se_mc=se, se_pair=pair_se(row['outcomes']), reply_outcomes={str(a): {str(b): groups[i][b] for b in support} for i, a in enumerate(row['actions'])},
                 children=children, root_obs=0, draws=draws)
    return index, observations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bank');parser.add_argument('out');parser.add_argument('--collection', required=True)
    parser.add_argument('--checkpoint', required=True);parser.add_argument('--draws', type=int, default=2)
    parser.add_argument('--worker', type=int, default=0);parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--kinds', default='foulplay,self');parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    from gpu_engine import fp_reply_data as fpd, mc_ladder_r0prime as R, mc_root_labeller as L
    kinds = set(args.kinds.split(','))
    rows = [r for r in R.bank_rows(args.bank) if r['kind'] in kinds][args.worker::args.workers][:args.limit]
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True);index_path = out / f'index-{args.worker}.jsonl'
    if index_path.exists() and index_path.stat().st_size:raise SystemExit(f'{index_path} already holds rows')
    collection = Path(args.collection);first = sorted(glob.glob(str(collection / 'requests' / '*.json')))[0]
    base = json.loads(Path(first).read_bytes())['base_config'];champion = L.Model(args.checkpoint, torch)
    env = L.make_env(base, champion.result, caching=True)
    counts = dict(roots=0, written=0, not_replayed=0, root_mismatch=0, errors=0, observations=0, terminal_children=0)
    tokens, valid, owner, old_value = [], [], [], []
    try:
        with open(index_path, 'w') as stream:
            for row in rows:
                counts['roots'] += 1
                try:
                    why = R.replay_root(env, row, champion, collection, fpd)
                    if why is not None:counts[why] += 1;continue
                    snapshot = env.snapshot();index, observations = expand(env, snapshot, row, champion, args.draws, torch)
                    t, v, old = encoder_tokens(champion.model, torch, observations)
                except Exception as error:
                    counts['errors'] += 1;print(json.dumps(dict(error=f'{type(error).__name__}: {error}'[:300], kind=row['kind'], ordinal=row.get('ordinal'),
                                                                seed=row['seed'], ours=row['ours'], target=row['target'])), flush=True);continue
                base_obs = sum(len(x) for x in owner)
                identity = dict(kind=row['kind'], ordinal=row.get('ordinal'), seed=row['seed'], ours=row['ours'], target=row['target'])
                stream.write(json.dumps(dict(identity, **index, obs_offset=base_obs, obs_count=len(observations))) + '\n');stream.flush()
                tokens.append(t);valid.append(v);old_value.append(old);owner.append(np.full(len(observations), counts['written'], np.int32))
                counts['written'] += 1;counts['observations'] += len(observations)
                counts['terminal_children'] += sum(c['terminal'] is not None for c in index['children'])
    finally:
        env.close()
    if tokens:   # plain .npy arrays (memory-mappable by the trainer)
        shard = out / f'features-{args.worker}';shard.mkdir(exist_ok=True)
        for name, parts in (('tokens', tokens), ('valid', valid), ('owner', owner), ('old_value', old_value)):np.save(shard / f'{name}.npy', np.concatenate(parts))
    (out / f'summary-{args.worker}.json').write_text(json.dumps(dict(counts, draws=args.draws, worker=args.worker, workers=args.workers)) + '\n')
    print(json.dumps(dict(event='DONE', **counts)), flush=True)
    return 0 if counts['written'] and counts['written'] == counts['roots'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
