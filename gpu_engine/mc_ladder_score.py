"""Monte Carlo-truth decomposition ladder: score each rung's chosen action at the gate bank's roots by the bank's own Monte Carlo
truth (mc_root_labeller rows, K rollouts per legal action), against the policy's argmax, battle-clustered.

A rung is a JSONL of choices, one per root it decided:
  {"kind": "foulplay", "ordinal": 812, "target": 7, "action": 3}            Foul Play roots: the collection game and our decision
  {"kind": "self", "seed": 900123456, "ours": "p1", "target": 12, "action": 0}  self roots: the battle seed, our seat, our decision
optionally "guard_tripped" (bool) and "greedy_action" (int): with --guards on, a root whose guard tripped plays greedy_action (the
live fallback); with --guards policy it plays the bank's policy argmax (D = 0 exactly: the fallback without a policy-computation
mismatch); with --guards off the rung's own action is scored. `action` is the engine's action index (0-8), as the bank's
`actions` list spells them. A rung row may echo the root's legal "actions" (and "seed" for a Foul Play root); a row whose echo
differs from the bank's is not scored (identity_mismatch): the key matched but the state did not. R0, the bank's own real-Showdown depth-1 value-head choice, is built in (rung name "R0"): the
argmax of each row's baseline completed_q.

Per kind and rung: roots scored, coverage of the bank, override rate (choice != policy argmax), D = Q(choice) - Q(policy argmax)
over all scored roots and where the choice differs, and D against R0 on the same roots, each the mean with a 90% interval
resampling battle seeds (mc_root_report.cluster_ci). Q is the action's Monte Carlo mean with censored (capped) rollouts left out;
a root whose chosen or compared action has every rollout censored is skipped and counted, as is a choice outside the root's
legal actions. Also per rung: guard trips and how often the fallback's greedy_action differs from the bank's policy argmax, rung
rows with no bank root (unmatched), the share of capped rollouts on the chosen and the policy action, the intent-to-treat D
(undecided roots scored as the greedy fallback, D = 0, over every bank root), and minus_r0's own skips (no R0 at the root).
Unbiasedness assumes a rung's producer never reads the bank's outcomes and never reuses its rollout seed streams.

usage: python -m gpu_engine.mc_ladder_score BANK_DIR --rung NAME=PATH [--rung NAME=PATH ...] [--guards on|policy|off] [--out REPORT.json]
"""
import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

from gpu_engine.mc_root_report import CAPPED, censored, cluster_ci

SCHEMA = 'mc-ladder-score.v2'


def root_key(row):
    """The identity a rung row and a bank row share."""
    if row['kind'] == 'foulplay':
        return ('foulplay', int(row['ordinal']), int(row['target']))
    if row['kind'] == 'self':
        return ('self', int(row['seed']), str(row['ours']), int(row['target']))
    raise ValueError(f"unknown root kind {row.get('kind')!r}")


def load_bank(directory):
    """{key: bank row} over a gate bank's roots-*.jsonl; a duplicate key is an error (roots are drawn without replacement), and so
    is a kind whose rows mix continuations or root reply models (never pool M0's v1 rows with a gate bank's)."""
    bank, setups = {}, collections.defaultdict(set)
    for path in sorted(glob.glob(str(Path(directory) / 'roots-*.jsonl'))):
        for line in open(path):
            if not line.strip():continue
            row = json.loads(line);key = root_key(row)
            if key in bank:raise ValueError(f'duplicate root {key} in {path}')
            bank[key] = row;setups[row['kind']].add((row.get('continuation', 'v1_model_continuation'), row.get('opp_root_model')))
    mixed = {kind: sorted(map(str, s)) for kind, s in setups.items() if len(s) > 1}
    if mixed:raise ValueError(f'a bank kind mixes continuations or root reply models: {mixed}')
    return bank


def truth(row):
    """Each legal action's Monte Carlo mean, censored rollouts left out (NaN when all are censored), keyed by action index."""
    c = censored(row['outcomes'])
    return {int(a): (float(np.nanmean(r)) if np.any(~np.isnan(r)) else float('nan')) for a, r in zip(row['actions'], c)}


def policy_action(row):
    return int(row['actions'][int(np.argmax([row['policy'][a] for a in row['actions']]))])


def r0_choices(bank):
    """R0: the bank's real-Showdown depth-1 value-head choice (the argmax of baseline completed_q) at every root with a baseline."""
    out = {}
    for key, row in bank.items():
        b = row.get('baseline')
        if b and row.get('policy'):out[key] = int(row['actions'][int(np.argmax(b['completed_q']))])
    return out


def load_rung(path, guards='on'):
    """{key: record} from a rung JSONL: the action to score (guards on: a tripped guard plays the row's greedy_action; guards
    policy: it plays None, scored as the bank's policy argmax; guards off: the rung's own action, None when it has none), whether
    the guard tripped, its greedy_action, and any echoed identity (actions, seed)."""
    if guards not in ('on', 'policy', 'off'):raise ValueError(f'guards must be on, policy or off, not {guards!r}')
    out = {}
    for line in open(path):
        if not line.strip():continue
        row = json.loads(line);key = root_key(row)
        if key in out:raise ValueError(f'duplicate rung root {key} in {path}')
        tripped = bool(row.get('guard_tripped'));action = row.get('action')
        if tripped and guards == 'on':
            if row.get('greedy_action') is None:raise ValueError(f'{key}: guard tripped without greedy_action')
            action = row['greedy_action']
        elif tripped and guards == 'policy':action = 'policy'
        out[key] = dict(action=None if action is None else (action if action == 'policy' else int(action)), tripped=tripped,
                        greedy=row.get('greedy_action'), actions=row.get('actions'), seed=row.get('seed'))
    return out


def as_records(choices):
    """R0's {key: action} as rung records."""
    return {key: dict(action=a, tripped=False, greedy=None, actions=None, seed=None) for key, a in choices.items()}


def capped_share(row, action):
    o = np.asarray(row['outcomes'], float)[row['actions'].index(action)];return float(np.mean(o == CAPPED))


def score(bank, records, r0, kind):
    """One rung (records from load_rung or as_records) on one root kind."""
    keys = [k for k in bank if k[0] == kind];skipped, r0_skipped = collections.Counter(), collections.Counter()
    d, cl, dd, dcl, vs0, vs0cl, itt, ittcl, cap_c, cap_p = [], [], [], [], [], [], [], [], [], []
    overrides = scored = trips = greedy_ne_policy = 0
    for key in keys:
        row = bank[key];q = truth(row);p = policy_action(row);seed = row['seed']
        record = records.get(key)
        if record is None or record['action'] is None:
            skipped['not_decided'] += 1
            if not np.isnan(q[p]):itt.append(0.);ittcl.append(seed)   # the live fallback: greedy, D = 0
            continue
        if (record['actions'] is not None and [int(a) for a in record['actions']] != row['actions']) or \
                (record['seed'] is not None and int(record['seed']) != int(seed)):
            skipped['identity_mismatch'] += 1;continue
        trips += record['tripped']
        if record['tripped'] and record['greedy'] is not None:greedy_ne_policy += int(record['greedy']) != p
        a = p if record['action'] == 'policy' else record['action']
        if a not in q:skipped['choice_not_legal'] += 1;continue
        if np.isnan(q[a]) or np.isnan(q[p]):skipped['censored_choice'] += 1;continue
        scored += 1;x = q[a] - q[p];d.append(x);cl.append(seed);itt.append(x);ittcl.append(seed)
        cap_c.append(capped_share(row, a));cap_p.append(capped_share(row, p))
        if a != p:overrides += 1;dd.append(x);dcl.append(seed)
        if key not in r0:r0_skipped['no_r0'] += 1
        elif np.isnan(q[r0[key]]):r0_skipped['r0_censored'] += 1
        else:vs0.append(q[a] - q[r0[key]]);vs0cl.append(seed)
    mean = lambda x: None if not x else round(float(np.mean(x)), 5)
    return dict(bank_roots=len(keys), scored=scored, coverage=round(scored / len(keys), 4) if keys else None, skipped=dict(skipped),
                override_rate=round(overrides / scored, 4) if scored else None, guard_trips=trips, greedy_ne_policy=greedy_ne_policy,
                capped_share_choice=mean(cap_c), capped_share_policy=mean(cap_p),
                minus_policy=cluster_ci(d, cl), minus_policy_where_different=cluster_ci(dd, dcl),
                minus_policy_intent_to_treat=cluster_ci(itt, ittcl), minus_r0=cluster_ci(vs0, vs0cl), minus_r0_skipped=dict(r0_skipped))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bank');parser.add_argument('--rung', action='append', default=[], help='NAME=PATH of a rung JSONL')
    parser.add_argument('--guards', choices=('on', 'policy', 'off'), default='on');parser.add_argument('--out')
    args = parser.parse_args(argv)
    bank = load_bank(args.bank);r0 = r0_choices(bank);rungs = {'R0': as_records(r0)};unmatched = {}
    for spec in args.rung:
        name, _, path = spec.partition('=')
        if not name or not path or name in rungs:raise SystemExit(f'--rung NAME=PATH with a new name required: {spec!r}')
        rungs[name] = load_rung(path, args.guards);unmatched[name] = sum(1 for key in rungs[name] if key not in bank)
    kinds = sorted({k[0] for k in bank})
    report = dict(schema=SCHEMA, bank=str(args.bank), guards=args.guards, roots={k: sum(1 for x in bank if x[0] == k) for k in kinds},
                  unmatched_rung_rows=unmatched,
                  rungs={name: {kind: score(bank, choices, r0, kind) for kind in kinds} for name, choices in rungs.items()})
    text = json.dumps(report, indent=1)
    if args.out:Path(args.out).write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
