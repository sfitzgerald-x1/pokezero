"""Monte Carlo-truth decomposition ladder: score each rung's chosen action at the gate bank's roots by the bank's own Monte Carlo
truth (mc_root_labeller rows, K rollouts per legal action), against the policy's argmax, battle-clustered.

A rung is a JSONL of choices, one per root it decided:
  {"kind": "foulplay", "ordinal": 812, "target": 7, "action": 3}            Foul Play roots: the collection game and our decision
  {"kind": "self", "seed": 900123456, "ours": "p1", "target": 12, "action": 0}  self roots: the battle seed, our seat, our decision
optionally "guard_tripped" (bool) and "greedy_action" (int): with --guards on, a root whose guard tripped plays greedy_action (the
live fallback); with --guards off the rung's own action is scored. `action` is the engine's action index (0-8), as the bank's
`actions` list spells them. R0, the bank's own real-Showdown depth-1 value-head choice, is built in (rung name "R0"): the
argmax of each row's baseline completed_q.

Per kind and rung: roots scored, coverage of the bank, override rate (choice != policy argmax), D = Q(choice) - Q(policy argmax)
over all scored roots and where the choice differs, and D against R0 on the same roots, each the mean with a 90% interval
resampling battle seeds (mc_root_report.cluster_ci). Q is the action's Monte Carlo mean with censored (capped) rollouts left out;
a root whose chosen or compared action has every rollout censored is skipped and counted, as is a choice outside the root's
legal actions.

usage: python -m gpu_engine.mc_ladder_score BANK_DIR --rung NAME=PATH [--rung NAME=PATH ...] [--guards on|off] [--out REPORT.json]
"""
import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

from gpu_engine.mc_root_report import censored, cluster_ci

SCHEMA = 'mc-ladder-score.v1'


def root_key(row):
    """The identity a rung row and a bank row share."""
    if row['kind'] == 'foulplay':
        return ('foulplay', int(row['ordinal']), int(row['target']))
    if row['kind'] == 'self':
        return ('self', int(row['seed']), str(row['ours']), int(row['target']))
    raise ValueError(f"unknown root kind {row.get('kind')!r}")


def load_bank(directory):
    """{key: bank row} over a gate bank's roots-*.jsonl; a duplicate key is an error (roots are drawn without replacement)."""
    bank = {}
    for path in sorted(glob.glob(str(Path(directory) / 'roots-*.jsonl'))):
        for line in open(path):
            if not line.strip():continue
            row = json.loads(line);key = root_key(row)
            if key in bank:raise ValueError(f'duplicate root {key} in {path}')
            bank[key] = row
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
    """{key: chosen action} from a rung JSONL; with guards on, a tripped guard plays the row's greedy_action."""
    out = {}
    for line in open(path):
        if not line.strip():continue
        row = json.loads(line);key = root_key(row)
        if key in out:raise ValueError(f'duplicate rung root {key} in {path}')
        action = row['action']
        if guards == 'on' and row.get('guard_tripped'):
            if row.get('greedy_action') is None:raise ValueError(f'{key}: guard tripped without greedy_action')
            action = row['greedy_action']
        out[key] = int(action)
    return out


def score(bank, choices, r0, kind):
    """One rung on one root kind."""
    keys = [k for k in bank if k[0] == kind];skipped = collections.Counter()
    d, cl, dd, dcl, vs0, vs0cl, overrides, scored = [], [], [], [], [], [], 0, 0
    for key in keys:
        if key not in choices:skipped['not_decided'] += 1;continue
        row = bank[key];q = truth(row);a, p = choices[key], policy_action(row)
        if a not in q:skipped['choice_not_legal'] += 1;continue
        if np.isnan(q[a]) or np.isnan(q[p]):skipped['censored_choice'] += 1;continue
        seed = row['seed'];scored += 1;x = q[a] - q[p];d.append(x);cl.append(seed)
        if a != p:overrides += 1;dd.append(x);dcl.append(seed)
        if key in r0 and not np.isnan(q[r0[key]]):vs0.append(q[a] - q[r0[key]]);vs0cl.append(seed)
    return dict(bank_roots=len(keys), scored=scored, coverage=round(scored / len(keys), 4) if keys else None, skipped=dict(skipped),
                override_rate=round(overrides / scored, 4) if scored else None, minus_policy=cluster_ci(d, cl),
                minus_policy_where_different=cluster_ci(dd, dcl), minus_r0=cluster_ci(vs0, vs0cl))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bank');parser.add_argument('--rung', action='append', default=[], help='NAME=PATH of a rung JSONL')
    parser.add_argument('--guards', choices=('on', 'off'), default='on');parser.add_argument('--out')
    args = parser.parse_args(argv)
    bank = load_bank(args.bank);r0 = r0_choices(bank);rungs = {'R0': r0}
    for spec in args.rung:
        name, _, path = spec.partition('=')
        if not name or not path or name in rungs:raise SystemExit(f'--rung NAME=PATH with a new name required: {spec!r}')
        rungs[name] = load_rung(path, args.guards)
    kinds = sorted({k[0] for k in bank})
    report = dict(schema=SCHEMA, bank=str(args.bank), guards=args.guards, roots={k: sum(1 for x in bank if x[0] == k) for k in kinds},
                  rungs={name: {kind: score(bank, choices, r0, kind) for kind in kinds} for name, choices in rungs.items()})
    text = json.dumps(report, indent=1)
    if args.out:Path(args.out).write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
