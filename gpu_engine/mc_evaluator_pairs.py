"""Backfill the worst-case paired bounds (d_lo, se_lo) into an extracted feature directory whose index rows predate them.

Feature rows written before mc_evaluator_features stored d_lo carry only se_pair, which the trainer's significance test no longer
uses. This recomputes, per index row, mc_evaluator_features.pair_bounds from the bank row the extraction read (matched by identity:
kind, ordinal, seed, ours, target) and writes NEW_DIR/index-<w>.jsonl with d_lo and se_lo added, every other field unchanged;
NEW_DIR/features-<w> links to the source's shard (the arrays are not copied).

It refuses (no output) unless every index row has exactly one bank row, the actions and K agree, and the row's stored targets are the
bank row's (se_pair, Q_MC and its SE recomputed from the bank outcomes, within 1e-9): the backfill reads the very outcomes the
extraction did, sign included.

usage: python -m gpu_engine.mc_evaluator_pairs --features DIR --bank BANK_DIR --out NEW_DIR
"""
import argparse
import glob
import json
from pathlib import Path

from gpu_engine.mc_evaluator_features import censored_q, pair_bounds, pair_se


def identity(row):
    return (row['kind'], row.get('ordinal') if row['kind'] == 'foulplay' else None, row['seed'], row['ours'], row['target'])


def bank_index(bank):
    out = {}
    for path in sorted(glob.glob(str(Path(bank) / 'roots-*.jsonl'))):
        for line in open(path):
            if not line.strip():continue
            row = json.loads(line);key = identity(row)
            if key in out:raise ValueError(f'duplicate bank root {key}')
            out[key] = row
    return out


def same(a, b, tol=1e-9):
    if a is None or b is None:return a is None and b is None
    return abs(a - b) <= tol


def backfill(index_row, bank_row):
    """index_row with d_lo and se_lo (pair_bounds of the bank row's outcomes); ValueError when the bank row is not the extraction's."""
    if bank_row['actions'] != index_row['actions'] or len(bank_row['outcomes'][0]) != index_row['K']:
        raise ValueError(f'actions or K differ for {identity(index_row)}')
    q, se = censored_q(bank_row['outcomes'])
    for name, mine, theirs in (('se_pair', pair_se(bank_row['outcomes']), index_row['se_pair']), ('q_mc', [q], [index_row['q_mc']]),
                               ('se_mc', [se], [index_row['se_mc']])):
        if len(mine) != len(theirs) or any(len(r) != len(s) or any(not same(x, y) for x, y in zip(r, s)) for r, s in zip(mine, theirs)):
            raise ValueError(f'{name} differs from the bank row for {identity(index_row)}')
    d_lo, se_lo = pair_bounds(bank_row['outcomes'])
    return dict(index_row, d_lo=d_lo, se_lo=se_lo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', required=True);parser.add_argument('--bank', required=True);parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    source = Path(args.features).resolve();bank = bank_index(args.bank);written = {}
    rows = {}
    for index in sorted(source.glob('index-*.jsonl')):
        out = []
        for line in open(index):
            if not line.strip():continue
            row = json.loads(line);key = identity(row)
            if key not in bank:raise SystemExit(f'no bank row for {key}')
            out.append(backfill(row, bank[key]))
        rows[index.name] = out
    target = Path(args.out);target.mkdir(parents=True, exist_ok=False)
    for name, out in rows.items():
        (target / name).write_text(''.join(json.dumps(r) + '\n' for r in out));written[name] = len(out)
        shard = source / f"features-{name[len('index-'):-len('.jsonl')]}"
        if shard.exists():(target / shard.name).symlink_to(shard, target_is_directory=True)
    summary = dict(source=str(source), bank=str(Path(args.bank).resolve()), rows=sum(written.values()), files=len(written))
    (target / 'pairs-summary.json').write_text(json.dumps(summary) + '\n');print(json.dumps(dict(event='DONE', **summary)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
