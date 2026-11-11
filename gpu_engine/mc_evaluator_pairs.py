"""Backfill the CRN-paired mean difference (d_pair) into an extracted feature directory whose index rows predate it.

Feature rows written before mc_evaluator_features stored d_pair carry only se_pair, so the trainer's significance test cannot pair
its numerator with its SE. This recomputes, per index row, mc_evaluator_features.pair_stats from the bank row the extraction read
(matched by identity: kind, ordinal, seed, ours, target) and writes NEW_DIR/index-<w>.jsonl with d_pair added, every other field
unchanged; NEW_DIR/features-<w> links to the source's shard (the arrays are not copied).

It refuses (no output) unless every index row has exactly one bank row, the actions agree, and the recomputed se_pair equals the
stored one (the same outcomes, within 1e-9): the backfill can only add the mean of the very samples the stored SE describes.

usage: python -m gpu_engine.mc_evaluator_pairs --features DIR --bank BANK_DIR --out NEW_DIR
"""
import argparse
import glob
import json
from pathlib import Path

from gpu_engine.mc_evaluator_features import pair_stats


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
    """index_row with d_pair (pair_stats of the bank row's outcomes); ValueError when the bank row is not the extraction's."""
    if bank_row['actions'] != index_row['actions']:raise ValueError(f'actions differ for {identity(index_row)}')
    d_pair, se_pair = pair_stats(bank_row['outcomes']);stored = index_row['se_pair']
    if len(stored) != len(se_pair) or any(not same(x, y) for r, s in zip(stored, se_pair) for x, y in zip(r, s)):
        raise ValueError(f'se_pair differs from the bank row for {identity(index_row)}')
    return dict(index_row, d_pair=d_pair)


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
