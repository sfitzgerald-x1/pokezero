#!/usr/bin/env python3
"""The W x S latency trade-off at the live shape: depthn_live_bench once per (W worlds, S chance draws), then one table.

Design. Each configuration WxS is one run of scripts/depthn_live_bench.py (the live server's own DepthNLiveSearch at bucket
1, max_requests 1, DN2: prepare, classify, build, ONE price call, the backups, averaging and the effect guard, nothing
re-implemented) in its own process, one after another on the one GPU, with the bench arguments given after `--` plus:
- the rung: DN2W16 for W = 16, DN2W8 for W = 8, and DN2W8 with --worlds W for a W the constructor does not build (W = 4:
  the first 4 of DN2W8's 8 worlds; the constructor's draws are a prefix-stable sequence). That root's prepare still
  constructs the rung's 8 worlds, so its prepare and decision walls are upper bounds for a 4-world constructor; its price
  is exact;
- --chance-draws S: each world priced under chance seeds 0..S-1 (seed 0: the constructed keys, the live draw) as W x S
  world lanes of the one price call, averaging over all of them (depthn_live_search's module docstring, chance seeds);
- --out DIR/W<W>xS<S>.
Then summary.json and summary.md tabulate, per configuration over its COMPLETE roots, the bench's own stage percentiles
(p50 / p95 / p99 / max): price_seconds (the price call), decision_seconds (batch start to result: the server-side
decision), wall_seconds (process_many's wall, what the client waits for less the network), prepare_seconds; and over every
eligible root (any status) the share whose wall plus the client's reserve fits the 3 s ladder cap. Informational: no
verdict.

  python scripts/depthn_chance_bench.py --out NEW_DIR [--configs 16x1,16x2,16x4,8x2,4x4] -- <depthn_live_bench.py arguments
      without --rung, --worlds, --chance-draws, --out>
  python scripts/depthn_chance_bench.py --out EXISTING_DIR --combine-only   (re-tabulate finished runs)
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'scripts' / 'depthn_live_bench.py'
CONFIGS = ((16, 1), (16, 2), (16, 4), (8, 2), (4, 4))
RUNG_WORLDS = {'DN2W8': 8, 'DN2W16': 16}   # depthn_live_search.RUNGS (kept here so tabulating needs no JAX)
OWNED = ('--rung', '--worlds', '--chance-draws', '--out')
STAGES = ('price_seconds', 'decision_seconds', 'wall_seconds', 'prepare_seconds')
LADDER_SECONDS, RESERVE_SECONDS = 3.0, .15   # depthn_live_bench's cap and client reserve


def config_arg(text):
    out = []
    for item in text.split(','):
        try:w, s = (int(x) for x in item.lower().split('x'))
        except ValueError:raise argparse.ArgumentTypeError(f'WxS items (e.g. 16x2) required, got {item!r}') from None
        if w < 2 or w % 2 or w > max(RUNG_WORLDS.values()) or s < 1:raise argparse.ArgumentTypeError(f'{item}: an even W <= 16 and S >= 1 required')
        out.append((w, s))
    if len(set(out)) != len(out):raise argparse.ArgumentTypeError('distinct configurations required')
    return tuple(out)


def rung_for(worlds):
    """(rung, --worlds value or None): the smallest rung with at least `worlds` worlds, a prefix when it has more."""
    rung = min((r for r, w in RUNG_WORLDS.items() if w >= worlds), key=RUNG_WORLDS.get)
    return rung, (None if RUNG_WORLDS[rung] == worlds else worlds)


def name(worlds, draws):return f'W{worlds}xS{draws}'


def command(bench_args, worlds, draws, out, python=sys.executable):
    """The depthn_live_bench.py command line of one configuration."""
    rung, prefix = rung_for(worlds)
    return [python, str(BENCH), *bench_args, '--rung', rung, *(() if prefix is None else ('--worlds', str(prefix))), '--chance-draws', str(draws),
        '--out', str(Path(out) / name(worlds, draws))]


def row(worlds, draws, summary):
    """One table row from a configuration's bench summary.json (None stages when it has no COMPLETE root)."""
    pooled = summary['pooled'];stages = pooled.get('stages') or {};latency = pooled.get('latency') or {}
    rung, prefix = rung_for(worlds)
    return dict(W=worlds, S=draws, lanes=worlds * draws, rung=rung, worlds_prefix=prefix, roots=pooled.get('roots'), priced=pooled.get('priced'),
        eligible=pooled.get('eligible'), complete=pooled.get('complete'), causes=pooled.get('causes'), **{s: stages.get(s) for s in STAGES},
        within_ladder_eligible=latency.get('within_ladder'), wall_eligible=pooled.get('wall_eligible'), chance=summary.get('chance'),
        prepare_upper_bound=prefix is not None)


def combine(out, configs):
    """summary.json and summary.md over the finished configurations under `out` (a missing one is listed, never invented)."""
    out = Path(out);rows, missing = [], []
    for worlds, draws in configs:
        path = out / name(worlds, draws) / 'summary.json'
        if not path.exists():missing.append(name(worlds, draws));continue
        rows.append(row(worlds, draws, json.loads(path.read_text())))
    summary = dict(schema='pokezero.depthn-chance-bench.v1', configs=[name(w, s) for w, s in configs], rows=rows, missing=missing,
        cap_seconds=LADDER_SECONDS, reserve_seconds=RESERVE_SECONDS,
        scope='INFORMATIONAL: warm per-decision cost at bucket 1, max_requests 1, DN2, per (W worlds, S chance draws); no verdict')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    (out / 'summary.md').write_text(markdown(summary))
    return summary


def _p(p, key):return '-' if p is None or p.get(key) is None or not math.isfinite(p[key]) else f'{p[key]:.3f}'


def markdown(summary):
    lines = ['# DN2 latency: W worlds x S chance draws (bucket 1, max_requests 1)', '',
        f"Per configuration over its COMPLETE roots (seconds); 'within cap' over every eligible root: wall + {summary['reserve_seconds']} s <= "
        f"{summary['cap_seconds']} s. W=4 runs DN2W8 with --worlds 4 (8 worlds constructed): its prepare and decision walls are upper bounds.", '',
        '| W x S | lanes | rung | complete / eligible | price p50 | price p95 | price p99 | decision p50 | decision p95 | decision p99 | wall p50 | wall p95 '
        '| wall p99 | prepare p50 | prepare p95 | within cap |', '|' + ' --- |' * 16]
    for r in summary['rows']:
        rung = r['rung'] + ('' if r['worlds_prefix'] is None else ' (first %d)' % r['worlds_prefix'])
        lines.append(f"| {r['W']} x {r['S']} | {r['lanes']} | {rung} "
            f"| {r['complete']} / {r['eligible']} | " + ' | '.join(_p(r[s], q) for s in ('price_seconds', 'decision_seconds', 'wall_seconds') for q in ('p50', 'p95', 'p99'))
            + f" | {_p(r['prepare_seconds'], 'p50')} | {_p(r['prepare_seconds'], 'p95')} | "
            + ('-' if r['within_ladder_eligible'] is None else f"{100 * r['within_ladder_eligible']:.1f}%") + ' |')
    if summary['missing']:lines += ['', f"Missing (not run or not finished): {', '.join(summary['missing'])}"]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    own, bench_args = (argv[:argv.index('--')], argv[argv.index('--') + 1:]) if '--' in argv else (argv, [])
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', required=True, help='a new directory (an existing one with --combine-only)')
    p.add_argument('--configs', type=config_arg, default=CONFIGS, help='comma-separated WxS (default ' + ','.join(f'{w}x{s}' for w, s in CONFIGS) + ')')
    p.add_argument('--combine-only', action='store_true', help='tabulate the finished configurations under --out, run nothing')
    p.add_argument('--dry-run', action='store_true', help='print the bench command lines and run nothing')
    args = p.parse_args(own)
    clash = [x for x in bench_args if x.split('=')[0] in OWNED]
    if clash:p.error(f'{", ".join(OWNED)} are set per configuration here; drop {clash} from the bench arguments')
    if args.combine_only:
        summary = combine(args.out, args.configs);print(json.dumps(dict(done=args.out, rows=len(summary['rows']), missing=summary['missing'])));return 0
    commands = [command(bench_args, w, s, args.out) for w, s in args.configs]
    if args.dry_run:
        for c in commands:print(' '.join(c))
        return 0
    if not bench_args:p.error('the depthn_live_bench.py arguments go after --')
    Path(args.out).mkdir(parents=True, exist_ok=False);failed = []
    for (w, s), c in zip(args.configs, commands):
        print(json.dumps(dict(config=name(w, s), command=c)), flush=True)
        with (Path(args.out) / f'{name(w, s)}.log').open('x') as log:code = subprocess.call(c, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(config=name(w, s), exit=code)), flush=True)
        if code not in (0, 2):failed.append(dict(config=name(w, s), exit=code))
    summary = combine(args.out, args.configs)
    print(json.dumps(dict(done=args.out, rows=len(summary['rows']), missing=summary['missing'], failed=failed)), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':raise SystemExit(main())
