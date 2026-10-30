#!/usr/bin/env python3
"""Prepare/run/reduce the fixed owning offline cost attempt. No game launcher.

Prepare reads metadata and chosen owning literals on remote storage. Run loads
the champion once and performs exactly the fixed 20 ordinary actor decisions.
Execution remains subject to B2 admission and independent source review.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pokezero.owning_encoder_cost import prepare_cost, run_cost, summarize
from pokezero.owning_encoder_study import atomic_json, read_bound


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--input', required=True)
    prepare.add_argument('--out', required=True)
    for name in ('run', 'reduce'):
        command = sub.add_parser(name)
        command.add_argument('--prespec', required=True)
        command.add_argument('--prespec-sha256', required=True)
        command.add_argument('--out', required=True)
        if name == 'reduce':
            command.add_argument('--ledger', required=True)
            command.add_argument('--ledger-sha256', required=True)
    args = parser.parse_args(argv)
    if args.mode == 'prepare':
        result = prepare_cost(**json.loads(Path(args.input).read_text()))
    else:
        spec = read_bound(dict(path=args.prespec,sha256=args.prespec_sha256))
        if args.mode == 'run':
            result = run_cost(spec,args.out)
            print(json.dumps(result['summary']))
            return result['actual_exit']
        result = summarize(spec,read_bound(dict(path=args.ledger,sha256=args.ledger_sha256)))
    target = Path(args.out)
    if target.exists():
        raise FileExistsError('cost artifacts are never overwritten/resumed')
    atomic_json(target,result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
