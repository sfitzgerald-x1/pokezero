#!/usr/bin/env python3
"""Prepare, explicitly run, or reduce one non-resumable B3 owning study.

Preparation/reduction read compact JSON only. `run` starts real games and
requires separately authorized B2 admission and independent scientific review.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pokezero.owning_encoder_study import (canonical, make_prespec, read_bound,
    run_study, summarize)


def write_new(path, document):
    with Path(path).open('xb') as stream:
        stream.write(canonical(document) + b'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--input', required=True, help='JSON: study_id, runtime, admission and optional fixed bounds')
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
        spec = make_prespec(**json.loads(Path(args.input).read_text()))
        write_new(args.out, spec)
        print(json.dumps(dict(status='PREPARED_NOT_EXECUTED', new_games=0,
            prespec_sha256=hashlib.sha256(Path(args.out).read_bytes()).hexdigest())))
        return 0
    spec = read_bound(dict(path=args.prespec, sha256=args.prespec_sha256))
    if args.mode == 'reduce':
        ledger = read_bound(dict(path=args.ledger, sha256=args.ledger_sha256))
        write_new(args.out, summarize(spec, ledger))
        return 0
    result = asyncio.run(run_study(spec, args.out))
    print(json.dumps(dict(status=result['status'], actual_exit=result['actual_exit'],
                          recorded_games=result['recorded_games'], B3_qualified=False)))
    return result['actual_exit']


if __name__ == '__main__':
    raise SystemExit(main())
