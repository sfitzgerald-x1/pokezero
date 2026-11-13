"""Record gpu_engine/effect_probe_device_pins.json from a device-mode run's per-scenario result files.

usage: python scripts/record_effect_probe_device_pins.py RUN_JSON... [--out PATH] [--check]

Every diff at an ALLOWLIST key (open or allowlisted in the run: a stale pin leaves it open, carrying the new values) and
every refusal at an allowlisted 'b2_refusal' key becomes that key's pin; pins of scenarios outside the run are kept.
It refuses, writing nothing, when one key shows two values, when a refusal's streams refuse differently, when the run
has a diff or a refusal outside the allowlist (a real open diff is never pinned), or when an allowlist entry ends with
no pin. --check compares with the file instead of writing it. Then confirm with a run that reads the pins: every case
EXACT or B2_LIMIT.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from gpu_engine import effect_probe_device as dev   # noqa: E402


def record(paths, allowlist=None, existing=None):
    """(pins, problems) from the run's result files; existing pins of scenarios outside the run are kept."""
    allowlist = dev.ALLOWLIST if allowlist is None else allowlist
    existing = dev.PINS if existing is None else existing
    pins, problems, scenarios = {}, [], set()

    def put(key, value):
        if key in pins and pins[key] != value:problems.append(f'two values at {key}: {pins[key]} and {value}')
        pins[key] = value

    for path in paths:
        data = json.loads(Path(path).read_text())
        if 'results' not in data or 'move' not in data:continue   # a run summary
        move = data['move'];scenarios.add(move)
        for r in data['results']:
            device = r.get('device')
            if device is None:problems.append(f"{move} {r.get('case_id')}: no device record");continue
            for s in device.get('steps', []):
                for d in s['diffs'] + s.get('allowlisted', []):
                    key = (move, r['case_id'], s['step'], d['field'])
                    if key in allowlist:put(key, dev._canon([d['b2'], d['device']]))
                    else:problems.append(f'open diff outside the allowlist at {key}')
            refusal = device.get('refusal')
            if refusal:
                key = (move, r['case_id'], refusal['step'], 'b2_refusal')
                kinds = {json.dumps([x.get('kind'), x.get('error')]) for x in [refusal] + refusal.get('other_streams', [])}
                if key not in allowlist:problems.append(f'refusal outside the allowlist at {key}: {refusal}')
                elif len(kinds) > 1:problems.append(f'streams refuse differently at {key}: {sorted(kinds)}')
                else:put(key, dict(kind=refusal.get('kind'), error=refusal.get('error')))
    merged = {k: v for k, v in existing.items() if k[0] not in scenarios and k in allowlist}
    merged.update(pins)
    problems += [f'allowlist entry with no pin: {k}' for k in sorted(set(allowlist) - set(merged), key=str)]
    return merged, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('runs', nargs='+', help="the run's per-scenario result files")
    parser.add_argument('--out', default=str(dev.PINS_PATH))
    parser.add_argument('--check', action='store_true', help='compare with --out instead of writing it')
    args = parser.parse_args(argv)
    pins, problems = record(args.runs, existing=dev.load_pins(args.out))
    for problem in problems:print(problem, file=sys.stderr)
    if problems:return 1
    if args.check:
        same = dev.load_pins(args.out) == pins
        print(f"{len(pins)} pins {'match' if same else 'differ from'} {args.out}");return 0 if same else 1
    dev.write_pins(pins, args.out);print(f'wrote {len(pins)} pins to {args.out}');return 0


if __name__ == '__main__':
    raise SystemExit(main())
