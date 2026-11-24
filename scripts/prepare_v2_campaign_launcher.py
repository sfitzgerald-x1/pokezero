"""Extend the bound drill launcher to lock and resume readmitted first attempts."""

import argparse
import hashlib
from pathlib import Path


def render(source):
    changes = (
        ("args.resume_run and args.kueue and args.nodes == 1 and",
         "args.kueue and args.nodes == 1 and not args.require_fresh and"),
        ("re.fullmatch(r'q-[a-f0-9]{16}',args.resilient_support)",
         "re.fullmatch(r'/shared/scott/v2p/v2[a-z]-[a-f0-9]{6,12}-cpu/q-[a-f0-9]{16}', "
         "args.resilient_support)"),
        ("raise SystemExit('resilience support requires a bound queued resume')",
         "raise SystemExit('resilience support requires a bound queued single-node Job')"),
        ("        script = script.replace('flock -n 9', 'flock -w 120 9')\n"
         "        support = '$OUT/' + args.resilient_support",
         "        if args.resume_run:\n"
         "            script = script.replace('flock -n 9', 'flock -w 120 9')\n"
         "        else:\n"
         "            anchor = 'mkdir -p \"$OUT\" /code/source\\n'\n"
         "            if script.count(anchor) != 1:\n"
         "                raise SystemExit('resilience wrapper needs the standard output setup')\n"
         "            script = script.replace(anchor, anchor + 'exec 9>\"$OUT/.resume.lock\"\\n'\n"
         "                                    'flock -w 120 9 || exit 1\\n')\n"
         "        support = args.resilient_support"),
    )
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError('unsupported drill launcher template')
        source = source.replace(old, new)
    compile(source, 'launch.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drill-launcher', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = args.drill_launcher.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.sha256:
        parser.error('drill launcher differs from its reviewed binding')
    with args.out.open('x') as output:
        output.write(render(data.decode()))


if __name__ == '__main__':
    main()
