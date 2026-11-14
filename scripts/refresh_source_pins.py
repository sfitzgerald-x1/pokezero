#!/usr/bin/env python3
"""Refresh the engine's reviewed source pins and domain digests to the current bytes (stdlib only).

Five modules pin the sources they were reviewed against: engine_source_frame_witness (SOURCE_PINS), chance_consumers
and phase_d_chance (SOURCE_PINS, DOMAIN_SHA256 over DOMAIN_MODULES), ordinary_law (MAPPING_PINS, and DOMAIN_SHA256
over chance_consumers.DOMAIN_MODULES) and conditional_law (MAPPING_PINS, KERNEL_SHA256 over KERNEL_MODULES). Pinned
files pin each other (ordinary_law pins chance_consumers.py), so a refresh iterates to a fixed point in memory.

Only the pin VALUES are rewritten: which files are pinned, and the module lists, are review decisions made by hand.
A refresh is a deliberate review step, run last, after every other source edit of the change it certifies.

usage: refresh_source_pins.py [--root ROOT] [--check]
  --check  write nothing; exit 1 and name every stale pin when the tree's pins differ from its bytes.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import re
import sys

FILES = ('engine_source_frame_witness.py', 'chance_consumers.py', 'phase_d_chance.py', 'ordinary_law.py',
         'conditional_law.py')
PIN_DICTS = ('SOURCE_PINS', 'MAPPING_PINS')
# (digest, module list in the same file, or the file whose list it shares)
DIGESTS = (('DOMAIN_SHA256', 'DOMAIN_MODULES'), ('KERNEL_SHA256', 'KERNEL_MODULES'))
SHARED_DOMAIN = {'ordinary_law.py': 'chance_consumers.py'}   # ordinary_law hashes chance_consumers.DOMAIN_MODULES
ROUNDS = 10


def _modules(text, name):
    match = re.search(rf'^{name} = tuple\("([^"]*)"\.split\(\)\)$', text, re.M)
    return match.group(1).split() if match else None


def refreshed(engine):
    """{file: (old text, refreshed text)} at the fixed point, computed in memory; and the rounds it took."""
    texts = {name: (engine/name).read_bytes().decode('utf-8') for name in FILES}
    original = dict(texts)

    def data(name):
        return texts[name].encode('utf-8') if name in texts else (engine/name).read_bytes()

    def sha(name):
        return hashlib.sha256(data(name)).hexdigest()

    def domain(modules):
        return hashlib.sha256(b''.join(m.encode() + b'\0' + data(m + '.py') + b'\0' for m in modules)).hexdigest()

    for rounds in range(1, ROUNDS + 1):
        changed = False
        for name in FILES:
            text = texts[name]
            for var in PIN_DICTS:
                match = re.search(rf'^{var} = (\{{.*\}})$', text, re.M)
                if match:
                    pins = ast.literal_eval(match.group(1))
                    text = text[:match.start(1)] + repr({key: sha(key) for key in pins}) + text[match.end(1):]
            for var, listing in DIGESTS:
                match = re.search(rf"^{var} = '([0-9a-f]{{64}})'$", text, re.M)
                if not match:
                    continue
                modules = _modules(text, listing)
                if modules is None and name in SHARED_DOMAIN:
                    modules = _modules(texts[SHARED_DOMAIN[name]], listing)
                if modules is None:
                    raise SystemExit(f'{name}: {var} has no {listing} to hash')
                text = text[:match.start(1)] + domain(modules) + text[match.end(1):]
            if text != texts[name]:
                texts[name] = text
                changed = True
        if not changed:
            return {name: (original[name], texts[name]) for name in FILES}, rounds
    raise SystemExit(f'no fixed point after {ROUNDS} rounds')


def stale(old, new):
    """The pin entries whose value differs between two versions of one file."""
    found = []
    for var in PIN_DICTS:
        a, b = (re.search(rf'^{var} = (\{{.*\}})$', t, re.M) for t in (old, new))
        if a and b:
            x, y = ast.literal_eval(a.group(1)), ast.literal_eval(b.group(1))
            found += [f'{var}[{key!r}]' for key in x if x[key] != y.get(key)]
    for var, _ in DIGESTS:
        a, b = (re.search(rf"^{var} = '([0-9a-f]{{64}})'$", t, re.M) for t in (old, new))
        if a and b and a.group(1) != b.group(1):
            found.append(var)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='repository root')
    parser.add_argument('--check', action='store_true', help='write nothing; exit 1 on any stale pin')
    args = parser.parse_args(argv)
    engine = args.root / 'gpu_engine'
    result, rounds = refreshed(engine)
    differing = {name: stale(old, new) for name, (old, new) in result.items() if old != new}
    if args.check:
        for name, entries in differing.items():
            print(f'STALE {name}: ' + ', '.join(entries))
        print('pins current' if not differing else f'{sum(map(len, differing.values()))} stale pins in {len(differing)} files')
        return 1 if differing else 0
    for name, entries in differing.items():
        (engine/name).write_bytes(result[name][1].encode('utf-8'))
        print(f'refreshed {name}: ' + ', '.join(entries))
    print(f'fixed point after {rounds} rounds; {len(differing)} files rewritten')
    return 0


if __name__ == '__main__':
    sys.exit(main())
