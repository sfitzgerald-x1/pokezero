#!/usr/bin/env python3
"""Refresh the engine's reviewed source pins to the current bytes (stdlib only).

Five modules gate on the sources they were reviewed against, every pin a whole-file sha256:
- engine_source_frame_witness: SOURCE_PINS;
- chance_consumers, through chance_consumers_pins.py: SOURCE_PINS, and DOMAIN_PINS over chance_consumers.DOMAIN_MODULES;
- phase_d_chance: SOURCE_PINS, and DOMAIN_PINS over its DOMAIN_MODULES;
- ordinary_law: MAPPING_PINS, and DOMAIN_PINS over chance_consumers.DOMAIN_MODULES;
- conditional_law: MAPPING_PINS, and KERNEL_PINS over KERNEL_MODULES.
Pinned files pin each other (ordinary_law pins chance_consumers.py), so a refresh iterates to a fixed point in memory.

Only the pin VALUES are rewritten: which files are pinned, and the module lists, are review decisions made by hand.
A refresh is a deliberate review step, run last, after every other source edit of the change it certifies.

Layout, so concurrent changes merge: every pin dict holds one `'file': 'sha256',` entry per line. SOURCE_PINS and
MAPPING_PINS are sorted by file; DOMAIN_PINS and KERNEL_PINS follow their module list's order, which the gates check.
Two changes that refresh different files edit different lines; two that refresh the same file still conflict, as they
should. The domain digests were once single DOMAIN_SHA256 / KERNEL_SHA256 literals over all domain modules, which any two
domain edits changed on the same line. Problems --check reports besides stale values:
- FORMAT: a single-line or unsorted table, or a per-module table out of its list's order (a refresh rewrites it);
- AGGREGATE: a DOMAIN_SHA256 / KERNEL_SHA256 literal is back (from a branch older than per-module pins): remove it by hand;
- LIST: a per-module table no longer matches its module list, or the module list contains duplicates;
- TABLE: an expected table is missing, duplicated, or not a simple dict assignment, or a table is defined in the wrong file.
AGGREGATE, LIST and TABLE issues require a review decision. A refresh refuses to write until they are resolved.

usage: refresh_source_pins.py [--root ROOT] [--check]
  --check  write nothing; exit 1 and name every stale pin and every FORMAT, AGGREGATE or LIST problem.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import re
import sys

TABLES = {'engine_source_frame_witness.py': ('SOURCE_PINS',),
          'chance_consumers_pins.py': ('SOURCE_PINS', 'DOMAIN_PINS'),
          'phase_d_chance.py': ('SOURCE_PINS', 'DOMAIN_PINS'),
          'ordinary_law.py': ('MAPPING_PINS', 'DOMAIN_PINS'),
          'conditional_law.py': ('MAPPING_PINS', 'KERNEL_PINS')}
FILES = tuple(TABLES)
# per-module tables: (file, dict) -> (file holding its module list, list name)
LISTED = {('chance_consumers_pins.py', 'DOMAIN_PINS'): ('chance_consumers.py', 'DOMAIN_MODULES'),
          ('phase_d_chance.py', 'DOMAIN_PINS'): ('phase_d_chance.py', 'DOMAIN_MODULES'),
          ('ordinary_law.py', 'DOMAIN_PINS'): ('chance_consumers.py', 'DOMAIN_MODULES'),
          ('conditional_law.py', 'KERNEL_PINS'): ('conditional_law.py', 'KERNEL_MODULES')}
PIN_DICTS = ('SOURCE_PINS', 'MAPPING_PINS', 'DOMAIN_PINS', 'KERNEL_PINS')
AGGREGATES = ('DOMAIN_SHA256', 'KERNEL_SHA256')
ROUNDS = 10


def _pins(text, var):
    """The `VAR = {...}` dict literal's match (one line or one entry per line; pin dicts hold no braces)."""
    return re.search(rf'^{var}[ \t]*=[ \t]*(\{{[^{{}}]*\}})[ \t]*$', text, re.M)


def assignments(text):
    """Top-level assignment names, with repeats preserved."""
    for node in ast.parse(text).body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
        for target in targets:
            if isinstance(target, ast.Name):
                yield target.id


def _modules(text, name):
    match = re.search(rf'^{name} = tuple\("([^"]*)"\.split\(\)\)$', text, re.M)
    return match.group(1).split() if match else None


def canonical(pins, order=None):
    """A pin dict's one layout: one `'file': 'sha256',` entry per line, sorted by file, or in `order` (module names)."""
    keys = sorted(pins) if order is None else [name + '.py' for name in order]
    return '{\n' + ''.join(f'    {key!r}: {pins[key]!r},\n' for key in keys) + '}'


def order_of(texts, engine, name, var):
    """The module list a per-module table follows; None for a sorted table."""
    if (name, var) not in LISTED:
        return None
    holder, listing = LISTED[(name, var)]
    modules = _modules(texts[holder] if holder in texts else (engine/holder).read_text(), listing)
    if modules is None:
        raise SystemExit(f'{name}: {var} follows {holder}:{listing}, which is missing')
    return modules


def names_match(pins, order):
    return order is None or len(order) == len(pins) and set(pins) == {m + '.py' for m in order}


def problems(texts, engine):
    """Check table definitions, module lists and canonical spelling before writing."""
    found = []
    for name, text in {**texts, 'chance_consumers.py': (engine/'chance_consumers.py').read_text()}.items():
        assigned = list(assignments(text))
        found += [('AGGREGATE', name, var) for var in AGGREGATES if var in assigned]
        for var in PIN_DICTS:
            if var not in TABLES.get(name, ()):
                if var in assigned:
                    found.append(('TABLE', name, f'{var} is defined in the wrong file'))
                continue
            match = _pins(text, var)
            if assigned.count(var) != 1 or not match:
                found.append(('TABLE', name, f'{var} needs exactly one simple dict assignment'))
                continue
            pins, order = ast.literal_eval(match.group(1)), order_of(texts, engine, name, var)
            if not names_match(pins, order):
                found.append(('LIST', name, f'{var} does not name {LISTED[(name, var)][1]} exactly'))
            elif match.group(0) != f'{var} = {canonical(pins, order)}':
                found.append(('FORMAT', name, var))
    return found


def refreshed(engine, original):
    """{file: (old text, refreshed text)} at the fixed point, computed in memory; and the rounds it took. A per-module
    table with LIST drift is left as it is (problems() reports it; main() refuses to write)."""
    texts = dict(original)

    def sha(name):
        return hashlib.sha256(texts[name].encode('utf-8') if name in texts else (engine/name).read_bytes()).hexdigest()

    for rounds in range(1, ROUNDS + 1):
        changed = False
        for name in FILES:
            text = texts[name]
            for var in TABLES[name]:
                match = _pins(text, var)
                if match:
                    pins, order = ast.literal_eval(match.group(1)), order_of(texts, engine, name, var)
                    if names_match(pins, order):
                        rows = canonical({key: sha(key) for key in pins}, order)
                        text = text[:match.start()] + f'{var} = {rows}' + text[match.end():]
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
        a, b = (_pins(t, var) for t in (old, new))
        if a and b:
            x, y = ast.literal_eval(a.group(1)), ast.literal_eval(b.group(1))
            found += [f'{var}[{key!r}]' for key in x if x[key] != y.get(key)]
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='repository root')
    parser.add_argument('--check', action='store_true', help='write nothing; exit 1 on any stale pin or problem')
    args = parser.parse_args(argv)
    engine = args.root / 'gpu_engine'
    original = {name: (engine/name).read_bytes().decode('utf-8') for name in FILES}
    found = problems(original, engine)
    if any(kind == 'TABLE' for kind, _, _ in found):
        for kind, name, detail in found:
            print(f'{kind} {name}: {detail}')
        return 1
    result, rounds = refreshed(engine, original)
    differing = {name: stale(old, new) for name, (old, new) in result.items() if old != new}
    if args.check:
        for name, entries in differing.items():
            if entries:
                print(f'STALE {name}: ' + ', '.join(entries))
        for kind, name, detail in found:
            print(f'{kind} {name}: {detail}')
        print('pins current' if not differing and not found else
              f'{sum(map(len, differing.values()))} stale pins; {len(found)} layout/list problems')
        return 1 if differing or found else 0
    refused = [(kind, name, detail) for kind, name, detail in found if kind != 'FORMAT']
    if refused:
        for kind, name, detail in refused:
            print(f'REFUSED {kind} {name}: {detail} (a review decision: fix it by hand, then refresh)')
        return 1
    for name, (old, new) in result.items():
        if old != new:
            (engine/name).write_bytes(new.encode('utf-8'))
            layout = [f'{detail} layout' for kind, file, detail in found if file == name]
            print(f'refreshed {name}: ' + ', '.join(differing[name] + layout))
    print(f'fixed point after {rounds} rounds; {len(differing)} files rewritten')
    return 0


if __name__ == '__main__':
    sys.exit(main())
