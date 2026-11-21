#!/usr/bin/env python3
"""Check, refresh or show the content-anchored source citations in tests/source_anchors/ (stdlib only).

Prose cites code as `<path>#<id>`; the entry in tests/source_anchors/<path>.json pins a scope, a needle on exactly one
line of it, an extent and a digest of the span's code (gpu_engine/source_anchor.py). A moved citation passes as it
stands. A changed cited statement fails --check as CHANGED until --write re-pins it, and that is a review step:
--write prints each refreshed span, and with --base REF the span at REF too, so the reviewer reads old against new.
A new entry is added by hand with "digest": "" and filled by --write.

MISSING, AMBIGUOUS, INVALID, ORPHAN and UNANCHORED need a person to re-point the citation or its entry, and a store
that does not parse or sits at the wrong path needs one too, so --write refuses, writing nothing, while any is in
scope. Run it last, after every other source edit of the change it certifies.

usage: refresh_source_anchors.py [--root ROOT] (--check | --write [PATH...] | --show PATH#ID) [--base REF]
  --check         write nothing; exit 1 and name every problem.
  --write [PATH]  re-pin the CHANGED digests (of the cited PATHs only, when given) and rewrite those stores canonically.
  --show PATH#ID  print where the citation resolves now, and its span.
  --base REF      with --write or --show, also resolve each span at git REF (`git show REF:path`).
"""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_engine import source_anchor as sa  # noqa: E402


def _body(lines):
    return [f'    {n}: {text}' for n, text in lines]


def at_base(root, anchor, ref):
    """The anchor's span at git `ref`, as printable lines, or the reason it does not resolve there."""
    shown = subprocess.run(['git', '-C', str(root), 'show', f'{ref}:./{anchor.path}'], capture_output=True,
                           encoding='utf-8', errors='replace')
    if shown.returncode:
        return [f'  at {ref}: {anchor.path} is not there ({shown.stderr.strip()})']
    try:
        span = sa.resolve(anchor, root, text=shown.stdout)
    except sa.AnchorError as error:
        return [f'  at {ref}: does not resolve ({type(error).__name__}: {error})']
    return ([f'  at {ref}: {sa.render(span)}, digest {sa.digest(span, root, text=shown.stdout)}']
            + _body(sa.span_lines(span, root, text=shown.stdout)))


def check(root):
    found = sa.problems(root)
    for problem in found:
        print(problem)
    print('anchors current' if not found else f'{len(found)} anchor problems')
    return 1 if found else 0


def write(root, paths, base):
    unknown = [path for path in paths if not sa.store_file(root, path).is_file()]
    if unknown:
        print('no store for ' + ', '.join(unknown))
        return 1
    found = sa.problems(root, paths=paths or None)
    blocking = [problem for problem in found if not problem.fixable]
    if blocking:
        for problem in blocking:
            print(problem)
        print(f'refusing to write: {len(blocking)} problems need a person to re-point the citation; nothing written')
        return 1
    changed = {problem.cite for problem in found if problem.kind == 'CHANGED'}
    refreshed = rewritten = 0
    for path in sorted(set(paths)) or sa.store_paths(root):
        file = sa.store_file(root, path)
        text = file.read_text(encoding='utf-8')
        store = sa.parse_store(text)
        for key, entry in sorted(store['anchors'].items()):
            if f'{path}#{key}' not in changed:
                continue
            anchor = sa.Anchor.from_entry(path, entry)
            span = sa.resolve(anchor, root)
            new = sa.digest(span, root)
            print(f'refreshed {path}#{key}: {entry["digest"] or "nothing"} -> {new}, now {sa.render(span)}')
            print('\n'.join(_body(sa.span_lines(span, root))))
            if base:
                print('\n'.join(at_base(root, anchor, base)))
            entry['digest'] = new
            refreshed += 1
        canonical = sa.dump_store(store)
        if canonical != text:
            file.write_text(canonical, encoding='utf-8')
            rewritten += 1
    print(f'{refreshed} digests refreshed; {rewritten} store files rewritten')
    return 0


def show(root, cite, base):
    path, _, key = cite.rpartition('#')
    file = sa.store_file(root, path)
    entry = sa.load_store(file)['anchors'].get(key) if path and file.is_file() else None
    if entry is None:
        print(f'no entry {cite}')
        return 1
    anchor = sa.Anchor.from_entry(path, entry)
    try:
        span = sa.resolve(anchor, root)
    except sa.AnchorError as error:
        print(f'{cite}: {type(error).__name__}: {error}')
        return 1
    print(f'{cite} -> {sa.render(span)}')
    print('\n'.join(_body(sa.span_lines(span, root))))
    if base:
        print('\n'.join(at_base(root, anchor, base)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='repository root')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='write nothing; exit 1 on any problem')
    mode.add_argument('--write', nargs='*', metavar='PATH', help='re-pin CHANGED digests, of these cited paths only')
    mode.add_argument('--show', metavar='PATH#ID', help='print where a citation resolves now, and its span')
    parser.add_argument('--base', metavar='REF', help='with --write or --show, also show each span at this git ref')
    args = parser.parse_args(argv)
    if args.check:
        return check(args.root)
    if args.show:
        return show(args.root, args.show, args.base)
    return write(args.root, args.write, args.base)


if __name__ == '__main__':
    sys.exit(main())
