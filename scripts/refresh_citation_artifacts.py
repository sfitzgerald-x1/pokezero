#!/usr/bin/env python3
"""Regenerate C153/C154 source citations and C155 Appendix A from the current tree.

Committed experimental measurements and the recorded pool census are inputs.
This does not run experiments, fetch a pool, mask comparisons, or change verdicts.
Existing tests still re-derive and compare every source-derived field exactly.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]

from gpu_engine import source_citation
import c153_wide_negative_census as c153
import c154_unreachable_readjudication as c154
import test_terminal_disposition_register as c155

C153 = 'reports/artifacts/c153_wide_negative_census.json'
C154 = 'reports/artifacts/c154_unreachable_readjudication.json'
C155 = 'reports/c155_terminal_disposition_register.md'


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():yield from strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:yield from strings(child)


def check_citations(value):
    text = '\n'.join(strings(value))
    if not source_citation.PATTERN.search(text):
        raise ValueError('artifact has no identity-plus-digest citations')
    errors = source_citation.verify_text(text, ROOT)
    if errors:raise ValueError('\n'.join(errors))


def regenerate_c153(document):
    out = copy.deepcopy(document)
    entries, sites = c153.inventory(), c153.emission_sites()
    for name, record in out['verdicts'].items():
        if name in c153.CENSUS_CANNOT_REACH:
            record['census_cannot_reach'] = c153.CENSUS_CANNOT_REACH[name]
        prefix, _, kind = name.partition(':')
        if prefix == 'divergence_class' and kind in c153.STRUCTURAL_DIVERGENCE_CLASSES:
            record['structural_demonstration'] = c153.STRUCTURAL_DIVERGENCE_CLASSES[kind]
        if 'sites' in record:
            derived = c153.granularity(entries[name], sites)
            for field in ('granularity', 'denominator'):
                if record[field] != derived[field]:
                    raise ValueError(f'C153 {name}: {field} changed; citation regeneration cannot re-adjudicate it')
            record['sites'] = derived['sites']
    check_citations(out)
    return out


def regenerate_c154(document):
    out = copy.deepcopy(document)
    out['pool']['heal_subcase_call_graph'] = c154.rust_call_graph(c154.EV, 'heal_subcase', c154.HEAL_SUBCASE_ROOT)
    verdicts = c154.build_verdicts(out['pool'])
    if set(verdicts) != set(document['verdicts']) or c154.correction_counts(verdicts) != document['counts']:
        raise ValueError('C154 inventory/counts changed; citation regeneration cannot re-adjudicate it')
    for key, row in verdicts.items():
        for field in ('row', 'candidate', 'verdict', 'foreclosure', 'ledger_reason_status', 'measurements'):
            if row.get(field) != document['verdicts'][key].get(field):
                raise ValueError(f'C154 {key}: {field} changed')
    out['verdicts'] = verdicts
    check_citations(out)
    return out


def regenerate_c155(text, commit):
    facts = c155.derive()
    marker = '<!-- Citation generation commit: '+commit+' -->'
    text = re.sub(r'\n<!-- Citation generation commit: [0-9a-f]+ -->\n', '\n', text)
    title = '## Appendix A'
    start = text.index('\n', text.index(title))
    text = text[:start+1]+marker+'\n'+text[start+1:]
    rows = text.splitlines(keepends=True)
    first = next(i for i, row in enumerate(rows) if row.strip() == '| key | value |')+2
    last = first
    while last < len(rows) and rows[last].startswith('|'):last += 1
    rows[first:last] = [f'| `{key}` | {value} |\n' for key, value in sorted(facts.items())]
    result = ''.join(rows)
    check_citations(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    generated = {}
    for name, build, indent in ((C153, regenerate_c153, 1), (C154, regenerate_c154, 2)):
        original = json.loads((ROOT/name).read_text())
        value = build(original)
        value['citation_source_commit'] = commit if args.write else original.get('citation_source_commit', commit)
        generated[name] = json.dumps(value, indent=indent, sort_keys=True)+'\n'
    previous = (ROOT/C155).read_text()
    recorded = re.search(r'<!-- Citation generation commit: ([0-9a-f]{40}) -->', previous)
    generated[C155] = regenerate_c155(previous, commit if args.write or recorded is None else recorded[1])
    # Generate and verify every artifact before writing any of them.
    changed = [name for name, value in generated.items() if (ROOT/name).read_text() != value]
    if args.write:
        for name in changed:(ROOT/name).write_text(generated[name], encoding='utf-8')
    for name in changed:print(('regenerated ' if args.write else 'would regenerate ')+name)
    return 0 if args.write or not changed else 1


if __name__ == '__main__':
    raise SystemExit(main())
