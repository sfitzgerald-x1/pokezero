#!/usr/bin/env python3
"""Equivalence of the live reach guard (screen-r2) with #1873's bounds: CPU only, no GPU.

Inputs: --rows, the re-pricer rows (scripts/depthn_rr_reprice.py rows-*.jsonl) of a run on the reach-live tree, each with its
live effect_guard record (rule 'reach': m_own, m_opp, reach.tripped); --decisions, decisions.jsonl of an earlier RR score
(scripts/depthn_rr_score.py at #1873, 977326a0 or later) of the same decisions, holding reach_turn2 = {m_own_lower,
m_own_upper, m_opp_lower, m_opp_upper, tripped} (the bounds) and reach_scope_tripped_lower / _upper. Rows join on (ordinal,
decision_index).

Checks (the live turn-2 arm is the reach M; THRESHOLD from the record, else pokezero.depthn_effect_guard.THRESHOLD):
- lower: on every joined row whose lower bound trips (max(m_own_lower, m_opp_lower) >= THRESHOLD), the live reach trips: 100%;
- upper: on every joined row whose upper bound passes (max(m_own_upper, m_opp_upper) < THRESHOLD), the live reach passes: 100%;
- ambiguous (the bounds straddle THRESHOLD): the live m_own and m_opp each lie within [lower, upper] (to 1e-12), and how
  the live reach decided them;
- counts: rows, decisions, joined, unmatched on either side, rows without a live record or without bounds, unavailable on
  either side (fail closed: not compared), and (informational) the live scope (turn-1 (b) or the reach) against the old
  reach scopes.
Output: the report as JSON on stdout (and --out, a new file). Exit 0 when every check passes, 1 when one fails, 2 when no
row joined.

  python scripts/depthn_guard_equivalence.py --rows NEW/rows-*.jsonl --decisions OLD_SCORE/decisions.jsonl [--out REPORT.json]
"""
import argparse
import collections
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / 'src')):
    if _path not in sys.path:sys.path.insert(0, _path)

from pokezero import depthn_effect_guard as effect_guard  # noqa: E402

SCHEMA = 'pokezero.depthn-guard-equivalence.v1'
TOLERANCE = 1e-12
EXAMPLES = 20


def read_jsonl(paths):
    for path in paths:
        with open(path, 'rb') as stream:
            for raw in stream:
                if raw.strip():yield json.loads(raw)


def compare(rows, decisions):
    """The report (module docstring) over the new rows and the old decisions (iterables of dicts)."""
    old = {};counts = collections.Counter()
    for d in decisions:
        counts['decisions'] += 1;old[(d.get('ordinal'), d.get('decision_index'))] = d
    seen = set();checks = dict(lower=collections.Counter(), upper=collections.Counter(), ambiguous=collections.Counter(), scope=collections.Counter())
    failures = collections.defaultdict(list)
    for row in rows:
        counts['rows'] += 1;key = (row.get('ordinal'), row.get('decision_index'))
        d = old.get(key)
        if d is None:counts['rows_without_decision'] += 1;continue
        seen.add(key);counts['joined'] += 1
        record, bounds = row.get('effect_guard'), d.get('reach_turn2')
        if not isinstance(record, dict) or record.get('rule') != effect_guard.RULE:counts['without_live_record'] += 1;continue
        if not isinstance(bounds, dict):counts['without_bounds'] += 1;continue
        live = record.get('reach') or {};threshold = record.get('threshold', effect_guard.THRESHOLD)
        if live.get('unavailable') or record.get('m_own') is None:counts['live_unavailable'] += 1;continue
        values = [bounds.get(k) for k in ('m_own_lower', 'm_own_upper', 'm_opp_lower', 'm_opp_upper')]
        if any(type(v) not in (int, float) for v in values):counts['bounds_unavailable'] += 1;continue
        counts['compared'] += 1;own_lo, own_hi, opp_lo, opp_hi = values;m_own, m_opp = record['m_own'], record['m_opp']
        tripped = bool(live.get('tripped'));address = dict(ordinal=key[0], decision_index=key[1], m_own=m_own, m_opp=m_opp, bounds=bounds)
        if max(own_lo, opp_lo) >= threshold:
            checks['lower']['rows'] += 1;checks['lower']['agree'] += tripped
            if not tripped and len(failures['lower']) < EXAMPLES:failures['lower'].append(address)
        if max(own_hi, opp_hi) < threshold:
            checks['upper']['rows'] += 1;checks['upper']['agree'] += not tripped
            if tripped and len(failures['upper']) < EXAMPLES:failures['upper'].append(address)
        if bounds.get('tripped') is None:
            within = own_lo - TOLERANCE <= m_own <= own_hi + TOLERANCE and opp_lo - TOLERANCE <= m_opp <= opp_hi + TOLERANCE
            checks['ambiguous']['rows'] += 1;checks['ambiguous']['within'] += within;checks['ambiguous']['live_tripped'] += tripped
            if not within and len(failures['ambiguous']) < EXAMPLES:failures['ambiguous'].append(address)
        scope = bool(record.get('b'))
        for name in ('lower', 'upper'):
            old_scope = d.get(f'reach_scope_tripped_{name}')
            if type(old_scope) is bool:checks['scope'][f'{name}_rows'] += 1;checks['scope'][f'{name}_agree'] += scope == old_scope
    counts['decisions_without_row'] = len(set(old) - seen)
    share = lambda a, b: a / b if b else None
    lower, upper, ambiguous = checks['lower'], checks['upper'], checks['ambiguous']
    out = dict(schema=SCHEMA, tolerance=TOLERANCE, counts=dict(counts),
        lower=dict(rule='the lower bound trips -> the live reach trips (must be 100%)', rows=lower['rows'], agree=lower['agree'],
                   share=share(lower['agree'], lower['rows']), failures=failures['lower']),
        upper=dict(rule='the upper bound passes -> the live reach passes (must be 100%)', rows=upper['rows'], agree=upper['agree'],
                   share=share(upper['agree'], upper['rows']), failures=failures['upper']),
        ambiguous=dict(rule='the bounds straddle the threshold -> the live m_own and m_opp lie within [lower, upper]', rows=ambiguous['rows'],
                       within=ambiguous['within'], live_tripped=ambiguous['live_tripped'], live_passed=ambiguous['rows'] - ambiguous['live_tripped'],
                       failures=failures['ambiguous']),
        scope=dict(note='informational: the live scope (turn-1 (b) or the reach) against the old reach scopes', **dict(checks['scope'])))
    out['passed'] = (lower['agree'] == lower['rows'] and upper['agree'] == upper['rows'] and ambiguous['within'] == ambiguous['rows']
                     and counts['compared'] > 0)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--rows', nargs='+', required=True, help='the reach-live re-pricer rows-*.jsonl')
    parser.add_argument('--decisions', nargs='+', required=True, help="an earlier RR score's decisions.jsonl (#1873's reach_turn2 bounds)")
    parser.add_argument('--out', help='also write the report here (a new file)')
    args = parser.parse_args(argv)
    report = compare(read_jsonl(args.rows), read_jsonl(args.decisions))
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, 'x') as stream:stream.write(text + '\n')
    print(text)
    if not report['counts'].get('joined'):return 2
    return 0 if report['passed'] else 1


if __name__ == '__main__':sys.exit(main())
