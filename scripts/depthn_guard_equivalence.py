#!/usr/bin/env python3
"""Equivalence of the live reach guard (screen-r2) with #1873's bounds: CPU only, no GPU.

Inputs: --rows, the re-pricer rows (scripts/depthn_rr_reprice.py rows-*.jsonl) of a run on the reach-live tree, each with its
live effect_guard record (rule 'reach': m_own, m_opp, reach.tripped); --decisions, decisions.jsonl of an earlier RR score
(scripts/depthn_rr_score.py at #1873, 977326a0 or later) of the same decisions, holding reach_turn2 = {m_own_lower,
m_own_upper, m_opp_lower, m_opp_upper, tripped} (the bounds) and reach_scope_tripped_lower / _upper. Rows join on (ordinal,
decision_index); a joined pair must agree on battle_id and owner (the seat when either lacks the owner), else the join is
wrong: a hard error (exit 3), every mismatch listed. A repeated key keeps its first row / decision (counted).

Availability: a row's live side is available when it carries a guard record whose turn-2 capture is (turn2_available, no
error: an unavailable capture or an error record fails closed and is not compared); a decision's when reach_turn2 holds its
four bounds as numbers. A joined row available on both sides is compared when the live record is the reach rule's with
numbers m_own and m_opp; any other is counted under `uncompared` by cause.

Checks (the live turn-2 arm is the reach M; THRESHOLD from the record, else pokezero.depthn_effect_guard.THRESHOLD):
- lower: on every compared row whose lower bound trips (max(m_own_lower, m_opp_lower) >= THRESHOLD), the live reach trips: 100%;
- upper: on every compared row whose upper bound passes (max(m_own_upper, m_opp_upper) < THRESHOLD), the live reach passes: 100%;
- bounds: on every compared row the live m_own and m_opp each lie within their [lower, upper] to TOLERANCE (1e-12); every row
  outside is listed with its signed deltas (M minus the bound it crossed), largest first: a re-price that is not
  bit-reproducible fails loudly;
- coverage: compared >= COVERAGE_FLOOR (99%) of the joined rows available on both sides, the share reported;
- compared > 0;
- informational: the ambiguous rows (the bounds straddle THRESHOLD) and how the live reach decided them; counts (rows,
  decisions, joined, unmatched on either side, without a record or unavailable on either side); the live scope (turn-1 (b)
  or the reach) against the old reach scopes.
Output: the report as JSON on stdout (and --out, a new file); a failure's summary also on stderr. Exit 0 when every check
passes, 1 when one fails, 2 when no row joined, 3 on a join mismatch.

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

SCHEMA = 'pokezero.depthn-guard-equivalence.v2'
TOLERANCE = 1e-12
COVERAGE_FLOOR = .99
EXAMPLES = 20
BOUNDS = ('m_own_lower', 'm_own_upper', 'm_opp_lower', 'm_opp_upper')


class JoinMismatch(ValueError):
    """Joined rows whose battle_id or owner (seat) disagree with their decision's: the join is wrong."""

    def __init__(self, mismatches):
        self.mismatches = mismatches
        super().__init__(f'{len(mismatches)} joined row(s) disagree with their decision on battle_id or owner/seat; first: {mismatches[:3]}')


def read_jsonl(paths):
    for path in paths:
        with open(path, 'rb') as stream:
            for raw in stream:
                if raw.strip():yield json.loads(raw)


def identity_mismatch(row, d):
    """The fields on which a joined row and decision disagree (battle_id; owner, else the seat when either lacks the owner);
    a field missing on either side is a mismatch (the join cannot be verified)."""
    bad = {}
    if row.get('battle_id') is None or row.get('battle_id') != d.get('battle_id'):bad['battle_id'] = [row.get('battle_id'), d.get('battle_id')]
    field = 'owner' if row.get('owner') is not None and d.get('owner') is not None else 'seat'
    if row.get(field) is None or row.get(field) != d.get(field):bad[field] = [row.get(field), d.get(field)]
    return bad


def live_available(record):
    """A guard record whose turn-2 capture is available (not an unavailable capture, not an error record)."""
    return isinstance(record, dict) and record.get('turn2_available') is True and record.get('error') is None


def bounds_available(bounds):
    return isinstance(bounds, dict) and all(type(bounds.get(k)) in (int, float) for k in BOUNDS)


def outside(m, lower, upper):
    """M's signed distance past [lower, upper] (M - the crossed bound) beyond TOLERANCE, else None."""
    if m < lower - TOLERANCE:return m - lower
    if m > upper + TOLERANCE:return m - upper
    return None


def compare(rows, decisions):
    """The report (module docstring) over the new rows and the old decisions (iterables of dicts). Raises JoinMismatch."""
    old = {};counts = collections.Counter();uncompared = collections.Counter()
    for d in decisions:
        counts['decisions'] += 1;key = (d.get('ordinal'), d.get('decision_index'))
        if key in old:counts['duplicate_decisions'] += 1;continue
        old[key] = d
    seen = set();mismatches = [];checks = dict(lower=collections.Counter(), upper=collections.Counter(), ambiguous=collections.Counter(), scope=collections.Counter())
    failures = collections.defaultdict(list);out_of_bounds = [];within = 0
    for row in rows:
        counts['rows'] += 1;key = (row.get('ordinal'), row.get('decision_index'))
        d = old.get(key)
        if d is None:counts['rows_without_decision'] += 1;continue
        if key in seen:counts['duplicate_rows'] += 1;continue
        seen.add(key);counts['joined'] += 1
        bad = identity_mismatch(row, d)
        if bad:mismatches.append(dict(ordinal=key[0], decision_index=key[1], **bad));continue
        record, bounds = row.get('effect_guard'), d.get('reach_turn2')
        live_ok, old_ok = live_available(record), bounds_available(bounds)
        if not live_ok:counts['live_unavailable' if isinstance(record, dict) else 'live_without_record'] += 1
        if not old_ok:counts['bounds_unavailable' if isinstance(bounds, dict) else 'decision_without_bounds'] += 1
        if not (live_ok and old_ok):continue
        counts['available_both'] += 1
        if record.get('rule') != effect_guard.RULE:uncompared['live_rule_not_reach'] += 1;continue
        live = record.get('reach') or {};threshold = record.get('threshold', effect_guard.THRESHOLD)
        m_own, m_opp = record.get('m_own'), record.get('m_opp')
        if live.get('unavailable') or type(m_own) not in (int, float) or type(m_opp) not in (int, float):uncompared['live_m_missing'] += 1;continue
        counts['compared'] += 1;own_lo, own_hi, opp_lo, opp_hi = (bounds[k] for k in BOUNDS)
        tripped = bool(live.get('tripped'))
        address = dict(ordinal=key[0], decision_index=key[1], battle_id=row.get('battle_id'), owner=row.get('owner'), m_own=m_own, m_opp=m_opp,
                       bounds={k: bounds[k] for k in BOUNDS})
        if max(own_lo, opp_lo) >= threshold:
            checks['lower']['rows'] += 1;checks['lower']['agree'] += tripped
            if not tripped and len(failures['lower']) < EXAMPLES:failures['lower'].append(address)
        if max(own_hi, opp_hi) < threshold:
            checks['upper']['rows'] += 1;checks['upper']['agree'] += not tripped
            if tripped and len(failures['upper']) < EXAMPLES:failures['upper'].append(address)
        deltas = {side: delta for side, delta in (('own', outside(m_own, own_lo, own_hi)), ('opp', outside(m_opp, opp_lo, opp_hi))) if delta is not None}
        if deltas:out_of_bounds.append(dict(address, deltas=deltas, max_abs_delta=max(abs(v) for v in deltas.values())))
        else:within += 1
        if bounds.get('tripped') is None:
            checks['ambiguous']['rows'] += 1;checks['ambiguous']['live_tripped'] += tripped
        scope = bool(record.get('b'))
        for name in ('lower', 'upper'):
            old_scope = d.get(f'reach_scope_tripped_{name}')
            if type(old_scope) is bool:checks['scope'][f'{name}_rows'] += 1;checks['scope'][f'{name}_agree'] += scope == old_scope
    if mismatches:raise JoinMismatch(mismatches)
    counts['decisions_without_row'] = len(set(old) - seen)
    share = lambda a, b: a / b if b else None
    lower, upper, ambiguous = checks['lower'], checks['upper'], checks['ambiguous']
    out_of_bounds.sort(key=lambda x: (-x['max_abs_delta'], x['ordinal'], x['decision_index']))
    coverage = share(counts['compared'], counts['available_both'])
    out = dict(schema=SCHEMA, tolerance=TOLERANCE, counts=dict(counts),
        coverage=dict(rule=f'compared >= {COVERAGE_FLOOR:.0%} of the joined rows available on both sides', floor=COVERAGE_FLOOR,
                      compared=counts['compared'], available_both=counts['available_both'], share=coverage, uncompared=dict(uncompared),
                      passed=coverage is not None and coverage >= COVERAGE_FLOOR),
        lower=dict(rule='the lower bound trips -> the live reach trips (must be 100%)', rows=lower['rows'], agree=lower['agree'],
                   share=share(lower['agree'], lower['rows']), failures=failures['lower']),
        upper=dict(rule='the upper bound passes -> the live reach passes (must be 100%)', rows=upper['rows'], agree=upper['agree'],
                   share=share(upper['agree'], upper['rows']), failures=failures['upper']),
        bounds=dict(rule=f'every compared row: the live m_own and m_opp lie within [lower, upper] (to {TOLERANCE:g})', rows=counts['compared'],
                    within=within, outside=len(out_of_bounds), max_abs_delta=out_of_bounds[0]['max_abs_delta'] if out_of_bounds else None,
                    rows_outside=out_of_bounds),
        ambiguous=dict(note='informational: the bounds straddle the threshold; how the live reach decided them', rows=ambiguous['rows'],
                       live_tripped=ambiguous['live_tripped'], live_passed=ambiguous['rows'] - ambiguous['live_tripped']),
        scope=dict(note='informational: the live scope (turn-1 (b) or the reach) against the old reach scopes', **dict(checks['scope'])))
    out['failed'] = [name for name, ok in (('compared', counts['compared'] > 0), ('coverage', out['coverage']['passed']),
        ('lower', lower['agree'] == lower['rows']), ('upper', upper['agree'] == upper['rows']), ('bounds', not out_of_bounds)) if not ok]
    out['passed'] = not out['failed']
    return out


def summary(report):
    """A failing report's one-paragraph summary (stderr)."""
    lines = [f"FAILED: {', '.join(report['failed'])}"]
    c = report['coverage'];lines.append(f"compared {c['compared']} of {c['available_both']} available on both sides ({c['share']})")
    for name in ('lower', 'upper'):
        if name in report['failed']:lines.append(f"{name}: {report[name]['agree']} of {report[name]['rows']} agree")
    b = report['bounds']
    if b['outside']:
        lines.append(f"bounds: {b['outside']} row(s) outside [lower, upper] by more than {TOLERANCE:g}; max |delta| {b['max_abs_delta']!r}")
        lines += [f"  ordinal {x['ordinal']} decision {x['decision_index']}: {x['deltas']}" for x in b['rows_outside'][:EXAMPLES]]
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--rows', nargs='+', required=True, help='the reach-live re-pricer rows-*.jsonl')
    parser.add_argument('--decisions', nargs='+', required=True, help="an earlier RR score's decisions.jsonl (#1873's reach_turn2 bounds)")
    parser.add_argument('--out', help='also write the report here (a new file)')
    args = parser.parse_args(argv)
    try:report = compare(read_jsonl(args.rows), read_jsonl(args.decisions));code = None
    except JoinMismatch as error:
        report = dict(schema=SCHEMA, passed=False, failed=['join'], error='join_mismatch', mismatches=error.mismatches);code = 3
        print(f'JOIN MISMATCH: {error}', file=sys.stderr)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, 'x') as stream:stream.write(text + '\n')
    print(text)
    if code is not None:return code
    if not report['counts'].get('joined'):return 2
    if not report['passed']:print(summary(report), file=sys.stderr)
    return 0 if report['passed'] else 1


if __name__ == '__main__':sys.exit(main())
