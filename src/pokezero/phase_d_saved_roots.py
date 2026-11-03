"""Outcome-blind fixed metadata selector and full-denominator Phase D readout."""
from collections import Counter
import hashlib
import math

SCHEMA = 'pokezero.phase-d.saved-roots.v1'
TURN_BANDS = ((1, 3), (4, 9), (10, 19), (20, None))


def address(row):
    return (row['cell']['ordinal'], row['owner'], row['decision_index'], row['prefix_sha256'])


def band(turn):
    if type(turn) is not int or turn < 1:
        raise ValueError('literal positive captured turn required')
    return next(i for i, (lo, hi) in enumerate(TURN_BANDS) if turn >= lo and (hi is None or turn <= hi))


def declare(*, count=200, seed=20261103):
    if type(count) is not int or count < 200 or count % len(TURN_BANDS) or type(seed) is not int:
        raise ValueError('at least200 roots equally allocated over four turn bands required')
    return dict(schema=SCHEMA, count=count, seed=seed, turn_bands=TURN_BANDS,
                roots_per_band=count // len(TURN_BANDS), worlds=(16, 32), coverage_bar=.9,
                p95_seconds_bar=30., source='GENUINE_OWNING_CAPTURE_ONLY',
                selection='SHA256(metadata-address,seed) within turn band; no outcome/refusal inputs',
                stop_rule='If coverage<90%, one refusal-driven source batch and one remeasurement; then stop.',
                missing_rule='Preserve all declared slots. No outcome/constructor-based refill or selection.')


def select(rows, declaration):
    """Commit chosen metadata BEFORE opening any constructor results or timing.

    Missing turn-band supply blocks declaration rather than reallocating roots
    to easy early positions. Same capture prefix/address cannot appear twice.
    """
    if declaration != declare(count=declaration['count'], seed=declaration['seed']):
        raise ValueError('original Phase D declaration differs')
    groups = [[] for _ in TURN_BANDS]
    seen = set()
    for row in rows:
        key = address(row)
        if key in seen:
            raise ValueError('duplicate saved root address')
        seen.add(key)
        rank = hashlib.sha256(repr((SCHEMA, declaration['seed'], key)).encode()).digest()
        groups[band(row['turn'])].append((rank, key, row))
    quota = declaration['roots_per_band']
    if any(len(group) < quota for group in groups):
        raise ValueError('insufficient genuine saved roots by turn band: ' + repr(tuple(map(len, groups))))
    # No constructor, outcome, legal-action count or survival field is inspected.
    return tuple(item[2] for group in groups for item in sorted(group)[:quota])


def summarize(roster, records, *, worlds):
    """Missing records remain unavailable full slots; never survivor-renormalize."""
    if worlds not in (16, 32) or len(roster) < 200:
        raise ValueError('declared >=200 roots and16/32 worlds required')
    expected = {address(row): row for row in roster}
    if len(expected) != len(roster):
        raise ValueError('duplicate declared root')
    actual = {}
    for row in records:
        key = tuple(row['address'])
        if key not in expected or key in actual or row['requested_worlds'] != worlds:
            raise ValueError('foreign/duplicate root result or changed world denominator')
        constructed = row['constructed_worlds']
        if type(constructed) is not int or not 0 <= constructed <= worlds:
            raise ValueError('literal constructed-world count required')
        actual[key] = row
    by_turn, refusals, timings = {}, Counter(), []
    for key, source in expected.items():
        turn = source['turn']
        counts = by_turn.setdefault(turn, dict(requested=0, fully_constructed=0, timed=0))
        counts['requested'] += 1
        row = actual.get(key)
        complete = row is not None and row['constructed_worlds'] == worlds
        counts['fully_constructed'] += complete
        if row is None:
            refusals['UNAVAILABLE_RESULT'] += 1
            continue
        for reason in set(row.get('refusals', ())):
            refusals[reason] += 1
        elapsed = row.get('decision_seconds')
        if elapsed is not None:
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0 or not complete:
                raise ValueError('full constructed valid timed decisions required')
            timings.append(elapsed)
            counts['timed'] += 1
    n = len(expected)
    full = sum(v['fully_constructed'] for v in by_turn.values())
    # Untimed declared roots carry infinite decision time: p95 cannot erase
    # hard roots by timing only constructor successes.
    ordered = sorted(timings + [math.inf] * (n - len(timings)))
    p95 = ordered[math.ceil(.95 * n) - 1]
    return dict(requested_roots=n, requested_worlds=worlds, fully_constructed_roots=full,
                full_constructor_coverage=full / n, timed_roots=len(timings),
                decision_p95_seconds=p95 if math.isfinite(p95) else None,
                decision_p95_unavailable=not math.isfinite(p95),
                by_turn=by_turn, refusals_by_root=dict(refusals),
                admitted=full / n >= .9 and p95 <= 30.)
