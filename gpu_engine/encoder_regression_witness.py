"""Independent host fold of the encoder regression's ordered numeric events.

No production arithmetic helper or actual post-state value enters this fold.
Event capture and legacy replay are separate obligations of the trace harness.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np


ENCODER_SHAPES = {
    'trackers.damage_history.taken': (2, 2, 2),
    'trackers.damage_history.dealt': (2, 2, 2),
    'trackers.damage_exact_taken': (2, 6),
    'trackers.damage_exact_dealt': (2, 6),
    "trackers.mon_features['last_damage_taken']": (2, 6),
    "trackers.mon_features['last_damage_dealt']": (2, 6),
    'trackers.public_port.hazards': (2, 2, 2),
}
ENCODER_WITNESS_VERSION = 1
HISTORY_HP = 'trackers.damage_history.hp'
HISTORY_MAXIMUM = 'trackers.damage_history.maximum'
MAPPING = 'trackers.physical_to_public'
NEXT_SLOT = 'trackers.next_slot'
TURN = 'trackers.turn'
TAKEN, DEALT = 'trackers.damage_history.taken', 'trackers.damage_history.dealt'
HAZARDS = 'trackers.public_port.hazards'
SETTLED = (
    ("trackers.mon_features['last_damage_taken']", TAKEN, 0),
    ("trackers.mon_features['last_damage_dealt']", DEALT, 0),
    ('trackers.damage_exact_taken', TAKEN, 1),
    ('trackers.damage_exact_dealt', DEALT, 1),
)

# Payloads carry the leading lane dimension, including scalar hook arguments.
I, B = 'int32', 'bool'
EVENT_FIELDS = {
    'ARRIVE': dict(side=(I, ()), use=(B, ()), initial=(B, ()), active=(I, (2,)),
                   hp=(I, ()), maximum=(I, ()), mapping=(I, (2, 6)), next_slot=(I, (2,))),
    'OBSERVE': dict(use=(B, ()), record_damage=(B, ()), hp=(I, (2,)), maximum=(I, (2,)),
                    history_hp=(I, (2,)), history_maximum=(I, (2,))),
    'DIRECT': dict(side=(I, ()), use=(B, ()), dealt=(I, ()), taken=(I, ()), active=(I, (2,)),
                   hp=(I, ()), maximum=(I, ()), hp_percent=(I, (2, 6)), mapping=(I, (2, 6))),
    'SPIKES': dict(side=(I, ()), use=(B, ()), physical=(I, ()), hp=(I, ()),
                   maximum=(I, ()), new_hp=(I, ())),
    'TURN': dict(use=(B, ()), number=(I, ()), active=(I, (2,)), mapping=(I, (2, 6))),
    'BOUNDARY': dict(before_done=(B, ()), after_done=(B, ()), pending_residual=(B, ()),
                     forced_switch=(B, (2,)), after_turn=(I, ()), incoming_turn=(I, ()), ordinary=(B, ())),
}
EXPECTED_GUARDS = {
    'ARRIVE': dict(arrive=1, feature=2, write=2), 'OBSERVE': dict(observe=1),
    'DIRECT': dict(direct=1), 'SPIKES': {}, 'TURN': dict(settle=1, feature=4, write=4), 'BOUNDARY': {},
}


@dataclass
class FoldResult:
    expected: dict
    errors: tuple
    event_counts: dict


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _plus(a, b):
    return np.add(a, b, dtype=np.float32)


def _minus(a, b):
    return np.subtract(a, b, dtype=np.float32)


def _two_sum(a, b):
    total = _plus(a, b)
    recovered = _minus(total, a)
    error = _plus(_minus(a, _minus(total, recovered)), _minus(b, recovered))
    return total, error


def add_pair(accumulator, increment):
    """Float32-rounded storage recurrence; no high-plus-low reconstruction."""
    total, error = _two_sum(accumulator[..., 0], increment[..., 0])
    correction = _plus(_plus(accumulator[..., 1], increment[..., 1]), error)
    high, low = _two_sum(total, correction)
    return np.stack((high, low), axis=-1)


def ratio_pair(numerator, denominator):
    """Python binary64 quotient split into two float32 limbs, in the HP domain."""
    n, d = int(numerator), int(denominator)
    _require(0 <= n <= np.iinfo(np.int32).max and 0 <= d <= 714, 'ratio outside supported integer HP domain')
    value = min(1.0, max(0.0, float(n) / float(d or 1)))
    high = np.float32(value)
    return np.array((high, np.float32(value - float(high))), dtype=np.float32)


def _hp(hp, maximum):
    h, m = int(hp), int(maximum)
    _require(0 <= m <= 714 and 0 <= h <= max(m, 1), 'invalid HP operand')
    return h, m


def percent(hp, maximum):
    h, m = _hp(hp, maximum)
    positive = max(m, 1)
    _require(100 * h + positive - 1 <= np.iinfo(np.int32).max, 'shown-percent int32 overflow')
    shown = (100 * h + positive - 1) // positive
    return 99 if h < positive and shown == 100 else shown


def _side(value):
    side = int(value)
    _require(side in (0, 1), 'invalid side')
    return side


def _physical(value):
    physical = int(value)
    _require(0 <= physical < 6, 'invalid active physical index')
    return physical


def _array(value, shape, dtype, name):
    out = np.asarray(value)
    _require(out.shape == shape and out.dtype == np.dtype(dtype), f'{name}: shape/dtype mismatch')
    return out


def _same(a, b, message):
    _require(np.array_equal(a, b), message)


def _validate_tape(schema, events, lanes):
    _require(len(schema) == len(events), 'schema/event count mismatch')
    children = Counter()
    for seq, (spec, payload) in enumerate(zip(schema, events)):
        kind = spec['kind'];parent = spec['parent'];fields = EVENT_FIELDS[kind]
        _require(spec['seq'] == seq, 'nonchronological event schema')
        _require(dict(spec['guards']) == EXPECTED_GUARDS[kind], f'{kind}: incomplete writer guards')
        _require(tuple(spec['fields']) == tuple((n, d, s) for n, (d, s) in fields.items()), 'payload schema mismatch')
        _require(set(payload) == set(fields), 'payload field mismatch')
        if parent is not None:
            _require(isinstance(parent, int) and 0 <= parent < seq, 'invalid event parent')
            _require(schema[parent]['kind'] == 'BOUNDARY' and kind in ('OBSERVE', 'TURN'), 'unexpected nested writer')
            children[parent, kind] += 1
        _require(kind != 'TURN' or parent is not None, 'TURN without boundary evidence')
        for name, (dtype, shape) in fields.items():
            _array(payload[name], (lanes,) + shape, dtype, f'{seq}.{name}')
    for seq, spec in enumerate(schema):
        if spec['kind'] == 'BOUNDARY':
            _require(children[seq, 'OBSERVE'] == children[seq, 'TURN'] == 1, 'incomplete boundary sequence')


def fold_encoder(pre, schema, events):
    """Fold batched raw hook operands, returning complete expected arrays.

    Schema/shape defects invalidate every lane. Operand/continuity defects are
    reported per lane; that lane's expected arrays must not earn admission.
    Inputs are copied and never changed. Actual post-state is not an argument.
    """
    expected, errors, counts = {}, [], Counter()
    try:
        lanes = np.asarray(pre[TAKEN]).shape[0]
        for name, shape in ENCODER_SHAPES.items():
            expected[name] = _array(pre[name], (lanes,) + shape, 'float32', name).copy()
            _require(bool(np.isfinite(expected[name]).all()), f'{name}: nonfinite pre-value')
        extra = {name: _array(pre[name], (lanes,) + shape, I, name).copy() for name, shape in
                 ((HISTORY_HP, (2,)), (HISTORY_MAXIMUM, (2,)), (MAPPING, (2, 6)), (NEXT_SLOT, (2,)), (TURN, ()))}
        _validate_tape(schema, events, lanes)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return FoldResult(expected, (dict(lane=None, seq=None, detail=str(error)),), {})
    counts.update(spec['kind'] for spec in schema)
    for lane in range(lanes):
        values = {n: a[lane] for n, a in expected.items()}
        hp, maximum = extra[HISTORY_HP][lane], extra[HISTORY_MAXIMUM][lane]
        mapping, next_slot = extra[MAPPING][lane], extra[NEXT_SLOT][lane]
        turn = int(extra[TURN][lane]);boundaries = {};seq = None
        try:
            _require(bool(((mapping >= -1) & (mapping < 6)).all()), 'invalid pre mapping')
            _require(bool(((next_slot >= 0) & (next_slot <= 6)).all()), 'invalid pre next-slot counter')
            for seq, (spec, payload) in enumerate(zip(schema, events)):
                kind = spec['kind'];e = {n: a[lane] for n, a in payload.items()}
                if kind == 'OBSERVE':
                    _same(e['history_hp'], hp, 'history HP continuity mismatch')
                    _same(e['history_maximum'], maximum, 'history maximum continuity mismatch')
                    for side in (0, 1):
                        previous, denom = _hp(hp[side], maximum[side]);new, new_max = _hp(e['hp'][side], e['maximum'][side])
                        increment = np.stack((ratio_pair(max(0, percent(previous, denom) - percent(new, new_max)), 100),
                                              ratio_pair(max(0, previous - new), denom)))
                        increment = np.multiply(increment, np.bool_(e['record_damage']), dtype=np.float32)
                        if not e['use']:increment = np.zeros((2, 2), np.float32)
                        values[TAKEN][side] = add_pair(values[TAKEN][side], increment)
                    if e['use']:hp[:] = e['hp'];maximum[:] = e['maximum']
                elif kind == 'DIRECT':
                    actor = _side(e['side']);victim = 1 - actor;p = _physical(e['active'][victim])
                    _same(e['mapping'], mapping, 'direct mapping continuity mismatch')
                    h, m = _hp(e['hp'], e['maximum']);slot = int(mapping[victim, p])
                    _require(not e['use'] or slot >= 0, 'direct victim has no public slot')
                    shown = int(e['hp_percent'][victim, max(slot, 0)])
                    _require(0 <= shown <= 100, 'invalid shown HP operand')
                    increment = np.stack((ratio_pair(max(0, shown - percent(h, m)), 100), ratio_pair(e['taken'], m)))
                    if e['use'] and e['taken'] > 0:values[DEALT][actor] = add_pair(values[DEALT][actor], increment)
                elif kind == 'SPIKES':
                    side = _side(e['side']);_physical(e['physical'])
                    old, m = _hp(e['hp'], e['maximum']);new, _ = _hp(e['new_hp'], m)
                    yes = e['use'] and m > 0 and old > new
                    if yes:
                        increment = np.stack((ratio_pair(max(0, percent(old, m) - percent(new, m)), 100),
                                              ratio_pair(old - new, m)))
                        values[HAZARDS][side] = add_pair(values[HAZARDS][side], increment)
                elif kind == 'ARRIVE':
                    side = _side(e['side']);p = _physical(e['active'][side]);h, m = _hp(e['hp'], e['maximum'])
                    _same(e['mapping'], mapping, 'arrival mapping continuity mismatch')
                    _same(e['next_slot'], next_slot, 'arrival next-slot continuity mismatch')
                    old = int(mapping[side, p]);fresh = old < 0;slot = int(next_slot[side]) if fresh else old
                    good = 0 <= slot < 6
                    if e['use']:
                        if good:
                            mapping[side, p] = slot;next_slot[side] += int(fresh)
                            for name, _, _ in SETTLED:values[name][side, slot] = np.float32(0)
                        values[TAKEN][side] = np.float32(0);values[DEALT][side] = np.float32(0)
                        hp[side] = h;maximum[side] = m
                elif kind == 'BOUNDARY':
                    _require(int(e['incoming_turn']) == turn, 'boundary turn continuity mismatch')
                    ready = (not e['before_done'] and not e['after_done'] and not e['pending_residual']
                             and not bool(e['forced_switch'].any()) and turn < int(e['after_turn']) + 1)
                    boundaries[seq] = (bool(ready), int(e['after_turn']) + 1)
                elif kind == 'TURN':
                    ready, number = boundaries[spec['parent']]
                    _require(bool(e['use']) == ready and int(e['number']) == number, 'TURN boundary mask/number mismatch')
                    _same(e['mapping'], mapping, 'TURN mapping continuity mismatch')
                    if e['use']:
                        for side in (0, 1):
                            p = _physical(e['active'][side]);slot = int(mapping[side, p])
                            if slot >= 0:
                                for name, history, channel in SETTLED:
                                    pair = values[history][side, channel]
                                    values[name][side, slot] = _plus(pair[0], pair[1])
                        values[TAKEN][:] = np.float32(0);values[DEALT][:] = np.float32(0)
                        turn = number
        except (KeyError, TypeError, ValueError, IndexError) as error:
            errors.append(dict(lane=lane, seq=seq, detail=str(error)))
    return FoldResult(expected, tuple(errors), dict(counts))


def encoder_array_errors(actual, folded):
    """Validate all seven arrays, including lanes with no baseline digest change.

    The harness must treat any returned entry as fatal before classification,
    even when the original changed-entry set is empty.
    """
    errors = list(folded.errors)
    if any(e['lane'] is None for e in errors):return tuple(errors)
    try:
        _require(set(actual) == set(folded.expected) == set(ENCODER_SHAPES), 'incomplete encoder output arrays')
        lanes = folded.expected[TAKEN].shape[0]
        for name, shape in ENCODER_SHAPES.items():
            a = _array(actual[name], (lanes,) + shape, 'float32', name)
            expected = _array(folded.expected[name], (lanes,) + shape, 'float32', name)
            bits, reference = a.view(np.uint32), expected.view(np.uint32)
            different = (bits != reference) | ~np.isfinite(a) | ~np.isfinite(expected)
            for lane in np.flatnonzero(different.reshape(lanes, -1).any(axis=1)):
                coordinate = tuple(int(x) for x in np.argwhere(different[lane])[0])
                at = (int(lane),) + coordinate
                errors.append(dict(lane=int(lane), leaf=name, coordinate=coordinate,
                                   actual_bits=f'{int(bits[at]):08x}', expected_bits=f'{int(reference[at]):08x}'))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(dict(lane=None, seq=None, detail=str(error)))
    return tuple(errors)


def make_encoder_witness(pre, trace, baseline_digests):
    """Build per-lane admission evidence and fatal errors before classification.

    The caller aligns legacy/baseline digest names and slices all dynamic
    arrays to the same real lanes, excluding padding. No verdict is cached.
    """
    lanes = np.asarray(pre[TAKEN]).shape[0]
    folded = fold_encoder(pre, trace['schema'], trace['events'])
    errors = list(encoder_array_errors(trace['actual'], folded))
    outside, inputs = np.zeros(lanes, bool), np.zeros(lanes, bool)
    try:
        outside = _array(trace['outside_equal'], (lanes,), B, 'outside_equal')
        inputs = _array(trace['event_inputs_equal'], (lanes,), B, 'event_inputs_equal')
        for name in ENCODER_SHAPES:
            _array(trace['legacy'][name], (lanes, 2, 2), 'uint32', f'legacy {name}')
            _array(baseline_digests[name], (lanes, 2, 2), 'uint32', f'baseline {name}')
        for lane in range(lanes):
            if not outside[lane]:errors.append(dict(lane=lane, detail='corrected/legacy output differs outside encoder scope'))
            if not inputs[lane]:errors.append(dict(lane=lane, detail='corrected/legacy event operands differ'))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(dict(lane=None, detail=str(error)))
    global_bad = any(e['lane'] is None for e in errors)
    witnesses = []
    for lane in range(lanes):
        lane_errors = [e for e in errors if e['lane'] is None or e['lane'] == lane]
        witnesses.append(dict(version=ENCODER_WITNESS_VERSION, events_complete=not bool(lane_errors),
            arrays_valid=not bool(lane_errors), outside_equal=bool(outside[lane]),
            actual={n: a[lane] for n, a in trace['actual'].items()},
            expected={n: a[lane] for n, a in folded.expected.items()},
            legacy={} if global_bad else {n: a[lane] for n, a in trace['legacy'].items()},
            base={} if global_bad else {n: a[lane] for n, a in baseline_digests.items()}, errors=lane_errors))
    return dict(lanes=witnesses, errors=errors, event_counts=folded.event_counts, version=ENCODER_WITNESS_VERSION)
