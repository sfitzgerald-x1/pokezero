"""Project Native identities and Transform saves using the sealed vocabulary."""

from .fresh200_native_identity import reference_identifier, to_id
from .fresh200_native_initial import STATS, Vocabulary, _boolean, _field, _layout, _need, _typed_projection
from .fresh200_native_ledger import fields, items, member
from .fresh200_native_projector import bound_boundary
from .fresh200_witness_audit import _same


MATERIAL_FIELDS = frozenset((
    'species', 'hidden_power_type', 'hidden_power_power', 'types', 'base_types',
    'moves', 'last_move', 'item', 'ability', 'base_ability',
    'transformed', 'transform_species', 'transform_stats', 'transform_moves',
    'transform_pp', 'transform_hidden_power',
))
_CLEAR_TRANSFORM = dict(transformed=False, transform_species=0, transform_stats=(0,) * 5,
                        transform_moves=(0,) * 4, transform_pp=(0,) * 4, transform_hidden_power=(0, 0))


def _original_moves(vocab, original):
    moves = _field(original, 'moves')
    _need(type(moves) is list and 1 <= len(moves) <= 4, 'sealed original move roster required')
    return [vocab.id('moves', move) for move in moves] + [0] * (4 - len(moves))


def _type_slots(vocab, source):
    _need(type(source) is list and 1 <= len(source) <= 2, 'exact ordered Native type slots required')
    return [vocab.id('types', value) for value in source] + [0] * (2 - len(source))


def _last_move(vocab, value, roster, events, side, original):
    move = reference_identifier(value, 'Move', allow_none=True)
    if move is None:
        return 0
    if move != 'hiddenpower':
        return vocab.id('moves', move)
    slots = [slot for slot, code in enumerate(roster) if code and to_id(vocab.values['moves'][code]).startswith('hiddenpower')]
    candidates = {roster[slot] for slot in slots}
    if len(candidates) == 1:
        return next(iter(candidates))
    matches = [event for event in events if event['kind'] == 'MOVE_USED'
               and event['side'] == side and event['original_actor'] == original]
    _need(matches and matches[-1]['move_id'] == move and matches[-1]['physical_slot'] in slots,
          'Native last Hidden Power lacks a bound physical dispatch slot')
    return roster[matches[-1]['physical_slot']]


def _transform(vocab, coordinates, observed, mon, row, history, teams, sequence):
    side, original = coordinates
    live = member(observed, coordinates)
    records = [r for r in history['transform_records'] if r['actor'] == [side, original]]
    _need(len(records) == 1 and records[0]['sequence'] <= sequence, 'Native Transform lacks its latest successful copy')
    record = records[0]
    donor_side, donor_index = record['donor']
    _need(donor_side == 1 - side and type(donor_index) is int and 0 <= donor_index < 6
          and _same(live['species'], record['donor_effective_species']),
          'Native Transform copy does not match the observed donor')
    base = _field(mon, 'baseMoveSlots')
    base_moves = vocab.move_slots(base, _original_moves(vocab, row))
    saved_type = vocab.id('types', _field(mon, 'baseHpType'))
    saved_power = _field(mon, 'baseHpPower')
    saved_pp = [_field(slot, 'pp') for slot in base]
    stats = [_field(_field(mon, 'baseStoredStats'), name) for name in STATS]
    saved = record['saved']
    live_slots = items(live['baseMoveSlots'])
    event_slots = items(saved['baseMoveSlots'])
    _need(len(base) == len(live_slots) == len(event_slots)
          and all(actual['object'] == expected['object'] and fields(actual)['pp'] == fields(expected)['pp'] == pp
                  for actual, expected, pp in zip(live_slots, event_slots, saved_pp, strict=True))
          and _same(live['baseStoredStats'], saved['baseStoredStats'])
          and stats == [_field(fields(saved['baseStoredStats']), name) for name in STATS]
          and saved['baseHpType'] == mon['baseHpType'] and saved['baseHpPower'] == saved_power,
          'Native Transform base restoration state differs from the saved copy event')
    donor = teams[donor_side][donor_index]
    restoration = dict(transformed=True, transform_species=vocab.id('species', donor['species']),
                       transform_stats=stats, transform_moves=base_moves,
                       transform_pp=saved_pp + [0] * (4 - len(saved_pp)), transform_hidden_power=(saved_type, saved_power))
    return _original_moves(vocab, donor), restoration


def project_material(boundary, *, teams, vocabulary):
    """Bind original and copied slots before translating their categorical IDs."""
    species = tuple(tuple(mon['species'] for mon in team) for team in teams)
    party, observed = bound_boundary(boundary, species)
    vocab = Vocabulary(vocabulary)
    history = boundary.history
    result, restore = [[], []], []
    for side in (0, 1):
        active, side_restore = party.active(side), _CLEAR_TRANSFORM
        for original in range(6):
            row, mon = teams[side][original], party.mon(side, original)
            _need(not _field(mon, 'addedType'), 'unrepresented Native added type')
            transformed = _boolean(_field(mon, 'transformed'))
            if transformed:
                _need(original == active, 'unrepresented transformed bench member')
                bound, side_restore = _transform(vocab, (side, original), observed, mon, row, history, teams,
                                                  boundary.phase.snapshot_sequence)
            else:
                bound = _original_moves(vocab, row)
            moves = vocab.move_slots(_field(mon, 'moveSlots'), bound)
            base_ability = vocab.id('abilities', _field(mon, 'baseAbility'))
            _need(base_ability == vocab.id('abilities', row['ability']), 'Native base ability differs from the original set')
            result[side].append(dict(
                species=vocab.id('species', _field(mon['set'], 'species')),
                types=_type_slots(vocab, _field(mon, 'types')),
                base_types=_type_slots(vocab, _field(mon, 'baseTypes')),
                hidden_power_type=vocab.id('types', _field(mon, 'hpType')),
                hidden_power_power=_field(mon, 'hpPower'),
                item=vocab.id('items', _field(mon, 'item'), empty=True),
                ability=vocab.id('abilities', _field(mon, 'ability')), base_ability=base_ability,
                moves=moves, last_move=_last_move(vocab, _field(mon, 'lastMove'), moves, history['events'], side, original),
            ))
        restore.append(side_restore)
    values = {key: [[mon[key] for mon in side] for side in result] for key in result[0][0]}
    values.update({key: [side[key] for side in restore] for key in _CLEAR_TRANSFORM})
    layout = _layout()
    return _typed_projection(values, {name: spec for name, spec in layout.items() if name in MATERIAL_FIELDS})
