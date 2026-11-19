"""Project closed Native move restrictions and volatile state."""

from .fresh200_native_initial import Vocabulary, _boolean, _field, _integer, _layout, _need, _typed_projection
from .fresh200_native_identity import to_id
from .fresh200_native_ledger import fields, items, member, pokemon
from .fresh200_native_material import MATERIAL_FIELDS, _last_move
from .fresh200_native_projector import bound_boundary


CONDITION_FIELDS = frozenset((
    'disabled', 'choice_move', 'encore_move', 'encore_counter', 'charge_move', 'charge_counter',
    'volatile', 'confusion_counter', 'protect_counter', 'substitute_hp', 'recharge_counter',
    'perish_counter', 'yawn_counter', 'partial_trap_counter', 'partial_trap_source', 'trap_source',
))
BITS = dict(flinch=1, flashfire=2, leechseed=4, ingrain=8, focuspunch=16, destinybond=64, attract=128)
COUNTERS = dict(confusion_counter=('confusion', 'time'), substitute_hp=('substitute', 'hp'),
                recharge_counter=('mustrecharge', 'duration'), perish_counter=('perishsong', 'duration'),
                yawn_counter=('yawn', 'duration'), partial_trap_counter=('partiallytrapped', 'duration'))
REPRESENTED = (
    BITS.keys() | {condition for condition, _ in COUNTERS.values()}
    | {'choicelock', 'encore', 'twoturnmove', 'solarbeam', 'stall', 'protect', 'endure', 'trapped', 'trapper'}
)


def _counter(conditions, condition, name):
    if condition not in conditions:
        return 0
    value = _integer(_field(conditions[condition], name))
    _need(value > 0, f'active Native {condition} lacks a positive {name}')
    return value


def _opponent(party, side, reference):
    owner, index = party.resolve_pokemon(reference)
    _need(owner == 1 - side, 'Native volatile has an unexpected same-side source')
    return owner, index


def _linked_source(party, side, original, conditions, name, companion):
    if name not in conditions:
        return -1
    state = conditions[name]
    links = _field(state, 'linkedPokemon')
    _need(type(links) is list and len(links) == 1 and state['linkedStatus'] == companion,
          'Native linked trap requires one exact counterpart')
    foe, source = _opponent(party, side, links[0])
    reciprocal = _field(_field(party.mon(foe, source), 'volatiles'), companion)
    targets = _field(reciprocal, 'linkedPokemon')
    _need(type(targets) is list and len(targets) == 1 and reciprocal['linkedStatus'] == name
          and party.resolve_pokemon(targets[0]) == (side, original), 'Native linked trap is not reciprocal')
    return source


def _disable_condition(effect, live, item):
    name = effect['id']
    if name in ('choicelock', 'encore') and name in live:
        return
    if name == 'choiceband' and item == 'choiceband' and 'choicelock' in live:
        source = _field(fields(live['choicelock']), 'sourceEffect')
        _need(source['kind'] == 'Effect' and source['effectType'] == effect['effect_type'] == 'Item'
              and source['id'] == name and source['object'] == effect['object'],
              'Native Choice Band disable source differs from its live lock')
        return
    raise ValueError(f'unrepresented Native disable cause {name!r} or missing live lock')


def _disabled(side, original, slots, live, item, history, phase, move_request):
    disabled = []
    for index, slot in enumerate(slots):
        slot_values = fields(slot)
        flag = _field(slot_values, 'disabled')
        _need(type(flag) is bool or flag == 'hidden', 'literal Native move disable flag required')
        tracked = [r for r in history['disable_causes']
                   if r['actor'] == [side, original] and r['slot_object'] == slot['object']]
        if not flag:
            disabled.append(False)
            continue
        _need(len(tracked) == 1 and tracked[0]['causes'], f'disabled Native move lacks its actual slot causes: '
              f'p{side+1} original slot {original}, move slot {index}, source={slot_values.get("disabledSource")!r}')
        causes = tracked[0]['causes']
        _need(all(cause['physical_slot'] == index and cause['sequence'] <= phase.snapshot_sequence and cause['effect']
                  for cause in causes), 'Native disable cause does not bind the closed slot')
        for cause in causes:
            name = cause['effect']['id']
            lock = 'choicelock' if name == 'choiceband' else name
            retired = cause.get('clear_volatile_sequence')
            if not move_request and lock not in live and lock in ('choicelock', 'encore'):
                if (type(cause.get('lock_object')) is int and type(retired) is int
                        and cause['sequence'] < retired <= phase.snapshot_sequence):
                    continue
            _disable_condition(cause['effect'], live, item)
        disabled.append(False)
    return disabled + [False] * (4 - len(disabled))


def _proof_slot(vocab, roster, slots, record, move):
    lane = _integer(_field(record, 'physical_slot'))
    _need(0 <= lane < len(slots) and slots[lane]['object'] == record['slot_object']
          and fields(slots[lane])['id'] == move and roster[lane] > 0,
          'Native move condition points to a different physical slot')
    code = roster[lane]
    if move != 'hiddenpower':
        _need(code == vocab.id('moves', move), 'Native condition and typed move identities differ')
    return code


def _locks(vocab, side, original, conditions, live, slots, roster, history, sequence):
    choice = encore = charge = 0
    actor = [side, original]
    if 'choicelock' in conditions:
        choice = _last_move(vocab, _field(conditions['choicelock'], 'move'), roster, history['events'], side, original)
    if 'encore' in conditions:
        token = live['encore']['object']
        matches = [r for r in history['encore_slots'] if r['actor'] == actor and r['state_object'] == token]
        _need(len(matches) == 1 and matches[0]['sequence'] <= sequence, 'Native Encore lacks its live slot witness')
        encore = _proof_slot(vocab, roster, slots, matches[0], _field(conditions['encore'], 'move'))
    _need(('twoturnmove' in conditions) == ('solarbeam' in conditions), 'Native charge companion condition is missing')
    if 'twoturnmove' in conditions:
        token = live['twoturnmove']['object']
        move = _field(conditions['twoturnmove'], 'move')
        matches = [r for r in history['charge_records'] if r['actor'] == actor and r['state_object'] == token]
        _need(move == 'solarbeam' and len(matches) == 1 and matches[0]['move_id'] == move,
              'Native charge lacks its represented move and actual slot witness')
        charge = _proof_slot(vocab, roster, slots, matches[0], move)
    return dict(choice_move=choice, encore_move=encore, encore_counter=_counter(conditions, 'encore', 'duration'),
                charge_move=charge, charge_counter=_counter(conditions, 'twoturnmove', 'duration'))


def _protect_counter(phase, history, side, original, active, conditions):
    counter = _counter(conditions, 'stall', 'counter')
    effects = {'protect', 'endure'} & conditions.keys()
    if effects:
        _need(len(effects) == 1 and original == active and counter > 0 and phase.pending_residual,
              'current Native protect/endure lacks its suspended stall state')
    if counter and phase.pending_residual:
        refreshed = [r for r in history['stall_refreshed']
                     if r['actor'] == [side, original] and r['epoch'] == history['phase_epoch']]
        if not refreshed:
            _need(not effects and original == active and history['old_stall'] is not None
                  and history['old_stall'][side] == counter, 'Native old stall counter lacks its frozen move-phase value')
            return 0
        _need(len(refreshed) == 1 and refreshed[0]['sequence'] <= phase.snapshot_sequence,
              'Native stall refresh is outside the closed phase')
    return counter


def _validate_substitute_break_marker(party, vocab, side, original, serialized, observed):
    # In this pinned Gen 3 ruleset, breaking a Substitute can leave the
    # inherited Gen 4 marker. Its only ability reader checks U-turn. An
    # actual Substitute still has its own HP projection.
    marker = fields(observed)
    keys = {'id', 'name', 'target', 'effectOrder', 'source', 'sourceSlot', 'sourceEffect', 'move'}
    _need(type(serialized) is dict and set(serialized) == set(marker) == keys,
          'unexpected Native Substitute break marker fields')
    move = serialized['move']
    move_ids = vocab.ids['moves']
    _need(type(move) is str and move == marker['move'] and move != 'uturn'
          and (move in move_ids or (move == 'hiddenpower' and any(key.startswith('hiddenpower') for key in move_ids))),
          'Native Substitute break marker has an unsupported or inconsistent move')
    source = _opponent(party, side, serialized['source'])
    _need(serialized['id'] == marker['id'] == serialized['name'] == marker['name'] == 'substitutebroken'
          and _integer(serialized['effectOrder']) == _integer(marker['effectOrder']) >= 0
          and party.resolve_pokemon(serialized['target']) == pokemon(marker['target']) == (side, original)
          and pokemon(marker['source']) == source and serialized['sourceSlot'] == marker['sourceSlot'] == f'p{source[0]+1}a',
          'Native Substitute break marker differs from the admitted source')
    effect = marker['sourceEffect']
    _need(serialized['sourceEffect'] == '[Condition:substitute]' and type(effect) is dict
          and set(effect) == {'kind', 'effectType', 'id', 'object'} and effect['kind'] == 'Effect'
          and effect['effectType'] == 'Condition' and effect['id'] == 'substitute' and _integer(effect['object']) >= 0,
          'Native Substitute break marker has a different originating condition')


def _member_conditions(party, observed, boundary, vocab, side, original, roster):
    phase, history = boundary.phase, boundary.history
    mon = party.mon(side, original)
    cache = member(observed, (side, original))
    conditions = _field(mon, 'volatiles')
    live = fields(_field(cache, 'volatiles'))
    _need(type(conditions) is dict and conditions.keys() == live.keys(),
          'Native observed volatile roster differs from its serialized state')
    unrepresented = set(conditions) - REPRESENTED - {'substitutebroken'}
    _need(not unrepresented, f'unrepresented Native volatiles: {sorted(unrepresented)}')
    if 'substitutebroken' in conditions:
        _need(party.battle['formatid'] == 'gen3customgame', 'Native Substitute break marker is specific to the admitted format')
        _validate_substitute_break_marker(party, vocab, side, original, conditions['substitutebroken'], live['substitutebroken'])
    mask = sum(bit for name, bit in BITS.items() if name in conditions)
    if 'focuspunch' in conditions and _boolean(conditions['focuspunch'].get('lostFocus', False)):
        mask |= 32
    for source_condition in ('attract', 'leechseed'):
        if source_condition in conditions:
            _opponent(party, side, _field(conditions[source_condition], 'source'))
    partial_source = -1
    if 'partiallytrapped' in conditions:
        _, partial_source = _opponent(party, side, _field(conditions['partiallytrapped'], 'source'))
    _linked_source(party, side, original, conditions, 'trapper', 'trapped')
    trap_source = _linked_source(party, side, original, conditions, 'trapped', 'trapper')
    slots = items(_field(cache, 'moveSlots'))
    _need(1 <= len(slots) <= 4 and len(slots) == sum(code > 0 for code in roster),
          'closed Native slot roster differs from the typed identity roster')
    request = boundary.snapshot['boundaryRequests'].get(f'p{side+1}') or {}
    move_request = original == party.active(side) and phase.requested[side] and bool(request.get('active'))
    return dict(
        volatile=mask, trap_source=trap_source, partial_trap_source=partial_source,
        protect_counter=_protect_counter(phase, history, side, original, party.active(side), conditions),
        disabled=_disabled(side, original, slots, live, mon['item'], history, phase, move_request),
        **_locks(vocab, side, original, conditions, live, slots, roster, history, phase.snapshot_sequence),
        **{name: _counter(conditions, *address) for name, address in COUNTERS.items()},
    )


def project_conditions(boundary, *, original_species, vocabulary, material):
    """Derive condition leaves after binding the Native material slot roster."""
    _need(set(material) == MATERIAL_FIELDS, 'complete typed Native material fields required')
    party, observed = bound_boundary(boundary, original_species)
    vocab = Vocabulary(vocabulary)
    moves = material['moves']
    _need(moves.shape == (1, 2, 6, 4), 'typed Native material move roster has the wrong shape')
    members = [[_member_conditions(party, observed, boundary, vocab, side, original, moves[0, side, original].tolist())
                for original in range(6)] for side in (0, 1)]
    values = {name: [[mon[name] for mon in side] for side in members] for name in members[0][0]}
    layout = _layout()
    return _typed_projection(values, {name: spec for name, spec in layout.items() if name in CONDITION_FIELDS})


def _represented_traps(party, side, original, conditions):
    foe = party.active(1 - side)
    mon = party.mon(side, original)
    opponent = party.mon(1 - side, foe)
    live_foe = _integer(opponent['hp']) > 0
    effects = set()
    types = {to_id(value) for value in mon['types']}
    ability = opponent['ability']
    if live_foe and ability == 'shadowtag':
        effects.add('shadowtag')
    if live_foe and ability == 'arenatrap' and 'flying' not in types and mon['ability'] != 'levitate':
        effects.add('arenatrap')
    if live_foe and ability == 'magnetpull' and 'steel' in types:
        effects.add('magnetpull')
    if conditions['volatile'][0, side, original] & BITS['ingrain']:
        effects.add('ingrain')
    if live_foe and conditions['trap_source'][0, side, original] == foe:
        effects.add('trapped')
    if live_foe and conditions['partial_trap_source'][0, side, original] == foe:
        effects.add('partiallytrapped')
    if conditions['charge_move'][0, side, original] > 0:
        effects.update(('solarbeam', 'twoturnmove'))
    if conditions['recharge_counter'][0, side, original] > 0:
        effects.add('mustrecharge')
    return effects


def project_trapped(boundary, *, original_species, conditions):
    """Account for the Native combined trap cache before deriving residual traps."""
    _need(set(conditions) == CONDITION_FIELDS, 'typed Native conditions required before trapped projection')
    party, observed = bound_boundary(boundary, original_species)
    for side in (0, 1):
        original = party.active(side)
        actor = [side, original]
        live = member(observed, (side, original))
        cache = _field(live, 'trapped')
        _need(type(cache) is bool or cache == 'hidden', 'literal Native trap cache required')
        represented = _represented_traps(party, side, original, conditions)
        if cache:
            recorded = [cause for row in boundary.history['trap_causes'] if row['actor'] == actor
                        for cause in row['causes']]
            _need(represented and all(cause['effect'] and cause['effect']['id'] in represented
                                     and cause['sequence'] <= boundary.phase.snapshot_sequence for cause in recorded),
                  f'Native trap has an unexplained contribution for p{side+1} original slot {original}')
            # Shadow Tag writes the cache directly. The two hard-lock paths
            # are bound to the actual lock return. Other true writes require
            # the recorded tryTrap cause.
            hard = {r['hard'] for r in boundary.history['lock_results'] if r['actor'] == actor}
            direct = 'shadowtag' in represented
            locked = (('recharge' in hard and 'mustrecharge' in represented)
                      or ('solarbeam' in hard and 'solarbeam' in represented))
            _need(recorded or direct or locked, 'Native combined trap cache lacks an observed or direct tracked cause')
        relation = boundary.relations[side]
        if relation['kind'] == 'inactive':
            continue
        forced = relation['kind'] == 'forced-replacement'
        mask = [party.mon(side, target)['hp'] > 0 and (forced or not represented)
                for target in range(6) if target != original]
        _need(mask == relation['physical_mask'][4:], 'derived trap causes disagree with the full Native switch lanes')
    return _typed_projection({'trapped': [False, False]}, {'trapped': _layout()['trapped']})
