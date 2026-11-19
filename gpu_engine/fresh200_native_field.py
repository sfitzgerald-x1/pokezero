"""Project side conditions, field conditions and PRE result flags."""

from .fresh200_native_initial import WEATHER, _boolean, _field, _integer, _layout, _need, _typed_projection
from .fresh200_native_ledger import fields, member, pokemon
from .fresh200_native_projector import bound_boundary


FIELD_LEAVES = frozenset((
    'entry_pending', 'forced_switch', 'baton_pass', 'spikes', 'wish_counter',
    'weather', 'weather_counter', 'done', 'winner',
))
PERMANENT_WEATHER = {'raindance': 'drizzle', 'sunnyday': 'drought', 'sandstorm': 'sandstream'}
UNASSIGNED = ({'$native': 'undefined'}, {'$native': 'missing'})


def _format_conditions(battle, live):
    pseudo = fields(live['pseudoWeather'])
    _need(battle['formatid'] == 'gen3customgame' and live['terrain'] == ''
          and set(pseudo) == {'sleepclausemod'},
          f'unrepresented Native field rules: format={battle["formatid"]!r}, '
          f'terrain={live["terrain"]!r}, pseudo-weather={sorted(pseudo)}')
    # This is the pinned format-wide rule, not a timed or source-owned field
    # effect. Sleep and Rest evidence are projected from the actual members.
    rule = fields(pseudo['sleepclausemod'])
    undefined = {'$native': 'undefined'}
    _need(set(rule) == {'id', 'source', 'sourceSlot', 'duration', 'effectOrder'}
          and rule['id'] == 'sleepclausemod' and rule['source'] is None
          and rule['sourceSlot'] == undefined and rule['duration'] == undefined
          and _integer(rule['effectOrder']) == 0, 'Native Sleep Clause rule has unexpected material state')


def _weather(battle, observed, history, sequence):
    field = _field(battle, 'field')
    live = observed['fields']['field']
    weather = _field(field, 'weather')
    _format_conditions(battle, live)
    _need(weather in WEATHER and weather == live['weather'], 'unsupported or inconsistent Native weather')
    if not weather:
        return 0, 0
    state = live['weatherState']
    state_fields = fields(state)
    creation = _field(history, 'weather_creation')
    _need(creation is not None and creation['weather'] == weather
          and creation['state_object'] == state['object'] and state_fields['id'] == weather
          and creation['sequence'] <= sequence and list(pokemon(state_fields['source'])) == creation['source'],
          'active Native weather lacks the successful creation and source binding')
    duration = _integer(_field(state_fields, 'duration'))
    _need(duration >= 0 and _field(field['weatherState'], 'duration') == duration,
          'literal Native weather duration differs from the live state')
    if duration == 0:
        effect = creation['effect']
        _need(weather in PERMANENT_WEATHER and effect is not None and effect['effect_type'] == 'Ability'
              and effect['id'] == PERMANENT_WEATHER[weather],
              'permanent Native weather lacks the matching ability creation')
    return WEATHER[weather], duration


def _side_conditions(battle, side, party, history, sequence):
    record = battle['sides'][side]
    conditions = _field(record, 'sideConditions')
    slots = _field(record, 'slotConditions')
    _need(type(conditions) is dict and set(conditions) <= {'spikes'}
          and type(slots) is list and len(slots) == 1 and type(slots[0]) is dict and set(slots[0]) <= {'wish'},
          'unrepresented Native side or slot condition')
    layers = 0
    if 'spikes' in conditions:
        layers = _integer(_field(conditions['spikes'], 'layers'))
        _need(1 <= layers <= 3, 'Native Spikes layers outside the active condition range')
    duration = 0
    if 'wish' in slots[0]:
        wish = slots[0]['wish']
        duration = _integer(_field(wish, 'duration'))
        source = list(party.resolve_pokemon(_field(wish, 'source')))
        events = [event for event in history['events'] if event['kind'] == 'CREATED_WISH' and event['side'] == side]
        _need(duration > 0 and events and events[-1]['sequence'] <= sequence
              and events[-1]['source'] == source and events[-1]['amount'] == history['wish_amount'][side],
              'active Native Wish did not join to its slot creation')
    return layers, duration


def _switch_flags(boundary, party, observed):
    battle, phase = party.battle, boundary.phase
    _need(_boolean(_field(battle, 'started')), 'closed Native decision has unfinished battle startup')
    forced, baton = [], []
    for side in (0, 1):
        original = party.active(side)
        current = member(observed, (side, original))
        hp = _integer(_field(current, 'hp'))
        _need(hp == 0 or _boolean(_field(current, 'isStarted')),
              'living Native active member still has an unresolved entry')
        request = boundary.snapshot['boundaryRequests'].get(f'p{side+1}') or {}
        force = request.get('forceSwitch', [])
        _need(type(force) is list and len(force) <= 1 and all(type(bit) is bool for bit in force),
              'literal Native singles switch request required')
        requested = any(force)
        flag = _field(current, 'switchFlag')
        _need(type(flag) is bool or flag == 'batonpass', 'unrepresented Native self-switch cause')
        active_switch = bool(flag) and phase.kind.startswith('replacement-')
        _need(requested == active_switch and (not requested or phase.requested[side]),
              'Native forced request and switch flag disagree')
        if requested:
            _need(any(party.mon(side, index)['hp'] > 0 for index in range(6) if index != original),
                  'Native forced request has no living replacement')
            if flag == 'batonpass':
                events = [event for event in boundary.history['events']
                          if event['kind'] == 'MOVE_USED' and event['side'] == side]
                _need(hp > 0 and events and events[-1]['sequence'] <= phase.snapshot_sequence
                      and events[-1]['original_actor'] == original and events[-1]['move_id'] == 'batonpass',
                      'Native Baton Pass request lacks the current living actor and dispatch proof')
            else:
                _need(hp == 0 and _boolean(_field(current, 'fainted')), 'unrepresented living Native forced switch')
        elif phase.kind != 'terminal':
            _need(flag is False, 'Native switch flag survives outside its request phase')
        forced.append(requested)
        baton.append(requested and flag == 'batonpass')
    return forced, baton


def _result(battle, observed, phase):
    live = observed['fields']
    ended = _boolean(_field(battle, 'ended'))
    winner = _field(live, 'winner')
    _need(ended is _boolean(_field(live, 'ended')) and ended == (phase.kind == 'terminal'),
          'Native terminal phase and result bit disagree')
    if not ended:
        _need(winner in ('', *UNASSIGNED) and battle.get('winner', '') == '',
              'live Native decision has a resolved winner')
        return False, -1
    _need(type(winner) is str and winner == _field(battle, 'winner'), 'terminal Native winner lacks literal binding')
    if winner == '':
        return True, 2
    names = [_field(side, 'name') for side in battle['sides']]
    _need(all(type(name) is str and name for name in names) and names.count(winner) == 1,
          'terminal Native winner is not a unique owning side')
    return True, names.index(winner)


def project_field(boundary, *, original_species):
    """Project literal PRE flags without an external collector stop overlay."""
    party, observed = bound_boundary(boundary, original_species)
    phase, history = boundary.phase, boundary.history
    weather, duration = _weather(party.battle, observed, history, phase.snapshot_sequence)
    side_conditions = [_side_conditions(party.battle, side, party, history, phase.snapshot_sequence) for side in (0, 1)]
    forced, baton = _switch_flags(boundary, party, observed)
    done, winner = _result(party.battle, observed, phase)
    values = dict(entry_pending=[False, False], forced_switch=forced, baton_pass=baton,
                  spikes=[entry[0] for entry in side_conditions], wish_counter=[entry[1] for entry in side_conditions],
                  weather=weather, weather_counter=duration, done=done, winner=winner)
    layout = _layout()
    return _typed_projection(values, {name: spec for name, spec in layout.items() if name in FIELD_LEAVES})
