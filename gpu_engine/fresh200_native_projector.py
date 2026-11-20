"""Derive typed physical leaves from a closed Native decision boundary."""

from .fresh200_native_archive import RecordedBoundary
from .fresh200_native_identity import PartyMap
from .fresh200_native_initial import (
    BOOSTS, STATS, STATUS, _boolean, _field, _integer, _layout, _need, _typed_projection,
)
from .fresh200_native_ledger import fields, member
from .fresh200_witness_audit import _same


_CHANCE = {
    'draws': 'residual_draws',
    'phazing': 'phazing_draws',
    'secondary': 'secondary_draws',
    'sleep_talk': 'sleep_talk_draws',
    'effect': 'effect_draws',
}
CORE_FIELDS = frozenset({
    'level', 'happiness', 'gender', 'stats', 'hp', 'max_hp', 'pp', 'boosts', 'active',
    'truant_loaf', 'item_knocked', 'status', 'status_counter', 'sleep_skipped', 'rest_origin',
    'active_turns', 'last_action_kind', 'last_action_index', 'queued_actor', 'queued_actions',
    'queued_stall', 'pending_residual', 'replacement_count', 'turn', 'wish_amount',
    *_CHANCE.values(),
})


def _scalars(witness, party):
    facts = {}
    for fact in _field(witness, 'elidedProperties'):
        side, original = _integer(fact['side']), _integer(fact['original'])
        _need(side in (0, 1) and 0 <= original < 6 and (side, original) not in facts,
              'unique original party coordinate required for elided Native scalars')
        _need(_integer(fact['current']) == party.original_to_current[side][original],
              'elided Native scalar position differs from the party permutation')
        level, happiness = _integer(fact['level']), _integer(fact['happiness'])
        _need(1 <= level <= 9999 and 0 <= happiness <= 255, 'out-of-range Native constructor scalar')
        facts[side, original] = (level, happiness)
    _need(len(facts) == 12, 'elided Native scalars are incomplete')
    return facts


def _major_status(status, state, rest_states):
    _need(status in STATUS, 'unsupported Native major status')
    counter = skipped = 0
    rest = False
    if status in ('slp', 'tox'):
        values = fields(state)
        _need(_field(values, 'id') == status, 'Native major status differs from its state token')
        counter = _integer(_field(values, 'time' if status == 'slp' else 'stage'))
        _need(counter >= 0, 'negative Native major-status counter')
        if status == 'slp':
            skipped = _integer(_field(values, 'skippedTime'))
            rest = state['object'] in rest_states
            _need(skipped >= 0 and rest == (values.get('startTime') == 3),
                  'Native sleep history differs from its Rest creation')
    return dict(status=STATUS[status], status_counter=counter, sleep_skipped=skipped, rest_origin=rest)


def _members(party, observed, scalars, history, sequence):
    retained = {}
    for entry in _field(history, 'retained_status'):
        identity = tuple(entry['actor'])
        _need(identity in scalars and identity not in retained and _integer(entry['sequence']) <= sequence,
              'pre-faint observation is outside the closed Native boundary')
        retained[identity] = entry
    rest_states = set(_field(history, 'rest_state_objects'))
    result = [[], []]
    for side, original in scalars:
        mon = party.mon(side, original)
        live = member(observed, (side, original))
        hp, maximum, base_maximum = (_integer(_field(mon, field)) for field in ('hp', 'maxhp', 'baseMaxhp'))
        status = _field(live, 'status')
        fainted = _boolean(_field(live, 'fainted'))
        _need(status == _field(mon, 'status') and _field(live, 'hp') == hp
              and fainted == _boolean(_field(mon, 'fainted')), 'Native member cache differs from the serialized member')
        coordinate = f'p{side+1} original slot {original}'
        _need(0 <= hp <= maximum == base_maximum and maximum > 0,
              f'Native HP and max-HP disagree for {coordinate}: hp={hp}, max={maximum}, baseMax={base_maximum}')
        _need((hp == 0) == fainted and (status != 'fnt' or fainted),
              f'Native HP and faint flag disagree for {coordinate}: hp={hp}, fainted={fainted}, status={status!r}')
        state = _field(live, 'statusState')
        if fainted:
            _need((side, original) in retained, 'fainted Native member lacks a pre-faint major-status observation')
            prior = retained[side, original]
            # faintMessages sets fainted. checkFainted adds the fnt marker to
            # active members at the switch check, which can be bypassed.
            _need(status in ('fnt', prior['status']),
                  f'Native status after faint differs from the retained pre-faint status for {coordinate}')
            status, state = prior['status'], prior['state']
        level, happiness = scalars[side, original]
        slots = _field(mon, 'moveSlots')
        _need(type(slots) is list and 1 <= len(slots) <= 4, 'Native physical move-slot count differs')
        pp = [_integer(_field(slot, 'pp')) for slot in slots]
        _need(all(value >= 0 for value in pp), 'negative Native PP')
        result[side].append((original, dict(
            level=level, happiness=happiness, gender={'': 0, 'N': 0, 'M': 1, 'F': 2}[_field(mon, 'gender')],
            stats=[base_maximum, *[_field(mon['storedStats'], name) for name in STATS]],
            hp=hp, max_hp=maximum, pp=pp + [0] * (4 - len(pp)),
            boosts=[_field(mon['boosts'], name) for name in BOOSTS],
            truant_loaf=_field(mon, 'truantTurn'), item_knocked=_field(mon, 'itemKnockedOff'),
            **_major_status(status, state, rest_states),
        )))
    return [[values for _, values in sorted(side)] for side in result]


def bound_boundary(boundary, original_species):
    _need(isinstance(boundary, RecordedBoundary), 'closed archived Native boundary required')
    snapshot, phase, history = boundary.snapshot, boundary.phase, boundary.history
    witness = _field(snapshot, 'witness')
    receipt = _field(witness, 'receipt')
    observed = _field(witness, 'nativeObservation')
    _need(receipt['lastSequence'] == phase.snapshot_sequence
          and receipt['sourceRegistryHash'] == history['source_registry_hash']
          and _same(observed, receipt['boundary']['nativeObservation'])
          and _same(_field(witness, 'elidedProperties'), receipt['boundary']['elidedProperties']),
          'Native phase, causal history and literal member data must describe the same boundary')
    party = PartyMap(_field(snapshot, 'battle'), original_species)
    return party, _field(observed, 'state')


def project_core(boundary, *, original_species):
    """Type-check the covered leaves without opening any expected State values."""
    party, observed = bound_boundary(boundary, original_species)
    phase, history = boundary.phase, boundary.history
    members = _members(party, observed, _scalars(boundary.snapshot['witness'], party), history, phase.snapshot_sequence)
    values = {key: [[mon[key] for mon in side] for side in members] for key in members[0][0]}
    values.update(
        active=[party.active(0), party.active(1)], active_turns=phase.active_turns,
        last_action_kind=history['last_action_kind'], last_action_index=history['last_action_index'],
        queued_actor=phase.queued_actor, queued_actions=phase.queued_actions, queued_stall=phase.queued_stall,
        pending_residual=phase.pending_residual, replacement_count=boundary.replacement_packets,
        turn=phase.turn, wish_amount=history['wish_amount'],
    )
    values.update({name: boundary.retained_chance[key] for key, name in _CHANCE.items()})
    layout = _layout()
    return _typed_projection(values, {name: spec for name, spec in layout.items() if name in CORE_FIELDS})
