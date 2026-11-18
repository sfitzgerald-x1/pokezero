"""Complete 73-field projection of ONE fresh Native startup boundary.

This consumes saved Native observations and sealed original constructor
coordinates only. It never accepts an expected initialized device state. The
fresh-battle/no-choice proof is mandatory; this is deliberately inapplicable
after any accepted action, suspension or historical condition. General causal
replay, literal owner streams and actual-device parity remain separate gates.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .fresh200_native_identity import PartyMap, to_id, reference_identifier

STATUS = {'':0, 'slp':1, 'par':2, 'brn':3, 'frz':4, 'psn':5, 'tox':6}
WEATHER = {'':0, 'raindance':1, 'sunnyday':2, 'sandstorm':3, 'hail':4}
STATS = ('atk', 'def', 'spa', 'spd', 'spe')
BOOSTS = (*STATS, 'accuracy', 'evasion')
EXTERNAL = frozenset(('residual_draws','phazing_draws','secondary_draws','sleep_talk_draws',
    'effect_draws','replacement_count','keys'))
HISTORY = frozenset(('last_action_kind','last_action_index','entry_pending','queued_actor',
    'queued_actions','queued_stall','pending_residual','wish_amount','transform_species',
    'transform_stats','transform_moves','transform_pp','transform_hidden_power'))


class MissingInitialProjection(ValueError):
    pass


def _need(value, message):
    if not value:
        raise MissingInitialProjection(message)


def _field(value, name):
    _need(type(value) is dict and name in value, 'missing required Native field: '+name)
    return value[name]


def _integer(value):
    _need(type(value) is int, 'literal Native integer required')
    return value


def _boolean(value):
    _need(type(value) is bool, 'literal Native boolean required')
    return value


def _leaves(value):
    if type(value) in (list,tuple):
        for child in value:yield from _leaves(child)
    else:yield value


@dataclass(frozen=True)
class ConstructorCoordinates:
    """Only literal genotype slot IDs and external PRNG coordinates, no history."""
    species: tuple
    move_ids: np.ndarray
    keys: np.ndarray


def _layout():
    schema=json.loads(Path(__file__).with_name('fresh200_storage_schema.json').read_text())
    _need(schema['source_commit']=='f9b0c48ef0d14c15f017c3e4b97dddc6695c5b3d', 'trusted startup schema source changed')
    rows=schema['records']['gpu_engine.state.BattleState']
    _need(len(rows)==73, 'complete 73-field state schema required')
    return {row['name']:row['spec'] for row in rows}


class Vocabulary:
    def __init__(self, vocabulary):
        self.values=vocabulary;self.ids={}
        for kind in ('species','types','moves','items','abilities'):
            values=_field(vocabulary,kind)
            _need(type(values) is list and values and all(type(x) is str for x in values), 'literal bound vocabulary required')
            index={}
            for ordinal,value in enumerate(values):
                if ordinal==0:
                    continue
                key=to_id(value)
                _need(key not in index, 'ambiguous vocabulary identity: '+kind)
                index[key]=ordinal
            self.ids[kind]=index

    def id(self, kind, value, *, empty=False):
        if empty and value=='':
            return 0
        key=to_id(value)
        _need(key in self.ids[kind], 'unknown Native '+kind+': '+key)
        return self.ids[kind][key]

    def move_slots(self, native, bound):
        _need(type(native) is list and 1<=len(native)<=4 and len(bound)==4, 'literal bound Native move slots required')
        _need(np.asarray(bound).dtype.kind in 'iu', 'constructor slot IDs must be integers')
        selected=[int(x) for x in bound]
        _need(all(0<x<len(self.values['moves']) for x in selected[:len(native)]) and not any(selected[len(native):]),
            'constructor and Native move-slot cardinality differs')
        for slot,member in enumerate(native):
            expected=to_id(self.values['moves'][selected[slot]])
            if expected.startswith('hiddenpower'):
                expected='hiddenpower'
            _need(_field(member,'id')==expected, 'positional constructor/Native move identity differs')
        return selected


def _initial_winner(snapshot):
    battle=snapshot['battle']
    if 'winner' in battle:
        _need(battle['winner']=='', 'unresolved fresh Native winner required')
        return dict(kind='literal_snapshot',value='')
    # Pinned Battle.winner is optional and is assigned by win(), which also
    # sets ended=true. Native State serialization omits undefined. Require
    # the actual live data/absence observation bound to THIS snapshot receipt;
    # never fill an omitted property from expected device state or a default.
    proof=snapshot['witness'];native=_field(proof,'nativeObservation')
    boundary=_field(_field(proof['receipt'],'boundary'),'nativeObservation')
    _need(native==boundary, 'elided Native winner observation is not snapshot-bound')
    fields=_field(_field(native,'state'),'fields');winner=_field(fields,'winner')
    _need(winner in ({'$native':'missing'},{'$native':'undefined'}) and _field(fields,'ended') is False,
        'actual live undecided Native winner observation required')
    return dict(kind='receipt_bound_live_optional_property',value=winner,ended=False)


def _fresh_start(snapshot, rows):
    """Prove the restricted history domain from actual closed Native commands."""
    battle=_field(snapshot,'battle');proof=_field(snapshot,'witness');receipt=_field(proof,'receipt')
    _need(receipt['poisoned'] is False and receipt['openFrameCount']==0, 'unclean Native snapshot receipt')
    _need(receipt['control']['phase']=='snapshot', 'closed snapshot boundary required')
    _need(_field(battle,'started') is True and _field(battle,'ended') is False, 'live fresh Native battle required')
    _need(_integer(_field(battle,'turn'))==1 and _field(battle,'midTurn') is False, 'startup ordinary-turn phase required')
    _need(_field(battle,'requestState')=='move' and _field(battle,'queue')==[], 'startup queue/request phase differs')
    winner=_initial_winner(snapshot)
    _need(type(rows) is list and rows, 'complete previously verified Native startup rows required')
    begins=[r for r in rows if r['operation']=='COMMAND_BEGIN']
    _need([r['phase'] for r in begins]==['start','snapshot','end','close'], 'one fresh Native constructor/snapshot lifecycle required')
    _need({r['battleId'] for r in rows}=={receipt['battleId']} and all(r['phase'] in ('start','snapshot','end','close') for r in rows),
        'fresh Native battle/audit identity differs')
    _need(not any(r['operation']=='CALL_THROW' for r in rows), 'original Native exception makes startup evidence unsuccessful')
    forbidden={'Battle.choose','Side.chooseMove','Side.chooseSwitch','Side.commitChoices','Pokemon.moveUsed',
        'Pokemon.deductPP','Pokemon.transformInto','Pokemon.copyVolatileFrom','Side.addSlotCondition'}
    calls=[r for r in rows if r['operation']=='CALL_ENTER']
    _need(not any(r['frame']['operation'] in forbidden for r in calls), 'Native historical operation makes startup-only projection inapplicable')
    starts=[r for r in calls if r['frame']['operation']=='Battle.setPlayer']
    _need(len(starts)==2 and all(r['phase']=='start' for r in starts), 'both literal Native constructor sides required')
    for row in calls:
        if row['frame']['operation']=='Battle.runAction':
            arguments=row['evidence']['arguments']['items'];action=arguments[0]
            _need(type(action) is dict and 'fields' in action, 'literal Native startup action required')
            fields=dict(action['fields'])
            _need(fields.get('choice') in ('start','runSwitch'), 'non-startup Native queue action present')
    return dict(snapshot_sequence=receipt['lastSequence'],source_registry_hash=receipt['sourceRegistryHash'],
        start_call_ids=[r['frame']['callId'] for r in starts],initial_history_absence_proved=True,
        winner_observation=winner,causal_coverage_complete=False,qualification=False)


def project_initial(snapshot, rows, vocabulary, constructor: ConstructorCoordinates):
    """Freeze all 73 independently projected leaves before device comparison.

    Startup restrictions are strict refusals, not exemptions: an active status,
    volatile/history condition, unresolved trap, suspension or transform needs
    the general causal interpreter. No original/device historical values are
    substituted. Absent initial conditions and zero external packet retention
    use the pinned fresh-battle initialization contract.
    """
    certificate=_fresh_start(snapshot,rows);battle=snapshot['battle'];vocab=Vocabulary(vocabulary)
    _need(isinstance(constructor,ConstructorCoordinates), 'sealed literal constructor coordinates required')
    _need(constructor.move_ids.shape==(2,6,4) and constructor.keys.shape==(2,) and constructor.keys.dtype==np.uint32,
        'literal constructor slot/key layout differs')
    party=PartyMap(battle,constructor.species);members=[[party.mon(s,p) for p in range(6)] for s in (0,1)]
    facts=_field(snapshot['witness'],'elidedProperties');live={}
    for fact in facts:
        key=(fact['side'],fact['original']);_need(key not in live, 'duplicate elided Native identity')
        _need(fact['current']==party.original_to_current[key[0]][key[1]], 'elided Native original/current identity differs')
        live[key]=fact
    _need(set(live)=={(s,p) for s in (0,1) for p in range(6)}, 'all twelve actual live Native scalar receipts required')
    for side in (0,1):
        for original,mon in enumerate(members[side]):
            _need(_field(mon,'status')=='' and _field(mon,'volatiles')=={}, 'active status/volatile needs general causal projection')
            _need(_field(mon,'transformed') is False and _field(mon,'lastMove') is None,
                'transform/move history needs general causal projection')
            _need(not _field(mon,'addedType'), 'unsupported Native added type')
            _need(_field(mon,'maxhp')==_field(mon,'baseMaxhp'), 'Gen3 HP/max-HP invariant differs')
            _need(_field(mon,'fainted') is False and 0<_integer(_field(mon,'hp'))<=_integer(mon['maxhp']),
                'fresh Native faint/HP correspondence differs')
            _need(all(_field(slot,'disabled') is False for slot in _field(mon,'moveSlots')),
                'Native disabling cause needs general causal projection')
        active=members[side][party.active(side)]
        _need(_field(active,'isStarted') is True and _field(active,'isActive') is True,
            'Native startup entry is incomplete')
        _need(_field(active,'trapped') is False, 'resolved Native trap needs complete cause decomposition')
        request=_field(snapshot['boundaryRequests'],f'p{side+1}')
        _need(not request.get('wait') and not any(request.get('forceSwitch') or []), 'startup request is not an ordinary decision')
    def matrix(fn):return [[fn(mon,s,p) for p,mon in enumerate(members[s])] for s in (0,1)]
    def scalar(name):return matrix(lambda m,s,p:_field(m,name))
    def condition_zero():return [[0]*6 for _ in (0,1)]
    values={
        'species':matrix(lambda m,s,p:vocab.id('species',_field(m['set'],'species'))),
        'level':matrix(lambda m,s,p:_field(live[s,p],'level')),
        'happiness':matrix(lambda m,s,p:_field(live[s,p],'happiness')),
        'hidden_power_type':matrix(lambda m,s,p:vocab.id('types',_field(m,'hpType'))),
        'hidden_power_power':scalar('hpPower'),
        'gender':matrix(lambda m,s,p:{'':0,'N':0,'M':1,'F':2}[_field(m,'gender')]),
        'stats':matrix(lambda m,s,p:[_field(m,'baseMaxhp'),*[_field(m['storedStats'],x) for x in STATS]]),
        'hp':scalar('hp'),'max_hp':scalar('maxhp'),
        'moves':matrix(lambda m,s,p:vocab.move_slots(m['moveSlots'],constructor.move_ids[s,p])),
        'pp':matrix(lambda m,s,p:[*[_field(x,'pp') for x in m['moveSlots']],*[0]*(4-len(m['moveSlots']))]),
        'disabled':matrix(lambda m,s,p:[False]*4),
        'item':matrix(lambda m,s,p:vocab.id('items',_field(m,'item'),empty=True)),
        'ability':matrix(lambda m,s,p:vocab.id('abilities',_field(m,'ability'))),
        'base_ability':matrix(lambda m,s,p:vocab.id('abilities',_field(m,'baseAbility'))),
        'boosts':matrix(lambda m,s,p:[_field(m['boosts'],x) for x in BOOSTS]),
        'active':[party.active(s) for s in (0,1)],
        'active_turns':[_integer(_field(members[s][party.active(s)],'activeTurns'))-1 for s in (0,1)],
        'trapped':[False,False],'forced_switch':[False,False],'baton_pass':[False,False],
        'last_action_kind':[0,0],'last_action_index':[-1,-1],'entry_pending':[False,False],
        'queued_actor':-1,'queued_actions':[0,0],'queued_stall':[0,0],'pending_residual':False,
        'replacement_count':0,'wish_amount':[0,0],
        'residual_draws':[[0]*12 for _ in (0,1)],'phazing_draws':[0,0],'secondary_draws':[0,0],
        'sleep_talk_draws':[0,0],'effect_draws':[[0]*5 for _ in (0,1)],'keys':constructor.keys.tolist(),
        'turn':_integer(battle['turn'])-1,'done':_boolean(battle['ended']),'winner':-1,
        'truant_loaf':scalar('truantTurn'),'item_knocked':scalar('itemKnockedOff'),
        'transformed':[False,False],'transform_species':[0,0],'transform_stats':[[0]*5 for _ in (0,1)],
        'transform_moves':[[0]*4 for _ in (0,1)],'transform_pp':[[0]*4 for _ in (0,1)],
        'transform_hidden_power':[[0,0] for _ in (0,1)],
    }
    for name in ('types','base_types'):
        native='types' if name=='types' else 'baseTypes'
        def types(mon,s,p):
            actual=_field(mon,native);_need(type(actual) is list and 1<=len(actual)<=2,'ordered Native type slots required')
            return [*[vocab.id('types',x) for x in actual],*[0]*(2-len(actual))]
        values[name]=matrix(types)
    for name in ('status','status_counter','sleep_skipped','choice_move','last_move','encore_move','encore_counter',
            'charge_move','charge_counter','volatile','confusion_counter','protect_counter','substitute_hp',
            'recharge_counter','perish_counter','yawn_counter','partial_trap_counter'):
        values[name]=condition_zero()
    values['rest_origin']=[[False]*6 for _ in (0,1)]
    for name in ('trap_source','partial_trap_source'):values[name]=[[-1]*6 for _ in (0,1)]
    values['spikes']=[];values['wish_counter']=[]
    for side in battle['sides']:
        _need(_field(side,'sideConditions')=={}, 'startup side condition needs general projection')
        _need(_field(side,'slotConditions')==[{}], 'startup slot condition needs general projection')
        values['spikes'].append(0);values['wish_counter'].append(0)
    field=_field(battle,'field');weather=_field(field,'weather')
    _need(weather in WEATHER, 'unsupported Native weather')
    values['weather']=WEATHER[weather]
    if weather:
        raise MissingInitialProjection('active Native weather needs source-bound causal provenance')
    else:values['weather_counter']=0
    layout=_layout();_need(set(values)==set(layout), 'complete 73-field projector coverage differs')
    result={}
    for name,spec in layout.items():
        raw=np.asarray(values[name]);dtype=np.dtype(spec['dtype']);shape=tuple(spec['shape'][1:])
        _need(raw.shape==shape, 'Native projection shape differs: '+name)
        if dtype.kind=='b':
            _need(all(type(x) is bool for x in _leaves(values[name])), 'Native projection boolean differs: '+name)
        else:
            _need(all(type(x) is int for x in _leaves(values[name])), 'Native projection integer type differs: '+name)
            _need(raw.dtype.kind in 'iu', 'Native projection integer differs: '+name)
            limits=np.iinfo(dtype);_need(np.all(raw>=limits.min) and np.all(raw<=limits.max), 'Native projection integer overflow: '+name)
        result[name]=np.asarray(raw,dtype=dtype)[None]
    return result,dict(certificate,field_count=len(result),physical_fields=sorted(set(layout)-EXTERNAL),
        external_fields=sorted(EXTERNAL),initial_history_fields=sorted(HISTORY),
        done_binding='literal live Native terminal bit; no external stop/refusal latch at this startup boundary',
        slot_binding='sealed original constructor coordinates, positional ID equality including explicit Hidden Power aliases',
        actual_device_parity=False,general_prefix_applicable=False)
