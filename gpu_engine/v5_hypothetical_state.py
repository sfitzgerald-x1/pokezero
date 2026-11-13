"""Pure typed public-world spec -> JAX storage for hypothetical branches.

This does not mint an ordinary driver Session, constructor certificate, or
scientific root. Unsupported state is returned explicitly; no native engine,
model, team generator, or scalar leaf callback is invoked.
"""
from dataclasses import dataclass
import re

import jax.numpy as j
import numpy as np

from pokezero.dex import normalize_id
from pokezero.poke_engine_adapter import BattleSpec
from .state import empty_state, LEECH_SEED, DESTINY_BOND

PERISH=('perish1','perish2','perish3','perish4')


@dataclass(frozen=True)
class Mapping:
    state: object | None
    refusals: tuple[str, ...]


def map_spec(spec, *, metadata, seed, turn, active_turns, happiness, dex=None, held_volatiles=False):
    """Trusted lower mapping; the public-root factory supplies source inputs.

    ``held_volatiles`` maps the public volatiles the engine holds since fidelity A2/M2 instead of refusing them:
    destinybond to the DESTINY_BOND bit (Showdown volatiles.destinybond: presence only, cleared at the holder's next
    move attempt, a switch-out or a faint) and perishN to perish_counter N (volatiles.perishsong.duration: Showdown's
    residual decrements it before printing |-start|X|perishN, so the shown N is the remaining count).

    ``turn`` is the displayed public request turn. ORIGINAL BattleState.turn
    counts completed ordinary turns, so its exact value here is turn-1.
    ``active_turns`` and happiness are absent from BattleSpec. They must be
    supplied by actual public arrival/TURN chronology and sampled packed-set
    semantics respectively, never recovered from neural numeric features.
    """
    if type(spec) is not BattleSpec:raise ValueError('original typed BattleSpec required')
    sides=(spec.side_one,spec.side_two);reasons=[]
    if spec.terrain!='none' or spec.trick_room:reasons.append('global-field-unrepresentable')
    if type(turn) is not int or turn<1:raise ValueError('literal positive public turn required')
    if len(active_turns)!=2 or any(type(v) is not int or v<0 for v in active_turns):
        raise ValueError('two exact public active-turn counters required')
    if len(happiness)!=2 or any(len(values)!=len(side.pokemon) for values,side in zip(happiness,sides)):
        raise ValueError('one packed-set happiness per physical member required')
    for seat,side in enumerate(sides):
        if not 1<=len(side.pokemon)<=6:reasons.append(f'party-size:{seat}')
        if side.force_switch or side.baton_passing or side.slow_uturn_move or side.switch_out_move_second_saved_move:
            reasons.append(f'deferred-or-replacement-phase:{seat}')
        if set(side.side_conditions)-{'spikes','protect','toxic_count'}:reasons.append(f'side-condition:{seat}')
        held={'destinybond',*PERISH} if held_volatiles else set()
        if set(side.volatile_statuses)-{'leechseed','substitute','truant'}-held:reasons.append(f'volatile:{seat}')
        if sum(v in PERISH for v in side.volatile_statuses)>1:reasons.append(f'perish-count:{seat}')
        if side.volatile_status_durations:reasons.append(f'volatile-clock:{seat}')
        for slot,mon in enumerate(side.pokemon):
            if mon.pre_transform is not None or mon.base_types not in (None,mon.types):reasons.append(f'transformed-types:{seat}:{slot}')
            if mon.base_ability not in (None,mon.ability):reasons.append(f'copied-ability:{seat}:{slot}')
            if mon.status=='sleep' or mon.rest_turns or mon.rest_sleep_pending_refund or mon.sleep_turns:
                reasons.append(f'sleep-clock:{seat}:{slot}')
            if mon.weight_kg is not None:
                info=None if dex is None else dex.species_info(mon.id)
                if info is None or mon.weight_kg!=info.weight_kg:reasons.append(f'custom-weight:{seat}:{slot}')
    if reasons:return Mapping(None,tuple(reasons))
    state=empty_state(1,seed=seed)
    arrays={name:np.array(value,copy=True) for name,value in state._asdict().items()}
    vocab=metadata['vocabulary']
    namespaces={kind:{normalize_id(name):i for i,name in enumerate(vocab[kind])}
        for kind in ('species','types','abilities','items','moves')}
    def identity(kind,value,*,none=False):
        name=normalize_id(value or '')
        if none and name in ('','none','typeless'):return 0
        if name not in namespaces[kind] or namespaces[kind][name]<=0:
            raise ValueError('namespace:'+kind+':'+name)
        return namespaces[kind][name]
    boosts=('attack','defense','special_attack','special_defense','speed','accuracy','evasion')
    statuses={'none':0,'paralyze':1,'burn':2,'poison':3,'toxic':4,'freeze':6}
    try:
        for seat,side in enumerate(sides):
            if type(side.active_index) is not int or not 0<=side.active_index<len(side.pokemon):raise ValueError('active-index')
            active=side.active_index
            if side.pokemon[active].hp<=0:raise ValueError('ordinary-root-active-fainted')
            arrays['active'][0,seat]=active
            arrays['active_turns'][0,seat]=active_turns[seat]
            if set(side.boosts)-set(boosts):raise ValueError('boost-key')
            for index,key in enumerate(boosts):
                value=side.boosts.get(key,0)
                if type(value) is not int or not -6<=value<=6:raise ValueError('boost-stage')
                arrays['boosts'][0,seat,active,index]=value
            for key,destination in (('spikes','spikes'),):
                value=side.side_conditions.get(key,0)
                if type(value) is not int or not 0<=value<=3:raise ValueError('spikes')
                arrays[destination][0,seat]=value
            protect=side.side_conditions.get('protect',0);toxic=side.side_conditions.get('toxic_count',0)
            if type(protect) is not int or not 0<=protect<2**31 or type(toxic) is not int or not 0<=toxic<=14:
                raise ValueError('public-status-or-stall-clock')
            if type(side.substitute_health) is not int or not 0<=side.substitute_health<=side.pokemon[active].maxhp:
                raise ValueError('substitute-hp')
            if len(side.wish)!=2 or any(type(value) is not int or not 0<=value<2**31 for value in side.wish):raise ValueError('wish-clock')
            arrays['protect_counter'][0,seat,active]=protect
            arrays['status_counter'][0,seat,active]=toxic
            arrays['substitute_hp'][0,seat,active]=side.substitute_health
            if 'leechseed' in side.volatile_statuses:arrays['volatile'][0,seat,active]|=LEECH_SEED
            if held_volatiles and 'destinybond' in side.volatile_statuses:arrays['volatile'][0,seat,active]|=DESTINY_BOND
            for name in PERISH:
                if held_volatiles and name in side.volatile_statuses:arrays['perish_counter'][0,seat,active]=int(name[-1])
            arrays['truant_loaf'][0,seat,active]='truant' in side.volatile_statuses
            arrays['wish_counter'][0,seat],arrays['wish_amount'][0,seat]=side.wish
            for slot,mon in enumerate(side.pokemon):
                if any(type(value) is not int for value in (mon.level,mon.hp,mon.maxhp,mon.attack,
                        mon.defense,mon.special_attack,mon.special_defense,mon.speed)):
                    raise ValueError('noninteger-mon-field')
                if not 1<=mon.level<=100 or not 0<=mon.hp<=mon.maxhp or mon.maxhp<=0:
                    raise ValueError('mon-level-or-hp')
                values=(mon.maxhp,mon.attack,mon.defense,mon.special_attack,mon.special_defense,mon.speed)
                if any(value<=0 or value>=2**31 for value in values):raise ValueError('mon-stats')
                address=(0,seat,slot)
                arrays['species'][address]=identity('species',mon.id)
                arrays['level'][address]=mon.level
                arrays['hp'][address]=mon.hp;arrays['max_hp'][address]=mon.maxhp
                arrays['stats'][address]=values
                arrays['ability'][address]=identity('abilities',mon.ability,none=True)
                arrays['base_ability'][address]=identity('abilities',mon.base_ability or mon.ability,none=True)
                arrays['item'][address]=identity('items',mon.item,none=True)
                if mon.status not in statuses:raise ValueError('status:'+mon.status)
                arrays['status'][address]=statuses[mon.status]
                if mon.gender not in (None,'M','F','N'):raise ValueError('gender')
                # None is genuinely unknown; no fabricated gender draw.
                arrays['gender'][address]={None:-1,'N':0,'M':1,'F':2}[mon.gender]
                if not 1<=len(mon.types)<=2:raise ValueError('type-count')
                arrays['types'][address]=[identity('types',value,none=True) for value in mon.types]+[0]*(2-len(mon.types))
                # A public Color Change can replace mon.types while the older
                # materializer leaves base_types=None. Never seal that live
                # replacement as constructor identity. The original static dex
                # supplies an unmodified species row or this root is unknown.
                base=mon.base_types
                info=None if dex is None else dex.species_info(mon.id)
                if base is None:
                    if info is None:raise ValueError('base-types-unavailable')
                    base=tuple(info.types)
                elif dex is not None:
                    if info is None:raise ValueError('base-types-unavailable')
                    if tuple(normalize_id(v)for v in base)!=tuple(normalize_id(v)for v in info.types):
                        raise ValueError('base-types-differ-original-species')
                base=[identity('types',value,none=True) for value in base]
                if not 1<=len(base)<=2:raise ValueError('base-type-count')
                base=base+[0]*(2-len(base))
                if base!=arrays['types'][address].tolist():raise ValueError('changed-types-base-unavailable')
                arrays['base_types'][address]=base
                mood=happiness[seat][slot]
                if type(mood) is not int or not 0<=mood<=255:raise ValueError('packed-happiness')
                arrays['happiness'][address]=mood
                if len(mon.moves)>4:raise ValueError('move-count')
                for index,move in enumerate(mon.moves):
                    if move.id=='none':
                        if move.pp!=0:raise ValueError('absent-move-pp')
                        continue
                    move_id=normalize_id(move.id)
                    hp=re.fullmatch(r'hiddenpower([a-z]+)([0-9]+)',move_id)
                    if hp:
                        typ,power=hp.groups();power=int(power)
                        if not 30<=power<=70:raise ValueError('hidden-power-power')
                        arrays['hidden_power_type'][address]=identity('types',typ)
                        arrays['hidden_power_power'][address]=power
                        move_id=next((name for name in ('hiddenpower'+typ,'hiddenpower') if name in namespaces['moves']),move_id)
                    arrays['moves'][address+(index,)]=identity('moves',move_id)
                    if type(move.pp) is not int or not 0<=move.pp<2**31 or type(move.disabled) is not bool:
                        raise ValueError('move-pp-or-disabled')
                    arrays['pp'][address+(index,)]=move.pp
                    arrays['disabled'][address+(index,)]=move.disabled
            last=side.last_used_move
            if last:
                if last=='move:struggle':
                    arrays['last_action_kind'][0,seat]=3
                    arrays['last_move'][0,seat,active]=identity('moves','struggle')
                elif last.startswith('move:'):
                    index=int(last.split(':')[1])
                    if not 0<=index<4 or arrays['moves'][0,seat,active,index]<=0:raise ValueError('last-move-slot')
                    arrays['last_action_kind'][0,seat]=2;arrays['last_action_index'][0,seat]=index
                    arrays['last_move'][0,seat,active]=arrays['moves'][0,seat,active,index]
                    # With unsupported item mutations refused at the public
                    # factory, a sampled birth ChoiceBand + actual last move
                    # implies its own private current lock, never a peer guess.
                    if normalize_id(side.pokemon[active].item or '')=='choiceband':
                        arrays['choice_move'][0,seat,active]=arrays['last_move'][0,seat,active]
                elif last.startswith('switch:'):
                    arrays['last_action_kind'][0,seat]=1;arrays['last_action_index'][0,seat]=active
                else:raise ValueError('last-action')
        weather={'none':0,'rain':1,'sun':2,'sand':3,'hail':4}
        if spec.weather not in weather:raise ValueError('weather')
        arrays['weather'][0]=weather[spec.weather]
        if spec.weather!='none':
            if type(spec.weather_turns_remaining) is not int or not -1<=spec.weather_turns_remaining<=5:raise ValueError('weather-clock')
            arrays['weather_counter'][0]=max(0,spec.weather_turns_remaining)
        arrays['turn'][0]=turn-1;arrays['done'][0]=False;arrays['winner'][0]=-1
    except (ValueError,KeyError,IndexError) as error:return Mapping(None,(str(error),))
    return Mapping(type(state)(**{name:j.asarray(value) for name,value in arrays.items()}),())
