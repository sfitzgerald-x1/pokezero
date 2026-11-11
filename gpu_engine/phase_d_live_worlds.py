"""Owning B2 protocol/request -> public device worlds, without a live donor.

Own private facts come only from the literal own request (with static Gen3
source spreads checked against its exact stats). Opponent identities, HP,
status, reveals and belief masks come only from the original B2 public fold.
Unsupported clocks are named refusals. No true opponent request/state API exists.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import random
import re

import jax
import jax.numpy as j
import numpy as np

from pokezero.dex import normalize_id
from pokezero.tier2 import canonical_move_id
from pokezero import showdown
from pokezero.determinization import (_fixture_from_variant_payload, _sample_gender,
    _sample_hidden_backline, _normalize_species_id)
from pokezero.gen3_damage import gen3_stat, gen3_hp_stat
from pokezero.engine_world import hidden_power_engine_id
from pokezero.v5_contract import DecisionAddress
from pokezero.v5_gumbel import Recipe
from . import device_public_worlds as original, device_public_world_state as materializer
from . import engine_device_trackers as trackers, engine_device_public_hooks as hooks
from . import engine_device_public_port as port, engine_device_damage_history as damage
from .state import empty_state, LEECH_SEED, ROOTED

SCHEMA = 'pokezero.phase-d.live-public-worlds.v1'
STATUS = (0, 5, 1, 2, 6, 3, 4)


class Refusal(ValueError):
    def __init__(self, kind, detail=''):
        self.kind, self.detail = kind, detail
        super().__init__(kind + (': ' + detail if detail else ''))


def _host(x):
    return np.asarray(jax.block_until_ready(x))


def _lookup(names, value):
    name = normalize_id(value or '')
    if not name:
        return 0
    found = {normalize_id(n): i for i, n in enumerate(names)}.get(name)
    if found is None and name=='hiddenpower':
        found=next((i for i,n in enumerate(names) if normalize_id(n).startswith('hiddenpower')),None)
    if found is None:
        raise Refusal('NAMESPACE_UNAVAILABLE', name)
    return found


def _remap(values, names, target):
    values = _host(values)
    if np.any(values < 0) or np.any(values >= len(names)):
        raise Refusal('PUBLIC_NAMESPACE_INVALID')
    # B2 knows a wider dex than the engine. Only actually referenced public
    # IDs need an engine representation; absent dex entries are not facts.
    lookup = np.zeros(len(names), np.int32)
    for index in np.unique(values):
        lookup[index] = _lookup(target, names[int(index)])
    return lookup[values]


def _source_move(name):
    value = normalize_id(name)
    return 'hiddenpower' if value.startswith('hiddenpower') else canonical_move_id(value)


def _own_move(names,name,hp_move):
    # map_spec uses this same device representation: the move ID carries the
    # type, while hidden_power_type/power hold the exact IV-derived facts.
    if _source_move(name)=='hiddenpower':
        value=canonical_move_id(hp_move);literal=canonical_move_id(name)
        if (not value.startswith('hiddenpower') or value=='hiddenpower'
                or literal not in ('hiddenpower',value)
                or (normalize_id(name)!=literal and normalize_id(name)!=normalize_id(hp_move))):
            raise Refusal('OWN_HIDDEN_POWER_LITERAL_DIFFERS_FROM_CHECKED_IVS',normalize_id(name))
    else:value=_source_move(name)
    return _lookup(names,value)


def _own_happiness(mon,moves,species):
    """Invert a literal own request's resolved power only when it is unique.

    Pinned Showdown f76228a sim/pokemon.ts getSwitchRequestData appends the
    actual data/moves.ts Return/Frustration callback to the move ID. Happiness
    is an integer in 0..255. Return102 therefore proves 255; plain Return and most
    other powers remain ambiguous. No generator default or donor is used.
    """
    explicit='happiness'in mon
    if explicit and(type(mon['happiness'])is not int or not 0<=mon['happiness']<=255):
        raise Refusal('OWN_HAPPINESS_INVALID')
    relevant=any(_source_move(x)in ('return','frustration')for x in moves)
    if not explicit and not relevant:return None
    candidates={mon['happiness']}if explicit else set(range(256))
    for move in moves:
        match=re.fullmatch(r'(return|frustration)([0-9]+)',normalize_id(move))
        if match is None:continue
        kind,power=match.groups();power=int(power)
        candidates={h for h in candidates if max(1,((h if kind=='return'else 255-h)*10)//25)==power}
    if not candidates:raise Refusal('OWN_HAPPINESS_INVALID')
    if len(candidates)!=1:raise Refusal('OWN_HAPPINESS_UNAVAILABLE',species)
    return next(iter(candidates))


def _physical_protect_counter(successes):
    """B2 public success streak -> engine's next-attempt denominator."""
    streak = int(successes)
    if not 0 <= streak <= showdown._STALL_COUNTER_CAP:
        raise Refusal('PUBLIC_PROTECT_STREAK_INVALID')
    return 0 if streak == 0 else 1 << min(streak, 3)


def _physical_toxic_counter(upcoming_stage):
    """B2 upcoming display stage -> completed residuals, with reset/cap."""
    stage = int(upcoming_stage)
    if not 0 <= stage <= 16:
        raise Refusal('PUBLIC_TOXIC_STAGE_INVALID')
    return max(0, min(15, stage - 1))


def _exact_own(request, state, catalog, source, metadata, owner, *, defer_sleep=False, protocol_facts=None):
    """Copy literal request fields, checking static latent IVs against own stats.

    The original public set source is used as a static format rule, never as a
    substitute for a missing private request. Ambiguous IV/HP effects refuse.
    """
    if request.get('side', {}).get('id') != f'p{owner+1}':
        raise Refusal('OWN_REQUEST_OWNER_MISMATCH')
    party = request['side'].get('pokemon')
    replacement=protocol_facts is not None and protocol_facts['replacement'] is not None
    if not isinstance(party, list) or len(party) != 6 or (not replacement and len(request.get('active', ())) != 1):
        raise Refusal('OWN_COMPLETE_ORDINARY_REQUEST_REQUIRED')
    if request.get('wait') or request.get('teamPreview') or (any(request.get('forceSwitch', ())) and not replacement):
        raise Refusal('SUSPENDED_OR_REPLACEMENT_BOUNDARY')
    template = empty_state(1)
    own = {name: np.array(getattr(template, name)[0, owner], copy=True)
           for name in original.OWN_MON_FIELDS + original.OWN_SIDE_FIELDS}
    v = source.vocabulary
    active = [i for i, mon in enumerate(party) if mon.get('active') is True]
    if len(active) != 1:
        raise Refusal('OWN_ACTIVE_UNAVAILABLE')
    own['active'][()] = active[0]
    current = {} if replacement else request['active'][0]
    own['forced_switch'][()] = replacement
    if current.get('maybeTrapped') and 'trapped' not in current:
        raise Refusal('OWN_TRAP_STATE_UNAVAILABLE')
    own['trapped'][()] = bool(current.get('trapped', False))
    public_names = [normalize_id(catalog.species.names[int(x)]) for x in _host(state.mons.species)[0, owner]]
    joined = {}
    for slot, mon in enumerate(party):
        details = str(mon.get('details', '')).split(',')
        species = normalize_id(details[0])
        universe = source.original.universe_for(species)
        if universe is None:
            raise Refusal('OWN_STATIC_SOURCE_UNAVAILABLE', species)
        level = next((int(x.strip()[1:]) for x in details[1:] if x.strip().startswith('L')), 100)
        gender = next((x.strip() for x in details[1:] if x.strip() in ('M', 'F')), 'N')
        condition = str(mon.get('condition', '')).split()
        if not condition:
            raise Refusal('OWN_EXACT_HP_UNAVAILABLE')
        if condition[0] == '0' and 'fnt' in condition:
            hp, maximum = 0, None
        elif '/' in condition[0]:
            hp, maximum = map(int, condition[0].split('/'))
        else:
            raise Refusal('OWN_EXACT_HP_UNAVAILABLE')
        stats = mon.get('stats')
        if type(stats) is not dict or any(type(stats.get(k)) is not int or stats[k] <= 0 for k in ('atk','def','spa','spd','spe')):
            raise Refusal('OWN_EXACT_STATS_UNAVAILABLE')
        moves = tuple(mon.get('moves', ()))
        if not 1 <= len(moves) <= 4 or any(type(m) is not str for m in moves):
            raise Refusal('OWN_MOVES_UNAVAILABLE')
        base_ability = mon.get('baseAbility') or mon.get('ability')
        if not base_ability or 'item' not in mon:
            raise Refusal('OWN_ABILITY_OR_ITEM_UNAVAILABLE')
        possible = []
        from .phase_d_constructor_rules import literal_hidden_power_agrees
        typed_hp=tuple(normalize_id(x)for x in moves if normalize_id(x).startswith('hiddenpower')and normalize_id(x)!='hiddenpower')
        for variant in universe.variants:
            if (variant.level != level or sorted(map(_source_move, variant.moves)) != sorted(map(_source_move, moves))
                    or normalize_id(variant.ability) != normalize_id(base_ability)):
                continue
            fixture = _fixture_from_variant_payload(variant.to_summary(), fallback_species=species,
                set_source=source.original, gender=gender)
            if fixture is None:
                continue
            info = source.original.species_metadata[_normalize_species_id(species)]
            base = info['baseStats']
            computed = [gen3_stat(base[k], fixture.ivs[k], fixture.evs[k], level) for k in ('atk','def','spa','spd','spe')]
            mx = 1 if base['hp'] == 1 else gen3_hp_stat(base['hp'], fixture.ivs['hp'], fixture.evs['hp'], level)
            if computed != [stats[k] for k in ('atk','def','spa','spd','spe')] or maximum is not None and maximum != mx:
                continue
            hp_move = next((x for x in fixture.moves if normalize_id(x).startswith('hiddenpower')), None)
            resolved = hidden_power_engine_id(normalize_id(hp_move), fixture.ivs) if hp_move else None
            if not literal_hidden_power_agrees(typed_hp,resolved):continue
            possible.append((mx, computed, info['types'], resolved))
        identities = {(mx, tuple(st), tuple(types), hp_move) for mx, st, types, hp_move in possible}
        if len(identities) != 1:
            raise Refusal('OWN_STATIC_IV_FACTS_UNAVAILABLE_OR_AMBIGUOUS', species)
        maximum, computed, types, hp_move = next(iter(identities))
        if not 0 <= hp <= maximum:
            raise Refusal('OWN_EXACT_HP_INVALID')
        own['species'][slot] = _lookup(v['species'], species)
        own['level'][slot] = level
        own['gender'][slot] = {'N':0, 'M':1, 'F':2}[gender]
        own['hp'][slot], own['max_hp'][slot] = hp, maximum
        own['stats'][slot] = [maximum, *computed]
        own['types'][slot] = own['base_types'][slot] = [_lookup(v['types'], t) for t in types] + [0]*(2-len(types))
        own['base_ability'][slot] = own['ability'][slot] = _lookup(v['abilities'], base_ability)
        own['item'][slot] = _lookup(v['items'], mon['item'])
        status = next((x for x in condition[1:] if x in ('slp','par','brn','frz','psn','tox')), '')
        own['status'][slot] = {'':0,'par':1,'brn':2,'psn':3,'tox':4,'slp':5,'frz':6}[status]
        if status=='slp' and not defer_sleep:raise Refusal('OWN_SLEEP_LATENT_CLOCK_UNAVAILABLE',species)
        # Known sleep status is exact; its hidden counter is restored per world
        # only after the protocol route validates public sleep evidence.
        happiness=_own_happiness(mon,moves,species)
        if happiness is not None:own['happiness'][slot]=happiness
        if hp_move:
            own['hidden_power_type'][slot] = _lookup(v['types'], hp_move[len('hiddenpower'):-2])
            own['hidden_power_power'][slot] = int(hp_move[-2:])
        matches = [i for i, n in enumerate(public_names) if n == species and int(_host(state.mons.encounter)[0,owner,i]) > 0]
        if len(matches) > 1:
            raise Refusal('OWN_PUBLIC_IDENTITY_AMBIGUOUS')
        public_slot = matches[0] if matches else None
        if public_slot is not None:
            joined[slot] = public_slot
        for k, name in enumerate(moves):
            move = _own_move(v['moves'],name,hp_move)
            own['moves'][slot,k] = move
            pp = metadata['moves'][move].get('pp')
            if type(pp) is not int or pp <= 0:
                raise Refusal('OWN_MOVE_MAX_PP_UNAVAILABLE')
            used = 0
            if public_slot is not None:
                ids = _host(state.context.moves.move_ids)[0,owner,public_slot]
                uses = _host(state.context.moves.uses)[0,owner,public_slot]
                used = sum(int(n) for m,n in zip(ids,uses) if m and _source_move(catalog.moves.names[int(m)]) == _source_move(name))
            own['pp'][slot,k] = max(0, pp*8//5-used)
            if slot == active[0] and not replacement:
                rows = [m for m in current.get('moves', ()) if _source_move(m.get('id','')) == _source_move(name)]
                if len(rows) != 1 or type(rows[0].get('pp')) is not int:
                    raise Refusal('OWN_ACTIVE_PP_UNAVAILABLE')
                own['pp'][slot,k] = rows[0]['pp']
                own['disabled'][slot,k] = bool(rows[0].get('disabled',False))
    if active[0] not in joined:
        raise Refusal('OWN_ACTIVE_PUBLIC_IDENTITY_UNAVAILABLE')
    return own, joined


def from_b2(adapter, request, source, metadata, *, address, defer_sleep=False, protocol_facts=None):
    """One original owning fold plus own request; no engine donor accepted."""
    from pokezero.recorded_encoder_runner import JaxRecordedAdapter
    from . import v4_encoder
    if (type(adapter) is not JaxRecordedAdapter or type(source) is not original.Source
            or type(address) is not DecisionAddress or adapter.owner_side != address.owner):
        raise Refusal('ORIGINAL_OWNING_B2_ADAPTER_REQUIRED')
    catalog, state, owner = adapter.catalog, adapter.state, address.owner
    if catalog.source.source_hash != source.source_hash:
        raise Refusal('PUBLIC_SOURCE_BINDING_DIFFERS')
    # Authoritative B2 admission happens before any materialization. This parses
    # ONLY request and reads original public fold, never another owner's packet.
    expected_inputs = adapter.encode(request, owner_side=owner)
    if type(defer_sleep)is not bool:raise Refusal('EXPLICIT_PROTOCOL_SLEEP_DEFER_REQUIRED')
    own, joined = _exact_own(request,state,catalog,source,metadata,owner,defer_sleep=defer_sleep,protocol_facts=protocol_facts)
    from . import public_context
    ctx,beliefs,ok=public_context.observation_boundary_copy(state.context,state.beliefs,state.mons,
        catalog.source.device,catalog.context,valid=j.ones(1,bool))
    if not bool(_host(ok).all()):raise Refusal('B2_OBSERVATION_BOUNDARY_COPY_REFUSED')
    state=state._replace(context=ctx,beliefs=beliefs)
    mons,v=state.mons,source.vocabulary
    if np.any(_host(ctx.abilities.transformed)):
        raise Refusal('PUBLIC_TRANSFORM_NOT_MATERIALIZED')
    if np.any(_host(ctx.types.overridden)):
        raise Refusal('PUBLIC_LIVE_RETYPE_NOT_MATERIALIZED')
    z = np.zeros((1,2,6),np.int32)
    present = _host(mons.encounter) > 0
    species = _remap(mons.species,catalog.species.names,v['species'])
    status = np.asarray(STATUS,np.int32)[_host(ctx.sleep.status)]
    hp_num, hp_den = _host(mons.hp_numerator), _host(mons.hp_denominator)
    foe = 1-owner
    fainted=_host(mons.fainted).astype(bool)
    if np.any(present[:,foe] & np.where(fainted[:,foe],(hp_num[:,foe]!=0)|(hp_den[:,foe]!=0),hp_den[:,foe]!=100)):
        raise Refusal('OPPONENT_LITERAL_PUBLIC_PERCENT_REQUIRED')
    percent = np.where(present, np.ceil(100*hp_num/np.maximum(1,hp_den)),0).astype(np.int32)
    percent = np.where((hp_num<hp_den)&(percent==100),99,percent).astype(np.int32)
    boosts = np.zeros((1,2,6,7),np.int32)
    active_slots = _host(mons.active)
    for side in range(2):
        if not 0 <= int(active_slots[0,side]) < 6:
            raise Refusal('PUBLIC_ACTIVE_UNAVAILABLE')
        boosts[0,side,active_slots[0,side]] = _host(mons.boosts)[0,side]
    mon_features = {k: z.astype(np.float32) if k.startswith('last_damage') else z.copy() for k in hooks.MON_KEYS}
    for key, field in (('turns_active','age'),('sleep_turns','count'),('rest_sleep','rest'),('sleep_skipped_turns','skipped'),('sleep_cant_pending','pending')):
        mon_features[key] = _host(getattr(ctx.sleep,field)).astype(np.int32)
    mon_features.update(choice_locked=z.copy(),wrap_trap_turns=z.copy())
    vol = np.zeros((1,2,6,len(hooks.VOLATILE_NAMES)),bool)
    last = z.copy()
    original_ability = _remap(ctx.abilities.original,catalog.abilities.names,v['abilities'])
    running = _remap(ctx.abilities.running,catalog.abilities.names,v['abilities'])
    shown_item = _remap(ctx.items.revealed,catalog.items.names,v['items'])
    observed = np.zeros((1,2,6,4),np.int32);used = np.zeros_like(observed)
    for side in range(2):
        at = int(active_slots[0,side])
        vol[0,side,at] = _host(ctx.active.volatiles)[0,side]
        for key, value in (('stall_counter',ctx.active.stall),('meanlook_trap',ctx.active.meanlook),
                           ('item_swapped',ctx.active.item_swapped),('choice_locked',ctx.active.choice_locked)):
            mon_features[key][0,side,at] = int(_host(value)[0,side])
        for k,key in enumerate(('confusion_turns','encore_turns','wrap_trap_turns')):
            mon_features[key][0,side,at] = int(_host(ctx.active.elapsed)[0,side,k])
        mon_features['sub_present'][0,side,at] = int(vol[0,side,at,hooks.VOLATILE_NAMES.index('substitute')])
        mon_features['toxic_stage'][0,side,at] = int(_host(ctx.toxic.stage)[0,side])
        kind = int(_host(ctx.moves.last_kind)[0,side]);mid = int(_host(ctx.moves.last_id)[0,side])
        last[0,side,at] = _lookup(v['moves'],catalog.moves.names[mid]) if kind == 2 else -1 if kind == 1 else 0
        for slot in range(6):
            ids = _host(ctx.moves.move_ids)[0,side,slot];charges = _host(ctx.moves.uses)[0,side,slot]
            ids = [(int(m),int(n)) for m,n in zip(ids,charges) if m]
            if len(ids) > 4:
                raise Refusal('PUBLIC_MORE_THAN_FOUR_SOURCE_MOVES')
            for k,(m,n) in enumerate(ids):
                observed[0,side,slot,k] = _lookup(v['moves'],catalog.moves.names[m])
                used[0,side,slot,k] = n
    own_slot = int(own['active']);pub_slot = joined[own_slot]
    if protocol_facts is not None:
        for seat,name in protocol_facts['fresh_substitutes'].items():
            side=int(seat[1])-1;at=int(active_slots[0,side])
            if normalize_id(v['species'][int(species[0,side,at])])!=name or not vol[0,side,at,hooks.VOLATILE_NAMES.index('substitute')]:
                raise Refusal('PUBLIC_SUBSTITUTE_B2_PRESENCE_DISAGREEMENT')
        if protocol_facts['replacement'] is not None and not fainted[0,owner,pub_slot]:
            raise Refusal('PUBLIC_REPLACEMENT_FAINT_DISAGREEMENT')
    replacement=protocol_facts is not None and protocol_facts['replacement'] is not None
    fresh=protocol_facts.get('fresh_substitutes',{}) if protocol_facts is not None else {}
    own_species=normalize_id(v['species'][int(own['species'][own_slot])])
    allowed = {'', 'leechseed', 'ingrain', 'trapped'}
    if fresh.get('p'+str(owner+1))==own_species:
        allowed.add('substitute');own['substitute_hp'][own_slot]=own['max_hp'][own_slot]//4
    own_vols = tuple(hooks.VOLATILE_NAMES[k] for k in np.flatnonzero(vol[0,owner,pub_slot]))
    if any(name not in allowed for name in own_vols):
        raise Refusal('OWN_VOLATILE_OR_LATENT_CLOCK_UNAVAILABLE', ','.join(own_vols))
    own['boosts'][own_slot] = boosts[0,owner,pub_slot]
    own['active_turns'][()] = max(0,int(mon_features['turns_active'][0,owner,pub_slot])-1)
    own['protect_counter'][own_slot] = _physical_protect_counter(mon_features['stall_counter'][0,owner,pub_slot])
    own['status_counter'][own_slot] = _physical_toxic_counter(mon_features['toxic_stage'][0,owner,pub_slot]) if own['status'][own_slot] == 4 else 0
    own['ability'][own_slot] = running[0,owner,pub_slot] or own['ability'][own_slot]
    for name, bit in (('leechseed',LEECH_SEED),('ingrain',ROOTED)):
        if name in own_vols:own['volatile'][own_slot] |= bit
    # Trap link names the current opposing active in the sampled physical map;
    # the existing materializer rebases this link using the retained public fact.
    if 'trapped' in own_vols or mon_features['meanlook_trap'][0,owner,pub_slot]:own['trap_source'][own_slot] = 0
    own_last = int(last[0,owner,pub_slot])
    if own_last > 0:
        names = [_source_move(v['moves'][int(m)]) for m in own['moves'][own_slot] if m]
        key = _source_move(v['moves'][own_last])
        if names.count(key) != 1:raise Refusal('OWN_LAST_MOVE_SLOT_UNAVAILABLE')
        k=names.index(key);own['last_move'][own_slot]=own['moves'][own_slot,k]
        own['last_action_kind'][()]=2;own['last_action_index'][()]=k
        if normalize_id(v['items'][int(own['item'][own_slot])]) == 'choiceband':
            own['choice_move'][own_slot]=own['last_move'][own_slot]
    elif own_last == -1:
        own['last_action_kind'][()]=1;own['last_action_index'][()]=own_slot
    if normalize_id(v['abilities'][int(own['ability'][own_slot])]) == 'truant':
        if not bool(_host(ctx.active.truant_known)[0,owner]):raise Refusal('OWN_TRUANT_PHASE_UNAVAILABLE')
        own['truant_loaf'][own_slot]=bool(_host(ctx.active.truant_loaf)[0,owner])
    field=ctx.field
    if not replacement and (np.any(_host(ctx.damage.current_dealt)) or np.any(_host(ctx.damage.current_taken))):
        raise Refusal('MIDTURN_DAMAGE_WINDOW_NOT_MATERIALIZED')
    starts=_host(field.wish_set_turn)[0];turn=int(_host(field.turn)[0])
    pending={'p'+str(side+1):int(start) for side,start in enumerate(starts) if start>=0 and turn-int(start)<=1}
    if protocol_facts is None and pending:raise Refusal('PUBLIC_WISH_CASTER_AMOUNT_UNAVAILABLE')
    if protocol_facts is not None:
        if pending!=protocol_facts['wish_set_turns']:raise Refusal('PUBLIC_WISH_PUBLIC_HISTORY_DISAGREEMENT')
        if protocol_facts['turn']!=turn:raise Refusal('PUBLIC_TAIL_TURN_DISAGREEMENT')
        if 'p'+str(owner+1) in pending:
            own['wish_counter'][()]=2-(turn-pending['p'+str(owner+1)])
            own['wish_amount'][()]=max(1,int(own['max_hp'][own_slot])//2)
    if replacement:own['active_turns'][()]=protocol_facts['replacement']['ages'][owner]
    weather_name=catalog.field.weather_names[int(_host(field.weather)[0])]
    weather={'':0,'sunnyday':2,'raindance':1,'sandstorm':3,'hail':4}.get(weather_name)
    if weather is None:raise Refusal('PUBLIC_WEATHER_UNAVAILABLE')
    # Initialize the PUBLIC carry directly from B2 arrays, never by observing a
    # sampled/empty physical opponent. Only owner cursor uses request indices.
    mapping=np.full((1,2,6),-1,np.int32);order=np.full_like(mapping,-1);physical=np.full((1,2),-1,np.int32)
    for private,public in joined.items():mapping[0,owner,private]=public
    order[0,owner]=np.arange(6);physical[0,owner]=own_slot
    # Join tendency keys by actual public species, never physical identities.
    td=ctx.tendency
    td_species=_host(td.species);public_species=_host(mons.species)
    joins=np.zeros((2,6),np.int32)
    for side in range(2):
        at=int(active_slots[0,side])
        mon_features['last_damage_dealt'][0,side,at]=float(_host(ctx.damage.last_dealt)[0,side].sum())
        mon_features['last_damage_taken'][0,side,at]=float(_host(ctx.damage.last_taken)[0,side].sum())
        for slot in np.flatnonzero(present[0,side]):
            found=np.flatnonzero(td_species[0,side]==public_species[0,side,slot])
            if len(found)!=1:raise Refusal('PUBLIC_TENDENCY_IDENTITY_UNAVAILABLE')
            joins[side,slot]=found[0]
            for k,key in enumerate(('switched_before_attack','stayed_and_attacked','turns_active_total')):
                mon_features[key][0,side,slot]=int(_host(td.mon)[0,side,found[0],k])
    matchup=np.zeros((1,2,6,6,2),np.int32)
    for side in range(2):
        for a in range(6):
            for b in range(6):
                if present[0,side,a] and present[0,1-side,b]:
                    matchup[0,side,a,b]=_host(td.matchup)[0,side,joins[side,a],joins[1-side,b]]
    tendency=trackers.Tendency(td.moved,td.opportunity_turn,td.switches,td.opportunities,j.asarray(matchup))
    visible_moves=j.asarray(observed);zero_float=j.zeros((1,2,6),j.float32)
    exact_taken=zero_float.at[0,owner,active_slots[0,owner]].set(ctx.damage.last_taken[0,owner].sum())
    exact_dealt=zero_float.at[0,foe,active_slots[0,foe]].set(ctx.damage.last_dealt[0,foe].sum())
    hazards=j.zeros((1,2,2,2),j.float32).at[0,owner,1].set(ctx.damage.hazard_suffered[0,owner]).at[0,foe,0].set(ctx.damage.hazard_suffered[0,foe])
    public_port=port.initialize(1)._replace(holder=ctx.sleep_clause.holder,victims=ctx.sleep_clause.victims,
        known=ctx.active.truant_known,loaf=ctx.active.truant_loaf,traced=ctx.active.traced_truant,
        skip_flip=ctx.active.truant_skip_flip,post_upkeep=ctx.active.post_upkeep,
        hazards=hazards,blocked=ctx.tendency.blocked,pursuit=ctx.tendency.pursuits,weather=ctx.tendency.weather)
    t=trackers.Trackers(j.asarray(mapping),j.asarray(physical),j.asarray(active_slots),j.asarray(present.sum(-1),j.int32),j.asarray(order),
        j.asarray(species),mons.level,mons.gender,j.asarray(percent),j.asarray(status),j.asarray(boosts),j.asarray(present),mons.fainted,
        field.turn,j.asarray([weather],j.int32),field.spikes,j.zeros(1,j.int32),
        observed_moves=visible_moves,pp_used=j.asarray(used),last_moves=j.asarray(last),belief_mask=state.beliefs.hard[:,::-1],
        shown_ability=j.asarray(original_ability),original_ability=j.asarray(original_ability),shown_item=j.asarray(shown_item),
        shown_types=j.zeros((1,2,6,2),j.int32),borrowed_ability=j.asarray(running),
        mon_features={k:j.asarray(x) for k,x in mon_features.items()},public_volatiles=j.asarray(vol),
        item_removed=ctx.items.removed,damage_exact_dealt=exact_dealt,damage_exact_taken=exact_taken,tendency=tendency,
        field_features=dict(weather_upkeeps=field.weather_upkeeps,weather_permanent=field.weather_ability,
            self_items_removed_credit=ctx.damage.items_removed[:,owner],opp_items_removed_credit=ctx.damage.items_removed[:,foe],
            wish_set_turn=field.wish_set_turn,wish_turns=j.zeros((1,2),j.int32),wish_pending=j.zeros((1,2),bool)),
        public_port=public_port)
    # Current ongoing damage history has exact data only for our own HP. The
    # peer's exact own history remains explicitly unavailable to reply policy.
    hist=damage.History(j.zeros((1,2),j.int32).at[0,owner].set(own['hp'][own_slot]),
        j.zeros((1,2),j.int32).at[0,owner].set(own['max_hp'][own_slot]),
        j.zeros((1,2,2,2),j.float32),j.zeros((1,2,2,2),j.float32))
    if replacement:
        # Original B2 owner readout: owner's dealt is shown foe fraction;
        # foe's dealt is exact own fraction. Unknown peer-private halves stay redacted.
        from .phase_d_tail import carry_current_damage
        hist=carry_current_damage(hist,ctx.damage.current_dealt,ctx.damage.current_taken,owner)
    from .engine_device_visible_belief import State as Visible
    hard=state.beliefs.hard[:,::-1];pin=state.beliefs.pin[:,::-1];pins=state.beliefs.pin_active[:,::-1]
    visible=Visible(pin,pins,state.beliefs.conflicts[:,::-1],hard,pin,pins,
        j.zeros((1,2,6,(len(v['abilities'])+31)//32),j.uint32),j.zeros((1,2,6,(len(v['items'])+31)//32),j.uint32))
    t=t._replace(damage_history=hist,visible_belief=visible)
    view=trackers.readout(t,j.asarray([owner],j.int32))
    snapshot=original._freeze({**view._asdict(),'hp_percent':j.asarray(percent[:,foe]),'_b2_root_inputs':expected_inputs})
    return original.Root(address,original._freeze(own),snapshot,original._redact_peer_private(t,owner),source.binding_sha256,False)


@dataclass(frozen=True)
class WorldSet:
    root: object
    worlds: tuple
    seeds: tuple
    requested_worlds: int
    mapper_diagnostics: tuple = ()

    @property
    def fully_constructed(self):
        return len(self.worlds)==self.requested_worlds and all(w.state is not None for w in self.worlds)


def _materialize_phase_d(root, start, tables, recipe):
    """Convert only Phase D's sampled foe physical units after original mapping.

    Keep the shared D materializer and every public tracker/feature untouched.
    The owner's detached physical counters were converted in from_b2 already.
    """
    world = materializer.materialize(root, start, tables, recipe)
    if world.state is None:
        return world
    try:
        foe = 1 - root.address.owner
        state = world.state
        features = root.view['mon_features']
        counters = state.status_counter
        protect = state.protect_counter
        active = int(state.active[0, foe])
        for physical, (public_slot, _) in enumerate(start.revealed_source_indices):
            if int(state.status[0, foe, physical]) == 4:
                counters = counters.at[0, foe, physical].set(
                    _physical_toxic_counter(features['toxic_stage'][0, public_slot]))
            if physical == active:
                protect = protect.at[0, foe, physical].set(
                    _physical_protect_counter(features['stall_counter'][0, public_slot]))
        return replace(world, state=state._replace(status_counter=counters, protect_counter=protect))
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return replace(world, state=None, public=None, refusals=(str(error),),
            reply_private_history_available=False)


def sample_worlds(root, source, tables, *, count, seed):
    """16/32 equally weighted original-public-source draws, never rerolled.

    Reuse the original revealed/source sampler and materializer. A separate
    domain-derived two-world recipe supplies each draw's existing world0 slot;
    no training quarter-selection or repeated two-world pair is used here.
    """
    if (count not in (16,32) or type(seed) is not int or not 0<=seed<2**64
            or type(root) is not original.Root or type(source) is not original.Source
            or type(tables) is not materializer.Tables or tables.source is not source):
        raise Refusal('BOUND_PUBLIC_SOURCE_16_OR_32_REQUIRED')
    try:
        revealed=original._revealed(root,source);error=None
    except ValueError as exc:
        revealed=();error=str(exc)
    worlds=[];seeds=[]
    for index in range(count):
        digest=hashlib.sha256(json.dumps((SCHEMA,seed,root.address.stream,root.address.frame,root.address.owner,index),separators=(',',':')).encode()).digest()
        recipe=Recipe(int.from_bytes(digest[:8],'big'))
        world_seed=recipe.world_seed(root.address,0);seeds.append(world_seed)
        rng=random.Random(world_seed);fixtures=[];indices=[];failure=error
        if failure is None:
            for slot,universe,candidates,shown_gender,shown_species in revealed:
                gender=shown_gender or _sample_gender(shown_species,set_source=source.original,rng=rng)
                selected,variant=candidates[rng.randrange(len(candidates))]
                fixture=_fixture_from_variant_payload(variant.to_summary(),fallback_species=shown_species,set_source=source.original,gender=gender)
                if fixture is None:failure='original revealed fixture spread unavailable';break
                fixtures.append(fixture);indices.append((slot,selected))
            if failure is None:
                hidden=_sample_hidden_backline(source.original,used_species={_normalize_species_id(x.species) for x in fixtures},count=6-len(fixtures),rng=rng)
                if hidden is None:failure='original hidden species/variant source unavailable'
                else:fixtures.extend(hidden)
        start=original.Start(0,world_seed,None if failure else tuple(fixtures),tuple(indices),() if failure is None else (failure,))
        world=_materialize_phase_d(root,start,tables,recipe)
        worlds.append(replace(world,index=index))
    return WorldSet(root,tuple(worlds),tuple(seeds),count)
