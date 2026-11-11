"""Genuine protocol roots through v5_public_world's pure Showdown mapper.

ProtocolInputs is a separate provenance type. It is not a retained driver,
Session, witness or History. The original materialize route for retained-driver
roots is untouched; its explicit protocol route calls the same typed
battle_spec_from_payload -> map_spec construction and rebases B2 public carry only.
"""
import copy
from dataclasses import dataclass,field,replace
import hashlib
import json
import random

import jax.numpy as j
import numpy as np

from pokezero.dex import normalize_id
from pokezero.engine_world import battle_spec_from_payload,EngineWorldUnsupported,unpack_team
from pokezero.env import BattleStartOverride
from pokezero.v5_teacher_public_root import PublicRoot,capture_root
from pokezero.determinization import _gen3_randbat_belief_start_override_result
from . import device_public_worlds as public_worlds,device_public_world_state as states
from . import phase_d_live_worlds as live,phase_d_tail as tail
from .v5_hypothetical_state import map_spec
from .phase_d_constructor_rules import public_hp_bin,request_ordered_packed,own_pp_carry,sleep_support

SCHEMA='pokezero.phase-d.protocol-materialization.v1'


@dataclass(frozen=True)
class ProtocolInputs:
    root:public_worlds.Root
    lines:tuple
    catalog:object
    metadata:dict
    source:object
    # One root's world-independent line scans, computed once for its worlds (_once); not part of the inputs' identity.
    memo:dict=field(default_factory=dict,compare=False,repr=False)


def _once(inputs,root,name,compute,*,shared=False):
    """compute() once per (inputs, root) for every world of one sample(): the same value (a deep copy per world unless shared,
    for read-only results) or the same exception re-raised at the same call site, exactly as computing it in every world."""
    key=(name,id(root))
    if key not in inputs.memo:
        try:inputs.memo[key]=(True,compute())
        except Exception as error:inputs.memo[key]=(False,error)
    ok,value=inputs.memo[key]
    if not ok:raise value.with_traceback(None)   # each raise starts a fresh traceback: the memo keeps one world's frames at most
    return value if shared else copy.deepcopy(value)


@dataclass(frozen=True)
class ProtocolWorld:
    state:object|None
    public:object|None
    refusals:tuple
    source_world:object|None=None
    origin_sha256:str|None=None
    mapper_diagnostics:tuple=()


def _chronology(lines):
    since=[None,None]
    for line in lines:
        p=line.split('|')
        if p[1]in ('switch','drag','replace'):
            if p[2][:2]not in ('p1','p2'):raise ValueError('public-active-owner-unavailable')
            since[int(p[2][1])-1]=0
        elif p[1]=='turn':since=[None if x is None else x+1 for x in since]
    if any(x is None for x in since):raise ValueError('public-active-arrival-unavailable')
    return tuple(max(0,x-1)for x in since)


def preflight_checks(payload,removed=None):
    """Same predicates used by production and collect-all; missing is UNKNOWN."""
    checks=[dict(kind='native-probed-hard-trapped-own-request',status='TRIGGERED' if payload.get('selfActiveRequestState',{}).get('trapped') else 'CLEAR',fatal=True),
        dict(kind='deferred-baton-phase',status='TRIGGERED' if payload.get('pendingBatonPassSides') else 'CLEAR',fatal=True)]
    for seat,side in payload['sides'].items():
        fail=bool(side.get('meanlookTrap') or set(side['volatiles'])&{'trapped','solarbeam','encore'})
        checks.append(dict(kind='native-probed-or-hidden-volatile-clock',status='TRIGGERED' if fail else 'CLEAR',side=seat,fatal=True))
        blockers=side.get('materializationBlockers',())
        failed=[kind for kind in blockers if not (str(kind).startswith('item-state-removed:') and normalize_id(str(kind).split(':',1)[1]) in (removed or {}).get(seat,()))]
        checks.append(dict(kind='public-materialization-blocker',status='TRIGGERED' if failed else 'CLEAR',side=seat,blockers=list(failed),fatal=True))
    return tuple(checks)


def _preflight(payload,removed=None):
    for check in preflight_checks(payload,removed):
        if check['status']=='TRIGGERED':raise ValueError(check['kind'])


def _public_removals(detached,metadata):
    carry=detached.public;removed=np.asarray(carry.item_removed);present=np.asarray(carry.present)
    species=np.asarray(carry.species)
    if removed.shape!=(1,2,6) or removed.dtype!=np.bool_:raise ValueError('PUBLIC_ITEM_REMOVAL_FACTS_UNAVAILABLE')
    return {'p'+str(side+1):tuple(normalize_id(metadata['vocabulary']['species'][int(species[0,side,k])]) for k in range(6) if present[0,side,k] and removed[0,side,k]) for side in (0,1)}


def _removed_items(payload,detached,metadata,lines):
    """Positive public removal only; retain source/reveal posterior unchanged."""
    for line in lines:
        p=line.split('|')
        if len(p)>1 and p[1] in ('-item','-enditem','-activate') and any('[from] move: '+move in line for move in ('Trick','Thief','Covet')):
            raise ValueError('PUBLIC_ITEM_SWAP_NOT_SUPPORTED')
    for side in payload['sides'].values():
        if any('currentItem' in row for row in side['pokemon']):raise ValueError('PUBLIC_ITEM_SWAP_NOT_SUPPORTED')
    return _public_removals(detached,metadata)


def _sleep_origins(lines):
    """Public attempt/refund history for either origin, with no hidden clock.

    Gen3 refunds one skippedTime unit at entry; Early Bird consumes two on
    each attempt. Keep these separate (the B2 net count alone loses that
    distinction). Normal prefixes reduce exactly to the declared count rule.
    """
    from .phase_d_constructor_rules import _identity,_move_key
    from pokezero.belief import _called_move_source
    identities={};origins={};abilities={}
    def settle():
        for value in origins.values():
            if value['pending']:value['pending']=False;value['skipped']=0
    for line in lines:
        p=line.split('|');kind=p[1] if len(p)>1 else ''
        if kind in ('upkeep','turn'):settle()
        if kind=='request':
            raw=json.loads(line[len('|request|'):]);side=raw['side']['id']
            for row in raw['side']['pokemon']:identities[(side,_identity(row['ident']))]=normalize_id(row['details'].split(',')[0])
            if len(raw.get('active',()))==1 and not(raw.get('wait') or any(raw.get('forceSwitch',()))):settle()
        elif kind in ('switch','drag','replace') and len(p)>4:
            identity=(p[2][:2],_identity(p[2]));name=normalize_id(p[3].split(',')[0]);identities[identity]=name
            key=(identity[0],name);abilities.pop(key,None)
            if key in origins:
                if 'slp' not in p[4].split():origins.pop(key)
                else:
                    value=origins[key];value['refunded']+=value['skipped'];value['skipped']=0;value['pending']=False
        if len(p)<3:continue
        key=(p[2][:2],identities.get((p[2][:2],_identity(p[2]))))
        if kind in ('-ability','-endability') and len(p)>3:abilities[key]=normalize_id(p[3]) if kind=='-ability' else ''
        if kind=='-status' and len(p)>3 and p[3]=='slp':
            origins[key]=dict(rest=_called_move_source(line)=='rest',attempt_abilities=[],attempt_refunds=[],refunded=0,skipped=0,pending=False)
        elif kind in ('-curestatus','faint'):origins.pop(key,None)
        elif kind=='-cureteam':
            origins={k:v for k,v in origins.items() if k[0]!=p[2][:2]}
        elif key in origins:
            value=origins[key]
            if kind=='cant' and len(p)>3 and p[3]=='slp':
                if value['pending']:value['skipped']=0
                value['attempt_abilities'].append(abilities.get(key));value['attempt_refunds'].append(value['refunded']);value['pending']=True
            elif value['pending'] and ((kind=='move' and len(p)>3 and _move_key(p[3]) in ('sleeptalk','snore') and _called_move_source(line)is None)
                    or kind=='cant' or(kind=='-activate' and len(p)>3 and normalize_id(p[3]) in ('confusion','moveattract'))):
                value['skipped']+=1;value['pending']=False
    return origins


def _defer_sleep(payload):
    """Typed bridge projection, never an admitted healthy world.

    Only literal slp is deferred. Unknown statuses/other guards stay unchanged.
    Every deferred member MUST be restored from public facts before return.
    """
    projected=json.loads(json.dumps(payload));sleepers=[]
    for seat,side in projected['sides'].items():
        for row in side['pokemon']:
            condition=str(row.get('condition','')).split()
            if 'slp' not in condition[1:]:continue
            if condition[0].split('/')[0]=='0':raise ValueError('PUBLIC_SLEEP_FAINT_CONFLICT')
            sleepers.append((int(seat[1])-1,normalize_id(row['species'])))
            row['condition']=' '.join(x for x in condition if x!='slp')
            for key in tuple(row):
                if key.startswith('restSleep'):del row[key]
    return projected,tuple(sleepers)


def _restore_sleep(physical,detached,metadata,sleepers,origins,rng,base_abilities,*,diagnostics=None):
    """Restore known status and draw hidden clock; no partial world escapes."""
    carry=detached.public;public_species=np.asarray(carry.species);present=np.asarray(carry.present)
    status=np.asarray(carry.status);features=carry.mon_features
    restored=physical;diagnostics=[] if diagnostics is None else diagnostics;restored_slots=set();failures=[]
    for side,name in sleepers:
        try:
            physical_slots=[k for k in range(6) if normalize_id(metadata['vocabulary']['species'][int(physical.species[0,side,k])])==name]
            public_slots=[k for k in range(6) if present[0,side,k] and normalize_id(metadata['vocabulary']['species'][int(public_species[0,side,k])])==name]
            if len(physical_slots)!=1 or len(public_slots)!=1:raise ValueError('PUBLIC_SLEEP_IDENTITY_UNAVAILABLE')
            at=physical_slots[0];public=public_slots[0]
            if status[0,side,public]!=5:raise ValueError('PUBLIC_SLEEP_STATUS_DISAGREEMENT')
            evidence=origins.get(('p'+str(side+1),name))
            if evidence is None:raise ValueError('PUBLIC_SLEEP_STINT_UNAVAILABLE')
            count=int(features['sleep_turns'][0,side,public]);rest=bool(features['rest_sleep'][0,side,public]);skipped=int(features['sleep_skipped_turns'][0,side,public])
            if evidence['rest']!=rest:raise ValueError('PUBLIC_SLEEP_ORIGIN_DISAGREEMENT')
            base=base_abilities[(side,name)]
            early=base=='earlybird'
            attempts=evidence['attempt_abilities'];refunded=evidence['refunded']
            if len(attempts)-refunded!=count:raise ValueError('PUBLIC_SLEEP_COUNTER_HISTORY_DISAGREEMENT')
            # Settle a proven ordinary request boundary on a value copy, just as
            # the original public payload settles pending Rest attempts. Persistent
            # B2 carry and encoder inputs are not modified.
            pending=bool(features['sleep_cant_pending'][0,side,public])
            if skipped!=evidence['skipped'] and not pending:raise ValueError('PUBLIC_SLEEP_SKIPPED_HISTORY_DISAGREEMENT')
            skipped=evidence['skipped']
            rates=tuple(2 if (ability if ability is not None else base)=='earlybird' else 1 for ability in attempts)
            support=sleep_support(count,rest,early_bird=early,skipped=skipped,pending=evidence['pending'],refunded=refunded,attempt_rates=rates,attempt_refunds=evidence['attempt_refunds'])
            counter=support[0] if len(support)==1 else rng.choice(support)
            restored=restored._replace(status=restored.status.at[0,side,at].set(5),status_counter=restored.status_counter.at[0,side,at].set(counter),
                rest_origin=restored.rest_origin.at[0,side,at].set(rest),sleep_skipped=restored.sleep_skipped.at[0,side,at].set(skipped))
            restored_slots.add((side,at))
            diagnostics.append(dict(kind='public-sleep-posterior',status='CLEAR',fatal=False,side=side,slot=at,support=list(support),sampled_counter=counter,rest=rest,skipped=skipped,
                validation='HIDDEN_CLOCK_SUPPORT_NOT_EXACT_OWN_EQUALITY'))
        except (ValueError,KeyError,IndexError,TypeError) as error:
            failures.append(str(error));diagnostics.append(dict(kind=str(error),status='TRIGGERED',fatal=True,side=side,species=name))
    if failures:raise ValueError(failures[0])
    actual={tuple(map(int,x)) for x in np.argwhere(np.asarray(restored.status)[0]==5)}
    if actual!=restored_slots:raise ValueError('PUBLIC_SLEEP_UNRESTORED_STATUS')
    return restored,tuple(diagnostics)


def _carry_public_foe(physical,detached,metadata,world,foe,rng,diagnostics):
    """Original postmap carries; collect every independently evaluable guard."""
    assignment={};pp=np.array(physical.pp,copy=True);hp=np.array(physical.hp,copy=True);failures=[]
    present=np.asarray(detached.view['present'])[0];species=np.asarray(detached.view['species'])[0]
    moves=np.asarray(detached.view['revealed_moves'])[0];used=np.asarray(detached.view['move_pp_used'])[0]
    move_identity=lambda mid:live._source_move(metadata['vocabulary']['moves'][int(mid)])
    def record(kind,failed,**details):
        diagnostics.append(dict(kind=kind,status='TRIGGERED' if failed else 'CLEAR',fatal=True,**details))
        if failed:failures.append(kind)
    for slot,name in enumerate(world.party_species[f'p{foe+1}']):
        matches=[i for i in range(6)if present[i]and normalize_id(metadata['vocabulary']['species'][int(species[i])])==normalize_id(name)]
        record('ambiguous-public-encounter',len(matches)>1,slot=slot)
        if len(matches)>1:
            diagnostics.append(dict(kind='PUBLIC_FOE_SLOT_CARRIES',status='NOT_EVALUABLE',fatal=True,slot=slot,reason='Ambiguous actual public encounter; no surrogate join'))
            continue
        if not matches:continue
        public=matches[0];assignment[public]=slot
        try:
            lo,hi=public_hp_bin(int(physical.max_hp[0,foe,slot]),int(detached.view['hp_percent'][0,public]))
            hp[0,foe,slot]=rng.randint(lo,hi)
            record('PUBLIC_FOE_HP_BIN',False,slot=slot,lower=lo,upper=hi)
        except (ValueError,KeyError,TypeError,IndexError) as error:record(str(error),True,slot=slot)
        for index,mid in enumerate(np.asarray(physical.moves)[0,foe,slot]):
            charge=sum(int(n)for known,n in zip(moves[public],used[public])
                if int(known)>0 and int(mid)>0 and move_identity(known)==move_identity(mid))
            failed=charge>pp[0,foe,slot,index]
            record('public-opponent-PP-charge-exceeds-source-max',failed,slot=slot,move=index)
            if not failed:pp[0,foe,slot,index]-=charge
        if int(physical.status[0,foe,slot])==4:
            try:
                counter=live._physical_toxic_counter(detached.view['mon_features']['toxic_stage'][0,public])
                physical=physical._replace(status_counter=physical.status_counter.at[0,foe,slot].set(counter))
                record('PUBLIC_TOXIC_STAGE_INVALID',False,slot=slot)
            except ValueError as error:record(getattr(error,'kind',str(error)),True,slot=slot)
    active=int(physical.active[0,foe]);active_public=[p for p,v in assignment.items()if v==active]
    record('sampled-active-public-join-unavailable',len(active_public)!=1)
    if len(active_public)==1:
        try:
            counter=live._physical_protect_counter(detached.view['mon_features']['stall_counter'][0,active_public[0]])
            physical=physical._replace(protect_counter=physical.protect_counter.at[0,foe,active].set(counter))
            record('PUBLIC_PROTECT_STREAK_INVALID',False)
        except ValueError as error:record(getattr(error,'kind',str(error)),True)
    else:diagnostics.append(dict(kind='PUBLIC_PROTECT_STREAK_INVALID',status='NOT_EVALUABLE',fatal=True,reason='Actual active public join unavailable'))
    return physical._replace(pp=j.asarray(pp),hp=j.asarray(hp)),assignment,active,failures


def _downstream_unknowns(diagnostics,reason):
    for kind in ('OWN_MAPPER_FIELDS','PUBLIC_FOE_SLOT_CARRIES','PUBLIC_SLEEP_WORLD_POSTERIOR','FINAL_PUBLIC_CARRY'):
        if not any(row['kind']==kind for row in diagnostics):
            diagnostics.append(dict(kind=kind,status='NOT_EVALUABLE',fatal=True,reason=reason))


def materialize_protocol(root,start,*,inputs,seed):
    """Pure typed mapping with exact own request and no true-opponent donor."""
    if type(root)is not PublicRoot or type(start)is not BattleStartOverride or type(inputs)is not ProtocolInputs:
        raise ValueError('typed genuine public root/start/protocol inputs required')
    detached=inputs.root;catalog=inputs.catalog;metadata=inputs.metadata;owner=detached.address.owner;foe=1-owner
    diagnostics=[]
    try:
        if root.address!=detached.address or root.owner_lines_sha256!=_once(inputs,root,'prefix',lambda:hashlib.sha256(''.join(x+'\n'for x in inputs.lines).encode()).hexdigest()):
            raise ValueError('same genuine owning prefix/address required')
        if catalog.source.source_hash!=root.context.public_materialization_state.belief_engine.set_source.metadata.source_hash:
            raise ValueError('same public source namespace required')
        state=root.context.public_materialization_state
        tail_facts=_once(inputs,root,'tail_facts',lambda:tail.facts(root,state.self_request,inputs.lines))
        removed=_once(inputs,root,'removed',lambda:_removed_items(root.materialization_payload,detached,metadata,inputs.lines))
        diagnostics.extend(preflight_checks(root.materialization_payload,removed))
        _preflight(root.materialization_payload,removed)
        projected,sleepers=_defer_sleep(root.materialization_payload)
        projected=tail.defer_substitutes(projected,tail_facts['fresh_substitutes'])
        origins=_once(inputs,root,'origins',lambda:_sleep_origins(inputs.lines))
        revealed={normalize_id(name):candidates for slot,universe,candidates,gender,name in
                  _once(inputs,root,'revealed',lambda:tuple(public_worlds._revealed(detached,inputs.source)),shared=True)}
        for mon in unpack_team(start.player_teams[f'p{foe+1}']):
            candidates=revealed.get(normalize_id(mon.species))
            if candidates is None:continue
            move_key=lambda xs:tuple(sorted(live._source_move(x)for x in xs))
            if not any(v.level==mon.level and move_key(v.moves)==move_key(mon.moves)
                    and normalize_id(v.ability or '')==normalize_id(mon.ability or '')
                    and normalize_id(v.item or '')==normalize_id(mon.item or '')for index,v in candidates):
                raise ValueError('sampled-foe-outside-b2-public-posterior')
        state=root.context.public_materialization_state
        own_names=tuple(normalize_id(metadata['vocabulary']['species'][int(x)])for x in detached.own['species'])
        own_key=f'p{owner+1}'
        sampled_names=tuple(normalize_id(mon.species)for mon in unpack_team(start.player_teams[own_key]))
        teams=dict(start.player_teams)
        teams[own_key]=request_ordered_packed(teams[own_key],sampled_names,own_names)
        start=replace(start,player_teams=teams)
        world=battle_spec_from_payload(projected,start,dex=catalog.dex,approximate_sleep_turns=False,
            approximate_substitute_health=False,approximate_partial_trap_turns=False,approximate_hidden_duration_volatiles=False,removed_item_species=removed)
        if any(len(x.pokemon)!=6 for x in (world.spec.side_one,world.spec.side_two)):raise ValueError('full-source-six-world-required')
        mood=[]
        for seat in ('p1','p2'):
            values=[]
            for member in start.player_teams[seat].split(']'):
                value=member.split('|')[11];values.append(255 if value==''else int(value))
            mood.append(tuple(values))
        mapped_names=tuple(normalize_id(x)for x in world.party_species[f'p{owner+1}'])
        if own_names!=mapped_names:raise ValueError('same-own-request-physical-order-required')
        mood[owner]=tuple(map(int,detached.own['happiness']))
        mapping_spec,cursors=tail.mapping_spec(world.spec,tail_facts['replacement'])
        mapped=map_spec(mapping_spec,metadata=metadata,dex=catalog.dex,seed=seed,
            turn=root.materialization_payload['turn'],active_turns=_once(inputs,root,'chronology',lambda:_chronology(inputs.lines)),happiness=tuple(mood))
        if mapped.state is None:
            _downstream_unknowns(diagnostics,'Typed mapper refused: '+repr(mapped.refusals))
            return ProtocolWorld(None,None,mapped.refusals,mapper_diagnostics=tuple(diagnostics))
        physical=tail.restore(mapped.state,world.spec,cursors,tail_facts,metadata=metadata,checks=diagnostics)
        maximum_pp={live._source_move(metadata['vocabulary']['moves'][int(mid)]):int(metadata['moves'][int(mid)]['pp'])*8//5 for row in detached.own['moves'] for mid in row if mid}
        own=dict(detached.own);own['pp']=np.asarray(_once(inputs,root,'own_pp',lambda:own_pp_carry(inputs.lines,state.self_request,maximum_pp)),np.int32)
        # Exact facts were read from the owning packet by B2's detached root.
        # Check all mapper-provided literal own fields before preserving the
        # complete own packet state (including exact private request flags).
        failures=[]
        for name in ('species','level','hp','max_hp','stats','moves','item','ability'):
            actual=np.asarray(getattr(physical,name))[0,owner];expected=own[name]
            fail=not np.array_equal(actual,expected)
            kind='own-request-mapper-disagreement:'+name
            diagnostics.append(dict(kind=kind,status='TRIGGERED' if fail else 'CLEAR',fatal=True))
            if fail:failures.append(kind)
        diagnostics.append(dict(kind='OWN_MAPPER_FIELDS',status='TRIGGERED' if failures else 'CLEAR',fatal=True))
        pp_fail=not np.array_equal(np.asarray(physical.pp)[0,owner],own['pp'])
        diagnostics.append(dict(kind='own-request-mapper-disagreement:pp',status='TRIGGERED' if pp_fail else 'CLEAR',fatal=False,
            source='LATEST_OWN_ACTIVE_REQUEST_PLUS_PUBLIC_POST_SNAPSHOT_COSTS',
            entries=[dict(slot=int(a),move=int(b),mapped=int(physical.pp[0,owner,a,b]),carried=int(own['pp'][a,b])) for a,b in zip(*np.where(np.asarray(physical.pp)[0,owner]!=own['pp']))]))
        if not failures:
            updates={name:getattr(physical,name).at[0,owner].set(j.asarray(own[name]))
                for name in public_worlds.OWN_MON_FIELDS+public_worlds.OWN_SIDE_FIELDS}
            physical=physical._replace(**updates)
        # A known public PP charge subtracts from sampled source max; no
        # unknown peer-private ledger or exact HP history is synthesized.
        hp_rng=random.Random(seed)
        physical,assignment,active,foe_failures=_carry_public_foe(physical,detached,metadata,world,foe,hp_rng,diagnostics)
        failures.extend(foe_failures)
        diagnostics.append(dict(kind='PUBLIC_FOE_SLOT_CARRIES',status='TRIGGERED' if foe_failures else 'CLEAR',fatal=True))
        base_abilities={(side,normalize_id(mon.species)):normalize_id(mon.ability or '') for side in (0,1) for mon in unpack_team(start.player_teams['p'+str(side+1)])}
        try:physical,_=_restore_sleep(physical,detached,metadata,sleepers,origins,hp_rng,base_abilities,diagnostics=diagnostics)
        except (ValueError,KeyError,IndexError,TypeError) as error:
            failures.append(str(error));diagnostics.append(dict(kind='PUBLIC_SLEEP_WORLD_POSTERIOR',status='TRIGGERED',fatal=True,reason=str(error)))
        else:diagnostics.append(dict(kind='PUBLIC_SLEEP_WORLD_POSTERIOR',status='CLEAR',fatal=True,applicability='CURRENT_SLEEP' if sleepers else 'NO_CURRENT_SLEEP'))
        if failures:raise ValueError(failures[0])
        carry=states._rebase(detached,assignment,active,state=physical)
        diagnostics.append(dict(kind='FINAL_PUBLIC_CARRY',status='CLEAR',fatal=True))
        body=dict(schema=SCHEMA,prefix=root.owner_lines_sha256,address=vars(root.address),seed=seed,
            public_source=catalog.source.source_hash,sampled_teams=dict(start.player_teams))
        digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return ProtocolWorld(physical,carry,(),world,digest,tuple(diagnostics))
    except EngineWorldUnsupported as error:
        _downstream_unknowns(diagnostics,'Original public mapper refused: '+error.reason)
        return ProtocolWorld(None,None,('original-public-mapper:'+error.reason,),mapper_diagnostics=tuple(diagnostics))
    except (ValueError,KeyError,IndexError,TypeError)as error:
        _downstream_unknowns(diagnostics,'Preceding production prerequisite refused: '+str(error))
        return ProtocolWorld(None,None,(str(error),),mapper_diagnostics=tuple(diagnostics))


def _carried_root(public,detached,lines,metadata):
    maximum={live._source_move(metadata['vocabulary']['moves'][int(mid)]):int(metadata['moves'][int(mid)]['pp'])*8//5 for row in detached.own['moves'] for mid in row if mid}
    carried=np.asarray(own_pp_carry(lines,public.context.public_materialization_state.self_request,maximum),np.int32)
    return replace(detached,own=public_worlds._freeze({**detached.own,'pp':carried}))


def diagnostic_inputs(public,detached,*,lines,metadata):
    """No sample/spec fabricated: current-source independent input checks.

    World-dependent statuses remain UNKNOWN until production supplies them.
    The caller must combine these with unchanged from_b2 raw checks and each
    actual world's refusals/mapper_diagnostics, never count nonfatal PP as a
    blocking refusal or infer CLEAR from an absent diagnostic row.
    """
    checks=list(tail.inspect_facts(public.context.public_materialization_state.self_request,public.materialization_payload,lines)[0])
    try:_carried_root(public,detached,lines,metadata)
    except (ValueError,KeyError,IndexError,TypeError) as error:checks.append(dict(kind=str(error),status='TRIGGERED',fatal=True))
    else:checks.append(dict(kind='OWN_PP_EXACT_CARRY',status='CLEAR',fatal=False))
    try:removed=_removed_items(public.materialization_payload,detached,metadata,lines)
    except (ValueError,KeyError,IndexError,TypeError) as error:
        checks.append(dict(kind=str(error),status='TRIGGERED',fatal=True));removed={}
    # Positive removal facts remain independently usable behind a mutation
    # refusal; this does not waive the production mutation guard.
    try:independent_removals=_public_removals(detached,metadata)
    except (ValueError,KeyError,IndexError,TypeError):independent_removals=None
    for check in preflight_checks(public.materialization_payload,independent_removals):
        if (independent_removals is None and check['kind']=='public-materialization-blocker'
                and check.get('blockers') and all(str(x).startswith('item-state-removed:') for x in check['blockers'])):
            check={**check,'status':'NOT_EVALUABLE','reason':'Public removal facts unavailable; matching discharge cannot be evaluated'}
        checks.append(check)
    try:_defer_sleep(public.materialization_payload);_sleep_origins(lines)
    except (ValueError,KeyError,IndexError,TypeError) as error:checks.append(dict(kind=str(error),status='TRIGGERED',fatal=True))
    checks.append(dict(kind='PUBLIC_SLEEP_WORLD_POSTERIOR',status='NOT_EVALUABLE',fatal=True,reason='Requires actual sampled/effective ability and typed world slots; no surrogate world generated'))
    return tuple(checks)


def sample(root,detached,*,lines,catalog,source,metadata,config,battle_id,seed,count):
    """Same public posterior draws and their prefix at all ladder budgets."""
    from . import v5_public_world
    if count not in (2,8,16,32):raise ValueError('declared2/8/16/32 world budget required')
    if detached.source_binding_sha256!=source.binding_sha256:
        raise live.Refusal('SAME_PUBLIC_ROOT_SOURCE_BINDING_REQUIRED')
    detached=_carried_root(root,detached,lines,metadata)
    inputs=ProtocolInputs(detached,tuple(lines),catalog,metadata,source)
    result=[];seeds=[];diagnostics=[]
    for index in range(count):
        identity=(SCHEMA,seed,root.context.battle_id,root.owner_lines_sha256,detached.address.stream,detached.address.frame,detached.address.owner,index)
        digest=hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).digest()
        world_seed=int.from_bytes(digest[:8],'big');kernel_seed=int.from_bytes(digest[8:12],'big');seeds.append(world_seed)
        try:
            start,reason=_gen3_randbat_belief_start_override_result(context=root.context,set_source=source.original,
                rng=random.Random(world_seed),team_size=6,canonicalize_revealed_variants=True)
            if start is None:
                unknown=[];_downstream_unknowns(unknown,'Original public sampler unavailable: '+str(reason))
                made=ProtocolWorld(None,None,(reason or 'public-sampler-unavailable',),mapper_diagnostics=tuple(unknown))
            else:made=v5_public_world.materialize(root,start,prepared=inputs,history=None,own_static=None,seed=kernel_seed)
            diagnostics.append(dict(world=index,checks=list(made.mapper_diagnostics)))
            result.append(states.World(index,made.state,made.public,made.refusals,kernel_seed,False))
        except (ValueError,KeyError,TypeError,IndexError)as error:
            unknown=[];_downstream_unknowns(unknown,'Production sample prerequisite refused: '+str(error))
            diagnostics.append(dict(world=index,checks=unknown))
            result.append(states.World(index,None,None,(str(error),),kernel_seed,False))
    return live.WorldSet(detached,tuple(result),tuple(seeds),count,tuple(diagnostics))
