"""Genuine protocol roots through v5_public_world's pure Showdown mapper.

ProtocolInputs is a separate provenance type. It is not a retained driver,
Session, witness or History. The original materialize route for retained-driver
roots is untouched; its explicit protocol route calls the same typed
world_battle_spec -> map_spec construction and rebases B2 public carry only.
"""
from dataclasses import dataclass
import hashlib
import json
import random

import jax.numpy as j
import numpy as np

from pokezero.dex import normalize_id
from pokezero.engine_world import world_battle_spec,EngineWorldUnsupported,unpack_team
from pokezero.env import BattleStartOverride
from pokezero.v5_teacher_public_root import PublicRoot,capture_root
from pokezero.determinization import _gen3_randbat_belief_start_override_result
from . import device_public_worlds as public_worlds,device_public_world_state as states
from . import phase_d_live_worlds as live
from .v5_hypothetical_state import map_spec

SCHEMA='pokezero.phase-d.protocol-materialization.v1'


@dataclass(frozen=True)
class ProtocolInputs:
    root:public_worlds.Root
    lines:tuple
    catalog:object
    metadata:dict
    source:object


@dataclass(frozen=True)
class ProtocolWorld:
    state:object|None
    public:object|None
    refusals:tuple
    source_world:object|None=None
    origin_sha256:str|None=None


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


def _preflight(payload):
    """Reject original Native-probe paths before pure spec construction."""
    if payload.get('selfActiveRequestState',{}).get('trapped'):raise ValueError('native-probed-hard-trapped-own-request')
    if payload.get('pendingBatonPassSides'):raise ValueError('deferred-baton-phase')
    for side in payload['sides'].values():
        if side.get('meanlookTrap')or set(side['volatiles'])&{'trapped','solarbeam','encore'}:
            raise ValueError('native-probed-or-hidden-volatile-clock')
        if any('slp'in str(row.get('condition','')).split()for row in side['pokemon']):raise ValueError('native-probed-sleep-clock')
        if side.get('materializationBlockers'):raise ValueError('public-materialization-blocker')


def materialize_protocol(root,start,*,inputs,seed):
    """Pure typed mapping with exact own request and no true-opponent donor."""
    if type(root)is not PublicRoot or type(start)is not BattleStartOverride or type(inputs)is not ProtocolInputs:
        raise ValueError('typed genuine public root/start/protocol inputs required')
    detached=inputs.root;catalog=inputs.catalog;metadata=inputs.metadata;owner=detached.address.owner;foe=1-owner
    try:
        if root.address!=detached.address or root.owner_lines_sha256!=hashlib.sha256(''.join(x+'\n'for x in inputs.lines).encode()).hexdigest():
            raise ValueError('same genuine owning prefix/address required')
        if catalog.source.source_hash!=root.context.public_materialization_state.belief_engine.set_source.metadata.source_hash:
            raise ValueError('same public source namespace required')
        _preflight(root.materialization_payload)
        revealed={normalize_id(name):candidates for slot,universe,candidates,gender,name in public_worlds._revealed(detached,inputs.source)}
        for mon in unpack_team(start.player_teams[f'p{foe+1}']):
            candidates=revealed.get(normalize_id(mon.species))
            if candidates is None:continue
            move_key=lambda xs:tuple(sorted(live._source_move(x)for x in xs))
            if not any(v.level==mon.level and move_key(v.moves)==move_key(mon.moves)
                    and normalize_id(v.ability or '')==normalize_id(mon.ability or '')
                    and normalize_id(v.item or '')==normalize_id(mon.item or '')for index,v in candidates):
                raise ValueError('sampled-foe-outside-b2-public-posterior')
        state=root.context.public_materialization_state
        world=world_battle_spec(state,start,dex=catalog.dex,approximate_sleep_turns=False,
            approximate_substitute_health=False,approximate_partial_trap_turns=False,approximate_hidden_duration_volatiles=False)
        if any(len(x.pokemon)!=6 for x in (world.spec.side_one,world.spec.side_two)):raise ValueError('full-source-six-world-required')
        mood=[]
        for seat in ('p1','p2'):
            values=[]
            for member in start.player_teams[seat].split(']'):
                value=member.split('|')[11];values.append(255 if value==''else int(value))
            mood.append(tuple(values))
        own_names=tuple(normalize_id(metadata['vocabulary']['species'][int(x)])for x in detached.own['species'])
        mapped_names=tuple(normalize_id(x)for x in world.party_species[f'p{owner+1}'])
        if own_names!=mapped_names:raise ValueError('same-own-request-physical-order-required')
        mood[owner]=tuple(map(int,detached.own['happiness']))
        mapped=map_spec(world.spec,metadata=metadata,dex=catalog.dex,seed=seed,
            turn=root.materialization_payload['turn'],active_turns=_chronology(inputs.lines),happiness=tuple(mood))
        if mapped.state is None:return ProtocolWorld(None,None,mapped.refusals)
        physical=mapped.state
        # Exact facts were read from the owning packet by B2's detached root.
        # Check all mapper-provided literal own fields before preserving the
        # complete own packet state (including exact private request flags).
        for name in ('species','level','hp','max_hp','stats','moves','pp','item','ability'):
            if not np.array_equal(np.asarray(getattr(physical,name))[0,owner],detached.own[name]):
                raise ValueError('own-request-mapper-disagreement:'+name)
        updates={name:getattr(physical,name).at[0,owner].set(j.asarray(detached.own[name]))
            for name in public_worlds.OWN_MON_FIELDS+public_worlds.OWN_SIDE_FIELDS}
        physical=physical._replace(**updates)
        # A known public PP charge subtracts from sampled source max; no
        # unknown peer-private ledger or exact HP history is synthesized.
        assignment={};pp=np.array(physical.pp,copy=True)
        present=np.asarray(detached.view['present'])[0];species=np.asarray(detached.view['species'])[0]
        moves=np.asarray(detached.view['revealed_moves'])[0];used=np.asarray(detached.view['move_pp_used'])[0]
        # Public Hidden Power is deliberately generic, while sampled engine
        # slots retain their typed move IDs. Debit the same canonical move.
        move_identity=lambda mid:live._source_move(metadata['vocabulary']['moves'][int(mid)])
        for slot,name in enumerate(world.party_species[f'p{foe+1}']):
            matches=[i for i in range(6)if present[i]and normalize_id(metadata['vocabulary']['species'][int(species[i])])==normalize_id(name)]
            if len(matches)>1:raise ValueError('ambiguous-public-encounter')
            if not matches:continue
            public=matches[0];assignment[public]=slot
            for index,mid in enumerate(np.asarray(physical.moves)[0,foe,slot]):
                charge=sum(int(n)for known,n in zip(moves[public],used[public])
                    if int(known)>0 and int(mid)>0 and move_identity(known)==move_identity(mid))
                if charge>pp[0,foe,slot,index]:raise ValueError('public-opponent-PP-charge-exceeds-source-max')
                pp[0,foe,slot,index]-=charge
            if int(physical.status[0,foe,slot])==4:
                physical=physical._replace(status_counter=physical.status_counter.at[0,foe,slot].set(
                    live._physical_toxic_counter(detached.view['mon_features']['toxic_stage'][0,public])))
        active=int(physical.active[0,foe]);active_public=[p for p,v in assignment.items()if v==active]
        if len(active_public)!=1:raise ValueError('sampled-active-public-join-unavailable')
        physical=physical._replace(pp=j.asarray(pp),protect_counter=physical.protect_counter.at[0,foe,active].set(
            live._physical_protect_counter(detached.view['mon_features']['stall_counter'][0,active_public[0]])))
        carry=states._rebase(detached,assignment,active,state=physical)
        body=dict(schema=SCHEMA,prefix=root.owner_lines_sha256,address=vars(root.address),seed=seed,
            public_source=catalog.source.source_hash,sampled_teams=dict(start.player_teams))
        digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return ProtocolWorld(physical,carry,(),world,digest)
    except EngineWorldUnsupported as error:return ProtocolWorld(None,None,('original-public-mapper:'+error.reason,))
    except (ValueError,KeyError,IndexError,TypeError)as error:return ProtocolWorld(None,None,(str(error),))


def sample(root,detached,*,lines,catalog,source,metadata,config,battle_id,seed,count):
    """Same public posterior draws and their prefix at all ladder budgets."""
    from . import v5_public_world
    if count not in (2,8,16,32):raise ValueError('declared2/8/16/32 world budget required')
    if detached.source_binding_sha256!=source.binding_sha256:
        raise live.Refusal('SAME_PUBLIC_ROOT_SOURCE_BINDING_REQUIRED')
    inputs=ProtocolInputs(detached,tuple(lines),catalog,metadata,source)
    result=[];seeds=[]
    for index in range(count):
        identity=(SCHEMA,seed,root.context.battle_id,root.owner_lines_sha256,detached.address.stream,detached.address.frame,detached.address.owner,index)
        digest=hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).digest()
        world_seed=int.from_bytes(digest[:8],'big');kernel_seed=int.from_bytes(digest[8:12],'big');seeds.append(world_seed)
        try:
            start,reason=_gen3_randbat_belief_start_override_result(context=root.context,set_source=source.original,
                rng=random.Random(world_seed),team_size=6,canonicalize_revealed_variants=True)
            if start is None:made=ProtocolWorld(None,None,(reason or 'public-sampler-unavailable',))
            else:made=v5_public_world.materialize(root,start,prepared=inputs,history=None,own_static=None,seed=kernel_seed)
            result.append(states.World(index,made.state,made.public,made.refusals,kernel_seed,False))
        except (ValueError,KeyError,TypeError,IndexError)as error:
            result.append(states.World(index,None,None,(str(error),),kernel_seed,False))
    return live.WorldSet(detached,tuple(result),tuple(seeds),count)
