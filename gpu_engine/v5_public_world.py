"""Original PURE public mapper -> separately named hypothetical JAX root.

Never calls Native, creates a game or rewrites an ordinary Session. The raw
owning history fixes our public carry across worlds. Sampled foe PP is reduced
by its ORIGINAL public ledger charges; the old full-PP exemption is not used.
"""
from dataclasses import dataclass, replace
from collections import OrderedDict
import hashlib
import json

import jax.numpy as j
import jax
import numpy as np

from pokezero.dex import normalize_id
from pokezero.engine_world import world_battle_spec, EngineWorldUnsupported, unpack_team
from pokezero.determinization import _gen3_randbat_fixture_spread
from pokezero.tier2 import canonical_move_id
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import _request_pokemon_identity
from pokezero.v5_teacher_public_root import PublicRoot
from pokezero.saved_player_projection import _public_condition
from . import engine_retained_encoder_driver as driver, v4_encoder, public_inputs
from . import public_context, public_belief
from . import v5_owning_history as capture
from .v5_hypothetical_state import map_spec

_FOLDS=OrderedDict()  # Bounded STATIC compiled functions only, no game carries.


def fold_public(state, chunk, catalog):
    key=id(catalog)
    if key not in _FOLDS or _FOLDS[key][0] is not catalog:
        _FOLDS[key]=(catalog,jax.jit(lambda value,packet:v4_encoder.fold_chunk(value,packet,catalog)))
    _FOLDS.move_to_end(key)
    while len(_FOLDS)>4:_FOLDS.popitem(last=False)
    return _FOLDS[key][1](state,chunk)


@dataclass(frozen=True)
class HypotheticalRoot:
    state: object
    public: object
    world: object
    prepared: object
    history: capture.History
    root: PublicRoot
    origin_sha256: str


@dataclass(frozen=True)
class Result:
    branch: HypotheticalRoot | None
    refusals: tuple[str,...]


@dataclass(frozen=True)
class OwnStatic:
    stream: int
    owner: int
    constructor_binding: tuple[int,...]
    happiness: tuple[int,...]


def capture_own_static(session, *, row, owner):
    """ONLY original own constructor happiness, never any peer physical cell."""
    record=driver._session(session);runtime=driver._checked(session.prepared)
    driver._checked_witness(runtime,session=session)
    core=session.state.core
    if type(row) is not int or not 0<=row<len(core.frame_number) or type(owner) is not int or owner not in (0,1):
        raise ValueError('exact original owning row/seat required')
    if int(core.frame_number[row])!=0:raise ValueError('capture immutable own static at actual constructor frame0')
    result=OwnStatic(int(core.cursor.stream_id[row]),owner,
        tuple(map(int,np.asarray(core.constructors[owner].binding[row]))),
        tuple(map(int,np.asarray(core.engine.happiness[row,owner]))))
    if driver._session(session) is not record:raise ValueError('original constructor changed during own-static capture')
    return result


def public_carry(history, runtime):
    """Original exact public words, folded once independently of any world."""
    catalog=runtime['catalog']
    if history.catalog is not catalog:raise ValueError('same original owning catalog required')
    if any(frame.refusal for frame in history.frames):raise ValueError('refused-owner-history')
    rows=[word for frame in history.frames for chunk,valid in zip(frame.words,frame.valid)
        for word,use in zip(chunk,valid) if use]
    events=[public_inputs.PublicEvent(public_inputs.Kind(w[0]),w[1],w[2],public_inputs.Phase(w[3]),
        w[4],w[5],w[6],w[7],tuple(w[8:])) for w in rows]
    namespaces=runtime['public'].device.namespaces._asdict()
    chunks=public_inputs.pack_chunks(np.array([history.stream_id],np.int32),[events],**namespaces)
    state=v4_encoder.initialize(catalog,stream_ids=j.array([history.stream_id],j.int32),
        owner_sides=j.array([history.owner],j.int32))
    for chunk in chunks:
        state,error=fold_public(state,chunk,catalog)
        if bool(np.asarray(error).any()):raise ValueError('original-public-carry-fold-refused')
    return state


def _own_pp_state(state, history, metadata):
    """Known unvisited own bench PP from actual owning constructor, not Dex."""
    cache=dict(state.self_move_states)
    original=json.loads(history.frames[0].request_json)['side']['pokemon']
    constructor=dict(history.own_constructor)
    species=metadata['vocabulary']['species'];moves=metadata['vocabulary']['moves']
    by_species={normalize_id(species[int(sid)]):i for i,sid in enumerate(constructor['species']) if sid}
    active_seen={normalize_id(row['details'].split(',')[0]) for frame in history.frames
        if frame.request_json for row in json.loads(frame.request_json)['side']['pokemon'] if row['active']}
    for row in original:
        identity=_request_pokemon_identity(row)
        name=normalize_id(row['details'].split(',')[0])
        if identity in cache:continue
        if name in active_seen:raise ValueError('previously-active-own-PP-unavailable')
        if name not in by_species:raise ValueError('own-constructor-species-join')
        index=by_species[name]
        cache[identity]=tuple(dict(id=moves[int(move)],pp=int(pp),maxpp=int(pp),disabled=False)
            for move,pp in zip(constructor['moves'][index],constructor['max_pp'][index]) if move)
    return replace(state,self_move_states=cache)


def _chronology(history):
    since=[None,None]
    for frame in history.frames:
        for chunk,valid in zip(frame.words,frame.valid):
            for word,use in zip(chunk,valid):
                if not use:continue
                if word[0] in (public_inputs.Kind.SWITCH,public_inputs.Kind.DRAG,public_inputs.Kind.REPLACE):since[word[6]]=0
                if word[0]==public_inputs.Kind.TURN:
                    since=[None if value is None else value+1 for value in since]
    if any(value is None for value in since):raise ValueError('public-active-arrival-unavailable')
    return tuple(max(0,value-1) for value in since)


def _base_move(name):
    name=normalize_id(name)
    return 'hiddenpower' if name.startswith('hiddenpower') else name


def _source_foe(root, start, history, catalog, public):
    """Full source-row identity + ORIGINAL sampler spread, including HP type."""
    source=root.context.public_materialization_state.belief_engine.set_source
    _,beliefs,ok=public_context.observation_boundary_copy(public.context,public.beliefs,public.mons,
        catalog.source.device,catalog.context,valid=True)
    if not bool(np.asarray(ok).all()):raise ValueError('original-public-posterior-boundary-refused')
    _,summary=public_belief.materialize(beliefs,catalog.source.device)
    foe=1-history.owner
    for mon in unpack_team(start.player_teams[f'p{foe+1}']):
        sid=catalog.source.species_id(mon.species)
        key=lambda moves:tuple(sorted(canonical_move_id(value) for value in moves))
        matches=[i for i,variant in enumerate(catalog.source.variants[sid]) if sid>0
            and variant.level==mon.level and key(variant.moves)==key(mon.moves)
            and normalize_id(variant.ability or '')==normalize_id(mon.ability or '')
            and normalize_id(variant.item or '')==normalize_id(mon.item or '')]
        if not matches:raise ValueError('sampled-foe-outside-full-source-axis')
        fixed=source.species_metadata.get(normalize_id(mon.species),{}).get('gender')
        if (fixed in ('M','F','N') and mon.gender!=fixed) or (fixed not in ('M','F','N') and mon.gender not in ('M','F')):
            raise ValueError('sampled-foe-outside-original-gender-recipe')
        spread=_gen3_randbat_fixture_spread({},species=mon.species,moves=tuple(mon.moves),
            item=mon.item,level=mon.level,set_source=source)
        if spread is None or dict(mon.ivs)!=dict(spread['ivs']) or dict(mon.evs)!=dict(spread['evs']):
            raise ValueError('sampled-foe-outside-original-spread-recipe')
        slots=[i for i,species in enumerate(np.asarray(public.mons.source_species)[0,foe]) if int(species)==sid]
        if len(slots)>1:raise ValueError('ambiguous-public-source-species')
        if slots:
            bits=np.asarray(summary.effective)[0,history.owner,slots[0]]
            if not any((int(bits[index//32])>>(index%32))&1 for index in matches):
                raise ValueError('sampled-foe-outside-actual-root-posterior')


def materialize(root, start, *, prepared, history, own_static, seed, retained=None):
    """One source-sampled start -> exact ordinary current root or refusal.

    Caller may reuse the SAME ``retained`` result of public_carry across both
    worlds. It never contains sampled latent values and is not reconditioned
    on a world. Passing a world-derived neural observation is not an API here.
    """
    if type(root) is not PublicRoot or type(start) is not BattleStartOverride or type(history) is not capture.History or type(own_static) is not OwnStatic:
        raise ValueError('original public root, owning history and sampled packed override required')
    runtime=driver._checked(prepared);catalog=runtime['catalog']
    if history.catalog is not catalog or history.session_identity[1]!=id(prepared):
        raise ValueError('original frozen owning prepared namespace differs')
    try:
        decoded=capture.decode(history)
        if not decoded.complete:raise ValueError('whole-owning-history-inverse-unavailable')
        prefix=hashlib.sha256(''.join(line+'\n' for line in decoded.lines).encode()).hexdigest()
        if prefix!=root.owner_lines_sha256:raise ValueError('public-root-owning-history-bytes')
        # This increment reconstructs the sampled opponent's private view only
        # from proven fresh birth and NO subsequent foe HP-altering packet.
        # Percentage100 alone (including100->100 damage/heal) is not full-HP
        # history evidence. Never recover private exact history from that bin.
        if tuple(frame.number for frame in history.frames)!=tuple(range(len(history.frames))):
            raise ValueError('fresh-uninterrupted-owning-constructor-history-required')
        foe=1-history.owner
        if any(use and word[0]==public_inputs.Kind.CONDITION and word[6]==foe
                for frame in history.frames for chunk,valid in zip(frame.words,frame.valid)
                for word,use in zip(chunk,valid)):
            raise ValueError('opponent-own-private-hp-history-unrepresented')
        if root.address.stream!=history.stream_id or root.address.frame!=history.frames[-1].number or root.address.owner!=history.owner:
            raise ValueError('public-root-owning-history-address')
        birth=dict(history.own_constructor)
        if ((own_static.stream,own_static.owner)!=(history.stream_id,history.owner)
                or own_static.constructor_binding!=birth['binding'] or len(own_static.happiness)!=6):
            raise ValueError('same-actual-own-static-constructor-required')
        state=root.context.public_materialization_state
        if state.belief_engine.set_source.metadata.source_hash!=catalog.source.source_hash:
            raise ValueError('public-root-source-binding')
        payload=root.materialization_payload
        # These original public mapper paths invoke Native capability probes;
        # they are deliberately refused BEFORE calling it, not monkeypatched.
        if payload.get('selfActiveRequestState',{}).get('trapped'):
            raise ValueError('native-probed-hard-trapped-own-request')
        for side in payload['sides'].values():
            if side.get('meanlookTrap') or set(side['volatiles'])&{'trapped','solarbeam','encore'}:
                raise ValueError('native-probed-or-hidden-volatile-clock')
            if any('slp' in str(row.get('condition','')).split() for row in side['pokemon']):
                raise ValueError('native-probed-sleep-clock')
            if side.get('materializationBlockers'):raise ValueError('public-materialization-blocker')
        if payload['pendingBatonPassSides']:raise ValueError('deferred-baton-phase')
        metadata=json.loads(runtime['request']['metadata'])
        state=_own_pp_state(state,history,metadata)
        world=world_battle_spec(state,start,dex=catalog.dex,approximate_sleep_turns=False,
            approximate_substitute_health=False,approximate_partial_trap_turns=False,
            approximate_hidden_duration_volatiles=False)
        if any(len(side.pokemon)!=6 for side in (world.spec.side_one,world.spec.side_two)):
            raise ValueError('full-six-member-source-world-required')
        moods=[]
        for side in ('p1','p2'):
            values=[]
            for packed in start.player_teams[side].split(']'):
                # Showdown's ORIGINAL packed-set grammar: omitted happiness
                # means255. This describes sampled sets, never original IVs.
                value=packed.split('|')[11]
                values.append(255 if value=='' else int(value))
            moods.append(tuple(values))
        own_moods=[]
        for name in world.party_species[f'p{history.owner+1}']:
            matches=[i for i,species in enumerate(birth['species'])
                if int(species)>0 and normalize_id(metadata['vocabulary']['species'][int(species)])==normalize_id(name)]
            if len(matches)!=1:raise ValueError('own-happiness-source-party-join')
            own_moods.append(own_static.happiness[matches[0]])
        moods[history.owner]=tuple(own_moods)
        mapped=map_spec(world.spec,metadata=metadata,dex=catalog.dex,seed=seed,
            turn=payload['turn'],active_turns=_chronology(history),happiness=tuple(moods))
        if mapped.state is None:return Result(None,mapped.refusals)
        physical_gender=np.array(mapped.state.gender,copy=True)
        for row in state.self_request['side']['pokemon']:
            name=normalize_id(row['details'].split(',')[0]);side=history.owner
            slots=[i for i,value in enumerate(world.party_species[f'p{side+1}']) if normalize_id(value)==name]
            if len(slots)!=1:raise ValueError('same-own-public-root-party')
            slot=slots[0];head=row['condition'].split()[0]
            if head=='0':
                if int(mapped.state.hp[0,side,slot])!=0:raise ValueError('same-own-public-root-faint')
            else:
                hp,maximum=map(int,head.split('/'))
                if (int(mapped.state.hp[0,side,slot]),int(mapped.state.max_hp[0,side,slot]))!=(hp,maximum):
                    raise ValueError('same-own-public-root-exact-hp')
            for index,key in enumerate(('atk','def','spa','spd','spe'),1):
                if int(mapped.state.stats[0,side,slot,index])!=row['stats'][key]:raise ValueError('same-own-public-root-stat:'+key)
            if int(mapped.state.gender[0,side,slot])==-1:
                # ORIGINAL source completeness rejects physical gender-1.
                # Thus a complete owning packet with no M/F is source gender0,
                # not an invitation to draw a replacement gender.
                physical_gender[0,side,slot]=0
        public=public_carry(history,runtime) if retained is None else retained
        # A supplied root carry must match the exact raw history, not merely
        # have plausible shape/tags. No caller accepted flag is consumed.
        if retained is not None:
            expected=public_carry(history,runtime)
            if any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in
                    zip(jax.tree.leaves(expected),jax.tree.leaves(public),strict=True)):
                raise ValueError('same-world-independent-root-public-carry-required')
        _source_foe(root,start,history,catalog,public)
        foe=1-history.owner;physical=np.array(mapped.state.pp,copy=True)
        for slot,name in enumerate(world.party_species[f'p{foe+1}']):
            matches=[i for i,sid in enumerate(np.asarray(public.mons.species)[0,foe])
                if int(sid)>0 and normalize_id(catalog.species.names[int(sid)])==normalize_id(name)]
            if len(matches)>1:raise ValueError('ambiguous-public-opponent-encounter')
            if not matches:continue
            public_slot=matches[0]
            hp=int(mapped.state.hp[0,foe,slot]);maximum=int(mapped.state.max_hp[0,foe,slot])
            shown='0 fnt' if hp==0 else _public_condition(f'{hp}/{maximum}')
            expected='0 fnt' if bool(public.mons.fainted[0,foe,public_slot]) else (
                str(int(public.mons.hp_numerator[0,foe,public_slot]))+'/'+str(int(public.mons.hp_denominator[0,foe,public_slot])))
            if shown!=expected:raise ValueError('sampled-hp-does-not-reproduce-public-bin')
            ids=np.asarray(public.context.moves.move_ids)[0,foe,public_slot]
            uses=np.asarray(public.context.moves.uses)[0,foe,public_slot]
            for index,move in enumerate((world.spec.side_one,world.spec.side_two)[foe].pokemon[slot].moves):
                if move.id=='none':continue
                name=_base_move(move.id)
                spent=sum(int(count) for mid,count in zip(ids,uses) if int(mid)>0 and _base_move(catalog.moves.names[int(mid)])==name)
                # The typed mapper's full foe PP is only the source max; all
                # actual public charges are now subtracted without truncating.
                if spent>physical[0,foe,slot,index]:raise ValueError('public-opponent-PP-charge-exceeds-source-max')
                physical[0,foe,slot,index]-=spent
        mapped_state=mapped.state._replace(pp=j.asarray(physical),gender=j.asarray(physical_gender))
        body=dict(scope='HYPOTHETICAL_PUBLIC_WORLD_NOT_ORDINARY_SESSION',root=root.owner_lines_sha256,
            engine=np.asarray(runtime['request']['rules'].device.engine_binding).tolist(),
            sampled_teams=dict(start.player_teams),seed=seed)
        digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return Result(HypotheticalRoot(mapped_state,public,world,prepared,history,root,digest),())
    except EngineWorldUnsupported as error:return Result(None,('original-public-mapper:'+error.reason,))
    except (ValueError,KeyError,IndexError) as error:return Result(None,(str(error),))
