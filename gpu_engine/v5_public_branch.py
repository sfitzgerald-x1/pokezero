"""One ORIGINAL kernel/FULL fold from a separately materialized hypothesis.

There is no ordinary Session registration or model callback. Our retained root
is shared across worlds. Opponent inputs are reconstructed from shared public
history plus that sampled world's OWN request, never the live peer's data.
"""
from dataclasses import dataclass
import hashlib
import json

import jax
import jax.numpy as j
import numpy as np

from pokezero.dex import normalize_id
from pokezero.saved_player_projection import _public_condition
from . import engine_retained_encoder_driver as driver, engine_lifecycle_callbacks as life
from . import engine_request_source as source, engine_request_order as order, engine_private_actions as actions
from . import engine_full_v4_event_device as full, engine_lifecycle_event_device as causal
from . import engine_execution_completeness as execution, engine_kernel_frame_risk as risks
from . import engine_source_frame_witness as witness, engine_public_events as events
from . import public_wire, public_inputs, v4_encoder, v5_observation, private_requests
from . import v5_owning_history as capture
from .v5_public_world import HypotheticalRoot,fold_public


def _physical(branch):
    """Hypothetical party -> original PUBLIC encounter indices only."""
    public=branch.public;catalog=branch.history.catalog
    mapping=np.full((1,2,6),-1,np.int32)
    for side in (0,1):
        names=branch.world.party_species[f'p{side+1}']
        for slot,species in enumerate(np.asarray(public.mons.species)[0,side]):
            if not int(species):continue
            name=normalize_id(catalog.species.names[int(species)])
            candidates=[p for p,value in enumerate(names) if normalize_id(value)==name]
            if len(candidates)!=1:raise ValueError('hypothetical-public-encounter-join')
            mapping[0,side,candidates[0]]=slot
    return mapping


def core(branch, runtime):
    if type(branch) is not HypotheticalRoot:raise ValueError('separate hypothetical root required')
    state=branch.state;public=branch.public;history=branch.history
    streams=j.array([history.stream_id],j.int32);mapping=_physical(branch)
    generation=np.zeros((1,2),np.int32);permutations=[list(range(6)),list(range(6))];seen=[False,False]
    for frame in history.frames:
        for chunk,valid in zip(frame.words,frame.valid):
            for word,use in zip(chunk,valid):
                if not use or word[0] not in (public_inputs.Kind.SWITCH,public_inputs.Kind.DRAG,public_inputs.Kind.REPLACE):continue
                side=word[6];incoming=np.flatnonzero(mapping[0,side]==word[7])
                if len(incoming)!=1:raise ValueError('public-switch-hypothetical-party-join')
                incoming=int(incoming[0]);position=permutations[side].index(incoming)
                if not seen[side]:
                    if incoming!=0:raise ValueError('sampled-initial-lead-order')
                    seen[side]=True
                else:
                    permutations[side][0],permutations[side][position]=permutations[side][position],permutations[side][0]
                    generation[0,side]+=1
    if any(permutations[side][0]!=int(state.active[0,side]) for side in (0,1)):
        raise ValueError('current-hypothetical-active-history')
    active=np.asarray(state.active);slots=mapping[np.arange(1)[:,None],np.arange(2)[None,:],active]
    if np.any(slots<0):raise ValueError('unobserved-active-hypothesis')
    binding=j.broadcast_to(runtime['request']['rules'].device.engine_binding,(1,8))
    cursor=life.Cursor(streams,binding,j.asarray(generation),state.turn+1,j.zeros(1,j.int32),j.ones(1,bool),j.ones(1,bool),
        state.active,j.asarray(slots),j.asarray(slots+1),j.ones((1,2),bool),
        j.take_along_axis(state.species,state.active[...,None],axis=2)[...,0],j.asarray(mapping),
        j.sum(public.mons.encounter>0,axis=2,dtype=j.int32))
    if not bool(life._valid(cursor).all()):raise ValueError('hypothetical-lifecycle-cursor')
    ledgers=[];constructors=[];originals=[];requests=[]
    metadata=json.loads(runtime['request']['metadata'])
    pp=np.array([int((row or {}).get('pp',0))*8//5 for row in metadata['moves']],np.int32)
    for side in (0,1):
        ledger=order.initialize_order(streams,j.array([side],j.int32),j.array([6],j.int32),engine_binding=binding)
        ledger=ledger._replace(request_to_physical=j.array([permutations[side]],j.int32),generation=j.asarray(generation[:,side]))
        original=source.Original(*(getattr(state,name)[:,side] for name in source.Original._fields))
        digest=hashlib.sha256((branch.origin_sha256+':own-request:'+str(side)).encode()).digest()
        words=j.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32))[None,:]
        maximum=pp[np.asarray(state.moves[:,side])]
        if side==history.owner:
            birth=dict(history.own_constructor)
            for slot,sid in enumerate(np.asarray(state.species)[0,side]):
                candidates=[i for i,value in enumerate(birth['species']) if value==int(sid)]
                if len(candidates)!=1:raise ValueError('own-exact-constructor-maxPP-join')
                maximum[0,slot]=birth['max_pp'][candidates[0]]
        constructor=actions.ConstructorData(words,streams,j.array([side],j.int32),binding,
            state.species[:,side],state.moves[:,side],j.asarray(maximum))
        produced=runtime['request_fast'](state,ledger,original,constructor,j.zeros(1,j.int32),
            life.snapshot(cursor),j.ones(1,bool))
        if not bool(np.asarray(produced.actions.admitted).all()):raise ValueError('hypothetical-own-request-refused:'+str(side))
        if side==history.owner:
            legal=private_requests.action_inputs(produced.actions.facts.request,observer_side=side).legal
            if tuple(map(bool,np.asarray(legal)[0]))!=tuple(branch.root.context.observation.legal_action_mask):
                raise ValueError('same-own-root-legal-action-denominator-required')
        ledgers.append(ledger);constructors.append(constructor);originals.append(original);requests.append(produced)
    account=execution.initialize(streams,compiled=runtime['execution_compiled'])
    risk=risks.initialize_device(streams,runtime['kernel_risk_compiled'].device)
    return driver.Core(state,cursor,tuple(ledgers),tuple(originals),tuple(constructors),
        (j.zeros(1,j.int32),j.zeros(1,j.int32)),tuple(requests),j.ones(1,bool),j.zeros(1,j.int32),None,account,risk)


def _opponent_public(branch, core, runtime):
    """Hypothetical opponent-own root, without an actual opposing request.

    Public exact HP belonging to OUR seat is downgraded using the original
    owner-wire visibility helper. Our private Pressure Start is absent. Prior
    opposing HP is public-only; current exact own HP enters solely through the
    sampled owner's request. Unexpressed private-clock histories are refused.
    """
    decoded=capture.decode(branch.history)
    if not decoded.complete:raise ValueError('hypothetical-opponent-public-history-incomplete')
    # Full-health reconstruction rests on ORIGINAL capture's constructor0
    # lineage and complete uninterrupted frames, plus no later HP event below.
    # A standalone percentage100, even on an arrival, is not that proof.
    if (not branch.history.frames or tuple(frame.number for frame in branch.history.frames)!=tuple(range(len(branch.history.frames)))
            or branch.history.session_identity[1]!=id(branch.prepared)):
        raise ValueError('opponent-fresh-birth-history-unproved')
    mine=f'p{branch.history.owner+1}';peer=1-branch.history.owner;lines=[]
    for line in decoded.lines:
        if line.startswith('|request|'):continue
        parts=line.split('|')
        if len(parts)>3 and parts[1] in ('-ability','ability') and parts[2].startswith(mine) and normalize_id(parts[3])=='pressure':continue
        if parts[1] in ('-status','-curestatus','cant') or (parts[1]=='-start' and 'confusion' in parts):
            raise ValueError('opponent-own-private-clock-history-unrepresented')
        positions=[]
        if len(parts)>4 and parts[1] in ('switch','drag','replace'):positions=[(2,4)]
        if len(parts)>3 and parts[1] in ('-damage','-heal','-sethp'):positions=[(2,3)]
        if parts[1]=='-sethp' and len(parts)>5:positions.append((4,5))
        for ident,condition in positions:
            if parts[ident].startswith(mine):parts[condition]=_public_condition(parts[condition])
            elif parts[ident].startswith(f'p{peer+1}'):
                # Public percentage damage does not disclose the opponent's
                # prior private exact-HP/tendency history. Never fabricate it.
                # Only the proven full-health birth with NO intervening HP
                # event makes this arrival exact in every sampled world.
                if parts[1] not in ('switch','drag','replace') or parts[condition].split()[0]!='100/100':
                    raise ValueError('opponent-own-private-hp-history-unrepresented')
                name=normalize_id(parts[3].split(',')[0])
                slots=[i for i,value in enumerate(branch.world.party_species[f'p{peer+1}']) if normalize_id(value)==name]
                if len(slots)!=1:raise ValueError('opponent-own-arrival-source-species')
                maximum=int(branch.state.max_hp[0,peer,slots[0]])
                parts[condition]=f'{maximum}/{maximum}'
        lines.append('|'.join(parts))
        if parts[1] in ('switch','drag','replace') and parts[2].startswith(f'p{peer+1}'):
            # Gen3's ORIGINAL owner-only onStart schema. This is that sampled
            # owner's private ability, never a revelation to our actor. All HP
            # change/hidden-clock entry histories above already refuse.
            slot=slots[0]
            mon=(branch.world.spec.side_one,branch.world.spec.side_two)[peer].pokemon[slot]
            if normalize_id(mon.ability or '')=='pressure':
                lines.append(f'|-ability|{parts[2]}|Pressure|[silent]')
    wire,packets=public_wire.project_lines(public_wire.WireState(),lines,**v4_encoder.wire_kwargs(runtime['catalog']))
    namespaces=runtime['public'].device.namespaces._asdict()
    chunks=public_inputs.pack_chunks(np.array([branch.history.stream_id],np.int32),
        [[event for packet in packets for event in packet.events]],**namespaces)
    result=v4_encoder.initialize(runtime['catalog'],stream_ids=core.cursor.stream_id,owner_sides=j.array([peer],j.int32))
    for chunk in chunks:
        result,error=fold_public(result,chunk,runtime['catalog'])
        if bool(np.asarray(error).any()):raise ValueError('hypothetical-opponent-public-fold-refused')
    return result


@dataclass(frozen=True)
class PolicyView:
    inputs: dict  # Original public metadata + five model tensors only.
    legal_action_mask: object  # Separate, same-request full reply denominator.


def policy_observation(branch, adapter, *, owner):
    """Original own-facts/retained encoder and append-only public29 adapter."""
    runtime=driver._checked(branch.prepared);current=core(branch,runtime)
    if type(owner) is not int or owner not in (0,1):raise ValueError('literal owner required')
    if v5_observation.check_adapter(adapter) is not runtime['catalog']:
        raise ValueError('same original public v5 adapter namespace required')
    public=branch.public if owner==branch.history.owner else _opponent_public(branch,current,runtime)
    facts,encoded=runtime['encode_fast'](current,_observer(public,owner))
    appended=v5_observation.append_device(encoded,public,adapter,observer_sides=j.array([owner],j.int32))
    if not bool(np.asarray(facts.admitted&encoded.observation.admitted&appended.admitted).all()):
        raise ValueError('hypothetical-public-policy-observation-refused')
    return PolicyView(v5_observation.actor_inputs(appended,adapter),appended.legal_action_mask)


def _stream(public, cursor, owner, runtime):
    result=full.register(cursor.stream_id,j.array([owner],j.int32),compiled=runtime['public'])
    lower=causal.StreamState(cursor,j.array([owner],j.int32),j.zeros(1,j.int32),j.ones(1,bool))
    field=public.context.field
    live=(public.mons.active>=0)&~j.take_along_axis(public.mons.fainted,j.maximum(public.mons.active,0)[...,None],axis=2)[...,0]
    return result._replace(causal=lower,sequence=public.cursor.sequence,live=live,
        species=public.mons.species,status=public.mons.status,hp=public.mons.hp_numerator,maximum=public.mons.hp_denominator,
        weather=field.weather,weather_permanent=field.weather_ability,
        weather_remaining=j.where((field.weather>0)&~field.weather_ability,j.maximum(0,5-field.weather_upkeeps),0))


def _observer(public,owner):
    return driver.Observer(j.array([owner],j.int32),None,None,public,j.ones(1,bool),j.zeros(1,j.int32),None)


def _requests_after_frame(runtime,frame,before,publics,orders,admitted):
    """Original request/type proof from PRE-fold owning history and SAME tape.

    This is the owning driver's existing Color Change join, not permission to
    trust private current types. Unsupported root-history reconstruction in
    core() remains refused. Only an existing public override or a genuine
    current claimed receipt can differ from sealed constructor types.
    """
    current_types=(None,None)
    context=runtime['direct_hit_context']
    if context is not None and context.data.color_change is not None and runtime['direct_hit_mode']=='kernel-event':
        from . import engine_color_change_callbacks as color
        current_types=tuple(color.expected_owned_types(frame.state,before.engine,public,
            owner=owner,tape=frame.callbacks,data=context.data.color_change,
            categories=runtime['public'].device.causal.color_categories,
            type_tables=runtime['catalog'].types.device,healthy=before.admitted)
            for owner,public in enumerate(publics))
    return tuple(runtime['request_fast'](frame.state,ledger,original,constructor,unsafe,
        frame.after_registry,admitted,types) for ledger,original,constructor,unsafe,types in
        zip(orders,before.originals,before.constructors,before.unsafe,current_types,strict=True))


@dataclass(frozen=True)
class Step:
    state: object | None
    public: tuple
    encoded: tuple
    refusals: tuple[str,...]
    raw_frame: object | None
    diagnostics: tuple[tuple[str,int],...]=()


def step(branch, choices, *, chance_inputs,hypothetical_profiles=False):
    """One actual original numerical step and both FULL folds, no model.

    Ordinary birth profiles are never fabricated. The explicit hypothetical
    mode uses its own current-origin record and original conditional math.
    Unsupported histories retain refusals. This component is NOT a ready D
    provider or a claim of full counterfactual-clock/profile coverage.
    """
    runtime=driver._checked(branch.prepared)
    try:
        if type(hypothetical_profiles) is not bool:raise ValueError('explicit hypothetical observer mode required')
        current=core(branch,runtime)
        if len(choices)!=2 or any(type(choice) is not int or not 0<=choice<9 for choice in choices):raise ValueError('two exact root/reply action slots required')
        routes=[actions.route_choice(current.requests[owner].actions,current.orders[owner],j.array([choices[owner]],j.int32)) for owner in (0,1)]
        if any(not bool(np.asarray(route.admitted).all()) for route in routes):raise ValueError('hypothetical-action-route-refused')
        publics=[branch.public if owner==branch.history.owner else _opponent_public(branch,current,runtime) for owner in (0,1)]
        streams=[_stream(public,current.cursor,owner,runtime) for owner,public in enumerate(publics)]
        left=runtime['leftovers_context'];status=runtime['status_residual_context']
        profile=None;silent=None
        if hypothetical_profiles:
            from . import v5_hypothetical_profile as hypothesis
            profile=hypothesis.qualify_origin(branch)
            current=current._replace(weather_profile=profile.record)
            silent=None if runtime['no_output_compiled'] is None else runtime['no_output_compiled'].device
            left=None if left is None else hypothesis.LeftoversContext(left.data,profile.data,silent)
            status=None if status is None else hypothesis.StatusContext(status.data,profile.data)
        frame=driver._frame(current,(None,runtime['public'].causal.projector.device,runtime['request']['engine_table']),
            j.stack([route.engine_action for route in routes],axis=1),chance_inputs,j.ones(1,bool),
            leftovers_context=None if left is None else left if hypothetical_profiles else life.LeftoversContext(left.data),
            status_residual_context=None if status is None else status if hypothetical_profiles else life.StatusResidualContext(status.data),
            execution_tables=runtime['execution_compiled'].device,kernel_risk_tables=runtime['kernel_risk_compiled'].device,
            direct_hit_context=runtime['direct_hit_context'],substitute_context=runtime['substitute_context'],
            toxic_context=runtime['toxic_context'],recoil_context=runtime['recoil_context'],
            calm_mind_context=runtime['calm_mind_context'],swords_dance_context=runtime['swords_dance_context'],
            spikes_context=runtime['spikes_context'],thunder_wave_context=runtime['thunder_wave_context'],sleep_powder_context=runtime['sleep_powder_context'],hypothetical_profile=profile)
        if profile is not None:profile=profile._replace(record=frame.weather_profile)
        consumed=[];ledgers=[]
        for stream,public in zip(streams,publics):
            ledger,result=execution.consume_frame_device(frame.callbacks.execution.ledger,frame,stream,public,
                active=j.ones(1,bool),tables=runtime['execution_compiled'].device,
                adapter_tables=runtime['public'].device,catalog=runtime['catalog'],
                leftovers=None if left is None else left.data,status_residual=None if status is None else status.data,
                leftovers_no_output=silent,hypothetical_hp=profile)
            consumed.append(result);ledgers.append(ledger)
        joined,complete=execution.join_consumed_owners(*ledgers,tables=runtime['execution_compiled'].device)
        raw=witness._raw_complete(frame.callbacks,runtime['frame_witness_compiled'].device)
        # The SAME-call risk frame is an actual ordinary hypothetical step,
        # not an invented ordinary constructor/witness carry. Reuse ORIGINAL
        # cutpoint geometry; a missing dispatch remains UNKNOWN even if no
        # unsupported write happened to emit a callback.
        actual_risks=frame.callbacks.execution.risks.carry
        expected,observed,geometry=witness._geometry(actual_risks,frame.callbacks)
        cutpoints=geometry&j.all(actual_risks.counts==expected,axis=1)&(actual_risks.observed==observed)
        good=complete&raw&~frame.invalid_actions&(frame.callbacks.execution.risks.carry.risk==0)
        good &= cutpoints
        good &= (frame.callbacks.refusal & ~events.UNINSTRUMENTED)==0
        diagnostic=(('callback_refusal',int(frame.callbacks.refusal[0])),
            ('kernel_risk',int(frame.callbacks.execution.risks.carry.risk[0])),
            ('accounting_refusal',int(joined.refusal[0])),('raw_complete',int(raw[0])),
            ('cutpoints_complete',int(cutpoints[0])),
            *((f'owner{owner}_FULL_error',int(value.projection.error[0])) for owner,value in enumerate(consumed)))
        if not bool(np.asarray(good).all()):
            return Step(frame.state,tuple(value.public for value in consumed),(),('actual-hypothetical-kernel-or-FULL-refused',),frame,diagnostic)
        ledgers=tuple(runtime['installations'](ledger,frame) for ledger in current.orders)
        requests=_requests_after_frame(runtime,frame,current,publics,ledgers,good)
        after=current._replace(engine=frame.state,cursor=frame.cursor,orders=ledgers,requests=requests,admitted=good)
        encoded=[]
        for owner,value in enumerate(consumed):
            facts,observation=runtime['encode_fast'](after,_observer(value.public,owner))
            if not bool(np.asarray(facts.admitted&observation.observation.admitted).all()):
                return Step(frame.state,tuple(v.public for v in consumed),(),('actual-branch-public-encoding-refused',),frame,diagnostic)
            encoded.append(observation)
        return Step(frame.state,tuple(value.public for value in consumed),tuple(encoded),(),frame,diagnostic)
    except (ValueError,KeyError,IndexError) as error:return Step(None,(),(),(str(error),),None)
