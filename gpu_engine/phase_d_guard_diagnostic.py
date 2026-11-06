"""Collect independent Phase D guard evidence, never admit a diagnostic world.

A failed prerequisite never licenses a fabricated root, counter or array.
Sample/spec-dependent checks are explicitly NOT_EVALUABLE. This diagnostic
does not alter ordinary constructor admission and makes no coverage claim.
"""
from collections import Counter


class Checks:
    def __init__(self):self.rows=[]
    def add(self,kind,failed,**details):
        self.rows.append(dict(kind=kind,status='NOT_EVALUABLE' if failed is None else 'TRIGGERED' if failed else 'CLEAR',details=details))
    def call(self,kind,fn,**details):
        try:self.add(kind,bool(fn()),**details)
        except (ValueError,KeyError,TypeError,IndexError,AttributeError) as error:
            # A typed refusal is evidence, while an absent prerequisite is not.
            if getattr(error,'kind',None)==kind:self.add(kind,True,error=str(error),**details)
            else:self.add(kind,None,error=type(error).__name__+': '+str(error),**details)


def hp_range_invalid(hp,maximum):
    return None if hp is None else not 0<=hp<=maximum


def own_toxic_check(checks,condition,convert):
    if not condition:checks.add('PUBLIC_TOXIC_STAGE_INVALID',None,reason='literal own status unavailable')
    elif next((x for x in condition[1:] if x in ('slp','par','brn','frz','psn','tox')),'')=='tox':
        checks.call('PUBLIC_TOXIC_STAGE_INVALID',lambda:convert()<0)
    else:checks.add('PUBLIC_TOXIC_STAGE_INVALID',False,applicability='NOT_APPLICABLE_OWN_STATUS_NOT_TOX')


def payload_checks(request,payload,lines,*,owner,checks=None):
    c=Checks() if checks is None else checks
    party=request.get('side',{}).get('pokemon')
    c.add('OWN_REQUEST_OWNER_MISMATCH',request.get('side',{}).get('id')!='p'+str(owner+1))
    c.add('OWN_COMPLETE_ORDINARY_REQUEST_REQUIRED',not isinstance(party,list) or len(party)!=6 or len(request.get('active',()))!=1)
    c.add('SUSPENDED_OR_REPLACEMENT_BOUNDARY',bool(request.get('wait')or request.get('teamPreview')or any(request.get('forceSwitch',()))))
    party=party if isinstance(party,list)else []
    active=[i for i,row in enumerate(party)if isinstance(row,dict)and row.get('active')is True]
    c.add('OWN_ACTIVE_UNAVAILABLE',len(active)!=1)
    current=request.get('active',())
    c.add('OWN_TRAP_STATE_UNAVAILABLE',bool(current[0].get('maybeTrapped')and 'trapped'not in current[0])if len(current)==1 else None)
    for slot,row in enumerate(party or ()):
        if not isinstance(row,dict):c.add('OWN_PARTY_ROW_MALFORMED',True,slot=slot);continue
        # Sleep is now a deferred public posterior, not an unconditional refusal.
    c.add('native-probed-hard-trapped-own-request',bool(payload.get('selfActiveRequestState',{}).get('trapped')))
    c.add('deferred-baton-phase',bool(payload.get('pendingBatonPassSides')))
    for side,data in payload.get('sides',{}).items():
        vol=set(data.get('volatiles',()))
        c.add('native-probed-or-hidden-volatile-clock',bool(data.get('meanlookTrap')or vol&{'trapped','solarbeam','encore'}),side=side,volatiles=sorted(vol))
        # Discharge requires positive B2 removal evidence; evaluated below by
        # the same protocol input predicates production uses.
    since=[None,None];bad_owner=False
    for line in lines:
        p=line.split('|')
        if len(p)>2 and p[1]in ('switch','drag','replace'):
            if p[2][:2]not in ('p1','p2'):bad_owner=True
            else:since[int(p[2][1])-1]=0
        elif len(p)>1 and p[1]=='turn':since=[None if x is None else x+1 for x in since]
    c.add('public-active-owner-unavailable',bad_owner)
    c.add('public-active-arrival-unavailable',any(x is None for x in since))
    return c


def scan(adapter,request,public,source,metadata,*,lines,checks=None):
    import numpy as np
    from pokezero.dex import normalize_id
    from . import phase_d_live_worlds as w,public_context
    hooks=w.hooks
    c=payload_checks(request,public.materialization_payload,lines,owner=public.address.owner,checks=checks)
    owner=public.address.owner;foe=1-owner;catalog=adapter.catalog;state=adapter.state
    c.add('PUBLIC_SOURCE_BINDING_DIFFERS',catalog.source.source_hash!=source.source_hash)
    try:adapter.encode(request,owner_side=owner);c.add('B2_OWNING_OBSERVATION_REFUSED',False)
    except (ValueError,TypeError,KeyError,IndexError)as error:c.add('B2_OWNING_OBSERVATION_REFUSED',True,error=str(error))
    party=request.get('side',{}).get('pokemon',[]);party=party if isinstance(party,list)else []
    current=request.get('active',())
    own_names=[];joined={};own_moves={}
    public_names=[normalize_id(catalog.species.names[int(x)])for x in w._host(state.mons.species)[0,owner]]
    encounter=w._host(state.mons.encounter)[0,owner]
    for slot,mon in enumerate(party):
        if not isinstance(mon,dict):continue
        details=str(mon.get('details','')).split(',');name=normalize_id(details[0]);own_names.append(name)
        condition=str(mon.get('condition','')).split();moves=mon.get('moves',());stats=mon.get('stats');ability=mon.get('baseAbility')or mon.get('ability')
        hp=maximum=None
        try:
            if condition and condition[0]=='0'and'fnt'in condition:hp=0
            elif condition and '/'in condition[0]:hp,maximum=map(int,condition[0].split('/'))
            c.add('OWN_EXACT_HP_UNAVAILABLE',hp is None,slot=slot)
        except (ValueError,TypeError):c.add('OWN_EXACT_HP_UNAVAILABLE',True,slot=slot)
        stats_ok=isinstance(stats,dict)and all(type(stats.get(k))is int and stats[k]>0 for k in ('atk','def','spa','spd','spe'))
        moves_ok=isinstance(moves,(list,tuple))and 1<=len(moves)<=4 and all(type(x)is str for x in moves)
        c.add('OWN_EXACT_STATS_UNAVAILABLE',not stats_ok,slot=slot)
        c.add('OWN_MOVES_UNAVAILABLE',not moves_ok,slot=slot)
        c.add('OWN_ABILITY_OR_ITEM_UNAVAILABLE',not ability or 'item'not in mon,slot=slot)
        universe=source.original.universe_for(name);c.add('OWN_STATIC_SOURCE_UNAVAILABLE',universe is None,slot=slot)
        matches=[i for i,n in enumerate(public_names)if n==name and encounter[i]>0]
        c.add('OWN_PUBLIC_IDENTITY_AMBIGUOUS',len(matches)>1,slot=slot)
        if len(matches)==1:joined[slot]=matches[0]
        identities=set();typed=[];resolved=None
        if universe is not None and stats_ok and moves_ok and ability:
            try:
                level=next((int(x.strip()[1:])for x in details[1:]if x.strip().startswith('L')),100)
                gender=next((x.strip()for x in details[1:]if x.strip()in('M','F')),'N')
                info=source.original.species_metadata[w._normalize_species_id(name)]
                for variant in universe.variants:
                    if variant.level!=level or sorted(map(w._source_move,variant.moves))!=sorted(map(w._source_move,moves))or normalize_id(variant.ability)!=normalize_id(ability):continue
                    fixture=w._fixture_from_variant_payload(variant.to_summary(),fallback_species=name,set_source=source.original,gender=gender)
                    if fixture is None:continue
                    base=info['baseStats'];st=[w.gen3_stat(base[k],fixture.ivs[k],fixture.evs[k],level)for k in('atk','def','spa','spd','spe')]
                    mx=1 if base['hp']==1 else w.gen3_hp_stat(base['hp'],fixture.ivs['hp'],fixture.evs['hp'],level)
                    if st!=[stats[k]for k in('atk','def','spa','spd','spe')]or maximum is not None and maximum!=mx:continue
                    move=next((x for x in fixture.moves if normalize_id(x).startswith('hiddenpower')),None)
                    hp_move=w.hidden_power_engine_id(normalize_id(move),fixture.ivs)if move else None
                    from .phase_d_constructor_rules import literal_hidden_power_agrees
                    typed_literals=tuple(normalize_id(x) for x in moves if normalize_id(x).startswith('hiddenpower') and normalize_id(x)!='hiddenpower')
                    if not literal_hidden_power_agrees(typed_literals,hp_move):continue
                    identities.add((mx,tuple(st),tuple(info['types']),hp_move))
                typed=[normalize_id(x)for x in moves if normalize_id(x).startswith('hiddenpower')and normalize_id(x)!='hiddenpower']
                c.add('OWN_STATIC_IV_FACTS_UNAVAILABLE_OR_AMBIGUOUS',len(identities)!=1,slot=slot,stat_compatible_identities=len(identities),typed_literals=typed,
                    typed_compatible_identities=sum(any(identity[3]and normalize_id(identity[3]).startswith(literal)for literal in typed)for identity in identities))
                if len(identities)==1:
                    mx,_,_,resolved=next(iter(identities));c.add('OWN_EXACT_HP_INVALID',hp_range_invalid(hp,mx),slot=slot,reason='literal own exact HP required' if hp is None else 'literal exact HP and checked maximum available')
            except (ValueError,KeyError,TypeError,IndexError)as error:c.add('OWN_STATIC_IV_FACTS_UNAVAILABLE_OR_AMBIGUOUS',None,slot=slot,error=str(error))
        else:c.add('OWN_STATIC_IV_FACTS_UNAVAILABLE_OR_AMBIGUOUS',None,slot=slot,reason='missing original static prerequisites')
        if moves_ok:
            try:w._own_happiness(mon,moves,name);c.add('OWN_HAPPINESS_UNAVAILABLE',False,slot=slot)
            except w.Refusal as error:c.add(str(error).split(':')[0],True,slot=slot)
            ids=[]
            for move in moves:
                if len(identities)!=1:c.add('OWN_CHECKED_MOVE_NAMESPACE',None,slot=slot,move=move);continue
                try:
                    mid=w._own_move(source.vocabulary['moves'],move,resolved);ids.append(mid)
                    pp=metadata['moves'][mid].get('pp');c.add('OWN_MOVE_MAX_PP_UNAVAILABLE',type(pp)is not int or pp<=0,slot=slot,move=move)
                except w.Refusal as error:c.add(error.kind if hasattr(error,'kind')else str(error).split(':')[0],True,slot=slot,move=move)
            own_moves[slot]=tuple(moves)
            if mon.get('active')is True:
                for move in moves:
                    rows=[r for r in current[0].get('moves',())if w._source_move(r.get('id',''))==w._source_move(move)]if len(current)==1 else []
                    c.add('OWN_ACTIVE_PP_UNAVAILABLE',len(rows)!=1 or type(rows[0].get('pp'))is not int if len(current)==1 else None,slot=slot,move=move,reason='Requires an ordinary active request')
    active=[i for i,m in enumerate(party)if isinstance(m,dict)and m.get('active')is True]
    c.add('OWN_ACTIVE_PUBLIC_IDENTITY_UNAVAILABLE',active[0]not in joined if len(active)==1 else None)
    ctx,_,ok=public_context.observation_boundary_copy(state.context,state.beliefs,state.mons,catalog.source.device,catalog.context,valid=w.j.ones(1,bool))
    boundary_ok=bool(w._host(ok).all());c.add('B2_OBSERVATION_BOUNDARY_COPY_REFUSED',not boundary_ok)
    mons=state.mons;present=w._host(mons.encounter)>0;num,den=w._host(mons.hp_numerator),w._host(mons.hp_denominator)
    for slot in range(6):
        c.add('OPPONENT_LITERAL_PUBLIC_PERCENT_REQUIRED',bool(present[0,foe,slot] and ((num[0,foe,slot]!=0 or den[0,foe,slot]!=0) if w._host(mons.fainted)[0,foe,slot] else den[0,foe,slot]!=100)),slot=slot,numerator=int(num[0,foe,slot]),denominator=int(den[0,foe,slot]),bare_fainted=bool(num[0,foe,slot]==den[0,foe,slot]==0))
    if boundary_ok:
        c.add('PUBLIC_TRANSFORM_NOT_MATERIALIZED',bool(np.any(w._host(ctx.abilities.transformed))))
        c.add('PUBLIC_LIVE_RETYPE_NOT_MATERIALIZED',bool(np.any(w._host(ctx.types.overridden))))
        slots=w._host(mons.active)
        for side in(0,1):
            at=int(slots[0,side]);c.add('PUBLIC_ACTIVE_UNAVAILABLE',not 0<=at<6,side=side)
            for slot in range(6):
                c.add('PUBLIC_MORE_THAN_FOUR_SOURCE_MOVES',int(np.count_nonzero(w._host(ctx.moves.move_ids)[0,side,slot]))>4,side=side,slot=slot)
                if present[0,side,slot]:c.add('PUBLIC_TENDENCY_IDENTITY_UNAVAILABLE',np.count_nonzero(w._host(ctx.tendency.species)[0,side]==w._host(mons.species)[0,side,slot])!=1,side=side,slot=slot)
        own_public=int(slots[0,owner])
        if 0<=own_public<6:
            vols=[hooks.VOLATILE_NAMES[i]for i in np.flatnonzero(w._host(ctx.active.volatiles)[0,owner])]
            unsupported=[x for x in vols if x not in ('','leechseed','ingrain','trapped')]
            c.add('OWN_VOLATILE_OR_LATENT_CLOCK_UNAVAILABLE',bool(unsupported),volatiles=unsupported)
            c.call('PUBLIC_PROTECT_STREAK_INVALID',lambda:w._physical_protect_counter(w._host(ctx.active.stall)[0,owner])<0)
            own_packet=party[active[0]] if len(active)==1 else {}
            own_condition=str(own_packet.get('condition','')).split()
            own_toxic_check(c,own_condition,lambda:w._physical_toxic_counter(w._host(ctx.toxic.stage)[0,owner]))
            ability=int(w._host(ctx.abilities.running)[0,owner,own_public])
            own_packet=party[active[0]]if len(active)==1 else {}
            ability_name=catalog.abilities.names[ability]if ability else own_packet.get('baseAbility')or own_packet.get('ability','')
            c.add('OWN_TRUANT_PHASE_UNAVAILABLE',normalize_id(ability_name)=='truant'and not bool(w._host(ctx.active.truant_known)[0,owner])if ability_name else None)
            if len(active)==1 and active[0]in own_moves:
                kind=int(w._host(ctx.moves.last_kind)[0,owner]);mid=int(w._host(ctx.moves.last_id)[0,owner])
                names=[w._source_move(x)for x in own_moves[active[0]]]
                c.add('OWN_LAST_MOVE_SLOT_UNAVAILABLE',kind==2 and names.count(w._source_move(catalog.moves.names[mid]))!=1)
        c.add('MIDTURN_DAMAGE_WINDOW_NOT_MATERIALIZED',bool(np.any(w._host(ctx.damage.current_dealt))or np.any(w._host(ctx.damage.current_taken))))
        field=ctx.field;wish=w._host(field.wish_set_turn);turn=w._host(field.turn)[:,None]
        c.add('PUBLIC_WISH_CASTER_AMOUNT_UNAVAILABLE',bool(np.any((wish>=0)&(turn-wish<=1))))
        weather=catalog.field.weather_names[int(w._host(field.weather)[0])]
        c.add('PUBLIC_WEATHER_UNAVAILABLE',weather not in('','sunnyday','raindance','sandstorm','hail'))
    else:c.add('PUBLIC_BOUNDARY_DEPENDENT_TAIL_GUARDS',None)
    from . import phase_d_protocol_world as protocol
    try:detached=w.from_b2(adapter,request,source,metadata,address=public.address,defer_sleep=True)
    except (ValueError,KeyError,IndexError,TypeError,AttributeError) as error:
        c.add(getattr(error,'kind',None) or str(error).split(':',1)[0],True,original_refusal=str(error))
        c.add('PROTOCOL_INPUTS_DEPENDENT_CHECKS',None,reason='Original from_b2 refused; no detached surrogate made')
    else:
        for check in protocol.diagnostic_inputs(public,detached,lines=lines,metadata=metadata):
            c.rows.append({**check,'details':{k:v for k,v in check.items() if k not in ('kind','status','fatal')}})
    return dict(checks=c.rows,all_refusals=sorted({r['kind']for r in c.rows if r['status']=='TRIGGERED' and r.get('fatal',True)}),
                not_evaluable=sorted({r['kind']for r in c.rows if r['status']=='NOT_EVALUABLE'}),coverage_claim=False)


def summarize(records):
    tables={};occurrences=Counter();unavailable=Counter();bands=Counter();statuses={}
    for record in records:
        band=str(record['band']);table=tables.setdefault(band,Counter())
        bands[band]+=1
        table.update(set(record['diagnostic']['all_refusals']))
        occurrences.update(r['kind']for r in record['diagnostic']['checks']if r['status']=='TRIGGERED')
        unavailable.update(set(record['diagnostic']['not_evaluable']))
        # A kind may occur for several slots: retain every root-level status,
        # rather than letting one unavailable slot hide another known trigger.
        per_kind={}
        for row in record['diagnostic']['checks']:per_kind.setdefault(row['kind'],set()).add(row['status'])
        for kind,found in per_kind.items():
            counts=statuses.setdefault(band,{}).setdefault(kind,Counter())
            counts.update(found)
    return dict(roots=len(records),roots_affected_by_band={k:dict(v)for k,v in tables.items()},
                roots_by_band=dict(bands),guard_status_roots_by_band={b:{k:dict(v)for k,v in kinds.items()}for b,kinds in statuses.items()},
                guard_status_accounting='Per kind/status unique roots; statuses may overlap across distinct slots within one root. A missing kind is not CLEAR.',
                trigger_occurrences=dict(occurrences),not_evaluable_roots=dict(unavailable),coverage_claim=False,
                scope='Independent current Phase D guard census; unknown prerequisites retained; no coverage or fidelity claim')
