"""Final bounded Phase D tail: positive public facts, no truth donor."""
from dataclasses import replace
from pokezero.dex import normalize_id


def replacement_phase(request,payload,lines):
    if not any(request.get('forceSwitch',())):return None
    party=request.get('side',{}).get('pokemon',());owner=request.get('side',{}).get('id')
    active=[row for row in party if row.get('active')is True]
    if (request.get('forceSwitch')!=[True] or request['forceSwitch'][0] is not True or request.get('active') or request.get('wait') or request.get('teamPreview')
            or len(party)!=6 or len(active)!=1 or str(active[0].get('condition','')).split()!=['0','fnt']
            or payload.get('selfRequestKind')!='force-switch' or payload.get('selfPlayer')!=owner):
        raise ValueError('PUBLIC_REPLACEMENT_PHASE_UNAVAILABLE')
    if payload.get('pendingBatonPassSides'):raise ValueError('PUBLIC_REPLACEMENT_PIVOT_UNSUPPORTED')
    turns=[i for i,line in enumerate(lines) if line.startswith('|turn|')]
    if not turns or lines[turns[-1]]!='|turn|'+str(payload['turn']):raise ValueError('PUBLIC_REPLACEMENT_TURN_UNAVAILABLE')
    window=[line.split('|') for line in lines[turns[-1]+1:] if not line.startswith('|request|')]
    paid=any(len(row)>1 and row[1]=='upkeep' for row in window)
    if any(len(row)>1 and row[1] in ('drag','replace') for row in window):raise ValueError('PUBLIC_REPLACEMENT_QUEUE_UNAVAILABLE')
    if not paid and any(len(row)>1 and (row[1]=='-weather' and '[upkeep]' in row or
            any(tag in '|'.join(row) for tag in ('[from] item: Leftovers','[from] psn','[from] brn','[from] Sandstorm','[from] Hail'))) for row in window):
        raise ValueError('PUBLIC_REPLACEMENT_PARTIAL_RESIDUAL_UNSUPPORTED')
    for side in ('p1','p2'):
        members=[row for row in payload['sides'][side]['pokemon'] if row.get('active')is True]
        if len(members)!=1:raise ValueError('PUBLIC_REPLACEMENT_ACTIVE_UNAVAILABLE')
        condition=str(members[0].get('condition','')).split()
        if not condition:raise ValueError('PUBLIC_REPLACEMENT_ACTIVE_UNAVAILABLE')
        dead=condition==['0','fnt']
        if side==owner and not dead:raise ValueError('PUBLIC_REPLACEMENT_PHASE_DISAGREEMENT')
        acted=any(len(row)>2 and row[2].startswith(side+'a:') and
            (row[1] in ('cant','switch') or row[1]=='move' and not any(x.startswith('[from]') and x!='[from] lockedmove' for x in row[4:])
             or row[1]=='-activate' and len(row)>3 and normalize_id(row[3])=='confusion') for row in window)
        if not paid and not dead and not acted:raise ValueError('PUBLIC_REPLACEMENT_QUEUE_UNAVAILABLE')
    ages=[None,None]
    for line in lines:
        p=line.split('|')
        if len(p)>2 and p[1] in ('switch','drag','replace'):ages[int(p[2][1])-1]=0
        elif len(p)>1 and p[1]=='turn':ages=[None if age is None else age+1 for age in ages]
    if any(age is None for age in ages):raise ValueError('PUBLIC_REPLACEMENT_AGE_UNAVAILABLE')
    return dict(pending_residual=not paid,queued_actor=-1,turn=payload['turn'],ages=tuple(ages))


def fresh_substitutes(payload,lines):
    active={};stints={}
    for line in lines:
        if line.startswith('|request|'):continue
        p=line.split('|')
        if len(p)<3:continue
        side=p[2][:2]
        if side not in ('p1','p2'):continue
        if p[1] in ('switch','drag','replace'):
            active[side]=normalize_id(p[3].split(',')[0]);stints.pop(side,None)
        elif len(p)>3 and p[1]=='move' and normalize_id(p[3])=='batonpass':stints.pop(side,None)
        elif len(p)>3 and normalize_id(p[3])=='substitute':
            if p[1]=='-start':stints[side]=active.get(side)
            elif p[1] in ('-activate','-end'):stints.pop(side,None)
        elif p[1]=='faint':stints.pop(side,None)
    result={}
    for side,data in payload['sides'].items():
        if 'substitute' not in data.get('volatiles',()):continue
        rows=[row for row in data['pokemon'] if row.get('active')is True]
        name=normalize_id(rows[0]['species']) if len(rows)==1 else ''
        if (not name or data.get('substituteHealthState')!='full' or type(data.get('substituteDepletion'))is not int
                or data['substituteDepletion']!=0 or type(data.get('substituteMinDepletion'))is not int or data['substituteMinDepletion']!=0
                or normalize_id(data.get('substituteOriginSpecies') or '')!=name or stints.get(side)!=name):
            raise ValueError('PUBLIC_SUBSTITUTE_NOT_FRESH_OR_PASSED')
        result[side]=name
    return result


def wish_facts(payload):
    result={}
    for side,start in payload.get('wishSetTurns',{}).items():
        if side not in ('p1','p2') or type(start)is not int or type(payload['turn'])is not int or payload['turn']-start not in (0,1):
            raise ValueError('PUBLIC_WISH_TIMING_UNAVAILABLE')
        result[side]=start
    return result


def inspect_facts(request,payload,lines):
    checks=[];result=dict(turn=payload.get('turn'))
    for key,kind,fn in (
            ('replacement','PUBLIC_REPLACEMENT_PHASE',lambda:replacement_phase(request,payload,lines)),
            ('wish_set_turns','PUBLIC_WISH_TIMING',lambda:wish_facts(payload))):
        try:result[key]=fn()
        except (ValueError,KeyError,IndexError,TypeError) as error:
            result[key]=None;checks.append(dict(kind=str(error),status='TRIGGERED',fatal=True))
        else:checks.append(dict(kind=kind,status='CLEAR',fatal=True))
    result['fresh_substitutes']={}
    for side,data in payload['sides'].items():
        try:result['fresh_substitutes'].update(fresh_substitutes(dict(sides={side:data}),lines))
        except (ValueError,KeyError,IndexError,TypeError) as error:checks.append(dict(kind=str(error),status='TRIGGERED',fatal=True,side=side))
        else:checks.append(dict(kind='PUBLIC_FRESH_SUBSTITUTE',status='CLEAR',fatal=True,side=side))
    return checks,result


def facts(public,request,lines):
    checks,result=inspect_facts(request,public.materialization_payload,lines)
    for check in checks:
        if check['status']=='TRIGGERED':raise ValueError(check['kind'])
    return result


def defer_substitutes(payload,fresh):
    import copy
    projected=copy.deepcopy(payload)
    for side in fresh:
        data=projected['sides'][side];data['volatiles']=[v for v in data['volatiles'] if v!='substitute']
        data.update(substituteHealthState='absent',substituteOriginSpecies=None,substituteDepletion=None,substituteMinDepletion=0)
    return projected


def mapping_spec(spec,phase):
    """Storage-only projection; no invented HP and no new party member.

    The generic mapper requires an ordinary living cursor. Select an existing
    living bench solely while mapping storage, keeping every party HP/status
    literal. Restore the actual cursor and its effects before any exposure.
    """
    if phase is None:return spec,()
    sides=[];cursors=[]
    for side in (spec.side_one,spec.side_two):
        if side.baton_passing or side.slow_uturn_move or side.switch_out_move_second_saved_move:
            raise ValueError('PUBLIC_REPLACEMENT_QUEUE_UNAVAILABLE')
        index=side.active_index;cursors.append(index)
        if side.pokemon[index].hp<=0:
            living=[k for k,mon in enumerate(side.pokemon) if mon.hp>0]
            if not living:raise ValueError('PUBLIC_REPLACEMENT_NO_LIVING_MEMBER')
            # Effects belong to the actual fainted cursor, never the temporary
            # storage cursor. Unsupported effects are not erased by projection.
            if (any(side.boosts.values()) or side.volatile_statuses or side.volatile_status_durations or side.substitute_health
                    or side.side_conditions.get('protect',0) or side.side_conditions.get('toxic_count',0)):
                raise ValueError('PUBLIC_REPLACEMENT_FAINTED_EFFECT_UNSUPPORTED')
            conditions={k:v for k,v in side.side_conditions.items() if k not in ('protect','toxic_count')}
            side=replace(side,active_index=living[0],side_conditions=conditions,last_used_move='')
        sides.append(replace(side,force_switch=False))
    return replace(spec,side_one=sides[0],side_two=sides[1]),tuple(cursors)


def restore(state,original_spec,cursors,tail,*,metadata,checks):
    import jax.numpy as j
    phase=tail['replacement']
    if phase is not None:
        for seat,(side,cursor) in enumerate(zip((original_spec.side_one,original_spec.side_two),cursors)):
            state=state._replace(active=state.active.at[0,seat].set(cursor))
            if side.pokemon[cursor].hp<=0 and side.last_used_move:
                last=side.last_used_move
                if last=='move:struggle':
                    try:mid=metadata['vocabulary']['moves'].index('struggle')
                    except ValueError:raise ValueError('PUBLIC_REPLACEMENT_LAST_MOVE_UNAVAILABLE')
                    state=state._replace(last_action_kind=state.last_action_kind.at[0,seat].set(3),last_move=state.last_move.at[0,seat,cursor].set(mid))
                elif last.startswith('move:'):
                    index=int(last.split(':')[1])
                    if not 0<=index<4 or int(state.moves[0,seat,cursor,index])<=0:raise ValueError('PUBLIC_REPLACEMENT_LAST_MOVE_UNAVAILABLE')
                    mid=state.moves[0,seat,cursor,index]
                    state=state._replace(last_action_kind=state.last_action_kind.at[0,seat].set(2),last_action_index=state.last_action_index.at[0,seat].set(index),last_move=state.last_move.at[0,seat,cursor].set(mid))
                    if normalize_id(side.pokemon[cursor].item or '')=='choiceband':state=state._replace(choice_move=state.choice_move.at[0,seat,cursor].set(mid))
                elif last.startswith('switch:'):state=state._replace(last_action_kind=state.last_action_kind.at[0,seat].set(1),last_action_index=state.last_action_index.at[0,seat].set(cursor))
                else:raise ValueError('PUBLIC_REPLACEMENT_LAST_MOVE_UNAVAILABLE')
        alive=j.any(state.hp>0,axis=2);hp=j.take_along_axis(state.hp,state.active[...,None],axis=2)[...,0]
        state=state._replace(forced_switch=(hp<=0)&alive,pending_residual=j.array([phase['pending_residual']]),
            queued_actor=j.array([-1],j.int32),queued_actions=j.zeros_like(state.queued_actions),queued_stall=j.zeros_like(state.queued_stall),
            turn=j.array([phase['turn']],j.int32),active_turns=j.array([phase['ages']],j.int32))
        if phase['pending_residual']:
            # Unchanged finite law, same kernel key. This is a hypothetical
            # future capsule; it is NEVER labeled recovered original draws.
            from .transition import sample_chance_inputs
            chance=sample_chance_inputs(state)
            state=state._replace(residual_draws=chance.draws,phazing_draws=chance.phazing,
                secondary_draws=chance.secondary,sleep_talk_draws=chance.sleep_talk)
            if 'effect_draws' in state._fields:state=state._replace(effect_draws=chance.effect)
        checks.append(dict(kind='PUBLIC_REPLACEMENT_PHASE',status='CLEAR',fatal=True,**phase,
            randomness='SAMPLED_HYPOTHETICAL_FUTURE_FROZEN_CHANCE_LAW_SAME_KERNEL_KEY'))
    for side,name in tail['fresh_substitutes'].items():
        seat=int(side[1])-1;slot=int(state.active[0,seat]);actual=normalize_id(metadata['vocabulary']['species'][int(state.species[0,seat,slot])])
        if actual!=name:raise ValueError('PUBLIC_SUBSTITUTE_ACTIVE_DISAGREEMENT')
        state=state._replace(substitute_hp=state.substitute_hp.at[0,seat,slot].set(state.max_hp[0,seat,slot]//4))
        checks.append(dict(kind='PUBLIC_FRESH_SUBSTITUTE',status='CLEAR',fatal=True,side=seat,slot=slot))
    for side,start in tail['wish_set_turns'].items():
        seat=int(side[1])-1;slot=int(state.active[0,seat]);remaining=2-(tail['turn']-start)
        state=state._replace(wish_counter=state.wish_counter.at[0,seat].set(remaining),
            wish_amount=state.wish_amount.at[0,seat].set(j.maximum(1,state.max_hp[0,seat,slot]//2)))
        checks.append(dict(kind='PUBLIC_WISH_RECIPIENT_STATE',status='CLEAR',fatal=True,side=seat,remaining=remaining,
            amount_semantics='CURRENT_RECIPIENT_COMPATIBILITY_FIELD;FROZEN_RESIDUAL_USES_RESOLVING_RECIPIENT'))
    return state


def carry_current_damage(history,dealt,taken,owner):
    """Only known B2 cells: shown foe damage and victim-owned exact fractions."""
    foe=1-owner
    return history._replace(taken=history.taken.at[0,owner,1].set(taken[0,owner]).at[0,foe,0].set(taken[0,foe]),
        dealt=history.dealt.at[0,owner,0].set(dealt[0,owner]).at[0,foe,1].set(dealt[0,foe]))
