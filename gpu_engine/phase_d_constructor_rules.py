"""Pure Phase D request-order and public-percent rules; no engine changes."""


def public_hp_bin(maximum, shown):
    """Inverse of Gen3 public ceil percent, capped at99 below full HP.

    100 is exactly full; 99 also includes the below-full ceil100 tail.
    No sampled maximum or exact opponent HP is accepted as public evidence.
    """
    if type(maximum)is not int or maximum<=0 or type(shown)is not int or not 0<=shown<=100:
        raise ValueError('literal positive maxHP and public0..100percent required')
    if shown==0:return 0,0
    if shown==100:return maximum,maximum
    lower=((shown-1)*maximum)//100+1
    upper=maximum-1 if shown==99 else(shown*maximum)//100
    if lower>upper:raise ValueError('sampled maximum has no HP in public bin')
    return lower,upper


def request_ordered_packed(packed, sampled_names, request_names):
    """Move whole literal packed entries, preserving IVs, gender and happiness."""
    entries=packed.split(']');a=tuple(sampled_names);b=tuple(request_names)
    if len(entries)!=len(a)or len(a)!=len(b)or len(a)!=6 or len(set(a))!=6 or len(set(b))!=6 or set(a)!=set(b):
        raise ValueError('same-six-unique-own-species-required-for-request-order')
    by_name=dict(zip(a,entries))
    return ']'.join(by_name[name]for name in b)


def literal_hidden_power_agrees(literals, resolved):
    """The typed own request is evidence before static-identity uniqueness."""
    return all(resolved is not None and resolved.startswith(literal)for literal in literals)


def _move_key(value):
    from pokezero.dex import normalize_id
    from pokezero.tier2 import canonical_move_id
    value=normalize_id(value)
    return 'hiddenpower' if value.startswith('hiddenpower') else canonical_move_id(value)


def _identity(value):
    return str(value).split(':',1)[-1].strip().casefold()


def own_pp_carry(lines,request,maximum_pp):
    """Latest owning active PP, plus only public costs after that snapshot.

    Costs before a later owning request are overwritten by that exact packet.
    The only full-max case is a provably never-active member of the complete
    battle-start prefix. No peer request, sampled ability, or hidden PP is used.
    """
    from pokezero.live_protocol_actor import _packet
    from pokezero.belief import _called_move_source,_CALLER_MOVES,_NEVER_PRESSURED_POOL_MOVES
    from pokezero.dex import normalize_id
    owner=request['side']['id'];party=request['side']['pokemon']
    identities=[_identity(p.get('ident','')) for p in party]
    if len(identities)!=6 or not all(identities) or len(set(identities))!=6 or '|start' not in lines:
        raise ValueError('OWN_PP_COMPLETE_UNIQUE_START_HISTORY_REQUIRED')
    ledger={};seen=set();active={};ability={};last=None
    for line in lines:
        p=line.split('|');kind=p[1] if len(p)>1 else ''
        if kind=='request':
            packet=_packet(line,owner);last=packet
            rows=packet['side']['pokemon']
            if {_identity(r.get('ident','')) for r in rows}!=set(identities):
                raise ValueError('OWN_PP_REQUEST_IDENTITY_CHANGED')
            current=[r for r in rows if r.get('active')is True]
            if len(current)!=1:continue
            identity=_identity(current[0]['ident']);seen.add(identity)
            if packet.get('wait') or any(packet.get('forceSwitch',())) or packet.get('teamPreview'):continue
            moves=packet.get('active',());moves=moves[0].get('moves',()) if len(moves)==1 else ()
            expected=list(map(_move_key,current[0].get('moves',())))
            values={}
            for row in moves:
                key=_move_key(row.get('id',''));pp=row.get('pp');maximum=row.get('maxpp')
                if key in values or type(pp)is not int or type(maximum)is not int or not 0<=pp<=maximum:
                    values={};break
                values[key]=pp
            if values and len(expected)==len(set(expected)) and set(values)==set(expected):ledger[identity]=values
            elif not moves or any(_move_key(row.get('id','')) not in ('struggle','recharge','solarbeam') for row in moves):
                ledger[identity]='OWN_PP_LAST_ACTIVE_REQUEST_UNAVAILABLE'
            # Source-proven pp-less locked Solar Beam/recharge/Struggle packets
            # retain the previous exact PP; public primary/locked move tags
            # below distinguish the one paid declaration from free continuations.
        elif kind in ('switch','drag','replace') and len(p)>3:
            side=p[2][:2];identity=_identity(p[2]);active[side]=identity;ability[side]=''
            if side==owner:
                if identity not in identities:raise ValueError('OWN_PP_PUBLIC_IDENTITY_UNAVAILABLE')
                seen.add(identity)
        elif kind in ('-ability','-endability') and len(p)>3:
            side=p[2][:2]
            if active.get(side)==_identity(p[2]):ability[side]=normalize_id(p[3]) if kind=='-ability' else ''
        elif kind=='move' and len(p)>3 and p[2][:2]==owner:
            identity=_identity(p[2]);seen.add(identity);value=ledger.get(identity)
            if not isinstance(value,dict):continue
            move=_move_key(p[3]);caller=_called_move_source(line)
            if move=='struggle' or caller in _CALLER_MOVES|{'sleeptalk','lockedmove'}:continue
            if caller is not None:
                ledger[identity]='OWN_PP_MOVE_COST_UNAVAILABLE';continue
            if active.get(owner)!=identity or move not in value:
                ledger[identity]='OWN_PP_MOVE_SLOT_UNAVAILABLE';continue
            cost=1+int(ability.get('p2' if owner=='p1' else 'p1')=='pressure' and move not in _NEVER_PRESSURED_POOL_MOVES)
            value[move]=max(0,value[move]-cost)
        if any(token in line.lower() for token in ('spite','grudge','leppaberry','leppa berry')):
            identity=active.get(owner)
            if identity in ledger:ledger[identity]='OWN_PP_NONMOVE_CHANGE_UNAVAILABLE'
    if last!=request:raise ValueError('OWN_PP_CURRENT_OWN_REQUEST_REQUIRED')
    result=[]
    for row,identity in zip(party,identities):
        moves=list(map(_move_key,row['moves']));value=ledger.get(identity)
        if identity not in seen:value={move:maximum_pp[move] for move in moves}
        if not isinstance(value,dict):raise ValueError(value or 'OWN_PP_LAST_ACTIVE_REQUEST_UNAVAILABLE')
        if set(value)!=set(moves):raise ValueError('OWN_PP_REQUEST_MOVESET_DIFFERS')
        result.append([value[move] for move in moves]+[0]*(4-len(moves)))
    return result


def sleep_support(count,rest,*,early_bird=False,skipped=0,pending=False,refunded=0,attempt_rates=None,attempt_refunds=None):
    """Public posterior in the frozen engine's PRE-attempt clock convention."""
    if type(count)is not int or type(skipped)is not int or type(refunded)is not int or min(count,skipped,refunded)<0 or skipped>count+refunded or type(rest)is not bool or type(early_bird)is not bool or type(pending)is not bool:
        raise ValueError('PUBLIC_SLEEP_EVIDENCE_INVALID')
    if pending:raise ValueError('PUBLIC_SLEEP_ATTEMPT_UNSETTLED')
    if attempt_rates is None:attempt_rates=(2 if early_bird else 1,)*(count+refunded)
    if len(attempt_rates)!=count+refunded or any(type(rate)is not int or rate not in (1,2) for rate in attempt_rates):
        raise ValueError('PUBLIC_SLEEP_ATTEMPT_HISTORY_INVALID')
    # B2 count has one unit removed per actual switch-in refund, even under
    # Early Bird. A refund adds ONE clock unit; it never refunds two.
    if attempt_refunds is None:
        if refunded:raise ValueError('PUBLIC_SLEEP_REFUND_CHRONOLOGY_UNAVAILABLE')
        attempt_refunds=(0,)*len(attempt_rates)
    if (len(attempt_refunds)!=len(attempt_rates) or any(type(n)is not int or not 0<=n<=refunded for n in attempt_refunds)
            or tuple(attempt_refunds)!=tuple(sorted(attempt_refunds))):
        raise ValueError('PUBLIC_SLEEP_REFUND_CHRONOLOGY_INVALID')
    # Every observed cant|slp proves the post-decrement clock was positive.
    # A later one-unit refund cannot erase that earlier survival constraint.
    lower=3 if rest else 2;upper=3 if rest else 5;spent=0
    for rate,prior_refunds in zip(attempt_rates,attempt_refunds):
        spent+=rate;lower=max(lower,spent-prior_refunds+1)
    used=spent-refunded;lo,hi=lower-used,upper-used
    if lo<1 or hi<lo:raise ValueError('PUBLIC_SLEEP_POSTERIOR_EMPTY')
    return tuple(range(lo,hi+1))
