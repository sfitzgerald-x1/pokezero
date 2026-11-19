"""Owner-request and public-history facts for the constructor's Transform paths.

No sampled donor supplies private move/PP/Hidden Power facts. Showdown f76228a,
sim/pokemon.ts getMoves/getSwitchRequestData name copied Hidden Power (including
power in the active Gen 3 request); transformInto preserves baseMoveSlots and
baseStoredStats, and clearVolatile restores them.
"""
import re

import numpy as np

from pokezero.dex import normalize_id


def copied_request(current,side_moves,vocabulary):
    from .phase_d_live_worlds import Refusal,_lookup,_own_move,_source_move
    rows=current.get('moves',())
    if (not 1<=len(rows)<=4 or
            [_source_move(r.get('id','')) for r in rows]!=list(map(_source_move,side_moves))):
        raise Refusal('OWN_TRANSFORM_COPIED_MOVES_UNAVAILABLE')
    ids=np.zeros(4,np.int32);pp=np.zeros(4,np.int32);disabled=np.zeros(4,bool)
    hp_type,hp_power=0,70
    for index,row in enumerate(rows):
        name=row.get('id','');resolved=None
        if _source_move(name)=='hiddenpower':
            match=re.fullmatch(r'hiddenpower([a-z]+)([0-9]+)',normalize_id(row.get('move','')))
            if match is None or not 30<=int(match[2])<=70:
                raise Refusal('OWN_TRANSFORM_HIDDEN_POWER_UNAVAILABLE')
            hp_type=_lookup(vocabulary['types'],match[1]);hp_power=int(match[2])
            resolved='hiddenpower'+match[1]+match[2]
        ids[index]=_own_move(vocabulary['moves'],name,resolved)
        if type(row.get('pp')) is not int or not 0<=row['pp']<=5:
            raise Refusal('OWN_TRANSFORM_COPIED_PP_UNAVAILABLE')
        pp[index]=row['pp'];disabled[index]=bool(row.get('disabled',False))
    return ids,pp,disabled,hp_type,hp_power


def history(lines):
    """Resolve encounter identities and the donor's public state AT Transform.

    An encounter's last borrow survives its departure, as B2's running does;
    its next arrival clears it. A later Trace announcement supersedes Transform.
    Side-private silent ability starts never enter the public ability ledger.
    """
    from .constructor_worlds import _seat_name
    from pokezero.belief import _confirmed_ability_from_event,_trace_copy_source
    from pokezero.showdown import _public_event_from_line
    names={};borrowed={};original={};running={};retyped={};copies={}
    member=lambda seat,name:names.get((seat,name),normalize_id(name))
    current=lambda who:running.get(who,original.get(who,''))
    def reveal(who,ability,authoritative=False):
        # B2 confirmations do not update an existing copy, even when that
        # owner's copied ability was unknown (notably private Gen 3 Pressure).
        if running.get(who) or (who in borrowed and who[0] in copies):return
        if authoritative or not original.get(who):original[who]=ability
    for line in lines:
        parts=line.split('|')
        if len(parts)<3 or parts[1]=='request':continue
        kind=parts[1];seat,name=_seat_name(parts[2])
        who=(seat,member(seat,name))
        if kind=='faint':copies.pop(seat,None)
        if len(parts)<4:continue
        if kind in ('switch','drag','replace'):
            species=normalize_id(parts[3].split(',')[0]);names[seat,name]=species
            who=(seat,species);borrowed.pop(who,None);running.pop(who,None)
            retyped[seat]=False;copies.pop(seat,None)
        tags=[x.strip().lower() for x in parts[4:]]
        traced,_=_trace_copy_source(line) if kind in ('-ability','ability') else (False,None)
        if traced:
            if seat not in copies:original[who]='trace'
            borrowed[who]='trace';running[who]=normalize_id(parts[3])
        # Use B2's holder attribution, including field procs, Trace's donor,
        # and ability: activation lines. The redaction helper intentionally
        # over-attributes and is not a current-ability ledger.
        if not (kind in ('-ability','ability') and tags==['[silent]']):
            ident,ability=_confirmed_ability_from_event(_public_event_from_line(line))
            if ident and ability:
                holder,holder_name=_seat_name(ident)
                if holder in ('p1','p2'):reveal((holder,member(holder,holder_name)),normalize_id(ability),kind in ('-ability','ability') and not traced)
        if seat not in ('p1','p2'):continue
        # Forme events can change the donor's live types just like typechange.
        # Save that uncertainty at the copy; a later weather change or departure
        # cannot change the types the transformer already copied.
        if kind=='-formechange' or (kind=='-start' and normalize_id(parts[3])=='typechange'):
            retyped[seat]=True
        if kind=='-transform':
            donor_seat,donor_name=_seat_name(parts[3]);donor=(donor_seat,member(donor_seat,donor_name))
            copies[seat]=dict(donor=donor[1],overridden=bool(retyped.get(donor_seat) or borrowed.get(donor)))
            copied=current(donor);borrowed[who]='transform';running[who]=copied
    public={who:current(who) for who in original.keys()|running.keys()}
    return dict(borrowed=borrowed,public=public,copies=copies,lines=tuple(lines))


def check_evidence(state,catalog,facts):
    from .phase_d_live_worlds import Refusal,_host
    for side in range(2):
        slot=int(_host(state.mons.active)[0,side]);marked=bool(_host(state.context.abilities.transformed)[0,side,slot])
        copy=facts['copies'].get('p'+str(side+1))
        if marked!=(copy is not None):raise Refusal('PUBLIC_TRANSFORM_EVIDENCE_DISAGREES')
        if not marked:continue
        donor=int(_host(state.context.abilities.transform_species)[0,side,slot])
        if normalize_id(catalog.species.names[donor])!=copy['donor']:
            raise Refusal('PUBLIC_TRANSFORM_EVIDENCE_DISAGREES')
        if copy['overridden']:raise Refusal('PUBLIC_TRANSFORM_DONOR_OVERRIDDEN')


def tracker_leaves(state,catalog,vocabulary,owner,facts,own,running):
    import jax.numpy as j
    from .phase_d_live_worlds import _host,_lookup,_remap
    marked=_host(state.context.abilities.transformed).astype(bool)
    borrowed=np.zeros((1,2,6),bool);view=np.zeros((1,2,6),np.int32)
    for side in range(2):
        for slot,sid in enumerate(_host(state.mons.species)[0,side]):
            who=('p'+str(side+1),normalize_id(catalog.species.names[int(sid)]))
            borrowed[0,side,slot]=running[0,side,slot]>0 and (marked[0,side,slot] or facts['borrowed'].get(who)=='transform')
            if marked[0,side,slot]:
                view[0,side,slot]=running[0,side,slot] if side==owner else _lookup(vocabulary['abilities'],facts['public'].get(who,''))
    used=np.zeros((1,2,4),np.int32);peer=1-owner
    if marked[0,peer].any():
        from .constructor_worlds import line_facts
        from .phase_d_live_worlds import _source_move
        fact=line_facts(facts['lines'],owner).transform[peer]
        d=donor_slot(own['species'],_lookup(vocabulary['species'],fact.donor))
        moves=own['transform_moves'] if bool(own['transformed']) and d==int(own['active']) else own['moves'][d]
        charges=dict(fact.copied_uses)
        used[0,peer]=[charges.get(_source_move(vocabulary['moves'][int(mid)]),0) if mid else 0 for mid in moves]
    return dict(transformed=j.asarray(marked),
        transform_species=j.asarray(_remap(state.context.abilities.transform_species,catalog.species.names,vocabulary['species'])),
        transform_borrowed=j.asarray(borrowed),transform_own_view=j.asarray(view),copied_pp_used=j.asarray(used))


def donor_slot(species,donor):
    matches=np.flatnonzero(np.asarray(species)==int(donor))
    if len(matches)!=1:raise ValueError('PUBLIC_TRANSFORM_DONOR_UNAVAILABLE')
    return int(matches[0])


def public_slots(public,vocabulary):
    marked=getattr(public,'transformed',None)
    if marked is None:return {}
    return {f'p{side+1}':normalize_id(vocabulary['species'][int(public.transform_species[0,side,int(public.active_slot[0,side])])])
        for side in range(2) if bool(marked[0,side,int(public.active_slot[0,side])])}


def donor_values(arrays,side,slot):
    """The sampled original donor, including when it has subsequently transformed."""
    transformed=bool(arrays['transformed'][0,side]) and slot==int(arrays['active'][0,side])
    return dict(stats=arrays['transform_stats'][0,side] if transformed else arrays['stats'][0,side,slot,1:],
        moves=arrays['transform_moves'][0,side] if transformed else arrays['moves'][0,side,slot],
        hidden=arrays['transform_hidden_power'][0,side] if transformed else
            (arrays['hidden_power_type'][0,side,slot],arrays['hidden_power_power'][0,side,slot]),
        ability=arrays['base_ability'][0,side,slot],types=arrays['base_types'][0,side,slot])


def check_donor(moves,hp_type,hp_power,donor_moves,donor_type,donor_power,names):
    """The request's copied set and HP are constraints on a sampled donor."""
    if sorted(int(m) for m in moves if m)!=sorted(int(m) for m in donor_moves if m):
        raise ValueError('PUBLIC_TRANSFORM_DONOR_DISAGREES')
    if any(normalize_id(names[int(m)]).startswith('hiddenpower') for m in moves if m):
        if (int(hp_type),int(hp_power))!=(int(donor_type),int(donor_power)):
            raise ValueError('PUBLIC_TRANSFORM_DONOR_DISAGREES')


def materialize_copy(root,tables,arrays,owner,foe,active,assignment):
    """Phase D's device route, after from_b2 checked donor history.

    The retained-device route has no protocol history and keeps its conservative
    donor refusal. This route can admit an unmodified Color Change donor.
    """
    own=root.own
    if bool(own['transformed']):
        at=int(own['active']);d=donor_slot(arrays['species'][0,foe],own['transform_species'])
        donor=donor_values(arrays,foe,d)
        check_donor(own['moves'][at],own['hidden_power_type'][at],own['hidden_power_power'][at],
            donor['moves'],*donor['hidden'],tables.source.vocabulary['moves'])
        arrays['stats'][0,owner,at,1:]=donor['stats']
        arrays['ability'][0,owner,at]=donor['ability']
        arrays['types'][0,owner,at]=donor['types']
    public=next((p for p,k in assignment.items() if k==active),None)
    if public is None or not bool(np.asarray(root.public.transformed)[0,foe,public]):
        restore_truant(arrays,root.public,tables.source.vocabulary)
        return
    donor=int(np.asarray(root.public.transform_species)[0,foe,public]);d=donor_slot(arrays['species'][0,owner],donor)
    at=(0,foe,active);mine=bool(own['transformed']) and d==int(own['active'])
    stats=own['transform_stats'] if mine else arrays['stats'][0,owner,d,1:]
    moves=own['transform_moves'] if mine else arrays['moves'][0,owner,d]
    hidden=own['transform_hidden_power'] if mine else (arrays['hidden_power_type'][0,owner,d],arrays['hidden_power_power'][0,owner,d])
    arrays['transform_stats'][0,foe]=arrays['stats'][at][1:]
    arrays['transform_moves'][0,foe]=arrays['moves'][at]
    arrays['transform_pp'][0,foe]=arrays['pp'][at]
    arrays['transform_hidden_power'][0,foe]=(arrays['hidden_power_type'][at],arrays['hidden_power_power'][at])
    arrays['transformed'][0,foe]=True;arrays['transform_species'][0,foe]=donor
    arrays['stats'][at][1:]=stats;arrays['moves'][at]=moves
    used=np.asarray(root.public.copied_pp_used)[0,foe]
    arrays['pp'][at]=[max(0,5-int(n)) if m else 0 for m,n in zip(moves,used)]
    arrays['disabled'][at]=False;arrays['ability'][at]=arrays['base_ability'][0,owner,d]
    arrays['types'][at]=arrays['base_types'][0,owner,d]
    arrays['hidden_power_type'][at],arrays['hidden_power_power'][at]=hidden
    mid=int(np.asarray(root.public.last_moves)[0,foe,public])
    if mid>0:
        from .phase_d_live_worlds import _source_move
        names=tables.source.vocabulary['moves'];key=_source_move(names[mid])
        matches=[k for k,m in enumerate(moves) if m and _source_move(names[int(m)])==key]
        index=-1 if key=='transform' else matches[0] if len(matches)==1 else None
        if index is None:raise ValueError('PUBLIC_TRANSFORM_LAST_MOVE_UNAVAILABLE')
        if index>=0:mid=int(moves[index])
        arrays['last_move'][at]=mid;arrays['last_action_kind'][0,foe]=2;arrays['last_action_index'][0,foe]=index
        if arrays['item'][at]==tables.namespaces['items'].get('choiceband',-1):arrays['choice_move'][at]=mid if matches else 0
    restore_truant(arrays,root.public,tables.source.vocabulary)


def restore_truant(arrays,public,vocabulary):
    """A copy keeps the transformer's own Truant clock; never use the donor's."""
    ids=[i for i,name in enumerate(vocabulary['abilities']) if normalize_id(name)=='truant']
    if not ids:return
    for side in range(2):
        at=int(arrays['active'][0,side])
        if not arrays['transformed'][0,side] or int(arrays['ability'][0,side,at])!=ids[0]:continue
        port=public.public_port
        if port is None or not bool(np.asarray(port.known)[0,side]):
            raise ValueError('PUBLIC_TRANSFORM_TRUANT_PHASE_UNAVAILABLE')
        arrays['truant_loaf'][0,side,at]=bool(np.asarray(port.loaf)[0,side])


def retained_copy(state,public,request,lines,metadata,catalog,owner):
    """The retained-history route uses the same request and line rules as B2's protocol route."""
    import jax.numpy as j
    from types import SimpleNamespace
    from .phase_d_constructor_rules import own_pp_carry
    from .phase_d_live_worlds import _source_move,_own_move,_lookup
    arrays={name:np.array(value,copy=True) for name,value in state._asdict().items()};v=metadata['vocabulary']
    if arrays['transformed'][0,owner]:
        at=int(arrays['active'][0,owner]);species=normalize_id(v['species'][int(arrays['species'][0,owner,at])])
        matches=[i for i,row in enumerate(request['side']['pokemon']) if normalize_id(row['details'].split(',')[0])==species]
        if len(matches)!=1:raise ValueError('OWN_TRANSFORM_REQUEST_IDENTITY_UNAVAILABLE')
        request_slot=matches[0];row=request['side']['pokemon'][request_slot]
        maximum={_source_move(n):int(m['pp'])*8//5 for n,m in zip(v['moves'],metadata['moves']) if m and m.get('pp')}
        _,saved=own_pp_carry(lines,request,maximum,transform_base=True);ledger=saved[request_slot]
        if ledger is None:raise ValueError('OWN_TRANSFORM_EVIDENCE_DISAGREES')
        typ,power=arrays['transform_hidden_power'][0,owner]
        resolved='hiddenpower'+normalize_id(v['types'][int(typ)])+str(int(power)) if typ else None
        ids=[_own_move(v['moves'],name,resolved) for name,_ in ledger]+[0]*(4-len(ledger))
        if sorted(ids)!=sorted(arrays['transform_moves'][0,owner]):raise ValueError('OWN_TRANSFORM_BASE_MOVES_DISAGREE')
        arrays['transform_moves'][0,owner]=ids;arrays['transform_pp'][0,owner]=[pp for _,pp in ledger]+[0]*(4-len(ledger))
        ids,pp,disabled,typ,power=copied_request(request['active'][0],row['moves'],v)
        d=donor_slot(arrays['species'][0,1-owner],arrays['transform_species'][0,owner]);donor=donor_values(arrays,1-owner,d)
        check_donor(ids,typ,power,donor['moves'],*donor['hidden'],v['moves'])
        arrays['moves'][0,owner,at]=ids;arrays['pp'][0,owner,at]=pp;arrays['disabled'][0,owner,at]=disabled
        arrays['hidden_power_type'][0,owner,at]=typ;arrays['hidden_power_power'][0,owner,at]=power
        # The world mapper starts from the copied types and folds any later
        # observed type change. Resampling the donor must preserve that result.
        arrays['stats'][0,owner,at,1:]=donor['stats']
        arrays['ability'][0,owner,at]=donor['ability']
    for side in range(2):
        if not arrays['transformed'][0,side]:continue
        at=int(arrays['active'][0,side]);mid=int(np.asarray(public.context.moves.last_id)[0,side])
        if int(np.asarray(public.context.moves.last_kind)[0,side])!=2:continue
        move=_lookup(v['moves'],catalog.moves.names[mid]);key=_source_move(v['moves'][move])
        matches=[k for k,m in enumerate(arrays['moves'][0,side,at]) if m and _source_move(v['moves'][int(m)])==key]
        index=-1 if key=='transform' else matches[0] if len(matches)==1 else None
        if index is None:raise ValueError('PUBLIC_TRANSFORM_LAST_MOVE_UNAVAILABLE')
        if index>=0:move=int(arrays['moves'][0,side,at,index])
        arrays['last_move'][0,side,at]=move;arrays['last_action_kind'][0,side]=2;arrays['last_action_index'][0,side]=index
        if normalize_id(v['items'][int(arrays['item'][0,side,at])])=='choiceband':arrays['choice_move'][0,side,at]=move if matches else 0
    port=SimpleNamespace(known=public.context.active.truant_known,loaf=public.context.active.truant_loaf)
    restore_truant(arrays,SimpleNamespace(public_port=port),v)
    return state._replace(**{name:j.asarray(value) for name,value in arrays.items()})
