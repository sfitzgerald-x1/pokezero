"""Raw transaction envelope for ordered source-summary consumers.

The first integrated families are arrival, move, turn and public abilities.
Unsupported source callback families refuse at the lexical boundary until their
intermediate evidence producers are installed. This is not replay admission.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from pokezero.belief import _trace_copy_source, _confirmed_ability_from_event
from pokezero.showdown import _public_event_from_line
from pokezero.dex import normalize_id
from . import public_belief_refresh as refresh, public_belief_snapshot as snapshot
from . import public_tendency as tendency, public_mon_state as mon

NOOP, ARRIVAL, MOVE, TURN, ABILITY, TRACE, TRANSFORM=range(1,8)
STATUS,CANT,CURE,TEAM,FAINT,HP,CANT_OTHER=range(8,15)
ITEM=15
UPKEEP=16
LAST_FAMILY=UPKEEP
SCHEMA='pokezero.public_refresh_stream.v7'
# HP keeps its CONDITION flags in bits0..1. An absorb heal additionally binds
# the literal ability proc and healed address to its OUTCOME0 frame closure.
HP_ABSORB=4


def hp_absorb(word):
    return (word&HP_ABSORB)!=0


def hp_absorb_address(word):
    return (word>>6)&1,(word>>3)&7,word>>7
FAMILIES=('noop','arrival','move','turn','ability','trace','transform','status','sleep-cant','cure','team','faint','hp','other-cant','item','upkeep')


class StreamState(NamedTuple):
    controller: refresh.RefreshState
    family: jax.Array
    main_seen: jax.Array
    main_proof: jax.Array          # side, tendency kind, action/species, target, flags, turn
    raw_meta: jax.Array
    proc_seen: jax.Array
    common_seen: jax.Array
    common_proof: jax.Array
    switch_seen: jax.Array
    item_expected: jax.Array
    item_seen: jax.Array
    item_eat: jax.Array
    item_mode: jax.Array
    item_target: jax.Array
    nonproc_seen: jax.Array
    nonproc_word: jax.Array
    mudshot_seen: jax.Array
    mudshot_effect_seen: jax.Array
    mudshot_word: jax.Array
    mudshot_target: jax.Array


def validate_source(source,*,pending_switch=False):
    for variants in source.variants:
        abilities={normalize_id(v.ability or '') for v in variants}
        if 'intimidate' in abilities and len(abilities)>1 and not pending_switch:
            raise ValueError('ordered refresh requires the pending Intimidate source consumer for this source')


def family(line,*,lifecycle=False,nonproc=False,mudshot=False):
    """Lexical source family, never a CPU summary or callback count."""
    parts=line.split('|');kind=parts[1] if len(parts)>1 else ''
    from . import public_annotations
    if public_annotations.classify(line) is not None:return NOOP
    if line=='|upkeep' and nonproc:return UPKEEP
    if lifecycle:
        if kind in ('-item','-enditem'):return ITEM
        if kind=='-status':return STATUS
        if kind=='-curestatus':return CURE
        if kind=='-cureteam':return TEAM
        if kind=='faint':return FAINT
        if kind in ('-damage','-heal','-sethp'):return HP
        if kind=='cant':
            if len(parts)<4:raise ValueError('source cant requires an explicit reason')
            if '|slp' in line and parts[3]!='slp':raise ValueError('ambiguous source sleep-attempt spelling')
            return CANT if parts[3]=='slp' else CANT_OTHER
    if kind in ('switch','drag','replace'):return ARRIVAL
    if kind=='move':
        if len(parts)<4 or (normalize_id(parts[3])=='mudshot' and not mudshot):
            raise ValueError('pending MudShot evidence requires its ordered refresh consumer')
        return MOVE
    if kind=='turn':return TURN
    if kind in ('ability','-ability'):return TRACE if _trace_copy_source(line)[0] else ABILITY
    if kind=='-transform':return TRANSFORM
    if line=='|' or kind in ('t:','-weather','-sidestart','-sideend','-boost','-unboost','-setboost',
            '-clearboost','-clearpositiveboost','-clearnegativeboost','-clearallboost','-copyboost',
            '-start','-end','-activate','-formechange','-immune','-fail','-block','-miss','-crit','-supereffective',
            '-resisted','-anim','-mustrecharge','-singleturn','-singlemove','-prepare'):
        return NOOP
    raise ValueError('raw source summary family still requires its ordered evidence consumer: '+kind)


def initialize(beliefs,source_tables,tables):
    batch=beliefs.species.shape[0]
    return StreamState(refresh.initialize(beliefs,source_tables,tables),jnp.zeros(batch,jnp.int32),
        jnp.zeros(batch,jnp.int32),jnp.zeros((batch,6),jnp.int32),
        jnp.zeros((batch,4),jnp.int32),jnp.zeros(batch,jnp.int32),jnp.zeros(batch,jnp.int32),jnp.zeros((batch,10),jnp.int32),jnp.zeros(batch,jnp.int32),
        jnp.zeros(batch,jnp.int32),jnp.zeros(batch,jnp.int32),jnp.zeros(batch,bool),
        jnp.zeros(batch,jnp.int32),jnp.broadcast_to(jnp.array([-1,-1,0],jnp.int32),(batch,3)),jnp.zeros(batch,jnp.int32),jnp.zeros(batch,jnp.int32),
        jnp.zeros(batch,jnp.int32),jnp.zeros(batch,jnp.int32),jnp.zeros(batch,jnp.int32),
        jnp.broadcast_to(jnp.array([-1,-1,0],jnp.int32),(batch,3)))


def item_word_ok(word,*,framed,code=None):
    if not framed:return (word>=0)&(word<=2)
    kind=word&3;count=(word>>2)&3;eat=(word&16)!=0
    ok=(word>=0)&(word<2**18)&(kind<=2)&(count<=2)&(((word&16)==0)|((count==1)&(kind==1)))
    if code is not None:
        ok &= (kind==1)==(code==ITEM)
        ok &= (kind!=2)|((code==NOOP)&(count==0))
        ok &= (count!=2)|(code==ITEM)
    return ok


def metadata(code,line,events,lifecycle_tables,*,addresses=None,clause=False):
    """Lexical proofs from decoded public facts, never Python summaries."""
    if lifecycle_tables is None:return (0,0,0,0)
    from .public_inputs import Kind
    from .public_mon_protocol import STATUS as statuses
    from . import public_sleep as sleep
    parts=line.split('|')
    confirmed=_confirmed_ability_from_event(_public_event_from_line(line))
    if code==NOOP and parts[1]=='-activate' and all(confirmed) and normalize_id(confirmed[1])=='shedskin':
        event=next(e for e in events if e.kind==Kind.ABILITY)
        return (1,event.target_side,event.target_slot,event.payload[0])
    if code in (STATUS,CURE,HP):
        operation={STATUS:sleep.STATUS,CURE:sleep.CURE,HP:sleep.CONDITION}[code]
        control=next(e for e in events if e.kind==Kind.SLEEP and e.payload[1]==operation)
        if code==STATUS:return (2*control.payload[3]+4*int('[from] ability:' in line),control.payload[2],0,0)
        if code==CURE:
            bench=control.payload[2]
            # An exact benched species lookup upserts even an existing row.
            # The source's cosmetic base-forme redirect returns it directly.
            if bench and addresses is None:raise ValueError('bench cure requires its public species lookup proof')
            named=parts[2].split(':',1)[1].strip()
            exact=bench and any(address.side==control.target_side and address.slot==control.target_slot and
                normalize_id(species)==normalize_id(named) for address,species in addresses.public_species.items())
            parser_match=0
            if clause:
                from pokezero.showdown import _species_from_ident, _normalize_identifier, _ident_has_position
                registered={_normalize_identifier(_species_from_ident(ident)) for ident,address in addresses.known_idents.items()
                    if address.side==control.target_side and address.slot==control.target_slot and _ident_has_position(ident)}
                if len(registered)!=1:raise ValueError('clause cure requires one registered public nickname key')
                parser_match=int(_normalize_identifier(_species_from_ident(parts[2])) in registered)
            return (int(exact),bench,statuses[parts[3]],parser_match)
        common=next(e for e in events if e.kind==Kind.CONDITION)
        flags=control.payload[3]
        if parts[1]=='-heal' and all(confirmed) and normalize_id(confirmed[1]) in ('waterabsorb','voltabsorb'):
            proc=next(e for e in events if e.kind==Kind.ABILITY)
            if (confirmed[0]!=parts[2] or flags!=0 or proc.payload[2]!=0
                    or (proc.target_side,proc.target_slot,proc.payload[0])!=(control.target_side,control.target_slot,control.payload[0])
                    or not 0<proc.payload[1]<2**24):
                raise ValueError('absorb HP closure requires its exact living healed ability proc')
            flags=HP_ABSORB|(proc.target_slot<<3)|(proc.target_side<<6)|(proc.payload[1]<<7)
        return (flags,*common.payload[1:4])
    return (0,0,0,0)


def metadata_ok(code,meta,*,lifecycle,clause=False):
    flags,a,b,c=(meta[:,i] for i in range(4))
    zero=jnp.all(meta==0,axis=-1)
    if not lifecycle:return (code>=NOOP)&(code<=TRANSFORM)&zero
    shed=(flags==1)&(a>=0)&(a<2)&(b>=0)&(b<6)&(c>0)
    status=(flags>=0)&(flags<=6)&((flags&1)==0)&(a>=mon.SLEEP)&(a<=mon.TOXIC)&(b==0)&(c==0)
    status &= ((flags&2)==0)|(a==mon.SLEEP)
    cure=((flags==0)|(flags==1))&((a==0)|(a==1))&((flags==0)|(a==1))&(b>=mon.SLEEP)&(b<=mon.TOXIC)&jnp.where(clause,(c==0)|(c==1),c==0)
    absorb=hp_absorb(flags);_,slot,aid=hp_absorb_address(flags)
    hp_flags=jnp.where(absorb,flags&3,flags)
    hp=((hp_flags==0)|(hp_flags==2)|(hp_flags==3))&(a>=0)&(b>=0)&(a<=b)&(c>=mon.NONE)&(c<=mon.TOXIC)
    hp &= ~absorb|((flags>=0)&(hp_flags==0)&(slot<6)&(aid>0))
    hp &= (b>0)|((a==0)&(c==mon.NONE))
    hp &= (b==0)==(hp_flags!=0)
    return (code>=NOOP)&(code<=LAST_FAMILY)&jnp.where(code==NOOP,zero|shed,
        jnp.where(code==STATUS,status,jnp.where(code==CURE,cure,jnp.where(code==HP,hp,zero))))


def metadata_words_ok(code,meta):
    flags,a,b,c=meta;zero=not any(meta)
    if not NOOP<=code<=LAST_FAMILY:return False
    if code==NOOP:return zero or flags==1 and 0<=a<2 and 0<=b<6 and c>0
    if code==STATUS:return flags in (0,2,4,6) and mon.SLEEP<=a<=mon.TOXIC and b==c==0 and (not flags&2 or a==mon.SLEEP)
    if code==CURE:return flags in (0,1) and a in (0,1) and (flags==0 or a==1) and mon.SLEEP<=b<=mon.TOXIC and c in (0,1)
    if code==HP:
        absorb=hp_absorb(flags);_,slot,aid=hp_absorb_address(flags);low=flags&3 if absorb else flags
        return (low in (0,2,3) and (not absorb or flags>=0 and low==0 and slot<6 and aid>0)
            and 0<=a<=b and mon.NONE<=c<=mon.TOXIC and (b>0 or a==c==0) and ((b==0)==(low!=0)))
    return zero


def common(state,row,*,status_kind,condition_kind,valid):
    code=state.family;data=row[:,8:]
    hp=code==HP;status=(code==STATUS)|(code==CURE)
    ok=(state.common_seen==0)&jnp.where(hp,row[:,0]==condition_kind,status&(row[:,0]==status_kind))
    expected=jnp.where(code==CURE,mon.NONE,state.raw_meta[:,1])
    ok &= jnp.where(hp,jnp.all(data[:,1:4]==state.raw_meta[:,1:4],axis=-1),data[:,1]==expected)
    marked=hp&hp_absorb(state.raw_meta[:,0]);side,slot,_=hp_absorb_address(state.raw_meta[:,0])
    ok &= ~marked|((state.proc_seen==1)&(row[:,6]==side)&(row[:,7]==slot))
    use=valid&ok
    proof=jnp.concatenate((row[:,6:8],data),axis=-1)
    return state._replace(common_seen=jnp.where(use,1,state.common_seen),
        common_proof=jnp.where(use[:,None],proof,state.common_proof)),ok|~valid


def common_matches(state,row):
    return (state.common_seen==1)&jnp.all(state.common_proof[:,:3]==row[:,6:9],axis=-1)


def validate(state,beliefs,source_tables,tables,move_tables,*,lifecycle=False,clause=False,mons=None):
    batch=beliefs.species.shape[0]
    refresh._tables(tables,source_tables,move_tables);refresh._state_shape(state.controller,batch)
    for value in (state.family,state.main_seen,state.proc_seen,state.common_seen,state.switch_seen,state.item_expected,state.item_seen,state.item_mode,state.nonproc_seen,state.nonproc_word,state.mudshot_seen,state.mudshot_effect_seen,state.mudshot_word):
        if value.shape!=(batch,) or value.dtype!=jnp.int32:
            raise ValueError('source refresh frame requires B int32 state words')
    if state.item_eat.shape!=(batch,) or state.item_eat.dtype!=jnp.bool_:
        raise ValueError('source item eat proof requires B boolean flags')
    if state.item_target.shape!=(batch,3) or state.item_target.dtype!=jnp.int32:
        raise ValueError('source item target requires B by3 int32 words')
    if state.main_proof.shape!=(batch,6) or state.main_proof.dtype!=jnp.int32:
        raise ValueError('source refresh main proof requires B by6 int32 words')
    for name,width in (('raw_meta',4),('common_proof',10)):
        value=getattr(state,name)
        if value.shape!=(batch,width) or value.dtype!=jnp.int32:raise ValueError('source frame shape/dtype: '+name)
    _,visible_ok=snapshot.read(state.controller.visible,beliefs,source_tables)
    ok=jnp.all(state.controller.binding==tables.binding,axis=-1)
    ok &= jnp.all(tables.source_binding==source_tables.binding)&jnp.all(tables.move_binding==move_tables.binding)
    ok &= (tables.choice_predicate>=0)&(tables.choice_predicate<source_tables.predicates.shape[0])
    selected=state.controller.first_selected
    ok &= jnp.all((selected>=0)&(selected<tables.sleep_usable.shape[0]),axis=(1,2))
    ok &= (state.family>=0)&(state.family<=LAST_FAMILY)&(state.main_seen>=0)&(state.main_seen<=1)
    ok &= (state.proc_seen>=0)&(state.proc_seen<=1)&(state.common_seen>=0)&(state.common_seen<=1)
    ok &= (state.switch_seen>=0)&(state.switch_seen<=1)&((state.family!=0)|(state.switch_seen==0))
    ok &= (state.nonproc_seen>=0)&(state.nonproc_seen<=1)&((state.family!=0)|(state.nonproc_seen==0))
    ok &= (state.nonproc_word>=0)&(state.nonproc_word<256)&((state.family!=0)|(state.nonproc_word==0))
    if state.mudshot_target.shape!=(batch,3) or state.mudshot_target.dtype!=jnp.int32:raise ValueError('Mud Shot target requires B by3 int32 words')
    ok &= (state.mudshot_seen>=0)&(state.mudshot_seen<=1)&(state.mudshot_effect_seen>=0)&(state.mudshot_effect_seen<=1)
    code=state.mudshot_word&7
    ok &= (state.mudshot_word>=0)&(state.mudshot_word<32)&((code==0)|((code>=2)&(code<=5)))
    ok &= ((state.mudshot_word&8)==0)|(code==5)
    ok &= ((state.mudshot_word&16)==0)|(code==2)
    ok &= (state.family!=0)|((state.mudshot_seen==0)&(state.mudshot_effect_seen==0)&(state.mudshot_word==0))
    ok &= (state.mudshot_seen!=0)|jnp.all(state.mudshot_target==jnp.array([-1,-1,0],jnp.int32),axis=-1)
    address=state.mudshot_target;none=jnp.all(address==jnp.array([-1,-1,0],jnp.int32),axis=-1)
    target=(address[:,0]>=0)&(address[:,0]<2)&(address[:,1]>=0)&(address[:,1]<6)&(address[:,2]>0)
    if mons is not None:
        rows=jnp.arange(batch);seat=jnp.clip(address[:,0],0,1);slot=jnp.clip(address[:,1],0,5)
        target &= mons.encounter[rows,seat,slot]==address[:,2]
    targeted=(code==2)|(code==3)|(code==4)
    queue=(state.mudshot_word&8)!=0
    ok &= (state.mudshot_seen==0)|jnp.where(targeted,target,jnp.where(queue,target|none,none))
    ok &= jnp.where(state.family==0,jnp.all(state.raw_meta==0,axis=-1),metadata_ok(state.family,state.raw_meta,lifecycle=lifecycle,clause=clause))
    ok &= (state.common_seen!=0)|jnp.all(state.common_proof==0,axis=-1)
    ok &= (state.family!=0)|((state.proc_seen==0)&(state.common_seen==0))
    ok &= (state.item_expected>=0)&(state.item_expected<=2)&(state.item_seen>=0)&(state.item_seen<=state.item_expected)
    item_word=state.item_mode|(state.item_expected<<2)|(state.item_eat.astype(jnp.int32)<<4)
    ok &= (state.item_mode>=0)&(state.item_mode<=2)
    ok &= jnp.where(state.family!=0,item_word_ok(item_word,framed=True,code=state.family),item_word==0)
    address=state.item_target;seen=state.item_seen>0
    target_ok=(address[:,0]>=0)&(address[:,0]<2)&(address[:,1]>=0)&(address[:,1]<6)&(address[:,2]>0)
    if mons is not None:
        rows=jnp.arange(batch);side=jnp.clip(address[:,0],0,1);slot=jnp.clip(address[:,1],0,5)
        target_ok &= mons.encounter[rows,side,slot]==address[:,2]
    ok &= jnp.where(seen,target_ok,jnp.all(address==jnp.array([-1,-1,0],jnp.int32),axis=-1))
    ok &= ~state.item_eat|(state.item_expected==1)
    ok &= (state.family!=ITEM)|(state.item_expected>0)
    ok &= (state.family!=0)|((state.item_expected==0)&(state.item_seen==0)&~state.item_eat)
    ok &= (state.family!=0)|(state.main_seen==0)
    ok &= (state.main_seen!=0)|jnp.all(state.main_proof==0,axis=-1)
    return ok&jnp.all(visible_ok,axis=(1,2))


def open_frame(state,code,meta,valid,*,lifecycle=False,clause=False,item_word=0):
    ok=(state.family==0)&(state.main_seen==0)&(code>=NOOP)&(code<=LAST_FAMILY)
    ok &= (state.proc_seen==0)&(state.common_seen==0)&(state.switch_seen==0)&metadata_ok(code,meta,lifecycle=lifecycle,clause=clause)
    ok &= state.nonproc_seen==0
    ok &= state.nonproc_word==0
    ok &= (state.mudshot_seen==0)&(state.mudshot_effect_seen==0)&(state.mudshot_word==0)
    count=(item_word>>2)&3;eat=(item_word&16)!=0
    ok &= item_word_ok(item_word,framed=True,code=code)&((code!=ITEM)|(count>0))
    ok &= (state.item_expected==0)&(state.item_seen==0)&~state.item_eat
    use=valid&ok
    return state._replace(family=jnp.where(use,code,state.family),raw_meta=jnp.where(use[:,None],meta,state.raw_meta),
        item_expected=jnp.where(use,count,state.item_expected),item_eat=jnp.where(use,eat,state.item_eat),item_mode=jnp.where(use,item_word&3,state.item_mode),
        nonproc_word=jnp.where(use,(item_word>>5)&255,state.nonproc_word),
        mudshot_word=jnp.where(use,item_word>>13,state.mudshot_word)),ok|~valid


def main(state,code,proof,valid):
    ok=(state.family==code)&(state.main_seen==0)
    return state._replace(main_seen=jnp.where(valid&ok,1,state.main_seen),
        main_proof=jnp.where((valid&ok)[:,None],proof,state.main_proof)),ok|~valid


def checkpoint(state,beliefs,source_tables,mask):
    cache,beliefs,ok=snapshot.refresh(state.controller.visible,beliefs,source_tables,source_refresh=mask)
    return state._replace(controller=state.controller._replace(visible=cache)),beliefs,jnp.all(ok,axis=(1,2))


def close_frame(state,beliefs,source_tables,control,control_side,valid,*,pending_switch=False,nonproc=False,mudshot=False):
    _,basis_ok=snapshot.read(state.controller.visible,beliefs,source_tables)
    ok=(state.family>=NOOP)&(state.family<=LAST_FAMILY)
    ok &= state.switch_seen==int(pending_switch)
    ok &= state.nonproc_seen==int(nonproc)
    ok &= state.mudshot_seen==int(mudshot)
    expected_mudshot_effect=((state.mudshot_word&7)!=0)&mudshot
    ok &= state.mudshot_effect_seen==expected_mudshot_effect.astype(jnp.int32)
    ok &= state.item_seen==state.item_expected
    ok &= state.main_seen==jnp.where((state.family==NOOP)|(state.family==CANT_OTHER)|(state.family==ITEM),0,1)
    control_kind=control[:,0]
    arrival=(control_kind>=tendency.SWITCH)&(control_kind<=tendency.REPLACE)
    callback_main=arrival|(control_kind==tendency.MOVE)|(control_kind==tendency.TURN)
    unsupported=(control_kind==tendency.FAINT)|(control_kind==tendency.CANT)|(control_kind==tendency.UPKEEP)
    expected_common=(state.family==STATUS)|(state.family==CURE)|(state.family==HP)
    ok &= state.common_seen==expected_common.astype(jnp.int32)
    absorb_hp=(state.family==HP)&hp_absorb(state.raw_meta[:,0])
    ok &= state.proc_seen==(((state.family==NOOP)&((state.raw_meta[:,0]&1)!=0))|absorb_hp).astype(jnp.int32)
    expected=jnp.where(state.family==ARRIVAL,arrival,jnp.where(state.family==MOVE,control_kind==tendency.MOVE,
        jnp.where(state.family==TURN,control_kind==tendency.TURN,
        jnp.where(state.family==NOOP,~callback_main&~unsupported,
        jnp.where(state.family==UPKEEP,control_kind==tendency.UPKEEP,jnp.where(state.family==FAINT,control_kind==tendency.FAINT,
        jnp.where((state.family==CANT)|(state.family==CANT_OTHER),control_kind==tendency.CANT,
        jnp.where(absorb_hp,(control_kind==tendency.OUTCOME)&(control[:,7]==0),control_kind==tendency.OTHER))))))))
    ok &= expected
    proof=jnp.stack((control_side,control_kind,jnp.where(arrival,control[:,1],control[:,2]),
        control[:,3],control[:,5],control[:,4]),axis=-1)
    ok &= ((state.family==NOOP)|(state.family==CANT_OTHER)|(state.family==ITEM))|jnp.all(proof==state.main_proof,axis=-1)
    ok &= jnp.all(basis_ok,axis=(1,2))
    use=valid&ok
    return state._replace(family=jnp.where(use,0,state.family),main_seen=jnp.where(use,0,state.main_seen),
        main_proof=jnp.where(use[:,None],0,state.main_proof),raw_meta=jnp.where(use[:,None],0,state.raw_meta),
        proc_seen=jnp.where(use,0,state.proc_seen),common_seen=jnp.where(use,0,state.common_seen),
        common_proof=jnp.where(use[:,None],0,state.common_proof),switch_seen=jnp.where(use,0,state.switch_seen),
        item_expected=jnp.where(use,0,state.item_expected),item_seen=jnp.where(use,0,state.item_seen),
        item_eat=jnp.where(use,False,state.item_eat),item_mode=jnp.where(use,0,state.item_mode),
        item_target=jnp.where(use[:,None],jnp.array([-1,-1,0],jnp.int32),state.item_target),
        nonproc_seen=jnp.where(use,0,state.nonproc_seen),nonproc_word=jnp.where(use,0,state.nonproc_word),
        mudshot_seen=jnp.where(use,0,state.mudshot_seen),mudshot_effect_seen=jnp.where(use,0,state.mudshot_effect_seen),
        mudshot_word=jnp.where(use,0,state.mudshot_word),
        mudshot_target=jnp.where(use[:,None],jnp.array([-1,-1,0],jnp.int32),state.mudshot_target)),ok|~valid


def commit(ok,before,after):
    """Commit batched state while retaining the cache's global namespace words."""
    choose=lambda a,b:refresh._choose(ok,a,b)
    controller=before.controller._replace(
        binding=choose(before.controller.binding,after.controller.binding),
        visible=refresh._cache_commit(ok,before.controller.visible,after.controller.visible),
        first_selected=choose(before.controller.first_selected,after.controller.first_selected),
        choice_excluded=choose(before.controller.choice_excluded,after.controller.choice_excluded))
    return after._replace(controller=controller,**{name:choose(getattr(before,name),getattr(after,name)) for name in before._fields if name!='controller'})
