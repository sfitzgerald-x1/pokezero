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
from pokezero.engine_world import battle_spec_from_payload,EngineWorldUnsupported,unpack_team,_engine_species_id
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
        # encore is not refused: its count is drawn from the public posterior (_encore_facts, _write_encores); trapped,
        # solarbeam and a Mean Look trap still are.
        fail=bool(side.get('meanlookTrap') or set(side['volatiles'])&{'trapped','solarbeam'})
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


def _defer_yawn(payload,lines):
    """The payload without its public Yawn volatiles, and (side, count) for each side whose active holds one.

    Showdown's Yawn (data/moves.ts: volatileStatus yawn, duration 2) starts at |-start|<target>|move: Yawn in a move phase;
    each residual decrements the duration (sim/battle.ts residualEvent, the |upkeep| that ends it) and at 0 the holder
    falls asleep. The count is therefore 2 before that turn's residual (a replacement boundary) and 1 after it: public,
    with no hidden roll. The engine holds it as yawn_counter (fidelity M2, the same duration), so the protocol world
    writes it there instead of passing a clock the CPU materializer cannot express exactly.
    """
    projected=json.loads(json.dumps(payload));yawns=[]
    for seat,side in projected['sides'].items():
        if 'yawn' not in side.get('volatiles',()):continue
        rows=[str(x).split('|') for x in lines]
        start=max((i for i,p in enumerate(rows) if len(p)>3 and p[1]=='-start' and p[2].startswith(seat+'a') and p[3]=='move: Yawn'),default=None)
        if start is None:raise ValueError('PUBLIC_YAWN_START_UNAVAILABLE')
        count=2-sum(1 for p in rows[start+1:] if len(p)>1 and p[1]=='upkeep')
        if not 1<=count<=2:raise ValueError('PUBLIC_YAWN_COUNT_UNAVAILABLE')
        side['volatiles']=[v for v in side['volatiles'] if v!='yawn'];yawns.append((int(seat[1])-1,count))
    return projected,tuple(yawns)


def _perish_counts(lines):
    """Per side (p1, p2): the Perish Song duration Showdown holds on that side's active, from the side's latest
    |-start|<side>a: X|perishN line, or None without one.

    Showdown's Perish Song (data/moves.ts) adds volatiles.perishsong with duration 4 and prints a SILENT
    |-start|X|perish3|[silent] at the use; each residual decrements the duration, then prints |-start|X|perishN (not
    silent) before that turn's |upkeep|. So a side whose latest line is the silent use line has had no residual since:
    4 (a replacement boundary in the Song's own turn); otherwise N. A Baton Pass copies the duration to the recipient
    (copyVolatileFrom) with no new line, so the side's latest line still gives the recipient's count.
    """
    counts=[None,None]
    for line in lines:
        p=str(line).split('|')
        if len(p)>3 and p[1]=='-start' and p[2][:3] in ('p1a','p2a') and p[3] in ('perish0','perish1','perish2','perish3'):
            n=int(p[3][-1]);counts[int(p[2][1])-1]=4 if n==3 and '[silent]' in p[4:] else n
    return tuple(counts)


def encore_support(k,b):
    """The remaining Encore count Showdown can hold after k residuals since the start, b = 1 when the target had already
    acted that turn: Gen 3 rolls D = random(3, 7) (data/mods/gen3/moves.ts), onStart adds one when the target no longer
    has a queued move (data/moves.ts: !this.queue.willMove(target)), and each residual decrements before the end check. Still
    active after k residuals means D + b > k, so the remaining count r = D + b - k is uniform on this support (empty: the
    lines contradict an active Encore). Pre-registered against Showdown f76228a1 (tests/fixtures/phase_d_held_volatiles)."""
    return tuple(range(max(1,3+b-k),7+b-k))


def _encore_facts(lines):
    """Per side (p1, p2): (k, b) of the side's latest |-start|<side>a: X|Encore line, or None without one, or when the lines
    show it over (a later |-end|<side>a: ...|Encore) or name another mon than the side's current active. k counts the
    residuals (|upkeep|) since it; b = 1 when that side had already acted earlier in the same turn: its own |move| or |cant|
    line, or a Gen 3 confusion self-hit (|-activate|<side>a: X|confusion, which uses the action and prints neither; the
    replacement phase's rule, phase_d_tail)."""
    rows=[str(x).split('|') for x in lines];out=[None,None]
    for side in (0,1):
        seat=f'p{side+1}a'
        start=max((i for i,p in enumerate(rows) if len(p)>3 and p[1]=='-start' and p[2].startswith(seat) and p[3]=='Encore'),default=None)
        if start is None:continue
        if any(len(p)>3 and p[1]=='-end' and p[2].startswith(seat) and p[3]=='Encore' for p in rows[start+1:]):continue
        active=max((i for i,p in enumerate(rows) if len(p)>2 and p[1] in ('switch','drag','replace') and p[2].startswith(seat)),default=None)
        if active is None or rows[active][2]!=rows[start][2] or active>start:continue
        turn=max((i for i in range(start) if rows[i][1:2]==['turn']),default=-1)
        acted=any(len(p)>2 and p[2].startswith(seat) and (p[1] in ('move','cant') or p[1]=='-activate' and len(p)>3 and normalize_id(p[3])=='confusion')
            for p in rows[turn+1:start])
        out[side]=(sum(1 for p in rows[start+1:] if p[1:2]==['upkeep']),int(acted))
    return tuple(out)


def _mapped_encores(physical):
    """(side, active, locked move, its slot) for each side whose active map_spec marked encored (held_volatiles)."""
    out=[]
    for side in (0,1):
        at=int(physical.active[0,side])
        if int(physical.encore_counter[0,side,at])>0:
            out.append((side,at,int(physical.encore_move[0,side,at]),int(physical.last_action_index[0,side])))
    return tuple(out)


def _write_encores(physical,encored,facts,seed,*,replacement,owner):
    """The world with each encored active's lock and a remaining count drawn uniformly from encore_support. The draw uses
    its own RNG per world and side, so no other draw of the world changes. Refuses: no start line, a support the lines
    rule out, a last action that is not the lock, or the locked move at 0 PP at an ordinary decision (Showdown's residual
    ends an Encore whose move has no PP before any request; a replacement boundary comes before that residual).

    Our own encored active: Showdown's request marks every move but the lock disabled (Encore's onDisableMove), and the own
    update copied those flags into the world, where nothing clears them, so the mon would stay locked after the drawn count
    ends. The engine masks the lock itself (encore_move/encore_counter), Choice (choice_move) and empty PP (pp) on its own,
    so once the request shows exactly that pattern (the lock enabled, every other move disabled or out of PP) the flags are
    cleared; any other pattern refuses."""
    for side,at,move,slot in encored:
        if facts[side] is None:raise ValueError('PUBLIC_ENCORE_START_UNAVAILABLE')
        support=encore_support(*facts[side])
        if not support:raise ValueError('PUBLIC_ENCORE_COUNT_UNAVAILABLE')
        if int(physical.last_action_kind[0,side])!=2 or int(physical.last_action_index[0,side])!=slot or int(physical.moves[0,side,at,slot])!=move:
            raise ValueError('PUBLIC_ENCORE_LOCK_DISAGREEMENT')
        if int(physical.pp[0,side,at,slot])==0 and not replacement:raise ValueError('PUBLIC_ENCORE_ZERO_PP')
        if side==owner:
            ids=np.asarray(physical.moves)[0,side,at];off=np.asarray(physical.disabled)[0,side,at];pp=np.asarray(physical.pp)[0,side,at]
            if off[slot] or any(ids[k]>0 and not off[k] and pp[k]>0 for k in range(len(ids)) if k!=slot):
                raise ValueError('PUBLIC_ENCORE_REQUEST_DISAGREEMENT')
            physical=physical._replace(disabled=physical.disabled.at[0,side,at].set(False))
        count=random.Random(f'encore:{seed}:{side}').choice(support)
        physical=physical._replace(encore_move=physical.encore_move.at[0,side,at].set(move),
            encore_counter=physical.encore_counter.at[0,side,at].set(count))
    return physical


def _write_held_counts(physical,yawns,perish):
    """The mapped world with Yawn's public count (_defer_yawn) and Perish Song's duration (_perish_counts) on each side's
    active; neither is an own field. map_spec wrote the shown perishN; a count the lines do not explain refuses."""
    for side,count in yawns:
        physical=physical._replace(yawn_counter=physical.yawn_counter.at[0,side,physical.active[0,side]].set(count))
    for side in (0,1):
        at=int(physical.active[0,side]);shown=int(physical.perish_counter[0,side,at])
        if not shown:continue
        if perish[side] is None or min(perish[side],3)!=shown:raise ValueError('PUBLIC_PERISH_COUNT_UNAVAILABLE')
        physical=physical._replace(perish_counter=physical.perish_counter.at[0,side,at].set(perish[side]))
    return physical


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
        projected,yawns=_defer_yawn(projected,inputs.lines)
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
            turn=root.materialization_payload['turn'],active_turns=_once(inputs,root,'chronology',lambda:_chronology(inputs.lines)),happiness=tuple(mood),held_volatiles=True)
        if mapped.state is None:
            _downstream_unknowns(diagnostics,'Typed mapper refused: '+repr(mapped.refusals))
            return ProtocolWorld(None,None,mapped.refusals,mapper_diagnostics=tuple(diagnostics))
        physical=tail.restore(mapped.state,world.spec,cursors,tail_facts,metadata=metadata,checks=diagnostics)
        physical=_write_held_counts(physical,yawns,_perish_counts(inputs.lines))
        encored=_mapped_encores(physical)   # read before the own update, which writes our own (zero) clock
        maximum_pp={live._source_move(metadata['vocabulary']['moves'][int(mid)]):int(metadata['moves'][int(mid)]['pp'])*8//5 for row in detached.own['moves'] for mid in row if mid}
        own=dict(detached.own);own['pp']=np.asarray(_once(inputs,root,'own_pp',lambda:own_pp_carry(inputs.lines,state.self_request,maximum_pp)),np.int32)
        # Exact facts were read from the owning packet by B2's detached root.
        # Check all mapper-provided literal own fields before preserving the
        # complete own packet state (including exact private request flags).
        failures=[]
        for name in ('species','level','hp','max_hp','stats','moves','item','ability'):
            actual=np.asarray(getattr(physical,name))[0,owner];expected=own[name]
            fail=not np.array_equal(actual,expected)
            if fail and name=='species':
                # The mapper writes the engine id (engine_world._engine_species_id collapses Unown letters); B2 keeps the
                # request's forme. Equal after the same collapse: the own update below writes B2's literal forme.
                engine=lambda xs:[_engine_species_id(normalize_id(metadata['vocabulary']['species'][int(x)]))for x in xs]
                fail=engine(actual)!=engine(expected)
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
        try:physical=_write_encores(physical,encored,_once(inputs,root,'encore_facts',lambda:_encore_facts(inputs.lines)) if encored else (None,None),
            seed,replacement=tail_facts['replacement'] is not None,owner=owner)
        except ValueError as error:failures.append(str(error))
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


# Showdown protocol lines that carry no game state: wall clock (|t:| is the turn's unix time; |inactive| / |inactiveoff| the
# timer) and room transport (chat, join, leave, rename, raw HTML). Two runs of the same game differ only in these; r4a3's
# owning wire carries |t:| alone (25,823 of its lines), the rest are listed defensively.
NON_GAME_LINE_TAGS=frozenset(('t:','c','c:','chat','j','J','join','l','L','leave','n','N','name','raw','html','uhtml',
    'uhtmlchange','inactive','inactiveoff'))


def game_lines_sha256(lines):
    """sha256 of the owner's protocol lines without the non-game ones (NON_GAME_LINE_TAGS), in the owner_lines_sha256 format:
    the same game state hashes the same in every run, whatever the wall clock said. It seeds the drawn worlds;
    owner_lines_sha256 stays the exact-bytes record of the root."""
    keep=(x for x in lines if not (x.startswith('|') and x.split('|',2)[1] in NON_GAME_LINE_TAGS))
    return hashlib.sha256(''.join(x+'\n'for x in keep).encode()).hexdigest()


def world_seeds(root,detached,game_lines,seed,index):
    """World `index`'s (world_seed, kernel_seed): sha256 over the run's seed, the battle, the game content of the owner's lines
    (game_lines = game_lines_sha256(lines), never the exact bytes: |t:| would make the same state draw different worlds in
    each run), the decision address and the index. world_seed drives the start sampler, kernel_seed everything after it."""
    identity=(SCHEMA,seed,root.context.battle_id,game_lines,detached.address.stream,detached.address.frame,detached.address.owner,index)
    digest=hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).digest()
    return int.from_bytes(digest[:8],'big'),int.from_bytes(digest[8:12],'big')


def sample(root,detached,*,lines,catalog,source,metadata,config,battle_id,seed,count):
    """Same public posterior draws and their prefix at all ladder budgets."""
    from . import v5_public_world
    if count not in (2,8,16,32,64):raise ValueError('declared2/8/16/32/64 world budget required')
    if detached.source_binding_sha256!=source.binding_sha256:
        raise live.Refusal('SAME_PUBLIC_ROOT_SOURCE_BINDING_REQUIRED')
    detached=_carried_root(root,detached,lines,metadata)
    inputs=ProtocolInputs(detached,tuple(lines),catalog,metadata,source)
    result=[];seeds=[];diagnostics=[];game=game_lines_sha256(lines)
    for index in range(count):
        world_seed,kernel_seed=world_seeds(root,detached,game,seed,index);seeds.append(world_seed)
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
