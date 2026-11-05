"""Actual witness FULL frames + owning request facts, before any token encoding.

No engine state/peer request/raw private callback tape is read. The original
current Session is checked at both ends of capture. Canonical words remain the
authority; any unsupported inverse refuses the entire parser materialization,
while retaining every attempted frame and its denominator for diagnostics.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from . import engine_retained_encoder_driver as driver, public_inputs, public_wire, v4_encoder
from . import private_actions, private_requests


def _freeze(value):
    if isinstance(value, np.ndarray):return _freeze(value.tolist())
    if isinstance(value, (list, tuple)):return tuple(_freeze(x) for x in value)
    if isinstance(value, np.generic):return value.item()
    return value


@dataclass(frozen=True)
class Frame:
    number: int
    words: tuple
    valid: tuple
    more: tuple
    request_json: str | None
    refusal: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class History:
    session_identity: tuple[int, int, int]
    row: int
    owner: int
    stream_id: int
    catalog: object
    own_constructor: tuple  # exact own binding/species/moves/maxPP, never peer
    frames: tuple[Frame, ...]


def _request(core, row, owner, catalog, metadata):
    """Exact source-fact inverse, accepted only after ORIGINAL parser roundtrip.

    The engine has no source JSON text. This emits canonical request JSON,
    rather than claiming byte identity with a simulator request. Party moves
    come from the ORIGINAL own constructor, not inferred from tokens/peer facts.
    """
    facts = core.requests[owner].actions.facts
    request, team = facts.request, facts.team
    if not bool(np.asarray(core.requests[owner].actions.admitted[row])):
        raise ValueError('own-source-request-refused')
    if int(np.asarray(request.owner_side[row])) != owner or int(np.asarray(team.owner_side[row])) != owner:
        raise ValueError('own-request-owner-mismatch')
    cache={}
    def at(value):
        key=id(value)
        if key not in cache:cache[key]=np.asarray(value[row])
        return cache[key]
    def token(index, prefix):
        if not 0 < int(index) <= len(catalog.vocabulary.tokens):raise ValueError('own-request-unknown-category')
        text = catalog.vocabulary.tokens[int(index)-1]
        if not text.startswith(prefix):raise ValueError('own-request-category-namespace')
        return text[len(prefix):]
    mapping = at(core.orders[owner].request_to_physical)
    original = core.originals[owner]
    engine_moves = metadata['vocabulary']['moves']
    rows = []
    for slot in range(int(at(request.team_size))):
        if not bool(at(team.present)[slot]):raise ValueError('own-request-noncontiguous-party')
        physical = int(mapping[slot])
        if not 0 <= physical < 6:raise ValueError('own-request-party-map')
        categories = at(team.categories)[slot]
        name = token(categories[0], 'species:')
        status = token(categories[1], 'status:') if categories[1] else ''
        if status == 'none':status = ''
        hp, maximum = int(at(team.hp_numerator)[slot]), int(at(team.hp_denominator)[slot])
        gender = ('', ', M', ', F')[int(at(team.gender)[slot])]
        ability = token(categories[5], 'belief:possible_ability:') if categories[5] else ''
        item = token(categories[6], 'belief:possible_item:') if categories[6] else ''
        moves = []
        for move in at(original.moves)[physical]:
            if int(move):moves.append(engine_moves[int(move)])
        stats = at(team.actual_stats)[slot]
        if any(not float(value).is_integer() for value in stats):raise ValueError('own-request-noninteger-source-stat')
        rows.append(dict(ident=f'p{owner+1}: {name}', details=f'{name}, L{int(at(team.level)[slot])}'+gender,
            condition='0 fnt' if bool(at(team.fainted)[slot]) else f'{hp}/{maximum}'+(' '+status if status else ''),
            active=bool(at(team.active)[slot]), moves=moves, baseAbility=ability, item=item,
            stats={key:int(stats[i+1]) for i,key in enumerate(('atk','def','spa','spd','spe'))}))
    result = dict(side=dict(id=f'p{owner+1}', pokemon=rows))
    rqid = int(at(request.request_id))
    if rqid >= 0:result['rqid'] = rqid
    kind = int(at(request.kind))
    if kind == private_requests.RequestKind.WAIT:result['wait'] = True
    elif kind == private_requests.RequestKind.FORCE_SWITCH:result['forceSwitch'] = [True]
    elif kind == private_requests.RequestKind.MOVE:
        moves = []
        for index in range(4):
            if not bool(at(request.move_present)[index]):continue
            name = catalog.actions.names[int(at(facts.mechanics)[index])]
            move = dict(id=name, disabled=bool(at(request.move_disabled)[index]))
            if bool(at(request.pp_known)[index]):
                pp, maximum = float(at(request.pp)[index]),float(at(request.max_pp)[index])
                if not pp.is_integer() or not maximum.is_integer():raise ValueError('own-request-noninteger-source-pp')
                move.update(pp=int(pp), maxpp=int(maximum))
            moves.append(move)
        result['active'] = [dict(moves=moves, trapped=bool(at(request.trapped)))]
    else:raise ValueError('own-request-phase-unsupported')
    parsed = private_actions.parse_action_facts(result, owner_side=owner,
        vocabulary=catalog.vocabulary, catalog=catalog.actions)
    for left, right in zip(parsed.request, request, strict=True):
        if not np.array_equal(np.asarray(left), at(right)):raise ValueError('own-request-action-roundtrip')
    for field,left,right in zip(parsed.team._fields,parsed.team,team,strict=True):
        # Vocabulary binding is global in host parser and row-bound in source.
        if not np.array_equal(np.asarray(left), at(right)):raise ValueError('own-request-team-roundtrip:'+field)
    if not np.array_equal(np.asarray(parsed.mechanics), at(facts.mechanics)):
        raise ValueError('own-request-move-slot-roundtrip')
    return json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)


def capture(session, *, row, owner, history=None):
    """Call at construction and after EACH original advance, including refusals."""
    record = driver._session(session)
    runtime = driver._checked(session.prepared)
    driver._checked_witness(runtime, session=session)
    core = session.state.core
    if type(row) is not int or not 0 <= row < core.frame_number.shape[0] or type(owner) is not int or owner not in (0,1):
        raise ValueError('one exact owning row/seat required')
    number = int(np.asarray(core.frame_number[row]))
    stream = int(np.asarray(core.cursor.stream_id[row]))
    if history is None:
        if number != 0:raise ValueError('owning history must begin at original constructor frame0')
        constructor=core.constructors[owner]
        own_constructor=tuple((name,_freeze(np.asarray(getattr(constructor,name)[row])))
            for name in ('binding','species','moves','max_pp'))
        # Only numeric lineage is retained. No Session/Core handle (which also
        # reaches latent peer data) crosses into the captured owner history.
        history = History((id(session),id(session.prepared),id(record['constructed'])),
            row,owner,stream,runtime['catalog'],own_constructor,())
    elif (type(history) is not History or history.session_identity !=
            (id(session),id(session.prepared),id(record['constructed'])) or history.row != row
            or history.owner != owner or history.stream_id != stream or history.catalog is not runtime['catalog']
            or number != history.frames[-1].number + 1):
        raise ValueError('same original session and consecutive unreplayed owning frames required')
    report = core.frame_input_report
    if report is None:raise ValueError('original current frame-witness FULL capture required')
    # Slice on device before host transfer; never copy another owner's words.
    words = np.asarray(report.owner_full_words[row,owner])
    valid = np.asarray(report.owner_full_valid[row,owner])
    more = np.asarray(report.owner_full_more[row,owner])
    if words.dtype != np.int32 or words.shape != (*valid.shape,16) or valid.dtype != bool or more.dtype != bool or more.shape != valid.shape[:1]:
        raise ValueError('original FULL chunk shape/dtype required')
    if np.any(valid[:,1:] & ~valid[:,:-1]):raise ValueError('FULL valid words must be contiguous')
    if any(bool(more[index]) != bool(valid[index+1:].any()) for index in range(len(more))):
        raise ValueError('FULL more flags must drain every original chunk')
    selected = words[valid]
    previous = [word[1] for frame in history.frames for chunk, mask in zip(frame.words,frame.valid) for word,use in zip(chunk,mask) if use]
    first = previous[-1]+1 if previous else 1
    if selected.size and not np.array_equal(selected[:,1], np.arange(first, first+len(selected))):
        raise ValueError('owning FULL sequence gap/replay')
    refusals = []
    if not bool(np.asarray(report.frame_inputs_complete[row])):refusals.append(('source-frame-refused',1))
    if int(np.asarray(report.refusal[row])):refusals.append(('source-frame-refusal-bits',int(np.asarray(report.refusal[row]))))
    try:request = _request(core,row,owner,runtime['catalog'],json.loads(runtime['request']['metadata']))
    except (ValueError,IndexError,KeyError) as error:
        request = None;refusals.append((str(error),1))
    frame = Frame(number,_freeze(words),_freeze(valid),_freeze(more),request,tuple(refusals))
    if driver._session(session) is not record or session.state is not record['state']:
        raise ValueError('original current Session changed during owning capture')
    return History(history.session_identity,row,owner,stream,runtime['catalog'],
        history.own_constructor,history.frames+(frame,))


@dataclass(frozen=True)
class Decoded:
    lines: tuple[str, ...]
    required_raw_groups: int
    unsupported_raw_groups: int
    refusal_counts: tuple[tuple[str, int], ...]

    @property
    def complete(self):return not self.refusal_counts


def decode(history):
    """Only source-invertible schemas, VERIFIED by original FULL writer equality.

    These are canonical equivalents of engine records, never claimed literal
    simulator text. A failed inverse is not skipped to produce a usable prefix.
    """
    catalog = history.catalog; state = public_wire.WireState(); output=[]; counts={}; required=unknown=0
    def note(reason):counts[reason]=counts.get(reason,0)+1
    for frame in history.frames:
        for reason,count in frame.refusal:counts[reason]=counts.get(reason,0)+count
        rows = [word for chunk,mask in zip(frame.words,frame.valid) for word,use in zip(chunk,mask) if use]
        groups=[]
        for word in rows:
            if word[0] == public_inputs.Kind.RAW_BEGIN:groups.append([])
            if not groups:raise ValueError('FULL frame starts without original raw boundary')
            groups[-1].append(word)
        for group in groups:
            required += 1
            try:candidates = _candidates(group,state,catalog)
            except (ValueError,IndexError,KeyError):candidates=()
            accepted = None
            for line in candidates:
                try:
                    after, projected = public_wire.project_line(state,line,**v4_encoder.wire_kwargs(catalog))
                    if tuple(tuple(map(int,event.words())) for event in projected.events) == tuple(group):
                        accepted=(after,line);break
                except (ValueError,IndexError):pass
            if accepted is None:
                unknown += 1;note('canonical-source-family-not-invertible:'+str(group[0][11]))
            else:
                state,line=accepted;output.append(line)
        if frame.request_json is None:note('own-request-unavailable')
        else:output.append('|request|'+frame.request_json)
    return Decoded(tuple(output),required,unknown,tuple(sorted(counts.items())))


def _candidates(group,state,catalog):
    """Bounded original lexical schemas; no inference from final state/tokens."""
    K=public_inputs.Kind
    main=[w for w in group if w[0] in (K.SWITCH,K.DRAG,K.REPLACE,K.TURN,K.UPKEEP,K.ACTION_CHUNK_END,K.MOVE,K.CONDITION,K.FAINT,K.ABILITY)]
    if len(main)!=1:return ()
    word=main[0];kind=word[0];p=word[8:]
    def ident(side):return state.active_idents()[side]
    if kind in (K.SWITCH,K.DRAG,K.REPLACE):
        name=catalog.species.names[p[1]];gender=('',', M',', F')[p[4]]
        status=('', ' slp',' par',' brn',' frz',' psn',' tox')[p[7]]
        return (f'|{K(kind).name.lower()}|p{word[6]+1}a: {name}|{name}, L{p[3]}{gender}|{p[5]}/{p[6]}{status}',)
    if kind==K.TURN:return (f'|turn|{word[2]}',)
    if kind==K.UPKEEP:return ('|upkeep',)
    if kind==K.ACTION_CHUNK_END:return ('|',)
    try:
        if kind==K.MOVE:
            if p[2] or p[3]:return ()
            return (f'|move|{ident(word[4])}|{catalog.moves.names[p[1]]}|{ident(word[6])}',)
        if kind==K.FAINT:return (f'|faint|{ident(word[6])}',)
        if kind==K.ABILITY:
            if p[2]!=1:return ()
            prefix=f'|-ability|{ident(word[6])}|{catalog.abilities.names[p[1]]}'
            return (prefix,prefix+'|[silent]')
        if kind==K.CONDITION:
            condition='0 fnt' if p[4] else f'{p[1]}/{p[2]}'+('', ' slp',' par',' brn',' frz',' psn',' tox')[p[3]]
            kinds=('-damage',) if p[5] else ('-heal','-sethp')
            return tuple(f'|{name}|{ident(word[6])}|{condition}'+tag for name in kinds for tag in
                ('','|[from] item: Leftovers','|[from] psn','|[from] brn','|[from] Sandstorm','|[from] Hail'))
    except (KeyError,IndexError):return ()
    return ()


@dataclass(frozen=True)
class Materialized:
    root: object | None
    decoded: Decoded


def public_root(history, *, config, contract, battle_id, seed):
    """Bridge fully invertible ORIGINAL engine history to the original planner.

    No invented |start/format/header text is inserted. The original constructor
    frame0 capture establishes a complete prefix, and the production owning
    parser already initializes complete-prefix clocks. The hash names the
    canonical equivalent lines, NOT nonexistent original simulator wire bytes.
    Unsupported canonical groups return no root, never a partial public state.
    """
    from dataclasses import replace
    from pokezero import v5_contract as c
    from pokezero.live_protocol_actor import PythonBackend
    from pokezero.local_showdown import _public_materialization_payload
    from pokezero.policy import PolicyContext
    from pokezero.trajectory import BattleTrajectory
    from pokezero.v5_teacher_public_root import PublicRoot
    decoded=decode(history)
    if not decoded.complete:return Materialized(None,decoded)
    if type(seed) is not int or not 0<=seed<2**32 or type(battle_id) is not str or not battle_id:
        raise ValueError('literal original battle identity required')
    owner=f'p{history.owner+1}'
    backend=PythonBackend(contract,config=config,battle_id=battle_id,owner=owner,stream_id=history.stream_id)
    try:
        first=None
        for line in decoded.lines:
            backend.ingest((line,))
            if first is None and line.startswith('|request|'):first=backend.env.observe(owner)
        observation=backend.env.observe(owner)
        state=backend.env.public_materialization_state(owner)
        observation=replace(observation,metadata={**dict(observation.metadata),
            'self_team':first.metadata['self_team'],
            'v5_owning_history_source':'ACTUAL_ENGINE_CANONICAL_FULL_NOT_LITERAL_SIMULATOR_WIRE'})
        address=c.DecisionAddress(history.stream_id,history.frames[-1].number,history.owner)
        context=PolicyContext(owner,address.frame,battle_id,state.format_id,seed,observation,
            (owner,),BattleTrajectory(battle_id,state.format_id,seed),
            {owner:tuple(observation.legal_action_mask)},{owner:observation},state)
        payload=_public_materialization_payload(state)
        counts={}
        for side in payload['sides'].values():
            for reason in side['materializationBlockers']:counts[reason]=counts.get(reason,0)+1
        if payload['pendingBatonPassSides']:counts['pending-baton-phase-needs-device-constructor']=1
        digest=hashlib.sha256(''.join(line+'\n' for line in decoded.lines).encode()).hexdigest()
        return Materialized(PublicRoot(address,digest,context,payload,tuple(sorted(counts.items()))),decoded)
    except (ValueError,KeyError,IndexError) as error:
        counts=dict(decoded.refusal_counts)
        reason='original-owning-parser:'+type(error).__name__+':'+str(error)
        counts[reason]=counts.get(reason,0)+1
        return Materialized(None,Decoded(decoded.lines,decoded.required_raw_groups,
            decoded.unsupported_raw_groups,tuple(sorted(counts.items()))))
    finally:backend.close()
