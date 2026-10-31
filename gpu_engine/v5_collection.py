"""Explicit bounded original-engine-session -> v5 raw actor -> capture bridge.

No constructor, team generator, pool iteration, search, CLI, or default loader
dispatch is installed. A caller supplies an already constructed original driver
session and a fixed declaration. Refused/failed roots stay in that declaration;
this module never retries, replaces teams, or constructs another session.
"""
from dataclasses import dataclass
import json
from types import MappingProxyType
import weakref

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_contract as c, v5_data
from pokezero.v5_model import MODEL_INPUTS
from . import engine_retained_encoder_driver as driver, engine_private_actions as actions
from . import private_requests, source_provenance, v5_model, v5_observation
from . import v5_collection_data as data


SCHEMA = 'pokezero.actual_engine_collector.v5'


@dataclass(frozen=True)
class Cell:
    stream_id: int
    battle_id: str
    seed: int
    team_indices: tuple[int, int]

    def __post_init__(self):
        object.__setattr__(self, 'team_indices', tuple(self.team_indices))
        if (type(self.stream_id) is not int or not 0 < self.stream_id < 2**31
                or type(self.battle_id) is not str or not self.battle_id
                or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or len(self.team_indices) != 2 or any(type(i) is not int or i < 0 for i in self.team_indices)
                or len(set(self.team_indices)) != 2):
            raise ValueError('explicit fixed stream/battle/seed/two original team rows required')


@dataclass(frozen=True, eq=False)
class Ledger:
    cells: tuple[Cell, ...]

    def report(self):
        record = _ledger(self); status = record['status']; detail = record['detail']
        return dict(schema=SCHEMA, cells=[dict(stream_id=cell.stream_id,battle_id=cell.battle_id,seed=cell.seed,
            team_indices=cell.team_indices,status=status,detail=dict(detail))
            for cell,status,detail in zip(self.cells,status,detail)],
            counts={name:self.status.count(name) for name in ('NOT_ATTEMPTED','RUNNING','COMPLETE','CAPPED','REFUSED','FAILED')})

    @property
    def status(self):return tuple(_ledger(self)['status'])


_LEDGERS = weakref.WeakKeyDictionary()


def _ledger(ledger):
    record = _LEDGERS.get(ledger) if type(ledger) is Ledger else None
    if record is None or ledger.cells is not record['cells']:raise ValueError('original fixed collection ledger required')
    return record


def declare_cells(cells):
    cells = tuple(cells)
    if (not cells or any(type(cell) is not Cell for cell in cells)
            or len({cell.stream_id for cell in cells}) != len(cells)
            or len({cell.battle_id for cell in cells}) != len(cells)
            or len({i for cell in cells for i in cell.team_indices}) != 2 * len(cells)):
        raise ValueError('finite unique declared cells and distinct fixed team rows required')
    ledger = Ledger(cells)
    _LEDGERS[ledger] = dict(cells=cells,status=['NOT_ATTEMPTED']*len(cells),detail=[{} for _ in cells])
    return ledger


@dataclass(frozen=True, eq=False)
class Actor:
    params: object
    config: object
    adapter: object
    metadata: object


_ACTORS = weakref.WeakKeyDictionary()


def load_actor(prepared, archive, *, expected_archive_sha256, expected_checkpoint_sha256,
        expected_category_vocabulary_sha256, expected_label_vocabulary_sha256,
        compute_dtype='float32', enable_v5=False):
    if enable_v5 is not True:raise ValueError('explicit v5 actual collector enablement required')
    if compute_dtype not in ('float32','bfloat16'):raise ValueError('explicit inference precision required')
    r = driver._checked(prepared); driver._checked_witness(r)
    params, config, metadata = v5_model.load(archive, enable_v5=True,
        expected_archive_sha256=expected_archive_sha256, expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_category_vocabulary_sha256=expected_category_vocabulary_sha256,
        expected_label_vocabulary_sha256=expected_label_vocabulary_sha256)
    adapter = v5_observation.compile_adapter(r['catalog'],model_config=config)
    params = MappingProxyType(dict(params)); metadata = json.dumps(metadata,sort_keys=True,allow_nan=False)
    actor = Actor(params,config,adapter,metadata)
    _ACTORS[actor] = dict(prepared=weakref.ref(prepared),params=params,config=config,adapter=adapter,metadata=metadata,
        forward=jax.jit(lambda inputs:v5_model.forward(params,inputs,config,compute_dtype=compute_dtype)),
        append=jax.jit(lambda encoded,state,owner:v5_observation.append_device(encoded,state,adapter,observer_sides=owner)))
    return actor


def _checked(actor, session):
    record = _ACTORS.get(actor) if type(actor) is Actor else None
    if (record is None or record['prepared']() is None or record['prepared']() is not session.prepared
            or any(getattr(actor,name) is not record[name] for name in ('params','config','adapter','metadata'))):
        raise ValueError('original actor and same prepared runtime required')
    current = driver._session(session)
    r = driver._checked(session.prepared); driver._checked_witness(r,session=session)
    if v5_observation.check_adapter(actor.adapter) is not r['catalog']:
        raise ValueError('same original retained catalog required')
    return record,r,current


def _pool_truth(pool, indices, session, runtime, current):
    """Training-only original constructor set identity; never actor kwargs."""
    receipt = source_provenance._pool(pool,rehash=True)
    constructed = driver.source._constructed(current['constructed'])
    birth = constructed.birth_state
    if birth is None:raise ValueError('actual original pre-entry constructor fields required')
    metadata = pool.metadata()
    if json.dumps(metadata,sort_keys=True) != runtime['request']['metadata']:
        raise ValueError('same original constructor metadata required')
    b = birth.hp.shape[0]
    if indices.shape != (b,2) or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= len(pool.arrays['species'])):
        raise ValueError('same fixed original constructor team indices required')
    streams = np.asarray(session.state.core.cursor.stream_id)
    for side, registration in enumerate(constructed.registrations):
        driver.source._registered(registration)
        retained = actions._constructor(registration.constructor)
        for row in range(b):
            payload = dict(schema=actions.SCHEMA+'.constructor',pool=receipt['pool_sha256'],
                receipt=pool.receipt_sha256,selection=int(indices[row,side]),stream=int(streams[row]),owner=side,
                engine=np.asarray(retained.engine_binding[row]).tolist())
            expected = np.frombuffer(bytes.fromhex(data.sha(payload)),dtype='>u4').astype(np.uint32)
            if not np.array_equal(retained.binding[row],expected):raise ValueError('original constructor pool/selection binding differs')
    for actual, original in (('species','species'),('level','level'),('moves','moves'),('base_ability','ability'),
            ('item','item'),('hidden_power_type','hidden_power_type'),('hidden_power_power','hidden_power_power')):
        if not np.array_equal(np.asarray(getattr(birth,actual)),pool.arrays[original][indices]):
            raise ValueError('literal original constructor set differs: '+actual)
    vocabulary = metadata['vocabulary']; source = runtime['catalog'].source
    truth = []
    for row in range(b):
        pair = []
        for side in (0,1):
            index = indices[row,side]; team = []
            for physical in range(6):
                value = lambda name:int(pool.arrays[name][index,physical])
                species = vocabulary['species'][value('species')]
                moves = tuple(vocabulary['moves'][int(move)] for move in pool.arrays['moves'][index,physical] if move)
                team.append(data.matching_hidden_set(species,level=value('level'),moves=moves,
                    ability=vocabulary['abilities'][value('ability')],item=vocabulary['items'][value('item')],
                    hidden_power_type=vocabulary['types'][value('hidden_power_type')],source=source))
            pair.append(tuple(team))
        truth.append(tuple(pair))
    return tuple(truth),receipt


def _public_inputs(actor_record, result, active):
    appended = []; fields = {name:[] for name in MODEL_INPUTS}; eligible = []
    for owner,(report,observer) in enumerate(zip(result.owners,result.state.observers)):
        output = actor_record['append'](report.encoded,observer.retained,observer.owner)
        jax.block_until_ready(output); appended.append(output)
        request = result.state.core.requests[owner].actions
        ready = private_requests.action_inputs(request.facts.request,observer_side=observer.owner).ready
        use = np.asarray(active) & np.asarray(ready) & np.asarray(report.policy_admitted) & np.asarray(output.admitted)
        eligible.append(use)
        values = v5_observation.actor_inputs(output,actor_record['adapter'])
        for name in MODEL_INPUTS:
            value = values[name]
            if name == 'history_mask':value = j.asarray(use[:,None])
            fields[name].append(value)
    inputs = {name:j.stack(pair,axis=1).reshape((-1,*pair[0].shape[1:])) for name,pair in fields.items()}
    return tuple(appended),inputs,np.stack(eligible,axis=1)


def _decide(actor_record, inputs, legal, eligible, streams, frames, sampling_seed):
    """One public-only raw actor call; record its literal sampled probability."""
    b = len(streams); choices = np.full((b,2),-1,np.int32)
    probabilities = np.zeros((b,2),np.float32); values = np.zeros((b,2),np.float32)
    failed = np.zeros(b,bool)
    if not eligible.any():return choices,probabilities,values,failed
    output = jax.block_until_ready(actor_record['forward'](inputs))
    logits = np.asarray(output.policy_logits).reshape(b,2,9); value = np.asarray(output.value).reshape(b,2)
    finite = np.asarray(output.valid).reshape(b,2) & np.isfinite(logits).all(-1) & np.isfinite(value) & (np.abs(value)<=1)
    failed |= np.any(eligible & ~finite,axis=1)
    for row,owner in zip(*np.nonzero(eligible & finite)):
        allowed = np.asarray(legal[row,owner],bool)
        if not allowed.any():failed[row]=True;continue
        scores = j.where(j.asarray(allowed),j.asarray(logits[row,owner]),-j.inf)
        key = jax.random.fold_in(jax.random.PRNGKey(sampling_seed),int(streams[row]))
        key = jax.random.fold_in(jax.random.fold_in(key,int(frames[row])),int(owner))
        choice = int(jax.random.categorical(key,scores))
        probability = float(jax.nn.softmax(scores)[choice])
        if not np.isfinite(probability) or not 0 < probability <= 1:failed[row]=True;continue
        choices[row,owner]=choice;probabilities[row,owner]=probability;values[row,owner]=value[row,owner]
    return choices,probabilities,values,failed


def _physical(core, row, owner):
    own = tuple(map(int,np.asarray(core.orders[owner].request_to_physical)[row]))
    mapping = np.asarray(core.cursor.physical_to_public)[row,1-owner]
    opponent = [-1]*6
    for physical,slot in enumerate(mapping):
        if slot >= 0:
            if slot >= 6 or opponent[int(slot)] != -1:raise ValueError('ambiguous actual public encounter map')
            opponent[int(slot)]=physical
    return own+tuple(opponent)


def _semantic(core, routes, choices, row, owner, runtime):
    choice = int(choices[row,owner])
    if choice < 0:return None
    if choice < 4:
        category = int(np.asarray(core.requests[owner].actions.facts.move_primary)[row,choice])
        vocabulary = runtime['catalog'].vocabulary
        if not 1 <= category <= len(vocabulary.tokens):return None
        label = vocabulary.tokens[category-1]
        if not label.startswith('move:') or not label[5:]:return None
        return c.OpponentAction('MOVE',label[5:])
    physical = int(np.asarray(routes[owner].physical_switch_target)[row])
    if not 0 <= physical < 6:return None
    species = int(np.asarray(core.originals[owner].species)[row,physical])
    names = json.loads(runtime['request']['metadata'])['vocabulary']['species']
    if not 0 < species < len(names):return None
    return c.OpponentAction('SWITCH',data.canonical_species(names[species]))


@dataclass(frozen=True)
class Collection:
    decisions: tuple[data.CapturedDecision,...]
    terminals: tuple[data.TerminalCapture,...]
    ledger: dict


def collect_session(session, *, actor, pool, team_indices, ledger, max_requests, sampling_seed,
        include_privileged=False, enable_v5=False):
    """Advance only the supplied actual session for a declared finite prefix.

    Fakes in tests replace the original driver's methods; no production fake
    backend or caller acceptance flag exists. Returned rows are training data,
    not scientific strength, throughput, or physics qualification.
    """
    if enable_v5 is not True:raise ValueError('explicit v5 actual collector enablement required')
    if (type(ledger) is not Ledger or type(max_requests) is not int or max_requests <= 0
            or type(sampling_seed) is not int or not 0 <= sampling_seed < 2**32
            or type(include_privileged) is not bool):raise ValueError('finite explicit collection recipe required')
    book = _ledger(ledger); status = book['status']; detail = book['detail']
    driver._session(session)  # A supplied actual constructor has attempted these cells.
    indices = np.asarray(team_indices); core = session.state.core
    streams = np.asarray(core.cursor.stream_id); b = len(streams)
    by_stream = {cell.stream_id:i for i,cell in enumerate(ledger.cells)}
    if len(set(map(int,streams))) != b or any(int(stream) not in by_stream for stream in streams):
        raise ValueError('actual session differs from declared stream roster')
    positions = [by_stream[int(stream)] for stream in streams]
    expected = [index for index,value in enumerate(status) if value=='NOT_ATTEMPTED'][:b]
    if positions != expected:
        raise ValueError('session must use the next not-attempted declared prefix in original order')
    for row,pos in enumerate(positions):
        if status[pos] != 'NOT_ATTEMPTED' or tuple(map(int,indices[row])) != ledger.cells[pos].team_indices:
            raise ValueError('declared attempted cells cannot be replayed/refilled or replaced')
    for pos in positions:status[pos]='RUNNING';detail[pos]={'decisions':0,'executed_frames':0}
    try:
        actor_record,r,current = _checked(actor,session)
        truth,receipt = _pool_truth(pool,indices,session,r,current)
    except Exception as error:
        terminals=[]
        for row,pos in enumerate(positions):
            reason=type(error).__name__+': '+str(error)
            frame=int(np.asarray(core.frame_number)[row])
            terminal=data.TerminalCapture(int(streams[row]),frame,'FAILED',data.sha(dict(
                schema=SCHEMA,stream=int(streams[row]),frame=frame,status='FAILED',reason=reason)))
            terminals.append(terminal);status[pos]='FAILED';detail[pos].update(reason=reason,source_sha256=terminal.source_sha256)
        return Collection((),tuple(terminals),ledger.report())
    decisions=[]; terminals={}

    def close(row,disposition,reason):
        pos=positions[row]
        if status[pos] != 'RUNNING':return
        engine=session.state.core.engine; frame=int(np.asarray(session.state.core.frame_number)[row])
        body=dict(schema=SCHEMA,stream=int(streams[row]),frame=frame,status=disposition,
            pool_sha256=receipt['pool_sha256'],reason=reason)
        if disposition=='COMPLETE':body.update(winner=int(np.asarray(engine.winner)[row]),
            hp=np.asarray(engine.hp)[row].tolist(),max_hp=np.asarray(engine.max_hp)[row].tolist())
        terminal=data.TerminalCapture(int(streams[row]),frame,disposition,data.sha(body),
            body.get('winner'),_as_tuple(body.get('hp',())),_as_tuple(body.get('max_hp',())))
        terminals[int(streams[row])]=terminal
        status[pos]=disposition;detail[pos].update(reason=reason,source_sha256=terminal.source_sha256)

    try:
        for _ in range(max_requests):
            actor_record,r,current=_checked(actor,session); core=session.state.core; result=session.result
            active=np.asarray([status[pos]=='RUNNING' for pos in positions])
            for row in np.flatnonzero(active & np.asarray(core.engine.done)):close(int(row),'REFUSED','terminal-without-observed-transition')
            active=np.asarray([status[pos]=='RUNNING' for pos in positions])
            if not active.any():break
            appended,inputs,eligible=_public_inputs(actor_record,result,active)
            ready=np.stack([np.asarray(private_requests.action_inputs(request.actions.facts.request,
                observer_side=session.state.observers[owner].owner).ready) for owner,request in enumerate(core.requests)],axis=1)
            for row in np.flatnonzero(active & np.any(ready & ~eligible,axis=1)):close(int(row),'REFUSED','current-owner-input-refused')
            active=np.asarray([status[pos]=='RUNNING' for pos in positions]);eligible &= active[:,None]
            legal=np.stack([np.asarray(x.encoded.observation.legal_action_mask) for x in result.owners],axis=1)
            choices,probabilities,values,failed=_decide(actor_record,inputs,legal,eligible,streams,
                np.asarray(core.frame_number),sampling_seed)
            for row in np.flatnonzero(active & failed):close(int(row),'REFUSED','raw-actor-nonfinite-or-no-legal-choice')
            routes=[]
            for owner in (0,1):
                route=actions.route_choice(core.requests[owner].actions,core.orders[owner],j.asarray(choices[:,owner]))
                jax.block_until_ready(route);routes.append(route)
                for row in np.flatnonzero(eligible[:,owner] & ~np.asarray(route.admitted)):
                    close(int(row),'REFUSED','actual-request-route-refused')
            active=np.asarray([status[pos]=='RUNNING' for pos in positions])
            if not active.any():break
            def capture_row(row):
                row_pending=[]
                publics={}; originals={}; mappings={}; addresses={}
                for owner in (0,1):
                    address=c.DecisionAddress(int(streams[row]),int(np.asarray(core.frame_number)[row]),owner)
                    addresses[owner]=address;mappings[owner]=_physical(core,int(row),owner)
                    if bool(np.asarray(appended[owner].admitted)[row]):
                        originals[owner],publics[owner]=data.public_observation(result.owners[owner].encoded,appended[owner],
                            row=int(row),owner=owner,address=address,adapter=actor.adapter,
                            public_species=np.asarray(result.state.observers[owner].retained.beliefs.species)[row,owner])
                for owner in (0,1):
                    if not eligible[row,owner]:continue
                    cell=ledger.cells[positions[row]];physical=mappings[owner]
                    hidden=tuple(None if p<0 else truth[row][1-owner][p] for p in physical[6:])
                    semantic=_semantic(core,routes,choices,int(row),1-owner,r)
                    if semantic is not None:
                        semantic=v5_data.semantic_opponent_action(addresses[owner],addresses[1-owner],kind=semantic.kind,identity=semantic.identity)
                        try:semantic.index(actor.config.label_vocabulary)
                        except ValueError:semantic=None
                    capture=dict(stream=int(streams[row]),frame=addresses[owner].frame,owner=owner,
                        turn=int(np.asarray(core.engine.turn)[row]),pool_sha256=receipt['pool_sha256'],
                        team_indices=cell.team_indices,physical=physical,
                        public_inputs=publics[owner].model_inputs(),request_binding=np.asarray(core.requests[owner].actions.facts.catalog_binding)[row].tolist(),
                        engine_action=int(np.asarray(routes[owner].engine_action)[row]),choice=int(choices[row,owner]),
                        behavior_probability=float(probabilities[row,owner]),value_estimate=float(values[row,owner]),
                        actor_archive_sha256=json.loads(actor.metadata)['weights_archive_sha256'])
                    source_sha=data.sha(capture);privileged=None
                    if include_privileged and len(set(physical[6:]))==6 and -1 not in physical[6:] and 1-owner in originals:
                        peer=mappings[1-owner][:6]
                        if len(set(peer))==6 and -1 not in peer:
                            privileged=c.privileged_view(publics[owner],originals[1-owner],peer_address=addresses[1-owner],
                                opponent_slot_map=tuple(peer.index(p) for p in physical[6:]),
                                peer_category_vocabulary_sha256=actor.config.category_vocabulary_sha256)
                    row_pending.append((int(row),data.capture_decision(originals[owner],publics[owner],battle_id=cell.battle_id,
                        seed=cell.seed,turn=capture['turn'],choice=int(choices[row,owner]),probability=float(probabilities[row,owner]),
                        value=float(values[row,owner]),physical=physical,hidden_sets=hidden,source_frame_sha256=source_sha,
                        capture_metadata={k:v for k,v in capture.items() if k!='public_inputs'},opponent_action=semantic,privileged=privileged)))
                return row_pending
            pending=[]
            for row in np.flatnonzero(active):
                try:pending.extend(capture_row(int(row)))
                except ValueError as error:close(int(row),'REFUSED','source-row-capture: '+str(error))
            active=np.asarray([status[pos]=='RUNNING' for pos in positions])
            if not active.any():break
            transport=j.stack([j.where(route.admitted,route.engine_action,0) for route in routes],axis=1)
            before=np.asarray(core.frame_number).copy()
            driver.advance(session,transport,active_rows=j.asarray(active));jax.block_until_ready(session.result)
            diagnostics=driver.latest_diagnostics(session);jax.block_until_ready(diagnostics)
            executed=np.asarray(diagnostics.executed_rows)
            if (executed.shape!=(b,) or executed.dtype!=np.bool_ or np.any(executed & ~active)
                    or not np.array_equal(np.asarray(diagnostics.active_rows),active)
                    or not np.array_equal(np.asarray(session.state.core.frame_number)-before,executed.astype(np.int32))):
                raise ValueError('actual driver executed a different frame/row set')
            invalid=np.asarray(diagnostics.invalid_actions)
            for row,captured in pending:
                if executed[row] and not invalid[row]:
                    decisions.append(captured);detail[positions[row]]['decisions']+=1
                    detail[positions[row]]['unknown_semantic_labels']=detail[positions[row]].get('unknown_semantic_labels',0)+int(captured.row.labels.opponent_next_action is None)
                    detail[positions[row]]['unknown_hidden_labels']=detail[positions[row]].get('unknown_hidden_labels',0)+sum(label is None for label in captured.row.labels.hidden_sets)
            public_ok=np.logical_and.reduce([(np.asarray(report.basic_error)==0)&(np.asarray(report.retained_error)==0)
                & np.asarray(observer.retained_healthy) for report,observer in zip(session.result.owners,session.state.observers)])
            for row in np.flatnonzero(active):
                detail[positions[row]]['executed_frames']+=int(executed[row])
                if not executed[row] or invalid[row] or not bool(np.asarray(session.state.core.admitted)[row]) or not public_ok[row]:
                    close(int(row),'REFUSED','actual-engine-or-public-frame-refused')
                elif bool(np.asarray(session.state.core.engine.done)[row]):close(int(row),'COMPLETE','observed-admitted-terminal')
        for row,pos in enumerate(positions):
            if status[pos]=='RUNNING':close(row,'CAPPED','finite-request-cap')
    except Exception as error:
        for row,pos in enumerate(positions):
            if status[pos]=='RUNNING':close(row,'FAILED',type(error).__name__+': '+str(error))
    closed=tuple(terminals[int(stream)] for stream in streams)
    return Collection(data.finish(decisions,closed),closed,ledger.report())


def _as_tuple(value):
    return tuple(_as_tuple(x) for x in value) if isinstance(value,(tuple,list)) else value
