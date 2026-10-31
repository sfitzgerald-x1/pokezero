"""Prospective ORIGINAL nontraining collector and public-sibling capture.

No collection is performed on import. The one public entry constructs the
declared original sessions, then reuses collect_paired/capture_decision/finish.
Sibling generation owns the original sampler/materializer/one-step calls; it
never accepts an arbitrary supplied Step or a returned outcome label. Missing
terminal continuations stay UNKNOWN. Source receipts are not external attestation.
"""
from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import weakref

import numpy as np

from pokezero import v5_contract as c, v5_data
from pokezero import v5_evaluation_capture as schema
from pokezero import v5_run as run


def _host(value):
    if hasattr(value,'_asdict'):return {k:_host(v) for k,v in value._asdict().items()}
    if isinstance(value,dict):return {k:_host(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_host(v) for v in value]
    if value is None or type(value) in (str,int,float,bool):return value
    return np.asarray(value).tolist()


def _state(value,row=0):
    # BattleState's ORIGINAL leaves all lead with B. Global namespace arrays
    # live on callback/context records and are copied whole by _host instead.
    return {name:np.asarray(leaf)[row:row+1].tolist() for name,leaf in value._asdict().items()}


def _frame(value):
    return dict(cursor=_host(value.cursor),callbacks=_host(value.callbacks),
        invalid_actions=_host(value.invalid_actions),placeholder_moves=_host(value.placeholder_moves))


@dataclass(frozen=True,eq=False)
class Recorder:
    bank:schema.CaptureBank


_RECORDERS=weakref.WeakKeyDictionary()


def _record(recorder):
    record=_RECORDERS.get(recorder) if type(recorder) is Recorder else None
    if (record is None or recorder.bank is not record['bank'] or schema.source_binding()!=record['source']
            or run._sha(recorder.bank.document)!=record['bank_sha256']):
        raise ValueError('original source-bound nontraining recorder required')
    return record


def _start(recorder,session,*,learner,actors,ledger,team_indices,sampling_seed):
    from . import v5_collection as original
    record=_record(recorder);current=record.get('current')
    if current is None or current['session'] is not session:raise ValueError('only the actual declared constructor opens a capture session')
    batch=current['batch'];actual=original.driver._session(session)
    if (session.result is not current['initial'] or int(np.asarray(session.state.core.frame_number).max())!=0
            or sampling_seed!=batch.sampling_seed or ledger is not current['ledger']
            or learner is not record['learner'] or actors!=current['actors']
            or not np.array_equal(team_indices,[g.team_indices for g in batch.games])):
        raise ValueError('original constructor/batch/actor/sampling identity differs')
    original._checked(learner,session)
    if actual['constructed'] is not current['constructed']:raise ValueError('original constructed snapshot changed')


def _unknown(address,world,reply,reason):
    return dict(schema=schema.SIBLING,selector=asdict(address),world=world,reply=reply,status='UNKNOWN',reason=reason,
        root_source_sha256=None,origin=None,chance=None,root_legal=None,branches=[])


def _siblings(recorder,captured,history,own_static):
    """Exactly two original draws; all original legal actions, no model calls."""
    from pokezero import v5_gumbel, v5_teacher_public_root as planner
    from . import v5_owning_history as histories, v5_public_world as worlds
    from . import v5_public_branch as branches, transition, v5_observation, v5_collection_data as data
    from .v5_depth1_provider import _same_root
    record=_record(recorder);address=captured.row.observation.address;bank=recorder.bank
    index=bank.selectors.index(address);replies=bank.document['opponent_replies'][index]
    result=[_unknown(address,w,replies[w],'original-root-unavailable') for w in (0,1)]
    try:
        if history is None or own_static is None:raise ValueError('missing-original-owning-history-or-static')
        if (history.stream_id,history.frames[-1].number,history.owner)!=(address.stream,address.frame,address.owner):
            raise ValueError('original history is not the current selected root')
        source=histories.public_root(history,config=record['public_config'],contract=record['contract'],
            battle_id=captured.row.battle_id,seed=captured.row.seed)
        if source.root is None:raise ValueError('original-public-root:'+str(source.decoded.refusal_counts))
        root=source.root;recipe=v5_gumbel.Recipe(**bank.document['world_recipe'])
        plans=planner.plan_starts(root,recipe,set_source=root.context.public_materialization_state.belief_engine.set_source)
        shared=worlds.public_carry(history,record['runtime'])
        history_body=dict(stream=history.stream_id,owner=history.owner,own_constructor=_host(history.own_constructor),
            frames=[asdict(f) for f in history.frames])
        history_sha=run._sha(history_body)
        for world,(start,reason) in enumerate(zip(plans.starts,plans.refusals,strict=True)):
            if start is None:
                result[world]=_unknown(address,world,replies[world],reason or 'original-posterior-draw-refused');continue
            seed=int.from_bytes(recipe._digest(address,'hypothetical-kernel',world)[:4],'big')
            materialized=worlds.materialize(root,start,prepared=record['prepared'],history=history,own_static=own_static,
                seed=seed,retained=shared)
            branch=materialized.branch
            if branch is None:
                result[world]=_unknown(address,world,replies[world],'original-materializer:'+','.join(materialized.refusals));continue
            try:
                root_view=branches.policy_observation(branch,record['learner'].adapter,owner=address.owner)
                _same_root(root_view,captured.row.observation)
                peer=branches.policy_observation(branch,record['learner'].adapter,owner=1-address.owner)
                if not bool(np.asarray(peer.legal_action_mask)[0,replies[world]]):raise ValueError('declared-reply-not-legal-in-sampled-world')
            except (ValueError,KeyError,IndexError) as error:
                result[world]=_unknown(address,world,replies[world],str(error));continue
            chance=transition.sample_chance_inputs(branch.state) # ONCE, shared across all original legal actions.
            chance_fields=_host(chance)
            origin_body=dict(scope='HYPOTHETICAL_PUBLIC_WORLD_NOT_ORDINARY_SESSION',root=root.owner_lines_sha256,
                engine=np.asarray(record['runtime']['request']['rules'].device.engine_binding).tolist(),
                sampled_teams=dict(start.player_teams),seed=seed)
            if run._sha(origin_body)!=branch.origin_sha256:raise ValueError('original sampled origin receipt differs')
            group=dict(schema=schema.SIBLING,selector=asdict(address),world=world,reply=replies[world],status='CAPTURED',reason=None,
                root_source_sha256=captured.row.source_frame_sha256,
                origin=dict(body=origin_body,sha256=branch.origin_sha256,history_sha256=history_sha,
                    history=history_body,owning_lines=list(source.decoded.lines),before_state=_state(branch.state),
                    root_view=_host(root_view.inputs),sampled_start=_host(asdict(start)),
                    posterior_seed=recipe.world_seed(address,world),kernel_seed=seed),
                chance=dict(fields=chance_fields,sha256=run._sha(chance_fields)),
                root_legal=list(captured.row.observation.legal_action_mask),branches=[])
            for action,legal in enumerate(captured.row.observation.legal_action_mask):
                if not legal:continue
                choices=[replies[world]]*2;choices[address.owner]=action
                actual=branches.step(branch,tuple(choices),chance_inputs=chance,hypothetical_profiles=True)
                child=dict(action=action,choices=choices,status='UNKNOWN',refusals=list(actual.refusals),
                    state=None if actual.state is None else _state(actual.state),raw_frame=None if actual.raw_frame is None else _frame(actual.raw_frame),
                    diagnostics=_host(actual.diagnostics),observation=None,terminal=None)
                if not actual.refusals:
                    try:
                        if actual.state is None or actual.raw_frame is None:raise ValueError('missing-original-returned-branch')
                        if bool(np.asarray(actual.state.done)[0]):
                            winner=int(np.asarray(actual.state.winner)[0])
                            if winner not in (0,1,2):raise ValueError('unrepresented-original-child-winner')
                            child['terminal']=dict(state_sha256=run._sha(child['state']),winner=winner)
                        else:
                            encoded=actual.encoded[address.owner];public=actual.public[address.owner]
                            appended=v5_observation.append_device(encoded,public,record['learner'].adapter,
                                observer_sides=np.array([address.owner],np.int32))
                            _,observation=data.public_observation(encoded,appended,row=0,owner=address.owner,
                                address=c.DecisionAddress(address.stream,address.frame+1,address.owner),adapter=record['learner'].adapter,
                                public_species=np.asarray(public.beliefs.species)[0,address.owner])
                            child['observation']=schema.observation_to_dict(observation)
                        child['status']='CAPTURED'
                    except (ValueError,KeyError,IndexError) as error:child['refusals']=[str(error)]
                if child['status']=='UNKNOWN' and not child['refusals']:child['refusals']=['missing-original-child-input']
                child['receipt_sha256']=run._sha(dict(selector=asdict(address),world=world,reply=replies[world],
                    origin=branch.origin_sha256,chance=group['chance']['sha256'],returned=child))
                group['branches'].append(child)
            result[world]=group
    except (ValueError,KeyError,IndexError) as error:
        result=[_unknown(address,w,replies[w],type(error).__name__+': '+str(error)) for w in (0,1)]
    return result


def _before(recorder,session,*,pending,histories,own_statics):
    from . import v5_collection as original
    record=_record(recorder);current=record['current'];original._checked(record['learner'],session)
    if session is not current['session']:raise ValueError('different original capture Session')
    if current.get('pending') is not None:raise ValueError('unconsumed capture frame ticket')
    selected=[];before=session.result
    for row,captured in pending:
        address=captured.row.observation.address
        if address not in recorder.bank.selectors:continue
        if address in record['positions']:raise ValueError('selected source frame replay')
        if (address.stream!=int(np.asarray(before.state.core.cursor.stream_id)[row])
                or address.frame!=int(np.asarray(before.state.core.frame_number)[row])):
            raise ValueError('selected source frame is not current original Session')
        # Exact metadata is produced by capture_decision at this callsite; the
        # private training view is stripped from the archived DecisionRow.
        captured=replace(captured,row=replace(captured.row,privileged=None))
        groups=_siblings(recorder,captured,histories.get((int(row),address.owner)),own_statics.get((int(row),address.owner)))
        original._checked(record['learner'],session)
        if session.result is not before:raise ValueError('sibling capture mutated original owning Session')
        selected.append((int(row),captured,groups))
    current['pending']=(before,tuple(selected))


def _after(recorder,session,*,diagnostics):
    from . import v5_collection as original
    record=_record(recorder);current=record['current'];original._checked(record['learner'],session)
    if session is not current['session'] or current.get('pending') is None:raise ValueError('original open same-call ticket required')
    before,selected=current.pop('pending');current['pending']=None
    if diagnostics is not original.driver.latest_diagnostics(session):raise ValueError('actual original driver diagnostic object required')
    if session.result is before:raise ValueError('original advance did not return a new frame')
    report=session.state.core.frame_input_report
    if report is None:raise ValueError('actual original source-witness report required')
    for row,captured,groups in selected:
        if not bool(np.asarray(diagnostics.executed_rows)[row]) or bool(np.asarray(diagnostics.invalid_actions)[row]):continue
        address=captured.row.observation.address
        if int(np.asarray(session.state.core.frame_number)[row])!=address.frame+1:raise ValueError('original frame delta differs')
        record['positions'][address]=dict(decision=v5_data.row_to_dict(captured.row),execution=dict(stream=address.stream,
            before_frame=address.frame,after_frame=address.frame+1,executed=True,invalid=False,
            source_frame_sha256=captured.row.source_frame_sha256,after_state=_state(session.state.core.engine,row),
            raw_frame=dict(row=row,source_witness_report=_host(report))))
        record['groups'][address]=groups


def _terminal(recorder,session,*,terminal,body):
    from . import v5_collection as original, v5_collection_data as data
    record=_record(recorder);original._checked(record['learner'],session)
    if session is not record['current']['session'] or type(terminal) is not data.TerminalCapture or data.sha(body)!=terminal.source_sha256:
        raise ValueError('actual original TerminalCapture/body required')
    if terminal.stream in record['terminals']:raise ValueError('terminal capture replay')
    core=session.state.core;row=next(i for i,s in enumerate(np.asarray(core.cursor.stream_id)) if int(s)==terminal.stream)
    if terminal.frame!=int(np.asarray(core.frame_number)[row]):raise ValueError('terminal is not current actual frame')
    if terminal.status=='COMPLETE':
        if (not bool(np.asarray(core.engine.done)[row]) or terminal.winner!=int(np.asarray(core.engine.winner)[row])
                or terminal.hp!=data._tuple(np.asarray(core.engine.hp)[row].tolist())
                or terminal.max_hp!=data._tuple(np.asarray(core.engine.max_hp)[row].tolist())):
            raise ValueError('literal actual returned terminal state differs')
    record['terminals'][terminal.stream]=dict(body=body,capture=asdict(terminal))


def _finish(recorder,session,*,decisions,terminals,finished):
    from . import v5_collection as original
    record=_record(recorder);original._checked(record['learner'],session)
    if (len(finished)!=len(decisions) or any(a.row.source_frame_sha256!=b.row.source_frame_sha256
            for a,b in zip(decisions,finished,strict=True))):
        raise ValueError('original finish lost/reordered a captured decision')
    if record['current']['session'] is not session or record['current'].get('pending') is not None:
        raise ValueError('unclosed original same-call frame ticket')
    record['current']=None


def _publish(recorder,directory):
    from pokezero import v5_evaluation as evaluation
    record=_record(recorder);directory=Path(directory)
    if record['current'] is not None:raise ValueError('original collection is still open')
    directory.mkdir(parents=True,exist_ok=True)
    header=dict(split='PHASE_C_FIXED_HELDOUT',category_vocabulary_sha256=record['learner'].config.category_vocabulary_sha256,
        label_vocabulary_sha256=record['learner'].config.label_vocabulary.sha256,
        capture_bank=asdict(recorder.bank.reference),capture_source=record['source'])
    positions=[];groups=[]
    for index,address in enumerate(recorder.bank.selectors):
        actual=record['positions'].get(address);game=recorder.bank.games[address.stream]
        positions.append(dict(schema=schema.POSITION,selector=asdict(address),game=asdict(game),
            status='CAPTURED' if actual is not None else 'UNKNOWN',decision=None if actual is None else actual['decision'],
            execution=None if actual is None else actual['execution'],terminal=record['terminals'].get(address.stream),
            reason=None if actual is not None else 'selected-original-decision-not-observed'))
        groups.extend(record['groups'].get(address,[_unknown(address,w,recorder.bank.document['opponent_replies'][index][w],
            'selected-original-decision-not-observed') for w in (0,1)]))
    references={}
    for key,schema_name,rows in (('positions',evaluation.POSITIONS,positions),('siblings',evaluation.SIBLINGS,groups)):
        path=directory/(key+'.jsonl');blob='\n'.join(run._json(v) for v in (dict(header,schema=schema_name,rows=len(rows)),*rows))+'\n'
        run._publish(path,blob.encode());references[key]=asdict(evaluation.Artifact(str(path.resolve()),run._file_sha(path)))
    return references


@contextmanager
def _ownership(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'capture.lock').open('ab') as handle:
        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('original capture directory already has an owner') from error
        try:yield
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def _claim(path,event):
    """Durable original attempt BEFORE constructor; interruption never replays."""
    with path.open('ab') as stream:
        stream.write((run._json(event)+'\n').encode());stream.flush();os.fsync(stream.fileno())


def _journal(directory,recorder,*,resume):
    record=_record(recorder);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    declaration=dict(schema=schema.CAPTURE,bank=asdict(recorder.bank.reference),source=record['source'])
    path=directory/'capture-declaration.json';journal=directory/'capture-journal.jsonl'
    if not path.exists():
        if resume or journal.exists():raise ValueError('original capture declaration missing')
        run._publish(path,run._json(declaration).encode())
    elif not resume:raise ValueError('existing capture requires explicit resume; no automatic retry')
    if json.loads(path.read_bytes())!=declaration:raise ValueError('original capture bank/source changed on resume')
    events=[] if not journal.exists() else [json.loads(line) for line in journal.read_text().splitlines()]
    started=[];completed=set()
    for event in events:
        index=event.get('batch')
        if type(index) is not int or not 0<=index<len(recorder.bank.batches):raise ValueError('capture journal batch differs')
        if event.get('kind')=='BATCH_STARTED':
            if index!=len(started):raise ValueError('capture journal order/replay differs')
            if event!=json.loads(run._json(dict(kind='BATCH_STARTED',batch=index,binding=run._sha(declaration),recipe=asdict(recorder.bank.batches[index])))):
                raise ValueError('capture attempt recipe/binding changed')
            started.append(index)
        elif event.get('kind')=='BATCH_END':
            if not started or index!=started[-1] or index in completed:raise ValueError('capture completion lacks original attempt')
            expected=directory/('batch-'+str(index)+'.json')
            if event.get('path')!=expected.name or run._file_sha(expected)!=event.get('sha256'):raise ValueError('original capture batch file differs')
            saved=json.loads(expected.read_bytes())
            if saved['binding']!=run._sha(declaration) or saved['batch']!=index:raise ValueError('saved capture batch identity differs')
            streams={g.stream_id for g in recorder.bank.batches[index].games}
            for position in saved['positions']:
                address=c.DecisionAddress(**position['selector'])
                if address in record['positions'] or address not in recorder.bank.selectors or address.stream not in streams:
                    raise ValueError('capture position replay/foreign batch')
                record['positions'][address]=position['value']
            for group in saved['groups']:
                address=c.DecisionAddress(**group['selector'])
                if address not in recorder.bank.selectors or address.stream not in streams or address in record['groups']:
                    raise ValueError('capture sibling replay/foreign batch')
                record['groups'][address]=group['value']
            for terminal in saved['terminals']:
                stream=terminal['capture']['stream']
                if stream in record['terminals'] or stream not in streams:raise ValueError('capture terminal replay/foreign batch')
                record['terminals'][stream]=terminal
            completed.add(index)
        else:raise ValueError('unknown capture journal event')
    # START without END remains an attempted UNKNOWN batch. No source state or
    # completed row is reconstructed; only the next never-attempted batch runs.
    return journal,declaration,len(started)


def _save_batch(directory,recorder,index,binding):
    record=_record(recorder);streams={g.stream_id for g in recorder.bank.batches[index].games}
    value=dict(binding=binding,batch=index,
        positions=[dict(selector=asdict(a),value=v) for a,v in record['positions'].items() if a.stream in streams],
        groups=[dict(selector=asdict(a),value=v) for a,v in record['groups'].items() if a.stream in streams],
        terminals=[v for stream,v in record['terminals'].items() if stream in streams])
    path=Path(directory)/('batch-'+str(index)+'.json');run._publish(path,run._json(value).encode())
    return dict(kind='BATCH_END',batch=index,path=path.name,sha256=run._file_sha(path))


def collect_declared_bank(bank_reference,*,prepared,pool,learner,opponent,public_config,contract,
        directory,max_requests,on_end,check_progress,gender_policy=None,teacher=None,resume=False):
    """Explicit future data operation, never called by the evaluator or import.

    Only original externally loaded actors/runtime/pool are accepted. No new
    seed, bank, retry or continuation is generated. This path is NONTRAINING;
    it returns capture artifacts and NEVER writes a training cache.
    """
    from . import v5_collection as original, v5_selfplay as paired, source_provenance
    from pokezero.live_protocol_actor import ActorContract,_mask_values,_vocab_sha
    from pokezero.local_showdown import LocalShowdownConfig
    bank=schema.load_bank(bank_reference);runtime=original.driver._checked(prepared)
    original.driver._checked_witness(runtime)
    if runtime['frame_witness_compiled'] is None:raise ValueError('actual original source-witness capture mode required')
    if type(max_requests) is not int or max_requests!=bank.document['max_requests'] or not callable(on_end) or not callable(check_progress):
        raise ValueError('explicit finite capture recipe/callbacks required')
    if (type(public_config) is not LocalShowdownConfig or type(contract) is not ActorContract
            or public_config.set_belief_source is not True or public_config.category_vocab is not runtime['catalog'].vocabulary
            or _vocab_sha(runtime['catalog'].vocabulary)!=contract.category_vocab_sha256
            or _mask_values(public_config.feature_masks)!=contract.feature_masks
            or contract.belief_set_source_hash!=runtime['catalog'].source.source_hash):
        raise ValueError('same original public parser/catalog/source required')
    actual_pool=source_provenance._pool(pool,rehash=True)
    heldout=schema._artifact(bank.document['heldout_roster']).document()
    if actual_pool['pool_sha256']!=heldout['pool_sha256']:raise ValueError('original heldout constructor pool differs')
    from .v5_depth1_provider import _actor
    _,archive=_actor(learner,prepared);_,peer_archive=_actor(opponent,prepared)
    if (learner.config!=opponent.config or archive!=bank.document['collector_archive_sha256']
            or peer_archive!=bank.document['opponent_archive_sha256']):
        raise ValueError('predeclared fixed collector-policy archives differ')
    recorder=Recorder(bank);_RECORDERS[recorder]=dict(bank=bank,bank_sha256=run._sha(bank.document),source=schema.source_binding(),prepared=prepared,runtime=runtime,
        learner=learner,public_config=public_config,contract=contract,current=None,positions={},groups={},terminals={})
    if type(resume) is not bool:raise ValueError('explicit capture resume flag required')
    with _ownership(directory):
        record=_record(recorder);journal,declaration,attempted=_journal(directory,recorder,resume=resume);offset=0
        for index,batch in enumerate(bank.batches):
            if index<attempted:offset+=len(batch.games);continue
            check_progress();indices=np.asarray([g.team_indices for g in batch.games],np.int32)
            streams=np.asarray([g.stream_id for g in batch.games],np.int32);seats=bank.document['collector_seats'][offset:offset+len(batch.games)]
            actors=tuple((learner,opponent) if seat==0 else (opponent,learner) for seat in seats)
            ledger=original.declare_cells(tuple(original.Cell(g.stream_id,g.battle_id,g.seed,g.team_indices) for g in batch.games))
            gender=None
            if batch.gender_seed is not None:
                if gender_policy is None:raise ValueError('original declared gender policy required')
                gender=dict(policy=gender_policy,seed=batch.gender_seed,domain=batch.gender_domain,root_ids=batch.gender_root_ids)
            _claim(journal,dict(kind='BATCH_STARTED',batch=index,binding=run._sha(declaration),recipe=asdict(batch)))
            session=original.driver.construct(pool,indices,prepared=prepared,stream_ids=streams,
                seed=batch.constructor_seed,gender_initialization=gender)
            original._checked(learner,session);original._checked(opponent,session)
            current=original.driver._session(session)
            record['current']=dict(session=session,batch=batch,initial=session.result,constructed=current['constructed'],ledger=ledger,actors=actors,pending=None)
            paired.collect_paired(session,learner=learner,actors=actors,pool=pool,team_indices=indices,ledger=ledger,
                max_requests=max_requests,sampling_seed=batch.sampling_seed,on_end=on_end,check_progress=check_progress,
                teacher=teacher,evaluation_recorder=recorder)
            _claim(journal,_save_batch(directory,recorder,index,run._sha(declaration)))
            offset+=len(batch.games)
        return _publish(recorder,directory)
