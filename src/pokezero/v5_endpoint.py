"""Explicit final-arm endpoint, separate from the unchanged 500k milestones.

Four original arm slots share 1,000 declared seed blocks, both seats. The
explicit v2 contract holds D after failed cost admission without requiring its
checkpoint or dispatching its cells. All 8,000 cells remain in the denominator.
This module
executes nothing on import and never generates/replaces a bank or trains a model.
"""
import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from types import MappingProxyType, SimpleNamespace
import uuid

from . import v5_evaluation as evaluation, v5_foulplay as fp, v5_run as run

SCHEMA='pokezero.v5.endpoint.v1'
HELD_SCHEMA='pokezero.v5.endpoint.v2'
HELD_D={'D0':'D_T_NOT_ADMITTED'}
ARMS=('A0','B0','D0','B1')
CONTRASTS=(('B0-A0','B0','A0'),('D0-B0','D0','B0'),('B1-B0','B1','B0'))
DISPOSITIONS=('NOT_ATTEMPTED','RUNNING','COMPLETE','CAPPED','REFUSED','FAILED','INTERRUPTED','HELD')
METHOD='PAIRED_SEED_BLOCK_PERCENTILE_MISSING_CELL_ENVELOPE_90'
TRAINING_GAMES=2_000_000
CORE_SCHEDULE=dict(rule='FIXED_ARM_INDEX_PLUS_SEED_OFFSET_MOD_WORKERS',
    reason='PRE_OUTCOME_ROTATION_AVOIDS_PERSISTENT_ARM_CORE_CONFOUNDING')


def _keys(value,names):evaluation._keys(value,names)
def _ref(value):return evaluation.Artifact(**value)


def _held(protocol):
    if protocol['schema']==SCHEMA:
        if 'held_arms' in protocol:raise ValueError('held arms require explicit v2 endpoint contract')
        return {}
    if protocol['schema']!=HELD_SCHEMA or protocol.get('held_arms')!=HELD_D:
        raise ValueError('v2 endpoint holds only D0 after failed cost admission')
    return HELD_D


def _active(protocol):return tuple(arm for arm in ARMS if arm not in _held(protocol))


def _next_ordinal(rows):
    return next((row['ordinal'] for row in rows if row['status']=='NOT_ATTEMPTED'),len(rows))


def _sync(directory):
    fd=os.open(directory,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _publish(path,value):
    run._publish(path,run._json(value).encode());_sync(path.parent)


@dataclass(frozen=True)
class Configs:
    by_arm: object
    models: object
    declared_games: int

    def __getitem__(self,arm):return self.by_arm[arm]


def _final_publication(arm,declaration,snapshot):
    """Three original publication receipts, not an exhaustive ledger audit."""
    _keys(arm['final_events'],('checkpoint','snapshot','iteration_complete'))
    events=[];directory=Path(arm['run_directory'])
    for key,kind in (('checkpoint','CHECKPOINT'),('snapshot','SNAPSHOT'),('iteration_complete','ITERATION_COMPLETE')):
        reference=_ref(arm['final_events'][key]);event=reference.document();digest=event.pop('sha256',None)
        _keys(event,('sequence','previous','time','kind','payload'))
        if (run._sha(event)!=digest or event['kind']!=kind or type(event['sequence']) is not int or event['sequence']<0
                or type(event['time']) not in (int,float) or not math.isfinite(event['time'])
                or Path(reference.path).resolve()!=(directory/'journal'/f"{event['sequence']:09d}.json").resolve()):
            raise ValueError('original typed final publication event differs')
        if event['previous'] is not None:run.c._binding(event['previous'])
        event['sha256']=digest;events.append(event)
    if not events[0]['sequence']<events[1]['sequence']<events[2]['sequence']:
        raise ValueError('original checkpoint/snapshot/completion order required')
    for before,after in zip(events,events[1:]):
        if after['sequence']==before['sequence']+1 and after['previous']!=before['sha256']:
            raise ValueError('adjacent original publication link differs')
    checkpoint,actual,complete=(e['payload'] for e in events)
    _keys(checkpoint,('iteration','path','sha256','steps','rows'));_keys(actual,run.Snapshot.__dataclass_fields__)
    _keys(complete,('iteration','snapshot','counts'))
    if (checkpoint['iteration']!=snapshot.iteration or checkpoint['path']!=snapshot.checkpoint
            or checkpoint['sha256']!=snapshot.checkpoint_sha256 or actual!=asdict(snapshot)
            or complete['iteration']!=snapshot.iteration or complete['snapshot']!=asdict(snapshot)
            or any(type(checkpoint[k]) is not int or checkpoint[k]<0 for k in ('steps','rows'))):
        raise ValueError('exact original final checkpoint/snapshot publication required')
    counts=complete['counts'];_keys(counts,('NOT_ATTEMPTED','RUNNING','COMPLETE','CAPPED','REFUSED','FAILED'))
    if (any(type(n) is not int or n<0 for n in counts.values()) or sum(counts.values())!=TRAINING_GAMES
            or counts['NOT_ATTEMPTED'] or counts['RUNNING']):
        raise ValueError('literal full terminal2M training declaration counts required')


def _protocol(reference,*,parse_configs=False):
    if type(reference) is not evaluation.Artifact:raise ValueError('original digest-bound endpoint Artifact required')
    value=reference.document()
    _keys(value,('schema','seeds','arms','worker_python','foulplay_python','foulplay_commit','foulplay_source',
        'actor_cpus','foulplay_cpus','watchdog','bootstrap','deterministic','core_schedule')
        + (('held_arms',) if value.get('schema')==HELD_SCHEMA else ()))
    active=_active(value)
    if value['deterministic'] is not True:raise ValueError('explicit raw deterministic v5 endpoint required')
    if value['core_schedule']!=CORE_SCHEDULE:raise ValueError('fixed declared pre-outcome core rotation required')
    seeds=value['seeds']
    if (type(seeds) is not list or len(seeds)!=1000 or any(type(x) is not int or not 0<=x<2**32 for x in seeds)
            or seeds!=list(range(seeds[0],seeds[0]+1000))):
        raise ValueError('exact original 1,000 ordered seeds, both seats per arm required')
    if [a.get('name') for a in value['arms']]!=list(active):raise ValueError('all active final arms in fixed original order required')
    for key in ('worker_python','foulplay_python','foulplay_source'):_ref(value[key]).verify()
    commit=value['foulplay_commit']
    if type(commit) is not str or len(commit)!=40 or any(x not in '0123456789abcdef' for x in commit):raise ValueError('original FP source revision required')
    actor=value['actor_cpus'];cores=value['foulplay_cpus']
    if (type(actor) is not list or not actor or type(cores) is not list or not 1<=len(cores)<=4
            or any(type(x) is not int or x<0 for x in actor+cores) or len(set(actor+cores))!=len(actor+cores)):
        raise ValueError('disjoint actor CPUs and at most four dedicated FP cores required')
    run.Watchdog(**value['watchdog'])
    bootstrap=value['bootstrap'];_keys(bootstrap,('method','replicates','seed','confidence'))
    if (bootstrap['method']!=METHOD or type(bootstrap['replicates']) is not int or bootstrap['replicates']!=10000
            or type(bootstrap['seed']) is not int or not 0<=bootstrap['seed']<2**64 or bootstrap['confidence']!=.9):
        raise ValueError('prespecified 10,000-draw 90% paired seed-block method required')
    configs={};models={};namespaces=set();fp_seeds=None;memory=[];shared_config=None
    for arm in value['arms']:
        _keys(arm,('name','run_directory','declaration','final_events','snapshot','runtime','configs'))
        directory=Path(arm['run_directory'])
        if not directory.is_absolute():raise ValueError('original absolute final-run directory required')
        declaration=_ref(arm['declaration']).document();snapshot=run.Snapshot(**arm['snapshot'])
        if declaration.get('schema')!=run.SCHEMA:raise ValueError('original v5 final-run declaration schema required')
        if Path(arm['declaration']['path']).resolve()!=(directory/'declaration.json').resolve():raise ValueError('original accepted declaration path required')
        from .v5_model import V5ModelConfig
        model=declaration['model'];models[arm['name']]=V5ModelConfig.from_metadata(model)
        declared_games=sum(len(batch['games']) for iteration in declaration['iterations'] for batch in iteration)
        if declared_games!=TRAINING_GAMES:raise ValueError('exact fresh2M declared games per final arm required')
        if (arm['name']!=model['arm']+str(model['replicate']) or snapshot.iteration!=len(declaration['iterations'])-1
                or snapshot.model_sha256!=run._sha(model)):
            raise ValueError('exact final snapshot for the declared arm required')
        _final_publication(arm,declaration,snapshot)
        for name,digest in ((snapshot.checkpoint,snapshot.checkpoint_sha256),(snapshot.archive,snapshot.archive_sha256)):
            path=(directory/name).resolve()
            if not path.is_relative_to(directory.resolve()) or run._file_sha(path)!=digest:raise ValueError('original immutable endpoint snapshot bytes differ')
        runtime=_ref(arm['runtime']).document()
        from .v5_worker import SCHEMA as runtime_schema,_memory
        if runtime['schema']!=runtime_schema:raise ValueError('original bound worker runtime required')
        _memory(runtime['memory_policy'],apply=False)
        memory.append(runtime['memory_policy'])
        if runtime['precision_policy']['inference']!='bfloat16':raise ValueError('original bf16 endpoint inference required')
        vocabulary=_ref(runtime['category_vocabulary']).document()
        category=run._sha(dict(tokens=vocabulary['tokens'],aliases=vocabulary['aliases'],oov_buckets=vocabulary['oov_buckets']))
        if category!=model['category_vocabulary_sha256']:raise ValueError('endpoint category source differs')
        namespaces.add((category,model['label_vocabulary']['source_binding_sha256'],run._sha(model['label_vocabulary'])))
        if len(arm['configs'])!=2:raise ValueError('literal both-seat endpoint configurations required')
        actual=[]
        for seat,document in zip(('p1','p2'),arm['configs'],strict=True):
            from .foulplay_bridge import ControlledFoulPlayConfig,_per_seed_foulplay_random_seed_schedule_for_offsets
            from dataclasses import fields
            _keys(document,(f.name for f in fields(ControlledFoulPlayConfig)))
            config=SimpleNamespace(**document)
            common={k:v for k,v in document.items() if k not in ('checkpoint','pokezero_player')}
            if shared_config is None:shared_config=common
            elif common!=shared_config:raise ValueError('same prespecified FP configuration across arms/seats required')
            if (config.pokezero_player!=seat or config.games!=1000 or config.seed_start!=seeds[0]
                    or Path(config.checkpoint).resolve()!=(directory/snapshot.checkpoint).resolve()
                    or config.search_time_ms!=1000 or config.owning_decision_timeout_seconds!=60.
                    or config.foulplay_mcts_iterations is not None or config.policy_mode!='raw'
                    or config.opponent_policy_mode!='foul-play' or config.encoder_kind!='jax-owning'
                    or config.allow_search_fallback or config.belief_set_source is not True or config.opponent_crash_retries!=0
                    or config.live_continuation_smoke or config.live_continuation_oracle or config.belief_start_overrides
                    or config.value_checkpoint is not None or config.foulplay_cpu_affinity is not None
                    or Path(config.showdown_root)!=Path(runtime['showdown']['root'])
                    or Path(config.node_binary)!=Path(runtime['showdown']['node']['path'])
                    or config.foulplay_python is None or Path(config.foulplay_python)!=Path(value['foulplay_python']['path'])):
                raise ValueError('original FP1000ms/raw owning60s/no fallback endpoint config required')
            # The original helper owns FP-seed arithmetic; no endpoint-specific RNG.
            schedule=_per_seed_foulplay_random_seed_schedule_for_offsets(config,offsets=tuple(range(1000)))
            if fp_seeds is None:fp_seeds=schedule
            elif schedule!=fp_seeds:raise ValueError('same paired original FP seed bank required across arms/seats')
            actual.append(evaluation._config(document) if parse_configs else config)
        configs[arm['name']]=tuple(actual)
    if len(namespaces)!=1:raise ValueError('same original category/label source across all active arms required')
    if any(p!=memory[0] for p in memory):raise ValueError('one explicit original worker memory policy required')
    return value,Configs(MappingProxyType(configs),MappingProxyType(models),TRAINING_GAMES)


def _rows(protocol,configs):
    from .foulplay_bridge import _per_seed_foulplay_random_seed_schedule_for_offsets
    rows=[];seeds=protocol['seeds'];schedule=_per_seed_foulplay_random_seed_schedule_for_offsets(configs['A0'][0],offsets=tuple(range(1000)))
    for offset,seed in enumerate(seeds):
        for seat_index,seat in enumerate(('p1','p2')):
            for arm_index,arm in enumerate(ARMS):
                ordinal=len(rows);slot=(arm_index+offset)%len(protocol['foulplay_cpus'])
                rows.append(dict(ordinal=ordinal,arm=arm,offset=offset,seed=seed,seat=seat,fp_seed=schedule[offset],
                    fp_core=protocol['foulplay_cpus'][slot],slot=slot,
                    status='HELD' if arm in _held(protocol) else 'NOT_ATTEMPTED',result=None,invocation_id=None))
    return rows


def _config_for(row,configs):return replace(configs[row['arm']][row['seat']=='p2'],foulplay_cpu_affinity=row['fp_core'])


def _checked_result(row,result,configs,protocol):
    if row['arm'] in _held(protocol):raise ValueError('held endpoint cells cannot have source results')
    _keys(result,('row','recipe','namespace','invocation_id'))
    actual=result['row'];recipe=result['recipe'];config=configs[row['arm']][row['seat']=='p2']
    _keys(actual,('offset','seed','fp_seed','status','score','duration_seconds','reason','source_result_sha256','source_game','invocation_id')
        + (('worker_compilation',) if 'worker_compilation' in actual else ()))
    for key in ('offset','seed','fp_seed'):
        if actual.get(key)!=row[key]:raise ValueError('original endpoint game/seed pairing differs')
    if actual.get('status') not in ('COMPLETE','CAPPED','REFUSED','FAILED'):raise ValueError('actual terminal endpoint disposition required')
    duration=actual.get('duration_seconds')
    if type(duration) not in (int,float) or not math.isfinite(duration) or duration<0:raise ValueError('actual completed/refused cell duration required')
    original=next(a for a in protocol['arms'] if a['name']==row['arm']);snapshot=original['snapshot']
    expected=json.loads(json.dumps(asdict(config),default=fp._json_default,allow_nan=False)) if not isinstance(config,SimpleNamespace) else dict(vars(config))
    expected['foulplay_cpu_affinity']=row['fp_core']
    from .live_protocol_actor import ActorContract
    contract=replace(ActorContract.champion(),checkpoint_sha256=snapshot['checkpoint_sha256'],model_config_sha256=snapshot['model_sha256'])
    model=configs.models[row['arm']]
    if (recipe.get('schema')!=fp.SCHEMA or recipe.get('actor_version')!='v5' or recipe.get('config')!=expected
            or recipe.get('game_offsets')!=[row['offset']] or recipe.get('archive_sha256')!=snapshot['archive_sha256']
            or recipe.get('checkpoint_sha256')!=snapshot['checkpoint_sha256'] or recipe.get('learner_seconds')!=60.
            or recipe.get('actor_namespace')!=contract.namespace
            or recipe.get('category_vocabulary_sha256')!=model.category_vocabulary_sha256
            or recipe.get('label_vocabulary_sha256')!=model.label_vocabulary.sha256
            or recipe.get('deterministic') is not True or recipe.get('inference_precision')!='bfloat16'
            or result['namespace']!=fp._sha(recipe) or type(result['invocation_id']) is not str or not result['invocation_id']
            or actual['invocation_id']!=result['invocation_id']):
        raise ValueError('original endpoint panel/frozen actor binding differs')
    source=actual.get('source_game');digest=actual.get('source_result_sha256')
    if source is not None and (run._sha(source)!=digest or source.get('seed')!=row['seed']):raise ValueError('literal endpoint source game changed')
    if source is None and digest is not None:raise ValueError('source hash without literal game is not evidence')
    if actual['status']=='COMPLETE':
        if source is None or type(actual.get('score')) not in (int,float) or actual['score']!=fp._score(config,source):
            raise ValueError('complete endpoint score requires original literal terminal')
    elif actual.get('score') is not None:raise ValueError('unknown endpoint cannot supply a zero/tie score')
    return actual


@contextmanager
def _ownership(directory):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'endpoint.lock').open('ab') as handle:
        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('endpoint already has an owner') from error
        try:yield
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


class Store:
    """Original fixed order and immutable per-cell attempt/result events."""
    def __init__(self,directory,reference,protocol,configs,*,resume=False):
        self.path=Path(directory);self.reference=reference;self.protocol=protocol;self.configs=configs
        self.rows=_rows(protocol,configs);self.events=[];self.path.mkdir(parents=True,exist_ok=True)
        declaration=dict(schema=protocol['schema'],recipe=asdict(reference))
        path=self.path/'declaration.json'
        if path.exists():
            if not resume or json.loads(path.read_bytes())!=declaration:raise ValueError('same original endpoint declaration and explicit resume required')
        elif resume:raise ValueError('original endpoint declaration missing')
        else:
            if any(p.name!='endpoint.lock' for p in self.path.iterdir()):raise ValueError('fresh owned endpoint directory required')
            _publish(path,declaration)
        self.refresh()

    def refresh(self,*,full=True):
        # Owned writers append immutable files. Poll/append reads only the tail;
        # resume and final publication recheck the WHOLE original chain.
        events=[] if full else list(self.events)
        previous=events[-1]['sha256'] if events else None
        rows=_rows(self.protocol,self.configs) if full else self.rows
        next_cell=_next_ordinal(rows)
        paths=sorted((self.path/'journal').glob('*.json'))
        if len(paths)<len(events):raise ValueError('observed endpoint source disappeared')
        for index in range(len(events),len(paths)):
            path=paths[index]
            event=json.loads(path.read_bytes());digest=event.pop('sha256',None)
            _keys(event,('sequence','previous','recipe_sha256','kind','ordinal','invocation_id','identity','result'))
            if (path.name!=f'{index:09d}.json' or event['sequence']!=index or event['previous']!=previous
                    or event['recipe_sha256']!=self.reference.sha256 or run._sha(event)!=digest):
                raise ValueError('original endpoint journal changed')
            ordinal=event['ordinal']
            if type(ordinal) is not int or not 0<=ordinal<len(rows):raise ValueError('original endpoint ordinal differs')
            row=rows[ordinal]
            if event['identity']!={k:row[k] for k in ('ordinal','arm','offset','seed','seat','fp_seed','fp_core','slot')}:
                raise ValueError('endpoint source/cell order changed')
            if event['kind']=='START':
                if ordinal!=next_cell:raise ValueError('only next original not-attempted cell may start')
                if row['status']!='NOT_ATTEMPTED':raise ValueError('held or closed endpoint cell cannot start')
                row.update(status='RUNNING',invocation_id=event['invocation_id']);next_cell=_next_ordinal(rows)
            elif event['kind']=='END':
                if row['status']!='RUNNING' or row['invocation_id']!=event['invocation_id']:raise ValueError('missing/replayed endpoint attempt')
                actual=_checked_result(row,event['result'],self.configs,self.protocol)
                row.update(status=actual['status'],result=event['result'])
            elif event['kind']=='INTERRUPTED':
                if row['status']!='RUNNING' or event['invocation_id']==row['invocation_id'] or event['result'] is not None:
                    raise ValueError('only an older inflight invocation may be retained as unknown')
                row.update(status='INTERRUPTED')
            else:raise ValueError('unknown endpoint event')
            event['sha256']=digest;events.append(event);previous=digest
        if self.events and events[:len(self.events)]!=self.events:raise ValueError('observed endpoint result disappeared or mutated')
        self.rows,self.events=rows,events

    def emit(self,kind,ordinal,invocation,*,result=None):
        if type(ordinal) is not int or not 0<=ordinal<len(self.rows):raise ValueError('original endpoint ordinal required')
        row=self.rows[ordinal]
        if type(invocation) is not str or not invocation:raise ValueError('explicit owned endpoint invocation required')
        if kind=='START' and (ordinal!=_next_ordinal(self.rows) or row['status']!='NOT_ATTEMPTED' or result is not None):
            raise ValueError('only next original not-attempted cell may dispatch')
        if kind=='END':
            if row['status']!='RUNNING' or row['invocation_id']!=invocation:raise ValueError('original open endpoint attempt required')
            _checked_result(row,result,self.configs,self.protocol)
        elif kind=='INTERRUPTED':
            if row['status']!='RUNNING' or row['invocation_id']==invocation or result is not None:raise ValueError('only older inflight cells may be closed unknown')
        elif kind!='START':raise ValueError('unknown endpoint event')
        event=dict(sequence=len(self.events),previous=self.events[-1]['sha256'] if self.events else None,
            recipe_sha256=self.reference.sha256,kind=kind,ordinal=ordinal,invocation_id=invocation,
            identity={k:row[k] for k in ('ordinal','arm','offset','seed','seat','fp_seed','fp_core','slot')},result=result)
        event['sha256']=run._sha(event);_publish(self.path/'journal'/f'{len(self.events):09d}.json',event);self.refresh(full=False)

    def interrupt_older(self,invocation):
        for row in tuple(self.rows):
            if row['status']=='RUNNING':self.emit('INTERRUPTED',row['ordinal'],invocation)

    def report(self,invocation):
        return dict(schema=self.protocol['schema'],recipe=asdict(self.reference),rows=self.rows,invocation_id=invocation,
            held_arms=dict(_held(self.protocol)),
            counts={s:sum(r['status']==s for r in self.rows) for s in DISPOSITIONS},scientific_qualification=False)


async def _execute_cells(store,factories,*,invocation):
    """Fixed original runner, no arbitrary executor callback or scalar fallback."""
    if set(factories)!=set(_active(store.protocol)):raise ValueError('all and only active endpoint factories required')
    for arm in _active(store.protocol):
        first=next(r for r in store.rows if r['arm']==arm)
        fp._recipe(_config_for(first,store.configs),factories[arm]) # Original typed/runtime guard, before dispatch.
    active={};stop=False
    async def cell(index):
        nonlocal stop
        row=store.rows[index];config=_config_for(row,store.configs);ended=False
        def updated(report):
            nonlocal ended
            actual=report['rows'][0]
            if actual['status'] in ('NOT_ATTEMPTED','RUNNING'):return
            result=dict(row=actual,recipe=report['recipe'],namespace=report['namespace'],invocation_id=report['invocation_id'])
            if ended:
                if result!=store.rows[index]['result']:raise ValueError('actual completed endpoint source mutated')
                return
            store.emit('END',index,invocation,result=result);ended=True
        report=await fp.run_panel(config,factories[row['arm']],game_offsets=(row['offset'],),progress_callback=updated,enable_v5=True)
        updated(report)
        if not ended:raise ValueError('original endpoint runner returned without source disposition')
        if str(store.rows[index]['result']['row'].get('reason','')).startswith('TimeoutError:'):stop=True
    try:
        while True:
            next_row=next((r for r in store.rows if r['status']=='NOT_ATTEMPTED'),None)
            while next_row is not None and not stop and next_row['slot'] not in active:
                index=next_row['ordinal'];store.emit('START',index,invocation)
                active[next_row['slot']]=asyncio.create_task(cell(index))
                next_row=next((r for r in store.rows if r['status']=='NOT_ATTEMPTED'),None)
            if not active:break
            done,_=await asyncio.wait(tuple(active.values()),return_when=asyncio.FIRST_COMPLETED)
            for slot,task in list(active.items()):
                if task in done:del active[slot];task.result()
    finally:
        for task in active.values():task.cancel()
        if active:
            closed=await asyncio.gather(*active.values(),return_exceptions=True)
            failed=[value for value in closed if isinstance(value,BaseException) and not isinstance(value,asyncio.CancelledError)]
            if failed:raise RuntimeError('original endpoint cell cleanup failed') from failed[0]
    store.refresh();return store.report(invocation)


async def execute_cells(store,factories,*,invocation):
    """Own cancellation/reaping, restoring only the signal entry we installed."""
    loop=asyncio.get_running_loop();prior=signal.getsignal(signal.SIGTERM)
    registry=getattr(loop,'_signal_handlers',{})
    previous=registry.get(signal.SIGTERM)
    callback=getattr(previous,'_callback',None);args=getattr(previous,'_args',())
    task=asyncio.create_task(_execute_cells(store,factories,invocation=invocation))
    installed=False;installed_handle=None;installed_python=None;cancelled=False
    def terminate():
        nonlocal cancelled
        if not cancelled:cancelled=True;task.cancel()
    try:
        loop.add_signal_handler(signal.SIGTERM,terminate);installed=True
        installed_handle=registry.get(signal.SIGTERM);installed_python=signal.getsignal(signal.SIGTERM)
        try:return await asyncio.shield(task)
        except asyncio.CancelledError:
            terminate()
            while not task.done():
                try:await asyncio.shield(task)
                except asyncio.CancelledError:pass
            if not task.cancelled():task.result() # A failed owned cleanup is never hidden by cancellation.
            raise
    finally:
        if not task.done():
            terminate()
            while not task.done():
                try:await asyncio.shield(task)
                except asyncio.CancelledError:pass
            if not task.cancelled():task.result()
        # A later parent handler owns its entry. Never overwrite it on return.
        if (installed and getattr(loop,'_signal_handlers',None) is registry
                and registry.get(signal.SIGTERM) is installed_handle
                and signal.getsignal(signal.SIGTERM) is installed_python):
            loop.remove_signal_handler(signal.SIGTERM)
            if callback is not None:
                loop.add_signal_handler(signal.SIGTERM,callback,*args)
                registry[signal.SIGTERM]=previous # Preserve its original Handle and contextvars context.
            else:signal.signal(signal.SIGTERM,prior)


def reduce_endpoint(protocol,rows,configs):
    """All 1,000 original seed blocks; caps/refusals are unknown, never ties."""
    expected=_rows(protocol,configs)
    if (protocol['bootstrap'].get('method')!=METHOD or protocol['bootstrap'].get('replicates')!=10000
            or type(protocol['bootstrap'].get('seed')) is not int or protocol['bootstrap'].get('confidence')!=.9
            or len(protocol['seeds'])!=1000):raise ValueError('original prespecified full-roster endpoint bootstrap required')
    if type(rows) is not list or len(rows)!=len(expected):raise ValueError('full original8,000cell endpoint denominator required')
    outcomes={a:{} for a in ARMS};arm_counts={a:{s:0 for s in DISPOSITIONS} for a in ARMS}
    for actual,target in zip(rows,expected,strict=True):
        if any(actual[k]!=target[k] for k in ('ordinal','arm','offset','seed','seat','fp_seed','fp_core','slot')):
            raise ValueError('original endpoint pairing/order differs')
        status=actual['status']
        if status not in DISPOSITIONS:raise ValueError('endpoint disposition differs')
        if target['status']=='HELD':
            if status!='HELD' or actual['result'] is not None or actual.get('invocation_id') is not None:
                raise ValueError('held endpoint cells must remain unattempted unknowns')
        elif status=='HELD':raise ValueError('an active endpoint cell cannot be relabeled held')
        arm_counts[actual['arm']][status]+=1;score=None
        if actual['result'] is not None:
            result=_checked_result(actual,actual['result'],configs,protocol)
            if result['status']!=status:raise ValueError('source disposition changed')
            if status=='COMPLETE':score=float(result['score'])
        elif status not in ('NOT_ATTEMPTED','RUNNING','INTERRUPTED','HELD'):raise ValueError('observed disposition requires actual original source result')
        outcomes[actual['arm']][(actual['offset'],actual['seat'])]=score
    from .capstone_analysis import _percentile,_stratified_bootstrap_means
    bootstrap=protocol['bootstrap'];contrasts={}
    for label,candidate,baseline in CONTRASTS:
        lower=[];upper=[];known_blocks=[];matched=[];unknown=0
        for offset in range(1000):
            lo=hi=0.;block=[]
            for seat in ('p1','p2'):
                a,b=outcomes[candidate][offset,seat],outcomes[baseline][offset,seat]
                lo+=(0. if a is None else a)-(1. if b is None else b)
                hi+=(1. if a is None else a)-(0. if b is None else b)
                unknown+=int(a is None)+int(b is None)
                if a is not None and b is not None:matched.append(a-b);block.append(a-b)
            lower.append(lo/2);upper.append(hi/2)
            if len(block)==2:known_blocks.append(sum(block)/2)
        def draws(values):
            if len(set(values))==1:return tuple([values[0]]*bootstrap['replicates'])
            groups=tuple(SimpleNamespace(band='ENDPOINT',delta=value) for value in values)
            return _stratified_bootstrap_means(groups,replicates=bootstrap['replicates'],seed=bootstrap['seed'])
        draw_lower,draw_upper=draws(lower),draws(upper)
        interval=[_percentile(tuple(draw_lower),.05),_percentile(tuple(draw_upper),.95)]
        bounds=[sum(lower)/1000,sum(upper)/1000]
        contrasts[label]=dict(candidate=candidate,baseline=baseline,primary=label!='B1-B0',declared_games_per_arm=2000,
            status='HELD_UNMEASURED' if candidate in _held(protocol) or baseline in _held(protocol) else 'ACTIVE_ARM_CONTRAST',
            declared_seed_blocks=1000,complete_seed_blocks=len(known_blocks),matched_complete_cells=len(matched),unknown_cells_across_arms=unknown,
            matched_complete_cell_mean=sum(matched)/len(matched) if matched else None,
            complete_seed_point=sum(known_blocks)/len(known_blocks) if known_blocks else None,
            fully_identified_point=bounds[0] if unknown==0 else None,full_roster_identification_bounds=bounds,
            paired_seed_bootstrap_90=interval,uncertainty_method=METHOD,
            supported_positive_primary=(label!='B1-B0' and candidate not in _held(protocol)
                and baseline not in _held(protocol) and interval[0]>0 and bounds[0]>0),
            equivalence_established=False)
    arm_results={}
    for arm in ARMS:
        known=[score for score in outcomes[arm].values() if score is not None];total=sum(known);unknown=2000-len(known)
        arm_results[arm]=dict(counts=arm_counts[arm],declared_cells=2000,known_completed=len(known),unknown_cells=unknown,
            completed_only_score=total/len(known) if known else None,
            full_roster_identification_bounds=[total/2000,(total+unknown)/2000])
    return dict(schema=protocol['schema'],declared_cells=8000,held_arms=dict(_held(protocol)),
        held_declared_cells=2000*len(_held(protocol)),attemptable_declared_cells=2000*len(_active(protocol)),
        arms=arm_results,contrasts=contrasts,bootstrap=dict(bootstrap,
        unit='ORIGINAL_SEED_WITH_BOTH_SEATS_TOGETHER',common_draws_across_contrasts=True),
        selection_rule='PRIMARY_CONSERVATIVE_LOWER90_GT_ZERO_AND_FULL_ROSTER_IDENTIFICATION_LOWER_GT_ZERO',
        interpretation='Missing-cell identification bounds are not confidence intervals; bootstrap uncertainty is approximate seed-block percentile inference, not equivalence.',
        B0_shared_without_pooling_B1=True,training_declared_games_per_arm=configs.declared_games,
        publication_binding_scope='THREE_FROZEN_FINAL_EVENTS_NOT_EXHAUSTIVE_TRAINING_LEDGER_AUDIT',
        confidence_scope='MARGINAL_PER_DECLARED_CONTRAST_NOT_SIMULTANEOUS',scientific_qualification=False)


def _factories(protocol,configs):
    from .v5_worker import _source,_probe
    factories={}
    evaluation._require_foulplay_source(protocol,tuple(config['foulplay_root'] for arm in protocol['arms'] for config in arm['configs']))
    for arm in protocol['arms']:
        runtime=_ref(arm['runtime']).document();_source(runtime);_probe(runtime)
        directory=Path(arm['run_directory']);snapshot=run.Snapshot(**arm['snapshot'])
        view=dict(runtime=arm['runtime'],label_vocabulary_sha256=configs.models[arm['name']].label_vocabulary.sha256,
            model_sha256=snapshot.model_sha256,deterministic=True)
        request=SimpleNamespace(snapshot=snapshot,checkpoint_path=(directory/snapshot.checkpoint).resolve(),archive_path=(directory/snapshot.archive).resolve())
        factories[arm['name']]=evaluation._factory(view,request)
    return factories


def child(reference):
    request=reference.document();_keys(request,('recipe','directory','resume','invocation'))
    if (type(request['resume']) is not bool or type(request['directory']) is not str or not Path(request['directory']).is_absolute()
            or type(request['invocation']) is not str or not request['invocation']):raise ValueError('explicit owned endpoint request required')
    recipe=_ref(request['recipe']);raw=recipe.document()
    from .v5_worker import _memory
    evaluation._pin_actor(raw)
    policies=[_ref(a['runtime']).document()['memory_policy'] for a in raw['arms']]
    if any(p!=policies[0] for p in policies):raise ValueError('one original owned-worker memory policy required')
    _memory(policies[0],apply=True)
    protocol,configs=_protocol(recipe,parse_configs=True);directory=Path(request['directory'])
    from .owning_worker_runtime import configure_compilation_cache
    configure_compilation_cache(directory/'jax-compilation-cache')
    with _ownership(directory):
        store=Store(directory,recipe,protocol,configs,resume=request['resume'])
        store.interrupt_older(request['invocation'])
        report=asyncio.run(execute_cells(store,_factories(protocol,configs),invocation=request['invocation']))
        report['analysis']=reduce_endpoint(protocol,report['rows'],configs)
        _publish(directory/('report-'+request['invocation']+'.json'),report)
        return 0 if all(row['status'] not in ('NOT_ATTEMPTED','RUNNING') for row in store.rows) else 1


def _run_endpoint(reference,directory,*,resume=False,enable_v5_endpoint=False):
    """Fixed owned child/supervisor. Actual source operation requires opt-in."""
    if enable_v5_endpoint is not True:raise ValueError('explicit endpoint enablement required before artifact access')
    protocol,configs=_protocol(reference,parse_configs=True);directory=Path(directory).resolve()
    evaluation._pin_actor(protocol) # Parent metadata/watchdog work also stays off dedicated FP cores.
    if type(resume) is not bool:raise ValueError('explicit original endpoint resume flag required')
    with _ownership(directory):store=Store(directory,reference,protocol,configs,resume=resume)
    invocation=uuid.uuid4().hex
    request=dict(recipe=asdict(reference),directory=str(directory),resume=True,invocation=invocation)
    path=directory/('request-'+invocation+'.json');_publish(path,request)
    proxy=directory/('verified-progress-'+invocation+'.json')
    def checked_poll(seconds):
        # Read only genuine, source-validated END events before the original
        # supervisor sees a completed cell. Bare heartbeat-shaped rows cannot reset it.
        store.refresh(full=False);rows=[]
        for row in store.rows:
            result=None if row['result'] is None else row['result']['row']
            rows.append(dict(status=row['status'],duration_seconds=None if result is None else result['duration_seconds'],
                reason=None if result is None else result.get('reason'),invocation_id=row['invocation_id']))
        evaluation._atomic(proxy,dict(schema=fp.SCHEMA,namespace=reference.sha256,invocation_id=invocation,rows=rows))
        time.sleep(seconds)
    policy=run.Watchdog(**protocol['watchdog'])
    result=fp.supervise_panel([protocol['worker_python']['path'],'-m','pokezero.v5_endpoint','--child',
        '--request',str(path),'--sha256',run._file_sha(path)],progress_path=proxy,
        initial_window_seconds=policy.initial_window_seconds,longest_multiplier=policy.longest_multiplier,
        backstop_seconds=policy.backstop_seconds,poll_seconds=policy.poll_seconds,grace_seconds=policy.terminate_grace_seconds,sleep=checked_poll)
    store.refresh();report=store.report(invocation);report.update(supervisor=result,analysis=reduce_endpoint(protocol,store.rows,configs))
    report['status']='RESULT' if result['exit_code']==0 and not result['stop_reason'] and result.get('cleanup_confirmed') is True and all(
        row['status'] not in ('NOT_ATTEMPTED','RUNNING') for row in store.rows) else 'UNKNOWN'
    final=directory/('endpoint-'+invocation+'.json');_publish(final,report)
    return dict(status=report['status'],report=str(final),sha256=run._file_sha(final),counts=report['counts'])


def run_endpoint(reference,directory,*,resume=False,enable_v5_endpoint=False):
    """The public synchronous owner also routes SIGTERM through group cleanup."""
    if enable_v5_endpoint is not True:raise ValueError('explicit endpoint enablement required before artifact access')
    previous=signal.getsignal(signal.SIGTERM);interrupted=False
    def terminate(signum,frame):
        nonlocal interrupted
        if not interrupted:
            interrupted=True;raise KeyboardInterrupt('owned endpoint invocation interrupted')
    signal.signal(signal.SIGTERM,terminate)
    try:return _run_endpoint(reference,directory,resume=resume,enable_v5_endpoint=True)
    finally:
        if signal.getsignal(signal.SIGTERM) is terminate:signal.signal(signal.SIGTERM,previous)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');parser.add_argument('--request')
    parser.add_argument('--recipe');parser.add_argument('--sha256',required=True);parser.add_argument('--directory')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--enable-v5-endpoint',action='store_true')
    args=parser.parse_args(argv)
    if args.child:return child(evaluation.Artifact(args.request,args.sha256))
    if not args.enable_v5_endpoint:raise ValueError('explicit endpoint enablement required')
    result=run_endpoint(evaluation.Artifact(args.recipe,args.sha256),args.directory,resume=args.resume,enable_v5_endpoint=True)
    print(run._json(result));return 0 if result['status']=='RESULT' else 1


if __name__=='__main__':raise SystemExit(main())
