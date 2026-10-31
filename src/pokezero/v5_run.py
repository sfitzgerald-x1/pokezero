"""Explicit finite v5 iterations and immutable historical snapshots.

The caller supplies a predeclared game/seed bank and an engine backend. This
module never draws teams, allocates a job, loads a legacy warm start, evaluates
a holdout, or promotes an arm. Real execution needs Torch and the separately
prepared original engine/JAX backend. Source readiness is not qualification.
"""
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from numbers import Real
import os
import signal
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from . import v5_contract as c, v5_training as training
from .v5_model import V5ModelConfig
from .v5_loss import V5LossConfig


SCHEMA = 'pokezero.finite_run.v5'


class MissingDependency(RuntimeError):pass


def _json_default(value):
    if is_dataclass(value):return asdict(value)
    raise TypeError('typed run JSON required')


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,default=_json_default)
def _sha(value):return hashlib.sha256(_json(value).encode()).hexdigest()
def _file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seed(value):
    if type(value) is not int or not 0<=value<2**32:raise ValueError('explicit uint32 seed required')


@dataclass(frozen=True)
class Game:
    """Cell/trajectory identity seed; Batch owns actual constructor/action RNG."""
    stream_id:int
    battle_id:str
    seed:int
    team_indices:tuple[int,int]

    def __post_init__(self):
        object.__setattr__(self,'team_indices',tuple(self.team_indices));_seed(self.seed)
        if (type(self.stream_id) is not int or not 0<self.stream_id<2**31 or type(self.battle_id) is not str
                or not self.battle_id or len(self.team_indices)!=2 or len(set(self.team_indices))!=2
                or any(type(x) is not int or x<0 for x in self.team_indices)):
            raise ValueError('fixed original game identity/team rows required')


@dataclass(frozen=True)
class Batch:
    games:tuple[Game,...]
    constructor_seed:int
    sampling_seed:int
    gender_seed:int|None=None
    gender_domain:int|None=None
    gender_root_ids:tuple[int,...]=()

    def __post_init__(self):
        object.__setattr__(self,'games',tuple(self.games));object.__setattr__(self,'gender_root_ids',tuple(self.gender_root_ids))
        _seed(self.constructor_seed);_seed(self.sampling_seed)
        if not self.games or any(type(g) is not Game for g in self.games):raise ValueError('finite nonempty declared batch required')
        if self.gender_seed is not None:
            _seed(self.gender_seed);_seed(self.gender_domain)
            if len(self.gender_root_ids)!=len(self.games):raise ValueError('literal per-game gender root IDs required')
            for value in self.gender_root_ids:_seed(value)
        elif self.gender_domain is not None or self.gender_root_ids:raise ValueError('complete explicit gender recipe required')


@dataclass(frozen=True)
class Snapshot:
    iteration:int
    checkpoint:str
    checkpoint_sha256:str
    archive:str
    archive_sha256:str
    model_sha256:str

    def __post_init__(self):
        if (type(self.iteration) is not int or self.iteration < -1 or type(self.checkpoint) is not str
                or not self.checkpoint or type(self.archive) is not str or not self.archive):
            raise ValueError('immutable historical snapshot identity required')
        for value in (self.checkpoint_sha256,self.archive_sha256,self.model_sha256):c._binding(value)

    def __getitem__(self,key):return getattr(self,key)


@dataclass(frozen=True)
class Watchdog:
    initial_window_seconds:float
    longest_multiplier:float=5.
    backstop_seconds:float=30*24*3600.
    poll_seconds:float=1.
    terminate_grace_seconds:float=10.

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('finite positive watchdog durations required')
        if self.longest_multiplier<3 or self.backstop_seconds<10*self.initial_window_seconds:
            raise ValueError('several-longest-game stall window and distant backstop required')


@dataclass(frozen=True)
class Plan:
    run_id:str
    model:V5ModelConfig
    loss:V5LossConfig
    train:training.TrainConfig
    iterations:tuple[tuple[Batch,...],...]
    pool_sha256:str
    selection_seed:int
    historical_fraction:float
    historical_window:int
    max_requests:int
    watchdog:Watchdog
    checkpoint_updates:int=32
    discount:float=1.0

    def __post_init__(self):
        object.__setattr__(self,'iterations',tuple(tuple(x) for x in self.iterations))
        _seed(self.selection_seed);c._binding(self.pool_sha256)
        if isinstance(self.discount,bool) or not isinstance(self.discount,Real) or not math.isfinite(self.discount) or not 0<=self.discount<=1:
            raise ValueError('finite explicit discount in0..1 required')
        object.__setattr__(self,'discount',float(self.discount))
        if (type(self.run_id) is not str or not self.run_id or type(self.model) is not V5ModelConfig
                or type(self.loss) is not V5LossConfig or type(self.train) is not training.TrainConfig
                or self.loss.arm!=self.model.arm or type(self.watchdog) is not Watchdog
                or not self.iterations or any(not x or any(type(b) is not Batch for b in x) for x in self.iterations)):
            raise ValueError('explicit same-arm scratch model/loss/trainer and finite declaration required')
        if (self.model.width,self.model.layers,self.model.attention_heads,self.model.expert_width,self.model.value_hidden,self.model.value_bins)!=(512,3,8,512,(512,512),101):
            raise ValueError('initial Phase C512x3/three512-role-expert architecture required')
        if self.model.arm!='A' and (self.model.value_sigma_ratio!=.75 or self.loss.entropy_coefficient<=0
                or self.loss.uniform_kl_coefficient<=0 or self.loss.auxiliary_weight<=0):
            raise ValueError('declared Phase C HL-Gauss sigma/entropy/uniform-KL/auxiliary treatment required')
        if (type(self.historical_window) is not int or self.historical_window<=0
                or type(self.max_requests) is not int or self.max_requests<=0
                or type(self.checkpoint_updates) is not int or self.checkpoint_updates<=0
                or isinstance(self.historical_fraction,bool) or not isinstance(self.historical_fraction,(int,float))
                or not math.isfinite(self.historical_fraction) or not 0<=self.historical_fraction<=1
                or (self.model.arm=='A')!=(self.historical_fraction==0)):
            raise ValueError('declared A selfplay or treatment historical mixture/window required')
        games=self.games
        if (len({g.stream_id for g in games})!=len(games) or len({g.battle_id for g in games})!=len(games)
                or len({t for g in games for t in g.team_indices})!=2*len(games)):
            raise ValueError('one fixed ordered bank, no reused roots/teams required')

    @property
    def games(self):return tuple(g for iteration in self.iterations for batch in iteration for g in batch.games)

    def manifest(self):
        return dict(schema=SCHEMA,run_id=self.run_id,model=self.model.metadata(),loss=self.loss.metadata(),
            train=asdict(self.train),iterations=[[asdict(b) for b in x] for x in self.iterations],pool_sha256=self.pool_sha256,
            selection_seed=self.selection_seed,historical_fraction=self.historical_fraction,historical_window=self.historical_window,
            max_requests=self.max_requests,watchdog=asdict(self.watchdog),checkpoint_updates=self.checkpoint_updates,discount=self.discount,
            contract_sha256=_sha(c.contract_descriptor()),opponent_rule='UNIFORM_RECENT_PAST_SNAPSHOTS_NO_OUTCOME_ADAPTATION',
            initial_history='EMPTY_POOL_EXPLICIT_SELFPLAY',actor='RAW_POLICY',
            game_seed_scope='TRAJECTORY_IDENTITY_ONLY_BATCH_CONSTRUCTOR_AND_SAMPLING_SEEDS_DRIVE_RNG',scientific_qualification=False)


def _publish(path,blob):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.pending-',delete=False) as stream:
        temporary=Path(stream.name);stream.write(blob);stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path);temporary.unlink();path.chmod(0o444)


class Store:
    def __init__(self,path,plan,*,resume=False):
        self.path=Path(path);self.plan=plan;self.path.mkdir(parents=True,exist_ok=True)
        declaration=self.path/'declaration.json';blob=_json(plan.manifest()).encode()
        if resume:
            # Earlier declarations omitted discount and used exactly1.0. Accept
            # only that identical historical recipe; never rewrite its bytes.
            legacy=plan.manifest();legacy.pop('discount')
            compatible=plan.discount==1.0 and declaration.is_file() and declaration.read_bytes()==_json(legacy).encode()
            if not declaration.is_file() or declaration.read_bytes()!=blob and not compatible:
                raise ValueError('original immutable run declaration required')
        else:
            if any(self.path.iterdir()):raise FileExistsError('fresh run directory required')
            _publish(declaration,blob)
        self.events=[];previous=None
        for index,path in enumerate(sorted((self.path/'journal').glob('*.json'))):
            event=json.loads(path.read_text());digest=event.pop('sha256',None)
            if path.name!=f'{index:09d}.json' or event['sequence']!=index or event['previous']!=previous or _sha(event)!=digest:
                raise ValueError('original ordered run journal changed')
            event['sha256']=digest;self.events.append(event);previous=digest
        if not self.events:self.emit('RUN_STARTED',declaration_sha256=_sha(plan.manifest()))

    def emit(self,kind,**payload):
        event=dict(sequence=len(self.events),previous=self.events[-1]['sha256'] if self.events else None,
            time=time.time(),kind=kind,payload=payload)
        event['sha256']=_sha(event)
        _publish(self.path/'journal'/f'{len(self.events):09d}.json',_json(event).encode())
        self.events.append(event);return event

    def matching(self,kind,**fields):
        return [event['payload'] for event in self.events if event['kind']==kind
            and all(event['payload'].get(key)==value for key,value in fields.items())]

    def resolve(self,relative,digest):
        path=(self.path/relative).resolve()
        actual=path/'metadata.json' if path.is_dir() else path
        if not path.is_relative_to(self.path.resolve()) or not actual.is_file() or _file_sha(actual)!=digest:
            raise ValueError('immutable run artifact changed/missing')
        return path

    def artifact(self,relative,blob,*,kind,**facts):
        path=(self.path/relative).resolve()
        if not path.is_relative_to(self.path.resolve()):raise ValueError('owned run artifact path required')
        _publish(path,blob)
        value=dict(path=relative,sha256=_file_sha(path),**facts)
        self.emit(kind,**value);return value

    def checkpoint(self,runtime,*,iteration):
        relative=f'artifacts/train-{iteration}-{uuid.uuid4().hex}.pt';path=self.path/relative;path.parent.mkdir(exist_ok=True)
        receipt=training.save_checkpoint(path,runtime);path.chmod(0o444)
        value=dict(iteration=iteration,path=relative,sha256=receipt['sha256'],steps=runtime.steps,rows=runtime.rows)
        self.emit('CHECKPOINT',**value);return value

    def snapshot(self,checkpoint,*,iteration):
        from gpu_engine import v5_model
        path=self.resolve(checkpoint['path'],checkpoint['sha256'])
        relative=f'artifacts/actor-{iteration}-{uuid.uuid4().hex}.npz'
        metadata=v5_model.export(path,self.path/relative,expected_checkpoint_sha256=checkpoint['sha256'],enable_v5=True)
        (self.path/relative).chmod(0o444)
        value=Snapshot(iteration=iteration,checkpoint=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'],
            archive=relative,archive_sha256=metadata['weights_archive_sha256'],model_sha256=_sha(self.plan.model.metadata()))
        self.emit('SNAPSHOT',**asdict(value));return value


def assignment(plan,game,ordinal,current,past):
    """Stateless declared-seed selection, independent of outcomes and compaction."""
    digest=hashlib.sha256(_json([plan.selection_seed,game.stream_id,ordinal]).encode()).digest()
    draw=int.from_bytes(digest[:8],'big')/2**64
    history=tuple(past[-plan.historical_window:])
    use=plan.model.arm!='A' and bool(history) and draw<plan.historical_fraction
    opponent=history[int.from_bytes(digest[8:16],'big')%len(history)] if use else current
    return dict(stream_id=game.stream_id,learner_side=ordinal%2,opponent=opponent,
        opponent_kind='HISTORICAL' if use else 'SELFPLAY',selection_digest=digest.hex())


def require_treatment_ready(plan,backend):
    if plan.model.arm=='D':
        checker=getattr(backend,'require_teacher',None)
        if checker is None:raise MissingDependency('D current-root public-world materializer/teacher is not integrated')
        checker()  # Must refuse missing materializer before any actual construction.


class Progress:
    def __init__(self,policy,*,started=None):
        self.policy=policy;self.started=time.monotonic() if started is None else started
        self.last=self.started;self.longest=0.;self.completed=0

    def completed_game(self,duration):
        if not math.isfinite(duration) or duration<0:raise ValueError('actual per-game completion time required')
        self.completed+=1;self.longest=max(self.longest,duration);self.last=time.monotonic()

    def check(self):
        now=time.monotonic()
        if now-self.started>self.policy.backstop_seconds:raise TimeoutError('distant total-time backstop')
        window=max(self.policy.initial_window_seconds,self.policy.longest_multiplier*self.longest)
        if now-self.last>window:raise TimeoutError('no actual completed game within adaptive stall window')


def run(plan,path,*,backend,resume=False,enable_v5=False):
    """Bounded collect/cache/train iterations, with explicit original RNG resume.

    Backend collects only fixed supplied batches and writes a genuine v5 cache;
    no outcome-dependent allocation exists. Completed-game panels/promotion are
    separate; their source readiness/launch decisions are never inferred here.
    """
    if enable_v5 is not True or type(plan) is not Plan:raise ValueError('explicit v5 finite run enablement required')
    require_treatment_ready(plan,backend)
    backend.check_plan(plan)
    store=Store(path,plan,resume=resume)
    checkpoints=store.matching('CHECKPOINT')
    if checkpoints:
        checkpoint=checkpoints[-1];verified=store.resolve(checkpoint['path'],checkpoint['sha256'])
        # Deserialize an owned immutable snapshot, not a second read of a mutable path.
        with tempfile.TemporaryDirectory(prefix='v5-resume-') as directory:
            snapshot=Path(directory)/'snapshot.pt';snapshot.write_bytes(verified.read_bytes())
            if _file_sha(snapshot)!=checkpoint['sha256']:raise ValueError('checkpoint snapshot digest differs')
            runtime=training.load_checkpoint(snapshot,enable_v5=True,expected_model_config=plan.model,
                expected_loss_config=plan.loss,expected_train_config=plan.train)
    else:
        runtime=training.create(plan.model,plan.loss,plan.train,enable_v5=True)
        checkpoint=store.checkpoint(runtime,iteration=-1)
    snapshots=[Snapshot(**value) for value in store.matching('SNAPSHOT')]
    if not snapshots:snapshots=[store.snapshot(checkpoint,iteration=-1)]
    for snapshot in snapshots:
        store.resolve(snapshot['checkpoint'],snapshot['checkpoint_sha256'])
        store.resolve(snapshot['archive'],snapshot['archive_sha256'])
        if snapshot['model_sha256']!=_sha(plan.model.metadata()):raise ValueError('historical model namespace differs')
    statuses={g.stream_id:'NOT_ATTEMPTED' for g in plan.games}
    for event in store.events:
        if event['kind']=='BATCH_STARTED':
            for stream in event['payload']['streams']:statuses[stream]='RUNNING'
        elif event['kind']=='GAME_END':statuses[event['payload']['stream_id']]=event['payload']['status']
    # In-flight roots have no restorable original Session. Preserve attempts,
    # never replay them; only the next declared not-attempted roots may start.
    for stream,status in tuple(statuses.items()):
        if status=='RUNNING':statuses[stream]='FAILED';store.emit('GAME_END',stream_id=stream,status='FAILED',reason='interrupted-original-session-not-replayed')
    ordinal={g.stream_id:i for i,g in enumerate(plan.games)}
    for iteration,batches in enumerate(plan.iterations):
        if store.matching('ITERATION_COMPLETE',iteration=iteration):continue
        current=max((x for x in snapshots if x['iteration']<iteration),key=lambda x:x['iteration'])
        past=[x for x in snapshots if x['iteration']<current['iteration']]
        progress=Progress(plan.watchdog)
        for index,batch in enumerate(batches):
            if all(statuses[g.stream_id]!='NOT_ATTEMPTED' for g in batch.games):continue
            remaining=[g.stream_id for g in plan.games if statuses[g.stream_id]=='NOT_ATTEMPTED']
            if [g.stream_id for g in batch.games]!=remaining[:len(batch.games)]:raise ValueError('next fixed not-attempted prefix required')
            assignments=tuple(assignment(plan,g,ordinal[g.stream_id],current,past) for g in batch.games)
            try:prepared_batch=backend.prepare_batch(plan,batch,current,assignments,store)
            except Exception as error:
                store.emit('STOP',iteration=iteration,reason='PREPARATION: '+type(error).__name__+': '+str(error))
                return dict(status='STOPPED',reason='PREPARATION: '+str(error),statuses=statuses)
            store.emit('BATCH_STARTED',iteration=iteration,batch=index,streams=[g.stream_id for g in batch.games],assignments=assignments)
            for g in batch.games:statuses[g.stream_id]='RUNNING'
            def ended(stream_id,status,**facts):
                if statuses.get(stream_id)!='RUNNING':raise ValueError('unexpected/replayed game completion')
                statuses[stream_id]=status
                if status=='COMPLETE':progress.completed_game(facts['duration_seconds'])
                store.emit('GAME_END',stream_id=stream_id,status=status,iteration=iteration,batch=index,**facts)
            try:result=backend.collect(prepared_batch,store,ended,progress.check)
            except Exception as error:
                for g in batch.games:
                    if statuses[g.stream_id]=='RUNNING':ended(g.stream_id,'FAILED',reason=type(error).__name__+': '+str(error))
                store.emit('STOP',iteration=iteration,reason=type(error).__name__+': '+str(error))
                return dict(status='STOPPED',reason=type(error).__name__+': '+str(error),statuses=statuses)
            for g in batch.games:
                if statuses[g.stream_id]=='RUNNING':ended(g.stream_id,'FAILED',reason='collector-returned-without-final-disposition')
            if result is not None:
                cache_path,digest=result
                store.resolve(cache_path,digest)
                store.emit('CACHE',iteration=iteration,batch=index,path=cache_path,sha256=digest)
        caches=sorted(store.matching('CACHE',iteration=iteration),key=lambda x:x['batch'])
        if not caches:
            store.emit('STOP',iteration=iteration,reason='NO_ACTUAL_TRAINING_ROWS_NO_REFILL')
            return dict(status='STOPPED',reason='NO_ACTUAL_TRAINING_ROWS_NO_REFILL',statuses=statuses)
        paths=[store.resolve(x['path'],x['sha256']) for x in caches]
        existing=store.matching('SNAPSHOT',iteration=iteration)
        if not existing:
            if runtime.progress is not None and runtime.progress['complete'] and checkpoint['iteration']!=iteration:
                store.emit('DATASET_TRANSITION',iteration=iteration,prior_complete_cursor=runtime.progress)
                runtime.progress=None  # Only after the exact previous finite cursor completed.
            while runtime.progress is None or not runtime.progress['complete']:
                report=training.train_caches(runtime,paths,max_updates=plan.checkpoint_updates)
                store.emit('TRAIN_PROGRESS',iteration=iteration,steps=runtime.steps,rows=runtime.rows)
                checkpoint=store.checkpoint(runtime,iteration=iteration)
                if report['complete']:break
            current=store.snapshot(checkpoint,iteration=iteration);snapshots.append(current)
        store.emit('ITERATION_COMPLETE',iteration=iteration,snapshot=existing[0] if existing else current,
            counts={name:list(statuses.values()).count(name) for name in ('NOT_ATTEMPTED','RUNNING','COMPLETE','CAPPED','REFUSED','FAILED')})
    return dict(status='COMPLETE_DECLARATION',statuses=statuses,steps=runtime.steps,rows=runtime.rows,
        scientific_qualification=False)


def _stop_owned_group(process,grace,*,sleep):
    """Only the private session created below; finite waits before returning."""
    def alive():
        try:os.killpg(process.pid,0);return True
        except ProcessLookupError:return False
    def signal_group(value):
        try:os.killpg(process.pid,value)
        except ProcessLookupError:pass
    if alive():signal_group(signal.SIGTERM)
    end=time.monotonic()+grace
    while alive() and time.monotonic()<end:sleep(min(.1,max(0.,end-time.monotonic())))
    if alive():
        signal_group(signal.SIGKILL)
        end=time.monotonic()+grace
        while alive() and time.monotonic()<end:sleep(min(.1,max(0.,end-time.monotonic())))
    if alive():raise RuntimeError('owned worker process group remains after bounded cleanup')
    try:process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        try:process.wait(timeout=grace)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('owned worker leader remains after bounded cleanup') from error


def supervise_owned(command,*,plan,path,popen=subprocess.Popen,sleep=time.sleep):
    """Bound THIS owned invocation, including its privately owned descendants.

    Original journal time remains provenance. A later authorized resume gets a
    new invocation backstop while retaining completed/unknown root dispositions.
    Observation errors and interruption clean only this new process session.
    """
    started=time.monotonic();last=started;longest=0.;seen=0;previous=None;phase='PREPARE';last_steps=-1
    process=popen(command,start_new_session=True)
    reason=None;observer_error=None
    try:
        while process.poll() is None:
            try:
                events=sorted((Path(path)/'journal').glob('*.json'))
                if len(events)<seen:reason='PROGRESS_JOURNAL_CHANGED'
                for item in events[seen:]:
                    event=json.loads(item.read_text());digest=event.pop('sha256',None)
                    if event.get('sequence')!=seen or event.get('previous')!=previous or _sha(event)!=digest:
                        reason='PROGRESS_JOURNAL_CHANGED';break
                    previous=digest;seen+=1;payload=event['payload']
                    if event['kind']=='GAME_END' and payload['status']=='COMPLETE':
                        longest=max(longest,payload['duration_seconds']);last=time.monotonic()
                    elif event['kind']=='TRAIN_PROGRESS' and payload['steps']>last_steps:
                        phase='TRAIN';last_steps=payload['steps'];last=time.monotonic()
                    elif event['kind']=='BATCH_STARTED' and phase!='COLLECT':phase='COLLECT';last=time.monotonic()
            except (OSError,ValueError,KeyError,TypeError) as error:
                observer_error=type(error).__name__+': '+str(error)
                reason='PROGRESS_JOURNAL_CHANGED'
            now=time.monotonic()
            if reason is None:
                if now-started>plan.watchdog.backstop_seconds:reason='DISTANT_BACKSTOP'
                elif now-last>max(plan.watchdog.initial_window_seconds,plan.watchdog.longest_multiplier*longest):reason='NO_DURABLE_WORK_PROGRESS'
            if reason:break
            sleep(plan.watchdog.poll_seconds)
    finally:
        # Also reap descendants after a leader exits or an observer raises;
        # an exited leader alone does not establish resource/slot release.
        _stop_owned_group(process,plan.watchdog.terminate_grace_seconds,sleep=sleep)
    return dict(exit_code=process.returncode,stop_reason=reason,owned_pid=getattr(process,'pid',None),observer_error=observer_error)
