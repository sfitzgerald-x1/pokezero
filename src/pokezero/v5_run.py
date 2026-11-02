"""Explicit finite v5 iterations and immutable historical snapshots.

The caller supplies a predeclared game/seed bank and an engine backend. This
module never draws teams, allocates a job, loads a legacy warm start, evaluates
a holdout, or promotes an arm. Real execution needs Torch and the separately
prepared original engine/JAX backend. Source readiness is not qualification.
"""
from dataclasses import asdict, dataclass, is_dataclass, replace
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
EVALUATION_INTERVAL_GAMES = 500_000


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
class EvaluationProtocol:
    """Explicit external panel recipe; never a qualification or launch decision."""
    protocol_sha256:str

    def __post_init__(self):c._binding(self.protocol_sha256)

    def metadata(self):
        return dict(protocol_sha256=self.protocol_sha256,interval_games=EVALUATION_INTERVAL_GAMES,
            progress='DECLARED_GAME_ORDINAL',attempts='RESUME_NOT_ATTEMPTED_NO_CELL_REPLAY')


@dataclass(frozen=True)
class EvaluationRequest:
    declared_games:int
    declaration_sha256:str
    protocol_sha256:str
    snapshot:Snapshot
    run_directory:Path
    checkpoint_path:Path
    archive_path:Path
    progress_callback:object=None  # Host-only, excluded from immutable binding.
    prior_completions:tuple=()  # Original journal facts, not new game dispatches.


@dataclass(frozen=True)
class EvaluationCompletion:
    cell_ordinal:int
    offset:int
    seed:int
    seat:str
    duration_seconds:float
    source_game_json:str
    source_result_sha256:str
    learner_username:str
    opponent_username:str
    protocol_path:str|None=None
    foulplay_seed:int|None=None


def _checked_evaluation_completion(receipt):
    from types import SimpleNamespace
    from .v5_foulplay import _score
    if (type(receipt) is not EvaluationCompletion or type(receipt.offset) is not int or not 0<=receipt.offset<500
            or type(receipt.cell_ordinal) is not int or receipt.seat not in ('p1','p2')
            or receipt.cell_ordinal!=2*receipt.offset+(receipt.seat=='p2')
            or type(receipt.seed) is not int or not 0<=receipt.seed<2**32
            or isinstance(receipt.duration_seconds,bool) or not isinstance(receipt.duration_seconds,Real)
            or not math.isfinite(receipt.duration_seconds) or receipt.duration_seconds<0
            or any(type(x) is not str or not x for x in (receipt.learner_username,receipt.opponent_username))):
        raise ValueError('literal completed original paired FP cell receipt required')
    game=json.loads(receipt.source_game_json)
    if _sha(game)!=receipt.source_result_sha256 or game.get('seed')!=receipt.seed:
        raise ValueError('actual completed FP source game differs')
    config=SimpleNamespace(pokezero_player=receipt.seat,pokezero_username=receipt.learner_username,
        foulplay_username=receipt.opponent_username)
    if _score(config,game) is None:raise ValueError('unknown/capped FP game cannot advance completion watchdog')
    return receipt


def _checked_completion_protocol(receipt,binding):
    """Join the literal terminal to its actual declared paired cell, not a tag."""
    if type(receipt.protocol_path) is not str or not Path(receipt.protocol_path).is_absolute():
        raise ValueError('original evaluation protocol artifact required for completion')
    blob=Path(receipt.protocol_path).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=binding['protocol_sha256']:
        raise ValueError('completed cell protocol digest differs')
    protocol=json.loads(blob)
    if (type(protocol) is not dict or protocol.get('schema')!='pokezero.v5.milestone_evaluation.v1'
            or protocol.get('model_sha256')!=binding['snapshot']['model_sha256']):
        raise ValueError('completed cell snapshot model differs')
    matches=[p for p in protocol['panels'] if p['declared_games']==binding['declared_games']]
    if len(matches)!=1:raise ValueError('original exact milestone cell declaration required')
    panel=matches[0]
    if len(panel['seeds'])!=500 or len(panel['configs'])!=2:raise ValueError('complete original paired cell roster required')
    config=panel['configs'][receipt.seat=='p2']
    from .foulplay_bridge import _per_seed_foulplay_random_seed_schedule_for_offsets
    from types import SimpleNamespace
    fp_seed=_per_seed_foulplay_random_seed_schedule_for_offsets(SimpleNamespace(
        foulplay_random_seed=config['foulplay_random_seed'],seed_start=config['seed_start']),offsets=(receipt.offset,))[0]
    if (panel['seeds'][receipt.offset]!=receipt.seed
            or type(receipt.foulplay_seed) is not int or receipt.foulplay_seed!=fp_seed
            or config['pokezero_player']!=receipt.seat or config['seed_start']+receipt.offset!=receipt.seed
            or config['pokezero_username']!=receipt.learner_username or config['foulplay_username']!=receipt.opponent_username):
        raise ValueError('completed source seed/seat/cell differs from declared protocol')


def _checked_evaluation_progress(payload,expected,seen):
    if (set(payload)!=set(expected)|{'completion','status','duration_seconds'}
            or any(payload[k]!=v for k,v in expected.items())):
        raise ValueError('original open evaluation progress binding required')
    receipt=_checked_evaluation_completion(EvaluationCompletion(**payload['completion']))
    _checked_completion_protocol(receipt,expected)
    if receipt.cell_ordinal in seen or payload['status']!='COMPLETE' or payload['duration_seconds']!=receipt.duration_seconds:
        raise ValueError('replayed/mismatched evaluation completion')
    seen.add(receipt.cell_ordinal)
    return receipt


@dataclass(frozen=True)
class EvaluationOutcome:
    """An owned immutable report receipt, or an explicit unknown disposition."""
    status:str
    report:str|None=None
    report_sha256:str|None=None
    reason:str|None=None

    def __post_init__(self):
        if self.status not in ('RESULT','UNKNOWN'):raise ValueError('evaluation RESULT or UNKNOWN required')
        if (self.report is None)!=(self.report_sha256 is None):raise ValueError('complete evaluation report receipt required')
        if self.report is not None:
            if type(self.report) is not str or not self.report:raise ValueError('owned evaluation report path required')
            c._binding(self.report_sha256)
        if self.status=='RESULT' and (self.report is None or self.reason is not None):
            raise ValueError('evaluation RESULT requires exact report receipt')
        if self.status=='UNKNOWN' and (type(self.reason) is not str or not self.reason):
            raise ValueError('evaluation UNKNOWN requires a reason')


def evaluation_milestones(iteration_game_counts):
    """Every fixed 500k boundary must be an exact declared iteration end."""
    counts=tuple(iteration_game_counts)
    if not counts or any(type(count) is not int or count<=0 for count in counts):
        raise ValueError('positive declared iteration game counts required')
    total=0;milestones={}
    for iteration,count in enumerate(counts):
        end=total+count
        boundary=(total//EVALUATION_INTERVAL_GAMES+1)*EVALUATION_INTERVAL_GAMES
        if boundary<end:raise ValueError('declared iteration crosses an exact500k evaluation checkpoint')
        if boundary==end:milestones[iteration]=end
        total=end
    return milestones


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
    evaluation:EvaluationProtocol|None=None

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
        if self.evaluation is not None:
            if type(self.evaluation) is not EvaluationProtocol:raise ValueError('typed evaluation protocol required')
            evaluation_milestones(sum(len(b.games) for b in batches) for batches in self.iterations)

    @property
    def games(self):return tuple(g for iteration in self.iterations for batch in iteration for g in batch.games)

    def manifest(self):
        value=dict(schema=SCHEMA,run_id=self.run_id,model=self.model.metadata(),loss=self.loss.metadata(),
            train=training.train_config_metadata(self.train),iterations=[[asdict(b) for b in x] for x in self.iterations],pool_sha256=self.pool_sha256,
            selection_seed=self.selection_seed,historical_fraction=self.historical_fraction,historical_window=self.historical_window,
            max_requests=self.max_requests,watchdog=asdict(self.watchdog),checkpoint_updates=self.checkpoint_updates,discount=self.discount,
            contract_sha256=_sha(c.contract_descriptor()),opponent_rule='UNIFORM_RECENT_PAST_SNAPSHOTS_NO_OUTCOME_ADAPTATION',
            initial_history='EMPTY_POOL_EXPLICIT_SELFPLAY',actor='RAW_POLICY',
            game_seed_scope='TRAJECTORY_IDENTITY_ONLY_BATCH_CONSTRUCTOR_AND_SAMPLING_SEEDS_DRIVE_RNG',scientific_qualification=False)
        if self.evaluation is not None:value['evaluation']=self.evaluation.metadata()
        return value


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
        if not self.events:self.emit('RUN_STARTED',declaration_sha256=_sha(json.loads(declaration.read_bytes())))

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


def learning_rate_window(plan,iteration):
    """Progress uses every DECLARED game ordinal, never accepted/cache rows."""
    if type(plan) is not Plan or type(iteration) is not int or not 0<=iteration<len(plan.iterations):
        raise ValueError('original declared iteration required for schedule window')
    counts=[sum(len(batch.games) for batch in batches) for batches in plan.iterations]
    start=sum(counts[:iteration])
    return training.DeclaredGameWindow(start,start+counts[iteration],sum(counts))


def _evaluation_binding(store,snapshot):
    milestones=evaluation_milestones(sum(len(b.games) for b in batches) for batches in store.plan.iterations)
    if snapshot.iteration not in milestones:raise ValueError('evaluation requires exact declared milestone snapshot')
    return dict(iteration=snapshot.iteration,declared_games=milestones[snapshot.iteration],
        declaration_sha256=_sha(json.loads((store.path/'declaration.json').read_bytes())),
        protocol_sha256=store.plan.evaluation.protocol_sha256,snapshot=asdict(snapshot))


def _evaluation_request(store,snapshot):
    binding=_evaluation_binding(store,snapshot)
    checkpoint=store.resolve(snapshot.checkpoint,snapshot.checkpoint_sha256)
    archive=store.resolve(snapshot.archive,snapshot.archive_sha256)
    if snapshot.model_sha256!=_sha(store.plan.model.metadata()):raise ValueError('evaluation snapshot model namespace differs')
    return EvaluationRequest(binding['declared_games'],binding['declaration_sha256'],binding['protocol_sha256'],
        snapshot,store.path.resolve(),checkpoint,archive)


def _checked_evaluation_start(store,event):
    payload=event['payload'];iteration=payload.get('iteration');sequence=event['sequence']
    if store.plan.evaluation is None:raise ValueError('evaluation progress requires opted-in original plan')
    prior=[e for e in store.events if e['sequence']<sequence]
    snapshots=[e['payload'] for e in prior if e['kind']=='SNAPSHOT' and e['payload'].get('iteration')==iteration]
    if len(snapshots)!=1:raise ValueError('original prior milestone snapshot required')
    snapshot=Snapshot(**snapshots[0]);binding=_evaluation_binding(store,snapshot)
    expected=dict(**binding,evaluation_sha256=_sha(binding))
    if payload!=expected:raise ValueError('original evaluation start binding differs')
    checkpoints=[e for e in prior if e['kind']=='CHECKPOINT' and e['payload'].get('iteration')==iteration
        and e['payload'].get('path')==snapshot.checkpoint and e['payload'].get('sha256')==snapshot.checkpoint_sha256]
    if len(checkpoints)!=1:raise ValueError('evaluation snapshot requires original prior checkpoint')
    if store.matching('RUN_STARTED')!=[dict(declaration_sha256=binding['declaration_sha256'])]:
        raise ValueError('original accepted evaluation run declaration required')
    milestones=evaluation_milestones(sum(len(b.games) for b in batches) for batches in store.plan.iterations)
    for earlier in (i for i in milestones if i<iteration):
        if not any(e['kind'] in ('EVALUATION_RESULT','EVALUATION_UNKNOWN') and e['payload'].get('iteration')==earlier for e in prior):
            raise ValueError('evaluation milestone order differs')
    _evaluation_request(store,snapshot)
    return expected


def _evaluation_history(store):
    """Verify panel attempts; an interrupted panel keeps its untouched suffix."""
    declaration_sha256=_sha(json.loads((store.path/'declaration.json').read_bytes()))
    if store.matching('RUN_STARTED')!=[dict(declaration_sha256=declaration_sha256)]:
        raise ValueError('original accepted run declaration binding required')
    milestones=evaluation_milestones(sum(len(b.games) for b in batches) for batches in store.plan.iterations)
    starts={};closed={};published={};completed={}
    for event in store.events:
        if event['kind']=='SNAPSHOT':
            snapshot=Snapshot(**event['payload'])
            if snapshot.iteration in published:raise ValueError('one original snapshot per iteration required')
            published[snapshot.iteration]=snapshot
        if event['kind']=='ITERATION_COMPLETE' and event['payload'].get('iteration') in milestones:
            iteration=event['payload']['iteration']
            if iteration not in published or _sha(_evaluation_binding(store,published[iteration])) not in closed:
                raise ValueError('milestone iteration completed without an attempted evaluation disposition')
        if event['kind'] not in ('EVALUATION_STARTED','EVALUATION_RESUMED','EVALUATION_RESULT','EVALUATION_UNKNOWN','EVALUATION_GAME_END'):continue
        if store.plan.evaluation is None:raise ValueError('evaluation journal requires original opted-in declaration')
        payload=event['payload'];iteration=payload.get('iteration')
        if iteration not in published:raise ValueError('evaluation requires one original published snapshot')
        snapshot=published[iteration];binding=_evaluation_binding(store,snapshot)
        expected=dict(**binding,evaluation_sha256=_sha(binding))
        if {key:payload.get(key) for key in expected}!=expected:
            raise ValueError('original evaluation snapshot/protocol/declaration binding changed')
        key=expected['evaluation_sha256']
        if event['kind']=='EVALUATION_STARTED':
            if key in starts or set(payload)!=set(expected):raise ValueError('evaluation attempt replay/extra fields')
            starts[key]=_checked_evaluation_start(store,event)
        elif event['kind']=='EVALUATION_RESUMED':
            if (key not in starts or payload!=starts[key]
                    or key in closed and closed[key].status!='UNKNOWN'):
                raise ValueError('only an unfinished original evaluation can resume')
            closed.pop(key,None)
        elif event['kind']=='EVALUATION_GAME_END':
            if key not in starts or key in closed:
                raise ValueError('evaluation completion requires open original attempt')
            _checked_evaluation_progress(payload,starts[key],completed.setdefault(key,set()))
        else:
            if key not in starts or key in closed or set(payload)!=set(expected)|{'outcome'}:
                raise ValueError('evaluation result requires one durable original attempt')
            outcome=EvaluationOutcome(**payload['outcome'])
            if event['kind']!='EVALUATION_'+outcome.status:raise ValueError('evaluation disposition differs')
            if outcome.report is not None:store.resolve(outcome.report,outcome.report_sha256)
            closed[key]=outcome
    return closed


def _evaluate_snapshot(store,snapshot,hook):
    """Resume the same panel, with idempotent completed-cell delivery only."""
    binding=_evaluation_binding(store,snapshot);key=_sha(binding)
    outcomes=[e['payload']['outcome']for e in store.events
        if e['kind'] in ('EVALUATION_RESULT','EVALUATION_UNKNOWN') and e['payload'].get('evaluation_sha256')==key]
    if outcomes and outcomes[-1]['status']=='RESULT':return True
    request=_evaluation_request(store,snapshot)
    bound=dict(**binding,evaluation_sha256=key)
    store.emit('EVALUATION_RESUMED' if store.matching('EVALUATION_STARTED',evaluation_sha256=key) else 'EVALUATION_STARTED',**bound)
    completed={e['completion']['cell_ordinal']:EvaluationCompletion(**e['completion'])for e in
        store.matching('EVALUATION_GAME_END',evaluation_sha256=key)}
    carried=set(completed);delivered=set()
    def actual_completion(receipt):
        _checked_evaluation_completion(receipt)
        _checked_completion_protocol(receipt,bound)
        if receipt.cell_ordinal in completed:
            if completed[receipt.cell_ordinal]!=receipt:raise ValueError('completed evaluation source changed on resume')
            if receipt.cell_ordinal not in carried or receipt.cell_ordinal in delivered:
                raise ValueError('completed evaluation cell replayed')
            delivered.add(receipt.cell_ordinal)
            return  # The stored panel ledger is delivered again, never its game.
        store.emit('EVALUATION_GAME_END',**bound,status='COMPLETE',duration_seconds=receipt.duration_seconds,
            completion=asdict(receipt))
        completed[receipt.cell_ordinal]=receipt
    request=replace(request,progress_callback=actual_completion,prior_completions=tuple(completed.values()))
    try:
        outcome=hook(request)
        if type(outcome) is not EvaluationOutcome:raise ValueError('typed evaluation outcome required')
        # The callback cannot replace its published input or return a foreign
        # output receipt. Snapshot metadata itself stays the original object.
        if _evaluation_binding(store,snapshot)!=binding:raise ValueError('evaluation input metadata changed during callback')
        _evaluation_request(store,snapshot)
        if outcome.report is not None:store.resolve(outcome.report,outcome.report_sha256)
    except Exception as error:
        outcome=EvaluationOutcome('UNKNOWN',reason=type(error).__name__+': '+str(error))
    store.emit('EVALUATION_'+outcome.status,**bound,outcome=asdict(outcome))
    return outcome.status=='RESULT'


def run(plan,path,*,backend,resume=False,enable_v5=False,evaluation_hook=None,measure_timings=False):
    """Bounded collect/cache/train iterations, with explicit original RNG resume.

    Backend collects only fixed supplied batches and writes a genuine v5 cache;
    no outcome-dependent allocation exists. An explicitly declared external
    milestone hook records its result; it never promotes or retunes an arm.
    """
    if type(measure_timings)is not bool:raise ValueError('explicit runtime timing choice required')
    if enable_v5 is not True or type(plan) is not Plan:raise ValueError('explicit v5 finite run enablement required')
    milestones={}
    if plan.evaluation is not None:
        if type(plan.evaluation) is not EvaluationProtocol or not callable(evaluation_hook):
            raise ValueError('explicit declared evaluation protocol and callable hook required')
        milestones=evaluation_milestones(sum(len(b.games) for b in batches) for batches in plan.iterations)
    elif evaluation_hook is not None:raise ValueError('evaluation hook requires opted-in immutable declaration')
    require_treatment_ready(plan,backend)
    backend.check_plan(plan)
    store=Store(path,plan,resume=resume)
    def measured(scope,callback,*args,**kwargs):
        if not measure_timings:return callback(*args,**kwargs)
        clock=time.perf_counter();status='FAILED'
        try:
            result=callback(*args,**kwargs);status='COMPLETE';return result
        finally:
            store.emit('RUNTIME_PHASE',scope=scope,wall_seconds=time.perf_counter()-clock,status=status,
                timing_scope='host wall inclusive of original operation and its data I/O; not pure GPU compute')
    if plan.evaluation is not None:_evaluation_history(store)
    checkpoints=store.matching('CHECKPOINT')
    if checkpoints:
        checkpoint=checkpoints[-1];verified=store.resolve(checkpoint['path'],checkpoint['sha256'])
        # Deserialize an owned immutable snapshot, not a second read of a mutable path.
        with tempfile.TemporaryDirectory(prefix='v5-resume-') as directory:
            snapshot=Path(directory)/'snapshot.pt';snapshot.write_bytes(verified.read_bytes())
            if _file_sha(snapshot)!=checkpoint['sha256']:raise ValueError('checkpoint snapshot digest differs')
            scheduled=plan.train.learning_rate_schedule!='constant'
            options=(dict(expected_learning_rate_window=learning_rate_window(plan,checkpoint['iteration']))
                if scheduled and checkpoint['iteration']>=0 else {})
            runtime=measured('TRAINER_RESUME',training.load_checkpoint,snapshot,enable_v5=True,expected_model_config=plan.model,
                expected_loss_config=plan.loss,expected_train_config=plan.train,**options)
    else:
        runtime=measured('TRAINER_CREATE_COLD',training.create,plan.model,plan.loss,plan.train,enable_v5=True)
        checkpoint=measured('CHECKPOINT_WRITE',store.checkpoint,runtime,iteration=-1)
    snapshots=[Snapshot(**value) for value in store.matching('SNAPSHOT')]
    if not snapshots:snapshots=[measured('ACTOR_EXPORT',store.snapshot,checkpoint,iteration=-1)]
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
                if plan.train.learning_rate_schedule=='constant':
                    runtime.progress=None  # Only after the exact previous finite cursor completed.
            options=(dict(learning_rate_window=learning_rate_window(plan,iteration))
                if plan.train.learning_rate_schedule!='constant' else {})
            while (runtime.progress is None or not runtime.progress['complete']
                    or options and checkpoint['iteration']!=iteration):
                report=measured('TRAIN_CACHES_INCLUSIVE',training.train_caches,runtime,paths,max_updates=plan.checkpoint_updates,**options)
                store.emit('TRAIN_PROGRESS',iteration=iteration,steps=runtime.steps,rows=runtime.rows)
                checkpoint=measured('CHECKPOINT_WRITE',store.checkpoint,runtime,iteration=iteration)
                if report['complete']:break
            current=measured('ACTOR_EXPORT',store.snapshot,checkpoint,iteration=iteration);snapshots.append(current)
        published=Snapshot(**existing[0]) if existing else current
        if iteration in milestones and not _evaluate_snapshot(store,published,evaluation_hook):
            store.emit('STOP',iteration=iteration,reason='EVALUATION_UNKNOWN_NO_AUTOMATIC_REPLAY')
            return dict(status='STOPPED',reason='EVALUATION_UNKNOWN_NO_AUTOMATIC_REPLAY',statuses=statuses,
                steps=runtime.steps,rows=runtime.rows,scientific_qualification=False)
        store.emit('ITERATION_COMPLETE',iteration=iteration,snapshot=published,
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
    evaluation_store=Store.__new__(Store);evaluation_store.path=Path(path);evaluation_store.plan=plan;evaluation_store.events=[]
    evaluation_starts={};evaluation_closed={};evaluation_cells={}
    milestones=(evaluation_milestones(sum(len(b.games) for b in batches) for batches in plan.iterations)
        if plan.evaluation is not None else {})
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
                    if type(payload) is not dict:raise ValueError('original journal payload required')
                    kind=event['kind']
                    if kind in ('RUN_STARTED','CHECKPOINT','SNAPSHOT','EVALUATION_STARTED','EVALUATION_RESUMED','EVALUATION_GAME_END',
                            'EVALUATION_RESULT','EVALUATION_UNKNOWN','ITERATION_COMPLETE'):
                        evaluation_store.events.append(event)
                    if kind=='EVALUATION_STARTED':
                        key=payload.get('evaluation_sha256')
                        if key in evaluation_starts:raise ValueError('evaluation start replayed')
                        evaluation_starts[key]=_checked_evaluation_start(evaluation_store,event)
                    elif kind=='EVALUATION_RESUMED':
                        key=payload.get('evaluation_sha256')
                        if (key not in evaluation_starts or payload!=evaluation_starts[key]
                                or key in evaluation_closed and evaluation_closed[key]!='UNKNOWN'):
                            raise ValueError('only an unfinished original evaluation can resume')
                        evaluation_closed.pop(key,None)
                        last=time.monotonic()
                    elif kind=='EVALUATION_GAME_END':
                        key=payload.get('evaluation_sha256')
                        if key not in evaluation_starts or key in evaluation_closed:
                            raise ValueError('evaluation progress requires an open original milestone')
                        receipt=_checked_evaluation_progress(payload,evaluation_starts[key],evaluation_cells.setdefault(key,set()))
                        longest=max(longest,receipt.duration_seconds);last=time.monotonic()
                    elif kind in ('EVALUATION_RESULT','EVALUATION_UNKNOWN'):
                        key=payload.get('evaluation_sha256');expected=evaluation_starts.get(key)
                        if (expected is None or key in evaluation_closed or set(payload)!=set(expected)|{'outcome'}
                                or any(payload[k]!=v for k,v in expected.items())):
                            raise ValueError('evaluation disposition requires one original open milestone')
                        outcome=EvaluationOutcome(**payload['outcome'])
                        if kind!='EVALUATION_'+outcome.status:raise ValueError('evaluation disposition changed')
                        if outcome.report is not None:evaluation_store.resolve(outcome.report,outcome.report_sha256)
                        evaluation_closed[key]=outcome.status
                    elif kind=='ITERATION_COMPLETE' and payload.get('iteration') in milestones:
                        if not any(key in evaluation_closed and binding['iteration']==payload['iteration']
                                for key,binding in evaluation_starts.items()):
                            raise ValueError('milestone iteration completed without original evaluation disposition')
                    if kind=='GAME_END' and payload['status']=='COMPLETE':
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
