"""Explicit v5 cache trainer and resumable v5-only checkpoint format.

No rollout, world sampler, opponent pool, old checkpoint migration or implicit
training entrypoint exists here. The owner supplies reviewed v5 caches and an
explicit seed/configuration. Each runtime owns its RNG stream and exact finite
cache cursor; resume continues that stream rather than replaying prior batches.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import weakref

import numpy as np

from . import v5_contract as contract, v5_data
from .v5_model import MODEL_INPUTS, V5ModelConfig, build_v5_model, forward_v5, require_v5_bindings
from .v5_loss import (BASE_PPO_FIELDS, SEARCH_FIELDS, FLOAT_FIELDS, INTEGER_FIELDS, MASK_FIELDS,
    V5LossConfig, v5_batch_to_torch, v5_loss)

CHECKPOINT_SCHEMA = 'pokezero.checkpoint.v5'
TRAINER_SCHEMA = 'pokezero.cache_trainer.v5'
BOOTSTRAP_SCHEMA = 'pokezero.public_value_bootstrap.v5'
_SCHEDULED = weakref.WeakKeyDictionary()
_CACHE_READERS = weakref.WeakKeyDictionary()
_RESIDENT_POLICY = ContextVar('v5_resident_cache_bytes',default=None)


@contextmanager
def resident_cache_policy(byte_budget):
    """Process-local resource choice; never part of optimizer or row state."""
    if type(byte_budget)is not int or byte_budget<=0:raise ValueError('positive resident cache byte budget required')
    token=_RESIDENT_POLICY.set(byte_budget)
    try:yield
    finally:_RESIDENT_POLICY.reset(token)


@dataclass(frozen=True)
class DeclaredGameWindow:
    """Immutable ordinal interval in the original declared game budget."""
    start_game: int
    end_game: int
    total_games: int

    def __post_init__(self):
        if (any(type(x) is not int for x in (self.start_game,self.end_game,self.total_games))
                or not 0<=self.start_game<self.end_game<=self.total_games):
            raise ValueError('ordered declared-game window within explicit total required')


@dataclass(frozen=True)
class _LearningRateState:
    window: DeclaredGameWindow
    manifest_sha256: str
    base_steps: int
    base_rows: int
    progress_sha256: str


def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    batch_size: int = 256
    epochs: int = 1
    learning_rate: float = 0.0003
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: float | None = None
    device: str = 'cpu'
    precision: str = 'float32'
    learning_rate_schedule: str = 'constant'

    def __post_init__(self):
        if (type(self.seed) is not int or not 0<=self.seed<2**32
                or any(type(v) is not int or v<=0 for v in (self.batch_size,self.epochs))
                or type(self.device) is not str or not self.device
                or self.precision not in ('float32','bfloat16')
                or type(self.learning_rate_schedule) is not str
                or self.learning_rate_schedule not in ('constant','mit-thesis')):
            raise ValueError('explicit uint32 seed, positive finite batch/epochs and precision/device required')
        for name in ('learning_rate','weight_decay','beta1','beta2','epsilon','max_grad_norm'):
            value=getattr(self,name)
            if name=='max_grad_norm' and value is None:continue
            if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value):
                raise ValueError('finite real optimizer settings required')
            object.__setattr__(self,name,float(value))
        if (self.learning_rate<=0 or self.weight_decay<0 or self.epsilon<=0
                or not 0<=self.beta1<1 or not 0<=self.beta2<1
                or self.max_grad_norm is not None and self.max_grad_norm<=0):
            raise ValueError('valid AdamW/gradient settings required')


@dataclass(eq=False)
class Runtime:
    model_config: V5ModelConfig
    loss_config: V5LossConfig
    train_config: TrainConfig
    model: object
    optimizer: object
    rng: dict
    source_contract: dict
    steps: int = 0
    rows: int = 0
    progress: dict | None = None
    learning_rate_state: _LearningRateState | None = None


def train_config_metadata(config):
    """The constant recipe keeps the original checkpoint/declaration bytes."""
    value=asdict(config)
    if config.learning_rate_schedule=='constant':value.pop('learning_rate_schedule')
    return value


def _epoch_learning_rate(config,window,epoch):
    from .neural_policy import _learning_rate_for_epoch
    # Reuse the existing epoch interpolation, including epochs==1 -> start.
    recipe=SimpleNamespace(epochs=config.epochs,learning_rate=config.learning_rate,
        learning_rate_schedule=config.learning_rate_schedule,
        learning_rate_progress_start=window.start_game/window.total_games,
        learning_rate_progress_end=window.end_game/window.total_games,
        learning_rate_warmup_progress=0.,learning_rate_resume_warmup_start_progress=None,
        learning_rate_resume_warmup_end_progress=None,learning_rate_resume_warmup_start_lr=None)
    return _learning_rate_for_epoch(recipe,epoch)


def _cursor_totals(progress,batch_size):
    per_epoch=sum(x['batches'] for x in progress['manifest'])
    examples=sum(x['examples'] for x in progress['manifest'])
    steps=progress['epoch']*per_epoch+sum(x['batches'] for x in progress['manifest'][:progress['cache']])+progress['batch']
    rows=progress['epoch']*examples+sum(x['examples'] for x in progress['manifest'][:progress['cache']])
    if not progress['complete']:rows+=min(progress['batch']*batch_size,progress['manifest'][progress['cache']]['examples'])
    return steps,rows,per_epoch


def _expected_learning_rate(runtime):
    config=runtime.train_config;record=_SCHEDULED.get(runtime)
    if config.learning_rate_schedule=='constant' and record is None:
        if runtime.learning_rate_state is not None:raise ValueError('constant runtime cannot carry scheduled state')
        return config.learning_rate
    if record is None or record[0] is not config or runtime.learning_rate_state is not record[1]:
        raise ValueError('original scheduled configuration/state required')
    state=runtime.learning_rate_state
    if state is None:
        if runtime.progress is not None or runtime.steps!=0 or runtime.rows!=0:
            raise ValueError('fresh scheduled runtime has no cache cursor')
        return config.learning_rate
    _validate_progress(runtime.progress)
    if (runtime.progress is None or runtime.progress['epochs']!=config.epochs
            or _sha(runtime.progress)!=state.progress_sha256
            or _sha(runtime.progress['manifest'])!=state.manifest_sha256):
        raise ValueError('original scheduled finite cache cursor required')
    steps,rows,per_epoch=_cursor_totals(runtime.progress,config.batch_size)
    if runtime.steps!=state.base_steps+steps or runtime.rows!=state.base_rows+rows:
        raise ValueError('scheduled training counters differ from original cache cursor')
    last_epoch=1 if steps==0 else (steps-1)//per_epoch+1
    return _epoch_learning_rate(config,state.window,last_epoch)


def _bind_learning_rate_state(runtime,state):
    runtime.learning_rate_state=state
    _SCHEDULED[runtime]=(runtime.train_config,state)


def _torch():
    from .neural_policy import require_torch
    return require_torch()


def _rng_state(torch, cuda):
    numpy=np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=[state.clone() for state in torch.cuda.get_rng_state_all()] if cuda else [],
        python=random.getstate(),numpy=dict(algorithm=numpy[0],keys=numpy[1].tolist(),
            position=int(numpy[2]),has_gaussian=int(numpy[3]),cached_gaussian=float(numpy[4])))


def _restore_rng(torch, state):
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])
    random.setstate(state['python'])
    value=state['numpy']
    np.random.set_state((value['algorithm'],np.asarray(value['keys'],np.uint32),value['position'],
        value['has_gaussian'],value['cached_gaussian']))


@contextmanager
def _rng_scope(torch, device, state=None, seed=None):
    cuda=device.startswith('cuda')
    previous=_rng_state(torch,cuda)
    try:
        if state is not None:_restore_rng(torch,state)
        else:
            torch.random.default_generator.manual_seed(seed)
            if cuda:torch.cuda.manual_seed_all(seed)
            random.seed(seed);np.random.seed(seed)
        yield
    finally:
        _restore_rng(torch,previous)


def create(model_config,loss_config,train_config,*,enable_v5=False):
    """Scratch initialization only; opt-in precedes Torch/model allocation."""
    if enable_v5 is not True:raise ValueError('explicit v5 trainer enablement required')
    if (type(model_config) is not V5ModelConfig or type(loss_config) is not V5LossConfig
            or type(train_config) is not TrainConfig or model_config.arm!=loss_config.arm):
        raise ValueError('same-arm explicit v5 model/loss/train configuration required')
    torch=_torch()
    with _rng_scope(torch,train_config.device,seed=train_config.seed):
        model=build_v5_model(model_config,enable_v5=True).to(train_config.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,betas=(train_config.beta1,train_config.beta2),
            eps=train_config.epsilon,foreach=False,amsgrad=False,maximize=False,
            capturable=False,differentiable=False,fused=None)
        rng=_rng_state(torch,train_config.device.startswith('cuda'))
    runtime=Runtime(model_config,loss_config,train_config,model,optimizer,rng,contract.contract_descriptor())
    if train_config.learning_rate_schedule!='constant':_bind_learning_rate_state(runtime,None)
    return runtime


def validate_bootstraps(batch):
    """Capped critic labels need an explicit actual public bootstrap capture.

    v5_bootstrap in step_metadata is ordinary source data, not an attestation:
    schema, later same-stream/owner address, finite value, and source-frame SHA.
    Its collector must have evaluated that original public boundary. No return,
    GAE number or capped-terminal zero can create this record automatically.
    """
    active=np.asarray(batch.terminal_capped,dtype=bool)&np.asarray(batch.ppo_value_target_mask,dtype=bool)
    for index in np.flatnonzero(active):
        capture=batch.step_metadata[index].get('v5_bootstrap')
        if not isinstance(capture,Mapping) or capture.get('schema')!=BOOTSTRAP_SCHEMA:
            raise ValueError('active capped critic target requires actual v5_bootstrap capture')
        original=contract.DecisionAddress(**batch.v5_columns['addresses'][index])
        address=contract.DecisionAddress(**capture.get('address',{}))
        if (address.stream!=original.stream or address.owner!=original.owner or address.frame<=original.frame):
            raise ValueError('capped bootstrap must use a later original stream/owner boundary')
        contract._binding(capture.get('source_frame_sha256'))
        contract._real(capture.get('value'),low=-1,high=1)


def _validate_optimizer(runtime):
    _validate_optimizer_groups(runtime.optimizer.param_groups,runtime.train_config,
        learning_rate=_expected_learning_rate(runtime))


def _validate_optimizer_groups(groups,config,*,learning_rate=None):
    if len(groups)!=1:raise ValueError('original single-group v5 AdamW required')
    actual=groups[0]
    expected=dict(lr=config.learning_rate if learning_rate is None else learning_rate,weight_decay=config.weight_decay,
        betas=(config.beta1,config.beta2),eps=config.epsilon,foreach=False,
        amsgrad=False,maximize=False,capturable=False,differentiable=False,fused=None,
        decoupled_weight_decay=True)
    if (set(actual)!={'params',*expected}
            or any(actual.get(name)!=value for name,value in expected.items())
            or any(type(actual[name]) is not type(expected[name]) for name in
                ('foreach','amsgrad','maximize','capturable','differentiable','fused','decoupled_weight_decay'))):
        raise ValueError('v5 optimizer configuration differs from checkpoint training metadata')


def _training_tensors(batch,model_config,device):
    # Memmapped cache views are read-only. Own the named tensor payload before
    # Torch conversion, without tensorizing optional private views or metadata.
    resolved=dict(batch.v5_columns)
    for name in (*MODEL_INPUTS,'legal_action_mask',*BASE_PPO_FIELDS,*SEARCH_FIELDS):
        value=getattr(batch,name,None)
        if value is not None:resolved[name]=value
    for name in set(FLOAT_FIELDS)|set(INTEGER_FIELDS)|set(MASK_FIELDS):
        value=resolved.get(name)
        if isinstance(value,np.ndarray) and not value.flags.writeable:resolved[name]=value.copy()
    return v5_batch_to_torch(resolved,model_config,device=device,training=True)


def update(runtime,batch):
    """One actual optimizer update from original public inputs and masked labels."""
    if type(runtime) is Runtime and runtime.train_config.learning_rate_schedule!='constant':
        raise ValueError('scheduled updates require the original finite train_caches cursor')
    return _update(runtime,batch)


def _update(runtime,batch):
    if type(runtime) is not Runtime:raise ValueError('v5 runtime required')
    if batch.v5_columns is None:raise ValueError('explicit v5 batch required')
    if not v5_data._validated_resident_slice(batch):
        v5_data.validate_batch(batch)
        validate_bootstraps(batch)
    require_v5_bindings(batch.v5_columns,runtime.model_config)
    _validate_optimizer(runtime)
    torch=_torch();config=runtime.train_config
    scheduled=config.learning_rate_schedule!='constant'
    if scheduled:
        progress=runtime.progress
        if progress is None or progress['complete']:raise ValueError('active scheduled cache cursor required')
        record=progress['manifest'][progress['cache']]
        expected_rows=min(config.batch_size,record['examples']-progress['batch']*config.batch_size)
        if batch.batch_size!=expected_rows:raise ValueError('scheduled batch differs from original cache cursor')
        learning_rate=_epoch_learning_rate(config,runtime.learning_rate_state.window,progress['epoch']+1)
        for group in runtime.optimizer.param_groups:group['lr']=learning_rate
    with _rng_scope(torch,config.device,state=runtime.rng):
        runtime.model.train()
        runtime.optimizer.zero_grad(set_to_none=True)
        tensors=_training_tensors(batch,runtime.model_config,config.device)
        scope=(torch.autocast(device_type=torch.device(config.device).type,dtype=torch.bfloat16)
            if config.precision=='bfloat16' else nullcontext())
        with scope:
            output=forward_v5(runtime.model,tensors,include_auxiliary=runtime.model_config.treatment)
            loss,metrics=v5_loss(output,tensors,runtime.model_config,runtime.loss_config)
        if not torch.isfinite(loss):raise ValueError('finite v5 optimizer loss required')
        loss.backward()
        gradients=[parameter.grad for parameter in runtime.model.parameters() if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError('finite v5 optimizer gradients required')
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in gradients]))
        if config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(runtime.model.parameters(),config.max_grad_norm,error_if_nonfinite=True)
        runtime.optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in runtime.model.parameters()):
            raise ValueError('finite v5 optimizer parameters required')
        runtime.rng=_rng_state(torch,config.device.startswith('cuda'))
    runtime.steps+=1;runtime.rows+=batch.batch_size
    if scheduled:
        epoch=runtime.progress['epoch']+1
        _advance_progress(runtime.progress)
        _bind_learning_rate_state(runtime,replace(runtime.learning_rate_state,
            progress_sha256=_sha(runtime.progress)))
        return dict(metrics,optimizer_step=runtime.steps,owner_rows=batch.batch_size,
            gradient_norm=float(norm),learning_rate=learning_rate,learning_rate_epoch=epoch)
    return dict(metrics,optimizer_step=runtime.steps,owner_rows=batch.batch_size,gradient_norm=float(norm))


def _cache_manifest(paths,batch_size):
    result=[]
    for path in paths:
        metadata=json.loads((Path(path)/'metadata.json').read_text())
        copied=dict(metadata);digest=copied.pop('metadata_sha256',None)
        if metadata.get('schema_version')!=v5_data.CACHE_SCHEMA or digest!=_sha(copied):
            raise ValueError('explicit intact v5 cache metadata required')
        count=metadata.get('example_count')
        if type(count) is not int or count<=0:raise ValueError('positive fixed cache example count required')
        result.append(dict(sha256=digest,examples=count,batches=(count+batch_size-1)//batch_size))
    if not result:raise ValueError('finite nonempty ordered cache roster required')
    return result


class _CacheReader:
    """Original order, with one mapped shard or a bounded resident iteration."""
    def __init__(self,paths,batch_size,resident_cache_bytes=None):
        self.paths=paths;self.batch_size=batch_size;self.resident_cache_bytes=resident_cache_bytes
        self.identities={path/'metadata.json':v5_data._file_identity(path/'metadata.json') for path in paths}
        self.manifest=_cache_manifest(paths,batch_size)
        v5_data._require_cache_identities(self.identities)
        self.active=None;self.shard=None
        self.resident=None
        if resident_cache_bytes is not None:
            if type(resident_cache_bytes) is not int or resident_cache_bytes<=0:
                raise ValueError('positive finite resident cache byte budget required')
            required=0;per_cache=[]
            for path in paths:
                metadata=json.loads((path/'metadata.json').read_text())
                size=0
                for description in metadata['arrays'].values():
                    shape=description['shape'];dtype=np.dtype(description['dtype'])
                    if dtype.hasobject or any(type(x) is not int or x<0 for x in shape):
                        raise ValueError('finite typed resident cache shape required')
                    size+=math.prod(shape)*dtype.itemsize
                required+=size;per_cache.append(size)
            if required>resident_cache_bytes:
                raise ValueError('resident iteration exceeds explicit cache byte budget')
            self.resident=tuple(v5_data.VerifiedCache(path,resident_bytes=size)
                for path,size in zip(paths,per_cache))
            v5_data._require_cache_identities(self.identities)

    def batches(self,shard,start):
        if self.shard!=shard:
            self.active=None;self.shard=None
            active=(self.resident[shard] if self.resident is not None else
                v5_data.VerifiedCache(self.paths[shard]))
            v5_data._require_cache_identities(self.identities)
            self.active=active;self.shard=shard
        return self.active.batches(batch_size=self.batch_size,start_batch=start)


def _cache_reader(runtime,paths,resident_cache_bytes=None):
    reader=_CACHE_READERS.get(runtime)
    if (reader is None or reader.paths!=paths or reader.batch_size!=runtime.train_config.batch_size
            or reader.resident_cache_bytes!=resident_cache_bytes):
        reader=_CacheReader(paths,runtime.train_config.batch_size,resident_cache_bytes)
        _CACHE_READERS[runtime]=reader
    v5_data._require_cache_identities(reader.identities)
    return reader


def _validate_progress(progress):
    if progress is None:return
    if (set(progress)!={'manifest','epochs','epoch','cache','batch','complete'}
            or type(progress['complete']) is not bool
            or any(type(progress[k]) is not int or progress[k]<0 for k in ('epochs','epoch','cache','batch'))
            or progress['epochs']<1 or not progress['manifest']):
        raise ValueError('valid finite v5 cache cursor required')
    for record in progress['manifest']:
        contract._binding(record['sha256'])
        if any(type(record[k]) is not int or record[k]<=0 for k in ('examples','batches')):
            raise ValueError('positive cache manifest counts required')
    if progress['complete']:
        valid=progress['epoch']==progress['epochs'] and progress['cache']==progress['batch']==0
    else:
        valid=(progress['epoch']<progress['epochs'] and progress['cache']<len(progress['manifest'])
            and progress['batch']<progress['manifest'][progress['cache']]['batches'])
    if not valid:raise ValueError('v5 cache cursor outside original finite roster')


def _advance_progress(progress):
    shard=progress['cache'];progress['batch']+=1
    if progress['batch']==progress['manifest'][shard]['batches']:
        progress['batch']=0;progress['cache']+=1
        if progress['cache']==len(progress['manifest']):
            progress['cache']=0;progress['epoch']+=1
            progress['complete']=progress['epoch']==progress['epochs']


def train_caches(runtime,paths,*,max_updates=None,learning_rate_window=None,resident_cache_bytes=None):
    """Continue the exact ordered finite dataset; no refill, shuffle or replay."""
    if type(runtime) is not Runtime:raise ValueError('v5 runtime required')
    policy=_RESIDENT_POLICY.get()
    if policy is not None:
        if resident_cache_bytes is not None and resident_cache_bytes!=policy:
            raise ValueError('resident cache resource policy differs')
        resident_cache_bytes=policy
    if max_updates is not None and (type(max_updates) is not int or max_updates<0):
        raise ValueError('nonnegative update budget required')
    if runtime in _SCHEDULED:_validate_optimizer(runtime)
    paths=tuple(Path(path) for path in paths)
    reader=_cache_reader(runtime,paths,resident_cache_bytes);manifest=json.loads(json.dumps(reader.manifest))
    scheduled=runtime.train_config.learning_rate_schedule!='constant'
    if scheduled:
        if type(learning_rate_window) is not DeclaredGameWindow:
            raise ValueError('explicit immutable declared-game learning-rate window required')
        _validate_optimizer(runtime)  # Validate the LAST applied rate before changing it.
        state=runtime.learning_rate_state
        if state is None:
            if learning_rate_window.start_game!=0:raise ValueError('fresh schedule must start at declared game zero')
        elif state.window!=learning_rate_window:
            if (not runtime.progress['complete'] or state.window.end_game!=learning_rate_window.start_game
                    or state.window.total_games!=learning_rate_window.total_games):
                raise ValueError('scheduled dataset transition requires completed cursor and contiguous declared windows')
            if state.manifest_sha256==_sha(manifest):
                raise ValueError('new declared window cannot replay the previous cache roster')
            runtime.progress=None
    elif learning_rate_window is not None:
        raise ValueError('constant learning rate does not accept a scheduled window')
    if runtime.progress is None:
        runtime.progress=dict(manifest=manifest,epochs=runtime.train_config.epochs,epoch=0,cache=0,batch=0,complete=False)
        if scheduled:
            _bind_learning_rate_state(runtime,_LearningRateState(learning_rate_window,_sha(manifest),
                runtime.steps,runtime.rows,_sha(runtime.progress)))
            for group in runtime.optimizer.param_groups:
                group['lr']=_epoch_learning_rate(runtime.train_config,learning_rate_window,1)
    elif runtime.progress['manifest']!=manifest or runtime.progress['epochs']!=runtime.train_config.epochs:
        raise ValueError('resume requires original ordered v5 caches and epoch budget')
    _validate_progress(runtime.progress)
    metrics=[];progress=runtime.progress
    while not progress['complete'] and (max_updates is None or len(metrics)<max_updates):
        shard=progress['cache'];start=progress['batch']
        batches=reader.batches(shard,start)
        for batch in batches:
            metrics.append(_update(runtime,batch) if scheduled else update(runtime,batch))
            if not scheduled:_advance_progress(progress)
            if max_updates is not None and len(metrics)>=max_updates:break
    return dict(schema=TRAINER_SCHEMA,steps=runtime.steps,rows=runtime.rows,
        complete=progress['complete'],progress=json.loads(json.dumps(progress)),updates=metrics,
        full_B4_qualified=False,scope='V5_CACHE_TRAINER_ONLY_NO_ROLLOUT_OR_PHASE_C_RESULT')


def _metadata(runtime):
    return dict(schema=CHECKPOINT_SCHEMA,training_schema=contract.TRAINING_CONTRACT_SCHEMA,
        observation_schema=contract.OBSERVATION_SCHEMA_VERSION_V5,model=runtime.model_config.metadata(),
        loss=runtime.loss_config.metadata(),training=train_config_metadata(runtime.train_config),
        source_contract=runtime.source_contract,source_contract_sha256=_sha(runtime.source_contract),
        category_vocabulary_sha256=runtime.model_config.category_vocabulary_sha256,
        label_vocabulary_sha256=runtime.model_config.label_vocabulary.sha256,
        optimizer='AdamW',steps=runtime.steps,rows=runtime.rows,progress=runtime.progress,
        model_training=runtime.model.training,
        **(dict(learning_rate_state=None if runtime.learning_rate_state is None else asdict(runtime.learning_rate_state))
            if runtime.train_config.learning_rate_schedule!='constant' else {}))


def save_checkpoint(path,runtime):
    """Create one versioned checkpoint; existing files are never overwritten."""
    if type(runtime) is not Runtime:raise ValueError('v5 runtime required')
    _validate_progress(runtime.progress)
    _validate_optimizer(runtime)
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    metadata=_metadata(runtime)
    payload=dict(metadata=metadata,metadata_sha256=_sha(metadata),state_dict=runtime.model.state_dict(),
        optimizer_state=runtime.optimizer.state_dict(),rng=runtime.rng)
    torch=_torch()
    # Temporary file belongs to this operation; leave it recoverable on error.
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name);torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    # Hard-link publication is atomic and refuses a concurrent existing target.
    os.link(temporary,path);temporary.unlink()
    return dict(path=str(path),sha256=v5_data._file_sha(path),steps=runtime.steps,rows=runtime.rows,
        schema=CHECKPOINT_SCHEMA)


def load_checkpoint(path,*,enable_v5=False,expected_model_config=None,expected_loss_config=None,
                    expected_train_config=None,expected_learning_rate_window=None):
    """Explicit same-v5 resume; no old loader, permissive warmstart or fresh head."""
    if enable_v5 is not True:raise ValueError('explicit v5 trainer enablement required')
    torch=_torch();payload=torch.load(Path(path),map_location='cpu',weights_only=True)
    metadata=payload.get('metadata',{})
    if (metadata.get('schema')!=CHECKPOINT_SCHEMA or payload.get('metadata_sha256')!=_sha(metadata)
            or metadata.get('training_schema')!=contract.TRAINING_CONTRACT_SCHEMA
            or metadata.get('observation_schema')!=contract.OBSERVATION_SCHEMA_VERSION_V5
            or metadata.get('optimizer')!='AdamW'
            or metadata.get('source_contract_sha256')!=_sha(metadata.get('source_contract'))
            or metadata.get('source_contract_sha256')!=_sha(contract.contract_descriptor())):
        raise ValueError('explicit intact same-source v5 checkpoint required')
    model=V5ModelConfig.from_metadata(metadata['model'])
    loss=V5LossConfig(**{field.name:metadata['loss'][field.name] for field in fields(V5LossConfig)})
    training=TrainConfig(**metadata['training'])
    if (_sha(loss.metadata())!=_sha(metadata['loss'])
            or _sha(train_config_metadata(training))!=_sha(metadata['training'])
            or model.category_vocabulary_sha256!=metadata['category_vocabulary_sha256']
            or model.label_vocabulary.sha256!=metadata['label_vocabulary_sha256']):
        raise ValueError('checkpoint source/model/loss namespace differs')
    for expected,actual in ((expected_model_config,model),(expected_loss_config,loss),(expected_train_config,training)):
        if expected is not None and expected!=actual:raise ValueError('explicit resume configuration differs')
    for name in ('steps','rows'):
        if type(metadata[name]) is not int or metadata[name]<0:raise ValueError('nonnegative training counters required')
    if type(metadata['model_training']) is not bool:raise ValueError('explicit model mode required')
    _validate_progress(metadata['progress'])
    runtime=create(model,loss,training,enable_v5=True)
    runtime.steps=metadata['steps'];runtime.rows=metadata['rows'];runtime.progress=metadata['progress']
    if training.learning_rate_schedule!='constant':
        if 'learning_rate_state' not in metadata:raise ValueError('explicit scheduled checkpoint state required')
        saved=metadata.get('learning_rate_state')
        if saved is None:
            if expected_learning_rate_window is not None:raise ValueError('fresh scheduled checkpoint has no bound window')
            state=None
        else:
            if (not isinstance(saved,dict) or set(saved)!={f.name for f in fields(_LearningRateState)}
                    or type(saved['window']) is not dict
                    or set(saved['window'])!={f.name for f in fields(DeclaredGameWindow)}
                    or any(type(saved[k]) is not int or saved[k]<0 for k in ('base_steps','base_rows'))):
                raise ValueError('original scheduled checkpoint state required')
            contract._binding(saved['manifest_sha256']);contract._binding(saved['progress_sha256'])
            window=DeclaredGameWindow(**saved['window'])
            if type(expected_learning_rate_window) is not DeclaredGameWindow or expected_learning_rate_window!=window:
                raise ValueError('explicit original declared-game resume window required')
            state=_LearningRateState(**dict(saved,window=window))
        _bind_learning_rate_state(runtime,state)
    elif 'learning_rate_state' in metadata or expected_learning_rate_window is not None:
        raise ValueError('constant checkpoint cannot accept scheduled state/window')
    expected_rate=_expected_learning_rate(runtime)
    runtime.model.load_state_dict(payload['state_dict'],strict=True)
    # Validate the literal saved recipe before Torch can normalize defaults.
    _validate_optimizer_groups(payload['optimizer_state']['param_groups'],training,learning_rate=expected_rate)
    runtime.optimizer.load_state_dict(payload['optimizer_state'])
    _validate_optimizer(runtime)
    if any(not torch.isfinite(parameter).all() for parameter in runtime.model.parameters()):
        raise ValueError('finite v5 checkpoint parameters required')
    runtime.rng=payload['rng']
    with _rng_scope(torch,training.device,state=runtime.rng):pass
    runtime.model.train(metadata['model_training'])
    return runtime
