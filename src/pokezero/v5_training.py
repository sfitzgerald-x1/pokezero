"""Explicit v5 cache trainer and resumable v5-only checkpoint format.

No rollout, world sampler, opponent pool, old checkpoint migration or implicit
training entrypoint exists here. The owner supplies reviewed v5 caches and an
explicit seed/configuration. Each runtime owns its RNG stream and exact finite
cache cursor; resume continues that stream rather than replaying prior batches.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import random
import tempfile

import numpy as np

from . import v5_contract as contract, v5_data
from .v5_model import MODEL_INPUTS, V5ModelConfig, build_v5_model, forward_v5, require_v5_bindings
from .v5_loss import (BASE_PPO_FIELDS, SEARCH_FIELDS, FLOAT_FIELDS, INTEGER_FIELDS, MASK_FIELDS,
    V5LossConfig, v5_batch_to_torch, v5_loss)

CHECKPOINT_SCHEMA = 'pokezero.checkpoint.v5'
TRAINER_SCHEMA = 'pokezero.cache_trainer.v5'
BOOTSTRAP_SCHEMA = 'pokezero.public_value_bootstrap.v5'


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

    def __post_init__(self):
        if (type(self.seed) is not int or not 0<=self.seed<2**32
                or any(type(v) is not int or v<=0 for v in (self.batch_size,self.epochs))
                or type(self.device) is not str or not self.device
                or self.precision not in ('float32','bfloat16')):
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
    return Runtime(model_config,loss_config,train_config,model,optimizer,rng,contract.contract_descriptor())


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
        if not isinstance(capture,dict) or capture.get('schema')!=BOOTSTRAP_SCHEMA:
            raise ValueError('active capped critic target requires actual v5_bootstrap capture')
        original=contract.DecisionAddress(**batch.v5_columns['addresses'][index])
        address=contract.DecisionAddress(**capture.get('address',{}))
        if (address.stream!=original.stream or address.owner!=original.owner or address.frame<=original.frame):
            raise ValueError('capped bootstrap must use a later original stream/owner boundary')
        contract._binding(capture.get('source_frame_sha256'))
        contract._real(capture.get('value'),low=-1,high=1)


def _validate_optimizer(runtime):
    _validate_optimizer_groups(runtime.optimizer.param_groups,runtime.train_config)


def _validate_optimizer_groups(groups,config):
    if len(groups)!=1:raise ValueError('original single-group v5 AdamW required')
    actual=groups[0]
    expected=dict(lr=config.learning_rate,weight_decay=config.weight_decay,
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
    if type(runtime) is not Runtime:raise ValueError('v5 runtime required')
    if batch.v5_columns is None:raise ValueError('explicit v5 batch required')
    v5_data.validate_batch(batch);require_v5_bindings(batch.v5_columns,runtime.model_config)
    validate_bootstraps(batch)
    _validate_optimizer(runtime)
    torch=_torch();config=runtime.train_config
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


def train_caches(runtime,paths,*,max_updates=None):
    """Continue the exact ordered finite dataset; no refill, shuffle or replay."""
    if type(runtime) is not Runtime:raise ValueError('v5 runtime required')
    if max_updates is not None and (type(max_updates) is not int or max_updates<0):
        raise ValueError('nonnegative update budget required')
    paths=tuple(Path(path) for path in paths)
    manifest=_cache_manifest(paths,runtime.train_config.batch_size)
    if runtime.progress is None:
        runtime.progress=dict(manifest=manifest,epochs=runtime.train_config.epochs,epoch=0,cache=0,batch=0,complete=False)
    elif runtime.progress['manifest']!=manifest or runtime.progress['epochs']!=runtime.train_config.epochs:
        raise ValueError('resume requires original ordered v5 caches and epoch budget')
    _validate_progress(runtime.progress)
    metrics=[];progress=runtime.progress
    while not progress['complete'] and (max_updates is None or len(metrics)<max_updates):
        shard=progress['cache'];start=progress['batch']
        batches=v5_data.iter_cache_batches(paths[shard],batch_size=runtime.train_config.batch_size)
        for index,batch in enumerate(batches):
            if index<start:continue
            metrics.append(update(runtime,batch))
            progress['batch']=index+1
            if progress['batch']==manifest[shard]['batches']:
                progress['batch']=0;progress['cache']+=1
                if progress['cache']==len(paths):
                    progress['cache']=0;progress['epoch']+=1
                    progress['complete']=progress['epoch']==progress['epochs']
            if max_updates is not None and len(metrics)>=max_updates:break
    return dict(schema=TRAINER_SCHEMA,steps=runtime.steps,rows=runtime.rows,
        complete=progress['complete'],progress=json.loads(json.dumps(progress)),updates=metrics,
        full_B4_qualified=False,scope='V5_CACHE_TRAINER_ONLY_NO_ROLLOUT_OR_PHASE_C_RESULT')


def _metadata(runtime):
    return dict(schema=CHECKPOINT_SCHEMA,training_schema=contract.TRAINING_CONTRACT_SCHEMA,
        observation_schema=contract.OBSERVATION_SCHEMA_VERSION_V5,model=runtime.model_config.metadata(),
        loss=runtime.loss_config.metadata(),training=asdict(runtime.train_config),
        source_contract=runtime.source_contract,source_contract_sha256=_sha(runtime.source_contract),
        category_vocabulary_sha256=runtime.model_config.category_vocabulary_sha256,
        label_vocabulary_sha256=runtime.model_config.label_vocabulary.sha256,
        optimizer='AdamW',steps=runtime.steps,rows=runtime.rows,progress=runtime.progress,
        model_training=runtime.model.training)


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
                    expected_train_config=None):
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
    runtime.model.load_state_dict(payload['state_dict'],strict=True)
    # Validate the literal saved recipe before Torch can normalize defaults.
    _validate_optimizer_groups(payload['optimizer_state']['param_groups'],training)
    runtime.optimizer.load_state_dict(payload['optimizer_state'])
    _validate_optimizer(runtime)
    if any(not torch.isfinite(parameter).all() for parameter in runtime.model.parameters()):
        raise ValueError('finite v5 checkpoint parameters required')
    runtime.rng=payload['rng']
    with _rng_scope(torch,training.device,state=runtime.rng):pass
    runtime.model.train(metadata['model_training']);runtime.steps=metadata['steps'];runtime.rows=metadata['rows']
    runtime.progress=metadata['progress']
    return runtime
