"""Private Phase C preparation/configuration glue; import performs no work.

Four opt-in source arms, fresh initialization, caller-supplied fixed banks and
original runtime objects. This module never allocates jobs, creates teams,
loads a checkpoint/model, samples worlds, trains or evaluates a holdout.
"""
from dataclasses import dataclass,fields
import hashlib
import json
from pathlib import Path
import re

from . import v5_contract as c,v5_run as runs,v5_training as training
from .v5_model import V5ModelConfig
from .v5_loss import V5LossConfig
from .live_protocol_actor import ActorContract
from .category_vocab import CategoryVocabulary

SCHEMA='pokezero.phasec.preparation.v1'
GAMES_PER_ARM=2_000_000
ARMS=(('A',0),('B',0),('D',0),('B',1))
CHOICES=(
    'Use direct9375 champion PPO/optimizer controls, never historical2.5M profile.',
    'Primary arms share declared initialization seed0; B replicate1 uses1; no RNG is consumed during preparation.',
    'Common512x3/preRMS/three512experts/dropout from champion; scratch, no checkpoint warmstart.',
    'A has no auxiliary heads/loss; old slot-index opponent loss is not copied.',
    'Policy lambda stays current; only treatment value lambda becomes.95 and value clip is removed.',
    'Use existing v5 treatment auxiliary.1, legal-uniform KL.01, search-policy1 onlyD; no tuning.',
    'Historical mixture50percent over recent32 snapshots; initial empty pool is explicit selfplay.',
    'Reset shared original mit-thesis schedule over the fixed2M declaration; original epoch interpolation, no inherited9375 progress.',
    'D labels onequarter by address, two equal posterior worlds/all legal replies/current public value leaf; actor staysraw.',
)


def _sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def category_binding(vocabulary):
    if type(vocabulary) is not CategoryVocabulary:raise ValueError('original category vocabulary required')
    return _sha(dict(tokens=vocabulary.tokens,oov_buckets=vocabulary.oov_buckets,aliases=dict(vocabulary.aliases)))


def _role(arm,replicate):
    if (arm,replicate) not in ARMS:raise ValueError('initial A/B/D/Brep1 only')
    return arm+str(replicate)


@dataclass(frozen=True)
class ChampionProfile:
    inventory_sha256:str
    payload:str

    def __post_init__(self):
        c._binding(self.inventory_sha256)
        if type(self.payload) is not str or hashlib.sha256(self.payload.encode()).hexdigest()!=self.inventory_sha256:
            raise ValueError('intact original metadata byte snapshot required')
        payload=json.loads(self.payload)
        if payload.get('checkpoint_sha256')!=ActorContract.champion().checkpoint_sha256:
            raise ValueError('direct selected9375 champion metadata required')
        model=payload.get('metadata',{}).get('model_config',{});train=payload.get('metadata',{}).get('training_config',{})
        if model.get('policy_id')!='foundation-midscale-iter-9375' or train.get('objective')!='ppo' or train.get('ppo_target_mode')!='gae':
            raise ValueError('original9375 raw PPO metadata required')
        for name in ('batch_size','epochs','learning_rate','weight_decay','discount','clip_epsilon',
                'entropy_coef','value_loss_weight','value_clip_range','gae_lambda','max_grad_norm','normalize_advantage'):
            if name not in train:raise ValueError('missing champion control:'+name)
        if train.get('learning_rate_schedule')!='mit-thesis' or train.get('learning_rate_progress_end')!=1.:
            raise ValueError('original completed champion schedule provenance required')
        c._real(model.get('dropout'),low=0,high=1)

    @classmethod
    def read(cls,path,*,expected_sha256):
        c._binding(expected_sha256);blob=Path(path).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=expected_sha256:raise ValueError('bound direct metadata inventory differs')
        return cls(expected_sha256,blob.decode())

    @property
    def metadata(self):return json.loads(self.payload)['metadata']

    def configurations(self,vocabulary,*,category_vocabulary,arm,replicate=0):
        role=_role(arm,replicate);meta=self.metadata;old=meta['training_config'];contract=ActorContract.champion()
        if type(vocabulary) is not c.LabelVocabulary:raise ValueError('original public source label vocabulary required')
        category_sha=category_binding(category_vocabulary)
        if category_vocabulary.size!=meta['model_config']['categorical_vocab_size']:raise ValueError('original category size differs')
        model=V5ModelConfig(category_vocabulary.size,category_sha,vocabulary,
            arm=arm,replicate=replicate,width=512,layers=3,attention_heads=8,expert_width=512,
            dropout=meta['model_config']['dropout'])
        loss=V5LossConfig(arm=arm,clip_epsilon=old['clip_epsilon'],value_weight=old['value_loss_weight'],
            entropy_coefficient=old['entropy_coef'],uniform_kl_coefficient=0. if arm=='A' else .01,
            auxiliary_weight=0. if arm=='A' else .1,search_policy_weight=1. if arm=='D' else 0.,
            value_lambda=old['gae_lambda'] if arm=='A' else .95,policy_lambda=old['gae_lambda'],
            value_clip=old['value_clip_range'] if arm=='A' else None,normalize_advantage=old['normalize_advantage'],
            historical_opponent_pool=arm!='A')
        train=dict(seed=replicate,batch_size=old['batch_size'],epochs=old['epochs'],learning_rate=old['learning_rate'],
            learning_rate_schedule='mit-thesis',weight_decay=old['weight_decay'],max_grad_norm=old['max_grad_norm'],
            beta1=.9,beta2=.999,epsilon=1e-8,device='cuda:0',precision='bfloat16')
        return dict(role=role,model=model,loss=loss,training=train,discount=old['discount'],
            games=GAMES_PER_ARM,historical_fraction=0. if arm=='A' else .5,historical_window=32)


def bind_plan(profile,vocabulary,*,category_vocabulary,run_id,arm,replicate,iterations,pool_sha256,selection_seed):
    """Bind only an already declared full bank, never create/redraw its cells."""
    config=profile.configurations(vocabulary,category_vocabulary=category_vocabulary,arm=arm,replicate=replicate)
    iterations=tuple(tuple(batch for batch in iteration) for iteration in iterations)
    if sum(len(batch.games) for iteration in iterations for batch in iteration)!=GAMES_PER_ARM:
        raise ValueError('complete predeclared2M game bank required; no refill or smaller production substitute')
    if 'learning_rate_schedule' not in {f.name for f in fields(training.TrainConfig)}:
        raise runs.MissingDependency('reviewed scheduled v5 trainer is not integrated')
    return runs.Plan(run_id,config['model'],config['loss'],training.TrainConfig(**config['training']),iterations,
        pool_sha256,selection_seed,config['historical_fraction'],config['historical_window'],1024,
        runs.Watchdog(1800,longest_multiplier=5.,backstop_seconds=30*24*3600.),discount=config['discount'])


def build_backend(plan,prepared,pool,*,public_config=None,actor_contract=None,gender_policy=None,device_rollout=False,device_teacher=False,cache_workers=0,teacher_frame_batch=False,
        raw_retention_every=1,transient_raw_root=None,teacher_queue_chunks=0,teacher_vectorized=False):
    """Explicit device rollout or the unchanged original protocol backend."""
    from gpu_engine import v5_selfplay,v5_depth1_provider
    from .v5_gumbel import Recipe
    if type(plan) is not runs.Plan:raise ValueError('original finite plan required')
    if type(device_rollout)is not bool or type(device_teacher)is not bool:raise ValueError('explicit device rollout/teacher factory flags required')
    if device_teacher and (not device_rollout or plan.model.arm!='D'):raise ValueError('device teacher is only the explicit D device route')
    if type(teacher_vectorized)is not bool or teacher_vectorized and not teacher_frame_batch:raise ValueError('vectorized labeling requires the explicit frame-batched D route')
    if type(teacher_frame_batch)is not bool or teacher_frame_batch and not device_teacher:raise ValueError('frame batching needs the explicit D teacher')
    if (type(teacher_queue_chunks)is not int or not 0<=teacher_queue_chunks<=4
            or teacher_queue_chunks and not teacher_frame_batch):
        raise ValueError('queued labeling requires the explicit frame-batched D route')
    if type(cache_workers)is not int or cache_workers<0 or cache_workers and not device_rollout:
        raise ValueError('CPU cache workers require the explicit device route')
    if type(raw_retention_every)is not int or raw_retention_every<=0:
        raise ValueError('positive fixed raw retention interval required')
    if not device_rollout and (raw_retention_every!=1 or transient_raw_root is not None):
        raise ValueError('raw retention choice requires the explicit device route')
    if device_rollout:
        from gpu_engine.v5_device_backend import RuntimeBackend
        teacher=None
        if plan.model.arm=='D':
            if not device_teacher:raise runs.MissingDependency('device D requires explicit public-world teacher enablement; no FULL fallback')
            from gpu_engine.v5_device_teacher import Provider
            teacher=Provider(prepared,plan,**(dict(frame_batch=True,**(dict(chunk_batch=True)if teacher_queue_chunks else{}))if teacher_frame_batch else{}),**(dict(vectorized=True)if teacher_vectorized else{}))
        options={}if raw_retention_every==1 and transient_raw_root is None else dict(raw_retention_every=raw_retention_every,transient_raw_root=transient_raw_root)
        if teacher_queue_chunks:options['teacher_queue_chunks']=teacher_queue_chunks
        return RuntimeBackend(prepared,pool,gender_policy=gender_policy,teacher=teacher,cache_workers=cache_workers,**options)
    teacher=None
    if plan.model.arm=='D':
        if public_config is None or actor_contract is None:raise runs.MissingDependency('D needs bound original public parser/contract')
        teacher=v5_depth1_provider.Provider(prepared,public_config,actor_contract,Recipe(plan.selection_seed))
    return v5_selfplay.EngineBackend(prepared,pool,gender_policy=gender_policy,compute_dtype='bfloat16',teacher=teacher)


def preparation(profile,vocabulary,*,category_vocabulary):
    """Compact PREPONLY declaration, with undiscovered runtime bindings explicit."""
    arms=[]
    for arm,replicate in ARMS:
        row=profile.configurations(vocabulary,category_vocabulary=category_vocabulary,arm=arm,replicate=replicate)
        arms.append(dict(row,model=row['model'].metadata(),loss=row['loss'].metadata()))
    missing=['four disjoint fixed game banks/poolSHA/gender declarations',
        'actual original prepared runtime/category/source handles','fixed FP panel/pairedendpoint/position-set bindings',
        'exact500k panel/snapshot orchestration (existing run has no evaluation hook)',
        'private worker argv/image digest/resources/volume bindings']
    if 'learning_rate_schedule' not in {f.name for f in fields(training.TrainConfig)}:
        missing.insert(0,'current reviewed scheduled-trainer source integration')
    return dict(schema=SCHEMA,status='PREPARATION_ONLY_NO_LAUNCH',control_inventory_sha256=profile.inventory_sha256,
        control_checkpoint_sha256=ActorContract.champion().checkpoint_sha256,arms=arms,choices=CHOICES,
        learning_rate=dict(source_function='pokezero.neural_policy._learning_rate_for_epoch',
            total_declared_games=GAMES_PER_ARM,reset_progress=[0.,1.],constant_within_epoch=True),
        panels=dict(every_declared_games=500_000,games=1000,at=[500_000,1_000_000,1_500_000,2_000_000],
            metrics=['FoulPlay_win_rate','value_calibration','top_two_resolution'],fixed_position_set_required=True,
            execution='UNBOUND_EXISTING_RUN_HAS_NO_EVALUATION_HOOK'),
        endpoint=dict(paired_games_per_contrast=2000,contrasts=[['B0','A0'],['D0','B0']],confidence=.9,
            advance='contrast interval entirely abovezero',fallback='C onB on freedGPUs onlyif neither primarycontrastpasses'),
        teacher=dict(selection_fraction=.25,worlds=2,depth=1,leaf='CURRENT_PUBLIC_VALUE',
            reply_policy='SAMPLED_OPPONENT_OWN_PUBLIC_VIEW',acting_policy='RAW',outcome_root_mix=[.5,.5],
            mixing_site='V5Labels.columns exactlyonce',unknown_positive_reply_mass='WHOLE_LABEL_REFUSED'),
        gpu_count=4,same_node_affinity=True,initial_privileged_critic=False,no_refill=True,
        launch_conditions=['B4 passes','source/pipeline reviews complete','actual fixed banks and artifacts bound'],
        missing_bindings=missing)


def job_templates(prepared,*,name,namespace,image,worker_command,volumes,volume_mounts,cpu,memory,private_manifest_root):
    """Four private suspended manifests; no API/subprocess/cluster interaction.

    Caller supplies actual private paths/image/resources. Pending bindings are
    not guessed. Each allocated pod requests one distinct GPU from Kubernetes.
    """
    if prepared.get('schema')!=SCHEMA or prepared.get('status')!='PREPARATION_ONLY_NO_LAUNCH':raise ValueError('original preparation document required')
    if not re.fullmatch(r'.+@sha256:[0-9a-f]{64}',image):raise ValueError('immutable supplied worker image required')
    if (type(cpu) is not int or cpu<=0 or type(memory) is not str or not memory
            or type(worker_command) not in (tuple,list) or not worker_command
            or any(type(x) is not str or not x for x in worker_command)):
        raise ValueError('explicit worker/resource bindings required')
    if tuple(row['role'] for row in prepared['arms'])!=tuple(_role(*x) for x in ARMS):
        raise ValueError('exact four declared arms required')
    if type(private_manifest_root) is not str or not private_manifest_root.startswith('/'):raise ValueError('private absolute manifest mount required')
    result=[]
    for row in prepared['arms']:
        role=row['role'];labels={'pokezero-phasec':name,'pokezero-arm':role}
        affinity=dict(podAffinity=dict(requiredDuringSchedulingIgnoredDuringExecution=[dict(
            labelSelector=dict(matchLabels={'pokezero-phasec':name}),namespaces=[namespace],topologyKey='kubernetes.io/hostname')]))
        container=dict(name='worker',image=image,command=list(worker_command),args=['--arm',role,'--declaration',private_manifest_root+'/preparation.json'],
            resources=dict(requests={'cpu':str(cpu),'memory':memory,'nvidia.com/gpu':'1'},limits={'cpu':str(cpu),'memory':memory,'nvidia.com/gpu':'1'}),
            volumeMounts=volume_mounts)
        result.append(dict(apiVersion='batch/v1',kind='Job',metadata=dict(name=name+'-'+role.lower(),namespace=namespace,labels=labels,
            annotations={'pokezero.preparation-sha256':_sha(prepared),'pokezero.launch-state':'PREPONLY_B4_GATE_PENDING'}),
            spec=dict(suspend=True,backoffLimit=0,activeDeadlineSeconds=30*24*3600,
                template=dict(metadata=dict(labels=labels),spec=dict(restartPolicy='Never',affinity=affinity,containers=[container],volumes=volumes)))))
    return tuple(result)


@dataclass(frozen=True)
class WorkerBinding:
    """Caller-supplied mounted files, with their raw byte digests.

    These are launch arguments, not a scientific admission receipt. The worker
    must still validate the files, accepted declaration and all launch gates.
    Constructing this value reads no files and allocates no resources.
    """
    declaration:str
    declaration_sha256:str
    runtime:str
    runtime_sha256:str
    evaluation_protocol:str
    evaluation_protocol_sha256:str
    run_root:str

    def __post_init__(self):
        for name in ('declaration','runtime','evaluation_protocol','run_root'):
            value=getattr(self,name)
            if type(value) is not str or not value or not Path(value).is_absolute() or '..' in Path(value).parts:
                raise ValueError('explicit absolute mounted path required:'+name)
        for name in ('declaration_sha256','runtime_sha256','evaluation_protocol_sha256'):
            c._binding(getattr(self,name))

    def argv(self,role):
        if role not in tuple(_role(*x) for x in ARMS):raise ValueError('exact declared role required')
        return ['--enable-v5','--arm',role,'--declaration',self.declaration,
            '--expected-declaration-sha256',self.declaration_sha256,
            '--runtime',self.runtime,'--expected-runtime-sha256',self.runtime_sha256,
            '--run-root',self.run_root,'--evaluation-protocol',self.evaluation_protocol,
            '--expected-evaluation-protocol-sha256',self.evaluation_protocol_sha256]


def worker_job_templates(prepared,*,bindings,**template_arguments):
    """Bind four suspended worker jobs to distinct actual run declarations.

    No preparation document is passed as a run declaration. Suspended state and
    pending gate annotation remain intact; supplying paths does not pass B4 or
    turn an unbound template into permission to execute it.
    """
    roles=tuple(_role(*x) for x in ARMS)
    if type(bindings) is not dict or set(bindings)!=set(roles) or any(type(bindings[x]) is not WorkerBinding for x in roles):
        raise ValueError('explicit binding for each of the four declared roles required')
    if len({bindings[x].run_root for x in roles})!=len(roles):raise ValueError('distinct run roots required')
    if len({bindings[x].declaration for x in roles})!=len(roles):raise ValueError('distinct run declaration paths required')
    if len({bindings[x].declaration_sha256 for x in roles})!=len(roles):raise ValueError('distinct run declaration bytes required')
    jobs=job_templates(prepared,**template_arguments)
    for role,job in zip(roles,jobs):
        binding=bindings[role]
        job['spec']['template']['spec']['containers'][0]['args']=binding.argv(role)
        job['metadata']['annotations']['pokezero.worker-binding-sha256']=_sha(dict(
            (field.name,getattr(binding,field.name)) for field in fields(binding)))
    return jobs
