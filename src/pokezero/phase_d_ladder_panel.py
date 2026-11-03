"""Fixed champion ladder cells on the original CPU B2/FP1000 runner.

This module creates no declaration or process on import. The RAW screen is the
actual reusable ladder control, not a pilot. Every cell is claimed once before
any numerical work; interruption leaves it unknown and dispatch resumes only
not-attempted cells. Literal owning-wire capture precedes action selection.
"""
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import time

SCHEMA='pokezero.champion-search-ladder.declaration.v1'
CELL_SCHEMA='pokezero.champion-search-ladder.cell.v1'
RUNG_NAMES=('RAW','R1','R3','R4','R7','R5')


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(value):return hashlib.sha256(encoded(value)).hexdigest()


def read_bound(path,digest):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('immutable ladder input differs')
    return json.loads(raw)


def validate(declaration):
    from .live_protocol_actor import ActorContract
    if (declaration.get('schema')!=SCHEMA or declaration.get('checkpoint_sha256')!=ActorContract.champion().checkpoint_sha256
            or declaration.get('max_paired_cells')!=2000 or declaration.get('screen_paired_cells')!=500
            or declaration.get('seeds')!=1000 or declaration.get('seats')!=[0,1]
            or declaration.get('foulplay_ms')!=1000 or declaration.get('learner_limit_seconds')!=60
            or declaration.get('confidence')!=.9 or declaration.get('screen_threshold_pp')!=3.
            or declaration.get('root_operator')!='ARGMAX_MEAN_COMPLETED_Q_COMMON_ALL_RUNGS'):
        raise ValueError('original fixed champion500/2000 paired ladder declaration required')
    for name in ('battle_seed_start','foulplay_seed_start','stream_start'):
        value=declaration.get(name)
        if type(value)is not int or not 0<value<2**31-12000:raise ValueError('literal fixed ladder seed/address required')
    expected=(('R1',2,1,1,'VALUE'),('R3',32,1,1,'VALUE'),('R4',32,8,1,'VALUE'),
              ('R7',32,1,2,'VALUE'),('R5',8,1,1,'ROLLOUT'))
    if tuple((r['name'],r['worlds'],r['chance_samples'],r['depth'],r['leaf'])for r in declaration['rungs'])!=expected:
        raise ValueError('declared breadth/chance/depth/rollout rung sequence differs')
    if declaration['rungs'][-1].get('rollout_leaves')!=8:raise ValueError('eight rollout leaves required')
    return declaration


def cells(declaration,arm,*,limit=2000):
    validate(declaration)
    if arm not in RUNG_NAMES or limit not in (500,2000):raise ValueError('fixed500 screen or2000 prefix required')
    axis=RUNG_NAMES.index(arm)
    return tuple(dict(ordinal=6*pair+axis,pair_ordinal=pair,offset=pair//2,seat=pair%2,arm=arm,
        seed=declaration['battle_seed_start']+pair//2,fp_seed=declaration['foulplay_seed_start']+pair//2)
        for pair in range(limit))


def checked_cell(declaration,cell):
    if type(cell)is not dict or type(cell.get('pair_ordinal'))is not int or not 0<=cell['pair_ordinal']<2000:
        raise ValueError('original fixed ladder cell required')
    expected=cells(declaration,cell.get('arm'))[cell['pair_ordinal']]
    if cell!=expected:raise ValueError('fixed ladder cell address/seed/seat differs')
    return cell


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb')as stream:stream.write(encoded(value));stream.flush();os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)


def configuration(base,cell,*,checkpoint,cpu_pair):
    """All scientific/FP controls retained; only declared cell fields vary."""
    from .foulplay_bridge import ControlledFoulPlayConfig
    from .live_protocol_bridge import validate_encoder_config
    if (type(cpu_pair)not in (tuple,list)or len(cpu_pair)!=2 or cpu_pair[0]==cpu_pair[1]
            or any(type(x)is not int or x<0 for x in cpu_pair)):
        raise ValueError('one dedicated actor/FP physical-core pair required')
    values=dict(base,checkpoint=checkpoint,games=1,seed_start=cell['seed'],foulplay_random_seed=cell['fp_seed'],
        pokezero_player=('p1','p2')[cell['seat']],foulplay_cpu_affinity=cpu_pair[1])
    for field in fields(ControlledFoulPlayConfig):
        if values.get(field.name)is not None and 'Path'in str(field.type):values[field.name]=Path(values[field.name])
    config=ControlledFoulPlayConfig(**values);validate_encoder_config(config)
    if config.encoder_kind!='jax-owning' or config.owning_decision_timeout_seconds!=60.:
        raise ValueError('B2 protocol-fed encoder and original60-second learner required')
    return config


@contextmanager
def owning_capture(directory,*,cell,declaration_sha256):
    """Same actor factory and returns; one game per dedicated worker process.

    The temporary return hook changes neither provider/contract nor model/RNG.
    It is installed within the cell worker, so it crosses no process boundary.
    The original bridge's to_thread copies this exact capture ContextVar.
    """
    from .phase_d_owning_capture import capture_actor,capture_to
    from .live_protocol_bridge import OwningActorFactory
    original=OwningActorFactory.actor
    def actor(self,*args,**kwargs):return capture_actor(original(self,*args,**kwargs))
    OwningActorFactory.actor=actor
    try:
        with capture_to(directory,cell=cell,declaration_sha256=declaration_sha256):yield
    finally:
        if OwningActorFactory.actor is not actor:raise ValueError('owning capture factory changed during one cell')
        OwningActorFactory.actor=original


async def raw_worker(request_path,expected_sha256,*,runtime=None,expected_pair=None):
    """Run exactly one RAW cell using unchanged benchmark and terminal scorer."""
    request=read_bound(request_path,expected_sha256)
    required={'schema','run_directory','declaration','declaration_sha256','cell','checkpoint','checkpoint_sha256','cpu_pair','base_config'}
    if set(request)!=required or request['schema']!='pokezero.champion-search-ladder.raw-request.v1':
        raise ValueError('literal raw ladder request required')
    root=Path(request['run_directory']).resolve();declaration=validate(read_bound(request['declaration'],request['declaration_sha256']))
    cell=checked_cell(declaration,request['cell'])
    if cell['arm']!='RAW' or cell['pair_ordinal']>=500:raise ValueError('only original RAW500 screen enabled')
    if request['checkpoint_sha256']!=declaration['checkpoint_sha256']:raise ValueError('untouched9375 control required')
    pair=tuple(request['cpu_pair'])
    if expected_pair is None:
        from .b4_canary_endpoint import pin_owned_actor
        pin_owned_actor(pair)
    elif pair!=tuple(expected_pair):raise ValueError('persistent worker CPU pair changed')
    if Path(request_path).resolve()!=root/'requests'/f"{cell['ordinal']}.json":raise ValueError('owned fixed-cell request path differs')
    # Claim and durable START precede model loading, capture, or game dispatch.
    write_new(root/'claims'/f"{cell['ordinal']}.json",request)
    write_new(root/'starts'/f"{cell['ordinal']}.json",dict(request_sha256=expected_sha256,cell=cell))
    config=configuration(request['base_config'],cell,checkpoint=request['checkpoint'],cpu_pair=pair)
    from .live_protocol_runtime import _checkpoint_sha
    if _checkpoint_sha(config.checkpoint)!=request['checkpoint_sha256']:raise ValueError('original champion bytes differ')
    from .owning_worker_runtime import ChampionWorkerRuntime
    from . import v5_foulplay
    runtime=ChampionWorkerRuntime()if runtime is None else runtime
    started=time.monotonic();document=game=None
    try:
        with owning_capture(root/'owning-wire',cell=cell,declaration_sha256=request['declaration_sha256']):
            result=await runtime.run(config)
        document=result.to_dict()
        if (result.config!=config or result.checkpoint_sha256!=request['checkpoint_sha256']
                or result.foulplay_random_seed_schedule!=(config.foulplay_random_seed,)or len(result.games)!=1):
            raise ValueError('original one-cell benchmark binding differs')
        game=result.games[0].to_dict()
        expected_header=dict(checkpoint_sha256=request['checkpoint_sha256'],policy_mode='raw',
            seed_start=cell['seed'],pokezero_player=('p1','p2')[cell['seat']])
        for key in ('encoder_kind','capture_driver','format_id'):
            if hasattr(config,key):expected_header[key]=getattr(config,key)
        if (game.get('seed')!=cell['seed'] or game.get('seed')!=config.seed_start
                or document.get('game_results')!=[game]
                or any(document.get(key)!=value for key,value in expected_header.items())
                or document.get('foulplay_random_seed_schedule',{}).get('seeds')!=[cell['fp_seed']]):
            raise ValueError('literal scored game/benchmark must join original fixed cell and controlled header')
        score=v5_foulplay._score(config,game)
        value=dict(status='COMPLETE'if score is not None else'UNKNOWN',score=score,error=None)
    except Exception as error:value=dict(status='UNKNOWN',score=None,error=type(error).__name__+': '+str(error))
    value.update(schema=CELL_SCHEMA,request_sha256=expected_sha256,declaration_sha256=request['declaration_sha256'],
        cell=cell,checkpoint_sha256=request['checkpoint_sha256'],cpu_pair=list(pair),duration_seconds=time.monotonic()-started,
        source_game=game,benchmark=document)
    write_new(root/'results'/f"{cell['ordinal']}.json",value)
    return 0


async def serve_raw(pair,cache_directory):
    """Persistent CPU-only original model/compiled-fold holder, fresh game state."""
    from .b4_canary_endpoint import pin_owned_actor
    pin_owned_actor(pair)
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    import jax,torch,sys,contextlib
    jax.config.update('jax_enable_x64',False);jax.config.update('jax_default_matmul_precision','highest')
    from .owning_worker_runtime import ChampionWorkerRuntime,configure_compilation_cache
    configure_compilation_cache(cache_directory)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime=ChampionWorkerRuntime();output=sys.stdout
    for line in sys.stdin:
        request=json.loads(line)
        if set(request)!={'path','sha256'}:raise ValueError('one owned fixed-cell request required')
        with contextlib.redirect_stdout(sys.stderr):
            code=await raw_worker(request['path'],request['sha256'],runtime=runtime,expected_pair=pair)
        output.write(json.dumps({'exit':code})+'\n');output.flush()


if __name__=='__main__':
    import argparse,asyncio
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-pair',nargs=2,type=int,required=True)
    parser.add_argument('--cache-directory',required=True)
    args=parser.parse_args();asyncio.run(serve_raw(tuple(args.cpu_pair),args.cache_directory))
