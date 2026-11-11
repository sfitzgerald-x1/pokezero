"""One immutable ladder search/suffix cell through original CPU panel/scorer."""
import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import sys
import time
from . import phase_d_ladder_panel as common


async def worker(path,digest,*,runtime,expected_pair):
    request=common.read_bound(path,digest);keys={'schema','run_directory','declaration','declaration_sha256','cell','checkpoint','checkpoint_sha256','cpu_pair','base_config','server'}
    if set(request)not in (keys,keys|{'promotion'})or request['schema']!='pokezero.champion-search-ladder.search-request.v1':raise ValueError('fixed ladder search request required')
    declaration=common.validate(common.read_bound(request['declaration'],request['declaration_sha256']));cell=common.checked_cell(declaration,request['cell'])
    if tuple(request['cpu_pair'])!=tuple(expected_pair):raise ValueError('persistent core pair differs')
    if cell['pair_ordinal']>=500:
        binding=request.get('promotion')
        if type(binding)is not dict or set(binding)!={'path','sha256'}:raise ValueError('fixed screen promotion artifact required')
        promotion=common.read_bound(binding['path'],binding['sha256'])
        if (promotion.get('declaration_sha256')!=request['declaration_sha256']or promotion.get('decision')!='PROMOTE_SAME_DECLARED_PREFIX_TO_2000'or promotion.get('paired_cells')!=500
                or promotion.get('rung')!=cell['arm']and cell['arm']!='RAW'):
            raise ValueError('same original500screen promotion required')
    if request['checkpoint_sha256']!=declaration['checkpoint_sha256']:raise ValueError('untouched champion evaluator required')
    root=Path(request['run_directory']).resolve()
    if Path(path).resolve()!=root/'requests'/f"{cell['ordinal']}.json":raise ValueError('same fixed-cell request path required')
    if cell['arm']=='RAW':
        if request['server']is not None:raise ValueError('RAW suffix uses original raw policy')
    elif request['server'].get('rung')!=cell['arm']:raise ValueError('same declared GPU rung required')
    common.write_new(root/'claims'/f"{cell['ordinal']}.json",request)
    common.write_new(root/'starts'/f"{cell['ordinal']}.json",dict(request_sha256=digest,cell=cell))
    config=common.configuration(request['base_config'],cell,checkpoint=request['checkpoint'],cpu_pair=expected_pair)
    from .live_protocol_runtime import _checkpoint_sha
    if _checkpoint_sha(config.checkpoint)!=request['checkpoint_sha256']:raise ValueError('original champion bytes differ')
    started=time.monotonic();game=document=None
    try:
        with common.owning_capture(root/'owning-wire',cell=cell,declaration_sha256=request['declaration_sha256']):result=await runtime.run(config)
        document=result.to_dict()
        if (result.config!=config or result.checkpoint_sha256!=request['checkpoint_sha256']or result.foulplay_random_seed_schedule!=(cell['fp_seed'],)or len(result.games)!=1):
            raise ValueError('literal one-cell original runtime result required')
        game=result.games[0].to_dict()
        headers=dict(checkpoint_sha256=request['checkpoint_sha256'],policy_mode='raw',seed_start=cell['seed'],
            pokezero_player=config.pokezero_player,encoder_kind=config.encoder_kind,capture_driver=config.capture_driver,format_id=config.format_id)
        if (game.get('seed')!=cell['seed']or document.get('game_results')!=[game]
                or any(document.get(k)!=v for k,v in headers.items())
                or document.get('foulplay_random_seed_schedule',{}).get('seeds')!=[cell['fp_seed']]):raise ValueError('original cell/scored game/serialized benchmark join required')
        from .v5_foulplay import _score
        score=_score(config,game);value=dict(status='COMPLETE'if score is not None else'UNKNOWN',score=score,error=None)
    except Exception as error:value=dict(status='UNKNOWN',score=None,error=type(error).__name__+': '+str(error))
    value.update(schema=common.CELL_SCHEMA,request_sha256=digest,declaration_sha256=request['declaration_sha256'],
        cell=cell,checkpoint_sha256=request['checkpoint_sha256'],cpu_pair=list(expected_pair),duration_seconds=time.monotonic()-started,
        source_game=game,benchmark=document,acting_axis=cell['arm'],transport_policy_mode='raw',
        search_server=request['server'],search_telemetry=None if cell['arm']=='RAW'else str(root/'search-telemetry'/f"{cell['ordinal']}.jsonl"))
    common.write_new(root/'results'/f"{cell['ordinal']}.json",value);return 0


async def serve(pair,cache_directory):
    from .b4_canary_endpoint import pin_owned_actor
    pin_owned_actor(pair);os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    import jax,torch
    jax.config.update('jax_enable_x64',False);jax.config.update('jax_default_matmul_precision','highest')
    from .owning_worker_runtime import ChampionWorkerRuntime,configure_compilation_cache
    configure_compilation_cache(cache_directory);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtimes={};output=sys.stdout
    for line in sys.stdin:
        packet=json.loads(line)
        if set(packet)!={'path','sha256'}:raise ValueError('one fixed-cell owned request required')
        request=common.read_bound(packet['path'],packet['sha256']);arm=request['cell']['arm'];root=Path(request['run_directory'])
        if arm=='RAW':runtime=runtimes.setdefault(arm,ChampionWorkerRuntime())
        else:
            from .phase_d_ladder_transport import Client
            from .phase_d_ladder_actor import Runtime
            server=request['server'];binding=(arm,server['host'],server['port'])
            if binding not in runtimes:runtimes[binding]=Runtime(client=Client(server['host'],server['port'],rung=arm),telemetry_path=root/'search-telemetry'/f"{request['cell']['ordinal']}.jsonl")
            runtime=runtimes[binding]
            # Fresh cell log path with the same immutable model/compiled holder.
            runtime.telemetry_path=root/'search-telemetry'/f"{request['cell']['ordinal']}.jsonl";runtime.factory=None
        with contextlib.redirect_stdout(sys.stderr):code=await worker(packet['path'],packet['sha256'],runtime=runtime,expected_pair=pair)
        output.write(json.dumps({'exit':code})+'\n');output.flush()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-pair',nargs=2,type=int,required=True);p.add_argument('--cache-directory',required=True)
    args=p.parse_args();asyncio.run(serve(tuple(args.cpu_pair),args.cache_directory))
