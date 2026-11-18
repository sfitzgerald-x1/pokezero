"""Profile the real V2 distributed learner on SHA-bound frozen collector rows.

This is a cost diagnostic: updated weights are discarded. Paths and rank
placement come from the caller. Profiling ranges wrap existing methods and do
not replace loss, backward, collectives, clipping or optimizer implementations.
Chrome traces stay at the caller's output path; copy only summary JSON locally.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import fields, replace
import datetime
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def compare_statistics(reference, candidate, *, atol=2e-5, rtol=2e-4):
    """Fail closed on missing fields, nonfinite values, or changed accounting."""
    exact = {'optimizer_step', 'owner_rows', 'learning_rate_epoch', 'epoch', 'epoch_step'}
    if not reference or len(reference) != len(candidate):
        return dict(passed=False, error='nonempty equal update counts required')
    failures = []
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if expected.keys() != actual.keys():
            failures.append(dict(update=index, error='metric fields differ'))
            continue
        for name, value in expected.items():
            other = actual[name]
            if (isinstance(value, bool) or isinstance(other, bool)
                    or not isinstance(value, (int, float)) or not isinstance(other, (int, float))
                    or not math.isfinite(value) or not math.isfinite(other)):
                failures.append(dict(update=index, field=name, error='finite numeric metric required'))
                continue
            if name in exact or name.endswith('_count') or name == 'learning_rate':
                allowed = 0.
            else:
                allowed = atol + rtol * abs(value)
            delta = abs(other - value)
            if delta > allowed:
                failures.append(dict(update=index, field=name, expected=value, actual=other,
                    absolute_error=delta, allowed_error=allowed))
    return dict(passed=not failures, atol=atol, rtol=rtol, updates=len(reference), failures=failures)


@contextmanager
def annotated(trainer):
    """Install reversible profiling labels without changing update arithmetic."""
    import torch

    def label(function, name):
        @wraps(function)
        def invoke(*args, **kwargs):
            with torch.profiler.record_function('v2/' + name):
                return function(*args, **kwargs)
        return invoke

    with ExitStack() as stack:
        for name, region in (('_step', 'update'), ('_forward_backward', 'forward_backward'),
                ('_call', 'forward_loss'), ('_all_reduce_gradients', 'gradient_allreduce'),
                ('_sum', 'scalar_allreduce'), ('_batch', 'batch'), ('_flush', 'report_flush')):
            if hasattr(trainer, name):
                stack.enter_context(patch.object(trainer, name, label(getattr(trainer, name), region)))
        optimizer = trainer.runtime.optimizer
        stack.enter_context(patch.object(optimizer, 'step', label(optimizer.step, 'optimizer')))
        stack.enter_context(patch.object(torch.autograd, 'backward', label(torch.autograd.backward, 'backward')))
        stack.enter_context(patch.object(torch.nn.utils, 'clip_grad_norm_',
            label(torch.nn.utils.clip_grad_norm_, 'gradient_clip')))
        yield


def compile_counters():
    from torch._dynamo.utils import counters
    return {name: {str(key): value for key, value in counts.items() if isinstance(value, int)}
        for name, counts in counters.items() if name in ('stats', 'frames', 'graph_break', 'inductor')}


def counter_delta(before, after):
    return {group: {key: after.get(group, {}).get(key, 0) - before.get(group, {}).get(key, 0)
        for key in sorted(before.get(group, {}).keys() | after.get(group, {}).keys())}
        for group in sorted(before.keys() | after.keys())}


def device_activity_summary(events, cuda_type, updates):
    """Sum actual CUDA events, excluding GPU copies of user annotation ranges.

    CUDA graph annotations also have DeviceType.CUDA. Device type alone would
    count an enclosing range and its kernels twice. Fail closed if annotation
    metadata is absent. Copies remain device activities, not compute kernels.
    """
    totals = {}
    for event in events:
        if event.device_type != cuda_type:
            continue
        annotation = getattr(event, 'is_user_annotation', None)
        if annotation is None:
            raise ValueError('CUDA event lacks annotation classification')
        if annotation:
            continue
        duration = getattr(event, 'self_device_time_total', getattr(event, 'self_cuda_time_total', 0.))
        record = totals.setdefault(event.key, dict(calls=0, microseconds=0.))
        record['calls'] += event.count
        record['microseconds'] += duration
    return dict(device_activity_ms_per_update=sum(value['microseconds'] for value in totals.values())/1000/updates,
        device_activity_calls_per_update=sum(value['calls'] for value in totals.values())/updates,
        includes_device_copies=True, excludes_user_annotations=True, summed_time_is_not_union=True,
        top_device_activities=[dict(name=name, calls_per_update=value['calls']/updates,
            device_ms_per_update=value['microseconds']/1000/updates)
            for name, value in sorted(totals.items(), key=lambda pair:pair[1]['microseconds'], reverse=True)[:20]])


def profile_updates(trainer, store, window, updates, trace):
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    before = compile_counters()
    torch.cuda.synchronize()
    with annotated(trainer), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        result = trainer.train(store, window=window, max_updates=updates)
        torch.cuda.synchronize()
    if len(result['updates']) != updates:
        raise ValueError('profiling reached the end of the row store before its update budget')
    averages = prof.key_averages()
    ranges = [event for event in averages if event.key.startswith('v2/')]

    def device_us(event):
        return getattr(event, 'device_time_total', getattr(event, 'cuda_time_total', 0.))

    prof.export_chrome_trace(str(trace))
    return dict(updates=updates, trace=str(trace),
        ranges=[dict(name=event.key, calls=event.count, cpu_ms_per_update=event.cpu_time_total/1000/updates,
            device_type=str(event.device_type), device_ms_per_update=device_us(event)/1000/updates) for event in ranges],
        ranges_are_nested_not_additive=True,
        cuda_graph_range_attribution_is_not_complete_forward_backward_timing=True,
        **device_activity_summary(prof.events(), DeviceType.CUDA, updates),
        compile_counter_delta=counter_delta(before, compile_counters()))


def shard_rows(data, rank, world):
    """Keep original roster/bindings while selecting disjoint game-owned rows."""
    import torch
    streams = [stream for batch in data['batches'] for stream in batch]
    if len(set(streams)) != len(streams) or len(streams) % world:
        raise ValueError('unique declared games divisible across ranks required')
    keys = data['keys']
    declared = torch.tensor(streams, dtype=keys.dtype)
    if not bool(torch.isin(keys[:, 0], declared).all()):
        raise ValueError('row outside the declared frozen game roster')
    owned = declared[len(streams)*rank//world:len(streams)*(rank+1)//world]
    selected = torch.nonzero(torch.isin(keys[:, 0], owned)).flatten()
    if not len(selected):
        raise ValueError('empty frozen-input rank')
    values = {}
    for name, value in data['tensors'].items():
        if isinstance(value, torch.Tensor):
            if value.shape[0] != len(keys):
                raise ValueError('non-row tensor: ' + name)
            values[name] = value.index_select(0, selected)
        elif name == 'terminal_status':
            values[name] = tuple(value[int(index)] for index in selected)
        else:
            values[name] = value
    return values, keys.index_select(0, selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--rows-sha256', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--local-batch', type=int, default=128)
    parser.add_argument('--precision', choices=('float32', 'bfloat16'), default='bfloat16')
    parser.add_argument('--mode', choices=('eager', 'compiled', 'cudagraphs'), default='cudagraphs')
    parser.add_argument('--optimizer', choices=('fused', 'foreach', 'single'), default='fused')
    parser.add_argument('--role-routing', choices=('indexed', 'sliced'), default='indexed')
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--updates', type=int, default=64)
    parser.add_argument('--profile', type=int, default=4)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, choices=(1, 4), default=1)
    parser.add_argument('--rendezvous', type=Path)
    args = parser.parse_args(argv)
    if (args.local_batch <= 0 or args.warmup < 0 or args.updates <= 0 or args.profile < 0
            or not 0 <= args.rank < args.world_size):
        parser.error('valid batch/update budgets and rank required')
    if sha256(args.rows) != args.rows_sha256 or sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('frozen-input/checkpoint SHA256 mismatch')

    import os
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='4')
    import torch
    import torch.distributed as dist
    from pokezero import v2_distributed as vd, v2_trainer as v2, v5_training
    from pokezero.v5_loss import V5LossConfig
    from pokezero.v5_model import V5ModelConfig
    if torch.cuda.device_count() != 1:
        raise ValueError('each process must see exactly its assigned GPU')
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    group = None
    if args.world_size > 1:
        if args.rendezvous is None:
            parser.error('four ranks require a shared rendezvous path')
        dist.init_process_group('nccl', init_method='file://' + str(args.rendezvous.resolve()),
            rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=300))
        group = dist.group.WORLD
    try:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        metadata = payload['metadata']
        model = V5ModelConfig.from_metadata(metadata['model'])
        loss = V5LossConfig(**{field.name:metadata['loss'][field.name] for field in fields(V5LossConfig)})
        train = v5_training.TrainConfig(**metadata['training'])
        provenance = dict(steps=metadata.get('steps'), rows=metadata.get('rows'))
        if (model.width, model.layers, model.dropout) != (512, 3, .1):
            raise ValueError('production width512/layers3/dropout0.1 required')
        train = replace(train, batch_size=args.local_batch*args.world_size, precision=args.precision)
        config = v2.V2Config(mode=args.mode, optimizer=args.optimizer, distributed_rows=True,
            role_routing=args.role_routing)
        runtime = v2.create(model, loss, train, config, enable_v5=True, device='cuda', process_group=group)
        runtime.model.load_state_dict(payload['state_dict'])
        v2._load_optimizer_state(runtime.optimizer, payload['optimizer_state'])
        runtime.rng = payload['v2']['rng_by_rank'][args.rank]
        del payload
        if any(parameter.dtype != torch.float32 for parameter in runtime.model.parameters()):
            raise ValueError('fp32 master weights required')
        started = time.perf_counter()
        data = torch.load(args.rows, map_location='cpu', weights_only=True)
        window_data = data['window']
        tensors, keys = shard_rows(data, args.rank, args.world_size)
        tensors = {name: value.cuda() if isinstance(value, torch.Tensor) else value for name, value in tensors.items()}
        store = v2.row_store(tensors, model, loss, store_id=args.rows_sha256)
        store = vd.distribute_store(store, keys.numpy(), data['batches'], runtime)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        del tensors, keys, data
        trainer = vd.DistributedTrainer(runtime)
        required = args.warmup + args.updates + args.profile
        if len(store.distributed_layout.counts)*train.epochs < required:
            raise ValueError('frozen fixture has insufficient updates;no silent epoch extension or row recycling')
        # Keep the artifact's original declared-game learning-rate window.
        window = v5_training.DeclaredGameWindow(**window_data)
        torch.cuda.reset_peak_memory_stats()
        before = compile_counters()
        started = time.perf_counter()
        warm = trainer.train(store, window=window, max_updates=args.warmup)
        torch.cuda.synchronize()
        warm_seconds = time.perf_counter() - started
        if len(warm['updates']) != args.warmup:
            raise ValueError('insufficient warmup updates')
        after_warm = compile_counters()
        torch.cuda.synchronize()
        started = time.perf_counter()
        timed = trainer.train(store, window=window, max_updates=args.updates)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if len(timed['updates']) != args.updates:
            raise ValueError('insufficient timed updates')
        after_timed = compile_counters()
        mode = trainer.mode_report()
        if mode['active'] != args.mode or mode['fallbacks']:
            raise ValueError('requested execution path fell back: ' + str(mode))
        args.out.mkdir(parents=True, exist_ok=True)
        profiling = profile_updates(trainer, store, window, args.profile,
            args.out/f'trace-{args.rank}.json') if args.profile else None
        report = dict(schema='v2-learner-profile.v2', rank=args.rank, world_size=args.world_size,
            local_batch=args.local_batch, global_batch=train.batch_size, model=model.metadata(),
            precision=train.precision, fp32_master_weights=True, mode=mode,
            rows_sha256=args.rows_sha256, checkpoint_sha256=args.checkpoint_sha256,
            checkpoint_provenance=provenance, warmup_updates=args.warmup, warmup_seconds=warm_seconds,
            fixture='saved model/optimizer/rank RNG;fresh cursor/counters on frozen rows and their declared window',
            fixture_is_production_continuation=False,
            updates=args.updates, elapsed_seconds=elapsed, ms_per_update=elapsed*1000/args.updates,
            initial_stats=warm['updates']+timed['updates'], window=window_data,
            load_redistribute_seconds=load_seconds, profile=profiling,
            warm_compile_counter_delta=counter_delta(before, after_warm),
            timed_compile_counter_delta=counter_delta(after_warm, after_timed),
            timed_new_graphs=after_timed.get('stats', {}).get('unique_graphs', 0)
                - after_warm.get('stats', {}).get('unique_graphs', 0),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                uuid=str(torch.cuda.get_device_properties(0).uuid), deterministic=True,
                tf32=False, cublas=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                attention_flags={name:getattr(torch.backends.cuda, name+'_sdp_enabled')()
                    for name in ('flash', 'mem_efficient', 'math', 'cudnn')}),
            includes=['batch selection', 'global loss statistics', 'forward/loss', 'backward',
                'gradient reduction', 'gradient clipping', 'AdamW', 'metrics/finiteness flush', 'cursor/Python'],
            excludes=['collection', 'initial H2D/redistribution', 'checkpoint/export', 'warmup/initial compile'],
            qualification='cost diagnostic only;no exact-resume or frozen-stat gate claimed')
        (args.out/f'result-{args.rank}.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(rank=args.rank, ms_per_update=report['ms_per_update'], updates=args.updates)), flush=True)
    finally:
        if group is not None:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
