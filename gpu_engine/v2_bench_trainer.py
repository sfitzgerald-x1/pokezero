"""V2-c trainer benchmark: warm rows/s and GPU kernel time from a device row store.

Loads real v5 caches into a GPU row store with ``v2_trainer.load_cache_store``.
Model/loss/optimizer configuration and initial weights come from an existing v5
checkpoint. Then, for each batch size and execution mode, it measures warm
updates and profiles a few of them. All paths are CLI arguments (glob
patterns are expanded here). Nothing is written outside ``--out``.

Example:
  python -m gpu_engine.v2_bench_trainer --caches '<run>/caches/cache-*' --max-caches 2 \\
      --checkpoint '<run>/artifacts/train-0-*.pt' --runtime '<run>/runtime.json' --out /out
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import gc
import glob
import json
import math
from pathlib import Path
import time


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def expand(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f'no path matches {pattern!r}')
    return matches


def configurations(checkpoint):
    import torch
    from pokezero.v5_loss import V5LossConfig
    from pokezero.v5_model import V5ModelConfig
    from pokezero import v5_training
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    model = V5ModelConfig.from_metadata(metadata['model'])
    loss = V5LossConfig(**{field.name: metadata['loss'][field.name] for field in fields(V5LossConfig)})
    train = v5_training.TrainConfig(**metadata['training'])
    return model, loss, train, payload['state_dict'], dict(steps=metadata.get('steps'), rows=metadata.get('rows'))


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def kernel_profile(trainer, store, window, updates):
    import torch
    from torch.profiler import ProfilerActivity, profile
    synchronize()
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=activities) as prof:
        report = trainer.train(store, window=window, max_updates=updates)
        synchronize()
    # key_averages lists each op *and* the kernels it launched; count device events only.
    from torch.autograd import DeviceType
    averages = prof.key_averages()
    attribute = 'self_device_time_total' if hasattr(averages[0], 'self_device_time_total') else 'self_cuda_time_total'
    kernels = [item for item in averages if item.device_type in (DeviceType.CUDA, getattr(DeviceType, 'PrivateUse1', None))]
    total_us = sum(getattr(item, attribute) for item in kernels)
    ranked = sorted(kernels, key=lambda item: getattr(item, attribute), reverse=True)[:25]
    top = [dict(name=item.key[:160], device_ms_per_update=getattr(item, attribute) / 1e3 / updates,
        calls_per_update=item.count / updates) for item in ranked]
    operators = sorted((item for item in averages if item.device_type == DeviceType.CPU),
        key=lambda item: getattr(item, attribute), reverse=True)[:15]
    top_ops = [dict(name=item.key[:120], device_ms_per_update=getattr(item, attribute) / 1e3 / updates,
        calls_per_update=item.count / updates) for item in operators]
    return dict(updates=len(report['updates']), kernel_ms_per_update=total_us / 1e3 / updates,
        kernel_launches_per_update=sum(item.count for item in kernels) / updates, top=top, top_operators=top_ops)


def parity(store, model, loss, train, state, batch_size, device, emulate):
    """Dropout-free loss/gradient parity on one real batch: original vs static eager vs compiled.

    With dropout 0 this runtime's SDPA dispatcher prefers cuDNN attention, which
    fails here (sublibrary version mismatch). Training (dropout > 0) uses the
    memory-efficient kernel, so parity pins that backend for all three paths.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from pokezero import v2_trainer as v2
    torch._dynamo.reset()
    quiet = replace(model, dropout=0.)
    runtime = v2.create(quiet, loss, replace(train, batch_size=batch_size), enable_v5=True, device=device)
    runtime.model.load_state_dict(state); runtime.model.train()
    batch = {name: value[:batch_size] for name, value in store.tensors.items()}
    experts = v2.expert_indices(runtime.model, device); counts = v2._variant_counts(quiet, device)
    compiled = torch.compile(v2._static_forward_loss, dynamic=False, options={'emulate_precision_casts': emulate})
    results = {}
    for name in ('original', 'static', 'compiled'):
        runtime.model.zero_grad(set_to_none=True)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            total = _parity_loss(name, runtime, batch, store, quiet, loss, train, device, experts, counts, compiled, batch_size)
            total.backward()
        results[name] = (float(total), torch.cat([p.grad.reshape(-1).float() for p in runtime.model.parameters()]))
    reference_loss, reference = results['original']
    scale = reference.norm().item()
    report = {}
    for name, (value, gradient) in results.items():
        report[name] = dict(loss=value, loss_abs_diff=abs(value - reference_loss),
            gradient_rel_l2_diff=(gradient - reference).norm().item() / scale,
            gradient_cosine=torch.nn.functional.cosine_similarity(gradient, reference, dim=0).item())
    del runtime, compiled
    torch._dynamo.reset()
    return dict(batch_size=batch_size, dropout_disabled_for_parity=True, sdpa_backends=['EFFICIENT_ATTENTION', 'MATH'],
        gradient_norm=scale, **report)


def _parity_loss(name, runtime, batch, store, quiet, loss, train, device, experts, counts, compiled, batch_size):
    import torch
    from pokezero import v2_trainer as v2
    from pokezero.v5_loss import v5_loss
    from pokezero.v5_model import forward_v5
    if name == 'original':
        tensors = dict(batch, **store.bindings, terminal_status=store.terminal_status[:batch_size])
        with v2._autocast(torch, device, train.precision):
            return v5_loss(forward_v5(runtime.model, tensors, include_auxiliary=quiet.treatment), tensors, quiet, loss)[0]
    function = v2._static_forward_loss if name == 'static' else compiled
    return function(runtime.model, batch, quiet, loss, train.precision, device, experts, counts)[0]


def measure(args, store, model, loss, train, state, batch_size, mode, window):
    import torch
    from pokezero import v2_trainer as v2
    torch._dynamo.reset(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    config = replace(train, batch_size=batch_size)
    runtime = v2.create(model, loss, config, v2.V2Config(mode=mode, order=args.order, check_interval=args.check_interval,
        optimizer=args.optimizer), enable_v5=True, device=args.device)
    runtime.model.load_state_dict(state)
    trainer = v2.Trainer(runtime)
    begin = time.perf_counter()
    warm = trainer.train(store, window=window, max_updates=args.warmup)
    synchronize(); warm_seconds = time.perf_counter() - begin
    first_losses = [u['loss'] for u in warm['updates'][:3]]
    rows_before = runtime.rows
    synchronize(); begin = time.perf_counter()
    timed = trainer.train(store, window=window, max_updates=args.measure)
    synchronize(); seconds = time.perf_counter() - begin
    rows = runtime.rows - rows_before; updates = len(timed['updates'])
    profile = kernel_profile(trainer, store, window, args.profile) if args.profile else None
    counters = {}
    try:
        from torch._dynamo.utils import counters as dynamo
        counters = {group: dict(values) for group, values in dynamo.items() if group in ('stats', 'graph_break', 'inductor')}
        counters['inductor'] = {k: v for k, v in counters.get('inductor', {}).items() if 'cudagraph' in k or 'fallback' in k}
    except Exception:
        pass
    result = dict(batch_size=batch_size, requested_mode=mode, mode=trainer.mode_report(), updates=updates, rows=rows,
        seconds=seconds, rows_per_second=rows / seconds, ms_per_update=1e3 * seconds / updates,
        warmup_updates=len(warm['updates']), warmup_seconds=warm_seconds, first_losses=first_losses,
        last_loss=timed['updates'][-1]['loss'], peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.,
        profile=profile, dynamo_counters=counters)
    if profile:
        result['gpu_busy_fraction'] = profile['kernel_ms_per_update'] / result['ms_per_update']
    del trainer, runtime
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--caches', required=True, help='glob of v5 cache directories')
    parser.add_argument('--max-caches', type=int, default=1)
    parser.add_argument('--checkpoint', required=True, help='glob; the first match supplies configs and weights')
    parser.add_argument('--runtime', default=None, help='glob of runtime.json for the torch matmul precision')
    parser.add_argument('--batch-sizes', default='512,2048,4096')
    parser.add_argument('--modes', default='eager,compiled')
    parser.add_argument('--order', default='shuffled', choices=('ordered', 'shuffled'))
    parser.add_argument('--optimizer', default='fused', choices=('fused', 'foreach', 'single'))
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--measure', type=int, default=40)
    parser.add_argument('--profile', type=int, default=4)
    parser.add_argument('--check-interval', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    import torch
    from pokezero import v2_trainer as v2, v5_training
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    caches = expand(args.caches)[:args.max_caches]
    checkpoint = expand(args.checkpoint)[0]
    precision = 'highest'
    if args.runtime:
        document = json.loads(Path(expand(args.runtime)[0]).read_text())
        precision = document.get('precision_policy', {}).get('torch_matmul', precision)
    torch.set_float32_matmul_precision(precision)
    device_name = torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else args.device
    log(f'torch {torch.__version__} cuda {torch.version.cuda} device {device_name}; '
        f'matmul precision {precision}')
    model, loss, train, state, provenance = configurations(checkpoint)
    log(f'config arm={model.arm} width={model.width} layers={model.layers} vocab={model.category_vocab_size} '
        f'dropout={model.dropout} batch={train.batch_size} epochs={train.epochs} lr={train.learning_rate} '
        f'schedule={train.learning_rate_schedule} precision={train.precision} max_grad_norm={train.max_grad_norm}')
    begin = time.perf_counter()
    store = v2.load_cache_store(caches, model, loss, device=args.device)
    synchronize(); load_seconds = time.perf_counter() - begin
    store_bytes = sum(t.numel() * t.element_size() for t in store.tensors.values())
    log(f'store rows={store.rows} segments={store.segments} {store_bytes / 2**30:.2f} GiB loaded in {load_seconds:.1f}s')
    window = None
    if train.learning_rate_schedule != 'constant':
        window = v5_training.DeclaredGameWindow(0, 20_000, 2_000_000)
    results = []
    log('parity: original forward_v5/v5_loss vs static eager vs compiled (dropout off)')
    try:
        check = parity(store, model, loss, train, state, max(int(x) for x in args.batch_sizes.split(',')), args.device, True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        check = dict(error=f'{type(error).__name__}: {error}'[:2000])
    log(f'parity {json.dumps(check)}')
    report = dict(parity=check, schema='v2c-trainer-bench.v1', torch=torch.__version__, cuda=torch.version.cuda,
        device=device_name, matmul_precision=precision, caches=len(caches),
        store_rows=store.rows, store_gib=store_bytes / 2**30, load_seconds=load_seconds,
        model=dict(arm=model.arm, width=model.width, layers=model.layers, dropout=model.dropout,
            category_vocab_size=model.category_vocab_size, parameters=sum(v.numel() for v in state.values())),
        train=dict(learning_rate=train.learning_rate, weight_decay=train.weight_decay, max_grad_norm=train.max_grad_norm,
            precision=train.precision, schedule=train.learning_rate_schedule), checkpoint_provenance=provenance,
        order=args.order, optimizer=args.optimizer, results=results)
    for batch_size in (int(x) for x in args.batch_sizes.split(',')):
        for mode in args.modes.split(','):
            log(f'measure batch={batch_size} mode={mode}')
            try:
                result = measure(args, store, model, loss, train, state, batch_size, mode, window)
            except Exception as error:
                import traceback
                traceback.print_exc()
                result = dict(batch_size=batch_size, requested_mode=mode, error=f'{type(error).__name__}: {error}'[:2000])
            results.append(result)
            if 'error' not in result:
                kernel = result['profile']['kernel_ms_per_update'] if result['profile'] else float('nan')
                log(f'  active={result["mode"]["active"]} {result["rows_per_second"]:,.0f} rows/s '
                    f'{result["ms_per_update"]:.2f} ms/update kernel {kernel:.2f} ms/update '
                    f'peak {result["peak_memory_gib"]:.1f} GiB first_losses {result["first_losses"]}')
            (out / 'bench.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    lines = ['| batch | mode (active) | rows/s | ms/update | GPU kernel ms/update | GPU busy | peak GiB |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        if 'error' in r:
            lines.append(f'| {r["batch_size"]} | {r["requested_mode"]} (error) | - | - | - | - | - |'); continue
        kernel = r['profile']['kernel_ms_per_update'] if r['profile'] else math.nan
        lines.append(f'| {r["batch_size"]} | {r["requested_mode"]} ({r["mode"]["active"]}) | {r["rows_per_second"]:,.0f} | '
            f'{r["ms_per_update"]:.2f} | {kernel:.2f} | {r.get("gpu_busy_fraction", math.nan):.0%} | {r["peak_memory_gib"]:.1f} |')
    table = '\n'.join(lines)
    (out / 'bench.md').write_text(table + '\n')
    print(table, flush=True)
    for r in results:
        if r.get('profile'):
            log(f'top kernels batch={r["batch_size"]} mode={r["mode"]["active"]}:')
            for item in r['profile']['top'][:12]:
                print(f'    {item["device_ms_per_update"]:8.3f} ms  x{item["calls_per_update"]:.0f}  {item["name"]}', flush=True)
    passed = any(r.get('batch_size') == 4096 and r.get('rows_per_second', 0) >= 40_000 for r in results)
    log(f'V2-c bar (>= 40,000 rows/s at batch 4096): {"PASS" if passed else "FAIL"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
