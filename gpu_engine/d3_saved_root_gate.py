"""The D3 original-40 saved-root gate, re-runnable with or without batched preparation.

Loads the saved cost bank exactly as the original D3 preflight did (runtime, pool, cost root index
checked by sha256, declaration, cost plan and snapshot), builds the original serial Provider and
runs gpu_engine.device_d_frame_parity in its D3 vectorized mode. --batched-prep builds the batch
side's roots with worlds.detach_roots and prices them with FramePricer(host_prep=True); --cuda-leaves
prices the batch side with FramePricer(cuda_leaves=True) (generated CUDA for the per-world engine
and encoder work); the serial reference is the same either way. The serial reference runs from THIS tree (not the bank's original
source commit), so a flag-off run must reproduce the published record before a flag-on run counts.
--compare A B checks that two result files have identical rows (timings excluded). All paths are
command-line arguments; the output directory must be new.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def log(message, **values):
    print(json.dumps(dict(message=message, **values), default=str), flush=True)


def rows_without_timing(result):
    return [{k: v for k, v in row.items() if 'seconds' not in k} for row in result['rows']]


def compare(a, b):
    ra, rb = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    rows_a, rows_b = rows_without_timing(ra), rows_without_timing(rb)
    differing = [i for i, (x, y) in enumerate(zip(rows_a, rows_b)) if x != y]
    report = dict(equal=len(rows_a) == len(rows_b) and not differing and ra['totals'] == rb['totals'],
        statuses=[ra['status'], rb['status']], totals=[ra['totals'], rb['totals']], rows=[len(rows_a), len(rows_b)],
        differing_rows=differing[:10], target_summaries=[ra.get('target_summary'), rb.get('target_summary')])
    log('COMPARED', **report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-root', help='saved cost bank attempt directory (runtime.json, bank/, cost-snapshot.json, prespec.json)')
    parser.add_argument('--index-sha256', help='sha256 of bank/cost-root-index.json')
    parser.add_argument('--out', help='new output directory')
    parser.add_argument('--batched-prep', action='store_true')
    parser.add_argument('--cuda-leaves', action='store_true', help='price the batch side with FramePricer(cuda_leaves=True)')
    parser.add_argument('--compare', nargs=2, metavar=('A', 'B'), help='compare two saved-root-parity.json files and exit')
    args = parser.parse_args(argv)
    if args.compare:return 0 if compare(*args.compare)['equal'] else 1
    if not (args.bank_root and args.index_sha256 and args.out):parser.error('--bank-root, --index-sha256 and --out required')
    root, out = Path(args.bank_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    from pokezero import v5_worker as worker, v5_run as runs
    from pokezero import v5_device_evaluation_bank as schema
    runtime = json.loads((root / 'runtime.json').read_text());pool_sha = runtime['constructor_pool']['sha256']
    profile, vocabulary, gender, category = worker._runtime_metadata(runtime, expected_pool_sha256=pool_sha)
    # worker._source would require this tree to BE the bank's source commit. Here the serial reference
    # runs from this tree on purpose; the flag-off run reproducing the published record is the check.
    log('SOURCE', bank_source_commit=runtime.get('source_commit'), running_source_commit=os.environ.get('V2_SOURCE_COMMIT'))
    worker._memory(runtime['memory_policy'], apply=True)   # must precede the first JAX import
    import jax
    try:worker._probe(runtime)
    except ValueError as error:
        # The probe pins the device count/kind of the original worker; keep its numeric settings.
        log('PROBE_DIFFERS', error=str(error));jax.config.update('jax_enable_x64', False);jax.config.update('jax_default_matmul_precision', 'highest')
    prepared, pool, _, _ = worker.prepare_device_runtime(SimpleNamespace(runtime=runtime, vocabulary=vocabulary, plan=SimpleNamespace(pool_sha256=pool_sha)))
    from gpu_engine import device_d_cost as cost, v5_device_teacher as teacher, device_d_frame_parity as parity
    index = worker._record(dict(path=str(root / 'bank/cost-root-index.json'), sha256=args.index_sha256))
    declaration, batches, _, labels = schema.declaration(index['declaration'])
    prespec = json.loads((root / 'prespec.json').read_text())
    plan = cost.plan_from_bank(profile=profile, labels=labels, category=category, batches=batches, pool_sha256=pool_sha,
        selection_seed=declaration['cost_recipe_seed'], run_id=prespec['cost_run_id'], max_requests=declaration['max_requests'])
    snapshot = runs.Snapshot(**json.loads((root / 'cost-snapshot.json').read_text()))
    provider = teacher.Provider(prepared, plan, batch_joint=True)
    def progress(record):
        with (out / 'parity-progress.jsonl').open('a') as f:f.write(json.dumps(record, allow_nan=False) + '\n')
        log('SERIAL_PARITY_SLOT', ordinal=record['ordinal'], status=record['status'])
    source = os.environ.get('V2_SOURCE_COMMIT') or ''
    result = parity.run(provider, snapshot, snapshot.archive, index, execution_source=source, enable_parity=True,
        progress_callback=progress, chunk_batch=True, vectorized=True, batched_prep=args.batched_prep, cuda_leaves=args.cuda_leaves)
    (out / 'saved-root-parity.json').write_text(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
    log('SAVED_ROOT_PARITY_COMPLETE', status=result['status'], totals=result['totals'], batching=result['batching'],
        cuda_leaves=result.get('cuda_leaves'), timings=result['batch_timings'])
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':raise SystemExit(main())
