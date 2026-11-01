"""Explicit persistent B3 worker transport; no work occurs at import.

One sequential worker owns each declared actor/FP core pair. A durable parent
attempt and exclusive original claim precede every game. Receipts acknowledge
the original runner's completed cleanup, not a new per-game process exit.
"""
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import time

from . import owning_encoder_continuation as continuation
from . import owning_encoder_study as study
from .owning_worker_runtime import ChampionWorkerRuntime, compilation_snapshot, configure_compilation_cache


async def serve(plan, output, pair):
    continuation.validate_plan(plan)
    study.require(plan['model_factory_lifecycle'] == 'PERSISTENT_PER_CORE_WORKER'
        and pair in plan['cpu_pairs'], 'bound persistent worker/core pair required')
    topology = continuation.cpu_allocation(plan['cpu_pairs'])
    os.sched_setaffinity(0, {pair[0]})
    study.require(os.sched_getaffinity(0) == {pair[0]}, 'actor CPU placement failed')
    initialized = time.monotonic()
    # Match the original environment BEFORE the cache setup imports JAX. The
    # unchanged precision routine then configures Torch/JAX and checks devices.
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','POKEZERO_TORCH_NUM_THREADS','POKEZERO_TORCH_NUM_INTEROP_THREADS'):
        os.environ[name] = '1'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    cache = configure_compilation_cache(plan['compilation_cache_directory'])
    precision = study._configure_precision()
    startup_seconds = time.monotonic() - initialized
    spec = study.read_bound(plan['references']['prespec'])
    runtime = ChampionWorkerRuntime()
    completed = 0
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:return 0
        value = json.loads(line)
        study.require(type(value) is dict and set(value) == {'index'}
            and type(value['index']) is int and value['index'] in plan['ordered_indices'],
            'one declared original cell command required')
        index = value['index'];path = output/f'cell-{index}.json'
        continuation.claim_cell(plan, index, pair, path)
        started_epoch, started = time.time(), time.monotonic()
        before = compilation_snapshot(None)
        kind = None
        try:
            config = continuation.single_config(spec, index, pair[1])
            kind = config.encoder_kind
            before = compilation_snapshot(runtime.factories.get(config.encoder_kind))
            result = await runtime.run(config)
            document = result.to_dict()
            continuation.checked_result(document, spec, index)
            after = compilation_snapshot(runtime.factories[config.encoder_kind])
            receipt = dict(index=index, document=document, precision=precision,
                cpu_pair=pair, cpu_topology=topology,
                timing=dict(started_epoch=started_epoch, completed_epoch=time.time(),
                    duration_seconds=time.monotonic()-started),
                worker_compilation=dict(before=before, after=after, game_ordinal=completed,
                    initialization_seconds=startup_seconds, cache=cache,
                    lifecycle='PERSISTENT_PER_CORE_WORKER_FRESH_GAME_STATE'))
        except BaseException as error:
            capped = isinstance(error, study.CappedGame) or (isinstance(error, RuntimeError)
                and str(error) == 'owning game round cap without actual owner terminal')
            receipt = dict(index=index, status='CAPPED' if capped else 'FAILED',
                failure=dict(error_type=type(error).__name__, **study._failure_diagnostics(error)),
                timing=dict(started_epoch=started_epoch, completed_epoch=time.time(),duration_seconds=time.monotonic()-started),
                worker_compilation=dict(before=before,after=compilation_snapshot(runtime.factories.get(kind)),
                    game_ordinal=completed,initialization_seconds=startup_seconds,cache=cache,
                    lifecycle='PERSISTENT_PER_CORE_WORKER_FRESH_GAME_STATE'))
            study.atomic_json(path, receipt)
            return 1  # No next cell in a failed or uncertain worker.
        study.atomic_json(path, receipt)
        completed += 1


class Pool:
    """Lazy per-pair owned processes; no model/game in the controller."""
    def __init__(self, plan_path, plan, output):
        self.plan_path, self.plan, self.output = Path(plan_path), plan, Path(output)
        self.processes = {}
        self.busy = set()
        self.closed = False

    async def invoke(self, index, pair):
        key = tuple(pair)
        study.require(not self.closed and key in map(tuple, self.plan['cpu_pairs'])
            and key not in self.busy, 'exclusive declared persistent worker slot required')
        self.busy.add(key)
        try:
            process = self.processes.get(key)
            if process is None:
                process = await asyncio.create_subprocess_exec(sys.executable, '-m',
                    'pokezero.owning_encoder_worker', str(self.plan_path),
                    str(self.output), str(pair[0]), str(pair[1]),
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
                self.processes[key] = process
            study.require(process.returncode is None, 'terminated worker cannot be retried')
            path = self.output/f'cell-{index}.json'
            study.require(not path.exists() and not path.with_suffix('.attempted').exists(),
                'attempted persistent cell cannot be resent')
            process.stdin.write(study.canonical(dict(index=index))+b'\n')
            await process.stdin.drain()
            while not path.exists():
                study.require(process.returncode is None, 'persistent worker failed without receipt; no retry')
                await asyncio.sleep(.05)
            receipt = json.loads(path.read_bytes())
            if receipt.get('status') in ('FAILED', 'CAPPED'):
                raise continuation.WorkerFailure(receipt)
            study.require(process.returncode is None, 'worker exited after receipt; no retry')
            return receipt
        finally:
            self.busy.remove(key)

    async def close(self):
        if self.closed:return
        self.closed = True
        failed = []
        for process in self.processes.values():
            try:
                if process.returncode is None:
                    process.stdin.close()
                    try:await asyncio.wait_for(process.wait(), 5)
                    except asyncio.TimeoutError:pass
                await continuation._stop_owned_group(process)
            except BaseException as error:failed.append(error)
        if failed:raise RuntimeError('persistent worker group cleanup failed') from failed[0]


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'output', 'actor_cpu', 'fp_cpu'):parser.add_argument(name)
    args = parser.parse_args(argv)
    async def execute():
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(serve(json.loads(Path(args.plan).read_bytes()),
            Path(args.output), [int(args.actor_cpu), int(args.fp_cpu)]))
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        try:return await task
        finally:loop.remove_signal_handler(signal.SIGTERM)
    return asyncio.run(execute())


if __name__ == '__main__':raise SystemExit(main())
