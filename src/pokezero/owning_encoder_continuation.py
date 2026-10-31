"""Explicit fixed-order B3 continuation. Import/preparation never starts games.

An external, hash-bound closure receipt must establish that the prior producer
has stopped. Legacy ledgers did not journal starts: their first unrecorded
in-flight cell is conservatively unknown, never silently retried. New starts
are durable before child creation. Completed cell documents stay immutable.
"""
import asyncio
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from . import owning_encoder_study as study

SCHEMA = 'pokezero.B3-fixed-order-continuation.v1'
CLOSURE_SCHEMA = 'pokezero.B3-closed-attempt.v1'
FLOOR_SECONDS = 8 * 60 * 60
TOTAL_SECONDS = 30 * 24 * 60 * 60
IDENTITY = ('invocation', 'encoder_kind', 'seat', 'seed', 'foulplay_seed')
UNKNOWN = ('ATTEMPTED', 'UNKNOWN_ATTEMPTED')
require = study.require


class WorkerFailure(RuntimeError):
    def __init__(self, receipt):
        super().__init__('cell worker reported ' + receipt['failure']['error_type'])
        self.receipt = receipt


def _finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_ledger(spec, ledger):
    # Reuse all original roster, terminal-witness and score checks. Unknown
    # attempted cells are mapped only in this temporary reducer input.
    reduced = deepcopy(ledger)
    for cell in reduced['cells']:
        if cell['status'] in UNKNOWN:
            cell['status'] = 'FAILED'
    study.summarize(spec, reduced)
    return ledger


def _legacy_ambiguous(ledger):
    if ledger.get('continuation', {}).get('schema') == SCHEMA:
        return []  # This producer journals every attempt before dispatch.
    result = []
    for invocation in ledger['invocations']:
        if invocation['status'] == 'RUNNING':
            pending = [i for i, cell in enumerate(ledger['cells'])
                if cell['invocation'] == invocation['ordinal'] and cell['status'] == 'NOT_ATTEMPTED']
            if pending:
                result.append(pending[0])
    return result


def make_plan(*, prespec, declaration, ledger, closure, cpu_pairs, total_timeout_seconds=TOTAL_SECONDS):
    """Bind compact original artifacts and a separately captured closure receipt.

    Closure records source_run_closed=True, original prespec/declaration/ledger
    SHA256s, potentially_attempted_indices, and a hash-bound timing evidence
    reference plus a conservative completed-game duration upper bound. The
    entire previous run elapsed time is a safe bound if per-game time is absent.
    This function does not infer closure from a stale RUNNING ledger.
    """
    refs = {key: study._reference(value) for key, value in
        dict(prespec=prespec, declaration=declaration, ledger=ledger, closure=closure).items()}
    spec, original, closed = (study.read_bound(refs[key]) for key in ('prespec', 'ledger', 'closure'))
    declared = study.read_bound(refs['declaration'])
    study.validate_prespec(spec)
    require(refs['prespec']['sha256'] == study.prespec_file_sha256(spec), 'canonical original prespec bytes required')
    require(type(declared) is dict, 'original declaration document required')
    _validate_ledger(spec, original)
    require(closed.get('schema') == CLOSURE_SCHEMA and closed.get('source_run_closed') is True
        and all(closed.get(key + '_sha256') == refs[key]['sha256'] for key in ('prespec', 'declaration', 'ledger')),
        'closed source-run receipt and exact original artifact bindings required')
    possible = closed.get('potentially_attempted_indices')
    require(type(possible) is list and all(type(i) is int and 0 <= i < len(original['cells']) for i in possible)
        and possible == sorted(set(possible)) and set(_legacy_ambiguous(original)) <= set(possible)
        and all(original['cells'][i]['status'] == 'NOT_ATTEMPTED' for i in possible),
        'legacy in-flight cells must remain potentially attempted unknowns')
    timing = study._reference(closed['timing_evidence'])
    # Content need not be a particular telemetry schema, but the exact evidence
    # bytes must exist and agree with the closure receipt before planning.
    require(_file_sha(timing['path']) == timing['sha256'], 'original timing evidence changed')
    bound = closed.get('completed_game_duration_upper_bound_seconds')
    require(_finite(bound), 'conservative prior completion duration bound required')
    measured = [c['timing']['duration_seconds'] for c in original['cells']
        if c['status'] == 'COMPLETE' and 'timing' in c]
    require(all(_finite(x) for x in measured) and bound >= max(measured, default=0),
        'prior timing bound cannot weaken recorded game durations')
    require(type(cpu_pairs) in (list, tuple) and 1 <= len(cpu_pairs) <= 16
        and all(type(pair) in (list, tuple) and len(pair) == 2
            and all(type(cpu) is int and cpu >= 0 for cpu in pair) for pair in cpu_pairs),
        'one to sixteen explicit actor/FoulPlay CPU pairs required')
    flattened = [cpu for pair in cpu_pairs for cpu in pair]
    require(len(flattened) == len(set(flattened)), 'actor and FoulPlay CPUs must be globally disjoint')
    require(_finite(total_timeout_seconds, positive=True) and total_timeout_seconds >= FLOOR_SECONDS,
        'distant total-time backstop required')
    remaining = [i for i, cell in enumerate(original['cells'])
        if cell['status'] == 'NOT_ATTEMPTED' and i not in possible]
    return dict(schema=SCHEMA, references=refs, cpu_pairs=[list(pair) for pair in cpu_pairs],
        max_workers=len(cpu_pairs), ordered_indices=remaining, potentially_attempted_indices=possible,
        original_completed_sha256=study.digest([(i, cell) for i, cell in enumerate(original['cells'])
            if cell['status'] == 'COMPLETE']), prior_duration_upper_bound_seconds=bound,
        stall_floor_seconds=FLOOR_SECONDS, stall_longest_multiplier=4,
        total_timeout_seconds=total_timeout_seconds, model_factory_lifecycle='FRESH_PER_CELL_PROCESS',
        policy_rng='EXISTING_SEED_OWNER_DECISION_ROUND', retries=0, refill=False,
        original_prespec_unchanged=True, execution_authority='SCOTT_FIXED_ORDER_CONTINUATION_DIRECTIVE')


def validate_plan(plan):
    require(type(plan) is dict and plan.get('schema') == SCHEMA, 'explicit continuation plan required')
    expected = make_plan(**plan['references'], cpu_pairs=plan['cpu_pairs'],
        total_timeout_seconds=plan['total_timeout_seconds'])
    require(study.canonical(plan) == study.canonical(expected), 'continuation plan drift')


def single_config(spec, index, fp_cpu):
    ordinal, offset = divmod(index, study.COUNT)
    return replace(study._config(spec, ordinal), games=1,
        seed_start=spec['seed_start'] + offset, foulplay_random_seed=spec['foulplay_seed_start'] + offset,
        foulplay_cpu_affinity=fp_cpu)


def checked_result(document, spec, index):
    ordinal, offset = divmod(index, study.COUNT)
    require(document.get('encoder_kind') == study.ORDER[ordinal][0]
        and document.get('pokezero_player') == study.ORDER[ordinal][1]
        and document.get('checkpoint_sha256') == study.ActorContract.champion().checkpoint_sha256
        and type(document.get('seed_start')) is int and document['seed_start'] == spec['seed_start'] + offset
        and type(document.get('games')) is int and document['games'] == 1
        and type(document.get('completed_games')) is int and document['completed_games'] == 1
        and document.get('policy_mode') == 'raw' and document.get('format_id') == 'gen3randombattle'
        and document.get('foulplay_random_seed_schedule', {}).get('seeds') == [spec['foulplay_seed_start'] + offset]
        and all(type(document.get('execution_integrity', {}).get(key)) is int
            and document['execution_integrity'][key] == 0 for key in
            ('retries', 'errors', 'refusal_records', 'refusal_recorder_instrument_errors', 'refusal_records_unrowed')),
        'single original cell benchmark identity or execution integrity differs')
    require(type(document.get('game_results')) is list and len(document['game_results']) == 1,
        'exactly one original game result required')
    return study.checked_game(document['game_results'][0], spec, ordinal, offset)


def cpu_allocation(pairs, *, available=None, topology=None):
    """Check Linux topology so an actor cannot share its FP core's SMT sibling.

    This enforces placement within our workers. Exclusive reservation from
    unrelated host workloads remains the launcher's resource responsibility.
    """
    if available is None:
        require(hasattr(os, 'sched_getaffinity'), 'Linux CPU affinity required before workers')
        available = os.sched_getaffinity(0)
    cpus = {cpu for pair in pairs for cpu in pair}
    require(cpus <= set(available), 'assigned dedicated CPUs absent from process allocation')
    if topology is None:
        def topology(cpu):
            root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            return tuple(int((root/name).read_text()) for name in ('physical_package_id', 'core_id'))
    mapping = {cpu: tuple(topology(cpu)) for cpu in cpus}
    require(all(len(key) == 2 and all(type(x) is int and x >= 0 for x in key) for key in mapping.values()),
        'physical CPU package/core topology required')
    actors = {mapping[pair[0]] for pair in pairs}
    peers = [mapping[pair[1]] for pair in pairs]
    require(len(peers) == len(set(peers)) and not actors.intersection(peers),
        'FoulPlay requires distinct physical cores without actor SMT sharing')
    return {str(cpu): list(mapping[cpu]) for cpu in sorted(cpus)}


def claim_cell(plan, index, pair, receipt):
    """Exclusive, durable child claim forbids re-running the worker entry point."""
    receipt = Path(receipt)
    require(receipt.name == f'cell-{index}.json' and not receipt.exists(), 'new per-cell receipt path required')
    ledger = json.loads((receipt.parent/'ledger.json').read_bytes())
    require(ledger.get('continuation', {}).get('plan_sha256') == study.prespec_file_sha256(plan)
        and index in ledger['continuation']['dispatch_order']
        and ledger['cells'][index]['status'] == 'ATTEMPTED'
        and ledger['cells'][index]['attempt']['cpu_pair'] == list(pair),
        'durable parent attempt must precede a worker claim')
    with receipt.with_suffix('.attempted').open('xb') as stream:
        stream.write(study.canonical(dict(index=index, plan_sha256=study.prespec_file_sha256(plan)))+b'\n')
        stream.flush();os.fsync(stream.fileno())
    study._sync_directory(receipt.parent)


async def worker_game(plan, index, pair, receipt):
    """Actual child-only entry point; never called during source preparation."""
    validate_plan(plan)
    require(index in plan['ordered_indices'] and list(pair) in plan['cpu_pairs'], 'declared cell/worker required')
    claim_cell(plan, index, pair, receipt)
    require(hasattr(os, 'sched_setaffinity'), 'Linux CPU affinity is required before models or games')
    placement = cpu_allocation(plan['cpu_pairs'])
    os.sched_setaffinity(0, {pair[0]})
    require(os.sched_getaffinity(0) == {pair[0]}, 'actor CPU placement failed')
    spec = study.read_bound(plan['references']['prespec'])
    precision = study._configure_precision()
    from .live_protocol_runtime import champion_actor_factory
    from .foulplay_bridge import run_controlled_foulplay_benchmark
    config = single_config(spec, index, pair[1])
    started_epoch, started = time.time(), time.monotonic()
    result = await run_controlled_foulplay_benchmark(config, owning_actor_factory=champion_actor_factory(config))
    document = result.to_dict()
    checked_result(document, spec, index)
    return dict(index=index, document=document, precision=precision, cpu_pair=list(pair), cpu_topology=placement,
        timing=dict(started_epoch=started_epoch, completed_epoch=time.time(), duration_seconds=time.monotonic()-started))


async def _stop_owned_group(process):
    """Bound cleanup by the owned group, independently of the leader's exit.

    Fail closed if group disappearance cannot be confirmed: the scheduler must
    not reuse that slot/core while a descendant might still be running.
    """
    def exists():
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
    def send(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
    async def gone_within(seconds):
        deadline = time.monotonic() + seconds
        while exists():
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)
        return True
    send(signal.SIGTERM)
    if not await gone_within(5):
        send(signal.SIGKILL)
        require(await gone_within(5), 'owned worker group still exists after bounded SIGKILL cleanup')
    if process.returncode is None:
        await asyncio.wait_for(process.wait(), 5)


async def _child(plan_path, index, pair, receipt):
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'pokezero.owning_encoder_continuation',
        'worker', str(plan_path), str(index), str(pair[0]), str(pair[1]), str(receipt),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
    try:
        exit_code = await process.wait()
        if Path(receipt).exists():
            result = json.loads(Path(receipt).read_bytes())
            if result.get('status') in ('FAILED', 'CAPPED'):
                raise WorkerFailure(result)
            require(exit_code == 0, 'worker exited unsuccessfully after result publication; no retry')
            return result
        raise RuntimeError('cell worker failed without a terminal receipt; no retry')
    finally:
        await _stop_owned_group(process)


async def run_continuation(plan, output, *, worker=None, clock=time.monotonic, epoch=time.time):
    """Explicit execution. Test injection uses tiny fake workers only.

    Global dispatch order is fixed; finishes may arrive in any order. Every
    dispatched cell is journaled before invocation. First failure/stall stops
    new work and preserves other dispatched cells as unknown attempted.
    """
    validate_plan(plan)
    plan = json.loads(study.canonical(plan))
    spec = study.read_bound(plan['references']['prespec'])
    original = study.read_bound(plan['references']['ledger'])
    # Recheck the original independent B2 admission before any child creation.
    study.validate_admission(spec)
    if worker is None:
        cpu_allocation(plan['cpu_pairs'])
    output = Path(output)
    output.mkdir(exist_ok=False)
    study._sync_directory(output.parent)
    study.atomic_json(output/'plan.json', plan)
    ledger = deepcopy(original)
    for i in plan['potentially_attempted_indices']:
        ledger['cells'][i]['status'] = 'UNKNOWN_ATTEMPTED'
    ledger.update(status='RUNNING', actual_exit=None, failure=None, B3_qualified=False,
        continuation=dict(schema=SCHEMA, plan_sha256=study.prespec_file_sha256(plan),
            parent_ledger_sha256=plan['references']['ledger']['sha256'],
            dispatch_order=[], started_epoch=epoch(), completed_cell_bytes_preserved=True))
    ledger.pop('readout', None)
    def publish():
        for i, cell in enumerate(original['cells']):
            if cell['status'] != 'NOT_ATTEMPTED':
                require(study.canonical(cell) == study.canonical(ledger['cells'][i]), 'previous cell changed')
        ledger['recorded_games'] = sum(cell['status'] == 'COMPLETE' for cell in ledger['cells'])
        study.atomic_json(output/'ledger.json', ledger)
    publish()
    started = last_completion = clock()
    longest = plan['prior_duration_upper_bound_seconds']
    pending = iter(plan['ordered_indices'])
    free = list(range(plan['max_workers']))
    active = {}
    exhausted = False
    try:
        while active or not exhausted:
            while free and not exhausted:
                index = next(pending, None)
                if index is None:
                    exhausted = True
                    break
                slot = free.pop(0); pair = plan['cpu_pairs'][slot]
                require(ledger['cells'][index]['status'] == 'NOT_ATTEMPTED', 'attempted cells cannot restart')
                ledger['cells'][index].update(status='ATTEMPTED', attempt=dict(
                    index=index, slot=slot, cpu_pair=pair, started_epoch=epoch()))
                ledger['continuation']['dispatch_order'].append(index)
                publish()  # Durable start precedes process/task creation.
                call = worker(index, pair) if worker else _child(output/'plan.json', index, pair, output/f'cell-{index}.json')
                active[asyncio.create_task(call)] = (index, slot)
            if not active:
                break
            done, _ = await asyncio.wait(active, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(done, key=lambda t: active[t][0]):
                index, slot = active.pop(task)
                receipt = task.result()
                require(receipt.get('index') == index and receipt.get('cpu_pair') == plan['cpu_pairs'][slot],
                    'worker receipt identity or CPU allocation changed')
                checked = checked_result(receipt['document'], spec, index)
                timing = receipt['timing']
                require(set(timing) == {'started_epoch', 'completed_epoch', 'duration_seconds'}
                    and all(_finite(value) for value in timing.values())
                    and timing['completed_epoch'] >= timing['started_epoch'], 'actual per-game timing required')
                ledger['cells'][index].update(checked, timing=timing, worker_precision=receipt.get('precision'))
                longest = max(longest, timing['duration_seconds'])
                last_completion = clock()
                free.append(slot); free.sort()
                publish()
            now = clock()
            require(now - last_completion < max(FLOOR_SECONDS, 4 * longest), 'no completed game within stall window')
            require(now - started < plan['total_timeout_seconds'], 'distant total-time backstop reached')
        unresolved = any(c['status'] != 'COMPLETE' for c in ledger['cells'])
        ledger['status'], ledger['actual_exit'] = ('FAILED', 1) if unresolved else ('COMPLETE', 0)
        if unresolved:
            ledger['failure'] = dict(error_type='UnresolvedPriorCells', no_retry=True)
    except BaseException as error:
        ledger['status'], ledger['actual_exit'] = 'FAILED', 1
        ledger['failure'] = dict(error_type=type(error).__name__, **study._failure_diagnostics(error), no_retry=True)
        if isinstance(error, WorkerFailure):
            receipt = error.receipt
            index = receipt['index']
            require(index in ledger['continuation']['dispatch_order'], 'failed worker identity changed')
            ledger['cells'][index]['status'] = receipt['status']
            ledger['failure']['worker_receipt'] = receipt
    finally:
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        for index in ledger['continuation']['dispatch_order']:
            if ledger['cells'][index]['status'] == 'ATTEMPTED':
                ledger['cells'][index]['status'] = 'UNKNOWN_ATTEMPTED'
        for invocation in ledger['invocations']:
            rows = [c for c in ledger['cells'] if c['invocation'] == invocation['ordinal']]
            invocation['status'] = 'COMPLETE' if all(c['status'] == 'COMPLETE' for c in rows) else 'FAILED'
        ledger['continuation'].update(completed_epoch=epoch(), longest_completion_or_bound_seconds=longest,
            stall_window_seconds=max(FLOOR_SECONDS, 4 * longest))
        reduced = deepcopy(ledger)
        for cell in reduced['cells']:
            if cell['status'] in UNKNOWN:
                cell['status'] = 'FAILED'
        ledger['readout'] = study.summarize(spec, reduced)
        publish()
    return ledger


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prepare = sub.add_parser('prepare'); prepare.add_argument('input'); prepare.add_argument('out')
    run = sub.add_parser('run'); run.add_argument('plan'); run.add_argument('plan_sha256'); run.add_argument('out')
    child = sub.add_parser('worker')
    for name in ('plan', 'index', 'actor_cpu', 'fp_cpu', 'out'):child.add_argument(name)
    args = parser.parse_args(argv)
    if args.mode == 'prepare':
        document = make_plan(**json.loads(Path(args.input).read_bytes()))
        with Path(args.out).open('xb') as stream:stream.write(study.canonical(document)+b'\n')
        return 0
    if args.mode == 'worker':
        require(not Path(args.out).exists(), 'existing cell receipt cannot be replaced or retried')
        try:
            result = asyncio.run(worker_game(json.loads(Path(args.plan).read_bytes()), int(args.index),
                [int(args.actor_cpu), int(args.fp_cpu)], args.out))
            exit_code = 0
        except BaseException as error:
            capped = isinstance(error, study.CappedGame) or (isinstance(error, RuntimeError)
                and str(error) == 'owning game round cap without actual owner terminal')
            result = dict(index=int(args.index), status='CAPPED' if capped else 'FAILED',
                failure=dict(error_type=type(error).__name__, **study._failure_diagnostics(error)))
            if isinstance(error, PermissionError):
                result['failure']['missing_permission'] = dict(errno=error.errno, filename=error.filename)
            exit_code = 1
        study.atomic_json(args.out, result)
        return exit_code
    plan = study.read_bound(dict(path=args.plan, sha256=args.plan_sha256))
    async def execute():
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(run_continuation(plan, args.out))
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        try:
            return await task
        finally:
            loop.remove_signal_handler(signal.SIGTERM)
    return asyncio.run(execute())['actual_exit']


if __name__ == '__main__':
    raise SystemExit(main())
