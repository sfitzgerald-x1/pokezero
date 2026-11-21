"""Same-node V2 fused ranks, one visible CUDA device per child process.

Example: python -m gpu_engine.v2_distributed --ranks 4 --timeout-seconds 3600
followed by the original v2_fused declaration/runtime/reference/out arguments.
The parent has a finite deadline and terminates its remaining children if any
rank fails outside a coordinated checkpoint write failure. A completed round
commits one shared checkpoint/state manifest.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import queue
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from uuid import UUID


CHECKPOINT_WRITE_EXIT = 75
CHECKPOINT_WRITE_EXIT_GRACE = 10
IDLE_WINDOW_SECONDS = 900
GPU_SAMPLE_SECONDS = 15
GPU_PROBE_TIMEOUT = 8
PROGRESS_SAMPLE_SECONDS = 300
NEAR_ZERO_CPU = 1.0


class Distribution:
    def __init__(self, rank, world_size, process_group):
        self.rank = rank; self.world_size = world_size; self.process_group = process_group
        self.contract = None

    def gather(self, value):
        from pokezero.v2_distributed import gather
        return gather(value, self)

    def broadcast(self, value):
        if self.world_size == 1:
            return value
        import torch.distributed as dist
        values = [value]; dist.broadcast_object_list(values, src=0, group=self.process_group)
        return values[0]

    def barrier(self):
        if self.world_size > 1:
            import torch.distributed as dist
            dist.barrier(group=self.process_group)

    def bind(self, context, args):
        from pokezero.v2_distributed import shard_bounds
        plan = context['plan']
        if args.row_order not in (None, 'file') or args.order != 'ordered' or args.mode == 'reference':
            raise ValueError('distributed V2 requires file row order and static ordered learner')
        if plan.train.batch_size % self.world_size:
            raise ValueError('global batch must divide into equal per-rank batches')
        for batches in plan.iterations:
            shard_bounds(sum(len(b.games) for b in batches), self.rank, self.world_size)
        self.contract = dict(schema='v2-fused-distributed.v1', world_size=self.world_size,
            global_batch_size=plan.train.batch_size, local_batch_size=plan.train.batch_size // self.world_size,
            lanes_per_rank=args.lanes, declared=context['declared'],
            sharding='contiguous_game_indices_original_block_constructor',
            rows='canonical_block_frame_row_owner', loss='global_weighted_terms_v1',
            rng='independent_rank_dropout_streams_checkpointed', checkpoint='shared_round_boundary')
        self.games_by_round = [sum(len(b.games) for b in batches) for batches in plan.iterations]
        if any(c != self.contract for c in self.gather(self.contract)):
            raise ValueError('ranks disagree about the declared training contract')

    def check_resume(self, saved):
        if saved.get('distribution') != self.contract:
            raise ValueError('resume requires identical source/runtime/rank/game/batch contract')

    def shard(self, plan, iteration):
        from pokezero.v2_distributed import shard_bounds
        return shard_bounds(sum(len(b.games) for b in plan.iterations[iteration]), self.rank, self.world_size)

    def reduce_report(self, report):
        reports = self.gather(report)
        if any(r['snapshot'] != report['snapshot'] or r['training']['steps'] != report['training']['steps']
                or r['training']['rows'] != report['training']['rows']
                or r['training'].get('lr_start') != report['training'].get('lr_start')
                or r['training'].get('campaign_schedule') != report['training'].get('campaign_schedule') for r in reports):
            raise ValueError('rank snapshot/update accounting diverged')
        combined = dict(reports[0], ranks=reports, world_size=self.world_size)
        for key in ('games', 'rows'):
            combined[key] = sum(r[key] for r in reports)
        if combined['games'] != self.games_by_round[report['iteration']]:
            raise ValueError('rank game totals differ from the declared global round')
        for key in ('dispositions', 'opponents'):
            combined[key] = {name: sum(r[key].get(name, 0) for r in reports)
                for name in set().union(*(r[key] for r in reports))}
        for key in ('iteration_seconds', 'steady_seconds'):
            combined[key] = max(r[key] for r in reports)
        combined['phase_seconds'] = {name: max(r['phase_seconds'][name] for r in reports)
            for name in report['phase_seconds']}
        combined['games_per_second'] = combined['games'] / combined['iteration_seconds']
        combined['games_per_second_steady'] = combined['games'] / combined['steady_seconds']
        combined['audit'] = [a for r in reports for a in r['audit']]
        if report['games_matching_reference'] is not None:
            combined['games_matching_reference'] = {k: sum(r['games_matching_reference'][k] for r in reports)
                for k in report['games_matching_reference']}
        return combined


def _worker(args, fused_args):
    import torch
    import torch.distributed as dist
    if torch.cuda.device_count() != 1:
        raise ValueError('each rank must see exactly its one assigned CUDA device')
    torch.cuda.set_device(0)
    group = None
    if args.ranks > 1:
        dist.init_process_group('nccl', init_method='file://' + str(Path(args.rendezvous).resolve()),
            rank=args.rank, world_size=args.ranks, timeout=datetime.timedelta(seconds=args.collective_timeout))
        group = dist.group.WORLD
    try:
        from gpu_engine import v2_fused
        from pokezero.v2_trainer import CheckpointWriteError
        try:
            return v2_fused.main(fused_args, distribution=Distribution(args.rank, args.ranks, group))
        except CheckpointWriteError as error:
            print(str(error), file=sys.stderr, flush=True)
            return CHECKPOINT_WRITE_EXIT
    finally:
        if group is not None:
            dist.destroy_process_group()


def _stop(processes):
    """Terminate only these owned worker process groups, with a finite grace."""
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


class _IdleEvidence:
    """Track overlapping idle intervals using complete, consecutive samples."""
    def __init__(self, ticks_per_second):
        self.ticks_per_second = ticks_per_second
        self.gpu_time = self.gpu_since = None
        self.progress = self.quiet_since = None

    def observe_gpu(self, now, idle):
        consecutive = self.gpu_time is not None and 0 < now - self.gpu_time <= GPU_SAMPLE_SECONDS + GPU_PROBE_TIMEOUT
        if not idle or not consecutive or self.gpu_since is None:
            self.gpu_since = now if idle else None
        self.gpu_time = now

    def observe_progress(self, now, cache, ranks):
        previous = self.progress
        self.progress = None if cache is None else (now, cache, ranks)
        if previous is not None and self.progress is not None:
            last_time, last_cache, last_ranks = previous
            elapsed = now - last_time
            if (0 < elapsed <= PROGRESS_SAMPLE_SECONDS + GPU_SAMPLE_SECONDS + GPU_PROBE_TIMEOUT
                    and cache == last_cache and len(last_ranks) == len(ranks)
                    and all(
                        start == old_start and log == old_log
                        and 0 <= 100 * (ticks - old_ticks) < NEAR_ZERO_CPU * elapsed * self.ticks_per_second
                        for (start, ticks, log), (old_start, old_ticks, old_log) in zip(ranks, last_ranks)
                    )):
                if self.quiet_since is None:
                    self.quiet_since = last_time
                return
        self.quiet_since = None

    def stalled(self, now, started):
        return (self.gpu_since is not None and self.quiet_since is not None
                and 0 <= now - self.gpu_time <= GPU_SAMPLE_SECONDS + GPU_PROBE_TIMEOUT
                and now - started >= IDLE_WINDOW_SECONDS
                and now - max(self.gpu_since, self.quiet_since) > IDLE_WINDOW_SECONDS)


def _gpu_idle(identifiers):
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,utilization.gpu', '--format=csv,noheader,nounits',
    ], timeout=GPU_PROBE_TIMEOUT, text=True)
    expected = {identifier.lower() for identifier in identifiers}
    observed = {}
    for line in output.splitlines():
        uuid, utilization = (field.strip() for field in line.split(','))
        uuid = uuid.lower()
        if uuid not in expected:
            continue
        value = int(utilization)
        if uuid in observed or not 0 <= value <= 100:
            raise ValueError('invalid utilization for an allocated GPU')
        observed[uuid] = value
    if observed.keys() != expected:
        raise ValueError('incomplete allocated GPU utilization sample')
    return not any(observed.values())


def _rank_progress(process, log):
    fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
    if fields[0] == 'Z':
        raise ProcessLookupError(process.pid)
    stat = os.fstat(log.fileno())
    return int(fields[19]), int(fields[11]) + int(fields[12]), (stat.st_size, stat.st_mtime_ns)


def _cache_progress(cache):
    root = Path(cache).resolve(strict=True)
    pending = [root]
    inventory = {}

    def signature(info):
        return info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    while pending:
        directory = pending.pop()
        before = directory.stat(follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            raise OSError('cache directory changed during observation')
        directory_key = str(directory.relative_to(root))
        before_key = signature(before)
        if directory_key in inventory and inventory[directory_key] != before_key:
            raise OSError('cache directory changed during observation')
        inventory[directory_key] = before_key
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise OSError('cache sample contains a symbolic link')
                inventory[str(path.relative_to(root))] = signature(info)
                # JAX stores its kernel cache file at the root and autotune entries here.
                if stat.S_ISDIR(info.st_mode) and (
                    directory != root or entry.name == 'xla_gpu_per_fusion_autotune_cache_dir'
                ):
                    pending.append(path)
        if signature(directory.stat(follow_symlinks=False)) != before_key:
            raise OSError('cache directory changed during observation')
    return inventory


def _observe_idle(completed, processes, logs, identifiers, cache, started, retire):
    if not cache:
        print('S0 idle guard needs JAX_COMPILATION_CACHE_DIR', file=sys.stderr, flush=True)
        return
    evidence = _IdleEvidence(os.sysconf('SC_CLK_TCK'))
    next_progress = started
    while not retire.is_set() and all(process.poll() is None for process in processes):
        try:
            gpu_idle = _gpu_idle(identifiers)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(f'S0 idle guard GPU sample unavailable: {error}', file=sys.stderr, flush=True)
            gpu_idle = False
        now = time.monotonic()
        evidence.observe_gpu(now, gpu_idle)
        if now >= next_progress:
            try:
                cache_progress = _cache_progress(cache)
                ranks = [_rank_progress(process, log) for process, log in zip(processes, logs)]
            except (OSError, ValueError, IndexError) as error:
                print(f'S0 idle guard progress sample unavailable: {error}', file=sys.stderr, flush=True)
                cache_progress, ranks = None, ()
            now = time.monotonic()
            evidence.observe_progress(now, cache_progress, ranks)
            next_progress = now + PROGRESS_SAMPLE_SECONDS
            if evidence.stalled(now, started) and not retire.is_set():
                completed.put(('idle', now))
        retire.wait(GPU_SAMPLE_SECONDS)


def _wait_workers(completed, ranks, deadline, *, processes=(), retire=None):
    codes = {}
    checkpoint_deadline = None
    while len(codes) < ranks:
        end = deadline if checkpoint_deadline is None else checkpoint_deadline
        try:
            rank, code = completed.get(timeout=max(.001, end - time.monotonic()))
        except queue.Empty:
            codes['deadline'] = 124
            break
        if rank == 'idle':
            now = time.monotonic()
            if now >= end:
                codes['deadline'] = 124
                break
            if codes or not processes or not 0 <= now - code <= GPU_SAMPLE_SECONDS:
                continue
            try:
                active = all(process.poll() is None for process in processes)
            except OSError as error:
                print(f'S0 idle guard could not confirm worker liveness: {error}', file=sys.stderr, flush=True)
                continue
            if active:
                codes['idle'] = 124
                break
            continue
        if retire is not None:
            retire.set()
        codes[rank] = code
        if code == CHECKPOINT_WRITE_EXIT:
            if checkpoint_deadline is None:
                checkpoint_deadline = min(deadline, time.monotonic() + CHECKPOINT_WRITE_EXIT_GRACE)
        elif code:
            break
    return codes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', required=True, type=int, choices=(1, 2, 4))
    parser.add_argument('--timeout-seconds', required=True, type=int)
    parser.add_argument('--collective-timeout', type=int, default=1800)
    parser.add_argument('--s0-idle-guard', action='store_true',
                        help='add a timeout for verified GPU, rank, cache and log inactivity')
    parser.add_argument('--rank', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--rendezvous', default=None, help=argparse.SUPPRESS)
    args, fused_args = parser.parse_known_args(argv)
    if args.timeout_seconds <= 0 or args.collective_timeout <= 0:
        raise ValueError('positive finite worker/collective deadlines required')
    if args.rank is not None:
        if not 0 <= args.rank < args.ranks or args.ranks > 1 and not args.rendezvous:
            raise ValueError('valid rank and rendezvous required')
        return _worker(args, fused_args)
    from gpu_engine.v2_fused import parse_args as parse_fused_args
    from pokezero.v2_fused_state import validate_output

    fused = parse_fused_args(fused_args)
    out = Path(fused.out)
    validate_output(out, Path(fused.cooldown_child) if fused.cooldown_child else None)
    if fused.require_resume and not (out / 'state.json').is_file():
        raise ValueError('--require-resume needs a committed state.json in --out')
    out.mkdir(parents=True, exist_ok=True)
    # Respect the container's allocated GPU visibility. A parent allocation
    # larger than the declared rank count must not silently leave GPUs idle.
    import torch
    count = torch.cuda.device_count()
    if count != args.ranks:
        raise ValueError(f'exactly {args.ranks} allocated CUDA devices required; visible={count}')
    # Torch can return a bare UUID. CUDA_VISIBLE_DEVICES expects the GPU- form.
    identifiers = [f"GPU-{UUID(str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-'))}"
                   for i in range(count)]
    if len(set(identifiers)) != count:
        raise ValueError('distinct allocated GPU UUIDs required')
    with tempfile.TemporaryDirectory(prefix='v2-rendezvous-') as temporary:
        rendezvous = str(Path(temporary) / 'store')
        processes = []; logs = []; completed = queue.Queue(); codes = {}
        started = time.monotonic(); deadline = started + args.timeout_seconds
        retire = threading.Event() if args.s0_idle_guard else None
        try:
            for rank, identifier in enumerate(identifiers):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=identifier, JAX_PLATFORMS='cuda',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                command = [sys.executable, '-m', __name__ if __name__ != '__main__' else 'gpu_engine.v2_distributed',
                    '--ranks', str(args.ranks), '--rank', str(rank), '--rendezvous', rendezvous,
                    '--timeout-seconds', str(args.timeout_seconds), '--collective-timeout', str(args.collective_timeout), *fused_args]
                log = (out / f'rank-{rank}.log').open('ab'); logs.append(log)
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                threading.Thread(target=lambda r=rank, p=process: completed.put((r, p.wait())), daemon=True).start()
            if retire is not None:
                try:
                    threading.Thread(target=_observe_idle, args=(
                        completed, processes, logs, identifiers,
                        os.environ.get('JAX_COMPILATION_CACHE_DIR'), started, retire,
                    ), daemon=True).start()
                except RuntimeError as error:
                    print(f'S0 idle guard observer unavailable: {error}', file=sys.stderr, flush=True)
            codes = _wait_workers(completed, args.ranks, deadline, processes=processes, retire=retire)
        finally:
            if retire is not None:
                retire.set()
            _stop(processes)
            for log in logs:
                log.close()
            for rank, process in enumerate(processes):
                codes.setdefault(rank, process.returncode)
            (out / 'rank-exit.json').write_text(json.dumps(dict(codes=codes,
                wall_seconds=time.monotonic() - started, world_size=args.ranks), indent=2) + '\n')
    return 0 if len(codes) == args.ranks and all(code == 0 for code in codes.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
