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
import subprocess
import sys
import tempfile
import threading
import time
from uuid import UUID


CHECKPOINT_WRITE_EXIT = 75
CHECKPOINT_WRITE_EXIT_GRACE = 10


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


def _wait_workers(completed, ranks, deadline):
    codes = {}
    checkpoint_deadline = None
    while len(codes) < ranks:
        end = deadline if checkpoint_deadline is None else checkpoint_deadline
        try:
            rank, code = completed.get(timeout=max(.001, end - time.monotonic()))
        except queue.Empty:
            codes['deadline'] = 124
            break
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
    parser.add_argument('--rank', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--rendezvous', default=None, help=argparse.SUPPRESS)
    args, fused_args = parser.parse_known_args(argv)
    if args.timeout_seconds <= 0 or args.collective_timeout <= 0:
        raise ValueError('positive finite worker/collective deadlines required')
    if args.rank is not None:
        if not 0 <= args.rank < args.ranks or args.ranks > 1 and not args.rendezvous:
            raise ValueError('valid rank and rendezvous required')
        return _worker(args, fused_args)
    if '--out' not in fused_args:
        raise ValueError('explicit shared output directory required')
    out = Path(fused_args[fused_args.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)
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
            codes = _wait_workers(completed, args.ranks, deadline)
        finally:
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
