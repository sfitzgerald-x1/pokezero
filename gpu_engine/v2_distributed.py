"""V2 fused ranks over one or more nodes, one visible CUDA device per child process.

Example (one node): python -m gpu_engine.v2_distributed --ranks 4 --timeout-seconds 3600
followed by the original v2_fused declaration/runtime/reference/out arguments.
Multi-node: every node runs the same command with --nodes N --node-index I and a
shared --master HOST:PORT (the node-0 rendezvous address); global rank is
node_index * ranks + local rank and the world size is nodes * ranks. All ranks
must join within --rendezvous-timeout seconds, otherwise every node exits
nonzero (all nodes or nothing). A node whose partner disappears later fails
its next collective after --collective-timeout seconds and exits nonzero too.
The parent has a finite deadline and terminates its remaining children if any
rank fails outside a coordinated checkpoint write failure. A completed round
commits one shared checkpoint/state manifest on the shared output directory.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from uuid import UUID


CHECKPOINT_WRITE_EXIT = 75
CHECKPOINT_WRITE_EXIT_GRACE = 10
RANKS_PER_NODE = (1, 2, 4)


TOPOLOGIES = ('ddp', 'learner-node')


class Distribution:
    """Rank placement. ``ddp``: every rank collects and learns (data parallel over the whole world).
    ``learner-node``: every rank collects its global-index shard, rows are routed to the first node's
    ranks (the learner world, ``ranks_per_node`` wide), which train exactly as a one-node run; the new
    actor reaches every node through the shared output directory. Only rows cross nodes."""

    def __init__(self, rank, world_size, process_group, *, nodes=1, ranks_per_node=None, store=None,
                 topology='ddp', learner_group=None):
        if topology not in TOPOLOGIES:
            raise ValueError(f'topology must be one of {TOPOLOGIES}')
        self.rank = rank; self.world_size = world_size; self.process_group = process_group
        self.nodes = nodes; self.ranks_per_node = world_size // nodes if ranks_per_node is None else ranks_per_node
        self.node_index = rank // self.ranks_per_node; self.local_rank = rank % self.ranks_per_node
        self.store = store   # rendezvous key-value store shared by every rank (for cross-rank coordination)
        self.topology = topology
        self.learner_world = self.ranks_per_node if topology == 'learner-node' else world_size
        self.learner_group = process_group if self.learner_world == world_size else learner_group
        if self.learner_world < world_size and learner_group is None and world_size > 1:
            raise ValueError('learner-node topology requires the learner process group')
        self.is_learner = rank < self.learner_world
        self.contract = None

    def exchange(self):
        from pokezero.v2_distributed import Exchange
        return Exchange(self.rank, self.world_size, self.process_group, self.learner_world)

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

    def placement(self):
        return dict(world_size=self.world_size, nodes=self.nodes, ranks_per_node=self.ranks_per_node,
                    topology=self.topology, learner_world=self.learner_world)

    def bind(self, context, args):
        from pokezero.v2_distributed import shard_bounds
        plan = context['plan']
        if args.row_order not in (None, 'file') or args.order != 'ordered' or args.mode == 'reference':
            raise ValueError('distributed V2 requires file row order and static ordered learner')
        if plan.train.batch_size % self.learner_world:
            raise ValueError('global batch must divide into equal per-learner-rank batches')
        for batches in plan.iterations:
            shard_bounds(sum(len(b.games) for b in batches), self.rank, self.world_size)
        # The contract is placement-free on purpose: the same world size on a
        # different node split (for example 1x8 against 2x4) trains identically,
        # so only the world size decides a resume. Node topology is reported.
        # The ddp contract is unchanged from single-node runs so they still resume.
        self.contract = dict(schema='v2-fused-distributed.v1', world_size=self.world_size,
            global_batch_size=plan.train.batch_size, local_batch_size=plan.train.batch_size // self.learner_world,
            lanes_per_rank=args.lanes, declared=context['declared'],
            sharding='contiguous_game_indices_original_block_constructor',
            rows='canonical_block_frame_row_owner', loss='global_weighted_terms_v1',
            rng='independent_rank_dropout_streams_checkpointed', checkpoint='shared_round_boundary',
            **({'topology': self.topology, 'learner_world': self.learner_world} if self.topology != 'ddp' else {}))
        self.games_by_round = [sum(len(b.games) for b in batches) for batches in plan.iterations]
        if any(c != self.contract for c in self.gather(self.contract)):
            raise ValueError('ranks disagree about the declared training contract')

    def check_resume(self, saved):
        recorded = saved.get('distribution')
        if type(recorded) is dict and (recorded.get('world_size'), recorded.get('topology', 'ddp')) != (self.world_size, self.topology):
            raise ValueError(f'resume topology mismatch: the run was recorded with world size {recorded.get("world_size")} '
                             f'({recorded.get("topology", "ddp")}) and this attempt has world size {self.world_size} '
                             f'({self.nodes} node(s) x {self.ranks_per_node}, {self.topology}); restart with the original placement')
        if recorded != self.contract:
            raise ValueError('resume requires identical source/runtime/rank/game/batch contract')

    def shard(self, plan, iteration):
        from pokezero.v2_distributed import shard_bounds
        return shard_bounds(sum(len(b.games) for b in plan.iterations[iteration]), self.rank, self.world_size)

    def reduce_report(self, report):
        reports = self.gather(report)
        if any(r['snapshot'] != report['snapshot'] or r['training']['steps'] != report['training']['steps']
                or r['training']['rows'] != report['training']['rows'] for r in reports):
            raise ValueError('rank snapshot/update accounting diverged')
        combined = dict(reports[0], ranks=reports, **self.placement())
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


def parse_master(value):
    host, _, port = str(value).rpartition(':')
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError('--master requires HOST:PORT')
    return host, int(port)


def rendezvous(args, rank, world_size):
    """One TCP store hosted by global rank 0; every rank joins within the rendezvous deadline."""
    import torch.distributed as dist
    host, port = parse_master(args.master)
    deadline = datetime.timedelta(seconds=args.rendezvous_timeout)
    try:
        store = dist.TCPStore(host, port, world_size, is_master=rank == 0, timeout=deadline,
                              wait_for_workers=True)
    except Exception as error:   # DistStoreError/DistNetworkError/TimeoutError depend on the torch version
        raise RuntimeError(f'rank {rank} could not rendezvous with all {world_size} ranks at {host}:{port} '
                           f'within {args.rendezvous_timeout} s: {type(error).__name__}: {error}') from error
    dist.init_process_group(args.backend, store=store, rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=args.collective_timeout))
    return store


def _bench(distribution, device, *, floats):
    """Pre-flight fabric numbers: node-0 <-> node-1 ping-pong latency/bandwidth, a learner-sized all-reduce, an all-to-all."""
    import torch
    import torch.distributed as dist
    group = distribution.process_group; rank = distribution.rank; sync = torch.cuda.synchronize if device == 'cuda' else (lambda: None)

    def timed(action, reps):
        sync(); start = time.perf_counter()
        for _ in range(reps):
            action()
        sync(); return (time.perf_counter() - start) / reps

    results = dict(device=device, floats=floats)
    if distribution.nodes > 1:
        peer = distribution.ranks_per_node                      # first rank of node 1
        if rank in (0, peer):
            other = peer if rank == 0 else 0
            small, big = torch.zeros(1, device=device), torch.zeros(floats, device=device)
            def pingpong(tensor):
                if rank == 0:
                    dist.send(tensor, other, group=group); dist.recv(tensor, other, group=group)
                else:
                    dist.recv(tensor, other, group=group); dist.send(tensor, other, group=group)
            timed(lambda: pingpong(small), 5); timed(lambda: pingpong(big), 2)
            results['cross_node_latency_us'] = timed(lambda: pingpong(small), 50) / 2 * 1e6
            results['cross_node_bandwidth_gbps'] = 2 * big.numel() * 4 * 8 / timed(lambda: pingpong(big), 5) / 1e9
            del big
    distribution.barrier()
    gradient = torch.ones(floats, device=device)                 # the learner gradient is ~10M fp32
    timed(lambda: dist.all_reduce(gradient, group=group), 3)
    results['all_reduce_ms'] = timed(lambda: dist.all_reduce(gradient, group=group), 20) * 1e3
    world = distribution.world_size
    chunk = floats // world * world
    sent, received = torch.ones(chunk, device=device), torch.empty(chunk, device=device)
    timed(lambda: dist.all_to_all_single(received, sent, group=group), 2)
    seconds = timed(lambda: dist.all_to_all_single(received, sent, group=group), 5)
    results['all_to_all_ms'] = seconds * 1e3
    results['all_to_all_gbps_per_rank'] = chunk * 4 * 8 * (world - 1) / world / seconds / 1e9
    return results


def _probe(args, fused_args, rank, world_size, distribution):
    """Collective smoke used by the CPU/gloo launch tests and the GPU pre-flight: every rank sees every other rank."""
    import torch
    import torch.distributed as dist
    device = 'cuda' if args.backend == 'nccl' else 'cpu'
    ranks = distribution.gather(dict(rank=rank, node=distribution.node_index, local=distribution.local_rank,
                                     host=socket.gethostname()))
    total = torch.tensor([float(rank)], device=device); dist.all_reduce(total, group=distribution.process_group)
    if [r['rank'] for r in ranks] != list(range(world_size)) or int(total.item()) != world_size * (world_size - 1) // 2:
        raise ValueError('probe collectives disagree with the declared world')
    bench = _bench(distribution, device, floats=10_000_000 if device == 'cuda' else 100_000) if world_size > 1 else None
    distribution.barrier()
    out = Path(fused_args[fused_args.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)
    (out / f'probe-rank-{rank}.json').write_text(json.dumps(dict(ranks=ranks, bench=bench, **distribution.placement())) + '\n')
    return 0


def _worker(args, fused_args):
    import torch
    import torch.distributed as dist
    world_size = args.nodes * args.ranks
    rank = args.node_index * args.ranks + args.rank
    if args.backend == 'nccl':
        if torch.cuda.device_count() != 1:
            raise ValueError('each rank must see exactly its one assigned CUDA device')
        torch.cuda.set_device(0)
    group = store = learners = None
    if world_size > 1:
        store = rendezvous(args, rank, world_size)
        group = dist.group.WORLD
        if args.topology == 'learner-node' and args.ranks < world_size:
            learners = dist.new_group(ranks=list(range(args.ranks)), timeout=datetime.timedelta(seconds=args.collective_timeout))
    distribution = Distribution(rank, world_size, group, nodes=args.nodes, ranks_per_node=args.ranks, store=store,
                                topology=args.topology, learner_group=learners)
    try:
        if args.probe:
            return _probe(args, fused_args, rank, world_size, distribution)
        from gpu_engine import v2_fused
        from pokezero.v2_trainer import CheckpointWriteError
        try:
            return v2_fused.main(fused_args, distribution=distribution)
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


def _free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _device_selectors(args):
    """CUDA_VISIBLE_DEVICES per local rank from the container's allocated devices (none under gloo)."""
    if args.backend != 'nccl':
        return [''] * args.ranks
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
    return identifiers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', required=True, type=int, choices=RANKS_PER_NODE, help='ranks (GPUs) on this node')
    parser.add_argument('--nodes', type=int, default=1, help='nodes in the run; every node runs --ranks ranks')
    parser.add_argument('--node-index', type=int, default=0, help='this node, 0 <= index < --nodes')
    parser.add_argument('--master', default=None, help='HOST:PORT of the global rank-0 rendezvous (node 0); '
                        'required when --nodes > 1, defaults to a loopback port on one node')
    parser.add_argument('--timeout-seconds', required=True, type=int)
    parser.add_argument('--collective-timeout', type=int, default=1800)
    parser.add_argument('--rendezvous-timeout', type=int, default=1200,
                        help='seconds for every rank of every node to join; a late or missing node fails all nodes')
    parser.add_argument('--topology', choices=TOPOLOGIES, default='ddp',
                        help='ddp: every rank learns (gradients cross nodes every step); learner-node: every rank collects, '
                             'only node 0 learns (only rows cross nodes, once per round)')
    parser.add_argument('--backend', choices=('nccl', 'gloo'), default='nccl', help=argparse.SUPPRESS)
    parser.add_argument('--probe', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--rank', type=int, default=None, help=argparse.SUPPRESS)
    args, fused_args = parser.parse_known_args(argv)
    if args.timeout_seconds <= 0 or args.collective_timeout <= 0 or args.rendezvous_timeout <= 0:
        raise ValueError('positive finite worker/collective/rendezvous deadlines required')
    if args.nodes <= 0 or not 0 <= args.node_index < args.nodes:
        raise ValueError('valid --nodes and --node-index required')
    if args.nodes > 1 and not args.master:
        raise ValueError('--master HOST:PORT required for a multi-node run')
    if args.master:
        parse_master(args.master)
    world_size = args.nodes * args.ranks
    if args.rank is not None:
        if not 0 <= args.rank < args.ranks or world_size > 1 and not args.master:
            raise ValueError('valid rank and rendezvous required')
        return _worker(args, fused_args)
    if '--out' not in fused_args:
        raise ValueError('explicit shared output directory required')
    out = Path(fused_args[fused_args.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)
    identifiers = _device_selectors(args)
    master = args.master or f'127.0.0.1:{_free_port()}'
    processes = []; logs = []; completed = queue.Queue(); codes = {}
    started = time.monotonic(); deadline = started + args.timeout_seconds
    try:
        for local, identifier in enumerate(identifiers):
            rank = args.node_index * args.ranks + local
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=identifier, JAX_PLATFORMS='cuda' if args.backend == 'nccl' else 'cpu',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            command = [sys.executable, '-m', __name__ if __name__ != '__main__' else 'gpu_engine.v2_distributed',
                '--ranks', str(args.ranks), '--nodes', str(args.nodes), '--node-index', str(args.node_index),
                '--rank', str(local), '--master', master, '--backend', args.backend, '--topology', args.topology,
                *(['--probe'] if args.probe else []),
                '--timeout-seconds', str(args.timeout_seconds), '--collective-timeout', str(args.collective_timeout),
                '--rendezvous-timeout', str(args.rendezvous_timeout), *fused_args]
            log = (out / f'rank-{rank}.log').open('ab'); logs.append(log)
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            threading.Thread(target=lambda r=rank, p=process: completed.put((r, p.wait())), daemon=True).start()
        codes = _wait_workers(completed, args.ranks, deadline)
    finally:
        _stop(processes)
        for log in logs:
            log.close()
        for local, process in enumerate(processes):
            codes.setdefault(args.node_index * args.ranks + local, process.returncode)
        if any(code for code in codes.values()):
            print(f'node {args.node_index}: rank exit codes {codes}; see rank-*.log in {out}', file=sys.stderr, flush=True)
        name = 'rank-exit.json' if args.nodes == 1 else f'rank-exit-node-{args.node_index}.json'
        (out / name).write_text(json.dumps(dict(codes=codes, wall_seconds=time.monotonic() - started,
            world_size=world_size, nodes=args.nodes, node_index=args.node_index, ranks_per_node=args.ranks,
            topology=args.topology, master=master), indent=2) + '\n')
    return 0 if len(codes) == args.ranks and all(code == 0 for code in codes.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
