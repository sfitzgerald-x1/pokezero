"""Run fused round coordination with real V2 training, checkpoints and CPU/Gloo ranks."""
from contextlib import ExitStack
from dataclasses import asdict
from datetime import timedelta
from itertools import count
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import v2_distributed as launcher, v2_fused
from pokezero import checkpoint_self_moves as encoding, v2_trainer as trainer, v5_run as run, v5_training
from test_v2_trainer import configurations, varied_batch

try:
    import torch
except ImportError:
    torch = None


def run_fixture(root, out, *, distribution=None, evaluation=None, limit=None, writer_failure=False,
                publication_failure=False, stop=False, invalid_resume_rank=None, fused_options=()):
    from gpu_engine import v2_bench_trainer, v2_collector, v5_device_backend, v5_model

    torch.set_num_threads(1)
    rank = 0 if distribution is None else distribution.rank
    model, loss, train = configurations(batch_size=8, epochs=2, dropout=.2, schedule='mit-thesis')
    batches = tuple((run.Batch(tuple(run.Game(i+1, f'game-{i}', 100+i, (2*i, 2*i+1))
                              for i in range(8*j, 8*j+8)), 10+j, 20+j),) for j in range(5))
    declaration = dict(fixture='fused-boundary', evaluation=None if evaluation is None else evaluation.metadata())
    plan = SimpleNamespace(model=model, loss=loss, train=train, evaluation=evaluation, self_moves=False,
        iterations=batches, games=tuple(g for b, in batches for g in b.games), discount=1.,
        max_requests=64, historical_fraction=.5, historical_window=2,
        manifest=lambda: declaration, evaluation_milestones=lambda: {1: 16, 3: 32})
    context = dict(plan=plan, declared={}, preparation_seconds=1, gender=None, pool=None,
        prepared=SimpleNamespace(engine_tables=None, observation_tables=None))
    snapshot = run.Snapshot(-1, 'initial.pt', 'a'*64, 'initial.npz', 'b'*64, run._sha(model.metadata()))
    events = [{'kind': 'SNAPSHOT', 'payload': asdict(snapshot)}]
    initial = trainer.create(model, loss, train, enable_v5=True, device='cpu')
    argv = ['--config', 'unused.json', '--reference-run', str(root), '--out', str(out),
            '--torch-memory-fraction', '0', '--mode', 'eager', '--lanes', '8', *fused_options]
    if limit is not None:
        argv += ['--max-iterations', str(limit)]
    phases = dict(saves=[], exports=[], state=[], collect=[], issues=[])

    class Rows:
        def __init__(self, iteration, begin, end):
            self.n = end - begin
            self.stats = {'dispositions': {'COMPLETE': self.n}}
            self.arrays = {}
            self.addresses = np.column_stack((np.arange(8*iteration+begin+1, 8*iteration+end+1),
                np.zeros(self.n, dtype=int), np.zeros(self.n, dtype=int)))
            all_tensors = v5_training._training_tensors(varied_batch(rows=8, seed=71+iteration), model, 'cpu')
            self.tensors = {name: value[begin:end] if isinstance(value, (torch.Tensor, tuple)) else value
                            for name, value in all_tensors.items()}

        def keys(self):
            return self.addresses

        def to_torch(self, **kwargs):
            return self.tensors

    class Collector:
        def __init__(self, engine, observations, config, labels, roster, **kwargs):
            self.roster = roster

        def rebind(self, roster):
            self.roster = roster
            return self

        def compile(self, tree, **kwargs):
            return {'fixture_compile': 1}

        def run(self, tree):
            iteration, begin, end = self.roster.fixture
            phases['collect'].append(iteration)
            return SimpleNamespace(rows=Rows(iteration, begin, end), segments=[], stats={'occupancy': 1.})

    def construct(context, backend, iteration, current, past, ordinal, *, shard=None):
        begin, end = shard or (0, 8)
        roster = SimpleNamespace(groups=(None,), games=end-begin, actor_archives=(current.archive_sha256,),
                                 fixture=(iteration, begin, end))
        return roster, dict(SELFPLAY=end-begin, HISTORICAL=0), 0

    real_save, load, real_export = trainer.save_checkpoint, trainer.load_checkpoint, v5_model.export
    real_write, real_link = v2_fused.write_json, trainer.os.link
    def save(path, runtime):
        phases['saves'].append(path.name)
        if evaluation is not None and rank == 0:
            time.sleep(.08)
        return real_save(path, runtime)

    def export(checkpoint, archive, **options):
        phases['exports'].append(archive.name)
        return real_export(checkpoint, archive, **options)

    def write(path, payload, **options):
        if path.name == 'state.json':
            phases['state'].append(len(payload['reports']))
            assert options.get('durable', False) == (evaluation is not None)
        return real_write(path, payload, **options)

    def link(source, destination, **options):
        if writer_failure and Path(destination).suffix == '.pt':
            raise OSError('injected rank zero checkpoint write failure')
        return real_link(source, destination, **options)

    def actor(plan, snapshot, directory):
        if snapshot.iteration >= 0:
            assert v2_fused.sha(directory/snapshot.archive) == snapshot.archive_sha256
        return None

    def cpu(function):
        def placed(*args, **kwargs):
            if function is load and rank == invalid_resume_rank:
                kwargs['expected_checkpoint_sha256'] = '0'*64
            return function(*args, **dict(kwargs, device='cpu'))
        return placed

    def log(message, **values):
        if message in ('STOP', 'EVALUATION_REQUEST_ERROR'):
            phases['issues'].append([message, values])

    clock = count()
    with ExitStack() as patches:
        def replace(module, **options):
            for name, replacement in options.items():
                patches.enter_context(patch.object(module, name, replacement))

        replace(v2_fused, prepare=lambda args: context, construct=construct, load_actor=actor,
                audit=lambda *_: [], write_json=write, log=log,
                time=SimpleNamespace(perf_counter=lambda: float(next(clock))))
        replace(encoding, verified_run=lambda *_: (None, None, events), original_initializer=lambda *_: (None, None))
        replace(v2_bench_trainer, configurations=lambda *a, **k: (model, loss, train, initial.model.state_dict(), None))
        replace(v2_collector, RefillCollector=Collector, expected_counts=lambda *a: None,
                finalize=lambda result, **k: result.rows, game_records=lambda result: {})
        replace(v5_device_backend, RuntimeBackend=lambda *a, **k: SimpleNamespace(labels=None, check_plan=lambda _: None))
        replace(run, learning_rate_window=lambda _, i: v5_training.DeclaredGameWindow(8*i, 8*i+8, 40))
        replace(torch.cuda, synchronize=lambda: None)
        replace(trainer, create=cpu(trainer.create), load_checkpoint=cpu(trainer.load_checkpoint), save_checkpoint=save)
        replace(trainer.os, link=link)
        replace(v5_model, export=export)
        if evaluation is not None:
            from pokezero import v5_deferred_evaluation as deferred
            key = b'fixture-route-key'.ljust(32, b'.')
            patches.enter_context(patch.dict(trainer.os.environ, {deferred.STOP_KEY_ENV: key.hex()}))
            publish_dispatch = deferred.publish_dispatch

            def publish(request, *a, **k):
                committed = json.loads((out/'state.json').read_bytes())
                assert request.snapshot == run.Snapshot(**committed['snapshots'][request.snapshot.iteration+1])
                if publication_failure:
                    raise OSError('injected outbox failure')
                dispatch = publish_dispatch(request, *a, **k)
                if stop and request.declared_games == 32:
                    payload = dict(schema=deferred.STOP, verdict='FAIL', route_schema='s0.transfer.r2',
                        route_declaration_sha256=evaluation.route_declaration_sha256, declaration_sha256=request.declaration_sha256,
                        protocol_sha256=request.protocol_sha256, iteration=request.snapshot.iteration,
                        declared_games=request.declared_games, dispatch_sha256=dispatch.request_sha256,
                        independent_verdict_sha256='d'*64)
                    deferred._publish_equal(out/'evaluation-stop.json', dict(payload=payload,
                        signature=deferred._signature(payload, key)))
                return dispatch
            replace(deferred, publish_dispatch=publish)
        try:
            phases['exit'] = v2_fused.main(argv, distribution=distribution)
        except Exception as error:
            if writer_failure:
                assert isinstance(error, trainer.CheckpointWriteError), repr(error)
            elif invalid_resume_rank is not None:
                assert isinstance(error, ValueError), repr(error)
            else:
                raise
            phases['error'] = str(error)
    return phases


def run_rank(rank, directory, attempt, options):
    import torch.distributed as dist

    root = Path(directory)
    dist.init_process_group('gloo', init_method=(root / f'rendezvous-{attempt}').as_uri(), rank=rank,
                            world_size=4, timeout=timedelta(seconds=45))
    try:
        phases = run_fixture(root, root/'run', distribution=launcher.Distribution(rank, 4, dist.group.WORLD), **options)
    finally:
        dist.destroy_process_group()
    (root / f'{attempt}-{rank}.json').write_text(json.dumps(phases))


def run_four(root, attempt, **options):
    import torch.multiprocessing as mp

    # Child startup uses the real process group. Fail well before its collective
    # deadline if peers get stranded by an injected checkpoint write error.
    processes = mp.spawn(run_rank, args=(str(root), attempt, options), nprocs=4, join=False)
    seconds = 30 if options.get('writer_failure') else 85
    deadline = time.monotonic() + seconds
    while not processes.join(timeout=max(.01, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for child in processes.processes:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=2)
            raise AssertionError(f'four fixture ranks did not exit within {seconds}s')
    return [json.loads((root / f'{attempt}-{rank}.json').read_bytes()) for rank in range(4)]


@unittest.skipIf(torch is None, 'torch required')
class FusedDeferredTests(unittest.TestCase):
    def test_production_profile_freezes_after_first_commit_of_each_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'run'
            freezes = []
            collections = []

            def freeze():
                state = json.loads((out / 'state.json').read_bytes())
                freezes.append(state['reports'][-1]['iteration'])

            gc = SimpleNamespace(collect=lambda: collections.append(None) or 0,
                                 freeze=freeze, get_freeze_count=lambda: 1)
            profile = ('--profile', 'v2-production')
            with patch.object(v2_fused, 'gc', gc):
                initial = run_fixture(root, out, limit=2, fused_options=profile)
                resumed = run_fixture(root, out, limit=4, fused_options=profile)
                disabled = run_fixture(root, root / 'disabled', limit=2,
                                       fused_options=(*profile, '--no-freeze-gc-after-warmup'))
            self.assertEqual([initial['exit'], resumed['exit'], disabled['exit']], [0, 0, 0])
            self.assertEqual(freezes, [0, 2])
            self.assertEqual(len(collections), 14)

    def test_rank_zero_writer_failure_reaches_every_rank_promptly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            started = time.monotonic()
            ranks = run_four(root, 'write-failure', writer_failure=True, limit=1)
            self.assertEqual({r['error'] for r in ranks},
                {'rank 0 checkpoint write failed: OSError: injected rank zero checkpoint write failure'})
            self.assertEqual([r['saves'] for r in ranks], [['train-0.pt']]*4)
            self.assertFalse((root/'run/state.json').exists())
            self.assertEqual(list((root/'run/artifacts').iterdir()), [])
            print('ALL_RANKS_REPORTED_CHECKPOINT_FAILURE', len(ranks), round(time.monotonic()-started, 3), flush=True)

    def test_four_rank_commit_restart_retention_and_authenticated_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/'run'
            (out/'artifacts').mkdir(parents=True)
            orphan = out/'artifacts/train-0-orphan.pt'
            orphan.touch()
            policy = run.EvaluationProtocol('c'*64, 16, 'deferred', 'e'*64)
            first = run_four(root, 'prefix', evaluation=policy, limit=2, publication_failure=True)
            self.assertEqual([r['exit'] for r in first], [0]*4)
            self.assertEqual([r['state'] for r in first], [[1, 2], [], [], []])
            self.assertEqual([len(r['exports']) for r in first], [2, 0, 0, 0])
            self.assertEqual(len({tuple(r['saves']) for r in first}), 1)
            self.assertRegex(first[0]['saves'][0], r'^train-0-[0-9a-f]{32}\.pt$')
            request = out/'artifacts/cpu-panels/16/request.json'
            self.assertFalse(request.exists())
            refused = run_four(root, 'invalid-resume', evaluation=policy, invalid_resume_rank=3)
            self.assertEqual({r['error'] for r in refused},
                {'rank 3 checkpoint resume failed: original checkpoint SHA256 differs'})
            self.assertEqual([r['collect'] for r in refused], [[], [], [], []])
            self.assertFalse(request.exists())
            second = run_four(root, 'resume', evaluation=policy, stop=True)
            self.assertEqual([r['exit'] for r in second], [0]*4)
            self.assertEqual([r['state'] for r in second], [[3, 4], [], [], []])
            self.assertEqual([r['collect'] for r in second], [[2, 3]]*4)
            self.assertEqual([r['issues'][0][0] if r['issues'] else None for r in second], ['STOP', None, None, None])
            self.assertEqual(len({tuple(r['saves']) for r in second}), 1)
            state = json.loads((out/'state.json').read_bytes())
            snapshots = state['snapshots'][1:]
            self.assertEqual(json.loads(request.read_bytes())['request']['snapshot'], snapshots[1])
            self.assertEqual(json.loads((out/'report.json').read_bytes())['status'], 'STOPPED')
            self.assertEqual([(out/s['checkpoint']).is_file() for s in snapshots], [False, True, False, True])
            self.assertTrue(all((out/s['archive']).is_file() for s in snapshots))
            self.assertTrue(orphan.exists())


if __name__ == '__main__':
    unittest.main()
