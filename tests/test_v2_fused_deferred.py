"""Run fused round coordination with real V2 training, checkpoints and CPU/Gloo ranks."""
from contextlib import ExitStack
from dataclasses import asdict, make_dataclass
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
                publication_failure=False, stop=False, invalid_resume_rank=None, fused_options=(), campaign=False, advance=None,
                parent_window=False, cooldown_from=None):
    from gpu_engine import v2_bench_trainer, v2_collector, v5_device_backend, v5_model

    torch.set_num_threads(1)
    rank = 0 if distribution is None else distribution.rank
    model, loss, train = configurations(batch_size=8, epochs=2, dropout=.2, schedule='mit-thesis')
    batches = tuple((run.Batch(tuple(run.Game(i+1, f'game-{i}', 100+i, (2*i, 2*i+1))
                              for i in range(8*j, 8*j+8)), 10+j, 20+j),) for j in range(5))
    declaration = dict(fixture='fused-boundary', evaluation=None if evaluation is None else evaluation.metadata())
    plan_fields = dict(model=model, loss=loss, train=train, evaluation=evaluation, self_moves=False,
        iterations=batches, games=tuple(g for b, in batches for g in b.games), discount=1.,
        max_requests=64, historical_fraction=.5, historical_window=2,
        manifest=lambda: declaration, evaluation_milestones=lambda: {1: 16, 3: 32})
    if cooldown_from is not None:
        from pokezero.v5_phasec import campaign_iterations
        plan_fields['compact'] = True
        plan_fields['iterations'] = campaign_iterations(
            'fixture', 20, stream_start=10, team_start=20, identity_seed_start=30,
            constructor_seed_start=40, sampling_seed_start=50, gender_seed=60, gender_domain=7, gender_root_start=80)
        plan_fields['game_offsets'] = tuple(range(0, 400_001, 20_000))
    plan_type = make_dataclass('CampaignFixturePlan', plan_fields) if campaign else SimpleNamespace
    plan = plan_type(**plan_fields)
    context = dict(plan=plan, declared={}, preparation_seconds=1, gender=None, pool=None,
        prepared=SimpleNamespace(engine_tables=None, observation_tables=None))
    snapshot = run.Snapshot(-1, 'initial.pt', 'a'*64, 'initial.npz', 'b'*64, run._sha(model.metadata()))
    events = [{'kind': 'SNAPSHOT', 'payload': asdict(snapshot)}]
    initial = trainer.create(model, loss, train, enable_v5=True, device='cpu')
    argv = ['--config', 'unused.json', '--reference-run', str(root), '--out', str(out),
            '--torch-memory-fraction', '0', '--mode', 'eager', '--lanes', '8', *fused_options]
    if limit is not None:
        argv += ['--max-iterations', str(limit)]
    if campaign:
        argv += ['--lr-start', 'sqrt', '--lr-warmup-updates', '2', '--campaign-anneal']
    if cooldown_from is not None:
        argv += ['--cooldown-child', str(root / 'run'), '--cooldown-parent-sha256', cooldown_from]
    if advance is not None:
        argv += ['--advance-campaign-step-at', str(advance)]
    phases = dict(saves=[], exports=[], state=[], collect=[], issues=[])

    class Rows:
        def __init__(self, iteration, begin, end):
            self.n = end - begin
            self.stats = {'dispositions': {'COMPLETE': self.n}}
            self.arrays = {}
            start = plan.iterations[iteration][0].games.stream_start if cooldown_from is not None else 8 * iteration + 1
            self.addresses = np.column_stack((np.arange(start + begin, start + end),
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
        rows = (2 * rank, 2 * (rank + 1)) if cooldown_from is not None else (begin, end)
        roster = SimpleNamespace(groups=(None,), games=end-begin, actor_archives=(current.archive_sha256,),
                                 fixture=(iteration, *rows))
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
            phases['state'].append(payload['round_ledger']['completed_rounds'] if cooldown_from else len(payload['reports']))
            assert options.get('durable', False) == (evaluation is not None or cooldown_from is not None)
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
        games_per_round = 2_000_000 if parent_window else (20_000 if cooldown_from else 8)
        declared_games = 2_000_000 if parent_window else (400_000 if cooldown_from else 40)
        replace(run, learning_rate_window=lambda _, i: v5_training.DeclaredGameWindow(
            games_per_round * i, games_per_round * (i + 1), declared_games))
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
        output = 'cooled' if options.get('cooldown_from') else 'run'
        phases = run_fixture(root, root/output, distribution=launcher.Distribution(rank, 4, dist.group.WORLD), **options)
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


class CollectorAuditTests(unittest.TestCase):
    def test_empty_audit_does_not_touch_the_device_store(self):
        self.assertEqual(v2_fused.audit(None, [], None), [])

    @unittest.skipUnless(torch is not None, 'torch required')
    def test_audit_gathers_only_valid_rows_without_jax_gather(self):
        import jax.numpy as j
        from gpu_engine.v2_collector import RowStore

        arrays = dict(streams=j.array([3, 7, 5, 7], dtype=j.int32),
                      mask=j.array([True, False, True, True]),
                      values=j.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=j.float32))
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(j, 'take', side_effect=AssertionError),
              patch.object(v2_fused, '_AUDIT_GATHER_ROWS', 2),
              patch.object(torch, 'as_tensor', wraps=torch.as_tensor) as take):
            for n, indices in ((4, [1, 2, 3]), (3, [1, 2]), (2, [1])):
                store = RowStore(arrays, n, {}, {})
                sampled, empty = v2_fused.audit(store, [(1, 5, 7), (2, 8, 9)], directory)
                self.assertEqual(sampled['rows'], len(indices))
                self.assertEqual(empty['rows'], 0)
                for report, expected in ((sampled, indices), (empty, [])):
                    with np.load(Path(report['path']), allow_pickle=False) as written:
                        self.assertEqual(set(written.files), set(arrays))
                        for name, field in arrays.items():
                            self.assertEqual(written[name].dtype, field.dtype)
                            np.testing.assert_array_equal(written[name], np.asarray(field)[expected])
            self.assertEqual([call.args[0].shape for call in take.call_args_list], [(2,)] * 4)


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

    def test_four_rank_campaign_advances_once_and_restores_the_sealed_schedule(self):
        from pokezero.v2_campaign_schedule import BACKSTOP_STEPS, FINAL_COOLDOWN
        from pokezero.v2_lr_start import scaled_peak_rate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_four(root, 'campaign-prefix', campaign=True, limit=1)
            self.assertEqual([entry['exit'] for entry in first], [0] * 4)
            second = run_four(root, 'campaign-cut', campaign=True, advance=8, limit=2)
            self.assertEqual([entry['exit'] for entry in second], [0] * 4)
            third = run_four(root, 'campaign-resume', campaign=True, limit=3)
            self.assertEqual([entry['exit'] for entry in third], [0] * 4)
            state = json.loads((root/'run/state.json').read_bytes())
            self.assertEqual(state['distribution']['declared']['campaign_anneal']['final_cooldown'], list(FINAL_COOLDOWN))
            reports = state['reports']
            self.assertEqual([report['training']['campaign_schedule']['step_down_games'] for report in reports],
                             [list(BACKSTOP_STEPS), [8, BACKSTOP_STEPS[1]], [8, BACKSTOP_STEPS[1]]])
            peak = scaled_peak_rate(8, 'sqrt')
            rates = [report['training']['iteration_mean']['learning_rate'] for report in reports]
            self.assertEqual(rates, [.75 * peak, .3 * peak, .3 * peak])

    def test_four_rank_cooldown_forks_and_resumes_without_touching_parent(self):
        from pokezero import v2_fused_state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_four(root, 'parent', campaign=True, parent_window=True, limit=1)
            manifest = root / 'run' / 'state.json'
            state = json.loads(manifest.read_bytes())
            parent = state['snapshots'][1]
            parent['iteration'] = 99
            # The tiny optimized checkpoint has the true 2M completed window.
            # Expand only the fixture ledger up to that production boundary.
            first = state['reports'][0]
            state['reports'] = [dict(first, iteration=index) for index in range(100)]
            original = json.dumps(state).encode()
            manifest.write_bytes(original)
            checkpoint = root / 'run' / parent['checkpoint']
            checkpoint_sha = v2_fused.sha(checkpoint)
            options = dict(campaign=True, cooldown_from=checkpoint_sha)
            mismatched = json.loads(original)
            mismatched['reports'][-1]['training']['rows'] += 1
            manifest.write_text(json.dumps(mismatched))
            from torch.multiprocessing.spawn import ProcessRaisedException
            with self.assertRaisesRegex(ProcessRaisedException, 'checkpoint counters differ'):
                run_four(root, 'cooled-bad-origin', limit=1, **options)
            self.assertFalse((root / 'cooled' / v2_fused_state.PARENT_RECEIPT).exists())
            manifest.write_bytes(original)
            self.assertEqual([rank['exit'] for rank in run_four(root, 'cooled-first', limit=1, **options)], [0] * 4)
            self.assertEqual([rank['exit'] for rank in run_four(root, 'cooled-resume', limit=2,
                             fused_options=('--require-resume',), **options)], [0] * 4)
            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual(v2_fused.sha(checkpoint), checkpoint_sha)
            cooled = v2_fused_state.read_state(root / 'cooled')
            windows = [report['training']['lr_start']['declared_window'] for report in cooled['reports']]
            self.assertEqual([(window['start_game'], window['end_game']) for window in windows],
                             [(2_000_000, 2_020_000), (2_020_000, 2_040_000)])
            self.assertEqual(cooled['cooldown_parent']['snapshot']['checkpoint_sha256'], checkpoint_sha)

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
