"""Global masked-loss equivalence, row accounting, and distributed resume.

CPU/gloo controls deliberately use dropout zero for a rank-count comparison;
independent rank dropout streams are checked separately by exact same-rank
resume. The actual NCCL/collector gates remain required on allocated GPUs.
"""
from dataclasses import replace
import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pokezero import v2_distributed as vd, v2_trainer as v2
import test_v2_trainer as fixture

try:
    import torch
except ImportError:
    torch = None


def example(dropout=0.):
    model, loss, train = fixture.configurations(batch_size=8, epochs=2, dropout=dropout)
    # Nonuniform masks/weights are essential: averaging rank-local losses
    # would otherwise accidentally look like the correct global reduction.
    batch = fixture.varied_batch(rows=23, seed=93)
    store = fixture.store_of(batch, model, loss, segments=(5, 11, 7), store_id='fixed')
    keys = np.column_stack([np.arange(100, 123), np.zeros(23, dtype=int), np.arange(23) % 2])
    batches = [list(range(100, 105)), list(range(105, 116)), list(range(116, 123))]
    return model, loss, train, store, keys, batches


def rank_test(rank, world, directory, dropout):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(Path(directory) / 'rendezvous'),
        rank=rank, world_size=world, timeout=datetime.timedelta(seconds=90))
    try:
        model, loss, train, full, keys, batches = example(dropout)
        selected = np.flatnonzero(np.arange(full.rows) % world == rank)
        local = v2.row_store({**{k: t[selected] for k, t in full.tensors.items()}, **full.bindings,
            'terminal_status': tuple(full.terminal_status[i] for i in selected)}, model, loss, store_id='fixed')
        config = v2.V2Config(mode='eager', optimizer='single', distributed_rows=True, check_interval=3)
        runtime = v2.create(model, loss, train, config, enable_v5=True, process_group=dist.group.WORLD)
        store = vd.distribute_store(local, keys[selected], batches, runtime)
        trainer = vd.DistributedTrainer(runtime)
        prefix = trainer.train(store, max_updates=3)
        checkpoint = Path(directory) / 'resume.pt'; v2.save_checkpoint(checkpoint, runtime)
        suffix = trainer.train(store)
        expected = {k: t.clone() for k, t in runtime.model.state_dict().items()}
        resumed = v2.load_checkpoint(checkpoint, enable_v5=True, process_group=dist.group.WORLD)
        again = vd.DistributedTrainer(resumed).train(store)
        for k, t in resumed.model.state_dict().items():
            if not torch.equal(t, expected[k]):
                raise AssertionError('same-rank resume differs: ' + k)
        if again['updates'] != suffix['updates']:
            raise AssertionError('resume metrics/cursor differ')
        try:
            v2.load_checkpoint(checkpoint, enable_v5=True, v2_config=v2.V2Config(), process_group=dist.group.WORLD)
        except ValueError as exc:
            if 'semantics' not in str(exc):
                raise
        else:
            raise AssertionError('legacy loss semantics accepted a distributed checkpoint')
        if rank == 0:
            torch.save(dict(state=expected, updates=prefix['updates'] + suffix['updates'],
                steps=runtime.steps, rows=runtime.rows, counts=store.distributed_layout.counts,
                outputs=fixture.eval_outputs(runtime.model, full.tensors)), Path(directory) / 'result.pt')
    finally:
        dist.destroy_process_group()


class LayoutTests(unittest.TestCase):
    def test_constructor_shards_slice_original_blocks_after_rng(self):
        from gpu_engine import v2_fused, v2_collector
        from pokezero import v5_run as runs
        model, _, _ = fixture.configurations()
        games = tuple(runs.Game(i + 1, str(i), i, (2*i, 2*i+1)) for i in range(8))
        batches = (runs.Batch(games[:5], 31, 41), runs.Batch(games[5:], 32, 42))
        plan = SimpleNamespace(model=model, iterations=(batches,), selection_seed=5,
            historical_window=32, historical_fraction=.5)
        context = dict(plan=plan, prepared=SimpleNamespace(source_sha256='a'*64))
        snapshot = runs.Snapshot(-1, 'initial.pt', 'b'*64, 'initial.npz', 'c'*64, 'd'*64)
        called = []
        def constructor(batch):
            called.append(len(batch.games))
            return dict(stream=np.asarray([g.stream_id for g in batch.games]),
                original_position=np.arange(len(batch.games)))
        backend = SimpleNamespace(declaration_sha256='e'*64, _construct=constructor)
        captured = []
        def roster(blocks, **kwargs):
            captured.extend(blocks)
            return SimpleNamespace(device=())
        with patch.object(v2_collector, 'roster_from_blocks', roster):
            v2_fused.construct(context, backend, 0, snapshot, (), {g.stream_id:i for i,g in enumerate(games)}, shard=(4,8))
        self.assertEqual(called, [5, 3])
        self.assertEqual(captured[0][0]['stream'].tolist(), [5])
        self.assertEqual(captured[0][0]['original_position'].tolist(), [4])
        self.assertEqual(captured[1][0]['original_position'].tolist(), [0,1,2])
        self.assertEqual(captured[0][1].streams, (5,))
        self.assertEqual(captured[1][1].streams, (6,7,8))

    def test_recipe_packages_preserve_other_controls(self):
        _, _, reference = fixture.configurations(batch_size=512, epochs=5)
        for name, batch, epochs in [('R5', 512, 5), ('R2', 2048, 2), ('R1', 4096, 1)]:
            actual = vd.recipe_train(reference, name)
            self.assertEqual(actual, replace(reference, batch_size=batch, epochs=epochs,
                learning_rate=reference.learning_rate * batch / 512))
        with self.assertRaises(ValueError):
            vd.recipe_train(replace(reference, epochs=2), 'R2')

    def test_shards_cover_global_games_once(self):
        self.assertEqual([vd.shard_bounds(20000, r, 4) for r in range(4)],
            [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)])
        with self.assertRaises(ValueError):
            vd.shard_bounds(7, 0, 4)

    def test_canonical_order_preserves_segment_tails(self):
        keys = np.array([[12, 1, 0], [11, 1, 0], [11, 0, 1], [12, 0, 0], [13, 0, 0]])
        layout = vd.canonical_layout(keys, [[11, 12], [13]], 4, 2)
        self.assertEqual(layout.counts, (4, 1))
        self.assertEqual(layout.segments, (2, 2))
        self.assertEqual(layout.destinations.tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(layout.slots.tolist(), [1, 0, 0, 1, 2])
        reordered = vd.canonical_layout(keys[::-1], [[11, 12], [13]], 4, 2)
        self.assertEqual(layout.digest, reordered.digest)

    def test_duplicate_foreign_and_invalid_rows_fail_closed(self):
        for keys in ([[[1, 0, 0], [1, 0, 0]]], [[[2, 0, 0]]], [[[1, -1, 0]]], [[[1, 0, 2]]]):
            with self.assertRaises(ValueError):
                vd.canonical_layout(keys[0], [[1]], 4, 2)
        with self.assertRaises(ValueError):
            vd.canonical_layout([[1, 0, 0]], [[1]], 3, 2)


@unittest.skipIf(torch is None, 'torch required')
class DistributedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def run_ranks(self, world, dropout=0.):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(rank_test, args=(world, directory, dropout), nprocs=world, join=True)
            return torch.load(Path(directory) / 'result.pt', weights_only=True)

    def test_four_ranks_match_original_global_update_and_resume(self):
        model, loss, train, store, _, _ = example()
        runtime = v2.create(model, loss, train, v2.V2Config(mode='eager', optimizer='single'), enable_v5=True)
        reference = v2.Trainer(runtime).train(store)
        result = self.run_ranks(4)
        self.assertEqual(result['steps'], reference['steps'])
        self.assertEqual(result['rows'], 46)
        self.assertEqual(result['counts'], (5, 8, 3, 7))
        for actual, expected in zip(result['updates'], reference['updates']):
            for key in ('loss', 'policy_loss', 'value_loss', 'entropy', 'uniform_kl', 'gradient_norm'):
                self.assertAlmostEqual(actual[key], expected[key], delta=2e-5, msg=key)
            for key in ('optimizer_step', 'learning_rate', 'epoch_step'):
                self.assertEqual(actual[key], expected[key])
        # Adam amplifies roundoff in analytically invariant parameter directions;
        # compare predictions as well as update statistics, not these gauge shifts.
        expected = fixture.eval_outputs(runtime.model, store.tensors)
        for actual, wanted in zip(result['outputs'], expected):
            torch.testing.assert_close(actual, wanted, atol=2e-4, rtol=2e-4)

    def test_one_rank_adapter_matches_original(self):
        model, loss, train, store, _, _ = example()
        runtime = v2.create(model, loss, train, v2.V2Config(mode='eager', optimizer='single'), enable_v5=True)
        reference = v2.Trainer(runtime).train(store)
        result = self.run_ranks(1)
        self.assertEqual(result['steps'], reference['steps'])
        for actual, expected in zip(result['updates'], reference['updates']):
            self.assertAlmostEqual(actual['loss'], expected['loss'], delta=2e-5)

    def test_rank_dropout_streams_resume_exactly(self):
        self.run_ranks(2, dropout=.2)


if __name__ == '__main__':
    unittest.main()
