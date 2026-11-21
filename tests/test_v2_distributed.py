"""Global masked-loss equivalence, row accounting, and distributed resume.

CPU/gloo controls deliberately use dropout zero for a rank-count comparison;
independent rank dropout streams are checked separately by exact same-rank
resume. The actual NCCL/collector gates remain required on allocated GPUs.
"""
from dataclasses import replace
import datetime
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pokezero import checkpoint_self_moves as encoding, v2_distributed as vd, v2_trainer as v2
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
        checkpoint = Path(directory) / 'resume.pt'
        receipt = [v2.save_checkpoint(checkpoint, runtime)]
        dist.broadcast_object_list(receipt, src=0, group=dist.group.WORLD)
        binding = dict(expected_self_moves=False, expected_checkpoint_sha256=receipt[0]['sha256'],
                       off_provenance=encoding.off_from_receipt(receipt[0]['encoding_receipt']))
        suffix = trainer.train(store)
        expected = {k: t.clone() for k, t in runtime.model.state_dict().items()}
        resumed = v2.load_checkpoint(checkpoint, enable_v5=True, process_group=dist.group.WORLD, **binding)
        again = vd.DistributedTrainer(resumed).train(store)
        for k, t in resumed.model.state_dict().items():
            if not torch.equal(t, expected[k]):
                raise AssertionError('same-rank resume differs: ' + k)
        if again['updates'] != suffix['updates']:
            raise AssertionError('resume metrics/cursor differ')
        try:
            v2.load_checkpoint(checkpoint, enable_v5=True, v2_config=v2.V2Config(), process_group=dist.group.WORLD,
                               **binding)
        except ValueError as exc:
            if 'semantics' not in str(exc):
                raise
        else:
            raise AssertionError('legacy loss semantics accepted a distributed checkpoint')
        if rank == 0:
            torch.save(dict(state=expected, updates=prefix['updates'] + suffix['updates'],
                steps=runtime.steps, rows=runtime.rows, counts=store.distributed_layout.counts,
                outputs=fixture.eval_outputs(runtime.model, full.tensors),
                receipt=dict(sha256=receipt[0]['sha256'], encoding_sha256=receipt[0]['encoding_receipt']['sha256'])),
                Path(directory) / 'result.pt')
    finally:
        dist.destroy_process_group()


def learner_node_test(rank, world, directory, learners, chunk_bytes=None):
    """Every rank collects a row shard; only ranks < learners train (as a learners-wide run)."""
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(Path(directory) / 'rendezvous'),
        rank=rank, world_size=world, timeout=datetime.timedelta(seconds=90))
    group = dist.new_group(ranks=list(range(learners)))
    try:
        model, loss, train, full, keys, batches = example()
        selected = np.flatnonzero(np.arange(full.rows) % world == rank)
        local = v2.row_store({**{k: t[selected] for k, t in full.tensors.items()}, **full.bindings,
            'terminal_status': tuple(full.terminal_status[i] for i in selected)}, model, loss, store_id='fixed')
        config = v2.V2Config(mode='eager', optimizer='single', distributed_rows=True, check_interval=3)
        exchange = vd.Exchange(rank, world, dist.group.WORLD, learners)
        runtime = v2.create(model, loss, train, config, enable_v5=True, process_group=group) if rank < learners else None
        store = vd.distribute_store(local, keys[selected], batches, runtime, exchange=exchange,
                                    configs=(model, loss, train, config), chunk_bytes=chunk_bytes)
        if rank >= learners:
            if store is not None:
                raise AssertionError('collector-only ranks must not receive rows')
            return
        report = vd.DistributedTrainer(runtime).train(store)
        if rank == 0:
            torch.save(dict(state={k: t.clone() for k, t in runtime.model.state_dict().items()}, updates=report['updates'],
                steps=runtime.steps, rows=runtime.rows, counts=store.distributed_layout.counts, store_id=store.store_id,
                outputs=fixture.eval_outputs(runtime.model, full.tensors)), Path(directory) / 'learner.pt')
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

    def run_ranks(self, world, dropout=0., keep=None):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(rank_test, args=(world, directory, dropout), nprocs=world, join=True)
            if keep is not None:   # the shared checkpoint and its writer receipt, for cross-topology resume checks
                for name in ('resume.pt', 'resume.pt.encoding.json'):
                    shutil.copy(Path(directory) / name, Path(keep) / name)
            return torch.load(Path(directory) / 'result.pt', weights_only=True)

    def assert_matches_reference(self, result, reference, runtime, store):
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

    def test_four_ranks_match_original_global_update_and_resume(self):
        model, loss, train, store, _, _ = example()
        runtime = v2.create(model, loss, train, v2.V2Config(mode='eager', optimizer='single'), enable_v5=True)
        reference = v2.Trainer(runtime).train(store)
        self.assert_matches_reference(self.run_ranks(4), reference, runtime, store)

    def test_eight_ranks_match_original_global_update_and_resume(self):
        # Two nodes x four ranks: the learner only sees the world size (local batch 1 of the global 8),
        # so the same statistics tolerance as four ranks applies despite a different reduction order.
        model, loss, train, store, _, _ = example()
        runtime = v2.create(model, loss, train, v2.V2Config(mode='eager', optimizer='single'), enable_v5=True)
        reference = v2.Trainer(runtime).train(store)
        self.assert_matches_reference(self.run_ranks(8), reference, runtime, store)

    def test_checkpoint_from_eight_ranks_refuses_a_four_rank_or_single_resume(self):
        with tempfile.TemporaryDirectory() as keep:
            result = self.run_ranks(8, keep=keep)
            checkpoint = Path(keep) / 'resume.pt'
            provenance = encoding.off_from_receipt(dict(path=str(checkpoint) + '.encoding.json',
                                                        sha256=result['receipt']['encoding_sha256']))
            binding = dict(enable_v5=True, expected_self_moves=False, off_provenance=provenance,
                           expected_checkpoint_sha256=result['receipt']['sha256'],
                           v2_config=v2.V2Config(mode='eager', optimizer='single', distributed_rows=True, check_interval=3))
            for world in (1, 4):   # a single process and a four-rank group both differ from the recorded eight
                with self.assertRaisesRegex(ValueError, 'original world size'):
                    v2._restore_checkpoint(checkpoint, expected_model_config=None, expected_loss_config=None,
                                           expected_train_config=None, device=None, process_group=None,
                                           world_size=world, **binding)

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

    def ddp_reference(self, world):
        """Uninterrupted ddp run of ``world`` learners (no mid-run checkpoint) for bit comparisons."""
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(learner_node_test, args=(world, directory, world), nprocs=world, join=True)
            return torch.load(Path(directory) / 'learner.pt', weights_only=True)

    def test_learner_node_eight_collectors_four_learners_is_bit_identical_to_four_rank_ddp(self):
        self.check_learner_node(chunk_bytes=None)

    def test_chunked_exchange_is_bit_identical(self):
        # 64-byte messages force many slabs per field (rows are hundreds of bytes); same rows, same slots.
        self.check_learner_node(chunk_bytes=64)

    def check_learner_node(self, *, chunk_bytes):
        import torch.multiprocessing as mp
        reference = self.ddp_reference(4)
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(learner_node_test, args=(8, directory, 4, chunk_bytes), nprocs=8, join=True)
            result = torch.load(Path(directory) / 'learner.pt', weights_only=True)
        self.assertEqual((result['steps'], result['rows'], result['counts']), (reference['steps'], reference['rows'], reference['counts']))
        self.assertEqual(result['store_id'], reference['store_id'])   # same canonical digest, learner world 4
        self.assertEqual(result['updates'], reference['updates'])
        for key, tensor in result['state'].items():
            self.assertTrue(torch.equal(tensor, reference['state'][key]), key)
        for actual, wanted in zip(result['outputs'], reference['outputs']):
            self.assertTrue(torch.equal(actual, wanted))


if __name__ == '__main__':
    unittest.main()
