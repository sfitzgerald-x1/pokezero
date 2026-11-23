"""CPU fixture for the fused planner, per-seat rows and distributed exchange."""

from dataclasses import replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from gpu_engine import transition, v2_collector as collector, v2_fused as fused, v5_device_rollout as rollout, v5_model
from pokezero import v2_distributed as distributed, v2_trainer as trainer, v5_run as runs
from pokezero.v5_loss import V5LossConfig
from tests import test_gpu_engine_v2_collector as fixture


def collect(*, both=False, flip=False):
    arc = fixture.ARCHIVES
    batches = tuple(runs.Batch(tuple(runs.Game(1000 + 17*i, str(i), i, (2*i, 2*i+1))
        for i in range(lo, hi)), seed, seed) for lo, hi, seed in ((0, 4, 7), (4, 9, 2**31 + 9)))
    _, _, labels, model, _ = fixture.fixture()
    plan = SimpleNamespace(model=model, iterations=(batches,), selection_seed=5,
        historical_fraction=.5, historical_window=32, train_both_seats=both)
    current = runs.Snapshot(0, 'current.pt', '1'*64, 'current.npz', arc[0], '4'*64)
    old = replace(current, iteration=-1, archive_sha256=arc[1])
    ordinal = {game.stream_id: int(game.battle_id) for batch in batches for game in batch.games}

    def constructor(batch):
        return fixture.block([ordinal[g.stream_id] for g in batch.games], [g.stream_id for g in batch.games])[0]

    original_assignment = runs.assignment
    game_kinds = {}

    def assign(*args, **kwargs):
        row = original_assignment(*args, **kwargs)
        game_kinds[row['stream_id']] = row['opponent_kind']
        if flip and row['opponent_kind'] == 'SELFPLAY':
            row['learner_side'] = 1 - row['learner_side']
        return row

    banks = rollout.actor_bank_from_assignments
    bindings = []

    def bind(*args, **kwargs):
        bank = banks(*args, **kwargs)
        bindings.append(bank.assignment_sha256)
        return bank

    context = dict(plan=plan, prepared=SimpleNamespace(source_sha256='2'*64))
    backend = SimpleNamespace(declaration_sha256='3'*64, _construct=constructor)
    with patch.object(runs, 'assignment', assign), patch.object(rollout, 'actor_bank_from_assignments', bind):
        roster, kinds, _ = fused.construct(context, backend, 0, current, (old,), ordinal)
    loss = V5LossConfig()

    def seat_values(params, inputs, config, *, compute_dtype):
        output = fixture.forward(params, inputs, config, compute_dtype=compute_dtype)
        side = inputs['numeric_features'][:, 0, 0, 3]
        return output._replace(value=jnp.clip(output.value + .125 * side, -.99, .99))

    def seat_masks(state, tables):
        return fixture.action_mask(state, tables).at[:, 0, 8].set(False).at[:, 1, 7].set(False)

    with (fixture.seams(), patch.object(v5_model, 'forward', seat_values),
          patch.object(transition, 'action_mask', seat_masks)):
        refill = collector.RefillCollector(SimpleNamespace(struggle=jnp.int32(2)), SimpleNamespace(belief=None),
            model, labels, roster, lanes=6, min_lanes=2, max_requests=fixture.MAX_REQUESTS,
            chunk_steps=fixture.CHUNK, rows_per_game=16 if both else 8)
        result = refill.run(fixture.params_for(roster.actor_archives))
        store = collector.finalize(result, loss_config=loss, discount=.9999)
    return SimpleNamespace(model=model, loss=loss, store=store, keys=store.keys(), ordinal=ordinal,
        batches=tuple(tuple(game.stream_id for game in batch.games) for batch in batches),
        records=collector.game_records(result), kinds=kinds, game_kinds=game_kinds, bindings=bindings)


def row_digest(round_):
    digest = hashlib.sha256()
    for name, value in sorted(round_.store.to_numpy().items()):
        digest.update(json.dumps((name, value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return dict(rows=len(round_.keys), row_sha256=digest.hexdigest(), assignment_sha256=round_.bindings,
        exchange_sha256=distributed.canonical_layout(round_.keys, round_.batches, 8, 2).digest)


def exchange_rank(rank, directory):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    path = Path(directory)
    state = torch.load(path / 'rows.pt', weights_only=False)
    keys, batches, model, loss = (state[k] for k in ('keys', 'batches', 'model', 'loss'))
    source = np.flatnonzero(np.isin(keys[:, 0], batches[rank]))
    columns = state['columns']
    local = trainer.row_store({name: value[source] if isinstance(value, torch.Tensor) else
        tuple(value[i] for i in source) if name == 'terminal_status' else value
        for name, value in columns.items()}, model, loss, store_id='fixture')
    dist.init_process_group('gloo', init_method='file://' + str(path / 'rendezvous'), rank=rank,
        world_size=2, timeout=timedelta(seconds=60))
    try:
        runtime = SimpleNamespace(rank=rank, world_size=2, process_group=dist.group.WORLD,
            train_config=SimpleNamespace(batch_size=8), model_config=model, loss_config=loss,
            v2_config=SimpleNamespace(order='ordered', distributed_rows=True))
        exchanged = distributed.distribute_store(local, keys[source], batches, runtime)
    finally:
        dist.destroy_process_group()
    expected = distributed.canonical_layout(keys, batches, 8, 2)
    received = np.flatnonzero(expected.destinations == rank)
    slots = expected.slots[received]
    np.testing.assert_array_equal(exchanged.tensors['training_weights'].nonzero().flatten().numpy(), np.sort(slots))
    for name, value in exchanged.tensors.items():
        torch.testing.assert_close(value[slots], columns[name][received], atol=0, rtol=0)
    if exchanged.distributed_layout.digest != expected.digest:
        raise AssertionError('exchange digest differs from canonical addresses')
    (path / f'{rank}.json').write_text(json.dumps(received.tolist()))


class BothSeatsTests(unittest.TestCase):
    def test_seat_projections_terminal_targets_and_exchange(self):
        import torch
        import torch.multiprocessing as mp

        original, flipped, both = collect(), collect(flip=True), collect(both=True)
        expected_parities = {both.ordinal[stream] % 2 for stream, kind in both.game_kinds.items() if kind == 'SELFPLAY'}
        self.assertEqual(expected_parities, {0, 1})
        self.assertGreater(both.kinds['HISTORICAL'], 0)
        self.assertEqual(original.kinds, both.kinds)
        for name in both.records:
            np.testing.assert_array_equal(original.records[name], both.records[name], err_msg=name)
        self.assertNotEqual(original.bindings, both.bindings)
        expected = {tuple(key): (original, idx) for idx, key in enumerate(original.keys)}
        expected.update({tuple(key): (flipped, idx) for idx, key in enumerate(flipped.keys)
            if flipped.game_kinds[key[0]] == 'SELFPLAY'})
        self.assertEqual(len(both.keys), len(expected))
        self.assertEqual(set(map(tuple, both.keys)), set(expected))
        projections = {id(round_): round_.store.to_numpy() for round_ in (original, flipped)}
        columns = both.store.to_numpy()
        indices = {tuple(key): i for i, key in enumerate(both.keys)}
        paired = [(i, indices[(stream, frame, 1)]) for (stream, frame, owner), i in indices.items()
                  if owner == 0 and (stream, frame, 1) in indices]
        self.assertTrue(paired)
        for i, j in paired:
            self.assertNotEqual(columns['value_estimates'][i], columns['value_estimates'][j])
            self.assertFalse(np.array_equal(columns['legal_action_mask'][i], columns['legal_action_mask'][j]))
        for i, key in enumerate(both.keys):
            match, idx = expected[tuple(key)]
            for name, value in columns.items():
                self.assertEqual(value[i].tobytes(), projections[id(match)][name][idx].tobytes(), (tuple(key), name))
        complete = []
        for stream in both.ordinal:
            eligible = (both.keys[:, 0] == stream) & columns['ppo_value_target_mask']
            if both.game_kinds[stream] != 'SELFPLAY' or not eligible.any():
                continue
            last = [np.flatnonzero(eligible & (both.keys[:, 2] == owner))[-1] for owner in (0, 1)]
            value = columns['ppo_value_targets'][last]
            outcome = columns['outcome_targets'][last]
            np.testing.assert_allclose(value, outcome, atol=1e-6, rtol=0)
            self.assertEqual(outcome[0], -outcome[1])
            if outcome[0] != 0:
                complete.append(int(stream))
        self.assertTrue(complete)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.save(dict(keys=both.keys, batches=both.batches, columns=both.store.to_torch(),
                model=both.model, loss=both.loss), path / 'rows.pt')
            mp.spawn(exchange_rank, args=(directory,), nprocs=2)
            seen = [index for rank in range(2) for index in json.loads((path / f'{rank}.json').read_text())]
        self.assertEqual(sorted(seen), list(range(len(both.keys))))
        off = row_digest(original)
        if bound := os.environ.get('V2_BOUND_FIXTURE'):
            self.assertEqual(off, json.loads(bound))
        print(json.dumps(dict(off=off, on=row_digest(both),
            completed_selfplay=complete, kinds=both.kinds, exchanged=len(seen)), sort_keys=True))

    def test_cgroup_only_memory_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / 'slice' / 'scope'
            leaf.mkdir(parents=True)
            proc = root / 'proc'
            proc.write_text('0::/slice/scope\n')
            for parent, cap, current in ((root, 'max', 1000), (leaf.parent, '800', 500), (leaf, '900', 300)):
                (parent / 'memory.max').write_text(cap)
                (parent / 'memory.current').write_text(str(current))
            (leaf / 'memory.peak').write_text('350')
            self.assertEqual(fused.host_memory(root, proc), dict(cgroup_limit_bytes=800,
                cgroup_available_bytes=300, cgroup_peak_bytes=350))
