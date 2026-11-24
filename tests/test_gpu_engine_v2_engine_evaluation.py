"""Fixtures for the evaluation-only V2 PROGRESS panel."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from gpu_engine import v2_engine_evaluation as evaluation


def _bank(count):
    return [dict(offset=i, seed=2**32 - count + i, teams={'p2': f'right{i}', 'p1': f'left{i}'})
            for i in range(count)]


class SchedulingTests(unittest.TestCase):
    def test_registered_shard_subset_preserves_both_seats_and_teams(self):
        bank = _bank(4000)
        cells = evaluation.schedule(bank)
        for pair, row in zip(zip(cells[::2], cells[1::2], strict=True), bank, strict=True):
            left, right = pair
            self.assertEqual(right, replace(left, seat=1))
            self.assertEqual((left.seed, left.seat, left.teams),
                             (row['seed'], 0, (row['teams']['p1'], row['teams']['p2'])))
        registered = (12, 28, 4, 24, 9, 1, 13, 21, 0, 10)
        offsets = {i for shard in registered for i in range(shard, len(bank), 36)}
        outcomes = [evaluation.Terminal(c.seed, c.seat, 10, False, c.seat if c.offset in offsets else 2)
                    for c in cells]
        result = evaluation.score(cells, outcomes, max_requests=250, interim=True)
        self.assertEqual(result['games'], 8000)
        self.assertEqual(result['wins'], 2224)
        self.assertEqual(result['registered_subset']['games'], 2224)
        self.assertEqual(result['registered_subset']['win_rate'], 1)

    def test_terminal_precedence_and_cap_loss(self):
        cap = 250
        classify = evaluation.disposition
        self.assertEqual(classify(live=True, refused=False, done=False, requests=cap, cap=cap), (False, False, True))
        self.assertEqual(classify(live=True, refused=False, done=True, requests=cap, cap=cap), (False, True, False))
        self.assertEqual(classify(live=True, refused=True, done=True, requests=cap, cap=cap), (True, False, False))
        self.assertEqual(classify(live=False, refused=True, done=True, requests=cap, cap=cap), (False, False, False))
        cells = evaluation.schedule(_bank(2))
        outcomes = [evaluation.Terminal(c.seed, c.seat, cap, False, c.seat) for c in cells]
        outcomes[0] = replace(outcomes[0], capped=True, winner=None)
        outcomes[1] = replace(outcomes[1], winner=2)
        stats = evaluation.score(cells, reversed(outcomes), max_requests=cap)
        self.assertEqual((stats['games'], stats['wins'], stats['caps'], stats['turn_cap_rate']), (4, 2, 1, .25))
        with self.assertRaisesRegex(ValueError, 'invalid terminal'):
            evaluation.score(cells, [replace(outcomes[0], winner=0), *outcomes[1:]], max_requests=cap)

    def test_missing_duplicate_and_unregistered_ends_are_refused(self):
        cells = evaluation.schedule(_bank(1))
        outcomes = [evaluation.Terminal(c.seed, c.seat, 1, False, 0) for c in cells]
        with self.assertRaisesRegex(ValueError, 'missing terminals'):
            evaluation.score(cells, outcomes[:1], max_requests=10)
        with self.assertRaisesRegex(ValueError, 'duplicate terminal'):
            evaluation.score(cells, outcomes + outcomes[:1], max_requests=10)
        with self.assertRaisesRegex(ValueError, 'unexpected.*terminal'):
            evaluation.score(cells, [replace(outcomes[0], seed=42), outcomes[1]], max_requests=10)


class DeviceTests(unittest.TestCase):
    def test_registered_set_matches_showdown_worker(self):
        from pokezero.v5_progress_evaluation import shard_offsets

        count = evaluation.PANEL_PAIRS['champion_9375']
        selected = {i for shard in evaluation.INTERIM_SHARDS for i in shard_offsets(count, 36, shard, 1)[0]}
        self.assertEqual(selected, {i for i in range(count) if i % 36 in evaluation.INTERIM_SHARDS})
        self.assertEqual(len(selected) * 2, 2224)

    def test_champion_slot_reads_the_forward_layout(self):
        import jax
        import numpy as np
        from gpu_engine import champion, engine_device_observation as encoder
        from gpu_engine.v2_engine_evaluation_device import champion_slot
        from tests.test_gpu_engine_champion_evaluator import tiny_champion, v2_inputs

        evaluator = tiny_champion('float32')
        inputs = v2_inputs(evaluator.config, 4)
        observation = SimpleNamespace(**{k: v[:, 0] for k, v in inputs.items() if k != 'history_mask'},
                                      error=np.zeros(4, np.int32))
        paired = jax.device_get(champion_slot(evaluator, evaluator.params, observation))
        direct = jax.device_get(champion.forward(evaluator.params, **encoder.champion_inputs(observation),
                                                c=evaluator.config, compute_dtype='float32'))
        self.assertTrue(paired.valid.all())
        np.testing.assert_array_equal(paired.policy_logits, direct.policy_logits)
        np.testing.assert_array_equal(paired.value, direct.value)

    def test_max_damage_matches_the_host_scores(self):
        import jax.numpy as jnp
        import numpy as np
        from gpu_engine import v2_engine_eval_max_damage as device
        from pokezero.dex import ShowdownDex, SpeciesInfo
        from pokezero.policy import _max_damage_estimate

        def move(id, power, kind, *, accuracy=100, priority=0, recoil=False, selfdestruct=False, status=False):
            return SimpleNamespace(id=id, name=id, type=kind, gen3_category='Status' if status else 'Physical',
                                   base_power=power, accuracy=accuracy, priority=priority, recoil=recoil,
                                   selfdestruct=selfdestruct)

        moves = [move('first', 33, 'Fire', accuracy=80, recoil=True),
                 move('second', 120, 'Fire', accuracy=20, selfdestruct=True),
                 move('third', 40, 'Normal', priority=1), move('fourth', 15, 'Fire', status=True),
                 move('hiddenpower', 0, 'Normal'), move('hiddenpowerice', 70, 'Ice')]
        own = SpeciesInfo('firstmon', 'firstmon', ('Fire',), {})
        foe = SpeciesInfo('secondmon', 'secondmon', ('Ghost', 'Steel'), {})
        dex = ShowdownDex({m.id: m for m in moves}, {own.id: own, foe.id: foe},
                          {'ghost': {'normal': 3}, 'steel': {'fire': 1}})
        vocab = {'species': ['_none', own.id, foe.id], 'moves': ['_none', *(m.id for m in moves)]}
        tables = device.compile_tables(vocab, dex)
        hps = [6, 7, 8]
        actual = device.scores(tables, jnp.tile(jnp.arange(1, len(moves) + 1), (3, 1)), jnp.ones(3, jnp.int32),
                               jnp.full(3, 2), jnp.asarray(hps), jnp.full(3, 20))
        request_ids = [m.id for m in moves[:-1]] + ['hiddenpower']
        expected = [[_max_damage_estimate(dict(action_index=i, move_id=move_id), dex, own.types, foe.types, hp / 20)
                     for i, move_id in enumerate(request_ids)] for hp in hps]
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float64) / 80000, expected, atol=1e-9, rtol=0)

    def test_evaluate_refuses_relative_receipts_inside_training_root(self):
        from contextlib import chdir
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from gpu_engine import v2_engine_eval_runner as runner
        from pokezero import v5_deferred_evaluation as deferred

        ref = SimpleNamespace(sha256='a' * 64)
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            training = root / 'training'
            training.mkdir()
            alias = root / 'alias'
            alias.symlink_to(training, target_is_directory=True)
            request = SimpleNamespace(run_directory=training)
            with chdir(training), patch.object(deferred, 'read_owned_milestone', return_value=request):
                for output in (Path('receipt.json'), Path('../alias/receipt.json')):
                    with self.subTest(output=output):
                        with self.assertRaisesRegex(ValueError, 'outside the immutable training root'):
                            runner.evaluate(ref, ref, output, source_sha='b' * 40, batch_size=1)
            self.assertFalse((training / 'receipt.json').exists())


def verify_registered_starts(config):
    """CPU fixture using the pinned 9375 archive and real registered starts."""
    import os
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpu_engine import champion, engine_device_observation as encoder, transition
    from gpu_engine import v2_engine_eval_runner as runner, v2_engine_evaluation_device as device

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise ValueError('launch the device fixtures with CPU-only settings')
    if any(backend.platform != 'cpu' for backend in jax.devices()):
        raise ValueError('device fixtures require CPU')
    profile, show, metadata, _, _, _, engine, actor, actor_tables = runner._prepare(config)
    bank = runner._artifact(config['banks']['champion_9375']).document()[:2]
    cells = evaluation.schedule(bank)
    arrays, gender = runner._decode(bank, metadata=metadata, show=show)
    raw, _ = runner._raw(arrays, gender, cells)
    for value in (raw.species, raw.moves, raw.hp, raw.keys, raw.gender):
        np.testing.assert_array_equal(value[::2], value[1::2])
    print('Mirrored starts verified', flush=True)
    state, public = transition.initialize_with_public(raw, engine, belief_tables=actor_tables.belief)
    owners = 1 - jnp.asarray([cell.seat for cell in cells], jnp.int32)
    observation = device._observe(state, public, owners, transition.action_mask(state, engine), actor_tables)
    paired = jax.device_get(device.champion_slot(actor, actor.params, observation))
    reference = jax.device_get(champion.forward(actor.params, **encoder.champion_inputs(observation),
                                              c=actor.config, compute_dtype='float32'))
    np.testing.assert_array_equal(paired.policy_logits, reference.policy_logits)
    np.testing.assert_array_equal(paired.value, reference.value)
    np.testing.assert_array_equal(paired.valid, np.ones(len(cells), bool))
    np.testing.assert_array_equal(observation.error, np.zeros(len(cells), np.int32))
    return dict(champion=actor.checkpoint_sha256,
                logits_bitwise=True, values_bitwise=True, registered_request_cap=profile['max_requests'],
                mirror_states=len(cells))


if __name__ == '__main__':
    unittest.main()
