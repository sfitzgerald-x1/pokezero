"""V2-c trainer: equivalence to v5_training, exact resume, schedule, finiteness, DDP, export.

CPU only, with tiny models and the existing v5 fixtures.
"""
from dataclasses import replace
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pokezero import v2_trainer as v2, v5_data, v5_training as v5
from pokezero.dataset import _combine_training_batches
import test_v5_training as fixture

try:
    import torch
except ImportError:
    torch = None


def varied_batch(arm='B', rows=9, seed=0):
    """Original fixture rows with varied inputs, actions, advantages, masks and weights."""
    rng = np.random.default_rng(seed)
    parts = [fixture.batch(arm, index=i, status='COMPLETE' if i % 3 else 'IN_PROGRESS') for i in range(rows)]
    batch = _combine_training_batches(parts)
    actions = rng.integers(0, 9, rows)
    legal = rng.random((rows, 9)) < .5
    legal[np.arange(rows), actions] = True
    if arm == 'D':
        legal = np.asarray(batch.legal_action_mask, bool).copy()
        actions = np.asarray(batch.action_indices)
    attention = np.ones((rows, 1, 29), bool)
    for i in range(rows):
        attention[i, 0, rng.choice(np.arange(1, 23), size=int(rng.integers(0, 6)), replace=False)] = False
    advantage = rng.normal(size=rows); advantage[::4] = 0.
    weights = rng.uniform(.5, 2., rows); weights[[1, rows - 2]] = 0.
    return replace(batch,
        numeric_features=rng.normal(0, .5, (rows, 1, 29, 132)).astype(np.float32),
        categorical_ids=rng.integers(0, 128, (rows, 1, 29, 41)),
        attention_mask=attention, legal_action_mask=legal, action_indices=actions,
        action_probabilities=rng.uniform(.05, 1., rows), ppo_advantages=advantage,
        ppo_advantage_mask=rng.random(rows) < .8, ppo_value_targets=rng.uniform(-1, 1, rows),
        ppo_value_target_mask=rng.random(rows) < .8, training_weights=weights)


def configurations(arm='B', *, epochs=1, batch_size=4, precision='float32', schedule='constant', dropout=.2):
    model, loss, train = fixture.configurations(arm, epochs=epochs, precision=precision)
    model = replace(model, dropout=dropout)
    return model, loss, replace(train, batch_size=batch_size, learning_rate_schedule=schedule)


def store_of(batch, model, loss, **kwargs):
    return v2.row_store(v5._training_tensors(batch, model, 'cpu'), model, loss, **kwargs)


def max_difference(left, right):
    return max((a.float() - b.float()).abs().max().item() for a, b in zip(left.values(), right.values()))


def invariant(name, width):
    """Elements whose value cannot change any model output (analytically zero gradient).

    The attention key bias adds q.b to every score of a query (softmax is shift
    invariant). The policy head bias adds one constant to all nine logits
    (log-softmax is shift invariant). Rounding noise in their zero gradient is
    amplified by AdamW's normalized step, so any reordering moves them by O(lr).
    """
    if name.endswith('self_attn.in_proj_bias'):
        return slice(width, 2 * width)
    if name == 'policy_head.bias':
        return slice(0, 1)
    return None


def split_differences(left, right, width):
    """Max |difference| over function-relevant elements, and over invariant ones."""
    relevant = invariant_part = 0.
    for name, a in left.items():
        d = (a.float() - right[name].float()).abs()
        part = invariant(name, width)
        if part is not None:
            invariant_part = max(invariant_part, d[part].max().item())
            d = d.clone(); d[part] = 0
        relevant = max(relevant, d.max().item() if d.numel() else 0.)
    return relevant, invariant_part


def eval_outputs(model, tensors):
    model.eval()
    with torch.no_grad():
        output = model(**{name: tensors[name] for name in v2.MODEL_INPUTS})
    model.train()
    legal = tensors['legal_action_mask']
    return output.policy_logits.masked_fill(~legal, -torch.inf).log_softmax(-1).masked_fill(~legal, 0), output.value


@unittest.skipIf(torch is None, 'requires optional local Torch')
class V2TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    # -- equivalence -----------------------------------------------------------
    def test_reference_mode_with_v5_optimizer_is_bitwise_train_caches_including_schedule(self):
        """Original forward_v5/v5_loss + foreach=False AdamW: every bit equals train_caches."""
        for arm, precision in (('A', 'float32'), ('B', 'float32'), ('D', 'float32'), ('B', 'bfloat16')):
            with self.subTest(arm=arm, precision=precision), tempfile.TemporaryDirectory() as directory:
                paths = []
                for i, rows in enumerate((5, 3)):
                    path = Path(directory) / f'cache{i}'
                    v5_data.write_cache(path, varied_batch(arm, rows=rows, seed=i)); paths.append(path)
                model, loss, train = configurations(arm, epochs=3, batch_size=2, precision=precision, schedule='mit-thesis')
                window = v5.DeclaredGameWindow(0, 10, 20)
                original = v5.create(model, loss, train, enable_v5=True)
                expected = v5.train_caches(original, paths, learning_rate_window=window)
                runtime = v2.create(model, loss, train, v2.V2Config(mode='reference', optimizer='single'), enable_v5=True)
                store = v2.load_cache_store(paths, model, loss, device='cpu')
                self.assertEqual(store.segments, (5, 3))
                actual = v2.Trainer(runtime).train(store, window=window)
                self.assertEqual(len(actual['updates']), 15)
                for left, right in zip(expected['updates'], actual['updates']):
                    for name in ('loss', 'policy_loss', 'value_loss', 'entropy', 'owner_rows', 'learning_rate',
                            'learning_rate_epoch', 'optimizer_step'):
                        self.assertEqual(left[name], right[name], name)
                self.assertEqual((original.steps, original.rows), (runtime.steps, runtime.rows))
                fixture.assert_tree_equal(self, original.model.state_dict(), runtime.model.state_dict())
                fixture.assert_tree_equal(self, original.optimizer.state_dict()['state'], runtime.optimizer.state_dict()['state'])
                fixture.assert_tree_equal(self, original.rng, runtime.rng)

    def test_learning_rate_schedule_matches_train_caches_for_every_update(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, rows in enumerate((5, 3)):
                path = Path(directory) / f'cache{i}'
                v5_data.write_cache(path, varied_batch(rows=rows, seed=i)); paths.append(path)
            model, loss, train = configurations(epochs=4, batch_size=2, schedule='mit-thesis')
            store = v2.load_cache_store(paths, model, loss, device='cpu')
            for window in (v5.DeclaredGameWindow(0, 10, 40), v5.DeclaredGameWindow(0, 40, 40), v5.DeclaredGameWindow(10, 30, 40)):
                original = v5.create(model, loss, train, enable_v5=True)
                if window.start_game:
                    v5.train_caches(original, paths, learning_rate_window=v5.DeclaredGameWindow(0, 10, 40))
                    other = [Path(directory) / 'next']
                    if not other[0].exists():
                        v5_data.write_cache(other[0], varied_batch(rows=8, seed=9))
                    expected = v5.train_caches(original, other, learning_rate_window=window)['updates']
                else:
                    expected = v5.train_caches(original, paths, learning_rate_window=window)['updates']
                by_epoch = {u['learning_rate_epoch']: u['learning_rate'] for u in expected}
                for order in ('ordered', 'shuffled'):
                    runtime = v2.create(model, loss, train, v2.V2Config(order=order), enable_v5=True)
                    trainer = v2.Trainer(runtime)
                    if window.start_game:
                        trainer.train(store, window=v5.DeclaredGameWindow(0, 10, 40))
                        updates = trainer.train(v2.load_cache_store(other, model, loss, device='cpu'), window=window)['updates']
                    else:
                        updates = trainer.train(store, window=window)['updates']
                    if order == 'ordered':
                        self.assertEqual([(u['learning_rate'], u['learning_rate_epoch']) for u in updates],
                            [(u['learning_rate'], u['learning_rate_epoch']) for u in expected])
                    self.assertEqual({u['learning_rate_epoch']: u['learning_rate'] for u in updates}, by_epoch)
                    self.assertEqual(by_epoch, {e: v5._epoch_learning_rate(train, window, e) for e in (1, 2, 3, 4)})
                    self.assertEqual(len(set(by_epoch.values())), 4)

    def test_static_eager_update_matches_original_update_loss_and_parameters(self):
        """One step against v5_training._update, from identical state and RNG.

        Loss: the static loss differs only in reduction order (|diff| <= 1e-6 relative).
        Parameters: every function-relevant element agrees to <= 1e-6 (measured
        <= 1.2e-7) with the single-tensor, fused and foreach AdamW kernels alike.
        The function-invariant biases (see ``invariant``) have an analytically
        zero gradient. AdamW's first step is about lr*sign(g), so their rounding
        noise moves them by up to lr either way; the bound is 2*lr. Moving them
        cannot change an output, so the stepped models' log-policies and values
        agree to <= 1e-5.
        """
        for arm, precision in (('B', 'float32'), ('A', 'float32'), ('D', 'float32'), ('B', 'bfloat16')):
            with self.subTest(arm=arm, precision=precision):
                batch = varied_batch(arm, rows=9, seed=3)
                model, loss, train = configurations(arm, batch_size=9, precision=precision)
                original = v5.create(model, loss, train, enable_v5=True)
                expected = v5._update(original, batch)
                results = {}
                for optimizer in ('single', 'fused', 'foreach'):
                    runtime = v2.create(model, loss, train, v2.V2Config(optimizer=optimizer), enable_v5=True)
                    updates = v2.Trainer(runtime).train(store_of(batch, model, loss))['updates']
                    self.assertEqual(len(updates), 1)
                    results[optimizer] = (updates[0], runtime)
                for optimizer, (update, runtime) in results.items():
                    self.assertLessEqual(abs(update['loss'] - expected['loss']), 1e-6 * max(1., abs(expected['loss'])))
                    for name in ('policy_loss', 'value_loss', 'entropy', 'uniform_kl'):
                        self.assertLessEqual(abs(update[name] - expected[name]), 1e-6 * max(1., abs(expected[name])), name)
                        self.assertEqual(update[name + '_count'], expected[name + '_count'])
                    self.assertLessEqual(abs(update['gradient_norm'] - expected['gradient_norm']), 1e-5 * expected['gradient_norm'])
                    self.assertEqual(update['owner_rows'], expected['owner_rows'])
                    relevant, gauge = split_differences(original.model.state_dict(), runtime.model.state_dict(), model.width)
                    self.assertLessEqual(relevant, 1e-6, optimizer)
                    self.assertLessEqual(gauge, 2 * train.learning_rate, optimizer)
                    tensors = v5._training_tensors(batch, model, 'cpu')
                    for left, right in zip(eval_outputs(original.model, tensors), eval_outputs(runtime.model, tensors)):
                        self.assertLessEqual((left - right).abs().max().item(), 1e-5, optimizer)
                fixture.assert_tree_equal(self, original.rng, results['fused'][1].rng)

    def test_static_gradients_match_original_forward_and_loss(self):
        """Same RNG and weights: static vs original gradient, per parameter."""
        from pokezero.v5_loss import v5_loss
        from pokezero.v5_model import forward_v5
        for arm in ('A', 'B', 'D'):
            with self.subTest(arm=arm):
                batch = varied_batch(arm, rows=9, seed=5)
                model, loss, train = configurations(arm, batch_size=9)
                runtime = v2.create(model, loss, train, enable_v5=True)
                tensors = v5._training_tensors(batch, model, 'cpu')
                gradients = []
                for implementation in ('original', 'static'):
                    runtime.model.zero_grad(set_to_none=True)
                    with v5._rng_scope(torch, 'cpu', state=runtime.rng):
                        if implementation == 'original':
                            total, _ = v5_loss(forward_v5(runtime.model, tensors, include_auxiliary=model.treatment),
                                tensors, model, loss)
                        else:
                            total, _, _ = v2._static_forward_loss(runtime.model, tensors, model, loss, 'float32', 'cpu',
                                v2.expert_indices(runtime.model, 'cpu'), v2._variant_counts(model, 'cpu'))
                        total.backward()
                    gradients.append([p.grad.clone() for p in runtime.model.parameters()])
                scale = torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients[0]])).item()
                worst = max((a - b).abs().max().item() for a, b in zip(*gradients))
                self.assertLessEqual(worst, 1e-5 * scale)

    # -- resume ----------------------------------------------------------------
    def test_checkpoint_resume_is_bitwise_uninterrupted_run(self):
        batch = varied_batch(rows=7, seed=11)
        for order, split in (('shuffled', 5), ('ordered', 4), ('shuffled', 1)):
            with self.subTest(order=order, split=split), tempfile.TemporaryDirectory() as directory:
                model, loss, train = configurations(epochs=3, batch_size=2, schedule='mit-thesis')
                config = v2.V2Config(order=order, check_interval=3, optimizer='fused')
                window = v5.DeclaredGameWindow(0, 5, 10)
                store = store_of(batch, model, loss, store_id='fixture')
                full = v2.create(model, loss, train, config, enable_v5=True)
                expected = v2.Trainer(full).train(store, window=window)
                self.assertEqual(len(expected['updates']), 12)
                partial = v2.create(model, loss, train, config, enable_v5=True)
                first = v2.Trainer(partial).train(store, window=window, max_updates=split)
                checkpoint = Path(directory) / 'v2.pt'
                v2.save_checkpoint(checkpoint, partial)
                torch.rand(7); np.random.random(3)
                resumed = v2.load_checkpoint(checkpoint, enable_v5=True, expected_train_config=train)
                self.assertEqual(resumed.cursor, partial.cursor)
                rest = v2.Trainer(resumed).train(store, window=window)
                self.assertEqual(first['updates'] + rest['updates'], expected['updates'])
                self.assertEqual((resumed.steps, resumed.rows, resumed.cursor), (full.steps, full.rows, full.cursor))
                fixture.assert_tree_equal(self, full.model.state_dict(), resumed.model.state_dict())
                fixture.assert_tree_equal(self, full.optimizer.state_dict(), resumed.optimizer.state_dict())
                fixture.assert_tree_equal(self, full.rng, resumed.rng)
                self.assertEqual(v2.Trainer(resumed).train(store, window=window)['updates'], [])
                with self.assertRaises(FileExistsError):
                    v2.save_checkpoint(checkpoint, resumed)

    def test_resume_refuses_different_store_order_or_window(self):
        batch = varied_batch(rows=7, seed=11)
        with tempfile.TemporaryDirectory() as directory:
            model, loss, train = configurations(epochs=2, batch_size=2, schedule='mit-thesis')
            window = v5.DeclaredGameWindow(0, 5, 10)
            store = store_of(batch, model, loss, store_id='first')
            runtime = v2.create(model, loss, train, v2.V2Config(order='shuffled'), enable_v5=True)
            v2.Trainer(runtime).train(store, window=window, max_updates=2)
            path = Path(directory) / 'v2.pt'; v2.save_checkpoint(path, runtime)
            resumed = v2.load_checkpoint(path, enable_v5=True)
            with self.assertRaisesRegex(ValueError, 'original row store'):
                v2.Trainer(resumed).train(store_of(batch, model, loss, store_id='other'), window=window)
            with self.assertRaisesRegex(ValueError, 'original row store'):
                v2.Trainer(resumed).train(store, window=v5.DeclaredGameWindow(0, 6, 10))
            with self.assertRaisesRegex(ValueError, 'original batch order'):
                v2.load_checkpoint(path, enable_v5=True, v2_config=v2.V2Config(order='ordered'))
            with self.assertRaisesRegex(ValueError, 'game zero'):
                v2.Trainer(v2.create(model, loss, train, enable_v5=True)).train(store, window=v5.DeclaredGameWindow(1, 5, 10))
            # A completed cursor accepts the next contiguous window on a new store, never a replay.
            v2.Trainer(resumed).train(store, window=window)
            with self.assertRaisesRegex(ValueError, 'replay'):
                v2.Trainer(resumed).train(store, window=v5.DeclaredGameWindow(5, 10, 10))
            report = v2.Trainer(resumed).train(store_of(batch, model, loss, store_id='next'), window=v5.DeclaredGameWindow(5, 10, 10))
            self.assertEqual(report['updates'][0]['learning_rate'],
                v5._epoch_learning_rate(train, v5.DeclaredGameWindow(5, 10, 10), 1))

    # -- finiteness ------------------------------------------------------------
    def test_nonfinite_parameter_or_loss_raises_with_its_update_window(self):
        batch = varied_batch(rows=8, seed=2)
        model, loss, train = configurations(epochs=4, batch_size=2)
        store = store_of(batch, model, loss)
        runtime = v2.create(model, loss, train, v2.V2Config(check_interval=4), enable_v5=True)
        trainer = v2.Trainer(runtime)
        self.assertEqual(len(trainer.train(store, max_updates=3)['updates']), 3)
        with torch.no_grad():
            runtime.model.policy_head.bias.fill_(float('nan'))
        with self.assertRaises(v2.NonFiniteUpdate) as raised:
            trainer.train(store)
        self.assertEqual((raised.exception.first_step, raised.exception.window), (4, (4, 4)))
        self.assertIn('update window steps 4..4', str(raised.exception))
        with self.assertRaisesRegex(RuntimeError, 'reload the last checkpoint'):
            trainer.train(store)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, 'refusing'):
            v2.save_checkpoint(Path(directory) / 'bad.pt', runtime)
        # A non-finite row slipped in after validation trips the loss flag inside the window.
        poisoned = store_of(batch, model, loss)
        poisoned.tensors['numeric_features'][5, 0, 0, 0] = float('inf')
        runtime = v2.create(model, loss, train, v2.V2Config(check_interval=8), enable_v5=True)
        with self.assertRaises(v2.NonFiniteUpdate) as raised:
            v2.Trainer(runtime).train(poisoned)
        self.assertEqual((raised.exception.first_step, raised.exception.window), (3, (1, 4)))

    def test_store_is_validated_once_with_the_original_messages(self):
        batch = varied_batch(rows=8, seed=2)
        model, loss, train = configurations(epochs=2, batch_size=2)
        tensors = v5._training_tensors(batch, model, 'cpu')
        for name, change, message in (
                ('numeric_features', lambda t: t.__setitem__((0, 0, 0, 0), float('nan')), 'finite original public'),
                ('categorical_ids', lambda t: t.__setitem__((0, 0, 0, 0), 128), 'finite original public'),
                ('training_weights', lambda t: t.__setitem__(0, -1.), 'nonnegative training weights'),
                ('ppo_value_targets', lambda t: t.__setitem__(slice(None), 2.), 'value targets in'),
                ('hidden_set_targets', lambda t: t.__setitem__((2, 0, 0), 5.), 'hidden-set probability'),
                ('action_probabilities', lambda t: t.__setitem__(slice(None), 0.), 'behavior probability')):
            with self.subTest(name=name):
                changed = dict(tensors); changed[name] = tensors[name].clone(); change(changed[name])
                with self.assertRaisesRegex(ValueError, message):
                    v2.row_store(changed, model, loss)
        calls = []
        original = v2.validate_store
        with patch.object(v2, 'validate_store', side_effect=lambda *a: calls.append(1) or original(*a)), \
                patch.object(v5_data, 'validate_batch', side_effect=AssertionError('per-batch validation')):
            store = v2.row_store(tensors, model, loss)
            v2.Trainer(v2.create(model, loss, train, enable_v5=True)).train(store)
        self.assertEqual(len(calls), 1)

    # -- order -----------------------------------------------------------------
    def test_batch_orders_are_deterministic_and_cover_each_row_once(self):
        order = v2.BatchOrder(10, (6, 4), 4, order='ordered', seed=0)
        self.assertEqual([order.bounds(i) for i in range(order.batches_per_epoch)], [(0, 4), (4, 6), (6, 10)])
        shuffled = v2.BatchOrder(10, (10,), 4, order='shuffled', seed=5)
        self.assertEqual([shuffled.batch_rows(i) for i in range(shuffled.batches_per_epoch)], [4, 4, 2])
        for epoch in range(3):
            self.assertEqual(sorted(shuffled.permutation(epoch).tolist()), list(range(10)))
        again = v2.BatchOrder(10, (10,), 4, order='shuffled', seed=5)
        self.assertTrue(torch.equal(shuffled.permutation(1), again.permutation(1)))
        self.assertFalse(torch.equal(again.permutation(0), again.permutation(1)))
        self.assertFalse(torch.equal(again.permutation(0), v2.BatchOrder(10, (10,), 4, order='shuffled', seed=6).permutation(0)))
        ranks = v2.BatchOrder(10, (6, 4), 2, order='ordered', seed=0, world_size=2)
        self.assertEqual((ranks.batches_per_epoch, ranks.steps_per_epoch), (5, 3))
        self.assertEqual([ranks.rank_batch(2, r) for r in (0, 1)], [4, None])
        self.assertEqual([ranks.step_rows(s) for s in range(3)], [4, 4, 2])

    def test_padded_partial_batch_is_the_same_update(self):
        batch = varied_batch(rows=7, seed=4)
        model, loss, train = configurations(batch_size=4, dropout=0.)
        states = []
        for pad in (False, True):
            runtime = v2.create(model, loss, train, v2.V2Config(pad_partial=pad, optimizer='single'), enable_v5=True)
            updates = v2.Trainer(runtime).train(store_of(batch, model, loss))['updates']
            self.assertEqual([u['owner_rows'] for u in updates], [4, 3])
            states.append((updates, runtime.model.state_dict()))
        for left, right in zip(states[0][0], states[1][0]):
            self.assertAlmostEqual(left['loss'], right['loss'], places=6)
            self.assertEqual(left['policy_loss_count'], right['policy_loss_count'])
        relevant, gauge = split_differences(states[0][1], states[1][1], model.width)
        self.assertLessEqual(relevant, 1e-6)
        self.assertLessEqual(gauge, 2 * 2 * train.learning_rate)

    def test_compile_failure_falls_back_to_eager_and_reports_it(self):
        batch = varied_batch(rows=6, seed=4)
        model, loss, train = configurations(epochs=2, batch_size=3)
        store = store_of(batch, model, loss)

        def broken(*args, **kwargs):
            def call(*a, **k):
                raise RuntimeError('synthetic inductor failure')
            return call
        runtime = v2.create(model, loss, train, v2.V2Config(mode='cudagraphs', pad_partial=False), enable_v5=True)
        with patch.object(torch, 'compile', broken):
            trainer = v2.Trainer(runtime)
            report = trainer.train(store)
        self.assertEqual(report['mode']['active'], 'eager')
        self.assertEqual([f['mode'] for f in report['mode']['fallbacks']], ['cudagraphs', 'compiled'])
        eager = v2.create(model, loss, train, enable_v5=True)
        expected = v2.Trainer(eager).train(store)
        self.assertEqual(report['updates'], expected['updates'])
        fixture.assert_tree_equal(self, eager.model.state_dict(), runtime.model.state_dict())

    def test_compiled_update_matches_eager_on_cpu(self):
        """torch.compile of the static forward+loss (dropout off, padded batches).

        float32: losses within 2e-7, function-relevant parameters within 1e-6.
        bfloat16 with emulate_precision_casts: bitwise eager where Inductor's CPU
        backend can compile it (torch 2.10 CPU codegen rejects it; that case
        exercises the fallback instead).
        """
        batch = varied_batch(rows=7, seed=4)
        for precision, emulate in (('float32', False), ('bfloat16', True)):
            with self.subTest(precision=precision):
                model, loss, train = configurations(epochs=2, batch_size=4, dropout=0., precision=precision)
                store = store_of(batch, model, loss)
                torch._dynamo.reset()
                runs = {}
                for mode in ('eager', 'compiled'):
                    config = v2.V2Config(mode=mode, pad_partial=True, emulate_precision_casts=emulate)
                    runtime = v2.create(model, loss, train, config, enable_v5=True)
                    runs[mode] = (v2.Trainer(runtime).train(store), runtime)
                torch._dynamo.reset()
                compiled = runs['compiled'][0]
                if compiled['mode']['active'] != 'compiled':
                    self.assertEqual(compiled['updates'], runs['eager'][0]['updates'])
                    self.skipTest('Inductor CPU compile unavailable: ' + compiled['mode']['fallbacks'][0]['error'][:200])
                pairs = list(zip(runs['eager'][0]['updates'], compiled['updates']))
                if emulate:
                    self.assertEqual([a['loss'] for a, _ in pairs], [b['loss'] for _, b in pairs])
                    fixture.assert_tree_equal(self, runs['eager'][1].model.state_dict(), runs['compiled'][1].model.state_dict())
                else:
                    self.assertLessEqual(max(abs(a['loss'] - b['loss']) for a, b in pairs), 2e-7)
                    relevant, _ = split_differences(runs['eager'][1].model.state_dict(), runs['compiled'][1].model.state_dict(), model.width)
                    self.assertLessEqual(relevant, 1e-6)

    # -- export ----------------------------------------------------------------
    def test_v2_checkpoint_exports_and_loads_as_jax_actor(self):
        try:
            from gpu_engine import v5_model as actor
            import jax  # noqa: F401
        except ImportError as error:
            self.skipTest(f'JAX actor unavailable: {error}')
        batch = varied_batch(rows=6, seed=8)
        model, loss, train = configurations(epochs=1, batch_size=3, schedule='mit-thesis')
        runtime = v2.create(model, loss, train, v2.V2Config(order='shuffled'), enable_v5=True)
        v2.Trainer(runtime).train(store_of(batch, model, loss), window=v5.DeclaredGameWindow(0, 4, 4))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'v2.pt'; archive = Path(directory) / 'actor.npz'
            saved = v2.save_checkpoint(checkpoint, runtime)
            receipt = v2.export_actor(checkpoint, archive)
            params, config, metadata = actor.load(archive, expected_archive_sha256=receipt['weights_archive_sha256'],
                expected_checkpoint_sha256=saved['sha256'], expected_category_vocabulary_sha256=model.category_vocabulary_sha256,
                expected_label_vocabulary_sha256=model.label_vocabulary.sha256, enable_v5=True)
        self.assertEqual(config, model)
        for name, value in params.items():
            np.testing.assert_array_equal(np.asarray(value), runtime.model.state_dict()[name].numpy())
        columns = dict(v5._training_tensors(batch, model, 'cpu'))
        inputs = actor.prepare_inputs({k: (v.numpy() if torch.is_tensor(v) else v) for k, v in columns.items()}, model)
        output = actor.forward(params, inputs, config)
        runtime.model.eval()
        with torch.no_grad():
            expected = runtime.model(**{name: columns[name] for name in v2.MODEL_INPUTS})
        np.testing.assert_allclose(np.asarray(output.policy_logits), expected.policy_logits.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.asarray(output.value), expected.value.numpy(), rtol=1e-4, atol=1e-5)

    # -- DDP -------------------------------------------------------------------
    def test_two_process_gloo_ddp_matches_rank_averaged_reference(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
            mp.spawn(_ddp_worker, args=(2, port, directory), nprocs=2, join=True)
            ranks = [torch.load(Path(directory) / f'rank{r}.pt', weights_only=True) for r in range(2)]
            fixture.assert_tree_equal(self, ranks[0]['state'], ranks[1]['state'])
            reference = _ddp_reference()
            self.assertEqual(ranks[0]['steps'], 3)
            self.assertEqual(ranks[0]['rows'], 10)
            self.assertLessEqual(max_difference(reference, ranks[0]['state']), 1e-6)
            payload = torch.load(Path(directory) / 'ddp.pt', weights_only=True)
            self.assertEqual(len(payload['v2']['rng_by_rank']), 2)


def _ddp_setup():
    model, loss, train = configurations(epochs=1, batch_size=2, dropout=0.)
    return model, loss, train, varied_batch(rows=10, seed=21)


def _ddp_worker(rank, world_size, port, directory):
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    try:
        model, loss, train, batch = _ddp_setup()
        runtime = v2.create(model, loss, train, v2.V2Config(optimizer='single'), enable_v5=True,
            process_group=dist.group.WORLD)
        report = v2.Trainer(runtime).train(store_of(batch, model, loss))
        torch.save(dict(state=runtime.model.state_dict(), steps=report['steps'], rows=report['rows']),
            Path(directory) / f'rank{rank}.pt')
        v2.save_checkpoint(Path(directory) / 'ddp.pt', runtime)
    finally:
        dist.destroy_process_group()


def _ddp_reference():
    """One process: per step, average the gradients of order batches 2s and 2s+1."""
    model, loss, train, batch = _ddp_setup()
    runtime = v2.create(model, loss, train, v2.V2Config(optimizer='single'), enable_v5=True)
    store = store_of(batch, model, loss)
    order = v2.BatchOrder(store.rows, store.segments, train.batch_size, order='ordered', seed=0, world_size=2)
    experts = v2.expert_indices(runtime.model, 'cpu'); counts = v2._variant_counts(model, 'cpu')
    parameters = list(runtime.model.parameters())
    for step in range(order.steps_per_epoch):
        totals = None; count = 0
        for rank in range(2):
            index = order.rank_batch(step, rank)
            if index is None:
                continue
            start, stop = order.bounds(index)
            runtime.model.zero_grad(set_to_none=True)
            total, _, _ = v2._static_forward_loss(runtime.model, {k: v[start:stop] for k, v in store.tensors.items()},
                model, loss, 'float32', 'cpu', experts, counts)
            total.backward()
            grads = [p.grad.clone() for p in parameters]
            totals = grads if totals is None else [a + b for a, b in zip(totals, grads)]
            count += 1
        for p, g in zip(parameters, totals):
            p.grad = g / count
        torch.nn.utils.clip_grad_norm_(parameters, train.max_grad_norm)
        runtime.optimizer.step()
    return runtime.model.state_dict()


if __name__ == '__main__':
    unittest.main()
