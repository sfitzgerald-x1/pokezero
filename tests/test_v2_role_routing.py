"""Opt-in fixed routing: math/RNG, layout refusal, tail resume and old metadata."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    import numpy
except ImportError:
    torch = None

if torch is not None:
    import test_v2_trainer as fixture
    from pokezero import checkpoint_self_moves as encoding, v2_trainer as v2, v5_training as v5


@unittest.skipIf(torch is None, 'requires optional Torch/numpy runtime')
class RoleRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def tree_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.tree_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.tree_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def example(self, *, routing='sliced', layers=1):
        model, loss, train = fixture.configurations(epochs=3, batch_size=4, dropout=.2)
        model = replace(model, layers=layers)
        config = v2.V2Config(mode='eager', optimizer='single', pad_partial=True,
            check_interval=3, role_routing=routing)
        runtime = v2.create(model, loss, train, config, enable_v5=True, device='cpu')
        store = fixture.store_of(fixture.varied_batch(rows=9, seed=81), model, loss,
            segments=(5, 4), store_id='routing-fixed')
        return runtime, store

    def test_missing_old_option_defaults_to_original_and_invalid_modes_refuse(self):
        self.assertEqual(v2.V2Config(**dict(mode='eager')).role_routing, 'indexed')
        for kwargs in (dict(role_routing='unknown'), dict(role_routing='sliced', mode='reference')):
            with self.assertRaises(ValueError):
                v2.V2Config(**kwargs)

    def test_every_layer_role_mapping_is_checked(self):
        runtime, _ = self.example(layers=2)
        runtime.model.encoder.layers[1].role_experts.roles[0] = 1
        with self.assertRaisesRegex(ValueError, 'every layer'):
            v2.Trainer(runtime)

    def test_wrong_token_count_refuses_sliced_routing(self):
        runtime, _ = self.example()
        with self.assertRaisesRegex(ValueError, '29 tokens'):
            v2._sliced_expert_output(runtime.model.encoder.layers[0], torch.zeros(2, 28, runtime.model_config.width))

    def test_isolated_route_forward_and_input_parameter_gradients_are_exact(self):
        runtime, _ = self.example()
        block = runtime.model.encoder.layers[0]
        hidden = torch.randn(3, 29, runtime.model_config.width)
        upstream = torch.randn_like(hidden)
        results = []
        for route in ('indexed', 'sliced'):
            runtime.model.zero_grad(set_to_none=True)
            value = hidden.clone().requires_grad_()
            output = block.role_experts(value) if route == 'indexed' else v2._sliced_expert_output(block, value)
            output.backward(upstream)
            results.append((output.detach().clone(), value.grad.clone(),
                {name:p.grad.clone() for name,p in block.role_experts.named_parameters()}))
        self.tree_equal(*results)

    def test_static_forward_loss_gradients_and_dropout_rng_are_exact(self):
        for arm in ('A', 'B', 'D'):
            for precision in ('float32', 'bfloat16'):
                with self.subTest(arm=arm, precision=precision):
                    model, loss, train = fixture.configurations(arm, precision=precision, dropout=.2)
                    runtime = v2.create(model, loss, train, enable_v5=True, device='cpu')
                    tensors = v5._training_tensors(fixture.varied_batch(arm, rows=9, seed=12), model, 'cpu')
                    experts = v2.expert_indices(runtime.model, 'cpu')
                    counts = v2._variant_counts(model, 'cpu')
                    results = []
                    for route in ('indexed', 'sliced'):
                        runtime.model.zero_grad(set_to_none=True)
                        with v5._rng_scope(torch, 'cpu', state=runtime.rng):
                            total, vector, ok = v2._static_forward_loss(runtime.model, tensors, model, loss,
                                precision, 'cpu', experts, counts, route)
                            total.backward()
                            results.append((total.detach().clone(), vector.clone(), ok.clone(),
                                {name:p.grad.clone() for name,p in runtime.model.named_parameters()},
                                v5._rng_state(torch, False)))
                    self.tree_equal(*results)

    def test_same_process_resume_is_exact_across_padding_and_epoch_boundary(self):
        runtime, store = self.example()
        trainer = v2.Trainer(runtime)
        prefix = trainer.train(store, max_updates=2)
        self.assertEqual(prefix['cursor']['step'], 2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'prefix.pt'
            receipt = v2.save_checkpoint(checkpoint, runtime)
            suffix = trainer.train(store, max_updates=4)
            expected = copy.deepcopy((runtime.model.state_dict(), v2._canonical_optimizer_state(runtime.optimizer),
                runtime.rng, runtime.cursor, runtime.steps, runtime.rows))
            resumed = v2.load_checkpoint(checkpoint, enable_v5=True, device='cpu', expected_self_moves=False,
                expected_checkpoint_sha256=receipt['sha256'],
                off_provenance=encoding.off_from_receipt(receipt['encoding_receipt']))
            actual = v2.Trainer(resumed).train(store, max_updates=4)
            self.assertEqual(suffix['updates'], actual['updates'])
            self.tree_equal(expected, (resumed.model.state_dict(), v2._canonical_optimizer_state(resumed.optimizer),
                resumed.rng, resumed.cursor, resumed.steps, resumed.rows))

    def test_legacy_serialized_option_absence_loads_as_indexed(self):
        runtime, store = self.example(routing='indexed')
        v2.Trainer(runtime).train(store, max_updates=2)
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory)/'legacy.pt'
            original_section = v2._v2_section

            def legacy_section(runtime, rng_by_rank):
                section = original_section(runtime, rng_by_rank)
                section['config'].pop('role_routing')
                return section

            with patch.object(v2, '_v2_section', side_effect=legacy_section):
                receipt = v2.save_checkpoint(legacy, runtime)
            resumed = v2.load_checkpoint(legacy, enable_v5=True, device='cpu', expected_self_moves=False,
                expected_checkpoint_sha256=receipt['sha256'],
                off_provenance=encoding.off_from_receipt(receipt['encoding_receipt']))
            self.assertEqual(resumed.v2_config.role_routing, 'indexed')
            self.assertEqual(resumed.cursor, runtime.cursor)

    def test_checkpoint_routing_switch_is_rejected(self):
        runtime, store = self.example(routing='indexed')
        v2.Trainer(runtime).train(store, max_updates=2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'saved.pt'
            receipt = v2.save_checkpoint(checkpoint, runtime)
            binding = dict(expected_self_moves=False, expected_checkpoint_sha256=receipt['sha256'],
                           off_provenance=encoding.off_from_receipt(receipt['encoding_receipt']))
            with self.assertRaisesRegex(ValueError, 'original role routing'):
                v2.load_checkpoint(checkpoint, enable_v5=True, device='cpu',
                    v2_config=replace(runtime.v2_config, role_routing='sliced'), **binding)


if __name__ == '__main__':
    unittest.main()
