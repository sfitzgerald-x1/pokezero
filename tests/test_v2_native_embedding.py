"""Native embedding semantics, opaque AOT backward, execution settings/resume.

These are CPU checks. They do not qualify CUDA capture, eager-compatible GPU
dropout, or an operator-to-GPU-kernel attribution.
"""
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
    from pokezero import v2_trainer as v2, v5_training as v5
    from pokezero.v2_native_embedding import _Embedding, embedding_dense_gradient, lookup, validate


@unittest.skipIf(torch is None, 'requires optional Torch/numpy runtime')
class NativeEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def equal_tree(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.equal_tree(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal_tree(a, b)
        else:
            self.assertEqual(left, right)

    def test_forward_and_native_weight_gradient_exact_with_repeats_padding_and_layouts(self):
        for shape in ((4, 29, 41), (2, 3), (1,)):
            for padding_only in (False, True):
                with self.subTest(shape=shape, padding_only=padding_only):
                    embedding = torch.nn.Embedding(32, 8, padding_idx=0)
                    validate(embedding)
                    indices = torch.zeros(shape, dtype=torch.long)
                    if not padding_only:
                        indices.reshape(-1)[::11] = 3
                        indices.reshape(-1)[::17] = 7
                    # Expanded upstream gradients exercise noncontiguous/zero strides.
                    gradient = torch.randn((*shape, 1)).expand(*shape, 8)
                    results = []
                    for native in (False, True):
                        embedding.zero_grad(set_to_none=True)
                        output = lookup(embedding, indices) if native else embedding(indices)
                        output.backward(gradient)
                        results.append((output.detach().clone(), embedding.weight.grad.clone()))
                    self.equal_tree(*results)
                    self.assertTrue(torch.equal(results[1][1][0], torch.zeros(8)))

    def test_attributes_refuse_unsupported_semantics(self):
        for module in (torch.nn.Embedding(8, 4), torch.nn.Embedding(8, 4, padding_idx=0, sparse=True),
                torch.nn.Embedding(8, 4, padding_idx=0, scale_grad_by_freq=True),
                torch.nn.Embedding(8, 4, padding_idx=0).to(dtype=torch.float64),
                torch.nn.Embedding(8, 4, padding_idx=0, max_norm=1.)):
            with self.assertRaises(ValueError):
                validate(module)

    def test_custom_op_schema_fake_and_aot_checks(self):
        indices = torch.tensor([[0, 1, 1], [3, 0, 2]])
        grad = torch.randn(2, 3, 8)
        # This backward-only operator is tested without higher-order autograd.
        result = torch.library.opcheck(embedding_dense_gradient, (grad, indices, 16, 0))
        self.assertTrue(all(value == 'SUCCESS' for value in result.values()), result)

    def test_aot_backward_preserves_opaque_native_op_under_inductor_decompositions(self):
        from functorch.compile import make_boxed_func
        from torch._functorch.aot_autograd import aot_function
        from torch._inductor.decomposition import select_decomp_table
        graphs = {}
        def compiler(kind):
            def compile_graph(graph, inputs):
                graphs[kind] = [str(node.target) for node in graph.graph.nodes if node.op == 'call_function']
                return make_boxed_func(graph.forward)
            return compile_graph
        def function(weight, indices):
            return _Embedding.apply(weight, indices, 0).sum(-2).square().sum()
        indices = torch.zeros(2, 29, 41, dtype=torch.long)
        indices[:, :, ::13] = 2
        weight = torch.randn(16, 8, requires_grad=True)
        reference = weight.detach().clone().requires_grad_()
        expected = function(reference, indices)
        expected.backward()
        with torch._inductor.config.patch(fallback_random=True):
            compiled = aot_function(function, fw_compiler=compiler('forward'), bw_compiler=compiler('backward'),
                decompositions=select_decomp_table())
            actual = compiled(weight, indices)
            actual.backward()
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(reference.grad, weight.grad))
        self.assertTrue(any('v2_runtime.embedding_dense_gradient' in target for target in graphs['backward']), graphs)
        self.assertFalse(any('index_put' in target for target in graphs['backward']), graphs)

    def test_full_forward_loss_gradient_rng_and_state_names_unchanged(self):
        for precision in ('float32', 'bfloat16'):
            with self.subTest(precision=precision):
                model, loss, train = fixture.configurations(precision=precision, dropout=.2)
                runtime = v2.create(model, loss, train, enable_v5=True, device='cpu')
                tensors = v5._training_tensors(fixture.varied_batch(rows=9, seed=12), model, 'cpu')
                names = tuple(runtime.model.state_dict())
                results = []
                for native in (False, True):
                    runtime.model.zero_grad(set_to_none=True)
                    with v5._rng_scope(torch, 'cpu', state=runtime.rng):
                        total, vector, ok = v2._static_forward_loss(runtime.model, tensors, model, loss, precision,
                            'cpu', v2.expert_indices(runtime.model, 'cpu'), v2._variant_counts(model, 'cpu'), 'indexed', native)
                        total.backward()
                        results.append((total.detach().clone(), vector.clone(), ok.clone(),
                            {key:p.grad.clone() for key,p in runtime.model.named_parameters()}, v5._rng_state(torch, False)))
                self.equal_tree(*results)
                self.assertEqual(names, tuple(runtime.model.state_dict()))

    def test_flags_default_off_and_compiler_options_are_explicit(self):
        config = v2.V2Config(**dict(role_routing='indexed', mode='eager'))
        self.assertFalse(config.native_embedding_backward)
        self.assertFalse(config.fallback_random)
        for kwargs in (dict(mode='reference', native_embedding_backward=True), dict(fallback_random=True),
                dict(native_embedding_backward=1), dict(mode='compiled', fallback_random=1)):
            with self.assertRaises(ValueError):
                v2.V2Config(**kwargs)
        model, loss, train = fixture.configurations()
        with patch.object(torch, 'compile') as compiler:
            runtime = v2.create(model, loss, train, v2.V2Config(mode='compiled', native_embedding_backward=True,
                fallback_random=True), enable_v5=True, device='cpu')
            v2.Trainer(runtime)
        self.assertTrue(compiler.call_args.kwargs['options']['fallback_random'])

    def test_same_process_native_eager_resume_and_flag_switch_refusal(self):
        model, loss, train = fixture.configurations(epochs=3, batch_size=4, dropout=.2)
        config = v2.V2Config(mode='eager', optimizer='single', pad_partial=True, native_embedding_backward=True)
        runtime = v2.create(model, loss, train, config, enable_v5=True, device='cpu')
        store = fixture.store_of(fixture.varied_batch(rows=9, seed=92), model, loss,
            segments=(5, 4), store_id='native-fixed')
        trainer = v2.Trainer(runtime)
        trainer.train(store, max_updates=2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'prefix.pt'
            v2.save_checkpoint(checkpoint, runtime)
            suffix = trainer.train(store, max_updates=4)
            state = copy.deepcopy((runtime.model.state_dict(), v2._canonical_optimizer_state(runtime.optimizer),
                runtime.rng, runtime.cursor, runtime.steps, runtime.rows))
            resumed = v2.load_checkpoint(checkpoint, enable_v5=True, device='cpu')
            again = v2.Trainer(resumed).train(store, max_updates=4)
            self.assertEqual(suffix['updates'], again['updates'])
            self.equal_tree(state, (resumed.model.state_dict(), v2._canonical_optimizer_state(resumed.optimizer),
                resumed.rng, resumed.cursor, resumed.steps, resumed.rows))
            for changed in (replace(config, native_embedding_backward=False), replace(config, mode='compiled', fallback_random=True)):
                with self.assertRaisesRegex(ValueError, 'original embedding/RNG'):
                    v2.load_checkpoint(checkpoint, enable_v5=True, device='cpu', v2_config=changed)

    def test_legacy_checkpoint_without_both_new_flags_loads_off(self):
        model, loss, train = fixture.configurations()
        runtime = v2.create(model, loss, train, enable_v5=True, device='cpu')
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'saved.pt'
            legacy = Path(directory)/'legacy.pt'
            v2.save_checkpoint(checkpoint, runtime)
            data = torch.load(checkpoint, weights_only=True)
            for name in ('native_embedding_backward', 'fallback_random'):
                data['v2']['config'].pop(name)
            data['v2_sha256'] = v2._v2_digest(data['v2'])
            torch.save(data, legacy)
            restored = v2.load_checkpoint(legacy, enable_v5=True, device='cpu')
            self.assertFalse(restored.v2_config.native_embedding_backward)
            self.assertFalse(restored.v2_config.fallback_random)


if __name__ == '__main__':
    unittest.main()
