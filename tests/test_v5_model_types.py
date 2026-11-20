"""Stable compiler-facing types with fresh model state and initialization."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from pokezero import v5_model as models
import test_v5_model_loss as fixture

try:
    import torch
except ImportError:
    torch = None


def family(model):
    block = model.encoder.layers[0]
    return dict(Model=type(model), Encoder=type(model.encoder), Block=type(block),
                RMSNorm=type(model.final_norm), RoleExperts=type(block.role_experts))


@unittest.skipIf(torch is None, 'Torch unavailable')
class ModelTypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_equal_configs_share_all_types_and_keep_supplied_config(self):
        config = fixture.config()
        equal = models.V5ModelConfig.from_metadata(json.loads(json.dumps(config.metadata())))
        self.assertIsNot(config, equal)
        a = models.build_v5_model(config, enable_v5=True)
        b = models.build_v5_model(equal, enable_v5=True)
        self.assertEqual(family(a), family(b))
        self.assertIs(a.config, config)
        self.assertIs(b.config, equal)
        self.assertEqual(a.config.metadata(), b.config.metadata())

    def test_instances_never_share_parameters_or_buffers(self):
        config = fixture.config()
        a = models.build_v5_model(config, enable_v5=True)
        b = models.build_v5_model(config, enable_v5=True)
        state = {name: value.clone() for name, value in b.state_dict().items()}
        for left, right in ((dict(a.named_parameters()), dict(b.named_parameters())),
                            (dict(a.named_buffers()), dict(b.named_buffers()))):
            self.assertEqual(left.keys(), right.keys())
            for name in left: self.assertNotEqual(left[name].data_ptr(), right[name].data_ptr(), name)
        with torch.no_grad():
            for value in a.parameters(): value.add_(1)
            for value in a.buffers(): value.zero_()
        for name, value in b.state_dict().items(): self.assertTrue(torch.equal(value, state[name]), name)

    def test_complete_configuration_is_the_key(self):
        config = fixture.config()
        base = models.build_v5_model(config, enable_v5=True)
        variants = [replace(config, arm='A'), replace(config, replicate=1), replace(config, dropout=.1),
                    replace(config, width=32), replace(config, category_vocabulary_sha256='a'*64),
                    replace(config, label_vocabulary=replace(config.label_vocabulary, source_binding_sha256='c'*64))]
        for changed in variants:
            with self.subTest(changed=changed):
                model = models.build_v5_model(changed, enable_v5=True)
                for name, custom_type in family(model).items(): self.assertIsNot(custom_type, family(base)[name])
                self.assertIs(model.config, changed)
                self.assertEqual(model.category_embedding.weight.shape[1], changed.width)

    def test_cache_miss_and_hit_keep_initialization_and_rng_identical(self):
        config = replace(fixture.config(), norm_epsilon=1.23456789e-6)
        torch.manual_seed(513)
        miss = models.build_v5_model(config, enable_v5=True)
        after_miss = torch.get_rng_state().clone()
        torch.manual_seed(513)
        hit = models.build_v5_model(replace(config), enable_v5=True)
        self.assertTrue(torch.equal(after_miss, torch.get_rng_state()))
        self.assertEqual(list(miss.state_dict()), list(hit.state_dict()))
        for name, value in miss.state_dict().items(): self.assertTrue(torch.equal(value, hit.state_dict()[name]), name)

    def test_opt_in_validation_precedes_factory_and_torch(self):
        with patch.object(models, '_make_v5_model_type') as factory, patch('pokezero.neural_policy.require_torch') as require:
            with self.assertRaises(ValueError): models.build_v5_model(fixture.config())
            with self.assertRaises(ValueError): models.build_v5_model(object(), enable_v5=True)
            factory.assert_not_called(); require.assert_not_called()

    def test_concurrent_first_lookup_creates_one_class_family(self):
        config = replace(fixture.config(), norm_epsilon=2.34567891e-6)
        factory = models._make_v5_model_type
        with patch.object(models, '_make_v5_model_type', wraps=factory) as counted:
            with ThreadPoolExecutor(max_workers=4) as pool:
                instances = list(pool.map(lambda _: models.build_v5_model(replace(config), enable_v5=True), range(8)))
        self.assertEqual(counted.call_count, 1)
        self.assertTrue(all(family(model) == family(instances[0]) for model in instances))
        self.assertEqual(len({model.category_embedding.weight.data_ptr() for model in instances}), len(instances))

    def test_strict_state_and_optimizer_parameter_order_compatible(self):
        config = fixture.config()
        a = models.build_v5_model(config, enable_v5=True)
        b = models.build_v5_model(config, enable_v5=True)
        self.assertEqual(list(dict(a.named_parameters())), list(dict(b.named_parameters())))
        optimizer = torch.optim.AdamW(a.parameters(), lr=.01, foreach=False)
        for value in a.parameters(): value.grad = torch.ones_like(value)
        optimizer.step(); b.load_state_dict(a.state_dict(), strict=True)
        restored = torch.optim.AdamW(b.parameters(), lr=.01, foreach=False)
        restored.load_state_dict(optimizer.state_dict())
        for (_, left), (_, right) in zip(a.named_parameters(), b.named_parameters()):
            self.assertTrue(torch.equal(left, right))
            self.assertTrue(torch.equal(optimizer.state[left]['exp_avg'], restored.state[right]['exp_avg']))
            self.assertTrue(torch.equal(optimizer.state[left]['exp_avg_sq'], restored.state[right]['exp_avg_sq']))

    def test_fresh_instance_reuses_compiler_type_guard(self):
        torch._dynamo.reset()
        config = fixture.config()
        a = models.build_v5_model(config, enable_v5=True)
        b = models.build_v5_model(replace(config), enable_v5=True)
        compiled_graphs = []
        def backend(graph, inputs):
            compiled_graphs.append(graph)
            return graph.forward
        def probe(model, inputs): return model.policy_head(model.final_norm(inputs))
        compiled = torch.compile(probe, backend=backend, fullgraph=True)
        inputs = torch.ones(3, config.width)
        torch.testing.assert_close(compiled(a, inputs), probe(a, inputs))
        torch.testing.assert_close(compiled(b, inputs), probe(b, inputs))
        self.assertEqual(len(compiled_graphs), 1, 'same-config instance changed the compiler specialization')


if __name__ == '__main__': unittest.main()
