"""Tiny untrained CPU models and hand loss/gradient controls; no scientific run."""
from dataclasses import replace
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from pokezero import v5_contract as c
from pokezero.v5_model import V5ModelConfig, V5Output, build_v5_model, forward_v5, EXPERT_ROLES
from pokezero.v5_loss import V5LossConfig, gae_targets, v5_batch_to_torch, v5_loss
from tests.test_v5_contract import observation, improvement, vocabulary


def config(arm='B'):
    return V5ModelConfig(128, 'b' * 64, vocabulary(), arm=arm, width=16,
        layers=1, attention_heads=2, expert_width=12, value_hidden=(16, 16))


def batch_columns(arm='B'):
    first = observation()
    rows = tuple(replace(first, address=replace(first.address, frame=first.address.frame + i)) for i in range(3))
    labels = [c.V5Labels(rows[0].address, terminal_status='COMPLETE', outcome=1,
        opponent_next_action=c.OpponentAction('MOVE', 'surf'),
        hidden_sets=(c.HiddenSet('blissey', (0, 2)),) + (None,) * 5,
        pokemon_end=(c.PokemonEnd(False, 0), c.PokemonEnd(True, 0.4)) + (None,) * 10),
        c.V5Labels(rows[1].address, terminal_status='CAPPED'), c.V5Labels(rows[2].address)]
    if arm == 'D':
        labels[0] = replace(labels[0], root_improvement=improvement(), search_status='LABELED')
        labels[1] = replace(labels[1], root_improvement=improvement(), search_status='LABELED')
    batch = c.batch_columns(rows, labels)
    batch.update(action_indices=(0, 1, -1), action_probabilities=(0.6, 0.4, math.nan),
        action_probability_mask=(True, True, False), ppo_advantages=(0.5, -0.3, math.nan),
        ppo_advantage_mask=(True, True, False), ppo_value_targets=(0.8, -0.6, math.nan),
        ppo_value_target_mask=(True, True, False), training_weights=(1.0, 2.0, 0.0),
        value_estimates=(0.1, -0.2, math.nan), value_estimate_mask=(True, True, False),
        outcome_targets=(1.0, 0.0, 0.0), outcome_target_mask=(True, False, False),
        terminal_status=('COMPLETE', 'CAPPED', 'IN_PROGRESS'))
    return batch


def outputs(cfg):
    logits = torch.tensor([[0.2, -0.1] + [2.0] * 7, [0.4, 0.6] + [3.0] * 7,
        [float('nan')] * 9], requires_grad=True)
    value = torch.tensor([0.3, -0.4, float('nan')], requires_grad=True)
    if not cfg.treatment:
        return V5Output(logits, value)
    distribution = torch.stack((torch.arange(101) * 0.03, torch.arange(101).flip(0) * 0.02,
        torch.full((101,), float('nan')))).requires_grad_()
    semantic = torch.full((3, 5), float('nan')); semantic[0] = 0
    species = torch.full((3, 6, 2), float('nan')); species[0, 0] = 0
    variants = torch.full((3, 6, 64), float('nan')); variants[0, 0, :3] = 0
    survival = torch.full((3, 12), float('nan')); survival[0, :2] = 0
    hp = torch.full((3, 12), float('nan')); hp[0, :2] = 0.5
    return V5Output(logits, value, distribution, semantic.requires_grad_(), species.requires_grad_(),
        variants.requires_grad_(), survival.requires_grad_(), hp.requires_grad_())


def isolated_loss_config(arm='B', **options):
    return V5LossConfig(arm=arm, value_weight=0, entropy_coefficient=0, uniform_kl_coefficient=0,
        auxiliary_weight=0, search_policy_weight=0, normalize_advantage=False, **options)


@unittest.skipIf(torch is None, 'requires optional local Torch runtime')
class V5ModelLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_opt_in_config_metadata_and_initial_arm_constraints(self):
        cfg = config('D')
        with self.assertRaises(ValueError):build_v5_model(cfg)
        self.assertEqual(V5ModelConfig.from_metadata(json.loads(json.dumps(cfg.metadata()))), cfg)
        for changed in (dict(privileged_critic=True), dict(actor='SEARCH'), dict(search_mode='ACT_WITH_SEARCH')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                V5ModelConfig.from_metadata(dict(cfg.metadata(), **changed))
        with self.assertRaises(ValueError):replace(cfg, arm='C')
        with self.assertRaises(ValueError):replace(cfg, arm='D', replicate=1)
        with self.assertRaises(ValueError):V5LossConfig(arm='B', value_clip=0.1)
        with self.assertRaises(ValueError):V5LossConfig(arm='B', value_lambda=0.94)
        with self.assertRaises(ValueError):V5LossConfig(arm='B', historical_opponent_pool=False)
        self.assertTrue(V5LossConfig(arm='D').metadata()['historical_opponent_pool'])
        self.assertEqual(V5ModelConfig(128, 'b' * 64, vocabulary()).width, 512)
        self.assertEqual(V5ModelConfig(128, 'b' * 64, vocabulary()).layers, 3)

    def test_tiny_A_B_D_forward_value_semantics_and_finite_model_gradients(self):
        for arm in ('A', 'B', 'D'):
            with self.subTest(arm=arm):
                torch.manual_seed(7)
                cfg = config(arm); model = build_v5_model(cfg, enable_v5=True)
                tensors = v5_batch_to_torch(batch_columns(arm), cfg, training=True)
                out = forward_v5(model, tensors, include_auxiliary=cfg.treatment)
                self.assertEqual(tuple(out.policy_logits.shape), (3, 9))
                self.assertEqual(tuple(out.value.shape), (3,))
                self.assertTrue(torch.isfinite(out.value).all())
                self.assertTrue((out.value.abs() <= 1).all())
                if cfg.treatment:
                    expected = (out.value_logits.float().softmax(-1) * model.value_centers).sum(-1)
                    torch.testing.assert_close(out.value, expected)
                    self.assertEqual(tuple(out.hidden_variant_logits.shape), (3, 6, 64))
                    self.assertEqual(tuple(out.survival_logits.shape), (3, 12))
                else:
                    self.assertIsNone(out.value_logits)
                    self.assertIsNone(out.opponent_semantic_logits)
                loss, metrics = v5_loss(out, tensors, cfg, V5LossConfig(arm=arm))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertEqual(metrics['policy_loss_count'], 2)

    def test_actor_never_evaluates_auxiliary_heads_or_reads_truth_sidecars(self):
        cfg = config(); model = build_v5_model(cfg, enable_v5=True).eval()
        columns = batch_columns(); columns['privileged_numeric_features'] = 'private poison'
        tensors = v5_batch_to_torch(columns, cfg)
        self.assertNotIn('privileged_numeric_features', tensors)
        with patch.object(model.opponent_semantic_head, 'forward', side_effect=AssertionError('auxiliary inference')):
            first = forward_v5(model, tensors)
            tensors['hidden_set_species'] = 'different training truth'
            tensors['privileged_numeric_features'] = float('nan')
            second = forward_v5(model, tensors)
        self.assertTrue(torch.equal(first.policy_logits, second.policy_logits))
        self.assertTrue(torch.equal(first.value, second.value))
        self.assertIsNone(first.hidden_species_logits)
        with self.assertRaises(TypeError):model(**{name: tensors[name] for name in
            ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask', 'history_mask')}, privileged_view=True)

    def test_every_declared_belief_token_must_remain_attended(self):
        cfg = config(); model = build_v5_model(cfg, enable_v5=True)
        original = v5_batch_to_torch(batch_columns(), cfg)
        for slot in range(6):
            tensors = dict(original); tensors['attention_mask'] = original['attention_mask'].clone()
            tensors['attention_mask'][1, 0, 23 + slot] = False
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                forward_v5(model, tensors)

    def test_role_experts_route_pokemon_actions_and_field_belief_without_gating(self):
        cfg = config(); model = build_v5_model(cfg, enable_v5=True)
        experts = model.encoder.layers[0].role_experts
        with torch.no_grad():
            for i, expert in enumerate(experts.experts):
                for parameter in expert.parameters():parameter.zero_()
                expert[2].bias.fill_(i + 1)
        result = experts(torch.randn(2, 29, cfg.width))
        expected = torch.tensor([role + 1 for role in EXPERT_ROLES], dtype=torch.float32)
        self.assertTrue(torch.equal(result[0, :, 0], expected))
        self.assertTrue(torch.equal(result[1, :, -1], expected))

    def test_normal_shape_warm_start_retains_v4_attention_parameter_names(self):
        from pokezero.neural_policy import EntityTokenTransformerPolicy, TransformerPolicyConfig
        old = EntityTokenTransformerPolicy(TransformerPolicyConfig.compact_category(
            category_vocab=('species:unit',), category_oov_buckets=126,
            embedding_dim=16, transformer_layers=1, attention_heads=2, feedforward_dim=64,
            observation_schema_version=c.OBSERVATION_SCHEMA_VERSION_V4, transition_token_budget=0, dropout=0))
        new = build_v5_model(config(), enable_v5=True)
        plan = c.warm_start_shape_plan({k: tuple(v.shape) for k, v in old.state_dict().items()},
            {k: tuple(v.shape) for k, v in new.state_dict().items()},
            source_category_vocabulary_sha256='b' * 64, target_category_vocabulary_sha256='b' * 64)
        self.assertIn('encoder.layers.0.self_attn.in_proj_weight', plan['copy'])
        self.assertIn('encoder.layers.0.norm1.weight', plan['copy'])
        self.assertIn('encoder.layers.0.role_experts.experts.0.0.weight', plan['initialize'])
        self.assertIn('encoder.layers.0.norm1.bias', plan['unused_source'])

    def test_tensor_adapter_reads_existing_PPO_fields_without_mutating_them(self):
        columns = batch_columns()
        # The actual data adapter retains inputs/PPO/search on TrainingBatch;
        # v5_columns contains only appended truth and namespace metadata.
        base_fields = ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask',
            'history_mask', 'legal_action_mask', 'action_indices', 'action_probabilities',
            'action_probability_mask', 'ppo_advantages', 'ppo_advantage_mask', 'ppo_value_targets',
            'ppo_value_target_mask', 'training_weights', 'search_policy_targets',
            'search_policy_target_mask', 'search_value_targets', 'search_value_target_mask')
        base = SimpleNamespace(v5_columns={k: v for k, v in columns.items() if k not in base_fields},
            **{k: columns[k] for k in base_fields})
        tensors = v5_batch_to_torch(base, config(), training=True)
        self.assertEqual(tensors['action_indices'].tolist(), [0, 1, -1])
        self.assertEqual(base.action_probabilities, columns['action_probabilities'])
        with self.assertRaises(ValueError):v5_batch_to_torch(dict(columns, label_vocabulary_sha256='c' * 64), config())
        with self.assertRaises(ValueError):v5_batch_to_torch(dict(columns, legal_action_mask=((1,) * 9,) * 3), config())
        with self.assertRaises(ValueError):v5_batch_to_torch(SimpleNamespace(v5_columns=None), config())

    def test_PPO_loss_and_gradients_match_literal_reference_and_ignore_masked_poison(self):
        cfg = config(); out = outputs(cfg); tensors = v5_batch_to_torch(batch_columns(), cfg, training=True)
        actual, _ = v5_loss(out, tensors, cfg, isolated_loss_config())
        lp = out.policy_logits[:2, :2].log_softmax(-1)
        ratio = torch.stack((lp[0, 0].exp() / 0.6, lp[1, 1].exp() / 0.4))
        advantage = torch.tensor([0.5, -0.3]); weights = torch.tensor([1.0, 2.0])
        expected = -(torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage) * weights).sum() / 3
        torch.testing.assert_close(actual, expected)
        a = torch.autograd.grad(actual, out.policy_logits, retain_graph=True)[0]
        b = torch.autograd.grad(expected, out.policy_logits)[0]
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual(a[2].abs().sum().item(), 0)

    def test_HL_Gauss_gradient_uses_D_target_exactly_once_and_preserves_advantages(self):
        cfg = config('D'); out = outputs(cfg); tensors = v5_batch_to_torch(batch_columns('D'), cfg, training=True)
        tensors['ppo_advantages'].zero_()
        loss_cfg = replace(isolated_loss_config('D'), value_weight=1)
        actual, _ = v5_loss(out, tensors, cfg, loss_cfg)
        # Literal targets: (outcome1+root.65)/2=.825; capped row keeps its own
        # valid GAE target-.6, never the absent literal outcome placeholder0.
        targets = torch.tensor([0.825, -0.6], dtype=torch.float64)
        edges = torch.linspace(-1, 1, 102, dtype=torch.float64)
        cdf = (1 + torch.erf((edges[None] - targets[:, None]) / (0.75 * 2 / 101 * math.sqrt(2)))) / 2
        masses = torch.diff(cdf, dim=1); labels = (masses / masses.sum(-1, keepdim=True)).float()
        per_row = -(labels * out.value_logits[:2].log_softmax(-1)).sum(-1)
        expected = (per_row * torch.tensor([1.0, 2.0])).sum() / 3
        torch.testing.assert_close(actual, expected)
        a = torch.autograd.grad(actual, out.value_logits, retain_graph=True)[0]
        b = torch.autograd.grad(expected, out.value_logits)[0]
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.isfinite(a).all())

    def test_reverse_uniform_KL_keeps_gradient_when_a_legal_action_probability_underflows(self):
        cfg = config(); out = outputs(cfg); tensors = v5_batch_to_torch(batch_columns(), cfg, training=True)
        tensors['ppo_advantages'].zero_()
        logits = out.policy_logits.detach().clone(); logits[0, :2] = torch.tensor([0.0, -1000.0])
        logits[1, :2] = 0; out = replace(out, policy_logits=logits.requires_grad_())
        actual, _ = v5_loss(out, tensors, cfg, replace(isolated_loss_config(), uniform_kl_coefficient=0.2))
        expected = 0.2 * (500 - math.log(2)) / 3
        self.assertAlmostEqual(actual.item(), expected, places=5)
        grad = torch.autograd.grad(actual, out.policy_logits)[0]
        self.assertLess(grad[0, 1].item(), 0)
        self.assertEqual(grad[:, 2:].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(grad).all())

    def test_auxiliary_hand_loss_and_gradients_preserve_unknown_masks_and_full_source_axis(self):
        cfg = config(); out = outputs(cfg); tensors = v5_batch_to_torch(batch_columns(), cfg, training=True)
        tensors['ppo_advantages'].zero_()
        actual, metrics = v5_loss(out, tensors, cfg, replace(isolated_loss_config(), auxiliary_weight=0.1))
        expected = 0.1 * (math.log(5) + math.log(2) + math.log(3) + math.log(2) + 0.13)
        self.assertAlmostEqual(actual.item(), expected, places=6)
        actual.backward()
        self.assertTrue(torch.isfinite(out.hidden_variant_logits.grad).all())
        self.assertEqual(out.hidden_variant_logits.grad[0, 0, 3:].abs().sum().item(), 0)
        self.assertEqual(out.hidden_variant_logits.grad[1:].abs().sum().item(), 0)
        self.assertEqual(out.survival_logits.grad[1:].abs().sum().item(), 0)
        self.assertEqual(metrics['survival_loss_count'], 2)
        corrupted = dict(tensors); corrupted['pokemon_end_target_mask'] = tensors['pokemon_end_target_mask'].clone()
        corrupted['pokemon_end_target_mask'][1, 0] = True
        with self.assertRaises(ValueError):v5_loss(outputs(cfg), corrupted, cfg, V5LossConfig())

    def test_D_teacher_policy_loss_is_separate_and_A_B_refuse_treatment_labels(self):
        cfg = config('D'); out = outputs(cfg); tensors = v5_batch_to_torch(batch_columns('D'), cfg, training=True)
        tensors['ppo_advantages'].zero_()
        actual, _ = v5_loss(out, tensors, cfg, replace(isolated_loss_config('D'), search_policy_weight=1))
        lp = out.policy_logits[:2, :2].log_softmax(-1)
        expected = -((lp * torch.tensor([0.25, 0.75])).sum(-1) * torch.tensor([1.0, 2.0])).sum() / 3
        torch.testing.assert_close(actual, expected)
        for arm in ('A', 'B'):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                v5_loss(outputs(config(arm)), tensors, config(arm), V5LossConfig(arm=arm))

    def test_separate_value_GAE_and_nonterminal_truncation_bootstrap(self):
        rewards = torch.tensor([0.0, 1.0, 0.0]); values = torch.tensor([0.2, 0.4, 0.5], requires_grad=True)
        next_values = torch.tensor([0.4, 0.0, 0.8]); terminal = torch.tensor([False, True, False])
        continues = torch.tensor([True, False, False])
        advantage, value_target = gae_targets(rewards, values, next_values, terminal, continues,
            policy_lambda=0.5, value_lambda=0.95)
        torch.testing.assert_close(advantage, torch.tensor([0.5, 0.6, 0.3]))
        torch.testing.assert_close(value_target, torch.tensor([0.97, 1.0, 0.8]))
        self.assertFalse(advantage.requires_grad); self.assertFalse(value_target.requires_grad)
        self.assertEqual(values.detach().tolist(), [float(torch.tensor(x)) for x in (0.2, 0.4, 0.5)])

    def test_zero_weight_batch_is_finite_zero_with_zero_gradients_even_for_unknown_poison(self):
        cfg = config('D'); out = outputs(cfg)
        columns = batch_columns('D'); columns['training_weights'] = (0.0,) * 3
        tensors = v5_batch_to_torch(columns, cfg, training=True)
        loss, metrics = v5_loss(out, tensors, cfg, V5LossConfig(arm='D'))
        self.assertEqual(loss.item(), 0)
        self.assertEqual(metrics['value_loss_count'], 0)
        loss.backward()
        for value in vars(out).values():
            if value is not None:
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.equal(value.grad, torch.zeros_like(value)))

    def test_active_label_mutations_are_refused_and_never_clamped_or_inferred(self):
        cfg = config('D'); original = v5_batch_to_torch(batch_columns('D'), cfg, training=True)
        for field, index, value in (
            ('ppo_value_targets', 1, 99.0),
            ('outcome_target_mask', 0, False),
            ('search_policy_targets', (0, 2), 0.1),
            ('hidden_set_targets', (0, 0, 3), 0.1),
            ('final_hp_targets', (0, 0), 0.1),
            ('action_probability_mask', 0, False),
        ):
            tensors = dict(original); tensors[field] = original[field].clone(); tensors[field][index] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                v5_loss(outputs(cfg), tensors, cfg, V5LossConfig(arm='D'))
        for field in ('terminal_status', 'outcome_target_mask'):
            tensors = dict(original); tensors.pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                v5_loss(outputs(cfg), tensors, cfg, V5LossConfig(arm='D'))


if __name__ == '__main__':
    unittest.main()
