"""Actual sibling admission, head-specific gaps and artifact-selection isolation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from pokezero.value_sibling_analysis import (analyze, capture_group, canonical_bytes,
    sha256, summarize_groups, validate_group, write_capture)
from pokezero.neural_policy import torch_available


def observation(signal=0):
    numeric = [[0.] * 132 for _ in range(23)]
    numeric[0][0] = signal
    return dict(categorical_ids=[[1] * 41 for _ in range(23)], numeric_features=numeric,
        token_type_ids=[0] * 23, attention_mask=[True] * 23, legal_action_mask=[True] * 9,
        schema_version="pokezero.observation.v4")


def group():
    root = dict(battle_id="battle-fixture", turn=3, seat="p2", root_sha256="a" * 64,
        source_receipt_sha256="b" * 64, world_id="world-1", chance_id="chance-2",
        source_commit="fixture", collector_id="fixture", split="test", legal_actions=[0, 1, 2],
        opponent_action=4, selection="all_legal_actions_before_values",
        observation_view="public_plus_root_private", belief_update="root_belief_plus_branch_reveals")
    branches = [dict(action=i, root_sha256=root["root_sha256"], world_id="world-1",
        chance_id="chance-2", opponent_action=4, observation_seat="p2", child_state_sha256=str(i) * 64,
        terminal=False, terminal_outcome=None, observation=observation(i)) for i in range(3)]
    return capture_group(root=root, branches=branches)


class SiblingContractTests(unittest.TestCase):
    def test_each_head_selects_actual_value_top_two_not_policy_probability(self):
        g = group()
        result = summarize_groups([g], lambda obs: dict(champion=[.3, .2, -.9], refit=[-.8, .2, .4]),
                                  dict(champion=.1, refit=.19))
        heads = result["rows"][0]["heads"]
        self.assertEqual(heads["champion"]["top_two_actions"], [0, 1])
        self.assertEqual(heads["refit"]["top_two_actions"], [2, 1])
        self.assertAlmostEqual(heads["refit"]["gap_return_scale"], .2)
        self.assertEqual(result["heads"]["refit"]["share_gap_exceeds_external_outcome_rmse"], 1.)
        self.assertIn("UNMEASURED", result["actual_sibling_error"])

    def test_complete_roster_context_seat_and_child_bytes_are_required(self):
        mutations = [lambda g: g["branches"].pop(),
            lambda g: g["branches"][0].update(opponent_action=5),
            lambda g: g["branches"][0].update(root_sha256="c" * 64),
            lambda g: g["branches"][0].update(world_id="another"),
            lambda g: g["branches"][0].update(chance_id="another"),
            lambda g: g["branches"][0].update(observation_seat="p1"),
            lambda g: g["branches"][0]["observation"]["numeric_features"][0].__setitem__(0, 99),
            lambda g: g["root"].update(selection="policy_top_two"),
            lambda g: g["root"].update(split="validation"),
            lambda g: g["branches"].append(deepcopy(g["branches"][0]))]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                g = group(); mutate(g)
                with self.assertRaises(ValueError): validate_group(g)

    def test_observation_shape_boolean_finiteness_and_tanh_scale_are_not_guessed(self):
        for field, value in [("numeric_features", [[0.] * 131] * 23),
                             ("attention_mask", [1] * 23),
                             ("attention_mask", [False] * 23),
                             ("categorical_ids", [[-1] * 41] * 23)]:
            g = group(); g["branches"][0]["observation"][field] = value
            with self.assertRaises(ValueError):
                capture_group(root=g["root"], branches=g["branches"])
        g = group(); g["branches"][0]["observation"]["numeric_features"][0][0] = float("nan")
        with self.assertRaises(ValueError): capture_group(root=g["root"], branches=g["branches"])
        # Refit linear/MLP values are unbounded, so their finite output is not clipped.
        result = summarize_groups([group()], lambda _: dict(champion=[0, 0, 0], refit=[3., -2., 1.]),
                                  dict(champion=0, refit=1.))
        self.assertEqual(result["rows"][0]["heads"]["refit"]["gap_return_scale"], 2.)

    def test_terminal_groups_preserved_excluded_without_root_forward_substitution(self):
        g = group(); b = g["branches"][0]
        b.update(terminal=True, terminal_capped=False, terminal_outcome=-1,
                 terminal_receipt_sha256="e" * 64, observation=None)
        result = summarize_groups([g], lambda _: self.fail("terminal group must not enter head metric"),
                                  dict(champion=.8, refit=.8))
        self.assertEqual(result["terminal_groups_excluded"], 1)
        self.assertEqual(result["immediate_terminal_actions"], 1)
        self.assertIsNone(result["heads"]["champion"]["share_gap_exceeds_external_outcome_rmse"])
        b["terminal_capped"] = True
        with self.assertRaises(ValueError): validate_group(g)

    def test_ties_strict_threshold_and_duplicate_roots(self):
        result = summarize_groups([group()], lambda _: dict(champion=[.5, .25, 0], refit=[.5, .5, .5]),
                                  dict(champion=.25, refit=0))
        for head in result["heads"].values():
            self.assertEqual(head["share_gap_exceeds_external_outcome_rmse"], 0.)
        self.assertEqual(result["rows"][0]["heads"]["refit"]["top_two_actions"], [0, 1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_groups([group(), group()], lambda _: {}, dict(champion=0, refit=0))

    def test_capture_copies_actual_inputs_and_is_create_only(self):
        g = group()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            digest = write_capture(path, groups=[g], checkpoint_contract={}, admission_receipt_sha256="a" * 64)
            self.assertEqual(digest, sha256(path.read_bytes()))
            with self.assertRaises(FileExistsError):
                write_capture(path, groups=[g], checkpoint_contract={}, admission_receipt_sha256="a" * 64)


@unittest.skipUnless(torch_available(), "PyTorch required")
class TwoHeadRunnerTests(unittest.TestCase):
    def fixture(self, root):
        from pokezero.neural_policy import (EntityTokenTransformerPolicy, TransformerPolicyConfig,
            require_torch, neural_policy_schema_version_for_model_config)
        from pokezero.value_precision_probe import checkpoint_contract, make_value_head
        torch = require_torch(); torch.manual_seed(18)
        config = TransformerPolicyConfig.compact_category(category_vocab=["kind:a", "kind:b"],
            window_size=1, embedding_dim=16, attention_heads=4, feedforward_dim=32,
            transformer_layers=1, dropout=0., observation_schema_version="pokezero.observation.v4")
        model = EntityTokenTransformerPolicy(config)
        checkpoint = root / "checkpoint.pt"
        torch.save(dict(schema_version=neural_policy_schema_version_for_model_config(config),
            model_config=config.to_dict(), state_dict=model.state_dict(), belief_set_source_hash="fixture"), checkpoint)
        contract = checkpoint_contract(checkpoint)
        fitted = root / "head.pt"
        head = make_value_head("attention_hl_gauss", 16)
        torch.save(dict(schema="pokezero.value_probe_head.v1", kind="attention_hl_gauss", width=16,
            selected_epoch=1, features_manifest_sha256="f" * 64, state_dict=head.state_dict()), fitted)
        summary = dict(schema="pokezero.value_probe_head_refits.v1",
            status="THREE_OUTCOME_HEAD_REFITS_COMPLETE_ZEROS_UNRESOLVED", checkpoint_contract=contract,
            features_manifest_sha256="f" * 64, baseline_test=dict(mse=.74), heads={
                "linear": dict(validation_mse=.9), "attention_mlp": dict(validation_mse=.8),
                "attention_hl_gauss": dict(validation_mse=.7, selected_epoch=1,
                    artifact_sha256=sha256(fitted.read_bytes()), test=dict(mse=.73))})
        summary_path = root / "summary.json"; summary_path.write_bytes(canonical_bytes(summary))
        capture = root / "capture.json"
        write_capture(capture, groups=[group()], checkpoint_contract=contract, admission_receipt_sha256="a" * 64)
        paths = [capture, checkpoint, summary_path, fitted]
        binding = dict(zip(("capture_sha256", "checkpoint_sha256", "summary_sha256", "head_sha256"),
                           (sha256(p.read_bytes()) for p in paths)))
        return paths, binding

    def test_real_cpu_forward_both_heads_binding_and_no_production_change(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, binding = self.fixture(Path(directory))
            before = [p.read_bytes() for p in paths]
            result = analyze(*paths, **binding)
            self.assertEqual(result["selected_head_validation_only"], "attention_hl_gauss")
            self.assertEqual(result["heads"]["champion"]["scored_root_contexts"], 1)
            self.assertEqual(result["heads"]["refit"]["scored_root_contexts"], 1)
            self.assertEqual(result["model_fits"], 0); self.assertEqual(result["replay_calls"], 0)
            self.assertEqual(before, [p.read_bytes() for p in paths])
            # Repeat deterministic forward, no head fit or stochastic dropout.
            self.assertEqual(result, analyze(*paths, **binding))
            with self.assertRaisesRegex(ValueError, "SHA binding"):
                analyze(*paths, **dict(binding, capture_sha256="0" * 64))

    def test_test_metrics_never_select_head_and_unselected_head_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, binding = self.fixture(Path(directory))
            summary = json.loads(paths[2].read_bytes())
            summary["heads"]["attention_mlp"]["test"] = dict(mse=0.)
            paths[2].write_bytes(canonical_bytes(summary)); binding["summary_sha256"] = sha256(paths[2].read_bytes())
            self.assertEqual(analyze(*paths, **binding)["selected_head_validation_only"], "attention_hl_gauss")
            summary["heads"]["attention_mlp"]["validation_mse"] = .6
            paths[2].write_bytes(canonical_bytes(summary)); binding["summary_sha256"] = sha256(paths[2].read_bytes())
            with self.assertRaisesRegex((ValueError, KeyError), "artifact|selected"):
                analyze(*paths, **binding)


if __name__ == "__main__": unittest.main()
