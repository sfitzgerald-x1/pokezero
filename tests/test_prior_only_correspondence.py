"""Authored priors only; no native work, source reset or scientific replay."""
from copy import deepcopy
from contextlib import ExitStack
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.prior_only_correspondence import prior_only_binding, prior_only_collector_gate
from pokezero.root_prior_correspondence import SCHEMA, exact_collector_gate


class PriorOnlyCorrespondenceTests(unittest.TestCase):
    def reports(self, sims=1024):
        binding = dict(source_configuration={"format_id": "gen3randombattle"},
            selector_configurations={"sampled4": dict(worlds=4, search_sims=sims,
                leaf_eval="model", model_leaf_override=None, strict_fallbacks=True,
                root_prior_exact_aggregate=True, c_puct=1.4)},
            world_rng="per-decision", source_files={"engine_search.py": "a"*64},
            **{k+"_sha256": "b"*64 for k in ("checkpoint", "model", "tables", "engine", "search")})
        collector = dict(schema=SCHEMA, complete=True, bindings=binding,
            expected=[dict(seed=12, round=i, variant="sampled4") for i in (1, 2)],
            rows=[dict(seed=12, round=i, boundary_sha256="c"*64, legal=[0, 2],
                variants={"sampled4": dict(native_prior_pre_display={"0": .75, "2": .25},
                    engine_action=2, iterations=sims*4)}) for i in (1, 2)])
        replay = deepcopy(collector)
        replay["bindings"] = prior_only_binding(binding)
        for row in replay["rows"]:
            row["variants"]["sampled4"].update(engine_action=0, iterations=64)
        return collector, replay

    def test_only_authorized_budgets_pass_without_action_or_work_equality(self):
        for sims in (1024, 4096):
            left, right = self.reports(sims)
            before = deepcopy((left, right))
            verdict = prior_only_collector_gate(left, right)
            self.assertTrue(verdict["passed"])
            self.assertEqual(verdict["compared"], 2)
            self.assertEqual(verdict["gate_tv"], 0.)
            self.assertFalse(verdict["selected_action_compared"])
            self.assertEqual((left, right), before)
            self.assertFalse(exact_collector_gate(left, right)["passed"])

    def test_every_declared_root_is_required(self):
        left, right = self.reports()
        right["rows"].pop()
        self.assertFalse(prior_only_collector_gate(left, right)["passed"])

    def test_duplicate_wrong_seed_boundary_and_mapping_fail(self):
        for mode in ("duplicate", "seed", "boundary", "mapping", "fallback", "rounded"):
            left, right = self.reports()
            row = right["rows"][0]
            if mode == "duplicate": right["rows"].append(deepcopy(row))
            elif mode == "seed": row["seed"] = 13
            elif mode == "boundary": row["boundary_sha256"] = "d"*64
            elif mode == "mapping": row["legal"] = [0, 1]
            elif mode == "fallback": row["variants"]["sampled4"]["error"] = "unavailable"
            else: row["variants"]["sampled4"].pop("native_prior_pre_display")
            with self.subTest(mode=mode):
                self.assertFalse(prior_only_collector_gate(left, right)["passed"])

    def test_even_sub_display_prior_difference_fails(self):
        left, right = self.reports()
        right["rows"][0]["variants"]["sampled4"]["native_prior_pre_display"] = {
            "0": .750000001, "2": .249999999}
        self.assertFalse(prior_only_collector_gate(left, right)["passed"])

    def test_no_other_configuration_or_artifact_drift_is_permitted(self):
        for mode in ("worlds", "leaf", "cpuct", "rng", "checkpoint", "source", "budget"):
            left, right = self.reports()
            config = right["bindings"]["selector_configurations"]["sampled4"]
            if mode == "worlds": config["worlds"] = 1
            elif mode == "leaf": config["model_leaf_override"] = "hp_fraction"
            elif mode == "cpuct": config["c_puct"] = 1.5
            elif mode == "rng": right["bindings"]["world_rng"] = "per-game"
            elif mode == "checkpoint": right["bindings"]["checkpoint_sha256"] = "d"*64
            elif mode == "source": right["bindings"]["source_files"]["engine_search.py"] = "d"*64
            else: config["search_sims"] = 64
            with self.subTest(mode=mode):
                self.assertFalse(prior_only_collector_gate(left, right)["passed"])

    def test_unapproved_selection_recipe_is_not_normalized(self):
        for mode in ("budget", "worlds", "leaf", "fallback", "prior", "labels", "rng"):
            left, right = self.reports()
            config = left["bindings"]["selector_configurations"]["sampled4"]
            if mode == "budget": config["search_sims"] = 80
            elif mode == "worlds": config["worlds"] = 1
            elif mode == "leaf": config["model_leaf_override"] = "hp_fraction"
            elif mode == "fallback": config["strict_fallbacks"] = False
            elif mode == "prior": config["root_prior_exact_aggregate"] = False
            elif mode == "labels": left["bindings"]["selector_configurations"]["oracle"] = config
            else: left["bindings"]["world_rng"] = "per-game"
            with self.subTest(mode=mode):
                self.assertFalse(prior_only_collector_gate(left, right)["passed"])

    def test_open_or_invalid_reports_do_not_pass(self):
        for side in (0, 1):
            reports = self.reports()
            reports[side]["complete"] = False
            self.assertFalse(prior_only_collector_gate(*reports)["passed"])
        self.assertFalse(prior_only_collector_gate({}, {})["passed"])

    def test_cli_prior_only_replays_every_root_and_not_selected_actions(self):
        script = Path(__file__).resolve().parents[1]/"scripts/live_root_prior_check.py"
        spec = importlib.util.spec_from_file_location("authored_prior_only_cli", script)
        live = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(live)
        for drift in (False, True):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                artifact = root/"input"
                artifact.write_bytes(b"authored")
                sha = hashlib.sha256(b"authored").hexdigest()
                collector, replay = self.reports()
                for binding in (collector["bindings"], replay["bindings"]):
                    for key in ("checkpoint_sha256", "model_sha256", "tables_sha256"):
                        binding[key] = sha
                if drift: replay["bindings"]["selector_configurations"]["sampled4"]["c_puct"] = 9.
                collector_path = root/"collector.json"
                collector_path.write_text(json.dumps(collector))
                stack.enter_context(patch.object(live, "input_binding", return_value=replay["bindings"]))
                config = SimpleNamespace(resolved_showdown_root=lambda: root)
                stack.enter_context(patch.object(live, "env_config_with_policy_spec_masks", return_value=config))
                env = Mock()
                env.snapshot.return_value.bridge_snapshot = {}
                create = stack.enter_context(patch.object(live, "LocalShowdownEnv", return_value=env))
                stack.enter_context(patch.object(live, "load_showdown_dex_cached"))
                stack.enter_context(patch.object(live, "load_gen3_randbat_source_cached"))
                stack.enter_context(patch.object(live, "_true_teams_from_bridge_snapshot", return_value={
                    "p1": {"packed": "own"}, "p2": {"packed": "opponent"}}))
                stack.enter_context(patch.object(live, "load_transformer_policy"))
                engine = Mock()
                engine.select_action_with_context.return_value = SimpleNamespace(action_index=0, metadata={
                    "root_allocation": {"arms": [dict(action_index=a, model_prior_pre_display=p)
                        for a, p in ((0, .75), (2, .25))]}})
                construct = stack.enter_context(patch.object(live, "EngineMctsPolicy", return_value=engine))
                stack.enter_context(patch.object(live, "_CapturingPolicy",
                    side_effect=lambda policy, sink: SimpleNamespace(sink=sink)))
                stack.enter_context(patch.object(live, "boundary_fingerprint", return_value="c"*64))
                def rollout(**kwargs):
                    for index in (1, 2):
                        context = SimpleNamespace(player_id="p1", decision_round_index=index,
                            observation=SimpleNamespace(legal_action_mask=[True, False, True], metadata={}))
                        decision = SimpleNamespace(action_index=2, metadata={"policy_distribution": [.75, 0., .25]})
                        kwargs["policies"]["p1"].sink(context, decision)
                    return SimpleNamespace(terminal=SimpleNamespace(turn_count=2))
                run = stack.enter_context(patch.object(live, "continue_rollout_from_current_state", side_effect=rollout))
                out = root/"out.json"
                args = ["--checkpoint", str(artifact), "--model-ts", str(artifact), "--tables", str(artifact),
                    "--showdown-root", str(root), "--seeds", "12", "--variants", "sampled4", "--sims", "16",
                    "--strict-fallbacks", "--prior-only", "--collector-report", str(collector_path), "--out", str(out)]
                if drift:
                    with self.assertRaisesRegex(ValueError, "beyond the declared iteration budget"):
                        live.main(args)
                    create.assert_not_called(); run.assert_not_called()
                    self.assertFalse(out.exists())
                else:
                    self.assertEqual(live.main(args), 0)
                    result = json.loads(out.read_text())
                    self.assertTrue(result["collector_gate"]["passed"])
                    self.assertEqual(result["collector_gate"]["compared"], 2)
                    self.assertEqual(construct.call_args.kwargs["config"].search_sims, 16)
                    self.assertEqual(engine.select_action_with_context.call_count, 2)
                    env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
