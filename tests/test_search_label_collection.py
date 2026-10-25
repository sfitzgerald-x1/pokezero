"""Label-only collection, mandatory teacher capture, and refresh integration."""
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import io
import json
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import read_rollout_records, CollectionMetrics
from pokezero.dataset import iter_training_cache_batches
from pokezero.neural_policy import TransformerTrainingConfig, training_batch_to_torch, _transformer_loss, torch_available
from pokezero.neural_selfplay import run_neural_selfplay_iterations
from pokezero.rollout import RolloutConfig
from pokezero.search_labels import with_search_labels
from pokezero.search_label_teacher import SearchLabelCollectionConfig, SearchLabelTeacher, prepare_search_label_teacher
from pokezero.selfplay import collect_selfplay_rollouts, _run_selfplay_game_record
from tests.test_neural_selfplay import _entity_test_model_config, patched_neural_selfplay_dependencies
from tests.test_search_labels import RawActor, Teacher, context
from tests.test_selfplay import MultiActionEnv


@unittest.skipUnless(torch_available(), "requires optional neural dependencies")
class SearchLabelCollectionTest(unittest.TestCase):
    def test_teacher_is_lazy_reused_per_battle_and_reset_between_battles(self):
        teachers = []
        def factory():
            teacher = Teacher(); teachers.append(teacher); return teacher
        actor = with_search_labels(RawActor(), fraction=1, teacher_factory=factory, teacher_identity={"unit": True})
        self.assertEqual(teachers, [])
        for i in range(3):
            actor.select_action_with_context(context(decision=i), rng=random.Random(i))
        self.assertEqual(len(teachers), 1); self.assertEqual(len(teachers[0].seen), 3)
        actor.reset(); actor.select_action_with_context(context(), rng=random.Random(0))
        self.assertEqual(len(teachers), 2)

    def test_collection_preserves_actions_and_wraps_only_learner_on_both_seats_and_threads(self):
        for workers in (1, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as t:
                root = Path(t); records = []
                for enabled in (False, True):
                    extra = {}
                    if enabled:
                        extra["learner_policy_wrapper"] = lambda actor: with_search_labels(
                            actor, fraction=1, teacher_factory=Teacher,
                            teacher_identity={"checkpoint_sha256": "a" * 64})
                    name = "labels" if enabled else "off"
                    collect_selfplay_rollouts(output_path=root / (name + ".jsonl"),
                        training_cache_output_path=root / name, training_cache_max_root_bytes=None,
                        games=4, seed_start=77, env_factory=MultiActionEnv,
                        rollout_config=RolloutConfig(max_decision_rounds=2),
                        current_policy_spec="unit-raw", opponent_policy_specs=("unit-raw",),
                        policy_factory_overrides={"unit-raw": RawActor}, worker_count=workers, **extra)
                    records.append(read_rollout_records(root / (name + ".jsonl")))
                for a, b in zip(*records, strict=True):
                    self.assertEqual(a.policy_ids, b.policy_ids)
                    for sa, sb in zip(a.trajectory.steps, b.trajectory.steps, strict=True):
                        for key in ("action_index", "action_probability", "value_estimate", "reward", "observation"):
                            self.assertEqual(getattr(sa, key), getattr(sb, key), key)
                    labels = [s for s in b.trajectory.steps if "search_label" in s.metadata]
                    self.assertEqual(len(labels), 1)
                    self.assertEqual(labels[0].player_id, "p1" if (b.seed - 77) % 2 == 0 else "p2")
                batch = next(iter_training_cache_batches(root / "labels", batch_size=4))
                self.assertEqual(list(batch.search_policy_target_mask), [True] * 4)
                tensors = training_batch_to_torch(batch)
                self.assertEqual(tensors["search_value_target_mask"].tolist(), [True] * 4)
                import torch
                from pokezero.neural_policy import TransformerPolicyOutput
                output = TransformerPolicyOutput(torch.zeros((4, 9), requires_grad=True),
                    torch.zeros(4, requires_grad=True), torch.zeros((4, 9)))
                base = TransformerTrainingConfig(objective="ppo", value_loss_weight=0.4375,
                                                  normalize_advantage=False, opponent_action_loss_weight=0)
                treatment = replace(base, search_policy_loss_weight=1, search_value_mix_weight=0.5)
                off_loss, _ = _transformer_loss(output, tensors, base)
                labeled_loss, _ = _transformer_loss(output, tensors, treatment)
                blended = 0.5 * tensors["returns"] + 0.5 * tensors["search_value_targets"]
                expected_delta = torch.log(torch.tensor(6.0)) + 0.4375 * (
                    blended.square().mean() - tensors["returns"].square().mean())
                torch.testing.assert_close(labeled_loss - off_loss, expected_delta, rtol=0, atol=2e-7)
                labeled_loss.backward()
                self.assertTrue(torch.isfinite(output.policy_logits.grad).all())

    def test_prepare_uses_existing_exports_frozen_search_and_mandatory_capture(self):
        with tempfile.TemporaryDirectory() as t, ExitStack() as stack:
            root = Path(t); model = root / "model.pt"; tables = root / "tables.json"
            model.write_bytes(b"unit model"); tables.write_bytes(b"unit tables")
            checkpoint = root / "checkpoint.pt"; checkpoint.write_bytes(b"unit checkpoint")
            quarantine = SimpleNamespace(build={"unit_build": "a" * 64}, directory=root / "quarantine")
            stack.enter_context(patch("pokezero.failure_capture.configured_quarantine", return_value=quarantine))
            admission = stack.enter_context(patch("pokezero.failure_capture.require_capture_admission"))
            stack.enter_context(patch.dict("os.environ", {"POKEZERO_CAPTURE_ADMISSION": "unit-admission.json"}))
            stack.enter_context(patch("pokezero.randbat.load_gen3_randbat_source_cached",
                return_value=SimpleNamespace(metadata=SimpleNamespace(source_hash="source"))))
            resolve = stack.enter_context(patch("pokezero.mcts_eval.resolver.resolve_checkpoint_contract",
                return_value=SimpleNamespace(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
            stack.enter_context(patch("pokezero.mcts_eval.lattice.materialize_search_artifacts",
                return_value={"model_path": str(model), "tables_path": str(tables)}))
            settings = SearchLabelCollectionConfig(root, device="cpu")
            teacher = prepare_search_label_teacher(settings, root / "checkpoint.pt", iteration=26)
            self.assertTrue(resolve.called)
            self.assertEqual(admission.call_args.kwargs["build"], quarantine.build)
            self.assertEqual(admission.call_args.kwargs["scientific_path"], "engine-mcts")
            self.assertEqual(teacher.identity["refresh_iteration"], 26)
            self.assertEqual(teacher.identity["model_sha256"], hashlib.sha256(model.read_bytes()).hexdigest())
            cfg = teacher.search_config
            self.assertEqual((cfg.search_sims, cfg.search_depth, cfg.search_batch, cfg.worlds), (2048, 8, 64, 1))
            self.assertTrue(cfg.use_opponent_priors); self.assertTrue(cfg.record_training_targets)
            self.assertFalse(cfg.early_stop); self.assertIsNone(cfg.fpu_reduction)
            stack.enter_context(patch("pokezero.dex.load_showdown_dex_cached"))
            constructor = stack.enter_context(patch("pokezero.engine_search.EngineMctsPolicy"))
            teacher.make_policy()
            self.assertIs(constructor.call_args.kwargs["scientific_capture"], True)

    def test_original_env_and_policy_initialization_order_is_preserved(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                events = []
                def env():
                    events.append("env"); return MultiActionEnv()
                def factory(name):
                    def make():
                        events.append(name); return RawActor()
                    return make
                def wrap(actor):
                    events.append("wrap"); return actor
                _run_selfplay_game_record(game_index=0, seed_start=1, env_provider=env,
                    rollout_config=RolloutConfig(max_decision_rounds=2), current_policy_spec="learner",
                    opponent_specs=("opponent",), opponent_pool_entries=None,
                    policy_factories={"learner": factory("p1"), "opponent": factory("p2")},
                    learner_policy_wrapper=wrap if enabled else None)
                self.assertEqual(events, ["env", "p1", "p2"] + (["wrap"] if enabled else []))

    def test_external_manifest_refuses_config_build_and_late_artifact_drift(self):
        from pokezero.engine_search import EngineMctsConfig
        for mutation in ("valid", "checkpoint", "model", "tables", "build", "depth", "priors", "early_stop", "worlds", "strict", "belief_source"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as t, ExitStack() as stack:
                root = Path(t); paths = {}
                for key in ("checkpoint", "model", "tables"):
                    paths[key] = root / (key + ".unit"); paths[key].write_bytes(key.encode())
                settings = SearchLabelCollectionConfig(root)
                config = EngineMctsConfig(leaf_eval="model", worlds=1, model_world_workers=1,
                    checkpoint_path=str(paths["checkpoint"]), model_path=str(paths["model"]), tables_path=str(paths["tables"]),
                    search_sims=2048, search_batch=64, search_depth=8, use_opponent_priors=True, record_training_targets=True)
                identity = {"schema": "pokezero.search-label-teacher.v1", "collection": settings.to_dict(),
                    "checkpoint_path": str(paths["checkpoint"]), "search_config": asdict(config),
                    "capture_build": {"unit_build": "abc"}, "belief_source_sha256": "source", "refresh_iteration": 1,
                    **{key + "_sha256": hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}}
                manifest = root / "teacher.json"; manifest.write_text(json.dumps(identity))
                quarantine = SimpleNamespace(build=identity["capture_build"], directory=root)
                stack.enter_context(patch("pokezero.search_label_teacher._require_capture", return_value=quarantine))
                stack.enter_context(patch("pokezero.failure_capture.configured_quarantine", return_value=quarantine))
                source = SimpleNamespace(metadata=SimpleNamespace(source_hash="source"))
                stack.enter_context(patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=source))
                stack.enter_context(patch("pokezero.dex.load_showdown_dex_cached"))
                stack.enter_context(patch("pokezero.mcts_eval.resolver.resolve_checkpoint_contract"))
                stack.enter_context(patch("pokezero.mcts_eval.resolver.validate_encoder_tables"))
                constructor = stack.enter_context(patch("pokezero.engine_search.EngineMctsPolicy"))
                teacher = SearchLabelTeacher.from_manifest(manifest)
                if mutation == "valid":
                    teacher.make_policy(); self.assertTrue(constructor.call_args.kwargs["scientific_capture"])
                elif mutation in paths:
                    paths[mutation].write_bytes(b"changed after load")
                    # Factory failure is outside ordinary search exception masking.
                    actor = with_search_labels(RawActor(), fraction=1,
                        teacher_factory=teacher.make_policy, teacher_identity=teacher.identity)
                    with self.assertRaisesRegex(ValueError, "provenance drift"):
                        actor.select_action_with_context(context(), rng=random.Random(3))
                    constructor.assert_not_called()
                else:
                    if mutation == "build":
                        identity["capture_build"] = {"unit_build": "different"}
                    elif mutation == "belief_source":
                        identity["belief_source_sha256"] = "different"
                    else:
                        key, value = {"depth": ("search_depth", 2), "priors": ("use_opponent_priors", False),
                            "early_stop": ("early_stop", True), "worlds": ("worlds", 4), "strict": ("strict_fallbacks", True)}[mutation]
                        identity["search_config"][key] = value
                    manifest.write_text(json.dumps(identity))
                    with self.assertRaises(ValueError):
                        SearchLabelTeacher.from_manifest(manifest)
                    constructor.assert_not_called()

    def test_external_collector_cli_wires_manifest_and_keeps_off_kwargs_unchanged(self):
        from pokezero.rollout_cli import main
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as t, ExitStack() as stack:
                metrics = CollectionMetrics(games=2, elapsed_seconds=1, total_decision_rounds=2,
                    total_simulator_turns=2, p1_wins=1, p2_wins=1, ties=0, capped_games=0)
                collect = stack.enter_context(patch("pokezero.rollout_cli.collect_selfplay_rollouts", return_value=metrics))
                teacher = SimpleNamespace(settings=SearchLabelCollectionConfig(Path("/unit/showdown")), wrap=lambda actor: actor)
                load = stack.enter_context(patch("pokezero.search_label_teacher.SearchLabelTeacher.from_manifest", return_value=teacher))
                stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
                argv = ["collect-selfplay-training-cache", "--games", "2", "--out", str(Path(t) / "cache"),
                        "--showdown-root", "/unit/showdown", "--current-policy", "simple-legal", "--opponent-policy", "random-legal"]
                if enabled:
                    argv += ["--search-label-teacher", "unit-teacher.json"]
                self.assertEqual(main(argv), 0)
                if enabled:
                    self.assertIs(collect.call_args.kwargs["learner_policy_wrapper"], teacher.wrap)
                    load.assert_called_once_with(Path("unit-teacher.json"))
                else:
                    load.assert_not_called()
                    self.assertNotIn("learner_policy_wrapper", collect.call_args.kwargs)

    def test_iteration_cli_wires_label_mode_and_explicit_treatment_weights(self):
        from pokezero.neural_cli import main
        from pokezero.category_vocab import build_category_vocabulary
        with ExitStack() as stack:
            stack.enter_context(patch("pokezero.randbat_vocab.gen3_category_vocabulary",
                return_value=build_category_vocabulary(["species:unit"], oov_buckets=16)))
            stack.enter_context(patch("pokezero.randbat_vocab.gen3_category_string_aliases", return_value={}))
            run = stack.enter_context(patch("pokezero.neural_cli.run_neural_selfplay_iterations",
                return_value=SimpleNamespace(run_dir=Path("unit"), iterations=(), latest_checkpoint_path=None)))
            stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            self.assertEqual(main(["iterate", "--run-dir", "unit", "--iterations", "1", "--games-per-iteration", "2",
                "--initial-policy", "random-legal", "--showdown-root", "/unit/showdown", "--objective", "ppo", "--search-labels", "--search-label-device", "cpu",
                "--search-policy-loss-weight", "1", "--search-value-mix-weight", ".5"]), 0)
            args = run.call_args.kwargs
            self.assertEqual(args["search_label_config"].fraction, 0.1)
            self.assertEqual(args["search_label_config"].refresh_iterations, 25)
            self.assertEqual(args["training_config"].search_policy_loss_weight, 1)
            self.assertEqual(args["training_config"].search_value_mix_weight, 0.5)

    def test_refresh_at_start_each_25_iterations_and_immediately_on_resume(self):
        collected = []; trained = []; models = []; generations = []
        def prepare(settings, checkpoint, *, iteration):
            generations.append((iteration, str(checkpoint)))
            return SimpleNamespace(identity={"refresh_iteration": iteration}, wrap=lambda actor: actor)
        with tempfile.TemporaryDirectory() as t, ExitStack() as stack:
            run = Path(t) / "run"
            stack.enter_context(patched_neural_selfplay_dependencies(
                collected=collected, trained_paths=trained, trained_initial_models=models))
            # Unit fixture dependencies normally have no initial checkpoint.
            stack.enter_context(patch("pokezero.neural_selfplay._initial_neural_model_from_policy_spec", return_value=None))
            stack.enter_context(patch("pokezero.neural_selfplay.prepare_search_label_teacher", side_effect=prepare))
            options = dict(run_dir=run, games_per_iteration=2, env_factory=lambda: None,
                rollout_config=RolloutConfig(max_decision_rounds=2), model_config=_entity_test_model_config(),
                training_config=TransformerTrainingConfig(objective="ppo", batch_size=2),
                initial_policy_spec="neural:/unit/initial.pt", fixed_opponent_policy_specs=("random-legal",),
                collector_advancement_mode="always", evaluation_games=1,
                search_label_config=SearchLabelCollectionConfig(Path("/unit/showdown")))
            run_neural_selfplay_iterations(iterations=27, **options)
            self.assertEqual([i for i, _ in generations], [1, 26])
            self.assertIn("iteration-0025", generations[1][1])
            run_neural_selfplay_iterations(iterations=1, resume=True, **options)
            self.assertEqual([i for i, _ in generations], [1, 26, 28])
            self.assertIn("iteration-0027", generations[2][1])
            self.assertTrue(all("learner_policy_wrapper" in call for call in collected))
            self.assertEqual(json.loads((run / "iteration-0028" / "search-teacher.json").read_text())["refresh_iteration"], 28)
