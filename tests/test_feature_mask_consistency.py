"""HIGH-1 latch tests: checkpoint-stamped feature masks must be read back into every
env-construction-from-checkpoint path (the mask-axis twin of the #492 belief mismatch)."""

import contextlib
import importlib.util
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pokezero.local_showdown import LocalShowdownConfig, env_config_with_checkpoint_masks
from pokezero.observation import (
    DEFAULT_OBSERVATION_FEATURE_MASKS,
    TRANSITION_TOKEN_COUNT,
    ObservationFeatureMasks,
)

K32_MASKS = ObservationFeatureMasks(transition_token_budget=32)
STATS_OFF_MASKS = ObservationFeatureMasks(stats_block=False)


class EnvConfigMaskResolutionTest(unittest.TestCase):
    def test_no_transformer_checkpoints_leaves_config_unchanged(self) -> None:
        config = LocalShowdownConfig()
        self.assertIs(env_config_with_checkpoint_masks(config, (), context="t"), config)

    def test_default_env_adopts_the_checkpoint_masks(self) -> None:
        config = LocalShowdownConfig()
        resolved = env_config_with_checkpoint_masks(config, K32_MASKS, context="t")
        self.assertEqual(resolved.feature_masks, K32_MASKS)

    def test_matching_masks_are_a_no_op(self) -> None:
        config = LocalShowdownConfig(feature_masks=K32_MASKS)
        resolved = env_config_with_checkpoint_masks(config, (K32_MASKS, K32_MASKS), context="t")
        self.assertIs(resolved, config)

    def test_conflicting_checkpoints_hard_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting observation feature masks"):
            env_config_with_checkpoint_masks(
                LocalShowdownConfig(), (K32_MASKS, STATS_OFF_MASKS), context="t"
            )

    def test_explicit_env_override_conflicting_with_checkpoint_hard_fails(self) -> None:
        config = LocalShowdownConfig(feature_masks=STATS_OFF_MASKS)
        with self.assertRaisesRegex(ValueError, "conflict with the loaded checkpoint"):
            env_config_with_checkpoint_masks(config, K32_MASKS, context="t")

    def test_full_default_checkpoint_keeps_default_env(self) -> None:
        config = LocalShowdownConfig()
        resolved = env_config_with_checkpoint_masks(
            config, DEFAULT_OBSERVATION_FEATURE_MASKS, context="t"
        )
        self.assertEqual(resolved.feature_masks, DEFAULT_OBSERVATION_FEATURE_MASKS)


def _torch_available() -> bool:
    from pokezero.neural_policy import torch_available

    return torch_available()


def _save_k32_checkpoint(path: Path):
    """A real saved checkpoint whose model config carries the K=32 ablation budget."""
    from pokezero.neural_policy import (
        EntityTokenTransformerPolicy,
        TransformerPolicyConfig,
        TransformerTrainingConfig,
        TransformerTrainingResult,
        save_transformer_checkpoint,
    )

    config = TransformerPolicyConfig.compact_category(
        policy_id="k32-arm",
        category_vocab=tuple(f"token-{index}" for index in range(8)),
        category_oov_buckets=2,
        window_size=1,
        embedding_dim=8,
        transformer_layers=0,
        attention_heads=1,
        feedforward_dim=8,
        dropout=0.0,
        transition_token_budget=32,
    )
    model = EntityTokenTransformerPolicy(config)
    result = TransformerTrainingResult(
        model_config=config,
        training_config=TransformerTrainingConfig(window_size=1),
        epochs=(),
    )
    save_transformer_checkpoint(path, model, result=result)
    return config


class MaskDerivationTest(unittest.TestCase):
    def test_feature_masks_from_model_config_round_trips(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import TransformerPolicyConfig, feature_masks_from_model_config

        config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",),
            category_oov_buckets=2,
            stats_block_enabled=False,
            exact_state_enabled=True,
            transition_token_budget=32,
        )
        masks = feature_masks_from_model_config(config)
        self.assertEqual(
            masks,
            ObservationFeatureMasks(
                stats_block=False, exact_state=True, transition_token_budget=32
            ),
        )
        default_config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",), category_oov_buckets=2
        )
        self.assertEqual(
            feature_masks_from_model_config(default_config), DEFAULT_OBSERVATION_FEATURE_MASKS
        )
        self.assertEqual(default_config.transition_token_budget, TRANSITION_TOKEN_COUNT)

    def test_transformer_policy_sweep_finds_model_configs(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import (
            TransformerPolicyConfig,
            TransformerTrainingConfig,
            TransformerTrainingResult,
            transformer_model_configs_from_policies,
        )
        from pokezero.policy import RandomLegalPolicy

        config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",), category_oov_buckets=2, transition_token_budget=32
        )

        class _FakeNeuralPolicy:
            result = TransformerTrainingResult(
                model_config=config,
                training_config=TransformerTrainingConfig(window_size=1),
                epochs=(),
            )

        configs = transformer_model_configs_from_policies(
            [RandomLegalPolicy(), _FakeNeuralPolicy(), object()]
        )
        self.assertEqual(configs, (config,))


class K32HarnessPathTest(unittest.TestCase):
    """Each harness path must build a K=32 env for a K=32 checkpoint."""

    def test_neural_cli_benchmark_builds_k32_env(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_cli import main as neural_cli_main

        captured: dict[str, object] = {}

        def fake_benchmark_rollouts(*, games, env_factory, rollout_config, seed_start, matchups):
            captured["env"] = env_factory()

            class _Report:
                def to_dict(self):
                    return {}

            return _Report()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "k32.pt"
            _save_k32_checkpoint(checkpoint_path)
            with (
                patch("pokezero.neural_cli.benchmark_rollouts", fake_benchmark_rollouts),
                patch("pokezero.neural_cli.print_benchmark_report"),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = neural_cli_main(
                    [
                        "benchmark",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--games",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )
        self.assertEqual(exit_code, 0)
        env = captured["env"]
        self.assertEqual(env.config.feature_masks, K32_MASKS)

    def test_policy_spec_resolver_builds_k32_env_config(self) -> None:
        # The shared path used by rollout_cli collect/benchmark/replay and the bootstrap
        # teacher harnesses: neural: specs contribute their stamped masks.
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.collection import env_config_with_policy_spec_masks

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "k32.pt"
            _save_k32_checkpoint(checkpoint_path)
            resolved = env_config_with_policy_spec_masks(
                LocalShowdownConfig(),
                (f"neural:{checkpoint_path}", "random-legal", None),
                context="spec harness",
            )
        self.assertEqual(resolved.feature_masks, K32_MASKS)

    def test_neural_cli_spec_mask_helper_covers_iterate_and_root_puct_paths(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_cli import _env_config_with_spec_masks
        from pokezero.neural_policy import load_transformer_model_config

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "k32.pt"
            _save_k32_checkpoint(checkpoint_path)
            loaded_config = load_transformer_model_config(checkpoint_path)
            # iterate / root-puct shape: a directly loaded model config plus policy specs.
            resolved = _env_config_with_spec_masks(
                LocalShowdownConfig(),
                (f"neural:{checkpoint_path}", "random-legal"),
                extra_model_configs=(loaded_config,),
                context="iterate",
            )
            self.assertEqual(resolved.feature_masks, K32_MASKS)
            # A conflicting full-default checkpoint alongside the K=32 arm must hard-fail.
            from pokezero.neural_policy import TransformerPolicyConfig

            default_config = TransformerPolicyConfig.compact_category(
                category_vocab=("species:a",), category_oov_buckets=2
            )
            with self.assertRaisesRegex(ValueError, "conflicting observation feature masks"):
                _env_config_with_spec_masks(
                    LocalShowdownConfig(),
                    (f"neural:{checkpoint_path}",),
                    extra_model_configs=(default_config,),
                    context="iterate",
                )

    def test_build_agent_carries_k32_masks_for_online_and_factor_paths(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.category_vocab import build_category_vocabulary
        from pokezero.online_client import build_agent

        fake_vocab = build_category_vocabulary(["species:a"], oov_buckets=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "k32.pt"
            _save_k32_checkpoint(checkpoint_path)
            with (
                patch("pokezero.randbat_vocab.gen3_category_vocabulary", return_value=fake_vocab),
                patch("pokezero.dex.load_showdown_dex_cached", return_value=object()),
            ):
                agent = build_agent(checkpoint_path, temp_dir, our_name="bot")
        self.assertEqual(agent.feature_masks, K32_MASKS)


_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    """Load a standalone scripts/*.py tool as a module (they live outside the package)."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROBE_LINES = [
    "|player|p1|Us|",
    "|player|p2|Them|",
    '|request|{"active":[{"moves":[{"move":"Flamethrower","id":"flamethrower"}]}],"side":{"id":"p1","name":"Us","pokemon":[{"ident":"p1a: Charizard","details":"Charizard, L78","condition":"100/100","active":true}]}}',
    "|switch|p1a: Charizard|Charizard, L78|100/100",
    "|switch|p2a: Xatu|Xatu, L78|100/100",
    "|turn|1",
]


def _probe_state():
    """A real mid-game PlayerRelativeBattleState (same shape every probe corpus produces)."""
    from pokezero.showdown import normalize_for_player, parse_showdown_replay

    replay = parse_showdown_replay(_PROBE_LINES, battle_id="battle-1")
    return normalize_for_player(replay, player_id="agent", player_name="Us")


def _k32_agent():
    """A build_agent-shaped agent whose checkpoint provenance stamped the K=32 masks."""
    from pokezero.category_vocab import build_category_vocabulary
    from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC

    vocab = build_category_vocabulary(
        ["species:Charizard", "species:Xatu", "status:tox", "status:none"], oov_buckets=4
    )
    policy = SimpleNamespace(
        model=object(),
        result=SimpleNamespace(model_config=SimpleNamespace(window_size=1)),
        select_action=lambda observation, rng=None: SimpleNamespace(action_index=0),
    )
    return SimpleNamespace(
        policy=policy,
        vocab=vocab,
        dex=None,
        spec=DEFAULT_REPLAY_OBSERVATION_SPEC,
        feature_masks=K32_MASKS,
        rng=random.Random(0),
        set_source=None,
    )


def _spy_encode(recorded: list):
    """Real observation_from_player_state, recording the feature_masks each call encoded with.

    Recording the signature default when the caller omits the kwarg is the point: a script
    that drops the masks records DEFAULT_OBSERVATION_FEATURE_MASKS and fails the assertion.
    """
    from pokezero.showdown import observation_from_player_state as real_encode

    def encode(state, *, feature_masks=DEFAULT_OBSERVATION_FEATURE_MASKS, **kwargs):
        recorded.append(feature_masks)
        return real_encode(state, feature_masks=feature_masks, **kwargs)

    return encode


def _fake_priors(*, model, result, observations):
    from pokezero.actions import ACTION_COUNT

    return [1.0 / ACTION_COUNT] * ACTION_COUNT


def _fake_value(*, model, result, observations):
    return 0.0


class K32ProbeScriptPathTest(unittest.TestCase):
    """Standalone probe/play scripts must encode with the checkpoint's stamped masks (the
    WS-3 probe-poisoning residual of the #502 review): a K=64/K=32 arm probed with
    default-mask encodes reads a model on observations it never trained on."""

    def _drive_behavior_probe(self):
        module = _load_script("behavior_probe")
        recorded: list = []
        state = _probe_state()

        class _FakeEnv:
            def __init__(self, config):
                self.steps = 0

            def reset(self, seed=None):
                pass

            def terminal(self):
                return None if self.steps == 0 else object()

            def requested_players(self):
                return ("p1",)

            def _state_for_player(self, player):
                return state

            def step(self, actions):
                self.steps += 1

        with (
            patch.object(module, "LocalShowdownEnv", _FakeEnv),
            patch.object(module, "observation_from_player_state", _spy_encode(recorded)),
        ):
            module._self_play_behavior(_k32_agent(), "showdown-root", 1, 1, None)
        return recorded

    def _drive_collapse_probe(self):
        module = _load_script("collapse_probe")
        recorded: list = []
        entry = SimpleNamespace(
            state=_probe_state(), legal_switch=False, setup_slots=(), active_hp=1.0
        )
        with (
            patch.object(module, "build_agent", return_value=_k32_agent()),
            patch.object(module, "observation_from_player_state", _spy_encode(recorded)),
            patch.object(module, "evaluate_transformer_action_priors", _fake_priors),
        ):
            module.probe_checkpoint("k32", "k32.pt", "showdown-root", [entry])
        return recorded

    def _drive_hazard_probe(self):
        module = _load_script("hazard_probe")
        recorded: list = []
        entry = SimpleNamespace(state=_probe_state(), turn=1)
        with (
            patch.object(module, "build_agent", return_value=_k32_agent()),
            patch.object(module, "observation_from_player_state", _spy_encode(recorded)),
            patch.object(module, "evaluate_transformer_action_priors", _fake_priors),
            patch.object(module, "evaluate_transformer_observation_value", _fake_value),
        ):
            # The dV hazard-injection section (value_self_hazard_response, the #501 ΔV
            # read) encodes 1 base + 8 counterfactual states per corpus entry.
            module.probe_checkpoint("k32", "k32.pt", "showdown-root", [entry], value_states=1)
        return recorded

    def _drive_choice_sample(self):
        module = _load_script("choice_sample")
        recorded: list = []
        state = _probe_state()
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "choice_sample.py",
                "--checkpoint", "k32.pt=k32",
                "--showdown-root", "showdown-root",
                "--num-games", "1",
                "--out", str(Path(temp_dir) / "out.json"),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(module, "sample_states_at_turn", lambda *args, **kwargs: [(state, 7)]),
                patch.object(module, "build_agent", return_value=_k32_agent()),
                patch.object(module, "observation_from_player_state", _spy_encode(recorded)),
                patch.object(module, "evaluate_transformer_action_priors", _fake_priors),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(module.main(), 0)
        return recorded

    def _drive_policy_probe(self):
        module = _load_script("policy_probe")
        recorded: list = []
        with (
            patch.object(module, "build_agent", return_value=_k32_agent()),
            patch.object(module, "observation_from_player_state", _spy_encode(recorded)),
            patch.object(module, "evaluate_transformer_action_priors", _fake_priors),
        ):
            # Runs the script's own engineered-feature assertions against the real
            # encoder output, plus the temporal-history sweep.
            module.probe_checkpoint("k32", "k32.pt", "showdown-root", _probe_state())
        return recorded

    def test_probe_script_encodes_carry_checkpoint_masks(self) -> None:
        drivers = (
            ("behavior_probe", self._drive_behavior_probe),
            ("collapse_probe", self._drive_collapse_probe),
            ("hazard_probe", self._drive_hazard_probe),
            ("choice_sample", self._drive_choice_sample),
            ("policy_probe", self._drive_policy_probe),
        )
        for name, driver in drivers:
            with self.subTest(script=name):
                recorded = driver()
                self.assertTrue(recorded, f"{name} never encoded an observation")
                self.assertEqual(
                    recorded,
                    [K32_MASKS] * len(recorded),
                    f"{name} encoded with masks other than the checkpoint's",
                )

    def test_policy_probe_capture_driver_env_adopts_masks(self) -> None:
        # capture_base_state drives real games through env.observe(), so the env itself
        # must encode with the driver checkpoint's masks.
        module = _load_script("policy_probe")
        captured: list = []

        class _FakeEnv:
            def __init__(self, config):
                captured.append(config)
                self.protocol_lines = ()

            def reset(self, seed=None):
                pass

            def terminal(self):
                return object()  # every game ends immediately: no capture, no encode

        with patch.object(module, "build_agent", return_value=_k32_agent()):
            with self.assertRaisesRegex(RuntimeError, "no target staller"):
                with patch.object(module, "LocalShowdownEnv", _FakeEnv):
                    module.capture_base_state(
                        "showdown-root", "k32.pt", 1, ("vaporeon",), max_seeds=1
                    )
        self.assertTrue(captured)
        self.assertEqual(captured[0].feature_masks, K32_MASKS)

    def test_play_against_checkpoint_env_adopts_k32_masks(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        module = _load_script("play_against_checkpoint")
        captured: list = []

        class _FakeEnv:
            def __init__(self, config):
                captured.append(config)

            def reset(self, seed=None):
                pass

            def requested_players(self):
                return ()

            def terminal(self):
                return None

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "k32.pt"
            _save_k32_checkpoint(checkpoint_path)
            with (
                patch.object(module, "LocalShowdownEnv", _FakeEnv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                module.play(
                    checkpoint=str(checkpoint_path),
                    showdown_root=temp_dir,
                    seed=1,
                    human_player="p2",
                    deterministic=True,
                )
        self.assertTrue(captured)
        self.assertEqual(captured[0].feature_masks, K32_MASKS)


class SelfplayCliSpecMaskTest(unittest.TestCase):
    def test_selfplay_iterate_neural_opponent_spec_builds_k32_env(self) -> None:
        # The linear-era harness accepts neural: specs for opponents/benchmarks; those
        # policies observe through the env, so their stamped masks must be adopted.
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.selfplay_cli import main as selfplay_cli_main

        captured: dict = {}

        def fake_run_selfplay_iterations(**kwargs):
            captured["env"] = kwargs["env_factory"]()
            return SimpleNamespace(run_dir="run", iterations=(), latest_checkpoint_path=None)

        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "k32.pt"
            _save_k32_checkpoint(checkpoint_path)
            with (
                patch("pokezero.selfplay_cli.run_selfplay_iterations", fake_run_selfplay_iterations),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = selfplay_cli_main(
                    [
                        "iterate",
                        "--run-dir", str(Path(temp_dir) / "run"),
                        "--iterations", "1",
                        "--games-per-iteration", "1",
                        "--showdown-root", temp_dir,
                        "--opponent-policy", f"neural:{checkpoint_path}",
                    ]
                )
        self.assertEqual(exit_code, 0, stderr.getvalue())
        self.assertEqual(captured["env"].config.feature_masks, K32_MASKS)


if __name__ == "__main__":
    unittest.main()
