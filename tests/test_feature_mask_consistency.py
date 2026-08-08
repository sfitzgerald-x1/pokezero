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

from pokezero.category_vocab import build_category_vocabulary
from pokezero.local_showdown import LocalShowdownConfig, env_config_from_checkpoint_provenance
from pokezero.observation import (
    DEFAULT_OBSERVATION_FEATURE_MASKS,
    TRANSITION_TOKEN_COUNT,
    ObservationFeatureMasks,
)

K32_MASKS = ObservationFeatureMasks(transition_token_budget=32)
STATS_OFF_MASKS = ObservationFeatureMasks(opponent_tendency_stats_block=False)
# The latch is fail-closed on the vocabulary axis: any checkpoint provenance at all must
# arrive with the enumeration it was trained on. These stand in for a checkpoint's
# `category_vocab_from_model_config(...)` in the axis tests below.
VOCAB = build_category_vocabulary(("species:pikachu", "move:thunderbolt"))
# Same tokens, different ORDER — the shape that silently mis-indexes embeddings.
SHIFTED_VOCAB = build_category_vocabulary(("aaa:inserted", "species:pikachu", "move:thunderbolt"))


class EnvConfigMaskResolutionTest(unittest.TestCase):
    def test_no_transformer_checkpoints_leaves_config_unchanged(self) -> None:
        config = LocalShowdownConfig()
        self.assertIs(env_config_from_checkpoint_provenance(config, (), context="t"), config)

    def test_default_env_adopts_the_checkpoint_masks(self) -> None:
        config = LocalShowdownConfig()
        resolved = env_config_from_checkpoint_provenance(config, K32_MASKS, context="t", required_vocabs=VOCAB)
        self.assertEqual(resolved.feature_masks, K32_MASKS)

    def test_matching_masks_are_a_no_op(self) -> None:
        config = LocalShowdownConfig(feature_masks=K32_MASKS, category_vocab=VOCAB)
        resolved = env_config_from_checkpoint_provenance(
            config, (K32_MASKS, K32_MASKS), context="t", required_vocabs=VOCAB
        )
        self.assertIs(resolved, config)

    def test_conflicting_checkpoints_hard_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting observation feature masks"):
            env_config_from_checkpoint_provenance(
                LocalShowdownConfig(), (K32_MASKS, STATS_OFF_MASKS), context="t", required_vocabs=VOCAB
            )

    def test_explicit_env_override_conflicting_with_checkpoint_hard_fails(self) -> None:
        config = LocalShowdownConfig(feature_masks=STATS_OFF_MASKS)
        with self.assertRaisesRegex(ValueError, "conflict with the loaded checkpoint"):
            env_config_from_checkpoint_provenance(
                config, K32_MASKS, context="t", required_vocabs=VOCAB
            )

    def test_full_default_checkpoint_keeps_default_env(self) -> None:
        config = LocalShowdownConfig()
        resolved = env_config_from_checkpoint_provenance(
            config, DEFAULT_OBSERVATION_FEATURE_MASKS, context="t", required_vocabs=VOCAB
        )
        self.assertEqual(resolved.feature_masks, DEFAULT_OBSERVATION_FEATURE_MASKS)


def _torch_available() -> bool:
    from pokezero.neural_policy import torch_available

    return torch_available()


#: Vocabulary for the v2.2 fixtures below. The eight placeholders keep them small; the
#: tt_phase/tt2_* families are what make it a LEGAL v2.2 vocabulary.
#:
#: These fixtures declare `pokezero.observation.v2.2`, and
#: `category_vocab_from_model_config` now latches the encode vocabulary FROM the checkpoint --
#: deliberately, since re-deriving it from the build renumbers rows the model trained against.
#: Bare `token-N` placeholders therefore reached the encoder, which refused at
#: showdown.py:4202 because `tt_phase:turn` was not enumerated. The guard is right and the
#: fixture was the lie: no real v2.2 model trained against a vocabulary lacking tt_phase/tt2_*.
#: The placeholders were harmless only while nothing latched the vocabulary -- the axis
#: `category_vocab_from_model_config`'s docstring calls "the axis nobody latched".
_K32_FIXTURE_VOCAB = tuple(f"token-{index}" for index in range(8)) + (
    "tt_phase:lead",
    "tt_phase:turn",
    "tt_phase:replacement",
    "tt2_kind:switch",
    "tt2_kind:move",
    "tt2_kind:cant",
)


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
        category_vocab=_K32_FIXTURE_VOCAB,
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


def _save_v2_checkpoint(path: Path):
    """A real saved checkpoint stamped with the v2 observation schema (121-column census) —
    the dual-schema window's live-training-run artifact shape."""
    from pokezero.neural_policy import (
        EntityTokenTransformerPolicy,
        TransformerPolicyConfig,
        TransformerTrainingConfig,
        TransformerTrainingResult,
        save_transformer_checkpoint,
    )

    config = TransformerPolicyConfig.compact_category(
        policy_id="v2-arm",
        category_vocab=_K32_FIXTURE_VOCAB,
        category_oov_buckets=2,
        window_size=1,
        embedding_dim=8,
        transformer_layers=0,
        attention_heads=1,
        feedforward_dim=8,
        dropout=0.0,
        observation_schema_version="pokezero.observation.v2",
        numeric_feature_count=121,
        # Real v2-era artifacts carry the 39-column categorical census (the constructor
        # default tracks the CURRENT schema — v2.2's 51 — post-flip).
        categorical_feature_count=39,
    )
    model = EntityTokenTransformerPolicy(config)
    result = TransformerTrainingResult(
        model_config=config,
        training_config=TransformerTrainingConfig(window_size=1),
        epochs=(),
    )
    save_transformer_checkpoint(path, model, result=result)
    return config


class CategoryVocabDerivationTest(unittest.TestCase):
    """`category_vocab_from_model_config` — the derivation this whole change hangs on."""

    def test_tokens_come_from_the_checkpoint_not_the_build(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import (
            TransformerPolicyConfig,
            category_vocab_from_model_config,
        )

        trained = ("move:thunderbolt", "species:pikachu")
        config = TransformerPolicyConfig.compact_category(
            category_vocab=trained, category_oov_buckets=2
        )
        with patch("pokezero.randbat_vocab.gen3_category_string_aliases", return_value={}):
            vocab = category_vocab_from_model_config(config, "/nonexistent-showdown-root")
        # The build is not consulted for tokens at all — note the root above does not exist.
        self.assertEqual(vocab.tokens, trained)
        self.assertEqual(vocab.oov_buckets, 2)

    def test_a_build_that_inserted_a_token_does_not_shift_the_trained_rows(self) -> None:
        """The exact 2026-07-29 failure, in miniature.

        A token inserted ahead of an existing one renumbers it under the build's
        enumeration. Deriving from the checkpoint must leave the trained row alone.
        """
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import (
            TransformerPolicyConfig,
            category_vocab_from_model_config,
        )

        trained = ("move:thunderbolt", "species:pikachu")
        config = TransformerPolicyConfig.compact_category(
            category_vocab=trained, category_oov_buckets=2
        )
        # What a newer build would enumerate: one extra token sorting before the others.
        newer_build = build_category_vocabulary(("aaa:added-later", *trained))
        self.assertNotEqual(
            newer_build.encode("species:pikachu"),
            build_category_vocabulary(trained).encode("species:pikachu"),
        )
        with patch("pokezero.randbat_vocab.gen3_category_string_aliases", return_value={}):
            derived = category_vocab_from_model_config(config, "/nonexistent-showdown-root")
        self.assertEqual(
            derived.encode("species:pikachu"),
            build_category_vocabulary(trained).encode("species:pikachu"),
        )

    def test_a_config_without_a_stamped_vocabulary_is_refused(self) -> None:
        from pokezero.neural_policy import category_vocab_from_model_config

        # Never silently fall back to the build — that IS the bug.
        with self.assertRaisesRegex(ValueError, "no stamped category_vocab"):
            category_vocab_from_model_config(SimpleNamespace(), "/nonexistent-showdown-root")


class EnvConfigVocabResolutionTest(unittest.TestCase):
    """The enumeration half of the latch, and the axis that fails SILENTLY.

    Masks and spec disagreements change the observation's SHAPE, so they tend to surface as
    a shape error in the forward pass. The vocabulary is a positional list of the same width
    whichever build wrote it: a token inserted mid-list renumbers everything after it and the
    encoder still produces a well-formed tensor — of rows the model learned as other tokens.
    Nothing crashes, so only these assertions stand between that and a silent wrong score.
    """

    def test_provenance_without_a_vocabulary_fails_closed(self) -> None:
        # The regression that motivated the change: masks/spec latched from the checkpoint
        # while the vocabulary was left to be re-derived from the build. Its old contract was
        # a comment on LocalShowdownConfig.category_vocab that nothing enforced.
        with self.assertRaisesRegex(ValueError, "without required_vocabs"):
            env_config_from_checkpoint_provenance(LocalShowdownConfig(), K32_MASKS, context="t")

    def test_spec_only_provenance_also_fails_closed(self) -> None:
        from pokezero.showdown import V2_REPLAY_OBSERVATION_SPEC

        # Spec alone is provenance too; the vocabulary is no less required for it.
        with self.assertRaisesRegex(ValueError, "without required_vocabs"):
            env_config_from_checkpoint_provenance(
                LocalShowdownConfig(), (), context="t",
                required_specs=V2_REPLAY_OBSERVATION_SPEC,
            )

    def test_no_provenance_at_all_is_still_a_no_op(self) -> None:
        # The control: fail-closed must not fire for envs with no model in play, which are
        # entitled to enumerate from the build.
        config = LocalShowdownConfig()
        self.assertIs(env_config_from_checkpoint_provenance(config, (), context="t"), config)

    def test_default_env_adopts_the_checkpoint_vocabulary(self) -> None:
        resolved = env_config_from_checkpoint_provenance(
            LocalShowdownConfig(), K32_MASKS, context="t", required_vocabs=VOCAB
        )
        self.assertEqual(resolved.category_vocab, VOCAB)

    def test_agreeing_checkpoints_adopt_once(self) -> None:
        resolved = env_config_from_checkpoint_provenance(
            LocalShowdownConfig(), K32_MASKS, context="t", required_vocabs=(VOCAB, VOCAB)
        )
        self.assertEqual(resolved.category_vocab, VOCAB)

    def test_conflicting_vocabularies_hard_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "different categorical vocabularies"):
            env_config_from_checkpoint_provenance(
                LocalShowdownConfig(), K32_MASKS, context="t",
                required_vocabs=(VOCAB, SHIFTED_VOCAB),
            )

    def test_explicit_env_vocabulary_conflicting_with_checkpoint_hard_fails(self) -> None:
        config = LocalShowdownConfig(category_vocab=SHIFTED_VOCAB)
        with self.assertRaisesRegex(ValueError, "conflicts with the loaded checkpoint"):
            env_config_from_checkpoint_provenance(
                config, K32_MASKS, context="t", required_vocabs=VOCAB
            )

    def test_a_reordered_vocabulary_of_the_same_tokens_is_still_a_conflict(self) -> None:
        # The whole point of the axis. SHIFTED_VOCAB is a superset here, but the failure this
        # guards is ORDER: rows are positions, so equal-length differently-ordered lists must
        # not compare equal either. Pin it directly rather than trusting dataclass equality.
        reordered = build_category_vocabulary(("move:thunderbolt", "species:pikachu"))
        same_tokens = build_category_vocabulary(("species:pikachu", "move:thunderbolt"))
        # build_category_vocabulary sorts, so these ARE equal — the sorted invariant is what
        # makes the enumeration reproducible. Construct the unsorted case explicitly to prove
        # the comparison is positional and not set-based.
        from pokezero.category_vocab import CategoryVocabulary

        self.assertEqual(reordered, same_tokens)
        a = CategoryVocabulary(tokens=("species:pikachu", "move:thunderbolt"))
        b = CategoryVocabulary(tokens=("move:thunderbolt", "species:pikachu"))
        self.assertNotEqual(a, b)
        self.assertNotEqual(a.encode("species:pikachu"), b.encode("species:pikachu"))
        with self.assertRaisesRegex(ValueError, "different categorical vocabularies"):
            env_config_from_checkpoint_provenance(
                LocalShowdownConfig(), K32_MASKS, context="t", required_vocabs=(a, b)
            )

    def test_matching_vocabulary_is_a_no_op(self) -> None:
        config = LocalShowdownConfig(feature_masks=K32_MASKS, category_vocab=VOCAB)
        resolved = env_config_from_checkpoint_provenance(
            config, K32_MASKS, context="t", required_vocabs=VOCAB
        )
        self.assertIs(resolved, config)


class EnvConfigSpecResolutionTest(unittest.TestCase):
    """The dual-schema half of the latch: checkpoint-stamped observation specs resolve the
    env's encode schema + width with the same adopt/agree/conflict semantics as masks."""

    def test_default_env_adopts_the_checkpoint_v2_spec(self) -> None:
        from pokezero.showdown import V2_REPLAY_OBSERVATION_SPEC

        resolved = env_config_from_checkpoint_provenance(
            LocalShowdownConfig(), (), context="t", required_specs=V2_REPLAY_OBSERVATION_SPEC,
            required_vocabs=VOCAB
        )
        self.assertEqual(resolved.observation_spec, V2_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(resolved.observation_spec.numeric_feature_count, 121)

    def test_matching_v2_1_spec_is_a_no_op(self) -> None:
        from pokezero.showdown import V2_1_REPLAY_OBSERVATION_SPEC

        # Pinned explicitly post-flip: the default env spec is v2.2 now, so a matching
        # v2.1 pair needs a v2.1 env to stay a no-op.
        config = LocalShowdownConfig(
            observation_spec=V2_1_REPLAY_OBSERVATION_SPEC, category_vocab=VOCAB
        )
        resolved = env_config_from_checkpoint_provenance(
            config,
            (),
            context="t",
            required_specs=(V2_1_REPLAY_OBSERVATION_SPEC, V2_1_REPLAY_OBSERVATION_SPEC),
            required_vocabs=VOCAB,
        )
        self.assertIs(resolved, config)

    def test_conflicting_schemas_hard_fail(self) -> None:
        from pokezero.showdown import V2_1_REPLAY_OBSERVATION_SPEC, V2_REPLAY_OBSERVATION_SPEC

        with self.assertRaisesRegex(ValueError, "conflicting observation specs"):
            env_config_from_checkpoint_provenance(
                LocalShowdownConfig(),
                (),
                context="t",
                required_specs=(V2_REPLAY_OBSERVATION_SPEC, V2_1_REPLAY_OBSERVATION_SPEC),
                required_vocabs=VOCAB,
            )

    def test_explicit_env_spec_conflicting_with_checkpoint_hard_fails(self) -> None:
        from pokezero.showdown import V2_1_REPLAY_OBSERVATION_SPEC, V2_REPLAY_OBSERVATION_SPEC

        config = LocalShowdownConfig(observation_spec=V2_REPLAY_OBSERVATION_SPEC)
        with self.assertRaisesRegex(ValueError, "conflicts with the loaded checkpoint"):
            env_config_from_checkpoint_provenance(
                config, (), context="t", required_specs=V2_1_REPLAY_OBSERVATION_SPEC,
                required_vocabs=VOCAB,
            )

    def test_masks_and_specs_resolve_together(self) -> None:
        from pokezero.showdown import V2_REPLAY_OBSERVATION_SPEC

        resolved = env_config_from_checkpoint_provenance(
            LocalShowdownConfig(),
            K32_MASKS,
            context="t",
            required_specs=V2_REPLAY_OBSERVATION_SPEC,
            required_vocabs=VOCAB,
        )
        self.assertEqual(resolved.feature_masks, K32_MASKS)
        self.assertEqual(resolved.observation_spec, V2_REPLAY_OBSERVATION_SPEC)


class MaskDerivationTest(unittest.TestCase):
    def test_v3_checkpoint_budget_is_schema_bounded_and_defaults_to_64_on_load(self) -> None:
        from pokezero.neural_policy import TransformerPolicyConfig
        from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V3
        from pokezero.showdown import V3_REPLAY_OBSERVATION_SPEC

        config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",),
            category_oov_buckets=2,
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V3,
            categorical_feature_count=V3_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
            numeric_feature_count=V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            token_count=V3_REPLAY_OBSERVATION_SPEC.token_count,
            transition_token_budget=64,
        )
        payload = config.to_dict()
        payload.pop("transition_token_budget")
        self.assertEqual(TransformerPolicyConfig.from_dict(payload).transition_token_budget, 64)

        with self.assertRaisesRegex(ValueError, "0..64"):
            TransformerPolicyConfig.compact_category(
                category_vocab=("species:a",),
                category_oov_buckets=2,
                observation_schema_version=OBSERVATION_SCHEMA_VERSION_V3,
                categorical_feature_count=V3_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
                numeric_feature_count=V3_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
                token_count=V3_REPLAY_OBSERVATION_SPEC.token_count,
                transition_token_budget=65,
            )

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
                opponent_tendency_stats_block=False, exact_state=True, transition_token_budget=32
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
                        "--policy-id",
                        "candidate-k32",
                        "--games",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )
        self.assertEqual(exit_code, 0)
        env = captured["env"]
        self.assertEqual(env.config.feature_masks, K32_MASKS)

    def test_neural_cli_benchmark_reference_alias_preserves_k32_mask_latch(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_cli import main as neural_cli_main

        class FakeCandidatePolicy:
            policy_id = "candidate"

        captured: dict[str, object] = {}

        def fake_benchmark_rollouts(*, games, env_factory, rollout_config, seed_start, matchups):
            captured["env"] = env_factory()

            class _Report:
                def to_dict(self):
                    return {}

            return _Report()

        with tempfile.TemporaryDirectory() as temp_dir:
            # The candidate USED to be unread -- patching `_policy_from_checkpoint` was enough,
            # hence the name. That stopped being true when the encode vocabulary became
            # checkpoint-latched: `neural_cli` now reads the candidate's `model_config` for
            # `category_vocab_from_model_config`, independently of how the policy is built. The
            # file was never written, so the CLI exited 1 on [Errno 2] and the test saw only
            # `1 != 0` because it redirects stderr.
            checkpoint_path = Path(temp_dir) / "candidate.pt"
            reference_path = Path(temp_dir) / "k32-reference.pt"
            _save_k32_checkpoint(reference_path)
            _save_k32_checkpoint(checkpoint_path)
            with (
                patch("pokezero.neural_cli._policy_from_checkpoint", return_value=FakeCandidatePolicy()),
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
                        "--benchmark-reference-policy",
                        f"neural:{reference_path}",
                        "--benchmark-reference-policy-id",
                        "pool-k32",
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
                # build_agent now derives the vocabulary from the CHECKPOINT's stamped
                # tokens; showdown_root supplies only the aliases, so that is what a fake
                # root has to stand in for here.
                patch("pokezero.randbat_vocab.gen3_category_string_aliases", return_value={}),
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
    """A build_agent-shaped agent whose checkpoint provenance stamped the K=32 masks.

    Pinned to the v2.1 spec the K=32 arm was trained under: the probe corpus states are
    normalized without the turn-merged stream, and the point of these tests is the MASK
    latch, not the schema default (which is v2.2 post-flip).
    """
    from pokezero.category_vocab import build_category_vocabulary
    from pokezero.showdown import V2_1_REPLAY_OBSERVATION_SPEC

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
        spec=V2_1_REPLAY_OBSERVATION_SPEC,
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


class V2CheckpointHarnessPathTest(unittest.TestCase):
    """Dual-schema window: a saved v2 (121-column) checkpoint must load without refusal and
    drive every env-construction path to the v2 encode — the live-training-run guarantee."""

    def test_v2_checkpoint_loads_and_derives_the_v2_spec(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import (
            load_transformer_model_config,
            observation_spec_from_model_config,
        )
        from pokezero.showdown import V2_REPLAY_OBSERVATION_SPEC

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "v2.pt"
            _save_v2_checkpoint(checkpoint_path)
            config = load_transformer_model_config(checkpoint_path)
        self.assertEqual(config.observation_schema_version, "pokezero.observation.v2")
        self.assertEqual(config.numeric_feature_count, 121)
        self.assertEqual(
            observation_spec_from_model_config(config), V2_REPLAY_OBSERVATION_SPEC
        )

    def test_policy_spec_resolver_builds_v2_env_config(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.collection import env_config_with_policy_spec_masks
        from pokezero.showdown import V2_REPLAY_OBSERVATION_SPEC

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "v2.pt"
            _save_v2_checkpoint(checkpoint_path)
            resolved = env_config_with_policy_spec_masks(
                LocalShowdownConfig(),
                (f"neural:{checkpoint_path}", "random-legal", None),
                context="spec harness",
            )
        self.assertEqual(resolved.observation_spec, V2_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(resolved.observation_spec.schema_version, "pokezero.observation.v2")

    def test_neural_cli_benchmark_builds_v2_env(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_cli import main as neural_cli_main
        from pokezero.showdown import V2_REPLAY_OBSERVATION_SPEC

        captured: dict[str, object] = {}

        def fake_benchmark_rollouts(*, games, env_factory, rollout_config, seed_start, matchups):
            captured["env"] = env_factory()

            class _Report:
                def to_dict(self):
                    return {}

            return _Report()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "v2.pt"
            _save_v2_checkpoint(checkpoint_path)
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
                        "--policy-id",
                        "candidate-v2",
                        "--games",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )
        self.assertEqual(exit_code, 0)
        env = captured["env"]
        self.assertEqual(env.config.observation_spec, V2_REPLAY_OBSERVATION_SPEC)

    def test_v2_and_v2_1_checkpoints_in_one_env_hard_fail(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.collection import env_config_with_policy_spec_masks

        from pokezero.neural_policy import (
            EntityTokenTransformerPolicy,
            TransformerPolicyConfig,
            TransformerTrainingConfig,
            TransformerTrainingResult,
            save_transformer_checkpoint,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            v2_path = Path(temp_dir) / "v2.pt"
            _save_v2_checkpoint(v2_path)
            # An explicitly v2.1-stamped checkpoint (the default stamps v2.2 post-flip)
            # with the SAME default masks, so the failure is unambiguously the schema
            # axis, not the mask axis.
            v21_config = TransformerPolicyConfig.compact_category(
                policy_id="v21-arm",
                category_vocab=_K32_FIXTURE_VOCAB,
                category_oov_buckets=2,
                window_size=1,
                embedding_dim=8,
                transformer_layers=0,
                attention_heads=1,
                feedforward_dim=8,
                dropout=0.0,
                observation_schema_version="pokezero.observation.v2.1",
                numeric_feature_count=140,
                categorical_feature_count=39,
            )
            v21_path = Path(temp_dir) / "v21.pt"
            save_transformer_checkpoint(
                v21_path,
                EntityTokenTransformerPolicy(v21_config),
                result=TransformerTrainingResult(
                    model_config=v21_config,
                    training_config=TransformerTrainingConfig(window_size=1),
                    epochs=(),
                ),
            )
            with self.assertRaisesRegex(ValueError, "conflicting observation specs"):
                env_config_with_policy_spec_masks(
                    LocalShowdownConfig(),
                    (f"neural:{v2_path}", f"neural:{v21_path}"),
                    context="spec harness",
                )


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
                # temp_dir is a stand-in showdown_root with no dex on disk. The vocabulary
                # now resolves EAGERLY at latch time (it used to be built lazily on the
                # first observe), so the alias lookup is reached here and needs stubbing.
                patch("pokezero.randbat_vocab.gen3_category_string_aliases", return_value={}),
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
                # See play_against_checkpoint above: temp_dir has no dex, and the vocabulary
                # axis now resolves eagerly through the latch.
                patch("pokezero.randbat_vocab.gen3_category_string_aliases", return_value={}),
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




TIER2_OFF_MASKS = ObservationFeatureMasks(tier2_residuals=False)


class Tier2ProvenanceLatchTest(unittest.TestCase):
    """#505 follow-up MED: tier2_residuals must latch through checkpoint provenance —
    a pre-#505 checkpoint (payload lacking the field) resolves to mask-OFF, never the
    dataclass default."""

    def test_payload_missing_field_resolves_off(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import TransformerPolicyConfig, feature_masks_from_model_config

        config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",), category_oov_buckets=2
        )
        self.assertTrue(config.tier2_residuals)  # new checkpoints self-describe as on
        payload = config.to_dict()
        payload.pop("tier2_residuals")  # a pre-#505 checkpoint payload
        legacy = TransformerPolicyConfig.from_dict(payload)
        self.assertFalse(legacy.tier2_residuals)
        self.assertFalse(feature_masks_from_model_config(legacy).tier2_residuals)

    def test_explicit_value_round_trips_and_derives(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import TransformerPolicyConfig, feature_masks_from_model_config

        for value in (True, False):
            config = TransformerPolicyConfig.compact_category(
                category_vocab=("species:a",), category_oov_buckets=2, tier2_residuals=value
            )
            round_tripped = TransformerPolicyConfig.from_dict(config.to_dict())
            self.assertEqual(round_tripped.tier2_residuals, value)
            self.assertEqual(feature_masks_from_model_config(round_tripped).tier2_residuals, value)

    def test_pre_505_v2_checkpoint_file_resolves_off(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        import torch

        from pokezero.neural_policy import load_transformer_model_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "k32.pt"
            _save_k32_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            # Rewrite the file as a pre-#505 spec-v2 checkpoint: same schema, no field.
            payload["model_config"].pop("tier2_residuals", None)
            torch.save(payload, path)
            legacy_config = load_transformer_model_config(path)
            self.assertFalse(legacy_config.tier2_residuals)
            fresh_path = Path(tmp) / "fresh.pt"
            _save_k32_checkpoint(fresh_path)
            self.assertTrue(load_transformer_model_config(fresh_path).tier2_residuals)

    def test_env_adopts_tier2_off_and_conflicts_fail(self) -> None:
        resolved = env_config_from_checkpoint_provenance(
            LocalShowdownConfig(), TIER2_OFF_MASKS, context="t", required_vocabs=VOCAB
        )
        self.assertEqual(resolved.feature_masks, TIER2_OFF_MASKS)
        with self.assertRaisesRegex(ValueError, "conflict with the loaded checkpoint"):
            env_config_from_checkpoint_provenance(
                LocalShowdownConfig(feature_masks=K32_MASKS), TIER2_OFF_MASKS, context="t",
                required_vocabs=VOCAB,
            )


class InvestmentProvenanceLatchTest(unittest.TestCase):
    """v2.1 batch 2: tier2_investment latches like tier2_residuals but with a FALSE
    dataclass default too — no current training run consumes the column, so both a
    missing payload field and a fresh config resolve to mask-OFF until v2.1 training
    flips the default."""

    def test_defaults_and_missing_field_resolve_off(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import TransformerPolicyConfig, feature_masks_from_model_config

        config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",), category_oov_buckets=2
        )
        self.assertFalse(config.tier2_investment)  # default off even for new configs
        payload = config.to_dict()
        payload.pop("tier2_investment")  # a pre-investment checkpoint payload
        legacy = TransformerPolicyConfig.from_dict(payload)
        self.assertFalse(legacy.tier2_investment)
        self.assertFalse(feature_masks_from_model_config(legacy).tier2_investment)

    def test_explicit_value_round_trips_and_derives(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import TransformerPolicyConfig, feature_masks_from_model_config

        for value in (True, False):
            config = TransformerPolicyConfig.compact_category(
                category_vocab=("species:a",), category_oov_buckets=2, tier2_investment=value
            )
            round_tripped = TransformerPolicyConfig.from_dict(config.to_dict())
            self.assertEqual(round_tripped.tier2_investment, value)
            self.assertEqual(
                feature_masks_from_model_config(round_tripped).tier2_investment, value
            )

    def test_residuals_only_checkpoint_masks_investment_off(self) -> None:
        # The separate-switch justification: a post-#505/pre-investment checkpoint
        # latches residuals ON with investment OFF — inexpressible with one switch.
        masks = ObservationFeatureMasks(tier2_residuals=True)
        self.assertTrue(masks.tier2_residuals)
        self.assertFalse(masks.tier2_investment)


class TrainMaskFlagsTest(unittest.TestCase):
    """Controller-path parity (#502 gap): the mask flags exist on `neural_cli train`
    and `rollout_cli collect-selfplay-training-cache`, set fresh configs, and hard-fail
    against a disagreeing checkpoint or cache."""

    def _train_args(self, extra=()):
        from pokezero.neural_cli import build_arg_parser

        return build_arg_parser().parse_args(
            ["train", "--data", "d", "--out", "o", *extra]
        )

    def test_fresh_train_flags_reach_model_config_fields(self) -> None:
        args = self._train_args(
            ["--transition-token-budget", "32", "--no-stats-block", "--no-tier2-residuals"]
        )
        self.assertEqual(args.transition_token_budget, 32)
        self.assertTrue(args.no_stats_block)
        self.assertFalse(args.no_exact_state)
        self.assertIs(args.tier2_residuals, False)
        defaults = self._train_args()
        self.assertIsNone(defaults.transition_token_budget)
        self.assertIsNone(defaults.tier2_residuals)

    def test_resume_agreement_hard_fails_on_disagreement(self) -> None:
        from types import SimpleNamespace

        from pokezero.neural_cli import _require_mask_flags_agree_with_checkpoint

        checkpoint_config = SimpleNamespace(
            stats_block_enabled=True,
            exact_state_enabled=True,
            transition_token_budget=32,
            tier2_residuals=False,
        )
        agreeing = SimpleNamespace(
            transition_token_budget=32, no_stats_block=False, no_exact_state=False, tier2_residuals=False
        )
        _require_mask_flags_agree_with_checkpoint(agreeing, checkpoint_config)  # no raise
        omitted = SimpleNamespace(
            transition_token_budget=None, no_stats_block=False, no_exact_state=False, tier2_residuals=None
        )
        _require_mask_flags_agree_with_checkpoint(omitted, checkpoint_config)  # adoption, no raise
        for kwargs, message in (
            (dict(transition_token_budget=128, no_stats_block=False, no_exact_state=False, tier2_residuals=None), "transition_token_budget"),
            (dict(transition_token_budget=None, no_stats_block=True, no_exact_state=False, tier2_residuals=None), "stats_block_enabled"),
            (dict(transition_token_budget=None, no_stats_block=False, no_exact_state=False, tier2_residuals=True), "tier2_residuals"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                _require_mask_flags_agree_with_checkpoint(SimpleNamespace(**kwargs), checkpoint_config)

    def test_cache_mask_cross_check_both_directions(self) -> None:
        import json
        from types import SimpleNamespace

        from pokezero.neural_cli import _require_cache_masks_match_model_config

        model_config = SimpleNamespace(
            stats_block_enabled=True,
            exact_state_enabled=True,
            transition_token_budget=32,
            tier2_residuals=True,
            tier2_investment=False,
        )
        # No tier2_investment key on purpose: a pre-investment cache payload must
        # resolve to investment-off and still match an investment-off model.
        matching = {
            "stats_block": True,
            "exact_state": True,
            "transition_token_budget": 32,
            "tier2_residuals": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            (cache / "metadata.json").write_text(json.dumps({"feature_masks": matching}))
            _require_cache_masks_match_model_config([cache], model_config)  # no raise
            # Direction 1: cache K=32, model K=128 -> fail.
            wide_model = SimpleNamespace(**{**vars(model_config), "transition_token_budget": 128})
            with self.assertRaisesRegex(ValueError, "mask-mismatched"):
                _require_cache_masks_match_model_config([cache], wide_model)
            # Direction 2: cache tier2-on, model tier2-off -> fail.
            masked_model = SimpleNamespace(**{**vars(model_config), "tier2_residuals": False})
            with self.assertRaisesRegex(ValueError, "mask-mismatched"):
                _require_cache_masks_match_model_config([cache], masked_model)
            # Legacy cache (no field) cannot be checked -> passes.
            (cache / "metadata.json").write_text(json.dumps({}))
            _require_cache_masks_match_model_config([cache], wide_model)

    def test_a_zero_capacity_is_a_real_capacity_not_a_missing_one(self) -> None:
        """v4 REMOVES the transition region, so its capacity is 0 and its only budget is 0.

        Resolving `transition_token_capacity or TRANSITION_TOKEN_COUNT` sent that zero through a
        FALSY-zero to the legacy 128, so a non-zero budget on a v4 spec passed validation, was
        accepted, and was then silently ignored by an encoder with no region to apply it to. That
        made v4's own contract text -- "there is no transition_token_budget knob left to mis-set"
        -- false, and failed QUIETLY, which is the worse half.
        """
        from pokezero.observation import V4_TRANSITION_TOKEN_COUNT
        from pokezero.rollout_cli import _explicit_feature_masks_from_args, build_arg_parser

        self.assertEqual(V4_TRANSITION_TOKEN_COUNT, 0, "premise: v4 carries no region")

        def resolve(budget: str):
            args = build_arg_parser().parse_args(
                [
                    "collect-selfplay-training-cache",
                    "--games", "1", "--out", "cache-out",
                    "--transition-token-budget", budget,
                ]
            )
            return _explicit_feature_masks_from_args(
                args, transition_token_capacity=V4_TRANSITION_TOKEN_COUNT
            )

        # 0 is the only admissible budget under v4, and it must still resolve.
        self.assertEqual(resolve("0").transition_token_budget, 0)
        # Anything else must be REFUSED rather than accepted-then-ignored.
        for budget in ("1", "32", "128"):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    resolve(budget)

    def test_collect_flags_resolve_explicit_masks_and_cache_metadata_records_them(self) -> None:
        import json

        from pokezero.dataset import TrainingCacheBuilder, TrajectoryDatasetConfig
        from pokezero.rollout_cli import _explicit_feature_masks_from_args, build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "collect-selfplay-training-cache",
                "--games", "1", "--out", "cache-out",
                "--transition-token-budget", "32",
            ]
        )
        masks = _explicit_feature_masks_from_args(args)
        self.assertEqual(masks, K32_MASKS)
        no_flags = build_arg_parser().parse_args(
            ["collect-selfplay-training-cache", "--games", "1", "--out", "cache-out"]
        )
        self.assertIsNone(_explicit_feature_masks_from_args(no_flags))
        # The resolved masks land in the cache metadata payload.
        builder = TrainingCacheBuilder(config=TrajectoryDatasetConfig(), feature_masks=masks)
        self.assertEqual(
            builder._feature_masks_payload,
            {
                "stats_block": True,
                "exact_state": True,
                "transition_token_budget": 32,
                "tier2_residuals": True,
                "tier2_investment": False,
                # The v4 pack's A2 ablation switch, pack-whole by default. Recorded here like
                # every other mask so the train-side cross-check can refuse a cache collected
                # under a different pack shape; caches predating the field are defaulted to
                # True on read, the same asymmetry tier2_investment uses.
                "feature_pack_last_move": True,
                # The belief-narrowing switch. Recorded like every other mask, and defaulted
                # FALSE on read: it is the one mask whose "on" state changes columns that
                # exist under every schema (candidate-set count, uncertainty), so a cache
                # predating the field cannot have been collected with it on.
                "investment_belief_narrowing": False,
                # The item-certainty narrowing switch. Same FALSE default on read and for the
                # same reason: it moves the candidate-set count and uncertainty columns, which
                # exist under every schema, so a cache predating the field cannot have had it on.
                "item_belief_narrowing": False,
            },
        )

    def test_tier2_investment_flag_tri_state_and_defaults_off(self) -> None:
        # The switch this PR adds: --tier2-investment sets the field; its ABSENCE resolves
        # OFF (asymmetric vs --tier2-residuals, which defaults on). All three subcommands
        # expose it as a BooleanOptionalAction defaulting None (the latch tri-state).
        from pokezero.neural_cli import build_arg_parser as neural_parser
        from pokezero.rollout_cli import build_arg_parser as rollout_parser

        subs = (
            (neural_parser, ["train", "--data", "d", "--out", "o"]),
            (
                neural_parser,
                [
                    "iterate", "--run-dir", "r", "--iterations", "1",
                    "--games-per-iteration", "1", "--initial-policy", "random-legal",
                ],
            ),
            (
                rollout_parser,
                ["collect-selfplay-training-cache", "--games", "1", "--out", "o"],
            ),
        )
        for parser, sub in subs:
            self.assertIsNone(parser().parse_args(sub).tier2_investment)  # absent -> None -> OFF
            self.assertIs(parser().parse_args([*sub, "--tier2-investment"]).tier2_investment, True)
            self.assertIs(
                parser().parse_args([*sub, "--no-tier2-investment"]).tier2_investment, False
            )

    def test_tier2_investment_fresh_train_config_defaults_off_and_enable_stamps_on(self) -> None:
        # A fresh train builds a config whose tier2_investment is OFF when the flag is absent
        # (byte-identity property) and ON when --tier2-investment is passed. Exercised through
        # the same _explicit_mask_requests the resume-agreement check consumes.
        from pokezero.neural_cli import _explicit_mask_requests

        omitted = self._train_args()
        self.assertIsNone(omitted.tier2_investment)
        self.assertNotIn("tier2_investment", _explicit_mask_requests(omitted))
        enabled = self._train_args(["--tier2-investment"])
        self.assertIs(_explicit_mask_requests(enabled)["tier2_investment"], True)
        disabled = self._train_args(["--no-tier2-investment"])
        self.assertIs(_explicit_mask_requests(disabled)["tier2_investment"], False)

    def test_tier2_investment_resume_agreement_hard_fails_both_directions(self) -> None:
        from types import SimpleNamespace

        from pokezero.neural_cli import _require_mask_flags_agree_with_checkpoint

        # A checkpoint trained WITH the investment channel: an explicit --no-tier2-investment
        # (disable) must hard-fail, and adoption (omitted flag) must pass.
        on_checkpoint = SimpleNamespace(
            stats_block_enabled=True, exact_state_enabled=True,
            transition_token_budget=32, tier2_residuals=True, tier2_investment=True,
        )
        omitted = SimpleNamespace(
            transition_token_budget=None, no_stats_block=False, no_exact_state=False,
            tier2_residuals=None, tier2_investment=None,
        )
        _require_mask_flags_agree_with_checkpoint(omitted, on_checkpoint)  # adoption, no raise
        agreeing = SimpleNamespace(**{**vars(omitted), "tier2_investment": True})
        _require_mask_flags_agree_with_checkpoint(agreeing, on_checkpoint)  # no raise
        disable = SimpleNamespace(**{**vars(omitted), "tier2_investment": False})
        with self.assertRaisesRegex(ValueError, "tier2_investment"):
            _require_mask_flags_agree_with_checkpoint(disable, on_checkpoint)
        # A checkpoint trained WITHOUT it: an explicit --tier2-investment (enable) hard-fails.
        off_checkpoint = SimpleNamespace(**{**vars(on_checkpoint), "tier2_investment": False})
        enable = SimpleNamespace(**{**vars(omitted), "tier2_investment": True})
        with self.assertRaisesRegex(ValueError, "tier2_investment"):
            _require_mask_flags_agree_with_checkpoint(enable, off_checkpoint)

    def test_tier2_investment_cache_cross_check_and_legacy_asymmetry(self) -> None:
        import json
        from types import SimpleNamespace

        from pokezero.neural_cli import _require_cache_masks_match_model_config

        investment_model = SimpleNamespace(
            stats_block_enabled=True, exact_state_enabled=True,
            transition_token_budget=32, tier2_residuals=True, tier2_investment=True,
        )
        off_model = SimpleNamespace(**{**vars(investment_model), "tier2_investment": False})
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            invest_cache = {
                "stats_block": True, "exact_state": True,
                "transition_token_budget": 32, "tier2_residuals": True,
                "tier2_investment": True,
            }
            (cache / "metadata.json").write_text(json.dumps({"feature_masks": invest_cache}))
            _require_cache_masks_match_model_config([cache], investment_model)  # match, no raise
            # Direction 1: cache investment-on, model investment-off -> fail.
            with self.assertRaisesRegex(ValueError, "mask-mismatched"):
                _require_cache_masks_match_model_config([cache], off_model)
            # Direction 2 (legacy asymmetry): a cache lacking the field is pre-flag, encoded
            # on the constant-zero column -> passes an investment-OFF train but REFUSES an
            # investment-ON train.
            legacy = {k: v for k, v in invest_cache.items() if k != "tier2_investment"}
            (cache / "metadata.json").write_text(json.dumps({"feature_masks": legacy}))
            _require_cache_masks_match_model_config([cache], off_model)  # pre-flag, passes off
            with self.assertRaisesRegex(ValueError, "mask-mismatched"):
                _require_cache_masks_match_model_config([cache], investment_model)

    def test_tier2_investment_collect_flag_resolves_explicit_masks(self) -> None:
        from pokezero.rollout_cli import _explicit_feature_masks_from_args, build_arg_parser

        # --tier2-investment alone triggers the explicit-masks path (was previously None).
        args = build_arg_parser().parse_args(
            ["collect-selfplay-training-cache", "--games", "1", "--out", "o", "--tier2-investment"]
        )
        masks = _explicit_feature_masks_from_args(args)
        self.assertIsNotNone(masks)
        self.assertTrue(masks.tier2_investment)
        self.assertTrue(masks.tier2_residuals)  # untouched sibling keeps its default-on
        # No flags at all -> None (adopt defaults / checkpoint), unchanged by this PR.
        no_flags = build_arg_parser().parse_args(
            ["collect-selfplay-training-cache", "--games", "1", "--out", "o"]
        )
        self.assertIsNone(_explicit_feature_masks_from_args(no_flags))

    def test_window_size_defaults_are_spec_v2_consistent(self) -> None:
        from pokezero.neural_cli import build_arg_parser as neural_parser
        from pokezero.rollout_cli import build_arg_parser as rollout_parser

        collect_args = rollout_parser().parse_args(
            ["collect-selfplay-training-cache", "--games", "1", "--out", "x"]
        )
        self.assertEqual(collect_args.window_size, 1)
        train_args = neural_parser().parse_args(["train", "--data", "d", "--out", "o"])
        self.assertEqual(train_args.window_size, 1)



class NumericCensusGuardTest(unittest.TestCase):
    """A 119-column artifact meeting 121-column code (or vice versa) must fail LOUDLY
    with the census named — never a downstream matmul error."""

    def test_numeric_width_mismatch_names_both_censuses(self) -> None:
        from types import SimpleNamespace

        from pokezero.neural_policy import _validate_tensor_shapes

        config = SimpleNamespace(
            window_size=1, token_count=4, categorical_feature_count=3, numeric_feature_count=121
        )
        def fake(shape):
            return SimpleNamespace(shape=shape)

        with self.assertRaisesRegex(ValueError, r"119.*121|121.*119"):
            _validate_tensor_shapes(
                fake((2, 1, 4, 3)),
                fake((2, 1, 4, 119)),  # pre-widening artifact width
                fake((2, 1, 4)),
                fake((2, 1, 4)),
                fake((2, 1)),
                config,
            )
        try:
            _validate_tensor_shapes(
                fake((2, 1, 4, 3)), fake((2, 1, 4, 119)), fake((2, 1, 4)), fake((2, 1, 4)), fake((2, 1)), config
            )
        except ValueError as error:
            self.assertIn("Tier-2", str(error))
            self.assertIn("must not be mixed", str(error))


if __name__ == "__main__":
    unittest.main()
