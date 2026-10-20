"""Real fresh-process Torch initialization; authored environment, no games."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


SCRIPT = textwrap.dedent("""
    import hashlib
    import os
    from pathlib import Path
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from pokezero import neural_policy
    from pokezero.mcts_eval.paper_reference import ReferenceRefusal
    from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory

    environment = Mock()
    def load(*args, **kwargs):
        # The real model loader reaches this same shared initialization path.
        neural_policy.require_torch()
        return SimpleNamespace()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "authored-checkpoint"
        checkpoint.write_bytes(b"not-a-model; loader is replaced")
        binding = ShowdownWorkerFactory(
            str(checkpoint), hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            directory, "authored-catalog",
        )
        with (
            patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=
                  SimpleNamespace(metadata=SimpleNamespace(source_hash="authored-catalog"))),
            patch("pokezero.neural_policy.load_transformer_policy", side_effect=load) as loader,
            patch("pokezero.mcts_eval.paper_reference_runtime.ChampionEvaluator"),
            patch("pokezero.collection.env_config_with_policy_spec_masks",
                  side_effect=lambda config, *args, **kwargs: config),
            patch("pokezero.local_showdown.LocalShowdownEnv", return_value=environment),
        ):
            if os.environ.get("TEST_EXPECT_THREAD_REFUSAL") == "1":
                try:
                    binding(0)
                except ReferenceRefusal as error:
                    assert "single-thread" in str(error), str(error)
                else:
                    raise AssertionError("non-single-thread worker was accepted")
                loader.assert_not_called()
                environment.reset.assert_not_called()
            else:
                runtime = binding(0)
                loader.assert_called_once()
                environment.reset.assert_called_once_with(seed=2026100400)
                torch = neural_policy.require_torch()
                assert torch.get_num_threads() == 1
                assert torch.get_num_interop_threads() == 1
                runtime.close()
                environment.close.assert_called_once()
""")


@unittest.skipUnless(importlib.util.find_spec("torch"), "real Torch dependency required")
class WorkerThreadInitializationTests(unittest.TestCase):
    def run_child(self, setting, *, refuse=False):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[name] = "1"
        for name in ("POKEZERO_TORCH_NUM_THREADS", "POKEZERO_TORCH_NUM_INTEROP_THREADS"):
            if setting is None:
                environment.pop(name, None)
            else:
                environment[name] = setting
        environment["TEST_EXPECT_THREAD_REFUSAL"] = "1" if refuse else "0"
        result = subprocess.run(
            [sys.executable, "-B", "-c", SCRIPT], env=environment,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_exact_inherited_environment_and_default_initialization(self):
        for setting in ("1", None, ""):
            with self.subTest(setting=setting):
                self.run_child(setting)

    def test_explicit_non_single_thread_setting_refuses_before_model_or_bridge(self):
        self.run_child("2", refuse=True)
