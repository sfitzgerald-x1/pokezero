from __future__ import annotations

import contextlib
import io
import json
from importlib import metadata
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.engine_cli import ENGINE_NOT_READY_EXIT_CODE, main as engine_cli_main
from pokezero.poke_engine_backend import (
    POKE_ENGINE_GEN3_INSTALL_COMMAND,
    PokeEngineProbe,
    PokeEngineUnavailableError,
    inspect_poke_engine_api,
    inspect_poke_engine_optional_api,
    probe_poke_engine,
    require_poke_engine,
)


class FakeState:
    @classmethod
    def from_string(cls, value: str) -> "FakeState":
        return cls()

    def to_string(self) -> str:
        return ""

    def apply_instructions(self, instructions: object) -> "FakeState":
        return self

    def reverse_instructions(self, instructions: object) -> "FakeState":
        return self


def fake_engine_module() -> ModuleType:
    module = ModuleType("poke_engine")
    module.State = FakeState
    module.generate_instructions = lambda *args, **kwargs: []
    module.calculate_damage = lambda *args, **kwargs: ([], [])
    module.monte_carlo_tree_search = lambda *args, **kwargs: None
    return module


class PokeEngineBackendTest(unittest.TestCase):
    def test_install_command_is_gen3_specific(self) -> None:
        self.assertIn("poke-engine==0.0.47", POKE_ENGINE_GEN3_INSTALL_COMMAND)
        self.assertIn("poke-engine/gen3", POKE_ENGINE_GEN3_INSTALL_COMMAND)
        self.assertIn("--no-default-features", POKE_ENGINE_GEN3_INSTALL_COMMAND)

    def test_inspect_api_accepts_expected_reversible_surface(self) -> None:
        self.assertEqual(inspect_poke_engine_api(fake_engine_module()), ())

    def test_optional_helpers_are_reported_separately(self) -> None:
        module = SimpleNamespace(
            State=FakeState,
            generate_instructions=lambda *args, **kwargs: [],
        )

        self.assertEqual(inspect_poke_engine_api(module), ())
        self.assertEqual(
            inspect_poke_engine_optional_api(module),
            ("calculate_damage", "monte_carlo_tree_search"),
        )

    def test_inspect_api_reports_missing_state_methods(self) -> None:
        module = SimpleNamespace(
            State=object,
            generate_instructions=lambda *args, **kwargs: [],
            calculate_damage=lambda *args, **kwargs: ([], []),
            monte_carlo_tree_search=lambda *args, **kwargs: None,
        )

        self.assertEqual(
            inspect_poke_engine_api(module),
            (
                "State.apply_instructions",
                "State.reverse_instructions",
                "State.from_string",
                "State.to_string",
            ),
        )

    def test_probe_reports_ready_fake_engine(self) -> None:
        probe = probe_poke_engine(
            importer=lambda name: fake_engine_module(),
            version_lookup=lambda name: "0.0.47",
        )

        self.assertTrue(probe.available)
        self.assertTrue(probe.ready)
        self.assertEqual(probe.version, "0.0.47")
        self.assertEqual(probe.missing_api, ())
        self.assertEqual(probe.missing_optional_api, ())
        self.assertIn("not full Gen 3 mechanics equivalence", probe.message())

    def test_probe_reports_missing_import_with_install_command(self) -> None:
        def missing_import(name: str) -> ModuleType:
            raise ModuleNotFoundError("No module named 'poke_engine'", name=name)

        probe = probe_poke_engine(importer=missing_import)

        self.assertFalse(probe.available)
        self.assertFalse(probe.ready)
        self.assertIn("poke-engine", probe.message())
        self.assertIn("poke-engine/gen3", probe.message())

    def test_probe_allows_missing_distribution_metadata(self) -> None:
        def missing_version(name: str) -> str:
            raise metadata.PackageNotFoundError(name)

        probe = probe_poke_engine(
            importer=lambda name: fake_engine_module(),
            version_lookup=missing_version,
        )

        self.assertTrue(probe.ready)
        self.assertIsNone(probe.version)

    def test_require_poke_engine_raises_targeted_error_when_missing(self) -> None:
        probe = PokeEngineProbe(
            available=False,
            ready=False,
            version=None,
            missing_api=(),
            missing_optional_api=(),
            import_error="No module named 'poke_engine'",
        )
        with patch("pokezero.poke_engine_backend.probe_poke_engine", return_value=probe):
            with self.assertRaises(PokeEngineUnavailableError) as context:
                require_poke_engine()

        self.assertIn("Install/rebuild the recommended Gen 3 wheel", str(context.exception))

    def test_doctor_cli_reports_missing_engine_as_not_ready(self) -> None:
        probe = PokeEngineProbe(
            available=False,
            ready=False,
            version=None,
            missing_api=(),
            missing_optional_api=(),
            import_error="No module named 'poke_engine'",
        )
        with patch("pokezero.engine_cli.probe_poke_engine", return_value=probe):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = engine_cli_main(["doctor", "--json"])

        self.assertEqual(exit_code, ENGINE_NOT_READY_EXIT_CODE)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["ready"])
        self.assertIn("poke-engine/gen3", payload["install_command"])
        self.assertFalse(payload["gen3_feature_verified"])

    def test_installed_poke_engine_contract_when_available(self) -> None:
        probe = probe_poke_engine()
        if not probe.available:
            self.skipTest("poke-engine is not installed")

        self.assertTrue(probe.ready, probe.message())


if __name__ == "__main__":
    unittest.main()
