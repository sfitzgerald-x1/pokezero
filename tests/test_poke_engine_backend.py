from __future__ import annotations

import contextlib
import io
import json
from importlib import metadata
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.engine_cli import (
    ENGINE_NOT_READY_EXIT_CODE,
    SMOKE_FAILED_EXIT_CODE,
    main as engine_cli_main,
)
from pokezero.poke_engine_backend import (
    POKE_ENGINE_GEN3_INSTALL_COMMAND,
    PokeEngineProbe,
    PokeEngineReversibleSmokeResult,
    PokeEngineUnavailableError,
    inspect_poke_engine_api,
    inspect_poke_engine_optional_api,
    probe_poke_engine,
    require_poke_engine,
    run_poke_engine_reversible_smoke,
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


def make_smoke_engine(
    instructions: list,
    *,
    broken_reverse: bool = False,
    broken_from_string: bool = False,
) -> ModuleType:
    """Build a deterministic fake poke-engine module for the reversible smoke.

    Each instruction carries a ``delta`` string. ``apply_instructions`` appends
    it to the serialized state (so a non-empty delta means "this branch mutated")
    and ``reverse_instructions`` strips it back off. ``broken_reverse`` simulates
    a non-reversible branch by failing to restore the original serialized state.
    """

    class SmokeState:
        def __init__(self, serialized: str = "ORIG", **_kwargs: object) -> None:
            self.serialized = serialized

        @classmethod
        def from_string(cls, value: str) -> "SmokeState":
            if broken_from_string:
                return cls(value + "_BROKEN")
            return cls(value)

        def to_string(self) -> str:
            return self.serialized

        def apply_instructions(self, instruction: object) -> "SmokeState":
            return SmokeState(self.serialized + instruction.delta)

        def reverse_instructions(self, instruction: object) -> "SmokeState":
            if broken_reverse:
                return SmokeState(self.serialized + "_BROKEN")
            delta = instruction.delta
            if delta and self.serialized.endswith(delta):
                return SmokeState(self.serialized[: -len(delta)])
            return SmokeState(self.serialized)

    module = ModuleType("poke_engine")
    module.State = SmokeState
    module.Move = lambda **kwargs: SimpleNamespace(**kwargs)
    module.Pokemon = lambda **kwargs: SimpleNamespace(**kwargs)
    module.Side = lambda **kwargs: SimpleNamespace(**kwargs)
    module.generate_instructions = lambda *args, **kwargs: list(instructions)
    return module


def smoke_instruction(percentage: float, delta: str) -> SimpleNamespace:
    return SimpleNamespace(percentage=percentage, delta=delta)


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


class PokeEngineReversibleSmokeTest(unittest.TestCase):
    def test_smoke_passes_when_branches_mutate_and_reverse(self) -> None:
        engine = make_smoke_engine(
            [
                smoke_instruction(50.0, "A"),
                smoke_instruction(30.0, "B"),
                smoke_instruction(20.0, ""),
            ]
        )

        result = run_poke_engine_reversible_smoke(module=engine)

        self.assertEqual(result.instruction_count, 3)
        self.assertEqual(result.checked_instruction_count, 3)
        self.assertTrue(result.mutated_any_state)
        self.assertTrue(result.round_trip_ok)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.instruction_percentages, (50.0, 30.0, 20.0))

    def test_smoke_not_successful_when_no_branch_mutates(self) -> None:
        # A vacuous round-trip (nothing changes) must not be reported as a win.
        engine = make_smoke_engine([smoke_instruction(100.0, "")])

        result = run_poke_engine_reversible_smoke(module=engine)

        self.assertFalse(result.mutated_any_state)
        self.assertTrue(result.round_trip_ok)
        self.assertFalse(result.succeeded)

    def test_smoke_detects_broken_reverse(self) -> None:
        engine = make_smoke_engine(
            [smoke_instruction(50.0, "A"), smoke_instruction(50.0, "B")],
            broken_reverse=True,
        )

        result = run_poke_engine_reversible_smoke(module=engine)

        self.assertTrue(result.mutated_any_state)
        self.assertFalse(result.round_trip_ok)
        self.assertFalse(result.succeeded)
        self.assertEqual(result.checked_instruction_count, 1)
        self.assertEqual(len(result.instruction_percentages), result.checked_instruction_count)

    def test_smoke_raises_when_serialization_does_not_round_trip(self) -> None:
        engine = make_smoke_engine(
            [smoke_instruction(100.0, "A")],
            broken_from_string=True,
        )

        with self.assertRaises(PokeEngineUnavailableError) as context:
            run_poke_engine_reversible_smoke(module=engine)

        self.assertIn("State.from_string", str(context.exception))

    def test_smoke_respects_max_instruction_checks(self) -> None:
        engine = make_smoke_engine(
            [smoke_instruction(float(i), f"D{i}") for i in range(10)]
        )

        result = run_poke_engine_reversible_smoke(module=engine, max_instruction_checks=3)

        self.assertEqual(result.instruction_count, 10)
        self.assertEqual(result.checked_instruction_count, 3)
        self.assertEqual(result.instruction_percentages, (0.0, 1.0, 2.0))
        self.assertTrue(result.succeeded)

    def test_smoke_raises_when_no_instructions(self) -> None:
        engine = make_smoke_engine([])

        with self.assertRaises(PokeEngineUnavailableError) as context:
            run_poke_engine_reversible_smoke(module=engine)

        self.assertIn("no instructions", str(context.exception))

    def test_smoke_raises_when_smoke_api_missing(self) -> None:
        engine = make_smoke_engine([smoke_instruction(100.0, "A")])
        del engine.Move

        with self.assertRaises(PokeEngineUnavailableError) as context:
            run_poke_engine_reversible_smoke(module=engine)

        self.assertIn("Missing smoke-test API", str(context.exception))
        self.assertIn("Move", str(context.exception))

    def test_result_summary_and_dict_round_trip(self) -> None:
        result = PokeEngineReversibleSmokeResult(
            instruction_count=13,
            checked_instruction_count=8,
            mutated_any_state=True,
            round_trip_ok=True,
            instruction_percentages=(1.0, 2.0),
        )

        self.assertIn("PASS", result.summary())
        payload = result.to_dict()
        self.assertTrue(payload["succeeded"])
        self.assertEqual(payload["instruction_count"], 13)
        self.assertEqual(payload["instruction_percentages"], [1.0, 2.0])

    def test_real_reversible_smoke_when_installed(self) -> None:
        probe = probe_poke_engine()
        if not probe.ready:
            self.skipTest("poke-engine is not installed/ready")

        result = run_poke_engine_reversible_smoke()

        self.assertGreater(result.instruction_count, 0)
        self.assertGreater(result.checked_instruction_count, 0)
        self.assertTrue(result.mutated_any_state, "expected at least one branch to change state")
        self.assertTrue(result.round_trip_ok, "expected every checked branch to reverse cleanly")
        self.assertTrue(result.succeeded)


def _ready_probe() -> PokeEngineProbe:
    return PokeEngineProbe(
        available=True,
        ready=True,
        version="0.0.47",
        missing_api=(),
        missing_optional_api=(),
        import_error=None,
    )


def _not_ready_probe() -> PokeEngineProbe:
    return PokeEngineProbe(
        available=False,
        ready=False,
        version=None,
        missing_api=(),
        missing_optional_api=(),
        import_error="No module named 'poke_engine'",
    )


def _passing_smoke() -> PokeEngineReversibleSmokeResult:
    return PokeEngineReversibleSmokeResult(
        instruction_count=13,
        checked_instruction_count=8,
        mutated_any_state=True,
        round_trip_ok=True,
        instruction_percentages=(50.0, 25.0),
    )


def _failing_smoke() -> PokeEngineReversibleSmokeResult:
    return PokeEngineReversibleSmokeResult(
        instruction_count=13,
        checked_instruction_count=2,
        mutated_any_state=True,
        round_trip_ok=False,
        instruction_percentages=(50.0, 25.0),
    )


class PokeEngineDoctorSmokeCliTest(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = engine_cli_main(argv)
        return exit_code, output.getvalue()

    def test_default_doctor_does_not_run_smoke(self) -> None:
        with patch("pokezero.engine_cli.probe_poke_engine", return_value=_ready_probe()):
            with patch("pokezero.engine_cli.run_poke_engine_reversible_smoke") as smoke:
                exit_code, out = self._run_cli(["doctor", "--json"])

        smoke.assert_not_called()
        self.assertEqual(exit_code, 0)
        self.assertNotIn("smoke", json.loads(out))

    def test_smoke_runs_when_ready_and_passes(self) -> None:
        with patch("pokezero.engine_cli.probe_poke_engine", return_value=_ready_probe()):
            with patch(
                "pokezero.engine_cli.run_poke_engine_reversible_smoke",
                return_value=_passing_smoke(),
            ) as smoke:
                exit_code, out = self._run_cli(["doctor", "--smoke", "--json"])

        smoke.assert_called_once()
        self.assertEqual(exit_code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["smoke"]["ran"])
        self.assertTrue(payload["smoke"]["succeeded"])

    def test_smoke_skipped_when_not_ready(self) -> None:
        with patch("pokezero.engine_cli.probe_poke_engine", return_value=_not_ready_probe()):
            with patch("pokezero.engine_cli.run_poke_engine_reversible_smoke") as smoke:
                exit_code, out = self._run_cli(["doctor", "--smoke", "--json"])

        smoke.assert_not_called()
        self.assertEqual(exit_code, ENGINE_NOT_READY_EXIT_CODE)
        payload = json.loads(out)
        self.assertFalse(payload["smoke"]["ran"])
        self.assertIn("not ready", payload["smoke"]["reason"])

    def test_smoke_failure_exits_nonzero(self) -> None:
        with patch("pokezero.engine_cli.probe_poke_engine", return_value=_ready_probe()):
            with patch(
                "pokezero.engine_cli.run_poke_engine_reversible_smoke",
                return_value=_failing_smoke(),
            ):
                exit_code, out = self._run_cli(["doctor", "--smoke", "--json"])

        self.assertEqual(exit_code, SMOKE_FAILED_EXIT_CODE)
        payload = json.loads(out)
        self.assertTrue(payload["smoke"]["ran"])
        self.assertFalse(payload["smoke"]["succeeded"])

    def test_smoke_engine_error_exits_nonzero(self) -> None:
        with patch("pokezero.engine_cli.probe_poke_engine", return_value=_ready_probe()):
            with patch(
                "pokezero.engine_cli.run_poke_engine_reversible_smoke",
                side_effect=PokeEngineUnavailableError("boom"),
            ):
                exit_code, out = self._run_cli(["doctor", "--smoke", "--json"])

        self.assertEqual(exit_code, SMOKE_FAILED_EXIT_CODE)
        payload = json.loads(out)
        self.assertTrue(payload["smoke"]["ran"])
        self.assertFalse(payload["smoke"]["succeeded"])
        self.assertIn("boom", payload["smoke"]["reason"])

    def test_smoke_unexpected_engine_error_exits_nonzero(self) -> None:
        with patch("pokezero.engine_cli.probe_poke_engine", return_value=_ready_probe()):
            with patch(
                "pokezero.engine_cli.run_poke_engine_reversible_smoke",
                side_effect=RuntimeError("native crash"),
            ):
                exit_code, out = self._run_cli(["doctor", "--smoke", "--json"])

        self.assertEqual(exit_code, SMOKE_FAILED_EXIT_CODE)
        payload = json.loads(out)
        self.assertTrue(payload["smoke"]["ran"])
        self.assertFalse(payload["smoke"]["succeeded"])
        self.assertIn("RuntimeError", payload["smoke"]["reason"])

    def test_smoke_text_output_prints_summary(self) -> None:
        with patch("pokezero.engine_cli.probe_poke_engine", return_value=_ready_probe()):
            with patch(
                "pokezero.engine_cli.run_poke_engine_reversible_smoke",
                return_value=_passing_smoke(),
            ):
                exit_code, out = self._run_cli(["doctor", "--smoke"])

        self.assertEqual(exit_code, 0)
        self.assertIn("reversible smoke PASS", out)


if __name__ == "__main__":
    unittest.main()
