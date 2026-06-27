"""Optional poke-engine integration checks.

The core package intentionally does not depend on the Rust-backed
``poke-engine`` wheel. This module is the narrow seam used by experiments that
want to validate or adopt poke-engine without making normal self-play imports
pay that dependency cost.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from importlib import metadata
from types import ModuleType
from typing import Any, Callable


POKE_ENGINE_IMPORT_NAME = "poke_engine"
POKE_ENGINE_PACKAGE_NAME = "poke-engine"
POKE_ENGINE_SUPPORTED_VERSION = "0.0.47"
POKE_ENGINE_GEN3_INSTALL_COMMAND = (
    "python -m pip uninstall -y poke-engine && "
    "python -m pip install -v --force-reinstall --no-cache-dir "
    f"{POKE_ENGINE_PACKAGE_NAME}=={POKE_ENGINE_SUPPORTED_VERSION} "
    '--config-settings="build-args=--features poke-engine/gen3 --no-default-features"'
)
POKE_ENGINE_REQUIRED_MODULE_ATTRIBUTES = (
    "State",
    "generate_instructions",
)
POKE_ENGINE_OPTIONAL_MODULE_ATTRIBUTES = (
    "calculate_damage",
    "monte_carlo_tree_search",
)
POKE_ENGINE_SMOKE_MODULE_ATTRIBUTES = (
    "Move",
    "Pokemon",
    "Side",
)
POKE_ENGINE_REQUIRED_STATE_METHODS = (
    "apply_instructions",
    "reverse_instructions",
    "from_string",
    "to_string",
)
POKE_ENGINE_GEN3_FEATURE_NOTE = (
    "This probe verifies the Python reversible API seam, not full Gen 3 mechanics equivalence. "
    "Use Showdown equivalence fixtures before adopting poke-engine for training/search."
)


class PokeEngineUnavailableError(RuntimeError):
    """Raised when optional poke-engine functionality is requested but unavailable."""


@dataclass(frozen=True)
class PokeEngineProbe:
    """Probe result for the optional poke-engine backend."""

    available: bool
    ready: bool
    version: str | None
    missing_api: tuple[str, ...]
    missing_optional_api: tuple[str, ...]
    import_error: str | None
    install_command: str = POKE_ENGINE_GEN3_INSTALL_COMMAND

    def message(self) -> str:
        if self.ready:
            version = self.version or "unknown"
            parts = [
                f"poke-engine is available and exposes the expected reversible API seam (version: {version}).",
                POKE_ENGINE_GEN3_FEATURE_NOTE,
            ]
            if self.missing_optional_api:
                parts.append("Optional API missing: " + ", ".join(self.missing_optional_api))
            return "\n".join(parts)

        parts = ["poke-engine is not ready for the reversible API spike."]
        if self.import_error:
            parts.append(f"Import error: {self.import_error}")
        if self.missing_api:
            parts.append("Missing API: " + ", ".join(self.missing_api))
        if self.missing_optional_api:
            parts.append("Optional API missing: " + ", ".join(self.missing_optional_api))
        parts.append("Install/rebuild the recommended Gen 3 wheel with:")
        parts.append(f"  {self.install_command}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "ready": self.ready,
            "version": self.version,
            "missing_api": list(self.missing_api),
            "missing_optional_api": list(self.missing_optional_api),
            "import_error": self.import_error,
            "install_command": self.install_command,
            "gen3_feature_verified": False,
            "gen3_feature_note": POKE_ENGINE_GEN3_FEATURE_NOTE,
        }


@dataclass(frozen=True)
class PokeEngineReversibleSmokeResult:
    """Result from a minimal real-engine apply/reverse smoke check."""

    instruction_count: int
    checked_instruction_count: int
    mutated_any_state: bool
    round_trip_ok: bool
    instruction_percentages: tuple[float, ...]

    @property
    def succeeded(self) -> bool:
        """A smoke run is only meaningful if it reversed cleanly AND changed something.

        ``round_trip_ok`` on its own can pass vacuously when no checked branch
        mutates the serialized state, so the spike requires at least one real
        mutation before it counts the reversible seam as exercised.
        """

        return (
            self.checked_instruction_count > 0
            and self.mutated_any_state
            and self.round_trip_ok
        )

    def summary(self) -> str:
        status = "PASS" if self.succeeded else "FAIL"
        return (
            f"reversible smoke {status}: "
            f"{self.checked_instruction_count}/{self.instruction_count} branches checked, "
            f"mutated_any_state={self.mutated_any_state}, round_trip_ok={self.round_trip_ok}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_count": self.instruction_count,
            "checked_instruction_count": self.checked_instruction_count,
            "mutated_any_state": self.mutated_any_state,
            "round_trip_ok": self.round_trip_ok,
            "succeeded": self.succeeded,
            "instruction_percentages": list(self.instruction_percentages),
        }


def probe_poke_engine(
    *,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    version_lookup: Callable[[str], str] = metadata.version,
) -> PokeEngineProbe:
    """Check whether poke-engine can be imported and has the expected API."""

    try:
        module = importer(POKE_ENGINE_IMPORT_NAME)
    except ModuleNotFoundError as exc:
        if exc.name == POKE_ENGINE_IMPORT_NAME:
            return PokeEngineProbe(
                available=False,
                ready=False,
                version=None,
                missing_api=(),
                missing_optional_api=(),
                import_error=str(exc),
            )
        return PokeEngineProbe(
            available=False,
            ready=False,
            version=None,
            missing_api=(),
            missing_optional_api=(),
            import_error=f"import failed while loading {POKE_ENGINE_IMPORT_NAME}: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive around native extension import failures.
        return PokeEngineProbe(
            available=False,
            ready=False,
            version=None,
            missing_api=(),
            missing_optional_api=(),
            import_error=f"{type(exc).__name__}: {exc}",
        )

    version = _lookup_version(version_lookup)
    missing_api = inspect_poke_engine_api(module)
    missing_optional_api = inspect_poke_engine_optional_api(module)
    return PokeEngineProbe(
        available=True,
        ready=not missing_api,
        version=version,
        missing_api=missing_api,
        missing_optional_api=missing_optional_api,
        import_error=None,
    )


def require_poke_engine() -> ModuleType:
    """Import poke-engine or raise a targeted install/rebuild error."""

    probe = probe_poke_engine()
    if not probe.ready:
        raise PokeEngineUnavailableError(probe.message())
    return importlib.import_module(POKE_ENGINE_IMPORT_NAME)


def inspect_poke_engine_api(module: Any) -> tuple[str, ...]:
    """Return missing required API names needed for reversible-state experiments."""

    missing: list[str] = []
    for attribute in POKE_ENGINE_REQUIRED_MODULE_ATTRIBUTES:
        if not hasattr(module, attribute):
            missing.append(attribute)

    state_type = getattr(module, "State", None)
    for method in POKE_ENGINE_REQUIRED_STATE_METHODS:
        if state_type is None or not hasattr(state_type, method):
            missing.append(f"State.{method}")

    return tuple(missing)


def inspect_poke_engine_optional_api(module: Any) -> tuple[str, ...]:
    """Return missing optional helpers that are useful but not required for the first spike."""

    missing: list[str] = []
    for attribute in POKE_ENGINE_OPTIONAL_MODULE_ATTRIBUTES:
        if not hasattr(module, attribute):
            missing.append(attribute)
    return tuple(missing)


def run_poke_engine_reversible_smoke(
    *,
    module: Any | None = None,
    max_instruction_checks: int = 8,
) -> PokeEngineReversibleSmokeResult:
    """Run a minimal real-engine instruction apply/reverse smoke check.

    This intentionally does not compare against Showdown. It only verifies that
    a locally installed poke-engine wheel can construct a basic battle state,
    generate possible instructions, apply them, and reverse them back to the
    original serialized state.
    """

    engine = require_poke_engine() if module is None else module
    missing = tuple(attribute for attribute in POKE_ENGINE_SMOKE_MODULE_ATTRIBUTES if not hasattr(engine, attribute))
    if missing:
        raise PokeEngineUnavailableError("Missing smoke-test API: " + ", ".join(missing))

    state = _basic_gen3_smoke_state(engine)
    return run_reversible_smoke_on_state(
        engine,
        state,
        "ember",
        "watergun",
        max_instruction_checks=max_instruction_checks,
    )


def run_reversible_smoke_on_state(
    engine: Any,
    state: Any,
    move_one: str,
    move_two: str,
    *,
    max_instruction_checks: int = 8,
) -> PokeEngineReversibleSmokeResult:
    """Generate instructions for ``state`` and verify apply/reverse round-trips.

    This is the shared core used both by :func:`run_poke_engine_reversible_smoke`
    (which builds its own minimal state) and by the fixture adapter, which builds
    a state from a curated :class:`~pokezero.poke_engine_adapter.BattleSpec`. It
    deliberately makes no Showdown comparison; it only proves the reversible seam.
    """

    original = state.to_string()
    if engine.State.from_string(original).to_string() != original:
        raise PokeEngineUnavailableError("poke-engine State.from_string did not round-trip the state")

    instructions = tuple(engine.generate_instructions(state, move_one, move_two))
    if not instructions:
        raise PokeEngineUnavailableError("poke-engine generated no instructions for the state")

    checked = instructions[:max_instruction_checks]
    mutated_any_state = False
    round_trip_ok = True
    percentages: list[float] = []
    for instruction in checked:
        percentages.append(float(getattr(instruction, "percentage", 0.0)))
        # ``apply_instructions``/``reverse_instructions`` return fresh states and
        # leave ``state`` untouched, so each branch is applied to the same original.
        after = state.apply_instructions(instruction)
        if after.to_string() != original:
            mutated_any_state = True
        restored = after.reverse_instructions(instruction)
        if restored.to_string() != original:
            round_trip_ok = False
            break

    return PokeEngineReversibleSmokeResult(
        instruction_count=len(instructions),
        checked_instruction_count=len(percentages),
        mutated_any_state=mutated_any_state,
        round_trip_ok=round_trip_ok,
        instruction_percentages=tuple(percentages),
    )


def _basic_gen3_smoke_state(engine: Any) -> Any:
    move = engine.Move
    pokemon = engine.Pokemon
    side = engine.Side
    state = engine.State
    return state(
        side_one=side(
            pokemon=[
                pokemon(
                    id="charmander",
                    level=100,
                    types=("fire", "typeless"),
                    hp=100,
                    maxhp=100,
                    attack=100,
                    defense=100,
                    special_attack=100,
                    special_defense=100,
                    speed=100,
                    status="none",
                    moves=[move(id="ember", pp=32), move(id="tackle", pp=32)],
                )
            ]
        ),
        side_two=side(
            pokemon=[
                pokemon(
                    id="squirtle",
                    level=100,
                    types=("water", "typeless"),
                    hp=100,
                    maxhp=100,
                    attack=100,
                    defense=100,
                    special_attack=100,
                    special_defense=100,
                    speed=100,
                    status="none",
                    moves=[move(id="watergun", pp=32), move(id="tackle", pp=32)],
                )
            ]
        ),
    )


def _lookup_version(version_lookup: Callable[[str], str]) -> str | None:
    try:
        return version_lookup(POKE_ENGINE_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return None
