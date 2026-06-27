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
    "calculate_damage",
    "monte_carlo_tree_search",
)
POKE_ENGINE_REQUIRED_STATE_METHODS = (
    "apply_instructions",
    "reverse_instructions",
    "from_string",
    "to_string",
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
    import_error: str | None
    install_command: str = POKE_ENGINE_GEN3_INSTALL_COMMAND

    def message(self) -> str:
        if self.ready:
            version = self.version or "unknown"
            return f"poke-engine is available and exposes the expected Gen 3 spike API (version: {version})."

        parts = ["poke-engine is not ready for the Gen 3 spike."]
        if self.import_error:
            parts.append(f"Import error: {self.import_error}")
        if self.missing_api:
            parts.append("Missing API: " + ", ".join(self.missing_api))
        parts.append("Install/rebuild for Gen 3 with:")
        parts.append(f"  {self.install_command}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "ready": self.ready,
            "version": self.version,
            "missing_api": list(self.missing_api),
            "import_error": self.import_error,
            "install_command": self.install_command,
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
                import_error=str(exc),
            )
        return PokeEngineProbe(
            available=False,
            ready=False,
            version=None,
            missing_api=(),
            import_error=f"import failed while loading {POKE_ENGINE_IMPORT_NAME}: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive around native extension import failures.
        return PokeEngineProbe(
            available=False,
            ready=False,
            version=None,
            missing_api=(),
            import_error=f"{type(exc).__name__}: {exc}",
        )

    version = _lookup_version(version_lookup)
    missing_api = inspect_poke_engine_api(module)
    return PokeEngineProbe(
        available=True,
        ready=not missing_api,
        version=version,
        missing_api=missing_api,
        import_error=None,
    )


def require_poke_engine() -> ModuleType:
    """Import poke-engine or raise a targeted install/rebuild error."""

    probe = probe_poke_engine()
    if not probe.ready:
        raise PokeEngineUnavailableError(probe.message())
    return importlib.import_module(POKE_ENGINE_IMPORT_NAME)


def inspect_poke_engine_api(module: Any) -> tuple[str, ...]:
    """Return missing API names needed for the first Gen 3 spike."""

    missing: list[str] = []
    for attribute in POKE_ENGINE_REQUIRED_MODULE_ATTRIBUTES:
        if not hasattr(module, attribute):
            missing.append(attribute)

    state_type = getattr(module, "State", None)
    for method in POKE_ENGINE_REQUIRED_STATE_METHODS:
        if state_type is None or not hasattr(state_type, method):
            missing.append(f"State.{method}")

    return tuple(missing)


def _lookup_version(version_lookup: Callable[[str], str]) -> str | None:
    try:
        return version_lookup(POKE_ENGINE_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return None

