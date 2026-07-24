"""Safe filesystem persistence for canonical scenario JSON."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .domain import EndgameScenario, ScenarioValidationError


_SLUG_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")


class ScenarioRepository:
    """A small atomic JSON repository rooted at one explicitly configured directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                scenario = self.load(path.stem)
            except (OSError, ValueError, ScenarioValidationError):
                records.append({"slug": path.stem, "invalid": True})
                continue
            records.append(
                {
                    "slug": path.stem,
                    "scenario_id": scenario.scenario_id,
                    "title": scenario.title,
                    "tags": list(scenario.tags),
                    "source_hash": scenario.randbat_source_hash,
                    "replay_proven": scenario.replay_proven,
                    "modified_at": path.stat().st_mtime,
                }
            )
        return records

    def load(self, slug: str) -> EndgameScenario:
        path = self._path_for_slug(slug)
        if not path.exists():
            raise FileNotFoundError(f"Scenario {slug!r} does not exist.")
        if path.is_symlink() or not path.is_file():
            raise ScenarioValidationError("scenario path must be a regular file")
        return EndgameScenario.from_payload_json(path.read_text(encoding="utf-8"))

    def save(self, slug: str, scenario: EndgameScenario) -> Path:
        path = self._path_for_slug(slug)
        if path.exists() and path.is_symlink():
            raise ScenarioValidationError("refusing to overwrite a symlinked scenario")
        encoded = scenario.canonical_json()
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{slug}-", suffix=".tmp", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return path

    def delete(self, slug: str) -> bool:
        path = self._path_for_slug(slug)
        if not path.exists():
            return False
        if path.is_symlink() or not path.is_file():
            raise ScenarioValidationError("scenario path must be a regular file")
        path.unlink()
        return True

    def _path_for_slug(self, slug: str) -> Path:
        if not isinstance(slug, str) or not _SLUG_RE.fullmatch(slug):
            raise ScenarioValidationError("slug must use lowercase letters, digits, and single hyphens")
        path = (self.root / f"{slug}.json").resolve(strict=False)
        if path.parent != self.root:
            raise ScenarioValidationError("slug escapes the scenario directory")
        return path
