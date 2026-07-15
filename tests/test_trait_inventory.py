"""Regression tests for the parameterized checkpoint-trait inventory script."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trait_inventory.py"


class TraitInventoryTest(unittest.TestCase):
    def run_inventory(self, environment: dict[str, str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            cwd=cwd,
        )

    def test_requires_explicit_storage_configuration(self) -> None:
        environment = dict(os.environ)
        environment.pop("POKEZERO_SHARED_ROOT", None)
        environment.pop("POKEZERO_TRAIT_INVENTORY_OUT", None)

        result = self.run_inventory(environment)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("set POKEZERO_SHARED_ROOT", result.stderr)

    def test_requires_an_explicit_inventory_output_path(self) -> None:
        environment = dict(os.environ)
        environment["POKEZERO_SHARED_ROOT"] = "."
        environment.pop("POKEZERO_TRAIT_INVENTORY_OUT", None)

        result = self.run_inventory(environment)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("set POKEZERO_TRAIT_INVENTORY_OUT", result.stderr)

    def test_writes_a_filename_only_configured_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = dict(os.environ)
            environment["POKEZERO_SHARED_ROOT"] = str(root)
            environment["POKEZERO_TRAIT_INVENTORY_OUT"] = "inventory.json"

            result = self.run_inventory(environment, cwd=root)

            self.assertEqual(result.returncode, 0, result.stderr)
            inventory = json.loads((root / "inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory["schema"], "pokezero.trait_inventory.v1")
