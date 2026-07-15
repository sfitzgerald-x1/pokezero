"""Regression tests for the parameterized checkpoint-trait inventory script."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trait_inventory.py"


class TraitInventoryTest(unittest.TestCase):
    def test_requires_explicit_storage_configuration(self) -> None:
        environment = dict(os.environ)
        environment.pop("POKEZERO_SHARED_ROOT", None)
        environment.pop("POKEZERO_TRAIT_INVENTORY_OUT", None)

        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("set POKEZERO_SHARED_ROOT", result.stderr)
