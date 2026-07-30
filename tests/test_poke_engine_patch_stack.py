"""Integration pin for the frozen poke-engine source and patch ordering.

The patch stack is a source-level build input. This test downloads the pinned
sdist, verifies its digest, and applies the complete ordered stack at fuzz=0.
The separate engine behavior pins verify the zero-heal markers this stack adds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apply_poke_engine_patches as patch_stack  # noqa: E402
import verify_poke_engine_source as source_verifier  # noqa: E402


class PokeEnginePatchStackTests(unittest.TestCase):
    def _download_archive(self, directory: Path) -> Path:
        supplied = os.environ.get("POKEZERO_POKE_ENGINE_SDIST")
        if supplied:
            archive = Path(supplied)
            self.assertTrue(archive.is_file(), f"missing supplied sdist: {archive}")
            return archive

        uv = shutil.which("uv")
        if not uv:
            self.skipTest("uv is required to fetch the pinned poke-engine sdist")
        subprocess.run(
            [
                uv,
                "run",
                "--isolated",
                "--python",
                "3.11",
                "pip",
                "download",
                "poke-engine==0.0.47",
                "--no-deps",
                "--no-binary",
                ":all:",
                "-d",
                str(directory),
            ],
            cwd=ROOT,
            check=True,
        )
        return directory / "poke_engine-0.0.47.tar.gz"

    def test_pinned_stack_applies_without_fuzz_and_keeps_public_noop_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            archive = self._download_archive(directory)
            source_verifier.verify(archive, expected_version="0.0.47")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(directory, filter="data")
            source = directory / "poke_engine-0.0.47"
            applied = patch_stack.apply_patch_stack(source)

            self.assertEqual(applied, patch_stack.patch_names())
            self.assertEqual(applied[-5], "poke-engine-gen3-public-noop-branches.patch")
            generated = (source / "src/gen3/generate_instructions.rs").read_text(
                encoding="utf-8"
            )
            self.assertIn("let original_accuracy = choice.accuracy;", generated)
            self.assertIn("heal_amount: 0", generated)
            self.assertIn("blocked_by_protect && incoming_instructions.percentage != 0.0", generated)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
