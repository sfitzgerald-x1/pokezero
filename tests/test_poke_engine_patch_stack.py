"""Integration pin for the frozen poke-engine source and patch ordering.

The patch stack is a source-level build input. This test downloads the pinned
sdist, verifies its digest, and applies the complete ordered stack at fuzz=0.
The separate engine behavior pins verify the zero-heal markers this stack adds.
"""

from __future__ import annotations

import os
import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apply_poke_engine_patches as patch_stack  # noqa: E402
import verify_poke_engine_source as source_verifier  # noqa: E402


# Post-patch content pins for the 53-patch stack. Only generate_instructions.rs
# moved: crit-kill-split and substitute-hp-gate both touch it. choice_effects.rs
# and abilities.rs are BOTH byte-identical to the 52-patch pins -- the dropped
# trick-attacker-item patch was the only thing that would have moved
# choice_effects.rs, and it is gone (review: it encoded a rule that is not gen3;
# Showdown succeeds where it forced a fail). Two unchanged digests either side of
# one changed digest is what makes the update a measurement rather than a paste:
# drift in the vendored source would have moved all three.
EXPECTED_FINAL_SHA256 = {
    "src/gen3/generate_instructions.rs": "a83419fba666545de3d26aaefde8b0c00680537f0f88cf91b70f16bde16662ff",
    "src/gen3/abilities.rs": "5bd46cc2517588fa380182e3e0c0d42676a596a90160735050beb3e5ab382294",
    "src/gen3/choice_effects.rs": "88101a4e475b7f9a99e3780dde56b39c9dcc6eb66a9458d516fa468ba8a13dc5",
}


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

    def _extract_source(self, directory: Path) -> Path:
        archive = self._download_archive(directory)
        source_verifier.verify(archive, expected_version="0.0.47")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(directory, filter="data")
        return directory / "poke_engine-0.0.47"

    def test_pinned_stack_applies_without_fuzz_and_keeps_public_noop_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = self._extract_source(directory)
            applied = patch_stack.apply_patch_stack(source)

            self.assertEqual([entry.name for entry in applied], patch_stack.patch_names())
            # Pinned by POSITION of the two zero-context patches rather than by
            # a hardcoded stack length. The stack grows; it went 52 -> 53 in this
            # branch and the old `applied[:46]` / `applied[46:]` split silently
            # became wrong, which is how a green guard on "the new patches apply
            # cleanly to a fresh upstream sdist" went red without anyone noticing.
            fallbacks = [
                index
                for index, entry in enumerate(applied)
                if entry.backend == "patch-fallback"
            ]
            self.assertEqual(len(fallbacks), 2, applied)
            self.assertEqual(fallbacks[1], fallbacks[0] + 1, "the two are adjacent")
            self.assertTrue(
                all(entry.backend == "git-apply" for entry in applied[: fallbacks[0]]),
                applied,
            )
            self.assertTrue(
                all(
                    entry.backend == "git-apply" for entry in applied[fallbacks[1] + 1 :]
                ),
                # Everything after the two zero-context patches, including the
                # Toxic-stage cap and both patches this branch adds, must keep
                # applying cleanly through git without fuzz.
                applied,
            )
            # Tail pin. Grown, not dropped: the stack is append-only at the end
            # and the order matters, so a new patch has to be recorded here
            # deliberately rather than sliding in under a length-agnostic check.
            self.assertEqual(
                [entry.name for entry in applied[-2:]],
                [
                    "poke-engine-gen3-substitute-hp-gate.patch",
                    "poke-engine-gen3-confusion-snapout-timing.patch",
                ],
            )
            # The dropped Trick patch must stay gone: no file, no registration.
            self.assertNotIn(
                "poke-engine-gen3-trick-attacker-item.patch",
                [entry.name for entry in applied],
            )
            self.assertIn(
                "poke-engine-gen3-public-noop-branches.patch",
                [entry.name for entry in applied],
            )
            generated = (source / "src/gen3/generate_instructions.rs").read_text(
                encoding="utf-8"
            )
            self.assertIn("let original_accuracy = choice.accuracy;", generated)
            self.assertIn("heal_amount: 0", generated)
            self.assertIn("blocked_by_protect && incoming_instructions.percentage != 0.0", generated)
            for relative_path, expected_sha256 in EXPECTED_FINAL_SHA256.items():
                actual_sha256 = hashlib.sha256((source / relative_path).read_bytes()).hexdigest()
                self.assertEqual(actual_sha256, expected_sha256, relative_path)

            targets = [path.as_posix() for path in patch_stack.patch_target_paths()]
            self.assertEqual(
                targets,
                [
                    "poke-engine-py/python/poke_engine/poke_engine.pyi",
                    "poke-engine-py/src/lib.rs",
                    "src/choices.rs",
                    "src/gen3/abilities.rs",
                    "src/gen3/choice_effects.rs",
                    "src/gen3/damage_calc.rs",
                    "src/gen3/generate_instructions.rs",
                    "src/gen3/items.rs",
                    "src/gen3/state.rs",
                    "src/instruction.rs",
                    "src/mcts.rs",
                    "src/state.rs",
                    "tests/test_gen3.rs",
                ],
            )
            self.assertEqual(
                patch_stack.patched_target_tree_sha256(source),
                patch_stack.PATCHED_TARGET_TREE_SHA256,
            )

    def test_shifted_zero_context_fallback_is_rejected_by_full_target_tree_digest(self) -> None:
        """A line-number-only hunk can apply while still landing in the wrong place."""

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = self._extract_source(directory)
            patch_root = directory / "patches"
            patch_root.mkdir()
            patch_list = directory / "patches.txt"
            names = patch_stack.patch_names()
            patch_list.write_text("\n".join(names) + "\n", encoding="utf-8")
            for name in names:
                shutil.copy2(patch_stack.PATCH_ROOT / name, patch_root / name)

            shifted = patch_root / "poke-engine-gen3-forecast-suppressor-handoff.patch"
            shifted.write_text(
                shifted.read_text(encoding="utf-8").replace(
                    "@@ -424,0 +425,23 @@",
                    "@@ -1,0 +2,23 @@",
                    1,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError, "patched target tree digest mismatch"
            ) as raised:
                patch_stack.apply_patch_stack(
                    source, patch_list=patch_list, patch_root=patch_root
                )
            self.assertIn("expected:", str(raised.exception))
            self.assertIn("actual:", str(raised.exception))

    def test_backup_suffix_environment_cannot_create_patch_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = self._extract_source(directory)
            with mock.patch.dict(
                os.environ,
                {
                    "SIMPLE_BACKUP_SUFFIX": ".bak",
                    "VERSION_CONTROL": "numbered",
                    "PATCH_OPTIONS": "--backup --suffix=.custom-backup",
                },
                clear=False,
            ), mock.patch.object(
                patch_stack, "_run", wraps=patch_stack._run
            ) as run:
                patch_stack.apply_patch_stack(source)
            patch_environments = [
                call.kwargs["environment"]
                for call in run.call_args_list
                if call.args[0][0] == "patch"
            ]
            self.assertTrue(patch_environments)
            for environment in patch_environments:
                self.assertIsNotNone(environment)
                for name in patch_stack._PATCH_ENVIRONMENT_OVERRIDES:
                    self.assertNotIn(name, environment)
            self.assertFalse(list(source.rglob("*.bak")))
            self.assertFalse(list(source.rglob("*.custom-backup")))
            self.assertFalse(list(source.rglob("*.orig")))
            self.assertFalse(list(source.rglob("*.rej")))

    def test_actual_patch_failure_output_is_not_masked_by_artifact_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / "source"
            source.mkdir()
            patch = directory / "bad.patch"
            patch.write_text("--- a/file\n+++ b/file\n@@ -1 +1 @@\n-a\n+b\n", encoding="utf-8")
            patch_list = directory / "patches.txt"
            patch_list.write_text("bad.patch\n", encoding="utf-8")
            results = [
                subprocess.CompletedProcess([], 1, b"git preflight output"),
                subprocess.CompletedProcess([], 0, b"patch dry run output"),
                subprocess.CompletedProcess([], 1, b"actual patch apply output"),
            ]
            with mock.patch.object(patch_stack, "_run", side_effect=results):
                with self.assertRaisesRegex(
                    RuntimeError, "actual patch apply output"
                ) as raised:
                    patch_stack.apply_patch_stack(
                        source,
                        patch_list=patch_list,
                        patch_root=directory,
                        expected_target_tree_sha256=None,
                    )
            self.assertNotIn("patch artifacts:", str(raised.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
