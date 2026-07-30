"""Pins for the two-consumer engine build stamp."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import engine_build_fingerprint as fingerprint  # noqa: E402


class EngineBuildStampTests(unittest.TestCase):
    def test_write_stamp_records_both_installed_consumers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stamp = Path(tmp) / "stamp.json"
            artifacts = {
                "poke_engine": {"module_path": "/tmp/engine.py", "module_sha256": "a" * 64, "extensions": []},
                "pokezero_search": {"module_path": "/tmp/search.py", "module_sha256": "b" * 64, "extensions": []},
            }
            with (
                patch.object(fingerprint, "_stamp_path", return_value=stamp),
                patch.object(fingerprint, "compute_fingerprint", return_value={"fingerprint": "c" * 64}),
                patch.object(fingerprint, "_installed_artifacts", return_value=artifacts),
            ):
                fingerprint.write_stamp()
            payload = json.loads(stamp.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "pokezero-engine-build/2")
        self.assertEqual(payload["artifacts"], artifacts)

    def test_check_rejects_legacy_input_only_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stamp = Path(tmp) / "stamp.json"
            stamp.write_text(json.dumps({"fingerprint": "c" * 64, "count": 1}), encoding="utf-8")
            with (
                patch.object(fingerprint, "_stamp_path", return_value=stamp),
                patch.object(fingerprint, "compute_fingerprint", return_value={"fingerprint": "c" * 64, "count": 1}),
                patch.object(fingerprint, "_installed_artifacts", return_value={}),
            ):
                problems = fingerprint.check(strict_mtime=False)
        self.assertTrue(any("does not attest both installed consumers" in problem for problem in problems))
        self.assertTrue(any("artifacts do not match" in problem for problem in problems))

    def test_cargo_manifests_and_locks_change_native_build_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patch_list = root / "third_party" / "poke-engine-gen3-patches.txt"
            patch_list.parent.mkdir(parents=True)
            patch_list.write_text("fixture.patch\n", encoding="utf-8")
            (root / "third_party" / "fixture.patch").write_text("patch\n", encoding="utf-8")
            crate = root / "rust" / "pokezero-search"
            vendored = root / "third_party" / "poke-engine-src"
            (crate / "src").mkdir(parents=True)
            vendored.mkdir(parents=True)
            (crate / "src" / "lib.rs").write_text("fn x() {}\n", encoding="utf-8")
            for directory in (crate, vendored):
                (directory / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
                (directory / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
            with (
                patch.object(fingerprint, "REPO_ROOT", root),
                patch.object(fingerprint, "PATCH_LIST", patch_list),
                patch.object(fingerprint, "VENDORED", vendored),
                patch.object(fingerprint, "CRATE_ROOT", crate),
                patch.object(fingerprint, "CRATE_SRC", crate / "src"),
            ):
                before = fingerprint.compute_fingerprint()["fingerprint"]
                (vendored / "Cargo.lock").write_text("version = 4\n# changed\n", encoding="utf-8")
                after_vendored_lock = fingerprint.compute_fingerprint()["fingerprint"]
                (crate / "Cargo.toml").write_text("[package]\nname='changed'\n", encoding="utf-8")
                after_crate_manifest = fingerprint.compute_fingerprint()["fingerprint"]
        self.assertNotEqual(before, after_vendored_lock)
        self.assertNotEqual(after_vendored_lock, after_crate_manifest)


if __name__ == "__main__":
    unittest.main()
