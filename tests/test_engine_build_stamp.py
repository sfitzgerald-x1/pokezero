"""Pins for the two-consumer engine build stamp."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import engine_build_fingerprint as fingerprint  # noqa: E402
import verify_poke_engine_source as source_verifier  # noqa: E402


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

    def test_fingerprint_is_reproducible_without_gitignored_vendor_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patch_list = root / "third_party" / "poke-engine-gen3-patches.txt"
            patch_list.parent.mkdir(parents=True)
            patch_list.write_text("fixture.patch\n", encoding="utf-8")
            (root / "third_party" / "fixture.patch").write_text("patch\n", encoding="utf-8")
            base_source = root / "third_party" / "poke-engine-base-source.json"
            base_source.write_text(
                json.dumps(
                    {
                        "schema": "pokezero-engine-upstream-source/1",
                        "distribution": "poke-engine",
                        "version": "0.0.47",
                        "archive": "poke_engine-0.0.47.tar.gz",
                        "sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            crate = root / "rust" / "pokezero-search"
            vendored = root / "third_party" / "poke-engine-src"
            (crate / "src").mkdir(parents=True)
            (crate / "src" / "lib.rs").write_text("fn x() {}\n", encoding="utf-8")
            (crate / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
            (crate / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
            crate_pyproject = crate / "pyproject.toml"
            crate_pyproject.write_text(
                "[tool.maturin]\nfeatures=['pyo3/extension-module']\n",
                encoding="utf-8",
            )
            crate_build = crate / "build.rs"
            crate_build.write_text("fn main() {}\n", encoding="utf-8")
            target_manifest = crate / "target" / "package" / "fixture" / "Cargo.toml"
            target_manifest.parent.mkdir(parents=True)
            target_manifest.write_text("[package]\nname='generated'\n", encoding="utf-8")
            with (
                patch.object(fingerprint, "REPO_ROOT", root),
                patch.object(fingerprint, "PATCH_LIST", patch_list),
                patch.object(fingerprint, "BASE_SOURCE", base_source),
                patch.object(fingerprint, "VENDORED", vendored),
                patch.object(fingerprint, "CRATE_ROOT", crate),
                patch.object(fingerprint, "CRATE_SRC", crate / "src"),
            ):
                before = fingerprint.compute_fingerprint()["fingerprint"]
                vendored.mkdir(parents=True)
                (vendored / "Cargo.toml").write_text(
                    "[package]\nname='derived'\n", encoding="utf-8"
                )
                after_vendoring = fingerprint.compute_fingerprint()["fingerprint"]
                payload = json.loads(base_source.read_text(encoding="utf-8"))
                payload["sha256"] = "b" * 64
                base_source.write_text(json.dumps(payload), encoding="utf-8")
                after_source_pin = fingerprint.compute_fingerprint()["fingerprint"]
                (crate / "Cargo.toml").write_text("[package]\nname='changed'\n", encoding="utf-8")
                after_crate_manifest = fingerprint.compute_fingerprint()["fingerprint"]
                crate_pyproject.write_text(
                    "[tool.maturin]\nfeatures=['different-feature']\n",
                    encoding="utf-8",
                )
                after_pyproject = fingerprint.compute_fingerprint()["fingerprint"]
                crate_build.write_text("fn main() { println!(\"changed\"); }\n", encoding="utf-8")
                after_build_script = fingerprint.compute_fingerprint()["fingerprint"]
                target_manifest.write_text(
                    "[package]\nname='generated-change'\n",
                    encoding="utf-8",
                )
                after_target_output = fingerprint.compute_fingerprint()["fingerprint"]
        self.assertEqual(before, after_vendoring)
        self.assertNotEqual(after_vendoring, after_source_pin)
        self.assertNotEqual(after_source_pin, after_crate_manifest)
        self.assertNotEqual(after_crate_manifest, after_pyproject)
        self.assertNotEqual(after_pyproject, after_build_script)
        self.assertEqual(after_build_script, after_target_output)

    def test_upstream_source_pin_rejects_wrong_archive_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "poke_engine-0.0.47.tar.gz"
            archive.write_bytes(b"upstream source")
            pin = root / "poke-engine-base-source.json"
            pin.write_text(
                json.dumps(
                    {
                        "schema": "pokezero-engine-upstream-source/1",
                        "distribution": "poke-engine",
                        "version": "0.0.47",
                        "archive": archive.name,
                        "sha256": hashlib.sha256(b"different").hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(source_verifier, "SOURCE_PIN", pin):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    source_verifier.verify(archive, expected_version="0.0.47")

    def test_upstream_source_pin_rejects_non_hex_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pin = Path(tmp) / "poke-engine-base-source.json"
            pin.write_text(
                json.dumps(
                    {
                        "schema": "pokezero-engine-upstream-source/1",
                        "distribution": "poke-engine",
                        "version": "0.0.47",
                        "archive": "poke_engine-0.0.47.tar.gz",
                        "sha256": "z" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(source_verifier, "SOURCE_PIN", pin):
                with self.assertRaisesRegex(ValueError, "malformed sha256"):
                    source_verifier.source_pin()


if __name__ == "__main__":
    unittest.main()
