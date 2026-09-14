"""Contract tests for the create-only deadline source-receipt bridge."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "create_mcts_deadline_source_receipt_under_test",
    ROOT / "scripts" / "create_mcts_deadline_source_receipt.py",
)
assert SPEC is not None and SPEC.loader is not None
receipt_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = receipt_tool
SPEC.loader.exec_module(receipt_tool)


def _write_canonical(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


class DeadlineSourceReceiptTest(unittest.TestCase):
    def _source_fixture(self, temporary: Path) -> tuple[Path, Path, str]:
        source = temporary / "source"
        (source / "scripts").mkdir(parents=True)
        (source / "src" / "pokezero" / "mcts_eval").mkdir(parents=True)
        (source / "rust" / "pokezero-search").mkdir(parents=True)
        for relative in (
            "scripts/run_mcts_deadline_qualification.py",
            "scripts/engine_build_fingerprint.py",
            "src/pokezero/engine_search.py",
            "src/pokezero/mcts_eval/deadline_qualification.py",
            "src/pokezero/mcts_eval/lattice.py",
        ):
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        (source / "rust" / "pokezero-search" / "Cargo.toml").write_text(
            "[package]\nname = 'fixture'\nversion = '0.0.0'\n", encoding="utf-8"
        )
        (source / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.name", "Fixture"], check=True)
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", "-q", "HEAD"], check=True)
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        receipt = temporary / "b2-receipt.json"
        _write_canonical(
            receipt,
            {
                "schema_version": receipt_tool.B2_SOURCE_IMAGE_RECEIPT_SCHEMA_VERSION,
                "complete": True,
                "source_commit": commit,
                "image_digest": "sha256:" + "a" * 64,
                "immutable_image": "registry.example/pokezero@sha256:" + "a" * 64,
                "model_runtime": {
                    "engine_fingerprint": receipt_tool._assignment_literals(
                        source / "scripts" / "run_mcts_deadline_qualification.py"
                    )["REVIEWED_ENGINE_FINGERPRINT"],
                    "source": {"commit": commit, "tree_status": "clean_tracked_checkout"},
                },
            },
        )
        return source, receipt, commit

    def test_complete_receipt_binds_source_image_and_runner_pins(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, b2_receipt, commit = self._source_fixture(Path(raw))
            output = Path(raw) / "deadline-receipt.json"
            self.assertEqual(
                receipt_tool.main(
                    [
                        "--create",
                        "--source-root",
                        str(source),
                        "--b2-receipt",
                        str(b2_receipt),
                        "--out",
                        str(output),
                    ]
                ),
                0,
            )
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], receipt_tool.SOURCE_RECEIPT_SCHEMA_VERSION)
            self.assertEqual(result["source_commit"], commit)
            self.assertEqual(result["immutable_image"], "registry.example/pokezero@sha256:" + "a" * 64)
            self.assertEqual(
                result["engine_fingerprint"],
                receipt_tool._assignment_literals(source / "scripts" / "run_mcts_deadline_qualification.py")[
                    "REVIEWED_ENGINE_FINGERPRINT"
                ],
            )
            self.assertIn("src/pokezero/engine_search.py", result["source_files_sha256"])

    def test_rejects_a_native_fingerprint_that_does_not_match_the_runner(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, b2_receipt, _ = self._source_fixture(Path(raw))
            payload = json.loads(b2_receipt.read_text(encoding="utf-8"))
            payload["model_runtime"]["engine_fingerprint"] = "b" * 64
            _write_canonical(b2_receipt, payload)
            with self.assertRaisesRegex(receipt_tool.ReceiptError, "native fingerprint"):
                receipt_tool.build_receipt(source_root=source, b2_receipt_path=b2_receipt)

    def test_refuses_to_replace_an_existing_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, b2_receipt, _ = self._source_fixture(Path(raw))
            output = Path(raw) / "deadline-receipt.json"
            output.write_text('{"sentinel":"preserve"}\n', encoding="utf-8")
            self.assertEqual(
                receipt_tool.main(
                    [
                        "--create",
                        "--source-root",
                        str(source),
                        "--b2-receipt",
                        str(b2_receipt),
                        "--out",
                        str(output),
                    ]
                ),
                2,
            )
            self.assertEqual(output.read_text(encoding="utf-8"), '{"sentinel":"preserve"}\n')

    def test_dirty_source_checkout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, b2_receipt, _ = self._source_fixture(Path(raw))
            (source / "scripts" / "new-untracked.py").write_text("# drift\n", encoding="utf-8")
            with self.assertRaisesRegex(receipt_tool.ReceiptError, "checkout is dirty"):
                receipt_tool.build_receipt(source_root=source, b2_receipt_path=b2_receipt)
