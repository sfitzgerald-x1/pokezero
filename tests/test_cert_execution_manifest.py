"""Tests for file-backed certification execution manifest production."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cert_execution_manifest as manifest  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExecutionManifestProducerTests(unittest.TestCase):
    def _source_commit(self) -> str:
        return subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"), text=True
        ).strip()

    def _write(self, path: Path, text: str) -> Path:
        path.write_text(text, encoding="utf-8")
        return path

    def _inputs(self, root: Path) -> dict[str, Path]:
        source = self._source_commit()
        fingerprint = "b" * 64
        image = "c" * 40
        report = self._write(
            root / "shard.json",
            json.dumps({"seeds": {"min": 1000, "max": 1000, "distinct": 1}}),
        )
        checkpoint = self._write(
            root / "shard.jsonl",
            json.dumps(
                {
                    "schema": "engine-transition-differential/1",
                    "build_check": "gated",
                    "seed": 1000,
                    "provenance": {
                        "source_commit": source,
                        "engine_fingerprint": fingerprint,
                        "image_commit": image,
                    },
                }
            ) + "\n",
        )
        engine_artifacts = {}
        for name in ("poke_engine", "pokezero_search"):
            module = self._write(root / f"{name}.so", name)
            engine_artifacts[name] = {
                "module_path": str(module),
                "module_sha256": _sha256(module),
            }
        stamp = self._write(
            root / "engine-stamp.json",
            json.dumps(
                {
                    "schema": "pokezero-engine-build/2",
                    "fingerprint": fingerprint,
                    "artifacts": engine_artifacts,
                }
            ),
        )
        behavior = self._write(root / "behavior.log", "".join(f"[p{n}] PASS\n" for n in range(9)))
        branch = self._write(root / "branch.log", "[search-crate-branch-events] PASS mapped events\n")
        return {
            "contract": self._write(root / "contract.json", "{}\n"),
            "report": report,
            "checkpoint": checkpoint,
            "marker": self._write(root / "shard.complete", "complete\n"),
            "stamp": stamp,
            "behavior": behavior,
            "branch": branch,
        }

    def test_producer_hashes_logs_and_freezes_content_addressed_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            output = root / "manifest.json"
            payload = manifest.produce_manifest(
                contract=paths["contract"],
                readout=ROOT / "scripts" / "cert_sweep_readout.py",
                output=output,
                reports=[paths["report"]],
                checkpoints=[paths["checkpoint"]],
                completion_markers=[paths["marker"]],
                behavioral_logs=[paths["behavior"]],
                branch_logs=[paths["branch"]],
                aggregate_behavioral_log=paths["behavior"],
                aggregate_branch_log=paths["branch"],
                engine_stamp=paths["stamp"],
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload, persisted)
            self.assertEqual(payload["schema"], "engine-cert-execution-manifest/2")
            self.assertEqual(
                payload["aggregate_provenance"]["behavioral_probes"]["sha256"],
                _sha256(paths["behavior"]),
            )
            self.assertEqual(payload["shards"][0]["checkpoint"]["records"], 1)
            self.assertEqual(payload["shards"][0]["image_commit"], "c" * 40)
            self.assertEqual(
                _sha256(Path(payload["contract_blob"]["path"])), _sha256(paths["contract"])
            )

    def test_producer_rejects_mixed_checkpoint_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._inputs(root)
            row = json.loads(paths["checkpoint"].read_text())
            row["provenance"]["engine_fingerprint"] = "d" * 64
            paths["checkpoint"].write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "engine fingerprint"):
                manifest.produce_manifest(
                    contract=paths["contract"],
                    readout=ROOT / "scripts" / "cert_sweep_readout.py",
                    output=root / "manifest.json",
                    reports=[paths["report"]],
                    checkpoints=[paths["checkpoint"]],
                    completion_markers=[paths["marker"]],
                    behavioral_logs=[paths["behavior"]],
                    branch_logs=[paths["branch"]],
                    aggregate_behavioral_log=paths["behavior"],
                    aggregate_branch_log=paths["branch"],
                    engine_stamp=paths["stamp"],
                )


if __name__ == "__main__":
    unittest.main()
