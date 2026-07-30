#!/usr/bin/env python
"""Verify the exact upstream poke-engine sdist before applying local patches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PIN = REPO_ROOT / "third_party" / "poke-engine-base-source.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_pin() -> dict[str, str]:
    payload = json.loads(SOURCE_PIN.read_text(encoding="utf-8"))
    required = {
        "schema": "pokezero-engine-upstream-source/1",
        "distribution": "poke-engine",
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in required.items()
    ):
        raise ValueError(f"invalid upstream source pin: {SOURCE_PIN}")
    for field in ("version", "archive", "sha256"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"upstream source pin has no {field}: {SOURCE_PIN}")
    if not _SHA256_RE.fullmatch(payload["sha256"]):
        raise ValueError(f"upstream source pin has malformed sha256: {SOURCE_PIN}")
    return payload


def verify(archive: Path, *, expected_version: str) -> dict[str, str]:
    payload = source_pin()
    if payload["version"] != expected_version:
        raise ValueError(
            f"upstream source version mismatch: pin={payload['version']} "
            f"builder={expected_version}"
        )
    if archive.name != payload["archive"]:
        raise ValueError(
            f"upstream source archive mismatch: pin={payload['archive']} "
            f"download={archive.name}"
        )
    actual = _sha256(archive)
    if actual != payload["sha256"]:
        raise ValueError(
            f"upstream source SHA-256 mismatch: pin={payload['sha256']} "
            f"download={actual}"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    payload = verify(args.archive, expected_version=args.expected_version)
    print(
        f"verified {payload['distribution']}=={payload['version']} "
        f"sdist sha256={payload['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
