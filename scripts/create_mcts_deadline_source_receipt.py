#!/usr/bin/env python3
"""Create the immutable source receipt required by deadline qualification.

The source-image builder already proves a digest-qualified image and the native
model runtime.  The deadline replay additionally needs a compact receipt that
binds every executable source input and the reviewed deadline mechanism.  This
tool bridges those two contracts without accepting a mutable tag, a dirty
checkout, an unpinned engine, or an existing receipt path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SOURCE_RECEIPT_SCHEMA_VERSION = "pokezero.mcts-deadline-source-receipt.v1"
B2_SOURCE_IMAGE_RECEIPT_SCHEMA_VERSION = "pokezero.b2-source-image-receipt.v7"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ReceiptError(RuntimeError):
    """A source/image identity cannot support a deadline qualification."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _read_canonical_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ReceiptError(f"receipt must be a regular file: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ReceiptError(f"receipt is not JSON: {path}") from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise ReceiptError(f"receipt is not canonical JSON: {path}")
    return value


def _git(source_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReceiptError(f"cannot inspect source checkout: {source_root}") from error


def _clean_detached_commit(source_root: Path) -> str:
    if not source_root.is_dir() or source_root.is_symlink():
        raise ReceiptError("source root must be a regular directory")
    if _git(source_root, "rev-parse", "--is-inside-work-tree") != "true":
        raise ReceiptError("source root is not a Git checkout")
    if _git(source_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ReceiptError("source checkout is dirty")
    try:
        symbolic_ref = subprocess.run(
            ["git", "-C", str(source_root), "symbolic-ref", "-q", "HEAD"],
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ReceiptError("cannot determine source checkout HEAD mode") from error
    if symbolic_ref.returncode == 0:
        raise ReceiptError("source checkout must be detached")
    if symbolic_ref.returncode != 1:
        raise ReceiptError("cannot determine source checkout HEAD mode")
    commit = _git(source_root, "rev-parse", "HEAD").lower()
    if not SHA1.fullmatch(commit):
        raise ReceiptError("source checkout HEAD is not a full lowercase Git commit")
    return commit


def _assignment_literals(path: Path) -> dict[str, Any]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ReceiptError(f"cannot parse qualification runner: {path}") from error
    values: dict[str, Any] = {}
    names = {
        "REVIEWED_ENGINE_SEARCH_SHA256",
        "REVIEWED_ENGINE_FINGERPRINT",
        "REQUIRED_RECEIPT_FILES",
    }
    for node in tree.body:
        targets: Iterable[ast.expr]
        value: ast.expr | None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = (node.target,), node.value
        else:
            continue
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    values[target.id] = ast.literal_eval(value)
                except ValueError as error:
                    raise ReceiptError(f"runner pin is not a literal: {target.id}") from error
    if set(values) != names:
        raise ReceiptError("qualification runner omits a required source pin")
    if not SHA256.fullmatch(str(values["REVIEWED_ENGINE_SEARCH_SHA256"])):
        raise ReceiptError("qualification runner has an invalid engine-source pin")
    if not SHA256.fullmatch(str(values["REVIEWED_ENGINE_FINGERPRINT"])):
        raise ReceiptError("qualification runner has an invalid native fingerprint pin")
    files = values["REQUIRED_RECEIPT_FILES"]
    if not isinstance(files, tuple) or not files or any(not isinstance(item, str) for item in files):
        raise ReceiptError("qualification runner has an invalid required-file roster")
    return values


def _execution_source_files(source_root: Path) -> list[Path]:
    paths: list[Path] = []
    for root, patterns in (
        (source_root / "src" / "pokezero", ("*.py",)),
        (source_root / "scripts", ("*.py", "*.mjs")),
        (source_root / "rust" / "pokezero-search", ("*",)),
    ):
        if root.is_dir():
            for pattern in patterns:
                paths.extend(
                    path
                    for path in root.rglob(pattern)
                    if path.is_file()
                    and "__pycache__" not in path.parts
                    and "target" not in path.parts
                )
    pyproject = source_root / "pyproject.toml"
    if pyproject.is_file():
        paths.append(pyproject)
    result = sorted(set(paths))
    if not result:
        raise ReceiptError("source checkout has no executable source inputs")
    return result


def _execution_tree_sha256(source_root: Path, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(source_root).as_posix()):
        relative = path.relative_to(source_root).as_posix()
        blob = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)
    return digest.hexdigest()


def _validate_b2_receipt(receipt: Mapping[str, Any], *, commit: str, fingerprint: str) -> None:
    if receipt.get("schema_version") != B2_SOURCE_IMAGE_RECEIPT_SCHEMA_VERSION:
        raise ReceiptError("source-image receipt schema is not supported")
    if receipt.get("complete") is not True:
        raise ReceiptError("source-image receipt is not complete")
    if receipt.get("source_commit") != commit:
        raise ReceiptError("source-image receipt commit differs from detached source")
    digest = receipt.get("image_digest")
    image = receipt.get("immutable_image")
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise ReceiptError("source-image receipt has an invalid image digest")
    if not isinstance(image, str) or image.rsplit("@", 1)[-1] != digest:
        raise ReceiptError("source-image receipt is not digest-qualified")
    runtime = receipt.get("model_runtime")
    if not isinstance(runtime, Mapping):
        raise ReceiptError("source-image receipt omits model runtime evidence")
    if runtime.get("engine_fingerprint") != fingerprint:
        raise ReceiptError("source-image native fingerprint differs from the reviewed mechanism")
    source = runtime.get("source")
    if source != {"commit": commit, "tree_status": "clean_tracked_checkout"}:
        raise ReceiptError("source-image runtime provenance differs from detached source")


def build_receipt(*, source_root: Path, b2_receipt_path: Path) -> dict[str, Any]:
    commit = _clean_detached_commit(source_root)
    runner = source_root / "scripts" / "run_mcts_deadline_qualification.py"
    pins = _assignment_literals(runner)
    engine = source_root / "src" / "pokezero" / "engine_search.py"
    if _sha256(engine) != pins["REVIEWED_ENGINE_SEARCH_SHA256"]:
        raise ReceiptError("engine_search.py differs from the reviewed deadline mechanism")
    b2_receipt = _read_canonical_json(b2_receipt_path)
    _validate_b2_receipt(
        b2_receipt,
        commit=commit,
        fingerprint=str(pins["REVIEWED_ENGINE_FINGERPRINT"]),
    )
    required_files = tuple(pins["REQUIRED_RECEIPT_FILES"])
    source_files: dict[str, str] = {}
    for relative in required_files:
        candidate = (source_root / relative).resolve()
        try:
            candidate.relative_to(source_root.resolve())
        except ValueError as error:
            raise ReceiptError("qualification runner required file escapes source root") from error
        if not candidate.is_file() or candidate.is_symlink():
            raise ReceiptError(f"qualification runner required file is missing: {relative}")
        source_files[relative] = _sha256(candidate)
    inputs = _execution_source_files(source_root)
    return {
        "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
        "complete": True,
        "immutable_image": b2_receipt["immutable_image"],
        "source_commit": commit,
        "execution_tree_sha256": _execution_tree_sha256(source_root, inputs),
        "engine_fingerprint": pins["REVIEWED_ENGINE_FINGERPRINT"],
        "source_files_sha256": source_files,
    }


def _create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ReceiptError(f"refusing to replace existing source receipt: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ReceiptError("source receipt parent must be a regular existing directory")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReceiptError(f"refusing to replace existing source receipt: {path}") from error
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--create", action="store_true", help="Create one new receipt; never replace.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--b2-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.create:
        parser.error("--create is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        receipt = build_receipt(
            source_root=args.source_root.resolve(),
            b2_receipt_path=args.b2_receipt.resolve(),
        )
        _create_only_json(args.out.resolve(), receipt)
    except ReceiptError as error:
        print(f"CANNOT CREATE DEADLINE SOURCE RECEIPT: {error}", file=sys.stderr)
        return 2
    print(f"WROTE MCTS DEADLINE SOURCE RECEIPT {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
