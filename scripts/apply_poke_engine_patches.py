#!/usr/bin/env python
"""Apply PokeZero's pinned poke-engine patch stack without patch fuzz.

Both supported engine builders call this helper. Keeping the exact command in
one place makes the build regression exercise the same stack and ordering that
ships to the Python wheel and the Rust search crate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_LIST = REPO_ROOT / "third_party" / "poke-engine-gen3-patches.txt"
PATCH_ROOT = REPO_ROOT / "third_party"

# This is the canonical post-patch tree for every file touched by the frozen
# patch stack. It deliberately hashes only paths named by ``+++`` patch
# headers, rather than globbing the source tree: upstream has case-colliding
# README files which are not build inputs for this patch stack.
PATCHED_TARGET_TREE_SHA256 = "7334738d06c11894c67709a6b749deb9082d9ccf865ff5232daabdeadaa010ee"
_TARGET_TREE_DOMAIN = b"pokezero.poke-engine.patched-target-tree/v1\0"

# GNU and BSD patch both consult these variables when deciding whether to keep
# backups.  Do not let caller configuration turn a failed fallback into a
# successful-looking source tree containing a stale .bak or custom-suffix copy.
_PATCH_ENVIRONMENT_OVERRIDES = frozenset(
    {
        "BACKUP_CONTROL",
        "BACKUP_SUFFIX",
        "PATCH_BACKUP_SUFFIX",
        "PATCH_OPTIONS",
        "PATCH_VERSION_CONTROL",
        "SIMPLE_BACKUP_SUFFIX",
        "VERSION_CONTROL",
    }
)
_PATCH_ARTIFACT_GLOBS = ("*.orig", "*.rej", "*.bak", "*~")


@dataclasses.dataclass(frozen=True)
class PatchApplication:
    name: str
    backend: str


def patch_names(patch_list: Path = PATCH_LIST) -> list[str]:
    """Return the ordered non-comment patch names from the frozen manifest."""
    return [
        line.strip()
        for line in patch_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def patch_target_paths(
    *,
    patch_list: Path = PATCH_LIST,
    patch_root: Path = PATCH_ROOT,
) -> list[Path]:
    """Return exact postimage paths named by the ordered patch headers.

    The canonical digest is a manifest of these paths and their final bytes.
    Never infer the targets with a source-tree glob: that can accidentally pull
    an unrelated upstream file into the identity on case-insensitive hosts.
    """

    targets: set[Path] = set()
    for patch_name in patch_names(patch_list):
        patch_file = patch_root / patch_name
        if not patch_file.is_file():
            raise FileNotFoundError(f"missing poke-engine patch: {patch_file}")
        for line in patch_file.read_text(encoding="utf-8").splitlines():
            if not line.startswith("+++ "):
                continue
            header_path = line[4:].split("\t", 1)[0]
            if header_path == "/dev/null":
                continue
            # Git-format patches name the postimage as b/path.  Plain unified
            # diffs name it directly; both forms are accepted, but neither may
            # escape the supplied extracted sdist root.
            relative = header_path.removeprefix("b/")
            path = Path(relative)
            if path.is_absolute() or not path.parts or ".." in path.parts:
                raise RuntimeError(
                    f"unsafe postimage path in {patch_name}: {header_path!r}"
                )
            targets.add(path)
    if not targets:
        raise RuntimeError("poke-engine patch manifest names no postimage targets")
    return sorted(targets, key=lambda path: path.as_posix())


def patched_target_tree_sha256(
    source: Path,
    *,
    patch_list: Path = PATCH_LIST,
    patch_root: Path = PATCH_ROOT,
) -> str:
    """Hash the final bytes of every postimage target in the patch manifest."""

    digest = hashlib.sha256()
    digest.update(_TARGET_TREE_DOMAIN)
    for relative in patch_target_paths(patch_list=patch_list, patch_root=patch_root):
        target = source / relative
        if not target.is_file():
            raise RuntimeError(
                f"patched target listed by manifest is missing: {relative.as_posix()}"
            )
        content_sha256 = hashlib.sha256(target.read_bytes()).digest()
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_sha256)
    return digest.hexdigest()


def _assert_pinned_target_tree(
    source: Path,
    *,
    patch_list: Path,
    patch_root: Path,
    expected_target_tree_sha256: str | None,
) -> None:
    if expected_target_tree_sha256 is None:
        return
    actual = patched_target_tree_sha256(
        source, patch_list=patch_list, patch_root=patch_root
    )
    if actual != expected_target_tree_sha256:
        raise RuntimeError(
            "patched target tree digest mismatch; this rejects a zero-context "
            "patch applied at the wrong location\n"
            f"expected: {expected_target_tree_sha256}\n"
            f"actual:   {actual}"
        )


def _patch_environment() -> dict[str, str]:
    """Return an environment where patch cannot create caller-configured backups."""

    environment = os.environ.copy()
    for name in _PATCH_ENVIRONMENT_OVERRIDES:
        environment.pop(name, None)
    return environment


def _run(
    command: list[str],
    *,
    source: Path,
    patch_file: Path | None = None,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run one applicator command without allowing it to mutate on failure."""
    return subprocess.run(
        command,
        cwd=source,
        input=patch_file.read_bytes() if patch_file is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=environment,
    )


def _format_output(result: subprocess.CompletedProcess[bytes]) -> str:
    return result.stdout.decode("utf-8", errors="replace").rstrip()


def _reject_patch_artifacts(source: Path) -> None:
    artifacts = sorted(
        path.relative_to(source).as_posix()
        for suffix in _PATCH_ARTIFACT_GLOBS
        for path in source.rglob(suffix)
    )
    if artifacts:
        raise RuntimeError(
            "patch application left rejected or backup artifacts: " + ", ".join(artifacts)
        )


def apply_patch_stack(
    source: Path,
    *,
    patch_list: Path = PATCH_LIST,
    patch_root: Path = PATCH_ROOT,
    expected_target_tree_sha256: str | None = PATCHED_TARGET_TREE_SHA256,
) -> list[PatchApplication]:
    """Apply every frozen patch with deterministic, context-exact backends.

    ``git apply`` is the portable primary backend: its preflight cannot mutate
    the source and it does not perform patch fuzz. A small pair of historical
    zero-context patches needs GNU/BSD ``patch`` parsing, so only after Git
    rejects do we run ``patch --dry-run --fuzz=0`` before its real apply.
    """
    applied: list[PatchApplication] = []
    for patch_name in patch_names(patch_list):
        patch_file = patch_root / patch_name
        if not patch_file.is_file():
            raise FileNotFoundError(f"missing poke-engine patch: {patch_file}")
        _reject_patch_artifacts(source)
        git_check = _run(
            ["git", "apply", "--no-index", "--check", str(patch_file)], source=source
        )
        if not git_check.returncode:
            git_apply = _run(
                ["git", "apply", "--no-index", str(patch_file)], source=source
            )
            if git_apply.returncode:
                raise RuntimeError(
                    f"git preflight passed but apply failed for {patch_name}\n"
                    f"{_format_output(git_apply)}"
                )
            backend = "git-apply"
        else:
            patch_command = [
                "patch",
                "--batch",
                "--no-backup-if-mismatch",
                "-p1",
                "--forward",
                "--fuzz=0",
            ]
            patch_environment = _patch_environment()
            patch_check = _run(
                patch_command + ["--dry-run"],
                source=source,
                patch_file=patch_file,
                environment=patch_environment,
            )
            if patch_check.returncode:
                raise RuntimeError(
                    f"both strict applicators rejected {patch_name}\n"
                    f"git apply --check:\n{_format_output(git_check)}\n"
                    f"patch --dry-run --fuzz=0:\n{_format_output(patch_check)}"
                )
            patch_apply = _run(
                patch_command,
                source=source,
                patch_file=patch_file,
                environment=patch_environment,
            )
            if patch_apply.returncode:
                # Report patch's own output first.  Artifact rejection remains
                # enforced below, but must not obscure why the apply failed.
                artifacts = sorted(
                    path.relative_to(source).as_posix()
                    for suffix in _PATCH_ARTIFACT_GLOBS
                    for path in source.rglob(suffix)
                )
                artifact_note = (
                    "\npatch artifacts: " + ", ".join(artifacts) if artifacts else ""
                )
                raise RuntimeError(
                    f"patch preflight passed but apply failed for {patch_name}\n"
                    f"{_format_output(patch_apply)}{artifact_note}"
                )
            backend = "patch-fallback"
        _reject_patch_artifacts(source)
        applied.append(PatchApplication(name=patch_name, backend=backend))
        print(f"      {patch_name}: applied via {backend}")
    _assert_pinned_target_tree(
        source,
        patch_list=patch_list,
        patch_root=patch_root,
        expected_target_tree_sha256=expected_target_tree_sha256,
    )
    return applied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="extracted poke-engine source root")
    args = parser.parse_args()
    try:
        apply_patch_stack(args.source)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
