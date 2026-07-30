#!/usr/bin/env python
"""Apply PokeZero's pinned poke-engine patch stack without patch fuzz.

Both supported engine builders call this helper. Keeping the exact command in
one place makes the build regression exercise the same stack and ordering that
ships to the Python wheel and the Rust search crate.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_LIST = REPO_ROOT / "third_party" / "poke-engine-gen3-patches.txt"
PATCH_ROOT = REPO_ROOT / "third_party"


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


def _run(command: list[str], *, source: Path, patch_file: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    """Run one applicator command without allowing it to mutate on failure."""
    return subprocess.run(
        command,
        cwd=source,
        input=patch_file.read_bytes() if patch_file is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _format_output(result: subprocess.CompletedProcess[bytes]) -> str:
    return result.stdout.decode("utf-8", errors="replace").rstrip()


def _reject_patch_artifacts(source: Path) -> None:
    artifacts = sorted(
        path.relative_to(source).as_posix()
        for suffix in ("*.orig", "*.rej")
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
            patch_command = ["patch", "--batch", "-p1", "--forward", "--fuzz=0"]
            patch_check = _run(patch_command + ["--dry-run"], source=source, patch_file=patch_file)
            if patch_check.returncode:
                raise RuntimeError(
                    f"both strict applicators rejected {patch_name}\n"
                    f"git apply --check:\n{_format_output(git_check)}\n"
                    f"patch --dry-run --fuzz=0:\n{_format_output(patch_check)}"
                )
            patch_apply = _run(patch_command, source=source, patch_file=patch_file)
            if patch_apply.returncode:
                _reject_patch_artifacts(source)
                raise RuntimeError(
                    f"patch preflight passed but apply failed for {patch_name}\n"
                    f"{_format_output(patch_apply)}"
                )
            backend = "patch-fallback"
        _reject_patch_artifacts(source)
        applied.append(PatchApplication(name=patch_name, backend=backend))
        print(f"      {patch_name}: applied via {backend}")
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
