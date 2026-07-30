#!/usr/bin/env python
"""Apply PokeZero's pinned poke-engine patch stack without patch fuzz.

Both supported engine builders call this helper. Keeping the exact command in
one place makes the build regression exercise the same stack and ordering that
ships to the Python wheel and the Rust search crate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_LIST = REPO_ROOT / "third_party" / "poke-engine-gen3-patches.txt"
PATCH_ROOT = REPO_ROOT / "third_party"


def patch_names(patch_list: Path = PATCH_LIST) -> list[str]:
    """Return the ordered non-comment patch names from the frozen manifest."""
    return [
        line.strip()
        for line in patch_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def apply_patch_stack(
    source: Path,
    *,
    patch_list: Path = PATCH_LIST,
    patch_root: Path = PATCH_ROOT,
) -> list[str]:
    """Apply every frozen patch in order, rejecting any fuzzy hunk match."""
    applied: list[str] = []
    for patch_name in patch_names(patch_list):
        patch_file = patch_root / patch_name
        if not patch_file.is_file():
            raise FileNotFoundError(f"missing poke-engine patch: {patch_file}")
        result = subprocess.run(
            ["patch", "-p1", "--forward", "--fuzz=0"],
            cwd=source,
            input=patch_file.read_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode:
            output = result.stdout.decode("utf-8", errors="replace").rstrip()
            raise RuntimeError(
                f"failed to apply {patch_name} at fuzz=0\n{output}"
            )
        applied.append(patch_name)
        print(f"      {patch_name}: applied")
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
