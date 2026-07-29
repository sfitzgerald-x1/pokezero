#!/usr/bin/env python
"""Assert the INSTALLED engine artifacts were built from the CHECKED-OUT patch set.

The trap this closes has now bitten twice, one layer apart:

  * cargo's cache survived an rsync'd vendored-tree swap, so a "rebuilt" crate
    still linked the old engine;
  * a stale installed WHEEL against a current vendored tree produced 4.43 %
    divergence where a wheel rebuilt at HEAD produced 1.11 % — on identical
    seeds.

Both failure modes yield a *plausible number*, not an error, which is the worst
possible shape for a measurement that gates an acceptance criterion.

Two independent checks, because they catch different halves:

  FINGERPRINT (content).  A sha256 over the shared patch list and every patch
  file it names — exactly the inputs both builders read. The builders stamp it
  into the venv at build time; a mismatch means the installed wheel was built
  from a different patch set than the one checked out. This is exact and does
  not depend on timestamps.

  FRESHNESS (mtime).  The installed extension modules must be NEWER than every
  patch file and every vendored source file. This catches the crate half, which
  maturin builds outside the stamping scripts, and it catches a rebuild that
  silently no-op'd.

Usage::

    python scripts/engine_build_fingerprint.py --write        # after building
    python scripts/engine_build_fingerprint.py --check        # before measuring
    python scripts/engine_build_fingerprint.py --print        # show the hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_LIST = REPO_ROOT / "third_party" / "poke-engine-gen3-patches.txt"
VENDORED = REPO_ROOT / "third_party" / "poke-engine-src"
STAMP_NAME = ".engine-build-fingerprint.json"

REBUILD_HINT = """
  scripts/vendor_poke_engine_src.sh <venv-python>
  find third_party/poke-engine-src -name '*.rs' -exec touch {} +
  scripts/setup_poke_engine.sh <venv-python>
  (cd rust/pokezero-search && touch src/lib.rs && maturin build --release \\
       --interpreter <venv-python> -o ../dist)
  uv pip install --python <venv-python> --force-reinstall rust/dist/*.whl
  python scripts/engine_build_fingerprint.py --write
""".rstrip()


def patch_files() -> list[Path]:
    """The patch files named by the shared list, in apply order."""

    if not PATCH_LIST.exists():
        raise FileNotFoundError(f"missing patch list: {PATCH_LIST}")
    names = [
        line.strip()
        for line in PATCH_LIST.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return [REPO_ROOT / "third_party" / name for name in names]


def compute_fingerprint() -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(PATCH_LIST.read_bytes())
    entries = []
    for path in patch_files():
        if not path.exists():
            raise FileNotFoundError(f"patch listed but missing: {path}")
        blob = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(blob).digest())
        entries.append(path.name)
    return {"fingerprint": digest.hexdigest(), "patches": entries, "count": len(entries)}


# maturin stamps extension modules with the reproducible-build epoch
# 315561600 (1980-01-01), and archive-extracting installers preserve it. Such a
# file's own mtime carries NO provenance, so comparing it against source mtimes
# reports a freshly built engine as STALE forever — which pushes operators
# toward --skip-build-check and trains exactly the habit the gate exists to
# prevent. Anything before 2000 is treated as "unknown", never as "old".
_REPRODUCIBLE_EPOCH_CUTOFF = 946684800.0  # 2000-01-01


def _dist_info_record(binary: Path) -> Path | None:
    """The installing wheel's RECORD, whose mtime is real INSTALL time."""

    site_packages = binary.parent.parent
    stem = binary.parent.name
    for dist_info in sorted(site_packages.glob(f"{stem}-*.dist-info")):
        record = dist_info / "RECORD"
        if record.exists():
            return record
    return None


def _artifact_time(binary: Path) -> tuple[float | None, str]:
    """Best available install time for an artifact, and where it came from.

    Returns ``(None, "unknown")`` when no timestamp can be trusted — the caller
    then relies on the content fingerprint, which is exact and does not depend
    on timestamps at all.
    """

    own = binary.stat().st_mtime
    if own >= _REPRODUCIBLE_EPOCH_CUTOFF:
        return own, "mtime"
    record = _dist_info_record(binary)
    if record is not None:
        recorded = record.stat().st_mtime
        if recorded >= _REPRODUCIBLE_EPOCH_CUTOFF:
            return recorded, "dist-info RECORD"
    return None, "unknown"


def _installed_binaries() -> list[Path]:
    """Installed extension modules for the two engine-bearing packages."""

    found: list[Path] = []
    for name in ("poke_engine", "pokezero_search"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 — reported by the caller as unusable
            continue
        base = Path(getattr(module, "__file__", "") or "").parent
        if base.exists():
            found.extend(base.glob("*.so"))
            found.extend(base.glob("*.pyd"))
    return found


def _stamp_path() -> Path:
    return Path(sys.prefix) / STAMP_NAME


def write_stamp() -> Path:
    payload = compute_fingerprint()
    path = _stamp_path()
    path.write_text(json.dumps(payload, indent=2))
    return path


def check(*, strict_mtime: bool = True) -> list[str]:
    """Return a list of problems; empty means the build matches HEAD."""

    problems: list[str] = []
    expected = compute_fingerprint()

    stamp = _stamp_path()
    if not stamp.exists():
        problems.append(
            f"no build stamp at {stamp} — the installed engine's provenance is unknown"
        )
    else:
        recorded = json.loads(stamp.read_text())
        if recorded.get("fingerprint") != expected["fingerprint"]:
            problems.append(
                "patch-set fingerprint MISMATCH: installed engine was built from a "
                f"different patch set\n    stamped : {recorded.get('fingerprint','?')[:16]} "
                f"({recorded.get('count','?')} patches)\n"
                f"    HEAD    : {expected['fingerprint'][:16]} ({expected['count']} patches)"
            )

    if strict_mtime:
        sources = list(patch_files()) + [PATCH_LIST]
        if VENDORED.exists():
            sources += list(VENDORED.rglob("*.rs"))
        newest_source = max((p.stat().st_mtime for p in sources if p.exists()), default=0.0)
        binaries = _installed_binaries()
        if not binaries:
            problems.append("no installed poke_engine / pokezero_search extension found")
        undatable = []
        for binary in binaries:
            stamped, origin = _artifact_time(binary)
            if stamped is None:
                # Not evidence of staleness — evidence of nothing. The content
                # fingerprint above is the authority in this case.
                undatable.append(binary.name)
                continue
            if stamped < newest_source:
                problems.append(
                    f"STALE artifact: {binary.name} predates the engine sources "
                    f"(by {origin}; built before the current patch set / vendored tree)"
                )
        if undatable:
            print(
                "note: reproducible-build timestamps on "
                f"{', '.join(sorted(undatable))} — freshness rests on the content "
                "fingerprint alone. That fingerprint spans the PATCH SET, not the "
                "built artifact, so an artifact built from the right patches at the "
                "wrong crate-source commit is invisible in this state (states 1-2 "
                "catch it by mtime). Unreachable for the acceptance run, where each "
                "shard rebuilds from a clean vendor.",
                file=sys.stderr,
            )
    return problems


def assert_fresh(*, skip: bool = False) -> None:
    """Fail loudly rather than let a stale build produce a plausible number."""

    if skip:
        print("WARNING: engine build-freshness check SKIPPED (--skip-build-check)",
              file=sys.stderr)
        return
    problems = check()
    if not problems:
        return
    message = "\n".join(f"  - {p}" for p in problems)
    raise SystemExit(
        "\nENGINE BUILD IS NOT CURRENT — refusing to measure.\n\n"
        f"{message}\n\n"
        "A stale build does not error, it produces a PLAUSIBLE NUMBER (a stale wheel\n"
        "measured 4.43 % where HEAD measured 1.11 % on identical seeds), so this is a\n"
        "hard stop. Rebuild and re-stamp:\n"
        f"{REBUILD_HINT}\n\n"
        "Pass --skip-build-check only for offline analysis (--merge-from) where no\n"
        "engine call is made.\n"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="stamp the current patch set")
    group.add_argument("--check", action="store_true", help="verify installed vs HEAD")
    group.add_argument("--print", dest="show", action="store_true")
    args = ap.parse_args(argv)

    if args.write:
        path = write_stamp()
        payload = json.loads(path.read_text())
        print(f"stamped {payload['count']} patches -> {path}")
        print(f"  fingerprint {payload['fingerprint']}")
        return 0
    if args.show:
        print(json.dumps(compute_fingerprint(), indent=2))
        return 0
    problems = check()
    if problems:
        for problem in problems:
            print(f"STALE: {problem}", file=sys.stderr)
        return 1
    payload = compute_fingerprint()
    print(f"engine build is current ({payload['count']} patches, "
          f"{payload['fingerprint'][:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
