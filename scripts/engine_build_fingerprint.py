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

  FINGERPRINT (content).  A sha256 over the pinned upstream sdist digest, shared
  patch list, every patch file it names, and all tracked search-crate Rust,
  Cargo, build-script, and pyproject inputs. The gitignored vendored tree is a
  verified derivation of the pinned sdist plus those ordered patches, so a clean
  checkout computes the same identity as a post-vendoring build. The builders
  stamp it into the venv at build time; a mismatch means the installed
  artifacts were built from different inputs than the ones checked out.
  Exact, reproducible from tracked bytes, and independent of timestamps.

  The stamp must be written at the END of a FULL rebuild (wheel AND crate), which
  is what the sequence below does — a stamp written after rebuilding only one of
  them would claim currency the other has not earned.

  FRESHNESS (mtime).  The installed extension modules must be NEWER than every
  patch file and every vendored source file. This catches the crate half, which
  maturin builds outside the stamping scripts, and it catches a rebuild that
  silently no-op'd.

Usage::

    python scripts/engine_build_fingerprint.py --write --after-two-consumer-rebuild
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
BASE_SOURCE = REPO_ROOT / "third_party" / "poke-engine-base-source.json"
VENDORED = REPO_ROOT / "third_party" / "poke-engine-src"
# The crate's OWN sources are build inputs too. Without them a .so built before
# an events.rs edit passes the content check whenever timestamps are in the
# provenance-unknown state — the mapper changes, the fingerprint does not. Hit
# in practice while landing the positional attributor.
CRATE_SRC = REPO_ROOT / "rust" / "pokezero-search" / "src"
CRATE_ROOT = REPO_ROOT / "rust" / "pokezero-search"
STAMP_NAME = ".engine-build-fingerprint.json"
STAMP_SCHEMA = "pokezero-engine-build/2"

REBUILD_HINT = """
  scripts/vendor_poke_engine_src.sh <venv-python>
  find third_party/poke-engine-src -name '*.rs' -exec touch {} +
  scripts/setup_poke_engine.sh <venv-python>
  (cd rust/pokezero-search && touch src/lib.rs && maturin build --release \\
       --interpreter <venv-python> -o ../dist)
  uv pip install --python <venv-python> --force-reinstall rust/dist/*.whl
  python scripts/engine_build_fingerprint.py --write --after-two-consumer-rebuild
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


def crate_sources() -> list[Path]:
    """Every `.rs` the search crate compiles, sorted for a stable hash."""

    if not CRATE_SRC.exists():
        return []
    return sorted(CRATE_SRC.rglob("*.rs"))


def cargo_inputs() -> list[Path]:
    """Tracked Cargo manifests/locks compiled by the search crate."""

    return _checked_tree_inputs({"Cargo.toml", "Cargo.lock"})


def build_metadata_inputs() -> list[Path]:
    """Tracked build scripts and Python/maturin feature configuration."""

    return _checked_tree_inputs({"build.rs", "pyproject.toml"})


def _checked_tree_inputs(names: set[str]) -> list[Path]:
    """Find checked build inputs while excluding generated and repository metadata."""

    paths: set[Path] = set()
    for root in (CRATE_ROOT,):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.name in names
                and not {".git", "target"}.intersection(path.relative_to(root).parts)
            ):
                paths.add(path)
    return sorted(paths)


def build_inputs() -> list[Path]:
    """All checked source inputs that can change either installed consumer."""

    return (
        list(patch_files())
        + [BASE_SOURCE]
        + crate_sources()
        + cargo_inputs()
        + build_metadata_inputs()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_fingerprint() -> dict[str, Any]:
    digest = hashlib.sha256()
    if not BASE_SOURCE.exists():
        raise FileNotFoundError(f"missing upstream source pin: {BASE_SOURCE}")
    digest.update(BASE_SOURCE.name.encode())
    digest.update(hashlib.sha256(BASE_SOURCE.read_bytes()).digest())
    digest.update(PATCH_LIST.read_bytes())
    entries = []
    for path in patch_files():
        if not path.exists():
            raise FileNotFoundError(f"patch listed but missing: {path}")
        blob = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(blob).digest())
        entries.append(path.name)
    # Native source and dependency-resolution inputs are hashed by repo-relative
    # path so the digest is location independent. The gitignored vendored engine
    # tree is a deterministic derivation of BASE_SOURCE plus the ordered patch
    # set; both builders verify the upstream archive before applying patches.
    crate = crate_sources()
    cargo = cargo_inputs()
    build_metadata = build_metadata_inputs()
    digest.update(b"--native-inputs--")
    for path in crate + cargo + build_metadata:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "fingerprint": digest.hexdigest(),
        "patches": entries,
        "count": len(entries),
        "base_source": json.loads(BASE_SOURCE.read_text(encoding="utf-8")),
        "crate_sources": len(crate),
        "cargo_inputs": [str(path.relative_to(REPO_ROOT)) for path in cargo],
        "build_metadata_inputs": [
            str(path.relative_to(REPO_ROOT)) for path in build_metadata
        ],
    }


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


def _installed_artifacts() -> dict[str, dict[str, Any]]:
    """Content identities for both installed consumers of the patched engine."""

    artifacts: dict[str, dict[str, Any]] = {}
    for name in ("poke_engine", "pokezero_search"):
        try:
            module = __import__(name)
        except Exception as error:  # noqa: BLE001 -- emitted as a useful rebuild failure
            raise RuntimeError(f"cannot import installed {name}: {type(error).__name__}: {error}") from error
        module_path = Path(getattr(module, "__file__", "") or "").resolve()
        if not module_path.is_file():
            raise RuntimeError(f"installed {name} has no module file")
        extension_paths = sorted(
            path.resolve()
            for suffix in ("*.so", "*.pyd")
            for path in module_path.parent.glob(suffix)
        )
        if not extension_paths:
            raise RuntimeError(f"installed {name} has no native extension artifact")
        artifacts[name] = {
            "module_path": str(module_path),
            "module_sha256": _sha256(module_path),
            "extensions": [
                {"path": str(path), "sha256": _sha256(path)} for path in extension_paths
            ],
        }
    return artifacts


def _stamp_path() -> Path:
    return Path(sys.prefix) / STAMP_NAME


def write_stamp() -> Path:
    """Record the just-built source and installed artifacts for both consumers."""

    payload = compute_fingerprint()
    payload.update(
        {
            "schema": STAMP_SCHEMA,
            "artifacts": _installed_artifacts(),
        }
    )
    path = _stamp_path()
    path.write_text(json.dumps(payload, indent=2))
    return path


def check(*, strict_mtime: bool = True) -> list[str]:
    """Return a list of problems; empty means the build matches HEAD."""

    problems: list[str] = []
    expected = compute_fingerprint()

    fingerprint_ok = False
    stamp = _stamp_path()
    if not stamp.exists():
        problems.append(
            f"no build stamp at {stamp} — the installed engine's provenance is unknown"
        )
    else:
        try:
            recorded = json.loads(stamp.read_text())
        except json.JSONDecodeError:
            recorded = {}
            problems.append(f"build stamp at {stamp} is not valid JSON")
        if not isinstance(recorded, dict):
            recorded = {}
            problems.append(f"build stamp at {stamp} is not a JSON object")
        if recorded.get("schema") != STAMP_SCHEMA:
            problems.append(
                f"build stamp at {stamp} does not attest both installed consumers "
                f"({STAMP_SCHEMA} required)"
            )
        fingerprint_ok = recorded.get("fingerprint") == expected["fingerprint"]
        if not fingerprint_ok:
            problems.append(
                "patch-set fingerprint MISMATCH: installed engine was built from a "
                f"different patch set\n    stamped : {recorded.get('fingerprint','?')[:16]} "
                f"({recorded.get('count','?')} patches)\n"
                f"    HEAD    : {expected['fingerprint'][:16]} ({expected['count']} patches)"
            )
        try:
            actual_artifacts = _installed_artifacts()
        except RuntimeError as error:
            problems.append(str(error))
        else:
            if recorded.get("artifacts") != actual_artifacts:
                problems.append(
                    "installed poke_engine/pokezero_search artifacts do not match the "
                    "two-consumer build stamp"
                )

    if strict_mtime:
        sources = [PATCH_LIST] + build_inputs()
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
                message = (
                    f"{binary.name} predates the engine sources "
                    f"(by {origin}; built before the current patch set / vendored tree)"
                )
                if fingerprint_ok:
                    # The CONTENT fingerprint — which now spans the patch set,
                    # the patch list AND every crate source — matched exactly.
                    # The inputs are therefore identical and a newer mtime just
                    # means a file was touched, not changed. The exact check
                    # outranks the heuristic one; reporting it as an error here
                    # would be the same unsatisfiable false positive the
                    # reproducible-epoch fix removed.
                    print(f"note: {message} — content fingerprint matches, so this is "
                          "a touched file, not a stale build.", file=sys.stderr)
                else:
                    problems.append(f"STALE artifact: {message}")
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
    group.add_argument("--write", action="store_true", help="stamp a completed two-consumer rebuild")
    group.add_argument("--check", action="store_true", help="verify installed vs HEAD")
    group.add_argument("--print", dest="show", action="store_true")
    ap.add_argument(
        "--after-two-consumer-rebuild",
        action="store_true",
        help="required acknowledgement from scripts/build_search_crate_engine.sh",
    )
    args = ap.parse_args(argv)

    if args.write:
        if not args.after_two_consumer_rebuild:
            ap.error("--write requires --after-two-consumer-rebuild; one-consumer stamps are forbidden")
        path = write_stamp()
        payload = json.loads(path.read_text())
        print(
            f"stamped {payload['count']} patches + "
            f"{payload.get('crate_sources', 0)} crate sources -> {path}"
        )
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
