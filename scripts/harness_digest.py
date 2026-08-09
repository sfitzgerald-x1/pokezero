#!/usr/bin/env python
"""A content identity for the PYTHON HALF of the measuring instrument.

WHY THIS EXISTS, and why it is not `engine_build_fingerprint.py`.

`scripts/engine_build_fingerprint.py` answers one question: *were the installed
native artifacts built from the checked-out inputs?* Its `build_inputs()` is
therefore a REBUILD TRIGGER — patch files, `BASE_SOURCE`, the search crate's
`.rs`/Cargo/`build.rs`/`pyproject`. Nothing under `scripts/` or `src/pokezero/`
belongs in it: those files change nothing about the wheel, and adding them would
fire the 60-minute `mass-gate` engine build on every prose or test edit. That set
is correct as it stands and this module does not touch it.

But a sweep number is produced by TWO things: the engine, and the Python harness
that drives it, renders both sides and decides what "matched" means. The engine
fingerprint identifies the first and says nothing at all about the second, so two
sweeps can report the SAME fingerprint and have been produced by different
instruments. That is not hypothetical:

  * Over the committed corpus (glob `reports/**/*.json` + `docs/**/*.json`,
    recursive, 375 files, 82 of which carry a `checkpoint_provenance` blob with a
    non-null `engine_fingerprint`), there are 29 distinct engine fingerprints and
    SIX of them span more than one `source_commit` — groups the fingerprint calls
    "the same build" that different harnesses produced. See
    `tests/test_harness_digest_provenance.py`, which pins that exact set.
  * Confirmed live once: dev `strict:diverged_on_full_branch_set` = 1 on a fresh
    base build, absent from the C142 artifact at the SAME fingerprint `5fa147ff`,
    because the counter was added to `engine_transition_differential.py` after
    C142's sweeps ran.

RETRO-VALIDATED AGAINST THAT CASE, rather than argued from the design. This module
was checked out into worktrees of three of the four commits in the `5fa147ff`
group and run there (the fourth, `4c0ded45`, is no longer reachable in this clone):

    662d9db8  engine_fp 5fa147ffa325c887   harness 86abc2dba0377271
    ce962c6e  engine_fp 5fa147ffa325c887   harness 86abc2dba0377271
    e0a23e4e  engine_fp 5fa147ffa325c887   harness 45d2b12d962ada12

One engine fingerprint, two instruments. The digest AGREES across C142's own pair
-- same instrument, and that matters, because a hash that simply differed
everywhere would separate nothing -- and SEPARATES C147's base sweeps from them.
The single closure file that differs between `ce962c6e` and `e0a23e4e` is
`engine_transition_differential.py`, and `strict:diverged_on_full_branch_set`
occurs 0 times in it at the two C142 commits and 2 times at `e0a23e4e`. So the
digest would have caught the one drift this program has confirmed by hand, and it
identifies it as the same event.

The owner has deferred the ratified `19,300,000`-`19,300,199` sweep until "the
ledger is terminal and the engine fingerprint is frozen"
(`RATIFIED_SWEEP_PRECONDITION`). Freezing the engine fingerprint alone does not
freeze the instrument, so the terminal claim could shift under a matcher change
while still reporting the frozen fingerprint. This module is the missing half.

WHY A DIGEST AND NOT A `source_commit` PIN. Every artifact already records
`source_commit`, so the drift above is already DETECTABLE and merely unasserted, and
a pin comparing the two is much cheaper. It was weighed and rejected on measured
grounds, in both directions:

  * `source_commit` is `git rev-parse HEAD` and NOTHING ELSE. Of the 82 provenance
    blobs above, 8 record `source_tree: "dirty"` and 56 predate the field
    entirely — so for 64 of 82 the stamped commit is not known to describe the
    tree that ran. A commit-equality pin certifies nothing on those, and sweeping
    a dirty tree is the normal way to measure a change before committing it.
  * In the other direction `source_commit` moves for every prose edit. A pin
    demanding one commit per fingerprint would redden on a README, which is how
    pins get widened until they are inert.

A content digest has neither failure: it is exact on a dirty tree and silent on a
prose edit.

WHAT IT COVERS. The full static first-party import closure of
`scripts/engine_transition_differential.py`, resolved by AST from the filesystem:
`scripts/<mod>.py` for bare top-level imports, `src/pokezero/<...>.py` for
`pokezero.*`, and — this is the part a first revision got wrong — RELATIVE imports
resolved against the containing package. 73 files at the time of writing.

THE TRUNCATION THAT WAS CAUGHT IN REVIEW, recorded because the number of files a
digest covers is exactly the sort of claim that goes stale silently. The first
revision of `_absolutize` dropped every `node.level > 0` import, with a comment
asserting relative imports "do not occur in either layout". `src/pokezero/**`
carries 70 relative `ImportFrom` statements across 9 of the 16 files that revision
hashed, so the closure came out at 16 files instead of 73 — and the missing 57
included `gen3_damage.py` (reached from `engine_world.py`), `showdown_fixture.py`
(module-level in `engine_fidelity.py`, `engine_fidelity_multiturn.py` and
`engine_world.py`) and `poke_engine_backend.py` (from `poke_engine_adapter.py`),
all three on the live sweep path. A semantic off-by-one in `gen3_stat` — 10,385
calls on a single dev game — left the digest byte-identical. The digest carried
the very defect it exists to catch, one level in, while its own docstring claimed
the full instrument.

WHY THE FULL CLOSURE AND NOT A NARROWER, TIDIER ONE. Following relative imports
pulls in `neural_policy.py`, `search.py`, `showdown.py` and the rest of the
training tree through `engine_search.py` and `pokezero/__init__.py`, which looks
like unwanted churn. It was MEASURED rather than argued: over the last 300 commits
of `origin/main`, 42 touch at least one of the truncated 16 and 49 touch at least
one of the honest 73 — SEVEN commits of difference, 2% of commits. That is the
entire cost of the honest answer, and it does not come close to justifying a
truncation that would have to be defended by proving the excluded 57 cannot affect
a sweep. That proof is strictly harder than the claim which just failed.

The residual coupling is real and worth naming as a follow-up rather than hiding
in a resolver: the differential should not be importing the training tree at all.
Decoupling `engine_search.py` and `pokezero/__init__.py` is the right fix for the
churn. Truncating the digest is not.

CROSS-CHECKED AGAINST RUNTIME, because a static resolver that agrees with itself
proves nothing. Importing the differential in a built venv loads 72 first-party
modules. Every one is in this closure; the closure additionally holds
`inference_service.py`, which is imported lazily and so is not resident at import
time. Static ⊇ runtime, over-capturing by exactly one file — the safe direction.

WHAT IT DOES NOT COVER, stated as narrowly as it was measured:

  * The native engine. Deliberate — that is the engine fingerprint's job, and the
    two are recorded side by side rather than merged.
  * Anything reached by a COMPUTED import name. `__import__`/`importlib` with a
    non-literal argument is invisible to this resolver, and this is the one class
    of import the completeness pin cannot re-derive either, so it is the residual
    gap. Re-measured over the 73 rather than restated:
    `grep -nE 'importlib|__import__'` matches FOUR files, and every site resolves
    to the native engine or to nothing at all —

      - `engine_build_fingerprint.py`: two `__import__(name)` calls over the
        literal tuple `("poke_engine", "pokezero_search")`;
      - `poke_engine_backend.py`: `probe_poke_engine`'s injectable
        `importer=importlib.import_module`, called once on the module constant
        `POKE_ENGINE_IMPORT_NAME`;
      - `engine_transition_differential.py`: a comment;
      - this file: this paragraph.

    Both live sites import the native engine, which the ENGINE fingerprint covers,
    so as of this commit the computed-import gap admits no first-party Python. That
    is a statement about these 73 files at this commit, not a structural guarantee.
    An earlier revision of this paragraph said "plus importlib sites in the training
    tree", which the grep does not support; it was written from expectation rather
    than from the output.
  * Import cycles in `src/pokezero/**` are followed, not flagged: the walk is a
    visited-set traversal, so a cycle terminates rather than recursing.
  * Third-party dependency versions, the Showdown checkout, and the interpreter.
    Those are runtime environment, not tracked bytes.
  * `tests/`, `docs/`, `reports/`, and every `scripts/` tool the differential does
    not import. Editing them cannot change a sweep number and must not move the
    digest.

HOW A FUTURE TRUNCATION IS CAUGHT. The exact-membership pin cannot do it on its
own — that was the second half of the review finding. A pin listing the 16 files
that ARE found stays green forever over a truncated graph, because it can only
detect growth that never happened. So
`tests/test_harness_digest_provenance.py` also carries a COMPLETENESS pin, which
re-derives every relative import target with an independent, deliberately naive
textual implementation and asserts each one is already in the closure. A resolver
that drops an import class goes red there immediately, by construction, without
anyone having to notice the count stopped moving.

Usage::

    python scripts/harness_digest.py --print
    python scripts/harness_digest.py --files
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PACKAGE_ROOT = REPO_ROOT / "src"

# The one module whose closure IS the harness. The differential is the measuring
# instrument; everything it reaches is part of the instrument by construction, and
# everything it does not reach cannot change a sweep number.
HARNESS_ROOT = SCRIPTS_DIR / "engine_transition_differential.py"

DIGEST_SCHEMA = "pokezero-harness-digest/1"


def _resolve(module_name: str) -> Path | None:
    """Map an ABSOLUTE dotted import name onto a tracked first-party file, or None.

    First party means exactly two layouts, which are the only two this repo has:
    a bare top-level name that is a sibling script (`scripts/` is prepended to
    `sys.path` by the differential), and a `pokezero.*` name under `src/`.
    Everything else — stdlib, third party, and the native engine — resolves to
    None and is not part of this digest.
    """

    parts = module_name.split(".")
    if parts[0] == "pokezero":
        base = PACKAGE_ROOT.joinpath(*parts)
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None
    if len(parts) == 1:
        candidate = SCRIPTS_DIR / f"{parts[0]}.py"
        if candidate.is_file():
            return candidate
    return None


def _containing_package(path: Path) -> str | None:
    """The dotted package a file lives in, or None if it is not under `src/`.

    `src/pokezero/engine_world.py` -> `pokezero`.
    `src/pokezero/__init__.py`     -> `pokezero` (a package's own `__init__` is
    INSIDE the package it names, which is what makes `from .x import y` there
    resolve to `pokezero.x` and not to `x`).
    """

    try:
        relative = path.resolve().relative_to(PACKAGE_ROOT)
    except ValueError:
        return None
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        return ".".join(parts[:-1])
    return ".".join(parts[:-1])


def _absolutize(node: "ast.ImportFrom", source: Path) -> list[str]:
    """Every absolute module name a single `from ... import ...` can denote.

    Handles BOTH forms, and the relative one is not optional. An earlier revision
    of this function dropped every `node.level > 0` import with a comment claiming
    relative imports "do not occur in either layout above". That was false and it
    was measurable: `src/pokezero/**` carries 70 relative `ImportFrom` statements
    across 9 of the files that were being hashed, and dropping them truncated the
    closure to 16 files where the honest answer is 73. `engine_world.py` reaches
    `gen3_damage.py` that way, `engine_fidelity.py` reaches `showdown_fixture.py`,
    and `poke_engine_adapter.py` reaches `poke_engine_backend.py` — all three on
    the live sweep path. The digest was therefore blind to the majority of the
    instrument while its docstring claimed the opposite. See
    `tests/test_harness_digest_provenance.py` for the pin that now makes a
    truncating resolver red rather than quietly green.

    `from .thing import name` is ambiguous between a module `pkg.thing` and an
    attribute of it, so both are returned and whichever resolves to a file wins.
    """

    if not node.level:
        if not node.module:
            return []
        return [node.module] + [f"{node.module}.{a.name}" for a in node.names]

    package = _containing_package(source)
    if package is None:
        # A relative import from outside `src/` — `scripts/` is a flat directory of
        # top-level modules, so this cannot resolve to a first-party file. Returning
        # nothing here is correct, and it is NOT the silent drop described above:
        # the completeness pin re-derives relative targets independently and would
        # flag it if such a file ever appeared.
        return []
    parts = package.split(".") if package else []
    # `level == 1` is "this package"; each extra level strips one more component.
    ascend = node.level - 1
    if ascend:
        if ascend > len(parts):
            return []
        parts = parts[: len(parts) - ascend]
    prefix = ".".join(parts + ([node.module] if node.module else []))
    if not prefix:
        return []
    return [prefix] + [f"{prefix}.{a.name}" for a in node.names]


def harness_files(root: Path | None = None) -> list[Path]:
    """Every first-party source file the harness transitively imports, sorted.

    `ast.walk` visits function bodies too, so an import deferred inside a function
    — the usual way a heavy dependency is loaded late — is still found. Only a
    COMPUTED module name escapes; see the module docstring for the measurement
    showing that set is currently empty.
    """

    start = (root or HARNESS_ROOT).resolve()
    if not start.is_file():
        raise FileNotFoundError(f"harness root is missing: {start}")

    seen: set[Path] = set()
    pending = [start]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            raise RuntimeError(f"cannot parse harness source {path}: {error}") from error
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = _absolutize(node, path)
            else:
                continue
            for name in names:
                resolved = _resolve(name)
                if resolved is not None:
                    pending.append(resolved.resolve())
    return sorted(seen)


def compute_harness_digest(root: Path | None = None) -> dict[str, object]:
    """A sha256 over the harness closure, by repo-relative path and content.

    Path-keyed as well as content-keyed so that MOVING a file changes the digest:
    two trees with the same bytes under different names are different instruments.
    Location independent otherwise — no absolute path enters the hash.
    """

    files = harness_files(root)
    digest = hashlib.sha256()
    digest.update(DIGEST_SCHEMA.encode())
    relative: list[str] = []
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        relative.append(rel)
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "schema": DIGEST_SCHEMA,
        "harness_digest": digest.hexdigest(),
        "root": HARNESS_ROOT.relative_to(REPO_ROOT).as_posix(),
        "count": len(relative),
        "files": relative,
    }


def harness_digest() -> str:
    """The digest alone, for stamping into provenance."""

    return str(compute_harness_digest()["harness_digest"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--print", dest="show", action="store_true", help="print the digest")
    group.add_argument("--files", action="store_true", help="print the full closure as JSON")
    args = ap.parse_args(argv)
    payload = compute_harness_digest()
    if args.files:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"{payload['harness_digest']}  ({payload['count']} harness sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
