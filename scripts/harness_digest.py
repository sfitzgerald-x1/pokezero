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

WHAT IT COVERS. The static first-party import closure of
`scripts/engine_transition_differential.py`, resolved by AST from the filesystem —
`scripts/<mod>.py` for top-level imports and `src/pokezero/<...>.py` for
`pokezero.*`. Self-maintaining: a new import into the harness enters the closure
automatically, and `tests/test_harness_digest_provenance.py` pins the exact
membership so one silently LEAVING is red (a vanishing member is this shape's
fail-open, and pure addition masks it). At the time of writing the closure is 16
files and includes the world model (`engine_world.py`), the matcher
(`engine_fidelity.py`, `engine_fidelity_multiturn.py`) and the Showdown adapter
(`local_showdown.py`, `poke_engine_adapter.py`) — the three named halves of the
instrument. It includes this module, because the differential imports it, so
tampering with the digest moves the digest.

WHAT IT DOES NOT COVER, stated as narrowly as it was measured:

  * The native engine. Deliberate — that is the engine fingerprint's job, and the
    two are recorded side by side rather than merged.
  * Anything reached by a COMPUTED import name. `__import__`/`importlib` with a
    non-literal argument is invisible to this resolver. Measured, not assumed:
    grepping `importlib|__import__` across the closure returns exactly three hits
    — two `__import__(name)` calls in `engine_build_fingerprint.py` over the
    literal tuple `("poke_engine", "pokezero_search")`, both of which are the
    native engine and therefore covered by the engine fingerprint, and one comment
    in the differential. So the gap is empty as of this commit; it is not
    structurally closed.
  * Third-party dependency versions, the Showdown checkout, and the interpreter.
    Those are runtime environment, not tracked bytes.
  * `tests/`, `docs/`, `reports/`, and every `scripts/` tool the differential does
    not import. Editing them cannot change a sweep number and must not move the
    digest.

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
    """Map a dotted import name onto a tracked first-party file, or None.

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
                if node.level or not node.module:
                    # Relative imports do not occur in either layout above; if one
                    # ever does, it is a resolver gap and the membership pin in
                    # tests/test_harness_digest_provenance.py will show it as a
                    # closure that stopped growing.
                    continue
                # `from pkg import thing` — `thing` may itself be a submodule.
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
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
