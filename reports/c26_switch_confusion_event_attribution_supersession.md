# C26 Supersession: Switch-Prefixed Confusion Attribution

## Decision

The retired renderer stack is superseded. It must not be integrated with the
public implementation.

The retired implementation recognized a structural
`ChangeVolatileStatusDuration(confusion, +1), Damage(self)` pair and emitted
`|-damage|...|[from] confusion`. That tagged line is not the public contract:
the V2 freeze and V3 corrective path require an untagged self-hit damage line
after `|-activate|...|confusion`. A tag prevents the fold from recording the
carried self-hit fraction, which would silently defeat V3's correction.

Public merge `8af4f42e99ef9b6a0b809027976a27a8d135cd3c` (PR #993) supersedes
the mechanism. Its renderer consumes the pre-move confusion duration increment,
derives the Gen 3 fixed 40-power self-hit damage from live state, and emits the
untagged canonical pair only when that identity is unique. It rejects crash and
self-faint collisions rather than attributing them to confusion. Switch
handling completes before this common move phase.

## Evidence Limits

The C26 hardcoded identity ledger is a historical transcription, not
independent provenance. The repository retains no C26 row payload, checkpoint
JSONL shard, or C26 result artifact. C27 gives secondary historical
corroboration for `3001000/57` and `3300122/21` only; neither is a retained
replay payload. `3300017/60` and the previously reported magnitude `31` have
no retained in-repository payload. The verifier deliberately does not assert
any of these labels or preserve a fresh classifier outcome.

Consequently, this record does **not** claim a row replay, independent residual
clearance, binding certification clearance, or a fresh classifier result.

## Reproducible Check

Run from this worktree:

```bash
uv run --isolated --python 3.12 python scripts/verify_c26_switch_confusion_supersession.py
(cd rust/pokezero-search && cargo test --test gen3_confusion_event_renderer)
uv run --isolated --python 3.12 python tests/test_poke_engine_patch_stack.py
uv run --isolated --python 3.12 python tests/test_public_invariant.py
git diff --check
```

The checker separates two claims. It reads immutable historical provenance from
public merge `8af4f42` with `git show`, first force-refreshes `origin/main`
from the authoritative remote, and then proves that merge is an ancestor of the
freshly fetched commit. The same fetched commit is the base for the current
ancestry and public-input-diff checks, so a stale tracking ref cannot satisfy
either claim. It then validates the current tracked engine source pin and
patch-list digest, checks the vendored patch target digest, and runs the current
switch-prefixed Rust regression alongside the patch-stack and public-invariant
tests. Cargo output must contain the exact named regression with `... ok`, a
nonzero runnable-test count, zero ignored tests, and zero filtered-out tests.
The ordinary Recoil control remains in the same renderer integration suite.
None of these commands reruns a C26 classifier or clears certification.
