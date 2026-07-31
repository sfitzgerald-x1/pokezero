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

## Verification Lifecycle

Before merge, branch validation is deliberately limited to parser/unit and
current-artifact checks. These commands are required, but they are **not** a
C26 supersession-verifier PASS:

```bash
uv run --isolated --python 3.12 python tests/test_c26_switch_confusion_supersession.py
(cd rust/pokezero-search && cargo test --test gen3_confusion_event_renderer)
uv run --isolated --python 3.12 python tests/test_poke_engine_patch_stack.py
uv run --isolated --python 3.12 python tests/test_public_invariant.py
git diff --check
```

After the final merge commit is fetched to `origin/main` and checked out as
`HEAD`, the full verifier is required and expected to pass:

```bash
uv run --isolated --python 3.12 python scripts/verify_c26_switch_confusion_supersession.py
```

It force-refreshes `origin/main`, rejects any `HEAD` that is not exactly that
commit, and retains the exact public-input equality gate for `events.rs`,
`Cargo.toml`, the engine source pin, and patch list. Only then does it validate
the engine artifacts and rerun the eleven parser/unit tests, 22-test renderer
suite, patch-stack test, and public-invariant test. A feature branch that
changes those authenticated inputs is therefore expected to fail the full
verifier before merge, not to report a pre-merge PASS. The ordinary Recoil
control remains in the renderer suite. None of these commands reruns a C26
classifier or clears certification.
