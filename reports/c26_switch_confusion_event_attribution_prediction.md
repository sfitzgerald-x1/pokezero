# C26 Prediction: Switch-Prefixed Confusion Self-Hit Attribution

## Scope

This document records the renderer contract for the final public merge, not a
request to restore the retired feature-lane implementation. C26's hardcoded
identity ledger is historical transcription, not independent provenance: this
repository retains no C26 row payload, checkpoint JSONL shard, or classifier
result artifact.

`3001000/57` and `3300122/21` have secondary historical corroboration in the
C27 damage-tail record, where each is labelled `confusion self-hit`. That C27
record is not a retained replay payload. `3300017/60`, including the previously
reported magnitude `31`, has no retained in-repository payload and is not used
as evidence by the verifier.

This is an event-rendering evidence record only. It does not change battle
mechanics, world construction, hidden-information handling, or damage
arithmetic.

## Prediction

Immutable historical proof: public merge
`8af4f42e99ef9b6a0b809027976a27a8d135cd3c` contains the exact-damage,
fail-closed renderer mechanism. The verifier proves that this merge remains an
ancestor of `origin/main` and reads its original sources with `git show`.

Current regression proof: the checked-out renderer has a native integration
test that generates one voluntary switch against an opposing exact confusion
self-hit and requires this canonical protocol shape from that one rendered
branch:

```text
|switch|<switching active>|...
|-activate|<confused active>|confusion
|-damage|<confused active>|<hp>/<maxhp>
```

The regression asserts the switch line, confusion activation, the exact
untagged damage line, and empty `attribution_unsafe` and lossy/fallback output.
The self-hit damage is deliberately untagged so the fold records it in the
active confusion move window. This is a renderer-contract regression, not a
fresh classifier result, a row replay, or evidence that any residual clears.

## Controls

1. Exact confusion self-hit emits activation plus untagged damage.
2. Crash and self-faint collisions fail closed rather than inventing confusion.
3. Ordinary executed-move recoil remains explicitly attributed `[from] Recoil`,
   not to confusion.

## Exact Commands

Run from this worktree:

```bash
uv run --isolated --python 3.12 python scripts/verify_c26_switch_confusion_supersession.py
(cd rust/pokezero-search && cargo test --test gen3_confusion_event_renderer)
uv run --isolated --python 3.12 python tests/test_poke_engine_patch_stack.py
uv run --isolated --python 3.12 python tests/test_public_invariant.py
git diff --check
```

The verifier runs the complete renderer integration suite, including the switch
regression and Recoil negative control, plus the patch-stack and public-invariant
tests after checking the pinned engine source and patch-list digest. A post-fix
certification classifier replay with retained inputs remains required before any
binding certification or sweep claim.
