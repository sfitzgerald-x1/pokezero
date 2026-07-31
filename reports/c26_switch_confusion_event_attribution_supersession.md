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
handling has already completed before this common move phase, so the classifier
is independent of the preceding voluntary switch.

## Identity Ledger

The C26 prediction names these retained identities directly:

| Identity | Documented retained shape | Replacement handling |
| --- | --- | --- |
| `3001000/57` | Voluntary switch, opposing exact confusion self-hit | Post-switch exact classifier |
| `3300017/60` | Voluntary switch, opposing exact confusion self-hit; magnitude `31` | Post-switch exact classifier |
| `3300122/21` | Voluntary switch, opposing exact confusion self-hit | Post-switch exact classifier |

The repository retains no C26 row payload, checkpoint JSONL shard, or C26
result artifact. Consequently, this record does **not** claim a fresh
classifier outcome or that independent residuals clear. It proves the final
public renderer-contract replacement for each explicitly named identity without
inventing unavailable inputs.

## Reproducible Check

Run from this worktree:

```bash
python3 scripts/verify_c26_switch_confusion_supersession.py
```

The checker reads only immutable files from public merge `8af4f42` with
`git show`; it does not read another worktree or modify files. It verifies the
exact identity ledger, final exact-damage and untagged-emission behavior,
fail-closed crash/self-faint controls, and the native Recoil ordinary-damage
negative control.
