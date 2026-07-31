# C26 Supersession: Switch-Prefixed Confusion Attribution

## Decision

The renderer stack in `3ec869c`/`f5b71a4`/`62947e9` is superseded and is
reverted in this lane. It should not be integrated with the active clean lane.

The retired implementation recognized a structural
`ChangeVolatileStatusDuration(confusion, +1), Damage(self)` pair and emitted
`|-damage|...|[from] confusion`. That tagged line is not the public contract:
the V2 freeze and V3 corrective path require an untagged self-hit damage line
after `|-activate|...|confusion`. A tag prevents the fold from recording the
carried self-hit fraction, which would silently defeat V3's correction.

Committed clean-lane revision `51faf308a8a3bb626b1c3d2e5b12b0491abaea5c`
supersedes the mechanism. Its renderer consumes the pre-move confusion duration
increment, derives the Gen 3 fixed 40-power self-hit damage from the live
state, and emits the untagged canonical pair only when that damage identity is
unique. It rejects crash and self-faint collisions rather than attributing them
to confusion. Switch handling has already completed before this common move
phase, so the classifier is independent of the preceding voluntary switch.

## Identity Ledger

The C26 prediction names these retained identities directly:

| Identity | Documented retained shape | Replacement handling |
| --- | --- | --- |
| `3001000/57` | Voluntary switch, opposing exact confusion self-hit | Post-switch exact classifier |
| `3300017/60` | Voluntary switch, opposing exact confusion self-hit; magnitude `31` | Post-switch exact classifier |
| `3300122/21` | Voluntary switch, opposing exact confusion self-hit | Post-switch exact classifier |

The repository retains no C26 row payload, checkpoint JSONL shard, or C26
result artifact. Consequently, this record does **not** claim a fresh
classifier outcome or that the independent 2-HP/1-HP residuals clear. It
proves the committed renderer-contract replacement for each explicitly named
identity without inventing unavailable inputs.

## Reproducible Check

Run from this worktree:

```bash
python3 scripts/verify_c26_switch_confusion_supersession.py \
  --candidate ../pokezero-confusion-before-substitute-clean
```

The checker reads the candidate commit with `git show`, so it does not read or
modify that worktree's uncommitted files. It verifies the exact identity ledger,
the retired tagged implementation, the clean lane's exact-damage and
fail-closed collision guards, and its native crash/self-faint controls.
