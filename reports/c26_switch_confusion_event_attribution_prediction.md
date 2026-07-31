# C26 Prediction: Switch-Prefixed Confusion Self-Hit Attribution

## Scope

Current-source retained certification identities `3001000/57`, `3300017/60`,
and `3300122/21` show a voluntary switch followed by the opposing active's
confusion self-hit. The engine generates the no-status self-hit branch, but
the event mapper reaches its generic attacker-side damage fallback and labels
the branch `unattributed_self_damage`. Strict matching consequently rejects a
real confusion branch.

This is an event-rendering defect only. It does not change battle mechanics,
world construction, hidden-information handling, or damage arithmetic.

## Prediction

For a joint action in which one side voluntarily switches and the other side
is confused, the branch whose post-switch action is a confusion self-hit will
render the canonical existing protocol shape:

```text
|switch|<switching active>|...
|-activate|<confused active>|confusion
|-damage|<confused active>|<hp>/<maxhp>|[from] confusion
```

That branch will not contain `unattributed_self_damage`. The exact retained
`3300017/60` magnitude remains `31`; this change only restores the lost
provenance. The `3001000/57` and `3300122/21` identities may retain their
separate 2-HP and 1-HP damage-state residuals after matching, but their
confusion branch will no longer be discarded for missing attribution.

## Controls

1. A confused action with no opposing switch still renders the same canonical
   confusion activation and tagged damage, with no lossy self-damage label.
2. A non-confused action with attacker-side damage from an established source
   such as recoil remains attributed to that source, not confusion.
3. An unexplained non-confusion attacker-side damage remains fail-closed as
   `unattributed_self_damage`; the new recognition must not infer confusion
   from damage alone.
4. Ordinary opponent-directed damage retains its existing target and has no
   confusion activation or confusion damage tag.

## Verification Plan

Add native event-renderer coverage using the real engine instruction stream
for a switch-prefixed confusion branch, plus Python `branch_events` and fold
mapping coverage. Run focused Rust tests, native Python mapper/matcher tests,
the public-invariant suite, formatting/lint, and whitespace checks. A post-fix
certification classifier replay remains required before a binding sweep.
