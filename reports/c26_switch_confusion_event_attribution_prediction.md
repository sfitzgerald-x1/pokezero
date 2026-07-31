# C26 Prediction: Switch-Prefixed Confusion Self-Hit Attribution

## Scope

The retained certification identities below show a voluntary switch followed
by the opposing active's confusion self-hit. This document is an evidence
record for the final public merge, not a request to restore the retired
feature-lane implementation.

| Identity | Retained shape | Public-merge handling |
| --- | --- | --- |
| `3001000/57` | Voluntary switch, then opposing exact confusion self-hit | Post-switch exact classifier |
| `3300017/60` | Voluntary switch, then opposing exact confusion self-hit; magnitude `31` | Post-switch exact classifier |
| `3300122/21` | Voluntary switch, then opposing exact confusion self-hit | Post-switch exact classifier |

This is an event-rendering evidence record only. It does not change battle
mechanics, world construction, hidden-information handling, or damage
arithmetic.

## Prediction

For a joint action in which one side voluntarily switches and the other side
is confused, public merge `8af4f42e99ef9b6a0b809027976a27a8d135cd3c` renders
the exact post-switch self-hit branch with the canonical protocol shape:

```text
|switch|<switching active>|...
|-activate|<confused active>|confusion
|-damage|<confused active>|<hp>/<maxhp>
```

The self-hit damage is deliberately untagged: the fold records it in the
active confusion move window, preserving the V2 freeze and V3 correction. The
public classifier derives Gen 3's fixed 40-power self-hit damage from live
state and fails closed when crash or self-faint damage can share the same
delta. The retained identities name no replay payload, so this is a renderer
contract proof, not a new classifier result or a claim that separate residuals
clear.

## Controls

1. Exact confusion self-hit emits activation plus untagged damage.
2. Crash and self-faint collisions fail closed rather than inventing confusion.
3. Ordinary executed-move recoil remains explicitly attributed `[from] Recoil`,
   not to confusion.

## Verification Plan

The dedicated verifier reads the immutable public merge with `git show` and
checks the identity ledger, exact classifier, untagged emission, collision
controls, and ordinary-damage negative control. A post-fix certification
classifier replay remains required before any binding sweep claim.
