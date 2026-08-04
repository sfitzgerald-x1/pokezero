# C118 — A5's site was wrong AND its mechanism is refuted; the rows are unattributed

C116 Phase 3 item 8 begins "reorder contact-ability trigger vs. same-turn wake **at the
pinned lines**". This is that re-pinning, and it changes the diagnosis. Nothing is fixed
here, deliberately.

Era: `main` `012d8451`, engine fingerprint from the build stamp (58 patches).

## What C111 recorded

> **A5** — contact-ability trigger precedes the same-turn wake:
> `ability_modify_attack_against` at `:2682` runs before the wake `ChangeStatus` at
> `:2694`, so `contact_status_is_valid` sees SLEEP.

Two rows depend on it: `19000125/226` (dev) and `19100012/61` (holdout, filed A5 in C117
§4 on the Wrap + Poison Point signature).

## What the source says now

The line numbers moved, which is expected and not the finding. The **hook** is different,
which is:

| thing | where it actually is |
|---|---|
| `contact_status_is_valid` | `gen3/abilities.rs:22`, refuses when `target.status != PokemonStatus::NONE` (`:29-33`) |
| Poison Point's arm | `gen3/abilities.rs:672-678`, inside **`ability_after_damage_hit`** (`:174`) |
| Effect Spore / Flame Body / Static arms | `abilities.rs:710`, `:750`, `:784` — same function |
| `ability_after_damage_hit` call site | `gen3/generate_instructions.rs:1881` |
| `ability_modify_attack_against` | called from `before_move` at `:2103` — an attack **modifier** (this is the Thick Fat hook), not the contact-secondary hook |
| the wake `ChangeStatus` | generated in `generate_instructions_from_existing_status_conditions` (`:2155`), called at `:2832` |

**So C111 pinned the wrong hook.** `ability_modify_attack_against` is where Thick Fat
halves `base_power` — the C100/C102 defect — and it has nothing to do with applying a
contact secondary. The contact secondary fires from `ability_after_damage_hit`. A fix
written "at the pinned lines" would have reordered the attack-modifier hook, which cannot
affect `contact_status_is_valid` at all.

## Why the mechanism looked survivable — and then did not

**This section is superseded by the one below it, and is kept as the reasoning that was
wrong.** On finding the wrong hook I concluded the *mechanism* still held: that
`contact_status_is_valid` refuses when the target already has a status, that in both rows the
secondary's target had just woken, and that a wake not yet applied would produce exactly the
observed absence. I listed two things as unestablished — the runtime order of `:1881` against
`:2832`, and whether the check reads applied state or the instruction list — and said they
needed an instrumented run.

They needed no run. The replay had the answer already.

## The ordering question is settled, and it refutes the mechanism

The two open questions needed no instrumentation. The replay prints instructions in
**applied** order, which is exactly what `contact_status_is_valid` reads. Seed 19000125
step 226, highest-probability branch:

```
Damage SideTwo: 16                        <- p1 Nidoqueen's Sludge Bomb hits Shuckle
ChangeStatus SideTwo-P0: SLEEP -> NONE    <- Shuckle WAKES, here
DecrementRestTurns SideTwo
SetLastUsedMove SideTwo: Move(M0) -> Move(M2)
Damage SideOne: 5                         <- Shuckle's Wrap contacts Nidoqueen
ApplyVolatileStatus SideOne: PARTIALLYTRAPPED
```

**The wake precedes the Wrap.** When `ability_after_damage_hit` evaluates Poison Point on
that contact, the secondary's target (Shuckle) already has `status == NONE`, so
`contact_status_is_valid` returns **true**, not false.

**C111's A5 mechanism is refuted.** The engine does not fail to poison because it thinks the
attacker is asleep. It simply does not poison: no `ChangeStatus … -> POISON` appears in the
branch, and the three arms C111 recorded (14.06 / 74.71 / 4.98 = 93.75%) carry no `psn` on
either side, with 6.25% of mass unaccounted for in that listing.

Untested candidates, recorded so the next attempt does not restart from zero: Poison Point's
activation probability and how its arms are enumerated; whether Wrap's contact flag reaches
this hook at all (Wrap is `contact: true` in gen3, verified during the c103–c109 landing, so
this is plumbing rather than data); and whether a partial-trap move's **first** turn routes
through `ability_after_damage_hit`.

## Disposition

**A5 is withdrawn as a cause. `19000125/226` and `19100012/61` are UNATTRIBUTED** and return
to the queue as such. C111's A5 entry and C117 §4's A5 filing both need amending, in separate
PRs — C111 owns the cause namespace, C117 owns the holdout row table.

Accounting consequences, stated because they move against the program's headline: the dev
window's 6 rows now carry **five** named causes and one unattributed; C117's "15 of 25 on
C111's six causes" becomes **14 of 25**; and A5 has **zero** rows in either window. C117's
"the number of causes rose by three" survives, because A5 was an existing cause being reused
rather than a new one.

**Do not write a fix for A5.** There is currently no mechanism to fix.

## A note on how this was found

Phase 3 item 8 looked like the cheapest remaining item — two rows, "site pinned to two
lines". It was cheap only because the pin was wrong, and the pin was wrong because C111
recorded a hook name from a plausible reading rather than from following the call. That is
the same failure mode as filing rows by class name, one layer down: filing a *cause* by
function name.
