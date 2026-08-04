# C118 — A5's site is not where C111 pinned it, and the fix must not be written yet

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

## Why this is a re-pin and not a re-diagnosis

The *mechanism* C111 describes is still the best candidate: `contact_status_is_valid`
refuses outright when the target already has a status, and in both rows the target of the
secondary is the **attacker**, which had just woken (`-curestatus …|slp|[msg]` immediately
before its move in each protocol). So a wake not yet applied when the secondary is
evaluated would produce exactly the observed absence.

What is **not** established, and must be before any code changes:

1. **The runtime order of `:1881` against `:2832`.** They sit in different functions, so
   source order proves nothing. `:2832` is in `generate_instructions_from_move`; `:1881` is
   in the damage-application path it calls. If the wake is generated *before* `run_move`
   is entered, the attacker is already `NONE` when the secondary is evaluated and this
   mechanism is refuted — in which case the rows have a different cause.
2. **Whether the engine reads the wake at all.** `contact_status_is_valid` reads
   `target.status` from live state, not from the instruction list, so what matters is
   whether the `ChangeStatus` has been *applied*, not whether it has been *generated*.

Both are one instrumented run to settle. I did not run it, and I am recording the gap
rather than reasoning across it, because reasoning across exactly this kind of gap is what
produced C117's three misfiled A2 rows — a cause whose fix had already shipped.

## Disposition

**A5 stays open, with its site corrected and its mechanism downgraded from established to
candidate.** The next action is instrumentation, not a patch:

- print the applied-state `status` of the secondary's target at the moment
  `contact_status_is_valid` is called, on seed 19000125 step 226;
- if it reads SLEEP, the mechanism is confirmed and the fix is an ordering change whose
  blast radius is `generate_instructions_from_existing_status_conditions` — wide, and worth
  a dedicated PR;
- if it reads NONE, A5 is refuted for these rows and both need re-attribution.

**Do not write the fix from this report.** It names where to look, not what to change.

## A note on how this was found

Phase 3 item 8 looked like the cheapest remaining item — two rows, "site pinned to two
lines". It was cheap only because the pin was wrong, and the pin was wrong because C111
recorded a hook name from a plausible reading rather than from following the call. That is
the same failure mode as filing rows by class name, one layer down: filing a *cause* by
function name.
