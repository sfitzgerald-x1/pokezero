# C128 — Phase 2 decision: reject blanket enumeration; Pain Split was the whole of it

C116 items 6 and 7. Item 6 asks for enumerate-then-merge behind a flag, measured on dev residue,
matched count, mass gate and throughput. Item 7 asks for the branch to be taken and **written
down with its numbers**. This is that.

**Decision: branch (c) — reject.** But not for the reason the plan anticipated. Blanket
enumeration is not rejected because it costs too much; it is rejected because **it was not
needed**. Everything it bought is obtainable by adding one arm to a four-item predicate.

> **On the C116 citation.** The plan is not in this repository; it is provenance, never evidence.

## 1. The spike

The engine **already ships** enumerate-then-merge. `gen3/generate_instructions.rs:3117` gates it:

```rust
if branch_on_damage
    && choice.first_move
    && pending_hp_reading_move(defender_choice)
    && fixed_damage.is_none()
```

and the guarded block enumerates `for random in 85..=100` through `run_move` per roll at mass
`chance/16`, then merges with `combine_duplicate_instructions`. So the spike is not a
mechanism, it is a flag: `POKEZERO_ENUMERATE_ALL_ROLLS=1` ORs into the third clause. Off by
default, shipped behaviour bit-identical. Verified live — a first-mover Tackle goes **4 branches
→ 46**, an 11.5× fan-out.

## 2. The measurement

Holdout, 200 games, engine otherwise identical, enumeration **off → on**:

| | boundaries | matched | diverged | throughput |
|---|---|---|---|---|
| off | 15,396 | 15,386 | 10 | 1,427.8 games/h |
| on | 15,396 | **15,388** | **8** | **1,463.6 games/h** |

Closed `19100148/76` and `19100179/21`. Opened none. **No measurable throughput cost** — the
run was marginally faster, i.e. inside run-to-run variance.

## 3. Three of my own predictions were falsified, and that is the spike's value

**(a) "Three measured absorptions" was wrong where it counted.** `reports/c123` and the task
ledger recorded `19100113/62`, `19100107/135` and `19100191/5` as Phase 2 absorptions. The
damage arithmetic behind that is correct — the fan really does contain the needed roll — but
the gate has **four** clauses and I had widened one. **`choice.first_move` blocks all three
independently.** On `19100113/62` Registeel (speed 123) outspeeds Marowak (122) but is asleep,
so Bonemerang is the *second* move: branch count is 4 with the flag off and **4 with it on**.
All three are still divergent under enumeration. The correct earlier statement would have been
"the fan contains the needed roll", not "the row is absorbed".

**(b) The "~8x slower per call" cost was mine to misuse.** That figure is real and is in the
engine's own comment at `:2911-2916`, but it is a *per-call* figure, and I repeated it as though
it priced the workload. Measured at the differential's workload, the aggregate cost is zero.

**(c) I expected nothing else to move.** Two rows closed, and they were the informative ones.

## 4. What the two closed rows actually revealed

Both are Pain Split. Chasing why they closed found the defect:

```rust
fn pending_hp_reading_move(choice: &Choice) -> bool {
    matches!(
        choice.move_id,
        Choices::FLAIL | Choices::REVERSAL | Choices::SUBSTITUTE | Choices::BELLYDRUM
    )
}
```

**`PAINSPLIT` is missing.** Pain Split *averages the two actives' HP*, so the first move's damage
roll changes its result directly — it is as HP-reading as Flail. The list was simply incomplete.

Adding `| Choices::PAINSPLIT` and running with the flag **off**:

| window | boundaries | matched | diverged | closed | opened |
|---|---|---|---|---|---|
| dev | 15,224 | 15,222 | 2 → **2** | none | none |
| validation holdout | 15,396 | 15,386 → **15,388** | 10 → **8** | `19100148/76`, `19100179/21` | **none** |

**The same two rows, without enumerating anywhere else.** Dev is unchanged because it contains no
Pain Split row — the control that says the change touches only what it should.

## 5. Gates

| gate | result |
|---|---|
| `tests/test_poke_engine_patch_stack` | Ran 4, OK |
| `tests/test_engine_gen3_abilities` | Ran 50, OK |
| `tests/test_branch_mass_reconstruction` (mass gate) | Ran 5, OK |
| `tests/test_crit_kill_split_patch` | Ran 8, OK |
| `tests/test_a1_residuals_already_ran` | Ran 7, OK |
| `scripts/engine_behavioral_probes.py` | exit 0, all PASS |

## 6. The decision, and what it does not license

**(c) Reject blanket enumeration.** Land the targeted `PAINSPLIT` arm instead. The partition
stack, the residual mirror and the f32 comparator all stay — branch (a) would have deleted them,
and nothing here justifies that.

This closes M5 as the plan says all three branches do, and it does **not** license the following:

- It does **not** show enumeration is cheap in general. It shows the *current gate* rarely fires,
  so widening one of its four clauses changes little. Relaxing `first_move` too would enumerate on
  every damaging move and has **not** been measured.
- It does **not** absorb the three fan rows. They need `first_move` relaxed, which is a
  materially larger change with an unmeasured cost.
- The 11.5× per-call fan-out is real. It simply does not reach the aggregate at this workload.

**The spike is dropped, not shipped.** It was an instrument, and it has produced its measurement.
Keeping it would mean carrying `POKEZERO_ENUMERATE_ALL_ROLLS` in the engine with no CI step
exercising either arm — precisely the "a gate nothing runs" shape this program keeps finding in
review. What ships is `poke-engine-gen3-painsplit-hp-reading.patch`, a **single hunk** adding one
arm to the predicate. This report is the record of what the flag measured, and the spike patch is
reconstructible from §1 if the `first_move` question is ever taken up.
