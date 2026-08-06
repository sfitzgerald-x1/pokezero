# C131 — Sand Veil was the plan's real failure; the fallback reorder was half a fix

> **This report's first revision named the wrong mechanism and claimed a row was fixed that its own
> committed artifact showed was not. Both are corrected below rather than quietly amended.** The
> reorder is kept — it is correct and monotone — but the row `19100014/35` closes because of a
> **missing Sand Veil gate**, found by the review of #1120, not because of the ordering.

C116 Phase 4 item 12: one more row disposed of, as a **harness fix**. Era: branch
`harness-leechseed-heal-label` off `main` `a4132d16`; engine+crate fingerprint `12e05f6e8a…` →
`7f0e61be89…` (`events.rs` is inside the fingerprint, so the crate change moves it even though the
patch stack does not).

> **On the C116 citation.** The plan lives outside this repository at the owner's instruction and is
> not verifiable by a future reader. Read "Phase 4 item 12" as provenance for *why this was queued*,
> never as evidence for a claim about the engine.

## 1. The defect is a LABEL, not an HP value

`rust/pokezero-search/src/events.rs` `residual_heal_cause` is the fallback used when
`ResidualPlan` — the structure that knows the speed order and can therefore attribute a cross-side
drain heal — has been discarded. The fallback tested the **Leech Seed drain before Leftovers**, so
an ordinary Leftovers tick on a side whose opponent happened to be seeded came back labelled
`leechseed`. That is the H.1 fall-through the plan exists to prevent.

The fallback now tests Leftovers first. **The phase-order reasoning an earlier revision of this
section gave for that is retracted — see §5.** `residual_heal_cause` takes no heal index, so it is a
constant function of state and cannot implement "answer with the earlier phase" at all; and the
premise is false anyway, because a faster victim's drain precedes the seeder's Leftovers tick. The
narrow true reason is that the drain is rendered *silently*, so `"Leech Seed"` is never a correct
answer for a `[from]`-tagged heal and nothing is displaced by preferring Leftovers.

```rust
if s.get_active_immutable().item == Items::LEFTOVERS { return "item: Leftovers".to_string(); }
if opponent.volatile_statuses.contains(&PokemonVolatileStatus::LEECHSEED)
    && opponent.get_active_immutable().ability != Abilities::LIQUIDOOZE
{ return String::new(); }   // renders `[silent]`, which is what Showdown emits
```

## 2. The row, and why it is only a label

`19100193/46`, `component_mismatch:itemleftovers|leechseed`, 100 % mass:

```
observed_only=[('itemleftovers', 18)]   engine_only=[('leechseed', 18)]
```

Cacturne has 290 maxhp, and `290 / 16 = 18` — a Leftovers tick. A *real* drain from Miltank would
be `273 / 8 = 34`, a different number. Walked step by step from the recorded protocol: Cacturne is
the seeder, and it **dies to poison at phase 10.6 before Miltank's 10.5 sap**, so no drain occurs
at all and both sides agree on every HP value. `ResidualPlan` had nonetheless reserved a drain slot
(the opponent was seeded and both were alive when the plan was built), came up one heal short, was
discarded, and the fallback then mislabelled the tick.

So this row was never an engine defect: the engine's HP arithmetic was right and its **attribution
string** was wrong.

## 3. Pins, and which one is the control

| pin | on `main`'s ordering | on this branch |
|---|---|---|
| `a_seeded_opponent_does_not_steal_the_leftovers_tag` | **RED** | green |
| `without_leftovers_a_seeded_opponent_still_yields_the_drain_label` | **RED** | green |
| `liquid_ooze_on_the_seeder_means_a_heal_here_is_not_the_drain` | n/a (new) | green; red when the guard is deleted |
| `sand_veil_is_exempt_so_the_plan_does_not_book_a_chip_that_never_fires` | n/a (new) | green; **red** when the exemption is deleted |
| `expiring_weather_books_no_chip_so_the_drain_keeps_its_label` | n/a (new) | green; **red** when the expiry gate is deleted |

> **`without_leftovers_…` is NOT a control, and calling it one was wrong.** Once its assertion was
> flipped to `[silent]` it became a second *regression* pin: on `main` the fallback returns
> `"Leech Seed"`, so the line has no `[silent]` and the pin is **RED**. An earlier revision of this
> table recorded it as green in both eras — I flipped the assertion and did not re-run the revert
> check, then reported a result that no longer held. Verified by restoring `main`'s drain return:
> that pin, and only that pin, fails.
>
> The LIQUID OOZE guard was **unpinned** — the review deleted it and the whole crate suite stayed
> green — and so was the **Sand Veil exemption**, the one line worth two rows.
>
> **Both of my first Sand Veil and expiry pins were vacuous, in the same commit, for the same
> reason.** They asserted on DAMAGE tags, and an unfilled chip slot does not corrupt damage
> attribution — it corrupts the HEAL labels, because that is where the fallback has to guess between
> Leftovers and the cross-side drain. Deleting the exemption left both green. Rewritten on heal
> labels, both are now verified red against deleting their own line.

The revert check was run properly rather than assumed: `events.rs` was restored to `main` and the
two pin functions **grafted back on**, so the pins ran against the old ordering with nothing else
changed. One failed, one passed. Reverting the whole file would have deleted the pins along with
the fix and "passed" vacuously — the same shape that has cost this program a red gate before.

The control matters because the naive fix is to test Leftovers first and stop. It pins that a
seeded opponent **without** Leftovers still yields the drain label, so the reorder did not simply
starve the Leech Seed branch.

## 4. Gates

| gate | result |
|---|---|
| crate suite, `RUSTFLAGS="-C debug-assertions=yes"` | 0 failed, 32 test groups ok |
| the new pins | 5, all OK; each verified red against deleting the line it covers |

## 5. The mechanism I got wrong, and the one that was actually load-bearing

`events.rs` `weather_chips` did not model the engine's **Sand Veil** exemption
(`gen3/generate_instructions.rs:4223` skips the sand chip for `Abilities::SANDVEIL`). `SANDVEIL`
appeared **zero times** in `events.rs`.

Cacturne has Sand Veil. So `ResidualPlan` booked a sandstorm chip that never fired; **one unfilled
slot makes the whole side's plan unusable**; and every heal on that side then fell through to
`residual_heal_cause`. That fallback takes `(state, side, next_ins)` and **no heal index**, so it is
a *constant function of state* — it cannot label two heals on one side differently. Reordering it
could therefore only ever fix whichever of Cacturne's two heals happened to be the Leftovers tick.

The gate is one line, and with it the plan reconciles and **both** arms close:

```rust
|| active.ability == Abilities::SANDVEIL
```

Two further corrections the review forced, both verified:

- **`"Leech Seed"` was never a legal answer.** Showdown renders the drain heal silently:
  `sim/battle.ts:2293-2296` switches on `effect.id`, and `case 'leechseed'` emits
  `('-heal', target, getHealth, '[silent]')`, reached from `data/moves.ts:10218-10221`. There is no
  `[from] Leech Seed` heal line anywhere in Showdown. The fallback now returns `String::new()`,
  agreeing with what `ResidualPlan` already did for its own drain slot.
- **§3's control pin was asserting that non-existent label**, so it enshrined a wrong output and
  blocked the right one. The review built the correct fix and found this pin was the *only* thing
  failing, at zero measured cost. It now asserts `[silent]`.

`ResidualPlan` books no slot for **Rain Dish** or **Sitrus** either (`RAINDISH`/`SITRUS` also grep to
zero in `events.rs`) — but **both are UNREACHABLE in the gen3 randbats pool**, so neither can cost a
row, and an earlier revision of this line overstated them by omitting that check. Verified against
the live checkout: `data/random-battles/gen3/sets.json` has **zero** sets with Rain Dish (or Dry
Skin, Overcoat, Ice Body) across its 220 species, and `teams.ts:452-513` `getItem` can only return
Stick, Soul Dew, Silk Scarf, Thick Club, Light Ball, Lum Berry, Choice Band, Twisted Spoon, White
Herb, Salac/Liechi/Petaya Berry or Leftovers — **Sitrus Berry cannot be held**. The repo already
applies exactly this test to DRYSKIN at `gen3/generate_instructions.rs:1567`, so the instrument was
available and I skipped it.

**The reachable member of the same class is weather EXPIRY, and it is fixed here.**
`weather_is_active` ignores `turns_remaining` (`gen3/state.rs:1050-1060`) while the engine decrements
and clears the weather at `generate_instructions.rs:4144-4163` *before* its chip loop at `:4193`. So
on the turn sand or hail expires the engine emits no chip, the plan books one, and the side drops to
the constant fallback — the `19100193/46` signature. Measured: without the gate that state yields
`["item: Leftovers", "item: Leftovers"]`, the genuine drain mislabelled as a second Leftovers tick;
with it the drain renders `[silent]`. Exactly `== 1`, not `<= 1`: the engine's decrement is gated on
`turns_remaining > 0`, so a pre-residual 0 means *permanent* weather, which does keep chipping.

## 6. Sweep

| window | engine | measured | full_round | matched | diverged |
|---|---|---|---|---|---|
| dev `19,000,000–19,000,199` | `main` `a4132d16`, fp `12e05f6e8a…` | 15,432 | 15,968 | 15,430 | 2 |
| dev | branch, fp `7f0e61be89…` | 15,432 | 15,968 | 15,430 | **2 (unchanged)** |
| validation holdout `19,100,000–19,100,199` | `main` `a4132d16`, fp `12e05f6e8a…` | 15,551 | 16,155 | 15,546 | 5 |
| validation holdout | branch, fp `7f0e61be89…` | 15,551 | 16,155 | **15,548** | **3** |

**Closed `19100014/35` and `19100193/46`. Nothing opened.** `boundaries_measured` and
`boundaries_full_round` identical, identity holds on all four, `engine_errors` 0 in all four.

> **The baseline artifacts were re-run twice for provenance, and the reason is worth recording.** The
> first pair stamped `source_commit: 7762a81d` (66 patches, fingerprint `599c68a31e…`) against
> `engine_fingerprint: 12e05f6e8a…` (67 patches, `a4132d16`) — a **false machine-readable pin**, since
> the recorded commit did not describe the tree that produced the run. A second attempt, reverting
> only `events.rs` in the worktree, stamped the *branch* commit against main's fingerprint: consistent
> numbers, still a mismatched pin. The committed baselines are now taken from a **clean `main`
> checkout**, so `source_commit` and `engine_fingerprint` agree. Prose explaining a wrong pin does not
> fix it; validation reads the field.

## 7. What the prediction got right, and what it got wrong

The registered prediction said `19100014/35` would **not** close, and under that patch it did not.
**But its stated reason was wrong, so it does not count as holding.** It said the surviving miss was
"a different mechanism this fix does not touch". In fact one matching branch suffices for the
transition, the 90 % arm was closable all along, and closing it closes the row — which a one-line
Sand Veil gate then did.

The earlier revision of this report also asserted that the 90 % arm was "now fixed". **Its own
committed artifact contradicted that**: two misses survived, the 90 % arm having merely traded
`engine_only=[('leechseed', 33)]` for `engine_only=[('itemleftovers', 33)]` — two wrong components
became one wrong component. I wrote "fixed" without reading the file I had just committed, which is
the same failure as the baseline error two reports earlier: the artifact was sitting there.

Being right about an outcome for the wrong reason is not a prediction holding, and a negative clause
is exactly where that distinction bites — it is the clause that *looks* like rigour.

## 8. Residue after this

**dev 2 / holdout 3 — five rows:**

| row | cause | disposition |
|---|---|---|
| `19100107/135`, `19100191/5` | `limit:roll_divergent_lethality` | **already classed as limits by the harness** |
| `19100180/24` | hazard applied to the non-replacing side on a forced-replacement ply (B1) | open |
| `19000191/63` | collapsed roll. The heal delta (28 vs 29) is **downstream and verified**: after a 109-vs-101 move roll Raichu sits at 14 vs 22, so `min(29, 14+14)=28` and `min(29, 22+14)=29` are both correct given their own HP | open |
| `19000074/27` | collapsed roll on the crit magnitude (93.75 % + 4.69 %), plus a 1.56 % crit-kill arm omitting the attacker's own sandstorm chip | open; the 1.56 % component is candidate **A12** — the engine skips the *whole* residual phase on a move-caused faint, which is measured, while whether Showdown agrees is **not** |

Also filed, not fixed: `ResidualPlan` books no Rain Dish or Sitrus slot; a pre-existing red in
`tests/test_engine_terminal_residual_roll_limit` that fails identically on `main`.
