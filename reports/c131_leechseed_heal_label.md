# C131 — the residual-heal fallback is ordered by phase, not by guess

C116 Phase 4 item 12: one more row disposed of, as a **harness fix**. Era: branch
`harness-leechseed-heal-label` off `main` `a4132d16`; engine+crate fingerprint `12e05f6e8a…` →
`15b9f280fc…` (`events.rs` is inside the fingerprint, so the crate change moves it even though the
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

The fix orders the fallback by the engine's own residual phase order: **Leftovers is 10.4 and the
drain is 10.5**, so Leftovers is tested first.

```rust
if s.get_active_immutable().item == Items::LEFTOVERS { return "item: Leftovers".to_string(); }
if opponent.volatile_statuses.contains(&PokemonVolatileStatus::LEECHSEED)
    && opponent.get_active_immutable().ability != Abilities::LIQUIDOOZE
{ return "Leech Seed".to_string(); }
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
| `a_seeded_opponent_does_not_steal_the_leftovers_tag` | **RED** (`events.rs:5930`) | green |
| control: `without_leftovers_a_seeded_opponent_still_yields_the_drain_label` | green | green |

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
| the two new pins | 3 ran, OK (2 pins + 1 pre-existing sibling) |

## 5. Sweep

| window | engine | measured | full_round | matched | diverged |
|---|---|---|---|---|---|
| dev `19,000,000–19,000,199` | main `12e05f6e8a…` | 15,432 | 15,968 | 15,430 | 2 |
| dev | branch `15b9f280fc…` | 15,432 | 15,968 | 15,430 | **2 (unchanged)** |
| validation holdout `19,100,000–19,100,199` | main `12e05f6e8a…` | 15,551 | 16,155 | 15,546 | 5 |
| validation holdout | branch `15b9f280fc…` | 15,551 | 16,155 | **15,547** | **4** |

**Closed exactly `19100193/46`. Nothing opened.** `boundaries_measured` and
`boundaries_full_round` are identical in both windows, identity holds on all four runs, and
`engine_errors` is 0 in all four. The `component_mismatch:itemleftovers|leechseed` class is gone.

The prediction registered before the sweep held in every part, **including its negative clause**:

> `19100014/35` does **NOT** close. Its other miss is a 10 % arm with
> `observed_only=[('leechseed', -33)] engine_only=[]` — the engine's Leech-Seed-**missed** branch,
> which this label fix does not touch. Predicting it closed would be predicting a fix I did not
> write.

It did not close, and its class is still `component_missing_in_engine:leechseed`. Naming in advance
what a fix will *not* do is the part of a prediction that can actually be wrong; this one was
registered with that clause precisely because the 90 % arm of that row *is* the same labelling bug
and it was tempting to claim both.

## 6. Residue after this

**dev 2 / holdout 4 — six rows, every one attributed:**

| row | cause | disposition |
|---|---|---|
| `19100014/35` | 90 % arm is the labelling bug (now fixed); the surviving 10 % arm is the engine's Leech-Seed-missed branch against a Showdown hit | open, distinct mechanism |
| `19100180/24` | hazard applied to the non-replacing side on a forced-replacement ply (B1) | open |
| `19100107/135`, `19100191/5` | `limit:roll_divergent_lethality` | already classed as limits by the harness |
| `19000191/63` | collapsed roll. The heal delta (28 vs 29) is **downstream and verified**: after a 109-vs-101 move roll Raichu sits at 14 vs 22, so `min(29, 14+14)=28` and `min(29, 22+14)=29` are both correct given their own HP | open |
| `19000074/27` | collapsed roll on the crit magnitude (93.75 % + 4.69 %), plus a 1.56 % crit-kill arm omitting the attacker's own sandstorm chip | open; the 1.56 % component is **reasoned** to be A1 residual placement and that attribution is **not yet measured** |
