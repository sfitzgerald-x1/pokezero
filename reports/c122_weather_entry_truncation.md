# C122 — battle-end residuals truncate BETWEEN entries, not mid-entry

A cause found by the adversarial review of #1087, which noticed that the two holdout rows C120
ejected as "needing a different cause" already had one. They did, and it was a
self-contradiction inside a single merged patch.

> **On the C116 citation.** As in `reports/c121_a5_wake_before_contact.md`: the C116 refocus
> plan is deliberately not in this repository, so it is provenance for why work was queued and
> never evidence for a claim. Everything load-bearing below is a tracked artifact.

## 1. A patch that contradicts its own rationale

`third_party/poke-engine-gen3-battle-end-residuals.patch` states the Showdown rule exactly
right:

> "The boundary is exact and it is BETWEEN entries, never mid-entry: the faint-causing entry
> runs to completion, `faintMessages()` resolves the faint and `checkWin` sets `ended`, and
> every REMAINING handler is skipped."

Its MECHANISM section then installs the guard "at the top of each per-side residual section
body (nine of them)". For weather those two are incompatible. Sandstorm and hail are **one**
entry that chips **both** actives, so a Pokémon killed by its own chip was cancelling the other
active's chip inside the same entry.

## 2. The artifact settles it

Recorded Showdown protocol at holdout row `19100002/53`, read from
`reports/artifacts/c121_a5_holdout_sweep.json`:

```
|-weather|Sandstorm|[upkeep]
|-damage|p1a: Slaking|0 fnt|[from] Sandstorm
|-damage|p2a: Venusaur|161/262|[from] Sandstorm
|faint|p1a: Slaking
|win|PokeZero p2
```

Both chips land, **then** the faint resolves, **then** the battle ends. That is exactly the
`observed_only=[('sandstorm', -16)] engine_only=[]` miss recorded on `19100002/53` and
`19100154/75` — Showdown chipped the winner and we did not.

## 3. The fix, and what it deliberately does not touch

The guard is hoisted out of the weather loop, with one left before it.

That retained guard is **defence in depth, not the safety argument** — a distinction the review
of #1092 drew and it is right. The function-top guard already covers battle-over-before-residuals,
and nothing at residual orders 1-7 (Reflect, Light Screen, Mist, Safeguard, Wish) can reduce HP,
so no fixture reaches the retained one and no test constrains it. The real guarantee for a
battle that ended in the move phase is the function-top guard, and that path is confirmed by
behaviour: with the battle already over, the walk emits no `DecrementWeatherTurnsRemaining` at
all.

The **order 10 per-Pokémon guards are untouched**, and that is the substance of the change, not
an omission. That class really is one handler per Pokémon — the engine's own comment says
Showdown "runs `this.faintMessages(); if (this.ended) return;` after EVERY handler, not after
every section" — so `ended` really is checked between them. Only the shared weather entry was
mis-sliced. The change is a granularity correction at one site, not a relaxation of truncation.

## 4. The pin asserted the defect as the rule

This is why the two rows never looked like a cause: the behaviour was gated as intended.
`rust/pokezero-search/tests/gen3_battle_end_residuals.rs` read

> "Neither side may take a residual after the battle ends — not just the winner. Sand chips
> BOTH actives, so without truncation the winning side would tick too."

and asserted `damages(SideOne).is_empty()`. It now asserts the winner takes **exactly** its sand
chip — `vec![18]`, being `300/16` — and never reaches its Leftovers heal or its own burn tick,
which are later entries. So a hoist that went too far fails just as loudly as no hoist at all.

**Red run (M3).** Against the 59-patch engine: `left: []  right: [18]`, with the instruction
list showing only `[DecrementWeatherTurnsRemaining, Damage SideTwo: 1]` — the loser's chip
present, the winner's absent. The defect in one line.

## 5. Gates

| gate | result |
|---|---|
| `gen3_battle_end_residuals` crate suite | 6 passed, 0 failed — including the two sibling truncation pins, which still hold |
| `tests/test_poke_engine_patch_stack` | Ran 4, OK — tail pin grown to 5 names |
| `tests/test_branch_mass_reconstruction` | Ran 5, OK |
| `tests/test_crit_kill_split_patch` | Ran 8, OK |
| `tests/test_drag_limit_is_a_last_resort` | Ran 3, OK |
| `tests/test_engine_gen3_abilities` | Ran 46, OK |
| `scripts/engine_behavioral_probes.py` | exit 0, `all behavioral probes PASS` |
| full crate suite, `RUSTFLAGS="-C debug-assertions=yes"` | **370 passed across 32 binaries, 0 failed** |

## 6. Sweep

Prediction registered before the sweeps were read: **holdout 13 → 11** on `19100002/53` and
`19100154/75`, **dev 5 unchanged** — neither dev row is battle-end sand, so dev is the control
that says the hoist touched only what it should.

| window | boundaries | matched | diverged |
|---|---|---|---|
| dev — before | 15,224 | 15,219 | 5 |
| dev — after | 15,224 | 15,219 | **5 (unchanged)** |
| validation holdout — before | 15,396 | 15,383 | 13 |
| validation holdout — after | 15,396 | **15,385** | **11** |

Row level: holdout closed exactly `19100002/53` and `19100154/75`; **nothing opened in either
window**; dev closed nothing and opened nothing.

**One disclosure the control does not license.** Dev row `19000074/27` carries
`observed_only=[('sandstorm', -18)] engine_only=[]` — the *same miss signature* as the two rows
just closed — and it is byte-identical before and after. So "neither dev row is battle-end sand"
is literally true (that mon's side has four live reserves, `battle_is_over()` is 0, and the
guard never fired there even before this patch), but describing dev purely as "the control that
says the hoist touched only what it should" would leave a reader thinking the sand-miss class is
now empty. It is not. That row is a different defect wearing the same signature and it is
**not** closed by this change; it needs its own investigation, and the first thing to rule out
is a damage-magnitude or partition-arm divergence rather than truncation. Identity `matched + diverged == boundaries`
holds on all four rows. Artifacts committed as `reports/artifacts/c122_weather_{dev,holdout}_sweep.json`.

Residue is now **dev 5 / holdout 11**, reported as an outcome.

## 7. Note

Two of the last three causes closed came from reading a recorded protocol, not from reading
source. This one was invisible for as long as it was because a *pin* encoded the defect, so
every gate agreed with the bug — which is the failure mode a gate cannot catch by construction.
The check that would have found it earlier is comparing a patch's stated rule against its own
implementation, and that comparison was available the whole time in one file.

Which makes it worth recording that the first version of this very change shipped a fresh
doc/reality contradiction of its own: the pin comment in `tests/test_poke_engine_patch_stack.py`
still credited the previous patch for a digest this one moved, and still claimed
`generate_instructions.rs` was byte-identical to a pin it had just changed. Caught in review.
The rationale paragraph in `battle-end-residuals.patch` has been amended in place rather than
deleted, because it is the evidence for how the defect stayed invisible — a pin was written to
match it.
