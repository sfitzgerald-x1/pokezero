# C121 — A5 closed: a contact secondary now sees the attacker wake on the same turn

C116 Phase 3 item 8. C118 v2 confirmed A5 by measurement and re-pointed it at the correct
site; this report is the fix, its red run, its gate, and its sweep.

> **On the C116 citation.** The C116 refocus plan is deliberately **not in this repository** --
> it lives at `agents/reports/rust-fidelity/c116_refocus_plan.md` outside the tree, at the
> repository owner's instruction, and is therefore not verifiable by a future reader of this
> file. Every claim below that matters is instead grounded in something tracked: the merged
> reports `reports/c118_a5_site_repin.md` and `reports/c117_validation_holdout_baseline.md`,
> the patch stack, the pins, and the sweep artifacts. Read "C116 Phase 3 item 8" as provenance
> for *why this work was queued*, never as evidence for any claim about the engine. A review of
> a sibling report found this same untracked citation used 8x as if it were falsifiable; the
> honest fix is to say so, not to keep citing it.

Era: branch `fidelity-a5-wake-before-contact` off `main`, patch stack grown 58 → 59.

## 1. The defect

`gen3/abilities.rs::ability_modify_attack_against` decides the contact secondaries — Poison
Point, Effect Spore, Flame Body, Static — and `gen3/generate_instructions.rs` reaches it from
`before_move`, **one call before**
`generate_instructions_from_existing_status_conditions` resolves the attacking mon's own sleep
or freeze. So `contact_status_is_valid` tested

```rust
|| target.status != PokemonStatus::NONE
```

against a status that was about to be cured, and refused the secondary outright. A Pokémon
that woke up and attacked on the same turn could never be poisoned by the Pokémon it hit.

Showdown does not have this ordering. Verified in the vendored gen3 mod, not inferred:

| claim | source | what it says |
|---|---|---|
| the secondary is decided *after* damage | `data/mods/gen3/abilities.ts:135-144` | `poisonpoint` overrides `onDamagingHit`, `randomChance(1, 3)` — matching the engine's `100.0/3.0`, and firing after the hit |
| the wake happens *in* BeforeMove and the move continues | `data/mods/gen3/conditions.ts` `slp.onBeforeMove` (priority 10) | on `time <= 0` it calls `pokemon.cureStatus(); return;` — undefined, not `false`, so the event continues and the move runs statusless |
| Sleep Talk is the exception | same handler | `if (move.sleepUsable) { skippedTime++; return; }` — announces `|cant|slp` and returns undefined, so the move runs **while still asleep** |
| freeze thaws and moves | `data/conditions.ts` `frz.onBeforeMove` (priority 10) | `randomChance(1, 5)` then `cureStatus(); return;` |

So at `onDamagingHit` Showdown's attacker is plainly statusless, and `trySetStatus('psn')`
succeeds where ours refused.

## 2. Why a predicate change and not a reorder

Reordering the two calls is the other repair and it is far wider: it moves every
`item_before_move` / `choice_before_move` instruction relative to every sleep, freeze and
paralysis branch in the game.

Ignoring an about-to-be-cured status is **exact rather than approximate**, because every
branch that keeps the status is a branch on which the move never lands:

- natural sleep and freeze clone their "no move" arm into `final_instructions` as a terminal
  branch and continue as the woke/thawed branch, where the status is already `NONE`;
- the Rest arms that fall through *still asleep* (`rest_turns` 2 or 3, which decrement and do
  not push) are caught by the `attacker_status == PokemonStatus::SLEEP && !choice.sleep_talk_move`
  gate later in `generate_instructions`, which reverses the instructions and returns;
- at `chance_to_wake == 0.0` with a non-Sleep-Talk move the continuing branch is left at 0%
  mass.

The one exception is a move that lands a hit *while* its user is asleep — Sleep Talk itself
and the move Sleep Talk calls, both of which that gate lets past. Those keep refusing, which
is what Showdown does too: their user really does hold a status at `onDamagingHit`.

PARALYZE, BURN, POISON and TOXIC are untouched. Nothing cures them between the check and the
hit, so `status != NONE` stays correct for them.

## 3. What this does NOT claim

C118 v2 said the fix should generalise "across Poison Point, Effect Spore, Flame Body, Static
and Cute Charm". That list of five was wrong by one:

- **Cute Charm is not affected.** It gates on `volatile_status_can_be_applied` for the ATTRACT
  *volatile*, never on `contact_status_is_valid`. A5 covers **four** abilities.
- **Effect Spore's SLEEP third stays refused** for a waking attacker. The sleep arm's
  `!target_side.has_alive_non_rested_sleeping_pkmn()` reads the same pre-wake status from a
  side-level helper with its own callers, so where Showdown's sleep clause would allow the
  re-sleep, we still decline. Recorded in the patch as a known residual rather than widened
  here. Effect Spore's POISON and PARALYZE thirds are fixed.

## 4. Red run

M3: a pin is not a pin until it has been seen red. Both pins were run against the installed
58-patch engine before the patch existed.

| pin case | expected | pre-patch |
|---|---|---|
| certain wake (`sleep_turns` 4, `chance_to_wake` 1.0) | 33.3333 % | **0** |
| forked wake (`sleep_turns` 1, `chance_to_wake` 1/4) | 8.3333 % | **0** |
| thaw (`randomChance(1, 5)`) | 6.6667 % | **0** |
| control: Sleep Talk stays asleep | 0 % | 0 % (green both eras) |
| control: paralysis | 0 % | 0 % (green both eras) |

gen3 `MAX_SLEEP_TURNS` is 4 and `chance_to_wake_up` is `1/(1 + MAX_SLEEP_TURNS - turns)`, so
the 1.0 and 1/4 figures are the engine's own arithmetic, not fitted constants.

The control pin carries an anti-vacuity assertion — that the contact move really lands 75 % of
the time — and **that guard caught its own first version**, which asserted on the string
`"Tackle"` and never fired because the branch text renders damage, not move names. Without it
the zero-poison assertion would also have passed on a state where no contact ever happened.

## 5. Gates

| gate | result |
|---|---|
| A5 pins | 2/2, red before, green after |
| `tests/test_engine_gen3_abilities` (whole suite) | Ran 46, OK |
| `tests/test_poke_engine_patch_stack` | Ran 4, OK — tail pin **grown** to 4 names, not slid |
| `tests/test_branch_mass_reconstruction` (mass gate) | Ran 5, OK |
| `tests/test_crit_kill_split_patch` | Ran 8, OK |
| `tests/test_drag_limit_is_a_last_resort` | Ran 3, OK |
| `scripts/engine_behavioral_probes.py` | all PASS |
| crate suite, `RUSTFLAGS="-C debug-assertions=yes"` | 0 failed, 366 passed, all four renderer pins ran |

CI did not run the A5 pins at all: `tests/test_engine_gen3_abilities.py` was neither a trigger
path nor executed by any step, so the patch would have triggered the workflow through the
`.patch` glob while its pins sat outside it. Added as a step naming both pins individually —
module-wide would be a false gate, because `AbilityCatalogTests` in the same file needs a
built Showdown checkout and legitimately skips in CI.

## 6. Sweep

Single-variable: same 200-game windows, same classifier, same matcher (`strict`), only the
patch differs. Baselines are the pre-A5 artifacts, not remembered figures.

| window | boundaries measured | matched | diverged | identity |
|---|---|---|---|---|
| dev `19,000,000–19,000,199` — before | 15,224 | 15,218 | 6 | holds |
| dev `19,000,000–19,000,199` — after | 15,224 | 15,219 | 5 | holds |
| validation holdout `19,100,000–19,100,199` — before | 15,396 | 15,382 | 14 | holds |
| validation holdout `19,100,000–19,100,199` — after | 15,396 | 15,383 | 13 | holds |

`matched` **rose** in both windows and `boundaries_measured` did not move, so the divergence
fall is a real match and not a boundary leaving the denominator.

Row level, which is what actually attributes the gain:

| window | rows closed | rows opened |
|---|---|---|
| dev | `19000125/226` | **none** |
| validation | `19100012/61` | **none** |

Those are exactly and only A5's two rows as filed by C118 §2 and C117 §4. Nothing opened in
either window, so no row was traded for another.

The prediction was registered before the sweeps were read: dev 6 → 5 on `19000125/226`,
matched 15,218 → 15,219, boundaries unchanged. It held exactly, and the holdout row closed too.

Residue is now **dev 5 / holdout 13**. Reported as an outcome, not a target: A5 is
closed as an *engine fix*, and neither window was retuned to reach it.

## 7. Process note: an exit code I nearly took on trust

The first replay of the grown stack printed `applied via git-apply` for all 59 patches and I
read that as success. It had **exited 1**: `apply_poke_engine_patches.py` asserts the
post-patch target-tree digest after the loop, and my `| tail -2` had cut the traceback off the
bottom of the output. The pins were still on the 58-patch digests.

The same shape has now cost this program four times — `OK (skipped=2)` from a venv with no
wheel, `OK (skipped=5)`, `depth-tactics OK` above a failing `patch-stack`, and this. The
mechanical rule that catches all four is to capture `$?` explicitly and assert on the summary
line, never on a tail of the log. Both new digests here were then read off a **replay** into a
scratch tree, never off the vendored tree on disk — the build rewrites that tree, so pinning
it can pin a stale preimage, which it once did and shipped a red gate.
