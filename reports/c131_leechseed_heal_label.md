# C131 — four residual-attribution fixes in the event renderer

C116 Phase 4 item 12. Branch `harness-leechseed-heal-label` off `main` `5a44c04e`. All four changes
are in `rust/pokezero-search/src/events.rs`. **The engine's HP arithmetic was already correct in
every case; only attribution was wrong.**

> **This report was rewritten at round 7, and the rewrite is the point.** Seven review rounds found
> errors in this document while the code was verified correct from round 2. The previous revision had
> grown to 303 lines carrying ~35 quantitative claims and **15 inline retractions** — more retraction
> than content, and each retraction was itself a new claim that could be wrong. Six of the seven
> BLOCKs were the same shape: a correction applied to the instance a reviewer named, while the same
> defect stood one surface over. The remedy is fewer claims, not more careful prose. Every number
> below is one I can point at an artifact for; the errors are listed once, in §6, rather than
> re-litigated where each occurred.

> **On the C116 citation.** The plan lives outside this repository at the owner's instruction and is
> not verifiable by a future reader. Read "Phase 4 item 12" as provenance for *why this was queued*,
> never as evidence for a claim about the engine.

## 1. The four changes

| # | change | why |
|---|---|---|
| 1 | `weather_chips` gains the Sand Veil exemption | the engine skips the sand chip for `SANDVEIL` (`gen3/generate_instructions.rs:4223`); `events.rs` did not, so the plan booked a chip that never fired |
| 2 | `weather_chips` books no chip when weather expires | `weather_is_active` ignores `turns_remaining` (`gen3/state.rs:1050-1060`), but the engine decrements and clears the weather (`gen3/generate_instructions.rs:4144-4163`) **before** its chip loop (`gen3/generate_instructions.rs:4193`) |
| 3 | `residual_heal_cause` tests Leftovers before the drain | a Leftovers tick on a side whose opponent was seeded came back labelled `leechseed` |
| 4 | the drain returns `String::new()`, rendering `[silent]` | Showdown renders it silently: `sim/battle.ts:2293-2296`, `case 'leechseed'`, reached from `data/moves.ts:10218-10221`. There is no `[from] Leech Seed` heal line in Showdown |

**Why one unfilled slot matters (changes 1 and 2).** `ResidualPlan` reconciles what it booked against
what was emitted; if it comes up short, the whole side falls back to `residual_heal_cause`, which
takes no heal index and is therefore a *constant function of state* — it cannot label two heals on
one side differently.

## 2. What closes what

| change | closes | holdout |
|---|---|---|
| change 3 (the reorder), alone | `19100193/46` | 5 → 4 |
| change 1 (Sand Veil), via that row's 90 % arm | `19100014/35` | 4 → 3 |
| changes 2 and 4 | nothing measured | — |

Evidence: the artifact committed at `87bcf351`, whose `events.rs` contains **zero** `SANDVEIL`
occurrences and which predates change 4 (that entered at `c9f6839b`), records holdout **4** with
`19100193/46` already closed.

- `19100193/46` is label-only. Cacturne has 290 maxhp and `290/16 = 18` is a Leftovers tick; a real
  drain from Miltank would be `273/8 = 34`. Cacturne dies to poison before Miltank's sap, so no drain
  occurs and both sides agree on every HP value.
- `19100014/35` closes through its **90 % arm**. The 10 % arm is the engine's Leech-Seed-*missed*
  branch against a Showdown hit (`observed_only=[('leechseed', -33)] engine_only=[]`); no rendering
  change can make a miss branch reproduce a hit. One matching branch closes a boundary.
- **Change 2 is worth zero rows: its trigger is unreachable here.** `weather_chips` returns `Some`
  only for sand or hail, and `data/random-battles/gen3/sets.json` has **0 of 220** species carrying
  `sandstorm` or `hail` (Snow Warning does not exist in gen3). Sand therefore always comes from Sand
  Stream, which writes `WEATHER_ABILITY_TURNS = -1` (`gen3/abilities.rs:20`), and the engine never
  decrements a non-positive value. It ships for fidelity, like the Liquid Ooze guard in §5.

The `== 1` boundary still matters: the **permanent** region *is* reachable (Tyranitar, Kyogre and
Groudon are all in the pool), and `<= 1` breaks it across all of it.

## 3. Pins

Five, all green on the branch, each verified red against deleting the line it covers.

| pin | on `main` |
|---|---|
| `a_seeded_opponent_does_not_steal_the_leftovers_tag` | **RED** |
| `without_leftovers_a_seeded_opponent_still_yields_the_drain_label` | **RED** |
| `liquid_ooze_on_the_seeder_means_a_heal_here_is_not_the_drain` | n/a (new) |
| `sand_veil_is_exempt_so_the_plan_does_not_book_a_chip_that_never_fires` | n/a (new) |
| `expiring_weather_books_no_chip_so_the_drain_keeps_its_label` (3 arms) | n/a (new) |

Neither of the first two is a *control*: both are regression pins, and restoring `main`'s fallback
fails both. Revert shapes, each run: whole fallback → 2 failures; reorder only → the first;
`String::new()` → `"Leech Seed"` only → the second.

Boundary mutations, all red: `<= 1`, `< 2`, `!= 0`, `>= 1`, `== 2`, `== 0`, `== -1`, gate deleted,
gate scoped to SAND, gate scoped to HAIL, gate moved below the hail branch.

## 4. Gates and sweep

Crate suite under `-C debug-assertions=yes`: **375 passed, 0 failed**. Patch stack `Ran 4, OK`; mass
gate `Ran 5, OK`; `test_final_holdout_guard` `Ran 14, OK`. The wider Python suite's FAIL/ERROR set is
unchanged from `main`.

| window | engine | measured | full_round | matched | diverged |
|---|---|---|---|---|---|
| dev `19,000,000–199` | `main` `5a44c04e` | 15,432 | 15,968 | 15,430 | 2 |
| dev | branch | 15,432 | 15,968 | 15,430 | 2 |
| holdout `19,100,000–199` | `main` `5a44c04e` | 15,551 | 16,155 | 15,546 | 5 |
| holdout | branch | 15,551 | 16,155 | **15,548** | **3** |

Nothing opened; identity holds on all four; `engine_errors: 0`. Artifacts:
`reports/artifacts/c131_leechseed_{main,fix}_{dev,holdout}_sweep.json`, each stamping a
`source_commit` whose tree produced the run. Re-measured after the rebase onto `5a44c04e` and again
after every code change — "the numbers probably did not move" is a prediction, not a measurement.

## 5. Filed, not fixed

- **Five unpinned exemptions in `weather_chips`**: ROCK, GROUND, STEEL (sand condition), ICE (hail
  branch), and the `hp <= 0` gate. Each verified to leave the suite green when deleted. Pre-existing,
  but this change added a fourth disjunct to that condition and pinned only its own.
- **The Liquid Ooze guard is dead code.** Liquid Ooze is emitted as a *negative* heal, which
  `events.rs:3261` routes to the damage renderer, so it never reaches `residual_heal_cause`. Its pin
  is real against deleting the guard; the guard protects nothing reachable.
- **Rain Dish and Sitrus book no plan slot either**, and both are unreachable: 0 Rain Dish sets in
  `sets.json`, and `teams.ts:452-513` `getItem` cannot return Sitrus Berry.
- A pre-existing red in `tests/test_engine_terminal_residual_roll_limit` that fails identically on
  `main`.

## 6. Errors this report made, listed once

Every one was caught by review, and every one was in this document rather than in the code.

1. **Named the wrong mechanism.** v1 credited the reorder with `19100014/35`; the cause is the
   missing Sand Veil gate, which the reorder cannot substitute for.
2. **Claimed a row was fixed that the committed artifact showed was not** (`19100014/35`'s 90 % arm),
   without opening the file I had just committed.
3. **Asserted `"Leech Seed"` as a heal label** Showdown never emits — and wrote a pin asserting it.
4. **Left the load-bearing line unpinned**, then wrote two vacuous pins for it in a single commit,
   both asserting on *damage* tags when the defect corrupts *heal* labels.
5. **Called the expiry gate "the reachable member"** of its class, two paragraphs below my own use of
   the reachability instrument, in a sentence criticising myself for having skipped it.
6. **Argued the `== 1` boundary instead of pinning it**, and named `0` as the permanent-weather
   sentinel when it is `-1`.
7. **Left false provenance pins** on the fix artifacts after fixing them on the baselines.
8. **Cited two measurements I had not made** — one of them in the same edit that removed the other.
9. **Over-credited Sand Veil with two rows**, then while correcting that over-credited `[silent]` and
   claimed "both arms", using as proof an artifact predating the change I was crediting.

The through-line: I corrected the instance a reviewer named and left the same defect one surface
over — baselines but not fix artifacts, paragraph but not heading, report but not PR body, one row
count but not the next.

## 7. Residue after this

**dev 2 / holdout 3.**

| row | cause | disposition |
|---|---|---|
| `19100107/135`, `19100191/5` | `limit:roll_divergent_lethality` | **NOT disposed — see below.** The `limit:` prefix is a classifier label, not an adjudicated limit |
| `19100180/24` | hazard applied to the non-replacing side on a forced-replacement ply (B1) | open |
| `19000191/63` | collapsed roll; the heal delta (28 vs 29) is downstream and verified — after a 109-vs-101 move roll Raichu sits at 14 vs 22, so `min(29, 14+14)=28` and `min(29, 22+14)=29` are each correct given their own HP | open |
| `19000074/27` | crit fan straddles the residual-lethality threshold; the crit-straddle path emits no residual arm | open. Its 1.56 % component was filed as candidate **A12**, now **retracted** — both simulators defer the residual phase to the replacement ply, so that was never a defect |

> **The `limit:` label on two of these rows is unearned, and I have been reporting it as a disposition.
> It is not one.** C116 item 12 requires each row to be an engine fix, a harness fix, or a limit **with
> a written demonstration**. A classifier prefix is none of those. Three of the repo's own records say
> so:
>
> - `scripts/family_bucket_audit.py:58` buckets this family as **"engine-gap (partially resolved)"**,
>   and `:12` defines engine-gap as the engine's branch support not containing the observed transition.
> - `reports/c105_retract_limit_overclaim.json` records the limit label as **"8-for-8 falsified"** across
>   the eight rows it was applied to — and `19000191/63` is explicitly one of the eight.
> - The signature the audit gives for an engine gap — a legal roll range straddling a discrete threshold
>   while the emitted arm sits on one side of it — holds for **all four** of these rows.
>
> On `19000074/27` the observed crit of **241 is roll 96 of the engine's own priced fan**
> (`[214 … 241, 244, 246, 249, 252]`); the engine's `227` is the mean of the 12 non-KO rolls and is not a
> fan member. The engine prices the roll and emits no arm at it.
>
> So the honest count is **five rows, none disposed** — not "five rows, two already limits". Corrected
> here rather than carried forward.
