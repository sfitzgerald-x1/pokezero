# C118 v2 — A5 is CONFIRMED, and both of v1's findings were inverted

> **v1 of this report claimed A5's hook was mis-pinned and its mechanism refuted. Both
> claims were wrong, in the same direction, and A5 is confirmed by measurement on both its
> rows. v1 should not be cited.** The retraction is the content of this document.

C116 Phase 3 item 8 opens "reorder contact-ability trigger vs. same-turn wake **at the
pinned lines**". Doing that re-pin produced two false findings before producing the right
answer.

Era: `main` `012d8451`, engine fingerprint `4098f204…`, 58 patches.

## 1. What v1 claimed, and why both halves were wrong

**v1 finding (a): "C111 pinned the wrong hook."** Inverted. Function extents in
`third_party/poke-engine-src/src/gen3/abilities.rs`, re-derived by walking `^pub fn`/`^fn`/`^}`:

| function | extent | ability arms it contains |
|---|---|---|
| `contact_status_is_valid` | 22–52 | — |
| `ability_after_damage_hit` | **174–207** | `COLORCHANGE`, `ROUGHSKIN` — **and nothing else** |
| `ability_modify_attack_against` | **648–925** | `POISONPOINT` :672, `EFFECTSPORE` :693, `CUTECHARM` :721, `FLAMEBODY` :745, `STATIC` :779, `THICKFAT` :793 |

So **every contact secondary lives inside `ability_modify_attack_against`** — exactly the
hook C111 pinned. v1's table asserted they were in `ability_after_damage_hit`, and its gloss
("an attack *modifier*, this is the Thick Fat hook, **not** the contact-secondary hook") was
false: it is *both*. I cited real line numbers, verified they existed, and then described
the wrong containing function — a check that confirms a line exists proves nothing about
what encloses it.

C111 v2 also does not pin an `abilities.rs` arm at all. It pins the two **call sites**:
`before_move → ability_modify_attack_against` against the wake. Those are `:2819` and
`:2832` today, against the `:2682`/`:2694` C111 recorded.

> **Correction (review of #1089).** A previous revision of this paragraph explained the
> 12-versus-13 discrepancy by saying `if !choice.sleep_talk_move {` "was added at `:2831` to
> wrap the call". **That insertion never happened and I did not check.** The guard is upstream
> `poke-engine` 0.0.47 code — pristine sdist `src/gen3/generate_instructions.rs:1652`, wrapping
> that same call — and no patch in the stack adds it. (`poke-engine-gen3-pp-ordering.patch`
> does add a line reading `if !choice.sleep_talk_move {`, but it wraps the **PP decrement**
> block ~174 lines below, at `:3005`. Precisely: that is how a grep of the PATCH FILES
> misleads. It does not mislead in the built source, where `lockedmove-pp.patch` later extends
> that line to `if !choice.sleep_talk_move && !in_locked_turn(...)`, so the exact string has
> exactly one hit there. Checking the built tree would have caught this; checking the patch
> files is what did not.)
>
> What actually moved: the whole region shifted a **uniform +137 lines**
> (2682 + 137 = 2819, 2694 + 137 = 2831, 2695 + 137 = 2832) as
> `residual-lethality-partition.patch` grew from +53 to +197 inserted lines across
> #1065/#1066/#1069. That is +144, not +137: the missing 7 are a second hunk of the same patch
> that sat *above* this site at #1062 and was relocated below it by #1065, so it stopped
> contributing to the offset. Nothing was inserted *between* the two call sites.
>
> And the discrepancy is an off-by-one, not a change in the code. To be fair to the record
> rather than generous to C111, it is C111's off-by-one: C111 gave the line number of the guard
> while its prose named the call. Stated plainly rather than as two conventions meeting:
> C111 pinned the **guard** (`:2694`, now `:2831`); this report pins the **call the guard
> wraps** (`:2695`, now `:2832`). Like for like the gap is **12 in both eras** —
> `2694 − 2682` and `2831 − 2819`. So the "same" half of the sentence this note was retracting
> was right, and only the "13" half was wrong; the retraction was itself made on a false
> premise. Fabricating a structural cause, in a report whose whole subject is fabricated
> structural causes, is the same error one level up.

**The pin was right; only the line numbers moved**, which v1 itself called "expected and not
the finding" before proceeding to make it the finding.

**v1 finding (b): "the mechanism is refuted."** A non sequitur. The Poison Point secondary
is decided during **choice modification** at `:2103`, reached from `before_move` at `:2819`
— *before any of the instructions v1 quoted exist*. The applied instruction order (wake
before the Wrap damage) is fully **consistent** with `contact_status_is_valid` seeing
`SLEEP`, and is silent on the question. I read an instruction listing and drew a conclusion
about a decision taken earlier in a different phase.

## 2. The measurement that settles it

Single-variable A/B on the recorded states — `SLEEP,1` → `NONE,0`, nothing else changed:

| row | state | branches | poison mass |
|---|---|---|---|
| `19000125/226` | asleep (as recorded) | 6 | **0.0000 %** |
| `19000125/226` | awake | 16 | 49.8333 % |
| `19000125/226` | awake, Poison Point removed | 12 | 30.0000 % (Sludge Bomb's own secondary) |
| `19100012/61` | asleep | 3 | **0.0000 %** |
| `19100012/61` | awake | 5 | **28.3333 %** |

`28.3333 % = 0.85 × 1/3` — Wrap's accuracy times Poison Point's activation, which is
**exactly the "1/3 of hit mass" C111 v2 had already measured** ("0 of 2 wake-and-hit
branches, versus 2 of 4 when already awake"). The same-turn wake is the sole cause.

**A5's mechanism is confirmed on both rows**, and C117 §4's signature-based filing of
`19100012/61` is correct. `19100012/61` is the cleaner witness: p2 switches, so Poison Point is
the sole poison source with no Sludge Bomb secondary to confound it, and its awake mass is
exactly `0.85 × 1/3`.

> **Retraction (review of #1089).** A previous revision added that `19100012/61` "carries the
> argument **alone**", on the reasoning that for `19000125/226` "already asleep" is a sufficient
> cause independent of ordering. That note was wrong twice over, and the sentence above it —
> which the note contradicted without amending — was right.
>
> It was wrong on its own terms: the attacker on `19100012/61` is *also* asleep, so the premise
> voids both rows equally and cannot be what separates them.
>
> And it is now settled by measurement rather than argument. The A5 fix (#1090) changes exactly
> one predicate and nothing else, and on a single-variable 200-game sweep of each window it
> closed **`19000125/226` on dev and `19100012/61` on the validation holdout, with nothing
> opened in either window** — matched 15,218 → 15,219 and 15,382 → 15,383, boundaries
> unchanged. Two rows filed to A5, both closed by A5's fix. They are two witnesses.
>
> This is the third time in this document's history that a structural argument was overturned
> by a measurement that was cheap and available. The pattern is not that the reasoning is
> careless; it is that I keep reaching for it when a two-line experiment would settle it.

v1's "6.25% unaccounted for" was also wrong: the branch listing has **six** arms, not three
— `14.06 + 74.71 + 4.98 + 0.94 + 4.98 + 0.33 = 100.00` — and the remainder is the Sludge
Bomb crit fan. **No arm carries `psn`.** So the row is not a mass-enumeration issue either;
the poison is absent everywhere, because the check refuses.

v1's "untested candidates" (Poison Point's activation probability; whether Wrap's contact
flag reaches the hook) were **already settled in C111 v2**. I proposed re-testing things the
ledger had measured.

## 3. Disposition

**A5 stands. Both rows stay attributed. Nothing is withdrawn, and C117 needs no amendment.**
v1's accounting changes are void: C117's "15 of 25 on C111's six causes" is correct as
written, and A5 has two rows, not zero.

**Update: A5 is closed.** The fix landed as #1090 — `contact_status_is_valid` now ignores a
SLEEP or FREEZE that is cured before contact, which is exact rather than approximate because
every branch that keeps the status is one on which the move never lands. See
`reports/c121_a5_wake_before_contact.md`. Two corrections this report owed and #1090 records:
**Cute Charm is not affected** (it gates on the ATTRACT volatile via
`volatile_status_can_be_applied`, not on `contact_status_is_valid`), so the fix generalises
across **four** abilities and not the five listed below; and Effect Spore's SLEEP third stays
refused for a waking attacker, because the sleep clause reads the same pre-wake status through
a side-level helper.

Phase 3 item 8 is re-pointed at its correct site: **`generate_instructions.rs:2819`**
(`before_move`, which reaches `ability_modify_attack_against` at `:2103`) runs before
**`:2832`** (`generate_instructions_from_existing_status_conditions`, which generates the
wake). ~~The fix is an ordering change and its blast radius is that second function — wide
enough to deserve a dedicated PR with its own pin, generalising across Poison Point, Effect
Spore, Flame Body, Static and Cute Charm × wake and thaw.~~ **Superseded:** #1090 is a
*predicate* change in `abilities.rs`, chosen deliberately over a reorder (see
`reports/c121_a5_wake_before_contact.md` §2 — a reorder would move every
`item_before_move`/`choice_before_move` instruction relative to every sleep, freeze and
paralysis branch in the game), and it generalises across **four** abilities, not five.

## 4. What went wrong here, twice, in the same direction

Both v1 findings came from **reading source structure instead of measuring behaviour**, and
both survived a check that looked rigorous:

- I verified every cited line **exists**. None of those greps could detect that a line sat
  inside a different function than I claimed. Verifying a citation resolves is not verifying
  it says what you think.
- I then read an *instruction listing* — real, correctly transcribed — and concluded
  something about a decision made in an earlier phase that emits no instruction at all.

The A/B that settles it takes two state edits and two `generate_instructions` calls. It was
available before v1 was written, and C111 v2 had already run its equivalent. The cheapest
correct move was to re-run the ledger's own measurement rather than re-derive its
conclusion from source.
