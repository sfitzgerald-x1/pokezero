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
`:2832` today — the **12**-line gap C111 recorded at `:2682`/`:2694`, now 13
because `if !choice.sleep_talk_move {` was added at `:2831` to wrap the call — so it is neither
13 *nor* the same, and an earlier revision said both. **The pin was right;
only the line numbers moved**, which v1 itself called "expected and not the finding" before
proceeding to make it the finding.

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
`19100012/61` is correct. `19100012/61` is the cleaner witness, **and it carries the argument
alone.** For `19000125/226`, "already asleep" is a *sufficient* cause independent of ordering —
gen3 admits no second non-volatile status — so that row's A/B cannot separate the two
explanations. Only `19100012/61` can: p2 switches, so Poison Point is the sole poison source
with no Sludge Bomb secondary to confound it, and its awake mass is exactly `0.85 × 1/3`.

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

Phase 3 item 8 is re-pointed at its correct site: **`generate_instructions.rs:2819`**
(`before_move`, which reaches `ability_modify_attack_against` at `:2103`) runs before
**`:2832`** (`generate_instructions_from_existing_status_conditions`, which generates the
wake). The fix is an ordering change and its blast radius is that second function — wide
enough to deserve a dedicated PR with its own pin, generalising across Poison Point, Effect
Spore, Flame Body, Static and Cute Charm × wake and thaw.

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
