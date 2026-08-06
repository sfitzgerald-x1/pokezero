# C130 — A11 closed: Belly Drum must fail at +6 Attack

C116 Phase 4 item 12 work: two of the nine remaining divergent rows disposed of as an **engine
fix**. Era: branch `fidelity-bellydrum-fail-clauses` off `main` `7762a81d`, patch stack grown
66 → 67.

> **On the C116 citation.** The C116 refocus plan lives outside this repository at the owner's
> instruction, so it is not verifiable by a future reader of this file. Read "Phase 4 item 12" as
> provenance for *why this was queued*, never as evidence for any claim about the engine.

## 1. The defect

`gen3/choice_effects.rs` `Choices::BELLYDRUM` gated on one condition:

```rust
if attacker.hp > attacker.maxhp / 2 {
```

Showdown fails the move on **three**. `data/moves.ts:1222-1228`, and there is **no gen3 or gen4
override** — verified by grepping both mod `moves.ts` files — so base is the effective handler:

```js
onHit(target) {
  if (target.hp <= target.maxhp / 2 || target.boosts.atk >= 6 || target.maxhp === 1) {
    return false;
  }
  this.directDamage(target.maxhp / 2);
  this.boost({ atk: 12 }, target);
}
```

So at +6 Attack the engine still halved the user's HP and pushed a `6 - 6 = 0` boost. A second
Belly Drum cost half the remaining HP for nothing. The red run says it verbatim:

```
Damage SideOne: 130 | Boost SideOne Attack: 0
```

## 2. The rows

`19100072/17` and `19100072/19`, both `roll_scaled_component`, both at 100 % mass:

```
observed=[]  engine=[('', -130)]
```

Linoone at **131/261** with Belly Drum PP 14/16 then 13/16 — already used, already +6, and
`131 = 261 - 130` is the earlier halving. Showdown emits `|-fail|p2a: Linoone` and applies
nothing; the engine applies −130.

The HP clause does **not** explain Showdown's refusal: `131 * 2 = 262 > 261`, so
`hp <= maxhp / 2` is false. The `boosts.atk >= 6` clause is the only one that fires, and
`maxhp / 2 = 130` is exactly the magnitude the engine applied. That is what makes the attribution
a measurement rather than a guess.

## 3. Reachability, checked before claiming anything

Instrument: `pokezero.randbat.Gen3RandbatSource.from_showdown_root(...)`, 220 species.

| clause | reachable in this pool? |
|---|---|
| `hp <= maxhp / 2` | yes — already implemented |
| `boosts.atk >= 6` | **yes** — Belly Drum is carried by exactly two species: `charizard` (2 of 6 variants) and `linoone` (3 of 4). Both filed rows are Linoone. |
| `maxhp === 1` (Shedinja) | **NO** |

Shedinja is in the pool, but its six variants are `agility` / `batonpass` /
`hiddenpowerfighting` / `shadowball` / `silverwind` / `toxic` — **none is Belly Drum**, so no legal
gen3 randbats team can reach the clause. It ships for parity with the source and is **not credited
with fixing anything**. Were it reachable it would matter: at `maxhp == 1`, `hp > maxhp / 2` is
`1 > 0` and passes, so the arm would grant +6 Attack for a `maxhp / 2 == 0` damage instruction.

## 4. Pins, and which of them are controls

| pin | pre-patch | post-patch |
|---|---|---|
| `..._fails_at_plus_six_attack` — no halving at +6, and none above +6 | **RED** (`Damage SideOne: 130 \| Boost SideOne Attack: 0`) | green |
| control: `..._still_works_from_plus_zero` | green | green |
| control: `..._still_fails_below_half_hp` | green | green |

The +0 control is **single-variable** — identical HP (131/261), only `attack_boost` differs — which
is what makes it evidence that this is a fail-clause and not a blanket disable. The half-HP control
pins that the new conditions were **ANDed into** the existing gate rather than replacing it. Both
controls are green in both eras; that is what makes them controls rather than pins.

> **The +0 control is its own test method because a review showed it otherwise proved nothing
> observable.** While it lived inside the +6 test, that whole method was RED pre-patch, so the
> "green in both eras" claim in this table could not be read off a test run at all — the reviewer had
> to hand-probe the engine to check it. A control folded into a failing test is not a control.
>
> The same review built a mutant writing the gate as `attack_boost != 6` instead of `>= 6`, and it
> **escaped every pin in the file**. Since the Python `State` API accepts an unclamped boost, that
> mutant spends half the user's HP to *lower* its own Attack (`attack_boost=7` →
> `Damage SideOne: 130 | Boost SideOne Attack: -1`). Real play clamps to +6, so it is unreachable and
> the shipped `>= 6` already handles it; the pin now asserts at +7 and +12 so that is a fact of
> record rather than a happy accident.

`attack_boost` is read **before** `get_active()` takes its mutable borrow, or the arm is `E0503`.
The first version of the patch did not, and did not compile.

## 5. Gates

| gate | result |
|---|---|
| `tests/test_poke_engine_patch_stack` | Ran 4, OK — tail pin **grown** 9 → 10, previous nine unchanged and in order |
| `tests/test_engine_gen3_abilities` | Ran 56, OK |
| `tests/test_branch_mass_reconstruction` (mass gate) | Ran 5, OK |
| `tests/test_crit_kill_split_patch` | Ran 8, OK |
| `tests/test_a1_residuals_already_ran` | Ran 13, OK |
| `tests/test_drag_limit_is_a_last_resort` | Ran 3, OK |
| `scripts/engine_behavioral_probes.py` | exit 0, **38** named probes PASS, 0 FAIL |

> The probe count is 38, not the 39 an earlier revision of this line claimed. `grep -c PASS`
> counts the trailing `all behavioral probes PASS` summary line as a 39th probe. The correct
> count comes from `grep -cE '^\[[^]]+\] PASS'`. Same class of error as reading a result off a
> log tail: the number came from the shape of the output rather than from the thing being
> counted.

Digests were re-derived **two independent ways that agree** — from my own edited scratch tree and
from a fresh `git apply` of the patch onto a replayed tree — both giving `choice_effects.rs` →
`4d2179c6…`. Never read off the vendored tree, which the build rewrites.

## 6. Sweep

| window | engine | measured | full_round | matched | diverged |
|---|---|---|---|---|---|
| dev `19,000,000–19,000,199` | main `599c68a31e…` | 15,432 | 15,968 | 15,430 | 2 |
| dev | A11 `12e05f6e8a…` | 15,432 | 15,968 | 15,430 | **2 (unchanged)** |
| validation holdout `19,100,000–19,100,199` | main `599c68a31e…` | 15,551 | 16,155 | 15,544 | 7 |
| validation holdout | A11 `12e05f6e8a…` | 15,551 | 16,155 | **15,546** | **5** |

**Closed exactly `19100072/17` and `19100072/19`. Nothing opened in either window.** Identity holds
on all four runs and `engine_errors` is 0 in all four. The prediction registered before any sweep —
holdout 7 → 5 closing those two rows, dev unchanged, `boundaries_measured` unchanged, nothing
opened — held in every part.

> **The baseline had to be measured twice, and the first one was wrong in the way I had just
> written a memo about.** My first "before" run used the pre-merge branch engine, whose sweep was
> taken on an **older harness**. Against it the holdout looked like 8 → 5 and
> `boundaries_measured` appeared to rise by 155, which I was briefly ready to explain as A11
> recovering unmeasurable boundaries. It was not. The counter that moved was
> `skip:world_unsupported:rest_sleep_active_refund_pending`, **158 → 0**, which has nothing to do
> with Belly Drum: `main` had advanced by **#1113 "supply the pending Rest refund instead of
> refusing"**. Re-running the baseline on the *current* harness with `main`'s engine — fingerprint
> printed at both builds, and identical at `599c68a31e…` because #1113 is Python-side — collapsed
> the delta to zero and left A11's two rows as the only change.
>
> The tell was that `boundaries_full_round` was identical (16,155) while `boundaries_measured`
> moved: the games were the same, so nothing about the *engine* could have changed the
> denominator. A "before" and an "after" that differ in the harness are not a differential.

## 7. Still open after this

Residue is **dev 2 / holdout 5**, seven rows, every one attributed:

| rows | cause | disposition |
|---|---|---|
| `19100193/46`, `19100014/35` | Leech Seed vs Leftovers residual-heal **labelling** | harness fix, written, not yet landed |
| `19100180/24` | hazard applied to the non-replacing side on a forced-replacement ply (B1) | open |
| `19100107/135`, `19100191/5` | `limit:roll_divergent_lethality` | already classed as limits by the harness |
| `19000191/63` | collapsed roll; the heal delta (28 vs 29) is **downstream**, verified — both values are correct given each side's own HP after a 109-vs-101 move roll | open |
| `19000074/27` | collapsed roll on the crit magnitude (93.75 % + 4.69 %), plus a 1.56 % crit-kill arm omitting the attacker's own sandstorm chip | open; the 1.56 % component is *reasoned* to be A1 residual placement and that attribution is **not yet measured** |
