# C115 — engine-fidelity program: objective, what landed, what remains

Written 2026-08-04. This is the orientation document for the gen3 engine-fidelity
program: what it is trying to do, what has merged in pursuit of it, what those
changes actually fixed, and what is still open. It supersedes nothing — the
per-row ledger is `c111_residue_row_causes.md` and remains the authority on
individual rows.

---

## 1. The objective, and why it is stated the way it is

**Goal.** Make the vendored gen3 engine agree with Pokémon Showdown's gen3
implementation, boundary by boundary, on a fixed 200-game window (seeds
19000000–19000199, 15,224 measured boundaries).

The framing that matters more than the target:

> The reference implementation is fully available — Showdown's gen3 mod and the
> vendored engine source. Every divergence therefore has a determinate cause that
> can be read out of those two sources. **Treat "unattributed" as a gap in the
> classifier, never as a gap in what is knowable.**

Each divergence shape must be worked to a source-level cause — reproduce the
boundary, read what Showdown does, read what the engine does, state the difference
— and then disposed of as exactly one of three kinds:

1. **engine fix** — a patch plus a regression gate;
2. **matcher/harness fix** — with a pin that fails when reverted;
3. **comparison limit** — reserved for what the methodology *genuinely cannot
   settle*, e.g. the engine enumerating a branch set against Showdown's single
   realised sample.

**Two rules that shaped everything below.** A row may never be closed by widening
or retuning an attribution rule; classifier changes are measured and reported
separately from fidelity changes, never in the same commit. And every change gets
an independent adversarial review before merge, verified by single-variable
measurement — mutate exactly one thing on one tree and re-sweep.

**Stopping condition.** Every shape has a written source-level cause and a
disposition, *and* the remaining divergence consists only of dispositions of the
third kind. The divergence count is an outcome, not a target.

---

## 2. Trajectory

**208 → 7** divergent boundaries on the window, with `boundaries_measured` held at
15,224 throughout and `engine_errors` at 0.

The table below reconstructs only the **39 → 7** span, which is the part worked in
this era; the 208 → 39 reduction predates it and is not itemised here. §3's
"eight of ten" therefore describes the 39 → 7 changes.

That the denominator held is not incidental. A divergence count can fall because
boundaries left the measured pool rather than because anything was fixed, so every
measurement below reports `boundaries_measured`, `transitions_matched`, and the
identity `matched + diverged == boundaries_measured`. A fidelity claim requires
`matched` to hold or rise.

| landed | delta |
|---|---|
| Liquid Ooze renderer guard (#1039) | 39 → 36 |
| Belly Drum roll gate (#1042) | 36 → 35 |
| Pain Split roll-awareness (#1054) | −4 |
| Crit-tag by reachability + pristine Choice (#1055) | −10 |
| Faint-without-damage synthesis + Air Lock speed guard (#1059) | 21 → 13 |
| Residual-lethality partition + gen3 contact flags (#1062) | 13 → 10 |
| Toxic stage `count + 1` (in #1062) | 10 → 11 *(deliberate, see §4)* |
| Pending read after this call's mutations (#1065) | 11 → 9 |
| Ordered residual simulation (#1066) | 9 → 8 |
| Case A three-way partition (open PR) | 8 → 7 |

---

## 3. What was actually fixed, by mechanism

Eight of the ten landed changes are instances of **three** underlying defects.
This is the program's central empirical finding: 36 rows was never 36
investigations.

### 3.1 A collapsed arm discards an observable

The engine represents a 16-roll damage fan by collapsing it to a representative
value. That is sound only when every roll in the arm is equivalent *in every
observable consequence*. Repeatedly it was not, and each fix recovers exactly one
discarded distinction:

- **Crit-kill split (C27, pre-existing)** — the crit fan straddles the KO
  threshold; collapsing puts all crit mass on one side.
- **Residual-lethality partition (#1062)** — a roll can decide lethality one phase
  *later*. The arm that cannot kill on the hit still collapses to `0.925 × max`,
  which picks one side of the *residual* threshold and drops the other.
- **Crit-fan extension (#1062)** — the same defect on the crit fan.
- **Case A three-way (open PR)** — the arm that *can* kill on the hit collapses
  its surviving sub-fan without consulting the residual threshold. When that
  representative lands on the lethal side, **no branch survives the turn at all**.
- **A7 (open, `19000191/63`)** — the residual-lethal arm collapses onto its
  minimum roll, valid only if death is the sole observable. It is not: Leech Seed's
  sap is `min(maxhp/8, hp)` clamped to the victim's HP at 10.5, and 10.5 heals the
  *other* side by the sapped amount, so the cross-side heal is roll-dependent
  (29 / 28 / 27 for rolls 108 / 109 / 110).

### 3.2 The residual mirror did not match the phase

`pending_residual_damage` predicted end-of-turn damage in order to place the
threshold. Three separate errors, each measured:

- **Toxic stage** — mirrored as `max(count, 1)` where the phase uses
  `normalized_toxic_count + 1`. Correct only at count 0; every higher rung
  understated by a full stage.
- **Read against pre-switch state (#1065)** — bound at *function* level, ~70 lines
  before `state.apply_instructions(...)`, so on any boundary where the defender
  switched in it read the **outgoing** Pokémon.
- **A sum instead of an ordered walk (#1066)** — heals precede every damage tick
  (7 Wish, 8 weather, 10.3 Rain Dish, 10.4 Leftovers, then 10.5 Leech Seed, 10.6
  status, 10.9 partial trap), so a damage-only sum sets the threshold too low.
  A *net* is unsound in the other direction: the order-8 chip can kill before the
  10.4 heal — `hp 10`, chip 15, Leftovers 18 nets `−3`, reads "never dies", and is
  dead at step 8. Replaced by an ordered simulation with each clamp applied, the
  threshold located as `hp − h* + 1` by bisection on a survival predicate.

Also discovered here: the weather **decrement runs before the chip**, so a weather
with one turn left ends and never chips. `weather_is_active` ignores
`turns_remaining`, so a mirror consulting it alone over-counts on the expiring
turn — the only direction that can make the engine *worse* than not partitioning.

### 3.3 Reference-data and renderer divergences

- **gen3 contact flags (#1062)** — Overheat and Ancient Power make contact in gen3;
  Covet, Fake Out and Feint Attack do not. Upstream carries Gen 4+ flags. The
  observable was Rough Skin failing to retaliate against Overheat.
- **Liquid Ooze, Air Lock speed, pristine Choice, crit-tag reachability,
  faint-without-damage synthesis** — renderer/matcher corrections landed earlier.

---

## 4. Decisions worth recording, because the counter argued against them

**The Toxic fix cost a row and landed anyway.** Correcting the stage to `count + 1`
moved the residue 10 → 11. `19000147/125` had been matching *by accident*: the old
mirror understated the tick by exactly one stage (18) and Leftovers heals exactly
18, so two errors cancelled. **"Accident" understates it**: the understated stage is
`maxhp/16` and Leftovers is also `maxhp/16`, so the cancellation is *structural* and
holds for every Toxic + Leftovers defender at every `maxhp` — not a coincidence of
one fixture. Keeping the bug to hold the counter down would have meant closing a
boundary with a magnitude the engine source contradicts.

**A mass leak was invisible to the sweep.** In review of #1062, the non-crit
residual split was found to call `update_percentage` *in place*, silently scaling
every crit arm cloned from that value afterwards. Totals still summed to 100%, so
no conservation check fired — and the transition differential compares *components*,
not branch *masses*, so it is structurally blind to the entire defect class. The
fix measured **neutral** on the sweep. Only a probe comparing masses against an
independent reconstruction could see it.

**Three "genuine comparison limits" were not limits.** C111 v1 claimed
`19000008/54`, `19000191/63` and `19000198/33` were irreducible. All three failed:
`19000198/33` closed on an engine fix; `19000191/63` reduces (it is A7);
`19000008/54` was never a drag divergence at all — the engine dragged the *same*
Pokémon and only a component *tag* differs. **Zero rows in the residue are
demonstrated comparison limits.**

---

## 5. What is outstanding

**Residue: 7 rows, six named causes, all with source-level attribution. None is a
comparison limit.**

| row | cause | disposition |
|---|---|---|
| 19000020/50, 19000059/27 | **A1** faint/forced-switch residual placement — the engine defers the phase on a faint (faithful gen3), Showdown runs it on the faint boundary | matcher/harness: needs a "residuals already ran" marker + a revert-failing pin |
| 19000074/27 | **A4** crit-**kill** split's survive arm unpartitioned on the residual | engine fix, symmetric with merged work |
| 19000112/32 | **A6** gen3 White Herb absent (`WHITEHERB` only under `src/genx/`), so Superpower's Def drop is never restored and every roll reaches lethal | engine fix, then re-ask the hidden-item question |
| 19000125/226 | **A5** contact-ability trigger precedes the same-turn wake: `ability_modify_attack_against` at `:2682` runs before the wake `ChangeStatus` at `:2694`, so `contact_status_is_valid` sees SLEEP | engine fix; generalises to Effect Spore, Static, Flame Body, Cute Charm |
| 19000191/63 | **A7** collapsed lethal arm discards the clamped sap | engine fix: split at `hp_after_move + leftovers < maxhp/8` |
| 19000008/54 | **B1** classifier applies the drag limit on the mere presence of a `\|drag\|` line; the real diff is a `spikes` vs `move` tag | renderer fix (closes the row) + classifier fix (outlives it), measured separately |

### Known gaps in shipped code, recorded not implied

- **Future Sight** (order 11, lethal-capable) is neither simulated nor declined by
  the residual mirror. It only *adds* damage, so it strictly **under**-partitions
  and can never invent a residual KO.
- **Leech Seed at 10.5 is cross-side and speed-major.** A faster attacker holding
  `LEECHSEED` heals the defender before the defender's own 10.x set; with
  `LIQUIDOOZE` it damages instead. Only the defender's own seed is modelled.
- **Dry Skin** heals `maxhp/8` in rain at 10.3. Dead code for gen3 randbats, but it
  falsifies "Rain Dish is the only HP-changing ability".
- **`compare_health_with_damage_multiples` accumulates in f32**, so its rungs can
  fall below the true `floor(max · r / 100)`. Re-derived against the shipped
  expression rather than transcribed: the **top** rung lands one below `max` for
  **173** of the first 400 `max` values, and over all `(max, threshold)` pairs in
  that range there are **195** kill-count mismatches, of which **22 are at
  interior thresholds** — the drift also hits rung `r=90` (14 values) and `r=95`
  (8 values), not only the top.
  So the defect is **not** confined to `threshold == max_damage`. Concretely, at
  `max_damage = 120`: threshold 108 counts 10 kill rolls against a true 11, and
  threshold 114 counts 5 against 6 — neither equals `max_damage`. That is 1/16 of
  non-crit mass on the wrong arm at an interior threshold.
  An earlier revision of this document said the divergence occurred "exactly where
  `threshold == max_damage`". That was **false**, and it was false in the worst
  direction: it would have sent the next reader looking only at the boundary case.
  I had transcribed it from a review note instead of deriving it. Pre-existing and
  now shared by **four** paths (Case A gained one); registered, not changed.

### Open PRs

- **#1068** — ledger consolidation (this document plus the A7 correction). A7 and
  the A2 spec had been committed and pushed but left on branches with no PR, so
  `main` served a superseded account. Replaces #1067, which was **closed** because a
  live engine patch had been committed onto the same branch by mistake.
- **`scott/a8-case-a-three-way`** — the Case A three-way partition, 8 → 7, measured
  and pinned, awaiting review.

---

## 6. How the work is verified, and where that verification failed

Every fidelity change is measured single-variable against the same tree without it,
on the same seed range, reporting the full counter set. Every change is
independently adversarially reviewed before merge; an approving review is
sufficient authorisation.

**Review caught defects the sweep could not**, three times: the probability-mass
leak (§4), an engine patch that modified both arms of a `cfg!` gen fork including
the non-gen3 arm (dead under gen3, wrong as data), and a rewrite of the entire
threshold model that **no test distinguished** — reverting it would have passed CI.

**The recurring failure mode on my side was fixtures that read PASS while asserting
less than intended**, five times: a composition case where neither split fired; a
threshold placed outside the fan so both eras agreed; pins that straddled the crit
arm instead of the non-crit arm they were meant to protect; a control that changed
two variables; and a probe reading the burn tick as a damage roll. In every case
the fix was to tie the metric to the mechanism rather than to branch shape, and to
print the structure instead of trusting the green line.

**A process failure worth naming:** every *code* change got a PR and a review, but
documentation commits drifted into done-when-pushed. Since the stopping condition
is a claim about the ledger file, that is the worst place for it to have drifted.
