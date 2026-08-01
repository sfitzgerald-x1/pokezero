# C26 structural residue: pre-launch diagnosis

## Result

**The registered C26 sweep would fail, and it should not be launched as-is.**

`structural_component_count_without_supported_sibling` is registered at
predicted-zero in `reports/c26_current_engine_resweep_spec.json`. On a fresh
60-game sample of the frozen 51-patch build it fires on **29 of 64 divergent
rows — 0.48 per game**, which projects to roughly **4,800 unattributed rows** in
the registered 10,000-game sweep. The certification standard requires zero.

Most of that residue is the rule firing wider than its own stated semantics,
and the bulk of what survives is the matcher's handling of capped heals. But a
small, hard core is a real engine gap: the branch set does not contain the
transition Showdown took. Fix specification below.

## Evidence

| population | scoping | unattributed | structural |
| --- | --- | ---: | ---: |
| fresh 60 games (64 rows) | registered | 29 | 29 |
| fresh 60 games | majority-scoped | 7 | 7 |
| fresh 60 games | majority-scoped + roll-tolerant sibling | 4 | 4 |
| retained archive (3,496 surviving rows) | registered | 883 | 776 |
| retained archive | majority-scoped | 607 | 495 |
| retained archive | majority-scoped + roll-tolerant sibling | 451 | 339 |

Reproduce with `scripts/c26_structural_residue_diagnosis.py`; the emitted
`reports/c26_structural_residue_diagnosis.json` pins the classifier hash and the
probe report it read. The probe is a **diagnostic** on seed band 17,000,000
(filed as consumed), run *after* the C26 contract was registered, so the
contract's `fresh_measurements_inspected_before_registration = 0` still holds.
It is not certification evidence: the binding measurement remains the registered
10,000-game sweep on the C26 blocks at 16,000,000, which has not run.

## Three defects in the rule, in order of impact

### 1. It fires from minority branch arms

`attribute_row` scans **every** branch arm for a component-count mismatch:

```python
if cls == "roll_scaled_component":
    for miss in misses:
        miss_obs, miss_eng = _miss_pairs(miss)
        if len(miss_obs) != len(miss_eng):
```

**22 of the 29** fresh rows have the deciding mismatch on a minority arm only —
the majority arm (median mass 84.4%) has no count mismatch at all and complains
about something a documented family explains. Seed 17000001/1 is decided by a
6.25%-mass arm.

Every neighbouring rule is majority-scoped, and I4 states the reason in its own
comment: *"A tie confined to a minority arm cannot explain the majority-arm
complaint."* The structural rule is the only documented-family-preempting rule
outside the marked new-mechanism block that reasons from a minority arm.

### 2. It precedes the documented families it overlaps

The rule sits ahead of I1, I2 and I5. **26 of the 29** fresh rows carry a
`_to_full` cap shape in the majority miss — which is I1_cap_state_shape's
literal trigger (`if "_to_full" in majority`). Those rows never reach I1.

This inverts the ordering standard this repository already recorded once, in the
Z12.6 amendment to `docs/engine_divergence_ledger_20260728.md`: *"a narrow
mechanism shape must be tested BEFORE any rule wide enough to swallow it."* Here
a wide rule precedes narrow documented families.

**Do not treat this one as a free repair.** Moving an UNATTRIBUTED rule after
attributing rules mechanically reduces the unattributed count — that is exactly
the shape of widening attribution to pass a gate. Measured on the archive, full
reordering drives the structural counter to *zero* and inflates I1 from 50 to
508, which reads as over-correction rather than a fix. Defect 1 is the
defensible minimal change; the ordering question needs its own per-shape
argument.

### 3. The sibling-carry check is stricter than the matcher

```python
if Counter(sibling_engine_components) == observed_components:
    return True
```

`_sibling_arm_carries_observed_components` compares roll-scaled components by
**exact magnitude**, while the matcher that produced the miss accepts a
roll-scaled component if it is equal, in the enumerated legal roll set, or
within ±9%. So a sibling arm carrying the observed shape at a different legal
roll reads as absent. Seed 17000025/32:

```
pct=46.88: observed=[('', -28), ('heal_to_full', 28)]  engine=[('', -30)]
pct=46.88: observed=[('', -28), ('heal_to_full', 28)]  engine=[('', -30), ('heal_to_full', 30)]
```

The second arm carries the observed component labels exactly; only the roll
differs, and the matcher already treats that roll as interchangeable. Aligning
the two tolerances removes 3 more fresh rows and 156 archive rows.

## What survives, and what it turned out to be

Four fresh rows (~0.07/game → ~**670 per 10,000 games**) and 339 archive rows
survive every scoping above. Replaying them against the engine splits them in
two: **most are a matcher artifact, and a hard core is a genuine engine gap.**

**26 of the 29 fresh rows carry a capped-heal component** (`heal_to_full` or
`itemleftovers_to_full`), and **21 of 29 have a majority arm whose net HP is
identical to the observed transition**. Those rows differ only in how a capped
heal is decomposed, because a heal that lands exactly on the HP cap absorbs the
damage roll by construction:

```
pct=93.75: observed=[('', -105), ('heal_to_full', 165)]   # net +60
           engine  =[('', -98),  ('heal_to_full', 158)]   # net +60
```

Same end state, same status, same weather. Only the intra-turn roll differs,
and the cap swallows it. Under the banded matcher these boundaries match; they
are strict-matcher-only divergences over a documented cap shape — I1's
territory, exactly as its own basis string says ("capped-heal component shape in
the majority miss (avg-roll world evolution)"). They are made UNATTRIBUTED by a
6.25%-mass `capped_lethal` sibling arm, not by the majority complaint.

### The count-mismatch subset is a bucket-migration bug

`_split_component_events` routes a component into the roll-scaled list iff its
source is in `_ROLL_SCALED_SOURCES` **or ends with `_to_full`**. Whether a heal
caps is a function of the damage roll — so *the length of the roll-scaled list
is itself roll-dependent*. `roll_component_events_agree` then rejects on length
before applying any tolerance:

```python
if len(observed) != len(engine):
    return False
```

Seed 17000013/12, replayed: Delcatty at 214, Earthquake for -84, a pending Wish
for +145, Leftovers +15 landing exactly on 290/290. Observed rolled list is
`[('', -84), ('itemleftovers_to_full', 15)]`. The engine's 93.75% arm rolls -90,
so after Wish the mon sits at 269 and Leftovers does *not* cap — the same
Leftovers heal is labelled plain `itemleftovers` and lands in the **exact**
bucket instead. Rolled lengths become 2 vs 1 and the row is rejected as
structurally different.

It is not. -84 is inside the engine's own tolerance for -90 (0.919·90−1 = 81.7 ≤
84 ≤ 1.09·90+1 = 99.1), so the matcher would have accepted the roll had it ever
compared them. Two arms differing only by a legal roll are declared structurally
different because the cap label moved a component between buckets.

### The three genuine outliers are a documented limit, mis-filed

Seeds 17000001/1, 17000028/53 and 17000037/55 are all **crit-arm** boundaries
where the roll changes a downstream discrete outcome:

- 17000001/1 — Alakazam takes a crit Thunderbolt to 4/209 and survives to take
  Leftovers; the engine's crit arm prices the same hit as lethal, so it emits no
  Leftovers heal at all.

  The crit *rate* is not the problem. Replaying the state gives exactly the
  right three arms:

  ```
  pct=84.38  Switch SideOne: P0 -> P5 | Damage SideOne: 106 | Heal SideOne: 13
  pct=9.38   Switch ... | Damage SideOne: 106 | ChangeStatus ... PARALYZE | Heal SideOne: 13
  pct=6.25   Switch ... | Damage SideOne: 209 | ToggleSideOneForceSwitch
  ```

  6.25% is exactly 1/16, the Gen 3 normal crit ratio, and the non-crit arm
  splits 90/10 for Thunderbolt's paralysis chance. The engine models the crit
  *rate* correctly and explores the crit arm. What each arm carries is a single
  representative damage value — 106 non-crit, 209 crit (the 2× Gen 3 crit
  multiplier) — not the roll spread. Showdown's crit rolled 205.

  Ordinarily that is harmless: the matcher accepts a magnitude inside the legal
  roll set or within ±9%. But a roll difference that crosses a **discrete
  threshold** cannot be absorbed by a magnitude tolerance. 209 ≥ Alakazam's 209
  HP faints it and cancels the Leftovers tick; 205 leaves it at 4 and the tick
  happens. One HP of roll changes the whole downstream event sequence.

  This is where `limit:roll_divergent_lethality` normally catches such rows —
  and the category that family asserts is itself questionable. See
  "Is roll-divergent lethality a comparison limit at all?" below.
- 17000028/53 — the same shape on a crit Hidden Power into Banette.
- 17000037/55 — Gyarados is crit to 58/258, which is below ¼ max, so its
  Substitute *fails*; in the engine's non-crit arm Gyarados is at 154 and
  Substitute succeeds, costing 64. Hence `engine=[('', -88), ('', -64)]` against
  `observed=[('', -184)]`.

These are roll-divergent lethality and its Substitute-viability cousin. The
`limit:roll_divergent_lethality` family already exists and is registered with a
bound, and the structural rule files them as an unexplained mechanism instead —
but routing them to that family is at best a better label, not a fix, for the
reason set out below.

The reason they never reach that family is worth stating precisely: the
lethality difference surfaces as an **absent residual component** (no Leftovers
tick, no Substitute cost) rather than as a `capped_lethal` component. So
`classify_divergence` labels the row `roll_scaled_component`, and the structural
rule — which triggers on exactly that class — claims it before any documented
family is consulted. A boundary whose whole story is "the representative roll
landed on the other side of a KO threshold" is reported as an unexplained
mechanism.

## Is roll-divergent lethality a comparison limit at all?

`classify_divergence` files these rows with an explicit category claim:

```python
# This is a limit of the comparison, not an engine fault; named so it can be
# excluded explicitly rather than sitting in an anonymous bucket.
return "limit:roll_divergent_lethality"
```

That claim should not be taken on trust, because it decides whether ~1,275 rows
per 10,000 games — the largest registered family in the contract — are tolerated
divergence or unfixed engine infidelity.

The differential's own stated question is whether the transition Showdown took
*lies in the branch support* `generate_instructions` enumerates. For these rows
it does not. And the reason it does not is a property of the engine, not of the
comparison: `generate_instructions` enumerates the crit dimension with exact
mass (6.25%) and the secondary-effect dimension with exact mass (10% paralysis),
then **collapses the 16-value damage roll dimension to one representative**.
Where that collapse only moves a magnitude, the matcher's legal-roll window
absorbs it and nothing is lost. Where it crosses a discrete threshold, the
support is genuinely missing a reachable state.

So within the engine's own design language — branches carrying probability mass
— omitting the roll dimension is an incompleteness of the support. Calling the
resulting mismatch a limit of the comparison records the symptom in the wrong
category. Collapsing rolls is defensible as a search-tractability decision; it
is not a reason the engine is faithful.

The consequence for the program is concrete: a PASS under the current contract
would certify an engine while tolerating roughly 1,275 transitions per 10,000
games that its branch support does not contain. The registered bound is not bad
arithmetic — it is the wrong category, and it is the largest one.

There is one residual sense in which "comparison limit" is fair: from a single
boundary you cannot distinguish "the engine cannot reach this state" from "the
engine could reach it but this branch did not." Enumerating the threshold split
resolves exactly that ambiguity, which is why the fix is a branch-set change
rather than a wider tolerance.

### The fourth candidate change is an engine patch, not a classifier tweak

Split a damage branch **only** where the legal roll set straddles a discrete
threshold — principally `damage ≥ the defender's current HP`. The split is exact
rather than sampled: the rolls are the 16 legal values, and the masses are
`(#lethal)/16` and `(#non-lethal)/16`. It costs nothing at the boundaries where
it changes nothing. `poke_engine.calculate_damage(state, c1, c2, True)` already
returns that roll vector and the matcher already consumes it via
`legal_roll_damages`; the information simply does not reach
`generate_instructions`.

On 17000001/1 this turns one 6.25% arm asserting a KO into two arms — crit and
it dies, crit and it survives to take the Leftovers tick — and the observed
transition lands inside the branch support instead of outside it.

Sampling rolls is not an alternative. The certification rests on exact replay:
`cert_sweep_reread.py` re-evaluates retained rows against a rebuilt engine and
its validation gate requires every archived row to re-read identically on the
same fingerprint. A sampled branch set would make the same row classify
differently on two runs, and no clearance could be distinguished from noise.

This is an engine change, so it needs a new fingerprint, a new source freeze,
and a re-registration — the sequence the handoff already prescribes for a patch
that resolves a certification failure.

### Where the archive disagrees, and why that is expected

On the retained archive the same breakdown gives only 36 net-identical majority
arms out of 776, against 21 of 29 on fresh games. That difference is a sampling
artifact and should not be read as two contradictory results: archive rows are
*conditioned on the 41-patch engine having diverged there*, which enriches for
large disagreements, while a fresh sample sees the current engine's actual
divergence mix. The fresh sample is the population the certification measures,
so it is the one that predicts the sweep.

## Fix specification for C27

Three independent changes, each justified from the code's own stated semantics
rather than from the count it produces:

1. **Scope the structural rule to the majority arm**, matching I4's stated
   rationale that a minority arm cannot explain the majority-arm complaint.
2. **Make bucket membership roll-independent.** Classify a heal by its source,
   not by whether it happened to cap, so the roll-scaled list length stops
   varying with the roll — or pair components by cap-normalized source before
   the length check.
3. **Align the sibling-carry tolerance with the matcher's** — labels plus the
   legal-roll window, not exact magnitudes.

A fourth question is open and should not be bundled in: whether crit-arm
lethality and Substitute-viability divergences belong in
`limit:roll_divergent_lethality` or need their own registered family.

Predict each change's effect **before** measuring it. These counterfactuals were
computed after the fact and are diagnosis, not a registered prediction.

## Consequences for the certification program

1. **C26 cannot pass**, and no classifier rescoping makes it pass: ~670
   unattributed rows per 10,000 games survive the most generous defensible
   scoping. Certification requires zero.
2. **The C26 registration stays as registered.** Its contract is immutable and
   its 16,000,000 seed blocks are unspent. Nothing here amends it.
3. **Any repair is a C27.** `scripts/cert_sweep_readout.py` is pinned by
   `required_readout_sha256` in the C26 contract and by the build-source
   lifecycle, so changing one line of the classifier invalidates the C26 source
   identity. The sequence is: fix, freeze a new source identity, re-register
   contract plus calibration, then sweep — exactly as the handoff requires for
   an engine patch.
4. **Register the prediction before measuring it.** Defects 1 and 3 are
   justified from the rule's own semantics and can be pre-registered on that
   basis. Defect 2 cannot, yet; it needs a per-shape argument that does not
   reduce to "it lowers the count."

## What was not done

The registered sweep was not launched. That is a deliberate hold, not an
omission: launching would spend the reserved blocks to obtain a FAIL this
document already predicts from a sample 167 times cheaper, and any fix requires
a new registration cycle regardless. Recording the binding FAIL remains
available if a formally attested negative result is wanted.
