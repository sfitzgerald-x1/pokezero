# C154 — section 4 re-adjudicated: 26 verdicts survive, 13 reasons do not

**Date** 2026-08-09 · **Base** `origin/main` at `2e1e1866` · **Instrument**
`scripts/c154_unreachable_readjudication.py` → `reports/artifacts/c154_unreachable_readjudication.json`,
pinned by `tests/test_unreachable_readjudication.py` · **No new sweep was run and none was needed.**

## 0. The question, and why it was worth asking

C153 adopted a rule for unreachability claims and wrote it into
`scripts/c153_wide_negative_census.py` above `CENSUS_CANNOT_REACH`:

> Trace the raise site to the caller that actually reaches it — not to a plausible sentence
> about it.

Applied to that map's seven entries it corrected **three**: one wrong verdict
(`public_effect_blocked`) and two wrong reasons (`deferred_opponent_action`,
`rest_sleep_refund_pending_precounts_legacy`). A fourth instance appeared *outside* the map, in a
granularity split written one commit after the rule was adopted.

`reports/c138_known_gaps_ledger.md` §4 holds **26** UNREACHABLE verdicts (R1–R27, R26 withdrawn)
that had never been through it. They are neither counters nor sweeps, so neither of the two
existing census instruments had ever looked at one; they were carried on prior work's word since
C138. At 3-in-7 that is not a formality.

## 0b. Review round, 2026-08-09

⚠ **Three of this pass's own load-bearing sentences were the defect it was removing, and the rule it
wrote down was evaded three more ways.** Review verified the substance independently — regenerating
the artifact against the live Showdown checkout at `f76228a1` produced **0 diffs** across the pool
block and all 27 verdict records, and all 26 verdicts stand — and then found that the corrections
had the same shape as the rows they corrected:

- R10's correction opened *"this cell reasoned from its NAME without tracing its caller"* and then
  asserted an untraced caller graph. §3 and the artifact now carry the derived one.
- The artifact's placement rested on a collision that **does not exist**, measured, with a control
  that **could not fail**. §5 and §6.
- The generator's docstring stated the corrections tally as SEVEN and as TEN against THIRTEEN — the
  fifth and sixth instances of this pass's own subject, inside the generator producing the number.
- The phrase guard fell to markdown emphasis, U+00AD, U+200B and an in-cell re-assertion. §5.

Every one is corrected in place and pinned, and the nine mutations review's findings imply are now
battery members 13–21. Nothing in the adjudication itself changed: 26 verdicts, 13 corrected reasons.

## 1. Result

**All 26 verdicts survive. Thirteen of the stated mechanisms do not** — four outright false, nine
incomplete. That is the `deferred_opponent_action` shape thirteen times over and the
`public_effect_blocked` shape zero times: **no §4 row turned out to be reachable.** Nothing was
closed; nothing new opened. The ledger is exactly as open as it was, with thirteen fewer sentences
a reader could rely on.

| Verdict | Rows | Reason SOUND | INCOMPLETE | FALSE |
|---|---|---|---|---|
| UNREACHABLE, traced | 26 | 13 | 9 | 4 |
| NOT OBSERVED AT SCOPE | 0 | — | — | — |
| WRONG | 0 | — | — | — |

Corrected: **R1, R4, R6, R7, R8, R9, R10, R14, R21, R22, R23, R24, R27**. Sound as written:
**R2, R3, R5, R11, R12, R13, R15, R16, R17, R18, R19, R20, R25**. R26 was already withdrawn and is
carried with no verdict of its own, because an inventory that silently drops a name is how a
"closed" row turns out to be a fourth category in disguise.

The per-row verdicts, demonstrations, corrections and measurements are in
`reports/artifacts/c154_unreachable_readjudication.json`. **This report deliberately does not restate
them**: a correction applied to data and not propagated to the prose describing it is a defect this
document has shipped three times, so the ledger cells and this report point at the artifact and the
pin holds all three against each other.

## 2. The four false sentences

**R4 — "The expiry path has no trigger."** It fires on every Rain Dance (7 species) and Sunny Day
(4), and `weather_survives_upkeep` evaluates false on their expiring turn. What has no trigger is
the narrower thing the row is titled after: a *chipping* weather holding a *finite* counter. Rain
and sun do not chip; sand and hail are the only chipping weathers and neither can hold a finite
counter in this pool. Two neighbouring over-readings go with it — permanent (`-1`) weather is not
Tyranitar-only, and the payload-seeding lane is closed by a different mechanism than the row cites.

**R9 — "The Liquid Ooze guard inside `residual_heal_cause` is therefore dead code."** A non
sequitur from the row's own previous sentence. The negative-heal interception is real and verified;
the guard is not in a negative-heal branch. It is the conjunct `ability != Abilities::LIQUIDOOZE`
inside the LEECHSEED arm, which runs only on a **positive** heal — exactly the heals the
interception lets through — and `residual_heal_cause` takes no heal amount, so the sign is not a
fact it can see. It is not dead in the literal sense either: the crate **pins** it, with a positive
heal and no Leftovers, and that pin exists because deleting the guard once left the suite green.
What forecloses the conjunct *in this format* is the two earlier returns plus the item universe.
The row also says the interception is what makes the row unreachable "alone"; there are three
ooze-aware sites, one of which — the move-phase renderer — is **reachable**.
`reports/c131_leechseed_heal_label.md` §5 is the origin of the false sentence and is struck in
place rather than deleted.

**R22 — "This closes … the `failencore` move-list edge cases."** `move_fails_encore` matches
`ENCORE | MIMIC | MIRRORMOVE | SKETCH | STRUGGLE | TRANSFORM`, and R22's eight names cover three of
the six: `encore` is 16 of 220, `transform` is 2, and Struggle is reachable by PP exhaustion. Half
that list is live in ordinary play. **Nothing opens, and that was checked rather than hoped:** the
crate's six are exactly the non-`Future` gen3 moves carrying Showdown's `failencore` flag, which is
the condition `encore.condition.onStart` actually tests. The clause is withdrawn; the patch is not.
Separately, "closes `_HIDDEN_INFORMATION_REQUEST_FLAGS`'s `maybeDisabled`/`maybeLocked`" uses the
wrong verb — that frozenset is a *tolerate*-list, so those flags never caused a refusal.

**R23 — "cannot fire from play."** `volatile_unsupported: side 'p1': ['taunt']` fires **today**, on
this repo's own scenario corpus (`struggle_taunt_stall`, recorded in
`docs/belief_edge_case_matrix.md`). The correct scope is "cannot fire from *sampled randbats*
play".

## 3. Three failure families, and the rules they earn

**One error class accounts for three rows at once, and it is R26's.** R6, R10 and R27 all closed on
"`getItem` cannot return it" — a **generation-time** argument that says nothing about acquisition
in play. That is the exact gap that made R26 wrong, in this document, about this mechanic, in
adjacent rows, and §8's R26 rule did not generalise to it because the rule was written about move
*targets*. The universe is closed at runtime too — `trick` (2 of 220) swaps, `knockoff` (4) removes,
`thief`/`covet`/`recycle`/`switcheroo`/`bugbite` are each 0, and no gen3 mechanism creates an item —
but nothing had said so.

**Three rows rested on an enumeration that had not been run.** R23 named one producer of the
`foresight` volatile where the dex has two (`odorsleuth`), and none of the `focusenergy` producer
that exists in gen3 as an item (Lansat Berry, whose `onEat` calls `addVolatile('focusenergy')`).
R14's Wonder Guard argument never asked whether the opponent can *remove* Wonder Guard — Skill Swap
and Role Play, both 0 of 220, closed but by nothing until measured — and its exhaustive
type enumeration is **vacuous**, because every pool move except Bonemerang is single-hit. R10
reasoned from a bucket's *name* and never traced its caller: every production path into
`heal_subcase` runs through `render_move_phase`'s Sleep Talk block, and **0 of 393 sets** pair
`sleeptalk` with a drain move, so the bucket is *unemittable*, not merely unambiguous. ⚠ **The
first revision of that correction made R10's own mistake in the sentence that names it**, asserting
untraced that the bucket is reached only through `ambiguous_unrenderable_slug_with_protect`. There
are two routes — the `sleeptalk_refusal_is_unsafe_with_protect` predicate at `:2138` as well as the
slug emit at `:2159` — and review found the second. The conclusion held; the sentence did not, and
nothing re-derived it. The caller graph is now built by reverse reachability and pinned.

**And one rule that is the R26 rule's missing half.** *A whole-pool "0 of 220" is side-independent;
a per-species one is not.* §8 has read since C138 as though every movepool check were suspect for a
`target: normal` move. R26 was wrong because it scoped the check to **two species**, not because it
used a movepool. Eighteen §4 rows are whole-pool absences and are safe for that reason. Saying so is
what let this pass spend its attention on the eight that are not.

The third new rule — *cite which guard* — is R9's: the conjunct text it cites occurs twice in
`events.rs`, and the row collapsed a dead guard and a live one into a single sentence.

## 4. Denominators, and why none of them is quoted as a bound

Five rows close a differential counter: R1, R7, R8 and R23 close
`skip:world_unsupported:<reason>` and R2 closes `world_prestate_mismatch:weather_HAIL`. For those
the artifact records the **AST-derived emission sites** and the **denominator name**, and the pin
re-derives both. `skip:world_unsupported:` has **two** increments, both in `_prepare_boundary`, so a
row citing "the" emission site would be inadmissible for the same reason a refusal reason with three
raise sites cannot be cited by line. Both patterns' denominator is **`boundaries_full_round`** —
these refusals fire *before* `boundaries_measured` increments — which on C153's strict arm is
**658,559** against **641,866** measured boundaries. Quoting the wrong one understated a result by
~80× once already.

No bound is quoted, because **no row resolved to NOT OBSERVED AT SCOPE.** Every one of the 26
resolved by tracing or by pool census, so a rule-of-three bound would be a number attached to a
verdict that does not rest on a measurement. Where a scope is mentioned it is C153's committed
strict arm (8,000 games, unregistered seeds `1,001,000`–`1,008,999`), cited as the widest scope the
program has and never as a new measurement.

## 5. What the pin covers, and the four things it cannot

`tests/test_unreachable_readjudication.py`, 31 tests, gated in
`.github/workflows/engine-fidelity-gates.yml` with an exact `Ran 31 tests` guard and a no-skip
check. Its load-bearing property is that **every demonstration is re-derived from source, byte for
byte, on every run** — `_anchor` / `_anchor_after` / `_raise_line` are *imported* from C153's census
rather than copied, so a stale anchor anywhere in the traced set is one loud failure. #1202 moved
fifteen citations in a single merge; that is the shape this exists to prevent. Three further claims
of this pass's own are re-derived rather than asserted, all three because review found them
over-claimed: R10's caller graph, the corrections tally, and R22's `move_fails_encore` set
equality.

It also pins the ledger prose against the artifact **in both directions** — a corrected row must
carry the C154 marker and an uncorrected one must not — and holds the four retracted sentences to a
quoting rule with three smuggle fixtures against it.

Four judgements are **human readings** and are named as such rather than covered by a green test:

1. R1's closure is that a committed Future Sight scenario sits in `interaction_registry_specs()`
   rather than `scenario_specs()`. The pin asserts exactly that, and asserts it reddens if the spec
   moves — but "no harness ever passes `specs=` explicitly" is a claim about every present and
   future caller. **This closure is one keyword argument deep**, which the ledger did not say.
2. R9's enumeration of the gen3 residual positive heals available to a non-Leftovers holder is an
   argument about a mechanic, not a grep.
3. R22's judgement that the crate's `move_fails_encore` list being gen3's `failencore`-flagged set
   *makes the shipped list correct* is not machine-checkable. ⚠ **The set equality is, and this
   report claimed it was while the pin asserted six memberships** — review defeated that by adding
   `Choices::TACKLE` to the arm and watching the module stay green. Both sides are measured now: the
   arm is parsed out of the committed patch and compared for equality against the dex-derived
   `failencore` set, with the TACKLE smuggle as a fixture.
4. R7's scenario-studio residual is closed by "the service never builds an engine world", an
   absence over one module. The pin asserts it by grep and the assertion message states the scope
   of the negative: the text of `src/pokezero/scenario_studio/service.py`, nothing wider.

**The pool half is not re-derived in CI.** CI builds no pokemon-showdown checkout, so the `pool`
block is a committed measurement at Showdown `f76228a1354b5d0f307ca2d16101294ad3a2308b`, recorded on
the artifact. A `sets.json` bump that added `taunt` to a set would leave the module green and the
ledger wrong. Bounded and nameable, exactly as `scripts/c152_pool_reachability_census.py` records
for its own artifact.

**Mutation battery: 21 applied, 21 caught**, enumerated in the pin's docstring. ⚠ **Number 8
survived the first version** — a retracted sentence re-inserted hard-wrapped *and lower-cased* walked
past a guard that already normalised whitespace, which is C153's own mutation 24 defeating a guard
written by someone who had just read about it. ⚠ **And 13–21 are review's**, added after this module
shipped claiming 12 of 12: a widened `foreclosure`, a removed caller-graph edge, `Choices::TACKLE` in
the encore arm, the artifact moved back out of the corpus, the tally detached from the records, and
four further evasions of the phrase guard — a re-assertion *inside* the correction's own cell, and
the same sentence written with `**bold**`, with U+00AD and with U+200B, **in a markdown file**. Each
of the nine corresponds to a sentence of this pass's own that was asserted beyond what it traced, so
they are a separate block rather than renumbered in.

The phrase guard now folds whitespace, case, markdown emphasis and zero-width characters, and each
of those four was added after something got past the previous three — the same instance-not-class
repair `reports/c131` §6 records about its own author. The in-cell re-assertion is *not* caught by
the quoting rule and cannot be, being indistinguishable from the legitimate shape; it is caught by an
exact per-phrase occurrence inventory, which is a second pin rather than a stricter first one.

**Test evidence.** `python -m unittest tests.test_unreachable_readjudication -v` → **Ran 31 tests,
OK**; the gated census family together → OK; exact `Ran N` guards re-checked individually —
never-fired **22**, wide-seed **36**, ledger uniformity **19**, this module **31**.

**The pin is red against the uncorrected ledger: 20 failures**, itemised — 13 missing C154 markers,
4 unquoted retractions, 1 occurrence-inventory mismatch, 1 corrections-tally mismatch, 1 `81 != 82`.
⚠ **A first revision of this report said 21 and that was wrong.** Against the 25-test version review
measured **18** (13 + 4 + 1), which reproduces exactly here. Every class fires; the count did not.

**Full `pytest tests/` adds nothing, and the absolute figure is environment-dependent, so the DELTA
is what is quoted.** Command: `pytest tests/ -q -p no:randomly --continue-on-collection-errors` — the
flag is required, because plain `pytest tests -q` aborts at collection where `numpy`/`torch` are
absent. Measured here: **157 failed at base, 157 failed at head.** Review measured **164 / 165** in a
different environment and identified the +1 as a load-sensitive liveness test that passes 3/3 in
isolation on both trees. Both agree on the only claim resting on this: **this PR adds zero
failures.** ⚠ A first revision quoted the bare 157 as though it were a property of the repository
rather than of one machine.

## 6. Base state, re-derived rather than trusted

| Quantity | Expected | Measured here |
|---|---|---|
| `main` | `2e1e1866` | `2e1e1866` ✓ |
| patch stack | 74 | 74 ✓ |
| `_EXPECTED_SWEEP_ARTIFACTS` | 115 | 115 ✓ |
| `_EXPECTED_COUNTER_ARTIFACTS` | 401 | 401 ✓ |
| §3 rows | 82 | 82 ✓ |
| engine fingerprint | `bfdbe1c04876edcd` | ⚠ **`9517aab98d56a9ba`** |
| pre-existing pytest failures | — | 157 base / 157 head (this environment) |

⚠ **The engine fingerprint has moved, and it moved legitimately.** `bfdbe1c04876edcd` is the
build at `7fcd9e19`, where C153's twelve shards were taken. Between there and `2e1e1866`,
`rust/pokezero-search/src/leaf.rs` gained 31 lines — one of the 11 crate sources the fingerprint
covers — so a rebuild of the committed tree at `main` now stamps
`9517aab98d56a9babe0f261cd83685b9fce20e513c8c5a023436b20052b72cda` at the same 74 patches. ⚠ **A
first revision of this table credited that change to #1202 (`49c31855`), which touches no `.rs` file
at all.** It is `21f484d4` (**#1197**), and that is not a guess either: `git log 7fcd9e19..2e1e1866`
restricted to the fingerprint's own inputs — the gen3 patches, `patches.txt` and
`rust/pokezero-search/src` — returns exactly one commit. Misattributing a commit inside a report
whose subject is untraced citations is the same defect one surface over, so it is corrected in place
rather than silently. No
committed artifact carries the new fingerprint yet, because this pass ran no sweep. Nothing in this
change depends on the engine build; it is recorded because the task named the old value as base
state to re-derive, and re-deriving it is how the move was found.

⚠ **This pass writes one artifact, and the first revision put it in the wrong place for a
measured-false reason.** It argued that under `reports/` the artifact's refusal-reason names beside
pool counts would read to `tests/test_never_fired_counter_census.py` as four counters firing, and
wrote it to `tests/data/` instead. **Review measured it: in the corpus, that census reports
`Ran 22 tests … OK`.** The names appear only as string *values* inside prose, and `_evidence_in`
admits a dotted-path token or C43's sibling-field shape and says in terms that "A name merely
mentioned inside prose is NOT evidence" — an exclusion written long before this pass assumed
otherwise. The placement bought nothing and **cost** the guard that census's own header warns about
by name: *"A future census written to `tests/data/` would leave the corpus and lose the check with no
test going red."*

Worse, the control meant to prove the hazard real was `assertIn(reason, json.dumps(document))` —
substring-in-prose, the exact shape the matcher excludes — so it **could not fail**. A hazard
asserted rather than measured, with a control that cannot fail, in the instrument built to remove
exactly that: the third and fourth instances in this pass of its own subject. The artifact is in
`reports/artifacts/` now (corpus 401 → 402, bumped by the four-step procedure with the set
difference confirmed as exactly one file and nothing removed), membership is asserted, and the
control feeds `_evidence_in` a counter-keyed copy and requires it to fire.

## 7. What this pass did not do

- **It did not manufacture terminality.** §4 is not "closed"; it is re-adjudicated. Thirteen rows
  now carry a correction, and three of the new standing rules exist because the same class of
  reasoning error is still being made in this document.
- **It ran no sweep**, and says so rather than dressing a trace as a measurement.
- **It did not touch** `OWNER_RATIFIED`, `RATIFIED_SWEEP_PRECONDITION`, `RATIFIED_FINAL_HOLDOUT`,
  `BURNED_FINAL_HOLDOUT`, the burn, C141's demotion, or
  `reports/c151_final_holdout_rereg_prediction.md`, and it swept no seed at all.
