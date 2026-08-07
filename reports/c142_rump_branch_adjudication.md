# C142 — the recoil magnitude row is a rump-branch adjudication artifact

Diagnosis of `19200131/129` in `reports/artifacts/c141_final_holdout_sweep.json`, reported as

```
pct=6.25: p1 roll-scaled components differ: observed=[('recoil', -18)] engine=[('recoil', -32)]
```

> The `c141` artifact and its prediction are now **on `main`** (PR #1159 merged), so every
> citation here resolves against a committed file. Nothing in this diagnosis re-measures that
> window — every number comes from the artifact's own retained state, from generated Showdown
> fixtures, or from the two development windows.

## Verdict

**HARNESS. Not an engine gap, and not a recoil defect of any kind.**

The boundary **matches on its full branch set**. Its 93.75 % arm was dropped before
comparison by `strict:lossy_render`, leaving the 6.25 % crit arm as the only thing the
observation was compared against. Both recoil magnitudes in the miss string are individually
correct gen3 arithmetic; they describe different arms of the same fork.

The verdict rested on **6.25 % of the enumerated probability mass**, and that number was
invisible in the artifact.

This is a considerably more important finding than a recoil bug, because it is not specific
to recoil: **any** boundary with a lossy-rendered branch can be reported divergent for a
reason that is not a divergence. The mechanism was already written down —
`docs/engine_divergence_ledger_20260728.md` — and never had a counter.

## 1. What actually happened, by replay of the retained state

`repros_complete: true`, so the retained state is the whole input. Replaying it (no
re-measurement of the reserved window) recovers both arms:

| | mass | render | p1 components | p2 components |
|---|---|---|---|---|
| non-crit arm | **93.75 %** | **dropped** — `attract_empty_tail_ambiguous:paralyzed+cannot_act` | `('recoil', -19)`, `('itemleftovers', 18)` | `('', -57)`, `('itemleftovers', 15)` |
| crit arm | 6.25 % | usable | `('recoil', -32)` | `('capped_lethal', -97)` |
| **observation** | — | — | `('recoil', -18)`, `('itemleftovers', 18)` | `('', -55)`, `('itemleftovers', 15)` |

`poke_engine.calculate_damage` on the pre-state returns p1 bases `[62, 124]` (non-crit,
crit). Gen3 rolls `floor(base * random(85,100) / 100)`, so the non-crit arm's legal damage is
52–62 and the observed 55 is inside it; the observed recoil 18 is inside that arm's window
around its representative 19. The dropped arm is the arm that happened.

Re-adjudicating the same state twice, changing nothing but the allowlist:

```
as shipped (rump branch set)                -> ('diverged', ["pct=6.25: ... observed=[('recoil', -18)] engine=[('recoil', -32)]"])
with that one slug allowlisted (full set)    -> ('matched', [])
```

**The allowlist is exact-set membership, not a prefix.** `branch_render_is_usable`
tests `set(lossy) <= _TELEMETRY_ONLY_LOSSY_MARKERS`, so what restores the arm is the full
suffixed slug `attract_empty_tail_ambiguous:paralyzed+cannot_act` — adding the bare family
name `attract_empty_tail_ambiguous` restores nothing. The distinction matters and is not
pedantry: a *prefix* allowlist would also admit the `noop` and `miss` slugs, which
`rust/pokezero-search/src/events.rs:2576-2650` records as **not downgradeable at any price**
because they erase a `|move|` reveal and the miss arm suppresses a PP decrement the fold
tracks. Only the `paralyzed`-dominant family is downgradeable. The new
`strict:lossy_render_marker:*` counter is deliberately prefix-*split* — that is a
low-cardinality reporting choice about the counter namespace and has no bearing on which
slugs the allowlist admits.

The first line reproduces the artifact's miss string character-for-character, so the replay
is of the same boundary and not a near-miss. The second line is the finding.

**Why only one miss for two branches.** `evaluate_boundary_strict` appends a miss on every
`ok = False` path and returns `"matched"` immediately on a match, so two *compared* arms
would have produced two misses. One miss plus `branch_count: 2` means exactly one arm was
dropped before comparison — which is what the replay shows, and which is recoverable from the
artifact alone.

**The other final-holdout row is genuine.** Replayed the same way, `19200244/115`
(`component_mismatch:heal|itemleftovers`) enumerates 9 branches, drops **none**, and diverges
on 100 % of its mass. So the corrected reading of that window is **1 divergence and 1
withheld boundary**, not 2 divergences.

## 2. The recoil arithmetic is right on both sides

The brief's premise — gen3 Double-Edge recoil is `floor(damage / 4)` — is wrong, and so is
the obvious reading of `data/moves.ts:3885` (`recoil: [33, 100]`), which is the *current-gen*
value. Gen3 inherits gen4 (`data/mods/gen3/scripts.ts`: `inherit: 'gen4'`), and
`data/mods/gen4/moves.ts:368` overrides `doubleedge` to **`recoil: [1, 3]`**. Gen3's own
`calcRecoilDamage` (`data/mods/gen3/scripts.ts:480-481`) is
`clampIntRange(Math.floor(damageDealt * recoil[0] / recoil[1]), 1)` — `floor`, not the
`Math.round` of `sim/battle-actions.ts:1402`.

Measured against real gen3 Showdown on **generated** fixtures (`FixturePokemon` +
`run_multi_turn_fixture`, gen3 Custom Game), 32 non-KO Double-Edge rows across seeds
2000–2031:

| candidate formula | rows agreeing |
|---|---|
| `floor(damage / 3)` | **32 / 32** |
| `floor(damage * 33 / 100)` | 10 / 32 |
| `round(damage / 3)` | 25 / 32 |
| `floor(damage / 4)` (the brief's premise) | **0 / 32** |

22 of the 32 rows discriminate `floor(d/3)` from `floor(33d/100)`, so this is not a
coincidence of small numbers — and the gate refuses to pass on a sample where fewer than a
quarter of rows discriminate, so a future re-run cannot accidentally prove nothing.

Both of these measurements are re-runnable: `scripts/gen3_recoil_differential.py` is
committed with the branch and asserts every number above against ground truth rather than
printing it (artifact `reports/artifacts/c142_recoil_ground_truth.json`).

The engine implements exactly this: `third_party/poke-engine-src/src/gen3/generate_instructions.rs:2397-2417`
maps the stored `0.33` back to integer `damage_dealt / 3` (and `0.25` → `/4`, `0.5` → `/2`),
with `.max(1)` standing in for `clampIntRange(..., 1)`. A prior cycle already fixed this; the
f32 form is still live in `genx`, `gen1` and `gen2`, which is correct for those gens' own
rules but is not what gen3 runs.

Both magnitudes in the miss string check out: `floor(55/3) = 18` and `floor(97/3) = 32`.

## 3. Recoil comes off the CAPPED damage — measured, both sides

Showdown: gen3 sets `move.totalDamage = damage` from `this.moveHit(...)`
(`data/mods/gen3/scripts.ts:456-457`), and `moveHit` returns what `battle.damage` actually
applied, which `Pokemon.damage` clamps to remaining HP. Rather than rest on that reading, the
same attacker, move and target species were run at two different target HPs:

| target HP before | outcome | recoil | `floor(hp/3)` |
|---|---|---|---|
| 221 (full) | faints | **73** | 73 |
| 21 (chipped by two Seismic Tosses) | faints | **7** | 7 |

The damage *roll* is identical in both runs — same attacker, same move, same target species.
Only the target's remaining HP differs, and the recoil moves with it. Recoil is therefore
computed from the capped damage dealt, not from the uncapped roll. (An uncapped reading
predicts the same recoil in both rows.)

The engine agrees: the crit arm's uncapped base is 124, so an uncapped recoil would be
`floor(105/3) … floor(124/3)` = 35–41. It emitted 32 = `floor(97/3)` against a
`capped_lethal` of 97 — the target's exact remaining HP.

**This also resolves the "suspicious factor near 1.78".** `32/18 = 1.778` is not a clean 2×
crit ratio because the crit arm's damage was cap-limited to 97 rather than its uncapped
105–124: `97/55 = 1.764`. The factor is an artifact of comparing a cap-limited crit arm
against a non-crit observation, and carries no information about a multiplier bug.

## 4. Baton Pass is incidental; Substitute and the KO are not required

The disqualifying marker is emitted at `rust/pokezero-search/src/events.rs:2576-2650` under
`attacker_attracted && !has_any_effect && called_tag.is_none()` plus **any one** of five
predicates. Two of them are live here: `attacker_paralyzed` (Scizor is paralysed) and
`!move_could_act` (Baton Pass with no live teammate). `_TELEMETRY_ONLY_LOSSY_MARKERS`
(`scripts/engine_transition_differential.py:443-446`) contains no
`attract_empty_tail_ambiguous` entry at all, so **every** predicate combination is
disqualifying. Baton Pass contributes the `cannot_act` token to the slug and nothing to the
outcome; the paralysis predicate alone would have dropped the same arm.

What *is* load-bearing is that the attracted seat's action produced no state change, so the
tail is empty and the engine cannot attribute the immobilisation. Baton Pass into a party
with no live switch target is one way to get there; being immobilised is another.

Ordering is not involved. The artifact records `active_changed: {p1: false, p2: false}`, and
the protocol shows `|cant|p2a: Scizor|Attract` — the target never switched, and the recoil was
computed against the correct, still-active target on both sides.

Neither a Substitute nor the KO is required for the artifact either. The false divergence
needs only that the surviving arm's roll-scaled components fall outside the observed arm's
legal window, and the crit/non-crit gap alone does that (uncapped crit recoil 35–41 against
an observed 18). The cap only fixes the particular number 32.

## 5. Reachability

**The recoil shape is common.** In `data/random-battles/gen3/sets.json` (220 species, 393
sets): Double-Edge appears in **56 sets across 45 species** (14 % of sets). Take Down,
Submission and Struggle appear in **zero** movepools; Volt Tackle in one (Pikachu). Struggle
is still reachable in play as the no-PP fallback — the same window counts
`skip:unmappable_choice:struggle_not_submittable: 205`. So Double-Edge is effectively the
only enumerated recoil move in this pool, and it is not rare.

**The rump shape is the one that matters, and it is unmeasured.** `strict:lossy_render`
across every sweep artifact in the repo:

| window | `strict:lossy_render` | `skip:strict_all_branches_lossy` | reported `diverged` |
|---|---|---|---|
| dev `19,000,000–199` | **0** (counter absent from all **29** dev artifacts) | 0 | 1 |
| validation holdout `19,100,000–199` | **3** (identical on every artifact `c121`→`c138`) | 0 | 0 |
| final holdout `19,200,060–259` (`c141`) | **14** | **4** | 2 → corrected to **1** |

The population is largest exactly where it cannot be re-measured, and it is invisible on both
windows that can be. Two honest caveats:

- `strict:lossy_render` counts **branch drops, not boundaries**, so the number of
  final-holdout boundaries adjudicated on a rump set **cannot be determined without re-running
  the reserved window**, which is forbidden. It can be bounded: that window also records
  `skip:strict_all_branches_lossy: 4`, and each of those 4 boundaries consumed at least one
  of the 14 drops, so at most 10 remain for rump boundaries. With the one proven by replay,
  the bound is **1–10**.

  Those 4 are the **only** `skip:strict_all_branches_lossy` in the sweep-artifact series. Of
  the **71** JSONs matching `reports/artifacts/*.json` on this branch, exactly one —
  `c141_final_holdout_sweep.json`, the artifact this diagnosis is about — carries the counter
  nonzero, at 4; the other 70 are absent or 0.

  ⚠ This sentence previously read "absent from all **64**", which was true when written and
  went false **silently** when `main` merged #1159 and brought `c141_final_holdout_sweep.json`
  into the directory the glob scans. Re-derived after the merge rather than carried forward.
  Worth naming as a class: the artifact-count pin in
  `tests/test_boundary_verdict_partition.py` fails **loudly** when the corpus moves, and this
  glob assertion fails **quietly**, so a prose count over a directory that other PRs write into
  has to be re-run on every merge or dropped. It is **not** the first time the counter has ever
  fired, and an earlier draft of this report said so. As hard counter values
  it is **2** in `reports/c26_structural_probe_report.json` and **2** in
  `reports/c27_structural_probe_report.json`. An older-era diagnosis,
  `reports/c32_fail_diagnosis.json`, records the same phenomenon at **372** under the
  differently named field `coverage_diagnosis.coverage_reducing_skips.strict_all_branches_lossy`
  — corroborating, but not the counter itself, and worth distinguishing because the counter is
  what the partition reads. An **eighth** instance, which neither this report nor the `c138`
  ledger listed until review found it: `reports/c43_coverage_shortfall_diagnosis.json`
  `decomposition.ranked[5]` = `{"rows": 372, "counter": "strict_all_branches_lossy"}` — the same
  372 as `c32`, under the same prefix-stripped name. The claim came from
  `reports/c138_known_gaps_ledger.md:268`/`:304`, which says "has never fired" and lists it
  among never-fired static counters; that is the ledger's error and it has now propagated into
  three separate PRs, including this one. **Do not cite it.** Correcting `c138` belongs to
  #1163 — see §8 for the residue this branch still owes after that lands.
- The 14-versus-3 jump is unexplained. The two windows differ by only ~4 % in boundaries
  measured (16,274 vs 15,579), so it is not a denominator effect. It may be window variance;
  nothing here establishes a trend from three points, and the dev-window row the ledger cites
  (`19000093/51`, "11 branches and ~10 % of its mass surviving") no longer diverges at all,
  so the population has previously been reduced by renderer work rather than grown.

## 6. The change

`scripts/engine_transition_differential.py`, `evaluate_boundary_strict`. The matcher's
contract is **existential**: *some* enumerated branch reproduces the observation. A dropped
branch makes that existential **unverifiable**, not false. So when nothing that survived the
filter reproduced the observation *and* at least one positive-mass branch was dropped, the
verdict is withheld:

- new verdict `skip_rump`, counted as `skip:rump_branch_set`, **never** folded into
  `transition:diverged`;
- `skip:rump_branch_set_surviving_decile:N` records how much mass a withheld verdict had;
- `skip:rump_branch_set_row:{seed}/{step}` indexes each withheld boundary, and
  `report["withheld_repros"]` carries its **state** — see §6.1, this is the part that makes a
  withheld row diagnosable at all;
- `strict:lossy_render_marker:{marker}` attributes each drop — the undifferentiated
  `strict:lossy_render` is why this row's marker had to be recovered by replay rather than
  read off the artifact;
- `strict:diverged_on_full_branch_set` makes the resulting invariant checkable from the
  artifact: **every reported divergence rests on 100 % of its enumerated mass.**

The trigger is `enumeration_incomplete`, which **three** paths set. The lossy-render filter is
the measured one; the other two are `BranchLegalRollError` (per-branch support could not be
priced) and `strict:branch_events_error` (a whole candidate state failed to render). Neither
fed the drop accounting in the first version of this patch, so
`strict:diverged_on_full_branch_set` could have fired asserting "100 % of its mass" over an
enumeration that was not complete. Neither counter appears in **any** artifact in
`reports/artifacts/`, so this is a fail-closed guard on unexercised paths rather than a change
to a live number — but a guard, not a comment, because the invariant is the whole point of the
counter. The state-render path loses the branch list itself, so its surviving fraction is
recorded as `unknown` rather than guessed at 0 %.

The rule is deliberately **mass-blind**. A threshold ("withhold only when the majority was
dropped") would put a tuned constant in front of a semantic question — the existential is
unverifiable at any positive dropped mass, and a 1 % dropped arm is still the arm that might
have matched. This is pinned rather than asserted: a mutant gated on `dropped_mass > 50.0`
passes the five original tests (the measured row dropped 93.75 %) and fails the
minority-drop pin added for exactly that reason. The surviving fraction is *recorded* instead
of gated on, so the population stays visible and the right way to attack it is to make the
renderer lossless, not to pick a number.

`skip_lossy` (every arm lossy) keeps precedence: it is the stronger statement.

### 6.1 A withheld row must stay replayable, or the exit destroys evidence

This is the sharpest thing in the change, and the first version of the patch got it wrong.

**Row `19200131/129` was diagnosable only because it was retained in `repros` as a
divergence.** Under a `skip_rump` exit that keeps nothing but a counter key, it would have
carried no state — and on a reserved window, re-running its seed to recover the state **is the
forbidden measurement**. The exit would have converted "a false divergence, diagnosable" into
"a boundary we declined to judge and can never look at again". On the last clean reserve that
is unrecoverable.

So withheld rows are retained with the same payload a divergence gets — `engine_states` (every
hidden-counter candidate), `choices`, `slot_sides`, `party_display`, `turn`, `pre_features`,
`observed`, `observed_boost_deltas`, `active_changed`, `protocol`, `branch_count`, and the
`withheld_misses` the verdict *would* have reported (whose first entry is the surviving-mass
line).

They go in a **separate** `withheld_repros` list, not in `repros`. That was the original
objection to retaining them and it is answered rather than overridden: `repros`' completeness
contract is `repros_retained == transitions_diverged`, which `scripts/cert_sweep_reread.py:128-132`
enforces with a hard `raise`, and a second population beside it leaves that identity exactly as
it was. `repro_retention` **declares** both symmetrically:

| | divergences | withheld |
|---|---|---|
| population size | `transitions_diverged` | `verdicts_withheld` |
| rows retained | `repros_retained` | `withheld_retained` |
| complete? | `repros_complete` | `withheld_complete` |

**Declaration is not enforcement, and the symmetry stops there.** Two gaps, both stated rather
than papered over:

- `withheld_complete` is written but checked nowhere. `scripts/cert_sweep_readout.py:884` and
  `:1465-1466` require `repros_complete is True` at shard level and have no withheld twin, so a
  truncated withheld population would not fail a readout. The count is still visible
  (`verdicts_withheld` vs `withheld_retained` in the artifact), so the truncation is legible —
  it is simply not gated.
- Nothing committed replays a withheld row end to end.
  `cert_sweep_reread.load_retained_rows` (`:125`, `:133-136`) takes only `data["repros"]` entries
  with `kind == "transition_diverged"`, so withheld rows are **replay-capable** — pinned by a
  round-trip through the matcher from the payload alone in
  `tests/test_rump_branch_adjudication.py` — without a CLI consumer that does it.

Both are follow-ups on the certification gates rather than on this exit, and both are cheaper
to do once a withheld population actually exists to gate. The property that mattered for the
block — that a withheld row carries enough state to be diagnosed at all — holds now.

`withheld_repros` is in `checkpoint_report_binding_failures`' field list, so a stale report
that dropped the withheld population fails certification instead of passing quietly. Records
written before this key existed read as an empty list — `--resume` and `--merge-from` over the
existing checkpoint archive keep working — while a *malformed* value still raises.

### 6.2 `cert_sweep_reread.py --expect diverged` now means what it says

Adjacent defect, found in review. The gate's docstring says "every row must re-read as
diverged", but it failed only on `tally["matched"]`, so **any other** non-divergent verdict
passed silently. `skip_lossy` already had this hole; `skip_rump` would have widened it. Both
are reconstruction infidelity on the sweep's own build, which is exactly what the gate exists
to catch, and both would have been reported as a clean pass.

Fixed as a **complement** — every verdict that is not `diverged` fails — rather than as a list
of known-bad verdicts, so a future verdict is caught when it appears rather than when someone
remembers to extend the list. Pinned in both directions, including an unknown future verdict.

### 6.3 The withheld verdict is a term in C144's mechanized partition

`main` gained #1163 (C144) while this branch was in review. C144 established that the boundary
verdict partition is four-term, **mechanized** it as `verdict_partition_failures()`, and made
`scripts/cert_sweep_readout.py` gate on it per shard. That turns this exit from a documentation
question into an integration requirement: a report carrying a withheld boundary would not
reconcile, and the gate would fail the shard.

C144's own comment in `scripts/engine_transition_differential.py` says what to do — "count the
escape explicitly … and add it to `VERDICT_PARTITION_SCALARS` or
`VERDICT_PARTITION_COUNTERS`, rather than discovering the drift from a report that no longer
reconciles". Done:

- `VERDICT_PARTITION_SKIP_COUNTERS` generalises the single lossy name to the **tuple** of
  post-measurement skip verdicts. `VERDICT_PARTITION_LOSSY_COUNTER` remains as an alias for
  C144's consumers, documented as the first of the skip terms rather than the only one. The
  identity is now five-term.
- `cert_sweep_readout`'s `unadjudicated` sums **every** skip term. Crediting a withheld boundary
  to the denominator would have understated `in_support_rate` by exactly the rows whose verdict
  was withheld.
- C144's structural pin was `len(COUNTERS) == len(SCALARS) + 1`, which a fifth term breaks. It
  is rewritten against the skip tuple, so it keeps holding as that tuple grows and still fails
  if the two vocabularies diverge — plus a new pin that the withheld term is in the partition
  **and load-bearing** (dropping it goes red).
- **Two stale C144 statements are corrected**, both of them premises of its exhaustiveness
  argument rather than decoration:
  - "every `skip:*` counter OTHER than `skip:strict_all_branches_lossy` fires before
    `boundaries_measured` increments" — `skip:rump_branch_set` is a `skip:*` counter, is not the
    lossy one, and fires after.
  - "the verdict domain is closed at three values … `evaluate_boundary_strict` and
    `evaluate_boundary` return only `matched`, `diverged` and `skip_lossy`" — it is now four;
    `evaluate_boundary_strict` also returns `skip_rump`. The *form* of the exhaustiveness
    argument survives (`skip_rump` has its own `continue` and its own counted term), but the
    enumeration it rests on had to move with it.

Both windows' artifacts pass `verdict_partition_failures()` on both sides of the comparison.

#### The membership rule, stated as measured

My first repair of C144's sentence was itself wrong, and it is worth recording because it would
have inherited C144's authority. I wrote: *membership is decided by when a counter fires, never
by its prefix.* **Firing late is necessary but not sufficient**, and the counterexamples were
already in the same comment four lines above the claim:

- `gating:exact` / `gating:support` increment on the line **immediately after**
  `counts["boundaries_measured"] += 1` (`_prepare_boundary`);
- every `strict:*` counter in `evaluate_boundary_strict` fires later still, because that function
  is called after `_prepare_boundary` has returned.

None of those is a partition term. The discriminator is the **second** condition:

> A counter is a partition term iff (1) it increments only after `boundaries_measured` has
> incremented for that boundary, **and** (2) the increment is that boundary's *terminal
> verdict* — at most one per boundary, and mutually exclusive with `transition:matched` /
> `transition:diverged`.

In the code, (2) is literally the `continue` in `run_game` that skips
`counts[f"transition:{verdict}"] += 1`. Verified for both skip terms: each has exactly **one**
increment site in the repo, and each `continue`s before that line. `strict:lossy_render` fails
(2) — it can fire several times for one boundary and leaves the boundary free to receive an
ordinary verdict.

This is pinned as arithmetic rather than left as prose
(`test_firing_after_boundaries_measured_is_NOT_sufficient_for_membership`), and the live
counterexample is **this branch's own validation-holdout artifact**: `strict:lossy_render` 3,
every boundary `matched`, identity closes. Folding `strict:lossy_render` into
`VERDICT_PARTITION_SKIP_COUNTERS` — the mutant a timing-only rule would permit — fails 45 tests,
including the corpus pin on real committed artifacts.

The two-condition form is now written into `reports/c144_boundary_identity_correction.md` §2,
which is the report of record for this identity, together with the retraction of the
timing-only version. The tuple itself was never in doubt and was independently confirmed to be
exactly those two members; what was wrong was the *rule* offered for why.

### Blast radius

- **Direction.** The change can only move boundaries **out of** `transition:diverged` into a
  named skip bucket. It cannot create a divergence and cannot create a match. It reduces
  coverage: `boundaries_measured` is unchanged, so withheld rows shrink the adjudicated
  denominator, which is exactly the "buy a lower divergence count by rendering less" hazard
  the ledger warns about. That is why the bucket is named, counted, per-row identified, and
  reported alongside `matched`/`diverged`.
- **Identity.** `boundaries_measured == matched + diverged + engine_error +
  skip:strict_all_branches_lossy + skip:rump_branch_set`. The ledger's Rule 2 has been
  amended in place.
- **Other consumers.** `scripts/attest_materialized_damage_stats.py` tests
  `verdict == "diverged"` explicitly, so it cannot mistake the new verdict for a divergence.
  `scripts/cert_sweep_reread.py` tallies by verdict string — and its `--expect diverged` gate
  was silently permissive about non-`matched` verdicts; §6.2 fixes that.
- **Checkpoint schema.** One additive key (`withheld_repros`), absent-tolerant on read, so the
  existing checkpoint archive resumes and merges unchanged.
- **Sensitivity.** Pinned by `tests/test_rump_branch_adjudication.py`: with a fully rendered
  branch set that fails to reproduce the observation, the verdict is still `diverged`.

## 7. Verification

Generated / replayed evidence, none of it fitted to the reserved window:

- `tests/test_rump_branch_adjudication.py` — 15 pins on synthetic branch data. The control
  (full set → `matched`); the withheld verdict and its recorded mass; marker attribution;
  unchanged sensitivity when nothing is dropped; `skip_lossy` precedence; a **minority** drop
  (6.25 % dropped, 93.75 % surviving) that a majority threshold would let through; an
  unpriceable branch (`BranchLegalRollError`) counting as a drop; the withheld payload carrying
  every field a replay consumes; a round trip that re-adjudicates from the payload alone back
  to `skip_rump`; `repros_retained == transitions_diverged` untouched while the withheld row
  survives; a report that drops `withheld_repros` — in **both** the emptied and the *absent*
  form, the latter being the only one the compatibility default governs and therefore the only
  one that discriminates it from a broken default; both compatibility halves (a pre-C142
  *record* still aggregating, a pre-C142 *report* still certifying); a **lost candidate state**
  withholding and reporting its surviving mass as `unknown` rather than as a computed 100 %;
  and every candidate state failing still taking the older all-lossy exit.

  Three of these are mutation-verified rather than merely green: a `dropped_mass > 50.0`
  threshold fails the minority pin; deleting `unrendered_states += 1` fails the lost-state pin;
  and defaulting the compatibility checks to the aggregate
  (`report.get("withheld_repros", aggregate["withheld_repros"])`) fails the absent-form pin.
  Each was applied, run, and reverted.
- `tests/test_cert_sweep_reread.py` — 5 new pins on the `--expect diverged` gate: `diverged`
  passes; `matched`, `skip_rump`, `skip_lossy` and an unknown future verdict all fail.
- `scripts/gen3_recoil_differential.py` — **committed**, so the recoil ground truth is
  re-runnable rather than quoted. Exits 0; artifact `c142_recoil_ground_truth.json`.
- Retained-state replay of both `c141` repros.
- `tests/test_transition_differential_matcher.py`, `tests/test_cert_sweep_readout_attribution.py`,
  `tests/test_roll_cascade_predicate.py`, `tests/test_family_bucket_audit.py`,
  `tests/test_matcher_tolerance_promotion.py`,
  `tests/test_engine_transition_checkpoint_provenance.py`, `tests/test_public_invariant.py`,
  `tests/test_c26_archival_recalibration.py` — pass.
- `tests/test_boundary_verdict_partition.py` — #1163's suite, extended (see §6.3) and passing,
  including its exact artifact-count pin. That pin **fired on the merge**, which is what an
  exact count is for; it was re-derived by running the selector over the merged tree rather
  than by adding four, and then confirmed still live by checking that both 73 and 75 fail.

**Full-suite failure delta, measured rather than characterised.** The earlier "7 pre-existing
failures" figure was from a `-k`-filtered selection and should not have been stated as if it
described the suite. Over the whole suite (`--ignore` the one module whose *collection* needs
NumPy), distinct failing ids: **84 at the merge base, 85 at this branch's head**. The single
difference is
`test_golden_corpus_scenarios::test_key_scenarios_search_or_fail_closed_with_known_reasons`,
and it is **flaky, not caused by this branch**: run in isolation it failed 1 of 6 times at the
base and 2 of 6 at head. The large standing population is environmental — NumPy and Torch are
absent from this venv, which accounts for the `test_neural_*`, `test_selfplay`, `test_dataset`
and `test_engine_env` families. Within the differential/certification area this branch actually
touches, the failing set is **identical at base and head**.

### Sweeps

Prediction registered before measuring in `reports/c142_rump_branch_prediction.md`
(commit `11bca7eb`). 200 games per window, `--matcher strict`, roll enumeration off (the
shipping path). Baseline from a `git worktree` at the merge base `ce962c6e`, same venv and same engine
`.so` — the build check reported a content-fingerprint match on both sides, so the two runs
differ only by the patch. **No run at or above seed 19,200,000.**

**Take counts differ per artifact pair, and the earlier "unchanged across all three takes"
overstated the baseline.** Per-path history: the **rumpfix** pair has three takes
(`6d055c42`, `c893c7df`, `ac05ba61`); the **baseline** pair has two (`6d055c42`, `ac05ba61`).
So the patched numbers are corroborated across three independent runs and the baseline across
**two**, not three.

The re-takes were forced rather than chosen: merging `main` moved the engine patch set and the
build gate **refused to measure on the stale build** (`c72e6523d8de6f64` installed against
`5fa147ffa325c887` at HEAD) instead of emitting plausible numbers. Every take agrees, which is
the useful part — the result does not depend on the engine revision.

Engine fingerprint `5fa147ffa325c887` on all four runs, `enumerate_rolls: false` on all four;
the provenance records differ in `source_commit`/`source_tree`, as they must.

**Provenance defect, disclosed rather than left to be derived.** The two `c142_rumpfix_*`
artifacts stamp `source_commit 4c0ded451db822d207cbf5d985652a3154c1d85c` with
`source_tree: dirty`, and **that commit does not resolve** — it was a local commit on the
pre-merge rebase that was undone (§9), so `git log -1` on it returns `fatal: bad object`. The
engine fingerprint pins the engine, but nothing in the record pins `scripts/`, which is exactly
the hazard the comment at `scripts/engine_transition_differential.py` (Rule 3, "`source_commit`
does not pin the code") exists to name. **This is a labelling defect, not a measurement one**:
the numbers reproduce on a clean tree — the reviewer re-ran the dev window at `5a6eae37` and got
the same counters with `verdict_partition_failures` empty — and the baseline pair carries a
resolvable commit. The correct reading is that these two artifacts' *numbers* are corroborated
and their *provenance stamp* is not usable for attribution. The
baseline also reproduces `main`'s shipped numbers exactly — `boundaries_measured` 15,503 on
dev and 15,579 on the validation holdout, identical to
`reports/artifacts/c138_collapsefix_merged_{dev,holdout}_sweep.json` — so the "before" side is
not a bad baseline.

Artifacts: `c142_base_dev_sweep.json`, `c142_rumpfix_dev_sweep.json`,
`c142_base_holdout_sweep.json`, `c142_rumpfix_holdout_sweep.json`.

**Dev `19,000,000–199`, 200 games**

| counter | baseline `ce962c6e` | + this patch |
|---|---|---|
| `boundaries_full_round` | 15968 | 15968 |
| `boundaries_measured` | 15503 | 15503 |
| `transition:matched` | 15502 | 15502 |
| `transition:diverged` | 1 | 1 |
| `strict:lossy_render` | 0 | 0 |
| `skip:strict_all_branches_lossy` | 0 | 0 |
| **`skip:rump_branch_set`** | — | **0** |
| **`strict:diverged_on_full_branch_set`** | — | **1** |

The **only** counter delta across the entire report is
`strict:diverged_on_full_branch_set: 0 → 1`, which is the new counter appearing. The single
divergent row is unchanged in identity and class: `19000191/63`,
`component_magnitude:heal`.

**Validation holdout `19,100,000–199`, 200 games**

| counter | baseline `ce962c6e` | + this patch |
|---|---|---|
| `boundaries_full_round` | 16155 | 16155 |
| `boundaries_measured` | 15579 | 15579 |
| `transition:matched` | 15579 | 15579 |
| `transition:diverged` | 0 | 0 |
| `strict:lossy_render` | 3 | 3 |
| `skip:strict_all_branches_lossy` | 0 | 0 |
| **`skip:rump_branch_set`** | — | **0** |
| **`strict:lossy_render_marker:attract_empty_tail_ambiguous`** | — | **3** |

Again the only delta is the new counter. **All three** of this window's branch drops are
`attract_empty_tail_ambiguous` — the same marker family behind the final-holdout row. That is
new information the old undifferentiated counter could not give, and it says where renderer
work would pay: this marker is the whole measured population outside dev.

**Both falsifier clauses cleared.** `transition:diverged` neither fell nor rose on either
window; `skip:rump_branch_set` is 0 on both; and
`transition:diverged == strict:diverged_on_full_branch_set` holds on both, as does
`matched + diverged + engine_error + skip:strict_all_branches_lossy + skip:rump_branch_set
== boundaries_measured` (15,503 and 15,579).

**Nothing opened and nothing closed** — which is exactly what was predicted, and which means
the sweeps are a safety result, not a confirmation. See §8.

## 8. What this does not establish

- **The sweeps cannot confirm the change.** Both permitted windows are silent on the new exit
  (see the table), so the evidence that the change does the right thing is the retained-state
  replay and the unit pins, not the sweeps. The sweeps establish the narrower claim that the
  change is behaviour-preserving on both development windows.
- **The size of the rump population on the final holdout is unknown**, and the measurement
  that would settle it — re-sweeping `19,200,060–259` with `skip:rump_branch_set` installed —
  is the one measurement that must not be taken. It will be answerable on the *next* reserved
  window, at zero extra cost, because the counter now exists.
- **Whether the 14-versus-3 gap is a trend or variance is unresolved.** The measurement that
  would settle it is a fresh, never-swept 200-game window below the reservation floor, swept
  with `strict:lossy_render_marker:*` installed.

  If `attract_empty_tail_ambiguous` dominates there too — as it does in all 3 of the
  validation holdout's drops — **the durable fix is renderer-side, not matcher-side**: retain
  the Attract-versus-paralysis attribution so the arm is never dropped, and the withheld
  boundary becomes an ordinary judged one. `rust/pokezero-search/src/events.rs:2576-2650` has
  already done the analysis and reaches the same place: the `paralyzed`-dominant family (which
  is the measured slug here, `paralyzed+cannot_act`) is *downgradeable* — "both outcomes are
  'no move used, no reveal, no PP', Attract dominates 4:1" — while the `noop` and `miss` slugs
  are **"not downgradeable at any price"**, because they erase a `|move|` reveal and the miss
  arm suppresses a PP decrement the fold tracks. That asymmetry is exactly why the fix must be
  a renderer change scoped to the downgradeable family and **not** an entry in
  `_TELEMETRY_ONLY_LOSSY_MARKERS`, which the withheld-verdict exit deliberately does not
  touch. Withholding the verdict is the correct interim: it stops the harness asserting
  something it cannot know, without paying for the asymmetry with a wrong allowlist.
- **No claim is made that the engine's recoil is correct in general.** What is measured is
  gen3 Double-Edge: the `[1,3]` fraction, the `floor`, the min-1 clamp and the cap. Volt
  Tackle (1 set) and the `[1,4]` moves (0 sets) were not exercised against Showdown here.

## 9. The ledger residue, now closed

`reports/c138_known_gaps_ledger.md` is the source of the false "never fired" claim about
`skip:strict_all_branches_lossy` (§5), which surfaced independently in three PRs — what a
shared wrong premise looks like. #1163 corrected the **H14 cell** and did not reach the
**summary list** two paragraphs below it, so the merged file contradicted itself: H14 read
"REACHED, three times over" while the list still published the counter under "Never-fired
static counters (10)", annotated "(H14)".

Fixed here, and only there. Matched on the list's text rather than a line number, because three
PRs are editing that file in sequence and every line number quoted during this review has since
moved.

**The count was re-derived, not decremented.** Each of the other nine names was searched for a
nonzero value across all 260 committed JSONs under `reports/`; all nine remain absent, so
10 − 1 = **9** is measured rather than asserted. `engine_error` in particular is still genuinely
never-fired, which is what H14's own closing sentence says.

The `docs/engine_divergence_ledger_20260728.md` Rule 2 conflict resolved in favour of #1163's
mechanized C144 correction, with a C142 addition recording that the identity is now five-term
and that **the mechanized form, not the prose, is the thing to extend** when the next
post-measurement exit is added.

One process note, disclosed rather than left in the reflog: `main` was **merged** into this
branch, not rebased onto. An earlier local rebase was performed and then undone before any
push — the published tip `c893c7df` is still an ancestor of this branch's head, so no
force-push was needed and no published history was rewritten.

### Owed at merge time: `_EXPECTED_SWEEP_ARTIFACTS`

`tests/test_boundary_verdict_partition.py` pins the sweep-artifact corpus size **exactly**, so
the value is a function of what is on `main` when this branch merges. It is **74** here, which
is the honest measurement against `main` at `ce962c6e` plus this branch's four artifacts.

It will be stale at merge time. #1165 lands four more artifacts ahead of this branch, and #1159
one more after that. **The number must be re-derived by running the selector over the merged
tree — not by adding 4 to 74, and not from any figure quoted in review** — and then re-confirmed
live by checking that one lower and one higher both fail. Both checks were run this round (74
correct; 73 and 75 both fail) and both must be re-run after each merge. The pin firing on a
merge is the mechanism working, not a problem to route around.
