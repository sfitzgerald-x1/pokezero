# C153 — one wide-seed census, and the instrument §8's newest rule was missing

**Status: measurement complete, no engine or renderer change built.** Read §1 first: this
document is not fidelity evidence and its divergence counts must never be quoted as the
program's.

---

## 1. Why this exists, and what it is not

#1200 (C152) added a standing rule to `reports/c138_known_gaps_ledger.md` §8:

> ⚠ **A negative measured only inside the two permitted windows is a claim about those
> windows.** … Widening the *corpus* cannot find this class of error; only widening the
> **measurement** can.

The rule is sound and it landed **with no instrument behind it**, which its reviewer said
in as many words. `tests/test_never_fired_counter_census.py` re-derives every absence over
every committed artifact on every run — and the rule's own sentence says a corpus scan
cannot catch this class of error. Nothing re-measured the affected negatives anywhere new,
so §3.5's inventory remained asserted at a scope it had never been measured at.

**The exposure is measured, not supposed.** Re-derived here rather than carried:
`counter_artifacts()` selects **388** committed JSON under `reports/` and `docs/`.
**103** of them carry a top-level `seeds.min`/`seeds.max` span, and they partition as:

| span | files | what it is |
|---|---|---|
| `19,000,000`–`19,000,199` | 39 | the dev window |
| `19,100,000`–`19,100,199` | 40 | the validation holdout |
| `19,100,170` single seed | 4 | C145's single-row replays, inside the holdout |
| `19,200,060`–`19,200,259` | 1 | the burned final-holdout block |
| `1,000,000`–`1,000,999` | 4 | C152's own wide census |
| `1,350,000` / `1,500,000` / `17,000,000` bands | 15 | pre-fingerprint c6–c13 and c26–c27, obsolete engines |

**83 of 103 are the two permitted windows.** Excluding C152's four, every sweep on an
engine anywhere near the current one is inside two 200-game windows this program has
iterated against for its entire history.

**What this is not.** Not a programme, not a fidelity measurement, not a coverage claim.
One bounded census on unregistered seeds, whose only job is to give every window-scoped
negative a verdict at a stated new scope.

---

## 2. The inventory, re-derived — and it is 46, not 45

Derived from source by AST in `scripts/c153_wide_negative_census.py`, never transcribed:
40 `EngineWorldUnsupported` reasons from `src/pokezero/engine_world.py`, 19
`classify_divergence` return classes and 8 `UnmappableChoice` reasons from
`scripts/engine_transition_differential.py`, plus the whole `counts[...]` **key space** so
an inventory entry the harness cannot emit is a loud failure rather than a row that
measures nothing.

§3.5's four verified-negative lists come to **50**: 8 static counters, 6 dynamic-family
prefixes, 7 `unmappable_choice` reasons, 29 `world_unsupported` reasons.

**Four** of the 50 carry an argument independent of measurement:

* `mapper_lossy` and `no_usable_branch` — structural, demonstrated in §6;
* `nature_not_neutral` — R7, natures unset on 24,000 of 24,000 generated Pokémon;
* `weather_unsupported` — R8, all four gen3 weathers are in `_WEATHER_IDS`.

So **46**, not 45, rest purely on window-scoped measurement. ⚠ **The brief this work
started from said ≈45, and the difference is a real one rather than rounding.** The fifth
name usually listed as measurement-independent, **`future_sight_pending` (R1)**, is **not
a member of the 50**: §3.5 removes it from the list of 33 *before* the four C146
corrections that take 33 to 29, and retires it under R1. Subtracting it from 50 subtracts
a name that was never in the set. It is carried through this census anyway, with an
`UNREACHABLE_POOL` verdict, because an inventory that silently drops a name is how a
"closed" row turns out to be a fourth category in disguise.

The census also carries the **row-level** negatives the §3.5 lists do not contain: H15's
other **ten** never-fired `divergence_class` values (the 12 minus `mapper_lossy` and
`no_usable_branch`, which §3.5 already counts). Total inventory: **61 entries**.

---

## 3. The scope, and what the sample size can and cannot rule out

**Extended rather than reinvented.** C152 used `1,000,000`–`1,000,999`. This census takes
the contiguous next block, so the whole unregistered census band reads as one range:

| arm | seeds | games | matcher | shards |
|---|---|---|---|---|
| strict | `1,001,000`–`1,008,999` | 8,000 | `strict` (the shipping configuration) | 8 × 1,000 |
| banded | `1,009,000`–`1,010,999` | 2,000 | `banded` | 4 × 500 |

Both are below `FIDELITY_SEED_FLOOR` (`19,000,000`), so `tests/test_seed_registry_coverage.py`
does not track them and no band registration is owed; they are in no acceptance namespace;
and they are disjoint from **every** reserved band — the dev window, the validation
holdout, the burned `19,200,000`–`19,200,259` block and the owner-ratified
`19,300,000`–`19,300,199` window. That disjointness is pinned, not asserted.

**Two arms because one of them answers a question the other structurally cannot.** H15's
own cell scopes four `divergence_class` values as "reachable only through the
`--matcher banded` path, which no committed artifact used". A strict-only census would
have left that negative exactly where it was, and would have reported it as measured.

### 3.1 Justifying the size

Zero events in *N* independent trials puts the 95 % upper bound on the per-trial rate at
**3/N** (the rule of three). Quoted at both granularities because the inventory mixes
per-game exits with per-boundary ones.

| arm | games | measured boundaries | classified divergences | 95 % bound / game | 95 % bound / boundary | 95 % bound / divergence |
|---|---|---|---|---|---|---|
| strict | 8,000 | 641,866 | 261 | 3.75 × 10⁻⁴ | 4.67 × 10⁻⁶ | 1.15 % |
| banded | 2,000 | 161,398 | 688 | 1.50 × 10⁻³ | 1.86 × 10⁻⁵ | 0.436 % |
| combined | 10,000 | 803,264 | 949 | 3.00 × 10⁻⁴ | 3.73 × 10⁻⁶ | 0.316 % |

⚠ **The third denominator is the one that matters for half the inventory, and getting it wrong
would overstate this census by four orders of magnitude.** A `divergence_class` negative is not a
claim about boundaries: `classify_divergence` runs only on a boundary that has **already
diverged**, so its trials number 949, not 803,264. The bound for "class X never fires" is
therefore **0.32 % of divergences**, not one in 267,755 boundaries. Both are recorded in the
artifact and both are re-derived by the pin.

**What it rules out.** §1.4 of the ledger names this program's blind spot in so many
words: *"a shape with a 1-in-50,000 boundary incidence is reachable and would show zero
rows"* in two 200-game windows. This census measures **803,264**, **25.8×** the two
windows' 31,082. A 1-in-50,000 per-boundary shape expects **16.1** hits here, so P(zero) ≈ 10⁻⁷.
That blind spot is closed for any counter incremented per boundary.

**What it demonstrably has the power to find.** The two negatives that fell in #1200 are
the calibration, not an analogy: `skip:rump_branch_set` fired **3 times in 1,000 games**
and `strict:branch_event_legal_error:BranchLegalRollError` **27 times in 1,000 games**.
At those rates this census expects ~24 and ~216 respectively, and **measured 14 and 146** on the
strict arm — the same order, running about 60–70 % of the C152 rate, which is what a different
seed block on a different build should look like and is recorded rather than smoothed over. So a
negative in the same incidence class as the two that fell **cannot** survive this sample by
chance, and the calibration is an observation rather than an assumption.

**What it cannot rule out**, stated so the verdicts are not over-read:

1. Anything below the per-boundary bound above. This is a bound, not a proof of absence.
2. Anything conditioned on a pool configuration these 10,000 games do not sample. 220
   species over 120,000 generated Pokémon is dense per species and sparse per
   species × set × item × move-draw combination.
3. Anything behind a non-default flag. Both arms run `--approximate-sleep` off,
   `--no-hidden-counter-support` off and `--enumerate-rolls` off. A refusal diverted by
   hidden-counter support is diverted in both arms.
4. Anything the differential's own payload builder cannot construct — enumerated per entry
   in §6, not left to be inferred from a zero.
5. Anything about the two permitted windows or the reserved ones. This census says nothing
   about them, in either direction.

---

## 4. Results

**61 entries, four verdicts, and every one carries its scope in the sentence.**

| family | entries | fired | not observed at scope | unreachable |
|---|---|---|---|---|
| §3.5 static counters | 8 | 0 | 6 | 2 structural |
| §3.5 dynamic families | 6 | 0 | 6 | — |
| §3.5 `unmappable_choice` | 7 | 0 | 7 | — |
| §3.5 `world_unsupported` | 29 | 0 | 27 | 2 pool (R7, R8) |
| R1, retired outside the 50 | 1 | 0 | — | 1 pool |
| H15, claimed banded-reachable-only | 4 | **3** | 1 | — |
| H15, claimed never produced | 6 | **4** | 2 | — |
| **total** | **61** | **7** | **49** | **5** |

### 4.1 §3.5's 46 all survive — and that is a result, not a null

Not one of the 46 fired, over 8,000 strict games and 641,866 measured boundaries. Read with §3.1
that is a real statement: the two negatives that fell in #1200 sit at 3 and 27 firings per 1,000
games, and this sample would have expected roughly 24 and 216 of each. **Whatever remains in
§3.5's four lists is at least an order of magnitude rarer than the two that fell.** The 46 keep
their status and gain a scope; they do not become theorems.

⚠ **And the census confirms two counters that were already known to fire, on the shipping build
for the first time.** `skip:world_unsupported:transform_unexpressible` **6** and
`skip:world_unsupported:status_unsupported` **4** over the 10,000 games. Both are outside the 46
(C146 had already refuted them from the archive) and both had been observed only in the c32 / c43
/ `docs/audit_artifacts` eras on engines long superseded. G50 is annotated in place.

### 4.2 H15's twelve did not survive — seven fired, and four of those were miscategorised

All seven fired on the **`--matcher banded` arm only**, over 688 classified divergences:

* `damage_band` **375** across 313 distinct seeds — first witness `1009001`
* `unclassified` **163** across 144 distinct seeds — first witness `1009048`
* `status_support` **84** across 71 distinct seeds — first witness `1009042`
* `faint_boundary` **30** across 29 distinct seeds — first witness `1009013`
* `evidence:faint_ply_no_upkeep` **30** across 30 distinct seeds — first witness `1009126`
* `evidence:crit_in_step` **3** — seeds `1009851`, `1010271`, `1010861`
* `evidence:spikes_in_step` **2** — seeds `1009458`, `1010900`

Three of the seven are from H15's own *"reachable only through `--matcher banded`, which no
committed artifact used"* list. Their firing **discharges** that scope caveat rather than
refuting the claim: the cell was right about the mechanism and honest about the scope, and twelve
committed artifacts now use that path.

⚠ **The other four are an error in the cell's categorisation, not in its count.** H15 files
`evidence:crit_in_step`, `evidence:faint_ply_no_upkeep`, `evidence:spikes_in_step` and
`unclassified` among *"strict-path classes the program has simply never produced"*. They are not
strict-path classes. `classify_divergence` marks the whole tail they live in *"Banded matcher (or
an unparsable miss): fall back to protocol evidence"*, so they belong to the same fallback family
as the three above and were unmeasured for exactly the same reason. Filing them as strict-path
made them look settled by 200-game strict sweeps that structurally cannot reach them — the
category, not the corpus, was doing the concealing.

**Five survive**, with the divergence-denominator bound: `boost_delta_support`,
`component_set_equal_but_unmatched` and `no_miss_recorded` are absent across 949 classified
divergences (95 % bound 0.32 %), and `mapper_lossy` / `no_usable_branch` remain structurally
unreachable — demonstrated in §6, not asserted.

### 4.3 One new ledger row: H22

`unclassified` at **163 of 688** is **23.7 %** of the banded arm's divergences, against
`classify_divergence`'s own opening contract: *"Name every divergence. No divergence may land in
an unnamed bucket."* The function records that an earlier revision *"left ~28 % of strict
divergences `unclassified`"* and treats that as fixed; it is fixed **on the strict path only**.
The banded path's misses come from `_transition_mismatch`, which the component regexes do not
parse, so control falls through the whole ladder. Filed as **H22**, class **H**, with its
reachability verdict recorded beside it: REACHABLE and observed, on the legacy comparator only,
zero on the shipping matcher.

Note what H22's reachability cell does *not* claim. §1.2's pool instruments do not apply: the
trigger is a **matcher selection**, not a gen3 mechanic, so "is it reachable" is answered by the
flag's existence plus a measurement of it, and the cell says so rather than borrowing a movepool
argument that would not be about anything.

---

## 5. C152's two refutations now have a build anyone can rebuild

⚠ **This was not the goal and it is the most load-bearing side effect.**
`reports/artifacts/c152_wide_census_*_sweep.json` were taken at engine fingerprint
`89797289…` — a **throwaway instrumented build**, which
`tests/test_harness_digest_provenance.py` records in its own comment as *"NOT reproducible
from any committed tree, by design"* — and at harness digest `e3459e1f…`. Until this
census, the entire evidence that `skip:rump_branch_set` and
`strict:branch_event_legal_error:BranchLegalRollError` fire **at all** lived on an engine
nobody can rebuild.

Over 8,000 games on the shipping `bfdbe1c04876edcd` at harness `e0617d12`:

| counter | strict arm | banded arm |
|---|---|---|
| `strict:branch_event_legal_error:BranchLegalRollError` | **146** | 0 |
| `skip:rump_branch_set` | **14** | 0 |

Both refutations stand, and now stand on an engine that can be rebuilt from a committed tree.
Pinned in `tests/test_wide_seed_negative_census.py`, so a future change that stops reproducing
them turns red instead of leaving §3.5's correction quietly unsupported.

⚠ **A second, smaller finding falls out of the same provenance.** These twelve shards make
`bfdbe1c04876edcd` the **seventh** multi-commit engine fingerprint in
`tests/test_harness_digest_provenance.py`'s allowlist: C152's four head/mutant sweeps put it at
`d20cf840` and these put it at `7fcd9e19`, and the **harness digest moved** between them,
`e3459e1f…` → `e0617d12…`, because #1199 changed `src/pokezero/local_showdown.py` inside the
differential's import closure. Same engine, two harnesses. That is exactly what the allowlist
exists to surface, and it means C152's wide census and this one may not be pooled into a
"9,000-game" figure: different engine *and* different harness.

---

## 6. What this census cannot settle, named

Seven entries carry a `census_cannot_reach` note in the artifact, and the pin asserts that none of
them is simultaneously reported as fired. **A zero produced by an instrument that could never have
produced a one is not the same measurement as a zero produced by an instrument that could.**

**Structural — not measurement results at all (2).**

* `mapper_lossy`. `evaluate_boundary_strict` returns the verdict `skip_lossy` carrying the trigger
  body *"every branch rendered lossy"*, and the run loop `continue`s at the
  `verdict == "skip_lossy"` branch **before** the `divergence_class:` line, which runs only under
  `verdict == "diverged"`. The classifier can never be handed the body that returns this class.
* `no_usable_branch`. Its trigger body *"mapper produced no usable branch"* is produced nowhere in
  the repository; its only occurrence is the classifier's own test of it.

**Reachable only by an input this instrument does not build (5).**

* `rest_sleep_refund_pending_precounts_legacy` and `rest_sleep_refund_pending_unsplit_legacy`.
  Both are raised only for rows written by a **pre-split producer** — the first guarded by
  `"restSleepAttempts" not in row`, the second by the pre-split flag arriving with neither
  producer flag. `engine_world.py` calls the second branch a **CANARY** whose expected count in a
  fresh post-split era is *exactly zero*. So a census zero here **confirms a design property** and
  is not evidence about coverage, and reporting it as a surviving negative would be the fourth
  category in disguise.
* `override_side_missing` — needs a caller-supplied team override missing a slot; the differential
  always supplies both packed teams.
* `public_effect_blocked` — one raise per entry of the caller-declared `blocked_slots` mapping,
  and the differential declares none.
* `deferred_opponent_action` — keyed on payload fields `deferredOpponentActions` /
  `deferredOpponentActionPriors` that the public-materialization payload never carries.

**Pool-unreachable, from §4 (3).** `future_sight_pending` (R1), `nature_not_neutral` (R7) and
`weather_unsupported` (R8). Carried through the census with an explicit `UNREACHABLE_POOL`
verdict rather than dropped, because an inventory that silently loses a name is how a "closed"
row turns out to be something else.

**And three things this census does not touch at all**, stated so completeness is not implied:

1. **G33b's speed-tie arm.** Its "no tie-arm divergence observed in 1,400 games" needs the
   `leftovers_slot_truncated` **predicate instrumentation** on a throwaway build to count tie
   calls; a verdict-level sweep cannot see the gate fire, which C147 already recorded. This census
   can say only that no tie-arm divergence appeared among the 261 strict divergences, which
   widens the "not observed" without recounting the predicate.
2. **Non-default flags.** `--approximate-sleep`, `--no-hidden-counter-support` and
   `--enumerate-rolls` are all off in both arms. A refusal diverted by hidden-counter support is
   diverted in both.
3. **The two permitted windows and every reserved band.** Nothing here is evidence about them, in
   either direction, and the pin asserts the shards are disjoint from all four.

---

## 7. Defects found in the instrument layer

**1. C152's thirteen artifacts were invisible to the workflow's path filter.** Found while
registering C153's own. `reports/artifacts/c152_*.json` are members of the exact-count pins in
both `tests/test_boundary_verdict_partition.py` and `tests/test_never_fired_counter_census.py`,
and `mass-gate` is the only job that runs either — but the job is gated on a path filter that
listed the C131 four and the C147 glob and **not** C152's. A PR whose only change was deleting one
of them would have matched no filter, skipped the job and gone green. `seed-registry` cannot
cover it either: the wide-census shards sit below `FIDELITY_SEED_FLOOR` and that module does not
see them. This is the exact failure mode the workflow file forbids twice in its own comments, in
the PR that added the standing rule about instruments. Fixed here, along with C153's entries.

**2. C152's refutations rested on an unreproducible engine.** §5. Fixed by re-measuring, not by
rewording.

**3. A self-certifying closure field, caught by this work's own mutation battery.** The first
revision of the per-seed closure pin asserted the artifact's `agrees` flag. Mutation 3 — perturb a
per-seed value and set `agrees` back to `true` — walked straight through it. Both sides are now
recomputed from the twelve committed shards and the per-entry seed map, and the recorded flag is
checked last as a transcription. Recorded rather than quietly fixed because it is the same shape
as the G8 cell this document withdrew: a number that certifies itself.

**Battery: 12 mutations applied, 12 caught** (after fixing the one above, which was 10 of 12 on
the first pass; the twelfth first-pass "survivor" was a defective mutation, `{} or {...}`, not an
inert pin). Pins verified under **both** `python -m unittest tests.test_wide_seed_negative_census`
and `python tests/test_wide_seed_negative_census.py` — 24 tests either way — after #1200 shipped
five pins that only one of those two invocations collected.

---

## 8. Reproduction

```
python3.14 -m venv .venv --system-site-packages
uv pip install --python ./.venv/bin/python maturin
bash scripts/build_search_crate_engine.sh "$(pwd)/.venv/bin/python"

# 8 strict shards
for k in 0 1 2 3 4 5 6 7; do s=$((1001000 + k*1000));
  PYTHONPATH=src python scripts/engine_transition_differential.py \
    --games 1000 --seed-start $s --keep-repro 25 \
    --checkpoint ckpt/strict_$s.jsonl \
    --json reports/artifacts/c153_wide_census_${s}_sweep.json & done; wait

# 4 banded shards
for k in 0 1 2 3; do s=$((1009000 + k*500));
  PYTHONPATH=src python scripts/engine_transition_differential.py \
    --games 500 --seed-start $s --matcher banded --keep-repro 25 \
    --checkpoint ckpt/banded_$s.jsonl \
    --json reports/artifacts/c153_banded_census_${s}_sweep.json & done; wait

PYTHONPATH=src python scripts/c153_wide_negative_census.py \
  --strict-shard reports/artifacts/c153_wide_census_*_sweep.json \
  --banded-shard reports/artifacts/c153_banded_census_*_sweep.json \
  --checkpoint-dir ckpt \
  --write reports/artifacts/c153_wide_negative_census.json
```

The per-game checkpoints are **not committed** — 10,000 records. They are only used to
attribute a firing to a seed, and that attribution is **closed against the committed shard
counters** in the artifact and in the pin: a missing or invented seed breaks the sum
rather than leaving an unfalsifiable sentence.
