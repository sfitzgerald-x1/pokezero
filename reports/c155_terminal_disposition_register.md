# C155 — the terminal-disposition register

**This document is the single maintained statement of what stands between the program and
`RATIFIED_SWEEP_PRECONDITION`.** It replaces a list that had been reconstructed on demand three
times by three agents, from `reports/c152_ledger_terminal_disposition.md` §7.1 plus C153 and C154,
with different numbering each time. A document rebuilt on demand is exactly the artifact this
program keeps finding drifted, so this one is committed, numbered permanently, and machine-checked
against the tree by `tests/test_terminal_disposition_register.py`.

**Reconciled against** `origin/main` at `578287e7` (#1207), merged into this branch — merged, never
rebased. The branch was cut at `6ef682bf` (#1204); `main` advanced twice while this document was in
review and every figure was re-derived after each merge. One of those advances moved the engine
fingerprint again, which is T1's subject rather than an inconvenience. Every figure below is re-derived from
tracked bytes or from a committed artifact — none is carried from the list it replaces, and **six**
statements did not survive that derivation, itemised in §5.

> **What this document does NOT do.** It runs no sweep, builds no engine and consumes no seed. It
> does not touch `OWNER_RATIFIED`, `RATIFIED_SWEEP_PRECONDITION`, `RATIFIED_FINAL_HOLDOUT`,
> `BURNED_FINAL_HOLDOUT`, the `19,200,000`–`19,200,259` burn, C141's demotion, or
> `reports/c151_final_holdout_rereg_prediction.md`. It writes no new JSON artifact, deliberately:
> every figure here is already derivable from a committed one, and C154 §6 records what a
> placement decision costs when the artifact was not needed.

---

## 0. The rule that makes this a register rather than a list

1. **Item ids are permanent.** `T1`–`T14` are allocated once and never reused or renumbered. An
   item that is discharged stays in place with its status changed, and the pin's inventory is
   ordered and exact in both directions, so an item cannot be dropped, added or reordered without
   an author touching the module beside it. Renumbering is what made the three reconstructions
   incomparable; it is now mechanically impossible.
2. **Every item states its evidence, its scope, and who can act on it.** The actor distinction has
   been wrong here before — the program spent time believing C116 item 13 was blocked on the owner
   when the owner had already done everything asked. It had: `OWNER_RATIFIED` carries
   `19,300,000-19,300,199`, `scott, 2026-08-08`, and the prediction was registered in C151. **The
   remaining work under both gates is agent work plus one owner declaration**, and no item below is
   waiting on a ratification that has not happened.
3. **Every figure is derived, and the pin says which.** Each item carries a `pin` verdict:
   `derived` means the pin re-derives the item's status from the tree; `derived + reading` means
   the pin re-derives the figures but the *disposition* is a human reading; `reading` means the pin
   covers nothing but the item's presence. Pretending a pin covers a reading is the failure mode
   C154 §5 had to add a fifth entry to its own list to avoid.
4. **A negative carries the measurement that produced it**, per the ledger's §8 rule as widened by
   C152: a negative measured only inside the two permitted windows is a claim about those windows.

---

## 1. The two gates, quoted

`RATIFIED_SWEEP_PRECONDITION`, verbatim from `scripts/engine_transition_differential.py`:

> `"ledger terminal AND engine fingerprint declared frozen for the claim"`

They are two gates, not one. **No amount of ledger work satisfies the second**, and the second is
the one the tree can say most about: it has moved, and nothing in the repository can declare it
frozen. C151 §3 records that the trigger is *"a condition on program state, not a date, and it is
**not machine-checkable**"* — so this register does not decide whether the gates are met. It
supplies the complete, derived input to that decision.

**What "terminal" is taken to mean here, and it is a reading.** Not "no gaps": the ledger is a
ledger *of* gaps and G0 will be open at the moment of the claim. It is read as *no entry leaves an
obligation undischarged* — no verdict withheld, no measurement owed, no claim whose scope the
document cannot state. Items below are classified against that reading and the reading is labelled
as one.

---

## 2. The register

Status vocabulary, exact: **OPEN** (an obligation the tree has not discharged) ·
**DISCHARGED-IN-SCOPE** (discharged over a stated population and not beyond it) · **STANDING** (a
permanent scope statement; nothing is owed).

Actor vocabulary, exact: **AGENT** · **OWNER** · **AGENT-THEN-OWNER**.

| id | item | gate | status | actor | pin |
|---|---|---|---|---|---|
| T1 | The engine fingerprint has moved and nothing in the tree can declare it frozen | G2 | OPEN | AGENT-THEN-OWNER | derived |
| T2 | G8's second remainder — the two `defender_active.hp`-ceiling `residual_disjoint_bands` call sites | G1 | OPEN | AGENT | derived + reading |
| T3 | G33b's speed-tie arm | G1 | OPEN | AGENT-THEN-OWNER | derived + reading |
| T4 | G33c — the truncation strands the winner's order-10 damage bookings | G1 | OPEN | AGENT-THEN-OWNER | derived |
| T5 | H1 — the single-seat population is counted and uncompared | G1 | OPEN | AGENT | derived + reading |
| T6 | §4's 26 UNREACHABLE verdicts, re-adjudicated for §4 and only §4 | G1 | DISCHARGED-IN-SCOPE | AGENT | derived + reading |
| T7 | §7 item 1 — whether G1 (Stick) produces a differential row | G1 | OPEN | AGENT | derived |
| T8 | §7 item 5 — co-occurrence counts are upper bounds, not draw rates | G1 | OPEN | AGENT | derived |
| T9 | §7 item 6 — how much of G0's population is a last-mon double faint | G1 | OPEN | AGENT | derived |
| T10 | §7 item 7 — the incidence of the missing Protect `\|-fail\|` line | G1 | OPEN | AGENT | derived |
| T11 | §7 item 8 — nothing measured on the reserved final holdout | G1 | STANDING | OWNER | derived + reading |
| T12 | §7 item 9 — every Observed column but C152's still reads the c136 pair | G1 | OPEN | AGENT | derived |
| T13 | §7 item 10 — the single-seat population has no taxonomy anywhere | G1 | OPEN | AGENT | derived |
| T14 | §7 item 11 — the two zeros are a property of two windows | G1 | STANDING | — | derived + reading |

**Fourteen items — 1 under G2, 13 under G1. Status tally: 11 OPEN, 1 DISCHARGED-IN-SCOPE,
2 STANDING.** Both sentences are re-derived from the table by the pin, so a status flip cannot
update the column and leave the summary stale — which is how four counts in the ledger went bad.
The *classification* of T11 and T14 as STANDING is a reading, argued in each item.

⚠ **And the register must state how much of the ledger these fourteen items are not about.** C152's
own §7.1 closed with *"G0 and every other §3 row C152 did not touch"* as its sixth open item; the
first revision of this register absorbed that into §1's "terminal is not the absence of gaps"
reading and dropped the number, which is exactly the kind of drop §5 exists to record. Derived: of
§3's **82** rows, **9** carry a C152, C153 or C154 marker — G8, G33b, G33c, G50, H8, H12, H15, H19,
H22 — and **73 have not been re-examined since C138**. Two of the nine (G33c, H22) are new filings
rather than re-examinations. C154's **13** corrections all landed in §4, not §3. Both the **ids** and
the count are published: the count is what an owner scopes from, and the ids are what a swap moves
when the count does not — one row leaving the set as another joins holds the count at 9, and battery
mutation B52 is that case. The marker is coarse by design (any `C152`/`C153`/`C154` mention in the
cell), so it can only over-count; the id list is what bounds it, and the number agrees with C152's
own tally of five dispositions plus the row it filed. So an owner reading "eleven OPEN items" should read it beside
"73 §3 rows last looked at in C138"; the fourteen are what the last three passes *found*, not a
survey of the ledger.

### Why this is fourteen and the reconstruction was six

The reconstruction listed six because it was assembled from a **delta** — what C152 opened, plus
C153 and C154 — and a delta is not an inventory. Derived from the ledger instead, §7 alone
contributes **eight** items that the document does not mark RESOLVED (T7–T14), of which the
reconstruction carried one. Two of those eight are standing scope statements and are labelled as
such rather than dropped, because a list of blind spots that only ever shrinks is losing them by
attrition — which is the reason §7 item 7 exists in the ledger at all.

---

## 3. The items

### T1 — the engine fingerprint has moved, and nothing can declare it frozen · G2 · OPEN · AGENT-THEN-OWNER

**Derived.** `scripts/engine_build_fingerprint.py::compute_fingerprint` over the tracked inputs at
this head stamps **`2ec5bfd1c7292ed6…`**. The newest sweeps the corpus carries were taken at
**`bfdbe1c04876edcd…`** — C152's two head windows and all twelve of C153's shards; earlier
artifacts carry earlier builds still — and **no committed JSON under `reports/` or `docs/` carries
the head value at all**. The move is
legitimate, and it has now happened **four times**. Restricting `git log 7fcd9e19..HEAD` to **all** of
the fingerprint's inputs — the 74 gen3 patches, `poke-engine-gen3-patches.txt`,
`poke-engine-base-source.json`, the 11 crate sources, and the `Cargo.toml` / `Cargo.lock` /
`build.rs` / `pyproject.toml` that `cargo_inputs` and `build_metadata_inputs` contribute — returns
exactly four commits. ⚠ **The `input` column is DERIVED, and the third row was wrong on
first write** — it carried `+403 −74`, which was neither this file's numstat nor any other
figure in the tree. `git diff --numstat <base> <head> -- rust/pokezero-search/src/events.rs`
is the derivation, and it returns the same pair against `83efbede`, `a6249971` and the
merge-base alike. ⚠ Note what moved the **#1211** row's FINGERPRINT: a `#[cfg(test)]`-only
addition. `compute_fingerprint` hashes crate SOURCES, so `t1.head_fingerprint` moved while
the shipped `.so` stayed byte-identical at `dd3658e4b52bd49e` — the stamp tracks the source
tree, not the artifact, and a reader must not infer a rebuilt binary from a moved stamp.
**#1221 is the contrasting case, and the pair is why the stamp cannot be read either way:**
23 of its 26 lines are comment, but the other 3 are a live guard in `vocab_encode`, so there
the `.so` really did change. A moved stamp implies a moved source tree and nothing more; to
learn whether the artifact moved, hash the artifact.
The column is NOT pinned by
`tests/test_terminal_disposition_register.py`; pinning it needs a stable base, which
`git show --numstat <commit> -- <path>` gives once a row's commit has landed, and that is
filed as the follow-up rather than done here because it would also newly pin the two rows
above.

| commit | input | fingerprint after |
|---|---|---|
| `21f484d4` (#1197) | `rust/pokezero-search/src/leaf.rs`, +31 lines | `9517aab98d56a9ba…` |
| `578287e7` (#1207) | `rust/pokezero-search/src/priors.rs`, +91 −4 | `236d1cac8a784898…` |
| #1211 | `rust/pokezero-search/src/events.rs`, +420 −50 | `028a4c52a4ad9fe7…` |
| #1221 | `rust/pokezero-search/src/encoder.rs`, +26 −0 | `2ec5bfd1c7292ed6…` |

⚠ **And the second landed while this register was in review**, three days after C153's build. That
is not an aside: it is T1's argument, live. C151 §3 deferred the terminal sweep precisely because
*"a sweep taken today buys an unbiased measurement of a fingerprint that is superseded within
days"*, and the fingerprint has moved twice in the window between the last committed sweep and this
document reaching approval. **The re-sweep T1 asks for is only worth taking after the freeze is
declared, not before** — otherwise it buys a third superseded build. ⚠ **A first revision of this
paragraph named only three of the five input classes**, a scope narrower than the claim it
supported, which is the rule this register enforces on everyone else. That derivation is recorded,
not pinned, because it is a statement about history rather than about this tree — but the
fingerprint VALUE is pinned, and it is what caught the second move.

**And the gate has no surface.** The window's ratification is a pinned constant, `OWNER_RATIFIED`.
The freeze has no equivalent: **zero** module-level constants in
`scripts/engine_transition_differential.py` name a freeze, and nothing anywhere in `scripts/`,
`tests/`, `src/` or `.github/` records one. Scope of that negative: a name-level AST scan of the
differential plus a literal scan of those four trees for `FROZEN`/`FREEZE` outside
`frozenset`/status contexts. So an owner asked to declare the fingerprint frozen today has nowhere
to write it down.

**Consequence for the other items, and this is the one the reconstruction did not state — with the
distinction review added.** Every measurement behind T2–T6 predates this build, but they do not all
predate it in the same way, and conflating the two inflates the re-sweep an owner would be asked
for:

| item | what its evidence rests on | needs an engine? |
|---|---|---|
| T3, T4 | C152's throwaway instrumented build `89797289f4a3b555` | yes |
| T5 | the two committed head sweeps at `bfdbe1c04876edcd` | yes |
| T2 | `scripts/c152_g8_survive_representative_census.py`, whose whole import list is `argparse` / `json` / `struct` / `pathlib` / `typing` | **no** |
| T6 | `reports/artifacts/c154_unreachable_readjudication.json`, a source-and-pool trace at `source_commit 8b3e5431` and Showdown `f76228a1`, carrying **no** `engine_fingerprint` key at all | **no** |

⚠ **A first revision of this paragraph said all of T2–T6 sat at one of the two engine
fingerprints.** T2's census is stdlib arithmetic over a synthetic `(max_damage, health)` plane and
T6's artifact builds no engine; neither artifact carries an `engine_fingerprint`. Attributing a
build to a measurement that has none makes the freeze look like a precondition for work that does
not depend on it — the opposite of what this register is for. So: the re-sweep below repairs T5's
and T12's provenance. **It does not touch T2 or T6, and neither of those needs a sweep or a build
to be discharged.**

**Who acts, and what it costs.** AGENT: rebuild the engine and re-sweep the two permitted windows
at the head fingerprint, then commit the artifacts. **Costed rather than merely named, because it is
the largest line item on this page:** `scripts/build_search_crate_engine.sh` is a full two-consumer
rebuild (vendor, `poke_engine` wheel, `pokezero_search` wheel, install, stamp), and the sweep itself
is **400 games** — the same 200 + 200 that produced `reports/artifacts/c152_head_{dev,holdout}_sweep.json`,
which record **19 minutes** of wall time between them at ~1,260 games/hour. Both windows are dev and
validation-holdout: **no reserved seed, no burn contact, and no contact with `19,300,000`+**. The
same run discharges T12. OWNER: declare the freeze. The declaration is not an agent act and there is
currently no place to put it; filing a `FROZEN_FOR_CLAIM`-shaped constant beside `OWNER_RATIFIED`
is the obvious surface and is **filed here unbuilt**, because that constant lives in the file this
task forbids touching.

### T2 — G8's second remainder · G1 · OPEN · AGENT · derived + reading

**Derived.** `residual_disjoint_bands` is called from **four** sites, all in the tracked
`third_party/poke-engine-gen3-status-aware-residual-threshold.patch`. Resolved by walking the patch
for call lines — excluding the function's own definition, which is the fifth occurrence of the name
and would put a parameter declaration in the ceiling column — and reading the **fourth positional
argument**. **Resolved, not copied**, and the outcome is worth stating: they resolve to the same
four values the reconstruction carried, unchanged at this head. That is a result of the derivation,
not a reason to have skipped it — the same four numbers were also once quoted from an instrumented
build, shifted by eleven lines, and looked exactly as plausible then.

| site | ceiling argument | reached by C149's split |
|---|---|---|
| `…status-aware-residual-threshold.patch:370` | `defender_active.hp` | no |
| `…status-aware-residual-threshold.patch:435` | `i16::MAX` | yes |
| `…status-aware-residual-threshold.patch:510` | `defender_active.hp` | no |
| `…status-aware-residual-threshold.patch:563` | `i16::MAX` | yes |

The right file to cite is the **patch**, not `generate_instructions.rs`:
`third_party/poke-engine-src/` is gitignored and absent from a clean checkout, so a
`generate_instructions.rs` line number is unresolvable by any check and was the citation that went
stale. The "reached by C149's split" column is derived rather than read off the scope comment: of
the five hunks in `third_party/poke-engine-gen3-leechseed-residual-band-split.patch`, exactly
**two** add the `if defender_leech_seeded {` gate, both carry an `i16::MAX,` context line, and
**none** carries a `defender_active.hp,` one.

**Scope of what IS measured.** C152 measured the *first* remainder only — the survive
representative, `reports/artifacts/c152_g8_survive_representative_census.json`: **16,205 of 27,655**
survive bands (**58.597 %**) price a representative off the exact integer fan and therefore
reproduce zero achievable rolls. That is an arithmetic census over a synthetic uniformly-weighted
plane, not an incidence. The consequence figure it is usually quoted beside — 0 divergent rows over
31,082 boundaries in the two permitted windows — **may not be quoted without both accept bars**:
support-gated acceptance at **8.689 % dev / 9.185 % holdout**, and the ±9 % roll window at
**167 dev / 140 holdout** (1.077 % / 0.899 %). The dominant class that window absorbs is
`roll_scaled_component`, which is exactly what an off-fan survive representative produces.

**What is owed.** The same census, run against the two `defender_active.hp`-ceiling sites. It needs
no sweep and no reserved seed: the first remainder's instrument is an arithmetic walk over
`(max_damage, health)`, and the second differs only in the ceiling it passes.

**Who acts.** AGENT, entirely.

### T3 — G33b's speed-tie arm · G1 · OPEN · AGENT-THEN-OWNER · derived + reading

**Derived** from `reports/artifacts/c152_g33b_open_arm_census.json`, over 1,400 games (the two
permitted windows plus four 250-game shards on unregistered seeds `1,000,000`–`1,000,999`):
**925** predicate calls at a battle-ending residual instruction, of which **24** are exact speed
ties and all 24 have a Leftovers winner. **20** of the 24 are `order_le_10` — the only arm a
truncation can expose, because Perish Song is order 12 and the other **4** ties are `perish` calls
where nothing in order 10 is skipped — and **3 of those 20** carry a winner-side heal before the
truncation, which an over-booked plan sends to a fallback answering `item: Leftovers`.

**The refusal is still shipped, and that is derived too.** `leftovers_slot_truncated` ends in
`match residual_speed_order(state)` with `_ => NO_TRUNCATION` at
`rust/pokezero-search/src/events.rs:5525`, resolved by a unique anchor. So the tie arm is unbuilt at
head, not merely unmeasured.

**Reading, and it is the disposition.** No tie-arm divergence has been *observed*: across the same
1,400 games the committed sweeps carry **12** divergent rows in the four wide shards (of 80,439
measured) and **0** in the two head windows (both bars: support-gated **8.689 % / 9.185 %**, roll window **167 / 140**). But "not observed" is not
"cannot happen", and the
attribution of any given divergence to a tie is **not re-derivable from a sweep artifact**: no
verdict-producing instrument records the residual speed order. It was recorded once, by C152's
throwaway `C152_TRUNC` instrumentation, which produced the census and no verdicts. So the tie arm's
divergence incidence has never been measured by anything that also decides verdicts, and the honest
form of "zero observed" is that sentence rather than a count. **Two** of the twelve are the
G33b-family mislabel class, and only one of them has ever been attributed — see T4.

**Power limit on any negative here.** C153 bounds a per-divergence class at **0.316 %** — one in
**316** — over **949** classified divergences. Ten times that many classified divergences,
**9,490**, would be needed to tighten it tenfold, so "the tie arm never diverges" is not a claim
any existing measurement licenses.

**Who acts.** AGENT to build the fix — the segment-order inference: `leftovers_slot_truncated` is
handed the segment as well as the state, and the two forks have different segments, so the order a
branch took is recoverable even though `residual_speed_order` returns `None`. OWNER to decide
whether a renderer change with a measured benefit of zero observed rows ships under C133 §7
discipline (registered prediction plus four sweeps) before the terminal measurement, or whether the
arm is retired with its scope stated the way the weather arm was.

### T4 — G33c · G1 · OPEN · AGENT-THEN-OWNER

**Derived.** The row is `1000513/121` in
`reports/artifacts/c152_wide_census_1000500_sweep.json`: divergence class `component_mismatch:heal`
against engine component `itemleftovers`, at branch-miss **pct 100.00**, with
`observed_only=[('heal', 36)] engine_only=[('itemleftovers', 36)]`.

**The fix is unbuilt, and that is derived exactly rather than argued.** The identifier
`leftovers_truncated` occurs **twice** in `rust/pokezero-search/src/events.rs`: the binding from
`leftovers_slot_truncated`, and **one** consumer, the Leftovers heal push at `:5466`. **Zero**
damage bookings consult it. So the flag the gate already computes suppresses the winner's 10.4 heal
and nothing else, `plan.damage[winner]` still books the status tick that the same truncation
skipped, `plan.usable[winner]` goes false anyway, and the heal drops to the fallback regardless —
which is why C147's gate is inert wherever the winner carries a residual status.

**⚠ And there is a second row of the same class that nothing has diagnosed.** Derived, not carried:
the four wide shards contain **2** rows of `component_mismatch:heal` against `itemleftovers`, and
the one the ledger's G33c cell and C152 both cite is `1000513/121`. The other is
**`1000321/102`**, in `reports/artifacts/c152_wide_census_1000250_sweep.json`, at pct 93.75 / 6.25
across two branches with `observed_only=[('heal', 26)] engine_only=[('itemleftovers', 26)]`. It may
well be the same mechanism — it has the same shape — but *may well be* is not an attribution, and
G33c's "measured benefit: one row" is a claim about the row that was diagnosed. Recorded here rather
than absorbed into the count, because absorbing it would be the sample-fitting this program keeps
finding. Attributing it is agent work and needs no sweep: the state is committed in the artifact.

**Scope.** One diagnosed row plus one undiagnosed sibling, on unregistered seeds, on C152's
instrumented build. Zero in both permitted windows (both bars: support-gated **8.689 % / 9.185 %**, roll window **167 / 140**), which is the point: the
shape was invisible to every sweep the program has run inside them. Unlike T3 the fix has a
**measured** benefit.

**Who acts.** AGENT to attribute `1000321/102` and to build the suppression; OWNER for the same
ship/retire decision as T3.

### T5 — H1, the single-seat population is counted and uncompared · G1 · OPEN · AGENT · derived + reading

**Derived** from the two committed head sweeps: **1,742** single-seat boundaries on dev and
**1,813** on holdout, **9.836 %** and **10.090 %** of all boundaries. They are counted in
`skip:single_seat_boundary`, they are disjoint from `boundaries_full_round`, and the verdict
partition closes exactly without them — so they are *visible* and *uncompared*, which is a
different thing from H12's refuted claim that a population was invisible to every counter.

**Scope.** Two 200-game windows at `bfdbe1c04876edcd`. See T1: this is not a head-build figure.

**Who acts.** AGENT. Comparing the single-seat arm needs an instrument, not a seed; T13 is its
prerequisite, since a comparison with no taxonomy produces one undifferentiated number.

### T6 — §4's 26 UNREACHABLE verdicts · G1 · DISCHARGED-IN-SCOPE · AGENT · derived + reading

**Discharged, and discharged for §4 and only §4.** C154 put all 26 through C153's tracing rule.
Derived from `reports/artifacts/c154_unreachable_readjudication.json`: **26** verdicts
UNREACHABLE-traced plus **1** withdrawn before the pass (R26), **13** reasons corrected (**4**
false, **9** incomplete) and **13** sound. **Nothing closed, nothing opened** — §3's inventory is
still 82 rows and §4's is still the same 27 candidates. This is the item where manufacturing
openness would be as wrong as manufacturing terminality: the verdicts are re-adjudicated, and
saying they are still "open" would be false.

**Four residues, each named because the discharge does not cover it:**

1. **The pool half is not re-derived in CI.** `scripts/c154_unreachable_readjudication.py` requires
   a pokemon-showdown checkout, and **zero** steps in `.github/workflows/engine-fidelity-gates.yml`
   check one out — the single mention of `pokemon-showdown` in that file is a comment. The `pool`
   block is therefore a committed measurement at Showdown
   `f76228a1354b5d0f307ca2d16101294ad3a2308b`, and a `sets.json` bump that added `taunt` to a set
   would leave the module green and the ledger wrong. Bounded and nameable.
2. **Five judgements are human readings**, enumerated in `reports/c154_unreachable_readjudication.md`
   §5 and derived here as a count of that numbered list: R1's one-keyword-argument closure, R9's
   mechanic enumeration, R22's correctness judgement, R7's one-module absence, and the
   `NARROW_FORECLOSURE` classification itself. No pin carries any of them.
3. **Three rows are foreclosed only over §4's own population** — **R1, R23, R24**, classified
   `RANDBATS_POPULATION` in the artifact. So "26 UNREACHABLE" must be read against §4's stated
   scope at `reports/c138_known_gaps_ledger.md:589`, *"cannot be reached in gen3 randbats"*,
   resolved here by a unique anchor. R23's counter fires today on this repo's own scenario corpus.
4. **The retraction guard is defeatable by deliberate obfuscation.** It folds whitespace, case,
   markdown emphasis and zero-width characters, each added after something got past the previous
   three, and the in-cell re-assertion is caught by a separate occurrence inventory rather than by a
   stricter phrase match. Its failure direction is safe for accidents and open to an author who
   wants past it. **That last sentence is a reading and no pin covers it** — the four normalisations
   are derived; "no fifth obfuscation exists" is not a claim any check makes.

**Who acts.** AGENT for residue 1 (a CI job that builds the checkout, or an accepted standing
scope). Residues 2–4 are dispositions, not measurements, and the register's position is that they
are correctly recorded as residues rather than as open rows.

### T7 — §7 item 1, whether G1 (Stick) produces a differential row · G1 · OPEN · AGENT

The pool reachability is settled and deterministic; what is not settled is whether the differential
ever hands the engine a Farfetch'd holding Stick, since items are not public until revealed and
Stick has no reveal event. The ledger's own settling measurement is a 200-game dev sweep asserting
on `PokemonSpec.item`. Derived: §7 does not mark this item RESOLVED, and no committed artifact
records the measurement. AGENT, inside the dev window, no reserved seed.

### T8 — §7 item 5, co-occurrence counts are upper bounds · G1 · OPEN · AGENT

Every *same-side* pairing verdict relied on for an UNREACHABLE came back zero, which is decisive.
The non-zero ones — `sleeptalk`+`rest` (44 sets), `wish`+`protect` (24), `leechseed`+`substitute`
(6) — are upper bounds and not measured draw rates, and G14 is the row where such an upper bound was
read as a reachability fact and was wrong. Derived: unresolved in §7. AGENT.

### T9 — §7 item 6, G0's last-mon fraction · G1 · OPEN · AGENT

Double faints are reachable and demonstrated; the fraction where both sides are down to their final
Pokémon is not measured, so G0's incidence is unbounded while its per-occurrence severity is large.
Settling measurement named in the ledger: count games in a 200-game dev sweep ending in a same-ply
double faint with both parties at one remaining Pokémon. Derived: unresolved in §7. AGENT.

### T10 — §7 item 7, the missing Protect `\|-fail\|` incidence · G1 · OPEN · AGENT

The mechanism is settled and the cost on the one observed shape is measured at zero. What is not
measured is whether any shape exists where the omission costs more than a protocol line. Derived:
unresolved in §7. AGENT.

### T11 — §7 item 8, the reserved final holdout is unmeasured · G1 · STANDING · OWNER · derived + reading

**This is the gated measurement itself, not a blocker on it**, and reading it as a blocker would
make the precondition circular. Nothing has been measured at or above `19,200,000` deliberately;
`19,200,000`–`19,200,259` is burned unconditionally and `19,300,000`–`19,300,199` is ratified and
awaiting the trigger. **The owner has already acted here**: the window is ratified and the
prediction is registered. Classifying this as STANDING is a reading; the derived part is that §7
does not mark it resolved and that the ratification exists.

### T12 — §7 item 9, the Observed column is as of c136 · G1 · OPEN · AGENT

Every "observed" column in §3 except the rows C152 dispositioned still reads
`reports/artifacts/c136_faintcancels_fix_{dev,holdout}_sweep.json`, i.e. is as of `aeaee2b1`. The
head sweeps C152 committed say only that both windows are at 0 divergences (both bars: support-gated **8.689 % / 9.185 %**, roll window **167 / 140**),
which cannot re-derive a per-row "observed" for a row that never had one. **This item and T1 share an instrument**: a pair
of head-fingerprint sweeps repairs the staleness and produces the artifacts T1 needs. Derived:
unresolved in §7. AGENT.

### T13 — §7 item 10, the single-seat population has no taxonomy · G1 · OPEN · AGENT

**Derived.** No sub-keyed counter `skip:single_seat_boundary:<reason>` appears in any of the 402
committed JSON under `reports/` and `docs/`, and nothing emits one. Scope of that negative: a
literal scan of the committed JSON corpus plus `src/` and `scripts/`, which is the glob, and it does
not extend to an instrument nobody has committed. `reports/c132_single_seat_coverage_bound.md` §3 is
the only mechanistic account and is argued from two hand-driven probes with no counts. Prerequisite
for T5. AGENT.

### T14 — §7 item 11, the two zeros are a property of two windows · G1 · STANDING · — · derived + reading

Both permitted windows are at 0 divergences (both bars: support-gated **8.689 % / 9.185 %**, roll window **167 / 140**) and the engine is not
divergence-free: the same
74-patch engine produced **12** divergent rows over **80,439** boundaries on unregistered seeds
`1,000,000`–`1,000,999`, and C153 measured **949** classified divergences over 803,264 boundaries.
Nothing is owed; the item exists so the two zeros are never quoted bare. STANDING is a reading; the
derived part is the item's presence in §7 and the two counts above.

---

## 4. Standing constraints on how any of this may be quoted

These are not items. They bind every item above and every claim built on one.

**The two accept bars.** No residue count from the permitted windows may be quoted without both:
support-gated acceptance at **8.689 % dev / 9.185 % holdout** (1,347 of 15,503 and 1,431 of
15,579), and the ±9 % roll window at **167 dev / 140 holdout** matched boundaries
(**1.077 % / 0.899 %**). A bare "0 divergences over the two windows' measured boundaries" is
forbidden by the ledger's §6 item 9 and by this register. ⚠ **The first revision of this sentence
said "the one place this document quotes that zero, T2" and the check behind it keyed on the
literal `31,082`.** The document quotes a two-window zero in **five** paragraphs and four of them
carried neither bar — this register breaking its own §4 rule, behind a check advertising coverage
it did not have, which is the #1205 shape in a document written after #1205. The rule and the
check are now the same width **over the phrasings this document uses**: every paragraph asserting a
zero over the two permitted windows carries both bar values. A detector that normalises markdown
emphasis before matching — because one of the five is written `**0** in the two head windows`, which
a literal scan walks past — finds **6** paragraphs, the five item-level assertions plus this one,
and the count is pinned. It is exercised on a constructed bare-zero paragraph and on a non-zero one,
so it cannot pass by finding nothing.

⚠ **Scope of that check, per this register's own §0 rule 4, because "the same width" would otherwise
be a stronger claim than the instrument supports.** The detector is a **phrasing set**, not a
semantic one. It fires on every form present today and would miss a zero written as *"Neither
permitted window carries a divergent row"*, *"No divergent rows survive in either window"*, *"Both
windows matched on every measured boundary"*, *"Dev and holdout are clean at head"* or *"The two
permitted windows carry none"* — all checked, all currently absent, none guaranteed absent tomorrow.
So the check is exact over the document as written and is **not** a guarantee against a future
rephrasing. Widening it to a semantic test is filed unbuilt; the honest form of the claim is this
paragraph rather than the words "the same width".

**The per-divergence power limit.** C153's combined arm classified **949** divergences over 803,264
boundaries. The rule of three puts the 95 % upper bound on a per-divergence class at **0.316 %** —
**one in 316**. That is far too weak to call a class-level negative settled: a tenfold tighter bound
needs tenfold the trials, **9,490** classified divergences, i.e. of order 10⁴. Per-boundary and
per-game negatives are much better served — the same census bounds a per-boundary class at
3.73 × 10⁻⁶ — and substituting the boundary denominator for the divergence one overstates a
per-class negative by **846×**, the ratio of the two denominators.

**A derived number is checked and a typed one is not, and this document has now paid for that
twice.** Three of the four claims review blocked in round one were *typed* self-descriptions; the
fourth round of failure was a *derived* one going red in CI while green on two local trees, because
GitHub gates on `refs/pull/1206/merge` and `main` had gained a step neither tree carried. **The
derived check caught a staleness that no reader, author or reviewer could have found locally — the
defect did not exist on either of their trees.** A typed `25` would have shipped wrong and silently.
Where CI gates on the merge, **local green and CI green are different measurements**; §6 declares
that coupling and the assertion carries the fix in its own failure message.

**Every measurement behind T2–T6 predates the head build**, and none is at `2ec5bfd1c7292ed6` —
nor at `028a4c52a4ad9fe7` or `9517aab98d56a9ba`, the builds this document was reconciled against
at the two preceding merges. Read the
table in T1 before quoting that as a re-sweep scope: T3, T4 and T5 rest on an engine build; **T2 and
T6 build no engine at all** and carry no `engine_fingerprint`.

---

## 5. What the derivation changed, against the list it replaces

Recorded rather than silently corrected, because "the previous version said X" is the only way a
maintained document proves it is maintained. **Six** entries, and the count is re-derived from the
list below rather than typed.

⚠ **Read this section as the author's account, not as a checkable comparison — and that limit is
inherent, not an oversight.** The list these six deltas are measured against **exists nowhere in the
repository**: it was assembled in conversation, three times, which is the entire reason this
document exists. Nothing a reader can open holds the "before". Every *right-hand* side is pinned —
the fourteen items, the §7 inventory, `1000321/102`, the four call sites, the two classification
names, the absent freeze surface are all re-derived by the pin — but the claim that a previous
version said otherwise rests on this author's word. It is written down so the **next** revision has
a checkable "before", which no revision until now has had.

1. **Six items became fourteen.** The reconstruction was a delta of what C152, C153 and C154 opened.
   Derived from the ledger, §7 contributes eight unresolved items where the reconstruction carried
   one, and two of those eight are standing scope statements rather than obligations.
2. **G8's call-site line numbers resolved to the same four values, in a different file than the one
   they are usually cited from.** `370` / `435` / `510` / `563` are lines in the tracked **patch**;
   C152's `4197` / `4300` / `4406` / `4456` are lines in the gitignored vendored
   `generate_instructions.rs`, which no check can resolve and which does not exist in a clean
   checkout. Both sets describe the same four sites. Only one of them is citable.
3. **"No tie-arm divergence observed" is weaker than it reads.** No verdict-producing instrument
   records the residual speed order, so a tie-arm divergence could not have been attributed by a
   sweep even if one occurred. See T3.
4. **The reconstruction called the §4 scope classification `NARROW_FORECLOSURE`.** That is the
   generator's Python constant; the value recorded on each of R1, R23 and R24 in the artifact is
   `RANDBATS_POPULATION`. Both are pinned here so the two names cannot drift apart.
5. **Gate 2 has no surface at all**, which no reconstruction said. The window's ratification is a
   pinned constant; the freeze has nowhere to be written down.
6. **G33c's class has two rows in the wide census, not one.** `1000513/121` is diagnosed;
   **`1000321/102`** is not, and neither C152 nor the ledger's G33c cell mentions it. "Measured
   benefit: one row" is a statement about the row that was attributed, and the second is filed
   inside T4 rather than folded into the count.

---

## 6. What the pin covers, and what it cannot

`tests/test_terminal_disposition_register.py`. Its load-bearing property is that the item
inventory, the item statuses and **every figure in Appendix A** are re-derived from the tree on each
run — from tracked bytes, from committed artifacts, and from anchors resolved with uniqueness
required, imported from C153's census rather than copied so a stale anchor is one loud failure
rather than a wrong number.

**It cannot decide whether the gates are met.** C151 §3 says the trigger is not machine-checkable
and this register does not make it so.

**It cannot cover a reading.** Six are named as such, and each carries `derived + reading` in the
`pin` column rather than `derived`. T3's disposition — the figures are derived; "not observed, and
not measurable by a verdict instrument" is a reading. T6's residue 4 — the four normalisations are
derived; "no fifth obfuscation exists" is not. The classification of T11 and of T14 as STANDING
rather than OPEN. And ⚠ **T2 and T5, downgraded in review**: their *figures* are derived, but the
**status** half of each rests on the ledger's prose — "still unmeasured" for T2's hp-ceiling sites,
"uncompared" for T5's single-seat population — rather than on an absence scan of the kind T13
carries. Both statuses are right; `derived` overstated by one word, and the word is the whole
distinction this section exists to draw.

**Two couplings, declared.** First, Appendix A pins `base.expected_counter_artifacts` and
`base.expected_sweep_artifacts` against the two census modules' own constants, so a PR that adds an
artifact must update this register in the same change. Second, the guard-scan counts above are
properties of the **whole** workflow, so a PR that adds any unittest step to it reddens this module
until the two numbers are re-derived. Both are deliberate and they are the whole point: the document
that goes stale is the one nothing forces an author through.

⚠ **The second coupling has a failure mode local runs cannot show, and it has already fired.**
Where CI gates on the merge rather than the branch — as this repo does — **local green and CI green
are different measurements**, and a step added to `main` after the branch was cut appears only in
the merge. The register documents that class; it now has a first-party instance, and the assertion's
failure message names the cause and the fix (merge `origin/main`, do not rebase, re-derive) rather
than leaving the next author to rediscover it.

**The `Ran N tests` guard.** The step carries an exact `Ran 42 tests`, re-derived from the module's
own AST and from a local run rather than copied. Issue **#1205** records that #1204's guard scan
covered only part of the workflow, so it was **not** assumed to cover this one. Measured on this tree
rather than inherited from #1205's figure. The workflow holds **32** lines containing the unittest
invocation, of which **one is a comment**, so there are **31 executable** invocation sites at this
head; the scan resolves **31** of them and leaves **none** unresolved. **This step is among the
resolved ones** — the number is not restated here, because a fourth copy of it is a fourth thing to
go stale — and `EveryWorkflowTestCountGuardMatchesItsModuleTests` derives 42 from the module's AST
and matches the guard. All three numbers are **re-derived by the pin** rather than typed.

⚠ **#1205 IS CLOSED (C156), AND THE SENTENCE ABOVE IS WHAT ITS CLOSURE LOOKS LIKE.** This register
recorded four unresolved sites —
`tests.test_differential_denominator`, `tests.test_engine_stat_attestation`,
`tests.test_seed_registry_coverage`, `tests.test_spread_gate_provenance` — and recorded that the set
was **invariant** across four trees. It was, and the reason it was invariant is that its cause was:
the scan looked at most **twelve lines** past each invocation, and those four steps put their guard
**12, 16, 23 and 30** lines out, behind long explanatory comments. One cause, four sites, no second
mechanism among them. `reports/c156_workflow_guard_scan_closure.md` carries the per-site diagnosis
and the mutation battery. The scan now derives each step's extent from its own `run:` body, so the
lookahead cannot go stale again the next time somebody writes a paragraph above a guard.

**The counts churn; the finding does not, and they are pinned differently.** `executable` and
`resolved` move whenever any step joins this workflow. The **unresolved set** is #1205's actual
subject, so it is pinned **by module name** rather than by count or by line — and it is pinned at
**empty**, which is a stronger statement than the four-member set it replaced, not a deletion of
it. A line shift does not touch it; a new *unguarded* step reddens it, which is a new blind spot
opening and should be loud.

⚠ **#1205's own counts went stale before it was closed, and this register says which.** #1205
records *"24 executable guards, 20 covered, 4 silently missed"*, measured at `6ef682bf` where that
was exact. `origin/main` then gained #1203's Taunt step, which **is** guarded (`Ran 13 tests`,
13 methods) and **is** resolved, so main read **25 executable / 21 resolved**; #1206's merge read
**26 / 22**. The four missed steps never changed. Restating #1205's 24/20 here would have been
quietly wrong; the honest form is that its arithmetic moved with the file and its subject did not,
until C156 moved the subject too.

⚠ **The same self-match trap, twice, and the second time it shipped.** The scan matches on the
invocation string, so the step's own comment — which quoted that string while explaining the scan —
counted as a site. Round one caught that for its visible effect, the step appearing twice, fixed the
comment, then round two **re-introduced it as an arithmetic error**, counting the comment into the
denominator: "20 of 25 / 21 of 26 / five" against the true 20 of 24 / 21 of 25 / four. Round one's
figures had been right, and the correction was reported as "re-measured on this tree", which made it
worse than a typo. Caught by review; the triple became derived.

⚠ **And then the derived check failed in CI while green on two reviewers' trees — the most useful
thing that happened to this document.** GitHub gates on `refs/pull/1206/merge`, not on the branch
head. #1203 landed on `main` after this branch was cut and added one guarded invocation; this branch
added its own; **each tree reads 25 executable in isolation and the merge reads 26.** A typed `25`
would have shipped wrong and silently, because nothing checks a typed number. This is the first
defect in three rounds that **neither the author nor the reviewer could have found locally** — it
does not exist on either tree. Resolved by merging `origin/main` into the branch (merge, never
rebase) and re-deriving all four trees; the assertion now carries that diagnosis in its own failure
message.

**Mutation evidence.** **53 mutations applied, 53 caught**, plus **1 negative control** verified green, enumerated in the module's docstring and
partitioned by *what is mutated*: **A1–A37** edit this document, **B38–B53** edit **only the tree
and never this document**. That second number is the one that matters, because a pin reading a
document against a hard-coded copy of itself passes every document-side mutation — and ⚠ **a first
revision put it at ten and two of those ten edited the register**, which is the property being
claimed, mis-stated. Sixteen is the measured figure: a patch line shift, the tie-refusal arm deleted,
a damage push made to consult the truncation flag (G33c *fixed*), a sub-keyed single-seat counter
added to a committed sweep, a freeze constant added to the differential, a real Showdown checkout
added to the workflow, the head fingerprint written into an artifact, an `hp`-ceiling context line
added to a split hunk, a §7 item marked RESOLVED, a §4 verdict flipped, the §4 population sentence
reworded, and every C152 marker stripped from a ledger row.

⚠ **The previous revision's account of a surviving mutation was wrong in both halves, and review
measured both.** It said a mutation had survived — one `(C152)` marker of six rewritten on a §3 row —
diagnosed the cause as *"the derivation published only a count"*, and cited that as what earned the
id list. Neither holds. The marker predicate is `any(tag in line)`, **insensitive to how many markers
a row carries by construction**, so a list of row *ids* is exactly as blind to that edit as a count
of rows: the id list could not have been the fix. And the mutation as shipped — every marker stripped
from a **one**-marker row — moves the count too, so a count alone would have caught it. The battery
total held only because a survivor had been silently replaced by a different mutation.

**Both halves are now correct.** That edit is recorded as **negative control C1: it must stay green,
because it does not change the fact being derived** — a row with five remaining markers is still a
row C152 re-examined, and a red there would be a false positive. Making the derivation
marker-sensitive was the alternative and was rejected: it would redden whenever anyone rewords a
ledger cell and drops one of several redundant mentions, which is brittleness rather than
sensitivity. The control is recorded **with the red mutations that prove the same assertion live**
(B49 and B52), because a negative control unpaired with one is indistinguishable from an inert pin.
And the id list is justified by the mutation that actually needs it: **B52**, a swap that strips H8's
marker and adds one to G1, holding the count at **9** while the membership changes — red on the ids,
green on the count, and the only mutation in the battery a count cannot catch.

**Test evidence.** `python -m unittest tests.test_terminal_disposition_register -v` → **Ran 42
tests, OK**. The gated family together: ledger uniformity **19**, never-fired **22**, wide-seed
**36**, C154 re-adjudication **34**, seed registry **41**, single-seat coverage **3** — all OK.
`tests/test_final_holdout_guard.py` and `tests/test_boundary_verdict_partition.py` cannot import
without a built engine in this environment and are unchanged by this branch. Full suite, with the
flag that is required rather than stylistic:
`pytest tests/ -q -p no:randomly --continue-on-collection-errors` — **165 failed at base, 165 failed
at head**, 4,420 → 4,462 passed (**+42**, exactly this module), 33 errors both sides, and the
`FAILED` id lists are **identical in both directions**. Re-measured against the merge-base after
each of the two `origin/main` merges rather than carried: at `6ef682bf` the pair read 164 / 4,416,
and #1203 brought one more engine-dependent module. The absolute figure is a property of the machine; the
delta is that this branch adds zero failures.

---

## Appendix A — the pinned facts

Every row is re-derived on each run of the pin, and the key set is exact in both directions: a key
added to the derivation and not to this table is red, and so is the reverse.

| key | value |
|---|---|
| `bar.roll_window_dev` | 167 |
| `bar.roll_window_dev_fraction` | 1.077 % |
| `bar.roll_window_holdout` | 140 |
| `bar.roll_window_holdout_fraction` | 0.899 % |
| `bar.support_gated_dev` | 8.689 % |
| `bar.support_gated_holdout` | 9.185 % |
| `base.expected_counter_artifacts` | 403 |
| `base.expected_sweep_artifacts` | 115 |
| `base.patch_stack` | 74 |
| `base.section3_rows` | 82 |
| `base.section4_candidates` | 27 |
| `base.section4_drops` | 26 |
| `base.section7_items` | 11 |
| `base.section7_unresolved` | 8 |
| `cost.head_resweep_games` | 400 |
| `cost.head_resweep_minutes` | 19 |
| `power.boundaries_per_classified_divergence` | 846 |
| `power.classified_divergences` | 949 |
| `power.divergences_for_a_tenfold_tighter_bound` | 9490 |
| `power.one_in_n_divergences` | 316 |
| `power.per_divergence_upper_95` | 0.316 % |
| `rule.two_window_zero_paragraphs` | 6 |
| `scope.section3_rows_touched_count` | 9 |
| `scope.section3_rows_touched_since_c138` | G8, G33b, G33c, G50, H8, H12, H15, H19, H22 |
| `scope.section3_rows_untouched_since_c138` | 73 |
| `scope.section4_rows_corrected_by_c154` | 13 |
| `t1.committed_json_carrying_head_fingerprint` | 0 |
| `t1.freeze_declaration_constants` | 0 |
| `t1.head_fingerprint` | 44dcfca90130ed91 |
| `t1.newest_committed_sweep_fingerprint` | bfdbe1c04876edcd |
| `t2.first_remainder_off_fan_bands` | 16205 of 27655 |
| `t2.first_remainder_off_fan_fraction` | 58.597 % |
| `t2.hp_ceiling_site_lines` | 370, 510 |
| `t2.hp_ceiling_sites` | 2 |
| `t2.i16max_ceiling_site_lines` | 435, 563 |
| `t2.i16max_ceiling_sites` | 2 |
| `t2.residual_disjoint_bands_call_sites` | 4 |
| `t2.split_hunks` | 2 |
| `t2.split_hunks_touching_an_hp_ceiling_site` | 0 |
| `t3.games` | 1400 |
| `t3.order_le_10_ties_carrying_a_winner_heal` | 3 |
| `t3.predicate_calls` | 925 |
| `t3.speed_ties` | 24 |
| `t3.speed_ties_order_le_10` | 20 |
| `t3.speed_ties_perish` | 4 |
| `t3.speed_ties_with_a_leftovers_winner` | 24 |
| `t3.tie_refusal_line` | rust/pokezero-search/src/events.rs:5767 |
| `t4.boundary` | 1000513/121 |
| `t4.branch_miss_pct` | 100.00 |
| `t4.engine_component` | itemleftovers |
| `t4.heal_mismatch_rows_in_the_wide_census` | 2 |
| `t4.leftovers_truncated_consumer_line` | rust/pokezero-search/src/events.rs:5839 |
| `t4.leftovers_truncated_consumers` | 1 |
| `t4.leftovers_truncated_references` | 2 |
| `t4.observed_component` | heal |
| `t4.undiagnosed_sibling_rows` | 1000321/102 |
| `t4.wide_census_divergent_rows` | 12 of 80439 |
| `t5.dev_single_seat_boundaries` | 1742 |
| `t5.dev_single_seat_fraction` | 9.836 % |
| `t5.holdout_single_seat_boundaries` | 1813 |
| `t5.holdout_single_seat_fraction` | 10.090 % |
| `t5.subkeyed_single_seat_counters_in_corpus` | 0 |
| `t6.human_readings` | 5 |
| `t6.pool_showdown_commit` | f76228a1354b5d0f307ca2d16101294ad3a2308b |
| `t6.reasons_corrected` | 13 |
| `t6.reasons_false` | 4 |
| `t6.reasons_incomplete` | 9 |
| `t6.reasons_sound` | 13 |
| `t6.rows_foreclosed_over_section_4_population_only` | R1, R23, R24 |
| `t6.section_4_population_anchor` | reports/c138_known_gaps_ledger.md:589 |
| `t6.verdicts_unreachable` | 26 |
| `t6.verdicts_withdrawn` | 1 |
| `t6.workflow_steps_checking_out_showdown` | 0 |
