# C147 — the G33b residual-bucket gate, built in `ResidualPlan` and swept

**G33b is now shipped rather than recommended.** The Leftovers heal slot is no longer booked when
the residual block was truncated by the opposing active's battle-ending faint. The gate is a real
change in `ResidualPlan::build`, the prediction was registered before any measurement of it
existed, both permitted windows are swept, and the pin that holds it is verified red on a
separately-vendored `origin/main` checkout.

Registered prediction: `reports/c147_g33b_residual_bucket_gate_prediction.md`, committed at
`e0b8ca0f`, ahead of every gate measurement in this report. Outcomes are appended to it in a §8
that leaves §1–§7 untouched.

**Every prediction held.** Nothing opened on either window, the built gate closes the replayed row
exactly as the model did, and the gate's reach in-window is non-zero — which is the one result
that could have made the sweeps vacuous.

## 0. What was and was not measured

| | |
|---|---|
| the defect | ledger **G33b** (`reports/c138_known_gaps_ledger.md`), diagnosed in `reports/c143_heal_attribution_diagnosis.md` §1–2 and §6, merged as a finding in PR #1161 (`1a929c57`) |
| what was missing | the gate was measured as a **model applied to the renderer's output**, on one row of a spent final holdout, and no window had been swept. C143 §6's own closing paragraph is the specification this branch implements |
| base build | `5fa147ffa325c8872d47fbe3645125af9dac2c94f3b84ca6dc8be5f96539d341`, 71 patches, fresh worktree of `origin/main` at `e0a23e4e`, `scripts/build_search_crate_engine.sh` with `exit=0` captured directly, `engine_build_fingerprint.py --check` green |
| gate build | `b8ff14456823adbe7b72f8e3dd7b65191a8ce84801fa8622b1286a71f40d7e86`, 71 patches, same script, `exit=0`, `--check` green |
| windows swept | dev `19,000,000–19,000,199` and validation holdout `19,100,000–19,100,199`, both on both builds |
| the final holdout | **not swept, at any point.** `19200244/115` is read out of the committed `reports/artifacts/c141_final_holdout_sweep.json` and replayed against the local engine, which is the same read-and-replay c143 §0 used on a row that has already been spent |
| build/source consistency | one hazard hit and corrected mid-run: adding the seven pins to `events.rs` changed the *source* fingerprint (`a0796bdd…` → `b8ff1445…`) while the installed wheel was still the earlier one, and `--check` said `STALE`. The wheel was rebuilt and every gate measurement in this report was re-run on `b8ff1445…`. The fingerprint's inputs include every `.rs` under `rust/pokezero-search/src` (`scripts/engine_build_fingerprint.py:93,168`), so a test-only edit moves it — which is the check doing its job, not a false alarm |

## 1. The mechanism, and the one thing no HP predicate can see

Read from source, not inferred. Gen 3 inherits gen 4's residual ordering:
`data/mods/gen4/items.ts:231` puts `leftovers` at `onResidualOrder 10 / subOrder 4`,
`data/mods/gen4/moves.ts:711-716` puts the `leechseed` condition at `10 / subOrder 5`, and
`data/mods/gen4/conditions.ts` puts `brn` at `10 / subOrder 6` — **one speed-sorted bucket per
Pokémon**, not two global phases. `sim/battle.ts:565-566` ends each entry with
`this.faintMessages(); if (this.ended) return;`.

The vendored engine already models this exactly. `add_end_of_turn_instructions` carries
`stop_residuals_if_battle_ended!` at every entry boundary
(`third_party/poke-engine-gen3-battle-end-residuals.patch`, with the order-8 granularity
correction in `third_party/poke-engine-gen3-weather-entry-truncation.patch`), and its order-10
block is the speed-major class — one Pokémon at a time, fastest first, each running its whole 10.x
set in subOrder before the other side runs any of its own. **The engine's HP arithmetic was never
wrong here.** Only the renderer's attribution was.

`ResidualPlan::build` books a heal slot for every side whose active holds Leftovers, and the
`NOTE:` at the booking site refuses an `hp < maxhp` guard for a measured reason: the plan is built
on the pre-residual state and Leftovers fires at 10.4, so a mon at full HP when the plan is built
is routinely below max when the tick fires, and that guard alone cost 5 rows on this dev window.

**G33b is the case that note cannot reach.** The slot is not skipped because a predicate evaluated
false; it is skipped because the engine never reached the handler. On `19200244/115` the winner
ends at 260/268 — comfortably below max — so no HP predicate of any kind can see it. The booked
slot goes unfilled, the count mismatch sets `plan.usable[winner] = false`, every heal on that side
drops to `residual_heal_cause`, and that function is a constant function of state which since C131
change 3 tests Leftovers first. So the bare Leech Seed drain mirror comes back tagged
`[from] item: Leftovers` on a line Showdown renders `[silent]`.

## 2. The gate

`rust/pokezero-search/src/events.rs`, `leftovers_slot_truncated`, one consumer:

```rust
if active.item == Items::LEFTOVERS && !leftovers_truncated[i] {
    plan.heal[i].push("item: Leftovers".to_string());
}
```

The predicate walks the residual segment for the instruction that ends the battle — the active
reaching 0 HP with no living reserve, reserves being unable to enter during the residual phase —
and then asks exactly one question: **was the winner's order-10.4 item slot behind that point?**
The arms are enumerated from the engine's own section order, and only two gate:

The third column is the fact; the fourth is what the shipped predicate does about it.
**They differ on exactly one row**, and that row is the weather under-reach recorded two
paragraphs down and again in §7. There is **one** speed test in the code and it guards both
gating rows, so the fourth column — not the third — is what ships:

| what delivered the battle-ending faint | section | is the winner's 10.4 behind it? | gated by the shipped predicate |
|---|---|---|---|
| the shared weather entry | order 8 | **yes, always** — order 8 precedes every order-10 handler on both sides | **only when the loser is faster**; the winner-faster half is not gated |
| the loser's own 10.5 sap / 10.6 status / 10.9 partial trap | order 10, loser's bucket | yes **iff the loser is faster** | yes, on exactly that condition |
| the winner's 10.5 Liquid Ooze recoil | order 10, winner's bucket | no, it already fired | no |
| Future Sight | order 11 | no, it already fired | no |
| Perish Song | order 12 | no, it already fired | no |

An earlier revision of this table, and of the matching one in `leftovers_slot_truncated`'s doc
comment, wrote the weather row's fourth column as an unconditional "yes" while the prose below it
and §7 both said otherwise. Independent review caught it. Nothing about the code or the
measurements changed — but this document's whole subject is claims being read off tables, so the
table is now the thing that is right rather than the paragraph that corrects it.

**Why the two "no" arms at order ≥ 10.5 are excluded by state predicate rather than by
classifying the instruction:** a lethal residual damage always equals the victim's remaining HP
exactly, so the instruction itself carries no information about which phase produced it. An
earlier draft of this predicate looked the phase up by position in `plan.damage[loser]`; that was
withdrawn because the lookup is only as sound as a list that is, at a truncation site, *itself*
over-booked. Future Sight and Perish Song are therefore excluded on `future_sight.0 == 1` and the
`PERISH1` volatile. Liquid Ooze is separable structurally instead: it writes a **negative `Heal`**,
never a `Damage`. That separation is exhaustive over the residual block rather than assumed —
enumerating every `heal_amount:` expression in `add_end_of_turn_instructions` and in the two
functions it calls into (`ability_end_of_turn`, `item_end_of_turn`) gives four sites, and
`-ooze_damage` is the sole negative one; the other three (`health_recovered` twice, Sitrus's
`heal_amount`) are each a `cmp::min` of non-negative terms. Searched with
`grep -n 'heal_amount:' third_party/poke-engine-src/src/gen3/{generate_instructions,abilities,items}.rs`
plus a function-body slice of the residual block, so the claim is scoped to the gen3 residual path
and to nothing wider.

Two deliberate under-reaches, both retaining the pre-gate booking:

* **A speed tie is not gated.** `residual_speed_order` returns `None` on an exact tie because
  `speedSort` shuffles it (`sim/battle.ts:455-457`) and the engine forks both orders, keeping both
  when they differ (`same_residual_outcome`). One live order fires the winner's tick and the other
  does not, so there is no single answer, and the gate declines to guess rather than fitting one.
* **A fatal weather chip with the winner faster is not gated.** Order 8 precedes all of order 10
  unconditionally, so this is a real instance of the same family. It is left unshipped because it
  was not measured, and the speed condition is what keeps it out. Filed in §7.

## 3. The closure, built rather than modelled

`19200244/115`, replayed from the committed C141 artifact by the committed
`scripts/c143_heal_attribution_probe.py --enumerated`, whose `shipped_renderer` block goes through
`cert_sweep_reread.reread_row` and the **installed** crate. Both builds, same row, same script:

| | base `5fa147ff…` | gate `b8ff1445…` |
|---|---|---|
| `shipped_renderer.verdict` | `diverged` | **`matched`** |
| `shipped_renderer.branches` | 416 | 416 |
| `shipped_renderer.misses` | 12 | **0** |
| `shipped_renderer.label_only_miss` | `pct=0.22: p1 attributed components differ: observed_only=[('heal', 36)] engine_only=[('itemleftovers', 36)]` | *(empty)* |
| `modelled_g33b_gate.verdict` | `matched` | `matched` |
| `modelled_g33b_gate.soundness.heals_relabelled` | 350 | **0** |
| `arms_reproducing_the_full_observed_trace` | 1 arm, 0.2189 % | 1 arm, 0.2189 % |

Artifacts `reports/artifacts/c147_g33b_row_replay_base.json` and
`…_gate.json`.

Two things worth reading off that table rather than the verdict alone:

* **The base column reproduces c143 §6 exactly** — 416 / diverged / 12, the label-only miss at
  0.2189 %, 350 relabels with every delta in [27, 47] and none equal to 16. So the replay is
  measuring the same thing c143 measured, on a build two merges later.
* **`heals_relabelled` falls 350 → 0**, which is the sharp result. The model rewrites p1 heals
  tagged `[from] item: Leftovers` in truncated arms; on the gate build it finds **nothing left to
  rewrite**. The built gate therefore covers exactly the set the model covered — not a subset,
  which is the outcome the prediction flagged as a finding-in-its-own-right if it had occurred.

## 4. The pin, red on a separately-vendored `origin/main`

Seven crate pins ship. Verified by **checking out and running**, in a fifth worktree
(`origin/main` at `e0a23e4e`) with its own venv and its own `vendor_poke_engine_src.sh` run — not
by reasoning from a fingerprint, and not by reverting a line in the branch's own tree. The three
end-to-end pins were applied verbatim to that pristine tree; the four predicate-level pins cannot
be, because they call a function that does not exist on `main`.

| pin | `origin/main` | branch |
|---|---|---|
| `a_truncated_leftovers_slot_is_not_booked_so_the_drain_stays_silent` | **FAILED** | ok |
| `a_faster_seeder_keeps_its_leftovers_tag` | ok | ok |
| `a_spare_pokemon_behind_the_victim_keeps_the_leftovers_tag` | ok | ok |
| `a_speed_tie_is_not_gated` | *(cannot compile there)* | ok |
| `a_future_sight_kill_is_not_gated` | *(cannot compile there)* | ok |
| `a_perish_song_kill_is_not_gated` | *(cannot compile there)* | ok |
| `a_liquid_ooze_kill_is_not_gated` | *(cannot compile there)* | ok |

Run as the FULL crate suite, not a name filter: on the `origin/main` tree the lib target reports
`135 passed; 1 failed`, and the one failure is the pin. (An earlier filtered `--lib events::tests::`
run showed a second failure, `the_refusing_seam_separates_ambiguous_from_none_matched`; that is an
artifact of the filter — the test needs the pyo3 interpreter another test in the same binary
initialises, and it passes in every unfiltered run on both trees. Recorded rather than dropped,
because "two tests failed on main" would have been a false red.)

The failure prints the defect itself rather than a generic mismatch:

```
thread 'events::tests::a_truncated_leftovers_slot_is_not_booked_so_the_drain_stays_silent'
panicked at src/events.rs: assertion `left == right` failed:
the drain mirror must render bare; a [from] tag here is the G33b mislabel
  left: ["item: Leftovers"]
 right: []
```

**Six of the seven are green on `main` too, and that is the point of having them.** Each is a case
the gate must *not* reach, so a gate that fired unconditionally — the obvious wrong version of
this change — would break all six. Only the first is revert-failing; presenting all seven as
regression pins would overstate them.

### 4a. And the six are not inert either — five mutants, five single kills

"Green on both trees" is also what an assertion that tests nothing looks like, and this program has
found five of those in one day. So each of the five arms the six pins guard was deleted or weakened
in turn, one at a time, with the source restored and byte-compared after every run
(`git status` clean afterwards, and `--check` back at `b8ff1445…`):

| mutant | pins that went red |
|---|---|
| drop the negative-`Heal` arm (Liquid Ooze) | `a_liquid_ooze_kill_is_not_gated` |
| drop the Future Sight exclusion | `a_future_sight_kill_is_not_gated` |
| drop the Perish Song exclusion | `a_perish_song_kill_is_not_gated` |
| gate the speed tie as well | `a_speed_tie_is_not_gated` |
| drop the living-reserve condition | `a_spare_pokemon_behind_the_victim_keeps_the_leftovers_tag` |

Each mutant kills **exactly one** pin and no other. That is the useful shape: it says each pin is
the sole killer of its own mutant, so none of the six is riding on another's coverage.
`a_faster_seeder_keeps_its_leftovers_tag` has no mutant listed because the mutation that kills it
is "gate unconditionally", which kills four of the seven at once and therefore attributes nothing;
it is the direct converse of the revert-failing pin (c143 §1a variant C) and is kept for that
reason.

**And these pins run in CI.** The crate suite is executed by the *Crate suites, including the
renderer pins* step of `.github/workflows/engine-fidelity-gates.yml`, whose test-count floor is
raised to the exact new total — measured by summing that step's own expression over a local run,
never by adding 7 — and whose named-pin list now names all seven. **The shipped figure is
`429 → 436`**, re-measured on the merged tree; this paragraph said `423 → 430` in an earlier
revision, which was the pre-merge measurement left standing in the present tense while §5b
re-derived the merged one correctly. Two figures for one shipped number is the defect, not the
arithmetic: §5b is the derivation and 436 is the value in the YAML. A count alone would let them
be deleted and replaced by any other seven; a name alone would survive `#[ignore]`. Both are
asserted, which is that step's existing discipline.

## 5. The sweeps

`scripts/engine_transition_differential.py --games 200`, strict matcher, collapsed roll path (the
shipping configuration), on both builds and both windows. Four artifacts committed as
`reports/artifacts/c147_g33b_{base,gate}_{dev,holdout}_sweep.json`.

| | dev `19,000,000–19,000,199` | | validation holdout `19,100,000–19,100,199` | |
|---|---|---|---|---|
| | **base** `5fa147ff…` | **gate** `b8ff1445…` | **base** `5fa147ff…` | **gate** `b8ff1445…` |
| `boundaries_full_round` | 15968 | 15968 | 16155 | 16155 |
| `boundaries_measured` | 15503 | 15503 | 15579 | 15579 |
| `transitions_matched` | 15502 | 15502 | 15579 | 15579 |
| `transitions_diverged` | 1 | 1 | 0 | 0 |
| `engine_errors` | 0 | 0 | 0 | 0 |
| `divergence_classes` | `{component_magnitude:heal: 1}` | same | `{}` | `{}` |
| `gating:exact` | 14156 | 14156 | 14148 | 14148 |
| `gating:support` | 1347 | 1347 | 1431 | 1431 |
| retained repro | `19000191/63` | `19000191/63` | — | — |

**The gate's whole `counters` block is byte-identical to the same-build base sweep's on both
windows** — compared as dicts, not eyeballed as a table, and that is stronger than the nine scalars
above: 23 keys on dev and 21 on holdout, including every `skip:*`, `strict:*`, `limit:*`,
`world_prestate_mismatch*` and `hidden_counter_support:*`. Nothing opened, nothing closed, nothing
moved. **P1, P2, P3 and P4 all hold**, and P4 — zero rows closed — was registered as the expected
outcome, not discovered.

Gate provenance on both windows:
`engine_fingerprint b8ff1445…`, `enumerate_rolls false`, `source_commit e0b8ca0f…`
(**the prediction commit** — the sweeps ran at the commit that registered the prediction, which is
the ordering evidence rather than a claim about it), `source_tree clean`,
`records_with_provenance 200`, `complete true`, one distinct provenance string.

The dev window's `exit=1` is the harness reporting its one divergence, not a failure; the holdout's
is `exit=0`. Both captured directly.

### 5a. The re-sweep of the base, and the one thing it caught

The prediction's registered baseline was the two committed C142 base artifacts, on the argument
that `main` at `e0a23e4e` reproduces their engine fingerprint exactly. **That argument is sound and
it is also incomplete, and only the control re-sweep shows how.** Every verdict scalar in §2 of the
prediction is reproduced exactly by the fresh base sweep. But two `counters` keys differ from
C142's artifacts:

| window | key | C142 artifact | fresh base sweep |
|---|---|---|---|
| dev | `strict:diverged_on_full_branch_set` | absent | 1 |
| holdout | `strict:lossy_render_marker:attract_empty_tail_ambiguous` | absent | 3 |

Neither is a behaviour change. Both keys were **added to the harness** after C142's sweeps were
taken — `git log ce962c6e..e0a23e4e -- scripts/engine_transition_differential.py` lists six
commits, and the keys are written at `engine_transition_differential.py:2279` and `:2115`. The
holdout's is a refinement rather than an addition to the total: `strict:lossy_render` is 3 in both,
and the marker sub-key labels the same three.

**The lesson is about the fingerprint's scope, and it is worth writing down**: the engine build
fingerprint covers the patch stack and `rust/pokezero-search/src/**`, and it does **not** cover
`scripts/`. So "same fingerprint" licenses a claim about the engine and the renderer and says
nothing about the measuring instrument. Had the gate sweeps been compared against the C142
artifacts instead of against a same-build, same-harness base, those two keys would have shown up as
a counter-block difference and P3 would have read as falsified by a change this branch did not
make. The control was run because C133 §7 asks for it; it earned its keep on a detail the argument
for skipping it could not have reached.

### 5b. `main` moved mid-branch, so every merge-sensitive figure was re-derived

`origin/main` advanced to **`f1c3b3aa`** (#1166, the attract immobilizer marker) after §5's pair was
taken. That merge adds an engine patch — the stack goes **71 → 72** — and changes the renderer, so
the pre-merge pair no longer measures the tree that ships. It was merged in (`git merge`, never a
rebase) and **all four sweeps were re-run on both sides of the merge**, rather than the pre-merge
numbers being carried across:

| | dev merged base | dev **merged gate** | holdout merged base | holdout **merged gate** |
|---|---|---|---|---|
| `boundaries_measured` | 15503 | 15503 | 15579 | 15579 |
| `transitions_matched` | 15502 | 15502 | 15579 | 15579 |
| `transitions_diverged` | 1 | 1 | 0 | 0 |
| `engine_errors` | 0 | 0 | 0 | 0 |
| `divergence_classes` | `{component_magnitude:heal: 1}` | same | `{}` | `{}` |

Merged base `770228825d53f717…`, merged gate `b3b0fde0b3fda523…`, both 72 patches, both `--check`
green. **The merged gate's `counters` block is again byte-identical to the merged base's** — 23 keys
on dev, 19 on holdout — and the row replay reproduces on the merged builds too: `diverged` / 12
misses → **`matched` / 0**, relabels **350 → 0**. Provenance `source_commit f68a3546…`, the merge
commit, `source_tree clean`.

**The merge was not a no-op, which is why re-running was not ceremony.** On the holdout window
#1166 moved three counters that this branch does not touch:

| key | pre-merge base | merged base |
|---|---|---|
| `strict:lossy_render` | 3 | absent |
| `strict:lossy_render_marker:attract_empty_tail_ambiguous` | 3 | absent |
| `strict:sleeptalk_union_branch` | 105 | 106 |

That is #1166's own effect — its marker de-collapses the three arms that used to refuse as
`attract_empty_tail_ambiguous`. On dev it moved nothing. Had the pre-merge gate sweep been compared
against the merged base, those three would have read as this branch opening something, and the
falsifier's clause 6 would have fired on a change made in another PR. **Both pairs are retained**:
the pre-merge pair is what the registered prediction was made against, the merged pair is what
certifies the head that ships.

**`main` then moved a second time, to `99c77eb7` (#1168), and that merge took the number `c146`.**
This report and its prediction are therefore **C147**, renumbered after the fact; the `c146_g33b_*`
paths quoted in the two commits before the merge are the same files under their old names. #1168
also adds a **second** exact corpus pin, `_EXPECTED_COUNTER_ARTIFACTS`, over every committed JSON
under `reports/` and `docs/` — bumped **347 → 360** here by importing that module into a worktree of
`99c77eb7` and calling `counter_artifacts()` there, set difference exactly the thirteen
`c147_g33b_*.json` this branch adds, nothing removed, and confirmed live at 359 and 361. **The two
corpora move independently**: only eight of the thirteen are sweep-corpus members, because that
selector requires a top-level `boundaries_measured` and this one does not, so neither count can be
used to check the other.

The crate-test floor was likewise re-measured on the merged tree rather than carried: **429 → 436**,
summed from that CI step's own expression over a local run (436 `... ok` lines, 0 failures, 1
ignored). `429 + 7` agrees, but the figure comes from the run. And `_EXPECTED_SWEEP_ARTIFACTS` was
re-derived twice — 79 → 83 before the merge, 79 → **87** after it, both times by running
`_sweep_reports` itself over both trees, and confirmed live at 82/84 and again at 86/88.

**No engine patch is added by this branch.** `git diff origin/main...HEAD -- third_party/
scripts/apply_poke_engine_patches.py` is empty, so `PATCHED_TARGET_TREE_SHA256`,
`EXPECTED_FINAL_SHA256`, the `--test test_gen3` count of 32 and the `Engine lib suite` count of 5 are
all untouched and unchanged. The only CI count this branch moves is the crate floor.

## 6. Reach — the one measurement that could have made §5 vacuous

**A verdict-level sweep cannot see this gate fire**, and the reason is structural rather than
incidental: at every site where the gate fires, the baseline plan was *already* unusable, so any
mutation that preserves the booking count reproduces the baseline's output exactly. The only
detectable change at a firing site is the one that makes the count reconcile — the gate itself.
So "the sweeps are unchanged" is consistent both with a safe gate and with a gate that never
executes, and §5 alone cannot tell them apart.

So reach was measured directly, with a **throwaway instrumented build** whose only difference from
the gate build is two `eprintln!` lines — one inside the gating arm of the predicate, one at the
booking site — swept over both windows and counted out of the sweep's own stderr:

| window | predicate returned true | **Leftovers slots actually skipped** |
|---|---|---|
| dev `19,000,000–19,000,199` | 54 | **52** |
| validation holdout `19,100,000–19,100,199` | 58 | **56** |

Artifact `reports/artifacts/c147_g33b_gate_reach.json`, which carries the instrumentation verbatim.
The two counts differ by 2 on each window because the predicate is evaluated for both sides and
reports truncation whether or not the truncated side holds Leftovers; the second column is the
gate's effective reach.

**So the gate fires 108 times across the 400 games and changes no verdict either way.** That is the
result §5 needed in order to mean anything: the sweeps are not silent about the gate, they measure
a change that executes 108 times and opens nothing.

Two controls on that number, because a firing count is easy to mis-read:

* The instrumented build's **`counters` block is byte-identical to the same-build base sweep's on
  both windows**, so the instrumentation itself changes no measurement — the `eprintln!` is on
  stderr and the harness reads none of it.
* The instrumented build is **not part of the patch**, and its two sweep artifacts are deliberately
  **not** committed: its fingerprint (`dee00fce…`) is not reproducible from any committed tree, and
  committing a sweep nobody can rebuild would be worse than quoting the count. The reach artifact
  records that fingerprint and the exact two lines instead.

**The condition under which this section would have withdrawn the sweeps' evidential value — a
firing count of zero on both windows — did not occur.** It was registered in the prediction as a
live possibility precisely because it was not knowable in advance.

## 7. Left open, and stated as such

* **The order-8 weather arm with the winner faster.** A fatal weather chip skips all of order 10
  on both sides, so the winner's Leftovers slot is over-booked there regardless of speed. The
  speed condition keeps that case out of this gate. It is a real, reachable instance of the same
  family, and it is unshipped because it is unmeasured — not because it is believed absent.
* **The loser's own over-booked slot in that same weather case.** Same reason.
* **Exact speed ties.** Both residual orders are live, so one of them is mislabelled either way.
  Nothing here improves that, and gating it would be choosing a fork to fit.
* **Whether the gate is safe beyond these two windows.** Two 200-game windows and one replayed
  row are what was measured. A relabelling can only ever *widen* what matches, so "nothing
  opened" is necessary and not sufficient, which is exactly why §3 and §6 are reported beside §5
  rather than instead of it.
* **G8 is untouched.** The magnitude half of `19200244/115` and of `19000191/63` is a separate
  defect with a separate disposition ("closed by enumeration, retained under the collapsed
  path"), and the dev window's surviving divergence is that row. Nothing in this branch is a
  representative change.
