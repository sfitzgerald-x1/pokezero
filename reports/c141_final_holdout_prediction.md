# C141 — final-holdout prediction, registered before the sweep

C116 Phase 4 item 13: the single measurement on the reserved final holdout. This document
is committed **before** the sweep runs, and the sweep is run **once**. There is no second
attempt, so a prediction registered afterwards would be worthless.

## The window, and why it is not the reserved floor

The reservation begins at seed **19,200,000** and is enforced by the guard added in #1122,
which checks the whole requested span and every seed present in the records, and refuses
without `--final-holdout-i-mean-it`.

I contaminated the first 60 seeds of that window earlier in this program: a convenience
shell loop executed 60 games of it before the guard existed. That is disclosed in
`reports/rust-fidelity/final_holdout_contamination_disclosure.md` (external) and the JSON
was deleted unread. So the measurement window is **`19,200,060`–`19,200,259`**, disjoint
from the contaminated 60, chosen by me rather than deferred, and 200 games like the other
two windows.

## The build

`main` `16857e06`, engine fingerprint **`44ee1430708cbb55`, 71 patches** — verified by
`engine_build_fingerprint.py --check` immediately before the run. This is the same
fingerprint the dev and holdout measurements carry, and #1155 was documentation-only, so
no engine input has moved since.

## State on the two open windows, on this build

| window | diverged | rows |
|---|---|---|
| dev `19,000,000–199` | **1** | `19000191/63` |
| holdout `19,100,000–199` | **0** | — |

## Prediction

**A small non-zero count, most likely 0–3, dominated by shapes already in the ledger.**

Reasoning, stated so it can be checked against the outcome rather than reinterpreted after
it:

- Dev and holdout are 1 and 0 on this build, so the per-window rate is now well under one
  row per 200 games. A third window drawn from the same distribution should look similar.
- The final holdout has never been swept, so **every latent shape gets its first
  opportunity here.** Six of the seven rows this program closed were closed after being
  seen; a window that has never been seen has had no such attrition. That argues for
  *more* than zero, not less.
- The known-gaps ledger (`c138`) lists reachable-but-unobserved shapes — Farfetch'd's
  Stick, the double-faint tie, PP above 10, the Toxic min-1 clamp — any of which could
  surface here for the first time.

**I am not predicting zero.** A zero would be a pleasant result and I would report it, but
predicting it would be predicting the flattering outcome on a window selected precisely
because nothing has been fitted to it.

## What this measurement can and cannot say

It is one unbiased sample of the real distribution on a window nothing was tuned against.
That is its whole value, and it is why it is spent once.

It **cannot** say the engine matches Showdown everywhere, and three standing caveats from
`c138` apply to this number exactly as they do to the others:

- **~9% of measured boundaries are accepted through a widened bar** — an underivable sleep
  counter is neither skipped nor guessed but enumerated over up to 64 worlds, accepted if
  any one matches (Constraint 7). Monotone toward matching.
- **The simultaneous last-mon double faint is a terminal-value divergence** that no
  differential counter can see; it corrupts search value rather than fidelity reporting.
- **~13% of boundaries are single-seat** and skipped before comparison, so the coverage
  denominator is ~87%, not 100% (`c132`).

## Falsifier

**If the count is large — say above 5 — or if any row is a shape the ledger does not
already contain, the program's claim that the residue is understood is wrong**, and that
is the outcome worth learning. A new shape here is more informative than a clean sweep,
because it would mean the two development windows had been exhausted rather than the
mechanism understood.

The run will be reported exactly as it comes out, including the seed and step of every
divergence, whatever the number.

---

# OUTCOME — appended after the single run, then corrected three times by review

Run once, on `19,200,060`–`19,200,259`, 200 games, build `44ee1430708cbb55` / 71 patches.
Artifact: `reports/artifacts/c141_final_holdout_sweep.json`. The run executed **at the
prediction commit** `3687d205` with `source_tree: clean`, so the pre-registration is
verifiable rather than asserted.

**Headline: 1 genuine divergence, 5 withheld boundaries, 16,268 matched, zero engine errors —
of 16,274 measured.** The five withheld are 1 rump-branch adjudication and 4 all-branches-lossy.
An earlier version of this line said "1 divergence and 1 withheld", which undercounts: the
four all-branches-lossy boundaries are withheld in exactly the same sense, and a reader summing
the old headline gets 16,272 adjudicated.

The four verdict counters, which partition the measured set:

| | dev | holdout | **final holdout** |
|---|---|---|---|
| boundaries_measured | 15,503 | 15,579 | **16,274** |
| `transition:matched` | 15,502 | 15,579 | **16,268** |
| `transition:diverged` (as reported) | 1 | 0 | **2** |
| `engine_errors` | 0 | 0 | **0** |
| `skip:strict_all_branches_lossy` | 0 | 0 | **4** |

`16,268 + 2 + 0 + 4 = 16,274` closes exactly — the four-term H14 form, not the two-term one
(§ below). Adjudication then splits the 2 reported divergences and counts what is withheld;
these rows re-partition the two columns above rather than adding to them:

| | dev | holdout | **final holdout** |
|---|---|---|---|
| **genuine divergences** | 1 | 0 | **1** |
| withheld: rump-branch artifact (from `transition:diverged`) | 0 | 0 | **1** |
| withheld: all-branches-lossy (from `skip:strict_all_branches_lossy`) | 0 | 0 | **4** |
| **total withheld** | 0 | 0 | **5** |

1 genuine + 1 rump = the 2 reported diverged; 1 rump + 4 lossy = the 5 withheld.

## Two defects in the pre-registration, recorded rather than edited

Everything above the `---` is frozen: **lines 1–80 are byte-identical to `3687d205`**, which is
what makes the pre-registration auditable at all. Two of its statements are defective, and the
fix is to say so here rather than to edit them, because editing them would destroy the only
property that makes a pre-registration worth anything.

- **`:15` cites a file that does not exist.**
  `reports/rust-fidelity/final_holdout_contamination_disclosure.md` is present in **no branch
  and as no object anywhere in history** (`git rev-list --all --objects` returns nothing for
  that name; `reports/rust-fidelity/` is not a directory in this tree). It is marked
  "(external)", so the existence of a disclosure is itself disclosed and not concealed — but it
  is the **sole** justification for narrowing the window from 260 seeds to 200, it sits inside
  the frozen pre-registration, and it is **unauditable from this repo**. A reader cannot check
  the contamination claim, only that I said it. That is a real weakness in the pre-registration
  and nothing in this report repairs it.
- **`:24` "#1155 was documentation-only" is loose.** #1155 (`16857e06`) also added
  `tests/test_roll_enumeration_scope.py` (8 lines). The **conclusion still holds**, and for a
  checkable reason rather than by assertion: `build_inputs()` in
  `scripts/engine_build_fingerprint.py` hashes the patch stack, `rust/pokezero-search/src/**`
  and the Cargo inputs, so nothing under `tests/` is a fingerprint input. The precise claim is
  "#1155 moved no engine fingerprint input", not "#1155 was documentation-only" — and the
  fingerprint was re-verified at `44ee1430708cbb55` / 71 patches on this tree.

## The falsifier did NOT fire, and my first outcome note said it did

I wrote that both rows were shapes the ledger does not contain. **Both halves were false.**

Both classes were already in the committed corpus. Re-derived here over the whole corpus —
glob `reports/**/*.json` (recursive, 254 files), inspecting **every** `divergence_classes`
block at any depth rather than only the top level:

| class | `reports/artifacts/` | `reports/*.json` | **total** | **predating this sweep** |
|---|---|---|---|---|
| `roll_scaled_component` | 13 | 14 | **27** | **26** |
| `component_mismatch:*` | 13 | 11 | **24** | **23** |

(The "predating" column subtracts only `c141_final_holdout_sweep.json`, which carries both.)

An earlier version of this line said "nine committed artifacts"; a later commit claimed to fix
it to **12** and never edited the line. Both were wrong, and in the same way: they globbed only
`reports/artifacts/*.json`, which is where the **12 prior** artifacts live, and missed the
older reports sitting directly in `reports/*.json` — **14** of them for `roll_scaled_component`
(`c6`–`c13`, plus `c26` and `c27`) and **11** for `component_mismatch:*` (`c6`–`c13`; `c26` and
`c27` carry only the roll-scaled class). `c138` H15 already lists both classes among the six
ever emitted.

Two qualifications that the "5× on dev" phrasing hid, and which matter more than the count:

- The **sole** dev-window artifact carrying `roll_scaled_component` is
  `reports/artifacts/c133_withdrawn_switchcancel_dev_sweep.json` (at 5) — a **withdrawn**
  experiment. It is not evidence about main.
- On **this** build, `44ee1430708cbb55`, the class has fired **0× on dev and 0× on holdout**
  (`c138_collapsefix_merged_{dev,holdout}_sweep.json`, and the `c134_collapsed` pair).
  So "already seen in the corpus" is true, and "recently seen on this build" is false; the
  honest statement is the first, not the second.

**`19200244/115` is ledger entry G8**, still open — the collapsed lethal Leech Seed drain,
`hp_after_move + leftovers < maxhp/8`. Verified: 11 + 25 = 36 < 50.875. It is the same
mechanism as dev row `19000191/63`, diagnosed in `c140` one commit before this sweep ran,
off by one in the same way (36 vs 37; 28 vs 29). The class string differs only because the
engine's arm labelled its component `itemleftovers` instead of `heal`. The committed replay
shows **all nine** enumerated arms missing — observed `heal +36` against `itemleftovers +37` on
the 49.03 % arm — which is why this row stands as a genuine divergence and the other does not.

**`19200131/129` is not a divergence at all.** Replaying the retained state recovers two
arms: the non-crit arm at **93.75 %** mass, carrying `recoil −19`, was **dropped** as
`attract_empty_tail_ambiguous:paralyzed+cannot_act`; the crit arm at 6.25 %, carrying
`−32`, survived; the observation is `−18`. The observation **falls inside the dropped
arm's accepted recoil roll band** — `recoil` is roll-scaled, and `−18` against `−19` is
5.3 % — which is what makes the boundary matchable. It is not an identity, and an earlier
version of this line said it was.

**None of that is derivable from the sweep artifact alone**, which is a defect in the
evidence and not only in the prose. `c141_final_holdout_sweep.json` records
`branch_count: 2` but only the *surviving* miss (`pct=6.25 … engine=[('recoil', -32)]`); the
93.75 % arm, its `recoil −19`, and the marker slug appear nowhere in it. So the replay is now
committed as **`reports/artifacts/c141_final_holdout_replay.json`**, and the headline
reduction from 2 to 1 is re-derivable offline from committed artifacts alone. It is a pure
re-execution of retained state — `branch_events` on the recorded `engine_state` with the
recorded joint action, on fingerprint `44ee1430708cbb55` — so it is not a re-measurement and
did not touch the window. The same file also replays `19200244/115`, where **all nine** arms
miss, which is why that row stands as genuine.

Allowlisting that marker turns the boundary into **matched**, and that is stated here as a
**counterfactual demonstration, not a recommendation.** The marker is emitted by
`mark_attribution_unsafe` (`rust/pokezero-search/src/events.rs:2529-2531`), the strong
refusal family — not the telemetry gap allowlisted two lines below — and the comment at
`:2475-2503` argues on the record against downgrading it: the "a third of that mass is the
case that erases a `|move|` reveal" sentence is at `:2501`. Actioning it as an allowlist change justified by
*this* boundary would be fitting the harness to the final holdout, which is what the
reservation exists to prevent. That is also why the right verdict is **withheld** rather
than matched.

The verdict rested on one-sixteenth of the enumerated mass.

So the corrected reading of this window is **1 genuine divergence + 5 withheld boundaries**
(1 rump-branch, 4 all-branches-lossy). An earlier version of this line said "1 divergence + 1
withheld boundary" — the same undercount the headline above already withdraws, left standing
here one section later.

## The identity finding was already ledgered — as H14, and already refuted on disk

`skip:strict_all_branches_lossy = 4`, so the two-term `matched + diverged == measured` fails
here. This is **not** a discovery, and — correcting this report a third time — it is also
**not the first firing of the counter**. The correct form is four-term:

> `boundaries_measured == matched + diverged + engine_error + skip:strict_all_branches_lossy`

`c138` H14 states that form. But H14 also says the counter "**has never fired**", and **that
is false**, so leaning on H14 rather than on the artifacts is exactly what produced the
overclaim this section previously carried. Verified against artifacts:

| prior instance | measured | matched | diverged | lossy | two-term |
|---|---|---|---|---|---|
| `reports/c26_structural_probe_report.json` | 4,738 | 4,672 | 64 | **2** | 4,736 ≠ 4,738 |
| `reports/c27_structural_probe_report.json` | 4,738 | 4,676 | 60 | **2** | 4,736 ≠ 4,738 |

Both have sat in this repo, committed, with the counter nonzero. And the closed-PR-#1037 pair
recorded in `docs/engine_divergence_ledger_20260728.md:6720-6726` carries it at **1** on the
baseline — described in that same section as "live on main today" — and **40** on the patch.
This report cited that very section two paragraphs earlier for the 40-row instance and then
called this window the first live firing, which cannot both be true.

So what is actually true, and all that is claimed here: **this is the first firing in the
`reports/artifacts` sweep series** — of the 56 sweep artifacts in that directory carrying a
`counters` block, **55 carry `skip:strict_all_branches_lossy` at 0 and only this one is
nonzero**. It is **not** the first firing in the repo, and it is not a discovery about the
instrument.

The ledger's own standing **rule 2** (`:6767-6770`) states the identity in a **three**-term
form, omitting `engine_error`; that is corrected in #1163, not here.

Nor is it true that "no report ever asserted the two-term form beyond its own data", as an
earlier version of this section claimed. #1163 audited it and found the two-term form asserted
across **twelve files** in `reports/`, `docs/` and `scripts/`, including **two registered
prediction clauses** (`c139`'s clauses 3 and 5) — i.e. asserted as a property of the
instrument, not merely of one run. What *is* true, and is the more useful finding, is that
**no test asserted the partition in either form**, so nothing was failing vacuously: there was
nothing to fail. #1163 mechanizes it; this report only records the firing.

## Coverage, corrected

My first note said "~3.3 % of full-round boundaries are single-seat" and called it unchanged
from the pre-registration. Wrong by 3×, and it was a silent revision. Single-seat
boundaries are counted **before** `boundaries_full_round`, so they are not in that
denominator at all. The reconciliation from `c132`:

- total 16,837 + 1,767 = **18,604**; single-seat **9.5 %**; coverage **16,274 / 18,604 =
  87.5 %** — which *vindicates* the pre-registered ~87 %.
- within-full-round exits total 563: 205 struggle, 169 substitute-health-unknown, 141
  volatile-unsupported, 24 prestate-mismatch, 18 self-request-state, 4 materialization,
  2 encore-move-unknown. `16,837 − 563 = 16,274`. My first list omitted the 169 and the 24
  and therefore did not reconcile. Only the 169 is the omission `c132` warns about by name:
  `c132:55` names `limit:world_substitute_health_unknown` as "a genuine exit; omit it and the
  reconciliation misses by exactly its count". The other `c132` warning, at `:53`, is the
  *opposite* trap — `world_prestate_mismatch`'s four sub-counters sum to the parent, so adding
  parent and children **double-counts**. Confirmed in this artifact: 7 + 7 + 4 + 6 = 24, the
  parent. Dropping the 24 was my own error, not one `c132` had flagged.
- `gating_support_based` 1,284 / 16,274 = **7.9 %** accepted through the enumerated
  sleep-counter widening (Constraint 7). That figure was right.

## What this measurement establishes

**One genuine divergence in 16,274 measured boundaries on an unbiased window nothing was
tuned against**, and it is an instance of a gap already diagnosed and already open in the
ledger. Zero engine errors.

My first outcome note concluded "the residue was not understood, it was exhausted on two
windows." That was over-correction in the pessimistic direction and it was wrong on the
facts. The residue looks **more** understood than that: the heal/Leech-Seed family
predicted the one real row, and the other row was never a divergence.

**The pre-registration was correct as registered, on count and on shape.** It predicted
"a small non-zero count, most likely 0–3, dominated by shapes already in the ledger": the
count is 1, and the one row is G8, already in the ledger. Saying only "the falsifier did
not fire" is a statement about the falsifier, not about the prediction, and withholding
the latter is the same reflex as the over-correction, pointed the other way.

The matching limit belongs beside it: **G8 is diagnosed but OPEN**, and `c140` §6a
establishes it is not closable by a representative without a trade. "Understood" here
means diagnosed and bounded, not fixed.

What the window surfaced is **instrumentation** — but stated precisely, because two earlier
versions of this paragraph overclaimed its novelty in both halves:

- The first firing of `skip:strict_all_branches_lossy` **on this sweep harness**, in the
  `reports/artifacts` series. Prior instances exist on committed artifacts (`c26`, `c27`) and
  in the #1037 pair; see the H14 section above.
- A rump-branch adjudication. This is **not** the first one in a reported result either:
  `docs/engine_divergence_ledger_20260728.md:6752-6754` already reports row `19000093/51`,
  evaluated as divergent on a rump set of 11 branches with ~10 % of its mass surviving. What
  is new here is only that this one was adjudicated as **withheld** rather than counted as a
  divergence — a verdict distinction #1162 makes first-class — not that the shape was unseen.

Neither needed this window to be discovered. Both are now measured on it.

## What it cannot say

- ~7.9 % of measured boundaries accepted through the sleep-counter widening (Constraint 7).
- Coverage is 87.5 %, not 100 % — 1,767 single-seat boundaries and 563 in-path exits are
  outside the measured set.
- The double-faint terminal-value tie is invisible to every counter here.
- The true rump-boundary count is **1–10**, and it is unrecoverable for a structural
  reason rather than merely a forbidden one. `strict:lossy_render = 14` counts branch
  drops, not boundaries; the 4 all-branches-lossy boundaries each consumed at least one
  drop, so at most 10 remain. And the skip path at
  `scripts/engine_transition_differential.py:2393-2395` does `continue` **before** the repro
  append at `:2400` (guarded at `:2399`; `:2398` is the `divergence_class` counter, which the
  skip also misses), so **no state was retained for any of the 4** — confirmed in the
  artifact, where `repros_retained: 2` and both repros are `transition_diverged`. Their
  markers were never recorded. The forward fix is filed on #1162: when `skip_rump` becomes
  a first-class verdict, retain repros for withheld boundaries too, or the next firing of this
  counter is equally unrecoverable in the same way this one is.

## The window is spent

`19,200,060`–`19,200,259` must never be measured again; `19,200,000`–`19,200,059` was
contaminated earlier and is also spent. Above `19,200,260` is the only clean reserve, and
#1122's guard still refuses it without an explicit opt-in.

**Neither row may be fixed against this window.** Both were diagnosed on generated
boundaries and retained-state replays, which is not a re-measurement.
