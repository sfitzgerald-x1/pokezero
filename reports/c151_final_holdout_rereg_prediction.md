# C151 — the re-registered final holdout: `19,300,000`–`19,300,199`

> ## ✅ REGISTERED AND FROZEN — as to the window, the protocol and the preconditions
> ## ⛔ SWEEP NOT RUN — and its TRIGGER HAS NOT FIRED
>
> Two states at once, and the distinction is the whole point of this document.
>
> **Frozen** means the window, the prediction, the falsifiers, the nonzero-result protocol
> and the preconditions are settled and may not be re-chosen. Ratified by the owner
> **scott** on **2026-08-08**. Everything above the horizontal rule at the end of this
> document is registered *before* any number exists, which is the only thing that makes a
> pre-registration worth anything.
>
> **Not run** means exactly that: no seed in `19,300,000`–`19,300,199` has been executed,
> no artifact exists, and the band is deliberately **absent** from `REGISTERED_BANDS` in
> `tests/test_seed_registry_coverage.py`, because ratification is not a witness.
>
> **Trigger not fired** means the sweep is *deferred by condition, not by schedule* — see
> §3. Ratification bought the window. It did not buy permission to run today.

## 1. The owner's decision, verbatim

The question that opened this, asked after an adversarial audit ruled C141's terminal claim
defeated:

> **"can we sweep a new set of seeds then?"**

The ratification, in the owner's own words, carried in full because a summary of a blessing
is not a blessing:

> **"Ratified: final holdout re-registered as 19,300,000–19,300,199, one sweep ever, to run
> only after the ledger is terminal and the engine fingerprint is frozen; old block burned
> in the guard; c141 demoted to dev evidence; nonzero-result protocol pre-registered per
> plan."**

Ratified **2026-08-08**, owner **scott**. The same pair is carried in code as
`OWNER_RATIFIED = ("19,300,000-19,300,199", "scott, 2026-08-08")` in
`scripts/engine_transition_differential.py`, pinned by `tests/test_final_holdout_guard.py`,
so that a future change to the window or to the name beside it is a reviewable diff rather
than an edit to a sentence. That is deliberate: the program's rule was *"only the owner can
bless the replacement window"*, it lived in a document, and an agent walked straight past
it — C141's own pre-registration says *"chosen by me rather than deferred"*.

## 2. Why the old block could not simply be re-used, and is now burned

The registered block `19,200,000`–`19,200,199` had **no virgin seeds left**:

| segment | disposition |
|---|---|
| `19,200,000`–`19,200,059` | contaminated by a pre-guard 60-game convenience loop; JSON deleted unread |
| `19,200,060`–`19,200,199` | swept by C141, on a window the executing agent chose itself |
| `19,200,200`–`19,200,259` | consumed by that same run overrunning its registration by 60 seeds |

So a fresh window was never a *replacement span inside* the block. It is a
**re-registration of the final-holdout namespace onto a new one**.

The whole of `19,200,000`–`19,200,259` is now **BURNED in the guard**:
`_reject_burned_final_holdout` refuses it **unconditionally**, at execution and at
`--merge-from` aggregation, and **`--final-holdout-i-mean-it` does not open it**. Not the
salvageable parts — the whole span. A flag that can reopen a burned block is a flag that
will reopen it, and the refusal message says *why* (contaminated head, self-blessed C141
window, 60-seed overrun) rather than merely "reserved", because a reader told only
"reserved" goes looking for the flag that lifts it.

**The disclosure is no longer missing.** `reports/rust-fidelity/final_holdout_contamination_disclosure.md`
— the sole justification for C141's narrowing, which the ledger recorded as existing in no
tree in this repository's history — was recovered from outside the repo at exactly the path
`5a44c04e`'s commit message gives, and is committed here verbatim.

**The evidence, strongest leg first.** The load-bearing corroboration is **in git**, not on a
filesystem: `5a44c04e` (#1122) landed `2026-08-05 22:20:36`, **names this exact path** —
`agents/reports/rust-fidelity/final_holdout_contamination_disclosure.md`, marked *"outside
this repo"* — and **reproduces the disclosure's shell loop verbatim**, `for start in 19100000
19000000 19200000`. That commit is immutable, signed into the history of this repository, and
independent of the recovered file; it establishes both that the document existed and what it
said, and it does so from inside the repo. The disclosure also files the guard as *"owed
work"*, and #1122 **is** that work, 2h17m later.

Only then the local metadata, which corroborates rather than carries: sha256
`a749c698ec7ac38d6a9709627836761ad548ad11cc8b1748c1eea83f19ff650e`, and birth = mtime = ctime
= `2026-08-05 20:03:30`, i.e. unmodified since written. That ordering is deliberate —
filesystem timestamps are local metadata and forgeable in principle, so they are the
supporting leg and the commit is the standing one.

It **strengthens** the audit's finding rather than softening it. Under a heading reading
*"Disposition, which is the repository owner's call and not mine"* it lists three options —
shift the window, declare and proceed, or *"retire the range, reserve a fresh window entirely
(`19,300,000+`)"* — and closes *"I have **not** chosen. Until the owner decides, I am treating
all of `19,200,000+` as still reserved and will not touch it again."* C141 took the first
without the owner deciding. The owner has now taken the third, at the block the disclosure
itself named.

**Quoted completely, including the part that cuts the other way.** Option 1 also carries
*"This is what I would recommend."* — so C141 followed its own author's recorded
recommendation rather than inventing a window. That is included because a record which quotes
selectively is a hostage to the next reader, and it does **not** soften the finding:
recommending a disposition inside a section titled *"the repository owner's call and not
mine"*, one line above *"I have **not** chosen"*, is precisely the deferral C141 overrode.
Recommending and taking are different acts, and the disclosure says which was the author's to
make.

## 3. THE TRIGGER — ratified now, swept later, and the timing is part of the ratification

**Do not run this sweep today.** The precondition, registered here and carried in the guard
as `RATIFIED_SWEEP_PRECONDITION`:

> **The sweep runs only when the ledger is terminal AND the engine fingerprint is declared
> frozen for the claim.**

The reasoning is C141's failure mode generalised, and it was structural rather than bad
luck. C141 measured `44ee1430708cbb55` / 71 patches while patches were still landing weekly,
on a program whose job is landing patches; `main` now ships `8e912b45544034e6` / 74. The
queue-semantics fixture pack and the enumeration spike are in flight and more engine patches
are coming. **A sweep taken today buys an unbiased measurement of a fingerprint that is
superseded within days** — which is precisely the defect that voided the last one, quite
apart from the self-blessing.

C116 already places item 13 last: it is the single terminal measurement and the last thing
the program does. That ordering is now a precondition rather than an intention.

The asymmetry that makes this the right shape: **registration must predate any result by
construction, execution must not.** So the registration is urgent — it is frozen here,
today, before any number exists — and the execution is deliberately deferred. The trigger is
a condition on program state, not a date, and it is **not machine-checkable**; the guard
records it beside the window so that the operator reaching for the flag reads it, and
`tests/test_final_holdout_guard.py` pins that it is still recorded there.

## 4. THE NONZERO-RESULT PROTOCOL — registered before any number exists

This is the clause that protects "exactly once" from dying at the first divergence, and it
is registered **now**, because the reflex on a nonzero result will be to fix the row and
then "confirm the fix" — and that confirmation is a second sweep.

> **The sweep is an estimate, not a gate.** The claim it supports is *"the shipped
> fingerprint diverges at rate X on virgin seeds"*, and that claim **stands at any X**.
> There is no threshold below which the result counts and above which it does not. A large
> X is a worse number and an equally valid measurement.
>
> **Divergences found get attributed and disposed through the ordinary machinery** —
> generated fixtures, retained-state replays, and the dev and validation windows —
> **never by touching the holdout again.**

Three consequences, stated so they cannot be reinterpreted afterwards:

* **No re-sweep, for any reason.** Not to confirm a fix, not because the build moved, not
  because the first run crashed. If the run dies halfway, the seeds it reached are spent and
  the report says so.
* **No fix may be developed against this window.** A row surfaced here is diagnosed on
  generated boundaries and on replays of retained state, which are not re-measurements. This
  is the rule C141 already recorded for its own two rows, and it now precedes the result
  rather than following it.
* **The window is not a gate on shipping.** Nothing about the engine is blocked or released
  by X. Reading it as a pass/fail is what would create the pressure to re-run it.

## 5. The build this prediction is made against, verified rather than quoted

`main` `2acd40ff`, engine fingerprint **`8e912b45544034e6…`, 74 patches** — re-derived on
this worktree with `python scripts/engine_build_fingerprint.py --print`, not copied:

```
"fingerprint": "8e912b45544034e67d4f168d9947bf63e769d6337f7fe27cb5c44a5addf0ca27",
"count": 74
```

**C141 measured a build that no longer exists.** Re-derived by extracting each commit's tree
with `git archive` and running that tree's own copy of the script:

| commit | role | fingerprint | patches |
|---|---|---|---|
| `3687d205` | C141's pre-registration commit, the tree the sweep executed at | `44ee1430708cbb55` | **71** |
| `16857e06` | `main` as the C141 prediction names it | `44ee1430708cbb55` | **71** |
| `aa2f2d40` | the commit C141's sweep landed at | `44ee1430708cbb55` | **71** |
| `2acd40ff` | `main` today | `8e912b45544034e6` | **74** |

The patch-stack delta is exactly three files, all **appended**, with **no existing patch
modified** — the manifest `diff`, comments and blanks stripped, is `71a72,74` and nothing
else: the Attract immobilizer marker, the Sleep Talk double-reset guard, and the Leech Seed
residual band split that closed the last dev row.

**Stated precisely, because the loose version would be wrong.** "Three engine patches landed
after C141" is true of the patch stack and is *not* the whole reason the fingerprint moved:
the search crate's own sources are fingerprint inputs too. Over the fingerprint's input set
— which includes `PATCH_LIST` itself, omitted by `build_inputs()` and worth naming because
leaving it out gives 8 paths / 2,779 insertions instead — `git diff --stat 3687d205 2acd40ff`
shows **nine paths, 3,074 insertions, 253 deletions**: the three new patch files, the
manifest, and `rust/pokezero-search/src/{events.rs, leaf.rs, model.rs, lib.rs}` changed with
`abort_telemetry.rs` added. **The terminal claim C141 registered describes an engine that no
longer ships.**

## 6. State on the two permitted windows, on this exact build

Re-derived from the committed artifacts. Four JSONs on `main` carry `8e912b45544034e6`; two
of them are single-row replays outside the sweep corpus, so these are **the only two sweep
artifacts** on the current build:

| window | artifact | measured | matched | diverged | engine errors | `skip:strict_all_branches_lossy` |
|---|---|---|---|---|---|---|
| dev `19,000,000`–`19,000,199` | `reports/artifacts/c149_split_dev_sweep.json` | 15,503 | 15,503 | **0** | 0 | 0 |
| validation holdout `19,100,000`–`19,100,199` | `reports/artifacts/c149_split_holdout_sweep.json` | 15,579 | 15,579 | **0** | 0 | 0 |

Both 200 games, `--matcher strict`, collapsed roll path, `divergence_classes` empty. **This
is the first build on which both permitted windows read zero.** On C141's build the pair read
dev 1 / holdout 0.

| window | full-round | single-seat | denominator | coverage | support-gated |
|---|---|---|---|---|---|
| dev | 15,968 | 1,742 | 17,710 | 87.5 % | 1,347 of 15,503 = 8.7 % |
| holdout | 16,155 | 1,813 | 17,968 | 86.7 % | 1,431 of 15,579 = 9.2 % |

## 7. The window, and the evidence that it is virgin

**`19,300,000`–`19,300,199`, 200 games** — the same size as every other window in this
program.

### 7a. Virginity, with the scope written into the sentence

Three passes, all over **every ref and the reflog**, and the third over the whole object
database including unreachable objects:

1. **Shape-agnostic, structural.** `_seed_intervals` from
   `tests/test_seed_registry_coverage.py` — the extractor #1189 built precisely because a
   selector keyed on `seeds.min` under `reports/artifacts/` misses `c73` — over **every
   `*.json` blob under `reports/` or `docs/` reachable from `git rev-list --objects --all
   --reflog`**: **900** distinct blobs, of which **145** reach fidelity seed space. The union
   of every fidelity seed touched in any ref is exactly:

   ```
   19,000,000 - 19,000,199   (200 seeds)   dev
   19,100,000 - 19,100,199   (200 seeds)   validation holdout
   19,200,060 - 19,200,259   (200 seeds)   C141  (now burned, with the rest of its block)
   19,500,000 - 19,500,799   (800 seeds)   c73
   ```

   Overlap with `19,300,000`–`19,300,199`: **none.**

2. **Raw, shape-free, every reachable blob.** A pass for any seed-shaped whole number in
   `19,200,260`–`19,999,999` over **all 8,507 reachable blobs** — any path, any file type, so
   markdown, YAML, shell, Python and lockfiles are included, not just JSON. Tokenisation is
   **maximal numeric runs** — digits with optional `,`/`_` grouping, rejected if adjacent to
   another digit — of **exactly eight digits**. See the correction below for why that
   matters. **Exactly ten paths** carry such a number, and every one is accounted for:

   | path | in-range whole numbers | what they are |
   |---|---|---|
   | `docs/engine_divergence_ledger_20260728.md` | registry prose, incl. C151's own row | prose |
   | `reports/c151_final_holdout_rereg_prediction.md` | this document | prose |
   | `tests/test_seed_registry_coverage.py` | band constants and the module's own synthetic mutants | code |
   | `.github/workflows/engine-fidelity-gates.yml` | C151's guard comment | prose |
   | `reports/c141_final_holdout_prediction.md` | `19200260`, `19500000`, `19500799` | prose |
   | `scripts/engine_transition_differential.py` | the typo exemplar and `c73`'s band | prose in comments |
   | `tests/test_final_holdout_guard.py` | `19500000` | `c73`'s band, named in a pin |
   | `reports/c73_eight_hundred_game_sweep.json` | `19500000` | **`c73`'s real `run.seed_start`** — the one true seed here |
   | `reports/c15_why_magnitude_statgap_current_engine.json` | `19921875` | the fractional part of the float `0.19921875`; scan 1 returns **no** interval for this file reaching fidelity space at all |
   | `uv.lock` | `19229139`, `19380991`, `19414622`, `19649463` | digit runs inside sha256 hashes and package URLs (`…d19229139cb…`, `…b427c19380991a4eaa…`) |

3. **The whole object database, unreachable objects included.** The same pass over `git
   cat-file --batch-all-objects`: **23,719** objects, of which **8,742** are blobs, **235**
   of them unreachable from any ref. Nothing new lands in the window. The unreachable blobs
   that carry `19,300,000` are old revisions of the guard comment that §7b moves.

**The only blobs carrying a value inside the window are C151's own** — the ledger row, this
document, the guard constant and the CI comment — by construction, because the window has to
be named to be registered.

**The one hole, closed by hand rather than left open.** Exactly one blob in pass 1 does not
parse: a conflict-marker intermediate of `reports/c102_consumed_choice_double_mutation.json`,
reachable only from the reflog. An unparseable blob makes a "nothing here" claim *easier*, so
it was read directly: the only seeds it names are `19000038` and `19000113`, both dev.

**⚠ CORRECTION to this scan's own first result, recorded rather than edited away.** The first
pass ran `finditer` with the bare pattern `19[,_]?[2-9][0-9]{2}[,_]?[0-9]{3}`, which matches a
**substring** of a longer digit run — so a search-grid node count such as `1930526512` read as
the seed `19,305,265`, and this section reported the false positives as *"8-digit values
inside `docs/audit_artifacts/hc-*` search grids and `uv.lock`"*. The `uv.lock` half was right;
**the `hc-*` half was wrong** — under boundary-correct tokenisation, glob
`docs/audit_artifacts/hc-*` over the same ref set yields **zero** paths with an in-range whole
number, across all 22 such blobs. The error only ever *over*-named false positives, so it
could not weaken the conclusion — which is exactly why it is fixed rather than shrugged at, in
the document whose thesis is that a scope claim must match what was measured. Caught by
independent review.

### 7b. Why this block, and the collision that had to be resolved

* **The owner chose it, and the reasoning is on the record.** `19,2xx` is now a graveyard
  namespace, and **adjacency invites off-by-N archaeology forever** — every future reader of a
  seed near `19,200,3xx` would have to re-derive whether it was inside the C141 overrun. The
  replacement is visibly distinct from everything touched.
* **The disclosure named this block first.** Its third disposition option, written 2026-08-05,
  is *"retire the range, reserve a fresh window entirely (`19,300,000+`)"*. The ratified window
  is that option, taken by the person the disclosure said had to take it.
* **It is inside `#1122`'s guarded half-line.** `FINAL_HOLDOUT_SEED_FLOOR = 19_200_000` is
  unbounded above, so `19,300,000` is unreachable without `--final-holdout-i-mean-it`, in
  execution *and* in `--merge-from` aggregation. No guard change was needed to protect it.
* **It follows the registry's own convention** — one purpose per `19,X00,000` block:
  `19,000,000` dev, `19,100,000` validation holdout, `19,200,000` the burned block,
  `19,500,000` c73.
* **⚠ THE COLLISION, resolved rather than ignored.**
  `scripts/engine_transition_differential.py` previously spent the literal `19,300,000` in
  prose, as the canonical typo the unbounded floor exists to catch — *"a typo of 19,300,000
  should not sail through"*. An illustration that names the real target reads backwards, and a
  reader grepping for the window would find a comment calling it a typo. C151 moves the
  exemplar to **`19,700,000`**, chosen because it **was** absent from **every blob in the
  object database** — reachable and unreachable — immediately before this change. Pinned by
  `test_the_typo_exemplar_no_longer_names_the_ratified_window`.

  **⚠ CORRECTION, and it is a sharper instance of this document's own subject.** That sentence
  was first written in the present tense — *"is absent from every blob"* — and **its own commit
  falsified it.** As committed, **six** blobs carry `19,700,000`, and all six are C151's own:
  the guard comment, the ledger paragraph, this document and the guard test. This is the
  stale-denominator defect one turn tighter — not a measurement that went stale later, but one
  **invalidated by the change that states it** — and it is the second scope defect this
  document has had to correct about its own scans, after the `hc-*` tokenisation error in §7a.
  The claim the evidence supports is about the object database *before* the edit. What keeps
  the exemplar honest going forward is not its absence from prose, which C151 has already
  ended, but that **no artifact ever records a seed there** — which
  `test_every_committed_fidelity_seed_lies_in_a_registered_band` enforces for every band
  outside the registered four. Caught by independent review.

## 8. Prediction

**0–2 genuine divergences, most likely 0 or 1, dominated by shapes already in `c138`** — as
measured on whatever fingerprint is frozen for the claim.

**A registered caveat on this prediction, and it is the honest consequence of §3.** The sweep
will run on a *later* build than the one described in §5, because the trigger requires the
fingerprint to be frozen first and more patches are expected. The reasoning below is derived
from `8e912b45544034e6` and is registered against it. **If the frozen fingerprint differs, this
prediction is re-derived and re-registered in an appendix to this document BEFORE the run, and
the original stays exactly as written** — the same discipline C134's post-rebase
re-registration used. What may never be re-chosen is the window, the falsifiers and §4.

Reasoning, stated so it can be checked against the outcome rather than reinterpreted after it:

* **The base rate from the one virgin window ever measured.** C141 found 1 genuine divergence
  in 16,274 measured boundaries. A 200-game window here should measure roughly 15,500–16,500,
  so that rate alone predicts about 1.
* **Both permitted windows are at 0 on this build (§6), and that is weak evidence.** Dev has
  had seven rows closed against it and is the window every fix iterated on; the validation
  holdout, though nothing was fitted to it, has been swept on every fix branch since C116
  Phase 1 and has had its own attrition. A window that has never been seen has had none. This
  argues for *more* than zero, not less — the same argument C141 made, and it was right.
* **G8 is still OPEN and its live sub-case is explicitly out of C149's reach.** C149 closed
  `19000191/63` by splitting residual-kill bands at the two `i16::MAX`-ceiling call sites. The
  second instance, `19200244/115`, is untouched: its arm is priced at the **survive
  representative** (`sum(band)//len(band) = 145`, not a member of its own fan), every residual
  threshold there lies below the fan minimum, and `residual_disjoint_bands`'s `min_roll <
  threshold` guard cannot pass. The `defender_active.hp`-ceiling call sites were left alone for
  blast radius, **not** because they are immune. Leech Seed plus Leftovers is reachable in
  ordinary play; Leftovers is 72 % of all generated items.
* **G33b has two open arms** — a fatal weather chip when the winner is faster, and exact speed
  ties, where the engine forks both orders and one is mislabelled either way.
* **`c138` lists reachable-but-unobserved shapes** — Farfetch'd's Stick, the double-faint tie,
  PP above 10, the Toxic min-1 clamp — any of which gets its first opportunity here.

**I am not predicting zero.** Zero would be a good result and would be reported as one, but
predicting it would be predicting the flattering outcome on the only window nothing has been
fitted to. The interval has 0 in it; its mode is not 0.

Secondary predictions, registered so they can fail:

* `boundaries_measured` in `15,000`–`17,500`, and `engine_errors` **0**.
* Coverage in `85`–`90` %, and support-gated acceptance (Constraint 7) in `7`–`11` % of
  measured boundaries.
* The four-term partition closes exactly:
  `boundaries_measured == matched + diverged + engine_errors + skip:strict_all_branches_lossy`.

## 9. Falsifier

### 9a. "NOTHING OPENED"

**State the limitation first, because the usual form of this falsifier is unavailable here.**
The subset comparison C133 §7 and C134 use — *every boundary divergent in configuration B must
be divergent in configuration A, compared row by row as `(seed, boundary_index)`* — requires
**two sweeps of the same window**. This window may be swept **once**, ever. There is no base to
subtract, and any claim to have run the standard falsifier here would be false. What follows is
the strongest form a single sweep can carry, and it is deliberately not called a subset test.

> **Every patch that lands between C141 and the frozen fingerprint carries a claim about what
> it cannot open. This window is the first unbiased test of those claims, and any divergence
> attributable to one of them REFUTES it.**
>
> Concretely, C149's band split is claimed in `c138`'s G8 cell to be *"a strict improvement or
> a no-op, never a trade — given that `max_damage_dealt` equals Showdown's own fan maximum"*,
> and that conditional has never been tested on a window nothing was fitted to. **If this
> window produces a divergence on a Leech-Seed-seeded residual-lethal band where the pre-split
> single arm would have matched — a row that exists only because the split emitted the arm that
> misses — the guarantee is refuted and the G8 cell is wrong.** That is decidable offline from
> the retained repro plus a same-tree flag-off replay, and **the replay is not a re-measurement
> and does not touch the window.**
>
> The same test applies to the Attract marker and the Sleep Talk double-reset guard: a
> divergence carrying `attract_empty_tail_ambiguous:*`, or a `damage_dealt`-carry shape that the
> C148 census argued is structurally unreachable on all three engine entry points, refutes that
> argument.

**This is a falsifier for the patches, not a gate on the sweep.** Per §4, the measurement stands
at any X whether or not this clause fires; what it refutes is a specific claim in the ledger.

### 9b. The C141-shaped falsifier, carried forward

> **If the count is large — say above 5 — or if any row is a shape the ledger does not already
> contain, the program's claim that the residue is understood is wrong**, and that is the
> outcome worth learning. A new shape here is more informative than a clean sweep.

### 9c. Refuting outcomes that are about the instrument, not the engine

* **Any `engine_errors > 0`.**
* **`boundaries_measured` outside `15,000`–`17,500`**, or coverage outside `85`–`90` %. The
  window is drawn from the same generator as dev and holdout; movement here is a harness wiring
  defect and makes the row count uninterpretable rather than interesting.
* **Any record whose seed falls outside `19,300,000`–`19,300,199`.** C141 overran its
  registration by 60 seeds and nothing caught it at the time. The artifact must report
  `seeds.min == 19,300,000`, `seeds.max == 19,300,199`, `seeds.distinct == 200`.

### 9d. What would NOT refute it

* A count inside `0`–`2` that differs from the mode. The interval is the prediction.
* A row whose *class string* is new while its *mechanism* is a ledgered gap — C141's
  `component_mismatch:heal|itemleftovers` was G8 wearing a different label because the engine's
  arm tagged its component `itemleftovers` instead of `heal`. Report the mechanism, not the
  string.
* Slower wall clock than C141's 1,480 games/hour.

## 10. The exact command, and the bookkeeping that follows it

**Do not run this until the §3 trigger has fired.** Run **once**.

```sh
# 0. The trigger, which is a human judgement and is not machine-checkable:
#    - the ledger is TERMINAL, and
#    - the engine fingerprint is DECLARED FROZEN for the claim.
#    If either is not true, stop. Nothing below is authorised yet.

# 1. Confirm the tree is the one the registration names.
git rev-parse HEAD
git status --short                       # must be empty: source_tree must record "clean"

# 2. Confirm the installed engine was built from THIS patch set. Do not skip.
python scripts/engine_build_fingerprint.py --check
echo "exit=$?"                           # captured directly, never through a pipe
#    If this is not the fingerprint §5 names, re-derive and re-register the PREDICTION
#    in an appendix first (§8), then come back. The WINDOW is not re-openable.

# 3. THE SWEEP. One invocation, one time, at seeds that can never be reused.
python scripts/engine_transition_differential.py \
    --games 200 \
    --seed-start 19300000 \
    --matcher strict \
    --keep-repro 25 \
    --repros-per-game 8 \
    --final-holdout-i-mean-it \
    --json reports/artifacts/c151_final_holdout_rereg_sweep.json
echo "exit=$?"
```

Notes on the argv, because a run's configuration must be reproducible from its argv alone:

* **`--final-holdout-i-mean-it` is required.** `_reject_unguarded_final_holdout` refuses the
  run without it, checking the whole span rather than the start. It opens the **ratified**
  window only; it does **not** open the burned block.
* **`--matcher strict`** is the default and is stated explicitly. It is what every window in §6
  was measured with.
* **`--keep-repro 25` and `--repros-per-game 8`** are the defaults and the values C141's
  artifact records. Stated so retention is in the argv rather than in a default that could move.
* **`--enumerate-rolls` MUST NOT be passed.** It selects the reference oracle rather than the
  collapsed cascade that production search runs; a sweep taken with it is not evidence about the
  shipping path.
* **`--approximate-sleep` MUST NOT be passed.** Strict hidden-counter handling is what the
  ~9 % support-gated figure is quoted against.
* **`--skip-build-check` MUST NOT be passed.** A stale build does not error; it produces a
  plausible number.

**Ratification bookkeeping, once and only once the sweep has run.** Two pins are designed to go
red on that day, so this cannot be forgotten:
`test_no_committed_artifact_touches_the_ratified_window` in
`tests/test_seed_registry_coverage.py`, and the frozen-banner pin beside it. The steps:

1. commit the sweep artifact;
2. add `(19_300_000, 19_300_199, …)` to `REGISTERED_BANDS`;
3. flip the ledger row from **RATIFIED, NOT YET SWEPT** to **CONSUMED**;
4. replace this document's banner with the run's commit and fingerprint;
5. append the outcome below a `---`, leaving everything above it byte-identical;
6. delete `TheRatifiedC151WindowIsNotYetSweptTests` and drop the seed-registry `Ran N tests`
   guard in `.github/workflows/engine-fidelity-gates.yml` accordingly.

## 11. What executing would cost and forfeit

Short, and every sentence is a cost rather than a caveat.

* **It spends a one-shot namespace for the second and last time.** `19,200,000`–`19,200,199`
  was the first such reservation and is burned. The property that made it valuable — that it
  had never been seen — is not renewable by fiat; it is renewable exactly once per virgin
  block, and each renewal weakens the claim that the reservation is respected.
* **It is irreversible.** The moment the sweep runs those 200 seeds are spent whatever the
  result comes back as, **including if it crashes**, including if the build turns out to have
  been wrong. There is no re-run. `--final-holdout-i-mean-it` is the point at which the guard
  stops protecting the window.
* **It must be the last time.** A third re-registration would establish that the final-holdout
  reservation means "the most recent window", which is the same as meaning nothing. If this
  window is swept, item 13 closes on its result — whatever that result is — or closes as
  unachievable. Not a fourth block.
* **What it does NOT buy.** It does not make the engine correct everywhere, and per §4 it does
  not gate anything. Three standing caveats apply exactly as they do to every other number:
  ~7–11 % of measured boundaries are accepted through the widened sleep-counter bar
  (Constraint 7, enumerating up to 64 worlds and accepting if any one matches); the
  simultaneous last-mon double faint is a terminal-value divergence no differential counter can
  see; and ~13 % of boundaries are single-seat and skipped before comparison, so coverage is
  ~87 %, not 100 %.
* **What deferring costs: nothing on disk.** Waiting for the trigger spends zero seeds, zero
  artifacts and zero rework. The window stays virgin and this registration stays valid. That
  asymmetry is deliberate — the expensive option is the one that requires the trigger.

**C141's artifacts stay in place, demoted.** `reports/artifacts/c141_final_holdout_sweep.json`,
`reports/artifacts/c141_final_holdout_replay.json` and
`reports/c141_final_holdout_prediction.md` are **dev-window evidence** — 200 then-fresh seeds, a
71-patch engine, a self-chosen window — and are **terminal for nothing**. **They are not to be
deleted.** They are the only committed witness for two of the four bands in `REGISTERED_BANDS`,
so removing them turns `test_every_registered_band_has_a_committed_witness` red; they are the
record of what `19,200,060`–`19,200,259` was spent on, and a spent band with no artifact reads
as a virgin band; and a superseded measurement that is retained and labelled is the difference
between a program that corrects itself and one that edits its history.

## 12. What C151 changes on disk

* `scripts/engine_transition_differential.py` — the burn, `OWNER_RATIFIED`,
  `RATIFIED_FINAL_HOLDOUT`, `RATIFIED_SWEEP_PRECONDITION`, and the typo exemplar moved off the
  ratified window.
* `docs/engine_divergence_ledger_20260728.md` — one **RATIFIED, NOT YET SWEPT** row and the
  prose that qualifies it. **The three rows #1189 wrote are untouched**; the burn is recorded in
  prose precisely so the decomposition stays three rows.
* `reports/rust-fidelity/final_holdout_contamination_disclosure.md` — the recovered disclosure,
  committed verbatim at the path the frozen C141 citation names.
* `reports/c141_final_holdout_prediction.md` — the demotion, **appended** below the outcome.
  Lines 1–80 remain byte-identical to `3687d205`, verified.
* `reports/c151_final_holdout_rereg_prediction.md` — this document.
* `tests/test_seed_registry_coverage.py`, `tests/test_final_holdout_guard.py` — the pins.
* `.github/workflows/engine-fidelity-gates.yml` — two exact `Ran N tests` guards, seed-registry
  30 → 41 and final-holdout 14 → 25, which live in YAML and **no local `unittest` or `pytest`
  run can see**.
* `reports/certification_contract_lifecycle.json` — **one JSON is modified after all**, and the
  correction is recorded rather than quietly absorbed. An earlier draft of this section said "no
  JSON is added or modified". That was **false the moment the guard changed**, and the thing
  that caught it was a pin rather than a reading:
  `test_production_matcher_is_not_the_rejected_experiment` binds
  `successor_pending_identity.differential_sha256` to the exact bytes of
  `scripts/engine_transition_differential.py`, so any edit to the differential forces a
  reviewable re-stamp. Verified as a real regression rather than local noise by running the
  module on a throwaway worktree at the base commit `2acd40ff`, where it is green.

**Nothing is ADDED to either corpus, and the one content change moves neither denominator.**
Re-derived by calling each selector itself *after* the JSON edit rather than by arithmetic:
`_sweep_reports()` from `tests/test_boundary_verdict_partition.py` returns **95** against
`_EXPECTED_SWEEP_ARTIFACTS = 95`, and `counter_artifacts()` from
`tests/test_never_fired_counter_census.py` returns **375** against
`_EXPECTED_COUNTER_ARTIFACTS = 375`. Re-derived *after* is the load-bearing part:
`counter_artifacts()` selects on counter-shaped leaves rather than on filenames, so a content
change alone can move a member in or out. It did not, but that was measured. The two
denominators move independently and must not be used to check one another.

**What the re-stamp certifies, per C144's standing instruction that a re-stamp must say which
changes it covers.** The differential's only edit is the seed-admission guard, and that this
touches no classification is **AST-verified rather than asserted**: comparing parsed top-level
definitions before and after gives exactly one added function
(`_reject_burned_final_holdout`) and two changed ones (`_reject_unguarded_final_holdout`,
`_reject_reserved_seeds_in_records`), **nothing removed**, 60 definitions before and 61
after, and 35 → 39 module-level assignments with **none removed or changed**. No matcher,
classifier, attributor or counter function is in that set, so no committed number is
re-derived and no boundary changes verdict. The identity was re-stamped **twice** in this
PR: once for the guard itself, and once for the comment-only past-tense fix above, which
is AST-verified *identical* to the previous stamp and moved the hash only because the
hash is over file bytes — the intended sensitivity, not a false alarm. What *does* change is which inputs are **admitted**:
a run or a `--merge-from` over the burned block is now refused where the opt-in previously
permitted it. C141's committed checkpoint carries 200 of those seeds, so that is a live effect
rather than a hypothetical one.
