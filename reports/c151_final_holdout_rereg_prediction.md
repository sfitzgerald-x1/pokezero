# C151 — UNREGISTERED: proposed re-registration of the final-holdout window

> ## ⚠ UNREGISTERED — NOT YET FROZEN, SWEEP NOT RUN
>
> **This is not a registered prediction.** Nothing in this document has been frozen, no
> sweep has been executed, and no seed at or above `19,200,000` has been spent by C151.
> It becomes a registered pre-registration only when the owner ratifies the window in
> writing and this banner is replaced by a freeze note naming the commit the freeze
> happened at. Until then, treat every number below as a *proposal* and the window as
> **virgin**.
>
> The corresponding ledger row — `docs/engine_divergence_ledger_20260728.md`, seed
> registry, `19,400,000`–`19,400,199` — is marked **PROPOSED, NOT RATIFIED** for the same
> reason. It proposes; it does not authorise.

## 1. Why this document exists, and what it is not

C134 §5.2 stands: *"only the owner can bless the replacement window. Item 13 stays frozen
until then."*

C141 spent the registered final holdout. An adversarial audit ruled that its window was
chosen by the executing agent rather than deferred to the owner — the pre-registration
says so in its own words, *"chosen by me rather than deferred"*
(`reports/c141_final_holdout_prediction.md:16`) — and that this defeats completion of
C116 Phase 4 item 13.

Asked how to proceed, the owner said, verbatim and in full:

> **"can we sweep a new set of seeds then?"**

That is the owner *initiating* a fresh window. It is not, by itself, ratification of what a
fresh window now requires, and the difference is the whole reason this document is
unregistered rather than frozen. **The registered block `19,200,000`–`19,200,199` has no
virgin seeds left:**

| segment | disposition |
|---|---|
| `19,200,000`–`19,200,059` | contaminated by a pre-guard 60-game convenience loop; JSON deleted unread |
| `19,200,060`–`19,200,199` | consumed by C141 |
| `19,200,200`–`19,200,259` | consumed by C141's overrun past the registered end |

So a fresh window cannot be a *replacement span inside the registered block*. It is a
**re-registration of the final-holdout namespace onto a new seed block**, which is
materially more than "sweep a new set of seeds" was asked to bless. Hence: prepared, not
executed. The decision is cheap to act on — §7 is one command — and costless to decline —
declining changes nothing on disk, because C151 commits no JSON and spends no seeds.

**Verbatim-words provenance.** Neither the owner's sentence above nor the C134 §5.2 clause
appears anywhere in this repository prior to this document. Scope of that negative: a
literal case-insensitive search for `can we sweep a new set of seeds`, `bless the
replacement` and `replacement window` over **all 8,499 blobs** reachable from **every ref
and the reflog** (`git rev-list --objects --all --reflog`), any path, any file type. Zero
hits for the first two; the one hit for the third is `src/pokezero/showdown.py`, an
unrelated forced-switch usage. This document is therefore the first in-repo record of
both, and it records them rather than paraphrasing them.

## 2. The build this prediction is made against, verified rather than quoted

`main` `2acd40ff`, engine fingerprint **`8e912b45544034e6…`, 74 patches**. Re-derived on
this worktree by running `python scripts/engine_build_fingerprint.py --print`, not copied
from a message:

```
"fingerprint": "8e912b45544034e67d4f168d9947bf63e769d6337f7fe27cb5c44a5addf0ca27",
"count": 74
```

**C141 measured a build that no longer exists.** Re-derived the same way, by extracting
each commit's tree with `git archive` and running that tree's own copy of the script:

| commit | role | fingerprint | patches |
|---|---|---|---|
| `3687d205` | C141's pre-registration commit, the tree the sweep executed at | `44ee1430708cbb55` | **71** |
| `16857e06` | `main` as the C141 prediction names it | `44ee1430708cbb55` | **71** |
| `aa2f2d40` | the commit C141's sweep landed at | `44ee1430708cbb55` | **71** |
| `2acd40ff` | `main` today | `8e912b45544034e6` | **74** |

The patch-stack delta is exactly three files, all **appended**, with **no existing patch
modified** — `diff` of the two `third_party/poke-engine-gen3-patches.txt` manifests,
comments and blanks stripped, shows `71a72,74` and nothing else:

1. `poke-engine-gen3-attract-marker.patch` — the Attract immobilizer marker
2. `poke-engine-gen3-sleeptalk-damage-dealt-double-reset.patch` — the Sleep Talk double-reset guard
3. `poke-engine-gen3-leechseed-residual-band-split.patch` — the Leech Seed residual band split, which closed the last dev row

**Stated precisely, because the loose version would be wrong.** "Three engine patches
landed after C141" is true of the patch stack and it is *not* the whole reason the
fingerprint moved: the search crate's own sources are fingerprint inputs too, and
`git diff --stat 3687d205 2acd40ff` over the fingerprint's input set shows
`rust/pokezero-search/src/{events.rs, leaf.rs, model.rs, lib.rs}` changed and
`abort_telemetry.rs` added, alongside the three new patch files and the manifest — nine
paths, 3,074 insertions, 253 deletions. So the exact claim is: **the terminal claim C141
registered describes an engine that no longer ships, the patch stack moved 71 → 74 by
three appended patches, and the crate sources moved as well.** Both halves are checkable
from the table and the diff above.

## 3. State on the two permitted windows, on this exact build

Re-derived from the committed artifacts, not carried from a message. The only two sweep
artifacts on `main` carrying fingerprint `8e912b45544034e6` are C149's pair:

| window | artifact | measured | matched | diverged | engine errors | `skip:strict_all_branches_lossy` |
|---|---|---|---|---|---|---|
| dev `19,000,000`–`19,000,199` | `reports/artifacts/c149_split_dev_sweep.json` | 15,503 | 15,503 | **0** | 0 | 0 |
| validation holdout `19,100,000`–`19,100,199` | `reports/artifacts/c149_split_holdout_sweep.json` | 15,579 | 15,579 | **0** | 0 | 0 |

Both are 200 games, `--matcher strict`, collapsed roll path, `divergence_classes` empty.
**This is the first build on which both permitted windows read zero.** On C141's build the
pair read dev 1 / holdout 0.

Coverage on the same two artifacts, re-derived rather than assumed:

| window | full-round | single-seat | denominator | coverage | support-gated |
|---|---|---|---|---|---|
| dev | 15,968 | 1,742 | 17,710 | 87.5 % | 1,347 of 15,503 = 8.7 % |
| holdout | 16,155 | 1,813 | 17,968 | 86.7 % | 1,431 of 15,579 = 9.2 % |

## 4. The proposed window, and the evidence that it is virgin

**`19,400,000`–`19,400,199`, 200 games** — the same size as every other window in this
program.

### 4a. Virginity, with the scope written into the sentence

Two independent scans, both over **every ref and the reflog**, not over `main`:

1. **Shape-agnostic, structural.** `_seed_intervals` from
   `tests/test_seed_registry_coverage.py` — the extractor #1189 built precisely because a
   selector keyed on `seeds.min` under `reports/artifacts/` misses `c73` — run against
   **every `*.json` blob under `reports/` or `docs/` reachable from
   `git rev-list --objects --all --reflog`**: 900 distinct blobs, of which 145 reach
   fidelity seed space. The union of every fidelity seed touched in any ref is exactly:

   ```
   19,000,000 - 19,000,199   (200 seeds)   dev
   19,100,000 - 19,100,199   (200 seeds)   validation holdout
   19,200,060 - 19,200,259   (200 seeds)   C141
   19,500,000 - 19,500,799   (800 seeds)   c73
   ```

   Overlap with `19,400,000`–`19,400,199`: **none.**

2. **Raw, shape-free, whole-repository.** A regex pass for any seed-shaped token in
   `19,200,260`–`19,999,999` over **all 8,499 blobs** in the same ref set — any path, any
   file type, so markdown, YAML, shell and Python are included, not just JSON. The only
   `19,3xx,xxx`/`19,4xx,xxx` hits are (a) `19,300,000` in
   `scripts/engine_transition_differential.py:3531`, which is prose in a comment and not a
   seed, and (b) 8-digit values inside `docs/audit_artifacts/hc-*` search grids and
   `uv.lock`, which scan 1 confirms sit under no seed-keyed path. Nothing lands in
   `19,400,000`–`19,400,199`.

**The one hole, closed by hand rather than left open.** Exactly one blob in scan 1 does not
parse: `18780107`, a conflict-marker intermediate of
`reports/c102_consumed_choice_double_mutation.json` reachable only from the reflog. An
unparseable blob makes a "nothing here" claim *easier*, so it was read directly: the only
seeds it names are `19000038` and `19000113`, both inside the dev band.

### 4b. Why this block

* **It is inside `#1122`'s guarded half-line.** `FINAL_HOLDOUT_SEED_FLOOR = 19_200_000` is
  unbounded above, so `19,400,000` is unreachable without `--final-holdout-i-mean-it`, in
  execution *and* in `--merge-from` aggregation. The window is protected by the existing
  guard on day one; no guard change is needed, and none is proposed.
* **It follows the convention this registry already uses** — one purpose per
  `19,X00,000` block: `19,000,000` dev, `19,100,000` validation holdout, `19,200,000` final
  holdout, `19,500,000` c73.
* **Clearance is large in both directions.** 199,741 seeds above C141's true end
  (`19,200,259`) and 99,801 below `c73`'s start (`19,500,000`). C141 overran its
  registration by 60 seeds; an overrun of that scale here, in either direction, cannot reach
  a consumed band, and an overrun is detectable long before it does.
* **`19,300,000` was rejected for a specific reason, not skipped.**
  `scripts/engine_transition_differential.py:3531` already spends that number in prose as
  the canonical typo the unbounded floor exists to catch — *"a typo of 19,300,000 should not
  sail through"*. Registering it as a real window would make the guard's own comment read
  backwards and would require editing the guard file for a documentation-only change.
* **It is not the span proposed earlier in this thread.** `19,200,300`–`19,200,499` was
  proposed on the false premise that "everything at or above `19,200,260` is virgin",
  asserted from a `seeds.min`-under-`reports/artifacts/` selector that structurally cannot
  see `c73`. That span happens to be virgin, but the reasoning that produced it was the
  exact defect #1189 exists to prevent, and neither the span nor the reasoning is inherited
  here.

## 5. Prediction

**0–2 genuine divergences, most likely 0 or 1, dominated by shapes already in `c138`.**

Reasoning, stated so it can be checked against the outcome rather than reinterpreted after
it:

* **The base rate from the one virgin window ever measured.** C141 found 1 genuine
  divergence in 16,274 measured boundaries. A 200-game window here should measure roughly
  15,500–16,500 boundaries, so the C141 rate alone predicts about 1.
* **Both permitted windows are at 0 on this build (§3), and that is weak evidence.** Dev has
  had seven rows closed against it and is the window every fix iterated on; the validation
  holdout, though nothing was fitted to it, has been swept on every fix branch since C116
  Phase 1 and has had its own attrition. A window that has never been seen has had none.
  This argues for *more* than zero, not less — the same argument C141 made, and it was
  right.
* **G8 is still OPEN and its live sub-case is explicitly out of C149's reach.** C149 closed
  `19000191/63` by splitting residual-kill bands at the two `i16::MAX`-ceiling call sites.
  The second instance, `19200244/115`, is untouched: its arm is priced at the **survive
  representative** (`sum(band)//len(band) = 145`, not a member of its own fan), every
  residual threshold there lies below the fan minimum, and `residual_disjoint_bands`'s
  `min_roll < threshold` guard cannot pass. The `defender_active.hp`-ceiling call sites were
  left alone for blast radius, **not** because they are immune. That mechanism is live on
  `8e912b45544034e6` and a Leech-Seed-plus-Leftovers configuration is reachable in ordinary
  play (Leftovers is 72 % of all generated items).
* **G33b has two open arms** — a fatal weather chip when the winner is faster, and exact
  speed ties, where the engine forks both orders and one is mislabelled either way.
* **`c138` lists reachable-but-unobserved shapes** — Farfetch'd's Stick, the double-faint
  tie, PP above 10, the Toxic min-1 clamp — any of which gets its first opportunity here.

**I am not predicting zero.** Zero would be a good result and would be reported as one, but
predicting it would be predicting the flattering outcome on the only window nothing has
been fitted to. The interval above has 0 in it; its mode is not 0.

Secondary predictions, registered so they can fail:

* `boundaries_measured` in `15,000`–`17,500`, and `engine_errors` **0**.
* Coverage in `85`–`90` %, and support-gated acceptance (Constraint 7) in `7`–`11` % of
  measured boundaries — the dev/holdout figures in §3 are 8.7 % and 9.2 %.
* The four-term partition closes exactly:
  `boundaries_measured == matched + diverged + engine_errors + skip:strict_all_branches_lossy`.

## 6. Falsifier

### 6a. "NOTHING OPENED"

**State the limitation first, because the usual form of this falsifier is unavailable
here.** The subset comparison C133 §7 and C134 use — *every boundary divergent in
configuration B must be divergent in configuration A, compared row by row as
`(seed, boundary_index)`* — requires **two sweeps of the same window**. This window may be
swept **once**, ever. So there is no base to subtract, and any claim to have run the
standard falsifier here would be false. What follows is the strongest form a single sweep
can carry, and it is deliberately not called a subset test.

> **The three patches that landed after C141 each carry a claim about what they cannot
> open. This window is the first unbiased test of those claims, and any divergence
> attributable to one of them REFUTES it.**
>
> Concretely, C149's band split is claimed in `c138`'s G8 cell to be *"a strict improvement
> or a no-op, never a trade — given that `max_damage_dealt` equals Showdown's own fan
> maximum"*, and that conditional has never been tested on a window nothing was fitted to.
> **If this window produces a divergence on a Leech-Seed-seeded residual-lethal band where
> the pre-split single arm would have matched — i.e. a row that exists only because the
> split emitted the arm that misses — the guarantee is refuted and the G8 cell is wrong.**
> That is decidable offline from the retained repro plus a same-tree flag-off replay, and
> **the replay is not a re-measurement and does not touch the window.**
>
> The same test applies to the Attract marker and the Sleep Talk double-reset guard: a
> divergence carrying `attract_empty_tail_ambiguous:*` or a `damage_dealt`-carry shape that
> the C148 census argued is structurally unreachable on all three engine entry points
> refutes that argument.

### 6b. The C141-shaped falsifier, carried forward

> **If the count is large — say above 5 — or if any row is a shape the ledger does not
> already contain, the program's claim that the residue is understood is wrong**, and that
> is the outcome worth learning. A new shape here is more informative than a clean sweep.

### 6c. Refuting outcomes that are about the instrument, not the engine

* **Any `engine_errors > 0`.**
* **`boundaries_measured` outside `15,000`–`17,500`**, or coverage outside `85`–`90` %. The
  window is drawn from the same generator as dev and holdout; movement here is a harness
  wiring defect and makes the row count uninterpretable rather than interesting.
* **Any record in the output whose seed falls outside `19,400,000`–`19,400,199`.** C141
  overran its registration by 60 seeds and nothing caught it at the time. The artifact must
  report `seeds.min == 19,400,000`, `seeds.max == 19,400,199`, `seeds.distinct == 200`.

### 6d. What would NOT refute it

* A count inside `0`–`2` that differs from the mode. The interval is the prediction.
* A row whose *class string* is new while its *mechanism* is a ledgered gap — C141's
  `component_mismatch:heal|itemleftovers` was G8 wearing a different label because the
  engine's arm tagged its component `itemleftovers` instead of `heal`. Report the mechanism,
  not the string.
* Slower wall clock than C141's 1,480 games/hour.

## 7. The exact command

Run **once**. Nothing else in this document authorises it; see §8.

```sh
# 0. Confirm the tree is the one this document names.
git rev-parse HEAD                       # must be the ratified commit
git status --short                       # must be empty: source_tree must record "clean"

# 1. Confirm the installed engine was built from THIS patch set. Do not skip.
python scripts/engine_build_fingerprint.py --check
echo "exit=$?"                           # captured directly, never through a pipe
#    Expected: 8e912b45544034e67d4f168d9947bf63e769d6337f7fe27cb5c44a5addf0ca27, 74 patches.

# 2. THE SWEEP. One invocation, one time, at seeds that can never be reused.
python scripts/engine_transition_differential.py \
    --games 200 \
    --seed-start 19400000 \
    --matcher strict \
    --keep-repro 25 \
    --repros-per-game 8 \
    --final-holdout-i-mean-it \
    --json reports/artifacts/c151_final_holdout_rereg_sweep.json
echo "exit=$?"
```

Notes on the argv, because a run's configuration must be reproducible from its argv alone:

* **`--final-holdout-i-mean-it` is required** and is the whole reason this is a decision.
  `_reject_unguarded_final_holdout` refuses the run without it, checking the whole span
  rather than the start.
* **`--matcher strict`** is the default and is stated explicitly. It is what every window in
  §3 was measured with.
* **`--keep-repro 25` and `--repros-per-game 8`** are the defaults and are the values C141's
  artifact records. Stated so the retention policy is in the argv rather than in a default
  that could move.
* **`--enumerate-rolls` MUST NOT be passed.** It selects the reference oracle rather than the
  collapsed cascade that production search runs; a sweep taken with it is not evidence about
  the shipping path.
* **`--approximate-sleep` MUST NOT be passed.** Strict hidden-counter handling is what the
  ~9 % support-gated figure is quoted against.
* **`--skip-build-check` MUST NOT be passed.** A stale build does not error; it produces a
  plausible number.

## 8. What executing would cost and forfeit

Short, and every sentence is a cost rather than a caveat.

* **It re-registers a one-shot namespace.** `19,200,000`–`19,200,199` was reserved as the
  single terminal window and is gone. Ratifying `19,400,000`–`19,400,199` creates a *second*
  such reservation. The property that made the first one valuable — that it had never been
  seen — is not renewable by fiat; it is renewable exactly once per virgin block, and each
  renewal weakens the claim that the reservation is respected.
* **It is irreversible.** The moment the sweep runs, those 200 seeds are spent whatever the
  result comes back as, including if the run crashes, including if the build turns out to
  have been wrong. There is no re-run. `--final-holdout-i-mean-it` is the point at which the
  guard stops protecting the window.
* **It must be the last time.** A third re-registration would establish that the
  final-holdout reservation means "the most recent window", which is the same as meaning
  nothing. If this window is ratified and swept, the honest disposition is that item 13 is
  closed on its result — whatever that result is — or that it is closed as unachievable. Not
  a fourth block.
* **What it does NOT buy.** It does not make the engine correct everywhere. The three
  standing caveats apply to this number exactly as they do to every other: ~7–11 % of
  measured boundaries are accepted through the widened sleep-counter bar (Constraint 7,
  enumerating up to 64 worlds and accepting if any one matches); the simultaneous last-mon
  double faint is a terminal-value divergence no differential counter can see; and ~13 % of
  boundaries are single-seat and skipped before comparison, so coverage is ~87 %, not 100 %.
* **What it forfeits if declined: nothing on disk.** Declining costs zero seeds, zero
  artifacts and zero test changes. C151 commits no JSON. The window stays virgin and this
  document stays unregistered. That asymmetry is deliberate — the expensive option should be
  the one that requires a decision.

**C141's artifact stays in place.** `reports/artifacts/c141_final_holdout_sweep.json`,
`reports/artifacts/c141_final_holdout_replay.json` and
`reports/c141_final_holdout_prediction.md` are a **superseded, unratified measurement** —
superseded because they measured `44ee1430708cbb55` / 71 patches and `main` ships
`8e912b45544034e6` / 74. **They are not to be deleted, and nothing here supersedes the
seeds they spent.** Three reasons, in increasing order of how badly deletion would go:
they are the only committed witness for two of the four bands in the seed registry, so
removing them turns `test_every_registered_band_has_a_committed_witness` red; they are the
record of what `19,200,060`–`19,200,259` was spent on, and a spent band with no artifact
reads as a virgin band; and a superseded measurement that is *retained and labelled* is the
difference between a program that corrects itself and one that edits its history.

## 9. What C151 changes on disk, and what it does not

* `docs/engine_divergence_ledger_20260728.md` — one **proposed, unratified** row added to
  the seed registry, plus the prose that qualifies it. **The three rows #1189 wrote are
  untouched.** They were deliberately silent on whether a future sweep is permitted; that
  silence is preserved.
* `reports/c151_final_holdout_rereg_prediction.md` — this file.
* `tests/test_seed_registry_coverage.py` — `TheProposedC151WindowIsNotRatifiedTests`, which
  pins the proposal's *unratified* state and re-derives the window's virginity from the live
  corpus, plus the row-count bump the new table row requires.
* `.github/workflows/engine-fidelity-gates.yml` — the `seed-registry` job's exact
  `Ran N tests` guard, 30 → 36. That guard lives in YAML and **no local `unittest` or
  `pytest` run can see it**; a module that grows without its count following is how an
  approved PR has reached CI red in this repository before.

**No JSON is added or modified**, so neither corpus denominator moves. Re-derived by calling
each selector itself on this tree rather than by arithmetic: `_sweep_reports()` from
`tests/test_boundary_verdict_partition.py` returns **95** against
`_EXPECTED_SWEEP_ARTIFACTS = 95`, and `counter_artifacts()` from
`tests/test_never_fired_counter_census.py` returns **375** against
`_EXPECTED_COUNTER_ARTIFACTS = 375`. The two move independently and must not be used to
check one another.

**`REGISTERED_BANDS` is deliberately NOT extended.** Its own rule is that a band joins it
only after the sweep that fills it is committed; an unwitnessed band would fail
`test_every_registered_band_has_a_committed_witness` — correctly, because the band has no
witness. When and only when the sweep runs, the ratification bookkeeping is: add
`(19_400_000, 19_400_199, …)` to `REGISTERED_BANDS`, flip the ledger row from PROPOSED to
CONSUMED, replace this document's banner with a freeze note, append the outcome below a
`---`, and delete `TheProposedC151WindowIsNotRatifiedTests`. That test class is designed to
go **red on the day the sweep lands**, so the bookkeeping cannot be forgotten.
