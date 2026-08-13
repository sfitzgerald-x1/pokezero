# Encode efficiency closeout — finishing the program #1217 started

Written 2026-08-12. **Amended 2026-08-13 from C0's measured evidence** — the amendment corrects
this plan's base figure (encode is 80.8% of search wall at production config, not the ~70–75%
forward estimate), restates the success criterion in denominator-proof units (the old share form
was ill-posed; see §5), adds a fifth gate class the C1 review proved necessary (§4), and marks
program state (§2). Owner-directed: close out the encode work. This document is the execution
contract for finishing what `docs/encode_optimization_plan_20260810.md` (#1217) opened; that
plan's gates and scope fences apply verbatim here and are not restated in full.

## 1. Why this is now the highest-leverage engineering line

When #1217 was written, encode efficiency was "eval-economics work": search was a marginal
operator and cheaper leaves mostly made reports faster. Two things changed:

1. **Search now measurably beats its prior.** With applied opponent priors (#1207) and the
   fallback ladder driven to zero (#1224, #1240, #1241, #1234, #1237, #1242 and the census line
   behind them), the first search cell has cleared parity against the reference bot with its
   whole confidence interval, ~13 points over the raw policy — and the leading cells are the
   high-sims ones. Sims are now a strength currency with a measured exchange rate.
2. **Every planned consumer multiplies the payoff.** Larger per-decision budgets, the
   budget-allocating dynamic search under design, and any future search-in-training experiment
   all spend sims; encode is the price of a sim. C0 measured the production config on the
   current build: **encode is 80.8% of search wall**, and the two staged targets are larger
   than #1217 estimated — `row_write` 27.4% (Stage 1) and `row_input` 22.6% (Stage 2),
   **50.0% of search wall combined**. (The original text here carried a ~70–75% forward
   estimate and #1217's 41.5%; C0 showed no post-#1221 measurement existed anywhere in the
   campaign data — every prior figure was from pre-#1221 builds.)

Retiring encode as the dominant cost is therefore no longer an optimization nicety; it is the
prerequisite economics for every search direction currently on the table.

## 2. State of the program, updated 2026-08-13

| item | state | evidence |
|---|---|---|
| Empty-cell short-circuit | **MERGED** | #1221 (`perf(encode): short-circuit empty cells before normalize_category`), −8.4% encode wall median |
| **C0 — the new "before"** | **DONE** | three-cell profile campaign on the current build, one render group; production config: encode 80.8% / tensor 63.0% / row_input 22.6% / products 3.4% / row_write 27.4%; the axis study never landed, so its method section landed with C0 per this plan's own fallback |
| Stage 1, first bite (#1219) | **APPROVED — merging** | all 8 per-leaf `format!` md keys retired via `md_key!`, which derives both spellings from one suffix token so a transposition is inexpressible; `get(md, &format!(…))` is gone from `encoder.rs` |
| Stage 1 proper | **MERGED** (01a06704) | #1249 (constant column/offset lookups resolved once at table load); adversarial review found the cited test evidence had zero detection power (a mutated field mapping stayed green) — fixed with a frozen 91-triple field↔constant table + per-field distinct-index test, mutation-verified; **residue: 63 constant names still string-hashed per leaf across 22 dynamic sites** |
| Stage 2 (typed Row, retire per-leaf clone) | **NOT STARTED**, conditional per #1217 §3 | `leaf_row_inputs` still starts from `self.root.clone()`; C0 sizes it at 22.6% of search wall |
| C3 remainder | **SIZED** | the three sub-timers cover only ~85% of `tensor` (stable within 0.3% across four independent measurements), and ~19% of `encode` lies outside `tensor` — the unattributed remainder is real, not noise |
| Byte-identity harnesses | **IN PLACE, WITH A PROVEN COVERAGE GAP** | golden corpus + fold comparison, `assert_vocab_alignment`, differential windows, live A/B shard-byte protocol per #1217 §4 — all schema-shaped; see §4's fifth gate |

Also relevant: world collapse (#1009) reduced searches issued since the 17.2%/24.3% attribution
was measured, and the fallback ladder reaching zero changed which decisions search at all — a
second reason the shares must be re-measured before Stage 2's go/no-go is written.

## 3. Closeout stages

**C0 — re-profile on current main (the new "before").** Re-run the profile cells (baseline
config plus the two highest-encode cells) on the current build, same seed block and node class
discipline as the study. Deliverable: per-phase walls (`row_input`, `tensor`, `products`,
`row_write`, `encode`, `fold_clone`) that (a) replace the stale before, (b) size Stage 1 vs
Stage 2 on the post-#1221, zero-fallback build. Half a day of fleet time; nothing merges before
it exists. If the study document itself has not landed by then, its method section lands with
C0's numbers so the baseline is citable.

**C1 — Stage 1 to completion.** (Amended with the review record.) The positional-resolution
core is #1249; its review is part of the program's evidence and its mapping test is now Gate 5.
Completing the stage means, beyond landing #1249: absorbing #1219, now DONE: all 8 sites are converted, so zero `format!` md-key
allocations per leaf remain (its original five were the smaller half), and migrating the **63 constant names still string-hashed per
leaf across 22 dynamic sites**, which outnumber the 58 already moved. C0 sizes the whole stage
at `row_write` = 27.4% of search wall. The C4 after-measurement must attribute against this
inventory — a partial C1 read as "Stage 1 complete" would overstate capture and corrupt the C2
go/no-go.

**C2 — Stage 2, go/no-go from C1's after-walls.** The typed struct holding the invariant root
plus a per-leaf delta (or copy-on-write on the mutated subtrees), subsuming the per-leaf
`serde_json` deep clone that `leaf_row_inputs` performs today (17.2% at last measurement). Per
#1217 §3 this proceeds **only if** C1's after-profile still shows the round-trip dominating; the
decision is written from C0/C1 numbers, not from this document.

**C3 — the remainder decision.** After C1 (+C2), either the success criterion (§5) is met and
the program closes, or the residual is the JSON mirror boundary itself, in which case the typed
path is completed for the 23-row contract. The known blocker from the corpus work gets decided
HERE, explicitly: corpus rows do not carry the event-stream inputs behind the tendency-family
columns, so those columns are not reproducible per-row from the corpus alone. Two resolutions,
one to be chosen on C2's evidence: extend corpus generation to capture the tendency inputs at
authoring time, or accept corpus coverage for the other columns and let the live A/B shard-byte
gate (which exercises real world construction end to end) carry the tendency columns. What is
not acceptable is silently narrowing the byte-identity claim — the closeout ledger states
per-column how identity was proven.

**C4 — after-measurement and re-anchoring.** The final profile at production config, side by
side with C0; refresh every consumer of the cost curves (the axis cost tables, and any
budget/depth-cap tables derived from realized-depth-vs-sims, which shift when sims-per-wall
does); close with a one-page ledger: per-stage before/after walls, bytes-identical attestations,
and the measured end state.

## 4. Gates — #1217 §4 applies verbatim, per stage, plus a fifth the C1 review proved necessary

Golden-corpus bit-exactness, `assert_vocab_alignment`, differential windows with the standing
"nothing opened" falsifier, and the live A/B shard-byte identity run on the same seed block —
all four per stage, with per-arm identity witnesses. The known tax rides in the same PR each
time: any Rust source edit moves the source-bytes fingerprint, so each stage carries its
surgical citation re-resolution (the c155 mechanism), never a cold regen. A performance win is
never traded against encoder agreement; a reddened gate stops the stage.

**Gate 5 — static mapping assertions for surfaces the byte gates cannot see.** The C1 review
proved by mutation that all four gates above are **schema-shaped**: they compare emitted bytes at
the schema version under test, and 18 of the 91 encoder fields do not exist at v4 (the
`CATEGORY_TM_*` family, `TM2_PRESENT`, and the `TT_*` transition block) while 2 do not exist at
v2.2/v3 (`LAST_USED_MOVE`, `TRACED_ABILITY`). A field mismapping confined to those fields passes
a single-schema corpus, a single-schema differential window, and a single-schema live A/B alike
— silently, forever. Any stage that rewires field resolution therefore also carries a
**Rust-side static mapping test**: a frozen table of every `(field, source, constant)` triple,
a per-field distinct-index assertion (so a swap cannot hide behind equal values), an
anti-vacuity floor on the table's size, and a demonstrated mutation kill. The closeout ledger
states per column whether identity was proven by bytes, by the static mapping, or both — the
byte-identity claim is never silently narrowed to "the fields this schema happens to exercise."

## 5. Success criteria and sequencing

**Numeric close condition — restated 2026-08-13, because the share form was ill-posed.** The
original "encode ≤ 40% of search wall" had a shifting denominator: removing encode work shrinks
the wall it is a share of. At C0's measured base (encode 80.8%, stages worth 50.0% combined),
even capturing the two stages *entirely* leaves encode at 30.8 points of the *original* wall —
which is **61.6% of the new wall**. Under the honest denominator the old target is unreachable
from stages 1–2 at any capture rate; under the original-wall denominator it quietly demanded
~80% capture. The criterion is therefore restated in the units the program actually exists to
move, which no denominator shift can distort:

- **Close condition (C1–C2): per-decision search wall at production config reduced by ≥ 40%**
  — equivalently **≥ 1.67× effective sims at fixed wall** — which at the measured shares means
  capturing ≥ 80% of the two stages' combined 50.0%. Judged on `c0-d4-s2048-b64-w1`'s cell,
  same render discipline, against C0's table.
- **Stretch (with C3): wall reduced ≥ 60% (≥ 2.5× sims at fixed wall).** C3 is where the
  stretch lives because C0 also sized what the stages cannot reach: ~15% of `tensor` is outside
  all three sub-timers and ~19% of `encode` is outside `tensor` — encode remains the majority
  share of the post-stage wall even at full stage capture, so closing past the stretch requires
  the remainder, not more of stages 1–2.

Every claim is a measured phase-wall from the C0-established harness, never an extrapolation,
and never differenced across render groups or builds.

**Sequencing fences:** no encode change lands mid-campaign (#1217 §2). Concretely: stages merge
only between strength-panel waves — a mid-panel merge silently shifts the timing baselines every
cell is compared on — and panel manifests record the search-crate fingerprint so any cross-build
comparison is detectable after the fact rather than assumed. The collection/training path is
untouched by construction (search is not in the training loop; this remains eval-and-play
economics, now with strength attached).

**Effort estimate:** C0 ~half a day; C1 ~1–2 days including #1219's review; C2 ~2–4 days if it
fires; C3 decision-scoped. The program is closeable inside two working weeks without touching
any campaign window.

## 6. Non-goals, restated once

No observation-schema or Python-encoder changes; no emitted-value changes of any kind; no
opportunistic refactors riding stage PRs; no batching or architectural redesign of the search
loop under this program — a stage diff is the optimization, its tests, and the citation
re-resolution, nothing else.
