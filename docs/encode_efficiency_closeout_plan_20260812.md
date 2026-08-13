# Encode efficiency closeout — finishing the program #1217 started

Written 2026-08-12. Owner-directed: close out the encode work. This document is the execution
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
   all spend sims; encode is the price of a sim. At the measured production config, encode-side
   work is ~70–75% of search wall (post-#1221), so the two staged changes of #1217 — worth
   ~41.5% of decision wall between them — are the difference between "s2048 economically
   behaves like ~s800" and near-parity between nominal and effective budget.

Retiring encode as the dominant cost is therefore no longer an optimization nicety; it is the
prerequisite economics for every search direction currently on the table.

## 2. State of the program, verified against main 2026-08-12

| item | state | evidence |
|---|---|---|
| Empty-cell short-circuit | **MERGED** | #1221 (`perf(encode): short-circuit empty cells before normalize_category`), −8.4% encode wall median |
| Stage 1, first bite | **OPEN, unreviewed** | #1219: five per-leaf `format!` md keys become `&'static str` (`rust/pokezero-search/src/encoder.rs`, +29/−6) |
| Stage 1 proper (~55 keyed reads → positional) | **NOT STARTED** | `encode_row_value` still reads row fields through string-keyed `get()`s |
| Stage 2 (typed Row, retire per-leaf clone) | **NOT STARTED**, conditional per #1217 §3 | `leaf_row_inputs` still starts from `self.root.clone()` |
| The "before" profile #1217 §2 relies on | **STALE BY THE PLAN'S OWN RULE** | the axis study was profiled on the pre-#1221 build; #1217 §2: wall-clock comparisons are valid only within one build |
| Byte-identity harnesses | **IN PLACE** | golden corpus + fold comparison (`src/pokezero/golden_corpus*.py`, `test_golden_corpus_fold.py`), `assert_vocab_alignment`, differential windows, live A/B shard-byte protocol per #1217 §4 |

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

**C1 — Stage 1 to completion.** Review and land #1219 (it is a strict subset of Stage 1 and has
sat unreviewed since 08-10), then the load-time resolution of the remaining string paths in
`encode_row_value` to positional indices — resolved once against the run-fixed schema at table
load, read positionally per leaf. Mechanical, no data-shape change, no Python-side change.
Target: the bulk of the (re-measured) lookup share, which #1217 sized at 24.3% of decision wall.

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

## 4. Gates — #1217 §4 applies verbatim, per stage, no exceptions

Golden-corpus bit-exactness, `assert_vocab_alignment`, differential windows with the standing
"nothing opened" falsifier, and the live A/B shard-byte identity run on the same seed block —
all four per stage, with per-arm identity witnesses. The known tax rides in the same PR each
time: any Rust source edit moves the source-bytes fingerprint, so each stage carries its
surgical citation re-resolution (the c155 mechanism), never a cold regen. A performance win is
never traded against encoder agreement; a reddened gate stops the stage.

## 5. Success criteria and sequencing

**Numeric close condition:** encode-side share of search wall **≤ 40%** at production config
after C1–C2 (from ~70–75% today) — equivalently ≥1.7× effective sims at fixed wall — with
**≤ 25%** the stretch condition that C3 exists to reach if C2's residual justifies it. Every
claim is a measured phase-wall from the C0-established harness, never an extrapolation.

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
