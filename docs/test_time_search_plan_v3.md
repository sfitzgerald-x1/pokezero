# Test-time search: working plan v3 — the engine swap

Status: 2026-07-18. Successor to `test_time_search_plan_v2.md`. v2 is left
untouched (it is actively read and written by in-flight agents finishing the
W5 Tier-1/Tier-2 validation recorded there); this doc is the fresh append
point and the working plan going forward. The measurement doctrine carries
forward unchanged: **paired seeds, both arms, the compare harness
(`scripts/compare_root_puct_vs_foulplay.py`), provenance logged but never
gating.**

## Where v2 ended (carried results)

- **Search works: paired +10.0 pts vs FoulPlay** (43.0% vs 33.0%, 200 seeds,
  McNemar p≈0.02) at root-PUCT-120 with the 1M v2.2 checkpoint and frozen
  isotonic leaf. Search never loses to its own prior.
- **W5 Tier 1 (snapshot-per-decision) landed and default-on**, probe-verified:
  `prefix_replay_count = 0`, world materialization ~5% of search wall, ~8s
  per decision at extra-120 (vs 19.2s baseline). Fallback trajectory:
  19.4% → ~15.3% after the W1 fix wave → 8.2% in the Tier-1 probe (small n).
- **W5 Tier 2 on the Node sim (direct materialization) is implemented**
  (`prepare_direct_materialization_prefix`, `materialize_public_world` —
  fail-closed on unsupported public effects) with validation in progress on
  the ~10-game telemetry probe. Whatever it delivers, it inherits the Node
  sim's per-step cost.
- **The remaining cost is not the sim mechanics but the loop around it:**
  ~72% of search wall is per-visit orchestration (bridge round-trips, Python
  loop). Instrumentation to attribute it landed; a guarded one-round-trip
  snapshot candidate is staged behind the probe's verdict.
- **Search cost is scale-independent** (~5–6.5 visits/sec at 50M and 200M on
  GPU) — the model is not the bottleneck at any scale we run.

## The ceiling, measured (2026-07-18 profile)

FoulPlay's engine (poke-engine, the Rust reimplementation of the showdown
engine, vendored via `third_party/foul-play`, gen3 feature build):

- **~0.86M MCTS iterations/sec midgame, ~1.2M/sec lategame, single thread**
  (~1.2µs per iteration all-in; sub-microsecond state transitions in-loop).
  FoulPlay's whole design rests on this: ~190k simulations per decision at
  its 100ms operating point — against our ~130 visits at ~8s.
- A single FFI-crossed call into that same engine costs ~20µs — 20–100× the
  in-loop per-node budget. Conclusion: no bridge- or IPC-based loop can ever
  approach this regime; the loop must be in-process with batched boundary
  crossings.
- `generate_instructions` returns exact chance distributions per joint
  action — strictly better chance-node handling than sampling outcomes
  through a stochastic sim.

**The plan: adopt poke-engine as the search branch simulator** and remove
the real sim from the search loop entirely. The real sim stays ground truth
everywhere else — training, evaluation games, the paired harness. Landing
zone: sim cost vanishes; the floor becomes batched NN leaf evaluation
(~10⁴+ evals/sec on GPU for the s-model) → **10³–10⁴ visits/sec vs today's
~16**. FoulPlay-class volume with a learned prior and value function; makes
multi-ply meaningful; extends the budget→value curve two decades right.

**Not chosen:** a Python engine reimplementation (~100µs–1ms/step — 10–100×
over the bridge but 100–1000× short of Rust, GIL-bound, and we own every
mechanics bug) and a from-scratch Rust engine (months to re-arrive at
poke-engine, which is battle-hardened by years of ladder play). poke-engine
is an approximation of the real engine — that risk is measured (track C
below), not inherited as an assumption.

## Contracts (freeze before any track forks)

1. **World contract.** Exactly which fields a belief-sampled world supplies
   to the poke-engine `State` constructor: teams, HP, statuses, boosts, PP,
   items/abilities as sampled, side conditions, field, trick-state flags.
   Worlds are constructed ONLY from public information plus belief-sampled
   hypotheses — never from the live battle's hidden state. The P-1
   anti-leakage checksum gate must pass unchanged.
2. **Encoder contract (revised 2026-07-18, owner-aligned).** The unit of
   correctness is the incremental **fold-state advance**, not a from-scratch
   encode: `advance(fold_state, events) -> (fold_state', transition_tokens')`
   where the fold state is a first-class serializable struct carrying
   everything cumulative the production encoder holds across decisions —
   the transition-token buffer AND the raw tendency counters (counts, not
   the normalized values the observation exposes), plus any other running
   state. Rationale: at search time the root's tokens already exist (real
   boundary); every branch shares the root prefix and appends only its own
   simulated events. IMPORTANT scoping correction (review-verified):
   production today RECOMPUTES the full fold on every observe — both the
   online client and the self-play observe path call
   `extract_transitions_and_tendencies` over the whole public log; no
   persistent transition/tendency fold state exists anywhere. The fold
   state must therefore be BUILT (refactor `_fold_replay` into an
   incremental accumulator) and its validity rests on the fold being
   deterministic and PREFIX-CLOSED — an unproven property the prerequisite
   probe below must establish before the schema lands. Known closure
   risks the probe must clear: `_flag_pursuit_intercepts` is a global
   post-pass over all windows, and `opportunity_turns` is a whole-game
   set. The refactor + closure proof are the two hardest parts of track B.
   NO FREEZING of history-derived columns at search leaves:
   a leaf whose transition tokens show simulated turns that its tendency
   columns ignore is internally inconsistent — an out-of-distribution input
   the model never saw in training (owner decision; early-game tendency
   denominators are small enough that even 3 turns shift them materially).
   Boundary-state tokens (0-22) keep the original bit-identity contract —
   already met (PR #710). Identity is defined by the golden corpus, not by
   code review.

## Tracks (parallel; only meet at integration)

| Track | Deliverable | Depends on | Owns |
|---|---|---|---|
| A | World constructor: belief world → poke-engine `State` (adapt foul-play's `battler_to_poke_engine_side` mapping to source from our belief engine) + anti-leakage tests | World contract | new module only |
| B | v2.2 encoder from engine state + instructions, developed against the golden corpus | Encoder contract | new module only |
| C | Fidelity differential harness: a few thousand (state, joint-action) cases stepped in both engines, outcome distributions compared | nothing | new script only |
| D | Batched NN leaf evaluation in the search loop (pays off on the Tier-1 path today; required to cash in visit volume later) | the v2 residual-probe verdict | `search.py` (sole owner) |

File-ownership rule: track D is the only track that touches `search.py`;
A/B/C land as new modules with their own tests. This is what makes
concurrent agents safe. Track D additionally waits for the in-flight v2
residual-attribution probe so batching targets the measured bucket.

Track C reports first by design — it is the go/no-go gate on poke-engine's
gen3 accuracy. If it fails badly, A and B pivot before they are deep. (One
gen3 bug is already known and patched locally: Rest/Sleep Talk PP
underflow.)

**Status ledger (2026-07-18 EOD):** A COMPLETE (world constructor + the
edge-case waves — Transform, Shedinja HP, recharge, Trick, Truant, Encore,
Baton Pass boundary; engine-search fallback 0.0% with three-tier loud
alerting; see docs/belief_edge_case_matrix.md). C COMPLETE for waves 1-2
(one-turn 15/15 on the patched build; multi-turn 6/6; tier-2 real-game
sweep still owes the per-source matcher). B: boundary tokens 0-22 bit-exact
in Rust (PR #710); fold-state advance built + closure-proven
(`transitions_fold.py`, PR #718); schema v2 COMPLETE — corpus regenerated
with per-row fold state + event slices + overlays, row-pair advance
validation green over every boundary of the random battery AND the full
scenario suite (`scripts/validate_corpus_v2.py`, backend seam ready for the
Rust advance; see docs/golden_corpus_notes.md "Corpus v2"); **Rust advance()
VALIDATED** — `pokezero_search.FoldState` (rust/pokezero-search `src/fold.rs`)
passes the row-pair harness byte-exact over ALL boundaries of both corpora
(golden-v2 1028/1028, golden-v2-scenarios 290/290; state + products; the
`--backend compare-backends` rust-vs-python diff shows zero divergences), with
the committed-sample chain test as the permanent no-Showdown gate; per-boundary
clone+advance is ~9.8µs vs the Python reference's ~92µs (~9x; see
docs/golden_corpus_notes.md "Rust backend"); **instruction→event mapping
LANDED** (`rust/pokezero-search/src/events.rs` + `branch_events` PyO3
surface): per-outcome engine instruction lists render as protocol lines via
engine re-generation phase segmentation, real-game fidelity gate over BOTH
corpora (scripts/fidelity_gate_events.py: scenarios 77a/87b/19c of 183
driven, random 378a/291b/106c of 775; every class-c case attributed —
engine-merged no-op branches, KO-capped crit labels, and three ENGINE-model
deviations incl. fixed-damage-through-Protect; see
docs/crate_search_design.md "Instruction→event mapping"); the leaf pricing
seam now carries the branch context (`BranchSeam`) at the
`multiply_batched_core` row write and the end-to-end leaf flow (root fold →
branch events → Rust advance → per-outcome products) is gated by
tests/test_instruction_event_mapping.py; **capstone integration LANDED
(2026-07-19)** — the in-crate encoder consumes fold products natively
(tokens 23-150 + tendency/stats/pinned cells; full observation surface
byte-exact over ALL 1318 corpus rows, `--backend rust-fold`), the leaf path
builds real per-outcome observations (engine-state recompute +
world-constant carry + per-branch fold advance chained through the tree),
the ROOT-PARITY GATE passes 1015/1015 + 235/235 driven rows byte-exact at
depth 0 (`scripts/leaf_root_parity.py`), and `search_batched_multi_encoded`
runs the full model-priced loop at ~115-190µs real-observation overhead per
eval (see docs/leaf_observation_column_map.md — the column contract, /100
resolution, bench, and the remaining-to-paired-read list).
D: crate model integration
LANDED (tch-rs behind the `model` feature, TorchScriptLeafEval, virtual-loss
batched leaf eval, bit-exact parity gate, CPU+MPS benches — see
docs/crate_model_integration.md); encoder hand-off (track B) LANDED via the
leaf path above; **INTEGRATION LANDED (2026-07-19)** — self-side model
priors in PUCT selection (mapping asserted against recorded request masks:
1015 + 235 driven corpus rows exact, `scripts/prior_mapping_assert.py`;
priors reweight exploration never values, cargo-tested), the live
incremental root fold in `EngineMctsPolicy` (per-battle
`transitions_fold.FoldState`, advanced over each decision's new lines;
batch-refold cross-check flag stayed 0-mismatch), and the config-gated full
pipeline `EngineMctsConfig(leaf_eval="model")` — per belief world
`search_batched_multi_encoded` with the real fold + TorchScript eval, root
aggregation across worlds, the full fallback taxonomy live. 15-game bench
(seeds 7000-7014, worlds=4 × sims=64 × batch=8 × depth=2, CPU, the real
emeta-v2-2-lr3m-belief FINAL checkpoint artifact — throughput/mechanics
only, NO strength claim): 334/394 decisions searched, 2.92 s wall per
decision (3.42 s search wall per searched decision, ~216k model evals,
lossy renders 0.12%), prior fallbacks 0, unmapped choices 0, live-fold
cross-check 394/394 clean. Fallback rate 15.2% — every one
`no_worlds_constructed` from the PRE-EXISTING fail-closed world walls
(per-battle composition, corrected from the logs in
docs/belief_edge_case_matrix.md: a genuine Trick swap 48/60 on 7013,
request-state flags 7/60 on 7010, flashfire 5/60 on 7005/7014); the
PAIRED hp_fraction CONTROL on the same seeds reads
15.9% with the same reasons on the same battles, so the new pipeline's own
taxonomy (crate_search / root_inputs / live_fold / choices_unmapped) is
EMPTY and the historical 0.0% simply does not reproduce on today's
build/seeds in either arm — a world-construction finding, not a search
regression (follow-up below). Argmax sanity vs hp_fraction: 10 decisions,
6/10 agree (agreement not expected — different leaf evals), 0 illegal.
Remaining = the 200-seed paired read + follow-ups
(opponent priors spec'd in docs/crate_search_design.md; Tier-2 overlay at
live boundaries; batch/virtual-loss re-pricing under real observation
costs — see docs/leaf_observation_column_map.md "Remaining"; the
determinized-world fallback walls below).
**Knock-Off removal recovery LANDED (2026-07-19 EOD)** — belief_view
gained the removal/swap distinction the blocked-slot comment enumerated:
`item_removed` marks a mon whose held item was publicly STRIPPED (Knock
Off, or an item-taking Trick that returned nothing) — publicly ITEMLESS
is exactly representable, so engine_search routes those to a
`removed_item_species` signal and engine_world clears the sampled item;
a live swap (the holder carries an item that is not the sampled
assignment) stays fail-closed. Same-seed paired re-run on the landed
code: model arm 334/394 searched, 15.2%, per-battle composition
identical to the corrected attribution (7013 Trick 48/60 BY DESIGN,
7010 request flags 7/60, 7005/7014 flashfire 5/60), fold cross-check
394/394, prior fallbacks 0; hp control 69/462 = 14.9%. NO rate change
on this band — as the corrected attribution predicted — and the new
`removed_item_decisions` telemetry (bench reports now carry per-game
fallback/world-failure deltas) reads 0 across all 15 games and across a
110-game hp-mode scan of seeds 7015-7124 (walls there: request-state,
Transform-Ditto on 7059/7073/7118, flashfire, and singleton
attract/destinybond/confusion volatiles — background rates 3.4%/7.0%
per 50/60-game band; organic Knock-Off walls need OUR policy to click
Knock Off, hence rare).
The removal path itself is proven end-to-end: a live-sim test
(`KnockOffRemovalLiveTests`: real protocol → belief flags → signals →
constructed world with the item cleared) and a directed paired repro on
the same post-Knock-Off state — pre-fix 0 searched / 1 fallback
(public_effect_blocked ×8 world attempts), landed 1 searched / 0
fallbacks / removed_item_decisions=1. Wall leverage, re-ranked from the
measured composition: (1) Trick-swap current-item override — post-swap
the CURRENT item is publicly revealed by the |-item| line, so worlds
could substitute it instead of failing closed (all 48/60 of 7013);
(2) request-state flags (7010); (3) seeding the publicly-derivable
flashfire/confusion volatiles (same shape as Truant/MUSTRECHARGE).
Speed POC complete; scenario corpus suite complete.
Multi-ply decision/chance tree per the search-tree contract LANDED in the
crate (exact-expectation backup, plies-1-2 damage branching + deep
KO-threshold splits, batched-through-chance virtual loss — see
docs/crate_search_design.md).

## The golden corpus (track B's definition of done)

The network only understands positions in the exact encoding it was trained
on, and "exact" cannot be verified by reading code — only against a
reference. The corpus is that reference: a few thousand decision points from
real games storing (a) the position in poke-engine representation (input to
the new encoder) and (b) the observation tensor the production encoder
emitted for that position (golden output). Definition of done (split per
the schema-v2 decision below): BOUNDARY cells (tokens 0-22) — every stored
tensor reproduced bit-for-bit from the single-row surface (met, PR #710);
HISTORY cells (transition tokens 23-150, tendency aggregates) — the
fold-state ADVANCE check passes row-pair by row-pair (single-row
reproduction is provably impossible for these; PR #710's phase-1 finding).

It guards against the failure mode that never crashes: encoding drift.
A transition token ordered differently, an HP fraction scaled off a
different base, a status categorical resolved through the wrong vocab
priority — every one produces a valid-looking tensor, no fallback fires,
and search strength quietly sags while the symptom points at the search
logic. (Precedents: the `include_turn_merged` capture-flag miss — the loud
version; the pipeline parity-lineage bug — the quiet version.)

Three roles for one artifact: definition of done for the encoder agent, a
binary ship gate for integration, and a permanent regression net for future
vocab/schema changes.

**Exemption rule:** fields the protocol stream knows that a reconstructed
engine state cannot (or vice versa) may be legitimately unequal. Each such
field gets an explicit documented exemption with a justification — never a
global loosening of the comparison. The exemption list IS the enumerated
residual domain-shift risk being accepted.

Corpus generation does not need track A: convert positions from real game
transcripts via foul-play's existing helpers.

**Schema v2 (decided 2026-07-18):** v1 rows cannot validate history-derived
content (transition tokens 23-150, tendency aggregates) — the stored surface
is a boundary snapshot, and even the production encoder cannot reproduce
those cells from it (PR #710's phase-1 finding). v2 adds, per row: the
exported **encoder fold state** at the previous same-seat decision plus the
**inter-decision event slice** (public events since that decision; filter
`|t:|` wall-clock lines for byte-determinism). The validation contract
becomes the advance operation itself, checked row-pair by row-pair —
exactly the operation search executes. This supersedes the earlier
full-stream + prefix-index proposal, which would have validated an
operation nothing runs. Prerequisite probe: confirm the production fold is
(state + slice)-closed and the state cleanly exportable from the parser.

## Validation gates (right-sized per the 2026-07-17 owner directive)

Minutes, not hours, per track: A = constructor unit tests + P-1 checksum;
B = golden corpus pass; C = differential harness summary; D = a 1–2 game
mechanics smoke with batching on. The expensive read runs once: after
integration, a single 200-seed paired FoulPlay comparison (both arms,
shared seeds, the standard harness). Success is more winrate at fixed
wall-clock, not more visits.

## POC checkpoint (2026-07-18)

The speed target is demonstrated end to end:
[`engine_search_poc.md`](engine_search_poc.md) — poke-engine MCTS over
belief-sampled worlds as a standard rollout policy, ~475k simulations per
searched decision at 0.44s (FoulPlay-class throughput, ~18× faster than
Tier-1 root-PUCT), with tradeoffs enumerated. Two findings feed back into
the plan: the belief sampler's deterministic dead-ends cap any determinized
search at ~45% of decisions searched (highest-leverage strength lever,
upstream of both search stacks), and speed remains decoupled from strength
until track B puts the learned model on this path.

## Integration endgame: the native search crate (2026-07-18)

The model-cost ladder settles the architecture question. A Python↔engine
crossing costs ~25µs while a CPU model forward costs ~3,200µs — language is
irrelevant today — but at the batched-GPU regime track B targets
(~100µs/leaf), Python orchestration becomes a 25–50% tax, and poke-engine's
built-in MCTS has no leaf-eval hook, so a custom search loop is required
regardless. The endgame is therefore NOT an upstream fork but our own
`pokezero-search` Rust crate (PyO3 extension) that:

- depends on poke-engine as a Cargo dependency with our gen3 patches
  applied via `[patch]` (the residual-order patch already establishes the
  vendored-patch mechanism);
- owns the PUCT tree, in-tree leaf batching, and native model inference
  (TorchScript via tch-rs or ONNX Runtime; the model is a plain
  transformer encoder and exports cleanly; fp16/int8 buys 2–4×);
- implements the v2.2 encoder ONCE, in Rust, exposed to Python via PyO3 so
  the golden corpus validates it (boundary cells bit-exactly per row; history cells via the advance check) — this becomes track B's
  deliverable, replacing a Python encoder that would need a Rust rewrite.

**Search-tree contract (owner-aligned 2026-07-18).** The value head
outputs win probability, so by the law of total expectation the optimal
policy maximizes plain expected value — no risk adjustment is ever correct
on top of it. Variance is handled structurally, in three places:

1. **Chance nodes are explicit and exact.** `generate_instructions` returns
   the enumerated branch distribution with exact probabilities (typically
   a handful of branches per joint action; speed-tie x crit x secondary
   tails can exceed that - unmeasured, hence the cutoff below). Decision nodes run PUCT over our actions; each
   joint-action edge resolves by exact expectation over the enumerated
   branches, not sampling — strictly lower estimator variance at equal
   budget on small supports. Sampling only past a depth/branch-product
   cutoff.
2. **Per-outcome fold-state advance.** Each chance-child advances its OWN
   copy of the fold state with that branch's events (the crit branch's
   history shows the crit). Shared/frozen history across outcomes is the
   same internal inconsistency rejected above.
3. **Lossy spots, mitigated:** the engine collapses damage rolls to a
   representative per branch — enable damage branching at plies 1-2
   (matching the engine's own MCTS policy) and split explicitly when the
   exact roll list straddles a KO/berry/Substitute threshold. Epistemic
   variance (hidden info -> belief worlds) stays a SEPARATE axis,
   aggregated at the root (likelihood-weighted eventually); the leaf value
   head absorbs all variance beyond the horizon — the thing it was trained
   on.

Interim (no new machinery): the Python engine loop + shared GPU inference
service clears >10³ model-priced evals/sec for throughput work (self-play
collection batches across 64–128 games), with per-leaf Python encoding as
the known next wall. Depth at the endgame: 10–30k model-priced visits/sec
supports PV depths of 4–6 turns, more with hybrid pricing (model at shallow
nodes, the fidelity-validated handcrafted eval below).

## Integration (serial, single owner)

After A+B land and C passes: swap the branch simulator behind the existing
search-policy interface, leaf evaluation batched via D. Then the one paired
read. If the corpus and fidelity gates pass, everything downstream — leaf
calibration, the W2 budget→value curve, W3 frontier reads — transfers
unchanged and re-runs cheaply on the fast path.

Carried from v2 unchanged: the W2 value curve and W3 frontier paired reads
remain wanted and become near-free after the swap; the owner criteria for
any binding claim (+3 md / +5 fp, paired CI > 0, never losing to the prior)
still stand. Deferred until after the swap: multi-ply, tree reuse, early
termination — all re-priced on the fast path.
