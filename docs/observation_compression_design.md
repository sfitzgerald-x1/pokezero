# Observation compression: opponent-signal stats replace the 4-turn window

Status: design, 2026-07-03. Motivated by the width result (512d arm leading
its cohort on trajectory: 74.4% max-damage at 208k games) and the cost
structure that blocks scaling width further. Companion to
[`wse_shaping_and_coverage_design.md`](wse_shaping_and_coverage_design.md)
(separate arm; this doc changes the *observation*, not the reward) and
[`research_synthesis.md`](research_synthesis.md) (T2/T6).

## Motivation

The encoder consumes `window_size=4` complete observation copies
(`neural_policy.py`): each turn is ~46 tokens (1 field + 6 self-mon +
6 opponent-mon + 9 action candidates + **24 recent-event tokens**), so
~3/4 of the sequence is stale board state — and the 24 event tokens per
turn already duplicate most of what the history window provides, paying
twice for temporal context. Cutting to `window_size=1` plus a compact
opponent-signal block frees roughly 4× the sequence compute. That saving
is the feasibility budget for width: the point is not a smaller model, it
is **the same FLOPs spent on a wider trunk** (512d-class and beyond),
which current evidence says is the productive axis.

The bet: within-game temporal context in this game is compressible into
sufficient-statistic-style summaries (the poker-HUD precedent — aggregate
frequency stats as the standard compression of action history), and the
capacity freed is worth more than raw history. This is a hypothesis; the
ablation section makes it falsifiable.

## Design principle: evidence mass, not rates

Every tendency feature is a **(count, opportunity) pair**, never a bare
rate. A 100% switch-out rate over one observation and over eight are
different evidence; the net can compute the ratio but cannot recover the
mass. All features are player-relative and computed from public
information only (events, revealed moves, belief candidate sets, damage
estimates) — no oracle leakage.

## v1 feature set

1. **Exact opponent PP ledger (belief engine).** For every revealed
   opponent move: remaining-PP fraction, maintained exactly — max PP from
   the randbats catalog, decrement per observed use, ×2 when our active
   mon has Pressure (our own ability is perfectly known, so the ledger is
   exact, not approximate). Exposed as a numeric feature on the opponent
   move slots, mirroring our own `NUMERIC_MOVE_PP_FRACTION`. Motivation:
   pp-stall is a real strategy in both directions; managing opponent PP
   (Pressure stalling, counting boosted-move uses) is decision-relevant
   signal the net currently cannot see at all.
2. **Global switch tendency:** (opponent switch count, opponent decision
   opportunities).
3. **Per-opponent-mon tendency triple** on the existing 6 opponent-mon
   tokens: (switched-out-before-attacking count, stayed-and-attacked
   count, turns-active). Deliberately *not* the full 6×6
   matchup-conditional matrix — sparse deadweight at battle length.
4. **Opponent weather/field reveals:** per weather type, one flag —
   "opponent has set this weather this game" (synergy reveal about their
   remaining team). Own-side weather usage is *not* tracked: the field
   token already carries active field state, and our own past usage
   changes no decision.
5. **Midground/prediction signal**, over turns where *we* switched:
   (my-switch-turn count, opponent-move-better-vs-incoming count
   [margin-thresholded damage-calc comparison, to exclude
   good-vs-both moves], pivot-move count). High ratio = opponent predicts
   switches / midgrounds; low = they tunnel the active mon. Noisy by
   nature; the (count, rate) form lets the net weigh confidence.
6. **Opponent setup usage** (optional, cheap): per-mon "has used a setup
   move" bit. Partially redundant with revealed-move features ("has used"
   vs "has"); include only if the token budget is comfortable.

**Descoped — per-candidate-move declined-opportunity counters.** The
"they didn't click the KO move they might have" evidence is real, but
representing it requires per-candidate-move accumulators across the whole
candidate set — expanding an already expensive feature family that is
often deadweight on the net. Revisit only if the v1 stats show value and
a later probe shows the net misreading stay-in bluffs; the principled
version (behavior-likelihood belief updates) is ReBeL-direction work, not
an observation feature.

## Encoding

One new **stats token** (or two) per observation carrying the global
pairs (items 2, 4, 5), plus per-mon fields added to the existing opponent
tokens (items 1, 3, 6). New observation spec version with checkpoint
compatibility guards per `model_versioning.md`; `window_size=1` becomes a
config choice, not a new code path (the window machinery already
supports it).

## Interplay notes (both load-bearing)

- **These features are near-constant in mirror self-play** — against a
  copy of yourself, tendency stats carry almost no signal. They become
  informative exactly when opponents vary. Ship alongside the cross-arm
  opponent pools (#487 arms as each other's population); the pools make
  the features informative and the features let the net exploit what the
  pools expose.
- **The same statistics are the self-derived z-descriptor vocabulary**
  (switch rate, weather/setup usage, midground rate) for the
  conditioning mechanism in `research_synthesis.md` stack item 3. One
  instrumentation effort, two consumers.

## Ablation plan (500k, same seed bands and yardstick as the width arms)

| arm | window | stats block | width |
|---|---|---|---|
| A (control) | 4 | no | 256d |
| B | 1 | yes | 256d |
| C (payoff) | 1 | yes | 512d |
| D (optional) | 1, GRU aggregator | no | 256d |

Reads: foul-play at matched milestones (primary), max-damage trajectory,
ΔV/behavior probes, and wall-clock/games-per-hour (the feasibility claim
is itself a measurable). Success for the compression hypothesis: B ≥ A on
foul-play at matched milestones with the expected sequence-compute
saving. Success for the program: C > both on trajectory at equal
wall-clock. D distinguishes "engineered stats" from "any temporal
compression."

## Explicit non-goals

- No reward changes (that is #487's arm).
- No cross-battle opponent memory — all counters reset per battle
  (randbats opponents are exchangeable across battles).
- No behavior-likelihood belief updates (descoped above).
