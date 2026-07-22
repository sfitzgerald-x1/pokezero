# Silent no-op sweep — plan (v3 schema-freeze gate)

Status: 2026-07-20, owner-directed. Companion to `observation_v3_spec.md`.
This sweep is the gate before the v3 observation schema is declared FROZEN
and the Rust mirror + corpus regeneration begin.

The final execution contract, including the token-format-refactor prerequisite,
source-identity rules, lane outputs, and completion criteria, is recorded in
[`v3_end_of_cycle_evaluation_plan.md`](v3_end_of_cycle_evaluation_plan.md).

## Principle (the meta-learning of the Toxic incident)

**Any engine outcome that renders as a silent no-op in the encoding — or
renders identically to a strategically distinct outcome — will eventually
produce pathological behavior.** The `-fail` gap was found reactively: a
behavioral pathology (deterministic Toxic re-clicking) forced a probe, which
found the encoder blind spot. This sweep is the proactive version: enumerate
every blind spot once, adjudicate each deliberately, and freeze the schema
with the accepted losses *written down* instead of latent.

## Method

1. **Emission inventory.** Enumerate every protocol line type the gen3 sim
   can emit: grep `this.add(` across the vendored showdown (`sim/`,
   `data/moves.ts`, `data/abilities.ts`, `data/items.ts`, gen3 mods),
   cross-checked against `sim/SIM-PROTOCOL.md`. Filter to lines REACHABLE in
   gen3randombattle: gen3 `Standard` ruleset + the actual generator pool
   (`data/random-battles/gen3/sets.json` — species, moves, abilities, items).
   Reachability evidence is mandatory per event (this repo has prior wins
   from checking: no screens in the pool; no Freeze Clause in the ruleset;
   all six Natural Cure species are mono-ability).
2. **Handled-set diff.** Extract the event dispatch tables actually consumed
   by `transitions.py`, `showdown.py` (`_ReplayParser` and public-state
   update helpers), and the belief engine. The unhandled remainder is the
   candidate list.
3. **Conflation test, per unhandled event:** name two strategically distinct
   situations this line distinguishes; check whether the current encoding
   distinguishes them by ANY other means (status tokens, volatiles, side
   conditions, transition outcomes, belief columns). Only a real conflation
   is a finding.
4. **Verdict per event**, one of:
   - `ADD` — proposed v3 column/bit (numeric bits preferred; no vocab sprawl)
   - `COVERED` — cite the exact covering feature/column
   - `UNREACHABLE` — cite ruleset/pool evidence
   - `ACCEPTED-LOSS` — conflation exists but marginal; rationale recorded
5. **Deliverable:** `docs/silent_noop_sweep_findings.md` — one table
   (event | reachable | encoded-where | conflation risk | verdict |
   evidence), plus a single batched ADD proposal for owner sign-off.
   **One batch decision, then the schema freezes.**

## Layered omission-detection program

An encoder-consistency oracle cannot discover a fact that every encoder path
omits: it can only prove that the encoder agrees with itself. Omission
coverage therefore needs independent references which know more than the
encoded observation. The four layers below are cumulative; a clean result in
an earlier layer is not a waiver for a later one.

### Layer 1: static emission inventory

The emission inventory and handled-set diff above are the first layer. They
are complete over **reachable top-level protocol event types**, and must be
extended to canonical event signatures that include meaningful subtypes and
arguments, not just a tag. For example, `-activate` identifiers and `cant`
reasons are distinct signatures. This layer is authoritative for what the
engine can emit, but cannot prove that a handler preserves an event's
semantics, nor see engine state changes with no protocol line.

### Layer 2: census differential

Use the existing protocol census machinery as a risk-weighted differential,
with canonical signatures rather than only top-level event tags. Collect:

- `E`: reachable engine-emittable signatures from Layer 1.
- `O`: signatures observed in bounded-depth fixtures, curated interaction
  fixtures, and existing production self-play trajectory logs. Each observed
  signature records its occurrence count and corpus provenance.
- `C`: signatures that the observation, replay, and belief dispatch paths
  actually consume.

Report these cells explicitly:

- `O - C`: observed-but-unconsumed candidates, ordered by real frequency.
- `E - O`: emittable-but-unobserved signatures, which need a targeted fixture
  or a reachability pruning decision.
- `C - E`: stale, dead, or incorrectly classified handlers.

This is a prioritization and fixture-planning tool, not evidence that an
observed signature is semantically represented. A consumed line can still
have an omitted subtype or argument.

### Layer 3: encoding-collision audit

Build a dedicated collision audit over approximately 100,000 decision states
from existing trajectory captures. For each state, hash exactly the
model-visible input arrays and masks: categorical ids, numeric values, token
types, attention structure, and legal-action mask. Do not include debug
metadata, raw protocol strings, timestamps, player names, or private request
state in this hash.

Within each byte-identical input group, compare canonical public-state and
public-log fingerprints. Fingerprints must be perspective-stable and scoped
to the same decision kind, so seat orientation or action type cannot create a
spurious collision. They should be parsed public facts plus a canonical
history signature, not a raw-line comparison subject to formatting changes.

Groups with distinct public fingerprints are concrete conflation candidates.
The 100k-state first pass writes a compact public collision sketch (input hash,
public fingerprint, and deterministic replay locator), not repeated model
tensors or complete public-state payloads. This keeps broad capture
storage-bounded. Candidate locators are then replayed to hydrate the exact
public-field differences before whitelist adjudication. Filter only through a
versioned, documented whitelist of intentional abstractions, initially HP
quantization, tendency bucketing, and transition window truncation. Every
remaining hydrated group records representative state pairs, the differing
public facts, frequency, schema version, corpus provenance, and whitelist
classification. This is the primary catch-all for tag-subtype and
argument-level omissions that a dispatch-table sweep cannot find.

### Layer 4: counterfactual harm probes

Run a focused probe only for the Layer 1--3 shortlist. Construct a minimal
public-state pair differing solely in the candidate fact, verify that the
trained policy receives identical inputs (and therefore identical logits),
then compare the simulator's action consequences or outcome distributions.
The result turns a collision into an evidence-based `ADD` or
`ACCEPTED-LOSS` verdict rather than a subjective schema preference. The
Toxic no-op probe is the model for this stage; it is not a reason to generate
one probe per mechanically possible event.

### Silent engine-mutation lane

Some relevant state transitions emit no public protocol line at all. They are
outside all four log-based layers. Add a bounded, test-only simulator
instrumentation pass that records state mutations and classifies each as
protocol-backed or silent. For every reachable silent mutation, the belief
layer must either track it, infer it from public deterministic information, or
document it as provably untrackable/accepted loss. Natural Cure with
`showCure=false` is the canonical example: the public line is absent, but the
mono-ability randbat universe makes the cure inferable on switch-out.

### Execution order and evidence

1. Extend the current census to Layer 2 signatures and production-log
   frequencies while completing the static Layer 1 table.
2. Build Layer 3 before expanding a large fixture matrix; collision evidence
   is the highest-leverage new omission detector.
3. Run Layer 4 only on the resulting shortlist.
4. Run the silent engine-mutation lane as a bounded companion audit.

`docs/silent_noop_sweep_findings.md` must record, for every candidate: the
canonical signature, reachability evidence, observed count and provenance,
consuming handler, collision evidence, silent-mutation classification when
applicable, harm-probe result, and final verdict. The existing protocol
coverage matrix and validated-interaction registry are useful inputs to this
record; neither substitutes for collision or silent-mutation evidence.

## Seed candidates (adjudicate, do not assume)

- `cant` reasons: is WHY the opponent lost its turn (full para vs sleep vs
  flinch vs recharge) distinguishable in the transition tokens, or only THAT
  it lost the turn? Para-rate and wake-timing inference depend on the reason.
- Yawn pending-sleep (`-start … move: Yawn`): a public two-turn sleep
  telegraph; interacts with the new sleep-clause bits (engine blocks the
  Yawn resolution under clause → renders as our new fail path).
- Weather turns remaining: move-set weather (5 turns) vs Sand Stream
  (permanent) — is the countdown encoded or only the weather id?
- `-notarget` (move aimed at a fainted slot), `-mustrecharge` beyond the
  charging side-effect, `-singleturn` (Protect/Endure visibility),
  `-sethp` (Pain Split), Perish Song counters (verify pool reachability),
  partial-trapping residual attribution (Wrap family), `-ohko` (verify
  reachability under OHKO Clause — likely UNREACHABLE).
- `-activate` subtypes currently unconsumed (grep the dispatcher for which
  identifiers are dropped).

## Already adjudicated — do NOT re-derive

- `-fail` / `-miss`: separate flags, landed in v3 (#779).
- Sleep-clause state bits: landed in v3 (#779); Freeze Clause UNREACHABLE.
- `item_removed` (publicly itemless ≠ unknown): COVERED, belief → obs.
- Sleep turn counter: COVERED (`NUMERIC_SLEEP_TURNS`).
- Natural Cure status staleness: NOT a schema item — all six pool NC species
  are mono-ability, so switch-out cure is deterministic. Fix is belief-side
  (clear status/toxic-stage on NC switch-out, derived from set data) PLUS
  the same clearing in the sleep-clause tracker (which otherwise goes
  stale-ON when an NC sleeper switches out and the engine cures silently —
  `showCure=false` emits no protocol line). Small follow-up patch to #779;
  lifecycle tests mirror the existing clause suite.
- BP-collapse fail loss: ACCEPTED-LOSS, recorded in the v3 spec; revisit at
  the Rust-mirror milestone.

## Constraints

- **v2.2 byte-identity is absolute** for any ADD — same dual-schema gating
  and V2.2 byte-identity plus V3 permutation-map tests as #779. All additions ride schema v3, pre-freeze
  only; after freeze, changes wait for v4.
- Layers 1--2 are local and bounded. Layer 3 is a new CPU tool and may sample
  existing captures, but must emit incremental artifacts and provenance rather
  than require a new training run. Layer 4 runs only on a small shortlist;
  the silent-mutation lane is bounded test instrumentation. No cluster-scale
  collection is required for the plan.
- Coordination: the Rust fold mirror implements against the FROZEN schema;
  regenerating the golden corpus at v3 happens once, after the freeze
  (spec §Coordination). The sweep must land its verdict table before that.
- Encoder edits follow the #779 review bar: independent review before merge;
  the fail-mode being guarded against is silent training-data corruption.

## Execution shape

Layers 1--2 are mechanical inventory and census work. Layer 3 is the one
substantive new tool and should be built before broad fixture expansion.
Layers 4 and the silent-mutation lane are deliberately narrow adjudication
passes. The output is the findings doc and, if any ADDs survive, one
implementation PR structured exactly like #779 (spec-first, both fold paths,
byte-identity tests, review).

---

# Part 2 — Cross-window automata sweep (the Protect-counter class)

Sibling lane, same freeze gate. Class definition: **publicly-derivable
running state (counters, countdowns, ramps) whose value requires more
history than one observation window**, and which the tendency columns
(long-run averages) cannot represent. Every event feeding the automaton IS
encoded per-turn; what the model cannot do is integrate them across
decisions — window-size 1 means transition tokens only cover events since
the last decision.

**Exemplar (found empirically, 2026-07-20 protocol probe):** consecutive
Protect/Detect count. Gen3 halves Protect's success odds per consecutive
use (100%→50%→25%); the count is public and central to Protect/Leftovers
stall wars; the model can see "he Protected last turn" but never "this is
his third in a row."

## Enumeration method (three sources, in order of authority)

1. **Engine state structs — the ground-truth list.** The engine must track
   every cross-turn automaton to implement it. Enumerate: showdown's
   `Battle#toJSON` volatile/side/field state (every field with a
   `duration`, `counter`, `time`, or `stage` semantic, filtered to gen3),
   and cross-check poke-engine's `VolatileStatusDurations` + side/state
   counter fields. The diff (engine counter fields) − (encoded features) is
   the candidate list. This source cannot miss an automaton the game
   actually has.
2. **Dex mechanics scan.** Grep vendored `data/moves.ts` + gen3 mods for
   `stallingMove`, `durationCallback`, consecutive-use mechanics
   (Rollout/Fury Cutter ramps), and counter-bearing volatiles; filter to
   pool reachability via `data/random-battles/gen3/sets.json`.
3. **Window-horizon test, per candidate:** is the state reconstructible
   from (a) boundary tokens, (b) ONE window of transition tokens,
   (c) tendencies? Only a "no" is a finding.

## Seed candidates (adjudicate with the Part-1 verdict scheme)

| Candidate | Public automaton | Prior |
|---|---|---|
| Consecutive Protect count | success halving | ADD-leaning (found) |
| Confusion turns-so-far | 2–5 turn hazard rate | check volatile encoding |
| Encore turns-so-far | 4–8 turn window | pool-reachable, check |
| Perish count (3/2/1) | explicit `-start perish{N}` events | verify pool reachability |
| Partial-trap turns-so-far (Wrap family) | 2–5 turns | pool check |
| Move-weather countdown | 5-turn clock vs permanent Sand Stream | carried from Part 1 |
| Rollout/Fury Cutter ramp | damage doubling per consecutive hit | pool check |
| Stockpile count (1–3) | explicit stages | pool check |
| Disable/Taunt turns-so-far | timed locks | gen3 semantics + pool check |
| Opp per-move public use counts | PP depletion / Struggle proximity | check belief columns first |
| `cant` reason subtype | full-para vs sleep vs flinch distinguishability | carried from Part 1 |

Already covered (do not re-derive): sleep turns (`NUMERIC_SLEEP_TURNS`),
toxic stage, substitute HP, Spikes layers, choice-lock inference (belief
tier2 CB columns), `item_removed`.

## Encoding conventions for ADDs

Numeric scalars on the OWNING token — per-mon automata (confusion, encore,
trap, ramp, stockpile) on that mon's token, side/field automata (weather
countdown, Protect count of the active) on the field token — normalized
like `NUMERIC_SLEEP_TURNS` (`min(1, x/cap)`). No categorical vocab growth.
Every ADD rides schema v3 pre-freeze with V2.2 byte identity and the V3
permutation-map pattern,
and each gets the same interpretability property as the sleep-clause bits:
a counterfactual flag/count-flip probe is the acceptance demonstration.

## Adjudication and sequencing

Shared with Part 1: reachability evidence mandatory, frequency from the
census/self-play, strategic-value argument per ADD (does the count change
optimal play?), ONE batched owner decision covering both parts, then the
v3 freeze. The engine-struct diff (source 1) is mechanical and should run
first — it bounds the class completely; sources 2–3 only classify what it
finds.
