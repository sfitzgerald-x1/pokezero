# Deep-line encoder audit plan (~2-hour, time-boxed)

**Audience:** an executing agent (Codex) running against the pokezero checkout.
**Status:** plan for a follow-on audit after the 2026-07-20 comprehensive encoder-fix batch.

## 0. Objective

Flag inconsistencies in the **production training observation encoder** that arise from
**deeper interaction lines** — multi-turn, multi-mechanic sequences and state-accumulation
paths that a single-mechanic, incidence-driven sweep cannot reach.

This is a **read-only audit**: find and flag, do **not** patch the encoder. Each confirmed
bug becomes a separate, checkpoint-compatible fix PR that can fast-follow the training runs.
It does **not** gate the live runs.

## 1. Why "deeper lines" is the right target

Every encoder bug found and fixed so far was essentially a **single-event** mis-encoding.
The residual tail is **state-accumulation**: the encoder carries belief / fold / counter state
across a game, and a bug only surfaces after a specific *sequence*. The archetype is the toxic
bug — a badly-poisoned mon read `status:tox` but `toxic_stage=0` **only after switch-out →
switch-back**; no single event was wrong, and it explained a real behavior (the model wouldn't
pivot poisoned mons). Random games hit these by luck. This audit **forces the sequences and
asserts continuously**.

## 2. Prior art — build on these, do NOT re-report already-fixed bugs

Reuse (do not rebuild):
- `scripts/leaf_vs_reality.py` — the leaf differential harness; the **template** for the
  oracle differential (drive recorded actions, encode, diff vs the next recorded golden row).
- `scripts/validate_corpus_v2.py` — the fold **row-pair** validator (batch ≡ incremental).
- `src/pokezero/golden_corpus_scenarios.py` — the `ScenarioSpec` / `ScriptedPreferencePolicy`
  / `gen3customgame` framework for scripting deterministic interaction chains.
- `src/pokezero/randbat.py` — `Gen3RandbatSource` + the `Teams.generate('gen3randombattle')`
  path used for the 36k-real-set belief oracle.
- Docs: `docs/belief_edge_case_matrix.md`, `docs/leaf_observation_column_map.md`,
  `docs/fold_closure_probe.md`, `docs/golden_corpus_notes.md`.

Encoder surfaces:
- `src/pokezero/showdown.py` — `observation_from_player_state`, the `_ReplayParser`, per-column
  encode sites.
- `src/pokezero/belief.py` — the belief engine (candidate sets, ability/item/move reveals,
  status/counter tracking).
- `src/pokezero/transitions.py` + `transitions_fold.py` + `turn_merged.py` — the transition /
  fold / history encoder.
- `src/pokezero/observation.py` — column layout / normalization.
- `src/pokezero/local_showdown.py` — `LocalShowdownEnv`; feeds the **omniscient** stream
  (exact HP both sides) — this is the ground-truth source for the oracle.

**Already fixed — do NOT re-flag** (2026-07-20 batch): absorb ability attribution; Shed Skin
confirmation + false Early Bird; opponent item currency (Knock Off / Trick / berry-eat);
toxic ramp counter after pivot; Hidden Power self type/BP + Return/Frustration; Sleep-Talk
reveals; Struggle belief poisoning; candidate-universe over-pruning (STAB/setup); L100 level +
opponent expected-stats zeroing; Unown cosmetic-forme dex fallback.
**Known-deferred / accepted (do NOT re-flag unless the incidence is wrong):** H1 switch-token
`side_effect` mislabel (Intimidate/Sand-Stream — zero training impact, gated out of encoding,
would ripple into the Rust fold + corpus); Substitute `maxhp/4` approximation; in-branch
screen set-turn approximation; sleep-turn approximation. Report only if the impact assessment
was wrong.

## 3. The oracle

For every **public** column the encoder emits, compute what it *should* be **independently**
from Showdown's omniscient battle state (both sides' exact HP/status/boosts/items/moves/field —
available because `local_showdown` feeds the omniscient stream) and assert equality. Public
columns: both sides' active+bench HP fraction; status + status counters (toxic stage, sleep
turns); stat-stage boosts; revealed moves + PP; revealed/current items; field (hazards/screens/
weather + turns); level; species/type/base-stats; fainted flags; the legal-action mask.

Do **not** assert **belief/uncertainty** columns against omniscient truth (they are *supposed*
to be uncertain). Their oracle is the generator's true set distribution (§ Lane 5 / the 36k-set
method) — a lighter regression check that off-script stays 0.

## 4. The five lanes

### Lane 1 — Long-game oracle differential (workhorse, scaled to depth)
Drive **full** games to natural termination (not just to first decision). Assert every public
column == the oracle at **every** decision, weighting turn-20+ decisions where state has
accumulated. Target thousands of decisions across hundreds of games. Any divergence at a late
decision is an accumulation bug. (This is the leaf-vs-reality method applied to the full public
surface across game depth.)

### Lane 2 — Mechanic-chain fuzzing (the combinatorial tail)
Script interacting **pairs/triples** deliberately (don't wait for them to occur), then assert
consistency at every step of the chain:
- status × movement: toxic → switch → switch-back → (stage?); sleep → Sleep-Talk → switch → wake-timing.
- boost × transfer: setup → Baton Pass → recipient → Haze; Intimidate on repeated switch-ins.
- item × mutation: Knock Off → Trick → berry-eat chains.
- transform × identity: Ditto Transform → switch-revert → re-Transform; Transform + Choice lock.
- forme × state: Castform weather `-formechange` chains; Deoxys (separate species) cross-contamination.
- faint × replacement: Explosion double-faint → force-switch → Pursuit-on-switch → Spikes chip on entry.

### Lane 3 — Invariant / property suite at scale (auto-flagging)
Assert the "impossible observation" properties at **every** decision of every long game and
fuzzed chain — violations auto-surface accumulation bugs with no hypothesis needed. Each maps
to a bug class already seen:
- consistency: `status==tox ⟹ toxic_stage≥1`; `status==slp ⟹ sleep_turns consistent`; a fainted
  mon has `hp==0` and no active volatiles; a revealed move's PP ≤ its max.
- monotonicity: opponent uncertainty / `candidate_set_count` non-increasing as reveals accumulate
  (over-pruning made it snap **up** at full scout).
- no-placeholder: no column reads the 0/None sentinel when the underlying public datum exists
  (the HP / level / Unown class).
- bounds: `hp∈[0,1]`; stat stages `∈[−6,6]`; `toxic_stage∈[0,15]`; all normalized columns in
  their declared range; categorical ids in-vocab or the OOV sentinel.
- sum/complement: probability-like columns sum to 1; possible-set counts ≥ 1 when revealed.

### Lane 4 — Self-consistency differentials (two independent paths must agree)
- **incremental-vs-batch**: the live-accumulated belief/fold vs a full re-derivation from
  Showdown state at the same decision — extends the fold's batch≡incremental proof to the
  **whole** observation. Divergence = an incremental-path accumulation bug.
- **snapshot-vs-live**: encode from a serialized snapshot vs live state — catches state that
  isn't captured/restored.
- **perspective symmetry**: p1's view of a symmetric position vs p2's mirrored view — public
  columns must mirror.

### Lane 5 — Protocol-chain census (depth extension)
Extend the single-message-type census to message-type **co-occurrences** within a turn,
cross-referenced against what the corpus + scenarios exercise. Uncovered high-frequency
co-occurrences = where to add scenarios. **Close the known gap:** `tests/data/golden_corpus_sample`
has **no** Intimidate / Sand-Stream / Baton-Pass switch-ins, so the fold-parity gate is blind
there — quantify the blind set and add fixtures.

## 5. Orchestration & time budget (~2h wall-clock)

| Time | Phase |
|---|---|
| 0:00–0:15 | **Harness prep** — one shared engine: game driver + omniscient oracle + per-column asserter + invariant checker. Every lane reuses it. |
| 0:15–1:30 | **Parallel fan-out** — ~6–8 workers, sharded: 2× Lane 1 (game-batch shards), 2× Lane 2 (chain-family shards), 1 each Lanes 3/4/5. Read-only; findings to a shared scratchpad. |
| 1:30–1:50 | **Synthesis** — dedupe by (column, mechanic, signature); classify real-bug / accepted-approx / oracle-gap; rank by incidence; minimize repros. |
| 1:50–2:00 | **Write-up** — ranked flag list + regression stubs for confirmed flags. |

Parallelism is what makes 2h feasible — the work is embarrassingly parallel across mechanic-chains
and game shards. Deterministic fan-out + synthesis is a natural fit for a single orchestrated
run.

## 6. Deliverable

A ranked flagged-inconsistency list — each with: the triggering mechanic-chain, the divergent
column(s), oracle-vs-encoder values, a **minimal repro**, TRAINING-AFFECTING yes/no + estimated
incidence, and a classification. **Plus:** the committable differential + invariant **harness**
as a permanent gate, and regression scenarios for confirmed flags.

## 7. Honest limits

**Catches:** state-accumulation bugs, sequence-dependent mis-encodings, invariant violations,
incremental-path divergences, uncovered protocol co-occurrences.
**Won't catch:** bugs gated on rare *species/moves* absent from the sample (mitigated by Lane 2
scripting + Lane 5 census); model-side issues (this is encoder-only — value target, policy loss,
architecture are out of scope); anything the oracle itself gets wrong (mitigated by Lane 4).
**The ultimate validator** remains behavior analysis on the retrained checkpoints — back-trace
any anomaly to its encoder cause (how absorb and toxic were both found).

## 8. Guardrails

- Read-only on production encoder code; flag, don't fix.
- Each confirmed bug → a **separate** fix PR, value-only / checkpoint-compatible (no observation
  schema change), so it can fast-follow the runs without a schema break.
- Any change touching the transition/fold surface must re-check the Rust-fold parity
  (`validate_corpus_v2.py --backend rust`) and regen the golden corpus if the fold products move
  — the H1 case showed the local suite does **not** catch this (the sample corpus lacks the
  triggering fixtures).
- Commit as the user; no AI co-author trailer.
