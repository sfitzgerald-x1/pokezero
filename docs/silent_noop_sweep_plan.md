# Silent no-op sweep — plan (v3 schema-freeze gate)

Status: 2026-07-20, owner-directed. Companion to `observation_v3_spec.md`.
This sweep is the gate before the v3 observation schema is declared FROZEN
and the Rust mirror + corpus regeneration begin.

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
  and byte-prefix tests as #779. All additions ride schema v3, pre-freeze
  only; after freeze, changes wait for v4.
- Validation is local and minutes-scale: static inventory + targeted unit
  probes with scripted logs; no cluster jobs.
- Coordination: the Rust fold mirror implements against the FROZEN schema;
  regenerating the golden corpus at v3 happens once, after the freeze
  (spec §Coordination). The sweep must land its verdict table before that.
- Encoder edits follow the #779 review bar: independent review before merge;
  the fail-mode being guarded against is silent training-data corruption.

## Execution shape

One agent, ~half a day: inventory + diff are mechanical; the conflation
tests need the vendored engine open. Output is the findings doc and (if any
ADDs survive) one implementation PR structured exactly like #779
(spec-first, both fold paths, byte-identity tests, review).
