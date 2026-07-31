# Rest Residual-Composition Comparison Limit

## Retained Identities

This closure covers only `2901076/41`, `3000156/47`, and `3500842/79`.

The earlier prediction described them as Rest turns with an apparent missing
Leftovers or Toxic tail. That description is historical context, not replayed
evidence: the raw retained reports are not tracked in this worktree or its
reachable local Git history. The recorded per-row negative evidence names the
searched scopes and references without exposing any private source path.

Attempts to materialize the rows also failed closed:

- The local `scripts/engine_transition_differential.py` entrypoint cannot load
  because the current Python environment has no importable `poke_engine` wheel;
  the harness therefore never reaches its downstream `pokezero_search` import,
  which is also absent as a standalone consumer. A full two-consumer build is
  required for a differential run.
- The opaque public Gen 3 Random Battle candidate label for each retained game
  identity returned HTTP 404. The machine artifact records the labels and
  statuses, not public paths.

A 404 does not prove that no original archive exists. It proves that this lane
cannot substitute a guessed public URL for the retained input.

## Executable Boundary

`rust/pokezero-search/tests/gen3_rest_residual_composition.rs` passes three
native, engine-only controls against the patched engine:

- A fixed nonterminal Seismic Toss hit is followed by Rest's full heal and the
  opposing survivor's Leftovers tail.
- The same survivor composition orders that Pokemon's Leftovers tick before its
  stage-two Toxic tick.
- A fixed terminal hit faints the final opposing Pokemon before Rest, after
  which neither Rest nor any Leftovers, Toxic, or Toxic-counter tail is emitted.

The native controls are not their own oracle. The matching Showdown-side
`restresidualtail` scenario in `scripts/gen3_switch_differential.py` independently
observes a 100 HP Seismic Toss, Rest from `361/461` to `461/461`, then Aipom's
`251` max-HP tail in the order Leftovers `+15` before stage-two Toxic `-30`
(`236 -> 251 -> 221`). This is the Gen 3 residual rule used by the sibling
standard: `Dex.mod('gen3')` resolves items at residual suborder 10.4 and status
damage at 10.6, while both magnitudes are `floor(maxhp / 16)` times the Toxic
stage where applicable. The scenario reads real Showdown protocol output;
the Rust controls only verify that the patched engine agrees for constructed
states.

These controls satisfy neither trigger in the existing stop rule. They show
the scheduler behavior for the constructed states only; they do not identify
the unreplayable retained rows' damage roll, world construction, or event
matching outcome.

The two-consumer engine build fingerprint is also not attested: the current
Python environment has neither importable consumer. Establishing that stamp
requires a full wheel and crate rebuild/write that records the same
consumer-visible fingerprint in both artifacts; merely installing
`pokezero_search` cannot create or verify it. The native terminal-residual
ablation is an additional focused control, not certification-quality replay
evidence.

## Final Disposition

**Comparison limit. No production change is licensed.**

There is no direct evidence of a residual scheduler defect, and there is no
direct evidence of a matcher or damage-composition defect for these identities.
The residual scheduler must not be changed to make the rows appear matched.
Likewise, the focused controls must not be laundered into an attribution to a
damage lattice or matcher mechanism.

To reopen this lane, supply immutable raw reports for all claimed rows, their
source and content hashes, the exact differential command, and a build with a
verified two-consumer engine fingerprint. A rerun must first establish the observed
row-level event tail, then compare it with every native branch. Only a
surviving branch missing an eligible residual instruction, or a terminal branch
emitting one, licenses a scheduler investigation.
