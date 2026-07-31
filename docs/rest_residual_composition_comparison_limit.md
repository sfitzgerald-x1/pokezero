# Rest Residual-Composition Comparison Limit

## Retained Identities

This closure covers only `2901076/41`, `3000156/47`, and `3500842/79`.

The earlier prediction described them as Rest turns with an apparent missing
Leftovers or Toxic tail. That description is historical context, not replayed
evidence: the raw retained reports are not tracked in this worktree, in
`pokezero-damage-composition-tail`, or in
`pokezero-engine-damage-arithmetic-tail`. A search of all available refs found
only the prior prediction and its focused Rust fixture.

Attempts to materialize the rows also failed closed:

- The local `scripts/engine_transition_differential.py` entrypoint cannot load
  because this checkout has no installed `pokezero_search` consumer. Installing
  the patched `poke_engine` wheel alone is intentionally insufficient for a
  differential run.
- Candidate public URLs of the form
  `https://replay.pokemonshowdown.com/gen3randombattle-<game>.json` returned
  HTTP 404 for all three numeric game identities.

A 404 does not prove that no original archive exists. It proves that this lane
cannot substitute a guessed public URL for the retained input.

## Executable Boundary

`rust/pokezero-search/tests/gen3_rest_residual_composition.rs` passes two
native controls against the patched engine:

- A surviving Rest turn preserves the opposing Toxic user's Leftovers tick and
  stage-two Toxic tick.
- A terminal damage branch that faints the final opposing Pokemon before Rest
  neither executes Rest nor emits a later Leftovers tick.

These controls satisfy neither trigger in the existing stop rule. They show
the scheduler behavior for the constructed states only; they do not identify
the unreplayable retained rows' damage roll, world construction, or event
matching outcome.

The two-consumer engine build fingerprint is also not attested: the local
environment has the patched `poke_engine` wheel but no installed
`pokezero_search` consumer. The wheel's terminal-residual ablation is an
additional focused control, not certification-quality replay evidence.

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
