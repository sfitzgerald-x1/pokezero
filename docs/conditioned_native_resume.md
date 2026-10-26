# Conditional continuation oracle

The opt-in `conditioned-oracle` build exposes
`pokezero_search.conditioned_resume(state_string, capsule_json, replacement_index, entry_ties)`.
It resumes a queued action using its saved move slot, critical/damage draw and
secondary draw, then applies the deferred residual block once. No new ordinary
chance inputs are accepted. The full suspended-turn capsule is retained and
validated; completed queue fields are cleared explicitly and reversibly.
The caller retains the original input on refusal. This API selects no action,
scores no game and is not connected to a scientific collector.

The current admitted domain is a clean direct Baton Pass with either queued
Thunderbolt/Body Slam or an already empty action queue, in either seat.
Body Slam's target paralysis uses the saved secondary coordinate; Static's
contact paralysis uses the actor's separate saved phazing coordinate (20/60).
An empty queue requires both native commitments and queued slots to be empty;
the replacement runs only the deferred residual, with no repeated move or PP.
It supports ordinary poison residuals and Leftovers, and refuses unsupported
abilities, statuses, volatiles, hazards, residual ties and other queued moves.
Sleep Talk, entry-tie correlations and hazard-KO chains remain follow-on work.
The general future-state mapper's
`LIVE_SUSPENSION_INTERFACE` refusal remains in place. These authored fixtures
provide no historical-panel reclassification, pool coverage or whole-engine
qualification.

From a clean checkout, vendor the pinned source with
`scripts/vendor_poke_engine_src.sh <isolated-python>`, then use that interpreter
to run `maturin build --no-default-features --features conditioned-oracle` in
both `third_party/poke-engine-src/poke-engine-py` and `rust/pokezero-search`.
Install both wheels only in the isolated environment. The feature enables
full-PP accounting; ordinary builds do not expose the resume entry point.
This recipe is for a clean checkout because vendoring replaces the ignored
native source directory. Never vendor over an independently modified tree.

Run `python -m unittest tests.test_conditioned_native_resume -v` with CPU JAX
and both opt-in bindings installed. CI requires the profile and rejects skips.
The tests compare complete canonical successors and every capsule field in
both seats, check the saved 9/10 secondary threshold against contradictory
new ordinary draws, and cover terminal poison KOs, minimum-one nonimmune
damage, Ground immunity, independent contact/target rolls, incoming versus
outgoing Static, poison and Ghost controls, and second-action Baton without a
queued move. Unsupported capsules must raise rather
than fall back to unconditional enumeration.
