# Vendored rejected-experiment matcher — provenance

## What this is

`rejected_experiment_761fc647_engine_transition_differential.py.txt` is a **byte-exact** copy of

    scripts/engine_transition_differential.py

as it stood at commit `761fc647a1c1a9ee57f09d44d2675d66c6d649b3` ("Fix retained damage
composition matcher tails", Scott Fitzgerald, 2026-07-30). That is the **rejected experiment**
matcher recorded in `reports/c26_damage_composition_tail_readout.json` under
`.rejected_experiment` — the variant production deliberately did **not** adopt.

    sha256 = 56ca68576587ad5a7fda64e28a1479d01a48d69374f13f440060b0bf32126f24

which is `tests/test_c26_damage_composition_readout.py::EXPERIMENT_MATCHER_SHA256` and
`reports/c26_damage_composition_tail_readout.json .rejected_experiment.matcher_source_sha256`.

It is stored with a `.txt` suffix so no test collector, linter, or import ever picks it up. It is
**not** production code and must never be imported, executed, or copied back into `scripts/`.
`tests/test_c26_damage_composition_readout.py::test_production_matcher_is_not_the_rejected_experiment`
exists precisely to catch that.

## Why it is vendored rather than read from git

**Commit `761fc647a1c1a9ee57f09d44d2675d66c6d649b3` exists on no remote.**

    gh api repos/sfitzgerald-x1/pokezero/commits/761fc647...  -> 422 "No commit found for SHA"
    git fetch --depth 1 origin 761fc647...                    -> "upload-pack: not our ref"

It was squash-merged, and the original commit was never pushed. Until 2026-08-07 it survived only
as a dangling object in one developer's local object store — reachable from no ref, one `git gc`
from being gone for good.

`test_matcher_sources_match_their_pinned_sha256` used to recover these bytes with
`git show 761fc647...:scripts/engine_transition_differential.py`. That can only ever work in the
single clone that happens to still hold the object; it fails in every fresh clone and in CI. The
test now hashes **this file** instead, so the check is clone-independent and passes anywhere.

## Do not "clean this up"

Deleting this file, or truncating/reformatting it, silently destroys the last copy of the bytes
the C26 readout's rejected-experiment provenance is pinned to. The digest above is the only thing
that proves the commit the readout cites actually held the code the readout reports. If it must
move, move it byte-for-byte and re-point
`tests/test_c26_damage_composition_readout.py::VENDORED_EXPERIMENT_MATCHER`.
