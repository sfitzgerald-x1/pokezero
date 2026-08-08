# Vendored rejected-experiment matcher — provenance

## What this is

`rejected_experiment_761fc647_engine_transition_differential.py.txt` is a **byte-exact** copy of

    scripts/engine_transition_differential.py

as it stood at commit `761fc647a1c1a9ee57f09d44d2675d66c6d649b3` ("Fix retained damage
composition matcher tails", Scott Fitzgerald, 2026-07-30). That is the **rejected experiment**
matcher recorded in `reports/c26_damage_composition_tail_readout.json` under
`.rejected_experiment` — the variant production deliberately did **not** adopt.

    sha256        = 56ca68576587ad5a7fda64e28a1479d01a48d69374f13f440060b0bf32126f24
    git blob OID  = 51f9025e4a5afc1e81e1cba31b470f33d113683d

The sha256 is `tests/test_c26_damage_composition_readout.py::EXPERIMENT_MATCHER_SHA256` and
`reports/c26_damage_composition_tail_readout.json .rejected_experiment.matcher_source_sha256`; it
is what the pin asserts. The blob OID adds no security — it is recorded because it is the id you
would grep an object store, a reflog, or `git fsck --lost-found` with if that dangling object ever
has to be re-found, and it exists in exactly one clone. `git hash-object` of this file equals
`git rev-parse 761fc647...:scripts/engine_transition_differential.py`.

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
single clone that happens to still hold the object; it failed in every fresh clone. The test now
hashes **this file** instead, so the check is clone-independent and passes anywhere.

Not "and in CI": this suite runs in **no workflow at all**. `tests/test_c26_damage_composition_
readout.py` appears in no `FILTERS` entry and in no step of any workflow, so it has never been
executed by CI and was never red there. That is a separate hole, and it is why
`test_production_matcher_is_not_the_rejected_experiment` sat red on main from #1167 until #1174
with nothing noticing. Closing it is **not** done here, because it is bigger than it looks:
`mass-gate` checks out at the default shallow depth, so the `PINNED_MAIN` leg of this very test
(and `test_c27_repro_provenance_is_current_main_baseline`) would fail there until that job also
sets `fetch-depth: 0`. Changing the checkout of the sole required status check is not something to
slip into a PR about a certification pin -- the note appended to
`reports/c112_leaf_state_divergence_ledger.md` in this same PR is specifically about why unscoped
"while I'm here" edits are how defects land. Filed as a follow-up: add
`tests/test_c26_damage_composition_readout.py`, `reports/c26_damage_composition_tail_readout.json`,
`reports/certification_contract_lifecycle.json` and this directory to `FILTERS`, set
`fetch-depth: 0` on `mass-gate`, and add a step pinning `Ran 10 tests` and `OK (skipped=1)` --
the skip count matters, because exactly one test here skips without `C26_RETAINED_SHARDS` and a
bare `OK` grep would accept a run that skipped everything.

## Do NOT push the preserving tag

An annotated tag `preserve/rejected-experiment-761fc647` anchors the commit in one local clone. It
is deliberately **not** pushed, and vendoring these bytes is what makes not pushing it safe.

Pushing a tag publishes the **entire object graph reachable from the commit**, so pushing that tag
would republish the whole tree at `761fc647` to a **public** repo. That snapshot predates the
2026-08-03 scrub: measured against the exact regex `tests/test_public_invariant.py` enforces,
`[/\\](Users|home)[/\\]+[A-Za-z0-9._-]+`, it still carries **92 occurrences across 41 files**
(`docs/audit_artifacts/**`, `scripts/*.py`, `docs/*.md`). The current tree has **zero**. Restoring
reachability that way would undo the scrub.

This single file is clean -- zero occurrences -- which is exactly why the right move was to vendor
one verified file rather than republish a commit. The tag preserves the object locally; the
vendored file is the artifact that may be public.

## Do not "clean this up"

Deleting this file, or truncating/reformatting it, silently destroys the last copy of the bytes
the C26 readout's rejected-experiment provenance is pinned to. The digest above is the only thing
that proves the commit the readout cites actually held the code the readout reports. If it must
move, move it byte-for-byte and re-point
`tests/test_c26_damage_composition_readout.py::VENDORED_EXPERIMENT_MATCHER`.
