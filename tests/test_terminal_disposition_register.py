"""Re-derive `reports/c155_terminal_disposition_register.md` from the tree on every run.

WHY THIS MODULE EXISTS.

`RATIFIED_SWEEP_PRECONDITION` gates the program's terminal measurement on *"the ledger is
terminal and the engine fingerprint is declared frozen for the claim."* The list of what
stands between here and "terminal" had been RECONSTRUCTED THREE TIMES by three different
agents -- from `reports/c152_ledger_terminal_disposition.md` §7.1 plus C153 and C154 --
with different numbering each time, and the third reconstruction's reviewer said so: it is
a reconstruction, not a maintained document, and its numbering wants reconciling before an
owner decision rests on it.

A document rebuilt on demand is the artifact this program keeps finding drifted. Four
counts in `reports/c138_known_gaps_ledger.md` went stale before anything re-derived them;
#1202 moved fifteen line citations in a single merge. This module is the register's
control: the item inventory, the item statuses and EVERY figure in its Appendix A are
re-derived here, so the register cannot be quoted at a value the tree no longer carries.

WHAT IS PINNED.

  1. **The item inventory is exact and ORDERED, in both directions.** `T1`-`T14` are
     permanent ids. An item added, dropped, renumbered or reordered is red. Renumbering is
     precisely what made the three reconstructions incomparable, so it is mechanically
     impossible rather than discouraged.

  2. **Statuses and actors come from a closed vocabulary**, and every item's status is
     re-derived from the tree where the tree can decide it. `T6` is DISCHARGED-IN-SCOPE
     because the C154 artifact says all 26 verdicts were re-adjudicated; flipping it to
     OPEN is manufacturing openness and is red, exactly as flipping `T2` to
     DISCHARGED-IN-SCOPE is manufacturing terminality and is red.

  3. **Appendix A is exact SET EQUALITY on keys, both directions, and exact on values.**
     A key derived here and missing from the document is red; so is a key in the document
     that nothing derives. That is the property the ledger's own counts lacked.

  4. **Every line citation is resolved by ANCHOR with uniqueness required**, using
     `_anchor` IMPORTED from `scripts/c153_wide_negative_census.py` rather than copied, so
     a moved or ambiguous anchor is one loud failure instead of a wrong number. The G8
     call sites are cited in the TRACKED PATCH, not in `generate_instructions.rs`:
     `third_party/poke-engine-src/` is gitignored, so a vendored line number is
     unresolvable by any check -- which is how C152's four citations came to be quoted
     from an instrumented build.

  5. **Anti-vacuity.** Every derivation that counts something is backed by a nonzero
     control, and the two parsers (items table, Appendix A) are asserted to find the
     tables at all. A loop over zero rows passes every assertion in this module; this repo
     has found eleven-plus inert pins, most recently a "control" that could not fail and a
     scan silently covering part of the workflow's guards (#1205).

WHAT IS NOT PINNED, ON PURPOSE, AND THE REGISTER SAYS SO PER ITEM.

  * Whether either gate is MET. C151 §3 records that the trigger is "a condition on
    program state, not a date, and it is not machine-checkable". Nothing here makes it so.
  * T3's disposition. The figures are derived; "no tie-arm divergence observed, and no
    verdict-producing instrument records the residual speed order" is a reading.
  * T6 residue 4. The four phrase-guard normalisations are derived from C154's pin; "no
    fifth obfuscation exists" is not a claim any check makes.
  * The classification of T11 and T14 as STANDING rather than OPEN.
  * T2's "still unmeasured" and T5's "uncompared". Their figures are derived; the STATUS
    rests on the ledger's prose, not on an absence scan. Both carry `derived + reading`.

A DECLARED COUPLING. `base.expected_counter_artifacts` and `base.expected_sweep_artifacts`
are pinned against the two census modules' own constants, read by AST rather than imported
(`tests/test_boundary_verdict_partition.py` imports the differential, which imports
`poke_engine`, which is absent without a built engine). So a PR that adds an artifact must
update the register in the same change. That is deliberate: the document that goes stale is
the one nothing forces an author through.

MUTATION BATTERY: 77 applied, 77 caught, plus 1 NEGATIVE CONTROL verified green
and 1 DISCLOSED SURVIVOR, enumerated below and NOT counted as caught.
Partitioned by WHAT IS MUTATED. Enumerated because
an unrecorded battery is what `tests/test_wide_seed_negative_census.py` records costing it a
surviving mutation, and because "the tests pass" is the same kind of claim this module
replaces. Each was applied to a clean tree, this module run, and the tree restored.

⚠ THE FIRST REVISION MIS-PARTITIONED ITS OWN BATTERY, and review caught it. It said "block
B's ten are applied only to the tree", when two of that block's ten -- deleting an accept bar
from a paragraph, adding a bare pipe to a table row -- edit the REGISTER and fire through
document-reading tests. Inflated, not fabricated; and tree-only is the property that
distinguishes this pin from a diary, so the split is now by target and the number is stated
where it can be checked. Three of the old block A (a patch line shift and two `events.rs`
edits) were likewise tree-side and have moved the other way.

BLOCK A -- A1-A37, applied to the REGISTER's own bytes.

  A1.  `T7` deleted from the item table.
  A2.  `T15` appended with a well-formed row. An item cannot join silently.
  A3.  `T3` and `T4` swapped. Order is part of the inventory.
  A4.  `T2`'s status OPEN -> DISCHARGED-IN-SCOPE, i.e. manufacturing terminality.
  A5.  `T6`'s status DISCHARGED-IN-SCOPE -> OPEN, i.e. manufacturing openness.
  A6.  `T1`'s actor AGENT-THEN-OWNER -> OWNER. The vocabulary is closed AND the value is
       pinned per item, so a plausible substitution inside the vocabulary still fails.
  A7.  `T11`'s actor -> AGENT.
  A8.  `T3`'s `pin` verdict `derived + reading` -> `derived`. Claiming a pin covers a
       judgement is the defect C154 §5 enumerates, and §6 must list every item carrying a
       reading, so this fails on both halves.
  A9.  `t1.head_fingerprint` changed by one hex digit.
  A10. `t1.committed_json_carrying_head_fingerprint` 0 -> 1.
  A11. `t1.freeze_declaration_constants` 0 -> 1.
  A12. `t2.hp_ceiling_site_lines` changed to the VENDORED `4197, 4406`. The historical
       defect replayed: a line number from a tree no check can resolve.
  A13. `t2.split_hunks_touching_an_hp_ceiling_site` 0 -> 1.
  A14. `t3.speed_ties_order_le_10` 20 -> 24, the over-broad population C152 itself
       corrected in review.
  A15. `t3.order_le_10_ties_carrying_a_winner_heal` 3 -> 7.
  A16. `t3.tie_refusal_line` decremented by one.
  A17. `t4.branch_miss_pct` 100.00 -> 99.00.
  A18. `t4.undiagnosed_sibling_rows` emptied. The second row of G33c's class is what
       deriving found and transcribing did not; it cannot be dropped again.
  A19. `t4.heal_mismatch_rows_in_the_wide_census` 2 -> 1, back to the count C152 and the
       ledger's G33c cell both imply.
  A20. `t5.subkeyed_single_seat_counters_in_corpus` 0 -> 1.
  A21. `t6.rows_foreclosed_over_section_4_population_only` widened to add R14.
  A22. `t6.workflow_steps_checking_out_showdown` 0 -> 1.
  A23. `power.divergences_for_a_tenfold_tighter_bound` 9490 -> 949.
  A24. An Appendix A row DELETED (`bar.support_gated_dev`). Removal is as red as
       alteration; a pin that only checks the keys present is defeated by deleting the
       inconvenient one.
  A25. The support-gated bar deleted from T2's paragraph.
  A26. A bare `|` introduced into a register table row. GFM DROPS the surplus cells, so a
       row can lose its actor column while keeping it in the bytes -- the G37 defect
       `tests/test_ledger_table_uniformity.py` exists for, on a new document.
  A27. §2's status tally 11 OPEN -> 10 OPEN. `EXPECTED_STATUS` forces an author through the
       column; before this check nothing forced them through the sentence a reader meets
       first, which is how the ledger's own counts went stale three times.
  A28. A §5 entry de-listed so the section names one fewer delta than it claims.
  A29. `scope.section3_rows_untouched_since_c138` 73 -> 0.
  A30. `cost.head_resweep_minutes` 19 -> 5.
  A31. The accept bars deleted from T14's paragraph, leaving a two-window zero quoted bare
       in a DIFFERENT paragraph from A25's. Both are needed: the first revision of that
       check keyed on the literal `31,082` and covered exactly one of the six paragraphs
       the widened detector finds.
  A32. The register's stated `Ran N tests` decremented while the module is unchanged.
  A33. The register's stated mutation total incremented while this list is unchanged.
  A34. The register's tree-only word ("Fifteen is the measured figure") decremented.
  A35. The stated count of EXECUTABLE workflow invocation sites incremented.
  A36. The stated count of guards the #1204 scan RESOLVES incremented.
  A37. The stated count of UNRESOLVED sites, "four", changed to "five" -- which is exactly
       the regression review found: counting the one comment line that carries the
       invocation string as a guard site. A32-A37 and B50-B51 exercise
       `TheDocumentsClaimsAboutItselfAreReDerivedTests`, which exists because review blocked
       two successive revisions on claims the document made about ITSELF.

BLOCK B -- B38-B77, FORTY mutations applied ONLY to the tree and never to the document.
Block A can be passed by a pin that reads the register against a hard-coded copy of itself.
These are the ones that prove each derivation reads what it claims to: every one MAKES A REAL
CHANGE TO THE TREE and the document, unedited, must go red. Six are the absences, and an
absence pin that cannot see its own subject appear is the eleven-plus-times defect this repo
has recorded.

  B38. A blank line inserted above the first `residual_disjoint_bands(` call in the tracked
       patch, shifting all four sites. The ANCHORS follow it; the document does not.
  B39. The `_ => NO_TRUNCATION,` arm deleted from `events.rs` -> loud anchor failure rather
       than a silently wrong line number.
  B40. A second `!leftovers_truncated[i]` guard added to a damage push in `events.rs`, i.e.
       G33c FIXED. An improvement has to be recorded rather than absorbed, which is the
       property `tests/test_ledger_table_uniformity.py` mutation 9 earned.
  B41. `skip:single_seat_boundary:phazing` added to a committed sweep. T13's absence is a
       scan of the corpus, and now it finds one.
  B42. `FROZEN_FOR_CLAIM = None` added to the differential. T1's "nothing can declare it
       frozen" is a name-level AST scan, and now something can.
  B43. A real `git clone .../pokemon-showdown` step added to the workflow. T6 residue 1's
       zero is a non-comment scan, and now the checkout exists.
  B44. A committed sweep's fingerprint rewritten to the head value. T1's "no committed
       artifact carries it" is progress when it happens and must be recorded, not absorbed.
  B45. A `defender_active.hp,` context line added inside one of C149's split hunks.
  B46. Ledger §7 item 5 marked RESOLVED. A resolved §7 item cannot keep a register slot, and
       the register's item count moves with the ledger.
  B47. A C154 verdict flipped `UNREACHABLE_TRACED` -> `NOT_OBSERVED_AT_SCOPE`. T6 is
       DISCHARGED-IN-SCOPE only while the artifact says all 26 were traced.
  B48. The ledger's §4 population sentence reworded -> loud anchor failure.
  B49. Every C152 marker removed from the ledger's H8 row, so the row leaves the
       re-examined set. Red on the id list AND on the count -- see C1 for what this
       mutation does NOT establish.
  B50. The workflow step's `Ran N tests` guard bumped while the module is unchanged. This is
       the #1205 shape applied to this step's own guard: the count it demands becomes one
       its suite can never print, so the guard stops failing closed.
  B51. The workflow comment's mutation total set to this module's TEST count -- the exact
       conflation review found in the first revision.
  B52. A SWAP: H8's marker stripped and a `C152` mention added to G1, so one row leaves the
       re-examined set and another joins. The COUNT stays at 9 and the MEMBERSHIP changes.
       This is the only mutation in the battery the count cannot catch, and it is therefore
       the whole justification for publishing the id list. Verified red on
       `scope.section3_rows_touched_since_c138` and green on
       `scope.section3_rows_touched_count`.
  B53. A NEW UNGUARDED unittest step added to the workflow, so #1205's unresolved set grows
       by one. This is the load-bearing half of the guard-scan pin: the counts churn with
       any step, the unresolved SET is what #1205 is about, and it is pinned by module name
       so a line shift cannot move it and a new blind spot cannot join quietly.
  B54. ⚠ THE ONE THAT SURVIVED, and the reason `TheFingerprintDerivationReadsTheTreeTests`
       exists. `head_fingerprint` re-pointed at Appendix A's own cell
       (`return register_facts()["t1.head_fingerprint"] + "0" * 48`), i.e. the pin made
       SELF-SATISFYING. All 42 tests then reported a clean OK **on a tree whose
       `rust/pokezero-search/src/tree.rs` had been edited** -- exactly the event T1's row
       exists to redden -- and the CI step's `Ran N tests` and `^OK$` guards passed too,
       because the method count never moved. Blocks A and B between them target the
       DOCUMENT and the TREE; neither targets the DERIVATION, so this whole battery was
       blind to the one edit that turns the row into a tautology. Now caught, sensitivity
       and specificity both, and the new class carries the negative control that keeps its
       sandbox honest.
  B55. ⚠ SURVIVED THE FIRST REVISION OF B54's FIX, and independent review found it.
       `derive()`'s `t1.head_fingerprint` line re-pointed at `register_facts()`. The row
       reaches CI through THREE seams -- `head_fingerprint`, `derive()`, the comparison
       loop -- and pinning only the first moved the tautology one frame up: clean 48/OK on
       an edited crate source. A guard that pins one seam of a three-seam path names the
       seam it pins.
  B56. ⚠ SURVIVED likewise: `test_every_value_re_derives`' loop taught to `continue` on
       `t1.head_fingerprint`. Now caught by a SECOND, independent comparison that routes
       through neither `derive()` nor the loop.
  B57. ⚠ SURVIVED: `cargo = []` inside `compute_fingerprint`, so `Cargo.toml`/`Cargo.lock`
       stop reaching the digest while still being listed as inputs.
  B58. ⚠ SURVIVED: `build_metadata = []` likewise, dropping `build.rs` and the crate's
       `pyproject.toml` -- the maturin FEATURE FLAGS, which decide what the extension can
       even do. One line, and green.
  B59. ⚠ SURVIVED: `digest.update(hashlib.sha256(BASE_SOURCE.read_bytes()).digest())`
       deleted, so the pinned upstream sdist could change under the whole patch stack.
  B60. ⚠ SURVIVED: `digest.update(PATCH_LIST.read_bytes())` deleted. `build_inputs()` does
       not contain `PATCH_LIST`, so a probe set taken from it alone leaves the manifest
       unprobed -- which is precisely why the probe set adds it explicitly.
       B57-B60 are one defect with four faces: the first revision claimed "sensitivity to
       each hashed input class" while probing TWO of the five. The unscoped negative this
       register exists to prosecute, committed in the guard against it. The sensitivity is
       now EXHAUSTIVE -- every file the hasher reads, derived from `build_inputs()` -- so a
       class added later is covered the day it lands.
  B61. `cargo_inputs()` and `build_metadata_inputs()` each returning `[]`. Distinct from
       B57/B58 and the reason the by-name floor exists: this shrinks the PROBE SET and the
       digest together, so an exhaustive loop derived from `build_inputs()` goes green over
       a smaller world. A floor derived from the thing it floors is not a floor, so the
       required input names are literals.
  B62. The exhaustive loop's `finally: path.write_bytes(original)` removed. Probes then
       compound and every one after the first measures an already-changed tree, which would
       make the whole loop pass on a single real sensitivity. Caught by a post-loop identity
       check against the baseline.
  B63. ⚠ SURVIVED THE SECOND REVISION, and it is the THIRD ROUND OF ONE DEFECT.
       `register_facts()` seeded with `{"t1.head_fingerprint": head_fingerprint()[:16]}`.
       Round 1 pinned `head_fingerprint` and the tautology moved to `derive()`; round 2
       pinned `derive()` and the comparison loop and it moved HERE, to the reader every
       document-side assertion funnels through -- including both of round 2's
       "independent" ones. The published claim "the row reaches CI through THREE seams"
       was false, and a fourth seam-specific pin would have been the same mistake again.
  B64. ⚠ THE SAME DEFECT ONE FRAME FURTHER OUT: `_text()` rewriting the row on the way
       past. This is the mutation that ends the chain, because it is killed by EXACTLY ONE
       test -- the one that reads the document with `Path.read_bytes` and a regex in its
       own body and the tree with `compute_fingerprint`, sharing no helper with the module.
       Verified: `FAILED (failures=1)`, and that one is
       `test_the_row_holds_when_read_through_none_of_this_modules_helpers`. Every other T1
       pin is green under this mutant. A guard chain has to terminate somewhere that reads
       primary bytes; until it does, each round moves the tautology outward.
  B65. FOUR SUBSET HASHERS, all of which SURVIVED the second revision's FLOORS:
       `crate_sources()` skipping `tree.rs`, `crate_sources()[:6]`, `patch_files()[:60]`,
       and `_checked_tree_inputs({"Cargo.toml"})` dropping `Cargo.lock`. The floors were
       `> 5` `.rs` against 11 actual, `> 50` `.patch` against 74 and `> 70` total against
       91 -- roughly twenty files of slack, i.e. a margin wider than the defect. The
       exhaustive loop cannot catch these on its own BY CONSTRUCTION: it derives its probe
       set from the hasher, so a hasher that reads less is probed less. Replaced by exact
       SET EQUALITY against a re-derivation that shares no code with the hasher.
  B66. TWO MUTATIONS THAT DECAY THE IDENTITY TO A BAG OF BYTES, and they need DIFFERENT
       probes, which is why both exist. Neither changes any file's bytes, so every per-file
       sensitivity probe is unaffected; both must be measured RESEATED, i.e. with Appendix A
       updated to the mutant's own value, because that is what the failure message tells an
       author to do.
       (a) The digest made order-insensitive (XOR of the per-file hashes). Caught by
           TRANSPOSING two crate sources' contents: the byte multiset is held fixed and only
           the association changes.
       (b) `digest.update(str(path.relative_to(REPO_ROOT)).encode())` deleted. ⚠ The
           transposition probe CANNOT see this one -- the native-input loop iterates in
           sorted-path order, so with the path gone the ORDER of content hashes still
           encodes which file held what, and a transposition moves the hash either way.
           Caught by a pure RENAME that PRESERVES the sort position (the last crate source
           renamed to a name that still sorts last): same bytes, same sequence, different
           path set. Recorded because a first fix here claimed the transposition covered
           both and measurement said otherwise.
  B67. ⚠ THE META-DEFECT, and it defeated the first fix for B65. The sandbox populated from
       `crate_sources()` / `cargo_inputs()` / `build_metadata_inputs()` -- the hasher's OWN
       accounting. A hasher skipping `tree.rs` then produced a sandbox missing `tree.rs`,
       the independent glob of that sandbox found the same short list, and the EXACT SET
       EQUALITY agreed with itself: B65a and B65b both survived reseated at a clean OK with
       the equality in place. A control populated by the thing it controls is not a control.
       `_tree_side_hashed_inputs` now enumerates from the real repo and populates the
       sandbox, so the copy is a copy of the TREE rather than of the hasher's opinion of it.
  B68. ⚠ AND B67's FIX MADE THE ENUMERATOR A SINGLE POINT OF FAILURE -- the same defect a
       FOURTH frame out, found by the author while re-reading B67 rather than by review, and
       recorded because the sequence is the lesson. `_tree_side_hashed_inputs` had become
       both the sandbox's source AND the equality's expected side, so a hasher and an
       enumerator that shrink IN THE SAME DIRECTION agree with each other:
       `crate_sources()` and the enumerator both skipping `tree.rs`, reseated, was a clean
       `Ran 51 tests, OK`. Four variants measured -- collude on a crate source, subset the
       enumerator alone, point the enumerator AT `build_inputs()` so it has no independence
       left, and collude on the patch manifest.
       THE FIX IS TO STOP TERMINATING IN CODE. Appendix A now records
       `t1.hashed_input_files` (91) and `t1.hashed_crate_sources` (11), derived from the
       enumerator, so a shrunken enumerator disagrees with the DOCUMENT even when the hasher
       shrinks with it. ⚠ RE-MEASURED AT HEAD, and the earlier claim "all four variants die on
       `test_every_value_re_derives`" was WRONG TWICE: the killer is now
       `test_the_hashed_crate_inputs_are_the_files_git_tracks` (the git anchor postdates this
       entry), and the FOURTH variant -- aliasing the git test's enumerator arm to the
       hasher's set -- SURVIVES. It is listed under DISCLOSED SURVIVORS below rather than
       counted here. Three of four: 1, 2, 1 failures respectively. A declared
       coupling and a free one: anything that changes this inventory moves the fingerprint
       too, so Appendix A is already being edited in the same change.
  B69. ⚠ AND B68's FIX HAD A FIFTH FRAME TOO, which is why the register now says to expect
       one rather than claiming there is none. Appendix A's inventory rows stop a hasher and
       an enumerator COLLUDING -- but not an author who reseats the inventory rows as well.
       Measured: hasher and enumerator both skipping `tree.rs`, with `t1.head_fingerprint`,
       `t1.hashed_input_files` AND `t1.hashed_crate_sources` all updated to the mutant's own
       values -- i.e. every number the failure messages name -- was a clean
       `Ran 51 tests, OK`. Nothing in the module OR the document knew that 11 was right.
       `git ls-files` does, and it is a THIRD anchor outside both. Against it the collusion
       is red however many rows are reseated, and it is killed by that one assertion alone:
       `FAILED (failures=1)`.
       SCOPE, stated because an unscoped anchor is this register's signature defect: git
       pins the GLOBBED classes exactly -- crate sources and the four build files -- and
       NOT the patch class, because `third_party/` tracks 77 `.patch` files while the
       manifest lists 74 and not hashing the other three is CORRECT. What is pinned there
       instead is that every patch the manifest names is tracked, so it cannot name a
       phantom.
  B70. ⚠ THE SIXTH FRAME, and it is the PATH -- review round 3, after I had claimed the
       chain terminated. `REGISTER` AND the literal beside it both re-pointed at a doctored
       copy of the register: the helper-free assertion then reads the doctored file, the
       published one stays stale, and the old drift check APPROVED it, because requiring the
       two literals to agree forces them to move together and pins neither to the file CI
       publishes. "Primary bytes" and "the document" were one anchor, not two. Closed by
       anchoring the path OUTSIDE this module -- to the workflow's `changes` paths filter,
       which is what decides whether `mass-gate` runs on this document at all, and to git's
       index, so a doctored copy has to be committed to exist in a CI checkout, at which
       point the filter names the wrong one.
  B71. ⚠ `_tracked` FILTERED THROUGH `crate_sources()` in three lines, with hasher and
       enumerator colluding and all three Appendix A rows reseated. B67's own sentence -- a
       control populated by the thing it controls is not a control -- landing on the git
       anchor added to close B69, one round later. Closed by NAMED LITERALS: the anchor must
       see `tree.rs`, `leaf.rs`, `model.rs`, `events.rs`, `encoder.rs` and `lib.rs`, so
       narrowing it is red however the narrowing is spelled. `tree.rs` is named explicitly
       because it is the file every collusion mutant in this battery drops.
       ALSO RECORDED, NOT CLOSED HERE: a module-level `load_tests` can drop the terminating
       assertion while holding the printed count at its pinned value.
       `tests/test_unreachable_readjudication.py` REFUSES a `load_tests` in any module a
       `Ran N` guard names -- measured, `FAILED (1)` with that exact message -- and
       `mass-gate` runs that module too, so the CI-level guard exists; it is simply not in
       this file, and the class docstring says so rather than implying this module is
       self-sufficient.
  B72. ⚠ B71's FIX WAS SIX OF ELEVEN -- review round 4, and the seventh frame. The named
       literals were `tree.rs`, `leaf.rs`, `model.rs`, `events.rs`, `encoder.rs`, `lib.rs`;
       git tracks ELEVEN crate sources. B71 re-run with `priors.rs` substituted for one of
       the six is a clean `Ran 52 tests, OK`, and with it landed and reseated a LIVE EDIT to
       `priors.rs` is invisible -- a file this register's own T1 commit table cites as having
       moved this row. Six of eleven is a floor with five files of slack, which is the same
       defect as every other floor in this battery, committed while fixing one of them.
       Closed by DERIVING the name set from `lib.rs`'s own `mod` declarations (10 + `lib.rs`
       = 11), which is exact, self-maintaining, and removes a literal instead of adding five.
       The pattern includes the visibility qualifier because `priors` is declared
       `pub(crate) mod` -- a first attempt matched only `pub mod` and missed exactly the file
       the finding was about.
  B73. ⚠ `_perturb` IS APPEND-ONLY, so A SIZE PASSED FOR A HASH -- and this one defeats every
       other probe in the class at once. `compute_fingerprint` hashing `path.stat().st_size`
       instead of the file's contents satisfies the exhaustive loop, the transposition probe
       AND the rename probe, all reseated, and an `f32` -> `f64` edit to `tree.rs` is then
       invisible with the fingerprint unmoved. Closed by swapping two ADJACENT DIFFERING
       bytes: content changes, length cannot. The probe asserts the length is unchanged
       first, because a probe that alters the length silently stops isolating the two.
  B74. THE FILTER ANCHOR WAS A POSITION-BLIND LINE GREP, defeated three ways in one round:
       DELETE the real `FILTERS` entry and satisfy the scan with a decoy indented inside a
       `run:` body (green, and register-only PRs then skip `mass-gate` -- the failure that
       filter's own comment calls the worst in the file); FALSE-TRIP it with a trailing
       comment on the real entry; FALSE-TRIP it with a second legitimate `reports/c155*`
       entry. Closed by extracting the `FILTERS: |` block BY INDENTATION, so only entries the
       match loop actually consumes are counted. Hand-parsed rather than via PyYAML, which is
       not in the CI job's install list.
  B75. ⚠ AND B74's FIX ADVERTISED THE HOLE AS THE FIX -- review round 5, and it is the worst
       single error in this row's history because the previous round ASKED for the wrong
       behaviour and I implemented it. The parser did `line.split("#", 1)[0].strip()`. The
       workflow does NEITHER: it skips a pattern only when its FIRST character is `#`, and
       otherwise passes the line VERBATIM to `case "$file" in $pattern)`. So a trailing
       comment on the register's entry -- and, separately, two extra spaces of indentation --
       produce a glob matching NOTHING, `fidelity=false`, and `mass-gate` never runs on a
       register-only PR, which is verbatim the outcome the assertion's own error message
       names. Both were green. Round 4 asked for the trailing-comment case to stay green and
       B74's note recorded that as a verified property; the correct answer was RED. The parser
       now replicates the shell -- YAML's block dedent, then the shell's own first-character
       rule, then exact membership -- because the shell is the only authority for what the
       gate actually consumes. Verified: trailing comment RED, extra indentation RED, decoy in
       a `run:` body RED.
  B76. ⚠ EVERY CONTENT PROBE IN THE CLASS PERTURBED A FILE'S HEAD OR ITS TAIL, NEVER ITS
       MIDDLE. `_perturb` appended; the length-preserving probe swapped the FIRST adjacent
       differing pair. So `sha256(blob[:512] + str(st_size))` survived all eleven tests
       reseated and hid an `f32` -> `f64` at byte 3010 of `tree.rs`. That is B73's own sentence
       one frame out: B73 was "a probe family that all change LENGTH is blind to a size
       proxy", and this is "a probe family that all perturb the same REGION is blind in that
       region". Closed twice over: `_perturb` splices into the MIDDLE, and the
       length-preserving probe now swaps at HEAD, MIDDLE and TAIL under separate subTests so
       the report names which region a proxy ignores. Verified: head+size RED,
       head+tail RED.
       ⚠ The mid-file splice had to learn the manifest's shape: a byte spliced mid-LINE in
       `poke-engine-gen3-patches.txt` renames a patch and `compute_fingerprint` RAISES rather
       than moving, so the probe would error instead of measure. It inserts a whole comment
       line at the newline nearest the middle instead. Found by running it.

  B77. ⚠ THE NINTH FRAME, and B76's own sentence one frame out for the THIRD time -- review
       round 6. B76's fix swapped two ADJACENT bytes, and an adjacent swap disturbs BOTH byte
       parities at once, so it cannot isolate a digest that reads only one of them; every
       other probe here changes length or path, so a residue-class digest moved under those
       too. `path.read_bytes()` -> `path.read_bytes()[::2]`, one line, survived all 53 tests
       reseated and hid a REAL SEMANTIC EDIT -- `DAMAGE_BRANCH_DEPTH: u8 = 2 -> 3` in
       `tree.rs` -- with the fingerprint byte-identical. The mirror `[1::2]` did the same.
       Closed by flipping a SINGLE byte (`^= 0xFF`) at BOTH PARITIES in each of three regions,
       so a digest skipping any residue class fails on the offsets it skips and the subTest
       names the region and the parity. Verified: `[::2]`, `[1::2]` and `[::3]` all RED, each
       by the length-preserving probe alone.
       THE SEQUENCE IS THE POINT: B73 was "all probes change LENGTH, so a size proxy is
       invisible"; B76 was "all probes touch one REGION"; B77 is "all probes disturb both
       PARITIES". Three rounds, one shape -- a probe family sharing a blind spot -- and each
       time the family looked exhaustive from inside.

BLOCK C -- NEGATIVE CONTROLS. Mutations that must stay GREEN because they do not change the
fact being derived. Recorded with the RED mutation that proves the same assertion is live,
because a negative control unpaired with one is indistinguishable from an inert pin -- which
is C154 §6's "control that could not fail", one surface over.

  C1.  ⚠ ONE of the six `C152` markers on the ledger's H19 row rewritten -> GREEN, and it
       SHOULD be. H19 with five remaining markers is still a row C152 re-examined, so
       ground truth has not moved and a red here would be a false positive. The predicate is
       `any(tag in line)`, insensitive to marker COUNT by construction.

       ⚠ **A previous revision of this module listed exactly this edit as a mutation that
       had SURVIVED, diagnosed the cause as "the derivation published only a count", and
       cited it as what earned the id list. Every part of that was wrong, and review
       measured all of it.** A row-id list is exactly as insensitive to marker count as a
       row count is, so the id list could not have been the fix; and B49 as shipped -- every
       marker stripped from a ONE-marker row -- moves the count too, so the count alone
       would have caught it. The battery total then held only because a survivor had been
       quietly replaced by a different mutation instead of being recorded. It is recorded
       now, in the block where a green result is the correct one, and the id list is
       justified by B52 instead.

       Paired live-assertion proof: B49 (all markers stripped from a one-marker row) and
       B52 (the swap) are both RED, so C1's green is the predicate being right rather than
       the assertion being absent.


DISCLOSED SURVIVORS. Mutations that are GREEN and arguably should not be. Enumerated because
the alternative is a battery that reads 77-of-77 while a reviewer can find a survivor in one
line, which is the exact credibility failure this document exists to prevent. NOT counted in
the totals above.

  S1.  The git anchor's ENUMERATOR arm aliased to the hasher's set
       (`enumerated = hashed`). Green. It is a REDUNDANCY rather than a hole -- the enumerator
       is pinned to the hasher by `test_the_hashed_input_set_is_exactly_what_the_tree_holds`,
       to the document by `t1.hashed_input_files` / `t1.hashed_crate_sources`, and the hasher
       is pinned to git -- so the property survives transitively while that one assertion
       stops carrying it. ⚠ RECORDED TWICE OVER because the bookkeeping went wrong here: I
       disclosed it, a review round told me it was killed, I EDITED THE CLAIM WITHOUT
       RE-RUNNING IT, and the next round caught that. Re-measured: it survives.
       Never edit a measured claim because someone says it is wrong -- re-measure it.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import engine_build_fingerprint  # noqa: E402
from c153_wide_negative_census import _anchor  # noqa: E402
from engine_build_fingerprint import compute_fingerprint  # noqa: E402
from test_ledger_table_uniformity import (  # noqa: E402
    _REACHABILITY_HEADER,
    _UNREACHABLE_HEADER,
    _tables_by_header,
    delimiter_anomalies,
    row_cells,
    tables,
)

REGISTER = "reports/c155_terminal_disposition_register.md"
PATCH_MANIFEST_NAME = "poke-engine-gen3-patches.txt"
LEDGER = "reports/c138_known_gaps_ledger.md"
WORKFLOW = ".github/workflows/engine-fidelity-gates.yml"
EVENTS = "rust/pokezero-search/src/events.rs"
THRESHOLD_PATCH = "third_party/poke-engine-gen3-status-aware-residual-threshold.patch"
SPLIT_PATCH = "third_party/poke-engine-gen3-leechseed-residual-band-split.patch"
C154_ARTIFACT = "reports/artifacts/c154_unreachable_readjudication.json"
C154_REPORT = "reports/c154_unreachable_readjudication.md"
C153_CENSUS = "reports/artifacts/c153_wide_negative_census.json"
G33B_CENSUS = "reports/artifacts/c152_g33b_open_arm_census.json"
G8_CENSUS = "reports/artifacts/c152_g8_survive_representative_census.json"
H8_CENSUS = "reports/artifacts/c152_h8_window_census.json"
HEAD_SWEEPS = {
    "dev": "reports/artifacts/c152_head_dev_sweep.json",
    "holdout": "reports/artifacts/c152_head_holdout_sweep.json",
}
G33C_SHARD = "reports/artifacts/c152_wide_census_1000500_sweep.json"
WIDE_SHARDS = tuple(
    f"reports/artifacts/c152_wide_census_100{n}_sweep.json"
    for n in ("0000", "0250", "0500", "0750")
)

#: The register's own item table. Ids are PERMANENT: allocated once, never reused, never
#: renumbered. The tuple is ordered, so a reorder is as red as a removal.
EXPECTED_ITEMS: tuple[str, ...] = tuple(f"T{n}" for n in range(1, 15))

#: Closed vocabularies. A value outside either set is a failure rather than a new category
#: absorbed silently -- R26 in C154's inventory is the standing example of what a fourth
#: category in disguise costs.
STATUS_VOCABULARY = frozenset({"OPEN", "DISCHARGED-IN-SCOPE", "STANDING"})
ACTOR_VOCABULARY = frozenset({"AGENT", "OWNER", "AGENT-THEN-OWNER", "—"})
PIN_VOCABULARY = frozenset({"derived", "derived + reading", "reading"})

EXPECTED_STATUS: dict[str, str] = {
    "T1": "OPEN",
    "T2": "OPEN",
    "T3": "OPEN",
    "T4": "OPEN",
    "T5": "OPEN",
    "T6": "DISCHARGED-IN-SCOPE",
    "T7": "OPEN",
    "T8": "OPEN",
    "T9": "OPEN",
    "T10": "OPEN",
    "T11": "STANDING",
    "T12": "OPEN",
    "T13": "OPEN",
    "T14": "STANDING",
}

EXPECTED_ACTOR: dict[str, str] = {
    "T1": "AGENT-THEN-OWNER",
    "T2": "AGENT",
    "T3": "AGENT-THEN-OWNER",
    "T4": "AGENT-THEN-OWNER",
    "T5": "AGENT",
    "T6": "AGENT",
    "T7": "AGENT",
    "T8": "AGENT",
    "T9": "AGENT",
    "T10": "AGENT",
    "T11": "OWNER",
    "T12": "AGENT",
    "T13": "AGENT",
    "T14": "—",
}

#: The `pin` column, per item. An item whose DISPOSITION is a human reading must say so here
#: rather than in prose a reader may not reach: C154 §5 had to add a fifth entry to its own
#: list of what no pin covers because the list omitted the classification introduced in the
#: same round. Four items carry `derived + reading` and each names its reading in §6.
EXPECTED_PIN: dict[str, str] = {
    "T1": "derived",
    # ⚠ T2 and T5 were `derived` until review. Their FIGURES are derived; the STATUS half
    # of each -- "still unmeasured", "uncompared" -- rests on the ledger's prose rather than
    # on an absence scan of the kind T13 carries. Both statuses are right; the label
    # overstated by one word, and the word is the distinction §6 exists to draw.
    "T2": "derived + reading",
    "T3": "derived + reading",
    "T4": "derived",
    "T5": "derived + reading",
    "T6": "derived + reading",
    "T7": "derived",
    "T8": "derived",
    "T9": "derived",
    "T10": "derived",
    "T11": "derived + reading",
    "T12": "derived",
    "T13": "derived",
    "T14": "derived + reading",
}

#: T7..T14 are the eight items §7 of the ledger does NOT mark RESOLVED, in §7's own order.
#: Pinned as a mapping so a §7 item that gets resolved cannot silently keep a register slot.
SECTION_7_ITEMS: dict[str, int] = {
    "T7": 1,
    "T8": 5,
    "T9": 6,
    "T10": 7,
    "T11": 8,
    "T12": 9,
    "T13": 10,
    "T14": 11,
}

_ITEM_HEADER = "| id | item | gate | status | actor | pin |"
_FACT_HEADER = "| key | value |"


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _text(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


def _artifact(relative: str) -> dict:
    return json.loads(_text(relative))


def _module_constant(relative: str, name: str):
    """A module-level literal, read by AST rather than imported.

    `tests/test_boundary_verdict_partition.py` imports the differential, which imports
    `poke_engine`; without a built engine that import raises. Reading the constant is the
    same fact without the dependency, and it is the fact the register cites.
    """

    tree = ast.parse(_text(relative))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level assignment in {relative}")


def _register_table(header: str) -> list[list[str]]:
    """The data rows of the register's table whose header line is `header`, as cells."""

    grouped = _tables_by_header(_text(REGISTER))
    found = grouped.get(header, [])
    if len(found) != 1:
        raise AssertionError(
            f"{REGISTER} carries {len(found)} tables with header {header!r}; expected 1. "
            "The register's machine-read tables are keyed on their header text so that "
            "reordering a section cannot silently retarget this pin."
        )
    return [row_cells(line) for _, line in found[0]["rows"]]


#: A paragraph asserting a zero over the two permitted windows. Matched against the
#: FLATTENED paragraph, because `**0** in the two head windows` is how one of the five is
#: actually written and a literal scan walks past it.
_TWO_WINDOW_ZERO = re.compile(
    r"(?:\b0 divergen|\bzero divergen|\b0 in the two\b|\b0 in both\b|\bzero in both\b"
    r"|windows are at 0\b|\bdivergence-free\b)",
    re.IGNORECASE,
)


def _flatten(paragraph: str) -> str:
    """Emphasis and code-span markers removed, whitespace folded."""

    return re.sub(r"\s+", " ", paragraph.replace("*", "").replace("`", ""))


def two_window_zero_paragraphs(text: str | None = None) -> list[str]:
    if text is None:
        text = _text(REGISTER)
    return [
        paragraph
        for paragraph in text.split("\n\n")
        if _TWO_WINDOW_ZERO.search(_flatten(paragraph))
    ]


def register_items() -> list[list[str]]:
    return _register_table(_ITEM_HEADER)


def register_facts() -> dict[str, str]:
    facts: dict[str, str] = {}
    for cells in _register_table(_FACT_HEADER):
        key = cells[0].strip("`")
        if key in facts:
            raise AssertionError(f"duplicate Appendix A key: {key}")
        facts[key] = cells[1]
    return facts


# ---------------------------------------------------------------------------
# Derivations. Every one of these answers a row of Appendix A.
# ---------------------------------------------------------------------------


def section_7_items() -> list[tuple[int, str]]:
    """Every numbered item of the ledger's §7, with its text."""

    body = _text(LEDGER).split("## 7. What I could not determine", 1)[1]
    body = body.split("\n## 8.", 1)[0]
    return [(int(n), t) for n, t in re.findall(r"(?m)^(\d+)\. (.*)$", body)]


def section_7_unresolved() -> list[int]:
    return [n for n, t in section_7_items() if not re.match(r"^(⚠ )?\*\*RESOLVED", t)]


def _ledger_rows(header: str) -> list[tuple[int, str]]:
    grouped = _tables_by_header(_text(LEDGER))
    return [
        entry for table in grouped.get(header, []) for entry in table["rows"]
    ]


def _ledger_row_ids(header: str) -> list[str]:
    return [row_cells(line)[0] for _, line in _ledger_rows(header)]


def committed_json() -> list[str]:
    """Every committed JSON under `reports/` and `docs/` -- the corpus both censuses use."""

    found: list[str] = []
    for tree in ("reports", "docs"):
        for root, dirs, files in os.walk(REPO / tree):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
            found += [
                os.path.relpath(os.path.join(root, name), REPO)
                for name in files
                if name.endswith(".json")
            ]
    return sorted(found)


def head_fingerprint() -> str:
    return compute_fingerprint()["fingerprint"]


def sweep_fingerprint(relative: str) -> str:
    distinct = _artifact(relative)["checkpoint_provenance"]["distinct"]
    fingerprints = {json.loads(entry)["engine_fingerprint"] for entry in distinct}
    assert len(fingerprints) == 1, (relative, fingerprints)
    return fingerprints.pop()


def freeze_declaration_constants() -> int:
    """Module-level names in the differential that would carry a freeze declaration.

    SCOPE OF THIS NEGATIVE, stated because an unscoped one is this program's signature
    defect: it is a NAME-level AST scan of one module. `OWNER_RATIFIED` exists there and is
    found by the sibling control below, so the scan is not vacuous. It says nothing about a
    freeze recorded somewhere this scan does not look, and the register's prose adds the
    literal scan of `scripts/`, `tests/`, `src/` and `.github/` that widens it.
    """

    tree = ast.parse(_text("scripts/engine_transition_differential.py"))
    names = [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    return sum(1 for name in names if "FROZEN" in name or "FREEZE" in name)


def differential_constant_names() -> list[str]:
    tree = ast.parse(_text("scripts/engine_transition_differential.py"))
    return [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]


def residual_disjoint_band_sites() -> list[tuple[int, str]]:
    """`(line, ceiling argument)` for every CALL of `residual_disjoint_bands`.

    Read out of the TRACKED PATCH. `third_party/poke-engine-src/` is gitignored and absent
    from a clean checkout, so a `generate_instructions.rs` line number cannot be resolved
    by any check -- which is exactly how C152's four citations came to be quoted from an
    instrumented build, shifted +11 by an `eprintln!` block. The ceiling is the fourth
    positional argument, taken by position rather than by matching a name.
    """

    lines = _text(THRESHOLD_PATCH).splitlines()
    sites: list[tuple[int, str]] = []
    for number, line in enumerate(lines, 1):
        stripped = line.lstrip("+ ")
        if "residual_disjoint_bands(" not in line or stripped.startswith("fn "):
            continue
        args = [lines[number + offset].lstrip("+ ").rstrip(",") for offset in range(4)]
        sites.append((number, args[3]))
    return sites


def split_hunks() -> list[dict]:
    """Every hunk of C149's split patch, tagged with what it gates and what it touches."""

    hunks: list[dict] = []
    current: list[str] | None = None
    for line in _text(SPLIT_PATCH).splitlines():
        if line.startswith("@@"):
            current = []
            hunks.append({"body": current})
        elif current is not None:
            current.append(line)
    for hunk in hunks:
        body = hunk["body"]
        hunk["adds_split_gate"] = any(
            line.startswith("+") and "if defender_leech_seeded {" in line for line in body
        )
        hunk["touches_i16max_site"] = any(
            re.match(r"^[ +]\s*i16::MAX,\s*$", line) for line in body
        )
        hunk["touches_hp_site"] = any(
            re.match(r"^[ +]\s*defender_active\.hp,\s*$", line) for line in body
        )
    return hunks


def leftovers_truncated_references() -> list[int]:
    return [
        number
        for number, line in enumerate(_text(EVENTS).splitlines(), 1)
        if "leftovers_truncated" in line
    ]


def single_seat_subkeys_in_corpus() -> list[str]:
    """Committed JSON carrying a SUB-KEYED `skip:single_seat_boundary:<reason>` counter.

    SCOPE: the committed JSON corpus, nothing wider. It cannot see an instrument nobody has
    committed, and the register says so.
    """

    hits: list[str] = []
    for name in committed_json():
        if "skip:single_seat_boundary:" in _text(name):
            hits.append(name)
    return hits


def workflow_showdown_checkouts() -> list[str]:
    """Non-comment workflow lines that would obtain a pokemon-showdown checkout."""

    return [
        line
        for line in _text(WORKFLOW).splitlines()
        if "pokemon-showdown" in line and not line.strip().startswith("#")
    ]


def c154_human_readings() -> int:
    body = _text(C154_REPORT).split(
        "## 5. What the pin covers, and the five things it cannot", 1
    )[1]
    body = body.split("\n## 6.", 1)[0]
    return len(re.findall(r"(?m)^(\d+)\. ", body))


def wide_shard_totals() -> tuple[int, int]:
    measured = diverged = 0
    for shard in WIDE_SHARDS:
        document = _artifact(shard)
        measured += document["boundaries_measured"]
        diverged += document["transitions_diverged"]
    return diverged, measured


def g33c_repro() -> dict:
    for repro in _artifact(G33C_SHARD)["repros"]:
        if repro.get("seed") == 1000513 and repro.get("step") == 121:
            return repro
    raise AssertionError(f"{G33C_SHARD} no longer carries 1000513/121")


def derive() -> dict[str, str]:
    """Every fact the register's Appendix A states, re-derived from the tree."""

    facts: dict[str, str] = {}

    # -- base state -------------------------------------------------------
    patches = _text("third_party/poke-engine-gen3-patches.txt").splitlines()
    facts["base.patch_stack"] = str(
        sum(1 for line in patches if line.strip() and not line.strip().startswith("#"))
    )
    facts["base.expected_sweep_artifacts"] = str(
        _module_constant(
            "tests/test_boundary_verdict_partition.py", "_EXPECTED_SWEEP_ARTIFACTS"
        )
    )
    facts["base.expected_counter_artifacts"] = str(
        _module_constant(
            "tests/test_never_fired_counter_census.py", "_EXPECTED_COUNTER_ARTIFACTS"
        )
    )
    facts["base.section3_rows"] = str(len(_ledger_row_ids(_REACHABILITY_HEADER)))
    unreachable = _ledger_row_ids(_UNREACHABLE_HEADER)
    facts["base.section4_candidates"] = str(len(unreachable))
    facts["base.section4_drops"] = str(
        sum(1 for row in unreachable if not row.startswith("~~"))
    )
    facts["base.section7_items"] = str(len(section_7_items()))
    facts["base.section7_unresolved"] = str(len(section_7_unresolved()))

    # -- T1, the fingerprint gate ----------------------------------------
    head = head_fingerprint()
    facts["t1.head_fingerprint"] = head[:16]
    facts["t1.newest_committed_sweep_fingerprint"] = sweep_fingerprint(
        HEAD_SWEEPS["dev"]
    )[:16]
    facts["t1.committed_json_carrying_head_fingerprint"] = str(
        sum(1 for name in committed_json() if head[:16] in _text(name))
    )
    facts["t1.freeze_declaration_constants"] = str(freeze_declaration_constants())
    # THE DOCUMENT IS WHERE THIS CHAIN TERMINATES, and these two rows are why.
    # `TheFingerprintDerivationReadsTheTreeTests` asserts the HASHER's input set equals the
    # TREE's, re-derived by `_tree_side_hashed_inputs`. That leaves the enumerator itself as
    # a single point of failure: MEASURED, a hasher and an enumerator that both skip
    # `tree.rs` agree with each other and the equality passes at a clean OK once Appendix A
    # is reseated -- the same defect one frame further out for the third time. Recording the
    # inventory HERE breaks it, because the document is not code: a shrunken enumerator
    # disagrees with these rows even when the hasher shrinks with it. Declared coupling, and
    # a free one -- anything that changes this inventory moves the fingerprint too, so the
    # author is already editing Appendix A in the same change.
    hashed = _tree_side_hashed_inputs()
    # Under `src/`, not "parent is named src": `crate_sources()` globs RECURSIVELY, so a
    # source added at `src/foo/bar.rs` would be counted by the hasher and missed here, and
    # the row would go stale in the one direction nothing else notices.
    crate_src = REPO / "rust" / "pokezero-search" / "src"
    facts["t1.hashed_crate_sources"] = str(
        len([
            path for path in hashed
            if path.suffix == ".rs" and path.is_relative_to(crate_src)
        ])
    )
    facts["t1.hashed_input_files"] = str(len(hashed))

    # -- T2, G8's second remainder ---------------------------------------
    sites = residual_disjoint_band_sites()
    facts["t2.residual_disjoint_bands_call_sites"] = str(len(sites))
    hp_sites = [line for line, ceiling in sites if ceiling == "defender_active.hp"]
    max_sites = [line for line, ceiling in sites if ceiling == "i16::MAX"]
    facts["t2.hp_ceiling_sites"] = str(len(hp_sites))
    facts["t2.i16max_ceiling_sites"] = str(len(max_sites))
    facts["t2.hp_ceiling_site_lines"] = ", ".join(str(line) for line in hp_sites)
    facts["t2.i16max_ceiling_site_lines"] = ", ".join(str(line) for line in max_sites)
    gates = [hunk for hunk in split_hunks() if hunk["adds_split_gate"]]
    facts["t2.split_hunks"] = str(len(gates))
    facts["t2.split_hunks_touching_an_hp_ceiling_site"] = str(
        sum(1 for hunk in gates if hunk["touches_hp_site"])
    )
    census = _artifact(G8_CENSUS)["census"]
    facts["t2.first_remainder_off_fan_bands"] = (
        f"{census['representative_is_OFF_fan']} of {census['windows_with_a_survive_band']}"
    )
    facts["t2.first_remainder_off_fan_fraction"] = (
        f"{census['off_fan_fraction_of_bands'] * 100:.3f} %"
    )

    # -- T3, G33b's speed-tie arm ----------------------------------------
    g33b = _artifact(G33B_CENSUS)
    combined = g33b["all_windows_combined"]
    facts["t3.games"] = str(sum(w["games"] for w in g33b["per_window"].values()))
    facts["t3.predicate_calls"] = str(combined["instructions_seen"])
    facts["t3.speed_ties"] = str(combined["by_order"]["tie"])
    facts["t3.speed_ties_order_le_10"] = str(
        combined["by_arm_and_order"]["order_le_10|tie"]
    )
    facts["t3.speed_ties_perish"] = str(combined["by_arm_and_order"]["perish|tie"])
    tie_rows = combined["tie_arm"]["rows"]
    facts["t3.speed_ties_with_a_leftovers_winner"] = str(
        sum(1 for row in tie_rows if row["winner_item_lefto"])
    )
    facts["t3.order_le_10_ties_carrying_a_winner_heal"] = str(
        sum(
            1
            for row in tie_rows
            if row["arm"] == "order_le_10" and row["winner_heals_before"] > 0
        )
    )
    facts["t3.tie_refusal_line"] = (
        f"{EVENTS}:{_anchor(EVENTS, '_ => NO_TRUNCATION,')}"
    )

    # -- T4, G33c ---------------------------------------------------------
    repro = g33c_repro()
    facts["t4.boundary"] = f"{repro['seed']}/{repro['step']}"
    observed, engine = repro["divergence_class"].split(":", 1)[1].split("|")
    facts["t4.observed_component"] = observed
    facts["t4.engine_component"] = engine
    percentages = re.findall(r"pct=([0-9.]+):", " ".join(repro["branch_misses"]))
    facts["t4.branch_miss_pct"] = percentages[0]
    references = leftovers_truncated_references()
    facts["t4.leftovers_truncated_references"] = str(len(references))
    consumer = _anchor(
        EVENTS, "if active.item == Items::LEFTOVERS && !leftovers_truncated[i] {"
    )
    facts["t4.leftovers_truncated_consumers"] = str(len(references) - 1)
    facts["t4.leftovers_truncated_consumer_line"] = f"{EVENTS}:{consumer}"
    diverged, measured = wide_shard_totals()
    facts["t4.wide_census_divergent_rows"] = f"{diverged} of {measured}"
    # THE SECOND ROW OF THE SAME CLASS, which is what deriving found and transcribing did
    # not: C152 and the ledger's G33c cell both cite `1000513/121` alone.
    siblings = [
        f"{row['seed']}/{row['step']}"
        for shard in WIDE_SHARDS
        for row in _artifact(shard)["repros"]
        if row["divergence_class"] == repro["divergence_class"]
    ]
    facts["t4.heal_mismatch_rows_in_the_wide_census"] = str(len(siblings))
    facts["t4.undiagnosed_sibling_rows"] = ", ".join(
        sorted(row for row in siblings if row != facts["t4.boundary"])
    )

    # -- T5, the single-seat population ----------------------------------
    for window, path in HEAD_SWEEPS.items():
        sweep = _artifact(path)
        single = sweep["counters"]["skip:single_seat_boundary"]
        total = sweep["boundaries_full_round"] + single
        facts[f"t5.{window}_single_seat_boundaries"] = str(single)
        facts[f"t5.{window}_single_seat_fraction"] = f"{single / total * 100:.3f} %"
    facts["t5.subkeyed_single_seat_counters_in_corpus"] = str(
        len(single_seat_subkeys_in_corpus())
    )

    # -- T6, §4's re-adjudication ----------------------------------------
    c154 = _artifact(C154_ARTIFACT)
    verdicts = c154["verdicts"]
    facts["t6.verdicts_unreachable"] = str(
        sum(1 for row in verdicts.values() if row["verdict"] == "UNREACHABLE_TRACED")
    )
    facts["t6.verdicts_withdrawn"] = str(
        sum(
            1
            for row in verdicts.values()
            if row["verdict"] == "WITHDRAWN_BEFORE_THIS_PASS"
        )
    )
    adjudicated = {
        name: row
        for name, row in verdicts.items()
        if row["verdict"] == "UNREACHABLE_TRACED"
    }
    statuses = [row["ledger_reason_status"] for row in adjudicated.values()]
    facts["t6.reasons_false"] = str(statuses.count("FALSE"))
    facts["t6.reasons_incomplete"] = str(statuses.count("INCOMPLETE"))
    facts["t6.reasons_sound"] = str(statuses.count("SOUND"))
    facts["t6.reasons_corrected"] = str(
        statuses.count("FALSE") + statuses.count("INCOMPLETE")
    )
    facts["t6.rows_foreclosed_over_section_4_population_only"] = ", ".join(
        sorted(
            (
                name
                for name, row in verdicts.items()
                if row.get("foreclosure") == "RANDBATS_POPULATION"
            ),
            key=lambda name: int(name[1:]),
        )
    )
    facts["t6.human_readings"] = str(c154_human_readings())
    facts["t6.pool_showdown_commit"] = c154["pool"]["showdown_commit"]
    facts["t6.workflow_steps_checking_out_showdown"] = str(
        len(workflow_showdown_checkouts())
    )
    facts["t6.section_4_population_anchor"] = (
        f"{LEDGER}:{_anchor(LEDGER, 'cannot be reached in gen3 randbats')}"
    )

    # -- how much of the ledger has actually been re-examined ---------------
    # C152 §7.1's own sixth open item was "G0 and every other §3 row C152 did not touch".
    # The first revision of this register absorbed that into §1's "terminal is not the
    # absence of gaps" reading and did not carry the NUMBER, which is the figure an owner
    # asking how much of the ledger has been looked at actually needs.
    ledger_rows = _ledger_rows(_REACHABILITY_HEADER)
    touched = sorted(
        {
            row_cells(line)[0].strip("*~ ")
            for _, line in ledger_rows
            if any(tag in line for tag in ("C152", "C153", "C154"))
        },
        key=lambda name: (name[0], int(re.sub(r"\D", "", name)), name),
    )
    # THE IDS AND THE COUNT. ⚠ The ids are NOT here because a mutation defeated the count
    # -- a previous revision said so and review measured it false. The marker predicate is
    # `any(tag in line)`, which is insensitive to how MANY markers a row carries by
    # construction, so a row-id list is exactly as blind to that as a row count; and
    # stripping every marker from a one-marker row moves the count too. What the ids catch
    # and the count cannot is a SWAP -- one row leaving the re-examined set while another
    # joins, which holds the count at 9 and changes the membership (battery B49). That, and
    # telling an owner WHICH rows, is why both are published.
    facts["scope.section3_rows_touched_since_c138"] = ", ".join(touched)
    facts["scope.section3_rows_touched_count"] = str(len(touched))
    facts["scope.section3_rows_untouched_since_c138"] = str(
        len(ledger_rows) - len(touched)
    )
    unreachable_rows = _ledger_rows(_UNREACHABLE_HEADER)
    facts["scope.section4_rows_corrected_by_c154"] = str(
        sum(1 for _, line in unreachable_rows if "C154" in line)
    )

    # -- what T1's agent half costs ------------------------------------------
    # Named as work in the first revision and not costed, which for the biggest line item
    # on the page is the same defect as an uncounted count. Taken from the two committed
    # sweeps' own recorded wall time rather than estimated.
    seconds = sum(_artifact(path)["elapsed_seconds"] for path in HEAD_SWEEPS.values())
    facts["cost.head_resweep_games"] = str(
        sum(_artifact(path)["games"] for path in HEAD_SWEEPS.values())
    )
    facts["cost.head_resweep_minutes"] = f"{seconds / 60:.0f}"

    # -- the register's own shape, so its summary sentences cannot go stale ---
    facts["rule.two_window_zero_paragraphs"] = str(len(two_window_zero_paragraphs()))

    # -- the standing bars -------------------------------------------------
    for window, path in HEAD_SWEEPS.items():
        sweep = _artifact(path)
        facts[f"bar.support_gated_{window}"] = (
            f"{sweep['gating_support_based'] / sweep['boundaries_measured'] * 100:.3f} %"
        )
    h8 = _artifact(H8_CENSUS)["boundaries_whose_accept_depended_on_the_window"]
    for window in ("dev", "holdout"):
        facts[f"bar.roll_window_{window}"] = str(h8[window])
        facts[f"bar.roll_window_{window}_fraction"] = (
            f"{h8[f'{window}_fraction_of_measured'] * 100:.3f} %"
        )

    # -- the per-divergence power limit -----------------------------------
    bounds = _artifact(C153_CENSUS)["statistical_bounds"]["combined"]
    classified = bounds["classified_divergences"]
    facts["power.classified_divergences"] = str(classified)
    facts["power.per_divergence_upper_95"] = (
        f"{bounds['rule_of_three_per_divergence_upper_95'] * 100:.3f} %"
    )
    facts["power.one_in_n_divergences"] = str(int(classified / 3))
    facts["power.boundaries_per_classified_divergence"] = str(
        round(bounds["boundaries_measured"] / classified)
    )
    # The rule of three is 3/N, so a tenfold tighter bound is exactly tenfold the trials.
    # Derived that way rather than by dividing the rounded percentage, which would make the
    # figure an artefact of the rounding in the row above it.
    facts["power.divergences_for_a_tenfold_tighter_bound"] = str(classified * 10)

    return facts


# ---------------------------------------------------------------------------
# 1. The derivations are not vacuous.
# ---------------------------------------------------------------------------


class TheDerivationsReadSomethingTests(unittest.TestCase):
    """Anti-vacuity. Every absence and every count below is a loop; a loop over nothing
    passes. These are the controls that make the zeroes mean something."""

    def test_the_register_exists_and_carries_both_machine_read_tables(self) -> None:
        found = tables(_text(REGISTER))
        self.assertGreaterEqual(len(found), 2, "the register lost its tables")
        self.assertEqual(len(register_items()), len(EXPECTED_ITEMS))
        self.assertGreater(len(register_facts()), 40)

    def test_the_committed_json_corpus_is_the_one_the_censuses_use(self) -> None:
        corpus = committed_json()
        self.assertEqual(
            len(corpus),
            int(derive()["base.expected_counter_artifacts"]),
            "the JSON walk here and `counter_artifacts()`'s must see the same corpus, or "
            "`t1.committed_json_carrying_head_fingerprint` is a scan of the wrong set",
        )
        self.assertIn(C154_ARTIFACT, corpus)
        self.assertIn(G33B_CENSUS, corpus)

    def test_the_freeze_scan_finds_the_constant_that_does_exist(self) -> None:
        # `t1.freeze_declaration_constants` is 0. A name-level scan that found NOTHING
        # would report 0 as well, so the control is that the sibling declaration -- the
        # window's ratification, which DOES exist -- is found by the same scan.
        names = differential_constant_names()
        self.assertIn("OWNER_RATIFIED", names)
        self.assertIn("RATIFIED_SWEEP_PRECONDITION", names)
        self.assertGreater(len(names), 20)

    def test_the_workflow_scan_sees_the_comment_it_excludes(self) -> None:
        # `t6.workflow_steps_checking_out_showdown` is 0 because the one mention is a
        # comment. If the scan stopped reading the file it would also report 0.
        mentions = [
            line
            for line in _text(WORKFLOW).splitlines()
            if "pokemon-showdown" in line
        ]
        self.assertEqual(len(mentions), 1)
        self.assertTrue(mentions[0].strip().startswith("#"))
        self.assertEqual(workflow_showdown_checkouts(), [])

    def test_the_single_seat_subkey_scan_finds_the_unkeyed_counter(self) -> None:
        # The absence is `skip:single_seat_boundary:<reason>`. The BARE counter is present
        # in the corpus, so a scan that read nothing is distinguishable from a real zero.
        self.assertEqual(single_seat_subkeys_in_corpus(), [])
        carriers = [
            name
            for name in committed_json()
            if '"skip:single_seat_boundary"' in _text(name)
        ]
        self.assertGreater(len(carriers), 10, carriers)

    def test_the_patch_reader_finds_the_definition_and_excludes_it(self) -> None:
        # Four CALLS, one DEFINITION. A reader that took every line mentioning the name
        # would report five sites and put the definition's fourth line in the ceiling
        # column, which is a wrong answer that looks like a right one.
        text = _text(THRESHOLD_PATCH)
        self.assertEqual(text.count("residual_disjoint_bands("), 5)
        self.assertEqual(len(residual_disjoint_band_sites()), 4)


# ---------------------------------------------------------------------------
# 2. The item inventory.
# ---------------------------------------------------------------------------


class TheItemInventoryIsExactTests(unittest.TestCase):
    def test_the_ids_are_exact_and_in_order(self) -> None:
        ids = tuple(cells[0] for cells in register_items())
        self.assertEqual(
            ids,
            EXPECTED_ITEMS,
            "an item was added, dropped, renumbered or reordered. Ids are PERMANENT: a "
            "discharged item stays in place with its status changed. Renumbering is what "
            "made three reconstructions of this list incomparable.",
        )

    def test_the_item_count_is_stated_and_re_derived(self) -> None:
        stated = re.search(
            r"\*\*Fourteen items — (\d+) under G2, (\d+) under G1\.", _text(REGISTER)
        )
        self.assertIsNotNone(
            stated,
            "the register's headline item count is gone or reworded; this pin reads it "
            "and must be updated with it rather than silently passing",
        )
        gates = [cells[2] for cells in register_items()]
        self.assertEqual(
            (int(stated.group(1)), int(stated.group(2))),
            (gates.count("G2"), gates.count("G1")),
        )
        self.assertEqual(len(gates), 14)

    def test_the_status_tally_sentence_is_re_derived(self) -> None:
        # Added in review. `EXPECTED_STATUS` already forces an author through the column
        # when a status flips; it does NOT force them through the summary sentence a reader
        # meets first, which is the same gap `test_the_stated_row_count_matches_the_rows`
        # closes in the ledger after that sentence went stale three times.
        stated = re.search(
            r"Status tally: (\d+) OPEN, (\d+) DISCHARGED-IN-SCOPE,\s*\n?(\d+) STANDING",
            _text(REGISTER),
        )
        self.assertIsNotNone(
            stated,
            "§2's status-tally sentence is gone or reworded; this pin reads it and must be "
            "updated with it rather than silently passing",
        )
        statuses = [cells[3] for cells in register_items()]
        self.assertEqual(
            tuple(int(stated.group(n)) for n in (1, 2, 3)),
            (
                statuses.count("OPEN"),
                statuses.count("DISCHARGED-IN-SCOPE"),
                statuses.count("STANDING"),
            ),
        )

    def test_the_derivation_delta_count_is_re_derived(self) -> None:
        # §5 says how many statements did not survive the derivation, in two places. Both
        # come from the list itself. Added in review for the same reason as the tally above.
        text = _text(REGISTER)
        body = text.split("## 5. What the derivation changed", 1)[1]
        body = body.split("\n## 6.", 1)[0]
        entries = re.findall(r"(?m)^(\d+)\. \*\*", body)
        self.assertEqual(
            [int(n) for n in entries],
            list(range(1, len(entries) + 1)),
            "§5's list is not numbered consecutively from 1",
        )
        words = {6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}
        self.assertIn(len(entries), words, f"§5 has {len(entries)} entries; extend `words`")
        word = words[len(entries)]
        self.assertRegex(body, rf"\*\*{word}\*\* entries")
        self.assertRegex(
            text, rf"and \*\*{word.lower()}\*\*\s*\nstatements did not survive"
        )

    def test_every_status_and_actor_is_from_the_closed_vocabulary(self) -> None:
        for cells in register_items():
            with self.subTest(item=cells[0]):
                self.assertIn(cells[3], STATUS_VOCABULARY)
                self.assertIn(cells[4], ACTOR_VOCABULARY)
                self.assertIn(cells[5], PIN_VOCABULARY)

    def test_every_status_matches_the_pinned_disposition(self) -> None:
        # BOTH directions matter. Flipping T2 to DISCHARGED-IN-SCOPE manufactures
        # terminality; flipping T6 to OPEN manufactures openness. This pin refuses both.
        measured = {cells[0]: cells[3] for cells in register_items()}
        self.assertEqual(measured, EXPECTED_STATUS)

    def test_every_actor_matches_the_pinned_assignment(self) -> None:
        # The actor column has been wrong before at program level: item 13 was believed
        # blocked on the owner when the owner had already ratified. The assignment is
        # pinned so a change to it is a diff someone reads.
        measured = {cells[0]: cells[4] for cells in register_items()}
        self.assertEqual(measured, EXPECTED_ACTOR)

    def test_every_pin_verdict_matches_the_pinned_assignment(self) -> None:
        # An item whose disposition is a reading must SAY `derived + reading`. Downgrading
        # one to `derived` is the claim that a pin covers a judgement, which is the defect
        # C154 §5 exists to enumerate.
        measured = {cells[0]: cells[5] for cells in register_items()}
        self.assertEqual(measured, EXPECTED_PIN)
        readings = sorted(
            key for key, value in EXPECTED_PIN.items() if value == "derived + reading"
        )
        section = _text(REGISTER).split("**It cannot cover a reading.**", 1)[1]
        section = section.split("\n\n", 1)[0]
        for identifier in readings:
            with self.subTest(item=identifier):
                self.assertIn(
                    identifier,
                    section,
                    "§6's list of what the pin cannot cover omits an item whose own row "
                    "says its disposition is a reading",
                )

    def test_the_owner_has_already_ratified_the_window(self) -> None:
        # The evidence for T11's "the owner has already acted". Read from the guard's own
        # constant rather than from prose about it, and NOT modified by this change.
        text = _text("scripts/engine_transition_differential.py")
        self.assertIn('OWNER_RATIFIED = ("19,300,000-19,300,199", "scott, 2026-08-08")', text)
        self.assertIn(
            "ledger terminal AND engine fingerprint declared frozen for the claim", text
        )

    def test_every_item_has_its_own_section(self) -> None:
        text = _text(REGISTER)
        for identifier in EXPECTED_ITEMS:
            with self.subTest(item=identifier):
                self.assertRegex(
                    text,
                    rf"(?m)^### {identifier} — ",
                    f"{identifier} is in the table with no section of its own; an item "
                    "without evidence, scope and an actor is a line, not a register entry",
                )


class TheSection7CoverageIsExactTests(unittest.TestCase):
    def test_the_register_carries_exactly_the_unresolved_section_7_items(self) -> None:
        # THE DERIVATION THAT TOOK SIX ITEMS TO FOURTEEN. The reconstructions were built
        # from a DELTA -- what C152, C153 and C154 opened -- and carried one §7 item. The
        # ledger's §7 does not mark eight of its eleven RESOLVED.
        self.assertEqual(sorted(SECTION_7_ITEMS.values()), section_7_unresolved())

    def test_a_resolved_section_7_item_cannot_keep_a_register_slot(self) -> None:
        resolved = {n for n, _ in section_7_items()} - set(section_7_unresolved())
        self.assertEqual(resolved, {2, 3, 4})
        self.assertFalse(resolved & set(SECTION_7_ITEMS.values()))

    def test_each_mapped_item_names_its_section_7_number(self) -> None:
        text = _text(REGISTER)
        for identifier, number in SECTION_7_ITEMS.items():
            with self.subTest(item=identifier):
                self.assertRegex(text, rf"(?m)^\| {identifier} \| §7 item {number} —")


# ---------------------------------------------------------------------------
# 3. Appendix A.
# ---------------------------------------------------------------------------


class TheAppendixIsReDerivedTests(unittest.TestCase):
    def test_the_key_set_is_exact_in_both_directions(self) -> None:
        derived = derive()
        stated = register_facts()
        self.assertEqual(
            sorted(stated),
            sorted(derived),
            "Appendix A's key set and the derivation's disagree. A key derived here and "
            "absent from the document is an unstated fact; a key in the document that "
            "nothing derives is an unpinned one, which is the defect this register exists "
            "to remove.",
        )

    def test_every_value_re_derives(self) -> None:
        derived = derive()
        stated = register_facts()
        for key in sorted(derived):
            with self.subTest(key=key):
                self.assertEqual(
                    stated.get(key),
                    derived[key],
                    f"Appendix A states {key} = {stated.get(key)!r}; the tree gives "
                    f"{derived[key]!r}. Re-derive it; never edit the document to match a "
                    "figure carried from a message. AND RE-DERIVE ON THE MERGE TREE: CI "
                    "gates refs/pull/<n>/merge, so `git merge origin/main` (never rebase) "
                    "FIRST, then re-run -- a value derived on the branch alone is stale "
                    "the moment main moves, which has cost this row three repair commits. "
                    "For t1.head_fingerprint the derivation is "
                    "`python scripts/engine_build_fingerprint.py --print`, and T1's table "
                    "wants a row naming the input that moved it.",
                )

    def test_the_appendix_is_sorted_so_a_diff_is_readable(self) -> None:
        keys = list(register_facts())
        self.assertEqual(keys, sorted(keys))


class TheFingerprintGateIsOpenTests(unittest.TestCase):
    def test_the_head_fingerprint_differs_from_every_committed_sweeps(self) -> None:
        head = head_fingerprint()
        for window, path in HEAD_SWEEPS.items():
            with self.subTest(window=window):
                self.assertNotEqual(sweep_fingerprint(path), head)

    def test_no_committed_artifact_carries_the_head_fingerprint(self) -> None:
        head = head_fingerprint()[:16]
        carriers = [name for name in committed_json() if head in _text(name)]
        self.assertEqual(
            carriers,
            [],
            "an artifact now carries the head fingerprint. That is progress on T1 and it "
            "must be recorded here rather than absorbed: update the register in the same "
            "change.",
        )

    def test_nothing_in_the_tree_can_declare_the_fingerprint_frozen(self) -> None:
        self.assertEqual(freeze_declaration_constants(), 0)


def _strip_rust_comments(source: str) -> str:
    """`/* */` and `//` removed, so a commented-out `mod x;` is not read as a declaration."""

    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", source, flags=re.S))


def _tree_side_hashed_inputs(root: Path | None = None) -> list[Path]:
    """Every file the fingerprint SHOULD read, enumerated without asking the hasher.

    ⚠ THIS FUNCTION EXISTS BECAUSE ITS ABSENCE DEFEATED THE PREVIOUS REVISION. The sandbox
    used to be populated from `crate_sources()` / `cargo_inputs()` /
    `build_metadata_inputs()` -- the hasher's OWN accounting -- so a hasher that skipped
    `tree.rs`, or took `crate_sources()[:6]`, produced a sandbox missing exactly the files
    it had stopped reading. The independent re-derivation then globbed that sandbox, found
    the same short list, and the "exact set equality" agreed with itself. Both mutants
    SURVIVED at a clean OK after Appendix A was reseated. A control populated by the thing
    it controls is not a control.

    So the enumeration starts from the REAL repo and shares no code with
    `engine_build_fingerprint`: the manifest parsed for its own non-comment lines, the
    crate's `src/**/*.rs` globbed directly, and the four build files globbed by name.
    """

    base = REPO if root is None else root
    manifest = base / "third_party" / "poke-engine-gen3-patches.txt"
    found = {manifest, base / "third_party" / "poke-engine-base-source.json"}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            found.add(base / "third_party" / name)
    crate = base / "rust" / "pokezero-search"
    found |= set((crate / "src").rglob("*.rs"))
    found |= {
        path
        for path in crate.rglob("*")
        if path.is_file()
        and path.name in {"Cargo.toml", "Cargo.lock", "build.rs", "pyproject.toml"}
        and not {".git", "target"}.intersection(path.relative_to(crate).parts)
    }
    return sorted(found)


@contextlib.contextmanager
def _fingerprint_sandbox():
    """A byte-copy of every hashed input, with the hasher pointed at it.

    Repo-relative layout is PRESERVED, because `compute_fingerprint` feeds
    `path.relative_to(REPO_ROOT)` into the digest for the native inputs -- a flat copy
    would compute a different hash for identical bytes and the negative control below
    would fail for the wrong reason.
    """

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        for path in _tree_side_hashed_inputs():
            target = root / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        module = engine_build_fingerprint
        saved = {
            name: getattr(module, name)
            for name in ("REPO_ROOT", "PATCH_LIST", "BASE_SOURCE", "CRATE_SRC", "CRATE_ROOT")
        }
        module.REPO_ROOT = root
        module.PATCH_LIST = root / "third_party" / "poke-engine-gen3-patches.txt"
        module.BASE_SOURCE = root / "third_party" / "poke-engine-base-source.json"
        module.CRATE_SRC = root / "rust" / "pokezero-search" / "src"
        module.CRATE_ROOT = root / "rust" / "pokezero-search"
        try:
            yield root
        finally:
            for name, value in saved.items():
                setattr(module, name, value)


class TheFingerprintDerivationReadsTheTreeTests(unittest.TestCase):
    """`head_fingerprint` must READ THE TREE, and nothing above asserts that it does.

    Appendix A's `t1.head_fingerprint` is the one row that moves on every crate-touching
    commit, so it is the row an author reaches for when the pin is inconvenient -- and the
    module's 53-mutation battery cannot see the reach. Every one of those 53 targets the
    DOCUMENT or the TREE; none targets the DERIVATION. MEASURED, not argued: replacing the
    body of `head_fingerprint` with `return register_facts()["t1.head_fingerprint"] +
    "0" * 48` leaves all four T1 tests, and the whole module, at a clean `OK` **on a tree
    with a mutated `rust/pokezero-search/src/tree.rs`** -- the exact condition the pin
    exists to redden. The CI step's `Ran N tests` and `^OK$` guards both pass too, because
    the count is unchanged and the count is what they read.

    So the pin was one line from being permanently satisfied.

    ⚠ AND THE FIRST REVISION OF THIS CLASS DID NOT CLOSE IT, which independent review
    established with eight surviving mutants and is recorded because the miss is the
    instructive part. Two defects, both of them this class's own claim landing on itself:

      1. It pinned `head_fingerprint` and nothing else, so the tautology simply moved one
         frame up. `derive()`'s `t1.head_fingerprint` line re-pointed at
         `register_facts()`, or the comparison loop taught to `continue` on that one key,
         each left the module at a clean 46/OK **on an edited crate source**. A guard that
         pins one seam of a three-seam path names the seam it pins.
      2. It claimed "SENSITIVITY to each hashed input class" while probing TWO of the five
         -- crate sources and patch bodies. Nothing touched the manifest bytes, the sdist
         digest pin, `cargo_inputs()` or `build_metadata_inputs()`, so `cargo = []`,
         `build_metadata = []`, and dropping either `BASE_SOURCE`'s or `PATCH_LIST`'s bytes
         from the digest were each ONE LINE and green. That is the unscoped-negative defect
         this whole register exists to prosecute, committed in the guard against it.

    Both are closed below, and the sensitivity is now EXHAUSTIVE rather than enumerated:
    every file the hasher reads is perturbed in turn, derived from `build_inputs()` so a
    newly added input class is covered the day it lands and cannot be forgotten. Because
    that probe set comes from the hasher's own accounting, it is floored INDEPENDENTLY by
    name -- otherwise deleting a class from `build_inputs()` shrinks the probe set with it
    and the loop goes quietly green over a smaller world.

    All of it drives `head_fingerprint()` and `derive()` -- the functions the register is
    actually read through -- rather than `compute_fingerprint`, because the seams the
    mutants cut are between them.
    """

    #: Floors the probe set BY NAME, independently of the hasher's own accounting. Without
    #: this, `build_metadata_inputs() -> []` shrinks both the digest and the probe set
    #: together and the exhaustive loop below certifies a smaller world in silence.
    REQUIRED_INPUT_NAMES = frozenset(
        {
            "poke-engine-gen3-patches.txt",
            "poke-engine-base-source.json",
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            "pyproject.toml",
        }
    )

    @staticmethod
    def _hashed_inputs() -> list[Path]:
        """Every file `compute_fingerprint` reads.

        `PATCH_LIST` is added explicitly: `build_inputs()` omits it (it contributes the
        ordered patch NAMES, and its bytes are hashed by `compute_fingerprint` directly),
        so a probe set taken from `build_inputs()` alone leaves the manifest unprobed --
        which is exactly one of the two classes review found unprobed.
        """

        seen: dict[Path, None] = {}
        for path in [engine_build_fingerprint.PATCH_LIST] + engine_build_fingerprint.build_inputs():
            seen.setdefault(path, None)
        return list(seen)

    @staticmethod
    def _perturb(blob: bytes, path: Path) -> bytes:
        """One byte-level change that keeps the file PARSEABLE.

        `poke-engine-base-source.json` is `json.loads`-ed by `compute_fingerprint` for its
        payload, so appending a byte would raise instead of moving the digest and the probe
        would pass for the wrong reason. Re-indenting changes the bytes and keeps the JSON.
        """

        if path.suffix == ".json":
            return json.dumps(json.loads(blob.decode()), indent=4).encode()
        # ⚠ SPLICED INTO THE MIDDLE, not appended. Review round 5: every content probe in
        # this class perturbed a file's HEAD or its TAIL, so a one-line proxy --
        # `sha256(blob[:512] + str(st_size))` -- survived all eleven of them reseated and hid
        # an `f32` -> `f64` edit in the middle of `tree.rs`. That is B73's own sentence one
        # frame out: a probe family that all perturb the same REGION is blind in the same
        # place, exactly as a probe family that all change LENGTH was blind to a size proxy.
        middle = len(blob) // 2
        if path.name == PATCH_MANIFEST_NAME:
            # The manifest is PARSED FOR FILENAMES, so a byte spliced mid-line renames a
            # patch and `compute_fingerprint` raises instead of moving -- the probe would then
            # error rather than measure. Insert a whole COMMENT LINE at the newline nearest
            # the middle: middle bytes change, every filename survives intact. Found by
            # running it; the first mid-file version raised `patch listed but missing`.
            boundary = blob.rfind(b"\n", 0, middle) + 1
            return blob[:boundary] + b"# sensitivity probe\n" + blob[boundary:]
        return blob[:middle] + b"\n" + blob[middle:]

    def test_the_sandbox_reproduces_the_head_fingerprint_exactly(self) -> None:
        # The negative control, and it is load-bearing twice over: it proves the copy
        # captures every input class (a missed one changes the digest), and it proves the
        # sandbox is what the two sensitivity tests below are perturbing.
        real = head_fingerprint()
        with _fingerprint_sandbox():
            self.assertEqual(
                head_fingerprint(),
                real,
                "the sandbox is not a byte-faithful copy of the hashed input set, so the "
                "probes below perturb something other than what ships. Most likely "
                "`build_inputs` gained a class this copy does not mirror.",
            )

    def test_the_hashed_input_set_is_exactly_what_the_tree_holds(self) -> None:
        """MEMBERSHIP EQUALITY against a re-derivation that shares no code with the hasher.

        ⚠ THE FIRST TWO REVISIONS OF THIS GUARD WERE FLOORS WITH ~20 FILES OF SLACK, and
        review walked straight through them: `> 5` `.rs` against 11 actual, `> 50` `.patch`
        against 74, `> 70` total against 91. So `crate_sources()` skipping `tree.rs`,
        `crate_sources()[:6]` and `patch_files()[:60]` all SURVIVED green -- the exhaustive
        loop below derives its probe set from the hasher, so a hasher that reads less is
        probed less and reports success over a smaller world. A floor whose margin exceeds
        the defect it is meant to catch is decoration.

        ⚠ AND THE FIRST FIX FOR THAT STILL LET TWO OF THEM THROUGH, which is the sharper
        lesson: the equality was exact, but the SANDBOX was populated from the hasher's own
        accounting, so a hasher that skipped `tree.rs` produced a sandbox missing `tree.rs`
        and the independent glob agreed with the short list. Both survived at a clean OK
        after Appendix A was reseated. `_tree_side_hashed_inputs` now enumerates from the
        REAL repo and shares no code with `engine_build_fingerprint`, and it populates the
        sandbox too -- so the copy is a copy of the TREE rather than of the hasher's opinion
        of it. It costs nothing to maintain: adding a crate source changes both sides
        together, and the fingerprint moves anyway, so no new churn is introduced.
        """

        with _fingerprint_sandbox() as root:
            expected = {path.resolve() for path in _tree_side_hashed_inputs(root)}
            got = {path.resolve() for path in self._hashed_inputs()}
            self.assertEqual(
                sorted(str(p.relative_to(root.resolve())) for p in got),
                sorted(str(p.relative_to(root.resolve())) for p in expected),
                "the set of files the hasher reads is not the set the tree holds. A file "
                "MISSING from the hasher is an input an engine can be built from without "
                "moving the fingerprint; a file EXTRA is a fingerprint that churns on "
                "something that cannot change the build.",
            )
            # Non-vacuity on the expected side itself, because a re-derivation that
            # collapsed to nothing would make the equality above trivially true.
            self.assertEqual(len(self.REQUIRED_INPUT_NAMES), 6)
            names = {path.name for path in expected}
            self.assertEqual(sorted(self.REQUIRED_INPUT_NAMES - names), [])
            self.assertEqual(
                len([p for p in expected if p.suffix == ".patch"]),
                int(register_facts()["base.patch_stack"]),
                "the patch stack the hasher reads is not the length Appendix A records. "
                "`base.patch_stack` is re-derived from the manifest by `derive()`, so this "
                "ties the hashed set to a figure the document is already policed on.",
            )

    def test_every_hashed_input_moves_the_head_fingerprint(self) -> None:
        # EXHAUSTIVE, not enumerated. The first revision probed a crate source and a patch
        # body and CLAIMED "each hashed input class"; that was two of five, and the three it
        # skipped were each one line from being dropped from the digest with nothing red.
        # Deriving the probe set from `build_inputs()` means a class added later is covered
        # the day it lands rather than the day someone remembers -- and the by-name floor
        # above is what stops the set being shrunk instead of satisfied.
        with _fingerprint_sandbox() as root:
            baseline = head_fingerprint()
            inputs = self._hashed_inputs()
            self.assertGreater(len(inputs), 70, "the hashed input set collapsed")
            for path in inputs:
                with self.subTest(input=str(path.relative_to(root))):
                    original = path.read_bytes()
                    try:
                        path.write_bytes(self._perturb(original, path))
                        self.assertNotEqual(
                            head_fingerprint(),
                            baseline,
                            f"editing {path.relative_to(root)} did not move the head "
                            "fingerprint, so the hasher reads it in name only. An engine "
                            "built from a different version of this file is invisible to "
                            "Appendix A's row and to the build stamp that shares its "
                            "derivation.",
                        )
                    finally:
                        path.write_bytes(original)
            self.assertEqual(
                head_fingerprint(),
                baseline,
                "the loop did not restore the sandbox, so a later probe measured a tree "
                "the earlier ones had already changed.",
            )

    @staticmethod
    def _workflow_filters() -> list[str]:
        """The `FILTERS` patterns the `changes` job's match loop actually receives.

        ⚠ REPLICATES THE SHELL'S SEMANTICS RATHER THAN A TIDIER VERSION OF THEM, and review
        round 5 is why. A first revision did `line.split("#", 1)[0].strip()`. The workflow
        (`engine-fidelity-gates.yml`, the `changes` job) does neither: it skips a pattern only
        when its FIRST character is `#`, and otherwise passes the line VERBATIM to
        `case "$file" in $pattern)`. A YAML block scalar cannot hold YAML comments, which is
        why the provenance comments live inside the block and why that first-character rule
        exists.

        So a trailing comment on the register's entry, or two extra spaces of indentation,
        produce a glob that matches NOTHING -- `fidelity=false`, `mass-gate` never runs on a
        register-only PR -- while the normalising parser reported the entry present and the
        module went green. Round 4 asked for the trailing-comment case to stay GREEN and I
        implemented that; the correct answer is RED, and the previous docstring advertised the
        hole as if it were the fix. Corrected in the direction of the shell, which is the only
        authority here.

        Block dedent follows YAML's literal-scalar rule -- the indentation of the block is
        taken from its least-indented line, so EXCESS indentation survives into the value
        exactly as the shell would see it.
        """

        lines = (
            REPO / ".github" / "workflows" / "engine-fidelity-gates.yml"
        ).read_text(encoding="utf-8").splitlines()
        starts = [i for i, line in enumerate(lines) if line.strip() == "FILTERS: |"]
        if len(starts) != 1:
            raise AssertionError(f"expected one `FILTERS: |` block, found {len(starts)}")
        head = starts[0]
        key_indent = len(lines[head]) - len(lines[head].lstrip())
        block: list[str] = []
        for line in lines[head + 1:]:
            if not line.strip():
                block.append("")
                continue
            if len(line) - len(line.lstrip()) <= key_indent:
                break
            block.append(line)
        body = [line for line in block if line.strip()]
        if not body:
            raise AssertionError("the FILTERS block is empty")
        base = min(len(line) - len(line.lstrip()) for line in body)
        patterns: list[str] = []
        for line in block:
            value = line[base:]
            if not value:                      # `[ -z "$pattern" ] && continue`
                continue
            if value.startswith("#"):          # `case "$pattern" in \#*) continue`
                continue
            patterns.append(value)
        return patterns

    @staticmethod
    def _tracked(pathspec: str) -> set[Path]:
        """What GIT says is in the tree. A third anchor, and deliberately not code."""

        done = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--", pathspec],
            capture_output=True,
            check=True,
        )
        return {
            (REPO / name).resolve()
            for name in done.stdout.decode("utf-8").split("\0")
            if name
        }

    def test_the_hashed_crate_inputs_are_the_files_git_tracks(self) -> None:
        """⚠ THE FIFTH FRAME, closed against an anchor that is neither code nor the document.

        Appendix A's inventory rows (B68) stop a hasher and an enumerator colluding, because
        the document disagrees with both. But MEASURED: an author who reseats the fingerprint
        AND both inventory rows -- i.e. one who updates every number the failure messages
        name -- lands a `crate_sources()` that skips `tree.rs` at a clean `Ran 51 tests, OK`.
        Nothing in the module or the document knows that 11 was the right number.

        `git ls-files` does. It is a third source, outside both, and it does not move when
        someone edits the register or the hasher. Against it the collusion is red no matter
        how many rows are reseated.

        SCOPE, because an unscoped anchor is this register's signature defect: this pins the
        GLOBBED classes -- crate sources and the four build files -- exactly, because for
        those "what the tree holds" and "what git tracks" are the same question. It does NOT
        pin the patch class to git: `third_party/` tracks MORE `.patch` files than the
        manifest lists (77 tracked, 74 listed), and not hashing the unlisted ones is correct
        -- the manifest is the definition of that class. What is pinned there instead is that
        every patch the manifest names is TRACKED, so the manifest cannot name a phantom.
        """

        hashed = {path.resolve() for path in self._hashed_inputs()}
        enumerated = {path.resolve() for path in _tree_side_hashed_inputs()}
        crate_src = (REPO / "rust" / "pokezero-search" / "src").resolve()

        with self.subTest(anchor="crate sources"):
            tracked = {
                path for path in self._tracked("rust/pokezero-search/src")
                if path.suffix == ".rs"
            }
            self.assertTrue(tracked, "git tracks no crate sources; the anchor read nothing")
            # ⚠ DERIVED FROM `lib.rs`'s OWN `mod` DECLARATIONS, not a hand-written list.
            # This arm exists because review round 3 filtered `_tracked` through
            # `crate_sources()` and the anchor then agreed with the hasher it checks. The
            # first fix was six literal names -- and round 4 walked through it by substituting
            # `priors.rs`, one of the five the list did not name, which is a file this
            # register's own commit table cites as having moved this row. Six of eleven is a
            # floor with five files of slack, the same defect as every other floor in this
            # battery. The crate declares its own module set, so DERIVE it: every `mod x;` in
            # `lib.rs`, plus `lib.rs` itself. Exact, self-maintaining, and it removes a
            # literal rather than adding five.
            declared = {
                f"{name}.rs"
                for name in re.findall(
                    # Visibility qualifier included: `priors` is `pub(crate) mod`, exactly the
                    # file round 4's literal list missed -- and a first version of this regex
                    # missed it too, which the derivation caught immediately.
                    r"(?m)^\s*(?:pub\s*(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
                    # COMMENTS STRIPPED FIRST. Review round 5: the raw regex false-trips on a
                    # `mod x;` inside `/* */` or behind `//`, demanding a source that does not
                    # exist. Stripping can only REMOVE candidates, so it cannot hide a real
                    # module -- a missing one still reddens the equality below.
                    _strip_rust_comments(
                        (REPO / "rust" / "pokezero-search" / "src" / "lib.rs").read_text("utf-8")
                    ),
                )
            } | {"lib.rs"}
            self.assertGreater(len(declared), 5, "`lib.rs` declares no modules; the derivation read nothing")
            # TOP-LEVEL sources only on the right. The nested layout (`mod tree;` served by
            # `src/tree/mod.rs` or `src/tree/helper.rs`) is legal Rust and would make a flat
            # name comparison spuriously red -- round 5 flagged that as future brittleness.
            # Nested sources are covered by the loop below instead, which asserts the property
            # that actually matters: the hasher must read them.
            self.assertEqual(
                sorted(declared),
                sorted(path.name for path in tracked if path.parent == crate_src),
                "the crate's declared module set and the top-level sources this anchor can "
                "see disagree. If the anchor is short, it has been narrowed -- most likely "
                "filtered through the hasher, which makes it agree with the thing it is "
                "supposed to check. If `lib.rs` is short, a tracked source is no longer "
                "compiled and should not be hashed.",
            )
            for path in sorted(item for item in tracked if item.parent != crate_src):
                self.assertIn(
                    path,
                    hashed,
                    f"{path.relative_to(REPO)} is a tracked crate source in a subdirectory "
                    "and the fingerprint does not hash it.",
                )
            for label, produced in (("hasher", hashed), ("enumerator", enumerated)):
                under = sorted(
                    str(path.relative_to(REPO))
                    for path in produced
                    if path.suffix == ".rs" and path.is_relative_to(crate_src)
                )
                self.assertEqual(
                    under,
                    sorted(str(path.relative_to(REPO)) for path in tracked),
                    f"the {label}'s crate-source set is not the set git tracks. A source "
                    "git knows about and the fingerprint does not is a file the engine is "
                    "built from without moving its identity -- and this is the one "
                    "assertion that stays red when the hasher, the enumerator AND every "
                    "Appendix A row have been reseated together.",
                )

        with self.subTest(anchor="build files"):
            for name in ("Cargo.toml", "Cargo.lock", "build.rs", "pyproject.toml"):
                tracked = self._tracked(f"rust/pokezero-search/{name}")
                self.assertEqual(
                    len(tracked), 1, f"git does not track exactly one {name} at the crate root"
                )
                self.assertLessEqual(
                    tracked,
                    hashed,
                    f"git tracks {name} at the crate root and the fingerprint does not hash "
                    "it. `pyproject.toml` carries the maturin feature flags, so this is the "
                    "shape where an extension is built with different features at the same "
                    "stamp.",
                )

        with self.subTest(anchor="the manifest names only tracked patches"):
            listed = {path.resolve() for path in engine_build_fingerprint.patch_files()}
            tracked = self._tracked("third_party/*.patch")
            self.assertTrue(listed, "the patch list resolved to nothing")
            self.assertEqual(
                sorted(str(p.relative_to(REPO)) for p in listed - tracked),
                [],
                "the patch manifest names files git does not track, so the patch stack this "
                "identity covers is not reconstructible from the checkout.",
            )

    def test_swapping_two_hashed_files_contents_moves_the_head_fingerprint(self) -> None:
        """WHICH file held WHICH bytes has to matter, not just the multiset of bytes.

        Review's NIT, promoted because it is cheap and it closes two mutants at once:
        dropping the repo-relative path from the native-input digest, and making that digest
        order-insensitive, both survive every other probe here -- the per-file sensitivity
        loop moves the hash either way, because it CHANGES bytes rather than moving them.
        Swapping two files' contents holds the byte multiset fixed and changes only the
        association, so it is red exactly when the identity is a set of (path, bytes) pairs
        and green when it has decayed to a bag of bytes. A crate whose `events.rs` and
        `tree.rs` had been transposed builds differently and would otherwise stamp the same.
        """

        with _fingerprint_sandbox() as root:
            baseline = head_fingerprint()
            sources = sorted((root / "rust" / "pokezero-search" / "src").rglob("*.rs"))
            self.assertGreater(len(sources), 1, "need two crate sources to swap")
            first, second = sources[0], sources[1]
            one, two = first.read_bytes(), second.read_bytes()
            self.assertNotEqual(one, two, "the two probe files are byte-identical; pick others")
            first.write_bytes(two)
            second.write_bytes(one)
            self.assertNotEqual(
                head_fingerprint(),
                baseline,
                f"transposing the contents of {first.name} and {second.name} left the head "
                "fingerprint unchanged, so the identity no longer records WHICH input held "
                "which bytes -- only that the bytes were present somewhere.",
            )

    def test_a_length_PRESERVING_edit_moves_the_head_fingerprint(self) -> None:
        """The digest must read BYTES, not a cheap proxy for them.

        ⚠ Review round 4's finding, and it is the sharpest one in this class's history because
        it defeats every probe above at once. `_perturb` only ever APPENDS -- so a
        `compute_fingerprint` that hashed `path.stat().st_size` instead of the file's contents
        satisfied the whole exhaustive loop, the transposition probe and the rename probe, and
        an `f32` -> `f64` edit to `tree.rs` was then invisible while the fingerprint sat
        unmoved. A size is not a hash, and until now nothing here could tell them apart.

        Two adjacent bytes that DIFFER are swapped, which changes the content and cannot
        change the length. A crate source is used rather than the JSON pin because
        `compute_fingerprint` parses that one, so a byte-level edit there would raise instead
        of moving the digest and the probe would pass for the wrong reason.
        """

        with _fingerprint_sandbox() as root:
            baseline = head_fingerprint()
            sources = sorted((root / "rust" / "pokezero-search" / "src").rglob("*.rs"))
            self.assertTrue(sources, "no crate sources in the sandbox to perturb")
            target = max(sources, key=lambda path: path.stat().st_size)
            original = target.read_bytes()
            self.assertGreater(len(original), 2048, "probe file too small to place offsets")
            # ⚠ ONE BYTE, XORed -- NOT a swap, and review round 6 is why. An adjacent swap
            # disturbs BOTH byte parities at once, so it cannot isolate a digest that reads
            # only one of them; every other probe here changes length or path, so a stride-2
            # digest moved under those too. Result: `path.read_bytes()[::2]` -- one line --
            # survived all 53 tests reseated and hid a real semantic edit
            # (`DAMAGE_BRANCH_DEPTH: u8 = 2 -> 3` in `tree.rs`) with the fingerprint
            # byte-identical. That is B73/B76's sentence one frame out again, with "residue
            # class" substituted for "region" and "length".
            #
            # So: a single byte, flipped, at BOTH PARITIES in each of three regions. A digest
            # that skips any residue class of offsets now fails on the offsets it skips, and
            # the subTest names the region and the parity so the report says which.
            for label, base in (
                ("head", 0),
                ("middle", len(original) // 2),
                ("tail", len(original) - 512),
            ):
                for parity in (0, 1):
                    offset = base + parity
                    with self.subTest(region=label, parity=("even" if offset % 2 == 0 else "odd")):
                        blob = bytearray(original)
                        blob[offset] ^= 0xFF
                        target.write_bytes(bytes(blob))
                        self.assertEqual(
                            target.stat().st_size,
                            len(original),
                            "the probe changed the file's LENGTH, so it no longer isolates "
                            "content from size and a size-based digest would still pass it.",
                        )
                        self.assertNotEqual(
                            head_fingerprint(),
                            baseline,
                            f"flipping the single byte at offset {offset} (the {label} of "
                            f"{target.name}) left the head fingerprint unchanged. The digest "
                            "is reading a PROXY for the bytes -- it skips this offset -- so "
                            "any edit landing there is invisible to this identity and to the "
                            "build stamp that shares its derivation.",
                        )
            target.write_bytes(original)

    def test_renaming_a_hashed_file_moves_the_head_fingerprint(self) -> None:
        """The digest must bind content to a NAME, not merely to a position in a sequence.

        This isolates the one mutant the transposition probe cannot see: dropping
        `digest.update(str(path.relative_to(REPO_ROOT)).encode())` from the native-input
        loop. That loop iterates in sorted-path order, so with the path gone the ORDER of
        the content hashes still encodes which file held what -- which is why a transposition
        moves the hash either way, and why B66a survived reseated with everything else here
        green. A pure RENAME is the discriminating case, and only if it PRESERVES the sort
        position: rename the last crate source to a name that still sorts last, and the
        sequence of content hashes is byte-identical while the path set is not.

        Without this, `compute_fingerprint` could be refactored into a bag-of-bytes identity
        that cannot tell `leaf.rs` from a copy of it saved under another name -- and the
        vendored tree is reconstructed BY NAME from the patch list, so names are load
        bearing here rather than incidental.
        """

        with _fingerprint_sandbox() as root:
            baseline = head_fingerprint()
            sources = sorted((root / "rust" / "pokezero-search" / "src").rglob("*.rs"))
            self.assertTrue(sources, "no crate sources in the sandbox to rename")
            last = sources[-1]
            renamed = last.with_name(f"{last.stem}zzzz{last.suffix}")
            last.rename(renamed)
            self.assertEqual(
                sorted((root / "rust" / "pokezero-search" / "src").rglob("*.rs"))[-1],
                renamed,
                "the rename did not preserve the sort position, so this probe is no longer "
                "isolating the path component -- it would also move the content order and "
                "pass for the wrong reason.",
            )
            self.assertNotEqual(
                head_fingerprint(),
                baseline,
                f"renaming {last.name} to {renamed.name} -- same bytes, same sort position, "
                "different path -- left the head fingerprint unchanged. The identity no "
                "longer binds content to a name, so it cannot distinguish two crate "
                "sources whose contents were saved under each other's names.",
            )

    def test_the_row_holds_when_read_through_none_of_this_modules_helpers(self) -> None:
        """⚠ THE THIRD ROUND OF ONE DEFECT, and the assertion that ends the class of it.

        Round 1 pinned `head_fingerprint` and the tautology moved to `derive()`. Round 2
        pinned `derive()` and the comparison loop, and review found the tautology had moved
        again -- to `register_facts()`, which EVERY document-side reader funnels through,
        and to `_text()` one frame further out. One added line there made all three of the
        previous round's "independent" assertions vacuous: document `65092feac14da111`,
        tree `15c2381f81e8f886`, module exit 0 at a clean `Ran 48 tests, OK` (the count at THAT
        revision; this module is larger now). The claim
        "the row reaches CI through THREE seams" was simply false, and incrementing it to
        four would have been the same mistake a fourth time.

        So this does not add a seam-specific pin. It reads the document with
        `Path.read_bytes` and a regex RIGHT HERE, and the tree with `compute_fingerprint`
        directly, bypassing every document-side reader in this module -- `_text`,
        `register_facts`, `_register_table`, `derive`, `head_fingerprint`, the appendix loop.
        Mutating any of those cannot reach this assertion.

        ⚠ AND THE CLAIM "SHARING NO HELPER" WAS FALSE AND IS RETRACTED. Review round 3
        showed it shares four things -- `REPO`, `REGISTER`, the module-level `re`, and
        `compute_fingerprint` -- and turned the first into a SIXTH frame: re-point `REGISTER`
        AND the literal below at a doctored copy of the register, and this assertion reads
        the doctored file while the published one stays stale. The old drift check APPROVED
        that, because requiring the two literals to agree forces them to move TOGETHER and
        pins neither to the file CI publishes. State the sharing rather than deny it:

          * `re` and `compute_fingerprint` are shared and that is the point -- the second IS
            the tree-side authority being checked, and shadowing the first is a mutation of
            the assertion itself, not a way around it.
          * `REGISTER` -- and only `REGISTER` -- is anchored OUTSIDE this module, below: to
            the `FILTERS` block the `changes` job feeds its match loop, which is what decides
            whether `mass-gate` runs on this document at all, and to git's index, so a
            doctored copy has to be committed to exist in a CI checkout, where the filter then
            names the wrong one.
          * ⚠ `REPO` IS **NOT** ANCHORED, and a previous revision of this list said it was.
            Re-pointing `REPO` alone is one line and leaves the published register stale. It
            is not CI-shippable -- `REPO` is derived from this file's own location, so any
            substitute is an absolute path that does not exist on a runner -- but that is a
            property of the environment, not a guard, and the claim is retracted rather than
            defended. Stated as a known reachable edit, which is what this register requires
            of everyone else.

        A separate bypass worth recording because it is NOT closed here: a module-level
        `load_tests` can drop this assertion while holding the printed count at its pinned
        value. `tests/test_unreachable_readjudication.py` refuses a `load_tests` in any module
        a `Ran N` guard names, and `mass-gate` runs that module too, so the CI-level guard
        exists -- it is just not in this file. Measured, not assumed.
        """

        relative = "reports/c155_terminal_disposition_register.md"
        self.assertEqual(
            relative,
            REGISTER,
            "this test's literal path and the module's REGISTER constant have drifted; one "
            "of them is reading a document nothing else in this module reads.",
        )
        # ANCHOR THE PATH OUTSIDE THIS MODULE. Agreement between the two literals above is
        # necessary and was, on its own, worthless: moving both together satisfied it while
        # pointing at a doctored copy. The workflow's `changes` paths filter is what decides
        # whether `mass-gate` runs on this document, so it is the one place where naming the
        # wrong file has a consequence that cannot be hidden inside this module.
        # ⚠ POSITION-AWARE, not a line grep. Review round 4 defeated the grep version by
        # DELETING the real entry and satisfying the scan with a decoy indented inside a
        # `run:` body -- which is the exact failure the filter's own comment calls the worst
        # in that file, since register-only PRs then skip `mass-gate` and go green. The grep
        # also FALSE-TRIPPED on a trailing comment and on a second legitimate `reports/c155*`
        # entry. So the block that the shell loop actually consumes is extracted by
        # indentation, and only entries inside it count. Hand-parsed rather than via PyYAML,
        # which is not in the CI job's install list.
        filters = self._workflow_filters()
        # NON-VACUITY on the parser itself before trusting it: the block carries the whole
        # fidelity trigger, so a parser that returned a short list would make the two
        # assertions below meaningless. It must also see this module's OWN path, which is a
        # second independent entry in the same block.
        self.assertGreater(len(filters), 20, "the FILTERS parser read almost nothing")
        self.assertIn("tests/test_terminal_disposition_register.py", filters)
        self.assertIn(
            relative,
            filters,
            "the workflow's `FILTERS` block does not name this register, so a PR whose only "
            "change is editing it matches no filter, skips `mass-gate` -- the only job that "
            "runs this pin -- and goes green. Either the path here was re-pointed at another "
            "document, or the filter was.",
        )
        self.assertEqual(
            [entry for entry in filters if entry.startswith("reports/c155")],
            [relative],
            "the FILTERS block names a different or additional `reports/c155*` path. Exact, "
            "because a decoy entry beside a deleted real one is how the grep this replaced "
            "was defeated.",
        )
        # ...and it must be a file git actually tracks, so a doctored copy cannot be an
        # untracked local file: to exist in a CI checkout it has to be committed, at which
        # point the filter above names the wrong one.
        tracked = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--", relative],
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8").split("\0")
        self.assertEqual(
            [name for name in tracked if name],
            [relative],
            "git does not track the register at this exact path, so what this assertion "
            "reads is not what a CI checkout publishes.",
        )
        text = (REPO / relative).read_bytes().decode("utf-8")
        rows = re.findall(
            r"(?m)^\|\s*`t1\.head_fingerprint`\s*\|\s*([0-9a-f]{16})\s*\|\s*$", text
        )
        self.assertEqual(
            len(rows),
            1,
            "Appendix A does not hold exactly one well-formed `t1.head_fingerprint` row. "
            f"Found {rows!r}. A duplicated row lets a reader that keeps the last one "
            "disagree with a reader that keeps the first; none at all means the pin is "
            "gone rather than failing.",
        )
        self.assertEqual(
            rows[0],
            compute_fingerprint()["fingerprint"][:16],
            "the document's fingerprint row and the tree's fingerprint disagree, read "
            "through NONE of this module's helpers. If the rest of the module is green "
            "while this is red, a reader between the document and the assertion has been "
            "re-pointed and every other T1 pin is currently a tautology.",
        )

    def test_the_appendix_row_is_compared_without_going_through_derive(self) -> None:
        # BLOCKING FINDING 1 of review, half one. The row reaches CI through THREE seams --
        # `head_fingerprint`, `derive()`, and the comparison loop -- and the first revision
        # pinned only the first, so the tautology moved up a frame: `derive()`'s line
        # re-pointed at `register_facts()`, or the loop taught to `continue` on this one
        # key, each left the module at a clean 46/OK on an edited crate source. This is a
        # SECOND, INDEPENDENT comparison that routes through neither.
        self.assertEqual(
            register_facts()["t1.head_fingerprint"],
            head_fingerprint()[:16],
            "Appendix A's fingerprint row disagrees with the tree, and the generic "
            "appendix comparison did not say so -- which means that comparison no longer "
            "reaches this key. Re-derive the row; do not soften the loop.",
        )

    def test_derive_reports_the_tree_and_not_the_document(self) -> None:
        # BLOCKING FINDING 1 of review, half two, and the direct one. Inside the sandbox
        # with a crate source edited, the DERIVED value must part company with the STATED
        # one. A `derive()` that reads Appendix A cannot part company with it, by
        # construction, so this is the assertion that mutation could not survive.
        stated = register_facts()["t1.head_fingerprint"]
        with _fingerprint_sandbox() as root:
            source = sorted((root / "rust" / "pokezero-search" / "src").rglob("*.rs"))
            self.assertTrue(source, "no crate sources in the sandbox to perturb")
            source[0].write_bytes(source[0].read_bytes() + b"\n// derivation probe\n")
            self.assertNotEqual(
                derive()["t1.head_fingerprint"],
                stated,
                "`derive()` reported the DOCUMENT's fingerprint for a tree whose crate "
                "sources had been edited. The appendix is then being compared against "
                "itself and the row certifies nothing.",
            )

    def test_an_unhashed_neighbour_leaves_the_head_fingerprint_alone(self) -> None:
        # Specificity. Without this the sensitivity tests above are satisfied by a
        # "fingerprint" that hashes the whole worktree, which would redden this gate on
        # every Python-only PR -- and a gate that reddens on everything is triaged as noise
        # and then bypassed, which is how a guard stops firing without anyone editing it.
        # ⚠ Each shape gets its OWN subTest and its own assertion. A first revision wrote
        # all three files and then asserted ONCE, so the first over-capture short-circuited
        # the other two and the report named a single failure for three different causes --
        # the "perturb one field per fixture" rule, broken inside the fixture that enforces
        # it. The writes are cumulative on purpose (an unhashed file must stay unhashed
        # whatever else is beside it) and the assertion is per shape.
        with _fingerprint_sandbox() as root:
            before = head_fingerprint()
            crate = root / "rust" / "pokezero-search"
            for relative, body in (
                ("src/notes.md", b"prose, not a build input\n"),
                ("tests/probe.py", b"# not compiled into the crate\n"),
                ("../../third_party/unlisted.patch", b"--- not in the patch list\n"),
            ):
                target = (crate / relative).resolve()
                self.assertTrue(
                    # `root.resolve()` on both sides: on macOS the temp dir is reached
                    # through a symlink, so an unresolved comparison fails on a path that
                    # never left the sandbox.
                    target.is_relative_to(root.resolve()),
                    f"the probe {relative} escaped the sandbox to {target}",
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
                with self.subTest(unhashed=relative):
                    self.assertEqual(
                        head_fingerprint(),
                        before,
                        f"{relative} is not a build input, and writing it moved the head "
                        "fingerprint. Appendix A would then have to be re-derived by every "
                        "PR in the repo, which is how this row gets softened rather than "
                        "maintained.",
                    )
            self.assertEqual(
                head_fingerprint(),
                before,
                "an UNHASHED file moved the head fingerprint. Appendix A would then have "
                "to be re-derived by every PR in the repo, which is how this row gets "
                "softened rather than maintained.",
            )


class TheSourceLevelClaimsAreReadFromSourceTests(unittest.TestCase):
    def test_the_tie_arm_refusal_is_still_shipped(self) -> None:
        # T3's "unbuilt". Resolved by a unique anchor, so deleting the arm raises loudly
        # instead of leaving the register asserting a line number that now means something
        # else.
        line = _anchor(EVENTS, "_ => NO_TRUNCATION,")
        order = _anchor(EVENTS, "return match residual_speed_order(state) {")
        self.assertLess(order, line, "the tie refusal left its match on the speed order")

    def test_the_truncation_flag_gates_the_heal_slot_and_nothing_else(self) -> None:
        # T4's "unbuilt", stated exactly rather than argued: the flag has one binding and
        # one consumer, and the consumer is the Leftovers HEAL push. A damage push that
        # started consulting it would be G33c FIXED, and that has to be recorded.
        references = leftovers_truncated_references()
        self.assertEqual(len(references), 2, references)
        binding = _anchor(EVENTS, "let leftovers_truncated = leftovers_slot_truncated(state, segment);")
        consumer = _anchor(
            EVENTS, "if active.item == Items::LEFTOVERS && !leftovers_truncated[i] {"
        )
        self.assertEqual(sorted(references), sorted([binding, consumer]))

    def test_c149s_split_reaches_only_the_i16max_ceiling_sites(self) -> None:
        # T2. Derived from the patch hunks, not read off the scope comment that asserts it.
        gates = [hunk for hunk in split_hunks() if hunk["adds_split_gate"]]
        self.assertEqual(len(gates), 2)
        for index, hunk in enumerate(gates):
            with self.subTest(hunk=index):
                self.assertTrue(hunk["touches_i16max_site"])
                self.assertFalse(hunk["touches_hp_site"])

    def test_the_four_ceiling_arguments_partition_two_and_two(self) -> None:
        ceilings = sorted(ceiling for _, ceiling in residual_disjoint_band_sites())
        self.assertEqual(
            ceilings,
            ["defender_active.hp", "defender_active.hp", "i16::MAX", "i16::MAX"],
        )


class TheSection4DischargeIsScopedTests(unittest.TestCase):
    def test_the_re_adjudication_closed_nothing_and_opened_nothing(self) -> None:
        # T6 is DISCHARGED-IN-SCOPE, and this is what discharges it. Saying these rows are
        # still open would be manufacturing openness; saying §4 is CLOSED would be
        # manufacturing terminality. Both are wrong and the artifact decides.
        c154 = _artifact(C154_ARTIFACT)
        self.assertEqual(c154["counts"]["rows"], 26)
        self.assertEqual(c154["counts"]["rows_corrected"], 13)
        self.assertEqual(
            len(_ledger_row_ids(_UNREACHABLE_HEADER)),
            27,
            "§4's candidate inventory moved; 'nothing closed, nothing opened' no longer "
            "describes it",
        )

    def test_the_narrow_foreclosure_names_agree_across_generator_and_artifact(
        self,
    ) -> None:
        # The reconstruction called this classification `NARROW_FORECLOSURE`, which is the
        # GENERATOR's constant; the value recorded on each row is `RANDBATS_POPULATION`.
        # Both are pinned so the two names cannot drift apart.
        generator = _text("scripts/c154_unreachable_readjudication.py")
        self.assertIn("NARROW_FORECLOSURE = {", generator)
        self.assertIn('"RANDBATS_POPULATION" if name in NARROW_FORECLOSURE', generator)
        recorded = {
            name
            for name, row in _artifact(C154_ARTIFACT)["verdicts"].items()
            if row.get("foreclosure") == "RANDBATS_POPULATION"
        }
        self.assertEqual(recorded, {"R1", "R23", "R24"})

    def test_the_section_4_population_sentence_is_where_the_register_says(self) -> None:
        line = _anchor(LEDGER, "cannot be reached in gen3 randbats")
        self.assertIn(
            "cannot be reached in gen3 randbats",
            _text(LEDGER).splitlines()[line - 1],
        )

    def test_the_phrase_guard_normalisations_are_the_four_the_register_names(
        self,
    ) -> None:
        # T6 residue 4's DERIVED half. "No fifth obfuscation exists" is a reading and is
        # named as one in the register; that the guard folds these four is not.
        pin = _text("tests/test_unreachable_readjudication.py")
        self.assertIn('_INVISIBLE = "\\u00ad\\u200b\\u200c\\u200d\\ufeff"', pin)
        for token in ("lower()", "\\u200b", "\\u00ad", "*"):
            with self.subTest(token=token):
                self.assertIn(token, pin)


class TheStandingBarsAreQuotedWithTheirMeasurementsTests(unittest.TestCase):
    def test_both_accept_bars_appear_in_every_paragraph_quoting_a_two_window_zero(
        self,
    ) -> None:
        """The ledger §6 item 9 rule, enforced on this document at the rule's own width.

        ⚠ THE FIRST REVISION OF THIS CHECK WAS THE DEFECT IT ENFORCES. It selected
        paragraphs by the literal `31,082`, which is ONE of the five places the register
        asserts a two-window zero, and its failure message read "the zero moved or was
        quoted twice" -- a check advertising coverage it did not have, in a module written
        after #1205 recorded exactly that shape in the workflow's guard scan. Four
        paragraphs carried neither bar. Found in review, not by the author.

        The detector normalises markdown emphasis and code spans before matching, because
        the T3 site is written `**0** in the two head windows` and a literal scan walks
        straight past it -- C154's phrase guard learned the same thing four times.
        """

        paragraphs = two_window_zero_paragraphs()
        self.assertGreaterEqual(
            len(paragraphs),
            5,
            "the detector found fewer paragraphs than the five this document is known to "
            "carry; it has stopped matching and would pass over a bare zero silently",
        )
        derived = derive()
        bars = (
            derived["bar.support_gated_dev"],
            derived["bar.support_gated_holdout"],
            derived["bar.roll_window_dev"],
            derived["bar.roll_window_holdout"],
        )
        for index, block in enumerate(paragraphs):
            flat = _flatten(block)
            for bar in bars:
                with self.subTest(paragraph=index, bar=bar, opening=block[:48]):
                    self.assertIn(
                        bar,
                        flat,
                        "a paragraph asserts a zero over the two permitted windows "
                        "without carrying both accept bars. §6 item 9 of the ledger "
                        "forbids the bare number, and §4 of this register repeats the "
                        f"rule.\n\n{block}",
                    )

    def test_the_zero_detector_fires_on_a_bare_zero_and_is_quiet_otherwise(self) -> None:
        # ANTI-VACUITY for the check above, and the control the first revision lacked. A
        # detector that matched nothing would make every paragraph compliant.
        bare = "The two permitted windows are at **0** divergences at head.\n"
        self.assertEqual(len(two_window_zero_paragraphs(bare)), 1)
        self.assertEqual(
            two_window_zero_paragraphs("The wide census carries 12 divergent rows.\n"), []
        )
        # ...and emphasis around the digit must not hide it, which is the real T3 shape.
        self.assertEqual(
            len(two_window_zero_paragraphs("carry 12 rows and **0** in the two windows\n")),
            1,
        )

    def test_the_rule_of_three_is_applied_to_the_right_denominator(self) -> None:
        bounds = _artifact(C153_CENSUS)["statistical_bounds"]["combined"]
        self.assertAlmostEqual(
            bounds["rule_of_three_per_divergence_upper_95"],
            3 / bounds["classified_divergences"],
            places=6,
        )
        # ...and the register's 846x is the RATIO OF THE DENOMINATORS, re-derived rather
        # than the "four orders of magnitude" C153's prose rounds it to. That substitution
        # is the one C153 records as the way to overstate a per-class negative.
        self.assertEqual(
            derive()["power.boundaries_per_classified_divergence"],
            str(round(bounds["boundaries_measured"] / bounds["classified_divergences"])),
        )
        self.assertLess(
            bounds["rule_of_three_per_boundary_upper_95"] * 100,
            bounds["rule_of_three_per_divergence_upper_95"],
        )


class TheDocumentsClaimsAboutItselfAreReDerivedTests(unittest.TestCase):
    """⚠ ADDED IN REVIEW, AND IT IS THE CLASS FIX RATHER THAN THE INSTANCE.

    Review blocked the first revision on four false claims, and every one of them was this
    document, its pin or its workflow step describing ITSELF: a coverage claim for a check
    that covered one of six paragraphs, a tree-only mutation count inflated from 8 to 10,
    the module's TEST count written into the workflow's MUTATION count, and an engine
    fingerprint attributed to two measurements that build no engine. The substantive
    findings all held. What did not hold was the self-description -- which is the one thing
    a register whose subject is documentary drift cannot get wrong.

    Fixing four sentences repairs the instance. This class repairs the class: every number
    the register, this docstring and the workflow step state ABOUT THIS CHANGE is re-derived
    from the thing it describes, so the next revision cannot bump one site and miss another.
    `reports/c131` §6 records the same author correcting "the instance a reviewer named and
    leaving the same defect one surface over", and C154 §0b hit it twice more.
    """

    @staticmethod
    def _step() -> str:
        """This module's workflow step, sliced off so a sibling step's numbers cannot
        satisfy a claim about this one. A first version of this check read the whole file
        and matched C150's `Battery: 10 mutations`, reporting agreement with a comment
        forty steps away."""

        text = _text(WORKFLOW)
        start = text.index("# C155. The terminal-disposition register.")
        return text[start:].split("      # C150. The ledger's G8 cell", 1)[0]

    @staticmethod
    def _battery() -> str:
        source = _text("tests/test_terminal_disposition_register.py")
        return source.split("MUTATION BATTERY:", 1)[1].split('\n"""', 1)[0]

    def _all(self, pattern: str, haystack: str) -> str:
        """First capture of `pattern` in `haystack`, matched against FOLDED whitespace.

        Folded because these are prose sentences in a hard-wrapped document: a sentence
        that reflows across a line break is the same sentence, and a check that stops
        seeing it silently passes. That is the failure direction this whole class exists
        to remove, so it must not be the failure mode of the class itself.
        """

        folded = re.sub(r"\s+", " ", haystack)
        found = re.search(pattern, folded)
        self.assertIsNotNone(found, f"the sentence matched by {pattern!r} is gone or reworded")
        return found.group(1)

    def test_every_stated_test_count_is_this_modules_own_method_count(self) -> None:
        source = _text("tests/test_terminal_disposition_register.py")
        derived = len(re.findall(r"(?m)^\s+def test", source))
        register, step = _text(REGISTER), self._step()
        for label, pattern, haystack in (
            ("register §6 guard", r"exact `Ran (\d+) tests`", register),
            ("register §6 AST", r"derives (\d+) from the module's AST", register),
            ("register test evidence", r"→ \*\*Ran (\d+) tests, OK\*\*", register),
            ("workflow guard", r"Ran (\d+) tests' /tmp/c155register", step),
            ("workflow error message", r"expected (\d+) register pins", step),
            ("workflow AST", r"deriving (\d+) from the module's AST", step),
        ):
            with self.subTest(site=label):
                self.assertEqual(int(self._all(pattern, haystack)), derived)

    def test_the_guard_scan_coverage_triple_is_re_derived(self) -> None:
        """⚠ FINDING A OF REVIEW ROUND TWO, and the check that then failed in CI.

        The register states how much of #1204's `Ran N tests` guard scan reaches this
        workflow. Round one typed "21 of the 25" and "four" and was RIGHT. Round two
        "re-corrected" it to 20-of-25 / 21-of-26 by counting a COMMENT line carrying the
        invocation string as a guard site -- the same self-match trap the sentence beside it
        discloses, caught for its visible effect and shipped for its arithmetic one. So the
        triple became derived rather than stated.

        ⚠ AND THEN IT WENT RED IN CI WHILE GREEN ON TWO REVIEWERS' TREES, which is the most
        useful thing that happened to this module. GitHub runs the gate on
        `refs/pull/<n>/merge`, not on the branch head. #1203 landed on `main` after this
        branch was cut and added one guarded invocation; this branch added its own; each
        tree reads 25 executable in isolation and the MERGE reads 26. A typed 25 would have
        shipped wrong and silently, because nothing checks a typed number. **This is the
        first defect in three rounds that neither the author nor the reviewer could have
        found locally -- it does not exist on either tree.** Local green and CI green are
        different measurements when a repo gates on the merge, and this module now has a
        first-party instance of the class the register documents.

        DERIVED, never reimplemented: EXECUTABLE sites are lines carrying the invocation
        that are not comments; RESOLVED comes from C154's own `_guards()`, IMPORTED, because
        a second copy of that scan free to drift is worse than one; unresolved is the
        difference.

        THE COUNTS CHURN AND THE FINDING DOES NOT, so both are pinned and they are pinned
        differently. `executable` and `resolved` move whenever ANY step is added to this
        workflow -- that is a declared coupling, stated in the register's §6, and it is what
        catches the merge skew above. The UNRESOLVED SET is the load-bearing half, it is
        #1205's actual subject, and it is pinned BY MODULE NAME rather than by count or by
        line, so a line shift does not touch it and a NEW unguarded step reddens it.

        ⚠ **#1205 IS CLOSED AND THE SET IS NOW EMPTY (C156).** It was invariant at four --
        the same four modules at the old merge-base, at `origin/main`, at #1206's branch
        head and at #1206's merge -- because the cause was invariant: the scan looked at
        most twelve lines past each invocation and those four steps put their guard 12, 16,
        23 and 30 lines out, behind explanatory comments. The scan now derives each step's
        extent from its `run:` body, so all executable invocations resolve. The expectation
        below is `[]` rather than deleted: an EMPTY set that must stay empty is a stronger
        pin than a four-member set that must stay itself, and it still reddens on exactly
        the event that mattered -- a new unguarded step.
        """

        from test_unreachable_readjudication import (  # noqa: E402
            EveryWorkflowTestCountGuardMatchesItsModuleTests as scan,
        )

        lines = _text(WORKFLOW).splitlines()
        # The invocation pattern is IMPORTED, not respelled here: review's residual G is
        # that `python3 -m unittest` and `python -munittest` are real spellings a literal
        # match drops, and a second literal in this module is a second thing to drift.
        carrying = [
            (n, line) for n, line in enumerate(lines, 1) if scan.INVOCATION.search(line)
        ]
        comments = [(n, line) for n, line in carrying if line.strip().startswith("#")]
        executable = len(carrying) - len(comments)
        # ⚠ RESOLUTION IS IMPORTED, INCLUDING ITS NEGATIVE HALF. A previous revision took
        # `resolved` from C154's `_guards()` and then re-implemented the pairing rule here
        # -- `0 <= g - number <= 12` -- to get `unresolved`. That is the second copy the
        # docstring above forbids, and it was a copy of the very window C156 replaced: it
        # would have reported four unresolved sites against a scan that resolves them all.
        # Both halves now come from `_sites()`, which returns unresolved sites rather than
        # dropping them, so the two cannot disagree.
        sites = scan._sites()
        guards = scan._guards()
        resolved = len(guards)
        unresolved = sorted(
            {
                target
                for _, targets, guard, _ in sites if guard is None
                for target in targets
            }
        )

        # Anti-vacuity, and this control is the specific one the finding earns: the file DOES
        # carry the invocation inside a comment, so `executable == len(carrying)` would mean
        # the scan had stopped telling them apart -- which is the defect itself.
        self.assertEqual(len(comments), 1, carrying)
        self.assertGreater(executable, 20)
        self.assertGreater(resolved, 15)
        # And the second anti-vacuity control, which zero unresolved makes necessary: with
        # the set empty, `executable == resolved + 0` is the whole arithmetic, so a scan
        # that had silently stopped seeing a step would satisfy it by shrinking BOTH sides.
        # It cannot, because `executable` is a flat scan of the file and `sites` is the
        # structured walk, and they are required to name the same lines.
        self.assertEqual(len(sites), executable)
        self.assertEqual(executable, resolved + len(unresolved))

        # THE LOAD-BEARING HALF: #1205's subject, by module name. Empty since C156.
        self.assertEqual(
            unresolved, [],
            "the set of workflow steps #1204's scan cannot resolve has changed. C156 "
            "closed #1205 by emptying it, so a module appearing here is a NEW blind spot: "
            "an executable step whose `Ran N tests` guard the scan cannot pair. Add the "
            "guard inside the step's own `run:` body and record the change in the "
            "register beside the counts.",
        )

        skew = (
            "\n\nIF THIS PASSES LOCALLY AND FAILS IN CI: `main` has advanced and added a "
            "unittest step. GitHub gates on `refs/pull/<n>/merge`, so CI measures the merged "
            "workflow and your branch does not. MERGE `origin/main` -- do not rebase -- and "
            "re-derive all three numbers with:\n"
            "  grep -c 'python -m unittest' .github/workflows/engine-fidelity-gates.yml\n"
            "  python -m unittest tests.test_terminal_disposition_register\n"
            "This coupling is deliberate and declared in the register's §6."
        )
        register = _text(REGISTER)
        self.assertEqual(
            int(self._all(r"there are \*\*(\d+) executable\*\* invocation sites", register)),
            executable,
            "the register's EXECUTABLE site count does not match this tree." + skew,
        )
        self.assertEqual(
            int(self._all(r"the scan resolves \*\*(\d+)\*\* of them", register)),
            resolved,
            "the register's RESOLVED guard count does not match this tree." + skew,
        )
        words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        self.assertIn(len(unresolved), words, f"{len(unresolved)} unresolved; extend `words`")
        self.assertRegex(
            re.sub(r"\s+", " ", register),
            rf"leaves \*\*{words[len(unresolved)]}\*\* unresolved",
        )
        # This step must be one of the resolved ones, and the guard it resolves to must be
        # this module's own method count -- what makes the figure load-bearing rather than
        # trivia about someone else's step.
        mine = [
            stated
            for _, targets, stated in guards
            if targets == ("tests.test_terminal_disposition_register",)
        ]
        source = _text("tests/test_terminal_disposition_register.py")
        self.assertEqual(mine, [len(re.findall(r"(?m)^\s+def test", source))])

    def test_every_stated_mutation_count_is_the_enumerated_batterys_size(self) -> None:
        battery = self._battery()
        block_a = re.findall(r"(?m)^  A(\d+)\.", battery)
        block_b = re.findall(r"(?m)^  B(\d+)\.", battery)
        total = len(block_a) + len(block_b)
        self.assertGreater(total, 30, "the battery list collapsed; every count below is vacuous")
        # The lists are consecutive and A runs into B, so a renumbering cannot hide a gap.
        self.assertEqual(
            [int(n) for n in block_a + block_b], list(range(1, total + 1))
        )
        register, step = _text(REGISTER), self._step()
        for label, pattern, haystack in (
            ("docstring header", r"^ (\d+) applied, \1 caught", battery),
            ("register §6", r"\*\*(\d+) mutations applied, \1 caught\*\*", register),
            ("workflow comment", r"Battery: (\d+) mutations applied, \1 caught", step),
        ):
            with self.subTest(site=label):
                self.assertEqual(int(self._all(pattern, haystack)), total)
        self.assertEqual(int(self._all(r"BLOCK A -- A1-A(\d+)", battery)), len(block_a))
        self.assertEqual(int(self._all(r"BLOCK B -- B\d+-B(\d+)", battery)), total)
        self.assertEqual(int(self._all(r"\*\*A1–A(\d+)\*\*", register)), len(block_a))
        self.assertEqual(int(self._all(r"\*\*B\d+–B(\d+)\*\*", register)), total)

    def test_the_tree_only_block_is_stated_at_its_measured_size(self) -> None:
        # THE CLAIM REVIEW CORRECTED FROM TEN TO EIGHT, THEN THE RE-PARTITION MOVED IT
        # AGAIN. Tree-only is the property that distinguishes this pin from a diary, so the
        # word is re-derived from block B's own length in all three places it appears, and
        # the number is never written into this comment -- a previous revision hard-coded
        # TWELVE here and in A34's description and both went stale in the same round the
        # class was added to stop exactly that. Neither was reachable by the class, because
        # the class checks the three STATED sites and not its own prose about them.
        battery = self._battery()
        size = len(re.findall(r"(?m)^  B\d+\.", battery))
        words = {
            8: "EIGHT", 9: "NINE", 10: "TEN", 11: "ELEVEN", 12: "TWELVE",
            13: "THIRTEEN", 14: "FOURTEEN", 15: "FIFTEEN", 16: "SIXTEEN",
            17: "SEVENTEEN", 18: "EIGHTEEN", 19: "NINETEEN", 20: "TWENTY",
            21: "TWENTY-ONE", 22: "TWENTY-TWO", 23: "TWENTY-THREE",
            24: "TWENTY-FOUR", 25: "TWENTY-FIVE", 26: "TWENTY-SIX",
            27: "TWENTY-SEVEN", 28: "TWENTY-EIGHT", 29: "TWENTY-NINE",
            30: "THIRTY", 31: "THIRTY-ONE", 32: "THIRTY-TWO",
            33: "THIRTY-THREE", 34: "THIRTY-FOUR", 35: "THIRTY-FIVE",
            36: "THIRTY-SIX", 37: "THIRTY-SEVEN", 38: "THIRTY-EIGHT",
            39: "THIRTY-NINE", 40: "FORTY", 41: "FORTY-ONE",
        }
        self.assertIn(size, words, f"block B has {size} entries; extend `words`")
        word = words[size]
        self.assertRegex(battery, rf"BLOCK B -- B\d+-B\d+, {word} mutations")
        self.assertRegex(self._step(), rf"\b{word} of the \d+ are applied only to the tree")
        self.assertRegex(_text(REGISTER), rf"(?i)\b{word.lower()} is the measured figure")

    def test_the_readings_and_zero_paragraph_counts_are_stated_as_measured(self) -> None:
        readings = [key for key, value in EXPECTED_PIN.items() if value == "derived + reading"]
        words = {3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven"}
        self.assertIn(len(readings), words)
        register = _text(REGISTER)
        self.assertEqual(
            self._all(r"It cannot cover a reading\.\*\* (\w+) are named", register),
            words[len(readings)],
        )
        self.assertEqual(
            int(self._all(r"finds\s*\n?\*\*(\d+)\*\* paragraphs", register)),
            len(two_window_zero_paragraphs()),
        )


class TheRegisterRendersTests(unittest.TestCase):
    def test_no_table_row_drops_a_cell(self) -> None:
        # `tests/test_ledger_table_uniformity.py` holds the repo-wide inventory of
        # over-delimited rows and this document must never join it. Checked here too, so a
        # register whose item table silently lost its `pin` column fails in its own module.
        self.assertEqual(
            delimiter_anomalies(_text(REGISTER)), {"over": [], "under": []}
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
