"""Hold section 4's 26 UNREACHABLE verdicts against the C154 re-adjudication artifact.

WHY THIS MODULE EXISTS, AND WHY IT IS NEITHER OF ITS TWO SIBLINGS.

`tests/test_never_fired_counter_census.py` re-derives every ABSENCE over the committed
corpus. `tests/test_wide_seed_negative_census.py` pins a MEASUREMENT taken outside the two
permitted windows. Neither can do this job, and section 4 is the part of the ledger where
neither ever ran: its rows are not counters and not sweeps, they are 26 sentences of the
form "this cannot happen", carried on prior work's word since C138.

C153 adopted the rule that decides them -- **trace the raise site to the caller that
actually reaches it, not to a plausible sentence about it** -- and applying it to
`CENSUS_CANNOT_REACH`'s seven entries corrected THREE. Section 4's 26 had never been
through it. This module is the instrument that keeps them through it: it re-derives every
citation in every demonstration on every run, so the class of failure C153 named -- "the
figure was correct when taken and false later because its tree was replaced" -- reddens
here instead of ageing quietly in a table cell.

WHAT IS PINNED, and in every case what would have to be true for the pin to be vacuous.

  1. **The row inventory is DERIVED TWICE and matched in both directions.** Once from
     `tests/test_ledger_table_uniformity.py::_EXPECTED_UNREACHABLE_ROWS` and once from the
     ledger's own section 4 table, then against the artifact's verdict keys. A row added
     to section 4 without an adjudication is red; an adjudication for a row that left the
     table is red.

  2. **EVERY DEMONSTRATION IS RE-DERIVED FROM SOURCE, BYTE FOR BYTE.** The pin re-runs
     `build_verdicts()` against the artifact's own committed `pool` block and asserts the
     result equals the committed `verdicts` block. Every `file:line` in every
     demonstration comes from `_anchor` / `_anchor_after` / `_raise_line`, so a moved
     anchor changes the string and fails here, and a DELETED anchor raises out of the
     resolver rather than silently pointing at whatever now occupies the line. This is the
     single most load-bearing assertion in the module: without it these are 26 paragraphs
     of prose with hard-coded line numbers, which is exactly what #1202 turned into
     fifteen stale citations overnight.

  3. **The pool half cannot be laundered.** `build_verdicts` re-runs the absence
     assertions (`absent()` raises if a move a row calls 0-of-220 is present), so hand
     editing a count in the artifact to keep a row green fails at re-derivation.

  4. **THE LEDGER PROSE IS PINNED AGAINST THE ARTIFACT, IN BOTH DIRECTIONS.** Every row
     the artifact marks INCOMPLETE or FALSE must carry a C154 correction marker in its
     section 4 cell, and every row it marks SOUND must NOT -- so a correction that lands
     in the data and not in the sentence is red, and so is a marker on a row that was
     never corrected. This repo has shipped that exact desync three times.

  5. **The retracted sentences appear exactly where they are withdrawn, and nowhere
     else.** Two pins, because one is not enough. The QUOTING rule requires each occurrence
     to sit inside a quotation adjacent to the C154 marker; the EXACT OCCURRENCE INVENTORY
     requires the count per phrase to match. The second exists because review's fourth
     smuggle -- a re-assertion inside the correction's own cell -- satisfies the first and
     must, since it is indistinguishable from the legitimate shape. ⚠ The normaliser folds
     whitespace, case, markdown emphasis and zero-width characters, and each of those four
     was added after something got past the previous three.

  6. **Cross-instrument coupling is DECLARED AND CHECKED. ⚠ It asserted the OPPOSITE one
     revision ago, on a hazard nobody had measured.** The artifact was written to
     `tests/data/` because its refusal-reason names beside pool counts would supposedly
     read to `tests/test_never_fired_counter_census.py` as four counters firing. Review
     measured it: in the corpus, that census reports `Ran 22 tests ... OK`. The names occur
     only as string VALUES in prose, which `_evidence_in` excludes in terms. The placement
     bought nothing and cost the guard that census's header warns about by name, and its
     control -- `assertIn(reason, json.dumps(document))` -- was substring-in-prose and
     could not fail. The artifact is in `reports/artifacts/` now, membership is asserted,
     and the control feeds `_evidence_in` a counter-keyed copy and requires it to FIRE.

  7. **The denominator matches the emission site.** For the five rows that close a
     differential counter the artifact records the AST-derived emission sites and the
     denominator NAME. The pin re-derives both and additionally asserts
     `skip:world_unsupported:` has TWO sites, because a row that cited "the" emission site
     would be inadmissible for the same reason a reason with three raise sites cannot be
     cited by line.

  8. **ANTI-VACUITY, and it is the reason any of the zeroes mean anything.** Every "0 of
     220" in this document is a lookup into one dict built from one file. A census that
     silently read nothing would make all 26 rows EASIER to assert. Ten controls are
     pinned nonzero -- `raindance` 7, `spikes`, `encore`, `trick` 2, `knockoff` 4,
     `gigadrain` 2, Cute Charm's three species, Liquid Ooze's two, 44 Sleep Talk sets, 55
     Rest sets -- plus the generative census's 24,000 Pokemon and 13 items.

  9. **The volatile-producer scan's OWN correction is controlled.** The first version of
     that scan walked the dex and regexed `JSON.stringify`, which drops functions, so it
     reported the same-named move as the only producer of all ten volatiles -- a scan that
     could not fail. The pin asserts the source scan still finds the two producers that
     version missed, Cute Charm to `attract` and Lansat Berry to `focusenergy`. If the
     scan silently stops walking files, this is what goes red.

 10. **Three sentences of this pass's own are re-derived rather than asserted**, all three
     because review found them over-claimed: R10's `heal_subcase` caller graph (built by
     reverse reachability, `render_move_phase` chokepoint and five dead wrappers), the
     corrections tally (which two docstring revisions stated as SEVEN and as TEN against
     the artifact's THIRTEEN), and R22's `move_fails_encore` set equality (previously six
     membership assertions, which review defeated by adding `Choices::TACKLE`).

 11. **The foreclosure SCOPE is carried per row.** `UNREACHABLE_TRACED` was documented as
     "cannot fire for any caller of X" and that is false for three rows -- R23's counter
     fires today on the scenario corpus. Rather than soften the definition for all 26 and
     lose the stronger result on the other 23, each row carries `ALL_CALLERS` or
     `RANDBATS_POPULATION`.

 12. **Section 8's own row count is pinned to the derived one.** It read **81** while the
     table held 82 and `test_the_stated_row_count_matches_the_rows` pinned 82 in the
     header -- a fourth instance of "a correction applied to data does not propagate to
     the prose describing it", found by this pass and fixed with a pin rather than an
     edit.

WHAT THIS MODULE DOES NOT AND CANNOT COVER, stated rather than left to be discovered.

  * **The pool half is not re-derived in CI.** CI builds no pokemon-showdown checkout
    (`tests/_showdown_root.py`), so the `pool` block is a committed measurement at one
    Showdown commit, recorded on the artifact. A `sets.json` bump that added `taunt` to a
    set would leave this module green and the ledger wrong. Bounded and nameable, exactly
    as `scripts/c152_pool_reachability_census.py` records for its own artifact --
    regenerating is a deliberate act that shows up in review.
  * **FIVE judgements are HUMAN READINGS and are marked as such on the rows themselves,
    because no pin can carry them.** (a) R1's closure is that a committed Future Sight
    scenario sits in `interaction_registry_specs()` rather than `scenario_specs()`; the
    pin can assert the two functions exist and that the spec is in the first, and it does,
    but "no harness passes `specs=` explicitly" is a claim about every present and future
    caller. (b) R9's enumeration of "the gen3 residual positive heals available to a
    non-Leftovers holder" is an argument about a mechanic, not a grep. (c) R22's "the
    crate's six `move_fails_encore` members are exactly gen3's `failencore`-flagged moves"
    is re-derivable from the dex but the JUDGEMENT that this makes the shipped list
    correct is not. (d) R7's scenario-studio producer is closed by "the service never
    builds an engine world", which is an absence over a module, and the pin asserts that
    absence by grep -- a grep-shaped claim, scoped in the assertion message to the module
    it was run over.

THE MUTATION BATTERY, ENUMERATED. A battery whose members are not written down costs
exactly what C153's cost: it claimed "12 applied, 12 caught" and a reviewer found the 13th
survived. Each below was applied to a copy of the artifact (or to the generator, the
ledger, the patch, the corpus or the workflow) and had to turn this module RED. All 35 do.

   1. a verdict word changed from `UNREACHABLE_TRACED` to `NOT_OBSERVED_AT_SCOPE`
   2. a row deleted from the artifact's verdict map
   3. `ledger_reason_status` flipped from `FALSE` to `SOUND` with the correction left in
   4. a correction emptied on a row the artifact marks `FALSE`
   5. a line number inside a demonstration incremented by one
   6. a pool move count changed from 0 to 1 for a move a row calls absent
   7. the C154 marker removed from one section 4 cell
   8. ⚠ a retracted phrase re-inserted into the ledger, HARD-WRAPPED *and lower-cased*
   9. the `skip:world_unsupported:` emission-site list truncated to one site
  10. a denominator name swapped from `boundaries_full_round` to `boundaries_measured`
  11. the volatile source scan's Lansat Berry hit deleted
  12. section 8's row count set back to 81

⚠ **Number 8 SURVIVED the first version**, and that is why 1-12 are written down rather than
counted. The guard it defeated already normalised whitespace -- C153's own fix -- but was
case-sensitive, so a hard-wrapped, lower-cased re-insertion walked through. That is C153's
mutation 24, one document over, in a guard written by someone who had just read about it.

**13-21 ARE REVIEW'S, ADDED AFTER THIS MODULE SHIPPED CLAIMING 12 OF 12.** Every one of them
corresponds to a sentence of this pass's own that was asserted beyond what it traced, which
is the defect the pass exists to remove -- so they are listed as a separate block rather
than renumbered into the first, and the count of blocks is the honest summary.

  13. a `foreclosure` flipped from `RANDBATS_POPULATION` to `ALL_CALLERS`
  14. an edge removed from the derived `heal_subcase` caller graph
  15. `Choices::TACKLE` smuggled into the patch's `move_fails_encore` arm -- the mutation
      that passed the previous six-membership version green
  16. the artifact moved back out of `counter_artifacts()`'s glob
  17. the corrections tally detached from the verdict records
  18. a retraction restated INSIDE the correction cell that withdraws it -- which the
      quoting rule admits and must, and which the exact occurrence inventory catches
  19. the same phrase re-inserted with `**bold**`
  20. the same phrase re-inserted with a soft hyphen (U+00AD) mid-word
  21. the same phrase re-inserted with a zero-width space (U+200B) mid-word

**22-23 ARE ROUND TWO'S, and they are the ones that matter most**, because the defect they cover was
introduced BY the fix for 13-21: bumping this module's own workflow guard with an unbounded string
replace also rewrote the final-holdout step's guard, which gates `OWNER_RATIFIED`,
`BURNED_FINAL_HOLDOUT` and the burn, to a count its suite can never print.

  22. the final-holdout step's `Ran N tests` guard set to any value other than 25
  23. this module's own guard left stale after adding a test

**24-35 ARE C156'S, AND 24-27 ARE THE FOUR THIS BATTERY COULD NOT HAVE CAUGHT BEFORE IT.**
Round two closed the instance and shipped the class one surface over for the third time:
the scan that re-derives every guard reached only 22 of the 26 executable steps, and the
four it missed were missing from its OUTPUT, not marked as missed. Each of 24-27 was run
against `origin/main`'s scan first and observed GREEN there -- that is the demonstration
that the coverage was absent, and without it "the fix works" would be a claim about a scan
nobody had made fail.

  24. the SEED-REGISTRY step's guard set to a count its suite can never print. This is the
      exact probe #1205 reported: green before, red now.
  25. the same, on the spread-gate provenance step
  26. the same, on the stat-attestation step
  27. the same, on the denominator pair -- whose guard sat at a gap of exactly 12, one line
      past the old window, and is therefore the boundary case
  28. a new executable unittest step added to the workflow with NO count guard at all, so
      coverage is asserted rather than the site being dropped
  29. `_run_bodies` made to return nothing, so the scan sees no sites at all. "Zero
      unresolved" is TRUE and worthless for an empty scan; this is the control that says
      so, and it is the same class as 24-27 rather than a different one
  30. `load_tests = <callable>` added to a scanned module as an ASSIGNMENT. #1205's review
      named this form as uncovered on the ground that nothing in scope uses it today.
      ⚠ Applied at `tests/test_drag_limit_is_a_last_resort.py`, whose step was RESOLVED at
      base -- so its green there cannot be explained by the site already being blind, which
      is the objection review raised against siting it anywhere else
  31. a subclass of a base declared in the same module, adding no method of its own, so
      `unittest` collects the base's tests twice and `_methods` counts them once. Same
      resolved-at-base site, for the same reason
  32. a guard's `if`-block deleted and the count restated as a comment ABOVE the next step,
      i.e. outside the `run:` body it grades. Green before, because a dropped site is
      silence; red now, because the site is represented. ⚠ Applied at the Classifier pins
      step (drag-limit), which was RESOLVED at base -- same siting requirement as 30 and 31,
      and review noted this entry had not stated it
  33. the battery total in the workflow comment left at its old value while this list grew
  34. a guard weakened from an exact count to the FLOOR shape `Ran [0-9]+ tests`, which is
      what `fleet-worker.yml:45` uses today. A floor cannot detect a suite that shrank, and
      #1163 recorded a floor as the one fail-open in its own battery. Green at base -- the
      site simply left the scan -- and red here, because a guard the pattern cannot read is
      an unresolved site rather than an absent one
  35. ⚠ a WORKFLOW LINE NUMBER re-typed into this module's prose at its correct current
      value, in each of the FOUR spellings the four historical citations actually used, plus
      the denominator-step residual review named, a guard line and an invocation line.
      Seven shapes,
      seven reds, and NC1 still green. This entry exists because the guard it exercises
      SHIPPED COVERING NOTHING: its predecessor forbade a phrase this pass invented, which
      appears nowhere in the repository but inside its own regex, so all four real citations
      passed it. A guard verified only against a string its author chose is not verified.
      `reports/c156` §3.1 carries the seven shapes and their verdicts

⚠ **AND TWO NEGATIVE CONTROLS, because the first one alone proves nothing.** Forty comment
lines inserted between an invocation and its guard -- the edit that created all four blind
spots -- must be GREEN, and it is green at `dbb40c5c` too. That is the trap: there it is
green by going BLIND, here by resolving the site. So the control is run a second time with
the padded step's guard ALSO falsified to a count its suite cannot print. At `dbb40c5c` that
stays green; here it is red. Same edit, same verdict on the first pass, opposite on the
second, and only the second distinguishes a check from a silence.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import c154_unreachable_readjudication as c154  # noqa: E402
from test_ledger_table_uniformity import (  # noqa: E402
    _EXPECTED_UNREACHABLE_ROWS,
    _UNREACHABLE_HEADER,
    _tables_by_header,
    row_cells,
)
from test_never_fired_counter_census import counter_artifacts  # noqa: E402

ARTIFACT = "reports/artifacts/c154_unreachable_readjudication.json"
LEDGER = "reports/c138_known_gaps_ledger.md"

#: The marker every corrected section 4 cell must carry, and that no uncorrected one may.
#: One token, so the pin is a membership test rather than a fuzzy match on prose.
C154_MARKER = "⚠ **C154"

#: Sentences this pass RETRACTED. Each must be absent from the ledger, and the check runs
#: over the whitespace-NORMALISED document because every one of them is long enough to be
#: hard-wrapped by an editor, which is how a previous phrase guard in this repo went blind.
#: Kept short and distinctive: a fragment that would also match the sentence RECORDING the
#: retraction would make this pin unsatisfiable, so each is a clause the correction quotes
#: only inside a `"` pair, and `test_the_retraction_quoting_rule_is_live` proves the
#: quoting exemption cannot be smuggled through.
#: EXACT occurrence count per phrase in the ledger, because the quoting rule alone cannot
#: catch a re-assertion smuggled INSIDE the correction's own cell -- review's fourth
#: smuggle, and it defeated the first version:
#:
#:   ⚠ **C154 correction ... FALSE.** On reflection the original was right after all:
#:   "The expiry path has no trigger"...
#:
#: That passes quoting AND marker-adjacency, because both are satisfied by the legitimate
#: retraction two sentences earlier. An exact inventory catches it on arithmetic instead of
#: on sentiment: the phrase may appear exactly where it is withdrawn and nowhere else.
#: R4's phrase is 2 because §8's fifth-round block quotes it as well.
RETRACTED_OCCURRENCES = {
    "The expiry path has no trigger": 2,
    "The Liquid Ooze guard inside `residual_heal_cause` is therefore dead code": 1,
    "what makes this row UNREACHABLE is the renderer interception alone": 1,
    "This closes `_HIDDEN_INFORMATION_REQUEST_FLAGS`'s `maybeDisabled`/`maybeLocked` "
    "(Imprison is their only producer), the `failencore` move-list edge cases, and G32": 1,
}

RETRACTED = (
    "The expiry path has no trigger",
    "The Liquid Ooze guard inside `residual_heal_cause` is therefore dead code",
    "what makes this row UNREACHABLE is the renderer interception alone",
    "This closes `_HIDDEN_INFORMATION_REQUEST_FLAGS`'s `maybeDisabled`/`maybeLocked` "
    "(Imprison is their only producer), the `failencore` move-list edge cases, and G32",
)

#: Nonzero controls. Every absence in this document is a lookup into one dict built from
#: one file; a census that read nothing would make all 26 rows easier to assert.
NONZERO_MOVE_CONTROLS = {
    "raindance": 7,
    "sunnyday": 4,
    "trick": 2,
    "knockoff": 4,
    "gigadrain": 2,
    "bonemerang": 1,
    "transform": 2,
}


def _document() -> dict:
    with open(os.path.join(REPO, ARTIFACT), encoding="utf-8") as handle:
        return json.load(handle)


def _ledger_text() -> str:
    with open(os.path.join(REPO, LEDGER), encoding="utf-8") as handle:
        return handle.read()


def _section_four_cells() -> dict[str, str]:
    grouped = _tables_by_header(_ledger_text())
    tables = grouped.get(_UNREACHABLE_HEADER, [])
    assert len(tables) == 1, f"section 4 table not found by header {_UNREACHABLE_HEADER!r}"
    out: dict[str, str] = {}
    for _number, line in tables[0]["rows"]:
        cells = row_cells(line)
        out[cells[0].strip("*~ ")] = cells[2]
    return out


#: How close the C154 marker must sit to a quoted retraction, in normalised characters.
#: Wide enough for a table cell, narrow enough that a marker elsewhere in the document
#: cannot launder an unrelated restatement.
_RETRACTION_MARKER_WINDOW = 1200


def _is_a_recorded_retraction(text: str, match: "re.Match[str]") -> bool:
    """Whether a retracted phrase occurrence is a QUOTED withdrawal rather than a claim.

    Two conditions, and the second is the one that matters. A bare quote mark is the
    cheapest possible smuggle -- anyone re-asserting the old sentence would type it in
    quotes without thinking -- so the quotation must additionally sit within
    `_RETRACTION_MARKER_WINDOW` characters of the C154 marker that withdraws it.
    """

    before = text[max(0, match.start() - 2):match.start()]
    if '"' not in before:
        return False
    window = text[max(0, match.start() - _RETRACTION_MARKER_WINDOW):
                  match.end() + _RETRACTION_MARKER_WINDOW]
    return _normalised(C154_MARKER) in window


#: Characters that render as nothing (or as a line break that never happens) and therefore
#: let a retracted sentence past a guard that only folds whitespace. Measured, not guessed:
#: review evaded the first version with markdown emphasis, U+00AD and U+200B, and NBSP was
#: already caught by `\s`.
_INVISIBLE = "\u00ad\u200b\u200c\u200d\ufeff"


def _normalised(text: str) -> str:
    """The document as one line of single-spaced, lower-cased, unemphasised words.

    ⚠ THREE EVASION CLASSES, ALL FOUND BY SOMEONE ELSE, EACH ONE TURN LATER THAN THE LAST.

      * Hard wrapping. A guard that matched per line was blind to any sentence an editor
        had wrapped. Fixed before this module shipped -- it is C153's fix.
      * Case. Battery mutation 8 re-inserted a retracted sentence hard-wrapped AND
        lower-cased and walked past the whitespace-only version. Found by this module's
        own battery.
      * **Markdown emphasis and zero-width characters, in a MARKDOWN file.** `**bold**`,
        `*italic*`, backticks, U+00AD and U+200B all split a phrase into pieces no literal
        match can see. Found by review, after the module shipped claiming the guard was
        sound. This is the same defect three times: each fix addressed the instance named
        and not the class, which is exactly the through-line `reports/c131` §6 records
        about its own author.

    Emphasis markers and invisibles are DELETED rather than replaced by a space, because
    `**The** expiry` and `The expiry` have to normalise to the same string.
    """

    text = text.translate({ord(c): None for c in _INVISIBLE})
    text = re.sub(r"[*_`~]+", "", text)
    return re.sub(r"\s+", " ", text).lower()


class TheInventoryIsDerivedAndExactTests(unittest.TestCase):
    def test_the_artifact_covers_exactly_the_ledgers_section_four_rows(self) -> None:
        document = _document()
        from_pin = tuple(row.strip("*~ ") for row in _EXPECTED_UNREACHABLE_ROWS)
        from_ledger = tuple(_section_four_cells())
        self.assertEqual(
            from_pin, from_ledger,
            "the uniformity pin's section 4 inventory and the ledger's own table "
            "disagree; one of them moved without the other",
        )
        self.assertEqual(
            tuple(sorted(document["verdicts"], key=lambda r: int(r[1:]))), from_ledger,
            "a section 4 row has no adjudication, or an adjudication has no row. A row "
            "may not join section 4 without going through the C153 rule.",
        )

    def test_every_verdict_word_is_from_the_closed_set(self) -> None:
        for name, record in _document()["verdicts"].items():
            with self.subTest(row=name):
                self.assertIn(
                    record["verdict"], (*c154.VERDICTS, "WITHDRAWN_BEFORE_THIS_PASS"),
                    "a fourth verdict category is a row this pass got wrong, not a new "
                    "kind of answer",
                )
                self.assertIn(record["ledger_reason_status"], c154.REASON_STATUSES)

    def test_every_non_sound_row_records_what_replaces_its_reason(self) -> None:
        for name, record in _document()["verdicts"].items():
            if record["ledger_reason_status"] == "SOUND":
                continue
            with self.subTest(row=name):
                self.assertTrue(
                    (record["correction"] or "").strip(),
                    f"{name}: the stated mechanism is "
                    f"{record['ledger_reason_status']} and nothing is written down to "
                    "replace it, which is the shape this pass exists to remove",
                )

    def test_no_row_carries_a_bare_absence_without_its_denominator(self) -> None:
        # "0 of 220" is the only form admitted. A bare "0" or "none" is a count with no
        # instrument, which section 8 has forbidden since C125.
        for name, record in _document()["verdicts"].items():
            demonstration = record["demonstration"] or ""
            if "0 of" not in demonstration:
                continue
            with self.subTest(row=name):
                self.assertRegex(
                    demonstration, r"0 of (?:those |its |the )?\d",
                    f"{name} states an absence without the denominator it was measured "
                    "against",
                )


class TheCitationsAreResolvedOnEveryRunTests(unittest.TestCase):
    def test_every_demonstration_re_derives_from_the_current_tree(self) -> None:
        """The anti-staleness pin, and the reason this module is worth running.

        `build_verdicts` resolves every citation through `_anchor` / `_anchor_after` /
        `_raise_line` against the tree as it is NOW, using the artifact's own committed
        pool measurements as the data half. A moved anchor changes a line number and this
        fails; a deleted one raises out of the resolver. #1202 moved fifteen citations in
        one merge on demonstrations whose entire value was that they had been traced.
        """

        document = _document()
        rebuilt = c154.build_verdicts(document["pool"])
        self.assertEqual(
            set(rebuilt), set(document["verdicts"]),
            "re-derivation produced a different row set than the artifact carries",
        )
        for name in sorted(rebuilt, key=lambda r: int(r[1:])):
            with self.subTest(row=name):
                self.assertEqual(
                    rebuilt[name], document["verdicts"][name],
                    f"{name} no longer re-derives from source. Either a citation has "
                    "moved -- re-run scripts/c154_unreachable_readjudication.py --write "
                    "and READ THE DIFF, because a line that moved may have moved into "
                    "different code -- or the artifact was hand-edited.",
                )

    def test_a_perturbed_citation_is_caught(self) -> None:
        """The control for the pin above: prove it can see a wrong line number."""

        document = _document()
        rebuilt = c154.build_verdicts(document["pool"])
        tampered = dict(document["verdicts"])
        record = dict(tampered["R9"])
        record["demonstration"] = re.sub(
            r":(\d+)", lambda m: f":{int(m.group(1)) + 1}", record["demonstration"], count=1
        )
        tampered["R9"] = record
        self.assertNotEqual(
            rebuilt["R9"], tampered["R9"],
            "a demonstration with a bumped line number compared EQUAL to the re-derived "
            "one, so the citation pin above is asserting nothing",
        )

    def test_an_absence_that_stopped_being_true_fails_re_derivation(self) -> None:
        """`absent()` is a gate, not a formatter. Prove it refuses."""

        document = _document()
        poisoned = json.loads(json.dumps(document["pool"]))
        poisoned["move_species"]["taunt"] = ["gengar"]
        with self.assertRaises(SystemExit):
            c154.build_verdicts(poisoned)


class TheLedgerProseMatchesTheDataTests(unittest.TestCase):
    def test_every_corrected_row_carries_the_marker_and_no_other_row_does(self) -> None:
        document = _document()
        cells = _section_four_cells()
        for name, record in document["verdicts"].items():
            corrected = record["ledger_reason_status"] != "SOUND" and name != "R26"
            with self.subTest(row=name, status=record["ledger_reason_status"]):
                if corrected:
                    self.assertIn(
                        C154_MARKER, cells[name],
                        f"{name}'s stated mechanism is "
                        f"{record['ledger_reason_status']} in the artifact and its "
                        "section 4 cell does not say so. A correction that lands in the "
                        "data and not in the sentence describing it is the desync this "
                        "repo has shipped three times.",
                    )
                else:
                    self.assertNotIn(
                        C154_MARKER, cells[name],
                        f"{name} carries a C154 correction marker but the artifact marks "
                        "its reason SOUND. The marker is not decoration.",
                    )

    def test_no_retracted_sentence_survives_unquoted_in_the_ledger(self) -> None:
        text = _normalised(_ledger_text())
        for phrase in RETRACTED:
            with self.subTest(phrase=phrase[:48]):
                occurrences = list(re.finditer(re.escape(_normalised(phrase)), text))
                self.assertTrue(
                    occurrences,
                    "a retracted sentence vanished from the ledger entirely. Nothing here "
                    "is quietly repaired: the withdrawal must be readable next to what it "
                    "withdraws, so deleting the sentence is as wrong as keeping it.",
                )
                for match in occurrences:
                    self.assertTrue(
                        _is_a_recorded_retraction(text, match),
                        f"{LEDGER} carries a retracted sentence that is not marked as one. "
                        "A correction may QUOTE what it withdraws, and the quotation has to "
                        "sit inside a cell that names the correction; a bare quote mark is "
                        "not enough, because that is the cheapest possible way to smuggle "
                        "the old claim back in.",
                    )

    def test_the_retraction_quoting_rule_is_live(self) -> None:
        """Three smuggles against the rule above. All three must be rejected.

        The exemption is the weak point of any phrase guard: too loose and the retracted
        claim walks straight back in wearing quote marks. C153's reviewer defeated the
        first version of its exemption three ways, so the smuggles are fixtures here
        rather than a claim that none exist.
        """

        phrase = _normalised(RETRACTED[0])
        smuggles = {
            "bare quote mark, no correction marker": f'and so "{phrase}" as always.',
            "correction marker far away": (
                "⚠ **C154 correction 2026-08-09 — something else entirely.** "
                + "filler " * 200
                + f'"{phrase}"'
            ),
            "unquoted next to a marker": (
                f"⚠ **C154 correction 2026-08-09 —** {phrase} still holds."
            ),
        }
        # ⚠ THE FOURTH SMUGGLE DEFEATS THIS RULE AND IS CAUGHT BY THE OTHER ONE. A
        # re-assertion inside the correction's OWN cell satisfies both conditions -- the
        # quote mark and the marker are supplied by the legitimate withdrawal two sentences
        # earlier -- so `_is_a_recorded_retraction` admits it and must. What catches it is
        # `test_each_retracted_phrase_appears_exactly_where_it_is_withdrawn`, on the count.
        # Asserted here, rather than left implicit, so nobody "fixes" the quoting rule to
        # cover a case the inventory already owns.
        in_cell = _normalised(
            f'"{phrase}." ⚠ **C154 correction 2026-08-09 — FALSE.** On reflection the '
            f'original was right after all: "{phrase}".'
        )
        matches = list(re.finditer(re.escape(phrase), in_cell))
        self.assertEqual(len(matches), 2)
        self.assertTrue(
            all(_is_a_recorded_retraction(in_cell, m) for m in matches),
            "the quoting rule now rejects the in-cell smuggle, which means it also rejects "
            "the legitimate shape it cannot be told apart from; the inventory pin is what "
            "separates them",
        )
        self.assertNotEqual(
            len(matches), RETRACTED_OCCURRENCES[RETRACTED[0]] - 1,
            "the inventory pin must see the extra occurrence this smuggle adds",
        )
        for label, blob in smuggles.items():
            with self.subTest(smuggle=label):
                text = _normalised(blob)
                match = re.search(re.escape(phrase), text)
                self.assertIsNotNone(match, "the smuggle fixture does not contain the phrase")
                self.assertFalse(
                    _is_a_recorded_retraction(text, match),
                    f"the quoting rule admits a smuggle ({label}), so "
                    "test_no_retracted_sentence_survives_unquoted_in_the_ledger is "
                    "asserting less than it says",
                )
        # ...and the shape the ledger actually uses must still be admitted, or the rule is
        # unsatisfiable rather than strict.
        legitimate = _normalised(
            f'... "{phrase}." ⚠ **C154 correction 2026-08-09 — the sentence quoted above '
            'is FALSE; the verdict survives.** ...'
        )
        match = re.search(re.escape(phrase), legitimate)
        self.assertTrue(_is_a_recorded_retraction(legitimate, match))

    def test_each_retracted_phrase_appears_exactly_where_it_is_withdrawn(self) -> None:
        """Exact inventory, not a floor.

        The quoting rule cannot catch a re-assertion smuggled inside the correction's own
        cell, because that cell legitimately contains both a quote mark and the marker.
        Arithmetic can: the phrase may appear exactly where it is withdrawn and nowhere
        else.
        """

        text = _normalised(_ledger_text())
        measured = {
            phrase: len(re.findall(re.escape(_normalised(phrase)), text))
            for phrase in RETRACTED_OCCURRENCES
        }
        self.assertEqual(
            measured, RETRACTED_OCCURRENCES,
            "the number of times a retracted sentence appears in the ledger has changed. "
            "MORE means it has been restated somewhere -- including, and this is the case "
            "the quoting rule cannot see, inside the correction that withdraws it. FEWER "
            "means a withdrawal was deleted rather than kept struck.",
        )

    def test_the_guard_survives_emphasis_and_invisible_characters(self) -> None:
        """Review's evasion classes, as fixtures.

        `**bold**`, `*italic*`, backticks, U+00AD and U+200B all render as the same
        sentence and all split it into pieces a literal match cannot see -- in a MARKDOWN
        file, where emphasis is the most likely thing an editor adds. NBSP was already
        caught by `\\s` and is kept as the control that the fixture set is not all
        positives.
        """

        phrase = RETRACTED[0]
        words = phrase.split()
        evasions = {
            "bold on one word": " ".join(["**" + words[0] + "**", *words[1:]]),
            "italic spanning two": " ".join([f"*{words[0]} {words[1]}*", *words[2:]]),
            "backticks": " ".join([f"`{words[0]}`", *words[1:]]),
            "soft hyphen mid-word": phrase.replace("expiry", "ex\u00adpiry"),
            "zero-width space mid-word": phrase.replace("trigger", "trig\u200bger"),
            "nbsp between words": phrase.replace(" ", "\u00a0", 1),
            "combined": "**" + words[0] + "**\u200b " + " ".join(words[1:]),
        }
        for label, blob in evasions.items():
            with self.subTest(evasion=label):
                self.assertNotEqual(blob, phrase, "the fixture is not actually evasive")
                self.assertIn(
                    _normalised(phrase), _normalised(blob),
                    f"a retracted sentence written with {label} walks past the guard",
                )

    def test_the_phrase_guard_catches_a_hard_wrapped_retraction(self) -> None:
        """Live control for the guard above.

        The failure it exists for is real: a phrase guard in this repo matched per line
        and passed on hard-wrapped text it was written to condemn. This feeds it a
        retracted sentence broken across three lines with the wrapping an editor would
        produce, and requires the normalised match to find it.
        """

        phrase = RETRACTED[0]
        words = phrase.split()
        wrapped = "\n".join(
            [" ".join(words[:2]), "   " + " ".join(words[2:4]), "\t" + " ".join(words[4:])]
        )
        self.assertNotIn(phrase, wrapped, "the fixture is not actually hard-wrapped")
        self.assertIn(
            _normalised(phrase), _normalised(wrapped),
            "the normalising guard cannot see a hard-wrapped retraction, which is exactly "
            "the defect it was written to fix",
        )
        # ...and the same fixture LOWERCASED, which is how mutation 8 survived the first
        # version of this guard. Whitespace normalisation alone is not enough.
        self.assertIn(
            _normalised(phrase), _normalised(wrapped.lower()),
            "the guard is case-sensitive again; a retracted sentence typed in lower case "
            "walks straight past it",
        )

    def test_section_eight_states_the_derived_row_count(self) -> None:
        """Section 8 read 81 while the table held 82 and the header pinned 82.

        Found by this pass, and fixed with a pin rather than an edit because an unpinned
        count in this document has now gone stale four times.
        """

        from test_ledger_table_uniformity import _EXPECTED_GAP_ROWS

        derived = sum(len(rows) for rows in _EXPECTED_GAP_ROWS)
        text = _normalised(_ledger_text())
        # `_normalised` case-folds AND strips `*`/`_`/backticks, so the pattern carries
        # neither capitals nor emphasis. A literal with either would silently never match
        # and this pin would fail open on the `assertIsNotNone` branch -- which is exactly
        # what happened when the normaliser gained emphasis-stripping and this line did not.
        stated = re.search(r"non-empty rendered reachability evidence cell on all (\d+)", text)
        self.assertIsNotNone(
            stated,
            "section 8's sentence about the reachability-evidence cell has moved or been "
            "reworded; re-point this pin rather than deleting it",
        )
        self.assertEqual(
            int(stated.group(1)), derived,
            f"section 8 states {stated.group(1)} gap rows; the tables carry {derived}",
        )


class TheDenominatorMatchesTheEmissionSiteTests(unittest.TestCase):
    def test_each_counter_row_records_every_emission_site(self) -> None:
        document = _document()
        for name, record in document["verdicts"].items():
            counter = record.get("counter")
            if not counter:
                continue
            with self.subTest(row=name, pattern=counter["pattern"]):
                rederived = c154.counter_emission(counter["pattern"])
                self.assertEqual(counter, rederived)

    def test_the_world_unsupported_counter_has_two_sites_not_one(self) -> None:
        # A bare citation is inadmissible for a counter with more than one increment, for
        # the same reason a refusal reason with three raise sites cannot be cited by line.
        sites = c154.counter_emission("skip:world_unsupported:")["sites"]
        self.assertEqual(
            len(sites), 2,
            "`skip:world_unsupported:` no longer has exactly two emission sites; a row "
            "citing them has to be re-traced, not re-numbered",
        )
        self.assertEqual({s["function"] for s in sites}, {"_prepare_boundary"})

    def test_the_denominator_is_boundaries_full_round_and_that_is_derived(self) -> None:
        """The refusals fire BEFORE `boundaries_measured` increments.

        Quoting the wrong one of the three scalars in use understated a result by ~80x
        once already, so the name is derived from the enclosing function rather than
        chosen.
        """

        for pattern in ("skip:world_unsupported:", "world_prestate_mismatch:"):
            with self.subTest(pattern=pattern):
                self.assertEqual(
                    c154.counter_emission(pattern)["denominator"], "boundaries_full_round"
                )

    def test_per_game_is_not_read_off_loop_depth(self) -> None:
        """`abort:no_legal_action` is per-game because its statement RETURNS out of
        `run_game`, not because of where it sits in the loop nest. A depth heuristic gets
        it backwards, so the derivation is structural -- asserted here because this module
        depends on that derivation being the structural one."""

        sites = [s for s in c154.emission_sites() if s["pattern"] == "abort:no_legal_action"]
        self.assertEqual(len(sites), 1)
        self.assertTrue(sites[0]["ends_the_game"])
        self.assertGreater(
            len(sites[0]["loops"]), 1,
            "the site is no longer nested, so it no longer demonstrates that per-game is "
            "not a loop-depth property; re-point this pin",
        )


class TheAbsencePinsAreNotVacuousTests(unittest.TestCase):
    """Every zero in this document is a lookup into one dict built from one file."""

    def test_the_pool_census_carries_its_nonzero_controls(self) -> None:
        pool = _document()["pool"]
        self.assertEqual(pool["species"], 220)
        self.assertEqual(pool["sets"], 393)
        self.assertEqual(pool["distinct_moves"], 125)
        self.assertEqual(pool["distinct_abilities"], 71)
        for move, expected in NONZERO_MOVE_CONTROLS.items():
            with self.subTest(move=move):
                self.assertEqual(len(pool["move_species"][move]), expected)
        self.assertGreater(len(pool["move_species"]["spikes"]), 0)
        self.assertGreater(len(pool["move_species"]["encore"]), 0)
        self.assertEqual(len(pool["ability_species"]["Cute Charm"]), 3)
        self.assertEqual(len(pool["ability_species"]["Liquid Ooze"]), 2)
        self.assertEqual(pool["ability_species"]["Sand Stream"], ["tyranitar"])
        self.assertEqual(pool["co_occurrence"]["sleeptalk_sets"], 44)
        self.assertEqual(pool["co_occurrence"]["rest_sets"], 55)
        self.assertEqual(pool["co_occurrence"]["drain_sets"], 3)

    def test_the_generative_census_is_a_real_run(self) -> None:
        gen = _document()["pool"]["generative"]
        self.assertEqual(gen["teams"], c154.GENERATION_TEAMS)
        self.assertEqual(gen["pokemon"], 24000)
        self.assertEqual(gen["distinct_items"], 13)
        self.assertEqual(gen["nature_unset"], gen["pokemon"])
        self.assertEqual(gen["nature_set"], 0)
        self.assertEqual(sum(gen["items"].values()), gen["pokemon"])
        self.assertIn("Leftovers", gen["items"])
        self.assertEqual(gen["maxhp_min"], 1)
        self.assertEqual(
            sorted(gen["species_at_or_below_maxhp_ceiling"]), ["Shedinja"],
            "R14's whole argument is that Shedinja is the pool's only species at or below "
            f"{c154.N5_MAXHP_CEILING} max HP",
        )

    def test_the_volatile_source_scan_still_finds_what_its_predecessor_missed(self) -> None:
        """The control for a corrected instrument.

        The first version of this scan regexed `JSON.stringify` of each dex entry, which
        DROPS FUNCTIONS, so it reported that the same-named move is the only producer of
        all ten volatiles -- and 'confirmed' R23's reason. The two producers it missed are
        the two that matter, and both are handler-installed.
        """

        producers = _document()["pool"]["volatile_producers_by_source"]
        attract = " ".join(producers["attract"])
        self.assertIn("abilities.ts", attract, "Cute Charm's `addVolatile('attract')` is gone")
        self.assertIn("mods/gen3/abilities.ts", attract)
        self.assertTrue(
            any("items.ts" in hit for hit in producers["focusenergy"]),
            "Lansat Berry's `addVolatile('focusenergy')` is gone; without it R23 closes "
            "on a move census that cannot see the item producer",
        )
        self.assertEqual(
            producers["mudsport"], [],
            "a volatile with no handler producer started reporting one; re-adjudicate R23",
        )

    def test_the_data_scan_finds_the_second_foresight_producer(self) -> None:
        by_data = _document()["pool"]["volatile_producers_by_data"]
        self.assertEqual(
            by_data["foresight"], ["foresight", "odorsleuth"],
            "R23's correction rests on `odorsleuth` being a second producer of the "
            "`foresight` volatile",
        )
        self.assertEqual(_document()["pool"]["move_species"]["odorsleuth"], [])


class TheCrossInstrumentCouplingIsCheckedTests(unittest.TestCase):
    """⚠ THIS CLASS ASSERTED THE OPPOSITE ONE REVISION AGO, ON A HAZARD NOBODY MEASURED.

    The artifact was written to `tests/data/` on the argument that its refusal-reason names
    beside pool counts would read to `tests/test_never_fired_counter_census.py` as four
    counters firing. Review copied it into `reports/artifacts/`, bumped the corpus count and
    ran the census: **`Ran 22 tests ... OK`.** The names appear only as string VALUES inside
    prose, and that census's `_evidence_in` says in terms that "A name merely mentioned
    inside prose is NOT evidence".

    So the placement bought nothing and cost the guard the census's own header warns about
    by name. Worse, the control that was supposed to prove the hazard real --
    `assertIn(reason, json.dumps(document))` -- was substring-in-prose, the exact shape the
    matcher excludes. **It could not fail.** Both are replaced here: membership in the
    corpus is asserted, and the control now feeds `_evidence_in` a counter-keyed copy and
    requires it to fire.
    """

    def test_the_artifact_is_inside_the_counter_census_corpus(self) -> None:
        self.assertIn(
            ARTIFACT, counter_artifacts(),
            "the artifact left `counter_artifacts()`'s glob, so the shape-matching census "
            "no longer checks it. That census caught a convenience field that nearly "
            "inverted 46 verdicts one PR ago; leaving the corpus is a silent loss of the "
            "only instrument that would notice the same thing here.",
        )

    def test_the_census_matcher_would_fire_on_a_counter_keyed_number(self) -> None:
        """The control, and unlike its predecessor it can fail.

        Feed `_evidence_in` the committed artifact and a copy carrying one counter-keyed
        number. The real one must be clean and the copy must be caught -- so a future edit
        that files a count under a counter name is provably visible to the corpus census
        rather than merely believed to be.
        """

        from test_never_fired_counter_census import _evidence_in

        reason = "nature_not_neutral"
        patterns = {reason: re.compile(r"(?:^|[^A-Za-z0-9_])" + reason + r"(?:$|[^A-Za-z0-9_])")}
        document = _document()

        self.assertEqual(
            _evidence_in(document, patterns), {},
            "the committed artifact ALREADY reads as a counter firing to the corpus "
            "census. Whatever was just added is keyed by a counter name and carries a "
            "number; move the number out of the name-keyed record, as C153 did.",
        )

        poisoned = json.loads(json.dumps(document))
        poisoned["counters"] = {f"skip:world_unsupported:{reason}": 3}
        self.assertIn(
            reason, _evidence_in(poisoned, patterns),
            "the corpus census does NOT see a counter-keyed nonzero number in this "
            "artifact's shape, so its membership in the corpus is not protecting anything "
            "and the claim in the module docstring is false.",
        )


class TheDerivedClaimsAreDerivedTests(unittest.TestCase):
    """The three sentences review found asserted beyond what was traced."""

    def test_the_heal_subcase_caller_graph_is_re_derived(self) -> None:
        """R10's corrected mechanism, which got R10's own mistake wrong once.

        The correction opens "this cell reasoned from its NAME without tracing its caller"
        and then asserted, untraced, that `heal_subcase` is reached only through
        `ambiguous_unrenderable_slug_with_protect`. There are two routes. The graph is now
        built by reverse reachability over the file and re-derived here.
        """

        committed = _document()["pool"]["heal_subcase_call_graph"]
        rederived = c154.rust_call_graph(c154.EV, "heal_subcase", c154.HEAL_SUBCASE_ROOT)
        self.assertEqual(committed, rederived)
        self.assertEqual(
            rederived["live_entry_points"], ["branch_events"],
            "a second live entry point reaches `heal_subcase`; R10's unemittability "
            "argument is scoped to the Sleep Talk block and must be re-traced",
        )
        # DE-NUMBERED, deliberately, and this is the one edit this PR makes to a gate.
        #
        # The assertion's own message says "the NUMBER of ways into the subgraph ...
        # changed", and that is the claim R10's unemittability argument rests on. The
        # literal it compared was a pair of LINE ADDRESSES inside `render_move_phase`, so it
        # fired on any edit that shifted that function -- a docstring insert would do it --
        # while reporting a semantic change that had not happened. That is the landmine shape
        # report 4 §4.8 records: "de-number a stale citation; never re-point it."
        #
        # Nothing is lost by dropping the addresses. `assertEqual(committed, rederived)`
        # above compares the WHOLE graph, line numbers included, against the committed
        # artifact, so a moved edge still forces a regeneration and a read of the diff; and
        # `test_the_r10_cell_states_the_derived_route_count` holds the ledger prose to this
        # same count. What was unique to the literal was only its address, which is the part
        # that is not a claim about anything.
        self.assertEqual(
            len(rederived["edges_out_of_the_chokepoint"]), 2,
            "the number of ways into the subgraph from `render_move_phase` changed; R10's "
            "unemittability argument is scoped to its Sleep Talk block and must be "
            "re-traced",
        )
        self.assertEqual(len(rederived["dead_wrappers_with_no_production_caller"]), 5)

    def test_the_r10_cell_states_the_derived_route_count(self) -> None:
        """The ledger sentence is held to the graph, not merely written from it.

        It read "reached through ONE non-test path" against a derived TWO, and against this
        document's own retraction of that very claim three sections away -- the fifth
        instance of a correction not reaching the prose describing it, inside the pass whose
        subject is that. Pinning the number is the only thing that stops a sixth.
        """

        routes = len(_document()["pool"]["heal_subcase_call_graph"]["edges_out_of_the_chokepoint"])
        cell = _normalised(_section_four_cells()["R10"])
        self.assertIn(
            _normalised(f"by **{routes}** routes"), cell,
            f"R10's cell does not state the derived route count ({routes}). The graph is "
            "in the artifact; the sentence has to agree with it.",
        )
        self.assertNotIn(
            _normalised("reached through one non-test path"), cell,
            "the withdrawn 'one non-test path' wording is back in R10's cell",
        )

    def test_the_corrections_tally_is_derived_not_typed(self) -> None:
        """Two revisions of the generator's docstring said SEVEN and TEN against THIRTEEN.

        Fifth and sixth instances of this PR's own subject, inside the generator that
        produces the number. The count now exists in one place.
        """

        document = _document()
        rederived = c154.correction_counts(document["verdicts"])
        self.assertEqual(document["counts"], rederived)
        live = {k: v for k, v in document["verdicts"].items() if k != "R26"}
        self.assertEqual(rederived["rows"], 26)
        self.assertEqual(
            rederived["rows_corrected"],
            sum(1 for v in live.values() if v["ledger_reason_status"] != "SOUND"),
        )
        self.assertEqual(
            rederived["rows_corrected"],
            sum(1 for name, v in _section_four_cells().items()
                if name != "R26" and C154_MARKER in v),
            "the corrections tally and the number of marked section 4 cells disagree",
        )
        # No prose in this PR's own files may state the tally as a literal.
        for name in ("scripts/c154_unreachable_readjudication.py",
                     "tests/test_unreachable_readjudication.py"):
            with self.subTest(file=name), open(os.path.join(REPO, name), encoding="utf-8") as fh:
                body = fh.read()
            # Assembled from fragments so this file does not contain the literals it
            # forbids -- a first version condemned itself, which is the same self-match
            # trap the retraction guard needed a quoting exemption for.
            for stale in ("corrections on " + "SEVEN", "ten wrong or " + "incomplete"):
                self.assertNotIn(
                    stale, body,
                    f"{name} states the corrections tally as a literal again; it is "
                    "derived by `correction_counts()` and pinned above",
                )

    def test_the_foreclosure_scope_is_recorded_where_it_is_narrow(self) -> None:
        """`UNREACHABLE_TRACED` was documented as "cannot fire for any caller of X".

        It is not true for every row: `volatile_unsupported` fires today for a caller
        (`struggle_taunt_stall`, a hand-written Custom Game fixture), and R1's raise is one
        keyword argument away. Both verdicts are right over section 4's population; the
        word over-claimed, so the scope is carried per row.
        """

        document = _document()
        for name, record in document["verdicts"].items():
            if name == "R26":
                continue
            with self.subTest(row=name):
                self.assertIn(record["foreclosure"], c154.FORECLOSURES)
                narrow = record["foreclosure"] == "RANDBATS_POPULATION"
                self.assertEqual(narrow, name in c154.NARROW_FORECLOSURE)
                self.assertEqual(bool(record["foreclosure_note"]), narrow)
        self.assertEqual(
            sorted(c154.NARROW_FORECLOSURE), ["R1", "R23", "R24"],
            "a row's foreclosure scope narrowed or widened without the report following",
        )
        self.assertEqual(document["counts"]["randbats_population"], 3)

    def test_the_encore_fail_list_is_a_set_equality_not_a_membership_sweep(self) -> None:
        """R22's claim, which the report called machine-checked while nothing checked it.

        The previous version asserted six MEMBERSHIPS, so adding `Choices::TACKLE` to the
        patch's match arm left the module green -- review demonstrated exactly that. Both
        sides are measured now: the arm is parsed out of the committed patch, and the gen3
        `failencore`-flagged set comes from the dex via the census.
        """

        shipped = c154._patch_match_arm(
            "poke-engine-gen3-encore-failencore.patch", "fn move_fails_encore"
        )
        flagged = tuple(_document()["pool"]["failencore_flagged_gen3"])
        self.assertEqual(
            shipped, flagged,
            "the crate's `move_fails_encore` arm and gen3's `failencore`-flagged move set "
            "are no longer equal. R22's withdrawn clause said the edge cases were closed; "
            "what keeps that from OPENING a row is this equality, so a difference here is "
            "a real finding and not a pin to relax.",
        )
        self.assertEqual(len(shipped), 6)

    def test_a_smuggled_member_breaks_the_set_equality(self) -> None:
        """The control review used, as a fixture: the previous pin passed with TACKLE in."""

        import re as _re

        path = os.path.join(REPO, "third_party/poke-engine-gen3-encore-failencore.patch")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        smuggled = text.replace("+            | Choices::TRANSFORM",
                                "+            | Choices::TACKLE\n+            | Choices::TRANSFORM", 1)
        self.assertNotEqual(smuggled, text, "the smuggle fixture did not apply")
        opening = smuggled.index("matches!(", smuggled.index("fn move_fails_encore"))
        closing = smuggled.index("\n+    )", opening)
        added = "\n".join(l for l in smuggled[opening:closing].splitlines() if l.startswith("+"))
        found = tuple(sorted(set(_re.findall(r"Choices::(\w+)", added))))
        self.assertIn("TACKLE", found, "the parser cannot see an added member at all")
        self.assertNotEqual(
            found, tuple(_document()["pool"]["failencore_flagged_gen3"]),
            "a smuggled member leaves the set equality intact, so it is not an equality",
        )


class EveryWorkflowTestCountGuardMatchesItsModuleTests(unittest.TestCase):
    """⚠ THIS PASS WEAKENED THE GUARD PROTECTING THE OWNER RATIFICATION AND THE BURN.

    Bumping this module's own `Ran N tests` guard was done with an UNBOUNDED string
    replace of `Ran 25 tests` -> `Ran 31 tests` over the whole workflow. Two steps carried
    `Ran 25` at that moment. One was this module's. The other was
    `tests/test_final_holdout_guard.py`, whose step gates `OWNER_RATIFIED`,
    `BURNED_FINAL_HOLDOUT` and the `19,200,000`-`19,200,259` burn -- and whose module this
    PR does not touch at all. It became `Ran 31 tests` for a suite that can only ever print
    25, so the guard could no longer fail closed: the count it demanded was unreachable.
    Nothing in the repository noticed, and the neighbouring `expected 25 final-holdout
    pins` message stayed at 25, which is what made it findable by eye.

    Restoring line 698 is the instance. THIS IS THE CLASS, and the distinction is the
    through-line of the whole pass -- `reports/c131` §6 records the same author correcting
    "the instance a reviewer named and leaving the same defect one surface over". Every
    `Ran N tests` guard in the workflow is re-derived here from its module's AST, so a
    guard that stops matching its suite is red locally instead of inert in CI.

    Two shapes are handled, both present today:

      * a step naming a MODULE -- expected count is the module's `test*` methods;
      * a step naming individual `Module.Class.test_method` paths -- expected count is the
        number of paths named. The contact-ability wake-ordering step is that shape, and a
        first
        version of this pin reported it as a mismatch because it looked for a file.

    Modules using `subTest` are still counted by method: `unittest` prints one `Ran N` per
    test method regardless of subtests, which is why the six `subTest` modules here match.
    A module defining `load_tests` would break that assumption, so its absence is checked
    BY AST rather than assumed -- and by AST rather than by substring, because a substring
    check condemned this module for describing the rule in this very docstring. The AST
    check covers BOTH forms `unittest` honours: `def load_tests(...)` and
    `load_tests = <callable>`. The assignment form was #1205's review residual, named there
    as uncovered and closed here, because "no module in scope uses it today" is a fact
    about today and this pin outlives today.

    ⚠ **THE SCAN ITSELF WAS THE NEXT INSTANCE OF ITS OWN CLASS, and #1205 is that.** Its
    first version looked at most TWELVE lines past each `python -m unittest`. Four steps put
    their `Ran N tests` guard further out, behind long explanatory comments, and for those
    four `_guards()` emitted **no entry at all** -- no error, no skip, just absent. A pin
    silently covering part of its subject is the same failure as a guard demanding a count
    its suite cannot print: neither can fail. Measured before the fix, the guard sat 12, 16,
    23 and 30 lines past its invocation for the denominator pair, the spread-gate pin, the
    seed registry and the stat attestation respectively -- the denominator step by EXACTLY
    the window width, i.e. one line out. Setting the seed-registry step's guard to a count
    its suite can never print left this whole module green.

    So the extent is now DERIVED rather than guessed. A step's shell body is a YAML block
    scalar, whose extent is fixed by indentation, and the pairing rule is: a guard belongs
    to the last executable invocation above it **within the same `run:` body**. That is why
    `_sites()` returns one entry per invocation whether or not it found a guard -- an
    unresolved site has to be REPRESENTED to be assertable, and
    `test_no_executable_unittest_invocation_escapes_the_scan` asserts there are none.
    Comment lines are excluded on both sides: the file carries the invocation string inside
    one comment, and `Ran N tests` inside seven more, and counting either is the self-match
    trap this module has now hit three times.

    ⚠ **THE LINE NUMBER OF THAT COMMENT IS NOT WRITTEN DOWN ANYWHERE, and the reason is
    C156's own defect.** Its first revision cited it as `:1202` in three places in this
    module and once in `reports/c156`, and C156's own eleven-line workflow comment moved it
    to a different line IN THE SAME COMMIT -- four citations stale inside the change that
    staled them, in a pass whose subject is stale typed numbers. Review found it, and then
    found the FIRST TWO FIXES for it defective in turn; the record is in
    `test_the_scan_sees_every_invocation_a_flat_scan_sees`, which is where the working one
    lives. (`#1202` elsewhere in this module is the PULL REQUEST, not a line, and is not
    affected.)
    """

    WORKFLOW = ".github/workflows/engine-fidelity-gates.yml"

    #: One executable invocation of the runner. Comment lines carrying the same string are
    #: NOT sites; the file carries exactly one such comment and counting it into the
    #: denominator already shipped once. A REGEX, not a literal: review's residual G was that `python3 -m unittest` and
    #: `python -munittest` are both real spellings `unittest` honours and a literal
    #: `"python -m unittest" in line` sees neither, so a step written either way would drop
    #: out of coverage with no error -- #1205's shape with a different cause.
    INVOCATION = re.compile(r"python3?\s+-m\s*unittest")

    #: The guard shape. Matched on non-comment lines only, for the same reason.
    GUARD = re.compile(r"Ran (\d+) tests")

    @classmethod
    def _lines(cls) -> list[str]:
        with open(os.path.join(REPO, cls.WORKFLOW), encoding="utf-8") as handle:
            return handle.read().splitlines()

    @staticmethod
    def _run_bodies(lines: list[str]) -> list[list[int]]:
        """Zero-based line indices of every `run:` shell body, by YAML indentation.

        Both scalar shapes are handled: a block (`run: |`), whose body is the following
        run of lines indented deeper than the key, and an inline `run: <command>`, whose
        body is the key line itself. Nothing in this workflow uses the inline shape today
        and `test_the_scan_sees_every_invocation_a_flat_scan_sees` is what would say so
        loudly if one appeared, rather than this returning a quietly short list.
        """

        bodies = []
        for index, line in enumerate(lines):
            head = re.match(r"^(\s*)run:(\s*)(\S.*)?$", line)
            if not head:
                continue
            indent, rest = len(head.group(1)), (head.group(3) or "")
            if not rest or rest[0] in "|>":
                body = []
                for offset in range(index + 1, len(lines)):
                    following = lines[offset]
                    if following.strip() and len(following) - len(following.lstrip()) <= indent:
                        break
                    body.append(offset)
                bodies.append(body)
            else:
                bodies.append([index])
        return bodies

    @classmethod
    def _sites(cls) -> list[tuple[int, tuple[str, ...], int | None, int | None]]:
        """`(invocation_line, targets, guard_line, stated)` per EXECUTABLE invocation.

        `guard_line` and `stated` are `None` for a site whose `run:` body carries no
        `Ran N tests` below the invocation -- #1205's shape, and the reason this returns
        unresolved sites instead of dropping them.
        """

        lines = cls._lines()
        code = [
            index for index, line in enumerate(lines)
            if line.strip() and not line.strip().startswith("#")
        ]
        sites = []
        for body in cls._run_bodies(lines):
            live = [index for index in body if index in set(code)]
            invocations = [
                index for index in live if cls.INVOCATION.search(lines[index])
            ]
            guards = [index for index in live if cls.GUARD.search(lines[index])]
            for position, index in enumerate(invocations):
                stop = invocations[position + 1] if position + 1 < len(invocations) else len(lines)
                # The whole shell command, followed across backslash continuations, so a
                # step invoking TWO modules on one command -- the Denominator rule step
                # is one site with two modules -- yields both targets and neither the next
                # command's. A fixed line window did both jobs wrong at once.
                command, cursor = [lines[index]], index
                while lines[cursor].rstrip().endswith("\\") and cursor + 1 < len(lines):
                    cursor += 1
                    command.append(lines[cursor])
                targets = tuple(re.findall(r"tests\.[A-Za-z0-9_.]+", " ".join(command)))
                mine = [g for g in guards if index < g < stop]
                if mine:
                    found = cls.GUARD.search(lines[mine[0]])
                    assert found is not None
                    sites.append((index + 1, targets, mine[0] + 1, int(found.group(1))))
                else:
                    sites.append((index + 1, targets, None, None))
        return sites

    @classmethod
    def _guards(cls) -> list[tuple[int, tuple[str, ...], int]]:
        return [
            (guard, targets, stated)
            for _, targets, guard, stated in cls._sites()
            if targets and guard is not None and stated is not None
        ]

    @staticmethod
    def _methods(module: str) -> int:
        path = os.path.join(REPO, module.replace(".", "/") + ".py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)
        # BY AST, not by substring. A substring check condemned THIS module, whose own
        # docstring names `load_tests` while explaining the assumption -- the third
        # self-match trap in this PR, after the retraction guard and the tally literals.
        # BOTH forms `unittest` honours, not just the one in scope: #1205's review named
        # `load_tests = <callable>` as the uncovered half, on the ground that no module
        # uses it TODAY. A pin scoped to today is the shape this module exists to remove.
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        } | {
            target.id
            for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        if "load_tests" in defined:
            raise AssertionError(
                f"{module} defines `load_tests`, so its printed count is no longer its "
                "method count and this pin's arithmetic does not apply to it"
            )
        return sum(
            1
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            for body in node.body
            if isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef))
            and body.name.startswith("test")
        )

    def test_every_ran_n_guard_equals_its_suites_test_count(self) -> None:
        # ⚠ THE FLOOR IS DERIVED, NOT TYPED, and #1205 names why: the previous form was
        # `>= 20` at a moment when the scan returned exactly 20, so it had ZERO margin and
        # could not tell "the scan stopped matching the workflow's shape" from "a step was
        # removed". A floor equal to its own subject is not a floor. The workflow's
        # executable `Ran N tests` lines are counted here by a scan that shares no code
        # with `_sites()` -- a flat, comment-excluding pass over the file -- so the two
        # cannot drift into agreement.
        guards = self._guards()
        lines = self._lines()
        written = [
            number for number, line in enumerate(lines, 1)
            if self.GUARD.search(line) and not line.strip().startswith("#")
        ]
        self.assertEqual(
            sorted(line for line, _, _ in guards), written,
            "the set of `Ran N tests` lines this scan RESOLVED is not the set the file "
            "WRITES. A guard the scan cannot reach is a guard nothing re-derives, which "
            "is #1205 exactly; an extra one means the scan has begun claiming comments.",
        )
        for line, targets, stated in guards:
            with self.subTest(line=line, targets=targets):
                modules = [t for t in targets if os.path.exists(
                    os.path.join(REPO, t.replace(".", "/") + ".py"))]
                if modules:
                    derived = sum(self._methods(m) for m in modules)
                else:
                    # Individually named `Module.Class.test_method` paths.
                    derived = len(targets)
                self.assertEqual(
                    derived, stated,
                    f"{self.WORKFLOW}:{line} demands `Ran {stated} tests` from "
                    f"{list(targets)}, which has {derived}. A guard that names a count its "
                    "suite cannot print never fails closed. This pin exists because an "
                    "unbounded string replace in this PR did exactly that to the "
                    "final-holdout step, which gates OWNER_RATIFIED and the burn.",
                )

    def test_no_executable_unittest_invocation_escapes_the_scan(self) -> None:
        """⚠ #1205's SUBJECT, and the assertion that closes it.

        The scan used to resolve 22 of the workflow's 26 executable invocations and say
        nothing about the other four. Coverage is now asserted rather than assumed: every
        executable invocation must carry a guard the scan can reach. A step added without
        a `Ran N tests` guard, or with one placed where the pairing rule cannot see it,
        reddens HERE instead of joining a blind spot nothing enumerates.
        """

        unresolved = sorted(
            (line, targets) for line, targets, guard, _ in self._sites() if guard is None
        )
        self.assertEqual(
            unresolved, [],
            f"{self.WORKFLOW} has executable `{self.INVOCATION.pattern}` steps whose "
            "`Ran N tests` "
            "guard the scan cannot pair: " + repr(unresolved) + ". Either the step has no "
            "count guard -- add one -- or its guard sits outside the step's own `run:` "
            "body, which is where #1205's four blind spots came from.",
        )

    def test_the_scan_sees_every_invocation_a_flat_scan_sees(self) -> None:
        """Anti-vacuity for the assertion above: zero unresolved is trivial if zero sites.

        The site list is built by walking `run:` bodies. A parser that stopped recognising
        a body would drop its invocations ENTIRELY, and "no unresolved sites" would then be
        true and worthless -- the exact shape of the defect being closed. So the site count
        is cross-checked against a flat, structure-free pass over the file, which excludes
        comments because the file carries the invocation string inside one of them and
        counting it into the denominator has already shipped once.

        ⚠ AND NO WORKFLOW LINE NUMBER IS TYPED IN THIS MODULE -- review's finding C, and
        then review's finding on each of the two fixes for it. All three are recorded,
        because the later ones are worse than the first.

          * **The defect.** C156's first revision cited the comment as `:1202` three times
            here and once in `reports/c156`, and C156's own workflow comment moved it in
            the same commit. Four citations stale inside the change that staled them.
          * **Fix 1 reddened on NOISE.** It pinned the typed citation to the computed line,
            so any edit above it turned this module red for a reason having nothing to do
            with guards; the negative control NC1 went red under it. Rejected on that
            measurement, and review reproduced it and withdrew the suggestion.
          * ⚠ **Fix 2 COVERED NOTHING, AND SHIPPED.** It forbade the phrase
            "invocation-carrying comment at `:NNNN`" -- a spelling THIS PASS INVENTED, which
            `grep -rn` finds nowhere in the repository except inside its own regex. None of
            the four real citations was written that way. Restoring two of them VERBATIM
            left this module green. Dead COVERAGE, inside the docstring advertising it as
            the closure, and the third recurrence in this lineage: C154's bullet, then
            #1205, then here.

        So the guard is scoped BY VALUE, not by phrase. Every workflow line this module
        could sensibly cite -- the comment, each executable invocation, each guard -- is
        computed, and this module's own text may not contain any of them as a `:NNNN`
        citation. A phrase is evaded by rewording; a value cannot be, because the value is
        what a citation IS. Verified to fire on all four historical strings and to stay
        green under NC1, which is the criterion fix 1 failed.

        SCOPE, stated because a negative claim is only as wide as the check.

          * Files: `tests/test_unreachable_readjudication.py`, and for the comment line
            only, `reports/c156`. The report's §1 table cites the four sites by line ON
            PURPOSE and says it measured them at `dbb40c5c`; a citation scoped to a commit
            cannot go stale and is not in scope here.
          * ⚠ **It catches a citation typed at its CORRECT value, and cannot catch one that
            is already wrong.** Authoring time is exactly when this defect is born -- C156
            typed `:1202` while the comment WAS at 1202 -- so the guard fires as the mistake
            is made, before the edit that stales it. But a citation that has ALREADY drifted
            equals no computed value and is invisible here. Not hypothetical: the
            contact-ability step's citation named line 469 and had been wrong since #1204
            (the invocation is at 482) and nothing found it -- review found it by hand. Both
            live citations in this module are de-numbered rather than left for the next
            reader.

            ⚠ THAT SENTENCE USED TO CARRY THE STALE NUMBER IN `:NNNN` FORM, AND THIS GUARD
            CAUGHT IT -- from the other end. A stale citation is invisible only while it
            equals no computed value; an unrelated edit above it can MAKE it equal one. The
            Q2 `encore_move_unknown` step added six lines to `FILTERS`, which slid a real
            `Ran N tests` guard onto 469, and the guard fired on a number that had been
            dead prose for two passes. De-numbered rather than exempted: an exemption would
            have to be phrase-scoped, which is the shape fix 1 already failed on. The
            historical fact is unchanged and both numbers are still stated.
        """

        lines = self._lines()
        carrying = [n for n, line in enumerate(lines, 1) if self.INVOCATION.search(line)]
        comments = [n for n in carrying if lines[n - 1].strip().startswith("#")]
        self.assertEqual(
            len(comments), 1,
            "the workflow no longer carries the invocation string inside exactly one "
            "comment; this control's own premise has moved and must be re-measured",
        )
        self.assertEqual(
            [line for line, _, _, _ in self._sites()],
            [n for n in carrying if n not in set(comments)],
            "the `run:`-body walk and a flat scan disagree about which lines invoke the "
            "runner. A body the walk stops recognising takes its steps' guards out of "
            "coverage silently, which is #1205 with a different cause.",
        )

        # ⚠ SCOPED BY VALUE, NOT BY PHRASE -- see the docstring. The phrase-scoped version
        # of this guard matched a spelling this pass invented and nothing else, so all four
        # real citations walked through it and it shipped. A value cannot be evaded by
        # rewording, because the value is what a citation IS.
        sites = self._sites()
        forbidden = {
            "the invocation-carrying comment": [comments[0]],
            "an executable invocation": [line for line, _, _, _ in sites],
            "a `Ran N tests` guard": [g for _, _, g, _ in sites if g is not None],
        }
        with open(os.path.join(REPO, "tests/test_unreachable_readjudication.py"),
                  encoding="utf-8") as handle:
            mine = handle.read()
        for what, numbers in forbidden.items():
            for number in numbers:
                with self.subTest(cites=what, line=number):
                    # The citing LINES are reported rather than `assertNotIn`'s haystack,
                    # which would print this whole module: a failure nobody can read is a
                    # failure nobody acts on.
                    hits = [
                        n for n, text in enumerate(mine.splitlines(), 1)
                        if ":%d" % number in text
                    ]
                    self.assertEqual(
                        hits, [],
                        "tests/test_unreachable_readjudication.py cites a WORKFLOW LINE BY "
                        f"NUMBER ({what}, currently line {number}) at its own lines {hits}."
                        " That citation is right today and goes stale on the next edit "
                        "above it, silently -- which is how C156's four stale citations "
                        "were born. Describe the step; do not number it.",
                    )
        # `reports/c156` may cite the four SITES by line, because its §1 scopes them to
        # `dbb40c5c` and a citation scoped to a commit cannot go stale. The comment line is
        # not one of those and is where the report's own stale citation was.
        with open(os.path.join(REPO, "reports/c156_workflow_guard_scan_closure.md"),
                  encoding="utf-8") as handle:
            report = handle.read()
        self.assertEqual(
            [n for n, text in enumerate(report.splitlines(), 1)
             if ":%d" % comments[0] in text], [],
            "reports/c156 cites the invocation-carrying comment BY LINE NUMBER "
            f"(currently {comments[0]}). Describe it; the line is derived here.",
        )

    def test_every_scanned_module_matches_the_ast_derivations_assumptions(self) -> None:
        """`derived == printed` rests on THREE things `_methods` does not itself check.

        `unittest` prints one line per test method it COLLECTS AND RUNS, and `_methods`
        counts `test*` methods declared directly in a class body. That equals the printed
        count only while:

          (a) **no scanned class inherits its test methods from a base declared in the same
              module** -- those are collected once per subclass and counted once;
          (b) **no non-`TestCase` class carries `test*` methods** -- those are counted here
              and never collected;
          (c) **no class SKIPS AT `setUpClass`** -- a class-level skip contributes ZERO to
              `testsRun`, so the printed total drops below the AST count.

        ⚠ (a) and (b) were pinned by C156's first revision and (b) WAS OVERSTATED: the
        predicate carried `and not bases`, which exempted any class with a plain `Name`
        base, so review appended `class MutantNotATestCase(object)` with a `test*` method,
        bumped the guard, and this module stayed GREEN. It is now the positive form -- a
        class with `test*` methods must name a `TestCase` base -- which is exactly as
        strong as the sentence above rather than a subset of it. Measured across the
        scanned modules: all 110 such classes name `unittest.TestCase` and nothing else, so
        the positive form costs no exemption.

        ⚠ (c) IS NOT PINNED AND IS NOT PINNABLE HERE, and it has a live counterexample in
        this very tree. `python -m unittest tests.test_spread_gate_provenance` prints
        `Ran 1 test ... OK (skipped=2)` on a developer machine against an AST-derived 6,
        because five of its six sit behind a Showdown-dependent `setUpClass` skip and the
        two `setUpClass` calls that raise take their whole classes out of `testsRun`. In
        CI, which is where the guard is graded, the dependency is present and the step
        prints `Ran 6`. So the guard is right and the divergence FAILS CLOSED -- a step
        that skipped at class level in CI would print a smaller number and its exact-count
        guard would go red. What is wrong is only the claim: C156's first revision said
        "two assumptions", and there are three. Named rather than pinned, because the
        printed count under (c) is a property of the CI environment and no local scan can
        establish it.
        """

        modules = sorted({
            target
            for _, targets, _, _ in self._sites()
            for target in targets
            if os.path.exists(os.path.join(REPO, target.replace(".", "/") + ".py"))
        })
        self.assertGreater(len(modules), 20, "the module list collapsed; this pin is vacuous")
        for module in modules:
            with self.subTest(module=module):
                path = os.path.join(REPO, module.replace(".", "/") + ".py")
                with open(path, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read())
                local = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
                for node in tree.body:
                    if not isinstance(node, ast.ClassDef):
                        continue
                    bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
                    methods = [
                        b.name for b in node.body
                        if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and b.name.startswith("test")
                    ]
                    self.assertFalse(
                        bases & local,
                        f"{module}.{node.name} inherits from a class declared in the same "
                        "module, so `_methods`' per-class count no longer equals what "
                        "`unittest` prints for it",
                    )
                    # THE POSITIVE FORM. The previous predicate was
                    # `methods and not (attributed & {"TestCase"}) and not bases`, whose
                    # trailing `and not bases` exempted every class with a plain `Name`
                    # base -- so `class MutantNotATestCase(object)` carrying a `test*`
                    # method passed. Review demonstrated it green. Requiring a TestCase
                    # base outright is the sentence the docstring actually makes.
                    named = {
                        b.attr if isinstance(b, ast.Attribute) else b.id
                        for b in node.bases
                        if isinstance(b, (ast.Attribute, ast.Name))
                    }
                    self.assertFalse(
                        methods and not any(n.endswith("TestCase") for n in named),
                        f"{module}.{node.name} carries {methods} and names no TestCase "
                        f"base (bases: {sorted(named) or 'none'}), so those methods are "
                        "counted by `_methods` and never collected by `unittest`",
                    )

    def test_the_stated_battery_size_is_the_enumerated_lists_length(self) -> None:
        """ALL THREE statements of this module's battery size, derived from the list itself.

        The size is written in three places -- this module's docstring header, the workflow
        step's comment and `reports/c156` -- and none was checked by anything, so C156
        adding entries could have left any of them stale exactly as `reports/c131` §6
        records happening elsewhere. `tests/test_terminal_disposition_register.py` pins its
        own battery this way; the sibling that taught it the rule did not follow it.

        ⚠ **THE REPORT WAS THE ONE THAT WENT STALE, and review found it.** C156's first
        revision pinned the docstring and the workflow comment and left
        `reports/c156_workflow_guard_scan_closure.md` describing the same edit as
        "23 -> 31" while the workflow said 33 -- a typed number nothing derived, in the
        report of the pass whose subject is typed numbers nothing derives. The report is in
        the loop now, and its sentence names the OLD and NEW totals so both move together.
        """

        # Scoped to the battery section: this docstring carries a SECOND numbered list
        # ("WHAT IS PINNED"), and a whole-docstring scan silently concatenated the two.
        battery = (__doc__ or "").split("THE MUTATION BATTERY, ENUMERATED.")[-1]
        self.assertNotEqual(battery, __doc__, "the battery section header moved or was "
                                              "reworded; this pin is scanning the whole "
                                              "docstring and its numbering is meaningless")
        entries = [int(n) for n in re.findall(r"(?m)^ {2,3}(\d+)\. ", battery)]
        self.assertGreater(len(entries), 20, "the battery list collapsed; the counts below "
                                             "are vacuous")
        self.assertEqual(entries, list(range(1, len(entries) + 1)),
                         "the battery is not consecutively numbered, so a gap could hide "
                         "an entry that was dropped rather than caught")
        self.assertEqual(
            int(self._all(r"had to turn this module RED\. All (\d+) do\.", battery)),
            len(entries),
        )
        self.assertEqual(
            int(self._all(r"Battery: (\d+) mutations applied, \1 caught", self._step())),
            len(entries),
            "the workflow comment states a battery size this module's enumerated list does "
            "not have. The comment is the copy a reader meets first.",
        )
        with open(os.path.join(REPO, "reports/c156_workflow_guard_scan_closure.md"),
                  encoding="utf-8") as handle:
            report = handle.read()
        for label, pattern in (
            ("headline", r"\*\*Battery: (\d+) applied, \1 caught"),
            ("change list", r"battery comment 23 . (\d+)\b"),
        ):
            with self.subTest(site=label):
                self.assertEqual(
                    int(self._all(pattern, report)), len(entries),
                    f"reports/c156's {label} states a battery size the enumerated list "
                    "does not have. This is the site review caught at '23 -> 31' against "
                    "a workflow saying 33.",
                )
        # The OLD total in that sentence is the workflow's value at `origin/main`, which is
        # a fact about a commit and cannot be derived from this tree. Pinned as the literal
        # it is, so a future edit cannot quietly redefine what the change was FROM.
        self.assertIn("battery comment 23 ", report)

    @staticmethod
    def _all(pattern: str, haystack: str) -> str:
        found = re.findall(pattern, haystack)
        assert len(found) == 1, f"{pattern!r} matched {len(found)} times, expected 1"
        return found[0]

    @classmethod
    def _step(cls) -> str:
        """The workflow step that runs this module, comment header included."""

        text = "\n".join(cls._lines())
        blocks = re.split(r"(?m)^\n(?=      )", text)
        owning = [b for b in blocks if f"unittest tests.{__name__.split('.')[-1]}" in b]
        assert len(owning) == 1, f"{len(owning)} workflow blocks run this module"
        return owning[0]

    def test_the_final_holdout_guard_is_pinned_at_its_measured_value(self) -> None:
        """Named explicitly, because it is the one that was broken and what it protects."""

        self.assertEqual(self._methods("tests.test_final_holdout_guard"), 25)
        guards = {line: stated for line, targets, stated in self._guards()
                  if targets == ("tests.test_final_holdout_guard",)}
        self.assertEqual(
            list(guards.values()), [25],
            "the final-holdout step's `Ran N tests` guard is not 25. That step gates "
            "OWNER_RATIFIED, BURNED_FINAL_HOLDOUT and the 19,200,000-19,200,259 burn.",
        )


class TheHumanReadingsAreNamedTests(unittest.TestCase):
    """Four judgements no pin carries. Named, with the part that IS checkable checked."""

    def test_the_future_sight_scenario_is_in_the_registry_list_not_the_corpus_list(self) -> None:
        path = os.path.join(REPO, "src/pokezero/golden_corpus_scenarios.py")
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        spec = next(i for i, line in enumerate(lines, 1) if '"future_sight_pending",' in line)
        corpus = next(i for i, line in enumerate(lines, 1) if line.startswith("def scenario_specs("))
        registry = next(
            i for i, line in enumerate(lines, 1) if line.startswith("def interaction_registry_specs(")
        )
        self.assertLess(corpus, registry)
        self.assertGreater(
            spec, registry,
            "the Future Sight scenario moved INTO scenario_specs(); R1's closure is that "
            "it sits in interaction_registry_specs(), which no world-building harness "
            "reads by default. This is now REACHABLE -- re-adjudicate R1.",
        )

    def test_the_scenario_studio_service_builds_no_engine_world(self) -> None:
        # A grep-shaped claim, and the assertion says which grep produced it.
        path = os.path.join(REPO, "src/pokezero/scenario_studio/service.py")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        for symbol in ("world_battle_spec", "battle_spec_from_payload", "engine_search"):
            with self.subTest(symbol=symbol):
                self.assertNotIn(
                    symbol, text,
                    "R7's residual -- scenario_studio parses `nature` with no vocabulary "
                    "check -- is closed only by this module never building an engine "
                    "world. Scope of this negative: the text of "
                    "src/pokezero/scenario_studio/service.py, nothing wider.",
                )

if __name__ == "__main__":
    unittest.main()
