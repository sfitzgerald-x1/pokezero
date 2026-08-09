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

  5. **The retracted sentences are gone, and the guard is whitespace-normalised.** A
     phrase guard that matches per line is blind to a hard-wrapped sentence; this repo has
     shipped that too. The check normalises the WHOLE document to single spaces first, and
     `test_the_phrase_guard_catches_a_hard_wrapped_retraction` feeds it a deliberately
     wrapped copy of a retracted phrase and requires it to fire.

  6. **Cross-instrument coupling is DECLARED AND CHECKED, not arranged.** The artifact is
     keyed by refusal-reason names and carries numbers beside them, so under `reports/` it
     would read to `tests/test_never_fired_counter_census.py` as four counters firing --
     the shape that took the corpus census from green to six failures in C153. The pin
     asserts `counter_artifacts()` does not select it, so the exclusion is a property
     rather than a filesystem accident.

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

 10. **Section 8's own row count is pinned to the derived one.** It read **81** while the
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
  * **Four judgements are HUMAN READINGS and are marked as such on the rows themselves,
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
survived. Each below was applied to a copy of the artifact (or to the generator, or to the
ledger) and had to turn this module RED. All 12 do now; **number 8 SURVIVED the first
version and is the reason this list is here rather than a sentence claiming one exists.**
The guard it defeated already normalised whitespace -- the fix C153 shipped -- but was
case-sensitive, so a retracted sentence re-inserted hard-wrapped AND lower-cased walked
straight through. That is C153's own mutation 24, one document over, in a guard written by
someone who had just read about it. `_normalised` now case-folds and
`test_the_phrase_guard_catches_a_hard_wrapped_retraction` asserts both halves.

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
"""

from __future__ import annotations

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

ARTIFACT = "tests/data/c154_unreachable_readjudication.json"
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


def _normalised(text: str) -> str:
    """The whole document as one line of single-spaced, LOWERCASED words.

    ⚠ CASE-FOLDED, and that was earned by the battery rather than foreseen: mutation 8
    re-inserted a retracted sentence hard-wrapped AND lowercased, and the first version of
    this guard -- which already normalised whitespace -- passed it. C153's reviewer defeated
    its phrase guard the same way and recorded it as mutation 24; the lesson did not travel
    with the rule, so it is enforced here instead of restated.


    ⚠ NOT `for line in text.splitlines()`. A phrase guard in this repo matched per line and
    was therefore blind to every sentence an editor had hard-wrapped -- it passed on text
    it was written to condemn. Normalising the WHOLE document first is the fix, and
    `test_the_phrase_guard_catches_a_hard_wrapped_retraction` is its live control.
    """

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
        # `_normalised` case-folds, so the pattern is lower case; a capitalised literal
        # here would silently never match and this pin would fail open on the None branch.
        stated = re.search(r"non-empty rendered \*\*reachability evidence\*\* cell on all \*\*(\d+)\*\*", text)
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
    def test_the_artifact_is_outside_the_counter_census_corpus(self) -> None:
        """Declared coupling, checked.

        `tests/test_never_fired_counter_census.py` flags a nonzero number under a dotted
        path containing a counter name. This artifact is keyed by refusal-reason names and
        carries pool counts beside them, so under `reports/` it would read as four
        counters firing across the corpus -- a convenience field of exactly this shape
        nearly inverted 46 verdicts in C153. Moving it is not a matter of taste.
        """

        self.assertNotIn(ARTIFACT, counter_artifacts())
        self.assertTrue(
            ARTIFACT.startswith("tests/data/"),
            "the artifact moved out of tests/data/; if it is now under reports/ or docs/ "
            "the corpus census will read its pool counts as counter firings",
        )

    def test_the_names_that_would_collide_are_actually_in_the_artifact(self) -> None:
        """Control for the pin above: prove the collision it avoids is real."""

        blob = json.dumps(_document())
        for reason in ("nature_not_neutral", "weather_unsupported", "volatile_unsupported"):
            with self.subTest(reason=reason):
                self.assertIn(reason, blob)


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

    def test_the_crate_encore_fail_list_matches_gen3s_failencore_flag(self) -> None:
        """R22's withdrawn clause, and the check that stops it from opening a row.

        The clause said the `failencore` edge cases were closed by eight absent moves.
        Three of the six members are reachable, so they are not closed -- but the shipped
        list is still right, because it is exactly gen3's `failencore`-flagged move set.
        Derived from the committed patch here; the dex side is recorded on the row.
        """

        path = os.path.join(REPO, "third_party/poke-engine-gen3-encore-failencore.patch")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("fn move_fails_encore", text)
        for member in ("ENCORE", "MIMIC", "MIRRORMOVE", "SKETCH", "STRUGGLE", "TRANSFORM"):
            with self.subTest(member=member):
                self.assertIn(f"Choices::{member}", text)


if __name__ == "__main__":
    unittest.main()
