"""Hold the zero-heal-guard mutation battery's RECORDED RUN to the harness and the tree.

WHY THIS IS NOT THE EXISTING TALLY PATTERN. Every other battery in this repo is pinned by a
regex of the shape ``Battery: (\\d+) mutations applied, \\1 caught``. Review established that
this is a null-world pass, and it is worth stating exactly why, because the shape is about to
be copied again by someone:

  * the BACKREFERENCE makes "applied == caught" unfalsifiable -- the pattern cannot express a
    survivor at all, so an honest "13 applied, 12 caught" REDDENS while both under-reporting
    shapes stay green;
  * it compares a STRING to the length of a hand-written LIST, never to an execution result,
    so a battery of pure prose that was never applied and never run passes.

This module compares a recorded EXECUTION against two independent things: the harness's own
`MUTANTS` table, imported rather than transcribed, and its declared `EXPECTED_EQUIVALENT` set.
The survivor count is not asserted to be zero -- it is asserted to be exactly the declared
equivalences, each with a written reason. A new survivor reddens; so does an equivalence that
has quietly started dying, which is the stale-justification direction the backreference form
cannot see either.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import mutate_zero_heal_guard as harness  # noqa: E402

ARTIFACT = ROOT / "reports/artifacts/zero_heal_guard_mutation_battery.json"


def _doc() -> dict:
    return json.loads(ARTIFACT.read_text())


class TheRecordedBatteryIsHeldToTheHarnessTests(unittest.TestCase):
    def test_the_recorded_mutant_set_is_the_harnesss_own(self) -> None:
        """A battery that exists only as prose fails here. The names are IMPORTED from the
        harness, so adding a mutant to the table without re-running, or describing one that
        the table does not contain, both redden."""

        recorded = [m["name"] for m in _doc()["mutants"]]
        declared = [name for name, _, _ in harness.MUTANTS]
        self.assertEqual(recorded, declared)
        self.assertEqual(len(set(recorded)), len(recorded), "duplicate mutant name")

    def test_the_harness_has_not_changed_since_the_recorded_run(self) -> None:
        """The digest is what stops a stale artifact reading as a fresh measurement -- the
        failure mode the string-and-list-length pattern cannot see."""

        live = hashlib.sha256(
            (ROOT / "scripts/mutate_zero_heal_guard.py").read_bytes()
        ).hexdigest()
        self.assertEqual(
            _doc()["harness_sha256"], live,
            "the harness changed after the recorded run; re-run it and commit the result "
            "rather than editing the artifact",
        )

    def test_every_mutant_was_actually_applied(self) -> None:
        """`NOT_APPLIED` is the third way a battery lies (report 4 section 4.4). A mutant whose
        anchor stopped matching, or that no longer compiles, is not a kill."""

        never = [m["name"] for m in _doc()["mutants"] if m["status"] == "NOT_APPLIED"]
        self.assertEqual(never, [], f"mutants never applied: {never}")

    def test_the_survivors_are_exactly_the_declared_equivalences(self) -> None:
        """THE ASSERTION THE BACKREFERENCE CANNOT MAKE. Survivors are allowed, named, and
        justified -- and the set is pinned in BOTH directions."""

        doc = _doc()
        survived = sorted(m["name"] for m in doc["mutants"] if m["status"] == "SURVIVED")
        self.assertEqual(
            survived, sorted(harness.EXPECTED_EQUIVALENT),
            "the survivor set moved. A NEW survivor is a suite gap; a survivor that has "
            "started dying means its equivalence justification is stale and should be "
            "deleted along with the entry.",
        )
        for name in survived:
            self.assertGreater(
                len(harness.EXPECTED_EQUIVALENT[name]), 120,
                f"{name} survives with no substantive equivalence argument",
            )

    def test_every_kill_names_a_test_that_exists(self) -> None:
        """A phantom killer is a fake kill. Each recorded killer name is grepped for in the
        tree, which is also what catches a rename silently orphaning a pin."""

        doc = _doc()
        for mutant in doc["mutants"]:
            if mutant["status"] != "KILLED":
                continue
            with self.subTest(mutant=mutant["name"]):
                self.assertTrue(mutant["killers"], "killed but named no killing test")
                for killer in mutant["killers"]:
                    leaf = killer.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
                    # SOURCE ONLY. `rust/pokezero-search/target` is a multi-GB build tree and
                    # grepping it turned this gate into a hang rather than a check.
                    haystacks = [
                        ROOT / "rust/pokezero-search/src",
                        ROOT / "rust/pokezero-search/tests",
                        ROOT / "tests",
                    ]
                    found = subprocess.run(
                        ["grep", "-rql", leaf, *[str(h) for h in haystacks]],
                        capture_output=True, text=True)
                    self.assertEqual(
                        found.returncode, 0,
                        f"{mutant['name']} records killer {killer!r}, which no source file "
                        "mentions -- a phantom killer is a fake kill",
                    )

    def test_both_directions_are_represented(self) -> None:
        """A battery of only fail-open mutants cannot see an over-refusal, which is the leak
        that survived elsewhere in this campaign with all 11 original mutants killed."""

        directions = [m["direction"] for m in _doc()["mutants"]]
        self.assertTrue(any(d.startswith("fail-open") for d in directions))
        self.assertTrue(any(d.startswith("fail-safe") for d in directions))
        self.assertTrue(any(d == "counter" for d in directions))

    def test_the_totals_are_derived_and_not_transcribed(self) -> None:
        doc = _doc()
        statuses = [m["status"] for m in doc["mutants"]]
        self.assertEqual(doc["killed"], statuses.count("KILLED"))
        self.assertEqual(doc["survived"], statuses.count("SURVIVED"))
        self.assertEqual(doc["never_applied"], statuses.count("NOT_APPLIED"))
        self.assertEqual(doc["applied"], len(statuses) - statuses.count("NOT_APPLIED"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
