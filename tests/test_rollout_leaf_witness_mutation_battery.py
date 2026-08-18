"""Hold the recorded rollout-leaf mutation battery to the harness that produced it.

WHY THIS FILE EXISTS. Round 2's finding was not "the mutants were wrong" -- it was
that "19 mutants, 19 KILLED" shipped WITHOUT ITS LIST, so the number was
unfalsifiable. `tests/test_zero_heal_guard_mutation_battery.py` is the precedent, and
its docstring names the pattern this rejects: a regex like
``Battery: (\\d+) mutations applied, \\1 caught`` over a PR body cannot express a
survivor at all, so an honest "13 applied, 12 caught" reddens while both
under-reporting shapes stay green.

So the artifact is in the repo, the mutant table is in the repo, and the assertions
below are the ones that make the pair checkable:

  * the recorded names ARE the harness's, in order -- not a hand-typed subset;
  * the harness has not changed since the run (sha256 of the script itself);
  * the target files have not changed since the run (sha256 of each);
  * NOT APPLIED is empty -- the third way a battery lies is to report a mutant that
    never reached the tree;
  * DID NOT RUN is a SEPARATE count from SURVIVED, and every entry in it names why;
  * survivors are exactly the declared equivalences, each with a written argument;
  * every kill names a test that exists;
  * every mutant family is represented, so the battery cannot narrow to one finding;
  * the CONTROLS produced their required verdicts -- all SEVEN of them, including the
    three that must read DID NOT RUN and never KILLED, the one that must read NOT
    APPLIED, and the one whose only failures are SUBTEST failures and must still read
    KILLED (requiring a bare `FAILED <nodeid>` line scored three real kills as DID NOT
    RUN);
  * the totals are recomputed from the mutant list rather than read off the header.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "reports" / "artifacts" / "rollout_leaf_witness_mutation_battery.json"

sys.path.insert(0, str(ROOT / "scripts"))
import mutate_rollout_leaf_witness as harness  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RolloutLeafWitnessMutationBatteryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = json.loads(ARTIFACT.read_text())

    def test_the_recorded_mutant_set_is_the_harnesss_own(self) -> None:
        """IMPORTED, not transcribed. A hand-typed list can omit a survivor."""
        self.assertEqual(
            [entry["name"] for entry in self.doc["mutants"]],
            [name for name, _family, _edits in harness.MUTANTS],
        )
        self.assertEqual(
            [entry["name"] for entry in self.doc["controls"]],
            [name for name, _required, _edits in harness.CONTROLS],
        )

    def test_the_harness_has_not_changed_since_the_recorded_run(self) -> None:
        self.assertEqual(
            self.doc["harness_sha256"],
            _sha256(Path(harness.__file__).resolve()),
            "the recorded run was produced by a different version of the harness; "
            "re-run it rather than editing the artifact",
        )

    def test_the_mutated_files_have_not_changed_since_the_recorded_run(self) -> None:
        """The counts are about a TREE. A recorded sweep over a tree that has since
        moved says nothing about this one."""
        # TARGETS ARE NO LONGER ALL IN ONE DIRECTORY. B1's independent-deletion
        # mutants edit `scripts/foulplay_paired_eval.py` and
        # `scripts/foulplay_power_report.py` -- the two READ-path call sites, which are
        # the whole point of that family -- so a hardcoded `src/pokezero/` prefix
        # reported them as "no longer exists". Resolved against the harness's own
        # target list rather than by adding a second hardcoded directory, so a third
        # location does not silently drop out of the pin.
        by_name = {path.name: path for path in harness.ALL_TARGETS}
        self.assertEqual(
            sorted(by_name),
            sorted(self.doc["targets"]),
            "the recorded targets must be exactly the harness's targets",
        )
        for name, digest in self.doc["targets"].items():
            with self.subTest(target=name):
                path = by_name[name]
                self.assertTrue(path.exists(), f"{name} no longer exists")
                self.assertEqual(_sha256(path), digest)

    def test_every_mutant_was_actually_applied(self) -> None:
        """A mutant whose anchor did not match is prose, not a mutant."""
        never = [e for e in self.doc["mutants"] if e["status"] == "NOT APPLIED"]
        self.assertEqual(never, [], "these mutants never reached the tree")
        self.assertEqual(self.doc["not_applied"], 0)

    def test_did_not_run_is_reported_separately_from_survived(self) -> None:
        """The distinction the classifier exists for.

        `pytest` exits 1 on a collection error, so a return-code-only runner scores
        an unimportable tree as a clean sweep of kills. Anything that DID NOT RUN is
        neither a kill nor a survival and is counted on its own line, with a reason.
        """
        stalled = [e for e in self.doc["mutants"] if e["status"] == "DID NOT RUN"]
        for entry in stalled:
            with self.subTest(mutant=entry["name"]):
                self.assertTrue(
                    entry["detail"], "a DID NOT RUN must say why it did not run"
                )
        self.assertEqual(self.doc["did_not_run"], len(stalled))

    def test_the_survivors_are_exactly_the_declared_equivalences(self) -> None:
        survivors = sorted(
            e["name"] for e in self.doc["mutants"] if e["status"] == "SURVIVED"
        )
        self.assertEqual(survivors, sorted(harness.EXPECTED_EQUIVALENT))
        self.assertEqual(
            sorted(self.doc["expected_equivalent"]), sorted(harness.EXPECTED_EQUIVALENT)
        )
        for name, argument in harness.EXPECTED_EQUIVALENT.items():
            with self.subTest(survivor=name):
                self.assertGreater(
                    len(argument),
                    120,
                    "an equivalence has to be argued, not shrugged at",
                )

    def test_every_kill_names_a_test_that_exists(self) -> None:
        """A phantom killer is a fake kill."""
        haystack = "\n".join(
            path.read_text()
            for path in (ROOT / "tests").glob("test_*.py")
        )
        for entry in self.doc["mutants"]:
            if entry["status"] != "KILLED":
                continue
            with self.subTest(mutant=entry["name"]):
                self.assertTrue(entry["killers"], "a kill must name its killer")
                for killer in entry["killers"]:
                    leaf = killer.split("::")[-1].split("[")[0]
                    self.assertIn(
                        leaf,
                        haystack,
                        f"{entry['name']} claims to be killed by {killer}, which "
                        "does not exist in tests/",
                    )

    def test_every_family_is_represented(self) -> None:
        """The battery must span the findings, not concentrate on one."""
        families = {e["family"] for e in self.doc["mutants"]}
        for required in (
            "A1 last hop",
            "A2 encode skip",
            "A4 instrumentation",
            "A5 retraction",
            "A6 schema",
            # Round 4's three families, required for the same reason the others are:
            # a battery that quietly narrows to the previous round's findings is a
            # battery about the code that was already fixed.
            "B1 independent guards",
            "B4 new survivors",
            "B5 narrower items",
            "prior rounds",
        ):
            with self.subTest(family=required):
                self.assertIn(required, families)

    def test_the_controls_produced_their_required_verdicts(self) -> None:
        """THE PART THAT MAKES THE COUNTS MEAN ANYTHING.

        A syntax error, a hang and a deleted killer module must each read DID NOT
        RUN and never KILLED; a null edit must read SURVIVED; a known break must
        read KILLED; an anchor that cannot match must read NOT APPLIED; a break whose
        only failures are SUBTEST failures must read KILLED. Without all SEVEN, the
        battery is a number about a runner nobody validated.

        The sixth was missing, and NOT APPLIED is not a cosmetic verdict: it already
        caught four ambiguous anchors in pass 1. A classifier that defaulted an
        unapplied mutant to SURVIVED would score a hole as a finding; one that
        defaulted it to KILLED would score a hole as a kill. Only a control says which.
        """
        controls = {e["name"]: e for e in self.doc["controls"]}
        required = {name: verdict for name, verdict, _edits in harness.CONTROLS}
        self.assertEqual(sorted(controls), sorted(required))
        # ALL SIX MODES THE CLASSIFIER CAN EMIT are represented, so no verdict is
        # asserted about a state nothing was ever driven into.
        self.assertEqual(
            {"SURVIVED", "KILLED", "DID NOT RUN", "NOT APPLIED"},
            set(required.values()),
            "every verdict the classifier can emit needs a control that produces it",
        )
        self.assertEqual(len(required), 7)
        # THE SEVENTH: a mutant whose only failures are `SUBFAILED(...)` lines. The
        # classifier required a `FAILED <nodeid>` line and therefore read THREE GENUINE
        # KILLS as DID NOT RUN -- flattering, because a DID NOT RUN is excluded from the
        # kill denominator and reads as an instrument problem rather than as a covered
        # defect. The three were all asserted `subTest`-per-field, which is itself
        # deliberate (a fixture that perturbs two things at once pins neither).
        self.assertEqual(controls["_control_subtest_only"]["status"], "KILLED")
        for name, verdict in required.items():
            with self.subTest(control=name):
                self.assertEqual(controls[name]["status"], verdict)
        for name in ("_control_syntax_error", "_control_deleted_killer", "_control_hang"):
            with self.subTest(control=name):
                self.assertEqual(required[name], "DID NOT RUN")
                self.assertNotEqual(controls[name]["status"], "KILLED")
        # THE SIXTH: it must be NOT APPLIED and must not be scored as either of the
        # two verdicts that would flatter the count.
        self.assertEqual(controls["_control_not_applied"]["status"], "NOT APPLIED")
        self.assertNotIn(
            controls["_control_not_applied"]["status"], ("SURVIVED", "KILLED")
        )
        self.assertIn("anchor matched 0 times", controls["_control_not_applied"]["detail"])

    def test_the_run_imported_this_tree(self) -> None:
        """`tests/conftest.py` MOVES its own `src` to the front of `sys.path`, so a
        PYTHONPATH-supplied mutant is overridden and appears to survive without ever
        loading. The harness prints the resolved module path from inside the run.

        B3. PATH-INDEPENDENT, and it was not. The recorded strings were ABSOLUTE
        (rooted in the recording clone's own checkout) while `ROOT` is derived from
        `__file__`, so both `assertIn`s failed in every checkout except the one
        directory that recorded the run -- this gate was RED IN ANY CLONE, which is the
        "works only in the single clone that happens to hold it" class the
        rejected-experiment provenance note in `tests/data/` already documents.

        The property is "the run imported THIS TREE, not a sibling reachable on
        PYTHONPATH". A repo-RELATIVE path states that and says nothing about which
        directory the repo is in. The sha256 pins were never the problem: they are
        content-addressed and relocate for free.
        """
        resolved = set(self.doc["resolved_modules"])
        self.assertIn("RESOLVED_ENGINE=src/pokezero/engine_search.py", resolved)
        self.assertIn("RESOLVED_BRIDGE=src/pokezero/foulplay_bridge.py", resolved)
        self.assertEqual(len(resolved), 2, f"a run imported something else: {resolved}")
        # NO ABSOLUTE PATH SURVIVES ANYWHERE IN THE ARTIFACT. Scoped by VALUE over the
        # whole document rather than by re-checking the two keys just asserted: the
        # defect was one field, and the class is "this artifact only reads correctly in
        # the directory that wrote it".
        def _strings(node):
            if isinstance(node, str):
                yield node
            elif isinstance(node, dict):
                for value in node.values():
                    yield from _strings(value)
            elif isinstance(node, list):
                for value in node:
                    yield from _strings(value)

        absolute = sorted(
            {s for s in _strings(self.doc) if s.startswith("/") or ":\\" in s}
        )
        self.assertEqual(
            absolute,
            [],
            "the recorded run must be readable in any checkout; these strings pin it "
            f"to one directory: {absolute}",
        )
        # And the walker is discriminating: it finds an absolute path when there is one.
        self.assertEqual(
            sorted(_strings({"a": {"b": ["/abs/path", "rel/path"]}})),
            ["/abs/path", "rel/path"],
        )

    def test_the_totals_are_derived_and_not_transcribed(self) -> None:
        mutants = self.doc["mutants"]
        self.assertEqual(
            self.doc["applied"],
            sum(1 for e in mutants if e["status"] != "NOT APPLIED"),
        )
        self.assertEqual(
            self.doc["killed"], sum(1 for e in mutants if e["status"] == "KILLED")
        )
        self.assertEqual(
            self.doc["survived"], sum(1 for e in mutants if e["status"] == "SURVIVED")
        )
        self.assertEqual(
            self.doc["did_not_run"],
            sum(1 for e in mutants if e["status"] == "DID NOT RUN"),
        )
        self.assertEqual(
            self.doc["applied"],
            self.doc["killed"] + self.doc["survived"] + self.doc["did_not_run"],
            "applied must partition into the three verdicts",
        )

    def test_the_killers_are_the_modules_the_finding_spans(self) -> None:
        """The finding was that two modules were tested to different depths, so both
        are killers -- a battery on `engine_search` alone could not see the last hop."""
        self.assertEqual(
            self.doc["killers"],
            [
                "tests/test_rollout_model_priors.py",
                "tests/test_mcts_acceptance_report.py",
            ],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
