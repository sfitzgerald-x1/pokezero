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
  * the CONTROLS produced their required verdicts -- including the three that must
    read DID NOT RUN and never KILLED;
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
        for name, digest in self.doc["targets"].items():
            with self.subTest(target=name):
                path = ROOT / "src" / "pokezero" / name
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
            "prior rounds",
        ):
            with self.subTest(family=required):
                self.assertIn(required, families)

    def test_the_controls_produced_their_required_verdicts(self) -> None:
        """THE PART THAT MAKES THE COUNTS MEAN ANYTHING.

        A syntax error, a hang and a deleted killer module must each read DID NOT
        RUN and never KILLED; a null edit must read SURVIVED; a known break must
        read KILLED. Without all five, the battery is a number about a runner nobody
        validated.
        """
        controls = {e["name"]: e for e in self.doc["controls"]}
        required = {name: verdict for name, verdict, _edits in harness.CONTROLS}
        self.assertEqual(sorted(controls), sorted(required))
        for name, verdict in required.items():
            with self.subTest(control=name):
                self.assertEqual(controls[name]["status"], verdict)
        for name in ("_control_syntax_error", "_control_deleted_killer", "_control_hang"):
            with self.subTest(control=name):
                self.assertEqual(required[name], "DID NOT RUN")
                self.assertNotEqual(controls[name]["status"], "KILLED")

    def test_the_run_imported_this_tree(self) -> None:
        """`tests/conftest.py` MOVES its own `src` to the front of `sys.path`, so a
        PYTHONPATH-supplied mutant is overridden and appears to survive without ever
        loading. The harness prints the resolved module path from inside the run."""
        resolved = set(self.doc["resolved_modules"])
        self.assertIn(
            f"RESOLVED_ENGINE={ROOT / 'src' / 'pokezero' / 'engine_search.py'}", resolved
        )
        self.assertIn(
            f"RESOLVED_BRIDGE={ROOT / 'src' / 'pokezero' / 'foulplay_bridge.py'}",
            resolved,
        )
        self.assertEqual(len(resolved), 2, f"a run imported something else: {resolved}")

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
