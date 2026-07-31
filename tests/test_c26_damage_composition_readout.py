"""Certification contract for the evidence-only C26 disposition."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
READOUT = REPO_ROOT / "reports" / "c26_damage_composition_tail_readout.json"
PREDICTION = REPO_ROOT / "reports" / "c26_damage_composition_tail_prediction.md"
VERIFIER = REPO_ROOT / "scripts" / "c26_damage_composition_verifier.py"
PINNED_MAIN = "8e81426e98cc9b490020026d455eb653903488c4"
ENGINE_FINGERPRINT = "992186c85b4809f768830fa544209d5c31fee1bbc06be1587fe68698d074ba6e"
WHAT_IDENTITIES = {
    "2000261/31": 4,
    "2000298/23": 2,
    "2000431/32": 2,
    "2000561/67": 5,
    "2100079/7": 4,
    "2400156/29": 12,
    "2401127/54": 20,
    "2500120/60": 4,
    "2500576/7": 4,
    "2600657/49": 4,
    "2601196/46": 4,
}
ARCHIVE_SHAS = [
    "b4a0d2c2a182e693554b3fbe2b241126d367178319cd6d9f690e28e8d524dd59",
    "3259aa315252f505002b5570343f74b2f305ed00f30e1437ad19d081590e2c3a",
    "8003fbc80d86c7d07b2fbb9f39c521093d047ce98b2cb4c2c9497d0df4722d7b",
    "442417c845c9046d7b7e4855dcdc4cba9d20f17a3acf76a447cddfa61882786c",
    "486b635854c62e6eecf0eae54c98dcf53f49940404f14b96c9d2ab5994985d0d",
    "5a9e87fd694da86a7fa36a2c8db0f05b424485328769b1729aee7fca28073a95",
    "962ac6efd52cd4b689154ac083678ae8ba6e28c96465c1b963c4453cbcb953c4",
    "5c0235d98fbc91f72338e51001a4b82506899c46eae3cbdfdfe9d86c5e9e462d",
]


class C26DamageCompositionReadoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.readout = json.loads(READOUT.read_text())

    def test_pinned_baseline_engine_and_archive_contract(self) -> None:
        baseline = self.readout["pinned_baseline"]
        archive = baseline["retained_archive"]

        self.assertEqual(self.readout["schema"], "c26-damage-composition-tail-readout/2")
        self.assertEqual(baseline["commit"], PINNED_MAIN)
        self.assertRegex(baseline["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(baseline["engine_fingerprint"], ENGINE_FINGERPRINT)
        self.assertRegex(baseline["engine_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertEqual(archive["population"], 3821)
        self.assertEqual(
            [shard["label"] for shard in archive["shards"]],
            [f"retained-certification-shard-{index:02d}" for index in range(8)],
        )
        self.assertEqual([shard["sha256"] for shard in archive["shards"]], ARCHIVE_SHAS)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in ARCHIVE_SHAS))

    def test_main_reread_and_exact_c15_identity_accounting(self) -> None:
        control = self.readout["current_main_control"]
        rows = {row["identity"]: row for row in control["rows"]}

        self.assertEqual(control["commit"], PINNED_MAIN)
        self.assertEqual(control["tally"], {"diverged": 3599, "matched": 221, "skip_lossy": 1})
        self.assertEqual(set(self.readout["source_population"]["identities"]), set(WHAT_IDENTITIES))
        self.assertEqual(set(rows), set(WHAT_IDENTITIES))
        self.assertEqual(
            {identity: row["branches"] for identity, row in rows.items()},
            WHAT_IDENTITIES,
        )
        self.assertTrue(all(row["verdict"] == "matched" for row in rows.values()))

        matrix = self.readout["ownership_matrix"]
        self.assertEqual(set(matrix["closed_by_current_main"]), set(WHAT_IDENTITIES))
        for key in ("poison_tail", "matcher", "c27", "rest", "refused"):
            self.assertEqual(matrix[key], [])
        self.assertTrue(self.readout["invariants"]["all_c15_identities_are_accounted_for"])
        self.assertTrue(self.readout["invariants"]["no_double_ownership"])

    def test_rejected_experiment_and_regressions_are_pinned(self) -> None:
        experiment = self.readout["rejected_experiment"]

        self.assertEqual(experiment["status"], "rejected")
        self.assertFalse(experiment["production_code_survives"])
        self.assertEqual(experiment["tally"], {"diverged": 3514, "matched": 306, "skip_lossy": 1})
        self.assertEqual(experiment["verdict_delta"], {"diverged_to_matched": 88, "matched_to_diverged": 3})
        self.assertEqual(
            experiment["clearances_by_recorded_class"],
            {
                "component_missing_in_engine:brn,itemleftovers": 3,
                "component_missing_in_engine:itemleftovers": 12,
                "component_missing_in_engine:itemleftovers,psn": 26,
                "component_missing_in_engine:psn": 15,
                "component_missing_in_engine:sandstorm": 1,
                "limit:roll_divergent_lethality": 17,
                "roll_scaled_component": 14,
            },
        )
        self.assertEqual(sum(experiment["clearances_by_recorded_class"].values()), 88)
        self.assertEqual(
            experiment["mechanism_isolation"]["generic_capped_source_promotion"]["regressions_removed_by_ablation"],
            3,
        )
        self.assertEqual(
            experiment["mechanism_isolation"]["cumulative_tail_scale"]["identity"],
            "2201132/57",
        )

        regressions = {row["identity"]: row for row in experiment["regressions"]}
        self.assertEqual(set(regressions), {"2200760/86", "2300983/40", "2700145/92"})
        self.assertEqual(regressions["2200760/86"]["experiment_class"], "limit:roll_divergent_lethality")
        self.assertEqual(regressions["2300983/40"]["experiment_class"], "roll_scaled_component")
        self.assertEqual(regressions["2700145/92"]["experiment_class"], "component_missing_in_engine:psn")
        self.assertTrue(all("No engine defect is claimed." in row["adjudication"] for row in regressions.values()))

    def test_historical_rows_are_uniformly_refused_and_bounded_runs_fail_closed(self) -> None:
        rows = {row["identity"]: row for row in self.readout["historical_control_rows"]}
        self.assertEqual(set(rows), {"2900889/126", "3400914/75", "1500037/28", "1500174/72"})
        self.assertTrue(all(row["disposition"] == "refused_archive_row_absent" for row in rows.values()))
        self.assertEqual(
            rows["2900889/126"]["bounded_gated_run"],
            {"boundaries_full_round": 119, "required_step": 126, "reached_required_step": False},
        )
        self.assertEqual(
            rows["3400914/75"]["bounded_gated_run"],
            {"boundaries_full_round": 66, "required_step": 75, "reached_required_step": False},
        )
        self.assertTrue(self.readout["invariants"]["no_historical_or_control_row_is_claimed_cleared"])

    def test_final_matcher_has_zero_diff_from_pinned_main(self) -> None:
        completed = subprocess.run(
            ["git", "diff", "--exit-code", PINNED_MAIN, "--", "scripts/engine_transition_differential.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(
            self.readout["final_main_equivalence"]["archive_reread_delta"],
            {"diverged_to_matched": 0, "matched_to_diverged": 0},
        )
        self.assertTrue(self.readout["invariants"]["no_production_matcher_change"])
        self.assertTrue(self.readout["invariants"]["baseline_is_immutable_commit_not_moving_ref"])

    def test_prediction_outcome_amendment_withdraws_stale_claims(self) -> None:
        prediction = PREDICTION.read_text()

        self.assertIn("## Outcome Amendment", prediction)
        self.assertRegex(
            prediction,
            r"withdraws the original reproduction, mechanism, and acceptance\s+claims",
        )
        self.assertIn("2900889/3", prediction)
        self.assertIn("2900889/93", prediction)
        self.assertIn("undispositioned", prediction)
        self.assertNotIn("Both target seeds reproduce", prediction)
        self.assertNotIn("The implementation is acceptable", prediction)

    def test_verifier_refuses_execution_without_retained_inputs(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(VERIFIER)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--shards", completed.stderr)

    @unittest.skipUnless(
        os.environ.get("C26_RETAINED_SHARDS"),
        "retained archive inputs unavailable: archive reread explicitly skipped, not passed",
    )
    def test_optional_full_archive_reread_is_zero_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "c26-verification.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VERIFIER),
                    "--shards",
                    os.environ["C26_RETAINED_SHARDS"],
                    "--json",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            payload = json.loads(output.read_text())
        self.assertEqual(payload["status"], "verified")
        self.assertEqual(payload["population"], 3821)
        self.assertEqual(payload["verdict_delta"], {"diverged_to_matched": 0, "matched_to_diverged": 0})


if __name__ == "__main__":
    unittest.main()
