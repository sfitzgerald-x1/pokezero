"""Sealed/public binding checks including actual excluded Showdown boundaries."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive
from pokezero.policy import PolicyDecision
from pokezero.rollout import RolloutConfig, RolloutDriver
import qualify_search_over_raw_source as DRIVER


class SourceQualificationAdmissionTests(unittest.TestCase):
    def registration(self):
        from test_search_over_raw_opening import registration
        row = registration()
        row.update(schema=DRIVER.SCHEMA, source_contract=DRIVER.engineering_contract())
        return row

    def test_attempt_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(len(DRIVER.claim_attempt(output, self.registration())), 3)
            before = (output / "attempt.json").read_bytes()
            with self.assertRaises(FileExistsError):
                DRIVER.claim_attempt(output, self.registration())
            self.assertEqual(before, (output / "attempt.json").read_bytes())

    def test_redraw_cap_or_scientific_scope_drift_fails_before_claim(self):
        for key, value in (("max_source_boundaries", 500), ("continuation_replicates", 9),
                           ("exclude_opening_requests", False), ("retry_authorized", True)):
            with tempfile.TemporaryDirectory() as directory:
                row = self.registration()
                row["source_contract"][key] = value
                with self.assertRaisesRegex(ValueError, "qualification changed"):
                    DRIVER.claim_attempt(Path(directory), row)
                self.assertFalse((Path(directory) / "attempt.json").exists())

    def test_scientific_admission_is_not_qualification(self):
        row = self.registration()
        row["phase_a_admission"] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "scientific outcomes"):
                DRIVER.claim_attempt(Path(directory), row)

    def test_startup_exposure_roster_cannot_be_changed(self):
        row = self.registration()
        row["seeds"].append(42)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "qualification changed"):
                DRIVER.claim_attempt(Path(directory), row)


class ActualSourceBoundaryArchiveTests(unittest.TestCase):
    def setUp(self):
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        path = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT", "third_party/pokemon-showdown"))
        if not (path / "dist" / "sim" / "battle.js").exists():
            self.skipTest("built Showdown fixture unavailable")
        self.env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=path, set_belief_source=True))
        self.addCleanup(self.env.close)
        self.archive = SealedSourceArchive()
        self.addCleanup(self.archive.close)
        self.boundaries = []

    def collect(self):
        archive = self.archive

        class Policy:
            policy_id = "excluded-boundary-test"

            def select_action_with_context(self, context, *, rng):
                action = next(i for i, legal in enumerate(context.observation.legal_action_mask) if legal)
                if context.player_id == "p1":
                    from pokezero.mcts_eval.head_to_head import public_only_context
                    archive.capture_public(public_only_context(context), action)
                return PolicyDecision(action, self.policy_id)

        def sealed(boundary):
            self.boundaries.append(boundary)
            archive.capture_private(boundary)

        return RolloutDriver(env=self.env, policies={s: Policy() for s in ("p1", "p2")},
            config=RolloutConfig(max_decision_rounds=3, hide_opponent_legal_action_masks=True,
                sealed_pre_step_sink=sealed)).run(seed=DRIVER.FIXTURE_SEED, battle_id="excluded-archive")

    def root(self, result):
        from pokezero.public_decision_corpus import public_decision_records_from_trajectory
        record = public_decision_records_from_trajectory(result.trajectory, acting_player="p1")[-1]
        public = record.to_dict()
        return dict(root_id="excluded:root", source_request_index=record.turn_index,
            public_record=public, public_record_sha256=digest(public))

    def test_actual_midgame_binds_canonical_record_to_private_snapshot(self):
        result = self.collect()
        root = self.root(result)
        context, pending, snapshot = self.archive.selected(root)
        self.assertGreater(context.decision_round_index, 0)
        self.assertIs(snapshot, self.boundaries[root["source_request_index"]].snapshot)
        self.assertTrue(all(step.player_id == "p1" for step in context.trajectory.steps))
        self.assertEqual(set(context.requested_observations), {"p1"})
        self.assertNotIn("snapshot", json.dumps(root))
        # Source and continuation environment are owned by the auditor, not a
        # public search adapter. Restore proves this is an actionable snapshot.
        self.env.restore(snapshot)
        self.assertEqual(self.env.observe("p1").legal_action_mask, context.observation.legal_action_mask)
        self.assertIsNone(self.env.terminal())

    def test_forged_source_action_binding_is_rejected(self):
        root = self.root(self.collect())
        root["public_record"]["recorded_action_index"] = 8
        root["public_record_sha256"] = digest(root["public_record"])
        with self.assertRaises(ValueError):
            self.archive.selected(root)

    def test_private_boundary_cannot_precede_public_selection(self):
        self.collect()
        empty = SealedSourceArchive()
        with self.assertRaisesRegex(ValueError, "public binding"):
            empty.capture_private(self.boundaries[0])

    def test_close_discards_source_snapshots(self):
        root = self.root(self.collect())
        self.archive.close()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.archive.selected(root)


if __name__ == "__main__":
    unittest.main()
