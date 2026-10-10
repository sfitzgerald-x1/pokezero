"""Prospective cap-safe collection planning; synthetic outcomes only."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import collect_search_over_raw_a1_exploration as exploration
import qualify_search_over_raw_a1 as a1
from pokezero.mcts_eval import search_over_raw_collection_amendment as amendment
from pokezero.mcts_eval.search_over_raw import digest, root_contrast
from tests import test_search_over_raw_a1 as fixtures
from tests import test_search_over_raw_a1_exploration as exploration_fixtures


class CollectionAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.helper = exploration_fixtures.ExplorationTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.output = self.helper.output

    def reviewed_fixture(self):
        r = self.helper.registration()
        p = exploration.new_progress(r)
        seeds = r["seeds"]
        complete = [slot["root_id"] for slot in r["source_contract"]["panels"]["exploration"]["root_slots"]
            if slot["source_seed"] == seeds[0] or (slot["source_seed"] == seeds[1] and slot["root_slot"] < 2)]
        failed = f"exploration:{seeds[1]}:2"
        for cell in p["fixed_roster"]:
            if cell["root_id"] in complete:
                cell.update(status="COMPLETE_EXPLORATION", action=0, contrast_interval=[0., 0.])
            elif cell["root_id"] == failed:
                cell.update(status="SELECTED_UNMEASURED", action=0)
        for cell in p["continuation_roster"]:
            if cell["root_id"] in complete or (cell["root_id"] == failed and cell["replicate"] < 3):
                cell.update(status="COMPLETE", action=0, signed_outcome=1)
            elif cell["root_id"] == failed and cell["replicate"] == 3:
                cell.update(status="CAPPED", action=0)
        for seed in seeds[:2]:
            p["source_roster"][str(seed)]["status"] = "COMPLETE"
        p.update(status="FAILED_NO_RETRY", exit_code=1, registration_sha256=digest(r),
            roots_completed=9, source_games_completed=2, retry_authorized=False, holdout_opened=False,
            scientific_strength_evidence=False, phase_a_admission=False, phase_b_authorized=False)
        review = dict(disposition=amendment.PARTIAL_REVIEW, registration_sha256=digest(r),
            source_commit=r["source_commit"], independent_reviewer="synthetic-other-reviewer",
            scientific_admission=False, holdout_authorized=False, retry_authorized=False,
            continuation_authorized=False, fixed_roster=dict(completed_roots=9, cells=dict(
                COMPLETE_EXPLORATION=45, SELECTED_UNMEASURED=5, UNSTARTED_UNCERTAIN=950)))
        return r, p, review

    def test_exact_original_nine_five_186_partition_full_denominators_and_closed_validation(self):
        r, p, review = self.reviewed_fixture()
        before = deepcopy((r, p, review))
        result = amendment.plan_collection_amendment(r, p, review)
        self.assertEqual((r, p, review), before)
        source = result["execution_source_contract"]
        self.assertEqual(source["panels"]["exploration"]["seeds"], r["seeds"][2:])
        self.assertEqual(len(source["panels"]["exploration"]["root_slots"]), 186)
        self.assertEqual((len(result["retained_root_ids"]), len(result["unavailable_root_ids"])), (9, 5))
        self.assertEqual((len(result["retained_cells"]), len(result["retained_aliases"])), (45, 360))
        self.assertEqual([result[key] for key in ("full_root_denominator", "full_seed_denominator",
            "full_selector_denominator", "full_continuation_alias_denominator")], [200, 32, 1000, 8000])
        self.assertEqual(result["phase_a_cohort"], r["phase_a_cohort"])
        self.assertEqual(source["namespace"], r["source_contract"]["namespace"])
        self.assertEqual(source["max_continuation_boundaries"], 250)
        self.assertEqual(result["external_runtime_cap_seconds"], 14400)
        self.assertFalse(result["runtime_authorized"])
        self.assertFalse(result["source_replay_authorized"])
        self.assertFalse(result["holdout_authorized"])
        self.assertNotIn("validation", source["panels"])
        self.assertIn("Outcome-informed", result["analytical_extension"])
        comparable = result["comparable_root_contract"]
        self.assertEqual(comparable["root_slots"], source["panels"]["exploration"]["root_slots"])
        self.assertEqual(comparable["capacity"], 186)
        self.assertEqual(len(comparable["old_source_a2_a3_uncertain_root_ids"]), 14)
        self.assertFalse(comparable["source_replay_authorized"])
        self.assertFalse(comparable["private_persistence_authorized"])
        self.assertIn("staged A2/A3 producer", " ".join(comparable["launch_barriers"]))
        self.assertIn("same sampler streams", comparable["a2"])

    def test_existing_source_replay_or_attempted_remaining_root_cannot_be_admitted(self):
        r, p, review = self.reviewed_fixture()
        p["source_roster"][str(r["seeds"][0])]["status"] = "UNSTARTED_UNCERTAIN"
        with self.assertRaisesRegex(ValueError, "exposed/attempted"):
            amendment.plan_collection_amendment(r, p, review)
        r, p, review = self.reviewed_fixture()
        root = f"exploration:{r['seeds'][2]}:0"
        next(cell for cell in p["fixed_roster"] if cell["root_id"] == root)["status"] = "SELECTION_ATTEMPTED_UNCERTAIN"
        with self.assertRaisesRegex(ValueError, "exposed/attempted"):
            amendment.plan_collection_amendment(r, p, review)

    def test_failed_root_identical_action_is_not_silently_scored_zero(self):
        r, p, review = self.reviewed_fixture()
        cell = next(cell for cell in p["fixed_roster"] if cell["status"] == "SELECTED_UNMEASURED")
        cell["contrast_interval"] = [0., 0.]
        with self.assertRaisesRegex(ValueError, "unavailable root was scored"):
            amendment.plan_collection_amendment(r, p, review)

    def test_duplicate_cell_alias_or_redrawn_root_rejected(self):
        for field in ("fixed_roster", "continuation_roster"):
            r, p, review = self.reviewed_fixture()
            p[field][-1] = deepcopy(p[field][0])
            with self.assertRaisesRegex(ValueError, "duplicate or replacement"):
                amendment.plan_collection_amendment(r, p, review)
        r, p, review = self.reviewed_fixture()
        p["fixed_roster"][-1]["root_id"] = "replacement:123:0"
        with self.assertRaisesRegex(ValueError, "duplicate or replacement"):
            amendment.plan_collection_amendment(r, p, review)

    def test_wrong_review_or_runtime_admission_rejected(self):
        for field in ("scientific_admission", "holdout_authorized", "retry_authorized", "continuation_authorized"):
            r, p, review = self.reviewed_fixture()
            review[field] = True
            with self.assertRaisesRegex(ValueError, "independent closed-attempt review"):
                amendment.plan_collection_amendment(r, p, review)
        r, p, review = self.reviewed_fixture()
        review["registration_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "independent closed-attempt review"):
            amendment.plan_collection_amendment(r, p, review)

    def test_cap_increase_is_not_an_amendment_option(self):
        r, p, review = self.reviewed_fixture()
        r["source_contract"]["max_continuation_boundaries"] = 500
        p["registration_sha256"] = review["registration_sha256"] = digest(r)
        with self.assertRaisesRegex(ValueError, "source roster/caps"):
            amendment.plan_collection_amendment(r, p, review)

    def test_finite_historical_manifest_rehash_and_drift_failure(self):
        r, p, review = self.reviewed_fixture()
        for name, payload in (("registration.json", r), ("terminal.json", p)):
            a1.save_new(self.output / name, payload)
        manifests = []
        selected = {row["root_id"] for row in p["fixed_roster"] if row["status"] != "UNSTARTED_UNCERTAIN"}
        complete = {row["root_id"] for row in p["fixed_roster"] if row["status"] == "COMPLETE_EXPLORATION"}
        for slot in r["source_contract"]["panels"]["exploration"]["root_slots"]:
            if slot["root_id"] not in selected:
                continue
            root = self.output / f"source-{slot['source_seed']}" / f"root-{slot['root_slot']}"
            root.mkdir(parents=True)
            a1.save_new(root / "one.json", dict(synthetic=True))
            hashes = {"one.json": amendment.sha256_file(root / "one.json")}
            manifests.append(dict(path=str(root), file_count=1, file_sha256=hashes,
                manifest_sha256=digest(hashes), fully_completed=slot["root_id"] in complete))
        review.update(evidence_file_sha256={name: amendment.sha256_file(self.output / name)
            for name in ("registration.json", "terminal.json")}, root_evidence_manifests=manifests)
        a1.save_new(self.output / "terminal-independent-review.json", review)
        expected = amendment.sha256_file(self.output / "terminal-independent-review.json")
        result = amendment.load_reviewed_collection(self.output, expected_review_sha256=expected)
        self.assertFalse(result["runtime_authorized"])
        with self.assertRaisesRegex(ValueError, "review file binding drift"):
            amendment.load_reviewed_collection(self.output, expected_review_sha256="0" * 64)
        for field in ("evidence_file_sha256", "root_evidence_manifests"):
            bad = deepcopy(review)
            if field == "evidence_file_sha256":
                del bad[field]["terminal.json"]
            else:
                bad[field].pop()
            original_loads = amendment.json.loads
            def amended_review(text):
                value = original_loads(text)
                return bad if "root_evidence_manifests" in value else value
            with patch.object(amendment.json, "loads", side_effect=amended_review):
                with self.assertRaisesRegex(ValueError, "core evidence|exact completed/failed root"):
                    amendment.load_reviewed_collection(self.output, expected_review_sha256=expected)

    def root_helper(self, **kwargs):
        helper = fixtures.StagedA1Tests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        return helper, helper.root_fixture(**kwargs)

    def test_shared_action_censoring_keeps_all_eight_null_receipts_and_forty_aliases(self):
        helper, (kwargs, p, _) = self.root_helper(capped=True)
        kwargs.update(boundary_censoring=True, completion_status=amendment.COLLECTED)
        a1.collect_root(**kwargs)
        self.assertEqual((p["roots_completed"], p["continuations_completed"], p["continuations_capped"]), (1, 0, 8))
        audit = json.loads((helper.output / "audit.json").read_text())
        self.assertEqual(audit["status"], "UNCERTAIN")
        self.assertTrue(all(row["status"] == "CAPPED" and row["signed_outcome"] is None for row in audit["outcomes"]))
        self.assertEqual(len(audit["outcomes"]), 8)
        self.assertTrue(all(interval == [0., 0.] for interval in audit["contrasts"].values()))
        aliases = [row for row in p["continuation_roster"] if row["root_id"] == kwargs["root"]["root_id"]]
        self.assertEqual(len(aliases), 40)
        self.assertTrue(all(row["status"] == "CAPPED" and row["signed_outcome"] is None for row in aliases))

    def test_distinct_action_censoring_keeps_sharp_uncertainty_not_loss_or_draw(self):
        helper, (kwargs, p, _) = self.root_helper(capped=True)
        original = a1.make_adapter
        def differing(cfg, **options):
            adapter = original(cfg, **options)
            select = adapter.select
            def choose(context, **kw):
                result = select(context, **kw)
                if cfg.arm != "raw":
                    result["action"] = 1
                return result
            adapter.select = choose
            return adapter
        with patch.object(a1, "make_adapter", side_effect=differing):
            a1.collect_root(**kwargs, boundary_censoring=True, completion_status=amendment.COLLECTED)
        audit = json.loads((helper.output / "audit.json").read_text())
        self.assertEqual((p["continuations_completed"], p["continuations_capped"]), (0, 16))
        self.assertEqual(root_contrast(audit, a1.key(a1.configurations()[1])), (-1., 1.))
        self.assertTrue(all(row["signed_outcome"] is None for row in audit["outcomes"]))

    def test_opt_in_does_not_swallow_selection_provenance_or_cleanup_failure(self):
        helper, (kwargs, p, _) = self.root_helper(fail=("incumbent", "oracle"))
        with self.assertRaisesRegex(ValueError, "synthetic selection failure"):
            a1.collect_root(**kwargs, boundary_censoring=True)
        self.assertEqual(p["roots_completed"], 0)
        self.assertFalse((helper.output / "continuation-attempt.json").exists())

    def test_opt_in_environment_exception_still_halts_without_completing_root(self):
        helper, (kwargs, p, _) = self.root_helper()
        kwargs["env"].step = lambda actions: (_ for _ in ()).throw(RuntimeError("synthetic materialization failure"))
        with self.assertRaisesRegex(RuntimeError, "materialization failure"):
            a1.collect_root(**kwargs, boundary_censoring=True)
        self.assertEqual(p["roots_completed"], 0)
        self.assertFalse((helper.output / "audit.json").exists())

    def test_opt_in_provenance_and_cleanup_failures_still_halt(self):
        helper, (kwargs, p, _) = self.root_helper()
        kwargs["verify"].side_effect = ValueError("synthetic provenance drift")
        with self.assertRaisesRegex(ValueError, "provenance drift"):
            a1.collect_root(**kwargs, boundary_censoring=True)
        self.assertEqual(p["roots_completed"], 0)
        helper, (kwargs, p, _) = self.root_helper()
        original = a1.make_adapter
        def unclosed(cfg, **options):
            adapter = original(cfg, **options)
            adapter.close = lambda: (_ for _ in ()).throw(RuntimeError("synthetic cleanup failure"))
            return adapter
        with patch.object(a1, "make_adapter", side_effect=unclosed):
            with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
                a1.collect_root(**kwargs, boundary_censoring=True)
        self.assertEqual(p["roots_completed"], 0)
        self.assertFalse((helper.output / "continuation-attempt.json").exists())

    def test_plan_writer_is_proposal_only_and_cannot_overwrite_old_evidence(self):
        proposal = dict(runtime_authorized=False)
        path = self.output / "proposal.json"
        original = self.output / "old-attempt"
        with patch.object(amendment, "load_reviewed_collection", return_value=proposal):
            self.assertEqual(amendment.main(["--original", str(original), "--output", str(path),
                "--review-sha256", "0" * 64]), 0)
        self.assertEqual(json.loads(path.read_text())["artifact_kind"], "PROPOSAL_ONLY_NOT_EXECUTABLE_REGISTRATION")
        with self.assertRaisesRegex(ValueError, "must not change old evidence"):
            amendment.main(["--original", str(original), "--output", str(original / "registration.json"),
                "--review-sha256", "0" * 64])
        with patch.object(amendment, "load_reviewed_collection", return_value=proposal):
            with self.assertRaises(FileExistsError):
                amendment.main(["--original", str(original), "--output", str(path), "--review-sha256", "0" * 64])

    def test_scientific_setting_change_cannot_enter_collection_only_plan(self):
        r, p, review = self.reviewed_fixture()
        r["configurations"][1]["seconds"] = 3.
        p["registration_sha256"] = review["registration_sha256"] = digest(r)
        with self.assertRaisesRegex(ValueError, "scientific settings"):
            amendment.plan_collection_amendment(r, p, review)

    def test_bank_cannot_enter_legacy_or_different_roster_before_runtime(self):
        from pokezero.mcts_eval.search_over_raw_archive import SealedComparisonBank
        slot = dict(root_id="exploration:123:0", source_seed=123, root_slot=0)
        bank = SealedComparisonBank([slot], parent_memory_limit_bytes=100, memory_usage=lambda: 1)
        self.addCleanup(bank.close)
        with patch.object(a1, "_runtime") as runtime:
            with self.assertRaisesRegex(ValueError, "explicit censored exploration"):
                a1.measure({}, self.output, {}, comparison_bank=bank)
            runtime.assert_not_called()
            contract = dict(boundary_censoring_policy=amendment.CENSORING,
                panels=dict(exploration=dict(root_slots=[dict(slot, root_slot=1)])))
            with self.assertRaisesRegex(ValueError, "differs from executable fixed roster"):
                a1.measure({}, self.output, {}, source_contract_value=contract,
                    panel="exploration", comparison_bank=bank)
            runtime.assert_not_called()

    def test_censored_full_roster_reconciles_then_next_source_collects(self):
        helper, _ = self.root_helper()
        contract = self.helper.small_contract()
        contract["boundary_censoring_policy"] = amendment.CENSORING
        p = exploration.new_progress(dict(source_contract=contract, seeds=contract["panels"]["exploration"]["seeds"]))
        for index, seed in enumerate(contract["panels"]["exploration"]["seeds"]):
            directory = self.output / f"source-{seed}"
            directory.mkdir()
            source = self.helper.source(contract, seed, 2)
            a1.save_new(directory / "source.json", source)
            p["source_roster"][str(seed)]["status"] = "COMPLETE"
            p["source_games_completed"] += 1
            for root in source["roots"]:
                kwargs, _, _ = helper.root_fixture(capped=index == 0)
                root_dir = directory / f"root-{root['root_id'].rsplit(':', 1)[1]}"
                root_dir.mkdir()
                kwargs.update(root=root, output=root_dir, progress=p, namespace=contract["namespace"],
                    boundary_censoring=True, completion_status=amendment.COLLECTED)
                a1.collect_root(**kwargs)
        a1.verify_completion(self.output, p, contract=contract, panel="exploration", completion_status=amendment.COLLECTED)
        self.assertEqual((p["roots_completed"], p["continuations_completed"], p["continuations_capped"]), (3, 8, 16))
        self.assertEqual(len(p["fixed_roster"]), 20)
        self.assertEqual(len(p["continuation_roster"]), 160)
        p["continuations_capped"] = 0
        with self.assertRaisesRegex(ValueError, "full staged work accounting"):
            a1.verify_completion(self.output, p, contract=contract, panel="exploration", completion_status=amendment.COLLECTED)


if __name__ == "__main__":
    unittest.main()
