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


class ExplorationStageTests(unittest.TestCase):
    def setUp(self):
        from pokezero.mcts_eval import search_over_raw_stages as stages
        self.stages = stages
        self.helper = CollectionAmendmentTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.plan = amendment.plan_collection_amendment(*self.helper.reviewed_fixture())
        self.plan["historical_directory"] = str(self.helper.output / "closed-historical-attempt")
        self.slots = self.plan["comparable_root_contract"]["root_slots"]
        self.slot = self.slots[0]
        self.manifest = dict(state="SEALED", capacity=186, captured_roots=1,
            root_public_bindings={self.slot["root_id"]: "f"*64},
            source_validated_missing_root_ids=[s["root_id"] for s in self.slots[1:]])

    def evidence(self, *, actions=None):
        root = self.slot["root_id"]
        configs = self.stages.a2_configurations()
        actions = actions or {self.stages.key(cfg): 0 for cfg in configs}
        selections = {self.stages.key(cfg): dict(status="SELECTED", action=actions[self.stages.key(cfg)],
            root_id=root + ":" + self.stages.key(cfg), configuration_sha256=cfg.identity,
            statistics_root_id=root, selection_seed=self.slot["source_seed"],
            evidence=dict(belief_draws=[fixtures.row()],
                worker_receipts=[dict(worker=0, evidence=dict(draws=[fixtures.row()]))])) for cfg in configs}
        audit = dict(root_id=root, actions=actions, status="COMPLETE", outcomes=[dict(
            action=action, replicate=rep, status="COMPLETE", signed_outcome=1)
            for action in sorted(set(actions.values())) for rep in range(8)])
        return audit, selections

    def record(self, ledger, audit=None, selections=None, **kwargs):
        if audit is None:
            audit, selections = self.evidence()
        ledger.record_root(audit, selections, public_record_sha256="f"*64,
            selection_seed=self.slot["source_seed"], legal_choices=2, **kwargs)

    def test_exact_a2_roster_and_a3_requires_sealed_global_choice(self):
        configs = self.stages.a2_configurations()
        self.assertEqual(len(configs), 13)
        self.assertEqual(len({self.stages.key(c) for c in configs}), 13)
        self.assertTrue(all(c.seconds == 1. and c.workers == (20 if c.arm == "reference" else 1) for c in configs))
        ledger = self.stages.ExplorationStages(self.plan, self.manifest)
        with self.assertRaisesRegex(ValueError, "prior sealed"):
            ledger.a3_configurations()
        with self.assertRaisesRegex(ValueError, "no early freeze"):
            ledger.freeze_for_a3()
        self.record(ledger)
        frozen = ledger.freeze_for_a3()
        self.assertEqual(len(ledger.a3_configurations()), 13)
        self.assertEqual(sum(c.deployable for c in ledger.a3_configurations()), 6)
        self.assertEqual({c.seconds for c in ledger.a3_configurations()}, {1., 3., 10.})
        self.assertEqual(sum(row["deployable"] for row in frozen["choices"]), 2)

    def test_all_200_slots_and_32_seeds_remain_in_lower_bound_ranking(self):
        ledger = self.stages.ExplorationStages(self.plan, self.manifest)
        self.record(ledger)
        frozen = ledger.freeze_for_a3()
        self.assertEqual((frozen["full_root_denominator"], frozen["full_seed_denominator"],
            frozen["accounted_new_roots"], frozen["unavailable_a2_roots"]), (200, 32, 1, 199))
        chosen = frozen["choices"][0]["candidate_scores"][0]
        quota = sum(s["source_seed"] == self.slot["source_seed"] for s in self.plan[
            "phase_a_cohort"]["panels"]["exploration"]["root_slots"])
        self.assertAlmostEqual(chosen["identification_interval"][0], -1+1/(32*quota))
        self.assertEqual(len(chosen["seed_intervals"]), 32)
        self.assertEqual(chosen["uncertain_roots"], 199)
        self.assertTrue(all(row["choice_basis"] == "PREDECLARED_TIE_DEFAULT_NOT_A_WINNER"
            and row["leaf"] == "model" and row["positive_gain_claim"] is False for row in frozen["choices"]))

    def test_freeze_is_irreversible_and_returns_copy_isolated_choices(self):
        ledger = self.stages.ExplorationStages(self.plan, self.manifest)
        self.record(ledger)
        frozen = ledger.freeze_for_a3()
        frozen["choices"][0]["leaf"] = "raw_rollout"
        self.assertEqual(ledger.a3_configurations()[1].leaf, "model")
        with self.assertRaisesRegex(ValueError, "reselection"):
            ledger.freeze_for_a3()
        with self.assertRaisesRegex(ValueError, "A2 frozen"):
            self.record(ledger)

    def test_original_team_equality_cannot_establish_full_world_or_tree_equality(self):
        ledger = self.stages.ExplorationStages(self.plan, self.manifest)
        self.record(ledger)
        chosen = ledger.freeze_for_a3()["choices"][0]
        self.assertEqual(chosen["original_team_matched_roots"], 1)
        self.assertEqual(chosen["strict_same_world_roots"], 0)
        self.assertFalse(chosen["full_panel_same_world_qualification"])
        self.assertFalse(chosen["causal_evaluator_claim"])

    def test_forced_roots_keep_real_nonempty_search_work_without_becoming_evaluator_evidence(self):
        configs = [c for c in self.stages.a2_configurations() if (c.arm, c.belief) == ("reference", "public")]
        _, selections = self.evidence()
        for cfg, count in zip(configs, (43, 48, 37)):
            selections[cfg.identity]["evidence"]["worker_receipts"][0]["evidence"]["draws"] = [
                fixtures.row(i) for i in range(count)]
        result = self.stages.compare_leaf_worlds(selections, configs, root_id=self.slot["root_id"],
            selection_seed=self.slot["source_seed"], legal_choices=1)
        self.assertEqual(result["attempted_world_counts"], dict(model=43, hp_fraction=48, raw_rollout=37))
        self.assertTrue(result["forced_action_not_evaluator_evidence"])
        self.assertFalse(result["strict_whole_population_match"])
        ledger = self.stages.ExplorationStages(self.plan, self.manifest)
        audit, selections = self.evidence()
        ledger.record_root(audit, selections, public_record_sha256="f"*64,
            selection_seed=self.slot["source_seed"], legal_choices=1)
        self.assertTrue(all(row["substantive_roots"] == 0 and row["strict_same_world_roots"] == 0
            for row in ledger.freeze_for_a3()["choices"]))

    def test_full_attempted_tails_and_cancelled_draws_are_never_common_prefix_filtered(self):
        _, selections = self.evidence()
        configs = [c for c in self.stages.a2_configurations() if (c.arm, c.belief) == ("reference", "public")]
        selections[configs[1].identity]["evidence"]["worker_receipts"][0]["evidence"]["draws"].append(
            dict(ordinal=1, status="DEADLINE_CANCELLED"))
        result = self.stages.compare_leaf_worlds(selections, configs, root_id=self.slot["root_id"],
            selection_seed=self.slot["source_seed"], legal_choices=2)
        self.assertEqual(result["attempted_world_counts"], dict(model=1, hp_fraction=2, raw_rollout=1))
        self.assertEqual(result["unresolved_world_counts"]["hp_fraction"], 1)
        self.assertFalse(result["original_team_population_match"])
        self.assertFalse(result["common_prefix_filtering"])

    def test_duplicate_draw_ordinals_and_stream_namespace_drift_refuse(self):
        configs = [c for c in self.stages.a2_configurations() if (c.arm, c.belief) == ("incumbent", "public")]
        _, selections = self.evidence()
        selections[configs[0].identity]["evidence"]["belief_draws"].append(fixtures.row())
        with self.assertRaisesRegex(ValueError, "duplicate or invalid"):
            self.stages.compare_leaf_worlds(selections, configs, root_id=self.slot["root_id"],
                selection_seed=self.slot["source_seed"], legal_choices=2)
        _, selections = self.evidence()
        selections[configs[1].identity]["statistics_root_id"] += ":different-leaf"
        with self.assertRaisesRegex(ValueError, "namespace differs"):
            self.stages.compare_leaf_worlds(selections, configs, root_id=self.slot["root_id"],
                selection_seed=self.slot["source_seed"], legal_choices=2)

    def test_bound_duplicate_and_action_or_configuration_forgery_refuses(self):
        for mutation in ("action", "configuration_sha256", "root_id"):
            ledger = self.stages.ExplorationStages(self.plan, self.manifest)
            audit, selections = self.evidence()
            row = selections[next(k for k in selections if k != "raw")]
            row[mutation] = 1 if mutation == "action" else "forged"
            with self.assertRaisesRegex(ValueError, "binding drift"):
                self.record(ledger, audit, selections)
        ledger = self.stages.ExplorationStages(self.plan, self.manifest)
        self.record(ledger)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.record(ledger)

    def test_short_source_manifest_cannot_silently_omit_or_replace_roots(self):
        for mutate in (lambda m: m["source_validated_missing_root_ids"].pop(),
                lambda m: m["source_validated_missing_root_ids"].append(self.slot["root_id"]),
                lambda m: m.update(state="CAPTURING")):
            manifest = deepcopy(self.manifest)
            mutate(manifest)
            with self.assertRaisesRegex(ValueError, "exact186"):
                self.stages.ExplorationStages(self.plan, manifest)

    def test_missing_continuation_or_fake_terminal_score_cannot_enter_ranking(self):
        for mutation in (lambda a: a["outcomes"].pop(),
                lambda a: a["outcomes"][0].update(status="CAPPED", signed_outcome=0)):
            ledger = self.stages.ExplorationStages(self.plan, self.manifest)
            audit, selections = self.evidence()
            mutation(audit)
            with self.assertRaises(ValueError):
                self.record(ledger, audit, selections)

    def test_performance_rank_can_choose_hp_without_manufacturing_a_mechanism_claim(self):
        configs = self.stages.a2_configurations()
        actions = {self.stages.key(c): int(c.leaf == "hp_fraction") for c in configs}
        audit, selections = self.evidence(actions=actions)
        for row in audit["outcomes"]:
            row["signed_outcome"] = 1 if row["action"] else -1
        ledger = self.stages.ExplorationStages(self.plan, self.manifest)
        self.record(ledger, audit, selections)
        frozen = ledger.freeze_for_a3()
        self.assertTrue(all(row["leaf"] == "hp_fraction" and row["choice_basis"] == "LOWER_BOUND_PERFORMANCE_RANK"
            and not row["positive_gain_claim"] and not row["causal_evaluator_claim"] for row in frozen["choices"]))

    def test_concrete_collector_runs_thirteen_selectors_and_all_capped_aliases(self):
        helper, (kwargs, p, events) = self.helper.root_helper(capped=True)
        root_id = kwargs["root"]["root_id"]
        roster = self.stages.a2_configurations()
        p["fixed_roster"] = [dict(root_id=root_id, configuration=self.stages.key(c),
            status="UNSTARTED_UNCERTAIN", action=None, contrast_interval=[-1, 1]) for c in roster]
        p["continuation_roster"] = [dict(root_id=root_id, configuration=self.stages.key(c), replicate=rep,
            status="UNSTARTED_UNCERTAIN", action=None, signed_outcome=None) for c in roster for rep in range(8)]
        a1.collect_root(**kwargs, configuration_roster=roster, boundary_censoring=True, matched_statistics_root=True)
        self.assertEqual((p["selections_completed"], p["roots_completed"], p["continuations_capped"]), (13, 1, 8))
        self.assertEqual(len(events), 26)
        self.assertEqual(len(p["continuation_roster"]), 104)
        self.assertTrue(all(row["status"] == "CAPPED" and row["signed_outcome"] is None for row in p["continuation_roster"]))

    def test_deadline_expiry_clears_bank_without_starting_stage_or_scientific_work(self):
        import collect_search_over_raw_a2_stage as stage
        from pokezero.mcts_eval.search_over_raw_archive import SealedComparisonBank
        class Bank(SealedComparisonBank):
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True
        bank = Bank()
        with patch.object(stage, "collect_root") as collector:
            with self.assertRaisesRegex(ValueError, "no new stage clock"):
                stage.collect_a2_from_bank(plan=self.plan, bank=bank, output=self.helper.output / "stage",
                    env=None, evaluator=None, factory=None, checkpoint_contract=None, source=None,
                    showdown="none", verify=lambda: None, deadline_at=10., clock=lambda: 10.)
            collector.assert_not_called()
        self.assertTrue(bank.closed)
        self.assertFalse((self.helper.output / "stage").exists())

    def test_invalid_stage_arguments_clear_bank_and_cannot_write_under_historical_evidence(self):
        import collect_search_over_raw_a2_stage as stage
        from pokezero.mcts_eval.search_over_raw_archive import SealedComparisonBank
        class Bank(SealedComparisonBank):
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True
        base = dict(plan=self.plan, output=self.helper.output / "stage", env=None, evaluator=None,
            factory=None, checkpoint_contract=None, source=None, showdown="none", verify=lambda: None,
            deadline_at=100., clock=lambda: 10.)
        for options in (dict(deadline_at=float("nan")), dict(verify=None),
                dict(output=Path(self.plan["historical_directory"]) / "new-stage")):
            bank = Bank()
            with self.assertRaises(ValueError):
                stage.collect_a2_from_bank(bank=bank, **(base | options))
            self.assertTrue(bank.closed)
        self.assertFalse(Path(self.plan["historical_directory"]).exists())

    def run_synthetic_stage(self, *, fail=None):
        import collect_search_over_raw_a2_stage as stage
        from pokezero.mcts_eval.search_over_raw_archive import SealedComparisonBank
        helper, (kwargs, _, events) = self.helper.root_helper(capped=True, fail=fail)
        root = kwargs["root"]
        root["root_id"] = self.slot["root_id"]
        root["public_record"]["seed"] = self.slot["source_seed"]
        manifest = self.manifest
        class Bank(SealedComparisonBank):
            def __init__(self):
                self.closed = False
            def manifest(self):
                return deepcopy(manifest)
            def validate_roster(self, slots):
                pass
            def selected_for_auditor(self, root_id):
                return root, kwargs["context"], kwargs["pending"], kwargs["snapshot"]
            def close(self):
                self.closed = True
        bank = Bank()
        original = a1.make_adapter
        def adapter_with_stream_receipt(cfg, **options):
            adapter = original(cfg, **options)
            select = adapter.select
            def select_with_binding(context, **opts):
                result = select(context, **opts)
                result.update(statistics_root_id=opts["statistics_root_id"], selection_seed=opts["selection_seed"])
                for receipt in result["evidence"]["worker_receipts"]:
                    receipt["worker"] = 0
                return result
            adapter.select = select_with_binding
            return adapter
        self.bank = bank
        self.stage_kwargs = kwargs
        self.stage_adapter = adapter_with_stream_receipt
        self.stage_output = helper.output / "a2-stage"
        with patch.object(a1, "make_adapter", side_effect=adapter_with_stream_receipt):
            return stage.collect_a2_from_bank(plan=self.plan, bank=bank, output=self.stage_output,
                env=kwargs["env"], evaluator=kwargs["evaluator"], factory=kwargs["factory"],
                checkpoint_contract=kwargs["contract"], source=kwargs["source"], showdown="none",
                verify=lambda: None, deadline_at=100., clock=lambda: 10.), events

    def test_concrete_stage_collects_then_freezes_and_retains_bank_for_a3(self):
        (ledger, progress), events = self.run_synthetic_stage()
        self.assertEqual(progress["status"], "A2_ACCOUNTED_NOT_MECHANISM_QUALIFIED")
        self.assertEqual((progress["roots_completed"], progress["selections_completed"],
            progress["continuations_capped"]), (1, 13, 8))
        self.assertEqual((len(progress["fixed_roster"]), len(progress["continuation_roster"])), (2600, 20800))
        self.assertFalse(self.bank.closed)
        frozen = json.loads((self.stage_output / "a3-exploration-freeze.json").read_text())
        self.assertEqual(frozen["unavailable_a2_roots"], 199)
        self.assertFalse(frozen["runtime_authorized"])
        self.assertEqual(len(ledger.a3_configurations()), 13)
        self.assertEqual(len(events), 26)

    def test_concrete_stage_operational_failure_closes_bank_and_never_freezes(self):
        with self.assertRaisesRegex(ValueError, "synthetic selection failure"):
            self.run_synthetic_stage(fail=("incumbent", "oracle"))
        self.assertTrue(self.bank.closed)
        self.assertFalse((self.stage_output / "a3-exploration-freeze.json").exists())
        terminal = json.loads((self.stage_output / "stage-failure.json").read_text())
        self.assertEqual(terminal["status"], "FAILED_NO_RETRY")
        self.assertEqual(terminal["roots_completed"], 0)
        self.assertFalse(terminal["retry_authorized"])


class A3StageTests(unittest.TestCase):
    def setUp(self):
        self.helper = ExplorationStageTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        (self.ledger, _), self.events = self.helper.run_synthetic_stage()
        self.output = self.helper.stage_output.parent / "a3-stage"

    def collect(self, *, options=None, adapter=None):
        import collect_search_over_raw_a3_stage as stage
        kw = self.helper.stage_kwargs
        kwargs = dict(plan=self.helper.plan, bank=self.helper.bank, ledger=self.ledger,
            a2_output=self.helper.stage_output, output=self.output, env=kw["env"],
            evaluator=kw["evaluator"], factory=kw["factory"], checkpoint_contract=kw["contract"],
            source=kw["source"], showdown="none", verify=lambda: None, deadline_at=100., clock=lambda: 10.)
        kwargs.update(options or {})
        with patch.object(a1, "make_adapter", side_effect=adapter or self.helper.stage_adapter) as factory:
            result = stage.collect_a3_from_bank(**kwargs)
            self.factory_calls = factory.call_count
            return result

    def test_sweep_reuses_all_one_second_selections_and_capped_shared_outcomes_without_retry(self):
        env = self.helper.stage_kwargs["env"]
        with patch.object(env, "restore", wraps=env.restore) as restore:
            readout, progress = self.collect()
        self.assertEqual(self.factory_calls, 8)
        restore.assert_not_called()
        self.assertEqual((progress["selections_completed"], progress["selections_reused"],
            progress["continuations_completed"], progress["continuations_capped"],
            progress["continuations_reused"], progress["continuations_reused_capped"]), (8, 5, 0, 0, 8, 8))
        self.assertEqual((len(progress["fixed_roster"]), len(progress["continuation_roster"])), (2600, 20800))
        self.assertEqual(progress["roots_completed"], 1)
        self.assertTrue(all(row["reused_from_a2"] and row["signed_outcome"] is None
            for row in progress["continuation_roster"] if row["root_id"] == self.helper.slot["root_id"]))
        self.assertFalse(self.helper.bank.closed)
        self.assertEqual(readout["observed_source_seed_count"], 1)
        self.assertFalse(readout["at_least_32_observed_sources"])
        self.assertFalse(readout["original_phase_a_measurement_requirement_satisfied"])
        self.assertTrue(readout["one_second_points_reused_not_independent"])
        self.assertFalse(readout["monotone_gain_claim"])
        self.assertEqual(len(readout["curves"]), 4)
        for curve in readout["curves"]:
            self.assertEqual([row["configuration"]["seconds"] for row in curve["budget_points"]], [1., 3., 10.])
            self.assertTrue(all(row["root_slots"] == 200 and row["source_seeds"] == 32
                and row["uncertain_roots"] == 199 for row in curve["budget_points"]))

    def test_only_new_action_continuations_execute_and_full_null_aliases_remain(self):
        original = self.helper.stage_adapter
        def new_action(cfg, **options):
            adapter = original(cfg, **options)
            select = adapter.select
            def choose(context, **kwargs):
                result = select(context, **kwargs)
                result["action"] = 1
                return result
            adapter.select = choose
            return adapter
        env = self.helper.stage_kwargs["env"]
        with patch.object(env, "restore", wraps=env.restore) as restore:
            readout, progress = self.collect(adapter=new_action)
        self.assertEqual(restore.call_count, 8)
        self.assertEqual((progress["continuations_capped"], progress["continuations_reused_capped"]), (8, 8))
        points = readout["curves"][0]["budget_points"]
        self.assertEqual(points[1]["identification_interval"], [-1., 1.])
        self.assertLess(points[0]["identification_interval"][1], 1.)

    def test_changed_a2_audit_is_refused_before_any_new_selector(self):
        path = self.helper.stage_output / f"source-{self.helper.slot['source_seed']}" / "root-0" / "audit.json"
        # Simulate read-time tampering without overwriting the retained evidence.
        original_loads = json.loads
        def forged(text):
            value = original_loads(text)
            if "continuation_contract" in value:
                value["outcomes"][0]["signed_outcome"] = 0
            return value
        import collect_search_over_raw_a3_stage as stage
        with patch.object(stage.json, "loads", side_effect=forged), patch.object(stage, "collect_root") as collect:
            with self.assertRaisesRegex(ValueError, "prior A2 evidence hash drift"):
                self.collect()
            collect.assert_not_called()
        self.assertTrue(self.helper.bank.closed)
        self.assertEqual(original_loads(path.read_text())["outcomes"][0]["signed_outcome"], None)

    def test_changed_freeze_cannot_select_a_new_leaf(self):
        import collect_search_over_raw_a3_stage as stage
        original_loads = json.loads
        def forged(text):
            value = original_loads(text)
            if "choices" in value:
                value["choices"][0]["leaf"] = "hp_fraction"
            return value
        with patch.object(stage.json, "loads", side_effect=forged), patch.object(stage, "collect_root") as collect:
            with self.assertRaisesRegex(ValueError, "freeze binding drift"):
                self.collect()
            collect.assert_not_called()
        self.assertTrue(self.helper.bank.closed)
        self.assertFalse(self.output.exists())

    def test_deadline_cannot_be_extended_and_expiry_closes_bank_before_stage(self):
        for options, pattern in ((dict(deadline_at=101.), "cannot extend"),
                (dict(clock=lambda: 100.), "no new stage clock")):
            with self.assertRaisesRegex(ValueError, pattern):
                self.collect(options=options)
            self.assertTrue(self.helper.bank.closed)
        self.assertFalse(self.output.exists())

    def test_stage_failure_preserves_a2_but_closes_bank_and_never_publishes_readout(self):
        original = self.helper.stage_adapter
        def refused(cfg, **options):
            adapter = original(cfg, **options)
            adapter.select = lambda *a, **k: (_ for _ in ()).throw(ValueError("new selection refused"))
            return adapter
        frozen = (self.helper.stage_output / "a3-exploration-freeze.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "new selection refused"):
            self.collect(adapter=refused)
        self.assertTrue(self.helper.bank.closed)
        self.assertEqual(frozen, (self.helper.stage_output / "a3-exploration-freeze.json").read_bytes())
        self.assertFalse((self.output / "exploratory-readout.json").exists())
        failure = json.loads((self.output / "stage-failure.json").read_text())
        self.assertEqual(failure["status"], "FAILED_NO_RETRY")
        self.assertEqual(failure["selections_reused"], 2)  # raw and first group's 1s; no fresh work completed.

    def test_forged_manifest_and_matching_argument_cannot_extend_live_a2_deadline(self):
        import collect_search_over_raw_a3_stage as stage
        original_loads = json.loads
        def forged(text):
            value = original_loads(text)
            if "bank" in value and "original_deadline_at" in value:
                value["original_deadline_at"] = 200.
            return value
        with patch.object(stage.json, "loads", side_effect=forged), patch.object(stage, "collect_root") as collect:
            with self.assertRaisesRegex(ValueError, "bound original A2 execution deadline"):
                self.collect(options=dict(deadline_at=200., clock=lambda: 150.))
            collect.assert_not_called()
        self.assertTrue(self.helper.bank.closed)
        self.assertFalse(self.output.exists())

    def test_a3_is_single_use_even_after_success_and_readout_requires_complete_pass(self):
        with self.assertRaisesRegex(ValueError, "full fixed accounting"):
            self.ledger.a3_readout()
        self.collect()
        with self.assertRaisesRegex(ValueError, "no retry"):
            self.collect(options=dict(output=self.output.parent / "a3-retry"))
        self.assertTrue(self.helper.bank.closed)

    def test_existing_stage_output_or_historical_target_never_overwrites_evidence(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.collect()
        self.assertTrue(self.helper.bank.closed)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_deadline_reached_between_selectors_keeps_partial_progress_uncertain(self):
        count = 0
        def clock():
            nonlocal count
            count += 1
            return 100. if count >= 6 else 10.
        with self.assertRaisesRegex(ValueError, "no new stage clock"):
            self.collect(options=dict(clock=clock))
        failure = json.loads((self.output / "stage-failure.json").read_text())
        self.assertEqual(failure["roots_completed"], 0)
        self.assertTrue(any(row["status"] == "UNSTARTED_UNCERTAIN" for row in failure["fixed_roster"]))
        self.assertTrue(self.helper.bank.closed)

    def test_prior_root_binding_change_cannot_reuse_even_hash_valid_outcomes(self):
        kw = self.helper.stage_kwargs
        prior_dir = self.helper.stage_output / f"source-{self.helper.slot['source_seed']}" / "root-0"
        audit = json.loads((prior_dir / "audit.json").read_text())
        selections = {self.helper.stages.key(cfg): json.loads((prior_dir / f"{self.helper.stages.key(cfg)}-selected.json").read_text())
            for cfg in self.helper.stages.a2_configurations()}
        self.ledger.begin_a3(self.helper.plan, self.helper.manifest,
            json.loads((self.helper.stage_output / "a3-exploration-freeze.json").read_text()), deadline_at=100.)
        reuse = self.ledger.a3_reuse(self.helper.slot["root_id"], audit, selections)
        kw = dict(kw, output=self.output, reused_root_evidence=reuse,
            configuration_roster=self.ledger.a3_configurations(), matched_statistics_root=True, boundary_censoring=True)
        self.output.mkdir()
        progress = deepcopy(kw["progress"])
        configs = self.ledger.a3_configurations()
        progress["fixed_roster"] = [dict(root_id=self.helper.slot["root_id"], configuration=self.helper.stages.key(c)) for c in configs]
        progress["continuation_roster"] = [dict(root_id=self.helper.slot["root_id"], configuration=self.helper.stages.key(c), replicate=r)
            for c in configs for r in range(8)]
        kw.update(progress=progress, namespace="different-continuation-namespace")
        with patch.object(a1, "make_adapter") as construct:
            with self.assertRaisesRegex(ValueError, "continuation/root contract drift"):
                a1.collect_root(**kw)
            construct.assert_not_called()

    def test_shared_capped_outcome_forgery_cannot_enter_a3_readout(self):
        self.collect()
        directory = self.output / f"source-{self.helper.slot['source_seed']}" / "root-0"
        audit = json.loads((directory / "audit.json").read_text())
        selections = {self.helper.stages.key(cfg): json.loads((directory / f"{self.helper.stages.key(cfg)}-selected.json").read_text())
            for cfg in self.ledger.a3_configurations()}
        # A separate ledger follows the same freeze, but has not recorded A3 yet.
        other = self.helper.stages.ExplorationStages(self.helper.plan, self.helper.manifest, original_deadline_at=100.)
        prior_dir = self.helper.stage_output / directory.relative_to(self.output)
        prior = json.loads((prior_dir / "audit.json").read_text())
        prior_selections = {self.helper.stages.key(cfg): json.loads((prior_dir / f"{self.helper.stages.key(cfg)}-selected.json").read_text())
            for cfg in self.helper.stages.a2_configurations()}
        self.helper.record(other, prior, prior_selections)
        frozen = other.freeze_for_a3()
        other.begin_a3(self.helper.plan, self.helper.manifest, frozen, deadline_at=100.)
        audit["outcomes"][0].update(status="COMPLETE", signed_outcome=1)
        with self.assertRaisesRegex(ValueError, "changed or retried"):
            other.record_a3_root(audit, selections)


class CachedContinuationTests(unittest.TestCase):
    def cache(self):
        return dict(root_id="root", namespace="namespace", subject="p1", max_boundaries=250,
            replicates=8, outcomes=[dict(action=0, replicate=r, boundaries=250,
                status="CAPPED", signed_outcome=None) for r in range(8)])

    def run_cache(self, cache):
        from pokezero.mcts_eval.search_over_raw import paired_continuations
        return paired_continuations(env=None, snapshot=None, subject="p1", actions=dict(raw=0),
            evaluator=None, namespace="namespace", root_id="root", max_boundaries=250, cached_outcomes=cache)

    def test_cache_requires_all_eight_and_refuses_duplicates_invalid_scores_or_refusals(self):
        for mutate in (lambda c: c["outcomes"].pop(),
                lambda c: c["outcomes"].append(deepcopy(c["outcomes"][0])),
                lambda c: c["outcomes"][0].update(status="REFUSED"),
                lambda c: c["outcomes"][0].update(signed_outcome=0),
                lambda c: c["outcomes"][0].update(boundaries=251),
                lambda c: c.update(namespace="other")):
            cache = self.cache()
            mutate(cache)
            with self.assertRaises(ValueError):
                self.run_cache(cache)

    def test_cache_is_copy_isolated_and_preserves_null_scores_and_zero_fresh_attempts(self):
        from pokezero.mcts_eval.search_over_raw import paired_continuations
        cache = self.cache()
        attempts, reused = [], []
        audit = paired_continuations(env=None, snapshot=None, subject="p1", actions=dict(raw=0),
            evaluator=None, namespace="namespace", root_id="root", max_boundaries=250,
            cached_outcomes=cache, attempt_sink=attempts.append, reuse_sink=reused.append)
        self.assertEqual(attempts, [])
        self.assertEqual(len(reused), 8)
        self.assertEqual(root_contrast(audit, "raw"), (0., 0.))
        self.assertEqual(audit["status"], "UNCERTAIN")
        audit["outcomes"][0]["signed_outcome"] = 0
        self.assertIsNone(cache["outcomes"][0]["signed_outcome"])


if __name__ == "__main__":
    unittest.main()
