"""Sealed/public binding checks including actual excluded Showdown boundaries."""
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from pokezero.mcts_eval.search_over_raw import digest, select_source_requests
from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive, SealedComparisonBank
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

    def leaf_registration(self):
        from dataclasses import asdict
        row = self.registration()
        row.update(schema=DRIVER.LEAF_SCHEMA, source_contract=DRIVER.engineering_contract(True),
            configurations=[asdict(c) for c in DRIVER.qualification_configurations(10., True)])
        return row

    def test_leaf_qualification_has_distinct_namespace_and_configurations(self):
        row = self.leaf_registration()
        self.assertNotEqual(row["source_contract"]["namespace"], DRIVER.NAMESPACE)
        with tempfile.TemporaryDirectory() as directory:
            configs = DRIVER.claim_attempt(Path(directory), row)
        self.assertEqual([DRIVER.configuration_key(c) for c in configs],
            ["raw", "incumbent", "reference", "reference-hp_fraction", "reference-raw_rollout"])
        self.assertEqual([c.leaf for c in configs[-3:]], ["model", "hp_fraction", "raw_rollout"])

    def test_leaf_qualification_rejects_uniform_or_model_fallback_relabeling(self):
        for key, value in (("raw_rollout_policy", "uniform"), ("raw_rollout_cap", 1000),
                ("capped_rollout", "hp_fallback"), ("expired_rollout", "model_fallback")):
            row = self.leaf_registration()
            row["source_contract"]["reference_leaf_ablation"][key] = value
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "qualification changed"):
                    DRIVER.claim_attempt(Path(directory), row)
                self.assertFalse((Path(directory) / "attempt.json").exists())

    def test_leaf_qualification_rejects_budget_arm_or_valuation_drift(self):
        for key, value in (("seconds", 3.), ("belief", "oracle"), ("leaf", "model"), ("workers", 1)):
            row = self.leaf_registration()
            row["configurations"][-1][key] = value
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "arm/resource drift"):
                    DRIVER.claim_attempt(Path(directory), row)
                self.assertFalse((Path(directory) / "attempt.json").exists())

    def test_source_qualification_cannot_admit_an_undeclared_budget(self):
        row = self.registration()
        for config in row["configurations"]:
            config["seconds"] = 2.
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "undeclared qualification budget"):
                DRIVER.claim_attempt(Path(directory), row)
            self.assertFalse((Path(directory) / "attempt.json").exists())

    def oracle_registration(self):
        from dataclasses import asdict
        row = self.registration()
        row.update(schema=DRIVER.ORACLE_SCHEMA, source_contract=DRIVER.engineering_contract(oracle=True),
            configurations=[asdict(c) for c in DRIVER.qualification_configurations(10., oracle=True)])
        return row

    def test_oracle_qualification_separate_namespace_roster_and_nondeployment(self):
        row = self.oracle_registration()
        self.assertNotIn(row['source_contract']['namespace'], (DRIVER.NAMESPACE, DRIVER.LEAF_NAMESPACE))
        with tempfile.TemporaryDirectory() as directory:
            configs = DRIVER.claim_attempt(Path(directory), row)
        self.assertEqual([DRIVER.configuration_key(c) for c in configs], ['raw', 'incumbent',
            'reference', 'incumbent-oracle', 'reference-oracle', 'reference-oracle-hp_fraction',
            'reference-oracle-raw_rollout'])
        self.assertTrue(all(not c.deployable for c in configs[3:]))
        self.assertFalse(row['source_contract']['team_oracle_diagnostic']['deployment_authorized'])

    def test_oracle_qualification_rejects_live_hidden_state_and_sampled_fallback(self):
        for key, value in (('information_scope', 'current_opponent_private_state'),
                ('sampled_fallback', 'allow'), ('deployment_authorized', True), ('raw_rollout_cap', 500)):
            row = self.oracle_registration()
            row['source_contract']['team_oracle_diagnostic'][key] = value
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, 'qualification changed'):
                    DRIVER.claim_attempt(Path(directory), row)
                self.assertFalse((Path(directory) / 'attempt.json').exists())

    def test_oracle_qualification_does_not_alias_leaf_or_default_mode(self):
        with self.assertRaisesRegex(ValueError, 'undeclared qualification'):
            DRIVER.qualification_configurations(10., reference_leaves=True, oracle=True)
        row = self.oracle_registration()
        row['configurations'][3]['belief'] = 'public'
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'arm/resource drift'):
                DRIVER.claim_attempt(Path(directory), row)

    def hp_registration(self):
        from dataclasses import asdict
        row = self.registration()
        row.update(schema=DRIVER.INCUMBENT_HP_SCHEMA,
            source_contract=DRIVER.engineering_contract(incumbent_hp=True),
            configurations=[asdict(c) for c in DRIVER.qualification_configurations(10., incumbent_hp=True)])
        return row

    def test_incumbent_hp_qualification_preserves_model_tree_and_separate_namespace(self):
        row = self.hp_registration()
        self.assertNotIn(row["source_contract"]["namespace"],
            (DRIVER.NAMESPACE, DRIVER.LEAF_NAMESPACE, DRIVER.ORACLE_NAMESPACE))
        with tempfile.TemporaryDirectory() as directory:
            configs = DRIVER.claim_attempt(Path(directory), row)
        self.assertEqual([DRIVER.configuration_key(c) for c in configs], ["raw", "incumbent", "reference",
            "incumbent-hp_fraction", "incumbent-oracle", "incumbent-oracle-hp_fraction"])
        ablation = row["source_contract"]["incumbent_hp_ablation"]
        self.assertEqual(ablation["tree"], "unchanged_encoded_model_tree")
        self.assertEqual(ablation["model_forwards"], "retained")
        self.assertFalse(ablation["oracle_deployable"])

    def test_incumbent_hp_qualification_refuses_relabeling_before_claim(self):
        for key, value in (("tree", "hp_fraction_crate"), ("priors", "uniform"),
                ("model_forwards", "skipped"), ("value_frame", "self_relative"), ("oracle_deployable", True)):
            row = self.hp_registration()
            row["source_contract"]["incumbent_hp_ablation"][key] = value
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "qualification changed"):
                    DRIVER.claim_attempt(Path(directory), row)
                self.assertFalse((Path(directory) / "attempt.json").exists())

    def test_incumbent_hp_budget_and_mode_are_explicit(self):
        for kwargs in (dict(incumbent_hp=True, oracle=True), dict(incumbent_hp=True, reference_leaves=True)):
            with self.assertRaisesRegex(ValueError, "undeclared qualification"):
                DRIVER.qualification_configurations(10., **kwargs)
        with self.assertRaisesRegex(ValueError, "ten-second budget"):
            DRIVER.qualification_configurations(3., incumbent_hp=True)
        row = self.hp_registration()
        row["configurations"][-1]["leaf"] = "model"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "arm/resource drift"):
                DRIVER.claim_attempt(Path(directory), row)


class IncumbentRawQualificationAdmissionTests(unittest.TestCase):
    def registration(self):
        from dataclasses import asdict
        row = SourceQualificationAdmissionTests().registration()
        row.update(schema=DRIVER.INCUMBENT_RAW_SCHEMA,
            source_contract=DRIVER.engineering_contract(incumbent_raw=True),
            configurations=[asdict(c) for c in DRIVER.qualification_configurations(10., incumbent_raw=True)])
        return row

    def test_distinct_namespace_same_tree_both_seats_and_no_scientific_admission(self):
        row = self.registration()
        self.assertNotIn(row["source_contract"]["namespace"],
            (DRIVER.NAMESPACE, DRIVER.LEAF_NAMESPACE, DRIVER.ORACLE_NAMESPACE, DRIVER.INCUMBENT_HP_NAMESPACE))
        self.assertNotEqual(row["source_contract"]["namespace"], DRIVER.CLOSED_INCUMBENT_RAW_NAMESPACE)
        self.assertNotEqual(row["schema"], DRIVER.CLOSED_INCUMBENT_RAW_SCHEMA)
        with tempfile.TemporaryDirectory() as directory:
            configs = DRIVER.claim_attempt(Path(directory), row)
        self.assertEqual([DRIVER.configuration_key(c) for c in configs], ["raw", "incumbent", "reference",
            "incumbent-raw_rollout", "incumbent-oracle", "incumbent-oracle-raw_rollout"])
        declared = row["source_contract"]["incumbent_raw_terminal_ablation"]
        self.assertEqual(declared["tree"], "unchanged_encoded_model_tree")
        self.assertEqual(declared["raw_rollout_policy"], "both_seats_own_raw_masked_argmax")
        self.assertEqual(declared["allocation"], dict(depth=6, sims=4096, batch=16, worlds=4, workers=1))
        self.assertFalse(declared["oracle_deployable"])
        self.assertFalse(row["source_contract"]["phase_a_admission"])
        self.assertFalse(row["source_contract"]["scientific_strength_evidence"])
        self.assertEqual(declared["private_request_protocol"], "private_choice_attempt_redecision_v1")
        self.assertEqual(declared["raw_work_witness"], "pokezero.model-tree-raw-leaf.v2")
        self.assertFalse(declared["historical_attempt_reused"])

    def test_relabelled_fallback_budget_seed_or_policy_refuses_before_claim(self):
        for key, value in (("tree", "rollout_crate"), ("priors", "uniform"),
                ("model_forwards", "skipped"), ("raw_rollout_policy", "uniform"),
                ("raw_rollout_cap", 500), ("capped_rollout", "hp_fallback"),
                ("expired_rollout", "draw"), ("seed", "tree_rng"), ("branch_on_damage", False),
                ("rollout_count", 32), ("rollout_threads", 12), ("oracle_deployable", True),
                ("private_request_protocol", "global_true_trapped"), ("raw_work_witness", "v1"),
                ("hidden_trap_information", "true_hidden_ability"), ("opponent_choice", "resample"),
                ("historical_attempt_reused", True)):
            row = self.registration()
            row["source_contract"]["incumbent_raw_terminal_ablation"][key] = value
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "qualification changed"):
                    DRIVER.claim_attempt(Path(directory), row)
                self.assertFalse((Path(directory) / "attempt.json").exists())

    def test_mode_budget_and_create_only_attempt_are_explicit(self):
        legacy = self.registration()
        legacy["schema"] = DRIVER.CLOSED_INCUMBENT_RAW_SCHEMA
        legacy["source_contract"]["namespace"] = DRIVER.CLOSED_INCUMBENT_RAW_NAMESPACE
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "original raw qualification is closed"):
                DRIVER.claim_attempt(Path(directory), legacy)
            self.assertFalse((Path(directory) / "attempt.json").exists())
        for mode in ("oracle", "reference_leaves", "incumbent_hp"):
            with self.assertRaisesRegex(ValueError, "undeclared qualification"):
                DRIVER.qualification_configurations(10., incumbent_raw=True, **{mode: True})
            with self.assertRaisesRegex(ValueError, "undeclared qualification"):
                DRIVER.engineering_contract(incumbent_raw=True, **{mode: True})
        with self.assertRaisesRegex(ValueError, "ten-second budget"):
            DRIVER.qualification_configurations(3., incumbent_raw=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            DRIVER.claim_attempt(output, self.registration())
            before = (output / "attempt.json").read_bytes()
            with self.assertRaises(FileExistsError):
                DRIVER.claim_attempt(output, self.registration())
            self.assertEqual(before, (output / "attempt.json").read_bytes())

    def test_source_schema_and_arm_resource_drift_cannot_alias_raw_qualification(self):
        for change in ("schema", "configuration"):
            row = self.registration()
            if change == "schema":
                row["schema"] = DRIVER.INCUMBENT_HP_SCHEMA
            else:
                row["configurations"][-1]["leaf"] = "model"
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    DRIVER.claim_attempt(Path(directory), row)
                self.assertFalse((Path(directory) / "attempt.json").exists())


class ComparisonBankTests(unittest.TestCase):
    def fixture(self, slots=1, memory_usage=lambda: 1, available=None):
        roster = [dict(root_id=f"exploration:123:{i}", source_seed=123, root_slot=i) for i in range(slots)]
        contract = dict(namespace="7e105338-51b1-44fd-8e0d-9fd526493b4c", exclude_opening_requests=True,
            panels=dict(exploration=dict(seeds=[123], root_slots=roster)))
        selected = select_source_requests(contract["namespace"], 123,
            list(range(1, 1 + (slots if available is None else available))), slots)
        roots = []
        for slot, index in zip(roster, selected):
            public = dict(seed=123, request=index)
            roots.append(dict(root_id=slot["root_id"], source_request_index=index, public_record=public,
                public_record_sha256=digest(public)))
        catalog = [dict(source_request_index=root["source_request_index"],
            public_record_sha256=root["public_record_sha256"]) for root in roots]
        source = dict(contract_sha256=digest(contract), source_seed=123, panel="exploration", status="COMPLETE",
            source_terminal_complete=True, source_policy="raw_argmax_both_seats", eligible_public_records=catalog,
            eligible_requests=len(catalog), eligible_catalog_sha256=digest(catalog), requested_root_slots=slots,
            roots=roots, missing_root_ids=[slot["root_id"] for slot in roster[len(roots):]])
        archive = SealedSourceArchive()
        snapshot = SimpleNamespace(bridge_snapshot=dict(battle={"private": "DO_NOT_EXPORT"}))
        archive.selected = Mock(return_value=(SimpleNamespace(public=True), None, snapshot))
        bank = SealedComparisonBank(roster, parent_memory_limit_bytes=100, memory_usage=memory_usage)
        self.addCleanup(bank.close)
        bank.account_source(source, contract=contract)
        return bank, archive, roots[0], snapshot

    def test_only_fixed_unique_exploration_slots_and_explicit_memory_limit(self):
        for slots in ([], [dict(root_id="validation:123:0", source_seed=123, root_slot=0)],
                [dict(root_id="exploration:123:0", source_seed=123, root_slot=0)] * 2):
            with self.assertRaisesRegex(ValueError, "fixed unique exploration"):
                SealedComparisonBank(slots, parent_memory_limit_bytes=100)
        for limit in (0, True, 8 * 1024**3 + 1):
            with self.assertRaisesRegex(ValueError, "memory limit"):
                SealedComparisonBank([dict(root_id="exploration:123:0", source_seed=123, root_slot=0)],
                    parent_memory_limit_bytes=limit)
        bank, _, _, _ = self.fixture()
        bank.validate_roster([dict(root_id="exploration:123:0", source_seed=123, root_slot=0)])
        with self.assertRaisesRegex(ValueError, "differs from executable fixed roster"):
            bank.validate_roster([dict(root_id="exploration:123:1", source_seed=123, root_slot=1)])

    def test_public_manifest_does_not_export_private_payload_or_allow_persistence(self):
        bank, archive, root, _ = self.fixture()
        bank.capture(archive, root)
        manifest = bank.seal()
        self.assertNotIn("DO_NOT_EXPORT", json.dumps(manifest))
        self.assertEqual(manifest["root_public_bindings"], {root["root_id"]: root["public_record_sha256"]})
        self.assertFalse(manifest["source_environments_retained"])
        with self.assertRaisesRegex(TypeError, "cannot be persisted"):
            pickle.dumps(bank)

    def test_unknown_duplicate_forged_seed_and_bridge_local_handle_fail(self):
        bank, archive, root, snapshot = self.fixture()
        forged = dict(root, root_id="exploration:456:0")
        with self.assertRaisesRegex(ValueError, "unregistered"):
            bank.capture(archive, forged)
        forged = dict(root, public_record=dict(seed=456))
        with self.assertRaisesRegex(ValueError, "seed drift"):
            bank.capture(archive, forged)
        snapshot.bridge_snapshot["snapshot_id"] = "resident-other-process"
        with self.assertRaisesRegex(ValueError, "bridge-local handle"):
            bank.capture(archive, root)
        del snapshot.bridge_snapshot["snapshot_id"]
        bank.capture(archive, root)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bank.capture(archive, root)

    def test_seal_requires_whole_registered_roster_and_forbids_later_capture(self):
        bank, archive, root, _ = self.fixture(slots=2)
        bank.capture(archive, root)
        with self.assertRaisesRegex(ValueError, "all fixed slots"):
            bank.seal()
        with self.assertRaisesRegex(ValueError, "sealed registered"):
            bank.selected_for_auditor(root["root_id"])
        with self.assertRaisesRegex(ValueError, "validated priority-selected"):
            bank.capture(archive, dict(root, root_id="exploration:123:1"))
        bank.capture(archive, bank._selected["exploration:123:1"])
        bank.seal()
        with self.assertRaisesRegex(ValueError, "sealed root capture"):
            bank.capture(archive, root)
        short, archive, root, _ = self.fixture(slots=2, available=1)
        short.capture(archive, root)
        self.assertEqual(short.seal()["source_validated_missing_root_ids"], ["exploration:123:1"])
        with self.assertRaisesRegex(ValueError, "sealed registered"):
            short.selected_for_auditor("exploration:123:1")

    def test_each_retrieval_is_independent_and_close_makes_bank_unusable(self):
        bank, archive, root, snapshot = self.fixture()
        bank.capture(archive, root)
        snapshot.bridge_snapshot["battle"]["private"] = "original-mutated"
        bank.seal()
        first = bank.selected_for_auditor(root["root_id"])
        first[3].bridge_snapshot["battle"]["private"] = "branch-mutated"
        self.assertEqual(bank.selected_for_auditor(root["root_id"])[3].bridge_snapshot["battle"]["private"],
            "DO_NOT_EXPORT")
        bank.close()
        self.assertEqual(bank._entries, {})
        with self.assertRaisesRegex(ValueError, "closed"):
            bank.selected_for_auditor(root["root_id"])

    def test_memory_stop_clears_entries_without_eviction_or_redraw(self):
        memory = Mock(return_value=1)
        bank, archive, root, _ = self.fixture(memory_usage=memory)
        memory.return_value = 101
        with self.assertRaisesRegex(ValueError, "no eviction or replay"):
            bank.capture(archive, root)
        self.assertEqual(bank._entries, {})
        self.assertEqual(bank._state, "CLOSED")


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
        from pokezero.public_decision_corpus import canonical_json_sha256
        original = root["public_record"]["recorded_action_index"]
        alternative = next(i for i, x in enumerate(root["public_record"]["current_legal_action_mask"])
            if x and i != original)
        root["public_record"]["recorded_action_index"] = alternative
        root["public_record"]["decision_id"] = canonical_json_sha256(
            {k: v for k, v in root["public_record"].items() if k != "decision_id"})
        root["public_record_sha256"] = digest(root["public_record"])
        with self.assertRaisesRegex(ValueError, "differs from sealed source"):
            self.archive.selected(root)

    def test_valid_checksum_cannot_replace_source_history(self):
        from pokezero.public_decision_corpus import canonical_json_sha256
        root = self.root(self.collect())
        root["public_record"]["history"][0]["observation"]["numeric_features"][0][0] += 1
        root["public_record"]["decision_id"] = canonical_json_sha256(
            {k: v for k, v in root["public_record"].items() if k != "decision_id"})
        root["public_record_sha256"] = digest(root["public_record"])
        with self.assertRaisesRegex(ValueError, "history/belief differs"):
            self.archive.selected(root)

    def test_valid_checksum_cannot_replace_source_belief(self):
        from pokezero.public_decision_corpus import canonical_json_sha256
        root = self.root(self.collect())
        root["public_record"]["public_belief_view"]["self_slot"] = "p2"
        root["public_record"]["decision_id"] = canonical_json_sha256(
            {k: v for k, v in root["public_record"].items() if k != "decision_id"})
        root["public_record_sha256"] = digest(root["public_record"])
        with self.assertRaisesRegex(ValueError, "history/belief differs"):
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

    def test_comparison_bank_restores_after_original_environment_and_archive_close(self):
        from pokezero.local_showdown import LocalShowdownEnv
        from pokezero.public_decision_corpus import PublicObservation
        # Existing three-boundary excluded engineering fixture, not a fresh
        # scientific source or a strength measurement.
        root = self.root(self.collect())
        root["root_id"] = f"exploration:{DRIVER.FIXTURE_SEED}:0"
        bank = SealedComparisonBank([dict(root_id=root["root_id"], source_seed=DRIVER.FIXTURE_SEED,
            root_slot=0)], parent_memory_limit_bytes=8 * 1024**3)
        self.addCleanup(bank.close)
        contract = dict(namespace="7e105338-51b1-44fd-8e0d-9fd526493b4c", exclude_opening_requests=True,
            panels=dict(exploration=dict(seeds=[DRIVER.FIXTURE_SEED], root_slots=[dict(
                root_id=root["root_id"], source_seed=DRIVER.FIXTURE_SEED, root_slot=0)])))
        catalog = [dict(source_request_index=root["source_request_index"],
            public_record_sha256=root["public_record_sha256"])]
        bank.account_source(dict(contract_sha256=digest(contract), source_seed=DRIVER.FIXTURE_SEED,
            panel="exploration", status="COMPLETE", source_terminal_complete=True,
            source_policy="raw_argmax_both_seats", eligible_public_records=catalog, eligible_requests=1,
            eligible_catalog_sha256=digest(catalog), requested_root_slots=1, roots=[root], missing_root_ids=[]),
            contract=contract)
        bank.capture(self.archive, root)
        bank.seal()
        expected = bank.selected_for_auditor(root["root_id"])
        self.archive.close()
        self.env.close()
        other = LocalShowdownEnv(self.env.config)
        self.addCleanup(other.close)
        other.reset(seed=DRIVER.FIXTURE_SEED, format_id="gen3randombattle")
        suffixes = []
        for _ in range(2):
            retained, context, pending, snapshot = bank.selected_for_auditor(root["root_id"])
            other.restore(snapshot)
            self.assertEqual(PublicObservation.from_observation(other.observe("p1")).to_dict(),
                PublicObservation.from_observation(context.observation).to_dict())
            self.assertEqual({seat: tuple(rows) for seat, rows in other._request_history.items()},
                snapshot.request_history)
            self.assertEqual(other._parser.snapshot(), snapshot.replay)
            self.assertEqual(retained, expected[0])
            self.assertEqual(pending, expected[2])
            other.reseed_simulator_rng(4711)
            other.step({seat: next(i for i, legal in enumerate(other.observe(seat).legal_action_mask) if legal)
                for seat in other.requested_players()})
            # Showdown inserts wall-clock |t: messages into its protocol log.
            # Only those timestamps are ignored; compare both observations,
            # all other public protocol lines, full battle sides and PRNG state.
            battle = other.snapshot().bridge_snapshot["battle"]
            suffixes.append((tuple(line for line in other._lines if not line.startswith("|t:")),
                {seat: PublicObservation.from_observation(other.observe(seat)).to_dict()
                    for seat in ("p1", "p2")}, battle["sides"], battle.get("prng"), battle["turn"]))
        self.assertEqual(suffixes[0], suffixes[1])
        after = bank.selected_for_auditor(root["root_id"])
        self.assertEqual(after[3].bridge_snapshot, expected[3].bridge_snapshot)
        self.assertEqual(after[3].replay, expected[3].replay)


if __name__ == "__main__":
    unittest.main()
