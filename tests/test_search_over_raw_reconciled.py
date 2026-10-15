"""Temporary synthetic controller ledgers only; no scientific panel allocation."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PACKAGE = "reconciled_ledger_fixture"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "src/pokezero/mcts_eval")]
sys.modules[PACKAGE] = package
m = importlib.import_module(PACKAGE + ".search_over_raw_reconciled")
legacy = importlib.import_module(PACKAGE + ".search_over_raw_ledger")
science = importlib.import_module(PACKAGE + ".search_over_raw")
NAMESPACE = "86fbb43c-dd13-47f5-8c91-82c1758765eb"


class ReconciledLedgerTest(unittest.TestCase):
    def test_reconciled_contract_rejects_numeric_aliases_in_frozen_design(self):
        m.validate_contract(self.contract)
        mutations = (
            ("bootstrap_draws", lambda c:c["validation_inference"].update(bootstrap_draws=10000.0)),
            ("lower_order_index", lambda c:c["validation_inference"].update(lower_order_index=500.0)),
            ("nominal_seconds", lambda c:c["configurations"][0].update(seconds=True)),
            ("worker_count", lambda c:c["a1_configurations"][0].update(workers=1.0)),
            ("replicates", lambda c:c.update(continuation_replicates=8.0)),
            ("phase_b", lambda c:c.update(phase_b_authorized=0)),
            ("history_pooling", lambda c:c.update(historical_outcomes_pooled=0)),
        )
        for name, mutate in mutations:
            changed = deepcopy(self.contract)
            mutate(changed)
            with self.subTest(mutation=name), self.assertRaisesRegex(ValueError, "core drift"):
                m.validate_contract(changed)
            destination = self.root / ("numeric-alias-"+name)
            with self.subTest(ledger= name), self.assertRaisesRegex(ValueError, "core drift"):
                legacy.PhaseALedger.create(destination, changed)
            self.assertFalse(destination.exists())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.panels = {}
        # Explicit fixture IDs, not a source-law draw or scientific registration.
        for name, offset in (("exploration", 0), ("validation", 1000)):
            seeds = list(range(offset, offset+32))
            self.panels[name] = dict(seeds=seeds, root_slots=[dict(root_id=f"fixture:{name}:{i}",
                source_seed=seeds[i % 32], root_slot=i//32) for i in range(200)])
        self.excluded = [9999999]
        self.source_review = dict(status="CLEAR_FRESH_SOURCE_DESIGN", panels_sha256=m.digest(self.panels),
            excluded_seeds_sha256=m.digest(self.excluded), all_known_exposures_included=True,
            nine_roots_and_two_sources_excluded=True)
        self.contract = m.reconciled_contract(NAMESPACE, panels=self.panels, excluded_seeds=self.excluded,
            source_review=self.source_review)
        self.ledger = legacy.PhaseALedger.create(self.root / "ledger", self.contract)

    def tearDown(self):
        self.temp.cleanup()

    def intervals(self, panel, value=.25):
        return {row["root_id"]: [value, value] for row in self.panels[panel]["root_slots"]}

    def record(self, stage, cfg, *, value=.25, status="QUALIFIED"):
        uncertain = status != "QUALIFIED"
        if getattr(self, "_ready_receipts", False) and not uncertain:
            if stage == "A3" and cfg.seconds == 1.:
                prior = json.loads((self.ledger.directory / "A2" / (cfg.identity+".json")).read_text())
                return self.ledger.record_stage(stage, cfg.identity, prior["root_intervals"],
                    accepted_units=prior["summary"]["accepted_units"], status=prior["summary"]["status"],
                    evidence_sha256=prior["evidence_sha256"])
            bundle = self.evidence_fixture(cfg)
            for row in bundle["outcomes"]:
                if row["action"] == 1:
                    row["signed_outcome"] = 1 if row["replicate"] < round(value*8) else -1
            return self.record_evidence(cfg, {bundle["root_binding"]["root_id"]: bundle}, stage=stage)
        return self.ledger.record_stage(stage, cfg.identity,
            {} if uncertain else self.intervals("exploration", value),
            accepted_units=0 if uncertain else 1, status=status, evidence_sha256="a"*64)

    def freeze(self, unavailable=()):
        for cfg in m.a2_configurations():
            self.record("A2", cfg, status="UNQUALIFIED_AT_CEILING" if cfg.arm in unavailable else "QUALIFIED")
        self.ledger.freeze_evaluators()
        for cfg in self.ledger._stage_configs("A3"):
            self.record("A3", cfg)
        return self.ledger.freeze()

    def ready(self):
        # A separate fixture simulates externally admitted runtime state.
        self.evidence_ledger(m.deployment_configurations())
        contract = dict(self.ledger.contract, execution_ready=True, collection_authorized=True, remaining_gates=[])
        self.ledger = legacy.PhaseALedger.create(self.root / "ready", contract)
        self._ready_receipts = True

    def evidence_fixture(self, cfg, root=None, *, raw_signed=-1):
        root = self.panels["exploration"]["root_slots"][0] if root is None else root
        raw = m.SearchConfiguration("raw", seconds=1.)
        binding = dict(root_id=root["root_id"], source_seed=root["source_seed"],
            public_record_sha256="a"*64, information_key="fixture-public-info", legal_action_mask=[True, True])
        selections = {}
        for c in (raw, cfg):
            name = "raw" if c.arm == "raw" else c.identity
            evidence = (dict(selector="deterministic_masked_argmax", model_forwards=1,
                legal_actions=[0, 1], priors=[.75, .25]) if c.arm == "raw" else
                dict(total_iterations=64) if c.arm == "incumbent" else dict(result=dict(trajectories=80)))
            selections[name] = dict(root_id=root["root_id"]+":"+name, statistics_root_id=root["root_id"],
                selection_seed=root["source_seed"], configuration_sha256=c.identity, runtime_sha256="a"*64,
                independent_root_statistics=True, action=0 if c.arm == "raw" else 1,
                status="SELECTED", elapsed_seconds=.5, nominal_search_seconds=c.seconds, evidence=evidence)
        return dict(root_binding=binding, continuation_contract=dict(root_binding=deepcopy(binding),
            root_id=root["root_id"], namespace=NAMESPACE, subject="p1", max_boundaries=250,
            replicates=8, selection_seed=root["source_seed"],
            policy="raw_argmax_after_initial_sampled_opponent_reply"), selections=selections,
            outcomes=[dict(root_id=root["root_id"], action=a, replicate=r, status="COMPLETE",
                signed_outcome=raw_signed if a == 0 else 1, boundaries=2) for a in (0, 1) for r in range(8)])

    def evidence_ledger(self, configs, *, zero_work_reviews=None):
        self.public_bindings = {r["root_id"]: "a"*64 for r in self.panels["exploration"]["root_slots"]}
        self.qualification_paths = {}
        hashes = m.controller_input_hashes()
        for cfg in configs:
            path = (self.root / (cfg.identity+"-qualification.json")).resolve()
            review=dict(status="CLEAR_CONFIGURATION_AT_CEILING",
                configuration_sha256=cfg.identity, runtime_sha256="a"*64,
                public_bindings_sha256=m.digest(self.public_bindings), atomic_work_and_cleanup_validated=True)
            if zero_work_reviews and cfg.identity in zero_work_reviews:
                review.update(zero_work_is_local_unqualified=True,status=zero_work_reviews[cfg.identity])
            path.write_text(json.dumps(review))
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.qualification_paths[cfg.identity] = path
        self.ledger = legacy.PhaseALedger.create(self.root / "evidence-ledger",
            dict(self.contract, input_hashes=hashes))

    def record_evidence(self, cfg, bundles, *, stage="A2", **overrides):
        args = dict(public_bindings=self.public_bindings, runtime_hashes={"raw": "a"*64, cfg.identity: "a"*64},
            qualification_review_path=self.qualification_paths[cfg.identity],
            validate_selector=lambda c, row, binding: m.digest(row),
            validate_outcome=lambda row, contract: m.digest(row) if row["status"] == "COMPLETE" else None)
        args.update(overrides)
        return self.ledger.record_stage_evidence(stage, cfg.identity, bundles, **args)

    def test_reconciled_evidence_stage_derives_bounds_and_keeps_full_roster(self):
        cfg = m.a2_configurations()[0]
        self.evidence_ledger((cfg,))
        bundle = self.evidence_fixture(cfg)
        root = bundle["root_binding"]["root_id"]
        bundle["outcomes"].pop(0)
        artifact = self.record_evidence(cfg, {root: bundle})
        self.assertEqual(artifact["root_intervals"], {root: [.875, 1.]})
        self.assertEqual(artifact["summary"]["accepted_units"], 64)
        self.assertLess(artifact["summary"]["lower_mean"], -.98)
        self.assertFalse(self.ledger.contract["collection_authorized"])
        with self.assertRaises(FileExistsError):
            self.record_evidence(cfg, {root: bundle})

    def zero_work_bundle(self, cfg, root=None):
        bundle=self.evidence_fixture(cfg,root)
        selected=bundle["selections"][cfg.identity]
        selected.update(status="UNQUALIFIED_AT_CEILING",action=None,attempt_consumed=True,
            retry_authorized=False,continuation_authorized=False)
        selected["evidence"]=(dict(total_iterations=0) if cfg.arm == "incumbent"
            else dict(result=dict(trajectories=0)))
        bundle["outcomes"]=[row for row in bundle["outcomes"] if row["action"] == 0]
        return bundle

    def test_owned_zero_work_is_local_unqualified_not_global_stop(self):
        configs=m.a2_configurations();zero,productive=configs[:2]
        self.evidence_ledger(configs,zero_work_reviews={zero.identity:"CLEAR_UNQUALIFIED_AT_CEILING"})
        bundle=self.zero_work_bundle(zero);root=bundle["root_binding"]["root_id"]
        artifact=self.record_evidence(zero,{root:bundle})
        self.assertEqual(artifact["summary"]["status"],"UNQUALIFIED_AT_CEILING")
        self.assertEqual(artifact["summary"]["accepted_units"],0)
        self.assertEqual(artifact["root_intervals"],{})
        evidence=json.loads((self.ledger.directory/"A2"/(zero.identity+"-root-evidence.json")).read_text())
        self.assertEqual(evidence["root_evidence"][root]["root_intervals"][zero.identity],[-1.,1.])
        self.assertNotIn(zero.identity,evidence["root_evidence"][root]["actions"])
        self.assertEqual(evidence["root_evidence"][root]["unique_action_cells"],8)
        normal=self.record_evidence(productive,{root:self.evidence_fixture(productive)})
        self.assertEqual(normal["summary"]["status"],"QUALIFIED")
        self.assertFalse((self.ledger.directory/"operational-stop.json").exists())
        self.assertFalse(self.ledger.contract["execution_ready"])
        for cfg in configs[2:]:self.record_evidence(cfg,{root:self.evidence_fixture(cfg)})
        report=self.ledger.report_stage("A2")
        self.assertEqual(report["configurations"][0]["identification_interval"],[-1.,1.])
        frozen=self.ledger.freeze_evaluators()
        self.assertEqual(dict(frozen["evaluators"])[zero.arm],productive.leaf)
        self.assertFalse(frozen["runtime_authorized"])

    def test_mixed_productive_zero_roots_keep_zero_in_full_denominator(self):
        cfg=m.a2_configurations()[0]
        self.evidence_ledger((cfg,),zero_work_reviews={cfg.identity:"CLEAR_CONFIGURATION_AT_CEILING"})
        zero=self.zero_work_bundle(cfg);normal=self.evidence_fixture(cfg,self.panels["exploration"]["root_slots"][1])
        root=zero["root_binding"]["root_id"];other=normal["root_binding"]["root_id"]
        artifact=self.record_evidence(cfg,{root:zero,other:normal})
        self.assertEqual(artifact["root_intervals"],{root:[-1.,1.],other:[1.,1.]})
        self.assertEqual(artifact["summary"]["accepted_units"],64)
        self.assertLess(artifact["summary"]["lower_mean"],-.98)

    def test_zero_work_requires_prospective_local_review_and_observed_attempt(self):
        cfg=m.a2_configurations()[0];self.evidence_ledger((cfg,))
        bundle=self.zero_work_bundle(cfg)
        with self.assertRaises(ValueError):self.record_evidence(cfg,{bundle["root_binding"]["root_id"]:bundle})
        self.assertTrue((self.ledger.directory/"operational-stop.json").exists())

    def test_empty_panel_cannot_be_called_observed_zero_work(self):
        cfg=m.a2_configurations()[0]
        self.evidence_ledger((cfg,),zero_work_reviews={cfg.identity:"CLEAR_UNQUALIFIED_AT_CEILING"})
        with self.assertRaisesRegex(ValueError,"observed owned attempts"):
            self.record_evidence(cfg,{})

    def test_unqualified_review_cannot_promote_productive_work(self):
        cfg=m.a2_configurations()[0]
        self.evidence_ledger((cfg,),zero_work_reviews={cfg.identity:"CLEAR_UNQUALIFIED_AT_CEILING"})
        bundle=self.evidence_fixture(cfg)
        with self.assertRaisesRegex(ValueError,"configuration qualification review"):
            self.record_evidence(cfg,{bundle["root_binding"]["root_id"]:bundle})

    def test_different_configuration_cannot_redraw_shared_raw_control(self):
        configs = m.a2_configurations()[:2]
        self.evidence_ledger(configs)
        for i, cfg in enumerate(configs):
            bundle = self.evidence_fixture(cfg, raw_signed=-1 if i == 0 else 1)
            root = bundle["root_binding"]["root_id"]
            if i == 0:
                self.record_evidence(cfg, {root: bundle})
            else:
                with self.assertRaisesRegex(ValueError, "shared action/control"):
                    self.record_evidence(cfg, {root: bundle})
        self.assertTrue((self.ledger.directory / "operational-stop.json").exists())

    def test_shared_raw_exact_reuse_across_configuration_consumers(self):
        configs = m.a2_configurations()[:2]
        self.evidence_ledger(configs)
        for cfg in configs:
            bundle = self.evidence_fixture(cfg)
            self.record_evidence(cfg, {bundle["root_binding"]["root_id"]: bundle})
        self.assertEqual(len(list((self.ledger.directory / "shared-actions").iterdir())), 1)

    def test_same_nonraw_action_cannot_receive_different_outcome_or_validation(self):
        configs = (m.a2_configurations()[0], m.a2_configurations()[2])
        self.evidence_ledger(configs)
        first = self.evidence_fixture(configs[0])
        root = first["root_binding"]["root_id"]
        self.record_evidence(configs[0], {root: first})
        second = self.evidence_fixture(configs[1])
        second["outcomes"][-1]["signed_outcome"] = -1
        with self.assertRaisesRegex(ValueError, "shared action/control"):
            self.record_evidence(configs[1], {root: second})
        self.assertTrue((self.ledger.directory / "operational-stop.json").exists())

    def test_missing_shared_candidate_cell_cannot_be_silently_completed_later(self):
        configs = m.a2_configurations()[:2]
        self.evidence_ledger(configs)
        first = self.evidence_fixture(configs[0])
        first["outcomes"].pop()
        root = first["root_binding"]["root_id"]
        self.record_evidence(configs[0], {root: first})
        second = self.evidence_fixture(configs[1])
        with self.assertRaisesRegex(ValueError, "shared action/control"):
            self.record_evidence(configs[1], {root: second})

    def test_execution_ready_manual_summary_cannot_bypass_reconciled_receipts(self):
        self.ready()
        cfg = m.a2_configurations()[0]
        with self.assertRaisesRegex(ValueError, "reconciled receipts"):
            self.ledger.record_stage("A2", cfg.identity, self.intervals("exploration"),
                accepted_units=1, status="QUALIFIED", evidence_sha256="a"*64)

    def test_unbound_configuration_review_prevents_claim_and_validators(self):
        cfg = m.a2_configurations()[0]
        self.evidence_ledger((cfg,))
        path = self.root / "unbound.json"
        path.write_text(self.qualification_paths[cfg.identity].read_text())
        bundle = self.evidence_fixture(cfg)
        with patch.object(m, "digest", wraps=m.digest), self.assertRaises(ValueError):
            self.record_evidence(cfg, {bundle["root_binding"]["root_id"]: bundle}, qualification_review_path=path)
        self.assertFalse((self.ledger.directory / "A2" / (cfg.identity+"-evidence-claim.json")).exists())

    def test_validator_failure_poisoned_and_manual_summary_cannot_bypass_orphan(self):
        cfg = m.a2_configurations()[0]
        self.evidence_ledger((cfg,))
        bundle = self.evidence_fixture(cfg)
        def fail(*args):
            raise ValueError("invalid actual backup")
        with self.assertRaises(ValueError):
            self.record_evidence(cfg, {bundle["root_binding"]["root_id"]: bundle}, validate_selector=fail)
        self.assertTrue((self.ledger.directory / "operational-stop.json").exists())
        with self.assertRaises(ValueError):
            self.ledger.record_stage("A2", cfg.identity, {}, accepted_units=0,
                status="UNSTARTED", evidence_sha256="a"*64)

    def test_manual_intervals_cannot_bypass_orphan_reconciliation_claim(self):
        cfg = m.a2_configurations()[0]
        path = self.ledger.directory / "A2" / (cfg.identity+"-evidence-claim.json")
        path.write_text("{}")  # Synthetic process-loss fixture; no terminal stop receipt.
        with self.assertRaisesRegex(ValueError, "bypassed"):
            self.ledger.record_stage("A2", cfg.identity, {}, accepted_units=0,
                status="UNSTARTED", evidence_sha256="a"*64)

    def test_collector_input_mutation_during_validation_fails_claimed_attempt(self):
        cfg = m.a2_configurations()[0]
        self.evidence_ledger((cfg,))
        bundle = self.evidence_fixture(cfg)
        def mutate(c, row, binding):
            self.public_bindings[self.panels["exploration"]["root_slots"][1]["root_id"]] = "b"*64
            return m.digest(row)
        with self.assertRaisesRegex(ValueError, "input evidence drift"):
            self.record_evidence(cfg, {bundle["root_binding"]["root_id"]: bundle}, validate_selector=mutate)
        self.assertTrue((self.ledger.directory / "operational-stop.json").exists())

    def review(self):
        calibration = self.root / "synthetic-calibration.json"
        calibration.write_text(json.dumps(dict(schema="pokezero.staged-bootstrap-synthetic-calibration.v1.results",
            calibration_complete=True, input_sha256={str(m.Path(m.__file__).with_name("search_over_raw_screening.py")):
                hashlib.sha256(m.Path(m.__file__).with_name("search_over_raw_screening.py").read_bytes()).hexdigest()})))
        runtime = self.root / "synthetic-runtime.json"
        runtime.write_text("{}")
        selection = self.ledger._selection()
        selected = [m.SearchConfiguration(arm, "public", leaf, seconds, 20 if arm == "reference" else 1).identity
                    for arm, leaf, seconds in selection["choices"] if leaf is not None]
        review = dict(status="CLEAR_ONE_HOLDOUT_SCREENING_ATTEMPT",
            contract_sha256=self.ledger.contract_sha256, selection_sha256=m.digest(selection),
            analysis_code_sha256=hashlib.sha256(m.Path(m.__file__).with_name("search_over_raw_screening.py").read_bytes()).hexdigest(),
            user_authorized_holdout=True, screening_risk_reviewed=True,
            qualified_configurations=selected,
            calibration_result_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
            runtime_qualification_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
            evidence_files=dict(calibration_result=str(calibration), runtime_qualification=str(runtime)))
        path = self.root / "synthetic-review.json"
        path.write_text(json.dumps(review))
        return path

    def test_existing_controller_dispatches_v3_without_changing_legacy(self):
        self.assertIsInstance(self.ledger, m.ReconciledPhaseALedger)
        self.assertIsInstance(legacy.PhaseALedger(self.root / "ledger"), m.ReconciledPhaseALedger)
        configs = [science.SearchConfiguration("raw"), science.SearchConfiguration("incumbent"),
                   science.SearchConfiguration("reference", workers=20)]
        old = science.phase_a_contract(NAMESPACE, excluded_seeds=[], configurations=configs)
        old_ledger = legacy.PhaseALedger.create(self.root / "legacy", old)
        self.assertIs(type(old_ledger), legacy.PhaseALedger)
        self.assertEqual(old_ledger.contract["validation_inference"]["stake"], .99)

    def test_source_review_overlap_exposure_and_wrong_quotas_rejected(self):
        for mode in ("review", "overlap", "exposed", "quota"):
            panels, excluded, review = deepcopy(self.panels), list(self.excluded), dict(self.source_review)
            if mode == "review":
                review["nine_roots_and_two_sources_excluded"] = False
            elif mode == "overlap":
                panels["validation"] = deepcopy(panels["exploration"])
            elif mode == "exposed":
                excluded.append(0)
                excluded.sort()
            else:
                panels["exploration"]["root_slots"].pop()
            review.update(panels_sha256=m.digest(panels), excluded_seeds_sha256=m.digest(excluded))
            with self.assertRaises(ValueError):
                m.reconciled_contract(NAMESPACE, panels=panels, excluded_seeds=excluded, source_review=review)

    def test_a1_independent_unqualified_oracle_cannot_block_a2(self):
        for cfg in m.a1_configurations():
            self.record("A1", cfg, status="UNQUALIFIED_AT_CEILING")
        selected = self.freeze()
        self.assertEqual(selected["choices"], (("incumbent", "model", 1.), ("reference", "model", 1.)))
        self.assertTrue(self.ledger.report_stage("A3")["other_stages_required"])

    def test_a1_independent_report_is_a1_and_retains_missing_full_roster(self):
        for cfg in m.a1_configurations():
            self.record("A1", cfg, status="UNQUALIFIED_AT_CEILING")
        report = self.ledger.report_stage("A1")
        self.assertEqual(report["stage"], "A1")
        self.assertFalse(report["other_stages_required"])
        for row in report["configurations"]:
            self.assertEqual((row["stage"], row["lane"]), ("A1", "wall_clock_diagnostic"))
            self.assertEqual(row["identification_interval"], [-1., 1.])
            self.assertEqual((row["root_slots"], row["source_seeds"], row["uncertain_roots"]), (200, 32, 200))
        self.assertFalse((self.ledger.directory / "a2-freeze.json").exists())

    def test_a2_independent_report_has_full_quota_without_a1(self):
        for cfg in m.a2_configurations():
            self.record("A2", cfg)
        report = self.ledger.report_stage("A2")
        self.assertTrue(all(row["stage"] == "A2" for row in report["configurations"]))
        self.assertTrue(all(row["identification_interval"] == [.25, .25] for row in report["configurations"]))
        self.assertEqual(list((self.ledger.directory / "A1").iterdir()), [])

    def test_shared_candidate_validation_cannot_change_between_consumers(self):
        configs = m.a2_configurations()[:2]
        self.evidence_ledger(configs)
        first = self.evidence_fixture(configs[0])
        root = first["root_binding"]["root_id"]
        self.record_evidence(configs[0], {root: first}, validate_outcome=lambda row, contract:
            m.digest(row) if row["action"] == 0 else None)
        second = self.evidence_fixture(configs[1])
        with self.assertRaisesRegex(ValueError, "shared action/control"):
            self.record_evidence(configs[1], {root: second})

    def test_zero_work_local_and_one_arm_unavailable(self):
        selected = self.freeze(unavailable=("incumbent",))
        self.assertEqual(selected["choices"][0], ("incumbent", None, None))
        self.assertEqual(selected["choices"][1], ("reference", "model", 1.))

    def test_a2_freeze_closes_reselection_and_changed_evidence(self):
        for cfg in m.a2_configurations():
            self.record("A2", cfg)
        self.ledger.freeze_evaluators()
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.record("A2", m.a2_configurations()[0])
        with self.assertRaises(FileExistsError):
            self.ledger.freeze_evaluators()
        cfg = m.a2_configurations()[0]
        path = self.ledger.directory / "A2" / f"{cfg.identity}.json"
        row = json.loads(path.read_text())
        row["evidence_sha256"] = "b"*64
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "A2 receipt drift"):
            self.ledger._stage_configs("A3")

    def test_a3_cannot_change_leaf_or_rerun_one_second(self):
        for cfg in m.a2_configurations():
            self.record("A2", cfg)
        self.ledger.freeze_evaluators()
        wrong = m.SearchConfiguration("incumbent", "public", "hp_fraction", 3.)
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.record("A3", wrong)
        with self.assertRaisesRegex(ValueError, "reuse"):
            self.record("A3", self.ledger._stage_configs("A3")[0], value=.5)

    def test_no_terminal_or_oracle_promotion_roster(self):
        self.assertEqual(len(m.deployment_configurations()), 12)
        self.assertTrue(all(c.belief == "public" and c.leaf != "raw_rollout" for c in m.deployment_configurations()))
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.record("A2", m.SearchConfiguration("incumbent", "oracle", "model", 1.))
        self.assertIsNone(self.contract["fixed_work_diagnostic"]["search_decision_ceiling"])

    def test_zero_work_cannot_score_action_or_claim_qualified(self):
        cfg = m.a2_configurations()[0]
        for status, roots in (("QUALIFIED", {}), ("UNQUALIFIED_AT_CEILING", self.intervals("exploration"))):
            with self.assertRaises(ValueError):
                self.ledger.record_stage("A2", cfg.identity, roots, accepted_units=0,
                    status=status, evidence_sha256="a"*64)

    def test_default_authority_cannot_call_holdout(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "admission"):
            self.ledger.validation_once(lambda *_: self.fail("must not call"), review_path=self.root / "absent")
        self.assertFalse((self.ledger.directory / "validation-claim.json").exists())

    def test_a1_a2_exploration_does_not_depend_on_downstream_holdout_gate(self):
        contract = dict(self.contract, input_hashes=m.controller_input_hashes(),
            execution_ready=True, collection_authorized=True, remaining_gates=[m.HOLDOUT_ONLY_GATE])
        ledger = legacy.PhaseALedger.create(self.root / "exploration-admitted", contract)
        for stage, cfg in (("A1", m.a1_configurations()[1]), ("A2", m.a2_configurations()[0])):
            result = ledger.verify_exploration(stage, cfg.identity)
            self.assertTrue(result["exploration_admission_validated"])
            self.assertFalse(result["holdout_authorized"])
        with self.assertRaisesRegex(ValueError, "execution admission"):
            ledger.verify(execution=True)
        with self.assertRaises((ValueError, FileNotFoundError)):
            ledger.verify_exploration("A3", m.deployment_configurations()[1].identity)
        self.assertFalse((ledger.directory / "validation-claim.json").exists())

    def test_exploration_does_not_waive_unknown_source_runtime_or_authority_gate(self):
        cfg = m.a2_configurations()[0]
        for i, changes in enumerate((dict(), dict(execution_ready=False), dict(collection_authorized=False),
                dict(remaining_gates=["source and per-configuration runtime qualification"]),
                dict(remaining_gates=["unknown upstream constraint"]))):
            contract = dict(self.contract, input_hashes=m.controller_input_hashes(),
                execution_ready=True, collection_authorized=True, remaining_gates=[m.HOLDOUT_ONLY_GATE])
            contract.update(changes)
            ledger = legacy.PhaseALedger.create(self.root / f"exploration-gate-{i}", contract)
            if i == 0:
                ledger.verify_exploration("A2", cfg.identity)
            else:
                with self.assertRaisesRegex(ValueError, "exploration.*incomplete"):
                    ledger.verify_exploration("A2", cfg.identity)

    def test_qualified_exploration_receipt_does_not_clear_holdout(self):
        cfg = m.a2_configurations()[0]
        self.evidence_ledger((cfg,))
        contract = dict(self.ledger.contract, execution_ready=True, collection_authorized=True,
            remaining_gates=[m.HOLDOUT_ONLY_GATE])
        self.ledger = legacy.PhaseALedger.create(self.root / "stage-only-ready", contract)
        bundle = self.evidence_fixture(cfg); root = bundle["root_binding"]["root_id"]
        result = self.record_evidence(cfg, {root: bundle})
        self.assertEqual(result["summary"]["status"], "QUALIFIED")
        with self.assertRaisesRegex(ValueError, "execution admission"):
            self.ledger.verify(execution=True)
        self.assertFalse((self.ledger.directory / "operational-stop.json").exists())

    def test_ready_booleans_do_not_replace_controller_provenance(self):
        contract = dict(self.contract, execution_ready=True, collection_authorized=True, remaining_gates=[])
        self.ledger = legacy.PhaseALedger.create(self.root / "unbound-ready", contract)
        with self.assertRaisesRegex(ValueError, "controller code"):
            self.freeze()

    def test_one_shot_paired_bootstrap_other_arm_unavailable(self):
        self.ready()
        self.freeze(unavailable=("incumbent",))
        review = self.review()
        result = self.ledger.validation_once(lambda *_: dict(incumbent=None,
            reference=self.intervals("validation", .1)), review_path=review)
        gate = result["screening"]
        self.assertEqual(gate["arms"]["incumbent"]["status"], "UNAVAILABLE")
        self.assertTrue(gate["arms"]["reference"]["screen_positive"])
        self.assertFalse(gate["both_arms_screen_positive"])
        self.assertFalse(result["phase_b_authorized"])
        with self.assertRaises(FileExistsError):
            legacy.PhaseALedger(self.ledger.directory).validation_once(lambda *_: self.fail("retry"), review_path=review)

    def test_missing_full_denominator_remains_uncertain(self):
        self.ready()
        self.freeze()
        result = self.ledger.validation_once(lambda *_: dict(incumbent={}, reference={}), review_path=self.review())
        self.assertEqual(result["screening"]["arms"]["reference"]["identification_interval"], [-1., 1.])
        self.assertFalse(result["screening"]["any_arm_screen_positive"])

    def test_failure_or_orphan_attempt_never_reopens(self):
        self.ready()
        self.freeze()
        review = self.review()
        def fail(*_):
            raise RuntimeError("PRIVATE_DETAIL")
        with self.assertRaises(RuntimeError):
            self.ledger.validation_once(fail, review_path=review)
        failure = (self.ledger.directory / "validation-failure.json").read_text()
        self.assertNotIn("PRIVATE_DETAIL", failure)
        with self.assertRaises(FileExistsError):
            self.ledger.validation_once(fail, review_path=review)

    def test_operational_stop_is_global_not_unqualification(self):
        self.ledger.stop("UnexpectedWorkerLoss")
        with self.assertRaisesRegex(ValueError, "integrity failure"):
            self.record("A2", m.a2_configurations()[0])
        with self.assertRaisesRegex(ValueError, "integrity failure"):
            self.ledger.freeze_evaluators()

    def test_clearance_must_bind_current_choice_and_acknowledge_risk(self):
        self.ready()
        self.freeze()
        review = self.review()
        original = json.loads(review.read_text())
        for key, value in (("screening_risk_reviewed", False), ("user_authorized_holdout", False),
                           ("selection_sha256", "b"*64), ("qualified_configurations", [])):
            changed = dict(original, **{key: value})
            review.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.ledger.validation_once(lambda *_: self.fail("must not call"), review_path=review)
            self.assertFalse((self.ledger.directory / "validation-claim.json").exists())

    def test_clearance_evidence_drift_blocks_callback(self):
        self.ready()
        self.freeze()
        review = self.review()
        (self.root / "synthetic-runtime.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence drift"):
            self.ledger.validation_once(lambda *_: self.fail("must not call"), review_path=review)

    def assert_screening_drift_fails_closed(self, during_screen):
        self.ready()
        self.freeze()
        review = self.review()
        kernel = Path(m.__file__).with_name("search_over_raw_screening.py").resolve()
        original_read, original_screen = Path.read_bytes, m.screen
        drifted = False
        def read_bytes(path):
            value = original_read(path)
            return value + b"\n" if drifted and path.resolve() == kernel else value
        def collect(*_):
            nonlocal drifted
            drifted = not during_screen
            return dict(incumbent=self.intervals("validation"), reference=self.intervals("validation"))
        def drifting_screen(*args, **kwargs):
            nonlocal drifted
            value = original_screen(*args, **kwargs)
            drifted = True
            return value
        # Simulate observed code bytes changing; never modify the real source.
        with patch.object(Path, "read_bytes", read_bytes), patch.object(m, "screen", drifting_screen):
            with self.assertRaisesRegex(ValueError, "drift"):
                self.ledger.validation_once(collect, review_path=review)
        self.assertTrue((self.ledger.directory / "validation-claim.json").exists())
        self.assertTrue((self.ledger.directory / "validation-failure.json").exists())
        self.assertFalse((self.ledger.directory / "validation-result.json").exists())
        with self.assertRaises(FileExistsError):
            self.ledger.validation_once(lambda *_: self.fail("retry"), review_path=review)

    def test_screening_code_drift_during_callback_fails_claimed_attempt(self):
        self.assert_screening_drift_fails_closed(during_screen=False)

    def test_screening_code_drift_during_bootstrap_fails_claimed_attempt(self):
        self.assert_screening_drift_fails_closed(during_screen=True)

    def test_neither_arm_available_keeps_holdout_unopened(self):
        self.ready()
        self.freeze(unavailable=m.ARMS)
        with self.assertRaisesRegex(ValueError, "neither arm"):
            self.ledger.validation_once(lambda *_: self.fail("must not call"), review_path=self.review())

    def test_unavailable_arm_cannot_adopt_fake_outcomes(self):
        self.ready()
        self.freeze(unavailable=("incumbent",))
        with self.assertRaisesRegex(ValueError, "unavailable arm"):
            self.ledger.validation_once(lambda *_: dict(incumbent={}, reference={}), review_path=self.review())

    def test_contract_core_cannot_change_to_old_betting(self):
        changed = deepcopy(self.contract)
        changed["validation_inference"]["method"] = "fixed_panel_betting"
        with self.assertRaisesRegex(ValueError, "core drift"):
            legacy.PhaseALedger.create(self.root / "wrong", changed)
        self.assertFalse((self.root / "wrong").exists())


if __name__ == "__main__":
    unittest.main()
