"""One trusted authored auditor, actual nested owners and staged freeze.

No scientific source/model/engine, allocation or holdout. All fixture recipes
precede the in-process archive; only a fixed redacted summary crosses out.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_ledger import _write_new
from pokezero.mcts_eval.search_over_raw_reconciled import configuration_review_template
from pokezero.mcts_eval.search_over_raw_root_plan import ArchivedRootEvaluationPlan
from pokezero.mcts_eval.search_over_raw_continuation_admission import ContinuationAdmissionJournal
from pokezero.mcts_eval.search_over_raw_continuation_producer import produce_reserved_continuation
from pokezero.mcts_eval.search_over_raw_supervision import supervise_fixed_work
import test_search_over_raw_selector_producer as selector_fixture
import test_search_over_raw_continuation_admission as admission_fixture
import test_search_over_raw_outcome_run as outcome_fixture


@dataclass(frozen=True)
class AuthoredStagedAuditor:
    from_driver: bool = False
    late_reader_mutation: bool = False
    synthetic_process_fixture = True

    def __call__(self, ownership, phase):
        phase("startup")
        directory = Path(ownership["directory"]).parent
        fixtures = directory / "authored-fixtures"
        fixtures.mkdir()
        # Nested launch evidence must survive worker exit for the ancestor's
        # independent closure read. The enclosing test owns final disposal.
        class RetainedFixtureDirectory:
            def __init__(self, *args, **kwargs):
                self.name = tempfile.mkdtemp(dir=fixtures)
            def cleanup(self): pass
        phase("selection")
        with patch.object(tempfile, "TemporaryDirectory", RetainedFixtureDirectory):
            helper = selector_fixture.SelectorProducerTests(); helper.setUp()
            hp = SearchConfiguration("incumbent", leaf="hp_fraction", seconds=1.)
            extra_reviews = {}
            source_inputs={}
            original_initialize = helper.h.initialize
            def prospectively_review_then_initialize():
                hashes = admission_fixture.controller_input_hashes()
                for cfg in (helper.cfg, hp):
                    path = helper.h.directory.parent / (cfg.identity + "-a2-review.json")
                    review = configuration_review_template(helper.h.contract, stage="A2", configuration=cfg.identity,
                        raw_runtime_sha256=helper.h.raw_sha, selector_recipe_sha256=digest(helper.c.recipe(cfg)),
                        qualification_evidence_hashes={str(helper.c.evidence):hashlib.sha256(helper.c.evidence.read_bytes()).hexdigest()},
                        cost_scope="Authored nested-owner engineering fixture only; no scientific or complete cost qualification.")
                    review.update(status="CLEAR_CONFIGURATION_RECIPE", reviewed_before_first_source_work=True,
                        atomic_work_and_cleanup_validated=True, resource_cost_qualification_reviewed=True)
                    _write_new(path, review)
                    extra_reviews[cfg.identity] = path
                    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                if self.from_driver:
                    from pokezero.local_showdown import LocalShowdownConfig
                    from pokezero.mcts_eval.resolver import CheckpointContract
                    from pokezero.mcts_eval.search_over_raw_source_run import source_recipe,source_review_template
                    from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
                    showdown=fixtures/"source-showdown";sim=showdown/"dist/sim";sim.mkdir(parents=True)
                    (sim/"index.js").write_text(json.dumps(dict(rounds=2,mode="complete")))
                    (sim/"state.js").write_text("authored-not-engine")
                    bridge=fixtures/"source-bridge";bridge.mkdir()
                    scripts=Path(__file__).resolve().parents[1]/"scripts"
                    (bridge/"battle_bridge.mjs").write_bytes(Path(__file__).with_name("source_bridge_fixture.mjs").read_bytes())
                    for name in ("battle_bridge_owned_snapshot.mjs","battle_bridge_owned_continuation.mjs"):
                        (bridge/name).write_bytes((scripts/name).read_bytes())
                    config=LocalShowdownConfig(showdown_root=str(showdown),bridge_path=str(bridge/"battle_bridge.mjs"),
                        set_belief_source=False,read_timeout_seconds=.5)
                    checkpoint=CheckpointContract(checkpoint_path=str(helper.o.checkpoint),
                        checkpoint_sha256=helper.o.files[str(helper.o.checkpoint)],policy_id="authored",schema_version="authored",
                        token_count=1,categorical_feature_count=1,numeric_feature_count=1,transition_token_count=0,
                        category_vocab=(),architecture={"window_size":1},feature_masks={},model_device="cpu",
                        showdown_root=str(showdown),showdown_source_sha256="a"*64)
                    runtime=outcome_runtime_binding(config,checkpoint_path=helper.o.checkpoint)
                    path=fixtures/"source-review.json"
                    review=source_review_template(helper.h.contract,source_recipe(config,checkpoint,runtime),
                        qualification_evidence_hashes={str(helper.c.evidence):hashlib.sha256(helper.c.evidence.read_bytes()).hexdigest()})
                    review.update(status="CLEAR_SOURCE_RECIPE",reviewed_before_first_source_work=True,
                        source_runtime_and_cleanup_reviewed=True,source_policy_and_terminal_reviewed=True)
                    _write_new(path,review);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                    source_inputs.update(config=config,checkpoint_contract=checkpoint,bridge_runtime=runtime,
                        source_review_path=path)
                with patch.object(admission_fixture, "controller_input_hashes", return_value=hashes):
                    original_initialize()
            try:
                with patch.object(helper.h, "initialize", side_effect=prospectively_review_then_initialize):
                    helper.initialize(stage="A1", continuation=True)
                ledger = helper.h.ledger
                source_owner=None
                if self.from_driver:
                    from pokezero.mcts_eval.search_over_raw_source_run import collect_owned_exploration_source
                    from pokezero.mcts_eval.search_over_raw_archive import archived_exploration_root_binding
                    from test_search_over_raw_source_run import AuthoredSourceEnv,AuthoredPolicy
                    # Actual driver/bridge/capture ordering, but an explicitly
                    # authored continuation snapshot. No simulator truth claim.
                    class SourceEnvWithContinuationSnapshot(AuthoredSourceEnv):
                        def snapshot_actionable_boundary(self):return deepcopy(helper.h.outcome.snapshot)
                    class SourceFactory:
                        synthetic_process_fixture=True
                        def __new__(cls,config,checkpoint):
                            env=SourceEnvWithContinuationSnapshot(config)
                            return env,{seat:AuthoredPolicy(checkpoint,env.mode) for seat in ("p1","p2")}
                    helper.a.archive.close()
                    source_owner=collect_owned_exploration_source(ledger,stage="A1",configuration=helper.cfg.identity,
                        source_seed=ledger.contract["panels"]["exploration"]["seeds"][0],
                        history_review_path=helper.h.review_path,_synthetic_factory=SourceFactory,**source_inputs)
                    helper.source_owner=source_owner
                    helper.a.archive=source_owner.archive;helper.a.source=source_owner.source
                    helper.a.root_id=source_owner.source["roots"][0]["root_id"]
                    helper.a.context,helper.a.pending,_=helper.a.archive.selected(source_owner.source["roots"][0])
                    helper.f.bound=archived_exploration_root_binding(helper.a.archive,helper.a.source,
                        contract=ledger.contract,root_id=helper.a.root_id)
                    helper.f.mask=helper.f.bound["public_evidence_binding"]["legal_action_mask"]
                raw, _ = helper.produce(raw=True)
                model, _ = helper.produce(zero=True)
                def plan(*owners):
                    configs = (helper.f.cfg,) + tuple(o.job.configuration for o in owners)
                    return ArchivedRootEvaluationPlan(helper.a.archive, helper.a.source, contract=helper.h.contract,
                        root_id=helper.a.root_id, configurations=configs,
                        selections={"raw":raw.selection, **{o.job.configuration.identity:o.selection for o in owners}},
                        selector_readers={"raw":raw.reader, **{o.job.configuration.identity:o.reader for o in owners}},
                        allow_synthetic=True,source_owner=source_owner)
                first = plan(model)
                if self.late_reader_mutation:
                    import pokezero.mcts_eval.search_over_raw_root_plan as root_plan
                    extract=root_plan.archived_exploration_root_binding
                    check_files=raw.reader._files
                    captures=0
                    def capture(*args,**kwargs):
                        nonlocal captures
                        value=extract(*args,**kwargs);captures+=1
                        return value
                    def files():
                        check_files()
                        if captures>=2:
                            # Only after final_bound/source validation, mutate
                            # a DIFFERENT retained root, not this plan's root.
                            index=source_owner.source["roots"][-1]["source_request_index"]
                            snapshot=source_owner.archive._private[index]
                            source_owner.archive._private[index]=type(snapshot)(**{
                                **snapshot.__dict__,"battle_id":"late-mutated-source"})
                    with patch.object(root_plan,"archived_exploration_root_binding",side_effect=capture),\
                            patch.object(raw.reader,"_files",side_effect=files):first.revalidate()
                    raise AssertionError("late source mutation was accepted")
                a1 = ledger.record_owned_stage_evidence("A1", helper.cfg.identity, plans={first.root_id:first},
                    continuations={first.root_id:None}, qualification_review_path=helper.c.review_path)
                assert a1["summary"]["status"] == "UNQUALIFIED_AT_CEILING"
                assert not (ledger.directory / "operational-stop.json").exists()
                # A1 cannot gate A2. Same public model owner is reused, not redrawn.
                helper.stage = helper.c.stage = "A2"
                helper.c.review_path = extra_reviews[helper.cfg.identity]
                prior, reused = helper.produce(prior=model)
                assert prior is model and reused["selector_launched"] is False
                a2zero = ledger.record_owned_stage_evidence("A2", helper.cfg.identity, plans={first.root_id:first},
                    continuations={first.root_id:None}, qualification_review_path=helper.c.review_path)
                helper.cfg = helper.c.cfg = hp
                helper.search_runtime = helper.c.recipe(hp)
                helper.c.review_path = extra_reviews[hp.identity]
                search, _ = helper.produce()
                combined = plan(search)  # Producer owns one config plus shared raw, not a factorial plan.
                journal = ContinuationAdmissionJournal.create(ledger, combined, history_review_path=helper.h.review_path)
                bank, continuation = produce_reserved_continuation(ledger, "A2", hp.identity, plan=combined,
                    journal=journal, config=helper.h.outcome.config, runtime=helper.h.outcome.runtime,
                    _synthetic_job_class=outcome_fixture.SyntheticContinuationJob)
                a2 = ledger.record_owned_stage_evidence("A2", hp.identity, plans={combined.root_id:combined},
                    continuations={combined.root_id:(bank,)}, qualification_review_path=helper.c.review_path)
                for cfg in ledger._stage_configs("A2"):
                    if cfg.identity not in {hp.identity, model.job.configuration.identity}:
                        ledger.record_stage("A2", cfg.identity, {}, accepted_units=0,
                            status="UNSTARTED", evidence_sha256=digest(dict(authored_unstarted=cfg.identity)))
                frozen = ledger.freeze_evaluators()
                assert dict(frozen["evaluators"])["incumbent"] == "hp_fraction"
                assert dict(frozen["evaluators"])["reference"] is None
                with patch("subprocess.Popen") as launch:
                    a3 = ledger.record_stage("A3", hp.identity, a2["root_intervals"],
                        accepted_units=a2["summary"]["accepted_units"], status=a2["summary"]["status"],
                        evidence_sha256=a2["evidence_sha256"])
                launch.assert_not_called()
                assert a3["evidence_sha256"] == a2["evidence_sha256"]
                result = dict(synthetic=True, a1_status=a1["summary"]["status"], a2_zero_status=a2zero["summary"]["status"],
                    a2_productive_status=a2["summary"]["status"], frozen_evaluator="hp_fraction",
                    a3_one_second_reused=True, source_seed_count=32, root_slot_count=200,
                    measured_authored_roots=1, reserved_cells=continuation["reserved_cells"],
                    continuation_owner_validated=bank.reader.last_validation["process_receipts_validated"],
                    private_archive_returned=False, collection_authorized=False, holdout_authorized=False,
                    runtime_qualified=False, scientific_measurements=0)
                if source_owner is not None:
                    source_owner.validate()
                    result.update(owned_source_driver_used=True,source_roots=len(source_owner.source["roots"]),
                        source_missing_slots=len(source_owner.source["missing_root_ids"]))
            except BaseException as failure:
                import traceback
                _write_new(directory / "authored-fixture-error.json", dict(error=type(failure).__name__,
                    message=str(failure),traceback=traceback.format_exc()))  # Authored inputs only.
                raise
            finally:
                phase("cleanup")
                helper.doCleanups()  # Close private archives; retain only disposable evidence paths.
        return result


class StagedAuditorTests(unittest.TestCase):
    def test_final_reader_callback_cannot_mutate_another_owned_source_root(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root).resolve()/"auditor"
            result=supervise_fixed_work(AuthoredStagedAuditor(from_driver=True,late_reader_mutation=True),
                directory=directory,binding_sha256="f"*64,verify_admission=lambda:True)
            self.assertEqual(result["supervisor_exit_code"],1,result)
            failure=json.loads((directory/"authored-fixture-error.json").read_bytes())
            self.assertEqual(failure["error"],"ValueError",failure)
            self.assertIn("archive or owners changed during handoff",failure["message"])

    def test_actual_authored_source_driver_to_owned_selectors_continuations_and_staged_freeze(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root).resolve()/"auditor"
            result=supervise_fixed_work(AuthoredStagedAuditor(from_driver=True),directory=directory,
                binding_sha256="f"*64,verify_admission=lambda:True)
            detail=directory/"authored-fixture-error.json"
            self.assertEqual(result["supervisor_exit_code"],0,json.loads(detail.read_bytes()) if detail.exists() else result)
            summary=result["result"]
            self.assertTrue(summary["owned_source_driver_used"])
            self.assertEqual((summary["source_roots"],summary["source_missing_slots"]),(2,5))
            self.assertEqual(summary["a1_status"],"UNQUALIFIED_AT_CEILING")
            self.assertEqual(summary["a2_productive_status"],"QUALIFIED")
            self.assertTrue(summary["a3_one_second_reused"])
            self.assertEqual(summary["reserved_cells"],16)
            self.assertFalse(summary["private_archive_returned"])
            self.assertEqual(summary["scientific_measurements"],0)

    def test_in_process_archive_a1_zero_a2_freeze_a3_reuse_actual_nested_owners(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root).resolve() / "auditor"
            result = supervise_fixed_work(AuthoredStagedAuditor(), directory=directory,
                binding_sha256="f"*64, verify_admission=lambda: True)
            detail = directory / "authored-fixture-error.json"
            self.assertEqual(result["supervisor_exit_code"], 0,
                json.loads(detail.read_bytes()) if detail.exists() else result)
            summary = result["result"]
            self.assertEqual(summary["a1_status"], "UNQUALIFIED_AT_CEILING")
            self.assertEqual(summary["a2_productive_status"], "QUALIFIED")
            self.assertTrue(summary["a3_one_second_reused"])
            self.assertEqual(summary["reserved_cells"], 16)
            self.assertEqual((summary["source_seed_count"], summary["root_slot_count"]), (32, 200))
            self.assertEqual(summary["scientific_measurements"], 0)
            self.assertFalse(result["resource_cost_complete"])
            self.assertNotIn("private-source-snapshot", (directory / "worker-result.json").read_text())
            self.assertEqual(len(list((directory / "delegated-groups").glob("child-*.json"))), 4)
            self.assertFalse(list((directory / "delegated-groups" / "active").iterdir()))


if __name__ == "__main__":
    unittest.main()
