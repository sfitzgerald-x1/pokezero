"""Authored fixed-budget/control-flow fixtures; no games or model inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from pokezero.mcts_eval import search_over_raw_fixed_exploration as fixed
from pokezero.mcts_eval.search_over_raw_root_priors import native_measure,RootInputUnavailable
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
from pokezero.mcts_eval.search_over_raw_world_inputs import NativeWorldInputs,NativeInputHandle,validate_fixed_work_inputs
from test_search_over_raw_world_inputs import BINDING,construct,arguments
from test_search_over_raw_native_atomic_work import native_report


class FixedExplorationTests(unittest.TestCase):
    def contract(self,total=320):
        return fixed.contract("51f8beec-d601-4fdc-ab73-f0517cd05001",list(range(100,132)),[1,2,3],
            dict(path="/authored/steer",sha256="a"*64),total)

    def test_total_budget_is_not_four_times_the_requested_work_or_a_deadline(self):
        for total,per_world,leaves in ((320,80,["model","hp_fraction"]),(1024,256,["model"]),(4096,1024,["model"])):
            c=self.contract(total);fixed.validate_contract(c)
            self.assertEqual(c["native_iterations_per_world"],per_world)
            self.assertEqual(c["native_worlds"]*per_world,total)
            self.assertEqual(c["leaves"],leaves)
            self.assertEqual(len(c["panels"]["exploration"]["root_slots"]),200)
            self.assertIsNone(c["wall_clock_search_budget"])
            for key in ("native_iterations_per_world","continuation_replicates","holdout_authorized"):
                broken=deepcopy(c);broken[key]=True
                with self.assertRaises(ValueError):fixed.validate_contract(broken)
        with self.assertRaises(ValueError):self.contract(64)
        with self.assertRaises(ValueError):FixedWorkDiagnosticContract(incumbent_iterations_per_world=80)

    def test_real_atomic_reader_accepts_exact_320_only_under_explicit_new_allocation(self):
        recorder=NativeWorldInputs(BINDING,80,deadline_limited=False,record_atomic_work=True)
        for i in range(4):
            text="authored-state-"+str(i);construct(recorder,text=text,hp=80+i)
            handle=NativeInputHandle(SimpleNamespace(search_batched_multi_encoded=Mock(
                return_value=json.dumps(native_report(80,text=text)))),recorder)
            handle.search_batched_multi_encoded(*arguments(text=text,iterations=80))
        evidence=dict(total_iterations=320)
        evidence["consumed_world_inputs"]=recorder.finish(evidence,attempted=4,constructed=4,searched=4)
        kwargs=dict(arm="incumbent",binding=BINDING,statistics_root_id="authored-root")
        with self.assertRaises(ValueError):validate_fixed_work_inputs(evidence,**kwargs)
        self.assertIsInstance(validate_fixed_work_inputs(evidence,iterations_per_world=80,**kwargs),str)
        for budget in (16,256,1024,True):
            with self.assertRaises(ValueError):validate_fixed_work_inputs(evidence,iterations_per_world=budget,**kwargs)
        changed=deepcopy(evidence);changed["total_iterations"]-=1
        with self.assertRaises(ValueError):validate_fixed_work_inputs(changed,iterations_per_world=80,**kwargs)

    def test_only_exact_unrepresentable_move_guard_is_unavailable_and_runtime_is_closed(self):
        for message,expected in (("side_one.last_used_move must be 'move:<slot>' or 'switch:<idx>' with a numeric index (engine panics on move ids), got 'move:struggle'",RootInputUnavailable),
                ("some unrelated validation failure",ValueError)):
            original=ValueError(message);native_factory=Mock()
            policy=SimpleNamespace(_native=native_factory,_search=Mock(side_effect=original),
                _config=SimpleNamespace(model_decision_time_ms=None,worlds=4))
            decider=SimpleNamespace(close=Mock());boundary=SimpleNamespace(validate=Mock())
            context=SimpleNamespace(public_materialization_state=None,observation=object())
            recorder=SimpleNamespace(attempt=Mock(),constructed=Mock(),world_built=Mock(),native_built=Mock(),rows=[],calls=[])
            with patch("pokezero.mcts_eval.search_over_raw_adapters._incumbent_runtime",return_value=(decider,policy,None)) as runtime, \
                    patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture",return_value=object()), \
                    patch("pokezero.mcts_eval.search_over_raw_world_inputs._binding",return_value=BINDING), \
                    patch("pokezero.mcts_eval.search_over_raw_root_priors.public_request_witness",return_value={}), \
                    patch("pokezero.mcts_eval.search_over_raw_world_inputs.NativeWorldInputs",return_value=recorder):
                with self.assertRaises(expected):native_measure(context,boundary,SimpleNamespace(showdown_root="/authored"),
                    100,accepted_iterations=320,leaf="hp_fraction")
                self.assertEqual(runtime.call_args.kwargs["fixed_work_iterations"],80)
            decider.close.assert_called_once();self.assertIs(policy._native,native_factory)

    def test_full_and_missing_denominators_and_descriptive_means_are_separate(self):
        c=self.contract();s=dict(contract=c)
        terminal=dict(supervisor_exit_code=0,worker_exit_code=0,status="COMPLETE",elapsed_wall_seconds=1.,cpu_seconds_including_reaped_children=.8)
        empty=fixed.summarize(s,[],terminal)
        self.assertEqual(empty["effects"]["model"]["fixed_seed_weighted_bounds"],[-1.,1.])
        self.assertIsNone(empty["effects"]["model"]["observed_complete_case_mean"])
        rows=[dict(root_id=r["root_id"],source_seed=r["source_seed"],intervals=dict(model=[.25,.25],hp_fraction=[-.125,-.125]),
            actions=dict(raw=0,model=1,hp_fraction=2)) for r in c["panels"]["exploration"]["root_slots"]]
        full=fixed.summarize(s,rows,terminal)
        for leaf,want in (("model",.25),("hp_fraction",-.125)):
            self.assertEqual(full["effects"][leaf]["fixed_seed_weighted_bounds"],[want,want])
            self.assertEqual(full["effects"][leaf]["observed_complete_case_mean"],want)
            self.assertEqual(full["effects"][leaf]["uncertain_fixed_slots"],0)
        self.assertFalse(full["scientific_strength_claim"])
        with self.assertRaises(ValueError):fixed.summarize(s,rows+[rows[0]],terminal)

    def test_actual_outer_job_continues_and_resets_until_three_unavailable_roots(self):
        for missing,expected in (([True]*7,3),([True,True,False,True,True,True,False],6),([True,False,False,False,False,False,False],7)):
            with tempfile.TemporaryDirectory() as temporary:
                out=Path(temporary);(out/"completed-roots").mkdir()
                s=dict(contract=self.contract(),output_directory=str(out))
                s["contract"]["panels"]["exploration"]["seeds"]=[100]
                path=out/"spec.json";path.write_text("authored fixture")
                archive=SimpleNamespace(close=Mock());roots=[dict(root_id=f"exploration:100:{i}",source_request_index=i) for i in range(7)]
                def source(*args,**kwargs):
                    for i in range(7):kwargs["root_capture"](None,archive,SimpleNamespace(decision_round_index=i),0,None)
                    return archive,dict(roots=roots,missing_root_ids=[]),{}
                rows=[dict(root_id=r["root_id"],actions={},intervals={},input_unavailable={"model":"known"} if v else {})
                    for r,v in zip(roots,missing)]
                phases=[]
                with patch.object(fixed,"read_spec",return_value=(s,None,None)), \
                        patch.object(fixed,"capture_actor_boundary",return_value=object()), \
                        patch("pokezero.mcts_eval.search_over_raw_bounded_a2.collect_source",side_effect=source), \
                        patch.object(fixed,"measure_root",side_effect=rows) as measure:
                    fixed.FixedExplorationJob(str(path),hashlib.sha256(path.read_bytes()).hexdigest())(None,phases.append)
                self.assertEqual(measure.call_count,expected)
                self.assertEqual(phases,["startup","selection","cleanup"])
                archive.close.assert_called_once()

    def test_measure_root_handoff_uses_actual_owned_synthetic_continuations_and_reader(self):
        from test_search_over_raw_outcome_run import OutcomeRunTests,SyntheticContinuationJob
        from test_search_over_raw_outcomes import BINDING,NAMESPACE
        from pokezero.mcts_eval.search_over_raw_outcome_run import OwningRunValidator
        for admit_synthetic in (True,False):
            fixture=OutcomeRunTests();fixture.setUp()
            try:
                binding=deepcopy(BINDING)
                context=SimpleNamespace(public_materialization_state=None,battle_id="authored-source",
                    observation=SimpleNamespace(legal_action_mask=binding["legal_action_mask"]))
                archive=SimpleNamespace(selected=Mock(return_value=(context,None,fixture.snapshot)))
                root=dict(root_id=binding["root_id"],source_request_index=0,
                    public_record_sha256=binding["public_record_sha256"])
                root["public_record"]=dict(current_legal_action_mask=binding["legal_action_mask"])
                source=dict(source_seed=binding["source_seed"],roots=[root],missing_root_ids=[])
                checkpoint=SimpleNamespace(showdown_root="/authored",checkpoint_sha256="a"*64,
                    showdown_source_sha256="a"*64)
                spec=dict(contract={**self.contract(),"namespace":NAMESPACE},source_runtime=fixture.runtime,
                    output_directory=str(fixture.root),checkpoint_contract={})
                raw_job=Mock(return_value=dict(action=0))
                def select(*args,leaf,**kwargs):
                    return dict(action=1 if leaf=="model" else 0,actual_native_evidence={},actual_model_argmax=0,
                        accepted_target=320,fixed_iterations=320)
                def reader(**kwargs):
                    return OwningRunValidator(**kwargs,allow_synthetic=admit_synthetic)
                with patch("pokezero.mcts_eval.search_over_raw_archive.retained_archive_root_identity",return_value="frozen"), \
                        patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture",return_value=object()), \
                        patch.object(fixed,"public_request_witness",return_value=dict(public_request_sha256="b"*64,
                            consumed_binding=dict(information_key=binding["information_key"]))), \
                        patch("pokezero.mcts_eval.search_over_raw_raw_run.RawSelectionJob",return_value=raw_job), \
                        patch.object(fixed,"native_measure",side_effect=select), \
                        patch.object(fixed,"compare_world_inputs",return_value=dict(entire_population_matched=True)), \
                        patch("pokezero.mcts_eval.search_over_raw_outcome_run.OwnedContinuationJob",SyntheticContinuationJob), \
                        patch("pokezero.mcts_eval.search_over_raw_outcome_run.OwningRunValidator",side_effect=reader), \
                        patch("pokezero.mcts_eval.search_over_raw_exploration_entry._decode",return_value=checkpoint), \
                        patch("pokezero.mcts_eval.search_over_raw_raw_run.read_raw_selection") as raw_reader, \
                        patch("pokezero.mcts_eval.search_over_raw_world_inputs.validate_fixed_work_inputs") as native_reader:
                    directory=fixture.root/"roots"/fixed.digest(root["root_id"])
                    row=fixed.measure_root(spec,fixture.config,checkpoint,archive,source,root,
                        SimpleNamespace(validate=Mock()),directory,{},Mock())
                    row["receipt_files"]=fixed._pins(directory)
                    self.assertEqual(fixed.read_completed_root(spec,row,source),row)
                    raw_reader.assert_called_once();self.assertEqual(native_reader.call_count,2)
                    for key,value in (("actions",dict(raw=0,model=0)),("intervals",{}),("cells",[])):
                        changed={**row,key:value}
                        with self.assertRaises(ValueError):fixed.read_completed_root(spec,changed,source)
                    (directory/"model-selected.json").write_text("changed after acceptance")
                    with self.assertRaises(ValueError):fixed.read_completed_root(spec,row,source)
                self.assertEqual(len(row["cells"]),16)  # HP aliases raw; never a third bank.
                self.assertEqual({c["validated"] for c in row["cells"]},{admit_synthetic})
                self.assertEqual(row["intervals"]["model"],[0.,0.] if admit_synthetic else [-1.,1.])
                self.assertEqual(row["intervals"]["hp_fraction"],[0.,0.])
                self.assertEqual(row["actions"],dict(raw=0,model=1,hp_fraction=0))
            finally:fixture.doCleanups()

    def test_prepared_entry_requires_complete_inventory_and_exact_checkpoint_runtime_join(self):
        from dataclasses import replace
        from test_search_over_raw_outcome_run import OutcomeRunTests
        from pokezero.mcts_eval.resolver import CheckpointContract
        from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
        fixture=OutcomeRunTests();fixture.setUp()
        try:
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            model=fixture.root/"authored-model";model.write_bytes(b"not a model")
            tables=fixture.root/"authored-tables";tables.write_text("{}")
            authority=fixture.root/"authored-steer";authority.write_text("synthetic fixed-work authority")
            checkpoint_path=fixture.runtime["checkpoint_path"]
            checkpoint=CheckpointContract(checkpoint_path,sha(checkpoint_path),"authored","v4",1,1,1,0,(),{},{},"cpu",
                showdown_root=str(fixture.config.resolved_showdown_root().resolve()),showdown_source_sha256="a"*64,
                model_path=str(model.resolve()),model_sha256=sha(model),tables_path=str(tables.resolve()),tables_sha256=sha(tables))
            parent=fixture.root/"parent.json"
            parent.write_text(json.dumps(dict(contract=self.contract(),source_config=_state_payload(fixture.config),
                checkpoint_contract=_state_payload(checkpoint),exposure_inventory={str(authority):sha(authority)},input_hashes={})))
            result=fixed.prepare(fixture.root/"new-entry",parent,authority,320)
            path=Path(result["entry_spec"]);spec,_,_=fixed.read_spec(path)
            for mutate in (lambda s:s["input_hashes"].pop(str(Path(fixed.__file__).resolve())),
                    lambda s:s.update(checkpoint_contract=_state_payload(replace(checkpoint,checkpoint_sha256="b"*64))),
                    lambda s:s.update(checkpoint_contract=_state_payload(replace(checkpoint,model_sha256="b"*64)))):
                changed=deepcopy(spec);mutate(changed);path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):fixed.read_spec(path)
        finally:fixture.doCleanups()


if __name__=="__main__":unittest.main()
