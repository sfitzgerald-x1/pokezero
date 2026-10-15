"""Actual supervised auditor/bridge ownership; authored policies/physics only."""
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import dataclass,replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
from pokezero.mcts_eval.search_over_raw_after_owner import execute_owned_after_cell,read_owned_after_cell
from pokezero.mcts_eval.search_over_raw_actor_projection import SCHEMA,ENCODING_LAW
from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
import test_search_over_raw_after_raw as private_fixture
from test_search_over_raw_construction import _dex


def authored_programs(root):
    result={}
    for seat in ("p1","p2"):
        request=deepcopy(root["boundaryRequests"][seat])
        lines=["|player|p1|PokeZero p1|","|player|p2|PokeZero p2|","|gen|3","|start",
            "|switch|p1a: Swampert|Swampert, L84|100/100","|switch|p2a: Snorlax|Snorlax, L80|100/100","|turn|1"]
        result[seat]=dict(schema=SCHEMA,actor=seat,encoding_law=ENCODING_LAW,opening_request=request,
            events=[dict(kind="protocol",lines=lines),dict(kind="own_request",request=request),dict(kind="observe",coordinate=0)])
    return result


@dataclass(frozen=True)
class AuthoredAfterOwnerJob:
    config: object
    runtime: dict
    fixture_path: str
    mode: str="complete"
    synthetic_process_fixture=True

    def __call__(self,ownership,phase):
        phase("startup")
        root=json.loads(Path(self.fixture_path).read_bytes())["snapshot"]
        programs=authored_programs(root)
        if self.mode=="missing-history":programs["p2"]=None
        policies={s:private_fixture.MaskedChampion() for s in ("p1","p2")}
        for policy in policies.values():policy.weights_sha256=self.runtime["files"][self.runtime["checkpoint_path"]]
        directory=Path(ownership["directory"]).parent/"after-cell"
        identity=dict(leaf_plan_sha256="b"*64,event_id="a"*64,cell_id=digest(dict(event_id="a"*64,replicate=0)),
            canonical_construction_ordinal=0,replicate=0)
        args=dict(ownership=ownership,directory=directory,raw_after=root,programs=programs,config=self.config,runtime=self.runtime,
            sample_seed=17,event_id="a"*64,replicate=0,subject="p1",policies=policies,cell_identity=identity)
        phase("continuations")
        try:
            with patch("pokezero.local_showdown.load_showdown_dex_cached",return_value=_dex()):
                if self.mode.startswith("bank"):
                    import pokezero.mcts_eval.search_over_raw_after_bank as bank_module
                    from pokezero.mcts_eval.search_over_raw_after_bank import execute_owned_after_bank
                    from test_search_over_raw_native_leaf_cells import NativeLeafCellPlanTests
                    helper=NativeLeafCellPlanTests();plan=helper.plan(helper.fixture())
                    event=plan.receipt()["events"][0]["event_id"]
                    if self.mode=="bank-missing":programs["p2"]=None
                    real_cell=bank_module.execute_owned_after_cell
                    def check_first_claim(**kwargs):
                        bank_directory=kwargs["directory"].parent
                        claim=json.loads((bank_directory/"claim.json").read_bytes())
                        if kwargs["replicate"]==0:
                            if len(claim["cell_roster"])!=8 or not all(r["status"]=="UNSTARTED_UNCERTAIN" for r in claim["cell_roster"]):
                                raise ValueError("all eight cells were not claimed before first launch")
                            if list(bank_directory.rglob("bridge-launch.json")):raise ValueError("bridge predates eight-cell claim")
                        if self.mode=="bank-identity-mutation":kwargs["cell_identity"]["canonical_construction_ordinal"]+=1
                        value=real_cell(**kwargs)
                        if self.mode=="bank-extra-receipt":(bank_directory/"unexpected.json").write_text("{}")
                        if self.mode=="bank-stop-after-first":
                            (bank_directory.parent/"operational-stop.json").write_bytes(b'{"schema":"authored-stop","original":true}\n')
                        return value
                    with ExitStack() as stack:
                        stack.enter_context(patch.object(bank_module,"execute_owned_after_cell",side_effect=check_first_claim))
                        if self.mode.startswith("bank-executor-tail"):
                            actual_bank_validate=bank_module.OwnedAfterRawBank.validate
                            actual_cell_validate=bank_module.OwnedAfterRawCell.validate
                            armed=[]
                            def arm_tail(value):
                                checked=actual_bank_validate(value);armed.append(value);return checked
                            def mutate_tail(value):
                                checked=actual_cell_validate(value)
                                if armed and value is armed[0].cells[-1]:
                                    target=armed.pop()
                                    if self.mode=="bank-executor-tail-result":target.result["bridge_owned_cells"]=0
                                    else:
                                        p=target.cells[0].owner.directory/"bridge-close.json";p.write_bytes(p.read_bytes()+b" ")
                                return checked
                            stack.enter_context(patch.object(bank_module.OwnedAfterRawBank,"validate",arm_tail))
                            stack.enter_context(patch.object(bank_module.OwnedAfterRawCell,"validate",mutate_tail))
                        bank=execute_owned_after_bank(ownership=ownership,plan=plan,expected_plan_sha256=plan.sha256,
                            event_id=event,raw_after=root,programs=programs,config=self.config,runtime=self.runtime,policies=policies)
                    if self.mode=="bank-replaced-cells":
                        for cells in ((None,)*8,tuple(reversed(bank.cells)),(bank.cells[0],)*8):
                            try:replace(bank,cells=cells).validate()
                            except ValueError:pass
                            else:raise ValueError("bank accepted replaced/reordered/duplicated original cells")
                    if self.mode in {"bank-late-result","bank-late-prior-cell","bank-late-receipt"}:
                        real_validate=bank_module.OwnedAfterRawCell.validate
                        def late_callback(value):
                            checked=real_validate(value)
                            if value is bank.cells[-1]:
                                if self.mode=="bank-late-result":bank.result["bridge_owned_cells"]=0
                                elif self.mode=="bank-late-prior-cell":bank.cells[0].readout["signed_outcome"]=-1
                                else:
                                    p=bank.cells[0].owner.directory/"bridge-close.json";p.write_bytes(p.read_bytes()+b" ")
                            return checked
                        with patch.object(bank_module.OwnedAfterRawCell,"validate",late_callback):bank.validate()
                    if self.mode=="bank-scope-mutation":
                        object.__setattr__(bank.scope,"depth",bank.scope.depth+1)
                        bank.validate()
                    result=bank.validate()
                    if self.mode=="bank-duplicate":
                        try:execute_owned_after_bank(ownership=ownership,plan=plan,expected_plan_sha256=plan.sha256,
                            event_id=event,raw_after=root,programs=programs,config=self.config,runtime=self.runtime,policies=policies)
                        except FileExistsError:pass
                        else:raise ValueError("consumed eight-cell bank was retried")
                    phase("cleanup")
                    return dict(bank=result,scientific_measurements=0)
                owned=execute_owned_after_cell(**args)
                if self.mode=="duplicate":
                    try:execute_owned_after_cell(**args)
                    except FileExistsError:pass
                    else:raise ValueError("completed cell was silently retried")
                if self.mode=="late-capture":
                    owned.capture.policy_calls[0]["observation_sha256"]="f"*64
                    owned.validate()
                if self.mode=="late-receipt":
                    p=directory/"bridge-close.json";p.write_bytes(p.read_bytes()+b" ")
                    owned.validate()
                if self.mode=="late-scope-runtime":
                    import pokezero.mcts_eval.search_over_raw_after_owner as owner_module
                    original_runtime=owner_module.verify_outcome_runtime
                    def mutate_scope(runtime,config):
                        value=original_runtime(runtime,config)
                        object.__setattr__(owned.owner.scope,"depth",owned.owner.scope.depth+1)
                        return value
                    with patch.object(owner_module,"verify_outcome_runtime",side_effect=mutate_scope):owned.validate()
                reread=read_owned_after_cell(owned.owner,owned.capture,
                    expected_claim=json.loads((directory/"claim.json").read_bytes()),raw_after=root,programs=programs,
                    config=self.config,runtime=self.runtime)
                if digest(reread.readout)!=digest(owned.readout):raise ValueError("original owning reread differs")
                try:pickle.dumps(owned)
                except TypeError:pass
                else:raise ValueError("private owning cell serialized")
            phase("cleanup")
            return dict(readout=owned.readout,bridge_pid=owned.owner.process.pid,
                bridge_returncode=owned.owner.process.returncode,duplicate_rejected=self.mode=="duplicate",
                serialized_actor_program=False,scientific_measurements=0)
        except BaseException as error:
            (Path(ownership["directory"]).parent/"authored-after-error.json").write_text(json.dumps(
                dict(error=type(error).__name__,message=str(error),notes=getattr(error,"__notes__",[]))))
            raise


class AfterOwnerTests(unittest.TestCase):
    def setUp(self):
        h=private_fixture.AfterRawTests();h.setUp();self.addCleanup(h.doCleanups);self.h=h
        self.runtime=outcome_runtime_binding(h.h.config,checkpoint_path=h.h.checkpoint,
            extra_files=[h.h.fixture_path,Path(__file__).resolve(),Path(private_fixture.__file__).resolve()])

    def run_job(self,mode="complete"):
        directory=self.h.h.root/"auditor"
        job=AuthoredAfterOwnerJob(self.h.h.config,self.runtime,str(self.h.h.fixture_path),mode)
        terminal=_supervise_owned_job(job,directory=directory,binding_sha256="c"*64,verify_admission=lambda:True,
            contract=FixedWorkDiagnosticContract(),kind="staged-exploration-run",work_phase="continuations",
            validate_result=lambda result:result["scientific_measurements"]==0)
        return directory,terminal

    def change_bridge_mode(self,mode):
        value=json.loads(self.h.h.fixture_path.read_bytes());value["mode"]=mode
        self.h.h.fixture_path.write_text(json.dumps(value))
        self.runtime=outcome_runtime_binding(self.h.h.config,checkpoint_path=self.h.h.checkpoint,
            extra_files=[self.h.h.fixture_path,Path(__file__).resolve(),Path(private_fixture.__file__).resolve()])

    def test_actual_auditor_launch_release_close_wait4_and_private_independent_reader(self):
        directory,terminal=self.run_job()
        error=directory/"authored-after-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
        result=terminal["result"];readout=result["readout"]
        self.assertEqual((readout["status"],readout["signed_outcome"]),("COMPLETE",1))
        self.assertTrue(readout["bridge_process_ownership_validated"])
        self.assertFalse(readout["enclosing_auditor_terminal_validated"])
        self.assertFalse(readout["independent_terminal_labels"])
        self.assertFalse(readout["original_endpoint_provenance_validated"])
        self.assertEqual(result["bridge_returncode"],0)
        closed=json.loads((directory/"after-cell/bridge-close.json").read_bytes())
        self.assertTrue(closed["normal_close"]);self.assertTrue(closed["reaped"]);self.assertTrue(closed["pumps_drained"])
        self.assertNotIn(b"opening_request",(directory/"job.pickle").read_bytes())
        with self.assertRaises(ProcessLookupError):os.kill(result["bridge_pid"],0)

    def test_closed_cell_cannot_retry_or_choose_a_new_alias(self):
        directory,terminal=self.run_job("duplicate")
        error=directory/"authored-after-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
        self.assertTrue(terminal["result"]["duplicate_rejected"])
        self.assertEqual(len(list(directory.rglob("bridge-launch.json"))),1)

    def test_missing_history_consumes_claim_without_launch_or_score(self):
        directory,terminal=self.run_job("missing-history")
        self.assertEqual(terminal["supervisor_exit_code"],1)
        self.assertTrue((directory/"after-cell/claim.json").is_file())
        self.assertFalse(list(directory.rglob("bridge-launch.json")))
        error=json.loads((directory/"authored-after-error.json").read_bytes())
        self.assertIn("both separately frozen actor histories",error["message"])

    def test_nonzero_bridge_close_and_missing_ack_remain_failed_not_reaped_success(self):
        for mode in ("nonzero","missing-close"):
            h=AfterOwnerTests();h.setUp();self.addCleanup(h.doCleanups);h.change_bridge_mode(mode)
            directory,terminal=h.run_job()
            with self.subTest(mode=mode):
                self.assertEqual(terminal["supervisor_exit_code"],1)
                closed=json.loads((directory/"after-cell/bridge-close.json").read_bytes())
                self.assertFalse(closed["normal_close"])
                self.assertTrue(closed["reaped"])

    def test_late_private_capture_or_receipt_mutation_cannot_revalidate(self):
        for mode in ("late-capture","late-receipt","late-scope-runtime"):
            h=AfterOwnerTests();h.setUp();self.addCleanup(h.doCleanups)
            directory,terminal=h.run_job(mode)
            with self.subTest(mode=mode):
                self.assertEqual(terminal["supervisor_exit_code"],1)
                self.assertEqual(len(list(directory.rglob("bridge-launch.json"))),1)
                error=json.loads((directory/"authored-after-error.json").read_bytes())
                self.assertIn("original",error["message"])

    def test_all_eight_bridge_cells_are_claimed_and_completed_without_scored_label_promotion(self):
        directory,terminal=self.run_job("bank")
        error=directory/"authored-after-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
        bank=terminal["result"]["bank"]
        self.assertEqual((bank["required_cells"],bank["bridge_owned_cells"],bank["complete_transport_results"],
            bank["scored_terminal_labels"]),(8,8,8,0))
        self.assertEqual([r["replicate"] for r in bank["cells"]],list(range(8)))
        self.assertEqual(len({r["cell_id"] for r in bank["cells"]}),8)
        self.assertEqual(len(list(directory.rglob("bridge-launch.json"))),8)
        claims=[json.loads(p.read_bytes()) for p in directory.glob("after-banks/*/claim.json")]
        self.assertEqual(len(claims),1)
        self.assertEqual(len(claims[0]["cell_roster"]),8)
        self.assertTrue(claims[0]["all_eight_cells_consumed"])
        self.assertFalse(bank["native_inventory_owning_validated"])
        self.assertFalse(bank["original_endpoint_provenance_validated"])
        self.assertFalse(bank["scientific_collection_authorized"])

    def test_eight_missing_history_cells_are_consumed_null_and_launch_nothing(self):
        directory,terminal=self.run_job("bank-missing")
        error=directory/"authored-after-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
        bank=terminal["result"]["bank"]
        self.assertEqual(bank["required_cells"],8);self.assertEqual(bank["bridge_owned_cells"],0)
        self.assertTrue(all(c["status"]=="ACTOR_HISTORY_UNAVAILABLE" and c["signed_outcome"] is None for c in bank["cells"]))
        self.assertFalse(list(directory.rglob("bridge-launch.json")))

    def test_second_bank_attempt_cannot_restart_completed_cells(self):
        directory,terminal=self.run_job("bank-duplicate")
        error=directory/"authored-after-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
        self.assertEqual(len(list(directory.rglob("bridge-launch.json"))),8)

    def test_first_failed_cell_poison_stops_remaining_seven_without_replacement(self):
        self.change_bridge_mode("nonzero")
        directory,terminal=self.run_job("bank")
        self.assertEqual(terminal["supervisor_exit_code"],1)
        self.assertEqual(len(list(directory.rglob("bridge-launch.json"))),1)
        self.assertTrue((directory/"after-banks/operational-stop.json").is_file())
        failed=json.loads(next(directory.glob("after-banks/*/failed-bank.json")).read_bytes())
        self.assertEqual(len(failed["cells"]),8)
        self.assertEqual(failed["cells"][0]["status"],"ATTEMPTED_UNCERTAIN")
        self.assertTrue(all(r["status"]=="UNSTARTED_UNCERTAIN" and r["signed_outcome"] is None for r in failed["cells"][1:]))

    def test_original_bank_cells_cannot_be_dropped_reordered_or_duplicated(self):
        directory,terminal=self.run_job("bank-replaced-cells")
        error=directory/"authored-after-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,error.read_text() if error.exists() else terminal)
        self.assertEqual(len(list(directory.rglob("bridge-launch.json"))),8)

    def test_last_cell_callback_cannot_change_bank_or_previously_validated_cell(self):
        for mode in ("bank-late-result","bank-late-prior-cell","bank-late-receipt","bank-scope-mutation"):
            h=AfterOwnerTests();h.setUp();self.addCleanup(h.doCleanups)
            directory,terminal=h.run_job(mode)
            with self.subTest(mode=mode):
                self.assertEqual(terminal["supervisor_exit_code"],1)
                self.assertEqual(len(list(directory.rglob("bridge-launch.json"))),8)
                error=json.loads((directory/"authored-after-error.json").read_bytes())
                self.assertIn("original",error["message"])

    def test_mutable_cell_identity_and_unexpected_receipt_stop_before_next_launch(self):
        for mode in ("bank-identity-mutation","bank-extra-receipt"):
            h=AfterOwnerTests();h.setUp();self.addCleanup(h.doCleanups)
            directory,terminal=h.run_job(mode)
            with self.subTest(mode=mode):
                self.assertEqual(terminal["supervisor_exit_code"],1)
                self.assertEqual(len(list(directory.rglob("bridge-launch.json"))),1)
                self.assertTrue((directory/"after-banks/operational-stop.json").is_file())

    def test_executor_final_callback_cannot_return_changed_result_or_earlier_receipt(self):
        for mode in ("bank-executor-tail-result","bank-executor-tail-receipt"):
            h=AfterOwnerTests();h.setUp();self.addCleanup(h.doCleanups)
            directory,terminal=h.run_job(mode)
            with self.subTest(mode=mode):
                self.assertEqual(terminal["supervisor_exit_code"],1)
                self.assertEqual(len(list(directory.rglob("bridge-launch.json"))),8)
                self.assertTrue((directory/"after-banks/operational-stop.json").is_file())
                self.assertTrue(list(directory.glob("after-banks/*/failed-bank.json")))
                error=json.loads((directory/"authored-after-error.json").read_bytes())
                self.assertIn("original",error["message"])

    def test_stop_created_during_first_cell_prevents_next_launch_and_is_preserved(self):
        directory,terminal=self.run_job("bank-stop-after-first")
        self.assertEqual(terminal["supervisor_exit_code"],1)
        self.assertEqual(len(list(directory.rglob("bridge-launch.json"))),1)
        marker=directory/"after-banks/operational-stop.json"
        self.assertEqual(marker.read_bytes(),b'{"schema":"authored-stop","original":true}\n')
        failed=json.loads(next(directory.glob("after-banks/*/failed-bank.json")).read_bytes())
        self.assertEqual(failed["cells"][0]["status"],"ATTEMPTED_UNCERTAIN")
        self.assertTrue(all(r["status"]=="UNSTARTED_UNCERTAIN" and r["signed_outcome"] is None for r in failed["cells"][1:]))
