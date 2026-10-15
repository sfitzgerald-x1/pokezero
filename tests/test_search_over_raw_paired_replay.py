"""Real constructor/event handling/owners over authored Node streams, no games."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
from pokezero.mcts_eval.search_over_raw_boundary_run import raw_root_json
from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
from pokezero.mcts_eval.search_over_raw_paired_replay import (
    ARTIFACT, OwnedPairedPrefixJob, PairedPrefixRunValidator, read_paired_prefix,
    supervise_paired_prefix_run, sources, validate_inputs,
)
import test_search_over_raw_paired_materialization as paired
import test_search_over_raw_showdown_boundary as showdown


class NoObservationEnv(paired.PairedSyntheticEnv):
    def _observe(self,*args,**kwargs):raise AssertionError("replay made an undeclared observation")
    def observe(self,*args,**kwargs):raise AssertionError("replay made an undeclared observation")


@dataclass(frozen=True)
class AuthoredPairedPrefixJob(OwnedPairedPrefixJob):
    synthetic_process_fixture=True
    def _components(self):return NoObservationEnv(self.config)
    def __call__(self,ownership,phase):
        try:return super().__call__(ownership,phase)
        except BaseException as error:
            (Path(ownership["directory"]).parent/"authored-error.json").write_text(str(error))
            raise


class PairedReplayTests(unittest.TestCase):
    def setUp(self):
        self.p=paired.PairedMaterializationTests();self.p.setUp();self.addCleanup(self.p.doCleanups)
        self.h=self.p.h
        # Same actual initializer as the reviewed constructor fixture, now with
        # a fully mapped raw boundary and actual per-transition stream events.
        self.h.snapshot=showdown.snapshot();self.h.snapshot["battle"]["log"]=[]
        for seat in ("p1","p2"):
            self.h.snapshot["boundaryRequests"][seat]["active"][0]["moves"][0]["pp"]=15
        bridge=self.h.bridge.read_text().replace("current.battle.log.push('|turn|2');",
            "current.battle.log.push('|turn|'+(generation+1)); "+
            "send({type:'stream',battleId:b,stream:'omniscient',lines:['|turn|'+(generation+1)]});")
        bridge=bridge.replace("send({type:'ready',battleId:b,requested:['p1','p2']});\n   } else {",
            "for(const seat of ['p1','p2']) send({type:'stream',battleId:b,stream:seat,lines:['|request|'+JSON.stringify(current.boundaryRequests[seat])]}); "+
            "send({type:'ready',battleId:b,requested:['p1','p2']});\n   } else {")
        self.h.bridge.write_text(bridge)

    def setup_job(self, *, transitions=2,disable=False,mode="complete",clone_stream=False):
        if clone_stream:
            text=self.h.bridge.read_text().replace("send({type:'cloned',...owned,requested:['p1','p2']});",
                "send({type:'stream',battleId:b,stream:'omniscient',lines:['|turn|98']}); "+
                "send({type:'cloned',...owned,requested:['p1','p2']});")
            self.h.bridge.write_text(text)
        self.h.configure(mode)
        value=json.loads(self.h.fixture_path.read_bytes())
        value.update(nonterminalReplay=True,disableNativeActionAfter=disable)
        self.h.fixture_path.write_text(json.dumps(value))
        self.h.runtime=outcome_runtime_binding(self.h.config,checkpoint_path=self.h.checkpoint,extra_files=[self.h.fixture_path])
        constructor,terminal,detail=self.p.run_job()
        self.assertEqual(terminal["supervisor_exit_code"],0,detail)
        result=terminal["result"]
        snapshot=self.p.reader(constructor).retained_context(result)
        root=deepcopy(snapshot.bridge_snapshot)
        origin=dict(materialization_capture_sha256=result["capture_sha256"],produced_bridge_snapshot_sha256=digest(root),
            native_selected_sha256="c"*64,native_capture_sha256="d"*64,construction_ordinal=0,evaluation_ordinal=0)
        root_binding=dict(root_binding=dict(id="authored-native-root"))
        actions=dict(p1=dict(kind="move",slot=0,engine_id="seismictoss"),p2=dict(kind="move",slot=0,engine_id="splash"))
        boundaries=[dict(ordinal=i,binding=dict(ordinal=i,capture_sha256="d"*64,evaluation_ordinal=0,
            root_binding=root_binding["root_binding"]),native=dict(action_identity=None if i==0 else deepcopy(actions)),
            native_export_sha256="e"*64,native_export_utf8_sha256="f"*64,independently_validated=False,
            party_mapping_validated=False,runtime_qualified=False,scientific_collection_authorized=False)
            for i in range(transitions+1)]
        clone=dict(battleId="paired-replay-fixture",cloneId="paired-replay-path",
            snapshotJsonSha256=hashlib.sha256(raw_root_json(root).encode()).hexdigest())
        job=AuthoredPairedPrefixJob(self.h.config,root,root_binding,origin,self.h.runtime,clone,
            ["paired-"+str(i) for i in range(transitions+1)],boundaries,showdown.mapping(root),snapshot,
            result["paired_context"])
        return job,snapshot

    def run_job(self,job):
        self.directory=self.h.root/"paired-replay"
        terminal=supervise_paired_prefix_run(job,directory=self.directory,binding_sha256="a"*64,
            verify_admission=lambda:True,validate_result=lambda _:True)
        detail=self.directory/"authored-error.json"
        self.assertEqual(terminal["supervisor_exit_code"],0,detail.read_text() if detail.exists() else terminal)
        self.endpoint=deepcopy(terminal["result"]["paired_endpoint"])
        return terminal["result"]

    def reader(self,job,allow_synthetic=True):
        return PairedPrefixRunValidator(directory=self.directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64,expected_inputs=job.input_binding(),expected_raw_root=job.raw_root,runtime=job.runtime,
            snapshot=job.snapshot,expected_endpoint=self.endpoint,allow_synthetic=allow_synthetic)

    def test_actual_constructor_to_paired_replay_and_independently_folded_endpoint(self):
        job,root=self.setup_job();result=self.run_job(job);reader=self.reader(job)
        self.assertEqual(reader(result["capture"]),digest(result["capture"]))
        endpoint=reader.retained_endpoint(result)
        self.assertEqual(endpoint.latest_turn,3)
        self.assertTrue(all(len(endpoint.request_history[s])==len(root.request_history[s])+2 for s in ("p1","p2")))
        self.assertEqual(endpoint.latest_requests,endpoint.bridge_snapshot["boundaryRequests"])
        self.assertNotEqual(digest(_state_payload(endpoint.replay)),digest(_state_payload(root.replay)))
        self.assertEqual(len(result["capture"]["paired_contexts"]),3)
        self.assertEqual(result["paired_replay_law"]["observation_players"],[])
        self.assertFalse(result["independent_terminal_truth"])
        self.assertFalse(result["scientific_collection_authorized"])
        self.assertIn(ARTIFACT,reader.last_validation["receipt_file_sha256"])
        with self.assertRaises(FileExistsError):self.run_job(job)
        self.assertIsNone(self.reader(job,False)(result["capture"]))
        with self.assertRaises(ValueError):self.reader(job,False).retained_endpoint(result)

    def test_unavailable_suffix_preserves_context_but_cannot_supply_after_endpoint(self):
        job,_=self.setup_job(disable=True);result=self.run_job(job);reader=self.reader(job)
        self.assertEqual(reader(result["capture"]),digest(result["capture"]))
        self.assertEqual(result["capture"]["termination"]["required_boundary_records"],3)
        self.assertEqual(len(result["capture"]["paired_contexts"]),2)
        with self.assertRaisesRegex(ValueError,"cannot substitute"):reader.retained_endpoint(result)
        self.assertIsNone(reader.last_validation)

    def test_clone_shell_streams_are_discarded_at_actual_restore_boundary(self):
        job,_=self.setup_job(clone_stream=True);result=self.run_job(job)
        endpoint=self.reader(job).retained_endpoint(result)
        self.assertEqual(endpoint.latest_turn,3)
        self.assertNotIn("|turn|98",endpoint.protocol_lines)
        self.assertTrue(any("|turn|98" in row["event"]["lines"] for row in result["capture"]["replay_streams"]))

    def test_original_transport_rejects_changed_private_streams_or_context(self):
        job,_=self.setup_job();result=self.run_job(job)
        for mode in ("stream","missing","context","alias"):
            capture=deepcopy(result["capture"])
            if mode=="stream":capture["replay_streams"][0]["event"]["lines"].append("|turn|99")
            elif mode=="missing":capture["replay_streams"].pop()
            elif mode=="context":capture["paired_contexts"][-1]["snapshot_sha256"]="f"*64
            else:capture["replay_streams"][0]["ordinal"]=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                read_paired_prefix(capture,process_records=result["capture"]["records"],snapshot=job.snapshot,expected=job.input_binding())

    def test_private_endpoint_and_external_pin_cannot_be_rehashed_into_acceptance(self):
        job,_=self.setup_job();result=self.run_job(job);reader=self.reader(job)
        path=self.directory/ARTIFACT;original=path.read_bytes();path.write_bytes(original+b" ")
        with patch("pokezero.mcts_eval.search_over_raw_paired_replay.pickle.loads",side_effect=AssertionError("unexpected load")):
            with self.assertRaisesRegex(ValueError,"byte drift"):reader.retained_endpoint(result)
        path.write_bytes(original)
        self.assertIsNotNone(reader.retained_endpoint(result))
        changed=deepcopy(result);changed["paired_endpoint"]["snapshot_sha256"]="f"*64
        with self.assertRaises(ValueError):reader.retained_endpoint(changed)

    def test_late_owning_callback_cannot_change_endpoint_file(self):
        job,_=self.setup_job();result=self.run_job(job);reader=self.reader(job)
        path=self.directory/ARTIFACT;original=path.read_bytes();check=reader._match_capture
        def alter(*args):check(*args);path.write_bytes(original+b" ")
        with patch.object(reader,"_match_capture",side_effect=alter):
            with self.assertRaisesRegex(ValueError,"owning callbacks"):reader(result["capture"])
        self.assertIsNone(reader.last_validation)
        path.write_bytes(original);self.assertIsNotNone(reader.retained_endpoint(result))

    def test_frozen_law_and_controller_inventory(self):
        job,_=self.setup_job();binding=job.input_binding();validate_inputs(binding)
        from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes
        inventory=controller_input_hashes();self.assertTrue(all(inventory.get(p)==sha for p,sha in sources().items()))
        binding["paired_replay_law"]["policy_inference_during_replay"]=0
        with self.assertRaises(ValueError):validate_inputs(binding)


class ExcludedShowdownPairedTests(unittest.TestCase):
    """Existing excluded root; real bridge, authored origin/action roster only.

    No native evaluator, champion inference, fresh panel, or scientific labels.
    This specifically checks Python/actual-Showdown context interoperability.
    """
    def test_real_paired_constructor_and_single_step_replay_fold_the_same_context(self):
        from dataclasses import asdict
        from pokezero.env import BattleStartOverride
        from pokezero.mcts_eval.search_over_raw_oracle import opening_team
        from pokezero.mcts_eval.search_over_raw_materialization import materialization_inputs,recorded_materialization_inputs
        from pokezero.mcts_eval.search_over_raw_paired_materialization import (
            OwnedPairedMaterializationJob,PairedMaterializationRunValidator,supervise_paired_materialization_run)
        from pokezero.mcts_eval.search_over_raw_showdown_boundary import freeze_party_mapping,project_showdown_boundary,_species
        from pokezero.showdown_fixture import pack_team
        from test_search_over_raw_archive import ActualSourceBoundaryArchiveTests
        import qualify_search_over_raw_source as driver
        fixture=ActualSourceBoundaryArchiveTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        root=fixture.root(fixture.collect())
        context,_,source_snapshot=fixture.archive.selected(root)
        state=context.public_materialization_state
        source_active=next(side for side in source_snapshot.bridge_snapshot["battle"]["sides"]
            if side["id"]==state.player_id)["pokemon"][0]
        self.assertTrue(source_active["fainted"])
        self.assertFalse(source_active["isActive"])
        # Both private opening teams are test inputs only. They do not enter a
        # public search adapter, nor assert genuine native consumption ancestry.
        override=BattleStartOverride(player_teams={seat:pack_team(opening_team(
            source_snapshot.first_requests[seat],seat,fixture.env._belief_set_source)) for seat in ("p1","p2")},
            observation_format_id=state.observation_format_id)
        binding=dict(subject=state.player_id,fixture_only=True,source_seed=driver.FIXTURE_SEED,
            root_binding=digest(dict(excluded_fixture_root=root["root_id"])))
        original=materialization_inputs(state=state,override=override,seed=driver.FIXTURE_SEED,root_binding=binding)
        authored_origin=dict(schema="pokezero.search-over-raw.recorded-native-materialization.v1",
            native_selected_sha256=digest(dict(authored_native_selection=True)),
            construction_capture_sha256=digest(dict(authored_native_constructor=True)),construction_ordinal=0,
            original_public_request_sha256=digest(dict(excluded_root=root["root_id"])),
            draw_sha256=original["draw_sha256"],public_payload_sha256=original["public_payload_sha256"])
        inputs=recorded_materialization_inputs(draw=asdict(override),public_payload=original["materialize"]["publicState"],
            seed=driver.FIXTURE_SEED,root_binding=binding,public_state_sha256=original["public_state_sha256"],
            recorded_native_origin=authored_origin,options=original["options"])
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve();checkpoint=directory/"unused-checkpoint-binding"
            checkpoint.write_bytes(b"not loaded: constructor and replay have no inference")
            runtime=outcome_runtime_binding(fixture.env.config,checkpoint_path=checkpoint)
            constructor=OwnedPairedMaterializationJob(fixture.env.config,inputs,runtime,state)
            terminal=supervise_paired_materialization_run(constructor,directory=directory/"constructor",binding_sha256="a"*64,
                verify_admission=lambda:True,validate_result=lambda _:True)
            self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
            result=terminal["result"]
            reader=PairedMaterializationRunValidator(directory=directory/"constructor",
                job_sha256=hashlib.sha256(pickle.dumps(constructor)).hexdigest(),binding_sha256="a"*64,
                expected_inputs=constructor.input_binding(),runtime=runtime,original_public_context=state,
                expected_context=result["paired_context"])
            snapshot=reader.retained_context(result);raw=snapshot.bridge_snapshot
            active=next(side for side in raw["battle"]["sides"] if side["id"]==state.player_id)["pokemon"][0]
            self.assertTrue(active["fainted"])
            self.assertFalse(active["isActive"])
            self.assertTrue(raw["boundaryRequests"][state.player_id]["forceSwitch"][0])
            native=dict(root=True)
            orders={}
            for side in raw["battle"]["sides"]:
                seat=side["id"];order=[int(v)-1 for v in side["team"]]
                native[seat]=dict(party_identities=[dict(slot=i,engine_id=_species(side["pokemon"][current]))
                    for i,current in enumerate(order)])
                orders[seat]=list(range(len(order)))
            try:mapping=freeze_party_mapping(raw,native,native_to_original=orders)
            except ValueError as error:
                self.fail(str(error)+": "+repr([{k:mon.get(k) for k in
                    ("position","isActive","hp","maxhp","fainted","status")} for side in raw["battle"]["sides"]
                    for mon in side["pokemon"]]))
            projection=project_showdown_boundary(raw,mapping)
            actions={seat:(projection["showdown"]["effective_legal_choices"][seat][0]
                if projection["showdown"]["effective_legal_choices"][seat] else dict(kind="none",slot=None,engine_id="No Move"))
                for seat in ("p1","p2")}
            capture_sha=digest(dict(authored_action_roster=actions))
            origin=dict(materialization_capture_sha256=result["capture_sha256"],produced_bridge_snapshot_sha256=digest(raw),
                native_selected_sha256=authored_origin["native_selected_sha256"],native_capture_sha256=capture_sha,
                construction_ordinal=0,evaluation_ordinal=0)
            boundaries=[dict(ordinal=i,binding=dict(ordinal=i,capture_sha256=capture_sha,evaluation_ordinal=0,
                root_binding=binding["root_binding"]),native=dict(action_identity=actions if i else None),
                native_export_sha256=digest(dict(authored_boundary=i)),
                native_export_utf8_sha256=digest(dict(authored_boundary_utf8=i)),
                independently_validated=False,party_mapping_validated=False,runtime_qualified=False,
                scientific_collection_authorized=False) for i in range(2)]
            clone=dict(battleId="excluded-paired-replay",cloneId="excluded-paired-path",
                snapshotJsonSha256=hashlib.sha256(raw_root_json(raw).encode()).hexdigest())
            job=OwnedPairedPrefixJob(fixture.env.config,raw,binding,origin,runtime,clone,["excluded-0","excluded-1"],
                boundaries,mapping,snapshot,result["paired_context"])
            replay_terminal=supervise_paired_prefix_run(job,directory=directory/"replay",binding_sha256="b"*64,
                verify_admission=lambda:True,validate_result=lambda _:True)
            self.assertEqual(replay_terminal["supervisor_exit_code"],0,replay_terminal)
            replay_result=replay_terminal["result"]
            owner=PairedPrefixRunValidator(directory=directory/"replay",job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
                binding_sha256="b"*64,expected_inputs=job.input_binding(),expected_raw_root=raw,runtime=runtime,
                snapshot=snapshot,expected_endpoint=replay_result["paired_endpoint"])
            endpoint=owner.retained_endpoint(replay_result)
            self.assertEqual(len(replay_result["capture"]["paired_contexts"]),2)
            self.assertGreater(len(endpoint.protocol_lines),len(snapshot.protocol_lines))
            self.assertEqual(endpoint.latest_requests,endpoint.bridge_snapshot["boundaryRequests"])
            self.assertFalse(replay_result["runtime_qualified"])
            self.assertFalse(replay_result["scientific_collection_authorized"])
