"""Authored snapshots and real disposable bridge owners; no scientific games."""
from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_boundary_run import raw_root_json, supervise_native_path_boundary_run
from pokezero.mcts_eval.search_over_raw_native_prefix import (
    OwnedNativePrefixBoundaryJob, NativePrefixBoundaryRunValidator, supervise_native_prefix_boundary_run,
    read_native_prefix_transport, termination_witness,
    _validate_inputs, prefix_source_hashes,
)
from pokezero.mcts_eval.search_over_raw_showdown_boundary import (
    commands_for_native_actions, NativeReplayUnavailable, project_showdown_boundary,
)
from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
from pokezero.mcts_eval.search_over_raw_consumed_materialization import join_owned_ancestry_roots
import test_search_over_raw_consumed_materialization as consumed
import test_search_over_raw_materialization_run as material
import test_search_over_raw_showdown_boundary as showdown


@dataclass(frozen=True)
class AuthoredPrefixJob(OwnedNativePrefixBoundaryJob):
    synthetic_process_fixture = True
    def _components(self): return material.SyntheticEnv(self.config)
    def __call__(self,ownership,phase):
        try:return super().__call__(ownership,phase)
        except BaseException as error:
            import traceback
            repo=Path(__file__).resolve().parents[1]
            frames=[dict(file=str(Path(f.filename).relative_to(repo)),line=f.lineno,function=f.name)
                for f in traceback.extract_tb(error.__traceback__) if Path(f.filename).is_relative_to(repo)]
            (Path(ownership["directory"]).parent/"authored-prefix-failure.json").write_text(json.dumps(
                dict(error_type=type(error).__name__,locations=frames[-6:])))
            raise


class NativePrefixTests(unittest.TestCase):
    def setUp(self):
        self.h=consumed.OwnedRawReplayJoinTests(); self.addCleanup(self.h.doCleanups)
        self.selected,self.mats,_,self.native=self.h.fixture(native_driven=True,skip_replay=True,with_replay_log=True)
        self.config=self.h.planner_config
        self.root=Path(self.config.resolved_showdown_root()).parent
        self.produced=self.mats[0]["result"]
        self.roots=join_owned_ancestry_roots(self.selected,self.mats,native_validator=self.native,
            ancestry_contract=consumed.CONTRACT,native_extract=self.h.extractor)
        self.event=self.roots["rows"][0]["sampled_events"][0]
        self.serial=0

    def job(self, *, transitions=2, nonterminal=True, disable=True, mode="complete", strict=False):
        self.serial+=1
        fixture=Path(self.config.resolved_showdown_root())/"fixture.json"
        value=json.loads(fixture.read_bytes())
        value.update(mode=mode,nonterminalReplay=nonterminal,disableNativeActionAfter=disable)
        fixture.write_text(json.dumps(value))
        checkpoint=self.mats[0]["validator"].runtime["checkpoint_path"]
        runtime=outcome_runtime_binding(self.config,checkpoint_path=checkpoint)
        root=deepcopy(self.produced["produced_bridge_snapshot"])
        boundaries=deepcopy(self.event["boundaries"])
        self.assertEqual(len(boundaries),2)
        while len(boundaries)<transitions+1:
            row=deepcopy(boundaries[-1]); row["ordinal"]=len(boundaries); row["binding"]["ordinal"]=len(boundaries)
            boundaries.append(row)  # Authored extra native target, not native ancestry evidence.
        boundaries=boundaries[:transitions+1]
        origin=dict(materialization_capture_sha256=self.produced["capture_sha256"],
            produced_bridge_snapshot_sha256=digest(root),native_selected_sha256=digest(self.selected),
            native_capture_sha256=self.event["capture_sha256"],construction_ordinal=0,
            evaluation_ordinal=self.event["evaluation_ordinal"])
        token="prefix-fixture-"+str(self.serial)
        clone=dict(battleId=token,cloneId=token+"-clone",snapshotJsonSha256=hashlib.sha256(raw_root_json(root).encode()).hexdigest())
        cls=consumed.AuthoredNativeReplayJob if strict else AuthoredPrefixJob
        return cls(self.config,root,deepcopy(consumed.INPUT_BINDING),origin,runtime,clone,
            [token+":"+str(i) for i in range(transitions+1)],boundaries,deepcopy(self.roots["rows"][0]["mapping"]))

    def run_job(self,job):
        directory=self.root/("prefix-attempt-"+str(self.serial))
        supervise=supervise_native_path_boundary_run if type(job) is consumed.AuthoredNativeReplayJob else supervise_native_prefix_boundary_run
        terminal=supervise(job,directory=directory,binding_sha256="a"*64,verify_admission=lambda:True,
            validate_result=lambda _:True)
        failure=directory/"authored-prefix-failure.json"
        if failure.exists(): print("AUTHORED_PREFIX_FAILURE:"+failure.read_text())
        return directory,terminal

    def reader(self,job,directory):
        return NativePrefixBoundaryRunValidator(directory=directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64,expected_inputs=job.input_binding(),expected_raw_root=job.raw_root,
            runtime=job.runtime,allow_synthetic=True)

    def test_prefix_law_sources_are_pinned_and_drift_or_omission_refuses(self):
        job=self.job();binding=job.input_binding()
        from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes
        sources=prefix_source_hashes();inventory=controller_input_hashes()
        self.assertTrue(all(inventory.get(p)==sha for p,sha in sources.items()))
        _validate_inputs(binding)
        changed=deepcopy(binding);changed.pop("prefix_derivation_sources")
        with self.assertRaises(ValueError):_validate_inputs(changed)
        changed=deepcopy(binding);changed["prefix_law"]["fallback_permitted"]=0
        with self.assertRaises(ValueError):_validate_inputs(changed)
        changed_sources=dict(sources);changed_sources[next(iter(sources))]="f"*64
        with patch("pokezero.mcts_eval.search_over_raw_native_prefix.prefix_source_hashes",return_value=changed_sources):
            with self.assertRaises(ValueError):_validate_inputs(binding)

    def test_later_unavailable_action_retains_prefix_full_roster_and_normal_cleanup(self):
        job=self.job(); directory,terminal=self.run_job(job)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        capture=terminal["result"]["capture"]
        self.assertEqual(self.reader(job,directory)(capture),digest(capture))
        self.assertEqual(len(capture["request_ids"]),3)
        self.assertEqual(len(capture["snapshots"]),2)
        end=capture["termination"]
        self.assertEqual((end["status"],end["reason"],end["next_original_ordinal"]),
            ("UNAVAILABLE_SUFFIX","NATIVE_ACTION_UNAVAILABLE",2))
        self.assertEqual(end["unavailable_boundary_records"],1)
        closed=json.loads((directory/"bridge-close.json").read_bytes())
        self.assertTrue(closed["normal_close"]); self.assertTrue(closed["reaped"])
        self.assertFalse(terminal["result"]["branch_transition_validated"])
        self.assertFalse(terminal["result"]["independent_terminal_truth"])
        with self.assertRaises(FileExistsError): self.run_job(job)

    def test_early_terminal_is_uncertain_suffix_not_a_scored_label_or_strict_v1_success(self):
        job=self.job(nonterminal=False); directory,terminal=self.run_job(job)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        capture=terminal["result"]["capture"]
        self.assertEqual(capture["termination"]["reason"],"REFERENCE_TERMINAL")
        self.assertEqual(self.reader(job,directory)(capture),digest(capture))
        strict=self.job(nonterminal=False,strict=True)
        _,failed=self.run_job(strict)
        self.assertEqual(failed["supervisor_exit_code"],1)
        self.assertIsNone(failed["result"])

    def test_full_path_and_subsequent_coordinate_are_not_blocked_by_unavailable_prefix(self):
        first=self.job(); _,partial=self.run_job(first)
        self.assertEqual(partial["supervisor_exit_code"],0)
        second=self.job(transitions=1,disable=False)
        directory,complete=self.run_job(second)
        self.assertEqual(complete["supervisor_exit_code"],0,complete)
        self.assertEqual(complete["result"]["capture"]["termination"]["status"],"FULL_PATH_OBSERVED")
        self.assertEqual(self.reader(second,directory)(complete["result"]["capture"]),digest(complete["result"]["capture"]))

    def test_executable_shortening_and_changed_stop_witness_refuse(self):
        job=self.job(); directory,terminal=self.run_job(job)
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        capture=terminal["result"]["capture"]
        for key,value in (("reason","REFERENCE_TERMINAL"),("next_original_ordinal",1),
                ("last_actual_snapshot_sha256","f"*64),("unavailable_boundary_records",0)):
            bad=deepcopy(capture); bad["termination"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                read_native_prefix_transport(bad,process_records=bad["records"],raw_root=job.raw_root,
                    clone_binding=job.clone_binding,request_ids=job.request_ids,native_boundaries=job.native_boundaries,
                    party_mapping=job.party_mapping)
        current=json.loads(capture["snapshots"][-1]["currentSnapshotJson"])
        future=deepcopy(job.native_boundaries)
        projected=project_showdown_boundary(current,job.party_mapping)
        future[2]["native"]["action_identity"]={s:projected["command_catalogs"][s][0]["choice"] for s in ("p1","p2")}
        with self.assertRaisesRegex(ValueError,"executable next"):
            read_native_prefix_transport(capture,process_records=capture["records"],raw_root=job.raw_root,
                clone_binding=job.clone_binding,request_ids=job.request_ids,native_boundaries=future,party_mapping=job.party_mapping)

    def test_malformed_or_ambiguous_request_cannot_hide_behind_withholding_flag(self):
        actions=dict(p1=dict(kind="move",slot=0,engine_id="seismictoss"),p2=dict(kind="move",slot=0,engine_id="splash"))
        for mode in ("flag","row","ambiguous","missing","mapping","wait","preview"):
            root=showdown.snapshot(); mapping=showdown.mapping(root)
            active=root["boundaryRequests"]["p1"]["active"][0]; active["maybeTrapped"]=True
            if mode=="flag":active["maybeTrapped"]="yes"
            if mode=="row":active["moves"]=[None]
            if mode=="ambiguous":root["battle"]["sides"][0]["pokemon"][0]["moveSlots"].append(dict(id="seismictoss",pp=1,disabled=False))
            if mode=="missing":root["boundaryRequests"].pop("p2")
            if mode=="mapping":mapping["native_to_original"]["p1"]=[0,0]
            if mode=="wait":root["boundaryRequests"]["p1"]["wait"]=1
            if mode=="preview":root["boundaryRequests"]["p1"]["teamPreview"]="yes"
            with self.subTest(mode=mode):
                try: commands_for_native_actions(root,mapping,actions)
                except NativeReplayUnavailable: self.fail("malformed input became normal unavailability")
                except ValueError: pass
                else: self.fail("malformed input accepted")
        root=showdown.snapshot(); mapping=showdown.mapping(root)
        root["boundaryRequests"]["p1"]["active"][0]["maybeTrapped"]=True
        with self.assertRaises(NativeReplayUnavailable) as caught:commands_for_native_actions(root,mapping,actions)
        self.assertEqual(caught.exception.reason,"REQUEST_LEGALITY_UNAVAILABLE")

    def test_close_failure_cannot_be_downgraded_to_valid_unavailability(self):
        job=self.job(mode="missing-close"); _,terminal=self.run_job(job)
        self.assertEqual(terminal["supervisor_exit_code"],1)
        self.assertIsNone(terminal["result"])
