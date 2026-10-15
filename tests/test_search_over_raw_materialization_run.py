"""Real Python constructor and POSIX owner, disposable authored JSON bridge.

The child is NOT battle_bridge.mjs or Showdown: no engine/model/game is run.
The core materializer transport tests separately exercise Python request laws.
"""
from copy import deepcopy
from dataclasses import dataclass,replace
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from pokezero.local_showdown import LocalShowdownConfig,PublicBattleMaterializationState
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_materialization_run import (
    OwnedMaterializationJob,MaterializationRunValidator,MaterializationRunContract,supervise_materialization_run,
)
from pokezero.mcts_eval.search_over_raw_outcome_run import outcome_runtime_binding
from test_search_over_raw_outcomes import EntrypointSyntheticEnv
from test_search_over_raw_materialization import fixture,BINDING
from test_engine_world import _override,_payload,_dex

BRIDGE = """
import readline from 'node:readline';
import fs from 'node:fs';
import crypto from 'node:crypto';
const root=process.argv[process.argv.indexOf('--showdown-root')+1];
const fixture=JSON.parse(fs.readFileSync(root+'/fixture.json','utf8'));
let owned, current, snapshotIndex=0, generation=0;
const hash=value=>crypto.createHash('sha256').update(value).digest('hex');
function send(value) { process.stdout.write(JSON.stringify(value)+'\\n'); }
const lines=readline.createInterface({input:process.stdin});
lines.on('line',line=>{
 const c=JSON.parse(line),b=c.battleId;
 if(c.type==='start') { send({type:'started',battleId:b,formatid:c.formatid,seed:c.seed});
   send({type:'ready',battleId:b,requested:['p1','p2']}); }
 else if(c.type==='materialize') {
   if(fixture.mode==='hang') return;
   if(fixture.mode==='failure') { send({type:'error',battleId:b,message:'synthetic refusal'}); return; }
   send({type:'materialized',battleId:b,boundaryRequests:fixture.snapshot.boundaryRequests,requested:['p1','p2']}); }
 else if(c.type==='snapshot') send({type:'snapshot',battleId:b,snapshot:fixture.snapshot});
 else if(c.type==='end') { for(const stream of ['omniscient','p1','p2']) send({type:'stream_end',battleId:b,stream});
   send({type:'ended',battleId:b}); }
 else if(c.type==='clone_snapshot') {
   owned={battleId:b,cloneId:c.cloneId,snapshotJsonSha256:hash(c.snapshotJson)};
   current=JSON.parse(c.snapshotJson);
   send({type:'cloned',...owned,requested:['p1','p2']}); }
 else if(c.type==='snapshot_owned_boundary') {
   const text=JSON.stringify(current);
   send({type:'owned_boundary_snapshot',...owned,requestId:c.requestId,snapshotIndex:snapshotIndex++,
     boundaryGeneration:generation,currentSnapshotJson:text,currentSnapshotJsonSha256:hash(text)}); }
 else if(c.type==='choices') {
   for(const [player,choice] of Object.entries(c.choices)) send({type:'choice_ack',battleId:b,player,choice});
   generation++;
   if(fixture.nonterminalReplay) {
     current.battle.log.push('|turn|2');
     for(const seat of ['p1','p2']) current.boundaryRequests[seat].active[0].moves[0].pp--;
     if(fixture.disableNativeActionAfter) current.boundaryRequests.p1.active[0].moves[0].disabled=true;
     send({type:'ready',battleId:b,requested:['p1','p2']});
   } else {
     current.battle.ended=true;current.boundaryRequests={};
     send({type:'terminal',battleId:b}); }
 }
 else if(c.type==='release_clone') {
   for(const stream of ['omniscient','p1','p2']) send({type:'stream_end',battleId:b,stream});
   send({type:'clone_released',...owned}); }
 else if(c.type==='close') {
   if(fixture.mode==='missing-close') { send({type:'error',message:'missing synthetic close'}); return; }
   send({type:'closed'});lines.close();process.stdin.destroy();process.exitCode=fixture.mode==='nonzero'?7:0; }
 else send({type:'error',battleId:b,message:'unknown synthetic command'});
});
"""


class SyntheticEnv(EntrypointSyntheticEnv):
    def _sync_incremental_state(self):pass


@dataclass(frozen=True)
class SyntheticMaterializationJob(OwnedMaterializationJob):
    synthetic_process_fixture=True

    def input_binding(self):
        with patch("pokezero.local_showdown._public_materialization_payload",return_value=_payload(_dex())):
            return super().input_binding()

    def _components(self):return SyntheticEnv(self.config)

    def __call__(self,ownership,phase):
        with patch("pokezero.local_showdown._public_materialization_payload",return_value=_payload(_dex())):
            return super().__call__(ownership,phase)


class MaterializationRunTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="pokezero-materialization-fixture-")
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.directory=self.root/"attempt"
        bridge_dir=self.root/"scripts";bridge_dir.mkdir()
        self.bridge=bridge_dir/"fixture.mjs";self.bridge.write_text(BRIDGE)
        repo=Path(__file__).resolve().parents[1]
        for name in ("battle_bridge_owned_continuation.mjs","battle_bridge_owned_snapshot.mjs"):
            (bridge_dir/name).write_bytes((repo/"scripts"/name).read_bytes())
        sim=self.root/"showdown/dist/sim";sim.mkdir(parents=True)
        for name in ("index.js","state.js"):(sim/name).write_text("// Never loaded synthetic dependency placeholder.\n")
        self.checkpoint=self.root/"not-a-model";self.checkpoint.write_bytes(b"never loaded")
        self.config=LocalShowdownConfig(showdown_root=str(self.root/"showdown"),bridge_path=str(self.bridge),
            set_belief_source=False,read_timeout_seconds=2.)
        env,state,_,_=fixture()
        self.state=state;env._process=None
        # Use the same fully authored raw root as the constructor transport fixture.
        from test_search_over_raw_construction import map_fixture
        self.snapshot=map_fixture()[0]
        self.snapshot["boundaryRequests"]={seat:dict(active=[dict(moves=[dict(id="surf",pp=5)])],
            side=dict(id=seat,pokemon=[])) for seat in ("p1","p2")}
        self.fixture_path=self.root/"showdown/fixture.json"
        self.configure("complete")

    def configure(self,mode):
        self.fixture_path.write_text(json.dumps(dict(mode=mode,snapshot=self.snapshot)))
        self.runtime=outcome_runtime_binding(self.config,checkpoint_path=self.checkpoint,extra_files=[self.fixture_path])

    def job(self):return SyntheticMaterializationJob(self.config,self.state,_override(),17,deepcopy(BINDING),deepcopy(self.runtime))
    def run_job(self,job=None,validate_result=lambda _:True,**kwargs):
        job=self.job() if job is None else job
        receipt=supervise_materialization_run(job,directory=self.directory,binding_sha256="a"*64,
            verify_admission=lambda:True,validate_result=validate_result,**kwargs)
        return job,receipt
    def reader(self,job,allow_synthetic=True,expected=None):
        return MaterializationRunValidator(directory=self.directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64,expected_inputs=job.input_binding() if expected is None else expected,
            runtime=self.runtime,allow_synthetic=allow_synthetic)
    def load(self,name):return json.loads((self.directory/name).read_bytes())
    def save(self,name,value):(self.directory/name).write_text(json.dumps(value))

    def test_actual_owned_constructor_and_distinct_independent_reader(self):
        job,receipt=self.run_job()
        self.assertEqual(receipt["status"],"MATERIALIZATION_RUN_COMPLETE",receipt)
        result=receipt["result"];capture=result["capture"]
        self.assertEqual(self.reader(job)(capture),digest(capture))
        self.assertIsNone(self.reader(job,False)(capture))
        self.assertEqual(result["produced_bridge_snapshot"],self.snapshot)
        self.assertIn("materialization_wall_seconds",receipt["phase_costs"])
        closed=self.load("bridge-close.json")
        self.assertEqual(closed["exit_code"],0);self.assertTrue(closed["reaped"])
        self.assertTrue(closed["normal_close"]);self.assertTrue(closed["pumps_drained"])
        self.assertFalse(closed["forced_cleanup"])
        self.assertEqual(closed["schema"],"pokezero.search-over-raw.materialization-bridge-close.v1")
        trace=self.load("bridge-transport.json")["records"]
        self.assertEqual([v["payload"]["type"] for v in trace if v["kind"]=="command"],
            ["start","materialize","snapshot","end","close"])
        text=json.dumps(trace)
        for private in ('"team":','"publicState":','"boundaryRequests":','"snapshot":'):
            self.assertNotIn(private,text)
        self.assertFalse(result["construction_provenance_validated"])
        self.assertFalse(result["full_world_equivalence"])
        self.assertFalse(result["scientific_collection_authorized"])
        with self.assertRaises(FileExistsError):self.run_job(job)

    def test_refusal_missing_close_and_actual_nonzero_exit_cannot_clear_or_retry(self):
        for mode in ("failure","missing-close","nonzero"):
            self.directory=self.root/mode;self.configure(mode);job,receipt=self.run_job()
            with self.subTest(mode=mode):
                self.assertEqual(receipt["status"],"OPERATIONAL_STOP",receipt)
                self.assertEqual(receipt["worker_exit_code"],1);self.assertIsNone(receipt["result"])
                closed=self.load("bridge-close.json")
                self.assertTrue(closed["reaped"]);self.assertFalse(closed["normal_close"])
                if mode=="nonzero":self.assertEqual(closed["exit_code"],7)
                with self.assertRaises(FileExistsError):self.run_job(job)

    def test_watchdog_remains_failure_not_exit_zero_inference(self):
        self.configure("hang");job,receipt=self.run_job(_synthetic_watchdog_seconds=.5)
        self.assertEqual(receipt["status"],"MATERIALIZATION_RUN_INCOMPLETE",receipt)
        self.assertEqual(receipt["supervisor_exit_code"],124);self.assertIsNone(receipt["result"])

    def test_reader_rejects_receipt_drift_and_rehashed_substitute_root(self):
        job,receipt=self.run_job();capture=receipt["result"]["capture"]
        cases=(("claim.json","contract",dict(MaterializationRunContract().receipt(),max_roots=2)),
            ("launch.json","pid",False),("terminal.json","status","BOUNDARY_RUN_COMPLETE"),
            ("terminal.json","worker_exit_code",False),("bridge-close.json","exit_code",False),
            ("bridge-close.json","forced_cleanup",True),("bridge-close.json","pid",1),
            ("bridge-close.json","wait_ownership_lost",True),("bridge-close.json","pumps_drained",False),
            ("bridge-launch.json","schema","pokezero.search-over-raw.boundary-bridge-launch.v1"))
        for name,key,value in cases:
            saved=self.load(name);self.save(name,dict(saved,**{key:value}))
            with self.subTest(name=name,key=key),self.assertRaises(ValueError):self.reader(job)(capture)
            self.save(name,saved)
        terminal=self.load("terminal.json");worker=self.load("worker-result.json")
        substituted=deepcopy(terminal["result"]);substituted["produced_bridge_snapshot"]["battle"]["turn"]=99
        substituted["produced_bridge_snapshot_sha256"]=digest(substituted["produced_bridge_snapshot"])
        self.save("terminal.json",dict(terminal,result=substituted));self.save("worker-result.json",dict(worker,result=substituted))
        with self.assertRaises(ValueError):self.reader(job)(capture)
        self.save("terminal.json",terminal);self.save("worker-result.json",worker)
        self.assertEqual(self.reader(job)(capture),digest(capture))

    def test_foreign_inputs_and_mutating_acceptance_cannot_publish_completion(self):
        job,receipt=self.run_job();capture=receipt["result"]["capture"]
        expected=job.input_binding();expected["materialization_inputs"]["seed"]+=1
        with self.assertRaises(ValueError):self.reader(job,expected=expected)
        self.directory=self.root/"mutating-acceptance"
        def mutate(result):result["produced_bridge_snapshot"]["battle"]["turn"]=99;return True
        _,receipt=self.run_job(validate_result=mutate)
        self.assertEqual(receipt["worker_exit_code"],0)
        self.assertEqual(receipt["status"],"OPERATIONAL_STOP");self.assertIsNone(receipt["result"])

    def test_prospective_controller_binds_both_materialization_modules(self):
        from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes
        hashes=controller_input_hashes()
        directory=Path(__file__).resolve().parents[1]/"src/pokezero/mcts_eval"
        for name in ("search_over_raw_materialization.py","search_over_raw_materialization_run.py"):
            path=directory/name
            self.assertEqual(hashes[str(path)],hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=="__main__":unittest.main()
