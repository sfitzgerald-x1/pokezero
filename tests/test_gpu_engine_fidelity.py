"""Authored trace/semantic-coupling checks, not the fixed scientific roster."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
# Native bridge fixtures import the src-layout API before fidelity itself.
# Do not depend on a caller's PYTHONPATH or another test's import side effect.
sys.path.insert(0,str(ROOT/'src'))
NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated spike JAX')
class TraceTests(unittest.TestCase):
    def test_faint_marker_is_not_a_major_status_but_hp_disagreement_remains(self):
        from unittest.mock import patch
        import jax.numpy as j
        from gpu_engine.fidelity import compare
        from gpu_engine.state import empty_state
        s=empty_state(1)._replace(hp=j.full((1,2,6),100).at[0,0,0].set(0),
            status=j.zeros((1,2,6),j.int32).at[0,0,0].set(2))
        parties=[[dict(hp=100,status='',fainted=False) for _ in range(6)] for _ in range(2)]
        parties[0][0]=dict(hp=0,status='fnt',fainted=True)
        with patch('gpu_engine.fidelity.native_view',return_value=({},parties,[0,0])):
            self.assertEqual(compare(s,None,None,None),[])
            parties[0][0]=dict(hp=1,status='',fainted=False)
            diff=compare(s,None,None,None)
            self.assertTrue(any(x['field']=='0:0:hp' for x in diff))
            self.assertTrue(any(x['field']=='0:0:fainted' for x in diff))

    def test_final_row_harness_failure_cannot_exit_zero(self):
        from gpu_engine.fidelity import protocol_completed
        rows=[dict(status='STATE_MISMATCH') for _ in range(200)]
        self.assertTrue(protocol_completed(rows));self.assertFalse(protocol_completed(rows[:-1]))
        rows[-1]['status']='HARNESS_ERROR'
        self.assertFalse(protocol_completed(rows))

    def test_requested_seats_do_not_inherit_the_reference_wait_mask(self):
        import jax.numpy as j
        from gpu_engine.fidelity import requested_seats
        from gpu_engine.state import empty_state
        s=empty_state(1)._replace(hp=j.full((1,2,6),100),done=j.array([False]))
        self.assertEqual(requested_seats(s).tolist(),[True,True])
        s=s._replace(forced_switch=j.array([[True,False]]))
        self.assertEqual(requested_seats(s).tolist(),[True,False])
        self.assertEqual(requested_seats(s._replace(done=j.array([True]))).tolist(),[False,False])

    def test_terminal_agreement_and_completed_twentieth_turn_required(self):
        import jax.numpy as j
        from gpu_engine.fidelity import completion
        from gpu_engine.state import empty_state
        from pokezero.env import TerminalState
        s=empty_state(1)._replace(hp=j.full((1,2,6),100),done=j.array([False]),turn=j.array([20]))
        self.assertEqual(completion(s,None)[0],'MATCHED_THROUGH_TURN20')
        self.assertIsNone(completion(s._replace(pending_residual=j.array([True])),None)[0])
        self.assertIsNone(completion(s._replace(forced_switch=j.array([[True,False]])),None)[0])
        self.assertEqual(completion(s,TerminalState('p1',20))[0],'TERMINAL_MISMATCH')
        s=s._replace(done=j.array([True]),winner=j.array([0]))
        self.assertEqual(completion(s,TerminalState('p2',20))[0],'TERMINAL_MISMATCH')
        self.assertEqual(completion(s,TerminalState('p1',20))[0],'MATCHED_TERMINAL')

    def test_raw_tape_preserves_saved_chance_across_deferred_requests(self):
        import jax.numpy as j
        from gpu_engine.fidelity import chance_tape
        from gpu_engine.state import empty_state
        s=empty_state(1,seed=19)
        a=chance_tape(s,50000,0)
        s=s._replace(pending_residual=j.array([True]),residual_draws=j.asarray([a['draws']]),
            secondary_draws=j.asarray([a['secondary']]),sleep_talk_draws=j.asarray([a['sleep_talk']]))
        b=chance_tape(s,50000,1)
        self.assertEqual(a['draws'],b['draws']);self.assertEqual(a['secondary'],b['secondary'])
        self.assertEqual(a['sleep_talk'],b['sleep_talk']);self.assertNotEqual(a['nonce'],b['nonce'])

    def test_hidden_power_move_identity_is_not_alias_slot_identity(self):
        from gpu_engine.fidelity import canonical,toid
        self.assertEqual(canonical('hiddenpowergrass'),'hiddenpower')
        self.assertEqual(toid('Deoxys-Attack'),'deoxysattack')


@unittest.skipUnless(NODE and SHOWDOWN,'requires pinned compiled Showdown')
class NativeChanceTests(unittest.TestCase):
    def test_switch_indices_are_normalized_back_to_original_party(self):
        import random
        from types import SimpleNamespace
        import jax.numpy as j
        from gpu_engine.fidelity import reference_actions,toid
        from gpu_engine.state import empty_state
        from pokezero.env import BattleStartOverride
        from pokezero.local_showdown import LocalShowdownEnv,LocalShowdownConfig
        from pokezero.observation import ObservationFeatureMasks
        names=['Mew','Pikachu','Blissey','Snorlax','Venusaur','Magikarp']
        code="const {Teams}=require(process.argv[1]+'/dist/sim/teams.js');process.stdout.write(Teams.pack(JSON.parse(process.argv[2]).map(species=>({species,ability:'Run Away',moves:['thunderbolt'],level:100}))));"
        team=subprocess.check_output([NODE,'-e',code,SHOWDOWN,json.dumps(names)],text=True)
        env=LocalShowdownEnv(LocalShowdownConfig(showdown_root=SHOWDOWN,node_binary=NODE,
            feature_masks=ObservationFeatureMasks(transition_token_budget=0)))
        s=empty_state(1)._replace(hp=j.full((1,2,6),100),done=j.array([False]),
            moves=j.zeros((1,2,6,4),j.int32).at[...,0].set(1),pp=j.zeros((1,2,6,4),j.int32).at[...,0].set(24))
        reference_actions.table=SimpleNamespace(combat=None)
        original=[[toid(x) for x in names]]*2;vocab={'moves':['_none','thunderbolt']}
        try:
            env.reset_with_start_override(seed=17,start_override=BattleStartOverride(player_teams={'p1':team,'p2':team}))
            _,_,diff,maps=reference_actions(env,s,env.snapshot(),original,vocab,random.Random(1))
            self.assertEqual(diff,[]);self.assertEqual(maps['p1'][5],5)
            from unittest.mock import patch
            # A native unilateral boundary cannot pass merely because GPU
            # believes turn20 is complete; final comparison consumes no RNG.
            policy=random.Random(1);before=policy.getstate()
            with patch.object(env,'requested_players',return_value=('p1',)):
                _,_,diff,_=reference_actions(env,s._replace(turn=j.array([20])),env.snapshot(),original,vocab,policy,choose=False)
            self.assertTrue(any(x['field']=='requested_players' for x in diff))
            self.assertEqual(policy.getstate(),before)
            env.step({'p1':5,'p2':0})
            s=s._replace(active=j.array([[2,0]],j.int32))
            _,_,diff,maps=reference_actions(env,s,env.snapshot(),original,vocab,random.Random(1))
            self.assertEqual(diff,[])
            self.assertEqual(maps['p1'][4],5);self.assertEqual(maps['p1'][5],4)
            # An unsupported native slot change is scientific nonmatch, not
            # an infrastructure error that aborts all remaining fixed games.
            moved=s._replace(moves=s.moves.at[...,0].set(2))
            _,_,diff,_=reference_actions(env,moved,env.snapshot(),original,
                {'moves':['_none','thunderbolt','tackle']},random.Random(1))
            self.assertTrue(any(x['field']=='p1:action_identity' for x in diff))
        finally:env.close()

    def test_stock_local_bridge_uses_owned_launcher_and_records_coupled_roles(self):
        from pokezero.env import BattleStartOverride
        from pokezero.local_showdown import LocalShowdownEnv,LocalShowdownConfig
        from pokezero.observation import ObservationFeatureMasks
        code="const {Teams}=require(process.argv[1]+'/dist/sim/teams.js');process.stdout.write(JSON.stringify(['Mew','Pikachu'].map(species=>Teams.pack([{species,ability:'Run Away',moves:['thunderbolt'],level:100}]))));"
        packed=json.loads(subprocess.check_output([NODE,'-e',code,SHOWDOWN],text=True))
        with tempfile.TemporaryDirectory(prefix='pokezero-fidelity-fixture-') as temp:
            temp=Path(temp);bridge=temp/'bridge.mjs'
            bridge.write_text('await import('+json.dumps((ROOT/'gpu_engine/fidelity_bridge.mjs').as_uri())+');\n')
            env=LocalShowdownEnv(LocalShowdownConfig(showdown_root=SHOWDOWN,node_binary=NODE,bridge_path=bridge,
                feature_masks=ObservationFeatureMasks(transition_token_budget=0)))
            try:
                (temp/'initializing.json').write_text(json.dumps(dict(allow_inert_gender=True)))
                env.reset_with_start_override(seed=17,start_override=BattleStartOverride(player_teams=dict(p1=packed[0],p2=packed[1])))
                (temp/'initializing.json').unlink()
                initial=[json.loads(line) for line in (temp/'chance-audit.jsonl').read_text().splitlines()]
                self.assertTrue(any(e['kind']=='INERT_GENDER' for e in initial))
                self.assertFalse(any(e['kind']=='INITIAL_RANDOM' for e in initial),initial)
                (temp/'chance-audit.jsonl').unlink()
                c=dict(schema='pokezero.gpu_chance_tape.v1',nonce='fixture:0',
                    draws=[[0,99,99,0,15,15,0,0,0,0,0,0]]*2,secondary=[99,99],sleep_talk=[0,0])
                (temp/'chance.json').write_text(json.dumps(c));env.step({'p1':0,'p2':0})
                (temp/'chance.json').unlink()
                events=[json.loads(line) for line in (temp/'chance-audit.jsonl').read_text().splitlines()]
                self.assertFalse(any(e['kind']=='UNRESOLVED' for e in events))
                self.assertEqual(sum(e.get('role')=='damage' for e in events),2)
            finally:env.close()

    def test_native_raw_damage_accuracy_critical_and_secondary_roles(self):
        code=r'''
const path=require('node:path'),assert=require('node:assert/strict');
const {Battle}=require(path.join(process.argv[1],'dist/sim/battle.js'));
const {PRNG}=require(path.join(process.argv[1],'dist/sim/prng.js'));
let armed=false;const events=[];
const c={schema:'pokezero.gpu_chance_tape.v1',nonce:'authored:0',
draws:[[0,99,99,0,15,15,0,0,0,0,0,0],[0,99,99,0,15,15,0,0,0,0,0,0]],secondary:[99,99],sleep_talk:[0,0]};
require(process.argv[2]).install(Battle,PRNG,()=>armed?c:null,row=>events.push(row));
const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
for(let i=1;i<=2;i++)b.setPlayer('p'+i,{name:'s'+i,team:[{species:'Mew',ability:'Run Away',moves:['thunderbolt'],level:100}]});
b.p1.active[0].storedStats.spe=300;b.p2.active[0].storedStats.spe=100;
armed=true;b.makeChoices('move 1','move 1');
assert.equal(b.p1.active[0].hp,260);assert.equal(b.p2.active[0].hp,260);
for(const role of ['accuracy','critical','damage','secondary'])assert.equal(events.filter(x=>x.role===role).length,2);
assert(!events.some(x=>x.kind==='UNRESOLVED'));
assert.throws(()=>b.prng.random(),/SPIKE_UNRESOLVED:UNMAPPED_RAW_RNG/);
armed=false;b.destroy();process.stdout.write(JSON.stringify(events));
'''
        result=subprocess.run([NODE,'-e',code,SHOWDOWN,str(ROOT/'gpu_engine/chance_hooks.cjs')],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(result.stdout))
