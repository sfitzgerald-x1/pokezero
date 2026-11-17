"""The v6 union retains entry coins, construction genders and the fifth effect column."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root

ROOT = Path(__file__).resolve().parents[1]
NODE = os.environ.get('NODE') or shutil.which('node')

SCRIPT = r'''
const assert=require('node:assert/strict'),path=require('node:path');
const root=process.argv[1],hooks=process.argv[2];
const {Battle}=require(path.join(root,'dist/sim/battle.js'));
const {PRNG}=require(path.join(root,'dist/sim/prng.js'));
let tape=null,events=[],nonce=0;
require(hooks).install(Battle,PRNG,()=>tape,e=>events.push(e),()=>null,path.join(root,'dist/sim/pokemon.js'));
const row=()=>[0,99,99,99,15,0,0,0,0,0,0,0];
function T(version){
 const t={schema:'pokezero.gpu_chance_tape.v'+version,nonce:String(++nonce),draws:[row(),row()],
  secondary:[99,99],sleep_talk:[0,0],phazing:[0,0],
  effect:Array.from({length:2},()=>version>=5?[15,0,0,0,1]:[15,0,0,0])};
 if(version===4||version===6){t.entry_ties=[0,0];t.resuming=false;}
 return t;
}
const mon=(species,ability,gender,moves)=>({species,ability,gender,moves,level:100,item:'',
 evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0},ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31}});
function battle(side=0,same=false,special=false){
 tape=null;
 const attacker=mon('Snorlax','Thick Fat','M',[special?'psychic':'tackle']);
 const holder=mon('Clefable','Cute Charm',same?'M':'F',['splash']);
 const teams=[[attacker],[holder,mon('Clefable','Cute Charm','F',['splash'])]];
 if(side)teams.reverse();
 const b=new Battle({formatid:'gen3customgame@@@sleepclausemod',seed:[1,2,3,4]});
 b.setPlayer('p1',{name:'p1',team:teams[0]});b.setPlayer('p2',{name:'p2',team:teams[1]});return b;
}
function turn(b,t){
 events=[];tape=t;let error=null;const first=b.log.length;
 try{b.makeChoices('move 1','move 1');}catch(e){error=e.message;}
 finally{tape=null;}
 return {error,events:events.slice(),blocked:b.log.slice(first).some(x=>x.includes('|cant|')&&x.includes('|Attract'))};
}
const coupled=(r,role)=>r.events.filter(e=>e.kind==='COUPLED'&&e.role===role);
const out={contact:[],infatuation:[],legacy:[],noncontact:[],invalid_context:[],repeated:[]};
for(const side of [0,1])for(const same of [false,true])for(const roll of [19,20]){
 const b=battle(side,same),t=T(6);t.phazing[side]=roll;
 const r=turn(b,t);out.contact.push({side,same,roll,...r,attracted:!!b.sides[side].active[0].volatiles.attract});
}
for(const version of [5,6])for(const side of [0,1])for(const bit of [0,1]){
 const b=battle(side);const first=turn(b,T(version));assert.equal(first.error,null);
 assert(b.sides[side].active[0].volatiles.attract);
 const t=T(version);t.effect[side][4]=bit;t.effect[1-side][4]=1-bit;
 const r=turn(b,t);out.infatuation.push({version,side,bit,...r});
}
for(const version of [3,4]){
 const b=battle();assert.equal(turn(b,T(version)).error,null);
 const r=turn(b,T(version));out.legacy.push({version,...r});
}
for(const side of [0,1]){
 const b=battle(side,false,true);out.noncontact.push({side,...turn(b,T(6)),attracted:!!b.sides[side].active[0].volatiles.attract});
 const wrong=battle(side),attacker=wrong.sides[side].active[0],bench=wrong.sides[1-side].pokemon[1];
 wrong.activePokemon=attacker;wrong.activeMove=wrong.dex.getActiveMove('tackle');tape=T(6);events=[];let error=null;
 // The active holder executes its real callback, but the mismatched state names a bench holder.
 try{wrong.singleEvent('DamagingHit',wrong.sides[1-side].active[0].getAbility(),bench.abilityState,
   wrong.sides[1-side].active[0],attacker,wrong.activeMove,1);}
 catch(e){error=e.message;}finally{tape=null;}
 out.invalid_context.push({side,error,events:events.slice()});
 const repeated=battle(side),a=repeated.sides[side].active[0],h=repeated.sides[1-side].active[0];
 repeated.activePokemon=a;tape=T(6);events=[];error=null;
 try{
  repeated.activeMove=repeated.dex.getActiveMove('whirlwind');repeated.getRandomSwitchable(repeated.sides[1-side]);
  repeated.activeMove=repeated.dex.getActiveMove('tackle');
  repeated.singleEvent('DamagingHit',h.getAbility(),h.abilityState,h,a,repeated.activeMove,1);
 }catch(e){error=e.message;}finally{tape=null;}
 out.repeated.push({side,error,events:events.slice()});
}
process.stdout.write(JSON.stringify(out));
'''


@requires_showdown()
@unittest.skipUnless(NODE, 'requires node')
class TapeV6ContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = json.loads(subprocess.check_output(
            [NODE, '-e', SCRIPT, str(showdown_root()), str(ROOT/'gpu_engine/chance_hooks.cjs')], text=True))

    def test_cute_charm_threshold_seat_and_gender_law(self):
        self.assertEqual(len(self.out['contact']), 8)
        for r in self.out['contact']:
            with self.subTest(side=r['side'], same=r['same'], roll=r['roll']):
                self.assertIsNone(r['error'], r)
                self.assertEqual(r['attracted'], not r['same'] and r['roll'] < 20)
                rows = [e for e in r['events'] if e['kind']=='COUPLED' and e['role']=='contact_cutecharm']
                self.assertEqual([(e['side'], e['raw']) for e in rows], [(r['side'], r['roll'])])

    def test_v5_and_v6_infatuation_read_only_the_actors_fifth_column(self):
        self.assertEqual(len(self.out['infatuation']), 8)
        for r in self.out['infatuation']:
            with self.subTest(version=r['version'], side=r['side'], bit=r['bit']):
                self.assertIsNone(r['error'], r)
                self.assertEqual(r['blocked'], r['bit']==0)
                rows = [e for e in r['events'] if e['kind']=='COUPLED' and e['role']=='infatuation']
                self.assertEqual([(e['side'], e['raw']) for e in rows], [(r['side'], r['bit'])])

    def test_legacy_four_column_tapes_refuse_infatuation(self):
        for r in self.out['legacy']:
            with self.subTest(version=r['version']):
                self.assertIn('SPIKE_UNRESOLVED:', r['error'])
                self.assertTrue(any(e['kind']=='UNRESOLVED' for e in r['events']))
                self.assertFalse(any(e['kind']=='COUPLED' and e['role']=='infatuation' for e in r['events']))

    def test_noncontact_and_wrong_holder_do_not_inherit_contact_role(self):
        for r in self.out['noncontact']:
            self.assertIsNone(r['error'], r)
            self.assertFalse(r['attracted'])
            self.assertFalse(any(e.get('role')=='contact_cutecharm' for e in r['events']))
        for r in self.out['invalid_context']:
            self.assertIsNotNone(r['error'], r)
            self.assertIn('SPIKE_UNRESOLVED:', r['error'])
            self.assertFalse(any(e['kind']=='COUPLED' and e['role']=='contact_cutecharm' for e in r['events']))

    def test_drag_and_contact_cannot_reuse_the_auxiliary_coordinate(self):
        for r in self.out['repeated']:
            self.assertIn('SPIKE_UNRESOLVED:REPEATED_ROLE', r['error'])


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires JAX')
class InitialChanceUnionTests(unittest.TestCase):
    def test_panel_initializer_keeps_retained_genders_and_initial_coin(self):
        import jax.numpy as j
        import numpy as np
        from gpu_engine.fidelity import initial_chance
        from gpu_engine.state import empty_state
        from gpu_engine.transition import sample_chance_inputs
        genders=j.array([[[0,1,2,1,2,0],[2,1,0,2,1,0]]], dtype=j.int8)
        for seed in (1, 19, 719, 50200):
            s=empty_state(1,seed=seed)._replace(gender=genders)
            before=np.asarray(s.keys).copy()
            for allow in (False, True):
                # This is exactly the public helper used by fidelity_panel.run.
                payload=initial_chance(s,allow_inert_gender=allow)
                self.assertEqual(payload['genders'],np.asarray(genders)[0].tolist())
                self.assertFalse(payload['allow_inert_gender'])
                self.assertEqual(payload['initial_tie'],int(sample_chance_inputs(s).initial_tie[0]))
                np.testing.assert_array_equal(s.keys,before)

    def test_unknown_legacy_gender_preserves_initializer_api(self):
        from gpu_engine.fidelity import initial_chance
        from gpu_engine.state import empty_state
        for allow in (False, True):
            payload=initial_chance(empty_state(1,seed=19),allow_inert_gender=allow)
            self.assertNotIn('genders',payload)
            self.assertEqual(payload['allow_inert_gender'],allow)
            self.assertIn(payload['initial_tie'],(0,1))

    def test_partially_known_or_malformed_genders_refuse_initialization(self):
        import jax.numpy as j
        from gpu_engine.fidelity import initial_chance
        from gpu_engine.state import empty_state
        s=empty_state(1,seed=19)
        for value in (-1,3):
            genders=j.ones((1,2,6),dtype=j.int8).at[0,1,5].set(value)
            for allow in (False,True):
                with self.subTest(value=value,allow=allow),self.assertRaises(ValueError):
                    initial_chance(s._replace(gender=genders),allow_inert_gender=allow)
        with self.assertRaises(ValueError):
            initial_chance(s._replace(gender=j.ones((1,2,5),dtype=j.int8)),allow_inert_gender=True)


if __name__=='__main__':
    unittest.main()
