"""Tape v3 (fidelity tier D): the contact abilities' chance roles on the phazing column, on real Showdown battles (no engine).

data/mods/gen3/abilities.ts at f76228a1: Static / Poison Point / Flame Body onDamagingHit randomChance(1,3) -> phazing[s] < 20;
Effect Spore randomChance(1,10) -> phazing[s] < 6, then sample(['slp','par','psn']) -> items[phazing[s] >> 1], and that
sleep's random(2,6) -> 2 + effect[s][2]. s is the attacker's seat.
"""
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
const path=require('path'),root=process.argv[1],hooks=process.argv[2];
const {Battle}=require(path.join(root,'dist/sim/battle.js'));
const {PRNG}=require(path.join(root,'dist/sim/prng.js'));
let tape=null;const events=[];
require(hooks).install(Battle,PRNG,()=>tape,e=>events.push(e),()=>null,path.join(root,'dist/sim/pokemon.js'));
const row=()=>[0,99,99,0,15,0,0,0,0,0,0,0];
function T(nonce,o={},schema='v3'){
  const t={schema:'pokezero.gpu_chance_tape.'+schema,nonce,draws:[row(),row()],secondary:[99,99],sleep_talk:[0,0]};
  if(schema!=='v1')t.effect=[[15,0,0,0],[15,0,0,0]];
  if(schema==='v3')t.phazing=[59,59];
  for(const [k,v] of Object.entries(o)){const [a,b,c]=k.split('.');if(c!==undefined)t[a][+b][+c]=v;else if(b!==undefined)t[a][+b]=v;else t[a]=v;}
  return t;
}
const Z={hp:0,atk:0,def:0,spa:0,spd:0,spe:0},I={hp:31,atk:31,def:31,spa:31,spd:31,spe:31};
const mon=(species,moves,ability)=>({species,moves,ability,item:'',level:100,evs:{...Z},ivs:{...I}});
function battle(p1,p2){
  const b=new Battle({formatid:'gen3customgame@@@sleepclausemod',seed:[1,2,3,4]});
  b.setPlayer('p1',{name:'p1',team:p1});b.setPlayer('p2',{name:'p2',team:p2});return b;
}
function turn(b,t,c1='move 1',c2='move 1'){
  tape=t;events.length=0;let error=null;
  try{b.makeChoices(c1,c2);}catch(e){error=e.message;}
  tape=null;return {error,events:events.slice(),log:b.log.slice(-30)};
}
function contact(holder,ability,o,schema){
  const b=battle([mon('Snorlax',['tackle'],'Thick Fat')],[mon(holder,['splash'],ability)]);
  const r=turn(b,T(holder+JSON.stringify(o)+schema,o,schema));
  return {...r,status:b.p1.active[0].status,time:b.p1.active[0].statusState.time??null};
}
const out={};
for(const [holder,ability] of [['Raichu','Static'],['Nidoking','Poison Point'],['Magcargo','Flame Body']])
  out[ability]=[19,20].map(v=>contact(holder,ability,{'phazing.0':v},'v3'));
out.spore=[0,1,2,3,4,5,6].map(v=>contact('Breloom','Effect Spore',{'phazing.0':v,'effect.0.2':3},'v3'));
out.static_v2=contact('Raichu','Static',{},'v2');
out.invalid={};
for(const [name,t] of Object.entries({
  v3_without_phazing:(()=>{const t=T('i1');delete t.phazing;return t;})(),
  v2_with_phazing:{...T('i2',{},'v2'),phazing:[0,0]},
  v3_phazing_out_of_range:T('i3',{'phazing.1':60}),
  v3_short_phazing:T('i4',{'phazing':[0]}),
})){const b=battle([mon('Snorlax',['splash'],'Thick Fat')],[mon('Blissey',['splash'],'Natural Cure')]);out.invalid[name]=turn(b,t).error;}
process.stdout.write(JSON.stringify(out));
'''


def coupled(result):
    return [(e['role'], e['side'], e['raw']) for e in result['events'] if e['kind'] == 'COUPLED']


@requires_showdown()
@unittest.skipUnless(NODE, 'requires node')
class ChanceHooksV3Tests(unittest.TestCase):
    schema = 'v3'
    @classmethod
    def setUpClass(cls):
        script = SCRIPT if cls.schema == 'v3' else SCRIPT.replace("'v3'", "'v4'").replace("t.phazing=[59,59];", "{t.phazing=[59,59];t.entry_ties=[0,0];t.resuming=false;}")
        raw = subprocess.check_output([NODE, '-e', script, str(showdown_root()), str(ROOT / 'gpu_engine/chance_hooks.cjs')], text=True)
        cls.out = json.loads(raw)

    def assertResolved(self, result):
        self.assertIsNone(result['error'], result)
        self.assertNotIn('UNRESOLVED', [e['kind'] for e in result['events']])

    def test_one_third_abilities_read_phazing_below_twenty(self):
        for ability, status in (('Static', 'par'), ('Poison Point', 'psn'), ('Flame Body', 'brn')):
            role = 'static' if ability == 'Static' and self.schema == 'v4' else 'contact_' + ability.lower().replace(' ', '')
            for raw, r in zip((19, 20), self.out[ability]):
                with self.subTest(ability=ability, raw=raw):
                    self.assertResolved(r)
                    self.assertIn((role, 0, raw), coupled(r))
                    self.assertEqual(r['status'], status if raw < 20 else '')

    def test_effect_spore_chance_pick_and_sleep_share_one_draw(self):
        expected = ['slp', 'slp', 'par', 'par', 'psn', 'psn', '']
        for raw, (r, status) in enumerate(zip(self.out['spore'], expected)):
            with self.subTest(raw=raw):
                self.assertResolved(r)
                roles = coupled(r)
                self.assertIn(('contact_effectspore', 0, raw), roles)
                self.assertEqual(('effect_spore_status', 0, raw) in roles, raw < 6)
                self.assertEqual(('effect_spore_sleep_duration', 0, 3) in roles, status == 'slp')
                self.assertNotIn('sleep_duration', [x[0] for x in roles])
                self.assertEqual(r['status'], status)
                if status == 'slp':self.assertEqual(r['time'], 2 + 3)

    def test_a_v2_tape_leaves_the_contact_roles_unresolved(self):
        self.assertIn('SPIKE_UNRESOLVED:PHAZING_STREAM_ABSENT', self.out['static_v2']['error'])

    def test_tape_schema_validation(self):
        for name, error in self.out['invalid'].items():
            with self.subTest(name=name):self.assertEqual(error, 'invalid chance tape')


class ChanceHooksV4ContactTests(ChanceHooksV3Tests):
    schema = 'v4'


if __name__ == '__main__':
    unittest.main()
