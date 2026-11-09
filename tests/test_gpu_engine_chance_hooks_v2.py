"""Tape v2 and the fidelity-patch chance-hook roles, on real Showdown battles (no engine)."""
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
function T(nonce,o={},schema='v2'){
  const t={schema:'pokezero.gpu_chance_tape.'+schema,nonce,draws:[row(),row()],secondary:[99,99],sleep_talk:[0,0]};
  if(schema==='v2')t.effect=[[15,0,0,0],[15,0,0,0]];
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
const out={};
{const b=battle([mon('Machamp',['superpower'],'Guts')],[mon('Snorlax',['splash'],'Thick Fat')]);
 out.superpower={...turn(b,T('sp')),boosts:b.p1.active[0].boosts};}
{const b=battle([mon('Snorlax',['curse'],'Thick Fat')],[mon('Blissey',['splash'],'Natural Cure')]);
 out.curse={...turn(b,T('cu')),boosts:b.p1.active[0].boosts};}
{const b=battle([mon('Marowak',['bonemerang'],'Rock Head')],[mon('Snorlax',['splash'],'Thick Fat')]);
 out.bonemerang={...turn(b,T('bo',{'effect.0.0':0,'effect.0.1':5}))};}
{const b=battle([mon('Marowak',['bonemerang'],'Rock Head')],[mon('Snorlax',['splash'],'Thick Fat')]);
 out.bonemerang_v1={...turn(b,T('bo1',{},'v1'))};}
{const b=battle([mon('Arbok',['wrap'],'Intimidate')],[mon('Snorlax',['splash'],'Thick Fat')]);
 const r=turn(b,T('wr',{'draws.0.7':2}));
 out.wrap={...r,duration:b.p2.active[0].volatiles.partiallytrapped?.duration??null};}
{const b=battle([mon('Ariados',['signalbeam'],'Insomnia')],[mon('Snorlax',['splash'],'Thick Fat')]);
 const r=turn(b,T('sb',{'secondary.0':0,'draws.0.7':1}));
 out.signalbeam={...r,time:b.p2.active[0].volatiles.confusion?.time??null};}
{const b=battle([mon('Swalot',['yawn','splash'],'Liquid Ooze')],[mon('Snorlax',['splash'],'Thick Fat')]);
 const first=turn(b,T('ya0'));
 const second=turn(b,T('ya1',{'effect.1.2':3}),'move 2','move 1');
 out.yawn={first,second,status:b.p2.active[0].status,time:b.p2.active[0].statusState.time??null};}
// Residual pair ties need equal Speed, where every action's eachEvent('Update') already
// meets an unmapped Pokemon-order tie; so drive the residual handler sort directly.
function pairTie(id,o){
  const b=battle([mon('Arbok',['splash'],'Shed Skin')],[mon('Arbok',['splash'],'Shed Skin')]);
  const h=p=>({effect:{id},effectHolder:p,order:10,priority:0,speed:200,subOrder:9,callback(){}});
  const list=[h(b.p1.active[0]),h(b.p2.active[0])];
  tape=T('tie:'+id+':'+JSON.stringify(o),o);events.length=0;let error=null;
  try{b.speedSort(list);}catch(e){error=e.message;}
  tape=null;return {error,events:events.slice(),first:list[0].effectHolder.side.n};
}
out.perish=[pairTie('perishsong',{'effect.1.3':0}),pairTie('perishsong',{'effect.1.3':1})];
out.wraptie=[pairTie('partiallytrapped',{'effect.0.3':0}),pairTie('partiallytrapped',{'effect.0.3':1})];
out.mixedtie=pairTie('leftovers',{});
{const b=battle([mon('Arbok',['wrap'],'Shed Skin')],[mon('Snorlax',['splash'],'Thick Fat')]);
 out.wrapv1=turn(b,T('wv1',{'draws.0.7':1},'v1'));}
{const b=battle([mon('Blissey',['protect'],'Natural Cure')],[mon('Snorlax',['splash'],'Thick Fat')]);
 out.protect=turn(b,T('pr'));}
out.invalid={};
for(const [name,t] of Object.entries({
  v1_with_effect:{...T('i1',{},'v1'),effect:[[0,0,0,0],[0,0,0,0]]},
  v2_without_effect:(()=>{const t=T('i2');delete t.effect;return t;})(),
  v2_effect_out_of_range:T('i3',{'effect.0.3':2}),
  v2_short_effect_row:T('i4',{'effect':[[0,0,0],[0,0,0,0]]}),
})){const b=battle([mon('Snorlax',['splash'],'Thick Fat')],[mon('Blissey',['splash'],'Natural Cure')]);out.invalid[name]=turn(b,t).error;}
{const b=battle([mon('Snorlax',['splash'],'Thick Fat')],[mon('Blissey',['splash'],'Natural Cure')]);out.v1_ok=turn(b,T('ok',{},'v1')).error;}
process.stdout.write(JSON.stringify(out));
'''


def coupled(result):
    return [(e['role'], e['side'], e['raw']) for e in result['events'] if e['kind'] == 'COUPLED']


def kinds(result):
    return [e['kind'] for e in result['events']]


@requires_showdown
@unittest.skipUnless(NODE, 'requires node')
class ChanceHooksV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = subprocess.check_output([NODE, '-e', SCRIPT, str(showdown_root()), str(ROOT / 'gpu_engine/chance_hooks.cjs')], text=True)
        cls.out = json.loads(raw)

    def assertResolved(self, result):
        self.assertIsNone(result['error'], result)
        self.assertNotIn('UNRESOLVED', kinds(result))

    def test_self_drops_roll_is_inert_and_the_drop_applies(self):
        for name, boosts in (('superpower', dict(atk=-1, def_=-1)), ('curse', dict(atk=1, def_=1, spe=-1))):
            with self.subTest(name=name):
                r = self.out[name];self.assertResolved(r)
                inert = [e for e in r['events'] if e['kind'] == 'INERT_SELF_DROPS']
                self.assertEqual([e['move'] for e in inert], [name])
                for stat, value in boosts.items():self.assertEqual(r['boosts'][stat.rstrip('_')], value)

    def test_second_multihit_critical_and_damage_read_the_effect_stream(self):
        r = self.out['bonemerang'];self.assertResolved(r)
        roles = coupled(r)
        self.assertIn(('critical', 0, 15), roles);self.assertIn(('damage', 0, 0), roles)
        self.assertIn(('critical_hit2', 0, 0), roles);self.assertIn(('damage_hit2', 0, 5), roles)
        self.assertEqual(sum('|-crit|' in line for line in r['log']), 1)
        v1 = self.out['bonemerang_v1']
        self.assertIn('SPIKE_UNRESOLVED:EFFECT_STREAM_ABSENT', v1['error'])

    def test_partial_trap_and_confusion_durations_share_draw_column_seven(self):
        r = self.out['wrap'];self.assertResolved(r)
        self.assertIn(('partial_trap_duration', 0, 2), coupled(r))
        self.assertEqual(r['duration'], 3 + 2 - 1)        # 3+d at the hit, one residual tick this turn
        r = self.out['signalbeam'];self.assertResolved(r)
        self.assertIn(('confusion_duration', 0, 1), coupled(r))
        self.assertEqual(r['time'], 2 + 1 - 1)            # 2+d, then the confused mon's own before-move

    def test_yawn_residual_sleep_reads_the_drowsy_sides_effect_column(self):
        r = self.out['yawn'];self.assertResolved(r['first']);self.assertResolved(r['second'])
        self.assertIn(('yawn_sleep_duration', 1, 3), coupled(r['second']))
        self.assertNotIn('sleep_duration', [x[0] for x in coupled(r['second'])])
        self.assertEqual((r['status'], r['time']), ('slp', 2 + 3))

    def test_residual_pair_ties_for_perish_and_partial_trap(self):
        for name, role, row in (('perish', 'perish_tie', 1), ('wraptie', 'partial_trap_tie', 0)):
            for value, r in enumerate(self.out[name]):
                with self.subTest(name=name, value=value):
                    self.assertResolved(r)
                    self.assertEqual(coupled(r), [(role, row, value)])
                    self.assertEqual(r['first'], value)      # the side whose n equals the tape value goes first
        self.assertIn('SPIKE_UNRESOLVED:TIE_ORDER', self.out['mixedtie']['error'])
        # A v1 tape still couples every draws-only role.
        self.assertResolved(self.out['wrapv1'])
        self.assertIn(('partial_trap_duration', 0, 1), coupled(self.out['wrapv1']))

    def test_one_mons_stall_and_protect_countdowns_are_an_inert_tie(self):
        r = self.out['protect'];self.assertResolved(r)
        ties = [e for e in r['events'] if e['kind'] == 'INERT_RESIDUAL_TIE']
        self.assertEqual(len(ties), 1);self.assertEqual(sorted(ties[0]['effects']), ['protect', 'stall'])

    def test_tape_schema_validation(self):
        for name, error in self.out['invalid'].items():
            with self.subTest(name=name):self.assertEqual(error, 'invalid chance tape')
        self.assertIsNone(self.out['v1_ok'])


if __name__ == '__main__':
    unittest.main()
