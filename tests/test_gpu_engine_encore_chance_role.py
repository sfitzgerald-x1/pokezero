"""Original callbacks on inert objects only; no Native battle/game/RNG sampling."""
import hashlib,json,subprocess,unittest
from pathlib import Path
from _showdown_root import showdown_root

SCRIPT=r'''
const path=require('path'),crypto=require('crypto'),hookPath=process.argv[1],root=process.argv[2],mode=process.argv[3];
const {Dex}=require(path.join(root,'dist/sim/dex.js')),{Pokemon}=require(path.join(root,'dist/sim/pokemon.js'));
const dex=Dex.mod('gen3'),condition=dex.conditions.get('encore'),originalDuration=condition.durationCallback,originalStart=condition.onStart,originalAdd=Pokemon.prototype.addVolatile;
const h=x=>crypto.createHash('sha256').update(x.toString()).digest('hex');
const sources={duration:h(originalDuration),start:h(originalStart),addVolatile:h(originalAdd)};
if(mode==='source_drift'){const get=dex.conditions.get;dex.conditions.get=function(name){const c=get.call(this,name);return c.id==='encore'?{...c,durationCallback(){return this.random(3,7);}}:c;};}
if(mode==='callsite_drift')Pokemon.prototype.addVolatile=function(...a){return originalAdd.apply(this,a);};
let current=null,events=[],fallback=[];
class UnitBattle{random(...a){fallback.push(a);return a[0]+1;}randomChance(){throw Error('unused original chance');}sample(x){return x[0];}speedSort(x){return x;}}
class UnitPRNG{random(){throw Error('no real PRNG allowed');}}
require(hookPath).install(UnitBattle,UnitPRNG,()=>current,x=>events.push(x),()=>null,mode==='no_path'?'':path.join(root,'dist/sim/pokemon.js'));
function control(nonce,draw){return {schema:'pokezero.gpu_chance_tape.v1',nonce,draws:[Array(12).fill(0),Array(12).fill(0)].map(x=>(x[7]=draw,x)),secondary:[0,0],sleep_talk:[0,0]};}
function fixture(actor,queued){
 const b=new UnitBattle();b.dex=dex;b.gen=3;b.effect={id:''};b.effectState={};b.event=null;b.sides=[{n:0,active:[]},{n:1,active:[]}];b.queue={willMove:()=>queued?{}:null};b.lines=[];
 const mons=[0,1].map(seat=>Object.assign(Object.create(Pokemon.prototype),{battle:b,side:b.sides[seat],hp:100,volatiles:{},lastMove:dex.moves.get('surf'),runStatusImmunity:()=>true,getMoveData:()=>({pp:5}),getSlot:()=>`p${seat+1}a`,toString:()=>`p${seat+1}a: UNIT`}));
 for(let s=0;s<2;s++)b.sides[s].active=[mons[s]];
 b.activePokemon=mons[actor];b.activeTarget=mons[1-actor];b.activeMove=dex.getActiveMove('encore');b.add=(...xs)=>b.lines.push(xs.map(String).join('|'));b.debug=()=>{};b.runEvent=()=>true;b.initEffectState=x=>({...x});
 b.singleEvent=function(name,status,data,target,source,effect){if(name!=='Start')throw Error('unexpected event');const old=this.effectState;this.effectState=data;try{const value=status.onStart.call(this,target,source,effect);return value===undefined?true:value;}finally{this.effectState=old;}};
 return {b,source:mons[actor],target:mons[1-actor],move:b.activeMove};
}
function invoke(f){return f.target.addVolatile('encore',f.source,f.move);}
const rows=[];
function run(name,actor,draw,queued,change=null,act=invoke){
 events=[];fallback=[];current=control(name,draw);const f=fixture(actor,queued);if(change)change(f);
 let error=null,value;try{value=act(f);}catch(e){error=e.message;}
 rows.push({name,actor,draw,queued,error,value:value===undefined?null:value,duration:f.target.volatiles.encore?.duration??null,lines:f.b.lines,events:[...events],fallback:[...fallback]});return f;
}
if(mode==='positive'||mode==='source_drift'||mode==='callsite_drift'||mode==='no_path'){
 for(let actor=0;actor<2;actor++)for(let draw=0;draw<4;draw++)for(const queued of [false,true])run(`p:${actor}:${draw}:${queued}`,actor,draw,queued);
}else if(mode==='negative'){
 run('foreign-range-only',0,2,true,null,f=>f.b.random(3,7));
 run('direct-original-callback-outside-addVolatile',0,2,true,null,f=>originalDuration.call(f.b,f.target,f.source,f.move));
 run('nonce-changed-within-original-call',0,2,true,f=>f.b.runEvent=()=>{current.nonce+=':foreign';return true;});
 run('foreign-wrong-upper',0,2,true,null,f=>f.b.random(3,8));
 run('foreign-wrong-lower',0,2,true,null,f=>f.b.random(4,7));
 run('no-battle',0,2,true,f=>f.target.battle=null);
 run('no-actor',0,2,true,f=>f.b.activePokemon=null);
 run('foreign-actor',0,2,true,f=>f.b.activePokemon=f.target);
 run('no-target',0,2,true,f=>f.b.activeTarget=null);
 run('called-sleeptalk',0,2,true,f=>f.move.sourceEffect='sleeptalk');
 run('foreign-active-move',0,2,true,f=>f.b.activeMove=dex.getActiveMove('surf'));
 run('foreign-effect-argument',0,2,true,f=>f.move=dex.getActiveMove('encore'));
 run('foreign-battle',0,2,true,f=>f.target.battle={dex,runEvent:()=>true,initEffectState:x=>x});
 run('encore-then-sleep',0,2,true,null,f=>{invoke(f);f.b.effect={id:'slp'};return f.b.random(2,6);});
 run('sleep-then-encore',1,2,true,null,f=>{f.b.effect={id:'slp'};f.b.random(2,6);f.b.effect={id:''};return invoke(f);});
 run('repeated-duration',0,2,true,null,f=>{invoke(f);delete f.target.volatiles.encore;return invoke(f);});
}else if(mode==='conditional'){
 run('pp-failure-still-original-duration-call',0,1,false,f=>f.target.getMoveData=()=>({pp:0}));
 run('existing-volatile-no-new-duration',0,1,false,f=>f.target.volatiles.encore={id:'encore',duration:2});
 run('opposite-actor-duration-roles',1,3,true,null,f=>{current.draws[0][7]=0;f.b.effect={id:'slp'};f.b.activePokemon=f.target;f.b.random(2,6);f.b.effect={id:''};f.b.activePokemon=f.source;return invoke(f);});
 events=[];fallback=[];current=null;const f=fixture(0,true);let error=null;try{invoke(f);}catch(e){error=e.message;}rows.push({name:'uncontrolled-delegates-original',error,events,fallback,duration:f.target.volatiles.encore?.duration});
}
process.stdout.write(JSON.stringify({sources,callbacks_unchanged:condition.durationCallback===originalDuration&&condition.onStart===originalStart,rows}));
'''

class EncoreChanceRoleTests(unittest.TestCase):
 def run_component(self,mode):
  raw=subprocess.check_output(['node','-e',SCRIPT,str(Path('gpu_engine/chance_hooks.cjs').resolve()),str(showdown_root()),mode],text=True)
  data=json.loads(raw);Path('/tmp/pokezero-encore-chance-'+mode+'-current.json').write_text(json.dumps(data,sort_keys=True)+'\n');return data
 def test_original_duration_and_queue_correspondence_both_actors_all_four_draws(self):
  data=self.run_component('positive');self.assertTrue(data['callbacks_unchanged']);self.assertEqual(data['sources'],dict(duration='5fa95c56ee1187568ff774129ba7816873d4d6a4806130268725463caea45786',start='65cca25468443dea05dd74c855318f3668d29f5a791838ef10d845536937eeca',addVolatile='cddf90eeb6711d33592f46b063e3f804f395d5dfe9ec8ccd833540d4280add51'))
  for row in data['rows']:
   with self.subTest(name=row['name']):
    self.assertIsNone(row['error']);self.assertEqual(row['duration'],3+row['draw']+int(not row['queued']));self.assertEqual(row['fallback'],[])
    coupled=[x for x in row['events']if x['kind']=='COUPLED'];self.assertEqual([(x['role'],x['side'],x['raw'])for x in coupled],[('encore_duration',row['actor'],row['draw'])])
    self.assertEqual(row['lines'],[f"-start|p{2-row['actor']}a: UNIT|Encore"])
  engine=Path('gpu_engine/encore.py').read_text();self.assertIn('duration=3+roll+(~jnp.asarray(queued)).astype(jnp.int32)',engine)
  self.assertIn('work=repeating.apply(work,foe,move,chance[:,7],table.encore',Path('gpu_engine/transition.py').read_text())
 def test_noncallback_called_missing_context_mixed_roles_and_source_drift_stay_unknown(self):
  for mode in ('negative','source_drift','callsite_drift','no_path'):
   for row in self.run_component(mode)['rows']:
    with self.subTest(mode=mode,name=row['name']):
     self.assertIsNotNone(row['error']);self.assertIn('SPIKE_UNRESOLVED:',row['error']);self.assertEqual(row['fallback'],[])
     self.assertTrue(any(x['kind']=='UNRESOLVED'for x in row['events']))
 def test_actual_conditional_call_counts_and_uncontrolled_legacy_delegate(self):
  data=self.run_component('conditional');rows={r['name']:r for r in data['rows']}
  pp=rows['pp-failure-still-original-duration-call'];self.assertIsNone(pp['error']);self.assertFalse(pp['value']);self.assertEqual(pp['lines'],[]);self.assertEqual(len([x for x in pp['events']if x['kind']=='COUPLED']),1)
  old=rows['existing-volatile-no-new-duration'];self.assertIsNone(old['error']);self.assertFalse(old['value']);self.assertEqual(old['events'],[])
  peer=rows['opposite-actor-duration-roles'];self.assertIsNone(peer['error']);self.assertEqual([(x['role'],x['side'],x['raw'])for x in peer['events']if x['kind']=='COUPLED'],[('sleep_duration',0,0),('encore_duration',1,3)]);self.assertEqual(peer['duration'],6)
  legacy=rows['uncontrolled-delegates-original'];self.assertIsNone(legacy['error']);self.assertEqual(legacy['events'],[]);self.assertEqual(legacy['fallback'],[[3,7]]);self.assertEqual(legacy['duration'],4)

if __name__=='__main__':unittest.main()
