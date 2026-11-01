"""One static full-namespace fixture; no generation, turn, FULL or model."""
import ast,copy,json,subprocess,unittest
from pathlib import Path
import jax.numpy as j
import numpy as np
from pokezero.dex import load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import load_gen3_randbat_source_cached
from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
from gpu_engine import engine_sleep_blocked_callbacks as slp,engine_before_move_blocked_callbacks as blocked
from gpu_engine import public_belief,v4_encoder,transition,engine_public_events as events
from gpu_engine import engine_request_source as request,engine_own_facts as own,engine_direct_hit_controls as direct
from gpu_engine import engine_leftovers_callbacks as leftovers,engine_retained_encoder_driver as driver
from gpu_engine import engine_lifecycle_callbacks as life,engine_execution_completeness as ex,engine_kernel_frame_risk as risk,state,mechanics
from _showdown_root import showdown_root
BASE='fd3256d2305a0770764e737640fc80157156cb29'

def original_compile(module):
 tree=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/'+Path(module.__file__).name],text=True));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='compile_data');node.name='old_compile';space=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module([node],[])),'<unchanged original compile_data>','exec'),space);return space['old_compile']

class BeforeMoveHiddenPowerAliasTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  # Evaluate ONLY the original generator's static metadata prefix. Its team
  # loop is excluded, and Teams/RNG dependency imports are not needed here.
  script=r"""
const fs=require('fs'),vm=require('vm'),s=fs.readFileSync(process.argv[1],'utf8'),start=s.indexOf('async function generate('),body=s.slice(s.indexOf('{',start)+1,s.indexOf('  async function emit(',start)).split('\n').filter(x=>!x.includes("const {Teams}")&&!x.includes('const rngPackage')).join('\n');
const code=s.slice(0,start)+'\nfunction metadata(root){'+body+"return {format:'gen3randombattle',source_commit_label:require('node:child_process').execFileSync('git',['-C',root,'rev-parse','HEAD'],{encoding:'utf8'}).trim(),compiled_tree:compiledTree(path.join(root,'dist')),vocabulary,moves,type_chart:chart,stats_order:STATS};}\nJSON.stringify(metadata(ROOT))";
process.stdout.write(vm.runInNewContext(code,{require,ROOT:process.argv[2]}));
"""
  cls.metadata=json.loads(subprocess.check_output(['node','-e',script,'gpu_engine/generate_teams.cjs',str(showdown_root())],text=True));v=json.loads(Path('/Users/scott/workspace/agents/pokezero-b4-full-gap-census-r21-ops-20261101/vocabulary.json').read_text());vocab=CategoryVocabulary(tuple(v['tokens']),v['oov_buckets'],v['aliases']);cls.dex=load_showdown_dex_cached(showdown_root())
  # Original pinned static source inputs, matching production compilation.
  # No generator, constructor, engine, policy or model is invoked.
  source=load_gen3_randbat_source_cached(showdown_root())
  cls.catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),cls.dex,vocab,checkpoint_sha256=v['checkpoint_sha256'],feature_masks=CHAMPION_CONTRACT['feature_masks']);cls.table=transition.tables_from_metadata(cls.metadata)
  cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=events.HealthExposure(request.SHOWDOWN,'gen3randombattle',True,False));cls.rules=request.load_source_rules(showdown_root(),cls.metadata)
  cls.old_slp=original_compile(slp)(cls.producer,cls.metadata,cls.rules);cls.old_blocked=original_compile(blocked)(cls.producer,cls.metadata,cls.rules)
  compiled=request.compile_request_source(cls.metadata,vocabulary=vocab,dex=cls.dex,checkpoint_sha256=v['checkpoint_sha256'],rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(compiled,cls.catalog);cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(compiled,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:None);cls.runtime=driver._checked(cls.prepared);cls.sleep=cls.helper.device.sleep_blocked;cls.blocked=cls.helper.device.before_move_blocked
  lookup=[]
  for i in(134,309):lookup.append(dict(index=i,name=cls.metadata['vocabulary']['moves'][i],public_id=cls.producer.ids[i],public_name=cls.catalog.moves.names[cls.producer.ids[i]],old_sleep=bool(cls.old_slp.moves[i]),new_sleep=bool(cls.sleep.moves[i]),old_par_frz=cls.old_blocked.moves[:,i].tolist(),new_par_frz=cls.blocked.moves[:,i].tolist()))
  cls.lookup=lookup;Path('/tmp/pokezero-before-move-hidden-power-alias-lookup.json').write_text(json.dumps({'scope':'STATIC_FULL_NAMESPACE_FIXTURE_NOT_SAVED_R21_ARRAYS','rows':lookup},sort_keys=True)+'\n')

 def tape(self,statuses,counters):
  r=j.arange(8);z=j.zeros(8,j.int32);yes=j.ones(8,bool);side=r%2
  s=state.empty_state(8)._replace(species=j.ones((8,2,6),j.int32),level=j.full((8,2,6),80,j.int32),gender=j.ones((8,2,6),j.int32),done=~yes,hp=j.full((8,2,6),100,j.int32),max_hp=j.full((8,2,6),100,j.int32),ability=j.full((8,2,6),self.metadata['vocabulary']['abilities'].index('pressure'),j.int32))
  s=s._replace(status=s.status.at[r,side,z].set(statuses),status_counter=s.status_counter.at[r,side,z].set(counters))
  cursor=life.register(r+93001,catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));t=life._tape(cursor,self.producer,64)
  for owner in(0,1):t=life._reveal(t,s,z+owner,z-1,z,eligible=yes,replacement=~yes,baton=~yes,initial=True)
  t=life._turn(t._replace(cursor=t.cursor._replace(initialized=yes)),yes)
  t=life._tape(t.cursor,self.producer,64);a=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device
  t=ex.begin_on_tape(t,ex.initialize(t.stream_id,compiled=a),a.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,rt),t,rt,active=yes);t=t._replace(execution=t.execution._replace(risks=risk.Context(carry,rt)));return life.begin(t,s,yes,~yes),s,side,z,yes

 def test_exact_full_namespace_masks_all_aliases_and_closed_foreign_sources(self):
  self.assertEqual(self.lookup[0]['name'],'hiddenpowerfighting');self.assertEqual(self.lookup[1]['name'],'spore');self.assertGreater(self.lookup[0]['public_id'],0)
  self.assertFalse(self.lookup[0]['old_sleep']);self.assertEqual(self.lookup[0]['old_par_frz'],[False,False]);self.assertTrue(self.lookup[1]['old_sleep']);self.assertEqual(self.lookup[1]['old_par_frz'],[True,True])
  source=Path('gpu_engine/generate_teams.cjs').read_text();literal=source.split('const TYPES = ',1)[1].split(';',1)[0];types=ast.literal_eval(literal);self.assertEqual(slp.HIDDEN_POWER_TYPES,tuple(x for x in types if x!='Normal'))
  names=self.metadata['vocabulary']['moves']
  script="const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>{const m=d.moves.get(n);return{requested:n,id:m.id,realMove:m.realMove||null,type:m.type,exists:m.exists,gen:m.gen,sleepUsable:!!m.sleepUsable}})));"
  rows=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root()),json.dumps(names)],text=True))
  for name in slp.HIDDEN_POWER_ALIASES:
   i=names.index(name);self.assertEqual(rows[i]['id'],'hiddenpower');self.assertEqual(rows[i]['gen'],2);self.assertFalse(rows[i]['sleepUsable']);self.assertTrue(bool(self.sleep.moves[i]));self.assertEqual(self.blocked.moves[:,i].tolist(),[True,True]);self.assertEqual(self.producer.ids[i],self.producer.ids[names.index('hiddenpower')])
  i=134
  for key,value in [('id','hiddenpowerfighting'),('requested','hiddenpowerbug'),('realMove','Other'),('type','Bug'),('gen',3),('sleepUsable',True),('exists',False)]:
   bad=copy.deepcopy(rows);bad[i][key]=value;self.assertFalse(bool(slp.canonical_move_rows(self.producer,names,bad)[i]),key)
  for unknown in('hiddenpowernormal','hiddenpowerfairy','hiddenpowerevil','hiddenpowerfightingextra'):
   badnames=list(names);badnames[i]=unknown;bad=copy.deepcopy(rows);bad[i]['requested']=unknown;self.assertFalse(bool(slp.canonical_move_rows(self.producer,badnames,bad)[i]),unknown)
  # All nonalias rows retain the old candidates, including defrost filtering.
  ordinary=np.asarray([n not in slp.HIDDEN_POWER_ALIASES for n in names]);np.testing.assert_array_equal(np.asarray(self.sleep.moves)[ordinary],np.asarray(self.old_slp.moves)[ordinary]);np.testing.assert_array_equal(np.asarray(self.blocked.moves)[:,ordinary],np.asarray(self.old_blocked.moves)[:,ordinary])

 def test_actual_same_call_sleep_and_par_frz_raw_receipts_both_actor_sides(self):
  statuses=j.full(8,mechanics.SLP,j.int32);counter=j.full(8,5,j.int32);t,s,side,z,yes=self.tape(statuses,counter);move=j.full(8,134,j.int32);context=self.runtime['direct_hit_context']
  t,p=slp.begin(t,s,side,move,context,self.table.sleep_talk,eligible=yes,sleep_usable=~yes,early_bird=~yes,loafing=~yes);self.assertEqual(p.requested.tolist(),[True]*8);self.assertEqual(p.valid.tolist(),[True]*8)
  before=mechanics.before_move(statuses,counter);r=j.arange(8);after=s._replace(status=s.status.at[r,side,z].set(before.status),status_counter=s.status_counter.at[r,side,z].set(before.counter));t,c=slp.finish(t,p,after,before)
  self.assertEqual(slp.owned(t,1).tolist(),[True]*8);self.assertEqual(slp.captured_matches(t,c,s,side,before).tolist(),[True]*8)
  expected=j.stack((z+47,side,z,move,z+5,z+4,z,z,z,z+5),axis=1);np.testing.assert_array_equal(t.words[:,1],expected);self.assertEqual(t.count.tolist(),[2]*8);np.testing.assert_array_equal(t.words[:,0],j.broadcast_to(j.array([35,-1,-1,-1,-1,0,0,0,0,0],j.int32),(8,10)))
  for flags in(dict(sleep_usable=yes,early_bird=~yes,loafing=~yes),dict(sleep_usable=~yes,early_bird=~yes,loafing=yes)):
   self.assertEqual(slp.begin(t,s,side,move,context,self.table.sleep_talk,eligible=yes,**flags)[1].requested.tolist(),[False]*8)
  statuses=j.where(j.arange(8)%3==0,mechanics.PAR,mechanics.FRZ).astype(j.int32);t,s,side,z,yes=self.tape(statuses,z);draws=j.broadcast_to(j.array([99,0,99],j.int32),(8,3))
  t,p=blocked.begin(t,s,side,move,move,draws,context,eligible=yes,sleep_usable=~yes,loafing=~yes);self.assertEqual(p.requested.tolist(),[True]*8);self.assertEqual(p.valid.tolist(),[True]*8)
  before=mechanics.before_move(statuses,z,thaw_roll=draws[:,0],paralysis_roll=draws[:,1],confusion_roll=draws[:,2]);t,c=blocked.finish(t,p,s,before);self.assertEqual(blocked.owned(t,1).tolist(),[True]*8);self.assertEqual(blocked.captured_matches(t,c,s,side,before).tolist(),[True]*8)
  expected=j.stack((z+56,side,z,move,statuses,draws[:,0],draws[:,1],draws[:,2],z,s.ability[j.arange(8),side,z]),axis=1);np.testing.assert_array_equal(t.words[:,1],expected);self.assertEqual(t.count.tolist(),[2]*8);np.testing.assert_array_equal(t.words[:,0],j.broadcast_to(j.array([35,-1,-1,-1,-1,0,0,0,0,0],j.int32),(8,10)))
  self.assertEqual(blocked.begin(t,s,side,move,move+1,draws,context,eligible=yes,sleep_usable=~yes,loafing=~yes)[1].requested.tolist(),[False]*8)

 def test_all_runtime_capture_physics_consumers_and_signatures_unchanged(self):
  for file in('engine_sleep_blocked_callbacks.py','engine_before_move_blocked_callbacks.py'):
   a=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/'+file],text=True));b=ast.parse(Path('gpu_engine',file).read_text());skip={'compile_data','canonical_move_rows'};f=lambda t:{n.name:ast.dump(n,include_attributes=False)for n in t.body if isinstance(n,ast.FunctionDef)and n.name not in skip};self.assertEqual(f(a),f(b))
  for file in('transition.py','mechanics.py','engine_retained_encoder_driver.py','engine_kernel_frame_risk.py','engine_full_v4_event_device.py','engine_lifecycle_event_device.py','engine_execution_completeness.py'):
   self.assertEqual(Path('gpu_engine',file).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+file]))

if __name__=='__main__':unittest.main()
