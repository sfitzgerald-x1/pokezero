"""Exact source-dormant callback and one current Ground transaction component."""
import ast,copy,json,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import jax.numpy as j
import numpy as np
from gpu_engine import engine_thunder_wave_callbacks as cb,engine_execution_completeness as ex
from gpu_engine import engine_kernel_frame_risk as risk,engine_lifecycle_callbacks as life,engine_public_events as public
from _showdown_root import showdown_root
import test_gpu_engine_thunder_wave_ground_immunity as original

BASE='e481228626c9cfdfe2a2989871672392aa3e5859'

class DormantWaterAbsorbTests(unittest.TestCase):
 setUpClass=classmethod(original.ThunderGroundTests.setUpClass.__func__)
 component=original.ThunderGroundTests.component

 def literal(self,full=False,**change):
  s,t,pubs,actor,phys=self.component(full);b=8;r=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);target=s.active[r,1-actor];g=self.thunder.device.ground_immunity
  wa=self.metadata['vocabulary']['abilities'].index('waterabsorb');quiet=int(next(i for i,x in enumerate(np.asarray(g.abilities))if x and i!=wa));self.assertGreater(quiet,0)
  types=j.full((b,2,6,2),self.metadata['vocabulary']['types'].index('normal'),j.int32).at[r,1-actor,target].set(j.stack((z+int(g.ground),z),axis=1))
  abilities=j.full((b,2,6),quiet,j.int32).at[r,1-actor,target].set(j.where(r%3==0,quiet,wa))
  s=s._replace(done=~yes,turn=t.cursor.turn-1,types=types,base_types=types,ability=abilities,base_ability=abilities,moves=s.moves.at[r,actor,phys,0].set(self.move_id),pp=s.pp.at[r,actor,phys,0].set(5))
  t=life.begin(t,s,yes,~yes);move=z+self.move_id;t=public.move(t,actor,phys,target,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.PRIMARY_PARALYSIS,1-actor,active=yes)
  kw=dict(requested=yes,resolved_type=z+int(g.electric),power=z,accuracy=yes,hit=yes,immune=yes,numerator=z,denominator=z+1,protected=~yes,absorbed=~yes);kw.update(change)
  p=cb.begin_ground(t,s,ticket,actor,move,move,z,self.runtime['thunder_wave_context'],**kw)
  a=cb.Attempt(yes,yes,~yes,~yes,~yes,yes,~yes,~yes,~yes,yes,~yes,z,z,z,z)
  t=cb.record_device(t,ticket,actor,phys,target,move,a,self.runtime['thunder_wave_context'],requested=yes,ground_pending=p,state=s)
  return s,t,pubs,actor,phys,target,move,a,ticket,p

 def test_source_inert_electric_and_water_branch_mask_source_guards(self):
  script=r"""
const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),a=d.abilities.get('waterabsorb'),rows=[];
for(const name of['thunderwave','hydropump'])for(const hp of[75,100]){
 const lines=[],source={},target={hp,baseMaxhp:100,status:'',counter:0,pp:[4,5],toString(){return'p2a: Target'}},before=JSON.stringify(target);let heals=0;
 const value=a.onTryHit.call({heal(amount){heals++;const old=target.hp;target.hp=Math.min(100,target.hp+amount);return target.hp!==old},add(...args){lines.push('|'+args.map(String).join('|'))}},target,source,d.moves.get(name));
 rows.push({name,type:d.moves.get(name).type,hp,after:target.hp,heals,lines,unchanged:before===JSON.stringify(target),value:value===undefined?'undefined':value});
}
console.log(JSON.stringify({sha:require('crypto').createHash('sha256').update(String(a.onTryHit)).digest('hex'),rows}));
"""
  result=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(result['sha'],cb.WATER_ABSORB_TRY_HIT_SHA256)
  for row in result['rows'][:2]:self.assertEqual((row['heals'],row['lines'],row['unchanged'],row['value']),(0,[],True,'undefined'))
  self.assertEqual(result['rows'][2]['heals'],1);self.assertEqual(result['rows'][2]['after'],100);self.assertIsNone(result['rows'][2]['value']);self.assertEqual(result['rows'][3]['lines'],['|-immune|p2a: Target|[from] ability: Water Absorb'])
  wa=self.metadata['vocabulary']['abilities'].index('waterabsorb');self.assertTrue(bool(self.thunder.device.ground_immunity.abilities[wa]));self.assertIsNone(self.legacy.device.ground_immunity)
  original_run=cb.subprocess.run;captured=[]
  def capture(*args,**kwargs):
   value=original_run(*args,**kwargs);captured.append(json.loads(value.stdout));return value
  with patch.object(cb.subprocess,'run',side_effect=capture):cb.compile_thunder_wave(engine_catalog=self.producer,metadata=self.metadata,source_rules=self.rules)
  descriptor=captured[0];hooks=descriptor['ground']['dormantWaterAbsorb'];self.assertEqual(hooks['otherHooks'],['undefined']*12)
  for field,value in [('id','voltabsorb'),('onTryHit','0'*64),('otherHooks',['function']+['undefined']*11)]:
   bad=copy.deepcopy(descriptor);bad['ground']['dormantWaterAbsorb'][field]=value
   with patch.object(cb.subprocess,'run',return_value=NS(stdout=json.dumps(bad))),self.assertRaisesRegex(ValueError,'dormant Water Absorb'):
    cb.compile_thunder_wave(engine_catalog=self.producer,metadata=self.metadata,source_rules=self.rules)
  with patch.object(cb,'WATER_ABSORB_TRY_HIT_SHA256','0'*64),self.assertRaisesRegex(ValueError,'dormant Water Absorb'):
   cb.compile_thunder_wave(engine_catalog=self.producer,metadata=self.metadata,source_rules=self.rules)
  # Exactly this one source-qualified mask bit changed. Electric absorption
  # (Volt Absorb), Flash Fire and every other old hook exclusion stay closed.
  script="const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),hooks=['onTryHit','onTryPrimaryHit','onTryMove','onPrepareHit','onModifyMove','onAnyModifyMove','onAllyModifyMove','onNegateImmunity','onAnyTryHit','onAllyTryHit','onFoeTryHit','onSourceTryHit','onDeductPP'];console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>{const e=d.abilities.get(n);return e.exists&&e.gen<=3&&!hooks.some(k=>e[k]!==undefined)})));"
  old=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root()),json.dumps(self.metadata['vocabulary']['abilities'])],text=True));new=self.thunder.device.ground_immunity.abilities.tolist();self.assertEqual([i for i,(a,b)in enumerate(zip(old,new))if a!=b],[wa])

 def test_current_mixed_ground_receipt_and_all_other_private_predicates_unchanged(self):
  s,t,_,actor,phys,target,move,attempt,ticket,p=self.literal();index=t.count-1;c=t.execution.thunder_wave_immunity[0];self.assertEqual(cb.owned_ground_immunity(t,index).tolist(),[True]*8)
  self.assertEqual(risk._captured_thunder_wave_matches(t,self.runtime['kernel_risk_compiled'].device,risk.CapturedThunderWave(self.runtime['thunder_wave_context'],ticket,actor,phys,target,move,attempt)).tolist(),[True]*8)
  for field,value in [('resolved_type',p.resolved_type.at[0].add(1)),('caller',p.caller.at[0].add(1)),('immune',p.immune.at[0].set(False)),('absorbed',p.absorbed.at[0].set(True))]:
   bad=t._replace(execution=t.execution._replace(thunder_wave_immunity=(c._replace(pending=p._replace(**{field:value})),)));self.assertEqual(cb.owned_ground_immunity(bad,index).tolist(),[False]+[True]*7,field)
  for field in('types','status','hp','item','base_ability','pp'):
   before=getattr(p.before,field);change=before.at[0].add(1);after=c.after._replace(**{field:change});bad=t._replace(execution=t.execution._replace(thunder_wave_immunity=(c._replace(after=after),)));self.assertEqual(cb.owned_ground_immunity(bad,index).tolist(),[False]+[True]*7,field)
  for field in('engine_binding','source_binding','move_binding','producer_binding'):
   with self.assertRaises(ValueError):cb.owned_ground_immunity(t._replace(**{field:j.broadcast_to(getattr(t,field),(8,8))}),index)
  for file in('transition.py','engine_retained_encoder_driver.py','engine_kernel_frame_risk.py','engine_full_v4_event_device.py','engine_lifecycle_event_device.py'):
   # Certify the original dormant callback delta separately from the
   # complete integrated source, including later device tracker hooks.
   self.assertEqual(subprocess.check_output(['git','show','856c49f8:gpu_engine/'+file]),subprocess.check_output(['git','show',BASE+':gpu_engine/'+file]))
   # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
   if file!='transition.py':self.assertEqual(Path('gpu_engine',file).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+file]))
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_thunder_wave_callbacks.py'],text=True));new=ast.parse(subprocess.check_output(['git','show','856c49f8:gpu_engine/engine_thunder_wave_callbacks.py'],text=True));functions=lambda tree:{n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name!='compile_thunder_wave'};self.assertEqual(functions(old),functions(new))
  integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/engine_thunder_wave_callbacks.py'],text=True));self.assertEqual(ast.dump(ast.parse(Path(cb.__file__).read_text())),ast.dump(integrated))

 # One existing BOTH-owner original word component, with the new mixed target
 # abilities supplied by literal() above. No engine turn or FULL matrix.
 test_original_both_full_plain_immunity=original.ThunderGroundTests.test_02_original_both_full_plain_immunity

if __name__=='__main__':unittest.main()
