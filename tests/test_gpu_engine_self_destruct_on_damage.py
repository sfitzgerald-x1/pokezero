"""Original source-only capability narrowing; no models/engine or FULL repeat."""
import ast,hashlib,json,subprocess,unittest
from pathlib import Path
from _showdown_root import showdown_root
class SelfDestructOnDamageCapabilityTests(unittest.TestCase):
 def test_actual_compiler_predicate_excludes_original_damage_callbacks(self):
  source=Path('gpu_engine/engine_self_destruct_callbacks.py').read_text();fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='compile_data');script=ast.literal_eval(next(n.value for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='script'for t in n.targets)))
  # Evaluate the literal function used for BOTH original compiler map vectors.
  predicate=next(line for line in script.splitlines()if line.startswith('const quiet='))
  node="""const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),crypto=require('crypto');"""+predicate+"""
const calls=[],lines=[],f=d.items.get('focusband'),r=d.abilities.get('rockhead'),other=d.items.get('leftovers'),safe=d.abilities.get('runaway');
const state={hp:100,toString:()=> 'p2a: Target'},fake={randomChance:(n,z)=>{calls.push([n,z]);return true;},add:(...x)=>lines.push(x.map(String).join('|'))};
const adjusted=f.onDamage.call(fake,100,state,null,{id:'explosion',effectType:'Move'});
const result={focusband:{exists:f.exists,gen:f.gen,quiet:quiet(f),callback_sha256:crypto.createHash('sha256').update(f.onDamage.toString()).digest('hex'),adjusted},rockhead:{exists:r.exists,gen:r.gen,onDamage:typeof r.onDamage,quiet:quiet(r)},leftovers:quiet(other),runaway:quiet(safe),calls,lines};console.log(JSON.stringify(result));"""
  v=json.loads(subprocess.check_output(['node','-e',node,str(showdown_root())],text=True));Path('/tmp/pokezero-self-destruct-on-damage-predicate-'+('closed'if not v['focusband']['quiet']else'parent-open')+'.json').write_text(json.dumps(v,sort_keys=True)+'\n')
  self.assertTrue(v['focusband']['exists']);self.assertEqual(v['focusband']['gen'],2);self.assertEqual(v['focusband']['callback_sha256'],'1656b8c4bc28e2ecea442479fe6b2646f475150a6a32b4307f9cd7c0441dc247');self.assertEqual(v['focusband']['adjusted'],99);self.assertEqual(v['calls'],[[1,10]]);self.assertEqual(v['lines'],['-activate|p2a: Target|item: Focus Band'])
  self.assertFalse(v['focusband']['quiet'],'unreconstructed original onDamage must not be a quiet self-KO item');self.assertFalse(v['rockhead']['quiet'],'the SAME ability predicate must close unported onDamage too');self.assertEqual(v['rockhead']['onDamage'],'function');self.assertTrue(v['leftovers']);self.assertTrue(v['runaway'])
if __name__=='__main__':unittest.main()
