"""Pure source maintenance checks; no runtime, device or gameplay imports."""
import ast,hashlib,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='4557de220d2fe51d0b5408577ac350a5bfd89fa9'
def constant(file,key):
 tree=ast.parse((ROOT/'gpu_engine'/file).read_text())
 return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==key for t in n.targets)))
class DoubleEdgeSourceBindingTests(unittest.TestCase):
 def test_all_three_domain_guards_and_every_owned_mapping_pin(self):
  modules='action_history absorption baton combat critical damage_moves effects encore entry focus_punch items linked_traps mechanics phazing pressure pursuit secondary self_destruct sleep_talk solar_beam state sustain transition variable_moves weather'.split()
  domain=hashlib.sha256(b''.join(n.encode()+b'\0'+(ROOT/'gpu_engine'/(n+'.py')).read_bytes()+b'\0'for n in modules)).hexdigest()
  for file,key in [('conditional_law.py','KERNEL_SHA256'),('chance_consumers.py','DOMAIN_SHA256'),('ordinary_law.py','DOMAIN_SHA256')]:self.assertEqual(constant(file,key),domain,(file,key))
  for file,key in [('engine_source_frame_witness.py','SOURCE_PINS'),('chance_consumers.py','SOURCE_PINS'),('conditional_law.py','MAPPING_PINS'),('ordinary_law.py','MAPPING_PINS')]:
   for name,want in constant(file,key).items():self.assertEqual(hashlib.sha256((ROOT/'gpu_engine'/name).read_bytes()).hexdigest(),want,(file,name))
 def test_literal_only_repair_leaves_all_guard_and_helper_bodies_exact(self):
  for file in ('chance_consumers.py','ordinary_law.py'):
   old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/'+file],cwd=ROOT,text=True));new=ast.parse((ROOT/'gpu_engine'/file).read_text())
   def body(t):return [ast.dump(n)for n in t.body if not(isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id in ('DOMAIN_SHA256','MAPPING_PINS')for x in n.targets))]
   self.assertEqual(body(old),body(new),file)
  for file in ('engine_no_damage_callbacks.py','engine_recoil_callbacks.py','engine_kernel_frame_risk.py','engine_source_frame_witness.py','transition.py','conditional_law.py','engine_full_v4_event_device.py','engine_lifecycle_event_device.py','engine_execution_completeness.py','engine_retained_encoder_driver.py','state.py'):
   self.assertEqual((ROOT/'gpu_engine'/file).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+file],cwd=ROOT),file)
if __name__=='__main__':unittest.main()
