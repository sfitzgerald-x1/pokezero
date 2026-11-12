"""Pure source/pin and default numerical mutation controls."""
import ast,hashlib,subprocess,unittest
from pathlib import Path
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
ROOT=Path('gpu_engine')
def val(path,name):
 tree=ast.parse((ROOT/path).read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets));return ast.literal_eval(node.value)
class SleepTalkRestSourceTests(unittest.TestCase):
 def test_all_actual_pins_and_domains(self):
  for file,key in [('engine_source_frame_witness.py','SOURCE_PINS'),('chance_consumers.py','SOURCE_PINS'),('ordinary_law.py','MAPPING_PINS'),('conditional_law.py','MAPPING_PINS')]:
   for name,digest in val(file,key).items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,(file,name))
  modules=tuple('action_history absorption baton combat critical damage_moves effects encore entry focus_punch items linked_traps mechanics phazing pressure pursuit secondary self_destruct sleep_talk solar_beam state sustain transition variable_moves weather'.split())
  domain=hashlib.sha256(b''.join(n.encode()+b'\0'+(ROOT/(n+'.py')).read_bytes()+b'\0'for n in modules)).hexdigest()
  for file,key in [('conditional_law.py','KERNEL_SHA256'),('chance_consumers.py','DOMAIN_SHA256'),('ordinary_law.py','DOMAIN_SHA256')]:self.assertEqual(val(file,key),domain)
 def test_default_numerics_and_detect_real_mutations(self):
  old=subprocess.check_output(['git','show','3a1655ad73da6f61fa5cbe5961c40600587162dc:gpu_engine/transition.py'],text=True)
  historical=ast.parse(subprocess.check_output(['git','show','76ef9e6b:gpu_engine/transition.py'],text=True))
  baseline=dump_default_program(ast.parse(old));self.assertEqual(dump_default_program(historical),baseline);assert_hp_rng_mutations_detected(self,historical,baseline)
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components; mechanics, sleep_talk, state, sustain and effects: byte-exact against reviewed SOURCE_PINS in transition_observer_wiring's source-binding subtests.
  for name in ('mechanics','sleep_talk','state','sustain','effects'):
   historical=subprocess.check_output(['git','show','76ef9e6b:gpu_engine/'+name+'.py'])
   self.assertEqual(historical,subprocess.check_output(['git','show','3a1655ad73da6f61fa5cbe5961c40600587162dc:gpu_engine/'+name+'.py']))
if __name__=='__main__':unittest.main()
