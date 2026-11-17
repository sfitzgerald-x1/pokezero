"""Pure source maintenance checks; no runtime, device or gameplay imports."""
import ast,hashlib,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='4557de220d2fe51d0b5408577ac350a5bfd89fa9'
REPAIR='77288c9d88030754cdc1164a4f29ffeb69473d3d'
def constant(file,key):
 tree=ast.parse((ROOT/'gpu_engine'/file).read_text())
 return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==key for t in n.targets)))
class DoubleEdgeSourceBindingTests(unittest.TestCase):
 def test_device_port_recertification_keeps_guards_and_inventory(self):
  # Re-certify the already integrated device/driver source, without changing
  # the consumer inventory, partition math, mapper or source-drift rejection.
  # Baseline: the integration commit main (#1819) merged into. Main certified b8092a85; after it, the
  # reviewed Fidelity R and B batches changed the chance_consumers.py and phase_d_chance.py bodies.
  # The other three files have the same body and mappings at both commits.
  baseline='0a3711dc9a833790a7b54547dda61cb4547cc88b'
  # Fidelity M2 (#1930, covered by its Opus review) re-certifies three bodies at its own reviewed commits; their
  # mappings are unchanged. 7cc1d9f6 changed chance_consumers.py's and phase_d_chance.py's CONSUMERS rows (Wrap's
  # duration on draws.{s}.7, Yawn's sleep on effect column 2, the Wrap and Perish pair ties on effect column 3) and
  # plan_for_state's draws[s][7] for a selected Wrap. 7476bce7 changed engine_source_frame_witness.py's Encore cutpoint
  # geometry (12 to 16, one per residual event). No later commit (M2's R1 included) changes these bodies.
  # §6 (the effect-coordinate chance proof) re-certifies four bodies and the pinned-file inventories at its own commits:
  # 72670171 and 3b0cc40e (plan_for_state's effect axes and SUSPENDED_ROOT_SAVED_BLOCKS, conditional_law's saved
  # effect block, residual_order.py in SOURCE_PINS and the domain lists) and 53dc3add (ordinary_law's effect bank).
  # Per-module domain pins (cf0bb33d; the plan gates' domain identity restored at a1b0218a) re-certify the four gate bodies: each aggregate DOMAIN_SHA256 / KERNEL_SHA256
  # literal became a per-module DOMAIN_PINS / KERNEL_PINS table, the gates check it module by module (and the table
  # against the module list, in order), and chance_consumers' tables moved to chance_consumers_pins.py, whose
  # inventory is bound here too. The witness body is unchanged (its SOURCE_PINS only took one entry per line).
  # 5166db32 binds each plan identity to the same source snapshot its per-module gate validated.
  certified={'chance_consumers.py':'5166db320a7681c6e0181409adc2425c176ece9c',
             'chance_consumers_pins.py':'cf0bb33db8c2ab0247c8c0988538eb29811d0a9b',
             'phase_d_chance.py':'5166db320a7681c6e0181409adc2425c176ece9c',
             'conditional_law.py':'cf0bb33db8c2ab0247c8c0988538eb29811d0a9b',
             'engine_source_frame_witness.py':'3b0cc40e6709bd19fe70a6295580a51590dd4f57',
             'ordinary_law.py':'cf0bb33db8c2ab0247c8c0988538eb29811d0a9b'}
  constants={'SOURCE_PINS','MAPPING_PINS','DOMAIN_PINS','KERNEL_PINS','DOMAIN_SHA256','KERNEL_SHA256'}
  for file in ('chance_consumers.py','chance_consumers_pins.py','ordinary_law.py','conditional_law.py',
               'engine_source_frame_witness.py','phase_d_chance.py'):
   old=ast.parse(subprocess.check_output(['git','show',certified.get(file,baseline)+':gpu_engine/'+file],cwd=ROOT,text=True))
   current=ast.parse((ROOT/'gpu_engine'/file).read_text())
   def body(tree):
    return [ast.dump(node) for node in tree.body if not (
     isinstance(node,ast.Assign) and any(isinstance(target,ast.Name)
      and target.id in constants for target in node.targets))]
   self.assertEqual(body(old),body(current),file)
   def mappings(tree):
    return {target.id:tuple(ast.literal_eval(node.value)) for node in tree.body
     if isinstance(node,ast.Assign) for target in node.targets
     if isinstance(target,ast.Name) and target.id in ('SOURCE_PINS','MAPPING_PINS','DOMAIN_PINS','KERNEL_PINS')}
   self.assertEqual(mappings(old),mappings(current),file)
 def test_all_three_domain_guards_and_every_owned_mapping_pin(self):
  modules='action_history absorption baton combat critical damage_moves effects encore entry focus_punch items linked_traps mechanics phazing pressure pursuit residual_order secondary self_destruct sleep_talk solar_beam state sustain transform transition variable_moves weather'.split()
  # the per-module domain pins (they replaced one KERNEL_SHA256 / DOMAIN_SHA256 over these modules): every module, in order
  domain=[(n+'.py',hashlib.sha256((ROOT/'gpu_engine'/(n+'.py')).read_bytes()).hexdigest())for n in modules]
  for file,key in [('conditional_law.py','KERNEL_PINS'),('chance_consumers_pins.py','DOMAIN_PINS'),('ordinary_law.py','DOMAIN_PINS'),('phase_d_chance.py','DOMAIN_PINS')]:self.assertEqual(list(constant(file,key).items()),domain,(file,key))
  for file,key in [('engine_source_frame_witness.py','SOURCE_PINS'),('chance_consumers_pins.py','SOURCE_PINS'),('conditional_law.py','MAPPING_PINS'),('ordinary_law.py','MAPPING_PINS')]:
   for name,want in constant(file,key).items():self.assertEqual(hashlib.sha256((ROOT/'gpu_engine'/name).read_bytes()).hexdigest(),want,(file,name))
 def test_literal_only_repair_leaves_all_guard_and_helper_bodies_exact(self):
  # This regression binds the historical repair, not later reviewed source batches.
  # Current source pins/domains are independently checked above.
  for file in ('chance_consumers.py','ordinary_law.py'):
   old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/'+file],cwd=ROOT,text=True));new=ast.parse(subprocess.check_output(['git','show',REPAIR+':gpu_engine/'+file],cwd=ROOT,text=True))
   def body(t):return [ast.dump(n)for n in t.body if not(isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id in ('DOMAIN_SHA256','MAPPING_PINS')for x in n.targets))]
   self.assertEqual(body(old),body(new),file)
  for file in ('engine_no_damage_callbacks.py','engine_recoil_callbacks.py','engine_kernel_frame_risk.py','engine_source_frame_witness.py','transition.py','conditional_law.py','engine_full_v4_event_device.py','engine_lifecycle_event_device.py','engine_execution_completeness.py','engine_retained_encoder_driver.py','state.py'):
   self.assertEqual(subprocess.check_output(['git','show',REPAIR+':gpu_engine/'+file],cwd=ROOT),subprocess.check_output(['git','show',BASE+':gpu_engine/'+file],cwd=ROOT),file)
if __name__=='__main__':unittest.main()
