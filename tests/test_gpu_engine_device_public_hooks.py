"""Public write fixtures only; no numerical engine turn/model/game."""
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from typing import NamedTuple
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_device_public_hooks as h, engine_device_trackers as tr
from gpu_engine import transition,secondary
from gpu_engine.state import empty_state

BASE='db4ae97b6e560bb69c6c0bf9c845c82535ec0930'
class Belief(NamedTuple):
 species_initial_mask:object;move_mask:object;ability_mask:object;item_mask:object
 species_types:object;public_move:object;source_move_evidence:object

def state(b=2):
 s=empty_state(b);shape=(b,2,6)
 return s._replace(species=j.broadcast_to(j.arange(1,13,dtype=j.int32).reshape(1,2,6),shape),level=j.full(shape,80,j.int32),gender=j.ones(shape,j.int32),hp=j.full(shape,200,j.int32),max_hp=j.full(shape,200,j.int32),done=j.zeros(b,bool))
def tables():
 pool=j.full((20,2),j.uint32(15));m=j.broadcast_to(pool[:,None],(20,10,2));a=m;i=m
 m=m.at[:,2].set(j.uint32(3));a=a.at[:,3].set(j.uint32(5));i=i.at[:,4].set(j.uint32(9))
 return Belief(pool,m,a,i,j.tile(j.array([[1,2]],j.int32),(20,1)),j.arange(10,dtype=j.int32).at[8].set(2),j.ones(10,bool).at[0].set(False).at[9].set(False))

class PublicHooksTests(unittest.TestCase):
 def test_actual_declarations_masks_private_noninterference_and_public_types(self):
  s=state();b=tables()
  @jax.jit
  def f(s):
   c=h.initialize(s,b)
   h.move(c,s,1,j.array([8,9],j.int32),j.ones(2,bool),pp_use=True)
   h.reveal_ability(c,s,1,j.array([True,False]),ability=j.full(2,3,j.int32))
   h.reveal_item(c,s,1,j.full(2,4,j.int32),j.array([True,False]))
   return c.trackers,tr.readout(c.trackers,j.zeros(2,j.int32))
  t,v=f(s)
  self.assertEqual(v.revealed_moves[:,0,0].tolist(),[2,0]);self.assertEqual(v.move_pp_used[:,0,0].tolist(),[1,0])
  self.assertEqual(v.belief_mask[:,0,0].tolist(),[1,15]);self.assertEqual(v.types[:,0].tolist(),[[1,2]]*2)
  changed=s._replace(ability=j.full_like(s.ability,7),item=j.full_like(s.item,7),types=j.full_like(s.types,9),moves=j.full_like(s.moves,9),pp=j.full_like(s.pp,1),stats=j.full_like(s.stats,999))
  _,v2=f(changed)
  for x,y in zip(jax.tree.leaves(v),jax.tree.leaves(v2)):np.testing.assert_array_equal(x,y)
  self.assertFalse(bool(v.present[:,1:].any()));self.assertFalse(bool(v.ability[1].any()))
 def test_multiple_actual_installs_and_borrowed_reveal_does_not_narrow_original(self):
  s=state();c=h.initialize(s,tables());yes=j.ones(2,bool)
  h.reveal_ability(c,s,1,yes,ability=j.full(2,3,j.int32))
  old=c.trackers.belief_mask.copy()
  h._write(c.trackers,'borrowed_ability',1,s.active[:,1],j.full(2,7,j.int32),yes) # immutable return deliberately ignored
  c.trackers=h._write(c.trackers,'borrowed_ability',1,s.active[:,1],j.full(2,7,j.int32),yes)
  h.reveal_ability(c,s,1,yes,ability=j.full(2,7,j.int32),original=False)
  np.testing.assert_array_equal(c.trackers.belief_mask,old)
  for p in (3,4,0):
   s=s._replace(active=s.active.at[:,1].set(p));h.install(c,s,1,yes)
  self.assertEqual(c.trackers.next_slot[:,1].tolist(),[3,3])
  self.assertEqual(c.trackers.own_order[0,1].tolist(),[0,1,2,4,3,5])
  self.assertEqual(tr.readout(c.trackers,j.zeros(2,j.int32)).species[0,:3].tolist(),[7,10,11])
 def test_transient_status_lum_and_reflected_ability_are_hooked_before_cure(self):
  s=state()._replace(item=state().item.at[:,1,0].set(4),status=state().status.at[:,1,0].set(1))
  b=tables();c=h.initialize(s,b)
  ability=j.zeros(10,bool).at[3].set(True)
  sec=NS(synchronize=ability,ability_immunity=j.zeros((10,7),bool),type_immunity=j.zeros((7,10),bool))
  s=s._replace(ability=s.ability.at[:,1,0].set(3));items=NS(lum=j.zeros(10,bool).at[4].set(True))
  out=secondary.after_status_event(s,0,1,j.full(2,1,j.int32),sec,applied=j.ones(2,bool),item_table=items,_device_public=c)
  self.assertEqual(out.status[:,0,0].tolist(),[1,1]);self.assertEqual(out.status[:,1,0].tolist(),[0,0])
  self.assertEqual(c.trackers.status[:,0,0].tolist(),[1,1]);self.assertEqual(c.trackers.status[:,1,0].tolist(),[0,0])
  self.assertEqual(c.trackers.shown_ability[:,1,0].tolist(),[3,3]);self.assertEqual(c.trackers.shown_item[:,1,0].tolist(),[4,4]);self.assertEqual(c.trackers.item_removed[:,1,0].tolist(),[True,True])
  # Public consumed item evidence survives even though current item is gone.
  self.assertEqual(c.trackers.belief_mask[:,1,0,0].tolist(),[1,1])
 def test_observer_relative_damage_and_known_party_cure(self):
  s=state();s=s._replace(hp=s.hp.at[:,1,0].set(199),max_hp=s.max_hp.at[:,1,0].set(513))
  c=h.initialize(s,tables());after=s._replace(hp=s.hp.at[:,1,0].set(99))
  h.damage(c,after,0,j.full(2,100,j.int32),j.full(2,100,j.int32),j.ones(2,bool));h.observe(c,after,j.ones(2,bool))
  h.public_turn(c,after,j.ones(2,bool),c.trackers.turn+1)
  a=tr.readout(c.trackers,j.zeros(2,j.int32));b=tr.readout(c.trackers,j.ones(2,j.int32))
  np.testing.assert_allclose(a.own_mon_features['last_damage_dealt'][:,0],.19)
  np.testing.assert_allclose(b.mon_features['last_damage_dealt'][:,0],100/513)
  np.testing.assert_allclose(a.mon_features['last_damage_taken'][:,0],.19)
  np.testing.assert_allclose(b.own_mon_features['last_damage_taken'][:,0],100/513)
  hidden=after._replace(status=after.status.at[:,1,5].set(4));cured=hidden._replace(status=j.zeros_like(hidden.status))
  h.party_cure(c,hidden,cured,1,j.ones(2,bool));self.assertFalse(bool(c.trackers.present[:,1,1:].any()))
 def test_actual_entry_trace_weather_and_baton_copied_public_state(self):
  s=state()._replace(ability=state().ability.at[:,0,0].set(3).at[:,1,0].set(7))
  c=h.initialize(s,tables());yes=j.ones(2,bool);flag=j.zeros(10,bool)
  entry=NS(weather_ability=j.zeros(10,j.int32),intimidate=flag,attack_drop_immune=flag,trace=flag.at[3].set(True),own_tempo=flag)
  after=s._replace(ability=s.ability.at[:,0,0].set(7))
  h.entry(c,s,after,0,entry,yes)
  self.assertEqual(c.trackers.original_ability[:,0,0].tolist(),[3,3]);self.assertEqual(c.trackers.shown_ability[:,0,0].tolist(),[7,7])
  view=tr.readout(c.trackers,j.ones(2,j.int32))
  self.assertEqual(view.ability[:,0].tolist(),[3,3]);self.assertEqual(view.borrowed_ability[:,0].tolist(),[7,7])
  self.assertEqual(view.borrowed_ability.dtype,j.int32)
  self.assertEqual(c.trackers.borrowed_ability[:,0,0].tolist(),[7,7]);self.assertEqual(c.trackers.belief_mask[:,0,0,0].tolist(),[5,5])
  h.volatile(c,after,0,'substitute',yes,yes);h.feature(c,0,after.active[:,0],'sub_present',j.ones(2,j.int32),yes)
  nextstate=after._replace(active=after.active.at[:,0].set(3))
  h.install(c,nextstate,0,yes);h.baton_copy(c,nextstate,0,j.zeros(2,j.int32),yes);h.withdraw(c,nextstate,0,j.zeros(2,j.int32),yes)
  v=tr.readout(c.trackers,j.ones(2,j.int32))
  self.assertTrue(bool(v.volatiles[:,1,h.VOLATILE_NAMES.index('substitute')].all()))
  self.assertFalse(bool(v.volatiles[:,0].any()));self.assertEqual(v.mon_features['sub_present'][:,1].tolist(),[1,1])
 def test_handler_private_changes_are_not_generic_reveals_and_inactive_writes_stay_local(self):
  s=state();c=h.initialize(s,tables());no=j.zeros(2,bool)
  h.reveal_ability(c,s,1,no);h.reveal_item(c,s,1,j.full(2,7,j.int32),no)
  after=s._replace(hp=s.hp.at[0,1,0].set(1),status=s.status.at[0,1,0].set(4))
  h.observe(c,after,j.array([True,False]))
  self.assertEqual(c.trackers.status[:,1,0].tolist(),[4,0]);self.assertFalse(bool(c.trackers.shown_ability.any()));self.assertFalse(bool(c.trackers.shown_item.any()))
  # Root-world clones carry exactly existing public leaves, never a sampled world.
  out=tr.clone_for_worlds(c.trackers,j.array([1,0,1],j.int32))
  for x,y in zip(jax.tree.leaves(c.trackers),jax.tree.leaves(out)):np.testing.assert_array_equal(y,np.asarray(x)[[1,0,1]])
 def test_public_pressure_charge_and_actual_known_bench_cure(self):
  s=state(6);yes=j.ones(6,bool);r=j.arange(6)
  base=tables();b=NS(**base._asdict(),public_foe_targeted=j.ones(10,bool).at[4].set(False),
       public_pressure=j.zeros(10,bool).at[2].set(True))
  # Actual private Pressure is deliberately present on EVERY foe. Only public
  # native or copied Pressure can double a caller's source-targeted charge.
  s=s._replace(ability=s.ability.at[:,1,0].set(2))
  c=h.initialize(s,b)
  h.reveal_ability(c,s,1,j.array([False,True,False,False,True,True]),ability=j.full(6,2,j.int32))
  h.reveal_ability(c,s,1,j.array([False,False,True,True,False,False]),ability=j.full(6,3,j.int32))
  c.trackers=h._write(c.trackers,'borrowed_ability',1,s.active[:,1],j.full(6,2,j.int32),j.array([False,False,True,False,False,False]))
  move=j.array([2,2,2,2,4,2],j.int32)
  h.move(c,s,0,move,yes,pp_use=j.array([True,True,True,True,True,False]))
  self.assertEqual(c.trackers.pp_used[:,0,0,0].tolist(),[1,2,2,1,1,0])
  # An observed bench member's genuine party cure clears only its public
  # major-status clocks. Unseen members do not become public encounters.
  asleep=s._replace(active=s.active.at[:,1].set(3),status=s.status.at[:,1,3].set(5))
  h.install(c,asleep,1,yes)
  h.feature(c,1,j.full(6,3,j.int32),'sleep_turns',j.full(6,3,j.int32),yes)
  h.feature(c,1,j.full(6,3,j.int32),'rest_sleep',j.ones(6,j.int32),yes)
  back=asleep._replace(active=asleep.active.at[:,1].set(0),status=asleep.status.at[:,1,5].set(5))
  h.install(c,back,1,yes)
  cured=back._replace(status=back.status.at[:,1,3].set(0).at[:,1,5].set(0))
  h.party_cure(c,back,cured,1,yes)
  view=tr.readout(c.trackers,j.zeros(6,j.int32))
  self.assertFalse(bool(view.mon_features['sleep_turns'][:,1].any()))
  self.assertFalse(bool(view.mon_features['rest_sleep'][:,1].any()))
  self.assertFalse(bool(view.present[:,2:].any()))
 def test_explicit_chance_same_original_call_and_sample_once(self):
  s=state();c=h.initialize(s);a=j.zeros((2,2),j.int32);chance=object();sentinel=object()
  with self.assertRaisesRegex(ValueError,'initialize_with_public'):
   transition.step_with_public(s,a,None,tr.initialize(s),chance_inputs=chance)
  with patch.object(transition,'sample_chance_inputs',return_value=chance) as sample,patch.object(transition,'step_with_chance',return_value=sentinel) as step:
   self.assertIs(transition.step_with_public(s,a,None,c.trackers),sentinel);sample.assert_called_once_with(s)
   self.assertIs(step.call_args.args[3],chance)
   sample.reset_mock();step.reset_mock()
   self.assertIs(transition.step_with_public(s,a,None,c.trackers,chance_inputs=chance),sentinel)
   sample.assert_not_called();self.assertIs(step.call_args.args[3],chance)
   self.assertIs(step.call_args.kwargs['_device_public'].trackers,c.trackers)
 def test_default_physical_ast_exact_after_only_optin_erasure(self):
  class Erase(ast.NodeTransformer):
   def visit_If(self,n):
    if any(isinstance(x,ast.Name) and x.id=='_device_public' for x in ast.walk(n.test)):return None
    return self.generic_visit(n)
   def visit_Call(self,n):
    n=self.generic_visit(n);n.keywords=[k for k in n.keywords if k.arg!='_device_public']
    if isinstance(n.func,ast.Name) and n.func.id=='MoveTable':
     n.args=[a for a in n.args if not(isinstance(a,ast.Call) and isinstance(a.func,ast.Attribute)
      and isinstance(a.func.value,ast.Name) and a.func.value.id=='public_port' and a.func.attr=='compile_tables')]
    return n
   def visit_ClassDef(self,n):
    n=self.generic_visit(n)
    if n.name=='MoveTable':
     n.body=[a for a in n.body if not(isinstance(a,ast.AnnAssign) and isinstance(a.target,ast.Name) and a.target.id=='public_port')]
    return n
   def visit_arguments(self,n):
    n=self.generic_visit(n);pairs=[(a,d) for a,d in zip(n.kwonlyargs,n.kw_defaults) if a.arg!='_device_public']
    n.kwonlyargs=[a for a,d in pairs];n.kw_defaults=[d for a,d in pairs];return n
  for name in ('transition.py','secondary.py'):
   path='gpu_engine/'+name;base=ast.parse(subprocess.check_output(['git','show',BASE+':'+path],text=True));current=ast.parse(Path(path).read_text())
   current.body=[n for n in current.body if not(isinstance(n,ast.ImportFrom) and any(a.name in ('engine_device_public_hooks','engine_device_public_port') for a in n.names)) and not(isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('step_with_public','initialize_with_public'))]
   current=Erase().visit(current)
   self.assertEqual(ast.dump(current,include_attributes=False),ast.dump(base,include_attributes=False),name)

if __name__=='__main__':unittest.main()
