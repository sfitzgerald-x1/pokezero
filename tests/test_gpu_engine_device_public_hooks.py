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
  # Schema v3: the same installs in public-slot coordinates (slots 1, 2, then 0).
  self.assertEqual(c.trackers.public_order[0,1].tolist(),[0,2,1,3,4,5])
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
  # These PP facts are read by seat1 about seat0. Seat1 knows its own active
  # ability even when the Pressure start line was not shared with seat0.
  s=s._replace(ability=s.ability.at[:,1,0].set(j.array([2,2,2,3,2,2],j.int32)))
  c=h.initialize(s,b)
  h.reveal_ability(c,s,1,j.array([False,True,False,False,True,True]),ability=j.full(6,2,j.int32))
  h.reveal_ability(c,s,1,j.array([False,False,True,True,False,False]),ability=j.full(6,3,j.int32))
  c.trackers=h._write(c.trackers,'borrowed_ability',1,s.active[:,1],j.full(6,2,j.int32),j.array([False,False,True,False,False,False]))
  move=j.array([2,2,2,2,4,2],j.int32)
  h.move(c,s,0,move,yes,pp_use=j.array([True,True,True,True,True,False]))
  self.assertEqual(c.trackers.pp_used[:,0,0,0].tolist(),[2,2,2,1,1,0])
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
 def test_encore_bag_bit_follows_the_engine_counter_and_clears_at_the_residual_end(self):
  # The device harness's OPEN BUG C: transition.py's Encore residual calls no hook, so before this the bit set at the
  # Encore move stayed after -end|X|Encore. derived_volatiles (run by turn_end) now writes encore = counter>0.
  E=h.VOLATILE_NAMES.index('encore')
  def bit(c,st,side):
   r=np.arange(2);p=np.asarray(st.active)[:,side];slot=np.clip(np.asarray(c.trackers.physical_to_public)[r,side,p],0,5)
   return np.asarray(c.trackers.public_volatiles)[r,side,slot,E].tolist()
  s=state();c=h.initialize(s,tables());yes=j.ones(2,bool)
  counter=lambda st,side,n:st._replace(encore_counter=st.encore_counter.at[:,side,0].set(j.asarray(n,j.int32)))
  held=counter(s,1,[2,2]);c=h.volatile(c,held,1,'encore',yes,yes)                 # the Encore move's write
  self.assertEqual(bit(c,held,1),[True,True])
  c=h.turn_end(c,held,held,yes);self.assertEqual(bit(c,held,1),[True,True])       # counter still above 0: kept
  ended=counter(held,1,[0,1])                                                       # lane 0's Encore ends at the residual
  c=h.turn_end(c,held,ended,yes);self.assertEqual(bit(c,ended,1),[False,True])
  self.assertEqual(bit(c,ended,0),[False,False])                                    # the other side untouched
  # a done lane is left alone; a counter that starts sets the bit (it agrees with the move's write)
  c2=h.initialize(s,tables());started=counter(s,0,[3,3])
  c2=h.derived_volatiles(c2,started,j.array([True,False]));self.assertEqual(bit(c2,started,0),[True,False])
  # the faint clear zeroes the engine counter (transition's faintMessages), so a fainted holder reads no Encore
  fainted=ended._replace(hp=ended.hp.at[:,1,0].set(0),encore_counter=ended.encore_counter.at[:,1,0].set(0))
  c=h.turn_end(c,ended,fainted,yes);self.assertEqual(bit(c,fainted,1),[False,False])
  # a Baton Pass recipient: Encore is noCopy (baton.py zeroes the recipient's counter, baton_copy omits the bit)
  c3=h.initialize(s,tables());c3=h.volatile(c3,held,1,'encore',yes,yes)
  passed=held._replace(active=held.active.at[:,1].set(3))
  c3=h.install(c3,passed,1,yes);c3=h.baton_copy(c3,passed,1,j.zeros(2,j.int32),yes);c3=h.withdraw(c3,passed,1,j.zeros(2,j.int32),yes)
  c3=h.turn_end(c3,held,passed,yes);self.assertEqual(bit(c3,passed,1),[False,False])
 def test_flashfire_bag_bit_follows_the_engine_bit_while_its_holder_lives(self):
  # The device harness's OPEN BUG E: no hook wrote B2's flashfire entry (-start|X|ability: Flash Fire). derived_volatiles
  # (run by public_move_state and turn_end) now writes flashfire = the engine's absorption.FLASH_FIRE bit and hp>0.
  from gpu_engine import baton
  from gpu_engine.absorption import FLASH_FIRE
  F=h.VOLATILE_NAMES.index('flashfire');yes=j.ones(2,bool);zero=j.zeros(2,j.int32)
  def bit(c,st,side):
   r=np.arange(2);p=np.asarray(st.active)[:,side];slot=np.clip(np.asarray(c.trackers.physical_to_public)[r,side,p],0,5)
   return np.asarray(c.trackers.public_volatiles)[r,side,slot,F].tolist()
  lit=lambda st,lanes:st._replace(volatile=st.volatile.at[:,1,0].set(j.where(j.asarray(lanes),j.uint32(FLASH_FIRE),j.uint32(0))))
  s=state();c=h.initialize(s,tables())
  fired=lit(s,[True,False])                                                         # lane 0's Fire move is absorbed
  c=h.turn_end(c,s,fired,yes);self.assertEqual(bit(c,fired,1),[True,False]);self.assertEqual(bit(c,fired,0),[False,False])
  again=lit(fired,[True,True])                                                      # lane 0: a second Fire move (-immune)
  c=h.turn_end(c,fired,again,yes);self.assertEqual(bit(c,again,1),[True,True])
  # the engine's faint leaves the bit; B2's FAINT clear empties the bag, and a later read of the fainted holder agrees
  fainted=again._replace(hp=again.hp.at[0,1,0].set(0))
  c=h.turn_end(c,again,fainted,yes);self.assertEqual(bit(c,fainted,1),[False,True])
  c=h.turn_end(c,fainted,fainted,yes);self.assertEqual(bit(c,fainted,1),[False,True])
  # Baton Pass: flashfire is noCopy; baton.copy_to_recipient carries LEECH_SEED only and baton_copy omits the bag entry
  c2=h.initialize(s,tables());c2=h.turn_end(c2,s,again,yes)
  passed=baton.copy_to_recipient(again,1,j.full(2,3,j.int32),eligible=yes)
  self.assertEqual((np.asarray(passed.volatile)[:,1,3]&FLASH_FIRE).tolist(),[0,0])
  passed=passed._replace(active=passed.active.at[:,1].set(3),volatile=passed.volatile.at[:,1,0].set(0))  # depart
  c2=h.install(c2,passed,1,yes);c2=h.baton_copy(c2,passed,1,zero,yes);c2=h.withdraw(c2,passed,1,zero,yes)
  self.assertEqual(bit(c2,passed,1),[False,False])
  c2=h.turn_end(c2,again,passed,yes);self.assertEqual(bit(c2,passed,1),[False,False])
 def test_fainted_active_keeps_its_public_stages_in_both_views_until_its_replacement_arrives(self):
  # The device harness's OPEN BUG D, by the lead's ruling (observation inputs follow the champion's B2/Python encoder
  # convention, engine state follows Showdown): the engine zeroes a fainted active's stages, B2 and the Python encoder
  # keep them until the replacement arrives. The carry keeps them; the replacement's install and withdraw reset them.
  s=state();yes=j.ones(2,bool);zero=j.zeros(2,j.int32)
  up=s._replace(boosts=s.boosts.at[:,1,0,0].set(2).at[:,1,0,4].set(-1))          # side 1's active: +2 Atk, -1 Spe
  c=h.initialize(s,tables());c=h.observe(c,up,yes)
  # The KO. Lane 0 reads it before the engine's clear, with a last -1 Spe from the same action; lane 1 only after it.
  pre=up._replace(hp=up.hp.at[:,1,0].set(0),boosts=up.boosts.at[0,1,0,4].set(-2))
  c=h.observe(c,pre,j.array([True,False]))
  cleared=pre._replace(boosts=pre.boosts.at[:,1,0].set(0))                         # transition._clear_fainted_boosts
  c=h.turn_end(c,up,cleared,yes)
  kept=[[2,0,0,0,-2,0,0],[2,0,0,0,-1,0,0]]
  foe=tr.readout(c.trackers,zero);own=tr.readout(c.trackers,j.ones(2,j.int32))
  self.assertEqual(np.asarray(foe.boosts)[:,0].tolist(),kept)                      # the opponent's view
  self.assertEqual(np.asarray(own.own_boosts).tolist(),kept)                       # the fainted mon's own view
  self.assertEqual(np.asarray(foe.own_boosts).tolist(),[[0]*7]*2)                  # side 0's own active is untouched
  self.assertEqual(np.asarray(c.trackers.fainted)[:,1,0].tolist(),[True,True])
  # The replacement's arrival resets them: install reads the entrant's engine stages, withdraw zeroes the fainted slot.
  new=cleared._replace(active=cleared.active.at[:,1].set(3))
  c=h.install(c,new,1,yes);c=h.withdraw(c,new,1,zero,yes);c=h.turn_end(c,new,new,yes)
  foe=tr.readout(c.trackers,zero);own=tr.readout(c.trackers,j.ones(2,j.int32))
  self.assertEqual(np.asarray(foe.boosts)[:,1].tolist(),[[0]*7]*2);self.assertEqual(np.asarray(own.own_boosts).tolist(),[[0]*7]*2)
  self.assertEqual(np.asarray(c.trackers.boosts)[:,1,0].tolist(),[[0]*7]*2)
  # Baton Pass, the one stage carrier: the recipient's stages are the engine's copy (baton.copy_to_recipient), read at
  # install. A fainted passer cannot arise (transition sets baton_pass only at hp>0 and the step ends there), and the
  # carry's kept stages never reach a recipient: an install reads only engine state.
  c2=h.initialize(s,tables());c2=h.observe(c2,up,yes)
  passed=up._replace(active=up.active.at[:,1].set(3),boosts=up.boosts.at[:,1,3].set(up.boosts[:,1,0]).at[:,1,0].set(0))
  c2=h.install(c2,passed,1,yes);c2=h.baton_copy(c2,passed,1,zero,yes);c2=h.withdraw(c2,passed,1,zero,yes)
  self.assertEqual(np.asarray(tr.readout(c2.trackers,zero).boosts)[:,1].tolist(),[[2,0,0,0,-1,0,0]]*2)
  self.assertEqual(np.asarray(c2.trackers.boosts)[:,1,0].tolist(),[[0]*7]*2)
  c3=h.initialize(s,tables());c3=h.observe(c3,up,yes);c3=h.observe(c3,cleared,yes)  # a kept fainted carry
  c3=h.install(c3,new,1,yes);c3=h.baton_copy(c3,new,1,zero,yes);c3=h.withdraw(c3,new,1,zero,yes)
  self.assertEqual(np.asarray(tr.readout(c3.trackers,zero).boosts)[:,1].tolist(),[[0]*7]*2)

 def test_a_drag_out_drops_the_slot_stages_so_a_drag_back_into_lethal_spikes_reads_zero(self):
  # Review of D: the hooks' drag never ran withdraw, so a phazed-out mon kept its stages in the slot carry; with the
  # fainted-active carry (_observe keeps a slot's stages at hp 0) a later drag back into lethal Spikes showed them.
  # B2's drag arrival resets the seat's stages (public_mon_state.py:111-112), and so does the Python encoder.
  s=state();yes=j.ones(2,bool);zero=j.zeros(2,j.int32);one=j.ones(2,j.int32)
  up=s._replace(boosts=s.boosts.at[:,1,0,0].set(2))                                 # side 1's lead at +2 Atk
  c=h.initialize(s,tables());c=h.observe(c,up,yes)
  out=up._replace(active=up.active.at[:,1].set(3),boosts=up.boosts.at[:,1,0].set(0))  # Whirlwind: entry.depart zeroes
  c=h.drag(c,up,out,1,None,yes)
  self.assertEqual(np.asarray(c.trackers.boosts)[:,1,0].tolist(),[[0]*7]*2)          # the dragged-out slot
  back=out._replace(active=out.active.at[:,1].set(0),hp=out.hp.at[:,1,0].set(0))     # dragged back, KO'd by Spikes
  c=h.drag(c,out,back,1,None,yes);c=h.turn_end(c,back,back,yes)
  self.assertEqual(np.asarray(tr.readout(c.trackers,zero).boosts)[:,0].tolist(),[[0]*7]*2)   # the opponent's view
  self.assertEqual(np.asarray(tr.readout(c.trackers,one).own_boosts).tolist(),[[0]*7]*2)     # the mon's own view
  self.assertEqual(np.asarray(c.trackers.fainted)[:,1,0].tolist(),[True,True])
  # Baton Pass into lethal Spikes keeps the passed stages, as B2 does (its Baton arrival copies them, its faint keeps
  # them): install reads the alive recipient's engine copy, and the entry's read precedes the engine's clear.
  c=h.initialize(s,tables());c=h.observe(c,up,yes)
  passed=up._replace(active=up.active.at[:,1].set(3),boosts=up.boosts.at[:,1,3].set(up.boosts[:,1,0]).at[:,1,0].set(0))
  c=h.install(c,passed,1,yes);c=h.baton_copy(c,passed,1,zero,yes);c=h.withdraw(c,passed,1,zero,yes)
  spiked=passed._replace(hp=passed.hp.at[:,1,3].set(0))
  c=h.entry(c,passed,spiked,1,None,yes)
  cleared=spiked._replace(boosts=spiked.boosts.at[:,1,3].set(0));c=h.turn_end(c,spiked,cleared,yes)
  self.assertEqual(np.asarray(tr.readout(c.trackers,zero).boosts)[:,1].tolist(),[[2,0,0,0,0,0,0]]*2)
 def test_encore_bag_bit_is_gated_on_a_living_holder(self):
  # The engine zeroes a fainted holder's encore_counter only at the step's end; a derived read before that sees a live
  # counter on a fainted holder, which B2's FAINT clear has already emptied.
  E=h.VOLATILE_NAMES.index('encore');yes=j.ones(2,bool)
  s=state();held=s._replace(encore_counter=s.encore_counter.at[:,1,0].set(2))
  c=h.initialize(s,tables());c=h.derived_volatiles(c,held,yes)
  self.assertEqual(np.asarray(c.trackers.public_volatiles)[:,1,0,E].tolist(),[True,True])
  fainted=held._replace(hp=held.hp.at[0,1,0].set(0))                                  # lane 0: KO'd, counter not yet 0
  c=h.derived_volatiles(c,fainted,yes)
  self.assertEqual(np.asarray(c.trackers.public_volatiles)[:,1,0,E].tolist(),[False,True])

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
   path='gpu_engine/'+name;base=ast.parse(subprocess.check_output(['git','show',BASE+':'+path],text=True))
   # Keep the original hook-only erasure proof on its immutable feature.
   current=ast.parse(subprocess.check_output(['git','show','a6b8da18:'+path],text=True))
   current.body=[n for n in current.body if not(isinstance(n,ast.ImportFrom) and any(a.name in ('engine_device_public_hooks','engine_device_public_port') for a in n.names)) and not(isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('step_with_public','initialize_with_public'))]
   current=Erase().visit(current)
   self.assertEqual(ast.dump(current,include_attributes=False),ast.dump(base,include_attributes=False),name)
  # Later hooks and engine changes are bound in full, never erased to pass: transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components;
  # secondary.py: byte-exact against reviewed SOURCE_PINS in transition_observer_wiring's source-binding subtests.

if __name__=='__main__':unittest.main()
