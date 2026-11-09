"""constructor_worlds' carry completion (fill_shown_types, fill_negative_items, orient_credits, complete_world, carry_tables): synthetic
tracker carries; the device public hooks' own writes (install, color_change) on the same reveals, with a fake belief table and with the
repo's compiled belief tables (engine_device_observation's fixture catalog); and the B2 -> engine species id space the tracker uses."""
from dataclasses import replace
from types import SimpleNamespace as NS
from typing import NamedTuple
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import constructor_worlds as cw,engine_device_negative_items as ni,engine_device_public_hooks as hooks
from gpu_engine import engine_device_trackers as tr
from gpu_engine.device_public_world_state import World
from gpu_engine.state import empty_state

# Engine species 0..4 -> engine type ids (0 = none); species 5+ are outside the table (unknown).
TABLE=np.asarray([[0,0],[1,0],[2,3],[4,0],[1,2]],np.int32)
WATER=5


def table(rows=TABLE):return cw._table(np.asarray(rows,np.int32))


def state(species,*,active=(0,0),types=None,base=None):
    """An engine state of `species` [2, 6] (engine ids), types = base types = TABLE rows unless given."""
    sp=np.asarray(species,np.int32);t=TABLE[np.clip(sp,0,len(TABLE)-1)]if types is None else np.asarray(types,np.int32)
    return empty_state(1)._replace(species=j.asarray(sp[None]),active=j.asarray([active],j.int32),types=j.asarray(t[None]),
        base_types=j.asarray((t if base is None else np.asarray(base,np.int32))[None]))


def carry(species,mapping,*,shown=None,credits=((2,),(1,)),negative=None,fields=True):
    """A from_b2-style tracker carry: public `species` [2, 6] per encounter slot, physical -> public `mapping` [2, 6], shown_types
    zeros (or `shown`), negative_items None, field_features with owner-oriented [B] items-removed credits."""
    z=j.zeros((1,2,6),j.int32);sp=j.asarray(np.asarray(species,np.int32)[None])
    ff=None if not fields else dict(weather_upkeeps=j.zeros(1,j.int32),weather_permanent=j.zeros(1,bool),wish_set_turn=j.full((1,2),-1,j.int32),
        **({}if credits is None else dict(self_items_removed_credit=j.asarray(credits[0],j.int32),opp_items_removed_credit=j.asarray(credits[1],j.int32))))
    return tr.Trackers(j.asarray(np.asarray(mapping,np.int32)[None]),j.zeros((1,2),j.int32),j.zeros((1,2),j.int32),j.asarray([[2,3]],j.int32),
        j.broadcast_to(j.arange(6,dtype=j.int32),(1,2,6)),sp,z,z,z,z,j.zeros((1,2,6,7),j.int32),sp>0,j.zeros((1,2,6),bool),j.ones(1,j.int32),
        j.zeros(1,j.int32),j.zeros((1,2),j.int32),j.zeros(1,j.int32),
        shown_types=j.zeros((1,2,6,2),j.int32)if shown is None else j.asarray(np.asarray(shown,np.int32)[None]),
        mon_features=dict(turns_active=z),field_features=ff,negative_items=negative)


SPECIES=[[3,1,0,0,0,0],[2,4,7,0,0,0]]   # side 1 slot 2: species 7, outside the table
MAPPING=[[0,1,-1,-1,-1,-1],[0,1,2,-1,-1,-1]]


def leaves(public,skip=()):
    return[(path,x)for path,x in jax.tree_util.tree_flatten_with_path(public)[0]if not any(s in jax.tree_util.keystr(path)for s in skip)]


class FillShownTypesTests(unittest.TestCase):
    def test_known_species_take_species_types_and_nothing_else_changes(self):
        preset=np.zeros((2,6,2),np.int32);preset[1,1]=[6,0]   # side 1 slot 1 already shown (Transform / Color Change): untouched
        public=carry(SPECIES,MAPPING,shown=preset);s=state([[3,1,2,0,0,0],[2,4,1,0,0,0]])
        out=cw.fill_shown_types(s,public,table())
        want=np.zeros((2,6,2),np.int32);want[0,0]=TABLE[3];want[0,1]=TABLE[1];want[1,0]=TABLE[2];want[1,1]=[6,0]
        np.testing.assert_array_equal(np.asarray(out.shown_types)[0],want)   # both sides; unknown (7) and empty slots stay 0
        self.assertEqual(out.shown_types.dtype,j.int32)
        before,after=leaves(public,('shown_types',)),leaves(out,('shown_types',))
        self.assertEqual([p for p,_ in before],[p for p,_ in after]);self.assertTrue(all(x is y for(_,x),(_,y)in zip(before,after)))   # same objects
        self.assertIs(cw.fill_shown_types(s,out,table()),out)   # idempotent: a filled carry is returned as it is

    def test_an_active_whose_engine_types_differ_from_its_base_types_takes_them(self):
        types=TABLE[np.asarray([[3,1,2,0,0,0],[2,4,1,0,0,0]])].copy();types[1,0]=[WATER,0]   # side 1's active (physical 0) changed type
        s=state([[3,1,2,0,0,0],[2,4,1,0,0,0]],types=types,base=TABLE[np.asarray([[3,1,2,0,0,0],[2,4,1,0,0,0]])])
        out=np.asarray(cw.fill_shown_types(s,carry(SPECIES,MAPPING),table()).shown_types)[0]
        np.testing.assert_array_equal(out[1,0],[WATER,0]);np.testing.assert_array_equal(out[1,1],TABLE[4])   # the bench keeps its species' types
        bench=types.copy();bench[1,0]=TABLE[2];bench[1,1]=[WATER,0]   # a changed BENCH mon (physical 1) is not the active: species types
        s=state([[3,1,2,0,0,0],[2,4,1,0,0,0]],types=bench,base=TABLE[np.asarray([[3,1,2,0,0,0],[2,4,1,0,0,0]])])
        np.testing.assert_array_equal(np.asarray(cw.fill_shown_types(s,carry(SPECIES,MAPPING),table()).shown_types)[0,1,1],TABLE[4])
        preset=np.zeros((2,6,2),np.int32);preset[1,0]=[3,0]   # a shown row is never overwritten, even for a changed active
        s=state([[3,1,2,0,0,0],[2,4,1,0,0,0]],types=types,base=TABLE[np.asarray([[3,1,2,0,0,0],[2,4,1,0,0,0]])])
        np.testing.assert_array_equal(np.asarray(cw.fill_shown_types(s,carry(SPECIES,MAPPING,shown=preset),table()).shown_types)[0,1,0],[3,0])

    def test_nothing_to_fill_returns_the_same_carry(self):
        public=carry([[0]*6]*2,MAPPING);self.assertIs(cw.fill_shown_types(state([[0]*6]*2),public,table()),public)
        public=carry([[1,0,0,0,0,0],[0]*6],MAPPING);self.assertIs(cw.fill_shown_types(state([[1,0,0,0,0,0],[0]*6]),public,table([[0,0],[0,0]])),public)

    def test_another_layout_fails_loudly(self):
        s=state(SPECIES);public=carry(SPECIES,MAPPING)
        class Other(NamedTuple):
            species:object
        cases=dict(type=(s,Other(public.species),table()),species=(s,public._replace(species=public.species[...,:5]),table()),
            shown_dtype=(s,public._replace(shown_types=np.zeros((1,2,6,2),np.int64)),table()),
            state_types=(s._replace(types=s.types[...,:1]),public,table()),tables=(s,public,TABLE))
        for name,(st,pub,tab)in cases.items():
            with self.subTest(name),self.assertRaises(cw.CarryLayoutError):cw.fill_shown_types(st,pub,tab)
        for bad in(np.zeros((5,3),np.int32),np.zeros((5,2),np.int64),np.full((5,2),-1,np.int32)):
            with self.assertRaises(cw.CarryLayoutError):cw._table(bad)
        with self.assertRaises(cw.CarryLayoutError):cw._table(TABLE,species_count=6)
        with self.assertRaises(cw.CarryLayoutError):cw._table(TABLE,type_count=4)
        self.assertTrue(issubclass(cw.CarryLayoutError,RuntimeError)and not issubclass(cw.CarryLayoutError,cw.ROOT_ERRORS))   # never a refusal


class CompletionTests(unittest.TestCase):
    def test_negative_items_become_the_device_paths_empty_state(self):
        public=carry(SPECIES,MAPPING);out=cw.fill_negative_items(public)
        self.assertIs(type(out.negative_items),ni.State)
        for got,want in zip(out.negative_items,ni.empty(1)):
            self.assertEqual((got.shape,got.dtype),(want.shape,want.dtype));np.testing.assert_array_equal(got,want)
        self.assertIs(cw.fill_negative_items(out),out)   # an existing State is kept
        self.assertTrue(all(x is y for(_,x),(_,y)in zip(leaves(public,('negative_items',)),leaves(out,('negative_items',)))))
        with self.assertRaises(cw.CarryLayoutError):cw.fill_negative_items(public._replace(negative_items=(1,2)))
        with self.assertRaises(cw.CarryLayoutError):cw.fill_negative_items(public._replace(negative_items=ni.State(j.zeros((1,2,6),j.float32),j.zeros((1,2,6),bool))))

    def test_items_removed_credits_become_per_side(self):
        for owner in(0,1):
            out=cw.orient_credits(carry(SPECIES,MAPPING,credits=([2],[1])),owner);f=out.field_features
            removed=np.zeros((1,2),np.int32);removed[0,owner],removed[0,1-owner]=2,1
            np.testing.assert_array_equal(f['self_items_removed_credit'],removed);np.testing.assert_array_equal(f['opp_items_removed_credit'],removed[:,::-1])
            # the readout's rule for a [B, 2] field (take the observer's column): the owner sees its own counts, the peer its own
            for observer,(mine,theirs)in((owner,(2,1)),(1-owner,(1,2))):
                self.assertEqual((int(f['self_items_removed_credit'][0,observer]),int(f['opp_items_removed_credit'][0,observer])),(mine,theirs))
            self.assertIs(cw.orient_credits(out,owner),out)   # already per side
            self.assertEqual(set(f),set(carry(SPECIES,MAPPING).field_features))
        public=carry(SPECIES,MAPPING,credits=None);self.assertIs(cw.orient_credits(public,0),public)   # no credits (the device path): unchanged
        public=carry(SPECIES,MAPPING,fields=False);self.assertIs(cw.orient_credits(public,0),public)
        one=carry(SPECIES,MAPPING);one=one._replace(field_features={k:v for k,v in one.field_features.items()if k!='opp_items_removed_credit'})
        for bad,owner in((one,0),(carry(SPECIES,MAPPING,credits=([[1,2,3]],[[1,2,3]])),0),(carry(SPECIES,MAPPING),2)):
            with self.assertRaises(cw.CarryLayoutError):cw.orient_credits(bad,owner)

    def test_complete_world_composes_the_three_and_leaves_the_rest(self):
        s=state([[3,1,2,0,0,0],[2,4,1,0,0,0]]);public=carry(SPECIES,MAPPING);world=World(3,s,public,(),77,False)
        done=cw.complete_world(world,table(),1)
        self.assertEqual((done.index,done.refusals,done.kernel_seed),(3,(),77));self.assertIs(done.state,s)
        self.assertIs(type(done.public.negative_items),ni.State);self.assertEqual(done.public.field_features['self_items_removed_credit'].shape,(1,2))
        self.assertTrue(np.asarray(done.public.shown_types).any())
        changed=('shown_types','negative_items','self_items_removed_credit','opp_items_removed_credit')
        self.assertTrue(all(x is y for(_,x),(_,y)in zip(leaves(public,changed),leaves(done.public,changed))))
        self.assertIs(cw.complete_world(done,table(),1),done)   # build's second pass on prepare_roots' worlds: a no-op
        refused=World(1,None,None,('x',),5,False);self.assertIs(cw.complete_world(refused,table(),0),refused)
        class Fake(NamedTuple):
            clock:object
        with self.assertRaisesRegex(cw.CarryLayoutError,'world 2 public carry'):cw.complete_world(World(2,s,Fake(1),(),5,False),table(),0)

    def test_tables_come_from_the_constructor_once(self):
        constructor=NS(metadata=dict(vocabulary=dict(species=['']*5,types=['']*6)),catalog=object())
        with patch.object(cw,'carry_tables',side_effect=lambda metadata,catalog:table())as compile_:
            a=cw.world_completer(constructor);cw.world_completer(constructor)
            compile_.assert_called_once_with(constructor.metadata,constructor.catalog)
            world=World(0,state([[3,1,2,0,0,0],[2,4,1,0,0,0]]),carry(SPECIES,MAPPING),(),1,False)
            self.assertTrue(np.asarray(a(world,0).public.shown_types).any());self.assertIs(cw.constructor_carry_tables(constructor),cw.constructor_carry_tables(constructor))


def fake_belief(rows):
    """The device hooks' belief tables, as tests/test_gpu_engine_device_public_teacher.facts builds them, with `rows` as species_types."""
    n=len(rows)
    return NS(species_initial_mask=j.tile(j.asarray([[3,0]],j.uint32),(n,1)),species_types=j.asarray(rows,j.int32),
        ability_mask=j.zeros((n,3,2),j.uint32),item_mask=j.zeros((n,3,2),j.uint32))


def battle(species,rows,*,active=(0,0)):
    sp=np.asarray(species,np.int32);t=np.asarray(rows,np.int32)[sp]
    return empty_state(1)._replace(species=j.asarray(sp[None]),hp=j.full((1,2,6),100,j.int32),max_hp=j.full((1,2,6),100,j.int32),
        stats=j.full((1,2,6,6),100,j.int32),level=j.full((1,2,6),80,j.int32),types=j.asarray(t[None]),base_types=j.asarray(t[None]),
        moves=j.broadcast_to(j.arange(1,5,dtype=j.int32),(1,2,6,4)),pp=j.full((1,2,6,4),10,j.int32),gender=j.zeros((1,2,6),j.int32),
        active=j.asarray([active],j.int32),entry_pending=j.zeros((1,2),bool),done=j.asarray([False]))


def as_from_b2(trackers):
    """The device carry as from_b2 leaves it: shown_types zeros, negative_items None (every other leaf as the hooks wrote it)."""
    return trackers._replace(shown_types=j.zeros_like(trackers.shown_types),negative_items=None)


def device_reveals(belief,species,rows):
    """The device path on one history: initialize (both leads installed), then side 0 switches to physical 2 and side 1 to physical
    3 (each install hook writes that member's shown_types). Returns (context, final state)."""
    s=battle(species,rows);c=hooks.initialize(s,belief)
    after=s._replace(active=j.asarray([[2,3]],j.int32))
    for side in(0,1):c=hooks.install(c,after,side,j.ones(1,bool))
    return c,after


ROWS=[[0,0],[1,0],[2,3],[4,0],[1,2],[3,0],[2,0],[4,1]]
TEAMS=[[1,2,3,4,5,6],[7,6,5,4,3,2]]


class DeviceHookComparisonTests(unittest.TestCase):
    """The fill on a from_b2-style carry (the device carry with shown_types zeroed and negative_items None) equals what the device public
    hooks wrote for the same revealed species, at the carry and through the readout (the encoder's TYPE_1/TYPE_2 source)."""
    def test_install_reveals(self):
        c,after=device_reveals(fake_belief(ROWS),TEAMS,ROWS);device=c.trackers
        self.assertTrue(np.asarray(device.shown_types).any())
        filled=cw.fill_shown_types(after,as_from_b2(device),table(ROWS))
        np.testing.assert_array_equal(np.asarray(filled.shown_types),np.asarray(device.shown_types))
        for observer in(0,1):
            seen=lambda t:np.asarray(tr.readout(t,j.asarray([observer],j.int32)).types)
            np.testing.assert_array_equal(seen(filled),seen(device));self.assertTrue(seen(filled).any())
            self.assertFalse(seen(as_from_b2(device)).any())   # the bug: every revealed member of the other side reads type 0
        world=World(0,after,as_from_b2(device),(),1,False);done=cw.complete_world(world,table(ROWS),0)
        np.testing.assert_array_equal(np.asarray(done.public.shown_types),np.asarray(device.shown_types))
        for got,want in zip(done.public.negative_items,ni.empty(1)):np.testing.assert_array_equal(got,want)

    def test_a_typechange_on_the_active_matches_the_color_change_hook(self):
        c,before=device_reveals(fake_belief(ROWS),TEAMS,ROWS)
        side=1;physical=int(before.active[0,side]);changed=before._replace(types=before.types.at[0,side,physical].set(j.asarray([WATER,0])))
        c=hooks.color_change(c,before,changed,side,j.ones(1,bool));device=c.trackers
        slot=int(device.physical_to_public[0,side,physical]);np.testing.assert_array_equal(np.asarray(device.shown_types)[0,side,slot],[WATER,0])
        filled=cw.fill_shown_types(changed,as_from_b2(device),table(ROWS))   # the state carries the change: the hook's own write
        np.testing.assert_array_equal(np.asarray(filled.shown_types),np.asarray(device.shown_types))
        # Benched after the change: the device carry keeps the changed types (no hook clears them before the next install); the
        # constructor's state and B2's overlay hold the species' types again (Gen 3 types reset on switching out), so the fill
        # writes the species' types: the one documented difference from the device path.
        out=changed._replace(active=changed.active.at[0,side].set(5),types=changed.base_types)
        c=hooks.install(c,out,side,j.ones(1,bool));device=c.trackers
        np.testing.assert_array_equal(np.asarray(device.shown_types)[0,side,slot],[WATER,0])
        filled=np.asarray(cw.fill_shown_types(out,as_from_b2(device),table(ROWS)).shown_types)
        np.testing.assert_array_equal(filled[0,side,slot],ROWS[TEAMS[side][physical]])
        mask=np.ones(filled.shape[:3],bool);mask[0,side,slot]=False
        np.testing.assert_array_equal(filled[mask],np.asarray(device.shown_types)[mask])   # every other row as the device wrote it


class RealTablesTests(unittest.TestCase):
    """The repo's compiled tables: engine_device_observation's fixture catalog (tests/test_gpu_engine_device_observation)."""
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_device_observation as fixtures
        fixtures.DeviceObservationTests.setUpClass()
        cls.catalog,cls.metadata,cls.tables=fixtures.DeviceObservationTests.catalog,fixtures.DeviceObservationTests.metadata,fixtures.DeviceObservationTests.tables

    def test_carry_tables_are_the_executors_species_types_in_the_engine_id_space(self):
        got=cw.carry_tables(self.metadata,self.catalog).species_types
        np.testing.assert_array_equal(got,np.asarray(self.tables.belief.species_types))   # the executor's own table (observation tables .belief)
        v=self.metadata['vocabulary'];types={n:i for i,n in enumerate(v['types'])}
        self.assertEqual(got.shape,(len(v['species']),2))
        np.testing.assert_array_equal(got[v['species'].index('snorlax')],[types['normal'],0])
        np.testing.assert_array_equal(got[v['species'].index('gardevoir')],[0,0])   # the fixture dex has no Gardevoir: 0 in the hook's table too
        # from_b2 remaps B2's public species ids into the engine namespace (phase_d_live_worlds._remap): the hook's index space.
        from gpu_engine.phase_d_live_worlds import _remap
        names=list(self.catalog.species.names);b2=[names.index(n)for n in('Gardevoir','Snorlax')]
        engine=_remap(j.asarray([b2],j.int32),names,v['species'])
        self.assertEqual(engine.tolist(),[[v['species'].index('gardevoir'),v['species'].index('snorlax')]])

    def test_the_fill_matches_the_install_hook_with_the_real_tables(self):
        rows=np.asarray(self.tables.belief.species_types)
        c,after=device_reveals(self.tables.belief,[[2,1,3,2,1,3],[1,3,2,2,1,3]],rows);device=c.trackers   # Snorlax revealed on both sides
        filled=cw.fill_shown_types(after,as_from_b2(device),cw.carry_tables(self.metadata,self.catalog))
        np.testing.assert_array_equal(np.asarray(filled.shown_types),np.asarray(device.shown_types))
        self.assertTrue(np.asarray(device.shown_types).any())


class FromB2WorldTests(unittest.TestCase):
    """Real phase_d_live_worlds.from_b2 roots (tests/test_phase_d_device_worlds' authored B2 fold: each side's lead revealed) and their
    16 sampled worlds through complete_world, for both owners."""
    @classmethod
    def setUpClass(cls):
        import test_phase_d_device_worlds as D
        D.WorldsTests.setUpClass();cls.fx=D.WorldsTests('test_real_b2_fold_to_16_and32_source_worlds_both_owners')
        cls.tables=cw.carry_tables(cls.fx.metadata,cls.fx.catalog)

    def test_real_from_b2_worlds_complete(self):
        from gpu_engine import phase_d_live_worlds as live
        from pokezero.v5_contract import DecisionAddress
        f=self.fx;st=self.tables.species_types;normal=f.metadata['vocabulary']['types'].index('Normal')
        for owner in(0,1):
            root=live.from_b2(f.adapter(owner),f.request(owner),f.source,f.metadata,address=DecisionAddress(1,0,owner))
            made=live.sample_worlds(root,f.source,f.tables,count=16,seed=9);self.assertTrue(made.fully_constructed)
            done=[cw.complete_world(w,self.tables,owner)for w in made.worlds]   # no CarryLayoutError on the real carry
            for w,d in zip(made.worlds,done):
                with self.subTest(owner=owner,world=w.index):
                    t,c=w.public,d.public
                    self.assertFalse(np.asarray(t.shown_types).any())   # from_b2's zeros
                    species,active,shown=np.asarray(c.species)[0],np.asarray(c.active_slot)[0],np.asarray(c.shown_types)[0]
                    for side in(0,1):   # each side's revealed lead is filled with its species' types; every other slot stays 0
                        at=int(active[side]);self.assertGreater(int(species[side,at]),0)
                        np.testing.assert_array_equal(shown[side,at],st[species[side,at]]);np.testing.assert_array_equal(shown[side,at],[normal,0])
                        rest=np.arange(6)!=at;self.assertFalse(species[side,rest].any());self.assertFalse(shown[side,rest].any())
                    for key in cw.CREDIT_KEYS:
                        self.assertEqual((np.shape(t.field_features[key]),np.shape(c.field_features[key])),((1,),(1,2)))
                    removed=np.asarray(c.field_features['self_items_removed_credit'])
                    self.assertEqual(removed[0,owner],int(np.asarray(t.field_features['self_items_removed_credit'])[0]))
                    self.assertIsNone(t.negative_items);self.assertIs(type(c.negative_items),ni.State)
                    changed=('shown_types','negative_items','self_items_removed_credit','opp_items_removed_credit')
                    self.assertTrue(all(x is y for(_,x),(_,y)in zip(leaves(t,changed),leaves(c,changed))))
                    self.assertIs(d.state,w.state);self.assertIs(cw.complete_world(d,self.tables,owner),d)   # a second pass is a no-op
            # every world completes alike (the completion reads only the public species and the engine types)
            for name in('shown_types',):
                for d in done[1:]:np.testing.assert_array_equal(getattr(d.public,name),getattr(done[0].public,name))
            for d in done[1:]:
                for key in cw.CREDIT_KEYS:np.testing.assert_array_equal(d.public.field_features[key],done[0].public.field_features[key])
                for x,y in zip(d.public.negative_items,done[0].public.negative_items):np.testing.assert_array_equal(x,y)


if __name__=='__main__':unittest.main()
