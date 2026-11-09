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

    def test_carry_tables_hold_the_engine_ability_names(self):
        v=self.metadata['vocabulary'];tables=cw.carry_tables(self.metadata,self.catalog)
        self.assertEqual(tables.ability_names,tuple(map(cw.normalize_id,v['abilities'])))
        self.assertEqual(len(tables.ability_names),len(tables.pressure))   # the same engine ability id space as the Pressure flags
        # This fixture's engine vocabulary has no Trace: a tracer on the lines fails loudly (CarryLayoutError), never silently.
        snorlax=v['species'].index('snorlax');species=np.zeros((1,2,6),np.int32);species[0,:,0]=snorlax
        state=empty_state(1)._replace(species=j.asarray(species))
        facts=cw.line_facts([switch('p1','snorlax'),switch('p2','snorlax'),'|-ability|p2a: Snorlax|Immunity|Trace|[from] ability: Trace|[of] p1a: Snorlax'],0)
        with self.assertRaisesRegex(cw.CarryLayoutError,'Trace is outside'):cw.fix_trace_base_ability(state,0,facts,tables)

    def test_the_peer_pressure_count_matches_the_device_move_hook_with_the_real_tables(self):
        v=self.metadata['vocabulary'];pressure,surf=v['abilities'].index('pressure'),v['moves'].index('surf')
        rows=np.asarray(self.tables.belief.species_types)
        s=battle([[2,1,3,0,0,0],[1,3,2,0,0,0]],rows)   # one member per species (species clause): the joins are by species
        s=s._replace(ability=s.ability.at[0,1,0].set(pressure),base_ability=s.base_ability.at[0,1,0].set(pressure))   # the peer's Gardevoir: Pressure
        c=hooks.initialize(s,self.tables.belief)
        for _ in range(2):c=hooks.move(c,s,0,j.asarray([surf],j.int32),j.ones(1,bool),pp_use=True)   # the owner's Snorlax uses Surf twice
        device=c.trackers;slot=int(device.active_slot[0,0]);column=int(np.flatnonzero(np.asarray(device.observed_moves)[0,0,slot]==surf)[0])
        self.assertEqual(int(device.pp_used[0,0,slot,column]),4)   # the device: the observer knows its own Pressure
        b2=device._replace(pp_used=device.pp_used.at[0,0,slot,column].set(2))   # B2's public count: Gen 3 Pressure is silent
        facts=cw.LineFacts(('gardevoir',),(),(False,False),(('snorlax','surf','gardevoir',True,False),)*2)
        out=cw.fill_peer_pressure(s,b2,0,facts,cw.carry_tables(self.metadata,self.catalog),b2)
        np.testing.assert_array_equal(np.asarray(out.pp_used),np.asarray(device.pp_used))

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
            # with the fold's own lines: the peer's lead is its request position 0 and the only switch-in, so the order stands; no Spikes
            lines=(f'|switch|p{owner+1}a: Alpha|Alpha, L80|291/291',f'|switch|p{2-owner}a: Beta|Beta, L80|100/100','|turn|1')
            facts=cw.line_facts(lines,owner);self.assertEqual(facts,cw.LineFacts(('beta',),()))
            for d in done:
                e=cw.complete_world(d,self.tables,owner,facts=facts);self.assertIs(e,d)
                order=np.asarray(d.public.own_order)[0,1-owner];self.assertEqual(int(order[0]),int(np.asarray(d.state.active)[0,1-owner]))
            # every world completes alike (the completion reads only the public species and the engine types)
            for name in('shown_types',):
                for d in done[1:]:np.testing.assert_array_equal(getattr(d.public,name),getattr(done[0].public,name))
            for d in done[1:]:
                for key in cw.CREDIT_KEYS:np.testing.assert_array_equal(d.public.field_features[key],done[0].public.field_features[key])
                for x,y in zip(d.public.negative_items,done[0].public.negative_items):np.testing.assert_array_equal(x,y)


    def test_b2s_fold_never_records_a_tracers_own_ability(self):
        """The real fold on Showdown's Gen 3 setAbility line form (the copied ability, then the tracer's own Trace as a positional
        argument): the tracer's public original stays 0 and its borrowed ability is the copy; the donor's original is the copy.
        fix_peer_base_ability's precondition (a public original) therefore never holds for a tracer."""
        from gpu_engine import phase_d_live_worlds as live
        from pokezero.v5_contract import DecisionAddress
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        f=self.fx;pressure=f.metadata['vocabulary']['abilities'].index('pressure')
        for owner,tracer in((0,1),(1,1),(0,0)):
            with self.subTest(owner=owner,tracer=tracer):
                donor=1-tracer;seat=lambda side:f'p{side+1}a: '+('Alpha'if side==owner else'Beta')
                lines=(f'|switch|p{owner+1}a: Alpha|Alpha, L80|291/291',f'|switch|p{2-owner}a: Beta|Beta, L80|100/100',
                    f'|-ability|{seat(tracer)}|Pressure|Trace|[from] ability: Trace|[of] {seat(donor)}','|turn|1')
                a=JaxRecordedAdapter(f.catalog,owner_side=owner);a.reset(stream_id=1,battle_id='authored');a.feed(lines)
                root=live.from_b2(a,f.request(owner),f.source,f.metadata,address=DecisionAddress(1,0,owner))
                original,borrowed=np.asarray(root.public.original_ability)[0],np.asarray(root.public.borrowed_ability)[0]
                self.assertEqual((int(original[tracer,0]),int(borrowed[tracer,0]),int(original[donor,0])),(0,pressure,pressure))
                self.assertEqual(cw.line_facts(lines,owner).tracers[tracer],('alpha'if tracer==owner else'beta',))


NAMES=['','alpha','beta','gamma','delta','epsilon','zeta','eta','theta']   # engine species 1..8
ORDER_TABLE=np.zeros((len(NAMES),2),np.int32)


def peer_world(team,*,active,revealed,owner=0,own_order=None):
    """(state, carry) of one constructed world: the peer (1 - owner) has `team` (engine species, physical order) with `active` the
    physical active and `revealed` its revealed physicals in public-slot order; own_order[peer] as _rebase leaves it
    ([active, the rest ascending]) unless given."""
    peer=1-owner;species=np.zeros((2,6),np.int32);species[peer]=team;species[owner]=[1,2,3,4,5,6]
    s=empty_state(1)._replace(species=j.asarray(species[None]),active=j.asarray([[0,0]],j.int32).at[0,peer].set(active))
    mapping=np.full((2,6),-1,np.int32);mapping[owner]=np.arange(6)
    for slot,physical in enumerate(revealed):mapping[peer,physical]=slot
    pub=np.zeros((2,6),np.int32);pub[owner]=species[owner];pub[peer,:len(revealed)]=[team[p]for p in revealed]
    t=carry(pub,mapping)
    order=np.broadcast_to(np.arange(6,dtype=np.int32),(1,2,6)).copy()
    order[0,peer]=[active,*[i for i in range(6)if i!=active]]if own_order is None else own_order
    return s,t._replace(own_order=j.asarray(order))


def switch(seat,name,hp='100/100',kind='switch'):return f'|{kind}|{seat}a: {name.title()}|{name.title()}, L80, M|{hp}'


class PeerOrderTests(unittest.TestCase):
    """constructor_worlds.order_peer and line_facts: the peer's Showdown request order from the public switch history."""
    tables=cw._table(ORDER_TABLE,species_names=NAMES)

    def order(self,lines,team,*,active,revealed,owner=0,**kw):
        s,t=peer_world(team,active=active,revealed=revealed,owner=owner,**kw)
        out=cw.order_peer(s,t,owner,cw.line_facts(lines,owner),self.tables)
        return np.asarray(out.own_order)[0,1-owner].tolist(),out,t

    def test_voluntary_switches_swap_into_position_zero_and_unrevealed_members_keep_their_slots(self):
        lines=['|init|battle',switch('p1','alpha','291/291'),switch('p2','alpha'),'|turn|1',switch('p2','gamma'),'|turn|2',switch('p2','beta'),'|turn|3']
        got,out,before=self.order(lines,[1,2,3,4,5,6],active=1,revealed=(0,2,1))
        self.assertEqual(got,[1,2,0,3,4,5])   # [A..F] -> C in: [C,B,A,..] -> B in: [B,C,A,D,E,F]; D, E, F never move
        self.assertEqual(np.asarray(before.own_order)[0,1].tolist(),[1,0,2,3,4,5])   # what _rebase leaves: active first, the rest ascending
        self.assertEqual(np.asarray(out.own_order)[0,0].tolist(),list(range(6)))   # the owner's own (request) order is untouched
        for a,b in zip(leaves(before,('own_order',)),leaves(out,('own_order',))):self.assertIs(a[1],b[1])
        self.assertIs(cw.order_peer(peer_world([1,2,3,4,5,6],active=1,revealed=(0,2,1))[0],out,0,cw.line_facts(lines,0),self.tables),out)   # idempotent

    def test_drags_faint_replacements_and_baton_pass_follow_the_same_rule(self):
        lines=['|init|battle',switch('p2','beta'),'|turn|1',
            '|move|p1a: Alpha|Roar|p2a: Beta',switch('p2','epsilon',kind='drag'),'|turn|2',   # a drag
            '|move|p1a: Alpha|Tackle|p2a: Epsilon','|faint|p2a: Epsilon','|upkeep|',switch('p2','zeta'),'|turn|3',   # a faint replacement
            '|move|p2a: Zeta|Baton Pass|p2a: Zeta',switch('p2','beta'),'|turn|4']   # a Baton Pass recipient
        got,_,_=self.order(lines,[2,3,4,5,6,1],active=0,revealed=(0,3,4))
        # [B,C,D,E,F,A]: E dragged in -> [E,C,D,B,F,A]; F replaces -> [F,C,D,B,E,A]; B passed in -> [B,C,D,F,E,A]
        self.assertEqual(got,[0,1,2,4,3,5])
        got,_,_=self.order([l.replace('p2','p1').replace('p1a: Alpha','p2a: Alpha')for l in lines],[2,3,4,5,6,1],active=0,revealed=(0,3,4),owner=1)
        self.assertEqual(got,[0,1,2,4,3,5])   # the same history with the seats exchanged, for owner 1

    def test_a_sampled_order_whose_lead_is_not_first_is_still_swapped(self):
        lines=[switch('p2','gamma'),'|turn|1']   # the lead at physical 2 of this world's sampled team
        got,_,_=self.order(lines,[1,2,3,4,5,6],active=2,revealed=(2,))
        self.assertEqual(got,[2,1,0,3,4,5])

    def test_what_cannot_be_joined_refuses(self):
        team=[1,2,3,4,5,6]
        # (lines, the world's peer active, its revealed members): each case trips one check only
        cases={'no switch-in':(['|turn|1'],0,(0,)),'not in the party':([switch('p2','theta')],0,(0,)),
            'not revealed here, though it is the active':([switch('p2','alpha'),switch('p2','delta')],3,(0,)),   # only the reveal check refuses
            'every switch-in revealed, not the active':([switch('p2','alpha'),switch('p2','beta')],0,(0,1))}   # only the active check refuses
        for name,(lines,active,revealed)in cases.items():
            with self.subTest(name),self.assertRaises(cw.PeerOrderUnavailable)as caught:self.order(lines,team,active=active,revealed=revealed)
            if name.startswith('not revealed'):self.assertIn('not a revealed member',str(caught.exception))
            if name.startswith('every'):self.assertIn('is not the world',str(caught.exception))
        self.assertEqual(self.order([switch('p2','alpha'),switch('p2','beta')],team,active=1,revealed=(0,1))[0],[1,0,2,3,4,5])   # both pass
        with self.assertRaises(cw.CarryLayoutError):
            s,t=peer_world(team,active=0,revealed=(0,));cw.order_peer(s,t,0,cw.line_facts([switch('p2','alpha')],0),table())   # no species names
        self.assertTrue(issubclass(cw.PeerOrderUnavailable,ValueError))

    def test_formes_join_exactly_else_by_a_unique_prefix(self):
        names=['','deoxysspeed','snorlax','unown','deoxysattack','swampert']
        tables=cw._table(np.zeros((len(names),2),np.int32),species_names=names)
        def joined(details,team,active):
            s,t=peer_world(team,active=active,revealed=(active,))
            return np.asarray(cw.order_peer(s,t,0,cw.line_facts([f'|switch|p2a: X|{details}, L80|100/100'],0),tables).own_order)[0,1].tolist()
        self.assertEqual(joined('Deoxys-Speed',[2,1,5,0,0,0],1),[1,0,2,3,4,5])   # exact: deoxysspeed
        self.assertEqual(joined('Unown-C',[2,3,5,0,0,0],1),[1,0,2,3,4,5])   # the engine collapses the forme (unown): a unique prefix
        with self.assertRaisesRegex(cw.PeerOrderUnavailable,'deoxys: 2 members'):joined('Deoxys',[1,4,2,0,0,0],0)   # two prefix matches refuse

    def test_line_facts(self):
        lines=['|init|battle','|switch|p1a: Alpha|Alpha, L80, F|300/300','|switch|p2a: Beta|Beta, L80|100/100','|turn|1',
            '|request|{"side":{"id":"p1"}}','|move|p2a: Beta|Spikes|p1a: Alpha','|-sidestart|p1: Player|Spikes',
            '|switch|p1a: Gamma|Gamma, L80|299/300','|-damage|p1a: Gamma|262/300|[from] Spikes','|-heal|p1a: Gamma|281/300 par|[from] item: Leftovers',
            '|drag|p2a: Delta|Delta, L80, M|100/100','|-damage|p2a: Delta|88/100|[from] Spikes','|turn|2',
            '|move|p2a: Delta|Tackle|p1a: Gamma','|-damage|p1a: Gamma|250/300','|switch|p1a: Alpha|Alpha, L80, F|300/300',   # not Spikes: no hit
            '|-damage|p1a: Alpha|263/300|[from] Spikes','|faint|p1a: Gamma',
            '|switch|p1a: Eta|Eta, L80|10/300','|-damage|p1a: Eta|6/300|[from] psn','|-damage|p1a: Eta|0 fnt|[from] Spikes',   # psn first: before = 6
            '|replace|p2a: Beta|Beta, L80|100/100']
        f=cw.line_facts(lines,0)
        self.assertEqual(f.peer_switches,('beta','delta','beta'))
        self.assertEqual(f.owner_spikes,((299,262,300),(300,263,300),(6,0,300)))
        g=cw.line_facts(lines,1)   # the other seat: p2 owns, p1 is the peer
        self.assertEqual(g.peer_switches,('alpha','gamma','alpha','eta'));self.assertEqual(g.owner_spikes,((100,88,100),))   # p2's hit, as its lines read
        self.assertEqual(cw.line_facts(['|-damage|p1a: Alpha|263/300|[from] Spikes'],0).owner_spikes,None)   # no previous HP on the lines


class Round4LineFactsTests(unittest.TestCase):
    def test_baton_pass_arrivals(self):
        bp=lambda seat,name:f'|move|{seat}a: {name.title()}|Baton Pass|{seat}a: {name.title()}'
        base=[switch('p1','alpha','300/300'),switch('p2','beta'),'|turn|1']
        cases={'a switch after the side\'s Baton Pass':(base+[bp('p2','beta'),switch('p2','gamma')],(False,True)),
            'another move of the side clears it':(base+[bp('p2','beta'),'|move|p2a: Beta|Tackle|p1a: Alpha',switch('p2','gamma')],(False,False)),
            'a drag is never one':(base+[bp('p2','beta'),switch('p2','gamma',kind='drag')],(False,False)),
            'the other side\'s move does not clear it':(base+[bp('p1','alpha'),'|move|p2a: Beta|Tackle|p1a: Alpha',switch('p1','delta','200/300')],(True,False)),
            'a later arrival replaces it':(base+[bp('p2','beta'),switch('p2','gamma'),'|turn|2',switch('p2','beta')],(False,False)),
            'a switch line naming Baton Pass':(base+[switch('p2','gamma')+'|[from] Baton Pass'],(False,True))}
        for name,(lines,want)in cases.items():
            with self.subTest(name):self.assertEqual(cw.line_facts(lines,0).baton,want)

    def test_owner_moves_carry_the_peers_active_and_whether_its_pressure_was_public(self):
        lines=[switch('p1','alpha','300/300'),switch('p2','beta'),'|turn|1',
            '|move|p1a: Alpha|Surf|p2a: Beta',   # charged; Beta's Pressure (if any) not public
            '|move|p1a: Alpha|Body Slam|p2a: Beta|[from]Sleep Talk',   # a called move: never charged
            '|move|p1a: Alpha|Struggle|p2a: Beta','|turn|2',   # Struggle: never charged
            '|-ability|p2a: Beta|Pressure|[from] ability: Trace|[of] p1a: Alpha','|move|p1a: Alpha|Surf|p2a: Beta',   # Beta runs a public (traced) Pressure
            switch('p2','gamma'),switch('p2','beta'),'|move|p1a: Alpha|Surf|p2a: Beta',   # Beta arrived again: the traced copy is gone
            '|-ability|p2a: Beta|Pressure','|move|p1a: Alpha|Hidden Power|p2a: Beta',   # an original Pressure reveal: public from now on
            '|faint|p2a: Beta','|move|p1a: Alpha|Surf|',   # the peer's active fainted
            '|-transform|p1a: Alpha|p2a: Beta','|move|p1a: Alpha|Surf|p2a: Beta']   # a transformed actor: never charged
        self.assertEqual(cw.line_facts(lines,0).owner_moves,(('alpha','surf','beta',True,False),('alpha','surf','beta',True,True),
            ('alpha','surf','beta',True,False),('alpha','hiddenpower','beta',True,True),('alpha','surf','beta',False,True)))


PRESSURE_NAMES=['','alpha','beta','gamma','delta','epsilon','zeta']
MOVES=['','surf','bodyslam','swordsdance','hiddenpowerfire70']
ABILITIES=['','pressure','trace','swiftswim']


def pressure_tables():
    return cw._table(np.zeros((len(PRESSURE_NAMES),2),np.int32),species_names=PRESSURE_NAMES,move_names=MOVES,
        foe_targeted=[False,True,True,False,True],pressure=[n=='pressure'for n in ABILITIES],ability_names=ABILITIES)


def pressure_world(*,peer_abilities=(1,3,0,0,0,0),uses=((2,1,0,0),)):
    """(state, carry, root carry): owner 0 has Alpha (public slot 0) whose revealed moves are surf, bodyslam, swordsdance, a
    Hidden Power with B2's counts `uses`; the peer has Beta, Gamma (physical 0, 1) with base abilities `peer_abilities`."""
    state=empty_state(1)._replace(species=j.asarray([[[1,4,5,6,0,0],[2,3,0,0,0,0]]],j.int32),base_ability=j.asarray([[[0]*6,list(peer_abilities)]],j.int32))
    t=carry([[1,4,5,6,0,0],[2,3,0,0,0,0]],[[0,1,2,3,-1,-1],[0,1,-1,-1,-1,-1]])
    observed=np.zeros((1,2,6,4),np.int32);observed[0,0,0]=[1,2,3,4];used=np.zeros((1,2,6,4),np.int32);used[0,0,0]=(list(uses[0])+[0,0,0,0])[:4]
    t=t._replace(observed_moves=j.asarray(observed),pp_used=j.asarray(used))
    return state,t,t


class PeerPressureTests(unittest.TestCase):
    def test_the_peers_own_silent_pressure_charges_the_owners_foe_targeted_moves_twice(self):
        state,t,root=pressure_world(uses=((3,1,1,2),))
        moves=(('alpha','surf','beta',True,False),('alpha','surf','beta',True,False),   # silent Pressure: +1 each
            ('alpha','surf','beta',True,True),   # public Pressure: B2 already charged two
            ('alpha','bodyslam','gamma',True,False),   # Gamma has no Pressure
            ('alpha','swordsdance','beta',True,False),   # not foe-targeted
            ('alpha','hiddenpower','beta',True,False),   # Hidden Power joins its typed column: +1
            ('alpha','surf','beta',False,False))   # the peer's active had fainted
        facts=cw.LineFacts(('beta',),(),(False,False),moves)
        out=cw.fill_peer_pressure(state,t,0,facts,pressure_tables(),root)
        np.testing.assert_array_equal(np.asarray(out.pp_used)[0,0,0],[5,1,1,3])
        np.testing.assert_array_equal(np.asarray(out.pp_used)[0,1],np.asarray(t.pp_used)[0,1])   # the peer's own PP: B2's
        self.assertIs(cw.fill_peer_pressure(state,out,0,facts,pressure_tables(),root),out)   # from the root's counts: the same on every pass
        for kw in(dict(root=None),dict(facts=None)):
            args=dict(dict(facts=facts,root=root),**kw);self.assertIs(cw.fill_peer_pressure(state,t,0,args['facts'],pressure_tables(),args['root']),t)
        swift=pressure_world(peer_abilities=(3,3,0,0,0,0),uses=((3,1,1,2),))   # a world whose Beta has no Pressure: nothing to add
        self.assertIs(cw.fill_peer_pressure(swift[0],swift[1],0,facts,pressure_tables(),swift[2]),swift[1])


    def test_a_tracers_copied_pressure_is_not_its_own_and_the_completion_is_idempotent(self):
        """A peer Beta (Trace) faced the owner's Alpha in an earlier stint and took a foe-targeted Surf (its copy then not Pressure),
        left, came back and traced the owner's Pressure. The spec wrote the copy into its base_ability; the corrected state says
        Trace, so the earlier Surf is not recharged, on the first pass (prepare_roots) or the second (build)."""
        state,t,root=pressure_world(peer_abilities=(1,3,0,0,0,0),uses=((3,1,1,2),))   # Beta: base_ability Pressure, as the spec wrote it
        original=np.zeros((1,2,6),np.int32);borrowed=np.zeros((1,2,6),np.int32);original[0,1,0]=2;borrowed[0,1,0]=1   # Trace, running Pressure
        t=t._replace(original_ability=j.asarray(original),borrowed_ability=j.asarray(borrowed),own_order=j.asarray([[[0,1,2,3,4,5]]*2],j.int32))
        lines=[switch('p1','alpha','300/300'),switch('p2','beta'),'|turn|1','|move|p1a: Alpha|Surf|p2a: Beta','|turn|2',switch('p2','gamma'),
            '|turn|3',switch('p2','beta'),'|-ability|p2a: Beta|Pressure|[from] ability: Trace|[of] p1a: Alpha','|turn|4']
        facts=cw.line_facts(lines,0);self.assertEqual(facts.owner_moves,(('alpha','surf','beta',True,False),))
        world=World(0,state,t,(),1,False)
        first=cw.complete_world(world,pressure_tables(),0,facts=facts,root=root)
        np.testing.assert_array_equal(np.asarray(first.public.pp_used)[0,0,0],[3,1,1,2])   # B2's count: nothing to add
        self.assertEqual(int(np.asarray(first.state.base_ability)[0,1,0]),2)   # Trace
        self.assertIs(cw.complete_world(first,pressure_tables(),0,facts=facts,root=root),first)   # the second pass changes nothing
        silent=World(0,state,t._replace(original_ability=j.zeros((1,2,6),j.int32),borrowed_ability=j.zeros((1,2,6),j.int32)),(),1,False)
        # B2's carry never holds a tracer's own ability; the Trace line alone proves Beta's is Trace (fix_trace_base_ability): no charge.
        traced=cw.complete_world(silent,pressure_tables(),0,facts=facts,root=root)
        np.testing.assert_array_equal(np.asarray(traced.public.pp_used)[0,0,0],[3,1,1,2]);self.assertEqual(int(np.asarray(traced.state.base_ability)[0,1,0]),2)
        quiet=cw.line_facts([x for x in lines if not x.startswith('|-ability|')],0)   # Beta never traced: its own Pressure, never public
        once=cw.complete_world(silent,pressure_tables(),0,facts=quiet,root=root)
        np.testing.assert_array_equal(np.asarray(once.public.pp_used)[0,0,0],[4,1,1,2])
        self.assertIs(cw.complete_world(once,pressure_tables(),0,facts=quiet,root=root),once)


class RechargeTests(unittest.TestCase):
    def test_line_facts_follow_b2s_recharge_rule(self):
        base=[switch('p1','alpha','300/300'),switch('p2','beta'),'|turn|1','|move|p2a: Beta|Hyper Beam|p1a: Alpha','|-damage|p1a: Alpha|100/300',
            '|-mustrecharge|p2a: Beta']
        cases={'set':(base+['|turn|2'],(False,True)),'its recharge turn clears it':(base+['|turn|2','|cant|p2a: Beta|recharge','|turn|3'],(False,False)),
            'an arrival clears it':(base+['|turn|2',switch('p2','gamma')],(False,False)),'a faint clears it':(base+['|faint|p2a: Beta'],(False,False)),
            'the other side\'s cant does not':(base+['|turn|2','|cant|p1a: Alpha|recharge'],(False,True)),
            'another cant does not':(base+['|turn|2','|cant|p2a: Beta|par'],(False,True))}
        for name,(lines,want)in cases.items():
            with self.subTest(name):self.assertEqual(cw.line_facts(lines,0).recharge,want)

    def test_the_fill_sets_the_volatile_on_the_sides_active(self):
        from gpu_engine.public_active import NAMES
        t=carry(SPECIES,MAPPING)._replace(public_volatiles=j.zeros((1,2,6,len(NAMES)),bool),active_slot=j.asarray([[0,1]],j.int32))
        out=cw.fill_recharge(t,cw.LineFacts(('beta',),(),recharge=(False,True)))
        bag=np.asarray(out.public_volatiles);self.assertEqual(np.argwhere(bag).tolist(),[[0,1,1,NAMES.index('mustrecharge')]])   # the peer's active slot 1
        self.assertIs(cw.fill_recharge(out,cw.LineFacts(('beta',),(),recharge=(False,True))),out)
        self.assertIs(cw.fill_recharge(t,cw.LineFacts(('beta',),())),t)


class PeerBaseAbilityTests(unittest.TestCase):
    def test_a_peer_tracers_base_ability_is_its_public_original(self):
        state=empty_state(1)._replace(base_ability=j.asarray([[[3,0,0,0,0,0],[1,3,0,0,0,0]]],j.int32),ability=j.asarray([[[3,0,0,0,0,0],[1,3,0,0,0,0]]],j.int32))
        t=carry([[1,0,0,0,0,0],[2,3,0,0,0,0]],[[0,-1,-1,-1,-1,-1],[1,0,-1,-1,-1,-1]])
        original=np.zeros((1,2,6),np.int32);borrowed=np.zeros((1,2,6),np.int32)
        original[0,1,1]=2;borrowed[0,1,1]=1   # the peer's public slot 1 (physical 0): Trace, running a copied Pressure
        original[0,0,0]=2;borrowed[0,0,0]=1   # the owner's own tracer: never touched (from_b2 wrote the request's baseAbility)
        t=t._replace(original_ability=j.asarray(original),borrowed_ability=j.asarray(borrowed))
        out=cw.fix_peer_base_ability(state,t,0)
        self.assertEqual(np.asarray(out.base_ability)[0].tolist(),[[3,0,0,0,0,0],[2,3,0,0,0,0]])
        self.assertIs(out.ability,state.ability);self.assertIs(cw.fix_peer_base_ability(out,t,0),out)
        world=World(0,state,t,(),1,False);done=cw.complete_world(world,table(),0)
        self.assertEqual(np.asarray(done.state.base_ability)[0,1].tolist(),[2,3,0,0,0,0]);self.assertIs(cw.complete_world(done,table(),0),done)


# The Gen 3 lines of the OV5 Trace misses (fp-collect-r1 owning-wire, verbatim): battles 335, 495 and 718.
TRACE_LINES={
    335:['|switch|p1a: Porygon2|Porygon2, L84|100/100','|switch|p2a: Sharpedo|Sharpedo, L83, M|100/100',
        '|-ability|p1a: Porygon2|Rough Skin|Trace|[from] ability: Trace|[of] p2a: Sharpedo','|turn|1'],
    495:['|switch|p1a: Gardevoir|Gardevoir, L82, F|100/100','|switch|p2a: Ampharos|Ampharos, L84, M|100/100',
        '|-ability|p1a: Gardevoir|Static|Trace|[from] ability: Trace|[of] p2a: Ampharos','|turn|1',
        '|switch|p2a: Jolteon|Jolteon, L82, M|100/100','|turn|2','|switch|p1a: Blissey|Blissey, L80, F|100/100','|turn|3',
        '|switch|p1a: Gardevoir|Gardevoir, L82, F|60/100','|-ability|p1a: Gardevoir|Volt Absorb|Trace|[from] ability: Trace|[of] p2a: Jolteon',
        '|turn|4'],
    718:['|switch|p1a: Nidoking|Nidoking, L84, M|100/100','|switch|p2a: Porygon2|Porygon2, L84|100/100',
        '|-ability|p2a: Porygon2|Poison Point|Trace|[from] ability: Trace|[of] p1a: Nidoking','|turn|1']}
TRACE_SPECIES=['','porygon2','sharpedo','gardevoir','ampharos','jolteon','blissey','nidoking']
TRACE_ABILITIES=['','roughskin','trace','static','voltabsorb','poisonpoint','swiftswim']


def trace_tables():
    return cw._table(np.zeros((len(TRACE_SPECIES),2),np.int32),species_names=TRACE_SPECIES,ability_names=TRACE_ABILITIES)


def trace_world(side_species,active,copied,owner):
    """(state, carry) as the materializer and B2's fold leave them: each side's party `side_species` (engine ids), the active
    tracer's base_ability and ability both its current copy `copied` = {side: ability id} (every other member Swift Swim, its
    set's), and B2's carry: the tracer's original 0, its borrowed the copy."""
    species=np.zeros((2,6),np.int32);base=np.full((2,6),6,np.int32)
    for side,team in enumerate(side_species):species[side,:len(team)]=team
    base[species==0]=0;mapping=np.where(species>0,np.arange(6),-1)
    for side,ability in copied.items():base[side,active[side]]=ability
    state=empty_state(1)._replace(species=j.asarray(species[None]),active=j.asarray([list(active)],j.int32),
        base_ability=j.asarray(base[None]),ability=j.asarray(base[None]))
    borrowed=np.zeros((1,2,6),np.int32)
    for side,ability in copied.items():borrowed[0,side,active[side]]=ability
    t=carry(species,mapping)._replace(original_ability=j.zeros((1,2,6),j.int32),borrowed_ability=j.asarray(borrowed))
    return state,t


class TraceTests(unittest.TestCase):
    def test_line_facts_name_each_side_s_tracers_on_the_real_line_form(self):
        self.assertEqual([cw.line_facts(TRACE_LINES[b],0).tracers for b in(335,495,718)],
            [(('porygon2',),()),(('gardevoir',),()),((),('porygon2',))])
        self.assertEqual(cw.line_facts(TRACE_LINES[718],1).tracers,((),('porygon2',)))   # per side, whoever owns the lines
        older=['|switch|p2a: Porygon2|Porygon2, L84|100/100','|-ability|p2a: Porygon2|Pressure|[from] ability: Trace|[of] p1a: Dusclops']
        self.assertEqual(cw.line_facts(older,0).tracers,((),('porygon2',)))   # without the positional own ability too
        plain=['|switch|p2a: Dusclops|Dusclops, L84|100/100','|-ability|p2a: Dusclops|Pressure']   # a direct reveal is no Trace
        self.assertEqual(cw.line_facts(plain,0).tracers,((),()))

    def test_the_tracer_takes_trace_as_its_base_on_either_side(self):
        """Battle 335 seen by p2 (the tracer Porygon2 is the peer) and by p1 (the tracer is the owner's own), 718 by p1 (peer):
        fix_peer_base_ability cannot fire (B2's original is 0); the completion sets base_ability Trace on the tracer only, keeps
        the running copy, and a second pass changes nothing."""
        cases=[(335,1,[[1,6],[2]],(0,0),{0:1},(0,0)),(335,0,[[1,6],[2]],(0,0),{0:1},(0,0)),(718,0,[[7],[1,5]],(0,0),{1:5},(1,0))]
        for battle,owner,team,active,copied,where in cases:
            with self.subTest(battle=battle,owner=owner):
                state,t=trace_world(team,active,copied,owner);facts=cw.line_facts(TRACE_LINES[battle],owner)
                self.assertIs(cw.fix_peer_base_ability(state,t,owner),state)   # the precondition that failed in OV5
                fixed,_=cw.complete_carry(state,t,trace_tables(),owner,facts=facts)
                base,run=np.asarray(fixed.base_ability)[0],np.asarray(fixed.ability)[0]
                self.assertEqual(int(base[where]),TRACE_ABILITIES.index('trace'));self.assertEqual(int(run[where]),copied[where[0]])
                want=np.asarray(state.base_ability)[0].copy();want[where]=2;np.testing.assert_array_equal(base,want)   # nothing else
                again,_=cw.complete_carry(fixed,t,trace_tables(),owner,facts=facts);self.assertIs(again,fixed)
        state,t=trace_world([[1,6],[2]],(0,0),{0:1},1);facts=cw.line_facts(TRACE_LINES[335],1)
        with self.assertRaises(cw.CarryLayoutError):cw.complete_carry(state,t,table(),1,facts=facts)   # no ability names
        with self.assertRaises(cw.CarryLayoutError):
            cw.fix_trace_base_ability(state,1,facts,cw._table(np.zeros((8,2),np.int32),species_names=TRACE_SPECIES,ability_names=['','roughskin']))
        with self.assertRaises(cw.PeerOrderUnavailable):cw.fix_trace_base_ability(state,1,cw.line_facts(TRACE_LINES[718],1),trace_tables())
        self.assertIs(cw.fix_trace_base_ability(state,1,cw.LineFacts(('sharpedo',),()),trace_tables()),state)   # no tracer: unchanged

    def test_gardevoirs_second_trace_keeps_one_tracer(self):
        state,t=trace_world([[3,6],[4,5]],(0,1),{0:4},1);facts=cw.line_facts(TRACE_LINES[495],1)
        self.assertEqual(facts.tracers,(('gardevoir',),()))
        fixed,_=cw.complete_carry(state,t,trace_tables(),1,facts=facts)
        self.assertEqual((int(np.asarray(fixed.base_ability)[0,0,0]),int(np.asarray(fixed.ability)[0,0,0])),(2,4))   # Trace; Volt Absorb


class OwnerSpikesTests(unittest.TestCase):
    def port_carry(self):
        from gpu_engine import engine_device_public_port as port
        t=carry(SPECIES,MAPPING);hazards=np.zeros((1,2,2,2),np.float32);hazards[0,0,1]=[.25,0.];hazards[0,1,0]=[.12,0.]
        return t._replace(public_port=port.initialize(1)._replace(hazards=j.asarray(hazards)))

    def test_the_owners_shown_spikes_credit_is_the_device_hooks(self):
        from gpu_engine import engine_device_damage_history as dh
        facts=cw.LineFacts(('alpha',),((300,263,300),(299,262,300),(10,0,300)))
        out=cw.fill_owner_spikes(self.port_carry(),0,facts);h=np.asarray(out.public_port.hazards)
        pair=j.zeros(2,j.float32)
        for drop in(12,11,4):pair=dh._add(pair,j.asarray(drop,j.float32)/100)   # 100->88, 99->88 (capped below max), 4->0
        np.testing.assert_array_equal(h[0,0,0],np.asarray(pair))
        np.testing.assert_array_equal(h[0,0,1],[.25,0.]);np.testing.assert_array_equal(h[0,1],np.asarray(self.port_carry().public_port.hazards)[0,1])
        self.assertAlmostEqual(float(np.float32(h[0,0,0,0])+np.float32(h[0,0,0,1])),.27,places=6)   # the peer's OPP_HAZARD_CREDIT (high + low)
        self.assertIs(cw.fill_owner_spikes(out,0,facts),out)   # filled once: idempotent
        base=self.port_carry()
        for f in(None,cw.LineFacts(('alpha',),None),cw.LineFacts(('alpha',),())):self.assertIs(cw.fill_owner_spikes(base,0,f),base)


class BatonFillTests(unittest.TestCase):
    def test_a_baton_pass_arrival_reads_the_devices_sentinel(self):
        t=carry(SPECIES,MAPPING)
        last=np.zeros((1,2,6),np.int32);last[0,0,0]=-1;last[0,1,0]=-1;active=np.asarray([[0,0]],np.int32)
        t=t._replace(last_moves=j.asarray(last),active_slot=j.asarray(active))
        out=cw.fill_baton(t,cw.LineFacts(('beta',),(),(False,True)))
        self.assertEqual(np.asarray(out.last_moves)[0,:,0].tolist(),[-1,-2])
        self.assertIs(cw.fill_baton(out,cw.LineFacts(('beta',),(),(False,True))),out)
        moved=t._replace(last_moves=j.asarray(np.where(np.arange(6)==0,7,0)[None,None].repeat(2,1).astype(np.int32)))   # it has moved since
        self.assertIs(cw.fill_baton(moved,cw.LineFacts(('beta',),(),(True,True))),moved)
        self.assertIs(cw.fill_baton(t,cw.LineFacts(('beta',),())),t)


class PeerViewLaneTests(unittest.TestCase):
    """The real encoder (engine_device_observation on the repo's fixture tables) on a peer whose own order is not physical: self
    tokens, switch tokens, switch lanes, map_actions and the self tokens' ability / item belief columns all follow own_order[peer]."""
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_device_observation as fixtures
        fixtures.DeviceObservationTests.setUpClass()
        cls.catalog,cls.tables=fixtures.DeviceObservationTests.catalog,fixtures.DeviceObservationTests.tables

    def test_lanes_tokens_and_belief_columns_follow_the_peers_request_order(self):
        from gpu_engine import device_public_leaf as leaf,engine_device_observation as e
        from gpu_engine.public_moves import _static_category
        rows=np.asarray(self.tables.belief.species_types);peer,active=1,2
        s=battle([[1,2,3,1,2,3],[2,3,1,3,1,2]],rows,active=(0,active))
        hp=np.asarray([100,90,80,70,60,50],np.int32);ability=np.asarray([1,2,1,2,1,2],np.int32);item=np.asarray([1,0,1,0,1,0],np.int32)
        s=s._replace(hp=s.hp.at[0,peer].set(hp),base_ability=s.base_ability.at[0,peer].set(ability),ability=s.ability.at[0,peer].set(ability),
            item=s.item.at[0,peer].set(item))
        c=hooks.initialize(s,self.tables.belief);order=[2,4,0,5,1,3]
        t=c.trackers._replace(own_order=c.trackers.own_order.at[0,peer].set(j.asarray(order,j.int32)))
        engine=np.asarray([[True,True,False,True,True,True,False,True,True]])   # engine lanes 4..8: physicals 0,1,3,4,5 (active 2 skipped)
        view=leaf.from_public(s,t,self.tables,observer_sides=j.asarray([peer],j.int32),legal_action_mask=j.asarray(engine),enable_device_leaf=True)
        obs=view.observation;num,cat=np.asarray(obs.numeric_features)[0],np.asarray(obs.categorical_ids)[0]
        self.assertEqual(np.asarray(obs.switch_targets)[0].tolist(),order[1:])
        v=self.catalog.vocabulary;names=lambda kind,n:_static_category(v,f'belief:possible_{kind}:{n}')
        for k,physical in enumerate(order):   # self token 1 + k is request position k
            self.assertAlmostEqual(float(num[1+k,e._index('HP_FRACTION')]),hp[physical]/100,places=6)
            self.assertEqual(int(cat[1+k,9]),names('ability',['','immunity','pressure'][ability[physical]]))
            self.assertEqual(int(cat[1+k,11]),names('item','leftovers')if item[physical]else 0)
        for i,physical in enumerate(order[1:]):   # switch lane 4 + i, token 17 + i: request position i + 1
            lane=int(np.asarray(leaf.map_actions(view,s,j.asarray([4+i],j.int32)))[0])
            self.assertEqual(lane,4+physical-(physical>active))   # the engine's own switch lane for that physical
            self.assertEqual(bool(np.asarray(obs.legal_action_mask)[0,4+i]),bool(engine[0,lane]))
            self.assertAlmostEqual(float(num[17+i,e._index('HP_FRACTION')]),hp[physical]/100,places=6)


class B2EncodingTests(unittest.TestCase):
    """A real from_b2 root on authored lines (tests/test_phase_d_device_worlds' B2 fold) whose foe must recharge: world 0's completed
    owner view, through the device encoder, equals B2's own encoding of the literal root on every token of the 23-token prefix."""
    @classmethod
    def setUpClass(cls):
        import test_phase_d_device_worlds as D
        from gpu_engine import engine_device_observation as e
        D.WorldsTests.setUpClass();cls.fx=D.WorldsTests('test_real_b2_fold_to_16_and32_source_worlds_both_owners')
        cls.carry=cw.carry_tables(cls.fx.metadata,cls.fx.catalog);cls.observation=e.compile_tables(cls.fx.metadata,catalog=cls.fx.catalog)

    def encodings(self,body,owner=0,*,complete=True):
        from gpu_engine import device_public_leaf as leaf,phase_d_live_worlds as live
        from pokezero import v5_contract
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        f=self.fx;own,peer=f'p{owner+1}',f'p{2-owner}'
        lines=(f'|switch|{own}a: Alpha|Alpha, L80|291/291',f'|switch|{peer}a: Beta|Beta, L80|100/100','|turn|1',
            *[x.replace('OWN',own).replace('PEER',peer)for x in body])
        a=JaxRecordedAdapter(f.catalog,owner_side=owner);a.reset(stream_id=1,battle_id='authored');a.feed(lines)
        root=live.from_b2(a,f.request(owner),f.source,f.metadata,address=v5_contract.DecisionAddress(1,0,owner))
        w=live.sample_worlds(root,f.source,f.tables,count=16,seed=9).worlds[0]
        if complete:w=cw.complete_world(w,self.carry,owner,facts=cw.line_facts(lines,owner),root=root.public)
        b2=root.view['_b2_root_inputs']
        out=leaf.from_public(w.state,w.public,self.observation,observer_sides=j.asarray([owner],j.int32),
            legal_action_mask=j.asarray(np.asarray(b2['legal_action_mask'],bool)[None]),enable_device_leaf=True).observation
        return b2,np.asarray(out.categorical_ids)[0,:23],np.asarray(out.numeric_features)[0,:23]

    def test_a_recharging_foe_matches_b2(self):
        recharge=['|move|PEERa: Beta|Tackle|OWNa: Alpha','|-damage|OWNa: Alpha|250/291','|-mustrecharge|PEERa: Beta',
            '|move|OWNa: Alpha|Tackle|PEERa: Beta','|-damage|PEERa: Beta|90/100','|','|upkeep','|turn|2']
        b2,cat,num=self.encodings(recharge)
        np.testing.assert_array_equal(cat,np.asarray(b2['categorical_ids']));np.testing.assert_allclose(num,np.asarray(b2['numeric_features']),atol=1e-6)
        from gpu_engine.public_moves import _static_category
        self.assertEqual(int(cat[7,33]),_static_category(self.fx.catalog.vocabulary,'volatile:mustrecharge'))   # the foe's active token
        _,raw,_=self.encodings(recharge,complete=False);self.assertEqual(int(raw[7,33]),0)   # from_b2's copy of B2's bag alone lacks it
        b2,cat,_=self.encodings(recharge[:-1]+['|turn|2','|cant|PEERa: Beta|recharge','|','|upkeep','|turn|3'])   # recharged: gone on both
        np.testing.assert_array_equal(cat,np.asarray(b2['categorical_ids']));self.assertEqual(int(cat[7,33]),0)


# Private reveals: Showdown's Gen 3 Pressure start reaches the holder's side alone (constructor_worlds' module docstring).
PRIVATE_LINES=['|switch|p1a: Birdy|Alpha, L80|100/100','|-ability|p1a: Birdy|Pressure|[silent]','|switch|p2a: Delta|Delta, L80|100/100',
    '|-ability|p2a: Delta|Pressure|[silent]',   # the opponent's own private line (never on our wire): never the owner's
    '|switch|p1a: Beta|Beta, L80|100/100','|-ability|p1a: Beta|Pressure|[silent]',
    '|-ability|p2a: Delta|Pressure|Trace|[from] ability: Trace|[of] p1a: Beta',   # the opponent's tracer copies Beta's Pressure: public
    '|switch|p1a: Gamma|Gamma, L80|100/100','|-ability|p1a: Gamma|Pressure|[silent]','|-ability|p1a: Gamma|Pressure',   # a public -ability
    '|switch|p1a: Epsilon|Epsilon, L80|100/100','|-ability|p1a: Epsilon|Pressure|[silent]',
    '|-heal|p1a: Epsilon|100/100|[from] ability: Volt Absorb|[of] p2a: Delta',   # another ability's attribution keeps nothing
    '|switch|p1a: Zeta|Zeta, L80|100/100','|-ability|p1a: Zeta|Pressure|[silent]','|-damage|p2a: Delta|50/100|[from] ability: Pressure|[of] p1a: Zeta',
    '|turn|1']


def reveal_tables():
    return cw._table(np.zeros((len(PRESSURE_NAMES),2),np.int32),species_names=PRESSURE_NAMES,ability_names=ABILITIES)


class PrivateRevealTests(unittest.TestCase):
    def test_line_facts_keep_only_the_owners_side_private_reveals(self):
        """Own [silent] Pressure (by species, through a nickname) is private; a Trace copy [of] our member, a public -ability line
        and a [from] ability: Pressure attribution keep it public; another ability's attribution does not; the opponent side is
        never the owner's."""
        facts=cw.line_facts(PRIVATE_LINES,0)
        self.assertEqual(facts.private_reveals,((('alpha','pressure'),('epsilon','pressure')),()))
        self.assertEqual(cw.line_facts(PRIVATE_LINES[:4],1).private_reveals,((),(('delta','pressure'),)))   # owner p2: its own line
        # later lines attribute Pressure to Delta (the subject of its Trace copy and of a [from] ability: Pressure line): an
        # over-attribution, which only keeps a reveal public (the conservative side)
        self.assertEqual(cw.line_facts(PRIVATE_LINES,1).private_reveals,((),()))
        self.assertEqual(cw.line_facts(PRIVATE_LINES[:2],0).private_reveals,((('alpha','pressure'),),()))
        tagged=['|switch|p1a: Alpha|Alpha, L80|100/100','|-ability|p1a: Alpha|Pressure|[silent]|[of] p2a: Delta']   # a tag: not the private form
        self.assertEqual(cw.line_facts(tagged,0).private_reveals,((),()))
        self.assertEqual(cw.line_facts(['|switch|p1a: Alpha|Alpha, L80|100/100','|turn|1'],0),cw.LineFacts((),()))

    def carry(self,*,borrowed=0):
        """A from_b2-style carry, owner p1: Alpha (Pressure, shown; `borrowed` its running ability), Beta (Pressure), Gamma (Swift
        Swim) on the owner side; Delta (Pressure) on the peer side."""
        t=carry([[1,2,3,0,0,0],[4,0,0,0,0,0]],[[0,1,2,-1,-1,-1],[0,-1,-1,-1,-1,-1]])
        original=np.asarray([[[1,1,3,0,0,0],[1,0,0,0,0,0]]],np.int32);borrow=np.zeros((1,2,6),np.int32);borrow[0,0,0]=borrowed
        shown=np.where(borrow>0,borrow,original).astype(np.int32)
        return t._replace(original_ability=j.asarray(original),shown_ability=j.asarray(shown),borrowed_ability=j.asarray(borrow))

    def test_the_redaction_clears_the_owners_private_members_only(self):
        lines=['|switch|p1a: Alpha|Alpha, L80|100/100','|-ability|p1a: Alpha|Pressure|[silent]','|switch|p2a: Delta|Delta, L80|100/100',
            '|-ability|p2a: Delta|Pressure|[silent]','|switch|p1a: Gamma|Gamma, L80|100/100','|-ability|p1a: Gamma|Pressure|[silent]',
            '|switch|p1a: Omega|Omega, L80|100/100','|-ability|p1a: Omega|Pressure|[silent]']
        facts=cw.line_facts(lines,0);t=self.carry();tables=reveal_tables()
        done=cw.redact_private_reveals(t,0,facts,tables)
        self.assertEqual(np.asarray(done.original_ability)[0].tolist(),[[0,1,3,0,0,0],[1,0,0,0,0,0]])   # Alpha only; Delta (peer) kept
        self.assertEqual(np.asarray(done.shown_ability)[0].tolist(),[[0,1,3,0,0,0],[1,0,0,0,0,0]])
        self.assertIs(done.borrowed_ability,t.borrowed_ability)
        same=('original_ability','shown_ability')
        self.assertTrue(all(x is y for(_,x),(_,y)in zip(leaves(t,same),leaves(done,same))))
        self.assertIs(cw.redact_private_reveals(done,0,facts,tables),done)   # a second pass is a no-op
        record=cw.private_reveal_record(t,0,facts,tables,False)
        self.assertEqual(record,dict(enabled=False,found=[dict(species=s,ability='pressure')for s in('alpha','gamma','omega')],
            applicable=[dict(species='alpha',ability='pressure',slot=0)],redacted=[],
            unmatched=[dict(species='gamma',ability='pressure',reason='not_shown'),dict(species='omega',ability='pressure',reason='no_member')]))
        self.assertEqual(cw.private_reveal_record(t,0,facts,tables,True)['redacted'],[dict(species='alpha',ability='pressure',slot=0)])
        self.assertIsNone(cw.private_reveal_record(t,0,None,tables,True))
        theirs=cw.redact_private_reveals(t,1,cw.line_facts(lines,1),tables)   # owner p2: its own Delta, never p1's members
        self.assertEqual(np.asarray(theirs.original_ability)[0].tolist(),[[1,1,3,0,0,0],[0,0,0,0,0,0]])
        running=self.carry(borrowed=2);done=cw.redact_private_reveals(running,0,facts,tables)   # a running copy stays shown
        self.assertEqual((np.asarray(done.original_ability)[0,0,0],np.asarray(done.shown_ability)[0,0,0],np.asarray(done.borrowed_ability)[0,0,0]),(0,2,2))


class PrivateRevealB2Tests(unittest.TestCase):
    """The real B2 fold (tests/test_phase_d_device_worlds' authored lines) on the owner's own side-private Pressure line: B2 records it
    (the leak), the redaction clears it from every world; the owner's own view (and its parity with B2's encoding) is unchanged and
    only the peer's view of that member changes (REVEALED_ABILITY); a Trace copy of it keeps it."""
    @classmethod
    def setUpClass(cls):
        import test_phase_d_device_worlds as D
        from gpu_engine import engine_device_observation as e
        D.WorldsTests.setUpClass();cls.fx=D.WorldsTests('test_real_b2_fold_to_16_and32_source_worlds_both_owners')
        cls.carry=cw.carry_tables(cls.fx.metadata,cls.fx.catalog);cls.observation=e.compile_tables(cls.fx.metadata,catalog=cls.fx.catalog)
        cls.e=e

    def root(self,owner,body):
        from gpu_engine import phase_d_live_worlds as live
        from pokezero import v5_contract
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        f=self.fx;own,peer=f'p{owner+1}',f'p{2-owner}'
        lines=(f'|switch|{own}a: Alpha|Alpha, L80|291/291',*[x.replace('OWN',own).replace('PEER',peer)for x in body])
        a=JaxRecordedAdapter(f.catalog,owner_side=owner);a.reset(stream_id=1,battle_id='authored');a.feed(lines)
        root=live.from_b2(a,f.request(owner),f.source,f.metadata,address=v5_contract.DecisionAddress(1,0,owner))
        return root,live.sample_worlds(root,f.source,f.tables,count=16,seed=9),cw.line_facts(lines,owner)

    def view(self,world,side,b2):
        from gpu_engine import device_public_leaf as leaf
        mask=j.asarray(np.asarray(b2['legal_action_mask'],bool)[None])
        out=leaf.from_public(world.state,world.public,self.observation,observer_sides=j.asarray([side],j.int32),legal_action_mask=mask,
            enable_device_leaf=True).observation
        return{k:np.asarray(getattr(out,k))for k in('numeric_features','categorical_ids','token_type_ids','attention_mask')}

    def test_the_owners_private_pressure_leaves_the_peers_view_only(self):
        pressure=self.carry.ability_names.index('pressure');revealed=self.e._index('REVEALED_ABILITY')
        for owner in(0,1):
            with self.subTest(owner=owner):
                root,made,facts=self.root(owner,['|-ability|OWNa: Alpha|Pressure|[silent]','|switch|PEERa: Beta|Beta, L80|100/100','|turn|1'])
                self.assertEqual(int(np.asarray(root.public.original_ability)[0,owner,0]),pressure)   # B2's fold records it: the leak
                self.assertEqual(facts.private_reveals[owner],(('alpha','pressure'),))
                b2=root.view['_b2_root_inputs'];peer=1-owner
                for w in made.worlds:
                    off=cw.complete_world(w,self.carry,owner,facts=facts,root=root.public)
                    on=cw.complete_world(w,self.carry,owner,facts=facts,root=root.public,redact_private_reveals=True)
                    o,f=np.asarray(on.public.original_ability)[0],np.asarray(off.public.original_ability)[0]
                    self.assertEqual((int(f[owner,0]),int(o[owner,0]),int(np.asarray(on.public.shown_ability)[0,owner,0])),(pressure,0,0))
                    np.testing.assert_array_equal(o[peer],f[peer]);self.assertIs(on.state,off.state)
                    self.assertIs(cw.complete_world(on,self.carry,owner,facts=facts,root=root.public,redact_private_reveals=True),on)
                    if w.index>1:continue   # the views on two worlds (every world's carry is redacted alike above)
                    mine_on,mine_off=self.view(on,owner,b2),self.view(off,owner,b2)
                    for k in mine_on:np.testing.assert_array_equal(mine_on[k],mine_off[k],err_msg=k)   # the owner's view: unchanged
                    if w.index==0:   # root/B2 parity of world 0's owner view holds with the redaction
                        np.testing.assert_array_equal(mine_on['categorical_ids'][0,:23],np.asarray(b2['categorical_ids']))
                        np.testing.assert_allclose(mine_on['numeric_features'][0,:23],np.asarray(b2['numeric_features']),atol=1e-6)
                    theirs_on,theirs_off=self.view(on,peer,b2),self.view(off,peer,b2)
                    changed=np.argwhere(theirs_on['numeric_features']!=theirs_off['numeric_features']).tolist()
                    self.assertEqual(changed,[[0,7,revealed]])   # the peer's view of our Alpha (its foe token 7): REVEALED_ABILITY only
                    self.assertEqual((theirs_off['numeric_features'][0,7,revealed],theirs_on['numeric_features'][0,7,revealed]),(1.,0.))
                    for k in('categorical_ids','token_type_ids','attention_mask'):np.testing.assert_array_equal(theirs_on[k],theirs_off[k],err_msg=k)

    def test_a_trace_copy_of_it_keeps_it_public(self):
        root,made,facts=self.root(0,['|-ability|OWNa: Alpha|Pressure|[silent]','|switch|PEERa: Beta|Beta, L80|100/100',
            '|-ability|PEERa: Beta|Pressure|Trace|[from] ability: Trace|[of] OWNa: Alpha','|turn|1'])
        self.assertEqual(facts.private_reveals,((),()))
        pressure=self.carry.ability_names.index('pressure')
        self.assertEqual(int(np.asarray(root.public.original_ability)[0,0,0]),pressure)   # public through the copy's [of]
        for w in made.worlds:self.assertIs(cw.redact_private_reveals(w.public,0,facts,self.carry),w.public)   # (the fixture has no Trace
        # in its vocabulary, so the full completion's Trace fix cannot run here: the redaction step alone)
        self.assertEqual(cw.private_reveal_record(root.public,0,facts,self.carry,True),dict(enabled=True,found=[],applicable=[],redacted=[],unmatched=[]))


if __name__=='__main__':unittest.main()
