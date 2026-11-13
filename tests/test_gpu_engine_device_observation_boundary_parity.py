"""Observed clock/stats/last-move meanings against original pure writers.

Synthetic source transactions only: no engine turn, actor, Native game or replay.
"""
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_device_observation as e, engine_device_public_hooks as h
from gpu_engine import engine_device_trackers as tr, engine_device_features as f, public_tendency as ref
from gpu_engine.state import empty_state
from pokezero import showdown as s
s_=s

BASE='03be68de16f4265f5f9c226ce6ada80b8dd8619d'


class DeviceBoundaryParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_device_observation as fixture
        fixture.DeviceObservationTests.setUpClass()
        cls.fixture=fixture.DeviceObservationTests();cls.catalog=fixture.DeviceObservationTests.catalog
        cls.tables=fixture.DeviceObservationTests.tables;cls.metadata=fixture.DeviceObservationTests.metadata
        cls.original=ref.compile_tendency(cls.catalog.species,cls.catalog.moves,cls.catalog.vocabulary)

    def state(self):
        b=2;z=empty_state(b);old,_,_,_=self.fixture.facts(b)
        values={name:getattr(old,name)for name in old._fields if name in z._fields}
        values.update(active=j.zeros((b,2),j.int32),turn=j.zeros(b,j.int32),
            species=j.broadcast_to(j.array([[2,1,0,0,0,0],[3,1,0,0,0,0]],j.int32),(b,2,6)))
        return z._replace(**values)

    def reference(self):
        return ref.initialize(2,self.original.device,species_binding=self.original.device.species_binding,complete_prefix=True)

    def line(self,state,line):
        fact=ref.decode_line(line,catalog=self.original)
        out,valid=ref.apply(state,self.original.device,fact,producer_binding=self.original.device.binding)
        self.assertTrue(bool(valid.all()),line);return out

    def compare_tendency(self,c,state,reference):
        for owner in (0,1):
            owners=j.full(2,owner,j.int32);view=tr.readout(c.trackers,owners)
            species=j.asarray([[self.catalog.species.species_id(self.metadata['vocabulary']['species'][int(x)])if int(x)else 0
                for x in row]for row in np.asarray(view.species)],j.int32)
            own=[self.catalog.species.species_id(self.metadata['vocabulary']['species'][int(state.species[r,owner,state.active[r,owner]])])for r in range(2)]
            expected=ref.project(reference,self.original.device,observer_side=owners,opponent_species=species,
                own_active_species=j.asarray(own,j.int32),expected_vocabulary=self.original.device.vocabulary_binding)
            n,_,_=f.project_mon(j.zeros((2,6,132),j.float32),j.zeros((2,6,41),j.int32),view.mon_features,
                active=view.active,present=view.present,tables=self.tables.features,status=view.status)
            stat,cat,on,bad=f.project_stats(view.stats_features,self.tables.features,2)
            np.testing.assert_allclose(n[...,j.asarray(ref.MON_COLUMNS)],expected.numeric[...,j.asarray(ref.MON_COLUMNS)],atol=1e-7)
            for key in ('opp_switch_count','opp_decision_opportunities','my_switch_turns'):
                col=s.v4_numeric_index(getattr(s,'NUMERIC_'+f.STATS_FIELDS[key][0]))
                np.testing.assert_allclose(stat[:,col],expected.stats_numeric[:,col],atol=1e-7)
            np.testing.assert_array_equal(cat[:,s.CATEGORY_ROLE],expected.stats_role)
            self.assertTrue(bool(on.all()));self.assertFalse(bool(bad.any()))
            self.assertFalse(any(key in view.own_mon_features for key in f.TENDENCY_FIELDS))

    def test_initial_and_next_boundary_literal_columns_original_writers(self):
        state=self.state();c=h.initialize(state,self.tables.belief)
        for turn in (1,2):
            if turn==2:
                h.move(c,state,0,j.ones(2,j.int32),j.ones(2,bool),pp_use=True)
                h.move(c,state,1,j.ones(2,j.int32),j.ones(2,bool),pp_use=True)
                after=state._replace(turn=j.ones(2,j.int32));h.turn_end(c,state,after,j.ones(2,bool));state=after
            for owner in (0,1):
                owners=j.full(2,owner,j.int32);view=tr.readout(c.trackers,owners)
                out=e.encode(state,view,self.tables,observer_sides=owners,legal_action_mask=j.ones((2,9),bool))
                self.assertAlmostEqual(float(out.numeric_features[0,0,e._index('TURN_COUNT')]),turn/1000,places=7)
                self.assertTrue(bool(out.attention_mask[:,22].all()))
                self.assertTrue(bool((out.numeric_features[:,22,e._index('PRESENT')]==1).all()))
                np.testing.assert_array_equal(out.categorical_ids[:,22,s.CATEGORY_ROLE],self.tables.features.stats_role)
                for token in (1,7):
                    # Use the original exact-state writer for active-age meaning.
                    nums=[0.]*200
                    exact=NS(turns_active=turn,revealed_ability=None,possible_abilities=(),ruled_out_abilities=())
                    s._encode_mon_exact_state(nums,NS(active=True,ability=None),exact,
                        role='self'if token==1 else'opponent',status='',fainted=False)
                    self.assertAlmostEqual(float(out.numeric_features[0,token,e._index('TURNS_ACTIVE')]),nums[s.NUMERIC_TURNS_ACTIVE],places=7)
                self.assertEqual(float(out.numeric_features[0,1,e._index('MON_TURNS_ACTIVE_TOTAL')]),0)
                self.assertEqual(float(out.numeric_features[0,1,e._index('MON_STAYED_AND_ATTACKED')]),0)
                if turn==1:
                    for token in (1,7,23):self.assertEqual(int(out.categorical_ids[0,token,s.CATEGORY_LAST_USED_MOVE]),int(self.tables.features.switch_categories[0]))
                else:self.assertAlmostEqual(float(out.numeric_features[0,7,e._index('MON_TURNS_ACTIVE_TOTAL')]),2/64)

    def test_original_first_stay_switch_matchup_and_opportunity_algebra(self):
        state=self.state();c=h.initialize(state,self.tables.belief);r=self.reference()
        for line in ('|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100','|turn|1'):
            r=self.line(r,line)
        self.compare_tendency(c,state,r)
        yes=j.ones(2,bool);old=state.active[:,1]
        h.switch_decision(c,state,1,yes,baton=j.zeros(2,bool))
        state=state._replace(active=state.active.at[:,1].set(1));h.install(c,state,1,yes);h.withdraw(c,state,1,old,yes)
        r=self.line(r,'|switch|p2a: Gardevoir|Gardevoir, L80|100/100');self.compare_tendency(c,state,r)
        for turn in (1,2):
            h.move(c,state,0,j.ones(2,j.int32),yes,pp_use=True)
            r=self.line(r,'|move|p1a: Snorlax|Surf|p2a: Gardevoir');self.compare_tendency(c,state,r)
            if turn==1:
                after=state._replace(turn=j.ones(2,j.int32));h.turn_end(c,state,after,yes);state=after
                r=self.line(r,'|turn|2');self.compare_tendency(c,state,r)
        # A second declaration of the same stay is NOT another stayed datum.
        self.assertEqual(int(c.trackers.mon_features['stayed_and_attacked'][0,0,0]),1)

    def test_mixed_rows_pending_replacement_terminal_and_stats_mask(self):
        state=self.state();c=h.initialize(state,self.tables.belief);before=c.trackers
        def move_tracker(st,trackers):
            current=h.Context(trackers,self.tables.belief)
            h.move(current,st,j.array([0,1],j.int32),j.ones(2,j.int32),j.array([True,False]),pp_use=True)
            return current.trackers
        c.trackers=jax.jit(move_tracker)(state,c.trackers)
        self.assertEqual(c.trackers.tendency.opportunities.tolist(),[[1,0],[0,0]])
        after=state._replace(turn=j.ones(2,j.int32),pending_residual=j.array([True,False]),forced_switch=j.array([[False,False],[True,False]]))
        h.turn_end(c,state,after,j.ones(2,bool));self.assertEqual(c.trackers.turn.tolist(),[1,1])
        ready=after._replace(pending_residual=j.zeros(2,bool),forced_switch=j.zeros((2,2),bool))
        h.turn_end(c,after,ready,j.zeros(2,bool));self.assertEqual(c.trackers.turn.tolist(),[2,2])
        h.turn_end(c,after,ready,j.zeros(2,bool));self.assertEqual(c.trackers.turn.tolist(),[2,2])
        terminal=ready._replace(done=j.ones(2,bool),turn=j.full(2,2,j.int32))
        h.turn_end(c,ready,terminal,j.ones(2,bool));self.assertEqual(c.trackers.turn.tolist(),[2,2])
        n,cat,on,bad=f.project_stats({'opp_switch_count':j.ones(2)},self.tables.features._replace(stats_enabled=j.array(False)),2)
        self.assertFalse(bool(n.any()));self.assertFalse(bool(cat.any()));self.assertFalse(bool(on.any()))
        # An explicit own tendency input cannot populate opponent-only columns.
        n,_,_=f.project_mon(j.zeros((2,6,132)),j.zeros((2,6,41),j.int32),{'stayed_and_attacked':j.ones((2,6))},
            active=j.ones((2,6),bool),present=j.ones((2,6),bool),tables=self.tables.features,opponent_tendency=False)
        self.assertFalse(bool(n.any()))
        n,_,_=f.project_mon(j.zeros((2,6,132)),j.zeros((2,6,41),j.int32),{'stayed_and_attacked':j.ones((2,6))},
            active=j.ones((2,6),bool),present=j.ones((2,6),bool),tables=self.tables.features._replace(stats_enabled=j.array(False)))
        self.assertFalse(bool(n.any()))

    def test_transition_only_changes_actual_observer_call_and_original_physics_bytes(self):
        root=Path(__file__).resolve().parents[1]
        original=subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],cwd=root,text=True)
        # This exact two-boundary-call proof concerns the original change.
        # Retain it at that immutable feature commit; current tracker behavior
        # is exercised by the other tests and the full integrated source pins.
        introduced='693e7ffb2acd420fc692619229d276e129f25bf5'
        current=subprocess.check_output(['git','show',introduced+':gpu_engine/transition.py'],cwd=root,text=True)
        new="device_public.switch_decision(_device_public,work,side,do_switch & ~replacement,baton=_device_baton)"
        old="device_public.feature(_device_public,side,old_active,'switched_before_attack',jnp.ones_like(old_active),do_switch & ~replacement,add=True)"
        self.assertEqual(current.count(new),1);self.assertEqual(current.replace(new,old),original)
        for name in ('state.py','mechanics.py','entry.py','damage_moves.py','secondary.py'):
            self.assertEqual(subprocess.check_output(['git','show',introduced+':gpu_engine/'+name],cwd=root),subprocess.check_output(['git','show',BASE+':gpu_engine/'+name],cwd=root))

    def test_original_called_child_lastmove_and_preparation_header_are_distinct(self):
        import test_gpu_engine_public_moves as fixture
        fixture.PublicMovesTests.setUpClass();original=fixture.PublicMovesTests()
        moves,beliefs,mons=original.initial(batch=2)
        for line in ('|move|p1a: Snorlax|Body Slam|p2a: Absol',
                     '|move|p1a: Snorlax|Surf|p2a: Absol|[from] move: Sleep Talk'):
            value=original.apply_line(moves,beliefs,mons,line)
            self.assertTrue(bool(value.admitted.all()));moves,beliefs=value.state,value.beliefs
        state=self.state();c=h.initialize(state,self.tables.belief);yes=j.ones(2,bool)
        h.move(c,state,0,j.full(2,2,j.int32),yes,pp_use=True)
        h.move(c,state,0,j.ones(2,j.int32),yes,pp_use=False,count_turn=False)
        self.assertEqual(original.catalog.names[int(moves.last_id[0,0])],
            self.metadata['vocabulary']['moves'][int(c.trackers.last_moves[0,0,0])])
        self.assertEqual(c.trackers.tendency.opportunities.tolist(),[[1,0],[1,0]])
        # Preparation-only -singleturn remains a reveal, not a MOVE header;
        # only the subsequent ordinary header supplies lastmove/tendency.
        fresh=h.initialize(state,self.tables.belief)
        h.move(fresh,state,0,j.ones(2,j.int32),yes,pp_use=False,count_turn=False,preparation=True)
        self.assertEqual(int(fresh.trackers.last_moves[0,0,0]),-1)
        self.assertEqual(int(fresh.trackers.mon_features['stayed_and_attacked'][0,0,0]),0)
        self.assertEqual(int(fresh.trackers.tendency.opportunities[0,0]),0)
        self.assertEqual(int(fresh.trackers.observed_moves[0,0,0,0]),1)
        h.move(fresh,state,0,j.ones(2,j.int32),yes,pp_use=True)
        self.assertEqual(int(fresh.trackers.mon_features['stayed_and_attacked'][0,0,0]),1)

    def b2_active(self):
        """B2's public active consumer (public_active.apply) over a fixture vocabulary, batch 1."""
        from gpu_engine import public_active as a, public_mon_state as m
        catalog=a.compile_active(self.catalog.vocabulary);t=catalog.device;sb=self.catalog.species.device.binding
        box=dict(state=a.initialize(1,t,species_binding=sb),mons=m.initialize(1,species_binding=sb))
        def fact(op,side=-1,value=0,slot=0,bp=False):
            st,mons=box['state'],box['mons']
            if op==a.ARRIVE:
                mons,ok=m.arrive(mons,side=side,slot=slot,encounter=slot+1,species=slot+1,source_species=0,level=80,gender=0,
                    hp_numerator=100,hp_denominator=100,status=0,kind=m.BATON_PASS if bp else m.SWITCH)
                self.assertTrue(bool(ok.all()));box['mons']=mons
            enc=0 if side<0 else int(mons.encounter[0,side,int(mons.active[0,side])])
            st,ok=a.apply(st,t,mons,operation=op,side=side,encounter=enc,value=value,baton_pass=bp,producer_binding=t.binding)
            self.assertTrue(bool(ok.all()),op);box['state']=st
        def project():
            return a.project(box['state'],t,box['mons'],expected_vocabulary=t.vocabulary_binding)
        return a,catalog,fact,project

    def compare_active(self,trackers,expected,a,*,equal=True):
        cols=j.asarray(a.NUMERIC_COLUMNS);cats=j.asarray(a.CATEGORY_COLUMNS);same=True
        for seat in (0,1):
            view=tr.readout(trackers,j.full(2,1-seat,j.int32))
            n,cat,_=f.project_mon(j.zeros((2,6,132),j.float32),j.zeros((2,6,41),j.int32),view.mon_features,
                active=view.active,present=view.present,tables=self.tables.features,volatiles=view.volatiles)
            slot=int(trackers.active_slot[0,seat])
            same &= bool(np.allclose(n[0,slot,cols],expected.numeric[0,seat,cols],atol=1e-7))
            same &= bool(np.array_equal(cat[0,slot,cats],expected.categorical[0,seat,cats]))
            if equal:
                np.testing.assert_allclose(n[0,slot,cols],expected.numeric[0,seat,cols],atol=1e-7,err_msg=str(seat))
                np.testing.assert_array_equal(cat[0,slot,cats],expected.categorical[0,seat,cats],err_msg=str(seat))
        return same

    def test_faint_clear_matches_b2_faint_on_the_fainted_seat_and_both_mean_looks(self):
        a,catalog,fact,project=self.b2_active()
        state=self.state();c=h.initialize(state,self.tables.belief);yes=j.ones(2,bool)
        # One synthetic transaction on both sides: seat 1 Substitute + confusion + Leech Seed,
        # one successful stall, a public Trick swap and one TURN of confusion; seat 0 Mean Looked.
        fact(a.ARRIVE,0);fact(a.ARRIVE,1)
        for name in ('confusion','substitute','leechseed'):
            fact(a.ADD,1,catalog.ids[name]);h.volatile(c,state,1,name,yes,yes)
        fact(a.STALL_ATTEMPT,1,1);fact(a.STALL_SUCCESS,1);fact(a.ITEM_SET,1,2);fact(a.TRAPPED,0);fact(a.TURN,-1,1)
        for side,name,value in ((1,'confusion_turns',1),(1,'stall_counter',1),(1,'item_swapped',1),(1,'sub_present',1),(0,'meanlook_trap',1)):
            h.feature(c,side,state.active[:,side],name,j.full(2,value,j.int32),yes)
        self.compare_active(c.trackers,project(),a)
        self.assertTrue(bool(project().numeric[0,1,j.asarray(a.NUMERIC_COLUMNS)].any()))
        # Seat 1's active faints: B2 FAINT; device turn_end over the same HP transition.
        fact(a.FAINT,1);expected=project()
        after=state._replace(hp=state.hp.at[:,1,0].set(0))
        before=c.trackers
        h.turn_end(c,state,after,j.zeros(2,bool))
        self.compare_active(c.trackers,expected,a)
        self.assertFalse(bool(expected.numeric[0,:,j.asarray(a.NUMERIC_COLUMNS)].any()))
        # Without the clear (the old device) the fainted active's bag and clocks stayed visible.
        old=h.Context(before,self.tables.belief);h.observe(old,after,yes)
        self.assertFalse(self.compare_active(old.trackers,expected,a,equal=False))

    def test_mean_look_clears_on_both_seats_at_every_arrival_like_b2(self):
        """B2 ARRIVE (public_active.py:246) clears Mean Look on both seats, Baton Pass included.

        Showdown agrees for a switch or drag (the linked trapped/trapper volatiles drop) but keeps
        the trap through Baton Pass in both directions (gen3: noCopy false; copyVolatileFrom relinks).
        The device matches B2 there on purpose (lead decision; the stale recharge bit precedent)."""
        import json,os
        root=os.environ['POKEZERO_SHOWDOWN_ROOT']
        code="""const {Battle}=require(process.argv[1]+'/dist/sim/battle.js');
function mon(species){return {name:species,species,ability:'Pressure',item:'',moves:['meanlook','batonpass','roar','splash'],nature:'Hardy',
 level:80,gender:'M',evs:{hp:84,atk:84,def:84,spa:84,spd:84,spe:84},ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31}};}
function run(choices){const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 b.setPlayer('p1',{name:'A',team:[mon('Umbreon'),mon('Espeon')]});b.setPlayer('p2',{name:'B',team:[mon('Pikachu'),mon('Raichu')]});
 for(const [x,y] of choices)b.makeChoices(x,y);const req=b.sides[1].activeRequest;
 return {trapped:!!req.active[0].trapped,volatile:!!b.sides[1].active[0].volatiles.trapped};}
console.log(JSON.stringify({switch:run([['move 1','move 4'],['switch 2','move 4']]),drag:run([['move 1','move 4'],['move 4','move 3']]),
 victim_baton:run([['move 1','move 4'],['move 4','move 2'],['','switch 2']]),trapper_baton:run([['move 1','move 4'],['move 2','move 4'],['switch 2','']])}));"""
        showdown=json.loads(subprocess.check_output(['node','-e',code,root],text=True))
        self.assertEqual({k:v['trapped'] for k,v in showdown.items()},
            dict(switch=False,drag=False,victim_baton=True,trapper_baton=True))
        for kind,side,bp in (('switch',0,False),('drag',0,False),('victim_baton',1,True),('trapper_baton',0,True)):
            a,catalog,fact,project=self.b2_active()
            state=self.state();c=h.initialize(state,self.tables.belief);yes=j.ones(2,bool)
            fact(a.ARRIVE,0);fact(a.ARRIVE,1);fact(a.TRAPPED,1);fact(a.STALL_ATTEMPT,1,1);fact(a.STALL_SUCCESS,1)
            for name,value in (('meanlook_trap',1),('stall_counter',1)):
                h.feature(c,1,state.active[:,1],name,j.full(2,value,j.int32),yes)
            self.compare_active(c.trackers,project(),a)
            self.assertTrue(bool(project().numeric[0,1,a.NUMERIC_COLUMNS[a.CORE.index('MEANLOOK_TRAP')]]))
            fact(a.ARRIVE,side,slot=1,bp=bp)
            old=state.active[:,side];after=state._replace(active=state.active.at[:,side].set(1))
            if kind=='drag':h.drag(c,state,after,side,None,yes)
            else:
                h.install(c,after,side,yes)
                if bp:h.baton_copy(c,after,side,old,yes)
                h.withdraw(c,after,side,old,yes)
            expected=project();self.compare_active(c.trackers,expected,a)
            self.assertFalse(bool(expected.numeric[0,:,a.NUMERIC_COLUMNS[a.CORE.index('MEANLOOK_TRAP')]].any()),kind)
            # Showdown's request says trapped after a Baton Pass; B2 and the device both show 0 (the quirk).
            self.assertEqual(showdown[kind]['trapped'],bp,kind)

    def test_entry_reveals_follow_showdowns_shared_lines(self):
        import json,os
        from types import SimpleNamespace as NS
        from pokezero.belief import _confirmed_ability_from_event
        from pokezero.showdown import _public_event_from_line
        root=os.environ['POKEZERO_SHOWDOWN_ROOT']
        code="""const {Battle}=require(process.argv[1]+'/dist/sim/battle.js');
function mon(species,ability,moves){return {name:species,species,ability,item:'',moves,nature:'Hardy',level:80,gender:'M',
 evs:{hp:84,atk:84,def:84,spa:84,spd:84,spe:84},ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31}};}
function run(p1,p2,choices){const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 b.setPlayer('p1',{name:'A',team:p1});b.setPlayer('p2',{name:'B',team:p2});for(const [x,y] of choices)b.makeChoices(x,y);return b.log;}
const karp=[mon('Magikarp','Swift Swim',['splash']),mon('Gyarados','Intimidate',['splash'])];
console.log(JSON.stringify({
 sub:run([mon('Smeargle','Own Tempo',['substitute','splash'])],karp,[['move 1','move 1'],['move 2','switch 2']]),
 open:run([mon('Smeargle','Own Tempo',['splash'])],karp,[['move 1','switch 2']]),
 sand:run([mon('Tyranitar','Sand Stream',['splash'])],[mon('Tyranitar','Sand Stream',['splash'])],[]),
 rain:run([mon('Smeargle','Own Tempo',['raindance','splash'])],[mon('Magikarp','Swift Swim',['splash']),mon('Politoed','Drizzle',['splash'])],
  [['move 1','move 1'],['move 2','switch 2']])}));"""
        logs=json.loads(subprocess.check_output(['node','-e',code,root],text=True))
        def seen_by(log,observer):
            out,i=[],0
            while i<len(log):
                if log[i].startswith('|split|'):
                    out.append(log[i+1] if log[i].split('|')[2]==observer else log[i+2]);i+=3
                else:out.append(log[i]);i+=1
            return out
        def revealed(log,holder):
            # The B2/champion ability attribution: -ability lines and [from] ability tags.
            found=set()
            for line in seen_by(log,'p1' if holder=='p2' else 'p2'):
                ident,name=_confirmed_ability_from_event(_public_event_from_line(line))
                if ident and name and ident.startswith(holder):found.add(name)
            return found
        self.assertEqual(revealed(logs['sub'],'p2'),set())
        self.assertTrue(any(x.startswith('|-hint|') for x in logs['sub']))
        self.assertEqual(revealed(logs['open'],'p2'),{'Intimidate'})
        sand={side:revealed(logs['sand'],side) for side in ('p1','p2')}
        self.assertEqual(sorted(len(x) for x in sand.values()),[0,1])          # exactly one Sand Stream is public
        self.assertEqual(revealed(logs['rain'],'p2'),{'Drizzle'})               # re-set over temporary Rain Dance
        # The device entry hook over the same transactions.
        # Fixture ability ids stand in for the holders: 1 Intimidate, 2 Sand Stream.
        flag=j.zeros(3,bool);SAND=3
        table=NS(weather_ability=j.zeros(3,j.int32).at[2].set(SAND),intimidate=flag.at[1].set(True),
            attack_drop_immune=flag,trace=None,own_tempo=None)
        state=self.state();yes=j.ones(2,bool)
        gyarados=state._replace(ability=state.ability.at[:,1,0].set(1),substitute_hp=state.substitute_hp.at[0,0,0].set(60))
        c=h.initialize(gyarados,self.tables.belief);h.entry(c,gyarados,gyarados,1,table,yes)
        self.assertEqual([bool(x) for x in c.trackers.shown_ability[:,1,0]!=0],
            [bool(revealed(logs['sub'],'p2')),bool(revealed(logs['open'],'p2'))])
        tyranitars=state._replace(ability=state.ability.at[:,:,0].set(2))
        c=h.initialize(tyranitars,self.tables.belief);first=1 if sand['p2'] else 0
        sanded=tyranitars._replace(weather=j.full(2,SAND,j.int32))
        h.entry(c,tyranitars,sanded,first,table,yes);h.entry(c,sanded,sanded,1-first,table,yes)
        for side in (0,1):
            self.assertEqual(bool(c.trackers.shown_ability[0,side,0]),bool(sand[f'p{side+1}']))
        # Drizzle into Rain Dance's temporary rain (row 0) sets it permanent with a line; into
        # permanent rain (row 1) it is the silent no-op of the Sand Stream case.
        rain=NS(weather_ability=j.zeros(3,j.int32).at[2].set(1),intimidate=flag,attack_drop_immune=flag,trace=None,own_tempo=None)
        politoed=state._replace(ability=state.ability.at[:,1,0].set(2),weather=j.ones(2,j.int32),weather_counter=j.array([4,0],j.int32))
        c=h.initialize(politoed,self.tables.belief)
        h.entry(c,politoed,politoed._replace(weather_counter=j.zeros(2,j.int32)),1,rain,yes)
        self.assertEqual((c.trackers.shown_ability[:,1,0]!=0).tolist(),[True,False])

    def test_public_order_replays_showdowns_request_order(self):
        """Showdown's side.pokemon after each switch, in public encounter slots, equals public_order.

        Members are revealed in team order here: the public view cannot know a hidden member's
        team position, so its rule places an unrevealed member at its public slot's position."""
        import json,os
        root=os.environ['POKEZERO_SHOWDOWN_ROOT']
        names=['Magikarp','Feebas','Goldeen','Horsea','Shellder','Staryu']
        code="""const {Battle}=require(process.argv[1]+'/dist/sim/battle.js');
function mon(species){return {name:species,species,ability:'Swift Swim',item:'',moves:['splash'],nature:'Hardy',level:80,gender:'M',
 evs:{hp:84,atk:84,def:84,spa:84,spd:84,spe:84},ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31}};}
const b=new Battle({formatid:'gen3customgame',seed:[5,6,7,8]});
b.setPlayer('p1',{name:'A',team:[mon('Smeargle')]});b.setPlayer('p2',{name:'B',team:JSON.parse(process.argv[2]).map(mon)});
const orders=[b.sides[1].pokemon.map(p=>p.name)];
for(const c of JSON.parse(process.argv[3])){b.makeChoices('move 1',c);orders.push(b.sides[1].pokemon.map(p=>p.name));}
console.log(JSON.stringify(orders));"""
        choices=['switch 2','switch 3','switch 2','switch 4','switch 3','switch 5']
        orders=json.loads(subprocess.check_output(['node','-e',code,root,json.dumps(names),json.dumps(choices)],text=True))
        state=self.state();state=state._replace(species=state.species.at[:,1].set(j.asarray([3,1,2,3,1,2],j.int32)))
        c=h.initialize(state);yes=j.ones(2,bool);public=[names[0]]
        for k,order in enumerate(orders):
            if k and order[0]!=orders[k-1][0]:
                if order[0] not in public:public.append(order[0])
                state=state._replace(active=state.active.at[:,1].set(names.index(order[0])));h.install(c,state,1,yes)
            hidden=iter(range(len(public),6))
            expected=[public.index(n) if n in public else next(hidden) for n in order]
            self.assertEqual(c.trackers.public_order[0,1].tolist(),expected,(k,order))
        self.assertEqual(orders[3][:3],['Magikarp','Goldeen','Feebas'])            # 0->1->2->0 gives [0,2,1,...]

    def test_departed_color_change_shows_b2_species_types(self):
        from gpu_engine import public_types as pt, public_mon_state as m
        state=self.state();yes=j.ones(2,bool);owners=j.zeros(2,j.int32)
        types=self.tables.belief.species_types[state.species];state=state._replace(types=types,base_types=types)
        c=h.initialize(state,self.tables.belief)
        gardevoir=state._replace(active=state.active.at[:,1].set(1));h.install(c,gardevoir,1,yes)
        water=self.metadata['vocabulary']['types'].index('water')
        hit=gardevoir._replace(types=gardevoir.types.at[:,1,1].set(j.asarray([water,0],j.int32)))
        h.color_change(c,gardevoir,hit,1,yes)
        stale=c.trackers
        reverted=hit._replace(types=hit.types.at[:,1,1].set(hit.base_types[:,1,1]))
        back=reverted._replace(active=reverted.active.at[:,1].set(0))
        h.depart(c,hit,reverted,1,None,yes);h.install(c,back,1,yes);h.withdraw(c,back,1,j.ones(2,j.int32),yes)
        cols=[s.CATEGORY_TYPE_1,s.CATEGORY_TYPE_2]
        def bench_types(trackers):
            out=e.encode(back,tr.readout(trackers,owners),self.tables,observer_sides=owners,legal_action_mask=j.ones((2,9),bool))
            slot=int(trackers.physical_to_public[0,1,1]);return np.asarray(out.categorical_ids[0,7+slot,cols])
        # B2: the team writer's species categories on the bench; the seat's overlay is cleared on arrival.
        sid=self.catalog.species.species_id('gardevoir')
        base=np.asarray(self.catalog.species.device.type_categories)[sid]
        np.testing.assert_array_equal(bench_types(c.trackers),base)
        self.assertFalse(np.array_equal(bench_types(stale._replace(active_slot=c.trackers.active_slot)),base))
        tt=self.catalog.types.device;sb=self.catalog.species.device.binding
        ts,mons=pt.initialize(1,tt,species_binding=sb),m.initialize(1,species_binding=sb)
        def arrive(mons,ts,slot,name):
            mons,ok=m.arrive(mons,side=1,slot=slot,encounter=slot+1,species=self.catalog.species.species_id(name),
                source_species=0,level=80,gender=0,hp_numerator=100,hp_denominator=100,status=0);self.assertTrue(bool(ok.all()))
            ts,ok=pt.apply(ts,tt,mons,operation=pt.ARRIVE,side=1,encounter=slot+1,producer_binding=tt.binding);self.assertTrue(bool(ok.all()))
            return mons,ts
        mons,ts=arrive(mons,ts,1,'gardevoir')
        from gpu_engine.public_moves import _static_category
        ts,ok=pt.apply(ts,tt,mons,operation=pt.TYPE,side=1,encounter=2,first=_static_category(self.catalog.vocabulary,'type:Water'),
            producer_binding=tt.binding);self.assertTrue(bool(ok.all()))
        self.assertTrue(bool(pt.project(ts,tt,mons,expected_vocabulary=tt.vocabulary_binding).write_mask[0,1].any()))
        # FAINT leaves the overlay on the still-current encounter; the replacement's ARRIVE clears it.
        mons,ok=m.faint(mons,side=1,slot=1,encounter=2);self.assertTrue(bool(ok.all()))
        self.assertTrue(bool(pt.project(ts,tt,mons,expected_vocabulary=tt.vocabulary_binding).write_mask[0,1].any()))
        mons,ts=arrive(mons,ts,0,'absol')
        self.assertFalse(bool(pt.project(ts,tt,mons,expected_vocabulary=tt.vocabulary_binding).write_mask.any()))
        # The device over the same faint-then-replacement: overlay kept at the faint, reverted at the arrival.
        c=h.initialize(state,self.tables.belief);h.install(c,gardevoir,1,yes);h.color_change(c,gardevoir,hit,1,yes)
        fainted=hit._replace(hp=hit.hp.at[:,1,1].set(0));h.turn_end(c,hit,fainted,j.zeros(2,bool))
        self.assertEqual(c.trackers.shown_types[:,1,int(c.trackers.physical_to_public[0,1,1])].tolist(),[[water,0]]*2)
        gone=fainted._replace(types=fainted.types.at[:,1,1].set(fainted.base_types[:,1,1]))
        replaced=gone._replace(active=gone.active.at[:,1].set(0))
        h.depart(c,fainted,gone,1,None,yes);h.install(c,replaced,1,yes);h.withdraw(c,replaced,1,j.ones(2,j.int32),yes)
        np.testing.assert_array_equal(bench_types(c.trackers),base)

    def b2_lines(self,a,catalog,fact,*lines):
        """B2's own line consumer: public_active.project_line facts, applied by public_active.apply."""
        for text in lines:
            for op,side,value in a.project_line(text,catalog=catalog).facts:fact(op,side,value)

    def test_destiny_bond_bag_matches_b2_singlemove_expiry_and_faint(self):
        """A2's destinybond producer (hooks.destiny_bond) against B2: -singlemove adds the bag entry, the holder's
        next move line expires it (EXPIRE_ACTION), and a faint clears it (review L5)."""
        a,catalog,fact,project=self.b2_active()
        state=self.state();c=h.initialize(state,self.tables.belief);yes=j.ones(2,bool)
        fact(a.ARRIVE,0);fact(a.ARRIVE,1)
        bonded=state._replace(volatile=state.volatile.at[:,1,0].set(64))
        h.destiny_bond(c,bonded,1,yes);self.b2_lines(a,catalog,fact,'|-singlemove|p2a: Absol|Destiny Bond')
        expected=project();self.compare_active(c.trackers,expected,a)
        self.assertTrue(bool(expected.categorical[0,1,j.asarray(a.CATEGORY_COLUMNS)].any()))
        # The holder's next move attempt: the engine clears the bit at BeforeMove; B2 expires on the move line.
        h.destiny_bond(c,state,1,yes);self.b2_lines(a,catalog,fact,'|move|p2a: Absol|Splash|p2a: Absol')
        expected=project();self.compare_active(c.trackers,expected,a)
        self.assertFalse(bool(expected.categorical[0,1,j.asarray(a.CATEGORY_COLUMNS)].any()))
        # Bonded again, then KO'd: B2's FAINT empties the bag; the device's faint clear in turn_end does too.
        h.destiny_bond(c,bonded,1,yes);self.b2_lines(a,catalog,fact,'|-singlemove|p2a: Absol|Destiny Bond')
        self.compare_active(c.trackers,project(),a)
        fact(a.FAINT,1);h.turn_end(c,bonded,bonded._replace(hp=bonded.hp.at[:,1,0].set(0)),j.zeros(2,bool))
        self.compare_active(c.trackers,project(),a)

    def confused(self,state,side,value,slot=0):
        return state._replace(confusion_counter=state.confusion_counter.at[:,side,slot].set(value))

    def confusion_turn(self,c,a,fact,state,turn):
        """One TURN while seat 1 is confused: turn_end's CONFUSION_TURNS tick and B2's TURN fact."""
        h.turn_end(c,state,state,j.ones(2,bool));fact(a.TURN,-1,turn)

    def test_confusion_snap_out_ends_the_bag_entry_and_clock_like_b2(self):
        """B (design 2.5): Signal Beam's -start adds the confusion bag entry (public_move_state), each TURN ticks
        CONFUSION_TURNS, a move attempt that does not end it prints -activate (no bag change), and the BeforeMove
        snap-out's -end removes the entry and resets the clock: hooks.before_move's confusion_end, since
        public_move_state's before is the post-BeforeMove state. The old device kept both."""
        from gpu_engine.mechanics import BeforeMove
        a,catalog,fact,project=self.b2_active()
        state=self.state();c=h.initialize(state,self.tables.belief);yes=j.ones(2,bool);no=j.zeros(2,bool)
        fact(a.ARRIVE,0);fact(a.ARRIVE,1)
        two=self.confused(state,1,2)
        h.public_move_state(c,state,two,0,j.zeros(2,j.int32),yes);self.b2_lines(a,catalog,fact,'|-start|p2a: Absol|confusion')
        self.compare_active(c.trackers,project(),a)
        self.confusion_turn(c,a,fact,two,1)
        cols=j.asarray(a.NUMERIC_COLUMNS);clock=a.NUMERIC_COLUMNS[a.CORE.index('CONFUSION_TURNS')]
        expected=project();self.compare_active(c.trackers,expected,a);self.assertTrue(bool(expected.numeric[0,1,clock]))
        def attempt(before,left):
            return BeforeMove(before.status[:,1,0],before.status_counter[:,1,0],j.full(2,left,j.int32),yes,no)
        one=self.confused(state,1,1)
        h.before_move(c,two,1,attempt(two,1),yes);self.b2_lines(a,catalog,fact,'|-activate|p2a: Absol|confusion')
        self.compare_active(c.trackers,project(),a)
        self.confusion_turn(c,a,fact,one,2)
        stale=c.trackers;self.compare_active(stale,project(),a)
        h.before_move(c,one,1,attempt(one,0),yes);self.b2_lines(a,catalog,fact,'|-end|p2a: Absol|confusion')
        expected=project();self.compare_active(c.trackers,expected,a)
        self.assertFalse(bool(expected.numeric[0,1,cols].any()));self.assertFalse(bool(expected.categorical[0,1,j.asarray(a.CATEGORY_COLUMNS)].any()))
        self.assertFalse(self.compare_active(stale,expected,a,equal=False))

    def test_confusion_lum_cure_ends_the_bag_entry_and_clock_like_b2(self):
        """B: a Lum Berry's onEat removes confusion (-enditem|X|Lum Berry|[eat], -end|X|confusion), e.g. one Baton-passed
        onto a holder: item_changes' confusion_end, from the item window's before/after confusion."""
        a,catalog,fact,project=self.b2_active()
        state=self.state();c=h.initialize(state,self.tables.belief);yes=j.ones(2,bool)
        fact(a.ARRIVE,0);fact(a.ARRIVE,1)
        held=self.confused(state,1,3)._replace(item=state.item.at[:,1,0].set(1))
        h.public_move_state(c,state,held,0,j.zeros(2,j.int32),yes);self.b2_lines(a,catalog,fact,'|-start|p2a: Absol|confusion')
        self.confusion_turn(c,a,fact,held,1)
        stale=c.trackers;self.compare_active(stale,project(),a)
        cured=self.confused(held,1,0)._replace(item=held.item.at[:,1,0].set(0))
        h.item_changes(c,held,cured,yes)
        self.b2_lines(a,catalog,fact,'|-enditem|p2a: Absol|Lum Berry|[eat]','|-end|p2a: Absol|confusion')
        expected=project();self.compare_active(c.trackers,expected,a)
        self.assertFalse(self.compare_active(stale,expected,a,equal=False))

    def test_baton_passed_confusion_and_own_tempo_cure_like_b2(self):
        """B: Baton Pass carries the confusion bag entry and its clock (B2 ARRIVE with baton_pass keeps transferable
        entries and their elapsed clock; hooks.baton_copy). An Own Tempo recipient's onUpdate then prints
        -activate|X|ability: Own Tempo and -end|X|confusion: entry's confusion_end removes both."""
        from types import SimpleNamespace as NS
        for own_tempo in (False,True):
            a,catalog,fact,project=self.b2_active()
            state=self.state()._replace(ability=j.zeros((2,2,6),j.int32).at[:,1,1].set(1))
            c=h.initialize(state,self.tables.belief);yes=j.ones(2,bool)
            fact(a.ARRIVE,0);fact(a.ARRIVE,1)
            two=self.confused(state,1,2)
            h.public_move_state(c,state,two,0,j.zeros(2,j.int32),yes);self.b2_lines(a,catalog,fact,'|-start|p2a: Absol|confusion')
            self.confusion_turn(c,a,fact,two,1)
            fact(a.ARRIVE,1,slot=1,bp=True)
            passed=self.confused(two,1,2,slot=1)._replace(active=two.active.at[:,1].set(1))
            h.install(c,passed,1,yes);h.baton_copy(c,passed,1,two.active[:,1],yes);h.withdraw(c,passed,1,two.active[:,1],yes)
            carried=project();self.compare_active(c.trackers,carried,a)
            self.assertTrue(bool(carried.numeric[0,1,a.NUMERIC_COLUMNS[a.CORE.index('CONFUSION_TURNS')]]),own_tempo)
            n=len(self.metadata['vocabulary']['abilities'])
            table=NS(weather_ability=j.zeros(n,j.int32),intimidate=j.zeros(n,bool),attack_drop_immune=j.zeros(n,bool),trace=None,
                own_tempo=j.zeros(n,bool).at[1].set(own_tempo))
            after=self.confused(passed,1,0,slot=1) if own_tempo else passed
            h.entry(c,passed,after,1,table,yes)
            if own_tempo:self.b2_lines(a,catalog,fact,'|-activate|p2a: Gardevoir|ability: Own Tempo','|-end|p2a: Gardevoir|confusion')
            expected=project();self.compare_active(c.trackers,expected,a)
            self.assertEqual(bool(expected.categorical[0,1,j.asarray(a.CATEGORY_COLUMNS)].any()),not own_tempo)

    def test_knock_off_clears_the_swap_flag_like_b2_item_end(self):
        """A2's knock_off producer against B2's ITEM_END (public_active.py): a Tricked Choice Band, locked by a move line,
        then knocked off; -enditem clears item_swapped, choice_public and choice_locked (tracked since M1) (L5)."""
        a,catalog,fact,project=self.b2_active()
        state=self.state();c=h.initialize(state,self.tables.belief);yes=j.ones(2,bool)
        fact(a.ARRIVE,0);fact(a.ARRIVE,1)
        self.b2_lines(a,catalog,fact,'|-item|p2a: Absol|Choice Band|[from] move: Trick','|move|p2a: Absol|Tackle|p1a: Snorlax')
        for name in ('item_swapped','choice_public','choice_locked'):
            h.feature(c,1,state.active[:,1],name,j.ones(2,j.int32),yes)
        swapped=project();self.compare_active(c.trackers,swapped,a)
        self.assertTrue(bool(swapped.numeric[0,1,a.NUMERIC_COLUMNS[a.CORE.index('CHOICE_LOCKED')]]))
        cols=j.asarray(a.NUMERIC_COLUMNS)
        self.assertTrue(bool(swapped.numeric[0,1,cols].any()))
        item=1;before=state._replace(item=state.item.at[:,1,0].set(item))
        after=before._replace(item=before.item.at[:,1,0].set(0),item_knocked=before.item_knocked.at[:,1,0].set(True))
        h.knock_off(c,before,after,1,yes)
        self.b2_lines(a,catalog,fact,'|-enditem|p2a: Absol|Leftovers|[from] move: Knock Off|[of] p1a: Snorlax')
        cleared=project();self.compare_active(c.trackers,cleared,a)
        self.assertFalse(bool(cleared.numeric[0,1,cols].any()))

    def test_white_herb_ends_the_item_flags_like_b2_item_end(self):
        """WH's white_herb producer against B2's ITEM_END (public_active.py): a Tricked Choice Band's flags, then a bare
        -enditem (White Herb's useItem; the item name does not enter public_active) clears item_swapped, choice_public and
        choice_locked; the bag is untouched."""
        a,catalog,fact,project=self.b2_active()
        state=self.state();c=h.initialize(state,self.tables.belief);yes=j.ones(2,bool)
        fact(a.ARRIVE,0);fact(a.ARRIVE,1)
        self.b2_lines(a,catalog,fact,'|-item|p2a: Absol|Choice Band|[from] move: Trick','|move|p2a: Absol|Tackle|p1a: Snorlax')
        for name in ('item_swapped','choice_public','choice_locked'):
            h.feature(c,1,state.active[:,1],name,j.ones(2,j.int32),yes)
        self.compare_active(c.trackers,project(),a)
        before=state._replace(item=state.item.at[:,1,0].set(2),boosts=state.boosts.at[:,1,0,0].set(-1))
        after=before._replace(item=before.item.at[:,1,0].set(0),boosts=before.boosts.at[:,1,0,0].set(0))
        h.white_herb(c,before,after,1,yes)
        self.b2_lines(a,catalog,fact,'|-enditem|p2a: Absol|White Herb','|-clearnegativeboost|p2a: Absol|[silent]')
        cleared=project();self.compare_active(c.trackers,cleared,a)
        self.assertFalse(bool(cleared.numeric[0,1,j.asarray(a.NUMERIC_COLUMNS)].any()))

    def test_trick_received_white_herb_flags_follow_b2(self):
        """WH review finding 1 (TrickWhiteHerbBoundaryParityTests has the ledger): B2's public_active fold of the four lines
        Showdown prints when a Trick hands a White Herb to a receiver with a negative stage, here Snorlax (Band, -2 Atk)
        Tricking Absol: the target's -item (ITEM_SET, choice_public for the Band), the receiver's -enditem (ITEM_END) and
        -clearnegativeboost, then the receiver's own -item (ITEM_SET). The device's white_herb then trick, in that order."""
        a,catalog,fact,project=self.b2_active()
        state=self.state();yes=j.ones(2,bool);band=j.asarray([False,False,True])
        state=state._replace(item=state.item.at[:,0,0].set(2).at[:,1,0].set(1))
        c=h.initialize(state,self.tables.belief);fact(a.ARRIVE,0);fact(a.ARRIVE,1)
        dropped=state._replace(boosts=state.boosts.at[:,0,0,0].set(-2))
        swapped=dropped._replace(item=dropped.item.at[:,0,0].set(1).at[:,1,0].set(2))
        used=swapped._replace(item=swapped.item.at[:,0,0].set(0),boosts=swapped.boosts.at[:,0,0,0].set(0))
        h.white_herb(c,swapped,used,0,yes);h.trick(c,dropped,swapped,0,yes,band,herb_used=j.asarray([[True,False]]*2))
        self.b2_lines(a,catalog,fact,'|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol','|-item|p2a: Absol|Choice Band|[from] move: Trick',
            '|-enditem|p1a: Snorlax|Leftovers','|-clearnegativeboost|p1a: Snorlax|[silent]','|-item|p1a: Snorlax|Leftovers|[from] move: Trick')
        got=project();self.compare_active(c.trackers,got,a)
        swapped_col=a.NUMERIC_COLUMNS[a.CORE.index('ITEM_SWAPPED')]
        self.assertEqual((bool(got.numeric[0,0,swapped_col]),bool(got.numeric[0,1,swapped_col])),(True,True))

    def test_trick_flags_follow_b2_item_set_item_end_and_choice_move(self):
        """M1's trick hook and the move hook's CHOICE_MOVE against B2's public_active fold of the same lines: -item is
        ITEM_SET (item_swapped; choice_public for a Choice item; choice_locked reset), -enditem [silent] is ITEM_END, and
        the holder's own move line locks a public Choice Band (column 40). The Tricker's move line precedes the -item
        lines, so the Tricker never locks on its Trick; a called child ([from]) or a preparation never locks."""
        a,catalog,fact,project=self.b2_active()
        state=self.state();yes=j.ones(2,bool);band=j.asarray([False,False,True])
        state=state._replace(item=state.item.at[:,0,0].set(2).at[:,1,0].set(1))
        c=h.initialize(state,self.tables.belief);fact(a.ARRIVE,0);fact(a.ARRIVE,1)
        locked=a.NUMERIC_COLUMNS[a.CORE.index('CHOICE_LOCKED')]
        swapped_col=a.NUMERIC_COLUMNS[a.CORE.index('ITEM_SWAPPED')]
        def move(st,side,line,**kw):
            h.move(c,st,side,j.ones(2,j.int32),yes,pp_use=True,**kw);self.b2_lines(a,catalog,fact,line)
        # Snorlax (Band) Tricks Absol (Leftovers): both -item lines; Absol's own move then locks it publicly.
        move(state,0,'|move|p1a: Snorlax|Trick|p2a: Absol')
        after=state._replace(item=state.item.at[:,0,0].set(1).at[:,1,0].set(2))
        h.trick(c,state,after,0,yes,band)
        self.b2_lines(a,catalog,fact,'|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol','|-item|p2a: Absol|Choice Band|[from] move: Trick',
            '|-item|p1a: Snorlax|Leftovers|[from] move: Trick')
        got=project();self.compare_active(c.trackers,got,a)
        self.assertEqual((bool(got.numeric[0,0,swapped_col]),bool(got.numeric[0,1,swapped_col]),bool(got.numeric[0,1,locked])),(True,True,False))
        h.move(c,after,1,j.ones(2,j.int32),yes,pp_use=False,count_turn=False)          # a called child ([from]): no lock
        self.b2_lines(a,catalog,fact,'|move|p2a: Absol|Tackle|p1a: Snorlax|[from]move: Sleep Talk')
        self.compare_active(c.trackers,project(),a);self.assertFalse(bool(project().numeric[0,1,locked]))
        move(after,1,'|move|p2a: Absol|Tackle|p1a: Snorlax')
        got=project();self.compare_active(c.trackers,got,a);self.assertTrue(bool(got.numeric[0,1,locked]))
        h.turn_end(c,after,after,yes);fact(a.TURN,-1,1);self.compare_active(c.trackers,project(),a)
        # Snorlax Tricks again: Absol gives the Band back and gets nothing (-enditem [silent]: ITEM_END).
        move(after,0,'|move|p1a: Snorlax|Trick|p2a: Absol')
        back=after._replace(item=after.item.at[:,0,0].set(2).at[:,1,0].set(1))
        h.trick(c,after,back,0,yes,band)
        self.b2_lines(a,catalog,fact,'|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol','|-item|p2a: Absol|Leftovers|[from] move: Trick',
            '|-item|p1a: Snorlax|Choice Band|[from] move: Trick')
        got=project();self.compare_active(c.trackers,got,a);self.assertFalse(bool(got.numeric[0,1,locked]))
        move(back,0,'|move|p1a: Snorlax|Surf|p2a: Absol')
        got=project();self.compare_active(c.trackers,got,a);self.assertTrue(bool(got.numeric[0,0,locked]))
        # A one-sided Trick: Snorlax's Band for nothing: Snorlax's -enditem is ITEM_END, Absol's -item ITEM_SET.
        empty=back._replace(item=back.item.at[:,1,0].set(0))
        h.trick(c,empty,empty._replace(item=empty.item.at[:,0,0].set(0).at[:,1,0].set(2)),0,yes,band)
        self.b2_lines(a,catalog,fact,'|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol','|-item|p2a: Absol|Choice Band|[from] move: Trick',
            '|-enditem|p1a: Snorlax|Choice Band|[silent]|[from] move: Trick')
        got=project();self.compare_active(c.trackers,got,a)
        self.assertEqual((bool(got.numeric[0,0,swapped_col]),bool(got.numeric[0,0,locked])),(False,False))
        # Band <-> Band (B2's identical-swap quirk): the state is unchanged but both -item lines are folded.
        same=back._replace(item=back.item.at[:,1,0].set(2))
        h.trick(c,same,same,0,yes,band)
        self.b2_lines(a,catalog,fact,'|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol','|-item|p2a: Absol|Choice Band|[from] move: Trick',
            '|-item|p1a: Snorlax|Choice Band|[from] move: Trick')
        got=project();self.compare_active(c.trackers,got,a)
        self.assertEqual((bool(got.numeric[0,0,swapped_col]),bool(got.numeric[0,1,swapped_col])),(True,True))
        # Arrival resets the seat's flags (ARRIVE), not the per-mon ledger.
        fact(a.ARRIVE,1,slot=1);moved=same._replace(active=same.active.at[:,1].set(1))
        h.install(c,moved,1,yes);h.withdraw(c,moved,1,same.active[:,1],yes)
        self.compare_active(c.trackers,project(),a)
        self.assertEqual(int(c.trackers.mon_features['item_mutated'][0,1,int(c.trackers.physical_to_public[0,1,0])]),1)

    def test_mustrecharge_bag_and_its_stale_bit_follow_b2(self):
        """M1: -mustrecharge (B2 RECHARGE) when the counter starts, cant|X|recharge (CLEAR_RECHARGE) at the recharge turn's
        hook; a recharge cancelled by a faint expires silently and both keep the bit (B2's stale bit, kept on purpose),
        through the holder's later move lines; a faint or an arrival clears it."""
        a,catalog,fact,project=self.b2_active()
        state=self.state();c=h.initialize(state,self.tables.belief);yes=j.ones(2,bool)
        fact(a.ARRIVE,0);fact(a.ARRIVE,1)
        name=a.CATEGORY_COLUMNS;bag=lambda got:bool(got.categorical[0,0,j.asarray(name)].any())
        hit=state._replace(recharge_counter=state.recharge_counter.at[:,0,0].set(2))
        h.public_move_state(c,state,hit,0,j.ones(2,j.int32),yes);self.b2_lines(a,catalog,fact,'|-mustrecharge|p1a: Snorlax')
        got=project();self.compare_active(c.trackers,got,a);self.assertTrue(bag(got))
        h.turn_end(c,hit,hit,yes);fact(a.TURN,-1,1);self.compare_active(c.trackers,project(),a)
        h.recharge_turn(c,state,0,yes);self.b2_lines(a,catalog,fact,'|cant|p1a: Snorlax|recharge')
        got=project();self.compare_active(c.trackers,got,a);self.assertFalse(bag(got))
        h.public_move_state(c,state,hit,0,j.ones(2,j.int32),yes);self.b2_lines(a,catalog,fact,'|-mustrecharge|p1a: Snorlax')
        h.turn_end(c,hit,hit,yes);fact(a.TURN,-1,2)
        h.turn_end(c,hit,state,yes);fact(a.TURN,-1,3)                              # silent expiry: no line, no hook
        got=project();self.compare_active(c.trackers,got,a);self.assertTrue(bag(got))
        h.destiny_bond(c,state,0,yes);h.move(c,state,0,j.ones(2,j.int32),yes,pp_use=True)
        self.b2_lines(a,catalog,fact,'|move|p1a: Snorlax|Surf|p2a: Absol')
        got=project();self.compare_active(c.trackers,got,a);self.assertTrue(bag(got))   # EXPIRE_ACTION leaves it
        fact(a.FAINT,0);h.turn_end(c,state,state._replace(hp=state.hp.at[:,0,0].set(0)),j.zeros(2,bool))
        got=project();self.compare_active(c.trackers,got,a);self.assertFalse(bag(got))

    def test_recharge_request_matches_the_original_action_writer(self):
        """M1: the device's recharge collapse against the original writer (showdown._encode_action_tokens) on the request
        Showdown sends: moves [{move: 'Recharge', id: 'recharge'}], trapped."""
        from typing import NamedTuple
        from types import SimpleNamespace
        state,public,owners,legal=self.fixture.facts(b=1)
        Recharging=NamedTuple('Recharging',[(n,object) for n in state._fields]+[('recharge_counter',object)])
        s=Recharging(*state,recharge_counter=j.zeros((1,2,6),j.int32).at[0,0,1].set(1))
        legal=j.asarray([[True]+[False]*8])
        out=e.encode(s,public,self.tables,observer_sides=owners,legal_action_mask=legal)
        raw={'side':{'id':'p1','pokemon':[dict(ident='p1: '+name,details=name+', L80, M',condition='120/240',active=i==0,
            ability='Immunity',item='Leftovers',stats={'atk':151,'def':160,'spa':170,'spd':180,'spe':190})
            for i,name in enumerate(('Snorlax','Gardevoir'))]},
            'active':[{'moves':[{'move':'Recharge','id':'recharge'}],'trapped':True}]}
        team=s_._self_team_from_request(raw,'p1')
        source=SimpleNamespace(request=raw,self_team=team,self_active=team[0],legal_action_mask=s_._legal_action_mask(raw))
        self.assertEqual(list(source.legal_action_mask),[True]+[False]*8)
        nums=[[0.]*s_.V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT for _ in range(23)];cats=[['']*60 for _ in range(23)]
        s_._encode_action_tokens(cats,nums,source,dex=self.catalog.dex)
        np.testing.assert_allclose(out.numeric_features[0,13:22],s_._project_v4_numeric_rows(nums)[13:22],rtol=0,atol=1e-6)
        expected=[[self.catalog.vocabulary.encode(label) for label in row[:41]] for row in cats[13:22]]
        np.testing.assert_array_equal(out.categorical_ids[0,13:22],expected)
        self.assertEqual(int(out.categorical_ids[0,13,0]),self.catalog.vocabulary.encode('move:recharge'))
        self.assertEqual(out.move_targets[0].tolist(),[0,1,2,3])
        # Without the leaf (an older struct) nothing collapses: the two ordinary rows come back.
        plain=e.encode(state,public,self.tables,observer_sides=owners,legal_action_mask=legal)
        self.assertEqual(int(plain.numeric_features[0,14,e._index('PRESENT')]),1)

    def test_effective_gen3_failed_focus_retains_move_before_cant_order(self):
        import json,os
        root=os.environ['POKEZERO_SHOWDOWN_ROOT']
        code="""const {Dex}=require(process.argv[1]+'/dist/sim/dex.js');const d=Dex.mod('gen3'),m=d.moves.get('focuspunch'),a=d.data.Scripts.actions;const events=[];const result=m.onTry.call({attrLastMove:x=>events.push(['attr',x]),add:(...x)=>events.push(['add',...x.map(String)])},{toString:()=> 'p1a: Snorlax',volatiles:{focuspunch:{lostFocus:true}}});const use=a.useMoveInner.toString(),hit=a.tryMoveHit.toString();console.log(JSON.stringify({beforeMove:typeof m.beforeMoveCallback,resultIsNull:result===null,events,move:use.indexOf('this.battle.addMove('),hit:use.indexOf('this.tryMoveHit('),onTry:hit.indexOf('singleEvent("Try",')}));"""
        actual=json.loads(subprocess.check_output(['node','-e',code,root],text=True))
        self.assertEqual(actual['beforeMove'],'undefined');self.assertTrue(actual['resultIsNull'])
        self.assertEqual(actual['events'],[['attr','[still]'],['add','cant','p1a: Snorlax','Focus Punch','Focus Punch']])
        self.assertGreaterEqual(actual['move'],0);self.assertGreater(actual['hit'],actual['move']);self.assertGreaterEqual(actual['onTry'],0)
        # The ordinary MOVE has already opened this original-source window;
        # the following CANT does not erase its header/first-stay effects.
        refstate=self.reference()
        for line in ('|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100','|turn|1',
                     '|move|p1a: Snorlax|Surf|p2a: Absol|[still]','|cant|p1a: Snorlax|Focus Punch|Focus Punch'):
            refstate=self.line(refstate,line)
        self.assertEqual(int(refstate.mon[0,0,0,1]),1);self.assertEqual(int(refstate.opportunities[0,0]),1)


class KnockOffLedgerBoundaryParityTests(unittest.TestCase):
    """A2's knock_off producer against B2's own fold of the Showdown lines (review L5): the item ledger's shown
    and removed item (public_items REMOVE) and the removal credit columns 119/120 (public_damage.project_credits)."""
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_damage_context as fixture
        fixture.PublicDamageContextTests.setUpClass()
        cls.f=fixture.PublicDamageContextTests()

    def b2(self,lines):
        result,_=self.f.run_lines(lines)
        d=self.f.d;tables=self.f.damage_tables
        credits={}
        for observer in (0,1):
            patch=d.project_credits(result.context.damage,tables,result.mon_state,observer_side=observer,
                expected_vocabulary=tables.vocabulary_binding)
            self.assertEqual(patch.admitted.tolist(),[True])
            credits[observer]=[float(patch.numeric[0,col]) for col in d.FIELD_COLUMNS[2:]]
        items=result.context.items
        return dict(credits=credits,shown=self.f.item_catalog.names[int(items.revealed[0,1,0])],
            removed=bool(items.removed[0,1,0]),mutated=bool(items.mutated[0,1,0]),
            original=self.f.item_catalog.names[int(items.original[0,1,0])],original_id=int(items.original[0,1,0]))

    def device(self,knocked):
        from gpu_engine import engine_device_trackers as dtr,engine_device_features as df
        s=empty_state(1)
        s=s._replace(species=s.species.at[0,0,0].set(1).at[0,1,0].set(2),level=j.full_like(s.level,80),
            hp=j.full_like(s.hp,100),max_hp=j.full_like(s.max_hp,100),item=s.item.at[0,1,0].set(7),done=j.zeros(1,bool))
        c=h.initialize(s,None);yes=j.ones(1,bool)
        if knocked:
            after=s._replace(item=s.item.at[0,1,0].set(0),item_knocked=s.item_knocked.at[0,1,0].set(True))
            h.knock_off(c,s,after,1,yes)
        credits={}
        for observer in (0,1):
            view=dtr.readout(c.trackers,j.full(1,observer,j.int32))
            numeric,bad=df.project_field(j.zeros((1,132),j.float32),view.field_features)
            self.assertFalse(bool(bad.any()))
            credits[observer]=[float(numeric[0,s_.v4_numeric_index(getattr(s_,'NUMERIC_'+n))])
                for n in ('SELF_ITEMS_REMOVED_CREDIT','OPP_ITEMS_REMOVED_CREDIT')]
        slot=int(c.trackers.physical_to_public[0,1,0])
        return dict(credits=credits,shown=int(c.trackers.shown_item[0,1,0]),removed=bool(c.trackers.item_removed[0,1,0]),
            mutated=bool(c.trackers.mon_features['item_mutated'][0,1,slot]),original=int(c.trackers.mon_features['item_original'][0,1,slot]))

    def test_knock_off_ledger_and_removal_credit_match_b2(self):
        prefix=[*self.f.prefix,'|turn|1','|move|p1a: Snorlax|Body Slam|p2a: Absol','|-damage|p2a: Absol|80/100']
        knocked=self.b2(prefix+['|-enditem|p2a: Absol|Leftovers|[from] move: Knock Off|[of] p1a: Snorlax'])
        plain=self.b2(prefix)
        self.assertEqual((knocked['shown'],knocked['removed'],knocked['mutated']),('leftovers',True,True))
        self.assertEqual((plain['removed'],plain['mutated']),(False,False))
        device,quiet=self.device(True),self.device(False)
        # The removed item is shown (the device's id for the knocked Leftovers) and marked removed, as in B2.
        self.assertEqual((device['shown'],device['removed']),(7,knocked['removed']))
        self.assertEqual((quiet['shown'],quiet['removed']),(0,plain['removed']))
        # Columns 119/120 for both observers: the victim's side (p2) is credited one removal of six.
        for observer in (0,1):
            np.testing.assert_allclose(device['credits'][observer],knocked['credits'][observer],atol=1e-7)
            np.testing.assert_allclose(quiet['credits'][observer],plain['credits'][observer],atol=1e-7)
        np.testing.assert_allclose(knocked['credits'][1],[1/6,0.],atol=1e-7);np.testing.assert_allclose(knocked['credits'][0],[0.,1/6],atol=1e-7)

    def test_white_herb_consumption_matches_b2_consume_and_clear_negative(self):
        """WH's white_herb producer against B2's own fold of -enditem|X|White Herb and -clearnegativeboost|X|[silent]:
        public_items CONSUME (the item revealed and removed, no mutation, no original: the ledger stays) and public_mon_state
        CLEAR_NEGATIVE on the active's stages (only the negative ones). B2's projection of a bare -enditem does not depend
        on the item's name (public_items.project_line: CONSUME), so the fixture's catalog item stands in for White Herb."""
        prefix=[*self.f.prefix,'|turn|1','|-boost|p2a: Absol|spa|1','|-unboost|p2a: Absol|atk|1','|-unboost|p2a: Absol|def|2']
        result,_=self.f.run_lines(prefix+['|-enditem|p2a: Absol|Leftovers','|-clearnegativeboost|p2a: Absol|[silent]'])
        items=result.context.items
        b2=dict(shown=self.f.item_catalog.names[int(items.revealed[0,1,0])],removed=bool(items.removed[0,1,0]),
            mutated=bool(items.mutated[0,1,0]),original=int(items.original[0,1,0]),boosts=np.asarray(result.mon_state.boosts[0,1]).tolist())
        self.assertEqual(b2,dict(shown='leftovers',removed=True,mutated=False,original=0,boosts=[0,0,1,0,0,0,0]))
        s=empty_state(1)
        s=s._replace(species=s.species.at[0,0,0].set(1).at[0,1,0].set(2),level=j.full_like(s.level,80),hp=j.full_like(s.hp,100),
            max_hp=j.full_like(s.max_hp,100),item=s.item.at[0,1,0].set(7),done=j.zeros(1,bool))
        c=h.initialize(s,None);yes=j.ones(1,bool)
        dropped=s._replace(boosts=s.boosts.at[0,1,0].set(j.asarray([-1,-2,1,0,0,0,0])))
        h.observe(c,dropped,yes)
        used=dropped._replace(item=dropped.item.at[0,1,0].set(0),boosts=dropped.boosts.at[0,1,0].set(j.asarray([0,0,1,0,0,0,0])))
        h.white_herb(c,dropped,used,1,yes)
        slot=int(c.trackers.physical_to_public[0,1,0])
        device=dict(shown=int(c.trackers.shown_item[0,1,slot]),removed=bool(c.trackers.item_removed[0,1,slot]),
            mutated=bool(c.trackers.mon_features['item_mutated'][0,1,slot]),original=int(c.trackers.mon_features['item_original'][0,1,slot]),
            boosts=np.asarray(c.trackers.boosts[0,1,slot]).tolist())
        self.assertEqual(device,dict(shown=7,removed=b2['removed'],mutated=b2['mutated'],original=b2['original'],boosts=b2['boosts']))

    def test_knock_off_item_ledger_matches_b2_mutated_and_original(self):
        """M1 review: the device's per-mon ledger written by knock_off (item_mutated, item_original) against B2's
        public_items fold of the same -enditem line (items.mutated, items.original: REMOVE's own removal credits the
        removed item as the original), and both untouched without the line."""
        prefix=[*self.f.prefix,'|turn|1','|move|p1a: Snorlax|Body Slam|p2a: Absol','|-damage|p2a: Absol|80/100']
        knocked=self.b2(prefix+['|-enditem|p2a: Absol|Leftovers|[from] move: Knock Off|[of] p1a: Snorlax'])
        plain=self.b2(prefix)
        device,quiet=self.device(True),self.device(False)
        self.assertEqual((knocked['mutated'],knocked['original']),(True,'leftovers'))
        self.assertEqual((device['mutated'],device['original']),(True,7))         # the knocked Leftovers' device id
        self.assertEqual(device['original'],device['shown']);self.assertEqual(knocked['original'],knocked['shown'])
        self.assertEqual((plain['mutated'],plain['original_id']),(False,0))
        self.assertEqual((quiet['mutated'],quiet['original']),(False,0))



class TrickWhiteHerbBoundaryParityTests(unittest.TestCase):
    """WH review finding 1: a White Herb received by Trick is used inside Trick's onHit (setItem's Start, sim/pokemon.ts:
    1868-1889; data/moves.ts:19904-19915; data/items.ts:7674-7686), before the receiver's own -item line. Showdown f76228a1,
    Deoxys-Attack (White Herb) Charming Kecleon, then Kecleon (Choice Band) Tricking it, prints these four lines:
        -item|p2a: Deoxys|Choice Band|[from] move: Trick
        -enditem|p1a: Kecleon|White Herb
        -clearnegativeboost|p1a: Kecleon|[silent]
        -item|p1a: Kecleon|White Herb|[from] move: Trick
    The transition's device order (white_herb on the swapped state, then trick with that pre-herb state, so its -item writes
    land last) against B2's fold of the same four lines, here as Snorlax (Choice Band, -2 Atk) Tricking Absol: the item
    ledger (revealed, removed, mutated, original) and the receiver's stages. B2 reads a bare -enditem alike for any item
    (public_items CONSUME), so the fixture's Leftovers stands in for White Herb. The -clearnegativeboost line ends B2's
    Trick pairing (public_items.begin_event), so the target's original, credited from the receiver's later -item line, stays
    unknown: the hook's herb_used. The old order (trick, then white_herb at AfterMove) leaves the herb removed where B2
    does not, and credits the target's original."""
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_fidelity_a2 as a2
        a2.KnockOffNarrowingB2Tests.setUpClass()
        cls.helper=a2.KnockOffNarrowingB2Tests();cls.f=cls.helper.f
        # The fixture's frozen public move namespace has neither Charm nor Trick: their effect lines stand alone (as M1's
        # Trick ledger test does).
        cls.lines=[*cls.f.prefix,'|turn|1','|-unboost|p1a: Snorlax|atk|2','|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol',
            '|-item|p2a: Absol|Choice Band|[from] move: Trick','|-enditem|p1a: Snorlax|Leftovers',
            '|-clearnegativeboost|p1a: Snorlax|[silent]','|-item|p1a: Snorlax|Leftovers|[from] move: Trick']

    def b2(self):
        result,_=self.f.run_lines(self.lines)
        names=self.f.item_catalog.names;items=result.context.items
        return {side:dict(revealed=names[int(items.revealed[0,side,0])],original=names[int(items.original[0,side,0])],
            removed=bool(items.removed[0,side,0]),mutated=bool(items.mutated[0,side,0]),
            boosts=np.asarray(result.mon_state.boosts[0,side]).tolist()) for side in (0,1)}

    def device(self,herb_first):
        names={0:'',1:'leftovers',2:'choiceband'}
        s0=empty_state(1)
        s0=s0._replace(species=s0.species.at[0,0,0].set(1).at[0,1,0].set(2),level=j.full_like(s0.level,80),
            hp=j.full_like(s0.hp,100),max_hp=j.full_like(s0.max_hp,100),item=s0.item.at[0,0,0].set(2).at[0,1,0].set(1),
            done=j.zeros(1,bool))
        c=h.initialize(s0,self.helper.tables());yes=j.ones(1,bool);band=j.asarray([False,False,True])
        dropped=s0._replace(boosts=s0.boosts.at[0,0,0,0].set(-2));h.observe(c,dropped,yes)
        swapped=dropped._replace(item=dropped.item.at[0,0,0].set(1).at[0,1,0].set(2))
        used=swapped._replace(item=swapped.item.at[0,0,0].set(0),boosts=swapped.boosts.at[0,0,0,0].set(0))
        if herb_first:h.white_herb(c,swapped,used,0,yes);h.trick(c,dropped,swapped,0,yes,band,herb_used=j.asarray([[True,False]]))
        else:h.trick(c,dropped,swapped,0,yes,band);h.white_herb(c,swapped,used,0,yes)
        out={};f_=c.trackers.mon_features
        for side in (0,1):
            slot=int(c.trackers.physical_to_public[0,side,0])
            out[side]=dict(revealed=names[int(c.trackers.shown_item[0,side,slot])],original=names[int(f_['item_original'][0,side,slot])],
                removed=bool(c.trackers.item_removed[0,side,slot]),mutated=bool(f_['item_mutated'][0,side,slot]),
                boosts=np.asarray(c.trackers.boosts[0,side,slot]).tolist())
        return out

    def test_trick_received_white_herb_four_lines_match_b2(self):
        b2=self.b2()
        self.assertEqual(b2[0],dict(revealed='leftovers',original='choiceband',removed=False,mutated=True,boosts=[0]*7))
        self.assertEqual(b2[1],dict(revealed='choiceband',original='',removed=False,mutated=True,boosts=[0]*7))
        self.assertEqual(self.device(True),b2)
        old=self.device(False)
        self.assertNotEqual(old[0]['removed'],b2[0]['removed'])        # the old order: the herb shown removed after the -item
        self.assertNotEqual(old[1]['original'],b2[1]['original'])      # and the target's original credited

if __name__=='__main__':unittest.main()
