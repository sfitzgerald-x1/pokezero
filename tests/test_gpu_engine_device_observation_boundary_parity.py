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
        current=(root/'gpu_engine/transition.py').read_text()
        new="device_public.switch_decision(_device_public,work,side,do_switch & ~replacement,baton=_device_baton)"
        old="device_public.feature(_device_public,side,old_active,'switched_before_attack',jnp.ones_like(old_active),do_switch & ~replacement,add=True)"
        self.assertEqual(current.count(new),1);self.assertEqual(current.replace(new,old),original)
        for name in ('state.py','mechanics.py','entry.py','damage_moves.py','secondary.py'):
            self.assertEqual((root/'gpu_engine'/name).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+name],cwd=root))

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
 sand:run([mon('Tyranitar','Sand Stream',['splash'])],[mon('Tyranitar','Sand Stream',['splash'])],[])}));"""
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
        mons,ts=arrive(mons,ts,0,'absol')
        self.assertFalse(bool(pt.project(ts,tt,mons,expected_vocabulary=tt.vocabulary_binding).write_mask.any()))

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


if __name__=='__main__':unittest.main()
