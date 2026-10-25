"""Native observation parity versus the pre-existing P2 path, not strength."""
import json
import os
import unittest
from unittest.mock import patch

from tests import test_leaf_encoder as fixture
from tests import test_one_ply_native_leaf_routing as routing
from pokezero.shallow_exact_native import NativeExactBackend


@unittest.skipUnless(fixture._wheel_has('ExactSession'),'requires rebuilt native session')
class NativeSessionParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        routing.NativeOnePlyLeafRoutingTests.setUpClass()
        cls.roots=routing.NativeOnePlyLeafRoutingTests.roots
        cls.native=fixture.pokezero_search

    def session(self,root,max_branches=200000,max_cells=81):
        return self.native.ExactSession(root.state_str,root.fold,
            json.dumps({**root.party_species,'turn':root.turn}),root.turn,
            max_branches,max_cells,root.encoder)

    def test_complete_chance_fans_all_five_arrays_and_pending_kos_in_both_frames(self):
        with patch.dict(os.environ,POKEZERO_ENUMERATE_ROLLS='1'):
            for root in self.roots.values():
                session=self.session(root)
                self.assertIsNone(session.last_enumeration_timing())
                old=NativeExactBackend(self.native,None)
                self.assertEqual(session.root_actions(),tuple(list(s) for s in old.actions(root,root=True)))
                self.assertIsNone(session.root_terminal())
                for row,col in [('earthquake','fireblast'),
                    (next(a for a in old.actions(root,root=True)[0] if a.startswith('switch ')),'fireblast')]:
                    display=(row,col)
                    previous=list(old.outcomes(root,*display))
                    cell,report,subcases=session.enumerate_cell(row.removeprefix('switch '),col)
                    timing=json.loads(session.last_enumeration_timing())
                    self.assertEqual((timing['status'],timing['phase']),('COMPLETE','post_generation'))
                    self.assertEqual((timing['cell_index'],timing['generated_branches']),(cell,len(report)))
                    self.assertGreater(timing['generation_seconds'],0)
                    self.assertGreaterEqual(timing['post_generation_seconds'],0)
                    self.assertLessEqual(timing['generation_seconds']+timing['post_generation_seconds'],
                        timing['native_body_seconds'])
                    self.assertEqual([p for p,_ in report],[b.probability for b in previous])
                    self.assertEqual([v for _,v in report],[old.terminal_value(b.state) for b in previous])
                    addresses=[(cell,i) for i,(_,v) in enumerate(report) if v is None]
                    expected=[root.encoder.encode_leaf(b.state.state_str,b.state.fold,b.state.turn,
                        list(b.state.lines) or None,False)
                        for b in previous if old.terminal_value(b.state) is None]
                    for workers in (1,8):
                        packed=session.encode_packed(addresses,workers)
                        self.assertEqual(packed['batch_rows'],len(expected))
                        for name in expected[0]:
                            self.assertEqual(packed[name],b''.join(r[name] for r in expected),
                                (root.self_side,display,workers,name))
                    for i in (0,len(previous)-1):
                        witness=json.loads(session.reconstruct(cell,i))
                        self.assertEqual(witness['post_state'],previous[i].state.state_str)
                        self.assertEqual(witness['instruction_list'],list(previous[i].state.instruction_list))
                        self.assertEqual(witness['lines'],list(previous[i].state.lines))
                        self.assertEqual(witness['turn'],previous[i].state.turn)

    def test_invalid_addresses_and_caps_refuse_without_mutating_the_root(self):
        root=next(iter(self.roots.values()));session=self.session(root,max_cells=1)
        actions=session.root_actions()
        cell,branches,_=session.enumerate_cell('earthquake','fireblast')
        with self.assertRaises(ValueError):session.enumerate_cell('earthquake','fireblast')
        timing=json.loads(session.last_enumeration_timing())
        self.assertEqual((timing['status'],timing['phase']),('FAILED','validation'))
        self.assertEqual((timing['generation_seconds'],timing['post_generation_seconds']),(0,0))
        for addresses,workers in (([],1),([(0,0)],0),([(999,0)],1),([(0,len(branches))],1)):
            with self.assertRaises(ValueError):session.encode_packed(addresses,workers)
        with self.assertRaises(ValueError):session.reconstruct(999,0)
        small=self.session(root,max_branches=1)
        with self.assertRaises(ValueError):small.enumerate_cell('earthquake','fireblast')
        timing=json.loads(small.last_enumeration_timing())
        self.assertEqual((timing['status'],timing['phase']),('FAILED','post_generation'))
        self.assertGreater(timing['generated_branches'],1)
        self.assertGreater(timing['generation_seconds'],0)
        self.assertEqual(session.root_actions(),actions)
        self.assertIsNone(session.root_terminal())
        # An invalid row must not poison a subsequent valid encode.
        packed=session.encode_packed([(cell,0),(cell,0)],1)
        self.assertEqual(packed['batch_rows'],2)
        self.assertEqual(json.loads(session.reconstruct(cell,0))['engine_state'],root.state_str)

    def test_short_party_terminals_draws_and_passive_pair_match_existing_p2(self):
        from dataclasses import replace
        from pokezero.poke_engine_adapter import (BattleSpec,MoveSpec,PokemonSpec,
            SideSpec,build_poke_engine_state)
        def mon(name,hp,move,speed):
            return PokemonSpec(name,100,('normal',),hp,100,200,100,100,100,speed,
                (MoveSpec(move,8),),ability='none',item='none')
        template=next(iter(self.roots.values()))
        victim=mon('blissey',1,'splash',100)
        for winner in (0,1):
            sides=[SideSpec((mon('snorlax',100,'seismictoss',200),)),SideSpec((victim,))]
            if winner:sides.reverse()
            state=build_poke_engine_state(BattleSpec(*sides)).to_string()
            root=replace(template,state_str=state,
                party_species={'p1':tuple(p.id for p in sides[0].pokemon),
                               'p2':tuple(p.id for p in sides[1].pokemon)},
                fold=self.native.FoldState.initial('p1'),turn=1)
            # Encoder is deliberately never invoked: terminal leaves bypass it.
            old=NativeExactBackend(self.native,None)
            actions=('splash','seismictoss') if winner else ('seismictoss','splash')
            previous=list(old.outcomes(root,*actions))
            session=self.session(root)
            cell,branches,_=session.enumerate_cell(*actions)
            self.assertEqual(branches,[(p.probability,old.terminal_value(p.state)) for p in previous])
            self.assertTrue(all(v==(-1. if winner else 1.) for _,v in branches))
            with self.assertRaises(ValueError):session.encode_packed([(cell,0)],1)
        sides=[SideSpec((mon('snorlax',100,'explosion',200),)),SideSpec((victim,))]
        root=replace(root,state_str=build_poke_engine_state(BattleSpec(*sides)).to_string(),
            party_species={'p1':('snorlax',),'p2':('blissey',)})
        previous=list(old.outcomes(root,'explosion','splash'))
        session=self.session(root)
        _,branches,_=session.enumerate_cell('explosion','splash')
        self.assertEqual(branches,[(p.probability,old.terminal_value(p.state)) for p in previous])
        self.assertTrue(any(v==0. for _,v in branches),'short-party double exhaustion is a draw')
        passive=list(old.outcomes(root,'No Move','No Move'))
        _,branches,_=self.session(root).enumerate_cell('none','none')
        self.assertEqual(branches,[(p.probability,old.terminal_value(p.state)) for p in passive])
