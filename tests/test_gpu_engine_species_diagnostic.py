"""Authored accounting fixtures, not scientific self-play measurements."""
import unittest

import numpy as np

from gpu_engine.species_diagnostic import accept_step, clustered_ratio, roster, summarize


class SpeciesDiagnosticTests(unittest.TestCase):
    def test_roster_reverses_each_fresh_pair_once(self):
        self.assertEqual(roster(20,2,30).tolist(),[[20,21],[21,20],[22,23],[23,22]])
        for args in ((-1,2,30),(20,0,30),(20,6,30),(20.,2,30)):
            with self.assertRaises(ValueError):roster(*args)

    def test_pair_clusters_are_not_independent_games(self):
        a=clustered_ratio([0,4],[4,4])
        self.assertEqual(a['estimate'],.5)
        self.assertEqual(a['clusters'],2)
        self.assertEqual(a['standard_error'],.5)
        self.assertIsNone(a['interval95'])
        self.assertIsNone(clustered_ratio([0],[0])['estimate'])

    @staticmethod
    def data(pairs=32):
        original=np.array([[[1]*6,[2]*6]])
        species=np.tile(np.stack((original,original[:,::-1]),axis=1).reshape(2,2,6),(pairs,1,1))
        return species,np.tile([0,1],pairs),np.ones(2*pairs,bool),np.ones_like(species),np.zeros_like(species)

    def test_genuinely_stronger_species_not_assumed_half(self):
        result=summarize(*self.data(),['empty','first','second'])
        first,second=result['species']
        self.assertEqual(first['team_score']['estimate'],1)
        self.assertEqual(second['team_score']['estimate'],0)
        self.assertEqual(first['team_score']['clusters'],32)
        self.assertEqual(first['presence_games'],64)
        self.assertEqual(first['seat0_minus_seat1_complete_pairs']['estimate'],0)
        self.assertEqual(first['fixed_roster_seat_difference_bounds'],[0,0])
        self.assertIn('No species-50% null',result['interpretation'])

    def test_caps_and_unstarted_cells_retained_in_bounds(self):
        species,winners,attempted,active,moves=self.data(2)
        winners[1:]=-1;attempted[2:]=False;active[2:]=0
        r=summarize(species,winners,attempted,active,moves,['empty','first','second'])
        self.assertEqual((r['terminal_games'],r['capped_or_interrupted_games'],r['unstarted_games']),(1,1,2))
        self.assertEqual(r['species'][0]['fixed_roster_score_bounds'],[.25,1])
        self.assertEqual(r['species'][0]['fixed_roster_seat_difference_bounds'],[-.5,1])
        self.assertIsNone(r['species'][0]['seat0_minus_seat1_complete_pairs']['estimate'])
        self.assertEqual(r['species'][0]['unstarted_presence_games'],2)

    def test_draws_and_validation(self):
        args=list(self.data(1));args[1][:]=2
        r=summarize(*args,['empty','first','second'])
        self.assertEqual(r['draws'],2);self.assertEqual(r['species'][0]['team_score']['estimate'],.5)
        args[2][:]=False
        with self.assertRaises(ValueError):summarize(*args,['empty','first','second'])

    def test_failed_step_cannot_admit_new_winners_or_counts(self):
        previous=([-1,0],[2,3],[1,2])
        accepted,valid=accept_step(previous,([1,0],[3,3],[2,2],True))
        self.assertFalse(valid);self.assertIs(accepted,previous)
        accepted,valid=accept_step(previous,([1,0],[3,3],[2,2],False))
        self.assertTrue(valid);self.assertEqual(accepted,([1,0],[3,3],[2,2]))

    def test_seat_ranking_requires_its_own_complete_pair_clusters(self):
        species,winners,attempted,active,moves=self.data(32)
        winners[1::2]=-1
        r=summarize(species,winners,attempted,active,moves,['empty','first','second'])
        self.assertEqual(r['exploratory_triage']['largest_absolute_seat_differences'],[])
        args=list(self.data(1));args[0][1]=args[0][0]
        with self.assertRaises(ValueError):summarize(*args,['empty','first','second'])

    def test_driver_does_not_count_terminal_actors(self):
        import jax
        import jax.numpy as j
        from gpu_engine.champion import Output
        from gpu_engine.species_diagnostic import driver
        from gpu_engine.state import empty_state
        from gpu_engine.transition import MoveTable
        table=MoveTable(j.array([0,35,50]),j.array([0,1,0]),j.ones(3,bool),j.full(3,100),
            j.ones(3,bool),j.zeros(3,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(2))
        s=empty_state(2,seed=7);s=s._replace(hp=s.hp.at[:,:,0].set(100),max_hp=j.full_like(s.hp,100),
            species=j.ones_like(s.species),level=j.full_like(s.level,100),stats=j.full_like(s.stats,100),
            moves=j.ones_like(s.moves),pp=j.full_like(s.pp,8),done=j.array([False,True]))
        forward=lambda o:Output(j.zeros((2,2,9)),j.zeros((2,2)),j.zeros((2,2,9)))
        counts=j.zeros((2,2,6),j.int32)
        state,active,moves,invalid=driver(forward,table)(s,j.array([7,8]),j.array(0),counts,counts)
        self.assertFalse(bool(invalid));self.assertEqual(active.sum(axis=2).tolist(),[[1,1],[0,0]])
        self.assertEqual(moves.sum(axis=2).tolist(),[[1,1],[0,0]])
        self.assertEqual(int(state.turn[0]),1)

        nonfinite=lambda o:Output(j.full((2,2,9),j.nan),j.zeros((2,2)),j.zeros((2,2,9)))
        candidate=driver(nonfinite,table)(s,j.array([7,8]),j.array(0),counts,counts)
        accepted,valid=accept_step((s,counts,counts),candidate)
        self.assertFalse(valid);self.assertIs(accepted[0],s)

        # One side must replace a fainted active. The healthy side is waiting,
        # even though its ordinary move mask still contains legal actions.
        s=s._replace(hp=s.hp.at[0,0,0].set(0).at[0,0,1].set(100))
        state,active,moves,invalid=driver(forward,table)(s,j.array([7,8]),j.array(0),counts,counts)
        self.assertFalse(bool(invalid));self.assertEqual(int(active.sum()),0)
        self.assertEqual(int(moves.sum()),0)


if __name__=='__main__':unittest.main()
