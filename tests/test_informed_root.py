"""Authored P2 information/seat contracts; no measured strength claims."""
from dataclasses import replace
from itertools import product
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.informed_root import (exact_informed_one_ply,
                                    sample_informed_equilibrium,
                                    solve_informed_root_lp)
from pokezero.one_ply_exact import solve_matrix_game_lp
from pokezero.shallow_exact import ChanceBranch, ExactSearchRefused


class InformedLPTests(unittest.TestCase):
    def test_compact_lp_equals_full_contingent_plan_game(self):
        tables = (((.9,.1),(.3,.6)), ((.2,.8,.4),(.7,.1,.5)), ((.8,),(.2,)))
        plans = tuple(product(*(range(len(m[0])) for m in tables)))
        full = [[math.fsum(m[i][j] for m,j in zip(tables,plan))/len(tables)
                 for plan in plans] for i in range(2)]
        result = solve_informed_root_lp(tables)
        reference = solve_matrix_game_lp(full)
        self.assertAlmostEqual(result.value, reference.value)
        self.assertAlmostEqual(result.lower_value, reference.lower_value)
        self.assertLessEqual(abs(result.gap), 1e-8)
        self.assertEqual(tuple(map(len,result.opponent_mixes)),(2,3,1))
        self.assertAlmostEqual(sum(result.own_mix),1)
        self.assertTrue(all(abs(sum(y)-1)<1e-10 for y in result.opponent_mixes))

    def test_independent_responses_are_not_a_shared_column_solve(self):
        tables = (((1.,0.),(1.,0.)), ((0.,1.),(0.,1.)))
        self.assertEqual(solve_matrix_game_lp([[.5,.5],[.5,.5]]).value,.5)
        result = solve_informed_root_lp(tables)
        self.assertEqual(result.value,0)
        self.assertEqual(result.world_lower_values,(0.,0.))
        self.assertEqual(result.opponent_mixes,((0.,1.),(1.,0.)))

    def test_world_specific_own_optima_do_not_get_optimized_independently(self):
        result = solve_informed_root_lp([[[1],[0]],[[0],[1]]])
        self.assertAlmostEqual(result.value,.5)  # NOT mean(per-world max)=1.
        self.assertEqual(len(result.own_mix),2)

    def test_independent_column_permutations_and_common_row_permutation(self):
        a = ((.9,.1),(.3,.6))
        b = ((.2,.8,.4),(.7,.1,.5))
        original = solve_informed_root_lp([a,b])
        changed = solve_informed_root_lp([[tuple(reversed(r)) for r in reversed(a)],
                                          [[r[2],r[0],r[1]] for r in reversed(b)]])
        self.assertAlmostEqual(original.value,changed.value)
        for p,q in zip(original.own_mix,reversed(changed.own_mix)):
            self.assertAlmostEqual(p,q)

    def test_invalid_matrices_limits_and_deadline_refuse(self):
        for tables in ([],[[]],[[[]]],[[[1],[2,3]]],[[[1]],[[1],[2]]],
                       [[[math.nan]]],[[[math.inf]]]):
            with self.subTest(tables=tables),self.assertRaises(ExactSearchRefused):
                solve_informed_root_lp(tables)
        for kwargs in (dict(tolerance=0),dict(tolerance=math.nan),dict(deadline=0)):
            with self.assertRaises(ExactSearchRefused):solve_informed_root_lp([[[.5]]],**kwargs)

    def test_solver_failures_bad_coordinates_and_false_success_refuse(self):
        with patch('scipy.optimize.linprog',return_value=SimpleNamespace(
                success=False,status=1,message='test')):
            with self.assertRaisesRegex(ExactSearchRefused,'LP refused'):
                solve_informed_root_lp([[[.5]]])
        for values in ([math.nan,.5],[.8,.5],[1.,.9],[1.,.5,.3]):
            with patch('scipy.optimize.linprog',return_value=SimpleNamespace(
                    success=True,status=0,x=values,nit=1)):
                with self.subTest(values=values),self.assertRaises(ExactSearchRefused):
                    solve_informed_root_lp([[[.5]]])

    def test_same_world_lp_and_negative_scores_remain_mathematical_games(self):
        for matrix in ([[1,0],[0,.5]],[[-2]],[[.3,.3],[.3,.3]]):
            actual=solve_informed_root_lp([matrix])
            reference=solve_matrix_game_lp(matrix)
            self.assertAlmostEqual(actual.value,reference.value)


class Backend:
    def __init__(self, roots, tables):
        self.roots,self.tables=roots,tables
        self.calls=[];self.evaluated=[];self.reset=0
    def begin_decision(self):self.reset+=1
    def terminal_value(self,state):
        return state[4] if len(state)==5 and state[3]=='terminal' else None
    def actions(self,state,*,root):
        assert root and len(state)==1,'NO child action/replacement search'
        return self.roots[state[0]]
    def outcomes(self,state,row,col):
        self.calls.append((state,row,col))
        value=self.tables[state[0]][row,col]
        return [ChanceBranch((state[0],row,col,'head',value),.25),
                ChanceBranch((state[0],row,col,'terminal',value),.75)]
    def evaluate_leaves(self,states):
        self.evaluated.extend(states)
        return [s[4] for s in states]


class InformedExpansionTests(unittest.TestCase):
    def backend(self, side=0, permute=False):
        # Actor two has two common column actions. Opponent action identities
        # and cardinalities differ by world; its first native row is not ours.
        roots=[(('u','v'),('a','b')), (('q',),('b','a') if permute else ('a','b'))]
        values=[{('u','a'):1.,('u','b'):-1.,('v','a'):.2,('v','b'):-.2},
                {('q','a'):.8,('q','b'):-.6}]
        if side==0:
            roots=[(cols,rows) for rows,cols in roots]
            values=[{(c,r):-v for (r,c),v in m.items()} for m in values]
        return Backend(roots,values)

    def test_actor_two_complements_transposes_and_samples_own_mix(self):
        b=self.backend(side=1,permute=True)
        result=exact_informed_one_ply([(0,),(1,)],b,acting_side=1,batch_size=3)
        self.assertEqual(result.own_actions,('a','b'))
        self.assertEqual(result.opponent_actions,(('u','v'),('q',)))
        for actual,expected in zip(result.world_matrices,(((0.,.4),(1.,.6)),((.1,),(.8,)))):
            for r,s in zip(actual,expected):
                for v,w in zip(r,s):self.assertAlmostEqual(v,w)
        self.assertEqual(result.equilibrium.own_mix,(0.,1.))
        receipt=sample_informed_equilibrium(result,game_seed=2,turn=7)
        self.assertEqual(receipt['action'],'b')
        self.assertEqual(receipt['mix'],(0.,1.))
        self.assertEqual(b.reset,1)
        self.assertEqual((result.joint_actions,result.chance_branches,result.leaf_evaluations),(6,12,6))

    def test_actor_symmetry_common_semantic_rows_and_batching(self):
        a,b=self.backend(0,True),self.backend(1,True)
        first=exact_informed_one_ply([(0,),(1,)],a,acting_side=0,batch_size=5)
        second=exact_informed_one_ply([(0,),(1,)],b,acting_side=1,batch_size=1,
                                      cross_cell_batches=False)
        self.assertEqual(first.own_actions,second.own_actions)
        self.assertEqual(first.equilibrium.own_mix,second.equilibrium.own_mix)
        self.assertAlmostEqual(first.equilibrium.value,second.equilibrium.value)
        self.assertEqual(first.evaluation_batches,2)
        self.assertEqual(second.evaluation_batches,6)

    def test_own_action_mismatch_refuses_before_any_expansion(self):
        b=self.backend(1);b.roots[1]=(('q',),('a','other'))
        with self.assertRaisesRegex(ExactSearchRefused,'acting-player action identities'):
            exact_informed_one_ply([(0,),(1,)],b,acting_side=1)
        self.assertFalse(b.calls)

    def test_whole_decision_refusal_keeps_failed_world_and_caps(self):
        for options in (dict(max_leaves=1),dict(max_chance_branches=1),
                        dict(max_joint_actions=5),dict(acting_side=True),dict(acting_side=2)):
            kwargs=dict(acting_side=1);kwargs.update(options)
            with self.subTest(options=options),self.assertRaises(ExactSearchRefused):
                exact_informed_one_ply([(0,),(1,)],self.backend(1),**kwargs)
        b=self.backend(1)
        original=b.outcomes
        def failure(state,row,col):
            if state[0]==1:raise ExactSearchRefused('unsupported world')
            return original(state,row,col)
        b.outcomes=failure
        with patch('pokezero.informed_root.solve_informed_root_lp') as solver:
            with self.assertRaisesRegex(ExactSearchRefused,'unsupported world'):
                exact_informed_one_ply([(0,),(1,)],b,acting_side=1)
            solver.assert_not_called()

    def test_sampling_is_repeatable_full_domain_and_no_support_floor(self):
        result=exact_informed_one_ply([(0,),(1,)],self.backend(1),acting_side=1)
        result=replace(result,equilibrium=replace(result.equilibrium,own_mix=(.999,.001)))
        receipt=sample_informed_equilibrium(result,game_seed=17,turn=5)
        self.assertEqual(receipt,sample_informed_equilibrium(result,game_seed=17,turn=5))
        self.assertEqual(receipt['mix'],(.999,.001))
        for kwargs in (dict(game_seed=18,turn=5),dict(game_seed=17,turn=6),
                       dict(game_seed=17,turn=5,decision_index=1)):
            self.assertNotEqual(receipt['seed_sha256'],sample_informed_equilibrium(result,**kwargs)['seed_sha256'])
        for changed in (replace(result,support_floor='2percent'),
                        replace(result,acting_side=True),
                        replace(result,equilibrium=replace(result.equilibrium,own_mix=(1.,1.)))):
            with self.assertRaises(ExactSearchRefused):
                sample_informed_equilibrium(changed,game_seed=17,turn=5)


if __name__=='__main__':unittest.main()
