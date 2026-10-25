"""Authored one-ply/LP contracts, not measured gameplay or fidelity credit."""
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.one_ply_exact import exact_one_ply, sample_equilibrium, solve_matrix_game_lp
from pokezero.shallow_exact import ChanceBranch, ExactSearchRefused


class LPTests(unittest.TestCase):
    def test_matching_pennies_and_asymmetric_probability_games(self):
        for matrix, mix, value in (([[1,0],[0,1]], .5, .5),
                                    ([[1,0],[0,.5]], 1/3, 1/3)):
            solution=solve_matrix_game_lp(matrix)
            self.assertAlmostEqual(solution.row_mix[0],mix)
            self.assertAlmostEqual(solution.column_mix[0],mix)
            self.assertAlmostEqual(solution.value,value)
            self.assertLessEqual(solution.gap,1e-8)
            self.assertEqual((solution.primal_status,solution.dual_status),(0,0))
            self.assertEqual(solution.solver,'highs-ds')
            self.assertTrue(solution.scipy_version)

    def test_rectangular_pure_negative_and_degenerate_games(self):
        for matrix in ([[.2,.7],[.1,.5],[.15,.1]], [[-2]], [[.3,.3],[.3,.3]]):
            s=solve_matrix_game_lp(matrix)
            self.assertAlmostEqual(sum(s.row_mix),1)
            self.assertAlmostEqual(sum(s.column_mix),1)
            lower=min(sum(s.row_mix[i]*matrix[i][j] for i in range(len(matrix)))
                      for j in range(len(matrix[0])))
            upper=max(sum(a*p for a,p in zip(row,s.column_mix)) for row in matrix)
            self.assertEqual((s.lower_value,s.upper_value),(lower,upper))
            self.assertLessEqual(s.gap,1e-8)

    def test_invalid_and_failed_solver_refuse_without_fallback(self):
        for matrix in ([],[[]],[[1],[1,2]],[[math.nan]],[[math.inf]]):
            with self.subTest(matrix=matrix),self.assertRaises(ExactSearchRefused):
                solve_matrix_game_lp(matrix)
        with patch('scipy.optimize.linprog',return_value=SimpleNamespace(
                success=False,status=1,message='bounded test')):
            with self.assertRaisesRegex(ExactSearchRefused,'LP refused'):
                solve_matrix_game_lp([[.5]])

    def test_false_success_bad_mix_and_gap_are_independently_refused(self):
        for values in ([math.nan,0,0],[-.1,1.1,0],[.8,.8,0],[1,0,.5]):
            with patch('scipy.optimize.linprog',return_value=SimpleNamespace(
                    success=True,status=0,x=values,nit=1)):
                with self.subTest(values=values),self.assertRaises(ExactSearchRefused):
                    solve_matrix_game_lp([[1,0],[0,1]])

    def test_deadline_is_not_solver_success(self):
        with self.assertRaisesRegex(ExactSearchRefused,'deadline'):
            solve_matrix_game_lp([[.5]],deadline=0)
        with patch('pokezero.one_ply_exact.time.monotonic',side_effect=[0.,2.]), \
                patch('scipy.optimize.linprog',return_value=SimpleNamespace(success=True,status=0)):
            with self.assertRaisesRegex(ExactSearchRefused,'deadline reached in LP'):
                solve_matrix_game_lp([[.5]],deadline=1.)


class Backend:
    def __init__(self):
        self.reset=0; self.calls=[]; self.evaluated=[]
    def begin_decision(self): self.reset+=1
    def terminal_value(self,state): return None
    def actions(self,state,*,root):
        assert root and len(state)==1, 'NO child action/replacement search'
        return ('a','b'),('x','y')
    def outcomes(self,state,row,col):
        self.calls.append((state,row,col))
        return [ChanceBranch((state[0],row,col,-1),.25),
                ChanceBranch((state[0],row,col,1),.75)]
    def evaluate_leaves(self,states):
        self.evaluated.extend(states)
        return [(1 if (s[1],s[2]) in (('a','x'),('b','y')) else -1)*s[3] for s in states]


class OnePlyTests(unittest.TestCase):
    def test_r4_bundle_deadline_fixture_retains_exact_failed_address_without_a_choice(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/p2_deadline_r4.json').read_text())
        clock = [0.0]
        class CapturedAddress(Backend):
            def actions(self,state,*,root):
                return (fixture['row'],),(fixture['column'],)
            def outcomes(self,state,row,col):
                count = fixture['branch_index']+1
                return [ChanceBranch((state[0],row,col,index),1/count) for index in range(count)]
            def terminal_value(self,state):
                if len(state) == 4 and state[0] == fixture['world_ordinal'] and state[3] == fixture['branch_index']:
                    clock[0] = fixture['deadline_seconds']+1
                return None
            def evaluate_leaves(self,states):
                return [0.0]*len(states)
        with patch('pokezero.one_ply_exact.time.monotonic',side_effect=lambda:clock[0]), \
                patch('pokezero.one_ply_exact.solve_matrix_game_lp') as solve:
            with self.assertRaisesRegex(ExactSearchRefused,fixture['original_error']) as caught:
                exact_one_ply([(i,) for i in range(4)],CapturedAddress(),
                    max_seconds=fixture['deadline_seconds'],cross_cell_batches=True)
        solve.assert_not_called()
        work = caught.exception.work
        self.assertEqual(caught.exception.failure_codes,['p2.decision.deadline'])
        self.assertEqual((work['world_ordinal'],work['row_action'],work['column_action'],work['branch_index']),
            (fixture['world_ordinal'],fixture['row'],fixture['column'],fixture['branch_index']))
        self.assertEqual(work['phase'],'leaf_terminal_check')
        self.assertEqual(work['worlds_enumerated'],fixture['world_ordinal'])

    def test_deadline_after_terminal_check_retains_guarded_work_and_never_solves(self):
        clock = [0.0]
        class LateTerminal(Backend):
            def terminal_value(self,state):
                if len(state) == 4:
                    clock[0] = 2.0
                return None
        with patch('pokezero.one_ply_exact.time.monotonic',side_effect=lambda:clock[0]), \
                patch('pokezero.one_ply_exact.solve_matrix_game_lp') as solve:
            with self.assertRaisesRegex(ExactSearchRefused,'deadline reached') as caught:
                exact_one_ply([(0,)],LateTerminal(),max_seconds=1)
        solve.assert_not_called()
        self.assertEqual(caught.exception.failure_codes,['p2.decision.deadline'])
        work = caught.exception.work
        self.assertEqual(work['phase'],'leaf_terminal_check')
        self.assertEqual((work['world_ordinal'],work['row_action'],work['column_action'],work['branch_index']),
                         (0,'a','x',0))
        self.assertEqual((work['chance_branches_checked'],work['leaves_queued'],
                          work['batches_accepted'],work['pending_leaves']),(2,0,0,0))
        self.assertEqual(work['worlds_enumerated'],0)

    def test_deadline_after_forward_distinguishes_accepted_batch_from_inflight_work(self):
        clock = [0.0]
        class LateForward(Backend):
            def evaluate_leaves(self,states):
                clock[0] = 2.0
                return super().evaluate_leaves(states)
        with patch('pokezero.one_ply_exact.time.monotonic',side_effect=lambda:clock[0]):
            with self.assertRaises(ExactSearchRefused) as caught:
                exact_one_ply([(0,)],LateForward(),max_seconds=1,batch_size=2)
        work = caught.exception.work
        self.assertEqual(work['phase'],'leaf_evaluation')
        self.assertEqual((work['leaves_queued'],work['pending_leaves'],work['batches_accepted']),(2,2,0))

    def test_cross_cell_batches_preserve_world_cell_weight_and_leaf_order(self):
        class Distinct(Backend):
            def evaluate_leaves(self,states):
                self.evaluated.extend(states)
                return [s[3]*(s[0]+1)/4*(1 if s[1]=='a' else -1) for s in states]
        worlds=[(i,) for i in range(4)];old,new=Distinct(),Distinct()
        serial=exact_one_ply(worlds,old,batch_size=7)
        bulk=exact_one_ply(worlds,new,batch_size=7,cross_cell_batches=True)
        self.assertEqual(serial.root_matrix,bulk.root_matrix)
        self.assertEqual(serial.equilibrium,bulk.equilibrium)
        self.assertEqual(old.evaluated,new.evaluated)
        self.assertEqual(serial.leaf_evaluations,bulk.leaf_evaluations)
        self.assertEqual((serial.evaluation_batches,bulk.evaluation_batches),(16,5))

    def test_cross_cell_terminal_holes_and_final_partial_batch_preserve_expectations(self):
        class Mixed(Backend):
            def terminal_value(self,state):
                return state[3] if len(state)==4 and state[0]%2==0 and state[2]=='x' else None
        old,new=Mixed(),Mixed();worlds=[(i,) for i in range(3)]
        serial=exact_one_ply(worlds,old,batch_size=5)
        bulk=exact_one_ply(worlds,new,batch_size=5,cross_cell_batches=True)
        self.assertEqual(serial.root_matrix,bulk.root_matrix)
        self.assertEqual(old.evaluated,new.evaluated)
        self.assertEqual(bulk.evaluation_batches,math.ceil(bulk.leaf_evaluations/5))

    def test_cross_cell_mode_keeps_caps_and_whole_decision_refusals(self):
        for options in (dict(max_leaves=1),dict(max_chance_branches=1),
                        dict(max_joint_actions=1),dict(cross_cell_batches=1)):
            kwargs=dict(cross_cell_batches=True);kwargs.update(options)
            with self.assertRaises(ExactSearchRefused):exact_one_ply([(0,)],Backend(),**kwargs)
        class Missing(Backend):
            def evaluate_leaves(self,states):return [0]*(len(states)-1)
        with self.assertRaisesRegex(ExactSearchRefused,'cardinality'):
            exact_one_ply([(0,)],Missing(),cross_cell_batches=True)
        class LateMass(Backend):
            def outcomes(self,state,row,col):
                return super().outcomes(state,row,col) if row=='a' else [ChanceBranch((0,row,col,0),.9)]
        with patch('pokezero.one_ply_exact.solve_matrix_game_lp') as solver:
            with self.assertRaisesRegex(ExactSearchRefused,'unit mass'):
                exact_one_ply([(0,)],LateMass(),cross_cell_batches=True)
            solver.assert_not_called()

    def test_four_worlds_complete_one_ply_and_single_tanh_conversion(self):
        backend=Backend()
        result=exact_one_ply([(i,) for i in range(4)],backend,batch_size=1)
        self.assertEqual(result.root_matrix,((.75,.25),(.25,.75)))
        self.assertEqual((result.worlds,result.joint_actions,result.chance_branches,
                          result.leaf_evaluations,result.evaluation_batches),(4,16,32,32,32))
        self.assertEqual(backend.reset,1)
        self.assertEqual(result.equilibrium.row_mix,(.5,.5))
        self.assertTrue(all(len(s)==4 for s in backend.evaluated))

    def test_average_world_matrices_before_single_solve(self):
        class Different(Backend):
            def evaluate_leaves(self,states):
                return [(-1 if s[0]==0 else 1) if s[1]=='a' else 0 for s in states]
        result=exact_one_ply([(0,),(1,),(1,),(1,)],Different())
        self.assertEqual(result.root_matrix,((.75,.75),(.5,.5)))
        self.assertEqual(result.equilibrium.row_mix,(1.,0.))

    def test_game_terminals_exact_but_pending_replacement_is_a_head_leaf(self):
        class Terminal(Backend):
            def terminal_value(self,state):
                return state[3] if len(state)==4 and state[1]=='a' else None
        b=Terminal();result=exact_one_ply([(0,)],b)
        self.assertEqual(result.root_matrix,((.75,.75),(.25,.75)))
        self.assertEqual(len(b.evaluated),4)
        self.assertTrue(all(s[1]=='b' for s in b.evaluated))
        class Draw(Terminal):
            def terminal_value(self,state): return 0 if len(state)==4 else None
        self.assertEqual(exact_one_ply([(0,)],Draw()).root_matrix,((.5,.5),(.5,.5)))

    def test_nine_by_nine_is_actual_actions_not_padded_illegal_cells(self):
        class Nine(Backend):
            def actions(self,state,*,root): return tuple(range(9)),tuple(range(9))
            def outcomes(self,state,row,col): return [ChanceBranch((0,row,col,0),1)]
            def evaluate_leaves(self,states): return [0]*len(states)
        r=exact_one_ply([(i,) for i in range(4)],Nine())
        self.assertEqual((r.joint_actions,r.leaf_evaluations),(324,324))
        self.assertEqual(len(r.root_matrix),9)

    def test_hidden_action_mismatch_refused_before_transitions(self):
        class Different(Backend):
            def actions(self,state,*,root): return (('a' if state[0]==0 else 'other'),),('x',)
        b=Different()
        with self.assertRaisesRegex(ExactSearchRefused,'identities differ'):
            exact_one_ply([(0,),(1,)],b)
        self.assertFalse(b.calls)

    def test_caps_bad_probability_or_bad_value_never_return_partial_strategy(self):
        for limits in (dict(max_leaves=1),dict(max_chance_branches=1),dict(max_joint_actions=1)):
            with self.subTest(limits=limits),self.assertRaises(ExactSearchRefused):
                exact_one_ply([(0,)],Backend(),**limits)
        class BadMass(Backend):
            def outcomes(self,*args): return [ChanceBranch((0,'a','x',1),.9)]
        with self.assertRaisesRegex(ExactSearchRefused,'unit mass'):
            exact_one_ply([(0,)],BadMass())
        class BadValue(Backend):
            def evaluate_leaves(self,states): return [2]*len(states)
        with self.assertRaisesRegex(ExactSearchRefused,'tanh scale'):
            exact_one_ply([(0,)],BadValue())

    def test_sampling_has_no_floor_and_logs_repeatable_full_domain(self):
        result=exact_one_ply([(0,)],Backend())
        first=sample_equilibrium(result,game_seed=17,turn=5,side=0)
        self.assertEqual(first,sample_equilibrium(result,game_seed=17,turn=5,side=0))
        for change in (dict(game_seed=18),dict(turn=6),dict(side=1),dict(decision_index=1)):
            args=dict(game_seed=17,turn=5,side=0);args.update(change)
            self.assertNotEqual(first['seed_sha256'],sample_equilibrium(result,**args)['seed_sha256'])
        self.assertEqual(first['actions'],result.row_actions)
        self.assertEqual(first['mix'],(.5,.5))
        self.assertEqual(first['support_floor'],'NONE')
        self.assertIn(first['action'],result.row_actions)
        tiny=replace(result,equilibrium=replace(result.equilibrium,row_mix=(.999,.001)))
        self.assertEqual(sample_equilibrium(tiny,game_seed=17,turn=5,side=0)['mix'],(.999,.001))
        with self.assertRaises(ExactSearchRefused):
            sample_equilibrium(replace(result,support_floor='2percent'),game_seed=17,turn=5,side=0)

    def test_sampling_uses_distinct_row_and_column_mixes(self):
        result=exact_one_ply([(0,)],Backend())
        asymmetric=replace(result,equilibrium=replace(result.equilibrium,
                            row_mix=(1.,0.),column_mix=(0.,1.)))
        self.assertEqual(sample_equilibrium(asymmetric,game_seed=2,turn=7,side=0)['action'],'a')
        self.assertEqual(sample_equilibrium(asymmetric,game_seed=2,turn=7,side=1)['action'],'y')

    def test_float_mass_correction_is_reported_and_never_drops_a_branch(self):
        class FloatMass(Backend):
            def outcomes(self,state,row,col):
                return [ChanceBranch((0,row,col,-1),.25*(1+1e-8)),
                        ChanceBranch((0,row,col,1),.75*(1+1e-8))]
        result=exact_one_ply([(0,)],FloatMass())
        self.assertAlmostEqual(result.max_chance_mass_error,1e-8)
        self.assertEqual(result.chance_branches,8)
        self.assertEqual(result.leaf_evaluations,8)
        self.assertEqual(result.root_matrix,((.75,.25),(.25,.75)))


if __name__=='__main__': unittest.main()
