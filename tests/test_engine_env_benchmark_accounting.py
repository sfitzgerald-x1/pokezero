"""Benchmark completion counting, using a fake environment, not game evidence."""
import importlib.util
from pathlib import Path
import sys
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'scripts/engine_env_benchmark.py'
spec=importlib.util.spec_from_file_location('engine_env_benchmark_accounting',path)
benchmark=importlib.util.module_from_spec(spec);sys.modules[spec.name]=benchmark;spec.loader.exec_module(benchmark)


class FakeEnv:
    def __init__(self,finish=None,empty=None,capped=False):self.finish,self.empty,self.round,self.capped=finish,empty,0,capped
    def reset(self,**kwargs):self.round=0
    def requested_players(self):return () if self.round==self.empty else ('p1','p2')
    def observe(self,player):return None
    def terminal(self):return SimpleNamespace(turn_count=7,capped=self.capped) if self.round==self.finish else None
    def step(self,actions):
        self.round+=1
        return SimpleNamespace(requested_players=self.requested_players(),observations={'p1':None,'p2':None})


class Policy:
    def select_action(self,observation,*,rng):return SimpleNamespace(action_index=0)


class BenchmarkAccountingTests(unittest.TestCase):
    def play(self,env,cap=2):
        arm=benchmark.ArmResult('fixture');arm.wall=10
        benchmark._play(env,{p:Policy() for p in ('p1','p2')},17,'authored-fixture',cap,arm)
        return arm

    def test_capped_attempt_is_not_a_completed_game(self):
        arm=self.play(FakeEnv())
        self.assertEqual((arm.attempted_games,arm.games,arm.incomplete_games,arm.games_per_second),(1,0,1,0))
        self.assertEqual(arm.outcomes,[dict(seed=17,completed=False,decision_rounds=2,turns=None,stop_reason='DECISION_ROUND_CAP')])

    def test_terminal_at_cap_is_completed(self):
        arm=self.play(FakeEnv(finish=2))
        self.assertEqual((arm.games,arm.incomplete_games,arm.games_per_second,arm.turns),(1,0,.1,7))
        self.assertEqual(arm.outcomes[0]['stop_reason'],'TERMINAL')

    def test_nonterminal_empty_request_is_retained_as_incomplete(self):
        arm=self.play(FakeEnv(empty=1))
        self.assertEqual(arm.outcomes[0]['stop_reason'],'NO_REQUESTED_PLAYERS')
        self.assertEqual((arm.games,arm.incomplete_games),(0,1))

    def test_environment_capped_terminal_is_not_completion(self):
        arm=self.play(FakeEnv(finish=1,capped=True))
        self.assertEqual((arm.games,arm.incomplete_games,arm.turns),(0,1,0))
        self.assertEqual(arm.outcomes[0]['stop_reason'],'ENVIRONMENT_CAP')
        self.assertEqual(arm.outcomes[0]['turns'],7)

    def test_report_zero_and_mixed_completion_metrics_are_not_invented(self):
        args=SimpleNamespace(games=2,seed_start=17,format_id='fixture',policy_label='fake')
        incomplete=self.play(FakeEnv())
        complete=self.play(FakeEnv(finish=1));complete.game_seconds=[2.5]
        for other in (incomplete,complete):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):benchmark.report(incomplete,other,args)
            lines=output.getvalue().splitlines()
            for label in ('all-attempt wall sec / completion','completed game sec (median)',
                    'all-attempt env sec / completion','completed mean turns'):
                row=next(line for line in lines if line.startswith(label))
                values=row[len(label):].split()
                self.assertEqual((values[0],values[-1]),('n/a','n/a'))
                self.assertEqual(values[1],'n/a') if other is incomplete else self.assertGreater(float(values[1]),0)
            self.assertIn('No 16k-game projection',output.getvalue())

    def test_policy_string_seed_is_repeatable(self):
        import random
        a=random.Random('17:p1');b=random.Random('17:p1')
        self.assertEqual([a.random() for _ in range(4)],[b.random() for _ in range(4)])

    def test_random_legal_report_needs_no_torch(self):
        arm=self.play(FakeEnv())
        args=SimpleNamespace(games=1,seed_start=17,format_id='fixture',policy_label='random-legal (no model forward)')
        output=io.StringIO()
        with patch.dict(sys.modules,{'torch':None}),contextlib.redirect_stdout(output):benchmark.report(arm,arm,args)
        self.assertIn('no neural policy or Torch forward',output.getvalue())
