"""Benchmark completion counting, using a fake environment, not game evidence."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/engine_env_benchmark.py'
spec=importlib.util.spec_from_file_location('engine_env_benchmark_accounting',path)
benchmark=importlib.util.module_from_spec(spec);sys.modules[spec.name]=benchmark;spec.loader.exec_module(benchmark)


class FakeEnv:
    def __init__(self,finish=None,empty=None):self.finish,self.empty,self.round=finish,empty,0
    def reset(self,**kwargs):self.round=0
    def requested_players(self):return () if self.round==self.empty else ('p1','p2')
    def observe(self,player):return None
    def terminal(self):return SimpleNamespace(turn_count=7) if self.round==self.finish else None
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

    def test_policy_string_seed_is_repeatable(self):
        import random
        a=random.Random('17:p1');b=random.Random('17:p1')
        self.assertEqual([a.random() for _ in range(4)],[b.random() for _ in range(4)])
