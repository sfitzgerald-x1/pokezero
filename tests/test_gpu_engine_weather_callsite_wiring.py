"""Actual initializer trace threading, literal engine states; no public claims."""
import ast
import subprocess
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import transition, weather
from gpu_engine import engine_weather_callbacks as callbacks
from test_gpu_engine_engine_weather_callbacks import table, state, tape


class WeatherCallsiteTests(unittest.TestCase):
    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def test_actual_initializer_orders_writes_once_and_preserves_default_state_rng(self):
        s=state(2); rows=j.arange(2)
        s=s._replace(entry_pending=j.ones((2,2),bool),
            ability=s.ability.at[rows,0,2].set(1).at[rows,1,4].set(2),
            base_ability=s.base_ability.at[rows,0,2].set(1).at[rows,1,4].set(2))
        engine=SimpleNamespace(entry=table(),weather=None,items=None)
        coin=j.array([0,1],j.int32)
        expected=transition.initialize(s,engine,tie_coin=coin)
        traced,receipt=transition.initialize(s,engine,tie_coin=coin,_weather_trace=tape(s))
        self.same(expected,traced)
        self.assertEqual(receipt.count.tolist(),[2,2])
        self.assertEqual(receipt.words[:,:2,1].tolist(),[[0,1],[1,0]])
        self.assertEqual(receipt.words[:,:2,2].tolist(),[[2,4],[4,2]])
        self.assertEqual(traced.weather.tolist(),[weather.SUN,weather.RAIN])
        self.assertFalse(bool(traced.entry_pending.any()))
        self.assertFalse(bool((receipt.refusal & callbacks.UNKNOWN_WEATHER).any()))
        fast=jax.jit(lambda x,c,r:transition.initialize(x,engine,tie_coin=c,_weather_trace=r))
        for _ in range(2):
            actual,events=fast(s,coin,tape(s));self.same(actual,expected);self.same(events,receipt)
        self.assertEqual(fast._cache_size(),1)
        again,events=fast(traced,coin,receipt)
        self.same(again,traced);self.same(events,receipt)

    def test_no_entry_table_retains_trace_and_normal_return(self):
        s=state(2);r=tape(s);engine=SimpleNamespace(entry=None)
        self.assertIs(transition.initialize(s,engine),s)
        actual,events=transition.initialize(s,engine,_weather_trace=r)
        self.assertIs(actual,s);self.assertIs(events,r)

    def test_whole_default_transition_ast_stays_equal_to_actual_main_parent(self):
        class Strip(ast.NodeTransformer):
            def visit_If(self,node):
                if ast.unparse(node.test) in ('_weather_trace is not None','_status_trace is not None',
                        '_public_trace is not None','_lifecycle_trace'):
                    return [self.visit(n) for n in node.orelse]
                return self.generic_visit(node)
            def visit_FunctionDef(self,node):
                for name in ('_weather_trace','_weather_hp_context'):
                    if name in [a.arg for a in node.args.kwonlyargs]:
                        i=[a.arg for a in node.args.kwonlyargs].index(name)
                        node.args.kwonlyargs.pop(i);node.args.kw_defaults.pop(i)
                return self.generic_visit(node)
            def visit_ClassDef(self,node):
                if node.name=='WeatherProfileTracedStep':
                    # Remove ONLY this non-executable tracing return type, not
                    # arbitrary classes or any default mathematical statement.
                    expected=ast.parse('''class WeatherProfileTracedStep(NamedTuple):
    """Private owning-profile path only; legacy StepResult/TracedStep unchanged."""
    step: StepResult
    callbacks: object
    weather_profile: object
''').body[0]
                    if ast.dump(node)!=ast.dump(expected):raise ValueError('changed observer type declaration')
                    return None
                return self.generic_visit(node)
        def normalized(raw):
            tree=Strip().visit(ast.parse(raw))
            if any(isinstance(x,ast.Name) and x.id=='WeatherProfileTracedStep' for x in ast.walk(tree)):
                raise ValueError('observer type is reachable from the default program')
            return ast.dump(tree,include_attributes=False)
        old=subprocess.check_output(['git','show','d38caf8808a87cb64319b8e3bf2f358c10729399:gpu_engine/transition.py'])
        from pathlib import Path
        current=Path(transition.__file__).read_text()
        self.assertEqual(normalized(old),normalized(current))
        with self.assertRaises(ValueError):normalized(current.replace('weather_profile: object', 'weather_profile: object = 1',1))
        with self.assertRaises(ValueError):normalized(current+'\nBAD=WeatherProfileTracedStep\n')


if __name__=='__main__':unittest.main()
