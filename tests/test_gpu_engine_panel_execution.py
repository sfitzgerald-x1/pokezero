"""Explicitly invoked native/JAX authored seam controls, not panel evidence."""
import itertools
import importlib.util
import os
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax") and importlib.util.find_spec("poke_engine"),
                     "explicit native/JAX authored execution profile required")
class PanelExecutionTests(unittest.TestCase):
    def fixture(self):
        # Scientific admission separately requires both actual tests, no skips.
        import jax.numpy as jnp
        from gpu_engine import chance_consumers as consumers
        from gpu_engine.state import empty_state
        from gpu_engine.transition import MoveTable
        from tests.test_gpu_engine_future_state import fixture
        data, metadata = fixture()
        metadata["vocabulary"]["moves"].append("struggle")
        for side in range(2):
            for mon in range(6):
                data["moves"][side][mon] = [1, 0, 0, 0]
                data["pp"][side][mon] = [8, 0, 0, 0]
        base = empty_state(1)
        state = base._replace(**{name:jnp.asarray([value], dtype=getattr(base, name).dtype)
                                 for name,value in data.items()})
        n = len(metadata["vocabulary"]["moves"])
        table = MoveTable(jnp.asarray([0,35]+[0]*(n-2), jnp.int32),
            jnp.asarray([0,1]+[0]*(n-2), jnp.int32), jnp.ones(n, bool),
            jnp.asarray([100,95]+[100]*(n-2), jnp.int32), jnp.zeros(n, bool),
            jnp.zeros(n, jnp.int32), jnp.ones((18,18), jnp.int32),
            jnp.ones((18,18), jnp.int32), jnp.asarray(n-1, jnp.int32))
        plan = consumers.plan_for_state(state, 0, (0,0), table, metadata, cap=262144)
        self.assertIsInstance(plan, consumers.ConsumerPlan, plan)
        return state, table, plan

    def test_actual_compiled_padded_batch_retains_one_result_per_real_case(self):
        from gpu_engine.fidelity_panel import JaxCaseExecutor, require_future
        state, table, plan = self.fixture()
        cases = tuple(itertools.islice(plan.bounded_cases(), 3))
        execute = JaxCaseExecutor(state, (0,0), table, plan, batch_size=4)
        result = execute(cases)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(require_future(value).raw_jax for value in result))
        with self.assertRaisesRegex(ValueError, "actions differ"):
            JaxCaseExecutor(state, (1,0), table, plan, batch_size=4)
        with self.assertRaisesRegex(ValueError, "root differs"):
            JaxCaseExecutor(state._replace(turn=state.turn+1), (0,0), table, plan, batch_size=4)

    def test_actual_native_full_future_branches_preserve_parent(self):
        self.assertEqual(os.environ.get("POKEZERO_ENUMERATE_ROLLS"), "1")
        from gpu_engine.fidelity_panel import native_executor, require_future
        _, _, plan = self.fixture()
        before = plan.native_input.state.to_string()
        rows = native_executor(plan)
        self.assertTrue(rows)
        self.assertAlmostEqual(sum(p for p,_ in rows), 1., places=5)
        self.assertTrue(all(require_future(value).raw_native for p,value in rows if p>0))
        self.assertEqual(plan.native_input.state.to_string(), before)

    def test_actual_requested_seats_have_canonical_ignored_actions(self):
        import numpy as np
        from gpu_engine.collect_fidelity_panel import requested_actions
        state, table, _ = self.fixture()
        forced = state._replace(hp=state.hp.at[0,0,0].set(0),
            forced_switch=state.forced_switch.at[0,0].set(True))
        actions, missing, requested = requested_actions(forced, table, salt=104729)
        self.assertEqual(np.asarray(requested).tolist(), [[True,False]])
        self.assertEqual(int(actions[0,1]), 0)
        self.assertGreaterEqual(int(actions[0,0]), 4)
        self.assertFalse(np.asarray(missing).any())
        terminal = state._replace(done=state.done.at[:].set(True))
        actions, missing, requested = requested_actions(terminal, table, salt=104729)
        self.assertEqual(np.asarray(actions).tolist(), [[0,0]])
        self.assertFalse(np.asarray(requested).any())
        self.assertFalse(np.asarray(missing).any())


if __name__ == "__main__":
    unittest.main()
