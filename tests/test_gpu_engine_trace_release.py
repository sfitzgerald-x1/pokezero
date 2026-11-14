"""A compiled DepthN program keeps no trace alive (gpu_engine.trace_release).

This is the retained-jaxpr GC stall in miniature. The depth-2 tests' fake executor holds jax.jit-wrapped legal, observe,
sample and advance on a long-lived Executor, as the live executor does. A block program's trace calls those jitted functions
on its own tracers, and jit's tracing caches then keep their jaxprs for as long as the functions live, long after the
program is compiled. CPU only: a few small programs and well under 1 GiB."""
import gc
import unittest

import jax
import jax.numpy as j

from gpu_engine import device_depthn_pricer as dn
from tests import test_gpu_engine_depth2_pricer as T


def live_equations():
    """JaxprEqn objects alive after a full collection."""
    gc.collect()
    return sum(1 for o in gc.get_objects() if type(o).__name__ == 'JaxprEqn')


class CompileEvents:
    """JAX's trace, MLIR-lowering and backend-compile events seen while active."""
    def __enter__(self):
        self.seen = []
        self.listener = lambda event, duration, **kw: self.seen.append(event) if event.startswith('/jax/core/compile/') else None
        jax.monitoring.register_event_duration_secs_listener(self.listener)
        return self

    def __exit__(self, *exc):
        from jax._src import monitoring
        monitoring.unregister_event_duration_listener(self.listener)


def block_inputs(nodes, k_own, k_opp, blocks=1):
    """One call's gathered batch for the fake executor's worlds: [blocks*nodes] node states, both seats requested."""
    n = blocks * nodes
    state = T.State(j.full((n, 1), .25, j.float32), j.zeros((n, 1), bool), j.zeros((n, 1), j.int32), j.zeros((n, 1), j.int32),
        j.arange(n, dtype=j.uint32)[:, None], j.zeros((n, 1), j.int32), j.ones((n, 1), j.int32), j.full((n, 1), -1, j.int32))
    public = T.Public(j.zeros((n,), j.float32), j.ones((n, 1), j.int32))
    return (state, public, j.zeros((n,), j.int32), j.ones((n, 2), bool), j.broadcast_to(j.arange(k_own, dtype=j.int32), (n, k_own)),
        j.broadcast_to(j.arange(k_opp, dtype=j.int32), (n, k_opp)), j.ones((n,), bool), j.zeros((blocks, 2), j.int32))


class TraceReleaseTests(unittest.TestCase):
    def test_a_compiled_block_program_keeps_no_traced_equation(self):
        ex, _ = T.executor();nodes, k_own, k_opp = 2, 2, 2
        pricer = dn.DepthNPricer(ex, object(), capacity=1, depth=2, k_own=k_own, k_opp=k_opp, blocks_per_call=1,
            block_lanes=nodes * k_own * k_opp, forward=T.forward, actor_capacity=3)
        pricer.bind(T.PARAMS[:1]);args = block_inputs(nodes, k_own, k_opp)
        one = T.State(*(x[:1] for x in args[0]))
        jax.block_until_ready(ex.legal(one))   # an eager call of an executor function, before the program: its executable is cached
        before = live_equations()
        key, build = pricer._block_program(2, nodes)
        compiled = pricer._program(key, build, *args);first = jax.device_get(compiled(*args))
        retained = live_equations() - before
        # Today's failure: the trace of every executor function the program called stays in jit's caches (about a thousand
        # equations here; about 66M objects after a DN2W16 server warmup).
        self.assertLessEqual(retained, 0, f'{retained} traced equations outlive the compiled program')
        # The executable alone runs the program again, and nothing traces or compiles again: neither the program (its table
        # entry is reused) nor the eager executor call at the signature it already ran.
        with CompileEvents() as events:
            again = jax.device_get(pricer._program(key, build, *args)(*args))
            jax.block_until_ready(ex.legal(one))
        self.assertEqual(events.seen, [])
        for name in first:
            self.assertTrue((first[name] == again[name]).all(), name)
        self.assertEqual(len(pricer.compiles), 1)

    def test_release_empties_both_tracing_caches_of_this_jax(self):
        from gpu_engine import trace_release
        caches = trace_release._caches()
        self.assertEqual(len(caches), 2, 'jit\'s tracing caches moved: trace_release no longer finds both')
        held = jax.jit(lambda x: j.sin(x) * 2)   # a long-lived jitted function, traced inside another program
        jax.jit(lambda x: held(x) + 1).lower(j.ones(3, j.float32)).compile()
        self.assertGreater(sum(c.cache_info().currsize for c in caches), 0)
        self.assertGreater(trace_release.release(), 0)
        self.assertEqual(sum(c.cache_info().currsize for c in caches), 0)


if __name__ == '__main__':
    unittest.main()
