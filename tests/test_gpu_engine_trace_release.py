"""No warmed program's trace outlives the live search's warmup (gpu_engine.trace_release).

This is the retained-jaxpr GC stall in miniature. The fake executors (as the live one) hold jax.jit-wrapped legal,
observe, sample and advance for as long as the process lives. A pricer program's trace calls those jitted functions on its
own tracers, and jit's tracing caches then keep their jaxprs for as long as the functions live, long after the program is
compiled. A DN2W16 server warmup kept about 66M objects that way. CPU only: small programs, well under 2 GiB."""
import gc
import unittest
from unittest.mock import patch
import weakref

import jax
import jax.numpy as j

from gpu_engine import depthn_live_search as live,device_depthn_pricer as dn
from tests import test_gpu_engine_depth2_pricer as T
from tests import test_gpu_engine_depthn_live_search as L


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


class WarmupReleaseTests(unittest.TestCase):
    """DepthNLiveSearch.warmup on LiveSearchTests' fake engine and harness."""
    setUpClass = classmethod(L.LiveSearchTests.setUpClass.__func__);make_pricers = classmethod(L.LiveSearchTests.make_pricers.__func__)
    setUp, search, outcome, run_batch = L.LiveSearchTests.setUp, L.LiveSearchTests.search, L.LiveSearchTests.outcome, L.LiveSearchTests.run_batch

    def test_no_trace_of_a_warmed_program_outlives_warmup(self):
        s = self.search(pricers=self.make_pricers(8, (1,)));outcomes = {('w0', 0):self.outcome(0), ('l0', 0):self.outcome(1)}
        gc.collect();before = {id(o) for o in gc.get_objects() if type(o).__name__ == 'Jaxpr'};traced = []
        warm_buckets = live.DepthNLiveSearch._warm_buckets

        def spy(search, pool):   # every jaxpr the bucket passes traced (P1, the selection, the gather, the leaf program) and still alive after them
            out = warm_buckets(search, pool);gc.collect()
            traced.extend(weakref.ref(o) for o in gc.get_objects() if type(o).__name__ == 'Jaxpr' and id(o) not in before)
            return out
        with L.Codex(outcomes), patch.object(live.DepthNLiveSearch, '_warm_buckets', spy):report = s.warmup([L.request('w0', 0)])
        gc.collect();alive = [r() for r in traced if r() is not None];equations = sum(len(x.eqns) for x in alive)
        self.assertGreater(len(traced), 100)
        # Today's failure: jit's tracing caches keep them all (the executor's jitted functions are alive, so their entries are).
        # What may stay is JAX's own strong caches of tiny fixed computations (a reduction's `add a b`), never a trace.
        self.assertLessEqual(equations, 4, f'{len(alive)} of the {len(traced)} jaxprs traced while warming the buckets outlive warmup '
            f'({equations} equations)')
        del alive
        self.assertGreater(report['trace_release']['entries'], 0)
        # After READY, serving a new root traces and compiles nothing: the re-warm pass recompiled what the release unkeyed.
        with CompileEvents() as events:
            rows, _ = self.run_batch(s, [L.request('l0', 0)], outcomes)
        self.assertEqual([r['status'] for r in rows], ['COMPLETE']);self.assertEqual(events.seen, [])


def block_inputs(nodes, k_own, k_opp, blocks=1):
    """One call's gathered batch for the depth-2 fake's worlds: [blocks*nodes] node states, both seats requested."""
    n = blocks * nodes
    state = T.State(j.full((n, 1), .25, j.float32), j.zeros((n, 1), bool), j.zeros((n, 1), j.int32), j.zeros((n, 1), j.int32),
        j.arange(n, dtype=j.uint32)[:, None], j.zeros((n, 1), j.int32), j.ones((n, 1), j.int32), j.full((n, 1), -1, j.int32))
    public = T.Public(j.zeros((n,), j.float32), j.ones((n, 1), j.int32))
    return (state, public, j.zeros((n,), j.int32), j.ones((n, 2), bool), j.broadcast_to(j.arange(k_own, dtype=j.int32), (n, k_own)),
        j.broadcast_to(j.arange(k_opp, dtype=j.int32), (n, k_opp)), j.ones((n,), bool), j.zeros((blocks, 2), j.int32))


class TraceReleaseTests(unittest.TestCase):
    def test_release_drops_a_compiled_programs_trace_and_nothing_recompiles(self):
        from gpu_engine import trace_release
        ex, _ = T.executor();nodes, k_own, k_opp = 2, 2, 2
        pricer = dn.DepthNPricer(ex, object(), capacity=1, depth=2, k_own=k_own, k_opp=k_opp, blocks_per_call=1,
            block_lanes=nodes * k_own * k_opp, forward=T.forward, actor_capacity=3)
        pricer.bind(T.PARAMS[:1]);args = block_inputs(nodes, k_own, k_opp)
        one = T.State(*(x[:1] for x in args[0]))
        jax.block_until_ready(ex.legal(one))   # an eager call of an executor function, before the program: its executable is cached
        before = live_equations()
        key, build = pricer._block_program(2, nodes)
        compiled = pricer._program(key, build, *args);first = jax.device_get(compiled(*args))
        self.assertGreater(live_equations() - before, 100)   # the executor functions' traces, held by jit's caches
        trace_release.release()
        self.assertLessEqual(live_equations() - before, 0)
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
        caches = trace_release.caches()   # raises when this JAX moved either cache (verified on trace_release.VERIFIED_JAX)
        self.assertEqual(len(caches), 2)
        self.assertEqual([type(c).__name__ for c in caches], ['WeakrefLRUCache'] * 2)
        held = jax.jit(lambda x: j.sin(x) * 2)   # a long-lived jitted function, traced inside another program
        jax.jit(lambda x: held(x) + 1).lower(j.ones(3, j.float32)).compile()
        self.assertGreater(sum(c.cache_info().currsize for c in caches), 0)
        self.assertGreater(trace_release.release(), 0)
        self.assertEqual(sum(c.cache_info().currsize for c in caches), 0)

    def test_a_moved_cache_fails_loudly(self):
        from gpu_engine import trace_release
        with patch.object(trace_release, 'CACHES', (('jax._src.pjit', 'no_such_cache'),)), patch.object(trace_release, '_found', None):
            with self.assertRaisesRegex(RuntimeError, r'jax\._src\.pjit\.no_such_cache not found in jax .* \(verified on 0\.12\.0\)'):
                trace_release.release()
        with patch.object(trace_release, 'CACHES', (('jax._src.pjit', 'jit_lower'),)), patch.object(trace_release, '_found', None):
            with self.assertRaisesRegex(RuntimeError, 'has no cache_clear'):trace_release.release()


if __name__ == '__main__':
    unittest.main()
