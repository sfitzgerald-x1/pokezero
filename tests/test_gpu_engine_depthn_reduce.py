"""The live search's reduce, vectorized: depthn_backups.backup_learner_max_fast against the reference backup_learner_max, bit
for bit, on synthetic depth-2 frames with the live pricer's shapes (k 9/9, 81 turn-1 cells, [81, 9, 9] leaves) and every
status the reference distinguishes. CPU, numpy only."""
from types import SimpleNamespace as NS
import unittest

import numpy as np

from gpu_engine import depthn_backups as backups


def frame(blocks, seed, *, expanded=.85, terminal=.05, error=.01, leaf_terminal=.05, leaf_error=.002, zeros=False, ties=False):
    """A depth-2 DepthNFrame stand-in: `blocks` roots x 2 worlds, the fields the backups read, statuses drawn per cell."""
    g = np.random.default_rng(seed);R = blocks
    pre = np.where(g.random((R, 2)) < .03, g.integers(-2, 3, (R, 2)), 0).astype(np.int32)
    requested = g.random((R, 2)) < .9
    t1_status = np.zeros((R, 2, 9, 9), np.int8);t1_value = g.uniform(-1, 1, (R, 2, 9, 9)).astype(np.float32)
    own_actions = np.full((R, 2, 9, 9, 9), -1, np.int32);opp_actions = np.full((R, 2, 9, 9, 9), -1, np.int32)
    for r in range(R):
        for w in range(2):
            n_own, n_opp = g.integers(1, 10), g.integers(1, 10)
            for a in range(n_own):
                for b in range(n_opp):
                    u = g.random()
                    t1_status[r, w, a, b] = 1 if u < terminal else 3 if u < terminal + error else 2 if u < terminal + error + expanded else 0
                    if t1_status[r, w, a, b] == 2:
                        ko, kp = g.integers(0, 10), g.integers(0, 10)   # 0 slots: the reference's None
                        own_actions[r, w, a, b, :ko] = g.permutation(9)[:ko];opp_actions[r, w, a, b, :kp] = g.permutation(9)[:kp]
                        if g.random() < .2:   # a gap in the slots (an action pruned): slots need not be a prefix
                            own_actions[r, w, a, b, g.integers(0, 9)] = -1
    for r in range(R):   # a requested opponent has an enumerated reply (WorldPrices' rule); a waiting one may have none
        for w in range(2):
            if requested[r, w] and not t1_status[r, w].any():
                t1_status[r, w, 0, 0] = 1
    leaf_status = np.where(g.random((R, 2, 81, 9, 9)) < leaf_terminal, 1, 4).astype(np.int8)
    leaf_status[g.random((R, 2, 81, 9, 9)) < leaf_error] = 3
    leaf_value = g.uniform(-1, 1, (R, 2, 81, 9, 9)).astype(np.float32)
    logits = g.normal(0, 2, (R, 2, 9, 9, 9)).astype(np.float32)
    if zeros:
        leaf_value[g.random(leaf_value.shape) < .2] = 0.;leaf_value[g.random(leaf_value.shape) < .1] = -0.
    if ties:
        leaf_value[..., 1] = leaf_value[..., 0];logits[..., 1] = logits[..., 0];leaf_value[:, :, :, 2, :] = leaf_value[:, :, :, 0, :]
    return NS(depth=2, k_own=(9,), k_opp=(9,), pre=pre, root_peer_requested=requested, root_peer_logits=g.normal(0, 2, (R, 2, 9)).astype(np.float32),
              t1_status=t1_status, t1_value=t1_value, t1_own_actions=own_actions, t1_opp_actions=opp_actions,
              t1_own_logits=g.normal(0, 2, (R, 2, 9, 9, 9)).astype(np.float32), t1_peer_logits=logits, nodes=(), leaf_status=leaf_status, leaf_value=leaf_value)


def same(a, b):
    """Equal WorldPrices tuples, floats compared bit for bit (-0.0 is not 0.0)."""
    if (a is None) != (b is None):
        return False
    if a is None:
        return True
    flat = lambda x: [x.opponent_requested, x.opponent_legal, [float(v).hex() for v in x.opponent_logits],
                      [[None if v is None else float(v).hex() for v in row] for row in x.joint_values]]
    return flat(a) == flat(b)


class FastBackupTests(unittest.TestCase):
    def check(self, f):
        for r in range(f.pre.shape[0]):
            outcome = []
            for fn in (backups.backup_learner_max, backups.backup_learner_max_fast):   # an exception is part of the outcome
                try:
                    outcome.append(('ok', fn(f, r)))
                except ValueError as error:
                    outcome.append(('error', str(error)))
            (kind, want), (kind2, got) = outcome
            self.assertEqual(kind, kind2, f'root {r}')
            if kind == 'error':
                self.assertEqual(want, got);continue
            self.assertEqual(len(want), len(got))
            for w, g in zip(want, got):
                self.assertTrue(same(w, g), f'root {r}')
        return f

    def test_bitwise_on_random_frames(self):
        for seed in range(12):
            self.check(frame(8, seed))

    def test_bitwise_on_every_status_mix(self):
        for seed, kw in enumerate((dict(expanded=1., terminal=0., error=0.), dict(expanded=0., terminal=.5, error=.5), dict(leaf_error=.3),
                                   dict(leaf_terminal=1.), dict(zeros=True), dict(ties=True), dict(expanded=.5, leaf_error=0.))):
            self.check(frame(4, 100 + seed, **kw))

    def test_an_invalid_surface_raises_as_the_reference(self):
        f = frame(2, 11);f.root_peer_requested[0, 0] = True;f.t1_status[0, 0] = 0   # requested, nothing enumerated: WorldPrices refuses
        self.check(f)
        with self.assertRaises(ValueError):
            backups.backup_learner_max_fast(f, 0)

    def test_a_refusal_anywhere_is_the_references(self):
        f = frame(2, 7, expanded=1., terminal=0., error=0., leaf_error=0.)
        f.leaf_status[0, 0, 5, :, :] = 3   # every leaf of one cell an error: that cell None, as the reference
        f.pre[1, 1] = 2   # a world without a price surface: None
        self.check(f)
        self.assertIsNone(backups.backup_learner_max_fast(f, 1)[1])

    def test_another_depth_is_the_reference(self):
        f = frame(1, 3);calls = []
        f.depth = 3;original = backups.backup
        backups.backup = lambda *a: calls.append(a) or 'reference'
        try:
            self.assertEqual(backups.backup_learner_max_fast(f, 0), 'reference')
        finally:
            backups.backup = original
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()


class HostStackTests(unittest.TestCase):
    """depthn_reply_prior.host_stack: jnp.stack per leaf, built on the host (numpy) and placed with one device_put."""
    def test_every_leaf_as_jnp_stack_makes_it(self):
        import jax
        import jax.numpy as jnp
        from gpu_engine.depthn_reply_prior import host_stack
        g = np.random.default_rng(0)
        def tree(i):   # strong float32, int8, bool, a weak float (a Python scalar's), a weak int, a nested pytree
            return dict(f=jnp.asarray(g.normal(size=(1, 3)).astype(np.float32)), i=jnp.asarray(g.integers(-5, 5, (2,)).astype(np.int8)),
                        b=jnp.asarray(g.random(4) < .5), weak=jnp.asarray(float(i)) * 2, weak_int=jnp.asarray(i) + 1,
                        nested=(jnp.zeros((2, 2), jnp.int32) + i, [jnp.full((1,), i, jnp.uint8)]))
        trees = [tree(i) for i in range(5)]
        want, got = jax.tree.map(lambda *x: jnp.stack(x), *trees), host_stack(trees)
        self.assertEqual(jax.tree.structure(want), jax.tree.structure(got))
        for w, x in zip(jax.tree.leaves(want), jax.tree.leaves(got)):
            self.assertEqual((w.dtype, w.shape, w.aval.weak_type, w.committed, set(w.devices())), (x.dtype, x.shape, x.aval.weak_type, x.committed, set(x.devices())))
            self.assertTrue(np.array_equal(np.asarray(w), np.asarray(x)))
        self.assertTrue(any(x.aval.weak_type for x in jax.tree.leaves(got)), 'the fixture has weak leaves')

    def test_a_column_that_differs_in_dtype_keeps_jnp_stack(self):
        import jax
        import jax.numpy as jnp
        from gpu_engine.depthn_reply_prior import host_stack
        trees = [dict(x=jnp.zeros((2,), jnp.float32)), dict(x=jnp.zeros((2,), jnp.int32))]
        want, got = jax.tree.map(lambda *x: jnp.stack(x), *trees), host_stack(trees)
        self.assertEqual((got['x'].dtype, got['x'].aval.weak_type), (want['x'].dtype, want['x'].aval.weak_type))
        with self.assertRaises(ValueError):
            host_stack([dict(x=jnp.zeros(2)), dict(y=jnp.zeros(2))])


class ReplyStackTests(unittest.TestCase):
    """ReplyPrior.jit_logits on host-stacked worlds: the same compiled executable (no compile after jnp.stack's inputs compiled
    it) and the same bits, on the fake engine's prepared worlds."""
    from tests import test_gpu_engine_depthn_live_search as L
    make_pricers = classmethod(L.LiveSearchTests.make_pricers.__func__)

    @classmethod
    def setUpClass(cls):
        cls.L.LiveSearchTests.setUpClass.__func__(cls)

    def test_the_jitted_reply_program_is_unchanged(self):
        import jax
        import jax.numpy as jnp
        from gpu_engine import depthn_live_search as live
        from gpu_engine.depthn_reply_prior import ReplyPrior
        from tests.test_gpu_engine_trace_release import CompileEvents
        L = self.L;L.LiveSearchTests.setUp(self)
        s = L.LiveSearchTests.search(self, 8, pricers=self.make_pricers(8, (1,)));outcomes = {('h', 0): L.LiveSearchTests.outcome(self, 0, 8)}
        with L.Codex(outcomes):
            req = L.request('h', 0);p = {k: req[k] for k in live.PROTOCOL_KEYS};row = s._response(req['request_id'], 1)
            item = s._prepare([(0, p, row)], {})[0]
        reply = ReplyPrior(self.executor, self.evaluator, L.PARAMS[0], mode='jit');worlds = item.made.worlds;owner = int(item.detached.address.owner)
        reply.jit_logits(worlds, owner)   # builds and compiles the seat's program (host-stacked inputs)
        stack = lambda xs: jax.tree.map(lambda *leaves: jnp.stack(leaves), *xs)
        states, publics = jax.block_until_ready((stack([w.state for w in worlds]), stack([w.public for w in worlds])))   # the previous code's inputs
        with CompileEvents() as events:   # they reach the seat's program without a trace, a lowering or a compile: the same avals and executable
            old = np.asarray(jax.device_get(reply._jitted[owner](reply.params, states, publics)), np.float32)
            new = reply.jit_logits(worlds, owner)
        self.assertEqual(events.seen, []);self.assertEqual(old.tobytes(), new.tobytes())
        fetched = np.asarray(jax.device_get(reply.jit_dispatch(worlds, owner)), np.float32)   # dispatched, fetched later: the same bits
        self.assertEqual(fetched.tobytes(), new.tobytes())
