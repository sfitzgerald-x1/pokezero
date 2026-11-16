"""gpu_engine.depthn_batching on the fake engine: the pipelined server answers as the serial search (prepare in real worker
processes, gpu_engine.prepare_pool, with Codex's real ProtocolConstructor.prepare under the fake harness), the flush rule,
request_id and order, a worker crash failing closed, a drain on close, serving after warmup compiling nothing at any batch
size, batches forming under concurrent load with the device never more than one batch ahead of reduce (backpressure), a root
that cannot be priced in time refused before the device, and a battle homed in process moving to a worker under load. CPU
only."""
import threading
import time
from types import SimpleNamespace as NS
import unittest

from unittest.mock import patch

from gpu_engine import constructor_worlds as cw, depthn_batching as batching, depthn_live_search as live, prepare_pool as pp
from tests import test_gpu_engine_depthn_live_search as L
from tests import test_gpu_engine_prepare_pool as PP
from tests.test_gpu_engine_trace_release import CompileEvents

STRIP = ('timings', 'decision_seconds', 'pipeline')
BATCH = ('batch_size', 'bucket')


def request(battle, decision, **kw):
    """The live-search tests' request with a 60 s budget by default: these tests are about the pipeline, not deadlines (a first
    price compiles its programs, seconds on a loaded CPU); the tests about deadlines pass their own."""
    return L.request(battle, decision, **{'budget': 60., **kw})


def strip(row, extra=()):
    return {k: v for k, v in row.items() if k not in STRIP + tuple(extra)}


KEYS = [('a', 0), ('b', 0), ('a', 1), ('c', 0), ('b', 1), ('a', 2), ('d', 0), ('e', 0), ('c', 1), ('f', 0), ('g', 0), ('h', 0)]
PAYLOAD = dict(W=8, outcomes=[(k, ('outcome', i, (('refused', (2,)),) if k == ('d', 0) else ())) for i, k in enumerate(KEYS) if k != ('e', 0)]
               + [(('e', 0), ('refusal', 'PUBLIC_ROOT_REFUSED'))] + [((f'w{i}', 0), ('outcome', 20 + i, ())) for i in range(2)]
               + [((f'l{i}', d), ('outcome', 30 + i, ())) for i in range(4) for d in range(4)] + [(('x', 0), ('outcome', 40, ()))],
               crash=[('x', 0)])


class Collector:
    """done callbacks: every response by request_id, in the order they came."""
    def __init__(self):
        self.rows = {};self.order = [];self.lock = threading.Lock();self.event = threading.Event();self.want = 0

    def done(self, rid):
        def call(response):
            with self.lock:
                self.rows[rid] = response;self.order.append(rid)
                if len(self.rows) >= self.want:
                    self.event.set()
        return call

    def wait(self, n, timeout=120.):
        self.want = n
        if len(self.rows) < n:
            self.event.wait(timeout)
        return self.rows


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L.LiveSearchTests.setUpClass.__func__(cls)
        cls.pool = pp.PreparePool(PP.FACTORY, PAYLOAD, workers=2, startup_seconds=300.).start()

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    make_pricers = classmethod(L.LiveSearchTests.make_pricers.__func__)
    setUp, search, outcome = L.LiveSearchTests.setUp, L.LiveSearchTests.search, L.LiveSearchTests.outcome

    def pooled(self, buckets=(1,), **kw):
        self.pool.reset();s = self.search(pricers=self.make_pricers(8, buckets), max_requests=buckets[-1], **kw)
        if self.pool.local is not None:
            self.pool.attach_local(s.constructor)   # the class's pool: its in-process slot follows this test's search
        s.attach_pool(self.pool)
        return s

    def pipeline(self, search, **kw):
        p = batching.Pipeline(search, self.pool, **kw).start();self.addCleanup(lambda: p.threads and p.close(30))
        return p

    def serve(self, pipeline, requests, timeout=120.):
        c = Collector()
        for r in requests:
            pipeline.submit(dict(r), c.done(r['request_id']))
        rows = c.wait(len(requests), timeout);self.assertEqual(len(rows), len(requests), 'not every request was answered')
        return [rows[r['request_id']] for r in requests], c

    def test_the_pipeline_answers_as_the_serial_search(self):
        requests = [request(b, d) for b, d in KEYS[:9]]
        requests.append(dict(request('zz', 0), rung='DN2W16'))   # bad_request: another rung
        requests.append(dict(request('f', 0), **{live.ARRIVAL: time.monotonic() - 100.}))   # expired_in_queue
        serial = self.search(pricers=self.make_pricers(8, (1,)), max_requests=1);outcomes = PP.outcomes_of(self.cw, PAYLOAD);expected = []
        for r in requests:   # batch 1, in order, on one constructor: the serial server
            rows, _ = self.run_batch(serial, [r], outcomes);expected.append(rows[0])
        causes = [r['cause'] for r in expected]
        self.assertEqual(causes, [None] * 6 + ['constructor_refused', 'constructor_refused', None, 'bad_request', 'expired_in_queue'])
        p = self.pipeline(self.pooled());rows, _ = self.serve(p, requests);p.close(60)
        for got, want in zip(rows, expected):
            if want['cause'] == 'expired_in_queue':   # its text names the measured wait
                self.assertEqual((got['cause'], got['status'], got['action_index']), ('expired_in_queue', 'REFUSED', None));continue
            self.assertEqual(strip(got), strip(want), want['request_id'])
        priced = [r for r in rows if r['status'] == 'COMPLETE']
        self.assertTrue(all(r['pipeline']['worker'] in (0, 1) and r['pipeline']['shipped_bytes'] > 0 for r in priced))
        self.assertTrue(all(r['decision_seconds'] >= r['timings']['prepare_seconds'] for r in priced))
        # Batched: the same decisions (bucket and batch size are the batch's own) ON THE FAKE ENGINE ONLY: its P1 and evaluator are
        # bucket-invariant. A real champion's batch forward is not shown to be (the bucket-invariance check of the GPU dump decides).
        rows, c = self.serve(self.pipeline(self.pooled((1, 2, 4)), window_seconds=5.), requests)
        for got, want in zip(rows, expected):
            if want['cause'] == 'expired_in_queue':
                continue
            self.assertEqual(strip(got, BATCH + ('pricer',)), strip(want, BATCH + ('pricer',)), want['request_id'])
        self.assertGreater(max(r['batch_size'] for r in rows), 1)

    def run_batch(self, search, requests, outcomes):
        return L.LiveSearchTests.run_batch(self, search, requests, outcomes)

    def test_batch_one_is_prepared_in_process_and_a_battle_stays_where_it_is_homed(self):
        outcomes = PP.outcomes_of(self.cw, PAYLOAD);serial = self.search(pricers=self.make_pricers(8, (1, 2, 4)));expected = {}
        order = [('a', 0), ('a', 1), ('a', 2), ('b', 0), ('c', 0)]
        for b, d in order:   # the serial server, one request at a time
            rows, _ = self.run_batch(serial, [request(b, d)], outcomes);expected[f'{b}:{d}'] = rows[0]
        s = self.pooled((1, 2, 4));local = self.pool.attach_local(s.constructor).index;p = self.pipeline(s, window_seconds=0., local_battles=2)
        with L.Codex(outcomes):   # (the in-process slot prepares with this process's harness, as the serial search does)
            alone = [self.serve(p, [request('a', d)])[0][0] for d in (0, 1)]   # nothing else in flight: homed here
            together, _ = self.serve(p, [request('a', 2), request('b', 0), request('c', 0)])   # b, c arrive with a in flight
        self.assertEqual(self.pool.affinity['a'], local);self.assertTrue(all(self.pool.affinity[x] < self.pool.size for x in 'bc'))
        self.assertEqual([r['pipeline']['worker'] for r in alone + together], [local, local, local, self.pool.affinity['b'], self.pool.affinity['c']])
        self.assertEqual([r['pipeline']['shipped_bytes'] for r in alone], [0, 0])   # nothing shipped: the serial server's own prepare
        for row in alone + together:
            want = expected[row['request_id']]
            self.assertEqual(strip(row, BATCH + ('pricer',)), strip(want, BATCH + ('pricer',)), row['request_id'])

    def test_batches_form_under_load_and_the_device_never_runs_ahead_of_reduce(self):
        # The GPU's shape (pod 1): device time linear in the batch's roots, reduce the slower stage. Twelve battles at once; the
        # device starts a batch only while at most one earlier batch is unreduced (itself included: <= 2), so ready roots
        # gather into batches > 1 instead of queueing behind reduce; every decision is the serial search's.
        outcomes = PP.outcomes_of(self.cw, PAYLOAD)
        keys = [('a', 0), ('b', 0), ('c', 0), ('f', 0), ('g', 0), ('h', 0)] + [(f'l{i}', 0) for i in range(4)] + [('w0', 0), ('w1', 0)]
        serial = self.search(pricers=self.make_pricers(8, (1,)), max_requests=1);expected = {}
        for b, d in keys:
            rows, _ = self.run_batch(serial, [request(b, d)], outcomes);expected[f'{b}:{d}'] = rows[0]
        s = self.pooled((1, 2, 4));p = self.pipeline(s, window_seconds=0.);price, decide = s._price, s._decide;seen = []
        def slow_price(gated, bucket):
            seen.append((len(gated), p.unreduced));time.sleep(.02 * len(gated))
            return price(gated, bucket)
        def slow_decide(todo, prepared, priced):
            time.sleep(.15 * len(todo))
            return decide(todo, prepared, priced)
        s._price, s._decide = slow_price, slow_decide
        rows, _ = self.serve(p, [request(b, d, budget=1000.) for b, d in keys])
        self.assertGreater(max(n for n, _ in seen), 1);self.assertLessEqual(max(u for _, u in seen), 2)
        settle = time.monotonic() + 30   # the last batch's 'batch_done' reaches the scheduler just after its responses
        while (p.unreduced or p.batches < sum(p.stats['batches'].values())) and time.monotonic() < settle:
            time.sleep(.01)
        self.assertEqual(sum(p.stats['batches'].values()), p.batches);self.assertEqual(p.unreduced, 0)
        self.assertGreater(max(r['batch_size'] for r in rows), 1)
        for row in rows:
            want = expected[row['request_id']]
            self.assertEqual(strip(row, BATCH + ('pricer',)), strip(want, BATCH + ('pricer',)), row['request_id'])

    def test_a_battle_homed_here_moves_to_a_worker_under_load(self):
        outcomes = PP.outcomes_of(self.cw, PAYLOAD);serial = self.search(pricers=self.make_pricers(8, (1, 2, 4)));expected = {}
        for b, d in [('a', 0), ('a', 1), ('b', 0), ('a', 2)]:   # the serial server, one request at a time
            rows, _ = self.run_batch(serial, [request(b, d)], outcomes);expected[f'{b}:{d}'] = rows[0]
        s = self.pooled((1, 2, 4));local = self.pool.attach_local(s.constructor).index;p = self.pipeline(s, window_seconds=0., local_battles=1)
        with L.Codex(outcomes):
            alone = [self.serve(p, [request('a', d)])[0][0] for d in (0, 1)]   # nothing else in flight: homed here
            # b first (a new battle; the one in-process home is a's: a worker), then a while b is in flight: a moves, and the
            # worker folds it from scratch to the state the in-process slot had
            together, _ = self.serve(p, [request('b', 0), request('a', 2)])
        self.assertEqual([r['pipeline']['worker'] for r in alone], [local, local]);self.assertEqual(p.stats['migrated'], 1)
        self.assertLess(self.pool.affinity['a'], self.pool.size);self.assertEqual(together[1]['pipeline']['worker'], self.pool.affinity['a'])
        self.assertGreater(together[1]['pipeline']['shipped_bytes'], 0)
        for row in alone + together:
            want = expected[row['request_id']]
            self.assertEqual(strip(row, BATCH + ('pricer',)), strip(want, BATCH + ('pricer',)), row['request_id'])

    def test_a_slow_in_process_prepare_under_a_tight_budget_answers_as_serial(self):
        # one stream, a 2.0 s budget, the 1.0 s minimum, a 1.1 s in-process prepare: the serial search prices and decides it (it
        # checks the minimum only at the batch's start), and so does the pipeline (no EXPIRED_BEFORE_PRICE after the prepare)
        outcomes = PP.outcomes_of(self.cw, PAYLOAD);real = cw.prepare_roots
        def slow(*a, **k):
            time.sleep(1.1)
            return real(*a, **k)
        serial = self.search(pricers=self.make_pricers(8, (1,)), max_requests=1);req = request('a', 0, budget=2.)
        with patch.object(cw, 'prepare_roots', slow):
            want, _ = self.run_batch(serial, [dict(req, **{live.ARRIVAL: time.monotonic()})], outcomes)
            s = self.pooled((1,));local = self.pool.attach_local(s.constructor).index;p = self.pipeline(s, window_seconds=0., local_battles=1)
            with L.Codex(outcomes):
                got, _ = self.serve(p, [dict(req, **{live.ARRIVAL: time.monotonic()})])
        self.assertEqual((want[0]['status'], got[0]['status'], got[0]['pipeline']['worker']), ('COMPLETE', 'COMPLETE', local))
        self.assertEqual(strip(got[0], BATCH + ('pricer',)), strip(want[0], BATCH + ('pricer',)));self.assertEqual(p.stats['expired_before_price'], 0)

    def test_session_cap_eviction_through_the_pipeline(self):
        # session_cap 2 over four battles: the least recent battles' sessions go (their worker drops them), and a battle evicted
        # and seen again folds from scratch: every decision is the serial search's under the same cap
        outcomes = PP.outcomes_of(self.cw, PAYLOAD);order = [('a', 0), ('b', 0), ('c', 0), ('f', 0), ('a', 1), ('b', 1)]
        serial = self.search(pricers=self.make_pricers(8, (1,)), max_requests=1, session_cap=2);expected = {}
        for b, d in order:
            rows, _ = self.run_batch(serial, [request(b, d)], outcomes);expected[f'{b}:{d}'] = rows[0]
        s = self.pooled((1,), session_cap=2);p = self.pipeline(s, window_seconds=0.)
        rows = [self.serve(p, [request(b, d)])[0][0] for b, d in order]
        self.assertGreater(s.evicted, 0)
        for row in rows:
            self.assertEqual(strip(row, BATCH + ('pricer',)), strip(expected[row['request_id']], BATCH + ('pricer',)), row['request_id'])

    def test_reset_forgets_every_session_so_an_address_prepares_again(self):
        s = self.pooled((1,));p = self.pipeline(s, window_seconds=0.)
        first, _ = self.serve(p, [request('a', 0)]);p.reset(60)
        again, _ = self.serve(p, [request('a', 0)])   # without the reset: SEARCH_REQUEST_ALREADY_ATTEMPTED
        self.assertEqual((first[0]['status'], again[0]['status']), ('COMPLETE', 'COMPLETE'))
        self.assertEqual(strip(again[0], BATCH + ('pricer',)), strip(first[0], BATCH + ('pricer',)))

    def test_a_battle_moves_while_its_in_process_prepare_runs(self):
        # a1 prepares here, slowly; b0 arrives (a worker) and a2 arrives with both in flight: 'a' moves to a worker while a1's
        # prepare still runs here (its drop queued behind it); a2 folds from scratch there; every decision is the serial one
        outcomes = PP.outcomes_of(self.cw, PAYLOAD);serial = self.search(pricers=self.make_pricers(8, (1,)));expected = {}
        for b, d in [('a', 0), ('a', 1), ('b', 0), ('a', 2)]:
            rows, _ = self.run_batch(serial, [request(b, d)], outcomes);expected[f'{b}:{d}'] = rows[0]
        s = self.pooled((1,));local = self.pool.attach_local(s.constructor).index;p = self.pipeline(s, window_seconds=0., local_battles=1)
        real = cw.prepare_roots
        def slow(constructor, requests, **kw):
            if any(r.get('battle_id') == 'a' and r.get('decision_index') == 1 for r in requests):
                time.sleep(1.)
            return real(constructor, requests, **kw)
        with L.Codex(outcomes), patch.object(cw, 'prepare_roots', slow):
            alone = self.serve(p, [request('a', 0)])[0][0]
            together, _ = self.serve(p, [request('a', 1), request('b', 0), request('a', 2)])
        self.assertEqual([alone['pipeline']['worker'], together[0]['pipeline']['worker']], [local, local]);self.assertEqual(p.stats['migrated'], 1)
        self.assertNotEqual(together[2]['pipeline']['worker'], local)
        for row in [alone] + together:
            self.assertEqual(strip(row, BATCH + ('pricer',)), strip(expected[row['request_id']], BATCH + ('pricer',)), row['request_id'])

    def test_a_worker_crash_fails_only_its_request_and_close_drains(self):
        p = self.pipeline(self.pooled((1, 2)), window_seconds=0.)
        requests = [request('x', 0)] + [request(f'l{i}', 0) for i in range(3)]
        rows, _ = self.serve(p, requests)
        self.assertEqual((rows[0]['status'], rows[0]['cause']), ('REFUSED', 'server_error'))
        self.assertRegex(rows[0]['refusal'], r'^RuntimeError: prepare worker \d exited \(code 3\)$')
        self.assertEqual([r['status'] for r in rows[1:]], ['COMPLETE'] * 3)
        # close() answers everything accepted before it, then stops.
        c = Collector()
        for i in range(3):
            p.submit(request(f'l{i}', 1), c.done(f'l{i}:1'))
        p.close(60);self.assertEqual(sorted(c.rows), ['l0:1', 'l1:1', 'l2:1']);self.assertTrue(all(not t.is_alive() for t in p.threads))
        self.assertEqual([c.rows[k]['status'] for k in sorted(c.rows)], ['COMPLETE'] * 3)

    def test_serving_after_warmup_compiles_nothing_at_any_batch_size(self):
        s = self.pooled((1, 2, 4));report = s.warmup([request('w0', 0), request('w1', 0)])
        self.assertEqual(set(report['buckets']), {1, 2, 4});self.assertTrue(all(report['buckets'][b]['warm_stages'] for b in (1, 2, 4)))
        p = self.pipeline(s, window_seconds=5.);seen = []
        with CompileEvents() as events:
            for n in (1, 2, 3, 4):   # each a batch of n (all submitted at once, the window long): buckets 1, 2, 4, 4
                rows, _ = self.serve(p, [request(f'l{i}', n - 1) for i in range(n)])
                seen.append(sorted({(r['batch_size'], r['bucket'], r['status']) for r in rows}))
        self.assertEqual(seen, [[(1, 1, 'COMPLETE')], [(2, 2, 'COMPLETE')], [(3, 4, 'COMPLETE')], [(4, 4, 'COMPLETE')]])
        self.assertEqual(events.seen, [])
        self.assertTrue(all(r['pipeline']['prepare_compiles'] == 0 for r in rows))   # the workers too: warmed before serving


class ReplyForwardTests(unittest.TestCase):
    """reply_forward='device': the reply model's forward once per batch and seat, padded to the bucket, beside the price. Its
    equality to the per-root forward below holds for the fake engine's small model only: a real reply model's forward is NOT
    batch-invariant (CPU: every logit of a 16-world root differs from its 32-world batch's, 0 of 194 served roots equal), which
    is why per root is the default and device is opt-in."""
    setUpClass = classmethod(PipelineTests.setUpClass.__func__);tearDownClass = classmethod(PipelineTests.tearDownClass.__func__)
    make_pricers = PipelineTests.make_pricers;setUp, search = PipelineTests.setUp, PipelineTests.search
    pooled, pipeline, serve, run_batch = PipelineTests.pooled, PipelineTests.pipeline, PipelineTests.serve, PipelineTests.run_batch

    def reply(self, mode):
        from gpu_engine.depthn_reply_prior import ReplyPrior
        return ReplyPrior(self.executor, self.evaluator, L.PARAMS[0], mode=mode)

    def test_the_batched_reply_forward_answers_as_per_root_and_compiles_at_warmup(self):
        requests = [request(f'l{i}', 0) for i in range(3)] + [request('a', 0), request('b', 0)]
        for mode in ('jit', 'eager'):
            with self.subTest(mode=mode):
                serial = self.search(pricers=self.make_pricers(8, (1, 2, 4)), reply=self.reply(mode));expected = []
                for r in requests:
                    rows, _ = self.run_batch(serial, [r], PP.outcomes_of(self.cw, PAYLOAD));expected.append(rows[0])
                s = self.pooled((1, 2, 4), reply=self.reply(mode));s.reply_batches = s.buckets
                s.warmup([request('w0', 0), request('w1', 0)]);p = self.pipeline(s, window_seconds=5., reply_forward='device')
                with CompileEvents() as events:
                    rows, _ = self.serve(p, requests)
                p.close(60)
                self.assertEqual(events.seen, []);self.assertGreater(max(r['batch_size'] for r in rows), 1)
                for got, want in zip(rows, expected):
                    self.assertEqual(strip(got, BATCH + ('pricer', 'reply_prior')), strip(want, BATCH + ('pricer', 'reply_prior')), want['request_id'])
                    self.assertEqual({k: v for k, v in got['reply_prior'].items() if k != 'seconds'}, {k: v for k, v in want['reply_prior'].items() if k != 'seconds'})


class FlushRuleTests(unittest.TestCase):
    """Pipeline._maybe_flush alone: a pipeline's scheduler state, a fixed clock."""
    def pipeline(self, now, *, max_batch=4, window=.05, guard=.15, device=1., prepare=1.):
        p = batching.Pipeline.__new__(batching.Pipeline)
        p.search = NS(buckets=(1, 2, 4), min_price_seconds=1., _encodable=lambda row, n: row);p.max_batch = max_batch;p.window = window
        p.guard = guard;p.clock = lambda: now;p.unreduced = 0;p.responded = [];p._respond = lambda t, r: p.responded.append((t.id, r))
        p.estimates = batching.Estimates((1, 2, 4), seed={b: (device * b ** .5, 0.) for b in (1, 2, 4)}, prepare=prepare)
        p.ready = [];p.tasks = {};p.device_busy = False;p.closing = False;p.stats = dict(flush={}, expired_before_price=0);p.device_q = []
        p.device_q = NS(put=lambda batch: p.flushed.append(batch));p.flushed = []
        return p

    def task(self, p, i, *, deadline, ready_at=None, started=None, item=True, rid=None):
        t = batching.Task(i, {}, dict(request_id=rid or f'r{i}'), dict(timings=dict(queue_seconds=.1)), None);t.deadline = deadline;t.ready_at = ready_at;t.started = started
        t.item = object() if item else None;p.tasks[i] = t
        if item:
            p.ready.append(t)
        return t

    def test_full_batches_go_at_once_earliest_deadlines_first(self):
        p = self.pipeline(10.)
        for i, d in enumerate((14., 12., 13., 11., 15.)):
            self.task(p, i, deadline=d, ready_at=10.)
        self.assertIsNone(p._maybe_flush());self.assertEqual([[t.id for t in b] for b in p.flushed], [[3, 1, 2, 0]])
        self.assertEqual(p.stats['flush'], {'full': 1});self.assertTrue(p.device_busy);self.assertEqual([t.id for t in p.ready], [4])
        self.assertIsNone(p._maybe_flush())   # the device is busy: nothing more

    def test_nothing_coming_flushes_now_and_a_prepare_about_to_land_is_waited_for(self):
        p = self.pipeline(10.);self.task(p, 0, deadline=13., ready_at=10.)
        p._maybe_flush();self.assertEqual(p.stats['flush'], {'nothing_coming': 1})
        p = self.pipeline(10.);self.task(p, 0, deadline=13., ready_at=10.);self.task(p, 1, deadline=13., started=9.02, item=False)   # lands at 10.02
        wake = p._maybe_flush();self.assertEqual(p.flushed, []);self.assertIsNotNone(wake)
        p.clock = lambda: 10.06;p._maybe_flush();self.assertEqual(p.stats['flush'], {'window': 1})   # the window caps the wait
        p = self.pipeline(10.);self.task(p, 0, deadline=13., ready_at=10.);self.task(p, 1, deadline=13., started=9.5, item=False)   # lands at 10.5
        p._maybe_flush();self.assertEqual(p.stats['flush'], {'nothing_coming': 1})   # past the window: not worth waiting for

    def test_the_deadline_rule_flushes_before_the_oldest_runs_out(self):
        # bucket 2's expected device + reduce seconds: 1.41; guard 0.15: the latest start is deadline - 1.56
        p = self.pipeline(10., window=5.);self.task(p, 0, deadline=11.6, ready_at=9.9);self.task(p, 1, deadline=15., started=9.03, item=False)
        self.assertIsNotNone(p._maybe_flush());self.assertEqual(p.flushed, [])   # a root lands at 10.03, before the latest start (10.036)
        p.clock = lambda: 10.05;p._maybe_flush();self.assertEqual(p.stats['flush'], {'deadline': 1})

    def test_backpressure_one_batch_in_reduce_one_on_the_device(self):
        p = self.pipeline(10.);self.task(p, 0, deadline=13., ready_at=10.)
        p.unreduced = 2;self.assertIsNone(p._maybe_flush());self.assertEqual(p.flushed, [])   # one reducing, one priced and waiting
        p.unreduced = 1;p._maybe_flush();self.assertEqual(len(p.flushed), 1);self.assertEqual(p.unreduced, 2)

    def test_a_root_that_lost_its_margin_waiting_for_the_device_is_refused_before_it(self):
        # ready with 1.1 s left (over the 1.0 s minimum), 0.6 s left at the device start: it waited the margin away -> refused
        p = self.pipeline(10.);self.task(p, 0, deadline=10.6, ready_at=9.5, rid='late');self.task(p, 1, deadline=13., ready_at=10.)
        p._maybe_flush();self.assertEqual([[t.id for t in b] for b in p.flushed], [[1]]);self.assertEqual(p.stats['expired_before_price'], 1)
        (tid, row), = p.responded
        self.assertEqual((tid, row['cause']), (0, 'expired_in_queue'))
        self.assertEqual(row['refusal'], 'EXPIRED_BEFORE_PRICE: 0.600s of the budget left at the device start, under the 1.000s minimum to price')
        p = self.pipeline(10.);self.task(p, 0, deadline=10.2, ready_at=9.)   # ready with 1.2 s, 0.2 s at the device start
        self.assertIsNone(p._maybe_flush());self.assertEqual((p.flushed, p.device_busy, len(p.responded)), ([], False, 1))   # nothing left to price

    def test_a_root_that_came_out_of_its_prepare_under_the_minimum_is_priced_as_serial(self):
        # the serial search checks the minimum only at the batch's start: a root ready with 0.7 s (or 0.3 s) left is priced
        for deadline in (10.6, 10.2):
            p = self.pipeline(10.);self.task(p, 0, deadline=deadline, ready_at=9.9)
            p._maybe_flush();self.assertEqual(([[t.id for t in b] for b in p.flushed], p.responded), ([[0]], []))

    def test_one_request_id_once_per_batch(self):
        p = self.pipeline(10.)
        for i in range(4):
            self.task(p, i, deadline=12. + i, ready_at=10., rid='same' if i < 2 else None)
        p._maybe_flush();self.assertEqual([t.id for t in p.flushed[0]], [0, 2, 3]);self.assertEqual([t.id for t in p.ready], [1])

    def test_estimates(self):
        e = batching.Estimates((1, 2, 4), seed={1: (.5, .25)})
        self.assertEqual(e.device(1), .75);self.assertEqual(e.device(4), 3.)   # an unseen bucket: the nearest seen, scaled
        e.observe(4, price=1., reduce=.5);self.assertEqual(e.device(4), 1.5);e.observe(4, price=2.);self.assertAlmostEqual(e.price[4], 1.2)
        self.assertEqual(batching.Estimates((1,)).device(1), 1.)


if __name__ == '__main__':
    unittest.main()
