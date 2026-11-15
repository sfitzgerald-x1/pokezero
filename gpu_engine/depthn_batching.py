"""Batched, pipelined serving of the live DepthN search (scripts/depthn_live_server --prepare-workers N).

Design. One server process per GPU owns the device (champion, bucket pricers, reply prior); prepare runs in N worker processes
(gpu_engine.prepare_pool: battle affinity, sessions there). Three threads here, each a stage, so the host stages overlap the
device price of the current batch:
- scheduler (owns every piece of search state: the session LRU, the carry reference, the pool): admits each request as it
  arrives (bad_request at once), queues it on its battle's worker, and starts it there when that worker is free. At that start,
  and only then, expired_in_queue is decided (its queue wait: arrival -> prepare start, the live search's "batch start"), the
  battle is touched in the session LRU and battles beyond session_cap are evicted (each eviction is queued on its owning worker
  behind that battle's pending prepares). A prepared root (shipped back with its classes) is gated (refused classes, carry
  structure) here; one that cannot be priced (prepare refused it, a refused world, a gate) is finished at once without the
  device; the rest wait in `ready` for the device.
- device: build (views and root forward) and ONE price call on the smallest bucket holding the batch (DepthNLiveSearch._price),
  then, with reply_forward='device', the reply prior's forward for every priced root of the batch (one call per owner seat,
  padded to the bucket's world count). Then the batch goes to the reduce thread and the device takes the next batch.
- reduce: per root backups, the reply prior (its forward too, with reply_forward='reduce', the default: per root, exactly as
  the serial search), decide, the noise gate, the effect guard; the responses (DepthNLiveSearch._decide/_assemble), each
  checked by phase_d_search_server.encode, go to their requests' `done` callbacks.
So prepare(N+2) || build+price(N+1) || reduce(N).

Batch 1 pays nothing for batching (local_battles > 0, the server's default): a new battle that arrives with nothing else in
flight while the server runs about one decision at a time (the in-flight count arriving requests see, EWMA, under LOCAL_LOAD) is
homed in this process (the pool's in-process slot: this search's own constructor, the serial server's prepare, nothing shipped),
at most local_battles live there (a request within local_idle_seconds); any other new battle goes to a worker. A
battle never moves (its sessions, attempts, held roots and peer folds live where it is homed), except by the evicted battle's
path (session_cap, a dead worker), which folds it from scratch to the same state. So a single stream is served entirely in
process, by the serial server's own stage functions, and a loaded server prepares in the workers.

Flush rule (the device idle, R ready roots, each with deadline = arrival + budget_seconds): the batch is the up to max_batch
ready roots with the earliest deadlines; it goes when
  (a) R >= max_batch, or
  (b) now >= latest = min(deadline) - est(bucket(R + 1)) - guard_seconds: waiting longer would leave the oldest root less than
      the expected device and reduce time of a batch one root larger, or
  (c) now >= first ready + window_seconds (a latency cap on batching while the device idles), or
  (d) no prepare in flight is expected to finish (its start + the prepare EWMA) before min(latest, first ready + window):
      waiting would add no root.
est(b) = EWMA (alpha 0.2) of the measured build + price and reduce seconds of batches at bucket b, seeded from warmup's warm
pass at b (a bucket never seen: the nearest seen bucket's, scaled by b / that bucket); the prepare EWMA starts at 1 s.

Responses are the serial search's (the same stage functions); per response: queue_seconds = arrival -> prepare start,
prepare/classify/structure_seconds the root's own, build/price/reduce_seconds its batch's, decision_seconds = prepare start ->
its batch's decisions (today's definitions: the "batch start" of a request is its prepare start), batch_size the device batch's
size. One added field, `pipeline`: worker, ready_wait_seconds (ready -> its batch's device start), prepare_wall_seconds (prepare
start -> the prepared root back here, shipping included), unpack_seconds. A prepare worker's failure answers that request
server_error; an exception of a device batch answers each of its requests server_error (the serial search's rule for its
shared batch); MemoryError and non-Exception errors end the requests with that error (the server closes their connections, as
the serial path does).
"""
import itertools
import math
import queue
import threading
import time

from . import constructor_worlds as cw, depthn_live_search as live

REPLY_FORWARDS = ('reduce', 'device')
LOCAL_LOAD = .5   # the arriving requests' mean in-flight count under which a new battle may be homed in process


class Task:
    __slots__ = ('id', 'request', 'p', 'row', 'done', 'arrival', 'deadline', 'started', 'item', 'labels', 'gate', 'worker',
                 'sent', 'received', 'unpack_seconds', 'ready_at', 'compiles', 'bytes', 'split')

    def __init__(self, id, request, p, row, done):
        self.id, self.request, self.p, self.row, self.done = id, request, p, row, done
        self.arrival = request.get(live.ARRIVAL)
        self.deadline = self.started = self.item = self.labels = self.gate = self.worker = self.sent = self.received = None
        self.unpack_seconds = self.ready_at = self.compiles = self.bytes = self.split = None


class Estimates:
    """EWMA seconds: per bucket, a batch's build + price and its reduce; prepare (start -> back here)."""
    def __init__(self, buckets, seed=None, alpha=.2, prepare=1.):
        self.buckets, self.alpha, self.prepare = tuple(buckets), alpha, prepare
        self.price = {b: None for b in self.buckets};self.reduce = {b: None for b in self.buckets}
        for b, (price, reduce) in (seed or {}).items():
            if b in self.price:
                self.price[b], self.reduce[b] = price, reduce

    def _ewma(self, old, new):
        return new if old is None else old + self.alpha * (new - old)

    def observe(self, b, *, price=None, reduce=None):
        if price is not None:
            self.price[b] = self._ewma(self.price[b], price)
        if reduce is not None:
            self.reduce[b] = self._ewma(self.reduce[b], reduce)

    def observe_prepare(self, seconds):
        self.prepare = self._ewma(self.prepare, seconds)

    def device(self, b):
        """Expected build + price + reduce seconds of a batch at bucket b."""
        seen = [x for x in self.buckets if self.price[x] is not None]
        if not seen:
            return 1.
        if b in seen:
            return self.price[b] + (self.reduce[b] or 0.)
        near = min(seen, key=lambda x: (abs(x - b), x))
        return (self.price[near] + (self.reduce[near] or 0.)) * b / near


class Pipeline:
    def __init__(self, search, pool, *, max_batch=None, window_seconds=.05, guard_seconds=.15, reply_forward='reduce', estimates=None,
                 clock=None, log=None, local_battles=0, local_idle_seconds=60., split_worlds=0, split_min_worlds=32):
        if pool is None or getattr(search, 'pool', None) is not pool:
            raise ValueError('the search must prepare through this pool (DepthNLiveSearch.attach_pool)')
        max_batch = search.max_requests if max_batch is None else max_batch
        if type(max_batch) is not int or not 1 <= max_batch <= search.buckets[-1]:
            raise ValueError(f'max_batch in 1..{search.buckets[-1]} (the largest bucket) required')
        for name, value in (('window_seconds', window_seconds), ('guard_seconds', guard_seconds)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'finite {name} >= 0 required')
        if reply_forward not in REPLY_FORWARDS:
            raise ValueError(f'reply_forward in {REPLY_FORWARDS} required')
        if type(local_battles) is not int or local_battles < 0 or local_battles and getattr(pool, 'local', None) is None:
            raise ValueError('local_battles: an int >= 0, and > 0 only with the pool\'s in-process slot (PreparePool.attach_local)')
        self.local_battles, self.local_idle = local_battles, float(local_idle_seconds);self.activity = {};self.load = 0.
        # MEASUREMENT ONLY (gpu_engine.world_split; off by default, never in production before worlds' world-range seam): a lone
        # in-process decision of W >= split_min_worlds farms its world loop out to up to split_worlds idle workers.
        if type(split_worlds) is not int or split_worlds < 0 or split_worlds and not local_battles:
            raise ValueError('split_worlds: an int >= 0, and > 0 only with local_battles (it splits in-process decisions)')
        self.split_k, self.split_min = split_worlds, int(split_min_worlds);self.splits = {};self.split_backends = {}
        if split_worlds:
            pool.local_extra = lambda task: self.split_backends.pop(task, None)
        self.search, self.pool, self.max_batch = search, pool, max_batch
        self.window, self.guard, self.reply_forward = float(window_seconds), float(guard_seconds), reply_forward
        self.clock = clock or search.clock;self.log = log or (lambda *a: None)
        self.estimates = estimates or Estimates(search.buckets, seed=warm_seed(search))
        self.inbox = pool.events = queue.SimpleQueue()   # one queue: requests, the pool's events (redirected here), the device's and reduce's ends
        self.device_q, self.reduce_q = queue.SimpleQueue(), queue.SimpleQueue()
        self.tasks = {};self.ready = [];self.device_busy = False;self.closing = False;self.ids = itertools.count()
        self.threads = [];self.batches = 0;self.stats = dict(batches={}, flush={});self.resets = [];self.deferred = []

    # lifecycle ------------------------------------------------------------------------------------------------------------------
    def start(self):
        for name, target in (('depthn-scheduler', self._schedule), ('depthn-device', self._device), ('depthn-reduce', self._reduce)):
            t = threading.Thread(target=target, name=name, daemon=True);t.start();self.threads.append(t)
        return self

    def submit(self, request, done):
        """Queue one request (any thread). done(response) is called once, from a pipeline thread, with its response (or, for a
        fatal error, the exception)."""
        self.inbox.put(('request', request, done))

    def reset(self, timeout=None):
        """Forget every session (DepthNLiveSearch.reset_sessions, the workers' too) once nothing is in flight; blocks until done
        (a bench between rosters; requests submitted meanwhile wait until after it)."""
        done = threading.Event();box = []
        self.inbox.put(('reset', done, box))
        if not done.wait(timeout):
            raise TimeoutError('the pipeline did not reset')
        if box:
            raise box[0]

    def close(self, timeout=None):
        """Stop admitting, finish every accepted request, stop the threads (the pool is the caller's). Idempotent."""
        if self.threads and self.threads[0].is_alive() and not getattr(self, '_close_sent', False):
            self._close_sent = True;self.inbox.put(('close',))
        for t in self.threads:
            t.join(timeout)
        if not any(t.is_alive() for t in self.threads) and self.pool.events is self.inbox:
            self.pool.events = queue.SimpleQueue()   # the pool's own again: nothing of this pipeline's left on it

    # scheduler ------------------------------------------------------------------------------------------------------------------
    def _schedule(self):
        wake = None
        while True:
            if self.closing and not self.tasks and not self.device_busy:
                self.device_q.put(None);self.reduce_q.put(None);return
            timeout = None if wake is None else max(0., wake - time.monotonic())
            nxt = self.pool.watchdog()
            if nxt is not None:
                timeout = nxt - time.monotonic() if timeout is None else min(timeout, nxt - time.monotonic())
            try:
                event = self.inbox.get(timeout=None if timeout is None else max(.001, timeout))
            except queue.Empty:
                event = None
            try:
                if event is not None:
                    self._event(event)
                if self.resets and not self.tasks and not self.device_busy:   # nothing in flight: forget every session now
                    for done, box in self.resets:
                        try:
                            self.search.reset_sessions()
                        except Exception as error:   # noqa: BLE001
                            box.append(error)
                        done.set()
                    self.resets = []
                    deferred, self.deferred = self.deferred, []
                    for request, done in deferred:
                        self._admit(request, done)
                for result in self.pool.pump():
                    self._result(result)
                wake = self._maybe_flush()
            except Exception as error:   # noqa: BLE001 -- a scheduler fault fails what it touched, never the loop
                self.log('scheduler error', f'{type(error).__name__}: {error}')
                wake = None

    def _event(self, event):
        kind = event[0]
        if kind == 'request':
            self._admit(event[1], event[2])
        elif kind == 'pool':
            for result in self.pool.handle(event):
                self._result(result)
        elif kind == 'device':
            _, tasks, bucket, seconds = event;self.device_busy = False
            if bucket is not None:
                self.estimates.observe(bucket, price=seconds)
        elif kind == 'reduced':
            _, bucket, seconds = event
            if bucket is not None:
                self.estimates.observe(bucket, reduce=seconds)
        elif kind == 'finished':
            self.tasks.pop(event[1], None)
        elif kind == 'close':
            self.closing = True
        elif kind == 'reset':
            self.resets.append(event[1:])
        elif kind == 'split':
            self._split(*event[1:])

    def _admit(self, request, done):
        search = self.search
        if self.closing:
            return self._respond_now(None, done, RuntimeError('the server is stopping'))
        if self.resets:   # admitted after the reset
            return self.deferred.append((request, done))
        try:
            row, problem = search._admit(request, 1)
        except Exception as error:   # noqa: BLE001
            row, problem = search._response(None, 1), f'MALFORMED_REQUEST: {type(error).__name__}'
        if problem is not None:
            return self._respond_now(None, done, search._encodable(dict(row, cause='bad_request', refusal=problem), 1))
        p = {k: request[k] for k in live.PROTOCOL_KEYS};task = Task(next(self.ids), request, p, row, done)
        budget = float(request['budget_seconds']);arrival = task.arrival if live._real(task.arrival) else self.clock()
        task.deadline = arrival + budget;self.tasks[task.id] = task
        battle = p['battle_id'] if type(p['battle_id']) is str else None
        try:
            task.worker = self.pool.submit(task.id, p, battle, on_start=self._on_start, prefer_local=self._local_first(battle, task.id))
        except Exception as error:   # noqa: BLE001
            self._respond(task, search._encodable(search._failed(row, None, error), 1))

    def _local_first(self, battle, task_id):
        """Home a new battle in this process (the serial server's own prepare, no shipping) when nothing else is in flight, the
        server has been running about one decision at a time (the in-flight count seen by arriving requests, EWMA alpha 0.1, under
        LOCAL_LOAD), and fewer than local_battles battles live here (a request within local_idle_seconds); every other new battle
        goes to a worker. A battle stays where it was homed (its sessions live there), so its requests are prepared where its
        history is: one decision at a time, every battle is served in process; under load, new battles go to the workers."""
        now = time.monotonic();others = sum(1 for t in self.tasks if t != task_id)
        self.load += .1 * (others - self.load)
        if battle is not None:
            self.activity[battle] = now
        if not self.local_battles or battle is None or battle in self.pool.affinity or others or self.load >= LOCAL_LOAD:
            return False
        index = self.pool.local.index
        live = sum(1 for b, i in self.pool.affinity.items() if i == index and now - self.activity.get(b, -1e9) < self.local_idle)
        return live < self.local_battles

    def _on_start(self, task_id):
        """The pool is about to send this task's prepare: the request's start (the live search's batch start). Returns the
        response that ends it instead (expired_in_queue, or server_error should this step fail), else None."""
        task = self.tasks[task_id];search = self.search
        try:
            now = self.clock();expired = search._expired(task.request, task.row, now)
            if expired is not None:
                return expired
            task.started = now;task.sent = time.monotonic();search._touch([task.p]);search._evict()
            if (self.split_k and task.worker == self.pool.local.index and search.W >= self.split_min
                    and not any(t != task_id for t in self.tasks) and len(self.pool.idle_workers()) >= 2):
                from . import world_split
                backend = world_split.Backend(min(self.split_k, len(self.pool.idle_workers())), self._split_ranges(task_id))
                self.split_backends[task_id] = backend;task.split = backend
        except Exception as error:   # noqa: BLE001
            return search._failed(task.row, None, error)
        return None

    def _split_ranges(self, task_id):
        """A world_split.Backend's `ranges` for this task (called on the in-process slot's thread): the pickled ranges go to the
        scheduler, which sends each to an idle worker (or answers 'local': that range runs here); the replies in order."""
        def ranges(blobs):
            reply = queue.SimpleQueue();self.inbox.put(('split', task_id, blobs, reply));out = [None] * len(blobs)
            deadline = time.monotonic() + self.pool.timeout
            for _ in blobs:
                j, kind, payload = reply.get(timeout=max(.1, deadline - time.monotonic()))
                if kind == 'local':
                    from . import world_split
                    payload = world_split.worker_range(self.search.constructor, blobs[j])
                elif kind != 'worlds':
                    raise RuntimeError(f'world split: range {j} {kind}: {payload}')
                out[j] = payload
            return out
        return ranges

    def _split(self, task_id, blobs, reply):
        """Scheduler side of a split: each range to an idle worker, the rest answered 'local'."""
        idle = self.pool.idle_workers()
        for j, blob in enumerate(blobs):
            if idle:
                index = idle.pop(0);key = ('split', task_id, j);self.splits[key] = (reply, j);self.pool.send_worlds(index, key, blob)
            else:
                reply.put((j, 'local', None))

    def _result(self, result):
        if type(result.task) is tuple and result.task[:1] == ('split',):   # a world range's end: to the waiting split
            reply, j = self.splits.pop(result.task, (None, None))
            if reply is not None:
                reply.put((j, result.kind, result.item if result.kind == 'worlds' else result.text))
            return
        task = self.tasks.get(result.task)
        if task is None:
            return
        try:
            self._prepared(task, result)
        except Exception as error:   # noqa: BLE001 -- this request's server_error; the scheduler goes on
            self._respond(task, self.search._encodable(self.search._failed(task.row, task.labels, error), 1))

    def _prepared(self, task, result):
        search = self.search
        if result.kind == 'refused':
            return self._respond(task, search._encodable(result.veto, 1))
        if result.kind != 'prepared':
            return self._respond(task, search._encodable(search._failed(task.row, None, live.RemoteError(result.text or result.kind)), 1))
        task.item, task.labels = result.item, dict(result.labels);task.received = time.monotonic();task.unpack_seconds = result.unpack_seconds
        task.compiles, task.bytes = result.compiles, result.bytes
        if task.started is not None:
            self.estimates.observe_prepare(max(0., self.clock() - task.started))
        classified = {0: task.labels}
        task.gate = search._gate(0, task.p, task.item, classified)
        full = type(task.item) is cw.Prepared and task.item.made.fully_constructed
        if task.gate is not None or not full:   # nothing to price: finished without the device
            return self.reduce_q.put(('single', task))
        task.ready_at = self.clock();self.ready.append(task)

    def _maybe_flush(self):
        """Hand the device a batch when the flush rule says so (module docstring); else the time to look again, or None."""
        if self.device_busy or not self.ready:
            return None
        now = self.clock();ready = sorted(self.ready, key=lambda t: (t.deadline, t.id))
        reason = None
        if len(ready) >= self.max_batch:
            reason = 'full'
        else:
            nxt = next((b for b in self.search.buckets if b >= len(ready) + 1), self.search.buckets[-1])
            latest = ready[0].deadline - self.estimates.device(nxt) - self.guard
            cap = min(t.ready_at for t in ready) + self.window;until = min(latest, cap)
            soon = [t.started + self.estimates.prepare for t in self.tasks.values()
                    if t.started is not None and t.item is None]
            if now >= latest:
                reason = 'deadline'
            elif now >= cap:
                reason = 'window'
            elif self.closing or not any(s <= until for s in soon):
                reason = 'nothing_coming'
            else:
                return time.monotonic() + max(0., until - now)
        batch, seen = [], set()
        for t in ready:   # one request_id once per batch (the stage functions key roots by it)
            if len(batch) < self.max_batch and t.p['request_id'] not in seen:
                batch.append(t);seen.add(t.p['request_id'])
        for t in batch:
            self.ready.remove(t)
        self.device_busy = True;self.stats['flush'][reason] = self.stats['flush'].get(reason, 0) + 1
        self.device_q.put(batch)
        return None

    # device ---------------------------------------------------------------------------------------------------------------------
    def _device(self):
        search = self.search
        while (batch := self.device_q.get()) is not None:
            started = self.clock();priced = error = None
            gated = [t.item for t in batch]
            try:
                priced = search._price(gated, None)
                if self.reply_forward == 'device' and getattr(search, 'reply', None) is not None:
                    priced['reply_logits'] = reply_logits(search, priced, {t.p['request_id']: t.item for t in batch})
            except BaseException as caught:   # noqa: BLE001 -- answered per request by the reduce thread
                error = caught
            seconds = None if priced is None else priced['build_seconds'] + priced['price_seconds']
            self.reduce_q.put(('batch', batch, priced, error, started))
            self.inbox.put(('device', batch, None if priced is None else priced['chosen'], seconds))

    # reduce ---------------------------------------------------------------------------------------------------------------------
    def _reduce(self):
        search = self.search
        while (job := self.reduce_q.get()) is not None:
            if job[0] == 'single':
                task = job[1];batch, error, started = [task], None, None
                try:
                    priced = search._price([live._gated(task.item) if task.gate is not None else task.item], None)
                except BaseException as caught:   # noqa: BLE001
                    priced, error = None, caught
            else:
                _, batch, priced, error, started = job
            n = len(batch);todo = [(k, t.p, dict(t.row, batch_size=n)) for k, t in enumerate(batch)];bucket = None
            try:
                if error is not None:
                    raise error
                results = search._decide(todo, [t.item for t in batch], priced);t_reduce = time.perf_counter();now = self.clock()
                out = search._assemble(todo, [t.item for t in batch], results, {k: dict(t.labels) for k, t in enumerate(batch)},
                                       {k: t.gate for k, t in enumerate(batch) if t.gate is not None}, priced, t_reduce,
                                       {k: now - t.started for k, t in enumerate(batch)})
                bucket = priced['chosen']
                if bucket is not None:
                    self.inbox.put(('reduced', bucket, t_reduce - priced['price_end']))
                for k, t in enumerate(batch):
                    row = dict(out[k], pipeline=dict(worker=t.worker, ready_wait_seconds=None if t.ready_at is None or started is None else started - t.ready_at,
                        prepare_wall_seconds=None if t.received is None or t.sent is None else t.received - t.sent, unpack_seconds=t.unpack_seconds,
                        prepare_compiles=t.compiles, shipped_bytes=t.bytes, **({} if t.split is None else dict(split=dict(t.split.parts)))))
                    self._respond(t, search._encodable(row, n))
            except live.FATAL as fatal:
                for t in batch:
                    self._respond(t, fatal)
            except Exception as caught:   # noqa: BLE001 -- the shared batch's rule: each request server_error
                for k, t in enumerate(batch):
                    self._respond(t, search._encodable(search._failed(todo[k][2], t.labels, caught), n))
            except BaseException as caught:
                for t in batch:
                    self._respond(t, caught)
            if job[0] == 'batch':
                self.batches += 1;self.stats['batches'][n] = self.stats['batches'].get(n, 0) + 1

    # responses ------------------------------------------------------------------------------------------------------------------
    def _respond(self, task, response):
        try:
            task.done(response)
        except Exception as error:   # noqa: BLE001 -- a callback's failure is the caller's; the pipeline goes on
            self.log('done callback error', f'{type(error).__name__}: {error}')
        self.inbox.put(('finished', task.id))

    def _respond_now(self, task, done, response):
        try:
            done(response)
        except Exception as error:   # noqa: BLE001
            self.log('done callback error', f'{type(error).__name__}: {error}')
        if task is not None:
            self.tasks.pop(task.id, None)


def warm_seed(search):
    """{bucket: (build + price seconds, reduce seconds)} from the search's warm report (warmup's warm pass per bucket), when it
    has them."""
    out = {}
    for b, row in ((getattr(search, 'warm', None) or {}).get('buckets') or {}).items():
        stages = row.get('warm_stages') if isinstance(row, dict) else None
        if stages:
            out[b] = (stages['build_seconds'] + stages['price_seconds'], stages['reduce_seconds'])
    return out


def reply_logits(search, priced, items):
    """{request_id: the reply model's logits [W, 9]} for every priced root of a batch: one forward per owner seat over the roots'
    worlds, padded (the first root's worlds repeated) to the batch bucket's world count, so each bucket compiles one program per
    seat (DepthNLiveSearch's reply prior; its mode's forward: eager views, or the jitted views)."""
    import numpy as np
    reply = search.reply;made = priced['made'];bucket = priced['chosen'];out = {}
    if bucket is None:
        return out
    seats = {}
    for root in made.roots:
        if root['status'] != 'PRICED':
            continue
        item = items[root['request_id']]
        seats.setdefault(int(item.detached.address.owner), []).append((root['request_id'], tuple(item.made.worlds)))
    for owner, roots in seats.items():
        flat = [w for _, worlds in roots for w in worlds];W = len(roots[0][1])
        flat += list(roots[0][1]) * (bucket - len(roots))
        logits = reply.jit_logits(flat, owner) if reply.mode == 'jit' else reply.logits(flat, owner)
        logits = np.asarray(logits)
        for k, (rid, worlds) in enumerate(roots):
            out[rid] = logits[k * W:(k + 1) * W]
    return out
