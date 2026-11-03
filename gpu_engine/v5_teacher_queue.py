"""Bounded FIFO label work; only the caller publishes original chunk records.

The consumer retains opaque device roots and verified archive references. It
never transfers a whole root to the host, touches the journal, or chooses an
action. All mutable provider state belongs to its single consumer thread.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time


class LabelQueue:
    def __init__(self,provider,*,capacity,check_progress,publish):
        if type(capacity)is not int or not 1<=capacity<=4:
            raise ValueError('bounded label queue capacity from1to4 required')
        self.provider=provider;self.capacity=capacity;self.check_progress=check_progress;self.publish=publish
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='device-labels')
        self.pending=deque();self.closed=False;self.submitted=0;self.maximum_pending=0;self.wait_seconds=0.

    def _label(self,roots,host,bank,params):
        self.provider.reset_metrics();clock=time.perf_counter()
        records=self.provider.label_chunk(roots,host.frames,bank,params)
        return records,self.provider.metrics(),time.perf_counter()-clock

    def check(self):
        self.check_progress()
        # Surface even a later failed task without publishing out of order.
        for _,future in self.pending:
            if future.done() and not future.cancelled():
                if future.exception() is not None:future.result()

    def _take(self):
        host,future=self.pending[0];clock=time.perf_counter()
        while True:
            self.check()
            try:result=future.result(timeout=.1);break
            except TimeoutError:
                if future.done():raise
                continue
        self.wait_seconds+=time.perf_counter()-clock
        self.publish(host,*result);self.pending.popleft()

    def submit(self,roots,host,bank,params):
        if self.closed:raise ValueError('closed label queue')
        self.make_room()
        self.pending.append((host,self.executor.submit(self._label,roots,host,bank,params)))
        self.submitted+=1;self.maximum_pending=max(self.maximum_pending,len(self.pending))

    def make_room(self):
        self.check()
        while len(self.pending)>=self.capacity:self._take()

    def drain(self):
        while self.pending:self._take()

    def close(self,*,failed=False):
        if self.closed:return
        self.closed=True
        for _,future in self.pending:future.cancel()
        # On failure no publication, cache merge or training may follow. The
        # original external owned-process stall/backstop policy remains live.
        self.executor.shutdown(wait=not failed,cancel_futures=True)

    def report(self):
        return dict(capacity=self.capacity,chunks=self.submitted,maximum_pending=self.maximum_pending,
            parent_wait_seconds=self.wait_seconds,publication='ORIGINAL_CHUNK_FIFO_PARENT_ONLY',
            roots='OPAQUE_DEVICE_REFERENCES_NO_WHOLE_ROOT_HOST_COPY',
            timing_scope='Consumer label wall overlaps acting; parent wait is a subset, not additive. Original full iteration wall remains the denominator.')
