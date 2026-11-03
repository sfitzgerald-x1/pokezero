"""Bounded CPU cache assembly and next-bank write overlap.

Workers run the original per-game adapter/writer; the original merge preserves
frame,row,owner order. Only the owner publishes journal events. No policy lag.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


class CpuWorkers:
    def __init__(self,count):
        cores=sorted(os.sched_getaffinity(0))if hasattr(os,'sched_getaffinity')else list(range(os.cpu_count()or 1))
        if type(count)is not int or not 1<=count<=len(cores)-1:
            raise ValueError('cache workers need dedicated available cores plus an owner core')
        self.cores=cores[-count:];self.executor=ThreadPoolExecutor(max_workers=count,thread_name_prefix='cache-cpu')
        self.local=threading.local();self.processes=[];self.lock=threading.Lock();self.closed=False

    def _request(self,request):
        started=time.perf_counter()
        if not hasattr(self.local,'process'):
            with self.lock:
                if self.closed:raise RuntimeError('cache workers closed')
                core=self.cores[len(self.processes)]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('v5_cache_worker.py')),'--core',str(core)],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,env=env)
                self.processes.append(process);self.local.process=process
        process=self.local.process
        process.stdin.write(json.dumps(request,allow_nan=False)+'\n');process.stdin.flush()
        line=process.stdout.readline()
        if not line:raise RuntimeError('CPU cache worker exited '+str(process.poll()))
        response=json.loads(line)
        if response.get('ok')is not True:raise RuntimeError('CPU cache assembly: '+response.get('error','invalid response'))
        return dict(response['result'],dispatch_wall_seconds=time.perf_counter()-started)

    def submit(self,request):
        if self.closed:raise RuntimeError('cache workers closed')
        return self.executor.submit(self._request,request)

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True;processes=tuple(self.processes)
        # A stopped study must not wait for queued data work, nor leave its own
        # workers alive. Children stay in the owner's process group as well.
        for process in processes:
            if process.poll()is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            process.stdin.close();process.stdout.close()
        self.executor.shutdown(wait=True,cancel_futures=True)


class BankAssembly:
    def __init__(self,workers,spool,*,bindings,loss_config,discount):
        if spool.bank.bindings!=bindings:raise ValueError('same original cache binding required')
        self.workers=workers;self.spool=spool;self.futures=[];self.rows=set()
        context=dict(store=str(spool.store.path),bank=asdict(spool.bank),bank_sha256=spool.bank_sha256,
            teacher_seed=None if spool.teacher_recipe is None else spool.teacher_recipe.seed,
            loss=asdict(loss_config),discount=discount)
        data=(json.dumps(context,sort_keys=True,allow_nan=False)+'\n').encode()
        path=spool.root/'cpu-context.json';path.write_bytes(data);path.chmod(0o444)
        self.reference=dict(context=str(path),sha256=hashlib.sha256(data).hexdigest())

    def submit(self,row,game):
        if row in self.rows:raise ValueError('cache row cannot be assembled twice')
        self.rows.add(row)
        self.futures.append(self.workers.submit(dict(self.reference,row=row,prefix=len(self.spool.records),
            records=list(self.spool.record_refs),has_training=bool(self.spool.has_training[row]),
            game=dict(stream_id=game.stream_id,battle_id=game.battle_id,seed=game.seed))))

    def check(self):
        for future in self.futures:
            if future.done():future.result()

    def finish(self,path):
        started=time.perf_counter();busy=0.;dispatched=0.
        for future in self.futures:
            result=future.result();busy+=result['seconds'];dispatched+=result.get('dispatch_wall_seconds',result['seconds'])
            if result['part']is not None:
                row,folder,digest=result['part'];self.spool.parts.append((row,Path(folder),digest))
        waited=time.perf_counter()-started;started=time.perf_counter()
        metadata=self.spool.merge(path)
        return dict(metadata=metadata,worker_seconds=busy,dispatch_seconds=dispatched,
            wait_seconds=waited,merge_seconds=time.perf_counter()-started)
