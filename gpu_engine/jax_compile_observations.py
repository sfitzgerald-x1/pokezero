"""Optional literal JAX0.12 monitoring, never compiler-only attribution.

No executable/config/cache changes. Missing/unmatched records remain UNKNOWN;
backend spans enclose compile_or_get_cached (including lookup/load).
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib.metadata
import math
from pathlib import Path
import threading
import time

SPAN_EVENTS={
 '/jax/core/compile/jaxpr_trace_duration':'jaxpr_trace',
 '/jax/core/compile/jaxpr_to_mlir_module_duration':'jaxpr_to_mlir',
 '/jax/core/compile/backend_compile_duration':'compile_or_get_cached_service'}
SOURCE_PINS={
 '_src/pjit.py':'4cbac41ed693853ca871599ba3a6b0d9db9c21998f93facb4066442dc1ac8a62',
 '_src/dispatch.py':'fe674386831eba96b681afe939accc2aabe1759ea1671189e98996b089d6e514',
 '_src/interpreters/pxla.py':'78aa89c3a410591ff240bbb46e0cf7ab0b7c76773a9430fbfaf1fa9befc07986',
 '_src/compiler.py':'2fd500e123543f69cd21540a87cd0a34c014ff9dd7a6cec71b2467235d2cef25',
 '_src/monitoring.py':'39c5d1a7c99a4497b08190e828b13c0fef6d1d99e320d5c0fe6df97b09bdb8f6',
 'monitoring.py':'8a4d0fb11e2b6e7d4db30138209fa2e9ba60995e2e797b9a2df0256a050c74d2'}
_CURRENT=ContextVar('pokezero_compile_observation',default=None)
_PHASE=ContextVar('pokezero_compile_phase',default=None)


def _runtime():
 import jax
 from jax import monitoring
 root=Path(jax.__file__).parent
 hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest()for name in SOURCE_PINS}
 jaxlib_version=importlib.metadata.version('jaxlib')
 methods=('register_event_time_span_listener','unregister_event_time_span_listener','register_event_listener','unregister_event_listener','register_event_duration_secs_listener','unregister_event_duration_listener')
 supported=jax.__version__==jaxlib_version=='0.12.0' and hashes==SOURCE_PINS and all(callable(getattr(monitoring,n,None))for n in methods)
 return monitoring,dict(jax_version=jax.__version__,jaxlib_version=jaxlib_version,source_sha256=hashes,supported=supported)


@contextmanager
def phase(name):
 observer=_CURRENT.get()
 if observer is None or not observer.available:yield;return
 token=_PHASE.set((observer,str(name),time.time(),threading.get_ident()))
 try:yield
 finally:_PHASE.reset(token)


def sample_step(step):
 observer=_CURRENT.get()
 if observer is not None:observer.step=None if step is None else int(step)


class Observer:
 def __init__(self,*,capacity=4096):
  if type(capacity)is not int or not 1<=capacity<=16384:raise ValueError('bounded compile observation capacity required')
  self.capacity=capacity;self.records=[];self.counts={};self.seen=0;self.dropped=0;self.errors=0;self.duplicate_duration_notifications=0
  self.available=False;self.closed=False;self.step=None;self.binding={};self.token=None;self.registered=[];self.lock=threading.Lock()
 def start(self):
  if self.closed or self.token is not None:raise ValueError('compile observer starts exactly once')
  try:
   if _CURRENT.get()is not None:self.binding={'supported':False,'reason':'NESTED_OBSERVATION_UNKNOWN'};return self
   self.monitor,self.binding=_runtime()
   if not self.binding['supported']:self.binding['reason']='VERSION_OR_SOURCE_OR_API_UNKNOWN';return self
   for register,unregister,callback in ((self.monitor.register_event_time_span_listener,self.monitor.unregister_event_time_span_listener,self.span),(self.monitor.register_event_listener,self.monitor.unregister_event_listener,self.event),(self.monitor.register_event_duration_secs_listener,self.monitor.unregister_event_duration_listener,self.duration)):
    register(callback);self.registered.append((unregister,callback))
   self.available=True;self.token=_CURRENT.set(self)
  except Exception as error:
   self.errors+=1;self.binding.update(supported=False,reason='INSTALL_UNAVAILABLE',error_type=type(error).__name__);self.close()
  return self
 def close(self):
  if self.closed:return
  if self.token is not None:_CURRENT.reset(self.token);self.token=None
  self.step=None;self.available=False
  for unregister,callback in reversed(self.registered):
   try:unregister(callback)
   except Exception:self.errors+=1
  self.registered=[];self.closed=True
 def _append(self,record):
  with self.lock:
   self.seen+=1
   key=record['sample_step'];counts=self.counts.setdefault(key,[0,0,0]);counts[0]+=1;counts[1]+=int(record['matched']);counts[2]+=int(not record['matched'])
   if len(self.records)<self.capacity:self.records.append(dict(index=self.seen-1,**record))
   else:self.dropped+=1
 def _where(self,start=None):
  current=_CURRENT.get();p=_PHASE.get();same=current is self
  step=self.step if same else None
  if p is None or p[0]is not self:return dict(phase=None,sample_step=step,matched=False,reason='NO_CURRENT_PHASE_OR_THREAD')
  if p[3]!=threading.get_ident():return dict(phase=None,sample_step=None,matched=False,reason='THREAD_UNKNOWN')
  if start is not None and start<p[2]:return dict(phase=None,sample_step=step,matched=False,reason='SPAN_STARTED_BEFORE_CURRENT_CALL')
  return dict(phase=p[1],sample_step=step,matched=True,reason=None)
 def span(self,event,start,end,**metadata):
  if not isinstance(event,str)or not event.startswith(('/jax/core/compile/','/jax/compilation_cache/')):return
  try:
   valid=type(start)in(int,float)and type(end)in(int,float)and math.isfinite(start)and math.isfinite(end)and end>=start
   where=self._where(start if valid else None)
   if not valid:where.update(matched=False,phase=None,reason='INVALID_LITERAL_SPAN')
   if event not in SPAN_EVENTS:where.update(matched=False,phase=None,reason='UNRECOGNIZED_SPAN_EVENT')
   self._append(dict(kind='span',event=event,service=SPAN_EVENTS.get(event),start=start if valid else None,end=end if valid else None,duration_secs=end-start if valid else None,fun_name=metadata.get('fun_name')if type(metadata.get('fun_name'))is str else None,**where))
  except Exception:self.errors+=1
 def event(self,event,**metadata):
  if not isinstance(event,str)or not event.startswith('/jax/compilation_cache/'):return
  try:self._append(dict(kind='cache_event',event=event,**self._where()))
  except Exception:self.errors+=1
 def duration(self,event,duration,**metadata):
  if not isinstance(event,str):return
  if event in SPAN_EVENTS:
   with self.lock:self.duplicate_duration_notifications+=1
   return # same JAX context also emits the authoritative time-span record
  if not event.startswith('/jax/compilation_cache/'):return
  try:
   valid=type(duration)in(int,float)and math.isfinite(duration)and duration>=0
   where=self._where()
   if not valid:where.update(matched=False,phase=None,reason='INVALID_LITERAL_DURATION')
   self._append(dict(kind='cache_duration',event=event,reported_duration_secs=duration if valid else None,**where))
  except Exception:self.errors+=1
 def snapshot(self,*,samples,warmup,steps):
  records=list(self.records);window=range(warmup,warmup+steps)
  fixed=[int(s['step'])for s in samples if s['step']in window]
  observed=[r for r in records if r['sample_step']in window]
  totals=[sum(v[i]for v in self.counts.values())for i in range(3)]
  fixed_counts=[sum(v[i]for step,v in self.counts.items()if step in window)for i in range(3)]
  return dict(schema='pokezero.literal-jax-compile-observations.v1',runtime=self.binding,closed=self.closed,
   status=('STRUCTURED_EVENTS_OBSERVED'if records else'NO_EVENTS_OBSERVED_COMPILER_UNKNOWN')if self.binding.get('supported')and not self.errors and not self.dropped else'INCOMPLETE_OR_UNAVAILABLE_UNKNOWN',
   event_count=self.seen if self.binding.get('supported')else None,stored_event_count=len(records),dropped_event_count=self.dropped,observer_errors=self.errors,
   matched_event_count=totals[1]if self.binding.get('supported')else None,unmatched_event_count=totals[2]if self.binding.get('supported')else None,
   duplicate_duration_notifications=self.duplicate_duration_notifications,records=records,event_count_scope='Canonical compile time-span records plus cache notifications/durations; sibling compile-duration notifications are separate, never additional compiles.',
   pure_compiler_only_ms=None,pure_compiler_only_status='UNMEASURED: backend service includes compile_or_get_cached lookup/load',
   fixed_declared_warm_window=dict(start=warmup,end_exclusive=warmup+steps,declared_samples=steps,recorded_steps=fixed,recorded_samples=len(fixed),
    event_count=fixed_counts[0]if self.binding.get('supported')else None,matched_events=fixed_counts[1]if self.binding.get('supported')else None,unmatched_events=fixed_counts[2]if self.binding.get('supported')else None,
    unassigned_step_events=self.counts.get(None,[0,0,0])[0],activity_status='UNKNOWN_UNASSIGNED_OR_DROPPED'if self.errors or self.dropped or self.counts.get(None,[0,0,0])[0]>0 or not self.binding.get('supported')else'EVENTS_OBSERVED'if observed else'NO_EVENTS_OBSERVED_NOT_PROOF_ZERO',sample_selection='Original fixed window, unchanged by observed compilation/cache events.'),
   limits=['Nested/overlapping event spans are not disjoint and must not be summed/subtracted as total compiler time.','Backend event is compile_or_get_cached SERVICE wall, including lookup/load. Cache events are literal phase-level notifications; no inferred cache-hit association.','Zero observed events does not prove zero compiler time; missing/version/thread/overflow gaps remain UNKNOWN.','No JIT/executable/compile/config/barrier/RNG/action/admission change; existing coldcall/explicit-compile metrics keep their original scopes.'])
