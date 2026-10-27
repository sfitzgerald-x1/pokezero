"""Attempt-bound comparator timers and durable witnesses, with no launch policy.

The operation ends after synchronized comparison and result/progress fsync.
Timing-witness publication is excluded, and its overhead is separately retained.
Missing DONE never proves how long the in-scope operation took.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import select
import stat
import time
import zlib

from .panel_cost import operation_budget,contract,summarize
from .collect_fidelity_panel import atomic_json,encoded

ENV_ATTEMPT='POKEZERO_PANEL_TIMING_ATTEMPT'
ENV_NOTIFY='POKEZERO_PANEL_TIMING_NOTIFY_FD'
SCHEMA='pokezero.comparator-slot-timing.v1'


class OperationBudgetExceeded(TimeoutError):
    pass


def notification_fd():
    text=os.environ.get(ENV_NOTIFY,'')
    if not text.isdigit():raise ValueError('owned live timing notification channel required')
    fd=int(text)
    if not stat.S_ISFIFO(os.fstat(fd).st_mode) or os.get_blocking(fd):
        raise ValueError('owned timing channel must be a nonblocking pipe')
    return fd


def notify_start(frame):
    """One bounded nonblocking pipe write BEFORE any in-scope filesystem I/O."""
    fd=notification_fd()
    # Compact fixed-order wire fits POSIX's minimum512-byte atomic pipe bound.
    payload=encoded([1,frame['attempt'],frame['worker_pid'],frame['entry_sha256'],
        frame['operation_budget_sha256'],frame['input_freeze_sha256'],
        frame['slot'],frame['start_monotonic']])+b'\n'
    if len(payload)>os.fpathconf(fd,'PC_PIPE_BUF'):
        raise ValueError('timing start exceeds atomic notification bound')
    if os.get_blocking(fd):raise ValueError('timing channel must be nonblocking')
    if os.write(fd,payload)!=len(payload):raise OSError('incomplete timing start notification')


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def sync_file(path):
    with path.open('rb') as stream:os.fsync(stream.fileno())


def exclusive_json(path,value):
    temporary=path.with_name(f'.{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp')
    with temporary.open('xb') as stream:
        stream.write(encoded(value));stream.flush();os.fsync(stream.fileno())
    # A hard link publishes complete bytes atomically and refuses an existing
    # final name. os.replace would silently overwrite an immutable witness.
    os.link(temporary,path)
    temporary.unlink()
    sync_directory(path.parent)


def binding(entry,attempt,pid):
    if (type(attempt) is not str or len(attempt)!=64 or any(c not in '0123456789abcdef' for c in attempt)
            or type(pid) is not int or pid<=0):
        raise ValueError('owned timing attempt nonce and exact child PID required')
    return dict(schema=SCHEMA,attempt=attempt,worker_pid=pid,entry_sha256=sha(entry),
        operation_budget_sha256=operation_budget(entry)[1])


class Session:
    def __init__(self,entry,panel):
        if 'cost_admission' in entry:contract(entry)
        self.entry=entry;self.panel=Path(panel)
        self.bound=binding(entry,os.environ.get(ENV_ATTEMPT),os.getpid())
        notification_fd()  # fail before device/source work when owner is absent
        self.budget_exceeded=False

    def begin(self,slot,frozen):
        if self.budget_exceeded:
            raise OperationBudgetExceeded('previous slot exceeded its operation budget; no continuation')
        if type(slot) is not int or not 0<=slot<self.entry['slots']:
            raise ValueError('bounded exact timing slot required')
        frame=dict(self.bound,slot=slot,input_freeze_sha256=sha(frozen),
            start_monotonic=time.monotonic(),status='STARTED')
        notify_start(frame)
        exclusive_json(self.panel/f'slot-{slot:04d}-timing-start.json',frame)
        atomic_json(self.panel/'timing-current.json',frame);sync_directory(self.panel)
        return frame

    def finish(self,frame,result,path,planned_cases):
        slot=frame['slot'];result_path=self.panel/f'slot-{slot:04d}.json.gz'
        result_sha=hashlib.sha256(result_path.read_bytes()).hexdigest()
        end=time.monotonic()
        self.budget_exceeded=end-frame['start_monotonic']>operation_budget(self.entry)[0]['limit_seconds']
        done=dict(frame,status='DONE',end_monotonic=end,seconds=end-frame['start_monotonic'],
            result_sha256=result_sha,path=path,planned_cases=planned_cases,
            completed_cases=result.get('exact_cases',result.get('completed_cases',0)),
            result_status=result['status'],timing_scope=operation_budget(self.entry)[0]['operation'])
        exclusive_json(self.panel/f'slot-{slot:04d}-timing-done.json',done)
        # This excludes publication of the overhead receipt itself; no
        # self-referential timestamp is presented as exact full runtime.
        published=time.monotonic()
        exclusive_json(self.panel/f'slot-{slot:04d}-timing-publication.json',
            dict(self.bound,slot=slot,done_sha256=sha(done),
                witness_publication_seconds=published-end,published_monotonic=published))
        if self.budget_exceeded:
            raise OperationBudgetExceeded('full slot exceeded the prospectively bound operation budget')


def _finite_number(value):
    return type(value) in (int,float) and math.isfinite(value)


def validate_start(frame,bound,frozen,slots,launch_monotonic,now):
    if (type(frame) is not dict or any(frame.get(k)!=v for k,v in bound.items())
            or frame.get('status')!='STARTED' or type(frame.get('slot')) is not int
            or not 0<=frame['slot']<slots or frame.get('input_freeze_sha256')!=sha(frozen)
            or not _finite_number(frame.get('start_monotonic'))
            or not launch_monotonic<=frame['start_monotonic']<=now):
        raise ValueError('invalid timing START identity or absolute monotonic time')
    return frame


def validate_done_header(done,start,entry,now):
    fixed={k:v for k,v in start.items() if k!='status'}
    if (type(done) is not dict or any(done.get(k)!=v for k,v in fixed.items())
            or done.get('status')!='DONE' or not _finite_number(done.get('end_monotonic'))
            or not start['start_monotonic']<=done['end_monotonic']<=now
            or not _finite_number(done.get('seconds'))
            or done['seconds']!=done['end_monotonic']-start['start_monotonic']
            or done.get('timing_scope')!=operation_budget(entry)[0]['operation']
            or type(done.get('result_sha256')) is not str or len(done['result_sha256'])!=64):
        raise ValueError('invalid timing DONE binding, duration or result digest')
    return done


def validate_publication(pub,done,bound,slot,now):
    if (type(pub) is not dict or any(pub.get(k)!=v for k,v in bound.items()) or pub.get('slot')!=slot
            or pub.get('done_sha256')!=sha(done) or not _finite_number(pub.get('published_monotonic'))
            or not done['end_monotonic']<=pub['published_monotonic']<=now
            or pub.get('witness_publication_seconds')!=pub['published_monotonic']-done['end_monotonic']):
        raise ValueError('invalid separately retained witness overhead')
    return pub


def validate_done(done,start,result_path,entry,now,*,root_record=None,constructor_context=None):
    from .panel_owner import validate_slot
    import gzip
    validate_done_header(done,start,entry,now)
    if done['result_sha256']!=hashlib.sha256(result_path.read_bytes()).hexdigest():
        raise ValueError('full result differs from timing DONE digest')
    with gzip.open(result_path,'rt') as stream:result=json.load(stream)
    frozen=json.loads((result_path.parent/'inputs-frozen.json').read_bytes())
    validate_slot(result,entry,frozen['input_jsonl_sha256'],start['slot'],
        root_record=root_record,constructor_context=constructor_context)
    certificate=result.get('consumer_certificate')
    expected=('ordinary_carry' if certificate.get('kind')=='ordinary_carry' else
        'conditional' if 'native_cases' in certificate else 'ordinary') if certificate else (
        'source_failure' if result.get('category') in {'SOURCE_CONSTRUCTOR_FAILURE','SOURCE_REQUEST_FAILURE',
            'TERMINAL_SOURCE_ROOT'} else 'refused')
    if (done.get('path')!=expected or done.get('result_status')!=result['status']
            or type(done.get('planned_cases')) is not int or done['planned_cases']<0
            or type(done.get('completed_cases')) is not int
            or not 0<=done['completed_cases']<=done['planned_cases']
            or done.get('completed_cases')!=result.get('exact_cases',result.get('completed_cases',0))
            or (certificate is None and done['planned_cases']!=0)
            or (certificate is not None and done['planned_cases']!=certificate['cardinality'])):
        raise ValueError('timing execution-path evidence disagrees with full result')
    return done


def small_json(path):
    # Live deadline checks must never deserialize/hash an arbitrarily large law.
    with path.open('rb') as stream:payload=stream.read(65537)
    if len(payload)>65536:raise ValueError('timing envelope exceeds its bounded format')
    return json.loads(payload)


class Watch:
    def __init__(self,entry,panel,attempt,pid,launch_monotonic,notify_fd=None):
        self.entry=entry;self.panel=Path(panel);self.bound=binding(entry,attempt,pid)
        self.launch=launch_monotonic;self.limit=operation_budget(entry)[0]['limit_seconds']
        self.notify_fd=notify_fd;self.buffer=b'';self.current=None;self.eof=False;self.live_frames={}

    def receive(self):
        if self.notify_fd is None:return
        while True:
            try:chunk=os.read(self.notify_fd,65536)
            except BlockingIOError:break
            if not chunk:
                self.eof=True;break
            self.buffer+=chunk
            while b'\n' in self.buffer:
                payload,self.buffer=self.buffer.split(b'\n',1)
                wire=json.loads(payload)
                if type(wire) is not list or len(wire)!=8 or wire[0]!=1:
                    raise ValueError('invalid live timing wire schema')
                frame=dict(schema=SCHEMA,attempt=wire[1],worker_pid=wire[2],entry_sha256=wire[3],
                    operation_budget_sha256=wire[4],input_freeze_sha256=wire[5],slot=wire[6],
                    start_monotonic=wire[7],status='STARTED')
                frozen=small_json(self.panel/'inputs-frozen.json')
                validate_start(frame,self.bound,frozen,self.entry['slots'],self.launch,time.monotonic())
                if self.current is not None and frame['slot']!=self.current['slot']+1:
                    raise ValueError('live timing sequence changed or repeated')
                if self.current is None and frame['slot']!=0:
                    raise ValueError('first live timing slot must be zero')
                self.current=frame
                self.live_frames[frame['slot']]=frame
            if len(self.buffer)>4096:raise ValueError('unbounded timing notification')
        if self.eof and self.buffer:raise ValueError('incomplete timing notification at owned pipe EOF')

    def wait(self,seconds):
        if self.notify_fd is None:
            time.sleep(seconds)
        else:
            select.select([self.notify_fd],[],[],seconds)
            self.receive()

    def observe(self):
        self.receive()
        frame=self.current
        if frame is None:
            if self.notify_fd is not None:return dict(expired=False,deadline=None,phase=None,reason=None)
            try:frame=small_json(self.panel/'timing-current.json')
            except FileNotFoundError:return dict(expired=False,deadline=None,phase=None,reason=None)
        frozen=small_json(self.panel/'inputs-frozen.json')
        now=time.monotonic()
        validate_start(frame,self.bound,frozen,self.entry['slots'],self.launch,now)
        start_path=self.panel/f"slot-{frame['slot']:04d}-timing-start.json"
        # A live pipe START remains authoritative even if immutable START fsync
        # is currently hung. Its absence is preserved at terminal, not ignored.
        if start_path.exists() and small_json(start_path)!=frame:
            raise ValueError('mutable timing pointer changed identity')
        deadline=frame['start_monotonic']+self.limit
        phase='operation_or_done_publication';reason='SLOT_DEADLINE_WITHOUT_DURABLE_DONE'
        done_path=self.panel/f"slot-{frame['slot']:04d}-timing-done.json"
        if done_path.exists():
            done=validate_done_header(small_json(done_path),frame,self.entry,now)
            if done['end_monotonic']>deadline:
                return dict(expired=True,deadline=deadline,slot=frame['slot'],
                    phase='operation',reason='PROVEN_OPERATION_OVERRUN')
            phase='witness_publication';reason='SLOT_DEADLINE_WITHOUT_PUBLICATION_WITNESS'
            pub_path=self.panel/f"slot-{frame['slot']:04d}-timing-publication.json"
            if pub_path.exists():
                validate_publication(small_json(pub_path),done,self.bound,frame['slot'],now)
                # A finished operation cannot disable owned liveness forever.
                # The excluded completion phase has one additional equal budget;
                # its expiry is not a claim that operation latency exceeded it.
                deadline=done['end_monotonic']+self.limit
                phase='completion_liveness';reason='COMPLETION_PHASE_LIVENESS_LIMIT'
        return dict(expired=now>=deadline,deadline=deadline,slot=frame['slot'],phase=phase,reason=reason)

    def overdue(self):
        return self.observe()['expired']

    def reconstruct(self,source_freeze_valid):
        observations=[];issues={};overhead={};execution={};overruns=[];now=time.monotonic();last_end=self.launch
        try:self.receive()
        except (OSError,ValueError,KeyError,TypeError) as error:
            issues['live_notification']=f'{type(error).__name__}: {error}'
        try:
            frozen=json.loads((self.panel/'inputs-frozen.json').read_bytes())
            if not source_freeze_valid:raise ValueError('input freeze invalid')
        except (OSError,ValueError):
            frozen=None
        root_records=None;constructor_context=None
        if frozen is not None and 'ordinary_carry' in self.entry:
            import gzip
            try:
                digest=hashlib.sha256();root_records=[]
                with gzip.open(self.panel/'inputs.jsonl.gz','rb') as stream:
                    for line in stream:
                        digest.update(line);root_records.append(json.loads(line))
                context_payload=(self.panel/'constructor-context.json').read_bytes()
                if (digest.hexdigest()!=frozen['input_jsonl_sha256']
                        or hashlib.sha256((self.panel/'inputs.jsonl.gz').read_bytes()).hexdigest()!=frozen['compressed_sha256']
                        or hashlib.sha256(context_payload).hexdigest()!=frozen['constructor_context_file_sha256']
                        or len(root_records)!=self.entry['slots']
                        or any(r['slot']!=i for i,r in enumerate(root_records))):
                    raise ValueError('TIMING_FROZEN_ROOTS_CHANGED')
                constructor_context=json.loads(context_payload)
            except (OSError,ValueError,KeyError,EOFError,zlib.error) as error:
                issues['root_dispatch_source']=str(error);frozen=None
        for slot in range(self.entry['slots']):
            start_path=self.panel/f'slot-{slot:04d}-timing-start.json'
            done_path=self.panel/f'slot-{slot:04d}-timing-done.json'
            result_path=self.panel/f'slot-{slot:04d}.json.gz'
            if not any(p.exists() for p in (start_path,done_path,result_path)) and slot not in self.live_frames:
                observations.append(dict(slot=slot,status='UNATTEMPTED'));continue
            try:
                if frozen is None:raise ValueError('input freeze invalid')
                frame=validate_start(json.loads(start_path.read_bytes()),self.bound,frozen,
                    self.entry['slots'],self.launch,now)
                if frame['slot']!=slot or frame['start_monotonic']<last_end:
                    raise ValueError('timing order or slot changed')
                done=validate_done(json.loads(done_path.read_bytes()),frame,result_path,self.entry,now,
                    root_record=None if root_records is None else root_records[slot],
                    constructor_context=constructor_context)
                last_end=done['end_monotonic']
                observations.append(dict(slot=slot,status='COMPLETE',seconds=done['seconds'],path=done['path']))
                execution[str(slot)]={k:done[k] for k in ('path','planned_cases','completed_cases','result_status')}
                if done['seconds']>self.limit:overruns.append(slot)
                pub=json.loads((self.panel/f'slot-{slot:04d}-timing-publication.json').read_bytes())
                validate_publication(pub,done,self.bound,slot,now)
                overhead[str(slot)]=pub['witness_publication_seconds']
            except (OSError,ValueError,KeyError,TypeError,AttributeError,EOFError,zlib.error,ZeroDivisionError) as error:
                issues[str(slot)]=f'{type(error).__name__}: {error}'
                if len(observations)<=slot:observations.append(dict(slot=slot,status='MISSING'))
        result=dict(observations=observations,timing_issues=issues,witness_publication_seconds=overhead,
            execution_evidence=execution,
            operation_overruns=overruns,
            live_starts=list(self.live_frames.values()),
            operation_budget_sha256=self.bound['operation_budget_sha256'],
            timing_evidence_valid=not issues and source_freeze_valid)
        if 'cost_admission' in self.entry:result['latency']=summarize(self.entry,observations)
        return result
