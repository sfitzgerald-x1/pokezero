"""Durable, finite IO recording for one independently admitted Native child.

The caller pins executable/source/runtime and sealed input applicability BEFORE
launch and supplies an independent expected registry. This module neither
launches/stages a child nor invents an action, chance operand or reference. It
records original bytes, waits on the existing completion FD and actual stdout
capture, then invokes the closed artifact verifier before another stdin write.

Capture/observer failures never terminate or restart the child. They preserve a
failed recording and leave its already-running work to the caller's real wall
deadline/memory guard. Only natural, explicitly commanded close is accepted by
finish. No context manager silently stops a child on an observer exception.
"""
import copy
import hashlib
import os
from pathlib import Path
import threading
import time

from .fresh200_native_consumer import CommandSpec, MAX_BYTES, _need
from . import fresh200_witness_audit as native


def _durable(path, data, *, replace=False):
    _need(type(data) is bytes, 'literal durable artifact bytes required')
    if not replace:
        with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    else:
        temporary=path.with_name(path.name+'.recording-tmp')
        with temporary.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def _json(value):
    # No runtime or scientific object is implicitly serialized here.
    import json
    return (json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')


def _identity(path):
    digest=hashlib.sha256();size=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):digest.update(block);size+=len(block)
    return dict(path=str(path),bytes=size,sha256=digest.hexdigest())


def _bounded_read(path):
    with path.open('rb') as stream:
        data = stream.read(MAX_BYTES + 1)
    _need(len(data) <= MAX_BYTES, 'recording input exceeds the byte limit')
    return data


class _Capture:
    """Drain an original FD even if the evidence sink fails; never parse IO."""
    def __init__(self, source, path, condition):
        self.source=source;self.path=path;self.condition=condition;self.lines=0;self.bytes=0;self.eof=False;self.failures=[]
        self.sink=path.open('xb');self.thread=threading.Thread(target=self._pump,daemon=True);self.thread.start()

    def _pump(self):
        try:
            while True:
                block=os.read(self.source.fileno(),65536)
                if not block:break
                try:self.sink.write(block);self.sink.flush();os.fsync(self.sink.fileno())
                except Exception as error:
                    with self.condition:self.failures.append(type(error).__name__+': '+str(error))
                    # Still drain the pipe so observer disk errors cannot stop
                    # Native at a full stdout/stderr/notification buffer.
                with self.condition:
                    self.bytes+=len(block);self.lines+=block.count(b'\n');self.condition.notify_all()
        except Exception as error:
            with self.condition:self.failures.append(type(error).__name__+': '+str(error))
        finally:
            try:self.sink.close()
            except Exception as error:
                with self.condition:self.failures.append(type(error).__name__+': '+str(error))
            with self.condition:self.eof=True;self.condition.notify_all()


class NativeCommandRecorder:
    """Record commands into a unique attempt; no hidden retries or extra reads.

    ``directory`` is the child's prepared bridge/control folder. ``artifacts``
    must be new and distinct; all logs/snapshots stay on that compute/storage.
    ``completion`` is the original read end of its separately declared FD.
    ``expected_registry_wire`` comes from independent source admission, never
    from witness-source-registry.json. Caller owns the finite process guard.
    """
    def __init__(self, *, directory, artifacts, child, completion, verifier,
            expected_registry_wire, deadline_monotonic, max_commands=20, visibility_policy_wire=None):
        _need(type(deadline_monotonic) in (int,float) and time.monotonic()<deadline_monotonic,
            'finite future recording wall deadline required')
        _need(type(max_commands) is int and 1<=max_commands<=20, 'bounded explicit command count required')
        _need(type(expected_registry_wire) is bytes and expected_registry_wire.endswith(b'\n')
            and len(expected_registry_wire)<=MAX_BYTES, 'independent literal source registry required')
        registry=native.document(expected_registry_wire.decode('utf8'))
        _need(type(registry) is dict and set(registry)=={'registry','sourceRegistryHash'}
            and registry['sourceRegistryHash']==verifier.source_registry_hash,
            'independent expected registry/causal verifier identity differs')
        self.visibility_sha256 = verifier.transport_cursor.visibility_policy_sha256
        _need(registry['registry'].get('ownerVisibilitySha256') == self.visibility_sha256,
            'independent expected registry/visibility identity differs')
        if self.visibility_sha256 is None:
            _need(visibility_policy_wire is None, 'unadmitted visibility policy')
        else:
            _need(type(visibility_policy_wire) is bytes and 0 < len(visibility_policy_wire) <= 4096
                and native.digest(visibility_policy_wire) == self.visibility_sha256
                and _bounded_read(Path(directory)/'owner-visibility.json') == visibility_policy_wire,
                'exact admitted resident visibility policy required')
        self.directory=Path(directory);self.artifacts=Path(artifacts);self.artifacts.mkdir(exist_ok=False)
        _need(self.directory.resolve()!=self.artifacts.resolve() and self.directory.is_dir(), 'separate prepared source and new artifact directories required')
        _need(child.stdin is not None and child.stdout is not None and child.stderr is not None,
            'original binary stdin/stdout/stderr pipes required')
        self.child=child;self.verifier=verifier;self.registry=expected_registry_wire;self.deadline=deadline_monotonic
        self.max_commands=max_commands;self.command_lock=threading.Lock();self.condition=threading.Condition()
        self.records=[];self.failed=None;self.finished=False;self.native_cursor=0;self.notification_cursor=0;self.transport_cursor=0
        self._captures={name:_Capture(source,self.artifacts/(name+'.bin'),self.condition)
            for name,source in (('stdout',child.stdout),('stderr',child.stderr),('completion',completion))}
        _durable(self.artifacts/'expected-source-registry.json',self.registry)
        self.admission_files = {}
        if self.visibility_sha256 is not None:
            path = self.artifacts/'owner-visibility.json'
            _durable(path, visibility_policy_wire)
            self.admission_files[path.name] = _identity(path)
        policy = self.verifier.hook_policy
        if policy is not None:
            for name, wire, digest in (
                ('hook-policy.json', policy.wire(), policy.sha256),
                ('hook-catalog.json', policy.catalog.wire(), policy.catalog.sha256),
            ):
                _need(len(wire) <= MAX_BYTES and native.digest(wire) == digest, 'independently admitted hook document differs')
                path = self.artifacts / name
                _durable(path, wire)
                self.admission_files[name] = _identity(path)
        self._manifest()

    def _manifest(self):
        value=dict(schema='pokezero.fresh200.raw_command_recording.v1',qualification=False,
            status='FAILED_RAW_RECORDING' if self.failed else 'RAW_CLOSED_ARTIFACT_SET_RECORDED_ONLY' if self.finished else 'RECORDING',
            source_registry_hash=self.verifier.source_registry_hash,commands=copy.deepcopy(self.records),failure=self.failed,
            open_gates=['sealed fixed-input program applicability','independent73-field physical replay','literal owning references','actual-device parity'])
        if self.admission_files:
            value['admission_files'] = copy.deepcopy(self.admission_files)
        if self.visibility_sha256 is not None:
            value['owner_visibility_sha256'] = self.visibility_sha256
        _durable(self.artifacts/'recording.json',_json(value),replace=(self.artifacts/'recording.json').exists())
        return value

    def _remaining(self):
        value=self.deadline-time.monotonic()
        if value<=0:raise TimeoutError('recording wall deadline reached; caller owns its real process guard')
        return value

    def _await(self, predicate):
        with self.condition:
            while not predicate():
                failures=[name+': '+failure for name,capture in self._captures.items() for failure in capture.failures]
                if failures:raise RuntimeError('original IO capture failed; child left running: '+'; '.join(failures))
                self.condition.wait(timeout=self._remaining())
            failures=[name+': '+failure for name,capture in self._captures.items() for failure in capture.failures]
            if failures:raise RuntimeError('original IO capture failed; child left running: '+'; '.join(failures))

    def _saved_line(self, path, position):
        with path.open('rb') as stream:
            stream.seek(position);value=stream.readline(MAX_BYTES+1);end=stream.tell()
        _need(0<len(value)<=MAX_BYTES and value.endswith(b'\n') and not value.endswith(b'\r\n'),
            'bounded complete saved receipt bytes required')
        return value,end

    def record(self, spec, *, chance_control_wire=None):
        """Record one command with the independently supplied chance control, if used."""
        _need(self.command_lock.acquire(blocking=False), 'one original command at a time required')
        index=len(self.records);record=None
        try:
            _need(not self.failed and not self.finished and index<self.max_commands, 'closed failed/finished/bounded attempt cannot accept input')
            _need(isinstance(spec,CommandSpec), 'independently sealed original command required')
            command,control,bound=self.verifier.admit_spec(spec);key=(control['run'],control['stream'],control['reset'])
            _need(control['phase'] in ('start','choices','snapshot','end','close')
                and control['command']==self.verifier.native_cursor.commands.get(key,-1)+1
                and control['previous_receipt_hash']==self.verifier.native_cursor.receipt_hash
                and bound['previous_completion_hash']==self.verifier.transport_cursor.previous_completion,
                'independent next command/receipt parents differ before stdin write')
            chance_path = self._validate_chance_control(control, chance_control_wire)
            self._remaining();folder=self.artifacts/('command-'+str(index));folder.mkdir(exist_ok=False)
            record=dict(index=index,command_id=control['nonce'],phase=command['type'],status='SEALED_BEFORE_STDIN',inputs={})
            self.records.append(record)
            inputs = {
                'stdin.jsonl': spec.stdin_wire,
                'native-control.json': spec.native_control_wire,
                'transport-control.json': spec.transport_control_wire,
            }
            if spec.hook_admission is not None:
                inputs['hook-admission.json'] = _json(vars(spec.hook_admission))
            else:
                inputs['hook-admission.json'] = _json(dict(hook_identity_digest=spec.hook_identity_digest))
            if chance_path is not None:
                inputs[chance_path.name] = chance_control_wire
            for name,data in inputs.items():
                path=folder/name;_durable(path,data);record['inputs'][name]=_identity(path)
            _durable(self.directory/'witness-control.json',spec.native_control_wire,replace=True)
            _durable(self.directory/'witness-transport-control.json',spec.transport_control_wire,replace=True)
            self._manifest()
            if chance_path is not None:
                _need(_bounded_read(chance_path) == chance_control_wire, 'resident chance control changed before stdin write')
            self._remaining()
            # Exactly the admitted bytes, once. No newline/protocol sentinel,
            # default action, read probe or implicit teardown is introduced.
            wrote=self.child.stdin.write(spec.stdin_wire)
            _need(wrote==len(spec.stdin_wire),'original stdin write was incomplete')
            self.child.stdin.flush();record['status']='SUBMITTED'
            notification=self._captures['completion'];stdout=self._captures['stdout']
            self._await(lambda:notification.lines>index or notification.eof)
            notification_wire,next_notification=self._saved_line(notification.path,self.notification_cursor)
            receipt=native.document(notification_wire.decode('utf8'));frontier=receipt.get('stdout_frontier') if type(receipt) is dict else None
            _need(type(frontier) is int and frontier>=0, 'literal completed stdout frontier required')
            self._await(lambda:stdout.lines>=frontier or stdout.eof)
            _need(stdout.lines==frontier,'captured original stdout frontier differs')
            _need((self.directory/'witness-source-registry.json').read_bytes()==self.registry,
                'running registry differs from independent prelaunch source admission')
            native_wire,next_native=self._saved_line(self.directory/'witness-receipts.jsonl',self.native_cursor)
            transport_wire,next_transport=self._saved_line(self.directory/'witness-transport-receipts.jsonl',self.transport_cursor)
            _need(transport_wire==notification_wire, 'durable completion differs from original dedicated FD bytes')
            _durable(folder/'native-receipt.jsonl',native_wire);_durable(folder/'transport-receipt.jsonl',transport_wire)
            positions=dict(audit=0 if not index else self.records[index-1]['verified_extents']['audit'][1],
                events=0 if not index else self.records[index-1]['verified_extents']['events'][1],
                stdout=0 if not index else self.records[index-1]['verified_extents']['stdout'][1])
            paths=dict(audit=self.directory/'witness-audit.jsonl',events=self.directory/'witness-transport-events.jsonl',stdout=stdout.path)
            opened={name:path.open('rb') for name,path in paths.items()}
            try:
                for name,stream in opened.items():stream.seek(positions[name])
                result=self.verifier.verify(spec,audit=opened['audit'],events=opened['events'],stdout=opened['stdout'],
                    native_receipt_wire=native_wire,transport_receipt_wire=transport_wire)
                record['verified_extents']={name:[positions[name],stream.tell()] for name,stream in opened.items()}
            finally:
                for stream in opened.values():stream.close()
            record.update(status='RAW_CLOSED_COMMAND_VERIFIED_ONLY',verification=result)
            self.native_cursor=next_native;self.notification_cursor=next_notification;self.transport_cursor=next_transport
            self._manifest();return copy.deepcopy(record)
        except Exception as error:
            if self.failed is None:
                self.failed=dict(command_index=index,error=type(error).__name__,message=str(error),child_stopped_by_recorder=False)
            if record is not None:record['status']='FAILED_RAW_RECORDING'
            self._manifest();raise
        finally:self.command_lock.release()

    def _validate_chance_control(self, control, wire):
        if wire is None:
            if self.verifier.hook_policy is not None:
                _need(not any((self.directory / name).exists() for name in ('initializing.json', 'chance.json')),
                      'active chance control requires independently sealed bytes')
            return None
        _need(type(wire) is bytes and 0 < len(wire) <= MAX_BYTES, 'bounded exact chance control bytes required')
        phase = control['phase']
        _need(phase in ('start', 'choices'), 'chance control belongs only to start or choices')
        name, other = ('initializing.json', 'chance.json') if phase == 'start' else ('chance.json', 'initializing.json')
        path = self.directory / name
        _need(not (self.directory / other).exists(), 'overlapping Native chance controls')
        value = native.document(wire.decode('utf8'))
        _need(type(value) is dict, 'literal chance control object required')
        if phase == 'choices':
            _need(type(control['chance_nonce']) is str and bool(control['chance_nonce'])
                  and value.get('nonce') == control['chance_nonce'], 'chance control nonce differs')
        _need(_bounded_read(path) == wire, 'resident chance control differs before stdin write')
        return path

    def archive_chance_before_retirement(self, *, retire):
        """Preserve exact bytes before existing initialized()/end() removes them."""
        _need(self.records and self.records[-1]['status']=='RAW_CLOSED_COMMAND_VERIFIED_ONLY' and callable(retire),
            'closed command and explicit existing chance retirement required')
        folder=self.artifacts/('command-'+str(len(self.records)-1));source=self.directory/'chance-audit.jsonl'
        data=_bounded_read(source) if source.exists() else b''
        expected=[]
        for line in data.splitlines(keepends=True):
            _need(line.endswith(b'\n'),'complete chance audit frames required');expected.append(native.document(line.decode('utf8')))
        path=folder/'chance-audit.jsonl';_durable(path,data)
        self.records[-1]['chance_audit']=_identity(path);self._manifest()
        actual=retire()
        _need(native._same(actual,expected),'existing chance retirement differs from archived literal bytes')
        return copy.deepcopy(actual)

    def finish(self, *, expected_command_ids):
        """Require natural close, original EOF, exit0 and no unaccounted tails."""
        _need(self.command_lock.acquire(blocking=False),'finish cannot overlap an original command')
        try:return self._finish(expected_command_ids=expected_command_ids)
        except Exception as error:
            if self.failed is None:
                self.failed=dict(command_index=len(self.records),error=type(error).__name__,message=str(error),child_stopped_by_recorder=False)
            self._manifest();raise
        finally:self.command_lock.release()

    def _finish(self, *, expected_command_ids):
        _need(not self.failed and not self.finished and self.records,'one clean completed recording required')
        self._await(lambda:all(x.eof for x in self._captures.values()))
        code=self.child.wait(timeout=self._remaining())
        _need(code==0, 'original Native child exited nonzero: '+str(code))
        for name,position in (('completion',self.notification_cursor),):
            _need(self._captures[name].path.stat().st_size==position,'unaccounted original dedicated FD tail')
        _need((self.directory/'witness-receipts.jsonl').stat().st_size==self.native_cursor
            and (self.directory/'witness-transport-receipts.jsonl').stat().st_size==self.transport_cursor,
            'unaccounted durable Native/transport receipt tail')
        paths=dict(audit=self.directory/'witness-audit.jsonl',events=self.directory/'witness-transport-events.jsonl',stdout=self._captures['stdout'].path)
        opened={name:path.open('rb') for name,path in paths.items()}
        try:
            for name,stream in opened.items():stream.seek(self.records[-1]['verified_extents'][name][1])
            result=self.verifier.finish_artifacts(**opened,expected_command_ids=expected_command_ids)
        finally:
            for stream in opened.values():stream.close()
        self.finished=True;manifest=self._manifest();manifest['files']={name:_identity(capture.path) for name,capture in self._captures.items()}
        manifest['files'].update({
            name: _identity(self.directory / name) for name in (
                'witness-audit.jsonl', 'witness-receipts.jsonl',
                'witness-transport-events.jsonl', 'witness-transport-receipts.jsonl',
                'witness-source-registry.json',
            )
        })
        manifest['closure']=result;manifest['child_exit_code']=code;_durable(self.artifacts/'completed.json',_json(manifest))
        return manifest
