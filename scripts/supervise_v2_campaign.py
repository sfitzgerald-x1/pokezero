"""Supervise one released V2 campaign through the approved single-node launcher."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_engine import v2_material_contract as material


ADMISSION_SECONDS = 6 * 3600
EXEC_MARGIN_SECONDS = 600
ATTEMPT_LIMIT = 3
TICK_SECONDS = 60
GPU_COUNT = 4
BASE = ('kubectl', '--context', 'potomac', '-n', 'scott', '--request-timeout=25s')
OPAQUE_GPU = re.compile(r'v2[a-z]-[0-9a-f]{6,12}-gpu')
OUT_HOST = '/shared/scott/v2p'


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def epoch(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def run(argv, *, timeout=35):
    result = subprocess.run(argv, timeout=timeout, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}: {result.stderr.strip()}')
    return result.stdout


def read_resource(*args):
    output = run([*BASE, 'get', *args, '-o', 'json'])
    return json.loads(output) if output.strip() else None


def shared_text(root, relative, *, line=None):
    path = f'{OUT_HOST}/{root}/{relative}'
    script = 'if test -s "$1"; then cat "$1"; fi'
    if line is not None:
        script = 'test -s "$1" && sed -n "$2{p;q;}" "$1"'
    return run([*BASE, 'exec', 'eigen-stage-0', '-c', 'stage', '--', 'sh', '-c',
                script, 'sh', path, str(line or 0)])


def shared_file(root, relative, *, line=None):
    output = shared_text(root, relative, line=line)
    return json.loads(output) if output.strip() else None



def material_bytes(path):
    result = run([*BASE, 'exec', 'eigen-stage-0', '-c', 'stage', '--', 'sh', '-c',
                  'if test -f "$1"; then printf "+\\n"; cat "$1"; fi', 'sh', path])
    if not result:
        raise FileNotFoundError(path)
    prefix, data = result.split('\n', 1)
    if prefix != '+':
        raise ValueError('invalid shared artifact read')
    return data.encode()


def verify_material_files(document):
    verify_material_refs(material.material_artifacts(document))


def verify_material_refs(refs):
    if any(any(char in ref['path'] for char in '\r\n\\') for ref in refs):
        raise ValueError('invalid shared artifact filename')
    checks = ''.join(f"{ref['sha256']}  {ref['path']}\n" for ref in refs)
    result = subprocess.run(
        [*BASE[:-1], 'exec', '-i', 'eigen-stage-0', '-c', 'stage', '--', 'sha256sum', '--check', '--quiet', '-'],
        input=checks, text=True, capture_output=True, timeout=900,
    )
    if result.returncode:
        detail = (result.stdout + result.stderr).strip()
        if any(message in detail for message in ('FAILED', 'No such file')):
            raise ValueError(f'material artifact could not be verified: {detail[:400]}')
        raise RuntimeError(f'material checksum command exited {result.returncode}: {detail[:400]}')


def missing_material(paths):
    output = run([*BASE, 'exec', 'eigen-stage-0', '-c', 'stage', '--', 'sh', '-c',
                  'for p do if ! test -f "$p"; then printf "%s\\n" "$p"; fi; done',
                  'sh', *map(str, paths)])
    return set(output.splitlines())

def verify_initial_root(output, text, state, restore):
    if (completed_rounds(state) != 100
            or hashlib.sha256(text.encode()).hexdigest() != restore['state_sha256']
            or state['declaration_sha256'] != restore['declaration_sha256']):
        raise ValueError('production root differs from the bound 2M pointer')
    snap = state['round_ledger']['latest_snapshot']
    checkpoint = Path(snap['checkpoint'])
    if (snap['checkpoint_sha256'] != restore['checkpoint_sha256']
            or checkpoint.parent != Path('artifacts') or checkpoint.suffix != '.pt'):
        raise ValueError('production root differs from the selected 2M checkpoint')
    path = f'{OUT_HOST}/{output}/{checkpoint}'
    digest = run([*BASE, 'exec', 'eigen-stage-0', '-c', 'stage', '--', 'sha256sum', path]).split()[0]
    if digest != restore['checkpoint_sha256']:
        raise ValueError('selected 2M checkpoint bytes differ at first admission')


def verify_signed_release(data, signature, signers, packet_data, ledger_path):
    result = subprocess.run(
        ['ssh-keygen', '-Y', 'verify', '-f', str(signers), '-I', 'scott',
         '-n', 'pokezero-100m', '-s', str(signature)],
        input=data, capture_output=True, timeout=15,
    )
    if result.returncode:
        raise ValueError('Scott’s release signature did not verify')
    release = json.loads(data)
    if (release['packet_sha256'] != hashlib.sha256(packet_data).hexdigest()
            or type(release['gpu_seconds']) is not int or release['gpu_seconds'] <= 0
            or str(ledger_path.resolve()) != release['ledger']):
        raise ValueError('signed release differs from the packet, finite budget or canonical ledger')
    return release


def completed_rounds(doc):
    if doc is None:
        return 0
    ledger = doc['round_ledger']
    count = ledger['completed_rounds']
    if type(count) is not int or count < 1 or ledger['latest_snapshot']['iteration'] != count - 1:
        raise ValueError('invalid committed round pointer')
    return count


def verify_attempt_pointer(output, raw_state, state, previous_attempt, *, restore=None):
    committed = completed_rounds(state)
    state_sha256 = '-' if state is None else hashlib.sha256(raw_state.encode()).hexdigest()
    if previous_attempt is None:
        if restore is not None:
            verify_initial_root(output, raw_state, state, restore)
        elif state is not None:
            raise ValueError('a child’s first admission requires an empty committed pointer')
    elif (committed, state_sha256) != (previous_attempt['committed_round'], previous_attempt['state_sha256']):
        raise ValueError('segment resume pointer differs from its prior owned attempt')
    return committed, state_sha256


def segment_order(main, children):
    """Run each cooldown after the first main stop that includes its parent."""
    remaining = sorted(children, key=lambda child: child['parent_round'])
    ordered = []
    for stop in main['stops']:
        ordered.append(dict(main, stop=stop))
        eligible = [child for child in remaining if child['parent_round'] <= stop]
        ordered.extend(eligible)
        remaining = [child for child in remaining if child['parent_round'] > stop]
    if remaining:
        raise ValueError('a cooldown parent is beyond the main-line horizon')
    return ordered


def observe_job(name):
    job = read_resource('job', name, '--ignore-not-found')
    if job is None:
        return None
    uid = job['metadata']['uid']
    pods = read_resource('pods', '-l', f'job-name={name}')['items']
    owned = [pod for pod in pods if any(
        ref['kind'] == 'Job' and ref['uid'] == uid for ref in pod['metadata'].get('ownerReferences', ())
    )]
    live = [pod for pod in owned if pod['status']['phase'] not in ('Succeeded', 'Failed')]
    conditions = {item['type'] for item in job.get('status', {}).get('conditions', ()) if item['status'] == 'True'}
    workloads = read_resource('workloads', '-l', f'kueue.x-k8s.io/job-uid={uid}')['items']
    queue = [
        dict(active=workload.get('spec', {}).get('active', True), conditions=[
            row for row in workload.get('status', {}).get('conditions', ())
            if row['type'] in ('Admitted', 'Evicted', 'Requeued', 'DeactivationTarget')
        ])
        for workload in workloads
    ]
    return dict(uid=uid, live=bool(live), complete='Complete' in conditions,
                failed='Failed' in conditions or any(not item['active'] for item in queue),
                admitted=bool(job.get('status', {}).get('startTime')) or any(
                    pod.get('spec', {}).get('nodeName') for pod in owned),
                queue=queue)


def disposition(observed, attempt, committed, stop, *, at, terminal_status=None):
    if observed is None:
        return 'AMBIGUOUS' if at - epoch(attempt['submitted_at']) >= ADMISSION_SECONDS else None
    if not attempt.get('ever_admitted') and not observed['admitted']:
        if at - epoch(attempt['submitted_at']) >= ADMISSION_SECONDS:
            return 'ADMISSION_EXPIRED'
    if observed['live'] or not (observed['complete'] or observed['failed']):
        return None
    if terminal_status == 'STOPPED':
        return 'SIGNED_STOP'
    if committed == stop and terminal_status == 'COMPLETE':
        return 'ADVANCE'
    if committed == stop:
        return 'RETRY'
    if observed['complete'] or committed > stop:
        return 'STOPPED_SHORT'
    return 'RETRY'


def build_launch(packet, segment, attempt, parent_checkpoint):
    output = segment['output']
    main = 'parent_round' not in segment
    resume = attempt['before_round'] > 0
    name = f"{output}-r{attempt['number']}" if resume else output
    runtime = packet['runtime']
    declaration = segment['declaration']
    resilience = packet['resilience']
    worker_seconds = attempt['cap_seconds'] - EXEC_MARGIN_SECONDS
    entry = [
        '-m', 'gpu_engine.v2_distributed', '--ranks', '4', '--timeout-seconds', str(worker_seconds),
        '--collective-timeout', '1800', '--s0-idle-guard', '--out', '/out',
        '--declaration', declaration['path'], '--declaration-sha256', declaration['sha256'],
        '--runtime', runtime['path'], '--runtime-sha256', runtime['sha256'],
        '--plan-arm', 'B0', '--reference-run', packet['reference_run'],
        '--max-iterations', str(segment['stop']), '--lanes', '2048', '--row-order', 'file',
        '--segment-outcome', '--segment-start', str(attempt['before_round']), attempt['before_state_sha256'],
        str(attempt['number']),
        '--torch-memory-fraction', '.45', '--compile-threads', '4', '--recipe', 'R2x1', '--mode', 'cudagraphs',
        '--learner-precision', 'float32', '--native-embedding-backward', '--fallback-random',
        '--profile', 'v2-production', '--campaign-anneal', '--lr-start', 'sqrt',
        '--lr-warmup-updates', '24000', '--freeze-gc-after-warmup', '--audit-every', '50',
    ]
    if 'material' in attempt:
        plan = packet['material']['plan']
        ref = attempt['material']
        entry.extend(['--material', plan['path'], plan['sha256'], str(attempt['segment']),
                      ref['path'], ref['sha256']])
    if main:
        if not resume:
            raise ValueError('the main campaign must start from the adopted 2M parent')
        milestones = segment['restore']['milestone_sha256s']
        entry.extend(['--adopt-2m-prefix-runtime', packet['previous_runtime'],
                      '--adopted-milestone-sha256s', milestones['1000000'], milestones['2000000']])
    else:
        entry.extend(['--cooldown-child', f"{OUT_HOST}/{packet['main']['output']}",
                      '--cooldown-parent-sha256', parent_checkpoint])
    # A replacement pod may resume a pointer advanced by this same Job.
    options = ['--resume-run', output] if resume else []
    argv = [
        sys.executable, packet['launcher']['path'], '--clone', packet['checkout'],
        '--context', 'potomac', '--name', name, '--gpus', '4',
        '--cpu', str(packet['cpu']), '--memory', packet['memory'],
        '--kueue', '--queue', 'medium', '--admission-hours', '6',
        '--max-exec-seconds', str(attempt['cap_seconds']), '--trace-jax-cache',
        '--resilient-support', resilience['path'],
        '--phase-observer-sha256', resilience['phase_observer_sha256'],
        '--signal-driver-sha256', resilience['signal_driver_sha256'],
        '--secret-env', packet['stop_key_reference'], *options, '--entry', shlex.join(entry),
    ]
    return f'scott-{name}', argv


def verify_source(packet):
    launcher = packet['launcher']
    if hashlib.sha256(Path(launcher['path']).read_bytes()).hexdigest() != launcher['sha256']:
        raise ValueError('launcher differs from the released binding')
    checkout = ['git', '-C', packet['checkout']]
    if (run([*checkout, 'rev-parse', 'HEAD']).strip() != packet['source_commit']
            or run([*checkout, 'status', '--porcelain'])):
        raise ValueError('the production checkout differs from the released source')


class Supervisor:
    def __init__(self, packet, ledger_path, binding, budget):
        self.packet = packet
        self.path = ledger_path
        self.segments = segment_order(packet['main'], packet['children'])
        self.budget = budget
        if ledger_path.exists():
            self.ledger = json.loads(ledger_path.read_bytes())
            if self.ledger['release_binding'] != binding:
                raise ValueError('ledger belongs to a different campaign release')
        else:
            self.ledger = dict(release_binding=binding, status='RUNNING', segment=0,
                               consecutive_failures=0, reserved_gpu_seconds=0, attempts=[])

    def material_plan(self):
        if not hasattr(self, '_material_plan'):
            ref = self.packet['material']['plan']
            self._material_plan = material.load_plan(ref, material_bytes)
            if len(self._material_plan['windows']) != len(self.segments):
                raise ValueError('material and supervisor segment counts differ')
        return self._material_plan

    def bind_segment(self, segment):
        stream = self.packet['material']
        plan = self.material_plan()
        index = self.ledger['segment']
        pins = self.ledger.setdefault('material_bindings', [])
        previous = pins[index - 1]['sha256'] if index else None
        try:
            ref, doc = material.bound_manifest(plan, stream['plan']['sha256'], index, previous, material_bytes)
            for bindings in (stream['ready'], pins):
                if index < len(bindings) and ref != bindings[index]:
                    raise ValueError('material differs from its admitted manifest')
            if doc['window']['rounds'][1] != segment['stop']:
                raise ValueError('material segment stop differs from the supervisor schedule')
            material.require_verification(plan, stream['plan']['sha256'], index, ref, material_bytes)
            verify_material_files(doc)
        except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
            reason = error if not isinstance(error, FileNotFoundError) else f'missing {error.filename or error}'
            self.wait_for_material(index, str(reason))
            return None
        if index == len(pins):
            pins.append(ref)
        self.ledger.pop('material_wait', None)
        return ref

    def wait_for_material(self, index, reason):
        self.ledger['material_wait'] = dict(at=now(), segment=index, detail=reason[:480])
        self.save()
        print(f'WAITING FOR MATERIAL {index + 1}: {reason[:480]}', flush=True)

    def fail_producer(self, active, reason):
        self.ledger['producer_failure'] = dict(job=active['job'], start=active['start'],
                                               stop=active['stop'], reason=reason)
        self.save()

    def advance_producer(self):
        stream = self.packet['material']
        plan = self.material_plan()
        cfg = stream['producer']
        active = self.ledger.get('producer')
        if active is not None:
            info = read_resource('job', active['job'], '--ignore-not-found')
            if info is None:
                elapsed = time.time() - epoch(active['submitted_at'])
                failed_launch = active.get('launcher_exit', 0) != 0 and elapsed >= 120
                if not active.get('terminal') and not failed_launch and elapsed < ADMISSION_SECONDS:
                    return
                if not active.get('terminal'):
                    self.fail_producer(active, 'producer admission could not be reconciled')
                del self.ledger['producer']
                self.save()
            else:
                uid = info['metadata']['uid']
                if active.get('uid', uid) != uid:
                    raise ValueError('producer Job UID differs from the owned admission')
                active['uid'] = uid
                if not active.get('terminal'):
                    conditions = {row['type'] for row in info.get('status', {}).get('conditions', ())
                                  if row['status'] == 'True'}
                    if not conditions & {'Complete', 'Failed'}:
                        self.save()
                        return
                    active['terminal'] = 'Failed' if 'Failed' in conditions else 'Complete'
                    if active['terminal'] == 'Failed':
                        self.fail_producer(active, 'producer job failed')
                    self.save()
                run([*BASE, 'delete', 'job', active['job'], '--ignore-not-found', '--wait=false'])
                return
        index = self.ledger['segment']
        final = Path(plan['root']) / 'z' / 'd.json'
        stop = min(len(self.segments), index + cfg['lookahead'] + 1)
        segments = {
            i: (material.manifest_path(plan, i), material.verification_path(plan, i)) for i in range(index, stop)
        }
        paths = [path for pair in segments.values() for path in pair]
        if stop == len(self.segments):
            paths.append(final)
        missing = missing_material(paths) if paths else set()
        if not missing:
            self.ledger.pop('producer_failure', None)
            return
        start = next((i for i, paths in segments.items() if missing.intersection(map(str, paths))),
                     min(index, stop - 1))
        failure = self.ledger.get('producer_failure')
        if failure is not None:
            if failure['start'] <= start < failure['stop']:
                self.wait_for_material(index, f"{failure['reason']}. Publish the same window after checking its log")
                return
            self.ledger.pop('producer_failure')
        verify_source(self.packet)
        plan_ref = stream['plan']
        name = f'v2m-{secrets.token_hex(5)}-cpu'
        entry = ['-m', 'gpu_engine.v2_material', '--plan', plan_ref['path'], '--sha256', plan_ref['sha256'],
                 '--start', str(start), '--stop', str(stop), '--workers', str(cfg['workers'])]
        argv = [
            sys.executable, self.packet['launcher']['path'], '--clone', self.packet['checkout'],
            '--context', BASE[2], '--name', name, '--gpus', '0', '--cpu', str(cfg['cpu']),
            '--memory', cfg['memory'], '--max-exec-seconds', str(cfg['max_exec_seconds']),
            '--entry', shlex.join(entry),
        ]
        for node in cfg['nodes']:
            argv.extend(('--only-node', node))
        self.ledger['producer'] = dict(job=f'scott-{name}', start=start, stop=stop, submitted_at=now())
        self.save()
        with self.path.with_name(f'{self.path.stem}-{name}.log').open('x') as log:
            result = subprocess.run(['timeout', '480', *argv], stdout=log, stderr=subprocess.STDOUT)
        self.ledger['producer']['launcher_exit'] = result.returncode
        self.save()

    def material_complete(self):
        if 'producer' in self.ledger:
            return False
        index = self.ledger['segment']
        try:
            path = Path(self.material_plan()['root']) / 'z' / 'd.json'
            receipt = json.loads(material_bytes(str(path)))
            if (receipt['plan_sha256'] != self.packet['material']['plan']['sha256']
                    or receipt['ledger_sha256'] != self.ledger['material_bindings'][-1]['sha256']
                    or receipt['team_rows'] != material.shard_end(self.material_plan()['shards'][-1])
                    or not re.fullmatch('[0-9a-f]{64}', receipt['pool_sha256'])):
                raise ValueError('final material digest differs from the admitted ledger')
            verify_material_refs([dict(path=str(path.parent / 't.npz'), sha256=receipt['pool_sha256'])])
        except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
            self.wait_for_material(index, str(error))
            return False
        self.ledger['material_result'] = receipt
        self.ledger.pop('material_wait', None)
        return True

    def save(self):
        partial = self.path.with_suffix('.partial')
        partial.write_text(json.dumps(self.ledger, indent=2) + '\n')
        partial.replace(self.path)

    def stop(self, outcome):
        self.ledger.update(status=outcome, finished_at=now())
        self.save()

    def latest_attempt(self, output):
        return next((attempt for attempt in reversed(self.ledger['attempts'])
                     if attempt['output'] == output), None)

    def reconciled_rounds(self, output):
        attempt = self.latest_attempt(output)
        if attempt is None or (attempt['status'], attempt['outcome']) != ('RELEASED', 'ADVANCE'):
            raise ValueError('segment output has no completed, released attempt')
        raw_state = shared_text(output, 'state.json')
        state = json.loads(raw_state) if raw_state.strip() else None
        rounds, _ = verify_attempt_pointer(output, raw_state, state, attempt)
        return rounds

    def finish(self):
        stops = {segment['output']: segment['stop'] for segment in self.segments}
        for output, stop in stops.items():
            if self.reconciled_rounds(output) != stop:
                raise ValueError('completed segment output differs from the declared stop')
        self.stop('COMPLETE')

    def submit(self, segment):
        binding = None
        if 'material' in self.packet:
            binding = self.bind_segment(segment)
            if binding is None:
                return
        remaining = self.budget - self.ledger['reserved_gpu_seconds']
        cap = min(segment['max_exec_seconds'], remaining // GPU_COUNT)
        if cap < segment['min_exec_seconds']:
            self.stop('GPU_BUDGET_EXHAUSTED')
            return
        raw_state = shared_text(segment['output'], 'state.json')
        doc = json.loads(raw_state) if raw_state.strip() else None
        prior = self.latest_attempt(segment['output'])
        before, state_sha256 = verify_attempt_pointer(segment['output'], raw_state, doc, prior,
                                                      restore=segment.get('restore'))
        if before > segment['stop']:
            raise ValueError('committed pointer exceeds the segment stop')
        parent_checkpoint = None
        if 'parent_round' in segment:
            main_output = self.packet['main']['output']
            parent_round = segment['parent_round']
            if self.reconciled_rounds(main_output) < parent_round:
                raise ValueError('cooldown parent has not committed')
            snapshot = shared_file(main_output, 'reports.jsonl', line=parent_round)['report']['snapshot']
            if snapshot['iteration'] != parent_round - 1:
                raise ValueError('cooldown parent is outside its committed round')
            parent_checkpoint = snapshot['checkpoint_sha256']
        attempt = dict(number=len(self.ledger['attempts'])+1, segment=self.ledger['segment'],
                       before_round=before, before_state_sha256=state_sha256,
                       submitted_at=now(), cap_seconds=cap, status='SUBMITTING')
        if binding is not None:
            bounds = self.material_plan()['windows'][attempt['segment']]['rounds']
            if not bounds[0] <= before <= bounds[1]:
                raise ValueError('committed pointer is outside the admitted material segment')
            if before > bounds[0]:
                expected = binding['sha256']
            else:
                previous = next((i for i in range(attempt['segment'] - 1, -1, -1)
                                 if self.segments[i]['output'] == segment['output']), None)
                expected = None if previous is None else self.ledger['material_bindings'][previous]['sha256']
            if expected is not None and doc.get('material_sha256') != expected:
                raise ValueError('committed pointer does not match material admitted on this lineage')
            attempt['material'] = binding
        name, argv = build_launch(self.packet, segment, attempt, parent_checkpoint)
        if read_resource('job', name, '--ignore-not-found') is not None:
            raise ValueError('a prior Job still owns this attempt name')
        verify_source(self.packet)
        attempt.update(job=name, output=segment['output'])
        self.ledger['attempts'].append(attempt)
        self.ledger['reserved_gpu_seconds'] += GPU_COUNT * cap
        self.save()
        with self.path.with_name(f"{self.path.stem}-{attempt['number']}.log").open('x') as log:
            result = subprocess.run(['timeout', '480', *argv], stdout=log, stderr=subprocess.STDOUT)
        attempt.update(launcher_exit=result.returncode,
                       status='SUBMITTED' if result.returncode == 0 else 'ADMISSION_UNKNOWN')
        self.save()

    def reconcile(self, segment, attempt):
        observed = observe_job(attempt['job'])
        if observed is not None:
            if attempt.get('uid', observed['uid']) != observed['uid']:
                raise ValueError('observed Job UID differs from the owned attempt')
            attempt['uid'] = observed['uid']
            attempt['ever_admitted'] = attempt.get('ever_admitted', False) or observed['admitted']
            if attempt.get('placement') != observed['queue']:
                attempt['placement'] = observed['queue']
                attempt.setdefault('queue_events', []).append(dict(at=now(), queue=observed['queue']))
        raw_state = shared_text(segment['output'], 'state.json')
        state = json.loads(raw_state) if raw_state.strip() else None
        committed = completed_rounds(state)
        state_sha256 = '-' if state is None else hashlib.sha256(raw_state.encode()).hexdigest()
        name = attempt['job'].removeprefix('scott-')
        if committed > attempt['before_round']:
            if state.get('segment_job') != f"{name}:{attempt['number']}":
                raise ValueError('committed pointer advanced outside the owned Job')
            if 'material' in self.packet and state.get('material_sha256') != attempt['material']['sha256']:
                raise ValueError('committed pointer differs from the admitted material manifest')
        elif (committed, state_sha256) != (attempt['before_round'], attempt['before_state_sha256']):
            raise ValueError('committed pointer differs from the admitted state')
        if committed > attempt['before_round'] and 'first_commit' not in attempt:
            attempt['first_commit'] = dict(at=now(), round=committed)
        terminal_status = None
        if observed and not observed['live'] and (observed['complete'] or observed['failed']):
            receipt = shared_file(segment['output'], f'segment-{name}.json')
            if receipt is not None:
                if (not committed or receipt['status'] not in ('STOPPED', 'COMPLETE')
                        or receipt.get('attempt') != attempt['number']
                        or receipt['completed_rounds'] != committed
                        or receipt['requested_round'] != segment['stop']
                        or receipt['declaration_sha256'] != segment['declaration']['effective_sha256']
                        or receipt['checkpoint_sha256'] != state['round_ledger']['latest_snapshot']['checkpoint_sha256']):
                    raise ValueError('terminal segment receipt differs from the committed checkpoint or declared stop')
                terminal_status = receipt['status']
        outcome = disposition(observed, attempt, committed, segment['stop'], at=time.time(),
                              terminal_status=terminal_status)
        self.save()
        if outcome is None:
            return
        if outcome == 'AMBIGUOUS':
            self.stop('ADMISSION_UNRECONCILED')
            return
        # Persist the terminal reconciliation before deleting the confirmed Job.
        checkpoint = None if not committed else state['round_ledger']['latest_snapshot']['checkpoint_sha256']
        attempt.update(status='RELEASING', outcome=outcome, committed_round=committed,
                       checkpoint_sha256=checkpoint, state_sha256=state_sha256, ended_at=now())
        self.save()
        self.release(attempt)

    def release(self, attempt):
        job = read_resource('job', attempt['job'], '--ignore-not-found')
        if job is not None:
            if job['metadata']['uid'] != attempt['uid']:
                raise ValueError('refusing to release a Job with a different UID')
            run([*BASE, 'delete', 'job', attempt['job'], '--ignore-not-found', '--wait=false'])
        attempt['status'] = 'RELEASED'
        outcome = attempt['outcome']
        if outcome == 'ADVANCE':
            self.ledger['segment'] += 1
            self.ledger['consecutive_failures'] = 0
        elif outcome == 'RETRY':
            self.ledger['consecutive_failures'] += 1
            if self.ledger['consecutive_failures'] >= ATTEMPT_LIMIT:
                self.stop('CONSECUTIVE_FAILURE_LIMIT')
        else:
            self.stop(outcome)
        self.save()

    def work(self):
        failures = 0
        while self.ledger['status'] == 'RUNNING':
            try:
                if self.ledger['segment'] == len(self.segments):
                    if 'material' in self.packet:
                        self.advance_producer()
                    if self.ledger['status'] == 'RUNNING' and (
                            'material' not in self.packet or self.material_complete()):
                        self.finish()
                        break
                else:
                    segment = self.segments[self.ledger['segment']]
                    attempts = self.ledger['attempts']
                    active = attempts[-1] if attempts and attempts[-1]['status'] != 'RELEASED' else None
                    if active is None:
                        self.submit(segment)
                    elif active['status'] == 'RELEASING':
                        self.release(active)
                    else:
                        self.reconcile(segment, active)
                    if self.ledger['status'] == 'RUNNING' and 'material' in self.packet:
                        self.advance_producer()
                failures = 0
            except (RuntimeError, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
                failures += 1
                self.ledger['last_error'] = dict(at=now(), detail=str(error), consecutive=failures)
                self.save()
                print(self.ledger['last_error'], file=sys.stderr, flush=True)
                if 'Forbidden' in str(error) or 'Unauthorized' in str(error) or failures >= 6:
                    # The operator can resume with this ledger after access or observation recovers.
                    return 1
                time.sleep(600)
                continue
            if self.ledger['status'] == 'RUNNING':
                time.sleep(TICK_SECONDS)
        return int(self.ledger['status'] != 'COMPLETE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True, help='Scott-approved packet digest and GPU-second budget')
    parser.add_argument('--release-signature', type=Path, required=True, help='Scott’s detached SSH release signature')
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    data = args.packet.read_bytes()
    packet = json.loads(data)
    release_data = args.release.read_bytes()
    signers = Path.home() / '.config' / 'pokezero' / '100m-release-allowed-signers'
    ledger_path = args.ledger.resolve()
    release = verify_signed_release(release_data, args.release_signature, signers, data, ledger_path)
    if not all(OPAQUE_GPU.fullmatch(row['output']) for row in [packet['main'], *packet['children']]):
        parser.error('expected opaque GPU output roots')
    if len({row['output'] for row in [packet['main'], *packet['children']]}) != 1 + len(packet['children']):
        parser.error('each campaign lineage needs a distinct output root')
    for row in [packet['main'], *packet['children']]:
        if not EXEC_MARGIN_SECONDS < row['min_exec_seconds'] <= row['max_exec_seconds'] <= 6 * 86400:
            parser.error('segment execution deadline outside the launcher envelope')
    if 'material' in packet:
        cfg = packet['material']['producer']
        memory = re.fullmatch(r'([1-9][0-9]*)(Mi|Gi)', cfg['memory'])
        if (not 1 <= cfg['lookahead'] or not 1 <= cfg['workers'] <= cfg['cpu'] <= 16
                or not 1 <= cfg['max_exec_seconds'] <= 7200 or not cfg['nodes']
                or not memory or int(memory[1]) > (128 if memory[2] == 'Gi' else 131072)):
            parser.error('invalid bounded CPU producer configuration')
    verify_source(packet)
    release_binding = hashlib.sha256(release_data).hexdigest()
    with ledger_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return Supervisor(packet, ledger_path, release_binding, release['gpu_seconds']).work()


if __name__ == '__main__':
    sys.exit(main())
