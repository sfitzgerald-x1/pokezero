"""Supervise one released V2 campaign through the approved single-node launcher."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


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


def shared_file(root, relative, *, line=None):
    path = f'{OUT_HOST}/{root}/{relative}'
    script = 'if test -s "$1"; then cat "$1"; fi'
    if line is not None:
        script = 'test -s "$1" && sed -n "$2{p;q;}" "$1"'
    output = run([*BASE, 'exec', 'eigen-stage-0', '-c', 'stage', '--', 'sh', '-c',
                  script, 'sh', path, str(line or 0)])
    return json.loads(output) if output.strip() else None


def completed_rounds(doc):
    if doc is None:
        return 0
    ledger = doc['round_ledger']
    count = ledger['completed_rounds']
    if type(count) is not int or count < 1 or ledger['latest_snapshot']['iteration'] != count - 1:
        raise ValueError('invalid committed round pointer')
    return count


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


def disposition(observed, attempt, committed, stop, *, at):
    if observed is None:
        return 'AMBIGUOUS' if at - epoch(attempt['submitted_at']) >= ADMISSION_SECONDS else None
    if not attempt.get('ever_admitted') and not observed['admitted']:
        if at - epoch(attempt['submitted_at']) >= ADMISSION_SECONDS:
            return 'ADMISSION_EXPIRED'
    if observed['live'] or not (observed['complete'] or observed['failed']):
        return None
    if committed == stop:
        return 'ADVANCE'
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
    worker_seconds = attempt['cap_seconds'] - EXEC_MARGIN_SECONDS
    entry = [
        '-m', 'gpu_engine.v2_distributed', '--ranks', '4', '--timeout-seconds', str(worker_seconds),
        '--collective-timeout', '1800', '--s0-idle-guard', '--out', '/out',
        '--declaration', declaration['path'], '--declaration-sha256', declaration['sha256'],
        '--runtime', runtime['path'], '--runtime-sha256', runtime['sha256'],
        '--plan-arm', 'B0', '--reference-run', packet['reference_run'],
        '--max-iterations', str(segment['stop']), '--lanes', '2048', '--row-order', 'file',
        '--torch-memory-fraction', '.45', '--compile-threads', '4', '--recipe', 'R2x1', '--mode', 'cudagraphs',
        '--learner-precision', 'float32', '--native-embedding-backward', '--fallback-random',
        '--profile', 'v2-production', '--campaign-anneal', '--lr-start', 'sqrt',
        '--lr-warmup-updates', '24000', '--freeze-gc-after-warmup', '--audit-every', '50',
    ]
    if main:
        if not resume:
            raise ValueError('the main campaign must start from the adopted 2M parent')
        entry.extend(['--adopt-2m-prefix-runtime', packet['previous_runtime']])
    else:
        entry.extend(['--cooldown-child', f"{OUT_HOST}/{packet['main']['output']}",
                      '--cooldown-parent-sha256', parent_checkpoint])
    options = ['--resume-run', output] if resume else ['--require-fresh']
    argv = [
        sys.executable, packet['launcher']['path'], '--clone', packet['checkout'],
        '--context', 'potomac', '--name', name, '--gpus', '4',
        '--cpu', str(packet['cpu']), '--memory', packet['memory'],
        '--kueue', '--queue', 'medium', '--admission-hours', '6',
        '--max-exec-seconds', str(attempt['cap_seconds']), '--trace-jax-cache',
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

    def save(self):
        partial = self.path.with_suffix('.partial')
        partial.write_text(json.dumps(self.ledger, indent=2) + '\n')
        partial.replace(self.path)

    def stop(self, outcome):
        self.ledger.update(status=outcome, finished_at=now())
        self.save()

    def submit(self, segment):
        remaining = self.budget - self.ledger['reserved_gpu_seconds']
        cap = min(segment['max_exec_seconds'], remaining // GPU_COUNT)
        if cap < segment['min_exec_seconds']:
            self.stop('GPU_BUDGET_EXHAUSTED')
            return
        doc = shared_file(segment['output'], 'state.json')
        before = completed_rounds(doc)
        if before >= segment['stop']:
            raise ValueError('unexpected committed segment without a reconciled attempt')
        parent_checkpoint = None
        if 'parent_round' in segment:
            main_output = self.packet['main']['output']
            parent_round = segment['parent_round']
            if completed_rounds(shared_file(main_output, 'state.json')) < parent_round:
                raise ValueError('cooldown parent has not committed')
            snapshot = shared_file(main_output, 'reports.jsonl', line=parent_round)['report']['snapshot']
            if snapshot['iteration'] != parent_round - 1:
                raise ValueError('cooldown parent is outside its committed round')
            parent_checkpoint = snapshot['checkpoint_sha256']
        attempt = dict(number=len(self.ledger['attempts'])+1, segment=self.ledger['segment'],
                       before_round=before, submitted_at=now(), cap_seconds=cap, status='SUBMITTING')
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
        committed = completed_rounds(shared_file(segment['output'], 'state.json'))
        if committed < attempt['before_round']:
            raise ValueError('committed pointer regressed below the attempt start')
        if committed > attempt['before_round'] and 'first_commit' not in attempt:
            attempt['first_commit'] = dict(at=now(), round=committed)
        outcome = disposition(observed, attempt, committed, segment['stop'], at=time.time())
        self.save()
        if outcome is None:
            return
        if outcome == 'AMBIGUOUS':
            self.stop('ADMISSION_UNRECONCILED')
            return
        # Persist the terminal reconciliation before deleting the confirmed Job.
        attempt.update(status='RELEASING', outcome=outcome, committed_round=committed, ended_at=now())
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
            if self.ledger['segment'] == len(self.segments):
                self.stop('COMPLETE')
                break
            segment = self.segments[self.ledger['segment']]
            attempts = self.ledger['attempts']
            active = attempts[-1] if attempts and attempts[-1]['status'] != 'RELEASED' else None
            try:
                if active is None:
                    self.submit(segment)
                elif active['status'] == 'RELEASING':
                    self.release(active)
                else:
                    self.reconcile(segment, active)
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
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    data = args.packet.read_bytes()
    packet = json.loads(data)
    release = json.loads(args.release.read_bytes())
    if (release['packet_sha256'] != hashlib.sha256(data).hexdigest()
            or type(release['gpu_seconds']) is not int or release['gpu_seconds'] <= 0):
        parser.error('release does not authorize this packet and finite GPU budget')
    if not all(OPAQUE_GPU.fullmatch(row['output']) for row in [packet['main'], *packet['children']]):
        parser.error('expected opaque GPU output roots')
    if len({row['output'] for row in [packet['main'], *packet['children']]}) != 1 + len(packet['children']):
        parser.error('each campaign lineage needs a distinct output root')
    for row in [packet['main'], *packet['children']]:
        if not EXEC_MARGIN_SECONDS < row['min_exec_seconds'] <= row['max_exec_seconds'] <= 6 * 86400:
            parser.error('segment execution deadline outside the launcher envelope')
    verify_source(packet)
    release_binding = hashlib.sha256(args.release.read_bytes()).hexdigest()
    with args.ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return Supervisor(packet, args.ledger, release_binding, release['gpu_seconds']).work()


if __name__ == '__main__':
    sys.exit(main())
