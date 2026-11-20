"""Monitor staged Showdown milestone requests and own only their eval jobs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

from . import v5_deferred_evaluation as deferred, v5_progress_evaluation as panels
from . import v5_progress_report as report
from .v5_evaluation import Artifact


def _artifact(ref):
    return Artifact(**ref)


def _write(path, value):
    report._atomic(path, json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def _read(path, default):
    return json.loads(path.read_bytes()) if path.is_file() else default


def _observe_job(name):
    argv = ['kubectl', '--context', 'jokla', '-n', 'scott', 'get', 'job',
            'scott-' + name, '--ignore-not-found', '-o', 'json']
    result = subprocess.check_output(argv, text=True, timeout=45)
    if not result.strip():
        return None
    job = json.loads(result)
    if job['metadata']['labels']['v2-run'] != name:
        raise ValueError('launcher ownership label differs')
    terminal = next((c for c in job.get('status', {}).get('conditions', [])
                     if c['type'] in ('Complete', 'Failed') and c['status'] == 'True'), {})
    return job['metadata']['uid'], terminal.get('type'), terminal.get('reason')


def _release_job(name, uid):
    observed = _observe_job(name)
    if observed is None:
        return
    if observed[0] != uid or observed[1] is None:
        raise ValueError('refusing to release a foreign or live job')
    subprocess.run(['kubectl', '--context', 'jokla', '-n', 'scott', 'delete', 'job',
                    'scott-' + name, '--ignore-not-found', '--wait=false'], check=True, timeout=45)


def _launch(config, dispatch, opponent, shard, name):
    executable = _artifact(config['launcher']).verify()
    protocol = config['protocol']
    resources = config['resources'][opponent]
    lanes, peak = resources['lanes'], resources['measured_peak_bytes']
    if type(lanes) is not int or not 1 <= lanes <= 4 or type(peak) is not int or peak <= 0:
        raise ValueError('measured job resources required before automatic admission')
    memory_mib = math.ceil(13 * peak / (10 * 2**20))
    if memory_mib > 128 * 1024:
        raise ValueError('measured shard peak plus headroom exceeds the CPU job cap')
    wall_seconds = resources['wall_seconds']
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 7200:
        raise ValueError('finite admitted shard wall time required')
    entry = shlex.join(['-m', 'pokezero.v5_progress_evaluation',
        '--dispatch', dispatch.path, '--dispatch-sha256', dispatch.sha256,
        '--protocol', protocol['path'], '--protocol-sha256', protocol['sha256'],
        '--registry', config['registry'], 'evaluate', '--opponent', opponent,
        '--shard', str(shard), '--lanes', str(lanes), '--wall-seconds', str(wall_seconds)])
    env = {key: value for key, value in os.environ.items() if not key.startswith('SLACK_')}
    return subprocess.run([sys.executable, str(executable), '--clone', config['source_checkout'],
        '--context', 'jokla', '--gpus', '0', '--cpu', str(4*lanes), '--memory', f'{memory_mib}Mi',
        '--name', name, '--entry', entry], env=env, timeout=90, check=False).returncode


def _message(row):
    fragments = []
    for name, label in (('max_damage', 'MD'), ('champion_9375', 'H2H'), ('foul_play', 'FP')):
        result = row['panel'][name]
        rate = result['win_rate']
        seen = f'{result["games_completed"]}/{result["games_declared"]} completed'
        score = 'unavailable' if rate is None else f'{rate:.1%}'
        fragments.append(f'{label}: {score} ({seen})')
    return f'Showdown at {row["declared_games"]:,} games · ' + ' · '.join(fragments)


def _slack_post(channel, row, receipt_id):
    # Use the same bot authentication as the foundation monitor. Keep the
    # credential in the monitor process and avoid reflecting response bodies.
    payload = dict(channel=channel, text=_message(row), client_msg_id=receipt_id,
                   unfurl_links=False, unfurl_media=False)
    request = Request('https://slack.com/api/chat.postMessage', method='POST',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + os.environ['SLACK_BOT_TOKEN']})
    with urlopen(request, timeout=20) as response:
        result = json.load(response)
    if result.get('ok') is not True:
        raise ValueError('Slack rejected the milestone post')
    return str(result['ts'])


def _notify(folder, row, channel, send):
    destination = folder / 'slack'
    destination.mkdir(exist_ok=True)
    identity = str(uuid5(NAMESPACE_URL, f'pokezero:{row["dispatch_sha256"]}:{channel}'))
    attempt = destination / f'{identity}.attempt.json'
    receipt = destination / f'{identity}.receipt.json'
    if attempt.exists():
        return
    deferred._publish_equal(attempt, dict(milestone=row['declared_games'], client_msg_id=identity))
    try:
        ts = send(channel, row, identity)
    except Exception as error:
        _write(receipt, dict(status='UNCONFIRMED', reason=type(error).__name__))
        # The request may have reached Slack. Reconcile before another post.
        return
    _write(receipt, dict(status='DELIVERED', ts=ts))


def _observe_owned(state, observe, release, log):
    now = time.time()
    for record in state['attempts']:
        if record['state'] == 'CLOSED':
            continue
        try:
            actual = observe(record['name'])
            if actual is None:
                # Retain a full startup/reconcile window after an ambiguous
                # launch. Observer errors never release jobs.
                if now-record['time'] >= 900:
                    record.update(state='CLOSED', reason='JobMissingAfterObservationWindow')
                continue
            uid, terminal, reason = actual
            if record.get('uid', uid) != uid:
                raise ValueError('owned job UID changed')
            record['uid'] = uid
            if terminal is not None:
                release(record['name'], uid)
                record.update(state='CLOSED', terminal=terminal, reason=reason)
        except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
            log('job observation deferred: ' + type(error).__name__)


def tick(config, folder, *, allow_launch=False, channel=None, send=_slack_post,
         observe=_observe_job, release=_release_job, launch=_launch, log=print):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'monitor.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _tick(config, folder, allow_launch, channel, send, observe, release, launch, log)


def _tick(config, folder, allow_launch, channel, send, observe, release, launch, log):
    protocol = _artifact(config['protocol'])
    protocol.verify()
    root = Path(config['staged_run_root'])
    registry = config['registry']
    state_path = folder/'job-ledger.json'
    digest = deferred.run._sha(config)
    state = _read(state_path, {'config_sha256': digest, 'attempts': []})
    if state['config_sha256'] != digest:
        raise ValueError('monitor configuration changed')
    if allow_launch:
        _observe_owned(state, observe, release, log)
        _write(state_path, state)

    index = folder/'milestones'
    index.mkdir(exist_ok=True)
    summaries, frontier, backlog, finalized = [], 0, [], []
    queue_path = folder/'eval-queue.json'
    queue = _read(queue_path, {'exhausted': []})
    void = {row['declared_games']: row for row in queue.get('void', [])}
    def record_void(games, receipt):
        if void.get(games) != receipt:
            log(f'EVALUATION_VOID_ALERT games={games} '
                f"existing={receipt['existing_request_sha256']} proposed={receipt['proposed_request_sha256']}")
        void[games] = receipt

    for step in range(1, config['milestones']+1):
        games = step * panels.INTERVAL_GAMES
        request = root/'artifacts'/'cpu-panels'/str(games)/'request.json'
        try:
            receipt = deferred.read_eval_void(request, games)
        except (OSError, ValueError, KeyError) as error:
            log(f'EVALUATION_VOID_ALERT games={games} receipt unreadable: {error}')
            continue
        if receipt is not None:
            frontier = games
            record_void(games, receipt)
        if games in void or not request.is_file():
            continue
        frontier = games
        dispatch = Artifact(str(request), deferred.run._file_sha(request))
        target = index/f'{games}.json'
        if target.is_file():
            try:
                receipt = deferred.read_published_eval_void(dispatch, games)
            except (OSError, ValueError, KeyError, TypeError) as error:
                log(f'milestone observation deferred: {type(error).__name__}')
                continue
            if receipt is not None:
                record_void(games, receipt)
                continue
            row = json.loads(target.read_bytes())
            if row['dispatch_sha256'] != dispatch.sha256 or row['protocol_sha256'] != protocol.sha256:
                raise ValueError('completed milestone request changed')
        else:
            try:
                row = panels.summarize(dispatch, protocol, registry)
            except deferred.VoidEvaluation as error:
                record_void(games, error.receipt)
                continue
            if row['complete']:
                deferred._publish_equal(target, row)
            else:
                backlog.append(dispatch)
        summaries.append(row)
        if row['complete']:
            finalized.append(row)
    report.update_report(folder, summaries, training_games=frontier, void_games=void)
    # Materialize the small timeline atomically. Each finalized milestone is
    # included once, even if several monitors restart or replay their reads.
    report._atomic(folder/'eval-timeline.jsonl', ''.join(
        json.dumps(row, sort_keys=True, allow_nan=False)+'\n' for row in finalized))
    if channel:
        for row in finalized:
            _notify(folder, row, channel, send)

    exhausted = []
    if allow_launch:
        limit = config['max_active_jobs']
        max_attempts = config['max_attempts']
        if type(limit) is not int or limit < 1 or type(max_attempts) is not int or not 1 <= max_attempts <= 999:
            raise ValueError('positive job slot and bounded retry count required')
        slots = limit - sum(item['state'] != 'CLOSED' for item in state['attempts'])
        for dispatch in backlog:
            dispatch_attempts = [a for a in state['attempts'] if a['key'][0] == dispatch.sha256]
            if slots <= 0 and len(dispatch_attempts) < max_attempts:
                continue
            try:
                work = panels.pending_shards(dispatch, protocol, registry)
            except deferred.VoidEvaluation as error:
                record_void(error.receipt['declared_games'], error.receipt)
                continue
            for item in sorted(work, key=lambda x: (x['shard'], x['opponent'])):
                opponent, shard = item['opponent'], item['shard']
                key = (dispatch.sha256, opponent, shard)
                attempts = [a for a in dispatch_attempts if tuple(a['key']) == key]
                if any(a['state'] != 'CLOSED' for a in attempts):
                    continue
                if len(attempts) >= max_attempts:
                    last = attempts[-1]
                    exhausted.append(dict(item, dispatch_sha256=dispatch.sha256,
                        status='EXHAUSTED', attempts=len(attempts), last_job=last['name'],
                        last_terminal=last.get('terminal'), last_reason=last.get('reason'),
                        last_launcher_exit_code=last.get('launcher_exit_code'),
                        last_observation_error=last.get('observation'),
                        action='Inspect the last shard Job receipt and reconcile its pending cells.'))
                    continue
                if slots <= 0:
                    continue
                name = f'v2q-{hashlib.sha256(repr(key).encode()).hexdigest()[:12]}-cpu-r{len(attempts)+1}'
                record = dict(key=key, name=name, time=time.time(), state='ADMITTING')
                state['attempts'].append(record)
                _write(state_path, state)
                try:
                    record['launcher_exit_code'] = launch(config, dispatch, opponent, shard, name)
                except (OSError, subprocess.SubprocessError) as error:
                    record['observation'] = type(error).__name__
                _write(state_path, state)
                slots -= 1
        queue['exhausted'] = exhausted
    if void:
        queue['void'] = [void[games] for games in sorted(void)]
    if allow_launch or void:
        _write(queue_path, queue)
    return dict(finalized=len(finalized), queued=len(backlog), published_games=frontier,
                exhausted_shards=len(exhausted))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--slack', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--monitor-seconds', type=int, default=0, help='0 performs one tick')
    args = parser.parse_args(argv)
    config = Artifact(str(args.config.resolve()), args.config_sha256).document()
    channel = os.environ.get('SLACK_EVAL_CHANNEL') if args.slack else None
    if args.slack and (not channel or not os.environ.get('SLACK_BOT_TOKEN')):
        parser.error('SLACK_EVAL_CHANNEL and SLACK_BOT_TOKEN required for --slack')
    if args.poll_seconds < 10 or args.monitor_seconds < 0:
        parser.error('invalid polling interval or deadline')
    end = time.monotonic() + args.monitor_seconds
    while True:
        try:
            print(json.dumps(tick(config, args.directory, allow_launch=args.launch, channel=channel)))
        except (OSError, subprocess.SubprocessError) as error:
            print(f'monitor observation deferred: {type(error).__name__}', file=sys.stderr)
        remaining = end - time.monotonic()
        if remaining <= 0:
            return 0
        time.sleep(min(remaining, args.poll_seconds))


if __name__ == '__main__':
    raise SystemExit(main())
