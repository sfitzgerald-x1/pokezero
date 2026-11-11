"""Stored-only recovery of drained d3 workers. Never signals or launches.

External process evidence is an operations input, like the existing closure
receipt. It must retain the actual stopped controller and unreaped child proc
records: a saved game document alone does not establish successful child exit.
"""
from copy import deepcopy
import json
from pathlib import Path

from . import owning_encoder_continuation as c
from . import owning_encoder_study as s


SCHEMA = 'pokezero.B3-drained-worker-reconciliation.v1'
PROCESS_ID = ('pid', 'ppid', 'pgid', 'sid', 'start_ticks', 'args')
TERMINAL_ID = ('pid', 'ppid', 'pgid', 'sid', 'start_ticks')


def _terminal_process(before, after):
    """Linux zombies lose cmdline; retain pre-exit argv, verify actual stat."""
    literal = after.get('stat_literal')
    s.require(type(literal) is str and '(' in literal and ')' in literal,
        'literal original terminal proc stat required')
    prefix, _, suffix = literal.rpartition(')')
    pid = int(prefix.split('(',1)[0].strip())
    fields = suffix.split()
    s.require(len(fields) >= 50, 'complete Linux proc stat including wait status required')
    values = dict(pid=pid,state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),
        sid=int(fields[3]),start_ticks=int(fields[19]),exit_code=int(fields[49]))
    s.require(all(after.get(k) == v for k,v in values.items())
        and all(after[k] == before[k] for k in TERMINAL_ID)
        and after['state'] == 'Z' and type(after['exit_code']) is int
        and after.get('args') == [],
        'same unreaped terminal child, empty zombie cmdline and exact literal wait status required')


def reconcile(*, plan, ledger, quiesce, terminal):
    refs = {k:s._reference(v) for k,v in dict(plan=plan,ledger=ledger,
        quiesce=quiesce,terminal=terminal).items()}
    p, old, pause, end = (s.read_bound(refs[k]) for k in ('plan','ledger','quiesce','terminal'))
    c.validate_plan(p)
    spec = s.read_bound(p['references']['prespec'])
    c._validate_ledger(spec, old)
    s.require(pause['kind'] == 'OWNED_CONTROLLER_ONLY_SIGSTOP_NO_WORKER_SIGNAL'
        and pause['ledger_sha256'] == refs['ledger']['sha256']
        and pause['dispatch_order'] == old['continuation']['dispatch_order']
        and old['continuation']['plan_sha256'] == refs['plan']['sha256'],
        'exact paused ledger/plan/dispatch evidence required')
    controller = pause['controller_stopped']
    s.require(controller['state'] == 'T'
        and all(pause['controller_before'][k] == controller[k] for k in PROCESS_ID)
        and end['schema'] == SCHEMA and end['controller_closed'] is True
        and end['quiesce_sha256'] == refs['quiesce']['sha256']
        and end['ledger_sha256'] == refs['ledger']['sha256']
        and end['plan_sha256'] == refs['plan']['sha256']
        and end['controller_pid'] == controller['pid']
        and end['controller_start_ticks'] == controller['start_ticks']
        and c._finite(end['observed_epoch']) and end['observed_epoch'] >= pause['observed_epoch'],
        'closed exact stopped controller and bound terminal evidence required')
    position = controller['args'].index('run')
    s.require(controller['args'][position-2:position] == ['-m','pokezero.owning_encoder_continuation']
        and controller['args'][position+2] == refs['plan']['sha256'], 'controller plan SHA differs')
    original_output = Path(controller['args'][position+3])
    current = [i for i in old['continuation']['dispatch_order'] if old['cells'][i]['status'] == 'ATTEMPTED']
    s.require([dict(index=i,attempt=old['cells'][i]['attempt']) for i in current] == pause['current_attempted']
        and type(end['workers']) is list,
        'exact original attempted roster required')
    workers = {}
    for worker in pause['workers_before']:
        argv = worker['args'];at = argv.index('worker')
        index = int(argv[at+2])
        s.require(index in current and index not in workers
            and argv[at-2:at] == ['-m','pokezero.owning_encoder_continuation']
            and argv[at+1] == str(original_output/'plan.json')
            and argv[at+5] == str(original_output/f'cell-{index}.json')
            and [int(argv[at+3]),int(argv[at+4])] == old['cells'][index]['attempt']['cpu_pair']
            and worker['ppid'] == controller['pid']
            and worker['pid'] == worker['pgid'] == worker['sid'],
            'original worker command/core/process identity required')
        workers[index] = worker
    endings = {item['index']:item for item in end['workers']}
    s.require(len(endings) == len(end['workers']) and set(endings) == set(current)
        and set(workers) == set(current), 'all and only originally attempted workers required')
    result = deepcopy(old);retained = []
    for index in current:
        item = endings[index];before = workers[index];after = item['process']
        _terminal_process(before, after)
        s.require(item['remaining_group_processes'] == [], 'fully drained owned process group required')
        receipt_ref,marker_ref = s._reference(item['receipt']),s._reference(item['attempted_marker'])
        receipt,marker = s.read_bound(receipt_ref),s.read_bound(marker_ref)
        s.require(Path(receipt_ref['path']).name == f'cell-{index}.json'
            and Path(marker_ref['path']).name == f'cell-{index}.attempted'
            and marker == dict(index=index,plan_sha256=refs['plan']['sha256'])
            and receipt.get('index') == index, 'exclusive original attempt/result identities required')
        row = result['cells'][index]
        if after['exit_code'] == 0:
            s.require(receipt.get('cpu_pair') == row['attempt']['cpu_pair'], 'original worker pair differs')
            checked = c.checked_result(receipt['document'], spec, index)
            timing = receipt['timing']
            s.require(set(timing) == {'started_epoch','completed_epoch','duration_seconds'}
                and all(c._finite(x) for x in timing.values())
                and row['attempt']['started_epoch'] <= timing['started_epoch'] <= timing['completed_epoch'] <= end['observed_epoch'],
                'original actual completed-game timing required')
            row.update(checked,timing=timing,worker_precision=receipt.get('precision'))
            retained.append(index)
        else:
            # No inference from apparent game output after a failed child exit.
            row['status'] = 'UNKNOWN_ATTEMPTED'
        row['drain_recovery'] = dict(receipt=receipt_ref,attempted_marker=marker_ref,
            terminal_evidence=refs['terminal'],original_wait_status=after['exit_code'])
    for i,cell in enumerate(old['cells']):
        if i not in current:s.require(s.canonical(cell) == s.canonical(result['cells'][i]), 'existing cell changed')
    result.update(status='FAILED',actual_exit=1,B3_qualified=False,
        failure=dict(error_type='DrainedControllerClosedForWorkerReuse',no_retry=True))
    result['recorded_games'] = sum(x['status'] == 'COMPLETE' for x in result['cells'])
    for invocation in result['invocations']:
        rows = [x for x in result['cells'] if x['invocation'] == invocation['ordinal']]
        invocation['status'] = 'COMPLETE' if all(x['status']=='COMPLETE' for x in rows) else 'FAILED'
    result['reconciliation'] = dict(schema=SCHEMA,references=refs,retained_completed_indices=retained,
        unknown_attempted_indices=[i for i in current if i not in retained],
        scientific_rerun=False,original_files_unchanged=True)
    c._validate_ledger(spec,result)
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input');parser.add_argument('out')
    args = parser.parse_args(argv)
    output = Path(args.out)
    s.require(not output.exists(), 'new reconciliation output required')
    document = reconcile(**json.loads(Path(args.input).read_bytes()))
    with output.open('xb') as stream:
        import os
        stream.write(s.canonical(document)+b'\n');stream.flush();os.fsync(stream.fileno())
    s._sync_directory(output.parent)
    return 0


if __name__ == '__main__':raise SystemExit(main())
