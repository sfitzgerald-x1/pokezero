"""Descriptive first-two-cohort update census; no model, game or kill decision.

Batch metrics have already applied their original training weights. Their
means below are means of recorded nonempty batch metrics, not reconstructed
population-state means. Legal-support mix is not present in these journals;
zero entropy alone does not establish collapse or a training cycle.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('nonnegative integer diagnostic counter required')
    return value


def _metric(value):
    if isinstance(value, dict):
        if (set(value) != {'kind', 'classification', 'sign'} or value['kind'] != 'NONFINITE_DIAGNOSTIC'
                or value['classification'] not in ('NAN', 'INFINITY') or value['sign'] not in ('positive', 'negative')):
            raise ValueError('original tagged unavailable diagnostic required')
        return dict(value)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite recorded batch metric required')
    return float(value)


def _mean(values):
    return None if not values else math.fsum(v/len(values) for v in values)


def _nonfinite(rows, name):
    values = [r[name] for r in rows if isinstance(r.get(name), dict)]
    return dict(nonfinite_updates=len(values), nonfinite_classifications=dict(Counter(
        v['classification'] + ':' + v['sign'] for v in values)))


def _group(rows):
    owner_rows = sum(r['owner_rows'] for r in rows)
    result = dict(captured_updates=len(rows), known_owner_rows=owner_rows)
    for name in ('entropy', 'policy_loss', 'value_loss'):
        available = [r for r in rows if name + '_count' in r]
        count = sum(r[name + '_count'] for r in available)
        denominator = sum(r['owner_rows'] for r in available)
        nonempty = [r for r in available if r[name + '_count'] > 0 and name in r]
        values = [r[name] for r in nonempty if not isinstance(r[name], dict)]
        result[name] = dict(known_target_rows=count, owner_rows_with_known_count=denominator,
            owner_rows_with_unknown_count=owner_rows-denominator,
            target_fraction=None if not denominator else count/denominator,
            nonempty_metric_updates=len(values), nonempty_updates_missing_metric=sum(
                r[name + '_count'] > 0 and name not in r for r in available),
            mean_batch_metric=_mean(values), zero_metric_updates=sum(v == 0 for v in values),
            **_nonfinite(rows, name))
    gradients = [r['gradient_norm'] for r in rows if 'gradient_norm' in r and not isinstance(r['gradient_norm'], dict)]
    result['gradient_norm'] = dict(known_updates=len(gradients),
        mean=_mean(gradients), maximum=None if not gradients else max(gradients),
        zero_updates=sum(v == 0 for v in gradients), missing_updates=sum('gradient_norm' not in r for r in rows),
        **_nonfinite(rows, 'gradient_norm'))
    return result


def summarize(journal):
    """All original hash-linked events through the first trained snapshot1."""
    journal = Path(journal)
    previous = None
    updates = {}
    duplicate_updates = 0
    checkpoints = []
    envelopes = []
    snapshots = {}
    ending = None
    for sequence, path in enumerate(sorted(journal.glob('[0-9]*.json'))):
        event = json.loads(path.read_bytes())
        digest = event.pop('sha256')
        if (path.stem != str(sequence).zfill(9) or event['sequence'] != sequence
                or event['previous'] != previous or _digest(event) != digest):
            raise ValueError('complete original journal hash chain required')
        previous = digest
        if sequence == 0 and event['kind'] != 'RUN_STARTED':
            raise ValueError('original run start required')
        payload = event['payload']
        if event['kind'] == 'CHECKPOINT':
            checkpoints.append(payload)
        elif event['kind'] == 'TRAIN_PROGRESS':
            iteration = payload['iteration']
            if type(iteration) is not int or iteration not in (0, 1):
                raise ValueError('original first two training cohorts only')
            envelopes.append(dict(iteration=iteration, steps=_integer(payload['steps'], 1),
                rows=_integer(payload['rows'], 1)))
            for original in payload.get('updates', []):
                if 'iteration' in original and original['iteration'] != iteration:
                    raise ValueError('update iteration differs from its original progress envelope')
                row = dict(original, iteration=iteration)
                step = _integer(row['optimizer_step'], 1)
                if step > payload['steps']:
                    raise ValueError('update follows its original progress envelope counter')
                _integer(row['owner_rows'], 1)
                epoch = row.get('learning_rate_epoch')
                if epoch is not None and (type(epoch) is not int or not 1 <= epoch <= 5):
                    raise ValueError('explicit original five-epoch index required')
                for name in ('entropy', 'policy_loss', 'value_loss'):
                    if name in row:
                        row[name] = _metric(row[name])
                    if name + '_count' in row:
                        count = _integer(row[name + '_count'])
                        if count > row['owner_rows']:
                            raise ValueError('target count exceeds original owner rows')
                if 'gradient_norm' in row:
                    row['gradient_norm'] = _metric(row['gradient_norm'])
                    if not isinstance(row['gradient_norm'], dict) and row['gradient_norm'] < 0:
                        raise ValueError('nonnegative recorded gradient norm required')
                if step in updates:
                    if updates[step] != row:
                        raise ValueError('conflicting original optimizer-step diagnostics')
                    duplicate_updates += 1
                else:
                    updates[step] = row
        elif event['kind'] == 'SNAPSHOT':
            iteration = payload['iteration']
            if type(iteration) is not int or iteration not in (-1, 0, 1):
                raise ValueError('original initial/first-two snapshot identity required')
            if iteration in snapshots:
                raise ValueError('one original trained publication per cohort required')
            snapshots[iteration] = payload
            if iteration == 1:
                ending = dict(event, sha256=digest)
                break
    if ending is None or 0 not in snapshots:
        raise ValueError('both actual trained/exported snapshots required before census')
    initial = [r for r in checkpoints if r['iteration'] == -1]
    if len(initial) != 1 or _integer(initial[0]['steps']) != 0 or _integer(initial[0]['rows']) != 0:
        raise ValueError('actual fresh-run initial checkpoint counters required')
    bound = {-1: initial[0]}
    for i in (0, 1):
        matches = [r for r in checkpoints if r['iteration'] == i and r['sha256'] == snapshots[i]['checkpoint_sha256']
                   and r['path'] == snapshots[i]['checkpoint']]
        if len(matches) != 1:
            raise ValueError('each trained snapshot must bind its actual checkpoint counters')
        bound[i] = matches[0]
        if (_integer(bound[i]['steps'], 1) <= bound[i-1]['steps']
                or _integer(bound[i]['rows'], 1) <= bound[i-1]['rows']):
            raise ValueError('positive original cohort step/row intervals required')
    total_steps, total_rows = bound[1]['steps'], bound[1]['rows']
    if any(step > total_steps for step in updates) or sum(r['owner_rows'] for r in updates.values()) > total_rows:
        raise ValueError('diagnostics exceed original final trained counters')
    rows = list(updates.values())
    cohorts = []
    for i in (0, 1):
        lo, hi = bound[i-1], bound[i]
        local = [r for r in rows if r['iteration'] == i]
        if any(not lo['steps'] < r['optimizer_step'] <= hi['steps'] for r in local):
            raise ValueError('optimizer identity falls outside its actual cohort interval')
        for envelope in (e for e in envelopes if e['iteration'] == i):
            if not (lo['steps'] < envelope['steps'] <= hi['steps'] and lo['rows'] < envelope['rows'] <= hi['rows']):
                raise ValueError('TRAIN_PROGRESS envelope falls outside its actual cohort interval')
            prefix = [r for r in local if r['optimizer_step'] <= envelope['steps']]
            prefix_unknown = envelope['steps']-lo['steps']-len(prefix)
            prefix_missing_rows = envelope['rows']-lo['rows']-sum(r['owner_rows'] for r in prefix)
            if prefix_missing_rows < prefix_unknown or (prefix_unknown == 0 and prefix_missing_rows != 0):
                raise ValueError('known update rows conflict with actual TRAIN_PROGRESS envelope')
        missing_steps = hi['steps']-lo['steps']-len(local)
        missing_rows = hi['rows']-lo['rows']-sum(r['owner_rows'] for r in local)
        if missing_rows < missing_steps or (missing_steps == 0 and missing_rows != 0):
            raise ValueError('known update rows conflict with actual cohort counters')
        cohorts.append(dict(iteration=i, start_step=lo['steps'], end_step=hi['steps'],
            declared_row_visits=hi['rows']-lo['rows'], captured_updates=len(local),
            updates_without_diagnostics=missing_steps, row_visits_without_diagnostics=missing_rows))
    unknown_epochs = [r for r in rows if r.get('learning_rate_epoch') is None]
    groups = [dict(iteration=i, epoch=e, **_group([r for r in rows if r['iteration'] == i and r.get('learning_rate_epoch') == e]))
              for i in (0, 1) for e in range(1, 6)]
    return dict(schema='pokezero.first-two-training-diagnostics.v1', journal=str(journal.resolve()),
        ending_snapshot=ending, checkpoint_steps=total_steps, checkpoint_row_visits=total_rows,
        captured_unique_updates=len(rows), updates_without_diagnostics=total_steps-len(rows),
        known_diagnostic_row_visits=sum(r['owner_rows'] for r in rows),
        row_visits_without_diagnostics=total_rows-sum(r['owner_rows'] for r in rows),
        duplicate_identical_update_records=duplicate_updates, cohorts=cohorts,
        updates_without_epoch=len(unknown_epochs), unknown_epoch_diagnostics=_group(unknown_epochs), epochs=groups,
        update_identity_capture_complete=(len(rows) == total_steps and not unknown_epochs
            and sum(r['owner_rows'] for r in rows) == total_rows),
        diagnostic_capture_complete=(len(rows) == total_steps and not unknown_epochs
            and sum(r['owner_rows'] for r in rows) == total_rows and all(
                all(n+'_count' in r and (r[n+'_count'] == 0 or n in r and not isinstance(r[n], dict))
                    for n in ('entropy', 'policy_loss', 'value_loss'))
                and 'gradient_norm' in r and not isinstance(r['gradient_norm'], dict) for r in rows)),
        decision='DESCRIPTIVE_ONLY_NO_AUTOMATIC_CYCLE_KILL_OR_PASS',
        scope='All emitted first-two-cohort diagnostics; missing updates/epochs/counts stay unknown. Means are recorded nonempty batch metrics, not population state metrics. No legal-support reconstruction, new games, model queries or training changes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.journal), allow_nan=False))


if __name__ == '__main__':
    main()
