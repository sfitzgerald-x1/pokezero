import json
from pathlib import Path
import tempfile
import unittest

from pokezero import v5_training_diagnostics as d


def update(step, epoch, **extra):
    return dict(optimizer_step=step, learning_rate_epoch=epoch, owner_rows=8,
        entropy=.4, entropy_count=8, policy_loss=.2, policy_loss_count=4,
        value_loss=.3, value_loss_count=6, gradient_norm=.1, **extra)


class TrainingDiagnosticsTests(unittest.TestCase):
    def write(self, root, *, first=None, second=None, missing=False, duplicate=None):
        rows=[]
        def event(kind, **payload):
            v=dict(kind=kind, payload=payload, sequence=len(rows), previous=rows[-1]['sha256'] if rows else None, time=len(rows))
            v['sha256']=d._digest(v);rows.append(v)
        event('RUN_STARTED', declaration_sha256='a'*64)
        event('CHECKPOINT', iteration=-1, steps=0, rows=0, path='initial.pt', sha256='0'*64)
        event('TRAIN_PROGRESS', iteration=0, steps=2, rows=16, updates=([update(1,1)] if missing else (first or [update(1,1),update(2,5)])))
        event('CHECKPOINT', iteration=0, steps=2, rows=16, path='zero.pt', sha256='b'*64)
        event('SNAPSHOT', iteration=0, checkpoint='zero.pt', checkpoint_sha256='b'*64)
        if duplicate is not None:event('TRAIN_PROGRESS', iteration=0, steps=2, rows=16, updates=[duplicate])
        event('TRAIN_PROGRESS', iteration=1, steps=4, rows=32, updates=second or [update(3,1),update(4,5)])
        event('CHECKPOINT', iteration=1, steps=4, rows=32, path='one.pt', sha256='c'*64)
        event('SNAPSHOT', iteration=1, checkpoint='one.pt', checkpoint_sha256='c'*64)
        # A future cohort must not enter this fixed prefix.
        event('TRAIN_PROGRESS', iteration=2, steps=5, rows=40, updates=[update(5,1)])
        for v in rows:(root/(str(v['sequence']).zfill(9)+'.json')).write_text(json.dumps(v))
        return root

    def census(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            return d.summarize(self.write(Path(tmp), **kwargs))

    def test_fixed_prefix_and_counts(self):
        v=self.census();self.assertTrue(v['diagnostic_capture_complete']);self.assertEqual(v['captured_unique_updates'],4)
        self.assertEqual(v['checkpoint_row_visits'],32);self.assertEqual(v['epochs'][0]['value_loss']['target_fraction'],.75)
        self.assertEqual(v['epochs'][1]['entropy']['mean_batch_metric'],None)
        self.assertEqual(v['decision'],'DESCRIPTIVE_ONLY_NO_AUTOMATIC_CYCLE_KILL_OR_PASS')

    def test_missing_updates_are_not_inferred(self):
        v=self.census(missing=True);self.assertEqual(v['updates_without_diagnostics'],1)
        self.assertEqual(v['row_visits_without_diagnostics'],8);self.assertFalse(v['diagnostic_capture_complete'])

    def test_missing_epoch_and_count_stay_unknown(self):
        row=update(1,1);del row['learning_rate_epoch'];del row['value_loss_count']
        v=self.census(first=[row,update(2,5)]);self.assertEqual(v['updates_without_epoch'],1)
        self.assertEqual(v['unknown_epoch_diagnostics']['value_loss']['owner_rows_with_unknown_count'],8)
        self.assertIsNone(v['unknown_epoch_diagnostics']['value_loss']['target_fraction'])

    def test_identical_duplicate_not_double_counted(self):
        v=self.census(duplicate=update(1,1));self.assertEqual(v['captured_unique_updates'],4)
        self.assertEqual(v['duplicate_identical_update_records'],1)

    def test_conflicting_duplicate_refuses(self):
        row=update(1,1);row['entropy']=.9
        with self.assertRaisesRegex(ValueError,'conflicting'):self.census(duplicate=row)

    def test_empty_loss_and_zero_entropy_not_fabricated_loss_or_diagnosis(self):
        row=update(1,1);row.update(entropy=0, policy_loss=0, policy_loss_count=0, value_loss=0, value_loss_count=0, gradient_norm=0)
        v=self.census(first=[row,update(2,5)]);group=v['epochs'][0]
        self.assertIsNone(group['value_loss']['mean_batch_metric']);self.assertEqual(group['entropy']['zero_metric_updates'],1)
        self.assertEqual(group['gradient_norm']['zero_updates'],1);self.assertIn('NO_AUTOMATIC',v['decision'])

    def test_invalid_count_and_nonfinite_metric_refuse(self):
        for change in [dict(value_loss_count=9),dict(entropy=float('nan'))]:
            row=update(1,1);row.update(change)
            with self.assertRaises(ValueError):self.census(first=[row,update(2,5)])

    def test_changed_hash_chain_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.write(Path(tmp));p=root/'000000001.json';v=json.loads(p.read_text());v['payload']['iteration']=1;p.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'hash chain'):d.summarize(root)

    def test_progress_envelope_counters_bind_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.write(Path(tmp));previous=None
            for p in sorted(root.glob('*.json')):
                e=json.loads(p.read_text());e.pop('sha256')
                if e['kind']=='TRAIN_PROGRESS' and e['payload']['iteration']==0:e['payload']['rows']=15
                e['previous']=previous;e['sha256']=d._digest(e);previous=e['sha256'];p.write_text(json.dumps(e))
            with self.assertRaisesRegex(ValueError,'envelope'):d.summarize(root)

    def test_tagged_entropy_keeps_count_but_no_mean_or_zero(self):
        row=update(1,1);row['entropy']=dict(kind='NONFINITE_DIAGNOSTIC',classification='NAN',sign='negative')
        v=self.census(first=[row,update(2,5)]);s=v['epochs'][0]['entropy']
        self.assertEqual(s['known_target_rows'],8);self.assertEqual(s['nonfinite_classifications'],{'NAN:negative':1})
        self.assertIsNone(s['mean_batch_metric']);self.assertEqual(s['zero_metric_updates'],0)
        self.assertFalse(v['diagnostic_capture_complete']);self.assertTrue(v['update_identity_capture_complete'])


if __name__=='__main__':unittest.main()
