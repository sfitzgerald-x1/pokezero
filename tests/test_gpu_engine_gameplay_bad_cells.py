"""Bounded recorded operands only; no model, engine, Native or sampler calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from gpu_engine import finite_gameplay_comparison as c
from test_gpu_engine_gameplay_saved_reference import fixture, SavedGameplayTests, Owner


def arrays():
    return {name:(np.ones(shape,np.bool_) if name.endswith('mask') else np.zeros(shape,np.float32 if name=='numeric_features' else np.int32))
            for name,shape in c.original.SHAPES.items()}


class BadCellTests(unittest.TestCase):
    def compare(self, actual, expected, enabled, eligible=True):
        calls=[]
        def observation():calls.append('single-original-encode');return expected
        boundary=dict(full=[{},{}],observations=[dict(admitted=True,arrays={k:v.tolist()for k,v in actual.items()})]*2)
        reference=dict(words=[],refused_lines=[],raw_kinds={})
        with patch.object(c,'captured_words',return_value=[]), \
             patch.object(c.original,'OwningShowdownEnv',side_effect=AssertionError('Native forbidden')), \
             patch.object(c,'ChanceControl',side_effect=AssertionError('chance forbidden')):
            result=c.compare_owner(boundary,0,reference,NS(observation=observation),eligible=eligible,record_bad_cells=enabled)
        return result,calls

    def test_single_encode_exact_operands_and_original_counts_unchanged(self):
        a,b=arrays(),arrays();a['numeric_features'][2,38]=.25;a['categorical_ids'][4,4]=7
        a['legal_action_mask'][7]=False
        default,calls=self.compare(a,b,False);self.assertEqual(len(calls),1)
        diagnostic,calls=self.compare(a,b,True);self.assertEqual(len(calls),1)
        cells=diagnostic['columns'].pop('bad_cells');self.assertEqual(diagnostic,default)
        self.assertEqual((cells['total'],cells['recorded'],cells['truncated']),(3,3,False))
        by={row['tensor']:row for row in cells['cells']}
        self.assertEqual(by['numeric_features'],dict(tensor='numeric_features',index=[2,38],gpu=.25,python=0.,absolute_error=.25))
        self.assertEqual(by['categorical_ids'],dict(tensor='categorical_ids',index=[4,4],gpu=7,python=0))
        self.assertEqual(by['legal_action_mask'],dict(tensor='legal_action_mask',index=[7],gpu=False,python=True))
        json.dumps(cells,allow_nan=False)

    def test_diagnostic_failure_never_erases_original_comparison_or_bad_counts(self):
        a,b=arrays(),arrays();a['numeric_features'][2,38]=.25
        expected,_=self.compare(a,b,False)
        with patch.object(c,'_bad_cell_diagnostics',side_effect=ValueError('UNIT diagnostic unavailable')):
            actual,calls=self.compare(a,b,True)
        self.assertEqual(len(calls),1);detail=actual['columns'].pop('bad_cells')
        self.assertEqual(detail['status'],'UNAVAILABLE');self.assertEqual(actual,expected)
        self.assertEqual(actual['columns']['numeric_bad'][38],1)

    def test_all21_records_bounded_truncation_and_exact_threshold(self):
        a,b=arrays(),arrays();a['numeric_features'].flat[:19]=1.;a['categorical_ids'].flat[:2]=3
        delta=np.abs(a['numeric_features'].astype(np.float64)-b['numeric_features'])
        all_cells=c._bad_cell_diagnostics(a,b,delta);self.assertEqual((all_cells['total'],all_cells['recorded']),(21,21))
        self.assertFalse(all_cells['truncated'])
        limited=c._bad_cell_diagnostics(a,b,delta,limit=3);self.assertEqual((limited['total'],limited['recorded'],limited['truncated']),(21,3,True))
        delta[0,0]=1e-6;self.assertEqual(c._bad_cell_diagnostics(a,b,delta)['total'],20)
        for limit in (0,-1,4097,True):
            with self.assertRaises(ValueError):c._bad_cell_diagnostics(a,b,delta,limit=limit)
        self.assertEqual(sum(np.prod(s)for s in c.original.SHAPES.values()),4034)

    def test_unknown_never_encodes_or_fabricates_operands(self):
        result,calls=self.compare(arrays(),arrays(),True,eligible=False)
        self.assertEqual(calls,[]);self.assertEqual(result['columns'],dict(status='UNKNOWN',reason='STATE_NOT_AGREED'))

    def test_saved_diagnostics_refuse_reference_drift_without_runtime_or_original_mutation(self):
        Owner.fail_p2=False
        with tempfile.TemporaryDirectory()as tmp:
            args=fixture(Path(tmp));before={name:args[name].read_bytes()for name in ('measurement_path','trace_path','native_result_path','native_decisions_path')}
            args['record_bad_cells']=True
            with self.assertRaisesRegex(ValueError,'diagnostic reference drift'):
                SavedGameplayTests().run_saved(args)
            self.assertEqual(before,{name:args[name].read_bytes()for name in before})


if __name__=='__main__':unittest.main()
