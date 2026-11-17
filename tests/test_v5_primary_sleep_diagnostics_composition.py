"""Exact reviewed source composition and pure B8 diagnostic transport only."""
from pathlib import Path
import subprocess
import unittest
import jax.numpy as j
import numpy as np
from gpu_engine import engine_outcome_diagnostics as diag,v5_hypothetical_batch as batch

class PrimarySleepCompositionTests(unittest.TestCase):
    def test_reviewed_ordinary_components_and_distinct_private_origins_unchanged(self):
        from gpu_engine import transition
        self.assertEqual(transition.EFFECT_BOUNDS,(16,16,4,2,2))
        groups=(('2b0ec76a',('engine_sleep_powder_callbacks.py','engine_outcome_diagnostics.py','engine_kernel_frame_risk.py','benchmark_retained_games.py')),
            ('c3166bdc',('v5_hypothetical_profile.py','v5_public_branch.py','v5_hypothetical_batch.py','engine_weather_speed_profile.py',
                'engine_leftovers_callbacks.py','engine_leftovers_no_output.py','engine_status_residual_callbacks.py','engine_weather_hp_segment.py')))
        for original,names in groups:
            for name in names:
                path='gpu_engine/'+name
                # Keep the original composition proof at its assembly commit.
                self.assertEqual(subprocess.check_output(['git','show','40228424:'+path]),
                    subprocess.check_output(['git','show',original+':'+path]))
                expected=subprocess.check_output(['git','show','6caf071b:'+path])
                if name=='v5_hypothetical_batch.py':
                    # Only the reviewed fifth-coordinate extension may differ
                    # from this branch's base; every other byte stays bound.
                    for before,after,count in ((b'(1,2,4)',b'(1,2,len(transition.EFFECT_BOUNDS))',2),
                            (b'np.array([16,16,4,2])',b'np.array(transition.EFFECT_BOUNDS)',1)):
                        self.assertEqual(expected.count(before),count)
                        expected=expected.replace(before,after)
                self.assertEqual(Path(path).read_bytes(),expected)

    def test_two_literal_outcome_slots_stack_and_take_as_rows_at_b8(self):
        rows=[]
        for n in range(8):
            value=diag.empty(1)
            value=value._replace(words=value.words.at[0,0].set(j.arange(len(diag.FIELDS),dtype=j.int32)+n*100),
                valid=value.valid.at[0,0].set(True),count=j.ones(1,j.int32),overflow=j.array([n==7]))
            rows.append(value)
        stacked,layout=batch._stack(rows)
        self.assertEqual(stacked.words.shape,(8,2,len(diag.FIELDS)))
        self.assertEqual(stacked.count.tolist(),[1]*8)
        for row,original in enumerate(rows):
            result=batch._take(stacked,layout,row,8)
            for before,after in zip(original,result,strict=True):np.testing.assert_array_equal(before,after)
        refused=diag.refused_rows(stacked,np.array([False,True,False,True,False,False,False,True]))
        self.assertEqual([r['row']for r in refused['rows']],[1,3,7])
        self.assertEqual([r['overflow']for r in refused['rows']],[False,False,True])
