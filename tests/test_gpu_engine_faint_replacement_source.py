"""Pinned-source checks for simultaneous Gen3 faint replacement ordering."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')


@unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown and Node')
class FaintReplacementSourceTests(unittest.TestCase):
    def test_both_installs_precede_hazard_callbacks_and_one_deferred_residual(self):
        run=subprocess.run([NODE,str(Path(__file__).with_name('gpu_faint_replacement_oracle.cjs')),SHOWDOWN],
            capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        packet=json.loads(run.stdout);self.assertFalse(packet['scientific_measurement'])
        self.assertEqual(len(packet['rows']),26)
        for row in packet['rows']:
            with self.subTest(row=row):
                self.assertEqual(row['requested'],[True,True])
                first=row['first'];self.assertEqual(first['activeIsIncoming'],[True,True])
                side=row['hazardSide'];layers=row['layers'];terminal=row['terminal']
                if not layers:
                    self.assertEqual(first['hp'],[737,737]);self.assertFalse(first['ended'])
                    self.assertEqual(first['requested'],[False,False]);self.assertIsNone(row['afterChain'])
                else:
                    self.assertEqual(first['hp'][side],0);self.assertEqual(first['hp'][1-side],800-((0,125,166,250)[layers] if row['late'] else 0))
                    self.assertEqual(first['ended'],terminal)
                    self.assertEqual(first['requested'],[not terminal and s==side for s in (0,1)])
                    if terminal:self.assertIsNone(row['afterChain'])
                    else:
                        final=row['afterChain'];self.assertFalse(final['ended'])
                        self.assertEqual(final['requested'],[False,False])
                        self.assertEqual(final['hp'][side],1000-(0,125,166,250)[layers])
                        self.assertEqual(final['hp'][1-side],737-((0,125,166,250)[layers] if row['late'] else 0))
