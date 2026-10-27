"""Pinned source law for the native cure regression; not JAX sleep parity."""
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')


@unittest.skipUnless(NODE and SHOWDOWN,'requires pinned source simulator')
class NaturalCureSourceTests(unittest.TestCase):
    def test_natural_cure_ends_rest_before_a_fresh_natural_sleep(self):
        rows=json.loads(subprocess.check_output([NODE,
            str(Path(__file__).with_name('gpu_natural_cure_sleep_oracle.cjs')),SHOWDOWN],text=True))
        for side in (0,1):
            cases=[r for r in rows if r['side']==side]
            self.assertEqual(sorted(r['forcedNaturalDuration'] for r in cases),[2,3,4,5])
            for row in cases:
                self.assertEqual((row['afterRest']['status'],row['afterRest']['time']),('slp',3))
                self.assertEqual((row['afterCure']['status'],row['afterCure']['time']),('',0))
                self.assertEqual(row['freshNaturalSleep']['time'],row['forcedNaturalDuration'])
            awake=[sum(Fraction(1,4) for r in cases if r['attempts'][i]['status']=='') for i in range(5)]
            self.assertEqual(awake,[0,Fraction(1,4),Fraction(1,2),Fraction(3,4),1])
