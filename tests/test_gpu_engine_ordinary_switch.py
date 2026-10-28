"""Pinned source controls for the authored ordinary-switch vocabulary."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')


class OrdinarySwitchSourceTests(unittest.TestCase):
    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown')
    def test_catalog_and_flash_fire_gates_match_pinned_showdown(self):
        catalog=json.loads((Path(__file__).parent/'data/gpu_ordinary_switch_catalog.json').read_text())
        result=subprocess.run([NODE,str(Path(__file__).with_name('gpu_ordinary_switch_oracle.cjs')),SHOWDOWN],
            text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),catalog)
        self.assertEqual(len(catalog['flash_fire']),8)
        for row in catalog['flash_fire']:
            self.assertEqual(row['threshold'],75)
            self.assertEqual(row['activated'],row['hit'] and not row['protect'])
        for row in catalog['terminal_switch']:
            self.assertTrue(row['ended'])
            self.assertEqual(row['winner'],'seat'+str(row['seat']))
            self.assertEqual((row['request'],row['incoming_hp'],row['remaining']),('',0,1))


if __name__=='__main__':unittest.main()
