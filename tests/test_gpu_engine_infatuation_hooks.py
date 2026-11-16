"""Real Showdown gender construction audit and malformed-control checks."""
from pathlib import Path
import shutil
import subprocess
import unittest

from tests._showdown_root import requires_showdown, showdown_root


class GenderHookTests(unittest.TestCase):
    @requires_showdown()
    @unittest.skipUnless(shutil.which('node'), 'requires node')
    def test_constructor_law_and_stream_position(self):
        subprocess.run(['node',str(Path(__file__).with_name('infatuation_gender_hooks.cjs')),
            str(showdown_root())],check=True,timeout=60)


if __name__=='__main__':unittest.main()
