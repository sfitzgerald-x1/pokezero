"""team_rows.cjs reproduces generate_teams.cjs's vocabulary and member rows byte for byte."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root

ROOT = Path(__file__).resolve().parents[1]
NODE = os.environ.get('NODE') or shutil.which('node')
LABEL = 'f76228a1354b5d0f307ca2d16101294ad3a2308b'
# The fixed 200-pair fidelity roster (fidelity.FIRST_GAME=50200) starts at pool seed 650100400.
SEED, COUNT = 650100400, 400


@requires_showdown
@unittest.skipUnless(NODE, 'requires node')
class TeamRowsTests(unittest.TestCase):
    def test_team_lines_are_byte_identical_to_generate_teams(self):
        original = subprocess.check_output([NODE, str(ROOT / 'gpu_engine/generate_teams.cjs'), str(showdown_root()),
                                            str(COUNT), str(SEED), LABEL], text=True).splitlines()
        factored = subprocess.check_output([NODE, str(ROOT / 'gpu_engine/team_rows.cjs'), str(showdown_root()),
                                            str(COUNT), str(SEED)], text=True).splitlines()
        header, vocabulary = json.loads(original[0]), json.loads(factored[0])
        self.assertEqual(vocabulary, dict(kind='vocabulary', vocabulary=header['vocabulary']))
        teams = [line for line in original if line.startswith('{"kind":"team"')]
        self.assertEqual(len(teams), COUNT)
        digest = lambda lines: hashlib.sha256('\n'.join(lines).encode()).hexdigest()
        self.assertEqual(digest(factored[1:]), digest(teams))


if __name__ == '__main__':
    unittest.main()
