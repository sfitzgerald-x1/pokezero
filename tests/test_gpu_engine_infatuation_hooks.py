"""Real Showdown gender construction audit and malformed-control checks."""
from pathlib import Path
import json
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

    @requires_showdown()
    @unittest.skipUnless(shutil.which('node'), 'requires node')
    def test_comparison_policy_checks_vocabulary_values_independent_of_object_key_order(self):
        root=Path(__file__).resolve().parents[1]/'gpu_engine'
        script="console.log(JSON.stringify(require(process.argv[1]).load(process.argv[2]).vocabulary))"
        vocab=json.loads(subprocess.check_output(['node','-e',script,str(root/'team_rows.cjs'),str(showdown_root())],text=True))
        metadata=dict(vocabulary=dict(reversed(list(vocab.items()))))
        command=['node',str(root/'comparison_gender_policy.cjs'),str(showdown_root())]
        policy=json.loads(subprocess.check_output(command,input=json.dumps(metadata),text=True))
        self.assertEqual(policy['species'],vocab['species'])
        metadata['vocabulary']['species']=vocab['species'][::-1]
        bad=subprocess.run(command,input=json.dumps(metadata),text=True,capture_output=True,timeout=60)
        self.assertNotEqual(bad.returncode,0)
        self.assertIn('comparison gender vocabulary differs',bad.stderr)


if __name__=='__main__':unittest.main()
