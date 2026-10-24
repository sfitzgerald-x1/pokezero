"""Authored owner routing only; no native model, collection or cost claim."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

DRIVER=Path(__file__).resolve().parents[1]/'scripts/bench_one_ply_exact_native.py'
spec=importlib.util.spec_from_file_location('one_ply_runner_test',DRIVER)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class OwnerTests(unittest.TestCase):
    def call(self,folder,owned,extra=()):
        fixture=folder/'fixture.json'
        fixture.write_text('{"authored_fixture":true}')
        output=folder/'receipt.json'
        argv=[str(DRIVER),'--fixture',str(fixture),'--model','unused',
              '--tables','unused','--output',str(output),*extra]
        with patch.object(sys,'argv',argv),patch.object(runner,'run_owned',return_value=owned) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            code=runner.main()
        return code,json.loads(output.read_text()),run.call_args

    def test_fixed_worker_launch_and_actual_exit_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            code,receipt,call=self.call(Path(tmp),dict(actual_exit_code=7,timed_out=False,
                owner_signal=None,owned_child_reaped=True,stdout='failed',stderr='error'))
        self.assertEqual(code,7)
        self.assertEqual(receipt['actual_exit_code'],7)
        self.assertEqual(call.args[0][-1],'--worker')
        self.assertEqual(call.kwargs['env']['POKEZERO_ENUMERATE_ROLLS'],'1')
        self.assertEqual(call.kwargs['hard_seconds'],150)
        self.assertFalse(receipt['scientific_collection'])

    def test_timeout_is_not_promoted_by_zero_child_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            code,receipt,_=self.call(Path(tmp),dict(actual_exit_code=0,timed_out=True,
                owner_signal=None,owned_child_reaped=True))
        self.assertEqual(code,1)
        self.assertTrue(receipt['timed_out'])

    def test_invalid_limits_or_existing_output_do_not_launch(self):
        for extra in (('--hard-seconds','0'),('--soft-seconds','150'),
                      ('--batch-size','0'),('--max-leaves','0'),('--max-branches','0')):
            with tempfile.TemporaryDirectory() as tmp,self.subTest(extra=extra):
                with self.assertRaisesRegex(RuntimeError,'bounded work'):
                    self.call(Path(tmp),{},extra)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'receipt.json').write_text('original')
            with self.assertRaisesRegex(RuntimeError,'overwrite'):
                self.call(folder,{})
            self.assertEqual((folder/'receipt.json').read_text(),'original')


if __name__=='__main__':unittest.main()
