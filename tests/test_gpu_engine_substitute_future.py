"""Inactive native Substitute storage, not a panel replay or coverage credit."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

from gpu_engine import future_state as f
from test_gpu_engine_future_state import FAKE, fixture

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')


def constructed():
    data,meta=fixture()
    return data,meta,f.native_input_from_jax(data,None,meta,module=FAKE)


class SubstituteFutureTests(unittest.TestCase):
    def test_absent_volatile_stale_scalar_is_inert_but_raw_wire_is_retained(self):
        for side in f.SIDE_NAMES:
            for stale in (0,1,60,32767):
                data,meta,native=constructed()
                getattr(native.state,side).substitute_health=stale
                raw=native.state.to_string()
                result=f.canonical_native_future(raw,native.certificate)
                expected=f.canonical_jax_future(data,None,meta,certificate=native.certificate)
                self.assertIsInstance(result,f.CanonicalFuture,result)
                self.assertEqual(result.key,expected.key)
                self.assertEqual(result.raw_native,raw)
                self.assertEqual(f._native_wire(result.raw_native)[side]['substitute_health'],stale)

    def test_active_substitute_health_is_never_erased_or_combined(self):
        for side in f.SIDE_NAMES:
            _,_,native=constructed()
            target=getattr(native.state,side);target.volatile_statuses.add('substitute')
            keys=[]
            for hp in (1,60,61):
                target.substitute_health=hp
                result=f.canonical_native_future(native.state,native.certificate)
                self.assertIsInstance(result,f.CanonicalFuture,result)
                self.assertEqual(dict(result.key)[side+'.substitute_health'],hp)
                keys.append(result.key)
            self.assertEqual(len(set(keys)),3)

    def test_invalid_health_and_live_baton_interfaces_are_not_hidden(self):
        for side in f.SIDE_NAMES:
            _,_,native=constructed();target=getattr(native.state,side)
            for malformed in (-32768,-1,32768,65536):
                for active in (False,True):
                    target.substitute_health=malformed
                    target.volatile_statuses={'substitute'} if active else set()
                    result=f.canonical_native_future(native.state,native.certificate)
                    self.assertTrue(isinstance(result,f.Refused),type(result).__name__)
                    self.assertIn('INVALID_NATIVE_SUBSTITUTE_HEALTH',{g.code for g in result.gaps})
                    self.assertEqual(f._native_wire(result.raw_native)[side]['substitute_health'],malformed)
            target.volatile_statuses=set()
            target.substitute_health=60;target.baton_passing=True
            result=f.canonical_native_future(native.state,native.certificate)
            self.assertIsInstance(result,f.Refused,result)
            self.assertIn(side+'.baton_passing',{g.field for g in result.gaps})
            self.assertEqual(f._native_wire(result.raw_native)[side]['substitute_health'],60)

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Gen3 Showdown')
    def test_showdown_ordinary_switch_clears_but_completed_baton_retains_substitute(self):
        specs=[dict(side=side,baton=baton) for side in (0,1) for baton in (False,True)]
        run=subprocess.run([NODE,str(Path(__file__).with_name('gpu_substitute_future_oracle.cjs')),SHOWDOWN],
            input=json.dumps(specs),text=True,capture_output=True,check=True)
        rows=json.loads(run.stdout);self.assertEqual(len(rows),4)
        for spec,row in zip(specs,rows,strict=True):
            self.assertEqual(row,dict(species='ditto',sub=73 if spec['baton'] else 0,request='move'))


if __name__=='__main__':unittest.main()
