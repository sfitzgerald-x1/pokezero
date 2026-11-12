"""Bounded actual callback operands, no games, model or kernel execution."""
import ast
from pathlib import Path
import subprocess
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_outcome_diagnostics as d
import test_gpu_engine_before_move_diagnostics as fixture

BASE='8c879093c6384610d24338e90f7cf6588fa589b0'

class SleepGuardDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        f=fixture.BeforeMoveDiagnosticsTests();f.setUp();self.t=f.t;self.args,self.kw=f.inputs()

    def observe(self,t,**changes):
        b=8;z=j.arange(b,dtype=j.int32)
        values=dict(rest_origin=z%2==0,skipped=z,volatile=j.full(b,0x80000001,j.uint32),
            requested=z%3==0,valid=z%3!=0,move_ok=z%4==0,eligible=j.ones(b,bool))
        values.update(changes)
        return d.record_sleep_guard_on_tape(t,**values)

    def test_literal_alignment_counts_masks_unsigned_and_overflow(self):
        active=j.array([True,False]*4);a=self.observe(self.t,eligible=active)
        self.assertIs(a.execution.ledger,self.t.execution.ledger);self.assertIs(a.execution.risks,self.t.execution.risks)
        for name in self.t._fields:
            if name!='execution':self.assertIs(getattr(a,name),getattr(self.t,name))
        np.testing.assert_array_equal(a.execution.diagnostics.before_move_count,0)
        b=d.record_before_move_on_tape(a,*self.args,**self.kw)
        records=d.refused_rows(b.execution.diagnostics,np.ones(8,bool))['rows']
        for i,row in enumerate(records):
            r=row['before_move']['records'][0]
            self.assertEqual(r['sleep_volatile_uint32'],0x80000001 if i%2==0 else None)
            self.assertEqual(r['sleep_skipped'],i if i%2==0 else None)
            self.assertEqual(r['sleep_block_requested'],int(i%3==0)if i%2==0 else None)
            self.assertEqual(r['sleep_block_move_ok'],int(i%4==0)if i%2==0 else None)
            self.assertEqual(r['sleep_rest_origin'],int(i%2==0)if i%2==0 else None)
        c=d.record_before_move_on_tape(self.observe(b),*self.args,**self.kw)
        e=self.observe(c)
        np.testing.assert_array_equal(e.execution.diagnostics.sleep_guard_words,c.execution.diagnostics.sleep_guard_words)
        np.testing.assert_array_equal(e.execution.diagnostics.before_move_count,2)

    def test_missing_malformed_legacy_and_row_commits(self):
        a=self.observe(self.t,skipped=j.ones(8,j.float32),valid=None)
        b=d.record_before_move_on_tape(a,*self.args,**self.kw)
        r=d.refused_rows(b.execution.diagnostics,np.array([True]+[False]*7))['rows'][0]['before_move']['records'][0]
        self.assertIsNone(r['sleep_skipped']);self.assertIsNone(r['sleep_block_valid']);self.assertEqual(r['sleep_volatile_uint32'],0x80000001)
        for cap in (None,d.Capture(*self.t.execution.diagnostics[:23]),self.t.execution.diagnostics._replace(sleep_guard_words=j.zeros((8,2,6),j.float32))):
            t=self.t._replace(execution=self.t.execution._replace(diagnostics=cap));self.assertIs(self.observe(t),t)
        legacy=d.Capture(*b.execution.diagnostics[:23]);old={}
        exec(compile(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_outcome_diagnostics.py'],text=True),'legacy','exec'),old)
        current_report=d.refused_rows(legacy,np.ones(8,bool))
        old_report=old['refused_rows'](old['Capture'](*legacy[:23]),np.ones(8,bool))
        for current_row,old_row in zip(current_report['rows'],old_report['rows'],strict=True):
            # The installed Baton diagnostic gained an explicit quiet-status
            # field; this new schema is unavailable to the historical decoder.
            current_baton=current_row.pop('baton_start');old_baton=old_row.pop('baton_start')
            self.assertTrue(current_baton['available']);self.assertEqual(current_baton['count'],0)
            self.assertEqual(current_baton['records'],[]);self.assertFalse(old_baton['available'])
        self.assertEqual(current_report,old_report)
        from gpu_engine import engine_retained_encoder_driver as driver
        got=driver._rows(j.array([True,False]*4),self.t.execution.diagnostics,b.execution.diagnostics)
        np.testing.assert_array_equal(got.sleep_guard_known[::2],b.execution.diagnostics.sleep_guard_known[::2])
        self.assertFalse(bool(got.sleep_guard_known[1::2].any()))

    def test_original_helper_body_and_physical_files_exact(self):
        # engine_sleep_blocked_callbacks, mechanics, kernel_frame_risk, execution_completeness and the driver: byte-exact against reviewed SOURCE_PINS in transition_observer_wiring's source-binding subtests.
        # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
        for p in ('benchmark_capture_gap_census.py',):
            self.assertEqual(Path('gpu_engine',p).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+p]))

    def test_fixed_array_carry_traces_without_kernel_execution(self):
        def fn(cap):
            t=self.t._replace(execution=self.t.execution._replace(diagnostics=cap))
            return d.record_before_move_on_tape(self.observe(t),*self.args,**self.kw).execution.diagnostics
        shaped=jax.eval_shape(fn,self.t.execution.diagnostics)
        self.assertEqual(shaped._fields,d.Capture._fields)
        self.assertEqual(len(shaped),32)
        self.assertEqual(shaped.sleep_guard_words.shape,(8,2,6))
        jax.make_jaxpr(fn)(self.t.execution.diagnostics)

if __name__=='__main__':unittest.main()
