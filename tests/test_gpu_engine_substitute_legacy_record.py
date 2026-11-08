"""Legacy no-capability direct contexts never inspect absent Substitute data."""
from types import SimpleNamespace as S
from unittest.mock import patch
import unittest
import jax.numpy as j
import numpy as np
from gpu_engine import engine_substitute_damage_callbacks as sd
from gpu_engine import engine_execution_completeness as ex


class LegacyRecordTests(unittest.TestCase):
    def test_absent_capability_preserves_no_attempt_and_marks_forged_attempt_unhandled(self):
        for requested in (j.zeros(2,bool),j.array([True,False])):
            tape=object();ticket=object();ledger=object();tables=object();updated=object();result=object()
            pending=S(context=S(data=S(substitute_damage=None)),ticket=ticket,requested=requested)
            with (patch.object(ex,'_context',return_value=S(ledger=ledger,tables=tables)),
                  patch.object(ex,'unhandled',return_value=updated)as unhandled,
                  patch.object(ex,'_put_context',return_value=result),
                  patch.object(sd,'raw_words',side_effect=AssertionError('Absent capability cannot produce raw words')),
                  patch.object(ex,'emitted',side_effect=AssertionError('No consumer credit'))):
                self.assertIs(sd.record(tape,pending,object()),result)
            args,kw=unhandled.call_args;self.assertEqual(args,(ledger,ticket))
            self.assertIs(kw['tables'],tables);np.testing.assert_array_equal(kw['active'],requested)

    def test_no_execution_context_returns_original_tape(self):
        tape=object();pending=S(context=S(data=S(substitute_damage=None)))
        with patch.object(ex,'_context',return_value=None):self.assertIs(sd.record(tape,pending,object()),tape)


if __name__=='__main__':unittest.main()
