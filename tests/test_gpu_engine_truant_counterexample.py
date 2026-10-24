"""Authored guard/accounting doubles; no native or JAX Truant measurement."""
from types import SimpleNamespace as NS
import io
import json
import unittest
from unittest.mock import patch

from gpu_engine import truant_counterexample as probe


class Parent:
    def __init__(self,states):self.states=states
    def to_string(self):return 'unchanged parent'
    def apply_instructions(self,branch):return self.states[branch.index]


def outcome(label):
    mon=NS(hp=600,moves=[NS(pp=7)],ability='truant')
    side=NS(pokemon=[mon],volatile_statuses={'truant'})
    return NS(side_one=side,side_two=side,to_string=lambda:label)


class TruantCounterexampleGuards(unittest.TestCase):
    def test_console_json_uses_print_flush_not_encoder_options(self):
        class Sink(io.StringIO):
            flush_count=0
            def flush(self):self.flush_count+=1
        stream=Sink()
        probe.emit_summary(dict(elapsed_seconds=1.25,cases=[dict(turns=[dict(total_variation_half_l1=0),dict(total_variation_half_l1=1)])]),stream=stream)
        self.assertEqual(json.loads(stream.getvalue()),dict(cases=1,elapsed_seconds=1.25,tv_by_case=[[0,1]]))
        self.assertEqual(stream.flush_count,1)

    def test_native_opt_in_is_caller_owned(self):
        with patch.object(probe.os,'environ',{}):
            with self.assertRaisesRegex(ValueError,'caller must opt'):
                probe.native_root(truant=True)

    def test_distinct_latent_outcomes_cannot_be_merged_by_hp_pp(self):
        branches=[NS(percentage=50,index=i,instruction_list=[]) for i in range(2)]
        engine=NS(generate_instructions=lambda *_:branches)
        parent=Parent([outcome('with truant'),outcome('without truant')])
        with patch.dict('sys.modules',{'poke_engine':engine}):
            with self.assertRaisesRegex(ValueError,'full-state branching'):probe.native_step(parent)

    def test_probability_one_singleton_and_full_state_preserved(self):
        branch=NS(percentage=100,index=0,instruction_list=['ApplyVolatileStatus TRUANT'])
        engine=NS(generate_instructions=lambda *_:[branch]);parent=Parent([outcome('full original state')])
        with patch.dict('sys.modules',{'poke_engine':engine}):
            _,distribution,rows=probe.native_step(parent)
        self.assertEqual(distribution,{(600,600,7,7):1})
        self.assertEqual(rows[0]['full_state'],'full original state')
        self.assertEqual(rows[0]['volatile_statuses'],[['truant'],['truant']])

    def test_invalid_native_mass_is_not_renormalized(self):
        branch=NS(percentage=99,index=0,instruction_list=[])
        engine=NS(generate_instructions=lambda *_:[branch]);parent=Parent([outcome('same')])
        with patch.dict('sys.modules',{'poke_engine':engine}):
            with self.assertRaisesRegex(ValueError,'mass invalid'):probe.native_step(parent)


if __name__=='__main__':unittest.main()
