"""Compiled actual frame through both real owner drains: heal and status spans, rollback; fake model only.

The one kept heavy owning-drain module. The light owning_status_drain module runs the same real
construct, advance and consume flow with the prepared frame closure's body run eagerly. This module
compiles that closure once at B=2, so the frame, requests, public, join, encode and score closures all
run compiled together with accounting, and the real kernel word stream meets the real consumers. It also
runs engine_retained_encoder_driver's hot-jit test with the basic-mode frame compiled.
"""
import unittest

from gpu_engine import engine_retained_encoder_driver as d

import test_gpu_engine_engine_retained_encoder_driver as encoder
import test_gpu_engine_owning_status_drain as light


class OwningDrainEndToEndTests(unittest.TestCase):
    # Bound inside the class: a module-level TestCase name would make unittest collect the light tests here.
    base=light.OwningStatusDrainTests
    compiled_frame=True
    setUpClass=classmethod(light.OwningStatusDrainTests.setUpClass.__func__)
    construct=light.OwningStatusDrainTests.construct
    chance=light.OwningStatusDrainTests.chance
    same=light.OwningStatusDrainTests.same
    check_final_public_hp=light.OwningStatusDrainTests.check_final_public_hp
    frame_call=light.OwningStatusDrainTests.frame_call
    frame_phase=light.OwningStatusDrainTests.frame_phase
    birth=light.OwningStatusDrainTests.birth
    heal_turn=light.OwningStatusDrainTests.heal_turn
    next_turn=light.OwningStatusDrainTests.next_turn
    warm=light.OwningStatusDrainTests.warm
    actual_frame=light.OwningStatusDrainTests.actual_frame
    final_hp=light.OwningStatusDrainTests.final_hp

    def test_heal_then_status_spans_drain_both_owners_through_the_compiled_frame(self):
        # Hail turn: both Leftovers heals captured by the kernel and drained by both owners (admitted,
        # observers' accounting equal, owner-correct public HP); then frame 3 drains the burn and
        # poison residual spans through the same compiled frame.
        self.base.test_actual_heal_turn_captures_and_drains_both_owner_leftovers_spans(self)
        self.base.test_actual_kernel_status_spans_drain_both_original_owner_consumers(self)
        # The heal turn, the next turn and the status turn reuse one compiled frame executable.
        self.assertEqual(d._checked(self.kernel)['frame']._cache_size(),1)

    def test_second_owner_fault_rolls_back_both_rows_through_real_encode_and_score(self):
        self.base.test_one_row_second_owner_failure_rolls_back_both_and_does_not_drain(self)

    def test_diagnostic_status_write_stays_unknown_through_the_compiled_frame(self):
        # The diagnostic status mode's frame, compiled (the light module runs it eagerly).
        self.base.test_internal_actual_status_write_stays_explicitly_unknown(self)



class BasicModeHotJitFrameTests(unittest.TestCase):
    """engine_retained_encoder_driver's frame tests with the basic-mode frame compiled (the light module runs
    them eagerly): one executable across every turn, the step equal to the plain one, fresh host guards on a
    hot frame, no host callbacks."""
    base=encoder.RetainedEncoderDriverTests   # bound in the class body, never at module level
    compiled_frame=True
    setUpClass=classmethod(encoder.RetainedEncoderDriverTests.setUpClass.__func__)
    construct=encoder.RetainedEncoderDriverTests.construct
    chance=encoder.RetainedEncoderDriverTests.chance
    same=encoder.RetainedEncoderDriverTests.same
    assert_refused_before_fake=encoder.RetainedEncoderDriverTests.assert_refused_before_fake

    # These two run first (alphabetical order) on the same compiled frame, so the hot-jit test's
    # _cache_size()==1 counts every frame call of the module, as in the original, and the host guards run
    # against a hot compiled frame.
    def test_actual_action_faint_then_forced_replacement_uses_source_wait_and_order(self):
        self.base.test_actual_action_faint_then_forced_replacement_uses_source_wait_and_order(self)

    def test_host_guards_run_after_hot_jit_and_stale_carry_cannot_be_replayed(self):
        self.base.test_host_guards_run_after_hot_jit_and_stale_carry_cannot_be_replayed(self)

    def test_same_core_one_invocation_nonphysical_bench_both_orders_and_hot_jit(self):
        self.base.test_same_core_one_invocation_nonphysical_bench_both_orders_and_hot_jit(self)


if __name__=='__main__':unittest.main()
