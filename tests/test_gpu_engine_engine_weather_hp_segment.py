"""Literal owning-seam controls; no shared wiring/models/full games."""
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_weather_hp_segment as segment, engine_weather_speed_profile as profile
from gpu_engine import engine_public_events as public, engine_lifecycle_callbacks as life
from gpu_engine import mechanics, transition, weather
import test_gpu_engine_engine_weather_speed_profile as fixtures
import test_gpu_engine_engine_item_callbacks as tapes


class WeatherHPSegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.WeatherSpeedProfileTests.setUpClass()
        cls.f = fixtures.WeatherSpeedProfileTests()
        cls.data = profile.check_profile(cls.f.profile)

    def same(self, a, b):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(x, y)

    def case(self, batch=1, *, speeds=(199, 301), types=None, wish=False, wish_both=False, capacity=64,
             prefix=True, prior=0):
        born = self.f.state(batch)._replace(active=j.broadcast_to(j.array([2, 5], j.int32), (batch, 2)))
        values = j.broadcast_to(j.array(speeds, j.int32), (batch, 2))
        born = born._replace(stats=born.stats.at[..., 5].set(j.broadcast_to(values[:, :, None], (batch, 2, 6))),
            types=self.f.rules.device.species_types[born.species])
        ids = self.f.streams(batch)
        record = profile.qualify_birth(born, ids, profile=self.f.profile)
        pre = born._replace(turn=j.full(batch, 2, j.int32), weather=j.full(batch, weather.SAND, j.int32),
            weather_counter=j.full(batch, 3, j.int32),
            hp=born.hp.at[:, 0, 2].set(100).at[:, 1, 5].set(120))
        if types is not None:
            pre = pre._replace(types=pre.types.at[:, 0, 2].set(j.array(types[0], j.int32))
                .at[:, 1, 5].set(j.array(types[1], j.int32)))
        wishes = (0, 1) if wish_both else (0,) if wish else ()
        for side in wishes:
            pre = pre._replace(wish_counter=pre.wish_counter.at[:, side].set(1))
        end_speed = transition._speeds(pre, self.f.table)
        cursor = tapes.tape(pre).cursor._replace(stream_id=ids,
            engine_binding=j.broadcast_to(self.data.engine_binding, (batch, 8)),
            revealed_species_id=pre.species[j.arange(batch)[:, None], j.arange(2)[None, :], pre.active])
        tape = life._tape(cursor, self.f.engine, capacity)
        tape = tape._replace(refusal=j.full(batch, prior, j.int32))
        if prefix:
            tape = public.move(tape, j.zeros(batch, j.int32), pre.active[:, 0], pre.active[:, 1],
                j.ones(batch, j.int32), eligible=j.ones(batch, bool))
        # Actual numerical primitives at the same before/after seam. The direct
        # internal append represents a future emitter deferring ONLY its own
        # UNKNOWN; this test does not register a public producer/callsite.
        work = pre
        for side in wishes:
            from gpu_engine import sustain
            rows = j.arange(batch); active = work.active[:, side]
            result = sustain.wish_residual(work.hp[rows, side, active], work.max_hp[rows, side, active],
                work.wish_counter[:, side], eligible=j.ones(batch, bool))
            work = work._replace(hp=work.hp.at[rows, side, active].set(result.hp),
                wish_counter=work.wish_counter.at[:, side].set(result.counter))
        changed, counter, tape = weather.residual_clock(work.weather, work.weather_counter,
            eligible=j.ones(batch, bool), _weather_trace=tape)
        work = work._replace(weather=changed, weather_counter=counter)
        before = tape.count
        pre_writes = work
        for side in (0, 1):
            rows = j.arange(batch); active = work.active[:, side]
            old = work.hp[rows, side, active]
            maximum = work.max_hp[rows, side, active]
            status = work.status[rows, side, active]
            new = weather.chip(self.f.table.weather, work.weather,
                work.types[rows, side, active], work.ability[rows, side, active], old, maximum,
                eligible=j.ones(batch, bool))
            work = work._replace(hp=work.hp.at[rows, side, active].set(new))
            tape = public._append(tape, 16, end_speed[:, 0], end_speed[:, 1],
                j.full(batch, side, j.int32), active, (old, new, maximum, status, work.weather), new < old)
        return tape, before, pre, work, end_speed, record, pre_writes

    def run_case(self, values, *, eligible=True, data=None):
        return segment.order_segment_device(*values[:6], self.data if data is None else data,
            pre_writes_state=values[6], eligible=eligible)

    def test_two_whole_receipts_sorted_only_by_checked_cached_speed(self):
        values = self.case(prior=128 | 256)
        tape, before, *_ = values
        result = self.run_case(values)
        self.assertEqual(result.supported.tolist(), [True]); self.assertEqual(result.reordered.tolist(), [True])
        start = int(before[0])
        self.assertEqual(result.tape.words[0, start:start + 2, 3].tolist(), [1, 0])
        self.same(result.tape.words[:, :start], tape.words[:, :start])
        self.same(result.tape.words[:, start:start + 2], tape.words[:, start:start + 2][:, ::-1])
        self.same(result.tape._replace(words=tape.words), tape)
        self.assertEqual(result.tape.refusal.tolist(), [128 | 256])
        self.assertEqual(result.tape.count.tolist(), tape.count.tolist())

    def test_already_ordered_singleton_and_zero_segments(self):
        ordered = self.run_case(self.case(speeds=(301, 199)))
        self.assertTrue(bool(ordered.supported[0])); self.assertFalse(bool(ordered.reordered[0]))
        # Ground is immune to SAND; the other literal active has one write.
        single_values = self.case(types=((5, 0), (1, 0)))
        self.assertEqual((single_values[0].count - single_values[1]).tolist(), [1])
        single = self.run_case(single_values)
        self.assertTrue(bool(single.supported[0])); self.same(single.tape, single_values[0])
        empty_values = self.case(types=((5, 0), (6, 0)), speeds=(200, 200))
        empty_values = (*empty_values[:5], empty_values[5]._replace(qualified_birth=j.array([False])), empty_values[6])
        empty = self.run_case(empty_values)
        self.assertEqual((empty.tape.count - empty_values[1]).tolist(), [0])
        self.assertTrue(bool(empty.supported[0])); self.same(empty.tape, empty_values[0])
        self.assertFalse(bool(empty.record.qualified_birth[0]))

    def test_wish_healing_is_not_reconstructed_and_does_not_change_order(self):
        values = self.case(wish=True)
        tape, before, pre, after, *_ = values
        old = int(tape.words[0, int(before[0]), 5])
        self.assertGreater(old, int(pre.hp[0, 0, 2]))
        result = self.run_case(values)
        self.assertTrue(bool(result.supported[0])); self.assertTrue(bool(result.reordered[0]))
        self.same(result.tape.words[0, int(before[0]) + 1, 5:], tape.words[0, int(before[0]), 5:])
        self.assertGreater(int(after.hp[0, 0, 2]), int(pre.hp[0, 0, 2]))

    def test_ties_stale_speed_nonpositive_and_false_birth_are_unknown_only(self):
        tied = self.case(speeds=(200, 200), prior=128)
        stale = self.case(prior=128); stale = (*stale[:4], stale[4].at[0, 0].add(1), *stale[5:])
        refused_birth = self.case(prior=128)
        refused_birth = (*refused_birth[:5], refused_birth[5]._replace(qualified_birth=j.array([False])), refused_birth[6])
        nonpositive = self.case(speeds=(1, 301), prior=128)
        t, n, pre, after, _, birth, pre_writes = nonpositive
        pre = pre._replace(boosts=pre.boosts.at[0, 0, 2, 4].set(-6))
        after = after._replace(boosts=after.boosts.at[0, 0, 2, 4].set(-6))
        pre_writes = pre_writes._replace(boosts=pre_writes.boosts.at[0, 0, 2, 4].set(-6))
        speed = transition._speeds(pre, self.f.table)
        for offset in (0, 1):
            t = t._replace(words=t.words.at[0, n[0] + offset, 1:3].set(speed[0]))
        nonpositive = (t, n, pre, after, speed, birth, pre_writes)
        for values in (tied, stale, refused_birth, nonpositive):
            result = self.run_case(values)
            self.assertFalse(bool(result.supported[0])); self.assertFalse(bool(result.reordered[0]))
            self.same(result.tape._replace(refusal=values[0].refusal), values[0])
            self.assertEqual(result.tape.refusal.tolist(), [128 | 256])

    def test_effective_gen3_cached_speed_keeps_values_above_base_13bit_domain(self):
        # Gen3 installs Scripts.pokemon.getActionSpeed on each instance. Its
        # override returns getStat's speed, unlike the BASE prototype's13bit
        # truncation. Never use that base method as this format's oracle.
        for stored in (2047,2048,2500):
            tape,before,pre,after,_,birth,pre_writes=self.case(speeds=(stored,2000),prior=128)
            self.assertTrue(bool(birth.qualified_birth[0]))
            pre=pre._replace(boosts=pre.boosts.at[:,0,2,4].set(6))
            pre_writes=pre_writes._replace(boosts=pre_writes.boosts.at[:,0,2,4].set(6))
            after=after._replace(boosts=after.boosts.at[:,0,2,4].set(6))
            speeds=transition._speeds(pre,self.f.table)
            self.assertEqual(int(speeds[0,0]),stored*4)
            for offset in (0,1):
                tape=tape._replace(words=tape.words.at[0,before[0]+offset,1:3].set(speeds[0]))
            result=self.run_case((tape,before,pre,after,speeds,birth,pre_writes))
            self.assertEqual(result.supported.tolist(),[True])
            self.assertEqual(result.tape.refusal.tolist(),[128])
            self.assertEqual(result.tape.words[0,before[0]:before[0]+2,3].tolist(),[0,1])

    def test_lethal_malformed_payload_identity_stamps_and_reserved_metadata_refuse(self):
        values = self.case(prior=128); tape, before, *_ = values; at = int(before[0])
        for col, value in ((0, 3), (1, 9), (3, 2), (4, 0), (5, 0), (6, 0),
                (6, 1), (7, 2), (8, 7), (9, weather.RAIN)):
            changed = tape._replace(words=tape.words.at[0, at, col].set(value))
            result = self.run_case((changed, *values[1:]))
            self.assertFalse(bool(result.supported[0]), (col, value))
            self.same(result.tape._replace(refusal=changed.refusal), changed)
        for name in ('event_turn', 'event_phase', 'event_generation', 'switch_details'):
            array = getattr(tape, name)
            index = (0, at) + ((0,) if array.ndim == 3 else ())
            changed = tape._replace(**{name:array.at[index].add(1)})
            self.assertFalse(bool(self.run_case((changed, *values[1:])).supported[0]))
        duplicate = tape._replace(words=tape.words.at[0, at + 1].set(tape.words[0, at]))
        self.assertFalse(bool(self.run_case((duplicate, *values[1:])).supported[0]))

    def test_late_physical_changes_and_missing_hp_write_refuse(self):
        values = self.case(); tape, before, pre, after, end_speed, birth, pre_writes = values
        variations = [after._replace(stats=after.stats.at[0, 1, 5, 5].add(1)),
            after._replace(boosts=after.boosts.at[0, 1, 5, 4].add(1)),
            after._replace(status=after.status.at[0, 1, 5].set(mechanics.PAR)),
            after._replace(item=after.item.at[0, 1, 4].set(0)),
            after._replace(ability=after.ability.at[0, 1, 5].set(1)),
            after._replace(active=after.active.at[0, 1].set(4)),
            after._replace(hp=after.hp.at[0, 1, 4].add(-1))]
        for changed in variations:
            result = self.run_case((tape, before, pre, changed, end_speed, birth, pre_writes))
            self.assertFalse(bool(result.supported[0])); self.same(result.tape.words, tape.words)
        # Drop the second receipt while retaining its actual HP decrease.
        missing = tape._replace(count=tape.count - 1)
        self.assertFalse(bool(self.run_case((missing, *values[1:])).supported[0]))

    def test_missing_both_hp_receipts_refuses_without_changing_tape_records(self):
        values = self.case(prior=128)
        tape, before, pre, after, *_ = values
        self.assertLess(int(after.hp[0, 0, 2]), int(pre.hp[0, 0, 2]))
        self.assertLess(int(after.hp[0, 1, 5]), int(pre.hp[0, 1, 5]))
        # Neither actual HP decrease has a counted receipt. The uncounted tail
        # is not a substitute for the missing events and must remain untouched.
        missing = tape._replace(count=before)
        result = self.run_case((missing, *values[1:]))
        self.assertEqual(result.supported.tolist(), [False])
        self.assertEqual(result.reordered.tolist(), [False])
        self.assertEqual(result.tape.refusal.tolist(), [128 | 256])
        self.same(result.tape._replace(refusal=missing.refusal), missing)

    def test_both_wish_heals_cannot_mask_dropped_receipts_but_immune_empty_is_valid(self):
        values = self.case(wish_both=True, prior=128)
        tape, before, pre, after, _, _, pre_writes = values
        # Both final HPs exceed their pre-Wish HP, concealing BOTH missing chip
        # receipts from a mere pre-Wish/final comparison.
        for side, active in ((0, 2), (1, 5)):
            self.assertGreaterEqual(int(after.hp[0, side, active]), int(pre.hp[0, side, active]))
            self.assertLess(int(after.hp[0, side, active]), int(pre_writes.hp[0, side, active]))
        missing = tape._replace(count=before)
        result = self.run_case((missing, *values[1:]))
        self.assertEqual(result.supported.tolist(), [False])
        self.assertEqual(result.tape.refusal.tolist(), [128 | 256])
        self.same(result.tape._replace(refusal=missing.refusal), missing)
        immune = self.case(wish_both=True, types=((5, 0), (6, 0)), speeds=(200, 200))
        self.assertEqual((immune[0].count - immune[1]).tolist(), [0])
        valid = self.run_case(immune)
        self.assertTrue(bool(valid.supported[0])); self.assertFalse(bool(valid.reordered[0]))
        self.same(valid.tape, immune[0])

    def test_actual_prewrite_old_hp_and_clock_window_and_intermediate_profile_are_required(self):
        values = self.case(wish=True, prior=128)
        tape, before, pre, after, speeds, birth, pre_writes = values
        at = int(before[0])
        wrong_old = tape._replace(words=tape.words.at[0, at, 5].add(-1))
        self.assertFalse(bool(self.run_case((wrong_old, *values[1:])).supported[0]))
        # Clocks/Wish may change before the HP window, never inside it.
        changed = [after._replace(weather=j.full_like(after.weather, weather.HAIL)),
            after._replace(weather_counter=after.weather_counter + 1),
            after._replace(wish_counter=after.wish_counter.at[0, 0].add(1))]
        for state in changed:
            result = self.run_case((tape, before, pre, state, speeds, birth, pre_writes))
            self.assertFalse(bool(result.supported[0]))
            self.same(result.tape._replace(refusal=tape.refusal), tape)
        unsafe = pre_writes._replace(item=pre_writes.item.at[0, 1, 4].set(4))
        result = self.run_case((tape, before, pre, after, speeds, birth, unsafe))
        self.assertFalse(bool(result.record.qualified_birth[0]))
        self.assertFalse(bool(result.supported[0]))
        with self.assertRaises(TypeError):
            segment.order_segment_device(*values[:6], self.data, eligible=True)

    def test_b8_empty_segments_validate_seam_without_inventing_order_or_profile(self):
        values = self.case(8, prior=128 | 4096)
        tape, before, pre, after, end_speed, birth, pre_writes = values
        # Rows0..5 and7 have no counted HP receipts. Row0 drops both actual
        # decreases. Rows1/2 are genuine no-event tie/false-profile cases. Row3
        # is inactive, row4 foreign, row5 has an impossible late status change.
        empty_rows = j.array([True] * 6 + [False, True])
        tape = tape._replace(count=j.where(empty_rows, before, tape.count))
        same_hp = j.array([False, True, True, False, True, True, False, True])
        after = after._replace(hp=j.where(same_hp[:, None, None], pre.hp, after.hp))
        pre = pre._replace(stats=pre.stats.at[1, ..., 5].set(200))
        pre_writes = pre_writes._replace(stats=pre_writes.stats.at[1, ..., 5].set(200))
        after = after._replace(stats=after.stats.at[1, ..., 5].set(200),
            status=after.status.at[5, 1, 5].set(mechanics.PAR))
        end_speed = end_speed.at[1].set(j.array([200, 200], j.int32))
        birth = birth._replace(qualified_birth=birth.qualified_birth.at[2].set(False),
            source_binding=birth.source_binding.at[4, 0].add(1))
        eligible = j.array([True, True, True, False, True, True, True, True])
        fast = jax.jit(lambda t, n, a, z, s, r, w:segment.order_segment_device(
            t, n, a, z, s, r, self.data, pre_writes_state=w, eligible=eligible))
        result = jax.block_until_ready(fast(tape, before, pre, after, end_speed, birth, pre_writes))
        self.assertEqual(result.supported.tolist(), [False, True, True, False, False, False, True, True])
        self.assertEqual(result.reordered.tolist(), [False] * 6 + [True, False])
        self.assertFalse(bool(result.record.qualified_birth[1]))
        self.assertFalse(bool(result.record.qualified_birth[2]))
        self.assertEqual(result.tape.refusal.tolist(),
            [(128 | 4096 | 256) if row in (0, 4, 5) else (128 | 4096) for row in range(8)])
        for row in (0, 1, 2, 3, 4, 5, 7):
            self.same(result.tape.words[row], tape.words[row])
            self.same(result.tape.event_generation[row], tape.event_generation[row])
        self.same(jax.tree.map(lambda a:a[3], result.record), jax.tree.map(lambda a:a[3], birth))

    def test_bad_counts_overflow_tail_and_ineligible_rows_cannot_rewrite(self):
        values = self.case(prior=128); tape, before, *_ = values
        for n in (before - 1, before + 1, j.array([-1], j.int32), tape.count + 1):
            result = self.run_case((tape, n, *values[2:]))
            self.assertFalse(bool(result.supported[0])); self.same(result.tape.words, tape.words)
        overflow = tape._replace(refusal=tape.refusal | public.OVERFLOW)
        result = self.run_case((overflow, *values[1:]))
        self.assertFalse(bool(result.supported[0])); self.same(result.tape.words, overflow.words)
        self.assertEqual(result.tape.refusal.tolist(), [128 | public.OVERFLOW | 256])
        inactive = self.run_case(values, eligible=False)
        self.same(inactive.tape, tape); self.same(inactive.record, values[5])
        self.assertFalse(bool(inactive.supported[0]))

    def test_b8_mixed_hot_jit_and_namespaces_preserve_global_bindings(self):
        speeds = j.array([[199, 301], [301, 199], [200, 200]] + [[199, 301]] * 5, j.int32)
        values = self.case(8, speeds=speeds, prior=128); tape, before, pre, after, speed, birth, pre_writes = values
        # Row1 already source-ordered; row2 tie; row3 unsafe birth; row4 stale
        # source namespace; row5 late state; row6 no operational processing.
        after = after._replace(status=after.status.at[5, 1, 5].set(1))
        birth = birth._replace(qualified_birth=birth.qualified_birth.at[3].set(False),
            source_binding=birth.source_binding.at[4, 0].add(1))
        at = before
        words = tape.words
        for offset in (0, 1):
            words = words.at[j.arange(8), at + offset, 1:3].set(speed)
        tape = tape._replace(words=words)
        eligible = j.array([True] * 6 + [False, True])
        fast = jax.jit(lambda t, n, a, z, s, r, w:segment.order_segment_device(t, n, a, z, s, r,
            self.data, pre_writes_state=w, eligible=eligible))
        result = jax.block_until_ready(fast(tape, before, pre, after, speed, birth, pre_writes))
        self.assertEqual(result.supported.tolist(), [True, True, False, False, False, False, False, True])
        self.assertEqual(result.reordered.tolist(), [True, False, False, False, False, False, False, True])
        self.same(jax.block_until_ready(fast(tape, before, pre, after, speed, birth, pre_writes)), result)
        self.assertEqual(fast._cache_size(), 1)
        for name in ('engine_binding', 'source_binding', 'move_binding', 'producer_binding'):
            self.assertEqual(getattr(result.tape, name).shape, (8,))
            self.same(getattr(result.tape, name), getattr(tape, name))
        for row in (2, 3, 4, 5, 6):self.same(result.tape.words[row], tape.words[row])

    def test_original_namespace_and_empty_overflow_are_not_supported(self):
        values = self.case(2, prior=128); tape, before, *_ = values
        for name in ('engine_binding', 'source_binding'):
            changed = tape._replace(**{name:getattr(tape, name).at[0].add(1)})
            result = self.run_case((changed, *values[1:]))
            self.assertEqual(result.supported.tolist(), [False, False])
            self.same(result.tape.words, changed.words)
            self.assertEqual(result.tape.refusal.tolist(), [128 | 256] * 2)
        changed = tape._replace(cursor=tape.cursor._replace(
            engine_binding=tape.cursor.engine_binding.at[0, 0].add(1)))
        self.assertEqual(self.run_case((changed, *values[1:])).supported.tolist(), [False, True])
        empty = self.case(types=((5, 0), (6, 0)), capacity=1, prefix=False)
        over = empty[0]._replace(refusal=empty[0].refusal | public.OVERFLOW)
        result = self.run_case((over, *empty[1:]))
        self.assertFalse(bool(result.supported[0])); self.same(result.tape, over)

    def test_strict_types_checked_host_profile_and_legacy_noop(self):
        values = self.case()
        with self.assertRaises(ValueError):self.run_case((*values[:4], values[4].astype(j.float32), *values[5:]))
        with self.assertRaises(ValueError):self.run_case((values[0], np.array([1], np.int64), *values[2:]))
        with self.assertRaises(ValueError):self.run_case((*values[:5], True, values[6]))
        with self.assertRaises(ValueError):self.run_case(values, eligible=j.ones(2, bool))
        result = segment.order_segment(*values[:6], pre_writes_state=values[6], profile_handle=self.f.profile, eligible=True)
        self.assertTrue(bool(result.supported[0]))
        original = self.f.profile.device
        try:
            object.__setattr__(self.f.profile, 'device', original._replace(current_items=j.ones_like(original.current_items)))
            with self.assertRaises(ValueError):segment.order_segment(*values[:6], pre_writes_state=values[6], profile_handle=self.f.profile, eligible=True)
        finally:object.__setattr__(self.f.profile, 'device', original)
        legacy = public.CallbackTape(*values[0][:8])
        old = segment.order_segment_device(legacy, None, None, None, None, None, None, pre_writes_state=None, eligible=True)
        self.same(old.tape, legacy); self.assertFalse(bool(old.supported[0]))


if __name__ == '__main__':unittest.main()
