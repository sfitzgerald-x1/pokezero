"""Literal numerical component controls; no games, models or full consumer port."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_direct_hit_controls as c, critical, mechanics
from gpu_engine import engine_lifecycle_callbacks as life, engine_public_events as public
import test_gpu_engine_engine_own_facts as fixture
import test_gpu_engine_engine_item_callbacks as tapes


class DirectHitControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.DeviceOwnFactsTests.setUpClass(); cls.f = fixture.DeviceOwnFactsTests
        cls.compiled = c.compile_direct_hit_controls(engine_catalog=cls.f.lifecycle)
        cls.data = c.check_compiled(cls.compiled)
        cls.move = int(np.flatnonzero(np.asarray(cls.f.table.power) > 0)[0])

    def same(self, x, y):
        for a, b in zip(jax.tree.leaves(x), jax.tree.leaves(y)):
            np.testing.assert_array_equal(a, b)

    def tape(self, b=1, capacity=8, prior=0, legacy=False):
        state = tapes.state(b)._replace(active=j.broadcast_to(j.array([2,4], j.int32), (b,2)))
        old = tapes.tape(state, capacity, legacy=legacy)
        if legacy:
            return old
        cursor = old.cursor._replace(engine_binding=j.broadcast_to(self.data.engine_binding, (b,8)))
        return life._tape(cursor, self.f.lifecycle, capacity)._replace(refusal=j.full(b, prior, j.int32))

    def write(self, b=1, *, critical_roll=0, numerator=1, denominator=1):
        """Same existing critical/damage helpers, then the literal HP assignment.

        This is a component seam test, not a substitute for actual transition
        callsite wiring or a policy input/consumer completeness test.
        """
        move = j.full(b, self.move, j.int32)
        crit = critical.is_critical(self.f.table.critical, move, j.zeros(b, j.int32), critical_roll)
        power = self.f.table.power[move]
        num = j.broadcast_to(j.asarray(numerator, j.int32), (b,))
        den = j.broadcast_to(j.asarray(denominator, j.int32), (b,))
        old = j.full(b, 1000, j.int32)
        damage = mechanics.gen3_damage(level=80, power=power, attack=175, defense=187,
            physical=False, critical=crit, effectiveness_num=num, effectiveness_den=den, roll=100)
        new = old-damage
        return c.HitWrite(power, crit, True, False, num, den, False, False, False, old, new, old)

    def record(self, tape=None, write=None, **kwargs):
        return c.record_direct_hit_controls_device(self.tape() if tape is None else tape,
            kwargs.pop('side', 0), kwargs.pop('actor', 2), kwargs.pop('target', 4),
            kwargs.pop('move', self.move), self.write() if write is None else write,
            kwargs.pop('data', self.data), eligible=kwargs.pop('eligible', True), **kwargs)

    def test_actual_numerical_helpers_literal_payload_and_control_order(self):
        for roll, num, den, expected in ((0,2,1,[21,22]), (0,1,2,[21,23]),
                (0,1,1,[21]), (15,2,1,[22]), (15,1,2,[23]), (15,1,1,[])):
            with self.subTest(roll=roll, numerator=num, denominator=den):
                write = self.write(critical_roll=roll, numerator=num, denominator=den)
                result = self.record(write=write)
                self.assertEqual(int(result.count[0]), len(expected))
                self.assertEqual(result.words[0,:len(expected),0].tolist(), expected)
                for row in result.words[0,:len(expected)]:
                    self.assertEqual(row[1:].tolist(), [0,2,1,4,self.move,
                        int(write.old_hp[0]),int(write.new_hp[0]),num,den])
                self.assertEqual(int(result.refusal[0]), c.UNKNOWN_DIRECT_HIT)
                self.assertFalse(bool((result.words[:,:,0] == public.DIRECT_FAINT).any()))
                self.assertEqual(result.event_turn[0,:len(expected)].tolist(), [1]*len(expected))
                self.assertEqual(result.event_phase[0,:len(expected)].tolist(), [0]*len(expected))
                for index in range(len(expected)):
                    self.same(result.event_generation[:,index], result.cursor.generation)
                followed = public.condition(result, j.ones(1,j.int32), j.array([4],j.int32),
                    write.new_hp, write.maximum, j.zeros(1,j.int32), eligible=j.ones(1,bool))
                self.assertEqual(followed.words[0,:len(expected)+1,0].tolist(), expected+[public.MAIN_HP])

    def test_flags_are_not_reconstructed_from_hp_amount_and_no_hp_event_is_fabricated(self):
        write = self.write()._replace(old_hp=1000, new_hp=999, maximum=1000)
        # Arbitrary literal positive loss is retained, never recalculated.
        out = self.record(write=write)
        self.assertEqual(out.words[0,0,6:8].tolist(), [1000,999])
        plain = self.record(write=write._replace(critical=False))
        self.assertEqual(int(plain.count[0]), 0)
        self.assertTrue(int(plain.refusal[0]) & c.UNKNOWN_DIRECT_HIT)
        self.assertFalse(hasattr(out, 'complete'))

    def test_unsupported_and_malformed_writes_remain_unknown_without_rows(self):
        base = self.write()
        for changes in (dict(hit=False), dict(immune=True), dict(substitute=True), dict(protected=True),
                dict(enduring=True), dict(new_hp=0), dict(new_hp=1000), dict(new_hp=1001),
                dict(old_hp=0), dict(maximum=999), dict(power=0), dict(power=-1),
                dict(numerator=0), dict(denominator=0), dict(numerator=-1), dict(denominator=-1),
                dict(numerator=3), dict(numerator=4,denominator=4),
                dict(power=True), dict(new_hp=1.5), dict(critical=1), dict(hit=j.ones((1,1),bool))):
            with self.subTest(changes=changes):
                out = self.record(write=base._replace(**changes))
                self.assertEqual(int(out.count[0]), 0)
                self.assertTrue(int(out.refusal[0]) & c.UNKNOWN_DIRECT_HIT)
        for args in (dict(side=-1), dict(side=2), dict(actor=-1), dict(actor=6),
                dict(target=-1), dict(target=6), dict(actor=0), dict(target=0),
                dict(move=0), dict(move=int(self.data.move_count)), dict(move=1.5)):
            with self.subTest(args=args):
                out = self.record(**args)
                self.assertEqual(int(out.count[0]), 0)
                self.assertTrue(int(out.refusal[0]) & c.UNKNOWN_DIRECT_HIT)

    def test_b8_globals_inactive_rows_and_invalid_claim_isolation(self):
        tape = self.tape(8, prior=128)
        mask = j.array([True,False]*4)
        write = self.write(8, numerator=2)._replace(new_hp=j.array([0,999,900,999,900,999,900,999],j.int32))
        out = self.record(tape, write, eligible=mask)
        self.assertEqual(out.count.tolist(), [0,0,2,0,2,0,2,0])
        self.assertEqual(out.refusal.tolist(), [384,128]*4)
        for name in ('words','count','refusal','turn','event_turn','event_phase','event_generation','switch_details'):
            self.same(getattr(out,name)[~mask], getattr(tape,name)[~mask])
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):
            with self.subTest(name=name):
                bad = self.data._replace(**{name:getattr(self.data,name).at[7].add(1)})
                result = self.record(tape, self.write(8,numerator=2), data=bad, eligible=mask)
                self.assertEqual(result.count.tolist(), [0]*8)
                self.assertEqual(result.refusal.tolist(), [384,128]*4)

    def test_raw_validator_finite_literal_ratios_kind_semantics_and_typed_storage(self):
        rows = []
        for num, den in c.POSITIVE_EFFECTIVENESS_RATIOS:
            rows.append([c.CRITICAL,0,2,1,4,self.move,1000,999,num,den])
            if num != den:
                rows.append([c.SUPER_EFFECTIVE if num > den else c.RESISTED,
                    0,2,1,4,self.move,1000,999,num,den])
        raw = j.asarray(rows,j.int32)
        self.assertTrue(bool(c.validate_raw_device(raw).all()))
        self.same(jax.jit(c.validate_raw_device)(raw), c.validate_raw_device(raw))
        base = raw[:2]
        for column, value in ((0,0),(0,24),(1,-1),(1,2),(2,6),(3,0),(4,-1),(4,6),
                (5,0),(6,999),(7,0),(7,1000),(8,0),(8,3),(8,8),(9,0),(9,3)):
            with self.subTest(column=column,value=value):
                changed = base.at[0,column].set(value)
                self.assertEqual(c.validate_raw_device(changed).tolist(), [False,True])
        neutral = raw[:1].at[0,0].set(c.SUPER_EFFECTIVE)
        self.assertFalse(bool(c.validate_raw_device(neutral)[0]))
        reversed_control = raw[:1].at[0,0].set(c.RESISTED).at[0,8].set(2)
        self.assertFalse(bool(c.validate_raw_device(reversed_control)[0]))
        for bad in (raw.astype(j.float32), raw[0], raw[:,:9], raw[:0]):
            with self.assertRaises(ValueError):
                c.validate_raw_device(bad)

    def test_atomic_capacity_prefix_stamps_prior_bits_and_legacy(self):
        write = self.write(numerator=2)
        tape = self.tape(capacity=1, prior=512)
        out = self.record(tape, write)
        self.assertEqual(int(out.count[0]), 0)
        self.assertEqual(int(out.refusal[0]), 512|256|public.OVERFLOW)
        self.same(out.words, tape.words)
        self.same(out.event_generation, tape.event_generation)
        tape = self.tape(capacity=3, prior=64)
        tape = public.condition(tape, j.ones(1,j.int32), j.array([4],j.int32), j.array([1000],j.int32),
            j.array([1000],j.int32), j.array([0],j.int32), eligible=j.array([True]))
        out = self.record(tape, write)
        self.assertEqual(out.words[0,:,0].tolist(), [public.MAIN_HP,21,22])
        self.same(out.words[:,0], tape.words[:,0])
        self.assertEqual(int(out.refusal[0]), 64|256)
        stale = self.tape(capacity=2)._replace(switch_details=j.zeros((1,2,6),j.int32).at[0,1,0].set(1))
        self.assertEqual(int(self.record(stale,write).count[0]), 0)
        legacy = self.tape(legacy=True)
        self.assertIs(self.record(legacy,write), legacy)

    def test_capacity_diagnostic_ignores_existing_prefix_switch_details(self):
        tape = self.tape(2,capacity=1,prior=512)
        tape = tape._replace(count=j.ones(2,j.int32), words=tape.words.at[:,0,0].set(life.SWITCH_INSTALLED),
            switch_details=tape.switch_details.at[0,0,0].set(1))
        write = self.write(2)._replace(critical=True,numerator=1,denominator=1)
        out = self.record(tape,write)
        self.same(out.words,tape.words)
        self.same(out.switch_details,tape.switch_details)
        self.same(out.event_turn,tape.event_turn)
        self.same(out.event_generation,tape.event_generation)
        self.assertEqual(out.count.tolist(), [1,1])
        self.assertEqual(out.refusal.tolist(), [512|256|public.OVERFLOW]*2)
        # Even stale NEW-slot details cannot hide simultaneous capacity loss.
        partial = self.tape(capacity=2)._replace(count=j.ones(1,j.int32),
            switch_details=j.zeros((1,2,6),j.int32).at[0,1,0].set(1))
        result = self.record(partial,self.write(numerator=2))
        self.assertEqual(int(result.refusal[0]), 256|public.OVERFLOW)
        self.same(result.words,partial.words)
        self.assertEqual(int(result.count[0]),1)

    def test_hot_jit_has_one_cache_no_callbacks_and_original_host_guards(self):
        tape = self.tape(8); write = self.write(8,numerator=2)
        call = lambda t,w:c.record_direct_hit_controls_device(t,0,2,4,self.move,w,self.data,eligible=True)
        fast = jax.jit(call); expected = call(tape,write)
        for _ in range(2):
            self.same(fast(tape,write), expected)
        self.assertEqual(fast._cache_size(), 1)
        text = str(jax.make_jaxpr(call)(tape,write))
        for forbidden in ('pure_callback','io_callback','debug_callback'):
            self.assertNotIn(forbidden, text)
        with self.assertRaises(ValueError):
            c.check_compiled(replace(self.compiled))
        old = self.compiled.device
        try:
            object.__setattr__(self.compiled, 'device', old._replace(move_count=j.array(old.move_count,copy=True)))
            with self.assertRaises(ValueError):
                c.check_compiled(self.compiled)
        finally:
            object.__setattr__(self.compiled, 'device', old)
        for host in (lambda:c.compile_direct_hit_controls(engine_catalog=self.f.lifecycle),
                lambda:c.check_compiled(self.compiled),
                lambda:c.record_direct_hit_controls(tape,0,2,4,self.move,write,compiled=self.compiled,eligible=True)):
            for dummy in (False,True):
                with patch.object(public, 'check_catalog', side_effect=AssertionError('late catalog guard')):
                    with self.assertRaisesRegex(ValueError, 'cannot be traced'):
                        jax.jit(lambda _:host())(j.array(1)) if dummy else jax.jit(lambda:host())()


if __name__ == '__main__':
    unittest.main()
