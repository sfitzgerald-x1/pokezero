"""Conditional INTERNAL ordering of one just-appended weather HP segment.

This is not a public formatter, damage simulator, source witness or admission
gate. The owning Session must supply its original BirthRecord/ProfileData and
the actual pre-Wish state/end_speed, pre-HP-write state AFTER Wish/weatherclock,
and after-BOTH-writes state from one core invocation. A later state, reset birth
record or caller boolean cannot establish
those facts. Host profile/catalog checks belong outside the trusted JIT closure.

Only nonlethal kind16 writes with positive unequal, source-profile-supported
cached speeds are supported. The bounded SAND/HAIL speed arithmetic is checked
against the pre-Wish snapshot; HP magnitude and Wish are never recomputed.
After-state data only validates existing receipts/late changes, never creates
receipts or determines order. Source AfterDamage/Update interleaving, a public
old-HP join, all-HP-before-faints and complete source fidelity remain separate
proofs. Ties/lethal writes stay UNKNOWN. No shared callsite is registered here.

Existing refusal bits, count, payload and cursor are retained. This helper can
only add UNKNOWN256 for unsupported eligible nonempty segments. It never clears
an emitter's UNKNOWN; a future caller may defer only that emitter's own latch.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import engine_lifecycle_callbacks as life, engine_public_events as public
from . import engine_weather_order_callbacks as annotation
from . import engine_weather_speed_profile as profile
from . import mechanics, public_mon_state, weather

WEATHER_CHIP = 16
UNKNOWN_WEATHER = 256


class SegmentResult(NamedTuple):
    tape: object
    record: profile.BirthRecord
    supported: object             # Segment only, never whole-frame admission
    reordered: object


def _data(data):
    if type(data) is not profile.ProfileData:
        raise ValueError('original checked profile device data required')
    for value in (data.binding, data.engine_binding, data.source_binding):
        profile._typed(value, (8,), jnp.uint32)
    for value in (data.species, data.birth_items, data.current_items, data.abilities):
        if value.ndim != 1 or not value.shape[0]:
            raise ValueError('nonempty profile namespace masks required')
        profile._typed(value, value.shape, jnp.bool_)
    if data.birth_items.shape != data.current_items.shape:
        raise ValueError('same physical birth/current item namespace required')


def _tape(tape):
    if not annotation._check(tape):
        return False
    b, c = tape.words.shape[:2]
    for value, shape, dtype in ((tape.turn, (b,), jnp.int32),
            (tape.source_binding, (8,), jnp.uint32),
            (tape.move_binding, (8,), jnp.uint32),
            (tape.producer_binding, (8,), jnp.uint32),
            (tape.switch_details, (b, c, 6), jnp.int32)):
        life._typed(value, shape, dtype)
    return True


def _stable_snapshots(before, after, b, *, before_writes):
    # Before the write window, Wish/clock may change these fields. Inside the
    # HP window only HP may change; no hidden clock/status/item mutation fits.
    ignored = dict(hp=None)
    if before_writes:
        ignored.update(weather=None, weather_counter=None, wish_counter=None)
    left = jax.tree.leaves(before._replace(**ignored))
    right = jax.tree.leaves(after._replace(**ignored))
    same = jnp.ones(b, bool)
    for a, z in zip(left, right, strict=True):
        if a.shape != z.shape or a.dtype != z.dtype or not a.ndim or a.shape[0] != b:
            raise ValueError('same physical before/after snapshot layout required')
        same &= jnp.all(a == z, axis=tuple(range(1, a.ndim)))
    return same


def order_segment_device(tape, before_count, pre_wish_state, after_writes_state,
                         end_speed, birth_record, data, *, pre_writes_state, eligible):
    """Validate/order at the owning seam immediately after BOTH chip writes.

    ``before_count`` is captured after weather-clock receipts and before either
    HP callback, together with the REQUIRED actual ``pre_writes_state``.
    ``eligible`` is only the operational row mask, not proof.
    A valid zero-length segment leaves its tape untouched; the returned record
    still retains the eligible rows' actual pre/post conditional profile checks.
    Callers must carry that returned record and preserve every other refusal.
    """
    if not _tape(tape):
        no = jnp.zeros_like(tape.count, dtype=bool)
        return SegmentResult(tape, birth_record, no, no)
    _data(data)
    b, capacity = tape.words.shape[:2]
    profile._state_shape(pre_wish_state, tape.stream_id)
    profile._state_shape(pre_writes_state, tape.stream_id)
    profile._state_shape(after_writes_state, tape.stream_id)
    for state in (pre_wish_state, pre_writes_state, after_writes_state):
        profile._typed(state.active, (b, 2), jnp.int32)
        profile._typed(state.weather, (b,), jnp.int32)
        profile._typed(state.weather_counter, (b,), jnp.int32)
        profile._typed(state.wish_counter, (b, 2), jnp.int32)
    profile._typed(end_speed, (b, 2), jnp.int32)
    eligible = annotation._eligible(tape, eligible)
    before_count = public_mon_state._int32(before_count)
    if before_count.shape not in ((), (b,)):
        raise ValueError('physical scalar/B before-count required')
    before_count = jnp.broadcast_to(before_count, (b,))

    retained = profile.retain_device(birth_record, pre_wish_state, tape.stream_id, data)
    retained = profile.retain_device(retained, pre_writes_state, tape.stream_id, data)
    retained = profile.retain_device(retained, after_writes_state, tape.stream_id, data)
    # BirthRecord has ONLY B-leading leaves, including all three bindings.
    retained = jax.tree.map(lambda a, z: jnp.where(
        eligible.reshape((b,) + (1,) * (a.ndim - 1)), z, a), birth_record, retained)

    rows = jnp.arange(b)
    active = pre_wish_state.active
    safe_active = jnp.clip(active, 0, 5)
    sides = jnp.arange(2)[None, :]
    hp_pre_wish = pre_wish_state.hp[rows[:, None], sides, safe_active]
    hp_before = pre_writes_state.hp[rows[:, None], sides, safe_active]
    hp_after = after_writes_state.hp[rows[:, None], sides, safe_active]
    maximum = pre_writes_state.max_hp[rows[:, None], sides, safe_active]
    status = pre_writes_state.status[rows[:, None], sides, safe_active]
    stored = pre_wish_state.stats[rows[:, None], sides, safe_active, 5]
    stages = pre_wish_state.boosts[rows[:, None], sides, safe_active, 4]
    expected_speed = mechanics.paralyzed_speed(mechanics.boosted_stat(stored, stages),
        status == mechanics.PAR)

    delta = tape.count - before_count
    bounds = ((before_count >= 0) & (before_count <= tape.count)
        & (tape.count >= 0) & (tape.count <= capacity) & (delta <= 2))
    empty = bounds & (delta == 0)
    nonempty = delta != 0
    i0 = jnp.clip(before_count, 0, capacity - 1)
    i1 = jnp.clip(before_count + 1, 0, capacity - 1)
    first, second = tape.words[rows, i0], tape.words[rows, i1]
    present0, present1 = delta >= 1, delta >= 2

    # Even an empty segment belongs to this exact seam/lifetime. A false
    # profile can be a legitimate no-event case; a foreign record cannot.
    seam_bound = (life._valid(tape.cursor)
        & (tape.stream_id == tape.cursor.stream_id)
        & jnp.all(tape.engine_binding == data.engine_binding)
        & jnp.all(tape.source_binding == data.source_binding)
        & jnp.all(tape.cursor.engine_binding == data.engine_binding, axis=1)
        & (retained.stream_id == tape.stream_id)
        & jnp.all(retained.engine_binding == data.engine_binding, axis=1)
        & jnp.all(retained.source_binding == data.source_binding, axis=1)
        & jnp.all(retained.profile_binding == data.binding, axis=1)
        & (tape.turn == tape.cursor.turn) & (tape.cursor.phase == 0)
        & ((tape.refusal & public.OVERFLOW) == 0))
    bound = retained.qualified_birth & seam_bound
    active_ok = (jnp.all((active >= 0) & (active < 6), axis=1)
        & jnp.all(active == pre_writes_state.active, axis=1)
        & jnp.all(active == after_writes_state.active, axis=1)
        & jnp.all(active == tape.cursor.physical_active, axis=1))
    math_ok = (jnp.all((end_speed > 0) & (end_speed == expected_speed), axis=1)
        & (end_speed[:, 0] != end_speed[:, 1])
        & ((pre_wish_state.weather == weather.SAND) | (pre_wish_state.weather == weather.HAIL))
        & (pre_writes_state.weather == pre_wish_state.weather)
        & (after_writes_state.weather == pre_writes_state.weather))
    active_mask = jnp.arange(6)[None, None, :] == active[:, :, None]
    snapshots = (_stable_snapshots(pre_wish_state, pre_writes_state, b, before_writes=True)
        & _stable_snapshots(pre_writes_state, after_writes_state, b, before_writes=False)
        & jnp.all(pre_wish_state.max_hp == pre_writes_state.max_hp, axis=(1, 2))
        & jnp.all(pre_writes_state.max_hp == after_writes_state.max_hp, axis=(1, 2))
        & jnp.all(jnp.where(active_mask, True, pre_wish_state.hp == pre_writes_state.hp), axis=(1, 2))
        & jnp.all(jnp.where(active_mask, True, pre_writes_state.hp == after_writes_state.hp), axis=(1, 2))
        & jnp.all((hp_pre_wish > 0) & (hp_before >= hp_pre_wish) & (hp_before <= maximum)
            & (hp_after > 0) & (hp_after <= maximum), axis=1)
        & ~pre_wish_state.done & ~pre_writes_state.done & ~after_writes_state.done)

    def receipt_ok(word, index):
        victim = word[:, 3]; safe = jnp.clip(victim, 0, 1)
        return ((word[:, 0] == WEATHER_CHIP)
            & jnp.all(word[:, 1:3] == end_speed, axis=1)
            & (victim >= 0) & (victim < 2)
            & (word[:, 4] == active[rows, safe])
            & (word[:, 5] > 0) & (word[:, 6] > 0) & (word[:, 6] < word[:, 5])
            & (word[:, 7] >= word[:, 5]) & (word[:, 7] > 0)
            & (word[:, 5] == hp_before[rows, safe])
            & (word[:, 6] == hp_after[rows, safe])
            & (word[:, 7] == maximum[rows, safe])
            & (word[:, 8] == status[rows, safe])
            & (word[:, 9] == pre_writes_state.weather)
            & (tape.event_turn[rows, index] == tape.cursor.turn)
            & (tape.event_phase[rows, index] == tape.cursor.phase)
            & jnp.all(tape.event_generation[rows, index] == tape.cursor.generation, axis=1)
            & jnp.all(tape.switch_details[rows, index] == 0, axis=1))

    seen0 = (present0 & (first[:, 3] == 0)) | (present1 & (second[:, 3] == 0))
    seen1 = (present0 & (first[:, 3] == 1)) | (present1 & (second[:, 3] == 1))
    seen = jnp.stack((seen0, seen1), axis=1)
    # Wish already finished before this exact snapshot. Every unwritten active
    # must retain HP exactly, including when Wish would mask a missing chip.
    # The public old-HP ledger join remains separate work.
    unwritten_ok = jnp.all(seen | (hp_after == hp_before), axis=1)
    valid = (bounds & bound & active_ok & math_ok & snapshots & unwritten_ok
        & (delta >= 1) & receipt_ok(first, i0)
        & (~present1 | (receipt_ok(second, i1) & (first[:, 3] != second[:, 3]))))
    use = eligible & valid
    fast_side = jnp.where(end_speed[:, 0] > end_speed[:, 1], 0, 1)
    swap = use & present1 & (first[:, 3] != fast_side)

    def swap_rows(array):
        a, z = array[rows, i0], array[rows, i1]
        mask = swap.reshape((b,) + (1,) * (a.ndim - 1))
        return array.at[rows, i0].set(jnp.where(mask, z, a)).at[rows, i1].set(jnp.where(mask, a, z))

    ordered = tape._replace(**{name: swap_rows(getattr(tape, name)) for name in
        ('words', 'event_turn', 'event_phase', 'event_generation', 'switch_details')})
    # Missing BOTH receipts must not evade the same no-unrecorded-decrease
    # validation that catches a missing singleton. No order/profile claim is
    # needed for a genuine no-event tie or a previously false birth profile.
    empty_valid = bounds & seam_bound & active_ok & snapshots & unwritten_ok
    empty_invalid = ~nonempty & ~empty_valid & ((tape.refusal & public.OVERFLOW) == 0)
    ordered = ordered._replace(refusal=tape.refusal | jnp.where(
        eligible & ((nonempty & ~valid) | empty_invalid), UNKNOWN_WEATHER, 0))
    # Overflow can hide attempted writes even when no count increment survived.
    # Keep that zero-length tape unchanged, without calling it supported.
    empty_supported = eligible & empty & empty_valid
    return SegmentResult(ordered, retained, empty_supported | use, swap)


def order_segment(tape, before_count, pre_wish_state, after_writes_state,
                  end_speed, birth_record, *, pre_writes_state, profile_handle, eligible):
    """Checked host entry; invoke outside JIT before each cached device call."""
    data = profile.check_profile(profile_handle)
    return order_segment_device(tape, before_count, pre_wish_state,
        after_writes_state, end_speed, birth_record, data,
        pre_writes_state=pre_writes_state, eligible=eligible)
