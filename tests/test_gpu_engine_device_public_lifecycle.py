"""PR0 public lifecycle writes: the faint-time clear, the entry reveal gates, the type revert.

Synthetic hook transactions only: no engine turn, model, fold or game. B2 references:
gpu_engine/public_active.py apply (FAINT), gpu_engine/public_types.py (ARRIVE clears the
seat's overlay), Showdown gen3 Intimidate/setWeather (see the boundary-parity test).
"""
from types import SimpleNamespace as NS
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_device_public_hooks as h, engine_device_trackers as tr
from test_gpu_engine_device_public_hooks import Belief, state, tables

KEYS = ('confusion_turns', 'encore_turns', 'stall_counter', 'sub_present', 'item_swapped', 'meanlook_trap')
SPECIES_TYPES = j.asarray([[1, 0]] * 7 + [[1, 0], [2, 3], [4, 0], [5, 6], [1, 2], [3, 0]] + [[1, 0]] * 7, j.int32)


def typed_tables():
    t = tables()
    return Belief(t.species_initial_mask, t.move_mask, t.ability_mask, t.item_mask, SPECIES_TYPES,
                  t.public_move, t.source_move_evidence)


def entry_table(*, intimidate=(), weather=(), immune=()):
    flag = j.zeros(10, bool)
    climate = j.zeros(10, j.int32)
    for ability, kind in weather:climate = climate.at[ability].set(kind)
    return NS(weather_ability=climate, intimidate=flag.at[j.asarray(intimidate, j.int32)].set(True) if intimidate else flag,
              attack_drop_immune=flag.at[j.asarray(immune, j.int32)].set(True) if immune else flag, trace=None, own_tempo=None)


def set_active(c, s, side, values, use):
    p = s.active[:, side]
    for name, value in values.items():h.feature(c, side, p, name, j.full(len(p), value, j.int32), use)


class FaintClearTests(unittest.TestCase):
    def prepared(self, b=3):
        s = state(b);c = h.initialize(s, tables());yes = j.ones(b, bool)
        # Side 1 reveals a bench member (physical 3, public slot 1) that keeps its own values.
        bench = s._replace(active=s.active.at[:, 1].set(3));h.install(c, bench, 1, yes)
        set_active(c, bench, 1, dict(confusion_turns=4, stall_counter=3), yes)
        h.install(c, s, 1, yes)
        for name in ('confusion', 'substitute', 'leechseed'):h.volatile(c, s, 1, name, yes, yes)
        h.volatile(c, s, 0, 'leechseed', yes, yes)
        set_active(c, s, 1, dict(confusion_turns=2, encore_turns=1, stall_counter=2, sub_present=1, item_swapped=1,
                                 meanlook_trap=1, turns_active=3), yes)
        set_active(c, s, 0, dict(meanlook_trap=1, stall_counter=1, confusion_turns=1), yes)
        return s, c

    def test_faint_clears_b2_fields_of_the_fainted_active_and_meanlook_on_both_seats(self):
        s, c = self.prepared();before = c.trackers
        after = s._replace(hp=s.hp.at[0, 1, 0].set(0).at[2, 1, 0].set(0), done=s.done.at[2].set(True))
        dead = s._replace(done=s.done.at[2].set(True))         # row 2: an already finished game is left alone
        h.turn_end(c, dead, after, j.zeros(3, bool))
        t = c.trackers;f = t.mon_features
        self.assertFalse(bool(t.public_volatiles[0, 1, 0].any()))
        for name in KEYS:self.assertEqual(int(f[name][0, 1, 0]), 0, name)
        self.assertEqual(int(f['meanlook_trap'][0, 0, 0]), 0)                 # the other seat's Mean Look
        # Not part of B2's FAINT: the survivor's own bag/clocks, the fainted mon's turn count, benches.
        self.assertTrue(bool(t.public_volatiles[0, 0, 0, h.VOLATILE_NAMES.index('leechseed')]))
        self.assertEqual((int(f['stall_counter'][0, 0, 0]), int(f['confusion_turns'][0, 0, 0])), (1, 1))
        self.assertEqual(int(f['turns_active'][0, 1, 0]), 3)
        self.assertEqual((int(f['confusion_turns'][0, 1, 1]), int(f['stall_counter'][0, 1, 1])), (4, 3))
        self.assertTrue(bool(t.fainted[0, 1, 0]))
        # Rows without a faint (1) and finished rows (2) are bitwise unchanged.
        for row in (1, 2):
            for name in ('public_volatiles', 'shown_types'):
                np.testing.assert_array_equal(getattr(t, name)[row], getattr(before, name)[row], err_msg=name)
            for name in KEYS:np.testing.assert_array_equal(f[name][row], before.mon_features[name][row], err_msg=name)

    def test_only_an_hp_transition_this_step_clears(self):
        s, c = self.prepared(2)
        fainted = s._replace(hp=s.hp.at[:, 1, 0].set(0))
        h.turn_end(c, s, fainted, j.zeros(2, bool))
        # A value written afterwards on the still-fainted active survives a later step without a transition.
        set_active(c, fainted, 1, dict(confusion_turns=5), j.ones(2, bool))
        h.turn_end(c, fainted, fainted, j.zeros(2, bool))
        self.assertEqual(c.trackers.mon_features['confusion_turns'][:, 1, 0].tolist(), [5, 5])
        # A mon that entered this step (healthy on the bench before) and fainted on entry is cleared too.
        s2, c2 = self.prepared(2);yes = j.ones(2, bool)
        entered = s2._replace(active=s2.active.at[:, 1].set(3));h.install(c2, entered, 1, yes)
        h.volatile(c2, entered, 1, 'substitute', yes, yes)
        spiked = entered._replace(hp=entered.hp.at[:, 1, 3].set(0))
        h.turn_end(c2, s2, spiked, j.zeros(2, bool))
        self.assertFalse(bool(c2.trackers.public_volatiles[:, 1, 1].any()))

    def test_ordinary_turn_end_keeps_the_cleared_clocks_and_projects_like_b2(self):
        from gpu_engine import engine_device_features as features
        s, c = self.prepared(2)
        live = s._replace(confusion_counter=s.confusion_counter.at[:, 1, 0].set(2), encore_counter=s.encore_counter.at[:, 1, 0].set(2))
        after = live._replace(hp=live.hp.at[0, 1, 0].set(0))
        h.turn_end(c, live, after, j.ones(2, bool))
        f = c.trackers.mon_features
        self.assertEqual(f['confusion_turns'][:, 1, 0].tolist(), [0, 3])          # fainted stays 0; survivor ticks
        self.assertEqual(f['encore_turns'][:, 1, 0].tolist(), [0, 2])
        view = tr.readout(c.trackers, j.zeros(2, j.int32))
        self.assertTrue(bool(view.active[0, 0]))                                   # still the active token
        vocab = features.Tables(j.arange(len(h.VOLATILE_NAMES), dtype=j.int32), j.zeros(4, j.int32), j.zeros(2, j.int32),
                                j.zeros(10, j.int32), j.asarray(0, j.int32))
        n, cat, _ = features.project_mon(j.zeros((2, 6, 132)), j.zeros((2, 6, 41), j.int32), view.mon_features,
            active=view.active, present=view.present, tables=vocab, volatiles=view.volatiles,
            substitute_fraction=j.where(view.mon_features['sub_present'], .25, 0))
        from pokezero import showdown as ref
        cols = [ref.v4_numeric_index(getattr(ref, 'NUMERIC_' + n)) for n in
                ('STALL_COUNTER', 'CONFUSION_TURNS', 'ENCORE_TURNS', 'MEANLOOK_TRAP', 'ITEM_SWAPPED', 'SUB_HP_FRACTION')]
        self.assertFalse(bool(n[0, 0, j.asarray(cols)].any()))
        self.assertFalse(bool(cat[0, 0, ref.CATEGORY_VOLATILE_OFFSET:ref.CATEGORY_VOLATILE_OFFSET + 6].any()))
        self.assertTrue(bool(n[1, 0, j.asarray(cols)].any()))

    def test_jit_matches_eager(self):
        s, c = self.prepared(2);start = c.trackers
        after = s._replace(hp=s.hp.at[1, 1, 0].set(0))
        eager = h.Context(start, tables());h.turn_end(eager, s, after, j.ones(2, bool))
        jitted = jax.jit(lambda t, b, a: (lambda ctx: (h.turn_end(ctx, b, a, j.ones(2, bool)), ctx.trackers)[1])(h.Context(t, tables())))
        for x, y in zip(jax.tree.leaves(eager.trackers), jax.tree.leaves(jitted(start, s, after))):np.testing.assert_array_equal(x, y)


class EntryRevealGateTests(unittest.TestCase):
    def test_intimidate_reveals_only_against_a_living_foe_without_substitute(self):
        b = 4;s = state(b)._replace(ability=state(b).ability.at[:, 0, 0].set(3).at[:, 1, 0].set(5))
        c = h.initialize(s, tables());table = entry_table(intimidate=(3,), immune=(5,))
        before = s._replace(substitute_hp=s.substitute_hp.at[1, 1, 0].set(50), hp=s.hp.at[2, 1, 0].set(0))
        h.entry(c, before, before, 0, table, j.array([True, True, True, False]))
        t = c.trackers
        self.assertEqual(t.shown_ability[:, 0, 0].tolist(), [3, 0, 0, 0])            # open foe / Sub / fainted / unused
        self.assertEqual(t.original_ability[:, 0, 0].tolist(), [3, 0, 0, 0])
        self.assertEqual(t.shown_ability[:, 1, 0].tolist(), [5, 0, 0, 0])            # Clear Body's own -fail line
        self.assertEqual(t.belief_mask[:, 0, 0, 0].tolist(), [5, 15, 15, 15])

    def test_weather_ability_reveals_only_when_the_field_changes(self):
        SAND, RAIN = 3, 1
        b = 5;s = state(b)._replace(ability=state(b).ability.at[:, 0, 0].set(4))
        c = h.initialize(s, tables());table = entry_table(weather=((4, SAND),))
        weather = j.array([0, SAND, SAND, RAIN, SAND], j.int32);counter = j.array([0, 0, 3, 2, 0], j.int32)
        before = s._replace(weather=weather, weather_counter=counter)
        after = before._replace(weather=j.full(b, SAND, j.int32), weather_counter=j.zeros(b, j.int32))
        h.entry(c, before, after, 0, table, j.array([True, True, True, True, False]))
        # None, same permanent (silent no-op), same temporary (re-set), different, unused row.
        self.assertEqual(c.trackers.shown_ability[:, 0, 0].tolist(), [4, 0, 4, 4, 0])
        self.assertEqual(c.trackers.field_features['weather_permanent'].tolist(), [True, False, True, True, False])
        # A fainted entrant (hazard KO) never starts its ability.
        c2 = h.initialize(s, tables());h.entry(c2, before, after._replace(hp=after.hp.at[:, 0, 0].set(0)), 0, table, j.ones(b, bool))
        self.assertFalse(bool(c2.trackers.shown_ability[:, 0, 0].any()))


class TypeRevertTests(unittest.TestCase):
    def kecleon(self, b=2, belief=True):
        s = state(b);types = SPECIES_TYPES[s.species]
        s = s._replace(types=types, base_types=types)
        return s, h.initialize(s, typed_tables() if belief else None)

    def depart_to(self, c, before, side, incoming, use, *, drag=False):
        """One actual departure: the engine reverts the outgoing types, then the incoming installs."""
        p = before.active[:, side];rows = j.arange(len(p))
        reverted = before._replace(types=before.types.at[rows, side, p].set(before.base_types[rows, side, p]))
        arrived = reverted._replace(active=reverted.active.at[:, side].set(incoming))
        if drag:
            h.drag(c, before, arrived, side, None, use);return arrived
        h.depart(c, before, reverted, side, None, use);h.install(c, arrived, side, use);h.withdraw(c, arrived, side, p, use)
        return arrived

    def test_switch_out_reverts_a_color_change_to_species_types(self):
        s, c = self.kecleon();yes = j.ones(2, bool)
        hit = s._replace(types=s.types.at[:, 1, 0].set(j.array([5, 0], j.int32)))
        h.color_change(c, s, hit, 1, yes)
        self.assertEqual(c.trackers.shown_types[:, 1, 0].tolist(), [[5, 0]] * 2)
        out = self.depart_to(c, hit, 1, j.full(2, 3, j.int32), j.array([True, False]))
        self.assertEqual(c.trackers.shown_types[:, 1, 0].tolist(), [[1, 0], [5, 0]])     # row 1 never left
        view = tr.readout(c.trackers, j.zeros(2, j.int32))
        self.assertEqual(view.types[0, :2].tolist(), [[1, 0], SPECIES_TYPES[10].tolist()])
        # Re-entry still shows the species types (install) until the next Color Change.
        back = self.depart_to(c, out, 1, j.zeros(2, j.int32), j.array([True, False]))
        self.assertEqual(c.trackers.shown_types[0, 1, 0].tolist(), [1, 0])
        self.assertEqual(back.active[:, 1].tolist(), [0, 0])

    def test_drag_and_replacement_departures_revert_and_faint_keeps_the_overlay(self):
        s, c = self.kecleon();yes = j.ones(2, bool)
        hit = s._replace(types=s.types.at[:, 1, 0].set(j.array([4, 0], j.int32)))
        h.color_change(c, s, hit, 1, yes)
        self.depart_to(c, hit, 1, j.full(2, 2, j.int32), j.array([True, False]), drag=True)
        self.assertEqual(c.trackers.shown_types[:, 1, 0].tolist(), [[1, 0], [4, 0]])
        # B2 keeps the overlay at faint (the fainted mon is still current); the replacement's arrival clears it.
        dead = hit._replace(hp=hit.hp.at[:, 1, 0].set(0))
        h.turn_end(c, hit, dead, j.zeros(2, bool))
        self.assertEqual(c.trackers.shown_types[1, 1, 0].tolist(), [4, 0])
        self.depart_to(c, dead, 1, j.full(2, 4, j.int32), j.array([False, True]))
        self.assertEqual(c.trackers.shown_types[:, 1, 0].tolist(), [[1, 0], [1, 0]])

    def test_departure_without_a_type_change_writes_nothing(self):
        s, c = self.kecleon();yes = j.ones(2, bool)
        c.trackers = h._write(c.trackers, 'shown_types', 1, s.active[:, 1], j.full((2, 2), 9, j.int32), yes)
        self.depart_to(c, s, 1, j.full(2, 3, j.int32), yes)
        self.assertEqual(c.trackers.shown_types[:, 1, 0].tolist(), [[9, 9]] * 2)

    def test_without_species_tables_the_overlay_is_cleared(self):
        s, c = self.kecleon(belief=False);yes = j.ones(2, bool)
        hit = s._replace(types=s.types.at[:, 1, 0].set(j.array([5, 0], j.int32)))
        h.color_change(c, s, hit, 1, yes)
        self.depart_to(c, hit, 1, j.full(2, 3, j.int32), yes)
        self.assertEqual(c.trackers.shown_types[:, 1, 0].tolist(), [[0, 0]] * 2)


class MeanLookArrivalTests(unittest.TestCase):
    """B2 clears Mean Look on both seats at every arrival (public_active.py:246), Baton included.

    Showdown keeps the trap through Baton Pass (a known B2 quirk the device matches on purpose)."""
    def trapped(self, b=3):
        s = state(b);c = h.initialize(s, tables());yes = j.ones(b, bool)
        set_active(c, s, 1, dict(meanlook_trap=1, stall_counter=2), yes)      # side 1 is trapped by side 0
        set_active(c, s, 0, dict(meanlook_trap=1), yes)                       # (and side 0 too: both seats clear)
        return s, c

    def flags(self, c, s):
        f = c.trackers.mon_features['meanlook_trap'];r = j.arange(len(s.done))
        return [[int(f[i, side, int(c.trackers.active_slot[i, side])]) for side in (0, 1)] for i in range(len(s.done))]

    def arrive(self, c, s, side, incoming, use, *, baton=False):
        old = s.active[:, side];after = s._replace(active=s.active.at[:, side].set(incoming))
        h.install(c, after, side, use)
        if baton:h.baton_copy(c, after, side, old, use)
        h.withdraw(c, after, side, old, use)
        return after

    def test_switch_and_replacement_arrivals_clear_both_seats(self):
        s, c = self.trapped();use = j.array([True, False, True])
        self.arrive(c, s, 0, j.full(3, 2, j.int32), use)                     # the trapper switches out
        self.assertEqual(self.flags(c, s._replace(active=s.active.at[:, 0].set(j.array([2, 0, 2], j.int32)))),
                         [[0, 0], [1, 1], [0, 0]])
        # The other seat keeps its other clocks; only Mean Look is a both-seat clear.
        self.assertEqual(c.trackers.mon_features['stall_counter'][:, 1, 0].tolist(), [2, 2, 2])

    def test_drag_clears_both_seats(self):
        s, c = self.trapped(2)
        after = s._replace(active=s.active.at[:, 0].set(3))
        h.drag(c, s, after, 0, None, j.array([True, False]))
        self.assertEqual(c.trackers.mon_features['meanlook_trap'][:, 1, 0].tolist(), [0, 1])

    def test_baton_target_clears_both_seats_in_both_directions(self):
        for passer in (0, 1):
            s, c = self.trapped(2);yes = j.ones(2, bool)
            after = self.arrive(c, s, passer, j.full(2, 4, j.int32), yes, baton=True)
            self.assertEqual(self.flags(c, after), [[0, 0], [0, 0]], passer)
        # Other Baton-transferable public state still rides the pass.
        s, c = self.trapped(2);yes = j.ones(2, bool)
        set_active(c, s, 1, dict(sub_present=1, confusion_turns=2), yes)
        after = self.arrive(c, s, 1, j.full(2, 4, j.int32), yes, baton=True)
        f = c.trackers.mon_features;slot = int(c.trackers.active_slot[0, 1])
        self.assertEqual((int(f['sub_present'][0, 1, slot]), int(f['confusion_turns'][0, 1, slot]),
                          int(f['meanlook_trap'][0, 1, slot])), (1, 2, 0))

    def test_jitted_vector_seat_install_clears_the_other_seat_per_row(self):
        s, c = self.trapped(2)
        side = j.array([0, 1], j.int32);incoming = j.array([2, 3], j.int32)
        after = s._replace(active=s.active.at[j.arange(2), side].set(incoming))
        out = jax.jit(lambda t, st, sd: h.install(h.Context(t, tables()), st, sd, j.ones(2, bool)).trackers)(c.trackers, after, side)
        f = out.mon_features['meanlook_trap']
        self.assertEqual((int(f[0, 1, 0]), int(f[1, 0, 0])), (0, 0))          # each row's other seat


if __name__ == '__main__':
    unittest.main()
