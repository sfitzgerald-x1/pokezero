"""The batched transition PR (bugs F, G, F1, F2) at the device public hooks, and its GPU-regression plug-in (TB).

Synthetic hook transactions (the device public hooks' own fixtures, test_gpu_engine_device_public_hooks), no engine turn:
- turn_end advances a held clock (confusion, Encore, partial trap) on the TURN line only, B2's rule (public_active.py:240):
  not on a battle-ending step, not on a step that ends in a replacement request, and on the replacement step that prints the
  TURN; it counts a weather upkeep only where the residual ran (`residual`), and runs the upkeep where `upkeep` says;
- move(from_tag=True), a Pursuit intercept's |move|...|[from] Pursuit line, records no last move, Choice lock or Truant anchor
  and still reveals the move and counts the tendencies;
- the TB plug-in's involvement predicates and allowances (pure numpy, gpu_engine.v2e_regress).
The engine paths (transition.step_with_chance's intercept, Sleep Talk call and Perish upkeep line) are covered end to end by
the effect probes (gpu_engine/effect_probes: pursuit_intercept_*, counter_sleeptalk, perishsong_end_*, the battle-ending
twins, encore_survivor_replacement_turn, wrap_sand_move_ko_distinct, perishsong_damaged_upkeep_sweep).
"""
import importlib.util
import unittest

import numpy as np

HAS_JAX = importlib.util.find_spec('jax') is not None


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class TurnEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax.numpy as j
        from gpu_engine import engine_device_public_hooks as h
        from test_gpu_engine_device_public_hooks import state, tables
        cls.j, cls.h, cls.state, cls.tables = j, h, staticmethod(state), staticmethod(tables)

    def held(self, b=4):
        """A carry and a state whose side-1 active holds confusion and Encore (counters 2) with its clocks at 1."""
        j, h = self.j, self.h
        s = self.state(b);c = h.initialize(s, self.tables());p = s.active[:, 1];yes = j.ones(b, bool)
        for name in ('confusion_turns', 'encore_turns'):h.feature(c, 1, p, name, j.ones(b, j.int32), yes)
        s = s._replace(confusion_counter=s.confusion_counter.at[:, 1, 0].set(2), encore_counter=s.encore_counter.at[:, 1, 0].set(2))
        return s, c

    def clocks(self, c):
        f = c.trackers.mon_features
        return [np.asarray(f[name][:, 1, 0]).tolist() for name in ('confusion_turns', 'encore_turns')]

    def test_held_clocks_advance_on_the_turn_line_only(self):
        """Row 0: an ordinary step that prints its TURN (+1). Row 1: a battle-ending step (no TURN: unchanged). Row 2: an
        ordinary step whose residual faint asks for a replacement (no TURN yet: unchanged). Row 3: that replacement step,
        not ordinary, prints the TURN (+1)."""
        j = self.j;s, c = self.held()
        after = s._replace(turn=s.turn + 1, done=s.done.at[1].set(True), forced_switch=s.forced_switch.at[2, 0].set(True))
        self.h.turn_end(c, s, after, j.asarray([True, True, True, False]))
        self.assertEqual(self.clocks(c), [[2, 1, 1, 2], [2, 1, 1, 2]])
        self.assertEqual(np.asarray(c.trackers.turn).tolist(), [2, 1, 1, 2])   # the TURN line's own gate: number after.turn + 1, else kept

    def test_a_cleared_clock_still_resets_on_an_ordinary_step(self):
        j = self.j;s, c = self.held(2)
        after = s._replace(turn=s.turn + 1, confusion_counter=s.confusion_counter.at[:, 1, 0].set(0), done=s.done.at[1].set(True))
        self.h.turn_end(c, s, after, j.ones(2, bool))
        self.assertEqual(self.clocks(c), [[0, 0], [2, 1]])

    def test_weather_upkeeps_count_where_the_residual_ran(self):
        """A battle ended in the move phase is `ordinary` but ran no residual, so printed no -weather ...|[upkeep] line."""
        j = self.j;s, c = self.held(3)
        after = s._replace(turn=s.turn + 1, weather=j.full(3, 3, j.int32), done=s.done.at[1].set(True).at[2].set(True))
        self.h.turn_end(c, s, after, j.ones(3, bool), residual=j.asarray([True, False, True]))
        self.assertEqual(np.asarray(c.trackers.field_features['weather_upkeeps']).tolist(), [1, 0, 1])
        s2, c2 = self.held(3)
        self.h.turn_end(c2, s2, after, j.ones(3, bool))                         # residual omitted: ordinary, as before
        self.assertEqual(np.asarray(c2.trackers.field_features['weather_upkeeps']).tolist(), [1, 1, 1])

    def test_the_upkeep_mask_runs_on_a_battle_ending_step(self):
        """F1: a battle ended by Perish Song's queued faints prints |upkeep (transition decides the mask); the port's
        post_upkeep follows it, done or not."""
        j = self.j;s, c = self.held(2)
        after = s._replace(turn=s.turn + 1, done=j.ones(2, bool))
        self.h.turn_end(c, s, after, j.ones(2, bool), upkeep=j.asarray([True, False]))
        self.assertEqual(np.asarray(c.trackers.public_port.post_upkeep).tolist(), [True, False])


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class FromTagMoveTests(unittest.TestCase):
    def test_a_from_tagged_header_records_no_last_move_and_still_reveals(self):
        import jax.numpy as j
        from gpu_engine import engine_device_public_hooks as h
        from test_gpu_engine_device_public_hooks import state, tables
        s = state();yes = j.ones(2, bool)
        plain, tagged = h.initialize(s, tables()), h.initialize(s, tables())
        for c in (plain, tagged):h.move(c, s, 1, j.full(2, 7, j.int32), yes, pp_use=False)        # a prior move: last move 7
        h.move(plain, s, 1, j.full(2, 8, j.int32), yes, pp_use=True)
        h.move(tagged, s, 1, j.full(2, 8, j.int32), yes, pp_use=True, from_tag=True)
        def last(c):return np.asarray(c.trackers.last_moves)[:, 1, 0].tolist()
        self.assertEqual(last(plain)[0], int(tables().public_move[8]));self.assertEqual(last(tagged)[0], int(tables().public_move[7]))
        for name in ('observed_moves', 'pp_used'):
            np.testing.assert_array_equal(getattr(plain.trackers, name), getattr(tagged.trackers, name))   # revealed and charged alike
        for name in ('moves_shown', 'stayed_and_attacked'):
            np.testing.assert_array_equal(plain.trackers.mon_features[name], tagged.trackers.mon_features[name])
        np.testing.assert_array_equal(plain.trackers.tendency.opportunities, tagged.trackers.tendency.opportunities)


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class PerishUpkeepSweepTests(unittest.TestCase):
    def test_a_perish_faint_after_the_upkeep_line_is_swept_as_living(self):
        """Showdown prints |upkeep before a Perish end-path |faint|, so B2's non-proc sweep sees the perishing mon alive: hit to
        80% (an action-phase HP line, no Leftovers heal) and observed fainted by the device at its perish0, it still loses
        Leftovers when turn_end's upkeep_living marks it; without the mark (a mon fainted before the |upkeep), it keeps it."""
        import jax
        import jax.numpy as j
        from gpu_engine import engine_device_negative_items as n, engine_device_public_hooks as h
        from test_gpu_engine_device_negative_items import fixture
        s, tables, leftovers, _ = fixture(survivors=(0, 1), count=8)
        def run(state, living):
            c = h.initialize(state, tables)
            hit = state._replace(hp=state.hp.at[:, 1, 0].set(160))
            h.negative_hp_line(c, hit, 1, j.ones(2, bool), source_class=n.HP_ACTION);h.observe(c, hit, j.ones(2, bool))
            dead = hit._replace(hp=hit.hp.at[:, 1, 0].set(0))
            h.observe(c, dead, j.ones(2, bool), record_damage=False)      # the perish0 faint, observed at the end path
            h.turn_end(c, hit, dead, j.ones(2, bool), upkeep=j.ones(2, bool), leftovers=leftovers, upkeep_living=living)
            return c.trackers.belief_mask
        swept = jax.jit(run)(s, j.asarray([[False, True], [False, False]]))
        self.assertEqual(np.asarray(swept)[:, 1, 0].tolist(), [[3, 0], [255, 0]])   # row 0 marked: Leftovers ruled out


class TBPlugInTests(unittest.TestCase):
    VOCAB = dict(moves=['_none', 'tackle', 'pursuit', 'sleeptalk', 'encore', 'wrap', 'counter', 'struggle', 'sunnyday'],
                 abilities=['_none', 'pressure', 'drought'], items=['_none', 'leftovers', 'figyberry'])
    TACKLE, PURSUIT, SLEEPTALK, ENCORE, WRAP, SUNNYDAY = 1, 2, 3, 4, 5, 8

    def pre(self, edit, L=12):
        from gpu_engine.v2e_regress import Pre
        z = lambda *s: np.zeros(s, np.int32)
        s = {k: z(L, 2, 6) for k in ('status', 'charge_move', 'encore_move', 'encore_counter', 'ability', 'item', 'confusion_counter',
                                      'partial_trap_counter', 'perish_counter')}
        s.update(hp=np.full((L, 2, 6), 50), rest_origin=np.zeros((L, 2, 6), bool), volatile=np.zeros((L, 2, 6), np.uint32), active=z(L, 2),
                 pending_residual=np.zeros(L, bool), queued_actor=np.full(L, -1), queued_actions=z(L, 2), last_action_kind=z(L, 2),
                 last_action_index=np.full((L, 2), -1), forced_switch=np.zeros((L, 2), bool), moves=np.full((L, 2, 6, 4), self.TACKLE),
                 weather=z(L))
        actions = np.zeros((L, 2), np.int32);edit(s, actions)
        named = {f'state.{k}': v for k, v in s.items()};named['actions'] = actions
        callable_ = np.array([False, True, True, False, True, True, True, False, True])
        return Pre(named, self.VOCAB, callable_, np.zeros((L, 2), bool))

    def test_predicates(self):
        from gpu_engine.v2e_regress import PLUGINS
        def edit(s, a):
            s['moves'][0, 1, 0, 0] = self.PURSUIT;a[0, 0] = 4                     # lane 0: side 1 Pursuits a switching side 0
            s['moves'][1, 1, 0, 0] = self.PURSUIT                                 # lane 1: no switch, no intercept
            s['moves'][2, 0, 0] = [self.SLEEPTALK, self.TACKLE, 0, 0]               # lane 2: Sleep Talk
            s['forced_switch'][3, 0] = True                                        # lane 3: a replacement step
            s['confusion_counter'][4, 1, 0] = 2                                   # lane 4: a held clock
            s['moves'][5, 0, 0, 0] = self.ENCORE                                  # lane 5: Encore this step
            s['hp'][6, 0, 1:] = 0                                                 # lane 6: side 0's last mon, no weather
            s['hp'][7, 0, 1:] = 0;s['weather'][7] = 3                             # lane 7: its last mon under sandstorm
            s['hp'][8, 1, 1:] = 0;s['moves'][8, 0, 0, 0] = self.SUNNYDAY           # lane 8: a last mon, Sunny Day this step
            s['hp'][9, 1, 1:] = 0;s['ability'][9, 0, 2] = 2;s['forced_switch'][9, 0] = True   # lane 9: a Drought holder may enter
            s['perish_counter'][10, 1, 0] = 1                                     # lane 10: a Perish count of 1
            s['item'][11, 0, 0] = 2                                               # lane 11: an active Figy Berry (confusion)
        tb = PLUGINS['TB'];p = self.pre(edit);no, both = [False, False], [True, True]
        self.assertEqual(tb['pursuit_intercept'](p).tolist(), [[False, True]] + [no] * 11)
        self.assertEqual(tb['sleeptalk_call'](p).tolist(), [no, no, [True, False]] + [no] * 9)
        self.assertEqual(tb['turn_clock'](p).tolist(), [no, no, no, both, both, both, no, no, no, both, no, both])
        self.assertEqual(tb['weather_end'](p).tolist(), [no] * 7 + [both, both, both, no, no])
        self.assertEqual(tb['perish_upkeep'](p).tolist(), [no] * 10 + [both, no])

    def test_allowances(self):
        from gpu_engine.v2e_regress import ALLOWED, PLUGINS, lane_verdict
        self.assertTrue(all(e in ALLOWED for e in PLUGINS['TB']))
        ok = {('trackers.last_moves', 1), ("trackers.mon_features['stall_counter']", 1), ('trackers.damage_history.dealt', 1),
              ("trackers.mon_features['last_damage_dealt']", 1), ('trackers.hp_percent', 0), ('trackers.negative_items.percent', 0),
              ('trackers.public_port.known', 1), ('trackers.public_port.loaf', 1)}
        self.assertEqual(lane_verdict(ok, {'pursuit_intercept': [1]}, None)[0], [])
        for bad in (('state.hp', 0), ('trackers.last_moves', 0), ('trackers.observed_moves', 1), ('trackers.public_port.blocked', 1),
                    ('trackers.public_port.post_upkeep', None)):
            self.assertEqual(lane_verdict({bad}, {'pursuit_intercept': [1]}, None)[0], [bad])
        self.assertEqual(lane_verdict({('trackers.observed_moves', 0), ("trackers.mon_features['moves_shown']", 0),
                                       ('trackers.belief_mask', 0)}, {'sleeptalk_call': [0]}, None)[0], [])
        self.assertEqual(lane_verdict({('trackers.observed_moves', 1)}, {'sleeptalk_call': [0]}, None)[0], [('trackers.observed_moves', 1)])
        clocks = {("trackers.mon_features['confusion_turns']", 0), ("trackers.mon_features['wrap_trap_turns']", 1)}
        self.assertEqual(lane_verdict(clocks, {'turn_clock': [0, 1]}, None)[0], [])
        self.assertEqual(lane_verdict({('trackers.turn', None)}, {'turn_clock': [0, 1]}, None)[0], [('trackers.turn', None)])
        upkeep = {('trackers.public_port.post_upkeep', None), ("trackers.mon_features['sleep_cant_pending']", 1),
                  ('trackers.visible_belief.visible_hard', 0), ('trackers.negative_items.known', 0), ('trackers.belief_mask', 1)}
        self.assertEqual(lane_verdict(upkeep, {'perish_upkeep': [0, 1]}, None)[0], [])
        for bad in (("trackers.field_features['weather_upkeeps']", None), ("trackers.mon_features['encore_turns']", 0), ('state.hp', 0)):
            self.assertEqual(lane_verdict({bad}, {'perish_upkeep': [0, 1]}, None)[0], [bad])
        self.assertEqual(lane_verdict({("trackers.field_features['weather_upkeeps']", None)}, {'weather_end': [0, 1]}, None)[0], [])
        for bad in (('trackers.public_port.post_upkeep', None), ("trackers.mon_features['encore_turns']", 1), ('trackers.belief_mask', 0)):
            self.assertEqual(lane_verdict({bad}, {'weather_end': [0, 1]}, None)[0], [bad])


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class PerishUpkeepLineTests(unittest.TestCase):
    """transition._perish_upkeep_line, the pure F1 rule, case by case: each later handler that reaches faintMessages after a
    battle-ending Perish end path suppresses the |upkeep line (ended False) and the living mark; without one, both stand."""
    def run_line(self, edit, *, count=(1, 1), alive=(True, True), first=0, hp_after=(0, 0)):
        import jax.numpy as j
        from types import SimpleNamespace as NS
        from gpu_engine import transition
        from gpu_engine.state import empty_state
        s = empty_state(1)
        s = s._replace(hp=j.zeros((1, 2, 6), j.int32).at[0, :, 0].set(j.asarray(hp_after, j.int32)), max_hp=j.full((1, 2, 6), 100, j.int32))
        s = edit(s)
        truant = j.zeros(4, bool).at[3].set(True);herb = j.zeros(4, bool).at[2].set(True)
        table = NS(entry=NS(truant=truant), items=NS(white_herb=herb))
        ended, living = transition._perish_upkeep_line(s, table, j.asarray([True]), j.asarray([first], j.int32),
            j.asarray([alive]), j.asarray([count], j.int32))
        return bool(ended[0]), np.asarray(living)[0].tolist()

    def test_a_plain_perish_end_prints_upkeep(self):
        self.assertEqual(self.run_line(lambda s: s), (True, [True, True]))

    def test_each_later_handler_suppresses_it(self):
        cases = {
            'Truant (order 27)': lambda s: s._replace(ability=s.ability.at[0, 1, 0].set(3)),
            'White Herb (order 29)': lambda s: s._replace(item=s.item.at[0, 0, 0].set(2)),
            'a stall volatile set this turn': lambda s: s._replace(protect_counter=s.protect_counter.at[0, 0, 0].set(2)),
            'mustrecharge at duration 2': lambda s: s._replace(recharge_counter=s.recharge_counter.at[0, 1, 0].set(2)),
            "Solar Beam's charge turn (twoturnmove at 2)": lambda s: s._replace(charge_counter=s.charge_counter.at[0, 0, 0].set(2)),
        }
        for name, edit in cases.items():
            with self.subTest(name):self.assertEqual(self.run_line(edit), (False, [False, False]))

    def test_spent_durations_are_end_paths_not_handlers(self):
        for name, edit in {'mustrecharge at 1': lambda s: s._replace(recharge_counter=s.recharge_counter.at[0, 1, 0].set(1)),
                           'Solar Beam released (charge 1)': lambda s: s._replace(charge_counter=s.charge_counter.at[0, 0, 0].set(1))}.items():
            with self.subTest(name):self.assertEqual(self.run_line(edit), (True, [True, True]))

    def test_the_other_holders_tick_counts_only_after_the_first_end_path(self):
        # side 0 ends (count 1), side 1 ticks (count 3, a callback): side 0 first -> the tick's faintMessages ends it, no |upkeep
        self.assertEqual(self.run_line(lambda s: s, count=(1, 3), first=0, hp_after=(0, 50)), (False, [False, False]))
        # side 1 first: its tick runs before side 0's end path, nothing after it -> |upkeep, then the faint
        self.assertEqual(self.run_line(lambda s: s, count=(1, 3), first=1, hp_after=(0, 50)), (True, [True, False]))

    def test_a_holder_fainted_before_the_pass_runs_no_handler(self):
        # side 1's Truant holder fainted earlier in the residual (not alive at the pass): its handler is skipped
        self.assertEqual(self.run_line(lambda s: s._replace(ability=s.ability.at[0, 1, 0].set(3)), alive=(True, False), count=(1, 0)),
                         (True, [True, False]))


if __name__ == '__main__':
    unittest.main()
