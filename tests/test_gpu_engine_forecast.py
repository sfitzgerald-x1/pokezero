"""Forecast (gpu_engine/forecast.py): its forme table, one WeatherChange event's update and the event masks (eager JAX on
tiny fixtures), and transition.py's call-site wiring (pure AST, no JAX import). Mechanics follow Showdown f76228a1:
data/abilities.ts forecast / cloudnine / airlock, data/mods/gen4/abilities.ts (no suppressor arrival event),
sim/field.ts setWeather / clearWeather / suppressingWeather. The effect probes (gpu_engine/effect_probes/forecast_*.json)
cover the whole step against Showdown."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TYPES = ['_none', 'normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost', 'steel', 'fire', 'water',
         'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark']
ABILITIES = ['_none', 'forecast', 'cloudnine', 'airlock', 'drizzle', 'trace', 'static']
SPECIES = ['_none', 'castform', 'castformrainy', 'ditto', 'golduck', 'rayquaza']
META = {'vocabulary': {'abilities': ABILITIES, 'types': TYPES, 'species': SPECIES,
                       'moves': ['_none', 'splash', 'raindance', 'sunnyday', 'sandstorm', 'hail', 'thunder', 'struggle'],
                       'items': ['_none']}}
NONE, RAIN, SUN, SAND, HAIL = range(5)
NORMAL, WATER, FIRE, ICE = (TYPES.index(n) for n in ('normal', 'water', 'fire', 'ice'))


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class ForecastUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax.numpy as j
        from gpu_engine import forecast, weather
        from gpu_engine.state import empty_state
        cls.j, cls.f, cls.empty = j, forecast, staticmethod(empty_state)
        cls.table = forecast.from_metadata(META)
        cls.wt = weather.from_metadata(META)

    def state(self, rows):
        """rows: per lane ((species, ability, hp), (species, ability, hp), weather); active slot 0 on both sides."""
        j = self.j;b = len(rows);s = self.empty(b)
        species = s.species;ability = s.ability;hp = s.hp;types = s.types
        for lane, (left, right, _) in enumerate(rows):
            for side, (name, held, live) in enumerate((left, right)):
                species = species.at[lane, side, 0].set(SPECIES.index(name))
                ability = ability.at[lane, side, 0].set(ABILITIES.index(held))
                hp = hp.at[lane, side, 0].set(live)
                types = types.at[lane, side, 0, 0].set(WATER if name in ('golduck',) else NORMAL)
        return s._replace(species=species, ability=ability, base_ability=ability, hp=hp, max_hp=j.full_like(hp, 100),
                          types=types, base_types=types, active=j.zeros_like(s.active),
                          weather=j.asarray([w for *_, w in rows], j.int32))

    def types(self, state):
        return [[list(map(int, state.types[lane, side, 0])) for side in range(2)] for lane in range(state.hp.shape[0])]

    def test_table_formes(self):
        t = self.table
        self.assertEqual(int(t.castform), SPECIES.index('castform'))
        self.assertEqual(t.forme.tolist(), [[0, 0], [WATER, 0], [FIRE, 0], [0, 0], [ICE, 0]])
        self.assertEqual(t.retyped.tolist(), [False, True, True, False, True])
        self.assertEqual(t.forecast.tolist(), [name == 'forecast' for name in ABILITIES])
        missing = self.f.from_metadata({'vocabulary': dict(META['vocabulary'], species=['_none', 'ditto'])})
        self.assertEqual(int(missing.castform), -1)

    def test_each_weather_takes_its_forme_and_reverts(self):
        cast = ('castform', 'forecast', 50);other = ('ditto', 'static', 50)
        s = self.state([(cast, other, w) for w in (NONE, RAIN, SUN, SAND, HAIL)])
        out, changed = self.f.update(s, self.table, self.wt, eligible=self.j.ones(5, bool))
        self.assertEqual([lane[0] for lane in self.types(out)], [[NORMAL, 0], [WATER, 0], [FIRE, 0], [NORMAL, 0], [ICE, 0]])
        self.assertEqual(changed.tolist(), [[False, False], [True, False], [True, False], [False, False], [True, False]])
        # The forme back to Castform: its base types (setSpecies(baseSpecies)), with a change only where it was re-formed.
        back, changed = self.f.update(out._replace(weather=self.j.zeros(5, self.j.int32)), self.table, self.wt,
                                      eligible=self.j.ones(5, bool))
        self.assertEqual([lane[0] for lane in self.types(back)], [[NORMAL, 0]] * 5)
        self.assertEqual(changed[:, 0].tolist(), [False, True, True, False, True])

    def test_public_history_uses_forme_events_and_clears_on_faint_or_switch(self):
        from unittest.mock import patch
        import jax
        import numpy as np
        from gpu_engine import engine_device_public_hooks as hooks
        from test_gpu_engine_device_public_hooks import tables

        j = self.j
        cast = ('castform', 'forecast', 50)
        native = self.state([(cast, cast, RAIN)] * 4)
        eligible = j.array([[True, False], [False, True], [True, True], [False, False]])
        changed_forme, mask = self.f.update(native, self.table, self.wt, eligible=eligible)
        self.assertEqual(mask.tolist(), eligible.tolist())
        live = hooks.initialize(native, tables())
        legacy = hooks.initialize(native, tables())

        def assert_same_old_leaves():
            live_leaves, live_schema = jax.tree.flatten(live.trackers._replace(type_changed=None))
            legacy_leaves, legacy_schema = jax.tree.flatten(legacy.trackers._replace(type_changed=None))
            self.assertEqual(live_schema, legacy_schema)
            for actual, expected in zip(live_leaves, legacy_leaves):
                actual, expected = np.asarray(actual), np.asarray(expected)
                self.assertEqual((actual.shape, actual.dtype), (expected.shape, expected.dtype))
                self.assertEqual(actual.tobytes(), expected.tobytes())

        def apply_forme_event(state, mask):
            hooks.forecast(live, state, mask)
            with patch.object(hooks, 'type_event', side_effect=lambda context, *_: context):
                hooks.forecast(legacy, state, mask)
            assert_same_old_leaves()

        apply_forme_event(changed_forme, mask)
        self.assertEqual(live.trackers.type_changed.tolist(), eligible.tolist())

        restored, mask = self.f.update(changed_forme._replace(weather=j.zeros_like(native.weather)),
                                      self.table, self.wt, eligible=j.ones(4, bool))
        self.assertEqual(mask.tolist(), eligible.tolist())
        np.testing.assert_array_equal(restored.types, native.types)
        apply_forme_event(restored, mask)
        self.assertEqual(live.trackers.type_changed.tolist(), eligible.tolist())
        restoring = hooks.initialize(changed_forme, tables())
        hooks.forecast(restoring, restored, mask)
        self.assertEqual(restoring.trackers.type_changed.tolist(), eligible.tolist())

        fainted = restored._replace(hp=restored.hp.at[0, 0, 0].set(0).at[2, 1, 0].set(0))
        for context in (live, legacy):
            hooks.faint_clear(context, restored, fainted, j.array([True, True, False, True]))
        self.assertEqual(live.trackers.type_changed.tolist(),
                         [[False, False], [False, True], [True, True], [False, False]])
        assert_same_old_leaves()

        switched = fainted._replace(active=fainted.active.at[1, 1].set(1))
        for context in (live, legacy):
            hooks.install(context, switched, 1, j.array([False, True, False, False]))
        self.assertEqual(live.trackers.type_changed.tolist(),
                         [[False, False], [False, False], [True, True], [False, False]])
        assert_same_old_leaves()

    def test_suppression_ending_and_weather_override(self):
        cast = ('castform', 'forecast', 50)
        rows = [(cast, ('golduck', 'cloudnine', 50), RAIN), (cast, ('rayquaza', 'airlock', 50), SUN),
                (cast, ('golduck', 'cloudnine', 0), RAIN), (('golduck', 'cloudnine', 50), cast, RAIN)]
        s = self.state(rows)
        out, changed = self.f.update(s, self.table, self.wt, eligible=self.j.ones(4, bool))
        # A living suppressor on either side: no effective weather (Normal); a fainted one does not suppress.
        self.assertEqual(self.types(out), [[[NORMAL, 0], [WATER, 0]], [[NORMAL, 0], [NORMAL, 0]],
                                           [[WATER, 0], [WATER, 0]], [[WATER, 0], [NORMAL, 0]]])
        self.assertEqual(changed.tolist(), [[False, False], [False, False], [True, False], [False, False]])
        # An End in progress (`ending`) stops that holder suppressing.
        ending = self.j.asarray([[False, True], [False, True], [False, False], [True, False]])
        out, changed = self.f.update(s, self.table, self.wt, eligible=self.j.ones(4, bool), ending=ending)
        self.assertEqual([lane[0] for lane in self.types(out)][:3], [[WATER, 0], [FIRE, 0], [WATER, 0]])
        self.assertEqual(self.types(out)[3][1], [WATER, 0])
        # The event's own weather (a drag's pre-entry weather) overrides the state's.
        out, _ = self.f.update(s, self.table, self.wt, eligible=self.j.ones(4, bool), ending=ending,
                               weather=self.j.full(4, SUN, self.j.int32))
        self.assertEqual(self.types(out)[0][0], [FIRE, 0])

    def test_inert_holders_and_masks(self):
        rows = [(('ditto', 'forecast', 50), ('castform', 'trace', 50), RAIN),   # a copy of Forecast; Castform without it
                (('castform', 'forecast', 0), ('ditto', 'static', 50), RAIN),   # a fainted Castform
                (('castform', 'forecast', 50), ('castform', 'forecast', 50), RAIN)]
        s = self.state(rows)
        out, changed = self.f.update(s, self.table, self.wt, eligible=self.j.ones(3, bool))
        self.assertEqual(changed.tolist(), [[False, False], [False, False], [True, True]])
        self.assertEqual(self.types(out)[0], self.types(s)[0])
        self.assertEqual(self.types(out)[1], self.types(s)[1])
        # [B] eligibility covers both actives, [B,2] one side only (a Forecast Start re-forms only its holder).
        _, changed = self.f.update(s, self.table, self.wt, eligible=self.j.asarray([[True, True], [True, True], [False, True]]))
        self.assertEqual(changed.tolist(), [[False, False], [False, False], [False, True]])
        _, changed = self.f.update(s, self.table, self.wt, eligible=self.j.asarray([True, True, False]))
        self.assertEqual(changed.tolist(), [[False, False]] * 3)

    def test_transformed_holder_is_inert_when_the_state_carries_the_flag(self):
        s = self.state([(('castform', 'forecast', 50), ('ditto', 'static', 50), RAIN)] * 2)
        flagged = s._replace(transformed=s.transformed.at[0, 0].set(True))
        out, changed = self.f.update(flagged, self.table, self.wt, eligible=self.j.ones(2, bool))
        self.assertEqual(changed.tolist(), [[False, False], [True, False]])

    def test_event_masks(self):
        j = self.j;cast = ('castform', 'forecast', 50)
        before = self.state([(cast, ('golduck', 'cloudnine', 50), RAIN), (cast, ('golduck', 'cloudnine', 0), RAIN),
                             (cast, ('rayquaza', 'airlock', 50), SUN)])
        after = before._replace(hp=before.hp.at[0, 1, 0].set(0))
        self.assertEqual(self.f.suppressor_fainted(after, before.hp, self.wt).tolist(), [True, False, False])
        self.assertEqual(self.f.suppressor_departs(before, j.asarray([1, 1, 0]), self.wt).tolist(), [True, False, False])
        self.assertEqual(self.f.suppressor_departs(before, 1, self.wt).tolist(), [True, False, True])
        # An entry: its own side's Start; both sides for a weather or clock change, or a suppressor's hazard faint.
        entered = after._replace(weather=after.weather.at[2].set(RAIN),
                                 weather_counter=after.weather_counter.at[1].set(5))
        self.assertEqual(self.f.entry_events(before, entered, j.asarray([0, 0, 1]), self.wt).tolist(),
                         [[True, True], [True, True], [True, True]])
        self.assertEqual(self.f.entry_events(before, before, j.asarray([0, 1, 1]), self.wt).tolist(),
                         [[True, False], [False, True], [False, True]])

    def test_faint_queue_reforms_only_a_recipient_not_yet_fainted(self):
        j=self.j;cast=('castform','forecast',50);duck=('golduck','cloudnine',50)
        before=self.state([(cast,duck,RAIN),(cast,duck,RAIN),(duck,cast,SUN),(duck,cast,SUN),
                           (('castform','forecast',0),duck,RAIN),(cast,duck,RAIN)])
        # Both HP writes can precede faintMessages. Queue order determines whether the zero-HP Castform is still
        # active when the suppressor's End runs: target before recoil/Bond, actor first for Explosion, Perish order.
        after=before._replace(hp=before.hp.at[:5,:,0].set(0).at[5,1,0].set(0))
        present=self.f.faint_recipients(after,before.hp,j.asarray([1,0,0,1,1,1]))
        event=self.f.suppressor_fainted(after,before.hp,self.wt)
        out,changed=self.f.update(after,self.table,self.wt,eligible=event,not_fainted=present)
        self.assertEqual(changed.tolist(),[[True,False],[False,False],[False,True],[False,False],
                                          [False,False],[True,False]])
        self.assertEqual(self.types(out)[0][0],[WATER,0])
        self.assertEqual(self.types(out)[2][1],[FIRE,0])
        self.assertEqual(out.hp.tolist(),after.hp.tolist())

    def test_separate_residual_faints_preserve_the_earlier_forme(self):
        j=self.j
        before=self.state([(('castform','forecast',11),('golduck','cloudnine',36),RAIN)])
        first=before._replace(hp=before.hp.at[0,1,0].set(0))
        first,changed=self.f.update(first,self.table,self.wt,
            eligible=self.f.suppressor_fainted(first,before.hp,self.wt))
        self.assertEqual(changed.tolist(),[[True,False]])
        later=first._replace(hp=first.hp.at[0,0,0].set(0))
        later,changed=self.f.update(later,self.table,self.wt,
            eligible=self.f.suppressor_fainted(later,first.hp,self.wt))
        self.assertEqual(changed.tolist(),[[False,False]])
        self.assertEqual(self.types(later)[0][0],[WATER,0])


def _step_source():
    tree = ast.parse((ROOT / 'gpu_engine' / 'transition.py').read_text())
    return tree, {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _calls(node, name):
    return sorted((n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name),
                  key=lambda n: (n.lineno, n.col_offset))


class ForecastWiringTests(unittest.TestCase):
    """transition.py calls _forecast once per Showdown WeatherChange point (pure AST): a dropped, moved or doubled call fails
    here as well as in the probes."""
    @classmethod
    def setUpClass(cls):
        cls.tree, cls.fns = _step_source()
        cls.step = cls.fns['step_with_chance']

    def kinds(self, node):
        out = []
        for call in _calls(node, '_forecast'):
            keys = {k.arg for k in call.keywords} & {'entered', 'departed', 'weather_since', 'fainted_since'}
            self.assertLessEqual(len(keys), 2, ast.unparse(call))
            out.append('+'.join(sorted(keys)) or 'every')
        return out

    def test_the_weather_change_points(self):
        self.assertEqual(self.kinds(self.fns['initialize']), ['entered'])
        self.assertEqual(self.kinds(self.step), ['fainted_since', 'departed', 'entered', 'entered', 'weather_since',
                                                 'departed', 'entered', 'fainted_since', 'weather_since', 'fainted_since',
                                                 'fainted_since'])
        # Each call threads the lifecycle tape and the device context, and reassigns the state.
        for call in _calls(self.step, '_forecast'):
            keys = {k.arg: ast.unparse(k.value) for k in call.keywords}
            self.assertEqual((keys['_public_trace'], keys['_lifecycle'], keys['_device_public']),
                             ('_public_trace', '_lifecycle_trace', '_device_public'))
        assigns = [n for n in ast.walk(self.step) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Name) and n.value.func.id == '_forecast']
        self.assertEqual({ast.unparse(n.targets[0]) for n in assigns}, {'(work, _public_trace)'})
        self.assertEqual(len(assigns), 11)

    def test_order_against_the_events_they_follow(self):
        lines = [c.lineno for c in _calls(self.step, '_forecast')]
        def first(pattern):
            found = [n.lineno for n in ast.walk(self.step) if isinstance(n, ast.stmt) and not hasattr(n, 'body')
                     and pattern in ast.unparse(n)]
            self.assertTrue(found, pattern)
            return found
        depart = first('arrival.depart(work, side, table.entry, eligible=do_switch)')[0]
        install = first('work = work._replace(active=work.active.at[rows, side].set(jnp.where(do_switch, target[rows, side], old_active))')[0]
        self.assertTrue(depart < lines[1] < install)
        set_move = first('climate.set_move(')[-1]
        self.assertTrue(set_move < lines[4])
        drag = first('dragging.drag(')[-1]
        self.assertTrue(drag < lines[5] < lines[6])
        turn_context = first('turn_context = live & (~replacement | state.pending_residual)')[0]
        actions = [n for n in self.step.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'range(2)'
                   and any(isinstance(s, ast.Assign) and ast.unparse(s) == '_faint_before_hp = work.hp' for s in n.body[:1])]
        self.assertEqual(len(actions), 1)
        self.assertTrue(actions[0].lineno < lines[7] < actions[0].end_lineno < turn_context)
        action_call=_calls(actions[0],'_forecast')[-1]
        self.assertEqual({k.arg:ast.unparse(k.value) for k in action_call.keywords}['queue_first'],
                         'jnp.where(explodes, side, foe)')
        clock = first('climate.residual_clock(')[-1]
        self.assertTrue(clock < lines[8])
        residual = [n for n in ast.walk(self.step) if isinstance(n, ast.For) and ast.unparse(n.iter) == 'range(residual_order.EVENTS)']
        self.assertEqual(len(residual), 1)
        self.assertEqual(len(_calls(residual[0], '_forecast')),1)
        self.assertTrue(residual[0].lineno < lines[9] < residual[0].end_lineno)
        herb = first('_white_herb_pair(work, table, 0, _herb_residual')[0]
        self.assertTrue(residual[0].end_lineno < herb < lines[10])
        perish_call=_calls(self.step,'_forecast')[-1]
        self.assertEqual({k.arg:ast.unparse(k.value) for k in perish_call.keywords}['queue_first'],'_perish_first')

    def test_the_device_line_is_the_forecast_hook(self):
        helper = self.fns['_forecast']
        hooks = [ast.unparse(n.func) for n in ast.walk(helper) if isinstance(n, ast.Call) and ast.unparse(n.func).startswith('device_public.')]
        self.assertEqual(hooks, ['device_public.forecast'])
        hooks_tree = ast.parse((ROOT / 'gpu_engine' / 'engine_device_public_hooks.py').read_text())
        forecast = next(n for n in hooks_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'forecast')
        calls = {ast.unparse(n.func) for n in ast.walk(forecast) if isinstance(n, ast.Call)}
        self.assertIn('reveal_ability', calls)
        self.assertTrue(any("'shown_types'" in ast.unparse(n) for n in ast.walk(forecast) if isinstance(n, ast.Call)))


if __name__ == '__main__':
    unittest.main()
