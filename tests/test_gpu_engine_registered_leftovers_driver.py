"""Actual registered Leftovers seam, tiny source fixtures; no full qualification.

The full consumer is integrated separately. Its forwarding test below is a
named spy; direct frame tests inspect actual numerical callbacks and carry.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_leftovers_callbacks as leftovers
from gpu_engine import engine_lifecycle_callbacks as life
from gpu_engine import engine_retained_encoder_driver as driver
from gpu_engine import engine_weather_speed_profile as speed_profile
from gpu_engine import source_provenance, transition, items
import test_gpu_engine_leftovers_owning_callers as fixture


class RegisteredLeftoversDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.LeftoversOwningCallerTests.setUpClass()
        cls.f = fixture.LeftoversOwningCallerTests
        cls.prepared = driver.prepare(cls.f.request, cls.f.own,
            public_mode='retained-full', leftovers_compiled=cls.f.leftovers,
            leftovers_mode='retained',
            fake_forward=lambda _: driver.FakeScores(j.zeros(9), j.array(1, j.int32)))

    same = fixture.LeftoversOwningCallerTests.same
    words = staticmethod(fixture.LeftoversOwningCallerTests.words)
    chance = fixture.LeftoversOwningCallerTests.chance

    def construct(self, b=2, *, tie=False, move='hail', stored_speeds=None):
        if not tie and stored_speeds is None:
            session = self.f().construct(b, move)
        else:
            arrays = self.f.f().arrays('pressure', 'pressure')
            arrays['moves'][0, 0, :2] = [self.f.f.ids[move], 1]
            arrays['stats'][..., 5] = (199 if stored_speeds is None
                else np.array(stored_speeds)[:, None])
            with TemporaryDirectory() as directory:
                path = Path(directory) / 'tiny.npz'
                np.savez_compressed(path, **dict(arrays, metadata_utf8=np.frombuffer(
                    json.dumps(self.f.f.metadata).encode(), np.uint8)))
                pool = source_provenance.load_constructor_pool(path)
                session = driver.construct(pool, np.tile([[0, 1]], (b, 1)).astype(np.int64),
                    prepared=self.f.prepared, stream_ids=j.arange(901, 901+b, dtype=j.int32), seed=31)
        # Retain the ACTUAL original pre-entry snapshot, never reconstruct a
        # birth from current state. This is a component frame test, not a new
        # Session or a public consumer admission claim.
        r = driver._checked(self.prepared)
        birth = driver._SESSIONS[session]['constructed'].birth_state
        core = session.state.core
        profile = speed_profile.qualify_birth(birth, core.cursor.stream_id, profile=r['speed_profile'])
        profile = speed_profile.retain(profile, core.engine, core.cursor.stream_id, profile=r['speed_profile'])
        return session, core._replace(weather_profile=profile)

    def frame(self, core, *, mask=None):
        b = core.engine.hp.shape[0]
        return driver._checked(self.prepared)['frame'](core,
            j.tile(j.array([[0, 8]], j.int32), (b, 1)), self.chance(b), mask)

    def test_actual_strict_cached_speed_defers_only_new_helper_latch(self):
        session, core = self.construct()
        actual = self.frame(core)
        diagnostic = driver._checked(self.f.prepared)['frame'](core,
            j.tile(j.array([[0, 8]], j.int32), (2, 1)), self.chance())
        self.same(actual.state, diagnostic.state)
        self.assertTrue(bool(actual.weather_profile.qualified_birth.all()))
        for row in range(2):
            words = self.words(actual, row)
            heals, chips = words[words[:, 0] == 17], words[words[:, 0] == 16]
            self.assertEqual(heals[:, 3].tolist(), [1, 0])  # source cached speeds301>199
            self.assertTrue(any(int(w[4]) > 0 for w in heals))
            self.assertEqual(len(heals), 2)
            for heal in heals:
                side, physical = map(int, heal[3:5])
                self.assertEqual(physical, int(actual.state.active[row, side]))
                chip = chips[chips[:, 3] == side][0]
                self.assertEqual(int(heal[5]), int(chip[6]))
                self.assertEqual(int(heal[6]), int(actual.state.hp[row, side, physical]))
                self.assertEqual(int(heal[9]), self.f.item)
            self.assertFalse(int(actual.callbacks.refusal[row]) & 256)
            self.assertTrue(int(diagnostic.callbacks.refusal[row]) & 256)
        self.assertFalse(any(bool(o.B4_qualified.any()) for o in session.result.owners))

    def test_tie_and_sticky_false_keep_unknown_with_actual_callbacks(self):
        _, tied = self.construct(tie=True)
        tied_frame = self.frame(tied)
        self.assertTrue(bool(tied_frame.weather_profile.qualified_birth.all()))
        self.assertTrue(bool(j.all((tied_frame.callbacks.refusal & 256) != 0)))
        _, core = self.construct()
        core = core._replace(weather_profile=core.weather_profile._replace(qualified_birth=j.array([False, True])))
        frame = self.frame(core)
        self.assertEqual(frame.weather_profile.qualified_birth.tolist(), [False, True])
        self.assertEqual(((frame.callbacks.refusal & 256) != 0).tolist(), [True, False])
        self.assertTrue(all((self.words(frame, row)[:, 0] == 17).any() for row in range(2)))

    def test_actual_constructor_effective_gen3_high_speed_and_safe_edge(self):
        # Qualify the original actual birth BEFORE adding the permitted stage.
        # The BASE prototype maps raw10000 to1808; the effective Gen3 override
        # returns10000. Cached source order is[0,1] in BOTH high-speed cases.
        for stored, cached in ((2500, 10000), (2047, 8188)):
            with self.subTest(stored=stored):
                _, core = self.construct(stored_speeds=(stored, 2000))
                core = core._replace(engine=core.engine._replace(
                    boosts=core.engine.boosts.at[:, 0, :, 4].set(6)))
                self.assertEqual(transition._speeds(core.engine, self.f.f.table).tolist(),
                    [[cached, 2000], [cached, 2000]])
                frame = self.frame(core, mask=j.array([True, False]))
                self.assertEqual(frame.weather_profile.qualified_birth.tolist(), [True, True])
                words = self.words(frame, 0)
                self.assertEqual(words[words[:, 0] == 17, 3].tolist(), [0, 1])
                self.assertFalse(bool(int(frame.callbacks.refusal[0]) & 256))
                self.assertEqual(int(frame.callbacks.count[1]), 0)
                self.same(jax.tree.map(lambda x: x[1], frame.state),
                    jax.tree.map(lambda x: x[1], core.engine))
                self.same(jax.tree.map(lambda x: x[1], frame.weather_profile),
                    jax.tree.map(lambda x: x[1], core.weather_profile))

    def test_original_effective_gen3_speed_override_and_cached_update(self):
        # Original method fixtures only: no Battle/Pokemon construction/game.
        script = r'''
const {Dex} = require('./dist/sim/dex.js');
const {Pokemon} = require('./dist/sim/pokemon.js');
const dex = Dex.mod('gen3'); dex.includeData();
const format = Dex.formats.get('gen3randombattle');
const battle = {ruleTable:{has:()=>false},field:{getPseudoWeather:()=>false},
  trunc:Dex.trunc,quickClawRoll:false};
const inputs = [10000,2000];
const base = inputs.map(speed => Pokemon.prototype.getActionSpeed.call({getStat:()=>speed,battle}));
const cached = inputs.map(speed => {
  const holder = Object.create(Pokemon.prototype);
  Object.assign(holder,dex.data.Scripts.pokemon);
  holder.getStat=()=>speed; holder.battle=battle; holder.hasItem=()=>false;
  holder.updateSpeed(); return holder.speed;
});
console.log(JSON.stringify({base,cached,mod:format.mod,formatOverride:!!format.pokemon}));
'''
        result = json.loads(subprocess.check_output([self.f.f.f.f.node, '-e', script],
            cwd=self.f.f.f.f.root, text=True))
        self.assertEqual(result, dict(base=[1808, 2000], cached=[10000, 2000],
            mod='gen3', formatOverride=False))

    def test_cached_speed_above_source_cap_remains_unknown(self):
        _, core = self.construct(move='raindance')
        core = core._replace(engine=core.engine._replace(hp=core.engine.hp-20))
        context = driver._checked(self.prepared)['leftovers_context']
        # A faulted actual cached-speed helper output exceeds source getStat's
        # cap. Rain has no HP segment to supply a separate weather refusal.
        with patch.object(transition, '_speeds', return_value=j.array([[10001, 2000]]*2, j.int32)):
            actual = transition.step_with_chance(core.engine,
                j.tile(j.array([[0, 8]], j.int32), (2, 1)), self.f.f.table, self.chance(),
                _public_trace=life._tape(core.cursor, self.f.f.producer, 64), _lifecycle_trace=True,
                _weather_hp_context=(core.weather_profile, context.weather_profile_data),
                _leftovers_context=context)
        self.assertEqual(actual.weather_profile.qualified_birth.tolist(), [True, True])
        for row in range(2):
            words = self.words(actual, row)
            self.assertFalse(bool((words[:, 0] == 16).any()))
            self.assertEqual(int((words[:, 0] == 17).sum()), 2)
            self.assertTrue(int(actual.callbacks.refusal[row]) & 256)

    def test_b8_inactive_false_profile_and_nonphysical_literal_targets(self):
        _, core = self.construct(8)
        mask = j.array([True, False, True, False, True, False, True, False])
        core = core._replace(weather_profile=core.weather_profile._replace(
            qualified_birth=j.array([True, True, False, True, True, True, True, True])))
        frame = self.frame(core, mask=mask)
        self.assertEqual(frame.weather_profile.qualified_birth.tolist(), core.weather_profile.qualified_birth.tolist())
        for row in range(8):
            if not bool(mask[row]):
                self.assertEqual(int(frame.callbacks.count[row]), 0)
                self.same(frame.state.hp[row], core.engine.hp[row])
                self.same(jax.tree.map(lambda x: x[row], frame.weather_profile),
                    jax.tree.map(lambda x: x[row], core.weather_profile))
            else:
                heals = self.words(frame, row)
                self.assertEqual(int((heals[:, 0] == 17).sum()), 2)
                self.assertEqual(bool(int(frame.callbacks.refusal[row]) & 256), row == 2)
        for name in ('engine_binding', 'source_binding', 'move_binding', 'producer_binding'):
            self.assertEqual(getattr(frame.callbacks, name).shape, (8,))
            self.same(getattr(frame.callbacks, name), getattr(self.f.leftovers.device, name))

    def test_item_write_intermediate_false_survives_later_safe_final(self):
        _, core = self.construct()
        r = driver._checked(self.prepared)
        context = r['leftovers_context']
        calls = [0]
        original = items.consume_pinch
        def transient(table, item, boosts, hp, maximum, *, eligible):
            calls[0] += 1
            current, after = original(table, item, boosts, hp, maximum, eligible=eligible)
            # Fault two literal numerical item-write results. The first is an
            # unsupported item, the second consumes it back to0. No source
            # history assertion can be recovered from the safe final item.
            if calls[0] == 2:
                current = j.where(eligible,
                    len(self.f.metadata['vocabulary']['items']) + 9, current)
            elif calls[0] == 3:
                current = j.zeros_like(current)
            return current, after
        tape = life._tape(core.cursor, self.f.f.producer, 64)
        with patch.object(items, 'consume_pinch', side_effect=transient):
            result = transition.step_with_chance(core.engine, j.tile(j.array([[0, 8]], j.int32), (2, 1)),
                self.f.f.table, self.chance(), _public_trace=tape, _lifecycle_trace=True,
                _weather_hp_context=(core.weather_profile, context.weather_profile_data),
                _leftovers_context=context)
        self.assertGreaterEqual(calls[0], 3)
        self.assertTrue(bool(j.all((result.step.state.item >= 0) &
            (result.step.state.item < len(self.f.metadata['vocabulary']['items'])))))
        self.assertEqual(result.weather_profile.qualified_birth.tolist(), [False, False])
        final = speed_profile.retain_device(result.weather_profile, result.step.state,
            core.cursor.stream_id, context.weather_profile_data)
        self.assertEqual(final.qualified_birth.tolist(), [False, False])
        self.assertTrue(bool(j.all((result.callbacks.refusal & 256) != 0)))

    def test_typed_context_ownership_modes_and_original_host_guards(self):
        r = driver._checked(self.prepared)
        self.assertIs(type(r['leftovers_context']), leftovers.RegisteredLeftoversContext)
        self.assertIs(r['leftovers_context'].weather_profile_data, r['speed_profile'].device)
        self.assertIs(type(driver._checked(self.f.prepared)['leftovers_context']), life.LeftoversContext)
        for kwargs in (dict(leftovers_mode='other'), dict(leftovers_mode='retained'),
                dict(leftovers_mode='retained', leftovers_compiled=self.f.leftovers)):
            with self.assertRaises(ValueError):
                driver.prepare(self.f.request, self.f.own, fake_forward=lambda _: None, **kwargs)
        with self.assertRaises(ValueError):
            driver.prepare(self.f.request, self.f.own, fake_forward=lambda _: None,
                public_mode='retained-full', leftovers_mode='retained', leftovers_compiled=replace(self.f.leftovers))
        _, core = self.construct()
        tape = life._tape(core.cursor, self.f.f.producer, 64)
        for context, weather_context in ((r['leftovers_context'], None),
                (r['leftovers_context'], (core.weather_profile, r['speed_profile'].device._replace())),
                (True, (core.weather_profile, r['speed_profile'].device))):
            with self.assertRaises(ValueError):
                transition.step_with_chance(core.engine, j.full((2, 2), 8, j.int32), self.f.f.table,
                    self.chance(), _public_trace=tape, _lifecycle_trace=True,
                    _weather_hp_context=weather_context, _leftovers_context=context)
        # Checked driver invocation still consults the original helper holder.
        self.f.metadata['vocabulary']['items'][self.f.item] = 'changed'
        try:
            with self.assertRaises(ValueError): driver._checked(self.prepared)
        finally:
            self.f.metadata['vocabulary']['items'][self.f.item] = 'leftovers'

    def test_private_helper_invalid_and_foreign_bits_never_defer(self):
        _, core = self.construct()
        data = self.f.leftovers.device
        tape = life._tape(core.cursor, self.f.f.producer, 64)._replace(refusal=j.full(2, 128, j.int32))
        # Valid private defer keeps existing foreign bits, never clears them.
        def heal(value, **kwargs):
            return leftovers.record_leftovers_heal_device(value, 0, core.engine.active[:, 0],
                self.f.item, 50, 56, 100, 0, data, eligible=j.array([True, False]),
                _defer_own_unknown=True, **kwargs)
        valid = heal(tape)
        self.assertEqual(valid.count.tolist(), [1, 0])
        self.assertEqual(valid.refusal.tolist(), [128, 128])
        invalid = leftovers.record_leftovers_heal_device(tape, 0, core.engine.active[:, 0],
            0, 50, 56, 100, 0, data, eligible=True, _defer_own_unknown=True)
        self.assertEqual(invalid.count.tolist(), [0, 0])
        self.assertEqual(invalid.refusal.tolist(), [384, 384])
        overflow = heal(tape._replace(count=j.full(2, 64, j.int32)))
        self.assertTrue(int(overflow.refusal[0]) & 256)
        self.assertTrue(int(overflow.refusal[0]) & 2)
        self.assertEqual(int(overflow.refusal[1]), 128)
        context = driver._checked(self.prepared)['leftovers_context']
        actual = transition.step_with_chance(core.engine,
            j.tile(j.array([[0, 8]], j.int32), (2, 1)), self.f.f.table, self.chance(),
            _public_trace=tape, _lifecycle_trace=True,
            _weather_hp_context=(core.weather_profile, context.weather_profile_data),
            _leftovers_context=context)
        self.assertTrue(bool(j.all((actual.callbacks.refusal & 128) != 0)))
        self.assertFalse(bool(j.any((actual.callbacks.refusal & 256) != 0)))
        self.assertTrue(all((self.words(actual, row)[:, 0] == 17).any() for row in range(2)))

    def test_full_consumer_data_forwarding_is_explicit_and_spy_only(self):
        session, core = self.construct()
        observer = session.state.observers[0]
        received = []
        def consumer_spy(*args, **kwargs):
            received.append(kwargs)
            return SimpleNamespace(projection=SimpleNamespace(error=j.zeros(2, j.int32),
                component_admitted=j.ones(2, j.bool_), state=observer.projection), public=observer.retained)
        profile = (core.weather_profile, driver._checked(self.prepared)['speed_profile'].device)
        driver._full_public(observer, None, None, None, consumer_spy, weather_hp=profile)
        driver._full_public(observer, None, None, None, consumer_spy,
            weather_hp=profile, leftovers=self.f.leftovers.device)
        self.assertNotIn('leftovers', received[0])
        self.assertIs(received[1]['leftovers'], self.f.leftovers.device)
        self.assertIs(received[1]['weather_hp'], profile)


if __name__ == '__main__': unittest.main()
