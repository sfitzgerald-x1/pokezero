"""The probe reference retains encounter pin activity independently of current hard support."""
import unittest
import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from gpu_engine.effect_probe_set_posterior import Narrowing


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.reference = Narrowing.__new__(Narrowing)
        self.reference.source = SimpleNamespace(universe_for=lambda _: SimpleNamespace(variants=(0, 1)))
        self.reference.dex = None
        self.reference.sp = SimpleNamespace(
            species_key=str.lower, _first_request=lambda _: None, SINGLE_VARIANT='single-variant',
            parse_strikes=lambda *_: [dict(attacker_key='tauros', reason=None, line=1, last=2)],
            _counts=Mock(return_value=(None, (1, 0))), _encounter=Mock())
        self.reference.reset()
        self.reference.feed(0, 'initial', ([], []))
        self.reference.feed(1, 'initial', ([], []))
        for q in (0, 2):self.reference.damage[q] = {2: dict(species='tauros', hard=(3, 0))}
        self.reference.damage[1] = {2: dict(species='tauros', hard=(1, 0))}

    def result(self, support, admitted):
        self.reference.sp._encounter.return_value = SimpleNamespace(
            support=support, strikes=[SimpleNamespace(admitted=admitted, line=1, move='return',
                                                     reason='' if admitted else 'single-variant')])

    def test_later_single_variant_reveal_preserves_activity(self):
        self.result((True, False), True)
        self.assertTrue(self.reference.reference(0, 0, 'Tauros', (3, 0))[1])
        self.reference.sp._counts.reset_mock()
        self.result((True, False), False)
        support, active = self.reference.reference(0, 0, 'Tauros', (1, 0))
        self.assertTrue(active)
        self.assertEqual(support, [True] + [False] * 63)
        self.reference.sp._counts.assert_not_called()

    def test_activity_does_not_reuse_prior_support(self):
        self.result((True, False), True)
        self.reference.reference(0, 0, 'Tauros', (3, 0))
        self.result((False, True), False)
        support, active = self.reference.reference(0, 0, 'Tauros', (2, 0))
        self.assertTrue(active)
        self.assertEqual(support, [False, True] + [False] * 62)

    def test_activity_is_isolated_and_reset_only_for_the_new_battle(self):
        self.result((True, False), True)
        self.reference.reference(0, 0, 'Tauros', (3, 0))
        self.reference.reference(1, 0, 'Tauros', (3, 0))
        self.result((True, False), False)
        self.assertFalse(self.reference.reference(0, 1, 'Tauros', (1, 0))[1])
        self.assertFalse(self.reference.reference(0, 0, 'Snorlax', (1, 0))[1])
        self.reference.feed(0, 'turn', ([], []))
        self.assertTrue(self.reference.reference(0, 0, 'Tauros', (1, 0))[1])
        self.reference.sp.parse_strikes = lambda *_: []
        self.reference.feed(0, 'initial', ([], []))
        self.assertFalse(self.reference.reference(0, 0, 'Tauros', (1, 0))[1])
        self.assertTrue(self.reference.reference(1, 0, 'Tauros', (1, 0))[1])

    def test_completed_ko_window_is_not_left_open_by_request_free_channels(self):
        if importlib.util.find_spec('pokezero.set_posterior') is None:
            self.skipTest('host set_posterior is not available in this tree')
        from pokezero import set_posterior
        lines = [
            '|switch|p1a: Snorlax|Snorlax, L100|200/400',
            '|switch|p2a: Tauros|Tauros, L100|100/100',
            '|move|p2a: Tauros|Return|p1a: Snorlax',
            '|-damage|p1a: Snorlax|0 fnt',
            '|faint|p1a: Snorlax',
        ]
        self.assertEqual(set_posterior.parse_strikes(lines, 'p1')[0]['reason'], 'window-open')
        self.reference.sp.parse_strikes = set_posterior.parse_strikes
        self.reference.feed(0, 'step', (lines, []))
        self.reference.damage[0] = {3: dict(species='tauros', hard=(3, 0))}
        self.result((True, False), True)
        self.assertTrue(self.reference.reference(0, 0, 'Tauros', (3, 0))[1])
        self.assertEqual(self.reference.sp._counts.call_count, 2)  # event-time activity and independent root-time support
        self.assertIsNone(self.reference.sp._encounter.call_args.args[4][0][1])
        self.assertEqual(self.reference.lines[0][0], lines)

    def test_same_step_singleton_still_activates_from_damage_time_hard(self):
        self.result((True, False), False)  # current hard is singleton: no strike admitted by the root-time reference
        support, active = self.reference.reference(0, 0, 'Tauros', (1, 0))
        self.assertTrue(active)
        self.assertEqual(support, [True] + [False] * 63)
        self.assertEqual(self.reference.sp._counts.call_args.args[3], (True, True))

    def test_root_admission_cannot_replace_event_time_singleton_or_refusal(self):
        self.result((True, False), True)
        self.reference.damage[0][2]['hard'] = (1, 0)
        self.assertFalse(self.reference.reference(0, 0, 'Tauros', (3, 0))[1])
        self.reference.evaluated[0].clear()
        self.reference.damage[0][2]['hard'] = (3, 0)
        self.reference.sp._counts.side_effect = [('effectiveness-disagreement', None), (None, (1, 0))]
        self.assertFalse(self.reference.reference(0, 0, 'Tauros', (3, 0))[1])

    def test_empty_fit_and_protocol_refusals_do_not_activate(self):
        self.result((True, True), False)
        self.reference.sp._counts.return_value = (None, (0, 0))
        self.assertFalse(self.reference.reference(0, 0, 'Tauros', (3, 0))[1])
        self.reference.evaluated[0].clear()
        self.reference.damage[0] = {}  # Substitute has no HP damage capture; multi-hit must not use one of its hits
        for reason in ('hit-substitute', 'multi-hit', 'called'):
            with self.subTest(reason=reason):
                self.reference.evaluated[0].clear()
                self.reference.sp.parse_strikes = lambda *_, reason=reason: [dict(
                    attacker_key='tauros', reason=reason, line=1, last=2)]
                self.assertFalse(self.reference.reference(0, 0, 'Tauros', (3, 0))[1])

    def test_missing_or_wrong_subject_capture_is_an_error(self):
        self.result((True, False), True)
        for captures in ({}, {2: dict(species='snorlax', hard=(3, 0))}):
            with self.subTest(captures=captures), self.assertRaisesRegex(ValueError, 'expected one B2 damage capture'):
                self.reference.damage[0] = captures
                self.reference.reference(0, 0, 'Tauros', (3, 0))


class CaptureTests(unittest.TestCase):
    def test_cuts_keep_original_offsets_and_copy_hard_before_later_reveals(self):
        reference = Narrowing.__new__(Narrowing)
        reference.sp = SimpleNamespace(species_key=str.lower)
        reference.reset()
        hard = np.zeros((2, 2, 6, 2), np.uint32);hard[..., 0] = 3
        host = SimpleNamespace(beliefs=SimpleNamespace(hard=hard), mons=SimpleNamespace(
            active=np.zeros((2, 2), np.int32), species=np.ones((2, 2, 6), np.int32),
            encounter=np.ones((2, 2, 6), np.int32)))
        b2 = SimpleNamespace(streams=2, lines=[10, 20], refused=[None, None], host=lambda: host,
            catalog=SimpleNamespace(species=SimpleNamespace(names=('_none', 'Tauros'))))
        batches = []
        def feed(parts):
            batches.append(parts)
            for q, lines in enumerate(parts):
                b2.lines[q] += len(lines or ())
                if lines and any(line.startswith('|-item|') for line in lines):hard[q, q % 2, 0, 0] = 1
            return []
        b2.feed = feed
        lines = ['|debug|filtered', '|move|p2a: Tauros|Return|p1a: Snorlax',
                 '|-damage|p1a: Snorlax|50/100', '|-item|p2a: Tauros|Leftovers',
                 '|-damage|p1a: Snorlax|40/100|[from] psn', '|-damage|p2a: Tauros|90/100']
        reference.feed_b2(b2, [lines, None])
        self.assertEqual(batches, [[lines[:3], None], [lines[3:], None]])
        self.assertEqual(set(reference.damage), {0})
        self.assertEqual(set(reference.damage[0]), {12})
        self.assertEqual(reference.damage[0][12]['hard'].tolist(), [3, 0])
        self.assertEqual(hard[0, 0, 0].tolist(), [1, 0])
        self.assertEqual(reference.damage[0][12]['encounter'], 1)
        b2.refused[0] = {'kind': 'FOLD_ERROR'}
        reference.feed_b2(b2, [[], None])
        self.assertNotIn(0, reference.damage)
        reference.reset()
        self.assertEqual(reference.damage, {})


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class B2CaptureParityTests(unittest.TestCase):
    def test_split_feed_matches_unsplit_state_and_captures_pre_reveal_support(self):
        from dataclasses import replace
        from gpu_engine import effect_probe_device as device, public_belief, v4_encoder
        from test_gpu_engine_v4_encoder import V4EncoderTests
        V4EncoderTests.setUpClass()
        fixture = V4EncoderTests
        source = fixture.public.source
        universes = dict(source.universes)
        original = universes['snorlax']
        variant = original.variants[0]
        universes['snorlax'] = replace(original, variants=(
            replace(variant, variant_id='immune', ability='Immunity'),
            replace(variant, variant_id='pressure', ability='Pressure')))
        source = replace(source, universes=universes,
                         metadata=replace(source.metadata, source_hash='probe-damage-capture-fixture'))
        catalog = v4_encoder.compile_encoder(public_belief.compile_source(source), fixture.inputdex, fixture.vocab,
            checkpoint_sha256='a' * 64, feature_masks=fixture.masks)
        ordinary, split = device.B2Batch(catalog, 2), device.B2Batch(catalog, 2)
        split.fold = ordinary.fold
        ordinary.reset([0, 1]);split.reset([0, 1])
        reference = Narrowing.__new__(Narrowing)
        reference.sp = SimpleNamespace(species_key=str.lower);reference.reset()
        prefix = ['|switch|p1a: Absol|Absol, L80|100/100',
                  '|switch|p2a: Snorlax|Snorlax, L80|100/100', '|turn|1']
        lines = ['|debug|filtered'] + ['|'] * 70 + [
            '|move|p2a: Snorlax|Body Slam|p1a: Absol', '|-damage|p1a: Absol|70/100',
            '|-ability|p2a: Snorlax|Pressure', '|', '|turn|2']
        mirror = lambda rows: [s.replace('p1', 'pX').replace('p2', 'p1').replace('pX', 'p2') for s in rows]
        for delta in ([prefix, mirror(prefix)], [lines, mirror(lines)]):
            self.assertEqual(ordinary.feed(delta), [])
            self.assertEqual(reference.feed_b2(split, delta), [])
        self.assertEqual(split.refused, [None, None])
        self.assertEqual(split.lines, ordinary.lines)
        self.assertTrue(device.trees_equal(ordinary.host(), split.host()))
        offset = len(prefix) + lines.index('|-damage|p1a: Absol|70/100')
        final = split.host()
        for q in (0, 1):
            self.assertEqual(set(reference.damage[q]), {offset})
            saved = reference.damage[q][offset]
            self.assertEqual(saved['species'], 'snorlax')
            self.assertEqual(saved['hard'].tolist(), [3, 0])
            self.assertEqual(final.beliefs.hard[q, q, saved['slot']].tolist(), [2, 0])


if __name__ == '__main__':
    unittest.main()
