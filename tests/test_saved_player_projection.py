"""Authored protocol/metadata fixtures only; no saved corpus or runtime execution."""
from contextlib import ExitStack
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from pokezero.belief import CandidateSetSummary
from pokezero.category_vocab import build_category_vocabulary
from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo
from pokezero.local_showdown import LocalShowdownConfig
from pokezero.observation import ObservationFeatureMasks
from pokezero.saved_player_projection import (INTIMIDATE_HINT, PINNED_SHOWDOWN_REVISION,
    RULES, SHARED_HINTS, PlayerSeatProjection, SavedSourcePlayerProjector, assert_struggle_omission_boundary,
    feed_player_reference_boundary, new_player_reference_environments)
from pokezero.showdown import V4_REPLAY_OBSERVATION_SPEC

TENSORS = ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask', 'legal_action_mask')


def setup_lines():
    return ['|gen|3', '|gametype|singles', '|tier|[Gen 3] Random Battle',
            '|player|p1|Authored one|', '|player|p2|Authored two|',
            '|teamsize|p1|6', '|teamsize|p2|6',
            *['|rule|' + rule for rule in sorted(RULES)], '|start']


def leads():
    return setup_lines() + ['|switch|p1a: Blissey|Blissey, L80, F|300/400',
                            '|switch|p2a: Gengar|Gengar, L80, M|200/301', '|turn|1']


def request(seat, *, active='Blissey', move='struggle', wait=False, forced=False, hp='300/400'):
    species = ['Blissey', 'Alakazam'] if seat == 'p1' else ['Gengar', 'Arcanine']
    active = active if seat == 'p1' else ('Gengar' if active == 'Blissey' else active)
    order = [active] + [name for name in species if name != active]
    pokemon = [{'ident': f'{seat}: {name}', 'details': name + ', L80', 'condition': hp if i == 0 else '200/300',
                'active': i == 0, 'stats': {'atk': 120, 'def': 130, 'spa': 180, 'spd': 160, 'spe': 140},
                'moves': ['thunderbolt', 'trick'] if name != 'Blissey' else ['softboiled', 'seismictoss'],
                'baseAbility': 'Levitate' if name == 'Gengar' else 'Natural Cure',
                'item': 'Choice Band' if name == 'Alakazam' else 'Leftovers'} for i, name in enumerate(order)]
    value = {'side': {'id': seat, 'pokemon': pokemon}}
    if wait:
        value['wait'] = True
    elif forced:
        value['forceSwitch'] = [True]
    else:
        value['active'] = [{'moves': [{'id': move, 'move': move, 'pp': 1, 'maxpp': 16,
                                     'target': 'normal', 'disabled': False}]}]
    return '|request|' + json.dumps(value)


def move_info(name, kind='Special', power=80):
    return MoveInfo(id=name.lower(), name=name, type='Electric' if kind == 'Special' else 'Normal',
        category=kind, gen3_category=kind, base_power=power, accuracy=100., priority=0,
        recoil=False, drain=False, heal=False, status=None, boosts={}, target='normal', selfdestruct=False, pp=16)


class AuthoredSource:
    def summarize(self, *, species, revealed_moves, **kwargs):
        moves = ('thunderbolt', 'trick', 'softboiled', 'seismictoss')
        variants = ({'variant_id': species + '-fixture', 'moves': moves, 'ability': 'Natural Cure',
                     'item': 'Leftovers', 'level': 80},)
        return CandidateSetSummary(species=species, candidate_count=1, uncertainty=.25,
            possible_abilities=('Natural Cure',), possible_items=('Leftovers',),
            possible_moves=moves, candidate_variants=variants)


def authored_dex():
    species = {name.lower(): SpeciesInfo(id=name.lower(), name=name, types=('Normal',),
               base_stats={'hp': 100, 'atk': 80, 'def': 90, 'spa': 100, 'spd': 100, 'spe': 80})
               for name in ('Blissey', 'Alakazam', 'Gengar', 'Arcanine')}
    moves = {name.lower(): move_info(name, kind, power) for name, kind, power in
             [('Thunderbolt', 'Special', 95), ('Trick', 'Status', 0), ('Struggle', 'Physical', 50),
              ('Softboiled', 'Status', 0), ('Seismictoss', 'Physical', 0)]}
    return ShowdownDex(moves=moves, species=species, type_chart={})


class ProjectorTests(unittest.TestCase):
    def test_health_split_pressure_trace_transform_and_pain_split(self):
        projector = SavedSourcePlayerProjector(battle_id='authored')
        batch = projector.project_nonrequest_delta(leads() + [
            '|-ability|p1a: Blissey|Pressure|[silent]',
            '|-ability|p2a: Gengar|Pressure|Trace|[from] ability: Trace|[of] p1a: Blissey',
            '|-damage|p2a: Gengar|300/301 tox', '|-heal|p1a: Blissey|1/100 par',
            '|-sethp|p1a: Blissey|40/400|[from] move: Pain Split',
            '|-sethp|p2a: Gengar|40/301|[from] move: Pain Split|[silent]',
            '|-transform|p1a: Blissey|p2a: Gengar', '|-damage|p1a: Blissey|0 fnt', '|faint|p1a: Blissey'])
        p1, p2 = batch.p1.lines, batch.p2.lines
        self.assertIn('|switch|p1a: Blissey|Blissey, L80, F|300/400', p1)
        self.assertIn('|switch|p1a: Blissey|Blissey, L80, F|75/100', p2)
        self.assertIn('|-damage|p2a: Gengar|99/100 tox', p1)
        self.assertIn('|-damage|p2a: Gengar|300/301 tox', p2)
        self.assertIn('|-heal|p1a: Blissey|1/100 par', p2)
        self.assertIn('|-sethp|p1a: Blissey|10/100|[from] move: Pain Split', p2)
        self.assertIn('|-sethp|p2a: Gengar|14/100|[from] move: Pain Split|[silent]', p1)
        private = '|-ability|p1a: Blissey|Pressure|[silent]'
        public = '|-ability|p2a: Gengar|Pressure|Trace|[from] ability: Trace|[of] p1a: Blissey'
        self.assertIn(private, p1)
        self.assertNotIn(private, p2)
        for lines in (p1, p2):
            self.assertIn(public, lines)
            self.assertIn('|-transform|p1a: Blissey|p2a: Gengar', lines)
            self.assertIn('|-damage|p1a: Blissey|0 fnt', lines)
        self.assertFalse(batch.public_wire_qualified or batch.captured_player_wire or batch.fullencoder)
        self.assertIn('NOT_FABRICATED', batch.struggle_warning_policy)

    def test_private_intimidate_uses_unique_substitute_not_last_switch(self):
        for substituted, owner in (('p1', 'p2'), ('p2', 'p1')):
            projector = SavedSourcePlayerProjector(battle_id='authored-' + owner)
            projector.project_nonrequest_delta(leads())
            # No adjacent switch at all: ownership follows the emitter's
            # substitute predicate, not a nickname or last-switch heuristic.
            batch = projector.project_nonrequest_delta([
                f'|-start|{substituted}a: Authored|Substitute', '|-hint|' + INTIMIDATE_HINT])
            self.assertIn('|-hint|' + INTIMIDATE_HINT, batch.for_seat(owner).lines)
            self.assertNotIn('|-hint|' + INTIMIDATE_HINT, batch.for_seat(substituted).lines)

    def test_ambiguous_hint_refuses_atomically_without_filtering(self):
        for sublines in ([], ['|-start|p1a: Blissey|Substitute', '|-start|p2a: Gengar|Substitute']):
            projector = SavedSourcePlayerProjector(battle_id='authored')
            initial = projector.project_nonrequest_delta(leads())
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                projector.project_nonrequest_delta(sublines + ['|-hint|' + INTIMIDATE_HINT])
            after = projector.project_nonrequest_delta(['|turn|2'])
            self.assertEqual(after.source_nonrequest_start, initial.source_nonrequest_end)
            self.assertFalse(any('substitute' in v for v in projector._parser.volatiles.values()))

    def test_shared_diagnostics_are_preserved(self):
        projector = SavedSourcePlayerProjector(battle_id='authored')
        diagnostics = ['|-hint|' + hint for hint in sorted(SHARED_HINTS)] + ['|-message|Sleep Clause Mod activated.']
        batch = projector.project_nonrequest_delta(leads() + diagnostics)
        for seat in ('p1', 'p2'):
            for line in diagnostics:
                self.assertIn(line, batch.for_seat(seat).lines)

    def test_unknown_families_payloads_formats_and_private_warning_refuse(self):
        for suffix in (['|split|p1'], ['|request|{}'], ['|-hint|unknown'], ['|-message|unknown'],
                       ['|-ability|p1a: Blissey|Pressure'], ['|-damage|p1a: Blissey|101/100'],
                       ['|-activate|p1a: Blissey|move: Struggle']):
            projector = SavedSourcePlayerProjector(battle_id='authored')
            with self.assertRaises(ValueError):
                projector.project_nonrequest_delta(leads() + suffix)
            self.assertEqual(projector._count, 0)
        with self.assertRaises(ValueError):
            SavedSourcePlayerProjector(battle_id='authored', showdown_revision='other')
        projector = SavedSourcePlayerProjector(battle_id='authored')
        with self.assertRaises(ValueError):
            projector.project_nonrequest_delta([line for line in leads() if 'HP Percentage Mod:' not in line])


class ProductionHistoryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Source execution, simulator processes, network and real model/NPZ
        # reads are not allowed by these authored source-consumer tests.
        self.stack.enter_context(patch('subprocess.run', side_effect=AssertionError('no runtime execution')))
        self.stack.enter_context(patch('subprocess.Popen', side_effect=AssertionError('no simulator process')))
        # This is an authored vocabulary, not the champion's enumeration; full
        # semantic state is compared separately from the five tensor arrays.
        self.stack.enter_context(patch('pokezero.category_vocab.logger.warning'))
        self.source = AuthoredSource()
        self.stack.enter_context(patch('pokezero.local_showdown.load_gen3_randbat_source_cached', return_value=self.source))
        self.stack.enter_context(patch('pokezero.local_showdown.load_showdown_dex_cached', return_value=authored_dex()))
        vocab = build_category_vocabulary(['request_kind:move', 'request_kind:force-switch',
            'species:Blissey', 'species:Alakazam', 'species:Gengar', 'species:Arcanine',
            'move:struggle', 'move:thunderbolt', 'move:trick', 'belief:possible_ability:naturalcure',
            'belief:possible_item:leftovers', 'volatile:substitute'])
        self.config = LocalShowdownConfig(showdown_root='/authored-no-runtime', set_belief_source=True,
            category_vocab=vocab, observation_spec=V4_REPLAY_OBSERVATION_SPEC,
            feature_masks=ObservationFeatureMasks(exact_state=True, transition_token_budget=0,
                opponent_tendency_stats_block=True, tier2_residuals=True, tier2_investment=True,
                investment_belief_narrowing=False, item_belief_narrowing=False, feature_pack_last_move=True))

    def make(self):
        return new_player_reference_environments(self.config, battle_id='authored')

    def feed(self, envs, batch, requests=None, requested=('p1', 'p2')):
        requests = requests or {'p1': (request('p1'),), 'p2': (request('p2', move='thunderbolt', hp='200/301'),)}
        return feed_player_reference_boundary(envs, batch, own_request_deltas=requests,
                    requested_seats=requested, request_tail_verified=True)

    def assert_tensors(self, first, second):
        for name in TENSORS:
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)
        self.assertEqual(np.asarray(first.categorical_ids).shape, (23, 41))
        self.assertEqual(np.asarray(first.numeric_features).shape, (23, 132))

    def assert_consumers(self, first, second, seat):
        # The two intentional differences are raw diagnostic history and the
        # diagnostic belief event counter. Neither is read by the five tensors.
        self.assertEqual(replace(first._parser.snapshot(), public_events=(), public_lines=()),
                         replace(second._parser.snapshot(), public_events=(), public_lines=()))
        a, b = vars(first._belief_engine), vars(second._belief_engine)
        self.assertEqual({k: v for k, v in a.items() if k != '_event_count'},
                         {k: v for k, v in b.items() if k != '_event_count'})
        a, b = first._state_for_player(seat), second._state_for_player(seat)
        self.assertEqual(replace(a, recent_events=(), recent_public_events=()),
                         replace(b, recent_events=(), recent_public_events=()))
        for attr in ('_tier2_trackers', '_investment_trackers'):
            left, right = getattr(first, attr)[seat], getattr(second, attr)[seat]
            self.assertEqual(left._assessed_until, right._assessed_until)
            self.assertEqual(left.belief_narrowing_count, right.belief_narrowing_count)
            # Raw line indices necessarily shift, but their action-aligned
            # strike contexts and every assessed result must be identical.
            af, bf = left._fold, right._fold
            self.assertEqual([af.contexts.get(i) for i in af.token_line_indices],
                             [bf.contexts.get(i) for i in bf.token_line_indices])
            if attr == '_tier2_trackers':
                for key in ('_residuals', '_cb_turns', '_cb_non_ko'):
                    self.assertEqual(getattr(left, key), getattr(right, key))
            else:
                self.assertEqual(left.token_codes, right.token_codes)
                self.assertEqual(left.conclusions, right.conclusions)

    def test_independent_owning_histories_waits_and_foreign_request_refusal(self):
        envs = self.make()
        projector = SavedSourcePlayerProjector(battle_id='authored')
        self.feed(envs, projector.project_nonrequest_delta(leads()))
        for seat in ('p1', 'p2'):
            self.assertEqual(set(envs[seat]._parser.requests), {seat})
            with self.assertRaises(ValueError):
                envs[seat].observe('p2' if seat == 'p1' else 'p1')
        self.assertIsNot(envs['p1']._belief_engine, envs['p2']._belief_engine)
        self.assertIsNot(envs['p1']._tier2_trackers['p1'], envs['p2']._tier2_trackers['p2'])
        batch = projector.project_nonrequest_delta(['|-damage|p2a: Gengar|0 fnt', '|faint|p2a: Gengar'])
        before = {s: envs[s].protocol_lines for s in envs}
        with self.assertRaisesRegex(ValueError, 'foreign request'):
            self.feed(envs, batch, {'p1': (request('p2'),), 'p2': ()}, requested=('p2',))
        self.assertEqual(before, {s: envs[s].protocol_lines for s in envs})
        observed = self.feed(envs, batch, {'p1': (request('p1', wait=True),),
            'p2': (request('p2', forced=True, hp='0 fnt'),)}, requested=('p2',))
        self.assertEqual(set(observed), {'p2'})
        self.assertTrue(envs['p1']._latest_requests['p1']['wait'])
        self.assertEqual(len(envs['p1']._request_history['p1']), 2)

    def test_request_tail_proof_and_poisoned_ingestion_are_required(self):
        envs = self.make()
        batch = SavedSourcePlayerProjector(battle_id='authored').project_nonrequest_delta(leads())
        with self.assertRaisesRegex(ValueError, 'tail'):
            feed_player_reference_boundary(envs, batch, own_request_deltas={'p1': (), 'p2': ()}, requested_seats=())
        self.assertTrue(all(not env.protocol_lines for env in envs.values()))
        with self.assertRaisesRegex(ValueError, 'roles'):
            self.feed(envs, batch, requested=('p1',))
        for seat in envs:
            with self.assertRaisesRegex(ValueError, 'failed'):
                envs[seat].observe(seat)

    def test_both_public_channels_are_snapshotted_and_validated_before_mutation(self):
        for bad_line in ('|request|{}', '|-damage|p1a: Blissey|300/400',
                         '|-ability|p1a: Blissey|Pressure|[silent]'):
            envs = self.make()
            batch = SavedSourcePlayerProjector(battle_id='authored').project_nonrequest_delta(leads())
            bad = replace(batch, p2=PlayerSeatProjection('p2', batch.p2.lines + (bad_line,)))
            with self.assertRaises(ValueError):
                self.feed(envs, bad)
            self.assertTrue(all(not env.protocol_lines for env in envs.values()))
        envs = self.make()
        envs['p2']._belief_engine = envs['p1']._belief_engine
        with self.assertRaisesRegex(ValueError, 'alias'):
            self.feed(envs, batch)

    def test_actual_reference_identity_and_schema_are_bound_to_owner(self):
        for mode in ('seat', 'schema', 'metadata_owner', 'boolean_turn'):
            envs = self.make()
            projector = SavedSourcePlayerProjector(battle_id='authored')
            observations = self.feed(envs, projector.project_nonrequest_delta(leads()))
            if mode == 'seat':
                bad = observations['p2']
            elif mode == 'schema':
                bad = replace(observations['p1'], schema_version='other')
            else:
                wrong = {'player_id': 'p2'} if mode == 'metadata_owner' else {'turn_number': True}
                bad = replace(observations['p1'], metadata={**observations['p1'].metadata, **wrong})
            with patch.object(envs['p1'], 'observe', return_value=bad):
                with self.assertRaisesRegex(ValueError, 'identity/schema'):
                    self.feed(envs, projector.project_nonrequest_delta(['|turn|2']))
            self.assertTrue(all(env._player_projection_failed for env in envs.values()))

    def test_both_owners_warning_omission_survives_the_next_boundary(self):
        first, second = self.make(), self.make()
        projector = SavedSourcePlayerProjector(battle_id='authored')
        requests = {'p1': (request('p1'),), 'p2': (request('p2', move='struggle', hp='200/301'),)}
        initial = projector.project_nonrequest_delta(leads())
        self.feed(first, initial, requests)
        self.feed(second, initial, requests)
        for seat, species in (('p1', 'Blissey'), ('p2', 'Gengar')):
            assert_struggle_omission_boundary(second[seat], seat)
            second[seat]._apply_event({'type': 'stream', 'stream': 'omniscient',
                'lines': [f'|-activate|{seat}a: {species}|move: Struggle']})
        next_batch = projector.project_nonrequest_delta([
            '|move|p1a: Blissey|Struggle|p2a: Gengar', '|-damage|p2a: Gengar|180/301',
            '|move|p2a: Gengar|Struggle|p1a: Blissey', '|-damage|p1a: Blissey|280/400',
            '|', '|upkeep', '|turn|2'])
        a, b = self.feed(first, next_batch, requests), self.feed(second, next_batch, requests)
        for seat in ('p1', 'p2'):
            self.assert_tensors(a[seat], b[seat])
            self.assert_consumers(first[seat], second[seat], seat)

    def test_struggle_omission_survives_subsequent_trick_clock_and_annotation_updates(self):
        canonical, captured = self.make(), self.make()
        projector = SavedSourcePlayerProjector(battle_id='authored')
        phases = [leads(),
            ['|move|p1a: Blissey|Struggle|p2a: Gengar', '|-damage|p2a: Gengar|180/301',
             '|-damage|p1a: Blissey|290/400|[from] recoil',
             '|move|p2a: Gengar|Thunderbolt|p1a: Blissey', '|-damage|p1a: Blissey|250/400',
             '|', '|-heal|p1a: Blissey|275/400|[from] item: Leftovers', '|upkeep', '|turn|2'],
            ['|switch|p1a: Alakazam|Alakazam, L80|200/300',
             '|move|p2a: Gengar|Thunderbolt|p1a: Alakazam', '|-damage|p1a: Alakazam|180/300',
             '|', '|upkeep', '|turn|3'],
            ['|move|p1a: Alakazam|Thunderbolt|p2a: Gengar', '|-damage|p2a: Gengar|160/301',
             '|move|p2a: Gengar|Thunderbolt|p1a: Alakazam', '|-damage|p1a: Alakazam|160/300',
             '|', '|upkeep', '|turn|4'],
            ['|move|p1a: Alakazam|Trick|p2a: Gengar',
             '|-activate|p1a: Alakazam|move: Trick|[of] p2a: Gengar',
             '|-item|p1a: Alakazam|Leftovers|[from] move: Trick',
             '|-item|p2a: Gengar|Choice Band|[from] move: Trick',
             '|move|p2a: Gengar|Thunderbolt|p1a: Alakazam', '|-damage|p1a: Alakazam|150/300',
             '|', '|upkeep', '|turn|5'],
            ['|switch|p1a: Blissey|Blissey, L80|275/400',
             '|move|p2a: Gengar|Thunderbolt|p1a: Blissey', '|-damage|p1a: Blissey|250/400',
             '|', '|upkeep', '|turn|6'],
            ['|move|p1a: Blissey|Struggle|p2a: Gengar', '|-damage|p2a: Gengar|150/301',
             '|-damage|p1a: Blissey|240/400|[from] recoil',
             '|move|p2a: Gengar|Thunderbolt|p1a: Blissey', '|-damage|p1a: Blissey|200/400',
             '|', '|upkeep', '|turn|7']]
        for index, lines in enumerate(phases):
            with self.subTest(boundary=index):
                active = 'Alakazam' if index in (2, 3, 4) else 'Blissey'
                move = ('thunderbolt' if index == 2 else 'trick') if active == 'Alakazam' else 'struggle'
                requests = {'p1': (request('p1', active=active, move=move),),
                            'p2': (request('p2', move='thunderbolt', hp='150/301'),)}
                batch = projector.project_nonrequest_delta(lines)
                first, second = self.feed(canonical, batch, requests), self.feed(captured, batch, requests)
                for seat in ('p1', 'p2'):
                    self.assert_tensors(first[seat], second[seat])
                    self.assert_consumers(canonical[seat], captured[seat], seat)
                # A Struggle-only request may still choose switch: index1 does,
                # so no warning is fabricated merely because it was requested.
                if index in (0, 5):
                    assert_struggle_omission_boundary(captured['p1'], 'p1')
                    captured['p1']._apply_event({'type': 'stream', 'stream': 'omniscient',
                        'lines': ['|-activate|p1a: Blissey|move: Struggle']})
                    self.assert_tensors(canonical['p1'].observe('p1'), captured['p1'].observe('p1'))
                    self.assert_consumers(canonical['p1'], captured['p1'], 'p1')
        self.assertGreater(len(canonical['p1']._tier2_trackers['p1']._fold.token_line_indices), 4)
        self.assertGreater(captured['p1']._belief_engine._event_count, canonical['p1']._belief_engine._event_count)

    def test_warning_is_not_a_generic_noop_pending_trick_is_a_real_negative_control(self):
        envs = self.make()
        projector = SavedSourcePlayerProjector(battle_id='authored')
        self.feed(envs, projector.project_nonrequest_delta(leads()))
        env = envs['p1']
        env._apply_event({'type': 'stream', 'stream': 'omniscient',
            'lines': ['|-activate|p1a: Blissey|move: Trick|[of] p2a: Gengar']})
        env._sync_incremental_state()
        self.assertIsNotNone(env._belief_engine._pending_trick)
        with self.assertRaisesRegex(ValueError, 'pending belief'):
            assert_struggle_omission_boundary(env, 'p1')
        env._apply_event({'type': 'stream', 'stream': 'omniscient',
            'lines': ['|-activate|p1a: Blissey|move: Struggle']})
        env._sync_incremental_state()
        self.assertIsNone(env._belief_engine._pending_trick)


if __name__ == '__main__':
    unittest.main()
