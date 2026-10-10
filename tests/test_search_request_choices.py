"""Request-only snapshot choices must agree with full player normalization."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from _showdown_root import requires_showdown, showdown_root
from pokezero.actions import ACTION_COUNT
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.observation import ObservationFeatureMasks
from pokezero.showdown import (
    V2_1_REPLAY_OBSERVATION_SPEC, normalize_for_player, parse_showdown_replay,
    showdown_choice_for_action, showdown_choices_for_request,
)
from pokezero.showdown_fixture import FixturePokemon, pack_team


def legacy_choices(env):
    result = {}
    for player in env.requested_players():
        state = env._state_for_player(player)
        result[player] = {i: showdown_choice_for_action(state, i)
                         for i in range(ACTION_COUNT) if state.legal_action_mask[i]}
    return result


class RequestChoiceTests(unittest.TestCase):
    def test_request_layouts_match_full_public_state_oracle_without_mutation(self):
        count = 0
        for player in ('p1', 'p2'):
            for size in range(1, 7):
                for active_index in range(size):
                    for kind in ('move', 'trapped', 'maybeTrapped', 'force', 'wait'):
                        party = [dict(ident=f'{player}: Pikachu', details='Pikachu, L50',
                                      active=i == active_index,
                                      condition='0 fnt' if i != active_index and i % 3 == 0 else '100/100',
                                      moves=['thunderbolt', 'tackle']) for i in range(size)]
                        request = dict(side=dict(id=player, pokemon=party), active=[dict(moves=[
                            dict(id='thunderbolt', disabled=False), dict(id='tackle', disabled=True),
                            dict(id='splash', disabled=False), dict(id='protect', disabled=False)])])
                        if kind in ('trapped', 'maybeTrapped'):
                            request['active'][0][kind] = True
                        elif kind == 'force':
                            request['forceSwitch'] = [True]
                        elif kind == 'wait':
                            request['wait'] = True
                        original = deepcopy(request)
                        replay = parse_showdown_replay([f'|{player}|request|{json.dumps(request)}'])
                        # Requests are player-scoped transport; set them explicitly
                        # as the parser's real request event does for a private stream.
                        replay = replace(replay, requests={player: request})
                        state = normalize_for_player(replay, player_id=player, configured_showdown_slot=player)
                        expected = {i: showdown_choice_for_action(state, i)
                                    for i in range(ACTION_COUNT) if state.legal_action_mask[i]}
                        self.assertEqual(showdown_choices_for_request(request, player), expected)
                        self.assertEqual(request, original)
                        count += 1
        self.assertEqual(count, 210)

    def test_empty_request_and_struggle_match_existing_legality(self):
        self.assertEqual(showdown_choices_for_request(None, 'p1'), {})
        self.assertEqual(showdown_choices_for_request({}, 'p1'), {})
        request = dict(active=[dict(moves=[dict(id='struggle', pp=0, disabled=False)])])
        self.assertEqual(showdown_choices_for_request(request, 'p1'), {0: 'move 1'})

    def test_shared_annotations_preserve_producer_creation_and_evaluation_order(self):
        calls, raw, annotated = [], ('raw',), ('annotated',)
        tracker = SimpleNamespace(annotate=lambda replay, tokens, belief:
            calls.append(('residual', tokens)) or annotated)
        investment = SimpleNamespace(observe=lambda replay, tokens, belief:
            calls.append(('investment', tokens)) or {})
        shell = SimpleNamespace(_belief_engine=object(),
            _tier2_tracker_for=lambda player: calls.append(('create_residual', player)) or tracker,
            _investment_tracker_for=lambda player: calls.append(('create_investment', player)) or investment)
        self.assertEqual(LocalShowdownEnv._annotate_transition_tokens(shell, 'p2', object(), raw),
                         (annotated, True))
        self.assertEqual(calls, [('create_residual', 'p2'), ('create_investment', 'p2'),
                                 ('residual', raw), ('investment', annotated)])


@requires_showdown()
class NativeRequestChoiceTests(unittest.TestCase):
    def config(self, annotations=False):
        return LocalShowdownConfig(showdown_root=showdown_root(),
            observation_spec=V2_1_REPLAY_OBSERVATION_SPEC,
            set_belief_source=annotations,
            feature_masks=ObservationFeatureMasks(tier2_investment=annotations))

    def override(self):
        return BattleStartOverride(observation_format_id='gen3randombattle', player_teams={
            'p1': pack_team((FixturePokemon('Tauros', ('Double-Edge', 'Earthquake'), ability='Intimidate', level=76),
                             FixturePokemon('Snorlax', ('Body Slam', 'Splash'), ability='Immunity', level=71))),
            'p2': pack_team((FixturePokemon('Snorlax', ('Body Slam', 'Earthquake'), ability='Immunity', level=71),
                             FixturePokemon('Tauros', ('Double-Edge', 'Splash'), ability='Intimidate', level=76)))})

    def test_native_snapshots_choices_observations_and_chance_match_legacy(self):
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=23, start_override=self.override())
            env.step({'p1': 0, 'p2': 0})
            before = env.snapshot()
            with mock.patch.object(env, '_state_for_player', side_effect=AssertionError('full state called')):
                fast = env.snapshot_for_search()
                repeated = env.snapshot_for_search()
            self.assertEqual(fast.search_choice_cache, repeated.search_choice_cache)
            after = env.snapshot()
            self.assertEqual(before.bridge_snapshot, after.bridge_snapshot)
            self.assertEqual(before.belief_engine.snapshot(), after.belief_engine.snapshot())
            self.assertEqual(before.annotation_cache, after.annotation_cache)
            with mock.patch.object(env, '_search_choice_cache', side_effect=lambda: legacy_choices(env)):
                legacy = env.snapshot_for_search()
            self.assertEqual(fast.search_choice_cache, legacy.search_choice_cache)
            self.assertEqual(fast.annotation_cache, legacy.annotation_cache)
            expected = {}
            for snapshot in (fast, legacy, repeated):
                env.restore_search_snapshot(snapshot)
                obs = {player: env.observe(player) for player in env.requested_players()}
                if expected:
                    self.assertEqual(obs, expected)
                expected = obs
            actions = {player: min(choices) for player, choices in fast.search_choice_cache.items()}
            first = env.step_from_search_snapshot_for_conditioning(fast, actions, chance_seed=117)
            first_history = tuple(e.raw_line for e in env._parser.public_events)
            second = env.step_from_search_snapshot_for_conditioning(legacy, actions, chance_seed=117)
            self.assertEqual(first, second)
            self.assertEqual(first_history, tuple(e.raw_line for e in env._parser.public_events))
            for snapshot in (fast, repeated, legacy):
                env.release_search_snapshot(snapshot)

    def test_empty_and_populated_annotation_trackers_match_legacy_snapshot_path(self):
        with LocalShowdownEnv(self.config(annotations=True)) as env:
            env.reset_with_start_override(seed=23, start_override=self.override())
            env.step({'p1': 0, 'p2': 0})
            for initially_empty in (True, False):
                if initially_empty:
                    env._tier2_trackers = {}
                    env._investment_trackers = {}
                before = env.snapshot()
                with mock.patch.object(env, '_state_for_player', wraps=env._state_for_player) as full:
                    snapshot = env.snapshot_for_search()
                self.assertEqual(full.call_count, 0)
                self.assertEqual(set(snapshot.annotation_cache.tier2_trackers), {'p1', 'p2'})
                self.assertEqual(set(snapshot.annotation_cache.investment_trackers), {'p1', 'p2'})
                self.assertEqual(env.snapshot().bridge_snapshot, before.bridge_snapshot)
                self.assertEqual(set(env._tier2_trackers), set(before.annotation_cache.tier2_trackers))
                with mock.patch.object(env, '_search_choice_cache', side_effect=lambda: legacy_choices(env)):
                    legacy = env.snapshot_for_search()
                self.assertEqual(snapshot.search_choice_cache, legacy.search_choice_cache)
                env.restore_search_snapshot(snapshot)
                expected = {p: env.observe(p) for p in env.requested_players()}
                env.restore_search_snapshot(legacy)
                self.assertEqual({p: env.observe(p) for p in env.requested_players()}, expected)
                env.release_search_snapshot(snapshot)
                env.release_search_snapshot(legacy)
