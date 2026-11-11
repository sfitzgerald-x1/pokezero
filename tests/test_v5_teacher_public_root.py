"""Real process-free owning parser, tiny authored packets, sampler mocked only.

No actual posterior draw, team/engine constructor, native, model or game runs.
"""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero import v5_teacher_public_root as r
from pokezero.live_protocol_actor import ActorContract, _vocab_sha
from pokezero.v5_contract import DecisionAddress
from pokezero.v5_gumbel import Recipe
from test_saved_player_projection import ProductionHistoryTests, leads, request


class PublicRootTests(unittest.TestCase):
    def setUp(self):
        self.f = ProductionHistoryTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.source.metadata = SimpleNamespace(source_hash='b'*16)
        self.contract = replace(ActorContract.champion(), belief_set_source_hash='b'*16,
            category_vocab_sha256=_vocab_sha(self.f.config.category_vocab),
            categorical_vocab_size=self.f.config.category_vocab.size)

    def lines(self, owner='p1'):
        lines = leads()
        if owner == 'p1':lines = [line.replace('200/301', '67/100') for line in lines]
        else:lines = [line.replace('300/400', '75/100') for line in lines]
        return lines + [request(owner, move='thunderbolt' if owner=='p2' else 'softboiled')]

    def capture(self, lines=None, owner=0):
        return r.capture_root(config=self.f.config, contract=self.contract,
            lines=self.lines(('p1','p2')[owner]) if lines is None else lines,
            address=DecisionAddress(17, 2, owner), battle_id='authored-only', seed=7)

    def test_original_owner_materialization_separate_views_and_exact_own_request(self):
        for owner in (0, 1):
            root = self.capture(owner=owner)
            side = ('p1','p2')[owner]
            peer = ('p2','p1')[owner]
            state = root.context.public_materialization_state
            self.assertFalse(root.device_branch_ready)
            self.assertEqual(state.replay.requests, {})
            self.assertEqual(state.self_request['side']['id'], side)
            self.assertEqual(set(root.context.requested_observations), {side})
            self.assertEqual(set(root.context.requested_legal_action_masks), {side})
            self.assertFalse(root.context.trajectory.steps)  # never fabricate historical slot choices
            payload = root.materialization_payload
            self.assertEqual(payload['sides'][peer]['pokemon'][0]['condition'], '67/100' if owner==0 else '75/100')
            self.assertTrue(state.self_initial_request)
            self.assertTrue(state.self_move_states)

    def test_foreign_private_request_exact_peer_hp_and_truncated_prefix_refused(self):
        for lines in (self.lines()+[request('p2')], leads()+[request('p1')],
                [line for line in self.lines() if line != '|start'], self.lines()+['|turn|2']):
            with self.assertRaises(ValueError):self.capture(lines)
        poisoned = self.lines()+['|-sethp|p1a: Blissey|100/400|p2a: Gengar|100/301', request('p1')]
        with self.assertRaisesRegex(ValueError, 'opponent exact HP'):self.capture(poisoned)

    def test_original_current_payload_keeps_public_clock_and_belief_reveals(self):
        lines = self.lines()+['|move|p2a: Gengar|Thunderbolt|p1a: Blissey',
            '|-start|p2a: Gengar|confusion', '|turn|2', request('p1')]
        root = self.capture(lines)
        self.assertIn('confusion', root.materialization_payload['sides']['p2']['volatiles'])
        self.assertEqual(root.materialization_payload['turn'], 2)
        self.assertIn('thunderbolt', str(root.context.observation.metadata['belief_view']).lower())
        self.assertIn('|turn|2', root.context.public_materialization_state.replay.public_lines)

    def test_original_initial_request_and_current_pp_do_not_alias(self):
        lines = self.lines()
        current = json.loads(lines[-1][len('|request|'):])
        current['side']['pokemon'][0]['condition'] = '100/400'
        current['side']['pokemon'][0]['stats']['atk'] = 999
        current['active'][0]['moves'][0]['pp'] = 7
        lines += ['|turn|2', '|request|'+json.dumps(current)]
        root = self.capture(lines)
        state = root.context.public_materialization_state
        self.assertEqual(state.self_initial_request['side']['pokemon'][0]['stats']['atk'], 120)
        self.assertEqual(state.self_request['side']['pokemon'][0]['stats']['atk'], 999)
        self.assertEqual(state.self_request['side']['pokemon'][0]['condition'], '100/400')
        self.assertEqual(next(iter(state.self_move_states.values()))[0]['pp'], 7)

    def test_original_public_blocker_is_nonfatal_and_counted(self):
        root = self.capture(self.lines()+['|-start|p2a: Gengar|Leech Seed', '|turn|2', request('p1')])
        self.assertIn(('leechseed-source-unknown', 1), root.unsupported_counts)
        self.assertFalse(root.device_branch_ready)

    def test_exact_two_original_sampler_calls_unknown_slot_not_refilled(self):
        root = self.capture(); marker = object()
        with patch.object(r, '_gen3_randbat_belief_start_override_result',
                side_effect=[(None, 'unknown-clock'), (marker, None)]) as sampler:
            result = r.plan_starts(root, Recipe(11), set_source=self.f.source)
        self.assertEqual(result.required_worlds, 2)
        self.assertEqual(result.starts, (None, marker))
        self.assertEqual(result.refusals, ('unknown-clock', None))
        self.assertFalse(result.device_branch_ready)
        self.assertEqual(sampler.call_count, 2)
        for call in sampler.call_args_list:
            self.assertIs(call.kwargs['context'], root.context)
            self.assertTrue(call.kwargs['canonicalize_revealed_variants'])
        self.assertNotEqual(sampler.call_args_list[0].kwargs['rng'].getstate(), sampler.call_args_list[1].kwargs['rng'].getstate())
        foreign = SimpleNamespace(metadata=SimpleNamespace(source_hash='foreign'))
        with self.assertRaisesRegex(ValueError, 'source differ'):r.plan_starts(root, Recipe(11), set_source=foreign)


def digest(root):
    """What two PublicRoots of one prefix must share: identity, payload, counts, and the context's public state and observation."""
    state = root.context.public_materialization_state;obs = root.context.observation
    return (root.address, root.owner_lines_sha256, json.dumps(root.materialization_payload, sort_keys=True, default=repr), root.unsupported_counts,
            state.player_id, state.format_id, state.replay.public_lines, json.dumps(state.self_request, sort_keys=True),
            json.dumps(state.self_initial_request, sort_keys=True), repr(state.self_move_states), repr(sorted(obs.metadata.items())),
            tuple(obs.legal_action_mask), root.context.requested_legal_action_masks, root.context.battle_id, root.context.seed)


class RootSessionTests(unittest.TestCase):
    setUp = PublicRootTests.setUp
    lines = PublicRootTests.lines

    def session(self, owner=0):
        return r.RootSession(config=self.f.config, contract=self.contract, battle_id='authored-only', owner=('p1', 'p2')[owner], stream_id=17)

    def root(self, session, lines, owner=0, index=2):
        return session.root(contract=self.contract, lines=lines, address=DecisionAddress(17, index, owner), battle_id='authored-only', seed=7)

    def prefixes(self, owner=0):
        first = self.lines(('p1', 'p2')[owner]);side = ('p1', 'p2')[owner]
        second = first + ['|move|p2a: Gengar|Thunderbolt|p1a: Blissey', '|-start|p2a: Gengar|confusion', '|turn|2', request(side)]
        third = second + ['|-start|p2a: Gengar|Leech Seed', '|turn|3', request(side)]
        return first, second, third

    def test_a_session_over_successive_requests_gives_capture_root_s_root_for_each_prefix(self):
        for owner in (0, 1):
            session = self.session(owner)
            for i, lines in enumerate(self.prefixes(owner)):
                self.assertTrue(session.extends(lines))
                held = self.root(session, lines, owner, index=i)
                fresh = r.capture_root(config=self.f.config, contract=self.contract, lines=lines, address=DecisionAddress(17, i, owner),
                    battle_id='authored-only', seed=7)
                self.assertEqual(digest(held), digest(fresh))
                self.assertFalse(session.extends(lines))   # a request is folded once

    def test_a_non_successor_or_any_error_closes_the_session(self):
        first, second, third = self.prefixes()
        session = self.session();self.root(session, second)
        self.assertFalse(session.extends(first))
        with self.assertRaisesRegex(ValueError, 'successor'):self.root(session, first)
        self.assertFalse(session.open);self.assertFalse(session.extends(third))
        with self.assertRaisesRegex(ValueError, 'closed'):self.root(session, third)
        session = self.session();self.root(session, first)
        poisoned = first + ['|-sethp|p1a: Blissey|100/400|p2a: Gengar|100/301', request('p1')]
        with self.assertRaisesRegex(ValueError, 'opponent exact HP'):self.root(session, poisoned)
        self.assertFalse(session.open)
        with self.assertRaisesRegex(ValueError, 'another owning stream'):self.root(self.session(), first, owner=1)


if __name__ == '__main__':unittest.main()
