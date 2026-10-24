"""Actual native boundary controls; tagged heads are not champion quality evidence."""
from dataclasses import replace
import json
import math
import os
import struct
import unittest
from unittest.mock import patch

from tests import test_leaf_encoder as fixture
from pokezero.one_ply_exact import exact_one_ply, sample_equilibrium
from pokezero.shallow_exact_native import NativeExactBackend, NativePosition, validate_encoder_frame


class TaggedHead:
    def __init__(self, value):
        self.value, self.rows = value, 0

    def eval_obs_flat(self, count, *arrays):
        self.rows += count
        return [self.value]*count, [], [], 9


@unittest.skipUnless(fixture._wheel_has('LeafEncoder'), 'requires actual native encoder')
class NativeOnePlyLeafRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.engine_world import battle_spec_from_payload
        from pokezero.env import BattleStartOverride
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
        from pokezero.poke_engine_adapter import build_poke_engine_state
        if not DEFAULT_SHOWDOWN_ROOT.exists():
            raise unittest.SkipTest('requires pinned Showdown Dex')
        tables = fixture._tables_json()
        if tables is None:
            raise unittest.SkipTest('requires encoder tables')
        cls.native, cls.roots = fixture.pokezero_search, {}
        corpus, folds = fixture._sample_rows_with_folds()
        games = {g.record.battle_id:g for g in corpus.games}
        dex = load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)
        for i, row in enumerate(corpus.decision_rows):
            if row.player_id in cls.roots:
                continue
            game = games[row.battle_id]
            packed = {s:game.record.true_teams[s]['packed'] for s in ('p1','p2')}
            world = battle_spec_from_payload(row.public_materialization,
                BattleStartOverride(player_teams=packed), dex=dex,
                approximate_sleep_turns=True, approximate_substitute_health=True)
            state = build_poke_engine_state(world.spec).to_string()
            ctx = dict(p1=list(world.party_species['p1']), p2=list(world.party_species['p2']),
                       turn=int(row.public_materialization.get('turn') or 0))
            inputs = fixture._row_inputs(row)
            side = {'p1':'side_one','p2':'side_two'}[row.player_id]
            validate_encoder_frame(inputs, side)
            encoder = cls.native.LeafEncoder(tables, inputs, json.dumps(ctx), state)
            root = NativePosition(state, {s:tuple(ctx[s]) for s in ('p1','p2')},
                ctx['turn'], cls.native.FoldState.from_payload(folds[i]), encoder, side)
            # The unmodified committed fixture has these legal moves in both
            # frames. No root HP, move, context or observation is fabricated.
            options = json.loads(cls.native.env_options(state, root=True))
            if 'earthquake' not in options['p1'] or 'fireblast' not in options['p2']:
                raise AssertionError('committed first-turn move identities changed')
            cls.roots[row.player_id] = root
        if set(cls.roots) != {'p1','p2'}:
            raise AssertionError('both actual encoder perspectives required')

    def setUp(self):
        # Scoped precondition only. The runner must set this before native load;
        # importing the suite never switches the process-local engine roll law.
        self.env = patch.dict(os.environ, {'POKEZERO_ENUMERATE_ROLLS':'1'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def branches(self, root):
        branches = list(NativeExactBackend(self.native, TaggedHead(.2)).outcomes(
            root, 'earthquake', 'fireblast'))
        # Literal positive fan on the unchanged committed first-turn fixture.
        # A process that latched the default collapsed law must fail loudly.
        self.assertEqual(len(branches),1306)
        return branches

    def test_real_ko_pending_inputs_keep_fainted_slot_bench_and_history_in_both_frames(self):
        from pokezero.showdown import NUMERIC_HP_FRACTION, SELF_POKEMON_TOKEN_OFFSET
        for seat, root in self.roots.items():
            side = 0 if seat == 'p1' else 1
            backend = NativeExactBackend(self.native, TaggedHead(.2))
            branches = self.branches(root)
            self.assertAlmostEqual(math.fsum(b.probability for b in branches), 1, places=6)
            candidates = [b.state for b in branches if b.state.teams_hp[side][0] == 0
                          and any(b.state.teams_hp[side][1:])]
            self.assertTrue(candidates, 'actual KO with a living bench never reached')
            leaf = candidates[0]
            self.assertIsNone(backend.terminal_value(leaf))
            self.assertEqual(leaf.turn, root.turn, 'pending replacement must not add a turn')
            self.assertTrue(any(line.startswith('|faint|'+seat) for line in leaf.lines))
            self.assertNotEqual(root.fold.to_payload(), leaf.fold.to_payload())
            row = json.loads(root.encoder.leaf_inputs_json(
                leaf.state_str, leaf.turn, list(leaf.lines), False))
            md = row['observation_metadata']
            self.assertEqual(md['request_kind'], 'force_switch')
            active = next(p for p in md['self_team'] if p['active'])
            self.assertEqual(active['condition'], '0 fnt')
            self.assertTrue(any(p['condition'] != '0 fnt' for p in md['self_team'] if not p['active']))
            legal = [p for p in md['action_candidates'] if p['legal']]
            self.assertTrue(legal, 'living bench must produce a legal replacement')
            self.assertTrue(all(p['kind'] == 'switch' for p in legal))
            encoded = root.encoder.encode_leaf(leaf.state_str, leaf.fold, leaf.turn,
                                                list(leaf.lines), False)
            # Specific fainted-HP token, not a generic byte difference that a
            # changed turn alone could satisfy. Self display order stays put.
            numeric = list(struct.unpack('<'+'d'*(len(encoded['numeric_features'])//8),
                                         encoded['numeric_features']))
            tokens = len(encoded['token_type_ids'])//2
            width = len(numeric)//tokens
            index = md['self_team'].index(active)
            self.assertEqual(numeric[(SELF_POKEMON_TOKEN_OFFSET+index)*width+NUMERIC_HP_FRACTION], 0)
            original = root.encoder.encode_leaf(root.state_str, root.fold, root.turn, None, False)
            self.assertGreater(struct.unpack('<'+'d'*(len(original['numeric_features'])//8),
                original['numeric_features'])[(SELF_POKEMON_TOKEN_OFFSET+index)*width+NUMERIC_HP_FRACTION], 0)
            self.assertNotEqual(original['numeric_features'], encoded['numeric_features'])

    def test_one_ply_values_pending_states_without_child_search_and_converts_each_frame_once(self):
        for seat, root in self.roots.items():
            head = TaggedHead(.2 if seat == 'p1' else .6)
            class OnePair(NativeExactBackend):
                def actions(inner, state, *, root):
                    self.assertTrue(root, 'child replacement actions must not be searched')
                    self.assertEqual(state.state_str, self.roots[seat].state_str)
                    return ('earthquake',), ('fireblast',)
                def outcomes(inner, state, row, col):
                    self.assertEqual(state.state_str, self.roots[seat].state_str)
                    return super().outcomes(state, row, col)
            backend = OnePair(self.native, head)
            result = exact_one_ply([root], backend, batch_size=256,
                max_joint_actions=1, max_leaves=5000, max_chance_branches=5000,
                max_seconds=60, cross_cell_batches=True)
            self.assertGreater(result.leaf_evaluations, 0)
            self.assertEqual(result.chance_branches,1306)
            self.assertEqual(head.rows, result.leaf_evaluations)
            self.assertEqual(result.leaf_evaluations, result.chance_branches)
            self.assertAlmostEqual(result.root_matrix[0][0], .6 if seat == 'p1' else .2)
            played = sample_equilibrium(result, game_seed=33, turn=root.turn,
                                        side=0 if seat == 'p1' else 1)
            self.assertEqual(played['action'], 'earthquake' if seat == 'p1' else 'fireblast')
            self.assertEqual(played['mix'], (1.,))

    def test_actual_terminal_successors_bypass_head_but_active_only_ko_does_not(self):
        from pokezero.poke_engine_adapter import (BattleSpec, MoveSpec, PokemonSpec,
                                                 SideSpec, build_poke_engine_state)
        def mon(name, hp, move, speed):
            return PokemonSpec(name,100,('normal',),hp,100,200,100,100,100,speed,
                               (MoveSpec(move,8),),ability='none',item='none')
        attacker, victim = mon('snorlax',100,'seismictoss',200), mon('blissey',1,'splash',100)
        class NoHead:
            def eval_obs_flat(inner, *args):
                self.fail('complete terminal successor must bypass the head')
        for winner in (0,1):
            sides = [SideSpec((attacker,)),SideSpec((victim,))]
            if winner:sides.reverse()
            state = build_poke_engine_state(BattleSpec(*sides)).to_string()
            root = NativePosition(state,{'p1':tuple(p.id for p in sides[0].pokemon),
                'p2':tuple(p.id for p in sides[1].pokemon)},1,
                self.native.FoldState.initial('p1'),None,'side_one')
            class OnePair(NativeExactBackend):
                def actions(inner, position, *, root):
                    self.assertTrue(root)
                    return (('splash',),('seismictoss',)) if winner else (('seismictoss',),('splash',))
            result = exact_one_ply([root],OnePair(self.native,NoHead()),max_joint_actions=1)
            self.assertEqual(result.root_matrix, ((0. if winner else 1.,),))
            self.assertEqual(result.leaf_evaluations,0)
        # Gen3 Explosion can exhaust both complete parties. The native options
        # terminal sign is not trusted for a double wipe: team HP owns the draw.
        exploder = mon('snorlax',100,'explosion',200)
        state = build_poke_engine_state(BattleSpec(SideSpec((exploder,)),SideSpec((victim,)))).to_string()
        root = replace(root,state_str=state,party_species={'p1':('snorlax',),'p2':('blissey',)})
        backend = NativeExactBackend(self.native,NoHead())
        branches = list(backend.outcomes(root,'explosion','splash'))
        self.assertTrue(branches)
        self.assertTrue(all(backend.terminal_value(b.state)==0 for b in branches))
        self.assertTrue(all(all(hp==0 for team in b.state.teams_hp for hp in team) for b in branches))


@unittest.skipUnless(fixture._wheel_has('branch_events'), 'requires native transitions')
class NativeForcedPassBoundaryTests(unittest.TestCase):
    def test_authored_forced_replacement_keeps_passive_seat_and_switch_identity(self):
        from pokezero.poke_engine_adapter import (
            BattleSpec, MoveSpec, PokemonSpec, SideSpec, build_poke_engine_state,
        )
        native = fixture.pokezero_search

        def mon(name, hp):
            return PokemonSpec(name, 100, ('normal',), hp, 100, 100, 100,
                               100, 100, 100, (MoveSpec('splash', 8),),
                               ability='none', item='none')

        with patch.dict(os.environ, {'POKEZERO_ENUMERATE_ROLLS': '1'}):
            for active_side in (0, 1):
                with self.subTest(active_side=active_side):
                    sides = [SideSpec((mon('snorlax', 0), mon('blissey', 100)),
                                      force_switch=True),
                             SideSpec((mon('ditto', 100),))]
                    if active_side:
                        sides.reverse()
                    state = build_poke_engine_state(BattleSpec(*sides)).to_string()
                    root = NativePosition(state,
                        {seat: tuple(p.id for p in side.pokemon)
                         for seat, side in zip(('p1', 'p2'), sides)},
                        3, native.FoldState.initial('p1'), None, 'side_one')
                    backend = NativeExactBackend(native, None)
                    actions = backend.actions(root, root=True)
                    self.assertEqual(actions[active_side], ('switch blissey',))
                    self.assertEqual(actions[1 - active_side], ('No Move',))
                    tokens = ('blissey', 'none') if active_side == 0 else ('none', 'blissey')
                    direct = json.loads(native.branch_events(state, *tokens,
                        json.dumps({**root.party_species, 'turn': root.turn}),
                        branch_on_damage=True, include_post_state=True))['branches']
                    branches = list(backend.outcomes(root, actions[0][0], actions[1][0]))
                    self.assertTrue(branches)
                    self.assertEqual(len(branches), len(direct))
                    self.assertAlmostEqual(math.fsum(b.probability for b in branches), 1)
                    for branch, expected in zip(branches, direct):
                        self.assertEqual(branch.state.state_str, expected['post_state'])
                        self.assertEqual(branch.state.lines, tuple(expected['events']))
                        self.assertEqual(branch.state.turn,
                                         root.turn + int(expected['turn_completed']))
                        self.assertEqual(branch.state.teams_hp[1 - active_side], (100,))
                        options = json.loads(native.env_options(branch.state.state_str, root=True))
                        self.assertFalse(options['p1_force_switch'])
                        self.assertFalse(options['p2_force_switch'])
                        self.assertTrue(any(line.startswith('|switch|' + ('p2' if active_side else 'p1'))
                                            for line in branch.state.lines))


if __name__ == '__main__':
    unittest.main()
