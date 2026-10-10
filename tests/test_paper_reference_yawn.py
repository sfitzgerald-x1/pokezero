"""Restore Yawn's PUBLIC fixed clock; native engine handles delayed sleep."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import (LocalShowdownConfig, LocalShowdownEnv,
    LocalShowdownError, _public_reference_yawn)
from pokezero.showdown_fixture import FixturePokemon, pack_team
from pokezero.mcts_eval.paper_reference_sleep import induced_sleep_certificates, induced_sleep_support

OPEN = ['|switch|p1a: Swalot|Swalot, M|300/300',
        '|switch|p2a: Butterfree|Butterfree, F|273/273', '|turn|1']
START = ['|move|p1a: Swalot|Yawn|p2a: Butterfree',
         '|-start|p2a: Butterfree|move: Yawn|[of] p1a: Swalot']
TAIL = ['|upkeep', '|turn|2']


def state(lines, *, present=True, force=False, pending=None):
    return SimpleNamespace(observation_format_id='gen3customgame',
        self_request={'forceSwitch': [True]} if force else {'active': [{}]},
        deferred_opponent_action_player=pending,
        replay=SimpleNamespace(volatiles={'p1': (), 'p2': ('yawn',) if present else ()},
            public_events=[SimpleNamespace(raw_line=line) for line in lines]))


class PublicYawnCertificateTests(unittest.TestCase):
    def test_fixed_countdown_and_disclosed_source(self):
        self.assertEqual(_public_reference_yawn(state(OPEN + START + TAIL), 'p2'),
            dict(sourceSide='p1', sourceIdent='p1a: Swalot', targetIdent='p2a: Butterfree',
                 sourceSpecies='Swalot', targetSpecies='Butterfree', start_turn=1,
                 residual_turns=[1], duration=1))

    def test_source_switch_or_faint_preserves_original_source(self):
        for event in ('|switch|p1a: Shiftry|Shiftry, M|300/300', '|faint|p1a: Swalot'):
            with self.subTest(event=event):
                c = _public_reference_yawn(state(OPEN + START + [event] + TAIL), 'p2')
                self.assertEqual((c['sourceSpecies'], c['duration']), ('Swalot', 1))

    def test_target_switch_drag_replace_faint_or_end_cannot_reuse_start(self):
        for event in ('|switch|p2a: Butterfree|Butterfree, F|273/273',
                      '|drag|p2a: Butterfree|Butterfree, F|273/273',
                      '|replace|p2a: Butterfree|Butterfree, F|273/273',
                      '|faint|p2a: Butterfree', '|-end|p2a: Butterfree|move: Yawn|[silent]'):
            with self.subTest(event=event), self.assertRaises(LocalShowdownError):
                _public_reference_yawn(state(OPEN + START + [event] + TAIL), 'p2')

    def test_unknown_mismatched_duplicate_or_failed_public_source_refuses(self):
        cases = [START + TAIL, OPEN + [START[1]] + TAIL,
                 OPEN + [START[0], START[1] + '|[of] p1a: Swalot'] + TAIL,
                 OPEN + [START[0], START[1].replace('p1a: Swalot', 'p1a: Shiftry')] + TAIL,
                 OPEN + START + [START[1]] + TAIL]
        for failure in ('|-fail|p1a: Swalot', '|-immune|p2a: Butterfree',
                        '|-miss|p1a: Swalot|p2a: Butterfree'):
            cases.append(OPEN + [START[0], failure, START[1]] + TAIL)
        for lines in cases:
            with self.subTest(lines=lines), self.assertRaises(LocalShowdownError):
                _public_reference_yawn(state(lines), 'p2')

    def test_public_clock_cannot_be_guessed(self):
        for tail in ([], ['|turn|2'], ['|upkeep'], ['|upkeep', '|upkeep', '|turn|2'],
                     TAIL + ['|upkeep', '|turn|3'], ['|upkeep', '|turn|4']):
            with self.subTest(tail=tail), self.assertRaises(LocalShowdownError):
                _public_reference_yawn(state(OPEN + START + tail), 'p2')
        truncated = [line.replace('|turn|1', '|turn|5') for line in OPEN] + START + ['|upkeep', '|turn|6']
        with self.assertRaisesRegex(LocalShowdownError, 'complete public opening turn'):
            _public_reference_yawn(state(truncated), 'p2')

    def test_reversed_source_target_seats(self):
        lines = [line.replace('p1', 'TMP').replace('p2', 'p1').replace('TMP', 'p2')
                 for line in OPEN + START + TAIL]
        root = state(lines, present=False)
        root.replay.volatiles = {'p1': ('yawn',), 'p2': ()}
        cert = _public_reference_yawn(root, 'p1')
        self.assertEqual((cert['sourceSide'], cert['targetIdent']), ('p2', 'p1a: Butterfree'))

    def test_midturn_and_non_gen3_refuse_only_when_present(self):
        for kwargs in ({'force': True}, {'pending': 'p1'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(LocalShowdownError):
                _public_reference_yawn(state(OPEN + START + TAIL, **kwargs), 'p2')
        invalid = state(OPEN + START + TAIL)
        invalid.observation_format_id = 'gen4randombattle'
        with self.assertRaises(LocalShowdownError):
            _public_reference_yawn(invalid, 'p2')
        self.assertIsNone(_public_reference_yawn(state([], present=False, force=True), 'p2'))

    def test_new_status_does_not_invent_or_remove_pending_yawn(self):
        c = _public_reference_yawn(state(OPEN + START + ['|-status|p2a: Butterfree|par'] + TAIL), 'p2')
        self.assertEqual(c['duration'], 1)

    def test_current_turn_resolution_cannot_pose_as_ordinary_frontier(self):
        for line in ('|move|p2a: Butterfree|Splash|p2a: Butterfree',
                     '|cant|p2a: Butterfree|slp', '|-status|p2a: Butterfree|par',
                     '|switch|p1a: Shiftry|Shiftry, M|300/300', '|-activate|p2a: Butterfree|move: Protect'):
            with self.subTest(line=line), self.assertRaises(LocalShowdownError):
                _public_reference_yawn(state(OPEN + START + TAIL + [line]), 'p2')

    def test_node_rebinds_source_in_bench_and_fainted_party_and_checks_clock(self):
        module = (Path(__file__).resolve().parents[1] / 'scripts/battle_bridge_reference_yawn.mjs').as_uri()
        script = """
        import {bindReferenceYawn as bind} from MODULE;
        const mon=(species,active,status='')=>({set:{species},isActive:active,status,
          hp:200,fainted:false,volatiles:{}});
        const fresh=()=>({midTurn:false,sides:[{id:'p1',pokemon:[mon('Shiftry',true),mon('Swalot',false)]},
          {id:'p2',pokemon:[mon('Scizor',false),mon('Butterfree',true)]}]});
        const pub={turn:2,selfRequestKind:'move',sides:{
          p1:{pokemon:[{species:'Swalot',active:false},{species:'Shiftry',active:true}],volatiles:[],referenceYawn:null},
          p2:{pokemon:[{species:'Butterfree',active:true}],volatiles:['yawn'],referenceYawn:{
            sourceSide:'p1',sourceIdent:'p1a: Swalot',sourceSpecies:'Swalot',
            targetIdent:'p2a: Butterfree',targetSpecies:'Butterfree',start_turn:1,residual_turns:[1],duration:1}}}};
        for(const dead of [false,true]){const s=fresh();if(dead){s.sides[0].pokemon[1].hp=0;s.sides[0].pokemon[1].fainted=true;}
          bind(s,pub,3);const v=s.sides[1].pokemon[1].volatiles.yawn;
          if(v.source!=='[Pokemon:p1b]'||v.target!=='[Pokemon:p2b]'||v.duration!==1)throw Error('ownership/clock drift');}
        const status=fresh();status.sides[1].pokemon[1].status='par';bind(status,pub,3);
        for(const mutate of [p=>p.turn=3,p=>p.sides.p2.referenceYawn.duration=2,
          p=>p.sides.p2.referenceYawn.residual_turns=[],p=>p.sides.p2.referenceYawn=null,
          p=>p.sides.p2.referenceYawn.sourceSpecies='Scizor',p=>p.sides.p2.volatiles=[],
          p=>p.selfRequestKind='force-switch']){let p=structuredClone(pub);mutate(p);let refused=false;
          try{bind(fresh(),p,3);}catch{refused=true;}if(!refused)throw Error('bad certificate admitted');}
        for(const mutate of [s=>s.sides[1].pokemon[1].hp=0,s=>s.sides[1].pokemon[1].fainted=true,
          s=>s.sides[1].pokemon[1].isActive=false,s=>s.sides[0].pokemon.push(mon('Swalot',false)),
          s=>s.midTurn=true]){const s=fresh();mutate(s);let refused=false;try{bind(s,pub,3);}catch{refused=true;}
          if(!refused)throw Error('bad target/source/frontier admitted');}
        const absent=structuredClone(pub);absent.sides.p2.volatiles=[];absent.sides.p2.referenceYawn=null;
        absent.selfRequestKind='force-switch';const s=fresh();s.midTurn=true;bind(s,absent,3);
        """.replace('MODULE', json.dumps(module))
        subprocess.run(['node', '--input-type=module', '-e', script], check=True, capture_output=True)


class YawnExpiryLedgerTests(unittest.TestCase):
    END = '|-end|p2a: Butterfree|move: Yawn|[silent]'
    SLEEP = '|-status|p2a: Butterfree|slp'

    def prefix(self):
        return OPEN + START + TAIL + ['|move|p2a: Butterfree|Splash|p2a: Butterfree',
                                      '|move|p1a: Swalot|Splash|p1a: Swalot']

    def test_delayed_source_and_initial_uniform_support(self):
        root = state(self.prefix() + [self.END, self.SLEEP, '|upkeep', '|turn|3'], present=False)
        cert = induced_sleep_certificates(root)['p2:butterfree']
        self.assertEqual((cert['source_player'], cert['source_name'], cert['attempts']), ('p1', 'Swalot', 0))
        self.assertEqual([draw['startTime'] for draw in induced_sleep_support(cert, 'Compound Eyes')], [2,3,4,5])

    def test_expiry_retains_source_after_switch_or_faint(self):
        for line in ('|switch|p1a: Shiftry|Shiftry, M|300/300', '|faint|p1a: Swalot'):
            with self.subTest(line=line):
                cert = induced_sleep_certificates(state(self.prefix() + [line, self.END, self.SLEEP], present=False))['p2:butterfree']
                self.assertEqual(cert['source_name'], 'Swalot')

    def test_invalid_expiry_receipts_cannot_lend_a_source(self):
        cases = [OPEN + TAIL + [self.END, self.SLEEP],
                 self.prefix() + [self.END.replace('|[silent]', ''), self.SLEEP],
                 self.prefix() + [self.END, '|-message|Sleep Clause Mod activated.', self.SLEEP],
                 self.prefix() + [self.END, '|faint|p2a: Butterfree', self.SLEEP],
                 self.prefix() + [self.END, self.SLEEP.replace('Butterfree', 'Scizor')],
                 self.prefix() + [self.END, self.SLEEP + '|[from] move: Yawn'],
                 self.prefix() + [self.END, self.SLEEP, self.SLEEP],
                 self.prefix() + ['|switch|p2a: Scizor|Scizor|300/300', self.END, self.SLEEP]]
        for lines in cases:
            with self.subTest(lines=lines), self.assertRaises(LocalShowdownError):
                induced_sleep_certificates(state(lines, present=False))

    def test_blocked_yawn_does_not_replace_later_explicit_sleep_source(self):
        lines = self.prefix() + [self.END, '|-message|Sleep Clause Mod activated.',
            '|move|p1a: Shiftry|Spore|p2a: Butterfree',
            self.SLEEP + '|[from] move: Spore']
        cert = induced_sleep_certificates(state(lines, present=False))['p2:butterfree']
        self.assertEqual(cert['source_name'], 'Shiftry')

    def test_cure_and_new_yawn_episode_binds_new_source_once(self):
        lines = self.prefix() + [self.END, self.SLEEP, '|upkeep', '|turn|3',
            '|-curestatus|p2a: Butterfree|slp', '|switch|p1a: Shiftry|Shiftry, M|300/300',
            '|move|p1a: Shiftry|Yawn|p2a: Butterfree',
            '|-start|p2a: Butterfree|move: Yawn|[of] p1a: Shiftry', '|upkeep', '|turn|4',
            self.END, self.SLEEP, '|upkeep', '|turn|5']
        cert = induced_sleep_certificates(state(lines, present=False))['p2:butterfree']
        self.assertEqual((cert['source_name'], cert['attempts']), ('Shiftry', 0))

    def test_sleep_talk_refund_survival_and_cure_remain_unchanged(self):
        lines = self.prefix() + [self.END, self.SLEEP, '|upkeep', '|turn|3',
            '|cant|p2a: Butterfree|slp', '|move|p2a: Butterfree|Sleep Talk|p2a: Butterfree',
            '|move|p2a: Butterfree|Splash|p2a: Butterfree|[from]move: Sleep Talk',
            '|upkeep', '|turn|4', '|switch|p2a: Scizor|Scizor|300/300', '|upkeep', '|turn|5',
            '|switch|p2a: Butterfree|Butterfree|200/273 slp', '|upkeep', '|turn|6']
        cert = induced_sleep_certificates(state(lines, present=False))['p2:butterfree']
        self.assertEqual((cert['attempts'], cert['refunded'], cert['skipped']), (1,1,0))
        self.assertEqual([draw['startTime'] for draw in induced_sleep_support(cert, 'Early Bird')], [3,4,5])
        self.assertEqual(induced_sleep_certificates(state(lines + ['|-curestatus|p2a: Butterfree|slp'], present=False)), {})


@requires_showdown()
class YawnServerTests(unittest.TestCase):
    def config(self):
        return LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)

    def override(self):
        return BattleStartOverride(player_teams={
            'p1': pack_team((FixturePokemon('Swalot', ('Yawn', 'Splash'), ability='Liquid Ooze'),
                             FixturePokemon('Shiftry', ('Splash',), ability='Chlorophyll'))),
            'p2': pack_team((FixturePokemon('Butterfree', ('Splash', 'Rest'), ability='Compound Eyes'),
                             FixturePokemon('Scizor', ('Splash',), ability='Swarm')))})

    def start(self, live, *, reversed_seats=False, rest=False):
        original = self.override()
        override = original
        if reversed_seats:
            override = BattleStartOverride(player_teams={'p1': original.player_teams['p2'], 'p2': original.player_teams['p1']})
        live.reset_with_start_override(seed=223, start_override=override)
        live.step({'p1': 1 if rest and reversed_seats else 0,
                   'p2': 1 if rest and not reversed_seats else 0})
        target = 'p1' if reversed_seats else 'p2'
        self.assertIn('yawn', live.public_materialization_state(target).replay.volatiles[target])
        return override, target

    def test_default_unchanged_and_opt_in_boolean(self):
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as branch:
            override, target = self.start(live)
            root = live.public_materialization_state(target)
            with self.assertRaisesRegex(LocalShowdownError, 'does not yet support volatile effect yawn'):
                branch.materialize_public_world(state=root, start_override=override, seed=19)
            with self.assertRaisesRegex(LocalShowdownError, 'opt-in must be boolean'):
                branch.materialize_public_world(state=root, start_override=override, seed=19, reference_yawn=1)

    def test_both_seats_permutations_and_no_constructor_rng_or_early_sleep(self):
        for reverse in (False, True):
            for actor in ('p1', 'p2'):
                with self.subTest(reverse=reverse, actor=actor), LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as branch:
                    override, target = self.start(live, reversed_seats=reverse)
                    sampled = BattleStartOverride(player_teams={side: ']'.join(team.split(']')[::-1])
                        for side, team in override.player_teams.items()})
                    branch.reset_with_start_override(seed=19, start_override=sampled)
                    before = branch.snapshot().bridge_snapshot['battle']['prng']
                    branch.materialize_public_world(state=live.public_materialization_state(actor),
                        start_override=sampled, seed=19, reference_turn_clocks=True, reference_yawn=True)
                    snapshot = branch.snapshot().bridge_snapshot['battle']
                    self.assertEqual(snapshot['prng'], before)
                    owner = next(s for s in snapshot['sides'] if s['id'] == target)
                    mon = next(p for p in owner['pokemon'] if p['isActive'])
                    self.assertEqual(mon['status'], '')
                    self.assertEqual(mon['volatiles']['yawn']['duration'], 1)
                    self.assertEqual(mon['volatiles']['yawn']['sourceEffect'],
                        {'hit': 0, 'move': '[Move:yawn]'})
                    source_side = 'p2' if target == 'p1' else 'p1'
                    source = next(s for s in snapshot['sides'] if s['id'] == source_side)
                    index = next(i for i, p in enumerate(source['pokemon']) if p['set']['species'] == 'Swalot')
                    self.assertEqual(mon['volatiles']['yawn']['source'], f'[Pokemon:{source_side}{"abcdef"[index]}]')

    def test_native_expiry_matches_live_sleep_status_source_and_rng(self):
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as branch:
            override, target = self.start(live)
            branch.materialize_public_world(state=live.public_materialization_state('p1'),
                start_override=override, seed=19, reference_turn_clocks=True, reference_yawn=True)
            for env in (live, branch):
                env.reseed_simulator_rng(1234)
                env.step({'p1': 1, 'p2': 0})
            a = live.snapshot().bridge_snapshot['battle']
            b = branch.snapshot().bridge_snapshot['battle']
            target_index = 1
            for key in ('status', 'statusState'):
                # Compare source, duration and PRNG, not unrelated live log metadata.
                if key == 'status':
                    self.assertEqual(a['sides'][target_index]['pokemon'][0][key], 'slp')
                    self.assertEqual(a['sides'][target_index]['pokemon'][0][key], b['sides'][target_index]['pokemon'][0][key])
                else:
                    for field in ('time', 'source', 'sourceSlot'):
                        self.assertEqual(a['sides'][target_index]['pokemon'][0][key].get(field),
                                         b['sides'][target_index]['pokemon'][0][key].get(field))
            self.assertEqual(a['prng'], b['prng'])
            self.assertNotIn('yawn', b['sides'][target_index]['pokemon'][0]['volatiles'])
            for env in (live, branch):
                cert = induced_sleep_certificates(env.public_materialization_state('p2'))['p2:butterfree']
                self.assertEqual((cert['source_player'], cert['source_name']), ('p1', 'Swalot'))
            # The next world must condition ordinary induced sleep, not refuse
            # because Yawn's public start happened on an earlier turn.
            root = live.public_materialization_state('p2')
            draw = induced_sleep_support(induced_sleep_certificates(root)['p2:butterfree'], 'Compound Eyes')[0]
            branch.materialize_public_world(state=root, start_override=override, seed=19,
                reference_rest_sleep=True, reference_induced_sleep={'p2:butterfree': draw},
                reference_turn_clocks=True, reference_yawn=True)
            status = branch.snapshot().bridge_snapshot['battle']['sides'][1]['pokemon'][0]['statusState']
            self.assertEqual(status['source'], '[Pokemon:p1a]')
            self.assertEqual(status['time'], draw['time'])

    def test_canonical_factory_bundle_sleep_clause_blocks_second_yawn_victim(self):
        override = BattleStartOverride(observation_format_id='gen3randombattle', player_teams={
            'p1': pack_team((FixturePokemon('Articuno', ('Splash',), ability='Pressure'),
                             FixturePokemon('Skarmory', ('Splash',), ability='Keen Eye'))),
            'p2': pack_team((FixturePokemon('Wailord', ('Spore', 'Yawn', 'Splash'), ability='Water Veil'),))})
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as branch:
            live.reset_with_start_override(seed=17, start_override=override)
            live.step({'p1': 0, 'p2': 0})
            live.step({'p1': 4, 'p2': 1})
            root = replace(live.public_materialization_state('p1'), format_id='gen3randombattle')
            cert = induced_sleep_certificates(root)['p1:articuno']
            draw = induced_sleep_support(cert, 'Pressure')[-1]
            branch.reset_with_start_override(seed=19, start_override=override)
            before = branch.snapshot().bridge_snapshot['battle']['prng']
            branch.materialize_public_world(state=root, start_override=override, seed=19,
                reference_rest_sleep=True, reference_consumed_items=True,
                reference_encore_durations={}, reference_induced_sleep={'p1:articuno': draw},
                reference_turn_clocks=True, reference_attract=True, reference_yawn=True)
            snapshot = branch.snapshot().bridge_snapshot['battle']
            self.assertEqual(snapshot['prng'], before)
            self.assertFalse(snapshot['debugMode'])
            self.assertIn('sleepclausemod', snapshot['field']['pseudoWeather'])
            branch.step({'p1': 0, 'p2': 2})
            mons = branch.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon']
            self.assertEqual(next(p for p in mons if p['set']['species'] == 'Skarmory')['status'], '')
            self.assertEqual(next(p for p in mons if p['set']['species'] == 'Articuno')['status'], 'slp')
            self.assertTrue(any('Sleep Clause Mod activated.' in line for line in branch.protocol_lines))

    def test_canonical_factory_bundle_rest_sleeper_does_not_block_yawn(self):
        override = BattleStartOverride(observation_format_id='gen3randombattle', player_teams={
            'p1': pack_team((FixturePokemon('Articuno', ('Rest', 'Splash'), ability='Pressure'),
                             FixturePokemon('Skarmory', ('Splash',), ability='Keen Eye'))),
            'p2': pack_team((FixturePokemon('Wailord', ('Water Gun', 'Yawn', 'Splash'), ability='Water Veil'),))})
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as branch:
            live.reset_with_start_override(seed=17, start_override=override)
            live.step({'p1': 1, 'p2': 0})
            live.step({'p1': 0, 'p2': 2})
            live.step({'p1': 4, 'p2': 1})
            root = replace(live.public_materialization_state('p1'), format_id='gen3randombattle')
            self.assertEqual(induced_sleep_certificates(root), {})
            branch.materialize_public_world(state=root, start_override=override, seed=19,
                reference_rest_sleep=True, reference_consumed_items=True,
                reference_encore_durations={}, reference_induced_sleep={},
                reference_turn_clocks=True, reference_attract=True, reference_yawn=True)
            branch.step({'p1': 0, 'p2': 2})
            mons = branch.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon']
            self.assertEqual(next(p for p in mons if p['set']['species'] == 'Skarmory')['status'], 'slp')
            self.assertEqual(next(p for p in mons if p['set']['species'] == 'Articuno')['status'], 'slp')
            self.assertFalse(any('Sleep Clause Mod activated.' in line for line in branch.protocol_lines))

    def test_source_switch_preserves_pending_delayed_sleep(self):
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as branch:
            override, target = self.start(live)
            branch.materialize_public_world(state=live.public_materialization_state('p2'),
                start_override=override, seed=19, reference_turn_clocks=True, reference_yawn=True)
            for env in (live, branch):
                env.reseed_simulator_rng(1234)
                env.step({'p1': 4, 'p2': 0})
                mon = env.snapshot().bridge_snapshot['battle']['sides'][1]['pokemon'][0]
                self.assertEqual(mon['status'], 'slp')
                self.assertEqual(mon['statusState']['source'], '[Pokemon:p1b]')

    def test_publicly_fainted_source_survives_binding_after_force_switch(self):
        override = BattleStartOverride(player_teams={
            'p1': pack_team((FixturePokemon('Shedinja', ('Yawn', 'Splash'), ability='Wonder Guard'),
                             FixturePokemon('Shiftry', ('Splash',), ability='Chlorophyll'))),
            'p2': pack_team((FixturePokemon('Slowpoke', ('Ember', 'Splash'), ability='Own Tempo'),
                             FixturePokemon('Scizor', ('Splash',), ability='Swarm')))})
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as branch:
            live.reset_with_start_override(seed=223, start_override=override)
            live.step({'p1': 0, 'p2': 0})
            self.assertEqual(live.requested_players(), ('p1',))
            live.step({'p1': 4})
            root = live.public_materialization_state('p1')
            self.assertEqual(_public_reference_yawn(root, 'p2')['sourceSpecies'], 'Shedinja')
            branch.materialize_public_world(state=root, start_override=override, seed=19,
                reference_turn_clocks=True, reference_yawn=True)
            snapshot = branch.snapshot().bridge_snapshot['battle']
            source = next(p for p in snapshot['sides'][0]['pokemon'] if p['set']['species'] == 'Shedinja')
            self.assertTrue(source['fainted'])
            for env in (live, branch):
                env.reseed_simulator_rng(1234)
                env.step({'p1': 0, 'p2': 1})
                mon = env.snapshot().bridge_snapshot['battle']['sides'][1]['pokemon'][0]
                self.assertEqual(mon['status'], 'slp')
                self.assertEqual(mon['statusState']['source'], '[Pokemon:p1b]')

    def test_target_switch_clears_without_sleep_or_delayed_yawn_on_return(self):
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as branch:
            override, target = self.start(live)
            branch.materialize_public_world(state=live.public_materialization_state('p1'),
                start_override=override, seed=19, reference_turn_clocks=True, reference_yawn=True)
            for env in (live, branch):
                env.step({'p1': 1, 'p2': 4})
                env.step({'p1': 1, 'p2': 4})
                mon = next(p for p in env.snapshot().bridge_snapshot['battle']['sides'][1]['pokemon']
                           if p['set']['species'] == 'Butterfree')
                self.assertEqual(mon['status'], '')
                self.assertNotIn('yawn', mon['volatiles'])


if __name__ == '__main__':
    unittest.main()
