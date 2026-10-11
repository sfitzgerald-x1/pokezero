"""Attract needs a public, surviving cross-Pokemon source, never a guessed one."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import (LocalShowdownConfig, LocalShowdownEnv,
    LocalShowdownError, _public_reference_attract)
from pokezero.showdown_fixture import FixturePokemon, pack_team


def public_state(lines, *, present=True):
    return SimpleNamespace(observation_format_id='gen3customgame', replay=SimpleNamespace(
        volatiles={'p1': (), 'p2': ('attract',) if present else ()},
        public_events=[SimpleNamespace(raw_line=line) for line in lines]))


OPENING = ['|switch|p1a: Clefable|Clefable, F|300/300',
           '|switch|p2a: Scizor|Scizor, M|249/249']
CUTE = '|-start|p2a: Scizor|Attract|[from] ability: Cute Charm|[of] p1a: Clefable'


class PublicAttractCertificateTests(unittest.TestCase):
    def test_cute_charm_uses_disclosed_source_not_hidden_state(self):
        cert = _public_reference_attract(public_state(OPENING + [CUTE]), 'p2')
        self.assertEqual(cert, dict(sourceSide='p1', sourceIdent='p1a: Clefable',
            targetIdent='p2a: Scizor', sourceSpecies='Clefable', targetSpecies='Scizor', cause='cutecharm'))

    def test_move_start_binds_exact_public_move_target(self):
        lines = OPENING + ['|move|p1a: Clefable|Attract|p2a: Scizor', '|-start|p2a: Scizor|Attract']
        self.assertEqual(_public_reference_attract(public_state(lines), 'p2')['cause'], 'attract')

    def test_reversed_source_and_target_public_seats(self):
        state = public_state([])
        state.replay.volatiles = {'p1': ('attract',), 'p2': ()}
        state.replay.public_events = [SimpleNamespace(raw_line=line) for line in [
            '|switch|p1a: Scizor|Scizor, M|249/249',
            '|switch|p2a: Clefable|Clefable, F|300/300',
            '|-start|p1a: Scizor|Attract|[from] ability: Cute Charm|[of] p2a: Clefable']]
        cert = _public_reference_attract(state, 'p1')
        self.assertEqual((cert['sourceSide'], cert['sourceIdent'], cert['targetIdent']),
                         ('p2', 'p2a: Clefable', 'p1a: Scizor'))

    def test_unknown_truncated_wrong_or_ambiguous_sources_refuse(self):
        for lines in ([CUTE], OPENING + ['|-start|p2a: Scizor|Attract'],
                      OPENING + [CUTE.replace('p1a: Clefable', 'p1a: Donphan')],
                      OPENING + [CUTE + '|[of] p1a: Clefable'],
                      OPENING + [CUTE + '|[from] item: Mystery'],
                      OPENING + ['|move|p1a: Clefable|Attract|p2a: Magmar', '|-start|p2a: Scizor|Attract']):
            with self.subTest(lines=lines), self.assertRaisesRegex(LocalShowdownError, 'surviving public source'):
                _public_reference_attract(public_state(lines), 'p2')

    def test_switching_out_and_back_cannot_reuse_an_old_relationship(self):
        for side, ident, details in [('p1', 'Clefable', 'Clefable, F'), ('p2', 'Scizor', 'Scizor, M')]:
            lines = OPENING + [CUTE, f'|switch|{side}a: Donphan|Donphan, M|300/300',
                               f'|switch|{side}a: {ident}|{details}|200/300']
            with self.subTest(side=side), self.assertRaisesRegex(LocalShowdownError, 'surviving public source'):
                _public_reference_attract(public_state(lines), 'p2')

    def test_public_end_invalidates_and_absent_effect_needs_no_certificate(self):
        with self.assertRaisesRegex(LocalShowdownError, 'surviving public source'):
            _public_reference_attract(public_state(OPENING + [CUTE, '|-end|p2a: Scizor|Attract|[silent]']), 'p2')
        self.assertIsNone(_public_reference_attract(public_state([CUTE], present=False), 'p2'))

    def test_unrelated_move_or_turn_cannot_lend_a_source(self):
        for middle in ['|turn|2', '|move|p2a: Scizor|Splash|p2a: Scizor']:
            lines = OPENING + ['|move|p1a: Clefable|Attract|p2a: Scizor', middle,
                               '|-start|p2a: Scizor|Attract']
            with self.subTest(middle=middle), self.assertRaises(LocalShowdownError):
                _public_reference_attract(public_state(lines), 'p2')

    def test_missed_failed_or_immune_moves_cannot_certify_a_later_start(self):
        move = '|move|p1a: Clefable|Attract|p2a: Scizor'
        for suffix in ([move + '|[miss]'],
                       [move, '|-miss|p1a: Clefable|p2a: Scizor'],
                       [move, '|-fail|p1a: Clefable'],
                       [move, '|-immune|p2a: Scizor']):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(LocalShowdownError, 'surviving public source'):
                _public_reference_attract(public_state(OPENING + suffix + ['|-start|p2a: Scizor|Attract']), 'p2')

    def test_fainted_source_or_target_cannot_certify_a_force_switch_frontier(self):
        for endpoint in ('p1a: Clefable', 'p2a: Scizor'):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(LocalShowdownError, 'surviving public source'):
                _public_reference_attract(public_state(OPENING + [CUTE, f'|faint|{endpoint}']), 'p2')

    def test_node_binds_final_party_indices_and_rejects_bad_certificates(self):
        module = (Path(__file__).resolve().parents[1] / 'scripts/battle_bridge_reference_attract.mjs').as_uri()
        script = """
          import {bindReferenceAttract as bind} from MODULE;
          const mon=(species,gender,active)=>({set:{species},gender,isActive:active,hp:200,fainted:false,volatiles:{}});
          const fresh=()=>({sides:[{id:'p1',pokemon:[mon('Donphan','M',false),mon('Clefable','F',true)]},
            {id:'p2',pokemon:[mon('Magmar','F',false),mon('Rhydon','M',false),mon('Scizor','M',true)]}]});
          const pub={sides:{p1:{pokemon:[{species:'Clefable',active:true}],volatiles:[],referenceAttract:null},
            p2:{pokemon:[{species:'Scizor',active:true}],volatiles:['attract'],referenceAttract:{
              sourceSide:'p1',sourceIdent:'p1a: Clefable',targetIdent:'p2a: Scizor',
              sourceSpecies:'Clefable',targetSpecies:'Scizor',cause:'cutecharm'}}}};
          const s=fresh();bind(s,pub,3);
          const v=s.sides[1].pokemon[2].volatiles.attract;
          if(v.source!=='[Pokemon:p1b]'||v.target!=='[Pokemon:p2c]')throw Error('party ownership drift');
          const capitalized=structuredClone(pub);capitalized.sides.p2.volatiles=['Attract'];
          const capitalizedState=fresh();bind(capitalizedState,capitalized,3);
          if(!capitalizedState.sides[1].pokemon[2].volatiles.attract)throw Error('normalized volatile dropped');
          if(s.sides[0].pokemon[0].volatiles.attract||s.sides[1].pokemon[0].volatiles.attract)throw Error('opening lead contaminated');
          for(const mutate of [p=>p.sides.p2.referenceAttract=null,
            p=>p.sides.p2.referenceAttract.sourceSide='p2',
            p=>p.sides.p2.referenceAttract.sourceSpecies='Donphan',
            p=>p.sides.p2.referenceAttract.targetSpecies='Rhydon',
            p=>p.sides.p2.volatiles=[],p=>p.sides.p2.referenceAttract.cause='unknown']){
            const p=structuredClone(pub);mutate(p);let refused=false;
            try{bind(fresh(),p,3);}catch{refused=true;}if(!refused)throw Error('bad provenance admitted');
          }
          const bad=fresh();bad.sides[1].pokemon[2].gender='F';let refused=false;
          try{bind(bad,pub,3);}catch{refused=true;}if(!refused)throw Error('same gender admitted');
          for(const [side,index] of [[0,1],[1,2]]){
            const fainted=fresh();fainted.sides[side].pokemon[index].hp=0;
            let refused=false;try{bind(fainted,pub,3);}catch{refused=true;}
            if(!refused)throw Error('zero HP endpoint admitted');
            const flagged=fresh();flagged.sides[side].pokemon[index].fainted=true;
            refused=false;try{bind(flagged,pub,3);}catch{refused=true;}
            if(!refused)throw Error('fainted endpoint admitted');
          }
          refused=false;try{bind(fresh(),pub,4);}catch{refused=true;}if(!refused)throw Error('wrong generation admitted');
        """.replace('MODULE', json.dumps(module))
        subprocess.run(['node', '--input-type=module', '-e', script], check=True, capture_output=True)


@requires_showdown()
class AttractServerTests(unittest.TestCase):
    def config(self):
        return LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)

    def override(self, *, cute=False):
        return BattleStartOverride(player_teams={
            'p1': pack_team((FixturePokemon('Clefable', ('Attract', 'Splash'), ability='Cute Charm', gender='F'),
                             FixturePokemon('Donphan', ('Splash',), ability='Sturdy', gender='M'))),
            'p2': pack_team((FixturePokemon('Scizor', ('Splash', 'Tackle'), ability='Swarm', gender='M'),
                             FixturePokemon('Magcargo', ('Splash',), ability='Flame Body', gender='F')))})

    def start_attracted(self, live, *, cute=False):
        override = self.override(cute=cute)
        if not cute:
            live.reset_with_start_override(seed=223, start_override=override)
            live.step({'p1': 0, 'p2': 0})
        else:
            for seed in range(1, 40):
                live.reset_with_start_override(seed=seed, start_override=override)
                live.step({'p1': 1, 'p2': 1})
                if 'attract' in live.public_materialization_state('p1').replay.volatiles['p2']:
                    break
            else:
                self.fail('fixture never triggered public Cute Charm')
        self.assertIn('attract', live.public_materialization_state('p1').replay.volatiles['p2'])
        return override

    def test_move_and_cute_charm_restore_for_both_actor_seats(self):
        for cute in (False, True):
            for actor in ('p1', 'p2'):
                with self.subTest(cute=cute, actor=actor), LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
                    override = self.start_attracted(live, cute=cute)
                    state = live.public_materialization_state(actor)
                    restored.materialize_public_world(state=state, start_override=override, seed=19,
                        reference_turn_clocks=True, reference_attract=True)
                    snapshot = restored.snapshot().bridge_snapshot['battle']
                    self.assertEqual(snapshot['sides'][1]['pokemon'][0]['volatiles']['attract']['source'], '[Pokemon:p1a]')
                    self.assertEqual(_public_reference_attract(state, 'p2')['cause'], 'cutecharm' if cute else 'attract')

    def test_default_path_still_refuses_and_bad_opt_in_refuses(self):
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
            override = self.start_attracted(live)
            state = live.public_materialization_state('p1')
            with self.assertRaisesRegex(LocalShowdownError, 'does not yet support volatile effect attract'):
                restored.materialize_public_world(state=state, start_override=override, seed=19)
            with self.assertRaisesRegex(LocalShowdownError, 'opt-in must be boolean'):
                restored.materialize_public_world(state=state, start_override=override, seed=19, reference_attract=1)

    def test_materialization_does_not_advance_sampled_shell_rng(self):
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
            override = self.start_attracted(live)
            state = live.public_materialization_state('p1')
            restored.reset_with_start_override(seed=19, start_override=override)
            before = restored.snapshot().bridge_snapshot['battle']['prng']
            restored.materialize_public_world(state=state, start_override=override, seed=19,
                reference_turn_clocks=True, reference_attract=True)
            after = restored.snapshot().bridge_snapshot['battle']['prng']
            self.assertEqual(after, before)

    def test_factory_flag_bundle_preserves_rng_and_canonical_debug_mode(self):
        for actor in ('p1', 'p2'):
            with self.subTest(actor=actor), LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
                override = replace(self.override(), observation_format_id='gen3randombattle')
                live.reset_with_start_override(seed=223, start_override=override)
                live.step({'p1': 0, 'p2': 0})
                state = replace(live.public_materialization_state(actor), format_id='gen3randombattle')
                restored.reset_with_start_override(seed=19, start_override=override)
                shell = restored.snapshot().bridge_snapshot['battle']
                self.assertTrue(shell['debugMode'])
                restored.materialize_public_world(state=state, start_override=override, seed=19,
                    reference_rest_sleep=True, reference_consumed_items=True,
                    reference_encore_durations={}, reference_induced_sleep={},
                    reference_turn_clocks=True, reference_attract=True)
                after = restored.snapshot().bridge_snapshot['battle']
                self.assertEqual(after['prng'], shell['prng'])
                self.assertFalse(after['debugMode'])
                self.assertEqual(after['formatid'], 'gen3randombattle')
                self.assertIn('sleepclausemod', after['field']['pseudoWeather'])

    def test_reversed_source_and_target_restore_with_permuted_sampled_leads(self):
        original = self.override()
        reversed_seats = BattleStartOverride(player_teams={
            'p1': original.player_teams['p2'], 'p2': original.player_teams['p1']})
        for actor in ('p1', 'p2'):
            with self.subTest(actor=actor), LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
                live.reset_with_start_override(seed=223, start_override=reversed_seats)
                live.step({'p1': 0, 'p2': 0})
                state = live.public_materialization_state(actor)
                self.assertIn('attract', state.replay.volatiles['p1'])
                # The sampled opening lead is deliberately not either public active.
                sampled = BattleStartOverride(player_teams={
                    side: ']'.join(reversed_seats.player_teams[side].split(']')[::-1]) for side in ('p1', 'p2')})
                restored.materialize_public_world(state=state, start_override=sampled, seed=19,
                    reference_turn_clocks=True, reference_attract=True)
                snapshot = restored.snapshot().bridge_snapshot['battle']
                target = next(row for row in snapshot['sides'][0]['pokemon'] if row['isActive'])
                source_index = next(index for index, row in enumerate(snapshot['sides'][1]['pokemon']) if row['isActive'])
                self.assertEqual(target['volatiles']['attract']['source'], f'[Pokemon:p2{"abcdef"[source_index]}]')

    def test_seeded_continuations_match_live_attract_chance_and_source_logs(self):
        outcomes = set()
        for seed in range(700, 708):
            with self.subTest(seed=seed), LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
                override = self.start_attracted(live)
                restored.materialize_public_world(state=live.public_materialization_state('p1'),
                    start_override=override, seed=19, reference_turn_clocks=True, reference_attract=True)
                for env in (live, restored):
                    env.reseed_simulator_rng(seed)
                    env.step({'p1': 1, 'p2': 1})
                def relevant(env):
                    return tuple(line for line in env._lines if line.startswith(('|-activate|', '|cant|', '|-damage|')))
                self.assertEqual(relevant(live), relevant(restored))
                outcomes.add(any('|cant|p2a: Scizor|Attract' in line for line in relevant(restored)))
        self.assertEqual(outcomes, {False, True})

    def test_source_and_target_switches_end_restored_attract(self):
        for switcher in ('p1', 'p2'):
            with self.subTest(switcher=switcher), LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
                override = self.start_attracted(live)
                restored.materialize_public_world(state=live.public_materialization_state('p1'),
                    start_override=override, seed=19, reference_turn_clocks=True, reference_attract=True)
                for env in (live, restored):
                    env.reseed_simulator_rng(701)
                    env.step({'p1': 4 if switcher == 'p1' else 1, 'p2': 4 if switcher == 'p2' else 1})
                    self.assertNotIn('attract', env.public_materialization_state('p1').replay.volatiles['p2'])
