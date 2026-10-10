"""A sampled opening lead must not lend stale traps to the public active."""
import json
from pathlib import Path
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError
from pokezero.showdown_fixture import FixturePokemon, pack_team


@requires_showdown()
class ReferenceTrappingTests(unittest.TestCase):
    def override(self, *, wobb_first=True, steel=False, trapper='Wobbuffet'):
        actors = (FixturePokemon('Skarmory' if steel else 'Tangela', ('Protect',),
                                  ability='Keen Eye' if steel else 'Chlorophyll'),
                  FixturePokemon('Donphan', ('Protect',), ability='Sturdy'))
        ability = {'Wobbuffet': 'Shadow Tag', 'Diglett': 'Arena Trap',
                   'Magneton': 'Magnet Pull'}[trapper]
        opponent = (FixturePokemon(trapper, ('Protect',), ability=ability),
                    FixturePokemon('Magcargo', ('Protect',), ability='Flame Body'))
        if not wobb_first:
            opponent = opponent[::-1]
        return BattleStartOverride(player_teams={'p1': pack_team(actors), 'p2': pack_team(opponent)})

    def config(self):
        return LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)

    def test_benched_shadow_tag_does_not_keep_opening_template_trap(self):
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
            live.reset_with_start_override(seed=223, start_override=self.override())
            live.step({'p1': 0, 'p2': 4})
            state = live.public_materialization_state('p1')
            self.assertNotIn('trapped', state.self_request['active'][0])
            restored.materialize_public_world(state=state, start_override=self.override(),
                                              seed=19, reference_turn_clocks=True)
            self.assertTrue(restored.observe('p1').legal_action_mask[4])
            # The same public root with an already-active Magcargo template
            # yields identical independently generated actor legality.
            first = restored._latest_requests['p1']['active']
            restored.materialize_public_world(state=state, start_override=self.override(wobb_first=False),
                                              seed=19, reference_turn_clocks=True)
            self.assertEqual(restored._latest_requests['p1']['active'], first)

    def test_current_shadow_tag_is_rederived_not_cleared(self):
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
            live.reset_with_start_override(seed=223, start_override=self.override(wobb_first=False))
            live.step({'p1': 0, 'p2': 4})
            state = live.public_materialization_state('p1')
            self.assertTrue(state.self_request['active'][0]['trapped'])
            restored.materialize_public_world(state=state, start_override=self.override(wobb_first=False),
                                              seed=19, reference_turn_clocks=True)
            self.assertFalse(restored.observe('p1').legal_action_mask[4])

    def test_current_hidden_ability_traps_and_request_validation_survive(self):
        for trapper, steel in [('Diglett', False), ('Magneton', True)]:
            with self.subTest(trapper=trapper):
                override = self.override(trapper=trapper, steel=steel)
                with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
                    live.reset_with_start_override(seed=223, start_override=override)
                    state = live.public_materialization_state('p1')
                    self.assertTrue(state.self_request['active'][0]['maybeTrapped'])
                    restored.materialize_public_world(state=state, start_override=override,
                                                      seed=19, reference_turn_clocks=True)
                    self.assertTrue(restored._latest_requests['p1']['active'][0]['maybeTrapped'])

    def test_bad_sampled_active_cannot_be_hidden_by_retained_legal_request(self):
        from dataclasses import replace
        with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
            live.reset_with_start_override(seed=223, start_override=self.override())
            state = live.public_materialization_state('p1')
            request = json.loads(json.dumps(state.self_request))
            request['active'][0].pop('trapped')
            with self.assertRaisesRegex(LocalShowdownError, 'generated actor boundary'):
                restored.materialize_public_world(state=replace(state, self_request=request),
                    start_override=self.override(), seed=19, reference_turn_clocks=True)

    def test_public_ingrain_and_recharge_locks_survive_cache_refresh(self):
        for move, ability, species in [('Ingrain', 'Chlorophyll', 'Tangela'),
                                       ('Hyper Beam', 'Truant', 'Slaking')]:
            with self.subTest(move=move):
                override = BattleStartOverride(player_teams={
                    'p1': pack_team((FixturePokemon(species, (move, 'Protect'), ability=ability),
                                     FixturePokemon('Donphan', ('Protect',), ability='Sturdy'))),
                    'p2': pack_team((FixturePokemon('Shuckle', ('Splash',), ability='Sturdy'),))})
                with LocalShowdownEnv(self.config()) as live, LocalShowdownEnv(self.config()) as restored:
                    live.reset_with_start_override(seed=223, start_override=override)
                    live.step({'p1': 0, 'p2': 0})
                    state = live.public_materialization_state('p1')
                    self.assertTrue(state.self_request['active'][0]['trapped'])
                    restored.materialize_public_world(state=state, start_override=override,
                                                      seed=19, reference_turn_clocks=True)
                    self.assertFalse(restored.observe('p1').legal_action_mask[4])

    def test_refresh_is_not_a_turn_and_does_not_advance_rng(self):
        module = (Path(__file__).resolve().parents[1] /
                  'scripts/battle_bridge_reference_trapping.mjs').as_uri()
        simulator = str(showdown_root() / 'dist/sim/index.js')
        script = """
            import {createRequire} from 'node:module';
            import {refreshReferenceTrapping as refresh} from MODULE;
            const {Battle,Teams}=createRequire(import.meta.url)(SIMULATOR);
            const battle=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
            battle.setPlayer('p1',{team:Teams.pack([
                {species:'Tangela',ability:'Chlorophyll',moves:['protect']},
                {species:'Donphan',ability:'Sturdy',moves:['protect']}])});
            battle.setPlayer('p2',{team:Teams.pack([
                {species:'Magcargo',ability:'Flame Body',moves:['protect']},
                {species:'Wobbuffet',ability:'Shadow Tag',moves:['protect']}])});
            const signature=()=>JSON.stringify({seed:battle.prng.getSeed(),turn:battle.turn,
                mons:battle.sides.flatMap(s=>s.pokemon.map(p=>({hp:p.hp,pp:p.moveSlots,
                    status:p.statusState,volatiles:p.volatiles,activeTurns:p.activeTurns})))},
                (k,v)=>['target','source','linkedPokemon'].includes(k)?undefined:v);
            const before=signature();
            battle.p1.active[0].trapped=true;
            battle.p1.active[0].maybeTrapped=true;
            refresh(battle,{selfPlayer:'p1',selfActiveRequestState:{}});
            if(battle.p1.active[0].trapped || battle.p1.active[0].maybeTrapped) throw Error('stale trap');
            if(signature()!==before) throw Error('turn or chance state changed');
            const original=battle.runEvent.bind(battle);
            battle.runEvent=(...args)=>{battle.random();return original(...args);};
            let refused=false;try {refresh(battle,{selfPlayer:'p1'});} catch(e) {
                refused=e.message.includes('consumed chance RNG');
            }
            if(!refused) throw Error('chance-consuming refresh accepted');
        """.replace('MODULE', json.dumps(module)).replace('SIMULATOR', json.dumps(simulator))
        subprocess.run(['node', '--input-type=module', '-e', script], check=True, capture_output=True)
