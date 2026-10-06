"""Recharge/Truant public clocks must survive actual sampled-world execution."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError, _validate_reference_actor_request
from pokezero.showdown_fixture import FixturePokemon, pack_team


class ActorBoundaryTests(unittest.TestCase):
    def test_retained_pseudo_move_cannot_hide_generated_ordinary_move(self):
        expected = {'active': [{'moves': [{'id': 'recharge'}], 'trapped': True}]}
        _validate_reference_actor_request(expected, expected)
        with self.assertRaisesRegex(LocalShowdownError, 'generated actor boundary'):
            _validate_reference_actor_request({'active': [{'moves': [{'id': 'earthquake', 'pp': 10, 'maxpp': 16}]}]}, expected)

    def test_disabled_and_pp_drift_refuse_but_irrelevant_request_metadata_does_not(self):
        expected = {'active': [{'moves': [{'id': 'hyperbeam', 'pp': 7, 'maxpp': 8, 'disabled': False}]}]}
        _validate_reference_actor_request(dict(expected, rqid=14), dict(expected, rqid=15))
        for field, value in [('disabled', True), ('pp', 8), ('maxpp', 9), ('id', 'earthquake')]:
            actual = json.loads(json.dumps(expected))
            actual['active'][0]['moves'][0][field] = value
            with self.assertRaisesRegex(LocalShowdownError, 'generated actor boundary'):
                _validate_reference_actor_request(actual, expected)

    def test_node_public_clock_guard_and_phase_assignment(self):
        module = (Path(__file__).resolve().parents[1]/'scripts/battle_bridge_reference_turn_clocks.mjs').as_uri()
        script = """
            import {applyReferenceTurnClocks as apply} from MODULE;
            const mon={ability:'Truant', fainted:false, volatiles:{}};
            apply(mon,{mustRecharge:true,truantPhase:true},'p1',3);
            if(mon.truantTurn!==true || mon.volatiles.mustrecharge.duration!==1) throw Error('clock drift');
            apply(mon,{mustRecharge:false,truantPhase:false},'p1',3);
            if(mon.truantTurn!==false) throw Error('phase not assigned');
            for(const row of [{mustRecharge:1,truantPhase:true},{mustRecharge:false,truantPhase:null}]) {
                let refused=false;try {apply(mon,row,'p1',3);} catch {refused=true;}
                if(!refused) throw Error('missing provenance accepted');
            }
        """.replace('MODULE', json.dumps(module))
        subprocess.run(['node', '--input-type=module', '-e', script], check=True, capture_output=True)


@requires_showdown()
class TurnClockServerTests(unittest.TestCase):
    def override(self, item='', target_ability='Sturdy'):
        return BattleStartOverride(player_teams={
            'p1': pack_team((FixturePokemon('Slaking', ('Earthquake', 'Hyper Beam'), ability='Truant', item=item),)),
            'p2': pack_team((FixturePokemon('Shuckle', ('Splash', 'Rest'), ability=target_ability),))})

    def row(self, env):
        return env.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]

    def test_recharge_and_truant_are_rebuilt_for_both_subject_seats(self):
        cfg = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        override = self.override()
        with LocalShowdownEnv(cfg) as live, LocalShowdownEnv(cfg) as restored:
            live.reset_with_start_override(seed=223, start_override=override)
            live.step({'p1': 1, 'p2': 0})
            self.assertTrue(live.public_materialization_state('p1').replay.must_recharge['p1'])
            for actor in ('p1', 'p2'):
                state = live.public_materialization_state(actor)
                restored.materialize_public_world(state=state, start_override=override, seed=19,
                    reference_turn_clocks=True)
                before = self.row(restored)
                self.assertEqual(before['volatiles']['mustrecharge']['duration'], 1)
                self.assertEqual(before['truantTurn'], self.row(live)['truantTurn'])
                restored.step({'p1': 0, 'p2': 0})
                self.assertNotIn('mustrecharge', self.row(restored)['volatiles'])
                self.assertTrue(any('|cant|p1a: Slaking|recharge' in line for line in restored._lines))
                self.assertNotEqual(self.row(restored)['truantTurn'], before['truantTurn'])

    def test_unknown_truant_phase_refuses_instead_of_guessing(self):
        cfg = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        override = self.override()
        with LocalShowdownEnv(cfg) as live, LocalShowdownEnv(cfg) as restored:
            live.reset_with_start_override(seed=223, start_override=override)
            state = live.public_materialization_state('p1')
            unknown = replace(state, replay=replace(state.replay, truant_phase={'p1': None, 'p2': None}))
            with self.assertRaisesRegex(LocalShowdownError, 'Truant.*publicly identifiable'):
                restored.materialize_public_world(state=unknown, start_override=override, seed=19,
                    reference_turn_clocks=True)

    def test_choice_band_lock_survives_recharge_and_continuation(self):
        cfg = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        override = self.override('Choice Band')
        with LocalShowdownEnv(cfg) as live, LocalShowdownEnv(cfg) as restored:
            live.reset_with_start_override(seed=223, start_override=override)
            live.step({'p1': 1, 'p2': 0})
            state = live.public_materialization_state('p1')
            restored.materialize_public_world(state=state, start_override=override, seed=19,
                reference_turn_clocks=True)
            self.assertEqual(self.row(restored)['volatiles']['choicelock']['move'], 'hyperbeam')
            for env in (live, restored):
                env.reseed_simulator_rng(701)
                env.step({'p1': 0, 'p2': 0})
            # It is no longer a Recharge request. Only the locked Hyper Beam
            # remains available; Earthquake must not become legal in the tree.
            _validate_reference_actor_request(restored._latest_requests['p1'], live._latest_requests['p1'])
            self.assertFalse(restored.observe('p1').legal_action_mask[0])
            self.assertTrue(restored.observe('p1').legal_action_mask[1])

    def test_pressure_charge_is_conditioned_on_sampled_target_not_true_target(self):
        cfg = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        with LocalShowdownEnv(cfg) as live, LocalShowdownEnv(cfg) as restored:
            live.reset_with_start_override(seed=223, start_override=self.override('Choice Band', 'Pressure'))
            live.step({'p1': 1, 'p2': 0})
            state = live.public_materialization_state('p1')
            # Gen 3 Pressure is privately disclosed to its holder. The actor
            # certificate contains no target ability or oracle PP measurement.
            self.assertEqual(state.self_recharge_pp_charge,
                {'move': 'hyperbeam', 'targetSide': 'p2', 'targetSpecies': 'Shuckle'})
            self.assertEqual(state.self_move_states['slaking'][1]['pp'], 8)
            for ability, expected in [('Pressure', 6), ('Sturdy', 7), ('Pressure', 6)]:
                restored.materialize_public_world(state=state,
                    start_override=self.override('Choice Band', ability), seed=19,
                    reference_turn_clocks=True)
                slot = self.row(restored)['moveSlots'][1]
                self.assertEqual(slot['pp'], expected)
                restored.step({'p1': 0, 'p2': 0})
                self.assertEqual(restored._latest_requests['p1']['active'][0]['moves'][1]['pp'], expected)
            # Repeated captures/materializations never spend the retained bank.
            self.assertEqual(live.public_materialization_state('p1').self_recharge_pp_charge,
                             state.self_recharge_pp_charge)
            self.assertEqual(state.self_move_states['slaking'][1]['pp'], 8)

    def test_unanchored_recharge_refuses_and_normal_request_supersedes_charge(self):
        cfg = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        override = self.override('Choice Band')
        with LocalShowdownEnv(cfg) as live, LocalShowdownEnv(cfg) as restored:
            live.reset_with_start_override(seed=223, start_override=override)
            live.step({'p1': 1, 'p2': 0})
            state = live.public_materialization_state('p1')
            with self.assertRaisesRegex(LocalShowdownError, 'certified actor PP charge'):
                restored.materialize_public_world(state=replace(state, self_recharge_pp_charge={}),
                    start_override=override, seed=19, reference_turn_clocks=True)
            live.step({'p1': 0, 'p2': 0})
            after = live.public_materialization_state('p1')
            self.assertEqual(after.self_recharge_pp_charge, {})
            restored.materialize_public_world(state=after, start_override=override,
                seed=19, reference_turn_clocks=True)
            self.assertEqual(self.row(restored)['moveSlots'][1]['pp'], 7)
