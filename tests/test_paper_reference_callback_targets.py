"""Effect callbacks must follow their owner after public party reconstruction."""
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.showdown_fixture import FixturePokemon, pack_team


@requires_showdown()
class ReferenceCallbackTargetTests(unittest.TestCase):
    def test_switched_active_and_bench_callbacks_and_next_leftovers_for_both_seats(self):
        config = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        override = BattleStartOverride(player_teams={
            'p1': pack_team((FixturePokemon('Shuckle', ('Splash',), ability='Sturdy'),
                FixturePokemon('Lugia', ('Substitute', 'Splash'), ability='Pressure', item='Leftovers'))),
            'p2': pack_team((FixturePokemon('Magcargo', ('Splash',), ability='Flame Body'),
                FixturePokemon('Clefable', ('Substitute', 'Splash'), ability='Cute Charm', item='Leftovers'))),
        })
        with LocalShowdownEnv(config) as live:
            live.reset_with_start_override(seed=223, start_override=override)
            live.step({'p1': 4, 'p2': 4})
            live.step({'p1': 0, 'p2': 0})
            public = {actor: live.public_materialization_state(actor) for actor in ('p1', 'p2')}
            live.reseed_simulator_rng(418)
            live.step({'p1': 1, 'p2': 1})
            expected = live.snapshot().bridge_snapshot['battle']
            for actor in ('p1', 'p2'):
                with self.subTest(actor=actor), LocalShowdownEnv(config) as sampled:
                    sampled.materialize_public_world(state=public[actor], start_override=override,
                        seed=19, reference_turn_clocks=True)
                    rebuilt = sampled.snapshot().bridge_snapshot['battle']
                    for side in rebuilt['sides']:
                        for index, pokemon in enumerate(side['pokemon']):
                            owner = f"[Pokemon:{side['id']}{'abcdef'[index]}]"
                            for field in ('itemState', 'abilityState'):
                                self.assertEqual(pokemon[field]['target'], owner, (actor, field, index))
                            for volatile in pokemon['volatiles'].values():
                                self.assertEqual(volatile['target'], owner)
                    sampled.reseed_simulator_rng(418)
                    sampled.step({'p1': 1, 'p2': 1})
                    actual = sampled.snapshot().bridge_snapshot['battle']
                    for i in (0, 1):
                        self.assertEqual(actual['sides'][i]['pokemon'][0]['hp'],
                            expected['sides'][i]['pokemon'][0]['hp'])
                    for player in ('p1', 'p2'):
                        self.assertEqual(sampled.observe(player).legal_action_mask,
                            live.observe(player).legal_action_mask)


if __name__ == '__main__':
    unittest.main()
