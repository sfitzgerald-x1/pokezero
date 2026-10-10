"""Borrowed abilities must not become original-set facts on activation."""
import random
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.belief import PublicBattleBeliefEngine
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.mcts_eval.followthrough import continuation_seed
from pokezero.mcts_eval.paper_reference import Evaluation
from pokezero.mcts_eval.paper_reference_factory import PublicRootWorldFactory
from pokezero.mcts_eval.paper_reference_showdown import decision_state
from pokezero.randbat import load_gen3_randbat_source_cached
from pokezero.showdown import parse_showdown_replay


OPENING = [
    '|start',
    '|switch|p1a: Flareon|Flareon, L84, M|100/100',
    '|switch|p2a: Porygon2|Porygon2, L80|100/100',
]
TRACE = '|-ability|p2a: Porygon2|Flash Fire|Trace|[from] ability: Trace|[of] p1a: Flareon'
ACTIVATION = '|-start|p2a: Porygon2|ability: Flash Fire'


def engine(lines):
    return PublicBattleBeliefEngine.from_events(parse_showdown_replay(
        lines, battle_id='battle-gen3randombattle-trace-activation').public_events)


def mon(state, side, species):
    return next(row for row in state.snapshot().side(side) if row.species == species)


class BorrowedAbilityActivationTests(unittest.TestCase):
    def test_trace_activation_preserves_original_set_and_running_copy(self):
        state = engine([*OPENING, TRACE, ACTIVATION])
        holder = mon(state, 'p2', 'Porygon2')
        self.assertIsNone(holder.revealed_ability)
        self.assertEqual(state._running_ability[holder.key], 'Flash Fire')
        self.assertEqual(mon(state, 'p1', 'Flareon').revealed_ability, 'Flash Fire')

    def test_confirmed_original_trace_is_not_flagged_as_conflicting_on_activation(self):
        state = engine([*OPENING, '|-ability|p2a: Porygon2|Trace', TRACE, ACTIVATION])
        holder = mon(state, 'p2', 'Porygon2')
        self.assertEqual(holder.revealed_ability, 'Trace')
        self.assertFalse(any(row.kind == 'conflicting-ability-evidence' for row in holder.evidence))

    def test_unborrowed_activation_still_confirms_native_ability(self):
        state = engine([*OPENING, '|-start|p1a: Flareon|ability: Flash Fire'])
        self.assertEqual(mon(state, 'p1', 'Flareon').revealed_ability, 'Flash Fire')

    def test_transform_activation_does_not_reveal_dittos_original_ability(self):
        state = engine([
            '|start', '|switch|p1a: Flareon|Flareon, L84, M|100/100',
            '|switch|p2a: Ditto|Ditto, L83|100/100',
            '|-transform|p2a: Ditto|p1a: Flareon',
            '|-start|p2a: Ditto|ability: Flash Fire',
        ])
        self.assertTrue(mon(state, 'p2', 'Ditto').transformed)
        self.assertIsNone(mon(state, 'p2', 'Ditto').revealed_ability)

    def test_a_new_switch_in_copy_does_not_preserve_the_old_activation_as_original(self):
        state = engine([*OPENING, TRACE, ACTIVATION,
            '|switch|p2a: Corsola|Corsola, L98, F|100/100',
            '|switch|p1a: Vaporeon|Vaporeon, L82, F|100/100',
            '|switch|p2a: Porygon2|Porygon2, L80|100/100',
            '|-ability|p2a: Porygon2|Water Absorb|Trace|[from] ability: Trace|[of] p1a: Vaporeon',
            '|-heal|p2a: Porygon2|100/100|[from] ability: Water Absorb',
        ])
        holder = mon(state, 'p2', 'Porygon2')
        self.assertIsNone(holder.revealed_ability)
        self.assertEqual(state._running_ability[holder.key], 'Water Absorb')


@requires_showdown()
class LiveRefusedRootTests(unittest.TestCase):
    def test_exact_refused_prefix_materializes_trace_and_flash_fire_boost(self):
        config = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        with LocalShowdownEnv(config) as live, LocalShowdownEnv(config) as sampled:
            seed = 3906631832
            live.reset(seed=seed)
            # All four accepted boundaries of the preserved R2 refusal. No new
            # action, position, private opponent request, or redraw is introduced.
            actions = [dict(p1=0, p2=7), dict(p1=7, p2=2),
                       dict(p1=1, p2=5), dict(p1=0, p2=1)]
            for boundary, row in enumerate(actions):
                live.reseed_simulator_rng(continuation_seed(
                    f'wider-search:{seed}:p1', 0, boundary, 'chance'))
                live.step(row)
            observation = live.observe('p1')
            public = live.public_materialization_state('p1')
            self.assertEqual(public.replay.traced_ability['p2'], 'flashfire')
            self.assertIn('flashfire', public.replay.volatiles['p2'])
            self.assertIsNone(mon(public.belief_engine, 'p2', 'Porygon2').revealed_ability)
            def evaluate(obs):
                legal = tuple(i for i, allowed in enumerate(obs.legal_action_mask) if allowed)
                return legal, Evaluation((1 / len(legal),) * len(legal), 0.)
            factory = PublicRootWorldFactory(env=sampled, state=public, observation=observation,
                evaluator=evaluate, set_source=load_gen3_randbat_source_cached(showdown_root()),
                allow_earlier_compatible_template=True, max_known_set_draws=128)
            world = factory(random.Random(1))
            try:
                self.assertEqual(world.frame().subject, decision_state(observation, player='p1'))
                reconstructed = sampled._parser.snapshot()
                self.assertEqual(reconstructed.traced_ability['p2'], 'flashfire')
                self.assertIn('flashfire', reconstructed.volatiles['p2'])
            finally:
                world.close()
            self.assertEqual(factory.receipts[0]['status'], 'ROOT_VALIDATED')
            self.assertTrue(factory.receipts[0]['released'])


if __name__ == '__main__':
    unittest.main()
