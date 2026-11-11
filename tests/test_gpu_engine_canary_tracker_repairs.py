"""Saved-parity causes isolated in finite device-write fixtures, without games."""
from types import SimpleNamespace as NS
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import effects, entry, items, transition
from gpu_engine import engine_device_public_hooks as hooks
from gpu_engine.state import empty_state
import test_gpu_engine_device_negative_items as absence


class CanaryTrackerRepairTests(unittest.TestCase):
    def test_generic_ground_immunity_does_not_publish_levitate(self):
        metadata = {'vocabulary': dict(moves=['_none', 'earthquake', 'calmmind', 'struggle'],
            abilities=['_none', 'levitate'], types=['_none', 'normal', 'ground', 'flying'],
            species=['_none', 'claydol'], items=['_none'])}
        table = transition.MoveTable(j.asarray([0,100,0,50],j.int32),j.asarray([0,2,0,1],j.int32),
            j.ones(4,bool),j.full(4,100,j.int32),j.ones(4,bool),j.zeros(4,j.int32),
            j.ones((4,4),j.int32),j.ones((4,4),j.int32),j.asarray(3,j.int32),entry=entry.from_metadata(metadata))
        state = empty_state(2)._replace(species=j.ones((2,2,6),j.int32),
            hp=j.full((2,2,6),200,j.int32),max_hp=j.full((2,2,6),200,j.int32),
            level=j.full((2,2,6),80,j.int32),done=j.zeros(2,bool))
        state = state._replace(moves=j.full_like(state.moves,2).at[:,0,0,0].set(1),
            pp=j.full_like(state.pp,20),types=state.types.at[...,0].set(1),
            stats=j.full_like(state.stats,100).at[:,0,:,5].set(200),
            ability=state.ability.at[0,1,0].set(1))
        actions=j.zeros((2,2),j.int32);chance=transition.sample_chance_inputs(state)
        actual=jax.jit(lambda state,public:transition.step_with_public(state,actions,table,public,
            chance_inputs=chance))(state,hooks.initialize(state).trackers)
        plain=jax.jit(lambda state:transition.step_with_chance(state,actions,table,chance))(state)
        for a,b in zip(jax.tree.leaves(actual.step),jax.tree.leaves(plain)):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(int(actual.step.state.hp[0,1,0]),200)
        self.assertLess(int(actual.step.state.hp[1,1,0]),200)
        # Literal saved prefix: a generic -immune, with no ability annotation.
        source=absence.Gen3RandbatSource(absence.RandbatSourceMetadata(
            'gen3randombattle',3,None,None,None,'fixture'),{'claydol':absence.Gen3RandbatSpeciesUniverse(
                'Claydol',80,(absence.Gen3RandbatVariant('0','fixture','Claydol','Fixture',80,
                    ('Earthquake','Calm Mind','Rest','Protect'),'Levitate','Leftovers'),))})
        reference=absence.PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=source)
        for line in ['|switch|p2a: Claydol|Claydol, L80|100/100','|turn|1',
                '|move|p1a: Groudon|Earthquake|p2a: Claydol','|-immune|p2a: Claydol']:
            reference.ingest_event(absence._public_event_from_line(line))
        self.assertIsNone(reference.snapshot().side('p2')[0].revealed_ability)
        self.assertFalse(bool(actual.trackers.shown_ability.any()))

    def test_self_hp_writes_replace_pre_item_snapshot_at_actual_step(self):
        state, base, leftovers, source = absence.fixture(batch=4, survivors=())
        names = ['_none', 'rest', 'recover', 'synthesis', 'substitute', 'calmmind', 'tackle', 'struggle']
        metadata = {'vocabulary': dict(moves=names, species=['_none', 'exploud'],
            abilities=['_none'], items=['_none', 'leftovers', 'choiceband'],
            types=['_none', 'normal', 'fire', 'ice', 'poison', 'steel', 'ground'])}
        count = len(names)
        table = transition.MoveTable(j.zeros(count, j.int32), j.ones(count, j.int32),
            j.zeros(count, bool), j.full(count, 100, j.int32), j.ones(count, bool),
            j.zeros(count, j.int32), j.ones((7, 7), j.int32), j.ones((7, 7), j.int32),
            j.asarray(7, j.int32), effects=effects.from_metadata(metadata), items=items.from_metadata(metadata))
        tables = NS(**vars(base), move_mask=j.broadcast_to(base.species_initial_mask[:, None], (2, count, 2)))
        # Real-set narrowing is inert in this fixture, isolating the absent-item
        # conclusion. The stored Rest example incorrectly eliminated Leftovers
        # after a public heal to full; all other numerical state stays identical.
        state = state._replace(hp=state.hp.at[:, 0, 0].set(j.asarray([100, 100, 100, 200])),
            moves=j.full_like(state.moves, 5).at[:, 0, 0, 0].set(j.arange(1, 5)),
            pp=j.full_like(state.pp, 10), level=j.full_like(state.level, 80),
            stats=j.full_like(state.stats, 100).at[:, 0, :, 5].set(200),
            types=state.types.at[..., 0].set(1))
        actions = j.zeros((4, 2), j.int32)
        chance = transition.sample_chance_inputs(state)
        public = hooks.initialize(state, tables).trackers
        actual = jax.jit(lambda state, public: transition.step_with_public(
            state, actions, table, public, belief_tables=tables, chance_inputs=chance))(state, public)
        plain = jax.jit(lambda state: transition.step_with_chance(state, actions, table, chance))(state)
        for a, b in zip(jax.tree.leaves(actual.step), jax.tree.leaves(plain)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(actual.step.state.hp[:, 0, 0].tolist(), [200, 200, 200, 150])
        self.assertEqual(actual.trackers.belief_mask[:, 0, 0, 0].tolist(), [255, 255, 255, 0])
        self.assertFalse(bool(actual.trackers.shown_item.any()))
        self.assertFalse(bool(actual.trackers.negative_items.known.any()))
        for percent, excluded in [(100, False), (75, True)]:
            result = absence.reference(source, [f'|-heal|p2a: Exploud|{percent}/100'], switch='50/100')
            self.assertEqual('leftovers' in result.ruled_out_items, excluded)


if __name__ == '__main__':
    unittest.main()
