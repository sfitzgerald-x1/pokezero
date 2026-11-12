"""Actual public-HP hook fixtures against the original absent-item rule.

No engine step, recorded replay, numerical damage calculation or model is run.
"""
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_device_public_hooks as h, engine_device_negative_items as n
from gpu_engine import engine_device_trackers as tr
from gpu_engine.state import empty_state
from pokezero.belief import PublicBattleBeliefEngine
from pokezero.randbat import (Gen3RandbatSource, Gen3RandbatSpeciesUniverse,
                             Gen3RandbatVariant, RandbatSourceMetadata)
from pokezero.showdown import _public_event_from_line

BASE = '564cffe68220f49741c549caa74ac5d2276ae34c'


def fixture(batch=2, survivors=(0, 1), count=8):
    variants = tuple(Gen3RandbatVariant(str(i), 'fixture', 'Exploud', 'Fixture', 80,
        ('Tackle', 'Protect', 'Rest', 'Sleep Talk'), 'Soundproof',
        'Choice Band' if i in survivors else 'Leftovers') for i in range(count))
    source = Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle', 3, None, None, None, 'fixture'),
        {'exploud': Gen3RandbatSpeciesUniverse('Exploud', 80, variants)})
    shape = (batch, 2, 6)
    state = empty_state(batch)._replace(species=j.ones(shape, j.int32),
        hp=j.full(shape, 200, j.int32), max_hp=j.full(shape, 200, j.int32),
        done=j.zeros(batch, bool))
    pool = j.asarray([[0, 0], [(1 << count) - 1, 0]], j.uint32)
    left = sum(1 << i for i in range(count) if i not in survivors)
    item = j.broadcast_to(pool[:, None, :], (2, 3, 2)).at[:, 1].set(j.asarray([left, 0], j.uint32))
    tables = NS(species_initial_mask=pool, item_mask=item,
        ability_mask=j.broadcast_to(pool[:,None,:],(2,2,2)))
    return state, tables, j.asarray([False, True, False]), source


def reference(source, hp_lines, *, switch='100/100'):
    engine = PublicBattleBeliefEngine(format_id='gen3randombattle', set_source=source, item_belief_narrowing=True)
    for line in ['|switch|p2a: Exploud|Exploud, L80|'+switch, '|turn|1', *hp_lines, '|upkeep']:
        engine.ingest_event(_public_event_from_line(line))
    return engine.snapshot().side('p2')[0]


class DeviceNegativeItemsTests(unittest.TestCase):
    def test_recorded_shapes_original_rule_and_both_owner_readouts(self):
        # The five retained examples have exactly these before/after index sets.
        # Source fixtures isolate the absent-item inference; these are not a
        # claim to have replayed/reconstructed their complete original worlds.
        cases = [(8, (0, 1), 80), (2, (1,), 86), (9, (1, 2, 4, 7, 8), 62),
                 (4, (0,), 58), (15, (5, 6, 7, 9, 10, 12), 87)]
        for count, survivors, percent in cases:
            s, tables, leftovers, source = fixture(survivors=survivors, count=count)
            @jax.jit
            def run(state):
                c = h.initialize(state, tables)
                after = state._replace(hp=state.hp.at[:, :, 0].set(percent*2))
                for side in range(2):
                    h.negative_hp_line(c, after, side, j.ones(2, bool), source_class=n.HP_ACTION)
                h.turn_end(c, state, after, j.zeros(2, bool), upkeep=j.ones(2, bool), leftovers=leftovers)
                return c.trackers, tr.readout(c.trackers, j.zeros(2, j.int32)), tr.readout(c.trackers, j.ones(2, j.int32))
            carry, owner0, owner1 = run(s)
            ref = reference(source, [f'|-damage|p2a: Exploud|{percent}/100'])
            self.assertEqual(tuple(int(v['variant_id']) for v in ref.candidate_variants), survivors)
            bits = sum(1 << i for i in survivors)
            self.assertEqual(owner0.belief_mask[:, 0].tolist(), [[bits, 0]]*2)
            np.testing.assert_array_equal(owner0.belief_mask, owner1.belief_mask)
            self.assertFalse(bool(carry.negative_items.known.any()))
            self.assertFalse(bool(owner0.item.any()))  # no hidden item reveal
            self.assertEqual(ref.possible_items, ('Choice Band',))

    def test_source_phase_order_unknown_and_no_stale_endpoint_evidence(self):
        s, tables, leftovers, source = fixture(batch=8)
        @jax.jit
        def run(state):
            c = h.initialize(state, tables)
            damaged = state._replace(hp=state.hp.at[:, 1, 0].set(160))
            h.negative_hp_line(c, damaged, 1, j.ones(8, bool), source_class=n.HP_ACTION)
            # Same current endpoint, different publicly ordered pre-slot facts.
            full = damaged._replace(hp=damaged.hp.at[:, 1, 0].set(200))
            h.negative_hp_line(c, full, 1, j.arange(8)==1, source_class=n.HP_PRE_ITEM)
            h.negative_hp_line(c, damaged, 1, j.arange(8)==2, source_class=n.HP_UNKNOWN)
            # Known post-slot poison must not overwrite the full pre-slot HP.
            h.negative_hp_line(c, damaged, 1, j.arange(8)==1, source_class=n.HP_POST_ITEM)
            h.reveal_item(c, damaged, 1, j.ones(8, j.int32), j.arange(8)==3)
            c.trackers = h._write(c.trackers, 'item_removed', 1, damaged.active[:, 1], j.ones(8, bool), j.arange(8)==4)
            h.feature(c, 1, damaged.active[:, 1], 'item_swapped', j.ones(8, j.int32), j.arange(8)==5)
            # Row6 pauses before upkeep, preserving this turn's facts for its
            # actual continuation. Row7 has no current HP evidence at all.
            c.trackers = c.trackers._replace(negative_items=c.trackers.negative_items._replace(
                known=c.trackers.negative_items.known.at[7].set(False)))
            h.turn_end(c, state, damaged, j.zeros(8, bool), upkeep=j.arange(8)!=6, leftovers=leftovers)
            first = c.trackers
            c.trackers = c.trackers._replace(belief_mask=c.trackers.belief_mask.at[:, 1, 0].set(tables.species_initial_mask[1]))
            # A later low endpoint is not a new HP publication. Only row6's
            # retained actual-turn snapshot may now eliminate Leftovers.
            h.turn_end(c, damaged, damaged, j.zeros(8, bool), upkeep=j.ones(8, bool), leftovers=leftovers)
            return first, c.trackers
        first, later = run(s)
        self.assertEqual(first.belief_mask[:, 1, 0, 0].tolist(), [3, 255, 255, 252, 255, 255, 255, 255])
        self.assertEqual(later.belief_mask[:, 1, 0, 0].tolist(), [255, 255, 255, 255, 255, 255, 3, 255])
        comparisons = [
            (['|-damage|p2a: Exploud|80/100'], True),
            (['|-damage|p2a: Exploud|80/100', '|', '|-heal|p2a: Exploud|100/100|[from] move: Wish',
              '|-damage|p2a: Exploud|80/100|[from] psn'], False),
            (['|-damage|p2a: Exploud|80/100', '|', '|-heal|p2a: Exploud|80/100|[silent]'], False),
            (['|-damage|p2a: Exploud|80/100', '|-heal|p2a: Exploud|86/100|[from] item: Leftovers'], False)]
        for lines, excluded in comparisons:
            self.assertEqual('leftovers' in reference(source, lines).ruled_out_items, excluded)

    def test_switch_condition_faint_missing_namespace_and_default_physics(self):
        s, tables, leftovers, _ = fixture()
        c = h.initialize(s, tables)
        new = s._replace(active=s.active.at[:, 1].set(3), hp=s.hp.at[:, 1, 3].set(170))
        h.install(c, new, 1, j.asarray([True, False]))
        h.turn_end(c, s, new, j.zeros(2, bool), upkeep=j.ones(2, bool), leftovers=leftovers)
        self.assertEqual(c.trackers.belief_mask[0, 1, 1].tolist(), [3, 0])
        self.assertFalse(bool(c.trackers.present[1, 1, 1]))
        # Unknown namespace is not silently mapped to a guessed item; missing
        # metadata expires evidence without manufacturing an exclusion.
        c = h.initialize(s, tables)
        low = s._replace(hp=s.hp.at[:, 1, 0].set(100))
        h.negative_hp_line(c, low, 1, True, source_class=n.HP_ACTION)
        with self.assertRaisesRegex(ValueError, 'namespaces'):
            n.upkeep(c.trackers, tables, j.ones(4, bool), True)
        t = n.upkeep(c.trackers, tables, None, True)
        self.assertFalse(bool(t.negative_items.known.any()))
        np.testing.assert_array_equal(t.belief_mask, c.trackers.belief_mask)
        dead = low._replace(hp=low.hp.at[:, 1, 0].set(0))
        h.turn_end(c, low, dead, j.zeros(2, bool), upkeep=j.ones(2, bool), leftovers=leftovers)
        self.assertEqual(c.trackers.belief_mask[:, 1, 0, 0].tolist(), [255, 255])
        # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.

    def test_unavailable_hp_cannot_become_snapshot_or_current_absence_evidence(self):
        s, tables, leftovers, _ = fixture(batch=3)
        c = h.initialize(s, tables)
        invalid = s._replace(max_hp=s.max_hp.at[0, 1, 0].set(0),
                             hp=s.hp.at[1, 1, 0].set(201))
        h.negative_hp_line(c, invalid, 1, True, source_class=n.HP_ACTION)
        self.assertEqual(c.trackers.negative_items.known[:, 1, 0].tolist(), [False, False, True])
        # No new HP-line notification: current generic observation itself
        # flags unavailable HP. Its clamped endpoint cannot authorize pruning.
        low = s._replace(hp=s.hp.at[:, 1, 0].set(160))
        h.negative_hp_line(c, low, 1, True, source_class=n.HP_ACTION)
        h.turn_end(c, s, invalid, j.zeros(3, bool), upkeep=j.ones(3, bool), leftovers=leftovers)
        self.assertEqual(c.trackers.belief_mask[:, 1, 0, 0].tolist(), [255, 255, 3])


if __name__ == '__main__':
    unittest.main()
