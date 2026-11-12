"""Static installed-hook descriptors, without a constructor or engine call."""
from collections import Counter
from types import SimpleNamespace
import unittest

from gpu_engine import engine_device_observation as observation
from gpu_engine import engine_device_public_hooks as hooks


def installed_readout():
    # Literal dictionary/tail presence of the current original readout; values
    # are deliberately opaque because manifest inspection must not read them.
    sentinel = object()
    mon = dict.fromkeys((*hooks.MON_KEYS, 'switched_vs_active', 'stayed_vs_active'), sentinel)
    own = {key: value for key, value in mon.items() if key not in (
        'switched_before_attack', 'stayed_and_attacked', 'turns_active_total',
        'switched_vs_active', 'stayed_vs_active')}
    fields = dict.fromkeys(('weather_upkeeps', 'weather_permanent', 'wish_turns', 'wish_pending',
        'self_wish_turns', 'opp_wish_turns', 'self_wish_pending', 'opp_wish_pending', 'weather_turns',
        'items_removed_credit', 'self_items_removed_credit', 'opp_items_removed_credit'), sentinel)
    return SimpleNamespace(types=sentinel, ability=sentinel, item=sentinel,
        mon_features=mon, own_mon_features=own, field_features=fields,
        stats_features=dict.fromkeys(('opp_switch_count', 'opp_decision_opportunities', 'my_switch_turns'), sentinel),
        last_moves=sentinel, own_last_moves=sentinel, borrowed_ability=sentinel,
        own_borrowed_ability=sentinel, volatiles=sentinel, own_volatiles=sentinel)


class ManifestApplicabilityTests(unittest.TestCase):
    def test_installed_roles_keep_real_missing_inputs_and_unavailable_mechanics(self):
        rows = observation.column_manifest(installed_readout())
        by = {(r['section'], r['kind'], r['column']): r for r in rows}
        self.assertEqual(len(by), 1038)
        self.assertEqual(Counter(r['status'] for r in rows), {'DIRECT': 76, 'TRACKED': 324, 'ZERO': 638})
        missing = {'field': (105, 106, 111, 112, 115, 116), 'own': (37,), 'opponent': (37,),
            'stats': (123, 124, 126, 127, 128, 129, 130, 131)}
        for section, columns in missing.items():
            for column in columns:
                row = by[section, 'numeric', column]
                self.assertEqual(row['status'], 'ZERO')
                self.assertTrue(row['reason'].startswith('deferred'), row)
        # Fidelity A2: Knock Off's removal credit (B2 public_damage) is a tracked field fact.
        for column in (119, 120):
            row = by['field', 'numeric', column]
            self.assertEqual(row['status'], 'TRACKED', row)
            self.assertIn('items_removed_credit', row['reason'])
        # Fidelity M2: WRAP_TRAP_TURNS (33) is tracked (hooks wrap_trap_turns, B2's partial-trap clock).
        for section in ('own', 'opponent'):
            row = by[section, 'numeric', 33]
            self.assertEqual(row['status'], 'TRACKED', row);self.assertIn('wrap_trap_turns', row['reason'])
        # Fidelity M1: Trick's -item lines and the holder's move lines write CHOICE_LOCKED (40) and ITEM_SWAPPED (41).
        for section in ('own', 'opponent'):
            for column, key in ((40, 'choice_locked'), (41, 'item_swapped')):
                row = by[section, 'numeric', column]
                self.assertEqual(row['status'], 'TRACKED', row);self.assertIn(key, row['reason'])
        zero = [r for r in rows if r['status'] == 'ZERO']
        self.assertEqual(sum(r['reason'].startswith('structurally') for r in zero), 616)
        self.assertEqual(sum(r['reason'].startswith('deferred') for r in zero), 16)
        self.assertEqual(sum(r['reason'].startswith('unavailable:') for r in zero), 0)
        self.assertEqual(sum(r['reason'].startswith('unavailable hidden') for r in zero), 6)

    def test_sparse_writer_roles_and_absent_readout_do_not_claim_hooks(self):
        by = {(r['section'], r['kind'], r['column']): r for r in observation.column_manifest()}
        # Field/stats writers never write mon/action or PP bucket coordinates;
        # unnamed range members are as structural as the named first member.
        for section, kind, column in [('field', 'numeric', 0), ('field', 'numeric', 80),
                ('field', 'categorical', 40), ('stats', 'numeric', 53), ('stats', 'numeric', 80),
                ('stats', 'categorical', 0), ('own', 'numeric', 80), ('own', 'categorical', 17),
                ('opponent', 'numeric', 99), ('opponent', 'categorical', 3)]:
            self.assertTrue(by[section, kind, column]['reason'].startswith('structurally'))
        for section, column in [('field', 107), ('own', 24), ('opponent', 31), ('stats', 121)]:
            row = by[section, 'numeric', column]
            self.assertEqual(row['status'], 'ZERO')
            self.assertTrue(row['reason'].startswith('deferred'), row)
        self.assertEqual(by['belief', 'numeric', 131]['status'], 'TRACKED')
        self.assertIn('frozen investment_belief_narrowing=False', by['belief', 'numeric', 131]['reason'])


if __name__ == '__main__':
    unittest.main()
