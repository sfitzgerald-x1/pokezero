from __future__ import annotations

import unittest

from scripts.attest_damage_arithmetic_tail import observed_direct_hit


class DamageArithmeticTailAttestationTests(unittest.TestCase):
    def test_switch_seeds_direct_damage_from_the_incoming_mon(self) -> None:
        row = {
            "pre_features": {"p1_hp": 300, "p2_hp": 200},
            "protocol": [
                "|switch|p1a: Raichu|Raichu, L83|141/235",
                "|move|p2a: Qwilfish|Sludge Bomb|p1a: Raichu",
                "|-damage|p1a: Raichu|19/235",
                "|-status|p1a: Raichu|psn",
            ],
        }

        hit = observed_direct_hit(row)

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.actor, "p2")
        self.assertEqual(hit.target, "p1")
        self.assertEqual(hit.move, "sludgebomb")
        self.assertEqual(hit.damage, 122)
        self.assertEqual(hit.secondary_status, "psn")

    def test_residual_damage_is_not_mistaken_for_the_move(self) -> None:
        row = {
            "pre_features": {"p1_hp": 100, "p2_hp": 100},
            "protocol": [
                "|move|p1a: A|Surf|p2a: B",
                "|-damage|p2a: B|70/100",
                "|-damage|p2a: B|64/100|[from] psn",
            ],
        }

        hit = observed_direct_hit(row)

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.damage, 30)
        self.assertIsNone(hit.secondary_status)
