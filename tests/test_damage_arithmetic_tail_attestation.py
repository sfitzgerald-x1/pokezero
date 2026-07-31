from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import attest_damage_arithmetic_tail as attestation
from scripts.attest_damage_arithmetic_tail import (
    DirectHit,
    _basic_oracle,
    _classify_branch_verdict,
    observed_direct_hit,
)


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

    def test_first_direct_hit_is_not_replaced_or_later_status_mutated(self) -> None:
        row = {
            "pre_features": {"p1_hp": 100, "p2_hp": 100},
            "protocol": [
                "|move|p1a: A|Surf|p2a: B",
                "|-damage|p2a: B|70/100",
                "|move|p2a: B|Sludge Bomb|p1a: A",
                "|-damage|p1a: A|80/100",
                "|-status|p1a: A|psn",
            ],
        }

        hit = observed_direct_hit(row)

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual((hit.actor, hit.target, hit.move, hit.damage), ("p1", "p2", "surf", 30))
        self.assertIsNone(hit.secondary_status)

    def test_status_binds_to_the_selected_first_direct_hit(self) -> None:
        row = {
            "pre_features": {"p1_hp": 100, "p2_hp": 100},
            "protocol": [
                "|move|p1a: A|Sludge Bomb|p2a: B",
                "|-damage|p2a: B|70/100",
                "|-supereffective|p2a: B",
                "|-status|p2a: B|psn",
            ],
        }

        hit = observed_direct_hit(row)

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.secondary_status, "psn")

    def test_complex_oracle_context_fails_closed(self) -> None:
        attacker = SimpleNamespace(
            id="Machamp", level=80, hp=300, maxhp=300, attack=200, defense=150,
            special_attack=150, special_defense=150, speed=100, ability="Guts", item="",
            status="", types=("Fighting",),
        )
        defender = SimpleNamespace(
            id="Snorlax", level=80, hp=400, maxhp=400, attack=150, defense=150,
            special_attack=150, special_defense=150, speed=80, ability="", item="",
            status="", types=("Normal",),
        )
        side = lambda member: SimpleNamespace(
            pokemon=(member,), active_index=0, attack_boost=0, defense_boost=0,
            special_attack_boost=0, special_defense_boost=0, speed_boost=0,
            side_conditions=SimpleNamespace(reflect=0, light_screen=0), volatile_statuses=(),
        )
        state = SimpleNamespace(weather="", side_one=side(attacker), side_two=side(defender))
        dex = SimpleNamespace(
            move_info=lambda _: SimpleNamespace(
                id="crosschop", base_power=100, gen3_category="Physical", type="Fighting"
            ),
            effectiveness=lambda *_: 1.0,
        )

        _, rolls, limit = _basic_oracle(
            state=state,
            direct=DirectHit("p1", "p2", "crosschop", 100, False, None),
            dex=dex,
        )

        self.assertIsNone(rolls)
        self.assertEqual(limit, "attacker_ability:guts")

    def test_branch_verdicts_fail_closed_and_require_secondary_coupling_evidence(self) -> None:
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=None, oracle_limit="attacker_ability:guts", native_max=100,
                observed_damage=90, nonterminal_damage=(92,), secondary_status="psn",
                secondary_branch_has_observed_damage=False,
            ),
            ("comparison_limit", "oracle_unavailable:attacker_ability:guts"),
        )
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(85, 100), oracle_limit=None, native_max=None,
                observed_damage=90, nonterminal_damage=(92,), secondary_status="psn",
                secondary_branch_has_observed_damage=False,
            ),
            ("comparison_limit", "native_damage_binding_unavailable"),
        )
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(85, 90, 100), oracle_limit=None, native_max=99,
                observed_damage=90, nonterminal_damage=(92,), secondary_status="psn",
                secondary_branch_has_observed_damage=False,
            )[0],
            "native_arithmetic_disagreement",
        )
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(85, 90, 100), oracle_limit=None, native_max=100,
                observed_damage=90, nonterminal_damage=(92,), secondary_status=None,
                secondary_branch_has_observed_damage=False,
            )[0],
            "no_arithmetic_disagreement",
        )
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(85, 90, 100), oracle_limit=None, native_max=100,
                observed_damage=90, nonterminal_damage=(92,), secondary_status="psn",
                secondary_branch_has_observed_damage=False,
            )[0],
            "fixed_single_roll_composition",
        )

    def test_stale_or_missing_provenance_refuses_to_emit_attestation(self) -> None:
        with patch.object(attestation, "assert_fresh", side_effect=SystemExit("stale build")):
            with self.assertRaisesRegex(SystemExit, "stale build"):
                attestation.main([
                    "--report", "missing.json", "--target", "1/2", "--showdown-root", "showdown",
                ])
        with patch.object(attestation.subprocess, "run", side_effect=OSError("git unavailable")):
            with self.assertRaisesRegex(SystemExit, "cannot record source provenance"):
                attestation._source_provenance()

    def test_emitted_json_has_deterministic_provenance_and_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "input.json"
            report.write_text(json.dumps({"repros": []}), encoding="utf-8")
            output = Path(directory) / "attestation.json"
            report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
            capture = io.StringIO()
            with (
                patch.object(attestation, "assert_fresh"),
                patch.object(attestation, "_source_provenance", return_value={"commit": "a" * 40, "tree_clean": True, "producer": {}}),
                patch.object(attestation, "compute_fingerprint", return_value={"fingerprint": "b" * 64}),
                patch.object(attestation, "load_showdown_dex", return_value=object()),
                redirect_stdout(capture),
            ):
                self.assertEqual(
                    attestation.main([
                        "--report", str(report), "--target", "2/3", "--target", "1/4",
                        "--showdown-root", "showdown", "--json", str(output),
                    ]),
                    0,
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "pokezero.damage-arithmetic-tail-attestation/v2")
        report_label = attestation._path_label(report)
        self.assertEqual(payload["input_reports"], [{"path": report_label, "sha256": report_hash}])
        self.assertEqual(payload["command"], [
            "python", "scripts/attest_damage_arithmetic_tail.py", "--report", report_label,
            "--target", "1/4", "--target", "2/3", "--showdown-root", "showdown",
        ])
        self.assertEqual([row["verdict"] for row in payload["targets"]], ["comparison_limit", "comparison_limit"])
