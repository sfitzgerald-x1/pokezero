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
    _branch_direct_damage,
    _branch_has_status,
    _branch_report,
    _classify_branch_verdict,
    _native_rolls,
    _validated_slot_sides,
    observed_direct_hit,
)


class DamageArithmeticTailAttestationTests(unittest.TestCase):
    def _state(self, *, ability: str = "", item: str = "", weather: str = "", volatiles: tuple[str, ...] = ()) -> SimpleNamespace:
        attacker = SimpleNamespace(
            id="Attacker", level=80, hp=300, maxhp=300, attack=200, defense=150,
            special_attack=150, special_defense=150, speed=100, ability=ability, item=item,
            status="", types=("Fighting",),
        )
        defender = SimpleNamespace(
            id="Defender", level=80, hp=400, maxhp=400, attack=150, defense=150,
            special_attack=150, special_defense=150, speed=80, ability="", item="",
            status="", types=("Normal",),
        )
        def side(member: SimpleNamespace) -> SimpleNamespace:
            return SimpleNamespace(
                pokemon=(member,), active_index=0, attack_boost=0, defense_boost=0,
                special_attack_boost=0, special_defense_boost=0, speed_boost=0,
                side_conditions=SimpleNamespace(reflect=0, light_screen=0), volatile_statuses=volatiles,
            )
        return SimpleNamespace(weather=weather, side_one=side(attacker), side_two=side(defender))

    @staticmethod
    def _dex() -> SimpleNamespace:
        return SimpleNamespace(
            move_info=lambda move: SimpleNamespace(
                id=move, base_power=100, gen3_category="Physical", type="Fighting"
            ),
            effectiveness=lambda *_: 1.0,
        )

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

    def test_status_attribution_ignores_from_sources(self) -> None:
        row = {
            "pre_features": {"p1_hp": 100, "p2_hp": 100},
            "protocol": [
                "|move|p1a: A|Sludge Bomb|p2a: B",
                "|-damage|p2a: B|70/100",
                "|-status|p2a: B|psn|[from] ability: Poison Point",
            ],
        }
        hit = observed_direct_hit(row)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertIsNone(hit.secondary_status)
        self.assertFalse(_branch_has_status(row["protocol"], "p2", "psn"))

    def test_branch_damage_seeds_pre_hit_hp_and_switch_overrides_it(self) -> None:
        self.assertEqual(
            _branch_direct_damage(["|-damage|p2a: B|70/100"], "p2", pre_hit_hp=100), 30
        )
        self.assertEqual(
            _branch_direct_damage(
                ["|switch|p2a: B|B, L50|90/100", "|-damage|p2a: B|70/100"],
                "p2", pre_hit_hp=100,
            ),
            20,
        )

    def test_complex_oracle_context_fails_closed(self) -> None:
        state = self._state(ability="Guts")

        _, rolls, limit = _basic_oracle(
            state=state,
            direct=DirectHit("p1", "p2", "crosschop", 100, False, None),
            dex=self._dex(),
        )

        self.assertIsNone(rolls)
        self.assertEqual(limit, "attacker_ability:guts")

    def test_oracle_rejects_every_named_untranscribed_class(self) -> None:
        direct = DirectHit("p1", "p2", "crosschop", 10, False, None)
        for move in (
            "hiddenpowerice", "return", "frustration", "rollout", "furycutter",
            "weatherball", "explosion", "selfdestruct", "charge", "mudsport", "watersport",
            "helpinghand", "armthrust", "dragonrage", "magnitude",
        ):
            _, rolls, limit = _basic_oracle(
                state=self._state(), direct=DirectHit("p1", "p2", move, 10, False, None), dex=self._dex()
            )
            self.assertIsNone(rolls, move)
            self.assertIsNotNone(limit, move)
        for state in (
            self._state(item="Deep Sea Tooth"), self._state(item="Deep Sea Scale"),
            self._state(item="Metal Powder"), self._state(item="Sea Incense"),
            self._state(weather="Sandstorm"),
            self._state(volatiles=("charge",)),
        ):
            _, rolls, limit = _basic_oracle(state=state, direct=direct, dex=self._dex())
            self.assertIsNone(rolls)
            self.assertIsNotNone(limit)

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
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(), oracle_limit=None, native_max=100, observed_damage=90,
                nonterminal_damage=(), secondary_status=None, secondary_branch_has_observed_damage=False,
            ),
            ("comparison_limit", "oracle_empty_roll_support"),
        )
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(85, 100), oracle_limit=None, native_max=99, observed_damage=100,
                observed_ko_clamped=True, nonterminal_damage=(), secondary_status=None,
                secondary_branch_has_observed_damage=False,
            ),
            ("comparison_limit", "observed_damage_ko_clamped"),
        )

    def test_native_controls_probe_both_orders_and_fail_closed(self) -> None:
        class Engine:
            def calculate_damage(self, _state: object, _one: str, _two: str, first: bool) -> tuple[tuple[int, int], tuple[int, int]]:
                return ((10, 20), (30 if first else 31, 40 if first else 41))

        rolls, reason = _native_rolls(
            native_engine=Engine(), state=object(), side_one_choice="one", side_two_choice="two",
            actor_side="side_two",
        )
        self.assertIsNone(reason)
        self.assertEqual(rolls, {True: (30, 40), False: (31, 41)})

        class PanicEngine:
            def calculate_damage(self, *_: object) -> object:
                raise BaseException("native panic")

        rolls, reason = _native_rolls(
            native_engine=PanicEngine(), state=object(), side_one_choice="one", side_two_choice="two",
            actor_side="side_one",
        )
        self.assertIsNone(rolls)
        self.assertEqual(reason, "native_damage_call_failed:BaseException")

    def test_slot_sides_are_required_and_inverted_orientation_reaches_native_correctly(self) -> None:
        self.assertEqual(_validated_slot_sides({}), (None, "missing_slot_sides"))
        self.assertEqual(
            _validated_slot_sides({"slot_sides": {"p1": "side_one", "p2": "side_one"}}),
            (None, "invalid_slot_sides"),
        )

        calls: list[tuple[object, ...]] = []
        class State:
            @staticmethod
            def from_string(_text: str) -> object:
                return object()
        class Engine:
            def calculate_damage(self, _state: object, one: str, two: str, first: bool) -> tuple[tuple[int, int], tuple[int, int]]:
                calls.append((one, two, first))
                return ((10, 20), (20, 30))
        Engine.State = State
        class Search:
            def branch_events(self, _state: str, one: str, two: str, context: str, *_: object) -> str:
                calls.append((one, two, json.loads(context)))
                return json.dumps({"branches": [{"events": ["|-damage|p1a: B|70/100"], "legal_roll_state": "state"}]})
        row = {
            "engine_state": "state", "engine_states": ["state"], "kind": "transition_diverged",
            "choices": {"p1": "left", "p2": "right"},
            "slot_sides": {"p1": "side_two", "p2": "side_one"},
            "party_display": {"p1": ["Left"], "p2": ["Right"]}, "turn": 7,
            "pre_features": {"p1_hp": 100, "p2_hp": 100},
        }
        with (
            patch.object(attestation, "_native_modules", return_value=(Engine(), Search(), None)),
            patch.object(attestation, "_basic_oracle", return_value=({}, (17, 20), None)),
        ):
            result = _branch_report(row, DirectHit("p1", "p2", "crosschop", 20, False, None), self._dex())
        self.assertEqual(result["verdict"], "no_arithmetic_disagreement")
        self.assertEqual(
            result["native_representative_damage"],
            {"value": 18, "derived": True, "formula": "floor(native_max * 0.925)"},
        )
        self.assertIn(("right", "left", {"p1": ["Right"], "p2": ["Left"], "turn": 7}), calls)
        self.assertTrue(all(call[:2] == ("right", "left") for call in calls if len(call) == 3 and isinstance(call[2], bool)))

    def test_candidate_aggregation_requires_identical_contexts(self) -> None:
        row = {
            "engine_state": "first", "engine_states": ["first", "second"],
            "choices": {"p1": "one", "p2": "two"},
            "slot_sides": {"p1": "side_one", "p2": "side_two"},
            "party_display": {"p1": [], "p2": []}, "pre_features": {"p1_hp": 100, "p2_hp": 100},
        }
        with (
            patch.object(attestation, "_native_modules", return_value=(object(), object(), None)),
            patch.object(attestation, "_candidate_report", side_effect=[
                {"candidate_index": 0, "verdict": "no_arithmetic_disagreement", "reason": None, "native_max": 100},
                {"candidate_index": 1, "verdict": "native_arithmetic_disagreement", "reason": None, "native_max": 101},
            ]),
        ):
            result = _branch_report(row, DirectHit("p1", "p2", "crosschop", 90, False, None), self._dex())
        self.assertEqual(result["verdict"], "comparison_limit")
        self.assertEqual(result["verdict_reason"], "candidate_contexts_or_results_differ")
        self.assertIsNone(result["native_representative_damage"])

    def test_pure_helpers_collect_without_a_native_wheel(self) -> None:
        self.assertFalse(hasattr(attestation, "poke_engine"))
        with patch.object(attestation.importlib, "import_module", side_effect=ModuleNotFoundError("no wheel")):
            self.assertEqual(attestation._native_modules()[2], "native_modules_unavailable:ModuleNotFoundError")

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
                patch.object(attestation, "_showdown_source_provenance", return_value={"content_sha256": "c" * 64, "inputs": [], "git_commit": None, "git_clean": None}),
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

        self.assertEqual(payload["schema_version"], "pokezero.damage-arithmetic-tail-attestation/v3")
        report_label = attestation._path_label(report)
        self.assertEqual(payload["input_reports"], [{"path": report_label, "sha256": report_hash}])
        self.assertEqual(payload["command"], [
            "python", "scripts/attest_damage_arithmetic_tail.py", "--report", report_label,
            "--target", "1/4", "--target", "2/3", "--showdown-root", "showdown",
        ])
        self.assertEqual([row["verdict"] for row in payload["targets"]], ["comparison_limit", "comparison_limit"])
        self.assertEqual(payload["showdown_source"]["content_sha256"], "c" * 64)
