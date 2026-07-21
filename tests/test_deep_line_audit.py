from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import unittest
from types import SimpleNamespace

from pokezero.category_vocab import CategoryVocabulary
from pokezero.deep_line_audit import (
    AuditFinding,
    DeepLineAuditReport,
    _audit_side_tokens,
    _encode_species_category,
    _forecast_form_types,
    _numeric_features_equal_except,
    _raw_request_action_mask,
    _raw_side_condition_counts,
    _canonical_protocol_signature,
    canonical_protocol_signature,
    PROTOCOL_SIGNATURE_SCHEMA_VERSION,
    audit_protocol_cut_fixture,
    protocol_cut_fixtures,
    census_protocol_cooccurrences,
    protocol_signature_counts,
)
from pokezero.observation import PokeZeroObservationV0
from pokezero.golden_corpus_scenarios import interaction_registry_specs
from pokezero.showdown import (
    CATEGORY_PRIMARY,
    CATEGORY_SECONDARY,
    CATEGORY_TYPE_1,
    CATEGORY_TYPE_2,
    NUMERIC_ACTIVE,
    NUMERIC_HP_FRACTION,
    NUMERIC_LEGAL,
    NUMERIC_PRESENT,
    SELF_POKEMON_TOKEN_OFFSET,
    V3_REPLAY_OBSERVATION_SPEC,
    parse_showdown_replay,
)


_CLI_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "deep_line_audit.py"
_CLI_SPEC = importlib.util.spec_from_file_location("deep_line_audit_cli_test", _CLI_MODULE_PATH)
if _CLI_SPEC is None or _CLI_SPEC.loader is None:  # pragma: no cover - importlib invariant
    raise RuntimeError(f"could not import deep-line audit driver from {_CLI_MODULE_PATH}")
deep_line_audit_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(deep_line_audit_cli)


class DeepLineAuditReportTest(unittest.TestCase):
    def test_report_stamps_protocol_signature_schema_version(self) -> None:
        self.assertEqual(
            DeepLineAuditReport().to_json_dict()["protocol_signature_schema_version"],
            PROTOCOL_SIGNATURE_SCHEMA_VERSION,
        )

    def test_protocol_signature_normalizes_effect_prefixes_and_fieldactivate(self) -> None:
        self.assertEqual(
            _canonical_protocol_signature(("", "-singleturn", "p1a: A", "move: Protect")),
            "-singleturn:protect",
        )
        self.assertEqual(
            _canonical_protocol_signature(("", "-activate", "p1a: A", "ability: Shed Skin")),
            "-activate:shedskin",
        )
        self.assertEqual(
            _canonical_protocol_signature(("", "-fieldactivate", "move: Perish Song")),
            "-fieldactivate:perishsong",
        )
        self.assertEqual(
            canonical_protocol_signature(("", "cant", "p1a: A", "slp")),
            "cant:slp",
        )

    def test_public_protocol_signature_counts_exclude_transport_only_lines(self) -> None:
        counts = protocol_signature_counts(
            (
                "|init|battle",
                "|request|{private-payload}",
                "|move|p1a: A|Protect|p2a: B",
                "|-singleturn|p1a: A|move: Protect",
                "|turn|2",
            )
        )

        self.assertEqual(dict(counts), {"-singleturn:protect": 1, "init": 1, "move:protect": 1})

    def test_begin_game_resets_candidate_history_without_losing_aggregate_counts(self) -> None:
        report = DeepLineAuditReport()
        report.begin_game("random-seed-1")
        report.decisions_checked = 12
        report.candidate_count_history[("p1", "snorlax")] = 3

        report.begin_game("random-seed-2")

        self.assertEqual(report.games_audited, 2)
        self.assertEqual(report.decisions_checked, 12)
        self.assertEqual(report.candidate_count_history, {})
        self.assertEqual(report.current_game_id, "random-seed-2")

    def test_protocol_census_records_set_and_ordered_cooccurrences(self) -> None:
        report = DeepLineAuditReport()

        census_protocol_cooccurrences(
            (
                "|switch|p1a: A|A|100/100",
                "|move|p1a: A|Toxic|p2a: B",
                "|-status|p2a: B|tox",
                "|-damage|p2a: B|90/100 tox",
                "|turn|2",
                "|move|p2a: B|Recover|p2a: B",
            ),
            report=report,
        )

        self.assertEqual(report.protocol_cooccurrences[("-damage", "-status", "move", "switch")], 1)
        self.assertEqual(report.protocol_ordered_pairs[("move", "-status")], 1)
        self.assertEqual(report.protocol_ordered_pairs[("-status", "-damage")], 1)
        self.assertEqual(report.protocol_ordered_triples[("move", "-status", "-damage")], 1)
        self.assertEqual(report.protocol_events["move"], 2)

    def test_merge_preserves_protocol_coverage_from_parallel_shards(self) -> None:
        first = DeepLineAuditReport(games_audited=1)
        second = DeepLineAuditReport(games_audited=2)
        first.protocol_ordered_pairs[("move", "-damage")] = 3
        second.protocol_ordered_pairs[("move", "-damage")] = 4
        first.protocol_signatures["cant:slp"] = 2
        second.protocol_signatures["cant:slp"] = 5

        first.merge(second)

        self.assertEqual(first.games_audited, 3)
        self.assertEqual(first.protocol_ordered_pairs[("move", "-damage")], 7)
        self.assertEqual(first.protocol_signatures["cant:slp"], 7)

    def test_bridge_compact_form_id_matches_public_vocabulary_spelling(self) -> None:
        vocab = CategoryVocabulary(tokens=("species:deoxys-speed", "species:deoxys"))

        self.assertEqual(
            _encode_species_category(vocab, "deoxysspeed"),
            vocab.encode("species:deoxys-speed"),
        )

    def test_numeric_comparison_can_exclude_live_only_annotations(self) -> None:
        expected = ((0.0, 0.0, 1.0),)
        annotation_only = ((0.0, 0.5, 1.0),)
        public_difference = ((0.0, 0.5, 0.0),)

        self.assertTrue(_numeric_features_equal_except(expected, annotation_only, frozenset({1})))
        self.assertFalse(_numeric_features_equal_except(expected, public_difference, frozenset({1})))

    def test_known_finding_suppression_preserves_audit_incidence(self) -> None:
        report = DeepLineAuditReport(suppressed_kinds=frozenset({"known"}))
        report.add(
            AuditFinding(
                kind="known",
                player_id="p1",
                turn=1,
                column="token[1]",
                expected=1,
                actual=0,
                detail="known regression",
            )
        )

        self.assertTrue(report.ok)
        self.assertEqual(report.suppressed_findings, {"known": 1})

    def test_protocol_cut_fixtures_are_parseable_public_sequences(self) -> None:
        fixtures = protocol_cut_fixtures()

        self.assertEqual({fixture.name for fixture in fixtures}, {
            "baton_pass_switch_boundary",
            "color_change_typechange",
            "cureteam_benched_toxic",
            "forecast_formechange",
            "intimidate_switch_in",
            "leech_seed_pending_snapshot",
            "sand_stream_switch_in",
        })
        for fixture in fixtures:
            replay = parse_showdown_replay(
                fixture.lines,
                battle_id=f"battle-gen3randombattle-test-{fixture.name}",
            )
            self.assertGreater(len(replay.public_events), 1, fixture.name)

    def test_forecast_fixture_keeps_base_species_and_records_live_type_source(self) -> None:
        fixture = next(item for item in protocol_cut_fixtures() if item.name == "forecast_formechange")
        report = DeepLineAuditReport()

        audit_protocol_cut_fixture(fixture, report=report)

        self.assertTrue(report.ok, report.findings)

    def test_forecast_bridge_oracle_checks_base_species_and_live_type_slots(self) -> None:
        vocab = CategoryVocabulary(tokens=("species:castform", "status:none", "type:water"))
        spec = V3_REPLAY_OBSERVATION_SPEC
        categorical = [[0] * spec.categorical_feature_count for _ in range(spec.token_count)]
        numeric = [[0.0] * spec.numeric_feature_count for _ in range(spec.token_count)]
        token = SELF_POKEMON_TOKEN_OFFSET
        categorical[token][CATEGORY_PRIMARY] = vocab.encode("species:castform")
        categorical[token][CATEGORY_SECONDARY] = vocab.encode("status:none")
        categorical[token][CATEGORY_TYPE_1] = vocab.encode("type:water")
        categorical[token][CATEGORY_TYPE_2] = vocab.encode("")
        numeric[token][NUMERIC_PRESENT] = 1.0
        numeric[token][NUMERIC_ACTIVE] = 1.0
        numeric[token][NUMERIC_LEGAL] = 1.0
        numeric[token][NUMERIC_HP_FRACTION] = 1.0
        observation = PokeZeroObservationV0(
            categorical_ids=tuple(tuple(row) for row in categorical),
            numeric_features=tuple(tuple(row) for row in numeric),
            token_type_ids=tuple(0 for _ in range(spec.token_count)),
            attention_mask=tuple(False for _ in range(spec.token_count)),
            legal_action_mask=(False,) * 9,
        )
        dex = SimpleNamespace(species_info=lambda species: SimpleNamespace(types=("Water",)))
        report = DeepLineAuditReport()

        _audit_side_tokens(
            observation,
            {"pokemon": ({
                "species": "Castform-Rainy",
                "baseSpecies": "Castform",
                "hp": 100,
                "maxhp": 100,
                "isActive": True,
                "boosts": {},
            },)},
            SELF_POKEMON_TOKEN_OFFSET,
            ({"species": "Castform"},),
            "p1",
            1,
            vocab,
            report,
            live_type_source="forme:Castform-Rainy",
            dex=dex,
        )

        self.assertTrue(report.ok, report.findings)
        self.assertEqual(
            _forecast_form_types(
                SimpleNamespace(species_info=lambda _species: None), "forme: Castform-Rainy"
            ),
            ("Water",),
        )
        self.assertIsNone(
            _forecast_form_types(SimpleNamespace(species_info=lambda _species: None), "forme: Missing")
        )

    def test_raw_request_action_mask_handles_moves_trap_and_forced_switches(self) -> None:
        request = {
            "active": [{"moves": [{"disabled": False}, {"disabled": True}]}],
            "side": {
                "pokemon": [
                    {"active": True, "condition": "100/100"},
                    {"active": False, "condition": "100/100"},
                    {"active": False, "condition": "0 fnt"},
                ]
            },
        }
        self.assertEqual(_raw_request_action_mask(request), (True, False, False, False, True, False, False, False, False))

        trapped = {**request, "active": [{"moves": [{"disabled": False}], "trapped": True}]}
        self.assertEqual(_raw_request_action_mask(trapped), (True, False, False, False, False, False, False, False, False))

        forced = {**trapped, "forceSwitch": [True]}
        self.assertEqual(_raw_request_action_mask(forced), (False, False, False, False, True, False, False, False, False))

    def test_raw_side_condition_counts_reads_layers_and_timed_durations(self) -> None:
        counts = _raw_side_condition_counts({
            "sideConditions": {
                "spikes": {"layers": 2},
                "reflect": {"duration": 4},
                "lightscreen": {"duration": 0},
            }
        })

        self.assertEqual(counts, {"spikes": 2, "reflect": 4})

    def test_source_manifest_separates_catalog_from_sampled_components(self) -> None:
        source = SimpleNamespace(
            metadata=SimpleNamespace(
                format_id="gen3randombattle",
                generation=3,
                source_hash="fixture-hash",
            ),
            universes={
                "xatu": SimpleNamespace(
                    species="Xatu",
                    variants=(
                        SimpleNamespace(
                            variant_id="xatu-1-variant-1",
                            moves=("Psychic", "Wish", "Protect", "Return"),
                            ability="Synchronize",
                            item="Leftovers",
                        ),
                        SimpleNamespace(
                            variant_id="xatu-1-variant-2",
                            moves=("Psychic", "Night Shade", "Wish", "Protect"),
                            ability="Early Bird",
                            item="Lum Berry",
                        ),
                    ),
                )
            },
        )
        source.universe_for = lambda species: source.universes.get(str(species).lower())
        report = DeepLineAuditReport()

        report.record_randbat_source(source)
        report.record_observed_randbat_team(({
            "species": "Xatu",
            "moves": ["Psychic", "Wish", "Protect", "Return102"],
            "ability": "Synchronize",
            "item": "Leftovers",
        },), source=source)

        coverage = report.to_json_dict()["randbat_source_coverage"]
        self.assertEqual(coverage["source_metadata"]["source_hash"], "fixture-hash")
        self.assertEqual(coverage["catalog_component_counts"]["variant"], 2)
        self.assertEqual(coverage["observed_component_counts"]["variant"], 1)
        self.assertEqual(coverage["observed_component_counts"]["species"], 1)
        self.assertEqual(coverage["unobserved_component_counts"]["ability"], 1)
        self.assertIn("return", report.randbat_observed_components["move"])
        self.assertNotIn("return102", report.randbat_observed_components["move"])


class DeepLineAuditDriverTests(unittest.TestCase):
    @staticmethod
    def _cli_error(*arguments: str) -> int:
        with redirect_stderr(io.StringIO()):
            with unittest.TestCase().assertRaises(SystemExit) as raised:
                deep_line_audit_cli.main(arguments)
        return raised.exception.code

    def test_cli_requires_explicit_v3_schema(self) -> None:
        missing = self._cli_error("--json", "/tmp/deep-line-audit.json")
        wrong = self._cli_error(
            "--json", "/tmp/deep-line-audit.json", "--observation-schema", "v2.2"
        )

        self.assertEqual(missing, 2)
        self.assertEqual(wrong, 2)

    def test_interaction_registry_is_explicit_opt_in(self) -> None:
        default = deep_line_audit_cli._available_scenarios(include_interaction_registry=False)
        expanded = deep_line_audit_cli._available_scenarios(include_interaction_registry=True)
        interaction_names = {spec.name for spec in interaction_registry_specs()}

        self.assertFalse(interaction_names & default.keys())
        self.assertTrue(interaction_names <= expanded.keys())

        selected, requested = deep_line_audit_cli._requested_scenario_names(
            named_scenarios=(),
            include_all_scenarios=False,
            include_interaction_registry=True,
        )
        self.assertEqual(selected.keys(), expanded.keys())
        self.assertEqual(requested, interaction_names)


if __name__ == "__main__":
    unittest.main()
