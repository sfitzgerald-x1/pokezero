import unittest

from pokezero.category_vocab import CategoryVocabulary
from pokezero.deep_line_audit import (
    AuditFinding,
    DeepLineAuditReport,
    _encode_species_category,
    _raw_request_action_mask,
    _raw_side_condition_counts,
    protocol_cut_fixtures,
    census_protocol_cooccurrences,
)
from pokezero.showdown import parse_showdown_replay


class DeepLineAuditReportTest(unittest.TestCase):
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

        first.merge(second)

        self.assertEqual(first.games_audited, 3)
        self.assertEqual(first.protocol_ordered_pairs[("move", "-damage")], 7)

    def test_bridge_compact_form_id_matches_public_vocabulary_spelling(self) -> None:
        vocab = CategoryVocabulary(tokens=("species:deoxys-speed", "species:deoxys"))

        self.assertEqual(
            _encode_species_category(vocab, "deoxysspeed"),
            vocab.encode("species:deoxys-speed"),
        )

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


if __name__ == "__main__":
    unittest.main()
