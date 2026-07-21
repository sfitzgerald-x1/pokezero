from __future__ import annotations

from contextlib import redirect_stderr
import io
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.coverage_enumeration_audit import (
    CoverageGame,
    CoverageSelection,
    build_coverage_plan,
    merge_coverage_ledgers,
    normalize_coverage_move,
    require_matching_audit_provenance,
)
from pokezero.deep_line_audit import DeepLineAuditReport
import pokezero.deep_line_audit as deep_line_audit
from pokezero.randbat import (
    Gen3RandbatSource,
    Gen3RandbatSpeciesUniverse,
    Gen3RandbatVariant,
    RandbatSourceMetadata,
)


_CLI_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coverage_enumeration_audit.py"
_CLI_SPEC = importlib.util.spec_from_file_location("coverage_enumeration_audit_cli_test", _CLI_MODULE_PATH)
if _CLI_SPEC is None or _CLI_SPEC.loader is None:  # pragma: no cover - importlib invariant
    raise RuntimeError(f"could not import coverage audit driver from {_CLI_MODULE_PATH}")
coverage_audit_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(coverage_audit_cli)


def _variant(
    variant_id: str,
    species: str,
    ability: str,
    item: str,
    moves: tuple[str, ...],
) -> Gen3RandbatVariant:
    return Gen3RandbatVariant(
        variant_id=variant_id,
        source_set_id=variant_id.rsplit("-", 1)[0],
        species=species,
        role="Fixture",
        level=80,
        moves=moves,
        ability=ability,
        item=item,
    )


def _source() -> Gen3RandbatSource:
    # Alpha's first-ability variants deliberately tie on four fresh moves. The
    # stable tiebreak picks alpha-a, leaving alpha-rare / Petaya Berry for the
    # gap-fill loop to prove it closes both atom kinds.
    alpha = (
        _variant("alpha-a", "Alpha", "Ability A", "Leftovers", ("movea", "moveb", "movec", "moved")),
        _variant("alpha-b", "Alpha", "Ability A", "Petaya Berry", ("alpharare", "movee", "movef", "moveg")),
        _variant("alpha-c", "Alpha", "Ability B", "Choice Band", ("moveh", "movei", "movej", "movek")),
    )
    beta = (
        _variant("beta-a", "Beta", "Ability C", "Salac Berry", ("movel", "movem", "moven", "moveo")),
    )
    return Gen3RandbatSource(
        metadata=RandbatSourceMetadata(
            format_id="gen3randombattle",
            generation=3,
            showdown_root=None,
            sets_path=None,
            generator_path=None,
            source_hash="fixture-source",
        ),
        universes={
            "alpha": Gen3RandbatSpeciesUniverse(species="Alpha", level=80, variants=alpha),
            "beta": Gen3RandbatSpeciesUniverse(species="Beta", level=80, variants=beta),
        },
    )


class CoverageEnumerationPlanTests(unittest.TestCase):
    def _plan(self, **kwargs):
        options = {
            "source_species": ("alpha", "beta"),
            "source_moves": (
                "movea", "moveb", "movec", "moved", "alpharare", "movee", "movef", "moveg",
                "moveh", "movei", "movej", "movek", "movel", "movem", "moven", "moveo",
            ),
            "source_items": ("Leftovers", "Petaya Berry", "Choice Band", "Salac Berry"),
        }
        options.update(kwargs)
        return build_coverage_plan(
            _source(),
            **options,
        )

    def test_both_passes_cover_every_species_reachable_ability_move_and_item(self) -> None:
        plan = self._plan()
        ledger = plan.coverage_ledger()

        self.assertTrue(ledger["complete"])
        self.assertEqual(ledger["uncovered"], {
            "species": [], "ability_pairs": [], "moves": [], "items": [], "variants": [],
        })
        self.assertEqual(len(ledger["expected"]["species"]), 2)
        self.assertEqual(len(ledger["expected"]["ability_pairs"]), 3)
        self.assertIn("alpharare", ledger["first_coverage"]["moves"])
        self.assertIn("Petaya Berry", ledger["first_coverage"]["items"])

    def test_gap_fill_is_explicit_when_the_two_pass_draft_misses_atoms(self) -> None:
        plan = self._plan()
        gap_games = [game for game in plan.games if game.purpose == "gap-fill"]

        self.assertEqual(len(gap_games), 1)
        self.assertEqual(gap_games[0].p1.variant_id, "alpha-b")
        self.assertIn("move:alpharare", gap_games[0].p1.targets)
        self.assertIn("item:petayaberry", gap_games[0].p1.targets)

    def test_single_pass_has_a_complete_single_pass_ability_scope(self) -> None:
        plan = self._plan(passes=("A",))
        ledger = plan.coverage_ledger()

        self.assertTrue(ledger["complete"])
        self.assertEqual(
            ledger["expected"]["ability_pairs"],
            [{"species": "alpha", "ability": "abilitya"}, {"species": "beta", "ability": "abilityc"}],
        )

    def test_shards_are_disjoint_and_reconstruct_the_full_game_plan(self) -> None:
        plan = self._plan()
        first = plan.games_for_shard(shard_index=0, shard_count=2)
        second = plan.games_for_shard(shard_index=1, shard_count=2)

        self.assertFalse({game.game_id for game in first} & {game.game_id for game in second})
        self.assertEqual(
            {game.game_id for game in first + second},
            {game.game_id for game in plan.games},
        )
        merged = merge_coverage_ledgers((plan.coverage_ledger(first), plan.coverage_ledger(second)))
        self.assertTrue(merged["complete"])
        self.assertEqual(merged["uncovered"], plan.coverage_ledger()["uncovered"])

    def test_exact_variant_mode_materializes_every_source_tuple_once(self) -> None:
        plan = self._plan(exact_variants=True)
        ledger = plan.coverage_ledger()

        self.assertEqual(len(plan.games), 2)
        self.assertTrue(all(game.purpose == "exact-variant" for game in plan.games))
        self.assertEqual(
            ledger["expected"]["variants"],
            ["alpha-a", "alpha-b", "alpha-c", "beta-a"],
        )
        self.assertEqual(
            ledger["first_coverage"]["variants"],
            {
                "alpha-a": "variant-0001",
                "alpha-b": "variant-0001",
                "alpha-c": "variant-0002",
                "beta-a": "variant-0002",
            },
        )
        self.assertTrue(ledger["complete"])
        self.assertEqual(ledger["uncovered"]["variants"], [])

    def test_exact_variant_ledger_is_incomplete_when_a_fixture_is_missing(self) -> None:
        plan = self._plan(exact_variants=True)
        partial = plan.coverage_ledger(plan.games[:1])

        self.assertFalse(partial["complete"])
        self.assertEqual(partial["uncovered"]["variants"], ["alpha-c", "beta-a"])

    def test_exact_variant_shards_merge_to_a_complete_tuple_ledger(self) -> None:
        plan = self._plan(exact_variants=True)
        merged = merge_coverage_ledgers(
            (
                plan.coverage_ledger(plan.games_for_shard(shard_index=0, shard_count=2)),
                plan.coverage_ledger(plan.games_for_shard(shard_index=1, shard_count=2)),
            )
        )

        self.assertTrue(merged["complete"])
        self.assertEqual(merged["uncovered"]["variants"], [])
        self.assertEqual(merged["first_coverage"]["variants"], plan.coverage_ledger()["first_coverage"]["variants"])

    def test_merged_ledger_requires_every_selected_game_to_complete(self) -> None:
        plan = self._plan(exact_variants=True)
        interrupted = dict(plan.coverage_ledger())
        interrupted["games_selected"] = 1

        merged = merge_coverage_ledgers((interrupted,))

        self.assertFalse(merged["complete"])
        self.assertEqual(merged["uncovered"], {"species": [], "ability_pairs": [], "moves": [], "items": [], "variants": []})

    def test_atom_only_ledger_keeps_atom_coverage_completion_semantics(self) -> None:
        plan = self._plan()
        atom_complete = dict(plan.coverage_ledger())
        atom_complete["games_selected"] = 1

        merged = merge_coverage_ledgers((atom_complete,))

        self.assertTrue(merged["complete"])

    def test_rejects_expected_atoms_without_a_materializable_source_variant(self) -> None:
        with self.assertRaisesRegex(ValueError, "lack a materializable source-variant carrier"):
            self._plan(source_moves=("movea", "missingmove"))

    def test_dynamic_aliases_collapse_but_typed_hidden_power_atoms_stay_distinct(self) -> None:
        self.assertEqual(normalize_coverage_move("Return102"), "return")
        self.assertEqual(normalize_coverage_move("frustration1"), "frustration")
        self.assertEqual(normalize_coverage_move("Hidden Power Fire"), "hiddenpowerfire")
        self.assertNotEqual(
            normalize_coverage_move("hiddenpowerfire"),
            normalize_coverage_move("hiddenpowergrass"),
        )

    def test_audit_provenance_rejects_missing_and_mixed_shards(self) -> None:
        baseline = {
            "public_repo_commit": "commit-a",
            "showdown_source_hash": "source-a",
            "observation_schema": "pokezero.observation.v3",
            "image_digest": "registry/pokezero@sha256:abc",
        }
        self.assertEqual(
            require_matching_audit_provenance(({"audit_provenance": baseline},)),
            baseline,
        )
        with self.assertRaisesRegex(ValueError, "no audit_provenance"):
            require_matching_audit_provenance(({},))
        with self.assertRaisesRegex(ValueError, "differs from shard 0"):
            require_matching_audit_provenance(
                (
                    {"audit_provenance": baseline},
                    {"audit_provenance": {**baseline, "image_digest": "registry/pokezero@sha256:def"}},
                )
            )


class CoverageEnumerationDriverTests(unittest.TestCase):
    @staticmethod
    def _cli_error(*arguments: str) -> int:
        with redirect_stderr(io.StringIO()):
            with unittest.TestCase().assertRaises(SystemExit) as raised:
                coverage_audit_cli.main(arguments)
        return raised.exception.code

    def test_exact_variant_cli_rejects_partial_ability_passes(self) -> None:
        code = self._cli_error(
            "--json", "/tmp/coverage-audit.json",
            "--coverage-json", "/tmp/coverage-ledger.json",
            "--observation-schema", "v3",
            "--exact-variants",
            "--pass", "A",
        )

        self.assertEqual(code, 2)

    def test_depth_cli_requires_exact_variants_and_failure_directory(self) -> None:
        missing_exact = self._cli_error(
            "--json", "/tmp/coverage-audit.json",
            "--coverage-json", "/tmp/coverage-ledger.json",
            "--observation-schema", "v3",
            "--depth-rounds", "1",
            "--failure-dir", "/tmp/coverage-failures",
        )
        missing_failure_dir = self._cli_error(
            "--json", "/tmp/coverage-audit.json",
            "--coverage-json", "/tmp/coverage-ledger.json",
            "--observation-schema", "v3",
            "--exact-variants",
            "--depth-rounds", "1",
        )

        self.assertEqual(missing_exact, 2)
        self.assertEqual(missing_failure_dir, 2)

    def test_failure_directory_requires_depth_mode(self) -> None:
        code = self._cli_error(
            "--json", "/tmp/coverage-audit.json",
            "--coverage-json", "/tmp/coverage-ledger.json",
            "--observation-schema", "v3",
            "--failure-dir", "/tmp/coverage-failures",
        )

        self.assertEqual(code, 2)

    def test_cli_requires_explicit_v3_schema(self) -> None:
        missing = self._cli_error(
            "--json", "/tmp/coverage-audit.json",
            "--coverage-json", "/tmp/coverage-ledger.json",
        )
        wrong = self._cli_error(
            "--json", "/tmp/coverage-audit.json",
            "--coverage-json", "/tmp/coverage-ledger.json",
            "--observation-schema", "v2.2",
        )

        self.assertEqual(missing, 2)
        self.assertEqual(wrong, 2)

    def test_protocol_signature_keeps_omission_prone_subtypes(self) -> None:
        self.assertEqual(
            deep_line_audit._canonical_protocol_signature(
                "|-activate|p1a: Gengar|ability: Cursed Body".split("|")
            ),
            "-activate:cursedbody",
        )
        self.assertEqual(
            deep_line_audit._canonical_protocol_signature(
                "|cant|p2a: Snorlax|slp".split("|")
            ),
            "cant:slp",
        )

    def test_depth_lane_preserves_submitted_moves_when_a_fixture_errors(self) -> None:
        selection = CoverageSelection(
            species="Alpha",
            ability="Ability A",
            item="Leftovers",
            level=80,
            moves=("movea",),
            variant_id="alpha-a",
            source_set_id="alpha",
            pass_name="A",
        )
        game = CoverageGame(
            game_id="variant-0001",
            seed=1,
            p1=selection,
            p2=selection,
            pass_name="both",
            purpose="exact-variant",
        )

        def fail_step(_actions: dict[str, int]) -> None:
            raise RuntimeError("bridge fault")

        env = SimpleNamespace(
            terminal=lambda: None,
            requested_players=lambda: ("p1", "p2"),
            observe=lambda _player_id: SimpleNamespace(legal_action_mask=(True,)),
            step=fail_step,
            protocol_lines=(),
        )

        with self.assertRaises(coverage_audit_cli._MoveUseLaneError) as raised:
            coverage_audit_cli._run_move_use_lane(
                env,
                game=game,
                report=DeepLineAuditReport(),
                max_rounds=1,
                depth_rounds=1,
            )

        self.assertIsInstance(raised.exception.cause, RuntimeError)
        self.assertEqual(raised.exception.move_use, {"p1": ["movea"], "p2": ["movea"]})

    def test_depth_lane_censuses_post_action_protocol_tags(self) -> None:
        selection = CoverageSelection(
            species="Alpha",
            ability="Ability A",
            item="Leftovers",
            level=80,
            moves=("movea",),
            variant_id="alpha-a",
            source_set_id="alpha",
            pass_name="A",
        )
        game = CoverageGame(
            game_id="variant-0001",
            seed=1,
            p1=selection,
            p2=selection,
            pass_name="both",
            purpose="exact-variant",
        )

        class ProtocolEnv:
            def __init__(self) -> None:
                self.protocol_lines: tuple[str, ...] = ()
                self._terminal = None

            def terminal(self):
                return self._terminal

            @staticmethod
            def requested_players():
                return ("p1", "p2")

            @staticmethod
            def observe(_player_id: str):
                return SimpleNamespace(legal_action_mask=(True,))

            def step(self, _actions: dict[str, int]) -> None:
                self.protocol_lines = (
                    "|move|p1a: Alpha|Move A|p2a: Alpha",
                    "|-fail|p2a: Alpha|move: Toxic",
                    "|turn|2",
                )
                self._terminal = SimpleNamespace()

        report = DeepLineAuditReport()
        with patch.object(coverage_audit_cli, "_audit_depth_boundary") as audit_boundary:
            coverage_audit_cli._run_move_use_lane(
                ProtocolEnv(),
                game=game,
                report=report,
                max_rounds=1,
                depth_rounds=1,
            )

        self.assertEqual(dict(report.protocol_events), {"move": 1, "-fail": 1})
        self.assertEqual(dict(report.protocol_signatures), {"-fail:toxic": 1, "move:movea": 1})
        audit_boundary.assert_not_called()

    def test_non_depth_lane_does_not_require_protocol_telemetry(self) -> None:
        selection = CoverageSelection(
            species="Alpha",
            ability="Ability A",
            item="Leftovers",
            level=80,
            moves=("movea",),
            variant_id="alpha-a",
            source_set_id="alpha",
            pass_name="A",
        )
        game = CoverageGame(
            game_id="variant-0001",
            seed=1,
            p1=selection,
            p2=selection,
            pass_name="both",
            purpose="exact-variant",
        )
        env = SimpleNamespace(
            terminal=lambda: None,
            requested_players=lambda: ("p1", "p2"),
            observe=lambda _player_id: SimpleNamespace(legal_action_mask=(True,)),
            step=lambda _actions: None,
        )

        report = DeepLineAuditReport()
        coverage_audit_cli._run_move_use_lane(
            env,
            game=game,
            report=report,
            max_rounds=1,
            depth_rounds=0,
        )

        self.assertEqual(dict(report.protocol_events), {})

    def test_depth_lane_censuses_each_round_without_double_counting(self) -> None:
        selection = CoverageSelection(
            species="Alpha",
            ability="Ability A",
            item="Leftovers",
            level=80,
            moves=("movea",),
            variant_id="alpha-a",
            source_set_id="alpha",
            pass_name="A",
        )
        game = CoverageGame(
            game_id="variant-0001",
            seed=1,
            p1=selection,
            p2=selection,
            pass_name="both",
            purpose="exact-variant",
        )

        class TwoRoundEnv:
            def __init__(self) -> None:
                self.protocol_lines: tuple[str, ...] = ()
                self.round = 0

            @staticmethod
            def terminal():
                return None

            @staticmethod
            def requested_players():
                return ("p1", "p2")

            @staticmethod
            def observe(_player_id: str):
                return SimpleNamespace(legal_action_mask=(True,))

            def step(self, _actions: dict[str, int]) -> None:
                self.round += 1
                self.protocol_lines += (
                    "|move|p1a: Alpha|Move A|p2a: Alpha",
                    *(("|-fail|p2a: Alpha|move: Toxic",) if self.round == 1 else ()),
                    f"|turn|{self.round + 1}",
                )

        report = DeepLineAuditReport()
        with patch.object(coverage_audit_cli, "_audit_depth_boundary") as audit_boundary:
            coverage_audit_cli._run_move_use_lane(
                TwoRoundEnv(),
                game=game,
                report=report,
                max_rounds=2,
                depth_rounds=2,
            )

        self.assertEqual(dict(report.protocol_events), {"move": 2, "-fail": 1})
        self.assertEqual(dict(report.protocol_signatures), {"-fail:toxic": 1, "move:movea": 2})
        self.assertEqual(audit_boundary.call_count, 2)

    def test_trace_keeps_native_candidate_while_exposing_copied_public_ability(self) -> None:
        """Trace must split its public copied fact from its native source atom."""

        categories = (
            "belief:possible_ability:synchronize",
            "belief:possible_item:leftovers",
            "belief:possible_ability:trace",
            "belief:possible_ability:overgrow",
            "belief:possible_move:psychic",
            "belief:possible_move:thunderbolt",
            "belief:possible_move:willowisp",
            "belief:possible_move:calmmind",
        )
        rows = {category: index + 101 for index, category in enumerate(categories)}
        vocab = SimpleNamespace(
            encode=lambda category: rows[category],
            is_enumerated=lambda category: category in rows,
        )
        categorical_ids = [
            [0] * 40
            for _ in range(coverage_audit_cli.OPPONENT_POKEMON_TOKEN_OFFSET + 1)
        ]
        self_token = coverage_audit_cli.SELF_POKEMON_TOKEN_OFFSET
        opponent_token = coverage_audit_cli.OPPONENT_POKEMON_TOKEN_OFFSET
        categorical_ids[self_token][coverage_audit_cli.CATEGORY_BELIEF_ABILITY_OFFSET] = rows[
            "belief:possible_ability:synchronize"
        ]
        categorical_ids[self_token][coverage_audit_cli.CATEGORY_BELIEF_ITEM_OFFSET] = rows[
            "belief:possible_item:leftovers"
        ]
        # Trace copied the opponent's Overgrow publicly, but Trace remains the
        # native source ability for belief-candidate membership.
        categorical_ids[opponent_token][coverage_audit_cli.CATEGORY_BELIEF_ABILITY_OFFSET] = rows[
            "belief:possible_ability:overgrow"
        ]
        categorical_ids[opponent_token][coverage_audit_cli.CATEGORY_BELIEF_ITEM_OFFSET] = rows[
            "belief:possible_item:leftovers"
        ]
        for index, move in enumerate(("psychic", "thunderbolt", "willowisp", "calmmind")):
            categorical_ids[opponent_token][coverage_audit_cli.CATEGORY_BELIEF_MOVE_OFFSET + index] = rows[
                f"belief:possible_move:{move}"
            ]
        observation = SimpleNamespace(categorical_ids=categorical_ids)
        belief = SimpleNamespace(
            revealed_ability="Overgrow",
            possible_abilities=("Trace",),
            candidate_variants=({"variant_id": "gardevoir-trace"},),
        )
        state = SimpleNamespace(
            turn_number=1,
            belief_view=SimpleNamespace(opponent_by_species=lambda: {"gardevoir": belief}),
        )
        source = SimpleNamespace(
            universe_for=lambda _species: SimpleNamespace(
                variants=(SimpleNamespace(variant_id="gardevoir-trace"),)
            )
        )
        env = SimpleNamespace(
            config=SimpleNamespace(category_vocab=vocab),
            _state_for_player=lambda _player_id: state,
            _belief_set_source=source,
            snapshot=lambda: SimpleNamespace(latest_requests={}),
        )
        self_selection = CoverageSelection(
            species="Alakazam",
            ability="Synchronize",
            item="Leftovers",
            level=80,
            moves=("psychic",),
            variant_id="alakazam-synchronize",
            source_set_id="alakazam",
            pass_name="A",
        )
        opponent_selection = CoverageSelection(
            species="Gardevoir",
            ability="Trace",
            item="Leftovers",
            level=80,
            moves=("psychic", "thunderbolt", "willowisp", "calmmind"),
            variant_id="gardevoir-trace",
            source_set_id="gardevoir",
            pass_name="A",
        )
        report = DeepLineAuditReport()

        with patch.object(coverage_audit_cli, "audit_live_decision", return_value=observation):
            coverage_audit_cli._audit_source_selection(
                env,
                player_id="p1",
                self_selection=self_selection,
                opponent_selection=opponent_selection,
                report=report,
            )

        self.assertEqual(report.findings, [])
        self.assertEqual(report.randbat_candidate_variants_checked, 1)
        self.assertIn(
            rows["belief:possible_ability:overgrow"],
            categorical_ids[opponent_token][
                coverage_audit_cli.CATEGORY_BELIEF_ABILITY_OFFSET : coverage_audit_cli.CATEGORY_BELIEF_ITEM_OFFSET
            ],
        )
        self.assertNotIn(
            rows["belief:possible_ability:trace"],
            categorical_ids[opponent_token][
                coverage_audit_cli.CATEGORY_BELIEF_ABILITY_OFFSET : coverage_audit_cli.CATEGORY_BELIEF_ITEM_OFFSET
            ],
        )


_DEFAULT_SHOWDOWN_ROOT = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
_SHOWDOWN_ROOT = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT", _DEFAULT_SHOWDOWN_ROOT))
_HAS_SHOWDOWN = (_SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists() and (
    _SHOWDOWN_ROOT / "dist" / "data" / "random-battles" / "gen3" / "teams.js"
).exists()


@unittest.skipUnless(_HAS_SHOWDOWN, "requires a built local Pokemon Showdown checkout")
class CoverageEnumerationSourceIntegrationTests(unittest.TestCase):
    def test_real_catalog_has_a_complete_two_pass_species_ability_move_and_item_plan(self) -> None:
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.randbat_vocab import gen3_randbat_entities

        source = load_gen3_randbat_source_cached(_SHOWDOWN_ROOT)
        entities = gen3_randbat_entities(_SHOWDOWN_ROOT)
        plan = build_coverage_plan(
            source,
            source_species=entities["species"],
            source_moves=entities["moves"],
            source_items=entities["items"],
        )
        ledger = plan.coverage_ledger()

        self.assertTrue(ledger["complete"])
        self.assertEqual(len(ledger["expected"]["species"]), 220)
        self.assertEqual(len(ledger["expected"]["moves"]), 125)
        self.assertEqual(len(ledger["expected"]["items"]), 13)
        self.assertEqual(len(plan.games), 220)
        self.assertEqual(plan.games[0].start_override().observation_format_id, "gen3randombattle")
        self.assertEqual(set(plan.source_metadata), {"format_id", "generation", "source_hash"})

    def test_real_catalog_exact_variant_plan_covers_every_source_tuple(self) -> None:
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.randbat_vocab import gen3_randbat_entities

        source = load_gen3_randbat_source_cached(_SHOWDOWN_ROOT)
        entities = gen3_randbat_entities(_SHOWDOWN_ROOT)
        plan = build_coverage_plan(
            source,
            source_species=entities["species"],
            source_moves=entities["moves"],
            source_items=entities["items"],
            exact_variants=True,
        )
        ledger = plan.coverage_ledger()
        source_variant_count = sum(len(universe.variants) for universe in source.universes.values())

        self.assertTrue(ledger["complete"])
        # The upstream Showdown/Dex data is intentionally live. The source hash in the
        # resulting artifact identifies its exact universe; this integration test verifies
        # that every tuple from that universe is planned rather than pinning a stale count.
        self.assertEqual(len(ledger["expected"]["variants"]), source_variant_count)
        self.assertEqual(len(ledger["first_coverage"]["variants"]), source_variant_count)
        self.assertEqual(len(plan.games), (source_variant_count + 1) // 2)


if __name__ == "__main__":
    unittest.main()
