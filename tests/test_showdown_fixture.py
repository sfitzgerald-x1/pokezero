import os
from pathlib import Path
import shutil
import unittest

from pokezero.showdown_fixture import (
    DEFAULT_GEN3_CUSTOM_FORMAT,
    FixturePokemon,
    OneTurnFixtureResult,
    build_start_payload,
    charmander_squirtle_fixture,
    pack_pokemon,
    pack_team,
    run_one_turn_fixture,
)


def integration_config():
    """Mirror tests/test_local_showdown.py: only run when a built checkout + node exist."""

    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig

    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=15.0)


class PackTeamTest(unittest.TestCase):
    def test_packs_simple_charmander_squirtle_fixture(self) -> None:
        p1_team, p2_team = charmander_squirtle_fixture()

        self.assertEqual(pack_team(p1_team), "Charmander|||Blaze|Ember,Tackle|||||||")
        self.assertEqual(pack_team(p2_team), "Squirtle|||Torrent|WaterGun,Tackle|||||||")

    def test_pack_pokemon_strips_spaces_from_move_and_ability_names(self) -> None:
        mon = FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun",))

        # Showdown's packName drops non-alphanumerics but keeps case.
        self.assertEqual(pack_pokemon(mon), "Squirtle|||Torrent|WaterGun|||||||")

    def test_pack_pokemon_emits_non_default_level(self) -> None:
        mon = FixturePokemon(species="Pikachu", moves=("Thunderbolt",), level=50)

        self.assertEqual(pack_pokemon(mon), "Pikachu||||Thunderbolt||||||50|")

    def test_pack_pokemon_packs_item_and_evs_when_supplied(self) -> None:
        mon = FixturePokemon(
            species="Snorlax",
            ability="Immunity",
            item="Leftovers",
            moves=("Body Slam",),
            evs={"hp": 252, "atk": 252},
        )

        self.assertEqual(
            pack_pokemon(mon),
            "Snorlax||Leftovers|Immunity|BodySlam||252,252,,,,|||||",
        )

    def test_pack_team_rejects_empty_team(self) -> None:
        with self.assertRaises(ValueError):
            pack_team([])

    def test_pack_pokemon_requires_moves(self) -> None:
        with self.assertRaises(ValueError):
            pack_pokemon(FixturePokemon(species="Ditto", moves=()))


class StartPayloadTest(unittest.TestCase):
    def test_build_start_payload_passes_teams_and_preserves_shape(self) -> None:
        payload = build_start_payload(
            battle_id="fixture",
            format_id=DEFAULT_GEN3_CUSTOM_FORMAT,
            seed=7,
            p1_team="Charmander|||Blaze|Ember|||||||",
            p2_team="Squirtle|||Torrent|WaterGun|||||||",
        )

        self.assertEqual(payload["type"], "start")
        self.assertEqual(payload["battleId"], "fixture")
        self.assertEqual(payload["formatid"], "gen3customgame")
        self.assertEqual(payload["players"]["p1"]["team"], "Charmander|||Blaze|Ember|||||||")
        self.assertEqual(payload["players"]["p2"]["team"], "Squirtle|||Torrent|WaterGun|||||||")
        self.assertEqual(payload["players"]["p1"]["name"], "PokeZero p1")
        self.assertEqual(payload["players"]["p2"]["name"], "PokeZero p2")

    def test_build_start_payload_seed_is_four_part_deterministic(self) -> None:
        from pokezero.local_showdown import showdown_seed_from_int

        payload = build_start_payload(
            battle_id="fixture",
            format_id=DEFAULT_GEN3_CUSTOM_FORMAT,
            seed=11,
            p1_team="a",
            p2_team="b",
        )

        self.assertEqual(payload["seed"], showdown_seed_from_int(11))
        self.assertEqual(len(payload["seed"].split(",")), 4)


@unittest.skipIf(integration_config() is None, "requires node and built Pokemon Showdown checkout")
class OneTurnFixtureIntegrationTest(unittest.TestCase):
    def test_runs_one_deterministic_turn_with_custom_one_mon_teams(self) -> None:
        config = integration_config()
        assert config is not None
        p1_team, p2_team = charmander_squirtle_fixture()

        result = run_one_turn_fixture(
            p1_team=p1_team,
            p2_team=p2_team,
            p1_choice="move 1",
            p2_choice="move 1",
            seed=7,
            config=config,
        )

        self.assertIsInstance(result, OneTurnFixtureResult)
        # Both seats were asked to act this turn (both choices were acknowledged/applied).
        self.assertTrue(result.p1_request and result.p1_request.get("active"))
        self.assertTrue(result.p2_request and result.p2_request.get("active"))
        # The submitted first-turn moves actually fired in the omniscient protocol.
        self.assertEqual(result.move_names(), ("Ember", "Water Gun"))
        self.assertIn("|-resisted|p2a: Squirtle", result.protocol_lines)
        self.assertIn("|-damage|p2a: Squirtle|209/229", result.protocol_lines)
        self.assertIn("|-supereffective|p1a: Charmander", result.protocol_lines)
        self.assertIn("|-damage|p1a: Charmander|127/219", result.protocol_lines)
        self.assertFalse(any(line.startswith("|error|") for line in result.protocol_lines))
        self.assertEqual(result.error_lines, ())
        # One turn of two healthy mons should not end the battle.
        self.assertFalse(result.terminal)

    def test_same_seed_reproduces_protocol_moves(self) -> None:
        config = integration_config()
        assert config is not None
        p1_team, p2_team = charmander_squirtle_fixture()

        def first_turn_moves() -> tuple[str, ...]:
            return run_one_turn_fixture(
                p1_team=p1_team,
                p2_team=p2_team,
                p1_choice="move 1",
                p2_choice="move 2",
                seed=21,
                config=config,
            ).move_names()

        self.assertEqual(first_turn_moves(), first_turn_moves())


if __name__ == "__main__":
    unittest.main()
