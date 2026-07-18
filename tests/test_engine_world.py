"""Tests for the belief-world -> poke-engine constructor (v3 plan, track A)."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    battle_spec_from_payload,
    unpack_pokemon,
    unpack_team,
)
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.gen3_damage import gen3_hp_stat  # noqa: E402
from pokezero.showdown_fixture import FixturePokemon, pack_pokemon, pack_team  # noqa: E402


def _dex() -> ShowdownDex:
    def species(species_id: str, name: str, types: tuple[str, ...], base: dict[str, int], weight: float) -> SpeciesInfo:
        return SpeciesInfo(id=species_id, name=name, types=types, base_stats=base, weight_kg=weight)

    def move(move_id: str, pp: int) -> MoveInfo:
        return MoveInfo(
            id=move_id, name=move_id, type="normal", category="physical",
            gen3_category="physical", base_power=50, accuracy=100.0, priority=0,
            recoil=False, drain=False, heal=False, status=None, boosts={},
            target="normal", selfdestruct=False, pp=pp,
        )

    return ShowdownDex(
        moves={
            "earthquake": move("earthquake", 10),
            "icebeam": move("icebeam", 10),
            "surf": move("surf", 15),
            "bodyslam": move("bodyslam", 15),
            "shadowball": move("shadowball", 15),
        },
        species={
            "swampert": species("swampert", "Swampert", ("water", "ground"), {"hp": 100, "atk": 110, "def": 90, "spa": 85, "spd": 90, "spe": 60}, 81.9),
            "snorlax": species("snorlax", "Snorlax", ("normal",), {"hp": 160, "atk": 110, "def": 65, "spa": 65, "spd": 110, "spe": 30}, 460.0),
            "starmie": species("starmie", "Starmie", ("water", "psychic"), {"hp": 60, "atk": 75, "def": 85, "spa": 100, "spd": 85, "spe": 115}, 80.0),
        },
        type_chart={},
    )


def _team(*mons: FixturePokemon) -> tuple[FixturePokemon, ...]:
    return mons


_SWAMPERT = FixturePokemon(
    species="Swampert", moves=("earthquake", "icebeam"), ability="Torrent",
    item="Leftovers", level=84, evs={stat: 85 for stat in ("hp", "atk", "def", "spa", "spd", "spe")},
)
_SNORLAX = FixturePokemon(
    species="Snorlax", moves=("bodyslam", "shadowball"), ability="Immunity",
    item="Leftovers", level=80, evs={stat: 85 for stat in ("hp", "atk", "def", "spa", "spd", "spe")},
)
_STARMIE = FixturePokemon(
    species="Starmie", moves=("surf",), ability="Natural Cure", item="Leftovers", level=79,
    evs={stat: 85 for stat in ("hp", "atk", "def", "spa", "spd", "spe")},
)


def _maxhp(mon: FixturePokemon, dex: ShowdownDex) -> int:
    info = dex.species_info(mon.species)
    return gen3_hp_stat(int(info.base_stats["hp"]), 31, int((mon.evs or {}).get("hp", 0)), mon.level)


def _payload(dex: ShowdownDex, **overrides):
    swampert_hp = _maxhp(_SWAMPERT, dex)
    payload = {
        "turn": 7,
        "weather": None,
        "weatherSetTurn": None,
        "weatherFromAbility": False,
        "futureSight": {"p1": 0, "p2": 0},
        "wishSetTurns": {},
        "leechSeedSourceSides": {},
        "pendingBatonPassSides": [],
        "deferredOpponentActions": {},
        "deferredOpponentActionPriors": {},
        "selfPlayer": "p1",
        "selfRequestKind": "move",
        "sides": {
            "p1": {
                "pokemon": [
                    {
                        "species": "Swampert",
                        "condition": f"{swampert_hp - 40}/{swampert_hp}",
                        "active": True,
                        "moves": [
                            {"id": "earthquake", "pp": 12, "maxpp": 16, "disabled": False},
                            {"id": "icebeam", "pp": 16, "maxpp": 16, "disabled": False},
                        ],
                    },
                    {"species": "Starmie", "condition": "0 fnt", "active": False, "moves": []},
                ],
                "boosts": {},
                "volatiles": [],
                "materializationBlockers": [],
                "toxicStage": 0,
                "sideConditions": {},
                "sideConditionSetTurns": {},
            },
            "p2": {
                "pokemon": [
                    {"species": "Snorlax", "condition": "73/100 par", "active": True},
                ],
                "boosts": {"atk": 1, "spe": -1},
                "volatiles": [],
                "materializationBlockers": [],
                "toxicStage": 0,
                "sideConditions": {"spikes": 2, "reflect": 1},
                "sideConditionSetTurns": {},
            },
        },
    }
    payload.update(overrides)
    return payload


def _override() -> BattleStartOverride:
    return BattleStartOverride(
        player_teams={
            "p1": pack_team(_team(_SWAMPERT, _STARMIE)),
            "p2": pack_team(_team(_SNORLAX, _STARMIE)),
        },
    )


class UnpackTeamTests(unittest.TestCase):
    def test_pack_unpack_round_trips_exactly(self) -> None:
        team = _team(_SWAMPERT, _SNORLAX, _STARMIE)
        packed = pack_team(team)
        unpacked = unpack_team(packed)
        self.assertEqual(pack_team(unpacked), packed)

    def test_unpack_defaults_match_showdown_conventions(self) -> None:
        mon = unpack_pokemon("Starmie||Leftovers|NaturalCure|surf||||||79|")
        self.assertEqual(mon.species, "Starmie")
        self.assertEqual(mon.level, 79)
        self.assertEqual(mon.evs, {s: 0 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
        self.assertEqual(mon.ivs, {s: 31 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
        default_level = unpack_pokemon("Starmie||||surf|||||||")
        self.assertEqual(default_level.level, 100)

    def test_unpack_partial_spreads(self) -> None:
        mon = unpack_pokemon("Swampert||||earthquake||,85,,85,,||,,30,,30,||84|")
        self.assertEqual(mon.evs["atk"], 85)
        self.assertEqual(mon.evs["hp"], 0)
        self.assertEqual(mon.ivs["def"], 30)
        self.assertEqual(mon.ivs["hp"], 31)


class BattleSpecConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dex = _dex()

    def test_constructs_midgame_world(self) -> None:
        world = battle_spec_from_payload(_payload(self.dex), _override(), dex=self.dex)
        spec = world.spec

        p1 = spec.side_one
        self.assertEqual(p1.active_index, 0)
        swampert = p1.pokemon[0]
        swampert_max = _maxhp(_SWAMPERT, self.dex)
        self.assertEqual(swampert.maxhp, swampert_max)
        self.assertEqual(swampert.hp, swampert_max - 40)
        self.assertEqual([m.id for m in swampert.moves], ["earthquake", "icebeam", "none", "none"])
        self.assertEqual(swampert.moves[0].pp, 12)  # request-known PP, not catalog PP
        self.assertTrue(swampert.moves[2].disabled)
        self.assertEqual(swampert.weight_kg, 81.9)
        self.assertEqual(p1.pokemon[1].hp, 0)  # fainted Starmie

        p2 = spec.side_two
        snorlax = p2.pokemon[0]
        self.assertEqual(snorlax.status, "paralyze")
        self.assertEqual(snorlax.hp, round(73 * snorlax.maxhp / 100))  # fraction scaled
        self.assertEqual(snorlax.moves[0].pp, (15 * 8) // 5)  # catalog randbats PP
        self.assertEqual(p2.boosts, {"attack": 1, "speed": -1})
        self.assertEqual(p2.side_conditions, {"spikes": 2, "reflect": 1})
        # Unrevealed sampled Starmie stays pristine.
        self.assertEqual(p2.pokemon[1].hp, p2.pokemon[1].maxhp)

        self.assertEqual(world.slot_sides, {"p1": "side_one", "p2": "side_two"})
        self.assertEqual(world.party_species["p2"], ("snorlax", "starmie"))

    def test_toxic_stage_maps_to_toxic_count(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["toxicStage"] = 3
        payload["sides"]["p2"]["pokemon"][0]["condition"] = "73/100 tox"
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertEqual(world.spec.side_two.side_conditions["toxic_count"], 3)
        self.assertEqual(world.spec.side_two.pokemon[0].status, "toxic")

    def test_ability_weather_is_indefinite(self) -> None:
        payload = _payload(self.dex, weather="sandstorm", weatherSetTurn=3, weatherFromAbility=True)
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertEqual(world.spec.weather, "sand")
        self.assertEqual(world.spec.weather_turns_remaining, -1)

    def test_manual_weather_counts_down(self) -> None:
        payload = _payload(self.dex, weather="raindance", weatherSetTurn=5, weatherFromAbility=False)
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertEqual(world.spec.weather, "rain")
        self.assertEqual(world.spec.weather_turns_remaining, 3)  # set turn 5, now turn 7

    def _assert_reason(self, payload, reason: str) -> None:
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertEqual(caught.exception.reason, reason)

    def test_fail_closed_taxonomy(self) -> None:
        self._assert_reason(_payload(self.dex, pendingBatonPassSides=["p2"]), "pending_baton_pass")
        self._assert_reason(_payload(self.dex, wishSetTurns={"p1": 6}), "wish_pending")
        self._assert_reason(_payload(self.dex, futureSight={"p1": 2, "p2": 0}), "future_sight_pending")
        self._assert_reason(_payload(self.dex, deferredOpponentActions={"p2": 3}), "deferred_opponent_action")
        self._assert_reason(_payload(self.dex, selfRequestKind="force-switch"), "boundary_not_move_request")

        sleeping = _payload(self.dex)
        sleeping["sides"]["p2"]["pokemon"][0]["condition"] = "73/100 slp"
        self._assert_reason(sleeping, "status_unsupported")

        substitute = _payload(self.dex)
        substitute["sides"]["p2"]["volatiles"] = ["Substitute"]
        self._assert_reason(substitute, "volatile_unsupported")

        blocked = _payload(self.dex)
        blocked["sides"]["p1"]["materializationBlockers"] = ["transform"]
        self._assert_reason(blocked, "materialization_blocker")

        stray = _payload(self.dex)
        stray["sides"]["p2"]["pokemon"].append({"species": "Blissey", "condition": "100/100", "active": False})
        self._assert_reason(stray, "public_species_not_in_world")

        expired = _payload(self.dex, weather="raindance", weatherSetTurn=1, weatherFromAbility=False)
        self._assert_reason(expired, "weather_turns_inconsistent")

    def test_self_maxhp_mismatch_fails_closed_instead_of_scaling(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p1"]["pokemon"][0]["condition"] = "200/999"
        self._assert_reason(payload, "self_maxhp_mismatch")

    def test_leechseed_volatile_is_supported(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["leechseed"]
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertEqual(world.spec.side_two.volatile_statuses, ("leechseed",))

    def test_anti_leakage_opponent_facts_come_only_from_inputs(self) -> None:
        """The constructed opponent side is a pure function of (payload, override).

        Hidden truths absent from both inputs must be absent from the output:
        the opponent's unrevealed move slots carry exactly the sampled moves,
        and HP derives only from the public fraction.
        """

        world = battle_spec_from_payload(_payload(self.dex), _override(), dex=self.dex)
        snorlax = world.spec.side_two.pokemon[0]
        sampled_moves = {"bodyslam", "shadowball"}
        real_moves = {m.id for m in snorlax.moves if m.id != "none"}
        self.assertEqual(real_moves, sampled_moves)
        # Same payload, different sampled world -> different constructed side,
        # proving the opponent data flows from the override alone.
        alt = BattleStartOverride(
            player_teams={
                "p1": _override().player_teams["p1"],
                "p2": pack_team(_team(
                    FixturePokemon(species="Snorlax", moves=("earthquake",), level=80,
                                   evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")}),
                    _STARMIE,
                )),
            },
        )
        alt_world = battle_spec_from_payload(_payload(self.dex), alt, dex=self.dex)
        alt_moves = {m.id for m in alt_world.spec.side_two.pokemon[0].moves if m.id != "none"}
        self.assertEqual(alt_moves, {"earthquake"})


class RealEngineSmokeTests(unittest.TestCase):
    def test_constructed_world_searches(self) -> None:
        try:
            import poke_engine
        except ImportError:
            self.skipTest("poke-engine wheel not installed")
        from pokezero.poke_engine_adapter import build_poke_engine_state

        world = battle_spec_from_payload(_payload(_dex()), _override(), dex=_dex())
        state = build_poke_engine_state(world.spec)
        result = poke_engine.monte_carlo_tree_search(state, 25, threads=1)
        self.assertGreater(result.total_visits, 0)
        choices = {entry.move_choice for entry in result.side_one}
        self.assertIn("earthquake", choices)


if __name__ == "__main__":
    unittest.main()
