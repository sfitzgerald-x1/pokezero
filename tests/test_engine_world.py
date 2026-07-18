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
    world_battle_spec,
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
            "hiddenpower": move("hiddenpower", 15),
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
        "selfTeamOrder": ["Swampert", "Starmie"],
        "selfActiveRequestState": {"trapped": False, "maybeTrapped": False, "maybeDisabled": False, "maybeLocked": False},
        "selfBenchedMoveHistory": False,
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
                "sideConditionSetTurns": {"reflect": 5},
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
        # Spikes is a layer count; Reflect is turns-remaining (set turn 5, now
        # turn 7, Gen 3 screens last 5 -> 3 left). Copying the presence flag
        # through would make the engine expire the screen after one turn.
        self.assertEqual(p2.side_conditions, {"spikes": 2, "reflect": 3})
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
        self._assert_reason(_payload(self.dex, wishSetTurns={"p1": 3}), "wish_turns_inconsistent")

    def test_pending_wish_constructs_with_engine_semantics(self) -> None:
        world = battle_spec_from_payload(
            _payload(self.dex, wishSetTurns={"p1": 6}), _override(), dex=self.dex
        )
        side = world.spec.side_one
        # Set turn 6, now turn 7 -> heals end of this turn (counter 1). The
        # engine ignores the amount (heals resolving active's maxhp/2); we
        # pass the active's value for forward compatibility.
        self.assertEqual(side.wish, (1, side.pokemon[side.active_index].maxhp // 2))
        self._assert_reason(_payload(self.dex, futureSight={"p1": 2, "p2": 0}), "future_sight_pending")
        self._assert_reason(_payload(self.dex, deferredOpponentActions={"p2": 3}), "deferred_opponent_action")
        self._assert_reason(_payload(self.dex, selfRequestKind="team-preview"), "boundary_not_move_request")

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

        trapped = _payload(self.dex)
        trapped["selfActiveRequestState"] = {"trapped": True}
        self._assert_reason(trapped, "self_request_state_unsupported")

        screen_no_set_turn = _payload(self.dex)
        screen_no_set_turn["sides"]["p2"]["sideConditionSetTurns"] = {}
        self._assert_reason(screen_no_set_turn, "side_condition_turns_unknown")

        screen_expired = _payload(self.dex)
        screen_expired["sides"]["p2"]["sideConditionSetTurns"] = {"reflect": 1}
        self._assert_reason(screen_expired, "side_condition_turns_inconsistent")

        order_mismatch = _payload(self.dex)
        order_mismatch["selfTeamOrder"] = ["Swampert", "Blissey"]
        self._assert_reason(order_mismatch, "self_world_mismatch")

    def test_benched_move_history_without_pp_snapshot_fails_closed(self) -> None:
        payload = _payload(self.dex)
        payload["selfBenchedMoveHistory"] = True
        # Starmie is benched (fainted here, but the PP rule is order-independent)
        # and its row carries no move states -> catalog PP would be a guess.
        self._assert_reason(payload, "self_pp_unknown")

    def test_benched_self_mon_without_history_uses_catalog_pp(self) -> None:
        world = battle_spec_from_payload(_payload(self.dex), _override(), dex=self.dex)
        starmie = world.spec.side_one.pokemon[1]
        self.assertEqual(starmie.moves[0].pp, (15 * 8) // 5)

    def test_self_maxhp_mismatch_fails_closed_instead_of_scaling(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p1"]["pokemon"][0]["condition"] = "200/999"
        self._assert_reason(payload, "self_maxhp_mismatch")

    def test_force_switch_boundary_constructs_with_flag(self) -> None:
        payload = _payload(self.dex, selfRequestKind="force-switch")
        starmie_max = _maxhp(_STARMIE, self.dex)
        payload["sides"]["p1"]["pokemon"][0]["condition"] = "0 fnt"
        payload["sides"]["p1"]["pokemon"][1]["condition"] = f"{starmie_max}/{starmie_max}"
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertTrue(world.spec.side_one.force_switch)
        self.assertFalse(world.spec.side_two.force_switch)
        self.assertEqual(world.spec.side_one.pokemon[0].hp, 0)

    def test_unown_letter_formes_collapse_to_base_species(self) -> None:
        unown_dex = _dex()
        unown_dex.species["unown"] = SpeciesInfo(
            id="unown", name="Unown", types=("psychic",),
            base_stats={"hp": 48, "atk": 72, "def": 48, "spa": 72, "spd": 48, "spe": 48},
            weight_kg=5.0,
        )
        payload = _payload(self.dex)
        payload["sides"]["p2"]["pokemon"] = [
            {"species": "Unown-C", "condition": "73/100", "active": True},
        ]
        override = BattleStartOverride(
            player_teams={
                "p1": _override().player_teams["p1"],
                "p2": pack_team(_team(
                    FixturePokemon(species="Unown-C", moves=("hiddenpower",), level=80,
                                   ivs={"hp": 31, "atk": 30, "def": 31, "spa": 30, "spd": 31, "spe": 31},
                                   evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")}),
                )),
            },
        )
        world = battle_spec_from_payload(payload, override, dex=unown_dex)
        self.assertEqual(world.spec.side_two.pokemon[0].id, "unown")
        self.assertEqual(world.party_species["p2"], ("unown",))

    def test_substitute_supported_only_with_approximation_flag(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Substitute"]
        self._assert_reason(payload, "volatile_unsupported")
        world = battle_spec_from_payload(
            payload, _override(), dex=self.dex, approximate_substitute_health=True
        )
        side = world.spec.side_two
        self.assertIn("substitute", side.volatile_statuses)
        self.assertEqual(side.substitute_health, side.pokemon[0].maxhp // 4)

    def test_sleep_approximation_flag(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["pokemon"][0]["condition"] = "73/100 slp"
        self._assert_reason(payload, "status_unsupported")
        world = battle_spec_from_payload(
            payload, _override(), dex=self.dex, approximate_sleep_turns=True
        )
        sleeper = world.spec.side_two.pokemon[0]
        self.assertEqual(sleeper.status, "sleep")
        self.assertEqual(sleeper.sleep_turns, 0)
        self.assertEqual(sleeper.rest_turns, 0)

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


class TransformAndEncoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dex = _dex()

    def _assert_reason(self, payload, reason, **kwargs) -> None:
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(payload, _override(), dex=self.dex, **kwargs)
        self.assertEqual(caught.exception.reason, reason)

    def test_blocked_slot_fails_closed(self) -> None:
        self._assert_reason(
            _payload(self.dex),
            "public_effect_blocked",
            blocked_slots={"p2": "active transformed into Snorlax"},
        )

    def test_self_moveset_mismatch_fails_closed(self) -> None:
        # A transformed self mon's request reports COPIED moves that are not
        # in the sampled (true) moveset -> must fail closed, never construct.
        payload = _payload(self.dex)
        payload["sides"]["p1"]["pokemon"][0]["moves"] = [
            {"id": "bodyslam", "pp": 15, "maxpp": 24, "disabled": False},
            {"id": "shadowball", "pp": 15, "maxpp": 24, "disabled": False},
        ]
        self._assert_reason(payload, "self_moveset_mismatch")

    def test_self_encore_derives_lock_from_disabled_pattern(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p1"]["volatiles"] = ["Encore"]
        payload["sides"]["p1"]["pokemon"][0]["moves"] = [
            {"id": "earthquake", "pp": 12, "maxpp": 16, "disabled": False},
            {"id": "icebeam", "pp": 16, "maxpp": 16, "disabled": True},
        ]
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        side = world.spec.side_one
        self.assertIn("encore", side.volatile_statuses)
        self.assertEqual(side.last_used_move, "move:0")  # earthquake slot
        self.assertEqual(dict(side.volatile_status_durations), {"encore": 1})

    def test_opponent_encore_uses_caller_supplied_move(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Encore"]
        world = battle_spec_from_payload(
            payload, _override(), dex=self.dex, encored_moves={"p2": "Body Slam"}
        )
        side = world.spec.side_two
        self.assertEqual(side.last_used_move, "move:0")  # bodyslam is snorlax slot 0
        self.assertIn("encore", side.volatile_statuses)

    def test_opponent_encore_without_move_fails_closed(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Encore"]
        self._assert_reason(payload, "encore_move_unknown")

    def test_encored_move_absent_from_sample_fails_closed(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Encore"]
        self._assert_reason(
            payload, "encore_move_unknown", encored_moves={"p2": "Hyper Beam"}
        )


@unittest.skipIf(
    not __import__("pathlib").Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/dist/sim/index.js").exists(),
    "requires a built local Showdown checkout",
)
class DittoTransformLiveTests(unittest.TestCase):
    """End-to-end fallback-detection edge case: a transformed Ditto must never
    construct as a silently wrong world (base stats + [transform] moveset)."""

    def test_transform_fails_closed_for_both_seats(self) -> None:
        from pokezero.dex import load_showdown_dex
        from pokezero.engine_search import EngineMctsPolicy, EngineMctsConfig
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

        root = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
        dex = load_showdown_dex(root)
        ditto = FixturePokemon(species="Ditto", moves=("Transform",), ability="Limber",
                               item="Quick Claw", level=100)
        lax = FixturePokemon(species="Snorlax", moves=("Body Slam", "Curse", "Rest", "Shadow Ball"),
                             ability="Immunity", item="Leftovers", level=80,
                             evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
        override = BattleStartOverride(player_teams={
            "p1": pack_team((ditto, _SWAMPERT)),
            "p2": pack_team((lax, _SWAMPERT)),
        })
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=root))
        try:
            env.reset_with_start_override(seed=99001, start_override=override)
            # Resolve one turn: Ditto transforms, Snorlax curses.
            actions = {}
            for player in env.requested_players():
                observation = env.observe(player)
                legal = [c for c in observation.metadata["action_candidates"] if c.get("legal")]
                want = "transform" if player == "p1" else "curse"
                pick = next((c for c in legal if c.get("kind") == "move" and want in str(c.get("move_id"))), legal[0])
                actions[player] = pick["action_index"]
            result = env.step(actions)
            self.assertIsNone(result.terminal)

            # Seat p1 (the transformed side itself): request now shows COPIED
            # moves; construction from the true world must fail closed.
            state_p1 = env.public_materialization_state("p1")
            with self.assertRaises(EngineWorldUnsupported) as caught:
                world_battle_spec(state_p1, override, dex=dex)
            self.assertEqual(caught.exception.reason, "self_moveset_mismatch")

            # Seat p2 (facing the transformed Ditto): the belief engine sees the
            # transform publicly; engine_search's signals must block the slot.
            observation_p2 = env.observe("p2")
            policy = EngineMctsPolicy(dex=dex, set_source=None, module=object(),
                                      config=EngineMctsConfig())

            context = type("Ctx", (), {
                "observation": observation_p2,
                "player_id": "p2",
                "public_materialization_state": env.public_materialization_state("p2"),
            })()
            blocked, _encored = policy._public_effect_signals(context)
            self.assertIn("p1", blocked)
            self.assertIn("transformed", blocked["p1"])
            state_p2 = env.public_materialization_state("p2")
            with self.assertRaises(EngineWorldUnsupported) as caught2:
                world_battle_spec(state_p2, override, dex=dex, blocked_slots=blocked)
            self.assertEqual(caught2.exception.reason, "public_effect_blocked")

            # F2 (review): the block must CLEAR after the transformed Ditto
            # switches out (gen3 transform reverts on switch) and back in.
            for _ in range(4):
                actions = {}
                requested = env.requested_players()
                for player in requested:
                    observation = env.observe(player)
                    legal = [c for c in observation.metadata["action_candidates"] if c.get("legal")]
                    switch = next((c for c in legal if c.get("kind") == "switch"), None)
                    pick = switch if (player == "p1" and switch is not None) else legal[0]
                    actions[player] = pick["action_index"]
                step_result = env.step(actions)
                if step_result.terminal is not None:
                    break
                observation_p2 = env.observe("p2")
                context_now = type("Ctx", (), {
                    "observation": observation_p2,
                    "player_id": "p2",
                    "public_materialization_state": env.public_materialization_state("p2"),
                })()
                blocked_now, _ = policy._public_effect_signals(context_now)
                belief_now = observation_p2.metadata.get("belief_view") or {}
                actives = [m for m in belief_now.get("opponent_pokemon") or [] if m.get("active")]
                if actives and "ditto" not in str(actives[0].get("species", "")).lower():
                    # Untransformed replacement active: block must be gone.
                    self.assertEqual(blocked_now, {})
                    break
        finally:
            env.close()

    def test_mirror_seat_p1_facing_transformed_p2_ditto(self) -> None:
        # F4 (review): the symmetric seat — p2 owns the Ditto, p1 must block p2.
        from pokezero.dex import load_showdown_dex
        from pokezero.engine_search import EngineMctsPolicy, EngineMctsConfig
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

        root = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
        dex = load_showdown_dex(root)
        ditto = FixturePokemon(species="Ditto", moves=("Transform",), ability="Limber",
                               item="Quick Claw", level=100)
        lax = FixturePokemon(species="Snorlax", moves=("Body Slam", "Curse", "Rest", "Shadow Ball"),
                             ability="Immunity", item="Leftovers", level=80,
                             evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
        override = BattleStartOverride(player_teams={
            "p1": pack_team((lax, _SWAMPERT)),
            "p2": pack_team((ditto, _SWAMPERT)),
        })
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=root))
        try:
            env.reset_with_start_override(seed=99002, start_override=override)
            actions = {}
            for player in env.requested_players():
                observation = env.observe(player)
                legal = [c for c in observation.metadata["action_candidates"] if c.get("legal")]
                want = "transform" if player == "p2" else "curse"
                pick = next((c for c in legal if c.get("kind") == "move" and want in str(c.get("move_id"))), legal[0])
                actions[player] = pick["action_index"]
            result = env.step(actions)
            self.assertIsNone(result.terminal)

            observation_p1 = env.observe("p1")
            policy = EngineMctsPolicy(dex=dex, set_source=None, module=object(),
                                      config=EngineMctsConfig())
            context = type("Ctx", (), {
                "observation": observation_p1,
                "player_id": "p1",
                "public_materialization_state": env.public_materialization_state("p1"),
            })()
            blocked, _ = policy._public_effect_signals(context)
            self.assertIn("p2", blocked)
            self.assertIn("transformed", blocked["p2"])
            with self.assertRaises(EngineWorldUnsupported) as caught:
                world_battle_spec(
                    env.public_materialization_state("p1"), override, dex=dex, blocked_slots=blocked
                )
            self.assertEqual(caught.exception.reason, "public_effect_blocked")
        finally:
            env.close()


class ShedinjaAndRechargeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dex = _dex()
        self.dex.species["shedinja"] = SpeciesInfo(
            id="shedinja", name="Shedinja", types=("bug", "ghost"),
            base_stats={"hp": 1, "atk": 90, "def": 45, "spa": 30, "spd": 30, "spe": 40},
            weight_kg=1.2,
        )

    def test_shedinja_maxhp_is_pinned_to_one(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["pokemon"] = [
            {"species": "Shedinja", "condition": "1/1", "active": True},
        ]
        override = BattleStartOverride(player_teams={
            "p1": _override().player_teams["p1"],
            "p2": pack_team(_team(
                FixturePokemon(species="Shedinja", moves=("shadowball",), level=100,
                               ability="Wonder Guard", item="Lum Berry",
                               evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")}),
            )),
        })
        world = battle_spec_from_payload(payload, override, dex=self.dex)
        shedinja = world.spec.side_two.pokemon[0]
        self.assertEqual((shedinja.hp, shedinja.maxhp), (1, 1))

    def test_recharging_slot_gets_mustrecharge_volatile(self) -> None:
        world = battle_spec_from_payload(
            _payload(self.dex), _override(), dex=self.dex, recharging_slots=("p2",)
        )
        self.assertIn("mustrecharge", world.spec.side_two.volatile_statuses)
        self.assertNotIn("mustrecharge", world.spec.side_one.volatile_statuses)


class BatonPassBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dex = _dex()

    def _bp_payload(self):
        payload = _payload(self.dex, selfRequestKind="force-switch", pendingBatonPassSides=["p1"])
        starmie_max = _maxhp(_STARMIE, self.dex)
        # Passer (Swampert) is leaving; Starmie is the live bench recipient.
        payload["sides"]["p1"]["pokemon"][1]["condition"] = f"{starmie_max}/{starmie_max}"
        payload["sides"]["p1"]["boosts"] = {"spa": 2}
        return payload

    def test_self_pending_baton_pass_constructs_and_populates_saved_move_field(self) -> None:
        # NOTE: the gen3 engine does not resolve the saved move after the pass
        # (probe-confirmed); this pins field population + determinism only.
        import random as _random

        world = battle_spec_from_payload(
            self._bp_payload(), _override(), dex=self.dex, rng=_random.Random(7)
        )
        p1 = world.spec.side_one
        p2 = world.spec.side_two
        self.assertTrue(p1.baton_passing)
        self.assertTrue(p1.force_switch)
        self.assertEqual(p1.boosts, {"special_attack": 2})
        self.assertTrue(p2.slow_uturn_move)
        self.assertIn(
            p2.switch_out_move_second_saved_move,
            {m.id for m in p2.pokemon[p2.active_index].moves if m.id != "none"},
        )
        # Seeded rng -> deterministic commitment sample.
        again = battle_spec_from_payload(
            self._bp_payload(), _override(), dex=self.dex, rng=_random.Random(7)
        )
        self.assertEqual(
            again.spec.side_two.switch_out_move_second_saved_move,
            p2.switch_out_move_second_saved_move,
        )

    def test_self_pending_without_rng_fails_closed(self) -> None:
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(self._bp_payload(), _override(), dex=self.dex)
        self.assertEqual(caught.exception.reason, "pending_baton_pass")

    def test_opponent_pending_still_fails_closed(self) -> None:
        import random as _random

        payload = _payload(self.dex, pendingBatonPassSides=["p2"])
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(payload, _override(), dex=self.dex, rng=_random.Random(1))
        self.assertEqual(caught.exception.reason, "pending_baton_pass")


if __name__ == "__main__":
    unittest.main()