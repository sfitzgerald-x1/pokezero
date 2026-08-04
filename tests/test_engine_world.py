"""Tests for the belief-world -> poke-engine constructor (v3 plan, track A)."""

from __future__ import annotations

import contextlib
from dataclasses import replace
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    _SUPPORTED_VOLATILES,
    _apply_forecast_types,
    _apply_transform,
    _require_world_reproduces_trap,
    _undischarged_materialization_blockers,
    battle_spec_from_payload,
    unpack_pokemon,
    unpack_team,
    world_battle_spec,
)
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.gen3_damage import gen3_hp_stat  # noqa: E402
from pokezero.poke_engine_adapter import (  # noqa: E402
    MoveSpec,
    PokemonSpec,
    PokeEngineMoveTrapUnsupportedError,
    SideSpec,
)
from pokezero.showdown_fixture import FixturePokemon, pack_pokemon, pack_team  # noqa: E402
from _showdown_root import showdown_root, showdown_root_str


@contextlib.contextmanager
def _move_trap_support(probe):
    """Swap the module-level move-trap capability probe for the duration.

    The real probe asks the installed native wheel whether it round-trips the
    TRAPPED volatile. These tests are about the WIRING either side of it, so they
    stub it; ``tests/test_engine_move_trap_wiring.py`` runs the real probe
    against a wheel built from the current patch list.
    """

    import pokezero.engine_world as engine_world

    original = engine_world.require_move_trap_support
    engine_world.require_move_trap_support = probe
    try:
        yield
    finally:
        engine_world.require_move_trap_support = original


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


def _override_with_trimmed_snorlax_hp() -> BattleStartOverride:
    trimmed = replace(_SNORLAX, evs={**(_SNORLAX.evs or {}), "hp": 0})
    return BattleStartOverride(
        player_teams={
            "p1": pack_team(_team(_SWAMPERT, _STARMIE)),
            "p2": pack_team(_team(trimmed, _STARMIE)),
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

    def test_public_gender_overrides_sampled_world_gender(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p1"]["pokemon"][0]["details"] = "Swampert, L84, M"
        payload["sides"]["p2"]["pokemon"][0]["details"] = "Snorlax, L80, F"
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)

        self.assertEqual(world.spec.side_one.pokemon[0].gender, "M")
        self.assertEqual(world.spec.side_two.pokemon[0].gender, "F")

    def test_removed_item_species_clears_only_the_named_mon(self) -> None:
        # Knock Off removal: the sampled set's item is the battle-START
        # assignment; the CURRENT public state is "holds nothing". The world
        # clears exactly the named mon's item — spread/stats stay the set's.
        world = battle_spec_from_payload(
            _payload(self.dex),
            _override(),
            dex=self.dex,
            removed_item_species={"p2": ("snorlax",)},
        )
        p2 = world.spec.side_two
        self.assertIsNone(p2.pokemon[0].item)  # knocked-off Snorlax
        self.assertEqual(p2.pokemon[1].item, "leftovers")  # backline untouched
        self.assertEqual(world.spec.side_one.pokemon[0].item, "leftovers")  # self side untouched

    def test_removed_item_species_normalizes_display_names(self) -> None:
        world = battle_spec_from_payload(
            _payload(self.dex),
            _override(),
            dex=self.dex,
            removed_item_species={"p2": ("Snorlax",)},
        )
        self.assertIsNone(world.spec.side_two.pokemon[0].item)

    def test_current_item_override_substitutes_only_the_named_mon(self) -> None:
        # Trick-swap override: the mon publicly holds the protocol-named
        # CURRENT item; the world substitutes exactly it, on exactly that mon
        # — spread/moves/ability stay the sampled assignment's (Trick moves
        # only the item).
        world = battle_spec_from_payload(
            _payload(self.dex),
            _override(),
            dex=self.dex,
            current_item_overrides={"p2": {"snorlax": "petayaberry"}},
        )
        p2 = world.spec.side_two
        self.assertEqual(p2.pokemon[0].item, "petayaberry")  # Tricked Snorlax
        self.assertEqual(p2.pokemon[1].item, "leftovers")  # backline untouched
        self.assertEqual(world.spec.side_one.pokemon[0].item, "leftovers")  # self side untouched

    def test_current_item_override_applies_to_the_self_slot_too(self) -> None:
        # The exchange's other half: OUR mon now holds the opponent's item;
        # the self side's packed team is the battle-start assignment, so it
        # needs the same substitution.
        world = battle_spec_from_payload(
            _payload(self.dex),
            _override(),
            dex=self.dex,
            current_item_overrides={"p1": {"swampert": "choiceband"}},
        )
        self.assertEqual(world.spec.side_one.pokemon[0].item, "choiceband")
        self.assertEqual(world.spec.side_two.pokemon[0].item, "leftovers")

    def test_current_item_override_normalizes_display_names(self) -> None:
        world = battle_spec_from_payload(
            _payload(self.dex),
            _override(),
            dex=self.dex,
            current_item_overrides={"p2": {"Snorlax": "Petaya Berry"}},
        )
        self.assertEqual(world.spec.side_two.pokemon[0].item, "petayaberry")

    def test_conflicting_removal_and_override_fails_closed(self) -> None:
        # "Holds nothing" and "holds X" for the same mon is contradictory
        # belief state: never guess — a wrong item in a searched world is
        # silent wrongness.
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(
                _payload(self.dex),
                _override(),
                dex=self.dex,
                removed_item_species={"p2": ("snorlax",)},
                current_item_overrides={"p2": {"snorlax": "petayaberry"}},
            )
        self.assertEqual(caught.exception.reason, "item_state_conflict")

    def test_toxic_stage_maps_to_toxic_count(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["toxicStage"] = 3
        payload["sides"]["p2"]["pokemon"][0]["condition"] = "73/100 tox"
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertEqual(world.spec.side_two.side_conditions["toxic_count"], 3)
        self.assertEqual(world.spec.side_two.pokemon[0].status, "toxic")

    def test_toxic_stage_zero_is_a_valid_first_residual_counter(self) -> None:
        # The replay/materialization seam admits this only for a post-upkeep
        # poisoned replacement. Once admitted, the Rust world convention is
        # the same: no stored count means the next residual is Toxic stage 1.
        payload = _payload(self.dex)
        payload["sides"]["p2"]["toxicStage"] = 0
        payload["sides"]["p2"]["pokemon"][0]["condition"] = "73/100 tox"
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertEqual(world.spec.side_two.pokemon[0].status, "toxic")
        self.assertNotIn("toxic_count", world.spec.side_two.side_conditions)

    def test_toxic_stage_fifteen_or_sentinel_fails_closed(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["pokemon"][0]["condition"] = "73/100 tox"
        payload["sides"]["p2"]["toxicStage"] = 15
        self._assert_reason(payload, "toxic_stage_unknown")

        payload["sides"]["p2"]["toxicStage"] = 16
        self._assert_reason(payload, "toxic_stage_unknown")

    def test_active_toxic_requires_explicit_public_counter(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["pokemon"][0]["condition"] = "73/100 tox"
        payload["sides"]["p2"]["toxicStage"] = None
        self._assert_reason(payload, "toxic_stage_unknown")

    def test_non_toxic_active_rejects_a_nonzero_toxic_counter(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["toxicStage"] = 2
        self._assert_reason(payload, "toxic_stage_inconsistent")

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
        # A fainted active cannot carry a stale Substitute into the forced
        # replacement world, even if the preceding snapshot had one.
        payload["sides"]["p1"]["substituteHealthState"] = "absent"
        payload["sides"]["p1"]["substituteDepletion"] = None
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertTrue(world.spec.side_one.force_switch)
        self.assertFalse(world.spec.side_two.force_switch)
        self.assertEqual(world.spec.side_one.pokemon[0].hp, 0)
        self.assertNotIn("substitute", world.spec.side_one.volatile_statuses)

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
        # party_species keeps the sampled team's OWN id (protocol/request
        # convention) — only the ENGINE-facing spec id collapses. The leaf
        # path's ctx contract needs the request-convention key ("unownc"),
        # and the self_world_mismatch guard compares request ids against
        # exactly this surface (seed-7001 bench repro).
        self.assertEqual(world.party_species["p2"], ("unownc",))

    def test_substitute_requires_publicly_known_health_state(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Substitute"]
        self._assert_reason(payload, "volatile_unsupported")
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(
                payload, _override(), dex=self.dex, approximate_substitute_health=True
            )
        self.assertEqual(caught.exception.reason, "substitute_health_provenance_contradiction")

        payload["sides"]["p2"]["substituteHealthState"] = "full"
        world = battle_spec_from_payload(
            payload, _override(), dex=self.dex, approximate_substitute_health=True
        )
        side = world.spec.side_two
        self.assertIn("substitute", side.volatile_statuses)
        self.assertEqual(side.substitute_health, side.pokemon[0].maxhp // 4)

        payload["sides"]["p2"]["substituteHealthState"] = "exact"
        payload["sides"]["p2"]["substituteDepletion"] = 7
        exact = battle_spec_from_payload(
            payload, _override(), dex=self.dex, approximate_substitute_health=True
        ).spec.side_two
        self.assertEqual(exact.substitute_health, exact.pokemon[0].maxhp // 4 - 7)

        payload["sides"]["p2"]["volatiles"] = []
        payload["sides"]["p2"]["substituteHealthState"] = "broken"
        payload["sides"]["p2"]["substituteDepletion"] = None
        broken = battle_spec_from_payload(
            payload, _override(), dex=self.dex, approximate_substitute_health=True
        ).spec.side_two
        self.assertNotIn("substitute", broken.volatile_statuses)
        self.assertEqual(broken.substitute_health, 0)

    def test_active_substitute_invalid_provenance_is_a_terminal_contradiction(self) -> None:
        for state, depletion in (
            (None, None),
            ("", None),
            ("absent", None),
            ("broken", None),
            ("UNKNOWN", None),
            ("arbitrary", None),
            ("exact", None),
            ("exact", 0),
            ("exact", -1),
            ("exact", True),
        ):
            with self.subTest(state=state, depletion=depletion):
                payload = _payload(self.dex)
                payload["sides"]["p2"]["volatiles"] = ["Substitute"]
                if state is not None:
                    payload["sides"]["p2"]["substituteHealthState"] = state
                payload["sides"]["p2"]["substituteDepletion"] = depletion
                with self.assertRaises(EngineWorldUnsupported) as caught:
                    battle_spec_from_payload(
                        payload, _override(), dex=self.dex, approximate_substitute_health=True
                    )
                self.assertEqual(
                    caught.exception.reason,
                    "substitute_health_provenance_contradiction",
                )

    def test_substitute_state_depletion_pair_matrix(self) -> None:
        cases = (
            # Active canonical pairs.
            (True, "full", None, "built"),
            (True, "full", 0, "built"),
            (True, "unknown", None, "unknown"),
            (True, "unknown", 0, "unknown"),
            (True, "exact", 7, "built"),
            # Active non-canonical companions and states.
            (True, "full", 50, "contradiction"),
            (True, "unknown", 50, "contradiction"),
            (True, "full", -1, "contradiction"),
            (True, "unknown", -1, "contradiction"),
            (True, "full", "0", "contradiction"),
            (True, "unknown", "0", "contradiction"),
            (True, "full", 0.0, "contradiction"),
            (True, "unknown", 0.0, "contradiction"),
            (True, "full", False, "contradiction"),
            (True, "unknown", False, "contradiction"),
            (True, "exact", None, "contradiction"),
            (True, "exact", 0, "contradiction"),
            (True, "exact", -1, "contradiction"),
            (True, "exact", "7", "contradiction"),
            (True, "exact", 7.0, "contradiction"),
            (True, "exact", True, "contradiction"),
            (True, "absent", None, "contradiction"),
            (True, "broken", None, "contradiction"),
            # Inactive canonical pairs require no depletion.
            (False, None, None, "built"),
            (False, "", None, "built"),
            (False, "absent", None, "built"),
            (False, "broken", None, "built"),
            (False, None, 0, "contradiction"),
            (False, "absent", 0, "contradiction"),
            (False, "broken", 50, "contradiction"),
            (False, "absent", "0", "contradiction"),
            (False, "broken", False, "contradiction"),
            (False, "full", None, "contradiction"),
            (False, "unknown", None, "contradiction"),
            (False, "exact", 7, "contradiction"),
        )
        for active, state, depletion, outcome in cases:
            with self.subTest(
                active=active,
                state=state,
                depletion=depletion,
                outcome=outcome,
            ):
                payload = _payload(self.dex)
                payload["sides"]["p2"]["volatiles"] = ["Substitute"] if active else []
                if state is not None:
                    payload["sides"]["p2"]["substituteHealthState"] = state
                payload["sides"]["p2"]["substituteDepletion"] = depletion
                if outcome == "built":
                    battle_spec_from_payload(
                        payload,
                        _override(),
                        dex=self.dex,
                        approximate_substitute_health=True,
                    )
                    continue
                with self.assertRaises(EngineWorldUnsupported) as caught:
                    battle_spec_from_payload(
                        payload,
                        _override(),
                        dex=self.dex,
                        approximate_substitute_health=True,
                    )
                expected = (
                    "substitute_health_unknown"
                    if outcome == "unknown"
                    else "substitute_health_provenance_contradiction"
                )
                self.assertEqual(caught.exception.reason, expected)

    def test_exact_depletion_is_applied_relative_to_sampled_max_hp(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["pokemon"][0]["condition"] = "387/387"
        payload["sides"]["p2"]["volatiles"] = ["Substitute"]
        payload["sides"]["p2"]["substituteHealthState"] = "exact"
        payload["sides"]["p2"]["substituteDepletion"] = 50

        side = battle_spec_from_payload(
            payload,
            _override_with_trimmed_snorlax_hp(),
            dex=self.dex,
            approximate_substitute_health=True,
        ).spec.side_two

        self.assertEqual(side.pokemon[0].maxhp, 370)
        self.assertEqual(side.substitute_health, 42)

    def test_exact_depletion_rejects_sample_that_could_not_have_survived(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["pokemon"][0]["condition"] = "387/387"
        payload["sides"]["p2"]["volatiles"] = ["Substitute"]
        payload["sides"]["p2"]["substituteHealthState"] = "exact"
        # The replay's 96 HP Substitute survives at 4; the sampled world's
        # 92 HP Substitute could not still be active.
        payload["sides"]["p2"]["substituteDepletion"] = 92
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(
                payload,
                _override_with_trimmed_snorlax_hp(),
                dex=self.dex,
                approximate_substitute_health=True,
            )
        self.assertEqual(caught.exception.reason, "substitute_depletion_world_incompatible")

    def test_substitute_unknown_after_public_hit_fails_closed(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Substitute"]
        payload["sides"]["p2"]["substituteHealthState"] = "unknown"
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(
                payload, _override(), dex=self.dex, approximate_substitute_health=True
            )
        self.assertEqual(caught.exception.reason, "substitute_health_unknown")

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

    def test_flashfire_volatile_is_supported_both_seats(self) -> None:
        # The parser sets flashfire on the public ``-start`` line and clears it
        # on ``-end``/switch, both seats; the engine models the volatile as
        # until-switch (probe-verified). No duration state needed — a pure
        # allow-list pass-through, and the volatile must land in the SideSpec
        # so the engine applies the 1.5x own-fire boost.
        payload = _payload(self.dex)
        payload["sides"]["p1"]["volatiles"] = ["flashfire"]
        payload["sides"]["p2"]["volatiles"] = ["flashfire"]
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertIn("flashfire", world.spec.side_one.volatile_statuses)
        self.assertIn("flashfire", world.spec.side_two.volatile_statuses)

    def test_attract_volatile_is_supported_both_seats(self) -> None:
        # Gen 3 infatuation runs until switch / source-leave (no countdown), so
        # like flashfire it is a pure allow-list pass-through — no duration
        # state. The volatile must land in the SideSpec so the patched engine
        # (poke-engine-gen3-attract.patch) prices the 50%-per-turn move
        # immobilization; pre-fix an attracted seat walled with
        # ``volatile_unsupported: attract``.
        payload = _payload(self.dex)
        payload["sides"]["p1"]["volatiles"] = ["attract"]
        payload["sides"]["p2"]["volatiles"] = ["attract"]
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertIn("attract", world.spec.side_one.volatile_statuses)
        self.assertIn("attract", world.spec.side_two.volatile_statuses)

    def test_trapped_volatile_is_supported_when_the_wheel_has_the_patch(self) -> None:
        # Showdown's move-trap (Mean Look / Spider Web / Block). Like flashfire
        # and attract this is a pure allow-list pass-through — the gen3 trap has
        # no duration and no residual, it simply lasts until the trapper leaves —
        # but it is gated on the installed wheel actually carrying
        # poke-engine-gen3-move-trapping.patch, so the probe is stubbed here and
        # exercised for real against the patched wheel in
        # tests/test_engine_move_trap_wiring.py.
        payload = _payload(self.dex)
        payload["sides"]["p1"]["volatiles"] = ["trapped"]
        payload["sides"]["p2"]["volatiles"] = ["trapped"]
        with _move_trap_support(lambda engine=None: None):
            world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertIn("trapped", world.spec.side_one.volatile_statuses)
        self.assertIn("trapped", world.spec.side_two.volatile_statuses)

    def test_move_trap_satisfies_the_request_trap_check_instead_of_falling_back(self) -> None:
        # The whole point of the wiring. A Mean Look / Spider Web turn discloses
        # ``trapped: true`` on OUR request, and _require_world_reproduces_trap
        # used to find no modelled cause for it — no trapping ability, no
        # partial trap — and raise self_request_state_unsupported, which fell the
        # entire root back to a non-search choice. The volatile is the cause, so
        # the check must now pass on it alone. The control below (same payload,
        # volatile removed) still fails closed, so this is the volatile doing the
        # work rather than the check going soft.
        payload = _payload(self.dex)
        payload["selfActiveRequestState"] = {"trapped": True}
        payload["sides"]["p1"]["volatiles"] = ["trapped"]
        with _move_trap_support(lambda engine=None: None):
            world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertIn("trapped", world.spec.side_one.volatile_statuses)

        payload["sides"]["p1"]["volatiles"] = []
        self._assert_reason(payload, "self_request_state_unsupported")

    def test_trapped_volatile_fails_loud_on_a_wheel_without_the_patch(self) -> None:
        # The stale-wheel path. An unpatched binding resolves the unknown TRAPPED
        # token to NONE and drops it silently, which would be strictly worse than
        # the fallback this replaces: search would hand the trapped seat its
        # switch options back. Construction must fail closed and name the patch.
        def _stale(engine=None):
            raise PokeEngineMoveTrapUnsupportedError(
                "the installed engine dropped the TRAPPED volatile instead of "
                "round-tripping it. Rebuild with scripts/setup_poke_engine.sh "
                "(third_party/poke-engine-gen3-move-trapping.patch)."
            )

        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["trapped"]
        with _move_trap_support(_stale):
            with self.assertRaises(PokeEngineMoveTrapUnsupportedError) as caught:
                battle_spec_from_payload(payload, _override(), dex=self.dex)
        message = str(caught.exception)
        self.assertIn("move-trapping.patch", message)
        self.assertIn("setup_poke_engine.sh", message)

    def test_other_unsupported_volatiles_still_fail_closed(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["confusion"]
        self._assert_reason(payload, "volatile_unsupported")

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
    not (showdown_root() / "dist/sim/index.js").exists(),
    "requires a built local Showdown checkout",
)
class DittoTransformLiveTests(unittest.TestCase):
    """End-to-end Transform against the REAL sim, from both seats.

    A transformed Ditto must never construct as a silently wrong world (base
    stats + the [transform] moveset). It used to fail closed to guarantee that;
    now the copied form is baked into the active's spec instead, so the world is
    both searchable AND correct -- these assert the second half.
    """

    def test_transform_fails_closed_for_both_seats(self) -> None:
        from pokezero.dex import load_showdown_dex
        from pokezero.engine_search import EngineMctsPolicy, EngineMctsConfig
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

        root = showdown_root_str()
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

            # Seat p1 (the transformed side itself): the request shows COPIED
            # moves while the sampled world still holds Ditto's real one. That
            # used to raise self_moveset_mismatch; the overlay now reconciles it.
            state_p1 = env.public_materialization_state("p1")
            world_p1 = world_battle_spec(
                state_p1, override, dex=dex, transformed_slots={"p1": "Snorlax"}
            )
            side_p1 = getattr(world_p1.spec, world_p1.slot_sides["p1"])
            copied = side_p1.pokemon[side_p1.active_index]
            self.assertEqual(copied.id, "snorlax")
            self.assertIn("bodyslam", [m.id for m in copied.moves])
            self.assertTrue(all(m.pp <= 5 for m in copied.moves if m.id != "none"))

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
            blocked, _encored, _removed, _overridden, transformed = (
                policy._public_effect_signals(context)
            )
            # The transform is reported as an expressible signal, not a block.
            self.assertEqual(blocked, {})
            self.assertEqual(transformed.get("p1", "").lower(), "snorlax")
            state_p2 = env.public_materialization_state("p2")
            world_p2 = world_battle_spec(
                state_p2, override, dex=dex, blocked_slots=blocked,
                transformed_slots=transformed,
            )
            foe = getattr(world_p2.spec, world_p2.slot_sides["p1"])
            foe_active = foe.pokemon[foe.active_index]
            self.assertEqual(foe_active.id, "snorlax")
            # HP stays Ditto's own -- the copy never touches it.
            ditto_side = getattr(world_p2.spec, world_p2.slot_sides["p1"])
            self.assertLess(foe_active.maxhp, max(m.maxhp for m in ditto_side.pokemon) + 1)

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
                blocked_now, _, _, _, _tf_now = policy._public_effect_signals(context_now)
                belief_now = observation_p2.metadata.get("belief_view") or {}
                actives = [m for m in belief_now.get("opponent_pokemon") or [] if m.get("active")]
                if actives and "ditto" not in str(actives[0].get("species", "")).lower():
                    # Untransformed replacement active: block must be gone.
                    self.assertEqual(blocked_now, {})
                    break
        finally:
            env.close()

    def test_world_still_builds_after_the_transformed_ditto_pivots_out(self) -> None:
        """The window #872 left open: every decision AFTER the Ditto switches out.

        Gen 3 ends Transform on switch-out, so ``transformed_slots`` empties and
        the moveset guard is armed again -- but the actor's cached move state was
        snapshotted WHILE transformed, so the benched Ditto's payload row still
        advertises the copied moveset. That stale row failed the guard for the
        rest of the battle, which is why self_moveset_mismatch stayed the
        dominant fallback even after Transform became expressible.
        """
        from pokezero.dex import load_showdown_dex
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

        # `showdown_root_str()`, not a hardcoded path. The cherry-picked commit predates
        # tests/_showdown_root.py, which exists precisely because ~23 files each wrote a
        # maintainer home directory into this PUBLIC repo -- and
        # test_public_invariant.py::test_no_internal_identifiers_in_tracked_files fails on
        # exactly that, which is how this was caught.
        root = showdown_root_str()
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
            actions = {}
            for player in env.requested_players():
                obs = env.observe(player)
                legal = [c for c in obs.metadata["action_candidates"] if c.get("legal")]
                want = "transform" if player == "p1" else "curse"
                pick = next((c for c in legal if c.get("kind") == "move"
                             and want in str(c.get("move_id"))), legal[0])
                actions[player] = pick["action_index"]
            self.assertIsNone(env.step(actions).terminal)

            # Pivot the Ditto out: Transform ends, the copied form is gone.
            actions = {}
            for player in env.requested_players():
                obs = env.observe(player)
                legal = [c for c in obs.metadata["action_candidates"] if c.get("legal")]
                switch = next((c for c in legal if c.get("kind") == "switch"), None)
                actions[player] = (switch or legal[0])["action_index"]
            self.assertIsNone(env.step(actions).terminal)

            # With no transform in play the world must construct cleanly. Before
            # the stale-snapshot fix this raised self_moveset_mismatch here and
            # on every later decision.
            world_battle_spec(env.public_materialization_state("p1"), override, dex=dex)
        finally:
            env.close()

    def test_mirror_seat_p1_facing_transformed_p2_ditto(self) -> None:
        # F4 (review): the symmetric seat — p2 owns the Ditto, p1 must block p2.
        from pokezero.dex import load_showdown_dex
        from pokezero.engine_search import EngineMctsPolicy, EngineMctsConfig
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

        root = showdown_root_str()
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
            blocked, _, _, _, transformed = policy._public_effect_signals(context)
            self.assertEqual(blocked, {})
            self.assertEqual(transformed.get("p2", "").lower(), "snorlax")
            world = world_battle_spec(
                env.public_materialization_state("p1"), override, dex=dex,
                blocked_slots=blocked, transformed_slots=transformed,
            )
            foe = getattr(world.spec, world.slot_sides["p2"])
            self.assertEqual(foe.pokemon[foe.active_index].id, "snorlax")
        finally:
            env.close()


@unittest.skipIf(
    not (showdown_root() / "dist/sim/index.js").exists(),
    "requires a built local Showdown checkout",
)
class KnockOffRemovalLiveTests(unittest.TestCase):
    """End-to-end removal path against the REAL sim protocol: a public Knock
    Off must not wall world construction — the built world clears the item."""

    def test_knock_off_removal_constructs_with_cleared_item(self) -> None:
        from pokezero.dex import load_showdown_dex
        from pokezero.engine_search import EngineMctsPolicy, EngineMctsConfig
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

        root = showdown_root_str()
        dex = load_showdown_dex(root)
        ttar = FixturePokemon(species="Tyranitar", moves=("Knock Off", "Rock Slide"),
                              ability="Sand Stream", item="Leftovers", level=74,
                              evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
        lax = FixturePokemon(species="Snorlax", moves=("Body Slam", "Curse"),
                             ability="Immunity", item="Leftovers", level=80,
                             evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
        override = BattleStartOverride(player_teams={
            "p1": pack_team((ttar, _SWAMPERT)),
            "p2": pack_team((lax, _SWAMPERT)),
        })
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=root))
        try:
            env.reset_with_start_override(seed=99003, start_override=override)
            actions = {}
            for player in env.requested_players():
                observation = env.observe(player)
                legal = [c for c in observation.metadata["action_candidates"] if c.get("legal")]
                want = "knockoff" if player == "p1" else "curse"
                pick = next((c for c in legal if c.get("kind") == "move" and want in str(c.get("move_id"))), legal[0])
                actions[player] = pick["action_index"]
            result = env.step(actions)
            self.assertIsNone(result.terminal)

            observation_p1 = env.observe("p1")
            belief_view = observation_p1.metadata.get("belief_view") or {}
            lax_belief = next(
                m for m in belief_view.get("opponent_pokemon") or []
                if "snorlax" in str(m.get("species", "")).lower()
            )
            # The real |-enditem|...|[from] move: Knock Off| line must have set BOTH flags.
            self.assertTrue(lax_belief.get("item_mutated"))
            self.assertTrue(lax_belief.get("item_removed"))

            policy = EngineMctsPolicy(dex=dex, set_source=None, module=object(),
                                      config=EngineMctsConfig())
            context = type("Ctx", (), {
                "observation": observation_p1,
                "player_id": "p1",
                "public_materialization_state": env.public_materialization_state("p1"),
            })()
            blocked, _encored, removed, _overridden, _tf = policy._public_effect_signals(context)
            self.assertEqual(blocked, {})
            self.assertEqual(removed, {"p2": ("snorlax",)})

            # Construction goes through, with the knocked-off item cleared.
            world = world_battle_spec(
                env.public_materialization_state("p1"), override, dex=dex,
                blocked_slots=blocked, removed_item_species=removed,
            )
            p2_side = getattr(world.spec, world.slot_sides["p2"])
            snorlax = next(m for m in p2_side.pokemon if m.id == "snorlax")
            self.assertIsNone(snorlax.item)
            ttar_side = getattr(world.spec, world.slot_sides["p1"])
            self.assertEqual(next(m for m in ttar_side.pokemon if m.id == "tyranitar").item, "leftovers")
        finally:
            env.close()


@unittest.skipIf(
    not (showdown_root() / "dist/sim/index.js").exists(),
    "requires a built local Showdown checkout",
)
class TrickSwapOverrideLiveTests(unittest.TestCase):
    """End-to-end Trick-swap override against the REAL sim protocol: a public
    exchange must not wall world construction — the built world carries the
    protocol-confirmed CURRENT item on BOTH mons of the exchange."""

    def test_trick_exchange_constructs_with_both_current_items(self) -> None:
        from pokezero.dex import load_showdown_dex
        from pokezero.engine_search import EngineMctsPolicy, EngineMctsConfig
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

        root = showdown_root_str()
        dex = load_showdown_dex(root)
        zam = FixturePokemon(species="Alakazam", moves=("Trick", "Psychic"),
                             ability="Synchronize", item="Petaya Berry", level=80,
                             evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
        lax = FixturePokemon(species="Snorlax", moves=("Body Slam", "Curse"),
                             ability="Immunity", item="Leftovers", level=80,
                             evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
        override = BattleStartOverride(player_teams={
            "p1": pack_team((zam, _SWAMPERT)),
            "p2": pack_team((lax, _SWAMPERT)),
        })
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=root))
        try:
            env.reset_with_start_override(seed=99005, start_override=override)
            actions = {}
            for player in env.requested_players():
                observation = env.observe(player)
                legal = [c for c in observation.metadata["action_candidates"] if c.get("legal")]
                want = "trick" if player == "p1" else "curse"
                pick = next((c for c in legal if c.get("kind") == "move" and want in str(c.get("move_id"))), legal[0])
                actions[player] = pick["action_index"]
            result = env.step(actions)
            self.assertIsNone(result.terminal)

            observation_p1 = env.observe("p1")
            belief_view = observation_p1.metadata.get("belief_view") or {}
            lax_belief = next(
                m for m in belief_view.get("opponent_pokemon") or []
                if "snorlax" in str(m.get("species", "")).lower()
            )
            # The real |-item|...|[from] move: Trick| lines must have set the
            # swap flags AND the protocol-confirmed current item on both mons.
            self.assertTrue(lax_belief.get("item_mutated"))
            self.assertFalse(lax_belief.get("item_removed"))
            self.assertEqual(lax_belief.get("current_public_item"), "Petaya Berry")
            zam_belief = next(
                m for m in belief_view.get("self_pokemon") or []
                if "alakazam" in str(m.get("species", "")).lower()
            )
            self.assertTrue(zam_belief.get("item_mutated"))
            self.assertEqual(zam_belief.get("current_public_item"), "Leftovers")

            policy = EngineMctsPolicy(dex=dex, set_source=None, module=object(),
                                      config=EngineMctsConfig())
            context = type("Ctx", (), {
                "observation": observation_p1,
                "player_id": "p1",
                "public_materialization_state": env.public_materialization_state("p1"),
            })()
            blocked, _encored, removed, overridden, _tf = policy._public_effect_signals(context)
            self.assertEqual(blocked, {})
            self.assertEqual(removed, {})
            self.assertEqual(overridden, {
                "p2": {"snorlax": "petayaberry"},
                "p1": {"alakazam": "leftovers"},
            })

            # Construction goes through, with BOTH current items substituted.
            world = world_battle_spec(
                env.public_materialization_state("p1"), override, dex=dex,
                blocked_slots=blocked, removed_item_species=removed,
                current_item_overrides=overridden,
            )
            p2_side = getattr(world.spec, world.slot_sides["p2"])
            snorlax = next(m for m in p2_side.pokemon if m.id == "snorlax")
            self.assertEqual(snorlax.item, "petayaberry")
            p1_side = getattr(world.spec, world.slot_sides["p1"])
            alakazam = next(m for m in p1_side.pokemon if m.id == "alakazam")
            self.assertEqual(alakazam.item, "leftovers")
            # Exchange partners' teammates keep their sampled items.
            self.assertEqual(next(m for m in p2_side.pokemon if m.id == "swampert").item, "leftovers")
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


class ForecastRootTypeTests(unittest.TestCase):
    @staticmethod
    def _mon(species: str, ability: str, types: tuple[str, ...]) -> PokemonSpec:
        return PokemonSpec(
            id=species,
            level=80,
            types=types,
            hp=200,
            maxhp=200,
            attack=100,
            defense=100,
            special_attack=100,
            special_defense=100,
            speed=100,
            moves=(MoveSpec("tackle"),),
            ability=ability,
        )

    def test_active_forecast_type_is_latched_from_root_weather(self) -> None:
        castform = self._mon("castform", "forecast", ("Normal",))
        other = self._mon("snorlax", "immunity", ("Normal",))
        sides = {"p1": SideSpec((castform,)), "p2": SideSpec((other,))}

        self.assertEqual(_apply_forecast_types(sides, weather="rain")["p1"].pokemon[0].types, ("Water",))
        self.assertEqual(_apply_forecast_types(sides, weather="sun")["p1"].pokemon[0].types, ("Fire",))
        self.assertEqual(_apply_forecast_types(sides, weather="hail")["p1"].pokemon[0].types, ("Ice",))
        self.assertEqual(_apply_forecast_types(sides, weather="sand")["p1"].pokemon[0].types, ("Normal",))

    def test_air_lock_or_cloud_nine_suppresses_forecast(self) -> None:
        castform = self._mon("castform", "forecast", ("Normal",))
        for ability in ("airlock", "cloudnine"):
            with self.subTest(ability=ability):
                suppressor = self._mon("golduck", ability, ("Water",))
                sides = {"p1": SideSpec((castform,)), "p2": SideSpec((suppressor,))}
                self.assertEqual(
                    _apply_forecast_types(sides, weather="rain")["p1"].pokemon[0].types,
                    ("Normal",),
                )


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


class MaterializationBlockerDischargeTests(unittest.TestCase):
    """A payload blocker the CALLER has positively expressed is not a blocker.

    The payload producer and the search caller derive item state from the same
    belief engine but describe it differently: the producer emits a fail-closed
    ``item-state-*`` token, the caller emits ``removed_item_species`` /
    ``current_item_overrides`` that make the world exact. Vetoing on the token
    discarded worlds the caller had already resolved.
    """

    def test_removal_token_is_discharged_by_removed_item_species(self) -> None:
        self.assertEqual(
            _undischarged_materialization_blockers(
                ("item-state-removed:Snorlax",),
                removed_item_species=frozenset({"snorlax"}),
                item_overrides={},
            ),
            (),
        )

    def test_removal_token_still_blocks_without_the_caller_signal(self) -> None:
        self.assertEqual(
            _undischarged_materialization_blockers(
                ("item-state-removed:Snorlax",),
                removed_item_species=frozenset(),
                item_overrides={},
            ),
            ("item-state-removed:Snorlax",),
        )

    def test_discharge_is_species_scoped_not_blanket(self) -> None:
        # Resolving one mon's item must not excuse another mon's unresolved one.
        self.assertEqual(
            _undischarged_materialization_blockers(
                ("item-state-removed:Snorlax", "item-state-removed:Zapdos"),
                removed_item_species=frozenset({"snorlax"}),
                item_overrides={},
            ),
            ("item-state-removed:Zapdos",),
        )

    def test_unconfirmed_mutation_is_discharged_only_by_a_confirmed_override(self) -> None:
        self.assertEqual(
            _undischarged_materialization_blockers(
                ("item-state-unconfirmed:Gengar",),
                removed_item_species=frozenset(),
                item_overrides={"gengar": "choiceband"},
            ),
            (),
        )
        self.assertEqual(
            _undischarged_materialization_blockers(
                ("item-state-unconfirmed:Gengar",),
                removed_item_species=frozenset(),
                item_overrides={},
            ),
            ("item-state-unconfirmed:Gengar",),
        )

    def test_non_item_blockers_are_never_discharged(self) -> None:
        # Nothing in the caller's item signals expresses a Baton-Passed volatile
        # or an unknown Leech Seed source, so these must stay fail-closed.
        blockers = ("baton-pass:confusion", "leechseed-source-unknown", "item-state-ambiguous:Unown")
        self.assertEqual(
            _undischarged_materialization_blockers(
                blockers, removed_item_species=frozenset({"unown"}), item_overrides={}
            ),
            blockers,
        )


class SelfRequestStateFlagTests(unittest.TestCase):
    """Hidden-information request flags must not wall a belief searcher.

    Showdown's ``maybe*`` flags mean "we decline to tell you", not "you may not".
    Each sampled world commits to a concrete opponent hypothesis and derives the
    truth itself, so refusing to search on them forfeits exactly the positions
    where hidden information matters. ``trapped`` is a real disclosed constraint
    and must still fail closed.
    """

    def _payload_with_flags(self, flags: dict[str, bool]) -> dict:
        return {
            "turn": 3,
            "selfPlayer": "p1",
            "selfRequestKind": "move",
            "selfActiveRequestState": flags,
            "sides": {"p1": {}, "p2": {}},
        }

    def _reason_for(self, flags: dict[str, bool]) -> str | None:
        # The flag gate runs before any team/side work, so an empty override is
        # enough to reach it: a non-flag payload defect surfaces as some OTHER
        # reason, which is exactly what these assertions distinguish.
        packed = pack_team((_SWAMPERT,))
        override = BattleStartOverride(player_teams={"p1": packed, "p2": packed})
        try:
            battle_spec_from_payload(self._payload_with_flags(flags), override, dex=_dex())
        except EngineWorldUnsupported as error:
            return error.reason
        return None

    def test_maybe_flags_do_not_raise_self_request_state_unsupported(self) -> None:
        for flag in ("maybeTrapped", "maybeDisabled", "maybeLocked"):
            with self.subTest(flag=flag):
                self.assertNotEqual(
                    self._reason_for({flag: True}), "self_request_state_unsupported"
                )

    def test_trapped_is_no_longer_refused_up_front(self) -> None:
        # trapped is now VERIFIED against the built world rather than refused
        # on sight, so it must not short-circuit ahead of side construction.
        self.assertNotEqual(
            self._reason_for({"trapped": True}), "self_request_state_unsupported"
        )


class TransformOverlayTests(unittest.TestCase):
    """A Transformed active is re-expressed as the mon it copied.

    The vendored gen3 engine has no TRANSFORM volatile, but Transform needs no
    volatile to express: gen3 copies species, types, the five non-HP stats, the
    ability and the moveset at 5 PP, and leaves HP alone (transformInto). All of
    those are PokemonSpec fields, so the copy is baked into the active's spec.
    """

    def _mon(self, species, **kw):
        base = dict(
            id=species, level=100, hp=200, maxhp=200, attack=100, defense=100,
            special_attack=100, special_defense=100, speed=100, types=("normal",),
            ability="limber", item=None, moves=(MoveSpec(id="transform", pp=8),),
        )
        base.update(kw)
        return PokemonSpec(**base)

    def _sides(self):
        ditto = self._mon("ditto", hp=180, maxhp=180)
        donor = self._mon(
            "snorlax", attack=250, defense=180, special_attack=120,
            special_defense=220, speed=60, ability="immunity", types=("normal",),
            moves=(MoveSpec(id="bodyslam", pp=24), MoveSpec(id="curse", pp=16)),
        )
        return {"p1": SideSpec(pokemon=(ditto,)), "p2": SideSpec(pokemon=(donor,))}

    def test_copies_species_stats_ability_and_moves(self) -> None:
        copied = _apply_transform(self._sides(), {"p1": "Snorlax"}, dex=_dex())["p1"].pokemon[0]
        self.assertEqual(copied.id, "snorlax")
        self.assertEqual(copied.attack, 250)
        self.assertEqual(copied.speed, 60)
        self.assertEqual(copied.ability, "immunity")
        self.assertEqual([m.id for m in copied.moves], ["bodyslam", "curse"])

    def test_hp_is_the_transformer_s_own(self) -> None:
        # The single most important thing Gen 3 does NOT copy -- it is why a
        # transformed Ditto stays frail, and copying it would make the searched
        # world wrong in the one dimension that decides KOs.
        copied = _apply_transform(self._sides(), {"p1": "Snorlax"}, dex=_dex())["p1"].pokemon[0]
        self.assertEqual((copied.hp, copied.maxhp), (180, 180))

    def test_copied_moves_get_five_pp_from_the_catalog_not_the_donor(self) -> None:
        # Showdown reads the DEX base PP (transformInto), not the donor's
        # remaining PP. The distinction only shows on a donor whose current PP is
        # already below 5 -- with a full donor both formulas agree, which is why
        # the original version of this test pinned nothing.
        sides = self._sides()
        spent = replace(
            sides["p2"],
            pokemon=(replace(
                sides["p2"].pokemon[0],
                moves=(MoveSpec(id="bodyslam", pp=2), MoveSpec(id="shadowball", pp=0)),
            ),),
        )
        copied = _apply_transform(
            {"p1": sides["p1"], "p2": spent}, {"p1": "Snorlax"}, dex=_dex()
        )["p1"].pokemon[0]
        self.assertEqual([(m.id, m.pp) for m in copied.moves],
                         [("bodyslam", 5), ("shadowball", 5)])

    def test_a_drained_donor_move_still_copies_at_five_pp(self) -> None:
        # Showdown copies `Math.min(5, move.pp)` off the DEX entry -- the move's
        # BASE PP, not the donor's remaining. No gen3 move has a base PP below 5,
        # so a copied slot is always exactly 5 even off a donor down to its last
        # one. Reading the donor's REMAINING pp here under-filled the copy and
        # disagreed with what the engine writes when Transform is clicked.
        sides = self._sides()
        drained = replace(
            sides["p2"].pokemon[0],
            moves=(MoveSpec(id="bodyslam", pp=1), MoveSpec(id="curse", pp=3)),
        )
        sides["p2"] = replace(sides["p2"], pokemon=(drained,))
        copied = _apply_transform(sides, {"p1": "Snorlax"}, dex=_dex())["p1"].pokemon[0]
        self.assertEqual([move.pp for move in copied.moves], [5, 5])

    def test_donor_side_is_left_untouched(self) -> None:
        sides = self._sides()
        donor_before = sides["p2"].pokemon[0]
        self.assertEqual(_apply_transform(sides, {"p1": "Snorlax"}, dex=_dex())["p2"].pokemon[0], donor_before)

    def test_base_identity_stays_the_transformer_s_own(self) -> None:
        # The half that used to be silently wrong. PokemonSpec plumbed neither
        # base field, so the binding defaulted base_ability to the ability this
        # function had just copied FROM THE DONOR -- a transformed Ditto reverted
        # into a Ditto with Immunity, and its types into a flat Normal.
        copied = _apply_transform(self._sides(), {"p1": "Snorlax"}, dex=_dex())["p1"].pokemon[0]
        self.assertEqual(copied.ability, "immunity", "the CURRENT form is the donor's")
        self.assertEqual(copied.base_ability, "limber", "the BASE form is Ditto's own")
        self.assertEqual(copied.base_types, ("normal",))

    def test_a_non_normal_transformer_keeps_its_own_base_types(self) -> None:
        # Mew is the other gen3 randbats Transform carrier and it is PSYCHIC --
        # the case the binding's flat ("normal", "typeless") default got wrong.
        sides = self._sides()
        mew = replace(sides["p1"].pokemon[0], id="mew", types=("psychic",), ability="synchronize")
        sides["p1"] = replace(sides["p1"], pokemon=(mew,))
        copied = _apply_transform(sides, {"p1": "Snorlax"}, dex=_dex())["p1"].pokemon[0]
        self.assertEqual(copied.types, ("normal",), "current typing is the donor's")
        self.assertEqual(copied.base_types, ("psychic",))
        self.assertEqual(copied.base_ability, "synchronize")

    def test_pre_transform_carries_the_pre_copy_spec(self) -> None:
        sides = self._sides()
        before = sides["p1"].pokemon[0]
        copied = _apply_transform(sides, {"p1": "Snorlax"}, dex=_dex())["p1"].pokemon[0]
        self.assertEqual(copied.pre_transform, before)
        self.assertIsNone(before.pre_transform, "a base form has no base form")

    def test_both_revert_volatiles_are_set(self) -> None:
        # TRANSFORMED drives the species/stats/moveset restore off pre_transform;
        # TYPECHANGE is the existing arm that restores types -> base_types.
        side = _apply_transform(self._sides(), {"p1": "Snorlax"}, dex=_dex())["p1"]
        self.assertEqual(set(side.volatile_statuses), {"transformed", "typechange"})

    def test_existing_volatiles_are_preserved(self) -> None:
        sides = self._sides()
        sides["p1"] = replace(sides["p1"], volatile_statuses=("substitute",))
        side = _apply_transform(sides, {"p1": "Snorlax"}, dex=_dex())["p1"]
        self.assertEqual(side.volatile_statuses[0], "substitute")
        self.assertEqual(set(side.volatile_statuses), {"substitute", "transformed", "typechange"})

    def test_absent_donor_fails_closed(self) -> None:
        # The copied mon is not in this world's opposing party, so its stats and
        # moves would have to be invented -- exactly the silent wrongness the
        # constructor exists to refuse.
        with self.assertRaises(EngineWorldUnsupported) as caught:
            _apply_transform(self._sides(), {"p1": "Rapidash"}, dex=_dex())
        self.assertEqual(caught.exception.reason, "transform_unexpressible")


class BridgeBuiltTransformRevertsLiveTests(unittest.TestCase):
    """The whole point, against the REAL engine: a constructed Transform ENDS.

    Regression guard for the divergence this change fixes -- a bridge-built
    transformed Ditto used to switch out and come back as a Ditto holding the
    DONOR's ability (Immunity, not Limber) and a flat Normal typing, because
    nothing carried its base identity across.
    """

    def setUp(self) -> None:
        from pokezero.poke_engine_adapter import (
            PokeEngineTransformRevertUnsupportedError,
            require_pre_transform_support,
        )
        from pokezero.poke_engine_backend import probe_poke_engine

        if not probe_poke_engine().ready:
            self.skipTest("poke-engine is not installed/ready")
        # A wheel predating the Transform patch rejects `pre_transform` outright,
        # which would surface here as a TypeError ERROR rather than a skip.
        try:
            require_pre_transform_support()
        except PokeEngineTransformRevertUnsupportedError as exc:
            self.skipTest(str(exc))

    def _switch_out_fields(self, transformer: PokemonSpec):
        from pokezero.poke_engine_adapter import BattleSpec, build_poke_engine_state
        import poke_engine

        reserve = PokemonSpec(
            id="swampert", level=100, types=("water", "ground"), hp=200, maxhp=200,
            attack=100, defense=100, special_attack=100, special_defense=100, speed=100,
            ability="torrent", moves=(MoveSpec(id="surf", pp=16),),
        )
        # A donor with a distinctive ability AND typing, so a wrong revert shows.
        donor = PokemonSpec(
            id="gengar", level=100, types=("ghost", "poison"), hp=300, maxhp=300,
            attack=200, defense=180, special_attack=250, special_defense=175, speed=220,
            # Splash keeps the ply deterministic; Shadow Ball's 20% secondary
            # forks it three ways.
            ability="levitate", moves=(MoveSpec(id="splash", pp=40),),
        )
        sides = _apply_transform(
            {"p1": SideSpec(pokemon=(transformer, reserve)), "p2": SideSpec(pokemon=(donor,))},
            {"p1": "Gengar"},
            # `dex` became required when the copied moveset started taking its PP from
            # the catalog rather than the donor's remaining PP. These two tests landed on
            # main after that change was written, so they are the only call sites that
            # had not been updated.
            dex=_dex(),
        )
        state = build_poke_engine_state(
            BattleSpec(side_one=sides["p1"], side_two=sides["p2"])
        )
        branches = list(poke_engine.generate_instructions(state, "swampert", "splash"))
        self.assertEqual(len(branches), 1, "expected one deterministic branch")
        after = state.apply_instructions(branches[0])
        self.assertEqual(
            after.reverse_instructions(branches[0]).to_string(),
            state.to_string(),
            "the switch-out must invert exactly",
        )
        # Pokemon fields are comma-separated; slot 0 of side one is the mon that
        # just left. See Pokemon::serialize in the vendored engine's state.rs.
        return after.to_string().split("/")[0].split("=")[0].split(",")

    def _ditto(self):
        return PokemonSpec(
            id="ditto", level=100, types=("normal",), hp=180, maxhp=180,
            attack=132, defense=132, special_attack=132, special_defense=132, speed=132,
            ability="limber", moves=(MoveSpec(id="transform", pp=8),),
        )

    def test_ditto_reverts_with_its_own_ability_and_typing(self) -> None:
        fields = self._switch_out_fields(self._ditto())
        self.assertEqual(fields[0], "DITTO", "species must revert")
        self.assertEqual(fields[2:4], ["NORMAL", "TYPELESS"], "typing must revert")
        self.assertEqual(fields[8], "LIMBER", "ability must revert to the TRANSFORMER's own")
        self.assertNotEqual(fields[8], "LEVITATE", "the donor's ability must not survive")
        self.assertEqual(fields[13:18], ["132"] * 5, "stats must revert")
        self.assertEqual(fields[22], "TRANSFORM;false;8", "the base moveset must come back")

    def test_a_psychic_transformer_reverts_to_psychic(self) -> None:
        mew = PokemonSpec(
            id="mew", level=72, types=("psychic",), hp=250, maxhp=250,
            attack=160, defense=160, special_attack=160, special_defense=160, speed=160,
            ability="synchronize",
            moves=(MoveSpec(id="transform", pp=8), MoveSpec(id="softboiled", pp=16)),
        )
        fields = self._switch_out_fields(mew)
        self.assertEqual(fields[0], "MEW")
        self.assertEqual(fields[2:4], ["PSYCHIC", "TYPELESS"], "not the flat Normal default")
        self.assertEqual(fields[8], "SYNCHRONIZE")

    def test_a_state_from_an_older_caller_still_fails_soft(self) -> None:
        # No snapshot and no volatiles: the base form was never observed, so the
        # engine keeps the copied form rather than panicking or half-reverting.
        from pokezero.poke_engine_adapter import BattleSpec, build_poke_engine_state
        import poke_engine

        stuck = replace(
            self._ditto(), id="gengar", types=("ghost", "poison"), ability="levitate",
            attack=200, moves=(MoveSpec(id="splash", pp=5),),
        )
        reserve = PokemonSpec(
            id="swampert", level=100, types=("water", "ground"), hp=200, maxhp=200,
            attack=100, defense=100, special_attack=100, special_defense=100, speed=100,
            ability="torrent", moves=(MoveSpec(id="surf", pp=16),),
        )
        donor = PokemonSpec(
            id="gengar", level=100, types=("ghost", "poison"), hp=300, maxhp=300,
            attack=200, defense=180, special_attack=250, special_defense=175, speed=220,
            ability="levitate", moves=(MoveSpec(id="splash", pp=40),),
        )
        state = build_poke_engine_state(BattleSpec(
            side_one=SideSpec(pokemon=(stuck, reserve)),
            side_two=SideSpec(pokemon=(donor,)),
        ))
        branch = list(poke_engine.generate_instructions(state, "swampert", "splash"))[0]
        after = state.apply_instructions(branch)
        fields = after.to_string().split("/")[0].split("=")[0].split(",")
        self.assertEqual(fields[0], "GENGAR", "nothing to restore from, so nothing reverts")
        self.assertEqual(
            after.reverse_instructions(branch).to_string(), state.to_string()
        )


class TrapReproductionTests(unittest.TestCase):
    """``trapped`` is discharged only when the built world traps us too.

    Showdown discloses ``trapped`` when its cause is public, and the belief
    filter carries a revealed trapping ability into every sample -- so the world
    usually reproduces the trap and refusing to search was over-strict. It is
    not always reproduced, though: the vendored gen3 engine models no Mean Look
    / Block / Spider Web, and a move-trapped mon would be free to switch in
    search. These pin the engine's own conditions (gen3/state.rs Side::trapped).
    """

    def _mon(self, **kw):
        base = dict(
            id="snorlax", level=100, hp=300, maxhp=300, attack=200, defense=200,
            special_attack=150, special_defense=200, speed=100, types=("normal",),
            ability="immunity", item=None, moves=(MoveSpec(id="bodyslam", pp=24),),
        )
        base.update(kw)
        return PokemonSpec(**base)

    def _check(self, *, foe_ability: str, self_types=("normal",), self_ability="immunity",
               self_volatiles=()) -> str | None:
        sides = {
            "p1": SideSpec(
                pokemon=(self._mon(types=self_types, ability=self_ability),),
                volatile_statuses=tuple(self_volatiles),
            ),
            "p2": SideSpec(pokemon=(self._mon(id="dugtrio", ability=foe_ability),)),
        }
        try:
            _require_world_reproduces_trap(sides, dex=_dex(), self_player="p1")
        except EngineWorldUnsupported as error:
            return error.reason
        return None

    def test_shadow_tag_reproduces_the_trap(self) -> None:
        self.assertIsNone(self._check(foe_ability="shadowtag"))

    def test_arena_trap_reproduces_the_trap_for_a_grounded_target(self) -> None:
        self.assertIsNone(self._check(foe_ability="arenatrap"))

    def test_arena_trap_does_not_trap_a_flyer_or_a_levitator(self) -> None:
        self.assertEqual(
            self._check(foe_ability="arenatrap", self_types=("flying",)),
            "self_request_state_unsupported",
        )
        self.assertEqual(
            self._check(foe_ability="arenatrap", self_ability="levitate"),
            "self_request_state_unsupported",
        )

    def test_magnet_pull_traps_only_steel(self) -> None:
        self.assertIsNone(self._check(foe_ability="magnetpull", self_types=("steel",)))
        self.assertEqual(
            self._check(foe_ability="magnetpull"), "self_request_state_unsupported"
        )

    def test_partial_trap_on_our_own_side_reproduces_the_trap(self) -> None:
        self.assertIsNone(
            self._check(foe_ability="immunity", self_volatiles=("partiallytrapped",))
        )

    def test_move_trap_stays_fail_closed(self) -> None:
        # Mean Look / Block / Spider Web have no engine expression at all, so a
        # world built from them would let us switch out of a real trap.
        self.assertEqual(
            self._check(foe_ability="immunity"), "self_request_state_unsupported"
        )


class SupportedVolatileTests(unittest.TestCase):
    """Volatiles the vendored gen3 engine reproduces exactly are searchable."""

    def test_destiny_bond_and_perish_counters_are_exact(self) -> None:
        # Presence-only in the engine (Destiny Bond) or a fully public counter the
        # engine decrements itself (Perish Song) — no hidden component to guess.
        for volatile in ("destinybond", "perish1", "perish2", "perish3", "perish4"):
            with self.subTest(volatile=volatile):
                self.assertIn(volatile, _SUPPORTED_VOLATILES)

    def test_partial_trap_is_an_opt_in_approximation_not_an_exact_volatile(self) -> None:
        # The engine has no duration counter for it, so it must never be treated
        # as exactly expressible — it is gated behind a named approximation.
        self.assertNotIn("partiallytrapped", _SUPPORTED_VOLATILES)


if __name__ == "__main__":
    unittest.main()
