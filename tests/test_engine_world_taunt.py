"""A publicly-Taunted side must build a world, and build it with the RIGHT clock.

``taunt`` was in ``showdown.TRACKED_VOLATILES`` and absent from
``engine_world._SUPPORTED_VOLATILES``, so ``_build_side_spec`` raised
``volatile_unsupported`` and every decision at a Taunt boundary refused with
``no_worlds_constructed``. Measured on the committed corpus scenario
``struggle_taunt_stall``: 9 refusals, ``world_failures
{"volatile_unsupported: side 'p1': ['taunt']": 8}`` on each.

WHY IT IS EXPRESSIBLE AT ALL. The vendored gen3 engine models Taunt as a
volatile PLUS a ``volatile_status_durations.taunt`` counter
(``src/gen3/state.rs`` ``PokemonVolatileStatus::TAUNT``,
``src/gen3/generate_instructions.rs`` block ``10.15``), it filters Status moves
out of the taunted side's options (``Pokemon::add_available_moves``'s ``taunted``
arm), and it clears both on switch-out. Nothing about it is fabricated here.

WHY THE COUNTER IS 1 AND NOT 0. The engine's counter is TICKS ELAPSED: ``0``
ticks up, ``1`` removes the volatile, anything else panics. gen3 Showdown pins
the condition at ``duration: 2`` with ``durationCallback: undefined``, and the
modern ``onStart`` that bumps duration against an already-moved target is
overridden by gen4's plain one, which gen3 inherits. So the target is taunted for
exactly one request after Taunt lands whichever side moved first -- measured both
ways, not read off -- and the world must say "one tick elapsed".

Both halves of that are pinned below against the real wheel rather than argued:
the Status filter, and the expiry after exactly one end-of-turn.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    _SUPPORTED_VOLATILES,
    battle_spec_from_payload,
)
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.gen3_damage import gen3_hp_stat  # noqa: E402
from pokezero.poke_engine_adapter import build_poke_engine_state  # noqa: E402
from pokezero.showdown_fixture import FixturePokemon, pack_team  # noqa: E402

# Imported unguarded, for the same reason test_engine_world_encore_transform.py
# does it: this file's central claim is about what the NATIVE gen3 engine does
# with TAUNT. A skip on a missing wheel would retire that pin while still
# printing OK, and the fidelity half is the half worth having.
import poke_engine  # noqa: E402

_EVS = {stat: 85 for stat in ("hp", "atk", "def", "spa", "spd", "spe")}
# The Status move. Taunt's whole effect is to remove exactly these from the
# option set, so a fixture without one cannot tell the fix from a no-op.
_STATUS_MOVE = "toxic"
_ATTACK_MOVE = "icebeam"


def _move(move_id: str, pp: int, *, status: bool = False) -> MoveInfo:
    return MoveInfo(
        id=move_id,
        name=move_id,
        type="normal",
        category="status" if status else "special",
        gen3_category="status" if status else "special",
        base_power=0 if status else 95,
        accuracy=100.0,
        priority=0,
        recoil=False,
        drain=False,
        heal=False,
        status="tox" if status else None,
        boosts={},
        target="normal",
        selfdestruct=False,
        pp=pp,
    )


def _dex() -> ShowdownDex:
    return ShowdownDex(
        moves={
            _ATTACK_MOVE: _move(_ATTACK_MOVE, 10),
            _STATUS_MOVE: _move(_STATUS_MOVE, 10, status=True),
            "bodyslam": _move("bodyslam", 15),
            "surf": _move("surf", 15),
        },
        species={
            "blissey": SpeciesInfo(
                id="blissey", name="Blissey", types=("normal",),
                base_stats={"hp": 255, "atk": 10, "def": 10, "spa": 75, "spd": 135, "spe": 55},
                weight_kg=46.8,
            ),
            "starmie": SpeciesInfo(
                id="starmie", name="Starmie", types=("water", "psychic"),
                base_stats={"hp": 60, "atk": 75, "def": 85, "spa": 100, "spd": 85, "spe": 115},
                weight_kg=80.0,
            ),
            "snorlax": SpeciesInfo(
                id="snorlax", name="Snorlax", types=("normal",),
                base_stats={"hp": 160, "atk": 110, "def": 65, "spa": 65, "spd": 110, "spe": 30},
                weight_kg=460.0,
            ),
        },
        type_chart={},
    )


_BLISSEY = FixturePokemon(
    species="Blissey", moves=(_STATUS_MOVE, _ATTACK_MOVE), ability="Natural Cure",
    item="Leftovers", level=80, evs=dict(_EVS),
)
_STARMIE = FixturePokemon(
    species="Starmie", moves=("surf",), ability="Natural Cure",
    item="Leftovers", level=79, evs=dict(_EVS),
)
_SNORLAX = FixturePokemon(
    species="Snorlax", moves=("bodyslam",), ability="Immunity",
    item="Leftovers", level=80, evs=dict(_EVS),
)


def _override() -> BattleStartOverride:
    return BattleStartOverride(
        player_teams={
            "p1": pack_team((_BLISSEY, _STARMIE)),
            "p2": pack_team((_SNORLAX,)),
        }
    )


def _payload(dex: ShowdownDex, *, p1_volatiles=(), p2_volatiles=()):
    maxhp = gen3_hp_stat(255, 31, 85, 80)
    return {
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
        "selfTeamOrder": ["Blissey", "Starmie"],
        "selfActiveRequestState": {
            "trapped": False, "maybeTrapped": False,
            "maybeDisabled": False, "maybeLocked": False,
        },
        "selfBenchedMoveHistory": False,
        "sides": {
            "p1": {
                "pokemon": [
                    {
                        "species": "Blissey",
                        "condition": f"{maxhp - 40}/{maxhp}",
                        "active": True,
                        "moves": [
                            # Showdown reports the Status slot as `disabled` at a
                            # Taunt boundary; keep it usable here so the ENGINE's
                            # own Status filter is what the fidelity tests read,
                            # not a flag copied out of the request.
                            {"id": _STATUS_MOVE, "pp": 10, "maxpp": 10, "disabled": False},
                            {"id": _ATTACK_MOVE, "pp": 10, "maxpp": 10, "disabled": False},
                        ],
                    },
                    {"species": "Starmie", "condition": "0 fnt", "active": False, "moves": []},
                ],
                "boosts": {},
                "volatiles": list(p1_volatiles),
                "materializationBlockers": [],
                "toxicStage": 0,
                "sideConditions": {},
                "sideConditionSetTurns": {},
            },
            "p2": {
                "pokemon": [
                    {"species": "Snorlax", "condition": "73/100", "active": True},
                ],
                "boosts": {},
                "volatiles": list(p2_volatiles),
                "materializationBlockers": [],
                "toxicStage": 0,
                "sideConditions": {},
                "sideConditionSetTurns": {},
            },
        },
    }


class TauntWorldConstructionTests(unittest.TestCase):
    """The refusal is gone, and what replaces it carries the right counter."""

    def setUp(self) -> None:
        self.dex = _dex()

    def _world(self, **kwargs):
        return battle_spec_from_payload(_payload(self.dex, **kwargs), _override(), dex=self.dex)

    def test_a_taunted_self_side_builds_instead_of_refusing(self) -> None:
        # RED before the fix with reason `volatile_unsupported`.
        world = self._world(p1_volatiles=("taunt",))
        self.assertIn("taunt", world.spec.side_one.volatile_statuses)

    def test_a_taunted_opponent_side_builds_too(self) -> None:
        # The opponent seat takes a different path into `_build_side_spec`
        # (sampled moveset, no request rows), so it is pinned separately.
        world = self._world(p2_volatiles=("taunt",))
        self.assertIn("taunt", world.spec.side_two.volatile_statuses)

    def test_the_counter_is_seeded_at_exactly_one_tick_elapsed(self) -> None:
        # 1, not 0 and not 2: the engine's counter is ticks ELAPSED and gen3
        # Taunt has exactly one left at any request that can observe it.
        for slot, kwargs in (
            ("side_one", {"p1_volatiles": ("taunt",)}),
            ("side_two", {"p2_volatiles": ("taunt",)}),
        ):
            with self.subTest(slot=slot):
                side = getattr(self._world(**kwargs), "spec")
                self.assertEqual(
                    getattr(side, slot).volatile_status_durations.get("taunt"), 1
                )

    def test_an_untaunted_side_carries_no_taunt_counter(self) -> None:
        # A counter written unconditionally would be a silent lie on every other
        # world, and the engine panics on a nonzero counter without the volatile.
        world = self._world()
        for slot in ("side_one", "side_two"):
            with self.subTest(slot=slot):
                durations = getattr(world.spec, slot).volatile_status_durations
                self.assertNotIn("taunt", durations)

    def test_taunt_is_exact_and_not_behind_an_approximation_flag(self) -> None:
        # It is in the EXACT set, so it needs none of the `approximate_*` opt-ins
        # that confusion/yawn/partiallytrapped ride on -- gen3's clock is fixed,
        # so there is no hidden duration to approximate.
        self.assertIn("taunt", _SUPPORTED_VOLATILES)

    def test_an_unsupported_volatile_still_fails_closed(self) -> None:
        # Guards the shape of the change: `taunt` was admitted, the allow-list was
        # not opened.
        with self.assertRaises(EngineWorldUnsupported) as caught:
            self._world(p1_volatiles=("torment",))
        self.assertEqual(caught.exception.reason, "volatile_unsupported")


class TauntEngineFidelityTests(unittest.TestCase):
    """What the vendored gen3 wheel actually does with the world we just built."""

    def setUp(self) -> None:
        self.dex = _dex()
        self.state = build_poke_engine_state(
            battle_spec_from_payload(
                _payload(self.dex, p1_volatiles=("taunt",)), _override(), dex=self.dex
            ).spec
        )
        self.untaunted = build_poke_engine_state(
            battle_spec_from_payload(_payload(self.dex), _override(), dex=self.dex).spec
        )

    def test_the_taunted_side_is_not_offered_its_status_move(self) -> None:
        taunted = {r.move_choice for r in poke_engine.monte_carlo_tree_search(self.state, 40).side_one}
        free = {r.move_choice for r in poke_engine.monte_carlo_tree_search(self.untaunted, 40).side_one}
        self.assertIn(_STATUS_MOVE, free)
        self.assertNotIn(_STATUS_MOVE, taunted)
        # ...and it still gets to attack, so this is a filter and not a wipe.
        self.assertIn(_ATTACK_MOVE, taunted)

    def test_the_volatile_expires_after_exactly_one_end_of_turn(self) -> None:
        # Seeded at 1 -> the very next residual removes it, which is what "one
        # taunted request remains" means. Seeded at 0 it would only tick UP, i.e.
        # the searched mon stays locked for a turn Showdown has already freed.
        instructions = self._taunt_instructions(self.state)
        self.assertIn("RemoveVolatileStatus SideOne: TAUNT", instructions)

        zeroed = self._taunt_instructions(self._state_with_taunt_duration(0))
        self.assertNotIn("RemoveVolatileStatus SideOne: TAUNT", zeroed)
        self.assertIn("ChangeVolatileStatusDuration SideOne TAUNT: 1", zeroed)

    def _state_with_taunt_duration(self, duration: int):
        """The same world, built with a different seeded counter, for contrast."""
        world = battle_spec_from_payload(
            _payload(self.dex, p1_volatiles=("taunt",)), _override(), dex=self.dex
        )
        side_one = dataclasses.replace(
            world.spec.side_one,
            volatile_status_durations={
                **world.spec.side_one.volatile_status_durations,
                "taunt": duration,
            },
        )
        return build_poke_engine_state(dataclasses.replace(world.spec, side_one=side_one))

    def test_the_counter_survives_serialization(self) -> None:
        # The native search crate is handed `state.to_string()`, so a duration
        # that does not round-trip would be silently dropped on that path only.
        text = self.state.to_string()
        self.assertIn("TAUNT", text.upper())
        self.assertEqual(poke_engine.State.from_string(text).to_string(), text)

    @staticmethod
    def _taunt_instructions(state) -> list[str]:
        branches = poke_engine.generate_instructions(state, _ATTACK_MOVE, "bodyslam")
        best = max(branches, key=lambda branch: branch.percentage)
        return [str(i) for i in best.instruction_list if "TAUNT" in str(i).upper()]


if __name__ == "__main__":
    unittest.main()
