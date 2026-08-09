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

AND WHERE IT IS NOT EXACT, IT REFUSES. "One tick elapsed" holds at an ordinary
move boundary because the applying turn's residual has always already run. At a
mid-turn REPLACEMENT boundary it has not, and the engine runs that deferred
residual on the replacement ply, so the right seed depends on how old the Taunt
is -- age 0 needs 0, age 1 needs 1, both are reachable, and the payload carries
neither. `taunt` is therefore withdrawn from `_SUPPORTED_VOLATILES` at that
boundary and the world fails closed, which is what `origin/main` did everywhere.
Pinned by ``test_a_replacement_boundary_refuses_rather_than_guessing_the_age``
and the wheel fact under it.

⚠ An earlier revision of this file asserted the OPPOSITE -- that a replacement
turn runs no residual and so cannot reach Taunt -- and built its evidence with
``hp = 0`` and no ``force_switch`` flag, an arm production never takes. That is
why the replacement tests below construct the payload and let
``battle_spec_from_payload`` set the flag, instead of hand-building a Side.
"""

from __future__ import annotations

import dataclasses
import json
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
        # A counter written unconditionally would be a silent lie on every other world.
        #
        # ⚠ An earlier version of this comment justified the test with "the engine panics
        # on a nonzero counter without the volatile", and that is FALSE: block 10.15 is
        # gated on `volatile_statuses.contains(TAUNT)`, so a stray counter with no
        # volatile is inert and the panic arm is unreachable from here. The test is still
        # worth having on the honest reason -- the world must not assert a fact it did not
        # observe, and it is what kills the seed-unconditionally mutant.
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
        """The native search crate is handed `state.to_string()`, so a duration that does
        not round-trip would be silently dropped on that path and that path only.

        ⚠ THE OBVIOUS VERSION OF THIS TEST IS WORTHLESS AND WAS WRITTEN FIRST.
        `assertIn("TAUNT", text.upper())` is satisfied by the volatile NAME alone, and
        `from_string(to_string(s)) == to_string(s)` is a property of the SERIALIZER, not
        of the counter. Measured: that pair killed only M1 (remove the volatile from the
        allow-list, so no world builds at all) and stayed green under M2 (drop the seed),
        M3 (seed 0), M4 (seed 2) and M8 (seed 3). It asserted nothing this file is about.

        The discriminating value was in the same string the whole time: the sixth field
        is `confusion;encore;lockedmove;slowstart;taunt;yawn`, so a seeded 1 is visible as
        `0;0;0;0;1;0`. Assert THAT, and assert it moves with the seed.
        """

        text = self.state.to_string()
        self.assertIn("TAUNT", text.upper())
        self.assertEqual(poke_engine.State.from_string(text).to_string(), text)

        # The counter itself, read out of the serialized form rather than assumed.
        self.assertEqual(self._durations_field(self.state), "0;0;0;0;1;0")
        # ...and it is the SEED that put it there, not a constant in the serializer.
        self.assertEqual(
            self._durations_field(self._state_with_taunt_duration(0)), "0;0;0;0;0;0"
        )
        # Survives the round trip with its VALUE, which is the property the crate needs.
        self.assertEqual(
            self._durations_field(poke_engine.State.from_string(text)), "0;0;0;0;1;0"
        )

    def test_the_serialized_durations_field_is_where_we_think_it_is(self) -> None:
        """`_durations_field` hard-codes index 9 of the `=`-delimited state.

        That index is an ordering decision inside a Rust `Display` impl with nothing on
        its side pinning it, so a reordering upstream would silently move every
        assertion above onto some other field. Nail it down from BOTH seats: side_one's
        durations are field 9 and side_two's are field 37, and only one of the two
        carries the seeded Taunt -- so a shifted layout cannot satisfy both.
        """

        world = battle_spec_from_payload(
            _payload(self.dex, p1_volatiles=("taunt",)), _override(), dex=self.dex
        )
        fields = build_poke_engine_state(world.spec).to_string().split("=")
        self.assertEqual(fields[9], "0;0;0;0;1;0", "side_one durations are not field 9")
        self.assertEqual(fields[37], "0;0;0;0;0;0", "side_two durations are not field 37")

        mirrored = battle_spec_from_payload(
            _payload(self.dex, p2_volatiles=("taunt",)), _override(), dex=self.dex
        )
        mirrored_fields = build_poke_engine_state(mirrored.spec).to_string().split("=")
        self.assertEqual(mirrored_fields[9], "0;0;0;0;0;0")
        self.assertEqual(mirrored_fields[37], "0;0;0;0;1;0")

    @staticmethod
    def _durations_field(state) -> str:
        """side_one's `VolatileStatusDurations` as serialized.

        Field 9 of the `=`-delimited state. The index is pinned from both seats by
        `test_the_serialized_durations_field_is_where_we_think_it_is`; the assertions
        that use this helper pin both a seeded and an unseeded value, so a wrong index
        cannot read as a pass in both.
        """

        return state.to_string().split("=")[9]

    def test_a_replacement_boundary_refuses_rather_than_guessing_the_age(self) -> None:
        """The seam this branch first got WRONG, now built the way production builds it.

        Round one asserted here that a replacement turn "contributes no move phase and no
        residual", so the seam could not reach Taunt -- and offered that as the structural
        discriminator against Yawn. The state it measured had `hp = 0` and NO
        `force_switch` flag. gen3 `get_all_options` checks the EXPLICIT flag first, and
        `end_of_turn_triggered` keys on that same flag, so the flagged arm RUNS the
        deferred residual on the replacement ply while the `hp<=0`-only arm does not.
        `_build_side_spec` sets the flag, so the test measured an arm production never
        takes. Claim withdrawn.

        With the flag set, the residual eats the Taunt on the replacement ply, and the
        correct seed depends on the Taunt's AGE, which the payload does not carry.
        Measured live on the simulator (`.probe/force_switch_taunt_live.py`):

            age 0 (Taunt landed on the faint turn) -> 1 taunted move phase left
            age 1 (Taunt landed the turn before)   -> 0 taunted move phases left

        and at a `force_switch` world the engine gives 0 phases for seed 1 and 1 for
        seed 0. No single seed is right, so the boundary fails closed instead.
        """

        payload = _payload(self.dex, p2_volatiles=("taunt",))
        payload["selfRequestKind"] = "force-switch"
        rows = payload["sides"]["p1"]["pokemon"]
        rows[0]["condition"] = "0 fnt"     # our active fainted...
        rows[1]["condition"] = "224/224"   # ...and this is the replacement we owe

        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertEqual(caught.exception.reason, "volatile_unsupported")
        self.assertIn("taunt", str(caught.exception))

        # Anti-vacuity, two ways. The refusal must be about the BOUNDARY, not about the
        # payload being malformed or about Taunt generally:
        #   * the identical payload at an ordinary boundary BUILDS;
        #   * the identical replacement boundary WITHOUT taunt also builds.
        ordinary = _payload(self.dex, p2_volatiles=("taunt",))
        self.assertIn(
            "taunt",
            battle_spec_from_payload(ordinary, _override(), dex=self.dex)
            .spec.side_two.volatile_statuses,
        )
        untaunted = dict(payload)
        untaunted["sides"] = json.loads(json.dumps(payload["sides"]))
        untaunted["sides"]["p2"]["volatiles"] = []
        built = battle_spec_from_payload(untaunted, _override(), dex=self.dex)
        self.assertTrue(built.spec.side_one.force_switch)

    def test_the_replacement_ply_really_does_run_the_residual(self) -> None:
        """The engine fact the refusal above rests on, pinned at the wheel.

        Both arms are exercised, because the difference between them is exactly the
        mistake round one made: the flagged arm consumes the Taunt on the replacement
        ply, the `hp<=0`-only arm does not.
        """

        def state(*, force_switch: bool):
            side_one = poke_engine.Side(
                pokemon=[
                    poke_engine.Pokemon(id="blissey", hp=0, maxhp=300,
                                        moves=[poke_engine.Move(id="tackle", pp=56)]),
                    poke_engine.Pokemon(id="starmie", hp=200, maxhp=200,
                                        moves=[poke_engine.Move(id="tackle", pp=56)]),
                ],
                active_index="0", force_switch=force_switch,
            )
            side_two = poke_engine.Side(
                pokemon=[poke_engine.Pokemon(
                    id="smeargle", hp=200, maxhp=200,
                    moves=[poke_engine.Move(id=_STATUS_MOVE, pp=16),
                           poke_engine.Move(id="tackle", pp=56)])],
                active_index="0", volatile_statuses={"taunt"},
                volatile_status_durations=poke_engine.VolatileStatusDurations(taunt=1),
            )
            return poke_engine.State(side_one=side_one, side_two=side_two)

        def replacement_ply(st):
            branches = poke_engine.generate_instructions(st, "starmie", "none")
            best = max(branches, key=lambda b: b.percentage)
            return [str(i) for i in best.instruction_list]

        flagged = replacement_ply(state(force_switch=True))
        derived = replacement_ply(state(force_switch=False))
        self.assertIn("RemoveVolatileStatus SideTwo: TAUNT", flagged)
        self.assertNotIn("RemoveVolatileStatus SideTwo: TAUNT", derived)

    def test_a_lone_taunted_all_status_side_leaves_the_engine_no_move(self) -> None:
        """The composition with #1202, asserted at the engine rather than argued.

        #1202 translates the crate's forced-no-move token onto the request's substituted
        `struggle`, gated on Struggle being the request's only legal action. That gate is
        reachable through Taunt only if the built world actually drives the engine to
        `MoveChoice::None` -- i.e. `add_available_moves` contributes nothing (every slot
        Status, and TAUNTed) AND `add_switches` contributes nothing (no live bench), so
        `get_all_options` falls through to its terminal `options.len() == 0` push.

        Both halves are needed and the test separates them: with a live bench the engine
        enumerates the switch instead, which is exactly why the corpus scenario
        `struggle_taunt_stall` never reached this shape and why
        `test_struggle_only_move_state.TauntStruggleOnlyReachesTheEngineTests` had to add
        a bench-less fixture.
        """

        def options(*, taunted: bool, bench: bool) -> set[str]:
            mon = poke_engine.Pokemon(
                id="blissey", hp=300, maxhp=300,
                moves=[poke_engine.Move(id=_STATUS_MOVE, pp=16),
                       poke_engine.Move(id="lightscreen", pp=48)],
            )
            party = [mon] + (
                [poke_engine.Pokemon(id="starmie", hp=200, maxhp=200,
                                     moves=[poke_engine.Move(id=_ATTACK_MOVE, pp=16)])]
                if bench else []
            )
            kwargs = {"pokemon": party, "active_index": "0"}
            if taunted:
                kwargs["volatile_statuses"] = {"taunt"}
                kwargs["volatile_status_durations"] = poke_engine.VolatileStatusDurations(
                    taunt=1
                )
            state = poke_engine.State(
                side_one=poke_engine.Side(**kwargs),
                side_two=poke_engine.Side(
                    pokemon=[poke_engine.Pokemon(id="smeargle", hp=200, maxhp=200,
                                                 moves=[poke_engine.Move(id="tackle", pp=56)])],
                    active_index="0",
                ),
            )
            return {
                str(r.move_choice)
                for r in poke_engine.monte_carlo_tree_search(state, 40).side_one
            }

        # Anti-vacuity: untaunted, the same all-Status side has real options.
        self.assertEqual(options(taunted=False, bench=False), {_STATUS_MOVE, "lightscreen"})
        # TAUNTed with no bench: nothing at all, so the engine emits its forced no-move.
        self.assertEqual(options(taunted=True, bench=False), {"No Move"})
        # TAUNTed WITH a bench: the switch is enumerated, so this is not the class.
        self.assertNotIn("No Move", options(taunted=True, bench=True))

    @staticmethod
    def _taunt_instructions(state) -> list[str]:
        branches = poke_engine.generate_instructions(state, _ATTACK_MOVE, "bodyslam")
        best = max(branches, key=lambda branch: branch.percentage)
        return [str(i) for i in best.instruction_list if "TAUNT" in str(i).upper()]


if __name__ == "__main__":
    unittest.main()
