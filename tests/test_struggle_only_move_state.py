"""A Struggle-only request must not leave the self moveset pinned to an older one.

THE DEFECT. ``sides[self].pokemon[].moves`` in the direct-materialization payload is
built ONLY from ``self_move_states``, folded by
``local_showdown.actor_move_states_from_request_history``. That fold skipped any request
where ``_request_active_moves`` returned ``[]``, and that helper drops move rows lacking
integer ``pp`` AND ``maxpp``. Showdown's Struggle branch
(``Pokemon.getMoveRequestData``'s ``!moves.length`` case) emits exactly one such row, so
the fold skipped the request and the retained entry stayed pinned to the last pp-BEARING
request. The world therefore answered a Struggle-only request with a moveset that still
advertised a usable move.

MEASURED, and this repo's first live capture of the branch. The provenance list in
``actor_move_states_from_request_history`` had classified Struggle as "SOURCE ONLY --
unverified by measurement; an attempt to produce a live Struggle request failed on
packed-team format". Driving a gen3 Custom Game with a single-move Bulbasaur (Sunny Day,
8 PP) for eight turns produces it. At that boundary, at the pre-fix commit::

    RAW REQUEST active moves : [{"move": "Struggle", "id": "struggle",
                                 "target": "randomNormal", "disabled": false}]
    VIEW A sides.p1.pokemon  : [{"species": "Bulbasaur", "active": true,
                                 "moves": [{"id": "sunnyday", "pp": 1, "maxpp": 8,
                                            "disabled": false}]}, ...]
    VIEW B selfActiveMoves   : []

-- one payload, two views of the same request, built from two DIFFERENT requests. View A
says Sunny Day is usable with 1 PP; the truth is 0 PP and unusable, and the request
offers only Struggle. ``StruggleOnlyPayloadViewTests`` re-runs that capture and asserts
the two views agree.

WHY "MARK EVERY SLOT DISABLED" IS A RESTORATION, NOT A GUESS. ``Pokemon.getMoves`` folds
``moveSlot.pp <= 0`` into ``disabled`` for every slot and returns
``hasValidMove ? moves : []``. An empty return therefore MEANS every slot came back
disabled; ``getMoveRequestData`` then discards that list and substitutes Struggle. The
fold now writes back the verdict Showdown had already reached.

SCOPE. The sibling pp-less shapes -- ``mustrecharge`` (measured 6 times) and the two-turn
charge lock -- are dropped by the SAME filter but come off ``getMoves``'s early
``if (lockedMove)`` return, which never evaluates per-slot ``disabled``. They keep the
pre-existing skip; ``MustRechargeIsOutOfScopeTests`` pins that boundary so a later edit
cannot widen it silently while the ``trapped`` work is in flight.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    _rows_report_nothing_usable,
    _sole_enabled_move_id,
    battle_spec_from_payload,
)
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.gen3_damage import gen3_hp_stat  # noqa: E402
from pokezero.local_showdown import (  # noqa: E402
    DEFAULT_SHOWDOWN_ROOT,
    LocalShowdownConfig,
    LocalShowdownEnv,
    _public_materialization_payload,
    _request_reports_only_struggle,
    actor_move_states_from_request_history,
)
from pokezero.showdown_fixture import FixturePokemon, pack_team  # noqa: E402


# --------------------------------------------------------------------------------------
# The fold
# --------------------------------------------------------------------------------------

TEAM = (
    ("p1: Bulbasaur", True, ["sunnyday", "tackle"]),
    ("p1: Charmander", False, ["ember"]),
)


def _pp_request(active_moves, team_rows=TEAM):
    """A normal request: `active[0].moves` carries pp/maxpp for every slot."""

    return {
        "active": [
            {
                "moves": [
                    {"id": mid, "pp": pp, "maxpp": maxpp, "disabled": disabled}
                    for mid, pp, maxpp, disabled in active_moves
                ]
            }
        ],
        "side": {
            "pokemon": [
                {
                    "ident": ident,
                    "details": ident.split(": ", 1)[1],
                    "active": active,
                    "moves": list(moves),
                }
                for ident, active, moves in team_rows
            ]
        },
    }


def _struggle_request(team_rows=TEAM):
    """Showdown's Struggle branch, VERBATIM from the live capture above."""

    request = _pp_request((), team_rows)
    request["active"] = [
        {
            "moves": [
                {
                    "move": "Struggle",
                    "id": "struggle",
                    "target": "randomNormal",
                    "disabled": False,
                }
            ]
        }
    ]
    return request


def _recharge_request(team_rows=TEAM):
    """Showdown's `getMoves(lockedMove='recharge')` early return. Also pp-less."""

    request = _pp_request((), team_rows)
    request["active"] = [{"moves": [{"move": "Recharge", "id": "recharge"}], "trapped": True}]
    return request


class StruggleOnlyFoldTests(unittest.TestCase):
    def test_the_fixture_is_the_real_branch(self) -> None:
        """Guard against a vacuous suite: the row must be pp-LESS, or nothing is tested.

        If a later edit gives this fixture `pp`/`maxpp`, `_request_active_moves` would
        keep it and every assertion below would pass for the wrong reason.
        """

        row = _struggle_request()["active"][0]["moves"][0]
        self.assertEqual(row["id"], "struggle")
        self.assertNotIn("pp", row)
        self.assertNotIn("maxpp", row)
        self.assertTrue(_request_reports_only_struggle(_struggle_request()))

    def test_a_struggle_request_marks_the_retained_snapshot_unusable(self) -> None:
        """The defect. Pre-fix this request was skipped and `disabled` stayed False."""

        pp = _pp_request((("sunnyday", 1, 8, False), ("tackle", 0, 56, True)))
        states = actor_move_states_from_request_history(
            [pp, _struggle_request()], initial_request=pp
        )

        self.assertEqual(
            [(row["id"], row["disabled"]) for row in states["bulbasaur"]],
            [("sunnyday", True), ("tackle", True)],
            "a Struggle request says every slot is disabled; the retained row must say so",
        )

    def test_the_identity_and_the_pp_survive(self) -> None:
        """Not erasure. Erasing strands the mon and `engine_world` says `self_pp_unknown`.

        Measured wrong for the sibling Transform case (7 refusals became 8). The PP
        numbers stay pinned to the last pp-bearing request because the Struggle request
        carries none -- that residual is stated, not fixed.
        """

        pp = _pp_request((("sunnyday", 1, 8, False),))
        states = actor_move_states_from_request_history(
            [pp, _struggle_request()], initial_request=pp
        )

        self.assertIn("bulbasaur", states)
        self.assertEqual(states["bulbasaur"][0]["pp"], 1)
        self.assertEqual(states["bulbasaur"][0]["maxpp"], 8)

    def test_the_pp_bearing_snapshot_is_not_mutated_in_place(self) -> None:
        """The marking must produce a NEW row, not edit the retained mapping's."""

        pp = _pp_request((("sunnyday", 1, 8, False),))
        before = actor_move_states_from_request_history([pp], initial_request=pp)
        after = actor_move_states_from_request_history(
            [pp, _struggle_request()], initial_request=pp
        )

        self.assertFalse(before["bulbasaur"][0]["disabled"])
        self.assertTrue(after["bulbasaur"][0]["disabled"])

    def test_a_later_pp_bearing_request_refreshes_the_marking(self) -> None:
        """The marking is about THIS boundary, not a permanent condemnation."""

        first = _pp_request((("sunnyday", 1, 8, False),))
        later = _pp_request((("sunnyday", 8, 8, False),))
        states = actor_move_states_from_request_history(
            [first, _struggle_request(), later], initial_request=first
        )

        self.assertEqual(
            [(row["id"], row["pp"], row["disabled"]) for row in states["bulbasaur"]],
            [("sunnyday", 8, False)],
        )

    def test_stale_by_one_decision_is_the_floor_not_the_bound(self) -> None:
        """Consecutive pp-less turns, which is what makes "skip" unbounded.

        Two Struggle turns with a `mustrecharge` between them: three consecutive requests
        the old filter dropped. Every one of them must leave the entry unusable, not just
        the first.
        """

        pp = _pp_request((("sunnyday", 1, 8, False),))
        states = actor_move_states_from_request_history(
            [pp, _struggle_request(), _recharge_request(), _struggle_request()],
            initial_request=pp,
        )

        self.assertTrue(states["bulbasaur"][0]["disabled"])

    def test_a_struggle_request_with_no_retained_entry_invents_nothing(self) -> None:
        pp_for_other_mon = _pp_request(
            (("ember", 25, 40, False),),
            (("p1: Bulbasaur", False, ["sunnyday"]), ("p1: Charmander", True, ["ember"])),
        )
        states = actor_move_states_from_request_history(
            [pp_for_other_mon, _struggle_request()], initial_request=pp_for_other_mon
        )

        self.assertNotIn("bulbasaur", states)
        self.assertEqual([row["id"] for row in states["charmander"]], ["ember"])

    def test_a_pp_bearing_struggle_row_is_not_the_branch(self) -> None:
        """The predicate keys on the MISSING pp fields, not on the id alone."""

        request = _struggle_request()
        request["active"][0]["moves"][0].update({"pp": 5, "maxpp": 5})
        self.assertFalse(_request_reports_only_struggle(request))


class MustRechargeIsOutOfScopeTests(unittest.TestCase):
    """A deliberate boundary, pinned so a later widening is visible in the diff.

    `mustrecharge` is dropped by the SAME filter, and a sibling change is routing it
    through the MUSTRECHARGE volatile. It is NOT marked here: `getMoves` returns early on
    `lockedMove` without ever computing per-slot `disabled`, so the real slots are usually
    perfectly usable and merely pre-empted for one turn. Marking them would be a
    fabrication rather than a restoration, and it would be inert anyway --
    `get_all_options` short-circuits on MUSTRECHARGE before `add_available_moves` runs.
    """

    def test_a_recharge_request_is_not_the_struggle_branch(self) -> None:
        self.assertFalse(_request_reports_only_struggle(_recharge_request()))

    def test_a_recharge_request_leaves_the_retained_snapshot_untouched(self) -> None:
        pp = _pp_request((("sunnyday", 3, 8, False),))
        states = actor_move_states_from_request_history(
            [pp, _recharge_request()], initial_request=pp
        )

        self.assertEqual(
            [(row["id"], row["pp"], row["disabled"]) for row in states["bulbasaur"]],
            [("sunnyday", 3, False)],
        )


# --------------------------------------------------------------------------------------
# Both payload views, on a LIVE Struggle request
# --------------------------------------------------------------------------------------


def _integration_config() -> LocalShowdownConfig | None:
    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=20.0)


_STRUGGLE_OVERRIDE = BattleStartOverride(
    player_teams={
        # ONE move, 5 base PP (-> 8 with full PP ups). Eight turns of Sunny Day exhausts
        # it and Showdown has nothing left to offer but Struggle. Sunny Day is chosen
        # because it is harmless: neither side faints and the battle stays at a move
        # request throughout.
        "p1": pack_team(
            (
                FixturePokemon(species="Bulbasaur", ability="Overgrow", moves=("Sunny Day",)),
                FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember",)),
            )
        ),
        "p2": pack_team(
            (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Harden",)),)
        ),
    },
)


def _enabled_ids(rows) -> list[str]:
    return [row["id"] for row in rows if not row.get("disabled")]


class StruggleOnlyPayloadViewTests(unittest.TestCase):
    """The two views of the SAME request must agree about what is usable.

    Not one assertion but two, because a fix that repairs `sides[...].moves` and leaves
    `selfActiveMoves` alone still ships an inconsistent payload. The pre-Struggle turn is
    asserted too: it is what shows the agreement is a property of the payload rather than
    an artefact of both views happening to be empty.
    """

    def setUp(self) -> None:
        config = _integration_config()
        if config is None:
            self.skipTest("a built pokemon-showdown checkout and node are required")
        self.config = config

    def _payloads(self):
        """(last pp-bearing boundary, Struggle boundary) as real payloads."""

        previous = None
        with LocalShowdownEnv(self.config) as env:
            env.reset_with_start_override(seed=11, start_override=_STRUGGLE_OVERRIDE)
            for _ in range(16):
                request = env._latest_requests.get("p1")
                active = (request or {}).get("active")
                rows = (active[0] if isinstance(active, list) and active else {}).get("moves")
                payload = _public_materialization_payload(
                    env.public_materialization_state("p1")
                )
                if rows and rows[0].get("id") == "struggle":
                    return previous, payload, rows
                previous = payload
                env.step({"p1": 0, "p2": 0})
        raise AssertionError("no Struggle request was produced in 16 turns")

    def test_the_two_views_agree_at_a_live_struggle_request(self) -> None:
        previous, struggle, raw_rows = self._payloads()

        # The captured branch, so the test states what it measured.
        self.assertEqual(
            json.dumps(raw_rows, sort_keys=True),
            json.dumps(
                [
                    {
                        "disabled": False,
                        "id": "struggle",
                        "move": "Struggle",
                        "target": "randomNormal",
                    }
                ],
                sort_keys=True,
            ),
        )

        # The turn BEFORE: both views name the same single usable move. Pre-fix this held
        # too -- it is the regression guard, not the fix.
        prev_active = next(r for r in previous["sides"]["p1"]["pokemon"] if r["active"])
        self.assertEqual(_enabled_ids(prev_active["moves"]), ["sunnyday"])
        self.assertEqual(_enabled_ids(previous["selfActiveMoves"]), ["sunnyday"])

        # The Struggle turn. VIEW B is empty by construction; VIEW A used to say
        # `sunnyday pp1 disabled:false`.
        active = next(r for r in struggle["sides"]["p1"]["pokemon"] if r["active"])
        self.assertEqual(struggle["selfActiveMoves"], [])
        self.assertEqual(
            _enabled_ids(active["moves"]),
            [],
            "VIEW A still advertised a usable move while VIEW B said nothing was usable",
        )

        # The PP snapshot survives -- this is a usability correction, not an erasure.
        self.assertEqual(
            [(row["id"], row["pp"]) for row in active["moves"]], [("sunnyday", 1)]
        )

        # And the single predicate every consumer applies to these rows now agrees across
        # both views instead of disagreeing.
        self.assertIsNone(_sole_enabled_move_id(active["moves"]))
        self.assertIsNone(_sole_enabled_move_id(struggle["selfActiveMoves"]))


# --------------------------------------------------------------------------------------
# What engine_world then builds
# --------------------------------------------------------------------------------------


def _move(move_id: str, pp: int) -> MoveInfo:
    return MoveInfo(
        id=move_id, name=move_id, type="normal", category="physical",
        gen3_category="physical", base_power=50, accuracy=100.0, priority=0,
        recoil=False, drain=False, heal=False, status=None, boosts={},
        target="normal", selfdestruct=False, pp=pp,
    )


def _species(species_id: str, name: str, types, base, weight: float) -> SpeciesInfo:
    return SpeciesInfo(
        id=species_id, name=name, types=types, base_stats=base, weight_kg=weight
    )


def _dex() -> ShowdownDex:
    return ShowdownDex(
        moves={
            "bodyslam": _move("bodyslam", 15),
            "healbell": _move("healbell", 5),
            "wish": _move("wish", 10),
            "protect": _move("protect", 10),
            "earthquake": _move("earthquake", 10),
            "surf": _move("surf", 15),
        },
        species={
            "delcatty": _species("delcatty", "Delcatty", ("normal",),
                                 {"hp": 70, "atk": 65, "def": 65, "spa": 55, "spd": 55, "spe": 70}, 32.6),
            "swampert": _species("swampert", "Swampert", ("water", "ground"),
                                 {"hp": 100, "atk": 110, "def": 90, "spa": 85, "spd": 90, "spe": 60}, 81.9),
        },
        type_chart={},
    )


_EVS = {stat: 85 for stat in ("hp", "atk", "def", "spa", "spd", "spe")}
_DELCATTY = FixturePokemon(
    species="Delcatty", moves=("bodyslam", "healbell", "wish", "protect"),
    ability="Cute Charm", item="Leftovers", level=96, evs=dict(_EVS),
)
_SWAMPERT = FixturePokemon(
    species="Swampert", moves=("earthquake", "surf"), ability="Torrent",
    item="Leftovers", level=84, evs=dict(_EVS),
)
_OVERRIDE = BattleStartOverride(
    player_teams={"p1": pack_team((_DELCATTY, _SWAMPERT)), "p2": pack_team((_SWAMPERT,))}
)

# The fold's Struggle marking, as it reaches engine_world: every slot disabled, PP pinned
# to the last pp-bearing request. Protect at 0 PP is the one that ran out and produced the
# branch; the other three are full and were disabled by Encore.
_STRUGGLE_ROWS = [
    {"id": "bodyslam", "pp": 24, "maxpp": 24, "disabled": True},
    {"id": "healbell", "pp": 5, "maxpp": 5, "disabled": True},
    {"id": "wish", "pp": 10, "maxpp": 10, "disabled": True},
    {"id": "protect", "pp": 0, "maxpp": 10, "disabled": True},
]


def _struggle_payload(dex: ShowdownDex, **overrides):
    delcatty_hp = gen3_hp_stat(
        int(dex.species_info("Delcatty").base_stats["hp"]), 31, 85, _DELCATTY.level
    )
    swampert_hp = gen3_hp_stat(
        int(dex.species_info("Swampert").base_stats["hp"]), 31, 85, _SWAMPERT.level
    )
    payload = {
        "turn": 26,
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
        "selfTeamOrder": ["Delcatty", "Swampert"],
        "selfActiveRequestState": {},
        "selfBenchedMoveHistory": False,
        # VIEW B under Struggle: Showdown offered nothing pp-bearing.
        "selfActiveMoves": [],
        "sides": {
            "p1": {
                "pokemon": [
                    {
                        "species": "Delcatty",
                        "condition": f"{delcatty_hp}/{delcatty_hp}",
                        "active": True,
                        "moves": [dict(row) for row in _STRUGGLE_ROWS],
                    },
                    {
                        "species": "Swampert",
                        "condition": f"{swampert_hp}/{swampert_hp}",
                        "active": False,
                        "moves": [],
                    },
                ],
                "boosts": {},
                "volatiles": ["Encore"],
                "lastUsedMove": "protect",
                "materializationBlockers": [],
                "toxicStage": 0,
                "sideConditions": {},
                "sideConditionSetTurns": {},
            },
            "p2": {
                "pokemon": [
                    {
                        "species": "Swampert",
                        "condition": f"{swampert_hp}/{swampert_hp}",
                        "active": True,
                    }
                ],
                "boosts": {},
                "volatiles": [],
                "lastUsedMove": "earthquake",
                "materializationBlockers": [],
                "toxicStage": 0,
                "sideConditions": {},
                "sideConditionSetTurns": {},
            },
        },
    }
    payload.update(overrides)
    return payload


class StruggleMarkedRowsReachTheEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dex = _dex()

    def test_the_signature_predicate_separates_struggle_from_everything_else(self) -> None:
        self.assertTrue(_rows_report_nothing_usable(_STRUGGLE_ROWS))
        # An ordinary request can never look like this: Showdown returns Struggle the
        # moment `hasValidMove` is false, so an all-disabled pp-bearing row does not exist.
        self.assertFalse(
            _rows_report_nothing_usable(
                [dict(row, disabled=(row["id"] != "protect")) for row in _STRUGGLE_ROWS]
            )
        )
        # An EMPTY list is a force-switch row or an absent snapshot, not this.
        self.assertFalse(_rows_report_nothing_usable([]))
        self.assertFalse(_rows_report_nothing_usable(None))

    def test_no_move_spec_is_selectable(self) -> None:
        """`_move_specs` copies (pp, disabled) verbatim, so the world offers switches only.

        Checked against poke-engine 0.0.47: `Side::add_available_moves` requires
        `!disabled && pp > 0`, and `get_all_options` then falls through to `add_switches`
        -- which is exactly what the live Struggle request also offers (its active row has
        no `trapped` key).
        """

        world = battle_spec_from_payload(_struggle_payload(self.dex), _OVERRIDE, dex=self.dex)
        side = world.spec.side_one
        active = side.pokemon[side.active_index]
        self.assertEqual(
            [(spec.id, spec.disabled) for spec in active.moves if spec.id != "none"],
            [("bodyslam", True), ("healbell", True), ("wish", True), ("protect", True)],
        )
        self.assertEqual([spec.pp for spec in active.moves if spec.id != "none"], [24, 5, 10, 0])

    def test_self_encore_at_a_struggle_request_resolves_from_last_used_move(self) -> None:
        """The consequence of the fold fix for `_sole_enabled_move_id`, addressed.

        The self seat gets no caller-supplied `encored_move` -- `_public_signals` fills
        that map for the OPPONENT slot only -- so self Encore was identified solely by
        "exactly one enabled row". Marking every row disabled removes that, and without
        the `lastUsedMove` fallback the fold fix would turn the population it repairs into
        an `encore_move_unknown` refusal. Protect is slot 3, so a silent slot-0 fallback is
        visible.
        """

        side = battle_spec_from_payload(
            _struggle_payload(self.dex), _OVERRIDE, dex=self.dex
        ).spec.side_one
        self.assertEqual(side.last_used_move, "move:3")
        self.assertEqual(dict(side.volatile_status_durations), {"encore": 1})

    def test_an_ordinary_encore_row_still_resolves_from_the_row(self) -> None:
        """Re-enable one row and the lock comes from THAT row, not from `lastUsedMove`.

        `lastUsedMove` names a different move here so the two sources are
        distinguishable: `wish` is slot 2, `protect` is slot 3.
        """

        payload = _struggle_payload(self.dex)
        rows = payload["sides"]["p1"]["pokemon"][0]["moves"]
        rows[2]["disabled"] = False  # wish, slot 2
        payload["sides"]["p1"]["lastUsedMove"] = "protect"
        side = battle_spec_from_payload(payload, _OVERRIDE, dex=self.dex).spec.side_one
        self.assertEqual(side.last_used_move, "move:2")

    def test_the_fallback_is_gated_on_the_struggle_signature(self) -> None:
        """The gate itself, and the mutant that proves it is load-bearing.

        Two enabled rows is the OTHER way `_sole_enabled_move_id` returns None: the
        snapshot does not identify the lock, and `engine_world` has always refused there.
        `lastUsedMove` is a perfectly resolvable move, so an UNGATED fallback would happily
        build a world -- widening self-seat Encore resolution to a population the fold fix
        has nothing to do with. Dropping `_rows_report_nothing_usable` from the condition
        must turn this refusal into a build.
        """

        payload = _struggle_payload(self.dex)
        rows = payload["sides"]["p1"]["pokemon"][0]["moves"]
        rows[1]["disabled"] = False  # healbell, slot 1
        rows[2]["disabled"] = False  # wish, slot 2
        payload["sides"]["p1"]["lastUsedMove"] = "protect"

        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(payload, _OVERRIDE, dex=self.dex)
        self.assertEqual(caught.exception.reason, "encore_move_unknown")

    def test_a_second_consecutive_struggle_turn_still_fails_closed(self) -> None:
        """Honest limit. After Struggling, `lastUsedMove` is `struggle`, in no moveset."""

        payload = _struggle_payload(self.dex)
        payload["sides"]["p1"]["lastUsedMove"] = "struggle"
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(payload, _OVERRIDE, dex=self.dex)
        self.assertEqual(caught.exception.reason, "encore_move_unknown")

    def test_a_struggle_request_without_encore_builds_with_no_lock(self) -> None:
        payload = _struggle_payload(self.dex)
        payload["sides"]["p1"]["volatiles"] = []
        side = battle_spec_from_payload(payload, _OVERRIDE, dex=self.dex).spec.side_one
        self.assertNotIn("encore", side.volatile_statuses)
        active = side.pokemon[side.active_index]
        self.assertTrue(all(spec.disabled for spec in active.moves))


if __name__ == "__main__":
    unittest.main()
