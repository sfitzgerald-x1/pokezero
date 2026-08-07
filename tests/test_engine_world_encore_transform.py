"""The Encore lock on a TRANSFORMED active must index the post-Transform moveset.

Showdown locks Encore by move ID; the vendored gen3 poke-engine locks by move
SLOT INDEX (``last_used_move = move:<i>``). ``engine_world`` bridges the two, and
for a transformed active the two lists it could bridge against are different:

  * ``_active_row_moves(side_payload)`` is deliberately the PRE-Transform
    snapshot -- ``local_showdown.actor_move_states_from_request_history`` skips
    requests taken while transformed so that PP stays honest. For a gen3
    randbats Ditto that is the single move ``transform``.
  * ``payload["selfActiveMoves"]`` is the RAW request's usable moveset, i.e. the
    COPIED moves, carrying Encore's disable pattern.

Resolving against the first satisfies the self-seat rule "exactly one enabled
move identifies the lock" SPURIOUSLY -- a one-move list always has exactly one
enabled entry -- and yields slot 0. ``_apply_transform`` then swaps the donor's
moveset in, so slot 0 names the donor's first move.

Measured on holdout ``19100170/71-72``: Showdown Encored Protect (donor slot 3),
the world locked donor slot 0 (Body Slam), and the resulting phantom KO made
``end_of_turn_is_deferred`` suppress the whole residual block -- surfacing as two
``component_missing_in_engine:itemleftovers`` divergences. Turns 76-78 of the
same game matched only because that Encore happened to lock Body Slam, which IS
donor slot 0.

``test_transformed_self_encore_locks_the_post_transform_slot`` is the row that
was RED before the fix, at ``move:0`` instead of ``move:3``.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    _resolve_encored_move_index,
    _sole_enabled_move_id,
    battle_spec_from_payload,
)
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.gen3_damage import gen3_hp_stat  # noqa: E402
from pokezero.poke_engine_adapter import MoveSpec  # noqa: E402
from pokezero.showdown_fixture import FixturePokemon, pack_team  # noqa: E402


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
            "transform": _move("transform", 10),
            "bodyslam": _move("bodyslam", 15),
            "healbell": _move("healbell", 5),
            "wish": _move("wish", 10),
            "protect": _move("protect", 10),
            "surf": _move("surf", 15),
            "earthquake": _move("earthquake", 10),
            "hiddenpower": _move("hiddenpower", 15),
        },
        species={
            "ditto": _species("ditto", "Ditto", ("normal",),
                              {"hp": 48, "atk": 48, "def": 48, "spa": 48, "spd": 48, "spe": 48}, 4.0),
            "delcatty": _species("delcatty", "Delcatty", ("normal",),
                                 {"hp": 70, "atk": 65, "def": 65, "spa": 55, "spd": 55, "spe": 70}, 32.6),
            "swampert": _species("swampert", "Swampert", ("water", "ground"),
                                 {"hp": 100, "atk": 110, "def": 90, "spa": 85, "spd": 90, "spe": 60}, 81.9),
        },
        type_chart={},
    )


_EVS = {stat: 85 for stat in ("hp", "atk", "def", "spa", "spd", "spe")}

_DITTO = FixturePokemon(
    species="Ditto", moves=("transform",), ability="Limber", item="Leftovers",
    level=100, evs=dict(_EVS),
)
_SWAMPERT = FixturePokemon(
    species="Swampert", moves=("earthquake", "surf"), ability="Torrent",
    item="Leftovers", level=84, evs=dict(_EVS),
)
# The donor. Protect is deliberately at slot 3, so a lock that silently falls
# back to slot 0 (Body Slam) is distinguishable from one that resolved by id.
_DELCATTY = FixturePokemon(
    species="Delcatty", moves=("bodyslam", "healbell", "wish", "protect"),
    ability="Cute Charm", item="Leftovers", level=96, evs=dict(_EVS),
)
_DELCATTY_HP = FixturePokemon(
    species="Delcatty", moves=("bodyslam", "healbell", "wish", "hiddenpower"),
    ability="Cute Charm", item="Leftovers", level=96, evs=dict(_EVS),
)


def _override(donor: FixturePokemon = _DELCATTY) -> BattleStartOverride:
    return BattleStartOverride(
        player_teams={
            "p1": pack_team((_DITTO, _SWAMPERT)),
            "p2": pack_team((donor, _SWAMPERT)),
        },
    )


def _maxhp(mon: FixturePokemon, dex: ShowdownDex) -> int:
    info = dex.species_info(mon.species)
    return gen3_hp_stat(
        int(info.base_stats["hp"]), 31, int((mon.evs or {}).get("hp", 0)), mon.level
    )


def _payload(dex: ShowdownDex, *, donor: FixturePokemon = _DELCATTY, **overrides):
    """The 19100170 shape: our own Ditto is transformed, encored, and active.

    ``sides.p1.pokemon[0].moves`` is the pre-Transform snapshot (``transform``
    only). ``selfActiveMoves`` is the raw request: the copied moveset with
    Encore's disable pattern.
    """

    ditto_hp = _maxhp(_DITTO, dex)
    donor_hp = _maxhp(donor, dex)
    payload = {
        "turn": 64,
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
        "selfTeamOrder": ["Ditto", "Swampert"],
        "selfActiveRequestState": {
            "trapped": False, "maybeTrapped": False,
            "maybeDisabled": False, "maybeLocked": False,
        },
        "selfBenchedMoveHistory": False,
        # Encore's request signature on the COPIED moveset: everything but the
        # locked move is disabled. Protect is donor slot 3.
        "selfActiveMoves": [
            {"id": "bodyslam", "pp": 5, "maxpp": 24, "disabled": True},
            {"id": "healbell", "pp": 5, "maxpp": 5, "disabled": True},
            {"id": "wish", "pp": 5, "maxpp": 10, "disabled": True},
            {"id": "protect", "pp": 2, "maxpp": 10, "disabled": False},
        ],
        "sides": {
            "p1": {
                "pokemon": [
                    {
                        "species": "Ditto",
                        "condition": f"{ditto_hp - 10}/{ditto_hp}",
                        "active": True,
                        # PRE-Transform snapshot, by design. One move.
                        "moves": [
                            {"id": "transform", "pp": 15, "maxpp": 16, "disabled": False},
                        ],
                    },
                    {"species": "Swampert", "condition": "0 fnt", "active": False, "moves": []},
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
                        "species": donor.species,
                        "condition": f"{donor_hp}/{donor_hp}",
                        "active": True,
                    },
                ],
                "boosts": {},
                "volatiles": [],
                "lastUsedMove": "bodyslam",
                "materializationBlockers": [],
                "toxicStage": 0,
                "sideConditions": {},
                "sideConditionSetTurns": {},
            },
        },
    }
    payload.update(overrides)
    return payload


def _override_opponent_transformed() -> BattleStartOverride:
    """Mirror image of ``_override``: the DONOR is ours, the Ditto is theirs."""

    return BattleStartOverride(
        player_teams={
            "p1": pack_team((_DELCATTY, _SWAMPERT)),
            "p2": pack_team((_DITTO, _SWAMPERT)),
        },
    )


def _payload_opponent_transformed(dex: ShowdownDex, **overrides):
    """We are p1; the OPPOSING active is a Ditto transformed into our Delcatty.

    ``sides.p2.lastUsedMove`` is deliberately ``protect`` -- the id the
    caller-supplied ``encored_move`` would also give -- so that a self-seat
    fallback leaking onto the opponent seat is visible as a build where a
    refusal is required.
    """

    delcatty_hp = _maxhp(_DELCATTY, dex)
    ditto_hp = _maxhp(_DITTO, dex)
    active_rows = [
        {"id": "bodyslam", "pp": 20, "maxpp": 24, "disabled": False},
        {"id": "healbell", "pp": 5, "maxpp": 5, "disabled": False},
        {"id": "wish", "pp": 8, "maxpp": 10, "disabled": False},
        {"id": "protect", "pp": 8, "maxpp": 10, "disabled": False},
    ]
    payload = {
        "turn": 40,
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
        "selfActiveRequestState": {
            "trapped": False, "maybeTrapped": False,
            "maybeDisabled": False, "maybeLocked": False,
        },
        "selfBenchedMoveHistory": False,
        "selfActiveMoves": list(active_rows),
        "sides": {
            "p1": {
                "pokemon": [
                    {
                        "species": "Delcatty",
                        "condition": f"{delcatty_hp}/{delcatty_hp}",
                        "active": True,
                        "moves": list(active_rows),
                    },
                    {"species": "Swampert", "condition": "0 fnt", "active": False, "moves": []},
                ],
                "boosts": {},
                "volatiles": [],
                "lastUsedMove": "bodyslam",
                "materializationBlockers": [],
                "toxicStage": 0,
                "sideConditions": {},
                "sideConditionSetTurns": {},
            },
            "p2": {
                "pokemon": [
                    {
                        "species": "Ditto",
                        "condition": f"{ditto_hp}/{ditto_hp}",
                        "active": True,
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
        },
    }
    payload.update(overrides)
    return payload


class EncoreOnATransformedActiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dex = _dex()

    def _build(self, payload, *, donor: FixturePokemon = _DELCATTY, **kwargs):
        return battle_spec_from_payload(
            payload,
            _override(donor),
            dex=self.dex,
            transformed_slots={"p1": donor.species},
            **kwargs,
        )

    def test_the_fixture_really_is_the_spurious_one_move_snapshot(self) -> None:
        """Guard against a vacuous pass.

        The whole defect needs the pre-Transform row to hold EXACTLY one enabled
        move, so that the old self-seat rule fires on it. If a later edit gives
        Ditto two moves here, the main assertion would pass for the wrong
        reason -- the old code would return None and fail closed instead of
        returning slot 0.
        """

        payload = _payload(self.dex)
        rows = payload["sides"]["p1"]["pokemon"][0]["moves"]
        enabled = [row for row in rows if not row["disabled"]]
        self.assertEqual(len(enabled), 1)
        self.assertEqual(enabled[0]["id"], "transform")
        # ...and the move it spuriously identifies is NOT the encored one.
        self.assertNotEqual(enabled[0]["id"], "protect")

    def test_transformed_self_encore_locks_the_post_transform_slot(self) -> None:
        """RED before the fix at ``move:0``; the locked move is donor slot 3."""

        world = self._build(_payload(self.dex))
        side = world.spec.side_one
        active = side.pokemon[side.active_index]
        # The copy really happened, in the donor's own slot order.
        self.assertEqual(
            [spec.id for spec in active.moves],
            ["bodyslam", "healbell", "wish", "protect"],
        )
        self.assertIn("encore", side.volatile_statuses)
        self.assertIn("transformed", side.volatile_statuses)
        self.assertEqual(side.last_used_move, "move:3")
        self.assertEqual(dict(side.volatile_status_durations), {"encore": 1})

    def test_transformed_self_encore_falls_back_to_last_used_move(self) -> None:
        """No ``selfActiveMoves`` -> the payload's public ``lastUsedMove`` (slot 2)."""

        payload = _payload(self.dex)
        payload["selfActiveMoves"] = []
        payload["sides"]["p1"]["lastUsedMove"] = "wish"
        side = self._build(payload).spec.side_one
        self.assertEqual(side.last_used_move, "move:2")

    def test_transformed_self_encore_ignores_the_pre_transform_snapshot(self) -> None:
        """Even a pre-Transform row that names a REAL donor move is not consulted.

        Ditto's own snapshot is rewritten to a lone enabled ``bodyslam`` -- donor
        slot 0, i.e. exactly the answer the old code produced by accident. The
        id-keyed sources still say Protect, so the lock must still be slot 3.
        """

        payload = _payload(self.dex)
        payload["sides"]["p1"]["pokemon"][0]["moves"] = [
            {"id": "bodyslam", "pp": 15, "maxpp": 16, "disabled": False},
        ]
        side = self._build(payload).spec.side_one
        self.assertEqual(side.last_used_move, "move:3")

    def test_transformed_self_encore_fails_closed_on_an_absent_move(self) -> None:
        """Fail-closed is PRESERVED: an id outside the copied moveset refuses."""

        payload = _payload(self.dex)
        payload["selfActiveMoves"] = [
            {"id": "surf", "pp": 5, "maxpp": 24, "disabled": False},
        ]
        payload["sides"]["p1"]["lastUsedMove"] = "surf"
        with self.assertRaises(EngineWorldUnsupported) as caught:
            self._build(payload)
        self.assertEqual(caught.exception.reason, "encore_move_unknown")

    def test_transformed_self_encore_fails_closed_when_no_id_is_available(self) -> None:
        payload = _payload(self.dex)
        payload["selfActiveMoves"] = []
        payload["sides"]["p1"]["lastUsedMove"] = None
        with self.assertRaises(EngineWorldUnsupported) as caught:
            self._build(payload)
        self.assertEqual(caught.exception.reason, "encore_move_unknown")

    def test_transformed_self_encore_keeps_the_hiddenpower_prefix_tolerance(self) -> None:
        """The request names the typed variant; the spec carries its own id."""

        payload = _payload(self.dex, donor=_DELCATTY_HP)
        payload["selfActiveMoves"] = [
            {"id": "bodyslam", "pp": 5, "maxpp": 24, "disabled": True},
            {"id": "healbell", "pp": 5, "maxpp": 5, "disabled": True},
            {"id": "wish", "pp": 5, "maxpp": 10, "disabled": True},
            {"id": "hiddenpowerice70", "pp": 5, "maxpp": 24, "disabled": False},
        ]
        payload["sides"]["p1"]["lastUsedMove"] = "hiddenpowerice70"
        world = self._build(payload, donor=_DELCATTY_HP)
        side = world.spec.side_one
        active = side.pokemon[side.active_index]
        self.assertEqual(side.last_used_move, "move:3")
        self.assertTrue(active.moves[3].id.startswith("hiddenpower"))
        # ...and the tolerance is doing real work: the two ids differ.
        self.assertNotEqual(active.moves[3].id, "hiddenpowerice70")

    def test_an_untransformed_self_encore_is_untouched(self) -> None:
        """The non-Transform path still resolves from the disabled pattern.

        Same payload, no ``transformed_slots``: Ditto keeps its own moveset, the
        pre-Transform row is the CURRENT row, and its single enabled ``transform``
        legitimately identifies the lock at slot 0.
        """

        payload = _payload(self.dex)
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        side = world.spec.side_one
        active = side.pokemon[side.active_index]
        self.assertEqual(
            [spec.id for spec in active.moves if spec.id != "none"], ["transform"]
        )
        self.assertNotIn("transformed", side.volatile_statuses)
        self.assertEqual(side.last_used_move, "move:0")

    def test_two_enabled_request_moves_do_not_identify_a_lock(self) -> None:
        """M6 pin, integration half: the rule is ONE enabled move, not "the first".

        With two enabled entries the request has not disclosed the lock, so
        ``selfActiveMoves`` must be discarded and the public ``lastUsedMove``
        used instead. A "first enabled wins" reading picks ``bodyslam`` (slot 0)
        -- which is exactly the wrong answer this whole fix exists to stop
        being produced by accident.
        """

        payload = _payload(self.dex)
        payload["selfActiveMoves"] = [
            {"id": "bodyslam", "pp": 5, "maxpp": 24, "disabled": False},
            {"id": "healbell", "pp": 5, "maxpp": 5, "disabled": True},
            {"id": "wish", "pp": 5, "maxpp": 10, "disabled": False},
            {"id": "protect", "pp": 2, "maxpp": 10, "disabled": True},
        ]
        payload["sides"]["p1"]["lastUsedMove"] = "wish"
        side = self._build(payload).spec.side_one
        self.assertEqual(side.last_used_move, "move:2")


class EncoreOnATransformedOpponentTests(unittest.TestCase):
    """The OPPONENT seat is deferred too, and that CHANGED its coverage.

    Deferral is not self-seat-only: any active in ``transformed_slots`` takes the
    new path. On the opponent seat the id still comes from the caller's
    ``encored_move`` -- the ``selfActiveMoves`` / ``lastUsedMove`` fallbacks are
    self-seat only -- but the SLOT is now resolved against the copied moveset
    instead of the transformer's own.

    That is a coverage change, and it is the one the PR body originally and
    wrongly claimed had been avoided. Before this fix the same input REFUSED:
    ``_move_index_by_id`` ran against Ditto's own ``[transform]``, ``protect``
    was absent, and construction raised ``encore_move_unknown`` -- a counted
    skip. It now builds. The direction is right (a correct world beats a
    refusal) and it still fails closed on an id the copy does not contain, but
    it can convert an ``encore_move_unknown`` skip into a measured boundary. It
    did not fire in either 200-game window; these tests exist so the path is
    pinned rather than merely unobserved.
    """

    def setUp(self) -> None:
        self.dex = _dex()

    def _build(self, **kwargs):
        return battle_spec_from_payload(
            _payload_opponent_transformed(self.dex),
            _override_opponent_transformed(),
            dex=self.dex,
            transformed_slots={"p2": "Delcatty"},
            **kwargs,
        )

    def test_transformed_opponent_encore_locks_the_post_transform_slot(self) -> None:
        world = self._build(encored_moves={"p2": "protect"})
        side = world.spec.side_two
        active = side.pokemon[side.active_index]
        self.assertEqual(
            [spec.id for spec in active.moves],
            ["bodyslam", "healbell", "wish", "protect"],
        )
        self.assertIn("encore", side.volatile_statuses)
        self.assertIn("transformed", side.volatile_statuses)
        self.assertEqual(side.last_used_move, "move:3")

    def test_transformed_opponent_encore_fails_closed_on_an_absent_move(self) -> None:
        """``surf`` is on the bench Swampert, never on the copied Delcatty."""

        with self.assertRaises(EngineWorldUnsupported) as caught:
            self._build(encored_moves={"p2": "surf"})
        self.assertEqual(caught.exception.reason, "encore_move_unknown")

    def test_transformed_opponent_encore_still_needs_a_caller_supplied_move(self) -> None:
        """No ``encored_move`` still refuses: the self-seat fallbacks are self-seat.

        ``sides.p2.lastUsedMove`` is set to ``protect`` in this fixture, so if
        the ``lastUsedMove`` fallback ever leaked across seats this would build
        at ``move:3`` instead of raising.
        """

        with self.assertRaises(EngineWorldUnsupported) as caught:
            self._build()
        self.assertEqual(caught.exception.reason, "encore_move_unknown")


class SoleEnabledMoveIdTests(unittest.TestCase):
    """M6 pin, unit half.

    ``_sole_enabled_move_id`` is the guard whose SPURIOUS satisfaction is this
    defect: a one-move pre-Transform snapshot has exactly one enabled entry and
    so "identified" a lock nobody set. Relaxing "exactly one" to "the first of
    many" leaves every other test in this file and in ``test_engine_world``
    green, so the invariant is pinned directly.
    """

    def test_exactly_one_enabled_row_identifies_the_lock(self) -> None:
        rows = [
            {"id": "bodyslam", "disabled": True},
            {"id": "protect", "disabled": False},
        ]
        self.assertEqual(_sole_enabled_move_id(rows), "protect")

    def test_two_enabled_rows_identify_nothing(self) -> None:
        rows = [
            {"id": "bodyslam", "disabled": False},
            {"id": "healbell", "disabled": True},
            {"id": "protect", "disabled": False},
        ]
        self.assertIsNone(_sole_enabled_move_id(rows))

    def test_four_enabled_rows_identify_nothing(self) -> None:
        """The ordinary un-encored request: every move usable, no lock at all."""

        rows = [{"id": name, "disabled": False}
                for name in ("bodyslam", "healbell", "wish", "protect")]
        self.assertIsNone(_sole_enabled_move_id(rows))

    def test_no_enabled_rows_identify_nothing(self) -> None:
        rows = [{"id": "bodyslam", "disabled": True}, {"id": "protect", "disabled": True}]
        self.assertIsNone(_sole_enabled_move_id(rows))

    def test_empty_and_missing_rows_identify_nothing(self) -> None:
        self.assertIsNone(_sole_enabled_move_id([]))
        self.assertIsNone(_sole_enabled_move_id(None))

    def test_the_non_transformed_path_inherits_the_same_rule(self) -> None:
        """``_resolve_encored_move_index`` must not guess from an ambiguous row set.

        This is the pre-existing invariant on the untouched path: two enabled
        moves mean the request has not disclosed a lock, so the caller must fail
        closed rather than take slot 0.
        """

        specs = [
            MoveSpec(id="bodyslam", pp=24),
            MoveSpec(id="healbell", pp=5),
            MoveSpec(id="protect", pp=10),
        ]
        ambiguous = [
            {"id": "bodyslam", "disabled": False},
            {"id": "protect", "disabled": False},
        ]
        self.assertIsNone(
            _resolve_encored_move_index(
                specs, rows_for_active=ambiguous, encored_move=None
            )
        )
        # Control: the same call with a real one-enabled pattern does resolve,
        # so the None above is the ambiguity rule and not a broken fixture.
        unambiguous = [
            {"id": "bodyslam", "disabled": True},
            {"id": "protect", "disabled": False},
        ]
        self.assertEqual(
            _resolve_encored_move_index(
                specs, rows_for_active=unambiguous, encored_move=None
            ),
            2,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
