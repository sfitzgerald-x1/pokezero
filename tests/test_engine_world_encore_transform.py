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

C145 adds ``LockIndexToResidualBlockTests``, which closes the chain that the
tests above stop short of. Everything above ends at the INDEX; nothing asserted
that the index change is what puts Showdown's ``|-heal|...|[from] item:
Leftovers`` line back. That link is the whole reason the row was divergent, and
without it a future change could keep ``move:3`` and still lose the tick.
Measured red/green in ``reports/c145_itemleftovers_row_adjudication.md`` §5.
"""

from __future__ import annotations

import json
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
from pokezero.poke_engine_adapter import MoveSpec, build_poke_engine_state  # noqa: E402
from pokezero.showdown_fixture import FixturePokemon, pack_team  # noqa: E402

# Imported unguarded, on purpose. Every other consumer of the native crate in
# tests/ wraps this in try/except and skips when the wheel is missing, which is
# right for a suite that is mostly crate-independent. Here it is not: a skip
# would silently retire the only pin that ties the Encore lock index to the
# residual block, and this program has twice read `OK (skipped=N)` as a pass.
# An ImportError is the correct, loud outcome of running this file without a
# built engine.
import poke_engine  # noqa: E402
import pokezero_search  # noqa: E402


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


class LockIndexToResidualBlockTests(unittest.TestCase):
    """C145: the lock index is only interesting because of what it SUPPRESSES.

    The `19100170/71-72` rows were never "the Encore index is wrong". They were
    ``component_missing_in_engine:itemleftovers`` at ``pct=100.00`` over a single
    branch: the phantom Body Slam KOs the incoming mon, ``end_of_turn_is_deferred``
    defers the whole residual block to the forced-replacement boundary, and BOTH
    sides' Leftovers ticks vanish from a turn where Showdown emits them.

    So this class renders the CONSTRUCTED world through the same mapper the
    differential's ``evaluate_boundary_strict`` calls, and asserts the heal line
    is there. Measured on `dc6e1e19` (the fix's parent) the same fixture builds
    ``move:0``, renders a faint, and emits no ``item: Leftovers`` event at all --
    red -- and green from `d27316b6` onward. The engine build is BYTE-IDENTICAL
    across that pair (fingerprint ``fdbf5937...``), which is the point: the
    closing change is world construction, so an engine-only pin cannot see it.

    Note the shape being pinned is a joint action where side one's move is
    IRRELEVANT to the opponent -- Protect against a switch. That is what makes
    the phantom lethal: the correct turn deals no damage whatsoever, so any
    damage the engine invents is a pure fabrication rather than a mispriced roll.
    """

    def setUp(self) -> None:
        self.dex = _dex()

    def _payload_with_a_lethal_switch_in(self):
        """The 19100170 shape, plus the two things the render needs.

        1. The transformed Ditto is 40 below max, so its Leftovers tick is the
           full ``maxhp // 16`` and not a clamp against the cap. A fixture that
           healed 0 would pass the "no faint" half while asserting nothing.
        2. The opponent has a benched mon at 5 HP to switch in, so the phantom
           Body Slam is LETHAL and actually trips the deferral. At full HP the
           block would still run and the pin would be vacuous -- which is what
           ``test_a_survivable_phantom_would_not_have_hidden_the_tick`` proves.
        """

        payload = _payload(self.dex)
        ditto_hp = _maxhp(_DITTO, self.dex)
        target_hp = _maxhp(_SWAMPERT, self.dex)
        payload["sides"]["p1"]["pokemon"][0]["condition"] = f"{ditto_hp - 40}/{ditto_hp}"
        payload["sides"]["p2"]["pokemon"].append(
            {"species": "Swampert", "condition": f"5/{target_hp}",
             "active": False, "moves": []}
        )
        return payload

    def _render(self, payload, *, force_last_used_move: str | None = None,
                expect_branches: int = 1):
        """Build the world, then render the joint action (protect, switch).

        ``force_last_used_move`` rewrites the built state's side-one
        ``last_used_move`` field in place. It exists ONLY for the negative
        control: it lets one test show that the identical engine build produces
        the suppressed turn when handed ``move:0``, so the difference between
        red and green is the constructed index and nothing else.

        ``expect_branches`` is asserted rather than defaulted-and-ignored. Both
        turns that reproduce the row are single-branch (the sweep recorded
        ``branch_count: 1``); the survivable-phantom control is deliberately NOT,
        because an invented attack that leaves the target alive splits into a
        roll fan, and pretending otherwise would hide the fan behind a
        highest-mass pick.
        """

        world = battle_spec_from_payload(
            payload, _override(), dex=self.dex, transformed_slots={"p1": _DELCATTY.species}
        )
        side_one, side_two = world.spec.side_one, world.spec.side_two
        state = build_poke_engine_state(world.spec, module=poke_engine)
        serialized = state.to_string()
        if force_last_used_move is not None:
            current = f"={side_one.last_used_move}="
            self.assertIn(current, serialized)
            serialized = serialized.replace(current, f"={force_last_used_move}=", 1)
        context = json.dumps({
            "p1": [mon.id for mon in side_one.pokemon],
            "p2": [mon.id for mon in side_two.pokemon],
            "turn": int(payload["turn"]),
        })
        rendered = json.loads(pokezero_search.branch_events(
            serialized, "protect", "swampert", context, True, True
        ))
        branches = rendered.get("branches") or []
        self.assertEqual(len(branches), expect_branches)
        for branch in branches:
            # A lossy render is the mapper declaring it cannot reproduce the
            # turn; the differential treats that as unmeasurable rather than
            # divergent, so a pin that allowed it would assert nothing.
            self.assertEqual(list(branch.get("lossy") or []), [])
        self.assertAlmostEqual(
            sum(float(branch["percentage"]) for branch in branches), 100.0, places=3
        )
        return world, [list(branch.get("events") or []) for branch in branches]

    def _sole_events(self, payload, **kwargs):
        """The single 100 %-mass branch, as the sweep recorded it for these rows."""

        world, per_branch = self._render(payload, expect_branches=1, **kwargs)
        return world, per_branch[0]

    def test_the_fixture_really_needs_the_residual_block(self) -> None:
        """Guard against a vacuous pass on either half of the render."""

        payload = self._payload_with_a_lethal_switch_in()
        world = battle_spec_from_payload(
            payload, _override(), dex=self.dex, transformed_slots={"p1": _DELCATTY.species}
        )
        active = world.spec.side_one.pokemon[world.spec.side_one.active_index]
        self.assertEqual(active.item, "leftovers")
        # Below max by MORE than one tick, so the heal is the full tick.
        self.assertGreater(active.maxhp - active.hp, active.maxhp // 16)
        self.assertGreater(active.maxhp // 16, 0)

    def test_the_constructed_world_renders_showdowns_leftovers_tick(self) -> None:
        """RED at `dc6e1e19`: no ``item: Leftovers`` event, and a faint instead.

        This is the row's actual symptom, asserted end to end from the payload
        the differential hands the constructor.
        """

        world, events = self._sole_events(self._payload_with_a_lethal_switch_in())
        side_one = world.spec.side_one
        active = side_one.pokemon[side_one.active_index]

        heals = [line for line in events if "[from] item: Leftovers" in line]
        # Both sides tick. The sweep's miss named only p1 because
        # `evaluate_boundary_strict` breaks out of its ("p1", "p2") slot loop on
        # the first failure -- p2's tick was lost too, and was never compared.
        self.assertEqual(len(heals), 2)
        self.assertIn(
            f"|-heal|p1a: {active.id}|{active.hp + active.maxhp // 16}/{active.maxhp}"
            "|[from] item: Leftovers",
            heals,
        )
        # The suppressor itself: no faint, so `end_of_turn_is_deferred` is not
        # armed and the block runs on THIS boundary.
        self.assertEqual([line for line in events if line.startswith("|faint|")], [])
        self.assertIn("|upkeep", events)
        # And Protect is what side one actually did, not Body Slam.
        self.assertIn(f"|move|p1a: {active.id}|protect||[still]", events)
        # Corroboration, asserted LAST on purpose. Checking the index first would
        # make this test fail with the same message as the index test above and
        # hide the symptom it exists to name.
        self.assertEqual(side_one.last_used_move, "move:3")

    def test_the_pre_fix_index_suppresses_the_block_on_the_same_build(self) -> None:
        """The negative control, and the reason this is a world-construction row.

        Same fixture, same engine build, same joint action -- only the side-one
        ``last_used_move`` differs. ``move:0`` reproduces the divergence exactly:
        a phantom Body Slam, a faint, and NO residual block. So nothing in the
        engine changed between the red and green eras; the constructed index did.
        """

        _world, events = self._sole_events(
            self._payload_with_a_lethal_switch_in(), force_last_used_move="move:0"
        )
        self.assertEqual([line for line in events if "item: Leftovers" in line], [])
        self.assertEqual(len([line for line in events if line.startswith("|faint|")]), 1)
        self.assertNotIn("|upkeep", events)
        self.assertTrue(any("|bodyslam|" in line for line in events))

    def test_a_survivable_phantom_would_not_have_hidden_the_tick(self) -> None:
        """Why the row needed a LETHAL phantom, not merely a wrong one.

        The same wrong lock against a healthy switch-in still emits the residual
        block, so the boundary would have diverged on the invented damage instead
        -- a different class. This is the control that keeps the fixture's 5 HP
        from looking arbitrary.
        """

        payload = self._payload_with_a_lethal_switch_in()
        target_hp = _maxhp(_SWAMPERT, self.dex)
        payload["sides"]["p2"]["pokemon"][-1]["condition"] = f"{target_hp}/{target_hp}"
        # A survivable invented attack is a roll fan, so every arm is checked
        # rather than the most likely one.
        _world, per_branch = self._render(
            payload, force_last_used_move="move:0", expect_branches=4
        )
        for events in per_branch:
            self.assertEqual([line for line in events if line.startswith("|faint|")], [])
            self.assertEqual(
                len([line for line in events if "[from] item: Leftovers" in line]), 2
            )
            self.assertTrue(any("|bodyslam|" in line for line in events))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
