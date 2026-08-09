"""A Struggle-only request must not be answered with an older request's moveset.

THE DEFECT. ``sides[self].pokemon[].moves`` in the direct-materialization payload comes
only from ``_request_materialization_rows``, whose move state is
``actor_move_states_from_request_history`` -- a fold that retains the most recent request
per own Pokemon and SKIPS any request where ``_request_active_moves`` is empty, because
that helper drops rows lacking integer ``pp`` AND ``maxpp``. Showdown's Struggle branch
emits exactly one such row, so the row stayed pinned to the last pp-BEARING request and
advertised a usable move at a boundary where Showdown offers only Struggle. Meanwhile
``selfActiveMoves``, built from the CURRENT request one call later, correctly said ``[]``.
One payload, two views of the same request, built from two different requests.

MEASURED, and this repo's first live capture of the branch. The provenance list in
``actor_move_states_from_request_history`` had classified Struggle as "SOURCE ONLY --
unverified by measurement; an attempt to produce a live Struggle request failed on
packed-team format". A gen3 Custom Game with a single-move Bulbasaur (Sunny Day, 8 PP)
produces it on turn 8. At that boundary, on ``origin/main``::

    RAW REQUEST active moves : [{"move": "Struggle", "id": "struggle",
                                 "target": "randomNormal", "disabled": false}]
    VIEW A sides.p1.pokemon  : [{"species": "Bulbasaur", "active": true,
                                 "moves": [{"id": "sunnyday", "pp": 1, "maxpp": 8,
                                            "disabled": false}]}, ...]
    VIEW B selfActiveMoves   : []

WHY THE FIX IS AT THE PAYLOAD BOUNDARY AND NOT IN THE FOLD. The first version of this
change marked the retained FOLD entry, and that was wrong in a way only a live run showed.
Showdown clears ``moveSlot.disabled`` on switch-out and recomputes it every turn, so
unusability belongs to ONE BOUNDARY; a fold entry outlives the request that wrote it.
Measured on that version -- Bulbasaur Taunted into Struggle, then switched out::

    fold version : bulbasaur (benched) [('sunnyday', 8, True),  ('growth', 64, True)]
    origin/main  : bulbasaur (benched) [('sunnyday', 8, False), ('growth', 64, False)]

-- full PP and no legal move in any searched line, because poke-engine's
``re_enable_disabled_moves`` runs on the OUTGOING active only. ``BenchLeakTests`` is the
regression pin for that; it fails against the fold placement.

WHICH STRUGGLE POPULATION ACTUALLY BUILDS A WORLD, enumerated rather than assumed. Every
gen3 ``disableMove`` caller is ``disable``, ``encore``, ``imprison``, ``taunt``,
``torment``. ``imprison`` reports ``disabled: 'hidden'``, which ``getMoves`` resolves to
``false`` in singles; ``disable``/``torment`` are in ``showdown.TRACKED_VOLATILES`` but
not in ``engine_world._SUPPORTED_VOLATILES``, so those boundaries raise
``volatile_unsupported`` and never reach move construction; and gen3 Encore's
``onResidual`` removes the volatile the same turn its move hits 0 PP, so an Encored
Struggle request does not exist.

⚠ ``taunt`` WAS IN THAT REFUSED LIST AND NO LONGER IS. It joined
``_SUPPORTED_VOLATILES`` (counter seeded at one tick elapsed) once the gen3 engine was
shown to model the volatile exactly, so a Taunt-induced Struggle-only request now BUILDS
a world and reaches move construction like a PP-exhausted one. Two populations now, not
one.

The PRE-EXISTING 19 tests here are unchanged by that and deliberately so: their subject
is the PAYLOAD view, every assertion is about ``sides[self].pokemon[].moves`` versus
``selfActiveMoves``, and both routes produce the identical payload. Measured, not
assumed: all 19 are green in BOTH worlds, i.e. they are completely insensitive to the
volatile change and none of them is evidence for it.

The evidence that the Taunt route is closed is therefore written where it can FAIL:
``tests/test_engine_world_taunt.py`` (construction, counter, engine fidelity) and
``TauntStruggleOnlyReachesTheEngineTests`` below -- the lone-Blissey line, the only shape
whose request offers ``['struggle']`` and NOTHING else, so the engine's
``MoveChoice::None`` has to land on the pseudo-move. That 20th test is red in the null
world.

``_TAUNT_STALL`` (used by ``BenchLeakTests``) keeps a live bench on purpose and so does
NOT reach that shape; it exercises the payload only, which is what the bench leak is
about.
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

from pokezero.dex import load_showdown_dex  # noqa: E402
from pokezero.engine_world import _sole_enabled_move_id  # noqa: E402
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.local_showdown import (  # noqa: E402
    DEFAULT_SHOWDOWN_ROOT,
    LocalShowdownConfig,
    LocalShowdownEnv,
    _public_materialization_payload,
    _request_materialization_rows,
    _request_reports_only_struggle,
    actor_move_states_from_request_history,
)
from pokezero.showdown_fixture import FixturePokemon, pack_team  # noqa: E402


# --------------------------------------------------------------------------------------
# The predicate, and the shapes it must NOT claim
# --------------------------------------------------------------------------------------

STRUGGLE_ROW = {
    "move": "Struggle",
    "id": "struggle",
    "target": "randomNormal",
    "disabled": False,
}


def _request_with_active_moves(moves):
    return {"active": [{"moves": moves}], "side": {"pokemon": []}}


class StruggleBranchPredicateTests(unittest.TestCase):
    def test_the_captured_branch_is_recognised(self) -> None:
        self.assertTrue(
            _request_reports_only_struggle(_request_with_active_moves([STRUGGLE_ROW]))
        )

    def test_the_fixture_really_is_pp_less(self) -> None:
        """Guard against a vacuous suite.

        If a later edit gives this row `pp`/`maxpp`, `_request_active_moves` would keep
        it, the fold would never skip, and the defect these tests describe would not
        exist -- so several of them would pass for the wrong reason.
        """

        self.assertNotIn("pp", STRUGGLE_ROW)
        self.assertNotIn("maxpp", STRUGGLE_ROW)

    def test_a_recharge_request_is_not_the_branch(self) -> None:
        """`mustrecharge` is dropped by the same pp filter but is NOT this.

        `getMoves` returns early on `lockedMove` (`sim/pokemon.ts:966`) without evaluating
        per-slot `disabled` at all, so its real slots are usually usable and merely
        pre-empted for one turn. A sibling change is routing it through the MUSTRECHARGE
        volatile; this boundary is pinned so a later widening is visible in the diff.
        """

        self.assertFalse(
            _request_reports_only_struggle(
                _request_with_active_moves([{"move": "Recharge", "id": "recharge"}])
            )
        )

    def test_a_locked_charge_move_is_not_the_branch(self) -> None:
        self.assertFalse(
            _request_reports_only_struggle(
                _request_with_active_moves([{"move": "Solar Beam", "id": "solarbeam"}])
            )
        )

    def test_more_than_one_row_is_not_the_branch(self) -> None:
        """Pins `len(moves) != 1`. Showdown's branch substitutes a list of exactly one."""

        self.assertFalse(
            _request_reports_only_struggle(
                _request_with_active_moves(
                    [STRUGGLE_ROW, {"id": "tackle", "pp": 5, "maxpp": 56, "disabled": False}]
                )
            )
        )

    def test_a_fully_pp_bearing_struggle_row_is_not_the_branch(self) -> None:
        self.assertFalse(
            _request_reports_only_struggle(
                _request_with_active_moves([{**STRUGGLE_ROW, "pp": 5, "maxpp": 5}])
            )
        )

    def test_a_half_pp_bearing_struggle_row_is_not_the_branch(self) -> None:
        """Pins the `and` in the pp check, which the both-fields case cannot.

        Requiring BOTH fields absent is deliberate: only the exact shape Showdown emits
        triggers the marking. An `or` here would accept a half-populated row that no
        branch of `getMoveRequestData` produces.
        """

        for partial in ({"pp": 5}, {"maxpp": 5}):
            with self.subTest(partial=partial):
                self.assertFalse(
                    _request_reports_only_struggle(
                        _request_with_active_moves([{**STRUGGLE_ROW, **partial}])
                    )
                )

    def test_an_absent_active_row_is_not_the_branch(self) -> None:
        for request in ({}, {"active": []}, {"active": [{}]}, {"active": [{"moves": []}]}):
            with self.subTest(request=request):
                self.assertFalse(_request_reports_only_struggle(request))


# --------------------------------------------------------------------------------------
# The row builder: scope of the marking
# --------------------------------------------------------------------------------------


def _team_request(active_moves, *, active_species="Bulbasaur"):
    return {
        "active": [{"moves": active_moves}],
        "side": {
            "pokemon": [
                {
                    "ident": "p1: Bulbasaur",
                    "details": "Bulbasaur, M",
                    "condition": "100/100",
                    "active": active_species == "Bulbasaur",
                    "moves": ["sunnyday", "growth"],
                },
                {
                    "ident": "p1: Charmander",
                    "details": "Charmander, F",
                    "condition": "100/100",
                    "active": active_species == "Charmander",
                    "moves": ["ember"],
                },
            ]
        },
    }


_STATES = {
    "bulbasaur": (
        {"id": "sunnyday", "pp": 1, "maxpp": 8, "disabled": False},
        {"id": "growth", "pp": 64, "maxpp": 64, "disabled": False},
    ),
    "charmander": ({"id": "ember", "pp": 40, "maxpp": 40, "disabled": False},),
}

_ORDINARY_ROW = {"id": "sunnyday", "pp": 1, "maxpp": 8, "disabled": False}


def _flags(rows):
    return {
        row["species"]: [(m["id"], m["pp"], m["disabled"]) for m in row["moves"]]
        for row in rows
    }


class RowMarkingScopeTests(unittest.TestCase):
    def test_a_struggle_request_disables_every_move_on_the_active_row(self) -> None:
        rows = _request_materialization_rows(
            _team_request([STRUGGLE_ROW]), self_move_states=_STATES
        )
        self.assertEqual(
            _flags(rows)["Bulbasaur"], [("sunnyday", 1, True), ("growth", 64, True)]
        )

    def test_the_benched_rows_are_untouched(self) -> None:
        """The scope the fold placement could not express.

        Nothing about a Struggle request says anything about a mon that is not active,
        and `_move_specs` copies `disabled` verbatim onto every team row.
        """

        rows = _request_materialization_rows(
            _team_request([STRUGGLE_ROW]), self_move_states=_STATES
        )
        self.assertEqual(_flags(rows)["Charmander"], [("ember", 40, False)])

    def test_an_ordinary_request_is_untouched(self) -> None:
        rows = _request_materialization_rows(
            _team_request([_ORDINARY_ROW]), self_move_states=_STATES
        )
        self.assertEqual(
            _flags(rows)["Bulbasaur"], [("sunnyday", 1, False), ("growth", 64, False)]
        )

    def test_the_retained_fold_state_is_not_written_through(self) -> None:
        """The marking must not reach `self_move_states`, which later boundaries reuse.

        `_request_materialization_rows` builds `dict(move)` copies; if a future edit marks
        the source instead, the bench leak returns by another route. This is the unit-level
        half of `BenchLeakTests`.
        """

        states = {k: tuple(dict(m) for m in v) for k, v in _STATES.items()}
        _request_materialization_rows(_team_request([STRUGGLE_ROW]), self_move_states=states)
        self.assertEqual(
            [(m["id"], m["disabled"]) for m in states["bulbasaur"]],
            [("sunnyday", False), ("growth", False)],
        )

    def test_a_later_boundary_from_the_same_state_is_unmarked(self) -> None:
        """What the write-through guard buys: the marking does not accumulate."""

        states = {k: tuple(dict(m) for m in v) for k, v in _STATES.items()}
        _request_materialization_rows(_team_request([STRUGGLE_ROW]), self_move_states=states)
        rows = _request_materialization_rows(
            _team_request([_ORDINARY_ROW]), self_move_states=states
        )
        self.assertEqual(
            _flags(rows)["Bulbasaur"], [("sunnyday", 1, False), ("growth", 64, False)]
        )

    def test_an_active_row_with_no_snapshot_invents_nothing(self) -> None:
        rows = _request_materialization_rows(
            _team_request([STRUGGLE_ROW]), self_move_states={}
        )
        self.assertEqual(_flags(rows), {"Bulbasaur": [], "Charmander": []})

    def test_duplicate_idents_only_mark_the_active_row(self) -> None:
        """`attract_snorlax` has two same-ident Blisseys sharing one retained entry.

        Keying the marking on `active` rather than on identity is what keeps one
        Blissey's Struggle out of the other's moveset.
        """

        request = {
            "active": [{"moves": [STRUGGLE_ROW]}],
            "side": {
                "pokemon": [
                    {"ident": "p1: Blissey", "details": "Blissey, F",
                     "condition": "100/100", "active": True, "moves": ["softboiled"]},
                    {"ident": "p1: Blissey", "details": "Blissey, F",
                     "condition": "90/100", "active": False, "moves": ["softboiled"]},
                ]
            },
        }
        states = {"blissey": ({"id": "softboiled", "pp": 8, "maxpp": 16, "disabled": False},)}
        rows = _request_materialization_rows(request, self_move_states=states)
        self.assertEqual(
            [(row["active"], row["moves"][0]["disabled"]) for row in rows],
            [(True, True), (False, False)],
        )


class FoldIsUnchangedTests(unittest.TestCase):
    """The fold stays pure history: a pp-less request contributes nothing to it.

    Not a leftover assertion -- this is the property whose violation produced the bench
    leak, so it is pinned rather than assumed.
    """

    def test_a_struggle_request_leaves_the_fold_alone(self) -> None:
        team = [
            {"ident": "p1: Bulbasaur", "details": "Bulbasaur, M", "active": True,
             "moves": ["sunnyday", "growth"]},
        ]
        pp_request = {
            "active": [{"moves": [
                {"id": "sunnyday", "pp": 1, "maxpp": 8, "disabled": False},
                {"id": "growth", "pp": 64, "maxpp": 64, "disabled": False},
            ]}],
            "side": {"pokemon": team},
        }
        struggle_request = {"active": [{"moves": [STRUGGLE_ROW]}], "side": {"pokemon": team}}

        states = actor_move_states_from_request_history(
            [pp_request, struggle_request], initial_request=pp_request
        )
        self.assertEqual(
            [(m["id"], m["pp"], m["disabled"]) for m in states["bulbasaur"]],
            [("sunnyday", 1, False), ("growth", 64, False)],
        )


# --------------------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------------------


def _integration_config() -> LocalShowdownConfig | None:
    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=20.0)


_CHARMANDER = FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember",))

# PP EXHAUSTION -- the only Struggle population that builds a world (see module docstring).
# One move, 5 base PP -> 8 with full PP ups, so turn 8 is Struggle. Sunny Day is harmless:
# neither side faints and the battle stays at a move request throughout.
_PP_STALL = BattleStartOverride(
    player_teams={
        "p1": pack_team(
            (FixturePokemon(species="Bulbasaur", ability="Overgrow", moves=("Sunny Day",)),
             _CHARMANDER)
        ),
        "p2": pack_team(
            (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Harden",)),)
        ),
    },
)
# TAUNT -- reaches Struggle on turn 1 instead of turn 8, which is what makes the bench-leak
# probe short. When this was written the boundary was refused downstream by
# `volatile_unsupported: taunt`, so it exercised the payload only -- which is exactly what
# the bench leak is about, and still all this fixture asserts. `taunt` is now in
# `engine_world._SUPPORTED_VOLATILES` (see tests/test_engine_world_taunt.py), so the same
# boundary DOES build a world now; that does not change what is checked here.
# TAUNT, WITH NO BENCH -- the only shape whose request is `['struggle']` and nothing
# else. `_TAUNT_STALL` below keeps a Charmander, so Showdown offers a switch alongside
# Struggle and poke-engine's `add_switches` has something to enumerate; with a LONE
# all-status Blissey both are empty and `get_all_options` reaches its terminal
# `MoveChoice::None` push. Blissey, not Bulbasaur, because every one of its four moves
# must be Status for Taunt alone to empty the moveset -- a PP-bearing attacking slot
# would keep the request non-Struggle.
_TAUNT_SOLO_STALL = BattleStartOverride(
    player_teams={
        "p1": pack_team(
            (FixturePokemon(species="Blissey", ability="Natural Cure",
                            moves=("Soft-Boiled", "Toxic", "Light Screen", "Sing")),)
        ),
        "p2": pack_team(
            (FixturePokemon(species="Smeargle", ability="Own Tempo",
                            moves=("Taunt", "Tackle")),)
        ),
    },
)
_TAUNT_STALL = BattleStartOverride(
    player_teams={
        "p1": pack_team(
            (FixturePokemon(species="Bulbasaur", ability="Overgrow",
                            moves=("Sunny Day", "Growth")),
             _CHARMANDER)
        ),
        "p2": pack_team(
            (FixturePokemon(species="Squirtle", ability="Torrent",
                            moves=("Taunt", "Water Gun")),)
        ),
    },
)


def _enabled_ids(rows) -> list[str]:
    return [row["id"] for row in rows if not row.get("disabled")]


def _self_rows(payload):
    return {
        row["species"]: (
            row["active"],
            [(m["id"], m["pp"], m["disabled"]) for m in row["moves"]],
        )
        for row in payload["sides"]["p1"]["pokemon"]
    }


class _LiveBase(unittest.TestCase):
    def setUp(self) -> None:
        config = _integration_config()
        if config is None:
            self.skipTest("a built pokemon-showdown checkout and node are required")
        self.config = config

    @staticmethod
    def _struggle_rows(env):
        request = env._latest_requests.get("p1")
        active = (request or {}).get("active")
        rows = (active[0] if isinstance(active, list) and active else {}).get("moves")
        return rows if rows and rows[0].get("id") == "struggle" else None

    def _advance_to_struggle(self, env, *, turns: int):
        for _ in range(turns):
            rows = self._struggle_rows(env)
            if rows is not None:
                return rows
            env.step({"p1": 0, "p2": 0})
        raise AssertionError(f"no Struggle request was produced in {turns} turns")


class StruggleOnlyPayloadViewTests(_LiveBase):
    """The two views of the SAME request must agree about what is usable.

    Two assertions, not one: a fix that repairs `sides[...].moves` and leaves
    `selfActiveMoves` alone still ships an inconsistent payload. The pre-Struggle turn is
    asserted too, so the agreement is a property of the payload rather than an artefact of
    both views being empty.
    """

    def _payloads(self):
        previous = None
        with LocalShowdownEnv(self.config) as env:
            env.reset_with_start_override(seed=11, start_override=_PP_STALL)
            for _ in range(16):
                payload = _public_materialization_payload(
                    env.public_materialization_state("p1")
                )
                rows = self._struggle_rows(env)
                if rows is not None:
                    return previous, payload, rows
                previous = payload
                env.step({"p1": 0, "p2": 0})
        raise AssertionError("no Struggle request was produced in 16 turns")

    def test_the_two_views_agree_at_a_live_struggle_request(self) -> None:
        previous, struggle, raw_rows = self._payloads()

        self.assertEqual(
            json.dumps(raw_rows, sort_keys=True),
            json.dumps([STRUGGLE_ROW], sort_keys=True),
            "the captured branch changed shape",
        )

        # The turn BEFORE: both views name the same single usable move. This holds pre-fix
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

        # A usability correction, not an erasure: the PP snapshot survives. It is still one
        # use too generous -- the request carries no PP and the true value is 0 -- and that
        # residual is stated rather than fixed.
        self.assertEqual([(m["id"], m["pp"]) for m in active["moves"]], [("sunnyday", 1)])

        # The single predicate every consumer applies to these rows now agrees across both
        # views instead of disagreeing.
        self.assertIsNone(_sole_enabled_move_id(active["moves"]))
        self.assertIsNone(_sole_enabled_move_id(struggle["selfActiveMoves"]))


class BenchLeakTests(_LiveBase):
    """The marking must not ride the mon onto the bench. Pins the rejected fold placement.

    Taunt clears on switch-out in Showdown, and poke-engine's `re_enable_disabled_moves`
    runs on the OUTGOING active only (`generate_instructions.rs`), so a benched row that
    still says `disabled` hands the search a mon with full PP and no legal move.
    """

    def test_switching_out_of_a_struggle_boundary_leaves_the_bench_usable(self) -> None:
        with LocalShowdownEnv(self.config) as env:
            env.reset_with_start_override(seed=5, start_override=_TAUNT_STALL)
            self._advance_to_struggle(env, turns=8)

            at_boundary = _self_rows(
                _public_materialization_payload(env.public_materialization_state("p1"))
            )
            self.assertEqual(
                at_boundary["Bulbasaur"],
                (True, [("sunnyday", 8, True), ("growth", 64, True)]),
                "the active row must report nothing usable at the Struggle boundary",
            )

            legal = env.legal_actions("p1")
            switch_index = next(i for i in range(4, len(legal)) if legal[i])
            env.step({"p1": switch_index, "p2": 1})

            after = _self_rows(
                _public_materialization_payload(env.public_materialization_state("p1"))
            )
            self.assertEqual(
                after["Bulbasaur"],
                (False, [("sunnyday", 8, False), ("growth", 64, False)]),
                "Taunt cleared on switch-out; a benched mon with full PP must stay usable",
            )


class TauntStruggleOnlyReachesTheEngineTests(_LiveBase):
    """The OTHER route to a Struggle-only request, on the only shape that isolates it.

    `_TAUNT_STALL` above keeps a live bench, so its request offers
    `['struggle', 'switch:...']` and the engine has a switch to enumerate -- which is
    why that fixture never exercised this. `_TAUNT_SOLO_STALL` removes the bench: a
    LONE all-status Blissey, Taunted, has no usable move and nothing to switch to, so
    `getMoveRequestData` substitutes Struggle and it is the request's ONLY candidate.

    That is the shape the docstring's Taunt paragraph claims is now closed, and this is
    where the claim can fail. Pre-fix `world_battle_spec` raised
    `EngineWorldUnsupported('volatile_unsupported')` here and no world existed at all;
    the `assertRaises`-shaped null world is covered by
    `tests/test_engine_world_taunt.py::test_a_taunted_self_side_builds_instead_of_refusing`,
    so this one asserts the positive end to end.

    What it does NOT assert, deliberately: which action search picks. That needs the
    native wheel and lives in `test_engine_world_taunt.py`
    (`test_a_lone_taunted_all_status_side_leaves_the_engine_no_move`).
    """

    def test_a_lone_taunted_all_status_side_builds_a_world_with_no_option(self) -> None:
        from pokezero.engine_world import world_battle_spec

        root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
        dex = load_showdown_dex(root)
        with LocalShowdownEnv(self.config) as env:
            env.reset_with_start_override(seed=11, start_override=_TAUNT_SOLO_STALL)
            rows = self._advance_to_struggle(env, turns=6)
            legal = env.legal_actions("p1")
            state = env.public_materialization_state("p1")
            payload = _public_materialization_payload(state)
            world = world_battle_spec(state, _TAUNT_SOLO_STALL, dex=dex)

        # 1. The request really is Struggle and NOTHING else -- no move, no switch.
        #    Without this the rest is a test of some other boundary.
        self.assertEqual([row["id"] for row in rows], ["struggle"])
        self.assertEqual(
            sum(1 for flag in legal if flag), 1, "a lone taunted mon has exactly one action"
        )

        # 2. Taunt is what put it there, and it is on the payload as a public volatile.
        self.assertIn("taunt", payload["sides"]["p1"]["volatiles"])

        # 3. The world BUILDS (pre-fix: volatile_unsupported), carries the volatile, and
        #    carries the counter at one tick elapsed.
        side = world.spec.side_one
        self.assertIn("taunt", side.volatile_statuses)
        self.assertEqual(side.volatile_status_durations.get("taunt"), 1)

        # 4. ...and it offers the engine nothing: every active slot unselectable, and no
        #    live bench for `add_switches` to fall through to. Those two together are the
        #    precondition for `MoveChoice::None`, which is what #1202's translation needs.
        active = side.pokemon[side.active_index]
        self.assertTrue(
            all(spec.disabled or spec.pp == 0 for spec in active.moves),
            [(spec.id, spec.pp, spec.disabled) for spec in active.moves],
        )
        self.assertEqual(
            [mon.id for i, mon in enumerate(side.pokemon) if i != side.active_index and mon.hp > 0],
            [],
            "the fixture must leave no live bench, or the engine enumerates a switch",
        )


class WhatReachesTheEngineTests(_LiveBase):
    """End to end on the population that actually builds a world: PP exhaustion.

    `_move_specs` copies `(pp, disabled)` verbatim and `Pokemon::add_available_moves`
    (poke-engine 0.0.47 `genx/state.rs`) requires `!disabled && pp > 0`, so an all-disabled
    active contributes no move option and `get_all_options` falls through to
    `add_switches`. The Python binding does not export the option vector
    (`poke_engine_legal_actions` records this as a known gap), so the assertion is on the
    MoveSpecs the engine reads.

    NOTE the residual this test makes concrete: `sunnyday` reaches the engine at pp 1,
    because the last pp-bearing request said 1 and the Struggle request carries no PP. Only
    `disabled` makes it unselectable.
    """

    def test_the_constructed_world_offers_no_move_and_keeps_the_bench_usable(self) -> None:
        from pokezero.engine_world import world_battle_spec

        root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
        dex = load_showdown_dex(root)
        with LocalShowdownEnv(self.config) as env:
            env.reset_with_start_override(seed=11, start_override=_PP_STALL)
            self._advance_to_struggle(env, turns=16)
            world = world_battle_spec(
                env.public_materialization_state("p1"), _PP_STALL, dex=dex
            )

        side = world.spec.side_one
        active = side.pokemon[side.active_index]
        self.assertEqual(
            [(spec.id, spec.pp, spec.disabled) for spec in active.moves],
            [("sunnyday", 1, True), ("none", 0, True), ("none", 0, True), ("none", 0, True)],
            "the world must offer the engine no selectable move at a Struggle boundary",
        )
        benched = [p for i, p in enumerate(side.pokemon) if i != side.active_index]
        self.assertTrue(
            any(
                not spec.disabled
                for mon in benched
                for spec in mon.moves
                if spec.id != "none"
            ),
            "a benched mon must keep a usable move",
        )


if __name__ == "__main__":
    unittest.main()
