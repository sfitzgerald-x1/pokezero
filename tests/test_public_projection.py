"""Pins for direction 2 of the oracle (fallback-burndown plan 4, sequencing 4).

Two things have to be true of this comparator and neither is implied by the
other:

1. **It can fire.** Every axis in :data:`AXES` is reachable from a state a real
   game can present, and this file constructs that state for each one. An axis
   nothing can trip is a row in a report that always reads zero.
2. **It can stay silent.** The same fixture, unperturbed, produces no mismatch
   at all. Direction 1 shipped a cross-stage probe that fired on 3,231 of 3,231
   decisions because it asked a question with only one answer; the mirror of
   that failure is a comparator that never fires, and both look like a working
   instrument from the outside.

The fixtures are REAL ``poke_engine`` states built through the production
adapter, not doubles. The observed side is a hand-built protocol log and a
hand-built request, which is the only part that can be synthesised without
losing the property under test.
"""

from __future__ import annotations

import types
import unittest

from pokezero.poke_engine_adapter import (
    BattleSpec,
    MoveSpec,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
)
from pokezero.public_projection import (
    AXES,
    ProjectionMismatch,
    WorldObserver,
    aggregate_projection_records,
    render_projection_mismatch,
    render_self_consistency_mismatches,
    state_projection_mismatches,
)

SLOT_SIDES = {"p1": "side_one", "p2": "side_two"}


def _mon(species: str, **overrides):
    base = dict(
        id=species,
        level=100,
        types=("normal",),
        hp=200,
        maxhp=200,
        attack=100,
        defense=100,
        special_attack=100,
        special_defense=100,
        speed=100,
        ability="static",
        item="leftovers",
        moves=(MoveSpec(id="tackle", pp=32), MoveSpec(id="splash", pp=40)),
    )
    base.update(overrides)
    return PokemonSpec(**base)


def _spec(**overrides) -> BattleSpec:
    side_one = SideSpec(pokemon=(_mon("pikachu"), _mon("dugtrio")), active_index=0)
    side_two = SideSpec(pokemon=(_mon("squirtle"), _mon("dodrio")), active_index=0)
    base = dict(side_one=side_one, side_two=side_two, weather="none")
    base.update(overrides)
    return BattleSpec(**base)


class _World:
    """Stand-in for `EngineWorld`: the comparator reads only these two fields."""

    def __init__(self, spec: BattleSpec) -> None:
        self.spec = spec
        self.slot_sides = SLOT_SIDES
        self.party_species = {"p1": ("pikachu", "dugtrio"), "p2": ("squirtle", "dodrio")}


#: The protocol log that describes `_spec()` exactly. Both actives at full HP,
#: no status, no weather, no side conditions.
BASE_LINES = (
    "|start",
    "|switch|p1a: Pikachu|Pikachu, L100, M|200/200",
    "|switch|p2a: Squirtle|Squirtle, L100, M|200/200",
    "|turn|1",
)


def _request(**overrides):
    base = {
        "active": [
            {
                "moves": [
                    {"id": "tackle", "move": "Tackle", "pp": 32, "maxpp": 32, "disabled": False},
                    {"id": "splash", "move": "Splash", "pp": 40, "maxpp": 40, "disabled": False},
                ]
            }
        ],
        "side": {
            "id": "p1",
            "pokemon": [
                {
                    "ident": "p1: Pikachu",
                    "details": "Pikachu, L100, M",
                    "condition": "200/200",
                    "active": True,
                    "item": "leftovers",
                    "ability": "static",
                    "moves": ["tackle", "splash"],
                },
                {
                    "ident": "p1: Dugtrio",
                    "details": "Dugtrio, L100, M",
                    "condition": "200/200",
                    "active": False,
                    "item": "leftovers",
                    "ability": "static",
                    "moves": ["tackle", "splash"],
                },
            ],
        },
    }
    base.update(overrides)
    return base


def _revealed(species: str, **overrides):
    base = dict(species=species, moves=(), item=None, ability=None)
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _context(
    *,
    lines=BASE_LINES,
    request=None,
    boosts=None,
    toxic_stage=None,
    revealed=(),
    slot="p1",
):
    replay = types.SimpleNamespace(
        public_events=tuple(types.SimpleNamespace(raw_line=line) for line in lines),
        boosts=boosts or {},
        toxic_stage=toxic_stage or {},
        public_revealed={"p2": tuple(revealed)} if slot == "p1" else {"p1": tuple(revealed)},
        turn_number=1,
    )
    state = types.SimpleNamespace(
        replay=replay, self_request=_request() if request is None else request
    )
    return types.SimpleNamespace(
        player_id=slot,
        battle_id="unit",
        decision_round_index=0,
        public_materialization_state=state,
        observation=types.SimpleNamespace(metadata={}),
    )


def _axes(mismatches):
    return sorted({m.axis for m in mismatches})


class SilenceTests(unittest.TestCase):
    """The comparator must be able to say nothing."""

    def test_matching_world_produces_no_mismatch(self):
        state = build_poke_engine_state(_spec())
        found = state_projection_mismatches(_context(), _World(_spec()), state)
        self.assertEqual([], [m.to_dict() for m in found])

    def test_silence_is_not_an_artifact_of_a_missing_observed_side(self):
        """A context with no materialization state returns [] -- and that is the
        one silence this oracle is allowed, so it must be distinguishable."""

        context = types.SimpleNamespace(player_id="p1", public_materialization_state=None)
        state = build_poke_engine_state(_spec())
        self.assertEqual([], state_projection_mismatches(context, _World(_spec()), state))

    def test_an_empty_world_is_not_silently_accepted(self):
        """The degenerate direction of vacuity: a world with nothing in it must
        NOT read as matching. If the comparator only ever compares what both
        sides happen to have, a constructor that produced an empty party would
        pass."""

        empty = _spec(
            side_one=SideSpec(pokemon=(_mon("ditto"),), active_index=0),
        )
        found = state_projection_mismatches(
            _context(), _World(empty), build_poke_engine_state(empty)
        )
        self.assertIn("self_party_species", _axes(found))


class AxisFiresTests(unittest.TestCase):
    """One test per axis. Each perturbs ONE fact and asserts the axis that owns
    it fires -- and, where the perturbation is genuinely single-axis, that no
    other axis fires with it."""

    def _fire(self, *, world_spec=None, **context_kwargs):
        world_spec = _spec() if world_spec is None else world_spec
        return state_projection_mismatches(
            _context(**context_kwargs),
            _World(world_spec),
            build_poke_engine_state(world_spec),
        )

    def test_active_hp(self):
        lines = BASE_LINES + ("|-damage|p2a: Squirtle|150/200",)
        found = self._fire(lines=lines)
        self.assertEqual(["active_hp"], _axes(found))

    def test_active_status(self):
        lines = BASE_LINES + ("|-status|p2a: Squirtle|brn",)
        found = self._fire(lines=lines)
        self.assertEqual(["active_status"], _axes(found))

    def test_weather(self):
        lines = BASE_LINES + ("|-weather|Sandstorm",)
        found = self._fire(lines=lines)
        self.assertEqual(["weather"], _axes(found))

    def test_side_conditions(self):
        lines = BASE_LINES + ("|-sidestart|p2: Squirtle|Spikes",)
        found = self._fire(lines=lines)
        self.assertEqual(["side_conditions"], _axes(found))

    def test_self_move_set(self):
        request = _request()
        request["active"][0]["moves"][0]["id"] = "thunderbolt"
        found = self._fire(request=request)
        self.assertEqual(["self_move_set"], _axes(found))

    def test_self_move_pp(self):
        """#1210's axis. The request publishes the live PP every round."""

        request = _request()
        request["active"][0]["moves"][0]["pp"] = 5
        found = self._fire(request=request)
        self.assertEqual(["self_move_pp"], _axes(found))
        self.assertEqual("self_move_pp:tackle", found[0].predicate)

    def test_self_move_disabled(self):
        """#1212's axis: under Encore the request disables every other move, so a
        lock resolved onto the WRONG move disagrees with it."""

        request = _request()
        request["active"][0]["moves"][0]["disabled"] = True
        found = self._fire(request=request)
        self.assertEqual(["self_move_disabled"], _axes(found))

    def test_self_party_species(self):
        request = _request()
        request["side"]["pokemon"][1]["details"] = "Snorlax, L100, M"
        found = self._fire(request=request)
        self.assertEqual(["self_party_species"], _axes(found))

    def test_self_party_hp(self):
        request = _request()
        request["side"]["pokemon"][1]["condition"] = "90/200"
        found = self._fire(request=request)
        self.assertEqual(["self_party_hp"], _axes(found))

    def test_self_item(self):
        request = _request()
        request["side"]["pokemon"][1]["item"] = "choiceband"
        found = self._fire(request=request)
        self.assertEqual(["self_item"], _axes(found))

    def test_self_ability(self):
        request = _request()
        request["side"]["pokemon"][1]["ability"] = "levitate"
        found = self._fire(request=request)
        self.assertEqual(["self_ability"], _axes(found))

    def test_opponent_revealed_species(self):
        found = self._fire(revealed=(_revealed("Snorlax"),))
        self.assertEqual(["opponent_revealed_species"], _axes(found))

    def test_opponent_revealed_moves(self):
        found = self._fire(revealed=(_revealed("Squirtle", moves=("surf",)),))
        self.assertEqual(["opponent_revealed_moves"], _axes(found))

    def test_opponent_revealed_item(self):
        found = self._fire(revealed=(_revealed("Squirtle", item="Choice Band"),))
        self.assertEqual(["opponent_revealed_item"], _axes(found))

    def test_opponent_revealed_ability(self):
        found = self._fire(revealed=(_revealed("Squirtle", ability="Torrent"),))
        self.assertEqual(["opponent_revealed_ability"], _axes(found))

    def test_boosts(self):
        found = self._fire(boosts={"p2": {"atk": 2}})
        self.assertEqual(["boosts"], _axes(found))
        self.assertEqual("boosts:atk", found[0].predicate)

    def test_toxic_count(self):
        """#1209's axis, and the one a real six-game shard did NOT reach.

        The census forcing `--force state-toxic` fired zero on six games because
        no active was Toxic-statused in them. That is exactly the situation where
        an axis is presumed working and is not, so it is pinned here instead of
        left to a game to reach.
        """

        spec = _spec(
            side_two=SideSpec(
                pokemon=(_mon("squirtle", status="toxic"), _mon("dodrio")),
                active_index=0,
                side_conditions={"toxic_count": 3},
            )
        )
        lines = BASE_LINES + ("|-status|p2a: Squirtle|tox",)
        found = self._fire(world_spec=spec, lines=lines, toxic_stage={"p2": 5})
        self.assertEqual(["toxic_count"], _axes(found))
        self.assertIn("observed stage 5 != world 3", found[0].detail)

    def test_toxic_count_is_silent_when_the_stage_agrees(self):
        spec = _spec(
            side_two=SideSpec(
                pokemon=(_mon("squirtle", status="toxic"), _mon("dodrio")),
                active_index=0,
                side_conditions={"toxic_count": 5},
            )
        )
        lines = BASE_LINES + ("|-status|p2a: Squirtle|tox",)
        found = self._fire(world_spec=spec, lines=lines, toxic_stage={"p2": 5})
        self.assertEqual([], _axes(found))

    def test_toxic_count_is_silent_at_the_saturation_sentinel(self):
        """`replay.toxic_stage == 16` is the parser saying "already capped and I
        cannot tell you the exact value", not a stage. Comparing against it would
        manufacture a mismatch out of the parser's own uncertainty."""

        spec = _spec(
            side_two=SideSpec(
                pokemon=(_mon("squirtle", status="toxic"), _mon("dodrio")),
                active_index=0,
                side_conditions={"toxic_count": 15},
            )
        )
        lines = BASE_LINES + ("|-status|p2a: Squirtle|tox",)
        found = self._fire(world_spec=spec, lines=lines, toxic_stage={"p2": 16})
        self.assertEqual([], _axes(found))


class KnownProducerExclusionTests(unittest.TestCase):
    """The two exclusions this comparator carries, each pinned in BOTH
    directions -- silent on the producer it excludes, still live otherwise.

    Both were found by running the comparator, not by reasoning about it, and
    both are the report 4 section 2.1 shape: the mechanism was wrong on contact
    until someone dumped the two sides of the comparison.
    """

    def _run(self, spec, request):
        return state_projection_mismatches(
            _context(request=request), _World(spec), build_poke_engine_state(spec)
        )

    def test_trace_is_excluded_because_the_world_is_the_correct_side(self):
        spec = _spec(
            side_one=SideSpec(
                pokemon=(_mon("pikachu", ability="rockhead"), _mon("dugtrio")),
                active_index=0,
            )
        )
        request = _request()
        request["side"]["pokemon"][0]["ability"] = "trace"
        found = self._run(spec, request)
        self.assertEqual([], _axes(found))

    def test_a_non_trace_ability_disagreement_still_fires(self):
        spec = _spec(
            side_one=SideSpec(
                pokemon=(_mon("pikachu", ability="rockhead"), _mon("dugtrio")),
                active_index=0,
            )
        )
        request = _request()
        request["side"]["pokemon"][0]["ability"] = "static"
        found = self._run(spec, request)
        self.assertEqual(["self_ability"], _axes(found))

    def test_a_transformed_active_is_matched_on_its_own_species(self):
        """A Ditto that copied Dugtrio is `p1a: Ditto` on every protocol line and
        `Ditto` in the request, while the world runs the donor. Measured firing on
        8 worlds of the first six-game shard before `pre_transform` was read."""

        copy = _mon("dugtrio", pre_transform=_mon("ditto"))
        spec = _spec(side_one=SideSpec(pokemon=(copy, _mon("pikachu")), active_index=0))
        request = _request()
        request["side"]["pokemon"][0]["details"] = "Ditto, L100"
        request["side"]["pokemon"][1]["details"] = "Pikachu, L100, M"
        found = self._run(spec, request)
        self.assertEqual([], _axes(found))

    def test_a_transformed_active_runs_the_donors_ability(self):
        """Measured: `ditto: request ability limber != world arenatrap`, 8 worlds
        on a six-game shard. Transform copies the ability; the request keeps
        reporting the transformer's own."""

        copy = _mon("dugtrio", ability="arenatrap", pre_transform=_mon("ditto"))
        spec = _spec(side_one=SideSpec(pokemon=(copy, _mon("pikachu")), active_index=0))
        request = _request()
        request["side"]["pokemon"][0]["details"] = "Ditto, L100"
        request["side"]["pokemon"][0]["ability"] = "limber"
        request["side"]["pokemon"][1]["details"] = "Pikachu, L100, M"
        self.assertEqual([], _axes(self._run(spec, request)))

    def test_an_untransformed_ability_disagreement_still_fires(self):
        """Safer-direction check on the exclusion above."""

        spec = _spec(
            side_one=SideSpec(
                pokemon=(_mon("dugtrio", ability="arenatrap"), _mon("pikachu")),
                active_index=0,
            )
        )
        request = _request()
        request["side"]["pokemon"][0]["details"] = "Dugtrio, L100, M"
        request["side"]["pokemon"][0]["ability"] = "limber"
        request["side"]["pokemon"][1]["details"] = "Pikachu, L100, M"
        self.assertEqual(["self_ability"], _axes(self._run(spec, request)))

    def test_a_transformed_opponent_is_not_held_to_the_transformers_moveset(self):
        """The opponent-side twin of the ability exclusion. A Ditto that copied
        Dugtrio is revealed as `Ditto` with `transform` in its move list, while
        the world's copy carries the DONOR's four moves and none of them is
        `transform`. Without this the axis fires on every opponent Transform."""

        copy = _mon("dugtrio", pre_transform=_mon("ditto"))
        spec = _spec(side_two=SideSpec(pokemon=(copy, _mon("dodrio")), active_index=0))
        context = _context(revealed=(_revealed("Ditto", moves=("transform",)),))
        found = state_projection_mismatches(
            context, _World(spec), build_poke_engine_state(spec)
        )
        self.assertEqual([], _axes(found))

    def test_an_untransformed_opponent_move_reveal_still_fires(self):
        """Safer-direction check: the exclusion above must be scoped to a
        transformed copy and to nothing else."""

        spec = _spec(side_two=SideSpec(pokemon=(_mon("dugtrio"), _mon("dodrio")), active_index=0))
        context = _context(revealed=(_revealed("Dugtrio", moves=("earthquake",)),))
        found = state_projection_mismatches(
            context, _World(spec), build_poke_engine_state(spec)
        )
        self.assertEqual(["opponent_revealed_moves"], _axes(found))

    def test_the_transform_exclusion_does_not_blind_the_species_axis(self):
        """The safer-direction check on the exclusion above: an UNtransformed
        species disagreement in the same shape must still fire."""

        spec = _spec(side_one=SideSpec(pokemon=(_mon("dugtrio"), _mon("pikachu")), active_index=0))
        request = _request()
        request["side"]["pokemon"][0]["details"] = "Ditto, L100"
        request["side"]["pokemon"][1]["details"] = "Pikachu, L100, M"
        found = self._run(spec, request)
        self.assertEqual(["self_party_species"], _axes(found))


class StickyFaintRegressionTests(unittest.TestCase):
    """`TurnFeatures.fainted` accumulates and never clears.

    Reading it over a whole-log fold marked a side permanently from its first
    faint; on the first smoke game that suppressed the status axis for the rest
    of the battle and (before the axis was removed) reported 272 of 312 worlds as
    mismatched. The pin is that a post-faint replacement is compared normally.
    """

    def test_status_is_still_compared_after_an_earlier_faint(self):
        lines = BASE_LINES + (
            "|faint|p2a: Squirtle",
            "|switch|p2a: Dodrio|Dodrio, L100, M|200/200",
            "|-status|p2a: Dodrio|brn",
        )
        spec = _spec(
            side_two=SideSpec(pokemon=(_mon("squirtle"), _mon("dodrio")), active_index=1)
        )
        found = state_projection_mismatches(
            _context(lines=lines), _World(spec), build_poke_engine_state(spec)
        )
        self.assertEqual(["active_status"], _axes(found))


class NarrowingsThatMustNotBeRemoved(unittest.TestCase):
    """Each of these makes the comparator SAFER when deleted -- it fires more.

    Report 4 section 4.4: a battery that only reverts a fix proves the fix does
    something, not the right thing. The mutants that matter are the ones that
    move toward MORE conservatism, because a survivor there means the suite is
    silent exactly where a too-permissive bound would live. These tests exist to
    kill that class.
    """

    def test_a_fainted_active_is_not_status_compared(self):
        """Showdown's `0 fnt` condition carries no status while the engine keeps
        it on the fainted mon. Comparing anyway fires on every faint."""

        lines = BASE_LINES + ("|-damage|p2a: Squirtle|0 fnt", "|faint|p2a: Squirtle")
        spec = _spec(
            side_two=SideSpec(
                pokemon=(_mon("squirtle", hp=0, status="burn"), _mon("dodrio")),
                active_index=0,
            )
        )
        found = state_projection_mismatches(
            _context(lines=lines), _World(spec), build_poke_engine_state(spec)
        )
        self.assertEqual([], _axes(found))

    def test_a_struggle_only_request_is_not_compared(self):
        """A Struggle-only request describes the ENGINE's substitution, not the
        active's moveset; the world legitimately still carries the real moves."""

        request = _request()
        request["active"][0]["moves"] = [
            {"id": "struggle", "move": "Struggle", "pp": 1, "maxpp": 1, "disabled": False}
        ]
        found = state_projection_mismatches(
            _context(request=request), _World(_spec()), build_poke_engine_state(_spec())
        )
        self.assertEqual([], _axes(found))

    def test_hidden_power_is_matched_through_its_typed_engine_id(self):
        """The request says `hiddenpower`; the engine carries `hiddenpowerice60`.
        Comparing the ids literally fires on every Hidden Power carrier."""

        spec = _spec(
            side_one=SideSpec(
                pokemon=(
                    _mon(
                        "pikachu",
                        moves=(MoveSpec(id="hiddenpowerice60", pp=24), MoveSpec(id="splash", pp=40)),
                    ),
                    _mon("dugtrio"),
                ),
                active_index=0,
            )
        )
        request = _request()
        request["active"][0]["moves"][0] = {
            "id": "hiddenpower", "move": "Hidden Power", "pp": 24, "maxpp": 24, "disabled": False
        }
        found = state_projection_mismatches(
            _context(request=request), _World(spec), build_poke_engine_state(spec)
        )
        self.assertEqual([], _axes(found))


class AxisClosureTests(unittest.TestCase):
    def test_every_axis_in_the_closed_set_is_pinned_by_a_test(self):
        """A row that can never be produced is a row that always reads zero.

        Enumerated from the tests in this file rather than asserted, so adding an
        axis without a pin fails here instead of shipping a permanently-empty
        queue row.
        """

        pinned = {
            name[len("test_") :]
            for name in dir(AxisFiresTests)
            if name.startswith("test_")
        }
        state_axes = {axis for axis in AXES if not axis.startswith("render_")}
        missing = {
            axis
            for axis in state_axes
            if axis not in pinned and not any(p.startswith(axis) for p in pinned)
        }
        self.assertEqual(set(), missing)

    def test_render_axes_are_pinned_separately(self):
        self.assertEqual(
            {
                "render_unmatched_transition",
                "render_no_usable_branch",
                "render_post_state_disagreement",
            },
            {axis for axis in AXES if axis.startswith("render_")},
        )


class _FakeCrate:
    """A `branch_events` double. The renderer under test in a census is the real
    crate; here the point is the COMPARATOR, so the branch payloads are supplied
    directly and the crate is not in the loop."""

    def __init__(self, branches):
        self._branches = branches
        self.calls = []

    def branch_events(self, state, s1, s2, ctx, branch_on_damage, include_post):
        import json

        self.calls.append((s1, s2, json.loads(ctx)))
        return json.dumps({"branches": self._branches})


class RenderProjectionTests(unittest.TestCase):
    """The renderer's own projection -- the only arm that can see #1211.

    #1211's over-broad direction renders `|-activate|...|Protect` where the
    engine emitted a real heal. In protocol terms that is a heal line that is
    simply absent from the render, and the fixtures below are exactly that shape.
    """

    PRE = types.SimpleNamespace(
        p1_hp=200,
        p2_hp=100,
        p1_status="NONE",
        p2_status="NONE",
        fainted=frozenset(),
        weather="NONE",
        side_conditions={},
        presence=lambda: {},
    )

    def _run(self, branches, observed_lines):
        return render_projection_mismatch(
            state_string="<state>",
            slot_sides=SLOT_SIDES,
            party_display={"p1": ["Pikachu"], "p2": ["Squirtle"]},
            turn=3,
            choices={"p1": "tackle", "p2": "surf"},
            observed_lines=observed_lines,
            pre_features=self.PRE,
            module=_FakeCrate(branches),
        )

    def test_a_matching_branch_is_silent(self):
        observed = ["|move|p1a: Pikachu|Tackle|p2a: Squirtle", "|-damage|p2a: Squirtle|60/200", "|turn|4"]
        branches = [{"events": list(observed), "lossy": [], "attribution_unsafe": False}]
        found, diagnostics = self._run(branches, observed)
        self.assertEqual([], found)
        self.assertTrue(diagnostics["matched"])

    def test_a_protect_marker_over_a_real_heal_is_caught(self):
        """The #1211 over-broad shape, in the vocabulary the comparator sees."""

        observed = [
            "|move|p1a: Pikachu|Water Gun|p2a: Squirtle",
            "|-heal|p2a: Squirtle|150/200|[from] ability: Water Absorb",
            "|turn|4",
        ]
        rendered = [
            "|move|p1a: Pikachu|Water Gun|p2a: Squirtle",
            "|-activate|p2a: Squirtle|move: Protect",
            "|turn|4",
        ]
        branches = [{"events": rendered, "lossy": [], "attribution_unsafe": False}]
        found, _ = self._run(branches, observed)
        self.assertEqual(["render_unmatched_transition"], _axes(found))

    def test_an_attribution_unsafe_only_branch_set_is_its_own_verdict(self):
        """Not `unmatched`. The renderer said it could not describe the branch,
        which is a different fact from "it described it wrongly", and folding the
        two together would let a census read a clean render rate off a block where
        nothing was comparable."""

        observed = ["|-damage|p2a: Squirtle|60/200", "|turn|4"]
        branches = [{"events": [], "lossy": ["x"], "attribution_unsafe": True}]
        found, _ = self._run(branches, observed)
        self.assertEqual(["render_no_usable_branch"], _axes(found))

    def test_one_matching_branch_among_many_is_enough(self):
        """Chance branching is real: the engine enumerates several outcomes and
        Showdown sampled one. Requiring ALL to match would fire on every
        multi-branch turn."""

        observed = ["|-damage|p2a: Squirtle|60/200", "|turn|4"]
        branches = [
            {"events": ["|-damage|p2a: Squirtle|10/200", "|turn|4"], "lossy": [], "attribution_unsafe": False},
            {"events": list(observed), "lossy": [], "attribution_unsafe": False},
        ]
        found, _ = self._run(branches, observed)
        self.assertEqual([], found)

    def test_a_telemetry_only_lossy_branch_stays_usable(self):
        """ALLOWLIST, not an exclusion. `sleeptalk_called_unidentified` is the
        marker the #1211 render path itself pushes; treating it as unusable would
        make the comparator blind to exactly the branch it exists to check."""

        observed = ["|-damage|p2a: Squirtle|60/200", "|turn|4"]
        branches = [
            {
                "events": ["|-damage|p2a: Squirtle|60/200", "|turn|4"],
                "lossy": ["sleeptalk_called_unidentified"],
                "attribution_unsafe": False,
            }
        ]
        found, diagnostics = self._run(branches, observed)
        self.assertEqual([], found)
        self.assertEqual(1, diagnostics["usable_branches"])

    def test_a_non_allowlisted_lossy_marker_makes_a_branch_unusable(self):
        """ALLOWLIST, in the direction that matters. A branch carrying a lossy
        marker nobody has cleared for matching is not comparable, and admitting
        it would match the census against renders the crate itself says are
        incomplete."""

        observed = ["|-damage|p2a: Squirtle|60/200", "|turn|4"]
        branches = [
            {
                "events": ["|-damage|p2a: Squirtle|60/200", "|turn|4"],
                "lossy": ["some_future_marker"],
                "attribution_unsafe": False,
            }
        ]
        found, diagnostics = self._run(branches, observed)
        self.assertEqual(0, diagnostics["usable_branches"])
        self.assertEqual(["render_no_usable_branch"], _axes(found))

    def test_the_band_does_not_swallow_a_wrong_mechanic(self):
        """The HP band is scoped to a roll, not to a mechanic. A heal rendered as
        a no-op moves HP by 25% of max, which is far outside it."""

        observed = ["|-heal|p2a: Squirtle|150/200", "|turn|4"]
        rendered = ["|-damage|p2a: Squirtle|98/200", "|turn|4"]
        branches = [{"events": rendered, "lossy": [], "attribution_unsafe": False}]
        found, _ = self._run(branches, observed)
        self.assertEqual(["render_unmatched_transition"], _axes(found))


class RenderSelfConsistencyTests(unittest.TestCase):
    """Every searched branch's render vs that branch's OWN post-state.

    The transition comparator only ever sees the one branch the game took. This
    one sees all of them, needs no log, and is the arm that catches #1211's
    over-broad direction: a Protect marker rendered over a real heal leaves the
    rendered HP where it started while the branch's post-state has moved.
    """

    PRE = types.SimpleNamespace(p1_hp=200, p2_hp=192, p1_status="NONE", p2_status="NONE")

    def _run(self, branches):
        return render_self_consistency_mismatches(
            branches, slot_sides=SLOT_SIDES, pre_features=self.PRE
        )

    def _branch(self, events, post_p2_hp, **overrides):
        base = {
            "events": events,
            "lossy": [],
            "attribution_unsafe": False,
            "post": {
                "p1": {"active_hp": 200},
                "p2": {"active_hp": post_p2_hp},
            },
        }
        base.update(overrides)
        return base

    def test_a_render_that_matches_its_own_post_state_is_silent(self):
        events = ["|-heal|p2a: Mantine|252/252", "|turn|2"]
        self.assertEqual([], self._run([self._branch(events, 252)]))

    def test_a_protect_marker_over_a_real_heal_is_caught(self):
        """#1211's over-broad shape, exactly: the marker renders, the heal does
        not, and the branch's own post-state says the defender healed."""

        events = ["|-activate|p2a: Mantine|Protect", "|turn|2"]
        found = self._run([self._branch(events, 252)])
        self.assertEqual(["render_post_state_disagreement"], _axes(found))
        self.assertIn("render says hp 192", found[0].detail)
        self.assertIn("post-state says 252", found[0].detail)

    def test_a_refused_branch_is_not_compared(self):
        """An attribution-unsafe branch never reaches the fold/encoder path, so
        holding its render to account would charge the abort channel twice --
        once to direction 1 as a refusal and again here as a wrong render."""

        events = ["|-activate|p2a: Mantine|Protect", "|turn|2"]
        branch = self._branch(events, 252, attribution_unsafe=True)
        self.assertEqual([], self._run([branch]))

    def test_a_zero_amount_marker_is_INVISIBLE_and_that_is_disclosed(self):
        """The documented blind spot, pinned so it cannot be quietly claimed.

        A full-HP absorber's no-op and Protect's no-op have the SAME post-state,
        which is precisely why the two are ambiguous. No state comparator can
        separate them; only the per-source line decomposition in
        `scripts/engine_transition_differential.py` can.
        """

        events = ["|-activate|p2a: Mantine|Protect", "|turn|2"]
        self.assertEqual([], self._run([self._branch(events, 192)]))

    def test_a_branch_with_no_post_state_is_skipped_rather_than_guessed(self):
        branch = {"events": [], "lossy": [], "attribution_unsafe": False}
        self.assertEqual([], self._run([branch]))


class AggregationTests(unittest.TestCase):
    def test_worlds_and_decisions_are_separate_units(self):
        """`world_failure_reasons` counts WORLDS and `fallback_reasons` counts
        DECISIONS; the same rule binds here. Two worlds of one decision carrying
        the same predicate is 2 worlds and 1 decision, never 2 of anything else."""

        record = {
            "battle_id": "b",
            "seed": 1,
            "seat": "p1",
            "round": 0,
            "turn": 1,
            "arm": "driver",
            "worlds": [
                {"world_index": 0, "mismatches": [{"axis": "boosts", "predicate": "boosts:atk", "slot": "p1", "detail": ""}]},
                {"world_index": 1, "mismatches": [{"axis": "boosts", "predicate": "boosts:atk", "slot": "p1", "detail": ""}]},
                {"world_index": 2, "mismatches": []},
            ],
            "render": None,
            "exemplar": None,
        }
        summary = aggregate_projection_records([record])
        self.assertEqual(3, summary["worlds_projected"])
        self.assertEqual(2, summary["projection_mismatched_worlds"])
        self.assertEqual(1, summary["projection_mismatched_decisions"])
        row = summary["predicates"][0]
        self.assertEqual((2, 1), (row["worlds"], row["decisions"]))

    def test_two_predicates_in_one_decision_is_still_one_decision(self):
        """The DECISIONS column counts decisions, not predicates. Summing
        predicates into it is the co-ranking the reporting rules forbid, and it
        would inflate a decision rate above 100% on a bad enough day."""

        record = {
            "battle_id": "b",
            "arm": "driver",
            "worlds": [
                {
                    "world_index": 0,
                    "mismatches": [
                        {"axis": "boosts", "predicate": "boosts:atk", "slot": "p1", "detail": ""},
                        {"axis": "active_hp", "predicate": "active_hp", "slot": "p2", "detail": ""},
                    ],
                }
            ],
        }
        summary = aggregate_projection_records([record])
        self.assertEqual(1, summary["projection_mismatched_decisions"])
        self.assertEqual(1, summary["decisions_seen"])

    def test_render_boundaries_are_a_third_unit(self):
        rows = [
            {"battle_id": "b", "arm": "driver", "worlds": [], "render": {"axes": []}},
            {
                "battle_id": "b",
                "arm": "driver",
                "worlds": [],
                "render": {"axes": ["render_unmatched_transition"]},
            },
        ]
        summary = aggregate_projection_records(rows)
        self.assertEqual(2, summary["render_boundaries_compared"])
        self.assertEqual(1, summary["render_mismatched_boundaries"])
        # And it is NOT folded into the world/decision numbers.
        self.assertEqual(0, summary["projection_mismatched_worlds"])


class ObserverTests(unittest.TestCase):
    def test_the_observer_records_one_entry_per_world_and_one_record_per_decision(self):
        records = []
        observer = WorldObserver(arm="driver", records=records)
        context = _context()
        state = build_poke_engine_state(_spec())
        for _ in range(3):
            observer(context, _World(_spec()), state)
        self.assertEqual(1, len(records))
        self.assertEqual(3, len(records[0].worlds))
        self.assertEqual(0, len(observer.errors))

    def test_a_raising_forcing_is_recorded_as_an_instrument_error_not_a_clean_zero(self):
        def boom(world, state):
            raise RuntimeError("nope")

        records = []
        observer = WorldObserver(arm="driver", records=records, forcing=boom)
        observer(_context(), _World(_spec()), build_poke_engine_state(_spec()))
        self.assertEqual(1, len(observer.errors))
        self.assertEqual([], records[0].worlds)


class EngineSearchHookTests(unittest.TestCase):
    def test_the_hook_defaults_to_absent_and_is_a_named_method(self):
        """Deleting `_notify_world_observer` must be a killable mutation, so it
        has to be a named attribute rather than an inline `if`."""

        from pokezero.engine_search import EngineMctsPolicy

        self.assertIn(
            "world_observer", EngineMctsPolicy.__init__.__code__.co_varnames
        )
        self.assertTrue(callable(EngineMctsPolicy._notify_world_observer))

    def test_a_raising_observer_does_not_break_the_search(self):
        from pokezero.engine_search import EngineMctsPolicy

        policy = EngineMctsPolicy.__new__(EngineMctsPolicy)
        policy._world_observer = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            policy._notify_world_observer(None, None, None)
        self.assertTrue(any("world_observer raised" in str(w.message) for w in caught))

    def test_the_observer_is_called_once_per_constructed_world(self):
        """The hook must fire from the construction loop, not from a wrapper the
        harness could have written itself -- otherwise the census measures the
        harness's own re-sample."""

        import inspect

        from pokezero.engine_search import EngineMctsPolicy

        source = inspect.getsource(EngineMctsPolicy._search)
        self.assertIn("worlds.append((world, state))", source)
        self.assertIn("self._notify_world_observer(context, world, state)", source)


if __name__ == "__main__":
    unittest.main()
