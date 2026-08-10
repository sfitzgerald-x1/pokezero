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

import inspect
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
    ObservedPublicView,
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
        """#1209's axis, on an observed side the constructor cannot have written.

        THE FIRST VERSION OF THIS TEST PINNED THE WRONG CONVENTION. It compared
        `replay.toxic_stage` against `side_conditions.toxic_count` and asserted
        they must be EQUAL -- but `_materialization_toxic_stage` computes the
        latter FROM the former as `tracked_stage - 1`, so the test asserted the
        opposite of the shipped convention and a mutant that corrected the axis
        would have turned this suite red. The observed side is now the multiplier
        the log shows was actually PAID, recovered from raw damage.

        maxhp 256 -> unit 16. A tick of 48 is multiplier 3, so the engine's
        pre-tick counter must be 3; 1 is a mismatch.
        """

        spec = _spec(
            side_two=SideSpec(
                pokemon=(_mon("squirtle", hp=100, maxhp=256, status="toxic"), _mon("dodrio")),
                active_index=0,
                side_conditions={"toxic_count": 1},
            )
        )
        lines = (
            "|start",
            "|switch|p1a: Pikachu|Pikachu, L100, M|200/200",
            "|switch|p2a: Squirtle|Squirtle, L100, M|148/256 tox",
            "|-damage|p2a: Squirtle|100/256 tox|[from] psn",
            "|turn|2",
        )
        found = self._fire(world_spec=spec, lines=lines)
        self.assertEqual(["toxic_count"], _axes(found))
        self.assertIn("paid multiplier 3", found[0].detail)

    def test_toxic_count_is_silent_when_the_counter_matches_the_paid_multiplier(self):
        spec = _spec(
            side_two=SideSpec(
                pokemon=(_mon("squirtle", hp=100, maxhp=256, status="toxic"), _mon("dodrio")),
                active_index=0,
                side_conditions={"toxic_count": 3},
            )
        )
        lines = (
            "|start",
            "|switch|p1a: Pikachu|Pikachu, L100, M|200/200",
            "|switch|p2a: Squirtle|Squirtle, L100, M|148/256 tox",
            "|-damage|p2a: Squirtle|100/256 tox|[from] psn",
            "|turn|2",
        )
        self.assertEqual([], _axes(self._fire(world_spec=spec, lines=lines)))

    def test_toxic_count_is_silent_before_any_tick_has_been_observed(self):
        """No tick since this active came in means no multiplier was determined,
        and an axis must never fire on an undetermined value."""

        spec = _spec(
            side_two=SideSpec(
                pokemon=(_mon("squirtle", status="toxic"), _mon("dodrio")),
                active_index=0,
                side_conditions={"toxic_count": 7},
            )
        )
        lines = BASE_LINES + ("|-status|p2a: Squirtle|tox",)
        self.assertEqual([], _axes(self._fire(world_spec=spec, lines=lines)))

    def test_a_switch_out_resets_what_the_log_can_prove(self):
        """Gen 3 clears the counter on switch-out, so ticks paid by the mon that
        left say nothing about the one standing there now."""

        spec = _spec(
            side_two=SideSpec(
                pokemon=(_mon("squirtle", hp=100, maxhp=256, status="toxic"), _mon("dodrio", status="toxic")),
                active_index=1,
                side_conditions={"toxic_count": 0},
            )
        )
        lines = (
            "|start",
            "|switch|p1a: Pikachu|Pikachu, L100, M|200/200",
            "|switch|p2a: Squirtle|Squirtle, L100, M|148/256 tox",
            "|-damage|p2a: Squirtle|100/256 tox|[from] psn",
            "|switch|p2a: Dodrio|Dodrio, L100, M|200/200 tox",
            "|turn|2",
        )
        self.assertEqual([], _axes(self._fire(world_spec=spec, lines=lines)))

    def test_a_non_integral_tick_is_not_turned_into_a_stage(self):
        """A percentage-mod grid, or a tick that hit the HP floor, cannot be
        divided cleanly. Silence beats a fabricated multiplier."""

        spec = _spec(
            side_two=SideSpec(
                pokemon=(_mon("squirtle", hp=100, maxhp=256, status="toxic"), _mon("dodrio")),
                active_index=0,
                side_conditions={"toxic_count": 1},
            )
        )
        lines = (
            "|start",
            "|switch|p1a: Pikachu|Pikachu, L100, M|200/200",
            "|switch|p2a: Squirtle|Squirtle, L100, M|141/256 tox",
            "|-damage|p2a: Squirtle|100/256 tox|[from] psn",
            "|turn|2",
        )
        self.assertEqual([], _axes(self._fire(world_spec=spec, lines=lines)))

    def test_the_observed_side_does_not_read_the_parsers_toxic_tracker(self):
        """The structural pin behind all of the above.

        `replay.toxic_stage` is what the constructor's own input is computed
        from, so reading it here made the axis compare `x` to `f(x)`. The observed
        side must be derivable with that field absent entirely.
        """

        from pokezero.public_projection import observed_toxic_multiplier

        lines = [
            "|switch|p2a: Squirtle|Squirtle, L100, M|148/256 tox",
            "|-damage|p2a: Squirtle|100/256 tox|[from] psn",
        ]
        self.assertEqual(3, observed_toxic_multiplier(lines)["p2"])
        # The CODE, with the docstring stripped -- the docstring necessarily
        # names the field it exists to explain the absence of.
        body = inspect.getsource(observed_toxic_multiplier)
        body = body.replace(observed_toxic_multiplier.__doc__ or "", "")
        self.assertNotIn("toxic_stage", body)
        self.assertNotIn("toxic_multiplier", ObservedPublicView.__doc__ or "x")


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

    def test_a_lock_restricted_request_is_not_a_wrong_moveset(self):
        """A charge lock / recharge / Choice lock restricts the request to one
        move while the world legitimately keeps all four. 304 of 710 firings on
        the first census were this."""

        request = _request()
        request["active"][0]["moves"] = [
            {"id": "splash", "move": "Splash", "pp": 40, "maxpp": 40, "disabled": False}
        ]
        self.assertEqual([], _axes(self._run(_spec(), request)))

    def test_a_transformed_active_keeps_slots_the_request_does_not_name(self):
        """`_copied_move_spec` leaves an unnamed donor slot usable ON PURPOSE, so
        the sampler's moveset fiction stays COUNTABLE in `unmapped_choices`. The
        published exemplar of this PR's first revision was exactly this shape --
        a Ditto transformed into Cradily -- and calling it a candidate defect was
        wrong."""

        copy = _mon(
            "cradily",
            pre_transform=_mon("ditto"),
            moves=(MoveSpec(id="earthquake", pp=5), MoveSpec(id="toxic", pp=5),
                   MoveSpec(id="recover", pp=5), MoveSpec(id="rockslide", pp=5)),
        )
        spec = _spec(side_one=SideSpec(pokemon=(copy, _mon("pikachu")), active_index=0))
        request = _request()
        request["side"]["pokemon"][0]["details"] = "Ditto, L100"
        request["side"]["pokemon"][1]["details"] = "Pikachu, L100, M"
        request["active"][0]["moves"] = [
            {"id": "earthquake", "pp": 5, "disabled": False},
            {"id": "recover", "pp": 5, "disabled": False},
            {"id": "rockslide", "pp": 5, "disabled": False},
        ]
        self.assertEqual([], _axes(self._run(spec, request)))

    def test_a_recharge_pseudo_move_is_not_a_missing_move(self):
        """`recharge` is not a move any engine moveset carries -- a recharging
        seat holds MUSTRECHARGE and the engine offers only "No Move". Measured
        firing 32 times on a 40-game slice before it was excluded."""

        request = _request()
        request["active"][0]["moves"] = [
            {"id": "recharge", "move": "Recharge", "pp": 0, "disabled": False}
        ]
        self.assertEqual([], _axes(self._run(_spec(), request)))

    def test_a_request_move_absent_from_the_world_still_fires(self):
        """The safer-direction pin on both narrowings above. Every move the
        request says this seat may pick MUST exist in the searched world."""

        request = _request()
        request["active"][0]["moves"] = [
            {"id": "thunderbolt", "pp": 24, "disabled": False}
        ]
        found = self._run(_spec(), request)
        self.assertEqual(["self_move_set"], _axes(found))
        self.assertEqual(
            "self_move_set:request_move_absent_from_world", found[0].predicate
        )

    def test_a_transform_copys_missing_move_is_a_DIFFERENT_predicate(self):
        """Different producer, different owner, different queue row. A
        non-transformed self active takes its moveset from the request rows, so a
        missing move there would be a self-team construction bug; on a Transform
        copy it is the belief sampler having drawn a donor variant the real copy
        is not."""

        copy = _mon(
            "cradily",
            pre_transform=_mon("ditto"),
            moves=(MoveSpec(id="earthquake", pp=5), MoveSpec(id="toxic", pp=5)),
        )
        spec = _spec(side_one=SideSpec(pokemon=(copy, _mon("pikachu")), active_index=0))
        request = _request()
        request["side"]["pokemon"][0]["details"] = "Ditto, L100"
        request["side"]["pokemon"][1]["details"] = "Pikachu, L100, M"
        request["active"][0]["moves"] = [{"id": "recover", "pp": 5, "disabled": False}]
        found = self._run(spec, request)
        self.assertEqual(
            ["self_move_set:request_move_absent_from_transformed_copy"],
            [m.predicate for m in found],
        )

    def test_pp_is_still_compared_on_a_transformed_active(self):
        """#1210's axis, and the ONLY place `self_move_pp` is not a tautology:
        `_copied_move_spec` is the one producer of a copied slot's PP that is not
        read straight from the request rows."""

        copy = _mon(
            "cradily",
            pre_transform=_mon("ditto"),
            moves=(MoveSpec(id="earthquake", pp=5), MoveSpec(id="recover", pp=5)),
        )
        spec = _spec(side_one=SideSpec(pokemon=(copy, _mon("pikachu")), active_index=0))
        request = _request()
        request["side"]["pokemon"][0]["details"] = "Ditto, L100"
        request["side"]["pokemon"][1]["details"] = "Pikachu, L100, M"
        request["active"][0]["moves"] = [{"id": "earthquake", "pp": 0, "disabled": True}]
        found = self._run(spec, request)
        self.assertEqual(
            ["self_move_disabled", "self_move_pp"], sorted(_axes(found))
        )

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

    def test_every_reason_is_reported_not_only_the_first(self):
        """The first revision returned the FIRST reason and tested status before
        HP and side conditions, so on the ~28% of boundaries where a status
        difference fired the HP and side-condition checks never ran. Measured by
        review as ~14 masked boundaries in 434."""

        observed = [
            "|-damage|p2a: Squirtle|10/200",
            "|-status|p2a: Squirtle|brn",
            "|-sidestart|p2: Squirtle|Spikes",
            "|turn|4",
        ]
        rendered = ["|-damage|p2a: Squirtle|190/200", "|turn|4"]
        branches = [{"events": rendered, "lossy": [], "attribution_unsafe": False}]
        _found, diagnostics = self._run(branches, observed)
        reasons = " | ".join(diagnostics["reasons"])
        self.assertIn("hp", reasons)
        self.assertIn("side conditions", reasons)
        self.assertIn("status", reasons)

    def test_an_ordinary_damage_line_does_not_wipe_the_folded_status(self):
        """`_STATUS_TO_ENGINE` carries "" as a KEY, so the obvious
        `.get(token, current)` never defaults and a bare `|-damage|` erased the
        status on both sides of the comparison. That erasure produced 830 of the
        969 "mismatched" boundaries in this PR's first census."""

        from pokezero.public_projection import fold_step_lines

        pre = types.SimpleNamespace(p1_hp=200, p2_hp=200, p1_status="NONE", p2_status="SLEEP")
        folded = fold_step_lines(["|-damage|p2a: X|150/200"], pre)
        self.assertEqual("SLEEP", folded.status["p2"])

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
    one sees all of them, needs no log, and is the arm that catches a
    renderer-side relaxation.

    It compares EVERY field `post_state_summary` publishes. The first revision
    compared `post["active_hp"]` and nothing else; independent review probed nine
    synthetic branches and found five silent regions -- status in both
    directions, boosts, a benched mon, and switch/active_index -- all using facts
    already in the payload. One test per region, so a re-narrowing is a red suite.
    """

    PRE = {
        "p1": {
            "active_index": 0,
            "active_hp": 200,
            "active_status": "none",
            "boosts": {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
            "pokemon": [{"hp": 200, "status": "none"}, {"hp": 180, "status": "none"}],
            "species": ["pikachu", "dugtrio"],
        },
        "p2": {
            "active_index": 0,
            "active_hp": 192,
            "active_status": "none",
            "boosts": {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
            "pokemon": [{"hp": 192, "status": "none"}, {"hp": 150, "status": "none"}],
            "species": ["mantine", "dodrio"],
        },
    }

    def _run(self, branches):
        return render_self_consistency_mismatches(
            branches, slot_sides=SLOT_SIDES, pre_summary=self.PRE
        )

    def _branch(self, events, post_p2=None, post_p1=None, **overrides):
        def side(active_index=0, hp=192, status="none", boosts=None, party=None):
            return {
                "active_index": active_index,
                "active_hp": hp,
                "active_status": status,
                "boosts": {**{k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
                           **(boosts or {})},
                "pokemon": party or [{"hp": hp, "status": status}, {"hp": 150, "status": "none"}],
            }

        base = {
            "events": events,
            "lossy": [],
            "attribution_unsafe": False,
            "post": {
                "p1": post_p1 or side(hp=200, party=[{"hp": 200, "status": "none"},
                                                     {"hp": 180, "status": "none"}]),
                "p2": post_p2 or side(),
            },
        }
        base.update(overrides)
        return base

    def _predicates(self, found):
        return sorted({m.predicate for m in found})

    # -- silence -------------------------------------------------------------

    def test_a_render_that_matches_its_own_post_state_is_silent(self):
        events = ["|-heal|p2a: Mantine|252/252", "|turn|2"]
        post = {"active_index": 0, "active_hp": 252, "active_status": "none",
                "boosts": {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
                "pokemon": [{"hp": 252, "status": "none"}, {"hp": 150, "status": "none"}]}
        self.assertEqual([], self._run([self._branch(events, post_p2=post)]))

    def test_a_refused_branch_is_not_compared(self):
        events = ["|-activate|p2a: Mantine|Protect", "|turn|2"]
        post = {"active_index": 0, "active_hp": 252, "active_status": "none",
                "boosts": {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
                "pokemon": [{"hp": 252, "status": "none"}, {"hp": 150, "status": "none"}]}
        branch = self._branch(events, post_p2=post, attribution_unsafe=True)
        self.assertEqual([], self._run([branch]))

    def test_a_branch_with_no_post_state_is_skipped_rather_than_guessed(self):
        self.assertEqual(
            [], self._run([{"events": [], "lossy": [], "attribution_unsafe": False}])
        )

    # -- the five regions review measured silent ------------------------------

    def test_active_hp_the_1211_shape(self):
        events = ["|-activate|p2a: Mantine|Protect", "|turn|2"]
        post = {"active_index": 0, "active_hp": 252, "active_status": "none",
                "boosts": {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
                "pokemon": [{"hp": 252, "status": "none"}, {"hp": 150, "status": "none"}]}
        found = self._run([self._branch(events, post_p2=post)])
        self.assertIn("render_post_state_disagreement:active_hp", self._predicates(found))

    def test_post_says_burn_and_the_render_never_said_so(self):
        events = ["|turn|2"]
        post = {"active_index": 0, "active_hp": 192, "active_status": "burn",
                "boosts": {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
                "pokemon": [{"hp": 192, "status": "burn"}, {"hp": 150, "status": "none"}]}
        found = self._run([self._branch(events, post_p2=post)])
        self.assertIn("render_post_state_disagreement:active_status", self._predicates(found))

    def test_the_render_asserts_a_status_the_post_state_denies(self):
        events = ["|-status|p2a: Mantine|brn", "|turn|2"]
        found = self._run([self._branch(events)])
        self.assertIn("render_post_state_disagreement:active_status", self._predicates(found))

    def test_a_boost_the_post_state_does_not_carry(self):
        events = ["|-boost|p2a: Mantine|atk|2", "|turn|2"]
        found = self._run([self._branch(events)])
        self.assertIn("render_post_state_disagreement:boost:atk", self._predicates(found))

    def test_a_benched_mon_the_render_never_mentions(self):
        events = ["|turn|2"]
        post = {"active_index": 0, "active_hp": 192, "active_status": "none",
                "boosts": {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
                "pokemon": [{"hp": 192, "status": "none"}, {"hp": 0, "status": "none"}]}
        found = self._run([self._branch(events, post_p2=post)])
        self.assertIn("render_post_state_disagreement:party_hp", self._predicates(found))

    def test_a_switch_the_post_state_does_not_agree_with(self):
        events = ["|switch|p2a: Dodrio|Dodrio, L100, M|150/150", "|turn|2"]
        found = self._run([self._branch(events)])
        self.assertIn("render_post_state_disagreement:active_index", self._predicates(found))

    def test_a_switch_the_post_state_DOES_agree_with_is_silent(self):
        """The safer-direction pin on the axis above: an ordinary rendered switch
        must not fire, or every switch turn becomes a defect."""

        events = ["|switch|p2a: Dodrio|Dodrio, L100, M|150/150", "|turn|2"]
        post = {"active_index": 1, "active_hp": 150, "active_status": "none",
                "boosts": {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
                "pokemon": [{"hp": 192, "status": "none"}, {"hp": 150, "status": "none"}]}
        self.assertEqual([], self._run([self._branch(events, post_p2=post)]))

    def test_a_switch_clears_boosts_with_no_protocol_echo(self):
        """Showdown emits nothing for it; the engine emits reset_boosts. Folding
        the boosts forward across a switch would fire on every switch after a
        Swords Dance."""

        pre = {slot: dict(value) for slot, value in self.PRE.items()}
        pre["p2"] = dict(pre["p2"], boosts={**pre["p2"]["boosts"], "atk": 2})
        events = ["|switch|p2a: Dodrio|Dodrio, L100, M|150/150", "|turn|2"]
        post = {"active_index": 1, "active_hp": 150, "active_status": "none",
                "boosts": {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
                "pokemon": [{"hp": 192, "status": "none"}, {"hp": 150, "status": "none"}]}
        found = render_self_consistency_mismatches(
            [self._branch(events, post_p2=post)], slot_sides=SLOT_SIDES, pre_summary=pre
        )
        self.assertEqual([], found)

    def test_a_cured_status_is_not_charged_to_the_renderer(self):
        """`|-curestatus|` is on `events.rs`'s documented never-rendered list, so
        a branch in which the sleeper woke shows `post.status == none` while the
        render still says sleep. Found on the FIRST run of the extended axis,
        firing twice on the base crate in the #1211 fixture's wake-up branch."""

        pre = {slot: dict(value) for slot, value in self.PRE.items()}
        pre["p2"] = dict(
            pre["p2"],
            active_status="sleep",
            pokemon=[{"hp": 192, "status": "sleep"}, {"hp": 150, "status": "none"}],
        )
        events = ["|turn|2"]
        found = render_self_consistency_mismatches(
            [self._branch(events)], slot_sides=SLOT_SIDES, pre_summary=pre
        )
        self.assertEqual([], found)

    def test_the_cure_carve_out_does_not_admit_a_fabricated_status(self):
        """Safer-direction pin. The carve-out is only `post says none AND the
        render is still showing what the world came in with`; a render asserting
        a status `post` denies must still fire."""

        pre = {slot: dict(value) for slot, value in self.PRE.items()}
        pre["p2"] = dict(
            pre["p2"],
            active_status="sleep",
            pokemon=[{"hp": 192, "status": "sleep"}, {"hp": 150, "status": "none"}],
        )
        events = ["|-status|p2a: Mantine|brn", "|turn|2"]
        found = render_self_consistency_mismatches(
            [self._branch(events)], slot_sides=SLOT_SIDES, pre_summary=pre
        )
        self.assertIn(
            "render_post_state_disagreement:active_status", self._predicates(found)
        )

    def test_a_zero_amount_marker_is_INVISIBLE_and_that_is_disclosed(self):
        """The documented blind spot, pinned so it cannot be quietly reclaimed.

        A full-HP absorber's no-op and Protect's no-op have the SAME post-state on
        EVERY published field, which is precisely why the two are ambiguous. Only
        the per-source line decomposition in
        `scripts/engine_transition_differential.py` can separate them.
        """

        events = ["|-activate|p2a: Mantine|Protect", "|turn|2"]
        self.assertEqual([], self._run([self._branch(events)]))


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
