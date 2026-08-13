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

THE ONE-DIRECTIONAL PIN: the defect class this file keeps producing
-------------------------------------------------------------------
Named because it has now recurred **three times in one review thread**, each
time in a different mechanism, and each time the same way: a boundary was pinned
on the side that had *already* failed, and left open on the side that had not.

* the render HP band was pinned against WIDENING and was green when TIGHTENED
  to zero;
* the module header was pinned against OVER-claiming coverage and was green when
  flipped back to UNDER-claiming -- which is the direction #1209 actually
  shipped wrong;
* ``_TOXIC_RAMP_RESET_TAGS`` was pinned against MISSING a reset and was green
  when given an extra one (``-heal``), which silences the axis permanently while
  every downstream number keeps reading exactly as it does today.

The asymmetry is not an accident of care. The failure you just fixed is vivid
and you write the test facing it; the opposite failure has never happened yet,
so nothing prompts the second assertion. **Whenever a pin is added here, write
the mutant that moves the boundary the OTHER way and confirm it dies** -- the
`safer=True` rows in the battery exist for exactly this and have caught every
one of the three above. Where the boundary is a SET rather than a scalar, prefer
deriving it from its source (see
``test_the_reset_tag_set_is_derived_from_the_parser_not_maintained_by_hand``),
because set equality closes both directions at once and cannot go stale by hand.
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
    dropped_move_lines,
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


#: The seven public stat stages, WRITTEN OUT rather than read from
#: `_BOOST_KEYS`. A loop over the constant under test shrinks with the constant,
#: so `_BOOST_KEYS` minus `evasion` would silently mean "six subTests instead of
#: seven, all green" -- the null-world shape. Every set-closure loop below
#: iterates a literal for the same reason; the literal itself is held honest
#: against the engine in `SetClosureTests`.
PUBLIC_BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")

#: Real gen3 moves a request can offer. Any one of them appearing in
#: `_REQUEST_PSEUDO_MOVES` blinds `self_move_set` for that move, so each is pinned
#: to still fire. `rest` is #1212's exemplar move; `recover` is the move behind
#: the census's one open predicate.
REAL_REQUEST_MOVES = ("rest", "recover", "sleeptalk", "protect", "toxic", "earthquake")

#: `AXES`, WRITTEN OUT, for the same reason as `PUBLIC_BOOST_KEYS` and found the
#: same way: `AxisClosureTests.test_every_axis_in_the_closed_set_is_pinned_by_a_test`
#: iterated `AXES` itself, so dropping an axis shrank the loop instead of failing
#: it. Measured per element on `424e1679`: all 20 single-axis deletions were green.
#: An axis deleted from the taxonomy is a queue row that silently stops existing,
#: which is worse than one that always reads zero.
CLOSED_AXES = (
    "active_hp",
    "active_status",
    "weather",
    "side_conditions",
    "self_move_set",
    "self_move_pp",
    "self_move_disabled",
    "self_party_species",
    "self_party_hp",
    "self_item",
    "self_ability",
    "opponent_revealed_species",
    "opponent_revealed_moves",
    "opponent_revealed_item",
    "opponent_revealed_ability",
    "boosts",
    "toxic_count",
    "render_unmatched_transition",
    "render_no_usable_branch",
    "render_post_state_disagreement",
    "render_move_line_dropped",
)

#: The seat-scoped axes, written out for the same reason as `PUBLIC_BOOST_KEYS`.
SEAT_SCOPED_AXES = frozenset(
    {
        "self_move_set",
        "self_move_pp",
        "self_move_disabled",
        "self_party_species",
        "self_party_hp",
        "self_item",
        "self_ability",
    }
)


def _ast_of(function):
    """The parsed body of one function, dedented so a method parses standalone."""

    import ast  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def _dispatch_constants(
    tree,
    *,
    names=("event_type",),
    subscripts=(("parts", 1),),
):
    """Every string literal `tree` COMPARES a protocol-tag expression against.

    #1222's walk keyed on `node.left` being `Name("event_type")` and on that
    alone, so two real spellings of the same dispatch were invisible to it:

    * `"-endability" == event_type` -- constant on the left;
    * `parts[1] == "-endability"` -- the raw field, which is literally what
      `event_type` is assigned FROM two lines above it in
      `showdown._update_toxic_stage`.

    Neither is exotic and both are fail-OPEN: a fifth parser reset written either
    way would leave the derived set short, the equality would fail... in the wrong
    direction, or -- worse -- if the constant were updated to match by hand, the
    derivation would silently stop being a derivation. The canonical spelling
    fails CLOSED already, so this is hardening, not a fix. Precedent for widening
    a scan like this: the `INVOCATION` regex in
    `EveryWorkflowTestCountGuardMatchesItsModuleTests`, widened because
    `python3 -m unittest` is a real spelling of `python -m unittest`.

    Deliberately NOT "any comparison against a string literal". `parts[3]` carries
    the STATUS token (`"tox"`), and `_normalize_identifier(parts[3]) == "tox"` is
    a Call on the left -- both must stay out or the derived set gains members the
    constant can never legitimately hold. The tag expression must be one of
    `names`, or one of the `(container, index)` pairs in `subscripts`.
    """

    import ast  # noqa: PLC0415

    def _is_tag(node) -> bool:
        if isinstance(node, ast.Name):
            return node.id in names
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            index = node.slice
            if not isinstance(index, ast.Constant) or isinstance(index.value, bool):
                return False
            return any(
                node.value.id == container and index.value == position
                for container, position in subscripts
            )
        return False

    def _literals(node) -> set:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
            return {
                element.value
                for element in node.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
        return set()

    found: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if not any(_is_tag(operand) for operand in operands):
            continue
        for operand in operands:
            if not _is_tag(operand):
                found |= _literals(operand)
    return found


def _emitted_axes(*functions) -> set:
    """The `axis=` literals a set of mismatch producers can emit."""

    import ast  # noqa: PLC0415

    found: set = set()
    for function in functions:
        for node in ast.walk(_ast_of(function)):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "axis"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                found.add(node.value.value)
    return found


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

    def test_a_silent_sethp_is_read_by_the_observed_side(self):
        """Pain Split writes `|-sethp|...|[silent]`, which the reused one-turn
        fold does not handle. The whole `active_hp` class on the first census --
        360 worlds, 45 decisions -- was that gap, with the WORLD right and the
        observed side stale. Exemplar `ppc-s0-9800144` p1 round 5."""

        spec = _spec(
            side_two=SideSpec(pokemon=(_mon("squirtle", hp=155), _mon("dodrio")), active_index=0)
        )
        lines = BASE_LINES + (
            "|-sethp|p2a: Squirtle|155/200|[from] move: Pain Split|[silent]",
        )
        self.assertEqual([], _axes(self._fire(world_spec=spec, lines=lines)))

    def test_a_sethp_the_world_disagrees_with_still_fires(self):
        """Safer-direction pin on the fix above."""

        lines = BASE_LINES + (
            "|-sethp|p2a: Squirtle|155/200|[from] move: Pain Split|[silent]",
        )
        self.assertEqual(["active_hp"], _axes(self._fire(lines=lines)))

    def test_cureteam_clears_the_observed_status(self):
        """Aromatherapy / Heal Bell announce `|-cureteam|`, not `|-curestatus|`.
        Missing it produced 1,168 worlds over 146 decisions on a census, with the
        WORLD right and the observed side stale. Exemplar `ppc-s1-10100064` p1
        round 75."""

        lines = BASE_LINES + (
            "|-damage|p2a: Squirtle|150/200 tox|[from] psn",
            "|-cureteam|p2a: Squirtle|[from] move: Aromatherapy",
        )
        spec = _spec(
            side_two=SideSpec(pokemon=(_mon("squirtle", hp=150), _mon("dodrio")), active_index=0)
        )
        self.assertEqual([], _axes(self._fire(world_spec=spec, lines=lines)))

    def test_a_status_the_log_still_asserts_after_a_cureteam_still_fires(self):
        """Safer-direction pin: the cure clears, then a NEW status must be seen."""

        lines = BASE_LINES + (
            "|-cureteam|p2a: Squirtle|[from] move: Aromatherapy",
            "|-status|p2a: Squirtle|tox",
        )
        self.assertEqual(["active_status"], _axes(self._fire(lines=lines)))

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

        # THE CODE, VIA ITS AST -- not via its source text. The previous form of
        # this pin stripped the docstring and then ran `assertNotIn` over what
        # was left, which still included every COMMENT, so it fired the moment a
        # comment named `showdown._reseed_toxic_stage_from_residual` to explain
        # which gate this function mirrors. That is report 4 section 4.8's
        # landmine exactly: a guard that fires on its own explanation, and a
        # source-text scan calling itself structural. Names, attributes and
        # string literals are what "reads the tracker" means; prose is not.
        import ast
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(observed_toxic_multiplier)))
        function = tree.body[0]
        if (
            isinstance(function.body[0], ast.Expr)
            and isinstance(function.body[0].value, ast.Constant)
            and isinstance(function.body[0].value.value, str)
        ):
            function.body = function.body[1:]  # its own docstring, which must discuss it
        touched = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Name):
                touched.add(node.id)
            elif isinstance(node, ast.Attribute):
                touched.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                touched.add(node.value)
        self.assertEqual(
            [],
            sorted(name for name in touched if "toxic_stage" in name),
            "the observed side reads the parser's toxic tracker in CODE, which is "
            "what made this axis compare `x` to `f(x)`",
        )
        # And the pin is not vacuous: the walk really does see this function's
        # identifiers, so an added read would land in `touched`.
        self.assertIn("multiplier", touched)
        self.assertIn("_TOXIC_DENOMINATOR", touched)

        self.assertNotIn("toxic_multiplier", ObservedPublicView.__doc__ or "x")

        # --- AND THE MODULE HEADER, which is where this pin could not see. ---
        # The function-level docstrings were corrected when the tautology was
        # retracted; the module header was not, and the header is the module's
        # OWN COVERAGE CONTRACT -- the first thing a reader is told about what
        # this oracle can and cannot falsify. It went on asserting both
        # retracted claims for a whole revision because every structural pin
        # stopped at a function body.
        from pokezero import public_projection

        header = public_projection.__doc__ or ""
        # WHITESPACE-NORMALISED, because a docstring is REFLOWED every time it is
        # edited. Matching the raw text made every phrase check below vacuous: a
        # mutant that re-asserted #1212's claim with the line break one word
        # further along SURVIVED, since the literal substring was no longer
        # contiguous. A pin on prose that any reflow defeats is not a pin.
        flat = " ".join(header.split())

        # TWO OF THE FOUR CHECKS ARE ABSOLUTE. These two phrasings only ever
        # existed as live assertions, so their presence anywhere is the defect:
        #  - #1209's TAUTOLOGY, named as the live comparison. The header said
        #    `side_conditions.toxic_count` "must equal the parser's public
        #    `replay.toxic_stage`", which is `x` against `f(x)`.
        #  - the miscount that followed from inventing an axis for #1212.
        self.assertNotIn("must equal the parser's public", flat)
        self.assertNotIn("three of which are visible to the state comparator", flat)

        # AND TWO ARE SCOPED, because the scoping is the whole point -- report 4
        # section 4.8's landmine is a guard that fires on its own explanation,
        # and this repo's retraction convention is to keep a wrong claim IN
        # PLACE, quoted, rather than to delete it. So a bare `assertNotIn` here
        # would forbid the CORRECTED header: it has to quote #1212's claim to
        # retract it, name `_materialization_toxic_stage` to say what the
        # producer mutant mutates, and name `_reseed_toxic_stage_from_residual`
        # to say why the two sides are not independent.
        #
        # The rule: MENTIONING either is allowed, making an UNMARKED claim is
        # not. Every paragraph that raises one must also say which way it is
        # being talked about. This is what would have caught the shipped header,
        # whose #1209 and #1212 bullets carried no marker at all.
        # #1212's CLAIM is held to the strict form of that rule, because the
        # loose form did not implement it. Asking only that the paragraph
        # containing the phrase also contain the word `RETRACTED` let the claim
        # be re-asserted as live INSIDE the retraction paragraph -- the marker
        # was already there for the quotation, so the mutant walked in under it
        # and all 87 tests passed. The phrase may therefore occur ONLY as the
        # quoted object of the retraction sentence, and every occurrence is
        # accounted for by counting.
        claim = "the axis that makes the relaxation falsifiable at all"
        retraction = (
            'RETRACTED: the claim that ``self_move_disabled`` is '
            f'"{claim}"'
        )
        self.assertEqual(1, flat.count(retraction), "the in-place retraction is gone")
        self.assertEqual(
            flat.count(retraction),
            flat.count(claim),
            "the module header states #1212's retracted claim somewhere OTHER than "
            "as the quoted object of its retraction -- which is how it would be "
            "re-asserted as live while still passing a marker check",
        )

        scoped = {
            # The parser's tracker may be NAMED to explain its absence; it is a
            # symbol, not a claim, so the paragraph-marker rule is the right
            # strength for it.
            "toxic_stage": ("RETRACTED", "NULL MUTANT", "not independent", "producer mutant"),
        }
        marked = 0
        for raw in header.split("\n\n"):
            paragraph = " ".join(raw.split())
            for phrase, markers in scoped.items():
                if phrase not in paragraph:
                    continue
                self.assertTrue(
                    any(marker in paragraph for marker in markers),
                    f"the module header raises {phrase!r} in a paragraph that does "
                    "not mark whether it is the live comparison or the retracted "
                    "one:\n" + paragraph[:300],
                )
                marked += 1
        # Anti-vacuity: a header that stopped discussing it would pass the loop
        # trivially, and the retraction is the useful part of this docstring.
        self.assertGreaterEqual(
            marked, 2, "the module header no longer discusses the retracted axis"
        )

        # UNDER-CLAIMING, WHICH IS THE DIRECTION THAT ACTUALLY WENT WRONG.
        # Everything above catches the header saying MORE than was measured.
        # Nothing caught it saying LESS -- and #1209 shipped as UNCOVERED for a
        # whole revision when a four-minute producer mutant showed the axis
        # fires. Flipping this bullet back to "UNCOVERED, and the producer mutant
        # was not run" was green. Same shape as the HP band being green when
        # TIGHTENED to zero: a boundary pinned on one side only.
        # Anchored defensively: a bare `split(...)[1]` dies of `IndexError` if
        # the anchor is edited, and an IndexError is a FAKE KILL -- it says the
        # test broke, not that the claim is wrong. Item 6's whole point.
        self.assertIn("#1209 (toxic stage", flat, "the #1209 bullet's anchor is gone")
        self.assertIn("* #1212", flat, "the #1212 bullet's anchor is gone")
        bullet = flat.split("#1209 (toxic stage")[1].split("* #1212")[0]
        self.assertIn(
            "COVERED-MEASURED",
            bullet,
            "the #1209 bullet no longer records that the producer mutant fires",
        )
        self.assertNotIn(
            "UNCOVERED",
            bullet,
            "the #1209 bullet calls the axis uncovered; the producer mutant fires",
        )
        # And "the mutant was not run" may appear ONLY as the quoted thing being
        # denied, by the same counting rule as the claim above.
        self.assertEqual(
            flat.count('That is not "the mutant was not run"'),
            flat.count("the mutant was not run"),
            "the module header says the producer mutant was not run; it was",
        )
        # AND THE RETRACTION IS IN PLACE, not deleted. This repo's convention is
        # that a wrong claim stays, quoted, with what replaces it beside it --
        # #1210's history is three commits doing exactly that. Deleting the claim
        # would satisfy every check above while destroying the record, so the
        # in-place form is pinned directly.
        self.assertIn("RETRACTED: the claim that", flat)

    def test_a_plain_poison_tick_is_not_priced_as_a_toxic_multiplier(self):
        """PLAIN POISON IS NOT TOXIC, and this side used to say it was.

        Gen 3 plain poison charges `maxhp / 8`, which is exactly
        `2 * (maxhp // 16)`. So a plain `|-damage|...|[from] psn` tick divided
        cleanly by the toxic unit and came back as TOXIC STAGE 2 -- a fabricated
        value on the side of the comparison whose whole job is to be observed.

        The PARSER applies the gate that prevents this explicitly, and this side
        omitted it: `showdown._reseed_toxic_stage_from_residual` opens with
        `if "tox" not in new_condition.split(): return`.

        WHY IT MATTERED even though `_axis_toxic_count` also gates on the
        engine's status: that engine-status gate was the ONLY thing standing
        between a fabricated 2 and a firing, and it was itself untested. Worse,
        where the engine DOES say TOXIC the fabricated 2 can MATCH a pre-tick
        counter of 2 and silently absorb a real status disagreement -- a defect
        that makes the oracle read CLEANER, which is the exact anti-instrument
        shape this axis was rebuilt once to escape.
        """

        from pokezero.public_projection import observed_toxic_multiplier

        # 256 // 16 == 16 is the toxic unit; plain poison charges 256 // 8 == 32,
        # a clean multiple of it, which is why the quotient guard cannot see this.
        plain = [
            "|switch|p2a: Squirtle|Squirtle, L100, M|256/256 psn",
            "|-damage|p2a: Squirtle|224/256 psn|[from] psn",
        ]
        self.assertEqual(32, 256 // 8, "plain poison is maxhp/8 in gen 3")
        self.assertEqual(2, 32 // (256 // 16), "and that is exactly 2 toxic units")
        self.assertIsNone(
            observed_toxic_multiplier(plain)["p2"],
            "a plain-poison tick was priced as toxic stage 2",
        )

        # THE OTHER DIRECTION, so this is a gate and not a mute button: the same
        # arithmetic on a `tox`-conditioned tick must still resolve.
        toxic = [
            "|switch|p2a: Squirtle|Squirtle, L100, M|148/256 tox",
            "|-damage|p2a: Squirtle|100/256 tox|[from] psn",
        ]
        self.assertEqual(3, observed_toxic_multiplier(toxic)["p2"])

        # THE GATE IS ON THE CONDITION TOKEN, NOT ON THE LINE. `"tox" in line`
        # reads the same on the fixture above and is wrong: a NICKNAME is
        # arbitrary text the opponent chose, and it sits on the very line the
        # multiplier is read from. This mon is plain-poisoned and nicknamed
        # `toxo`, so a substring gate admits it and prices it as stage 2 again.
        nicknamed = [
            "|switch|p2a: toxo|Squirtle, L100, M|256/256 psn",
            "|-damage|p2a: toxo|224/256 psn|[from] psn",
        ]
        self.assertIn("tox", nicknamed[1], "the fixture must trip a substring gate")
        self.assertIsNone(
            observed_toxic_multiplier(nicknamed)["p2"],
            "a nickname containing `tox` was read as the status token",
        )
        # NOT PINNED, AND SAID SO: `"tox" not in parts[3]` (substring on the
        # condition field rather than on its whitespace-split tokens) is an
        # EQUIVALENT mutant, not a hole. No gen 3 condition token contains `tox`
        # as a proper substring -- the set is {brn, par, slp, frz, psn, tox, fnt}
        # -- so nothing can distinguish the two, and no fixture should be
        # invented to pretend otherwise. Recorded so it is not miscounted as an
        # uncaught mutant later.

        # AN UNPRICEABLE TICK INVALIDATES WHAT WAS PAID; it does not merely fail
        # to add to it. Deleting the `multiplier[slot] = None` and leaving the
        # bare `continue` keeps the earlier, now-wrong multiplier standing, and
        # every assertion above still passes -- they all start from a slot with
        # nothing recorded.
        stale = [
            "|switch|p2a: Squirtle|Squirtle, L100, M|256/256 tox",
            "|-damage|p2a: Squirtle|208/256 tox|[from] psn",  # 48 = 3 units -> paid 3
            "|-damage|p2a: Squirtle|176/256 psn|[from] psn",  # no longer tox: unpriceable
        ]
        self.assertEqual(3, observed_toxic_multiplier(stale[:2])["p2"])
        self.assertIsNone(
            observed_toxic_multiplier(stale)["p2"],
            "an unpriceable tick left the previous multiplier standing",
        )

    def test_every_event_that_resets_the_parsers_ramp_invalidates_what_was_paid(self):
        """A LIVE DEFECT, and the loud direction: a FALSE POSITIVE on a CORRECT world.

        This function reset only on `switch`/`drag`/`replace`. The parser resets
        on four more, and `showdown._reseed_toxic_stage_from_residual`'s own
        comment names the reachable one: `Pokemon.setStatus` replaces
        `statusState` wholesale, so **Rest on an already-toxed mon ends the ramp**
        -- *"a LATER re-tox in the same stint was priced from a stage that no
        longer existed."*

        Measured before the fix, on exactly the sequence below: this returned
        **5**, and `_axis_toxic_count` fired
        `last tick paid multiplier 5, world pre-tick counter 0` against a world
        the parser had correctly licensed. Item 5's whole argument was "the
        parser applies this gate explicitly and this one did not" -- which was
        true of four more gates than the one it fixed.
        """

        from pokezero.public_projection import observed_toxic_multiplier

        paid = [
            "|switch|p2a: Squirtle|Squirtle, L100, M|256/256 tox",
            "|-damage|p2a: Squirtle|176/256 tox|[from] psn",  # 80 = 5 units
        ]
        self.assertEqual(5, observed_toxic_multiplier(paid)["p2"], "the fixture must pay 5")

        for label, tail in (
            ("Rest, cured, re-toxed", [
                "|-status|p2a: Squirtle|slp|[from] move: Rest",
                "|-curestatus|p2a: Squirtle|slp",
                "|-status|p2a: Squirtle|tox",
            ]),
            ("faint", ["|faint|p2a: Squirtle"]),
            ("-curestatus", ["|-curestatus|p2a: Squirtle|tox"]),
            ("-cureteam", ["|-cureteam|p2a: Squirtle|Aromatherapy"]),
            # A re-tox alone restarts Showdown's ramp at stage 1. Nothing has
            # been PAID at that ramp yet, so the answer is "not determined",
            # not "1" and certainly not the pre-Rest 5.
            ("re-tox alone", ["|-status|p2a: Squirtle|tox"]),
        ):
            with self.subTest(reset=label):
                self.assertIsNone(observed_toxic_multiplier(paid + tail)["p2"])

        # THE OTHER DIRECTION, and it is the parser's own rule rather than a
        # convenience: a cure naming a BENCHED mon (`p1: Name`, no `a`) cannot
        # touch the active's ramp, so it must NOT invalidate. Resetting on every
        # same-side cure is exactly the corruption
        # `_is_active_protocol_ident` exists to prevent.
        for bench in ("|-curestatus|p2: Benched|tox", "|-curestatus|p1: Benched|tox"):
            with self.subTest(bench=bench):
                self.assertEqual(5, observed_toxic_multiplier(paid + [bench])["p2"])

        # AND `-cureteam` IS EXEMPT FROM THAT RULE, which is the half no fixture
        # covered. The parser's `elif not active_target and event_type !=
        # "-cureteam"` lets a team-wide cure through with a BENCH ident, because
        # Aromatherapy cures the active too. Dropping the `tag == "-cureteam" or`
        # disjunct leaves every active-ident case passing, so this line is the
        # only thing standing between the docstring's "mirrors the parser" and a
        # rule it does not mirror.
        self.assertIsNone(
            observed_toxic_multiplier(paid + ["|-cureteam|p2: Benched|Aromatherapy"])["p2"],
            "`-cureteam` with a bench ident must still invalidate; the parser "
            "exempts it from the active-ident rule",
        )

    def test_the_reset_tag_set_is_derived_from_the_parser_not_maintained_by_hand(self):
        """The completeness claim, CHECKED rather than asserted.

        `observed_toxic_multiplier`'s docstring says its `None` list is CLOSED.
        That is a completeness claim held true by hand -- which is one level up
        the same defect this whole PR was opened to fix, and it is exactly how
        the four missing resets survived in the first place. If the parser grows
        a fifth reset, nothing else here notices.

        So the set is DERIVED. `showdown._update_toxic_stage` is called
        unconditionally from the parse loop and is the whole protocol reset
        surface outside the switch family, so the tags it dispatches on ARE the
        tags this module must mirror. Set equality, both directions: a parser
        reset we do not mirror fails, and a tag we invalidate on that the parser
        does not fails too.

        The second direction is the one that matters most and was completely
        unpinned. Adding `-heal` to the set silences the multiplier after any
        heal -- and Leftovers ticks every turn -- so the axis would go
        permanently quiet, the census would still read `toxic_count: 0`, and
        NEITHER the battery NOR a full 731-game census could tell that apart from
        a healthy instrument. That is the anti-instrument shape this module's own
        docstring warns about in capitals, sitting on the only coverage #1209
        has.

        The walk itself moved into `_dispatch_constants`, which is WIDER than the
        one this test shipped with: it also reads `"-endability" == event_type`
        and `parts[1] == "-endability"`. That widening changes nothing about
        today's source -- `_update_toxic_stage` uses the canonical spelling
        throughout -- so it is proven on synthetic sources in
        `SetClosureTests.test_the_tag_dispatch_scan_reads_every_real_spelling`
        rather than asserted here.
        """

        from pokezero import showdown
        from pokezero.public_projection import _TOXIC_RAMP_RESET_TAGS

        dispatched = _dispatch_constants(_ast_of(showdown._update_toxic_stage))

        # Anti-vacuity: an empty derivation would make the equality trivially
        # satisfiable by emptying the constant, and a walk that silently stops
        # finding anything is the failure mode of every AST pin.
        self.assertGreaterEqual(len(dispatched), 4, "the AST walk found no dispatch tags")
        self.assertEqual(
            dispatched,
            set(_TOXIC_RAMP_RESET_TAGS),
            "`_TOXIC_RAMP_RESET_TAGS` no longer mirrors the tags "
            "`showdown._update_toxic_stage` dispatches on",
        )

    def test_ordinary_events_between_the_tick_and_the_read_do_not_invalidate_it(self):
        """OVER-invalidation, which is the direction that silences the axis.

        Every other reset assertion says "this event MUST invalidate". Nothing
        said "this event must NOT", and the only must-survive anchors were two
        bench cures -- so widening the reset set was free. Measured: adding
        `-heal` to `_TOXIC_RAMP_RESET_TAGS` passed all 88 tests.

        A silenced axis is invisible to everything downstream: the census reads
        `toxic_count: 0` either way, so no amount of census evidence can
        distinguish it. The failure has to be caught here or not at all.

        `|turn|` and `|upkeep|` are pinned alongside for a different reason --
        see the note below.
        """

        from pokezero.public_projection import observed_toxic_multiplier

        paid = [
            "|switch|p2a: Squirtle|Squirtle, L100, M|256/256 tox",
            "|-damage|p2a: Squirtle|176/256 tox|[from] psn",
        ]
        for label, tail in (
            # The real hole. Leftovers alone puts one of these between almost
            # every tick and every read.
            ("-heal", ["|-heal|p2a: Squirtle|200/256 tox|[from] item: Leftovers"]),
            # These two are pinned as BEHAVIOUR, but note that adding either tag
            # to the reset set is an EQUIVALENT mutant, not a caught one: their
            # `parts[2]` is `45`/`` and the slot filter drops them before the tag
            # check ever runs. Recorded so a later reader does not count them as
            # kills, and so the slot filter itself is pinned.
            ("|turn|", ["|turn|45"]),
            ("|upkeep|", ["|upkeep|"]),
            # A tick on the OTHER side must not disturb this one.
            ("opponent tick", ["|-damage|p1a: Pikachu|100/200 tox|[from] psn"]),
            ("opponent cure", ["|-curestatus|p1a: Pikachu|tox"]),
        ):
            with self.subTest(survives=label):
                self.assertEqual(5, observed_toxic_multiplier(paid + tail)["p2"])

        # And a mon that switches in poisoned, ticks plain, then is re-Toxiced
        # must not carry the plain tick's fiction forward.
        mixed = [
            "|switch|p2a: Squirtle|Squirtle, L100, M|256/256 psn",
            "|-damage|p2a: Squirtle|224/256 psn|[from] psn",
            "|-status|p2a: Squirtle|tox",
            "|-damage|p2a: Squirtle|208/256 tox|[from] psn",
        ]
        self.assertEqual(1, observed_toxic_multiplier(mixed)["p2"])


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

    def test_the_engines_none_sentinel_is_not_a_transform(self):
        """The membership literal that decides whether the carve-out applies AT
        ALL, and it was unpinned in the direction that switches it ON.

        `Pokemon.pre_transform` reads back as the engine's SERIALIZED form, and an
        UNtransformed mon serializes it as `"none;..."` rather than as empty --
        which is why `_pre_transform_species` tests the token against `("",
        "none")`. Delete `"none"` from that tuple and `_is_transformed` flips
        False -> True for every ordinary mon: `_identity_species` starts calling
        them `none`, and the moveset, PP and ability exclusions written for a
        Transform start applying to mons that never transformed. The suite was
        green.

        DISCLOSED: the `""` element of the same tuple is an EQUIVALENT MUTANT and
        is deliberately not pinned here. `token` is `_norm(head)`, so when the
        token is empty the function returns `""` whether or not `""` is in the
        tuple. A test that appeared to kill it would be testing nothing.
        """

        from pokezero.public_projection import (
            _identity_species,
            _is_transformed,
            _pre_transform_species,
        )

        untransformed = types.SimpleNamespace(
            id="squirtle", pre_transform="none;0;0;0;0;0;none:0;none:0"
        )
        self.assertEqual("", _pre_transform_species(untransformed))
        self.assertFalse(_is_transformed(untransformed))
        self.assertEqual("squirtle", _identity_species(untransformed))

        # The positive control, so the assertions above are not satisfiable by a
        # `_is_transformed` that has become constantly False.
        transformed = types.SimpleNamespace(
            id="dugtrio", pre_transform="ditto;1;1;1;1;1;transform:32;none:0"
        )
        self.assertEqual("ditto", _pre_transform_species(transformed))
        self.assertTrue(_is_transformed(transformed))
        self.assertEqual("ditto", _identity_species(transformed))

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

        THE LOOP SOURCE IS A WRITTEN-OUT LITERAL, not `AXES`. Iterating the
        constant under test is the shrinking-loop null world this file's own
        `PUBLIC_BOOST_KEYS` comment names: measured on `424e1679`, deleting any one
        of the 20 axes left this test green, 10 of them still green even with the
        rest of this PR applied (the 7 self axes are caught only as a side effect
        of `_SELF_AXES <= AXES`, the 3 render axes by another literal). So `AXES`
        is pinned against `CLOSED_AXES` first, ORDER INCLUDED -- the tuple is what
        a report's column order comes from -- and everything else iterates the
        literal.
        """

        self.assertEqual(
            list(CLOSED_AXES),
            list(AXES),
            "`AXES` no longer matches the written-out taxonomy; an axis was added, "
            "removed or reordered",
        )
        pinned = {
            name[len("test_") :]
            for name in dir(AxisFiresTests)
            if name.startswith("test_")
        }
        state_axes = {axis for axis in CLOSED_AXES if not axis.startswith("render_")}
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
                "render_move_line_dropped",
            },
            {axis for axis in CLOSED_AXES if axis.startswith("render_")},
        )


class SetClosureTests(unittest.TestCase):
    """The hand-maintained membership sets in this module, closed.

    Sections 1-5 are #1223's, and its own review recorded that the sweep behind
    them was NOT exhaustive -- see the `RENDER_TELEMETRY_ONLY_LOSSY` bullet. It
    was then made exhaustive, over every string collection in the module,
    element by element: 25 survivors on `fb600899`. Sections 6 and 7 are those
    survivors' constants. The count, not the list, is the deliverable, and the
    remaining survivor is disclosed in
    `KnownProducerExclusionTests.test_the_engines_none_sentinel_is_not_a_transform`.

    Each of sections 1-5 was verified UNPINNED by a one-token sweep on #1215 as
    merged -- add or remove a single token and the whole suite stayed green:

    * `_REQUEST_PSEUDO_MOVES` + `"rest"`. It filters observed move ids OUT of the
      `self_move_set` comparison, so one real move id blinds that axis for that
      move on every world of every census, permanently and silently. `rest` occurs
      111 times in the gen3 randbat source, and `self_move_set` carries the
      census's ONLY open predicate.

      MEASURED, on `424e1679`, because the first version of this docstring
      overstated it and the overstatement is the thing that outlives the PR body.
      `PYTHONPATH=<base>/src .venv/bin/python -B -m unittest
      tests.test_public_projection` with one token added:

          + "rest"     Ran 90 tests ... OK        -- blinds `self_move_set` for a
                                                     real move, nothing red
          + "recover"  Ran 90 tests ... FAILED (failures=1)
                       test_a_transform_copys_missing_move_is_a_DIFFERENT_predicate

      So: **four sets survived a one-token mutation, and `rest` here blinds the
      axis to a real move with nothing red.** RETRACTED, as unmeasured: that one
      token takes the direction-2 headline to a clean zero with nothing red. It
      does not. The census's open predicate is on `recover`, and `recover` is
      caught -- by the fixture choice of an unrelated test, which is a defence
      nobody chose and the next fixture edit removes. That accident is what
      `test_a_real_move_the_request_offers_and_the_world_lacks_always_fires`
      replaces with something deliberate.
    * `_BOOST_KEYS` - `"evasion"`. Silences that boost in BOTH render arms and
      falsifies `render_self_consistency_mismatches`'s published claim to compare
      "all seven boosts".
    * `_CURE_TAGS` + `"-heal"`. Nothing pinned it. Note the CORRECTED mechanism:
      on the form Showdown actually emits this is an equivalent mutant, because
      all three folds test the HP tag families first -- see
      `test_both_cure_tags_clear_a_status_at_each_of_the_three_fold_sites`. What
      closes the set is the `-cure*` namespace assertion, and the shadowing is why
      it has to be a lexical assertion rather than a behavioural one.
    * `_SELF_AXES` - `"self_move_set"`. Survives because the constant had no
      consumer anywhere in `src/`, `scripts/` or `tests/` while the module header
      cites it as defining what the self axes evaluate.
    * `RENDER_TELEMETRY_ONLY_LOSSY`, found by independent review of this PR after
      the sweep above missed it -- so the sweep was NOT exhaustive over this
      module's membership sets, and this list is not evidence that it now is. Its
      own comment says it "Mirrors
      `engine_transition_differential._TELEMETRY_ONLY_LOSSY_MARKERS`", which is
      the hand-maintained-mirror construct #1222 exists to have replaced, and it
      had no reference in `tests/` or `scripts/`. Widening it is FAIL-OPEN on the
      render arm: `render_branch_is_usable` returns True, the branch becomes
      eligible to match, and `render_unmatched_transition` /
      `render_no_usable_branch` are suppressed while the published coverage floors
      go UP. That is direction 2's own silent-wrongness mode, inside direction 2.
      Derived here, like `_REQUEST_PSEUDO_MOVES`.

    THE METHOD, unchanged from #1222's `_TOXIC_RAMP_RESET_TAGS`: where the set has
    a real source of truth, DERIVE it and assert set equality in both directions
    with an anti-vacuity floor. Where it has none, pin both directions explicitly
    -- must-contain and must-not-contain. Set equality closes the direction the
    author is not facing, which is this file's named recurring defect class.

    NOTE ON ARM HYGIENE (report 4 section 4.2). Where a derivation reads a file
    rather than a loaded object, the path is resolved from the LOADED module's
    `__file__`, never from this test file's. `tests/conftest.py` prepends the
    running checkout's `src`, so a test resolving `../scripts` from `__file__`
    while `PYTHONPATH` selects a different arm would compare arm A's constant to
    arm B's source of truth and call it agreement.
    """

    # -- 1. `_REQUEST_PSEUDO_MOVES` ------------------------------------------

    def test_the_request_pseudo_move_set_is_derived_from_the_choice_mapper(self):
        """The set is exactly the ids the engine's choice mapper cannot resolve.

        `engine_transition_differential.engine_choice_for_action` is the one place
        in the tree that decides this question, and it decides it in the only
        order that can be authoritative: it tries `move_id in engine_moves` FIRST
        and special-cases only what fails that lookup. The ids it special-cases
        after the lookup ARE the request tokens no engine moveset carries, which
        is this constant's definition verbatim -- and the constant's own comment
        has always named that function. So the comment becomes the assertion.

        Both directions. A pseudo-move the mapper stopped translating fails; a
        REAL move added here fails, because the mapper resolves it and never names
        it.

        ONE PROPERTY OF THIS DERIVATION IS WORTH STATING, because it is an
        invitation to make things worse. It goes red whenever the mapper compares
        `move_id` to ANY new string literal -- including a dead
        `if move_id == "rest" and False:` -- and the obvious way to "fix" a red
        derivation is to add that id to the constant. That specific repair is
        caught by
        `test_a_real_move_the_request_offers_and_the_world_lacks_always_fires`, but
        only for the ids in `REAL_REQUEST_MOVES`. So the two tests are mutually
        protective for those six ids and loud for them; for a seventh real move
        they are not, and the reviewer of a mapper change that adds a `move_id`
        comparison has to decide whether the id is genuinely unresolvable.
        """

        import ast

        from pathlib import Path

        from pokezero import public_projection
        from pokezero.public_projection import _REQUEST_PSEUDO_MOVES

        # From the LOADED module, not from this file -- see the class docstring.
        source = (
            Path(public_projection.__file__).resolve().parents[2]
            / "scripts"
            / "engine_transition_differential.py"
        )
        self.assertTrue(
            source.is_file(),
            f"the derivation source is missing, so this pin cannot run: {source}",
        )
        module = ast.parse(source.read_text(encoding="utf-8"))
        mapper = next(
            (
                node
                for node in ast.walk(module)
                if isinstance(node, ast.FunctionDef)
                and node.name == "engine_choice_for_action"
            ),
            None,
        )
        self.assertIsNotNone(
            mapper, "`engine_choice_for_action` was renamed; the derivation is stale"
        )
        derived = _dispatch_constants(mapper, names=("move_id",), subscripts=())

        # Anti-vacuity, and it must be able to fire on its own: a walk that stops
        # matching returns the empty set, which would make the equality below
        # satisfiable by simply emptying the constant.
        self.assertGreaterEqual(
            len(derived),
            2,
            "the AST walk found no unresolvable-move-id dispatch in "
            "`engine_choice_for_action`",
        )
        self.assertEqual(
            derived,
            set(_REQUEST_PSEUDO_MOVES),
            "`_REQUEST_PSEUDO_MOVES` no longer mirrors the ids "
            "`engine_choice_for_action` cannot resolve on the engine's move list",
        )

    def test_a_real_move_the_request_offers_and_the_world_lacks_always_fires(self):
        """The must-not-contain direction, behaviourally, on both predicates.

        `test_self_move_set` uses `thunderbolt` and the transformed-copy pin uses
        `recover`, so today the census's one open predicate is caught BY ACCIDENT
        OF FIXTURE CHOICE: change either fixture's move and adding that move to
        `_REQUEST_PSEUDO_MOVES` becomes free again. This makes it deliberate for a
        list of real moves, and pins both producer predicates for each -- the
        transformed-copy one being the one the census actually reports.
        """

        from pokezero.public_projection import _REQUEST_PSEUDO_MOVES

        for move_id in REAL_REQUEST_MOVES:
            with self.subTest(move=move_id, producer="world"):
                self.assertNotIn(
                    move_id,
                    _REQUEST_PSEUDO_MOVES,
                    f"{move_id} is a real gen3 move; excluding it blinds "
                    "`self_move_set` for every world that carries it",
                )
                request = _request()
                request["active"][0]["moves"][0]["id"] = move_id
                found = state_projection_mismatches(
                    _context(request=request),
                    _World(_spec()),
                    build_poke_engine_state(_spec()),
                )
                self.assertEqual(["self_move_set"], _axes(found))
                self.assertEqual(
                    ["self_move_set:request_move_absent_from_world"],
                    [m.predicate for m in found],
                )

            with self.subTest(move=move_id, producer="transformed_copy"):
                copy = _mon(
                    "cradily",
                    pre_transform=_mon("ditto"),
                    # A donor moveset that shares nothing with the list above, so
                    # the request row is genuinely absent from the copy.
                    moves=(MoveSpec(id="tackle", pp=5), MoveSpec(id="splash", pp=5)),
                )
                spec = _spec(
                    side_one=SideSpec(pokemon=(copy, _mon("pikachu")), active_index=0)
                )
                request = _request()
                request["side"]["pokemon"][0]["details"] = "Ditto, L100"
                request["side"]["pokemon"][1]["details"] = "Pikachu, L100, M"
                request["active"][0]["moves"] = [
                    {"id": move_id, "pp": 5, "disabled": False}
                ]
                found = state_projection_mismatches(
                    _context(request=request), _World(spec), build_poke_engine_state(spec)
                )
                self.assertEqual(
                    ["self_move_set:request_move_absent_from_transformed_copy"],
                    [m.predicate for m in found],
                )

    # -- 2. `_BOOST_KEYS` -----------------------------------------------------

    def test_the_boost_key_sets_are_derived_from_the_engines_own_side(self):
        """The engine's Side is the source of truth for what a public boost IS.

        `poke_engine`'s Side exposes exactly seven `*_boost` attributes and those
        seven are the public stat stages, so no hand-maintained list here may
        differ from them. Three lists do exist -- `_BOOST_KEYS` (both render
        arms), `_ENGINE_BOOST_FIELD` (the state comparator) and the duplicate
        `_ENGINE_BOOST_ATTR` -- and all three are tied to the engine here, so the
        duplicate cannot drift either.
        """

        from pokezero.public_projection import (
            _BOOST_KEYS,
            _ENGINE_BOOST_ATTR,
            _ENGINE_BOOST_FIELD,
        )

        side = build_poke_engine_state(_spec()).side_one
        engine_attributes = {name for name in dir(side) if name.endswith("_boost")}

        # Anti-vacuity floor, able to fire alone: if `dir()` ever stops surfacing
        # the pyo3 getters, every equality below becomes trivially satisfiable by
        # emptying the constants.
        self.assertGreaterEqual(
            len(engine_attributes),
            7,
            "the engine Side exposed no `*_boost` attributes; the derivation is blind",
        )
        self.assertEqual(
            engine_attributes,
            set(_ENGINE_BOOST_FIELD.values()),
            "`_ENGINE_BOOST_FIELD` no longer mirrors the engine Side's stat stages",
        )
        self.assertEqual(
            _ENGINE_BOOST_FIELD,
            _ENGINE_BOOST_ATTR,
            "the two copies of the boost-key -> engine-attribute map have drifted",
        )
        self.assertEqual(
            set(PUBLIC_BOOST_KEYS),
            set(_ENGINE_BOOST_FIELD),
            "the public boost keys no longer match the engine-attribute map's keys",
        )
        self.assertEqual(
            list(PUBLIC_BOOST_KEYS),
            list(_BOOST_KEYS),
            "`_BOOST_KEYS` is not the seven public stat stages",
        )
        self.assertEqual(
            len(set(_BOOST_KEYS)), len(_BOOST_KEYS), "`_BOOST_KEYS` has a duplicate"
        )

    def test_every_one_of_the_seven_boosts_fires_on_both_arms(self):
        """The published claim, MEASURED on each key rather than asserted once.

        `render_self_consistency_mismatches`'s docstring says it compares "all
        seven boosts" and `_axis_boosts` implies the same for the state
        comparator, but only `atk` was ever pinned on either. Dropping any other
        key silenced that boost in both render arms with nothing red -- and
        `evasion` is the one a too-loose world is most likely to get wrong,
        because Double Team is the only common source and it is never announced
        with a value the constructor can check.
        """

        for key in PUBLIC_BOOST_KEYS:
            with self.subTest(boost=key, arm="state"):
                found = state_projection_mismatches(
                    _context(boosts={"p1": {key: 2}}),
                    _World(_spec()),
                    build_poke_engine_state(_spec()),
                )
                self.assertEqual(["boosts"], _axes(found))
                self.assertEqual([f"boosts:{key}"], [m.predicate for m in found])

            with self.subTest(boost=key, arm="render_self_consistency"):
                helper = RenderSelfConsistencyTests()
                found = helper._run(
                    [helper._branch([f"|-boost|p2a: Mantine|{key}|2", "|turn|2"])]
                )
                self.assertIn(
                    f"render_post_state_disagreement:boost:{key}",
                    sorted({m.predicate for m in found}),
                )

    # -- 3. `_CURE_TAGS` ------------------------------------------------------

    def test_the_cure_tag_set_is_closed_against_the_parsers_cure_dispatch(self):
        """No upstream spells out "tags that clear a status", so the closure is
        assembled from three assertions that together admit nothing else.

        (a) every member is a tag the parser's own public-condition dispatch
        handles -- a cure tag Showdown does not emit is a typo;
        (b) set equality over the `-cure*` namespace, which closes both directions
        inside it;
        (c) every member is IN that namespace, which is what stops `-heal`,
        `-damage`, `-sethp` or `faint` from being added -- the mutation that
        actually matters, since every member wipes status unconditionally at all
        three fold sites.
        """

        from pokezero import showdown
        from pokezero.public_projection import _CURE_TAGS

        sources = {
            "_update_public_pokemon_condition": _dispatch_constants(
                _ast_of(showdown._update_public_pokemon_condition)
            ),
            "_updated_public_condition": _dispatch_constants(
                _ast_of(showdown._updated_public_condition)
            ),
        }
        dispatched = set().union(*sources.values())

        # PER-SOURCE anti-vacuity, and on each source's `-cure*` CONTRIBUTION
        # rather than on its raw tag count -- because the contribution is what
        # the equality below actually consumes.
        #
        # A floor on the UNION cannot fire on a PARTIAL loss of one source, and
        # independent review of #1223 measured exactly that: rename
        # `_update_public_pokemon_condition`'s `-cureteam` dispatch out of the
        # `-cure*` namespace AND drop `-cureteam` from the constant, and this
        # test stays green -- the union floor of 6 is still met by the other
        # source on its own, and the `-cure*` subset equality still holds over
        # what is left. The whole suite killed that mutant only through a
        # positive control elsewhere. A per-source floor is strictly stronger and
        # this one fires on that mutant directly, because the renamed source
        # drops from two `-cure*` tags to one.
        for name, floor in (
            ("_update_public_pokemon_condition", 2),
            ("_updated_public_condition", 1),
        ):
            with self.subTest(source=name):
                contributed = {
                    tag for tag in sources[name] if tag.startswith("-cure")
                }
                self.assertGreaterEqual(
                    len(contributed),
                    floor,
                    f"`showdown.{name}` no longer dispatches on {floor} `-cure*` "
                    "tag(s); the derivation is now reading one source where it "
                    "claims to read two",
                )
        self.assertGreaterEqual(
            len(dispatched),
            6,
            "the AST walk found no public-condition dispatch tags in the parser",
        )
        self.assertTrue(
            set(_CURE_TAGS) <= dispatched,
            f"`_CURE_TAGS` names a tag the parser never dispatches on: "
            f"{sorted(set(_CURE_TAGS) - dispatched)}",
        )
        self.assertEqual(
            {tag for tag in dispatched if tag.startswith("-cure")},
            set(_CURE_TAGS),
            "`_CURE_TAGS` no longer mirrors the parser's `-cure*` dispatch",
        )
        self.assertEqual(
            [],
            [tag for tag in _CURE_TAGS if not tag.startswith("-cure")],
            "a non-`-cure` tag in `_CURE_TAGS` wipes status at all three fold sites",
        )

    def test_both_cure_tags_clear_a_status_at_each_of_the_three_fold_sites(self):
        """The behavioural half, and A CORRECTION TO WHAT IT CAN CLAIM.

        The first version of this test was titled "a heal never clears a status at
        any of the three fold sites" and was a NULL-WORLD TEST under a title
        asserting the opposite: independent review measured it GREEN with `-heal`
        added to `_CURE_TAGS`. Re-measured here, per site, before rewriting it:

            `.venv/bin/python -B` over the three folds with `_CURE_TAGS` patched
            to `("-curestatus", "-cureteam", "-heal")`:
              `|-heal|p2a: Squirtle|200/256 tox|[from] item: Leftovers`
                  base ('TOXIC', 'TOXIC', 'toxic')  mut (same)   -- SAME
              `|-heal|p2a: Squirtle`
                  base ('TOXIC', 'TOXIC', 'toxic')  mut ('NONE', 'NONE', 'none')

        The mechanism is branch ORDER, not the cure branch: all three folds test
        `_HP_TAGS_FIELD3` (`-damage`, `-heal`, `-sethp`) BEFORE `_CURE_TAGS`, so a
        `-heal` carrying a condition field is claimed first and never reaches the
        cure branch. On the only form Showdown emits, `-heal` in `_CURE_TAGS` is an
        EQUIVALENT MUTANT with zero behavioural effect. **What closes `-heal` is
        the `-cure*` namespace assertion in the derivation test above, not
        anything here** -- and the corollary is worth keeping in mind: a fold test
        is structurally blind to any tag the HP branches claim first, so no green
        fold test licenses an addition to `_CURE_TAGS`.

        What this test therefore claims, and no more:

        1. both cure tags DO clear a status at all three sites -- the positive
           control, which is what an emptied or shortened `_CURE_TAGS` fails;
        2. the three-field `|-heal|<ident>` form, the ONE form that reaches the
           cure branch, still preserves status -- a real behavioural kill of the
           `-heal` mutant, recorded as reached-only-by-an-unemitted-line so nobody
           mistakes it for the Leftovers case;
        3. `_CURE_TAGS` is disjoint from the HP tag families, which is the
           structural statement of the shadowing above.
        """

        from pokezero.public_projection import (
            _CURE_TAGS,
            _HP_TAGS_FIELD3,
            _HP_TAGS_FIELD4,
            _fold_public_lines,
            fold_lines_onto_summary,
            fold_step_lines,
        )

        entry = "|switch|p2a: Squirtle|Squirtle, L100, M|256/256 tox"
        # Showdown never emits this: a `-heal` always carries its condition. It is
        # the only `-heal` shape that falls through to the cure branch.
        unemitted_heal = "|-heal|p2a: Squirtle"
        summary = {
            "p2": {
                "active_index": 0,
                "active_hp": 256,
                "active_status": "toxic",
                "boosts": {key: 0 for key in PUBLIC_BOOST_KEYS},
                "pokemon": [{"hp": 256, "status": "toxic"}],
                "species": ["squirtle"],
            }
        }
        pre = _fold_public_lines([entry])

        def folded(line):
            """(whole-log fold, step fold, summary fold) status for one line."""

            return (
                _fold_public_lines([entry, line]).p2_status,
                fold_step_lines([line], pre).status["p2"],
                fold_lines_onto_summary(summary, [line])["p2"]["active_status"],
            )

        # 1. the positive control, on BOTH members, at all three sites.
        for tag, line in (
            ("-curestatus", "|-curestatus|p2a: Squirtle|tox"),
            ("-cureteam", "|-cureteam|p2a: Squirtle|[from] move: Aromatherapy"),
        ):
            with self.subTest(clears=tag):
                self.assertIn(tag, _CURE_TAGS)
                self.assertEqual(("NONE", "NONE", "none"), folded(line))

        # 2. the one form that reaches the cure branch.
        with self.subTest(preserves="the unemitted three-field heal"):
            self.assertEqual(("TOXIC", "TOXIC", "toxic"), folded(unemitted_heal))

        # 3. the shadowing, stated structurally.
        with self.subTest(structure="disjoint from the HP tag families"):
            self.assertEqual(
                set(),
                set(_CURE_TAGS)
                & (set(_HP_TAGS_FIELD3) | set(_HP_TAGS_FIELD4) | {"-status", "faint"}),
                "a cure tag that an earlier fold branch already claims is dead "
                "code at all three sites, so its membership means nothing",
            )

    # -- 4. `_SELF_AXES` ------------------------------------------------------

    #: The two seat-scoped producers, each with the floor on its OWN `axis=`
    #: contribution. One tuple, read by the real derivation below and by the
    #: mutant test after it, so the mutant cannot be pinned against a copy of the
    #: floors that has drifted from the ones that ship.
    SELF_AXIS_FLOORS = (("_axis_self_moves", 3), ("_axis_self_party", 4))

    @staticmethod
    def _self_axes_emitted():
        """`{producer name: the `axis=` literals it can emit}` for the two."""

        from pokezero.public_projection import _axis_self_moves, _axis_self_party

        return {
            "_axis_self_moves": _emitted_axes(_axis_self_moves),
            "_axis_self_party": _emitted_axes(_axis_self_party),
        }

    def _check_self_axes_derivation(self, emitted_by_producer, *, floors):
        """Every assertion the `_SELF_AXES` derivation makes, over a SUPPLIED
        per-producer mapping rather than over the live producers.

        Factored out so the mutant test below runs this exact assertion sequence
        against a perturbed mapping, instead of a re-typed imitation of it.

        DELIBERATELY NO `subTest` in the floor loop, and this is load-bearing:
        `subTest` reports a failure to the TestResult instead of raising, so a
        caller wrapping this in `assertRaises` would see nothing and conclude the
        floor does not fire. The producer name is carried in the message instead.
        """

        from pokezero.public_projection import AXES, _SELF_AXES

        # PER-SOURCE anti-vacuity, same reason as `_CURE_TAGS` above: a floor on
        # the union of two producers is satisfied by either one of them alone, so
        # it cannot fire when one producer stops emitting. Each is floored on its
        # own contribution.
        for name, floor in floors:
            self.assertGreaterEqual(
                len(emitted_by_producer[name]),
                floor,
                f"`{name}` no longer emits {floor} `axis=` literals; the "
                "derivation is reading one producer, not two",
            )
        emitted = set().union(*emitted_by_producer.values())
        self.assertGreaterEqual(
            len(emitted),
            7,
            "the AST walk found no `axis=` literals in the seat-scoped producers",
        )
        self.assertEqual(
            emitted,
            set(SEAT_SCOPED_AXES),
            "the seat-scoped producers emit a different axis set than this file pins",
        )
        self.assertEqual(
            emitted,
            set(_SELF_AXES),
            "`_SELF_AXES` no longer matches the axes `_axis_self_moves` and "
            "`_axis_self_party` emit, and the module header cites it as if it did",
        )
        self.assertTrue(
            set(_SELF_AXES) <= set(AXES),
            "`_SELF_AXES` names an axis outside the closed `AXES` taxonomy",
        )
        return emitted

    def test_the_self_axis_set_is_derived_from_its_two_producers(self):
        """`_SELF_AXES` had NO consumer -- in `src/`, `scripts/` or `tests/`.

        The module header cites it as defining which axes are evaluated only for
        `context.player_id`, but the seat scoping is structural (the two producers
        are handed `sides[observed.slot]` and `observed.self_request`), never a
        lookup against this set. Prose pointing at a constant nothing reads drifts
        silently, and it did: removing `self_move_set` from it was a green suite.

        The fix is to give it a reader, and the honest reader is a derivation --
        the `axis=` literals the two seat-scoped producers can actually emit. Both
        directions: a producer growing an axis that is not listed fails, and a
        member no producer emits fails too.

        What the per-source floor here is worth is pinned by the test below, not
        argued here; #1225 argued it and got it wrong in the safe direction.
        """

        self._check_self_axes_derivation(
            self._self_axes_emitted(), floors=self.SELF_AXIS_FLOORS
        )

    def test_only_the_per_source_floor_catches_an_axis_migrating_between_producers(
        self,
    ):
        """⚠ CORRECTION to #1225's own note on the floor above, which UNDER-claimed
        it -- wrongly, but in the safe direction.

        #1225 recorded: *"the same per-source treatment is applied to `_SELF_AXES`,
        and honestly it closes nothing there: that derivation's union floor of 7
        already equals the total its two producers emit, so any loss already trips
        it. It is applied for uniformity with the rule, not because a hole was
        measured."* The premise is true and the conclusion does not follow. A
        union floor sees LOSS. It cannot see MIGRATION, because migration is not a
        loss -- and a derivation whose whole subject is *which producer emits
        what* is exactly the thing migration corrupts.

        The discriminating case, which is what this test is: move
        `self_move_disabled` from `_axis_self_moves` to `_axis_self_party`. The
        union stays 7, the members stay equal to `_SELF_AXES` and to
        `SEAT_SCOPED_AXES`, `<= AXES` still holds -- every set assertion passes.
        Only the per-source floor fires. So the floor closes SOURCE-ATTRIBUTION
        DRIFT, a class a union floor structurally cannot express, and the note is
        corrected rather than relayed.

        Why it matters beyond bookkeeping: the module header's whole claim about
        these axes is that they are evaluated only for `context.player_id`, and it
        cites the two producers by name as the reason. An axis quietly emitted
        from a different producer than the one documented is that citation going
        stale with every equality still green.

        MEASURED AT SOURCE, not only simulated here. On `6af47d25`, with
        `axis="self_move_disabled"` removed from `_axis_self_moves`' literal and a
        `ProjectionMismatch(axis="self_move_disabled", ...)` construction added to
        `_axis_self_party`:

            PYTHONPATH=<arm>/src python -B -m unittest tests.test_public_projection
            -> Ran 113 tests ... FAILED (failures=1)
               AssertionError: 2 not greater than or equal to 3 :
               `_axis_self_moves` no longer emits 3 `axis=` literals; the
               derivation is reading one producer, not two

        One failure in the module, and it is the floor. With the floors alone
        neutered to `0` the same mutant reads `Ran 1 test ... OK`, which is the
        half of the claim a passing assertion cannot show -- so it is asserted
        below as well, by running the check with `floors=()`.
        """

        migrated_axis = "self_move_disabled"
        real = self._self_axes_emitted()

        # PREMISES, so a migration that migrates nothing cannot pass this test.
        self.assertIn(
            migrated_axis,
            real["_axis_self_moves"],
            f"{migrated_axis} is no longer emitted by `_axis_self_moves`, so the "
            "mutation below moves nothing and proves nothing; pick the axis this "
            "test should be migrating instead",
        )
        self.assertNotIn(migrated_axis, real["_axis_self_party"])

        migrated = {
            "_axis_self_moves": real["_axis_self_moves"] - {migrated_axis},
            "_axis_self_party": real["_axis_self_party"] | {migrated_axis},
        }
        self.assertNotEqual(migrated, real, "the mutation is a no-op")
        self.assertEqual(
            set().union(*migrated.values()),
            set().union(*real.values()),
            "the union MUST be unchanged, or this is a loss and not a migration, "
            "and the union floor would have caught it after all",
        )

        # 1. Every non-per-source assertion still passes. This is the half that
        #    makes the claim non-trivial: without the floor the mutant is green.
        self._check_self_axes_derivation(migrated, floors=())

        # 2. And the per-source floor is what catches it -- with its own message,
        #    so a future failure here cannot be some other assertion's.
        with self.assertRaises(AssertionError) as caught:
            self._check_self_axes_derivation(
                migrated, floors=self.SELF_AXIS_FLOORS
            )
        message = str(caught.exception)
        self.assertIn("the derivation is reading one producer, not two", message)
        self.assertIn("_axis_self_moves", message)

    # -- 5. `RENDER_TELEMETRY_ONLY_LOSSY` ------------------------------------

    def test_the_render_lossy_allowlist_is_derived_from_the_mapper(self):
        """The fifth set, and the one the original sweep MISSED.

        Found by independent review of this PR, which is the honest reason it is
        here: the sweep that found the other four was not exhaustive over this
        module's membership sets, so nothing about this list should be read as
        proof that it now is.

        It is the worst-shaped of the five. Its own comment says it "Mirrors
        `engine_transition_differential._TELEMETRY_ONLY_LOSSY_MARKERS`" -- a
        hand-maintained mirror, which is the construct #1222 replaced with a
        derivation -- and it had no reference in `tests/` or `scripts/`. And unlike
        the other four, widening it FAILS OPEN on the oracle itself:
        `render_branch_is_usable` returns True for the extra marker, the branch
        counts toward `usable`, it becomes eligible to report `matched: True`, and
        so `render_unmatched_transition` and `render_no_usable_branch` are
        suppressed while the published `usable_branches` /
        `self_consistency_branches` coverage floors go UP. A relaxation that raises
        the coverage figure while lowering the mismatch figure is the single most
        expensive shape this module exists to catch, and it was sitting inside it.

        Both directions, against the mapper's own allowlist, plus the fail-open
        direction pinned behaviourally on a real `mark_attribution_unsafe` marker.
        """

        import importlib.util
        import sys

        from pathlib import Path

        from pokezero import public_projection
        from pokezero.public_projection import (
            RENDER_TELEMETRY_ONLY_LOSSY,
            render_branch_is_usable,
        )

        # From the LOADED module -- see the class docstring on arm hygiene.
        source = (
            Path(public_projection.__file__).resolve().parents[2]
            / "scripts"
            / "engine_transition_differential.py"
        )
        self.assertTrue(
            source.is_file(),
            f"the derivation source is missing, so this pin cannot run: {source}",
        )
        spec = importlib.util.spec_from_file_location(
            "_setclosure_engine_transition_differential", source
        )
        mapper = importlib.util.module_from_spec(spec)
        # Registered before exec: the module defines frozen dataclasses, and
        # `dataclasses._is_type` resolves `cls.__module__` through `sys.modules`,
        # so an unregistered module dies with `AttributeError: 'NoneType'`.
        sys.modules[spec.name] = mapper
        spec.loader.exec_module(mapper)
        derived = frozenset(mapper._TELEMETRY_ONLY_LOSSY_MARKERS)

        # Anti-vacuity, able to fire alone: an empty allowlist upstream would make
        # the equality satisfiable by emptying this module's copy, and an empty
        # copy is the fail-CLOSED direction that hides itself as "no mismatches".
        self.assertGreaterEqual(
            len(derived),
            2,
            "the mapper's telemetry-only allowlist is empty; the derivation is blind",
        )
        self.assertEqual(
            derived,
            frozenset(RENDER_TELEMETRY_ONLY_LOSSY),
            "`RENDER_TELEMETRY_ONLY_LOSSY` no longer mirrors "
            "`engine_transition_differential._TELEMETRY_ONLY_LOSSY_MARKERS`",
        )

        # And the fail-open direction, behaviourally. `unattributed_self_damage` is
        # a real marker the renderer emits and it is NOT telemetry-only, so a
        # branch carrying it must stay unusable no matter what this set holds.
        self.assertTrue(render_branch_is_usable([]))
        for marker in sorted(derived):
            with self.subTest(usable=marker):
                self.assertTrue(render_branch_is_usable([marker]))
        for marker in ("unattributed_self_damage", "empty_instruction_list"):
            with self.subTest(refused=marker):
                self.assertFalse(
                    render_branch_is_usable([marker]),
                    f"{marker} is not telemetry-only; admitting it suppresses "
                    "render mismatches AND raises the published coverage floor",
                )

    # -- the remaining fold maps ----------------------------------------------

    def test_the_render_folds_status_and_boost_alias_maps_are_derived(self):
        """Two more maps in this module that nothing read, both raised in review.

        `_ENGINE_STATUS` is the lowercase spelling of `engine_fidelity`'s
        `_STATUS_TO_ENGINE`, which is the parser-side source of truth, so it is
        derivable outright. Dropping `"tox"` from it was green: fold site 3 then
        silently stops writing a toxic status at all, which is the quiet direction.

        `_BOOST_ALIAS` is the identity on the seven public boost keys. A spurious
        alias was green, and an alias is how a protocol key gets folded onto the
        WRONG stage -- so it is pinned as exactly the identity map.
        """

        from pokezero.engine_fidelity import _STATUS_TO_ENGINE
        from pokezero.public_projection import _BOOST_ALIAS, _ENGINE_STATUS

        self.assertGreaterEqual(
            len(_STATUS_TO_ENGINE), 7, "the parser status map is empty; derivation blind"
        )
        self.assertEqual(
            {token: spelling.lower() for token, spelling in _STATUS_TO_ENGINE.items()},
            _ENGINE_STATUS,
            "`_ENGINE_STATUS` is no longer the lowercase spelling of the parser's "
            "`_STATUS_TO_ENGINE`",
        )
        self.assertEqual(
            {key: key for key in PUBLIC_BOOST_KEYS},
            _BOOST_ALIAS,
            "`_BOOST_ALIAS` is not the identity on the seven public boost keys",
        )

    # -- 6. `_HP_TAGS_FIELD3` and `_HP_TAGS_FIELD4` ---------------------------

    def test_the_hp_tag_families_are_derived_from_the_parsers_own_fields(self):
        """The two families named for the protocol FIELD they read, derived from
        the two parser functions that decide which field that is.

        THESE ARE THE SETS FOLD SITE 3 ESCAPED. An exhaustive per-element sweep
        over this module -- delete one element of one string collection, run this
        module -- scored 25 survivors on `fb600899`, and ten of them were these
        two families: `_HP_TAGS_FIELD4`'s own `drag` and `replace`, plus the four
        inline copies `observed_toxic_multiplier` and `fold_lines_onto_summary`
        each kept of them. The copies are hoisted onto these names in the same
        commit as this test; these assertions are what the names are now worth.

        BEHAVIOURAL PROBES, not AST mirrors. A mirror keeps agreeing with an
        upstream that has been renamed or gutted and stops being a derivation
        without saying so. A probe agrees only with the function that still reads
        the field.

        * FIELD3 = the tags for which `showdown._updated_public_condition`
          returns `parts[3]` VERBATIM as the mon's new condition.
        * FIELD4 = the tags `showdown._canonical_replacement_marker` accepts as a
          canonical switch-family payload -- a five-field shape whose `parts[4]`
          it requires to be present and non-empty.

        Each probe runs over a universe that CONTAINS non-members and is asserted
        to reject some of them, so a probe that had degenerated into "accept
        everything" fails here instead of agreeing with whatever it is given.
        That is the anti-vacuity floor for a probe, and it is the shape the plain
        `len(...) >= n` floor cannot express.

        ⚠ THAT ASSERTION SHIPPED ONE-SIDED IN #1225, AND THIS IS THE OTHER SIDE.
        `universe - field` non-empty says the probe rejects something; it does
        NOT say the probe accepts anything, and the set equality below is
        satisfied by two empty sets. Measured on `6af47d25`, this module only:

            # field 3: `_updated_public_condition`'s field-3 branch returns None
            #          instead of `parts[3]`, and `_HP_TAGS_FIELD3 = ()`
            # field 4: `_canonical_replacement_marker` returns False outright,
            #          and `_HP_TAGS_FIELD4 = ()`
            PYTHONPATH=<arm>/src python -B -m unittest -k hp_tag_families
              tests.test_public_projection   ("Ran 1 test", this test alone)

        Both read `Ran 1 test ... OK` -- a probe that discriminates nothing at
        all, agreed with by a constant gutted to match, on both families. (The
        whole module still killed neither; the full suite did, elsewhere, which
        is why this was recorded as a follow-up and not as a hole in coverage.)
        It is nevertheless exactly the one-directional pin this module's own
        header names as its recurring defect class, so both directions are
        asserted here now: the probe must reject part of its universe AND accept
        part of it.

        SCOPE LIMIT OF THE FIELD-4 DERIVATION, stated because it is invisible
        otherwise. The field-4 universe is `event_tags` -- the 19 tags
        `showdown._public_event_from_line` dispatches on -- so the equality only
        constrains `_HP_TAGS_FIELD4` INSIDE that universe. If
        `_canonical_replacement_marker` grew a tag the public-event dispatch does
        not parse (`detailschange`, say), the probe would never offer it and the
        derivation would not notice; growing `faint`, which IS dispatched, is
        caught. That is defensible -- a tag the parser never parses cannot reach
        the fold, so it cannot move a folded HP or a mismatch count -- but it is
        a property of the universe, not of the probe, and it moves the moment the
        parser learns a new tag. The field-3 universe (`condition_tags`, from
        `_updated_public_condition`'s own dispatch) is the same shape.
        """

        from pokezero import showdown
        from pokezero.public_projection import _HP_TAGS_FIELD3, _HP_TAGS_FIELD4

        condition_tags = _dispatch_constants(
            _ast_of(showdown._updated_public_condition)
        )
        self.assertGreaterEqual(
            len(condition_tags),
            6,
            "the AST walk found no condition dispatch in the parser",
        )
        field3 = {
            tag
            for tag in condition_tags
            if showdown._updated_public_condition(
                "300/320 tox",
                event_type=tag,
                parts=["", tag, "p1a: Zigzagoon", "123/320 tox"],
            )
            == "123/320 tox"
        }
        self.assertTrue(
            condition_tags - field3,
            "the field-3 probe accepted every tag the parser dispatches on, so "
            "it discriminates nothing and the equality below is vacuous",
        )
        self.assertTrue(
            field3,
            "the field-3 probe accepted NO tag the parser dispatches on. That is "
            "the OTHER vacuity: with `_HP_TAGS_FIELD3` emptied to match, the "
            "equality below is two empty sets agreeing",
        )
        self.assertEqual(
            field3,
            set(_HP_TAGS_FIELD3),
            "`_HP_TAGS_FIELD3` is no longer the set of tags whose condition the "
            "parser reads out of protocol field 3",
        )

        event_tags = _dispatch_constants(_ast_of(showdown._public_event_from_line))
        self.assertGreaterEqual(
            len(event_tags),
            15,
            "the AST walk found no public-event dispatch in the parser",
        )

        def _switch_family(tag):
            parts = ["", tag, "p1a: Zigzagoon", "Zigzagoon, L84, M", "123/320"]
            return showdown._canonical_replacement_marker(
                "|".join(parts), event_type=tag, parts=parts
            )

        field4 = {tag for tag in event_tags if _switch_family(tag)}
        self.assertTrue(
            event_tags - field4,
            "the field-4 probe accepted every protocol tag, so it discriminates "
            "nothing and the equality below is vacuous",
        )
        self.assertTrue(
            field4,
            "the field-4 probe accepted NO protocol tag. That is the OTHER "
            "vacuity: with `_HP_TAGS_FIELD4` emptied to match, the equality "
            "below is two empty sets agreeing",
        )
        self.assertEqual(
            field4,
            set(_HP_TAGS_FIELD4),
            "`_HP_TAGS_FIELD4` is no longer the parser's switch family, whose "
            "condition lives in protocol field 4",
        )
        self.assertEqual(
            set(),
            set(_HP_TAGS_FIELD3) & set(_HP_TAGS_FIELD4),
            "a tag in both families is claimed by whichever fold branch is "
            "tested first, so its membership in the other one means nothing",
        )

    def test_every_member_of_both_hp_families_moves_the_folded_hp(self):
        """The behavioural half of the derivation, on EVERY member and at the two
        sites that used to hold their own copies.

        `replace` is the member nothing exercised: deleting it from
        `_HP_TAGS_FIELD4` itself was green while the step fold's HP went 123 ->
        -1. Gen 3 never emits `|replace|`, which is the honest reason it had no
        test -- not a reason to leave it unpinned, because the fold is what
        decides whether a mismatch is reported at all.
        """

        from pokezero.public_projection import (
            _HP_TAGS_FIELD3,
            _HP_TAGS_FIELD4,
            fold_lines_onto_summary,
            fold_step_lines,
        )

        pre = types.SimpleNamespace(
            p1_hp=320, p2_hp=320, p1_status="NONE", p2_status="NONE"
        )
        summary = {
            "p2": {
                "active_index": 0,
                "active_hp": 320,
                "active_status": "none",
                "boosts": {key: 0 for key in PUBLIC_BOOST_KEYS},
                "pokemon": [{"hp": 320, "status": "none"}],
                "species": ["zigzagoon"],
            }
        }
        for tag in _HP_TAGS_FIELD3:
            line = f"|{tag}|p2a: Zigzagoon|123/320"
            with self.subTest(field=3, tag=tag):
                self.assertEqual(123, fold_step_lines([line], pre).hp["p2"])
                self.assertEqual(
                    123, fold_lines_onto_summary(summary, [line])["p2"]["active_hp"]
                )
        for tag in _HP_TAGS_FIELD4:
            line = f"|{tag}|p2a: Zigzagoon|Zigzagoon, L84, M|123/320"
            with self.subTest(field=4, tag=tag):
                self.assertEqual(123, fold_step_lines([line], pre).hp["p2"])
                self.assertEqual(
                    123, fold_lines_onto_summary(summary, [line])["p2"]["active_hp"]
                )

    def test_a_leftovers_heal_does_not_silence_the_toxic_count_axis(self):
        """THE HEADLINE, and the reason `observed_toxic_multiplier`'s inline copy
        of `_HP_TAGS_FIELD3` was the highest-value survivor of the sweep.

        The axis prices a Toxic tick as `(previous_hp - hp) / (maxhp // 16)`, so
        it is only correct while `last_hp` is current. Every non-`-damage` member
        of `_HP_TAGS_FIELD3` exists in that loop to keep it current. Drop `-heal`
        from the copy and a Leftovers heal stops updating it: the observed
        multiplier goes 2 -> None and the axis reports nothing, permanently and
        silently, on the most common item in the format (1,371 occurrences in the
        gen 3 randbat source). `toxic_count` is #1209's ONLY coverage. It is the
        same shape #1222 fixed for `-heal` in `_TOXIC_RAMP_RESET_TAGS`, one
        function away.
        """

        from pokezero.public_projection import observed_toxic_multiplier

        for tag, restatement in (
            ("-heal", "|-heal|p2a: Zigzagoon|280/320 tox|[from] item: Leftovers"),
            ("-sethp", "|-sethp|p2a: Zigzagoon|280/320 tox|[from] move: Pain Split"),
        ):
            with self.subTest(restated_by=tag):
                lines = [
                    "|switch|p2a: Zigzagoon|Zigzagoon, L84, M|265/320 tox",
                    restatement,
                    "|-damage|p2a: Zigzagoon|240/320 tox|[from] psn",
                ]
                # 280 - 240 == 40 == 2 * (320 // 16). Without the restatement the
                # loop prices 265 - 240 == 25 against a unit of 20, which is not
                # integral, and the axis returns None.
                self.assertEqual(2, observed_toxic_multiplier(lines)["p2"])

    # -- 7. `_BOOST_WRITE_TAGS` and `_BOOST_CLEAR_TAGS` -----------------------

    def test_the_summary_fold_boost_tag_sets_are_pinned_in_both_directions(self):
        """Two tuples with no upstream, so pinned rather than derived -- and the
        must-NOT-contain half is the one that matters.

        `-clearallboost` is Haze. `fold_lines_onto_summary` has a separate arm for
        it that clears BOTH sides, and that arm is reached only because
        `_BOOST_CLEAR_TAGS` does not claim the tag first. Adding it here is the
        strictly-more-conservative-looking mutant that is actually FAIL-OPEN: Haze
        would clear one side, the other side's stale boosts would then agree with
        a world Haze had already emptied, and `render_post_state_disagreement`
        would go quiet exactly where a too-loose world lives.
        """

        from pokezero.public_projection import _BOOST_CLEAR_TAGS, _BOOST_WRITE_TAGS

        self.assertEqual(("-boost", "-unboost", "-setboost"), _BOOST_WRITE_TAGS)
        self.assertEqual(
            ("-clearboost", "-clearnegativeboost", "-invertboost"),
            _BOOST_CLEAR_TAGS,
        )
        self.assertEqual(
            set(),
            set(_BOOST_WRITE_TAGS) & set(_BOOST_CLEAR_TAGS),
            "a tag in both sets is claimed by whichever branch is tested first",
        )
        for name, tags in (
            ("_BOOST_WRITE_TAGS", _BOOST_WRITE_TAGS),
            ("_BOOST_CLEAR_TAGS", _BOOST_CLEAR_TAGS),
        ):
            with self.subTest(must_not_contain="-clearallboost", constant=name):
                self.assertNotIn(
                    "-clearallboost",
                    tags,
                    f"`-clearallboost` in `{name}` makes Haze clear the acting "
                    "side only, which is fail-open on the render arm",
                )

    def test_each_reachable_boost_tag_folds_the_way_the_render_arm_needs(self):
        """The behavioural half, on the members gen 3 can actually reach.

        `-boost` was already pinned, on all seven keys, by
        `test_every_one_of_the_seven_boosts_fires_on_both_arms`. `-unboost` and
        `-setboost` were not, and both survived deletion: an unpinned `-setboost`
        folds Belly Drum's `atk 6` to 0.

        NOT pinned behaviourally, and disclosed rather than faked:
        `-clearnegativeboost` and `-invertboost` do not exist in gen 3, and the
        fold's treatment of them (zero everything) is an approximation of what
        they mean. Asserting today's output for them would cement the
        approximation as a claim. They are held by the lexical pin above.
        """

        from pokezero.public_projection import fold_lines_onto_summary

        def _summary(**boosts):
            def side():
                return {
                    "active_index": 0,
                    "active_hp": 320,
                    "active_status": "none",
                    "boosts": {**{key: 0 for key in PUBLIC_BOOST_KEYS}, **boosts},
                    "pokemon": [{"hp": 320, "status": "none"}],
                    "species": ["zigzagoon"],
                }

            return {"p1": side(), "p2": side()}

        with self.subTest(tag="-unboost", pins="the sign of the delta"):
            folded = fold_lines_onto_summary(
                _summary(), ["|-unboost|p2a: Zigzagoon|atk|2"]
            )
            self.assertEqual(-2, folded["p2"]["boosts"]["atk"])

        with self.subTest(tag="-setboost", pins="assignment, not a delta"):
            # From -2, Belly Drum's `atk 6` is 6. Folded as a delta it is 4;
            # dropped from the set entirely it stays -2.
            folded = fold_lines_onto_summary(
                _summary(atk=-2), ["|-setboost|p2a: Zigzagoon|atk|6"]
            )
            self.assertEqual(6, folded["p2"]["boosts"]["atk"])

        with self.subTest(tag="-clearboost", pins="the acting side only"):
            folded = fold_lines_onto_summary(
                _summary(atk=2), ["|-clearboost|p2a: Zigzagoon"]
            )
            self.assertEqual(0, folded["p2"]["boosts"]["atk"])
            self.assertEqual(2, folded["p1"]["boosts"]["atk"])

        with self.subTest(tag="-clearallboost", pins="BOTH sides -- the Haze pin"):
            folded = fold_lines_onto_summary(
                _summary(atk=2), ["|-clearallboost|p2a: Zigzagoon"]
            )
            self.assertEqual(0, folded["p2"]["boosts"]["atk"])
            self.assertEqual(
                0,
                folded["p1"]["boosts"]["atk"],
                "Haze cleared one side; `_BOOST_CLEAR_TAGS` has claimed the tag "
                "ahead of the `-clearallboost` arm",
            )

    # -- the scan the derivations share ---------------------------------------

    def test_the_tag_dispatch_scan_reads_every_real_spelling(self):
        """The null-world test for the widening, because the widening is a no-op
        on today's sources.

        `_update_toxic_stage` and the parser's condition dispatch both use the
        canonical `event_type == "..."` throughout, so widening the scan cannot be
        observed anywhere in the tree and would be an untested change asserted to
        work. Synthetic sources are the only place it CAN be measured. A fifth
        parser reset written either non-canonical way was invisible to #1222's
        walk; the canonical form fails closed already, so this is hardening.

        The exclusions matter as much as the inclusions: `parts[3]` carries the
        status token and a normalising call is not a tag expression, so both must
        stay OUT or the derived sets gain members no constant may legitimately
        hold.
        """

        import ast

        for spelling, source in (
            ("canonical", 'if event_type == "-endability":\n    pass\n'),
            ("constant on the left", 'if "-endability" == event_type:\n    pass\n'),
            ("the raw field", 'if parts[1] == "-endability":\n    pass\n'),
            ("inequality", 'if event_type != "-endability":\n    pass\n'),
            ("membership", 'if event_type in {"-endability"}:\n    pass\n'),
            (
                "membership, raw field",
                'if parts[1] in ("-endability",):\n    pass\n',
            ),
        ):
            with self.subTest(reads=spelling):
                self.assertEqual(
                    {"-endability"}, _dispatch_constants(ast.parse(source))
                )

        for excluded, source in (
            ("parts[3] is the status token", 'if parts[3] == "tox":\n    pass\n'),
            (
                "a normalising call is not a tag expression",
                'if _normalize_identifier(parts[3]) == "tox":\n    pass\n',
            ),
            ("an unrelated name", 'if move_name == "rest":\n    pass\n'),
            ("a non-comparison", 'if event_type.startswith("-cure"):\n    pass\n'),
        ):
            with self.subTest(ignores=excluded):
                self.assertEqual(set(), _dispatch_constants(ast.parse(source)))


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


class DroppedMoveLineTests(unittest.TestCase):
    """A protocol line that is SILENTLY ABSENT from an otherwise matching render.

    The gap this closes, stated as the campaign measured it: `consume_move_prelude`'s
    FREEZE/thaw arm ate a benched party-cure clear, which left the ACTIVE's own clear
    next in line for the wake arm, so both were consumed and the render dropped
    `|move|..|healbell|..|[from] Sleep Talk` with `attribution_unsafe=[]` and
    `lossy=[]`. No refusal, no counter, no census row -- so every acceptance
    criterion of the direction-1 census was blind to it before and after #1242.

    WHY THE CHECK CAN LIVE HERE FOR THE PRICE OF TWO `Counter`s.
    `render_projection_mismatch` already holds the observed log and the rendered
    branch in one frame; `fold_step_lines` has no `move` arm, so the `|move|`
    population of BOTH sides falls through its tag chain unhandled. The
    discriminator was in hand and the consumer was not reading it. Direction 1
    cannot host the same check at any price: `truth_differential` has no log in
    scope at all.

    NULL WORLDS, one per test below and none of them shared: a genuine
    `|cant|..|slp` with no `|move|` on either side, an empty tail, a branch whose
    telemetry-only marker says the omission was deliberate, a sibling branch that
    does carry the line, and a display-name/slug pair that is one announcement.
    Reverting `dropped_move_lines` to `return []` turns the two firing tests red
    and leaves every silent one green, which is the direction that matters.
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
            choices={"p1": "sleeptalk", "p2": "splash"},
            observed_lines=observed_lines,
            pre_features=self.PRE,
            module=_FakeCrate(branches),
        )

    #: The exact shape of the defect, in protocol terms: Showdown announces Sleep
    #: Talk AND the move it called; the render announces only Sleep Talk. Nothing
    #: about the state moved, which is why every other axis agrees.
    OBSERVED_CALLEE = [
        "|cant|p1a: Pikachu|slp",
        "|move|p1a: Pikachu|Sleep Talk|p1a: Pikachu",
        "|move|p1a: Pikachu|Heal Bell|p1a: Pikachu|[from]Sleep Talk",
        "|turn|4",
    ]
    RENDERED_WITHOUT_CALLEE = [
        "|cant|p1a: Pikachu|slp",
        "|move|p1a: Pikachu|sleeptalk|p1a: Pikachu",
        "|turn|4",
    ]

    def test_a_silently_dropped_callee_line_is_caught(self):
        """MUST FIRE. The branch folds equal on every axis the comparator had --
        no HP moved, no status moved, nothing fainted -- so before this check the
        boundary was published as `matched` and the missing announcement was
        never counted anywhere in the tree."""

        branches = [
            {
                "events": list(self.RENDERED_WITHOUT_CALLEE),
                "lossy": [],
                "attribution_unsafe": False,
            }
        ]
        found, diagnostics = self._run(branches, self.OBSERVED_CALLEE)
        self.assertEqual(["render_move_line_dropped"], _axes(found))
        # NAMED, not merely counted. A tally that cannot say WHICH line went
        # missing cannot be triaged, and the predicate is the queue key.
        self.assertEqual(
            "render_move_line_dropped:|move|p1a|healbell", found[0].predicate
        )
        self.assertEqual(["|move|p1a|healbell"], diagnostics["move_lines_dropped"])
        # THE DENOMINATOR. Two announcements were compared, not zero.
        self.assertEqual(2, diagnostics["move_lines_compared"])

    def test_the_defect_free_render_of_the_same_boundary_is_silent(self):
        """THE CONTROL for the test above, and the post-#1242 render of the same
        position: the callee line is present, so the same comparator says nothing.
        Without this the firing test could be passing because the axis fires on
        every Sleep Talk boundary."""

        branches = [
            {
                "events": [
                    "|cant|p1a: Pikachu|slp",
                    "|move|p1a: Pikachu|sleeptalk|p1a: Pikachu",
                    "|move|p1a: Pikachu|healbell|p1a: Pikachu|[from] Sleep Talk",
                    "|turn|4",
                ],
                "lossy": [],
                "attribution_unsafe": False,
            }
        ]
        found, diagnostics = self._run(branches, self.OBSERVED_CALLEE)
        self.assertEqual([], _axes(found))
        self.assertEqual([], diagnostics["move_lines_dropped"])
        self.assertEqual(2, diagnostics["move_lines_compared"])

    def test_a_display_name_and_its_slug_are_one_announcement(self):
        """NULL WORLD: the log writes `Heal Bell` and the renderer writes
        `healbell`. Keying on the raw line would report every single announcement
        in the census as dropped -- an instrument that fires on everything is the
        same as one that fires on nothing."""

        self.assertEqual(
            [],
            dropped_move_lines(
                ["|move|p1a: Pikachu|Heal Bell|p1a: Pikachu|[from]Sleep Talk"],
                ["|move|p1a: Pikachu|healbell|p1a: Pikachu|[from] Sleep Talk"],
            ),
        )

    def test_a_target_field_disagreement_is_not_a_dropped_line(self):
        """NULL WORLD: the renderer blanks the target and adds `[still]` on a
        self-target failure where Showdown names the user. That is a different
        (and narrower) defect than an announcement that is not there, and
        conflating them would put a target-rendering question in a bucket whose
        whole claim is `this line does not exist`."""

        self.assertEqual(
            [],
            dropped_move_lines(
                ["|move|p1a: Pikachu|Splash|p1a: Pikachu"],
                ["|move|p1a: Pikachu|splash||[still]"],
            ),
        )

    def test_a_genuine_cant_slp_branch_with_no_move_line_is_silent(self):
        """NULL WORLD, the one the instrument is most likely to get wrong: a mon
        that is asleep and does not act announces NO move on EITHER side. The
        deficit is empty and the boundary must stay quiet."""

        observed = ["|cant|p1a: Pikachu|slp", "|turn|4"]
        branches = [
            {"events": list(observed), "lossy": [], "attribution_unsafe": False}
        ]
        found, diagnostics = self._run(branches, observed)
        self.assertEqual([], _axes(found))
        # And the denominator says so honestly: nothing was compared here, which
        # is NOT the same reading as "compared and clean".
        self.assertEqual(0, diagnostics["move_lines_compared"])

    def test_an_empty_tail_is_silent(self):
        """NULL WORLD: a branch that renders nothing at all. The deficit over an
        empty observed slice is empty, so a boundary with no protocol content
        cannot manufacture a verdict."""

        found, diagnostics = self._run(
            [{"events": [], "lossy": [], "attribution_unsafe": False}], []
        )
        self.assertEqual([], _axes(found))
        self.assertEqual(0, diagnostics["move_lines_compared"])

    def test_a_telemetry_only_marker_makes_the_omission_MARKED_not_a_verdict(self):
        """NULL WORLD for the verdict, MUST-FIRE for the counter.

        `sleeptalk_called_unidentified` renders the callee's damage with no
        `|move|` owner ON PURPOSE -- the class is already counted, by name, in
        direction 1. Reporting it here would bury this axis in the very class it
        exists to be independent of. It is still COUNTED, on its own key: if that
        class is ever closed, the closure has to show up as this counter going to
        zero rather than as the measurement quietly disappearing.
        """

        branches = [
            {
                "events": list(self.RENDERED_WITHOUT_CALLEE),
                "lossy": ["sleeptalk_called_unidentified"],
                "attribution_unsafe": False,
            }
        ]
        found, diagnostics = self._run(branches, self.OBSERVED_CALLEE)
        self.assertEqual([], _axes(found))
        self.assertEqual([], diagnostics["move_lines_dropped"])
        self.assertEqual(
            ["|move|p1a|healbell"], diagnostics["move_lines_dropped_marked"]
        )

    def test_a_sibling_branch_that_carries_the_line_clears_the_verdict(self):
        """MUTATED TOWARD THE SAFER BEHAVIOUR, and it found a real false positive.

        The first revision returned on the FIRST branch whose fold matched. On
        the 16-game control block that reported a dropped `|move|..|Toxic|..|[miss]`
        against a paralysed Pelipper: the full-paralysis sibling folds to exactly
        the same net state -- no damage, no status, no faint -- so which of the
        two the engine enumerated first decided the verdict. The support claim
        (`the observed transition lies in the rendered support`) is satisfied by
        ANY branch, so the deficit claim has to be as well.

        NULL WORLD: take the verdict on the first fold-matching branch instead of
        the whole matching set, and this goes red while every other test here
        stays green.
        """

        branches = [
            # Enumerated FIRST, folds equal, and announces nothing.
            {
                "events": ["|cant|p1a: Pikachu|par", "|turn|4"],
                "lossy": [],
                "attribution_unsafe": False,
            },
            # The branch the log actually took.
            {
                "events": [
                    "|move|p1a: Pikachu|toxic|p2a: Squirtle|[miss]",
                    "|-miss|p1a: Pikachu|p2a: Squirtle",
                    "|turn|4",
                ],
                "lossy": [],
                "attribution_unsafe": False,
            },
        ]
        observed = [
            "|move|p1a: Pikachu|Toxic|p2a: Squirtle|[miss]",
            "|-miss|p1a: Pikachu|p2a: Squirtle",
            "|turn|4",
        ]
        found, diagnostics = self._run(branches, observed)
        self.assertEqual([], _axes(found))
        self.assertEqual(2, diagnostics["matched_branches"])

    def test_every_matching_branch_dropping_it_is_still_a_verdict(self):
        """THE OTHER SIDE of the test above. Widening to "any matching branch
        clears it" must not become "nothing ever fires": when EVERY branch that
        folds equal omits the announcement, there is no render that produced it
        and the verdict stands."""

        branches = [
            {"events": ["|turn|4"], "lossy": [], "attribution_unsafe": False},
            {
                "events": ["|cant|p1a: Pikachu|par", "|turn|4"],
                "lossy": [],
                "attribution_unsafe": False,
            },
        ]
        observed = [
            "|move|p1a: Pikachu|Toxic|p2a: Squirtle|[miss]",
            "|-miss|p1a: Pikachu|p2a: Squirtle",
            "|turn|4",
        ]
        found, diagnostics = self._run(branches, observed)
        self.assertEqual(["render_move_line_dropped"], _axes(found))
        self.assertEqual(2, diagnostics["matched_branches"])

    def test_a_boundary_that_matched_nothing_publishes_a_zero_denominator(self):
        """The deficit is only read on a branch the fold ACCEPTED, so on an
        unmatched boundary the check does not run. It publishes an explicit 0
        rather than omitting the key, so a census summing `move_lines_compared`
        cannot read "no matched boundary here" as "no announcements here"."""

        observed = ["|move|p1a: Pikachu|Tackle|p2a: Squirtle", "|-damage|p2a: Squirtle|10/100"]
        branches = [
            {
                "events": ["|move|p1a: Pikachu|tackle|p2a: Squirtle"],
                "lossy": [],
                "attribution_unsafe": False,
            }
        ]
        found, diagnostics = self._run(branches, observed)
        self.assertEqual(["render_unmatched_transition"], _axes(found))
        self.assertEqual(0, diagnostics["move_lines_compared"])

    def test_an_extra_rendered_announcement_is_not_this_axis(self):
        """SCOPE, stated as a limit rather than left to be discovered. The
        opposite direction -- a `|move|` line the render invents and the log does
        not have -- is by this campaign's doctrine WORSE than a dropped one, and
        it is the same two `Counter`s away. It is deliberately not shipped here:
        it has no measured baseline, and an axis whose false-positive rate is
        unknown cannot be added to a census in the same change that establishes
        the first one's. Registered as a follow-up; this test pins that the
        current axis does NOT fire on it, so the follow-up cannot be mistaken for
        already done."""

        self.assertEqual(
            [],
            dropped_move_lines(
                ["|move|p1a: Pikachu|Tackle|p2a: Squirtle"],
                [
                    "|move|p1a: Pikachu|tackle|p2a: Squirtle",
                    "|move|p1a: Pikachu|splash||[still]",
                ],
            ),
        )


class RenderBandWidthTests(unittest.TestCase):
    """THE WIDTH OF THE HP BAND, pinned from BOTH sides. It was pinned by nothing.

    `_render_mismatch_reasons` is exact on everything deterministic and BANDED on
    HP, and the band is the dominant determinant of the largest figure the render
    arm publishes: HP is **159 of the 206** `render_unmatched_transition`
    reasons. So the band's width is very nearly the render number, and until this
    class existed the whole 80-test module passed with the band set anywhere
    across a range wider than the figure it produces.

    MEASURED, on the shipped tree, by mutating the two constants and re-running
    the module:

    * `_DAMAGE_TOLERANCE` 0.16 -> 0.20, 0.30, 0.40, 0.50, **0.75**: all green.
      First failure at 1.00. So the loose side was open to **4.7x**.
    * `_MIN_TOLERANCE_HP` 5 -> 20, **40**: all green. First failure at 80. So the
      loose side of the floor was open to **8x**.
    * AND THE SAFER DIRECTION, which is the half that says the suite was silent
      rather than lenient: `_MIN_TOLERANCE_HP` -> **1** and -> **0** are both
      green. A strictly-more-conservative variant surviving is report 4 section
      4.4's rule firing -- the boundary was unpinned in both directions, so a
      too-permissive bound and a too-strict one were equally invisible.

    Each test below sits ONE HP off the boundary, on opposite sides of it, at the
    two anchors where the floor and the proportional term each dominate. That
    pins `_DAMAGE_TOLERANCE` to **[0.16, 49/300)** -- that is `[0.16, 0.16333...)`,
    since the anchor fires iff `int(300 * T) + 1 < 50` -- and `_MIN_TOLERANCE_HP`
    to exactly 5. Not to within 2.5-6x.
    """

    #: The proportional anchor. 1000 HP so `int(moved * 0.16) + 1` is far above
    #: the 5 HP floor and the floor cannot mask the term under test.
    WIDE = types.SimpleNamespace(
        p1_hp=1000,
        p2_hp=1000,
        p1_status="NONE",
        p2_status="NONE",
        fainted=frozenset(),
        weather="NONE",
        side_conditions={},
        presence=lambda: {},
    )
    #: The floor anchor. `moved` = 10, so `int(10 * 0.16) + 1 == 2` and the 5 HP
    #: floor is what is actually being compared against.
    NARROW = types.SimpleNamespace(
        p1_hp=100,
        p2_hp=100,
        p1_status="NONE",
        p2_status="NONE",
        fainted=frozenset(),
        weather="NONE",
        side_conditions={},
        presence=lambda: {},
    )

    def _verdict(self, pre, maxhp, observed_hp, rendered_hp):
        """Axes for one boundary whose only disagreement is p2's HP.

        p1 is left unstated on both sides, so it resolves to the pre-state HP on
        both and is skipped -- the fixture isolates one banded comparison.
        """

        observed = [f"|-damage|p2a: Squirtle|{observed_hp}/{maxhp}", "|turn|4"]
        rendered = [f"|-damage|p2a: Squirtle|{rendered_hp}/{maxhp}", "|turn|4"]
        found, _ = render_projection_mismatch(
            state_string="<state>",
            slot_sides=SLOT_SIDES,
            party_display={"p1": ["Pikachu"], "p2": ["Squirtle"]},
            turn=3,
            choices={"p1": "tackle", "p2": "surf"},
            observed_lines=observed,
            pre_features=pre,
            module=_FakeCrate(
                [{"events": rendered, "lossy": [], "attribution_unsafe": False}]
            ),
        )
        return _axes(found)

    # --- the proportional term: `int(moved * _DAMAGE_TOLERANCE) + 1` ----------

    def test_an_hp_gap_one_inside_the_proportional_band_is_silent(self):
        """The SAFER-DIRECTION half. `moved` = 300, so the band is
        `int(300 * 0.16) + 1` = 49, and a 49 HP gap is the widest the band admits.

        TIGHTEN `_DAMAGE_TOLERANCE` below 0.16 and this fires: at 0.15 the band
        is 46 and 49 no longer fits. That is what makes the constant a floor and
        not just a ceiling -- a systematically over-strict band would inflate the
        render figure, and nothing else here would notice.
        """

        self.assertEqual([], self._verdict(self.WIDE, 1000, 700, 749))

    def test_an_hp_gap_one_outside_the_proportional_band_fires(self):
        """The TOLERANT half, and the one the reviewer's 16% -> 40% walk found.

        Same `moved` = 300 and the same band of 49, one HP further out. WIDEN
        `_DAMAGE_TOLERANCE` to **49/300 = 0.16333... or above** and this stops
        firing -- 0.164 already does, and at 0.17 the band is 52 and swallows the
        50 HP gap outright. 0.20, 0.30, 0.40 and 0.75 all swallow it, and all of
        those used to be green.
        """

        self.assertEqual(
            ["render_unmatched_transition"], self._verdict(self.WIDE, 1000, 700, 750)
        )

    # --- the floor: `max(_MIN_TOLERANCE_HP, ...)` -----------------------------

    def test_an_hp_gap_at_the_floor_is_silent(self):
        """`moved` = 10, so the proportional term is 2 and the 5 HP FLOOR governs.

        TIGHTEN `_MIN_TOLERANCE_HP` and this fires: at 4 the band is 4 and the
        5 HP gap no longer fits, and at 0 or 1 -- both green before this test --
        every rounding difference on a small trade becomes a render mismatch.
        """

        self.assertEqual([], self._verdict(self.NARROW, 100, 90, 95))

    def test_an_hp_gap_one_outside_the_floor_fires(self):
        """`moved` = 10 again, one HP further out.

        WIDEN `_MIN_TOLERANCE_HP` to 6 and this stops firing; so does widening
        `_DAMAGE_TOLERANCE` far enough for the proportional term to overtake the
        floor at `moved` = 10, which is what 0.75 does (`int(7.5) + 1` = 8). This
        one test therefore closes the loose side of BOTH constants at this anchor.
        """

        self.assertEqual(
            ["render_unmatched_transition"], self._verdict(self.NARROW, 100, 90, 96)
        )

    def test_the_band_is_anchored_on_movement_and_not_on_max_hp(self):
        """A percentage of MAX HP would be a different instrument, and a wrong one.

        The same 50 HP gap that fires at `moved` = 300 must be SILENT when the
        step moved far more, because a bigger roll spread genuinely admits a
        bigger disagreement. Replace `moved` with a constant or with `maxhp` and
        one of these two directions breaks.
        """

        # moved = 900: band is int(900 * 0.16) + 1 = 145, so 50 fits.
        self.assertEqual([], self._verdict(self.WIDE, 1000, 100, 150))
        # moved = 300: band is 49, so the same 50 does not.
        self.assertEqual(
            ["render_unmatched_transition"], self._verdict(self.WIDE, 1000, 700, 750)
        )


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

    def test_a_benched_mons_STATUS_disagreement_fires(self):
        """The other half of the party comparison, which nothing pinned.

        `party_hp` had the test above; the two-element field tuple it shares with
        `status` did not, so deleting `"status"` from it removed the
        `render_post_state_disagreement:party_status` predicate from the module
        ENTIRELY -- the string occurs once -- and the suite stayed green. That is
        the `_BOOST_KEYS` - `evasion` shape #1223 closed, one function along.

        Note the fixture deliberately puts the disagreement on a BENCHED mon with
        a status the render never mentions: `active_status` is compared
        separately, so an active-slot fixture would be killed by the wrong
        assertion.
        """

        events = ["|turn|2"]
        post = {"active_index": 0, "active_hp": 192, "active_status": "none",
                "boosts": {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")},
                "pokemon": [{"hp": 192, "status": "none"}, {"hp": 150, "status": "burn"}]}
        found = self._run([self._branch(events, post_p2=post)])
        self.assertIn(
            "render_post_state_disagreement:party_status", self._predicates(found)
        )

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

    def test_render_rows_and_render_boundaries_are_two_more_distinct_units(self):
        """`render_axis_boundaries` counts BOUNDARIES, and it used to count ROWS.

        `render_post_state_disagreement` emits one row per (branch, slot, field),
        so one boundary routinely carries several. The aggregate incremented once
        per ROW and published the total under a `_boundaries` key, and every
        surface downstream then labelled it BOUNDARIES: on the published census
        the figure is **80 rows over 23 boundaries**, so the artifact that plan 4
        section 2 designates as the deliverable overstated it by **3.5x**, in a
        table where the neighbouring `mismatched = 206` really is boundaries.

        The two units are now published side by side, because both are wanted and
        neither substitutes for the other: rows are the honest size of the arm's
        output, boundaries are the figure comparable with
        `render_boundaries_compared`.
        """

        def boundary(axes):
            return {"battle_id": "b", "arm": "driver", "worlds": [], "render": {"axes": axes}}

        rows = [
            # One boundary, four rows: two fields x two slots, the real shape.
            boundary(["render_post_state_disagreement"] * 4),
            # A second boundary carrying both axes; two rows, one of each.
            boundary(["render_post_state_disagreement", "render_unmatched_transition"]),
            boundary([]),
        ]
        summary = aggregate_projection_records(rows)

        self.assertEqual(3, summary["render_boundaries_compared"])
        self.assertEqual(2, summary["render_mismatched_boundaries"])
        # BOUNDARIES: the disagreement axis fired on two distinct boundaries.
        self.assertEqual(
            {"render_post_state_disagreement": 2, "render_unmatched_transition": 1},
            summary["render_axis_boundaries"],
        )
        # ROWS: five rows in total, and the key that says ROWS is the one that
        # carries the bigger number.
        self.assertEqual(
            {"render_post_state_disagreement": 5, "render_unmatched_transition": 1},
            summary["render_axis_rows"],
        )
        # A per-axis BOUNDARY count can never exceed the boundaries compared,
        # which is the invariant the row count silently violated: labelled
        # BOUNDARIES, 80 of 3,078 was reported where the truth was 23.
        for axis, count in summary["render_axis_boundaries"].items():
            self.assertLessEqual(
                count, summary["render_mismatched_boundaries"], f"{axis} exceeds boundaries"
            )


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


class BothSeatsAreFoldedTests(unittest.TestCase):
    """Every `("p1", "p2")` slot guard in this module, exercised on the P1 seat.

    Seven of the 25 survivors of the per-element sweep over this module were slot
    guards, and the cause is uniform rather than interesting: every other fixture
    in this file acts on `p2`, so deleting `"p1"` from a guard silences the p1
    seat -- half of every census -- with nothing red. That is not a theoretical
    half: `state_projection_mismatches` is evaluated for `context.player_id`,
    which is p1 as often as it is p2, and the seat a guard drops is the seat
    whose disagreements simply stop being counted.

    One site had BOTH elements surviving -- `_fold_public_lines`'s `fainted` set,
    which nothing asserted on at either seat.
    """

    def test_the_whole_log_fold_reads_both_seats(self):
        from pokezero.public_projection import _fold_public_lines

        folded = _fold_public_lines(
            [
                "|switch|p1a: Pikachu|Pikachu, L100, M|200/200",
                "|switch|p2a: Squirtle|Squirtle, L100, M|320/320",
                "|-damage|p1a: Pikachu|120/200",
                "|-damage|p2a: Squirtle|240/320",
            ]
        )
        self.assertEqual((120, 240), (folded.p1_hp, folded.p2_hp))

    def test_the_whole_log_fold_reports_a_faint_on_either_seat(self):
        from pokezero.public_projection import _fold_public_lines

        entry = [
            "|switch|p1a: Pikachu|Pikachu, L100, M|200/200",
            "|switch|p2a: Squirtle|Squirtle, L100, M|320/320",
        ]
        for slot, ident in (("p1", "p1a: Pikachu"), ("p2", "p2a: Squirtle")):
            with self.subTest(fainted=slot):
                folded = _fold_public_lines(entry + [f"|faint|{ident}"])
                self.assertEqual(frozenset({slot}), folded.fainted)

    def test_the_step_fold_reads_both_seats(self):
        from pokezero.public_projection import fold_step_lines

        pre = types.SimpleNamespace(
            p1_hp=200, p2_hp=320, p1_status="NONE", p2_status="NONE"
        )
        folded = fold_step_lines(
            ["|-damage|p1a: Pikachu|120/200", "|-damage|p2a: Squirtle|240/320"], pre
        )
        self.assertEqual({"p1": 120, "p2": 240}, dict(folded.hp))

    def test_the_toxic_multiplier_is_recovered_for_both_seats(self):
        from pokezero.public_projection import observed_toxic_multiplier

        lines = [
            "|switch|p1a: Pikachu|Pikachu, L100, M|280/320 tox",
            "|switch|p2a: Squirtle|Squirtle, L100, M|280/320 tox",
            "|-damage|p1a: Pikachu|240/320 tox|[from] psn",
            "|-damage|p2a: Squirtle|260/320 tox|[from] psn",
        ]
        # 320 // 16 == 20, so 40 is two ticks and 20 is one.
        self.assertEqual({"p1": 2, "p2": 1}, observed_toxic_multiplier(lines))

    def test_the_render_self_consistency_arm_compares_the_p1_side(self):
        """The predicate carries no slot, so this fixture makes p2 AGREE and only
        p1 disagree -- otherwise the assertion would be satisfied by the p2 arm
        that every other fixture already exercises."""

        helper = RenderSelfConsistencyTests()
        post_p1 = {
            "active_index": 0,
            "active_hp": 100,
            "active_status": "none",
            "boosts": {k: 0 for k in PUBLIC_BOOST_KEYS},
            "pokemon": [{"hp": 100, "status": "none"}, {"hp": 180, "status": "none"}],
        }
        found = helper._run([helper._branch(["|turn|2"], post_p1=post_p1)])
        self.assertIn(
            "render_post_state_disagreement:active_hp",
            sorted({m.predicate for m in found}),
        )

    def test_the_render_transition_arm_compares_the_p1_status(self):
        """`_render_mismatch_reasons`'s status loop, on p1. The log burns Pikachu
        and the rendered branch never says so."""

        helper = RenderProjectionTests()
        observed = ["|-status|p1a: Pikachu|brn", "|turn|4"]
        rendered = ["|turn|4"]
        branches = [{"events": rendered, "lossy": [], "attribution_unsafe": False}]
        _found, diagnostics = helper._run(branches, observed)
        self.assertIn("p1 status", " | ".join(diagnostics["reasons"]))


if __name__ == "__main__":
    unittest.main()
