"""Rest-sleep provenance, BUILD half: attempt count -> payload row -> ``rest_turns``.

The export half (``ShowdownReplayState.rest_sleep_counts``, covered in
``tests/test_observation_spec_v3.py``) records, per sleeping mon, that the sleep came
from its own Rest and how many move ATTEMPTS have already burned off it. This file
covers everything downstream of that: the direct-world row annotation, the world
constructor's ``3 - k`` reconstruction, the capability gate, and the claim that every
input is public-line-derived.

Ground truth for the behaviour all of this exists to reproduce is
``scripts/gen3_switch_differential.py --only hypnosisrestclause hypnosisrestclausecontrol``
(real Node sim), and the engine contract is pinned natively by
``rust/pokezero-search/tests/gen3_rest_sleep_clause.rs``.
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
    battle_spec_from_payload,
)
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.gen3_damage import gen3_hp_stat  # noqa: E402
from pokezero.local_showdown import _apply_rest_sleep_provenance  # noqa: E402
from pokezero.showdown import parse_showdown_replay  # noqa: E402
from pokezero.showdown_fixture import FixturePokemon, pack_team  # noqa: E402


# --- fixtures ---------------------------------------------------------------------

def _dex() -> ShowdownDex:
    def species(species_id, name, types, base, weight):
        return SpeciesInfo(id=species_id, name=name, types=types, base_stats=base, weight_kg=weight)

    def move(move_id, pp):
        return MoveInfo(
            id=move_id, name=move_id, type="normal", category="physical",
            gen3_category="physical", base_power=50, accuracy=100.0, priority=0,
            recoil=False, drain=False, heal=False, status=None, boosts={},
            target="normal", selfdestruct=False, pp=pp,
        )

    return ShowdownDex(
        moves={"bodyslam": move("bodyslam", 15), "rest": move("rest", 10), "surf": move("surf", 15)},
        species={
            "snorlax": species("snorlax", "Snorlax", ("normal",),
                               {"hp": 160, "atk": 110, "def": 65, "spa": 65, "spd": 110, "spe": 30}, 460.0),
            "skarmory": species("skarmory", "Skarmory", ("steel", "flying"),
                                {"hp": 65, "atk": 80, "def": 140, "spa": 40, "spd": 70, "spe": 70}, 50.5),
            "starmie": species("starmie", "Starmie", ("water", "psychic"),
                               {"hp": 60, "atk": 75, "def": 85, "spa": 100, "spd": 85, "spe": 115}, 80.0),
        },
        type_chart={},
    )


_SNORLAX = FixturePokemon(species="Snorlax", moves=("bodyslam",), ability="Immunity",
                          item="Leftovers", level=80,
                          evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
_SKARMORY = FixturePokemon(species="Skarmory", moves=("rest",), ability="Keen Eye",
                           item="Leftovers", level=76,
                           evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
_STARMIE = FixturePokemon(species="Starmie", moves=("surf",), ability="Natural Cure",
                          item="Leftovers", level=79,
                          evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})


def _maxhp(mon: FixturePokemon, dex: ShowdownDex) -> int:
    info = dex.species_info(mon.species)
    return gen3_hp_stat(int(info.base_stats["hp"]), 31, int((mon.evs or {}).get("hp", 0)), mon.level)


def _override() -> BattleStartOverride:
    return BattleStartOverride(
        player_teams={
            "p1": pack_team((_SNORLAX,)),
            "p2": pack_team((_SKARMORY, _STARMIE)),
        },
    )


def _payload(dex: ShowdownDex, *, rest_attempts=None, sleeper_active=False):
    """A minimal two-seat payload whose p2 Skarmory is asleep.

    ``rest_attempts`` is the row annotation under test; ``None`` leaves the row
    unannotated, which is exactly how an opponent-induced sleeper arrives.
    """

    snorlax_hp = _maxhp(_SNORLAX, dex)
    sleeper_row = {
        "species": "Skarmory",
        "condition": "88/100 slp",
        "active": sleeper_active,
    }
    if rest_attempts is not None:
        sleeper_row["restSleepAttempts"] = rest_attempts
    other_row = {"species": "Starmie", "condition": "100/100", "active": not sleeper_active}
    return {
        "turn": 9,
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
        "selfTeamOrder": ["Snorlax"],
        "selfActiveRequestState": {
            "trapped": False, "maybeTrapped": False, "maybeDisabled": False, "maybeLocked": False,
        },
        "selfBenchedMoveHistory": False,
        "sides": {
            "p1": {
                "pokemon": [{
                    "species": "Snorlax",
                    "condition": f"{snorlax_hp}/{snorlax_hp}",
                    "active": True,
                    "moves": [{"id": "bodyslam", "pp": 20, "maxpp": 24, "disabled": False}],
                }],
                "boosts": {}, "volatiles": [], "materializationBlockers": [],
                "toxicStage": 0, "sideConditions": {}, "sideConditionSetTurns": {},
            },
            "p2": {
                "pokemon": [sleeper_row, other_row],
                "boosts": {}, "volatiles": [], "materializationBlockers": [],
                "toxicStage": 0, "sideConditions": {}, "sideConditionSetTurns": {},
            },
        },
    }


_LEADS = [
    "|player|p1|Alice|",
    "|player|p2|Bob|",
    "|switch|p1a: Snorlax|Snorlax, L80|100/100",
    "|switch|p2a: Skarmory|Skarmory, L76|100/100",
    "|turn|1",
]


# --- the reconstruction -----------------------------------------------------------

class RestTurnsReconstructionTests(unittest.TestCase):
    """``rest_turns = 3 - k``, and what happens on either side of it."""

    def setUp(self) -> None:
        self.dex = _dex()

    def _sleeper(self, **kwargs):
        world = battle_spec_from_payload(_payload(self.dex, **kwargs), _override(), dex=self.dex)
        return world.spec.side_two.pokemon[0]

    def test_each_attempt_count_maps_onto_the_engines_own_counter(self) -> None:
        # The engine sets rest_turns 3 on Rest and decrements once per move ATTEMPT,
        # waking at 1. k counts those same attempts off the public |cant| lines, so
        # k attempts spent leave 3 - k -- 3, 2 or 1, never 0.
        for attempts, expected in ((0, 3), (1, 2), (2, 1)):
            with self.subTest(attempts=attempts):
                sleeper = self._sleeper(rest_attempts=attempts)
                self.assertEqual(sleeper.status, "sleep")
                self.assertEqual(sleeper.rest_turns, expected)

    def test_a_rest_sleep_carries_no_elapsed_turn_count(self) -> None:
        # The engine branches on rest_turns first and reads sleep_turns only in its
        # 0 arm, so a Rest sleep has no elapsed-turn count to carry. Setting one
        # would be read by nothing and would misdescribe the state on inspection.
        self.assertEqual(self._sleeper(rest_attempts=1).sleep_turns, 0)

    def test_a_rest_sleep_needs_no_approximation_flag(self) -> None:
        # The whole point: this is EXACT, so it constructs where the position used
        # to be declined outright.
        self.assertEqual(self._sleeper(rest_attempts=0).rest_turns, 3)

    def test_an_unannotated_sleeper_keeps_the_old_behaviour_exactly(self) -> None:
        # An opponent-induced sleeper is never annotated, so it must still fail
        # closed without the flag and still build as freshly-asleep with it. This
        # is the regression guard on the arm the fix does NOT change.
        payload = _payload(self.dex)
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertEqual(caught.exception.reason, "status_unsupported")

        world = battle_spec_from_payload(
            payload, _override(), dex=self.dex, approximate_sleep_turns=True
        )
        induced = world.spec.side_two.pokemon[0]
        self.assertEqual(induced.status, "sleep")
        self.assertEqual(induced.rest_turns, 0)
        self.assertEqual(induced.sleep_turns, 0)

    def test_the_exemption_rides_the_bench(self) -> None:
        # The population this exists to serve. An ACTIVE Rest-sleeper reveals itself
        # next turn either way; a BENCHED one is visible to nothing but the clause.
        benched = self._sleeper(rest_attempts=1, sleeper_active=False)
        active = self._sleeper(rest_attempts=1, sleeper_active=True)
        self.assertEqual(benched.rest_turns, 2)
        self.assertEqual(active.rest_turns, 2)

    def test_an_out_of_range_count_fails_closed_rather_than_clamping(self) -> None:
        # k is 0..2 by construction. A value outside it means the tracker and this
        # arithmetic have drifted apart, and a clamp would turn that into a
        # plausible-looking wrong world instead of a declined decision.
        for bad in (3, 7, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(EngineWorldUnsupported) as caught:
                    battle_spec_from_payload(
                        _payload(self.dex, rest_attempts=bad), _override(), dex=self.dex
                    )
                self.assertEqual(caught.exception.reason, "status_unsupported")

    def test_a_non_integer_count_is_not_coerced(self) -> None:
        # Bools are ints in Python: True must not read as "one attempt spent".
        for bad in (True, False, "1", 1.0, None):
            with self.subTest(bad=bad):
                with self.assertRaises(EngineWorldUnsupported):
                    battle_spec_from_payload(
                        _payload(self.dex, rest_attempts=bad), _override(), dex=self.dex
                    )

    def test_a_fainted_row_never_carries_a_rest_state(self) -> None:
        payload = _payload(self.dex, rest_attempts=1)
        payload["sides"]["p2"]["pokemon"][0]["condition"] = "0 fnt"
        payload["sides"]["p2"]["pokemon"][1]["active"] = True
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        fainted = world.spec.side_two.pokemon[0]
        self.assertEqual(fainted.hp, 0)
        self.assertEqual(fainted.status, "none")
        self.assertEqual(fainted.rest_turns, 0)


# --- the row annotation -----------------------------------------------------------

class RestSleepRowAnnotationTests(unittest.TestCase):
    """``local_showdown._apply_rest_sleep_provenance``: which rows get the count."""

    @staticmethod
    def _rows():
        return [
            {"species": "Skarmory", "condition": "88/100 slp", "active": False},
            {"species": "Starmie", "condition": "100/100", "active": True},
        ]

    @staticmethod
    def _annotate(lines, player="p2"):
        replay = parse_showdown_replay(lines, battle_id="rest-build")
        rows = RestSleepRowAnnotationTests._rows()
        _apply_rest_sleep_provenance(rows, replay, player)
        return rows

    _RESTED = _LEADS + [
        "|move|p2a: Skarmory|Rest|p2a: Skarmory",
        "|-status|p2a: Skarmory|slp|[from] move: Rest",
    ]

    def test_a_rest_sleeper_is_annotated_with_its_attempt_count(self) -> None:
        rows = self._annotate(self._RESTED)
        self.assertEqual(rows[0]["restSleepAttempts"], 0)

    def test_the_count_advances_with_the_public_cant_lines(self) -> None:
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|cant|p2a: Skarmory|slp",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 2)

    def test_bench_turns_do_not_advance_it(self) -> None:
        # The reason the tracker counts attempts: gen3's timer ticks only in
        # slp.onBeforeMove, so a benched Rest does not advance at all. Elapsed
        # turns would build this mon as closer to waking than it is.
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|turn|2",
            "|move|p1a: Snorlax|Body Slam|p2a: Starmie",
            "|turn|3",
            "|move|p1a: Snorlax|Body Slam|p2a: Starmie",
            "|turn|4",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 1)

    def test_an_induced_sleeper_is_never_annotated(self) -> None:
        rows = self._annotate(_LEADS + [
            "|move|p1a: Snorlax|Hypnosis|p2a: Skarmory",
            "|-status|p2a: Skarmory|slp",
        ])
        self.assertNotIn("restSleepAttempts", rows[0])

    def test_waking_retires_the_annotation(self) -> None:
        rows = self._annotate(self._RESTED + ["|-curestatus|p2a: Skarmory|slp"])
        self.assertNotIn("restSleepAttempts", rows[0])

    def test_only_the_sleeping_mons_row_is_touched(self) -> None:
        rows = self._annotate(self._RESTED)
        self.assertNotIn("restSleepAttempts", rows[1])

    def test_the_other_seats_rows_are_never_annotated(self) -> None:
        # Keys are side-scoped, so p2's Rest must not bleed onto a p1 row that
        # happens to carry the same species name.
        replay = parse_showdown_replay(self._RESTED, battle_id="rest-build")
        rows = [{"species": "Skarmory", "condition": "88/100 slp", "active": True}]
        _apply_rest_sleep_provenance(rows, replay, "p1")
        self.assertNotIn("restSleepAttempts", rows[0])

    def test_a_baton_pass_does_not_disturb_a_benched_rest_sleeper(self) -> None:
        # Composition guard against PR #879, which landed in the same parser file
        # and widened which Baton-Passed volatiles transfer and materialize
        # (perish counters, confusion, partial trap). Sleep is a STATUS, not a
        # volatile: ``copyVolatileFrom`` never moves it, and #879 changed only the
        # two BP allowlists -- none of the ``-curestatus`` / ``-cureteam`` /
        # ``faint`` handlers this tracker's clears ride on. A benched Rest must
        # therefore come through a pass with its provenance and count intact.
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|turn|2",
            "|-start|p2a: Starmie|perish3",
            "|move|p2a: Starmie|Baton Pass|p2a: Starmie",
            "|switch|p2a: Starmie|Starmie, L79|100/100|[from] Baton Pass",
            "|-start|p2a: Starmie|perish2",
            "|turn|3",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 1)

    def test_a_malformed_row_is_skipped_rather_than_crashing(self) -> None:
        replay = parse_showdown_replay(self._RESTED, battle_id="rest-build")
        rows = [{"condition": "88/100 slp"}, {"species": None, "condition": "1/1"}]
        _apply_rest_sleep_provenance(rows, replay, "p2")
        self.assertEqual(rows, [{"condition": "88/100 slp"}, {"species": None, "condition": "1/1"}])


# --- leakage ----------------------------------------------------------------------

class RestSleepLeakageTests(unittest.TestCase):
    """Every input is public-protocol-derived, so BOTH seats compute the same k."""

    _LINES = _LEADS + [
        "|move|p2a: Skarmory|Rest|p2a: Skarmory",
        "|-status|p2a: Skarmory|slp|[from] move: Rest",
        "|cant|p2a: Skarmory|slp",
        "|switch|p2a: Starmie|Starmie, L79|100/100",
        "|turn|2",
    ]

    def test_the_count_is_a_pure_function_of_the_public_lines(self) -> None:
        # No request, no packed team, no belief sample: the parser is fed protocol
        # lines and nothing else, and that is enough to produce the annotation.
        replay = parse_showdown_replay(self._LINES, battle_id="leak")
        rows = [{"species": "Skarmory", "condition": "88/100 slp", "active": False}]
        _apply_rest_sleep_provenance(rows, replay, "p2")
        self.assertEqual(rows[0]["restSleepAttempts"], 1)

    def test_both_seats_derive_the_same_count_from_the_same_lines(self) -> None:
        # The real leakage claim. Skarmory's Rest is p2's own business, but every
        # line it is derived from -- |move| Rest, |-status| ... [from] move: Rest,
        # |cant| ... slp -- is broadcast to both players, so p1 reconstructs the
        # identical counter. A value only one seat could compute would be hidden
        # state wearing a public field's name.
        p2_view = parse_showdown_replay(self._LINES, battle_id="leak-p2")
        p1_view = parse_showdown_replay(self._LINES, battle_id="leak-p1")
        self.assertEqual(dict(p1_view.rest_sleep_counts), dict(p2_view.rest_sleep_counts))
        self.assertEqual(set(p1_view.rest_sleep_counts), {"p2:skarmory"})

    def test_lines_the_public_never_sees_cannot_contribute(self) -> None:
        # A Rest whose -status line was never broadcast leaves no trace at all:
        # there is no other surface the tracker can read, so nothing can smuggle
        # a count in through it.
        silent = [line for line in self._LINES if "move: Rest" not in line]
        replay = parse_showdown_replay(silent, battle_id="leak-silent")
        self.assertEqual(dict(replay.rest_sleep_counts), {})
        rows = [{"species": "Skarmory", "condition": "88/100 slp", "active": False}]
        _apply_rest_sleep_provenance(rows, replay, "p2")
        self.assertNotIn("restSleepAttempts", rows[0])


# --- the capability gate ----------------------------------------------------------

class RestTurnsCapabilityTests(unittest.TestCase):
    """``require_rest_turns_support``: the move-trap/charge-state house pattern."""

    def test_the_probe_accepts_the_patched_wheel(self) -> None:
        from pokezero.poke_engine_adapter import require_rest_turns_support
        from pokezero.poke_engine_backend import probe_poke_engine

        if not probe_poke_engine().ready:
            self.skipTest(
                "poke-engine is not installed/ready; rebuild with "
                "scripts/setup_poke_engine.sh /path/to/venv/bin/python"
            )
        require_rest_turns_support()

    def test_the_probe_fails_closed_on_an_engine_that_drops_the_counter(self) -> None:
        # rest_turns is an UPSTREAM field, so a binding accepting the keyword proves
        # nothing. A binding that takes it and drops it builds the Rest-sleeper as an
        # ordinary sleeper and re-arms the clause -- silently. Only the round trip
        # separates the two.
        from types import SimpleNamespace

        from pokezero.poke_engine_adapter import (
            PokeEngineRestTurnsUnsupportedError,
            require_rest_turns_support,
        )

        class _DroppingState:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def to_string(self) -> str:
                return "SNORLAX,SLEEP,rest_turns-went-missing"

        engine = SimpleNamespace(
            State=_DroppingState,
            Side=lambda **kwargs: SimpleNamespace(**kwargs),
            Pokemon=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        with self.assertRaises(PokeEngineRestTurnsUnsupportedError) as caught:
            require_rest_turns_support(engine)
        self.assertIn("setup_poke_engine.sh", str(caught.exception))

    def test_the_probe_fails_closed_on_a_half_installed_binding(self) -> None:
        from types import SimpleNamespace

        from pokezero.poke_engine_adapter import (
            PokeEngineRestTurnsUnsupportedError,
            require_rest_turns_support,
        )

        with self.assertRaises(PokeEngineRestTurnsUnsupportedError):
            require_rest_turns_support(SimpleNamespace())

    def test_a_rest_spec_gates_on_the_capability_at_render_time(self) -> None:
        # The adapter-side half of the gate: a spec carrying rest_turns must not be
        # handed to an engine that would drop it, even if the world constructor was
        # bypassed.
        from types import SimpleNamespace

        from pokezero.poke_engine_adapter import (
            MoveSpec,
            PokeEngineRestTurnsUnsupportedError,
            PokemonSpec,
            SideSpec,
            build_poke_engine_state,
        )
        from pokezero.poke_engine_adapter import BattleSpec

        def _mon(rest_turns: int) -> PokemonSpec:
            return PokemonSpec(
                id="snorlax", level=80, types=("normal",), hp=100, maxhp=100,
                attack=100, defense=100, special_attack=100, special_defense=100,
                speed=100, moves=(MoveSpec(id="bodyslam"),), status="sleep",
                rest_turns=rest_turns,
            )

        class _DroppingState:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def to_string(self) -> str:
                return "rest_turns-went-missing"

        engine = SimpleNamespace(
            State=_DroppingState,
            Side=lambda **kwargs: SimpleNamespace(**kwargs),
            Pokemon=lambda **kwargs: SimpleNamespace(**kwargs),
            Move=lambda **kwargs: SimpleNamespace(**kwargs),
            SideConditions=lambda **kwargs: SimpleNamespace(**kwargs),
            VolatileStatusDurations=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        resting = BattleSpec(
            side_one=SideSpec(pokemon=(_mon(2),)),
            side_two=SideSpec(pokemon=(_mon(0),)),
        )
        with self.assertRaises(PokeEngineRestTurnsUnsupportedError) as caught:
            build_poke_engine_state(resting, module=engine)
        self.assertIn("setup_poke_engine.sh", str(caught.exception))

        # The control that makes the assertion above a measurement of rest_turns
        # rather than of the fake engine: the SAME engine renders a spec with no
        # Rest on it. Without this the test would pass on any construction error,
        # since every capability error subclasses PokeEngineUnavailableError.
        awake = BattleSpec(
            side_one=SideSpec(pokemon=(_mon(0),)),
            side_two=SideSpec(pokemon=(_mon(0),)),
        )
        build_poke_engine_state(awake, module=engine)


if __name__ == "__main__":
    unittest.main()
