"""Mid-charge (two-turn move) state: materialization + world construction.

The silent gap this closes: the public mid-charge state never reached world
construction, so a sampled world was built with the charging Pokemon FREE and the
engine started a FRESH charge instead of releasing (repro seed 1350004 step 66).
The engine itself needed no change — `active_is_charging_move` locks
`get_all_options` to the committed move and `generate_instructions` releases it —
so the volatile IS the commitment, and the only thing missing was surfacing it.

Both halves have to land together: the engine_world allowlist observes nothing
until the parser puts the volatile in the payload.

Protocol, read off the real sim rather than assumed:

    charge   |move|p1a: Exeggutor|Solar Beam||[still]
             |-prepare|p1a: Exeggutor|Solar Beam
    release  |move|p1a: Exeggutor|Solar Beam|p2a: Snorlax|[from] lockedmove
    cancel   |cant|p1a: Exeggutor|par        <- and NOTHING else
    in sun   |move|...||[still] + |-prepare| + |-anim| + damage, ALL one turn

Two lines here catch a wrong guess.

`cant` is the first: gen 3's `twoturnmove.onMoveAborted` drops the volatile with
no `-end`, so a full paralysis on the release turn cancels the charge outright
and the Pokemon re-charges from scratch afterwards. A parser that clears only on
release carries a phantom charge forever.

`-anim` is the second, and it is the one the differential control turned up:
Solar Beam in SUN still announces `-prepare` even though it skips the charge
turn, firing on the spot instead. So `-prepare` alone does NOT mean "is
charging" — the discriminator is whether a `[from] lockedmove` release follows or
an `-anim` lands immediately.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.showdown import (  # noqa: E402
    TRACKED_VOLATILES,
    _CHARGE_MOVE_VOLATILES,
    _ReplayParser,
    _update_volatiles,
)

P1 = "p1a: Exeggutor"
P2 = "p2a: Snorlax"


def _feed(*lines: str) -> set[str]:
    """Run raw protocol lines through the volatile fold and return p1's set."""

    volatiles: dict[str, set[str]] = {"p1": set(), "p2": set()}
    for line in lines:
        _update_volatiles(line.split("|"), volatiles)
    return volatiles["p1"]


class ChargeVolatileVocabularyTest(unittest.TestCase):
    def test_the_charge_state_is_tracked_and_distinct_from_the_charge_move(self) -> None:
        # `charge` is the Charge MOVE (the Electric-damage doubler). Conflating the
        # two would hand a Solar Beam user a phantom Electric boost.
        self.assertIn("solarbeam", TRACKED_VOLATILES)
        self.assertIn("charge", TRACKED_VOLATILES)
        self.assertNotIn("charge", _CHARGE_MOVE_VOLATILES)

    def test_only_the_reachable_charge_move_is_tracked(self) -> None:
        # Of the 17 dex moves flagged `charge: 1`, Solar Beam is the only one in the
        # gen3 randbats pool (4 sets). Tracking the rest would add vocabulary rows
        # for states that cannot occur.
        self.assertEqual(_CHARGE_MOVE_VOLATILES, frozenset({"solarbeam"}))


class ChargeStateParsingTest(unittest.TestCase):
    def test_prepare_sets_the_charge_state(self) -> None:
        self.assertEqual(
            _feed(f"|move|{P1}|Solar Beam||[still]", f"|-prepare|{P1}|Solar Beam"),
            {"solarbeam"},
        )

    def test_release_clears_it(self) -> None:
        self.assertEqual(
            _feed(
                f"|move|{P1}|Solar Beam||[still]",
                f"|-prepare|{P1}|Solar Beam",
                f"|move|{P1}|Solar Beam|{P2}|[from] lockedmove",
            ),
            set(),
        )

    def test_full_paralysis_on_the_release_turn_cancels_it(self) -> None:
        # THE pin. `cant` is the only announcement the charge is over — there is no
        # `-end` — so a parser that waits for a release keeps it forever.
        self.assertEqual(
            _feed(
                f"|move|{P1}|Solar Beam||[still]",
                f"|-prepare|{P1}|Solar Beam",
                f"|cant|{P1}|par",
            ),
            set(),
        )

    def test_any_cant_cancels_it(self) -> None:
        # onMoveAborted fires on every consumed action, not just paralysis.
        for reason in ("par", "slp", "frz", "flinch", "recharge", "move: Attract"):
            with self.subTest(reason=reason):
                self.assertEqual(
                    _feed(
                        f"|move|{P1}|Solar Beam||[still]",
                        f"|-prepare|{P1}|Solar Beam",
                        f"|cant|{P1}|{reason}",
                    ),
                    set(),
                )

    def test_a_cancelled_charge_re_arms_from_scratch(self) -> None:
        # The consequence: after the cancel the Pokemon charges again, and the state
        # is set by the NEW `-prepare`, not left over from the old one.
        self.assertEqual(
            _feed(
                f"|move|{P1}|Solar Beam||[still]",
                f"|-prepare|{P1}|Solar Beam",
                f"|cant|{P1}|par",
                f"|move|{P1}|Solar Beam||[still]",
                f"|-prepare|{P1}|Solar Beam",
            ),
            {"solarbeam"},
        )

    def test_the_charge_turns_own_move_line_does_not_clear_it(self) -> None:
        # `|move|...||[still]` is consumed by the same arm that handles the release,
        # and lands BEFORE the `-prepare` that arms the state. Ordering pin.
        self.assertEqual(
            _feed(f"|move|{P1}|Solar Beam||[still]", f"|-prepare|{P1}|Solar Beam"),
            {"solarbeam"},
        )

    def test_a_sun_boosted_solar_beam_leaves_no_charge_behind(self) -> None:
        # Caught by the differential control. In SUN the beam skips its charge turn
        # but Showdown STILL emits `|move|...||[still]` and `|-prepare|`, then fires
        # in the SAME turn via `|-anim|` + damage with no second `|move|` line. So
        # `-prepare` alone does not mean "is charging": without the `-anim` arm this
        # Pokemon carries a phantom charge until its next action, and the world
        # offers it only Solar Beam while it is in fact free.
        self.assertEqual(
            _feed(
                f"|move|{P1}|Solar Beam||[still]",
                f"|-prepare|{P1}|Solar Beam",
                f"|-anim|{P1}|Solar Beam|{P2}",
            ),
            set(),
        )

    def test_the_two_turn_release_emits_no_anim(self) -> None:
        # The other half of that discriminator, so the arm above cannot be
        # over-eager: a genuine two-turn release is a real `|move|` line tagged
        # `[from] lockedmove`, and the charge survives the turn it was armed on.
        self.assertEqual(
            _feed(f"|move|{P1}|Solar Beam||[still]", f"|-prepare|{P1}|Solar Beam"),
            {"solarbeam"},
        )

    def test_no_charge_state_without_a_prepare(self) -> None:
        # Leakage check: the state is set by the PUBLIC announcement only. A move
        # line alone — or an opponent's charge — never puts it on this slot.
        self.assertEqual(_feed(f"|move|{P1}|Solar Beam|{P2}"), set())
        self.assertEqual(_feed(f"|-prepare|{P2}|Solar Beam"), set())

    def test_an_untracked_charge_move_is_not_recorded(self) -> None:
        # Fly/Dig/Dive are real charge moves absent from the pool. They must not
        # produce an out-of-vocabulary token.
        for move in ("Fly", "Dig", "Dive", "Razor Wind"):
            with self.subTest(move=move):
                self.assertEqual(_feed(f"|-prepare|{P1}|{move}"), set())


class ChargeStateSwitchTest(unittest.TestCase):
    """Switching out ends the charge — the volatile belongs to the mon that left."""

    def _parser(self) -> _ReplayParser:
        parser = _ReplayParser()
        parser.feed([
            "|player|p1|Alice|1|",
            "|player|p2|Bob|2|",
            "|switch|p1a: Exeggutor|Exeggutor, M, L100|300/300",
            "|switch|p2a: Snorlax|Snorlax, M, L100|460/460",
            "|turn|1",
        ])
        return parser

    def test_switch_out_clears_the_charge_state(self) -> None:
        parser = self._parser()
        parser.feed([
            f"|move|{P1}|Solar Beam||[still]",
            f"|-prepare|{P1}|Solar Beam",
            "|turn|2",
        ])
        self.assertIn("solarbeam", parser.volatiles["p1"])

        parser.feed(["|switch|p1a: Tangela|Tangela, F, L100|250/250", "|turn|3"])
        self.assertNotIn(
            "solarbeam",
            parser.volatiles["p1"],
            "the charge belongs to the Pokemon that left the field",
        )

    def test_the_state_reaches_the_snapshot(self) -> None:
        parser = self._parser()
        parser.feed([f"|move|{P1}|Solar Beam||[still]", f"|-prepare|{P1}|Solar Beam", "|turn|2"])
        self.assertIn("solarbeam", set(parser.snapshot().volatiles.get("p1", ())))


class ChargeStateWorldConstructionTest(unittest.TestCase):
    """The engine_world half: the payload volatile becomes an expressed world."""

    def test_the_charge_volatile_is_in_the_allowlist_set(self) -> None:
        from pokezero.engine_world import _CHARGE_VOLATILES

        self.assertEqual(_CHARGE_VOLATILES, frozenset({"solarbeam"}))
        # Same key on both sides of the boundary, so the parser's token maps onto
        # the engine's own charge volatile without a translation table.
        self.assertEqual(_CHARGE_VOLATILES, set(_CHARGE_MOVE_VOLATILES))

    def test_a_charging_side_skips_the_trap_analysis(self) -> None:
        # `active_is_charging_move` already restricts the side to one move, so it is
        # the same hard lock as `lockedmove` beside it in that set.
        import inspect

        from pokezero import engine_world

        source = inspect.getsource(engine_world)
        self.assertIn("_CHARGE_VOLATILES", source)
        marker = '{"trapped", "partiallytrapped", "lockedmove", "mustrecharge"} | _CHARGE_VOLATILES'
        self.assertIn(marker, source, "charge state must join the hard-lock set")

    def test_capability_probe_accepts_the_patched_wheel(self) -> None:
        from pokezero.poke_engine_adapter import require_charge_state_support
        from pokezero.poke_engine_backend import probe_poke_engine

        if not probe_poke_engine().ready:
            self.skipTest("poke-engine is not installed/ready")
        require_charge_state_support()

    def test_capability_probe_fails_closed_on_a_dropping_engine(self) -> None:
        # `PokemonVolatileStatus::from_str` defaults to NONE, so an engine that does
        # not know SOLARBEAM ACCEPTS the token and silently discards it — building
        # the charging Pokemon FREE. Only a round trip tells the two apart.
        from types import SimpleNamespace

        from pokezero.poke_engine_adapter import (
            PokeEngineChargeStateUnsupportedError,
            require_charge_state_support,
        )

        class _DroppingState:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def to_string(self) -> str:
                return "no-volatiles-here"

        engine = SimpleNamespace(
            State=_DroppingState, Side=lambda **kwargs: SimpleNamespace(**kwargs)
        )
        with self.assertRaises(PokeEngineChargeStateUnsupportedError) as caught:
            require_charge_state_support(engine)
        self.assertIn("setup_poke_engine.sh", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
