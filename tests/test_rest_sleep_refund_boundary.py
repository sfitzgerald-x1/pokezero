"""Pin that a NONZERO rest_sleep_pending_refund survives Python -> engine.

Why this file exists, and why it asserts on a nonzero value specifically.

`rest_sleep_pending_refund` lives on the engine's `Pokemon`. pokezero builds
engine state through the pyo3 binding -- `poke_engine_adapter.py` calls
`engine.State(...)` with `Pokemon` objects -- NOT through `State::deserialize`.
So `impl Into<Pokemon> for PyPokemon` is the production path for getting this
value into search.

That conversion is the one the Rust compiler does not protect. Adding the field
to `PyPokemon` forces `E0063` in `From<Pokemon> for PyPokemon` and in the
`#[new]` signature, so those cannot be left behind. But `Into<Pokemon>` would
have gone on compiling with the literal `0` that the plumbing patch put there,
dropping every refund the harness emits. The wheel builds, every gate goes
green, and the field arrives as 0 -- a silent wrong answer rather than a
failure.

A test asserting `refund == 0` would pass against that bug. Only a nonzero
value distinguishes "the binding carries it" from "the binding hardcodes zero",
which is why every assertion below uses one.

The import is deliberately hard. `engine-fidelity-gates.yml` forbids
try/except ImportError + skipIf: a gate that skips when the wheel is missing is
how a previous era shipped six fixtures that read PASS while asserting nothing.
"""

import unittest

from poke_engine import Pokemon, Side, State

# Not 1. A refund of 1 could be produced by an off-by-one or a bool coerced to
# int; 2 is the smallest value that cannot.
PENDING = 2


def _sleeper(**overrides):
    """A Rest sleeper carrying an unspent refund, as row 2 constructs one."""

    kwargs = {
        "id": "snorlax",
        "status": "sleep",
        "rest_turns": 2,
        "rest_sleep_pending_refund": PENDING,
    }
    kwargs.update(overrides)
    return Pokemon(**kwargs)


def _state_with(pokemon):
    side = Side(pokemon=[pokemon] + [Pokemon() for _ in range(5)])
    return State(side_one=side, side_two=Side())


class RestSleepRefundBoundaryTests(unittest.TestCase):
    def test_constructor_keyword_is_accepted_and_readable(self):
        """The Python-visible half: the kwarg exists and round-trips as itself."""

        self.assertEqual(_sleeper().rest_sleep_pending_refund, PENDING)

    def test_default_is_zero_so_existing_callers_are_unaffected(self):
        """Every caller that predates this field must keep meaning 'nothing pending'."""

        self.assertEqual(Pokemon().rest_sleep_pending_refund, 0)

    def test_nonzero_refund_crosses_into_the_engine(self):
        """The assertion this file exists for.

        `State.to_string()` converts through `Into<Pokemon>` and then serializes
        the real engine struct, so a `refund:` tag in the output can only come
        from the value having crossed the boundary. Against the hardcoded-zero
        bug the tag is omitted entirely and this fails.
        """

        serialized = _state_with(_sleeper()).to_string()

        self.assertIn(
            f"refund:{PENDING}",
            serialized,
            "the refund did not reach the engine Pokemon; Into<Pokemon> is "
            "probably still writing a literal 0",
        )

    def test_zero_refund_emits_no_tag_so_the_field_stays_behaviour_neutral(self):
        """Guards the other direction: no tag unless something is actually pending.

        If a zero refund started emitting `refund:0`, every stored corpus row and
        every state hash would shift while nothing about the battle had changed.
        """

        self.assertNotIn(
            "refund:", _state_with(_sleeper(rest_sleep_pending_refund=0)).to_string()
        )

    def test_nonzero_refund_survives_a_full_state_round_trip(self):
        """Serialize -> deserialize -> serialize must be a fixed point.

        This is what makes the value safe to store: `from_string` reads the tag
        by prefix rather than by position, so it has to survive a trip through
        the wire format unchanged.
        """

        once = _state_with(_sleeper()).to_string()

        self.assertEqual(State.from_string(once).to_string(), once)
        self.assertIn(f"refund:{PENDING}", State.from_string(once).to_string())


if __name__ == "__main__":
    unittest.main()
