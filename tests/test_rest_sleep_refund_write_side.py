"""Row 2 step 4: the harness write side of the Rest skippedTime refund.

Era 58 measured `rest_sleep_active_refund_pending` at 781 killed decisions,
28.7% of all MCTS fallback and the largest exactly-counted class. It refused
because an ACTIVE Rest sleeper's skipped Sleep Talk/Snore turns had nowhere to
live: the value was known, and the engine had no field for a refund that only a
future switch-in applies. #1105 added the field, #1108 exposed it on the pyo3
binding, #1109 made the engine spend it. This is the half that supplies it.

Two failure modes are specifically pinned here because both are SILENT:

1. Double counting. `_rest_turns_from_row` folds the refund into `rest_turns`,
   which is correct for a BENCHED mon -- a switch-in necessarily precedes its
   next attempt -- and wrong for an active one, which may never switch while
   search explores the stay-in branch. Writing the bank without removing the
   fold credits the same turns twice and the engine clamps the result, so the
   mon simply sleeps too long and nothing errors.

2. A binding that drops the field. Adding it to `PyPokemon` forces `E0063` in
   `From<Pokemon>` and in `#[new]`, but `impl Into<Pokemon>` compiles fine while
   writing a literal 0 -- and that is the direction production uses. The wheel
   builds, every gate is green, and the refund is gone.

The import is deliberately hard: a skip here would assert nothing.
"""

import unittest

import poke_engine

from pokezero.poke_engine_adapter import (
    MoveSpec,
    PokemonSpec,
    PokeEngineRestSleepRefundUnsupportedError,
    _build_pokemon,
    _rest_sleep_refund_supported,
    require_rest_sleep_refund_support,
)

# Gen 3 Rest sets a 3-turn counter and the engine's wake match panics above it.
LEGAL_MAX = 3


def _sleeper(rest_turns: int, pending: int) -> PokemonSpec:
    return PokemonSpec(
        id="snorlax",
        level=50,
        types=("normal",),
        hp=100,
        maxhp=100,
        attack=100,
        defense=100,
        special_attack=100,
        special_defense=100,
        speed=100,
        moves=(MoveSpec(id="rest"),),
        status="sleep",
        rest_turns=rest_turns,
        rest_sleep_pending_refund=pending,
    )


def _dropping_pokemon(**kwargs):
    """A binding that accepts the keyword and silently discards it.

    MUST return a real ``poke_engine.Pokemon``. The first version of this control
    was a Python wrapper class, and it was VACUOUS: pyo3 rejects a foreign object
    at ``Side(pokemon=[...])`` with ``TypeError: ... cannot be converted to
    'Pokemon'``, the probe's blanket ``except Exception`` turned that into
    ``False``, and the test passed for a reason unrelated to dropping the field.
    Proof it asserted nothing: deleting the ``pop`` below, or deleting the
    difference check inside the probe, both left the whole suite green.

    ``poke_engine.Pokemon`` is not subclassable, so a factory is the only faithful
    way to model ``impl Into<Pokemon>`` writing a literal 0.
    """

    kwargs.pop("rest_sleep_pending_refund", None)
    return poke_engine.Pokemon(**kwargs)


class _DroppingEngine:
    State = poke_engine.State
    Side = poke_engine.Side
    Pokemon = staticmethod(_dropping_pokemon)


class RefundReachesTheEngineTests(unittest.TestCase):
    def test_a_legal_split_survives_into_the_engine_pokemon(self):
        """rest_turns 2 + bank 1 is the shape a one-attempt Sleep Talk row produces."""

        built = _build_pokemon(poke_engine, _sleeper(2, 1), "side_one.pokemon[0]")

        self.assertEqual(built.rest_turns, 2)
        self.assertEqual(built.rest_sleep_pending_refund, 1)

    def test_an_unrepresentable_sum_fails_closed_rather_than_clamping(self):
        """The engine clamps because it has no refusal channel; the adapter has one.

        `_rest_turns_from_row`'s own comment calls clamping here "exactly the
        silent wrongness the constructor fails closed on", so a sum above the
        legal maximum has to raise rather than quietly truncate into a plausible
        counter.
        """

        with self.assertRaises(ValueError) as caught:
            _build_pokemon(poke_engine, _sleeper(3, 2), "side_one.pokemon[0]")

        self.assertIn("rest_sleep_pending_refund", str(caught.exception))
        self.assertIn("5", str(caught.exception))

    def test_the_boundary_sum_is_allowed(self):
        """Exactly at the maximum must build: the guard is > , not >=."""

        built = _build_pokemon(poke_engine, _sleeper(1, 2), "side_one.pokemon[0]")

        self.assertEqual(built.rest_turns + built.rest_sleep_pending_refund, LEGAL_MAX)


class CapabilityProbeTests(unittest.TestCase):
    def test_the_installed_binding_round_trips_the_field(self):
        self.assertTrue(_rest_sleep_refund_supported(poke_engine))
        require_rest_sleep_refund_support(poke_engine)

    def test_a_binding_that_drops_the_field_is_rejected(self):
        """The negative control, and the reason this probe exists at all.

        Without this the probe could be asserting nothing -- exactly the shape
        that let #1105's wheel break reach CI. A dropping binding accepts the
        keyword and builds a state, so only a round-trip comparison can tell it
        from a working one.
        """

        self.assertFalse(_rest_sleep_refund_supported(_DroppingEngine))
        with self.assertRaises(PokeEngineRestSleepRefundUnsupportedError):
            require_rest_sleep_refund_support(_DroppingEngine)


if __name__ == "__main__":
    unittest.main()
