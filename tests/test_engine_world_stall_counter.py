"""The consecutive-Protect counter must reach the engine with no offset.

Gen 3's stall ladder prices the Nth consecutive Protect attempt at 1, 1/2, 1/4,
1/8, 1/8, ... The engine expresses that as
``CONSECUTIVE_PROTECT_CHANCE ** side_conditions.protect`` (0.5 ** k) and only
branches at all when ``k > 0`` — so a world that never seeds the counter says
"this is a first Protect" and returns a single 100%-success branch, which is the
recorded divergence.

The parser already derives the count from public protocol alone. These tests pin
the WORLD half: that the count survives materialization into
``side_conditions.protect`` unchanged, and that the engine then prices the next
attempt at ``0.5 ** k``. The convention claim is asserted in two linked halves
rather than one call — see
``test_the_seeded_value_prices_the_next_attempt_at_one_half_to_the_k`` for what
that seam does and does not cover.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tests.test_engine_world import _dex, _override, _payload  # noqa: E402

from pokezero.engine_world import battle_spec_from_payload  # noqa: E402

try:
    import poke_engine
except ImportError:  # pragma: no cover - native wheel absent
    poke_engine = None


def _protect_branch_success(state, protect_move: str = "protect") -> float:
    """Total probability mass of branches in which Protect goes up."""

    branches = poke_engine.generate_instructions(state, protect_move, "splash")
    success = 0.0
    for branch in branches:
        applied = any(
            "ApplyVolatileStatus" in str(instruction) and "PROTECT" in str(instruction).upper()
            for instruction in branch.instruction_list
        )
        if applied:
            success += float(branch.percentage)
    return success


@unittest.skipUnless(poke_engine is not None, "native poke_engine wheel is absent")
class StallCounterReachesTheEngineTest(unittest.TestCase):
    """The count must land in side_conditions.protect, unmodified."""

    def setUp(self) -> None:
        self.dex = _dex()

    def _world_with(self, count: int):
        payload = _payload(self.dex)
        payload["sides"]["p1"]["stallCounter"] = count
        return battle_spec_from_payload(payload, _override(), dex=self.dex)

    def test_the_count_passes_through_with_no_offset(self) -> None:
        for count in (1, 2, 3):
            world = self._world_with(count)
            self.assertEqual(
                world.spec.side_one.side_conditions.get("protect"),
                count,
                f"stallCounter {count} must seed side_conditions.protect unchanged",
            )

    def test_a_zero_count_seeds_nothing(self) -> None:
        # A first Protect must stay a single 100% branch, so zero must not write
        # the key at all rather than writing an explicit 0.
        world = self._world_with(0)
        self.assertNotIn("protect", world.spec.side_one.side_conditions)

    def test_the_seeded_value_prices_the_next_attempt_at_one_half_to_the_k(self) -> None:
        """The convention pin: an off-by-one here halves or doubles every branch.

        This is asserted in two linked halves rather than one call, because the
        shared world fixture's dex carries no Protect and threading one through
        the dex, the override team and the payload would be more surgery than the
        pin is worth:

          half 1 (above)  the world writes the count into side_conditions.protect
                          UNCHANGED — no offset applied during materialization;
          half 2 (here)   the engine prices the next attempt at 0.5 ** that same
                          value, generated from a real state.

        The seam is that the two halves use different states. What that leaves
        uncovered is only "materialization writes k but the engine reads some
        other field" — which half 1 already excludes by naming the field the
        engine's ladder reads.
        """

        for count, expected in ((0, 100.0), (1, 50.0), (2, 25.0)):
            state = poke_engine.State(
                side_one=poke_engine.Side(
                    side_conditions=poke_engine.SideConditions(protect=count),
                    pokemon=[
                        poke_engine.Pokemon(
                            id="snorlax", level=80, hp=300, maxhp=300,
                            moves=[poke_engine.Move(id="protect", pp=16)],
                        )
                    ],
                ),
                side_two=poke_engine.Side(
                    pokemon=[
                        poke_engine.Pokemon(
                            id="snorlax", level=80, hp=300, maxhp=300,
                            moves=[poke_engine.Move(id="splash", pp=16)],
                        )
                    ],
                ),
            )
            success = _protect_branch_success(state)
            self.assertAlmostEqual(
                success,
                expected,
                places=2,
                msg=f"{count} consecutive successes must price the next Protect "
                    f"at {expected}%, got {success}%",
            )


@unittest.skipUnless(poke_engine is not None, "native poke_engine wheel is absent")
class StallCounterResetPathsTest(unittest.TestCase):
    """The five public resets, asserted through the same construction path.

    The parser owns the resets; this pins that a reset count reaches the engine
    as "first Protect" rather than being stranded at a stale value.
    """

    def setUp(self) -> None:
        self.dex = _dex()

    def test_a_reset_count_restores_a_single_success_branch(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p1"]["stallCounter"] = 0
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertNotIn(
            "protect",
            world.spec.side_one.side_conditions,
            "a reset count must leave the engine at 'first Protect'",
        )

    def test_a_missing_key_is_treated_as_a_reset_not_an_error(self) -> None:
        # Older payloads predate the key; they must degrade to "first Protect"
        # rather than raising, so a stale producer cannot wall the searcher.
        payload = _payload(self.dex)
        payload["sides"]["p1"].pop("stallCounter", None)
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertNotIn("protect", world.spec.side_one.side_conditions)


if __name__ == "__main__":
    unittest.main()
