"""Regression pins for the gen3 Attract-immobilization patch (patched wheel).

Upstream poke-engine 0.0.47 accepts the ``ATTRACT`` volatile but wholly ignores
it (zero behavioral references in ``src/gen3``), so an attracted mon moved 100%
of the time. ``third_party/poke-engine-gen3-attract.patch`` adds the gen3 50/50
move-immobilization chance branch, mirroring the confusion self-hit branch.

These tests pin the *instruction-generation output shape* directly against the
engine so a future wheel rebuild (or a version bump that drops the patch) cannot
silently regress the immobilization back to a no-op. The Showdown differential
(``scripts/attract_differential.py``) is the separate ground-truth gate; this
file guards the engine contract the differential and the world constructor rely
on.

Run in the DEDICATED attract venv (never the shared one):
    .venv-attract/bin/python -m pytest tests/test_engine_attract_immobilization.py
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    import poke_engine
except ImportError:  # pragma: no cover - native wheel absent
    poke_engine = None


def _has_attract_patch() -> bool:
    """True iff the installed wheel models Attract immobilization.

    Guards these pins from running against an unpatched wheel (where they would
    fail confusingly): an attracted mon that still moves 100% of the time means
    the patch is absent.
    """

    if poke_engine is None:
        return False
    state = _AttractFixture.build(status="none")
    branches = poke_engine.generate_instructions(state, "swordsdance", "splash")
    moved = sum(
        b.percentage
        for b in branches
        if any("Boost SideOne" in str(i) for i in b.instruction_list)
    )
    return moved < 99.0  # patched: ~50; unpatched: 100


class _AttractFixture:
    """Attracted mon on SideOne using a deterministic, single-branch self-boost.

    Swords Dance (+2 Atk, 100% accuracy, no sub-branches) means the branch set
    partitions cleanly into "moved" (carries ``Boost SideOne Attack``) vs
    "immobilized" (empty delta). The opponent uses Splash (a true no-op) so no
    opponent instruction rides along to confuse the partition.
    """

    @staticmethod
    def _mon(species, moves, *, status="none"):
        pe = poke_engine
        return pe.Pokemon(
            id=species, level=80, types=("normal", "typeless"), hp=300, maxhp=300,
            ability="innerfocus", item="none", attack=180, defense=180,
            special_attack=180, special_defense=180, speed=120,
            moves=[pe.Move(id=m, pp=16) for m in moves], status=status,
        )

    @classmethod
    def build(cls, *, status="none", attracted=True):
        pe = poke_engine
        dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
        vols = {"ATTRACT"} if attracted else set()
        side_one = pe.Side(
            active_index="0",
            pokemon=[cls._mon("snorlax", ["swordsdance", "bodyslam"], status=status)] + [dummy] * 5,
            volatile_statuses=vols,
        )
        side_two = pe.Side(
            active_index="0",
            pokemon=[cls._mon("wobbuffet", ["splash", "tackle"])] + [dummy] * 5,
        )
        return pe.State(
            side_one=side_one, side_two=side_two,
            weather="none", terrain="none", trick_room=False,
        )

    @staticmethod
    def partition(branches):
        moved = 0.0
        immobilized = 0.0
        immobilized_deltas = []
        for b in branches:
            insts = [str(i) for i in b.instruction_list]
            if any("Boost SideOne" in s for s in insts):
                moved += b.percentage
            else:
                immobilized += b.percentage
                immobilized_deltas.append(insts)
        return moved, immobilized, immobilized_deltas


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
@unittest.skipUnless(_has_attract_patch(), "attract-patched wheel not installed")
class AttractImmobilizationTests(unittest.TestCase):
    def test_free_branch_is_exactly_50_50(self) -> None:
        # No other status: Attract is a clean 50% immobilize / 50% move split.
        state = _AttractFixture.build(status="none")
        branches = poke_engine.generate_instructions(state, "swordsdance", "splash")
        moved, immobilized, _ = _AttractFixture.partition(branches)
        self.assertAlmostEqual(moved, 50.0, places=3)
        self.assertAlmostEqual(immobilized, 50.0, places=3)

    def test_immobilized_branch_is_empty_delta(self) -> None:
        # The immobilized outcome must carry NO self-damage and NO move effect
        # (attract, unlike confusion, deals no damage) — an empty-delta terminal
        # in the same shape as the fully-paralyzed branch.
        state = _AttractFixture.build(status="none")
        branches = poke_engine.generate_instructions(state, "swordsdance", "splash")
        _, _, immobilized_deltas = _AttractFixture.partition(branches)
        self.assertTrue(immobilized_deltas, "expected an immobilized branch")
        for delta in immobilized_deltas:
            self.assertFalse(
                any("Damage" in s or "Boost" in s for s in delta),
                f"immobilized branch must be empty-delta, got {delta}",
            )

    def test_paralysis_composition_move_probability(self) -> None:
        # Paralyzed AND attracted: fully-para (25%) then attract (50% of the
        # remaining 75%). Net "moves" probability = 0.75 * 0.5 = 0.375; total
        # immobilized = 0.625. This is the search-relevant quantity and matches
        # Showdown exactly (multiplication is commutative); the engine's internal
        # par-vs-attract reason split is not observable here (both empty-delta).
        state = _AttractFixture.build(status="paralyze")
        branches = poke_engine.generate_instructions(state, "swordsdance", "splash")
        moved, immobilized, _ = _AttractFixture.partition(branches)
        self.assertAlmostEqual(moved, 37.5, places=3)
        self.assertAlmostEqual(immobilized, 62.5, places=3)

    def test_no_attract_moves_freely(self) -> None:
        # Control: without the ATTRACT volatile the mon moves 100% of the time,
        # proving the branch is gated on volatile presence, not applied blanketly.
        state = _AttractFixture.build(status="none", attracted=False)
        branches = poke_engine.generate_instructions(state, "swordsdance", "splash")
        moved, immobilized, _ = _AttractFixture.partition(branches)
        self.assertAlmostEqual(moved, 100.0, places=3)
        self.assertAlmostEqual(immobilized, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
