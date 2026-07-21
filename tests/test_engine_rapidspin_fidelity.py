"""Regression pins for the gen3 Rapid Spin / Protect fidelity patch (patched wheel).

Upstream poke-engine 0.0.47 detects Protect and calls ``remove_effects_for_protect()``,
which zeroes base_power/category and clears the declarative effect fields but leaves
``move_id`` intact. The post-hit dispatchers ``choice_hazard_clear`` and
``choice_special_effect`` are keyed on ``move_id`` and run in the same path, so a
Protect-blocked move still fired: a blocked Rapid Spin stripped the spinner's own
Spikes, and Seismic Toss / Super Fang dealt their fixed damage THROUGH Protect.
Separately, a *connecting* Rapid Spin never cleared the user's Leech Seed or freed
it from partial-trapping (both entirely unmodelled).

``third_party/poke-engine-gen3-rapidspin-fidelity.patch`` threads a
``blocked_by_protect`` bool from ``before_move`` into those dispatchers (early-return
when blocked, gated on Protect specifically — NOT on damage/hit_sub, so a spin that
connects on a Substitute STILL clears), and adds the Leech Seed + partial-trap clears
to the Rapid Spin arm.

These tests pin the *instruction-generation output shape* directly against the engine
so a future wheel rebuild (or a version bump that drops the patch) cannot silently
regress. The Showdown differential (``scripts/rapidspin_differential.py``) is the
separate ground-truth gate; this file guards the engine contract.

Run in the DEDICATED rapidspin venv (never the shared one):
    .venv-rapidspin/bin/python -m pytest tests/test_engine_rapidspin_fidelity.py
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


class _RapidSpinFixture:
    """Rapid Spin on SideOne, with SideOne hazards/volatiles and a SideTwo defender.

    Rapid Spin's spin effect (clearing the user's own entry hazards / Leech Seed /
    partial-trap) is emitted as ``ChangeSideCondition SideOne`` /
    ``RemoveVolatileStatus SideOne`` instructions on every branch when it fires, so
    each pin just asks whether the corresponding instruction is present.
    """

    @staticmethod
    def _mon(species, moves, *, types=("normal", "typeless")):
        pe = poke_engine
        return pe.Pokemon(
            id=species, level=80, types=types, hp=250, maxhp=300, ability="none",
            item="none", attack=150, defense=150, special_attack=150,
            special_defense=150, speed=100,
            moves=[pe.Move(id=m, pp=16) for m in moves],
        )

    @classmethod
    def build(cls, *, defender_vols=frozenset(), attacker_vols=frozenset(), spikes=2,
              defender_types=("normal", "typeless"), sub_health=0,
              attacker_move="rapidspin"):
        pe = poke_engine
        dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
        side_one = pe.Side(
            active_index="0",
            pokemon=[cls._mon("attacker", [attacker_move, "tackle"])] + [dummy] * 5,
            volatile_statuses=set(attacker_vols),
            side_conditions=pe.SideConditions(spikes=spikes),
        )
        side_two = pe.Side(
            active_index="0",
            pokemon=[cls._mon("defender", ["splash", "tackle"], types=defender_types)] + [dummy] * 5,
            volatile_statuses=set(defender_vols),
            substitute_health=sub_health,
        )
        return pe.State(side_one=side_one, side_two=side_two,
                        weather="none", terrain="none", trick_room=False)

    @staticmethod
    def instructions(state, attacker_move="rapidspin"):
        branches = poke_engine.generate_instructions(state, attacker_move, "splash")
        return [str(i) for b in branches for i in b.instruction_list]

    @staticmethod
    def spikes_cleared(insts):
        return any("ChangeSideCondition SideOne Spikes" in s for s in insts)

    @staticmethod
    def leech_cleared(insts):
        return any("RemoveVolatileStatus SideOne: LEECHSEED" in s for s in insts)

    @staticmethod
    def trap_cleared(insts):
        return any("RemoveVolatileStatus SideOne: PARTIALLYTRAPPED" in s for s in insts)

    @staticmethod
    def defender_damaged(insts):
        return any(("Damage SideTwo" in s or "DamageSubstitute SideTwo" in s) for s in insts)


def _has_rapidspin_patch() -> bool:
    """True iff the installed wheel guards the Protect-blocked hazard clear.

    On an unpatched wheel a Protect-blocked Rapid Spin still strips the user's own
    Spikes (the bug); on the patched wheel the hazards stay. Guards these pins from
    running (and failing confusingly) against an unpatched wheel.
    """

    if poke_engine is None:
        return False
    state = _RapidSpinFixture.build(defender_vols={"PROTECT"}, spikes=2)
    insts = _RapidSpinFixture.instructions(state)
    return not _RapidSpinFixture.spikes_cleared(insts)  # patched: hazards stay


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
@unittest.skipUnless(_has_rapidspin_patch(), "rapidspin-patched wheel not installed")
class RapidSpinFidelityTests(unittest.TestCase):
    F = _RapidSpinFixture

    def test_spin_into_protect_keeps_spikes(self) -> None:
        # THE bug: a Protect-blocked Rapid Spin must NOT strip the user's Spikes.
        state = self.F.build(defender_vols={"PROTECT"}, spikes=2)
        insts = self.F.instructions(state)
        self.assertFalse(self.F.spikes_cleared(insts),
                         f"Spikes must survive a Protect-blocked spin; got {insts}")

    def test_spin_into_substitute_clears_spikes(self) -> None:
        # The guard is on Protect, NOT on hit_sub: a spin that connects on a
        # Substitute STILL clears the user's hazards (Showdown-confirmed).
        state = self.F.build(defender_vols={"SUBSTITUTE"}, spikes=2, sub_health=80)
        insts = self.F.instructions(state)
        self.assertTrue(self.F.spikes_cleared(insts),
                        f"spin into a Substitute must still clear Spikes; got {insts}")

    def test_spin_connecting_clears_spikes_leech_and_trap(self) -> None:
        # A connecting spin clears the user's Spikes AND ends Leech Seed AND frees
        # the user from partial-trapping (the last two are the newly-modelled part).
        state = self.F.build(attacker_vols={"LEECHSEED", "PARTIALLYTRAPPED"}, spikes=2)
        insts = self.F.instructions(state)
        self.assertTrue(self.F.spikes_cleared(insts), f"expected Spikes clear; {insts}")
        self.assertTrue(self.F.leech_cleared(insts), f"expected Leech Seed clear; {insts}")
        self.assertTrue(self.F.trap_cleared(insts), f"expected partial-trap clear; {insts}")

    def test_spin_into_ghost_keeps_spikes(self) -> None:
        # Regression guard: Normal-type Rapid Spin is immune vs Ghost, so it never
        # spins (this path is correct in upstream via type immunity, not the patch).
        state = self.F.build(defender_types=("ghost", "poison"), spikes=2)
        insts = self.F.instructions(state)
        self.assertFalse(self.F.spikes_cleared(insts),
                         f"spin vs a Ghost must not clear Spikes; got {insts}")

    def test_spin_into_protect_leaves_no_leech_or_trap_clear(self) -> None:
        # A Protect-blocked spin must clear NOTHING, including Leech Seed / trap.
        state = self.F.build(defender_vols={"PROTECT"},
                             attacker_vols={"LEECHSEED", "PARTIALLYTRAPPED"}, spikes=2)
        insts = self.F.instructions(state)
        self.assertFalse(self.F.spikes_cleared(insts), insts)
        self.assertFalse(self.F.leech_cleared(insts), insts)
        self.assertFalse(self.F.trap_cleared(insts), insts)

    def test_seismictoss_into_protect_deals_no_damage(self) -> None:
        # Sibling sweep: Seismic Toss's fixed damage is keyed in choice_special_effect
        # (move_id survives the Protect strip) and must NOT fire through Protect.
        state = self.F.build(defender_vols={"PROTECT"}, spikes=0, attacker_move="seismictoss")
        insts = self.F.instructions(state, attacker_move="seismictoss")
        self.assertFalse(self.F.defender_damaged(insts),
                         f"Seismic Toss must be fully blocked by Protect; got {insts}")

    def test_superfang_into_protect_deals_no_damage(self) -> None:
        # Sibling sweep: Super Fang (half-HP) is likewise keyed in choice_special_effect.
        state = self.F.build(defender_vols={"PROTECT"}, spikes=0, attacker_move="superfang")
        insts = self.F.instructions(state, attacker_move="superfang")
        self.assertFalse(self.F.defender_damaged(insts),
                         f"Super Fang must be fully blocked by Protect; got {insts}")

    def test_no_protect_seismictoss_still_deals_damage(self) -> None:
        # Control: without Protect the sibling special effect fires normally, proving
        # the guard is gated on Protect and not a blanket disable.
        state = self.F.build(spikes=0, attacker_move="seismictoss")
        insts = self.F.instructions(state, attacker_move="seismictoss")
        self.assertTrue(self.F.defender_damaged(insts),
                        f"Seismic Toss must deal damage when not blocked; got {insts}")


if __name__ == "__main__":
    unittest.main()
