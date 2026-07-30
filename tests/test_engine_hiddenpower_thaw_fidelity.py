"""Regression pins for the gen3 typed-Hidden-Power thaw exclusion (patch 44).

Requires the patched wheel (``scripts/setup_poke_engine.sh``).

``poke-engine-gen3-typed-hiddenpower-thaw.patch``
    The thaw-on-fire-hit exclusion listed only the generic
    ``Choices::HIDDENPOWER`` (and WEATHERBALL); pokezero's typed variants
    (``HIDDENPOWERFIRE70`` etc.) slipped through, so a typed HP Fire hit
    thawed a frozen defender unconditionally and let it act (the
    certification sweep's 4-row family, e.g. 2000131/47: ChangeStatus
    FREEZE -> NONE at 100% of branches, Omastar attacking through a turn the
    sim spent frozen). The exclusion now enumerates all 33 Hidden Power
    variants — the same set as the Counter/Mirror Coat precedent.

Sim probes (live gen3customgame, hp_thaw_sim_probes.py): a generic Hidden
Power with HP-Fire IVs is supereffective on Skarmory (dynamically Fire) yet a
frozen Shuckle hit by it KEEPS frz and answers ``|cant|`` (seeds 14/37); the
20% self-thaw arm shows at the defender's own action (seed 40); Flamethrower
cures frz immediately after its damage line and the target acts the same turn
(the sim's runtime Hidden Power id is always the generic ``hiddenpower``,
dex type Normal in gen3's frz.onDamagingHit — the typed ids exist only in
the engine-side vocabulary, which is why the engine must exclude them all).
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


def _one_on_one(p1_kwargs, p2_kwargs):
    pe = poke_engine
    dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
    side_one = pe.Side(active_index="0", pokemon=[pe.Pokemon(**p1_kwargs)] + [dummy] * 5)
    side_two = pe.Side(active_index="0", pokemon=[pe.Pokemon(**p2_kwargs)] + [dummy] * 5)
    return pe.State(side_one=side_one, side_two=side_two,
                    weather="none", terrain="none", trick_room=False)


def _branches(state, m1, m2):
    return [
        (round(b.percentage, 2), [str(i) for i in b.instruction_list])
        for b in poke_engine.generate_instructions(state, m1, m2)
    ]


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class TypedHiddenPowerThawTests(unittest.TestCase):
    """The sweep's 4-row family shape: fast attacker, frozen slower defender."""

    def _state(self, attacker_moves):
        pe = poke_engine
        espeon = dict(id="espeon", level=80, types=("psychic", "typeless"), hp=230,
                      maxhp=230, ability="synchronize", item="leftovers", attack=120,
                      defense=130, special_attack=260, special_defense=190, speed=240,
                      moves=[pe.Move(id=m, pp=16) for m in attacker_moves])
        omastar = dict(id="omastar", level=84, types=("rock", "water"), hp=255,
                       maxhp=255, ability="shellarmor", item="leftovers", attack=107,
                       defense=258, special_attack=241, special_defense=166, speed=141,
                       status="freeze",
                       moves=[pe.Move(id="tackle", pp=16)])
        return _one_on_one(espeon, omastar)

    def _frozen_arm_weight(self, branches):
        """Total weight of branches where the defender stays frozen: no
        FREEZE -> NONE and no defender action (SideOne takes no move damage)."""
        weight = 0.0
        for pct, instrs in branches:
            thawed = any("FREEZE -> NONE" in i for i in instrs)
            defender_acted = any(i.startswith("Damage SideOne:") for i in instrs)
            if not thawed and not defender_acted:
                weight += pct
        return weight

    def test_typed_hp_fire_70_does_not_thaw(self) -> None:
        """Divergence pin (fails on the 43-patch wheel with the unpatched
        shape: thaw at 100%, frozen arm weight 0): the sim keeps the target
        frozen, so 80% of the engine's weight must be a still-frozen |cant|
        arm; only the 20% self-thaw arm may act."""
        branches = _branches(self._state(["hiddenpowerfire70"]),
                             "hiddenpowerfire70", "tackle")
        frozen = self._frozen_arm_weight(branches)
        self.assertAlmostEqual(frozen, 80.0, delta=1.0,
                               msg=f"still-frozen arm should carry ~80%: {branches}")

    def test_typed_hp_fire_60_does_not_thaw(self) -> None:
        """Divergence pin (fails unpatched): the 60-BP tier is a distinct
        enum variant and must be excluded too."""
        branches = _branches(self._state(["hiddenpowerfire60"]),
                             "hiddenpowerfire60", "tackle")
        frozen = self._frozen_arm_weight(branches)
        self.assertAlmostEqual(frozen, 80.0, delta=1.0,
                               msg=f"still-frozen arm should carry ~80%: {branches}")

    def test_self_thaw_arm_survives(self) -> None:
        """Divergence pin (fails unpatched, where thaw is 100% but carries the
        fire-hit's unconditional ChangeStatus): the 20% arm is the engine's
        80/20 self-thaw gate — present, weighted 20, and the defender acts."""
        branches = _branches(self._state(["hiddenpowerfire70"]),
                             "hiddenpowerfire70", "tackle")
        thaw_weight = sum(
            pct for pct, instrs in branches
            if any("FREEZE -> NONE" in i for i in instrs)
            and any(i.startswith("Damage SideOne:") for i in instrs))
        self.assertAlmostEqual(thaw_weight, 20.0, delta=1.0,
                               msg=f"self-thaw arm should carry ~20%: {branches}")

    def test_flamethrower_still_thaws(self) -> None:
        """Control (both builds): a real Fire move thaws on the hit — every
        branch carries FREEZE -> NONE and the thawed defender acts."""
        branches = _branches(self._state(["flamethrower"]),
                             "flamethrower", "tackle")
        self.assertTrue(branches)
        for pct, instrs in branches:
            self.assertTrue(any("FREEZE -> NONE" in i for i in instrs),
                            f"branch without thaw: {pct} {instrs}")
        self.assertAlmostEqual(
            sum(pct for pct, instrs in branches
                if any(i.startswith("Damage SideOne:") for i in instrs)),
            100.0, delta=1.0,
            msg=f"thawed defender should act in every branch: {branches}")

    def test_generic_hiddenpower_still_excluded(self) -> None:
        """Control (both builds): the generic id was already excluded and must
        keep the 80/20 shape."""
        branches = _branches(self._state(["hiddenpower"]),
                             "hiddenpower", "tackle")
        frozen = self._frozen_arm_weight(branches)
        self.assertAlmostEqual(frozen, 80.0, delta=1.0,
                               msg=f"still-frozen arm should carry ~80%: {branches}")


if __name__ == "__main__":
    unittest.main()
