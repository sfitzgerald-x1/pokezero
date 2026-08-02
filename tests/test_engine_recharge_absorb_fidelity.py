"""Regression pins for the gen3 recharge-turn-residuals and absorb-gate patches.

Requires the patched wheel (``scripts/setup_poke_engine.sh``). Two patches:

``poke-engine-gen3-recharge-turn-residuals.patch``
    end_of_turn_triggered's bare (Switch, None) exclusion — correct for
    faint-replacement plies — also swallowed voluntary switches beside a
    recharging Pokemon, dropping every end-of-turn residual on
    ``|cant|<mon>|recharge`` boundaries (the certification sweep's 56-row
    family). Sim probe (Snorlax Hyper Beam then recharge while toxic'd,
    opponent switches, seeds 1/3/5): the recharge turn runs the full block —
    ``-heal ... Leftovers`` then the toxic tick (387 -> 339, a 48 = 2/16 tick).

``poke-engine-gen3-absorb-protect-accuracy.patch``
    The absorb-conversion arms called remove_all_effects(), which clears
    flags.protect and forces accuracy to 100, so the quarter heal pierced
    Protect and healed on missed moves. Sim probes: a protecting Vaporeon
    answers ``-activate Protect`` with NO absorb; Hydro Pump (80) at a Water
    Absorb Politoed misses on seeds 4/10 of 12 with NO heal, and answers
    ``-immune`` on hits.

Audit note (family-scope correction): Truant-loaf boundaries were probed on
both (loaf, move) and (loaf, switch) shapes and the engine DOES run the
end-of-turn block there — the cant-source EOT gap is recharge-specific. The
loaf-shape control below pins that so the recharge fix cannot regress it.
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


def _one_on_one(p1_kwargs, p2_party, *, p1_side=None):
    pe = poke_engine
    dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
    side_one = pe.Side(active_index="0", pokemon=[pe.Pokemon(**p1_kwargs)] + [dummy] * 5,
                       **(p1_side or {}))
    side_two = pe.Side(active_index="0",
                       pokemon=p2_party + [dummy] * (6 - len(p2_party)))
    return pe.State(side_one=side_one, side_two=side_two,
                    weather="none", terrain="none", trick_room=False)


def _mon(**kwargs):
    base = dict(item="none", status="none")
    base.update(kwargs)
    return poke_engine.Pokemon(**base) if kwargs.get("_direct") else base


def _branches(state, m1, m2):
    return [
        (round(b.percentage, 2), [str(i) for i in b.instruction_list])
        for b in poke_engine.generate_instructions(state, m1, m2)
    ]


def _flat(branches):
    return [i for _, instrs in branches for i in instrs]


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class RechargeTurnResidualTests(unittest.TestCase):
    def _recharge_state(self):
        pe = poke_engine
        snorlax = dict(id="snorlax", level=80, types=("normal", "typeless"), hp=339,
                       maxhp=387, ability="thickfat", item="leftovers", attack=222,
                       defense=155, special_attack=155, special_defense=222, speed=85,
                       status="toxic",
                       moves=[pe.Move(id="hyperbeam", pp=8)])
        milotic = pe.Pokemon(id="milotic", level=80, types=("water", "typeless"), hp=250,
                             maxhp=283, ability="marvelscale", item="leftovers", attack=150,
                             defense=170, special_attack=200, special_defense=240, speed=170,
                             moves=[poke_engine.Move(id="splash", pp=16)])
        blissey = pe.Pokemon(id="blissey", level=80, types=("normal", "typeless"), hp=500,
                             maxhp=539, ability="naturalcure", item="leftovers", attack=60,
                             defense=77, special_attack=200, special_defense=250, speed=120,
                             moves=[poke_engine.Move(id="splash", pp=16)])
        return _one_on_one(snorlax, [milotic, blissey],
                           p1_side={"volatile_statuses": {"mustrecharge"}})

    def test_voluntary_switch_beside_recharge_runs_residuals(self) -> None:
        """The 56-row sweep family's exact shape (e.g. 2400315/32: sandstorm
        tick observed-only at 100%). Sim probe: the toxic'd recharger takes
        its Leftovers heal and toxic tick on the (switch, recharge) turn.
        Patched: the branch carries MUSTRECHARGE removal AND the p1 residuals
        (Heal 48 leftovers cap? here: heal 24 = 387/16, toxic tick 48 = 2*387/16
        on toxic stage advance). Unpatched: the branch is the bare switch +
        MUSTRECHARGE removal with NO residuals."""
        branches = _branches(self._recharge_state(), "none", "blissey")
        flat = _flat(branches)
        self.assertTrue(any("RemoveVolatileStatus SideOne: MUSTRECHARGE" in i for i in flat))
        self.assertTrue(any(i.startswith("Heal SideOne:") for i in flat),
                        f"leftovers residual missing: {branches}")
        self.assertTrue(any(i.startswith("Damage SideOne:") for i in flat),
                        f"toxic residual missing: {branches}")

    def test_recharge_beside_move_still_runs_residuals(self) -> None:
        """Control (both builds): a (move, recharge) turn never hit the
        (Switch, None) exclusion; residuals run."""
        branches = _branches(self._recharge_state(), "none", "splash")
        flat = _flat(branches)
        self.assertTrue(any(i.startswith("Heal SideOne:") for i in flat))
        self.assertTrue(any(i.startswith("Damage SideOne:") for i in flat))

    def test_hyper_beam_ko_replacement_defers_recharge_until_next_move_turn(self) -> None:
        """A forced replacement is not the recharger's onBeforeMove hook.

        The old engine consumed MUSTRECHARGE on the opponent's compulsory
        switch. Showdown keeps it through that boundary, then removes it when
        the recharger actually takes its next ordinary turn.
        """
        pe = poke_engine
        snorlax = dict(id="snorlax", level=80, types=("normal", "typeless"), hp=387,
                       maxhp=387, ability="thickfat", item="none", attack=222,
                       defense=155, special_attack=260, special_defense=222, speed=170,
                       moves=[pe.Move(id="hyperbeam", pp=8)])
        fainted = pe.Pokemon(id="pikachu", level=5, types=("electric", "typeless"),
                             hp=1, maxhp=20, ability="static", item="none", attack=10,
                             defense=10, special_attack=10, special_defense=10, speed=10,
                             moves=[pe.Move(id="splash", pp=16)])
        replacement = pe.Pokemon(id="blissey", level=80, types=("normal", "typeless"),
                                 hp=500, maxhp=539, ability="naturalcure", item="none",
                                 attack=60, defense=77, special_attack=200,
                                 special_defense=250, speed=120,
                                 moves=[pe.Move(id="splash", pp=16)])
        state = _one_on_one(snorlax, [fainted, replacement])

        ko_branch = next(
            branch for branch in pe.generate_instructions(state, "hyperbeam", "none")
            if any("ToggleSideTwoForceSwitch" in str(i) for i in branch.instruction_list)
        )
        after_ko = state.apply_instructions(ko_branch)
        self.assertTrue(after_ko.side_two.force_switch)
        self.assertIn("MUSTRECHARGE", after_ko.side_one.volatile_statuses)

        replacement_branch = pe.generate_instructions(after_ko, "none", "blissey")[0]
        replacement_text = [str(i) for i in replacement_branch.instruction_list]
        self.assertTrue(any(i.startswith("Switch SideTwo:") for i in replacement_text))
        self.assertFalse(any("RemoveVolatileStatus SideOne: MUSTRECHARGE" in i
                             for i in replacement_text), replacement_text)

        after_replacement = after_ko.apply_instructions(replacement_branch)
        self.assertIn("MUSTRECHARGE", after_replacement.side_one.volatile_statuses)
        next_turn = _flat(_branches(after_replacement, "none", "splash"))
        self.assertTrue(any("RemoveVolatileStatus SideOne: MUSTRECHARGE" in i
                            for i in next_turn), next_turn)

    def test_truant_loaf_boundaries_keep_their_residuals(self) -> None:
        """Audit control (both builds): the loaf early-return does NOT drop
        end-of-turn residuals on either (loaf, move) or (loaf, switch) shapes
        — probed against the engine during the family-scope audit. Pins the
        loaf+switch shape with sandstorm active."""
        pe = poke_engine
        dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
        slaking = pe.Pokemon(id="slaking", level=78, types=("normal", "typeless"), hp=274,
                             maxhp=362, ability="truant", item="leftovers", attack=295,
                             defense=201, special_attack=193, special_defense=146, speed=201,
                             moves=[pe.Move(id="earthquake", pp=16)])
        tyranitar = pe.Pokemon(id="tyranitar", level=74, types=("rock", "dark"), hp=156,
                               maxhp=270, ability="sandstream", item="leftovers", attack=200,
                               defense=180, special_attack=170, special_defense=170, speed=110,
                               status="toxic", moves=[pe.Move(id="splash", pp=16)])
        milotic = pe.Pokemon(id="milotic", level=80, types=("water", "typeless"), hp=250,
                             maxhp=283, ability="marvelscale", item="leftovers", attack=150,
                             defense=170, special_attack=200, special_defense=240, speed=170,
                             moves=[pe.Move(id="splash", pp=16)])
        state = pe.State(
            side_one=pe.Side(active_index="0", pokemon=[slaking] + [dummy] * 5,
                             volatile_statuses={"truant"}),
            side_two=pe.Side(active_index="0", pokemon=[tyranitar, milotic] + [dummy] * 4),
            weather="sand", terrain="none", trick_room=False,
        )
        flat = _flat(_branches(state, "earthquake", "milotic"))
        self.assertTrue(any("RemoveVolatileStatus SideOne: TRUANT" in i for i in flat))
        self.assertTrue(any(i.startswith("Damage SideOne:") for i in flat),
                        "loaf turn lost its sandstorm tick")


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class AbsorbGateTests(unittest.TestCase):
    def _absorb_state(self, attacker_move, defender_vols=None, defender_hp=200):
        pe = poke_engine
        seaking = dict(id="seaking", level=80, types=("water", "typeless"), hp=259,
                       maxhp=259, ability="swiftswim", item="none", attack=193,
                       defense=150, special_attack=150, special_defense=174, speed=155,
                       moves=[pe.Move(id=attacker_move, pp=16)])
        vaporeon = pe.Pokemon(id="vaporeon", level=80, types=("water", "typeless"),
                              hp=defender_hp, maxhp=371, ability="waterabsorb",
                              item="leftovers", attack=140, defense=130,
                              special_attack=230, special_defense=200, speed=140,
                              moves=[pe.Move(id="protect", pp=16)])
        dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
        return pe.State(
            side_one=pe.Side(active_index="0", pokemon=[pe.Pokemon(**seaking)] + [dummy] * 5),
            side_two=pe.Side(active_index="0", pokemon=[vaporeon] + [dummy] * 5,
                             volatile_statuses=defender_vols or set()),
            weather="none", terrain="none", trick_room=False,
        )

    def test_protect_blocks_the_absorb_heal(self) -> None:
        """Sim probe (Vaporeon protects vs Surf, seeds 1-3): -activate Protect,
        no absorb heal or immune line. Unpatched: the engine healed maxhp/4
        through Protect (the sweep's 23 protect-shape rows, e.g. 2000214/12:
        engine_only=[('abilitywaterabsorb', 83)])."""
        state = self._absorb_state("surf", defender_vols={"protect"})
        flat = _flat(_branches(state, "surf", "none"))
        self.assertFalse(any(i.startswith("Heal SideTwo:") and ": 92" in i for i in flat),
                         f"absorb quarter-heal pierced Protect: {flat}")
        # 371/4 = 92: no heal of that size; leftovers (23) may still tick at EOT.

    def test_inaccurate_absorb_move_keeps_miss_and_protect_branches(self) -> None:
        state = self._absorb_state("hydropump", defender_vols={"protect"})
        branches = _branches(state, "hydropump", "none")
        marker_pcts = [
            pct
            for pct, instrs in branches
            if any(i.startswith("Heal SideTwo: 0") for i in instrs)
        ]
        miss_pcts = [
            pct
            for pct, instrs in branches
            if not any(i.startswith("Heal SideTwo: 0") for i in instrs)
        ]
        self.assertTrue(marker_pcts and abs(sum(marker_pcts) - 80.0) < 0.6, branches)
        self.assertTrue(miss_pcts and abs(sum(miss_pcts) - 20.0) < 0.6, branches)

    def test_missed_water_move_does_not_heal(self) -> None:
        """Sim probe (Hydro Pump 80%% at Water Absorb Politoed, 12 seeds):
        misses on 2 seeds show -miss and NO heal. Patched: the converted heal
        rides the move's real accuracy — an 80%% heal branch and a 20%% no-heal
        branch. Unpatched: accuracy was forced to 100 and the heal appeared on
        a single 100%% branch."""
        state = self._absorb_state("hydropump")
        branches = _branches(state, "hydropump", "none")
        heal_pcts = [pct for pct, instrs in branches
                     if any(i.startswith("Heal SideTwo: 92") for i in instrs)]
        miss_pcts = [pct for pct, instrs in branches
                     if not any(i.startswith("Heal SideTwo: 92") for i in instrs)]
        self.assertTrue(heal_pcts and abs(sum(heal_pcts) - 80.0) < 0.6,
                        f"expected an ~80% heal branch, got {branches}")
        self.assertTrue(miss_pcts, f"expected a miss branch with no heal, got {branches}")

    def test_unprotected_hit_still_absorbs(self) -> None:
        """Control (both builds): Surf into an unprotected Water Absorb holder
        converts to the quarter heal (92 = floor(371/4)), no damage."""
        state = self._absorb_state("surf")
        flat = _flat(_branches(state, "surf", "none"))
        self.assertTrue(any(i.startswith("Heal SideTwo: 92") for i in flat))
        self.assertFalse(any(i.startswith("Damage SideTwo:") for i in flat))

    def test_full_hp_absorb_keeps_hit_and_miss_branches(self) -> None:
        state = self._absorb_state("hydropump", defender_hp=371)
        branches = _branches(state, "hydropump", "none")
        marker_pcts = [
            pct
            for pct, instrs in branches
            if any(i.startswith("Heal SideTwo: 0") for i in instrs)
        ]
        miss_pcts = [
            pct
            for pct, instrs in branches
            if not any(i.startswith("Heal SideTwo: 0") for i in instrs)
        ]
        self.assertTrue(marker_pcts and abs(sum(marker_pcts) - 80.0) < 0.6, branches)
        self.assertTrue(miss_pcts and abs(sum(miss_pcts) - 20.0) < 0.6, branches)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
