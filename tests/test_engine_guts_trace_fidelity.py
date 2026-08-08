"""Regression pins for the gen3 Guts/Facade wake-turn and Trace-activation patches.

Requires the patched wheel (``scripts/setup_poke_engine.sh``). Two patches:

``poke-engine-gen3-guts-facade-wake-turn.patch``
    before_move's choice modification ran BEFORE the sleep/freeze branching, so
    a sleeping Guts user's wake-turn attack carried a phantom 1.5x (Facade: 2x).
    Showdown cures sleep in ``slp.onBeforeMove`` (data/mods/gen3/conditions.ts)
    and evaluates Guts at damage time via ``onModifyAtk`` (data/abilities.ts),
    so the wake-turn attack is UNBOOSTED. Sleep Talk-called moves execute while
    still asleep and keep the boost.

``poke-engine-gen3-trace-no-activation.patch``
    ``sim/pokemon.ts setAbility()`` fires the copied ability's 'Start' event
    only when ``battle.gen > 3``; gen3's trace override calls plain setAbility.
    A gen3 traced Intimidate therefore never drops the foe's Attack. Upstream
    ran the switch-in activation on the copied ability.

Sim ground truth (gen3 Custom Game fixtures via pokezero.showdown_fixture,
2026-07-29, transcribed from live probe runs — see the class docstrings):
the engine-side stats below mirror the fixture requests exactly
(Machamp L80 atk 237, Smeargle L90 def 95 hp 226).
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


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class GutsFacadeWakeTurnTests(unittest.TestCase):
    """Sim probe: Smeargle Spores Machamp; Machamp clicks the attack every turn.

    Wake-turn Rock Slide damage observed across seeds 1-8:
    {116, 119, 122, 125, 126, 127, 129} — exactly the roll set of the UNBOOSTED
    max 129 (base 127 + 2, no STAB, neutral), never the Guts-boosted max 192
    (min legal roll 163). Wake-turn Facade damage observed across seeds 1-10:
    {108..120} — the roll set of the un-doubled, un-Guts-boosted max 120.
    Paralyzed control (Thunder Wave first): observed {165, 167, 176, 190} —
    inside the boosted max-192 roll set, so Guts still fires for a status that
    persists through the move.
    """

    def _machamp_vs_smeargle(self, *, move: str, moves=None, status="none",
                             sleep_turns=0):
        pe = poke_engine
        dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
        machamp = pe.Pokemon(
            id="machamp", level=80, types=("fighting", "typeless"), hp=258,
            maxhp=258, ability="guts", item="none", attack=237, defense=157,
            special_attack=133, special_defense=165, speed=117, status=status,
            sleep_turns=sleep_turns,
            moves=[pe.Move(id=m, pp=16) for m in (moves or (move,))],
        )
        smeargle = pe.Pokemon(
            id="smeargle", level=90, types=("normal", "typeless"), hp=226,
            maxhp=226, ability="owntempo", item="none", attack=68, defense=95,
            special_attack=68, special_defense=113, speed=167,
            moves=[pe.Move(id="splash", pp=16)],
        )
        state = pe.State(
            side_one=pe.Side(active_index="0", pokemon=[machamp] + [dummy] * 5),
            side_two=pe.Side(active_index="0", pokemon=[smeargle] + [dummy] * 5),
            weather="none", terrain="none", trick_room=False,
        )
        return pe.generate_instructions(state, move, "splash")

    @staticmethod
    def _damages(branches):
        out = []
        for branch in branches:
            for ins in branch.instruction_list:
                text = str(ins)
                if text.startswith("Damage SideTwo:"):
                    out.append(int(text.rsplit(":", 1)[1]))
        return out

    def test_guts_wake_turn_rockslide_is_unboosted(self) -> None:
        """Wake branch deals floor(129 * 0.925) = 119 — the sim's unboosted max.

        Unpatched, the branch dealt floor(192 * 0.925) = 177: the choice was
        boosted while the attacker was still flagged asleep, even though the
        move only executes on the branch that woke it.
        """
        branches = self._machamp_vs_smeargle(
            move="rockslide", status="sleep", sleep_turns=1)
        damages = self._damages(branches)
        self.assertIn(119, damages)
        self.assertNotIn(177, damages)

    def test_guts_paralyzed_control_still_boosted(self) -> None:
        """Paralysis persists through the move: floor(192 * 0.925) = 177 stays.

        Passes with and without the patch — a change moving this would be
        altering behaviour the sim endorses ({165,167,176,190} observed, all in
        the boosted max-192 roll set).
        """
        branches = self._machamp_vs_smeargle(move="rockslide", status="paralyze")
        damages = self._damages(branches)
        self.assertIn(177, damages)
        self.assertNotIn(119, damages)

    def test_sleep_talk_called_move_keeps_guts_boost(self) -> None:
        """A Sleep Talk-called Rock Slide executes while STILL asleep -> 177.

        Showdown's onModifyAtk sees status 'slp' at damage time for a
        sleepUsable move, so the boost is correct there and must be kept.
        """
        branches = self._machamp_vs_smeargle(
            move="sleeptalk", moves=("sleeptalk", "rockslide"),
            status="sleep", sleep_turns=1)
        damages = self._damages(branches)
        self.assertIn(177, damages)
        self.assertNotIn(119, damages)

    def test_facade_wake_turn_not_doubled(self) -> None:
        """Wake-turn Facade: floor(120 * 0.925) = 111 — no 2x, no Guts 1.5x.

        Unpatched, BP was doubled AND Guts-boosted while still flagged asleep.
        """
        branches = self._machamp_vs_smeargle(
            move="facade", status="sleep", sleep_turns=1)
        # Crit branches are excluded, because a crit legitimately looks boosted.
        #
        # `poke-engine-gen3-crit-kill-split.patch` (C27, commit 97fe2c60, #1007) applied the
        # kill-split identity to the crit arm, turning this state's two damage branches into
        # three. The original `assertNotIn(226, damages)` was written when the crit arm was a
        # single collapsed branch at floor(240 * 0.925) = 222, so 226 could only mean the
        # Guts/doubling bug. It no longer can:
        #
        #   111 @ 23.438%  25% wake x 15/16 non-crit
        #   226 @  0.586%  25% x 1/16 x 6/16 -- the crit KILL arm, priced at the DEFENDER'S HP
        #                  (226), not at a roll; 6 of the 16 rolls in the 240 crit fan reach it
        #   214 @  0.977%  25% x 1/16 x 10/16 -- mean of the ten sub-lethal crit rolls
        #
        # The two crit branches are 0.0625 of the wake mass: exactly the gen3 crit rate. So
        # `assertNotIn(226)` became unsatisfiable under CORRECT behaviour -- any engine emitting
        # a crit-kill arm against a 226-HP target emits 226. A raw damage value was the wrong
        # proxy for "boosted".
        #
        # (#1152's crit-straddle patch sub-splits this further, but only when residual_thresholds
        # is non-empty; this state has no residuals so it does not fire. An earlier revision of
        # this comment credited #1152 for the split, which was wrong by six days and the wrong
        # mechanism.)
        #
        # The Guts bug remains detectable: it lives in the 15/16 non-crit branch, which is NOT
        # excluded. The property is unchanged -- Facade on the wake turn is
        # floor(120 * 0.925) = 111, no 2x, no Guts 1.5x.
        wake = [b for b in branches if self._damages([b])]
        wake_mass = sum(float(b.percentage) for b in wake)
        crit = [b for b in wake if float(b.percentage) / wake_mass < 0.05]
        damages = self._damages([b for b in wake if b not in crit])

        # PROPERTY FIRST, guard second. Under a real Guts regression the wake branch is a
        # guaranteed kill, so there is no crit split and the excluded mass is 0 -- if the guard
        # ran first it would fail as "the baseline moved, re-derive this pin", which is the
        # wrong nudge for an actual regression and how a pin gets relaxed instead of fixed.
        # Ordered this way it fails as "111 not found in [226]".
        self.assertIn(111, damages)
        for boosted in (226, 258):  # HP-capped kill values of any boosted NON-CRIT calc
            self.assertNotIn(boosted, damages)

        # Now the guard: the exclusion must be the crit band and nothing else, so the filter
        # cannot quietly grow to swallow the branch under test.
        self.assertAlmostEqual(
            sum(float(b.percentage) for b in crit) / wake_mass,
            1.0 / 16.0,
            places=3,
            msg="the excluded branches are not the 1/16 crit band; re-derive this pin",
        )


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class TraceActivationTests(unittest.TestCase):
    """Sim probe (seed 42): Porygon2 (Trace) switches in against an active
    Intimidate Mightyena. Protocol shows

        |-ability|p2a: Porygon2|Intimidate|Trace|[from] ability: Trace|[of] p1a: Mightyena

    and NO ``-unboost|p1a: Mightyena`` — matching the repro row (seed 1500222
    step 17, Showdown Hidden Power damage 119 = unboosted). sim/pokemon.ts
    setAbility() fires 'Start' only for gen > 3.
    """

    @staticmethod
    def _mon(pe, id, ability, moves, types=("normal", "typeless")):
        return pe.Pokemon(
            id=id, level=80, types=types, hp=250, maxhp=250, ability=ability,
            item="none", attack=200, defense=150, special_attack=150,
            special_defense=150, speed=100,
            moves=[pe.Move(id=m, pp=16) for m in moves],
        )

    def test_traced_intimidate_does_not_activate(self) -> None:
        pe = poke_engine
        dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
        mightyena = self._mon(pe, "mightyena", "intimidate", ("tackle",),
                              types=("dark", "typeless"))
        castform = self._mon(pe, "castform", "forecast", ("splash",))
        porygon2 = self._mon(pe, "porygon2", "trace", ("splash",))
        state = pe.State(
            side_one=pe.Side(active_index="0", pokemon=[mightyena] + [dummy] * 5),
            side_two=pe.Side(active_index="0",
                             pokemon=[castform, porygon2] + [dummy] * 4),
            weather="none", terrain="none", trick_room=False,
        )
        branches = pe.generate_instructions(state, "tackle", "porygon2")
        all_instructions = [
            str(i) for b in branches for i in b.instruction_list]
        self.assertTrue(
            any(s.startswith("ChangeAbility SideTwo:") for s in all_instructions),
            f"trace must still copy the ability: {all_instructions}")
        self.assertFalse(
            any(s.startswith("Boost SideOne Attack:") for s in all_instructions),
            f"gen3 traced Intimidate must NOT activate: {all_instructions}")

    def test_real_intimidate_still_activates(self) -> None:
        """Control: a genuine Intimidate switch-in keeps its Attack drop."""
        pe = poke_engine
        dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
        castform_one = self._mon(pe, "castform", "forecast", ("splash",))
        mightyena = self._mon(pe, "mightyena", "intimidate", ("splash",),
                              types=("dark", "typeless"))
        castform_two = self._mon(pe, "castform", "forecast", ("splash",))
        state = pe.State(
            side_one=pe.Side(active_index="0",
                             pokemon=[castform_one, mightyena] + [dummy] * 4),
            side_two=pe.Side(active_index="0",
                             pokemon=[castform_two] + [dummy] * 5),
            weather="none", terrain="none", trick_room=False,
        )
        branches = pe.generate_instructions(state, "mightyena", "splash")
        all_instructions = [
            str(i) for b in branches for i in b.instruction_list]
        self.assertTrue(
            any(s.startswith("Boost SideTwo Attack: -1") for s in all_instructions),
            f"real Intimidate must still fire: {all_instructions}")


if __name__ == "__main__":
    unittest.main()
