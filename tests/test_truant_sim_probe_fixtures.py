"""The three `gen3customgame` probes that settled the Truant phase, as fixtures.

These are the measurements the state machine in `test_truant_phase_seeding.py` encodes. They
are kept executable rather than transcribed into prose because the derivation was WRONG once:
the composed rule for a traced holder predicted it loafs on its first move turn, and the
turn-0 probe showed it ACTS. A prose note would have preserved the wrong answer.

Deterministic and cheap (fixed seed, scripted turns, no sampling), so they run with the suite.
Skipped when the Showdown checkout is unavailable.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

SHOWDOWN = Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown")


def _tags(step) -> list[str]:
    out = []
    for line in step.protocol_lines:
        if "|cant|" in line and "Truant" in line:
            out.append("LOAF:" + line.split("|")[2].split(":")[0].strip())
        elif line.startswith("|move|"):
            out.append("MOVE:" + line.split("|")[2].split(":")[0].strip())
    return out


@unittest.skipUnless(SHOWDOWN.exists(), "vendored Showdown checkout not present")
class TruantSimProbes(unittest.TestCase):
    def _run(self, p1_team, p2_team, turns, seed=7):
        from pokezero.showdown_fixture import run_multi_turn_fixture

        return run_multi_turn_fixture(
            p1_team=p1_team, p2_team=p2_team, turns=turns, seed=seed
        )

    @staticmethod
    def _mons():
        from pokezero.showdown_fixture import FixturePokemon

        return (
            FixturePokemon(species="Porygon2", moves=["tackle"], ability="Trace", item="Leftovers"),
            FixturePokemon(species="Snorlax", moves=["tackle"], ability="Immunity", item="Leftovers"),
            FixturePokemon(species="Slaking", moves=["tackle"], ability="Truant", item="Leftovers"),
        )

    def test_native_and_turn0_tracer_both_act_first_then_alternate(self) -> None:
        """Probe 1. A tracer acquiring at turn 0 ACTS first — the derivation said LOAFS."""
        por, _bench, slak = self._mons()
        r = self._run([por], [slak], [("move tackle", "move tackle")] * 4)
        got = [_tags(s) for s in r.steps]
        self.assertIn("MOVE:p1a", got[0], f"turn 1 should ACT for both: {got[0]}")
        self.assertIn("MOVE:p2a", got[0])
        self.assertIn("LOAF:p1a", got[1], f"turn 2 should LOAF for both: {got[1]}")
        self.assertIn("LOAF:p2a", got[1])

    def test_mid_battle_tracer_loafs_first(self) -> None:
        """Probe 2. The same acquisition mid-battle inverts the parity."""
        por, bench, slak = self._mons()
        r = self._run(
            [bench, por], [slak],
            [("move tackle", "move tackle"), ("switch 2", "move tackle"),
             ("move tackle", "move tackle"), ("move tackle", "move tackle")],
        )
        got = [_tags(s) for s in r.steps]
        self.assertIn("LOAF:p1a", got[2], f"first move turn after tracing should LOAF: {got[2]}")
        self.assertIn("MOVE:p1a", got[3], f"and alternate back: {got[3]}")

    def test_post_residual_replacement_loafs_first(self) -> None:
        """Probe 3. The replacement guard: entering AFTER upkeep misses that turn's flip."""
        from pokezero.showdown_fixture import FixturePokemon

        frail = FixturePokemon(species="Shedinja", moves=["tackle"], ability="Wonder Guard", item="Leftovers")
        slak = FixturePokemon(species="Slaking", moves=["tackle"], ability="Truant", item="Leftovers")
        ttar = FixturePokemon(species="Tyranitar", moves=["tackle"], ability="Sand Stream", item="Leftovers")
        r = self._run(
            [frail, slak], [ttar],
            [("move tackle", "move tackle"), ("switch 2", None),
             ("move tackle", "move tackle"), ("move tackle", "move tackle")],
            seed=11,
        )
        got = [_tags(s) for s in r.steps]
        self.assertIn("LOAF:p1a", got[2], f"a post-residual replacement should LOAF first: {got[2]}")
        self.assertIn("MOVE:p1a", got[3], f"and alternate back: {got[3]}")


if __name__ == "__main__":
    unittest.main()
