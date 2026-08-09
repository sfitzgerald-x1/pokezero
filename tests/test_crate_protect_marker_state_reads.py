"""Pin the PRODUCTION READ that decides the Sleep Talk Protect marker.

The predicate half of this decision (`protect_blocked_marker_side`) is pinned inside the
crate. Its INPUTS are not: `render_move_phase` reads three facts off the live state --
the defender's `PROTECT` volatile, its ability, and its HP -- and every crate unit test
passes those in as literals. That gap is not theoretical. `protect_blocked_marker_side`'s
own doc block records that hardcoding `defender_protected = true` at the read site
survives the whole crate suite, because no crate test reaches the read.

#1211 moved the absorb axis from ability PRESENCE to "that ability could have emitted THIS
instruction", which is a fact about HP. So the read now carries the behaviour change, and
this module is where it is held.

WHAT THE FIXTURE IS. A sleeping Registeel-shaped attacker with Sleep Talk and two
protect-blockable callees whose blocked tails are byte-identical, so
`identify_sleep_talk_called` returns `Ambiguous` and the renderer takes the unnamed-callee
walk. The defender holds `PROTECT` and the engine's Protect-blocked branch marker -- a
zero-amount `Heal` on the defender -- is the whole tail. That is the shape captured at
`fb3m21-946004` round 45.

THE THREE ARMS, which are the reason this is a table rather than one case:

  * no absorb ability            -> rendered before #1211 and after      (control)
  * absorb ability, HP headroom  -> refused before #1211, rendered after (the change)
  * absorb ability, FULL HP      -> refused before and after             (the guard)

The third arm is the one that fails silently if it regresses: a full-HP absorber is
genuinely ambiguous, because `ability_modify_attack_against` runs BEFORE the Protect gate
and restores `flags.protect`, so a protect-bypassing Water move (WATERSPORT) keeps its
converted heal and produces the same instruction with a different meaning. Rendering
`|-activate|...|Protect` there would be a WRONG LINE in a SEARCHED world, which is worse
than the abort #1211 removes.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import pokezero_search
except ImportError:  # pragma: no cover - native module optional in some venvs
    pokezero_search = None

from pokezero.poke_engine_adapter import (  # noqa: E402
    BattleSpec,
    MoveSpec,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
)

CTX = json.dumps({"p1": ["Registeel"], "p2": ["Mantine"], "turn": 1})

# The marker's own line, and the sub-case tokens that count a render of it. The
# `_absorb_headroom` suffix is #1211's half; the bare token is #1157's and must not change
# meaning under a reader differencing eras.
PROTECT_LINE = "|-activate|p2a: Mantine|Protect"
RENDERED = "sleeptalk_called_unidentified:protect_marker_rendered"
RENDERED_HEADROOM = "sleeptalk_called_unidentified:protect_marker_rendered_absorb_headroom"
REFUSED = (
    "sleeptalk_called_unidentified:ambiguous_unrenderable:heal_zero_marker"
)


def _sleeper() -> PokemonSpec:
    # Two protect-blockable callees plus Sleep Talk. Blocked by Protect both collapse to
    # the same single-instruction tail, which is what makes the callee AMBIGUOUS -- the
    # precondition for this whole code path.
    return PokemonSpec(
        id="registeel", level=100, types=("steel",), hp=200, maxhp=200,
        attack=100, defense=100, special_attack=100, special_defense=100, speed=500,
        status="sleep", ability=None, item=None, sleep_turns=3,
        moves=(
            MoveSpec(id="sleeptalk", pp=16),
            MoveSpec(id="tackle", pp=32),
            MoveSpec(id="scratch", pp=32),
        ),
    )


def _defender(ability: str | None, hp: int) -> PokemonSpec:
    return PokemonSpec(
        id="mantine", level=100, types=("water",), hp=hp, maxhp=252,
        attack=100, defense=100, special_attack=100, special_defense=100, speed=100,
        status="none", ability=ability, item=None, sleep_turns=0,
        moves=(MoveSpec(id="splash", pp=32),),
    )


@unittest.skipIf(pokezero_search is None, "pokezero_search native module not built")
class ProtectMarkerStateReadTests(unittest.TestCase):
    def _branches(self, ability: str | None, hp: int) -> list[dict]:
        spec = BattleSpec(
            side_one=SideSpec(
                pokemon=(_sleeper(),), volatile_statuses=(), side_conditions={}, boosts={},
            ),
            side_two=SideSpec(
                pokemon=(_defender(ability, hp),),
                # PRE-SET, so the marker's producer fires on this very turn rather than
                # needing a second turn of setup the renderer would not see.
                volatile_statuses=("protect",),
                side_conditions={}, boosts={},
            ),
        )
        state = build_poke_engine_state(spec).to_string()
        report = json.loads(
            pokezero_search.branch_events(state, "sleeptalk", "splash", CTX, True, False)
        )
        return report["branches"]

    def _marker_branch(self, ability: str | None, hp: int) -> dict:
        """The branch carrying the Protect-blocked marker, however it was resolved.

        Located by its OUTCOME rather than by index, because a branch list that changed
        shape would otherwise make this module pass by looking at the wrong branch.
        """
        candidates = [
            branch
            for branch in self._branches(ability, hp)
            if PROTECT_LINE in branch["events"]
            or REFUSED in branch["attribution_unsafe_reasons"]
        ]
        self.assertEqual(
            len(candidates), 1,
            f"expected exactly one marker branch for ability={ability} hp={hp}: "
            f"{self._branches(ability, hp)}",
        )
        return candidates[0]

    def test_the_fixture_actually_reaches_the_marker(self) -> None:
        """NULL-WORLD GUARD, first, because every other assertion here is vacuous if the
        fixture stops producing a Protect-blocked ambiguous callee. A fixture that renders
        an ordinary turn would satisfy 'no refusal' trivially."""

        branch = self._marker_branch(None, 200)
        self.assertIn(PROTECT_LINE, branch["events"], branch)

    def test_no_absorb_ability_renders_the_marker(self) -> None:
        """CONTROL: unchanged by #1211, and counted under the ORIGINAL token."""

        branch = self._marker_branch(None, 200)
        self.assertFalse(branch["attribution_unsafe"], branch)
        self.assertIn(RENDERED, branch["lossy_subcases"], branch)
        self.assertNotIn(RENDERED_HEADROOM, branch["lossy_subcases"], branch)

    def test_absorb_ability_with_hp_headroom_now_renders(self) -> None:
        """THE CHANGE. 192/252 is the HP captured at `fb3m21-946004` round 45. Before
        #1211 this branch was attribution-unsafe and its world was thrown away."""

        branch = self._marker_branch("waterabsorb", 192)
        self.assertNotIn(REFUSED, branch["attribution_unsafe_reasons"], branch)
        self.assertFalse(branch["attribution_unsafe"], branch)
        self.assertIn(PROTECT_LINE, branch["events"], branch)

    def test_the_reclaimed_render_is_counted_under_its_own_token(self) -> None:
        """A class that stops refusing must not stop being visible, and #1211's reclaim
        must be separable from #1157's -- summing them would hide this one inside a series
        that already moves for another reason."""

        branch = self._marker_branch("waterabsorb", 192)
        self.assertIn(RENDERED_HEADROOM, branch["lossy_subcases"], branch)
        self.assertNotIn(RENDERED, branch["lossy_subcases"], branch)

    def test_a_full_hp_absorber_still_refuses(self) -> None:
        """THE GUARD, and the arm whose regression is silent. At full HP the absorb no-op
        is a real producer of this exact instruction, so the world must keep aborting
        rather than be searched against a fabricated `Protect` line."""

        branch = self._marker_branch("waterabsorb", 252)
        self.assertTrue(branch["attribution_unsafe"], branch)
        self.assertIn(REFUSED, branch["attribution_unsafe_reasons"], branch)
        self.assertNotIn(PROTECT_LINE, branch["events"], branch)

    def test_one_hp_below_full_is_the_boundary(self) -> None:
        """The clamp yields exactly 1 at 251/252, so the engine writes a REAL heal there
        and never the marker. Pins the boundary rather than a comfortable distance from
        it: a `>=` slip in the clamp arithmetic moves exactly this point."""

        self.assertIn(
            PROTECT_LINE, self._marker_branch("waterabsorb", 251)["events"]
        )
        self.assertNotIn(
            PROTECT_LINE, self._marker_branch("waterabsorb", 252)["events"]
        )

    def test_volt_absorb_is_read_the_same_way(self) -> None:
        """The read is on the ability SET, not on one ability. Volt Absorb reaching a
        different answer than Water Absorb would mean the read had been narrowed to a
        literal somewhere between here and `absorb_ability_can_emit_a_zero_heal`."""

        self.assertIn(PROTECT_LINE, self._marker_branch("voltabsorb", 192)["events"])
        self.assertNotIn(PROTECT_LINE, self._marker_branch("voltabsorb", 252)["events"])

    def test_a_non_absorb_ability_is_not_treated_as_one(self) -> None:
        """The other direction on the ability set: Flash Fire sets a volatile and never a
        heal, so it must not cost a world at any HP. Widening the guard back to
        `is_absorb_ability` would refuse the full-HP case here for nothing."""

        for hp in (192, 252):
            self.assertIn(
                PROTECT_LINE, self._marker_branch("flashfire", hp)["events"],
                f"flashfire at {hp}/252 must render",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
