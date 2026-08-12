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

THE SECOND AXIS, added because the first table pinned a PROXY. The read now carries a
third fact: whether any Sleep Talk callee's post-modification choice still holds the
absorb ability's converted `Heal{target: Opponent}`, which is producer 2's own and only
precondition. The old table had no such axis, so its full-HP arm was justified by the
protect-BYPASSING absorbed move (WATERSPORT) while its fixture used `tackle`/`scratch` --
Normal-typed and protect-flagged, which Water Absorb cannot convert at all. It therefore
asserted a refusal the engine could not need, and that is the whole of the census block's
`ambiguous_unrenderable:heal_zero_marker` class: all 31 decisions at `--truth-sims 64`
hold PROTECT at full HP with Water Absorb and have exactly two matching callees, both
protect-flagged with `heal == None` -- (Ice Beam, Toxic) x19 and (Surf, Toxic) x12.

THE ARMS, which are the reason this is a table rather than one case:

  * no absorb ability, unconvertible callees   -> renders, ORIGINAL token       (#1157)
  * absorb ability, HP headroom, unconvertible -> renders, HEADROOM token       (#1211)
  * absorb ability, FULL HP, unconvertible     -> renders, FULL_HP token        (this PR)
  * absorb ability, FULL HP, CONVERTIBLE       -> REFUSES                       (the guard)

The last arm is the one that fails silently if it regresses, and it is the one the old
table never built: `ability_modify_attack_against` runs BEFORE the Protect gate and
restores `flags.protect`, so a protect-bypassing Water move keeps its converted heal and
produces the same instruction with a different meaning. Rendering `|-activate|...|Protect`
there would be a WRONG LINE in a SEARCHED world, which is worse than any abort.
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
RENDERED_FULL_HP = "sleeptalk_called_unidentified:protect_marker_rendered_absorb_full_hp"
REFUSED = (
    "sleeptalk_called_unidentified:ambiguous_unrenderable:heal_zero_marker"
)

# The CALLEE axis. `tackle`/`scratch` are Normal-typed and protect-flagged, so no absorb
# ability can convert either one and `remove_effects_for_protect` would clear the heal even
# if it could -- producer 2 cannot reach the tail. `watersport` is Water-typed and carries NO
# protect flag, so Water Absorb converts it and Protect does not strip it; paired with a
# blocked `tackle` both callees regenerate the same zero-heal tail from DIFFERENT producers,
# which is the irreducible case.
UNCONVERTIBLE = ("tackle", "scratch")
CONVERTIBLE = ("watersport", "tackle")
# The same convertible pair with the bypassing callee LAST. `get_sleep_talk_choices` walks
# the move list in slot order, so this is the case a scan that stops at the second match --
# or that only looks at MATCHING candidates -- gets wrong, and it is the shape two surviving
# mutants exploited before it existed.
CONVERTIBLE_LAST = ("tackle", "scratch", "watersport")


def _sleeper(callees: tuple[str, ...] = UNCONVERTIBLE) -> PokemonSpec:
    # Two callees plus Sleep Talk, chosen so their tails are byte-identical and the callee is
    # therefore AMBIGUOUS -- the precondition for this whole code path.
    return PokemonSpec(
        id="registeel", level=100, types=("steel",), hp=200, maxhp=200,
        attack=100, defense=100, special_attack=100, special_defense=100, speed=500,
        status="sleep", ability=None, item=None, sleep_turns=3,
        moves=(MoveSpec(id="sleeptalk", pp=16),)
        + tuple(MoveSpec(id=move, pp=32) for move in callees),
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
    def _branches(
        self, ability: str | None, hp: int, callees: tuple[str, ...] = UNCONVERTIBLE
    ) -> list[dict]:
        spec = BattleSpec(
            side_one=SideSpec(
                pokemon=(_sleeper(callees),),
                volatile_statuses=(), side_conditions={}, boosts={},
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

    def _marker_branch(
        self, ability: str | None, hp: int, callees: tuple[str, ...] = UNCONVERTIBLE
    ) -> dict:
        """The branch carrying the Protect-blocked marker, however it was resolved.

        Located by its OUTCOME rather than by index, because a branch list that changed
        shape would otherwise make this module pass by looking at the wrong branch.
        """
        branches = self._branches(ability, hp, callees)
        candidates = [
            branch
            for branch in branches
            if PROTECT_LINE in branch["events"]
            or REFUSED in branch["attribution_unsafe_reasons"]
        ]
        self.assertEqual(
            len(candidates), 1,
            f"expected exactly one marker branch for ability={ability} hp={hp} "
            f"callees={callees}: {branches}",
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

    def test_a_full_hp_absorber_with_unconvertible_callees_now_renders(self) -> None:
        """THIS PR, and the arm the census block is made of.

        `tackle`/`scratch` are Normal-typed and protect-flagged, so Water Absorb cannot
        convert either one: producer 2 has no route to this tail whatever the defender's
        HP is, and the zero-amount `Heal` is provably the Protect-blocked branch marker.
        This is the assertion the old table had inverted, and inverting it back is what
        the whole PR does.

        SAFER-DIRECTION MUTANT'S GRAVE. Reverting the callee conjunct at the production
        read site -- or any strictly-more-conservative variant of it, including the exact
        pre-PR expression -- fails HERE and nowhere else in this module.
        """

        branch = self._marker_branch("waterabsorb", 252)
        self.assertNotIn(REFUSED, branch["attribution_unsafe_reasons"], branch)
        self.assertFalse(branch["attribution_unsafe"], branch)
        self.assertIn(PROTECT_LINE, branch["events"], branch)

    def test_the_full_hp_reclaim_is_counted_under_its_own_token(self) -> None:
        """A class that stops refusing must not stop being visible, and this reclaim must
        be separable from BOTH earlier ones. `_absorb_headroom` would be a false statement
        about a full-HP render, and would hide this change inside #1211's series; the bare
        token would hide it inside #1157's."""

        branch = self._marker_branch("waterabsorb", 252)
        self.assertIn(RENDERED_FULL_HP, branch["lossy_subcases"], branch)
        self.assertNotIn(RENDERED_HEADROOM, branch["lossy_subcases"], branch)
        self.assertNotIn(RENDERED, branch["lossy_subcases"], branch)

    def test_a_full_hp_absorber_with_a_convertible_callee_still_refuses(self) -> None:
        """THE GUARD, and the arm whose regression is silent -- now built with the
        counterexample its own rationale has always named.

        `watersport` is Water-typed and carries no protect flag, so the absorb conversion
        survives `remove_effects_for_protect` and its clamped-to-zero heal is a real
        producer of this exact instruction while PROTECT is set. The two callees therefore
        regenerate the same tail from DIFFERENT producers, the marker's meaning is
        genuinely unknown, and the world must keep aborting rather than be searched
        against a fabricated `Protect` line.
        """

        branch = self._marker_branch("waterabsorb", 252, CONVERTIBLE)
        self.assertTrue(branch["attribution_unsafe"], branch)
        self.assertIn(REFUSED, branch["attribution_unsafe_reasons"], branch)
        self.assertNotIn(PROTECT_LINE, branch["events"], branch)
        self.assertNotIn(RENDERED_FULL_HP, branch["lossy_subcases"], branch)

    def test_the_callee_scan_covers_candidates_after_the_second_match(self) -> None:
        """The fail-closed direction is only fail-closed if EVERY callee is scanned.

        Found by mutation, not by review: restoring the old early `return Ambiguous` on the
        second match, and narrowing the scan to MATCHING candidates, both survived the first
        version of this module because its convertible fixture put `watersport` FIRST. Here
        two protect-flagged callees match before it is reached, so a scan that stops early
        renders `Protect` over an ability activation.

        WHAT THIS CLOSES, corrected: the EARLY-RETURN mutant only. Review caught this
        docstring claiming both. The matching-only scan still survives, because `watersport`
        matches this tail too, so a matching-only scan still sees it -- it is an EQUIVALENCE on
        the `Ambiguous` arm, documented as one on `SleepTalkProbe`, not a gap this fixture
        forgot. A claimed kill that is not delivered is the same defect as a dangling citation.
        """

        branch = self._marker_branch("waterabsorb", 252, CONVERTIBLE_LAST)
        self.assertIn(REFUSED, branch["attribution_unsafe_reasons"], branch)
        self.assertNotIn(PROTECT_LINE, branch["events"], branch)
        self.assertNotIn(RENDERED_FULL_HP, branch["lossy_subcases"], branch)

    def test_one_hp_below_full_is_the_boundary(self) -> None:
        """The clamp yields exactly 1 at 251/252, so the engine writes a REAL heal there
        and never a zero marker. Pins the boundary rather than a comfortable distance from
        it: a `>=` slip in the clamp arithmetic moves exactly this point.

        Read on the CONVERTIBLE fixture, because that is now the only one where the clamp
        is observable: with unconvertible callees both HPs render, and rightly so. At 251
        `watersport` heals for 1 and its tail stops matching `tackle`'s, so the callee is
        NAMED and the named path renders Protect; at 252 both collapse onto the zero marker
        and the branch refuses.
        """

        self.assertIn(
            PROTECT_LINE, self._marker_branch("waterabsorb", 251, CONVERTIBLE)["events"]
        )
        self.assertNotIn(
            PROTECT_LINE, self._marker_branch("waterabsorb", 252, CONVERTIBLE)["events"]
        )

    def test_volt_absorb_is_read_the_same_way(self) -> None:
        """The read is on the ability SET, not on one ability. Volt Absorb reaching a
        different answer than Water Absorb would mean the read had been narrowed to a
        literal somewhere between here and `absorb_ability_can_emit_a_zero_heal`.

        Asserted through the COUNTER, not only through the line, because the line alone
        does not discriminate: see `test_volt_absorb_has_no_bypassing_producer_in_gen3` for
        why narrowing the ability set is behaviour-neutral for Volt Absorb in gen3. The
        token is not neutral -- a narrowed set would count these renders under #1157's bare
        token and silently redefine two series at once.
        """

        self.assertIn(PROTECT_LINE, self._marker_branch("voltabsorb", 192)["events"])
        headroom = self._marker_branch("voltabsorb", 192)
        self.assertIn(RENDERED_HEADROOM, headroom["lossy_subcases"], headroom)
        full = self._marker_branch("voltabsorb", 252)
        self.assertIn(PROTECT_LINE, full["events"], full)
        self.assertIn(RENDERED_FULL_HP, full["lossy_subcases"], full)
        self.assertNotIn(RENDERED, full["lossy_subcases"], full)

    def test_volt_absorb_has_no_bypassing_producer_in_gen3(self) -> None:
        """MEASURED unreachability, with the witness that it could have fired, because
        "absent" otherwise means "absent at the settings tried".

        Water Absorb's converted heal survives Protect through an unflagged Water move, and
        `watersport` demonstrates it two tests above. Volt Absorb has no such route in gen3,
        and the reason is an engine ASYMMETRY worth recording rather than an accident:
        `gen3/abilities.rs` gates Volt Absorb on `category != MoveCategory::Status` (a gen3
        fidelity fix; Water Absorb carries no such clause), and every Electric move in the
        table without the protect flag -- Charge, Magnet Rise, Electric Terrain, Ion Deluge,
        Magnetic Flux -- is a Status move. So the conversion never happens for the one class
        of callee that could keep it.

        Enumerated rather than argued, and paired with the Water witness in the same
        assertion so a change that broke BOTH would not read as this test still passing.
        """

        bypassing_electric = (
            "charge", "magnetrise", "electricterrain", "iondeluge", "magneticflux",
        )
        for move in bypassing_electric:
            branch = self._marker_branch("voltabsorb", 252, (move, "tackle"))
            self.assertNotIn(
                REFUSED, branch["attribution_unsafe_reasons"],
                f"{move} reached Volt Absorb's converted heal, so the gen3 Status gate no "
                f"longer holds and this claim must be re-derived: {branch}",
            )
        # THE WITNESS: the same fixture shape DOES refuse for the ability that has a
        # producer, so the zeros above are zeros from a fixture that can report a refusal.
        water = self._marker_branch("waterabsorb", 252, CONVERTIBLE)
        self.assertIn(REFUSED, water["attribution_unsafe_reasons"], water)

    def test_the_absorb_producer_at_headroom_emits_a_different_instruction(self) -> None:
        """THE PREMISE, demonstrated through the engine rather than argued from its source.

        The whole fix rests on one claim: a Water Absorb defender with HP headroom cannot
        have produced a ZERO heal, because the engine writes a REAL one instead. Every
        other test here pins the renderer's REACTION to that claim; this one pins the
        claim. Without it the suite would still be green if the engine started emitting
        zero markers at headroom, and the fix would be rendering `Protect` over them.

        No PROTECT volatile, so the absorb genuinely fires and nothing strips it. The
        callee set is Water-typed so the conversion happens. Below full HP the branch is
        refused as `heal_defender` -- a nonzero heal on the defender, a different
        instruction with a different protocol line. At full HP the same setup produces
        `heal_zero_marker`, which is the marker this module is about.
        """

        def reasons(hp: int) -> set[str]:
            spec = BattleSpec(
                side_one=SideSpec(
                    pokemon=(PokemonSpec(
                        id="registeel", level=100, types=("steel",), hp=200, maxhp=200,
                        attack=100, defense=100, special_attack=100,
                        special_defense=100, speed=500, status="sleep", ability=None,
                        item=None, sleep_turns=3,
                        moves=(MoveSpec(id="sleeptalk", pp=16),
                               MoveSpec(id="surf", pp=32),
                               MoveSpec(id="watergun", pp=32)),
                    ),),
                    volatile_statuses=(), side_conditions={}, boosts={},
                ),
                side_two=SideSpec(
                    pokemon=(_defender("waterabsorb", hp),),
                    volatile_statuses=(),  # NO Protect: the absorb is the only producer.
                    side_conditions={}, boosts={},
                ),
            )
            state = build_poke_engine_state(spec).to_string()
            report = json.loads(
                pokezero_search.branch_events(state, "sleeptalk", "splash", CTX, True, False)
            )
            out: set[str] = set()
            for branch in report["branches"]:
                out.update(
                    r.rsplit("ambiguous_unrenderable:", 1)[-1]
                    for r in branch["attribution_unsafe_reasons"]
                )
            return out

        self.assertEqual(reasons(192), {"heal_defender"})
        self.assertEqual(reasons(252), {"heal_zero_marker"})

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
