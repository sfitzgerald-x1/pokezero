"""Unknown Substitute health is SAMPLED from a bounded range, not refused.

`substitute_health_unknown` was 396 killed decisions in era 59 -- 48.6% of the
construction channel and its largest class once `self_moveset_mismatch` closed.

WHY IT SHOULD NEVER HAVE BEEN A REFUSAL. GOAL.md §0.2: *"Hidden information is not a
refusal category. The belief machinery's entire design is to sample any consistent
hypothesis -- it already does this for unrevealed sets, items, and abilities."* A
Substitute's remaining HP is one more belief dimension, and the same argument that made
`trapped`/`disabled` sample-not-refuse applies to it unchanged.

WHY SAMPLING IS HONEST HERE RATHER THAN A GUESS. `unknown` arises from exactly one
situation: a NON-BREAKING hit whose damage the public record does not reveal. Showdown
emits `|-activate|SLOT|Substitute|[damage]`, and `_update_substitute_health_state` can
resolve the amount only for gen 3's four public fixed-damage moves (Dragon Rage 40, Sonic
Boom 20, Seismic Toss and Night Shade at the attacker's level); anything else is
`unknown`.

But a non-breaking hit carries TWO facts, and the old code used neither:

  1. it removed at least 1 HP, and
  2. the Substitute SURVIVED it.

So each hit contributes a MINIMUM DEPLETION -- its proven damage when the record resolves it
(gen 3's four fixed-damage moves), otherwise 1 HP -- and remaining health lies in
`[1, initial - min_depletion]` where `initial` is `floor(maxhp / 4)`.

ONE accumulator, deliberately. Two of them (a hit count plus a proven-damage total) produced
an ORDERING ASYMMETRY that review measured twice: the proven-damage accumulation sat inside
the `exact` branch, whose guard fails after any unknown hit, so a fixed-damage hit landing
AFTER an unknown one contributed 1 instead of its real 77. Charging each hit exactly once
into a single minimum makes the result order-independent -- Seismic Toss then Ice Beam and Ice
Beam then Seismic Toss both give 78.

Honest about the width: with one unknown hit and nothing resolved the range is
`[1, initial - 1]`, which is 99.4% of `[1, initial]`, and 75% of real cases are exactly that.
The unknown-hit term is a 1-HP tightening. What does real work is a RESOLVED hit.

STRICTLY ADDITIVE. A payload that does not carry `substituteMinDepletion` refuses exactly
as before, under the same reason code. That matters beyond compatibility: minting a new
code for the no-bound case would rename a refusal rather than fix it, and the rename
would read as a new class in the next era's crosstab -- a discontinuity this campaign has
already had to document twice.
"""

import unittest

from pokezero.dex import load_showdown_dex
from pokezero.engine_world import EngineWorldUnsupported, battle_spec_from_payload
from pokezero.showdown import parse_showdown_replay

from ._showdown_root import requires_showdown, showdown_root
from .test_engine_world import _override, _payload


def _lines(*extra):
    """A minimal public transcript: p1 Breloom subs, p2 Blissey hits it."""

    return [
        "|player|p1|Alice|1|",
        "|player|p2|Bob|2|",
        "|teamsize|p1|2",
        "|teamsize|p2|2",
        "|gen|3",
        "|tier|[Gen 3] Random Battle",
        "|start",
        "|switch|p1a: Breloom|Breloom, L79, M|100/100",
        "|switch|p2a: Blissey|Blissey, L77, F|100/100",
        "|turn|1",
        *extra,
    ]


@requires_showdown()
class SubstituteUnknownHitCountingTests(unittest.TestCase):
    """The PRODUCER side: the parser must count what it cannot measure."""

    def _state(self, *extra):
        replay = parse_showdown_replay(
            _lines(*extra), battle_id="battle-sub-sampling-1", complete_prefix=True
        )
        return replay

    def test_a_non_fixed_damage_non_breaking_hit_is_counted(self):
        """The era-59 shape. Ice Beam is not one of the four public fixed-damage moves."""

        replay = self._state(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
        )

        self.assertEqual(replay.substitute_health_state.get("p1"), "unknown")
        self.assertEqual(
            replay.substitute_min_depletion.get("p1"),
            1,
            "the hit must be COUNTED even though its damage is unknowable -- the count is "
            "what bounds the sampled health instead of refusing the world",
        )

    def test_unknown_hits_accumulate_one_hp_each(self):
        replay = self._state(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|3",
        )

        self.assertEqual(
            replay.substitute_min_depletion.get("p1"),
            2,
            "two unresolved hits prove at least 2 HP gone -- 1 each, not a hit COUNT that "
            "later has to be reconciled with resolved damage",
        )

    def test_a_resolved_hit_after_an_unknown_one_still_contributes_its_damage(self):
        """ORDERING B, the defect review measured after the first fix.

        The proven-damage accumulation used to live inside the `exact` branch, whose guard is
        `substitute_depletion is not None` -- and that is None after any unknown hit. So
        Ice Beam then Seismic Toss charged 1 + 1 = 2 instead of 1 + 77 = 78, and the sampler
        drew from `[1, initial - 2]` while the record proved `<= initial - 78`. Ordering A was
        fixed and ordering B was not; only this test discriminates them.
        """

        unknown_first = self._state(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
            "|move|p2a: Blissey|Seismic Toss|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|3",
        )
        resolved_first = self._state(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|move|p2a: Blissey|Seismic Toss|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|3",
        )

        self.assertEqual(unknown_first.substitute_min_depletion.get("p1"), 78)
        self.assertEqual(
            resolved_first.substitute_min_depletion.get("p1"),
            78,
            "the bound must be ORDER-INDEPENDENT: 77 + 1 either way",
        )
        self.assertEqual(unknown_first.substitute_health_state.get("p1"), "unknown")
        self.assertEqual(resolved_first.substitute_health_state.get("p1"), "unknown")

    def test_a_fixed_damage_move_stays_exact_and_charges_its_real_damage(self):
        """Seismic Toss is public and deterministic, so provenance stays `exact`.

        Two things must hold together. Provenance stays `exact` -- the precise path is
        strictly better than a sampled range and must not be downgraded to one, and
        `substitute_depletion` still carries the exact total that `engine_world` uses
        directly. AND the minimum accumulator charges the real 77, not 1: under the
        single-accumulator design a resolved hit contributes its damage, which is what makes
        the bound order-independent.
        """

        replay = self._state(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|move|p2a: Blissey|Seismic Toss|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
        )

        self.assertEqual(replay.substitute_health_state.get("p1"), "exact")
        self.assertEqual(replay.substitute_depletion.get("p1"), 77, "Blissey is L77")
        self.assertEqual(
            replay.substitute_min_depletion.get("p1"),
            77,
            "a resolved hit charges its REAL damage, not 1 -- charging 1 here is what made "
            "the bound depend on hit ordering",
        )
        # THE INVARIANT that makes the exact and unknown paths independent, asserted rather
        # than argued. `engine_world`'s `exact` branch reads `substituteDepletion` and its
        # `unknown` branch reads `substituteMinDepletion`; the separation is safe only because
        # while provenance is `exact` the two are EQUAL -- both accumulate the same resolved
        # damage. Once one hit is unresolved, `substitute_depletion` goes `None` and can never
        # return, so the two can never diverge while `state == "exact"`. Review measured this
        # true (77/77 and 117/117) but only the consumer half was pinned by a test.
        self.assertEqual(
            replay.substitute_min_depletion.get("p1"),
            replay.substitute_depletion.get("p1"),
            "while provenance is `exact` the bound must equal the exact depletion",
        )

    def test_two_resolved_hits_keep_the_bound_and_the_exact_total_equal(self):
        """The same invariant with accumulation, so a one-hit coincidence cannot satisfy it."""

        replay = self._state(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|move|p2a: Blissey|Seismic Toss|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
            "|move|p2a: Blissey|Dragon Rage|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|3",
        )

        self.assertEqual(replay.substitute_health_state.get("p1"), "exact")
        self.assertEqual(
            replay.substitute_depletion.get("p1"), 117, "Seismic Toss 77 + Dragon Rage 40"
        )
        self.assertEqual(
            replay.substitute_min_depletion.get("p1"),
            replay.substitute_depletion.get("p1"),
        )

    def test_switching_out_resets_the_count(self):
        """The SWITCH path, which is the one this shape actually exercises.

        Two earlier versions of this test claimed to cover the `-start` reset and did not.
        The first put `-end` before the new `-start`, and `-end` resets too. The second used
        a switch, and the switch handler resets as well -- so deleting the `-start` reset
        still left both green.

        That is not a test gap, it is a redundancy: every path into `-start` has already
        zeroed the count (`-end`, switch/Baton Pass, faint, or never-subbed). The `-start`
        reset is kept as defence in depth and is documented as unreachable rather than
        claimed to be covered, because a stale count would otherwise bound a Substitute
        that no longer exists. This test asserts the switch path, which IS load-bearing.
        """

        replay = self._state(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
            "|switch|p1a: Swampert|Swampert, L79, M|100/100",
            "|turn|3",
            "|switch|p1a: Breloom|Breloom, L79, M|100/100",
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|turn|4",
        )

        self.assertEqual(replay.substitute_health_state.get("p1"), "full")
        self.assertEqual(replay.substitute_min_depletion.get("p1"), 0)

    def test_the_count_is_already_zero_at_the_moment_of_the_switch(self):
        """Isolate the switch reset by stopping BEFORE any new `-start`.

        This is the third attempt at this assertion and the first that discriminates. The
        two resets mask each other: any transcript that re-subs after switching passes
        whether the switch handler resets or the `-start` handler does, so deleting either
        left the suite green. Reading the count at the switch, with no new Substitute yet,
        is the only shape that isolates it.
        """

        replay = self._state(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            # A RESOLVED hit, so `min_depletion` is 77 entering the reset. With only an
            # Ice Beam here the assertion below was VACUOUS -- 1 or 0, both falsy-adjacent and
            # review showed the mutation surviving. Every reset transcript needs proven damage.
            "|move|p2a: Blissey|Seismic Toss|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
            "|switch|p1a: Swampert|Swampert, L79, M|100/100",
            "|turn|3",
        )

        self.assertEqual(
            replay.substitute_health_state.get("p1"),
            "absent",
            "the Substitute left the field with its owner",
        )
        self.assertEqual(
            replay.substitute_min_depletion.get("p1"),
            0,
            "a count surviving the switch would bound the NEXT Substitute by hits that "
            "landed on a previous one",
        )

    def test_the_count_is_already_zero_at_the_moment_the_substitute_breaks(self):
        """Isolate the `-end` reset, the same way the switch reset had to be isolated.

        Review found the `-end` mutation surviving: all three of `-start`, `-end` and faint
        reset, so any transcript that re-subs afterwards passes whichever one is deleted.
        Reading the count immediately after `-end`, with no new Substitute yet, is what
        discriminates it.
        """

        replay = self._state(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            # RESOLVED, for the same reason as the switch case: with only an Ice Beam the
            # post-reset assertion could not fail.
            "|move|p2a: Blissey|Seismic Toss|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-end|p1a: Breloom|Substitute",
            "|turn|3",
        )

        self.assertEqual(replay.substitute_health_state.get("p1"), "broken")
        self.assertEqual(
            replay.substitute_min_depletion.get("p1"),
            0,
            "a count surviving the break would bound the NEXT Substitute by hits that "
            "landed on the one that just died",
        )
        self.assertEqual(replay.substitute_min_depletion.get("p1"), 0)

    def test_a_faint_behind_a_live_substitute_resets_the_bound(self):
        """The faint reset is the one that is NOT redundant, and nothing covered it.

        `-start` is masked by `-end`/switch/faint. The faint path has no reset in front of it:
        a residual KO behind a live Substitute is routine, and it carries a non-zero bound into
        the reset. Review found both the old and new field's faint resets surviving mutation.
        """

        replay = self._state(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|move|p2a: Blissey|Seismic Toss|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|-damage|p1a: Breloom|0 fnt|[from] psn",
            "|faint|p1a: Breloom",
        )

        self.assertEqual(
            replay.substitute_health_state.get("p1"),
            "absent",
            "a fainted mon's Substitute must not survive as a phantom effect",
        )
        self.assertEqual(
            replay.substitute_min_depletion.get("p1"),
            0,
            "a bound surviving the faint would constrain the NEXT Substitute by damage dealt "
            "to a Pokemon that is already gone",
        )

    def test_a_broken_substitute_also_resets_the_count(self):
        """The `-end` path, kept as its own case now that the two are separated."""

        replay = self._state(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-end|p1a: Breloom|Substitute",
            "|turn|3",
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|turn|4",
        )

        self.assertEqual(replay.substitute_health_state.get("p1"), "full")
        self.assertEqual(
            replay.substitute_min_depletion.get("p1"),
            0,
            "carrying the broken Substitute's hits forward would bound the wrong effect",
        )


# The payload builder requires a non-empty acting-player team, so the seam test needs a real
# request rather than an empty stub -- `_request_materialization_rows` raises otherwise.
_SEAM_REQUEST = {
    "side": {
        "pokemon": [
            {
                "ident": "p1: Breloom",
                "details": "Breloom, L79, M",
                "condition": "100/100",
                "active": True,
                "moves": ["substitute", "machpunch", "spore", "seismictoss"],
            },
            {
                "ident": "p1: Swampert",
                "details": "Swampert, L79, M",
                "condition": "100/100",
                "active": False,
                "moves": ["earthquake", "surf", "icebeam", "protect"],
            },
        ]
    }
}


@requires_showdown()
class SubstituteProducerConsumerSeamTests(unittest.TestCase):
    """The one test that crosses the producer -> payload -> consumer seam.

    Review found three mutations that make this PR a COMPLETE NO-OP in production while the
    whole suite stays green: delete `"substituteMinDepletion"` from the payload builder,
    misspell the key, or change its default. The producer tests read
    `replay.substitute_min_depletion`; the consumer tests hand-inject the payload key. Nothing
    joined them, so the single line the entire change depends on was unasserted -- exactly
    the "tests that assert nothing" shape this repo keeps being bitten by.

    This drives real protocol lines through the production parser, builds the payload with
    the production builder, and feeds THAT payload to the world constructor.
    """

    @classmethod
    def setUpClass(cls):
        cls.dex = load_showdown_dex(showdown_root())

    def _payload_from_transcript(self, *extra):
        from pokezero.local_showdown import (
            PublicBattleMaterializationState,
            _public_materialization_payload,
        )
        from pokezero.belief import PublicBattleBeliefEngine

        replay = parse_showdown_replay(
            _lines(*extra), battle_id="battle-sub-seam-1", complete_prefix=True
        )
        state = PublicBattleMaterializationState(
            player_id="p1",
            format_id="gen3randombattle",
            observation_format_id="gen3randombattle",
            replay=replay,
            belief_engine=PublicBattleBeliefEngine.from_events(
                replay.public_events, format_id="gen3randombattle"
            ),
            self_request=_SEAM_REQUEST,
            self_move_states={},
            self_initial_request=_SEAM_REQUEST,
        )
        return _public_materialization_payload(state), replay

    def test_the_parsers_hit_count_reaches_the_payload(self):
        """Fails if the payload key is deleted, renamed, or defaulted differently."""

        payload, replay = self._payload_from_transcript(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
        )

        self.assertEqual(replay.substitute_min_depletion.get("p1"), 1)
        self.assertEqual(
            payload["sides"]["p1"].get("substituteMinDepletion"),
            1,
            "the parser's count must reach the payload under this exact key -- without it "
            "every world refuses and the whole change is a no-op in production",
        )
        self.assertEqual(payload["sides"]["p1"].get("substituteHealthState"), "unknown")

    def test_proven_depletion_also_reaches_the_payload(self):
        payload, replay = self._payload_from_transcript(
            "|move|p1a: Breloom|Substitute|p1a: Breloom",
            "|-start|p1a: Breloom|Substitute",
            "|move|p2a: Blissey|Seismic Toss|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|2",
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
            "|-activate|p1a: Breloom|Substitute|[damage]",
            "|turn|3",
        )

        self.assertEqual(replay.substitute_health_state.get("p1"), "unknown")
        self.assertEqual(
            payload["sides"]["p1"].get("substituteMinDepletion"),
            78,
            "the Seismic Toss 77 must survive the later unknown hit and reach the payload, "
            "plus 1 for that hit -- 78, not 1",
        )


@requires_showdown()
class SubstituteHealthSamplingTests(unittest.TestCase):
    """The CONSUMER side: bounded range sampled, unbounded case still refused."""

    @classmethod
    def setUpClass(cls):
        cls.dex = load_showdown_dex(showdown_root())

    def _sub_payload(self, *, hits=None):
        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Substitute"]
        payload["sides"]["p2"]["substituteHealthState"] = "unknown"
        payload["sides"]["p2"]["substituteDepletion"] = None
        if hits is not None:
            payload["sides"]["p2"]["substituteMinDepletion"] = hits
        return payload

    def _build(self, payload, *, rng=None):
        import random

        return battle_spec_from_payload(
            payload,
            _override(),
            dex=self.dex,
            approximate_substitute_health=True,
            # A REAL rng by default. Review found the first version of this helper defaulted
            # to `None`, which took the take-the-maximum branch -- so the `randint` path the
            # whole PR is about was never exercised by any consumer test.
            rng=random.Random(0) if rng is None else rng,
        )

    def test_a_bounded_unknown_health_is_sampled_not_refused(self):
        side = self._build(self._sub_payload(hits=1)).spec.side_two
        initial = side.pokemon[0].maxhp // 4

        self.assertIn("substitute", side.volatile_statuses)
        self.assertGreaterEqual(side.substitute_health, 1, "the Substitute survived the hit")
        self.assertLessEqual(
            side.substitute_health,
            initial - 1,
            "one non-breaking hit removed at least 1 HP, so full health is inconsistent",
        )

    def test_no_rng_is_an_error_rather_than_a_silent_maximum(self):
        """Defaulting to `upper` biases every world toward a near-full Substitute.

        Review found the original consumer test was itself taking that branch, so the
        sampler was untested. There is no live rng-less caller today, so this closes a
        latent trap rather than a live path.
        """

        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(
                self._sub_payload(hits=1),
                _override(),
                dex=self.dex,
                approximate_substitute_health=True,
                rng=None,
            )
        self.assertEqual(caught.exception.reason, "substitute_health_unknown")

    def test_the_lower_bound_is_asserted_without_relying_on_seed_luck(self):
        """A sampled 0 is a BROKEN Substitute with the volatile still set -- silent.

        Review's `randint(0, upper)` mutant survived because the 40-seed loop happened not to
        draw 0 for seeds 0..39. A stub rng that returns the minimum of whatever range it is
        given makes the lower bound deterministic instead of lucky.
        """

        class _MinRng:
            def __init__(self):
                self.calls = []

            def randint(self, low, high):
                self.calls.append((low, high))
                return low

        rng = _MinRng()
        side = self._build(self._sub_payload(hits=1), rng=rng).spec.side_two

        self.assertEqual(rng.calls and rng.calls[0][0], 1, "the low end must be 1, never 0")
        self.assertEqual(side.substitute_health, 1)
        self.assertGreater(
            side.substitute_health, 0, "0 HP with the volatile set is a broken sub, not a sub"
        )

    def test_every_sample_stays_inside_the_consistent_range(self):
        """Sampled per world, so the range is what must hold -- not one value."""

        import random

        hits = 3
        payload = self._sub_payload(hits=hits)
        initial = self._build(payload).spec.side_two.pokemon[0].maxhp // 4
        seen = set()
        for seed in range(40):
            side = self._build(payload, rng=random.Random(seed)).spec.side_two
            self.assertGreaterEqual(side.substitute_health, 1)
            self.assertLessEqual(side.substitute_health, initial - hits)
            seen.add(side.substitute_health)

        self.assertGreater(
            len(seen), 1, "an rng-driven sampler must actually vary across worlds"
        )

    def test_proven_exact_depletion_bounds_the_sample(self):
        """The correctness defect review MEASURED, in both orderings.

        `substitute_depletion` is cleared when provenance degrades to `unknown`, so a proven
        exact depletion used to VANISH: Seismic Toss for 100 then Ice Beam gave
        `upper = initial - 1` while the record proved remaining HP <= initial - 100 - 1. On the
        measured live numbers (initial 162) that put 62% of sampled worlds outside the allowed
        range, with a mean sample ABOVE the maximum possible -- worse than the refusal it
        replaced, and silent.
        """

        import random

        # Dragon Rage's fixed 40, not the live case's Seismic Toss 100: this fixture's world
        # has maxhp 387 so `initial` is 96, and 100 would make the world incompatible rather
        # than exercising the bound. Both are public fixed-damage gen 3 moves.
        proven = 40
        hits = 1
        payload = self._sub_payload(hits=hits)
        payload["sides"]["p2"]["substituteMinDepletion"] = proven + hits
        initial = self._build(payload).spec.side_two.pokemon[0].maxhp // 4
        self.assertGreater(
            initial - proven - hits, 0, "the fixture must leave a live Substitute"
        )

        for seed in range(40):
            side = self._build(payload, rng=random.Random(seed)).spec.side_two
            self.assertLessEqual(
                side.substitute_health,
                initial - proven - hits,
                "proven damage stays proven: an unknown hit must not restore it",
            )
            self.assertGreaterEqual(side.substitute_health, 1)

    def test_no_hit_count_still_refuses_under_the_SAME_reason_code(self):
        """Strictly additive: an unbounded `unknown` behaves exactly as before.

        Both the missing key and an explicit zero take this path. Asserting the reason
        code, not merely that it raises, is the point -- a renamed refusal would read as a
        new class in the next era's crosstab.
        """

        for hits in (None, 0):
            with self.subTest(hits=hits):
                with self.assertRaises(EngineWorldUnsupported) as caught:
                    self._build(self._sub_payload(hits=hits))
                self.assertEqual(caught.exception.reason, "substitute_health_unknown")

    def test_proven_depletion_alone_can_make_a_world_incompatible(self):
        """The `upper < 1` guard must account for proven depletion, not only hits."""

        payload = self._sub_payload(hits=1)
        payload["sides"]["p2"]["substituteMinDepletion"] = 10_000
        with self.assertRaises(EngineWorldUnsupported) as caught:
            self._build(payload)
        self.assertEqual(
            caught.exception.reason, "substitute_depletion_world_incompatible"
        )

    def test_a_malformed_bound_is_treated_as_absent_rather_than_trusted(self):
        """`True` is an `int` in Python, and a negative bound is not a bound.

        Both are unreachable in-process -- the only producer is the parser, whose dicts always
        carry `p1`/`p2` -- so these are input-validation guards rather than live paths. Pinned
        because a `bool` slipping through would make `upper = initial - 1` from a flag, and a
        negative one would WIDEN the range past `initial`, which is the opposite of a bound.
        """

        for bad in (True, False, -5, "3", 1.5, None):
            with self.subTest(bad=bad):
                payload = self._sub_payload()
                payload["sides"]["p2"]["substituteMinDepletion"] = bad
                with self.assertRaises(EngineWorldUnsupported) as caught:
                    self._build(payload)
                self.assertEqual(
                    caught.exception.reason,
                    "substitute_health_unknown",
                    "a malformed bound must fall back to the unchanged refusal, never be used",
                )

    def test_more_hits_than_the_world_can_absorb_refuses_that_world_only(self):
        """A world whose sampled max HP cannot absorb the public hits is inconsistent.

        This is a per-WORLD refusal, not a fallback: the retry budget samples another. It
        must not be reported as `substitute_health_unknown`, because the public record is
        fine here and the sampled world is the thing at fault.
        """

        payload = self._sub_payload(hits=10_000)
        with self.assertRaises(EngineWorldUnsupported) as caught:
            self._build(payload)
        self.assertEqual(
            caught.exception.reason, "substitute_depletion_world_incompatible"
        )


if __name__ == "__main__":
    unittest.main()
