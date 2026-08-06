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

So after `hits` such hits, remaining health lies in `[1, initial - hits]` where `initial`
is `floor(maxhp / 4)`. Both ends are informative and the window TIGHTENS with each hit.
That is a bounded range, not the open guess `[1, initial]` would be.

STRICTLY ADDITIVE. A payload that does not carry `substituteUnknownHits` refuses exactly
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
            replay.substitute_unknown_hits.get("p1"),
            1,
            "the hit must be COUNTED even though its damage is unknowable -- the count is "
            "what bounds the sampled health instead of refusing the world",
        )

    def test_hits_accumulate_so_the_window_tightens(self):
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

        self.assertEqual(replay.substitute_unknown_hits.get("p1"), 2)

    def test_a_fixed_damage_move_stays_exact_and_counts_nothing(self):
        """Seismic Toss is public and deterministic, so provenance stays `exact`.

        This is the boundary that keeps the counter from cannibalising the precise path:
        an exact depletion is strictly better than a sampled range and must not be
        downgraded to one.
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
            replay.substitute_unknown_hits.get("p1"),
            0,
            "an exact hit must not be counted as unknown, or the exact path loses HP twice",
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
        self.assertEqual(replay.substitute_unknown_hits.get("p1"), 0)

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
            "|move|p2a: Blissey|Ice Beam|p1a: Breloom",
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
            replay.substitute_unknown_hits.get("p1"),
            0,
            "a count surviving the switch would bound the NEXT Substitute by hits that "
            "landed on a previous one",
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
            replay.substitute_unknown_hits.get("p1"),
            0,
            "carrying the broken Substitute's hits forward would bound the wrong effect",
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
            payload["sides"]["p2"]["substituteUnknownHits"] = hits
        return payload

    def _build(self, payload, *, rng=None):
        return battle_spec_from_payload(
            payload,
            _override(),
            dex=self.dex,
            approximate_substitute_health=True,
            rng=rng,
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
