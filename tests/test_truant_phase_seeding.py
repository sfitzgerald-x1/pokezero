"""gen3 Truant loaf phase: parser state machine + world payload consumption.

gen3 OWNS Truant (`data/mods/gen3/abilities.ts`, `onStart: undefined`) and models it as a
free-running boolean rather than base's volatile::

    onSwitchIn(p) { p.truantTurn = this.turn !== 0; }
    onBeforeMove(p) { if (p.truantTurn) { add('cant', 'ability: Truant'); return false; } }
    onResidualOrder: 27
    onResidual(p) { p.truantTurn = !p.truantTurn; }

The bit flips at EVERY residual, unconditionally. The rule it replaced — "moved last round
-> loafs now" — is a proxy for the bit, and the first turn a holder is stopped by something
OTHER than Truant (sleep, paralysis, flinch, freeze, recharge, a switch) the two disagree and
the parity stays inverted for the rest of the stint.

**Every expectation below is transcribed from a `gen3customgame` probe, not derived.** That
distinction earned its place here: the composed derivation for a traced holder predicted it
loafs on its first move turn, and the probe showed a turn-0 tracer ACTS. Four formulations of
this fix fought each other before the probes settled it.

Probe results the pins encode:

| scenario | first move turn |
| --- | --- |
| native lead (turn 0)            | ACTS  |
| native mid-battle switch        | ACTS  |
| traced at turn 0                | ACTS  |
| traced by a one-sided action switch | LOAFS |
| traced during simultaneous switches | ACTS |
| traced by a pre-upkeep forced replacement | LOAFS |
| traced by a post-upkeep forced replacement | ACTS |
| native post-residual faint replacement | LOAFS |

The Trace rows prove that line position alone is insufficient: event-queue membership changes
whether copied Truant receives the residual. Traced holders therefore remain UNKNOWN until an
own move or Truant ``cant`` line publicly anchors the phase. The last row is the native
replacement guard: a holder entering between `|upkeep|` and `|turn|` missed that turn's
residual, so the following turn marker must not double-count it.
"""

from __future__ import annotations

import unittest

from pokezero.local_showdown import _public_materialization_payload  # noqa: F401  (import guard)
from pokezero.showdown import _ReplayParser

SLAKING = "|switch|p1a: Slaking|Slaking, L80, M|362/362"
OPP = "|switch|p2a: Snorlax|Snorlax, L80, M|400/400"


def _parse(lines: list[str]) -> _ReplayParser:
    p = _ReplayParser("battle-gen3randombattle-1")
    p.feed(["|player|p1|Us|", "|player|p2|Them|", *lines])
    return p


class NativeTruantPhaseTest(unittest.TestCase):
    """Native holders (`slakoth`/`slaking`, both mono-ability, so species is decisive)."""

    def test_a_lead_holder_acts_on_its_first_move_turn(self) -> None:
        # `truantTurn = this.turn !== 0` gives False at turn 0, and there is NO end-of-turn-0
        # residual to flip it, so turn 1 acts. Skipping that flip is a deliberate special
        # case; without it a lead's parity is inverted for the whole battle.
        p = _parse([SLAKING, OPP, "|turn|1"])
        self.assertIs(p.truant_phase["p1"], False)

    def test_a_lead_holder_loafs_on_the_second_turn(self) -> None:
        p = _parse([SLAKING, OPP, "|turn|1", "|move|p1a: Slaking|Tackle|p2a: Snorlax",
                    "|upkeep", "|turn|2"])
        self.assertIs(p.truant_phase["p1"], True)

    def test_a_mid_battle_switch_in_also_acts_on_its_first_move_turn(self) -> None:
        # Seeded True (turn != 0), then the end-of-turn residual flips it to False. The
        # `turn != 0` term exists precisely to make this case agree with the lead.
        p = _parse(["|switch|p1a: Snorlax|Snorlax, L80, M|400/400", OPP, "|turn|1",
                    "|move|p1a: Snorlax|Tackle|p2a: Snorlax", "|upkeep", "|turn|2",
                    SLAKING, "|upkeep", "|turn|3"])
        self.assertIs(p.truant_phase["p1"], False)

    def test_the_phase_alternates_every_turn_regardless_of_what_happened(self) -> None:
        # The point of the whole change: the bit is a toggle, not a consequence of acting.
        # Here the holder is ASLEEP throughout and never moves, and the parity still flips.
        lines = [SLAKING, OPP, "|turn|1"]
        seen = []
        for turn in range(2, 7):
            lines += ["|cant|p1a: Slaking|slp", "|upkeep", f"|turn|{turn}"]
            seen.append(_parse(list(lines)).truant_phase["p1"])
        self.assertEqual(seen, [True, False, True, False, True])

    def test_a_non_holder_switching_in_clears_the_phase(self) -> None:
        p = _parse([SLAKING, OPP, "|turn|1",
                    "|switch|p1a: Snorlax|Snorlax, L80, M|400/400", "|upkeep", "|turn|2"])
        self.assertIsNone(p.truant_phase["p1"])


class TruantAnchorTest(unittest.TestCase):
    """The sim publishes the answer; anchor on it rather than trusting the flip count."""

    def test_a_truant_cant_line_anchors_the_loafing_phase(self) -> None:
        p = _parse([SLAKING, OPP, "|turn|1", "|cant|p1a: Slaking|ability: Truant"])
        self.assertIs(p.truant_phase["p1"], True)

    def test_a_holders_own_move_anchors_the_acting_phase(self) -> None:
        p = _parse([SLAKING, OPP, "|turn|1", "|move|p1a: Slaking|Tackle|p2a: Snorlax"])
        self.assertIs(p.truant_phase["p1"], False)

    def test_an_anchor_corrects_a_drifted_parity(self) -> None:
        # An anchor is ground truth for its turn, so a derivation that has drifted is
        # corrected at the first public evidence rather than staying wrong until switch-out.
        drifted = [SLAKING, OPP, "|turn|1", "|cant|p1a: Slaking|ability: Truant"]
        p = _parse(drifted)
        self.assertIs(p.truant_phase["p1"], True)
        p2 = _parse(drifted + ["|upkeep", "|turn|2"])
        self.assertIs(p2.truant_phase["p1"], False)

    def test_a_called_move_does_not_anchor(self) -> None:
        # Sleep Talk's callee is not an independent action; the caller's line already
        # anchored the turn. Mirrors the last_used_move caller/callee split.
        p = _parse([SLAKING, OPP, "|turn|1",
                    "|move|p1a: Slaking|Sleep Talk|p1a: Slaking",
                    "|move|p1a: Slaking|Tackle|p2a: Snorlax|[from]Sleep Talk"])
        self.assertIs(p.truant_phase["p1"], False)


class TracedTruantTest(unittest.TestCase):
    """Trace holder detection is public; its initial boolean phase is not."""

    TRACE = "|-ability|p1a: Porygon2|Truant|Trace|[from] ability: Trace|[of] p2a: Slaking"
    POR = "|switch|p1a: Porygon2|Porygon2, L80|267/267"

    def test_a_tracer_is_recognised_but_left_unknown(self) -> None:
        p = _parse([self.POR, "|switch|p2a: Slaking|Slaking, L80, M|362/362", self.TRACE])
        self.assertEqual(p.traced_ability["p1"], "truant")
        self.assertIsNone(p.truant_phase["p1"])

    def test_retained_identity_3400443_step_2_anchors_only_on_cant(self) -> None:
        # Extracted current-source protocol: a one-sided switch traces Truant before upkeep,
        # then Showdown loafs at the next decision. The acquisition remains unknown; the
        # retained row's own ``cant`` line is the first honest phase anchor.
        p = _parse([
            self.POR,
            self.TRACE,
            "|move|p2a: Slaking|Return|p1a: Porygon2",
            "|-damage|p1a: Porygon2|49/267",
            "|-heal|p1a: Porygon2|65/267|[from] item: Leftovers",
            "|upkeep",
            "|turn|2",
        ])
        self.assertIsNone(p.truant_phase["p1"])
        p.feed([
            "|switch|p2a: Nidoqueen|Nidoqueen, L82, F|282/282",
            "|cant|p1a: Porygon2|ability: Truant",
        ])
        self.assertIs(p.truant_phase["p1"], True)

    def test_retained_identity_3400443_step_69_forced_replacement_anchor(self) -> None:
        # Extracted current-source protocol: a move KO creates a pre-upkeep forced replacement.
        # Trace is still unknown until both holders publicly loaf at the next boundary.
        p = _parse([
            "|switch|p1a: Moltres|Moltres, L80|300/300",
            "|switch|p2a: Slaking|Slaking, L80, M|362/362",
            "|move|p2a: Slaking|Return|p1a: Moltres",
            "|-damage|p1a: Moltres|0 fnt",
            "|faint|p1a: Moltres",
            self.POR,
            self.TRACE,
            "|upkeep",
            "|turn|59",
        ])
        self.assertIsNone(p.truant_phase["p1"])
        p.feed([
            "|cant|p2a: Slaking|ability: Truant",
            "|cant|p1a: Porygon2|ability: Truant",
        ])
        self.assertIs(p.truant_phase["p1"], True)

    def test_current_source_2200291_step_41_stays_unknown_until_it_acts(self) -> None:
        # Current-source control for the measured Z13.3 withdrawal. Both sides switch in on
        # the same action boundary; despite Trace preceding upkeep, Porygon2 ACTS next turn.
        # Any derived pre-upkeep boolean reintroduces this exact divergence.
        p = _parse([
            "|switch|p2a: Slaking|Slaking, L78, F|294/362",
            self.POR,
            self.TRACE,
            "|upkeep",
            "|turn|38",
        ])
        self.assertIsNone(p.truant_phase["p1"])
        p.feed([
            "|move|p2a: Slaking|Earthquake|p1a: Porygon2",
            "|move|p1a: Porygon2|Ice Beam|p2a: Slaking",
        ])
        self.assertIs(p.truant_phase["p1"], False)
        p.feed(["|upkeep", "|turn|39"])
        self.assertIs(p.truant_phase["p1"], True)

    def test_an_anchor_establishes_the_traced_phase(self) -> None:
        p = _parse([
            self.POR,
            "|switch|p2a: Slaking|Slaking, L80, M|362/362",
            self.TRACE,
            "|upkeep",
            "|turn|2",
            "|cant|p1a: Porygon2|ability: Truant",
        ])
        self.assertIs(p.truant_phase["p1"], True)

    def test_tracing_something_else_clears_the_holder_state(self) -> None:
        p = _parse([
            SLAKING,
            OPP,
            "|turn|1",
            self.POR,
            "|-ability|p1a: Porygon2|Levitate|Trace|[from] ability: Trace|[of] p2a: Snorlax",
        ])
        self.assertIsNone(p.truant_phase["p1"])

    def test_trace_without_an_active_truant_copy_makes_no_phase_claim(self) -> None:
        p = _parse([
            self.POR,
            "|switch|p2a: Snorlax|Snorlax, L80, M|400/400",
            "|-ability|p1a: Porygon2|Immunity|Trace|[from] ability: Trace|[of] p2a: Snorlax",
        ])
        self.assertIsNone(p.truant_phase["p1"])

    def test_reentry_and_retrace_discards_the_previous_anchor(self) -> None:
        p = _parse([
            self.POR,
            "|switch|p2a: Slaking|Slaking, L80, M|362/362",
            self.TRACE,
            "|cant|p1a: Porygon2|ability: Truant",
        ])
        self.assertIs(p.truant_phase["p1"], True)
        p.feed(["|switch|p1a: Snorlax|Snorlax, L80, M|400/400"])
        self.assertIsNone(p.truant_phase["p1"])
        p.feed([self.POR, self.TRACE])
        self.assertIsNone(p.truant_phase["p1"])


class ReplacementGuardTest(unittest.TestCase):
    """The sub-case that made the first ship understate its deviation."""

    def test_a_post_residual_replacement_loafs_on_its_first_move_turn(self) -> None:
        # Sim-probed: a holder replacing a mon that fainted at upkeep LOAFS, because that
        # turn's residual ran before it arrived. Without the guard the `|turn|` flip
        # double-counts and the parity is inverted for the whole stint -- which is what
        # produced the five newly-divergent rows the reviewer found.
        p = _parse(["|switch|p1a: Shedinja|Shedinja, L80|1/1", OPP, "|turn|1",
                    "|faint|p1a: Shedinja", "|upkeep",
                    SLAKING,          # replacement enters AFTER upkeep
                    "|turn|2"])
        self.assertIs(p.truant_phase["p1"], True)

    def test_a_switch_taken_as_the_turns_action_still_acts(self) -> None:
        # The control that gives the guard its meaning: same seed value, opposite outcome,
        # and the ONLY difference is which side of the residual the holder entered on.
        p = _parse(["|switch|p1a: Snorlax|Snorlax, L80, M|400/400", OPP, "|turn|1",
                    SLAKING,          # switch as the ACTION, before upkeep
                    "|upkeep", "|turn|2"])
        self.assertIs(p.truant_phase["p1"], False)

    def test_post_upkeep_guard_survives_snapshot_restore(self) -> None:
        live = _parse([
            "|switch|p1a: Shedinja|Shedinja, L80|1/1",
            OPP,
            "|turn|1",
            "|faint|p1a: Shedinja",
            "|upkeep",
            SLAKING,
        ])
        snapshot = live.snapshot()
        self.assertTrue(snapshot.post_upkeep_window)
        self.assertEqual(snapshot.truant_skip_next_flip, ("p1",))

        restored = _ReplayParser.from_snapshot(snapshot)
        live.feed(["|turn|2"])
        restored.feed(["|turn|2"])
        self.assertIs(live.truant_phase["p1"], True)
        self.assertEqual(restored.snapshot(), live.snapshot())

    def test_snapshot_before_post_upkeep_replacement_preserves_window(self) -> None:
        live = _parse([
            "|switch|p1a: Shedinja|Shedinja, L80|1/1",
            OPP,
            "|turn|1",
            "|faint|p1a: Shedinja",
            "|upkeep",
        ])
        restored = _ReplayParser.from_snapshot(live.snapshot())
        for parser in (live, restored):
            parser.feed([SLAKING, "|turn|2"])
        self.assertIs(restored.truant_phase["p1"], True)
        self.assertEqual(restored.snapshot(), live.snapshot())


class WorldPayloadTest(unittest.TestCase):
    """The world half: payload beats the caller-side proxy, and None is not False."""

    def _seed(self, phase, truant_loafs=False):
        from pokezero.engine_world import _truant_volatile_decision

        return _truant_volatile_decision({"truantPhase": phase}, truant_loafs)

    def test_payload_true_loafs(self) -> None:
        self.assertTrue(self._seed(True))

    def test_payload_false_beats_a_true_proxy(self) -> None:
        # The whole point of preferring the payload: the proxy is the thing being replaced,
        # so a False payload must override a True proxy rather than OR with it.
        self.assertFalse(self._seed(False, truant_loafs=True))

    def test_unknown_falls_back_to_the_proxy(self) -> None:
        # None means "no holder, or a truncated prefix whose switch-in was never seen".
        # Falling back preserves previous behaviour instead of asserting an acting phase.
        self.assertTrue(self._seed(None, truant_loafs=True))
        self.assertFalse(self._seed(None, truant_loafs=False))


if __name__ == "__main__":
    unittest.main()
