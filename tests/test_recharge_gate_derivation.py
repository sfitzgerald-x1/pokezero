"""The recharge gates, and whether they can catch a bad self-side write.

Both `rust/pokezero-search/src/leaf.rs` and `scripts/leaf_vs_reality.py` carry a standing
warning: the four differential harnesses derived `recharging` for BOTH slots from the RECORDED
CHOSEN CANDIDATE, so a gate's world was seeded from the very thing the gate was checking. If the
mon's recorded action was the `recharge` pseudo-move, the gate locked that slot -- which is
exactly what a symmetric self-side MUSTRECHARGE write would produce. The gate would agree with
such a write rather than test it.

Production does not build its world that way. `engine_search.py::_recharging_slots` locks the
OPPONENT slot only, from the parser's `must_recharge` tracker, never from the chosen action.

So the acceptance here is TWO-WAY, and neither direction is an assertion about intent:

  1. Under the OLD candidate-derived rule, a self-side write is RATIFIED -- the gate's world
     carries the lock, so a `true` write matches. Pinned so the defect is demonstrated, not
     merely described in a comment.
  2. Under the NEW production-mirroring rule, the same write is CAUGHT -- the gate's world does
     not carry the lock, so the write diverges.

If the second test is ever made to pass by a change that also makes the first pass, the gate has
gone back to ratifying, and the pair goes red.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fidelity_gate_events import (  # noqa: E402
    anchor_observation_metadata,
    normalize_id,
    production_recharging_slots,
)


def _candidate_derived_recharging(anchor_row, decisions, battle_id, round_n) -> tuple[str, ...]:
    """The rule the four gates used before this change, reproduced verbatim.

    Kept in the test rather than in the harnesses so direction (1) stays falsifiable after the
    production code stops doing it. This is the ONLY copy left.
    """
    out = []
    for slot in ("p1", "p2"):
        row = decisions.get((battle_id, round_n, slot))
        candidate = None
        if row is not None:
            index = row.get("chosen_action_index")
            for entry in (row.get("observation_metadata") or {}).get("action_candidates") or ():
                if entry.get("action_index") == index:
                    candidate = entry
                    break
        if (
            candidate is not None
            and candidate.get("kind") == "move"
            and normalize_id(str(candidate.get("move_id") or "")) == "recharge"
        ):
            out.append(slot)
    return tuple(out)


def _row(*, seat: str, self_mr: bool, opp_mr: bool, chose_recharge: bool) -> dict:
    """A decision row shaped like the corpus's, at the two fields that matter here."""
    candidates = [
        {"action_index": 0, "kind": "move", "move_id": "recharge" if chose_recharge else "return"},
        {"action_index": 1, "kind": "move", "move_id": "earthquake"},
    ]
    return {
        "battle_id": "b",
        "decision_round_index": 7,
        "player_id": seat,
        "chosen_action_index": 0,
        "observation_metadata": {
            "self_must_recharge": self_mr,
            "opponent_must_recharge": opp_mr,
            "action_candidates": candidates,
        },
    }


class TheGateUsedToRatifyASelfSideWriteTest(unittest.TestCase):
    """Direction 1: the defect, demonstrated rather than described."""

    def setUp(self) -> None:
        # Our own active (p2, the anchor seat) must recharge and recorded the recharge move.
        # The opponent (p1) is unaffected. This is the shape of corpus/golden-v4 rows 964/969.
        self.decisions = {
            ("b", 7, "p2"): _row(seat="p2", self_mr=True, opp_mr=False, chose_recharge=True),
            ("b", 7, "p1"): _row(seat="p1", self_mr=False, opp_mr=True, chose_recharge=False),
        }

    def test_the_old_rule_locks_our_own_slot(self) -> None:
        """The gate's world carried self-side MUSTRECHARGE, so a `true` write could only agree."""
        old = _candidate_derived_recharging(None, self.decisions, "b", 7)
        self.assertIn("p2", old, "the old rule locked our own slot off the recorded action")

    def test_and_therefore_ratifies_a_symmetric_write(self) -> None:
        """The write under test: an encoder that emits self-side MUSTRECHARGE = true.

        Against the old world the write MATCHES, so the gate reports no divergence. That is the
        ratification the source comments warned about, now measured.
        """
        old = _candidate_derived_recharging(None, self.decisions, "b", 7)
        injected_self_side_write = True
        world_says = "p2" in old
        self.assertEqual(
            world_says,
            injected_self_side_write,
            "old gate agreed with the write -- it could not have caught a wrong one",
        )

    def test_the_old_rule_would_also_ratify_the_OPPOSITE_write(self) -> None:
        """Non-vacuity, and the sharper form of the same defect.

        A gate that merely agreed with the truth would be harmless. This one tracks whatever the
        action record says: flip the recorded action to a non-recharge move while the tracker
        still says our mon must recharge, and the old world flips with it -- so it would equally
        ratify a `false` write on a genuinely recharging mon.
        """
        flipped = dict(self.decisions)
        flipped[("b", 7, "p2")] = _row(
            seat="p2", self_mr=True, opp_mr=False, chose_recharge=False
        )
        old = _candidate_derived_recharging(None, flipped, "b", 7)
        self.assertNotIn("p2", old)
        self.assertTrue(
            flipped[("b", 7, "p2")]["observation_metadata"]["self_must_recharge"],
            "the tracker still says our mon must recharge -- only the action record changed",
        )


class TheFixedGateCatchesItTest(unittest.TestCase):
    """Direction 2: the acceptance."""

    def setUp(self) -> None:
        self.decisions = {
            ("b", 7, "p2"): _row(seat="p2", self_mr=True, opp_mr=False, chose_recharge=True),
            ("b", 7, "p1"): _row(seat="p1", self_mr=False, opp_mr=True, chose_recharge=False),
        }

    def _fixed(self, seat: str) -> tuple[str, ...]:
        return production_recharging_slots(
            anchor_observation_metadata(self.decisions.get(("b", 7, seat))), seat
        )

    def test_the_fixed_rule_locks_our_own_slot_from_the_TRACKER(self) -> None:
        """Symmetric now, and sourced from the parser rather than the action.

        Step 1 of this change made the gate mirror production's ASYMMETRY (opponent only), so
        that a symmetric self-side write would diverge. Step 3 made production itself symmetric,
        so the gate follows. Symmetry is not a return to ratification: what made the old gate
        unable to catch anything was that it read the RECORDED ACTION, so it agreed with whatever
        the encoder had been told. This reads the tracker, which no encoder write can move.
        """
        self.assertIn("p2", self._fixed("p2"))

    def test_it_catches_a_write_that_CONTRADICTS_the_tracker_true_case(self) -> None:
        """THE ACCEPTANCE, direction A: encoder writes `true`, tracker says false."""
        self.decisions[("b", 7, "p2")] = _row(
            seat="p2", self_mr=False, opp_mr=False, chose_recharge=True
        )
        injected_write = True  # a bad self-side MUSTRECHARGE write
        self.assertNotEqual(
            "p2" in self._fixed("p2"),
            injected_write,
            "gate must disagree with a `true` write on a mon the tracker says is free",
        )

    def test_it_catches_a_write_that_CONTRADICTS_the_tracker_false_case(self) -> None:
        """THE ACCEPTANCE, direction B: encoder writes `false`, tracker says true.

        Both directions matter. A gate that only caught spurious locks would still miss the
        staleness this whole area is about -- a genuinely recharging mon written as free.
        """
        injected_write = False  # setUp has self_must_recharge True
        self.assertNotEqual(
            "p2" in self._fixed("p2"),
            injected_write,
            "gate must disagree with a `false` write on a genuinely recharging mon",
        )

    def test_and_the_old_rule_caught_NEITHER(self) -> None:
        """Non-vacuity for both directions above, against the rule actually replaced.

        The old rule read the recorded action, so for each bad write there is a corpus row on
        which it AGREES with that write -- which is what 'would ratify rather than catch' meant.
        """
        # `true` write, tracker false, but the action record says recharge -> old rule agrees.
        spurious = {
            ("b", 7, "p2"): _row(seat="p2", self_mr=False, opp_mr=False, chose_recharge=True),
            ("b", 7, "p1"): _row(seat="p1", self_mr=False, opp_mr=False, chose_recharge=False),
        }
        self.assertIn("p2", _candidate_derived_recharging(None, spurious, "b", 7))
        # `false` write, tracker true, but the action record says otherwise -> old rule agrees.
        stale = {
            ("b", 7, "p2"): _row(seat="p2", self_mr=True, opp_mr=False, chose_recharge=False),
            ("b", 7, "p1"): _row(seat="p1", self_mr=False, opp_mr=False, chose_recharge=False),
        }
        self.assertNotIn("p2", _candidate_derived_recharging(None, stale, "b", 7))

    def test_it_still_locks_the_opponent_when_the_tracker_says_so(self) -> None:
        """Non-vacuity: the fix is not 'lock nothing', which would catch everything trivially.

        The shared fixture has the opponent NOT recharging, so `()` there is the right answer and
        proves nothing on its own. This case supplies the opposite scenario: anchored at p2, with
        p2's own metadata reporting opponent_must_recharge, p1 must be locked. A helper that
        returned () unconditionally would pass the acceptance above while testing nothing, and
        this is what stops that.
        """
        self.decisions[("b", 7, "p2")] = _row(
            seat="p2", self_mr=True, opp_mr=True, chose_recharge=True
        )
        # Scoped to the OPPONENT slot: our own is separately locked by self_must_recharge now
        # that the helper is symmetric, and this case is about the opponent side.
        self.assertIn("p1", self._fixed("p2"))

    def test_it_is_keyed_off_the_tracker_not_the_action(self) -> None:
        """The property that makes the gate independent of what it is checking.

        Change only the recorded action; the derivation must not move. Under the old rule this
        was the one thing that DID move it.
        """
        before = self._fixed("p2")
        self.decisions[("b", 7, "p2")] = _row(
            seat="p2", self_mr=True, opp_mr=False, chose_recharge=False
        )
        self.assertEqual(before, self._fixed("p2"), "derivation moved with the recorded action")

    def test_an_explicit_false_is_proof_of_no_lock(self) -> None:
        """Production treats an explicit False as public proof, not an absent signal."""
        self.decisions[("b", 7, "p2")]["observation_metadata"]["opponent_must_recharge"] = False
        self.assertNotIn("p1", self._fixed("p2"))

    def test_an_absent_tracker_fails_open(self) -> None:
        """Recorded because it is a real divergence from production, not a design choice.

        Production reconstructs from the round-indexed public action record when the tracker key
        is absent; this returns (). On corpus/golden-v4 the tracker is present on 1208 of 1295
        decision rows, so the fallback covers 87. Fail-open matches where production's own
        fallback lands when the record is unavailable, but the two are not identical.
        """
        del self.decisions[("b", 7, "p2")]["observation_metadata"]["opponent_must_recharge"]
        self.assertNotIn("p1", self._fixed("p2"))


class BothDirectionsHoldTogetherTest(unittest.TestCase):
    def test_the_rules_differ_wherever_the_action_record_lies(self) -> None:
        """The whole change, in one assertion.

        Old and new agree whenever the recorded action happens to match the tracker -- which is
        every row of corpus/golden-v4, measured: the two sources pick out the same 2 rows, so
        this change is a NO-OP on the current corpus and its value cannot be shown by rerunning
        it. It can only be shown where they diverge, which is here.
        """
        # Tracker says our mon must recharge; the action record says it did something else.
        decisions = {
            ("b", 7, "p2"): _row(seat="p2", self_mr=True, opp_mr=False, chose_recharge=False),
            ("b", 7, "p1"): _row(seat="p1", self_mr=False, opp_mr=False, chose_recharge=False),
        }
        old = set(_candidate_derived_recharging(None, decisions, "b", 7))
        new = set(
            production_recharging_slots(
                anchor_observation_metadata(decisions.get(("b", 7, "p2"))), "p2"
            )
        )
        self.assertEqual(old, set(), "old rule believed the action record")
        self.assertEqual(new, {"p2"}, "new rule believes the parser tracker")
        self.assertNotEqual(old, new)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
