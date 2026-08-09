"""Our own forced-recharge turn must be searchable on every schema, not just v4.

`_recharging_slots` learned about the self side from `metadata["self_must_recharge"]`, published
ONLY by `_feature_pack_metadata` -- and that block is schema-gated to
FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS on purpose: an always-present key silently changed
world seeding for the v2.2/v3 arms in flight. The consequence was that under every earlier
schema the key was absent, `recharging_slots` came back empty, `engine_world` seeded no
`mustrecharge` volatile, and `_require_world_reproduces_trap` had nothing to discharge the
`trapped: true` Showdown sets on a recharge request. Those decisions were refused with

    self_request_state_unsupported: self active request flags ['trapped'] constrain legality
    beyond this construction (sampled world does not trap: foe ability 'X')

with `request_legal_choices == ['recharge']` and `recharging_slots=[]` in the record -- the same
key the Mean Look class lands on, and the same shape: the exemption exists and the signal never
arrives.

WHY THE REQUEST AND NOT A REPUBLISHED PACK. Republishing `self_must_recharge` on every schema is
exactly the mid-run behaviour change the gate was added to prevent. And the self side does not
need a reconstruction the way the opponent side does: Showdown's `Pokemon.getMoves(lockedMove)`
returns the single synthetic entry `{move: 'Recharge', id: 'recharge'}` if and only if
`lockedMove` is the `mustrecharge` volatile, and sets `this.trapped = true` in the same call. A
recharge-only request is not evidence about the lock, it IS the lock, disclosed to the seat that
has to act on it -- and it is schema-independent.

SCOPE, HONESTLY. This closes the recharge half on v2.2/v3. Under v4 the tracker key is already
present and True, so the union changes nothing there; if a v4 run is still refusing recharge
turns, the cause is elsewhere.
"""

from __future__ import annotations

import unittest
from collections import Counter
from types import SimpleNamespace

from pokezero.engine_search import EngineMctsPolicy, _self_request_forces_recharge


def _request(*move_ids: str) -> dict:
    """A Showdown move request whose active row offers exactly these moves."""

    return {
        "active": [
            {
                "moves": [{"move": move_id.title(), "id": move_id} for move_id in move_ids],
                "trapped": True,
            }
        ],
        "side": {"pokemon": []},
    }


#: Showdown's own spelling for the recharge turn: one synthetic move, no pp, `trapped: true`.
_RECHARGE_REQUEST = _request("recharge")


def _context(*, seat: str = "p1", metadata: dict | None = None, request=_RECHARGE_REQUEST):
    return SimpleNamespace(
        player_id=seat,
        observation=SimpleNamespace(metadata=dict(metadata or {})),
        public_materialization_state=(
            None if request is None else SimpleNamespace(self_request=request)
        ),
        trajectory=None,
        decision_round_index=None,
    )


def _policy() -> EngineMctsPolicy:
    policy = EngineMctsPolicy.__new__(EngineMctsPolicy)
    policy.stats = SimpleNamespace(choices_unmapped_causes=Counter(), unmapped_choices=Counter())
    return policy


class SelfRechargeIsDerivedFromOurOwnRequestTest(unittest.TestCase):
    """The production edit: `_recharging_slots`, not a harness twin."""

    def test_a_recharge_only_request_locks_our_slot_with_no_metadata_at_all(self) -> None:
        """The v2.2/v3 case. Pre-fix this returned `()` and the decision was refused."""

        got = EngineMctsPolicy._recharging_slots(_policy(), _context(metadata={}))
        self.assertIn("p1", got, "a recharge-only request did not lock our own slot")

    def test_it_follows_the_seat_rather_than_being_hardcoded(self) -> None:
        """Non-vacuity: `return ("p1",)` would satisfy the assertion above."""

        got = EngineMctsPolicy._recharging_slots(_policy(), _context(seat="p2", metadata={}))
        self.assertEqual(got, ("p2",))

    def test_an_ordinary_request_does_not_lock_us(self) -> None:
        """The other half. Without it, `always lock our slot` passes everything above."""

        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(metadata={}, request=_request("hyperbeam", "bodyslam"))
        )
        self.assertNotIn("p1", got)

    def test_the_tracker_still_locks_us_when_the_request_is_unavailable(self) -> None:
        """The v4 path is unchanged: no request on the context, key present and True."""

        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(metadata={"self_must_recharge": True}, request=None)
        )
        self.assertIn("p1", got)

    def test_the_two_proofs_do_not_double_count_or_cancel(self) -> None:
        """Under v4 both fire. The union must still name our slot exactly once."""

        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(metadata={"self_must_recharge": True})
        )
        self.assertEqual(got, ("p1",))

    def test_a_context_with_no_public_state_is_not_a_crash(self) -> None:
        """`_recharging_slots` is also called from hand-built and cached contexts."""

        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(metadata={"self_must_recharge": False}, request=None)
        )
        self.assertEqual(got, ())


class TheRequestProbeIsNarrowTest(unittest.TestCase):
    """`_self_request_forces_recharge` must not mistake other one-move locks for a recharge.

    An Encore lock, a Choice Band lock and a mid-charge Solar Beam all present a one-entry
    active moveset with `trapped: true`. None of them is a recharge, and seeding MUSTRECHARGE
    for them would model a mon that cannot act when it can -- strictly worse than the refusal
    this change removes, because the world would be silently wrong instead of declined.
    """

    def test_a_recharge_request_reads_true(self) -> None:
        self.assertTrue(_self_request_forces_recharge(_context()))

    def test_another_single_move_lock_reads_false(self) -> None:
        self.assertFalse(
            _self_request_forces_recharge(_context(request=_request("solarbeam"))),
            "a mid-charge Solar Beam is not a recharge turn",
        )

    def test_a_multi_move_request_that_includes_recharge_reads_false(self) -> None:
        self.assertFalse(
            _self_request_forces_recharge(_context(request=_request("recharge", "bodyslam")))
        )

    def test_a_force_switch_request_reads_false(self) -> None:
        self.assertFalse(
            _self_request_forces_recharge(
                _context(request={"forceSwitch": [True], "side": {"pokemon": []}})
            )
        )

    def test_malformed_and_missing_shapes_read_false_rather_than_raising(self) -> None:
        for label, request in {
            "no active key": {"side": {"pokemon": []}},
            "empty active list": {"active": []},
            "active is not a list": {"active": {"moves": [{"id": "recharge"}]}},
            "moves is not a list": {"active": [{"moves": "recharge"}]},
            "move is not a mapping": {"active": [{"moves": ["recharge"]}]},
            "no id or move name": {"active": [{"moves": [{"pp": 1}]}]},
        }.items():
            with self.subTest(label):
                self.assertFalse(_self_request_forces_recharge(_context(request=request)))
        self.assertFalse(_self_request_forces_recharge(_context(request=None)))
        self.assertFalse(_self_request_forces_recharge(SimpleNamespace()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
