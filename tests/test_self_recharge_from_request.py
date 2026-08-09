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

WHY A SECOND PROOF AND NOT A REPUBLISHED PACK. Republishing `self_must_recharge` on every schema
is exactly the mid-run behaviour change the gate was added to prevent. And the self side does
not need a reconstruction the way the opponent side does: `Pokemon.getMoveRequestData` sets
`this.trapped = true` whenever `getLockedMove()` returns anything, and `getMoves(lockedMove)`
short-circuits to `[{move: 'Recharge', id: 'recharge'}]` when that locked move is `recharge`
(`sim/pokemon.ts:968`, `:1084-1088`). `mustrecharge` is gen3's only `onLockMove: 'recharge'`, so
a request offering nothing but `recharge` is not evidence about the lock, it IS the lock.

WHY `action_candidates` AND NOT THE RAW REQUEST. Both work in production; only one is
MIRRORABLE. `scripts/fidelity_gate_events.py::production_recharging_slots` exists to be
"`recharging_slots` as production builds it", and it is handed observation metadata from a
recorded corpus row -- no `PolicyContext`, no raw request. The payload's `selfActiveMoves` drops
the synthetic recharge entry (it carries no `pp`/`maxpp`), so a corpus row has no request-side
trace of the lock at all; a request-based rule would leave the four fidelity gates seeding
v2.2/v3 self-recharge worlds WITHOUT MUSTRECHARGE while production seeds them with it. The
candidate fold is ungated metadata a row already carries, so the gate calls the very same
function. It also covers contexts holding metadata but no materialization state.

SCOPE, HONESTLY. This closes the recharge half on v2.2/v3. Under v4 the tracker key is already
present and True, so the union changes nothing there; if a v4 run is still refusing recharge
turns, the cause is elsewhere.
"""

from __future__ import annotations

import unittest
from collections import Counter
from types import SimpleNamespace

from pokezero.engine_search import EngineMctsPolicy, self_recharge_from_action_candidates


def _candidates(*legal_moves: str, legal_switches: tuple[str, ...] = ()) -> list[dict]:
    """`action_candidates` as `_action_candidate_metadata` builds it: all nine slots, always."""

    out: list[dict] = []
    for index in range(4):
        legal = index < len(legal_moves)
        out.append(
            {
                "action_index": index,
                "kind": "move",
                "legal": legal,
                "move_slot": index + 1,
                "move_id": legal_moves[index] if legal else f"slot:{index + 1}",
            }
        )
    for offset in range(5):
        legal = offset < len(legal_switches)
        out.append(
            {
                "action_index": 4 + offset,
                "kind": "switch",
                "legal": legal,
                "pokemon": {"species": legal_switches[offset]} if legal else None,
            }
        )
    return out


#: Showdown's own spelling for the recharge turn: one legal choice, the synthetic pseudo-move,
#: and no switch (the request carries `trapped: true`).
_RECHARGE_CANDIDATES = _candidates("recharge")


def _context(*, seat: str = "p1", metadata: dict | None = None, candidates=_RECHARGE_CANDIDATES):
    md: dict = dict(metadata or {})
    if candidates is not None:
        md["action_candidates"] = candidates
    return SimpleNamespace(
        player_id=seat,
        observation=SimpleNamespace(metadata=md),
        trajectory=None,
        decision_round_index=None,
    )


def _policy() -> EngineMctsPolicy:
    policy = EngineMctsPolicy.__new__(EngineMctsPolicy)
    policy.stats = SimpleNamespace(choices_unmapped_causes=Counter(), unmapped_choices=Counter())
    return policy


class SelfRechargeIsDerivedFromTheRequestsLegalSetTest(unittest.TestCase):
    """The production edit: `_recharging_slots`, not a harness twin."""

    def test_a_recharge_only_request_locks_our_slot_with_no_pack_key(self) -> None:
        """The v2.2/v3 case. Pre-fix this returned `()` and the decision was refused."""

        got = EngineMctsPolicy._recharging_slots(_policy(), _context())
        self.assertIn("p1", got, "a recharge-only request did not lock our own slot")

    def test_it_follows_the_seat_rather_than_being_hardcoded(self) -> None:
        """Non-vacuity: `return ("p1",)` would satisfy the assertion above."""

        got = EngineMctsPolicy._recharging_slots(_policy(), _context(seat="p2"))
        self.assertEqual(got, ("p2",))

    def test_an_ordinary_request_does_not_lock_us(self) -> None:
        """The other half. Without it, `always lock our slot` passes everything above."""

        got = EngineMctsPolicy._recharging_slots(
            _policy(),
            _context(candidates=_candidates("hyperbeam", "bodyslam", legal_switches=("Starmie",))),
        )
        self.assertNotIn("p1", got)

    def test_the_tracker_still_locks_us_when_candidates_are_absent(self) -> None:
        """The v4 path is unchanged: no candidates on the context, key present and True."""

        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(metadata={"self_must_recharge": True}, candidates=None)
        )
        self.assertIn("p1", got)

    def test_the_two_proofs_do_not_double_count_or_cancel(self) -> None:
        """Under v4 both fire. The union must still name our slot exactly once."""

        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(metadata={"self_must_recharge": True})
        )
        self.assertEqual(got, ("p1",))

    def test_an_empty_metadata_context_is_not_a_crash(self) -> None:
        """`_recharging_slots` is also called from hand-built and cached contexts."""

        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(metadata={"self_must_recharge": False}, candidates=None)
        )
        self.assertEqual(got, ())


class TheCandidateFoldIsNarrowTest(unittest.TestCase):
    """It must not mistake other one-move locks for a recharge.

    An Encore lock, a Choice Band lock and a mid-charge Solar Beam all present a single legal
    move with `trapped: true`. None is a recharge, and seeding MUSTRECHARGE for them would model
    a mon that cannot act when it can -- strictly worse than the refusal this change removes,
    because the world would be silently wrong instead of declined.
    """

    def test_a_recharge_request_reads_true(self) -> None:
        self.assertTrue(
            self_recharge_from_action_candidates({"action_candidates": _RECHARGE_CANDIDATES})
        )

    def test_another_single_move_lock_reads_false(self) -> None:
        for move_id in ("solarbeam", "struggle", "outrage", "bodyslam"):
            with self.subTest(move_id):
                self.assertFalse(
                    self_recharge_from_action_candidates(
                        {"action_candidates": _candidates(move_id)}
                    ),
                    f"a single legal {move_id} is not a recharge turn",
                )

    def test_recharge_alongside_another_legal_choice_reads_false(self) -> None:
        self.assertFalse(
            self_recharge_from_action_candidates(
                {"action_candidates": _candidates("recharge", "bodyslam")}
            )
        )

    def test_a_legal_switch_beside_the_recharge_reads_false(self) -> None:
        """A recharge request is `trapped: true`; a switchable one is a different position."""

        self.assertFalse(
            self_recharge_from_action_candidates(
                {"action_candidates": _candidates("recharge", legal_switches=("Starmie",))}
            )
        )

    def test_a_lone_legal_switch_reads_false(self) -> None:
        """A force-switch boundary is not a lock."""

        self.assertFalse(
            self_recharge_from_action_candidates(
                {"action_candidates": _candidates(legal_switches=("Starmie",))}
            )
        )

    def test_a_switch_candidate_carrying_a_move_id_reads_false(self) -> None:
        """Non-vacuity for the `kind == "move"` check, which nothing else reaches.

        Mutating that condition away SURVIVED the first version of this class: a real switch
        candidate has no `move_id`, so the id comparison rejects it anyway and the `kind` test
        never decides. Rather than delete a guard or call the mutant equivalent, this pins what
        the guard is FOR -- a row where the two disagree. `_action_candidate_metadata` cannot
        emit this shape today; the check exists so a future producer, or a hand-built context,
        cannot turn a switch into a forced-recharge lock.
        """

        self.assertFalse(
            self_recharge_from_action_candidates(
                {
                    "action_candidates": [
                        {
                            "action_index": 4,
                            "kind": "switch",
                            "legal": True,
                            "move_id": "recharge",
                            "pokemon": {"species": "Starmie"},
                        }
                    ]
                }
            ),
            "a switch candidate must never be read as the recharge pseudo-move",
        )

    def test_malformed_and_missing_shapes_read_false_rather_than_raising(self) -> None:
        for label, metadata in {
            "no candidates key": {},
            "candidates is not a list": {"action_candidates": "recharge"},
            "candidate is not a mapping": {"action_candidates": ["recharge"]},
            "no legal candidate at all": {"action_candidates": _candidates()},
            "legal move with no id": {
                "action_candidates": [{"action_index": 0, "kind": "move", "legal": True}]
            },
        }.items():
            with self.subTest(label):
                self.assertFalse(self_recharge_from_action_candidates(metadata))
        self.assertFalse(self_recharge_from_action_candidates(None))


class TheGateHarnessMirrorsProductionTest(unittest.TestCase):
    """`production_recharging_slots` must not drift from `_recharging_slots`.

    The four fidelity gates seed their worlds through that helper. B adds a second production
    input for the self side; left unmirrored the gates would seed v2.2/v3 self-recharge worlds
    without MUSTRECHARGE while production seeds them with it -- so the gates would stop measuring
    exactly the boundaries this change alters. Latent on today's corpus (both tracker keys are
    present on all 1295 golden-v4 rows), which is why it needs a test rather than a corpus run.
    """

    @staticmethod
    def _mirror():
        """Load the gate script for real.

        `pokezero_search` (the native crate) is stubbed rather than required: the harness imports
        it at module scope but `production_recharging_slots` never touches it, and a test that
        SKIPS whenever the crate is absent is exactly how a mirror is allowed to drift -- the
        divergence this class exists to catch would go unmeasured on every machine without a
        built crate, which is most of them.
        """

        import importlib.util
        import sys
        import types
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        sys.modules.setdefault("pokezero_search", types.ModuleType("pokezero_search"))
        spec = importlib.util.spec_from_file_location(
            "_fidelity_gate_events_under_test", root / "scripts" / "fidelity_gate_events.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.production_recharging_slots

    def test_the_mirror_agrees_with_production_on_every_shape(self) -> None:
        mirror = self._mirror()

        # The table must include the shapes where the fold's NARROWNESS is what decides, not
        # only the headline lock/no-lock pair. A mutation that restated the rule in this file
        # while dropping the `len(legal) == 1` and `kind == "move"` conditions SURVIVED the
        # first version of this test, which is exactly the drift it exists to catch: an
        # agreeing-on-the-easy-cases mirror is not a mirror.
        shapes = {
            "v2.2/v3 recharge (candidate fold only)": {
                "action_candidates": _RECHARGE_CANDIDATES
            },
            "v4 recharge (both inputs)": {
                "action_candidates": _RECHARGE_CANDIDATES,
                "self_must_recharge": True,
            },
            "v4 tracker only": {"self_must_recharge": True},
            "free mon": {"action_candidates": _candidates("bodyslam")},
            "explicit no-lock": {
                "action_candidates": _candidates("bodyslam"),
                "self_must_recharge": False,
            },
            "single non-recharge lock (Encore / Choice / mid-charge)": {
                "action_candidates": _candidates("solarbeam")
            },
            "recharge plus another legal move": {
                "action_candidates": _candidates("recharge", "bodyslam")
            },
            "recharge with a legal switch beside it": {
                "action_candidates": _candidates("recharge", legal_switches=("Starmie",))
            },
            "lone legal switch (force-switch boundary)": {
                "action_candidates": _candidates(legal_switches=("Starmie",))
            },
            "no legal choice at all": {"action_candidates": _candidates()},
            "no candidates key": {},
            # The two shapes where `kind == "move"` and `normalize_id` are what decide. Without
            # them a locally-restated mirror that dropped BOTH still agreed everywhere above.
            "switch candidate carrying a recharge move_id": {
                "action_candidates": [
                    {
                        "action_index": 4,
                        "kind": "switch",
                        "legal": True,
                        "move_id": "recharge",
                        "pokemon": {"species": "Starmie"},
                    }
                ]
            },
            "legal move with no id at all": {
                "action_candidates": [{"action_index": 0, "kind": "move", "legal": True}]
            },
        }
        for label, metadata in shapes.items():
            with self.subTest(label):
                production = EngineMctsPolicy._recharging_slots(
                    _policy(), _context(metadata=metadata, candidates=None)
                )
                self.assertEqual(
                    mirror(metadata, "p1"),
                    production,
                    "the gate harness no longer mirrors production on this shape",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
