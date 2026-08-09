"""The Encore lock's third id source: the payload's public ``lastUsedMove``.

Q2 / `encore_move_unknown` on the OPPONENT seat. Captured on
`fb3h0-955014` round 155 seat p1 (32 worlds attempted, 0 constructed) and
independently on `fb3h0-955041` round 54: the opposing active is asleep AND
Encored, so every recent public event is a `|cant|...|slp` line and the search
lane's 24-line scan of ``recent_public_events`` finds no `|move|` to name the
lock. The identity was never actually missing -- the world constructor was
already holding it in ``sides[<slot>]["lastUsedMove"]`` (probed at the live
refusal: `"rest"` on 955014, `"swordsdance"` on 955041) and simply never read
it on the non-transformed Encore branch.

Two coverage obligations, because the failure direction FLIPS with this change:

* Before: the risk is refusing too much, and it is LOUD (a counted
  `encore_move_unknown` world failure and a `no_worlds_constructed` fallback).
* After: the risk is answering WRONGLY -- locking a searched world onto a move
  nobody encored -- and that is SILENT.

So the classes below split into "the new source is read" (the refusal is gone)
and "the new source cannot lie" (precedence, the ``switch`` sentinel, and
fail-closed on an id the sampled world does not contain).
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    _resolve_encored_move_index,
    battle_spec_from_payload,
)
from pokezero.poke_engine_adapter import MoveSpec  # noqa: E402

from test_engine_world import _dex, _override, _payload  # noqa: E402


class OpponentEncoreFromPublicLastUsedMoveTests(unittest.TestCase):
    """The captured class: opponent Encored, event window carries no `|move|`.

    ``encored_moves`` is left EMPTY in every test here -- that is exactly what
    ``_public_effect_signals`` hands the constructor once the encored move's
    `|move|` line has scrolled out of the 24-line window.
    """

    def setUp(self) -> None:
        self.dex = _dex()

    def _payload_with(self, last_used: object) -> dict:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Encore"]
        payload["sides"]["p2"]["lastUsedMove"] = last_used
        return payload

    def test_opponent_encore_resolves_from_the_payload_latch(self) -> None:
        """The exemplar, reduced: no event-scan id, but the latch has one.

        Sampled Snorlax is ``bodyslam`` slot 0 / ``shadowball`` slot 1, so a
        latch of ``shadowball`` must produce slot 1 and not slot 0 -- a
        first-slot default would pass a weaker assertion.
        """

        world = battle_spec_from_payload(
            self._payload_with("shadowball"), _override(), dex=self.dex
        )
        side = world.spec.side_two
        active = side.pokemon[side.active_index]
        self.assertIn("encore", side.volatile_statuses)
        self.assertEqual(side.last_used_move, "move:1")
        self.assertEqual(active.moves[1].id, "shadowball")

    def test_the_lock_still_reaches_the_engine_for_the_first_slot_too(self) -> None:
        world = battle_spec_from_payload(
            self._payload_with("bodyslam"), _override(), dex=self.dex
        )
        self.assertEqual(world.spec.side_two.last_used_move, "move:0")

    def test_self_seat_falls_back_to_the_latch_when_rows_are_ambiguous(self) -> None:
        """Symmetric: the latch is public for BOTH seats.

        The self request's Encore signature is "exactly one enabled move". When
        the rows do not show that pattern (here: both enabled) the self seat had
        no third source either, and refused for the same reason.
        """

        payload = _payload(self.dex)
        payload["sides"]["p1"]["volatiles"] = ["Encore"]
        payload["sides"]["p1"]["pokemon"][0]["moves"] = [
            {"id": "earthquake", "pp": 12, "maxpp": 16, "disabled": False},
            {"id": "icebeam", "pp": 16, "maxpp": 16, "disabled": False},
        ]
        payload["sides"]["p1"]["lastUsedMove"] = "icebeam"
        side = battle_spec_from_payload(payload, _override(), dex=self.dex).spec.side_one
        self.assertEqual(side.last_used_move, "move:1")
        self.assertEqual(side.pokemon[side.active_index].moves[1].id, "icebeam")


class TheLatchCannotLieTests(unittest.TestCase):
    """The post-change failure direction: a wrong lock is silent, so bound it."""

    def setUp(self) -> None:
        self.dex = _dex()

    def _assert_refuses(self, payload: dict, **kwargs) -> None:
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(payload, _override(), dex=self.dex, **kwargs)
        self.assertEqual(caught.exception.reason, "encore_move_unknown")

    def test_the_event_scan_still_wins_when_both_sources_speak(self) -> None:
        """Precedence, not replacement.

        ``encored_moves`` is source 1 and the latch is source 3. Feeding them
        DIFFERENT ids and asserting source 1's answer is the only way to tell a
        strictly-additive third source from a silent re-ranking of the first.
        No world that constructs today may change.
        """

        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Encore"]
        payload["sides"]["p2"]["lastUsedMove"] = "shadowball"
        world = battle_spec_from_payload(
            payload, _override(), dex=self.dex, encored_moves={"p2": "bodyslam"}
        )
        self.assertEqual(world.spec.side_two.last_used_move, "move:0")

    def test_the_self_request_rows_still_win_over_the_latch(self) -> None:
        """Source 2 outranks source 3 on the self seat, for the same reason."""

        payload = _payload(self.dex)
        payload["sides"]["p1"]["volatiles"] = ["Encore"]
        payload["sides"]["p1"]["pokemon"][0]["moves"] = [
            {"id": "earthquake", "pp": 12, "maxpp": 16, "disabled": False},
            {"id": "icebeam", "pp": 16, "maxpp": 16, "disabled": True},
        ]
        payload["sides"]["p1"]["lastUsedMove"] = "icebeam"
        side = battle_spec_from_payload(payload, _override(), dex=self.dex).spec.side_one
        self.assertEqual(side.last_used_move, "move:0")

    def test_the_switch_sentinel_is_not_a_lock(self) -> None:
        """``"switch"`` is the positive fact "no last move", not a move id.

        Encore cannot start against it (Showdown: ``if (!move) return false``)
        and cannot survive the switch that writes it, so a payload carrying both
        is incoherent public state and must refuse rather than construct.
        """

        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Encore"]
        payload["sides"]["p2"]["lastUsedMove"] = "switch"
        self._assert_refuses(payload)

    def test_an_empty_or_absent_latch_still_refuses(self) -> None:
        for latch in ("", None):
            with self.subTest(latch=latch):
                payload = _payload(self.dex)
                payload["sides"]["p2"]["volatiles"] = ["Encore"]
                payload["sides"]["p2"]["lastUsedMove"] = latch
                self._assert_refuses(payload)

        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Encore"]
        payload["sides"]["p2"].pop("lastUsedMove", None)
        self._assert_refuses(payload)

    def test_a_latch_outside_the_sampled_moveset_refuses(self) -> None:
        """Fail-closed is preserved end to end.

        ``surf`` is a real move in this dex and a real move on the SAMPLED p1
        team, but sampled Snorlax does not have it. Locking slot 0 instead would
        be the silently-wrong world this class exists to prevent.
        """

        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Encore"]
        payload["sides"]["p2"]["lastUsedMove"] = "surf"
        self._assert_refuses(payload)

    def test_the_latch_is_read_from_the_encored_side_only(self) -> None:
        """p1's latch must not resolve p2's lock."""

        payload = _payload(self.dex)
        payload["sides"]["p2"]["volatiles"] = ["Encore"]
        payload["sides"]["p1"]["lastUsedMove"] = "bodyslam"
        self._assert_refuses(payload)

    def test_the_latch_does_not_create_an_encore_that_is_not_there(self) -> None:
        """No Encore volatile -> the latch keeps its pre-existing meaning.

        The un-encored branch already seeds ``last_used_move`` from the same
        latch; this pins that the new source did not change which BRANCH runs,
        only how the encore branch resolves an id. ``encore`` must be absent
        from the durations map.
        """

        payload = _payload(self.dex)
        payload["sides"]["p2"]["lastUsedMove"] = "shadowball"
        side = battle_spec_from_payload(payload, _override(), dex=self.dex).spec.side_two
        self.assertEqual(side.last_used_move, "move:1")
        self.assertNotIn("encore", side.volatile_statuses)
        self.assertNotIn("encore", dict(side.volatile_status_durations))


class ResolverPrecedenceUnitTests(unittest.TestCase):
    """``_resolve_encored_move_index`` in isolation: the three-source ladder."""

    SPECS = (
        MoveSpec(id="bodyslam", pp=24),
        MoveSpec(id="healbell", pp=5),
        MoveSpec(id="protect", pp=10),
    )

    def test_source_three_is_last(self) -> None:
        ambiguous = [
            {"id": "bodyslam", "disabled": False},
            {"id": "protect", "disabled": False},
        ]
        unambiguous = [
            {"id": "bodyslam", "disabled": True},
            {"id": "protect", "disabled": False},
        ]
        # 1 beats 2 and 3.
        self.assertEqual(
            _resolve_encored_move_index(
                self.SPECS,
                rows_for_active=unambiguous,
                encored_move="healbell",
                public_last_used_move="bodyslam",
            ),
            1,
        )
        # 2 beats 3.
        self.assertEqual(
            _resolve_encored_move_index(
                self.SPECS,
                rows_for_active=unambiguous,
                encored_move=None,
                public_last_used_move="bodyslam",
            ),
            2,
        )
        # 3 is consulted only when 1 and 2 are both silent.
        self.assertEqual(
            _resolve_encored_move_index(
                self.SPECS,
                rows_for_active=ambiguous,
                encored_move=None,
                public_last_used_move="bodyslam",
            ),
            0,
        )
        self.assertEqual(
            _resolve_encored_move_index(
                self.SPECS,
                rows_for_active=None,
                encored_move=None,
                public_last_used_move="protect",
            ),
            2,
        )

    def test_the_default_keeps_the_old_two_source_behaviour(self) -> None:
        """Callers that do not pass source 3 see exactly the old resolver."""

        self.assertIsNone(
            _resolve_encored_move_index(
                self.SPECS, rows_for_active=None, encored_move=None
            )
        )

    def test_source_three_fails_closed_on_an_unknown_id(self) -> None:
        self.assertIsNone(
            _resolve_encored_move_index(
                self.SPECS,
                rows_for_active=None,
                encored_move=None,
                public_last_used_move="surf",
            )
        )

    def test_the_switch_sentinel_is_rejected_before_the_moveset_lookup(self) -> None:
        """The sentinel filter must be a RULE, not an accident of the move list.

        Relying on "no gen3 move is called `switch`, so the lookup misses
        anyway" makes the guard behaviour-inert: a mutant that deletes it
        survives, which means nothing pins the boundary. So assert against a
        moveset that DOES contain a literal ``switch`` slot -- the lookup would
        happily return 0, and only the sentinel rule stops it.
        """

        specs = (MoveSpec(id="switch", pp=1), MoveSpec(id="bodyslam", pp=24))
        self.assertIsNone(
            _resolve_encored_move_index(
                specs,
                rows_for_active=None,
                encored_move=None,
                public_last_used_move="switch",
            )
        )
        # Control: the same call with a real id resolves, so the None above is
        # the sentinel rule and not a broken fixture.
        self.assertEqual(
            _resolve_encored_move_index(
                specs,
                rows_for_active=None,
                encored_move=None,
                public_last_used_move="bodyslam",
            ),
            1,
        )

    def test_source_three_keeps_the_hiddenpower_prefix_tolerance(self) -> None:
        """The latch names the typed variant; the sampled spec carries its own.

        gen3 has at most one Hidden Power slot, so the prefix is unambiguous --
        the same tolerance sources 1 and 2 already rely on.
        """

        specs = (MoveSpec(id="bodyslam", pp=24), MoveSpec(id="hiddenpowerice70", pp=24))
        self.assertEqual(
            _resolve_encored_move_index(
                specs,
                rows_for_active=None,
                encored_move=None,
                public_last_used_move="hiddenpowerground70",
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
