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

import json
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    _resolve_encored_move_index,
    battle_spec_from_payload,
)
from pokezero.poke_engine_adapter import MoveSpec  # noqa: E402
from pokezero.dex import normalize_id  # noqa: E402

from _showdown_root import has_showdown, requires_showdown, showdown_root  # noqa: E402
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


class TheSwitchSentinelSeedsTheEnginesOwnVariantTests(unittest.TestCase):
    """The UN-encored branch's ``switch:0`` seeding, which nothing pinned.

    ``_build_side_spec`` used to carry, immediately after reading the latch, the
    statement ``if last_used_move_id == "switch": last_used_move_id = "switch"``.
    That is a no-op, and it read like an intentional guard -- so a maintainer
    editing around it would reasonably assume it was load-bearing and tested. It
    was neither, and the seeding it appeared to protect was not tested either:
    renaming the ``"switch"`` literal on the un-encored branch sent the sentinel
    into ``_resolve_encored_move_index``, which cannot match it against any sampled
    moveset, so ``last_used_move`` fell back to ``""`` and the whole ``switch:0``
    seeding disappeared **with nothing red**.

    ``switch:0`` is a POSITIVE fact rather than ignorance: a fresh switch-in
    genuinely has no last move (``Pokemon.clearVolatile()``), Encore correctly
    fails against it, and the engine has a distinct ``LastUsedMove`` variant for
    exactly that. ``""`` is the engine's *unknown*, which is a different claim and
    the one Encore's ``onStart`` reads as "no move" only by accident.

    Distinct from ``ResolverPrecedenceUnitTests
    ::test_the_switch_sentinel_is_rejected_before_the_moveset_lookup``, which pins
    the sentinel's REJECTION inside the resolver against a synthetic moveset
    holding a move literally named ``switch``. That one is pinned-by-contract: no
    gen3 move normalises to ``switch``, so the resolver arm is not reachable from
    the pool. This class pins the reachable half -- what the un-encored branch
    WRITES when the latch is the sentinel.
    """

    def setUp(self) -> None:
        self.dex = _dex()

    def test_a_fresh_switch_in_seeds_the_engines_switch_variant(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["lastUsedMove"] = "switch"
        side = battle_spec_from_payload(payload, _override(), dex=self.dex).spec.side_two
        self.assertEqual(side.last_used_move, "switch:0")
        # Not the encore branch: the sentinel must not create a lock either.
        self.assertNotIn("encore", side.volatile_statuses)
        self.assertNotIn("encore", dict(side.volatile_status_durations))
        # Control, so the assertion above cannot pass on a fixture that seeds
        # nothing at all: the same payload with a real id still resolves a slot.
        payload["sides"]["p2"]["lastUsedMove"] = "shadowball"
        control = battle_spec_from_payload(payload, _override(), dex=self.dex).spec.side_two
        self.assertEqual(control.last_used_move, "move:1")

    def test_the_sentinel_is_seeded_on_the_SELF_seat_too(self) -> None:
        """Both seats, because the seeding block is not seat-scoped and must not become so.

        The un-encored seeding was added for EVERY side (an opponent that had
        visibly just moved was reaching the engine as ``LastUsedMove::None``), so a
        mutant narrowing this arm to ``is_self`` -- or to the opponent -- is a
        silent half-fix. One seat asserted alone cannot see it.
        """

        payload = _payload(self.dex)
        payload["sides"]["p1"]["lastUsedMove"] = "switch"
        side = battle_spec_from_payload(payload, _override(), dex=self.dex).spec.side_one
        self.assertEqual(side.last_used_move, "switch:0")


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


class PoolCannotReachTheLockedMoveDivergenceTests(unittest.TestCase):
    """The one shape where the latch and the event scan disagree, bounded to zero.

    The resolver's docstring argues source 3 is admissible partly because a CALLED
    move cannot move the latch. That is true for an ordinary caller, and NOT true in
    general: ``runMove`` takes the ``getLockedMove()`` branch, sets
    ``sourceEffect = lockedmove``, and STILL calls ``pokemon.moveUsed(...)``, so a
    locked continuation advances Showdown's ``lastMove`` while emitting
    ``|[from] lockedmove``. ``showdown._ReplayParser`` rejects every ``[from]``;
    ``determinization._called_move_line`` deliberately tolerates ``lockedmove``. The
    two sources use DIFFERENT rules, and on that one line the latch lags.

    Three reasons that is not a defect here, in increasing order of durability:

    1. Precedence. Source 1 (the faithful one) outranks source 3 (the lagging one),
       so wherever the event scan can see the line, its answer is the one used.
    2. Direction. Where source 1 is silent, source 3 lags rather than invents -- and
       for an Encore specifically the lock cannot BE the locked-move continuation,
       because ``encore.onStart`` requires a move slot the target owns with pp > 0.
    3. Reachability, pinned below at ZERO for this format.

    ⚠ WHAT WOULD BREAK IT. If the pool ever gained a locking move ALONGSIDE a caller,
    both the caller and the callee would be in the sampled moveset, so the stale latch
    would resolve to a real slot and build a SILENTLY WRONG world instead of refusing.
    That is why this is a pinned zero and not a comment.
    """

    #: Detectors, kept alongside the assertions so a reviewer can see they are precise
    #: rather than substring-matching a whole move block. `getLockedMove()` fires the
    #: LockMove event, so "can produce a `[from] lockedmove` line" is exactly "installs
    #: the lockedmove volatile, or supplies onLockMove".
    LOCKING = re.compile(r"volatileStatus: 'lockedmove'|\bonLockMove\b")
    CALLER = "useMove("

    #: Per-METHOD rather than per-class on purpose. A class-level skip collapses to a
    #: single skipped entry and drops unittest's reported test COUNT (21 -> 16), and the
    #: count is exactly what the CI step guards -- so the suite could shrink without the
    #: guard moving. Same shape as ``tests/test_spread_gate_provenance.py``, whose step
    #: pins ``Ran 6`` and ``OK (skipped=5)`` for this identical reason.
    SKIP = requires_showdown("the locking-move scan reads the real gen3 set pool")

    @classmethod
    def setUpClass(cls) -> None:
        cls.pool = set()
        cls.blocks = {}
        if not has_showdown():
            return  # the five pins below skip; nothing here may raise instead.
        root = showdown_root()
        sets_path = root / "data/random-battles/gen3/sets.json"
        moves_path = root / "data/moves.ts"
        for entry in json.loads(sets_path.read_text(encoding="utf-8")).values():
            rows = entry.get("sets", entry if isinstance(entry, list) else [])
            for row in rows:
                for move in row.get("movepool", row.get("moves", [])):
                    cls.pool.add(normalize_id(str(move)))
        # Brace-matched, NOT regex-delimited: a `\n\t},\n` terminator parses 477 of the
        # 953 blocks and its "no locking move found" would have been vacuously true.
        source = moves_path.read_text(encoding="utf-8")
        for match in re.finditer(r"^\t(\w+): \{$", source, re.M):
            index, depth = match.end(), 1
            while depth:
                depth += (source[index] == "{") - (source[index] == "}")
                index += 1
            cls.blocks[match.group(1)] = source[match.end() : index]

    @SKIP
    def test_the_detectors_are_not_vacuous(self) -> None:
        """Every zero below is a set intersection; an empty detector makes them all pass."""

        self.assertGreater(len(self.blocks), 900)
        self.assertGreater(len(self.pool), 100)
        locking = {m for m, b in self.blocks.items() if self.LOCKING.search(b)}
        callers = {m for m, b in self.blocks.items() if self.CALLER in b}
        self.assertLessEqual(
            {"thrash", "outrage", "petaldance", "uproar", "rollout", "iceball", "bide"},
            locking,
        )
        self.assertLessEqual({"sleeptalk", "metronome", "mirrormove"}, callers)
        # And the pool itself is populated with things this campaign knows are there.
        self.assertLessEqual(
            {"rest", "toxic", "protect", "sleeptalk", "solarbeam", "encore"}, self.pool
        )

    @SKIP
    def test_no_gen3_randbats_set_carries_a_locking_move(self) -> None:
        locking = {m for m, b in self.blocks.items() if self.LOCKING.search(b)}
        self.assertEqual(sorted(locking & self.pool), [])

    @SKIP
    def test_the_only_caller_in_the_pool_is_sleep_talk(self) -> None:
        """Bounds the other half of the pair: a `[from] lockedmove` line needs a caller
        only when the LOCKING move is called, but a lone caller with nothing lockable to
        call cannot produce one either."""

        callers = {m for m, b in self.blocks.items() if self.CALLER in b}
        self.assertEqual(sorted(callers & self.pool), ["sleeptalk"])

    @SKIP
    def test_the_only_charge_move_in_the_pool_sets_the_latch_to_itself(self) -> None:
        """Solar Beam is the pool's only two-turn move, and it is not a divergence.

        Its turn-1 line is an ordinary non-`[from]` `|move|`, so the latch takes
        `solarbeam`; the release turn names the same id. There is no caller/callee pair
        for the two rules to disagree about.
        """

        charge = {m for m, b in self.blocks.items() if re.search(r"(?<!re)charge: 1", b)}
        self.assertLessEqual({"solarbeam", "razorwind", "skullbash", "fly", "dig"}, charge)
        self.assertEqual(sorted(charge & self.pool), ["solarbeam"])

    @SKIP
    def test_the_other_latch_perturbers_are_absent_too(self) -> None:
        """Disable, Mimic and Sketch are the moves that could put a Struggle or a
        substituted id under a live Encore. All absent; see the resolver docstring's
        argument (4), which is scoped to PP exhaustion and not to these."""

        for move in ("disable", "mimic", "sketch", "struggle"):
            with self.subTest(move=move):
                self.assertNotIn(move, self.pool)


class TheReachabilityClaimIsTiedToAMeasuredPoolTests(unittest.TestCase):
    """Runs EVERYWHERE, including CI, which the scan above deliberately does not.

    The five pins above need the real Showdown checkout, and CI builds none -- the same
    honest gap `tests/test_spread_gate_provenance.py` documents. That would leave the
    resolver docstring's "no locking move in this pool" resting on a scan nothing in CI
    can execute, and a zero nobody re-measures is a zero that goes stale silently.

    So this ties the claim to the committed pool census instead. It does NOT re-derive
    reachability -- it asserts the POOL the reachability was measured against has not
    moved. If the set data is regenerated and the pool changes shape, this reddens in CI
    and a human re-runs the checkout-dependent scan. That is a staleness alarm, which is
    the honest thing to promise here, and not a proof.
    """

    def test_the_committed_census_still_describes_the_pool_that_was_scanned(self) -> None:
        census = json.loads(
            (Path(ROOT) / "tests/data/c152_pool_reachability_census.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            (census["species"], census["sets"], census["distinct_moves"]),
            (220, 393, 125),
            "the gen3 randbats pool has changed shape since the locking-move "
            "reachability scan in PoolCannotReachTheLockedMoveDivergenceTests was "
            "measured. Re-run that class against a real checkout "
            "(POKEZERO_SHOWDOWN_ROOT=...) and re-derive the zero before editing "
            "these numbers.",
        )


if __name__ == "__main__":
    unittest.main()
